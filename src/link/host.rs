//! The host: listens on the local network, pairs new devices with a code,
//! lets remembered ones in, and serves their calls.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::crypto::{self, Identity};
use super::peer::{self, BlobMeta, Handler, HostInfo, Incoming, Msg, Peer, Services, PROTO};
use super::trust::{self, Trust};
use super::wire;
use super::{addr, files, Backend};

/// How long a pairing code stays valid.
pub(crate) const CODE_LIFETIME: Duration = Duration::from_secs(10 * 60);
/// Wrong codes allowed before pairing closes (the owner has to open it again
/// for a new code): a stranger gets at most this many one-in-a-million
/// guesses per click on "Add a device".
pub(crate) const MAX_TRIES: u32 = 5;
/// Connections being set up at once (handshake + opening), so a flood can't
/// tie the host up — and from one address, so one PC can't take them all.
const MAX_OPENING: usize = 8;
const MAX_OPENING_PER_IP: usize = 2;
/// How long the opening (handshake, hello, pairing) may take in all.
pub(crate) const OPENING_LIMIT: Duration = Duration::from_secs(15);
/// Devices connected at once.
const MAX_SESSIONS: usize = 8;

/// What the host shares and how it presents itself.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct HostConfig {
    pub name: String,
    pub port: u16,
    pub services: Services,
    /// Save files from paired devices without asking.
    pub accept_files_unasked: bool,
    /// Running without a window (nobody to ask, no pairing).
    pub background: bool,
}

/// An open pairing window.
pub(crate) struct Pairing {
    pub code: String,
    pub until: Instant,
    pub tries: u32,
}

/// Something the Link tab should tell the owner.
pub(crate) enum HostEvent {
    Paired(String),
    WrongCode { name: String, closed: bool },
}

/// Listening state, for the Link tab.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Listening {
    Starting,
    /// Another copy of the app still holds the port (the background host
    /// finishing a job before it hands over).
    WaitingForPort,
    On,
    Failed(String),
}

pub(crate) struct HostShared {
    pub identity: Identity,
    pub dir: PathBuf,
    pub config: Mutex<HostConfig>,
    pub pairing: Mutex<Option<Pairing>>,
    pub sessions: Mutex<Vec<Arc<Peer>>>,
    pub events: Mutex<Vec<HostEvent>>,
    pub listening: Mutex<Listening>,
    pub stop: AtomicBool,
    /// Connections being opened, in all and per address.
    opening: Mutex<HashMap<IpAddr, usize>>,
    pub backend: Arc<dyn Backend>,
    pub files: Arc<files::FileHub>,
    /// Calls being served right now (all devices) — the background host
    /// waits for this to reach 0 before it exits.
    pub busy: AtomicUsize,
}

impl HostShared {
    pub fn info(&self) -> HostInfo {
        let c = self.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
        HostInfo {
            name: c.name,
            version: env!("CARGO_PKG_VERSION").to_string(),
            services: c.services,
            chat_model: if c.services.chat { self.backend.chat_model() } else { None },
            chat_thinks: c.services.chat && self.backend.chat_thinks(),
            background: c.background,
        }
    }

    fn event(&self, e: HostEvent) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(e);
        self.backend.wake();
    }

    /// Open pairing with a fresh code (closing any earlier one).
    pub fn open_pairing(&self) -> String {
        let code = crypto::new_code();
        *self.pairing.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Pairing { code: code.clone(), until: Instant::now() + CODE_LIFETIME, tries: 0 });
        code
    }

    pub fn close_pairing(&self) {
        *self.pairing.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Tell every connected device the offer changed (on a thread of its
    /// own: a device that stopped reading would hold up the caller, the UI).
    /// One announcement at a time, each with the offer as it is when it goes
    /// out, so the last one sent is always the latest.
    pub fn announce(self: &Arc<Self>) {
        static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());
        let shared = self.clone();
        let _ = std::thread::Builder::new().name("link-announce".into()).spawn(move || {
            let _turn = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let info = shared.info();
            let peers = shared.sessions.lock().unwrap_or_else(|e| e.into_inner()).clone();
            for p in peers {
                p.control(&Msg::Info { info: info.clone() });
            }
        });
    }

    /// Stop taking calls (and devices): `Host::stop`, or the background host
    /// handing over. Devices with nothing running here are let go now.
    pub fn wind_down(&self) {
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        self.stop.store(true, Ordering::SeqCst);
        for p in sessions.iter() {
            if !p.serving() {
                p.close();
            }
        }
    }

    /// Disconnect a device (it was forgotten).
    pub fn drop_device(&self, key: &[u8]) {
        for p in self.sessions.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            if p.key == key {
                p.close();
            }
        }
    }
}

/// A running host.
pub(crate) struct Host {
    pub shared: Arc<HostShared>,
}

impl Host {
    pub fn start(
        identity: Identity,
        dir: PathBuf,
        config: HostConfig,
        backend: Arc<dyn Backend>,
        files: Arc<files::FileHub>,
    ) -> Host {
        let shared = Arc::new(HostShared {
            identity,
            dir,
            config: Mutex::new(config),
            pairing: Mutex::new(None),
            sessions: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            listening: Mutex::new(Listening::Starting),
            stop: AtomicBool::new(false),
            opening: Mutex::new(HashMap::new()),
            backend,
            files,
            busy: AtomicUsize::new(0),
        });
        let s = shared.clone();
        let _ = std::thread::Builder::new().name("link-host".into()).spawn(move || listen(s));
        let s = shared.clone();
        let _ = std::thread::Builder::new().name("link-find".into()).spawn(move || super::discover::respond(s));
        Host { shared }
    }

    /// Stop listening and disconnect everyone. (Under the sessions lock, so a
    /// device just finishing its opening can't slip in after.)
    pub fn stop(&self) {
        let sessions = self.shared.sessions.lock().unwrap_or_else(|e| e.into_inner());
        self.shared.stop.store(true, Ordering::SeqCst);
        for p in sessions.iter() {
            p.close();
        }
    }
}

/// A connection being opened, counted until it's dropped.
struct OpeningSlot {
    shared: Arc<HostShared>,
    ip: IpAddr,
}

impl OpeningSlot {
    /// A slot for a connection from `ip`, if one's free.
    fn take(shared: &Arc<HostShared>, ip: IpAddr) -> Option<OpeningSlot> {
        let mut opening = shared.opening.lock().unwrap_or_else(|e| e.into_inner());
        let all: usize = opening.values().sum();
        let mine = opening.entry(ip).or_insert(0);
        if all >= MAX_OPENING || *mine >= MAX_OPENING_PER_IP {
            return None;
        }
        *mine += 1;
        Some(OpeningSlot { shared: shared.clone(), ip })
    }
}

impl Drop for OpeningSlot {
    fn drop(&mut self) {
        let mut opening = self.shared.opening.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = opening.get_mut(&self.ip) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                opening.remove(&self.ip);
            }
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.stop();
    }
}

fn set_listening(shared: &HostShared, l: Listening) {
    *shared.listening.lock().unwrap_or_else(|e| e.into_inner()) = l;
    shared.backend.wake();
}

fn listen(shared: Arc<HostShared>) {
    let (port, background) = {
        let c = shared.config.lock().unwrap_or_else(|e| e.into_inner());
        (c.port, c.background)
    };
    // The port may still be held for a while by the copy this one takes over
    // from: the app that just closed (for the background host), or the
    // background host stopping (for the app) — it frees the port at once
    // unless it's still answering a device.
    let started = Instant::now();
    let mut asked = Instant::now();
    let listener = loop {
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        match TcpListener::bind(("0.0.0.0", port)) {
            Ok(l) => break l,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::AddrInUse {
                    // The background host stays until its last job is done,
                    // however long (the app closing for good stays as long
                    // too); the app closing takes a moment otherwise.
                    let background_stopping = !background && super::background::is_running();
                    let wait = if background {
                        started.elapsed() < Duration::from_secs(30 * 60)
                    } else {
                        background_stopping || started.elapsed() < Duration::from_secs(5)
                    };
                    if wait {
                        if background_stopping {
                            set_listening(&shared, Listening::WaitingForPort);
                            // Asked again now and then: one that started
                            // just after the app asked didn't hear it.
                            if asked.elapsed() > Duration::from_secs(5) {
                                asked = Instant::now();
                                super::background::stop_running();
                            }
                        }
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                }
                let why = if e.kind() == std::io::ErrorKind::AddrInUse {
                    format!("Port {port} is in use by another program — pick another port")
                } else {
                    format!("Couldn't listen on port {port}: {e}")
                };
                set_listening(&shared, Listening::Failed(why));
                return;
            }
        }
    };
    if listener.set_nonblocking(true).is_err() {
        set_listening(&shared, Listening::Failed("Couldn't listen".into()));
        return;
    }
    set_listening(&shared, Listening::On);
    while !shared.stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, from)) => {
                // Local network only: an internet address is dropped before a
                // single byte is read.
                if !addr::is_local(from.ip()) {
                    continue;
                }
                // Windows and macOS hand out accepted sockets non-blocking,
                // like the listener: the session needs blocking reads.
                if stream.set_nonblocking(false).is_err() {
                    continue;
                }
                let live = shared.sessions.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|p| !p.is_closed()).count();
                if live >= MAX_SESSIONS {
                    continue;
                }
                let Some(slot) = OpeningSlot::take(&shared, from.ip()) else { continue };
                let s = shared.clone();
                let _ = std::thread::Builder::new().name("link-session".into()).spawn(move || {
                    session(&s, stream, from, slot);
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => std::thread::sleep(Duration::from_millis(250)),
        }
    }
}

/// Clean up a name another PC sent: printable, short.
pub(crate) fn clean_name(name: &str) -> String {
    let n: String = name.chars().filter(|c| !c.is_control()).take(48).collect();
    let n = n.trim().to_string();
    if n.is_empty() { "Unnamed PC".to_string() } else { n }
}

fn session(shared: &Arc<HostShared>, stream: TcpStream, from: SocketAddr, slot: OpeningSlot) {
    let deadline = wire::OpeningDeadline::start(&stream, OPENING_LIMIT);
    let opened = open_session(shared, stream, from);
    drop(deadline);
    drop(slot);
    let Some((peer, reader)) = opened else { return };
    {
        // Link turned off (or handing over) while this one was opening: it
        // doesn't get in.
        let mut sessions = shared.sessions.lock().unwrap_or_else(|e| e.into_inner());
        if shared.stop.load(Ordering::SeqCst) {
            peer.close();
            return;
        }
        sessions.push(peer.clone());
    }
    shared.backend.wake();
    let handler: Arc<dyn Handler> = Arc::new(HostHandler { shared: shared.clone() });
    let _why = peer::serve(peer.clone(), reader, handler);
    shared.sessions.lock().unwrap_or_else(|e| e.into_inner()).retain(|p| !Arc::ptr_eq(p, &peer));
    let key = crypto::to_hex(&peer.key);
    let _ = Trust::update(&shared.dir, |t| {
        if let Some(d) = t.devices.iter_mut().find(|d| d.key == key) {
            d.last_seen = trust::now();
        }
    });
    shared.backend.wake();
}

/// Handshake, then let the device in — by its remembered key, or by pairing.
fn open_session(shared: &Arc<HostShared>, stream: TcpStream, from: SocketAddr) -> Option<(Arc<Peer>, wire::Reader)> {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let mut s = wire::handshake_responder(stream, &shared.identity).ok()?;
    let refuse = |w: &wire::Writer, reason: String| {
        let _ = w.json(1, &Msg::Refused { reason });
    };
    let f = s.reader.next().ok()?;
    let Ok(Msg::Hello { name, version, proto, spake }) = serde_json::from_slice::<Msg>(&f.body) else {
        return None;
    };
    let host_name = shared.config.lock().unwrap_or_else(|e| e.into_inner()).name.clone();
    if proto != PROTO {
        refuse(
            &s.writer,
            format!(
                "{host_name} runs Clarity TagFlow {}, this PC {version} — update both to the same version",
                env!("CARGO_PKG_VERSION")
            ),
        );
        return None;
    }
    let name = clean_name(&name);
    let key_hex = crypto::to_hex(&s.remote_key);
    // A remembered device comes straight in — unless it asks to pair again
    // (it lost its own record of this host, so it can't trust it without
    // the code).
    if spake.is_none() && Trust::load(&shared.dir).device(&key_hex).is_some() {
        s.writer.json(1, &Msg::Welcome { info: shared.info() }).ok()?;
    } else {
        // Not remembered: it has to pair, with the code shown here.
        let Some(spake) = spake.and_then(|h| crypto::from_hex(&h)) else {
            s.writer.json(1, &Msg::PairRequired { name: host_name }).ok()?;
            return None;
        };
        let background = shared.config.lock().unwrap_or_else(|e| e.into_inner()).background;
        let code = {
            let mut p = shared.pairing.lock().unwrap_or_else(|e| e.into_inner());
            match p.as_mut() {
                Some(open) if open.until > Instant::now() && open.tries < MAX_TRIES => {
                    // Counted when it starts, not when it fails: tries made
                    // at the same time can't add up to more than allowed.
                    open.tries += 1;
                    open.code.clone()
                }
                _ => {
                    *p = None;
                    drop(p);
                    refuse(
                        &s.writer,
                        if background {
                            format!("{host_name} is running in the background — open Clarity TagFlow there, then Link → Add a device for a code")
                        } else {
                            format!("Pairing isn't open on {host_name} — click Add a device in its Link settings for a code")
                        },
                    );
                    return None;
                }
            }
        };
        let (key, reply) = match crypto::pair_host(&code, &spake) {
            Ok(k) => k,
            Err(_) => {
                refuse(&s.writer, "Pairing failed".into());
                return None;
            }
        };
        s.writer.json(1, &Msg::PairReply { spake: crypto::to_hex(&reply) }).ok()?;
        let f = s.reader.next().ok()?;
        let Ok(Msg::PairConfirm { proof }) = serde_json::from_slice::<Msg>(&f.body) else { return None };
        let ok = crypto::from_hex(&proof).is_some_and(|p| crypto::check_receiver_proof(&key, &s.hash, &p));
        if !ok {
            let closed = {
                let mut p = shared.pairing.lock().unwrap_or_else(|e| e.into_inner());
                let used_up = p.as_ref().is_some_and(|open| open.code == code && open.tries >= MAX_TRIES);
                if used_up {
                    *p = None;
                }
                used_up
            };
            refuse(&s.writer, "Wrong code".into());
            shared.event(HostEvent::WrongCode { name, closed });
            return None;
        }
        if Trust::update(&shared.dir, |t| t.add_device(&key_hex, &name)).is_err() {
            refuse(&s.writer, format!("{host_name} couldn't save the pairing"));
            return None;
        }
        // One code, one device: pairing another takes a new code.
        shared.close_pairing();
        shared.event(HostEvent::Paired(name.clone()));
        s.writer
            .json(1, &Msg::Paired { proof: crypto::to_hex(&crypto::host_proof(&key, &s.hash)), info: shared.info() })
            .ok()?;
    }
    let _ = Trust::update(&shared.dir, |t| {
        if let Some(d) = t.devices.iter_mut().find(|d| d.key == key_hex) {
            d.last_seen = trust::now();
            d.name.clone_from(&name);
        }
    });
    Some((Peer::new(name, s.remote_key, from, s.writer, true), s.reader))
}

/// What remembered devices may do here.
struct HostHandler {
    shared: Arc<HostShared>,
}

/// Largest attachment of a chat question (a video).
const CHAT_FILE_MAX: u64 = 1024 * 1024 * 1024;
/// Largest start image of an image-to-video job.
const GEN_FILE_MAX: u64 = 256 * 1024 * 1024;

impl Handler for HostHandler {
    fn admit(&self, peer: &Peer, service: &str, body: &Value, blobs: &[BlobMeta]) -> Result<PathBuf, String> {
        let (services, name) = {
            let c = self.shared.config.lock().unwrap_or_else(|e| e.into_inner());
            (c.services, c.name.clone())
        };
        let off = |what: &str| format!("{name} doesn't share {what}");
        if self.shared.stop.load(Ordering::SeqCst) {
            return Err(format!("{name} isn't taking requests right now — try again in a moment"));
        }
        let temp = self.shared.backend.temp_dir();
        if !matches!(service, "llm.ask" | "gen.run" | "files.send") && !blobs.is_empty() {
            return Err("Unexpected files".into());
        }
        match service {
            "info" => Ok(temp),
            "llm.ask" => {
                if !services.chat {
                    return Err(off("its AI chat"));
                }
                if blobs.len() > 8 || blobs.iter().any(|b| b.size > CHAT_FILE_MAX) {
                    return Err("That attachment is too large to send".into());
                }
                Ok(temp)
            }
            "gen.catalog" | "gen.thumbs" | "gen.lora_info" => {
                if !(services.images || services.video) {
                    return Err(off("image or video generation"));
                }
                Ok(temp)
            }
            "gen.run" => {
                match self.shared.backend.gen_is_video(body) {
                    Some(true) if !services.video => return Err(off("video generation")),
                    Some(false) if !services.images => return Err(off("image generation")),
                    None => return Err("Unknown generator".into()),
                    _ => {}
                }
                if blobs.len() > 1 || blobs.iter().any(|b| b.size > GEN_FILE_MAX) {
                    return Err("That start image is too large to send".into());
                }
                Ok(temp)
            }
            "files.offer" => {
                if !services.files {
                    return Err(off("files"));
                }
                Ok(temp)
            }
            "files.send" => {
                if !services.files {
                    return Err(off("files"));
                }
                self.shared.files.admit_send(&peer.key, body, blobs)
            }
            _ => Err("Not supported by this version".into()),
        }
    }

    fn handle(&self, call: Incoming) {
        /// Counts the call as running until it ends — even by a panic, or
        /// the background host would wait for it forever.
        struct Busy<'a>(&'a AtomicUsize);
        impl Drop for Busy<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        self.shared.busy.fetch_add(1, Ordering::SeqCst);
        let _busy = Busy(&self.shared.busy);
        let backend = self.shared.backend.clone();
        let label = match call.service.as_str() {
            "llm.ask" => Some("AI chat"),
            "gen.run" => Some(if backend.gen_is_video(&call.body) == Some(true) { "Generating a video" } else { "Generating an image" }),
            "files.send" => Some("Sending files"),
            _ => None,
        };
        call.peer.set_activity(call.id(), label);
        backend.wake();
        let result = match call.service.as_str() {
            "info" => Ok(serde_json::to_value(self.shared.info()).unwrap_or_default()),
            "llm.ask" => backend.llm_ask(&call),
            "gen.catalog" => backend.gen_catalog(),
            "gen.run" => backend.gen_run(&call),
            "gen.thumbs" => backend.gen_thumbs(&call),
            "gen.lora_info" => backend.gen_lora_info(&call),
            "files.offer" => {
                let (background, unasked) = {
                    let c = self.shared.config.lock().unwrap_or_else(|e| e.into_inner());
                    (c.background, c.accept_files_unasked)
                };
                let key = crypto::to_hex(&call.peer.key);
                let device_ok = Trust::load(&self.shared.dir).device(&key).is_some_and(|d| d.accept_files);
                let result = self.shared.files.offer(&call, unasked || device_ok, background);
                // "Always accept from this PC" was ticked: remember it.
                if result.as_ref().is_ok_and(|v| v.get("always").and_then(Value::as_bool) == Some(true)) {
                    let _ = Trust::update(&self.shared.dir, |t| {
                        if let Some(d) = t.devices.iter_mut().find(|d| d.key == key) {
                            d.accept_files = true;
                        }
                    });
                }
                result
            }
            "files.send" => self.shared.files.receive(&call),
            _ => Err("Not supported".into()),
        };
        call.peer.set_activity(call.id(), None);
        match result {
            Ok(body) => call.done(body),
            Err(e) => call.fail(e),
        }
        backend.wake();
    }
}
