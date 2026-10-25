//! The receiver: connects to a host — pairing with its code the first time —
//! keeps the session up, and reconnects after it drops.

use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::crypto::{self, Identity};
use super::peer::{self, BlobMeta, Handler, HostInfo, Incoming, Msg, Peer, PROTO};
use super::trust::{self, Trust};
use super::{addr, discover, files, wire, Backend};

/// Where the receiver stands, for the Link tab.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum Status {
    Connecting,
    /// The host doesn't know this PC yet: its code is needed.
    NeedCode { host: String },
    Connected,
    /// Lost the host; trying again shortly.
    Retrying { why: String },
    /// Gave up (refused, wrong code, not found…).
    Failed(String),
}

pub(crate) struct ClientShared {
    identity: Identity,
    dir: PathBuf,
    name: String,
    pub status: Mutex<Status>,
    pub peer: Mutex<Option<Arc<Peer>>>,
    pub info: Mutex<Option<HostInfo>>,
    stop: AtomicBool,
    backend: Arc<dyn Backend>,
    files: Arc<files::FileHub>,
}

impl ClientShared {
    fn set(&self, s: Status) {
        *self.status.lock().unwrap_or_else(|e| e.into_inner()) = s;
        self.backend.wake();
    }

    pub fn status(&self) -> Status {
        self.status.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn peer(&self) -> Option<Arc<Peer>> {
        self.peer.lock().unwrap_or_else(|e| e.into_inner()).clone().filter(|p| !p.is_closed())
    }

    pub fn info(&self) -> Option<HostInfo> {
        self.info.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// A running connection (and its reconnects).
pub(crate) struct Client {
    pub shared: Arc<ClientShared>,
    /// What it connects to: an address as typed, or a remembered host.
    pub target: String,
}

enum Fail {
    /// Worth trying again later (network trouble, the host not running).
    Network(String),
    /// Another PC answered at that address (the host's address went to
    /// someone else, or a stranger answered discovery): keep looking.
    WrongPc(String),
    NeedCode(String),
    /// Not worth retrying until something changes (wrong code, refused).
    Refused(String),
}

impl Client {
    /// Connect to `target` ("192.168.1.23", "host:port"), pairing with `code`
    /// when given. `known_key`: the remembered host this address is for, if
    /// any — then the PC that answers must be that host.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        identity: Identity,
        dir: PathBuf,
        name: String,
        target: String,
        known_key: Option<String>,
        code: Option<String>,
        backend: Arc<dyn Backend>,
        files: Arc<files::FileHub>,
    ) -> Client {
        let shared = Arc::new(ClientShared {
            identity,
            dir,
            name,
            status: Mutex::new(Status::Connecting),
            peer: Mutex::new(None),
            info: Mutex::new(None),
            stop: AtomicBool::new(false),
            backend,
            files,
        });
        let s = shared.clone();
        let t = target.clone();
        let _ = std::thread::Builder::new()
            .name("link-receiver".into())
            .spawn(move || run(s, t, known_key, code));
        Client { shared, target }
    }

    pub fn stop(&self) {
        // Set before taking the session: one still opening sees it when it
        // goes to store itself (see `connect`).
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(p) = self.shared.peer.lock().unwrap_or_else(|e| e.into_inner()).take() {
            p.close();
        }
        *self.shared.info.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(shared: Arc<ClientShared>, mut target: String, mut known_key: Option<String>, mut code: Option<String>) {
    let mut backoff = Duration::from_secs(3);
    // Where the host was last reached, and addresses where another PC
    // answered instead (skipped when looking for it).
    let mut home = target.clone();
    let mut not_it: Vec<String> = Vec::new();
    // The last try was at an address discovery just found (tried at once,
    // once a round: answers at ever new addresses can't make it spin).
    let mut tried_found = false;
    loop {
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        shared.set(Status::Connecting);
        let result = connect(&shared, &target, known_key.as_deref(), code.take());
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        let failed = match result {
            // Was connected; the session ended.
            Ok(key) => {
                known_key = Some(key);
                home = target.clone();
                not_it.clear();
                tried_found = false;
                backoff = Duration::from_secs(3);
                shared.set(Status::Retrying { why: "Connection lost — reconnecting…".into() });
                None
            }
            Err(Fail::NeedCode(host)) => {
                shared.set(Status::NeedCode { host });
                return;
            }
            Err(Fail::Refused(why)) => {
                shared.set(Status::Failed(why));
                return;
            }
            Err(Fail::Network(why)) => Some((why, false)),
            Err(Fail::WrongPc(why)) => Some((why, true)),
        };
        if let Some((why, wrong_pc)) = failed {
            // Only a remembered host is retried (and looked for, should its
            // address have changed); a typed address that doesn't answer is
            // just reported.
            let Some(key) = known_key.clone() else {
                shared.set(Status::Failed(why));
                return;
            };
            if wrong_pc && !not_it.contains(&target) {
                not_it.push(target.clone());
            }
            shared.set(Status::Retrying { why });
            // Found at a new address: tried at once. Otherwise back where it
            // was, after the wait.
            if !tried_found {
                let port =
                    addr::parse_host(&home).ok().and_then(|a| a.first().map(|a| a.port())).unwrap_or(addr::DEFAULT_PORT);
                let mut skip = not_it.clone();
                skip.extend([target.clone(), home.clone()]);
                if let Some(found) = find_known(&key, port, &skip) {
                    target = found;
                    tried_found = true;
                    continue;
                }
            }
            tried_found = false;
            target = home.clone();
        }
        // Wait before the next try, checking for Stop.
        let until = Instant::now() + backoff;
        while Instant::now() < until {
            if shared.stop.load(Ordering::SeqCst) {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

/// Where a remembered host answers discovery (its address may have changed),
/// leaving out addresses where another PC turned out to answer — anyone can
/// claim a fingerprint, the connection then checks the whole key.
fn find_known(key_hex: &str, port: u16, not_it: &[String]) -> Option<String> {
    let key = crypto::from_hex(key_hex)?;
    let fp = crypto::fingerprint(&key);
    let found = discover::find(port, Duration::from_millis(1200));
    found.into_iter().filter(|f| f.fp == fp).map(|f| f.addr()).find(|a| !not_it.contains(a))
}

/// Marks the reason another PC answered (the receiver keeps looking).
const WRONG_PC: &str = "isn't the host this PC paired with";

/// One connection: open it, run it until it ends. `Ok` carries the host's
/// key (it was connected).
fn connect(shared: &Arc<ClientShared>, target: &str, known_key: Option<&str>, code: Option<String>) -> Result<String, Fail> {
    let addrs = addr::parse_host(target).map_err(Fail::Refused)?;
    let mut last_err = String::new();
    let stream = addrs
        .iter()
        .find_map(|a| match TcpStream::connect_timeout(a, Duration::from_secs(4)) {
            Ok(s) => Some(s),
            Err(e) => {
                last_err = match e.kind() {
                    std::io::ErrorKind::ConnectionRefused => {
                        format!("Nothing answered at {target} — is Link set to Host there (or running in the background)?")
                    }
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => {
                        format!("{target} didn't answer — check the address and that both PCs are on the same network (a firewall may be blocking it)")
                    }
                    _ => format!("Couldn't reach {target}: {e}"),
                };
                None
            }
        })
        .ok_or_else(|| Fail::Network(last_err.clone()))?;
    let from = stream.peer_addr().map_err(|e| Fail::Network(e.to_string()))?;
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let deadline = wire::OpeningDeadline::start(&stream, super::host::OPENING_LIMIT);
    let mut s = wire::handshake_initiator(stream, &shared.identity).map_err(Fail::Network)?;
    let host_hex = crypto::to_hex(&s.remote_key);
    let known = Trust::load(&shared.dir).host(&host_hex).cloned();
    if let Some(expected) = known_key
        && expected != host_hex
        && code.is_none()
    {
        return Err(Fail::WrongPc(format!(
            "The PC at {target} {WRONG_PC} (its identity is different) — still looking. If the host was reinstalled, forget it and pair again with its code."
        )));
    }
    // An unknown host has to prove it knows the code before it's trusted.
    let pairing = match (&known, &code) {
        (Some(_), None) => None,
        (_, Some(c)) => Some(crypto::pair_receiver_start(c)),
        (None, None) => None,
    };
    s.writer
        .json(
            1,
            &Msg::Hello {
                name: shared.name.clone(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                proto: PROTO,
                spake: pairing.as_ref().map(|(_, m)| crypto::to_hex(m)),
            },
        )
        .map_err(|e| Fail::Network(e.to_string()))?;
    let read = |s: &mut wire::Session| -> Result<Msg, Fail> {
        let f = s.reader.next().map_err(|e| Fail::Network(e.to_string()))?;
        serde_json::from_slice::<Msg>(&f.body).map_err(|_| Fail::Refused("The host sent something unexpected".into()))
    };
    let info = match read(&mut s)? {
        Msg::Welcome { info } => {
            if known.is_none() {
                // It knows this PC, but this PC has no record of it: the
                // code is needed to trust it again.
                return Err(Fail::NeedCode(super::host::clean_name(&info.name)));
            }
            info
        }
        Msg::PairRequired { name } => return Err(Fail::NeedCode(super::host::clean_name(&name))),
        Msg::Refused { reason } => return Err(Fail::Refused(reason)),
        Msg::PairReply { spake } => {
            let Some((state, _)) = pairing else {
                return Err(Fail::Refused("The host started pairing unasked".into()));
            };
            let reply = crypto::from_hex(&spake).ok_or_else(|| Fail::Refused("Bad pairing reply".into()))?;
            let key = crypto::pair_receiver_finish(state, &reply).map_err(Fail::Refused)?;
            s.writer
                .json(1, &Msg::PairConfirm { proof: crypto::to_hex(&crypto::receiver_proof(&key, &s.hash)) })
                .map_err(|e| Fail::Network(e.to_string()))?;
            match read(&mut s)? {
                Msg::Paired { proof, info } => {
                    let ok = crypto::from_hex(&proof).is_some_and(|p| crypto::check_host_proof(&key, &s.hash, &p));
                    if !ok {
                        return Err(Fail::Refused(
                            "The host couldn't prove it knows the code — not connected (another PC may be posing as it)".into(),
                        ));
                    }
                    Trust::update(&shared.dir, |t| {
                        t.add_host(&host_hex, &super::host::clean_name(&info.name), &from.to_string())
                    })
                    .map_err(|e| Fail::Refused(format!("Couldn't save the pairing: {e}")))?;
                    info
                }
                Msg::Refused { reason } => return Err(Fail::Refused(reason)),
                _ => return Err(Fail::Refused("The host sent something unexpected".into())),
            }
        }
        _ => return Err(Fail::Refused("The host sent something unexpected".into())),
    };
    // Remember where the host was reached (its address may change).
    let _ = Trust::update(&shared.dir, |t| {
        if let Some(h) = t.hosts.iter_mut().find(|h| h.key == host_hex) {
            h.addr = from.to_string();
            h.last_seen = trust::now();
            h.name = super::host::clean_name(&info.name);
        }
    });
    drop(deadline);
    let peer = Peer::new(super::host::clean_name(&info.name), s.remote_key.clone(), from, s.writer.clone(), false);
    {
        // Disconnect / Off while this was opening: it doesn't stay.
        let mut slot = shared.peer.lock().unwrap_or_else(|e| e.into_inner());
        if shared.stop.load(Ordering::SeqCst) {
            peer.close();
            return Ok(host_hex);
        }
        *slot = Some(peer.clone());
        *shared.info.lock().unwrap_or_else(|e| e.into_inner()) = Some(info);
    }
    shared.set(Status::Connected);
    let handler: Arc<dyn Handler> = Arc::new(ReceiverHandler { shared: shared.clone() });
    let _why = peer::serve(peer.clone(), s.reader, handler);
    *shared.peer.lock().unwrap_or_else(|e| e.into_inner()) = None;
    *shared.info.lock().unwrap_or_else(|e| e.into_inner()) = None;
    Ok(host_hex)
}

/// What the host may do on this PC: send files (if the owner accepts).
struct ReceiverHandler {
    shared: Arc<ClientShared>,
}

impl Handler for ReceiverHandler {
    fn admit(&self, peer: &Peer, service: &str, body: &Value, blobs: &[BlobMeta]) -> Result<PathBuf, String> {
        match service {
            "files.offer" if blobs.is_empty() => Ok(self.shared.backend.temp_dir()),
            "files.send" => self.shared.files.admit_send(&peer.key, body, blobs),
            _ => Err("Not supported".into()),
        }
    }

    fn handle(&self, call: Incoming) {
        let result = match call.service.as_str() {
            "files.offer" => {
                let key = crypto::to_hex(&call.peer.key);
                let accept_now = Trust::load(&self.shared.dir).host(&key).is_some_and(|h| h.accept_files);
                let result = self.shared.files.offer(&call, accept_now, false);
                // "Always accept from this PC" was ticked: remember it.
                if result.as_ref().is_ok_and(|v| v.get("always").and_then(Value::as_bool) == Some(true)) {
                    let _ = Trust::update(&self.shared.dir, |t| {
                        if let Some(h) = t.hosts.iter_mut().find(|h| h.key == key) {
                            h.accept_files = true;
                        }
                    });
                }
                result
            }
            "files.send" => self.shared.files.receive(&call),
            _ => Err("Not supported".into()),
        };
        match result {
            Ok(v) => call.done(v),
            Err(e) => call.fail(e),
        }
    }

    fn on_info(&self, _peer: &Peer, info: HostInfo) {
        *self.shared.info.lock().unwrap_or_else(|e| e.into_inner()) = Some(info);
        self.shared.backend.wake();
    }
}
