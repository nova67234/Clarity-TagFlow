//! A live session with the other PC: calls in both directions over one
//! encrypted connection.
//!
//! A call is a request (`Call`: a service name, a JSON body and any files)
//! and a stream of replies (`Event`s and `Blob`s — files — then one `Done` or
//! `Fail`). Either side can start calls; the reader thread routes every frame
//! to its call, streams arriving files straight to disk, and starts a thread
//! for each incoming call once its files are in.

use std::collections::HashMap;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::wire::{Frame, JsonAssembler, Reader, Writer, KIND_DATA, KIND_JSON, KIND_JSON_MORE};

/// Bumped when the two sides could no longer understand each other.
pub(crate) const PROTO: u32 = 1;

/// How often an idle session pings, and how long a silent one lives (much
/// shorter in tests, so going silent can be tested).
const PING_EVERY: Duration = if cfg!(test) { Duration::from_millis(400) } else { Duration::from_secs(15) };
const SILENT_LIMIT: Duration = if cfg!(test) { Duration::from_secs(3) } else { Duration::from_secs(50) };

/// Calls one side may have running on the other at once.
const MAX_INCOMING: usize = 16;

/// What the host offers. Checked again by the host on every call — the
/// receiver uses it only to show what's available.
#[derive(Serialize, Deserialize, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Services {
    pub chat: bool,
    pub images: bool,
    pub video: bool,
    pub files: bool,
}

/// What a host tells a receiver about itself.
#[derive(Serialize, Deserialize, Clone, Default, PartialEq, Debug)]
#[serde(default)]
pub(crate) struct HostInfo {
    pub name: String,
    pub version: String,
    pub services: Services,
    /// The AI chat model the host runs (its label), when it has one set up.
    pub chat_model: Option<String>,
    /// That model thinks before it answers (the chat's thinking level
    /// applies).
    pub chat_thinks: bool,
    /// The host is running in the background (no window): it can't ask its
    /// owner anything, e.g. whether to accept files.
    pub background: bool,
}

/// One file of a call.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct BlobMeta {
    pub name: String,
    pub size: u64,
}

/// Every message, as JSON.
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "t", rename_all = "snake_case")]
pub(crate) enum Msg {
    // --- Opening (call 1). ---
    Hello {
        name: String,
        version: String,
        proto: u32,
        /// The receiver's SPAKE2 message (hex) when it has a pairing code.
        #[serde(default)]
        spake: Option<String>,
    },
    Welcome { info: HostInfo },
    PairRequired { name: String },
    PairReply { spake: String },
    PairConfirm { proof: String },
    Paired { proof: String, info: HostInfo },
    Refused { reason: String },
    // --- Control (call 0). ---
    Ping,
    Pong,
    Cancel { call: u32 },
    Info { info: HostInfo },
    // --- Calls. ---
    Call { service: String, body: Value, blobs: Vec<BlobMeta> },
    Event { body: Value },
    Blob { name: String, size: u64 },
    Done { body: Value },
    Fail { message: String },
}

/// What a caller hears back.
#[derive(Debug)]
pub(crate) enum CallEvent {
    Event(Value),
    /// A file arrived; it's the caller's now (move it or delete it).
    Blob { name: String, path: PathBuf },
    Done(Value),
    Fail(String),
    /// The session ended before the call did.
    Closed,
}

/// A file to send with a call.
pub(crate) enum BlobSrc {
    File { name: String, path: PathBuf },
    Bytes { name: String, data: Vec<u8> },
}

impl BlobSrc {
    fn meta(&self) -> Result<BlobMeta, String> {
        Ok(match self {
            BlobSrc::File { name, path } => BlobMeta {
                name: name.clone(),
                size: std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len(),
            },
            BlobSrc::Bytes { name, data } => BlobMeta { name: name.clone(), size: data.len() as u64 },
        })
    }
}

/// The app's side of a session: what it lets the other side do.
pub(crate) trait Handler: Send + Sync + 'static {
    /// May the other side run `service` (with this request and these
    /// files)? `Ok` names the folder its files are written into; `Err` is the
    /// reason it may not.
    fn admit(&self, peer: &Peer, service: &str, body: &Value, blobs: &[BlobMeta]) -> Result<PathBuf, String>;
    /// Serve an admitted call, on its own thread.
    fn handle(&self, call: Incoming);
    /// The host's offer changed (receivers only).
    fn on_info(&self, _peer: &Peer, _info: HostInfo) {}
}

struct CallSlot {
    tx: Sender<CallEvent>,
    /// Where files the other side sends back are written.
    blob_dir: PathBuf,
    /// Set once the other side has ended the call (Done / Fail): files still
    /// going out for it stop.
    over: Arc<AtomicBool>,
}

/// A connected PC.
pub(crate) struct Peer {
    /// Its device name, as it told us.
    pub name: String,
    /// Its identity key.
    pub key: Vec<u8>,
    pub addr: SocketAddr,
    writer: Arc<Writer>,
    next_id: AtomicU32,
    calls: Mutex<HashMap<u32, CallSlot>>,
    /// Cancel flags of the calls we're serving for it.
    incoming: Mutex<HashMap<u32, Arc<AtomicBool>>>,
    closed: AtomicBool,
    /// What it has running here right now (call id, label) — for the UI.
    pub activity: Mutex<Vec<(u32, String)>>,
    /// Replies the reader thread has to send (see `serve`).
    outbox: Mutex<Option<SyncSender<(u32, Msg)>>>,
}

impl Peer {
    /// `host_side`: this end is the host (it numbers its calls even).
    pub fn new(name: String, key: Vec<u8>, addr: SocketAddr, writer: Arc<Writer>, host_side: bool) -> Arc<Peer> {
        Arc::new(Peer {
            name,
            key,
            addr,
            writer,
            // Call 1 is the opening; the receiver's own calls start at 3.
            next_id: AtomicU32::new(if host_side { 2 } else { 3 }),
            calls: Mutex::new(HashMap::new()),
            incoming: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
            activity: Mutex::new(Vec::new()),
            outbox: Mutex::new(None),
        })
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Send a reply from the reader thread: queued for the outbox thread,
    /// so the reader never waits on a write. A peer that keeps it full
    /// without reading is let go.
    fn reply_later(&self, call: u32, msg: Msg) {
        let queued = self.outbox.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|tx| tx.try_send((call, msg)));
        if matches!(queued, Some(Err(_))) {
            self.close();
        }
    }

    /// It has calls running here (or files on their way for one).
    pub fn serving(&self) -> bool {
        !self.incoming.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// End the session (the reader thread sees the socket close and cleans up).
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.writer.shutdown();
    }

    fn ours(&self, call: u32) -> bool {
        // Both sides number their own calls with the parity they started on.
        call % 2 == self.next_id.load(Ordering::SeqCst) % 2
    }

    /// Send a control message (call 0).
    pub fn control(&self, msg: &Msg) -> bool {
        !self.is_closed() && self.writer.json(0, msg).is_ok()
    }

    /// Start a call: send the request and its files (this blocks while the
    /// files go out), then read the replies from the returned [`Call`].
    /// Files the other side sends back are written into `blob_dir`.
    pub fn call(self: &Arc<Self>, service: &str, body: Value, blobs: Vec<BlobSrc>, blob_dir: &Path) -> Result<Call, String> {
        if self.is_closed() {
            return Err("Not connected".into());
        }
        let metas = blobs.iter().map(BlobSrc::meta).collect::<Result<Vec<_>, _>>()?;
        let id = self.next_id.fetch_add(2, Ordering::SeqCst);
        let (tx, rx) = std::sync::mpsc::channel();
        let over = Arc::new(AtomicBool::new(false));
        self.calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, CallSlot { tx, blob_dir: blob_dir.to_path_buf(), over: over.clone() });
        let call = Call { id, peer: self.clone(), rx, finished: false, cancelled: AtomicBool::new(false), over };
        self.writer
            .json(id, &Msg::Call { service: service.to_string(), body, blobs: metas.clone() })
            .map_err(|_| "The connection was lost".to_string())?;
        let stop = || {
            self.is_closed() || call.cancelled.load(Ordering::SeqCst) || call.over.load(Ordering::SeqCst)
        };
        for (src, meta) in blobs.into_iter().zip(metas) {
            let sent = match src {
                BlobSrc::File { path, .. } => std::fs::File::open(&path)
                    .and_then(|mut f| self.writer.data(id, &mut f, meta.size, &stop)),
                BlobSrc::Bytes { data, .. } => self.writer.data(id, &mut data.as_slice(), meta.size, &stop),
            };
            if let Err(e) = sent {
                // Turned down part-way (its reply says why), or failed here:
                // then the other side is left waiting for the rest of the
                // file — cancelling lets it go.
                if call.over.load(Ordering::SeqCst) {
                    return Ok(call);
                }
                call.cancel();
                return Err(format!("Sending {} failed: {e}", meta.name));
            }
        }
        Ok(call)
    }

    /// Note what a call is doing (shown in the host's Link tab).
    pub fn set_activity(&self, call: u32, label: Option<&str>) {
        let mut a = self.activity.lock().unwrap_or_else(|e| e.into_inner());
        a.retain(|(id, _)| *id != call);
        if let Some(l) = label {
            a.push((call, l.to_string()));
        }
    }
}

/// A call this side started.
pub(crate) struct Call {
    pub id: u32,
    peer: Arc<Peer>,
    rx: Receiver<CallEvent>,
    finished: bool,
    cancelled: AtomicBool,
    over: Arc<AtomicBool>,
}

impl Call {
    /// The next reply, or None if none came within `timeout`.
    pub fn next(&mut self, timeout: Duration) -> Option<CallEvent> {
        match self.rx.recv_timeout(timeout) {
            Ok(ev) => {
                if matches!(ev, CallEvent::Done(_) | CallEvent::Fail(_) | CallEvent::Closed) {
                    self.finished = true;
                }
                Some(ev)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                self.finished = true;
                Some(CallEvent::Closed)
            }
        }
    }

    /// Ask the other side to stop. Replies may still arrive (ending in a
    /// `Done` or `Fail`).
    pub fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::SeqCst) && !self.over.load(Ordering::SeqCst) {
            self.peer.control(&Msg::Cancel { call: self.id });
        }
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        self.peer.calls.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
        if !self.finished {
            self.cancel();
        }
        // Files that arrived but were never picked up.
        while let Ok(ev) = self.rx.try_recv() {
            if let CallEvent::Blob { path, .. } = ev {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// A call the other side started, being served here.
pub(crate) struct Incoming {
    pub service: String,
    pub body: Value,
    /// Its files (name as sent, where it was written). The handler moves
    /// what it keeps; the rest is deleted when the call ends.
    pub blobs: Vec<(String, PathBuf)>,
    pub peer: Arc<Peer>,
    id: u32,
    cancel: Arc<AtomicBool>,
    finished: bool,
}

impl Incoming {
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Send a progress event. False once the session is gone.
    pub fn event(&self, body: Value) -> bool {
        !self.peer.is_closed() && self.peer.writer.json(self.id, &Msg::Event { body }).is_ok()
    }

    /// Send a file back.
    pub fn blob(&self, name: &str, path: &Path) -> Result<(), String> {
        let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let size = f.metadata().map_err(|e| e.to_string())?.len();
        self.peer
            .writer
            .json(self.id, &Msg::Blob { name: name.to_string(), size })
            .map_err(|_| "The connection was lost".to_string())?;
        let stop = || self.peer.is_closed() || self.cancel.load(Ordering::SeqCst);
        self.peer.writer.data(self.id, &mut f, size, &stop).map_err(|e| e.to_string())
    }

    pub fn done(mut self, body: Value) {
        self.finished = true;
        let _ = self.peer.writer.json(self.id, &Msg::Done { body });
    }

    pub fn fail(mut self, message: impl Into<String>) {
        self.finished = true;
        let _ = self.peer.writer.json(self.id, &Msg::Fail { message: message.into() });
    }

    /// The other side cancelled the call (or left).
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst) || self.peer.is_closed()
    }

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
}

impl Drop for Incoming {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.peer.writer.json(self.id, &Msg::Fail { message: "Stopped".into() });
        }
        self.peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
        self.peer.set_activity(self.id, None);
        for (_, path) in &self.blobs {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A file being received.
struct BlobRx {
    name: String,
    size: u64,
    written: u64,
    path: PathBuf,
    file: std::fs::File,
}

impl BlobRx {
    fn start(dir: &Path, name: &str, size: u64) -> std::io::Result<BlobRx> {
        std::fs::create_dir_all(dir)?;
        let path = (0..1000u32)
            .map(|n| dir.join(format!(".link-{}-{}-{n}.part", std::process::id(), unique())))
            .find(|p| !p.exists())
            .ok_or_else(|| std::io::Error::other("no free temporary name"))?;
        let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        Ok(BlobRx { name: name.to_string(), size, written: 0, path, file })
    }

    /// Write a frame's bytes; true when the file is complete.
    fn feed(&mut self, data: &[u8]) -> std::io::Result<bool> {
        if self.written + data.len() as u64 > self.size {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "more data than announced"));
        }
        self.file.write_all(data)?;
        self.written += data.len() as u64;
        Ok(self.written == self.size)
    }

    fn discard(self) {
        drop(self.file);
        let _ = std::fs::remove_file(&self.path);
    }
}

fn unique() -> u64 {
    static N: AtomicU32 = AtomicU32::new(0);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    (t << 16) ^ N.fetch_add(1, Ordering::Relaxed) as u64
}

/// An incoming call waiting for its files.
struct PendingCall {
    service: String,
    body: Value,
    metas: Vec<BlobMeta>,
    dir: PathBuf,
    received: Vec<(String, PathBuf)>,
}

/// Run the session until it ends: route frames, keep it alive, start
/// incoming calls. Returns why it ended.
pub(crate) fn serve(peer: Arc<Peer>, mut reader: Reader, handler: Arc<dyn Handler>) -> String {
    // A watchdog pings, and ends the session when the other side has gone
    // silent. Reads wait a second at a time (with a peek — see
    // `Reader::fill`), so a session closed from another thread (the watchdog,
    // Disconnect, Off) ends within a second on every OS: on Windows, shutting
    // the socket down doesn't wake a read that's waiting.
    reader.set_timeout(Some(Duration::from_secs(1)));
    let started = Instant::now();
    // When a frame last arrived (ms after `started`), and whether the
    // watchdog ended the session for silence.
    let heard = Arc::new(AtomicU64::new(0));
    let silent = Arc::new(AtomicBool::new(false));
    {
        let (peer, heard, silent) = (peer.clone(), heard.clone(), silent.clone());
        std::thread::spawn(move || {
            let mut pinged = Instant::now();
            while !peer.is_closed() {
                std::thread::sleep(Duration::from_millis(500));
                let quiet = started.elapsed().saturating_sub(Duration::from_millis(heard.load(Ordering::SeqCst)));
                if quiet > SILENT_LIMIT {
                    silent.store(true, Ordering::SeqCst);
                    peer.close();
                    break;
                }
                if pinged.elapsed() >= PING_EVERY {
                    pinged = Instant::now();
                    if !peer.control(&Msg::Ping) {
                        peer.close();
                        break;
                    }
                }
            }
        });
    }
    // Replies the reader itself sends (refusals, cancels) go out from a
    // thread of their own: if the reader waited on a write, two PCs both
    // sending flat out could each wait for the other to read.
    {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(u32, Msg)>(256);
        *peer.outbox.lock().unwrap_or_else(|e| e.into_inner()) = Some(tx);
        let peer = peer.clone();
        std::thread::spawn(move || {
            while let Ok((call, msg)) = rx.recv() {
                let sent = if call == 0 { peer.control(&msg) } else { peer.writer.json(call, &msg).is_ok() };
                if !sent {
                    break;
                }
            }
        });
    }
    let ended = |e: Option<std::io::Error>| -> String {
        if silent.load(Ordering::SeqCst) {
            return "The other PC stopped answering".to_string();
        }
        match e {
            Some(e) if matches!(e.kind(), std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset) => {
                "The other PC closed the connection".to_string()
            }
            _ if peer.is_closed() => "Disconnected".to_string(),
            Some(e) => e.to_string(),
            None => "Disconnected".to_string(),
        }
    };
    let mut json = JsonAssembler::default();
    let mut blobs: HashMap<u32, BlobRx> = HashMap::new();
    let mut pending: HashMap<u32, PendingCall> = HashMap::new();
    let why = loop {
        if peer.is_closed() {
            break ended(None);
        }
        let frame = match reader.next() {
            Ok(f) => f,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(e) => break ended(Some(e)),
        };
        heard.store(started.elapsed().as_millis() as u64, Ordering::SeqCst);
        match frame.kind {
            KIND_DATA => on_data(&peer, &handler, frame, &mut blobs, &mut pending),
            KIND_JSON | KIND_JSON_MORE => match json.push(&frame) {
                Ok(Some(bytes)) => match serde_json::from_slice::<Msg>(&bytes) {
                    Ok(msg) => on_msg(&peer, &handler, frame.call, msg, &mut blobs, &mut pending),
                    // A message this version doesn't know: answer a call
                    // with a failure, ignore anything else.
                    Err(_) if frame.call != 0 && !peer.ours(frame.call) => {
                        peer.reply_later(frame.call, Msg::Fail { message: "Not supported by this version".into() });
                    }
                    Err(_) => {}
                },
                Ok(None) => {}
                Err(e) => break e.to_string(),
            },
            _ => break "Unexpected data".to_string(),
        }
    };
    peer.close();
    // The outbox thread ends (it holds the peer).
    *peer.outbox.lock().unwrap_or_else(|e| e.into_inner()) = None;
    // Everything still waiting hears that the session is over.
    for (_, slot) in peer.calls.lock().unwrap_or_else(|e| e.into_inner()).drain() {
        let _ = slot.tx.send(CallEvent::Closed);
    }
    for flag in peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).values() {
        flag.store(true, Ordering::SeqCst);
    }
    for (_, b) in blobs.drain() {
        b.discard();
    }
    for (_, p) in pending.drain() {
        for (_, path) in p.received {
            let _ = std::fs::remove_file(path);
        }
    }
    why
}

fn on_data(
    peer: &Arc<Peer>,
    handler: &Arc<dyn Handler>,
    frame: Frame,
    blobs: &mut HashMap<u32, BlobRx>,
    pending: &mut HashMap<u32, PendingCall>,
) {
    let call = frame.call;
    // Data for a call that's been cancelled or failed: dropped.
    let Some(rx) = blobs.get_mut(&call) else { return };
    match rx.feed(&frame.body) {
        Ok(false) => {}
        Ok(true) => {
            let rx = blobs.remove(&call).expect("present");
            let (name, path) = (rx.name.clone(), rx.path.clone());
            drop(rx.file);
            blob_complete(peer, handler, call, name, path, blobs, pending);
        }
        Err(e) => {
            if let Some(rx) = blobs.remove(&call) {
                rx.discard();
            }
            fail_receiving(peer, call, &format!("Couldn't save a file: {e}"), pending);
        }
    }
}

/// A file finished arriving: hand it to its call, and start the next one (or
/// the call itself, once all of an incoming call's files are in).
fn blob_complete(
    peer: &Arc<Peer>,
    handler: &Arc<dyn Handler>,
    call: u32,
    name: String,
    path: PathBuf,
    blobs: &mut HashMap<u32, BlobRx>,
    pending: &mut HashMap<u32, PendingCall>,
) {
    if peer.ours(call) {
        let slot = peer.calls.lock().unwrap_or_else(|e| e.into_inner()).get(&call).map(|s| s.tx.clone());
        match slot {
            Some(tx) if tx.send(CallEvent::Blob { name, path: path.clone() }).is_ok() => {}
            _ => {
                let _ = std::fs::remove_file(&path);
            }
        }
        return;
    }
    let Some(p) = pending.get_mut(&call) else {
        let _ = std::fs::remove_file(&path);
        return;
    };
    p.received.push((name, path));
    if let Some(next) = p.metas.get(p.received.len()).cloned() {
        match BlobRx::start(&p.dir, &next.name, next.size) {
            Ok(rx) if next.size == 0 => {
                let (n, path) = (rx.name.clone(), rx.path.clone());
                drop(rx.file);
                blob_complete(peer, handler, call, n, path, blobs, pending);
            }
            Ok(rx) => {
                blobs.insert(call, rx);
            }
            Err(e) => fail_receiving(peer, call, &format!("Couldn't save a file: {e}"), pending),
        }
        return;
    }
    let p = pending.remove(&call).expect("present");
    start_incoming(peer, handler, call, p.service, p.body, p.received);
}

fn fail_receiving(peer: &Arc<Peer>, call: u32, why: &str, pending: &mut HashMap<u32, PendingCall>) {
    if peer.ours(call) {
        let slot = peer.calls.lock().unwrap_or_else(|e| e.into_inner()).get(&call).map(|s| s.tx.clone());
        if let Some(tx) = slot {
            let _ = tx.send(CallEvent::Fail(why.to_string()));
        }
        peer.reply_later(0, Msg::Cancel { call });
        return;
    }
    if let Some(p) = pending.remove(&call) {
        for (_, path) in p.received {
            let _ = std::fs::remove_file(path);
        }
    }
    peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).remove(&call);
    peer.reply_later(call, Msg::Fail { message: why.to_string() });
}

fn start_incoming(
    peer: &Arc<Peer>,
    handler: &Arc<dyn Handler>,
    call: u32,
    service: String,
    body: Value,
    received: Vec<(String, PathBuf)>,
) {
    let cancel = peer
        .incoming
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&call)
        .cloned()
        .unwrap_or_default();
    let incoming = Incoming { service, body, blobs: received, peer: peer.clone(), id: call, cancel, finished: false };
    let handler = handler.clone();
    let spawned = std::thread::Builder::new().name(format!("link-call-{call}")).spawn(move || handler.handle(incoming));
    if spawned.is_err() {
        peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).remove(&call);
        peer.reply_later(call, Msg::Fail { message: "The other PC is too busy".into() });
    }
}

fn on_msg(
    peer: &Arc<Peer>,
    handler: &Arc<dyn Handler>,
    call: u32,
    msg: Msg,
    blobs: &mut HashMap<u32, BlobRx>,
    pending: &mut HashMap<u32, PendingCall>,
) {
    if call == 0 {
        match msg {
            // Pings only show the other side is there (it counts as heard);
            // each side pings, so nothing is sent back.
            Msg::Ping | Msg::Pong => {}
            Msg::Cancel { call } => {
                if let Some(flag) = peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).get(&call) {
                    flag.store(true, Ordering::SeqCst);
                }
                // A call cancelled while its files were still coming in.
                if pending.remove(&call).is_some() {
                    if let Some(rx) = blobs.remove(&call) {
                        rx.discard();
                    }
                    peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).remove(&call);
                    peer.reply_later(call, Msg::Fail { message: "Cancelled".into() });
                }
            }
            Msg::Info { info } => handler.on_info(peer, info),
            _ => {}
        }
        return;
    }
    if peer.ours(call) {
        // A reply to a call we started. A message that arrives while one of
        // its files is half in means that file was abandoned.
        if let Some(rx) = blobs.remove(&call) {
            rx.discard();
        }
        let slot = peer
            .calls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&call)
            .map(|s| (s.tx.clone(), s.blob_dir.clone(), s.over.clone()));
        let Some((tx, dir, over)) = slot else { return };
        match msg {
            Msg::Event { body } => {
                let _ = tx.send(CallEvent::Event(body));
            }
            Msg::Blob { name, size } => match BlobRx::start(&dir, &name, size) {
                Ok(rx) if size == 0 => {
                    let path = rx.path.clone();
                    drop(rx.file);
                    if tx.send(CallEvent::Blob { name, path: path.clone() }).is_err() {
                        let _ = std::fs::remove_file(path);
                    }
                }
                Ok(rx) => {
                    blobs.insert(call, rx);
                }
                Err(e) => fail_receiving(peer, call, &format!("Couldn't save a file: {e}"), pending),
            },
            Msg::Done { body } => {
                over.store(true, Ordering::SeqCst);
                let _ = tx.send(CallEvent::Done(body));
                peer.calls.lock().unwrap_or_else(|e| e.into_inner()).remove(&call);
            }
            Msg::Fail { message } => {
                over.store(true, Ordering::SeqCst);
                let _ = tx.send(CallEvent::Fail(message));
                peer.calls.lock().unwrap_or_else(|e| e.into_inner()).remove(&call);
            }
            _ => {}
        }
        return;
    }
    // A call the other side is starting.
    let Msg::Call { service, body, blobs: metas } = msg else { return };
    let known = pending.contains_key(&call) || peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&call);
    if known {
        return;
    }
    if peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).len() >= MAX_INCOMING {
        peer.reply_later(call, Msg::Fail { message: "Too many requests at once — try again".into() });
        return;
    }
    let dir = match handler.admit(peer, &service, &body, &metas) {
        Ok(dir) => dir,
        Err(reason) => {
            // Its files (if any) are already on the way: they're dropped as
            // data for an unknown call.
            peer.reply_later(call, Msg::Fail { message: reason });
            return;
        }
    };
    peer.incoming.lock().unwrap_or_else(|e| e.into_inner()).insert(call, Arc::new(AtomicBool::new(false)));
    let Some(first) = metas.first().cloned() else {
        start_incoming(peer, handler, call, service, body, Vec::new());
        return;
    };
    pending.insert(call, PendingCall { service, body, metas, dir: dir.clone(), received: Vec::new() });
    match BlobRx::start(&dir, &first.name, first.size) {
        Ok(rx) if first.size == 0 => {
            let (name, path) = (rx.name.clone(), rx.path.clone());
            drop(rx.file);
            blob_complete(peer, handler, call, name, path, blobs, pending);
        }
        Ok(rx) => {
            blobs.insert(call, rx);
        }
        Err(e) => fail_receiving(peer, call, &format!("Couldn't save a file: {e}"), pending),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::crypto::Identity;
    use crate::link::wire::{handshake_initiator, handshake_responder};
    use std::net::{TcpListener, TcpStream};

    struct Refuse;

    impl Handler for Refuse {
        fn admit(&self, _: &Peer, _: &str, _: &Value, _: &[BlobMeta]) -> Result<PathBuf, String> {
            Err("no".into())
        }
        fn handle(&self, _: Incoming) {}
    }

    /// A peer that completes the handshake and then says nothing (not even
    /// answering pings) is dropped; one that answers stays.
    #[test]
    fn silent_peers_are_dropped() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let host_id = Identity::generate();
        let server = std::thread::spawn(move || {
            let (stream, from) = listener.accept().unwrap();
            let s = handshake_responder(stream, &host_id).unwrap();
            let peer = Peer::new("RX".into(), s.remote_key, from, s.writer, true);
            let started = Instant::now();
            let why = serve(peer, s.reader, Arc::new(Refuse));
            (why, started.elapsed())
        });
        // Handshake, then silence (the socket stays open).
        let _quiet = handshake_initiator(TcpStream::connect(addr).unwrap(), &Identity::generate()).unwrap();
        let (why, after) = server.join().unwrap();
        assert_eq!(why, "The other PC stopped answering");
        assert!(after >= SILENT_LIMIT && after < SILENT_LIMIT + Duration::from_secs(5), "{after:?}");
    }
}
