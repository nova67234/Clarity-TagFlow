//! Sending images and videos (with their .txt tags) between the two PCs.
//!
//! A send is two calls: an offer (the list of names and sizes), which the
//! other side's owner accepts or declines — or which is accepted at once for
//! a device they allowed to send without asking — and then the files
//! themselves, admitted only with the token the accepted offer handed out,
//! for exactly the files offered.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::peer::{BlobMeta, BlobSrc, CallEvent, Incoming, Peer};

/// Files in one send.
const MAX_FILES: usize = 500;
/// Largest single file.
const MAX_FILE: u64 = 64 * 1024 * 1024 * 1024;
/// How long the owner has to answer an offer.
const ANSWER_WAIT: Duration = Duration::from_secs(150);
/// How long an accepted offer's token stays good.
const TOKEN_LIFE: Duration = Duration::from_secs(15 * 60);

/// An offer waiting for the owner's answer, as the app shows it.
#[derive(Clone, Debug)]
pub(crate) struct OfferView {
    pub id: u64,
    pub from: String,
    pub names: Vec<String>,
    pub total: u64,
}

/// The owner's answer.
pub(crate) enum Answer {
    Accept { always: bool },
    Decline,
}

struct Pending {
    view: OfferView,
    reply: Sender<Answer>,
}

struct Approved {
    token: String,
    peer_key: Vec<u8>,
    files: Vec<BlobMeta>,
    until: Instant,
}

/// Something to tell the owner.
#[derive(Debug, Clone)]
pub(crate) enum FileEvent {
    Received { from: String, count: usize, dir: PathBuf },
    Sent { to: String, count: usize },
    NotSent { to: String, why: String },
}

/// File sharing state, shared by the host / receiver threads and the app.
pub(crate) struct FileHub {
    pending: Mutex<Vec<Pending>>,
    approved: Mutex<Vec<Approved>>,
    received_dir: Mutex<PathBuf>,
    events: Mutex<Vec<FileEvent>>,
    next_id: AtomicU64,
    /// Sends running from this side.
    pub sending: AtomicU64,
    wake: Box<dyn Fn() + Send + Sync>,
    quiet: AtomicBool,
}

/// Where received files go unless another folder is set (looked up once —
/// on Linux that reads the user-dirs file).
pub(crate) fn default_received_dir() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        dirs::picture_dir()
            .or_else(dirs::home_dir)
            .map(|d| d.join("Clarity TagFlow").join("Received"))
            .unwrap_or_else(|| crate::tagger::models_root().join("received"))
    })
    .clone()
}

impl FileHub {
    pub fn new(wake: Box<dyn Fn() + Send + Sync>) -> FileHub {
        FileHub {
            pending: Mutex::new(Vec::new()),
            approved: Mutex::new(Vec::new()),
            received_dir: Mutex::new(default_received_dir()),
            events: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            sending: AtomicU64::new(0),
            wake,
            quiet: AtomicBool::new(false),
        }
    }

    pub fn set_received_dir(&self, dir: PathBuf) {
        *self.received_dir.lock().unwrap_or_else(|e| e.into_inner()) = dir;
    }

    pub fn received_dir(&self) -> PathBuf {
        self.received_dir.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Offers waiting for an answer.
    pub fn pending(&self) -> Vec<OfferView> {
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|p| p.view.clone()).collect()
    }

    /// Answer offer `id`.
    pub fn answer(&self, id: u64, answer: Answer) {
        let mut p = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = p.iter().position(|o| o.view.id == id) {
            let o = p.remove(i);
            let _ = o.reply.send(answer);
        }
    }

    pub fn take_events(&self) -> Vec<FileEvent> {
        std::mem::take(&mut *self.events.lock().unwrap_or_else(|e| e.into_inner()))
    }

    fn event(&self, e: FileEvent) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(e);
        (self.wake)();
    }

    /// No window to ask in (the background host).
    pub fn set_quiet(&self, quiet: bool) {
        self.quiet.store(quiet, Ordering::SeqCst);
    }

    /// The other PC offers files. `accept_now`: its owner allowed it to send
    /// without asking. Returns `{accept, token | reason, always}` — `always`:
    /// the owner just ticked "always accept from this PC" (the caller
    /// remembers that).
    pub fn offer(&self, call: &Incoming, accept_now: bool, background: bool) -> Result<Value, String> {
        let files: Vec<BlobMeta> = serde_json::from_value(call.body.get("files").cloned().unwrap_or_default())
            .map_err(|_| "Bad file list".to_string())?;
        check_list(&files)?;
        let mut always = false;
        if !accept_now {
            if background || self.quiet.load(Ordering::SeqCst) {
                return Ok(json!({
                    "accept": false,
                    "reason": "it's running in the background, with nobody to accept them — its owner can allow your files in its Link settings",
                }));
            }
            let (tx, rx): (Sender<Answer>, Receiver<Answer>) = std::sync::mpsc::channel();
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let view = OfferView {
                id,
                from: call.peer.name.clone(),
                names: files.iter().map(|f| f.name.clone()).collect(),
                total: files.iter().map(|f| f.size).sum(),
            };
            self.pending.lock().unwrap_or_else(|e| e.into_inner()).push(Pending { view, reply: tx });
            (self.wake)();
            let started = Instant::now();
            let answer = loop {
                match rx.recv_timeout(Duration::from_millis(300)) {
                    Ok(a) => break Some(a),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                        if !call.cancelled() && started.elapsed() < ANSWER_WAIT => {}
                    _ => break None,
                }
            };
            self.pending.lock().unwrap_or_else(|e| e.into_inner()).retain(|p| p.view.id != id);
            (self.wake)();
            match answer {
                Some(Answer::Accept { always: a }) => always = a,
                Some(Answer::Decline) => return Ok(json!({ "accept": false, "reason": "they declined" })),
                None => return Ok(json!({ "accept": false, "reason": "nobody answered" })),
            }
        }
        let token = super::crypto::to_hex(&{
            let mut b = [0u8; 16];
            getrandom::getrandom(&mut b).map_err(|e| e.to_string())?;
            b
        });
        let mut a = self.approved.lock().unwrap_or_else(|e| e.into_inner());
        a.retain(|x| x.until > Instant::now());
        a.push(Approved { token: token.clone(), peer_key: call.peer.key.clone(), files, until: Instant::now() + TOKEN_LIFE });
        Ok(json!({ "accept": true, "token": token, "always": always }))
    }

    /// Admit the files of an accepted offer: the token must be this device's
    /// and the files exactly those offered. Returns where they're written.
    pub fn admit_send(&self, peer_key: &[u8], body: &Value, blobs: &[BlobMeta]) -> Result<PathBuf, String> {
        let token = body.get("token").and_then(Value::as_str).unwrap_or_default();
        let mut a = self.approved.lock().unwrap_or_else(|e| e.into_inner());
        let i = a
            .iter()
            .position(|x| x.token == token && x.peer_key == peer_key && x.until > Instant::now())
            .ok_or("Those files weren't accepted")?;
        let same = a[i].files.len() == blobs.len()
            && a[i].files.iter().zip(blobs).all(|(x, y)| x.name == y.name && x.size == y.size);
        if !same {
            return Err("Those aren't the files that were accepted".into());
        }
        a.remove(i);
        let dir = self.received_dir();
        std::fs::create_dir_all(&dir).map_err(|e| format!("Couldn't create {}: {e}", dir.display()))?;
        Ok(dir)
    }

    /// The accepted files arrived: give each its name (a free one — nothing
    /// in the folder is replaced), with each .txt following its image.
    pub fn receive(&self, call: &Incoming) -> Result<Value, String> {
        let dir = self.received_dir();
        let mut placed: Vec<(String, PathBuf)> = Vec::new();
        let mut saved = 0usize;
        let is_txt = |n: &str| Path::new(n).extension().is_some_and(|e| e.eq_ignore_ascii_case("txt"));
        for (name, path) in call.blobs.iter().filter(|(n, _)| !is_txt(n)) {
            let Some(dest) = crate::move_destination(Path::new(name), &dir) else { continue };
            if std::fs::rename(path, &dest).is_ok() {
                saved += 1;
                let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
                placed.push((stem, dest));
            }
        }
        // Tags go beside the image they belong to, under its (maybe new) name.
        for (name, path) in call.blobs.iter().filter(|(n, _)| is_txt(n)) {
            let stem = Path::new(name).file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
            if let Some((_, img)) = placed.iter().find(|(s, _)| *s == stem) {
                let txt = crate::right_details::sidecar_txt(img);
                if !txt.exists() {
                    let _ = std::fs::rename(path, txt);
                }
            }
        }
        if saved > 0 {
            self.event(FileEvent::Received { from: call.peer.name.clone(), count: saved, dir });
        }
        Ok(json!({ "saved": saved }))
    }

    /// Send `paths` (each with its .txt) to `peer`: offer, then — once
    /// accepted — the files. Blocks; run it on its own thread.
    pub fn send(&self, peer: &Arc<Peer>, paths: &[PathBuf], temp: &Path) {
        self.sending.fetch_add(1, Ordering::SeqCst);
        (self.wake)();
        let result = send_files(peer, paths, temp);
        self.sending.fetch_sub(1, Ordering::SeqCst);
        match result {
            Ok(count) => self.event(FileEvent::Sent { to: peer.name.clone(), count }),
            Err(why) => self.event(FileEvent::NotSent { to: peer.name.clone(), why }),
        }
    }
}

/// A file list another PC sent: sensible length, plain names (no folders —
/// nothing can land outside the received folder), images / videos and their
/// .txt files only.
fn check_list(files: &[BlobMeta]) -> Result<(), String> {
    if files.is_empty() || files.len() > MAX_FILES {
        return Err("Too many files".into());
    }
    for f in files {
        if !is_plain_name(&f.name) {
            return Err(format!("Unusable file name: {}", f.name));
        }
        let txt = Path::new(&f.name).extension().is_some_and(|e| e.eq_ignore_ascii_case("txt"));
        if !txt && !crate::has_sidecar_extension(Path::new(&f.name)) {
            return Err(format!("Not an image or video: {}", f.name));
        }
        if f.size > MAX_FILE {
            return Err(format!("{} is too large", f.name));
        }
    }
    Ok(())
}

/// A bare file name that's safe and valid on every OS: no folders (`/`, `\`),
/// no drive or stream marks (`:`), none of Windows' other reserved
/// characters, not hidden, no trailing dot or space.
fn is_plain_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 240
        && !name.starts_with('.')
        && !name.ends_with('.')
        && !name.ends_with(' ')
        && !name.chars().any(|c| c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
}

/// The files of a send, named so no two in it clash: each image / video,
/// then its .txt under the same (maybe renamed) stem.
fn send_list(paths: &[PathBuf]) -> Vec<(String, BlobSrc)> {
    let mut out: Vec<(String, BlobSrc)> = Vec::new();
    let mut used: Vec<String> = Vec::new();
    for p in paths {
        let Some(name) = p.file_name().map(|n| n.to_string_lossy().into_owned()) else { continue };
        let stem = Path::new(&name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let ext = Path::new(&name).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        let mut n = 1;
        let wire_stem = loop {
            let s = if n == 1 { stem.clone() } else { format!("{stem}_{n}") };
            if !used.iter().any(|u| u.eq_ignore_ascii_case(&s)) {
                break s;
            }
            n += 1;
        };
        used.push(wire_stem.clone());
        let wire_name = if ext.is_empty() { wire_stem.clone() } else { format!("{wire_stem}.{ext}") };
        // Images inside a mounted zip are read from it.
        let src = if crate::archive::is_entry(p) {
            match crate::archive::read(p) {
                Ok(data) => BlobSrc::Bytes { name: wire_name.clone(), data },
                Err(_) => continue,
            }
        } else {
            BlobSrc::File { name: wire_name.clone(), path: p.clone() }
        };
        out.push((wire_name, src));
        let txt = crate::right_details::sidecar_txt(p);
        let txt_name = format!("{wire_stem}.txt");
        if crate::archive::is_entry(&txt) {
            if let Ok(data) = crate::archive::read(&txt) {
                out.push((txt_name.clone(), BlobSrc::Bytes { name: txt_name, data }));
            }
        } else if txt.is_file() {
            out.push((txt_name.clone(), BlobSrc::File { name: txt_name, path: txt }));
        }
    }
    out
}

fn send_files(peer: &Arc<Peer>, paths: &[PathBuf], temp: &Path) -> Result<usize, String> {
    let list = send_list(paths);
    if list.is_empty() {
        return Err("Nothing to send".into());
    }
    let metas: Vec<Value> = list
        .iter()
        .map(|(name, src)| {
            let size = match src {
                BlobSrc::File { path, .. } => std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
                BlobSrc::Bytes { data, .. } => data.len() as u64,
            };
            json!({ "name": name, "size": size })
        })
        .collect();
    let media = list.iter().filter(|(n, _)| !n.to_lowercase().ends_with(".txt")).count();
    let mut offer = peer.call("files.offer", json!({ "files": metas }), Vec::new(), temp)?;
    let answer = wait_done(&mut offer, Duration::from_secs(200))?;
    if answer.get("accept").and_then(Value::as_bool) != Some(true) {
        let why = answer.get("reason").and_then(Value::as_str).unwrap_or("they declined");
        return Err(format!("Not sent — {why}"));
    }
    let token = answer.get("token").and_then(Value::as_str).unwrap_or_default().to_string();
    let blobs = list.into_iter().map(|(_, src)| src).collect();
    let mut send = peer.call("files.send", json!({ "token": token }), blobs, temp)?;
    let done = wait_done(&mut send, Duration::from_secs(120))?;
    let saved = done.get("saved").and_then(Value::as_u64).unwrap_or(0) as usize;
    if saved == 0 && media > 0 {
        return Err("The other PC couldn't save them".into());
    }
    Ok(saved)
}

/// Wait for a call's `Done` (progress events are skipped).
pub(crate) fn wait_done(call: &mut super::peer::Call, timeout: Duration) -> Result<Value, String> {
    let started = Instant::now();
    loop {
        match call.next(Duration::from_millis(500)) {
            Some(CallEvent::Done(v)) => return Ok(v),
            Some(CallEvent::Fail(e)) => return Err(e),
            Some(CallEvent::Closed) => return Err("The connection was lost".into()),
            Some(CallEvent::Blob { path, .. }) => {
                let _ = std::fs::remove_file(path);
            }
            Some(CallEvent::Event(_)) => {}
            None if started.elapsed() > timeout => {
                call.cancel();
                return Err("The other PC didn't answer".into());
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_lists_are_checked() {
        let f = |name: &str, size: u64| BlobMeta { name: name.into(), size };
        assert!(check_list(&[f("cat.png", 10), f("cat.txt", 3)]).is_ok());
        assert!(check_list(&[f("../cat.png", 10)]).is_err());
        assert!(check_list(&[f("a/cat.png", 10)]).is_err());
        assert!(check_list(&[f(".hidden.png", 10)]).is_err());
        assert!(check_list(&[f("run.exe", 10)]).is_err());
        assert!(check_list(&[f("cat:x.png", 10)]).is_err());
        assert!(check_list(&[f("a\\cat.png", 10)]).is_err());
        assert!(check_list(&[]).is_err());
    }

    #[test]
    fn send_names_never_clash() {
        let dir = std::env::temp_dir().join(format!("ctf_link_send_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::create_dir_all(dir.join("b")).unwrap();
        for p in ["a/cat.png", "a/cat.txt", "b/cat.png"] {
            std::fs::write(dir.join(p), b"x").unwrap();
        }
        let names: Vec<String> = send_list(&[dir.join("a/cat.png"), dir.join("b/cat.png")])
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["cat.png", "cat.txt", "cat_2.png"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
