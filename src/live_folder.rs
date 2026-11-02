//! Live file updates (Settings → General → Files): while it's on, the open
//! folder is watched, and images saved into it join the browser — and ones
//! deleted elsewhere leave it — without moving the selection, the image on
//! show or the Details panel (see `ViewerApp::apply_live_listing`).
//!
//! A worker thread looks at the folder twice a second: at its modified time
//! first (adding, removing or renaming a file changes it), and at the whole
//! listing every few seconds anyway, since some drives (FAT, some network
//! shares) don't keep that time. A new file joins only once it looks the
//! same at two looks in a row, so an image still being written — a generator
//! saving its output — isn't shown half there.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant, SystemTime};

use eframe::egui;

use crate::image_cache::{FileStamp, file_stamp};

/// How often the folder's modified time is checked.
const POLL: Duration = Duration::from_millis(500);
/// How often the folder is listed even when its modified time hasn't changed.
const FULL_EVERY: Duration = Duration::from_secs(3);

/// The watch on the open folder, if any, and its newest listing not used yet.
#[derive(Default)]
pub struct LiveFolder {
    watch: Option<Watch>,
    /// The watched folder and its newest listing, kept until the browser
    /// can take it (a generator view shows its own list meanwhile).
    latest: Option<(PathBuf, Vec<PathBuf>)>,
}

impl LiveFolder {
    /// Watch `dir`, or nothing. Called every frame: it only starts or stops
    /// the worker when that changes.
    pub fn watch(&mut self, dir: Option<&Path>, ctx: &egui::Context) {
        if self.watch.as_ref().map(|w| w.dir.as_path()) == dir {
            return;
        }
        self.watch = dir.map(|d| Watch::start(d.to_path_buf(), POLL, FULL_EVERY, ctx.clone()));
        self.latest = None;
    }

    /// The watched folder and its media files now (in name order), when
    /// they changed since the last time this returned them and `ready`.
    /// Called every frame either way, so listings never pile up.
    pub fn take(&mut self, ready: bool) -> Option<(PathBuf, Vec<PathBuf>)> {
        if let Some(w) = &self.watch {
            while let Ok(listing) = w.rx.try_recv() {
                self.latest = Some((w.dir.clone(), listing));
            }
        }
        if ready { self.latest.take() } else { None }
    }
}

/// A worker watching one folder; it stops when this is dropped.
struct Watch {
    dir: PathBuf,
    stop: Arc<AtomicBool>,
    rx: Receiver<Vec<PathBuf>>,
}

impl Watch {
    fn start(dir: PathBuf, poll: Duration, full_every: Duration, ctx: egui::Context) -> Watch {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let (d, s) = (dir.clone(), Arc::clone(&stop));
        // No thread (the system's out of them): no live updates, nothing else.
        let _ = std::thread::Builder::new()
            .name("live-folder".into())
            .spawn(move || run(&d, &s, &tx, poll, full_every, &ctx));
        Watch { dir, stop, rx }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// The worker: look at `dir` every `poll`, listing it when its modified time
/// changes (and every `full_every` regardless); send each listing that
/// differs from the last, and wake the UI to take it.
fn run(dir: &Path, stop: &AtomicBool, tx: &Sender<Vec<PathBuf>>, poll: Duration, full_every: Duration, ctx: &egui::Context) {
    let mut scan = Scan::default();
    let mut seen: Option<SystemTime> = None;
    let mut listed: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        let modified = std::fs::metadata(dir).and_then(|m| m.modified()).ok();
        let due = modified != seen || scan.waiting() || listed.is_none_or(|t| t.elapsed() >= full_every);
        if due {
            seen = modified;
            listed = Some(Instant::now());
            // Unreadable (gone, a drive or share dropped): nothing is sent —
            // never an empty list, which would empty the browser.
            if let Ok(files) = list_media(dir)
                && let Some(listing) = scan.update(files, file_stamp)
            {
                if tx.send(listing).is_err() {
                    return; // the watch is gone
                }
                ctx.request_repaint();
            }
        }
        std::thread::sleep(poll);
    }
}

/// The media files directly in `dir`, in name order (as the browser lists a
/// folder), or an error when it can't be read in full.
fn list_media(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !crate::is_media(&path) {
            continue;
        }
        // The listing already says what an entry is (no extra look at each
        // file); a link is followed.
        let is_file = match entry.file_type() {
            Ok(t) if !t.is_symlink() => t.is_file(),
            _ => path.is_file(),
        };
        if is_file {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

/// What the worker has sent, and the new files it's waiting on.
#[derive(Default)]
struct Scan {
    /// The last listing sent (`None` before the first).
    sent: Option<Vec<PathBuf>>,
    /// New files not sent yet, and how they looked (modified, length).
    waiting: HashMap<PathBuf, FileStamp>,
}

impl Scan {
    /// Whether a new file is waiting to look the same twice.
    fn waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    /// The listing to send for `files` (the folder's media files now, in
    /// name order), if it differs from the last one sent: at the first look,
    /// everything there; after that, a new file once it looks the same
    /// (`stamp`) as at the look before — it's done being written.
    fn update(&mut self, files: Vec<PathBuf>, stamp: impl Fn(&Path) -> FileStamp) -> Option<Vec<PathBuf>> {
        let listing = match &self.sent {
            None => files,
            Some(sent) => {
                let known: HashSet<&PathBuf> = sent.iter().collect();
                let mut waiting = HashMap::new();
                let mut listing = Vec::with_capacity(files.len());
                for f in files {
                    if known.contains(&f) {
                        listing.push(f);
                        continue;
                    }
                    let now = stamp(&f);
                    if now.is_some() && self.waiting.get(&f) == Some(&now) {
                        listing.push(f);
                    } else {
                        waiting.insert(f, now);
                    }
                }
                self.waiting = waiting;
                listing
            }
        };
        if self.sent.as_ref() == Some(&listing) {
            return None;
        }
        self.sent = Some(listing.clone());
        Some(listing)
    }
}

/// What a listing changes in the browser's list.
#[derive(Debug, PartialEq)]
pub struct Merged {
    /// The new list.
    pub images: Vec<PathBuf>,
    /// The files that joined it.
    pub added: Vec<PathBuf>,
    /// The files that left it.
    pub removed: Vec<PathBuf>,
}

/// `images` (the browser's list) with `listing` (`dir`'s media files now, in
/// name order) folded in, or `None` when nothing changes. New files join in
/// name order and the rest keep their places; files of `dir` no longer there
/// leave — but never `keep` (the image on show stays until you move on), and
/// files from elsewhere (dropped in) are left alone. Each change is checked
/// against the disk (`exists`) first: a listing can be a moment old — the
/// app may have just deleted a file, or written one (a crop) that's still
/// waiting to be listed.
pub fn merge(
    images: &[PathBuf],
    dir: &Path,
    listing: &[PathBuf],
    keep: Option<&Path>,
    exists: impl Fn(&Path) -> bool,
) -> Option<Merged> {
    let added: Vec<PathBuf> = {
        let listed: HashSet<&Path> = images.iter().map(PathBuf::as_path).collect();
        listing.iter().filter(|p| !listed.contains(p.as_path()) && exists(p)).cloned().collect()
    };
    let on_disk: HashSet<&Path> = listing.iter().map(PathBuf::as_path).collect();
    // A folder that's not there right now (a drive or share dropped) loses
    // nothing.
    let reachable = exists(dir);
    let gone = |p: &Path| {
        reachable && p.parent() == Some(dir) && Some(p) != keep && !on_disk.contains(p) && !exists(p)
    };
    let mut removed = Vec::new();
    let mut merged = Vec::with_capacity(images.len() + added.len());
    let mut new = added.iter().peekable();
    for p in images {
        if gone(p) {
            removed.push(p.clone());
            continue;
        }
        while let Some(a) = new.next_if(|a| *a < p) {
            merged.push(a.clone());
        }
        merged.push(p.clone());
    }
    merged.extend(new.cloned());
    if added.is_empty() && removed.is_empty() {
        return None;
    }
    Some(Merged { images: merged, added, removed })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(dir: &str, names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(|n| Path::new(dir).join(n)).collect()
    }

    /// New files join in name order; the rest keep their places, even out of
    /// order (a crop sits beneath its source); the image on show is unmoved.
    #[test]
    fn new_files_join_in_name_order() {
        let images = paths("/f", &["b.png", "b_crop.png", "a.png", "d.png"]);
        let listing = paths("/f", &["a.png", "b.png", "b_crop.png", "c.png", "d.png", "e.png"]);
        let m = merge(&images, Path::new("/f"), &listing, Some(Path::new("/f/d.png")), |_| true).unwrap();
        assert_eq!(m.images, paths("/f", &["b.png", "b_crop.png", "a.png", "c.png", "d.png", "e.png"]));
        assert_eq!(m.added, paths("/f", &["c.png", "e.png"]));
        assert!(m.removed.is_empty());
    }

    /// A file deleted elsewhere leaves, but not the one on show, nor files
    /// from other folders; an unchanged listing changes nothing.
    #[test]
    fn deleted_files_leave_but_not_the_one_on_show() {
        let mut images = paths("/f", &["a.png", "b.png", "c.png"]);
        images.push(PathBuf::from("/elsewhere/x.png"));
        let listing = paths("/f", &["a.png"]);
        let on_disk = |p: &Path| p == Path::new("/f") || p == Path::new("/f/a.png");
        let m = merge(&images, Path::new("/f"), &listing, Some(Path::new("/f/b.png")), on_disk).unwrap();
        assert_eq!(m.images, vec![PathBuf::from("/f/a.png"), PathBuf::from("/f/b.png"), PathBuf::from("/elsewhere/x.png")]);
        assert_eq!(m.removed, paths("/f", &["c.png"]));
        assert_eq!(merge(&m.images, Path::new("/f"), &listing, Some(Path::new("/f/b.png")), on_disk), None);
    }

    /// A listing a moment old: a file it has that's gone since isn't added,
    /// and one it lacks that's on disk (just written) stays. A folder that
    /// can't be reached loses nothing.
    #[test]
    fn changes_are_checked_against_the_disk() {
        let images = paths("/f", &["a.png", "new_crop.png"]);
        let listing = paths("/f", &["a.png", "just_deleted.png"]);
        assert_eq!(merge(&images, Path::new("/f"), &listing, None, |p| p != Path::new("/f/just_deleted.png")), None);
        let unreachable = |_: &Path| false;
        assert_eq!(merge(&images, Path::new("/f"), &paths("/f", &["a.png"]), None, unreachable), None);
    }

    /// At the first look everything is sent; after that a new file waits
    /// until it looks the same twice (it's done being written).
    #[test]
    fn a_new_file_waits_until_it_stops_changing() {
        use std::cell::Cell;
        let t = SystemTime::UNIX_EPOCH;
        let len = Cell::new(10u64);
        let stamp = |_: &Path| Some((t, len.get()));
        let mut scan = Scan::default();
        let first = paths("/f", &["a.png"]);
        assert_eq!(scan.update(first.clone(), stamp), Some(first.clone()));
        assert_eq!(scan.update(first.clone(), stamp), None, "nothing changed");

        let both = paths("/f", &["a.png", "b.png"]);
        assert_eq!(scan.update(both.clone(), stamp), None, "b.png is new: wait");
        assert!(scan.waiting());
        len.set(20); // still being written
        assert_eq!(scan.update(both.clone(), stamp), None);
        assert_eq!(scan.update(both.clone(), stamp), Some(both.clone()), "the same twice: in");
        assert!(!scan.waiting());

        assert_eq!(scan.update(first.clone(), stamp), Some(first), "a deleted file leaves at once");
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ctf_live_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Only media files, in name order — not folders or other files.
    #[test]
    fn listing_has_only_media_files_in_name_order() {
        let dir = scratch("list");
        for f in ["b.png", "a.jpg", "notes.txt", "c.mp4"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("sub.png")).unwrap();
        let names: Vec<String> = list_media(&dir)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["a.jpg", "b.png", "c.mp4"]);
        assert!(list_media(&dir.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The worker sends the folder as it is, then each file saved into it
    /// once it's written, then the list without a file deleted.
    #[test]
    fn the_worker_follows_the_folder() {
        let dir = scratch("watch");
        std::fs::write(dir.join("a.png"), b"x").unwrap();
        let w = Watch::start(dir.clone(), Duration::from_millis(10), Duration::from_millis(40), egui::Context::default());
        let next = || w.rx.recv_timeout(Duration::from_secs(5)).expect("a listing");
        assert_eq!(next(), vec![dir.join("a.png")]);
        std::fs::write(dir.join("b.png"), b"x").unwrap();
        assert_eq!(next(), vec![dir.join("a.png"), dir.join("b.png")]);
        std::fs::remove_file(dir.join("a.png")).unwrap();
        assert_eq!(next(), vec![dir.join("b.png")]);
        drop(w);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
