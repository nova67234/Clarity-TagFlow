//! Off-thread image decoding with a bounded LRU texture cache.
//!
//! This is the Rust port of terminus2's `ThumbnailService`, generalized to back
//! both the left-browser thumbnails (small, many) and the centre viewer (large,
//! few) via different `max_edge` / `capacity` settings. Like the Java version it:
//!   * decodes and downscales images on a small **worker pool** so the UI thread
//!     never blocks on a decode,
//!   * caches the resulting textures (capped to `max_edge` on the long side),
//!   * de-duplicates in-flight requests (a path is only ever queued once), and
//!   * **LRU-evicts** the least-recently-seen entries to bound GPU memory.
//!
//! A cache created with `animate = true` (the centre viewer) decodes every frame
//! of an animated GIF and plays it back; an `animate = false` cache (the browser
//! thumbnails) just takes the first frame, like the Java thumbnail service.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

use eframe::egui;

mod anim;
use anim::{current_frame, stream_gif, AnimFrame, ANIM_FRAMES_PER_FRAME, MIN_FRAME_DELAY};

/// Pixel count (≈ 4K: 3840×2160 ≈ 8.3 MP) at/above which a decode is "large".
const LARGE_PIXELS: u64 = 8_000_000;

/// Raw thumbnails up to this size come from the camera's embedded preview
/// (see `decode_still`); larger decodes — the viewer's — develop the raw.
#[cfg(feature = "avif")]
const RAW_PREVIEW_THUMB_EDGE: u32 = 1024;

/// Raw files read for their embedded preview at once, across all caches: each
/// read loads the whole raw (tens of MB) into memory, and a cache has up to
/// 64 workers.
#[cfg(feature = "avif")]
static RAW_PREVIEW_READS: Semaphore = Semaphore::new(4);

/// AVIF / HEIC images from this size on take a large-decode permit (they have
/// no reduced-size decode, so a thumbnail decodes every pixel). A 12 MP phone
/// photo decodes alongside others; a 48 MP one waits its turn.
#[cfg(feature = "avif")]
const HEAVY_EXTENDED_PIXELS: u64 = 16_000_000;

/// An AVIF / HEIC whose size the header doesn't give takes a large-decode
/// permit from this file size on (see `decode_still`).
#[cfg(feature = "avif")]
const HEAVY_EXTENDED_BYTES: u64 = 8 * 1024 * 1024;

/// A HEIF / AVIF file's pixel size from its container header: the largest
/// `ispe` (image spatial extents) property — the full picture, as a grid
/// image's tiles and its thumbnail are smaller. Read from the first 256 KiB,
/// where the `meta` box sits in files from phones and most encoders; `None`
/// when none is there.
#[cfg(feature = "avif")]
fn heif_extent(path: &Path) -> Option<(u32, u32)> {
    use std::io::Read as _;
    let mut head = Vec::new();
    crate::archive::open(path).ok()?.take(256 * 1024).read_to_end(&mut head).ok()?;
    let be = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    // A box: size, `ispe`, version + flags, width, height.
    head.windows(4)
        .enumerate()
        .filter(|&(at, w)| w == b"ispe" && at >= 4)
        .filter_map(|(at, _)| head.get(at + 8..at + 16).map(|b| (be(&b[0..4]), be(&b[4..8]))))
        .filter(|&(w, h)| w > 0 && h > 0)
        .max_by_key(|&(w, h)| w as u64 * h as u64)
}

/// Texture bytes a cache may keep in animations that weren't drawn last frame.
/// Every GIF frame is its own texture (up to 2048² px, 16 MB), so a few long
/// GIFs in the history could otherwise hold GBs of GPU memory.
const IDLE_ANIM_BYTES: usize = 512 << 20;

/// A file's (modification time, length) — what tells a file re-created at the
/// same path apart from the one a cached result came from. `None` when it
/// can't be read (a zip entry, which can't change; a missing file).
pub(crate) type FileStamp = Option<(SystemTime, u64)>;

pub(crate) fn file_stamp(path: &Path) -> FileStamp {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// Full-resolution sizes learned while decoding images whose header doesn't
/// give them — AVIF / HEIC (which `image` can't read) and camera raws (whose
/// header describes their small embedded thumbnail) — so the zoom viewer
/// knows what 100% is and when to load full resolution. See [`decoded_size`].
static DECODED_SIZES: LazyLock<Mutex<HashMap<PathBuf, (u32, u32)>>> = LazyLock::new(Default::default);

fn remember_decoded_size(path: &Path, w: u32, h: u32) {
    let mut sizes = DECODED_SIZES.lock().unwrap();
    if sizes.len() >= 4096 {
        sizes.clear(); // tiny entries, but never let them pile up forever
    }
    sizes.insert(path.to_path_buf(), (w, h));
}

/// The real (displayed) pixel size a decode found for `path`, for formats whose
/// header can't be trusted; `None` for other formats or before any decode.
pub fn decoded_size(path: &Path) -> Option<(u32, u32)> {
    DECODED_SIZES.lock().unwrap().get(path).copied()
}

/// A tiny counting semaphore (std-only) used to bound concurrent heavy decodes.
/// Each [`ImageCache`] owns its own, so the centre viewer and the browser
/// thumbnails never wait on each other's big decodes.
struct Semaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

impl Semaphore {
    const fn new(permits: usize) -> Self {
        Self { permits: Mutex::new(permits), available: Condvar::new() }
    }

    /// Block until a permit is free, then take it. The permit is released when
    /// the returned guard is dropped.
    fn acquire(&self) -> SemaphoreGuard<'_> {
        let mut permits = self.permits.lock().unwrap();
        while *permits == 0 {
            permits = self.available.wait(permits).unwrap();
        }
        *permits -= 1;
        SemaphoreGuard { sem: self }
    }
}

struct SemaphoreGuard<'a> {
    sem: &'a Semaphore,
}

impl Drop for SemaphoreGuard<'_> {
    fn drop(&mut self) {
        *self.sem.permits.lock().unwrap() += 1;
        self.sem.available.notify_one();
    }
}

/// `image::ImageReader::open`, but reading through [`crate::archive`] so
/// entries of a mounted zip decode from memory like plain files.
fn image_reader(
    path: &Path,
) -> std::io::Result<image::ImageReader<std::io::BufReader<crate::archive::Reader>>> {
    Ok(image::ImageReader::new(std::io::BufReader::new(crate::archive::open(path)?)))
}

/// `image::image_dimensions`, routed through [`crate::archive`] (header-only
/// read either way).
fn probe_dimensions(path: &Path) -> Option<(u32, u32)> {
    image_reader(path).ok()?.with_guessed_format().ok()?.into_dimensions().ok()
}

/// Take a permit from this cache's `gate` if `path` is a large (≥ [`LARGE_PIXELS`])
/// image; otherwise `None` (no throttling). `probe_dimensions` only reads the
/// header, so the size check is cheap. Hold the guard for the whole decode.
fn large_decode_permit<'a>(path: &Path, gate: &'a Semaphore) -> Option<SemaphoreGuard<'a>> {
    let pixels = probe_dimensions(path)
        .map(|(w, h)| w as u64 * h as u64)
        .unwrap_or(0);
    (pixels >= LARGE_PIXELS).then(|| gate.acquire())
}

/// State of a single cached image.
enum Slot {
    Loading,
    Ready(egui::TextureHandle),
    Animated {
        frames: Vec<AnimFrame>,
        total: Duration,
        /// egui time (seconds) when the first frame was uploaded — the playback
        /// clock's zero point, so position is measured from when the GIF started,
        /// not from absolute time.
        start: f64,
        /// `false` while more frames are still streaming in from the decoder.
        /// Playback starts on the first frame and is visible immediately; until
        /// `done`, it plays *forward only* and holds on the latest decoded frame
        /// rather than looping over a partial animation. Once `done`, it loops.
        done: bool,
        /// Texture bytes held by `frames` (for the [`IDLE_ANIM_BYTES`] budget).
        bytes: usize,
    },
    Failed,
}

struct Entry {
    slot: Slot,
    /// Frame index this image was last requested for (drives LRU eviction).
    last_used: u64,
    /// Id of the job decoding this entry. Results carrying another id belong to
    /// an earlier request for the same path — e.g. a GIF still streaming when
    /// its entry was evicted and the path re-opened — and are dropped (they
    /// used to start the new animation mid-way, with frames repeated).
    job: u64,
    /// Shared with that job and set when the entry is dropped (see `Drop`), so
    /// work nobody will see stops: a queued job is skipped, a GIF stream ends.
    cancel: Arc<AtomicBool>,
    /// The file's stamp when the decode began (`None` until a result lands).
    /// Re-checked when the entry comes back into use — see [`ImageCache::request`].
    stamp: FileStamp,
}

impl Drop for Entry {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// What [`ImageCache::request`] reports for a path this frame.
pub enum Cached {
    /// A still image — draw it as-is.
    Ready(egui::TextureHandle),
    /// The current frame of an animation — draw it, and keep repainting to play.
    Animated(egui::TextureHandle),
    Loading,
    Failed,
}

/// What a worker produces for a path. A still arrives in one message; a GIF is
/// streamed as a series of [`DecodedImage::AnimChunk`] messages (so the first
/// frames can be shown before the whole animation has decoded), the last of
/// which has `done = true`.
enum DecodedImage {
    Still(egui::ColorImage),
    AnimChunk {
        frames: Vec<(egui::ColorImage, Duration)>,
        done: bool,
    },
}

/// One message from a decode worker.
struct Decoded {
    path: PathBuf,
    /// The job it answers (see [`Entry::job`]).
    job: u64,
    /// The file's stamp when the decode began (see [`Entry::stamp`]).
    stamp: FileStamp,
    image: Option<DecodedImage>,
}

/// One queued decode: the path, plus what ties its results to the entry that
/// asked for it (see [`Entry::job`] / [`Entry::cancel`]).
struct Job {
    path: PathBuf,
    id: u64,
    cancel: Arc<AtomicBool>,
}

/// Shared FIFO of decode jobs; the condvar parks idle workers until work arrives.
type JobQueue = Arc<(Mutex<VecDeque<Job>>, Condvar)>;

pub struct ImageCache {
    entries: HashMap<PathBuf, Entry>,
    /// Aspect ratio (height / width) of each decoded image. Kept separately from
    /// `entries` so it survives texture eviction — the browser needs it to size
    /// variable-height tiles without forcing a re-decode.
    aspects: HashMap<PathBuf, f32>,
    queue: JobQueue,
    results: Receiver<Decoded>,
    frame: u64,
    capacity: usize,
    /// How many finished decodes to upload to the GPU per frame. Large viewer
    /// textures are many MB each, so uploading a whole burst in one frame
    /// stutters the UI thread — we spread them across frames instead.
    max_uploads_per_frame: usize,
    /// Decode resolution (longest side, px), shared live with the worker threads
    /// so it can be changed at runtime (e.g. toggling HD thumbnails) without
    /// rebuilding the pool.
    max_edge: Arc<AtomicU32>,
    /// Id of the most recently queued job (see [`Entry::job`]).
    next_job: u64,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl ImageCache {
    /// `max_edge` caps the longest side of decoded stills (in pixels); `capacity`
    /// is how many decoded entries stay resident before LRU eviction; `animate`
    /// enables full GIF playback (otherwise only the first frame is decoded);
    /// `decode_permits` is how many *large* (4K+) images this cache may decode at
    /// once — its own gate, so the viewer and the browser don't block each other.
    pub fn new(max_edge: u32, capacity: usize, animate: bool, decode_permits: usize) -> Self {
        let queue: JobQueue = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (res_tx, results) = mpsc::channel::<Decoded>();

        // This cache's own large-decode gate (see `Semaphore`).
        let gate = Arc::new(Semaphore::new(decode_permits.max(1)));

        // Decode resolution shared with the workers so it can change at runtime.
        let max_edge_shared = Arc::new(AtomicU32::new(max_edge));

        // MASSIVE PIPELINE UNLOCK:
        // We oversaturate the thread pool to mimic Java's virtual threads.
        // By spinning up double the logical cores (capped at a beefy 64),
        // we ensure that threads blocking on disk I/O don't hold up threads
        // ready to crush CPU decoding.
        let worker_count = thread::available_parallelism()
            .map(|n| n.get() * 2)
            .unwrap_or(16)
            .clamp(8, 64);

        let workers = (0..worker_count)
            .map(|_| {
                let queue = Arc::clone(&queue);
                let tx = res_tx.clone();
                let gate = Arc::clone(&gate);
                let max_edge = Arc::clone(&max_edge_shared);
                thread::spawn(move || worker_loop(queue, tx, max_edge, animate, gate))
            })
            .collect();

        // The workers hold their own sender clones; drop ours so the count is exact.
        drop(res_tx);

        Self {
            entries: HashMap::new(),
            aspects: HashMap::new(),
            queue,
            results,
            frame: 0,
            capacity: capacity.max(1),
            // Big viewer textures (max_edge ≥ 1024) trickle in one per frame;
            // small browser thumbnails can upload several without a hitch.
            max_uploads_per_frame: if max_edge >= 1024 { 1 } else { 6 },
            max_edge: max_edge_shared,
            next_job: 0,
            _workers: workers,
        }
    }

    /// Change the decode resolution (longest side, px). If it actually changes,
    /// the cached textures are dropped so tiles re-decode at the new size on next
    /// request. Aspect ratios are kept (resolution-independent), so the browser
    /// layout doesn't reflow.
    pub fn set_max_edge(&mut self, max_edge: u32) {
        if self.max_edge.swap(max_edge, Ordering::Relaxed) != max_edge {
            self.entries.clear();
        }
    }

    /// Call once per frame before drawing: advance the frame clock, upload any
    /// finished decodes into textures, and evict stale entries.
    pub fn begin_frame(&mut self, ctx: &egui::Context) {
        self.frame = self.frame.wrapping_add(1);

        // Upload finished decodes, but bound the GPU work per frame so a burst
        // can't stutter the UI thread. Big viewer stills are limited tightly
        // (`max_uploads_per_frame`); small streamed GIF frames get their own,
        // larger budget so an animation drains quickly while still playing.
        let mut stills = 0;
        let mut anim_frames = 0;
        let mut did_work = false;
        while stills < self.max_uploads_per_frame && anim_frames < ANIM_FRAMES_PER_FRAME {
            let Ok(msg) = self.results.try_recv() else { break };
            match &msg.image {
                Some(DecodedImage::Still(_)) | None => stills += 1,
                Some(DecodedImage::AnimChunk { frames, .. }) => anim_frames += frames.len(),
            }
            self.apply_decoded(ctx, msg);
            did_work = true;
        }
        if did_work {
            // Show what we just uploaded, and come back next frame to drain the rest.
            ctx.request_repaint();
        }
        self.evict();
    }

    /// Upload a finished decode (or a streamed GIF chunk) into GPU texture(s) and
    /// record its aspect ratio. GIF chunks append to the entry's existing frames,
    /// so playback can begin before the whole animation has arrived.
    fn apply_decoded(&mut self, ctx: &egui::Context, msg: Decoded) {
        let Decoded { path, job, stamp, image } = msg;
        let path = path.as_path();
        // Only the entry's own job counts (see `Entry::job`); with the entry
        // gone (evicted while in flight) there's nothing to fill either.
        if self.entries.get(path).is_none_or(|e| e.job != job) {
            return;
        }
        match image {
            Some(DecodedImage::Still(img)) => {
                self.record_aspect(path, img.size);
                let name = format!("img:{}", path.display());
                let tex = ctx.load_texture(name, img, egui::TextureOptions::LINEAR);
                if let Some(entry) = self.entries.get_mut(path) {
                    entry.slot = Slot::Ready(tex);
                    entry.stamp = stamp;
                }
            }
            Some(DecodedImage::AnimChunk { frames, done }) => {
                if let Some((img, _)) = frames.first() {
                    self.record_aspect(path, img.size);
                }
                let Some(entry) = self.entries.get_mut(path) else { return };
                entry.stamp = stamp;
                // Append onto an in-progress animation, or start a new one (with
                // the playback clock anchored to now, when its first frame lands).
                let (frame_vec, total, bytes) = match &mut entry.slot {
                    Slot::Animated { frames, total, bytes, .. } => (frames, total, bytes),
                    _ => {
                        entry.slot = Slot::Animated {
                            frames: Vec::new(),
                            total: Duration::ZERO,
                            start: ctx.input(|i| i.time),
                            done: false,
                            bytes: 0,
                        };
                        match &mut entry.slot {
                            Slot::Animated { frames, total, bytes, .. } => (frames, total, bytes),
                            _ => unreachable!(),
                        }
                    }
                };
                for (img, delay) in frames {
                    let delay = delay.max(MIN_FRAME_DELAY);
                    *total += delay;
                    *bytes += img.size[0] * img.size[1] * 4;
                    let name = format!("img:{}#{}", path.display(), frame_vec.len());
                    let tex = ctx.load_texture(name, img, egui::TextureOptions::LINEAR);
                    frame_vec.push(AnimFrame { tex, delay });
                }
                // Mark completion / fall back to Failed if a GIF yielded no frames.
                if let Slot::Animated { frames, done: d, .. } = &mut entry.slot {
                    if frames.is_empty() {
                        entry.slot = Slot::Failed;
                    } else {
                        *d = done;
                    }
                }
            }
            None => {
                if let Some(entry) = self.entries.get_mut(path)
                    && matches!(entry.slot, Slot::Loading) {
                        entry.slot = Slot::Failed;
                        entry.stamp = stamp;
                    }
            }
        }
    }

    fn record_aspect(&mut self, path: &Path, size: [usize; 2]) {
        let [w, h] = size;
        self.aspects
            .insert(path.to_owned(), h as f32 / w.max(1) as f32);
    }

    /// Get the image for `path`, starting a background decode on first sight.
    /// `now` is the egui time (seconds) used to pick the current animation frame.
    pub fn request(&mut self, path: &Path, now: f64) -> Cached {
        let frame = self.frame;
        // An entry coming back into use after not being drawn (another image
        // was on show, its tile scrolled away) may be for a file since replaced
        // at the same path: a crop or cutout re-created under a deleted one's
        // name, an outside edit. Compare the file's stamp once then — never per
        // frame while it stays on screen — and decode afresh if it changed.
        if let Some(entry) = self.entries.get(path)
            && frame.wrapping_sub(entry.last_used) > 1
            && !matches!(entry.slot, Slot::Loading)
            && file_stamp(path) != entry.stamp
        {
            self.entries.remove(path);
        }
        if let Some(entry) = self.entries.get_mut(path) {
            entry.last_used = frame;
            return match &entry.slot {
                Slot::Ready(tex) => Cached::Ready(tex.clone()),
                Slot::Animated { frames, total, start, done, .. } => {
                    Cached::Animated(current_frame(frames, *total, *start, now, *done))
                }
                Slot::Loading => Cached::Loading,
                Slot::Failed => Cached::Failed,
            };
        }
        // First request for this path: record it and enqueue a decode job.
        self.next_job += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.entries.insert(
            path.to_owned(),
            Entry {
                slot: Slot::Loading,
                last_used: frame,
                job: self.next_job,
                cancel: Arc::clone(&cancel),
                stamp: None,
            },
        );
        let (lock, cv) = &*self.queue;
        lock.lock().unwrap().push_back(Job { path: path.to_owned(), id: self.next_job, cancel });
        cv.notify_one();
        Cached::Loading
    }

    /// Known aspect ratio (height / width) of an image, once it has decoded.
    /// Returns `None` until then so the caller can use a default while it loads.
    pub fn aspect(&self, path: &Path) -> Option<f32> {
        self.aspects.get(path).copied()
    }

    /// Drop every resolved entry that wasn't requested this frame — i.e.
    /// everything outside the current viewport (+ prefetch margin) — freeing its
    /// GPU texture immediately. In-flight (`Loading`) entries are kept so their
    /// decode isn't wasted. Aspect ratios live in a separate map, so unloading a
    /// texture never reflows the list; the tile just re-decodes when scrolled
    /// back into view. Call this *after* the frame's `request` calls.
    pub fn retain_visible(&mut self) {
        let frame = self.frame;
        self.entries
            .retain(|_, e| e.last_used == frame || matches!(e.slot, Slot::Loading));
    }

    /// Drop the least-recently-used resolved entries once over capacity.
    /// In-flight (`Loading`) entries are kept so their decode isn't wasted.
    fn evict(&mut self) {
        self.evict_idle_animations();
        if self.entries.len() <= self.capacity {
            return;
        }
        let mut evictable: Vec<(PathBuf, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| !matches!(e.slot, Slot::Loading))
            .map(|(p, e)| (p.clone(), e.last_used))
            .collect();
        evictable.sort_by_key(|(_, used)| *used); // oldest first
        let to_remove = self.entries.len() - self.capacity;
        for (path, _) in evictable.into_iter().take(to_remove) {
            self.entries.remove(&path); // dropping the TextureHandle(s) frees the GPU texture(s)
        }
    }

    /// Keep animations that weren't drawn last frame within [`IDLE_ANIM_BYTES`],
    /// dropping the least recently used first. What's on screen (or prefetched,
    /// i.e. requested last frame) is never touched.
    fn evict_idle_animations(&mut self) {
        let frame = self.frame;
        let idle_bytes = |e: &Entry| match &e.slot {
            Slot::Animated { bytes, .. } if frame.wrapping_sub(e.last_used) > 1 => Some(*bytes),
            _ => None,
        };
        let mut total: usize = self.entries.values().filter_map(idle_bytes).sum();
        if total <= IDLE_ANIM_BYTES {
            return;
        }
        let mut idle: Vec<(PathBuf, u64, usize)> = self
            .entries
            .iter()
            .filter_map(|(p, e)| idle_bytes(e).map(|b| (p.clone(), e.last_used, b)))
            .collect();
        idle.sort_by_key(|(_, used, _)| *used); // oldest first
        for (path, _, bytes) in idle {
            if total <= IDLE_ANIM_BYTES {
                break;
            }
            self.entries.remove(&path);
            total -= bytes;
        }
    }
}

fn worker_loop(queue: JobQueue, results: Sender<Decoded>, max_edge: Arc<AtomicU32>, animate: bool, gate: Arc<Semaphore>) {
    let (lock, cv) = &*queue;
    loop {
        let job = {
            let mut q = lock.lock().unwrap();
            loop {
                if let Some(job) = q.pop_front() {
                    break job;
                }
                q = cv.wait(q).unwrap(); // park until a job is queued
            }
        };
        // Its entry is already gone (cleared, re-requested) — nobody would see it.
        if job.cancel.load(Ordering::Relaxed) {
            continue;
        }
        // Stamped before reading, so a change made mid-decode shows up as a
        // mismatch later instead of being missed.
        let stamp = file_stamp(&job.path);
        // Read the resolution per job so a live change (e.g. HD toggle) applies.
        let edge = max_edge.load(Ordering::Relaxed);
        let is_gif = job
            .path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gif"))
            .unwrap_or(false);

        let alive = if animate && is_gif {
            // GIF: stream frames as they decode (sends several messages itself).
            stream_gif(&job, stamp, &results, gate.as_ref())
        } else {
            let image = decode_still(&job.path, edge, &gate).map(DecodedImage::Still);
            results.send(Decoded { path: job.path, job: job.id, stamp, image }).is_ok()
        };
        if !alive {
            break; // UI side is gone; stop the worker.
        }
    }
}

/// Decode a single image downscaled to fit `max_edge` on the long side.
///
/// For images larger than `max_edge` we decode at a *reduced resolution* —
/// JPEG via DCT scaling, PNG by subsampling rows — so a 4K–8K source never
/// materializes a full ~100 MB buffer (this mirrors terminus2's
/// `ImageReader.setSourceSubsampling`). Everything else, and any failure of the
/// fast path, falls back to a full decode + downscale (memory-gated).
fn decode_still(path: &Path, max_edge: u32, gate: &Semaphore) -> Option<egui::ColorImage> {
    // SVG is vector — rasterize at the requested edge so thumbnails and the
    // big viewer are both pixel-sharp regardless of the document's size.
    if ext_eq(path, "svg") {
        let img = decode_svg(path, max_edge)?;
        return Some(color_image(&img));
    }

    // Radiance HDR (.hdr) decodes to linear, scene-referred floats whose values
    // exceed 1.0; the `image` crate's default 8-bit conversion just clamps, which
    // blows out highlights and skips gamma. Route it through our tone-mapper so it
    // displays well. Lightweight pure-Rust decoder, so always available.
    if ext_eq(path, "hdr") {
        // Always gated: there's no reduced-resolution path, and the f32
        // buffers peak at GBs for a 16k HDRI — a screenful decoding at once
        // on every worker could run out of memory.
        let _gate = gate.acquire();
        let img = decode_hdr(path)?;
        let img = image::DynamicImage::ImageRgba8(img).thumbnail(max_edge, max_edge);
        return Some(color_image(&img.to_rgba8()));
    }

    // AVIF/HEIC/HEIF go through the pure-Rust extended decoders (the `image`
    // crate can't handle them). Only compiled in with the `avif` feature;
    // downscale the result to fit like any other still.
    #[cfg(feature = "avif")]
    {
        let is_extended = path
            .extension()
            .and_then(|e| e.to_str())
            .map(crate::is_extended_extension)
            .unwrap_or(false);
        if is_extended {
            let is_raw = ["dng", "arw", "cr2", "nef"].iter().any(|e| ext_eq(path, e));
            // A raw's tile is the camera's own embedded preview when one is
            // big enough: no develop (seconds and gigabytes for a few hundred
            // pixels — a folder of raws filled in one tile at a time), and the
            // camera's colour. The viewer, at larger sizes, develops the raw.
            if is_raw && max_edge <= RAW_PREVIEW_THUMB_EDGE {
                let preview = {
                    let _reading = RAW_PREVIEW_READS.acquire();
                    crate::archive::read(path)
                        .ok()
                        .and_then(|bytes| crate::raw_preview::embedded_jpeg_for(&bytes, max_edge))
                };
                if let Some(rgba) = preview {
                    let img = image::DynamicImage::ImageRgba8(rgba).thumbnail(max_edge, max_edge);
                    return Some(color_image(&img.to_rgba8()));
                }
            }
            // These decode at full resolution, and `image` can't read their
            // headers to say how big they are (it reports a raw's small
            // embedded thumbnail). A raw develop always takes a permit; an
            // AVIF / HEIC one when its container header says it's large —
            // or, without one found there, when the file is — so a folder of
            // phone photos decodes side by side instead of one at a time.
            let heavy = is_raw
                || match heif_extent(path) {
                    Some((w, h)) => w as u64 * h as u64 >= HEAVY_EXTENDED_PIXELS,
                    None => std::fs::metadata(path).ok().is_none_or(|m| m.len() >= HEAVY_EXTENDED_BYTES),
                };
            let _gate = heavy.then(|| gate.acquire());
            let rgba = crate::avif::decode_avif(path)?;
            remember_decoded_size(path, rgba.width(), rgba.height());
            let img = image::DynamicImage::ImageRgba8(rgba).thumbnail(max_edge, max_edge);
            return Some(color_image(&img.to_rgba8()));
        }
    }

    // EXIF orientation isn't applied by `image` (nor read at all by the fast
    // `jpeg_decoder`/`png` subsample paths), so phone/drone/camera photos shot in
    // portrait (or any rotated orientation) would display sideways. Read the tag
    // once and apply it to whatever the decode paths below produce.
    let orientation = exif_orientation(path);

    if let Some((w, h)) = probe_dimensions(path)
        && (w > max_edge || h > max_edge)
            && let Some(mut img) = decode_downscaled(path, w, h, max_edge) {
                img.apply_orientation(orientation);
                // Cheap final fit: the result is already near `max_edge`.
                let img = img.thumbnail(max_edge, max_edge);
                return Some(color_image(&img.to_rgba8()));
            }

    // Fallback: full-resolution decode + downscale. Throttle it so a burst can't
    // spike RAM and freeze the UI (small images skip the gate inside the helper).
    let _gate = large_decode_permit(path, gate);

    match image_reader(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .decode()
    {
        Ok(mut img) => {
            img.apply_orientation(orientation);
            // `thumbnail` only ever downscales (small images pass through) and keeps aspect.
            let img = img.thumbnail(max_edge, max_edge);
            Some(color_image(&img.to_rgba8()))
        }
        // Some "raw" TIFFs store the rendered image JPEG-compressed inside the TIFF
        // (raw CFA data in a sub-IFD), which the `image` crate can't decode ("unknown
        // photometric interpretation"). Recover the embedded camera JPEG instead.
        Err(_) if matches!(path.extension().and_then(|e| e.to_str()), Some(e) if e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff")) =>
        {
            let bytes = crate::archive::read(path).ok()?;
            let rgba = crate::raw_preview::largest_embedded_jpeg(&bytes)?;
            remember_decoded_size(path, rgba.width(), rgba.height());
            let img = image::DynamicImage::ImageRgba8(rgba).thumbnail(max_edge, max_edge);
            Some(color_image(&img.to_rgba8()))
        }
        Err(_) => None,
    }
}

/// Read a file's EXIF orientation tag (defaulting to no-op). `image`'s decoders
/// expose it but never apply it; the fast subsample paths skip EXIF entirely, so
/// we read it here once and apply it to every decoded still. Reads only the
/// header/metadata, not the pixels.
pub(crate) fn exif_orientation(path: &Path) -> image::metadata::Orientation {
    use image::ImageDecoder;
    image_reader(path)
        .ok()
        .and_then(|r| r.with_guessed_format().ok())
        .and_then(|r| r.into_decoder().ok())
        .and_then(|mut d| d.orientation().ok())
        .unwrap_or(image::metadata::Orientation::NoTransforms)
}

/// `image::open` with the EXIF orientation applied — the picture the way the
/// viewer shows it. Phones often store portraits sideways with a "rotate
/// this" flag; anything that decodes stills outside the viewer (the AI
/// chat's thumbnails, its vision input and Detailed Vision sections) uses
/// this so it never sees them sideways. Like the viewer, it reads images
/// inside a mounted zip, and SVG, HDR and (with the `avif` feature)
/// AVIF/HEIC through the viewer's own decoders — so the AI can see anything
/// the browser shows.
pub(crate) fn open_oriented(path: &Path) -> image::ImageResult<image::DynamicImage> {
    let special = if ext_eq(path, "svg") {
        decode_svg(path, 2048)
    } else if ext_eq(path, "hdr") {
        decode_hdr(path)
    } else {
        None
    };
    if let Some(rgba) = special {
        return Ok(image::DynamicImage::ImageRgba8(rgba));
    }
    #[cfg(feature = "avif")]
    {
        if path.extension().and_then(|e| e.to_str()).is_some_and(crate::is_extended_extension)
            && let Some(rgba) = crate::avif::decode_avif(path)
        {
            return Ok(image::DynamicImage::ImageRgba8(rgba));
        }
    }
    let mut reader = image_reader(path).map_err(image::ImageError::IoError)?;
    // The content decides, like the viewer (`decode_still` sniffs too): a JPEG
    // or WebP saved as `.png` displays fine, so Crop / Copy must read it as
    // well. The extension is only the fallback when the bytes aren't
    // recognised (`with_guessed_format` keeps the format set here then).
    if let Ok(format) = image::ImageFormat::from_path(path) {
        reader.set_format(format);
    }
    let reader = reader.with_guessed_format().map_err(image::ImageError::IoError)?;
    let mut img = reader.decode()?;
    img.apply_orientation(exif_orientation(path));
    Ok(img)
}

/// Case-insensitive extension match (`ext` must be given lowercase).
fn ext_eq(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Rasterize an SVG to straight-alpha RGBA. Vector sources have no native
/// pixel size, so the long side renders at `max_edge` (up- OR downscaled —
/// always sharp); `max_edge == 0` renders at the SVG's own declared size
/// (used by the details panel so the reported dimensions are the document's).
/// resvg is already in the tree via egui_extras' svg feature (the UI icons).
pub fn decode_svg(path: &Path, max_edge: u32) -> Option<image::RgbaImage> {
    use std::sync::OnceLock;
    // System fonts for SVGs containing <text>, loaded once per run.
    static FONTDB: OnceLock<std::sync::Arc<resvg::usvg::fontdb::Database>> = OnceLock::new();
    let fontdb = FONTDB
        .get_or_init(|| {
            let mut db = resvg::usvg::fontdb::Database::new();
            db.load_system_fonts();
            std::sync::Arc::new(db)
        })
        .clone();
    // archive::read so SVGs inside a zip being viewed decode too (same as HDR).
    let data = crate::archive::read(path).ok()?;
    let opt = resvg::usvg::Options { fontdb, ..Default::default() };
    let tree = resvg::usvg::Tree::from_data(&data, &opt).ok()?;
    let size = tree.size();
    let (w0, h0) = (size.width(), size.height());
    if !(w0.is_finite() && h0.is_finite() && w0 > 0.0 && h0 > 0.0) {
        return None;
    }
    let long = w0.max(h0);
    let scale = if max_edge == 0 { 1.0f32 } else { max_edge as f32 / long };
    // Cap the raster so a huge declared viewBox can't allocate absurd buffers.
    let scale = scale.min(8192.0 / long);
    let w = ((w0 * scale).round() as u32).max(1);
    let h = ((h0 * scale).round() as u32).max(1);
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w, h)?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    // tiny-skia pixels are premultiplied; the viewer wants straight RGBA.
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for px in pixmap.pixels() {
        let c = px.demultiply();
        out.extend_from_slice(&[c.red(), c.green(), c.blue(), c.alpha()]);
    }
    image::RgbaImage::from_raw(w, h, out)
}

/// Decode a Radiance HDR (`.hdr`, RGBE) file to a display-ready 8-bit sRGB image.
///
/// HDR files store *linear, scene-referred* radiance with values well above 1.0,
/// so they can't be shown directly — we tone-map the high dynamic range down to
/// the [0,1] display range (global Reinhard, `c / (1 + c)`) and then sRGB
/// gamma-encode, the same two steps a viewer/photo app applies. A plain 8-bit
/// cast (what `image`'s `to_rgba8()` does) would instead clip every highlight and
/// skip gamma, giving a flat, blown-out result.
pub fn decode_hdr(path: &Path) -> Option<image::RgbaImage> {
    // `image`'s HDR decoder only accepts the `#?RADIANCE` signature, but the
    // format equally permits `#?RGBE` (and other program-name tokens). Read the
    // bytes, normalise the signature line, and decode from memory so those files
    // aren't rejected as "signature invalid".
    let bytes = normalize_hdr_signature(crate::archive::read(path).ok()?);
    let decoder = image::codecs::hdr::HdrDecoder::new(std::io::Cursor::new(&bytes)).ok()?;
    let src = image::DynamicImage::from_decoder(decoder).ok()?.to_rgb32f();
    let (w, h) = src.dimensions();
    let mut out = image::RgbaImage::new(w, h);
    for (dst, px) in out.pixels_mut().zip(src.pixels()) {
        let [r, g, b] = px.0;
        *dst = image::Rgba([tonemap_channel(r), tonemap_channel(g), tonemap_channel(b), 255]);
    }
    Some(out)
}

/// Rewrite a Radiance header's first line to the canonical `#?RADIANCE` signature
/// when it uses another valid program-name token (e.g. `#?RGBE`). The identifier
/// after `#?` is purely informational, so swapping it is lossless and lets the
/// stricter `image` decoder accept the file. Leaves non-`#?` data untouched.
fn normalize_hdr_signature(bytes: Vec<u8>) -> Vec<u8> {
    const RADIANCE: &[u8] = b"#?RADIANCE";
    if bytes.starts_with(b"#?") && !bytes.starts_with(RADIANCE)
        && let Some(nl) = bytes.iter().position(|&b| b == b'\n') {
            let mut out = Vec::with_capacity(RADIANCE.len() + (bytes.len() - nl));
            out.extend_from_slice(RADIANCE);
            out.extend_from_slice(&bytes[nl..]); // keep the newline + remaining header/body
            return out;
        }
    bytes
}

/// Reinhard tone-map + sRGB gamma encode for one linear HDR channel → 0..=255.
fn tonemap_channel(c: f32) -> u8 {
    let mapped = c.max(0.0) / (1.0 + c.max(0.0)); // Reinhard → [0,1)
    let srgb = if mapped <= 0.003_130_8 {
        12.92 * mapped
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Integer subsample factor that brings the long edge down to roughly `max_edge`,
/// matching terminus2's `ss`. Returns ≥ 1 (1 means "no reduction").
fn subsample_factor(src_w: u32, src_h: u32, max_edge: u32) -> u32 {
    let long = src_w.max(src_h) as f32;
    let target = max_edge.max(1) as f32;
    (long / target).round().max(1.0) as u32
}

/// Try to decode `path` at reduced resolution. Returns `None` for formats/cases
/// the fast path can't handle (the caller then does a full decode).
fn decode_downscaled(path: &Path, src_w: u32, src_h: u32, max_edge: u32) -> Option<image::DynamicImage> {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("jpg") | Some("jpeg") => decode_jpeg_scaled(path, max_edge),
        Some("png") => decode_png_subsampled(path, src_w, src_h, max_edge),
        Some("tif") | Some("tiff") => decode_tiff_subsampled(path, src_w, src_h, max_edge),
        Some("webp") => decode_webp_scaled(path, src_w, src_h, max_edge),
        _ => None,
    }
}

/// Decode a JPEG using the decoder's built-in DCT downscaling (1/1, 1/2, 1/4 or
/// 1/8), which is fast and never expands the full-resolution pixels.
fn decode_jpeg_scaled(path: &Path, max_edge: u32) -> Option<image::DynamicImage> {
    use jpeg_decoder::PixelFormat;

    let file = crate::archive::open(path).ok()?;
    let mut decoder = jpeg_decoder::Decoder::new(std::io::BufReader::new(file));
    // Ask for ~max_edge; the decoder snaps to the nearest power-of-two DCT scale.
    decoder.scale(max_edge as u16, max_edge as u16).ok()?;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let (w, h) = (info.width as u32, info.height as u32);

    match info.pixel_format {
        PixelFormat::RGB24 => image::RgbImage::from_raw(w, h, pixels).map(image::DynamicImage::ImageRgb8),
        PixelFormat::L8 => image::GrayImage::from_raw(w, h, pixels).map(image::DynamicImage::ImageLuma8),
        // L16 / CMYK32 are rare — let the full-decode fallback handle them.
        _ => None,
    }
}

/// Decode a PNG while subsampling: read it row-by-row, keeping every `ss`-th row
/// and every `ss`-th pixel, so only the small output (plus one source row) is
/// ever held. Interlaced PNGs don't arrive top-to-bottom, so they return `None`.
fn decode_png_subsampled(path: &Path, src_w: u32, src_h: u32, max_edge: u32) -> Option<image::DynamicImage> {
    let file = crate::archive::open(path).ok()?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    // Normalize rows to 8-bit and expand palette / low-bit-depth, so a row is a
    // simple array of 1–4 bytes-per-pixel that we can sample directly.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;

    if reader.info().interlaced {
        return None;
    }
    let channels = reader.output_color_type().0.samples();
    if !(1..=4).contains(&channels) {
        return None;
    }

    let ss = subsample_factor(src_w, src_h, max_edge);
    let out_w = src_w.div_ceil(ss);
    let out_h = src_h.div_ceil(ss);
    let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * 4];

    let mut src_y: u32 = 0;
    let mut dst_y: u32 = 0;
    loop {
        match reader.next_row() {
            Ok(Some(row)) => {
                if src_y.is_multiple_of(ss) && dst_y < out_h {
                    let data = row.data();
                    let row_base = (dst_y * out_w * 4) as usize;
                    for dst_x in 0..out_w {
                        let s = (dst_x * ss) as usize * channels;
                        let d = row_base + dst_x as usize * 4;
                        let (r, g, b, a) = match channels {
                            1 => (data[s], data[s], data[s], 255),
                            2 => (data[s], data[s], data[s], data[s + 1]),
                            3 => (data[s], data[s + 1], data[s + 2], 255),
                            _ => (data[s], data[s + 1], data[s + 2], data[s + 3]),
                        };
                        out[d] = r;
                        out[d + 1] = g;
                        out[d + 2] = b;
                        out[d + 3] = a;
                    }
                    dst_y += 1;
                }
                src_y += 1;
            }
            Ok(None) => break,
            Err(_) => return None,
        }
    }

    image::RgbaImage::from_raw(out_w, out_h, out).map(image::DynamicImage::ImageRgba8)
}

/// Decode a TIFF while subsampling, the same way as the PNG path: read it
/// strip-by-strip and keep every `ss`-th row/pixel. Only the common chunky,
/// 8-bit, stripped RGB/RGBA/Gray case is handled — anything else (tiled, planar,
/// 16-bit, CMYK, …) returns `None` and lets the full-decode fallback take over.
fn decode_tiff_subsampled(path: &Path, src_w: u32, src_h: u32, max_edge: u32) -> Option<image::DynamicImage> {
    let file = crate::archive::open(path).ok()?;
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).ok()?;

    let channels = match decoder.colortype().ok()? {
        tiff::ColorType::Gray(8) => 1usize,
        tiff::ColorType::RGB(8) => 3,
        tiff::ColorType::RGBA(8) => 4,
        _ => return None,
    };

    let (chunk_w, chunk_h) = decoder.chunk_dimensions();
    if chunk_w != src_w || chunk_h == 0 {
        return None; // tiled layout — not a simple top-to-bottom strip stream
    }
    let strip_count = decoder.strip_count().ok()?;
    if strip_count != src_h.div_ceil(chunk_h) {
        return None; // planar config would give more chunks (one plane each)
    }

    let ss = subsample_factor(src_w, src_h, max_edge);
    let out_w = src_w.div_ceil(ss);
    let out_h = src_h.div_ceil(ss);
    let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * 4];

    let mut src_y: u32 = 0;
    let mut dst_y: u32 = 0;
    for chunk in 0..strip_count {
        let (cw, ch) = decoder.chunk_data_dimensions(chunk);
        if cw != src_w {
            return None;
        }
        let data = match decoder.read_chunk(chunk).ok()? {
            tiff::decoder::DecodingResult::U8(v) => v,
            _ => return None,
        };
        for row in 0..ch {
            if src_y.is_multiple_of(ss) && dst_y < out_h {
                let row_off = row as usize * cw as usize * channels;
                let dst_base = (dst_y * out_w * 4) as usize;
                for dst_x in 0..out_w {
                    let s = row_off + (dst_x * ss) as usize * channels;
                    let d = dst_base + dst_x as usize * 4;
                    let (r, g, b, a) = match channels {
                        1 => (data[s], data[s], data[s], 255),
                        3 => (data[s], data[s + 1], data[s + 2], 255),
                        _ => (data[s], data[s + 1], data[s + 2], data[s + 3]),
                    };
                    out[d] = r;
                    out[d + 1] = g;
                    out[d + 2] = b;
                    out[d + 3] = a;
                }
                dst_y += 1;
            }
            src_y += 1;
        }
    }

    image::RgbaImage::from_raw(out_w, out_h, out).map(image::DynamicImage::ImageRgba8)
}

/// Decode a WebP at reduced resolution using libwebp's built-in scaler, so a
/// large WebP never expands to full resolution. Animated WebP falls through to
/// the full-decode fallback (this decoder only does still frames).
fn decode_webp_scaled(path: &Path, src_w: u32, src_h: u32, max_edge: u32) -> Option<image::DynamicImage> {
    use libwebp_sys as webp;

    let data = crate::archive::read(path).ok()?;

    // Fit the long edge to max_edge, preserving aspect (only ever downscaling).
    let scale = (max_edge as f64 / src_w.max(src_h).max(1) as f64).min(1.0);
    let out_w = ((src_w as f64 * scale).round() as i32).max(1);
    let out_h = ((src_h as f64 * scale).round() as i32).max(1);

    // SAFETY: we follow libwebp's documented decode flow — init the config,
    // probe features, decode into a library-owned buffer, copy it out, then free
    // it via WebPFreeDecBuffer on every return path.
    unsafe {
        let mut config = webp::WebPDecoderConfig::new().ok()?;
        if webp::WebPGetFeatures(data.as_ptr(), data.len(), &mut config.input)
            != webp::VP8StatusCode::VP8_STATUS_OK
        {
            return None;
        }
        if config.input.has_animation != 0 {
            return None; // let the fallback handle animated WebP
        }

        config.options.use_scaling = 1;
        config.options.scaled_width = out_w;
        config.options.scaled_height = out_h;
        config.output.colorspace = webp::WEBP_CSP_MODE::MODE_RGBA;

        if webp::WebPDecode(data.as_ptr(), data.len(), &mut config)
            != webp::VP8StatusCode::VP8_STATUS_OK
        {
            webp::WebPFreeDecBuffer(&mut config.output);
            return None;
        }

        let w = config.output.width as u32;
        let h = config.output.height as u32;
        let rgba = config.output.u.RGBA;
        let stride = rgba.stride as usize;
        let row_len = w as usize * 4;

        let result = if rgba.rgba.is_null() || w == 0 || h == 0 || stride < row_len {
            None
        } else {
            let mut buf = vec![0u8; h as usize * row_len];
            for row in 0..h as usize {
                let src = std::slice::from_raw_parts(rgba.rgba.add(row * stride), row_len);
                buf[row * row_len..(row + 1) * row_len].copy_from_slice(src);
            }
            image::RgbaImage::from_raw(w, h, buf).map(image::DynamicImage::ImageRgba8)
        };

        webp::WebPFreeDecBuffer(&mut config.output);
        result
    }
}

fn color_image(rgba: &image::RgbaImage) -> egui::ColorImage {
    let size = [rgba.width() as usize, rgba.height() as usize];
    egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())
}

/// Image dimensions as *displayed*, i.e. with EXIF orientation applied.
/// `image::image_dimensions` returns the stored (pre-rotation) size, so for a
/// portrait phone/camera photo (orientation 6/8) it reports width/height
/// transposed from what the user sees; we swap them for the 90°/270° rotations so
/// the Image Info panel matches the displayed image.
pub fn oriented_dimensions(path: &Path) -> Option<(u32, u32)> {
    use image::metadata::Orientation;
    let (w, h) = probe_dimensions(path)?;
    let rotated = matches!(
        exif_orientation(path),
        Orientation::Rotate90
            | Orientation::Rotate270
            | Orientation::Rotate90FlipH
            | Orientation::Rotate270FlipH
    );
    Some(if rotated { (h, w) } else { (w, h) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ctf_image_cache_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn entry(slot: Slot, last_used: u64, job: u64, stamp: FileStamp) -> Entry {
        Entry { slot, last_used, job, cancel: Arc::new(AtomicBool::new(false)), stamp }
    }

    fn texture(ctx: &egui::Context) -> egui::TextureHandle {
        ctx.load_texture("t", egui::ColorImage::filled([2, 2], egui::Color32::RED), egui::TextureOptions::LINEAR)
    }

    /// The HEIF size check finds the full picture among the header's
    /// extents (a grid's 512 px tiles and the thumbnail are smaller).
    #[cfg(feature = "avif")]
    #[test]
    fn heif_extent_reads_the_largest_ispe() {
        let ispe = |w: u32, h: u32| [&20u32.to_be_bytes()[..], b"ispe", &[0; 4], &w.to_be_bytes(), &h.to_be_bytes()].concat();
        let mut file = [&24u32.to_be_bytes()[..], b"ftypheic", &[0; 12]].concat();
        file.extend(ispe(320, 240));
        file.extend(ispe(512, 512));
        file.extend(ispe(4032, 3024));
        file.extend(b"mdat and the rest");
        let dir = test_dir("heif_extent");
        let p = dir.join("a.heic");
        std::fs::write(&p, &file).unwrap();
        assert_eq!(heif_extent(&p), Some((4032, 3024)));
        std::fs::write(&p, b"no header here").unwrap();
        assert_eq!(heif_extent(&p), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Crop / Copy read through `open_oriented`: a JPEG saved as `.png` (which
    // the viewer shows fine, since it sniffs) must decode by its content.
    #[test]
    fn open_oriented_sniffs_content_over_extension() {
        let dir = test_dir("sniff");
        let img = image::RgbImage::from_pixel(6, 4, image::Rgb([200, 30, 30]));
        let jpeg_as_png = dir.join("photo.png");
        img.save_with_format(&jpeg_as_png, image::ImageFormat::Jpeg).unwrap();
        let png_as_jpg = dir.join("other.jpg");
        img.save_with_format(&png_as_jpg, image::ImageFormat::Png).unwrap();
        for p in [&jpeg_as_png, &png_as_jpg] {
            let got = open_oriented(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert_eq!((got.width(), got.height()), (6, 4), "{}", p.display());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A GIF evicted mid-stream and re-opened gets a new job: the old stream's
    // remaining chunks must not start (or join) the new animation.
    #[test]
    fn stale_job_results_are_dropped() {
        let ctx = egui::Context::default();
        let mut cache = ImageCache::new(64, 8, true, 1);
        let path = PathBuf::from("/nonexistent/anim.gif");
        cache.entries.insert(path.clone(), entry(Slot::Loading, 0, 2, None));
        let chunk = |job| Decoded {
            path: path.clone(),
            job,
            stamp: None,
            image: Some(DecodedImage::AnimChunk {
                frames: vec![(egui::ColorImage::filled([4, 4], egui::Color32::RED), Duration::from_millis(40))],
                done: false,
            }),
        };
        let frames = |cache: &ImageCache| match &cache.entries[&path].slot {
            Slot::Animated { frames, bytes, .. } => Some((frames.len(), *bytes)),
            _ => None,
        };

        cache.apply_decoded(&ctx, chunk(1)); // the evicted request's stream
        assert!(frames(&cache).is_none(), "stale chunk started an animation");
        cache.apply_decoded(&ctx, chunk(2));
        cache.apply_decoded(&ctx, chunk(1));
        assert_eq!(frames(&cache), Some((1, 4 * 4 * 4)));
        cache.apply_decoded(&ctx, chunk(2));
        assert_eq!(frames(&cache), Some((2, 2 * 4 * 4 * 4)));

        // Dropping the entry cancels its job, so the stream stops.
        let cancel = Arc::clone(&cache.entries[&path].cancel);
        cache.entries.remove(&path);
        assert!(cancel.load(Ordering::Relaxed));
    }

    // A file re-created at a cached path (crop → delete → crop again) must be
    // decoded afresh once it's requested again — but an entry kept on screen
    // isn't stat'ed every frame.
    #[test]
    fn replaced_file_is_redecoded_on_return() {
        let ctx = egui::Context::default();
        let dir = test_dir("stamp");
        let path = dir.join("x_crop.png");
        std::fs::write(&path, b"first crop").unwrap();
        let mut cache = ImageCache::new(64, 8, false, 1);
        cache.frame = 10;
        cache.entries.insert(path.clone(), entry(Slot::Ready(texture(&ctx)), 9, 100, file_stamp(&path)));

        std::fs::write(&path, b"second, longer crop").unwrap();
        // Requested last frame (still on screen): served without a stat.
        assert!(matches!(cache.request(&path, 0.0), Cached::Ready(_)));
        // Not drawn for a while: the changed stamp forces a new decode job.
        cache.frame = 20;
        assert!(matches!(cache.request(&path, 0.0), Cached::Loading));
        assert_ne!(cache.entries[&path].job, 100);

        // An unchanged file keeps its texture across a gap.
        let stamp = file_stamp(&path);
        cache.entries.insert(path.clone(), entry(Slot::Ready(texture(&ctx)), 20, 7, stamp));
        cache.frame = 40;
        assert!(matches!(cache.request(&path, 0.0), Cached::Ready(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Animations not drawn last frame are trimmed to the byte budget, oldest
    // first; ones still in use are kept whatever their size.
    #[test]
    fn idle_animations_are_kept_within_budget() {
        let mut cache = ImageCache::new(64, 8, true, 1);
        let anim = |bytes| Slot::Animated { frames: Vec::new(), total: Duration::ZERO, start: 0.0, done: true, bytes };
        let half = IDLE_ANIM_BYTES / 2 + 1;
        cache.frame = 100;
        cache.entries.insert("old.gif".into(), entry(anim(half), 10, 1, None));
        cache.entries.insert("newer.gif".into(), entry(anim(half), 50, 2, None));
        cache.entries.insert("shown.gif".into(), entry(anim(IDLE_ANIM_BYTES * 2), 99, 3, None));
        cache.evict();
        let mut left: Vec<_> = cache.entries.keys().map(|p| p.display().to_string()).collect();
        left.sort();
        assert_eq!(left, ["newer.gif", "shown.gif"]);
    }

    // The sample carries EXIF Orientation=6 (rotate 90° CW). Confirm we report
    // the *displayed* (rotated) dimensions, i.e. width/height swapped. Skips
    // cleanly when the local sample isn't present.
    // cargo test --no-default-features --features avif exif_orientation_swaps_dims -- --nocapture
    #[test]
    fn exif_orientation_swaps_dims() {
        let p = std::path::Path::new("tests/bug/canon_hdr_YES.jpg");
        if !p.exists() {
            eprintln!("skipping: no tests/bug sample");
            return;
        }
        let (sw, sh) = image::image_dimensions(p).unwrap();
        let (dw, dh) = oriented_dimensions(p).unwrap();
        println!("stored {sw}x{sh} -> displayed {dw}x{dh} (orientation {:?})", exif_orientation(p));
        assert_eq!((dw, dh), (sh, sw), "portrait photo should report swapped dims");
    }

    // Rasterize one of the app's own UI icons: parses, honours the size
    // request, and produces non-empty pixels.
    // cargo test svg_smoke -- --nocapture
    #[test]
    fn svg_smoke() {
        let p = std::path::Path::new("icons/mic.svg");
        let img = decode_svg(p, 256).expect("decode failed");
        assert_eq!(img.width().max(img.height()), 256, "long side should match the request");
        assert!(img.pixels().any(|px| px.0[3] > 0), "raster came out fully transparent");
        // Natural-size render (max_edge = 0) must also produce something sane.
        let nat = decode_svg(p, 0).expect("natural-size decode failed");
        assert!(nat.width() > 0 && nat.height() > 0);
        println!("OK {}x{} (natural {}x{})", img.width(), img.height(), nat.width(), nat.height());
    }

    // Tone-map every Radiance HDR sample to a PNG so the result can be eyeballed.
    // Skips cleanly when the local sample folder is absent.
    // cargo test hdr_smoke -- --nocapture
    #[test]
    fn hdr_smoke() {
        let dir = std::path::Path::new("tests/HDR");
        if !dir.is_dir() {
            eprintln!("skipping hdr: no tests/HDR folder");
            return;
        }
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let p = entry.path();
            if !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("hdr")) {
                continue;
            }
            let img = decode_hdr(&p).unwrap_or_else(|| panic!("decode failed: {}", p.display()));
            let out = p.with_extension("hdr.decoded.png");
            img.save(&out).unwrap();
            println!("OK {} -> {}x{} -> {}", p.display(), img.width(), img.height(), out.display());
        }
    }
}