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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use eframe::egui;

mod anim;
use anim::{current_frame, stream_gif, AnimFrame, ANIM_FRAMES_PER_FRAME, MIN_FRAME_DELAY};

/// Pixel count (≈ 4K: 3840×2160 ≈ 8.3 MP) at/above which a decode is "large".
const LARGE_PIXELS: u64 = 8_000_000;

/// A tiny counting semaphore (std-only) used to bound concurrent heavy decodes.
/// Each [`ImageCache`] owns its own, so the centre viewer and the browser
/// thumbnails never wait on each other's big decodes.
struct Semaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

impl Semaphore {
    fn new(permits: usize) -> Self {
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

/// Take a permit from this cache's `gate` if `path` is a large (≥ [`LARGE_PIXELS`])
/// image; otherwise `None` (no throttling). `image_dimensions` only reads the
/// header, so the size check is cheap. Hold the guard for the whole decode.
fn large_decode_permit<'a>(path: &Path, gate: &'a Semaphore) -> Option<SemaphoreGuard<'a>> {
    let pixels = image::image_dimensions(path)
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
    },
    Failed,
}

struct Entry {
    slot: Slot,
    /// Frame index this image was last requested for (drives LRU eviction).
    last_used: u64,
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

type Decoded = (PathBuf, Option<DecodedImage>);
/// Shared FIFO of decode jobs; the condvar parks idle workers until work arrives.
type JobQueue = Arc<(Mutex<VecDeque<PathBuf>>, Condvar)>;

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
            let Ok((path, decoded)) = self.results.try_recv() else { break };
            match &decoded {
                Some(DecodedImage::Still(_)) | None => stills += 1,
                Some(DecodedImage::AnimChunk { frames, .. }) => anim_frames += frames.len(),
            }
            self.apply_decoded(ctx, &path, decoded);
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
    fn apply_decoded(&mut self, ctx: &egui::Context, path: &Path, decoded: Option<DecodedImage>) {
        match decoded {
            Some(DecodedImage::Still(img)) => {
                self.record_aspect(path, img.size);
                let name = format!("img:{}", path.display());
                let tex = ctx.load_texture(name, img, egui::TextureOptions::LINEAR);
                if let Some(entry) = self.entries.get_mut(path) {
                    entry.slot = Slot::Ready(tex);
                }
            }
            Some(DecodedImage::AnimChunk { frames, done }) => {
                if let Some((img, _)) = frames.first() {
                    self.record_aspect(path, img.size);
                }
                // Entry may have been evicted while the decode was in flight.
                let Some(entry) = self.entries.get_mut(path) else { return };
                // Append onto an in-progress animation, or start a new one (with
                // the playback clock anchored to now, when its first frame lands).
                let (frame_vec, total) = match &mut entry.slot {
                    Slot::Animated { frames, total, .. } => (frames, total),
                    _ => {
                        entry.slot = Slot::Animated {
                            frames: Vec::new(),
                            total: Duration::ZERO,
                            start: ctx.input(|i| i.time),
                            done: false,
                        };
                        match &mut entry.slot {
                            Slot::Animated { frames, total, .. } => (frames, total),
                            _ => unreachable!(),
                        }
                    }
                };
                for (img, delay) in frames {
                    let delay = delay.max(MIN_FRAME_DELAY);
                    *total += delay;
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
                if let Some(entry) = self.entries.get_mut(path) {
                    if matches!(entry.slot, Slot::Loading) {
                        entry.slot = Slot::Failed;
                    }
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
        if let Some(entry) = self.entries.get_mut(path) {
            entry.last_used = frame;
            return match &entry.slot {
                Slot::Ready(tex) => Cached::Ready(tex.clone()),
                Slot::Animated { frames, total, start, done } => {
                    Cached::Animated(current_frame(frames, *total, *start, now, *done))
                }
                Slot::Loading => Cached::Loading,
                Slot::Failed => Cached::Failed,
            };
        }
        // First request for this path: record it and enqueue a decode job.
        self.entries.insert(
            path.to_owned(),
            Entry {
                slot: Slot::Loading,
                last_used: frame,
            },
        );
        let (lock, cv) = &*self.queue;
        lock.lock().unwrap().push_back(path.to_owned());
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
        // Read the resolution per job so a live change (e.g. HD toggle) applies.
        let edge = max_edge.load(Ordering::Relaxed);
        let is_gif = job
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("gif"))
            .unwrap_or(false);

        let alive = if animate && is_gif {
            // GIF: stream frames as they decode (sends several messages itself).
            stream_gif(&job, &results, gate.as_ref())
        } else {
            let decoded = decode_still(&job, edge, &gate).map(DecodedImage::Still);
            results.send((job, decoded)).is_ok()
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
    // Radiance HDR (.hdr) decodes to linear, scene-referred floats whose values
    // exceed 1.0; the `image` crate's default 8-bit conversion just clamps, which
    // blows out highlights and skips gamma. Route it through our tone-mapper so it
    // displays well. Lightweight pure-Rust decoder, so always available (no gate).
    if ext_eq(path, "hdr") {
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
            let _gate = large_decode_permit(path, gate);
            let rgba = crate::avif::decode_avif(path)?;
            let img = image::DynamicImage::ImageRgba8(rgba).thumbnail(max_edge, max_edge);
            return Some(color_image(&img.to_rgba8()));
        }
    }

    // EXIF orientation isn't applied by `image` (nor read at all by the fast
    // `jpeg_decoder`/`png` subsample paths), so phone/drone/camera photos shot in
    // portrait (or any rotated orientation) would display sideways. Read the tag
    // once and apply it to whatever the decode paths below produce.
    let orientation = exif_orientation(path);

    if let Ok((w, h)) = image::image_dimensions(path) {
        if w > max_edge || h > max_edge {
            if let Some(mut img) = decode_downscaled(path, w, h, max_edge) {
                img.apply_orientation(orientation);
                // Cheap final fit: the result is already near `max_edge`.
                let img = img.thumbnail(max_edge, max_edge);
                return Some(color_image(&img.to_rgba8()));
            }
        }
    }

    // Fallback: full-resolution decode + downscale. Throttle it so a burst can't
    // spike RAM and freeze the UI (small images skip the gate inside the helper).
    let _gate = large_decode_permit(path, gate);

    match image::ImageReader::open(path)
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
            let bytes = std::fs::read(path).ok()?;
            let rgba = crate::raw_preview::largest_embedded_jpeg(&bytes)?;
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
fn exif_orientation(path: &Path) -> image::metadata::Orientation {
    use image::ImageDecoder;
    image::ImageReader::open(path)
        .ok()
        .and_then(|r| r.with_guessed_format().ok())
        .and_then(|r| r.into_decoder().ok())
        .and_then(|mut d| d.orientation().ok())
        .unwrap_or(image::metadata::Orientation::NoTransforms)
}

/// Case-insensitive extension match (`ext` must be given lowercase).
fn ext_eq(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
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
    let bytes = normalize_hdr_signature(std::fs::read(path).ok()?);
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
    if bytes.starts_with(b"#?") && !bytes.starts_with(RADIANCE) {
        if let Some(nl) = bytes.iter().position(|&b| b == b'\n') {
            let mut out = Vec::with_capacity(RADIANCE.len() + (bytes.len() - nl));
            out.extend_from_slice(RADIANCE);
            out.extend_from_slice(&bytes[nl..]); // keep the newline + remaining header/body
            return out;
        }
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

    let file = std::fs::File::open(path).ok()?;
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
    let file = std::fs::File::open(path).ok()?;
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
                if src_y % ss == 0 && dst_y < out_h {
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
    let file = std::fs::File::open(path).ok()?;
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
            if src_y % ss == 0 && dst_y < out_h {
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

    let data = std::fs::read(path).ok()?;

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
    let (w, h) = image::image_dimensions(path).ok()?;
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