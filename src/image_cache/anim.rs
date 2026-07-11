//! GIF animation support for [`super::ImageCache`].
//!
//! (Module named `anim`, not `gif`, to avoid colliding with the external `gif`
//! crate that `image` pulls in — a bare `gif::` path would be ambiguous.)
//!
//! Split out from `image_cache` so the still-image cache and the animation code
//! each stay readable. Two halves:
//!   * [`stream_gif`] — runs on a decode worker. It decodes a GIF frame-by-frame
//!     and *streams* the frames back in chunks so a long GIF starts playing
//!     before it has fully decoded.
//!   * [`current_frame`] — runs on the UI thread. Given the uploaded frames, it
//!     picks which one to show for the current playback time.
//!
//! Everything here is used only through the parent module (`pub(super)`); it
//! borrows the parent's `Semaphore`, `color_image`, etc. via `super::`.

use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Duration;

use eframe::egui;
use image::AnimationDecoder;

use super::{color_image, large_decode_permit, Decoded, DecodedImage, Semaphore};

/// GPU OFFLOAD: Raised from 512 to 2048.
/// By increasing this, most GIFs (up to 1080p) will bypass CPU downscaling completely.
/// This allows the raw high-def frames to hit VRAM, letting the GPU handle display
/// scaling natively (fixing blurriness) and removing the CPU bottleneck.
const GIF_MAX_EDGE: u32 = 2048;
/// Animation frames shorter than this are clamped to prevent hyper-speed rendering
/// or divide-by-zero on 0-delay GIFs. Set to 16ms to allow up to ~62.5 FPS playback.
pub(super) const MIN_FRAME_DELAY: Duration = Duration::from_millis(16);
/// Max streamed GIF frames uploaded to the GPU per UI frame.
pub(super) const ANIM_FRAMES_PER_FRAME: usize = 24;
/// Frames in the first streamed chunk — kept tiny so the very first frame paints
/// almost immediately; later chunks use [`GIF_STREAM_BATCH`].
const GIF_FIRST_BATCH: usize = 1;
/// Frames per streamed chunk after the first.
const GIF_STREAM_BATCH: usize = 8;

/// One uploaded frame of an animation.
pub(super) struct AnimFrame {
    pub(super) tex: egui::TextureHandle,
    pub(super) delay: Duration,
}

/// Pick the animation frame to show at `now` (egui seconds), measuring elapsed
/// time from `start`. A fully-decoded GIF (`done`) loops; a still-streaming one
/// plays forward and holds on the latest decoded frame, so it never loops over a
/// partial animation while the rest is still arriving.
pub(super) fn current_frame(
    frames: &[AnimFrame],
    total: Duration,
    start: f64,
    now: f64,
    done: bool,
) -> egui::TextureHandle {
    // Safety check fallback to prevent out-of-bounds panics
    if frames.is_empty() {
        unreachable!("current_frame called with empty frames");
    }

    if frames.len() == 1 || total.is_zero() {
        return frames[0].tex.clone();
    }

    // Use f64 precision over integer `.as_millis()` to prevent sub-millisecond truncation
    let total_ms = total.as_secs_f64() * 1000.0;
    let elapsed_ms = ((now - start) * 1000.0).max(0.0);

    let mut t = if done {
        // Fully decoded: loop.
        elapsed_ms.rem_euclid(total_ms)
    } else {
        // Still streaming: clamp to the frames we have (hold the last one) rather
        // than wrapping, so playback waits for more instead of repeating early.
        elapsed_ms.min(total_ms).max(0.0)
    };

    for f in frames {
        let ms = f.delay.as_secs_f64() * 1000.0;
        if t < ms {
            return f.tex.clone();
        }
        t -= ms;
    }

    frames.last().unwrap().tex.clone()
}

/// Decode an animated GIF frame-by-frame (each composited to the full canvas)
/// streaming the frames to the cache in chunks as they decode rather than waiting
/// for the whole animation. The first frame is sent on its own so it paints
/// almost immediately; the rest follow in [`GIF_STREAM_BATCH`]-sized chunks.
pub(super) fn stream_gif(path: &Path, results: &Sender<Decoded>, gate: &Semaphore) -> bool {
    // Throttle large decodes so a burst of them can't spike RAM and freeze the UI.
    let _gate = large_decode_permit(path, gate);

    let fail = || results.send((path.to_owned(), None)).is_ok();

    let Ok(file) = crate::archive::open(path) else { return fail() };

    // PERFORMANCE: Use a 64KB BufReader instead of the 8KB default.
    // GIF parsing executes thousands of micro-reads, so a larger buffer massively reduces syscalls.
    let reader = std::io::BufReader::with_capacity(64 * 1024, file);
    let Ok(decoder) = image::codecs::gif::GifDecoder::new(reader) else {
        return fail();
    };

    // Pre-allocate the vector capacity bounds to prevent constant re-allocations mid-loop
    let mut batch = Vec::with_capacity(GIF_FIRST_BATCH);
    let mut target = GIF_FIRST_BATCH;
    let mut sent_any = false;

    // Cache the scaled dimensions so we don't recalculate them on every frame.
    let mut resize_dims: Option<(u32, u32)> = None;

    for frame in decoder.into_frames() {
        let Ok(frame) = frame else { break }; // stop on a bad frame; flush what we have

        // Clamp to MIN_FRAME_DELAY (16ms) to allow 60 FPS while avoiding 0-delay warp speed.
        let delay = Duration::from(frame.delay()).max(MIN_FRAME_DELAY);
        let mut buf = frame.into_buffer();

        let (new_w, new_h) = *resize_dims.get_or_insert_with(|| {
            let (w, h) = buf.dimensions();
            if w > GIF_MAX_EDGE || h > GIF_MAX_EDGE {
                let ratio = (GIF_MAX_EDGE as f32 / w as f32).min(GIF_MAX_EDGE as f32 / h as f32);
                (
                    (w as f32 * ratio).round().max(1.0) as u32,
                    (h as f32 * ratio).round().max(1.0) as u32,
                )
            } else {
                (w, h)
            }
        });

        // OFF-LOAD TO GPU:
        // Because GIF_MAX_EDGE is now 2048, 99% of GIFs will completely skip this if-statement.
        // Bypassing CPU resizing hands the raw, full-definition frame over to the GPU directly!
        if buf.width() != new_w || buf.height() != new_h {
            // Fallback safety for absolutely massive GIFs to prevent out-of-memory crashes.
            // Using Triangle (bilinear) filtering here to preserve quality.
            buf = image::imageops::resize(&buf, new_w, new_h, image::imageops::FilterType::Triangle);
        }

        batch.push((color_image(&buf), delay));

        if batch.len() >= target {
            // Take the chunk and immediately pre-allocate memory for the next incoming chunk
            let chunk = std::mem::replace(&mut batch, Vec::with_capacity(GIF_STREAM_BATCH));

            if results
                .send((path.to_owned(), Some(DecodedImage::AnimChunk { frames: chunk, done: false })))
                .is_err()
            {
                return false;
            }
            sent_any = true;
            target = GIF_STREAM_BATCH;
        }
    }

    // Nothing decoded at all → report failure.
    if !sent_any && batch.is_empty() {
        return fail();
    }

    // Final chunk (possibly empty) marks the animation complete.
    results
        .send((path.to_owned(), Some(DecodedImage::AnimChunk { frames: batch, done: true })))
        .is_ok()
}