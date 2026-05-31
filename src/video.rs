//! Embedded video playback via libVLC (the `vlc-rs` crate), gated behind the
//! `vlc` cargo feature.
//!
//! * Without `--features vlc`: `VideoPlayer::start` returns `None`, so the app
//!   falls back to launching the external VLC player. The default build needs
//!   no VLC SDK.
//! * With `--features vlc`: libVLC decodes into an off-screen RGBA buffer via
//!   its "vmem" video callbacks, which we upload to an egui texture each frame.
//!   Requires the VLC SDK at build time and libVLC + plugins at runtime — see
//!   the build notes in the README.
//!
//! NOTE: the libVLC backend is raw `unsafe` FFI against `vlc::sys`. It type-checks
//! but could not be *run* in this environment (no VLC SDK here); expect to iterate.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use eframe::egui;

/// Whether videos should loop (restart at the end). Mirrors
/// `Settings::loop_video`; the main loop pushes the current value via
/// [`set_loop`] and the player reads it when a clip starts. A global atomic
/// keeps `VideoPlayer::start`'s signature stable across the stub / real
/// backends.
static LOOP_VIDEO: AtomicBool = AtomicBool::new(false);

/// Update the desired video-loop state (called from the app each frame).
pub fn set_loop(enabled: bool) {
    LOOP_VIDEO.store(enabled, Ordering::Relaxed);
}

/// Whether videos should currently loop.
#[allow(dead_code)] // only read by the libVLC backend (feature = "vlc")
pub fn loop_enabled() -> bool {
    LOOP_VIDEO.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Stub used when the `vlc` feature is off — keeps the rest of the app building
// with no VLC dependency. `start` returns `None`, so callers fall back to the
// external VLC launcher.
// ---------------------------------------------------------------------------
#[cfg(not(feature = "vlc"))]
pub struct VideoPlayer {}

#[cfg(not(feature = "vlc"))]
impl VideoPlayer {
    pub fn start(_path: &Path, _ctx: &egui::Context) -> Option<VideoPlayer> {
        None
    }

    pub fn frame(&mut self, _ctx: &egui::Context) -> Option<egui::TextureHandle> {
        None
    }
}

/// Stub video-thumbnail cache when the `vlc` feature is off — always returns
/// `None`, so the browser shows the video icon placeholder.
#[cfg(not(feature = "vlc"))]
pub struct VideoThumbs;

#[cfg(not(feature = "vlc"))]
impl VideoThumbs {
    pub fn new() -> Self {
        VideoThumbs
    }

    pub fn request(&mut self, _path: &Path, _ctx: &egui::Context) -> Option<egui::TextureHandle> {
        None
    }

    pub fn aspect(&self, _path: &Path) -> Option<f32> {
        None
    }

    pub fn set_busy(&mut self, _path: Option<&Path>) {}
}

// ---------------------------------------------------------------------------
// Real libVLC-backed player (feature = "vlc").
// ---------------------------------------------------------------------------
#[cfg(feature = "vlc")]
pub use backend::{VideoPlayer, VideoThumbs};

#[cfg(feature = "vlc")]
mod backend {
    use super::*;
    use std::collections::HashMap;
    use std::os::raw::{c_char, c_uint, c_void};
    use std::path::PathBuf;
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::{Duration, Instant};
    use vlc::sys;

    type SharedFrame = Arc<Mutex<Option<egui::ColorImage>>>;

    /// Opaque handed to libVLC's video callbacks. libVLC drives `setup`/`lock`/
    /// `unlock`/`display` serially from its own video thread, so `buffer` needs
    /// no locking there; `latest` is the hand-off to the egui (UI) thread.
    struct VideoCtx {
        width: u32,
        height: u32,
        buffer: Vec<u8>, // RGBA, width*height*4
        latest: SharedFrame,
        ctx: egui::Context,
    }

    pub struct VideoPlayer {
        // `_instance` must outlive `player`; both are released on drop.
        _instance: vlc::Instance,
        player: vlc::MediaPlayer,
        ctx_ptr: *mut VideoCtx,
        latest: SharedFrame,
        texture: Option<egui::TextureHandle>,
    }

    impl VideoPlayer {
        pub fn start(path: &Path, ctx: &egui::Context) -> Option<VideoPlayer> {
            let instance = vlc::Instance::new()?;
            let media = vlc::Media::new_path(&instance, path)?;
            // Loop the clip when the user enabled it in Settings. A large repeat
            // count stands in for "infinite" (libVLC has no unbounded value).
            if loop_enabled() {
                unsafe {
                    let opt = std::ffi::CString::new(":input-repeat=65535").unwrap();
                    sys::libvlc_media_add_option(media.raw(), opt.as_ptr());
                }
            }
            let player = vlc::MediaPlayer::new(&instance)?;
            player.set_media(&media);

            let latest: SharedFrame = Arc::new(Mutex::new(None));
            let ctx_ptr = Box::into_raw(Box::new(VideoCtx {
                width: 0,
                height: 0,
                buffer: Vec::new(),
                latest: latest.clone(),
                ctx: ctx.clone(),
            }));

            unsafe {
                let mp = player.raw();
                sys::libvlc_video_set_callbacks(
                    mp,
                    Some(lock_cb),
                    Some(unlock_cb),
                    Some(display_cb),
                    ctx_ptr as *mut c_void,
                );
                // vlc-rs 0.3 mis-declares the format callback's return type as `()`
                // instead of `unsigned`. Coerce our correctly-typed fn to a pointer
                // and transmute it so libVLC actually receives the picture count.
                let setup_fp: unsafe extern "C" fn(
                    *mut *mut c_void,
                    *mut c_char,
                    *mut c_uint,
                    *mut c_uint,
                    *mut c_uint,
                    *mut c_uint,
                ) -> c_uint = setup_cb;
                let setup: sys::libvlc_video_format_cb = Some(std::mem::transmute(setup_fp));
                sys::libvlc_video_set_format_callbacks(mp, setup, Some(cleanup_cb));
            }

            if player.play().is_err() {
                // No callbacks have fired yet, so it's safe to free the box.
                unsafe { drop(Box::from_raw(ctx_ptr)) };
                return None;
            }

            Some(VideoPlayer { _instance: instance, player, ctx_ptr, latest, texture: None })
        }

        /// Upload the newest decoded frame (if any) to a GPU texture and return a
        /// handle to draw. Returns the previous texture when no new frame arrived.
        pub fn frame(&mut self, ctx: &egui::Context) -> Option<egui::TextureHandle> {
            let new_img = self.latest.lock().ok().and_then(|mut g| g.take());
            if let Some(img) = new_img {
                match &mut self.texture {
                    Some(tex) => tex.set(img, egui::TextureOptions::LINEAR),
                    None => {
                        self.texture =
                            Some(ctx.load_texture("vlc_frame", img, egui::TextureOptions::LINEAR));
                    }
                }
            }
            self.texture.clone()
        }
    }

    impl Drop for VideoPlayer {
        fn drop(&mut self) {
            // Stop playback first (synchronous in VLC 3.x) so the video callbacks
            // can't fire again, then free the opaque they were writing into.
            self.player.stop();
            unsafe { drop(Box::from_raw(self.ctx_ptr)) };
        }
    }

    // --- libVLC vmem callbacks (called from libVLC's video thread) ---

    /// Negotiate the output format: keep VLC's proposed (native) size, request
    /// RGBA, and size our frame buffer to match. Returns the picture count.
    unsafe extern "C" fn setup_cb(
        opaque: *mut *mut c_void,
        chroma: *mut c_char,
        width: *mut c_uint,
        height: *mut c_uint,
        pitches: *mut c_uint,
        lines: *mut c_uint,
    ) -> c_uint {
        unsafe {
            let vc = &mut *(*opaque as *mut VideoCtx);
            let (w, h) = (*width, *height);
            if w == 0 || h == 0 || w > 8192 || h > 8192 {
                return 0; // reject implausible sizes
            }

            // RGBA byte order matches egui's ColorImage. If colours come out
            // wrong, switch this to b"BGRA" (the upload stays the same).
            std::ptr::copy_nonoverlapping(b"RGBA".as_ptr(), chroma as *mut u8, 4);
            *pitches = w * 4;
            *lines = h;

            vc.width = w;
            vc.height = h;
            vc.buffer = vec![0u8; (w * 4 * h) as usize];
            1 // one picture buffer
        }
    }

    unsafe extern "C" fn cleanup_cb(_opaque: *mut c_void) {
        // The buffer is owned by VideoCtx and freed when the player drops.
    }

    /// Give libVLC the buffer to write the next frame into.
    unsafe extern "C" fn lock_cb(opaque: *mut c_void, planes: *mut c_void) -> *mut c_void {
        unsafe {
            let vc = &mut *(opaque as *mut VideoCtx);
            *(planes as *mut *mut c_void) = vc.buffer.as_mut_ptr() as *mut c_void;
            std::ptr::null_mut()
        }
    }

    unsafe extern "C" fn unlock_cb(
        _opaque: *mut c_void,
        _picture: *mut c_void,
        _planes: *const *mut c_void,
    ) {
        // Nothing — the frame is copied out in display_cb.
    }

    /// Frame complete: copy it into the shared slot and wake the UI.
    unsafe extern "C" fn display_cb(opaque: *mut c_void, _picture: *mut c_void) {
        unsafe {
            let vc = &mut *(opaque as *mut VideoCtx);
            let expected = (vc.width * vc.height * 4) as usize;
            if expected == 0 || vc.buffer.len() != expected {
                return;
            }
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [vc.width as usize, vc.height as usize],
                &vc.buffer,
            );
            if let Ok(mut g) = vc.latest.lock() {
                *g = Some(img);
            }
            vc.ctx.request_repaint();
        }
    }

    // -----------------------------------------------------------------------
    // Video poster-frame thumbnails for the left browser.
    // -----------------------------------------------------------------------

    enum Thumb {
        Loading,
        Ready(egui::TextureHandle),
        Failed,
    }

    /// Lazily produces a poster frame for each video by briefly decoding it with
    /// libVLC on a background thread, then caching the resulting texture. Captures
    /// are bounded so a folder full of videos doesn't spin up many decoders.
    pub struct VideoThumbs {
        entries: HashMap<PathBuf, Thumb>,
        tx: mpsc::Sender<(PathBuf, Option<egui::ColorImage>)>,
        rx: mpsc::Receiver<(PathBuf, Option<egui::ColorImage>)>,
        in_flight: usize,
        /// Path of the video currently playing in the viewer, if any. We skip
        /// starting a *new* capture for just that file so the player's decoder
        /// isn't doing the same work twice. Every other video still loads its
        /// poster normally, and a cached poster for this file is still served.
        busy: Option<PathBuf>,
    }

    impl VideoThumbs {
        pub fn new() -> Self {
            let (tx, rx) = mpsc::channel();
            VideoThumbs { entries: HashMap::new(), tx, rx, in_flight: 0, busy: None }
        }

        /// Tell the cache which video is playing (so it won't double-decode it),
        /// or `None` when nothing is playing.
        pub fn set_busy(&mut self, path: Option<&Path>) {
            self.busy = path.map(|p| p.to_path_buf());
        }

        /// Cached poster texture for `path`, starting a capture on first sight.
        /// Returns `None` while loading or on failure.
        pub fn request(&mut self, path: &Path, ctx: &egui::Context) -> Option<egui::TextureHandle> {
            // Absorb any finished captures into textures.
            while let Ok((p, img)) = self.rx.try_recv() {
                self.in_flight = self.in_flight.saturating_sub(1);
                let state = match img {
                    Some(ci) => Thumb::Ready(ctx.load_texture(
                        format!("vthumb:{}", p.display()),
                        ci,
                        egui::TextureOptions::LINEAR,
                    )),
                    None => Thumb::Failed,
                };
                self.entries.insert(p, state);
            }

            match self.entries.get(path) {
                Some(Thumb::Ready(tex)) => return Some(tex.clone()),
                Some(_) => return None, // loading or failed
                None => {}
            }

            // Don't start a second decode of the file that's already playing —
            // that's the one that competes with the player. Its poster fills in
            // once playback stops.
            if self.busy.as_deref() == Some(path) {
                return None;
            }

            // Not seen yet — start a capture if we're under the concurrency cap.
            const MAX_IN_FLIGHT: usize = 2;
            if self.in_flight < MAX_IN_FLIGHT {
                self.entries.insert(path.to_path_buf(), Thumb::Loading);
                self.in_flight += 1;
                let tx = self.tx.clone();
                let p = path.to_path_buf();
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    let img = capture_poster(&p, 320);
                    let _ = tx.send((p, img));
                    ctx.request_repaint();
                });
            }
            None
        }

        /// Aspect ratio (height / width) of the cached poster, available once
        /// the capture has decoded. Lets the browser size the video tile to the
        /// real frame instead of stretching it into the placeholder square.
        pub fn aspect(&self, path: &Path) -> Option<f32> {
            match self.entries.get(path) {
                Some(Thumb::Ready(tex)) => {
                    let [w, h] = tex.size();
                    if w == 0 {
                        None
                    } else {
                        Some(h as f32 / w as f32)
                    }
                }
                _ => None,
            }
        }
    }

    struct CaptureCtx {
        max_edge: u32,
        width: u32,
        height: u32,
        buffer: Vec<u8>,
        frame_count: u32,
        latest: SharedFrame,
    }

    /// Initial frames to skip before grabbing the poster (avoids black lead-ins).
    const THUMB_SKIP_FRAMES: u32 = 3;

    /// Decode one representative frame of `path`, scaled so its long edge is about
    /// `max_edge`, as an RGBA image. `None` if no frame arrives within the timeout.
    fn capture_poster(path: &Path, max_edge: u32) -> Option<egui::ColorImage> {
        let instance = vlc::Instance::new()?;
        let media = vlc::Media::new_path(&instance, path)?;
        // Poster capture only needs a video frame. Disable the audio output for
        // this media so decoding a thumbnail never plays the soundtrack — otherwise
        // scrolling the browser starts background audio for each video it captures.
        unsafe {
            let opt = std::ffi::CString::new(":no-audio").unwrap();
            sys::libvlc_media_add_option(media.raw(), opt.as_ptr());
        }
        let player = vlc::MediaPlayer::new(&instance)?;
        player.set_media(&media);

        let latest: SharedFrame = Arc::new(Mutex::new(None));
        let ctx_ptr = Box::into_raw(Box::new(CaptureCtx {
            max_edge,
            width: 0,
            height: 0,
            buffer: Vec::new(),
            frame_count: 0,
            latest: latest.clone(),
        }));

        unsafe {
            let mp = player.raw();
            sys::libvlc_video_set_callbacks(
                mp,
                Some(thumb_lock_cb),
                Some(thumb_unlock_cb),
                Some(thumb_display_cb),
                ctx_ptr as *mut c_void,
            );
            let setup_fp: unsafe extern "C" fn(
                *mut *mut c_void,
                *mut c_char,
                *mut c_uint,
                *mut c_uint,
                *mut c_uint,
                *mut c_uint,
            ) -> c_uint = thumb_setup_cb;
            let setup: sys::libvlc_video_format_cb = Some(std::mem::transmute(setup_fp));
            sys::libvlc_video_set_format_callbacks(mp, setup, Some(thumb_cleanup_cb));
        }

        if player.play().is_err() {
            unsafe { drop(Box::from_raw(ctx_ptr)) };
            return None;
        }

        // Poll for the first settled frame, then stop.
        let deadline = Instant::now() + Duration::from_secs(6);
        let mut result = None;
        while Instant::now() < deadline {
            if let Some(img) = latest.lock().ok().and_then(|mut g| g.take()) {
                result = Some(img);
                break;
            }
            std::thread::sleep(Duration::from_millis(40));
        }

        player.stop();
        unsafe { drop(Box::from_raw(ctx_ptr)) };
        result
    }

    unsafe extern "C" fn thumb_setup_cb(
        opaque: *mut *mut c_void,
        chroma: *mut c_char,
        width: *mut c_uint,
        height: *mut c_uint,
        pitches: *mut c_uint,
        lines: *mut c_uint,
    ) -> c_uint {
        unsafe {
            let cc = &mut *(*opaque as *mut CaptureCtx);
            let (nw, nh) = (*width, *height);
            if nw == 0 || nh == 0 {
                return 0;
            }
            // Scale the long edge down to max_edge, preserving aspect.
            let scale = (cc.max_edge as f32 / nw.max(nh) as f32).min(1.0);
            let sw = ((nw as f32 * scale) as u32).max(2);
            let sh = ((nh as f32 * scale) as u32).max(2);

            std::ptr::copy_nonoverlapping(b"RGBA".as_ptr(), chroma as *mut u8, 4);
            *width = sw;
            *height = sh;
            *pitches = sw * 4;
            *lines = sh;

            cc.width = sw;
            cc.height = sh;
            cc.buffer = vec![0u8; (sw * 4 * sh) as usize];
            1
        }
    }

    unsafe extern "C" fn thumb_cleanup_cb(_opaque: *mut c_void) {}

    unsafe extern "C" fn thumb_lock_cb(opaque: *mut c_void, planes: *mut c_void) -> *mut c_void {
        unsafe {
            let cc = &mut *(opaque as *mut CaptureCtx);
            *(planes as *mut *mut c_void) = cc.buffer.as_mut_ptr() as *mut c_void;
            std::ptr::null_mut()
        }
    }

    unsafe extern "C" fn thumb_unlock_cb(
        _opaque: *mut c_void,
        _picture: *mut c_void,
        _planes: *const *mut c_void,
    ) {
    }

    unsafe extern "C" fn thumb_display_cb(opaque: *mut c_void, _picture: *mut c_void) {
        unsafe {
            let cc = &mut *(opaque as *mut CaptureCtx);
            cc.frame_count += 1;
            if cc.frame_count < THUMB_SKIP_FRAMES {
                return;
            }
            let expected = (cc.width * cc.height * 4) as usize;
            if expected == 0 || cc.buffer.len() != expected {
                return;
            }
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [cc.width as usize, cc.height as usize],
                &cc.buffer,
            );
            if let Ok(mut g) = cc.latest.lock() {
                *g = Some(img);
            }
        }
    }
}
