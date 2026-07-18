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
// VLC availability — drives whether the viewer plays in-app, offers to install
// VLC, or reports that this build has no video backend.
// ---------------------------------------------------------------------------

/// What the centre viewer should show for a video that has no running player.
// Which variants are constructed depends on the build: the `vlc` build returns
// `Available`/`NeedsInstall`, the stub build only `Unsupported` — so one variant
// is always cfg-unused. All three are still matched in `video_notice`.
#[allow(dead_code)]
pub enum VideoSupport {
    /// A libVLC runtime is available; playback should work (a `None` player then
    /// means the clip itself failed to start).
    Available,
    /// This build can play video but no libVLC runtime was found — offer to
    /// install VLC.
    NeedsInstall,
    /// This build has no video backend (compiled without the `vlc` feature).
    Unsupported,
}

/// Where the "Install VLC" button sends the user.
pub const VLC_DOWNLOAD_URL: &str = "https://www.videolan.org/vlc/";

/// Whether this build can play video and, if so, whether a libVLC runtime is
/// present. Cheap to call every frame (the Windows probe is memoised).
pub fn support() -> VideoSupport {
    #[cfg(not(feature = "vlc"))]
    {
        VideoSupport::Unsupported
    }
    #[cfg(feature = "vlc")]
    {
        if vlc_runtime_available() {
            VideoSupport::Available
        } else {
            VideoSupport::NeedsInstall
        }
    }
}

/// Whether a usable libVLC runtime is present.
///
/// On Windows libVLC is *delay-loaded* (see `.cargo/config.toml`), so the app
/// launches even when VLC isn't installed; we probe for a system/bundled install
/// here and register its folder on the DLL search path *before* any libVLC call,
/// so the delayed load (and `libvlccore.dll` / the `plugins\` folder beside it)
/// resolves. On other platforms libVLC is linked at load time, so if the process
/// is running it's necessarily present.
#[cfg(all(feature = "vlc", windows))]
fn vlc_runtime_available() -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    // Cache only *success*: while VLC is absent we keep re-probing (cheap — just a
    // few path checks), so installing it mid-session is picked up without a
    // restart. Once found, the DLL-dir setup has run and we stop probing.
    static READY: AtomicBool = AtomicBool::new(false);
    if READY.load(Ordering::Relaxed) {
        return true;
    }
    if setup_windows_vlc() {
        READY.store(true, Ordering::Relaxed);
        true
    } else {
        false
    }
}

#[cfg(all(feature = "vlc", not(windows)))]
fn vlc_runtime_available() -> bool {
    true
}

/// Locate a libVLC install, add its folder to the DLL search path, and pre-load
/// `libvlc.dll` by full path so later delay-loaded calls reuse that module.
/// Returns whether the load succeeded.
#[cfg(all(feature = "vlc", windows))]
fn setup_windows_vlc() -> bool {
    use std::os::raw::c_void;
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDefaultDllDirectories(flags: u32) -> i32;
        fn AddDllDirectory(path: *const u16) -> *mut c_void;
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    }
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x0000_0008;

    let wide = |p: &std::path::Path| -> Vec<u16> {
        p.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    };

    let Some(dir) = find_vlc_dir() else { return false };
    let dll = dir.join("libvlc.dll");
    if !dll.exists() {
        return false;
    }
    unsafe {
        // Let the loader honour AddDllDirectory for dependency resolution.
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        AddDllDirectory(wide(&dir).as_ptr());
        // Load by full path; ALTERED_SEARCH_PATH makes libvlccore.dll resolve from
        // the same folder. Keep the handle (no FreeLibrary) so the delay-loaded
        // by-name calls bind to this already-loaded module.
        let h = LoadLibraryExW(wide(&dll).as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH);
        !h.is_null()
    }
}

/// Find a folder containing `libvlc.dll`: next to our exe (bundled / `build.rs`
/// staged), then a `VLC_DIR` override, the standard install locations, and finally
/// the registry's recorded install dir.
#[cfg(all(feature = "vlc", windows))]
fn find_vlc_dir() -> Option<std::path::PathBuf> {
    let has_dll = |d: std::path::PathBuf| -> Option<std::path::PathBuf> {
        if d.join("libvlc.dll").exists() { Some(d) } else { None }
    };

    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()))
        && let Some(d) = has_dll(exe_dir) {
            return Some(d);
        }
    if let Some(d) = std::env::var_os("VLC_DIR").map(std::path::PathBuf::from).and_then(has_dll) {
        return Some(d);
    }
    for var in ["ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"] {
        if let Some(base) = std::env::var_os(var)
            && let Some(d) = has_dll(std::path::Path::new(&base).join("VideoLAN").join("VLC")) {
                return Some(d);
            }
    }
    registry_vlc_dir()
}

/// Read `HKLM\SOFTWARE\VideoLAN\VLC\InstallDir` (the path the VLC installer
/// records). Best-effort: any failure just yields `None`.
#[cfg(all(feature = "vlc", windows))]
fn registry_vlc_dir() -> Option<std::path::PathBuf> {
    use std::os::raw::c_void;

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn RegGetValueW(
            hkey: *mut c_void,
            subkey: *const u16,
            value: *const u16,
            flags: u32,
            ptype: *mut u32,
            pvdata: *mut c_void,
            pcbdata: *mut u32,
        ) -> i32;
    }
    const HKEY_LOCAL_MACHINE: *mut c_void = 0x8000_0002u32 as usize as *mut c_void;
    const RRF_RT_REG_SZ: u32 = 0x0000_0002;

    let subkey: Vec<u16> = "SOFTWARE\\VideoLAN\\VLC".encode_utf16().chain(std::iter::once(0)).collect();
    let value: Vec<u16> = "InstallDir".encode_utf16().chain(std::iter::once(0)).collect();
    let mut buf = [0u16; 520];
    let mut len = (buf.len() * 2) as u32; // capacity in bytes
    let rc = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            buf.as_mut_ptr() as *mut c_void,
            &mut len,
        )
    };
    if rc != 0 {
        return None; // ERROR_SUCCESS == 0
    }
    let chars = (len as usize / 2).saturating_sub(1); // drop the trailing NUL
    let dir = std::path::PathBuf::from(String::from_utf16_lossy(&buf[..chars]));
    if dir.join("libvlc.dll").exists() { Some(dir) } else { None }
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

    #[allow(dead_code)] // the stub VideoPreviews never starts a player
    pub fn start_preview(_path: &Path, _ctx: &egui::Context) -> Option<VideoPlayer> {
        None
    }

    pub fn frame(&mut self, _ctx: &egui::Context) -> Option<egui::TextureHandle> {
        None
    }
}

/// Stub thumbnail-preview manager when the `vlc` feature is off — always returns
/// `None`, so tiles show the static poster.
#[cfg(not(feature = "vlc"))]
pub struct VideoPreviews;

#[cfg(not(feature = "vlc"))]
impl VideoPreviews {
    pub fn new() -> Self {
        VideoPreviews
    }
    pub fn set_enabled(&mut self, _on: bool) {}
    pub fn begin_frame(&mut self) {}
    pub fn frame(&mut self, _path: &Path, _ctx: &egui::Context) -> Option<egui::TextureHandle> {
        None
    }
    pub fn end_frame(&mut self) {}
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

    pub fn set_max_edge(&mut self, _max_edge: u32) {}
}

/// Stub frame sampler when the `vlc` feature is off — the AI worker turns the
/// empty Vec into a readable "needs the VLC build" error.
#[cfg(not(feature = "vlc"))]
#[cfg_attr(not(feature = "llm"), allow(dead_code))] // only the AI worker samples frames
pub fn capture_frames(
    _path: &Path,
    _max_frames: usize,
    _max_edge: u32,
) -> Vec<(i64, egui::ColorImage)> {
    Vec::new()
}

/// Stub poster grab when the `vlc` feature is off — video attachments in the
/// AI chat keep their filename chip.
#[cfg(not(feature = "vlc"))]
pub fn capture_poster(_path: &Path, _max_edge: u32) -> Option<egui::ColorImage> {
    None
}

// ---------------------------------------------------------------------------
// Real libVLC-backed player (feature = "vlc").
// ---------------------------------------------------------------------------
#[cfg(feature = "vlc")]
pub use backend::{capture_poster, VideoPlayer, VideoPreviews, VideoThumbs};
// Only the AI worker samples frames, so this re-export is idle in non-llm builds.
#[cfg(feature = "vlc")]
#[cfg_attr(not(feature = "llm"), allow(unused_imports))]
pub use backend::capture_frames;

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

    /// The video-format ("setup") callback with its true libVLC signature —
    /// it returns the picture count. vlc-rs 0.3 mis-declares the return type
    /// as `()`, so the callbacks are transmuted from the real shape to the
    /// declared one (see the `set_format_callbacks` call sites).
    type SetupCb = unsafe extern "C" fn(
        *mut *mut c_void,
        *mut c_char,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
    ) -> c_uint;
    /// The `()`-returning shape vlc-rs declares for [`SetupCb`].
    type SetupCbAsDeclared = unsafe extern "C" fn(
        *mut *mut c_void,
        *mut c_char,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
    );

    // --- In-memory media (entries of a mounted zip archive) -----------------
    //
    // libVLC can only open real files by path, but zip entries are viewed
    // without ever being extracted (src/archive.rs). For those, the whole
    // entry is inflated to memory and handed to libVLC through
    // `libvlc_media_new_callbacks` — its "read a custom stream" API. vlc-rs
    // 0.3 doesn't bind it, so it's declared here; the symbol is exported by
    // the release import library (ci/vlc.def) and the VLC SDK alike.

    type MediaOpenCb =
        unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut u64) -> std::os::raw::c_int;
    type MediaReadCb = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> isize;
    type MediaSeekCb = unsafe extern "C" fn(*mut c_void, u64) -> std::os::raw::c_int;
    type MediaCloseCb = unsafe extern "C" fn(*mut c_void);

    unsafe extern "C" {
        fn libvlc_media_new_callbacks(
            p_instance: *mut sys::libvlc_instance_t,
            open_cb: Option<MediaOpenCb>,
            read_cb: Option<MediaReadCb>,
            seek_cb: Option<MediaSeekCb>,
            close_cb: Option<MediaCloseCb>,
            opaque: *mut c_void,
        ) -> *mut sys::libvlc_media_t;
    }

    /// The bytes behind an in-memory media. Owned by the player (or poster
    /// capture) and freed only after the player stops: looping playback
    /// re-opens the stream each pass, so the buffer must outlive every
    /// open/close cycle — a close callback can't own it.
    pub(super) struct MemShared {
        data: Vec<u8>,
    }

    /// One open stream over a [`MemShared`] buffer (libVLC may open the same
    /// media several times — probing, playing, looping). Created in
    /// `mem_open_cb`, freed in `mem_close_cb`.
    struct MemCursor {
        shared: *const MemShared,
        pos: usize,
    }

    unsafe extern "C" fn mem_open_cb(
        opaque: *mut c_void,
        datap: *mut *mut c_void,
        sizep: *mut u64,
    ) -> std::os::raw::c_int {
        unsafe {
            let shared = opaque as *const MemShared;
            *sizep = (*shared).data.len() as u64;
            *datap = Box::into_raw(Box::new(MemCursor { shared, pos: 0 })) as *mut c_void;
        }
        0
    }

    unsafe extern "C" fn mem_read_cb(datap: *mut c_void, buf: *mut u8, len: usize) -> isize {
        unsafe {
            let cur = &mut *(datap as *mut MemCursor);
            let data = &(*cur.shared).data;
            let n = len.min(data.len().saturating_sub(cur.pos));
            std::ptr::copy_nonoverlapping(data.as_ptr().add(cur.pos), buf, n);
            cur.pos += n;
            n as isize
        }
    }

    unsafe extern "C" fn mem_seek_cb(datap: *mut c_void, offset: u64) -> std::os::raw::c_int {
        unsafe {
            let cur = &mut *(datap as *mut MemCursor);
            if offset > (*cur.shared).data.len() as u64 {
                return -1;
            }
            cur.pos = offset as usize;
        }
        0
    }

    unsafe extern "C" fn mem_close_cb(datap: *mut c_void) {
        unsafe { drop(Box::from_raw(datap as *mut MemCursor)) };
    }

    /// A media handle for `path` — by path for real files, or fed from memory
    /// for zip entries. `mem` (null for path media) must be freed with
    /// `Box::from_raw` after the player using the media has stopped.
    struct MediaSource {
        /// Keeps the path-based `vlc::Media` alive until it's set on the
        /// player; `None` for memory media (we manage the raw refcount).
        media: Option<vlc::Media>,
        raw: *mut sys::libvlc_media_t,
        mem: *mut MemShared,
    }

    fn open_media(instance: &vlc::Instance, path: &Path) -> Option<MediaSource> {
        if crate::archive::is_entry(path) {
            let data = crate::archive::read(path).ok()?;
            let mem = Box::into_raw(Box::new(MemShared { data }));
            let raw = unsafe {
                libvlc_media_new_callbacks(
                    instance.raw(),
                    Some(mem_open_cb),
                    Some(mem_read_cb),
                    Some(mem_seek_cb),
                    Some(mem_close_cb),
                    mem as *mut c_void,
                )
            };
            if raw.is_null() {
                // No stream was opened, so the buffer is still exclusively ours.
                unsafe { drop(Box::from_raw(mem)) };
                return None;
            }
            Some(MediaSource { media: None, raw, mem })
        } else {
            let media = vlc::Media::new_path(instance, path)?;
            let raw = media.raw();
            Some(MediaSource { media: Some(media), raw, mem: std::ptr::null_mut() })
        }
    }

    impl MediaSource {
        /// Attach the media to `player` and drop our reference (the player
        /// retains its own). Returns the keep-alive buffer pointer the caller
        /// must free once the player has stopped (null for path media).
        fn set_on(self, player: &vlc::MediaPlayer) -> *mut MemShared {
            match &self.media {
                Some(m) => player.set_media(m), // vlc::Media's Drop releases ours
                None => unsafe {
                    sys::libvlc_media_player_set_media(player.raw(), self.raw);
                    sys::libvlc_media_release(self.raw);
                },
            }
            self.mem
        }
    }

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
        /// Bytes behind an in-memory (zip entry) media — null for file media.
        /// Freed on drop, after the player stops.
        mem_ptr: *mut MemShared,
        latest: SharedFrame,
        texture: Option<egui::TextureHandle>,
    }

    impl VideoPlayer {
        /// Start normal playback (audio on; loops per the user's Settings).
        pub fn start(path: &Path, ctx: &egui::Context) -> Option<VideoPlayer> {
            Self::start_with(path, ctx, loop_enabled(), false, None)
        }

        /// Start a muted thumbnail preview (no audio, loops so a short clip keeps
        /// moving instead of freezing) — used by [`VideoPreviews`] for the
        /// auto-playing tiles. Only the first [`PREVIEW_SECS`] seconds play; the
        /// loop restarts there, so a long video never decodes past that point.
        pub fn start_preview(path: &Path, ctx: &egui::Context) -> Option<VideoPlayer> {
            Self::start_with(path, ctx, true, true, Some(PREVIEW_SECS))
        }

        fn start_with(
            path: &Path,
            ctx: &egui::Context,
            looping: bool,
            muted: bool,
            stop_secs: Option<u64>,
        ) -> Option<VideoPlayer> {
            let instance = vlc::Instance::new()?;
            let media = open_media(&instance, path)?;
            // Loop the clip when requested. A large repeat count stands in for
            // "infinite" (libVLC has no unbounded value).
            if looping {
                unsafe {
                    let opt = std::ffi::CString::new(":input-repeat=65535").unwrap();
                    sys::libvlc_media_add_option(media.raw, opt.as_ptr());
                }
            }
            // End the input early when capped: with looping on, each repeat then
            // plays only the first `stop_secs` seconds (clips shorter than the
            // cap just loop at their natural end as before).
            if let Some(s) = stop_secs {
                unsafe {
                    let opt = std::ffi::CString::new(format!(":stop-time={s}")).unwrap();
                    sys::libvlc_media_add_option(media.raw, opt.as_ptr());
                }
            }
            // Preview tiles never play sound (a grid of chattering clips would be
            // unbearable) — disable audio output for this media.
            if muted {
                unsafe {
                    let opt = std::ffi::CString::new(":no-audio").unwrap();
                    sys::libvlc_media_add_option(media.raw, opt.as_ptr());
                }
            }
            let player = vlc::MediaPlayer::new(&instance)?;
            let mem_ptr = media.set_on(&player);

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
                let setup: sys::libvlc_video_format_cb =
                    Some(std::mem::transmute::<SetupCb, SetupCbAsDeclared>(setup_cb));
                sys::libvlc_video_set_format_callbacks(mp, setup, Some(cleanup_cb));
            }

            if player.play().is_err() {
                // No callbacks have fired yet, so it's safe to free the box.
                unsafe { drop(Box::from_raw(ctx_ptr)) };
                if !mem_ptr.is_null() {
                    // Play never started, so no stream is reading the buffer.
                    unsafe { drop(Box::from_raw(mem_ptr)) };
                }
                return None;
            }

            Some(VideoPlayer { _instance: instance, player, ctx_ptr, mem_ptr, latest, texture: None })
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
            // can't fire again, then free the opaque they were writing into —
            // and, for in-memory (zip entry) media, the stream's byte buffer
            // (the stopped input has closed every cursor over it).
            self.player.stop();
            unsafe { drop(Box::from_raw(self.ctx_ptr)) };
            if !self.mem_ptr.is_null() {
                unsafe { drop(Box::from_raw(self.mem_ptr)) };
            }
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
        /// Poster decode resolution (longest side, px). Mirrors the image
        /// thumbnail setting so "HD thumbnails" makes video posters crisp too.
        max_edge: u32,
    }

    impl VideoThumbs {
        pub fn new() -> Self {
            let (tx, rx) = mpsc::channel();
            VideoThumbs { entries: HashMap::new(), tx, rx, in_flight: 0, busy: None, max_edge: 320 }
        }

        /// Change the poster decode resolution (longest side, px). If it actually
        /// changes, cached posters are dropped so they re-capture at the new size.
        pub fn set_max_edge(&mut self, max_edge: u32) {
            if self.max_edge != max_edge {
                self.max_edge = max_edge;
                self.entries.clear();
            }
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

            // No libVLC runtime → don't spawn a decode (the delay-loaded call
            // would fail); show the placeholder video icon instead. This also runs
            // the one-time DLL-dir setup before the capture thread touches libVLC.
            if !super::vlc_runtime_available() {
                self.entries.insert(path.to_path_buf(), Thumb::Failed);
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
                let max_edge = self.max_edge;
                std::thread::spawn(move || {
                    let img = capture_poster(&p, max_edge);
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

    // -----------------------------------------------------------------------
    // Auto-playing thumbnail previews.
    // -----------------------------------------------------------------------

    /// Plays muted, looping previews on visible video tiles (the "Video thumbnail
    /// play" setting). Bounded in count so a folder of clips can't spin up dozens
    /// of decoders. A preview loops the first [`PREVIEW_SECS`] seconds of the
    /// clip the whole time its tile is on screen and is stopped as soon as the
    /// tile scrolls out of view.
    pub struct VideoPreviews {
        players: HashMap<PathBuf, VideoPlayer>,
        /// Paths a tile asked to preview this frame (mark-and-sweep for offscreen).
        requested: std::collections::HashSet<PathBuf>,
        enabled: bool,
    }

    /// Most previews decoding at once (each is a full libVLC pipeline).
    const MAX_PREVIEWS: usize = 6;

    /// How much of a clip a preview plays before looping back to the start —
    /// tiles tease the opening seconds instead of playing a long video through.
    const PREVIEW_SECS: u64 = 15;

    impl VideoPreviews {
        pub fn new() -> Self {
            VideoPreviews {
                players: HashMap::new(),
                requested: std::collections::HashSet::new(),
                enabled: false,
            }
        }

        /// Push the current "Video thumbnail play" setting. Turning it off stops
        /// and frees every running preview at once.
        pub fn set_enabled(&mut self, on: bool) {
            if self.enabled && !on {
                self.players.clear();
            }
            self.enabled = on;
        }

        /// Call once per frame before drawing tiles: reset the visibility marks.
        pub fn begin_frame(&mut self) {
            self.requested.clear();
        }

        /// The live preview texture for `path`, or `None` to show the poster
        /// instead (disabled, or over the concurrency cap, or still warming up).
        /// A tile calls this every frame it is visible; the preview plays for as
        /// long as the tile keeps calling (i.e. stays on screen).
        pub fn frame(&mut self, path: &Path, ctx: &egui::Context) -> Option<egui::TextureHandle> {
            if !self.enabled || !super::vlc_runtime_available() {
                return None;
            }
            self.requested.insert(path.to_path_buf());

            if let Some(player) = self.players.get_mut(path) {
                return player.frame(ctx);
            }
            // Not previewing yet — start one if we're under the cap.
            if self.players.len() < MAX_PREVIEWS
                && let Some(player) = VideoPlayer::start_preview(path, ctx) {
                    self.players.insert(path.to_path_buf(), player);
                }
            None
        }

        /// Call once per frame after drawing tiles: drop previews for tiles that
        /// were not visible this frame so they stop decoding (and replay fresh if
        /// they scroll back into view).
        pub fn end_frame(&mut self) {
            let req = &self.requested;
            self.players.retain(|p, _| req.contains(p));
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
    /// Also used by the AI chat for its video-attachment thumbnails (pub).
    pub fn capture_poster(path: &Path, max_edge: u32) -> Option<egui::ColorImage> {
        let instance = vlc::Instance::new()?;
        let media = open_media(&instance, path)?;
        // Poster capture only needs a video frame. Disable the audio output for
        // this media so decoding a thumbnail never plays the soundtrack — otherwise
        // scrolling the browser starts background audio for each video it captures.
        unsafe {
            let opt = std::ffi::CString::new(":no-audio").unwrap();
            sys::libvlc_media_add_option(media.raw, opt.as_ptr());
        }
        let player = vlc::MediaPlayer::new(&instance)?;
        let mem_ptr = media.set_on(&player);

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
            let setup: sys::libvlc_video_format_cb =
                Some(std::mem::transmute::<SetupCb, SetupCbAsDeclared>(thumb_setup_cb));
            sys::libvlc_video_set_format_callbacks(mp, setup, Some(thumb_cleanup_cb));
        }

        if player.play().is_err() {
            unsafe { drop(Box::from_raw(ctx_ptr)) };
            if !mem_ptr.is_null() {
                unsafe { drop(Box::from_raw(mem_ptr)) };
            }
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
        // The stopped input has closed all cursors over the in-memory buffer.
        if !mem_ptr.is_null() {
            unsafe { drop(Box::from_raw(mem_ptr)) };
        }
        result
    }

    /// Decode up to `max_frames` evenly-spaced frames of `path` (long edge ≈
    /// `max_edge`) for the AI chat's video input, as (timestamp ms, frame)
    /// pairs in playback order. The same muted throwaway player as
    /// [`capture_poster`], plus a position seek between grabs. An unplayable
    /// clip returns an empty Vec.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))] // only the AI worker samples frames
    pub fn capture_frames(
        path: &Path,
        max_frames: usize,
        max_edge: u32,
    ) -> Vec<(i64, egui::ColorImage)> {
        let Some(instance) = vlc::Instance::new() else { return Vec::new() };
        let Some(media) = open_media(&instance, path) else { return Vec::new() };
        unsafe {
            let no_audio = std::ffi::CString::new(":no-audio").unwrap();
            sys::libvlc_media_add_option(media.raw, no_audio.as_ptr());
            // A short clip could reach its end (closing the input) between
            // seeks — keep the throwaway player looping; it's stopped
            // explicitly below.
            let repeat = std::ffi::CString::new(":input-repeat=65535").unwrap();
            sys::libvlc_media_add_option(media.raw, repeat.as_ptr());
        }
        let Some(player) = vlc::MediaPlayer::new(&instance) else { return Vec::new() };
        let mem_ptr = media.set_on(&player);

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
            let setup: sys::libvlc_video_format_cb =
                Some(std::mem::transmute::<SetupCb, SetupCbAsDeclared>(thumb_setup_cb));
            sys::libvlc_video_set_format_callbacks(mp, setup, Some(thumb_cleanup_cb));
        }

        let wait_frame = |timeout: Duration| -> Option<egui::ColorImage> {
            let deadline = Instant::now() + timeout;
            while Instant::now() < deadline {
                if let Some(img) = latest.lock().ok().and_then(|mut g| g.take()) {
                    return Some(img);
                }
                std::thread::sleep(Duration::from_millis(40));
            }
            None
        };

        let mut out = Vec::new();
        if player.play().is_ok() {
            // The first settled frame proves the clip decodes and makes the
            // length readable; it isn't itself kept.
            if wait_frame(Duration::from_secs(6)).is_some() {
                let len_ms = unsafe { sys::libvlc_media_player_get_length(player.raw()) };
                // About one frame per second of clip, between 2 and max_frames
                // — longer clips just get sparser sampling.
                let n = if len_ms > 0 {
                    ((len_ms as f64 / 1000.0).round() as usize).clamp(2, max_frames.max(1))
                } else {
                    max_frames.clamp(1, 6)
                };
                for i in 0..n {
                    let pos = (i as f32 + 0.5) / n as f32;
                    player.set_position(pos);
                    // Let the seek land, then drop whatever frame was already
                    // waiting so the grab below is really post-seek.
                    std::thread::sleep(Duration::from_millis(100));
                    if let Ok(mut g) = latest.lock() {
                        *g = None;
                    }
                    if let Some(img) = wait_frame(Duration::from_secs(3)) {
                        let ts = player
                            .get_time()
                            .filter(|t| *t >= 0)
                            .unwrap_or((pos * len_ms.max(0) as f32) as i64);
                        out.push((ts, img));
                    }
                }
            }
            player.stop();
        }
        unsafe {
            drop(Box::from_raw(ctx_ptr));
            if !mem_ptr.is_null() {
                drop(Box::from_raw(mem_ptr));
            }
        }
        out
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
