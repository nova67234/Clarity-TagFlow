// Hide the console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;
use egui::{Color32, CornerRadius, Margin, Stroke};

/// Always-available image extensions (pure-Rust decoders, no heavy deps).
/// `jfif` is just JPEG with a different extension — the decoder content-sniffs it.
/// `hdr` (Radiance RGBE) decodes via the `image` crate's lightweight HDR decoder
/// and is tone-mapped for display (see `image_cache::decode_hdr`).
const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jfif", "gif", "bmp", "webp", "ico", "tif", "tiff", "hdr",
];

/// "Extended" image extensions decoded via the heavier pure-Rust crates: AVIF
/// (avif-parse + rav1d), HEIC/HEIF (heic), and camera raw — DNG, Sony ARW,
/// Canon CR2 and Nikon NEF — (zenraw). Only recognised when the user enables them in Settings
/// AND the app was built with the `avif` feature. Defined only in such builds so
/// a stale persisted setting can't make a normal build list files it can't decode.
#[cfg(feature = "avif")]
const EXTENDED_IMAGE_EXTENSIONS: &[&str] = &["avif", "heic", "heif", "dng", "arw", "cr2", "nef"];

/// True when `ext` (without the dot, any case) is one of the heavy "extended"
/// image formats — AVIF / HEIC / HEIF and the TIFF-based camera raws — that need
/// the pure-Rust decoders in `crate::avif` rather than `image::open`. Centralises
/// the list that was otherwise repeated across every decode path. Always `false`
/// in builds without the `avif` feature (those files aren't recognised anyway).
pub(crate) fn is_extended_extension(ext: &str) -> bool {
    #[cfg(feature = "avif")]
    {
        EXTENDED_IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
    }
    #[cfg(not(feature = "avif"))]
    {
        let _ = ext;
        false
    }
}

/// Runtime flag mirroring `Settings::enable_extended_formats`, so the free
/// `is_image()` helper (called all over) can gate the extended formats without
/// threading `Settings` through every call site.
#[cfg(feature = "avif")]
static EXTENDED_FORMATS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Browser thumbnail decode resolution (longest side, px). HD gives crisper
/// tiles at the cost of more memory and slower decoding.
const THUMB_MAX_EDGE: u32 = 320;
const THUMB_MAX_EDGE_HD: u32 = 768;

/// Video file extensions. These are listed, tag-able, move-able and delete-able
/// like images, but shown as a placeholder — there's no in-app frame decode or
/// playback (that would need a video backend such as ffmpeg).
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "mkv", "webm", "avi", "m4v", "wmv", "flv",
];

// ---------------------------------------------------------------------------
// Theme — mirrors the dark palette from terminus2's AppTheme.
// ---------------------------------------------------------------------------
mod theme;
use theme::*;

mod ai_models;
mod ai_orb;
// The full-window AI Chat view (Settings → AI Model → Activate AI Chat).
mod ai_chat;
// The OmniVoice neural voice for the chat's Listen buttons (Python sidecar).
mod voice;
// Role playing for the AI Chat: persona, names, shared memory diary.
mod roleplay;
#[cfg(feature = "avif")]
mod avif;
mod backup;
mod bgremove;
mod civitai;
mod depth;
mod detect;
mod emoji;
mod favorites;
mod ftp;
mod image_cache;
mod download;
mod gallery;
mod gallery_detail;
// Flux text-to-image generation (ComfyUI backend) — NVIDIA-only, like Pixal3D.
#[cfg(not(target_os = "macos"))]
mod generate;
mod gif_info;
mod left_browser;
mod left_panel_settings;
// Local AI model (Settings → AI Model): Gemma 4 vision via llama.cpp, fully
// in-process. The module always compiles; the heavy bindings sit behind the
// `llm` cargo feature.
mod llm;
mod mp4;
// Pixal3D image->3D requirement setup — Linux/Windows only (compiled out on macOS).
#[cfg(not(target_os = "macos"))]
mod pixal3d;
// Interactive 3D viewer for Pixal3D's GLB output (centre panel). Linux/Windows
// only, like Pixal3D — pulls in three-d, which is a not-macOS dependency.
#[cfg(not(target_os = "macos"))]
mod scene3d;
mod raw_preview;
mod right_details;
mod scan;
mod sd_metadata;
mod secret;
mod settings;
mod spellcheck;
mod splash;
mod tag_manager;
mod tag_manager_settings;
mod tagger;
mod top_bar;
mod update;
mod video; // embedded VLC playback (real backend only under --features vlc)
mod zoom; // zoom + pan for the centre image viewer

fn main() -> eframe::Result {
    // libVLC playback needs its DLLs + plugins beside the exe; build.rs stages
    // them there on Windows (libVLC finds the plugins relative to its own DLL).

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 720.0]) // Slightly wider default to accommodate both panels comfortably
            .with_min_inner_size([920.0, 460.0])
            .with_icon(load_app_icon()) // taskbar / title-bar icon
            .with_drag_and_drop(true),
        // Use the OpenGL backend: the Pixal3D 3D viewer (src/scene3d.rs) renders
        // with three-d into eframe's GL context. eframe 0.34 defaults to wgpu,
        // which would silently drop the glow paint callback.
        renderer: eframe::Renderer::Glow,
        // Request a depth buffer: egui itself doesn't need one, but the 3D viewer
        // does — without it three-d can't depth-test, so the model renders
        // see-through (back faces show through the front) while orbiting.
        depth_buffer: 24,
        ..Default::default()
    };

    eframe::run_native(
        "Clarity TagFlow",
        options,
        Box::new(|cc| {
            // REQUIRED: Install the image loaders so egui can parse SVG bytes
            egui_extras::install_image_loaders(&cc.egui_ctx);

            // Ctrl +/-/0 drive the image viewer's zoom (see src/zoom.rs), so turn
            // off egui's built-in keyboard zoom that would otherwise scale the whole
            // UI on those shortcuts. Ctrl+scroll / pinch still reach us via
            // zoom_delta(); we just don't want egui to also rescale the interface.
            cc.egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);

            // egui's bundled fonts have no CJK glyphs (Japanese / Chinese / Korean,
            // common in Civitai model names & tags, e.g. アルベド) nor the fancy
            // "Fraktur"/math letters & symbols people put in SD prompts (e.g. 𝔗ℜ𝔊),
            // so both render as tofu boxes. Append the platform's system CJK and
            // math/symbol fonts as fallbacks so those glyphs resolve app-wide.
            install_fallback_fonts(&cc.egui_ctx);

            let mut app = ViewerApp::default();
            // Restore saved settings (if any) from eframe's persistent storage.
            if let Some(storage) = cc.storage {
                if let Some(saved) = eframe::get_value::<settings::Settings>(storage, settings::STORAGE_KEY) {
                    app.settings = saved;
                    // Restore the last-used AI tagger model into the Tag Manager.
                    app.tag_manager.ai_model = app.settings.last_ai_model.clone();
                }
            }
            // Apply the saved colour theme before the first paint so a Light-mode
            // user doesn't see a dark flash on launch. The Glass config (incl.
            // dark/light panels) must be pushed first — the palette reads it.
            set_glass_config(app.settings.glass_bg, app.settings.glass_backdrop, app.settings.glass_light);
            set(app.settings.theme);
            app.last_theme = app.settings.theme;
            app.last_glass_light = app.settings.glass_light;
            apply_theme(&cc.egui_ctx);
            // Optional: open a folder passed on the command line (e.g. "Open with").
            if let Some(arg) = std::env::args().nth(1) {
                let dir = PathBuf::from(arg);
                if dir.is_dir() {
                    app.load_folder(&dir);
                }
            }
            Ok(Box::new(app))
        }),
    )
}

/// Decode the bundled PNG into an egui window icon (taskbar / title bar).
fn load_app_icon() -> egui::IconData {
    let bytes = include_bytes!("../icons/app-icon-256.png");
    match image::load_from_memory(bytes) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            egui::IconData { rgba: rgba.into_raw(), width, height }
        }
        Err(_) => egui::IconData::default(),
    }
}

/// Push the dark palette into egui's global visuals so stock widgets
/// (text fields, scrollbars, etc.) match the custom-painted panels.
fn apply_theme(ctx: &egui::Context) {
    // Delegates to the theme module, which picks the active Dark/Light palette.
    apply(ctx);
}

// ---------------------------------------------------------------------------
// Background removal (BiRefNet) — async right-click action
// ---------------------------------------------------------------------------

/// Catalog folder for the BiRefNet background-removal model (see ai_models.rs).
const BG_FOLDER: &str = "birefnet-lite-onnx";

/// In-flight background-removal job for one image. First downloads the model if
/// missing (best-effort progress), then runs inference on a worker thread.
struct BgRemoveJob {
    src: PathBuf,
    /// Set while the model is downloading on first use.
    download: Option<ai_models::DownloadHandle>,
    /// Set while inference runs; yields the written cutout path (or an error).
    rx: Option<std::sync::mpsc::Receiver<Result<PathBuf, String>>>,
    /// Status line shown in the floating overlay.
    status: String,
}

/// Spawn BiRefNet inference for `src` on a worker thread, returning the result
/// channel. The model path is resolved on the worker (it may have just landed).
fn spawn_bg_inference(src: PathBuf, ctx: egui::Context) -> std::sync::mpsc::Receiver<Result<PathBuf, String>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let res = match tagger::resolve(BG_FOLDER, "model.onnx") {
            Some(model) => bgremove::run_bgremove_job(model, src),
            None => Err("Background model file is missing".to_string()),
        };
        let _ = tx.send(res);
        ctx.request_repaint();
    });
    rx
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
struct ViewerApp {
    images: Vec<PathBuf>,
    /// The folder last opened via the folder button (the backup root). `None`
    /// when the list was built only from individually dropped files.
    current_folder: Option<PathBuf>,
    /// Last-seen value of `settings.enable_extended_formats`, so we can re-scan
    /// the folder when the user toggles AVIF/HEIC on or off and the new file
    /// types appear (or disappear) without needing to reopen the folder.
    last_extended_formats: bool,
    /// Last-applied colour theme, so we re-push the egui visuals only when the
    /// user actually changes it in the Appearance tab.
    last_theme: Theme,
    /// Last-applied Glass dark/light panel mode — a palette change that doesn't
    /// change `theme`, so it needs its own re-apply tracking.
    last_glass_light: bool,
    /// Last-applied media-type filter (Filter tab), so we re-filter the browser
    /// only when the user actually changes it.
    last_media_filter: left_panel_settings::MediaFilter,
    /// CACHED list of indexes mapping into `self.images` after the search string is applied
    filtered: Vec<usize>,
    selected: Option<usize>,
    search: String,
    /// Lowercased sidecar-tag text per image, validated by the sidecar's mtime,
    /// so the search can match tags without re-reading every .txt per keystroke.
    tag_search_cache: std::collections::HashMap<PathBuf, (Option<std::time::SystemTime>, String)>,
    /// Favorited ("hearted") files, tracked by content hash so they survive
    /// moves/renames. Shown as a heart badge on the browser thumbnails.
    favorites: favorites::Favorites,
    stats: top_bar::SystemStats,
    /// FTP/FTPS remote-browser state (Settings → FTP/FTPS).
    ftp: ftp::FtpState,
    /// Local AI model state (Settings → AI Model): setup download + the
    /// resident Gemma 4 inference worker.
    llm: llm::LlmState,

    // Panel States
    right_state: right_details::RightPanelState,
    settings: settings::Settings,
    /// Deep Scan ("Find Issues") window state.
    scan: scan::ScanState,
    /// Gallery-view image detail popup.
    detail_popup: gallery_detail::DetailPopup,
    /// Startup splash (the cursive "Clarity TagFlow" write-on).
    splash: splash::Splash,
    /// While the Generate (Flux) view is open, the browser/viewer show the session's
    /// generated images; these remember the real folder list to restore on exit.
    flux_active: bool,
    images_backup: Option<(Vec<PathBuf>, Option<usize>)>,
    /// (is_zimage, count) of the gen list currently shown, so switching between the
    /// Flux/Z-Image tabs or a new render refreshes the browser.
    flux_sig: (u8, usize),
    /// The "Create Backup" dialog (top bar).
    backup: backup::BackupState,
    /// Tag Manager view state, shown in the right panel when selected from its
    /// menu dropdown. In-memory only for now — not persisted or wired into
    /// tagging behaviour yet.
    tag_manager: tag_manager::TagManagerState,

    /// Small thumbnails for the left browser (many, downscaled hard).
    thumbs: image_cache::ImageCache,
    /// Larger images for the centre viewer (few, near-full resolution).
    viewer: image_cache::ImageCache,
    /// Poster-frame thumbnails for video files in the left browser.
    video_thumbs: video::VideoThumbs,
    /// Auto-playing muted previews on visible video tiles (the "Video thumbnail
    /// play" setting).
    video_previews: video::VideoPreviews,

    /// Embedded video player for the current selection (only ever `Some` when
    /// built with `--features vlc` and VLC starts successfully).
    video_player: Option<video::VideoPlayer>,
    /// Path the current `video_player` was started for, so we only (re)start it
    /// when the selection actually changes — and don't retry on failure.
    last_video_path: Option<PathBuf>,

    /// Zoom + pan state for the centre image viewer (resets per selection).
    zoom: zoom::ZoomState,

    /// Interactive 3D viewer shown in the centre when the Pixal3D view is active
    /// (displays the generated GLB instead of the selected image).
    #[cfg(not(target_os = "macos"))]
    scene3d: scene3d::Scene3D,

    /// In-flight background-removal job (right-click → Remove Background).
    bg_job: Option<BgRemoveJob>,
    /// Transient status/result message for background removal (text, hide-time).
    bg_toast: Option<(String, f64)>,
    /// Screen rect of the centre (image) panel, captured each frame so overlays
    /// can be positioned over the image rather than the whole window.
    last_center_rect: Option<egui::Rect>,
    /// App + ComfyUI update checker (drives the Updates tab and the gear's red dot).
    update: update::UpdateState,
}

impl Default for ViewerApp {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            current_folder: None,
            last_extended_formats: settings::Settings::default().enable_extended_formats,
            last_theme: Theme::default(),
            last_glass_light: false,
            last_media_filter: left_panel_settings::MediaFilter::default(),
            filtered: Vec::new(),
            selected: None,
            search: String::new(),
            tag_search_cache: std::collections::HashMap::new(),
            favorites: favorites::Favorites::load(),
            stats: top_bar::SystemStats::default(),
            ftp: ftp::FtpState::default(),
            llm: llm::LlmState::default(),
            right_state: right_details::RightPanelState::default(),
            settings: settings::Settings::default(),
            scan: scan::ScanState::default(),
            detail_popup: gallery_detail::DetailPopup::default(),
            splash: splash::Splash::default(),
            flux_active: false,
            images_backup: None,
            flux_sig: (0, 0),
            backup: backup::BackupState::default(),
            tag_manager: tag_manager::TagManagerState::default(),
            // Separate large-decode gates: the viewer gets a dedicated permit so
            // the clicked image always decodes immediately (priority); the browser
            // gets its own pair so it never blocks — or is blocked by — the viewer.
            thumbs: image_cache::ImageCache::new(THUMB_MAX_EDGE, 400, false, 2),
            viewer: image_cache::ImageCache::new(2048, 8, true, 1),
            video_thumbs: video::VideoThumbs::new(),
            video_previews: video::VideoPreviews::new(),
            video_player: None,
            last_video_path: None,
            zoom: zoom::ZoomState::default(),
            #[cfg(not(target_os = "macos"))]
            scene3d: scene3d::Scene3D::new(),
            bg_job: None,
            bg_toast: None,
            last_center_rect: None,
            update: update::UpdateState::default(),
        }
    }
}

impl ViewerApp {
    /// Update the cached list of filtered indices based on the search string.
    /// The query matches the file name OR the sidecar .txt tags (cached).
    fn update_filtered(&mut self) {
        use left_panel_settings::MediaFilter;
        let query = self.search.trim().to_lowercase();
        let filter = self.settings.media_filter;
        let mut filtered = Vec::with_capacity(self.images.len());
        for i in 0..self.images.len() {
            // Clone the path so the favorite look-up (which needs `&mut favorites`)
            // doesn't clash with borrowing `self.images`.
            let path = self.images[i].clone();
            if !query.is_empty()
                && !file_name(&path).to_lowercase().contains(&query)
                && !self.tags_match(&path, &query)
            {
                continue;
            }
            let keep = match filter {
                MediaFilter::All => true,
                // `is_image` counts GIFs as images, so exclude them here to keep
                // the "Images" and "GIFs" buckets distinct (matching the Java filter).
                MediaFilter::Images => is_image(&path) && !is_gif(&path),
                MediaFilter::Videos => is_video(&path),
                MediaFilter::Gifs => is_gif(&path),
                MediaFilter::Favorites => self.favorites.is_favorite(&path),
            };
            if keep {
                filtered.push(i);
            }
        }
        self.filtered = filtered;
    }

    /// Whether `path`'s sidecar tags contain `query` (lowercase). The tag text
    /// is cached and only re-read when the sidecar's mtime changes, so typing
    /// in the search box costs one `stat` per image, not a file read.
    fn tags_match(&mut self, path: &std::path::Path, query: &str) -> bool {
        let txt = right_details::sidecar_txt(path);
        let mtime = std::fs::metadata(&txt).and_then(|m| m.modified()).ok();
        let entry = self
            .tag_search_cache
            .entry(path.to_path_buf())
            .or_insert_with(|| (None, String::new()));
        if entry.0 != mtime {
            entry.0 = mtime;
            entry.1 = std::fs::read_to_string(&txt).unwrap_or_default().to_lowercase();
        }
        entry.1.contains(query)
    }

    /// Pick a folder and show every image in it (replacing the current list).
    fn open_dialog(&mut self) {
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            self.load_folder(&dir);
        }
    }

    /// Replace the browser contents with all images found directly in `dir`.
    fn load_folder(&mut self, dir: &std::path::Path) {
        self.images = images_in_dir(dir);
        self.current_folder = Some(dir.to_path_buf());
        self.selected = if self.images.is_empty() { None } else { Some(0) };
        self.tag_search_cache.clear(); // tags belong to the previous folder
        self.update_filtered();
    }

    /// Open the Create Backup dialog for the current folder and its media. The
    /// backup root is the last-opened folder, falling back to the first image's
    /// parent (e.g. when files were dropped in individually).
    fn start_backup(&mut self) {
        let source = self
            .current_folder
            .clone()
            .or_else(|| self.images.first().and_then(|p| p.parent().map(|d| d.to_path_buf())))
            .unwrap_or_default();
        self.backup.open(source, self.images.clone());
    }

    fn add_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        let first_new = self.images.len();
        for path in paths {
            if !self.images.contains(&path) {
                self.images.push(path);
            }
        }
        if self.images.len() > first_new {
            self.selected = Some(first_new);
            self.update_filtered();
        } else if self.selected.is_none() && !self.images.is_empty() {
            self.selected = Some(0);
            self.update_filtered();
        }
    }

    /// Save a cropped copy of `src` to `<name>_crop.png` next to the original
    /// (the original is left untouched), then add it to the browser and select it.
    /// `frac` is the crop region as fractions (0..1) of the full image. Re-reads the
    /// full-resolution original from disk so the crop isn't limited to the
    /// (possibly downscaled) on-screen texture. A port of ViewerPanel.cropTo().
    fn crop_current(&mut self, src: &std::path::Path, frac: zoom::CropFraction) {
        // `decode_full_rgba` covers common formats via `image::open` plus the
        // extended formats (AVIF/HEIC/RAW) through our own decoder, so cropping
        // works on everything the viewer can display.
        let Some(full) = decode_full_rgba(src).map(image::DynamicImage::ImageRgba8) else {
            return;
        };
        let (iw, ih) = (full.width(), full.height());
        if iw == 0 || ih == 0 {
            return;
        }
        let (iwf, ihf) = (iw as f32, ih as f32);
        let cx = (frac.x * iwf).round().clamp(0.0, iwf - 1.0) as u32;
        let cy = (frac.y * ihf).round().clamp(0.0, ihf - 1.0) as u32;
        let cw = (frac.w * iwf).round().clamp(1.0, iwf - cx as f32) as u32;
        let ch = (frac.h * ihf).round().clamp(1.0, ihf - cy as f32) as u32;

        let cropped = full.crop_imm(cx, cy, cw, ch);
        let dest = crop_destination(src);
        if cropped.save(&dest).is_ok() {
            // Place the crop directly beneath the image it came from in the
            // browser (rather than at the end of the list), then select it.
            if self.images.contains(&dest) {
                return;
            }
            let insert_at = self
                .images
                .iter()
                .position(|p| p == src)
                .map(|i| i + 1)
                .unwrap_or(self.images.len());
            self.images.insert(insert_at, dest);
            self.selected = Some(insert_at);
            self.update_filtered();
        }
    }

    /// Copy `src`'s pixels to the system clipboard. Re-reads the full-resolution
    /// original from disk (like cropping) so the clipboard gets full quality, not
    /// the downscaled on-screen texture. Extended formats (AVIF/HEIC/RAW) are
    /// routed to our own decoder first — same extension check as the display path
    /// in image_cache — since `image::open` can't handle them (and would mis-read
    /// TIFF-based raws like DNG).
    fn copy_current(&mut self, src: &std::path::Path) {
        // Decoding the full-resolution original and having arboard convert it to
        // the clipboard's DIB/PNG formats is heavy — on a large image it takes
        // seconds. Do it on a background thread so the UI never freezes; the
        // clipboard is populated a moment later. (arboard sets image data
        // immediately, so a short-lived thread is fine.)
        let src = src.to_path_buf();
        std::thread::spawn(move || {
            let Some(rgba) = decode_full_rgba(&src) else { return };
            let (w, h) = (rgba.width() as usize, rgba.height() as usize);
            // Build the plain-DIB payload first; the RGBA buffer then moves
            // into arboard.
            #[cfg(target_os = "windows")]
            let dib = Self::rgba_to_cf_dib(rgba.width(), rgba.height(), rgba.as_raw());
            let img = arboard::ImageData {
                width: w,
                height: h,
                bytes: std::borrow::Cow::Owned(rgba.into_raw()),
            };
            // arboard writes the "PNG" + CF_DIBV5 clipboard formats…
            if let Ok(mut clip) = arboard::Clipboard::new() {
                let _ = clip.set_image(img);
            }
            // …but Chromium-based browsers only ever request the legacy CF_DIB
            // when pasting an image (they ignore DIBV5 and PNG — see
            // PhotoDemon #343), so pasting into Chrome/Edge (e.g. a Gemini
            // upload box) found nothing. Append a plain DIB to what arboard
            // just set, without clearing it.
            #[cfg(target_os = "windows")]
            if let Ok(_clip) = clipboard_win::Clipboard::new_attempts(10) {
                let _ = clipboard_win::raw::set_without_clear(clipboard_win::formats::CF_DIB, &dib);
            }
        });
    }

    /// Encode RGBA pixels as a clipboard `CF_DIB` payload: a 40-byte
    /// BITMAPINFOHEADER (BI_RGB, 32 bpp) followed by bottom-up BGRX rows.
    /// Alpha is composited against white — plain-DIB consumers rarely honour
    /// an alpha channel, and a white backdrop matches what image editors put
    /// in the legacy format.
    #[cfg(target_os = "windows")]
    fn rgba_to_cf_dib(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
        let row = w as usize * 4;
        let mut dib = Vec::with_capacity(40 + rgba.len());
        dib.extend_from_slice(&40u32.to_le_bytes()); // biSize
        dib.extend_from_slice(&(w as i32).to_le_bytes()); // biWidth
        dib.extend_from_slice(&(h as i32).to_le_bytes()); // biHeight > 0: bottom-up
        dib.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
        dib.extend_from_slice(&32u16.to_le_bytes()); // biBitCount
        dib.extend_from_slice(&0u32.to_le_bytes()); // biCompression = BI_RGB
        dib.extend_from_slice(&(rgba.len() as u32).to_le_bytes()); // biSizeImage
        dib.extend_from_slice(&[0u8; 16]); // ppm + palette fields, all zero
        for y in (0..h as usize).rev() {
            for px in rgba[y * row..(y + 1) * row].chunks_exact(4) {
                let a = px[3] as u32;
                let blend = |c: u8| ((c as u32 * a + 255 * (255 - a)) / 255) as u8;
                dib.extend_from_slice(&[blend(px[2]), blend(px[1]), blend(px[0]), 0xFF]);
            }
        }
        dib
    }

    /// Start removing the background of `src` (right-click action). Downloads the
    /// BiRefNet model first if it isn't installed. One job at a time.
    fn start_bgremove(&mut self, src: &std::path::Path, ctx: &egui::Context) {
        if self.bg_job.is_some() {
            return;
        }
        let src = src.to_path_buf();
        let mut job = BgRemoveJob { src: src.clone(), download: None, rx: None, status: String::new() };
        if tagger::resolve(BG_FOLDER, "model.onnx").is_none() {
            job.download = ai_models::start_model_download(BG_FOLDER);
            if job.download.is_some() {
                job.status = "Downloading background model…".to_string();
            } else {
                self.bg_toast = Some(("Background model unavailable".to_string(), ctx.input(|i| i.time) + 5.0));
                return;
            }
        } else {
            job.status = "Removing background…".to_string();
            job.rx = Some(spawn_bg_inference(src, ctx.clone()));
        }
        self.bg_job = Some(job);
    }

    /// Advance the background-removal job each frame: poll the model download,
    /// kick inference once it's ready, and handle the result. Also expires the
    /// transient toast.
    fn drive_bg_job(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        if let Some((_, until)) = &self.bg_toast {
            if now >= *until {
                self.bg_toast = None;
            }
        }

        // Result of this frame's polling, applied after the &mut borrow ends.
        let mut done: Option<Result<PathBuf, String>> = None;
        if let Some(job) = &mut self.bg_job {
            if let Some(dl) = &job.download {
                if dl.done() {
                    if dl.ok() {
                        job.download = None;
                        job.status = "Removing background…".to_string();
                        job.rx = Some(spawn_bg_inference(job.src.clone(), ctx.clone()));
                    } else {
                        done = Some(Err(format!(
                            "Model download failed: {}",
                            dl.error().unwrap_or_else(|| "unknown error".into())
                        )));
                    }
                } else {
                    job.status = format!("Downloading background model… {}%", dl.pct());
                }
            } else if let Some(rx) = &job.rx {
                match rx.try_recv() {
                    Ok(r) => done = Some(r),
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        done = Some(Err("Background worker stopped".to_string()));
                    }
                }
            }
        }

        if let Some(result) = done {
            let src = self.bg_job.as_ref().map(|j| j.src.clone());
            self.bg_job = None;
            match result {
                Ok(dest) => {
                    if let Some(src) = src {
                        self.insert_derived(&src, dest);
                    }
                    self.bg_toast = Some(("Background removed ✓".to_string(), now + 3.0));
                }
                Err(e) => self.bg_toast = Some((format!("Background removal failed — {e}"), now + 6.0)),
            }
        }

        if self.bg_job.is_some() {
            ctx.request_repaint();
        }
    }

    /// Insert a derived image (`dest`) just beneath its source in the browser and
    /// select it; if it's already listed, just select it. Mirrors `crop_current`.
    fn insert_derived(&mut self, src: &std::path::Path, dest: PathBuf) {
        if let Some(i) = self.images.iter().position(|p| p == &dest) {
            self.selected = Some(i);
            self.update_filtered();
            return;
        }
        let at = self
            .images
            .iter()
            .position(|p| p == src)
            .map(|i| i + 1)
            .unwrap_or(self.images.len());
        self.images.insert(at, dest);
        self.selected = Some(at);
        self.update_filtered();
    }

    /// Floating top-centre overlay showing background-removal progress / result.
    fn paint_bg_status(&self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        let (text, busy) = if let Some(job) = &self.bg_job {
            (job.status.clone(), true)
        } else if let Some((msg, until)) = &self.bg_toast {
            if now < *until { (msg.clone(), false) } else { return }
        } else {
            return;
        };
        let mut area = egui::Area::new(egui::Id::new("bg_status_overlay")).interactable(false);
        // Centre over the image panel if we know its rect; else fall back to the
        // window centre.
        area = match self.last_center_rect {
            Some(rect) => area.pivot(egui::Align2::CENTER_CENTER).fixed_pos(rect.center()),
            None => area.anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0)),
        };
        area
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if busy {
                            ui.add(egui::Spinner::new().size(16.0));
                        }
                        ui.label(text);
                    });
                });
            });
    }

    fn step_selection(&mut self, delta: i32) {
        if self.images.is_empty() {
            return;
        }
        let current = self.selected.unwrap_or(0) as i32;
        let next = (current + delta).rem_euclid(self.images.len() as i32);
        self.selected = Some(next as usize);
    }

    /// Delete the selected image and its `.txt` sidecar, then fix up the list.
    /// While the Generate (Flux) view is open, show the session's generated images
    /// in the browser + viewer; restore the real folder list when it closes.
    fn sync_flux_browser(&mut self, in_flux: bool) {
        #[cfg(target_os = "macos")]
        {
            let _ = in_flux;
        }
        #[cfg(not(target_os = "macos"))]
        {
            if in_flux {
                // Which generation tab is active picks the source list (0 = Flux,
                // 1 = Z-Image, 2 = LTX, 3 = Wan, 4 = SDXL); the index is folded into
                // `sig` so a tab switch refreshes the browser.
                let (tab, gen_list) = match self.right_state.view {
                    right_details::RightView::ZImage => (1u8, self.right_state.zimage.gen_images().to_vec()),
                    right_details::RightView::Ltx => (2u8, self.right_state.ltx.gen_images().to_vec()),
                    right_details::RightView::Wan => (3u8, self.right_state.wan.gen_images().to_vec()),
                    right_details::RightView::Sdxl => (4u8, self.right_state.sdxl.gen_images().to_vec()),
                    right_details::RightView::Anima => (5u8, self.right_state.anima.gen_images().to_vec()),
                    right_details::RightView::Krea2 => (6u8, self.right_state.krea2.gen_images().to_vec()),
                    _ => (0u8, self.right_state.generate.gen_images().to_vec()),
                };
                // The LTX and Wan Directors are image-to-video, so they keep the
                // loaded input folder visible in the browser alongside their
                // generated videos (appended below) — you can pick another input
                // image and make more videos without reopening the folder. The
                // other (text-to-image) views just show their generated results.
                let combine = matches!(
                    self.right_state.view,
                    right_details::RightView::Ltx | right_details::RightView::Wan
                );
                let sig = (tab, gen_list.len());
                if !self.flux_active {
                    // Entering: stash the folder list.
                    self.images_backup = Some((std::mem::take(&mut self.images), self.selected.take()));
                    if combine {
                        // Show folder images, then the generated videos under them;
                        // keep the input image the user already had selected.
                        let orig = self.images_backup.as_ref().map(|(v, _)| v.clone()).unwrap_or_default();
                        let keep = self.images_backup.as_ref().and_then(|(_, s)| *s);
                        self.images = orig;
                        self.images.extend(gen_list);
                        self.selected = keep;
                    } else {
                        self.images = gen_list;
                        self.selected = if self.images.is_empty() { None } else { Some(self.images.len() - 1) };
                    }
                    self.update_filtered();
                    self.flux_active = true;
                    self.flux_sig = sig;
                } else if self.flux_sig != sig {
                    // Switched gen tab or a new output arrived — refresh + select newest.
                    if combine {
                        let orig = self.images_backup.as_ref().map(|(v, _)| v.clone()).unwrap_or_default();
                        let orig_len = orig.len();
                        self.images = orig;
                        self.images.extend(gen_list);
                        // A generated video is present → select the newest (at the
                        // end); otherwise keep whatever input image is selected.
                        if self.images.len() > orig_len {
                            self.selected = Some(self.images.len() - 1);
                        }
                    } else {
                        self.images = gen_list;
                        self.selected = if self.images.is_empty() { None } else { Some(self.images.len() - 1) };
                    }
                    self.update_filtered();
                    self.flux_sig = sig;
                }
            } else if self.flux_active {
                // Leaving: restore the real folder list.
                if let Some((imgs, sel)) = self.images_backup.take() {
                    self.images = imgs;
                    self.selected = sel;
                    self.update_filtered();
                }
                self.flux_active = false;
            }
        }
    }

    /// Move the selected image (and its `.txt` sidecar) to a folder the user picks,
    /// then drop it from the list. Shared by the right panel and the gallery popup.
    fn move_selected(&mut self) {
        let Some(idx) = self.selected else { return };
        let Some(target_dir) = rfd::FileDialog::new().pick_folder() else { return };
        let img_path = self.images[idx].clone();
        let txt_path = right_details::sidecar_txt(&img_path);

        if let Some(file_name) = img_path.file_name() {
            let _ = std::fs::rename(&img_path, target_dir.join(file_name));
        }
        if txt_path.exists() {
            if let Some(txt_name) = txt_path.file_name() {
                let _ = std::fs::rename(&txt_path, target_dir.join(txt_name));
            }
        }

        self.remove_image_at(idx);
    }

    /// Remove image `idx`, re-filter the browser, and clamp the selection to a
    /// still-valid index (or clear it when the list is now empty).
    fn remove_image_at(&mut self, idx: usize) {
        self.images.remove(idx);
        self.update_filtered();
        self.selected = if self.images.is_empty() {
            None
        } else {
            Some(idx.min(self.images.len().saturating_sub(1)))
        };
    }

    /// Apply a viewer right-click action (favorite / crop / copy / bg-remove) to
    /// `path`. Shared by the centre viewer and the gallery-detail popup.
    fn handle_viewer_action(&mut self, action: zoom::ViewerAction, path: &std::path::Path, ctx: &egui::Context) {
        match action {
            zoom::ViewerAction::ToggleFavorite => {
                self.favorites.toggle(path);
            }
            zoom::ViewerAction::Crop(frac) => self.crop_current(path, frac),
            zoom::ViewerAction::CopyImage => self.copy_current(path),
            zoom::ViewerAction::RemoveBackground => self.start_bgremove(path, ctx),
            zoom::ViewerAction::None => {}
        }
    }

    /// Re-scan the current folder and try to keep the same image selected. Used
    /// after the extended-format toggle changes and after a Deep Scan moves files.
    fn rescan_current_folder(&mut self) {
        if let Some(dir) = self.current_folder.clone() {
            let keep = self.selected.and_then(|i| self.images.get(i).cloned());
            self.images = images_in_dir(&dir);
            // Try to keep the same image selected across the re-scan.
            self.selected = keep
                .and_then(|p| self.images.iter().position(|q| *q == p))
                .or(if self.images.is_empty() { None } else { Some(0) });
            self.update_filtered();
        }
    }

    fn delete_selected(&mut self) {
        let Some(idx) = self.selected else { return };
        let img_path = self.images[idx].clone();
        let txt_path = right_details::sidecar_txt(&img_path);

        // Stop any embedded player so VLC releases the file before we delete it.
        self.video_player = None;
        self.last_video_path = None;

        let _ = std::fs::remove_file(&img_path);
        if txt_path.exists() {
            let _ = std::fs::remove_file(&txt_path);
        }

        self.remove_image_at(idx);
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if dropped.is_empty() {
            return;
        }
        // A drop onto a generator's prompt box is consumed by that view (it
        // imports the file's metadata + installs the workflow's custom nodes),
        // so don't also add the file to the gallery. (The generate module is
        // compiled out on macOS, so there's no prompt box to claim drops there.)
        #[cfg(not(target_os = "macos"))]
        {
            let drop_pos = ctx.input(|i| i.pointer.interact_pos().or(i.pointer.latest_pos()));
            if crate::generate::generator_claims_drop(ctx, drop_pos) {
                return;
            }
        }
        // Likewise, a drop onto the AI Chat's input card attaches the image
        // to the draft there (ai_chat.rs consumes it), not to the gallery.
        if self.settings.ai_chat && ai_chat::claims_drop(&self.llm, ctx) {
            return;
        }
        // A dropped folder loads all media inside it; dropped files are added directly.
        let mut to_add = Vec::new();
        for p in dropped {
            if p.is_dir() {
                to_add.extend(images_in_dir(&p));
            } else if is_media(&p) {
                to_add.push(p);
            }
        }
        if !to_add.is_empty() {
            self.add_paths(to_add);
        }
    }

    // ---- Center: the image display ------------------------------------
    fn center(&mut self, ui: &mut egui::Ui) {
        // Remember the centre-panel rect so overlays (e.g. the background-removal
        // loader) can be centred over the image rather than the whole window.
        self.last_center_rect = Some(ui.available_rect_before_wrap());

        // The centre viewer only actively PLAYS video while the right panel is on a
        // media-focused view: Details & Actions, Civitai Resources, or the LTX / Wan
        // Directors (which work with videos). On any other view (Generate,
        // Downloader, Tag Manager, Pixal3D, …) the player is released so a clip
        // doesn't keep decoding — and playing audio — in the background; the viewer
        // shows the still poster instead.
        let plays_media = matches!(
            self.right_state.view,
            right_details::RightView::Details | right_details::RightView::Civitai
        ) || {
            #[cfg(not(target_os = "macos"))]
            {
                matches!(
                    self.right_state.view,
                    right_details::RightView::Ltx | right_details::RightView::Wan
                )
            }
            #[cfg(target_os = "macos")]
            {
                false
            }
        };
        if !plays_media {
            self.video_player = None;
            self.last_video_path = None;
        }

        egui::CentralPanel::default()
            // Match the side panels' margins (top: 0) so the viewer rises to the
            // top bar and is the same height as the left/right panels.
            .frame(egui::Frame::new().fill(BG()).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
            .show_inside(ui, |ui| {
                // When the Pixal3D view is active, the centre shows the generated
                // 3D model (orbit viewer) instead of the selected image/video.
                #[cfg(not(target_os = "macos"))]
                if self.right_state.view == right_details::RightView::Pixal3D {
                    let glb = self.right_state.pixal3d.last_glb.clone();
                    self.scene3d.show(ui, glb.as_deref());
                    return;
                }

                let Some(idx) = self.selected else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Open a folder to get started\n\nClick the folder button, or drag a folder or images here",
                            )
                                .size(18.0)
                                .color(MUTED()),
                        );
                    });
                    return;
                };

                let now = ui.input(|i| i.time);
                let path = self.images[idx].clone();

                // The LTX / Wan Directors play videos but leave the centre blank for
                // still images — clicking through images doesn't load or show them.
                // Release any player so a clip stops when you move onto an image.
                #[cfg(not(target_os = "macos"))]
                if matches!(
                    self.right_state.view,
                    right_details::RightView::Ltx | right_details::RightView::Wan
                ) && !is_video(&path)
                {
                    self.video_player = None;
                    self.last_video_path = None;
                    return;
                }

                // Videos play in-app via libVLC. Only touch the player when a
                // libVLC runtime is actually available — otherwise we'd trigger the
                // delay-loaded DLL and crash; the notice below offers to install it.
                if is_video(&path) {
                    // On a non-media right-panel view, don't play — show the still
                    // poster frame (the player was already released above).
                    if !plays_media {
                        match self.video_thumbs.request(&path, ui.ctx()) {
                            Some(tex) => show_fitted(ui, &tex, false),
                            None => {
                                ui.centered_and_justified(|ui| {
                                    ui.add(
                                        egui::Image::new(egui::include_image!("../icons/video.svg"))
                                            .fit_to_exact_size(egui::vec2(64.0, 64.0))
                                            .tint(MUTED()),
                                    );
                                });
                            }
                        }
                        return;
                    }

                    let support = video::support();

                    if matches!(support, video::VideoSupport::Available) {
                        // Push the current loop preference so the player picks it up
                        // when it (re)starts this clip.
                        video::set_loop(self.settings.loop_video);
                        if self.last_video_path.as_deref() != Some(path.as_path()) {
                            self.last_video_path = Some(path.clone());
                            self.video_player = video::VideoPlayer::start(&path, ui.ctx());
                        }
                        if let Some(player) = &mut self.video_player {
                            match player.frame(ui.ctx()) {
                                Some(tex) => show_fitted(ui, &tex, false),
                                None => {
                                    ui.centered_and_justified(|ui| {
                                        ui.add(egui::Spinner::new().size(48.0).color(MUTED()));
                                    });
                                }
                            }
                            // Keep pulling frames, but cap the UI to ~60 Hz instead
                            // of repainting as fast as the monitor allows (e.g. 144
                            // Hz) — a full-app relayout every refresh steals CPU from
                            // decoding. New frames still wake us instantly via the
                            // player's display callback (request_repaint).
                            ui.ctx().request_repaint_after(Duration::from_millis(16));
                            return;
                        }
                    } else {
                        // No runtime (or unsupported build): drop any stale player so
                        // that, once VLC is installed, reselecting the clip restarts.
                        self.video_player = None;
                        self.last_video_path = None;
                    }

                    // No running player — show the appropriate notice (couldn't
                    // start / install VLC / unsupported build).
                    video_notice(ui, &path, support);
                    return;
                }

                // The selection isn't a video — release any running player so VLC
                // stops and frees the file.
                self.video_player = None;
                self.last_video_path = None;

                match self.viewer.request(&path, now) {
                    image_cache::Cached::Ready(tex) => {
                        let is_fav = self.favorites.is_favorite(&path);
                        let action = self.zoom.show(ui, &tex, &path, is_fav);
                        self.handle_viewer_action(action, &path, ui.ctx());
                    }
                    image_cache::Cached::Animated(frame) => {
                        show_fitted(ui, &frame, false);
                        // Keep playing the GIF, but cap to ~60 Hz instead of
                        // repainting (and relaying out the whole app) as fast as
                        // the monitor allows — GIFs top out at 50 fps, so this is
                        // smooth while leaving CPU for decoding.
                        ui.ctx().request_repaint_after(Duration::from_millis(16));
                    }
                    image_cache::Cached::Failed => {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("Couldn't load image").color(MUTED()));
                        });
                    }
                    image_cache::Cached::Loading => {
                        match self.thumbs.request(&path, now) {
                            image_cache::Cached::Ready(thumb)
                            | image_cache::Cached::Animated(thumb) => {
                                show_fitted(ui, &thumb, true);
                            }
                            _ => {
                                ui.centered_and_justified(|ui| {
                                    ui.add(egui::Spinner::new().size(48.0).color(MUTED()));
                                });
                            }
                        }
                        ui.ctx().request_repaint();
                    }
                }

                // Prefetch a few neighbours (configurable in Settings). A larger
                // radius means one selection kicks off several full-res decodes at
                // once, which is wasteful (and a freeze risk) for big 4K–8K images.
                let prefetch_radius = self.settings.prefetch_radius;
                for offset in 1..=prefetch_radius {
                    let forward_idx = (idx + offset).min(self.images.len().saturating_sub(1));
                    if forward_idx != idx && !is_video(&self.images[forward_idx]) {
                        let _ = self.viewer.request(&self.images[forward_idx], now);
                    }

                    let backward_idx = idx.saturating_sub(offset);
                    if backward_idx != idx && !is_video(&self.images[backward_idx]) {
                        let _ = self.viewer.request(&self.images[backward_idx], now);
                    }
                }
            });
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // While the splash is writing its wordmark, render ONLY the splash —
        // the panels (and the folder/thumbnail work they kick off) build after
        // it, so launch shows the text before the app UI exists. The fade-out
        // phase falls through and plays over the real UI.
        if self.splash.covers_ui(ui.ctx()) {
            self.splash.show(ui.ctx());
            return;
        }

        self.handle_dropped_files(ui.ctx());
        // Kick off the launch update-check and drain its workers (badge + Updates tab).
        self.update.tick(ui.ctx());
        // Advance any in-flight background-removal job (download → inference → result).
        self.drive_bg_job(ui.ctx());
        self.paint_bg_status(ui.ctx());
        // Only sample CPU/RAM when the top-bar graphs are actually shown.
        if self.settings.show_stats {
            self.stats.update();
        }

        // Mirror the movable-popups preference into the process-wide flag the popup
        // builders read (via PopupPlacement), so it isn't threaded through every
        // popup call site. Cheap; done each frame so the toggle applies live.
        MOVABLE_POPUPS.store(self.settings.movable_popups, std::sync::atomic::Ordering::Relaxed);

        // Apply the HD-thumbnail setting (cheap; only clears the cache on change).
        // Video poster frames follow the same setting so they're HD in step with
        // image tiles.
        let thumb_edge = if self.settings.hd_thumbnails {
            THUMB_MAX_EDGE_HD
        } else {
            THUMB_MAX_EDGE
        };
        self.thumbs.set_max_edge(thumb_edge);
        self.video_thumbs.set_max_edge(thumb_edge);

        self.thumbs.begin_frame(ui.ctx());
        self.viewer.begin_frame(ui.ctx());

        // Push the Glass theme's user-configurable background (colour + backdrop)
        // and its dark/light panel mode so the pickers update live; cheap, so
        // done every frame.
        set_glass_config(self.settings.glass_bg, self.settings.glass_backdrop, self.settings.glass_light);

        // Paint the theme's full-window background (the Space theme's animated
        // starfield, the Glass theme's configured backdrop) on the bottom layer,
        // beneath every panel. No-op otherwise.
        paint_background(ui.ctx());

        // --- Toggle fullscreen on F12 ---
        if ui.input(|i| i.key_pressed(egui::Key::F12)) {
            let is_fullscreen = ui.input(|i| i.viewport().fullscreen.unwrap_or(false));
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
        }

        // Arrow keys cycle through the opened images — but NOT while a text field
        // (tags box, search, blacklist, …) has keyboard focus, where the arrows
        // must move the text cursor instead.
        let typing = ui.ctx().egui_wants_keyboard_input();
        let delta = ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::ArrowDown) {
                1
            } else if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::ArrowUp) {
                -1
            } else {
                0
            }
        });
        if delta != 0 && !typing {
            self.step_selection(delta);
            // In the Gallery layout the arrow keys also page the open detail
            // popup to the newly selected image (it loads its content once on
            // open, so it must be re-pointed explicitly).
            if self.settings.layout == settings::Layout::Gallery && self.detail_popup.open {
                if let Some(i) = self.selected {
                    self.detail_popup.open_for(i, &self.images[i], ui.ctx());
                }
            }
        }

        let update_badge = self.update.badge(&self.settings);
        match top_bar::show(ui, &self.stats, self.settings.show_stats, update_badge, self.settings.ftp_enabled) {
            // In FTP mode the folder button opens the remote browser instead of
            // the local folder picker.
            top_bar::TopBarAction::OpenFolder => {
                if self.settings.ftp_enabled {
                    self.ftp.browser_open = !self.ftp.browser_open;
                } else {
                    self.open_dialog();
                }
            }
            top_bar::TopBarAction::OpenSettings => self.settings.open = !self.settings.open,
            top_bar::TopBarAction::CreateBackup => self.start_backup(),
            top_bar::TopBarAction::FindIssues(pos) => {
                self.scan
                    .open_with(self.current_folder.as_deref(), Some(pos));
            }
            top_bar::TopBarAction::None => {}
        }

        // While the Generate (Flux) view is open, swap the browser/viewer over to
        // the session's generated images (restored on exit).
        #[cfg(not(target_os = "macos"))]
        let in_flux = self.settings.layout == settings::Layout::Panels
            && matches!(
                self.right_state.view,
                right_details::RightView::Generate
                    | right_details::RightView::ZImage
                    | right_details::RightView::Ltx
                    | right_details::RightView::Wan
                    | right_details::RightView::Sdxl
                    | right_details::RightView::Anima
                    | right_details::RightView::Krea2
            );
        #[cfg(target_os = "macos")]
        let in_flux = false;
        self.sync_flux_browser(in_flux);

        // Drive the auto-playing video-tile previews: push the setting and reset
        // the per-frame visibility marks before drawing any tiles. `end_frame`
        // (after the layout below) stops previews for tiles that scrolled away.
        self.video_previews.set_enabled(self.settings.video_thumbnail_play);
        self.video_previews.begin_frame();

        // AI Chat mode replaces the panels with a full-window chat with the
        // local model (the top bar stays). Takes precedence over the layouts.
        if self.settings.ai_chat {
            // Release the centre viewer's player — like the Gallery layout, a
            // clip selected in Panels would keep playing behind the chat.
            self.video_player = None;
            self.last_video_path = None;
            ai_chat::show(ui, &mut self.llm, &mut self.settings);
        } else
        // When the Gallery layout is active, replace the three panels with a
        // full-window masonry grid of the open folder's images.
        if self.settings.layout == settings::Layout::Gallery {
            // Release the centre viewer's player — without this, a clip selected
            // in the Panels layout keeps playing (audibly) behind the grid.
            self.video_player = None;
            self.last_video_path = None;
            // Skip the poster capture for the clip playing in the detail popup,
            // like the browser does for the centre player.
            self.video_thumbs.set_busy(self.detail_popup.playing_video());
            if let Some(i) = gallery::show(
                ui,
                &self.images,
                &self.filtered,
                self.selected,
                &mut self.thumbs,
                &mut self.video_thumbs,
                &mut self.video_previews,
                &mut self.favorites,
                self.settings.thumbnail_size,
            ) {
                // Select the clicked tile (accent ring + arrow-key anchor) and
                // open its detail popup.
                self.selected = Some(i);
                self.detail_popup.open_for(i, &self.images[i], ui.ctx());
            }
            // Floating search/filter pill (bottom-right corner, movable).
            if gallery::search_pill(ui.ctx(), &mut self.search, &mut self.settings) {
                self.update_filtered();
            }
        } else {
        // 1. Left Panel
        // Only skip the poster capture for the file that's actually playing, so
        // its frame isn't decoded twice at once; every other video still loads
        // its thumbnail. (Reflects last frame's player state; that's fine.)
        let busy_video = self.video_player.as_ref().and(self.last_video_path.as_deref());
        self.video_thumbs.set_busy(busy_video);
        let search_changed = left_browser::show(
            ui,
            &self.images,
            &self.filtered, // Pass the cached list
            &mut self.search,
            &mut self.selected,
            &mut self.thumbs,
            &mut self.video_thumbs,
            &mut self.video_previews,
            &mut self.favorites,
            self.settings.thumbnail_size,
            &mut self.settings.media_filter,
        );

        // Recompute the cached indices list if the user typed in the search box or
        // changed the media-type filter in the Filter Settings panel.
        if search_changed || self.settings.media_filter != self.last_media_filter {
            self.last_media_filter = self.settings.media_filter;
            self.update_filtered();
        }

        // Immediately unload thumbnails that scrolled out of view (kept only for
        // the viewport + prefetch margin); they re-decode when scrolled back.
        // (When disabled, the cache's LRU keeps recently-seen tiles instead.)
        if self.settings.unload_offscreen_thumbs {
            self.thumbs.retain_visible();
        }

        // 2. Right Panel (Evaluate Action Logic)
        let current_image_path = self.selected.map(|idx| self.images[idx].as_path());
        let action = right_details::show(
            ui,
            &mut self.right_state,
            current_image_path,
            &mut self.settings.confirm_before_delete,
            &mut self.tag_manager,
            &self.images,
        );

        // Remember the chosen AI model so it's restored next launch.
        if self.settings.last_ai_model != self.tag_manager.ai_model {
            self.settings.last_ai_model = self.tag_manager.ai_model.clone();
        }

        match action {
            // The right panel handles its own confirmation (gated by the setting),
            // so by the time we get a DeleteCurrent the delete is already confirmed.
            right_details::RightPanelAction::DeleteCurrent => self.delete_selected(),
            right_details::RightPanelAction::MoveCurrent => self.move_selected(),
            right_details::RightPanelAction::None => {}
        }

        // 3. Central Panel (Fills remaining space)
        self.center(ui);
        } // end of the classic Panels layout

        // Stop previews for any video tile that wasn't drawn this frame (scrolled
        // off-screen, or the other layout), freeing their decoders.
        self.video_previews.end_frame();

        // 4. Settings window (floats on top when opened from the gear).
        // Keep the global extended-formats flag in sync with the setting so the
        // free `is_image()` helper recognises (or ignores) .avif/.heic live.
        // Only relevant in builds with a decoder compiled in (the `avif` feature).
        #[cfg(feature = "avif")]
        EXTENDED_FORMATS_ON.store(
            self.settings.enable_extended_formats,
            std::sync::atomic::Ordering::Relaxed,
        );

        // If the AVIF/HEIC toggle changed, re-scan the open folder so those files
        // appear (or disappear) right away instead of only after a reopen.
        if self.settings.enable_extended_formats != self.last_extended_formats {
            self.last_extended_formats = self.settings.enable_extended_formats;
            self.rescan_current_folder();
        }

        // Keep the voice's design prompt in sync with the persisted setting
        // (edited in the AI Model tab; read at Listen time). Migrate the old
        // default, which used attributes OmniVoice rejects.
        if self.settings.ai_voice_style == "female, warm, natural, conversational" {
            self.settings.ai_voice_style = voice::DEFAULT_STYLE.to_string();
        }
        if self.llm.voice.style != self.settings.ai_voice_style {
            self.llm.voice.style = self.settings.ai_voice_style.clone();
        }
        if self.llm.voice.ref_audio != self.settings.ai_voice_ref_audio {
            self.llm.voice.ref_audio = self.settings.ai_voice_ref_audio.clone();
        }
        if self.llm.voice.ref_text != self.settings.ai_voice_ref_text {
            self.llm.voice.ref_text = self.settings.ai_voice_ref_text.clone();
        }
        self.llm.auto_speak = self.settings.ai_auto_speak;
        self.llm.params = self.settings.ai_gen;
        self.llm.set_model(self.settings.ai_gemma_model);
        settings::show(ui.ctx(), &mut self.settings, &mut self.update, &mut self.ftp, &mut self.llm);

        // The floating voice-sample recorder (its own always-on-top window,
        // usable while other apps are focused). A finished recording becomes
        // the cloning sample; the transcript still needs typing.
        voice::recorder_window(ui.ctx(), &mut self.llm.voice);
        if let Some(p) = self.llm.voice.rec.saved.take() {
            self.settings.ai_voice_ref_audio = p.to_string_lossy().to_string();
            self.settings.ai_voice_ref_text.clear();
            self.llm.voice.ref_audio = self.settings.ai_voice_ref_audio.clone();
            self.llm.voice.ref_text.clear();
        }

        // FTP remote browser (opened by the top bar's globe button in FTP mode).
        // A finished directory download loads its local cache like any folder.
        ftp::show_browser(ui.ctx(), &mut self.ftp, &self.settings);
        if let Some(dir) = self.ftp.take_loaded() {
            self.load_folder(&dir);
        }

        // Gallery detail popup (opened by clicking a gallery tile). Push the loop
        // preference first so a clip the popup starts picks it up (the centre
        // viewer only pushes it when it is itself playing a video).
        video::set_loop(self.settings.loop_video);
        match gallery_detail::show(
            ui.ctx(),
            &mut self.detail_popup,
            &mut self.viewer,
            &mut self.right_state.civitai,
            &mut self.favorites,
            &mut self.settings.confirm_before_delete,
        ) {
            gallery_detail::DetailAction::Move(i) => {
                self.selected = Some(i);
                self.move_selected();
            }
            gallery_detail::DetailAction::Delete(i) => {
                self.selected = Some(i);
                self.delete_selected();
            }
            // Viewer right-click actions, handled exactly like the centre viewer.
            gallery_detail::DetailAction::Viewer(va, path) => {
                self.handle_viewer_action(va, &path, ui.ctx());
            }
            gallery_detail::DetailAction::None => {}
        }

        // Both Civitai hosts (right panel + detail popup) have rendered by now —
        // if neither showed the view this frame, cancel any in-flight lookup so
        // it stops fetching resource info / preview thumbnails in the background.
        self.right_state.civitai.end_frame();

        // Deep Scan window. When a scan finishes it may have moved files out of
        // the current folder, so refresh the browser list once.
        scan::show(ui.ctx(), &mut self.scan);
        if self.scan.finished_tick {
            self.scan.finished_tick = false;
            self.rescan_current_folder();
        }

        // Apply a theme change from the Appearance tab live (only when it
        // actually changed, so we don't re-push visuals every frame). The Glass
        // dark/light switch changes the palette without changing `theme`, so it
        // re-applies the visuals too. CRITICAL: re-push the glass config FIRST —
        // the frame-start push used this frame's pre-settings value, and apply()
        // derives every widget colour from the palette it reads. Applying before
        // the flag lands bakes the stale palette into the visuals permanently
        // (e.g. dark glass stuck with the light theme's blue toggles).
        if self.settings.theme != self.last_theme || self.settings.glass_light != self.last_glass_light {
            self.last_theme = self.settings.theme;
            self.last_glass_light = self.settings.glass_light;
            set_glass_config(self.settings.glass_bg, self.settings.glass_backdrop, self.settings.glass_light);
            set(self.settings.theme);
            apply(ui.ctx());
            ui.ctx().request_repaint();
        }

        // 5. Backup dialog (floats on top when opened from the top bar).
        self.backup.show(ui.ctx());

        // Startup splash — drawn last so it covers the whole UI while the
        // "Clarity TagFlow" wordmark writes itself on.
        self.splash.show(ui.ctx());

        // Keep the live graphs animating without busy-looping.
        ui.ctx().request_repaint_after(Duration::from_millis(250));
    }

    /// Persist settings to eframe's storage (called periodically and on exit).
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, settings::STORAGE_KEY, &self.settings);
    }
}

// ---------------------------------------------------------------------------
// Movable popups
// ---------------------------------------------------------------------------

/// Process-wide mirror of `Settings::movable_popups`, refreshed each frame in
/// `ViewerApp::ui`. Popup builders in other modules read it through the
/// [`PopupPlacement`] trait, so the preference doesn't have to be threaded through
/// every call site.
pub(crate) static MOVABLE_POPUPS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Whether popups should be draggable and remember where the user left them.
pub(crate) fn movable_popups() -> bool {
    MOVABLE_POPUPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Positioning for the app's floating popups (Civitai settings, LoRA picker,
/// gallery detail, Find Issues). When movable popups are enabled the window is
/// draggable and egui persists its position across runs (eframe's egui-memory
/// storage); otherwise it's pinned to its original spot and can't be moved.
///
/// Modal-style dialogs (Settings, Backup, Confirm Delete) deliberately do NOT use
/// this — they stay centred and fixed.
pub(crate) trait PopupPlacement<'a> {
    /// Centre the popup (its original placement). Draggable + remembered when
    /// movable popups are on; pinned dead-centre when off.
    fn placed_centered(self, ctx: &egui::Context) -> Self;
    /// Place the popup's top-left at `top_left` (e.g. dropped from a button).
    /// Draggable from there + remembered when on; pinned there when off.
    fn placed_at(self, top_left: impl Into<egui::Pos2>) -> Self;
}

/// Constrain a movable popup so it can only be dragged by its top strip.
///
/// egui's window move-sense covers the WHOLE window, and a drag that starts on
/// a click-only widget (e.g. a frameless ✕ / ☰ icon button) falls through to
/// it — so a click that slipped by a pixel dragged the popup around. Call this
/// FIRST inside the window's content closure: it registers an invisible
/// drag-consuming shield over everything below `strip_h` (measured from the
/// content top), leaving only the header strip draggable. Widgets added after
/// this still win their own interactions (buttons, scroll areas, sliders, text
/// selection) because later widgets sit on top of the shield.
pub(crate) fn popup_drag_strip(ui: &mut egui::Ui, strip_h: f32) {
    if !movable_popups() {
        return; // pinned popups can't be dragged anyway
    }
    // Cover the window generously (the frame margins too); the interact rect is
    // clipped to the window's clip rect, so the overshoot is harmless.
    let mut rect = ui.max_rect().expand(64.0);
    rect.min.y = ui.max_rect().min.y + strip_h;
    ui.interact(rect, ui.id().with("popup_drag_shield"), egui::Sense::drag());
}

impl<'a> PopupPlacement<'a> for egui::Window<'a> {
    fn placed_centered(self, ctx: &egui::Context) -> Self {
        if movable_popups() {
            // No anchor (anchoring forces immovability); centre via pivot so the
            // first appearance matches the old CENTER_CENTER placement, then egui
            // remembers any drag.
            self.movable(true)
                .pivot(egui::Align2::CENTER_CENTER)
                .default_pos(ctx.content_rect().center())
        } else {
            self.movable(false).anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        }
    }
    fn placed_at(self, top_left: impl Into<egui::Pos2>) -> Self {
        let pos = top_left.into();
        if movable_popups() {
            self.movable(true).default_pos(pos)
        } else {
            self.fixed_pos(pos)
        }
    }
}

// ---------------------------------------------------------------------------
// Reusable painting helpers
// ---------------------------------------------------------------------------

/// Show a texture centred and scaled to fit the available space, aspect-preserved.
/// Centred placeholder for a video that isn't playing: the video glyph, the file
/// name, and a message that depends on why there's no player — playback failed,
/// VLC needs installing (with a button), or this build has no video backend.
pub(crate) fn video_notice(ui: &mut egui::Ui, path: &std::path::Path, support: video::VideoSupport) {
    ui.vertical_centered(|ui| {
        let avail_h = ui.available_height();
        ui.add_space((avail_h * 0.5 - 84.0).max(8.0));

        let icon = egui::include_image!("../icons/video.svg");
        ui.add(egui::Image::new(icon).fit_to_exact_size(egui::vec2(84.0, 84.0)).tint(MUTED()));
        ui.add_space(10.0);
        ui.label(egui::RichText::new(file_name(path)).color(TEXT()).strong().size(15.0));
        ui.add_space(8.0);

        match support {
            video::VideoSupport::Available => {
                ui.label(egui::RichText::new("Couldn't start video playback.").color(MUTED()).size(13.0));
            }
            video::VideoSupport::NeedsInstall => {
                ui.label(
                    egui::RichText::new("In-app video playback needs VLC, which isn't installed.")
                        .color(MUTED())
                        .size(13.0),
                );
                ui.add_space(10.0);
                let install = egui::Button::new(
                    egui::RichText::new("Install VLC").color(Color32::WHITE).strong(),
                )
                .fill(ACCENT1());
                if ui.add(install).clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(video::VLC_DOWNLOAD_URL));
                }
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("After installing, select the video again.")
                        .color(MUTED())
                        .size(11.0),
                );
            }
            video::VideoSupport::Unsupported => {
                ui.label(
                    egui::RichText::new("This build has no video player.").color(MUTED()).size(13.0),
                );
            }
        }
    });
}

pub(crate) fn show_fitted(ui: &mut egui::Ui, tex: &egui::TextureHandle, is_loading: bool) {
    let avail = ui.available_size();
    let tex_size = tex.size_vec2();
    let aspect = tex_size.y / tex_size.x.max(1.0);

    // Calculate exact dimensions needed to fill the space without breaking aspect ratio.
    let h_at_full_width = avail.x * aspect;
    let fit_size = if h_at_full_width <= avail.y {
        egui::vec2(avail.x, h_at_full_width)
    } else {
        egui::vec2(avail.y / aspect, avail.y)
    };

    ui.centered_and_justified(|ui| {
        let mut img = egui::Image::from_texture(tex)
            .fit_to_exact_size(fit_size)
            .corner_radius(CornerRadius::same(22));

        if is_loading {
            img = img.tint(Color32::from_gray(180));
        }

        let resp = ui.add(img);

        if is_loading {
            let spinner_rect = egui::Rect::from_center_size(resp.rect.center(), egui::vec2(48.0, 48.0));
            egui::Spinner::new().color(MUTED()).paint_at(ui, spinner_rect);
        }
    });
}

/// Load the platform's system CJK and math/symbol fonts and append them as
/// fallbacks to egui's font families, so glyphs the bundled fonts lack — CJK
/// (Japanese / Chinese / Korean) and the Mathematical Alphanumeric Symbols /
/// Letterlike "fancy font" letters people use in SD prompts (𝔗 ℜ 𝔊 …) — render
/// instead of showing as tofu boxes. Loading the OS fonts at runtime avoids
/// bundling multi-MB fonts in the binary. Best-effort: any font that isn't present
/// is simply skipped, and Latin/Cyrillic text keeps using egui's default fonts
/// (these are appended *after* the defaults, so they're only consulted for glyphs
/// the defaults can't draw).
fn install_fallback_fonts(ctx: &egui::Context) {
    // Candidate fonts grouped by coverage. Within each group we load only the
    // FIRST file that exists (so we don't pull, say, three redundant Japanese
    // fonts into memory); across groups we load one each so JP + KR + CN + math
    // all resolve. `index` selects a face inside a TrueType Collection (.ttc) —
    // e.g. Cambria Math is face 1 of CAMBRIA.TTC (face 0 is plain Cambria).
    #[cfg(target_os = "windows")]
    let groups: &[&[(&str, u32)]] = &[
        // Japanese (kana + kanji): Meiryo → Yu Gothic → MS Gothic.
        &[
            (r"C:\Windows\Fonts\meiryo.ttc", 0),
            (r"C:\Windows\Fonts\YuGothM.ttc", 0),
            (r"C:\Windows\Fonts\msgothic.ttc", 0),
        ],
        // Korean: Malgun Gothic.
        &[(r"C:\Windows\Fonts\malgun.ttf", 0)],
        // Chinese Simplified: MS YaHei → SimSun.
        &[
            (r"C:\Windows\Fonts\msyh.ttc", 0),
            (r"C:\Windows\Fonts\simsun.ttc", 0),
        ],
        // Mathematical Alphanumeric Symbols + Letterlike (𝔗 ℜ 𝔊 …): Cambria Math.
        &[(r"C:\Windows\Fonts\CAMBRIA.TTC", 1)],
        // Broader symbols / dingbats / arrows people drop in prompts.
        &[(r"C:\Windows\Fonts\seguisym.ttf", 0)],
        // Emoji: Segoe UI Emoji's outline layer. egui rasterizes fonts
        // monochrome-only, so this is what editable fields (the AI chat's
        // send box, prompt boxes) show while typing — proper emoji shapes
        // with no tofu (it also maps U+FE0F to a zero-width glyph). Sent
        // messages render full-color Twemoji instead (src/emoji.rs).
        &[(r"C:\Windows\Fonts\seguiemj.ttf", 0)],
    ];
    #[cfg(target_os = "macos")]
    let groups: &[&[(&str, u32)]] = &[
        // PingFang covers CN/JP/KR; Hiragino is the JP fallback.
        &[
            ("/System/Library/Fonts/PingFang.ttc", 0),
            ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 0),
        ],
        &[("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0)], // Korean
        // Math alphanumerics: STIX Two Math if present; Apple Symbols for the rest.
        &[("/System/Library/Fonts/Supplemental/STIXTwoMath.otf", 0)],
        &[("/System/Library/Fonts/Apple Symbols.ttf", 0)],
    ];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let groups: &[&[(&str, u32)]] = &[
        // Noto Sans CJK (one .ttc) covers all of CJK; wqy is a fallback.
        &[
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
            ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
            ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 0),
            ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
        ],
        // Math alphanumerics + symbols: Noto Sans Math.
        &[
            ("/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf", 0),
            ("/usr/share/fonts/opentype/noto/NotoSansMath-Regular.ttf", 0),
            ("/usr/share/fonts/noto/NotoSansMath-Regular.ttf", 0),
        ],
    ];

    let mut fonts = egui::FontDefinitions::default();
    let mut added: Vec<String> = Vec::new();

    for group in groups {
        for (path, index) in *group {
            let Ok(bytes) = std::fs::read(path) else { continue };
            // Derive a stable family key from the file name.
            let key = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("cjk")
                .to_string();
            if fonts.font_data.contains_key(&key) {
                break;
            }
            let mut data = egui::FontData::from_owned(bytes);
            data.index = *index;
            fonts.font_data.insert(key.clone(), data.into());
            added.push(key);
            break; // one font per language group is enough
        }
    }

    // Append the CJK fonts after the existing fonts in both families, so they act
    // purely as fallbacks for glyphs the default fonts can't render.
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        let list = fonts.families.entry(family).or_default();
        for key in &added {
            list.push(key.clone());
        }
    }

    // Cursive script font for the startup splash ("Clarity TagFlow" write-on).
    // The family is ALWAYS registered — laying out with an unknown family
    // panics — and just maps to the proportional list when no script font is
    // installed (the splash then writes in the regular face).
    #[cfg(target_os = "windows")]
    let cursive_candidates: &[&str] = &[
        r"C:\Windows\Fonts\segoesc.ttf",  // Segoe Script
        r"C:\Windows\Fonts\segoepr.ttf",  // Segoe Print
        r"C:\Windows\Fonts\BRUSHSCI.TTF", // Brush Script MT
        r"C:\Windows\Fonts\FREESCPT.TTF", // Freestyle Script
    ];
    #[cfg(target_os = "macos")]
    let cursive_candidates: &[&str] = &[
        "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
        "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    ];
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let cursive_candidates: &[&str] = &[
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    ];
    let mut cursive_list: Vec<String> = Vec::new();
    for path in cursive_candidates {
        let Ok(bytes) = std::fs::read(path) else { continue };
        fonts
            .font_data
            .insert("splash_cursive".into(), egui::FontData::from_owned(bytes).into());
        cursive_list.push("splash_cursive".into());
        break;
    }
    if let Some(prop) = fonts.families.get(&egui::FontFamily::Proportional) {
        cursive_list.extend(prop.iter().cloned());
    }
    fonts
        .families
        .insert(egui::FontFamily::Name("cursive".into()), cursive_list);

    ctx.set_fonts(fonts);
}

/// A rounded panel with the PANEL() fill, faint edge, and a soft drop shadow.
pub(crate) fn card_frame(radius: u8) -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(CornerRadius::same(radius))
        .inner_margin(Margin::same(12))
        .stroke(Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 4],
            blur: 14,
            spread: 0,
            color: Color32::from_black_alpha(110),
        })
}

/// A borderless button using an SVG or raster image source, tinted with `tint`.
pub(crate) fn svg_button(ui: &mut egui::Ui, source: egui::ImageSource<'_>, tooltip: &str, icon_size: f32, tint: Color32) -> egui::Response {
    let img = egui::Image::new(source)
        .fit_to_exact_size(egui::vec2(icon_size, icon_size))
        .tint(tint);

    let resp = ui.add(
        egui::Button::image(img)
            .frame(false)
            .min_size(egui::vec2(icon_size + 12.0, icon_size + 12.0))
    );
    resp.on_hover_text(tooltip)
}

/// A hyperlink that ends in a right-pointing arrow drawn with the
/// `arrow_right_alt.svg` icon (tinted to the link colour) instead of a "→" text
/// glyph. The arrow is clickable and opens the same `url`. `size` sets the link
/// text size (and the icon scales to match); `None` uses the default body size.
pub(crate) fn arrow_link(ui: &mut egui::Ui, label: &str, url: &str, size: Option<f32>) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        let color = ui.visuals().hyperlink_color;
        let mut text = egui::RichText::new(label).color(color);
        if let Some(s) = size {
            text = text.size(s);
        }
        ui.hyperlink_to(text, url);

        let icon = size.unwrap_or_else(|| ui.text_style_height(&egui::TextStyle::Body));
        let img = egui::Image::new(egui::include_image!("../icons/arrow_right_alt.svg"))
            .fit_to_exact_size(egui::vec2(icon, icon))
            .tint(color);
        if ui
            .add(egui::Button::image(img).frame(false))
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        {
            ui.ctx().open_url(egui::OpenUrl::new_tab(url));
        }
    });
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------
pub(crate) fn is_image(p: &std::path::Path) -> bool {
    let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) else {
        return false;
    };
    if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
        return true;
    }
    // Extended (heavy) formats are only recognised when BOTH a decoder was
    // compiled in (the `avif` cargo feature) AND the user turned them on. Without
    // the feature this whole branch is gone, so a stale persisted "on" setting
    // can't make the app list .avif files it can't actually decode.
    #[cfg(feature = "avif")]
    {
        if EXTENDED_FORMATS_ON.load(std::sync::atomic::Ordering::Relaxed)
            && EXTENDED_IMAGE_EXTENSIONS.contains(&ext.as_str())
        {
            return true;
        }
    }
    false
}

/// True if `path` has a recognised video extension.
pub(crate) fn is_video(p: &std::path::Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// True if `path` is a GIF (used by the media-type filter to keep GIFs in their
/// own bucket, separate from still images).
pub(crate) fn is_gif(p: &std::path::Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gif"))
        .unwrap_or(false)
}

/// Any media file we list in the browser (image or video).
fn is_media(p: &std::path::Path) -> bool {
    is_image(p) || is_video(p)
}

/// All image files directly inside `dir` (non-recursive), sorted by name.
fn images_in_dir(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|p| p.is_file() && is_media(p))
        .collect();
    found.sort();
    found
}

/// Decode `src` to a full-resolution RGBA image for the clipboard. Extended
/// formats (AVIF/HEIC/HEIF + TIFF-based raws) are routed to our own decoder by
/// extension — matching the display path — since `image::open` can't handle them.
/// Everything else goes through `image::open`. Returns `None` on any failure.
fn decode_full_rgba(src: &std::path::Path) -> Option<image::RgbaImage> {
    #[cfg(feature = "avif")]
    {
        let is_extended = src
            .extension()
            .and_then(|e| e.to_str())
            .map(is_extended_extension)
            .unwrap_or(false);
        if is_extended {
            return avif::decode_avif(src);
        }
    }
    image::open(src).ok().map(|img| img.to_rgba8())
}

/// Where a crop of `src` is saved: `<name>_crop.png` next to the original, with a
/// ` (n)` suffix if that already exists. Mirrors ViewerPanel.resolveCropDestination.
fn crop_destination(src: &std::path::Path) -> PathBuf {
    let dir = src.parent().unwrap_or_else(|| std::path::Path::new("."));
    let base = src.file_stem().and_then(|s| s.to_str()).unwrap_or("crop");
    let mut dest = dir.join(format!("{base}_crop.png"));
    let mut n = 1;
    while dest.exists() {
        dest = dir.join(format!("{base}_crop ({n}).png"));
        n += 1;
    }
    dest
}

pub(crate) fn file_name(p: &std::path::Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>")
        .to_owned()
}

#[cfg(all(test, target_os = "windows"))]
mod cf_dib_tests {
    // Layout check for the clipboard CF_DIB payload: 40-byte BITMAPINFOHEADER,
    // bottom-up BGRX rows, alpha composited against white.
    #[test]
    fn dib_header_and_pixels() {
        // 2x2 RGBA: red | half-green / blue | fully transparent.
        let rgba = [
            255u8, 0, 0, 255,   0, 255, 0, 128,
            0, 0, 255, 255,     0, 0, 0, 0,
        ];
        let dib = super::ViewerApp::rgba_to_cf_dib(2, 2, &rgba);
        assert_eq!(dib.len(), 40 + 16);
        assert_eq!(&dib[0..4], &40u32.to_le_bytes()); // biSize
        assert_eq!(&dib[4..8], &2i32.to_le_bytes()); // biWidth
        assert_eq!(&dib[8..12], &2i32.to_le_bytes()); // biHeight (bottom-up)
        assert_eq!(u16::from_le_bytes([dib[14], dib[15]]), 32); // biBitCount
        assert_eq!(&dib[16..20], &0u32.to_le_bytes()); // BI_RGB
        // Bottom row first: blue -> BGRX(255,0,0); transparent -> white.
        assert_eq!(&dib[40..44], &[255, 0, 0, 255]);
        assert_eq!(&dib[44..48], &[255, 255, 255, 255]);
        // Top row: red -> BGRX(0,0,255); half-green on white -> (127,255,127).
        assert_eq!(&dib[48..52], &[0, 0, 255, 255]);
        assert_eq!(&dib[52..56], &[127, 255, 127, 255]);
    }
}
