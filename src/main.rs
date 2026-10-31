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
    "png", "jpg", "jpeg", "jfif", "gif", "bmp", "webp", "ico", "tif", "tiff", "hdr", "svg",
];

/// "Extended" image extensions decoded via the heavier pure-Rust crates: AVIF
/// (avif-parse + rav1d), HEIC/HEIF (heic), and camera raw — DNG, Sony ARW,
/// Canon CR2 and Nikon NEF — (zenraw). Only listed when the user enables them in
/// Settings AND the app was built with the `avif` feature (`is_image` and
/// `is_extended_extension` check both, so a stale persisted setting can't make a
/// normal build list files it can't decode). Such a file owns its `.txt` in
/// every build, though (`has_sidecar_extension`): deleting `cat.png` must not
/// take the `cat.txt` that `cat.heic` still uses.
const EXTENDED_IMAGE_EXTENSIONS: &[&str] = &["avif", "heic", "heif", "dng", "arw", "cr2", "nef"];

/// True when `ext` (without the dot, any case) is one of the heavy "extended"
/// image formats — AVIF / HEIC / HEIF and the TIFF-based camera raws — that need
/// the pure-Rust decoders in `crate::avif` rather than `image::open`. Centralises
/// the list that was otherwise repeated across every decode path. Always `false`
/// in builds without the `avif` feature (those files aren't recognised anyway).
#[cfg_attr(not(feature = "avif"), allow(dead_code))] // callers sit behind avif gates
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
// AI Settings window (the Tools menu's gear): replies, memory
// and role play tabs.
mod ai_settings;
mod album;
// The AI's long-term memory across chats (and role play's own, kept apart),
// and the 3D memory map.
mod memory;
mod memory_map;
// The OmniVoice neural voice for the chat's Listen buttons (Python sidecar).
mod voice;
// Role playing for the AI Chat: persona, names, its own memories.
mod roleplay;
// Krea 2 prompt writer for the AI Chat (tools menu).
mod krea_prompt;
// Tag images with the AI Chat (tools menu): the image, the chat and the Tag
// Manager side by side; the AI can change tags.
mod ai_tagging;
// Encoding-tolerant, minimal-change .txt tag sidecar reading/writing.
mod sidecar;
// The zip viewer only needs the (non-optional) `zip` crate, so it compiles in
// every build — including the no-default-features release/CI configurations.
mod archive;
// AVIF/HEIC/camera-raw decoding sits behind the `avif` feature (rav1d, heic
// and zenraw are heavy builds); the stub keeps every caller compiling — those
// formats simply fail to decode in builds without it.
#[cfg(feature = "avif")]
mod avif;
#[cfg(not(feature = "avif"))]
mod avif {
    // Today's callers all sit behind their own `avif` gates; the stub stays so
    // an ungated call keeps compiling (and simply fails to decode).
    #[allow(dead_code)]
    pub fn decode_avif(_path: &std::path::Path) -> Option<image::RgbaImage> {
        None
    }
}
mod age_gate;
mod backup;
mod bgremove;
mod civitai;
mod crash;
mod depth;
mod detect;
mod dictate;
mod emoji;
mod favorites;
mod ftp;
mod image_cache;
mod danbooru;
mod download;
mod pexels;
mod pixabay;
mod safebooru;
mod wallhaven;
mod gallery;
mod gallery_detail;
// Flux text-to-image generation (ComfyUI backend) — NVIDIA-only, like Pixal3D.
#[cfg(not(target_os = "macos"))]
mod generate;
mod gif_info;
mod left_browser;
mod left_panel_settings;
mod liquid_glass;
// Link (Settings → Link): another PC on the network uses this one's models.
mod link;
// Local AI model (Settings → AI Model): Gemma 4 vision via llama.cpp, fully
// in-process. The module always compiles; the heavy bindings sit behind the
// `llm` cargo feature.
mod llm;
mod mp4;
mod net;
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
    // First thing, before anything can fail: crash reports + stderr log into
    // %APPDATA%\Clarity TagFlow\crash reports (see src/crash.rs).
    crash::install();

    // Link's background host: no window, just hosting (see src/link/background.rs).
    if std::env::args().any(|a| a == link::background::FLAG) {
        std::process::exit(link::background::run());
    }

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

    let result = eframe::run_native(
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
            app.link.set_ctx(cc.egui_ctx.clone());
            // Restore saved settings (if any) from eframe's persistent storage.
            if let Some(storage) = cc.storage
                && let Some(saved) = eframe::get_value::<settings::Settings>(storage, settings::STORAGE_KEY) {
                    app.settings = saved.loaded();
                    // Restore the last-used AI tagger model into the Tag Manager.
                    app.tag_manager.ai_model.clone_from(&app.settings.last_ai_model);
                    // Restore the Tag Manager's gear-popup preferences.
                    app.tag_manager.settings = app.settings.tag_manager.clone();
                }
            // Apply the saved colour theme before the first paint so a Light-mode
            // user doesn't see a dark flash on launch. The Glass config (incl.
            // dark/light panels) must be pushed first — the palette reads it.
            set_glass_config(app.settings.glass_bg, app.settings.backdrop(), app.settings.glass_light);
            crate::theme::set_chat_text(app.settings.ai_chat_text_enabled.then_some(app.settings.ai_chat_text));
            crate::theme::set_chat_icon(app.settings.ai_chat_icon_enabled.then_some(app.settings.ai_chat_icon));
            crate::theme::set_orb_color(app.settings.ai_orb_color_enabled.then_some(app.settings.ai_orb_color));
            set(app.settings.theme);
            app.last_theme = app.settings.theme;
            app.last_glass_light = app.settings.glass_light;
            app.last_glass_bg = app.settings.glass_bg;
            apply_theme(&cc.egui_ctx);
            // Optional: open a folder passed on the command line (e.g. "Open
            // with"), falling back to the Input folder from the last session.
            // A `.zip` input never opens silently — it always goes through the
            // password prompt first.
            let is_zip = |p: &PathBuf| {
                p.is_file()
                    && p.extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
            };
            let cli = std::env::args().nth(1).map(PathBuf::from);
            let saved = (!app.settings.input_folder.trim().is_empty())
                .then(|| PathBuf::from(app.settings.input_folder.trim()));
            let restore = cli
                .filter(|p| p.is_dir() || is_zip(p))
                .or_else(|| saved.filter(|p| p.is_dir() || is_zip(p)));
            if let Some(target) = restore {
                if is_zip(&target) {
                    app.prompt_for_zip(target);
                } else {
                    app.load_folder(&target);
                }
            }
            Ok(Box::new(app))
        }),
    );
    // Link: jobs another PC still has running here finish before the app
    // exits (the window is already gone; its background copy has taken over).
    link::finish_handover();
    result
}

/// State of the zip password prompt (see `ViewerApp::show_zip_prompt`).
struct ZipPrompt {
    /// The `.zip` waiting to be opened.
    path: PathBuf,
    password: String,
    /// Shown under the password box after a failed attempt.
    error: Option<String>,
    /// The password box grabs focus once per prompt (and again after a wrong
    /// password clears it).
    focused: bool,
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
    /// The folder button's popup (input/output folder pickers).
    folder_popup: top_bar::FolderPopup,
    /// Password prompt for opening a zip archive as the input (see
    /// `src/archive.rs`). `Some` while the prompt is on screen.
    zip_prompt: Option<ZipPrompt>,
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
    /// The Glass background colour the visuals were last applied with, so
    /// picking a new colour re-applies them.
    last_glass_bg: [u8; 3],
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
    /// Sidecars being (re-)read for tag search on a background thread — the
    /// first search over a big folder used to read every .txt on the UI
    /// thread. Results land in `tag_search_cache` and re-run the filter.
    tag_index_rx: Option<std::sync::mpsc::Receiver<TagIndex>>,
    /// A crop being decoded/saved on a background thread (src, result).
    crop_rx: Option<std::sync::mpsc::Receiver<(PathBuf, Result<PathBuf, String>)>>,
    /// Favorited ("hearted") files, tracked by content hash so they survive
    /// moves/renames. Shown as a heart badge on the browser thumbnails.
    favorites: favorites::Favorites,
    stats: top_bar::SystemStats,
    /// FTP/FTPS remote-browser state (Settings → FTP/FTPS).
    ftp: ftp::FtpState,
    /// Local AI model state (Settings → AI Model): setup download + the
    /// resident Gemma 4 inference worker.
    llm: llm::LlmState,
    /// Link (Settings → Link): hosting this PC's models, or using another's.
    link: link::LinkState,

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
    /// The list shown is LTX/Wan's (folder images + videos), whose selection
    /// indexes the folder list — not a text-to-image tab's output list.
    flux_combine: bool,
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
    /// Full-resolution decodes for the centre viewer, loaded only when the
    /// view needs more pixels than `viewer`'s preview copy has (zoomed in,
    /// or fit on a big monitor). Capped at the GPU's texture limit (≤ 8192).
    viewer_full: image_cache::ImageCache,
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

    /// A browser tile's right-click Delete awaiting confirmation: the target's
    /// path (not index — the list can rescan while the dialog is open).
    /// Independent of the selection — the user keeps viewing what they're
    /// viewing.
    pending_tile_delete: Option<PathBuf>,
    /// The confirm dialog's "don't ask again" checkbox for the above.
    tile_delete_skip: bool,

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
    /// A message for the status overlay from code with no `Context` at hand
    /// (a failed move or delete); `drive_bg_job` shows it on the next frame.
    pending_toast: Option<String>,
    /// That message on show (text, hide-time) — its own line in the overlay,
    /// so a background-removal status running at the same time can't hide it.
    notice: Option<(String, f64)>,
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
            folder_popup: top_bar::FolderPopup::default(),
            zip_prompt: None,
            current_folder: None,
            last_extended_formats: settings::Settings::default().enable_extended_formats,
            last_theme: Theme::default(),
            last_glass_light: false,
            last_glass_bg: [0; 3],
            last_media_filter: left_panel_settings::MediaFilter::default(),
            filtered: Vec::new(),
            selected: None,
            search: String::new(),
            tag_search_cache: std::collections::HashMap::new(),
            tag_index_rx: None,
            crop_rx: None,
            favorites: favorites::Favorites::load(),
            stats: top_bar::SystemStats::default(),
            ftp: ftp::FtpState::default(),
            llm: llm::LlmState::default(),
            link: link::LinkState::new(egui::Context::default()),
            right_state: right_details::RightPanelState::default(),
            settings: settings::Settings::default(),
            scan: scan::ScanState::default(),
            detail_popup: gallery_detail::DetailPopup::default(),
            splash: splash::Splash::default(),
            flux_active: false,
            images_backup: None,
            flux_sig: (0, 0),
            flux_combine: false,
            backup: backup::BackupState::default(),
            tag_manager: tag_manager::TagManagerState::default(),
            // Separate large-decode gates: the viewer gets a dedicated permit so
            // the clicked image always decodes immediately (priority); the browser
            // gets its own pair so it never blocks — or is blocked by — the viewer.
            thumbs: image_cache::ImageCache::new(THUMB_MAX_EDGE, 400, false, 2),
            viewer: image_cache::ImageCache::new(zoom::PREVIEW_EDGE, 8, true, 1),
            viewer_full: image_cache::ImageCache::new(8192, 2, false, 1),
            video_thumbs: video::VideoThumbs::new(),
            video_previews: video::VideoPreviews::new(),
            video_player: None,
            last_video_path: None,
            pending_tile_delete: None,
            tile_delete_skip: false,
            zoom: zoom::ZoomState::default(),
            #[cfg(not(target_os = "macos"))]
            scene3d: scene3d::Scene3D::new(),
            bg_job: None,
            bg_toast: None,
            pending_toast: None,
            notice: None,
            last_center_rect: None,
            update: update::UpdateState::default(),
        }
    }
}

/// Sidecar text read for tag search: (image, sidecar mtime, lowercase text).
type TagIndex = Vec<(PathBuf, Option<std::time::SystemTime>, String)>;

impl ViewerApp {
    /// Update the cached list of filtered indices based on the search string.
    /// The query matches the file name OR the sidecar .txt tags (cached).
    fn update_filtered(&mut self) {
        let query = self.search.trim().to_lowercase();
        let filter = self.settings.media_filter;
        let mut filtered = Vec::with_capacity(self.images.len());
        // Sidecars not read yet (or changed on disk) — indexed in the background.
        let mut stale: Vec<PathBuf> = Vec::new();
        for i in 0..self.images.len() {
            // Clone the path so the favorite look-up (which needs `&mut favorites`)
            // doesn't clash with borrowing `self.images`.
            let path = self.images[i].clone();
            if !query.is_empty()
                && !file_name(&path).to_lowercase().contains(&query)
                && !self.tags_match(&path, &query, &mut stale)
            {
                continue;
            }
            // Selected type kinds union together; Favorites intersects (so
            // Images + Favorites = favorite images). No kinds selected = all
            // types pass. `is_image` counts GIFs as images, so exclude them
            // there to keep the "Images" and "GIFs" buckets distinct (matching
            // the Java filter).
            let type_ok = !(filter.images || filter.videos || filter.gifs)
                || (filter.images && is_image(&path) && !is_gif(&path))
                || (filter.videos && is_video(&path))
                || (filter.gifs && is_gif(&path));
            let keep = type_ok && (!filter.favorites || self.favorites.is_favorite(&path));
            if keep {
                filtered.push(i);
            }
        }
        self.filtered = filtered;
        if !stale.is_empty() && self.tag_index_rx.is_none() {
            self.start_tag_index(stale);
        }
    }

    /// Read `paths`' sidecars for tag search off the UI thread (through the
    /// archive layer, so tags inside an opened zip are searchable too).
    fn start_tag_index(&mut self, paths: Vec<PathBuf>) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.tag_index_rx = Some(rx);
        std::thread::spawn(move || {
            let index = paths
                .into_iter()
                .map(|p| {
                    let txt = right_details::sidecar_txt(&p);
                    let mtime = std::fs::metadata(&txt).and_then(|m| m.modified()).ok();
                    let text = sidecar::read_text_lossy(&txt).to_lowercase();
                    (p, mtime, text)
                })
                .collect();
            let _ = tx.send(index);
        });
    }

    /// Fold a finished tag index into the search cache and re-filter.
    fn poll_tag_index(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.tag_index_rx else { return };
        match rx.try_recv() {
            Ok(index) => {
                self.tag_index_rx = None;
                for (p, mtime, text) in index {
                    self.tag_search_cache.insert(p, (mtime, text));
                }
                self.update_filtered();
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint_after(Duration::from_millis(50)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.tag_index_rx = None,
        }
    }

    /// Whether `path`'s sidecar tags contain `query` (lowercase) as whole
    /// words — "cat" finds "cat" and "black cat", not "catgirl". The tag text
    /// is cached; a sidecar not read yet (or changed since) is queued in
    /// `stale` for the background indexer and matches once it lands.
    fn tags_match(&mut self, path: &std::path::Path, query: &str, stale: &mut Vec<PathBuf>) -> bool {
        let txt = right_details::sidecar_txt(path);
        let mtime = std::fs::metadata(&txt).and_then(|m| m.modified()).ok();
        match self.tag_search_cache.get(path) {
            Some((cached, text)) => {
                if *cached != mtime {
                    stale.push(path.to_path_buf()); // keep the old text meanwhile
                }
                contains_words(text, query)
            }
            None => {
                stale.push(path.to_path_buf());
                false
            }
        }
    }

    /// Unload the current folder (the folder popup's "Clear Input"), emptying
    /// the browser and viewer. Also forgets the remembered Input folder so the
    /// next launch starts empty, and unmounts any open zip archive.
    fn clear_folder(&mut self) {
        self.restore_folder_list();
        archive::close();
        self.images.clear();
        self.current_folder = None;
        self.selected = None;
        self.settings.input_folder.clear();
        self.tag_search_cache.clear();
        self.update_filtered();
    }

    /// Replace the browser contents with the (already opened) zip archive's
    /// media entries — virtual paths served from memory by `src/archive.rs`.
    fn load_zip_entries(&mut self, zip_path: &std::path::Path, mut entries: Vec<PathBuf>) {
        self.restore_folder_list();
        entries.retain(|p| is_media(p));
        entries.sort();
        self.images = entries;
        self.current_folder = None; // not a real directory
        // Remember the zip as the Input so the next launch re-prompts for it.
        self.settings.input_folder = zip_path.display().to_string();
        self.selected = if self.images.is_empty() { None } else { Some(0) };
        self.tag_search_cache.clear();
        self.update_filtered();
    }

    /// Replace the browser contents with all images found directly in `dir`.
    fn load_folder(&mut self, dir: &std::path::Path) {
        self.restore_folder_list();
        archive::close(); // a real folder replaces any mounted zip
        self.images = images_in_dir(dir);
        self.current_folder = Some(dir.to_path_buf());
        // Remember the Input folder so the next launch reopens it.
        self.settings.input_folder = dir.display().to_string();
        self.selected = if self.images.is_empty() { None } else { Some(0) };
        self.tag_search_cache.clear(); // tags belong to the previous folder
        self.update_filtered();
    }

    /// Open the Create Backup dialog for the current folder and its media. The
    /// backup root is the last-opened folder, falling back to the first image's
    /// parent (e.g. when files were dropped in individually).
    fn start_backup(&mut self) {
        // A mounted zip can't be backed up (its entries aren't files on disk).
        if archive::active().is_some() {
            return;
        }
        let source = self
            .current_folder
            .clone()
            .or_else(|| self.images.first().and_then(|p| p.parent().map(|d| d.to_path_buf())))
            .unwrap_or_default();
        self.backup.open(source, self.images.clone());
    }

    fn add_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        self.restore_folder_list();
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
    fn crop_current(&mut self, src: &std::path::Path, frac: zoom::CropFraction, ctx: &egui::Context) {
        // Zip entries are view-only (cropping writes a new file next to the
        // source); one crop at a time.
        if archive::is_entry(src) || self.crop_rx.is_some() {
            return;
        }
        // Decoding a big photo or RAW takes seconds — off the UI thread, with
        // a toast meanwhile; `drive_crop_job` adds the result to the browser.
        let (tx, rx) = std::sync::mpsc::channel();
        self.crop_rx = Some(rx);
        self.bg_toast = Some(("Cropping…".to_string(), f64::INFINITY));
        let src = src.to_path_buf();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = crop_to_file(&src, frac);
            let _ = tx.send((src, result));
            ctx.request_repaint();
        });
    }

    /// Finish a background crop: place it beneath its source and select it.
    fn drive_crop_job(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.crop_rx else { return };
        let (src, result) = match rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                (PathBuf::new(), Err("Crop worker stopped".to_string()))
            }
        };
        self.crop_rx = None;
        let now = ctx.input(|i| i.time);
        match result {
            Ok(dest) => {
                self.bg_toast = None;
                self.insert_derived(&src, dest);
            }
            Err(e) => self.bg_toast = Some((format!("Crop failed — {e}"), now + 5.0)),
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
        // Zip entries are view-only (the result is written next to the source).
        if self.bg_job.is_some() || archive::is_entry(src) {
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
        self.drive_crop_job(ctx);
        self.poll_tag_index(ctx);
        let now = ctx.input(|i| i.time);
        if let Some((_, until)) = &self.bg_toast
            && now >= *until {
                self.bg_toast = None;
            }
        if let Some((_, until)) = &self.notice
            && now >= *until {
                self.notice = None;
            }
        let fresh: Vec<String> =
            [self.pending_toast.take(), self.right_state.notice.take(), self.detail_popup.notice.take()]
                .into_iter()
                .flatten()
                .collect();
        if !fresh.is_empty() {
            // Messages arriving together, or while one is still on show, are
            // listed together (the last few) rather than replacing it.
            let mut lines: Vec<String> = self.notice.take().map(|(old, _)| old.lines().map(str::to_string).collect()).unwrap_or_default();
            lines.extend(fresh);
            let keep = lines.len().saturating_sub(3);
            self.notice = Some((lines[keep..].join("\n"), now + 6.0));
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(6.1));
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
        // A crop / cutout can reuse a deleted file's name: don't let it
        // inherit that file's cached heart.
        self.favorites.forget(&dest);
        // A crop of a folder image under LTX/Wan joins the stashed folder list
        // too, or it vanished on the view's next refresh and on leaving it.
        if self.flux_active
            && let Some((stash, sel)) = &mut self.images_backup
            && !stash.contains(&dest)
            && let Some(at) = stash.iter().position(|p| p == src).map(|i| i + 1)
        {
            stash.insert(at, dest.clone());
            if let Some(s) = sel.as_mut()
                && *s >= at
            {
                *s += 1;
            }
        }
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
        let status = if let Some(job) = &self.bg_job {
            Some((job.status.clone(), true))
        } else {
            self.bg_toast.as_ref().filter(|(_, until)| now < *until).map(|(msg, _)| (msg.clone(), false))
        };
        let notice = self.notice.as_ref().filter(|(_, until)| now < *until).map(|(msg, _)| msg.clone());
        if status.is_none() && notice.is_none() {
            return;
        }
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
                    if let Some((text, busy)) = &status {
                        ui.horizontal(|ui| {
                            if *busy {
                                ui.add(egui::Spinner::new().size(16.0));
                            }
                            ui.label(text.as_str());
                        });
                    }
                    if let Some(notice) = &notice {
                        ui.label(notice.as_str());
                    }
                });
            });
    }

    /// Arrow-key navigation through what the browser actually shows — the
    /// filtered list (search / media filters), wrapping at the ends.
    fn step_selection(&mut self, delta: i32) {
        if self.filtered.is_empty() {
            return;
        }
        let n = self.filtered.len() as i32;
        let pos = match self.selected.and_then(|s| self.filtered.iter().position(|&i| i == s)) {
            Some(pos) => (pos as i32 + delta).rem_euclid(n),
            // The selection is hidden by the filter: step to the nearest
            // visible image in the direction pressed.
            None => {
                let cur = self.selected.unwrap_or(0);
                if delta > 0 {
                    self.filtered.iter().position(|&i| i > cur).unwrap_or(0) as i32
                } else {
                    self.filtered.iter().rposition(|&i| i < cur).unwrap_or(n as usize - 1) as i32
                }
            }
        };
        self.selected = Some(self.filtered[pos as usize]);
    }

    /// A generator view shows its own list while the folder list waits in
    /// `images_backup`. Anything that changes the folder list (opening a
    /// folder or zip, dropping files) puts it back first and works on it —
    /// the view stashes it again, changed, on the next frame. Changing the
    /// shown list instead was undone when the view closed (the old folder's
    /// list came back under the new folder's name).
    fn restore_folder_list(&mut self) {
        if self.flux_active {
            #[cfg(not(target_os = "macos"))]
            self.remember_input_selection();
            if let Some((imgs, sel)) = self.images_backup.take() {
                self.images = imgs;
                self.selected = sel;
            }
            self.flux_active = false;
        }
    }

    /// Leaving LTX/Wan's list: a folder image picked there becomes the
    /// folder selection (found by path — a crop inserted into the shown list
    /// shifts positions), so the folder view — and LTX/Wan again later — come
    /// back to it.
    #[cfg(not(target_os = "macos"))]
    fn remember_input_selection(&mut self) {
        if self.flux_combine
            && let Some((stash, sel)) = &mut self.images_backup
            && let Some(picked) = self.selected.and_then(|i| self.images.get(i))
            && let Some(i) = stash.iter().position(|p| p == picked)
        {
            *sel = Some(i);
        }
    }

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
                // Outputs moved or deleted from the browser stay gone (the
                // generator still lists them). Checked only when the list is
                // rebuilt, not every frame.
                let rebuild = !self.flux_active || self.flux_sig != sig;
                let gen_list: Vec<PathBuf> =
                    if rebuild { gen_list.into_iter().filter(|p| p.exists()).collect() } else { gen_list };
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
                    self.flux_combine = combine;
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
                        } else if !self.flux_combine || self.selected.is_some_and(|i| i >= self.images.len()) {
                            // Coming from a text-to-image tab, `selected` indexed
                            // THAT tab's output list — out of range here (a panic
                            // the next frame) or, worse, in range and naming some
                            // other folder image, which LTX/Wan then animated.
                            // Fall back to the stashed folder selection (an index
                            // into `orig`, so always valid here).
                            self.selected = self
                                .images_backup
                                .as_ref()
                                .and_then(|(_, s)| *s)
                                .filter(|&i| i < self.images.len())
                                .or(if self.images.is_empty() { None } else { Some(0) });
                        }
                    } else {
                        self.remember_input_selection();
                        self.images = gen_list;
                        self.selected = if self.images.is_empty() { None } else { Some(self.images.len() - 1) };
                    }
                    self.update_filtered();
                    self.flux_sig = sig;
                    self.flux_combine = combine;
                }
            } else if self.flux_active {
                // Leaving: restore the real folder list.
                self.remember_input_selection();
                if let Some((imgs, sel)) = self.images_backup.take() {
                    self.images = imgs;
                    self.selected = sel;
                    self.update_filtered();
                }
                self.flux_active = false;
            }
        }
    }

    /// Move the selected image (and its `.txt` sidecar) to the Output folder
    /// (folder button's popup) — or a folder the user picks when none is set —
    /// then drop it from the list. Shared by the right panel and the gallery popup.
    fn move_selected(&mut self) {
        if let Some(idx) = self.selected {
            self.move_at(idx);
        }
    }

    /// Move the image at `idx` (and its sidecar) to the Output folder — or a
    /// picked folder when none is configured. Any list entry, not just the
    /// selection: also used by the browser tiles' right-click menu.
    fn move_at(&mut self, idx: usize) {
        let Some(img_path) = self.images.get(idx).cloned() else { return };
        // Zip entries are view-only — never move out of an archive (checked
        // before the picker so no dialog pops up either).
        if archive::is_entry(&img_path) {
            return;
        }
        let out = self.settings.output_folder.trim();
        let target_dir = if out.is_empty() {
            let Some(dir) = rfd::FileDialog::new().pick_folder() else { return };
            dir
        } else {
            std::path::PathBuf::from(out)
        };
        // Moving into the folder the image already lives in would only drop it
        // from the list — treat it as a no-op instead. Compared as resolved
        // paths: an Output path typed in another case, or through a link, is
        // the same folder.
        if img_path.parent().is_some_and(|dir| same_dir(dir, &target_dir)) {
            return;
        }
        let name = file_label(&img_path);
        if let Err(e) = std::fs::create_dir_all(&target_dir) {
            self.pending_toast = Some(format!("Couldn't move {name} — {e}"));
            return;
        }
        // An unsaved tag edit of this image goes into its .txt first, so it
        // moves with it (saved afterwards it left a stray .txt behind).
        if !self.right_state.save_before_move(&img_path) || !self.detail_popup.save_before_move(&img_path) {
            self.pending_toast = Some(format!("Couldn't save the tag edit of {name}, so it wasn't moved"));
            return;
        }
        self.release_file(&img_path);
        // Never replace a file already in the target folder: a same-named
        // image (or .txt) there gets this one renamed beside it instead.
        let Some(dest) = move_destination(&img_path, &target_dir) else {
            self.pending_toast = Some(format!("Couldn't move {name} — no free name in the folder"));
            return;
        };
        if let Err(e) = move_file(&img_path, &dest) {
            self.pending_toast = Some(format!("Couldn't move {name} — {e}"));
            return;
        }
        // The .txt goes along — copied instead when another file here shares
        // it (cat.png and cat.jpg both read cat.txt). If it can't follow, the
        // image goes back, so the two are never separated.
        let txt_path = right_details::sidecar_txt(&img_path);
        if txt_path.exists() {
            let dest_txt = right_details::sidecar_txt(&dest);
            let moved = if shares_sidecar(&img_path) {
                std::fs::copy(&txt_path, &dest_txt).map(|_| ())
            } else {
                move_file(&txt_path, &dest_txt)
            };
            if let Err(e) = moved {
                let undo = move_file(&dest, &img_path);
                self.pending_toast = Some(match undo {
                    Ok(()) => format!("Couldn't move the tags of {name}, so it stayed — {e}"),
                    Err(_) => format!("Moved {name} but not its tags — {e}"),
                });
                if undo.is_ok() {
                    return;
                }
            }
        }

        self.remove_image_at(idx);
    }

    /// Let go of `path` before it's moved or deleted: stop the centre player
    /// when it's playing that file (VLC holds it open, so on Windows the
    /// move/delete failed) and the browser's tile previews (they restart on
    /// the next frame for the tiles still listed).
    fn release_file(&mut self, path: &std::path::Path) {
        if self.last_video_path.as_deref() == Some(path) {
            self.video_player = None;
            self.last_video_path = None;
        }
        if is_video(path) {
            self.video_previews.set_enabled(false);
        }
    }

    /// Show the password prompt for `path` (a `.zip`). The archive is opened
    /// only after the prompt — even at startup for the remembered Input zip —
    /// and the password never leaves the process.
    fn prompt_for_zip(&mut self, path: PathBuf) {
        self.zip_prompt = Some(ZipPrompt {
            path,
            password: String::new(),
            error: None,
            focused: false,
        });
    }

    /// Draw the zip password prompt while one is pending and handle its
    /// Open/Cancel actions.
    fn show_zip_prompt(&mut self, ctx: &egui::Context) {
        let mut submit = false;
        let mut cancel;
        {
            let Some(prompt) = &mut self.zip_prompt else { return };
            cancel = ctx.input(|i| i.key_pressed(egui::Key::Escape));
            let name = prompt
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // A fixed, centred modal — deliberately not movable, like the other
            // modal dialogs (Settings / Backup / Confirm Delete).
            egui::Window::new("Open Zip")
                .id(egui::Id::new("zip_password_prompt"))
                .title_bar(false)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .frame(
                    egui::Frame::new()
                        .fill(FLOAT())
                        .corner_radius(CornerRadius::same(16))
                        .inner_margin(Margin::same(18))
                        .stroke(Stroke::new(1.0, EDGE()))
                        .shadow(egui::epaint::Shadow {
                            offset: [0, 6],
                            blur: 20,
                            spread: 0,
                            color: Color32::from_black_alpha(150),
                        }),
                )
                .show(ctx, |ui| {
                    ui.set_width(320.0);
                    ui.heading(egui::RichText::new("Open Zip").color(TEXT()).strong().size(17.0));
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(&name).color(MUTED()).size(12.0));
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Enter the archive password (leave it empty if the zip isn't \
                             encrypted). Contents are viewed straight from the zip — nothing \
                             is ever extracted.",
                        )
                        .color(MUTED())
                        .size(11.0),
                    );
                    ui.add_space(8.0);
                    let resp = ui.add_sized(
                        egui::vec2(ui.available_width(), 30.0),
                        egui::TextEdit::singleline(&mut prompt.password)
                            .password(true)
                            .hint_text("Password")
                            // Centre the text in the 30px box (a sized TextEdit
                            // top-aligns its one line by default).
                            .vertical_align(egui::Align::Center),
                    );
                    if !prompt.focused {
                        resp.request_focus();
                        prompt.focused = true;
                    }
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        submit = true;
                    }
                    if let Some(err) = &prompt.error {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(err)
                                .color(Color32::from_rgb(230, 90, 90))
                                .size(12.0),
                        );
                    }
                    ui.add_space(12.0);
                    // Wrapped in horizontal() so the right-to-left layout only
                    // claims one row, not all remaining window height.
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let open_btn = egui::Button::new(
                                egui::RichText::new("Open").color(Color32::WHITE).strong(),
                            )
                            .fill(ACCENT1())
                            .min_size(egui::vec2(84.0, 30.0));
                            if ui.add(open_btn).clicked() {
                                submit = true;
                            }
                            if ui
                                .add(egui::Button::new("Cancel").min_size(egui::vec2(84.0, 30.0)))
                                .clicked()
                            {
                                cancel = true;
                            }
                        });
                    });
                });
        }

        if submit {
            let (path, password) = {
                let p = self.zip_prompt.as_ref().unwrap();
                (p.path.clone(), p.password.clone())
            };
            match archive::open_archive(&path, &password) {
                Ok(entries) => {
                    self.zip_prompt = None;
                    self.load_zip_entries(&path, entries);
                }
                Err(archive::OpenError::WrongPassword) => {
                    if let Some(p) = &mut self.zip_prompt {
                        p.error = Some("Wrong password — try again.".to_string());
                        p.password.clear();
                        p.focused = false; // re-focus the emptied box
                    }
                }
                Err(archive::OpenError::Other(e)) => {
                    if let Some(p) = &mut self.zip_prompt {
                        p.error = Some(format!("Couldn't open the zip: {e}"));
                    }
                }
            }
        } else if cancel {
            self.zip_prompt = None;
        }
    }

    /// Remove image `idx`, re-filter the browser, and clamp the selection to a
    /// still-valid index (or clear it when the list is now empty).
    fn remove_image_at(&mut self, idx: usize) {
        let removed = self.images.remove(idx);
        // Whatever is created at this path later is hashed afresh.
        self.favorites.forget(&removed);
        // Under LTX/Wan the folder images shown are copies of the stashed
        // folder list: drop it there too, or it reappears on the next refresh
        // and when the view closes.
        self.drop_from_stash(&removed);
        self.update_filtered();
        self.selected = if self.images.is_empty() {
            None
        } else {
            match self.selected {
                // A different image was selected (e.g. a tile deleted from the
                // browser's right-click menu): keep viewing it — its index
                // shifts down when it sat after the removed one.
                Some(sel) if sel > idx => Some(sel - 1),
                Some(sel) if sel < idx => Some(sel),
                // The removed image itself (or none): the nearest one the
                // browser still shows (search / filters), else any.
                _ => self
                    .filtered
                    .iter()
                    .copied()
                    .find(|&i| i >= idx)
                    .or_else(|| self.filtered.last().copied())
                    .or(Some(idx.min(self.images.len().saturating_sub(1)))),
            }
        };
    }

    /// Mirror Tag Manager choices into the persisted Settings so they
    /// auto-save: the chosen tagger model (restored next launch) and the
    /// gear popup's settings (it has no Save button).
    fn sync_tag_manager_settings(&mut self) {
        if self.settings.last_ai_model != self.tag_manager.ai_model {
            self.settings.last_ai_model = self.tag_manager.ai_model.clone();
        }
        if self.settings.tag_manager != self.tag_manager.settings {
            self.settings.tag_manager = self.tag_manager.settings.clone();
        }
    }

    /// The AI chat's Tag images view, left panel: the selected image with
    /// previous/next (src/ai_tagging.rs draws it; the textures come from the
    /// centre viewer's caches, so switching views doesn't reload).
    fn ai_tag_image_panel(&mut self, ui: &mut egui::Ui, path: &std::path::Path, width: f32) {
        let now = ui.input(|i| i.time);
        let (tex, loading, failed) = if is_video(path) {
            (self.video_thumbs.request(path, ui.ctx()), false, false)
        } else {
            match self.viewer.request(path, now) {
                image_cache::Cached::Ready(tex) => (Some(tex), false, false),
                image_cache::Cached::Animated(frame) => {
                    // Keep a GIF playing, capped like the centre viewer.
                    ui.ctx().request_repaint_after(Duration::from_millis(16));
                    (Some(frame), false, false)
                }
                image_cache::Cached::Failed => (None, false, true),
                image_cache::Cached::Loading => {
                    ui.ctx().request_repaint_after(Duration::from_millis(60));
                    match self.thumbs.request(path, now) {
                        image_cache::Cached::Ready(t) | image_cache::Cached::Animated(t) => (Some(t), true, false),
                        _ => (None, true, false),
                    }
                }
            }
        };
        let position = self
            .selected
            .and_then(|sel| self.filtered.iter().position(|&i| i == sel))
            .map(|p| p + 1);
        let step = ai_tagging::image_panel(
            ui,
            ai_tagging::PanelImage {
                path,
                position,
                total: self.filtered.len(),
                tex: tex.as_ref(),
                loading,
                failed,
            },
            width,
        );
        if step != 0 {
            self.step_selection(step);
        }
    }

    /// Carry out the AI chat's tag edits and undos (Tag images mode): write
    /// the image's .txt, refresh the Tag Manager when it's the image on show,
    /// and keep the change on the reply for its note and Undo button.
    fn apply_ai_tag_requests(&mut self, ctx: &egui::Context) {
        for request in self.llm.take_tag_requests() {
            match request {
                ai_tagging::TagRequest::Apply { chat_id, index, image, edits } => {
                    let change = self.ai_apply_tags(image, &edits);
                    if let Some(m) = self.llm.msg_mut(chat_id, index) {
                        m.tag_change = Some(change);
                    }
                }
                ai_tagging::TagRequest::Undo { chat_id, index } => {
                    self.ai_undo_reply(chat_id, index, false);
                }
                // Regenerate / edit-and-resend: the replies being replaced
                // take their tag changes back first — newest first, so each
                // finds the file as it left it — then the chat resends.
                ai_tagging::TagRequest::Rewind { chat_id, from, then } => {
                    let len = self.llm.chats.iter().find(|c| c.id == chat_id).map_or(0, |c| c.msgs.len());
                    let undone = (from..len).rev().all(|index| self.ai_undo_reply(chat_id, index, true));
                    let still_open = self.llm.chats.get(self.llm.active_chat).is_some_and(|c| c.id == chat_id);
                    if !undone {
                        self.llm.run_err = Some(
                            "Couldn't take back a reply's tag changes, so nothing was resent — the note under \
                             it says why."
                                .to_string(),
                        );
                        // The rewritten message isn't lost: back into its editor.
                        if let ai_tagging::Resend::Edit(i, text) = then {
                            self.llm.editing = Some((chat_id, i));
                            self.llm.edit_draft = text;
                        }
                    } else if still_open {
                        match then {
                            ai_tagging::Resend::Retry(i) => self.llm.retry_now(i, ctx),
                            ai_tagging::Resend::Edit(i, text) => self.llm.apply_edit_now(i, text, ctx),
                        }
                    }
                }
            }
        }
    }

    /// Undo the tag change of reply `index` in chat `chat_id`, if it has one
    /// still in place. False when that failed: its note then says why, and
    /// keeps Undo to try again. `rewinding` (the reply is about to be
    /// replaced): an image that's gone since — moved or deleted, its tags
    /// with it — doesn't hold the resend up.
    fn ai_undo_reply(&mut self, chat_id: u64, index: usize, rewinding: bool) -> bool {
        let Some(change) = self.llm.msg_mut(chat_id, index).and_then(|m| m.tag_change.clone()) else {
            return true;
        };
        if change.undone || !change.changed() || (rewinding && !change.image.exists()) {
            return true;
        }
        let result = self.ai_undo_tags(&change);
        if let Some(c) = self.llm.msg_mut(chat_id, index).and_then(|m| m.tag_change.as_mut()) {
            match result {
                Ok(()) => {
                    c.undone = true;
                    c.undo_error = None;
                }
                Err(e) => c.undo_error = Some(e.to_string()),
            }
        }
        result.is_ok()
    }

    /// The image's .txt, if the AI may change it: not inside a zip backup,
    /// and the image still there.
    fn ai_tag_file(image: &std::path::Path) -> Result<PathBuf, &'static str> {
        if archive::is_entry(image) {
            return Err("it's inside a zip backup, which is read-only");
        }
        if !image.exists() {
            return Err("the image isn't there any more");
        }
        Ok(right_details::sidecar_txt(image))
    }

    /// Apply the AI's tag edits to `image`'s .txt like a manual edit (only
    /// what changed, same encoding). Returns the change for the reply's note.
    fn ai_apply_tags(&mut self, image: PathBuf, edits: &ai_tagging::TagEdits) -> ai_tagging::TagChange {
        let txt = match Self::ai_tag_file(&image) {
            Ok(txt) => txt,
            Err(why) => return ai_tagging::TagChange::failed(image, why),
        };
        // Never write under an edit the user hasn't saved yet.
        if self.right_state.unsaved_edit(&image) {
            return ai_tagging::TagChange::failed(
                image,
                "its tags are open in Edit Text with changes not saved yet, so nothing was written — save \
                 them and ask again",
            );
        }
        let Ok(sc) = sidecar::read(&txt) else {
            return ai_tagging::TagChange::failed(image, "its .txt file can't be read");
        };
        // The file's exact bytes, so Undo can put back the very same file.
        let before_bytes = if sc.existed {
            match std::fs::read(&txt) {
                Ok(bytes) => Some(bytes),
                Err(_) => return ai_tagging::TagChange::failed(image, "its .txt file can't be read"),
            }
        } else {
            None
        };
        let before = tag_manager::parse_tags(&sc.text);
        let after = ai_tagging::apply(&before, edits);
        let before_text = sc.existed.then(|| sc.text.clone());
        let after_text = if after == before {
            sc.text.clone()
        } else {
            match tag_manager::write_tags(&image, &after) {
                Some(text) => text,
                None => return ai_tagging::TagChange::failed(image, "its .txt file can't be written"),
            }
        };
        let change =
            ai_tagging::TagChange { image, before, after, before_text, after_text, before_bytes, ..Default::default() };
        if change.changed() {
            self.ai_tags_written(&change.image, &change.after_text, &change.before, &change.after, false);
        }
        change
    }

    /// Undo a reply's tag change. When the .txt is exactly as the AI left it,
    /// the very same file comes back, byte for byte (and a .txt the AI
    /// created is removed); otherwise only the AI's own change is taken back,
    /// keeping later edits.
    fn ai_undo_tags(&mut self, change: &ai_tagging::TagChange) -> Result<(), &'static str> {
        let txt = Self::ai_tag_file(&change.image)?;
        if self.right_state.unsaved_edit(&change.image) {
            return Err("its tags are open in Edit Text with changes not saved yet — save them first");
        }
        let sc = sidecar::read(&txt).map_err(|_| "its .txt file can't be read")?;
        let current = tag_manager::parse_tags(&sc.text);
        let (text, now) = if sc.text == change.after_text {
            match &change.before_bytes {
                Some(bytes) => {
                    std::fs::write(&txt, bytes).map_err(|_| "its .txt file can't be written")?;
                    (change.before_text.clone().unwrap_or_default(), change.before.clone())
                }
                None => {
                    if sc.existed {
                        std::fs::remove_file(&txt).map_err(|_| "its .txt file can't be removed")?;
                    }
                    (String::new(), Vec::new())
                }
            }
        } else {
            let tags = ai_tagging::undo(&current, change);
            let text = tag_manager::write_tags(&change.image, &tags).ok_or("its .txt file can't be written")?;
            (text, tags)
        };
        self.ai_tags_written(&change.image, &text, &current, &now, true);
        Ok(())
    }

    /// After the AI chat wrote an image's tags: the right panel shows the new
    /// text, and when it's the image on show the Tag Manager tints and
    /// flashes the change.
    fn ai_tags_written(&mut self, image: &std::path::Path, text: &str, before: &[String], after: &[String], undo: bool) {
        self.right_state.tags_written(image, text);
        if self.selected.and_then(|i| self.images.get(i)).is_some_and(|p| p == image) {
            tag_manager::ai_changed_tags(&mut self.tag_manager, before, after, undo);
        }
    }

    /// Apply a viewer right-click action (favorite / crop / copy / bg-remove) to
    /// `path`. Shared by the centre viewer and the gallery-detail popup.
    fn handle_viewer_action(&mut self, action: zoom::ViewerAction, path: &std::path::Path, ctx: &egui::Context) {
        match action {
            zoom::ViewerAction::ToggleFavorite => {
                self.favorites.toggle(path);
            }
            zoom::ViewerAction::Crop(frac) => self.crop_current(path, frac, ctx),
            zoom::ViewerAction::CopyImage => self.copy_current(path),
            zoom::ViewerAction::RemoveBackground => self.start_bgremove(path, ctx),
            zoom::ViewerAction::SendTo(key) => self.link.send(&key, vec![path.to_path_buf()]),
            zoom::ViewerAction::None => {}
        }
    }

    /// `path` is gone (deleted from Deep Scan's review list): take it off the
    /// list shown and, under a generator view, the folder list waiting behind
    /// it.
    fn remove_path(&mut self, path: &std::path::Path) {
        if let Some(i) = self.images.iter().position(|p| p == path) {
            self.remove_image_at(i);
        } else {
            self.favorites.forget(path);
            self.drop_from_stash(path);
        }
    }

    /// Drop `path` from the folder list stashed behind a generator view.
    fn drop_from_stash(&mut self, path: &std::path::Path) {
        if self.flux_active
            && let Some((stash, sel)) = &mut self.images_backup
            && let Some(j) = stash.iter().position(|p| p == path)
        {
            stash.remove(j);
            *sel = match *sel {
                Some(s) if s > j => Some(s - 1),
                Some(s) if s == j => (!stash.is_empty()).then(|| j.min(stash.len() - 1)),
                other => other,
            };
        }
    }

    /// Re-scan the current folder and try to keep the same image selected. Used
    /// after the extended-format toggle changes and after a Deep Scan moves files.
    fn rescan_current_folder(&mut self) {
        if let Some(dir) = self.current_folder.clone() {
            self.restore_folder_list();
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
        if let Some(idx) = self.selected {
            self.delete_at(idx);
        }
    }

    /// Delete the image at `idx` (any list entry, not just the selection) and
    /// its sidecar. Used by the Details panel's Delete button (via
    /// [`Self::delete_selected`]) and the browser tiles' right-click menu.
    fn delete_at(&mut self, idx: usize) {
        let Some(img_path) = self.images.get(idx).cloned() else { return };
        // Zip entries are view-only — never delete out of an archive.
        if archive::is_entry(&img_path) {
            return;
        }
        let txt_path = right_details::sidecar_txt(&img_path);
        let name = file_label(&img_path);

        // Stop the embedded player (and tile previews) when it's this file,
        // so VLC releases it before the delete (other tiles' deletes leave
        // playback alone).
        self.release_file(&img_path);

        // If the file can't be deleted (in use, read-only), its tags and its
        // place in the list stay too. One that's already gone (deleted in
        // Explorer) counts as deleted — but not when its whole folder is
        // unreachable (an unplugged drive, a dropped network share also say
        // "not found"): the file is still there.
        match std::fs::remove_file(&img_path) {
            Ok(()) => {}
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && img_path.parent().is_some_and(std::path::Path::is_dir) => {}
            Err(e) => {
                self.pending_toast = Some(format!("Couldn't delete {name} — {e}"));
                return;
            }
        }
        // An unsaved tag edit of the deleted image is dropped, not written
        // back afterwards (that recreated its .txt).
        self.right_state.discard_edit_of(&img_path);
        self.detail_popup.discard_edit_of(&img_path);
        // A .txt another file here still reads (cat.png and cat.jpg share
        // cat.txt) is kept for that file.
        if txt_path.exists()
            && !shares_sidecar(&img_path)
            && let Err(e) = std::fs::remove_file(&txt_path)
        {
            self.pending_toast = Some(format!("Deleted {name} but not its tags — {e}"));
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
        // A dropped folder loads all media inside it; dropped files are added
        // directly; a dropped zip goes through the password prompt and opens
        // as the (view-only, in-memory) input.
        let mut to_add = Vec::new();
        for p in dropped {
            if p.is_dir() {
                to_add.extend(images_in_dir(&p));
            } else if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
                self.prompt_for_zip(p);
                return;
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
                        // Swap in the full-resolution decode when the view needs
                        // more pixels than the preview has; until it lands the
                        // preview keeps showing (the zoom maths uses the real
                        // pixel size either way).
                        let tex = if self.zoom.wants_full_res(&path) {
                            match self.viewer_full.request(&path, now) {
                                image_cache::Cached::Ready(full) => full,
                                image_cache::Cached::Loading => {
                                    ui.ctx().request_repaint_after(Duration::from_millis(60));
                                    tex
                                }
                                _ => tex,
                            }
                        } else {
                            tex
                        };
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
        // Liquid Glass: once the frame's UI is done, the panels, windows and
        // menus it painted become real glass (src/liquid_glass.rs).
        let _glass = liquid_glass::at_frame_end(ui.ctx());
        // Interface size (Settings > Appearance). Applied before anything is laid
        // out so the whole UI, including fixed point sizes, scales together.
        {
            let ctx = ui.ctx().clone();
            let target = settings::effective_ui_scale(&ctx, &self.settings);
            if (ctx.zoom_factor() - target).abs() > 0.001 {
                ctx.set_zoom_factor(target);
            }
        }
        // While the splash is writing its wordmark, render ONLY the splash —
        // the panels (and the folder/thumbnail work they kick off) build after
        // it, so launch shows the text before the app UI exists. The fade-out
        // phase falls through and plays over the real UI.
        if self.splash.covers_ui(ui.ctx()) {
            self.splash.show(ui.ctx());
            return;
        }

        self.handle_dropped_files(ui.ctx());
        // Link: follow its settings; its messages join the status line, and a
        // browser showing the folder files just arrived in refreshes.
        self.link.tick(&self.settings);
        for n in self.link.take_notices() {
            self.pending_toast = Some(match self.pending_toast.take() {
                Some(prev) => format!("{prev}\n{n}"),
                None => n,
            });
        }
        for dir in self.link.take_received() {
            if self.current_folder.as_deref().is_some_and(|c| same_dir(c, &dir)) {
                self.rescan_current_folder();
            }
        }
        // A model loaded only for Link devices goes when they haven't used
        // it for 10 minutes (this PC's own chat keeps it while it's on).
        if !self.settings.ai_chat
            && !self.llm.needs_poll()
            && llm::worker_loaded()
            && llm::remote_idle_for().is_some_and(|d| d > Duration::from_secs(600))
        {
            llm::release_shared_worker();
        }
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
        // Full-res copies: as big as the GPU allows, at most 8192 px.
        let max_tex = ui.ctx().input(|i| i.max_texture_side).min(8192) as u32;
        self.viewer_full.set_max_edge(max_tex);
        self.viewer_full.begin_frame(ui.ctx());

        // Push the Glass theme's user-configurable background (colour + backdrop)
        // and its dark/light panel mode so the pickers update live; cheap, so
        // done every frame. Same for the AI chat's optional text-colour override.
        set_glass_config(self.settings.glass_bg, self.settings.backdrop(), self.settings.glass_light);
        crate::theme::set_chat_text(self.settings.ai_chat_text_enabled.then_some(self.settings.ai_chat_text));
        crate::theme::set_chat_icon(self.settings.ai_chat_icon_enabled.then_some(self.settings.ai_chat_icon));
        crate::theme::set_orb_color(self.settings.ai_orb_color_enabled.then_some(self.settings.ai_orb_color));

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
            if self.settings.layout == settings::Layout::Gallery && self.detail_popup.open
                && let Some(i) = self.selected {
                    self.detail_popup.open_for(&self.images[i], ui.ctx());
                }
        }

        let update_badge = self.update.badge(&self.settings);
        match top_bar::show(ui, &self.stats, self.settings.show_stats, update_badge, self.settings.ftp_enabled) {
            // In FTP mode the folder button opens the remote browser instead of
            // the input/output folder popup.
            top_bar::TopBarAction::OpenFolder(pos) => {
                if self.settings.ftp_enabled {
                    self.ftp.browser_open = !self.ftp.browser_open;
                    self.ftp.browser_just_opened = self.ftp.browser_open;
                } else {
                    self.folder_popup.open = !self.folder_popup.open;
                    self.folder_popup.just_opened = self.folder_popup.open;
                    self.folder_popup.anchor = Some(pos + egui::vec2(0.0, 8.0));
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

        // The folder button's popup: select/clear the Input (a folder or a zip
        // archive) and the Output folder used by the Move button.
        let shown_input = self.current_folder.clone().or_else(archive::active);
        match top_bar::folder_popup(
            ui.ctx(),
            &mut self.folder_popup,
            shown_input.as_deref(),
            &mut self.settings.output_folder,
        ) {
            top_bar::FolderMenuAction::LoadInput(dir) => self.load_folder(&dir),
            top_bar::FolderMenuAction::OpenZip(path) => self.prompt_for_zip(path),
            top_bar::FolderMenuAction::ClearInput => self.clear_folder(),
            top_bar::FolderMenuAction::None => {}
        }

        // Zip password prompt (opens the archive on success).
        self.show_zip_prompt(ui.ctx());

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

        // Chat switched off mid-reply (or with memory passes still due):
        // keep draining the worker so they finish and the deferred unload
        // frees the model weights (nothing else polls once the chat view and
        // Settings are closed).
        if !self.settings.ai_chat && self.llm.needs_poll() {
            self.llm.poll(ui.ctx());
        }

        // AI Chat mode replaces the panels with a full-window chat with the
        // local model (the top bar stays). Takes precedence over the layouts.
        if self.settings.ai_chat {
            // Release the centre viewer's player — like the Gallery layout, a
            // clip selected in Panels would keep playing behind the chat.
            self.video_player = None;
            self.last_video_path = None;
            // Tag images (Tools menu): the selected image on the left and the
            // Tag Manager on the right, around the chat. With no image
            // selected, only the chat shows.
            if let Some(i) = self.selected
                && i >= self.images.len()
            {
                self.selected = if self.images.is_empty() { None } else { Some(self.images.len() - 1) };
            }
            let tag_view = self.llm.tag_images && self.llm.usable();
            let current = if tag_view { self.selected.map(|i| self.images[i].clone()) } else { None };
            self.llm.tag_target = current.clone();
            if let Some(path) = current {
                let left_w = (ui.available_width() * 0.28).clamp(280.0, 560.0);
                right_details::show_tag_panel(ui, &mut self.right_state, Some(&path), &mut self.tag_manager, &self.images);
                self.sync_tag_manager_settings();
                self.ai_tag_image_panel(ui, &path, left_w);
            }
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
                self.detail_popup.open_for(&self.images[i], ui.ctx());
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
        let (search_changed, tile_action) = left_browser::show(
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

        // A tile's right-click menu action — acts on that tile directly, the
        // current selection stays put.
        match tile_action {
            left_browser::TileAction::Delete(i) => {
                if self.settings.confirm_before_delete {
                    self.pending_tile_delete = self.images.get(i).cloned();
                    self.tile_delete_skip = false; // fresh checkbox per prompt
                } else {
                    self.delete_at(i);
                }
            }
            left_browser::TileAction::Move(i) => self.move_at(i),
            left_browser::TileAction::Send(i, key) => {
                if let Some(p) = self.images.get(i).cloned() {
                    self.link.send(&key, vec![p]);
                }
            }
            left_browser::TileAction::None => {}
        }

        // The pending tile-delete's confirmation modal (same dialog the
        // Details panel uses, driven from here so no selection is needed).
        if self.pending_tile_delete.is_some() {
            match right_details::delete_confirm_dialog(
                ui.ctx(),
                "left_tile_confirm_delete",
                &mut self.tile_delete_skip,
            ) {
                Some(true) => {
                    // "Don't ask again" disables (and persists) future prompts.
                    if self.tile_delete_skip {
                        self.settings.confirm_before_delete = false;
                    }
                    // Re-resolve by path — the index may have shifted while
                    // the dialog was up.
                    if let Some(path) = self.pending_tile_delete.take()
                        && let Some(i) = self.images.iter().position(|p| *p == path)
                    {
                        self.delete_at(i);
                    }
                }
                Some(false) => self.pending_tile_delete = None,
                None => {}
            }
        }

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
        // Defensive clamp: the generator views swap `images` for their output
        // lists, and a missed selection fix-up there once left a stale index
        // that panicked here — never let that take the whole app down again.
        if let Some(i) = self.selected
            && i >= self.images.len()
        {
            self.selected = if self.images.is_empty() { None } else { Some(self.images.len() - 1) };
        }
        let current_image_path = self.selected.map(|idx| self.images[idx].as_path());
        let action = right_details::show(
            ui,
            &mut self.right_state,
            current_image_path,
            &mut self.settings.confirm_before_delete,
            &mut self.tag_manager,
            &self.images,
        );
        self.sync_tag_manager_settings();

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

        // Tag edits and undos the AI chat asked for (Tag images mode) — done
        // here because they touch the Tag Manager and the right panel.
        self.apply_ai_tag_requests(&ui.ctx().clone());

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
        self.llm.set_image_detail(self.settings.ai_image_detail);
        self.llm.detailed_vision = self.settings.ai_detailed_vision;
        self.llm.krea_prompts = self.settings.ai_krea_prompts;
        self.llm.tag_images = self.settings.ai_tag_images;
        self.llm.params = self.settings.ai_gen;
        self.llm.set_model(self.settings.ai_gemma_model);
        self.llm.set_memory_enabled(self.settings.ai_memory_enabled);
        self.llm.set_ai_chat_on(self.settings.ai_chat);
        settings::show(ui.ctx(), &mut self.settings, &mut self.update, &mut self.ftp, &mut self.llm, &mut self.link);
        // Files another PC offers over Link.
        link::offer_prompt(ui.ctx(), &mut self.link);

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
        let detail_action = gallery_detail::show(
            ui.ctx(),
            &mut self.detail_popup,
            &mut self.viewer,
            &mut self.right_state.civitai,
            &mut self.favorites,
            &mut self.settings.confirm_before_delete,
        );
        // Tags the popup saved show in the right panel too — before its Move
        // runs, or the panel's own older unsaved edit would be saved over them
        // (and a later save there or in the Tag Manager would undo them).
        if let Some((path, text)) = self.detail_popup.saved.take() {
            self.right_state.tags_written(&path, &text);
        }
        match detail_action {
            // Found again by path: the list can shift while the popup is open
            // (a crop or background removal inserts its result, a rescan…),
            // and a remembered position then named another file.
            gallery_detail::DetailAction::Move(path) => {
                if let Some(i) = self.images.iter().position(|p| *p == path) {
                    self.selected = Some(i);
                    self.move_selected();
                }
            }
            gallery_detail::DetailAction::Delete(path) => {
                if let Some(i) = self.images.iter().position(|p| *p == path) {
                    self.selected = Some(i);
                    self.delete_selected();
                }
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
        // Files deleted from Deep Scan's review list leave the browser too.
        for path in std::mem::take(&mut self.scan.deleted) {
            self.remove_path(&path);
        }
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
        if self.settings.theme != self.last_theme
            || self.settings.glass_light != self.last_glass_light
            || self.settings.glass_bg != self.last_glass_bg
        {
            self.last_theme = self.settings.theme;
            self.last_glass_light = self.settings.glass_light;
            self.last_glass_bg = self.settings.glass_bg;
            set_glass_config(self.settings.glass_bg, self.settings.backdrop(), self.settings.glass_light);
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
        eframe::set_value(storage, settings::STORAGE_KEY, self.settings.for_saving());
    }

    /// Closing the app saves a tag edit still open in the Details panel or
    /// the gallery popup, as moving to another image does — it was lost.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.right_state.save_pending_edit();
        self.detail_popup.save_edit();
        // What the AI's memory hasn't read yet waits (encrypted) for the
        // next run.
        self.llm.save_memory_for_later();
        // A Link host with background hosting on hands over to a background
        // copy of the app.
        self.link.on_exit(&self.settings);
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
    // ...but not past the window as drawn: a window that hugs its content
    // keeps its largest size as its max rect (and clip rect), so a popup that
    // got shorter left the shield over whatever was below it, taking those
    // clicks. Last frame's window rect bounds it.
    if let Some(window) = ui.ctx().memory(|m| m.area_rect(ui.layer_id().id)) {
        rect = rect.intersect(window);
    }
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

    // Bold weight for Markdown **bold** and headings in the AI chat (replies
    // and thoughts). egui has no synthetic bold (`strong()` only changes the
    // colour), so ship Ubuntu Medium: the same family and version (0.83) as
    // egui's built-in Ubuntu Light, under the same licence (assets/fonts/UFL.txt).
    // ALWAYS registered (laying out with an unknown family panics), with the
    // proportional list behind it so CJK / math / emoji still resolve.
    fonts.font_data.insert(
        "ubuntu_medium".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/Ubuntu-Medium.ttf")).into(),
    );
    let mut bold_list: Vec<String> = vec!["ubuntu_medium".into()];
    if let Some(prop) = fonts.families.get(&egui::FontFamily::Proportional) {
        bold_list.extend(prop.iter().cloned());
    }
    fonts.families.insert(egui::FontFamily::Name("bold".into()), bold_list);

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

/// [`card_frame`] for something floating over the UI (a popup, a dialog):
/// the same card with the [`FLOAT`] fill (see-through glass in the Glass
/// themes).
pub(crate) fn float_card_frame(radius: u8) -> egui::Frame {
    card_frame(radius).fill(FLOAT())
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

/// Child ui widened `gutter` px to the right, so a ScrollArea shown inside it
/// draws its floating scrollbar in the surrounding panel margin instead of on
/// top of the content — lay the content out `gutter` px narrower to keep it
/// clear of the bar. The handle is inked (the light-glass grey is invisible
/// against the panel margin), hidden while idle, and fades in on hover/scroll.
pub(crate) fn edge_scroll_ui(ui: &mut egui::Ui, gutter: f32) -> egui::Ui {
    let mut wide = ui.available_rect_before_wrap();
    wide.max.x += gutter;
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(wide));
    let scroll = &mut child.style_mut().spacing.scroll;
    scroll.foreground_color = true;
    scroll.dormant_handle_opacity = 0.0;
    scroll.active_handle_opacity = 0.9;
    child
}

/// Companion to [`edge_scroll_ui`]: the widened child never advances the
/// parent's cursor, so when more widgets follow below the scroll area (e.g.
/// in an auto-sized popup) call this after showing it — otherwise the
/// trailing widgets land on top of the list and the popup sizes without it.
pub(crate) fn edge_scroll_done(ui: &mut egui::Ui, child: &egui::Ui, gutter: f32) {
    let mut used = child.min_rect();
    used.max.x -= gutter; // the gutter overhang isn't the parent's to claim
    ui.allocate_rect(used, egui::Sense::hover());
}

/// Auto-scroll a vertical ScrollArea while the user drag-selects text past its
/// top or bottom edge — egui doesn't do this on its own, so a selection could
/// never grow beyond the visible part. Call inside the ScrollArea's closure.
/// Scrolls only while the left button is held AND the press started inside this
/// scroll viewport, with speed scaling by how far past the edge the pointer is.
pub(crate) fn drag_select_autoscroll(ui: &mut egui::Ui) {
    let clip = ui.clip_rect();
    drag_select_autoscroll_in(ui, clip);
}

/// [`drag_select_autoscroll`] for a viewport whose visible part is `clip`
/// rather than the whole of it (under Liquid Glass the AI chat's messages run
/// on under its composer, and the scrolling starts at the composer).
pub(crate) fn drag_select_autoscroll_in(ui: &mut egui::Ui, clip: egui::Rect) {
    let (down, origin, pos) = ui.input(|i| {
        (i.pointer.primary_down(), i.pointer.press_origin(), i.pointer.latest_pos())
    });
    let (Some(origin), Some(pos)) = (origin, pos) else { return };
    if !down || !clip.contains(origin) {
        return;
    }
    let speed = |overshoot: f32| (overshoot * 0.35).clamp(1.0, 25.0);
    let dy = if pos.y < clip.top() {
        speed(clip.top() - pos.y) // positive delta scrolls up (content moves down)
    } else if pos.y > clip.bottom() {
        -speed(pos.y - clip.bottom())
    } else {
        return;
    };
    ui.scroll_with_delta(egui::vec2(0.0, dy));
    ui.ctx().request_repaint(); // keep scrolling while the pointer holds still
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
    image_cache::open_oriented(src).ok().map(|img| img.to_rgba8())
}

/// Crop `src` by `frac` (fractions of the upright full image) and save it as
/// `<name>_crop.png` beside the original. Runs on a worker thread.
fn crop_to_file(src: &std::path::Path, frac: zoom::CropFraction) -> Result<PathBuf, String> {
    // `decode_full_rgba` covers common formats via `image` plus the extended
    // formats (AVIF/HEIC/RAW) through our own decoder, so cropping works on
    // everything the viewer can display.
    let full = decode_full_rgba(src)
        .map(image::DynamicImage::ImageRgba8)
        .ok_or("couldn't read the image")?;
    let (iw, ih) = (full.width(), full.height());
    if iw == 0 || ih == 0 {
        return Err("empty image".to_string());
    }
    let (iwf, ihf) = (iw as f32, ih as f32);
    let cx = (frac.x * iwf).round().clamp(0.0, iwf - 1.0) as u32;
    let cy = (frac.y * ihf).round().clamp(0.0, ihf - 1.0) as u32;
    let cw = (frac.w * iwf).round().clamp(1.0, iwf - cx as f32) as u32;
    let ch = (frac.h * ihf).round().clamp(1.0, ihf - cy as f32) as u32;
    let dest = crop_destination(src);
    full.crop_imm(cx, cy, cw, ch).save(&dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

/// `haystack` contains `needle` as whole words: the match isn't glued to a
/// letter or digit on either side ("cat" ≠ "catgirl", "scattered"; = "black
/// cat", "cat_ears"). Both lowercase.
fn contains_words(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack.match_indices(needle).any(|(i, _)| {
        let before = haystack[..i].chars().next_back();
        let after = haystack[i + needle.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

/// `path`'s file name, for messages.
pub(crate) fn file_label(path: &std::path::Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Where `img` lands when moved into `dir`: its own name when neither that
/// nor the matching `.txt` is taken there, else `name_2.ext`, `name_3.ext`, …
/// A move must never replace a file already in the folder (`rename` does,
/// silently, on every platform), and the image and its .txt keep matching
/// names.
fn move_destination(img: &std::path::Path, dir: &std::path::Path) -> Option<PathBuf> {
    let stem = img.file_stem()?.to_string_lossy().into_owned();
    let ext = img.extension().map(|e| e.to_string_lossy().into_owned());
    (1..10_000).find_map(|n| {
        let base = if n == 1 { stem.clone() } else { format!("{stem}_{n}") };
        let file = dir.join(match &ext {
            Some(e) => format!("{base}.{e}"),
            None => base.clone(),
        });
        (!file.exists() && !stem_taken(dir, &base)).then_some(file)
    })
}

/// A file named `base` exists in `dir` as a .txt or as any type that reads
/// one: then `base` is taken for a moved image — another image with that
/// name (an untagged 0001.jpg) would start reading the moved one's .txt.
/// Checked name by name (a few dozen lookups) rather than by listing the
/// folder, which took seconds for a big Output folder on a network drive.
fn stem_taken(dir: &std::path::Path, base: &str) -> bool {
    // Case-sensitive file systems: the extension's other usual spelling.
    let both_cases = !cfg!(any(windows, target_os = "macos"));
    std::iter::once("txt")
        .chain(IMAGE_EXTENSIONS.iter().copied())
        .chain(VIDEO_EXTENSIONS.iter().copied())
        .chain(EXTENDED_IMAGE_EXTENSIONS.iter().copied())
        .any(|ext| {
            dir.join(format!("{base}.{ext}")).exists()
                || (both_cases && dir.join(format!("{base}.{}", ext.to_ascii_uppercase())).exists())
        })
}

/// Two file names (or stems) that may be one file: compared ignoring case
/// (Unicode, `Éclair` = `éclair`) on every OS — Windows and macOS ignore
/// case, and so does Linux on an exFAT, NTFS or network drive. Used only
/// where a false match is harmless: a .txt kept or copied, a free name
/// skipped.
pub(crate) fn same_name(a: &std::ffi::OsStr, b: &std::ffi::OsStr) -> bool {
    a == b || a.to_string_lossy().to_lowercase() == b.to_string_lossy().to_lowercase()
}

/// `a` and `b` are the same folder, as the file system resolves them (case,
/// links); compared as written when either can't be resolved.
fn same_dir(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// A file that reads a `.txt` sidecar: every image and video type the app
/// knows, whether or not it lists it right now (AVIF / HEIC / raw count even
/// with extended formats off — they still own their .txt).
pub(crate) fn has_sidecar_extension(p: &std::path::Path) -> bool {
    let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()) else {
        return false;
    };
    IMAGE_EXTENSIONS.contains(&ext.as_str())
        || VIDEO_EXTENSIONS.contains(&ext.as_str())
        || EXTENDED_IMAGE_EXTENSIONS.contains(&ext.as_str())
}

/// Move a file, falling back to copy-then-delete where `rename` can't (the
/// target is on another drive). The copy goes to a temporary name and is
/// renamed into place only once complete, so a copy that fails part-way (a
/// full disk, a drive pulled) leaves no truncated file under the real name;
/// and if the original can't be removed after the copy, the copy is removed
/// again, so a failed move leaves no duplicate.
pub(crate) fn move_file(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    if to.exists() {
        return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "the target exists"));
    }
    // A temporary name nothing there has (a user's own `photo.png.part` is
    // never overwritten).
    let name = to.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let part = (0..100)
        .map(|n| to.with_file_name(format!(".{name}.{}-{n}.part", std::process::id())))
        .find(|p| !p.exists())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::AlreadyExists, "no free temporary name"))?;
    let copied = std::fs::copy(from, &part).and_then(|_| std::fs::rename(&part, to));
    if let Err(e) = copied {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    std::fs::remove_file(from).inspect_err(|_| {
        let _ = std::fs::remove_file(to);
    })
}

/// Another media file in `img`'s folder reads the same `.txt` (same name,
/// different extension — `cat.png` and `cat.jpg` both use `cat.txt`), so the
/// .txt must stay when `img` is moved or deleted.
fn shares_sidecar(img: &std::path::Path) -> bool {
    let (Some(dir), Some(stem)) = (img.parent(), img.file_stem()) else { return false };
    // A folder that can't be read: keep the .txt (the safe side).
    let Ok(entries) = std::fs::read_dir(dir) else { return true };
    // Names compare ignoring case (see `same_name`): keeping a .txt that
    // turns out not to be shared is harmless, losing a shared one isn't.
    entries.flatten().map(|e| e.path()).any(|p| {
        p != img
            && p.file_stem().is_some_and(|s| same_name(s, stem))
            && has_sidecar_extension(&p)
            && p.is_file()
    })
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


#[cfg(test)]
mod file_op_tests {
    use super::{move_destination, move_file, same_dir, shares_sidecar};
    use std::fs;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ctf_file_ops_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Moving never replaces a file in the target folder: a taken image name
    /// — or a taken .txt name — moves the pair to the next free `_N` name.
    #[test]
    fn move_destination_never_replaces() {
        let dir = scratch("dest");
        let img = std::path::Path::new("/somewhere/0001.png");
        assert_eq!(move_destination(img, &dir), Some(dir.join("0001.png")));
        fs::write(dir.join("0001.png"), b"x").unwrap();
        assert_eq!(move_destination(img, &dir), Some(dir.join("0001_2.png")));
        fs::write(dir.join("0001_2.txt"), b"tags").unwrap();
        assert_eq!(move_destination(img, &dir), Some(dir.join("0001_3.png")));
        // Another image with that stem (an untagged 0001_3.jpg) would start
        // reading the moved image's 0001_3.txt: that name is taken too.
        fs::write(dir.join("0001_3.jpg"), b"y").unwrap();
        assert_eq!(move_destination(img, &dir), Some(dir.join("0001_4.png")));
        // Files that don't read a .txt don't take a name.
        fs::write(dir.join("0001_4.json"), b"{}").unwrap();
        assert_eq!(move_destination(img, &dir), Some(dir.join("0001_4.png")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn move_file_moves_and_refuses_to_overwrite() {
        let dir = scratch("move");
        let (a, b) = (dir.join("a.png"), dir.join("b.png"));
        fs::write(&a, b"one").unwrap();
        move_file(&a, &b).unwrap();
        assert!(!a.exists());
        assert_eq!(fs::read(&b).unwrap(), b"one");
        // A missing source is an error, and nothing appears at the target —
        // nor a leftover temporary copy.
        assert!(move_file(&a, &dir.join("c.png")).is_err());
        assert!(!dir.join("c.png").exists());
        let names: Vec<_> = fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(names, [std::ffi::OsString::from("b.png")]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_dir_resolves_paths() {
        let dir = scratch("samedir");
        fs::create_dir_all(dir.join("sub")).unwrap();
        assert!(same_dir(&dir, &dir.join("sub").join("..")));
        assert!(!same_dir(&dir, &dir.join("sub")));
        let _ = fs::remove_dir_all(&dir);
    }

    /// `cat.png` and `cat.jpg` both read `cat.txt`; a lone image owns its own.
    #[test]
    fn a_shared_sidecar_is_detected() {
        let dir = scratch("share");
        for f in ["cat.png", "cat.jpg", "cat.txt", "dog.png", "dog.txt", "dog.json", "owl.png", "owl.heic", "owl.txt"] {
            fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(shares_sidecar(&dir.join("cat.png")));
        assert!(shares_sidecar(&dir.join("cat.jpg")));
        assert!(!shares_sidecar(&dir.join("dog.png")), "only media files count");
        // A HEIC reads its .txt even in a build (or with a setting) that
        // doesn't list it.
        assert!(shares_sidecar(&dir.join("owl.png")));
        // A folder that can't be read keeps the .txt.
        assert!(shares_sidecar(&dir.join("missing").join("x.png")));
        // Names compare ignoring case, on every OS (a Linux exFAT drive too).
        fs::write(dir.join("Éclair.png"), b"x").unwrap();
        fs::write(dir.join("éclair.jpg"), b"x").unwrap();
        assert!(shares_sidecar(&dir.join("éclair.jpg")));
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod viewer_review_tests {
    use super::contains_words;

    #[test]
    fn tag_search_matches_whole_words() {
        let tags = "1girl, black cat, cat_ears, scattered petals";
        assert!(contains_words(tags, "cat"));
        assert!(contains_words(tags, "black cat"));
        assert!(contains_words(tags, "ears"));
        assert!(!contains_words("catgirl, scattered", "cat"));
        assert!(!contains_words("1girl", "girl"));
        assert!(contains_words(tags, ""));
    }
}
