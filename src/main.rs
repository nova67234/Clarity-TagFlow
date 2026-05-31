// Hide the console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::time::Duration;

use eframe::egui;
use egui::{Color32, CornerRadius, Margin, Stroke};

/// Always-available image extensions (pure-Rust decoders, no heavy deps).
/// `jfif` is just JPEG with a different extension — the decoder content-sniffs it.
const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "jfif", "gif", "bmp", "webp", "ico", "tif", "tiff",
];

/// "Extended" image extensions decoded via the heavier pure-Rust crates: AVIF
/// (avif-parse + rav1d), HEIC/HEIF (heic), and camera raw — DNG and Sony ARW —
/// (zenraw). Only recognised when the user enables them in Settings AND the app
/// was built with the `avif` feature. Defined only in such builds so a stale
/// persisted setting can't make a normal build list files it can't decode.
#[cfg(feature = "avif")]
const EXTENDED_IMAGE_EXTENSIONS: &[&str] = &["avif", "heic", "heif", "dng", "arw"];

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
mod theme {
    use crate::egui::Color32;

    pub const BG: Color32 = Color32::from_rgb(24, 24, 26);
    pub const PANEL: Color32 = Color32::from_rgb(32, 32, 34);
    pub const FIELD: Color32 = Color32::from_rgb(45, 47, 50);
    pub const TEXT: Color32 = Color32::from_rgb(235, 235, 235);
    pub const MUTED: Color32 = Color32::from_rgb(170, 170, 170);
    pub const ACCENT1: Color32 = Color32::from_rgb(64, 140, 255);

    /// Faint light edge drawn around rounded panels (TEXT @ ~8% alpha).
    pub const EDGE: Color32 = Color32::from_rgba_premultiplied(18, 18, 18, 20);
}
use theme::*;

mod ai_models;
mod ai_orb;
#[cfg(feature = "avif")]
mod avif;
mod backup;
mod image_cache;
mod left_browser;
mod right_details;
mod settings;
mod tag_manager;
mod tag_manager_settings;
mod tagger;
mod top_bar;
mod video; // embedded VLC playback (real backend only under --features vlc)

fn main() -> eframe::Result {
    // libVLC playback needs its DLLs + plugins beside the exe; build.rs stages
    // them there on Windows (libVLC finds the plugins relative to its own DLL).

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 720.0]) // Slightly wider default to accommodate both panels comfortably
            .with_min_inner_size([920.0, 460.0])
            .with_icon(load_app_icon()) // taskbar / title-bar icon
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Clarity TagFlow — Image Viewer",
        options,
        Box::new(|cc| {
            // REQUIRED: Install the image loaders so egui can parse SVG bytes
            egui_extras::install_image_loaders(&cc.egui_ctx);

            apply_theme(&cc.egui_ctx);
            let mut app = ViewerApp::default();
            // Restore saved settings (if any) from eframe's persistent storage.
            if let Some(storage) = cc.storage {
                if let Some(saved) = eframe::get_value::<settings::Settings>(storage, settings::STORAGE_KEY) {
                    app.settings = saved;
                    // Restore the last-used AI tagger model into the Tag Manager.
                    app.tag_manager.ai_model = app.settings.last_ai_model.clone();
                }
            }
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
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = FIELD; // text-edit background
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = ACCENT1.gamma_multiply(0.45);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(8);
    }
    ctx.set_visuals(v);
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
    /// CACHED list of indexes mapping into `self.images` after the search string is applied
    filtered: Vec<usize>,
    selected: Option<usize>,
    search: String,
    stats: top_bar::SystemStats,

    // Panel States
    right_state: right_details::RightPanelState,
    settings: settings::Settings,
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

    /// Embedded video player for the current selection (only ever `Some` when
    /// built with `--features vlc` and VLC starts successfully).
    video_player: Option<video::VideoPlayer>,
    /// Path the current `video_player` was started for, so we only (re)start it
    /// when the selection actually changes — and don't retry on failure.
    last_video_path: Option<PathBuf>,
}

impl Default for ViewerApp {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            current_folder: None,
            last_extended_formats: settings::Settings::default().enable_extended_formats,
            filtered: Vec::new(),
            selected: None,
            search: String::new(),
            stats: top_bar::SystemStats::default(),
            right_state: right_details::RightPanelState::default(),
            settings: settings::Settings::default(),
            backup: backup::BackupState::default(),
            tag_manager: tag_manager::TagManagerState::default(),
            // Separate large-decode gates: the viewer gets a dedicated permit so
            // the clicked image always decodes immediately (priority); the browser
            // gets its own pair so it never blocks — or is blocked by — the viewer.
            thumbs: image_cache::ImageCache::new(THUMB_MAX_EDGE, 400, false, 2),
            viewer: image_cache::ImageCache::new(2048, 8, true, 1),
            video_thumbs: video::VideoThumbs::new(),
            video_player: None,
            last_video_path: None,
        }
    }
}

impl ViewerApp {
    /// Update the cached list of filtered indices based on the search string
    fn update_filtered(&mut self) {
        let query = self.search.trim().to_lowercase();
        self.filtered = (0..self.images.len())
            .filter(|&i| query.is_empty() || file_name(&self.images[i]).to_lowercase().contains(&query))
            .collect();
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

    fn step_selection(&mut self, delta: i32) {
        if self.images.is_empty() {
            return;
        }
        let current = self.selected.unwrap_or(0) as i32;
        let next = (current + delta).rem_euclid(self.images.len() as i32);
        self.selected = Some(next as usize);
    }

    /// Delete the selected image and its `.txt` sidecar, then fix up the list.
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

        self.images.remove(idx);
        self.update_filtered(); // indices shifted — re-filter
        self.selected = if self.images.is_empty() {
            None
        } else {
            Some(idx.min(self.images.len().saturating_sub(1)))
        };
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
        egui::CentralPanel::default()
            // Match the side panels' margins (top: 0) so the viewer rises to the
            // top bar and is the same height as the left/right panels.
            .frame(egui::Frame::new().fill(BG).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
            .show_inside(ui, |ui| {
                let Some(idx) = self.selected else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Open a folder to get started\n\nClick the folder button, or drag a folder or images here",
                            )
                                .size(18.0)
                                .color(MUTED),
                        );
                    });
                    return;
                };

                let now = ui.input(|i| i.time);
                let path = self.images[idx].clone();

                // Videos play in-app via embedded libVLC (the default build).
                if is_video(&path) {
                    // Start/refresh the embedded player when the selection changes.
                    // start() only returns None in a --no-default-features build
                    // (or if libVLC fails to init), handled by the notice below.
                    if self.last_video_path.as_deref() != Some(path.as_path()) {
                        self.last_video_path = Some(path.clone());
                        self.video_player = video::VideoPlayer::start(&path, ui.ctx());
                    }
                    if let Some(player) = &mut self.video_player {
                        match player.frame(ui.ctx()) {
                            Some(tex) => show_fitted(ui, &tex, false),
                            None => {
                                ui.centered_and_justified(|ui| {
                                    ui.add(egui::Spinner::new().size(48.0).color(MUTED));
                                });
                            }
                        }
                        // Keep pulling frames, but cap the UI to ~60 Hz instead
                        // of repainting as fast as the monitor allows (e.g. 144
                        // Hz) — a full-app relayout every refresh steals CPU from
                        // decoding. New frames still wake us instantly via the
                        // player's display callback (request_repaint).
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(16));
                        return;
                    }

                    // The embedded player couldn't start (only happens in a
                    // --no-default-features build, or if libVLC fails to init at
                    // runtime). Show a plain notice — no external launcher.
                    ui.vertical_centered(|ui| {
                        let avail_h = ui.available_height();
                        ui.add_space((avail_h * 0.5 - 72.0).max(8.0));

                        let icon = egui::include_image!("../icons/video.svg");
                        ui.add(
                            egui::Image::new(icon)
                                .fit_to_exact_size(egui::vec2(84.0, 84.0))
                                .tint(MUTED),
                        );
                        ui.add_space(10.0);
                        ui.label(egui::RichText::new(file_name(&path)).color(TEXT).strong().size(15.0));
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new("Couldn't start video playback.")
                                .color(MUTED)
                                .size(13.0),
                        );
                    });
                    return;
                }

                // The selection isn't a video — release any running player so VLC
                // stops and frees the file.
                self.video_player = None;
                self.last_video_path = None;

                match self.viewer.request(&path, now) {
                    image_cache::Cached::Ready(tex) => show_fitted(ui, &tex, false),
                    image_cache::Cached::Animated(frame) => {
                        show_fitted(ui, &frame, false);
                        // Keep playing the GIF, but cap to ~60 Hz instead of
                        // repainting (and relaying out the whole app) as fast as
                        // the monitor allows — GIFs top out at 50 fps, so this is
                        // smooth while leaving CPU for decoding.
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(16));
                    }
                    image_cache::Cached::Failed => {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("Couldn't load image").color(MUTED));
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
                                    ui.add(egui::Spinner::new().size(48.0).color(MUTED));
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
        self.handle_dropped_files(ui.ctx());
        self.stats.update();

        // Apply the HD-thumbnail setting (cheap; only clears the cache on change).
        self.thumbs.set_max_edge(if self.settings.hd_thumbnails {
            THUMB_MAX_EDGE_HD
        } else {
            THUMB_MAX_EDGE
        });

        self.thumbs.begin_frame(ui.ctx());
        self.viewer.begin_frame(ui.ctx());

        // --- Toggle fullscreen on F12 ---
        if ui.input(|i| i.key_pressed(egui::Key::F12)) {
            let is_fullscreen = ui.input(|i| i.viewport().fullscreen.unwrap_or(false));
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
        }

        // Arrow keys cycle through the opened images.
        let delta = ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::ArrowDown) {
                1
            } else if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::ArrowUp) {
                -1
            } else {
                0
            }
        });
        if delta != 0 {
            self.step_selection(delta);
        }

        match top_bar::show(ui, &self.stats) {
            top_bar::TopBarAction::OpenFolder => self.open_dialog(),
            top_bar::TopBarAction::OpenSettings => self.settings.open = !self.settings.open,
            top_bar::TopBarAction::CreateBackup => self.start_backup(),
            top_bar::TopBarAction::FindIssues => {}
            top_bar::TopBarAction::None => {}
        }

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
            self.settings.thumbnail_size,
        );

        // Recompute the cached indices list ONLY if the user actually typed a letter
        if search_changed {
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
            right_details::RightPanelAction::MoveCurrent => {
                if let Some(idx) = self.selected {
                    if let Some(target_dir) = rfd::FileDialog::new().pick_folder() {
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

                        self.images.remove(idx);

                        // RECOMPUTE filter so indices match correctly after shifting
                        self.update_filtered();

                        self.selected = if self.images.is_empty() {
                            None
                        } else {
                            Some(idx.min(self.images.len().saturating_sub(1)))
                        };
                    }
                }
            }
            right_details::RightPanelAction::None => {}
        }

        // 3. Central Panel (Fills remaining space)
        self.center(ui);

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

        settings::show(ui.ctx(), &mut self.settings);

        // 5. Backup dialog (floats on top when opened from the top bar).
        self.backup.show(ui.ctx());

        // Keep the live graphs animating without busy-looping.
        ui.ctx().request_repaint_after(Duration::from_millis(250));
    }

    /// Persist settings to eframe's storage (called periodically and on exit).
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, settings::STORAGE_KEY, &self.settings);
    }
}

// ---------------------------------------------------------------------------
// Reusable painting helpers
// ---------------------------------------------------------------------------

/// Show a texture centred and scaled to fit the available space, aspect-preserved.
fn show_fitted(ui: &mut egui::Ui, tex: &egui::TextureHandle, is_loading: bool) {
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
            .corner_radius(CornerRadius::same(12));

        if is_loading {
            img = img.tint(Color32::from_gray(180));
        }

        let resp = ui.add(img);

        if is_loading {
            let spinner_rect = egui::Rect::from_center_size(resp.rect.center(), egui::vec2(48.0, 48.0));
            egui::Spinner::new().color(MUTED).paint_at(ui, spinner_rect);
        }
    });
}

/// A rounded panel with the PANEL fill, faint edge, and a soft drop shadow.
pub(crate) fn card_frame(radius: u8) -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .corner_radius(CornerRadius::same(radius))
        .inner_margin(Margin::same(12))
        .stroke(Stroke::new(1.0, EDGE))
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

pub(crate) fn file_name(p: &std::path::Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>")
        .to_owned()
}