//! The settings window — opened from the top-bar gear. Holds user-tweakable
//! preferences; the relevant panels read these fields each frame. Not persisted
//! across runs (yet) — it resets to the defaults below on launch.

use eframe::egui;

use crate::theme::*;

/// Key under which the settings are saved in eframe's persistent storage.
pub const STORAGE_KEY: &str = "clarity_tagflow_settings";

/// User preferences. Lives on `ViewerApp` and is persisted across runs via
/// eframe's storage (see `main.rs`). `#[serde(default)]` lets older saved files
/// gain new fields gracefully.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Whether the settings window is currently shown (never persisted).
    #[serde(skip)]
    pub open: bool,
    /// How many neighbouring images the centre viewer decodes ahead of time,
    /// on each side, for smoother arrow-key navigation.
    pub prefetch_radius: usize,
    /// Free a thumbnail's GPU texture as soon as it scrolls off-screen (lowest
    /// memory) instead of keeping an LRU cache (smoother scroll-back).
    pub unload_offscreen_thumbs: bool,
    /// Largest height (px) a thumbnail tile can take in the browser.
    pub thumbnail_size: f32,
    /// Decode browser thumbnails at a higher resolution for crisper tiles
    /// (uses more memory and CPU).
    pub hd_thumbnails: bool,
    /// Ask for confirmation before deleting an image and its sidecar.
    pub confirm_before_delete: bool,
    /// Recognise extended image formats (AVIF / HEIC / RAW) that rely on heavy C-library
    /// decoders. Off by default. Only has an effect when the app was *built* with
    /// the matching support compiled in (the `avif` / `heic` cargo features);
    /// otherwise opening one shows a "couldn't load" notice.
    pub enable_extended_formats: bool,
    /// The last AI tagger model selected in the Tag Manager, restored on launch
    /// so the user doesn't have to re-pick it each run.
    pub last_ai_model: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            open: false,
            prefetch_radius: 1,
            unload_offscreen_thumbs: true,
            thumbnail_size: 300.0,
            hd_thumbnails: false,
            confirm_before_delete: true,
            enable_extended_formats: false,
            last_ai_model: "Select AI...".to_string(),
        }
    }
}

/// Render the settings window when it's open. Mutates `settings` in place; the
/// title-bar close button dismisses it (so does clicking the gear again).
pub fn show(ctx: &egui::Context, settings: &mut Settings) {
    if !settings.open {
        return;
    }

    let mut open = true;
    egui::Window::new("Settings")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(window_frame())
        .show(ctx, |ui| {
            // Shrink the UI smaller horizontally
            ui.set_width(260.0);

            // Force check boxes to be square with round edges.
            // This overrides heavy corner radiuses from the global theme that cause circular checkboxes.
            let square_radius = egui::CornerRadius::same(4);
            let visuals = ui.visuals_mut();
            visuals.widgets.noninteractive.corner_radius = square_radius;
            visuals.widgets.inactive.corner_radius = square_radius;
            visuals.widgets.hovered.corner_radius = square_radius;
            visuals.widgets.active.corner_radius = square_radius;
            visuals.widgets.open.corner_radius = square_radius;

            section(ui, "Viewer", |ui| {
                // Stack vertically to center nicely
                ui.label(egui::RichText::new("Prefetch radius").color(TEXT));
                ui.add(egui::Slider::new(&mut settings.prefetch_radius, 0..=3));
                hint(
                    ui,
                    "Images to decode ahead/behind the current one. Higher feels \
                     smoother when paging, but does more work per selection.",
                );
            });

            section(ui, "Browser", |ui| {
                ui.label(egui::RichText::new("Thumbnail size").color(TEXT));
                ui.add(
                    egui::Slider::new(&mut settings.thumbnail_size, 120.0..=400.0)
                        .step_by(10.0)
                        .suffix(" px"),
                );
                hint(ui, "Largest height a thumbnail tile can take in the list.");

                ui.add_space(6.0);
                ui.checkbox(
                    &mut settings.hd_thumbnails,
                    egui::RichText::new("HD thumbnails").color(TEXT),
                );
                hint(
                    ui,
                    "Decode thumbnails at a higher resolution for crisper tiles. \
                     Uses more memory and CPU, so loading and scrolling can be slower.",
                );

                ui.add_space(6.0);
                ui.checkbox(
                    &mut settings.unload_offscreen_thumbs,
                    egui::RichText::new("Unload off-screen thumbnails").color(TEXT),
                );
                hint(
                    ui,
                    "Frees thumbnail memory as you scroll; tiles re-decode when \
                     scrolled back. Turn off to cache them for instant scroll-back.",
                );
            });

            section(ui, "Files", |ui| {
                ui.checkbox(
                    &mut settings.confirm_before_delete,
                    egui::RichText::new("Confirm before deleting").color(TEXT),
                );
                hint(
                    ui,
                    "Show a confirmation dialog before deleting an image and its \
                     .txt sidecar.",
                );

                // Only shown in builds that actually compiled in a decoder for the
                // extended formats (the `avif` cargo feature). Without it the
                // toggle would do nothing useful, so it's hidden entirely.
                #[cfg(feature = "avif")]
                {
                    ui.add_space(6.0);
                    ui.checkbox(
                        &mut settings.enable_extended_formats,
                        egui::RichText::new("Enable extended formats (AVIF / HEIC / RAW)").color(TEXT),
                    );
                    hint(
                        ui,
                        "Recognise .avif, .heic, and camera raw (.dng, .arw) files. These \
                         use heavy decoders, so loading is slower.",
                    );
                }
            });

            section(ui, "About", |ui| {
                ui.label(egui::RichText::new("Clarity TagFlow").color(TEXT).strong());
                ui.label(
                    egui::RichText::new(concat!("Version ", env!("CARGO_PKG_VERSION")))
                        .color(MUTED)
                        .size(12.0),
                );
            });
        });

    // The title-bar close button flips `open` to false.
    settings.open = open;
}

/// A titled, rounded group card holding a few related controls.
fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(6.0);
    // Center the section title
    ui.vertical_centered(|ui| {
        ui.label(egui::RichText::new(title).color(MUTED).strong().size(12.0));
    });
    ui.add_space(4.0);

    egui::Frame::new()
        .fill(FIELD)
        .corner_radius(egui::CornerRadius::same(12)) // Shrunk to match the tighter UI
        .inner_margin(egui::Margin::symmetric(12, 10))
        .stroke(egui::Stroke::new(1.0, EDGE))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // Center the inner controls
            ui.vertical_centered(|ui| {
                add(ui);
            });
        });
    ui.add_space(2.0);
}

/// A small muted explanatory line, shown under a control.
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(text).color(MUTED).size(11.0));
}

/// A themed frame for the settings window body (rounded, soft drop shadow).
fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .corner_radius(egui::CornerRadius::same(16)) // Shrunk to match a more compact window
        .inner_margin(egui::Margin::same(12)) // Tighter padding
        .stroke(egui::Stroke::new(1.0, EDGE))
        .shadow(egui::epaint::Shadow {
            offset: [0, 4], // Softened shadow depth
            blur: 16,
            spread: 0,
            color: egui::Color32::from_black_alpha(140),
        })
}