//! The settings window — opened from the top-bar gear. Holds user-tweakable
//! preferences; the relevant panels read these fields each frame. Persisted
//! across runs via eframe's storage (see `main.rs`).

use eframe::egui;

use crate::left_panel_settings::MediaFilter;
use crate::theme::{Backdrop, Theme, EDGE, MUTED, PANEL, TEXT};

/// Key under which the settings are saved in eframe's persistent storage.
pub const STORAGE_KEY: &str = "clarity_tagflow_settings";

/// Which tab of the settings window is shown. Never persisted — resets to
/// `General` each launch.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SettingsTab {
    #[default]
    General,
    Appearance,
}

/// The overall UI layout: the classic three panels, or a full-window gallery.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Browser (left) · viewer (centre) · details (right).
    #[default]
    Panels,
    /// A full-window masonry grid of the open folder's images.
    Gallery,
}

/// User preferences. Lives on `ViewerApp` and is persisted across runs via
/// eframe's storage (see `main.rs`). `#[serde(default)]` lets older saved files
/// gain new fields gracefully.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Whether the settings window is currently shown (never persisted).
    #[serde(skip)]
    pub open: bool,
    /// Which tab is shown (never persisted).
    #[serde(skip)]
    pub tab: SettingsTab,
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
    /// The active app colour theme (Dark / Light). Applied on launch and live
    /// whenever changed from the Appearance tab.
    pub theme: Theme,
    /// The overall UI layout (classic panels, or full-window gallery).
    pub layout: Layout,
    /// Background colour (sRGB) for the Glass theme — painted behind its
    /// translucent panels. Independent of the panel colours, so changing it
    /// recolours the background without restyling the glass.
    pub glass_bg: [u8; 3],
    /// Which animated backdrop the Glass theme paints over `glass_bg`.
    pub glass_backdrop: Backdrop,
    /// Loop videos: restart playback from the beginning when a video reaches its
    /// end. Read by the embedded video player when a clip starts.
    pub loop_video: bool,
    /// Show the live CPU / RAM graphs in the top bar. Off gives a cleaner bar (and
    /// skips the periodic system sampling).
    pub show_stats: bool,
    /// Which media type the browser is narrowed to (Filter tab). Not persisted —
    /// resets to `All` each launch, matching the Java filter dialog, so a stored
    /// "Favorites" can't make the browser look empty after a restart.
    #[serde(skip)]
    pub media_filter: MediaFilter,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            open: false,
            tab: SettingsTab::General,
            prefetch_radius: 1,
            unload_offscreen_thumbs: true,
            thumbnail_size: 300.0,
            hd_thumbnails: false,
            confirm_before_delete: true,
            enable_extended_formats: false,
            last_ai_model: "Select AI...".to_string(),
            theme: Theme::default(),
            layout: Layout::default(),
            // A deep navy reads well behind the glass panels by default.
            glass_bg: [20, 22, 34],
            glass_backdrop: Backdrop::default(),
            loop_video: false,
            show_stats: true,
            media_filter: MediaFilter::default(),
        }
    }
}

/// Render the settings window when it's open. Mutates `settings` in place; the
/// title-bar close button dismisses it (so does clicking the gear again).
pub fn show(ctx: &egui::Context, settings: &mut Settings) {
    if !settings.open {
        return;
    }

    let mut want_close = false;
    egui::Window::new("Settings")
        .id(egui::Id::new("settings_window"))
        .title_bar(false) // custom header inside (matches the other popups)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(window_frame())
        .show(ctx, |ui| {
            ui.set_width(360.0);
            ui.set_max_width(360.0);

            // Square-but-rounded checkboxes (the global theme rounds them into
            // pills otherwise).
            let square_radius = egui::CornerRadius::same(4);
            let visuals = ui.visuals_mut();
            visuals.widgets.noninteractive.corner_radius = square_radius;
            visuals.widgets.inactive.corner_radius = square_radius;
            visuals.widgets.hovered.corner_radius = square_radius;
            visuals.widgets.active.corner_radius = square_radius;
            visuals.widgets.open.corner_radius = square_radius;

            // Title row: settings icon + "Settings" + close.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/settings.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(TEXT()),
                );
                ui.heading(egui::RichText::new("Settings").color(TEXT()).strong().size(17.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(egui::RichText::new("✕").size(14.0)).frame(false))
                        .on_hover_text("Close")
                        .clicked()
                    {
                        want_close = true;
                    }
                });
            });
            ui.add_space(12.0);

            // Tabs.
            ui.horizontal(|ui| {
                tab_button(ui, settings, SettingsTab::General, "General");
                tab_button(ui, settings, SettingsTab::Appearance, "Appearance");
            });
            ui.add_space(8.0);

            // Scroll the tab body so a long tab doesn't make the window tall; the
            // header + tabs stay pinned.
            let max_h = (ui.ctx().content_rect().height() - 140.0).clamp(240.0, 460.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .max_height(max_h)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    match settings.tab {
                        SettingsTab::General => general_tab(ui, settings),
                        SettingsTab::Appearance => appearance_tab(ui, settings),
                    }
                });
        });

    settings.open = !want_close;
}

/// The General tab: the original viewer / browser / files / about sections.
fn general_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    section(ui, "Interface", |ui| {
        ui.checkbox(
            &mut settings.show_stats,
            egui::RichText::new("Show CPU / RAM stats").color(TEXT()),
        );
        hint(
            ui,
            "The live CPU and memory graphs in the top bar. Turn off for a cleaner \
             bar (and to skip the periodic system sampling).",
        );
    });

    section(ui, "Viewer", |ui| {
        // Stack vertically to center nicely
        ui.label(egui::RichText::new("Prefetch radius").color(TEXT()));
        ui.add(egui::Slider::new(&mut settings.prefetch_radius, 0..=3));
        hint(
            ui,
            "Images to decode ahead/behind the current one. Higher feels \
             smoother when paging, but does more work per selection.",
        );
    });

    section(ui, "Browser", |ui| {
        ui.label(egui::RichText::new("Thumbnail size").color(TEXT()));
        ui.add(
            egui::Slider::new(&mut settings.thumbnail_size, 120.0..=400.0)
                .step_by(10.0)
                .suffix(" px"),
        );
        hint(ui, "Largest height a thumbnail tile can take in the list.");

        ui.add_space(6.0);
        ui.checkbox(
            &mut settings.hd_thumbnails,
            egui::RichText::new("HD thumbnails").color(TEXT()),
        );
        hint(
            ui,
            "Decode thumbnails at a higher resolution for crisper tiles. \
             Uses more memory and CPU, so loading and scrolling can be slower.",
        );

        ui.add_space(6.0);
        ui.checkbox(
            &mut settings.unload_offscreen_thumbs,
            egui::RichText::new("Unload off-screen thumbnails").color(TEXT()),
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
            egui::RichText::new("Confirm before deleting").color(TEXT()),
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
                egui::RichText::new("Enable extended formats (AVIF / HEIC / RAW)").color(TEXT()),
            );
            hint(
                ui,
                "Recognise .avif, .heic, and camera raw (.dng, .arw, .cr2, .nef) files. These \
                 use heavy decoders, so loading is slower.",
            );
        }
    });

    section(ui, "Video", |ui| {
        ui.checkbox(
            &mut settings.loop_video,
            egui::RichText::new("Loop videos").color(TEXT()),
        );
        hint(
            ui,
            "Restart a video from the beginning when it reaches the end. \
             Applies the next time a video starts playing.",
        );
    });

    section(ui, "About", |ui| {
        ui.label(egui::RichText::new("Clarity TagFlow").color(TEXT()).strong());
        ui.label(
            egui::RichText::new(concat!("Version ", env!("CARGO_PKG_VERSION")))
                .color(MUTED())
                .size(12.0),
        );
    });
}

/// The Appearance tab: pick the app colour theme. Changing it updates
/// `settings.theme`; `main.rs` applies the new palette live next frame.
fn appearance_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    section(ui, "Layout", |ui| {
        ui.radio_value(
            &mut settings.layout,
            Layout::Panels,
            egui::RichText::new("Panels").color(TEXT()),
        );
        ui.add_space(2.0);
        ui.radio_value(
            &mut settings.layout,
            Layout::Gallery,
            egui::RichText::new("Gallery").color(TEXT()),
        );
        hint(
            ui,
            "Panels is the classic browser · viewer · details layout. Gallery hides \
             the panels and shows every image in the open folder as a grid — click \
             one to open it back in Panels view.",
        );
    });

    section(ui, "Theme", |ui| {
        ui.label(egui::RichText::new("Colour theme").color(TEXT()));
        ui.add_space(6.0);
        ui.radio_value(
            &mut settings.theme,
            Theme::Dark,
            egui::RichText::new("Dark").color(TEXT()),
        );
        ui.add_space(2.0);
        ui.radio_value(
            &mut settings.theme,
            Theme::Light,
            egui::RichText::new("Light").color(TEXT()),
        );
        ui.add_space(2.0);
        ui.radio_value(
            &mut settings.theme,
            Theme::Space,
            egui::RichText::new("Space").color(TEXT()),
        );
        ui.add_space(2.0);
        ui.radio_value(
            &mut settings.theme,
            Theme::Aurora,
            egui::RichText::new("Aurora").color(TEXT()),
        );
        ui.add_space(2.0);
        ui.radio_value(
            &mut settings.theme,
            Theme::Glass,
            egui::RichText::new("Glass").color(TEXT()),
        );
        hint(
            ui,
            "Dark and Light recolour the whole app. Space is dark with an animated \
             starfield, and Aurora is light with a soft drifting glow behind the \
             panels and image. Glass has translucent dark panels over a background \
             you choose below. Applies instantly.",
        );
    });

    // Background controls for the Glass theme: a colour picker plus the animated
    // backdrop. Only shown when Glass is the active theme, since they don't apply
    // to the other themes.
    if settings.theme == Theme::Glass {
        section(ui, "Glass background", |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Colour").color(TEXT()));
                ui.add_space(6.0);

                // A pill-shaped colour swatch. egui's own colour button hard-caps
                // its corner radius at 2px (the alpha checker grid can't round), so
                // we paint our own pill and open the picker in a popup on click.
                let mut col = {
                    let [r, g, b] = settings.glass_bg;
                    egui::Color32::from_rgb(r, g, b)
                };
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(46.0, 18.0), egui::Sense::click());
                let radius = rect.height() / 2.0;
                ui.painter().rect_filled(rect, radius, col);
                ui.painter().rect_stroke(
                    rect,
                    radius,
                    egui::Stroke::new(1.0, EDGE()),
                    egui::StrokeKind::Inside,
                );
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                egui::Popup::from_toggle_button_response(&resp).show(|ui| {
                    ui.set_min_width(220.0);
                    if egui::widgets::color_picker::color_picker_color32(
                        ui,
                        &mut col,
                        egui::widgets::color_picker::Alpha::Opaque,
                    ) {
                        settings.glass_bg = [col.r(), col.g(), col.b()];
                    }
                });
            });
            hint(ui, "Shows through the translucent panels and fills the gutters.");

            ui.add_space(8.0);
            ui.label(egui::RichText::new("Backdrop").color(TEXT()));
            ui.add_space(4.0);
            ui.radio_value(
                &mut settings.glass_backdrop,
                Backdrop::Solid,
                egui::RichText::new("Solid").color(TEXT()),
            );
            ui.add_space(2.0);
            ui.radio_value(
                &mut settings.glass_backdrop,
                Backdrop::Starfield,
                egui::RichText::new("Starfield").color(TEXT()),
            );
            ui.add_space(2.0);
            ui.radio_value(
                &mut settings.glass_backdrop,
                Backdrop::Aurora,
                egui::RichText::new("Aurora glow").color(TEXT()),
            );
            hint(ui, "An optional animation painted over the background colour.");
        });
    }
}

/// A tab selector button across the top of the window. Highlights the active
/// tab and switches to it on click.
fn tab_button(ui: &mut egui::Ui, settings: &mut Settings, tab: SettingsTab, label: &str) {
    let selected = settings.tab == tab;
    let color = if selected { TEXT() } else { MUTED() };
    if ui
        .selectable_label(selected, egui::RichText::new(label).color(color).strong())
        .clicked()
    {
        settings.tab = tab;
    }
}

/// A flat section: an uppercase muted label with its controls directly below,
/// left-aligned (matches the Civitai / Backup popups — no bordered card).
fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
    ui.add_space(6.0);
    ui.scope(|ui| {
        ui.set_width(ui.available_width());
        add(ui);
    });
    ui.add_space(12.0);
}

/// A small muted explanatory line, shown under a control.
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(text).color(MUTED()).size(11.0));
}

/// A themed frame for the settings window body (matches the Civitai / Backup
/// popups: rounded-16 card with a soft drop shadow).
fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(egui::CornerRadius::same(16))
        .inner_margin(egui::Margin::same(18))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 6],
            blur: 20,
            spread: 0,
            color: egui::Color32::from_black_alpha(150),
        })
}
