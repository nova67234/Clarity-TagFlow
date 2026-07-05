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
    AiModel,
    Ftp,
    Updates,
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
    /// AI Chat mode: the main view swaps the three panels for a full-window
    /// chat with the local model (src/ai_chat.rs). Toggled from the AI Model
    /// tab; the top bar stays.
    pub ai_chat: bool,
    /// OmniVoice "voice design" description used by the chat's Listen buttons
    /// (gender, age, pitch, style, accent — free text).
    pub ai_voice_style: String,
    /// Auto-speak: read every finished AI reply aloud (tools menu toggle).
    pub ai_auto_speak: bool,
    /// Voice cloning: path to a short reference recording (empty = none —
    /// the description above is used instead).
    pub ai_voice_ref_audio: String,
    /// Word-for-word transcript of the reference recording (cloning quality
    /// hangs on this matching the audio).
    pub ai_voice_ref_text: String,
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
    /// Light-mode Glass: the frosted panels turn translucent white with dark-grey
    /// text/icons. Off keeps the original dark glass.
    pub glass_light: bool,
    /// Loop videos: restart playback from the beginning when a video reaches its
    /// end. Read by the embedded video player when a clip starts.
    pub loop_video: bool,
    /// Play a muted, looping live preview on each visible video thumbnail while
    /// it's in view (stops when scrolled away). Off shows only the static poster.
    pub video_thumbnail_play: bool,
    /// FTP mode: the top bar's folder button becomes a remote FTP/FTPS browser
    /// (see `src/ftp.rs`) instead of the local folder picker.
    pub ftp_enabled: bool,
    /// FTP server host (no scheme, e.g. "ftp.example.com" or "192.168.1.10").
    pub ftp_host: String,
    /// FTP control port (21 unless the server says otherwise).
    pub ftp_port: u16,
    /// FTP username; empty logs in as "anonymous".
    pub ftp_user: String,
    /// Upgrade the connection to FTPS (explicit TLS) before logging in.
    pub ftp_secure: bool,
    /// Show the live CPU / RAM graphs in the top bar. Off gives a cleaner bar (and
    /// skips the periodic system sampling).
    pub show_stats: bool,
    /// Let floating popups (Civitai settings, LoRA picker, image detail view, Find
    /// Issues) be dragged around and remember where they were left between runs.
    /// Off pins them to their original spot. Modal dialogs (Settings / Backup /
    /// delete confirm) are always fixed regardless.
    pub movable_popups: bool,
    /// Which media type the browser is narrowed to (Filter tab). Not persisted —
    /// resets to `All` each launch, matching the Java filter dialog, so a stored
    /// "Favorites" can't make the browser look empty after a restart.
    #[serde(skip)]
    pub media_filter: MediaFilter,
    /// The app release tag the user dismissed the update badge for, so the red dot
    /// stays hidden until a newer release appears. Empty = nothing dismissed.
    pub dismissed_app_version: String,
    /// Same, for the dismissed ComfyUI release tag.
    pub dismissed_comfy_version: String,
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
            ai_chat: false,
            ai_voice_style: crate::voice::DEFAULT_STYLE.to_string(),
            ai_auto_speak: false,
            ai_voice_ref_audio: String::new(),
            ai_voice_ref_text: String::new(),
            theme: Theme::default(),
            layout: Layout::default(),
            // A deep navy reads well behind the glass panels by default.
            glass_bg: [20, 22, 34],
            glass_backdrop: Backdrop::default(),
            glass_light: false,
            ftp_enabled: false,
            ftp_host: String::new(),
            ftp_port: 21,
            ftp_user: String::new(),
            ftp_secure: false,
            loop_video: false,
            video_thumbnail_play: false,
            show_stats: true,
            movable_popups: true,
            media_filter: MediaFilter::default(),
            dismissed_app_version: String::new(),
            dismissed_comfy_version: String::new(),
        }
    }
}

/// Render the settings window when it's open. Mutates `settings` in place; the
/// title-bar close button dismisses it (so does clicking the gear again).
/// `ftp` carries the FTP/FTPS tab's live state (password + connection test);
/// `llm` the AI Model tab's (setup download + inference worker).
pub fn show(
    ctx: &egui::Context,
    settings: &mut Settings,
    update: &mut crate::update::UpdateState,
    ftp: &mut crate::ftp::FtpState,
    llm: &mut crate::llm::LlmState,
) {
    if !settings.open {
        return;
    }

    let mut want_close = false;
    // One fixed window size for every tab (clamped only to the screen so it
    // still fits small displays) — the size never follows a tab's content.
    let win_h = (ctx.content_rect().height() - 160.0).clamp(320.0, 540.0);
    egui::Window::new("Settings")
        .id(egui::Id::new("settings_window"))
        .title_bar(false) // custom header inside (matches the other popups)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(window_frame())
        .fixed_size(egui::vec2(530.0, win_h))
        .show(ctx, |ui| {

            // Square-but-rounded checkboxes (the global theme rounds them into
            // pills otherwise).
            let square_radius = egui::CornerRadius::same(4);
            let visuals = ui.visuals_mut();
            visuals.widgets.noninteractive.corner_radius = square_radius;
            visuals.widgets.inactive.corner_radius = square_radius;
            visuals.widgets.hovered.corner_radius = square_radius;
            visuals.widgets.active.corner_radius = square_radius;
            visuals.widgets.open.corner_radius = square_radius;

            // Two columns: a left sidebar (gear + "Settings" title over a
            // vertical tab list) and the active tab's body on the right, with
            // the close button in its top-right corner. A hairline divider
            // between them spans the window's full height.
            let row = ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                // --- Sidebar ---
                ui.vertical(|ui| {
                    ui.set_width(136.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/settings.svg"))
                                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                .tint(TEXT()),
                        );
                        ui.heading(egui::RichText::new("Settings").color(TEXT()).strong().size(17.0));
                    });
                    ui.add_space(8.0);
                    // The underline beneath the title (as in the design sketch).
                    let w = ui.available_width() - 10.0;
                    let (line, _) = ui.allocate_exact_size(egui::vec2(w, 1.0), egui::Sense::hover());
                    ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0, EDGE()));
                    ui.add_space(10.0);

                    ui.spacing_mut().item_spacing.y = 4.0;
                    tab_button(ui, settings, SettingsTab::General, "General", false);
                    tab_button(ui, settings, SettingsTab::Appearance, "Appearance", false);
                    tab_button(ui, settings, SettingsTab::AiModel, "AI Model", false);
                    tab_button(ui, settings, SettingsTab::Ftp, "FTP/FTPS", false);
                    // The Updates tab carries a small red dot when an update is waiting.
                    let update_waiting = update.badge(settings);
                    tab_button(ui, settings, SettingsTab::Updates, "Updates", update_waiting);
                });
                ui.add_space(16.0);

                // --- Content ---
                ui.vertical(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        if ui
                            .add(egui::Button::image(
                                egui::Image::new(egui::include_image!("../icons/close.svg"))
                                    .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                    .tint(TEXT()),
                            ).frame(false))
                            .on_hover_text("Close")
                            .clicked()
                        {
                            want_close = true;
                        }
                    });
                    ui.add_space(2.0);

                    // The body fills the window's fixed height exactly (no
                    // shrinking to content) and scrolls when a tab is taller;
                    // the sidebar and close button stay pinned.
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            match settings.tab {
                                SettingsTab::General => general_tab(ui, settings),
                                SettingsTab::Appearance => appearance_tab(ui, settings),
                                SettingsTab::AiModel => ai_model_tab(ui, settings, llm),
                                SettingsTab::Ftp => ftp_tab(ui, settings, ftp),
                                SettingsTab::Updates => crate::update::updates_tab(ui, update, settings),
                            }
                        });
                });
            });

            // The sidebar/content divider, full height like the sketch.
            let rect = row.response.rect;
            ui.painter().vline(
                rect.left() + 144.0,
                rect.y_range(),
                egui::Stroke::new(1.0, EDGE()),
            );
        });

    settings.open = !want_close;
}

/// The General tab: the original viewer / browser / files / about sections.
fn general_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    section(ui, "Interface", |ui| {
        dot_toggle(ui, &mut settings.show_stats, "Show CPU / RAM stats");
        hint(
            ui,
            "The live CPU and memory graphs in the top bar. Turn off for a cleaner \
             bar (and to skip the periodic system sampling).",
        );

        ui.add_space(6.0);
        dot_toggle(ui, &mut settings.movable_popups, "Movable popups");
        hint(
            ui,
            "Let popups (Civitai settings, the LoRA picker, the image detail view, and \
             Find Issues) be dragged around — and remember where you leave them next \
             time. Turn off to keep them centred and fixed.",
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
        dot_toggle(ui, &mut settings.hd_thumbnails, "HD thumbnails");
        hint(
            ui,
            "Decode thumbnails at a higher resolution for crisper tiles. \
             Uses more memory and CPU, so loading and scrolling can be slower.",
        );

        ui.add_space(6.0);
        dot_toggle(ui, &mut settings.unload_offscreen_thumbs, "Unload off-screen thumbnails");
        hint(
            ui,
            "Frees thumbnail memory as you scroll; tiles re-decode when \
             scrolled back. Turn off to cache them for instant scroll-back.",
        );
    });

    section(ui, "Files", |ui| {
        dot_toggle(ui, &mut settings.confirm_before_delete, "Confirm before deleting");
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
            dot_toggle(ui, &mut settings.enable_extended_formats, "Enable extended formats (AVIF / HEIC / RAW)");
            hint(
                ui,
                "Recognise .avif, .heic, and camera raw (.dng, .arw, .cr2, .nef) files. These \
                 use heavy decoders, so loading is slower.",
            );
        }
    });

    section(ui, "Video", |ui| {
        dot_toggle(ui, &mut settings.loop_video, "Loop videos");
        hint(
            ui,
            "Restart a video from the beginning when it reaches the end. \
             Applies the next time a video starts playing.",
        );

        ui.add_space(6.0);
        dot_toggle(ui, &mut settings.video_thumbnail_play, "Video thumbnail play");
        hint(
            ui,
            "Play a live, muted, looping preview on each visible video thumbnail. \
             Previews keep playing while in view and stop when scrolled away. Uses \
             more CPU while playing; only a few play at once.",
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
        section(ui, "Glass panels", |ui| {
            // Dark keeps the original translucent-dark glass exactly as it is;
            // Light swaps to frosted-white panels with dark-grey text and icons.
            ui.radio_value(
                &mut settings.glass_light,
                false,
                egui::RichText::new("Dark").color(TEXT()),
            );
            ui.add_space(2.0);
            ui.radio_value(
                &mut settings.glass_light,
                true,
                egui::RichText::new("Light").color(TEXT()),
            );
            hint(
                ui,
                "Dark is the classic translucent-dark glass; Light turns the panels \
                 frosted white with dark-grey text and icons. Applies instantly.",
            );
        });

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
                    // The picker's saturation square / hue bar are sized by
                    // `slider_width`; the default leaves them much narrower than
                    // the U8/RGB header row, so the popup showed a big empty gap
                    // on the right. Widen them to fill the popup instead.
                    ui.spacing_mut().slider_width = 260.0;
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

/// The AI Model tab: one-click setup of the local Gemma 4 vision model
/// (src/llm.rs) plus the toggle that swaps the main view for the AI Chat
/// (src/ai_chat.rs). Everything runs inside the app — the setup button just
/// downloads the model weights.
fn ai_model_tab(ui: &mut egui::Ui, settings: &mut Settings, llm: &mut crate::llm::LlmState) {
    llm.poll(ui.ctx());

    section(ui, "Local model", |ui| {
        if llm.installed {
            ui.label(
                egui::RichText::new("Gemma 4 E4B (vision) — installed")
                    .color(egui::Color32::from_rgb(46, 160, 67))
                    .strong()
                    .size(12.5),
            );
        } else {
            ui.label(egui::RichText::new("Gemma 4 E4B (vision) — not set up yet").color(TEXT()).size(12.5));
        }
        hint(
            ui,
            "Google's Gemma 4 vision model, running fully inside the app. It \
             understands both text and images — no server, no account, and \
             nothing ever leaves this device.",
        );
        ui.add_space(2.0);
        // Which inference engine this exe was built with — makes it obvious
        // when a CPU-only build is running instead of the Vulkan one.
        if crate::llm::BUILT_WITH_GPU {
            ui.label(
                egui::RichText::new("Engine: GPU (Vulkan)")
                    .color(egui::Color32::from_rgb(46, 160, 67))
                    .size(11.5),
            );
        } else if crate::llm::BUILT_WITH_LLM {
            ui.label(egui::RichText::new("Engine: CPU").color(MUTED()).size(11.5));
            hint(ui, "For GPU acceleration, run the build made by scripts\\build-vulkan.cmd.");
        }

        if !crate::llm::BUILT_WITH_LLM {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("This build was compiled without the AI feature (`llm`).")
                    .color(egui::Color32::from_rgb(210, 70, 70))
                    .size(12.0),
            );
            return;
        }

        ui.add_space(6.0);
        if let Some(dl) = &llm.download {
            ui.add(
                egui::ProgressBar::new(dl.pct() as f32 / 100.0)
                    .desired_height(10.0)
                    .corner_radius(egui::CornerRadius::same(5)),
            );
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!("Downloading the model… {}%", dl.pct()))
                    .color(MUTED())
                    .size(12.0),
            );
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
        } else if !llm.installed {
            let btn = egui::Button::new(
                egui::RichText::new("Set up everything").color(egui::Color32::WHITE).strong(),
            )
            .fill(crate::theme::ACCENT1())
            .corner_radius(egui::CornerRadius::same(255));
            if ui.add_sized(egui::vec2(170.0, 32.0), btn).clicked() {
                llm.start_setup();
            }
            hint(
                ui,
                "Downloads the model weights and the vision projector (about \
                 6 GB) from HuggingFace into the models folder. You can keep \
                 using the app while it downloads.",
            );
        }
        if let Some(e) = &llm.download_err {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
        }
    });

    if llm.installed && crate::llm::BUILT_WITH_LLM {
        section(ui, "AI Chat", |ui| {
            dot_toggle(ui, &mut settings.ai_chat, "Activate AI Chat");
            hint(
                ui,
                "Swaps the main view for a full-window chat with the model — \
                 ask anything, attach images with the +, and switch between \
                 conversations with the tabs on the left. Turn it off to get \
                 the panels back.",
            );
            ui.add_space(4.0);
            hint(
                ui,
                "The first question loads the model into memory and can take \
                 a minute; after that it stays loaded.",
            );
        });

        section(ui, "Natural voice (OmniVoice)", |ui| {
            if llm.voice.installed {
                ui.label(
                    egui::RichText::new("Installed — the chat's Listen buttons use OmniVoice")
                        .color(egui::Color32::from_rgb(46, 160, 67))
                        .strong()
                        .size(12.5),
                );
                ui.add_space(6.0);
                ui.label(egui::RichText::new("Voice").color(TEXT()).size(12.5));
                ui.add_space(2.0);
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut settings.ai_voice_style)
                        .desired_width(f32::INFINITY)
                        .margin(egui::Margin::symmetric(8, 6))
                        .hint_text(crate::voice::DEFAULT_STYLE),
                );
                if resp.changed() {
                    llm.voice.style = settings.ai_voice_style.clone();
                }
                hint(
                    ui,
                    "Comma-separated voice attributes (OmniVoice accepts a fixed \
                     set): male/female · child/teenager/young adult/middle-aged/\
                     elderly · very low/low/moderate/high/very high pitch · \
                     whisper · american/british/australian/canadian/chinese/\
                     indian/japanese/korean/portuguese/russian accent. E.g. \
                     \"male, low pitch, british accent\". Takes effect on the \
                     next Listen; invalid words fall back to the default voice.",
                );

                // Voice cloning: speak in the voice of a short recording.
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Voice sample (cloning)").color(TEXT()).size(12.5));
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    if ui.button("Choose audio…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Audio", &["wav", "flac", "mp3", "ogg", "m4a"])
                            .pick_file()
                        {
                            settings.ai_voice_ref_audio = p.to_string_lossy().to_string();
                            llm.voice.ref_audio = settings.ai_voice_ref_audio.clone();
                        }
                    }
                    // The floating always-on-top recorder: capture a voice off
                    // whatever is playing (YouTube, a game) as the sample.
                    let mic = egui::Button::image(
                        egui::Image::new(egui::include_image!("../icons/mic.svg"))
                            .fit_to_exact_size(egui::vec2(15.0, 15.0))
                            .tint(crate::theme::icon_tint(TEXT())),
                    );
                    if ui
                        .add(mic)
                        .on_hover_text("Record what's playing (floating mic stays on top of other apps)")
                        .clicked()
                    {
                        llm.voice.rec.open = true;
                    }
                    if !settings.ai_voice_ref_audio.is_empty() {
                        let name = std::path::Path::new(&settings.ai_voice_ref_audio)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        ui.label(egui::RichText::new(name).color(MUTED()).size(11.5));
                        if ui.small_button("✕").on_hover_text("Remove the sample").clicked() {
                            settings.ai_voice_ref_audio.clear();
                            llm.voice.ref_audio.clear();
                        }
                    }
                });
                if !settings.ai_voice_ref_audio.is_empty() {
                    ui.add_space(4.0);
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut settings.ai_voice_ref_text)
                            .desired_width(f32::INFINITY)
                            .margin(egui::Margin::symmetric(8, 6))
                            .hint_text("Type exactly what is said in the recording"),
                    );
                    if resp.changed() {
                        llm.voice.ref_text = settings.ai_voice_ref_text.clone();
                    }
                }
                hint(
                    ui,
                    "Clone any voice from a clean 3–10 second recording of one \
                     speaker (plus its word-for-word transcript). The sample \
                     wins over the description above; remove it to go back. \
                     An unusable sample falls back automatically.",
                );
            } else {
                hint(
                    ui,
                    "A natural neural voice for the chat's Listen buttons \
                     (OmniVoice, runs fully on this PC) — great for role play \
                     or long sessions, instead of the robotic system voice.",
                );
                ui.add_space(6.0);
                if llm.voice.setting_up {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new().size(14.0).color(MUTED()));
                        ui.label(
                            egui::RichText::new(&llm.voice.setup_status).color(MUTED()).size(11.5),
                        );
                    });
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
                } else {
                    let btn = egui::Button::new(
                        egui::RichText::new("Set up voice").color(egui::Color32::WHITE).strong(),
                    )
                    .fill(crate::theme::ACCENT1())
                    .corner_radius(egui::CornerRadius::same(255));
                    if ui.add_sized(egui::vec2(150.0, 32.0), btn).clicked() {
                        llm.voice.start_setup(ui.ctx());
                    }
                    hint(
                        ui,
                        "Downloads its own Python, GPU PyTorch and the \
                         OmniVoice model (several GB — one time). Listen \
                         falls back to the system voice until this is done.",
                    );
                    if llm.voice.setup_failed {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!("Setup failed: {}", llm.voice.setup_status))
                                .color(egui::Color32::from_rgb(210, 70, 70))
                                .size(12.0),
                        );
                    }
                }
            }
        });
    }
}

/// The FTP/FTPS tab: connection details for the remote-folder browser. While
/// FTP mode is on, the top bar's folder button opens the remote browser
/// (`src/ftp.rs`) instead of the local folder picker.
fn ftp_tab(ui: &mut egui::Ui, settings: &mut Settings, ftp: &mut crate::ftp::FtpState) {
    // Round the input boxes (Host/Port/etc). The settings window pins widgets to
    // a near-square radius for its checkboxes; this tab's fields look better
    // pill-ish, and the scope keeps the override local to this tab.
    let r = egui::CornerRadius::same(10);
    let v = ui.visuals_mut();
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.corner_radius = r;

    section(ui, "FTP mode", |ui| {
        dot_toggle(ui, &mut settings.ftp_enabled, "Browse an FTP server");
        hint(
            ui,
            "Replaces the top bar's folder button with a remote browser: pick a \
             directory on the server and its images download into a local cache \
             for viewing and tagging.",
        );
    });

    section(ui, "Server", |ui| {
        let field = |ui: &mut egui::Ui, label: &str, value: &mut String, hint_text: &str, password: bool| {
            ui.label(egui::RichText::new(label).color(TEXT()).size(12.5));
            ui.add_space(2.0);
            let resp = ui.add(
                egui::TextEdit::singleline(value)
                    .password(password)
                    .desired_width(f32::INFINITY)
                    .margin(egui::Margin::symmetric(8, 6))
                    .hint_text(hint_text),
            );
            ui.add_space(6.0);
            resp
        };

        field(ui, "Host", &mut settings.ftp_host, "ftp.example.com", false);

        ui.label(egui::RichText::new("Port").color(TEXT()).size(12.5));
        ui.add_space(2.0);
        let mut port = settings.ftp_port as u32;
        ui.add(egui::DragValue::new(&mut port).range(1..=65535));
        settings.ftp_port = port as u16;
        ui.add_space(6.0);

        field(ui, "Username", &mut settings.ftp_user, "anonymous", false);

        // The password is stored encrypted (src/secret.rs), never in the plain
        // settings file — save it whenever the field changes.
        let pass = ftp.ensure_password();
        let mut pass_edit = pass.clone();
        let resp = field(ui, "Password", &mut pass_edit, "", true);
        if resp.changed() {
            *pass = pass_edit;
            crate::ftp::save_password(pass);
        }
        hint(ui, "Stored encrypted on this device.");

        ui.add_space(4.0);
        dot_toggle(ui, &mut settings.ftp_secure, "Use FTPS (TLS)");
        hint(
            ui,
            "Encrypts the connection before logging in (explicit TLS / AUTH TLS). \
             Servers that require TLS session resumption on transfers aren't \
             supported yet and will report it when listing.",
        );
    });

    section(ui, "Connection", |ui| {
        ftp.poll_test();
        ui.horizontal(|ui| {
            // A proper fixed-size accent button (like Civitai's Save), not a
            // text-hugging sliver.
            let btn = egui::Button::new(
                egui::RichText::new(if ftp.testing() { "Testing…" } else { "Test connection" })
                    .color(egui::Color32::WHITE)
                    .strong(),
            )
            .fill(crate::theme::ACCENT1())
            // A huge radius clamps to half the button height → a full pill.
            .corner_radius(egui::CornerRadius::same(255));
            if ui
                .add_enabled_ui(!ftp.testing(), |ui| ui.add_sized(egui::vec2(150.0, 32.0), btn))
                .inner
                .clicked()
            {
                let params = ftp.params(settings);
                ftp.start_test(params);
            }
            if ftp.testing() {
                ui.add(egui::Spinner::new().size(16.0).color(MUTED()));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
            }
        });
        if let Some(status) = &ftp.test_status {
            ui.add_space(4.0);
            match status {
                Ok(msg) => {
                    ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(46, 160, 67)).size(12.0));
                }
                Err(e) => {
                    ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
                }
            }
        }
    });
}

/// One row of the settings sidebar. The active tab is a full-width
/// accent-filled pill (white text); the others are quiet until hovered.
/// When `badge` is set, a small red dot sits at the row's right edge —
/// the "update waiting" mark.
fn tab_button(ui: &mut egui::Ui, settings: &mut Settings, tab: SettingsTab, label: &str, badge: bool) {
    let selected = settings.tab == tab;
    let text_color = if selected { egui::Color32::WHITE } else { TEXT() };
    let resp = ui
        .scope(|ui| {
            ui.spacing_mut().button_padding = egui::vec2(12.0, 6.0);
            let mut btn = egui::Button::new(egui::RichText::new(label).color(text_color).strong())
                .corner_radius(egui::CornerRadius::same(10))
                .min_size(egui::vec2(ui.available_width() - 10.0, 32.0));
            if selected {
                btn = btn.fill(crate::theme::ACCENT1());
            } else {
                // Quiet row: no fill until the theme's hover styling kicks in.
                btn = btn.fill(egui::Color32::TRANSPARENT);
            }
            ui.add(btn)
        })
        .inner;
    if resp.clicked() {
        settings.tab = tab;
    }
    if badge {
        // Paint the dot inside the row's right edge, on the label's vertical
        // midline. Matches the red of the top-bar gear's update badge.
        ui.painter().circle_filled(
            egui::pos2(resp.rect.right() - 12.0, resp.rect.center().y),
            3.5,
            egui::Color32::from_rgb(230, 70, 70),
        );
    }
}

/// A flat section: an uppercase muted label with its controls directly below,
/// left-aligned (matches the Civitai / Backup popups — no bordered card).
pub(crate) fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
    ui.add_space(6.0);
    ui.scope(|ui| {
        ui.set_width(ui.available_width());
        add(ui);
    });
    ui.add_space(12.0);
}

/// A boolean toggle drawn as a radio-style dot (matching the Appearance tab's
/// selectors) instead of a square checkbox tick. Clicking flips `value`.
fn dot_toggle(ui: &mut egui::Ui, value: &mut bool, label: &str) -> egui::Response {
    let mut resp = ui.radio(*value, egui::RichText::new(label).color(TEXT()));
    if resp.clicked() {
        *value = !*value;
        resp.mark_changed();
    }
    resp
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
