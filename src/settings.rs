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
            });
            ui.add_space(12.0);

            // Tabs. More tabs exist than fit the window (about four show at
            // once), so the row scrolls sideways — mouse wheel over it, or
            // click a partially visible tab and it pulls itself into view.
            // The scrollbar stays hidden; muted chevrons at the edges hint at
            // the hidden tabs instead.
            let tabs_out = egui::ScrollArea::horizontal()
                .id_salt("settings_tabs")
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        tab_button(ui, settings, SettingsTab::General, "General", false);
                        tab_button(ui, settings, SettingsTab::Appearance, "Appearance", false);
                        tab_button(ui, settings, SettingsTab::AiModel, "AI Model", false);
                        tab_button(ui, settings, SettingsTab::Ftp, "FTP/FTPS", false);
                        // The Updates tab carries a small red dot when an update is waiting.
                        let update_waiting = update.badge(settings);
                        tab_button(ui, settings, SettingsTab::Updates, "Updates", update_waiting);
                    });
                });
            // Edge chevrons while content is hidden on that side.
            {
                let rect = tabs_out.inner_rect;
                let max_off = (tabs_out.content_size.x - rect.width()).max(0.0);
                let off = tabs_out.state.offset.x;
                if off > 1.0 {
                    ui.painter().text(
                        rect.left_center(),
                        egui::Align2::LEFT_CENTER,
                        "‹",
                        egui::FontId::proportional(14.0),
                        MUTED(),
                    );
                }
                if off < max_off - 1.0 {
                    ui.painter().text(
                        rect.right_center(),
                        egui::Align2::RIGHT_CENTER,
                        "›",
                        egui::FontId::proportional(14.0),
                        MUTED(),
                    );
                }
            }
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
                        SettingsTab::AiModel => ai_model_tab(ui, llm),
                        SettingsTab::Ftp => ftp_tab(ui, settings, ftp),
                        SettingsTab::Updates => crate::update::updates_tab(ui, update, settings),
                    }
                });
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
/// (src/llm.rs) plus a small "try it" prompt box. Everything runs inside the
/// app — the setup button just downloads the model weights.
fn ai_model_tab(ui: &mut egui::Ui, llm: &mut crate::llm::LlmState) {
    llm.poll();

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
        section(ui, "Try it", |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut llm.prompt)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY)
                    .margin(egui::Margin::symmetric(8, 6))
                    .hint_text("Ask anything — attach an image to ask about it"),
            );
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                if ui.button("Attach image…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"])
                        .pick_file()
                    {
                        llm.image = Some(path);
                    }
                }
                if let Some(img) = &llm.image {
                    let name = img.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    ui.label(egui::RichText::new(name).color(MUTED()).size(11.5));
                    if ui.small_button("✕").on_hover_text("Remove the image").clicked() {
                        llm.image = None;
                    }
                }
            });
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                let btn = egui::Button::new(
                    egui::RichText::new(if llm.running { "Thinking…" } else { "Ask" })
                        .color(egui::Color32::WHITE)
                        .strong(),
                )
                .fill(crate::theme::ACCENT1())
                .corner_radius(egui::CornerRadius::same(255));
                if ui
                    .add_enabled_ui(!llm.running, |ui| ui.add_sized(egui::vec2(110.0, 32.0), btn))
                    .inner
                    .clicked()
                {
                    llm.generate(ui.ctx());
                }
                if llm.running {
                    ui.add(egui::Spinner::new().size(16.0).color(MUTED()));
                    ui.label(egui::RichText::new(&llm.status).color(MUTED()).size(11.5));
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
                }
            });

            if let Some(e) = &llm.run_err {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
            }
            if !llm.response.is_empty() {
                ui.add_space(8.0);
                egui::Frame::new()
                    .fill(ui.visuals().extreme_bg_color)
                    .corner_radius(egui::CornerRadius::same(10))
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.add(egui::Label::new(egui::RichText::new(&llm.response).color(TEXT()).size(12.5)).wrap());
                    });
            }
        });
        hint(
            ui,
            "The first question loads the model into memory and can take a \
             minute; after that it stays loaded. Runs on the CPU.",
        );
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

/// A tab selector button across the top of the window. Highlights the active
/// tab and switches to it on click. When `badge` is set, a small red dot is
/// painted just after the label (vertically centred) — the "update waiting" mark.
fn tab_button(ui: &mut egui::Ui, settings: &mut Settings, tab: SettingsTab, label: &str, badge: bool) {
    let selected = settings.tab == tab;
    // Every tab is a real pill button: the active one filled with the accent
    // (white text), the others with the theme's normal button fill + hover.
    let text_color = if selected { egui::Color32::WHITE } else { TEXT() };
    let resp = ui
        .scope(|ui| {
            // Roomier padding than the default so the pills read as buttons.
            ui.spacing_mut().button_padding = egui::vec2(14.0, 6.0);
            let mut btn = egui::Button::new(egui::RichText::new(label).color(text_color).strong())
                .corner_radius(egui::CornerRadius::same(255))
                .min_size(egui::vec2(0.0, 30.0));
            if selected {
                btn = btn.fill(crate::theme::ACCENT1());
            }
            ui.add(btn)
        })
        .inner;
    if resp.clicked() {
        settings.tab = tab;
        // The tab row scrolls sideways — bring the clicked tab fully into view.
        resp.scroll_to_me(None);
    }
    if badge {
        // Reserve a small slot so the dot sits in the row (not overlapping the
        // next tab), and paint it centred on the label's vertical midline. Matches
        // the red of the top-bar gear's update badge.
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, resp.rect.height()), egui::Sense::hover());
        ui.painter()
            .circle_filled(rect.center(), 3.5, egui::Color32::from_rgb(230, 70, 70));
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
