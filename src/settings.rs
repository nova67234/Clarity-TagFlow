//! The settings window — opened from the top-bar gear. Holds user-tweakable
//! preferences; the relevant panels read these fields each frame. Persisted
//! across runs via eframe's storage (see `main.rs`).

use eframe::egui;

use crate::left_panel_settings::MediaFilter;
use crate::theme::{Backdrop, Theme, EDGE, FLOAT, MUTED, PANEL, TEXT};

/// Key under which the settings are saved in eframe's persistent storage.
pub const STORAGE_KEY: &str = "clarity_tagflow_settings";

/// Which tab of the settings window is shown. Never persisted — resets to
/// `General` each launch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SettingsTab {
    #[default]
    General,
    Appearance,
    AiModel,
    Link,
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
    /// Tag Manager preferences (the gear popup). Values apply live and are
    /// auto-saved here — the popup has no Save/Cancel buttons.
    pub tag_manager: crate::tag_manager_settings::TagManagerSettings,
    /// AI Chat mode: the main view swaps the three panels for a full-window
    /// chat with the local model (src/ai_chat.rs). Toggled from the AI Model
    /// tab; the top bar stays.
    pub ai_chat: bool,
    /// OmniVoice "voice design" description used by the chat's Listen buttons
    /// (gender, age, pitch, style, accent — free text).
    pub ai_voice_style: String,
    /// Auto-speak: read every finished AI reply aloud (tools menu toggle).
    pub ai_auto_speak: bool,
    /// Detailed vision: the chat's newest image goes to the model as the full
    /// picture plus overlapping close-up sections (tools menu toggle).
    pub ai_detailed_vision: bool,
    /// Krea 2 prompts: the chat turns pictures and ideas into Krea 2
    /// text-to-image prompts (tools menu toggle; implies detailed vision).
    pub ai_krea_prompts: bool,
    /// Remember across chats (AI Settings → Memory): the AI keeps long-term
    /// memories (src/memory.rs) and looks them up when they're useful.
    pub ai_memory_enabled: bool,
    /// Tag images: the chat becomes a tagging view — the image, the chat and
    /// the Tag Manager — and the AI can change tags (Tools menu toggle;
    /// src/ai_tagging.rs).
    pub ai_tag_images: bool,
    /// Which local model the AI chat runs (Gemma E4B / 27B / 31B, Qwen3-VL
    /// 8B / 30B Thinking).
    pub ai_gemma_model: crate::llm::GemmaModel,
    /// Image detail for Gemma 4 (vision tokens per image). A request — the
    /// AI runs the highest level the GPU can safely handle.
    pub ai_image_detail: crate::llm::ImageDetail,
    /// AI chat sampling knobs (temperature, top-k/p, reply length) — edited
    /// from the gear in the AI Chat's Tools menu.
    pub ai_gen: crate::llm::GenParams,
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
    /// Use a custom colour for the AI chat's message text instead of the theme's
    /// text colour — for theme combinations where the chat is hard to read
    /// (e.g. light text over Aurora's bright glow). Edited in the Appearance tab.
    pub ai_chat_text_enabled: bool,
    /// The custom AI-chat text colour (sRGB), used when the switch above is on.
    pub ai_chat_text: [u8; 3],
    /// Custom tint for the icons in the AI chat's conversation area (copy,
    /// retry, listen, edit, …). The composer (message box) and icons
    /// elsewhere in the app keep the theme's tint.
    pub ai_chat_icon_enabled: bool,
    /// The custom AI-chat icon colour (sRGB), used when the switch above is on.
    pub ai_chat_icon: [u8; 3],
    /// Custom accent for the glowing AI orb (top bar, chat thinking rows).
    pub ai_orb_color_enabled: bool,
    /// The custom orb colour (sRGB), used when the switch above is on.
    pub ai_orb_color: [u8; 3],
    /// Loop videos: restart playback from the beginning when a video reaches its
    /// end. Read by the embedded video player when a clip starts.
    pub loop_video: bool,
    /// Play a muted, looping live preview on each visible video thumbnail while
    /// it's in view (stops when scrolled away). Off shows only the static poster.
    pub video_thumbnail_play: bool,
    /// The last Input folder loaded into the browser, reopened automatically
    /// on launch. Kept up to date by `load_folder` and emptied by the folder
    /// popup's "Clear Input".
    pub input_folder: String,
    /// Where the Move button (Details & Actions) sends images. Picked in the
    /// folder button's popup; empty falls back to asking with a folder dialog
    /// on every move.
    pub output_folder: String,
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
    /// Interface size multiplier (1.0 = 100%). Ignored while `ui_scale_auto` is on.
    pub ui_scale: f32,
    /// Pick the interface size from the monitor's resolution, so the app isn't
    /// tiny on large desktop screens running at 100% OS scaling.
    pub ui_scale_auto: bool,
    /// Link (Settings → Link): host this PC's models for another PC, or use
    /// another PC's (src/link.rs).
    pub link: crate::link::LinkSettings,
}

/// The interface size steps offered in Settings > Appearance.
pub const UI_SCALE_STEPS: [f32; 8] = [0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5];

/// Interface size picked for the current monitor when "Auto" is on.
///
/// The OS's own display scaling is already applied by egui, so this only adds
/// extra size when the screen is large in *OS logical* pixels, i.e. a 1440p or
/// 4K desktop monitor left at 100-125% scaling. A laptop at its usual 125-150%
/// scaling stays at 100%, so the layout you designed there is unchanged.
pub fn auto_ui_scale(ctx: &egui::Context) -> f32 {
    let (monitor, native) = ctx.input(|i| {
        let vp = i.viewport();
        (vp.monitor_size, vp.native_pixels_per_point)
    });
    let (Some(monitor), Some(native)) = (monitor, native) else {
        return 1.0;
    };
    // `monitor_size` is in points *including* our current zoom, so convert back
    // to physical pixels and then to OS logical pixels. This keeps the result
    // stable no matter what zoom is currently applied (no feedback loop).
    // Use the short side so ultrawide monitors are judged by their height.
    let physical_h = monitor.x.min(monitor.y) * ctx.pixels_per_point();
    let logical_h = physical_h / native.max(0.5);
    match logical_h {
        h if h >= 2000.0 => 1.5,  // 4K at 100%
        h if h >= 1700.0 => 1.25, // 4K at 125%
        h if h >= 1400.0 => 1.1,  // 1440p at 100%, 4K at 150%
        _ => 1.0,
    }
}

/// The interface size that should be applied right now.
pub fn effective_ui_scale(ctx: &egui::Context, settings: &Settings) -> f32 {
    let s = if settings.ui_scale_auto { auto_ui_scale(ctx) } else { settings.ui_scale };
    s.clamp(0.75, 3.0)
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
            tag_manager: Default::default(),
            ai_chat: false,
            ai_voice_style: crate::voice::DEFAULT_STYLE.to_string(),
            ai_auto_speak: false,
            ai_detailed_vision: false,
            ai_krea_prompts: false,
            ai_memory_enabled: false,
            ai_tag_images: false,
            ai_gemma_model: crate::llm::GemmaModel::default(),
            ai_image_detail: crate::llm::ImageDetail::default(),
            ai_gen: crate::llm::GenParams::default(),
            ai_voice_ref_audio: String::new(),
            ai_voice_ref_text: String::new(),
            theme: Theme::default(),
            layout: Layout::default(),
            // A deep navy reads well behind the glass panels by default.
            glass_bg: [20, 22, 34],
            glass_backdrop: Backdrop::default(),
            glass_light: false,
            ai_chat_text_enabled: false,
            // A dark ink — the useful pick for the "light text on a light
            // theme" case that motivates this setting.
            ai_chat_text: [35, 37, 42],
            ai_chat_icon_enabled: false,
            ai_chat_icon: [35, 37, 42],
            ai_orb_color_enabled: false,
            // The orb's classic light blue (its dark-theme accent).
            ai_orb_color: [64, 140, 255],
            input_folder: String::new(),
            output_folder: String::new(),
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
            ui_scale: 1.0,
            ui_scale_auto: true,
            link: crate::link::LinkSettings::default(),
        }
    }
}

/// Render the settings window when it's open. Mutates `settings` in place; the
/// title-bar close button dismisses it (so does clicking the gear again).
/// `ftp` carries the FTP/FTPS tab's live state (password + connection test);
/// `llm` the AI Model tab's (setup download + inference worker); `link` the
/// Link tab's (the running host or receiver).
pub fn show(
    ctx: &egui::Context,
    settings: &mut Settings,
    update: &mut crate::update::UpdateState,
    ftp: &mut crate::ftp::FtpState,
    llm: &mut crate::llm::LlmState,
    link: &mut crate::link::LinkState,
) {
    if !settings.open {
        return;
    }

    let mut want_close = false;
    // One fixed window size for every tab (clamped only to the screen so it
    // still fits small displays) — the size never follows a tab's content.
    let (win_w, win_h) = settings_window_size(ctx.content_rect());
    // The sections, cards and rows the tabs draw (here, in Link and in
    // Updates) take the Find Issues look while this window is drawn.
    FRESH.set(true);
    egui::Window::new("Settings")
        .id(egui::Id::new("settings_window"))
        .title_bar(false) // custom header inside (matches the other popups)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(fresh_window_frame())
        .fixed_size(egui::vec2(win_w, win_h))
        .show(ctx, |ui| {

            // Pill-ish inputs everywhere (text fields, buttons, combos) — the
            // rounded-10 look the FTP Server fields established, applied
            // window-wide so every tab's boxes match.
            let radius = egui::CornerRadius::same(10);
            let visuals = ui.visuals_mut();
            visuals.widgets.noninteractive.corner_radius = radius;
            visuals.widgets.inactive.corner_radius = radius;
            visuals.widgets.hovered.corner_radius = radius;
            visuals.widgets.active.corner_radius = radius;
            visuals.widgets.open.corner_radius = radius;

            // Two columns: a left sidebar (gear + "Settings" title over a
            // vertical tab list) and the active tab's body on the right, with
            // the close button in its top-right corner. A hairline divider
            // between them spans the window's full height.
            let row = ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                // --- Sidebar ---
                ui.vertical(|ui| {
                    ui.set_width(136.0);
                    // The title as Find Issues has it: bold, a muted icon
                    // beside it.
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/settings.svg"))
                                .fit_to_exact_size(egui::vec2(17.0, 17.0))
                                .tint(crate::theme::icon_tint(MUTED())),
                        );
                        ui.label(egui::RichText::new("Settings").color(TEXT()).font(bold(19.0)));
                    });
                    ui.add_space(10.0);
                    // The underline beneath the title (as in the design sketch).
                    let w = ui.available_width() - 10.0;
                    let (line, _) = ui.allocate_exact_size(egui::vec2(w, 1.0), egui::Sense::hover());
                    ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0, hairline()));
                    ui.add_space(12.0);

                    ui.spacing_mut().item_spacing.y = 4.0;
                    tab_button(ui, settings, SettingsTab::General, "General", false);
                    tab_button(ui, settings, SettingsTab::Appearance, "Appearance", false);
                    tab_button(ui, settings, SettingsTab::AiModel, "AI Model", false);
                    tab_button(ui, settings, SettingsTab::Link, "Link", false);
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
                    // Push the scrollbar into the window's margin so it rides
                    // the edge instead of sitting on the controls.
                    const SCROLL_GUTTER: f32 = 12.0;
                    let mut scroll_ui = crate::edge_scroll_ui(ui, SCROLL_GUTTER);
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(&mut scroll_ui, |ui| {
                            ui.set_width(ui.available_width() - SCROLL_GUTTER);
                            // Scope widget ids by tab: switches at the same
                            // layout position in two tabs would otherwise share
                            // an auto-id, and the animation state left behind by
                            // one tab's toggle plays back on the other's.
                            ui.push_id(settings.tab, |ui| match settings.tab {
                                SettingsTab::General => general_tab(ui, settings),
                                SettingsTab::Appearance => appearance_tab(ui, settings),
                                SettingsTab::AiModel => ai_model_tab(ui, settings, llm),
                                SettingsTab::Link => crate::link::settings_tab(ui, settings, link),
                                SettingsTab::Ftp => ftp_tab(ui, settings, ftp),
                                SettingsTab::Updates => crate::update::updates_tab(ui, update, settings),
                            });
                        });
                    crate::edge_scroll_done(ui, &scroll_ui, SCROLL_GUTTER);
                });
            });

            // The sidebar/content divider, full height like the sketch.
            let rect = row.response.rect;
            ui.painter().vline(
                rect.left() + 144.0,
                rect.y_range(),
                egui::Stroke::new(1.0, hairline()),
            );
        });
    FRESH.set(false);

    settings.open = !want_close;
}

/// The General tab: the original viewer / browser / files / about sections.
fn general_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    section(ui, "Interface", |ui| {
        row(
            ui,
            "Show CPU / RAM stats",
            Some("Live graphs in the top bar. Off is cleaner and skips the sampling."),
            |ui| {
                switch(ui, &mut settings.show_stats);
            },
        );
        row_sep(ui);
        row(
            ui,
            "Movable popups",
            Some("Drag popups around; they remember where you leave them."),
            |ui| {
                switch(ui, &mut settings.movable_popups);
            },
        );
    });

    section(ui, "Viewer", |ui| {
        row(
            ui,
            "Prefetch radius",
            Some("Images decoded ahead/behind the current one — smoother paging."),
            |ui| {
                ui.spacing_mut().slider_width = 110.0;
                // Air between the value box and the slider rail.
                ui.spacing_mut().item_spacing.x = 10.0;
                ui.add(egui::Slider::new(&mut settings.prefetch_radius, 0..=3));
            },
        );
    });

    section(ui, "Browser", |ui| {
        row(
            ui,
            "Thumbnail size",
            Some("Largest height a tile can take in the list."),
            |ui| {
                ui.spacing_mut().slider_width = 90.0;
                // Air between the value box and the slider rail.
                ui.spacing_mut().item_spacing.x = 10.0;
                ui.add(
                    egui::Slider::new(&mut settings.thumbnail_size, 120.0..=400.0)
                        .step_by(10.0)
                        .suffix(" px"),
                );
            },
        );
        row_sep(ui);
        row(
            ui,
            "HD thumbnails",
            Some("Crisper tiles; uses more memory and CPU."),
            |ui| {
                switch(ui, &mut settings.hd_thumbnails);
            },
        );
        row_sep(ui);
        row(
            ui,
            "Unload off-screen thumbnails",
            Some("Frees memory as you scroll; tiles re-decode when scrolled back."),
            |ui| {
                switch(ui, &mut settings.unload_offscreen_thumbs);
            },
        );
    });

    section(ui, "Files", |ui| {
        row(
            ui,
            "Confirm before deleting",
            Some("Ask before deleting an image and its .txt sidecar."),
            |ui| {
                switch(ui, &mut settings.confirm_before_delete);
            },
        );

        // Only shown in builds that actually compiled in a decoder for the
        // extended formats (the `avif` cargo feature). Without it the
        // toggle would do nothing useful, so it's hidden entirely.
        #[cfg(feature = "avif")]
        {
            row_sep(ui);
            row(
                ui,
                "Extended formats",
                Some("AVIF / HEIC / camera raw (.dng, .arw, .cr2, .nef). Heavier decoders — slower loads."),
                |ui| {
                    switch(ui, &mut settings.enable_extended_formats);
                },
            );
        }
    });

    section(ui, "Content", |ui| {
        row(
            ui,
            "Adult-content confirmation",
            Some(
                "The one-time 18+ notice shown before adult-capable download \
                 sources (Gelbooru, Danbooru, Wallhaven's NSFW purities). \
                 Reset it on a shared machine to ask again.",
            ),
            |ui| {
                if crate::age_gate::acknowledged() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        if ui.button("Reset").clicked() {
                            crate::age_gate::reset();
                        }
                        ui.label(
                            egui::RichText::new("Confirmed")
                                .color(egui::Color32::from_rgb(46, 160, 67))
                                .size(12.0),
                        );
                    });
                } else {
                    ui.label(egui::RichText::new("Not confirmed").color(MUTED()).size(12.0));
                }
            },
        );
    });

    section(ui, "Video", |ui| {
        row(
            ui,
            "Loop videos",
            Some("Restart from the beginning when a clip ends."),
            |ui| {
                switch(ui, &mut settings.loop_video);
            },
        );
        row_sep(ui);
        row(
            ui,
            "Video thumbnail play",
            Some("Muted looping previews on visible video tiles. Uses more CPU."),
            |ui| {
                switch(ui, &mut settings.video_thumbnail_play);
            },
        );
    });

    section(ui, "About", |ui| {
        row(ui, "Clarity TagFlow", None, |ui| {
            ui.label(
                egui::RichText::new(concat!("Version ", env!("CARGO_PKG_VERSION")))
                    .color(MUTED())
                    .size(12.0),
            );
        });
    });
}

/// The Appearance tab: pick the app colour theme. Changing it updates
/// `settings.theme`; `main.rs` applies the new palette live next frame.
fn appearance_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    section(ui, "Appearance", |ui| {
        row(
            ui,
            "Layout",
            Some("Gallery hides the panels and shows the folder as a grid."),
            |ui| {
                layout_dropdown(
                    ui,
                    "layout",
                    &mut settings.layout,
                    &[
                        (Layout::Panels, LayoutIcon::Panels, "Panels", "Browser, viewer & tags"),
                        (Layout::Gallery, LayoutIcon::Gallery, "Gallery", "Full-window image grid"),
                    ],
                );
            },
        );
        row_sep(ui);
        row(
            ui,
            "Interface size",
            Some("Makes text, icons and panels bigger. Auto picks a size from your monitor's resolution."),
            |ui| {
                let pct = |v: f32| format!("{}%", (v * 100.0).round() as i32);
                // `None` is Auto. A size from before the steps (not one of
                // them) stays listed, so the pill shows it.
                let mut choice = if settings.ui_scale_auto {
                    None
                } else {
                    let step = UI_SCALE_STEPS.into_iter().find(|s| (settings.ui_scale - s).abs() < 0.001);
                    Some(step.unwrap_or(settings.ui_scale))
                };
                let auto = format!("Auto ({})", pct(auto_ui_scale(ui.ctx())));
                let mut sizes: Vec<f32> = UI_SCALE_STEPS.to_vec();
                if let Some(v) = choice.filter(|v| !UI_SCALE_STEPS.contains(v)) {
                    sizes.push(v);
                    sizes.sort_by(f32::total_cmp);
                }
                let labels: Vec<String> = sizes.iter().map(|&v| pct(v)).collect();
                let mut options = vec![(None, auto.as_str(), None, None)];
                options.extend(sizes.iter().zip(&labels).map(|(&v, l)| (Some(v), l.as_str(), None, None)));
                if pill_dropdown(ui, "ui_scale", &mut choice, &options) {
                    match choice {
                        None => settings.ui_scale_auto = true,
                        Some(v) => {
                            settings.ui_scale_auto = false;
                            settings.ui_scale = v;
                        }
                    }
                }
            },
        );
        row_sep(ui);
        row(
            ui,
            "Colour theme",
            Some("Space and Aurora are animated; Glass is translucent. Applies instantly."),
            |ui| {
                // Each theme's dot: its surface, or the colour it's known by.
                let dot = |r, g, b| Some(egui::Color32::from_rgb(r, g, b));
                pill_dropdown(
                    ui,
                    "theme",
                    &mut settings.theme,
                    &[
                        (Theme::Dark, "Dark", dot(40, 40, 43), None),
                        (Theme::Light, "Light", dot(240, 241, 244), None),
                        (Theme::Space, "Space", dot(38, 36, 92), None),
                        (Theme::Aurora, "Aurora", dot(235, 130, 175), None),
                        (Theme::Glass, "Glass", dot(110, 150, 200), None),
                    ],
                );
            },
        );
    });

    // AI chat: optional custom colours for the chat's text, its icons, and the
    // AI orb — for theme combinations where the defaults are hard to read
    // (e.g. Aurora's bright pastel glow). Picking a colour activates the
    // override; Reset returns to the theme's own colour. Applied live —
    // pushed into the theme each frame from main.rs.
    section(ui, "AI chat", |ui| {
        row(
            ui,
            "Text colour",
            Some("Your messages and the AI's replies. Pick a colour to override the theme."),
            |ui| {
                // The colour the chat uses without an override (near-black
                // over Glass's Aurora glow, else the theme's text).
                let theme_col = crate::theme::chat_aurora_ink().unwrap_or(TEXT());
                override_pill(ui, &mut settings.ai_chat_text_enabled, &mut settings.ai_chat_text, theme_col);
            },
        );
        row_sep(ui);
        row(
            ui,
            "Icon colour",
            Some("Conversation icons only — the message box keeps the theme's."),
            |ui| {
                let theme_col = crate::theme::chat_aurora_ink().unwrap_or(crate::theme::icon_tint(TEXT()));
                override_pill(ui, &mut settings.ai_chat_icon_enabled, &mut settings.ai_chat_icon, theme_col);
            },
        );
        row_sep(ui);
        row(
            ui,
            "Orb colour",
            Some("The glowing AI orb's accent, in the chat and the top bar."),
            |ui| {
                override_pill(ui, &mut settings.ai_orb_color_enabled, &mut settings.ai_orb_color, crate::theme::orb_color());
            },
        );
    });

    // Glass-only controls: panel brightness, background colour, backdrop
    // animation. Hidden for the other themes, which they don't affect.
    if settings.theme == Theme::Glass {
        section(ui, "Glass", |ui| {
            row(
                ui,
                "Panels",
                Some("Dark is the classic glass; Light is frosted white."),
                |ui| {
                    pill_dropdown(
                        ui,
                        "glass_light",
                        &mut settings.glass_light,
                        &[(false, "Dark", None, None), (true, "Light", None, None)],
                    );
                },
            );
            row_sep(ui);
            row(
                ui,
                "Background colour",
                Some("Shows through the panels and fills the gutters."),
                |ui| {
                    color_pill(ui, &mut settings.glass_bg);
                },
            );
            row_sep(ui);
            row(
                ui,
                "Backdrop",
                Some("An optional animation over the background colour."),
                |ui| {
                    pill_dropdown(
                        ui,
                        "glass_backdrop",
                        &mut settings.glass_backdrop,
                        &[
                            (Backdrop::Solid, "Solid", None, None),
                            (Backdrop::Starfield, "Starfield", None, None),
                            (Backdrop::Aurora, "Aurora glow", None, None),
                        ],
                    );
                },
            );
        });
    }
}

/// The AI Model tab: one-click setup of the local Gemma 4 vision model
/// (src/llm.rs) plus the toggle that swaps the main view for the AI Chat
/// (src/ai_chat.rs). Everything runs inside the app — the setup button just
/// downloads the model weights.
fn ai_model_tab(ui: &mut egui::Ui, settings: &mut Settings, llm: &mut crate::llm::LlmState) {
    llm.poll(ui.ctx());

    // Linked to a host that runs the chat (Settings → Link).
    if let Some((_, host)) = crate::link::remote_chat() {
        section(ui, "Runs on another PC", |ui| {
            let model = crate::link::remote_chat_model().unwrap_or_else(|| "its model".to_string());
            ui.label(
                egui::RichText::new(format!("The AI chat runs on {host}, with {model}"))
                    .color(egui::Color32::from_rgb(46, 160, 67))
                    .strong()
                    .size(12.5),
            );
            hint(
                ui,
                "Through Link (Settings → Link → Use for). Your messages and pictures go \
                 to that PC encrypted, and nothing of them is kept there. The models \
                 below are this PC's own, used when Link is off.",
            );
        });
    }

    section(ui, "Local model", |ui| {
        hint(
            ui,
            "Local vision models (Google's Gemma, Alibaba's Qwen), running \
             fully inside the app. They understand both text and images — no \
             server, no account, and nothing ever leaves this device.",
        );
        ui.add_space(6.0);

        // Variant picker: pick the size that fits the hardware; the chat
        // switches models on its next question (the old one unloads).
        // Switch rows with radio semantics — turning one on selects it and
        // turns the rest off; the active model can't be switched off directly.
        let n = crate::llm::GemmaModel::ALL.len();
        for (i, m) in crate::llm::GemmaModel::ALL.into_iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(m.label()).color(TEXT()).size(13.0));
                if m == crate::llm::GemmaModel::D31B {
                    ui.label(
                        egui::RichText::new(" Developer's choice")
                            .color(crate::theme::ACCENT1())
                            .size(10.5),
                    );
                }
                if m.installed() {
                    ui.add_space(2.0);
                    ui.add(
                        egui::Image::new(egui::include_image!("../icons/check_circle.svg"))
                            .fit_to_exact_size(egui::vec2(13.0, 13.0))
                            // Status green stays green on every theme — icon_tint
                            // would grey it out on light Glass.
                            .tint(egui::Color32::from_rgb(46, 160, 67)),
                    )
                    .on_hover_text("Installed");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut on = settings.ai_gemma_model == m;
                    if switch(ui, &mut on).changed() && on {
                        settings.ai_gemma_model = m;
                        llm.set_model(m);
                    }
                });
            });
            hint(ui, m.hint());
            if i + 1 < n {
                row_sep(ui);
            }
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
                egui::RichText::new(format!("Download {}", settings.ai_gemma_model.label()))
                    .color(egui::Color32::WHITE)
                    .strong(),
            )
            .fill(crate::theme::ACCENT1())
            .corner_radius(egui::CornerRadius::same(255));
            if ui.add_sized(egui::vec2(210.0, 32.0), btn).clicked() {
                llm.start_setup();
            }
            hint(
                ui,
                "Downloads the selected model's weights and vision projector \
                 from HuggingFace into the models folder. You can keep using \
                 the app while it downloads.",
            );
        }
        if let Some(e) = &llm.download_err {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
        }
    });

    // Image detail — Gemma 4 only (Gemma 3 has a fixed image size, Qwen
    // counts image tokens differently).
    if crate::llm::BUILT_WITH_LLM && settings.ai_gemma_model.has_image_detail() {
        section(ui, "Image detail", |ui| {
            row(
                ui,
                "Image detail",
                Some(
                    "How closely the AI reads each picture. Higher levels catch small \
                     text, jewelry and eye colour, but replies start slower and need \
                     more GPU memory — the AI lowers the level itself when the GPU \
                     can't take it. Changing it reloads the model on the next question.",
                ),
                |ui| {
                    let levels = crate::llm::ImageDetail::ALL;
                    let notes: Vec<String> = levels.iter().map(|l| format!("{} tokens", l.tokens())).collect();
                    let options: Vec<_> =
                        levels.iter().zip(&notes).map(|(&l, note)| (l, l.label(), None, Some(note.as_str()))).collect();
                    let mut level = settings.ai_image_detail;
                    if pill_dropdown(ui, "ai_image_detail", &mut level, &options) {
                        settings.ai_image_detail = level;
                        llm.set_image_detail(level);
                    }
                },
            );
            // The level actually running, when the GPU check lowered it.
            if let Some(reason) = &llm.detail_reason {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(reason)
                        .color(egui::Color32::from_rgb(214, 150, 40))
                        .size(12.0),
                );
            }
        });
    }

    let local = llm.installed && crate::llm::BUILT_WITH_LLM;
    if llm.usable() {
        section(ui, "AI Chat", |ui| {
            row(
                ui,
                "Activate AI Chat",
                Some(
                    "Swaps the main view for a full-window chat with the model — \
                     attach images with the +, switch conversations with Chats. \
                     The first question loads the model and can take a minute; \
                     switching off frees the model from memory.",
                ),
                |ui| {
                    if switch(ui, &mut settings.ai_chat).changed() {
                        if settings.ai_chat {
                            llm.cancel_unload();
                        } else {
                            llm.unload();
                        }
                    }
                },
            );
            if !local {
                return;
            }
            row_sep(ui);
            // Which inference engine this exe was built with — makes it
            // obvious when a CPU-only build is running instead of the Vulkan
            // one.
            row(
                ui,
                "Engine",
                (!crate::llm::BUILT_WITH_GPU)
                    .then_some("For GPU acceleration, run the build made by scripts\\build-vulkan.cmd."),
                |ui| {
                    if crate::llm::BUILT_WITH_GPU {
                        ui.label(
                            egui::RichText::new("GPU (Vulkan)")
                                .color(egui::Color32::from_rgb(46, 160, 67))
                                .size(13.0),
                        );
                    } else {
                        ui.label(egui::RichText::new("CPU").color(MUTED()).size(13.0));
                    }
                },
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
                let resp = pill_field(
                    ui,
                    f32::INFINITY,
                    egui::TextEdit::singleline(&mut settings.ai_voice_style).hint_text(crate::voice::DEFAULT_STYLE),
                );
                if resp.changed() {
                    llm.voice.style.clone_from(&settings.ai_voice_style);
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
                    if ui.button("Choose audio…").clicked()
                        && let Some(p) = rfd::FileDialog::new()
                            .add_filter("Audio", &["wav", "flac", "mp3", "ogg", "m4a"])
                            .pick_file()
                        {
                            settings.ai_voice_ref_audio = p.to_string_lossy().to_string();
                            llm.voice.ref_audio.clone_from(&settings.ai_voice_ref_audio);
                        }
                    // The floating always-on-top recorder: capture a voice off
                    // whatever is playing (YouTube, a game) as the sample.
                    ui.add_space(6.0);
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
                    let resp = pill_field(
                        ui,
                        f32::INFINITY,
                        egui::TextEdit::singleline(&mut settings.ai_voice_ref_text)
                            .hint_text("Type exactly what is said in the recording"),
                    );
                    if resp.changed() {
                        llm.voice.ref_text.clone_from(&settings.ai_voice_ref_text);
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
    section(ui, "FTP mode", |ui| {
        row(
            ui,
            "Browse an FTP server",
            Some(
                "Replaces the folder button with a remote browser; images \
                 download into a local cache for viewing and tagging.",
            ),
            |ui| {
                switch(ui, &mut settings.ftp_enabled);
            },
        );
    });

    section(ui, "Server", |ui| {
        let field = |ui: &mut egui::Ui, value: &mut String, hint_text: &str, password: bool| {
            pill_field(
                ui,
                ROW_CONTROL_W - 10.0,
                egui::TextEdit::singleline(value).password(password).hint_text(hint_text),
            )
        };

        row(ui, "Host", None, |ui| {
            field(ui, &mut settings.ftp_host, "ftp.example.com", false);
        });
        row_sep(ui);
        row(ui, "Port", None, |ui| {
            let mut port = settings.ftp_port as u32;
            ui.add(egui::DragValue::new(&mut port).range(1..=65535));
            settings.ftp_port = port as u16;
        });
        row_sep(ui);
        row(ui, "Username", None, |ui| {
            field(ui, &mut settings.ftp_user, "anonymous", false);
        });
        row_sep(ui);
        row(ui, "Password", Some("Stored encrypted on this device."), |ui| {
            // The password is stored encrypted (src/secret.rs), never in the
            // plain settings file — save it whenever the field changes.
            let pass = ftp.ensure_password();
            let mut pass_edit = pass.clone();
            if field(ui, &mut pass_edit, "", true).changed() {
                *pass = pass_edit;
                crate::ftp::save_password(pass);
            }
        });
        row_sep(ui);
        row(
            ui,
            "Use FTPS (TLS)",
            Some("Encrypts the connection before logging in (explicit TLS)."),
            |ui| {
                switch(ui, &mut settings.ftp_secure);
            },
        );
    });

    section(ui, "Connection", |ui| {
        ftp.poll_test();
        row(ui, "Check the server details work", None, |ui| {
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
                .add_enabled_ui(!ftp.testing(), |ui| ui.add_sized(egui::vec2(150.0, 30.0), btn))
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
    if side_tab(ui, settings.tab == tab, label, badge) {
        settings.tab = tab;
    }
}

/// One sidebar tab row of a settings-style window (this window and AI
/// Settings): accent-filled when selected, flat otherwise, with an optional
/// red dot. True when clicked.
pub(crate) fn side_tab(ui: &mut egui::Ui, selected: bool, label: &str, badge: bool) -> bool {
    let text_color = if selected { egui::Color32::WHITE } else { TEXT() };
    let fresh = fresh();
    let resp = ui
        .scope(|ui| {
            ui.spacing_mut().button_padding = egui::vec2(12.0, 6.0);
            let text = egui::RichText::new(label).color(text_color);
            let text = if fresh { text.font(bold(13.0)) } else { text.strong() };
            let mut btn = egui::Button::new(text)
                // Fresh: a full pill, like Find Issues' buttons.
                .corner_radius(egui::CornerRadius::same(if fresh { 16 } else { 10 }))
                // Flat rows in every state: light Glass strokes all buttons by
                // default, which outlined the quiet tabs. Only the selected row
                // should stand out (accent fill below).
                .stroke(egui::Stroke::NONE)
                .min_size(egui::vec2(ui.available_width() - 10.0, 32.0));
            if selected {
                btn = btn.fill(if fresh { crate::theme::selection_outline() } else { crate::theme::ACCENT1() });
            } else {
                // Quiet row: no fill until the theme's hover styling kicks in.
                btn = btn.fill(egui::Color32::TRANSPARENT);
            }
            ui.add(btn)
        })
        .inner;
    if badge {
        // Paint the dot inside the row's right edge, on the label's vertical
        // midline. Matches the red of the top-bar gear's update badge.
        ui.painter().circle_filled(
            egui::pos2(resp.rect.right() - 12.0, resp.rect.center().y),
            3.5,
            egui::Color32::from_rgb(230, 70, 70),
        );
    }
    resp.clicked()
}

/// A settings group, macOS System Settings style: an uppercase muted header
/// above a rounded card (PANEL-flat with a faint edge) holding the controls —
/// typically [`row`]s separated by [`row_sep`] hairlines.
pub(crate) fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(4.0);
    if fresh() {
        // Small, bold and spaced out, as over Find Issues' cards.
        ui.label(
            egui::RichText::new(title.to_uppercase()).color(MUTED()).font(bold(9.5)).extra_letter_spacing(1.1),
        );
        ui.add_space(8.0);
    } else {
        ui.label(egui::RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
        ui.add_space(5.0);
    }
    card(ui, add);
}

/// A [`section`]'s rounded card without the header line.
pub(crate) fn card(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    let frame = if fresh() {
        egui::Frame::new()
            .fill(card_fill())
            .stroke(egui::Stroke::new(1.0, hairline()))
            .corner_radius(egui::CornerRadius::same(18))
            .inner_margin(egui::Margin::same(16))
    } else {
        egui::Frame::new()
            .fill(PANEL())
            .stroke(egui::Stroke::new(1.0, EDGE()))
            .corner_radius(egui::CornerRadius::same(12))
            .inner_margin(egui::Margin::symmetric(12, 10))
    };
    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        add(ui);
    });
    ui.add_space(14.0);
}

/// Width reserved for a row's right-aligned control (switch, slider, combo…).
const ROW_CONTROL_W: f32 = 180.0;

/// One settings row: the label (with an optional muted description wrapping
/// beneath it) on the left, `control` right-aligned — macOS style.
pub(crate) fn row(ui: &mut egui::Ui, label: &str, sub: Option<&str>, control: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        let text_w = ui.available_width() - ROW_CONTROL_W;
        ui.vertical(|ui| {
            ui.set_width(text_w);
            if fresh() {
                // A bold label over a muted line, as in Find Issues.
                ui.label(egui::RichText::new(label).color(TEXT()).font(bold(13.5)));
                if let Some(s) = sub {
                    ui.add_space(3.0);
                    ui.label(egui::RichText::new(s).color(MUTED()).size(12.0));
                }
            } else {
                ui.label(egui::RichText::new(label).color(TEXT()).size(13.0));
                if let Some(s) = sub {
                    ui.add_space(1.0);
                    ui.label(egui::RichText::new(s).color(MUTED()).size(10.5));
                }
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), control);
    });
}

/// A pill-shaped colour swatch that opens egui's colour picker in a popup on
/// click — the row-control companion for colour settings. egui's own colour
/// button hard-caps its corner radius at 2px (the alpha checker grid can't
/// round), so we paint our own pill. Returns true when the colour was edited.
fn color_pill(ui: &mut egui::Ui, rgb: &mut [u8; 3]) -> bool {
    let mut col = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    let mut changed = false;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(46.0, 18.0), egui::Sense::click());
    let radius = rect.height() / 2.0;
    ui.painter().rect_filled(rect, radius, col);
    ui.painter().rect_stroke(rect, radius, egui::Stroke::new(1.0, EDGE()), egui::StrokeKind::Inside);
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    // Closes on a click outside only: egui's default closes a popup on ANY
    // click, inside it too, so clicking in the colour square or the hue bar,
    // or into a number field to type a value, shut the picker at once.
    egui::Popup::from_toggle_button_response(&resp)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            // The picker's saturation square / hue bar are sized by `slider_width`;
            // the default leaves them much narrower than the U8/RGB header row, so
            // the popup showed a big empty gap on the right. Widen them to fill it.
            ui.spacing_mut().slider_width = 260.0;
            if egui::widgets::color_picker::color_picker_color32(
                ui,
                &mut col,
                egui::widgets::color_picker::Alpha::Opaque,
            ) {
                *rgb = [col.r(), col.g(), col.b()];
                changed = true;
            }
        });
    changed
}

/// Control for an optional colour override: the pill picker is always shown —
/// editing it activates the override — and once active a small Reset button
/// appears beside it that returns to the theme's own colour. While no override
/// is active the pill tracks `theme_col` (the theme's current colour), so it
/// always shows the colour actually in use and the picker starts from it.
fn override_pill(ui: &mut egui::Ui, enabled: &mut bool, rgb: &mut [u8; 3], theme_col: egui::Color32) {
    if !*enabled {
        *rgb = [theme_col.r(), theme_col.g(), theme_col.b()];
    }
    // The row control lays out right-to-left: pill first (rightmost), Reset
    // to its left.
    if color_pill(ui, rgb) {
        *enabled = true;
    }
    if *enabled {
        ui.add_space(8.0);
        if ui
            .add(egui::Button::new(egui::RichText::new("Reset").size(11.0)))
            .on_hover_text("Back to the theme's colour")
            .clicked()
        {
            *enabled = false;
        }
    }
}

/// The hairline between two [`row`]s of a card.
pub(crate) fn row_sep(ui: &mut egui::Ui) {
    let fresh = fresh();
    let gap = if fresh { 11.0 } else { 7.0 };
    ui.add_space(gap);
    let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
    if fresh {
        ui.painter().hline((r.left() + 4.0)..=(r.right() - 4.0), r.center().y, egui::Stroke::new(1.0, hairline()));
    } else {
        ui.painter().hline(r.x_range(), r.center().y, egui::Stroke::new(1.0, EDGE()));
    }
    ui.add_space(gap);
}

/// A macOS-style sliding toggle switch. Clicking flips `on`; the knob and
/// track colour animate between states.
pub(crate) fn switch(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let (rect, mut resp) = ui.allocate_exact_size(egui::vec2(40.0, 22.0), egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool(resp.id, *on);
    let off = if crate::theme::is_light() {
        egui::Color32::from_gray(200)
    } else {
        egui::Color32::from_gray(62)
    };
    let track = mix(off, crate::theme::ACCENT1(), t);
    let radius = rect.height() / 2.0;
    ui.painter().rect_filled(rect, radius, track);
    ui.painter().rect_stroke(rect, radius, egui::Stroke::new(1.0, EDGE()), egui::StrokeKind::Inside);
    let x = egui::lerp((rect.left() + 11.0)..=(rect.right() - 11.0), t);
    ui.painter().circle_filled(egui::pos2(x, rect.center().y), 8.0, egui::Color32::WHITE);
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Linear blend from `a` to `b` by `t` (0..1).
fn mix(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    egui::Color32::from(egui::Rgba::from(a) * (1.0 - t) + egui::Rgba::from(b) * t)
}

/// A small muted explanatory line, shown under a control.
pub(crate) fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(text).color(MUTED()).size(if fresh() { 12.0 } else { 11.0 }));
}

/// A one-line text box as Find Issues' folder field has it: a sunk pill,
/// ringed in the accent colour while it's being typed in. `width` may be
/// `f32::INFINITY` (all there is).
pub(crate) fn pill_field(ui: &mut egui::Ui, width: f32, edit: egui::TextEdit<'_>) -> egui::Response {
    let h = 32.0;
    let w = if width.is_finite() { width } else { ui.available_width() };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
    // The pill goes under the text: its slot is taken first, filled in once
    // the field says whether it has focus.
    let bg = ui.painter().add(egui::Shape::Noop);
    let inner = rect.shrink2(egui::vec2(14.0, 0.0));
    let resp = ui.put(
        inner,
        edit.frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .font(egui::FontId::proportional(13.0))
            // Explicit: the Light theme's widget text is white (for its blue
            // buttons), which made typed text white on a light field.
            .text_color(TEXT())
            .vertical_align(egui::Align::Center)
            .desired_width(inner.width()),
    );
    let ring = if resp.has_focus() { crate::theme::selection_outline() } else { egui::Color32::TRANSPARENT };
    ui.painter().set(
        bg,
        egui::epaint::RectShape::new(rect, h / 2.0, well_fill(), egui::Stroke::new(1.0, ring), egui::StrokeKind::Inside),
    );
    resp
}

/// Find Issues' folder field and list: sunk below the cards (darker in dark
/// themes, greyer in light ones).
fn well_fill() -> egui::Color32 {
    if crate::theme::is_light() {
        FLOAT().lerp_to_gamma(TEXT(), 0.07)
    } else {
        FLOAT().lerp_to_gamma(egui::Color32::BLACK, 0.28)
    }
}

thread_local! {
    /// The Settings window is being drawn: [`section`], [`card`], [`row`],
    /// [`row_sep`], [`hint`] and [`side_tab`] take the Find Issues look
    /// (src/scan.rs). The other popups that use them keep theirs.
    static FRESH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn fresh() -> bool {
    FRESH.get()
}

/// Draw `f` with the Find Issues look, as the Settings window is (for other
/// popups in that style: Create Backup).
pub(crate) fn fresh_look<R>(f: impl FnOnce() -> R) -> R {
    let was = FRESH.replace(true);
    let out = f();
    FRESH.set(was);
    out
}

/// Ubuntu Medium — the app's real bold face (`strong()` only brightens).
pub(crate) fn bold(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name("bold".into()))
}

/// Find Issues' cards: the window's colour lifted a little toward the text
/// colour (lighter in dark themes, a touch greyer in light ones).
fn card_fill() -> egui::Color32 {
    FLOAT().lerp_to_gamma(TEXT(), 0.035)
}

/// Find Issues' card outlines and dividers.
fn hairline() -> egui::Color32 {
    TEXT().gamma_multiply(0.08)
}

/// The Settings (and AI Settings) window's content size for a screen of
/// `screen`: 530 × up to 540, but never bigger than the screen — at a large
/// interface size on a small display the old 320-point minimum ran the
/// window off the bottom of the screen.
pub(crate) fn settings_window_size(screen: egui::Rect) -> (f32, f32) {
    // The frame's 18-point margins plus a little room on each side.
    let room = |side: f32| (side - 48.0).max(200.0);
    let w = 530.0_f32.min(room(screen.width()));
    let h = (screen.height() - 160.0).clamp(320.0, 540.0).min(room(screen.height()));
    (w, h)
}

/// A themed frame for the settings window body (matches the Civitai / Backup
/// popups: rounded-16 card with a soft drop shadow), filled with [`FLOAT`]
/// (see-through glass in the Glass themes).
pub(crate) fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(FLOAT())
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

/// The Settings window's frame, as Find Issues has it: rounder, a softer,
/// deeper shadow and a hairline edge. Same margins as [`window_frame`], so
/// [`settings_window_size`] holds.
pub(crate) fn fresh_window_frame() -> egui::Frame {
    window_frame()
        .corner_radius(egui::CornerRadius::same(24))
        .stroke(egui::Stroke::new(1.0, hairline()))
        .shadow(egui::epaint::Shadow { offset: [0, 8], blur: 28, spread: 0, color: egui::Color32::from_black_alpha(150) })
}

// ---------------------------------------------------------------------------
// Dropdowns: a pill that opens a rounded list under it
// ---------------------------------------------------------------------------

/// The pill a dropdown shows when closed (wide enough for the longest
/// option, inside a row's control column).
const DROP_W: f32 = 176.0;
const DROP_H: f32 = 34.0;
/// A list with more to show than names (the Layout list's lines, notes) is
/// wider than its pill; it opens leftward, under the row's description.
const WIDE_LIST_W: f32 = 214.0;

/// Which little diagram a [`layout_dropdown`] option shows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutIcon {
    /// Three columns: browser, viewer, tags.
    Panels,
    /// A grid of tiles.
    Gallery,
}

/// The Layout picker: a pill showing the current layout with an outline of
/// it, opening a list where each layout has a small diagram, its name and a
/// line on what it is. True when the choice changed.
pub(crate) fn layout_dropdown<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id_salt: &str,
    current: &mut T,
    options: &[(T, LayoutIcon, &str, &str)], // (value, diagram, name, line)
) -> bool {
    let Some(&(_, icon, title, _)) = options.iter().find(|o| o.0 == *current).or(options.first()) else {
        return false;
    };
    let id = egui::Id::new(("settings_dropdown", id_salt));
    let resp = dropdown_pill(ui, id, |p, rect, ink| {
        let glyph = egui::Rect::from_center_size(egui::pos2(rect.left() + 21.0, rect.center().y), egui::vec2(15.0, 12.0));
        paint_layout_glyph(p, glyph, icon, ink);
        p.text(
            egui::pos2(rect.left() + 37.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::proportional(13.0),
            TEXT(),
        );
    });

    let mut changed = false;
    dropdown_list(&resp, id, WIDE_LIST_W, |ui| {
        for &(value, icon, title, line) in options {
            let selected = *current == value;
            let (rect, item) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 44.0), egui::Sense::click());
            let item = item.on_hover_cursor(egui::CursorIcon::PointingHand);
            if ui.is_rect_visible(rect) {
                let p = ui.painter();
                paint_option_card(p, rect, 11, selected, item.hovered());
                let badge = egui::Rect::from_center_size(egui::pos2(rect.left() + 24.0, rect.center().y), egui::vec2(30.0, 22.0));
                paint_layout_badge(p, badge, icon, selected);
                let x = rect.left() + 48.0;
                let title_ink = if selected { TEXT() } else { TEXT().gamma_multiply(0.9) };
                p.text(egui::pos2(x, rect.top() + 15.0), egui::Align2::LEFT_CENTER, title, bold(12.5), title_ink);
                p.text(
                    egui::pos2(x, rect.top() + 30.5),
                    egui::Align2::LEFT_CENTER,
                    line,
                    egui::FontId::proportional(11.0),
                    MUTED(),
                );
                if selected {
                    paint_checkmark(p, egui::pos2(rect.right() - 16.0, rect.center().y), drop_accent_ink());
                }
            }
            if item.clicked() {
                changed |= *current != value;
                *current = value;
                ui.close();
            }
        }
    });
    changed
}

/// A dropdown in the same style for a plain list ("Interface size", "Colour
/// theme"). An option's colour, when it has one, shows as a dot before its
/// name; its note ("280 tokens"), when it has one, shows muted at the end of
/// its row in the list, while the pill shows just the name. True when the
/// choice changed.
pub(crate) fn pill_dropdown<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id_salt: &str,
    current: &mut T,
    options: &[(T, &str, Option<egui::Color32>, Option<&str>)], // (value, name, dot, note)
) -> bool {
    let Some(&(_, label, dot, _)) = options.iter().find(|o| o.0 == *current).or(options.first()) else {
        return false;
    };
    let id = egui::Id::new(("settings_dropdown", id_salt));
    let resp = dropdown_pill(ui, id, |p, rect, _| {
        let mut x = rect.left() + 15.0;
        if let Some(c) = dot {
            paint_dot(p, egui::pos2(rect.left() + 19.0, rect.center().y), c);
            x = rect.left() + 32.0;
        }
        p.text(egui::pos2(x, rect.center().y), egui::Align2::LEFT_CENTER, label, egui::FontId::proportional(13.0), TEXT());
    });

    let mut changed = false;
    let width = if options.iter().any(|o| o.3.is_some()) { WIDE_LIST_W } else { DROP_W };
    dropdown_list(&resp, id, width, |ui| {
        for &(value, label, dot, note) in options {
            let item = dropdown_option(ui, label, dot, note, *current == value);
            if item.clicked() {
                changed |= *current != value;
                *current = value;
                ui.close();
            }
        }
    });
    changed
}

/// The closed dropdown: a sunk pill (the pill text fields' well) with
/// `content` painted inside and a chevron that turns up while it's open, the
/// pill then ringed in the accent colour with a soft halo. `content` gets
/// the colour for icons (the accent while open).
fn dropdown_pill(
    ui: &mut egui::Ui,
    id: egui::Id,
    content: impl FnOnce(&egui::Painter, egui::Rect, egui::Color32),
) -> egui::Response {
    let open = egui::Popup::is_id_open(ui.ctx(), id);
    let t = ui.ctx().animate_bool(id.with("chevron"), open);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(DROP_W, DROP_H), egui::Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        let r = DROP_H / 2.0;
        let hot = resp.hovered() || resp.has_focus();
        if t > 0.01 {
            p.rect_stroke(rect, r, egui::Stroke::new(3.0 * t, drop_accent().gamma_multiply(0.18 * t)), egui::StrokeKind::Outside);
        }
        let fill = if open {
            sunk_deeper(well_fill())
        } else if hot {
            well_fill().lerp_to_gamma(TEXT(), 0.04)
        } else {
            well_fill()
        };
        let edge = if open {
            drop_accent_ink().gamma_multiply(0.65)
        } else if hot {
            TEXT().gamma_multiply(0.18)
        } else {
            hairline()
        };
        p.rect_filled(rect, r, fill);
        p.rect_stroke(rect, r, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);
        let ink = if open { drop_accent_ink() } else { MUTED() };
        content(p, rect, ink);
        paint_chevron(p, egui::pos2(rect.right() - 17.0, rect.center().y), t, ink);
    }
    resp
}

/// The open list: a rounded panel under the pill's right end, its options
/// `width` wide; a click outside closes it.
fn dropdown_list(resp: &egui::Response, id: egui::Id, width: f32, add: impl FnOnce(&mut egui::Ui)) {
    egui::Popup::menu(resp)
        .id(id)
        .align(egui::RectAlign::BOTTOM_END)
        .gap(6.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .frame(dropdown_frame())
        .show(|ui| {
            ui.set_width(width);
            ui.spacing_mut().item_spacing.y = 3.0;
            add(ui);
        });
}

/// The open list's rounded panel (also for menus in this style that open
/// from an icon: the right panel's view menu).
pub(crate) fn dropdown_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(popover_fill())
        .corner_radius(egui::CornerRadius::same(16))
        .inner_margin(egui::Margin::same(6))
        .stroke(egui::Stroke::new(1.0_f32, TEXT().gamma_multiply(0.1)))
        .shadow(egui::epaint::Shadow { offset: [0, 12], blur: 32, spread: 0, color: egui::Color32::from_black_alpha(150) })
}

/// One row of a dropdown list: an optional colour dot, the name, an
/// optional muted note at the end, and a tick on the chosen one.
pub(crate) fn dropdown_option(
    ui: &mut egui::Ui,
    label: &str,
    dot: Option<egui::Color32>,
    note: Option<&str>,
    selected: bool,
) -> egui::Response {
    let (rect, item) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 32.0), egui::Sense::click());
    let item = item.on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        paint_option_card(p, rect, 10, selected, item.hovered());
        let mut x = rect.left() + 11.0;
        if let Some(c) = dot {
            paint_dot(p, egui::pos2(rect.left() + 15.0, rect.center().y), c);
            x = rect.left() + 27.0;
        }
        let ink = if selected { TEXT() } else { TEXT().gamma_multiply(0.88) };
        p.text(egui::pos2(x, rect.center().y), egui::Align2::LEFT_CENTER, label, egui::FontId::proportional(12.5), ink);
        if let Some(note) = note {
            p.text(
                egui::pos2(rect.right() - 30.0, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                note,
                egui::FontId::proportional(11.5),
                MUTED(),
            );
        }
        if selected {
            paint_checkmark(p, egui::pos2(rect.right() - 14.0, rect.center().y), drop_accent_ink());
        }
    }
    item
}

/// An option's background: an accent-tinted card with a fine accent edge
/// when it's the chosen one, a faint one under the pointer.
fn paint_option_card(p: &egui::Painter, rect: egui::Rect, radius: u8, selected: bool, hovered: bool) {
    let r = egui::CornerRadius::same(radius);
    if selected {
        p.rect_filled(rect, r, drop_accent().gamma_multiply(0.18));
        p.rect_stroke(
            rect,
            r,
            egui::Stroke::new(1.0_f32, drop_accent_ink().gamma_multiply(0.38)),
            egui::StrokeKind::Inside,
        );
    } else if hovered {
        p.rect_filled(rect, r, TEXT().gamma_multiply(0.05));
    }
}

/// A theme's colour dot, ringed so a dark one shows on a dark list.
fn paint_dot(p: &egui::Painter, c: egui::Pos2, colour: egui::Color32) {
    p.circle_filled(c, 4.5, colour);
    p.circle_stroke(c, 4.5, egui::Stroke::new(1.0_f32, TEXT().gamma_multiply(0.25)));
}

/// The 30 × 22 layout preview in the Layout list.
fn paint_layout_badge(p: &egui::Painter, rect: egui::Rect, kind: LayoutIcon, selected: bool) {
    let (bg, edge) = if selected {
        (drop_accent().gamma_multiply(0.22), drop_accent_ink().gamma_multiply(0.45))
    } else if crate::theme::is_light() {
        (TEXT().gamma_multiply(0.06), hairline())
    } else {
        (egui::Color32::from_black_alpha(95), hairline())
    };
    let r6 = egui::CornerRadius::same(6);
    p.rect_filled(rect, r6, bg);
    p.rect_stroke(rect, r6, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);

    let inner = rect.shrink2(egui::vec2(5.0, 4.0));
    let (main, side) = if selected {
        (drop_accent_ink(), drop_accent().gamma_multiply(0.85))
    } else {
        (MUTED(), MUTED().gamma_multiply(0.65))
    };
    let r1 = egui::CornerRadius::same(1);
    match kind {
        LayoutIcon::Panels => {
            // Browser · viewer · tags.
            let left = egui::Rect::from_min_size(inner.min, egui::vec2(4.0, inner.height()));
            let mid = egui::Rect::from_min_size(
                egui::pos2(inner.left() + 5.5, inner.top()),
                egui::vec2(inner.width() - 11.0, inner.height()),
            );
            let right = egui::Rect::from_min_size(egui::pos2(inner.right() - 4.0, inner.top()), egui::vec2(4.0, inner.height()));
            p.rect_filled(left, r1, side);
            p.rect_filled(mid, r1, main);
            p.rect_filled(right, r1, side);
        }
        LayoutIcon::Gallery => {
            // Three by two tiles.
            let gap = 1.5;
            let w = (inner.width() - gap * 2.0) / 3.0;
            let h = (inner.height() - gap) / 2.0;
            for row in 0..2 {
                for col in 0..3 {
                    let at = egui::pos2(inner.left() + col as f32 * (w + gap), inner.top() + row as f32 * (h + gap));
                    p.rect_filled(egui::Rect::from_min_size(at, egui::vec2(w, h)), r1, main);
                }
            }
        }
    }
}

/// The 15 × 12 outline of the current layout, in the closed pill.
fn paint_layout_glyph(p: &egui::Painter, rect: egui::Rect, kind: LayoutIcon, colour: egui::Color32) {
    let stroke = egui::Stroke::new(1.3_f32, colour);
    p.rect_stroke(rect, egui::CornerRadius::same(2), stroke, egui::StrokeKind::Inside);
    match kind {
        LayoutIcon::Panels => {
            for x in [rect.left() + rect.width() * 0.28, rect.right() - rect.width() * 0.28] {
                p.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], stroke);
            }
        }
        LayoutIcon::Gallery => {
            let c = rect.center();
            p.line_segment([egui::pos2(c.x, rect.top()), egui::pos2(c.x, rect.bottom())], stroke);
            p.line_segment([egui::pos2(rect.left(), c.y), egui::pos2(rect.right(), c.y)], stroke);
        }
    }
}

/// A chevron that turns from pointing down (`t` = 0, closed) to up (1, open).
fn paint_chevron(p: &egui::Painter, c: egui::Pos2, t: f32, colour: egui::Color32) {
    let dy = 2.2 * (1.0 - 2.0 * t);
    let stroke = egui::Stroke::new(1.6_f32, colour);
    let mid = egui::pos2(c.x, c.y + dy);
    p.line_segment([egui::pos2(c.x - 4.0, c.y - dy), mid], stroke);
    p.line_segment([mid, egui::pos2(c.x + 4.0, c.y - dy)], stroke);
}

/// The chosen option's tick.
fn paint_checkmark(p: &egui::Painter, c: egui::Pos2, colour: egui::Color32) {
    let stroke = egui::Stroke::new(1.8_f32, colour);
    let (a, b, d) = (egui::pos2(c.x - 4.5, c.y + 0.2), egui::pos2(c.x - 1.2, c.y + 3.4), egui::pos2(c.x + 4.8, c.y - 3.2));
    p.line_segment([a, b], stroke);
    p.line_segment([b, d], stroke);
}

/// The dropdowns' accent: the theme's (its blue, Aurora's pink).
fn drop_accent() -> egui::Color32 {
    crate::theme::selection_outline()
}

/// The accent for ticks, icons and edges: a lighter tint of it on dark
/// surfaces, the accent itself on light ones.
fn drop_accent_ink() -> egui::Color32 {
    if crate::theme::is_light() { drop_accent() } else { drop_accent().lerp_to_gamma(egui::Color32::WHITE, 0.25) }
}

/// The open list's panel: a little below the window in dark themes, the
/// window's own white in light ones.
fn popover_fill() -> egui::Color32 {
    if crate::theme::is_light() { FLOAT() } else { FLOAT().lerp_to_gamma(egui::Color32::BLACK, 0.18) }
}

/// The pill while its list is open: sunk a little deeper.
fn sunk_deeper(c: egui::Color32) -> egui::Color32 {
    if crate::theme::is_light() { c.lerp_to_gamma(TEXT(), 0.04) } else { c.lerp_to_gamma(egui::Color32::BLACK, 0.15) }
}

#[cfg(test)]
mod tests {
    use super::settings_window_size;
    use eframe::egui;

    /// The Settings window (plus its 36-point frame) always fits the screen;
    /// on a normal one it keeps its usual 530 × 540.
    #[test]
    fn settings_window_fits_the_screen() {
        let screen = |w: f32, h: f32| egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h));
        assert_eq!(settings_window_size(screen(1600.0, 900.0)), (530.0, 540.0));
        // 1366 × 768 at 250%: 546 × 307 points.
        let (w, h) = settings_window_size(screen(546.4, 307.2));
        assert!(w + 36.0 <= 546.4 && h + 36.0 <= 307.2, "{w} × {h}");
    }
}
