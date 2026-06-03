//! Gelbooru downloader panel — a Rust port of terminus2's `Gelbooru.java`
//! (without the cookie / Cloudflare-WebView machinery).
//!
//! Shows a form (output folder, User ID + API key, tags, blacklist, limit,
//! delay, file-type toggles), a live log and a progress bar. The actual fetching
//! runs on a background thread that talks to the Gelbooru JSON API with `ureq`,
//! streams matching files to disk, writes a `.txt` tag sidecar next to each, and
//! keeps a small de-duplication log so re-runs skip files already pulled.

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::theme::{EDGE, FIELD, MUTED, PANEL, TEXT};

const API_URL: &str = "https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1";
const SITE_HOME: &str = "https://gelbooru.com/";
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
     (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

const PAGE_LIMIT: u32 = 100;
const MAX_TRANSIENT_RETRIES: u32 = 5;

/// Minimum delay (seconds) between downloads. Enforced everywhere — the user
/// can't go below this — to avoid hammering Gelbooru and getting rate-limited.
const MIN_DELAY: f32 = 3.0;

/// Maximum files a user may download per calendar day. A courtesy guard-rail so
/// the app can't be used (or accidentally left running) to mass-pull from
/// Gelbooru. The running count is kept in an *encrypted* file (DPAPI, same as the
/// API key) so it can't simply be edited back down — though it's a soft limit:
/// deleting the file or changing the system clock resets it.
const DAILY_CAP: u32 = 2000;

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "webp"];
const VIDEO_EXTS: &[&str] = &["mp4", "webm", "avi"];
const GIF_EXTS: &[&str] = &["gif"];

/// Messages the background download thread sends back to the UI.
enum DlMsg {
    Log(String),
    /// (downloaded so far, target total)
    Progress(u32, u32),
    Done,
}

/// Persisted-across-runs form values (credentials + last inputs).
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SavedConfig {
    #[serde(default)]
    user_id: String,
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    tags: String,
    #[serde(default)]
    blacklist: String,
    #[serde(default)]
    output_dir: String,
}

/// All UI + runtime state for the downloader view. Lives on `RightPanelState`.
pub struct DownloaderState {
    output_dir: String,
    user_id: String,
    api_key: String,
    tags: String,
    blacklist: String,
    limit: u32,
    delay: f32,
    include_img: bool,
    include_gif: bool,
    include_vid: bool,

    /// Rolling log lines shown in the console box.
    log: Vec<String>,
    /// `(done, total)` for the progress bar; `total == 0` means idle.
    progress: (u32, u32),
    status: String,

    /// `true` while a download thread is active.
    running: bool,
    /// Flipped to request the running thread stop.
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<DlMsg>>,

    /// Loaded once so credentials populate on first show.
    loaded: bool,

    /// Gelbooru reachability: 0 = checking, 1 = online, 2 = offline. Updated by a
    /// background monitor thread, read each frame to draw the status pill.
    api_status: Arc<AtomicU8>,
    /// Whether the monitor thread has been spawned (only once per session).
    monitor_started: bool,
}

/// API-status codes shared with the monitor thread.
const API_CHECKING: u8 = 0;
const API_ONLINE: u8 = 1;
const API_OFFLINE: u8 = 2;

impl Default for DownloaderState {
    fn default() -> Self {
        Self {
            output_dir: String::new(),
            user_id: String::new(),
            api_key: String::new(),
            tags: "example_tag".to_string(),
            blacklist: String::new(),
            limit: 100,
            delay: MIN_DELAY,
            include_img: true,
            include_gif: false,
            include_vid: false,
            log: Vec::new(),
            progress: (0, 0),
            status: "Idle".to_string(),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            loaded: false,
            api_status: Arc::new(AtomicU8::new(API_CHECKING)),
            monitor_started: false,
        }
    }
}

impl DownloaderState {
    fn push_log(&mut self, line: impl Into<String>) {
        self.log.push(line.into());
        // Cap the in-memory log so a long run can't grow unbounded.
        if self.log.len() > 1000 {
            let overflow = self.log.len() - 1000;
            self.log.drain(0..overflow);
        }
    }
}

/// Render the downloader view. Drains background messages, draws the form, and
/// starts / cancels the worker thread.
pub fn show(ui: &mut egui::Ui, state: &mut DownloaderState) {
    if !state.loaded {
        state.loaded = true;
        if let Some(cfg) = load_config() {
            state.user_id = cfg.user_id;
            state.api_key = cfg.api_key;
            if !cfg.tags.is_empty() {
                state.tags = cfg.tags;
            }
            state.blacklist = cfg.blacklist;
            state.output_dir = cfg.output_dir;
        }
        // Never let a persisted/old value drop below the safety floor.
        if state.delay < MIN_DELAY {
            state.delay = MIN_DELAY;
        }
    }

    // Spawn the API-status monitor once: it polls gelbooru.com every few seconds
    // and updates `api_status`, which drives the pill in the Destination header.
    if !state.monitor_started {
        state.monitor_started = true;
        start_api_monitor(Arc::clone(&state.api_status), ui.ctx().clone());
    }

    // Drain any messages from the worker.
    if let Some(rx) = &state.rx {
        let mut msgs = Vec::new();
        while let Ok(m) = rx.try_recv() {
            msgs.push(m);
        }
        for m in msgs {
            match m {
                DlMsg::Log(line) => state.push_log(line),
                DlMsg::Progress(done, total) => {
                    state.progress = (done, total);
                    state.status = format!("{done} / {total}");
                }
                DlMsg::Done => {
                    state.running = false;
                    state.rx = None;
                    if state.status.starts_with("Cancel") {
                        // keep the cancel label
                    } else {
                        state.status = "Done".to_string();
                    }
                }
            }
        }
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }

    // Round every widget in this view and give text fields an inset (PANEL)
    // background so they read as wells inside the lighter FIELD section cards.
    let radius = egui::CornerRadius::same(10);
    {
        let v = ui.visuals_mut();
        v.widgets.inactive.corner_radius = radius;
        v.widgets.hovered.corner_radius = radius;
        v.widgets.active.corner_radius = radius;
        v.widgets.noninteractive.corner_radius = radius;
        v.widgets.open.corner_radius = radius;
    }

    // Header.
    ui.add_space(2.0);
    ui.vertical_centered(|ui| {
        ui.heading(egui::RichText::new("Gelbooru Downloader").color(TEXT()).strong());
    });
    ui.add_space(8.0);

    let enabled = !state.running;

    // Pin the log + progress + buttons to the bottom (a bottom panel takes only
    // the height it needs), then let the form scroll in the remaining space. This
    // guarantees the action area never overflows past the panel's bottom edge.
    egui::Panel::bottom("dl_footer")
        .resizable(false)
        .show_separator_line(false)
        .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            ui.add_space(8.0);

            // Console / log — a dark inset well with a small header.
            field_label(ui, "Log");
            let log_bg = if crate::theme::is_light() {
                FIELD()
            } else {
                egui::Color32::from_rgb(15, 15, 17)
            };
            egui::Frame::new()
                .fill(log_bg)
                .corner_radius(egui::CornerRadius::same(12))
                .inner_margin(egui::Margin::same(10))
                .stroke(egui::Stroke::new(1.0, EDGE()))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::ScrollArea::vertical()
                        .id_salt("dl_log")
                        .max_height(120.0)
                        .auto_shrink([false, false])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            if state.log.is_empty() {
                                ui.label(
                                    egui::RichText::new("Log output will appear here.")
                                        .color(MUTED())
                                        .monospace()
                                        .size(12.0),
                                );
                            } else {
                                for line in &state.log {
                                    ui.label(egui::RichText::new(line).color(TEXT()).monospace().size(12.0));
                                }
                            }
                        });
                });

            ui.add_space(10.0);

            // Progress bar.
            let frac = if state.progress.1 > 0 {
                (state.progress.0 as f32 / state.progress.1 as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(egui::RichText::new(state.status.clone()).size(12.0))
                    .corner_radius(8)
                    .desired_height(18.0),
            );
            ui.add_space(8.0);

            // Start / Cancel buttons.
            ui.horizontal(|ui| {
                let gap = 10.0;
                ui.spacing_mut().item_spacing.x = gap;
                let btn_w = (ui.available_width() - gap) / 2.0;
                let size = egui::vec2(btn_w, 38.0);

                let start = egui::Button::new(
                    egui::RichText::new("Start Download").color(egui::Color32::WHITE).strong(),
                )
                .fill(egui::Color32::from_rgb(96, 99, 105));
                if ui.add_enabled_ui(!state.running, |ui| ui.add_sized(size, start)).inner.clicked() {
                    start_download(state, ui.ctx());
                }

                let cancel_bg = egui::Color32::from_rgb(180, 40, 40);
                let cancel = egui::Button::new(
                    egui::RichText::new("Cancel").color(egui::Color32::WHITE).strong(),
                )
                .fill(cancel_bg);
                if ui.add_enabled_ui(state.running, |ui| ui.add_sized(size, cancel)).inner.clicked() {
                    state.cancel.store(true, Ordering::SeqCst);
                    state.status = "Cancelling…".to_string();
                    state.push_log("Cancel requested…");
                }
            });
            ui.add_space(2.0);
        });

    // Form — fills the space above the footer and scrolls if it's too tall.
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let api = state.api_status.load(Ordering::Relaxed);
                    section_with_pill(ui, "Destination", api, |ui| {
                        field_label(ui, "Output folder");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            let folder_svg = egui::include_image!("../icons/folder.svg");
                            if crate::svg_button(ui, folder_svg, "Choose output folder", 34.0, crate::theme::icon_tint(MUTED()))
                                .clicked()
                            {
                                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                    state.output_dir = dir.display().to_string();
                                }
                            }
                            field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.output_dir)
                                .hint_text("Where files are saved"));
                        });
                    });

                    // Gelbooru disabled anonymous API access (mid-2025 ToS change):
                    // a User ID + API key are now required — explained via the info
                    // icon's hover next to the section title.
                    section_with_info(ui, "Account",
                        "Gelbooru no longer allows anonymous downloads — you must log in with \
                         your account's User ID and API key. Get them from gelbooru.com → \
                         Account → Options → 'API Access Credentials' (free account required).",
                        |ui| {
                        field_label(ui, "User ID");
                        field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.user_id)
                            .hint_text("Required"));
                        ui.add_space(8.0);
                        field_label(ui, "API key");
                        field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.api_key)
                            .password(true)
                            .hint_text("Required · stored encrypted"));
                    });

                    section(ui, "Search", |ui| {
                        field_label(ui, "Tags");
                        field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.tags)
                            .hint_text("space-separated, e.g. blue_sky 1girl"));
                        ui.add_space(8.0);
                        field_label(ui, "Blacklist");
                        field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.blacklist)
                            .hint_text("comma-separated tags to skip"));
                    });

                    section(ui, "Options", |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Limit").color(TEXT()));
                            ui.add_enabled(
                                enabled,
                                egui::DragValue::new(&mut state.limit).range(1..=10000).speed(1.0),
                            );
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("Delay (s)").color(TEXT()));
                            ui.add_enabled(
                                enabled,
                                egui::DragValue::new(&mut state.delay).range(MIN_DELAY..=60.0).speed(0.1),
                            );
                            // Info icon explaining the enforced minimum delay.
                            info_icon(
                                ui,
                                "The delay is the wait between downloads. It can't go below 3 \
                                 seconds: Gelbooru rate-limits frequent requests, so a shorter \
                                 delay risks being throttled or temporarily blocked.",
                            );
                        });
                        ui.add_space(8.0);
                        field_label(ui, "File types");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 14.0;
                            ui.add_enabled_ui(enabled, |ui| dot_checkbox(ui, &mut state.include_img, "Image"));
                            ui.add_enabled_ui(enabled, |ui| dot_checkbox(ui, &mut state.include_gif, "Gif"));
                            ui.add_enabled_ui(enabled, |ui| dot_checkbox(ui, &mut state.include_vid, "Video"));
                        });
                    });
                    ui.add_space(8.0);
                });
        });
}

/// The bundled info SVG at 16 px, tinted with the muted theme colour, with a
/// hover tooltip. Returns the response so callers can lay it out alongside text.
fn info_icon(ui: &mut egui::Ui, tooltip: &str) -> egui::Response {
    ui.add(
        egui::Image::new(egui::include_image!("../icons/info.svg"))
            .fit_to_exact_size(egui::vec2(16.0, 16.0))
            .tint(crate::theme::icon_tint(MUTED())),
    )
    .on_hover_text(tooltip)
}

/// A checkbox that shows a filled **dot** when on, instead of egui's checkmark.
/// Behaves like `ui.checkbox`: clicking the box or its label toggles `checked`.
/// egui's `Checkbox` only ever draws a tick, so we paint our own box + dot.
fn dot_checkbox(ui: &mut egui::Ui, checked: &mut bool, text: &str) -> egui::Response {
    let icon = ui.spacing().icon_width; // box edge length
    let gap = ui.spacing().icon_spacing;
    let galley = egui::WidgetText::from(text).into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        f32::INFINITY,
        egui::TextStyle::Button,
    );

    let mut desired = egui::vec2(icon + gap + galley.size().x, galley.size().y.max(icon));
    desired.y = desired.y.max(ui.spacing().interact_size.y);
    let (rect, mut response) = ui.allocate_exact_size(desired, egui::Sense::click());
    if response.clicked() {
        *checked = !*checked;
        response.mark_changed();
    }
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *checked, text)
    });

    if ui.is_rect_visible(rect) {
        let visuals = ui.style().interact(&response);
        let center = egui::pos2(rect.left() + icon / 2.0, rect.center().y);
        let outer = icon / 2.0;
        // Round indicator: an outline circle, with a filled dot when checked.
        ui.painter().circle(center, outer, visuals.bg_fill, visuals.bg_stroke);
        if *checked {
            ui.painter().circle_filled(center, outer * 0.5, visuals.fg_stroke.color);
        }
        let text_pos = egui::pos2(center.x + outer + gap, rect.center().y - galley.size().y / 2.0);
        ui.painter().galley(text_pos, galley, visuals.text_color());
    }
    response
}

/// A titled, rounded group card holding related controls — mirrors the look of
/// the Settings window's sections so the downloader feels native to the app.
fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
    ui.add_space(4.0);
    section_body(ui, add);
}

/// Like [`section`], but draws an info icon next to the title whose hover tooltip
/// shows `info`.
fn section_with_info(ui: &mut egui::Ui, title: &str, info: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(egui::RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
        info_icon(ui, info);
    });
    ui.add_space(4.0);
    section_body(ui, add);
}

/// Like [`section`], but draws a right-aligned API-status pill next to the title.
fn section_with_pill(ui: &mut egui::Ui, title: &str, api: u8, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            api_pill(ui, api);
        });
    });
    ui.add_space(4.0);
    section_body(ui, add);
}

/// The rounded card body shared by both section variants.
fn section_body(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(FIELD())
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(12, 10))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// Draw the coloured "API: …" status pill.
fn api_pill(ui: &mut egui::Ui, api: u8) {
    let (text, bg) = match api {
        API_ONLINE => ("Online", egui::Color32::from_rgb(35, 137, 58)),
        API_OFFLINE => ("Offline", egui::Color32::from_rgb(160, 60, 60)),
        _ => ("Checking…", egui::Color32::from_rgb(120, 120, 120)),
    };
    egui::Frame::new()
        .fill(bg)
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(format!("API: {text}"))
                    .color(egui::Color32::WHITE)
                    .size(11.0)
                    .strong(),
            );
        });
}

/// Spawn a daemon-style thread that probes gelbooru.com every 5s and stores the
/// result in `status`, repainting the UI when it changes.
fn start_api_monitor(status: Arc<AtomicU8>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::NativeTls)
                    // Validate against the OS cert store (with AIA intermediate
                    // fetching) instead of ureq's bundled webpki roots — see
                    // civitai.rs for the CDN incomplete-chain failure this avoids.
                    .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                    .build(),
            )
            .timeout_global(Some(Duration::from_secs(8)))
            .http_status_as_error(false)
            .build()
            .into();

        loop {
            let online = agent
                .get(SITE_HOME)
                .header("User-Agent", USER_AGENT)
                .call()
                .map(|r| {
                    let s = r.status().as_u16();
                    (200..500).contains(&s)
                })
                .unwrap_or(false);

            let new = if online { API_ONLINE } else { API_OFFLINE };
            if status.swap(new, Ordering::Relaxed) != new {
                ctx.request_repaint();
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    });
}

/// A small muted caption shown above a field.
fn field_label(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).color(MUTED()).size(12.0));
    ui.add_space(2.0);
}

/// A full-width text field with an inset (PANEL) background so it stands out
/// against the lighter section card.
fn field_edit(ui: &mut egui::Ui, enabled: bool, edit: egui::TextEdit<'_>) {
    ui.scope(|ui| {
        ui.visuals_mut().extreme_bg_color = PANEL();
        ui.add_enabled(
            enabled,
            edit.desired_width(f32::INFINITY).margin(egui::Margin::symmetric(10, 6)),
        );
    });
}

/// Validate the form and spawn the background worker.
fn start_download(state: &mut DownloaderState, ctx: &egui::Context) {
    if state.running {
        return;
    }
    state.log.clear();
    state.progress = (0, 0);

    // Hard floor on the delay — protects Gelbooru from being overloaded even if a
    // stale config or edge case slipped a smaller value through.
    if state.delay < MIN_DELAY {
        state.delay = MIN_DELAY;
    }

    let uid = state.user_id.trim().to_string();
    let key = state.api_key.trim().to_string();
    if uid.is_empty() || key.is_empty() {
        state.push_log("Error: User ID and API Key are required.");
        state.status = "Idle".to_string();
        return;
    }
    if !state.include_img && !state.include_gif && !state.include_vid {
        state.push_log("No file types selected. Nothing to download.");
        state.status = "Idle".to_string();
        return;
    }
    if state.output_dir.trim().is_empty() {
        state.push_log("Error: Output folder is blank.");
        state.status = "Idle".to_string();
        return;
    }

    // Persist the inputs for next time.
    save_config(&SavedConfig {
        user_id: uid.clone(),
        api_key: key.clone(),
        tags: state.tags.clone(),
        blacklist: state.blacklist.clone(),
        output_dir: state.output_dir.clone(),
    });

    let cfg = WorkerCfg {
        user_id: uid,
        api_key: key,
        tags: state.tags.clone(),
        blacklist: state.blacklist.clone(),
        limit: state.limit,
        delay: state.delay,
        include_img: state.include_img,
        include_gif: state.include_gif,
        include_vid: state.include_vid,
        output_dir: PathBuf::from(state.output_dir.trim()),
    };

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    state.cancel = Arc::clone(&cancel);
    state.rx = Some(rx);
    state.running = true;
    state.status = "Connecting…".to_string();

    let ctx = ctx.clone();
    std::thread::spawn(move || {
        run_download(cfg, tx, cancel, ctx);
    });
}

/// Immutable settings handed to the worker thread.
struct WorkerCfg {
    user_id: String,
    api_key: String,
    tags: String,
    blacklist: String,
    limit: u32,
    delay: f32,
    include_img: bool,
    include_gif: bool,
    include_vid: bool,
    output_dir: PathBuf,
}

/// A parsed Gelbooru post.
struct Post {
    md5: String,
    file_url: String,
    raw_tags: String,
}

fn run_download(cfg: WorkerCfg, tx: Sender<DlMsg>, cancel: Arc<AtomicBool>, ctx: egui::Context) {
    let log = |s: String| {
        let _ = tx.send(DlMsg::Log(s));
        ctx.request_repaint();
    };

    let mut downloaded_log = load_download_log();
    log(format!("Loaded {} previously downloaded file records.", downloaded_log.len()));

    let final_tags = build_final_tags(&cfg.tags, cfg.include_img, cfg.include_gif, cfg.include_vid);
    if final_tags.trim().is_empty() {
        log("No tags provided (and/or all filtered). Nothing to do.".into());
        let _ = tx.send(DlMsg::Done);
        return;
    }
    log(format!("Starting download with tags: {final_tags}"));

    if let Err(e) = std::fs::create_dir_all(&cfg.output_dir) {
        log(format!("Error: cannot create output folder: {e}"));
        let _ = tx.send(DlMsg::Done);
        return;
    }

    // native-tls => Windows SChannel. ureq 3.x defaults to rustls even with the
    // native-tls feature on, so the provider must be selected explicitly (rustls
    // isn't compiled in — see Cargo.toml / ai_models.rs). Without this the agent
    // fails on every HTTPS call, which on this worker thread looked like a silent
    // no-op. `http_status_as_error(false)` lets us inspect 4xx/5xx ourselves.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                // Use the OS cert store (with AIA intermediate fetching) rather
                // than ureq's bundled webpki roots — see civitai.rs.
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .http_status_as_error(false)
        .build()
        .into();

    // Enforce the daily cap: today's remaining allowance bounds this run.
    let mut used_today = quota_used_today();
    let remaining_today = DAILY_CAP.saturating_sub(used_today);
    if remaining_today == 0 {
        log(format!(
            "Daily limit reached ({DAILY_CAP}/day). Try again tomorrow."
        ));
        let _ = tx.send(DlMsg::Done);
        return;
    }
    let cap = cfg.limit.min(remaining_today);
    if cap < cfg.limit {
        log(format!(
            "Note: only {remaining_today} of today's {DAILY_CAP} daily allowance remain — \
             this run is capped at {cap}."
        ));
    }

    let _ = tx.send(DlMsg::Progress(0, cap));
    ctx.request_repaint();

    let mut total_downloaded: u32 = 0;
    let mut page: u32 = 0;

    'outer: while total_downloaded < cap && !cancel.load(Ordering::SeqCst) {
        let posts = match fetch_page(&agent, &final_tags, page, &cfg, &cancel, &log) {
            Some(p) => p,
            None => break,
        };
        if posts.is_empty() {
            log("No more posts found.".into());
            break;
        }

        for post in posts {
            if cancel.load(Ordering::SeqCst) || total_downloaded >= cap {
                break 'outer;
            }
            if post.file_url.is_empty() || post.md5.is_empty() {
                continue;
            }
            if post.file_url.to_lowercase().ends_with(".zip") {
                log(format!("Skipped zip file: {}", post.file_url));
                continue;
            }
            if is_blacklisted(&post.raw_tags, &cfg.blacklist) {
                log(format!("Skipped (blacklisted): {}", post.md5));
                continue;
            }
            if downloaded_log.contains(&post.md5) {
                log(format!("Skipped (already downloaded): {}", post.md5));
                continue;
            }

            let clean = post.file_url.split('?').next().unwrap_or(&post.file_url);
            let ext = clean.rsplit('.').next().unwrap_or("bin").to_lowercase();

            if !is_allowed_by_selection(&ext, cfg.include_img, cfg.include_gif, cfg.include_vid) {
                log(format!("Skipped (type not selected): {}.{}", post.md5, ext));
                continue;
            }

            let file_name = format!("{}.{}", post.md5, ext);
            let img_path = cfg.output_dir.join(&file_name);
            let txt_path = cfg.output_dir.join(format!("{}.txt", post.md5));

            if img_path.exists() {
                log(format!("Skipped (file exists): {file_name}"));
                append_download_log(&post.md5, &mut downloaded_log);
                continue;
            }

            log(format!("Downloading: {file_name}"));
            match download_file(&agent, &post.file_url, &img_path, &cancel) {
                Ok(true) => {}
                Ok(false) => {
                    if cancel.load(Ordering::SeqCst) {
                        break 'outer;
                    }
                    continue;
                }
                Err(e) => {
                    log(format!("Error downloading {}: {e}", post.md5));
                    let _ = std::fs::remove_file(&img_path);
                    continue;
                }
            }

            let formatted = format_gelbooru_tags(&post.raw_tags);
            if let Err(e) = std::fs::write(&txt_path, formatted) {
                log(format!("Warning: could not write tags for {}: {e}", post.md5));
            }

            append_download_log(&post.md5, &mut downloaded_log);
            total_downloaded += 1;
            // Count it against today's quota and persist immediately, so a crash
            // mid-run can't reset the running total.
            used_today += 1;
            quota_save(used_today);
            let _ = tx.send(DlMsg::Progress(total_downloaded, cap));
            ctx.request_repaint();

            if cfg.delay > 0.0 {
                // Sleep in small slices so Cancel feels responsive.
                let mut slept = 0.0;
                while slept < cfg.delay && !cancel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(100));
                    slept += 0.1;
                }
            }
        }

        page += 1;
        std::thread::sleep(Duration::from_millis(500)); // polite pacing
    }

    if cancel.load(Ordering::SeqCst) {
        log("Cancelled.".into());
    } else {
        log(format!("Download finished: {total_downloaded} new files this session."));
        let left = DAILY_CAP.saturating_sub(used_today);
        log(format!("Daily allowance remaining: {left} of {DAILY_CAP}."));
    }
    let _ = tx.send(DlMsg::Done);
    ctx.request_repaint();
}

/// Fetch one API page, with retry/backoff on transient failures. `None` means a
/// fatal error (caller should stop).
fn fetch_page(
    agent: &ureq::Agent,
    final_tags: &str,
    page: u32,
    cfg: &WorkerCfg,
    cancel: &AtomicBool,
    log: &impl Fn(String),
) -> Option<Vec<Post>> {
    let mut transient = 0u32;
    while !cancel.load(Ordering::SeqCst) {
        let url = build_api_url(final_tags, PAGE_LIMIT, page, &cfg.user_id, &cfg.api_key);
        let resp = agent
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "application/json,text/plain,*/*")
            .call();

        let mut resp = match resp {
            Ok(r) => r,
            Err(e) => {
                transient += 1;
                if transient > MAX_TRANSIENT_RETRIES {
                    log(format!("Error: network failure (max retries): {e}"));
                    return None;
                }
                let wait = backoff_ms(transient);
                log(format!("Network issue, retrying in {:.1}s…", wait as f64 / 1000.0));
                sleep_cancellable(wait, cancel);
                continue;
            }
        };

        let status = resp.status().as_u16();
        if status == 200 {
            let body = match resp.body_mut().read_to_string() {
                Ok(b) => b,
                Err(e) => {
                    log(format!("Error reading API response: {e}"));
                    return None;
                }
            };
            return Some(parse_posts(&body, log));
        }

        if status == 429 || status == 408 || (500..=599).contains(&status) {
            transient += 1;
            if transient > MAX_TRANSIENT_RETRIES {
                log(format!("Error: API returned {status} repeatedly (max retries)."));
                return None;
            }
            let wait = backoff_ms(transient);
            log(format!("API busy (HTTP {status}), retrying in {:.1}s…", wait as f64 / 1000.0));
            sleep_cancellable(wait, cancel);
            continue;
        }

        log(format!("Error: API returned status {status}."));
        return None;
    }
    None
}

/// Stream a file to `dest`, honouring cancellation. Returns `Ok(true)` on
/// success, `Ok(false)` on a non-success status (partial file removed).
fn download_file(
    agent: &ureq::Agent,
    file_url: &str,
    dest: &Path,
    cancel: &AtomicBool,
) -> Result<bool, String> {
    let url = normalize_file_url(file_url);
    let mut resp = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "*/*")
        // Many CDNs 403 a hotlink without a matching Referer / Origin.
        .header("Referer", SITE_HOME)
        .header("Origin", "https://gelbooru.com")
        .call()
        .map_err(|e| e.to_string())?;

    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        return Ok(false);
    }

    let mut file = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    let mut reader = resp.body_mut().as_reader();
    let mut buf = [0u8; 64 * 1024];
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file);
            let _ = std::fs::remove_file(dest);
            return Ok(false);
        }
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        use std::io::Write;
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
    Ok(true)
}

fn sleep_cancellable(total_ms: u64, cancel: &AtomicBool) {
    let mut slept = 0u64;
    while slept < total_ms && !cancel.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));
        slept += 100;
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (ported 1:1 from Gelbooru.java)
// ---------------------------------------------------------------------------

fn build_final_tags(user_tags: &str, img: bool, gif: bool, vid: bool) -> String {
    let mut tags: Vec<String> = Vec::new();
    let trimmed = user_tags.trim();
    if !trimmed.is_empty() {
        tags.extend(trimmed.split_whitespace().map(|s| s.to_string()));
    }

    let selected = img as u8 + gif as u8 + vid as u8;
    if selected == 1 && gif {
        // Gif-only: a single positive "gif" tag is far more reliable than
        // negating every other extension (which Gelbooru's tag limit truncates).
        tags.push("gif".to_string());
    } else {
        if !img {
            for e in IMAGE_EXTS {
                tags.push(format!("-{e}"));
            }
        }
        if !vid {
            for e in VIDEO_EXTS {
                tags.push(format!("-{e}"));
            }
        }
        if !gif {
            for e in GIF_EXTS {
                tags.push(format!("-{e}"));
            }
        }
    }
    tags.join(" ").trim().to_string()
}

fn is_blacklisted(raw_tags: &str, blacklist: &str) -> bool {
    let tags_lower = raw_tags.to_lowercase();
    blacklist
        .split(',')
        .map(|b| b.trim().to_lowercase())
        .filter(|b| !b.is_empty())
        .any(|b| tags_lower.contains(&b))
}

fn format_gelbooru_tags(raw: &str) -> String {
    let s = raw.trim();
    if s.is_empty() {
        return String::new();
    }
    s.split_whitespace().collect::<Vec<_>>().join(", ")
}

fn is_allowed_by_selection(ext: &str, img: bool, gif: bool, vid: bool) -> bool {
    let ext = ext.to_lowercase();
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return img;
    }
    if GIF_EXTS.contains(&ext.as_str()) {
        return gif;
    }
    if VIDEO_EXTS.contains(&ext.as_str()) {
        return vid;
    }
    false
}

fn build_api_url(final_tags: &str, per_page: u32, pid: u32, user_id: &str, api_key: &str) -> String {
    let mut s = String::from(API_URL);
    s.push_str("&tags=");
    s.push_str(&percent_encode(final_tags));
    s.push_str(&format!("&limit={per_page}&pid={pid}"));
    if !user_id.trim().is_empty() {
        s.push_str("&user_id=");
        s.push_str(&percent_encode(user_id.trim()));
    }
    if !api_key.trim().is_empty() {
        s.push_str("&api_key=");
        s.push_str(&percent_encode(api_key.trim()));
    }
    s
}

fn normalize_file_url(raw: &str) -> String {
    let s = raw.trim();
    if let Some(rest) = s.strip_prefix("//") {
        format!("https://{rest}")
    } else if s.starts_with('/') {
        format!("https://gelbooru.com{s}")
    } else {
        s.to_string()
    }
}

/// Percent-encode a query value (RFC 3986 unreserved set kept literal).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn parse_posts(body: &str, log: &impl Fn(String)) -> Vec<Post> {
    let root: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => {
            log("Error: failed to parse API JSON.".into());
            return Vec::new();
        }
    };
    let post_node = match root.get("post") {
        Some(n) if !n.is_null() => n,
        _ => return Vec::new(),
    };

    let mut out = Vec::new();
    let add = |p: &serde_json::Value, out: &mut Vec<Post>| {
        let mut file_url = p.get("file_url").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if let Some(rest) = file_url.strip_prefix("//") {
            file_url = format!("https:{}", format!("//{rest}"));
        } else if file_url.starts_with('/') {
            file_url = format!("https://gelbooru.com{file_url}");
        }
        let md5 = p.get("md5").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let raw_tags = p.get("tags").and_then(|v| v.as_str()).unwrap_or("").to_string();

        if file_url.is_empty() || file_url.eq_ignore_ascii_case("null") {
            return;
        }
        if md5.is_empty() || md5.eq_ignore_ascii_case("null") {
            return;
        }
        out.push(Post { md5, file_url, raw_tags });
    };

    if let Some(arr) = post_node.as_array() {
        for p in arr {
            add(p, &mut out);
        }
    } else if post_node.is_object() {
        add(post_node, &mut out);
    }
    out
}

fn backoff_ms(retry: u32) -> u64 {
    let shift = retry.saturating_sub(1).min(5);
    (1000u64 << shift).min(20_000)
}

// ---------------------------------------------------------------------------
// Config + download-log persistence
// ---------------------------------------------------------------------------

fn config_dir() -> PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow"))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn config_path() -> PathBuf {
    config_dir().join("gelbooru_credentials.json")
}

fn download_log_path() -> PathBuf {
    config_dir().join("gelbooru_download_log.json")
}

fn load_config() -> Option<SavedConfig> {
    let json = std::fs::read_to_string(config_path()).ok()?;
    let mut cfg: SavedConfig = serde_json::from_str(&json).ok()?;
    // The API key is stored encrypted (DPAPI on Windows); decrypt it back for use.
    cfg.api_key = crate::secret::unprotect(&cfg.api_key);
    Some(cfg)
}

fn save_config(cfg: &SavedConfig) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    // Never write the API key as plaintext: encrypt it (DPAPI on Windows, tied to
    // the current user account) so it can't be read straight out of the JSON.
    let on_disk = SavedConfig {
        user_id: cfg.user_id.clone(),
        api_key: crate::secret::protect(&cfg.api_key),
        tags: cfg.tags.clone(),
        blacklist: cfg.blacklist.clone(),
        output_dir: cfg.output_dir.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&on_disk) {
        let _ = std::fs::write(config_path(), json);
    }
}

fn load_download_log() -> HashSet<String> {
    let mut set = HashSet::new();
    if let Ok(json) = std::fs::read_to_string(download_log_path()) {
        if let Ok(v) = serde_json::from_str::<Vec<String>>(&json) {
            for s in v {
                let t = s.trim().to_string();
                if !t.is_empty() {
                    set.insert(t);
                }
            }
        }
    }
    set
}

// ---------------------------------------------------------------------------
// Daily download quota (encrypted, per calendar day)
// ---------------------------------------------------------------------------

/// `{ date, count }` for the daily cap, stored encrypted on disk.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct QuotaData {
    #[serde(default)]
    date: String,
    #[serde(default)]
    count: u32,
}

fn quota_path() -> PathBuf {
    config_dir().join("gelbooru_quota.dat")
}

/// Local calendar day as `YYYY-MM-DD`.
fn today_str() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// How many files have already been downloaded *today*. Returns 0 if the file is
/// missing, can't be decrypted, or holds a different (older) date.
fn quota_used_today() -> u32 {
    let Ok(stored) = std::fs::read_to_string(quota_path()) else {
        return 0;
    };
    let json = crate::secret::unprotect(stored.trim());
    let Ok(q) = serde_json::from_str::<QuotaData>(&json) else {
        return 0;
    };
    if q.date == today_str() {
        q.count
    } else {
        0
    }
}

/// Persist today's running count, encrypted.
fn quota_save(count: u32) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    let q = QuotaData { date: today_str(), count };
    if let Ok(json) = serde_json::to_string(&q) {
        let enc = crate::secret::protect(&json);
        let _ = std::fs::write(quota_path(), enc);
    }
}

fn append_download_log(md5: &str, set: &mut HashSet<String>) {
    if md5.trim().is_empty() {
        return;
    }
    set.insert(md5.trim().to_string());
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    let list: Vec<&String> = set.iter().collect();
    if let Ok(json) = serde_json::to_string(&list) {
        let _ = std::fs::write(download_log_path(), json);
    }
}
