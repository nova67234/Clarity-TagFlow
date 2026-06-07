//! Civitai resource info panel — a Rust port of terminus2's `CivitaiInfo.java`.
//!
//! Reads the embedded Stable-Diffusion generation metadata of the selected image
//! (the same string the Tags/Metadata switch shows), extracts the models / LoRAs /
//! VAE / embeddings that produced it, looks each up against the Civitai API, and
//! renders clickable cards (preview thumbnail + name + trigger words) plus an
//! "Original Upload" link when the metadata or filename points back to a Civitai
//! image/post.
//!
//! The metadata parsing (A1111, ComfyUI, and Civitai-generator formats) and the
//! API lookups run on a short-lived background thread so the UI never blocks; the
//! thread streams a status string and finally a fully-built result back over an
//! `mpsc` channel. Preview images are decoded to `egui::ColorImage` off-thread and
//! uploaded as textures on the UI thread (egui has no network image loader here).

use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::theme::{EDGE, FIELD, MUTED, PANEL, TEXT};

const USER_AGENT: &str = "Clarity TagFlow (Civitai resource lookup)";
const API_PING: &str = "https://civitai.com/api/v1/models?limit=1";

/// API-status codes shared with the monitor thread (mirrors the downloader).
const API_CHECKING: u8 = 0;
const API_ONLINE: u8 = 1;
const API_OFFLINE: u8 = 2;

// ---------------------------------------------------------------------------
// Resource taxonomy (ported from the Java enums)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ItemType {
    Model,
    Lora,
    Vae,
    Embedding,
}

#[derive(Clone, Copy)]
enum LookupMethod {
    Hash,
    Name,
    VersionId,
}

/// One thing to look up on Civitai, identified by (in priority order) a model
/// version id, a file hash, or a resource name.
struct Task {
    kind: ItemType,
    hash: Option<String>,
    name: Option<String>,
    version_id: Option<String>,
}

impl Task {
    fn version(kind: ItemType, version_id: String) -> Self {
        Self { kind, hash: None, name: None, version_id: Some(version_id) }
    }
    fn hash(kind: ItemType, hash: String) -> Self {
        Self { kind, hash: Some(hash), name: None, version_id: None }
    }
    fn name(kind: ItemType, name: String) -> Self {
        Self { kind, hash: None, name: Some(name), version_id: None }
    }
    fn embed(hash: String, name: String) -> Self {
        Self { kind: ItemType::Embedding, hash: Some(hash), name: Some(name), version_id: None }
    }
}

// ---------------------------------------------------------------------------
// Background → UI messages and the fetched data shapes
// ---------------------------------------------------------------------------

/// A resolved Civitai resource as produced by the worker (image decoded but not
/// yet uploaded to the GPU — that happens on the UI thread).
struct Fetched {
    name: String,
    url: String,
    triggers: Vec<String>,
    image: Option<egui::ColorImage>,
    has_video_only: bool,
    /// The Civitai `type` ("LORA" / "LoCon" / "DoRA" / …), used to split LoRAs
    /// from LyCORIS-style models into separate sections.
    resource_type: String,
    /// API download URL for the resolved version's primary file (for in-app
    /// download), and the file's suggested name.
    download_url: String,
    download_filename: Option<String>,
}

struct FetchedSection {
    title: String,
    items: Vec<Fetched>,
}

/// The full lookup result handed back to the UI in one message.
struct CivResult {
    source_url: Option<String>,
    sections: Vec<FetchedSection>,
}

enum CivMsg {
    Done(Box<CivResult>),
}

/// What the Download button asks for (collected during render, acted on after).
struct DownloadRequest {
    name: String,
    download_url: String,
    filename: Option<String>,
}

/// Progress from a model-download worker thread.
enum DlMsg {
    Progress(u64, u64), // received, total (total = 0 if unknown)
    Done(std::path::PathBuf),
    Error(String),
}

/// One in-flight / finished model download, shown as a progress row.
struct Download {
    name: String,
    rx: Option<Receiver<DlMsg>>,
    received: u64,
    total: u64,
    status: String,
    ok: bool,
}

// ---------------------------------------------------------------------------
// UI-side resolved state (textures live here)
// ---------------------------------------------------------------------------

struct UiResource {
    name: String,
    url: String,
    triggers: Vec<String>,
    tex: Option<egui::TextureHandle>,
    has_video_only: bool,
    download_url: String,
    download_filename: Option<String>,
}

struct UiSection {
    title: String,
    items: Vec<UiResource>,
}

struct UiResult {
    source_url: Option<String>,
    sections: Vec<UiSection>,
}

impl UiResult {
    fn is_empty(&self) -> bool {
        self.source_url.is_none() && self.sections.iter().all(|s| s.items.is_empty())
    }
}

/// All UI + runtime state for the Civitai view. Lives on `RightPanelState`.
pub struct CivitaiState {
    /// The image path we last kicked off a lookup for (so we refetch on change).
    loaded_key: Option<String>,
    /// Placeholder / status text shown when there's no result to display.
    status: String,
    /// The resolved, texture-loaded result, once the worker finishes.
    result: Option<UiResult>,
    rx: Option<Receiver<CivMsg>>,
    /// Flipped to ask the current worker to stop (set on every new lookup).
    cancel: Arc<AtomicBool>,

    /// Civitai reachability: 0 = checking, 1 = online, 2 = offline.
    api_status: Arc<AtomicU8>,
    monitor_started: bool,

    /// Civitai API key (plaintext in memory; stored encrypted on disk). Used for
    /// authenticated model downloads. Loaded once on first show.
    api_key: String,
    /// Folder downloaded models are saved into (like the Gelbooru downloader).
    download_dir: String,
    key_loaded: bool,
    /// Whether the API-key / download settings popup is open.
    show_settings: bool,
    /// In-flight / finished model downloads (progress rows).
    downloads: Vec<Download>,
}

impl Default for CivitaiState {
    fn default() -> Self {
        Self {
            loaded_key: None,
            status: "Select an image with Stable Diffusion metadata to see Civitai resources.".into(),
            result: None,
            rx: None,
            cancel: Arc::new(AtomicBool::new(false)),
            api_status: Arc::new(AtomicU8::new(API_CHECKING)),
            monitor_started: false,
            api_key: String::new(),
            download_dir: String::new(),
            key_loaded: false,
            show_settings: false,
            downloads: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the Civitai resources view. `metadata` is the embedded SD generation
/// metadata of the selected image (the same string the Metadata switch shows).
pub fn show(
    ui: &mut egui::Ui,
    state: &mut CivitaiState,
    current_image: Option<&Path>,
    metadata: Option<&str>,
) {
    // Spawn the API-status monitor once per session.
    if !state.monitor_started {
        state.monitor_started = true;
        start_api_monitor(Arc::clone(&state.api_status), ui.ctx().clone());
    }
    // Load the saved (encrypted) API key + download folder once.
    if !state.key_loaded {
        state.key_loaded = true;
        state.api_key = load_civitai_key();
        state.download_dir = load_download_dir();
    }

    // Drain model-download progress.
    let mut any_active = false;
    for d in &mut state.downloads {
        // Drain into a Vec first so the `&d.rx` borrow ends before we mutate `d`.
        let msgs: Vec<DlMsg> = match &d.rx {
            Some(rx) => std::iter::from_fn(|| rx.try_recv().ok()).collect(),
            None => Vec::new(),
        };
        for msg in msgs {
            match msg {
                DlMsg::Progress(r, t) => {
                    d.received = r;
                    d.total = t;
                    d.status = "Downloading…".into();
                }
                DlMsg::Done(path) => {
                    d.rx = None;
                    d.ok = true;
                    d.status = format!(
                        "Saved to {}",
                        path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
                    );
                }
                DlMsg::Error(e) => {
                    d.rx = None;
                    d.ok = false;
                    d.status = format!("Failed: {e}");
                }
            }
        }
        if d.rx.is_some() {
            any_active = true;
        }
    }
    if any_active {
        ui.ctx().request_repaint_after(Duration::from_millis(150));
    }

    // Drain worker messages.
    if let Some(rx) = &state.rx {
        let mut done = None;
        while let Ok(CivMsg::Done(res)) = rx.try_recv() {
            done = Some(res);
        }
        if let Some(res) = done {
            let resolved = resolve(ui.ctx(), *res);
            if resolved.is_empty() {
                state.status = "No Civitai resources found in metadata.".into();
                state.result = None;
            } else {
                state.result = Some(resolved);
            }
            state.rx = None;
        } else {
            ui.ctx().request_repaint_after(Duration::from_millis(150));
        }
    }

    // (Re)start a lookup whenever the selected image changes.
    let key = current_image.map(|p| p.display().to_string());
    if state.loaded_key != key {
        state.loaded_key = key;
        state.cancel.store(true, Ordering::SeqCst); // stop any in-flight worker
        state.result = None;
        state.rx = None;
        match current_image {
            Some(path) => {
                state.status = "Loading Civitai data…".into();
                start_fetch(state, path, metadata, ui.ctx());
            }
            None => {
                state.status =
                    "Select an image with Stable Diffusion metadata to see Civitai resources.".into();
            }
        }
    }

    // Round widgets in this view to match the rest of the app.
    {
        let r = egui::CornerRadius::same(10);
        let v = ui.visuals_mut();
        v.widgets.inactive.corner_radius = r;
        v.widgets.hovered.corner_radius = r;
        v.widgets.active.corner_radius = r;
    }

    // Header — title + API-status pill.
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let icon = egui::include_image!("../icons/civitai.svg");
        ui.add(egui::Image::new(icon).fit_to_exact_size(egui::vec2(22.0, 22.0)));
        ui.heading(egui::RichText::new("Civitai Resources").color(TEXT()).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            api_pill(ui, state.api_status.load(Ordering::Relaxed));
            // Settings gear next to the status: opens the API-key popup.
            let gear = egui::include_image!("../icons/settings.svg");
            if crate::svg_button(ui, gear, "Civitai API key", 18.0, crate::theme::icon_tint(MUTED())).clicked() {
                state.show_settings = !state.show_settings;
            }
        });
    });
    ui.add_space(8.0);

    // API-key / download settings popup.
    if state.show_settings {
        api_key_popup(ui.ctx(), state);
    }

    // Body. The Download buttons set `download_req`, acted on after rendering.
    let mut download_req: Option<DownloadRequest> = None;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            match &state.result {
                Some(res) => render_result(ui, res, &state.downloads, &state.download_dir, &mut download_req),
                None => {
                    ui.add_space(24.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(&state.status)
                                .color(MUTED())
                                .size(13.0),
                        );
                    });
                }
            }
        });
    if let Some(req) = download_req {
        start_download(state, req, ui.ctx());
    }
}

fn render_result(
    ui: &mut egui::Ui,
    res: &UiResult,
    downloads: &[Download],
    download_dir: &str,
    download_req: &mut Option<DownloadRequest>,
) {
    if let Some(url) = &res.source_url {
        section_label(ui, "Original Upload");
        source_link_card(ui, url);
    }
    for section in &res.sections {
        if section.items.is_empty() {
            continue;
        }
        section_label(ui, &section.title);
        for item in &section.items {
            resource_card(ui, item, downloads, download_dir, download_req);
        }
    }
}

fn section_label(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(egui::RichText::new(title).color(TEXT()).strong().size(13.0));
    ui.add_space(5.0);
}

/// The "View Original Post / Image on Civitai" link card.
fn source_link_card(ui: &mut egui::Ui, url: &str) {
    let inner = card_body(ui, |ui| {
        crate::emoji::label(
            ui,
            "🔗  View Original Post / Image on Civitai",
            TEXT(),
            12.0,
            true,
        );
    });
    let resp = inner.interact(egui::Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if resp.clicked() {
        open_url(ui.ctx(), url);
    }
}

/// A single resource card: preview thumbnail (or video placeholder) on the left,
/// name + trigger words on the right. The whole card is clickable.
fn resource_card(
    ui: &mut egui::Ui,
    data: &UiResource,
    downloads: &[Download],
    download_dir: &str,
    download_req: &mut Option<DownloadRequest>,
) {
    // Latest download for this resource (any state), driving the right-slot icon.
    let dl = downloads.iter().rev().find(|d| d.name == data.name);
    // Already installed? Check whether its file exists in the models folder.
    let installed = !download_dir.is_empty()
        && data
            .download_filename
            .as_deref()
            .map(|f| Path::new(download_dir).join(sanitize_filename(f)).exists())
            .unwrap_or(false);
    // Rect of the Download button, so a click there starts a download instead of
    // opening the page (the card-level interact below would otherwise win).
    let mut dl_rect: Option<egui::Rect> = None;
    let inner = card_body(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;

            if let Some(tex) = &data.tex {
                // Match the Java card: preview scaled to ~80px tall on the left,
                // width following the aspect ratio (capped so the text keeps room).
                ui.add(
                    egui::Image::from_texture(egui::load::SizedTexture::from_handle(tex))
                        .max_height(80.0)
                        .max_width(130.0)
                        .corner_radius(8),
                );
            } else if data.has_video_only {
                video_placeholder(ui);
            }

            ui.vertical(|ui| {
                crate::emoji::label(ui, &data.name, TEXT(), 12.0, true);
                if !data.triggers.is_empty() {
                    ui.add_space(2.0);
                    crate::emoji::label(
                        ui,
                        &format!("Triggers: {}", data.triggers.join(", ")),
                        MUTED(),
                        11.0,
                        false,
                    );
                }
            });

            // Right side, vertically centred. The slot reflects the download state:
            //   downloading → percentage · done → green check · failed → red
            //   warning (instant hover shows the error) · idle → download arrow.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(2.0);
                match dl {
                    Some(d) if d.rx.is_some() => {
                        let pct = if d.total > 0 {
                            (d.received as f64 / d.total as f64 * 100.0).round() as u32
                        } else {
                            0
                        };
                        ui.label(
                            egui::RichText::new(format!("{pct}%"))
                                .color(crate::theme::ACCENT1())
                                .strong()
                                .size(13.0),
                        );
                    }
                    Some(d) if d.ok => {
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/checkmark.svg"))
                                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                .tint(egui::Color32::from_rgb(46, 160, 67)),
                        )
                        .on_hover_text(&d.status);
                    }
                    Some(d) => {
                        // Failed — orange warning with an immediately-shown tooltip.
                        ui.scope(|ui| {
                            ui.style_mut().interaction.tooltip_delay = 0.0;
                            ui.add(
                                egui::Image::new(egui::include_image!("../icons/warning.svg"))
                                    .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                    .sense(egui::Sense::hover())
                                    .tint(egui::Color32::from_rgb(235, 150, 45)),
                            )
                            .on_hover_text(&d.status);
                        });
                    }
                    None if installed => {
                        // Already downloaded — green check.
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/checkmark.svg"))
                                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                .tint(egui::Color32::from_rgb(46, 160, 67)),
                        )
                        .on_hover_text("Already in your models folder");
                    }
                    None => {
                        // Blue download arrow.
                        let arrow = egui::include_image!("../icons/arrow_circle_down.svg");
                        let r = crate::svg_button(ui, arrow, "Download into your models folder", 22.0, crate::theme::ACCENT1());
                        dl_rect = Some(r.rect);
                    }
                }
            });
        });
    });

    let resp = inner.interact(egui::Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if resp.clicked() {
        // A click on the Download button starts a download; anywhere else on the
        // card opens the resource page. We test by click position because the
        // card's interact can swallow the button's own click event.
        let on_download = matches!(
            (resp.interact_pointer_pos(), dl_rect),
            (Some(p), Some(r)) if r.contains(p)
        );
        if on_download {
            *download_req = Some(DownloadRequest {
                name: data.name.clone(),
                download_url: data.download_url.clone(),
                filename: data.download_filename.clone(),
            });
        } else {
            open_url(ui.ctx(), &data.url);
        }
    }
}

/// A small "Video" preview placeholder (the resource only has a video preview).
fn video_placeholder(ui: &mut egui::Ui) {
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(80.0, 72.0), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 8, egui::Color32::from_black_alpha(60));
    // A simple white play triangle in the centre.
    let c = rect.center();
    let s = 12.0;
    let tri = [
        egui::pos2(c.x - s * 0.4, c.y - s * 0.6),
        egui::pos2(c.x - s * 0.4, c.y + s * 0.6),
        egui::pos2(c.x + s * 0.6, c.y),
    ];
    painter.add(egui::Shape::convex_polygon(
        tri.to_vec(),
        egui::Color32::from_white_alpha(200),
        egui::Stroke::NONE,
    ));
}

/// The shared rounded card frame; returns the frame's allocated response so the
/// caller can make the whole card clickable.
fn card_body(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) -> egui::Response {
    let r = egui::Frame::new()
        .fill(FIELD())
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::same(10))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
    ui.add_space(4.0);
    r.response
}

/// Draw the coloured "API: …" status pill (same look as the downloader's).
/// The Civitai settings popup: API key (stored encrypted via src/secret.rs) and a
/// download folder for models. A modern, sectioned card.
fn api_key_popup(ctx: &egui::Context, state: &mut CivitaiState) {
    egui::Window::new("")
        .id(egui::Id::new("civitai_settings"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .frame(
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
                }),
        )
        .show(ctx, |ui| {
            ui.set_width(380.0);
            let radius = egui::CornerRadius::same(10);
            {
                let v = ui.visuals_mut();
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.active.corner_radius = radius;
                v.extreme_bg_color = FIELD();
            }

            // Title row.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/civitai.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0)),
                );
                ui.heading(egui::RichText::new("Civitai Settings").color(TEXT()).strong().size(17.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(egui::RichText::new("✕").size(14.0)).frame(false))
                        .on_hover_text("Close")
                        .clicked()
                    {
                        state.show_settings = false;
                    }
                });
            });
            ui.add_space(14.0);

            // API key section.
            ui.label(egui::RichText::new("API KEY").color(MUTED()).strong().size(11.0));
            ui.add_space(4.0);
            ui.add(
                egui::TextEdit::singleline(&mut state.api_key)
                    .password(true)
                    .desired_width(f32::INFINITY)
                    .margin(egui::Margin::symmetric(10, 8))
                    .hint_text("Paste your Civitai API key"),
            );
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Stored encrypted on this device.").color(MUTED()).size(10.5),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.hyperlink_to(
                        egui::RichText::new("Get a key →").size(10.5),
                        "https://civitai.com/user/account",
                    );
                });
            });

            ui.add_space(14.0);

            // Download folder section.
            ui.label(egui::RichText::new("MODELS FOLDER").color(MUTED()).strong().size(11.0));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let folder = egui::include_image!("../icons/folder.svg");
                if crate::svg_button(ui, folder, "Choose folder", 30.0, crate::theme::icon_tint(MUTED())).clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        state.download_dir = dir.display().to_string();
                    }
                }
                ui.add(
                    egui::TextEdit::singleline(&mut state.download_dir)
                        .desired_width(f32::INFINITY)
                        .margin(egui::Margin::symmetric(10, 8))
                        .hint_text("Where downloaded models are saved"),
                );
            });

            ui.add_space(18.0);

            // Actions.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                let save = egui::Button::new(
                    egui::RichText::new("Save").color(egui::Color32::WHITE).strong(),
                )
                .fill(crate::theme::ACCENT1());
                if ui.add_sized(egui::vec2(90.0, 32.0), save).clicked() {
                    save_civitai_key(&state.api_key);
                    save_download_dir(&state.download_dir);
                    state.show_settings = false;
                }
                if ui.add_sized(egui::vec2(80.0, 32.0), egui::Button::new("Clear key")).clicked() {
                    state.api_key.clear();
                    save_civitai_key("");
                }
            });
        });
}

/// Validate + start a model download on a worker thread.
fn start_download(state: &mut CivitaiState, req: DownloadRequest, ctx: &egui::Context) {
    let dir = state.download_dir.trim().to_string();
    if dir.is_empty() {
        // No folder yet — open settings so the user can pick one.
        state.show_settings = true;
        state.downloads.push(Download {
            name: req.name,
            rx: None,
            received: 0,
            total: 0,
            status: "Failed: set a models folder in settings".into(),
            ok: false,
        });
        return;
    }
    let key = state.api_key.trim().to_string();
    let (tx, rx) = mpsc::channel();
    state.downloads.push(Download {
        name: req.name.clone(),
        rx: Some(rx),
        received: 0,
        total: 0,
        status: "Starting…".into(),
        ok: false,
    });
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        run_download(req.download_url, req.filename, key, std::path::PathBuf::from(dir), tx, ctx);
    });
}

/// Stream a Civitai model file to `dir`, reporting progress. The token goes in the
/// query string (not a header), since Civitai 302-redirects to S3 and strips
/// headers on the cross-domain hop.
fn run_download(
    mut url: String,
    filename_hint: Option<String>,
    key: String,
    dir: std::path::PathBuf,
    tx: Sender<DlMsg>,
    ctx: egui::Context,
) {
    use std::io::{Read, Write};

    if !key.is_empty() {
        url.push(if url.contains('?') { '&' } else { '?' });
        url.push_str("token=");
        url.push_str(&key);
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .http_status_as_error(false)
        // No global/body timeout — model files are large; only bound setup phases.
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_send_request(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .build()
        .into();

    let send_err = |e: String| {
        let _ = tx.send(DlMsg::Error(e));
        ctx.request_repaint();
    };

    let resp = match agent.get(&url).header("User-Agent", USER_AGENT).call() {
        Ok(r) => r,
        Err(e) => return send_err(format!("request failed: {e}")),
    };
    let mut resp = resp;
    let status = resp.status().as_u16();
    if status != 200 {
        return send_err(if status == 401 || status == 403 {
            "unauthorized — check your API key".into()
        } else {
            format!("HTTP {status}")
        });
    }

    // Prefer the server-suggested filename, then the API file name, then a default.
    let cd = resp
        .headers()
        .get("content-disposition")
        .and_then(|v| v.to_str().ok())
        .and_then(filename_from_content_disposition);
    let name = cd
        .or(filename_hint)
        .unwrap_or_else(|| "civitai-model.safetensors".to_string());
    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if let Err(e) = std::fs::create_dir_all(&dir) {
        return send_err(format!("cannot create folder: {e}"));
    }
    let dest = unique_path(&dir, &sanitize_filename(&name));
    let tmp = dest.with_extension("part");
    let mut out = match std::fs::File::create(&tmp) {
        Ok(f) => f,
        Err(e) => return send_err(format!("cannot write file: {e}")),
    };

    let mut reader = resp.body_mut().as_reader();
    let mut buf = vec![0u8; 1 << 16];
    let mut got: u64 = 0;
    let mut last_sent = 0u64;
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return send_err(format!("download error: {e}")),
        };
        if out.write_all(&buf[..n]).is_err() {
            return send_err("write error".into());
        }
        got += n as u64;
        if got - last_sent >= 1 << 20 {
            last_sent = got;
            let _ = tx.send(DlMsg::Progress(got, total));
            ctx.request_repaint();
        }
    }
    out.flush().ok();
    drop(out);
    if let Err(e) = std::fs::rename(&tmp, &dest) {
        return send_err(format!("could not finalize: {e}"));
    }
    let _ = tx.send(DlMsg::Done(dest));
    ctx.request_repaint();
}

/// Extract a filename from a `Content-Disposition` header value.
fn filename_from_content_disposition(cd: &str) -> Option<String> {
    // filename*=UTF-8''name takes priority over filename="name".
    if let Some(i) = cd.to_ascii_lowercase().find("filename*=") {
        let rest = &cd[i + "filename*=".len()..];
        let val = rest.split(';').next().unwrap_or(rest).trim();
        let val = val.rsplit("''").next().unwrap_or(val);
        let decoded = percent_decode(val.trim_matches('"'));
        if !decoded.is_empty() {
            return Some(decoded);
        }
    }
    if let Some(i) = cd.to_ascii_lowercase().find("filename=") {
        let rest = &cd[i + "filename=".len()..];
        let val = rest.split(';').next().unwrap_or(rest).trim().trim_matches('"');
        if !val.is_empty() {
            return Some(val.to_string());
        }
    }
    None
}

/// Minimal percent-decoding for Content-Disposition filenames.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Strip path separators / illegal characters from a download filename.
fn sanitize_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') { '_' } else { c })
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() { "civitai-model.safetensors".to_string() } else { cleaned.to_string() }
}

/// A non-clashing path in `dir`, appending " (n)" before the extension if needed.
fn unique_path(dir: &Path, name: &str) -> std::path::PathBuf {
    let base = dir.join(name);
    if !base.exists() {
        return base;
    }
    let (stem, ext) = match name.rfind('.') {
        Some(i) => (&name[..i], &name[i..]),
        None => (name, ""),
    };
    for i in 1..100_000 {
        let cand = dir.join(format!("{stem} ({i}){ext}"));
        if !cand.exists() {
            return cand;
        }
    }
    base
}

/// Path of the saved download-folder setting (plaintext — not sensitive).
fn download_dir_path() -> std::path::PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow").join("civitai_download_dir.txt"))
        .unwrap_or_else(|| std::path::PathBuf::from("civitai_download_dir.txt"))
}

fn load_download_dir() -> String {
    std::fs::read_to_string(download_dir_path()).map(|s| s.trim().to_string()).unwrap_or_default()
}

fn save_download_dir(dir: &str) {
    let path = download_dir_path();
    if let Some(d) = path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let _ = std::fs::write(&path, dir.trim());
}

/// Path of the encrypted Civitai API key file in the app config dir.
fn civitai_key_path() -> std::path::PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow").join("civitai_api_key.dat"))
        .unwrap_or_else(|| std::path::PathBuf::from("civitai_api_key.dat"))
}

/// Load the saved Civitai API key (decrypted), or "" if none/unreadable.
fn load_civitai_key() -> String {
    std::fs::read_to_string(civitai_key_path())
        .ok()
        .map(|s| crate::secret::unprotect(s.trim()))
        .unwrap_or_default()
}

/// Save the Civitai API key encrypted. An empty key removes the file.
fn save_civitai_key(key: &str) {
    let path = civitai_key_path();
    let trimmed = key.trim();
    if trimmed.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, crate::secret::protect(trimmed));
}

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

fn open_url(ctx: &egui::Context, url: &str) {
    if url.trim().is_empty() {
        return;
    }
    ctx.open_url(egui::OpenUrl::new_tab(url));
}

/// Convert a worker `CivResult` into UI state, uploading each decoded thumbnail
/// to the GPU as a texture (must happen on the UI thread).
fn resolve(ctx: &egui::Context, res: CivResult) -> UiResult {
    let sections = res
        .sections
        .into_iter()
        .map(|s| UiSection {
            title: s.title,
            items: s
                .items
                .into_iter()
                .map(|f| {
                    let tex = f.image.map(|img| {
                        ctx.load_texture(format!("civ_{}", f.url), img, egui::TextureOptions::LINEAR)
                    });
                    UiResource {
                        name: f.name,
                        url: f.url,
                        triggers: f.triggers,
                        tex,
                        has_video_only: f.has_video_only,
                        download_url: f.download_url,
                        download_filename: f.download_filename,
                    }
                })
                .collect(),
        })
        .collect();
    UiResult { source_url: res.source_url, sections }
}

// ---------------------------------------------------------------------------
// Worker plumbing
// ---------------------------------------------------------------------------

fn start_fetch(state: &mut CivitaiState, path: &Path, metadata: Option<&str>, ctx: &egui::Context) {
    let cancel = Arc::new(AtomicBool::new(false));
    state.cancel = Arc::clone(&cancel);
    let (tx, rx) = mpsc::channel();
    state.rx = Some(rx);

    let meta = metadata.map(|s| s.to_string());
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string());
    let ctx = ctx.clone();

    std::thread::spawn(move || run_fetch(meta, filename, cancel, tx, ctx));
}

fn run_fetch(
    metadata: Option<String>,
    filename: Option<String>,
    cancel: Arc<AtomicBool>,
    tx: Sender<CivMsg>,
    ctx: egui::Context,
) {
    let meta = metadata.unwrap_or_default();

    let source_url = extract_image_link(&meta, filename.as_deref());
    let tasks = if meta.trim().is_empty() {
        Vec::new()
    } else {
        parse_metadata_for_tasks(&meta)
    };

    // Nothing to do — return an (empty) result without touching the network.
    if tasks.is_empty() && source_url.is_none() {
        let _ = tx.send(CivMsg::Done(Box::new(CivResult { source_url: None, sections: Vec::new() })));
        ctx.request_repaint();
        return;
    }

    // native-tls => Windows SChannel (same rationale as download.rs / ai_models.rs).
    // PlatformVerifier => validate against the OS cert store (with its AIA
    // intermediate fetching) rather than ureq's bundled webpki roots. Civitai's
    // image CDN (image.civitai.com / image-b2.civitai.com) serves an incomplete
    // chain that fails webpki path-building ("unable to find any user-specified
    // roots in the final cert chain"), so without this the preview thumbnails
    // never download even though the api.civitai.com JSON lookups succeed.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .timeout_global(Some(Duration::from_secs(15)))
        .max_redirects(10)
        .http_status_as_error(false)
        .build()
        .into();

    let mut model: Option<Fetched> = None;
    let mut vae: Option<Fetched> = None;
    let mut loras: Vec<Fetched> = Vec::new();
    let mut lycoris: Vec<Fetched> = Vec::new();
    let mut embeddings: Vec<Fetched> = Vec::new();
    let mut processed: HashSet<String> = HashSet::new();

    for task in &tasks {
        if cancel.load(Ordering::SeqCst) {
            return; // navigated away — drop this stale lookup entirely
        }

        let mut info = None;
        if let Some(v) = &task.version_id {
            if !v.is_empty() {
                info = fetch_resource_info(&agent, task.kind, LookupMethod::VersionId, v);
            }
        }
        if info.is_none() {
            if let Some(h) = &task.hash {
                if !h.is_empty() {
                    info = fetch_resource_info(&agent, task.kind, LookupMethod::Hash, h);
                }
            }
        }
        if info.is_none() {
            if let Some(n) = &task.name {
                if !n.is_empty() {
                    info = fetch_resource_info(&agent, task.kind, LookupMethod::Name, n);
                }
            }
        }

        let Some(info) = info else { continue };
        if !processed.insert(info.url.clone()) {
            continue;
        }

        match task.kind {
            ItemType::Model => {
                if model.is_none() {
                    model = Some(info);
                }
            }
            ItemType::Vae => {
                if vae.is_none() {
                    vae = Some(info);
                }
            }
            ItemType::Lora => {
                let t = info.resource_type.to_lowercase();
                if t == "locon" || t == "lycoris" || t == "dora" {
                    lycoris.push(info);
                } else {
                    loras.push(info);
                }
            }
            ItemType::Embedding => embeddings.push(info),
        }
    }

    let mut sections = Vec::new();
    if let Some(m) = model {
        sections.push(FetchedSection { title: "Base Model".into(), items: vec![m] });
    }
    if let Some(v) = vae {
        sections.push(FetchedSection { title: "VAE Used".into(), items: vec![v] });
    }
    if !loras.is_empty() {
        sections.push(FetchedSection { title: "LoRA Models".into(), items: loras });
    }
    if !lycoris.is_empty() {
        sections.push(FetchedSection { title: "LyCORIS Models".into(), items: lycoris });
    }
    if !embeddings.is_empty() {
        sections.push(FetchedSection { title: "Embeddings".into(), items: embeddings });
    }

    let _ = tx.send(CivMsg::Done(Box::new(CivResult { source_url, sections })));
    ctx.request_repaint();
}

/// Poll civitai.com every 5s and store the reachability in `status`.
fn start_api_monitor(status: Arc<AtomicU8>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::NativeTls)
                    .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                    .build(),
            )
            .timeout_global(Some(Duration::from_secs(8)))
            .http_status_as_error(false)
            .build()
            .into();

        loop {
            let online = agent
                .get(API_PING)
                .header("User-Agent", USER_AGENT)
                .call()
                .map(|r| r.status().as_u16() > 0)
                .unwrap_or(false);
            let new = if online { API_ONLINE } else { API_OFFLINE };
            if status.swap(new, Ordering::Relaxed) != new {
                ctx.request_repaint();
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    });
}

// ---------------------------------------------------------------------------
// Civitai API lookups
// ---------------------------------------------------------------------------

fn fetch_resource_info(
    agent: &ureq::Agent,
    kind: ItemType,
    method: LookupMethod,
    value: &str,
) -> Option<Fetched> {
    let mut model_id: Option<String> = None;
    let mut version_id: Option<String> = None;

    match method {
        LookupMethod::VersionId => {
            let url = format!("https://civitai.com/api/v1/model-versions/{value}");
            let data = get_json(agent, &url)?;
            model_id = as_text(data.get("modelId")?);
            version_id = as_text(data.get("id").unwrap_or(&serde_json::Value::Null));
        }
        LookupMethod::Hash => {
            let url = format!("https://civitai.com/api/v1/model-versions/by-hash/{value}");
            let data = get_json(agent, &url)?;
            model_id = as_text(data.get("modelId")?);
            version_id = as_text(data.get("id").unwrap_or(&serde_json::Value::Null));
        }
        LookupMethod::Name => {
            // Clean the on-disk filename into a search query: drop any path, drop
            // the extension, turn separators into spaces.
            let mut clean = value.to_string();
            if let Some(pos) = clean.rfind(['/', '\\']) {
                clean = clean[pos + 1..].to_string();
            }
            if let Some(dot) = clean.rfind('.') {
                if dot > 0 {
                    clean = clean[..dot].to_string();
                }
            }
            clean = clean.replace(['_', '-'], " ");

            let type_filter = match kind {
                ItemType::Model => "&types=Checkpoint",
                ItemType::Lora => "&types=LORA&types=LoCon&types=Lycoris&types=DoRA",
                ItemType::Vae => "&types=VAE",
                ItemType::Embedding => "&types=TextualInversion",
            };
            let url = format!(
                "https://civitai.com/api/v1/models?query={}&limit=1{}",
                percent_encode(&clean),
                type_filter
            );
            let data = get_json(agent, &url)?;
            let items = data.get("items")?.as_array()?;
            let first = items.first()?;
            let fetched_name = first.get("name").and_then(|v| v.as_str()).unwrap_or("");

            if name_matches(&clean, fetched_name) {
                model_id = as_text(first.get("id").unwrap_or(&serde_json::Value::Null));
            }
        }
    }

    let model_id = model_id?;
    get_formatted_model_data(agent, &model_id, version_id.as_deref())
}

/// The fuzzy name-match the Java used to reject obviously-wrong search hits.
fn name_matches(query: &str, fetched: &str) -> bool {
    let norm = |s: &str| -> String {
        s.to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { ' ' })
            .collect()
    };
    let q = norm(query);
    let f = norm(fetched);
    let q = q.trim();
    let f = f.trim();

    if q.is_empty() || f.is_empty() || q.contains(f) || f.contains(q) {
        return true;
    }
    let q_words: HashSet<&str> = q.split_whitespace().collect();
    f.split_whitespace().any(|w| w.len() >= 4 && q_words.contains(w))
}

fn get_formatted_model_data(
    agent: &ureq::Agent,
    model_id: &str,
    specific_version_id: Option<&str>,
) -> Option<Fetched> {
    let url = format!("https://civitai.com/api/v1/models/{model_id}");
    let model_data = get_json(agent, &url)?;

    let resource_type = model_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();

    let versions = model_data.get("modelVersions").and_then(|v| v.as_array());

    // Prefer the exact version the metadata referenced; otherwise the newest.
    let version = match (specific_version_id, versions) {
        (Some(vid), Some(arr)) => arr
            .iter()
            .find(|v| as_text(v.get("id").unwrap_or(&serde_json::Value::Null)).as_deref() == Some(vid))
            .or_else(|| arr.first()),
        (None, Some(arr)) => arr.first(),
        _ => None,
    }?;

    let vid = as_text(version.get("id").unwrap_or(&serde_json::Value::Null)).unwrap_or_default();
    let page_url = format!("https://civitai.com/models/{model_id}?modelVersionId={vid}");

    // Pick the primary file for download (its own downloadUrl + name), falling
    // back to the version-level download endpoint.
    let primary_file = version
        .get("files")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|f| f.get("primary").and_then(|v| v.as_bool()).unwrap_or(false))
                .or_else(|| arr.first())
        });
    let download_url = primary_file
        .and_then(|f| f.get("downloadUrl").and_then(|v| v.as_str()))
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("https://civitai.com/api/download/models/{vid}"));
    let download_filename = primary_file
        .and_then(|f| f.get("name").and_then(|v| v.as_str()))
        .map(|s| s.to_string());

    // Pick the first still-image preview; note if the version only has videos.
    let mut image_url: Option<String> = None;
    let mut has_video = false;
    let mut has_image = false;
    if let Some(images) = version.get("images").and_then(|v| v.as_array()) {
        for img in images {
            let ty = img.get("type").and_then(|v| v.as_str()).unwrap_or("image").to_lowercase();
            if ty == "video" {
                has_video = true;
                continue;
            }
            if image_url.is_none() {
                image_url = img.get("url").and_then(|v| v.as_str()).map(|s| s.to_string());
                has_image = true;
            }
        }
    }
    let has_video_only = has_video && !has_image;

    let image = image_url.and_then(|u| {
        let sized = sized_image_url(&u, 200);
        get_bytes(agent, &sized).and_then(|b| decode_thumb(&b))
    });

    let model_name = model_data.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let version_name = version.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let name = format!("{model_name} - {version_name}");

    let triggers = version
        .get("trainedWords")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.as_str().map(|s| s.to_string()))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    Some(Fetched {
        name,
        url: page_url,
        triggers,
        image,
        has_video_only,
        resource_type,
        download_url,
        download_filename,
    })
}

/// Rewrite a Civitai image URL to request a width-limited render from the CDN.
///
/// Civitai media URLs carry their transform as a *path* segment
/// (`…/<bucket>/<uuid>/<transform>/<file>`), e.g. `original=true` or `width=450`.
/// A `?width=` *query* (what the API examples and the Java original use) is
/// silently ignored by the CDN, which then redirects to the full-size original —
/// for large previews that can be many MB (one CyberRealistic Z-Image preview is a
/// 12 MB PNG), blowing past [`get_bytes`]'s read cap so the truncated bytes fail to
/// decode and no thumbnail appears. Replacing the transform path segment instead
/// makes the CDN resize server-side (a ~50 KB JPEG).
fn sized_image_url(url: &str, width: u32) -> String {
    let mut parts: Vec<String> = url.split('/').map(|s| s.to_string()).collect();
    if parts.len() < 2 {
        return url.to_string();
    }
    let last = parts.len() - 1;
    let transform = format!("width={width}");
    // The transform sits just before the filename and always contains '='; if the
    // URL has none (older form `…/<uuid>/<file>`), insert one before the filename.
    if parts[last - 1].contains('=') {
        parts[last - 1] = transform;
    } else {
        parts.insert(last, transform);
    }
    parts.join("/")
}

/// Decode downloaded preview bytes into a small `ColorImage` (max ~200px).
fn decode_thumb(bytes: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory(bytes).ok()?;
    let thumb = img.thumbnail(200, 200).to_rgba8();
    let size = [thumb.width() as usize, thumb.height() as usize];
    Some(egui::ColorImage::from_rgba_unmultiplied(size, thumb.as_raw()))
}

fn get_json(agent: &ureq::Agent, url: &str) -> Option<serde_json::Value> {
    let mut resp = agent
        .get(url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/json")
        .call()
        .ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    let body = resp.body_mut().read_to_string().ok()?;
    serde_json::from_str(&body).ok()
}

fn get_bytes(agent: &ureq::Agent, url: &str) -> Option<Vec<u8>> {
    let mut resp = agent.get(url).header("User-Agent", USER_AGENT).call().ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    let mut buf = Vec::new();
    // Preview thumbnails are tiny; cap the read defensively at 8 MiB.
    resp.body_mut()
        .as_reader()
        .take(8 * 1024 * 1024)
        .read_to_end(&mut buf)
        .ok()?;
    Some(buf)
}

/// Treat a JSON value as text, coercing numbers/bools (Civitai ids are numeric).
fn as_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Percent-encode a query value (RFC 3986 unreserved set kept literal).
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Metadata parsing (ported 1:1 from CivitaiInfo.java, regex → manual scanning)
// ---------------------------------------------------------------------------

/// Find a link back to the source Civitai image/post, from the metadata text or
/// a purely-numeric filename.
fn extract_image_link(metadata: &str, filename: Option<&str>) -> Option<String> {
    // Earliest of the two prefixes wins (mirrors the single combined Java regex).
    let prefixes = ["https://civitai.com/images/", "https://civitai.com/posts/"];
    let mut best: Option<(usize, String)> = None;
    for prefix in prefixes {
        if let Some(pos) = metadata.find(prefix) {
            let rest = &metadata[pos + prefix.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                let url = format!("{prefix}{digits}");
                if best.as_ref().map(|(p, _)| pos < *p).unwrap_or(true) {
                    best = Some((pos, url));
                }
            }
        }
    }
    if let Some((_, url)) = best {
        return Some(url);
    }

    if let Some(fname) = filename {
        let base = fname.rsplit_once('.').map(|(b, _)| b).unwrap_or(fname);
        if (6..=15).contains(&base.len()) && base.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("https://civitai.com/images/{base}"));
        }
    }
    None
}

fn parse_metadata_for_tasks(metadata: &str) -> Vec<Task> {
    // Civitai-generator metadata is authoritative when present.
    let civitai = parse_civitai_generator(metadata);
    if !civitai.is_empty() {
        return civitai;
    }

    let mut tasks = parse_a1111(metadata);

    // ComfyUI: either the whole string is JSON, or it's embedded after "prompt:\n".
    let json_str = if metadata.trim_start().starts_with('{') {
        Some(metadata.to_string())
    } else if let Some(pi) = metadata.find("prompt:\n{") {
        let start = pi + "prompt:\n".len();
        let end = metadata[start..]
            .find("\nworkflow:\n")
            .map(|e| start + e)
            .unwrap_or(metadata.len());
        Some(metadata[start..end].trim().to_string())
    } else {
        None
    };
    if let Some(js) = json_str {
        if let Ok(root) = serde_json::from_str::<serde_json::Value>(&js) {
            if root.is_object() {
                tasks.extend(parse_comfyui(&root));
            }
        }
    }

    tasks
}

fn parse_civitai_generator(metadata: &str) -> Vec<Task> {
    let mut tasks = Vec::new();
    let Some(idx) = metadata.find("Civitai resources:") else {
        return tasks;
    };
    let start = idx + "Civitai resources:".len();
    let rest = &metadata[start..];
    let mut block = match rest.find("Civitai metadata:") {
        Some(m) => rest[..m].trim().to_string(),
        None => rest.trim().to_string(),
    };
    if block.ends_with(',') {
        block.pop();
        block = block.trim().to_string();
    }
    let block: String = block.chars().filter(|&c| c != '\n' && c != '\r').collect();
    // The metadata may have trailing junk after the `]` (e.g. EXIF/scan tail), so
    // clip to the outermost `[ … ]` before parsing.
    let block = match (block.find('['), block.rfind(']')) {
        (Some(a), Some(b)) if b >= a => block[a..=b].to_string(),
        _ => block,
    };

    if let Ok(arr) = serde_json::from_str::<serde_json::Value>(&block) {
        if let Some(items) = arr.as_array() {
            for node in items {
                // The resource type and model-version id come from either the old
                // `type` + `modelVersionId` fields, or the newer Civitai AIR id
                // (`"air":"urn:air:<eco>:<type>:<source>:<modelId>@<versionId>"`).
                let mut type_str =
                    node.get("type").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
                let mut ver = node.get("modelVersionId").and_then(as_text_opt).unwrap_or_default();

                if ver.is_empty() {
                    if let Some(air) = node.get("air").and_then(|v| v.as_str()) {
                        if let Some((air_type, version_id)) = parse_air(air) {
                            if type_str.is_empty() {
                                type_str = air_type;
                            }
                            ver = version_id;
                        }
                    }
                }

                if ver.is_empty() || ver == "0" {
                    continue;
                }
                let kind = match type_str.as_str() {
                    "checkpoint" => Some(ItemType::Model),
                    "lora" | "locon" | "lycoris" | "dora" => Some(ItemType::Lora),
                    "textualinversion" | "embedding" => Some(ItemType::Embedding),
                    "vae" => Some(ItemType::Vae),
                    _ => None,
                };
                if let Some(kind) = kind {
                    tasks.push(Task::version(kind, ver));
                }
            }
        }
    }
    tasks
}

/// Parse a Civitai AIR identifier
/// `urn:air:<ecosystem>:<type>:<source>:<modelId>@<versionId>` into
/// `(type, versionId)`. The id segment (with the `@`) is always last and the type
/// segment precedes the source; returns `None` if it doesn't match that shape.
fn parse_air(air: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = air.split(':').collect();
    if parts.len() < 6 || parts[0] != "urn" || parts[1] != "air" {
        return None;
    }
    let type_str = parts[3].to_lowercase();
    let (_model_id, version_id) = parts.last().unwrap().split_once('@')?;
    if version_id.is_empty() {
        return None;
    }
    Some((type_str, version_id.to_string()))
}

fn parse_comfyui(root: &serde_json::Value) -> Vec<Task> {
    let mut tasks = Vec::new();
    let node_dict = match root.get("nodes") {
        Some(n) if n.is_object() => n,
        _ => root,
    };
    let Some(obj) = node_dict.as_object() else {
        return tasks;
    };

    for node in obj.values() {
        if !node.is_object() {
            continue;
        }
        let class_type = node.get("class_type").and_then(|v| v.as_str()).unwrap_or("");
        let inputs = node.get("inputs");
        let input_str = |key: &str| -> Option<String> {
            inputs
                .and_then(|i| i.get(key))
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
        };

        match class_type {
            "CheckpointLoader" | "CheckpointLoaderSimple" => {
                if let Some(n) = input_str("ckpt_name") {
                    tasks.push(Task::name(ItemType::Model, n));
                }
            }
            "UNETLoader" => {
                if let Some(n) = input_str("unet_name") {
                    tasks.push(Task::name(ItemType::Model, n));
                }
            }
            "LoraLoader" | "LoraLoaderModelOnly" | "LycORISLoader" | "CR Lora Loader" => {
                if let Some(n) = input_str("lora_name") {
                    tasks.push(Task::name(ItemType::Lora, n));
                }
            }
            "Power Lora Loader (rgthree)" => {
                if let Some(io) = inputs.and_then(|i| i.as_object()) {
                    for (field, lnode) in io {
                        if field.starts_with("lora_") && lnode.is_object() {
                            let on = lnode.get("on").and_then(|v| v.as_bool()).unwrap_or(true);
                            if on {
                                if let Some(lname) =
                                    lnode.get("lora").and_then(|v| v.as_str()).filter(|s| !s.is_empty())
                                {
                                    tasks.push(Task::name(ItemType::Lora, lname.to_string()));
                                }
                            }
                        }
                    }
                }
            }
            "VAELoader" => {
                if let Some(n) = input_str("vae_name") {
                    tasks.push(Task::name(ItemType::Vae, n));
                }
            }
            _ => {}
        }
    }
    tasks
}

fn parse_a1111(metadata: &str) -> Vec<Task> {
    let mut tasks = Vec::new();
    let mut claimed_hashes: HashSet<String> = HashSet::new();
    let mut claimed_names: HashSet<String> = HashSet::new();

    // Model hash.
    if let Some(h) = scan_hash_after(metadata, "Model hash:") {
        if claimed_hashes.insert(h.clone()) {
            tasks.push(Task::hash(ItemType::Model, h));
        }
    }
    // VAE hash.
    if let Some(h) = scan_hash_after(metadata, "VAE hash:") {
        if claimed_hashes.insert(h.clone()) {
            tasks.push(Task::hash(ItemType::Vae, h));
        }
    }

    // "Lora hashes:" / "Lyco hashes:" / "LyCORIS hashes:" — a block of hashes.
    for label in ["Lora hashes:", "Lyco hashes:", "LyCORIS hashes:"] {
        if let Some(block) = label_value_block(metadata, label) {
            for h in hex_runs(&block) {
                if claimed_hashes.insert(h.clone()) {
                    tasks.push(Task::hash(ItemType::Lora, h));
                }
            }
        }
    }

    // "Hashes: { ... }" — a JSON object of resource → hash.
    if let Some(map) = hashes_block(metadata) {
        for (key, hash) in map {
            let hash = hash.trim().to_lowercase();
            if hash.is_empty() {
                continue;
            }
            let k = key.to_lowercase();
            if k.starts_with("lora:") || k.starts_with("lyco:") || k.starts_with("dora:") {
                if claimed_hashes.insert(hash.clone()) {
                    tasks.push(Task::hash(ItemType::Lora, hash));
                }
            } else if let Some(name) = k.strip_prefix("embed:") {
                let name = name.trim().to_string();
                if claimed_hashes.insert(hash.clone()) {
                    tasks.push(Task::embed(hash, name.clone()));
                    claimed_names.insert(name.to_lowercase());
                }
            }
        }
    }

    // "TI hashes:" — embeddings as `name: hash` pairs.
    if let Some(block) = label_value_block(metadata, "TI hashes:") {
        for part in block.split(',') {
            if let Some((name, tail)) = part.split_once(':') {
                let name = name.trim();
                let hash: String = tail
                    .trim()
                    .chars()
                    .take_while(|c| c.is_ascii_hexdigit())
                    .collect();
                let hash = hash.to_lowercase();
                if name.is_empty() || !(8..=12).contains(&hash.len()) {
                    continue;
                }
                if claimed_hashes.insert(hash.clone()) {
                    tasks.push(Task::embed(hash, name.to_string()));
                    claimed_names.insert(name.to_lowercase());
                }
            }
        }
    }

    // Prompt-embedded LoRA references: <lora:name:…>, <lyco:…>, <dora:…>.
    for tag in ["<lora:", "<lyco:", "<dora:"] {
        let mut search = metadata;
        while let Some(pos) = search.find(tag) {
            let after = &search[pos + tag.len()..];
            let name: String = after.chars().take_while(|&c| c != ':' && c != '>').collect();
            let name = name.trim().to_string();
            if !name.is_empty() && claimed_names.insert(name.to_lowercase()) {
                tasks.push(Task::name(ItemType::Lora, name));
            }
            search = after;
        }
    }

    tasks
}

/// Read a hex value immediately following `label` (e.g. "Model hash: a1b2c3").
fn scan_hash_after(metadata: &str, label: &str) -> Option<String> {
    let pos = metadata.find(label)?;
    let rest = metadata[pos + label.len()..].trim_start();
    let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    if hex.is_empty() {
        None
    } else {
        Some(hex.to_lowercase())
    }
}

/// The value after `label`: a quoted string if one follows, else the rest of the
/// line. Used for the various "… hashes:" blocks.
fn label_value_block(metadata: &str, label: &str) -> Option<String> {
    let pos = metadata.find(label)?;
    let rest = metadata[pos + label.len()..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"').unwrap_or(stripped.len());
        Some(stripped[..end].to_string())
    } else {
        let end = rest.find('\n').unwrap_or(rest.len());
        Some(rest[..end].to_string())
    }
}

/// Extract maximal hex runs, normalised to AutoV2-length (≤12) lowercase hashes.
/// Civitai's by-hash endpoint accepts the 12-char prefix of a full SHA-256.
fn hex_runs(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.len() >= 8 {
            out.push(cur.chars().take(12).collect::<String>().to_lowercase());
        }
        cur.clear();
    };
    for c in s.chars() {
        if c.is_ascii_hexdigit() {
            cur.push(c);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// Parse the `Hashes: { "key": "hash", … }` block into key→hash pairs.
fn hashes_block(metadata: &str) -> Option<Vec<(String, String)>> {
    let pos = metadata.find("Hashes:")?;
    let from_open = metadata[pos..].find('{')? + pos;
    let close = metadata[from_open..].find('}')? + from_open;
    let block = &metadata[from_open..=close]; // includes the braces
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(block).ok()?;
    Some(
        map.into_iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
            .collect(),
    )
}

/// Like [`as_text`] but as a free function usable in `Option::and_then`.
fn as_text_opt(v: &serde_json::Value) -> Option<String> {
    as_text(v)
}

#[cfg(test)]
mod sized_url_tests {
    use super::sized_image_url;

    #[test]
    fn rewrites_transform_segment() {
        // original=true -> width=200 (the failing CyberRealistic case)
        assert_eq!(
            sized_image_url("https://image.civitai.com/abc/uuid/original=true/130909706.jpeg", 200),
            "https://image.civitai.com/abc/uuid/width=200/130909706.jpeg"
        );
        // existing width=450 -> width=200
        assert_eq!(
            sized_image_url("https://image.civitai.com/abc/uuid/width=450/x.jpeg", 200),
            "https://image.civitai.com/abc/uuid/width=200/x.jpeg"
        );
        // no transform segment -> insert one before the filename
        assert_eq!(
            sized_image_url("https://image.civitai.com/abc/uuid/x.jpeg", 200),
            "https://image.civitai.com/abc/uuid/width=200/x.jpeg"
        );
    }
}
