//! Deep Scan — finds problem files in a folder. A Rust port of terminus2's
//! `Scan.java`, opened from the top bar's "Find Issues" button.
//!
//! Two phases run on a background thread:
//!   1. **Corruption scan** — every image is decoded; ones that fail (empty,
//!      truncated, undecodable) are **listed for review** in the window (Delete,
//!      or — later — Fix, per file). They are *not* moved.
//!   2. **Exact-duplicate scan** (optional) — remaining images are SHA-256
//!      hashed; all but one of each identical group move to `duplicates/`.
//!
//! Videos/GIFs are listed but skipped (decode-validating a video frame-by-frame
//! is pointless here). No log file is written — everything is shown in the UI.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::theme::{ACCENT1, EDGE, FIELD, MUTED, PANEL, TEXT};

const CORRUPTED_FOLDER: &str = "corrupted_files";
const DUPLICATES_FOLDER: &str = "duplicates";

/// Shared height for the two console column headers so their boxes line up.
const HEADER_H: f32 = 22.0;

/// Image extensions the corruption scan will try to decode.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "ico", "hdr"];
/// Extended formats only decodable when built with `--features avif`.
const EXTENDED_EXTS: &[&str] = &["avif", "heic", "heif", "dng", "arw", "cr2", "nef"];
/// Media we list but never decode-validate.
const SKIP_EXTS: &[&str] = &["gif", "mp4", "webm", "avi", "mov", "mkv", "m4v", "wmv", "flv"];

/// Messages from the worker thread to the UI.
enum Msg {
    Log(String),
    Progress(f32, String),
    /// A file found corrupt: (path, reason). Listed for review (not moved).
    Corrupt(PathBuf, String),
    Done(Summary),
}

/// Final tallies, shown in the status line when a scan ends.
#[derive(Default, Clone)]
struct Summary {
    scanned: u32,
    corrupted: u32,
    duplicates_moved: u32,
    skipped: u32,
    cancelled: bool,
    error: Option<String>,
}

/// UI + runtime state for the Deep Scan window. Lives on `ViewerApp`.
pub struct ScanState {
    pub open: bool,
    input_dir: String,
    scan_duplicates: bool,

    log: Vec<String>,
    /// Corrupt files found this scan (path + reason), shown in the review panel.
    corrupt_files: Vec<(PathBuf, String)>,
    progress: f32,
    status: String,

    running: bool,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Msg>>,
    /// Set true once after a scan finishes so the app can refresh the browser.
    pub finished_tick: bool,
    /// Anchor for the window — the Find Issues button's bottom-right, captured
    /// when opened. The window is right-aligned to it (drops down, extends left).
    anchor_pos: Option<egui::Pos2>,
}

impl Default for ScanState {
    fn default() -> Self {
        Self {
            open: false,
            input_dir: String::new(),
            scan_duplicates: true,
            log: Vec::new(),
            corrupt_files: Vec::new(),
            progress: 0.0,
            status: "Ready.".to_string(),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            finished_tick: false,
            anchor_pos: None,
        }
    }
}

impl ScanState {
    /// Open the window under `anchor` (the Find Issues button's bottom-right),
    /// pre-filling the folder with the currently-open one.
    pub fn open_with(&mut self, folder: Option<&Path>, anchor: Option<egui::Pos2>) {
        self.open = true;
        self.anchor_pos = anchor;
        if self.input_dir.trim().is_empty()
            && let Some(f) = folder {
                self.input_dir = f.display().to_string();
            }
    }

    fn push_log(&mut self, line: impl Into<String>) {
        self.log.push(line.into());
        if self.log.len() > 2000 {
            let n = self.log.len() - 2000;
            self.log.drain(0..n);
        }
    }
}

/// Render the Deep Scan window when open. Drains worker messages and draws the UI.
pub fn show(ctx: &egui::Context, state: &mut ScanState) {
    // Drain background messages every frame — even while the window is minimised
    // (closed), so a backgrounded scan keeps progressing and finishes.
    if let Some(rx) = &state.rx {
        let drained: Vec<Msg> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        for m in drained {
            match m {
                Msg::Log(l) => state.push_log(l),
                Msg::Progress(p, s) => {
                    state.progress = p;
                    state.status = s;
                }
                Msg::Corrupt(path, reason) => {
                    // The corrupt entry shows in BOTH the console (text) and the
                    // review panel (actionable row).
                    state.push_log(format!("Corrupt: {}  |  {reason}", file_name(&path)));
                    state.corrupt_files.push((path, reason));
                }
                Msg::Done(sum) => {
                    state.running = false;
                    state.rx = None;
                    state.progress = 0.0;
                    state.finished_tick = true; // tell the app to refresh the browser
                    state.status = if sum.cancelled {
                        "Cancelled.".to_string()
                    } else if let Some(e) = &sum.error {
                        format!("Failed: {e}")
                    } else {
                        format!(
                            "Done — {} scanned, {} corrupt, {} duplicates moved, {} skipped.",
                            sum.scanned, sum.corrupted, sum.duplicates_moved, sum.skipped
                        )
                    };
                }
            }
        }
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    // Minimised: keep draining (above) but don't draw the window.
    if !state.open {
        return;
    }

    // A compact, fixed-size window. Shrink to fit only on tiny screens. The
    // inner content width is pinned to `content_w` so the window can't auto-grow
    // to fill a large display.
    let screen = ctx.content_rect();
    let win_w = 460.0_f32.min(screen.width() - 40.0);
    let win_h = 440.0_f32.min(screen.height() - 40.0);
    let content_w = win_w - 28.0; // minus the window's 14px inner margins
    let console_h = 110.0_f32;

    // Drop the window down from the "Find Issues" button, right-aligned to it:
    // the window's right edge sits under the button (its captured bottom-right),
    // so it extends to the left. Clamped to stay on-screen.
    let anchor = state.anchor_pos.unwrap_or_else(|| {
        egui::pos2(screen.right() - 10.0, screen.top() + 80.0)
    });
    let x = (anchor.x - win_w).min(screen.right() - win_w - 10.0).max(screen.left() + 10.0);
    let y = anchor.y.min(screen.bottom() - win_h - 10.0).max(screen.top() + 10.0);

    let mut want_minimize = false;
    use crate::PopupPlacement;
    egui::Window::new("Find Issues")
        .id(egui::Id::new("deep_scan_window"))
        .title_bar(false) // custom header inside (matches the Civitai / Backup popups)
        .collapsible(false)
        .resizable(false)
        .fixed_size([win_w, win_h])
        .placed_at([x, y])
        .frame(window_frame())
        .show(ctx, |ui| {
            // Only the top strip drags the popup — not stray drags on the body.
            crate::popup_drag_strip(ui, 30.0);
            ui.set_width(content_w);
            let radius = egui::CornerRadius::same(10);
            {
                let v = ui.visuals_mut();
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.active.corner_radius = radius;
                v.widgets.noninteractive.corner_radius = radius;
            }

            // Title row: frame-inspect icon + "Find Issues" + close (which just
            // hides the window — a running scan keeps going in the background).
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/frame_inspect.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(TEXT()),
                );
                ui.heading(egui::RichText::new("Find Issues").color(TEXT()).strong().size(17.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // click_and_drag so a click that slips a pixel is swallowed
                    // by the button instead of dragging the popup.
                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/close.svg"))
                                .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                .tint(TEXT()),
                        ).frame(false).sense(egui::Sense::click_and_drag()))
                        .on_hover_text("Close (a running scan keeps going)")
                        .clicked()
                    {
                        want_minimize = true;
                    }
                });
            });
            ui.add_space(10.0);

            ui.label(
                egui::RichText::new("Find corrupt images and exact duplicates.")
                    .color(MUTED())
                    .size(12.0),
            );
            ui.add_space(8.0);

            let enabled = !state.running;

            // Input folder + browse.
            ui.label(egui::RichText::new("Input folder").color(MUTED()).size(12.0));
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let folder_svg = egui::include_image!("../icons/folder.svg");
                if crate::svg_button(ui, folder_svg, "Choose folder to scan", 32.0, crate::theme::icon_tint(MUTED())).clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        state.input_dir = dir.display().to_string();
                    }
                ui.scope(|ui| {
                    ui.visuals_mut().extreme_bg_color = PANEL();
                    ui.add_enabled(
                        enabled,
                        egui::TextEdit::singleline(&mut state.input_dir)
                            .desired_width(f32::INFINITY)
                            .margin(egui::Margin::symmetric(10, 6))
                            .hint_text("Folder to scan"),
                    );
                });
            });
            ui.add_space(6.0);
            ui.add_enabled(
                enabled,
                egui::Checkbox::new(&mut state.scan_duplicates, "Also scan for exact duplicates (SHA-256)"),
            );

            ui.add_space(8.0);
            // Status line (Gelbooru-downloader style — no progress bar): the
            // frame-inspect icon + percentage while scanning, a green checkmark
            // when done, plus the current status text.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let icon = egui::vec2(16.0, 16.0);
                if state.running {
                    ui.add(
                        egui::Image::new(egui::include_image!("../icons/frame_inspect.svg"))
                            .fit_to_exact_size(icon)
                            .tint(ACCENT1()),
                    );
                    let pct = (state.progress.clamp(0.0, 1.0) * 100.0).round() as u32;
                    ui.label(egui::RichText::new(format!("{pct}%")).color(ACCENT1()).strong().size(12.0));
                } else if state.status.starts_with("Failed") {
                    // Error → red warning.
                    ui.add(
                        egui::Image::new(egui::include_image!("../icons/warning.svg"))
                            .fit_to_exact_size(icon)
                            .tint(egui::Color32::from_rgb(220, 70, 70)),
                    );
                } else if !state.corrupt_files.is_empty() {
                    // Corrupt images found → orange warning.
                    ui.add(
                        egui::Image::new(egui::include_image!("../icons/warning.svg"))
                            .fit_to_exact_size(icon)
                            .tint(egui::Color32::from_rgb(235, 150, 45)),
                    );
                } else if state.status.starts_with("Done") {
                    // Clean → green checkmark.
                    ui.add(
                        egui::Image::new(egui::include_image!("../icons/checkmark.svg"))
                            .fit_to_exact_size(icon)
                            .tint(egui::Color32::from_rgb(46, 160, 67)),
                    );
                }
                ui.label(egui::RichText::new(&state.status).color(MUTED()).size(12.0));
            });
            ui.add_space(8.0);

            // Two columns: console (left, includes corrupt lines) and the corrupt
            // review panel (right). Fixed, bounded height.
            let mut to_open: Option<PathBuf> = None;
            let mut to_delete: Option<usize> = None;
            ui.columns(2, |cols| {
                console_box(&mut cols[0], "Console", &state.log, console_h, "scan_console");
                corrupt_review(&mut cols[1], &state.corrupt_files, console_h, &mut to_open, &mut to_delete);
            });
            if let Some(p) = to_open {
                open_in_default(&p);
            }
            if let Some(i) = to_delete
                && i < state.corrupt_files.len() {
                    let (p, _) = state.corrupt_files.remove(i);
                    match std::fs::remove_file(&p) {
                        Ok(_) => state.push_log(format!("Deleted: {}", file_name(&p))),
                        Err(e) => state.push_log(format!("ERROR deleting {}: {e}", file_name(&p))),
                    }
                }

            ui.add_space(8.0);

            // Right-aligned footer buttons (matches the Create Backup popup):
            // primary Start Scan / Cancel rightmost, Minimize to its left.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                if state.running {
                    if footer_button(ui, "Cancel", Some(egui::Color32::from_rgb(180, 40, 40))).clicked() {
                        state.cancel.store(true, Ordering::SeqCst);
                        state.status = "Cancelling…".to_string();
                    }
                } else if footer_button(ui, "Start Scan", Some(ACCENT1())).clicked() {
                    start_scan(state, ctx);
                }
                // Minimize: hide the window but keep any running scan going in the
                // background (messages are still drained at the top of `show`).
                if footer_button(ui, "Minimize", None).clicked() {
                    want_minimize = true;
                }
            });
        });
    // The ✕ and the Minimize button both just hide it (scan keeps running).
    state.open = !want_minimize;
}

/// A titled, scrollable, dark console well.
fn console_box(ui: &mut egui::Ui, title: &str, lines: &[String], height: f32, salt: &str) {
    // Title row with a Copy button, so users can grab the log on an error. Fixed
    // height (HEADER_H) so it lines up with the sibling column's header.
    ui.horizontal(|ui| {
        ui.set_min_height(HEADER_H);
        ui.label(egui::RichText::new(title).color(MUTED()).strong().size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Copy icon; flips to the green "copied" check for a moment after
            // a click, as feedback that the log is on the clipboard. Sized to
            // stay within HEADER_H so the sibling column header lines up.
            const FLASH_SECS: f64 = 1.2;
            let flash_id = egui::Id::new("console_copy_flash").with(salt);
            let now = ui.input(|i| i.time);
            let flashing = ui
                .data(|d| d.get_temp::<f64>(flash_id))
                .is_some_and(|t| now - t < FLASH_SECS);
            let icon = |src: egui::ImageSource<'static>, tint: egui::Color32| {
                egui::Button::image(
                    egui::Image::new(src)
                        .fit_to_exact_size(egui::vec2(15.0, 15.0))
                        .tint(tint),
                )
                .frame(false)
            };
            if flashing {
                // WHITE tint = keep the icon's own green.
                ui.add(icon(egui::include_image!("../icons/copied.svg"), egui::Color32::WHITE))
                    .on_hover_text("Copied");
                // Wake up to flip back to the copy icon when the flash ends.
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            } else {
                let copy = icon(
                    egui::include_image!("../icons/copy.svg"),
                    crate::theme::icon_tint(MUTED()),
                );
                if ui.add_enabled(!lines.is_empty(), copy).on_hover_text("Copy the log").clicked() {
                    ui.ctx().copy_text(lines.join("\n"));
                    ui.data_mut(|d| d.insert_temp(flash_id, now));
                }
            }
        });
    });
    ui.add_space(2.0);
    let bg = if crate::theme::is_light() { FIELD() } else { egui::Color32::from_rgb(15, 15, 17) };
    egui::Frame::new()
        .fill(bg)
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::same(10))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_height(height);
            egui::ScrollArea::vertical()
                .id_salt(salt)
                .auto_shrink([false, false])
                .stick_to_bottom(salt == "scan_console")
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    // Keep scrolling while a text selection is dragged past the edge.
                    crate::drag_select_autoscroll(ui);
                    if lines.is_empty() {
                        ui.label(egui::RichText::new("—").color(MUTED()).monospace().size(10.0));
                    } else {
                        for l in lines {
                            ui.label(egui::RichText::new(l).color(TEXT()).monospace().size(10.0));
                        }
                    }
                });
        });
}

/// The corrupt-image review panel: a scrollable list of corrupt files, each with
/// a warning icon + filename (click to open in the default viewer), a Delete
/// button, and a disabled Fix placeholder. Sets `to_open` / `to_delete` for the
/// caller to apply (we can't mutate the list while it's borrowed for rendering).
fn corrupt_review(
    ui: &mut egui::Ui,
    items: &[(PathBuf, String)],
    height: f32,
    to_open: &mut Option<PathBuf>,
    to_delete: &mut Option<usize>,
) {
    // Fixed-height header so it lines up with the Console column's header.
    ui.horizontal(|ui| {
        ui.set_min_height(HEADER_H);
        ui.label(
            egui::RichText::new(format!("Corrupt images ({})", items.len()))
                .color(MUTED())
                .strong()
                .size(11.0),
        );
    });
    ui.add_space(2.0);
    let bg = if crate::theme::is_light() { FIELD() } else { egui::Color32::from_rgb(15, 15, 17) };
    egui::Frame::new()
        .fill(bg)
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::same(10))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_height(height);
            egui::ScrollArea::vertical()
                .id_salt("scan_corrupt_review")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    if items.is_empty() {
                        ui.label(egui::RichText::new("—").color(MUTED()).monospace().size(12.0));
                        return;
                    }
                    for (i, (path, reason)) in items.iter().enumerate() {
                        // Line 1: warning icon + filename (click to open / review).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            ui.add(
                                egui::Image::new(egui::include_image!("../icons/warning.svg"))
                                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                                    .tint(egui::Color32::from_rgb(220, 160, 60)),
                            );
                            let label = egui::Label::new(
                                egui::RichText::new(file_name(path)).color(TEXT()).size(11.5),
                            )
                            .truncate()
                            .sense(egui::Sense::click());
                            if ui.add(label).on_hover_text(format!("{reason}\n(click to open)")).clicked() {
                                *to_open = Some(path.clone());
                            }
                        });
                        // Line 2: Delete (active) + Fix (placeholder — disabled).
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            let del = egui::Button::new(
                                egui::RichText::new("Delete")
                                    .color(egui::Color32::from_rgb(230, 90, 90))
                                    .size(11.0),
                            )
                            .corner_radius(egui::CornerRadius::same(8));
                            if ui.add(del).clicked() {
                                *to_delete = Some(i);
                            }
                            ui.add_enabled(
                                false,
                                egui::Button::new(egui::RichText::new("Fix").size(11.0))
                                    .corner_radius(egui::CornerRadius::same(8)),
                            )
                            .on_disabled_hover_text("Coming soon");
                        });
                        ui.add_space(6.0);
                    }
                });
        });
}

/// Open `path` in the OS default application (to review a flagged file).
fn open_in_default(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // `start` uses the file's default association; CREATE_NO_WINDOW avoids a
        // console flash.
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", &path.to_string_lossy()])
            .creation_flags(0x0800_0000)
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

/// A right-aligned footer button matching the Create Backup popup (corner-10,
/// fixed size). `fill` Some → filled with white text; None → subtle TEXT label.
fn footer_button(ui: &mut egui::Ui, label: &str, fill: Option<egui::Color32>) -> egui::Response {
    let text = if fill.is_some() { egui::Color32::WHITE } else { TEXT() };
    let mut btn = egui::Button::new(egui::RichText::new(label).color(text).strong().size(14.0))
        .corner_radius(egui::CornerRadius::same(10));
    if let Some(c) = fill {
        btn = btn.fill(c);
    }
    ui.add_sized(egui::vec2(96.0, 32.0), btn)
}

fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(egui::CornerRadius::same(16))
        .inner_margin(egui::Margin::same(14))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 6],
            blur: 20,
            spread: 0,
            color: egui::Color32::from_black_alpha(150),
        })
}

/// Validate the folder and spawn the worker.
fn start_scan(state: &mut ScanState, ctx: &egui::Context) {
    if state.running {
        return;
    }
    let dir = PathBuf::from(state.input_dir.trim());
    if state.input_dir.trim().is_empty() || !dir.is_dir() {
        state.log.clear();
        state.push_log("ERROR: Please choose a valid folder to scan.");
        state.status = "Ready.".to_string();
        return;
    }

    state.log.clear();
    state.corrupt_files.clear();
    state.progress = 0.0;
    state.status = "Scanning…".to_string();

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    state.cancel = Arc::clone(&cancel);
    state.rx = Some(rx);
    state.running = true;

    let scan_dupes = state.scan_duplicates;
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        run_scan(dir, scan_dupes, tx, cancel, ctx);
    });
}

fn run_scan(dir: PathBuf, scan_dupes: bool, tx: Sender<Msg>, cancel: Arc<AtomicBool>, ctx: egui::Context) {
    let log = |s: String| {
        let _ = tx.send(Msg::Log(s));
        ctx.request_repaint();
    };
    let prog = |p: f32, s: String| {
        let _ = tx.send(Msg::Progress(p, s));
        ctx.request_repaint();
    };

    let mut sum = Summary::default();

    log("Gathering files…".into());
    let all = gather_files(&dir);

    let mut images: Vec<PathBuf> = Vec::new();
    for p in &all {
        let ext = ext_of(p);
        if IMAGE_EXTS.contains(&ext.as_str())
            || (cfg!(feature = "avif") && EXTENDED_EXTS.contains(&ext.as_str()))
        {
            images.push(p.clone());
        } else if SKIP_EXTS.contains(&ext.as_str()) {
            sum.skipped += 1;
        }
    }
    log(format!(
        "Found {} image(s) to scan and {} media file(s) skipped.",
        images.len(),
        sum.skipped
    ));

    // --- Phase 1: corruption ---
    log("--- Phase 1: corrupted-file scan ---".into());
    let mut valid: Vec<PathBuf> = Vec::new();
    let mut corrupt_count = 0u32;

    let total = images.len().max(1);
    for (i, p) in images.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            finish(&tx, &ctx, sum, true);
            return;
        }
        prog(i as f32 / total as f32, format!("Scanning {} / {}", i + 1, images.len()));

        match validate_image(p) {
            // Corrupt files are NOT moved — they're listed for review (the UI logs
            // each one and offers Delete / Fix per file).
            Ok(()) => valid.push(p.clone()),
            Err(reason) => {
                let _ = tx.send(Msg::Corrupt(p.clone(), reason));
                corrupt_count += 1;
            }
        }
    }
    sum.scanned = images.len() as u32;
    sum.corrupted = corrupt_count;
    if corrupt_count == 0 {
        log("No corrupted images found.".into());
    } else {
        log(format!("{corrupt_count} corrupt image(s) found — review them in the panel."));
    }

    // --- Phase 2: exact duplicates ---
    let mut dupes_moved = 0u32;
    if scan_dupes && !valid.is_empty() {
        if cancel.load(Ordering::SeqCst) {
            finish(&tx, &ctx, sum, true);
            return;
        }
        log("--- Phase 2: exact-duplicate scan (SHA-256) ---".into());
        let dup_dir = dir.join(DUPLICATES_FOLDER);

        use std::collections::HashMap;
        let mut by_hash: HashMap<String, Vec<PathBuf>> = HashMap::new();
        let vtotal = valid.len().max(1);
        for (i, p) in valid.iter().enumerate() {
            if cancel.load(Ordering::SeqCst) {
                finish(&tx, &ctx, sum, true);
                return;
            }
            prog(i as f32 / vtotal as f32, format!("Hashing {} / {}", i + 1, valid.len()));
            if let Some(h) = sha256_file(p) {
                by_hash.entry(h).or_default().push(p.clone());
            }
        }

        let groups: Vec<Vec<PathBuf>> = by_hash.into_values().filter(|g| g.len() > 1).collect();
        if groups.is_empty() {
            log("No exact duplicates found.".into());
        } else {
            let to_move: usize = groups.iter().map(|g| g.len() - 1).sum();
            log(format!("Moving {to_move} duplicate file(s) to '{DUPLICATES_FOLDER}'…"));
            let _ = std::fs::create_dir_all(&dup_dir);
            for (gi, mut group) in groups.into_iter().enumerate() {
                // Keep the alphabetically-first file; move the rest.
                group.sort();
                for dup in group.into_iter().skip(1) {
                    if move_into(&dup, &dup_dir, &format!("dup_{}_", gi + 1), &log) {
                        dupes_moved += 1;
                    }
                }
            }
        }
    }
    sum.duplicates_moved = dupes_moved;

    log("Scan complete.".into());
    finish(&tx, &ctx, sum, false);
}

fn finish(tx: &Sender<Msg>, ctx: &egui::Context, mut sum: Summary, cancelled: bool) {
    sum.cancelled = cancelled;
    let _ = tx.send(Msg::Done(sum));
    ctx.request_repaint();
}

/// Decode-validate one image. `Ok(())` = readable; `Err(reason)` = corrupt.
fn validate_image(p: &Path) -> Result<(), String> {
    match std::fs::metadata(p) {
        Ok(m) if m.len() == 0 => return Err("Empty file".to_string()),
        Err(e) => return Err(format!("Unreadable: {e}")),
        _ => {}
    }

    let ext = ext_of(p);

    // HDR goes through our tone-mapping decoder — `image`'s reader rejects the
    // valid `#?RGBE` signature variant, which would wrongly flag good files.
    if ext == "hdr" {
        return match crate::image_cache::decode_hdr(p) {
            Some(_) => Ok(()),
            None => Err("Could not decode".to_string()),
        };
    }

    // Extended formats go through our own decoder.
    #[cfg(feature = "avif")]
    if EXTENDED_EXTS.contains(&ext.as_str()) {
        return match crate::avif::decode_avif(p) {
            Some(_) => Ok(()),
            None => Err("Could not decode".to_string()),
        };
    }

    // Everything else: the `image` crate reads + decodes it. Checking dimensions
    // forces a header parse; `decode()` forces the full pixel decode so truncated
    // files are caught, mirroring the Java reader's r.read(0).
    match image::ImageReader::open(p) {
        Ok(reader) => match reader.with_guessed_format() {
            Ok(reader) => match reader.decode() {
                Ok(img) => {
                    if img.width() == 0 || img.height() == 0 {
                        Err("Invalid dimensions".to_string())
                    } else {
                        Ok(())
                    }
                }
                Err(e) => Err(short_err(&e.to_string())),
            },
            Err(e) => Err(short_err(&e.to_string())),
        },
        Err(e) => Err(format!("Open failed: {e}")),
    }
}

/// Move `file` into `dest` with a name prefix, de-duplicating the target name.
/// Returns true on success.
fn move_into(file: &Path, dest: &Path, prefix: &str, log: &impl Fn(String)) -> bool {
    let base = file_name(file);
    let target = unique_target(dest, &format!("{prefix}{base}"));
    if let Err(e) = std::fs::rename(file, &target) {
        // rename fails across volumes; fall back to copy+delete.
        match std::fs::copy(file, &target).and_then(|_| std::fs::remove_file(file)) {
            Ok(_) => true,
            Err(_) => {
                log(format!("ERROR: could not move {base}: {e}"));
                false
            }
        }
    } else {
        true
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Recursively collect regular files, skipping our own quarantine folders.
fn gather_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.eq_ignore_ascii_case(CORRUPTED_FOLDER) || name.eq_ignore_ascii_case(DUPLICATES_FOLDER) {
                    continue;
                }
                stack.push(p);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

fn ext_of(p: &Path) -> String {
    p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase()
}

fn file_name(p: &Path) -> String {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("<unknown>").to_string()
}

/// Trim a decoder error to its first line so the list stays readable.
fn short_err(s: &str) -> String {
    let first = s.lines().next().unwrap_or(s).trim();
    if first.len() > 80 {
        format!("{}…", &first[..80])
    } else {
        first.to_string()
    }
}

/// A non-clashing path in `dest` for `name`, appending " (n)" if needed.
fn unique_target(dest: &Path, name: &str) -> PathBuf {
    let base = dest.join(name);
    if !base.exists() {
        return base;
    }
    let (stem, ext) = match name.rfind('.') {
        Some(i) => (&name[..i], &name[i..]),
        None => (name, ""),
    };
    for i in 1..100_000 {
        let cand = dest.join(format!("{stem} ({i}){ext}"));
        if !cand.exists() {
            return cand;
        }
    }
    dest.join(name)
}

// ---------------------------------------------------------------------------
// SHA-256 (self-contained — no external crate)
// ---------------------------------------------------------------------------

/// SHA-256 of a file as lowercase hex, streamed in 64 KiB chunks. `None` on I/O
/// error. (Also used by the Generate panels to look LoRAs up on Civitai.)
pub(crate) fn sha256_file(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Some(h.finish_hex())
}

/// A *fast* content fingerprint: SHA-256 over the file size plus its first and
/// last 4 KiB. It reads at most 8 KiB regardless of file size, so it's cheap even
/// for huge media, yet stable across moves/renames — which is what lets favorites
/// be tracked by content rather than by path (see `crate::favorites`, ported from
/// terminus2's `HeartManager`). `None` on I/O error.
pub(crate) fn fast_content_hash(path: &Path) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let size = f.metadata().ok()?.len();
    let mut h = Sha256::new();
    h.update(&size.to_le_bytes());

    let mut buf = [0u8; 4096];
    let n = f.read(&mut buf).ok()?;
    h.update(&buf[..n]);

    // Mix in the tail too, so files sharing a header (e.g. the same camera/JPEG
    // preamble) don't collide. Only when the file is larger than the head read.
    if size > 4096 {
        f.seek(SeekFrom::Start(size - 4096)).ok()?;
        let n = f.read(&mut buf).ok()?;
        h.update(&buf[..n]);
    }
    Some(h.finish_hex())
}

/// A small, dependency-free SHA-256 implementation (FIPS 180-4).
struct Sha256 {
    state: [u32; 8],
    len_bits: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            len_bits: 0,
            buf: [0u8; 64],
            buf_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.len_bits = self.len_bits.wrapping_add((data.len() as u64) * 8);
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finish_hex(mut self) -> String {
        // Pad: 0x80, zeros, then 64-bit big-endian length.
        let bits = self.len_bits;
        self.update_no_len(&[0x80]);
        while self.buf_len != 56 {
            self.update_no_len(&[0x00]);
        }
        self.update_no_len(&bits.to_be_bytes());

        let mut out = String::with_capacity(64);
        for word in self.state.iter() {
            out.push_str(&format!("{word:08x}"));
        }
        out
    }

    /// `update` but without counting toward the message length (used for padding).
    fn update_no_len(&mut self, data: &[u8]) {
        for &b in data {
            self.buf[self.buf_len] = b;
            self.buf_len += 1;
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
    }

    // Index loops mirror the FIPS 180-4 round notation.
    #[allow(clippy::needless_range_loop)]
    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[i * 4], block[i * 4 + 1], block[i * 4 + 2], block[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut h = self.state;
        for i in 0..64 {
            let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
            let ch = (h[4] & h[5]) ^ ((!h[4]) & h[6]);
            let t1 = h[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(Self::K[i])
                .wrapping_add(w[i]);
            let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let t2 = s0.wrapping_add(maj);
            h[7] = h[6];
            h[6] = h[5];
            h[5] = h[4];
            h[4] = h[3].wrapping_add(t1);
            h[3] = h[2];
            h[2] = h[1];
            h[1] = h[0];
            h[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            self.state[i] = self.state[i].wrapping_add(h[i]);
        }
    }
}

#[cfg(test)]
mod fast_hash_tests {
    use super::fast_content_hash;
    use std::io::Write;

    fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("clarity_fav_test_{name}"));
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(bytes).unwrap();
        p
    }

    #[test]
    fn stable_and_distinct() {
        let small = tmp("small", b"hello world");
        let big_a = tmp("big_a", &vec![7u8; 100_000]);
        let mut tail = vec![7u8; 100_000];
        *tail.last_mut().unwrap() = 9; // differs only in the last byte
        let big_b = tmp("big_b", &tail);

        // Deterministic.
        assert_eq!(fast_content_hash(&small), fast_content_hash(&small));
        assert_eq!(fast_content_hash(&big_a), fast_content_hash(&big_a));
        // The tail is mixed in, so a last-byte change is detected.
        assert_ne!(fast_content_hash(&big_a), fast_content_hash(&big_b));
        // Missing file -> None.
        assert!(fast_content_hash(std::path::Path::new("does_not_exist_xyz")).is_none());

        for p in [small, big_a, big_b] {
            let _ = std::fs::remove_file(p);
        }
    }
}
