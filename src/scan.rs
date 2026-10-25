//! Deep Scan — finds problem files in a folder. A Rust port of terminus2's
//! `Scan.java`, opened from the top bar's "Find Issues" button.
//!
//! Two phases run on a background thread:
//!   1. **Corruption scan** — every image is decoded; ones that fail (empty,
//!      truncated, undecodable) are **listed for review** in the window, each
//!      to open or delete. They are *not* moved.
//!   2. **Exact-duplicate scan** (optional) — remaining images are SHA-256
//!      hashed; all but one of each identical group move to `duplicates/`.
//!
//! Videos/GIFs are counted but skipped (decode-validating a video frame-by-frame
//! is pointless here). No log file is written — everything is shown in the UI.
//!
//! The window: a SCAN card (the folder, the duplicates switch), then — once a
//! scan runs — a results card with three counts (verified, duplicates,
//! corrupt) over one list of every file that needs a look: corrupt ones,
//! moved duplicates, and images it couldn't check. Copy list and Start Scan
//! sit at the bottom.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::backup::{FLASH, FLASH_GREEN, FLASH_RED};

use eframe::egui;
use egui::{Color32, RichText};

use crate::theme::{FLOAT, MUTED, TEXT};

const CORRUPTED_FOLDER: &str = "corrupted_files";
const DUPLICATES_FOLDER: &str = "duplicates";

/// The window's width and inner margin. Its height follows what it shows.
const WIN_W: f32 = 524.0;
const MARGIN: i8 = 28;
/// Corner radius of the cards; the list well inside the results card.
const CARD_R: u8 = 18;
const WELL_R: u8 = 12;
/// One row of the result list. The list shows this many rows before it
/// scrolls — half of the next one peeks out, so it's plain there are more.
const ROW_H: f32 = 36.0;
const LIST_ROWS: f32 = 6.5;
/// Share of the progress bar for checking the images when duplicates are
/// looked for too (hashing is much quicker than decoding).
const CHECK_SHARE: f32 = 0.85;

/// Image extensions the corruption scan will try to decode.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "ico", "hdr", "svg"];
/// Extended formats only decodable when built with `--features avif`.
const EXTENDED_EXTS: &[&str] = &["avif", "heic", "heif", "dng", "arw", "cr2", "nef"];
/// Media we count but never decode-validate.
const SKIP_EXTS: &[&str] = &["gif", "mp4", "webm", "avi", "mov", "mkv", "m4v", "wmv", "flv"];

/// The step the worker is on, for the progress line.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Stage {
    /// Walking the folder: how many images there are isn't known yet.
    #[default]
    Gathering,
    /// Decoding every image (phase 1).
    Checking,
    /// Hashing the readable images for exact duplicates (phase 2).
    Hashing,
}

/// Messages from the worker thread to the UI.
enum Msg {
    /// Working on file `at` (counting from 1) of `total` in `stage`.
    Progress { stage: Stage, at: usize, total: usize },
    /// A file found corrupt: (path, reason). Listed for review (not moved).
    Corrupt(PathBuf, String),
    /// An image that couldn't be decoded here, though it may well be fine:
    /// (path, reason).
    Unchecked(PathBuf, String),
    /// An exact copy of `kept`, and what happened when it was moved.
    Duplicate { path: PathBuf, kept: PathBuf, moved: Moved },
    Done(Summary),
}

/// What happened to a duplicate on its way to the duplicates folder.
#[derive(Clone, Debug, PartialEq)]
enum Moved {
    /// It's there now, with its tags if it had any.
    To(PathBuf),
    /// It's there now, but its tags (`.txt`) couldn't follow: why.
    WithoutTags(PathBuf, String),
    /// It couldn't be moved: why.
    Failed(String),
}

/// Final tallies of a scan.
#[derive(Default, Clone, Debug)]
struct Summary {
    scanned: u32,
    corrupted: u32,
    /// Images that couldn't be decoded here though they may well be fine (a
    /// format variant the decoder doesn't support, or too big).
    unchecked: u32,
    /// Extra copies found, and how many of them were moved.
    duplicates_found: u32,
    duplicates_moved: u32,
    skipped: u32,
    /// GIFs among the skipped (the rest are videos).
    skipped_gifs: u32,
    /// The duplicate pass ran (it was on, and some images were readable).
    dupes_checked: bool,
    cancelled: bool,
    error: Option<String>,
}

/// How the last scan ended.
#[derive(Debug)]
enum Outcome {
    Finished(Summary),
    Cancelled,
    /// It stopped part-way: what to tell the user.
    Failed(String),
}

/// One row of the result list.
struct Finding {
    path: PathBuf,
    what: What,
}

#[derive(Debug, PartialEq)]
enum What {
    /// Broken: listed to open or delete.
    Corrupt { reason: String, status: FindingStatus },
    /// Couldn't be decoded here; it may well be fine.
    Unchecked { reason: String },
    /// An extra copy of `kept`, moved (or not) to the duplicates folder.
    Duplicate { kept: PathBuf, moved: Moved },
}

impl What {
    /// Rows are grouped: corrupt files first, then the unchecked, then duplicates.
    fn rank(&self) -> u8 {
        match self {
            What::Corrupt { .. } => 0,
            What::Unchecked { .. } => 1,
            What::Duplicate { .. } => 2,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
enum FindingStatus {
    Listed,
    Deleted,
    /// Deleted, but its tags (`.txt`) couldn't be: why.
    DeletedTagsStayed(String),
    /// Deleting it failed: why.
    DeleteFailed(String),
}

impl FindingStatus {
    fn is_deleted(&self) -> bool {
        matches!(self, FindingStatus::Deleted | FindingStatus::DeletedTagsStayed(_))
    }
}

/// UI + runtime state for the Deep Scan window. Lives on `ViewerApp`.
pub struct ScanState {
    pub open: bool,
    input_dir: String,
    scan_duplicates: bool,

    /// Files the current / last scan found, in list order.
    findings: Vec<Finding>,
    /// Where the running scan is: its step, and file `at` of `total` in it.
    stage: Stage,
    at: usize,
    total: usize,
    /// The folder the current / last scan ran on. The list shows names
    /// relative to it; its `duplicates` folder is where duplicates go.
    root: PathBuf,
    /// How the last scan ended (`None` before the first, and while one runs).
    outcome: Option<Outcome>,
    /// Why Start didn't start, shown under the folder field.
    folder_error: Option<String>,
    /// Counts scans, so each one's progress bar and list start afresh.
    runs: u32,
    /// Height the list took last frame: the rest of the window is last
    /// frame's height minus this.
    list_h_drawn: f32,
    /// When Copy list was last clicked (it says "Copied" for a moment).
    copied_at: Option<Instant>,

    running: bool,
    /// Cancel was pressed: the worker stops at its next file.
    cancelling: bool,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Msg>>,
    /// Set true once after a scan finishes so the app can refresh the browser.
    pub finished_tick: bool,
    /// Files deleted from the list, for the app to take off the browser's
    /// list (taken by `main.rs` each frame).
    pub deleted: Vec<PathBuf>,
    /// Anchor for the window — the Find Issues button's bottom-right, captured
    /// when opened. The window is right-aligned to it (drops down, extends left).
    anchor_pos: Option<egui::Pos2>,
    /// True on the frame the window was opened, so the Find Issues button click
    /// that spawned it isn't counted as a click outside (instant hide).
    just_opened: bool,
    /// Button click-flash (Create Backup style): the button fills green/red
    /// for `FLASH`, then the action fires.
    start_flash: Option<Instant>,
    cancel_flash: Option<Instant>,
}

impl Default for ScanState {
    fn default() -> Self {
        Self {
            open: false,
            input_dir: String::new(),
            scan_duplicates: true,
            findings: Vec::new(),
            stage: Stage::Gathering,
            at: 0,
            total: 0,
            root: PathBuf::new(),
            outcome: None,
            folder_error: None,
            runs: 0,
            list_h_drawn: 0.0,
            copied_at: None,
            running: false,
            cancelling: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            finished_tick: false,
            deleted: Vec::new(),
            anchor_pos: None,
            just_opened: false,
            start_flash: None,
            cancel_flash: None,
        }
    }
}

impl ScanState {
    /// Open the window under `anchor` (the Find Issues button's bottom-right),
    /// pre-filling the folder with the currently-open one.
    pub fn open_with(&mut self, folder: Option<&Path>, anchor: Option<egui::Pos2>) {
        self.open = true;
        self.just_opened = true;
        self.anchor_pos = anchor;
        if self.input_dir.trim().is_empty()
            && let Some(f) = folder {
                self.input_dir = f.display().to_string();
            }
    }

    /// Add a row, keeping the groups together (corrupt, unchecked, duplicates).
    fn push_finding(&mut self, path: PathBuf, what: What) {
        let at = self.findings.iter().position(|f| f.what.rank() > what.rank()).unwrap_or(self.findings.len());
        self.findings.insert(at, Finding { path, what });
    }

    /// Take the worker's messages. Runs every frame, even while the window is
    /// hidden, so a scan left running in the background keeps going.
    fn drain(&mut self, ctx: &egui::Context) {
        let Some(rx) = &self.rx else { return };
        // A worker that died (a panic) hangs up without a Done; `try_recv().ok()`
        // read that as "no message yet", leaving the scan "running" forever.
        let mut died = false;
        let drained: Vec<Msg> = std::iter::from_fn(|| match rx.try_recv() {
            Ok(m) => Some(m),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                died = true;
                None
            }
        })
        .collect();
        for m in drained {
            match m {
                Msg::Progress { stage, at, total } => {
                    self.stage = stage;
                    self.at = at;
                    self.total = total;
                }
                Msg::Corrupt(path, reason) => {
                    self.push_finding(path, What::Corrupt { reason, status: FindingStatus::Listed });
                }
                Msg::Unchecked(path, reason) => self.push_finding(path, What::Unchecked { reason }),
                Msg::Duplicate { path, kept, moved } => {
                    // It may be listed as not checked: it's moved now.
                    self.findings.retain(|f| !(matches!(f.what, What::Unchecked { .. }) && f.path == path));
                    self.push_finding(path, What::Duplicate { kept, moved });
                }
                Msg::Done(sum) => {
                    self.end_run();
                    self.outcome = Some(if sum.cancelled {
                        Outcome::Cancelled
                    } else if let Some(e) = &sum.error {
                        Outcome::Failed(e.clone())
                    } else {
                        Outcome::Finished(sum)
                    });
                }
            }
        }
        if died && self.running {
            self.end_run();
            self.outcome = Some(Outcome::Failed(
                "The scan stopped unexpectedly. The list shows what it found before then.".to_string(),
            ));
        }
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    /// The worker is finished (or gone).
    fn end_run(&mut self) {
        self.running = false;
        self.cancelling = false;
        self.cancel_flash = None;
        self.rx = None;
        self.finished_tick = true; // tell the app to refresh the browser
    }

    /// The three counts over the list. `None` = not known (yet).
    fn counts(&self) -> [Option<u64>; 3] {
        let count = |rank| self.findings.iter().filter(|f| f.what.rank() == rank).count() as u64;
        let (corrupt, unchecked, dupes) = (count(0), count(1), count(2));
        let verified = match &self.outcome {
            Some(Outcome::Finished(sum)) => Some(u64::from(sum.scanned.saturating_sub(sum.corrupted + sum.unchecked))),
            // Running, or stopped part-way: the images checked so far minus
            // those that failed. While running, the one at `at` is still
            // being checked; a cancelled scan stops between files, so by
            // then it was.
            _ => match self.stage {
                Stage::Gathering => None,
                Stage::Checking => {
                    let cancelled = matches!(self.outcome, Some(Outcome::Cancelled));
                    let done = if cancelled { self.at } else { self.at.saturating_sub(1) };
                    Some((done as u64).saturating_sub(corrupt + unchecked))
                }
                // Everything readable is being hashed.
                Stage::Hashing => Some((self.total as u64).saturating_sub(unchecked)),
            },
        };
        let duplicates = match &self.outcome {
            Some(Outcome::Finished(sum)) if sum.dupes_checked => Some(u64::from(sum.duplicates_found)),
            _ if dupes > 0 => Some(dupes),
            _ => None,
        };
        let known = (self.running || self.outcome.is_some()) && self.stage != Stage::Gathering;
        let corrupt = (known || matches!(self.outcome, Some(Outcome::Finished(_)))).then_some(corrupt);
        [verified, duplicates, corrupt]
    }
}

/// What the user asked for this frame, applied once the window is drawn.
#[derive(Default)]
struct Actions {
    open: Option<PathBuf>,
    delete: Option<usize>,
    /// Height of the list drawn this frame.
    list_h: f32,
}

/// Render the Deep Scan window when open. Drains worker messages and draws the UI.
pub fn show(ctx: &egui::Context, state: &mut ScanState) {
    state.drain(ctx);
    fire_flash(state, ctx);
    // Minimised: keep draining (above) but don't draw the window.
    if !state.open {
        return;
    }

    let screen = ctx.content_rect();
    let win_w = WIN_W.min(screen.width() - 40.0);
    let content_w = win_w - 2.0 * f32::from(MARGIN);
    let id = egui::Id::new("deep_scan_window");
    // The window hugs its content, so it grows as a scan goes on: last
    // frame's height keeps it on-screen.
    let last_h = ctx.memory(|m| m.area_rect(id)).map_or(300.0, |r| r.height());
    // Drop the window down from the "Find Issues" button, right-aligned to it:
    // the window's right edge sits under the button (its captured bottom-right),
    // so it extends to the left. Clamped to stay on-screen.
    let anchor = state.anchor_pos.unwrap_or_else(|| {
        egui::pos2(screen.right() - 10.0, screen.top() + 80.0)
    });
    let x = (anchor.x - win_w).min(screen.right() - win_w - 10.0).max(screen.left() + 10.0);
    let y = anchor.y.min(screen.bottom() - last_h - 10.0).max(screen.top() + 10.0);
    // The list's height: its full size on a normal screen, what's left beside
    // the rest of the window on a short one.
    let top = ctx.memory(|m| m.area_rect(id)).map_or(y, |r| r.top());
    let room = screen.bottom() - 10.0 - top - (last_h - state.list_h_drawn);
    let list_h = (LIST_ROWS * ROW_H).min(room).max(2.5 * ROW_H);

    let mut act = Actions::default();
    use crate::PopupPlacement;
    let win = egui::Window::new("Find Issues")
        .id(id)
        .title_bar(false) // custom header inside
        .collapsible(false)
        .resizable(false)
        .placed_at([x, y])
        .frame(window_frame())
        .show(ctx, |ui| {
            // Only the top strip drags the popup — not stray drags on the body.
            crate::popup_drag_strip(ui, 30.0);
            ui.set_width(content_w);
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 0.0);
            header(ui);
            ui.add_space(18.0);
            section_label(ui, "SCAN");
            ui.add_space(8.0);
            let submitted = scan_card(ui, state);
            if state.running || state.outcome.is_some() {
                ui.add_space(14.0);
                results_card(ui, state, list_h, &mut act);
            }
            ui.add_space(20.0);
            footer(ui, state, submitted);
        });
    if win.is_some() {
        state.list_h_drawn = act.list_h;
    }

    if let Some(p) = act.open {
        open_in_default(&p);
    }
    if let Some(i) = act.delete {
        delete_finding(state, i);
    }

    // Clicking anywhere outside hides the window (a running scan keeps going —
    // messages are still drained at the top of `show`). Skipped on the opening
    // frame, when the Find Issues button click would count as "outside".
    if state.just_opened {
        state.just_opened = false;
    } else if let Some(win) = &win
        && win.response.clicked_elsewhere()
    {
        state.open = false;
    }
}

/// The title, bold, with its icon hanging to the left of it and the subtitle
/// under the title's text. No close button — clicking anywhere outside hides
/// the window (a running scan keeps going in the background).
fn header(ui: &mut egui::Ui) {
    let icon = 17.0;
    let gap = 15.0;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = gap;
        ui.add(
            egui::Image::new(egui::include_image!("../icons/frame_inspect.svg"))
                .fit_to_exact_size(egui::vec2(icon, icon))
                .tint(crate::theme::icon_tint(MUTED())),
        );
        ui.label(RichText::new("Find Issues").color(TEXT()).font(bold(19.0)));
    });
    ui.add_space(5.0);
    ui.horizontal(|ui| {
        ui.add_space(icon + gap);
        ui.label(RichText::new("Find corrupt images and exact duplicates.").color(MUTED()).size(12.5));
    });
}

/// A small spaced-out uppercase label over a card.
fn section_label(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).color(MUTED()).font(bold(9.5)).extra_letter_spacing(1.1));
}

/// The rounded card both sections sit in.
fn card(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(card_fill())
        .stroke(egui::Stroke::new(1.0_f32, hairline()))
        .corner_radius(egui::CornerRadius::same(CARD_R))
        .inner_margin(egui::Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// The folder and the duplicates switch. Returns true when Enter was
/// pressed in the folder field (start the scan).
fn scan_card(ui: &mut egui::Ui, state: &mut ScanState) -> bool {
    let enabled = !state.running;
    let mut submitted = false;
    card(ui, |ui| {
        let field = folder_pill(ui, &mut state.input_dir, enabled);
        submitted = field.submitted;
        if field.changed {
            state.folder_error = None;
        }
        if field.browse {
            let mut dialog = rfd::FileDialog::new();
            let current = Path::new(state.input_dir.trim());
            if current.is_dir() {
                dialog = dialog.set_directory(current);
            }
            if let Some(dir) = dialog.pick_folder() {
                state.input_dir = dir.display().to_string();
                state.folder_error = None;
            }
        }
        if let Some(e) = &state.folder_error {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.label(RichText::new(e).color(tone(Tone::Bad)).size(12.0));
            });
        }
        ui.add_space(14.0);
        divider(ui, 4.0);
        ui.add_space(14.0);
        ui.add_enabled_ui(enabled, |ui| {
            switch_row(
                ui,
                "Find exact duplicates",
                "SHA-256 \u{2014} extra copies move to a 'duplicates' folder.",
                &mut state.scan_duplicates,
            );
        });
    });
    submitted
}

#[derive(Default)]
struct FieldResult {
    changed: bool,
    /// Enter was pressed in it.
    submitted: bool,
    /// "Choose folder…" was clicked.
    browse: bool,
}

/// The folder as a pill: a folder icon, the path (editable — while it isn't
/// being edited it shows the end of a long path, the folder's own name), and
/// "Choose folder…" at the end, which opens the system's folder picker.
fn folder_pill(ui: &mut egui::Ui, text: &mut String, enabled: bool) -> FieldResult {
    let mut out = FieldResult::default();
    let id = ui.make_persistent_id("scan_folder_field");
    let focused = ui.memory(|m| m.has_focus(id));
    let h = 40.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let ring = if focused { accent() } else { Color32::TRANSPARENT };
    ui.painter().rect(rect, h / 2.0, well_fill(), egui::Stroke::new(1.0_f32, ring), egui::StrokeKind::Inside);

    let icon_at = egui::pos2(rect.left() + 24.0, rect.center().y);
    let icon_rect = egui::Rect::from_center_size(icon_at, egui::vec2(16.0, 16.0));
    let folder_tint = crate::theme::icon_tint(MUTED());
    egui::Image::new(egui::include_image!("../icons/folder.svg"))
        .tint(if enabled { folder_tint } else { folder_tint.gamma_multiply(0.5) })
        .paint_at(ui, icon_rect);

    // "Choose folder…", right-aligned in the pill.
    let font = egui::FontId::proportional(12.0);
    let choose = ui.painter().layout_no_wrap("Choose folder\u{2026}".to_string(), font, Color32::PLACEHOLDER);
    let choose_rect = egui::Rect::from_min_max(
        egui::pos2(rect.right() - 18.0 - choose.size().x - 8.0, rect.top()),
        egui::pos2(rect.right() - 8.0, rect.bottom()),
    );
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let browse = ui.interact(choose_rect, id.with("choose"), sense);
    let colour = if !enabled {
        MUTED().gamma_multiply(0.5)
    } else if browse.hovered() || browse.has_focus() {
        TEXT()
    } else {
        MUTED()
    };
    let at = egui::pos2(rect.right() - 18.0 - choose.size().x, rect.center().y - choose.size().y / 2.0);
    ui.painter().galley(at, choose, colour);
    out.browse = enabled && browse.on_hover_cursor(egui::CursorIcon::PointingHand).clicked();

    // The path, between the icon and "Choose folder…".
    let text_rect = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 46.0, rect.top()),
        egui::pos2(choose_rect.left() - 10.0, rect.bottom()),
    );
    let font = egui::FontId::proportional(13.0);
    if enabled && (focused || text.trim().is_empty()) {
        let resp = ui.put(
            text_rect,
            egui::TextEdit::singleline(text)
                .id(id)
                .frame(egui::Frame::NONE)
                .font(font)
                // Explicit: the Light theme's widget text is white (for its
                // blue buttons), which made a typed path white on a light field.
                .text_color(TEXT())
                .vertical_align(egui::Align::Center)
                .hint_text(RichText::new("No folder selected").color(MUTED()))
                .desired_width(text_rect.width()),
        );
        out.changed = resp.changed();
        out.submitted = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
    } else {
        let resp = ui.interact(text_rect, id.with("path"), sense);
        let colour = if enabled { TEXT() } else { MUTED() };
        let (galley, cut) = elide_front(ui, text.trim(), font, text_rect.width());
        let at = egui::pos2(text_rect.left(), text_rect.center().y - galley.size().y / 2.0);
        ui.painter().with_clip_rect(text_rect).galley(at, galley, colour);
        let resp = if cut { resp.on_hover_text(text.trim().to_string()) } else { resp };
        if enabled && resp.on_hover_cursor(egui::CursorIcon::Text).clicked() {
            // Edit it, with the cursor at the end.
            ui.memory_mut(|m| m.request_focus(id));
            let mut edit = egui::TextEdit::load_state(ui.ctx(), id).unwrap_or_default();
            let end = egui::text::CCursor::new(text.chars().count());
            edit.cursor.set_char_range(Some(egui::text::CCursorRange::one(end)));
            edit.store(ui.ctx(), id);
        }
    }
    out
}

/// `text` on one line, cut from the front with "…" until it fits `max_w`,
/// keeping its end. Says whether it had to cut. Laid out with the
/// placeholder colour: paint it with the colour to show.
fn elide_front(ui: &egui::Ui, text: &str, font: egui::FontId, max_w: f32) -> (Arc<egui::Galley>, bool) {
    let (shown, cut) = elide_front_text(ui, text, font.clone(), max_w);
    (ui.painter().layout_no_wrap(shown, font, Color32::PLACEHOLDER), cut)
}

/// [`elide_front`]'s text: `text`, or "…" and as much of its end as fits.
fn elide_front_text(ui: &egui::Ui, text: &str, font: egui::FontId, max_w: f32) -> (String, bool) {
    let width = |s: String| ui.painter().layout_no_wrap(s, font.clone(), Color32::PLACEHOLDER).size().x;
    if text.is_empty() || width(text.to_string()) <= max_w {
        return (text.to_string(), false);
    }
    let chars: Vec<char> = text.chars().collect();
    let cut = |skip: usize| -> String { std::iter::once('…').chain(chars[skip..].iter().copied()).collect() };
    // The fewest characters to drop for the rest to fit.
    let (mut lo, mut hi) = (1, chars.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if width(cut(mid)) <= max_w {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    (cut(lo), true)
}

/// A hairline across the card, `inset` in from both sides.
fn divider(ui: &mut egui::Ui, inset: f32) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
    let x = (r.left() + inset)..=(r.right() - inset);
    ui.painter().hline(x, r.center().y, egui::Stroke::new(1.0_f32, hairline()));
}

/// A bold label with a muted description under it, and a switch on the right.
fn switch_row(ui: &mut egui::Ui, label: &str, sub: &str, on: &mut bool) {
    ui.horizontal(|ui| {
        let text_w = ui.available_width() - 56.0;
        ui.vertical(|ui| {
            ui.set_width(text_w);
            ui.spacing_mut().item_spacing.y = 4.0;
            ui.label(RichText::new(label).color(TEXT()).font(bold(13.5)));
            ui.label(RichText::new(sub).color(MUTED()).size(12.0));
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            crate::settings::switch(ui, on);
        });
    });
}

/// The counts, a progress line while scanning (or how the scan ended), and
/// the list.
fn results_card(ui: &mut egui::Ui, state: &ScanState, list_h: f32, act: &mut Actions) {
    card(ui, |ui| {
        let sum = match &state.outcome {
            Some(Outcome::Finished(sum)) => Some(sum),
            _ => None,
        };
        let skipped = sum.and_then(skipped_media);
        let [verified, duplicates, corrupt] = state.counts();
        let verified_tip = match skipped {
            Some(s) => format!("Images that opened without errors. Videos and GIFs aren't checked: {s} skipped."),
            None => "Images that opened without errors.".to_string(),
        };
        let duplicates_tip = if sum.is_some_and(|s| !s.dupes_checked) && duplicates.is_none() {
            "Not looked for: Find exact duplicates was off, or nothing was readable."
        } else {
            "Extra copies of an image, moved to the \u{201c}duplicates\u{201d} folder. One of each is kept."
        };
        stats_row(
            ui,
            [
                ("VERIFIED", verified, verified_tip.as_str()),
                ("DUPLICATES", duplicates, duplicates_tip),
                ("CORRUPT", corrupt, "Images that are empty, cut off, or can't be decoded."),
            ],
        );

        if state.running {
            ui.add_space(14.0);
            progress_line(ui, state);
        } else if let Some(note) = ending_note(state) {
            ui.add_space(12.0);
            ui.label(RichText::new(note.0).color(note.1).size(12.0));
        }

        ui.add_space(14.0);
        if state.findings.is_empty() {
            if !state.running {
                empty_well(ui, sum);
            }
            return;
        }
        list(ui, state, list_h, act);
    });
}

/// "VERIFIED / 4,820" and its two neighbours, split by thin lines.
fn stats_row(ui: &mut egui::Ui, cols: [(&str, Option<u64>, &str); 3]) {
    let h = 38.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let w = rect.width() / 3.0;
    for (k, (label, value, tip)) in cols.into_iter().enumerate() {
        let col = egui::Rect::from_min_size(egui::pos2(rect.left() + w * k as f32, rect.top()), egui::vec2(w, h));
        let pad = if k == 0 { 2.0 } else { 18.0 };
        if k > 0 {
            ui.painter().vline(col.left(), col.y_range(), egui::Stroke::new(1.0_f32, hairline()));
        }
        let mut job = egui::text::LayoutJob::default();
        job.append(
            label,
            0.0,
            egui::TextFormat { font_id: bold(9.5), color: MUTED(), extra_letter_spacing: 1.1, ..Default::default() },
        );
        let label = ui.painter().layout_job(job);
        ui.painter().galley(egui::pos2(col.left() + pad, col.top()), label, MUTED());
        let (text, colour) = match value {
            Some(n) => (count(n), TEXT()),
            None => ("\u{2014}".to_string(), MUTED()),
        };
        let value = ui.painter().layout_no_wrap(text, bold(15.0), colour);
        ui.painter().galley(egui::pos2(col.left() + pad, col.bottom() - value.size().y), value, colour);
        ui.interact(col, ui.id().with(("scan_stat", k)), egui::Sense::hover()).on_hover_text(tip);
    }
}

/// While scanning: the step, "458 of 1,204", and a thin bar for the whole scan.
fn progress_line(ui: &mut egui::Ui, state: &ScanState) {
    let label = if state.cancelling {
        "Stopping\u{2026}"
    } else {
        match state.stage {
            Stage::Gathering => "Finding images\u{2026}",
            Stage::Checking => "Checking images",
            Stage::Hashing => "Looking for duplicates",
        }
    };
    let counted = state.stage != Stage::Gathering && state.total > 0;
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(MUTED()).size(12.0));
        if counted {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let n = format!("{} of {}", count(state.at as u64), count(state.total as u64));
                ui.label(RichText::new(n).color(MUTED()).size(12.0));
            });
        }
    });
    ui.add_space(8.0);
    // One bar for the whole scan: checking fills most of it, hashing the rest.
    let step = state.at.saturating_sub(1) as f32 / state.total.max(1) as f32;
    let frac = match state.stage {
        Stage::Gathering => None,
        Stage::Checking if state.scan_duplicates => Some(step * CHECK_SHARE),
        Stage::Checking => Some(step),
        Stage::Hashing => Some(CHECK_SHARE + step * (1.0 - CHECK_SHARE)),
    };
    progress_bar(ui, frac, egui::Id::new("scan_progress").with(state.runs));
}

/// A thin rounded bar in the accent colour, easing toward `frac`. `None`
/// (how much there is to do isn't known yet): a short piece slides along it.
pub(crate) fn progress_bar(ui: &mut egui::Ui, frac: Option<f32>, id: egui::Id) {
    let h = 4.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, h / 2.0, well_fill());
    match frac {
        Some(f) => {
            let f = ui.ctx().animate_value_with_time(id, f.clamp(0.0, 1.0), 0.25);
            if f > 0.0 {
                let w = (rect.width() * f).max(h);
                painter.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(w, h)), h / 2.0, accent());
            }
        }
        None => {
            let t = ui.input(|i| i.time) as f32;
            let piece = rect.width() * 0.28;
            let x = (t * 0.7).fract() * (rect.width() + piece) - piece;
            let bar = egui::Rect::from_min_size(egui::pos2(rect.left() + x, rect.top()), egui::vec2(piece, h))
                .intersect(rect);
            if bar.width() > 0.0 {
                painter.rect_filled(bar, h / 2.0, accent());
            }
            ui.ctx().request_repaint();
        }
    }
}

/// A line on how a scan that didn't finish ended (or on an empty folder).
fn ending_note(state: &ScanState) -> Option<(String, Color32)> {
    match state.outcome.as_ref()? {
        Outcome::Finished(_) => None,
        Outcome::Cancelled => {
            // The worker stops between files, so the last one it reported
            // was checked in full.
            let total = plural(state.total as u64, "image", "images");
            let done = match state.stage {
                Stage::Gathering => "before checking any images".to_string(),
                // Cancelled between checking the last image and hashing.
                Stage::Checking if state.at >= state.total => {
                    format!("after checking {total}, before looking for duplicates")
                }
                Stage::Checking => format!("after checking {} of {total}", count(state.at as u64)),
                Stage::Hashing => "while looking for duplicates".to_string(),
            };
            Some((format!("Scan cancelled {done}. Nothing was moved."), MUTED()))
        }
        Outcome::Failed(why) => Some((why.clone(), tone(Tone::Bad))),
    }
}

/// The list's place when there's nothing in it.
fn empty_well(ui: &mut egui::Ui, sum: Option<&Summary>) {
    let (icon, colour, text) = match sum {
        Some(s) if s.scanned == 0 => (
            egui::include_image!("../icons/info.svg"),
            MUTED(),
            format!("No images to check. Find Issues reads {}.", formats()),
        ),
        Some(s) => (
            egui::include_image!("../icons/check_circle.svg"),
            tone(Tone::Good),
            if s.dupes_checked { "No corrupt images or duplicates found." } else { "No corrupt images found." }
                .to_string(),
        ),
        // Cancelled or stopped before finding anything.
        None => (egui::include_image!("../icons/info.svg"), MUTED(), "Nothing found before it stopped.".to_string()),
    };
    egui::Frame::new()
        .fill(well_fill())
        .corner_radius(egui::CornerRadius::same(WELL_R))
        .inner_margin(egui::Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 10.0;
                ui.add(egui::Image::new(icon).fit_to_exact_size(egui::vec2(16.0, 16.0)).tint(colour));
                ui.add(egui::Label::new(RichText::new(text).color(MUTED()).size(12.5)).wrap());
            });
        });
}

/// Every file that needs a look, in a well: one fixed-height row each, only
/// those in view drawn.
fn list(ui: &mut egui::Ui, state: &ScanState, list_h: f32, act: &mut Actions) {
    let rows = state.findings.len();
    let scrolls = rows as f32 * ROW_H > list_h + 0.5;
    egui::Frame::new()
        .fill(well_fill())
        .corner_radius(egui::CornerRadius::same(WELL_R))
        .inner_margin(egui::Margin::symmetric(0, 3))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 0.0;
            let out = egui::ScrollArea::vertical()
                // A new scan's list starts at its top.
                .id_salt(("scan_findings", state.runs))
                .max_height(list_h)
                // Its full height at once: the window hugs its content, so it
                // never has room to spare below a scroll area to grow into.
                .min_scrolled_height(list_h)
                .auto_shrink([false, true])
                .show_rows(ui, ROW_H, rows, |ui, range| {
                    // Keep the row ends clear of the scroll bar.
                    ui.set_width(ui.available_width() - if scrolls { 10.0 } else { 0.0 });
                    for i in range {
                        finding_row(ui, &state.root, i, &state.findings[i], act);
                    }
                });
            act.list_h = out.inner_rect.height();
        });
}

/// One row: a kind icon, the name (subfolder muted), and its status on the
/// right — "Corrupt · Truncated file", "Duplicate · Moved". Hovering a row
/// swaps the status for its buttons (open; delete for a corrupt file);
/// clicking the row opens the file. A deleted file keeps its row, struck
/// through, so the rows below don't move up under the pointer.
fn finding_row(ui: &mut egui::Ui, root: &Path, i: usize, f: &Finding, act: &mut Actions) {
    let (rect, row) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H), egui::Sense::click());
    let inner = rect.shrink2(egui::vec2(12.0, 0.0));
    if i > 0 {
        ui.painter().hline(inner.x_range(), rect.top(), egui::Stroke::new(1.0_f32, hairline()));
    }
    let deleted = matches!(&f.what, What::Corrupt { status, .. } if status.is_deleted());
    // Where a click goes: the file, or a moved duplicate's new place.
    let target = match &f.what {
        What::Duplicate { moved: Moved::To(to) | Moved::WithoutTags(to, _), .. } => Some(to.clone()),
        _ if deleted => None,
        _ => Some(f.path.clone()),
    };
    let hot = target.is_some() && row.contains_pointer();
    if hot {
        ui.painter().rect_filled(rect.shrink2(egui::vec2(5.0, 2.0)), 8.0, TEXT().gamma_multiply(0.05));
    }

    // The kind icon.
    let icon_c = egui::pos2(inner.left() + 8.0, rect.center().y);
    match &f.what {
        What::Corrupt { .. } => alert_icon(ui, icon_c, if deleted { MUTED() } else { tone(Tone::Warn) }),
        What::Unchecked { .. } => {
            egui::Image::new(egui::include_image!("../icons/info.svg"))
                .tint(MUTED())
                .paint_at(ui, egui::Rect::from_center_size(icon_c, egui::vec2(15.0, 15.0)));
        }
        What::Duplicate { .. } => {
            egui::Image::new(egui::include_image!("../icons/copy.svg"))
                .tint(MUTED())
                .paint_at(ui, egui::Rect::from_center_size(icon_c, egui::vec2(15.0, 15.0)));
        }
    }

    // The status on the right, or — on hover — the row's buttons there.
    let (status, status_colour, tip) = row_status(f);
    let mut right = inner.right();
    // Over a button, its own tooltip — not the row's.
    let mut on_button = false;
    if hot {
        let mut button = |icon: egui::ImageSource<'static>, tip: &str, hover: Color32, salt: &str| {
            let r = egui::Rect::from_center_size(egui::pos2(right - 13.0, rect.center().y), egui::vec2(26.0, 26.0));
            right -= 30.0;
            let resp = ui.interact(r, ui.id().with((salt, i)), egui::Sense::click());
            on_button |= resp.contains_pointer();
            paint_icon_button(ui, r, &resp, icon, true, hover);
            resp.on_hover_text(tip).on_hover_cursor(egui::CursorIcon::PointingHand).clicked()
        };
        let delete = egui::include_image!("../icons/delete.svg");
        if let What::Corrupt { status: FindingStatus::Listed | FindingStatus::DeleteFailed(_), .. } = &f.what
            && button(delete, "Delete this file and its tags", tone(Tone::Bad), "del")
        {
            act.delete = Some(i);
        }
        let moved_copy = matches!(f.what, What::Duplicate { moved: Moved::To(_) | Moved::WithoutTags(..), .. });
        let open_tip = if moved_copy { "Open the moved copy" } else { "Open it" };
        if button(egui::include_image!("../icons/eye.svg"), open_tip, TEXT(), "open") {
            act.open = target.clone();
        }
    } else {
        let galley = ui.painter().layout_no_wrap(status, egui::FontId::proportional(11.5), status_colour);
        right -= galley.size().x;
        ui.painter().galley(egui::pos2(right, rect.center().y - galley.size().y / 2.0), galley, status_colour);
    }

    // The name. If it would run into the status, the folder part gives way
    // first (cut from its front), so the file's own name stays readable.
    let (folder, file) = name_parts(root, &f.path);
    let left = inner.left() + 26.0;
    let room = (right - 16.0 - left).max(40.0);
    let font = egui::FontId::proportional(12.5);
    let file_w = ui.painter().layout_no_wrap(file.clone(), font.clone(), Color32::PLACEHOLDER).size().x;
    // What's left beside the whole file name; too little, and it's left out.
    let folder_room = room - file_w;
    let folder = if folder_room >= 24.0 {
        elide_front_text(ui, &folder, font.clone(), folder_room).0
    } else {
        String::new()
    };
    let format = |color| egui::TextFormat {
        font_id: font.clone(),
        color,
        strikethrough: if deleted { egui::Stroke::new(1.0_f32, MUTED()) } else { egui::Stroke::NONE },
        ..Default::default()
    };
    let mut job = egui::text::LayoutJob::default();
    job.append(&folder, 0.0, format(MUTED()));
    job.append(&file, 0.0, format(if deleted { MUTED() } else { TEXT() }));
    job.wrap = egui::text::TextWrapping::truncate_at_width(room);
    let name = ui.painter().layout_job(job);
    ui.painter().galley(egui::pos2(left, rect.center().y - name.size().y / 2.0), name, TEXT());

    let row = if on_button { row } else { row.on_hover_text(tip) };
    if target.is_some() {
        let row = row.on_hover_cursor(egui::CursorIcon::PointingHand);
        if row.clicked() {
            act.open = target;
        }
    }
}

/// A row's status text, its colour, and its tooltip (the full path and the
/// whole story).
fn row_status(f: &Finding) -> (String, Color32, String) {
    let path = f.path.display();
    match &f.what {
        What::Corrupt { reason, status } => match status {
            FindingStatus::Listed => (
                format!("Corrupt \u{b7} {}", short_reason(reason)),
                tone(Tone::Warn),
                format!("{path}\n{reason}\nClick to open it."),
            ),
            FindingStatus::Deleted => ("Deleted".to_string(), MUTED(), format!("{path}\nDeleted.")),
            FindingStatus::DeletedTagsStayed(e) => (
                "Deleted \u{b7} tags stayed".to_string(),
                tone(Tone::Warn),
                format!("{path}\nDeleted, but its tags couldn't be: {e}"),
            ),
            FindingStatus::DeleteFailed(e) => {
                ("Couldn't delete".to_string(), tone(Tone::Bad), format!("{path}\nCouldn't delete it: {e}"))
            }
        },
        What::Unchecked { reason } => {
            let low = reason.to_ascii_lowercase();
            let why = if low.starts_with("can't read") {
                "Can't read"
            } else if low.contains("limit") {
                "Too large"
            } else {
                "Unsupported"
            };
            (
                format!("Not checked \u{b7} {why}"),
                MUTED(),
                format!("{path}\n{reason}\nIt couldn't be decoded here, but it may well be fine."),
            )
        }
        What::Duplicate { kept, moved } => {
            let kept = kept.display();
            match moved {
                Moved::To(to) => (
                    "Duplicate \u{b7} Moved".to_string(),
                    MUTED(),
                    format!("{path}\nSame as {kept}\nMoved to {}", to.display()),
                ),
                Moved::WithoutTags(to, e) => (
                    "Duplicate \u{b7} Moved, tags stayed".to_string(),
                    tone(Tone::Warn),
                    format!("{path}\nSame as {kept}\nMoved to {}\nIts tags couldn't follow: {e}", to.display()),
                ),
                Moved::Failed(e) => (
                    "Duplicate \u{b7} Couldn't move".to_string(),
                    tone(Tone::Bad),
                    format!("{path}\nSame as {kept}\nCouldn't move it: {e}"),
                ),
            }
        }
    }
}

/// A circle with an exclamation mark — the corrupt-file icon.
fn alert_icon(ui: &egui::Ui, c: egui::Pos2, colour: Color32) {
    let r = 6.0;
    let stroke = egui::Stroke::new(1.4_f32, colour);
    let painter = ui.painter();
    painter.circle_stroke(c, r, stroke);
    painter.line_segment([c - egui::vec2(0.0, 2.9), c + egui::vec2(0.0, 0.7)], stroke);
    painter.circle_filled(c + egui::vec2(0.0, 2.7), 0.85, colour);
}

/// The bottom row: Copy list on the left (once there is a list), the main
/// button on the right — Start Scan, or Cancel while running.
fn footer(ui: &mut egui::Ui, state: &mut ScanState, submitted: bool) {
    ui.horizontal(|ui| {
        ui.set_min_height(36.0);
        if !state.findings.is_empty() {
            let copied = state.copied_at.is_some_and(|t| t.elapsed() < Duration::from_millis(1200));
            if copied {
                ui.ctx().request_repaint_after(Duration::from_millis(100));
            }
            let label = if copied { "Copied" } else { "Copy list" };
            if pill(ui, label, Pill::Outline, None, true).on_hover_text("Copy the list as text").clicked() {
                ui.ctx().copy_text(list_text(state));
                state.copied_at = Some(Instant::now());
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Flash green/red on click; `fire_flash` acts once the flash has
            // shown for `FLASH` — same as Create Backup.
            if state.running {
                let flashing = state.cancel_flash.is_some();
                let label = if state.cancelling { "Stopping\u{2026}" } else { "Cancel" };
                let ready = !state.cancelling;
                let b = pill(ui, label, Pill::Outline, flashing.then_some(FLASH_RED), ready);
                if b.clicked() && ready && !flashing {
                    state.cancel_flash = Some(Instant::now());
                }
            } else {
                let flashing = state.start_flash.is_some();
                let ready = !state.input_dir.trim().is_empty();
                let b = pill(ui, "Start Scan", Pill::Accent, flashing.then_some(FLASH_GREEN), ready);
                if (b.clicked() || submitted) && ready && !flashing {
                    state.start_flash = Some(Instant::now());
                }
            }
        });
    });
}

/// The list as plain text, for the clipboard: what was scanned, the counts,
/// then one line per file.
fn list_text(state: &ScanState) -> String {
    let [verified, duplicates, corrupt] = state.counts();
    let n = |v: Option<u64>| v.map_or("\u{2014}".to_string(), count);
    let mut out = format!(
        "Find Issues: {}\nVerified {}, duplicates {}, corrupt {}\n",
        state.root.display(),
        n(verified),
        n(duplicates),
        n(corrupt)
    );
    for f in &state.findings {
        let status = match &f.what {
            What::Corrupt { status: FindingStatus::Listed, .. } => row_status(f).0,
            What::Corrupt { .. } => format!("Corrupt \u{b7} {}", row_status(f).0),
            _ => row_status(f).0,
        };
        let detail = match &f.what {
            What::Corrupt { reason, status: FindingStatus::DeleteFailed(e) | FindingStatus::DeletedTagsStayed(e) } => {
                format!(" ({reason}; {e})")
            }
            What::Corrupt { reason, .. } | What::Unchecked { reason } => format!(" ({reason})"),
            What::Duplicate { moved: Moved::To(to) | Moved::WithoutTags(to, _), .. } => {
                format!(" (now {})", to.display())
            }
            What::Duplicate { moved: Moved::Failed(e), .. } => format!(" ({e})"),
        };
        out.push_str(&format!("{status}: {}{detail}\n", f.path.display()));
    }
    out
}

/// A button's click-flash has shown for `FLASH`: do what was clicked. Runs
/// every frame, so a click still counts when the window is hidden during
/// its flash.
fn fire_flash(state: &mut ScanState, ctx: &egui::Context) {
    if state.start_flash.is_some_and(|t| t.elapsed() >= FLASH) {
        state.start_flash = None;
        start_scan(state, ctx);
    }
    if state.cancel_flash.is_some_and(|t| t.elapsed() >= FLASH) {
        state.cancel_flash = None;
        if state.running {
            state.cancel.store(true, Ordering::SeqCst);
            state.cancelling = true;
        }
    }
    if state.start_flash.is_some() || state.cancel_flash.is_some() {
        ctx.request_repaint();
    }
}

/// The footer buttons' two looks.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Pill {
    /// Filled with the accent colour, white bold label.
    Accent,
    /// A thin outline, muted label.
    Outline,
}

/// A pill-shaped footer button, painted the same in every theme. `flash`
/// fills it for the click-flash.
pub(crate) fn pill(ui: &mut egui::Ui, label: &str, kind: Pill, flash: Option<Color32>, enabled: bool) -> egui::Response {
    let font = if kind == Pill::Accent { bold(13.5) } else { egui::FontId::proportional(12.5) };
    let galley = ui.painter().layout_no_wrap(label.to_string(), font, Color32::PLACEHOLDER);
    let (pad, h) = if kind == Pill::Accent { (24.0, 34.0) } else { (17.0, 32.0) };
    let size = egui::vec2((galley.size().x + 2.0 * pad).max(if kind == Pill::Accent { 108.0 } else { 0.0 }), h);
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(size, sense);
    let hot = enabled && (resp.hovered() || resp.has_focus());
    let down = enabled && resp.is_pointer_button_down_on();
    let painter = ui.painter();
    let ink = match kind {
        _ if flash.is_some() => Color32::WHITE,
        Pill::Accent if enabled => Color32::WHITE,
        Pill::Accent => Color32::WHITE.gamma_multiply(0.7),
        Pill::Outline if !enabled => MUTED().gamma_multiply(0.5),
        Pill::Outline if hot => TEXT(),
        Pill::Outline => MUTED(),
    };
    match (kind, flash) {
        (_, Some(c)) => {
            painter.rect_filled(rect, h / 2.0, c);
        }
        (Pill::Accent, None) => {
            let fill = if !enabled {
                accent().gamma_multiply(0.45)
            } else if down {
                shade(accent(), -0.12)
            } else if hot {
                shade(accent(), 0.1)
            } else {
                accent()
            };
            painter.rect_filled(rect, h / 2.0, fill);
        }
        (Pill::Outline, None) => {
            let fill = if hot { card_fill().lerp_to_gamma(TEXT(), 0.05) } else { card_fill() };
            painter.rect_filled(rect, h / 2.0, fill);
            let edge = if hot { TEXT().gamma_multiply(0.3) } else { hairline_strong() };
            painter.rect_stroke(rect, h / 2.0, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);
        }
    }
    painter.galley(rect.center() - galley.size() / 2.0, galley, ink);
    if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp }
}

fn paint_icon_button(
    ui: &egui::Ui,
    rect: egui::Rect,
    resp: &egui::Response,
    icon: egui::ImageSource<'static>,
    enabled: bool,
    hover: Color32,
) {
    let hot = enabled && (resp.hovered() || resp.has_focus());
    if hot {
        ui.painter().rect_filled(rect, 7.0, TEXT().gamma_multiply(0.08));
    }
    let rest = crate::theme::icon_tint(MUTED());
    let tint = if !enabled {
        rest.gamma_multiply(0.4)
    } else if hot {
        hover
    } else {
        rest
    };
    egui::Image::new(icon).tint(tint).paint_at(ui, rect.shrink(5.0));
}

/// The theme's accent: its blue, or Aurora's pink (matching that theme's buttons).
fn accent() -> Color32 {
    crate::theme::selection_outline()
}

/// Ubuntu Medium — the app's real bold face (`strong()` only brightens).
fn bold(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name("bold".into()))
}

/// `c` lightened (`t` > 0, toward white) or darkened (`t` < 0, toward black).
fn shade(c: Color32, t: f32) -> Color32 {
    if t >= 0.0 {
        c.lerp_to_gamma(Color32::WHITE, t)
    } else {
        c.lerp_to_gamma(Color32::BLACK, -t)
    }
}

/// The cards: the window's colour lifted a little toward the text colour
/// (lighter in dark themes, a touch greyer in light ones).
fn card_fill() -> Color32 {
    FLOAT().lerp_to_gamma(TEXT(), 0.035)
}

/// The folder field and the list: sunk below the cards (darker in dark
/// themes, greyer in light ones).
fn well_fill() -> Color32 {
    if crate::theme::is_light() {
        FLOAT().lerp_to_gamma(TEXT(), 0.07)
    } else {
        FLOAT().lerp_to_gamma(Color32::BLACK, 0.28)
    }
}

/// Card outlines, dividers and row separators.
fn hairline() -> Color32 {
    TEXT().gamma_multiply(0.08)
}

/// The outline of the Copy list button, a little stronger.
fn hairline_strong() -> Color32 {
    TEXT().gamma_multiply(0.16)
}

#[derive(Clone, Copy)]
enum Tone {
    Good,
    Warn,
    Bad,
}

/// Green / amber / red, deeper on light themes so text in them stays readable.
fn tone(t: Tone) -> Color32 {
    let light = crate::theme::is_light();
    match t {
        Tone::Good if light => Color32::from_rgb(26, 135, 60),
        Tone::Good => Color32::from_rgb(88, 196, 112),
        Tone::Warn if light => Color32::from_rgb(170, 98, 0),
        Tone::Warn => Color32::from_rgb(232, 160, 88),
        Tone::Bad if light => Color32::from_rgb(196, 46, 46),
        Tone::Bad => Color32::from_rgb(240, 104, 104),
    }
}

/// `n` with thousands separators: 1204 → "1,204".
fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// "1 corrupt image", "1,204 corrupt images".
fn plural(n: u64, one: &str, many: &str) -> String {
    format!("{} {}", count(n), if n == 1 { one } else { many })
}

/// The skipped media in words: "3 videos", "2 GIFs", "3 videos and 2 GIFs".
fn skipped_media(sum: &Summary) -> Option<String> {
    let gifs = u64::from(sum.skipped_gifs);
    let videos = u64::from(sum.skipped).saturating_sub(gifs);
    match (videos, gifs) {
        (0, 0) => None,
        (v, 0) => Some(plural(v, "video", "videos")),
        (0, g) => Some(plural(g, "GIF", "GIFs")),
        (v, g) => Some(format!("{} and {}", plural(v, "video", "videos"), plural(g, "GIF", "GIFs"))),
    }
}

/// The formats the corruption scan reads, for an empty folder's note.
fn formats() -> &'static str {
    if cfg!(feature = "avif") {
        "PNG, JPEG, WebP, BMP, TIFF, ICO, HDR, SVG, AVIF, HEIC and camera RAW"
    } else {
        "PNG, JPEG, WebP, BMP, TIFF, ICO, HDR and SVG"
    }
}

/// A decoder's reason in a few words, for the row's status ("Truncated
/// file"). The whole reason is in the row's tooltip and the copied list.
fn short_reason(reason: &str) -> &'static str {
    let low = reason.to_ascii_lowercase();
    if low.contains("unexpected end of file")
        || low.contains("unexpected eof")
        || low.contains("not enough bytes")
        || low.contains("failed to fill whole buffer")
        || low.contains("truncated")
    {
        "Truncated file"
    } else if low.contains("signature") || low.contains("could not be determined") || low.contains("header") {
        "Bad header"
    } else if low == "empty file" {
        "Empty file"
    } else if low == "invalid dimensions" {
        "Bad dimensions"
    } else if low.contains("crashed") {
        "Decoder crashed"
    } else if low.starts_with("unreadable") || low.starts_with("open failed") {
        "Can't read file"
    } else {
        "Can't decode"
    }
}

/// `path`'s subfolder in the scanned folder, with a trailing separator
/// ("sketches/", or "" at the top), and its file name.
fn name_parts(root: &Path, path: &Path) -> (String, String) {
    let rel = rel_name(root, path);
    let rel = Path::new(&rel);
    let folder = match rel.parent() {
        Some(p) if !p.as_os_str().is_empty() => format!("{}{}", p.display(), std::path::MAIN_SEPARATOR),
        _ => String::new(),
    };
    (folder, file_name(rel))
}

/// `path` relative to the scanned folder ("sketches/lake.jpg"), or its file
/// name when it isn't inside it.
fn rel_name(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(rel) if !rel.as_os_str().is_empty() && !root.as_os_str().is_empty() => rel.display().to_string(),
        _ => file_name(path),
    }
}

/// Delete corrupt file `i`, and its tags unless another file uses them. Its
/// row stays (marked deleted), so the rows below don't shift under the
/// pointer between two clicks.
fn delete_finding(state: &mut ScanState, i: usize) {
    let Some(f) = state.findings.get_mut(i) else { return };
    let What::Corrupt { status, .. } = &mut f.what else { return };
    if status.is_deleted() {
        return;
    }
    let p = f.path.clone();
    // One already gone (deleted in the browser meanwhile) counts as deleted —
    // unless its folder is unreachable too (an unplugged drive). Either way
    // it leaves the browser's list.
    let gone = |e: &std::io::Error| e.kind() == std::io::ErrorKind::NotFound && p.parent().is_some_and(Path::is_dir);
    match std::fs::remove_file(&p).or_else(|e| if gone(&e) { Ok(()) } else { Err(e) }) {
        Ok(()) => {
            state.deleted.push(p.clone());
            // Its tags go too, like the browser's Delete — unless another
            // file there still uses that .txt.
            let txt = p.with_extension("txt");
            *status = if txt.is_file() && !sidecar_in_use(&txt) {
                match std::fs::remove_file(&txt) {
                    Ok(()) => FindingStatus::Deleted,
                    Err(e) => FindingStatus::DeletedTagsStayed(short_err(&e.to_string())),
                }
            } else {
                FindingStatus::Deleted
            };
        }
        Err(e) => *status = FindingStatus::DeleteFailed(short_err(&e.to_string())),
    }
}

/// Open `path` in the OS default application (to review a flagged file, or
/// a moved duplicate).
fn open_in_default(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // Explorer opens a file with its default association. Not `cmd /C
        // start`: Rust quotes an argument only when it has spaces, so a `&` in
        // a file name reached cmd as a command separator (`x&calc&.png` ran
        // calc). Explorer runs nothing from its arguments; the path goes as
        // one quoted argument (Windows names can't contain `"`) so a comma in
        // a name isn't read as an Explorer switch separator.
        let mut quoted = std::ffi::OsString::from("\"");
        quoted.push(path.as_os_str());
        quoted.push("\"");
        let _ = std::process::Command::new("explorer").raw_arg(quoted).spawn();
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

fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(FLOAT())
        .corner_radius(egui::CornerRadius::same(24))
        .inner_margin(egui::Margin::same(MARGIN))
        .stroke(egui::Stroke::new(1.0_f32, hairline()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 8],
            blur: 28,
            spread: 0,
            color: Color32::from_black_alpha(150),
        })
}

/// Validate the folder and spawn the worker.
fn start_scan(state: &mut ScanState, ctx: &egui::Context) {
    if state.running {
        return;
    }
    let dir = PathBuf::from(state.input_dir.trim());
    if state.input_dir.trim().is_empty() {
        state.folder_error = Some("Choose a folder to scan first.".to_string());
        return;
    }
    if !dir.is_dir() {
        state.folder_error = Some("Can't find that folder. Check the path, or use Choose folder\u{2026}".to_string());
        return;
    }

    state.folder_error = None;
    state.findings.clear();
    state.outcome = None;
    state.stage = Stage::Gathering;
    state.at = 0;
    state.total = 0;
    state.root = dir.clone();
    state.runs = state.runs.wrapping_add(1);
    state.cancelling = false;
    state.copied_at = None;

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
    let send = |m: Msg| {
        let _ = tx.send(m);
        ctx.request_repaint();
    };

    let mut sum = Summary::default();
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
            if ext == "gif" {
                sum.skipped_gifs += 1;
            }
        }
    }

    // --- Phase 1: corruption ---
    let mut valid: Vec<PathBuf> = Vec::new();
    for (i, p) in images.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            finish(&tx, &ctx, sum, true);
            return;
        }
        send(Msg::Progress { stage: Stage::Checking, at: i + 1, total: images.len() });

        // A decoder that panics on a damaged file marks that file corrupt
        // instead of killing the whole scan.
        let checked = std::panic::catch_unwind(|| validate_image(p))
            .unwrap_or_else(|_| Err(Problem::Corrupt("The decoder crashed on this file".to_string())));
        match checked {
            Ok(()) => valid.push(p.clone()),
            // Corrupt files are NOT moved — they're listed for review (open or
            // delete, per file).
            Err(Problem::Corrupt(reason)) => {
                send(Msg::Corrupt(p.clone(), reason));
                sum.corrupted += 1;
            }
            // Most likely fine, just not readable here: listed, never offered
            // for deletion, still looked at for duplicates.
            Err(Problem::Unchecked(reason)) => {
                send(Msg::Unchecked(p.clone(), reason));
                sum.unchecked += 1;
                valid.push(p.clone());
            }
        }
    }
    sum.scanned = images.len() as u32;

    // --- Phase 2: exact duplicates ---
    if scan_dupes && !valid.is_empty() {
        if cancel.load(Ordering::SeqCst) {
            finish(&tx, &ctx, sum, true);
            return;
        }
        let dup_dir = dir.join(DUPLICATES_FOLDER);

        use std::collections::HashMap;
        let mut by_hash: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for (i, p) in valid.iter().enumerate() {
            if cancel.load(Ordering::SeqCst) {
                finish(&tx, &ctx, sum, true);
                return;
            }
            send(Msg::Progress { stage: Stage::Hashing, at: i + 1, total: valid.len() });
            if let Some(h) = sha256_file(p) {
                by_hash.entry(h).or_default().push(p.clone());
            }
        }

        let mut groups: Vec<Vec<PathBuf>> = by_hash.into_values().filter(|g| g.len() > 1).collect();
        sum.dupes_checked = true;
        sum.duplicates_found = groups.iter().map(|g| g.len() - 1).sum::<usize>() as u32;
        if !groups.is_empty() {
            let _ = std::fs::create_dir_all(&dup_dir);
        }
        // Keep the alphabetically-first file of each group and move the rest.
        // Groups go in path order too, so the same folder numbers its moved
        // copies (dup_1_, dup_2_…) the same way every time.
        for g in &mut groups {
            g.sort();
        }
        groups.sort();
        for (gi, group) in groups.into_iter().enumerate() {
            let kept = group[0].clone();
            for dup in group.into_iter().skip(1) {
                let moved = move_into(&dup, &dup_dir, &format!("dup_{}_", gi + 1));
                if !matches!(moved, Moved::Failed(_)) {
                    sum.duplicates_moved += 1;
                }
                send(Msg::Duplicate { path: dup, kept: kept.clone(), moved });
            }
        }
    }
    finish(&tx, &ctx, sum, false);
}

fn finish(tx: &Sender<Msg>, ctx: &egui::Context, mut sum: Summary, cancelled: bool) {
    sum.cancelled = cancelled;
    let _ = tx.send(Msg::Done(sum));
    ctx.request_repaint();
}

/// Why an image didn't pass [`validate_image`].
#[derive(Debug, PartialEq)]
enum Problem {
    /// It's broken: empty, cut off, or not what its name says.
    Corrupt(String),
    /// It couldn't be decoded here, though it may well be fine: a variant of
    /// its format the decoder doesn't support, or too big to decode.
    Unchecked(String),
}

/// Decode-validate one image. `Ok(())` = readable.
fn validate_image(p: &Path) -> Result<(), Problem> {
    let corrupt = |s: &str| Err(Problem::Corrupt(s.to_string()));
    // Not being able to read a file (locked by another app, no permission,
    // gone mid-scan) says nothing about the image in it.
    let unreadable =
        |e: &dyn std::fmt::Display| Err(Problem::Unchecked(format!("Can't read it: {}", short_err(&e.to_string()))));
    match std::fs::metadata(p) {
        Ok(m) if m.len() == 0 => return corrupt("Empty file"),
        Err(e) => return unreadable(&e),
        _ => {}
    }

    let ext = ext_of(p);

    // SVG is a vector the `image` crate can't parse — validate it through the
    // rasterizer (small target: parsing is the check, size doesn't matter).
    if ext == "svg" {
        return match crate::image_cache::decode_svg(p, 64) {
            Some(_) => Ok(()),
            None => corrupt("Could not decode"),
        };
    }

    // HDR goes through our tone-mapping decoder — `image`'s reader rejects the
    // valid `#?RGBE` signature variant, which would wrongly flag good files.
    if ext == "hdr" {
        return match crate::image_cache::decode_hdr(p) {
            Some(_) => Ok(()),
            None => corrupt("Could not decode"),
        };
    }

    // Extended formats go through our own decoder.
    #[cfg(feature = "avif")]
    if EXTENDED_EXTS.contains(&ext.as_str()) {
        return match crate::avif::decode_avif(p) {
            Some(_) => Ok(()),
            None => corrupt("Could not decode"),
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
                        corrupt("Invalid dimensions")
                    } else {
                        Ok(())
                    }
                }
                // Not damage: a feature of the format the decoder lacks, or
                // over its memory limit. Calling those corrupt offered to
                // delete good files.
                Err(e @ (image::ImageError::Unsupported(_) | image::ImageError::Limits(_))) => {
                    Err(Problem::Unchecked(short_err(&e.to_string())))
                }
                // Running out of file part-way is damage (cut off); other
                // read errors aren't.
                Err(image::ImageError::IoError(e)) if e.kind() != std::io::ErrorKind::UnexpectedEof => unreadable(&e),
                Err(e) => Err(Problem::Corrupt(short_err(&e.to_string()))),
            },
            Err(e) => unreadable(&e),
        },
        Err(e) => unreadable(&e),
    }
}

/// Move `file` into `dest` with a name prefix, de-duplicating the target name.
/// Its tag sidecar (`.txt`) goes along under the same new name, so a moved
/// duplicate doesn't leave orphaned tags behind — copied rather than moved
/// when another file still uses that `.txt` (`cat.png` + `cat.jpg` share
/// `cat.txt`).
fn move_into(file: &Path, dest: &Path, prefix: &str) -> Moved {
    let base = file_name(file);
    let txt = file.with_extension("txt");
    let has_txt = txt.is_file();
    let target = unique_target(dest, &format!("{prefix}{base}"), has_txt);
    if let Err(e) = move_file(file, &target) {
        return Moved::Failed(short_err(&e.to_string()));
    }
    if has_txt {
        let txt_target = target.with_extension("txt");
        let done = if sidecar_in_use(&txt) {
            std::fs::copy(&txt, &txt_target).map(|_| ())
        } else {
            move_file(&txt, &txt_target)
        };
        if let Err(e) = done {
            return Moved::WithoutTags(target, short_err(&e.to_string()));
        }
    }
    Moved::To(target)
}

/// `rename`, falling back to copy + delete (rename fails across volumes) —
/// the browser's Move helper: the copy lands under a temporary name first and
/// a failed copy or delete leaves neither a half-written file nor a duplicate.
fn move_file(from: &Path, to: &Path) -> std::io::Result<()> {
    crate::move_file(from, to)
}

/// Whether another image or video in `txt`'s folder uses it as its tag
/// sidecar (`cat.png` and `cat.jpg` both use `cat.txt`) — then the `.txt`
/// must stay. The same rule as the browser's Delete / Move: every type the
/// app reads a .txt for counts (a `cat.npz` latent cache doesn't), and names
/// compare ignoring case on every OS (`Dog.JPG` uses `dog.txt` on Windows,
/// macOS and a Linux exFAT drive; elsewhere keeping the .txt is harmless). A
/// folder that can't be read counts as in use (keeping a `.txt` is the safe
/// side). Call it once the image itself has moved or gone.
pub(crate) fn sidecar_in_use(txt: &Path) -> bool {
    let (Some(dir), Some(stem)) = (txt.parent(), txt.file_stem()) else { return true };
    let Ok(entries) = std::fs::read_dir(dir) else { return true };
    entries.flatten().any(|e| {
        let p = e.path();
        !e.file_type().is_ok_and(|t| t.is_dir())
            && p.file_stem().is_some_and(|s| crate::same_name(s, stem))
            && crate::has_sidecar_extension(&p)
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Recursively collect regular files, skipping our own quarantine folders.
/// Symlinks (and Windows junctions) are neither followed nor listed: a file
/// reached two ways counted as its own duplicate, and moving "the copy" moved
/// the real file.
fn gather_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            // The entry's own type: unlike `Path::is_dir`/`is_file`, it doesn't
            // follow a link to what it points at.
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.eq_ignore_ascii_case(CORRUPTED_FOLDER) || name.eq_ignore_ascii_case(DUPLICATES_FOLDER) {
                    continue;
                }
                stack.push(p);
            } else if kind.is_file() {
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
    // Cut after 80 characters, not bytes: byte 80 can fall inside a
    // multi-byte character (a localized OS error message), and slicing there
    // panicked — killing the scan worker.
    match first.char_indices().nth(80) {
        Some((i, _)) => format!("{}…", &first[..i]),
        None => first.to_string(),
    }
}

/// A non-clashing path in `dest` for `name`, appending " (n)" if needed. With
/// `with_txt`, the name's `.txt` sibling must be free too (a tag sidecar
/// follows the file there).
fn unique_target(dest: &Path, name: &str, with_txt: bool) -> PathBuf {
    let free = |p: &Path| !p.exists() && (!with_txt || !p.with_extension("txt").exists());
    let base = dest.join(name);
    if free(&base) {
        return base;
    }
    let (stem, ext) = match name.rfind('.') {
        Some(i) => (&name[..i], &name[i..]),
        None => (name, ""),
    };
    for i in 1..100_000 {
        let cand = dest.join(format!("{stem} ({i}){ext}"));
        if free(&cand) {
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

        use std::fmt::Write as _;
        let mut out = String::with_capacity(64);
        for word in self.state.iter() {
            let _ = write!(out, "{word:08x}");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty scratch folder for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("clarity_scan_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn read(p: PathBuf) -> String {
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    fn corrupt(path: impl Into<PathBuf>, reason: &str) -> Finding {
        Finding { path: path.into(), what: What::Corrupt { reason: reason.into(), status: FindingStatus::Listed } }
    }

    #[test]
    fn short_err_cuts_on_a_char_boundary() {
        // 3-byte characters: byte 80 falls inside one, which used to panic.
        let jp = "指定されたファイルが見つかりません。".repeat(6);
        let cut = short_err(&jp);
        assert_eq!(cut.chars().count(), 81);
        assert!(cut.ends_with('…') && jp.starts_with(cut.trim_end_matches('…')));
        assert_eq!(short_err(&"x".repeat(100)), format!("{}…", "x".repeat(80)));
        assert_eq!(short_err("  short\nsecond line"), "short");
    }

    #[test]
    fn dead_worker_ends_the_scan() {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::channel();
        tx.send(Msg::Corrupt(PathBuf::from("/d/a.png"), "Empty file".into())).unwrap();
        drop(tx); // the worker panicked before its Done
        let mut state = ScanState { running: true, rx: Some(rx), ..Default::default() };
        show(&ctx, &mut state);
        assert!(!state.running && state.rx.is_none() && state.finished_tick);
        assert!(matches!(state.outcome, Some(Outcome::Failed(_))), "{:?}", state.outcome);
        assert_eq!(state.findings.len(), 1, "what it found before it died stays listed");

        // A worker that finished normally hangs up right after its Done.
        let (tx, rx) = mpsc::channel();
        tx.send(Msg::Done(Summary::default())).unwrap();
        drop(tx);
        let mut state = ScanState { running: true, rx: Some(rx), ..Default::default() };
        show(&ctx, &mut state);
        assert!(!state.running && matches!(state.outcome, Some(Outcome::Finished(_))), "{:?}", state.outcome);
    }

    #[test]
    fn duplicate_takes_its_tags_along() {
        let dir = scratch("dupes");
        let dup = dir.join(DUPLICATES_FOLDER);
        std::fs::create_dir_all(&dup).unwrap();
        for (name, body) in [("b.png", "img"), ("b.txt", "b tags"), ("c.png", "img"), ("c.jpg", "jpg"), ("c.txt", "c tags")] {
            std::fs::write(dir.join(name), body).unwrap();
        }
        // Left by an earlier scan: the next "dup_1_b" must not land on it.
        std::fs::write(dup.join("dup_1_b.txt"), "earlier tags").unwrap();

        assert_eq!(move_into(&dir.join("b.png"), &dup, "dup_1_"), Moved::To(dup.join("dup_1_b (1).png")));
        assert!(!dir.join("b.png").exists() && !dir.join("b.txt").exists());
        assert_eq!(read(dup.join("dup_1_b (1).png")), "img");
        assert_eq!(read(dup.join("dup_1_b (1).txt")), "b tags");
        assert_eq!(read(dup.join("dup_1_b.txt")), "earlier tags");

        // c.jpg still uses c.txt, so the moved c.png gets a copy.
        assert_eq!(move_into(&dir.join("c.png"), &dup, "dup_2_"), Moved::To(dup.join("dup_2_c.png")));
        assert_eq!(read(dir.join("c.txt")), "c tags");
        assert_eq!(read(dup.join("dup_2_c.png")), "img");
        assert_eq!(read(dup.join("dup_2_c.txt")), "c tags");

        // Gone from under it: not moved, and why.
        assert!(matches!(move_into(&dir.join("missing.png"), &dup, "dup_3_"), Moved::Failed(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sidecar_in_use_sees_other_files() {
        let dir = scratch("users");
        for name in ["cat.png", "cat.txt", "Dog.JPG", "dog.txt", "solo.txt", "folder.txt", "latent.npz", "latent.txt"] {
            std::fs::write(dir.join(name), "x").unwrap();
        }
        std::fs::create_dir_all(dir.join("folder")).unwrap();
        assert!(sidecar_in_use(&dir.join("cat.txt")));
        assert!(sidecar_in_use(&dir.join("dog.txt"))); // any case, on every OS
        assert!(!sidecar_in_use(&dir.join("solo.txt")));
        assert!(!sidecar_in_use(&dir.join("latent.txt"))); // not a file that reads a .txt
        assert!(!sidecar_in_use(&dir.join("folder.txt"))); // a folder isn't a user
        assert!(sidecar_in_use(&dir.join("missing").join("x.txt"))); // can't tell: keep it
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn counts_read_like_numbers() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1204), "1,204");
        assert_eq!(count(1_234_567), "1,234,567");
        assert_eq!(plural(1, "image", "images"), "1 image");
        assert_eq!(plural(12_000, "image", "images"), "12,000 images");
    }

    #[test]
    fn reasons_in_a_few_words() {
        assert_eq!(short_reason("unexpected end of file"), "Truncated file");
        let jpeg = "Format error decoding Jpeg: I/O errors Not enough bytes, expected 162";
        assert_eq!(short_reason(jpeg), "Truncated file");
        assert_eq!(short_reason("Format error decoding Bmp: failed to fill whole buffer"), "Truncated file");
        assert_eq!(short_reason("Format error decoding Png: Invalid PNG signature."), "Bad header");
        assert_eq!(short_reason("Empty file"), "Empty file");
        assert_eq!(short_reason("Invalid dimensions"), "Bad dimensions");
        assert_eq!(short_reason("The decoder crashed on this file"), "Decoder crashed");
        assert_eq!(short_reason("Unreadable: Permission denied (os error 13)"), "Can't read file");
        assert_eq!(short_reason("Could not decode"), "Can't decode");
    }

    #[test]
    fn names_are_relative_to_the_scanned_folder() {
        let root = Path::new("/data/set");
        assert_eq!(rel_name(root, &root.join("a.png")), "a.png");
        let sub = Path::new("sub").join("b.jpg");
        assert_eq!(rel_name(root, &root.join(&sub)), sub.display().to_string());
        assert_eq!(rel_name(root, Path::new("/elsewhere/c.png")), "c.png");
        assert_eq!(rel_name(Path::new(""), Path::new("/x/d.png")), "d.png");
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(name_parts(root, &root.join(&sub)), (format!("sub{sep}"), "b.jpg".to_string()));
        assert_eq!(name_parts(root, &root.join("a.png")), (String::new(), "a.png".to_string()));
    }

    #[test]
    fn a_deleted_file_keeps_its_row() {
        let dir = scratch("delete_rows");
        std::fs::write(dir.join("bad.png"), "x").unwrap();
        std::fs::write(dir.join("bad.txt"), "tags").unwrap();
        let unreachable = dir.join("unplugged").join("gone.png");
        let mut state = ScanState { root: dir.clone(), ..Default::default() };
        for path in [dir.join("bad.png"), unreachable.clone(), dir.join("second.png")] {
            state.findings.push(corrupt(path, "Empty file"));
        }
        let status = |state: &ScanState, i: usize| match &state.findings[i].what {
            What::Corrupt { status, .. } => status.clone(),
            other => panic!("{other:?}"),
        };
        delete_finding(&mut state, 0);
        assert_eq!(status(&state, 0), FindingStatus::Deleted);
        assert!(!dir.join("bad.png").exists() && !dir.join("bad.txt").exists(), "the file and its tags go");
        assert_eq!(state.deleted, [dir.join("bad.png")]);
        // A second click on the same spot does nothing: the row didn't move.
        delete_finding(&mut state, 0);
        assert_eq!(state.findings.len(), 3);
        assert_eq!(state.deleted.len(), 1);
        // Its folder is unreachable: that's an error, shown on its row.
        delete_finding(&mut state, 1);
        assert!(matches!(status(&state, 1), FindingStatus::DeleteFailed(_)), "{:?}", status(&state, 1));
        assert_eq!(status(&state, 2), FindingStatus::Listed);
        // Already gone from a folder that's there: it counts as deleted.
        delete_finding(&mut state, 2);
        assert_eq!(status(&state, 2), FindingStatus::Deleted);
        // Tags another file still uses stay.
        std::fs::write(dir.join("cat.png"), "x").unwrap();
        std::fs::write(dir.join("cat.jpg"), "x").unwrap();
        std::fs::write(dir.join("cat.txt"), "tags").unwrap();
        state.findings.push(corrupt(dir.join("cat.png"), "Empty file"));
        delete_finding(&mut state, 3);
        assert_eq!(status(&state, 3), FindingStatus::Deleted);
        assert!(!dir.join("cat.png").exists() && dir.join("cat.txt").exists());
        // A duplicate's row has nothing to delete.
        let kept = dir.join("kept.png");
        let what = What::Duplicate { kept, moved: Moved::Failed("x".into()) };
        state.findings.push(Finding { path: dir.join("d.png"), what });
        delete_finding(&mut state, 4);
        assert_eq!(state.deleted.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rows_keep_their_groups() {
        let mut state = ScanState::default();
        let kept = PathBuf::from("k.png");
        state.push_finding("d1.png".into(), What::Duplicate { kept: kept.clone(), moved: Moved::To("x".into()) });
        state.push_finding("c1.png".into(), What::Corrupt { reason: "r".into(), status: FindingStatus::Listed });
        state.push_finding("u1.png".into(), What::Unchecked { reason: "r".into() });
        state.push_finding("c2.png".into(), What::Corrupt { reason: "r".into(), status: FindingStatus::Listed });
        let order: Vec<_> = state.findings.iter().map(|f| f.path.display().to_string()).collect();
        assert_eq!(order, ["c1.png", "c2.png", "u1.png", "d1.png"]);
    }

    #[test]
    fn an_unchecked_image_that_moved_is_listed_once() {
        let (tx, rx) = mpsc::channel();
        let path = PathBuf::from("/d/big.tif");
        tx.send(Msg::Unchecked(path.clone(), "Limits are exceeded".into())).unwrap();
        let moved = Moved::To(PathBuf::from("/d/duplicates/dup_1_big.tif"));
        tx.send(Msg::Duplicate { path: path.clone(), kept: PathBuf::from("/d/a.tif"), moved }).unwrap();
        let mut state = ScanState { running: true, rx: Some(rx), ..Default::default() };
        state.drain(&egui::Context::default());
        assert_eq!(state.findings.len(), 1);
        assert!(matches!(state.findings[0].what, What::Duplicate { .. }));
    }

    #[test]
    fn start_explains_a_missing_folder() {
        let ctx = egui::Context::default();
        let mut state = ScanState::default();
        start_scan(&mut state, &ctx);
        assert!(!state.running && state.folder_error.is_some());
        state.input_dir = "/no/such/folder/for/clarity".into();
        start_scan(&mut state, &ctx);
        assert!(!state.running && state.rx.is_none());
        assert!(state.folder_error.as_deref().is_some_and(|e| e.contains("Can't find")));
        assert!(state.findings.is_empty() && state.outcome.is_none());
    }

    #[test]
    fn worker_reports_its_steps_and_findings() {
        let dir = scratch("worker");
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 120, 40]));
        img.save(dir.join("one.png")).unwrap();
        img.save(dir.join("copy of one.png")).unwrap();
        std::fs::write(dir.join("broken.png"), b"not a png at all").unwrap();
        std::fs::write(dir.join("clip.mp4"), b"video").unwrap();
        std::fs::write(dir.join("loop.gif"), b"gif").unwrap();
        let (tx, rx) = mpsc::channel();
        run_scan(dir.clone(), true, tx, Arc::new(AtomicBool::new(false)), egui::Context::default());
        let msgs: Vec<Msg> = rx.try_iter().collect();
        let stages: Vec<(Stage, usize, usize)> = msgs
            .iter()
            .filter_map(|m| match m {
                Msg::Progress { stage, at, total } => Some((*stage, *at, *total)),
                _ => None,
            })
            .collect();
        assert_eq!(
            stages,
            [
                (Stage::Checking, 1, 3),
                (Stage::Checking, 2, 3),
                (Stage::Checking, 3, 3),
                (Stage::Hashing, 1, 2),
                (Stage::Hashing, 2, 2),
            ]
        );
        let corrupt: Vec<&PathBuf> = msgs
            .iter()
            .filter_map(|m| if let Msg::Corrupt(p, _) = m { Some(p) } else { None })
            .collect();
        assert_eq!(corrupt, [&dir.join("broken.png")]);
        // "copy of one.png" sorts first, so it stays; "one.png" moves.
        let dupes: Vec<_> = msgs
            .iter()
            .filter_map(|m| if let Msg::Duplicate { path, kept, moved } = m { Some((path, kept, moved)) } else { None })
            .collect();
        let moved_to = dir.join(DUPLICATES_FOLDER).join("dup_1_one.png");
        assert_eq!(dupes, [(&dir.join("one.png"), &dir.join("copy of one.png"), &Moved::To(moved_to.clone()))]);
        assert!(moved_to.is_file());
        let Some(Msg::Done(sum)) = msgs.last() else { panic!("ends with Done") };
        assert_eq!((sum.scanned, sum.corrupted, sum.unchecked), (3, 1, 0));
        assert_eq!((sum.duplicates_found, sum.duplicates_moved), (1, 1));
        assert_eq!((sum.skipped, sum.skipped_gifs), (2, 1));
        assert!(sum.dupes_checked && !sum.cancelled && sum.error.is_none());

        // Through the window's state: the counts and the rows.
        let (tx, rx) = mpsc::channel();
        for m in msgs {
            tx.send(m).unwrap();
        }
        drop(tx);
        let mut state = ScanState { running: true, rx: Some(rx), root: dir.clone(), ..Default::default() };
        show(&egui::Context::default(), &mut state);
        assert!(matches!(&state.outcome, Some(Outcome::Finished(s)) if s.corrupted == 1));
        assert_eq!(state.counts(), [Some(2), Some(1), Some(1)]);
        let rows: Vec<String> = state.findings.iter().map(|f| row_status(f).0).collect();
        assert_eq!(rows, ["Corrupt \u{b7} Bad header", "Duplicate \u{b7} Moved"]);
        let text = list_text(&state);
        assert!(text.contains("Verified 2, duplicates 1, corrupt 1"), "{text}");
        let line = format!("Duplicate \u{b7} Moved: {} (now {})", dir.join("one.png").display(), moved_to.display());
        assert!(text.contains(&line), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn counts_while_scanning_and_after() {
        let mut state = ScanState { running: true, ..Default::default() };
        assert_eq!(state.counts(), [None, None, None], "nothing known while gathering");
        (state.stage, state.at, state.total) = (Stage::Checking, 11, 40);
        state.findings.push(corrupt("a.png", "Empty file"));
        assert_eq!(state.counts(), [Some(9), None, Some(1)], "10 done, 1 of them corrupt");
        // Cancelled there: the 11th was checked before it stopped, as its note says.
        state.outcome = Some(Outcome::Cancelled);
        assert_eq!(state.counts(), [Some(10), None, Some(1)]);
        assert!(ending_note(&state).unwrap().0.contains("after checking 11 of 40 images"));
        state.outcome = None;
        (state.stage, state.at, state.total) = (Stage::Hashing, 3, 38);
        assert_eq!(state.counts(), [Some(38), None, Some(1)]);
        state.running = false;
        state.outcome = Some(Outcome::Finished(Summary {
            scanned: 40,
            corrupted: 1,
            unchecked: 1,
            dupes_checked: true,
            ..Default::default()
        }));
        assert_eq!(state.counts(), [Some(38), Some(0), Some(1)]);
        // Duplicates weren't looked for: no number, not a zero.
        state.outcome = Some(Outcome::Finished(Summary { scanned: 40, corrupted: 1, ..Default::default() }));
        assert_eq!(state.counts()[1], None);
    }

    #[test]
    fn how_a_scan_ended() {
        let mut state = ScanState { outcome: Some(Outcome::Cancelled), ..Default::default() };
        (state.stage, state.at, state.total) = (Stage::Checking, 2, 5);
        assert_eq!(ending_note(&state).unwrap().0, "Scan cancelled after checking 2 of 5 images. Nothing was moved.");
        state.at = 5;
        assert_eq!(
            ending_note(&state).unwrap().0,
            "Scan cancelled after checking 5 images, before looking for duplicates. Nothing was moved."
        );
        state.stage = Stage::Gathering;
        assert_eq!(ending_note(&state).unwrap().0, "Scan cancelled before checking any images. Nothing was moved.");
        state.outcome = Some(Outcome::Finished(Summary::default()));
        assert!(ending_note(&state).is_none());
    }

    #[test]
    fn row_status_words() {
        let kept = PathBuf::from("/d/keep.png");
        let row = |what| Finding { path: PathBuf::from("/d/x.png"), what };
        let words = |f: Finding| row_status(&f).0;
        assert_eq!(words(corrupt("/d/x.png", "unexpected end of file")), "Corrupt \u{b7} Truncated file");
        let deleted = What::Corrupt { reason: "r".into(), status: FindingStatus::Deleted };
        assert_eq!(words(row(deleted)), "Deleted");
        let failed = What::Corrupt { reason: "r".into(), status: FindingStatus::DeleteFailed("denied".into()) };
        assert_eq!(words(row(failed)), "Couldn't delete");
        let big = What::Unchecked { reason: "Limits are exceeded".into() };
        assert_eq!(words(row(big)), "Not checked \u{b7} Too large");
        assert_eq!(words(row(What::Unchecked { reason: "unsupported".into() })), "Not checked \u{b7} Unsupported");
        let locked = What::Unchecked { reason: "Can't read it: The process cannot access the file".into() };
        assert_eq!(words(row(locked)), "Not checked \u{b7} Can't read");
        let stayed = What::Corrupt { reason: "r".into(), status: FindingStatus::DeletedTagsStayed("denied".into()) };
        assert_eq!(words(row(stayed)), "Deleted \u{b7} tags stayed");
        let to = PathBuf::from("/d/duplicates/dup_1_x.png");
        let dup = |moved| What::Duplicate { kept: kept.clone(), moved };
        assert_eq!(words(row(dup(Moved::To(to.clone())))), "Duplicate \u{b7} Moved");
        assert_eq!(words(row(dup(Moved::WithoutTags(to, "denied".into())))), "Duplicate \u{b7} Moved, tags stayed");
        assert_eq!(words(row(dup(Moved::Failed("denied".into())))), "Duplicate \u{b7} Couldn't move");
    }

    #[test]
    fn too_big_to_decode_is_not_corrupt() {
        fn crc32(bytes: &[u8]) -> u32 {
            let mut c = !0u32;
            for &b in bytes {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                }
            }
            !c
        }
        fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let start = out.len();
            out.extend_from_slice(kind);
            out.extend_from_slice(data);
            let crc = crc32(&out[start..]);
            out.extend_from_slice(&crc.to_be_bytes());
        }
        // A well-formed PNG of 60000 × 60000 RGBA: past the decoder's memory
        // limit, so it can't be checked here — but nothing says it's damaged.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = 60_000u32.to_be_bytes().repeat(2);
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        chunk(&mut png, b"IEND", &[]);
        let dir = scratch("unchecked");
        let p = dir.join("huge.png");
        std::fs::write(&p, &png).unwrap();
        assert!(matches!(validate_image(&p), Err(Problem::Unchecked(_))), "{:?}", validate_image(&p));
        // Cut off, it is damaged.
        std::fs::write(&p, &png[..png.len() / 2]).unwrap();
        assert!(matches!(validate_image(&p), Err(Problem::Corrupt(_))), "{:?}", validate_image(&p));
        // A file that can't be read at all says nothing about the image.
        let gone = dir.join("gone.png");
        assert!(matches!(validate_image(&gone), Err(Problem::Unchecked(r)) if r.starts_with("Can't read")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn walk_does_not_follow_symlinks() {
        let dir = scratch("links");
        std::fs::create_dir_all(dir.join("real")).unwrap();
        std::fs::write(dir.join("real").join("img.png"), "img").unwrap();
        std::os::unix::fs::symlink(dir.join("real"), dir.join("linked")).unwrap();
        std::os::unix::fs::symlink(dir.join("real").join("img.png"), dir.join("a_link.png")).unwrap();
        assert_eq!(gather_files(&dir), [dir.join("real").join("img.png")]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
