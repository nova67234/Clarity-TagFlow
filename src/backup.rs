//! Backup — the "Create Backup" dialog from the top bar.
//!
//! A Rust/egui port of terminus2's `Backup.java`. Zips the loaded folder's media
//! (images + a fixed set of video types) together with their `.txt` sidecars into
//! `<folder>/backups/<name>_<date>.zip`. Images are decode-validated first so a
//! corrupt file is skipped rather than archived. The archive can optionally be
//! AES-256 encrypted with a password.
//!
//! The zip work runs on a background thread (folders can be large), reporting
//! progress through a shared struct so the egui frame paints a live bar without
//! blocking — same pattern as the AI Model Manager.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, RichText};
use zip::write::SimpleFileOptions;
use zip::{AesMode, CompressionMethod};

use crate::scan::Pill;
use crate::theme::*;

/// Green flash on the Create button on click (matches Tag Manager settings Save).
pub const FLASH_GREEN: Color32 = Color32::from_rgb(46, 160, 67);
/// Red flash on the Cancel button on click (matches Tag Manager settings Cancel).
pub const FLASH_RED: Color32 = Color32::from_rgb(200, 55, 55);
/// How long a button flash lingers before its action fires.
pub const FLASH: Duration = Duration::from_millis(450);

/// Shortest password Create accepts for an encrypted archive. Zip AES derives
/// its key with low-iteration PBKDF2, so very short passwords give AES-256 in
/// name only.
const MIN_PASSWORD_LEN: usize = 8;

/// Fixed dialog body width. Everything inside uses finite widths derived from
/// this — NEVER `f32::INFINITY` or `ui.available_width()` — because in an
/// auto-sized egui window the available width is infinite, which leaks into the
/// persisted `last_content_size` as `inf` and reloads as a NaN rect that panics.
const DIALOG_WIDTH: f32 = 380.0;

/// GIF is animated — decode-validating only its first frame is pointless, so it's
/// included as-is like videos rather than corruption-scanned as a still image.
const SKIP_VALIDATION_EXTS: &[&str] = &["gif"];

/// Characters Windows (and the others) forbid in a file name.
fn has_illegal_chars(name: &str) -> bool {
    name.chars()
        .any(|c| matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'))
}

// ---------------------------------------------------------------------------
// Background worker ↔ UI shared state
// ---------------------------------------------------------------------------

/// Written by the backup thread, read by the UI each frame.
#[derive(Default)]
struct Progress {
    label: Mutex<String>,
    current: AtomicUsize,
    total: AtomicUsize, // 0 until the scan finishes (UI shows a spinner until then)
    done: AtomicBool,
    cancel: AtomicBool,
    outcome: Mutex<Option<Outcome>>,
}

impl Progress {
    fn set_label(&self, s: String) {
        *self.label.lock().unwrap() = s;
    }
    fn finish(&self, outcome: Outcome) {
        *self.outcome.lock().unwrap() = Some(outcome);
        self.done.store(true, Relaxed);
    }
}

/// What a finished (or aborted) backup produced.
enum Outcome {
    Success(Summary),
    NoFiles,
    Cancelled,
    Error(String),
}

struct Summary {
    zip_name: String,
    written: usize,
    /// Files that couldn't be read while archiving — left out, not written.
    failed: usize,
    total_selected: usize,
    skipped_corrupt: usize,
    encrypted: bool,
}

// ---------------------------------------------------------------------------
// Dialog state (lives on ViewerApp)
// ---------------------------------------------------------------------------

enum Phase {
    /// Collecting the name + optional password.
    Options,
    /// Zipping in the background.
    Running(Arc<Progress>),
    /// Finished — showing the result.
    Done(Outcome),
}

pub struct BackupState {
    pub open: bool,
    phase: Phase,

    // The folder being backed up and the media to include, captured on open.
    source: PathBuf,
    files: Vec<PathBuf>,

    // Options form.
    name: String,
    encrypt: bool,
    password: String,
    confirm: String,
    form_error: Option<String>,

    // Click-flash timers for the footer buttons (green = Create, red = Cancel).
    // The action fires once the flash has shown for `FLASH`.
    create_flash: Option<Instant>,
    cancel_flash: Option<Instant>,
}

impl Default for BackupState {
    fn default() -> Self {
        Self {
            open: false,
            phase: Phase::Options,
            source: PathBuf::new(),
            files: Vec::new(),
            name: String::new(),
            encrypt: false,
            password: String::new(),
            confirm: String::new(),
            form_error: None,
            create_flash: None,
            cancel_flash: None,
        }
    }
}

impl BackupState {
    /// Open the dialog to back up `files` rooted at `source`. Resets the form.
    pub fn open(&mut self, source: PathBuf, files: Vec<PathBuf>) {
        self.source = source;
        self.files = files;
        self.name.clear();
        self.encrypt = false;
        self.password.clear();
        self.confirm.clear();
        self.form_error = None;
        self.create_flash = None;
        self.cancel_flash = None;
        self.open = true;

        // Surface the "nothing to do" cases immediately, like the Java version.
        if self.source.as_os_str().is_empty() || !self.source.is_dir() {
            self.phase = Phase::Done(Outcome::Error("Source directory not found.".into()));
        } else if self.files.is_empty() {
            self.phase = Phase::Done(Outcome::NoFiles);
        } else {
            self.phase = Phase::Options;
        }
    }

    /// Draw the dialog (when open) and drive any running backup. Call every frame.
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }

        // Promote a finished background job into the Done phase.
        if let Phase::Running(prog) = &self.phase {
            if prog.done.load(Relaxed) {
                let outcome = prog
                    .outcome
                    .lock()
                    .unwrap()
                    .take()
                    .unwrap_or(Outcome::Error("Backup ended unexpectedly.".into()));
                self.phase = Phase::Done(outcome);
            } else {
                // Keep the bar animating while work is in flight.
                ctx.request_repaint();
            }
        }

        // The window title bar carries the (single) dialog name, phase-aware so
        // each stage reads correctly.
        let window_title = match &self.phase {
            Phase::Options => "New Backup",
            Phase::Running(_) => "Creating Backup",
            Phase::Done(Outcome::Success(_)) => "Backup Complete",
            Phase::Done(Outcome::NoFiles) => "Backup Failed",
            Phase::Done(Outcome::Cancelled) => "Cancelled",
            Phase::Done(Outcome::Error(_)) => "Error",
        };

        let running = match &self.phase {
            Phase::Running(p) => Some(p.clone()),
            _ => None,
        };

        let mut close_requested = false;

        // No window close (X) button — the in-dialog Cancel/Close buttons handle
        // dismissal (and a running backup must be cancelled, not orphaned).
        // NOTE: keep this exactly like the Settings window — anchored, auto-sized,
        // with the width set ONLY via `ui.set_width` inside. Do NOT add width
        // builder methods or read `ui.available_width()` for a layout extent: in
        // an auto-sized window that width is infinite, which stores a NaN rect and
        // panics on the next frame.
        let window = egui::Window::new(window_title)
            .id(egui::Id::new("backup_dialog"))
            .title_bar(false) // custom header inside (matches the Civitai popup)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .frame(crate::settings::fresh_window_frame());

        // Sections, rows and fields in the Find Issues look, as in Settings.
        crate::settings::fresh_look(|| window.show(ctx, |ui| {
            // Fix BOTH min and max width. `set_width` alone leaves the max at the
            // auto-sized window's infinite available width, so `available_width()`
            // stays infinite — and the `desired_width(INFINITY)` fields below then
            // resolve to inf, which gets persisted and reloads as a NaN rect that
            // panics on the next launch. Capping max keeps every width finite.
            ui.set_width(380.0);
            ui.set_max_width(380.0);

            // Custom title row (Find Issues-style): a muted backup icon, the
            // phase title in bold, close.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = TITLE_GAP;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/backup.svg"))
                        .fit_to_exact_size(egui::vec2(TITLE_ICON, TITLE_ICON))
                        .tint(crate::theme::icon_tint(MUTED())),
                );
                ui.label(RichText::new(window_title).color(TEXT()).font(crate::settings::bold(19.0)));
                // No close while a backup runs — it must be cancelled, not orphaned.
                if running.is_none() {
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
                            close_requested = true;
                        }
                    });
                }
            });
            ui.add_space(5.0);

            if let Some(prog) = &running {
                Self::running_body(ui, prog);
            } else if matches!(self.phase, Phase::Options) {
                self.options_body(ui);
            } else if let Phase::Done(outcome) = &self.phase {
                close_requested |= done_body(ui, outcome);
            }
        }));

        // Dismissed via an in-dialog Cancel/Close button.
        if close_requested {
            self.open = false;
        }
    }

    fn options_body(&mut self, ui: &mut egui::Ui) {
        header(ui, "Archive this folder's media and tags into a .zip.");

        crate::settings::section(ui, "Name", |ui| {
            text_field(ui, &mut self.name, "e.g. my_dataset", false);
            hint(ui, "Saved to a \"backups\" folder inside the source, dated automatically.");
        });

        crate::settings::section(ui, "Security", |ui| {
            crate::settings::row(
                ui,
                "Encrypt with password",
                Some("AES-256, at least 8 characters — the password can't be recovered if lost."),
                |ui| {
                    crate::settings::switch(ui, &mut self.encrypt);
                },
            );

            if self.encrypt {
                crate::settings::row_sep(ui);
                ui.label(RichText::new("Password").color(MUTED()).size(12.0));
                ui.add_space(4.0);
                text_field(ui, &mut self.password, "", true);
                ui.add_space(8.0);
                ui.label(RichText::new("Confirm password").color(MUTED()).size(12.0));
                ui.add_space(4.0);
                text_field(ui, &mut self.confirm, "", true);
            }
        });

        if let Some(err) = &self.form_error {
            ui.add_space(6.0);
            ui.label(RichText::new(err).color(Color32::from_rgb(235, 110, 110)).size(12.0));
        }

        ui.add_space(20.0);

        // Buttons flash a colour on click, then fire their action once the flash
        // has lingered for `FLASH` (green = Create/start, red = Cancel/close) —
        // matching the Tag Manager settings Save/Cancel behaviour.
        let create_flashing = self.create_flash.is_some();
        let cancel_flashing = self.cancel_flash.is_some();
        let start_backup = self.create_flash.is_some_and(|t| t.elapsed() >= FLASH);
        let do_cancel = self.cancel_flash.is_some_and(|t| t.elapsed() >= FLASH);

        footer_row(ui, |ui| {
            let create = footer_button(ui, "Create", Pill::Accent, create_flashing.then_some(FLASH_GREEN));
            if create.clicked() && self.create_flash.is_none() && !cancel_flashing {
                self.create_flash = Some(Instant::now());
            }
            ui.add_space(8.0);
            let cancel = footer_button(ui, "Cancel", Pill::Outline, cancel_flashing.then_some(FLASH_RED));
            if cancel.clicked() && self.cancel_flash.is_none() && !create_flashing {
                self.cancel_flash = Some(Instant::now());
            }
        });

        if create_flashing || cancel_flashing {
            ui.ctx().request_repaint(); // keep the flash animating toward its action
        }

        // Fire the deferred action once its flash has shown long enough.
        if start_backup {
            self.create_flash = None;
            self.try_start();
        } else if do_cancel {
            self.cancel_flash = None;
            self.open = false;
        }
    }

    /// Validate the form and, if good, spawn the backup thread.
    fn try_start(&mut self) {
        let name = self.name.trim().to_string();
        if name.is_empty() {
            self.form_error = Some("Please enter a backup name.".into());
            return;
        }
        if has_illegal_chars(&name) {
            self.form_error = Some("Name contains illegal characters (e.g. / \\ : ? *).".into());
            return;
        }

        let password = if self.encrypt {
            if self.password.is_empty() {
                self.form_error = Some(
                    "Enter a password, or uncheck \"Encrypt with password\".".into(),
                );
                return;
            }
            if self.password.chars().count() < MIN_PASSWORD_LEN {
                self.form_error = Some(format!(
                    "Password must be at least {MIN_PASSWORD_LEN} characters."
                ));
                return;
            }
            if self.password != self.confirm {
                self.form_error = Some("The password and confirmation don't match.".into());
                return;
            }
            Some(self.password.clone())
        } else {
            None
        };

        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        // The worker adds ".zip" (or " (2).zip"… when a same-name backup from
        // today is already there — see `create_archive`).
        let zip_base = format!("{name}_{date}");

        let prog = Arc::new(Progress::default());
        *prog.label.lock().unwrap() = "Scanning files…".to_string();

        let worker_prog = prog.clone();
        let source = self.source.clone();
        let files = self.files.clone();
        std::thread::spawn(move || {
            // Catch any panic in the backup so it surfaces as a clean error in the
            // dialog instead of silently killing the worker (which would leave the
            // UI spinning forever) or aborting the process.
            let panic_prog = worker_prog.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_backup(source, files, zip_base, password, &worker_prog);
            }));
            if result.is_err() && !panic_prog.done.load(Relaxed) {
                panic_prog.finish(Outcome::Error(
                    "The backup failed unexpectedly (internal error).".into(),
                ));
            }
        });

        // Clear the entered password from the form fields.
        self.password.clear();
        self.confirm.clear();
        self.form_error = None;
        self.phase = Phase::Running(prog);
    }

    fn running_body(ui: &mut egui::Ui, prog: &Progress) {
        header(ui, "Compressing and writing the archive…");

        crate::settings::section(ui, "Progress", |ui| {
            let label = prog.label.lock().unwrap().clone();
            let total = prog.total.load(Relaxed);
            let cur = prog.current.load(Relaxed);
            // The label, the count on the right, and Find Issues' thin bar.
            ui.horizontal(|ui| {
                ui.label(RichText::new(label).color(MUTED()).size(12.0));
                if total > 0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{cur} / {total}")).color(TEXT()).size(12.0));
                    });
                }
            });
            ui.add_space(10.0);
            let frac = (total > 0).then(|| (cur as f32 / total as f32).clamp(0.0, 1.0));
            crate::scan::progress_bar(ui, frac, egui::Id::new("backup_progress"));
        });

        ui.add_space(20.0);
        footer_row(ui, |ui| {
            let cancelling = prog.cancel.load(Relaxed);
            let btn_label = if cancelling { "Cancelling…" } else { "Cancel" };
            if crate::scan::pill(ui, btn_label, Pill::Outline, None, !cancelling).clicked() && !cancelling {
                prog.cancel.store(true, Relaxed);
                prog.set_label("Cancelling…".into());
            }
        });
    }
}

/// Draw the result screen. Returns `true` if the Close button was clicked.
fn done_body(ui: &mut egui::Ui, outcome: &Outcome) -> bool {
    match outcome {
        Outcome::Success(s) => {
            header(ui, "Your archive was created successfully.");
            crate::settings::section(ui, "Summary", |ui| {
                kv(ui, "Location", &s.zip_name);
                kv(ui, "Files written", &format!("{} / {}", s.written, s.total_selected));
                if s.encrypted {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/encrypted.svg"))
                                .fit_to_exact_size(egui::vec2(14.0, 14.0))
                                .tint(crate::theme::icon_tint(TEXT())),
                        );
                        ui.label(
                            RichText::new("Password-protected (AES-256).").color(TEXT()).size(12.0),
                        );
                    });
                }
                let warnings = [
                    (s.skipped_corrupt > 0)
                        .then(|| format!("Skipped {} potentially corrupt image(s).", s.skipped_corrupt)),
                    (s.failed > 0)
                        .then(|| format!("Couldn't read {} file(s) — they're not in the archive.", s.failed)),
                ];
                for warning in warnings.into_iter().flatten() {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/warning.svg"))
                                .fit_to_exact_size(egui::vec2(14.0, 14.0))
                                // The app's warning orange (matches the Civitai
                                // download-failed glyph).
                                .tint(Color32::from_rgb(235, 150, 45)),
                        );
                        ui.label(RichText::new(warning).color(TEXT()).size(12.0));
                    });
                }
            });
        }
        Outcome::NoFiles => {
            header(ui, "Nothing was archived.");
            crate::settings::section(ui, "Details", |ui| {
                ui.label(RichText::new("No valid files found to back up.").color(MUTED()).size(13.0));
            });
        }
        Outcome::Cancelled => {
            header(ui, "The backup was stopped.");
            crate::settings::section(ui, "Details", |ui| {
                ui.label(RichText::new("Backup operation canceled.").color(MUTED()).size(13.0));
            });
        }
        Outcome::Error(e) => {
            header(ui, "Something went wrong during the backup.");
            crate::settings::section(ui, "Details", |ui| {
                ui.label(
                    RichText::new(e)
                        .color(Color32::from_rgb(235, 110, 110))
                        .size(12.0),
                );
            });
        }
    }

    ui.add_space(20.0);
    let mut close = false;
    footer_row(ui, |ui| {
        if accent_button(ui, "Close").clicked() {
            close = true;
        }
    });
    close
}

// ---------------------------------------------------------------------------
// Shared layout helpers (mirroring the Settings window's section/hint style)
// ---------------------------------------------------------------------------

/// Dialog sub-header: a muted one-line description under the window title bar
/// (which already shows the dialog name, so there's no title duplicated here).
/// Lined up under the title's text, past its icon (as in Find Issues).
fn header(ui: &mut egui::Ui, subtitle: &str) {
    ui.horizontal(|ui| {
        ui.add_space(TITLE_ICON + TITLE_GAP);
        ui.label(RichText::new(subtitle).color(MUTED()).size(12.5));
    });
    ui.add_space(18.0);
}

/// The title row's icon size and the gap after it.
const TITLE_ICON: f32 = 17.0;
const TITLE_GAP: f32 = 10.0;

/// A small muted explanatory line, shown under a control.
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(text).color(MUTED()).size(12.0));
}

/// A label : value row, used in the result summary.
fn kv(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{key}:")).color(MUTED()).size(12.0));
        ui.label(RichText::new(value).color(TEXT()).size(12.0));
    });
}

/// A single-line text field: Find Issues' sunk pill, the full width.
fn text_field(ui: &mut egui::Ui, value: &mut String, hint: &str, password: bool) {
    crate::settings::pill_field(
        ui,
        f32::INFINITY,
        egui::TextEdit::singleline(value).password(password).hint_text(hint),
    );
}

/// The right-aligned footer button row. Allocated at a FIXED 380×32 — a bare
/// `with_layout(right_to_left)` here would claim all the height left in the
/// window, so once the window had grown (e.g. the password fields were shown)
/// the auto-sizer saw that leftover space as "content" and never shrank back,
/// leaving a big dead gap above the buttons.
fn footer_row(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    ui.allocate_ui_with_layout(
        egui::vec2(DIALOG_WIDTH, 36.0),
        egui::Layout::right_to_left(egui::Align::Center),
        add,
    );
}

/// A footer button: Find Issues' pill. Pass `flash = Some(color)` to fill
/// it for a click-flash.
fn footer_button(ui: &mut egui::Ui, label: &str, kind: Pill, flash: Option<Color32>) -> egui::Response {
    crate::scan::pill(ui, label, kind, flash, true)
}

/// The result screen's Close button — the accent pill, no flash.
fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    footer_button(ui, label, Pill::Accent, None)
}

// ---------------------------------------------------------------------------
// The actual backup work (background thread)
// ---------------------------------------------------------------------------

fn ext_lower(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// Header-only decode check — true if the image's dimensions read back as valid.
/// Mirrors Java's `reader.getWidth/getHeight > 0` without a full pixel decode.
/// Only a file positively found damaged is false: a backup must never drop a
/// file (and its tags) just because this check can't read its format.
fn is_image_valid(path: &Path) -> bool {
    // HDR's `#?RGBE` signature variant trips `image`'s header reader, so validate
    // it through our own decoder instead of dropping good files from the backup.
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("hdr")) {
        return crate::image_cache::decode_hdr(path).is_some();
    }
    // SVG: `image` can't parse vectors — validate through the rasterizer.
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("svg")) {
        return crate::image_cache::decode_svg(path, 64).is_some();
    }
    // The decoder is picked by content, not extension, so a PNG saved as
    // `.jpg` isn't mistaken for a broken JPEG.
    let Ok(reader) = image::ImageReader::open(path).and_then(|r| r.with_guessed_format()) else {
        return true; // can't be opened right now — archiving it reports that
    };
    match reader.into_dimensions() {
        Ok((w, h)) => w > 0 && h > 0,
        // A format `image` doesn't decode (or one past its limits) isn't corrupt.
        Err(image::ImageError::Unsupported(_) | image::ImageError::Limits(_)) => true,
        Err(_) => false,
    }
}

/// Swap a path's extension for `.txt` (its tag sidecar).
fn sidecar_txt(path: &Path) -> PathBuf {
    path.with_extension("txt")
}

/// `file` relative to `root`, using forward slashes — `None` when it isn't
/// inside `root` (or the rest isn't plain names, e.g. a `..`), so an entry
/// name built from it can never be absolute or point outside the archive.
fn safe_rel_path(root: &Path, file: &Path) -> Option<String> {
    let rel = file.strip_prefix(root).ok()?;
    let plain = rel.components().all(|c| matches!(c, std::path::Component::Normal(_)));
    (plain && rel.file_name().is_some()).then(|| rel.to_string_lossy().replace('\\', "/"))
}

/// The archive entry name for each file, in order, without repeats: a sidecar
/// shared by two images (`cat.png` + `cat.jpg` both use `cat.txt`) goes in
/// once — a second "cat.txt" made the zip writer fail the whole backup.
/// Names are always relative: a file from outside `root` (dropped in from
/// another folder) goes under `external/<its folder's name>/`, so an image
/// and its `.txt` stay side by side; another folder of the same name becomes
/// `<name> (2)`.
fn archive_entries(root: &Path, files: &[PathBuf]) -> Vec<(PathBuf, String)> {
    let mut seen: HashSet<&Path> = HashSet::new();
    let mut names: HashSet<String> = HashSet::new();
    // Each outside folder's name under external/, and those names lowercased
    // (folders differing only in case would merge when extracted on Windows).
    let mut folders: HashMap<&Path, String> = HashMap::new();
    let mut folder_keys: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for file in files {
        if !seen.insert(file.as_path()) {
            continue;
        }
        let name = safe_rel_path(root, file).unwrap_or_else(|| {
            let dir = file.parent().unwrap_or(Path::new(""));
            let folder = folders.entry(dir).or_insert_with(|| {
                let base = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "root".into()); // a drive / filesystem root
                let mut label = base.clone();
                let mut n = 1;
                while !folder_keys.insert(label.to_lowercase()) {
                    n += 1;
                    label = format!("{base} ({n})");
                }
                label
            });
            let file_name = file.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
            format!("external/{folder}/{file_name}")
        });
        // Distinct files can still meet on one name in odd cases (a `\` in a
        // Linux file name reads as a folder here); number the later ones.
        let dot = name.rfind('.').filter(|&i| i > name.rfind('/').map_or(0, |s| s + 1));
        let mut unique = name.clone();
        let mut n = 1;
        while !names.insert(unique.clone()) {
            n += 1;
            unique = match dot {
                Some(i) => format!("{} ({n}){}", &name[..i], &name[i..]),
                None => format!("{name} ({n})"),
            };
        }
        out.push((file.clone(), unique));
    }
    out
}

/// Create a new, empty `<dir>/<base>.zip`, or `<base> (2).zip`, `(3)`… when
/// that name is taken. `create_new` never opens (and truncates) an existing
/// file, so a second same-name backup on the same day can't overwrite the
/// first — and removing a failed or cancelled archive only ever removes this
/// run's own file.
fn create_archive(dir: &Path, base: &str) -> std::io::Result<(PathBuf, std::fs::File)> {
    let mut n = 1;
    loop {
        let name = if n == 1 { format!("{base}.zip") } else { format!("{base} ({n}).zip") };
        let path = dir.join(name);
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && n < 10_000 => n += 1,
            Err(e) => return Err(e),
        }
    }
}

fn run_backup(
    source: PathBuf,
    media: Vec<PathBuf>,
    zip_base: String,
    password: Option<String>,
    prog: &Progress,
) {
    // 1. Scan + validate, collecting valid media and their existing sidecars.
    let mut valid: Vec<PathBuf> = Vec::new();
    let mut skipped_corrupt = 0usize;

    for path in &media {
        if prog.cancel.load(Relaxed) {
            prog.finish(Outcome::Cancelled);
            return;
        }
        if !path.is_file() {
            continue;
        }

        // Mirror exactly what the app accepts (images + videos), so a backup never
        // silently drops a file the app itself shows. Still images are decode-
        // validated to skip corrupt ones; videos and GIFs are included as-is —
        // and so are AVIF/HEIC/camera raws, which `image` can't read (they go
        // through our own heavy decoders), rather than all counting as corrupt.
        let ext = ext_lower(path);
        let no_validate = crate::is_video(path)
            || SKIP_VALIDATION_EXTS.contains(&ext.as_str())
            || crate::is_extended_extension(&ext);
        let is_still_image = crate::is_image(path) && !no_validate;

        let include = if no_validate {
            true // a video or GIF the app accepts — include as-is
        } else if is_still_image {
            // A still image — corruption-scan it. A decoder that panics on it
            // can't judge it either: keep the file rather than fail the backup.
            std::panic::catch_unwind(|| is_image_valid(path)).unwrap_or(true)
        } else {
            continue; // not a media type the app accepts
        };

        if include {
            valid.push(path.clone());
            let txt = sidecar_txt(path);
            if txt.is_file() {
                valid.push(txt);
            }
        } else if is_still_image {
            skipped_corrupt += 1;
        }
    }

    if valid.is_empty() {
        prog.finish(Outcome::NoFiles);
        return;
    }

    let entries = archive_entries(&source, &valid);
    let total_selected = entries.len();
    prog.total.store(total_selected, Relaxed);

    // 2. Create <source>/backups/<name>_<date>.zip (numbered if taken).
    let backup_dir = source.join("backups");
    if let Err(e) = std::fs::create_dir_all(&backup_dir) {
        prog.finish(Outcome::Error(format!("Failed to create backups folder: {e}")));
        return;
    }
    let (zip_path, file) = match create_archive(&backup_dir, &zip_base) {
        Ok(created) => created,
        Err(e) => {
            prog.finish(Outcome::Error(format!("Failed to create the archive: {e}")));
            return;
        }
    };
    let zip_name = zip_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let result = write_zip(&entries, file, password.as_deref(), prog);

    // `zip_path` was created by this run (never an earlier backup), so it's
    // safe to remove on cancel or error.
    match result {
        Ok(WriteResult::Written { written, failed }) => prog.finish(Outcome::Success(Summary {
            zip_name,
            written,
            failed,
            total_selected,
            skipped_corrupt,
            encrypted: password.is_some(),
        })),
        Ok(WriteResult::Cancelled) => {
            let _ = std::fs::remove_file(&zip_path); // don't leave a partial archive
            prog.finish(Outcome::Cancelled);
        }
        Err(e) => {
            let _ = std::fs::remove_file(&zip_path);
            prog.finish(Outcome::Error(e));
        }
    }
}

enum WriteResult {
    Written { written: usize, failed: usize },
    Cancelled,
}

/// Files at least this big get zip64 headers (`large_file`): without them the
/// writer refuses an entry past 4 GiB part-way — a long video. The margin
/// covers deflate growing data that doesn't compress (and the AES overhead).
const LARGE_FILE: u64 = u32::MAX as u64 - (64 << 20);

fn write_zip(
    entries: &[(PathBuf, String)],
    file: std::fs::File,
    password: Option<&str>,
    prog: &Progress,
) -> Result<WriteResult, String> {
    use std::io::{Read as _, Seek as _, Write as _};

    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));

    let mut written = 0usize;
    let mut failed = 0usize;
    let mut buf = vec![0u8; 64 * 1024];

    for (i, (path, name)) in entries.iter().enumerate() {
        if prog.cancel.load(Relaxed) {
            return Ok(WriteResult::Cancelled);
        }

        let display = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        prog.set_label(format!("Archiving: {display}"));

        // A file that can't be read is logged, left out and counted as failed
        // — not archived as an empty or cut-off entry that counts as written —
        // so one unreadable file doesn't abort the whole backup (like the Java
        // version) but doesn't pass for backed up either. Opening it first
        // keeps most such files from getting an entry at all.
        let mut input = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Backup: failed to open {}: {e}", path.display());
                failed += 1;
                prog.current.store(i + 1, Relaxed);
                continue;
            }
        };
        let large = input.metadata().is_ok_and(|m| m.len() >= LARGE_FILE);

        let mut opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(6)) // "NORMAL"
            .large_file(large);
        if let Some(pw) = password {
            opts = opts.with_aes_encryption(AesMode::Aes256, pw);
        }

        zip.start_file(name.as_str(), opts).map_err(|e| e.to_string())?;

        let read_error = loop {
            match input.read(&mut buf) {
                Ok(0) => break None,
                // A failed write is the archive itself (disk full…) — that ends
                // the backup.
                Ok(n) => zip.write_all(&buf[..n]).map_err(|e| e.to_string())?,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => break Some(e),
            }
        };
        match read_error {
            None => written += 1,
            Some(e) => {
                eprintln!("Backup: failed to read {}: {e}", path.display());
                zip.abort_file().map_err(|e| e.to_string())?; // drop the cut-off entry
                failed += 1;
            }
        }
        prog.current.store(i + 1, Relaxed);
    }

    let out = zip.finish().map_err(|e| e.to_string())?;
    // An aborted entry's bytes can reach past the archive's new end (when it
    // was last, or later entries are smaller); cut them off — zip readers
    // look for the archive's directory at the very end of the file.
    let mut file = out.into_inner().map_err(|e| e.to_string())?;
    let end = file.stream_position().map_err(|e| e.to_string())?;
    file.set_len(end).map_err(|e| e.to_string())?;
    Ok(WriteResult::Written { written, failed })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty scratch folder for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("clarity_backup_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn names(entries: &[(PathBuf, String)]) -> Vec<&str> {
        entries.iter().map(|(_, n)| n.as_str()).collect()
    }

    #[test]
    fn entries_are_deduped_and_relative() {
        let base = std::env::temp_dir().join("clarity_backup_names");
        let root = base.join("set");
        let a = base.join("a").join("Photos");
        let b = base.join("b").join("photos");
        let files = vec![
            // cat.png + cat.jpg both push cat.txt.
            root.join("cat.png"),
            root.join("cat.txt"),
            root.join("cat.jpg"),
            root.join("cat.txt"),
            root.join("sub").join("x.png"),
            // Dropped in from elsewhere: two folders differing only in case.
            a.join("dog.png"),
            a.join("dog.txt"),
            b.join("dog.png"),
            b.join("dog.txt"),
            // A `..` can't climb out of the archive.
            root.join("sub").join("..").join("up.png"),
        ];
        let entries = archive_entries(&root, &files);
        assert_eq!(
            names(&entries),
            [
                "cat.png",
                "cat.txt",
                "cat.jpg",
                "sub/x.png",
                "external/Photos/dog.png",
                "external/Photos/dog.txt",
                "external/photos (2)/dog.png",
                "external/photos (2)/dog.txt",
                "external/root/up.png",
            ]
        );
        assert_eq!(entries[4].0, a.join("dog.png"));
    }

    #[test]
    fn colliding_entry_names_are_numbered() {
        let base = std::env::temp_dir().join("clarity_backup_collide");
        let root = base.join("set");
        let files = vec![
            root.join("external").join("Photos").join("dog.png"),
            base.join("Photos").join("dog.png"),
            base.join("Photos").join("README"),
            root.join("external").join("Photos").join("README"),
        ];
        assert_eq!(
            names(&archive_entries(&root, &files)),
            [
                "external/Photos/dog.png",
                "external/Photos/dog (2).png",
                "external/Photos/README",
                "external/Photos/README (2)",
            ]
        );
    }

    #[test]
    fn archive_name_never_reuses_an_existing_file() {
        let dir = scratch("create");
        std::fs::write(dir.join("set_2026-01-01.zip"), "yesterday's run").unwrap();
        let (first, _) = create_archive(&dir, "set_2026-01-01").unwrap();
        let (second, _) = create_archive(&dir, "set_2026-01-01").unwrap();
        assert_eq!(first, dir.join("set_2026-01-01 (2).zip"));
        assert_eq!(second, dir.join("set_2026-01-01 (3).zip"));
        assert_eq!(std::fs::read_to_string(dir.join("set_2026-01-01.zip")).unwrap(), "yesterday's run");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validator_reads_content_and_keeps_unknown_formats() {
        let dir = scratch("validate");
        image::RgbImage::new(2, 2).save(dir.join("a.png")).unwrap();
        std::fs::copy(dir.join("a.png"), dir.join("png_named.jpg")).unwrap();
        std::fs::write(dir.join("broken.png"), "not an image at all").unwrap();
        std::fs::write(dir.join("empty.png"), "").unwrap();
        std::fs::write(dir.join("photo.heic"), "a format `image` can't decode").unwrap();
        assert!(is_image_valid(&dir.join("a.png")));
        assert!(is_image_valid(&dir.join("png_named.jpg")));
        assert!(!is_image_valid(&dir.join("broken.png")));
        assert!(!is_image_valid(&dir.join("empty.png")));
        assert!(is_image_valid(&dir.join("photo.heic")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `cat.png` + `cat.jpg` share `cat.txt`: the backup used to fail on the
    /// second "cat.txt" — after truncating, then deleting, an earlier backup
    /// of the same name and day.
    #[test]
    fn shared_sidecar_backs_up_and_keeps_earlier_archive() {
        let dir = scratch("run");
        image::RgbImage::new(2, 2).save(dir.join("cat.png")).unwrap();
        std::fs::copy(dir.join("cat.png"), dir.join("cat.jpg")).unwrap();
        std::fs::write(dir.join("cat.txt"), "cat, tags").unwrap();
        std::fs::write(dir.join("bad.png"), "garbage").unwrap();
        std::fs::create_dir_all(dir.join("backups")).unwrap();
        std::fs::write(dir.join("backups").join("set.zip"), "earlier backup").unwrap();

        let prog = Progress::default();
        let media = vec![dir.join("cat.jpg"), dir.join("cat.png"), dir.join("bad.png")];
        run_backup(dir.clone(), media, "set".into(), None, &prog);
        let Some(Outcome::Success(s)) = prog.outcome.lock().unwrap().take() else {
            panic!("backup didn't succeed");
        };
        assert_eq!((s.zip_name.as_str(), s.written, s.failed, s.total_selected), ("set (2).zip", 3, 0, 3));
        assert_eq!(s.skipped_corrupt, 1);
        assert_eq!(std::fs::read_to_string(dir.join("backups").join("set.zip")).unwrap(), "earlier backup");

        let file = std::fs::File::open(dir.join("backups").join("set (2).zip")).unwrap();
        let mut zip = zip::ZipArchive::new(file).unwrap();
        let mut got: Vec<String> = zip.file_names().map(String::from).collect();
        got.sort();
        assert_eq!(got, ["cat.jpg", "cat.png", "cat.txt"]);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut zip.by_name("cat.txt").unwrap(), &mut text).unwrap();
        assert_eq!(text, "cat, tags");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A file that can't be read is counted as failed and left out — not
    /// archived as an empty entry that counts as written.
    #[test]
    fn unreadable_files_are_reported_not_archived_empty() {
        let dir = scratch("unreadable");
        std::fs::write(dir.join("ok.txt"), "fine").unwrap();
        std::fs::create_dir_all(dir.join("folder.png")).unwrap(); // opens on Linux, fails to read
        let entries = vec![
            (dir.join("gone.png"), "gone.png".to_string()),
            (dir.join("ok.txt"), "ok.txt".to_string()),
            (dir.join("folder.png"), "folder.png".to_string()),
        ];
        let zip_path = dir.join("out.zip");
        let file = std::fs::File::create(&zip_path).unwrap();
        let result = write_zip(&entries, file, None, &Progress::default()).unwrap();
        assert!(matches!(result, WriteResult::Written { written: 1, failed: 2 }));

        let bytes = std::fs::read(&zip_path).unwrap();
        // Nothing trails the end-of-central-directory record (no comment).
        assert_eq!(&bytes[bytes.len() - 22..bytes.len() - 18], b"PK\x05\x06");
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&zip_path).unwrap()).unwrap();
        assert_eq!(zip.file_names().collect::<Vec<_>>(), ["ok.txt"]);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut zip.by_name("ok.txt").unwrap(), &mut text).unwrap();
        assert_eq!(text, "fine");
        let _ = std::fs::remove_dir_all(&dir);
    }
}