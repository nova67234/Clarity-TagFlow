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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, RichText};
use zip::write::SimpleFileOptions;
use zip::{AesMode, CompressionMethod};

use crate::theme::*;

/// Green flash on the Create button on click (matches Tag Manager settings Save).
const FLASH_GREEN: Color32 = Color32::from_rgb(46, 160, 67);
/// Red flash on the Cancel button on click (matches Tag Manager settings Cancel).
const FLASH_RED: Color32 = Color32::from_rgb(200, 55, 55);
/// How long a button flash lingers before its action fires.
const FLASH: Duration = Duration::from_millis(450);

/// Fixed dialog body width. Everything inside uses finite widths derived from
/// this — NEVER `f32::INFINITY` or `ui.available_width()` — because in an
/// auto-sized egui window the available width is infinite, which leaks into the
/// persisted `last_content_size` as `inf` and reloads as a NaN rect that panics.
const DIALOG_WIDTH: f32 = 380.0;
/// Usable content width — fields fill it (flat sections, like the Civitai popup).
const FIELD_WIDTH: f32 = DIALOG_WIDTH - 4.0;

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
            .frame(window_frame());

        window.show(ctx, |ui| {
            // Fix BOTH min and max width. `set_width` alone leaves the max at the
            // auto-sized window's infinite available width, so `available_width()`
            // stays infinite — and the `desired_width(INFINITY)` fields below then
            // resolve to inf, which gets persisted and reloads as a NaN rect that
            // panics on the next launch. Capping max keeps every width finite.
            ui.set_width(380.0);
            ui.set_max_width(380.0);
            // Per-dialog widget styling — match the Settings / Tag Manager look:
            //  * square-but-rounded checkboxes (4px); the global theme rounds them
            //    into pills otherwise. Everything else (checkmark colour, fills)
            //    stays at the theme default so the checkbox looks identical to the
            //    Tag Manager settings one.
            //  * only the text wells use the darker BG() fill so the name/password
            //    boxes read as distinct inputs against the lighter FIELD() cards.
            let mut style = ui.style().as_ref().clone();
            let sq = egui::CornerRadius::same(4);
            for w in [
                &mut style.visuals.widgets.noninteractive,
                &mut style.visuals.widgets.inactive,
                &mut style.visuals.widgets.hovered,
                &mut style.visuals.widgets.active,
                &mut style.visuals.widgets.open,
            ] {
                w.corner_radius = sq;
            }
            // TextEdit fill comes from extreme_bg_color — set it to the section
            // card colour so the name/password boxes blend with the card behind
            // them, then give the input widgets a thin visible outline so the box
            // still reads as an input.
            style.visuals.extreme_bg_color = FIELD();
            let outline = egui::Stroke::new(1.0, Color32::from_gray(80));
            style.visuals.widgets.inactive.bg_stroke = outline;
            style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, Color32::from_gray(110));
            ui.set_style(style);

            // Custom title row (Civitai-style): backup icon + phase title + close.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/backup.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(TEXT()),
                );
                ui.heading(RichText::new(window_title).color(TEXT()).strong().size(17.0));
                // No close while a backup runs — it must be cancelled, not orphaned.
                if running.is_none() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(RichText::new("✕").size(14.0)).frame(false))
                            .on_hover_text("Close")
                            .clicked()
                        {
                            close_requested = true;
                        }
                    });
                }
            });
            ui.add_space(12.0);

            if let Some(prog) = &running {
                Self::running_body(ui, prog);
            } else if matches!(self.phase, Phase::Options) {
                self.options_body(ui);
            } else if let Phase::Done(outcome) = &self.phase {
                close_requested |= done_body(ui, outcome);
            }
        });

        // Dismissed via an in-dialog Cancel/Close button.
        if close_requested {
            self.open = false;
        }
    }

    fn options_body(&mut self, ui: &mut egui::Ui) {
        header(ui, "Archive this folder's media and tags into a .zip.");

        section(ui, "Details", |ui| {
            ui.label(RichText::new("Backup name").color(MUTED()).size(12.0));
            ui.add_space(4.0);
            text_field(ui, &mut self.name, "e.g. my_dataset", false);
            hint(ui, "Saved to a \"backups\" folder inside the source, dated automatically.");
        });

        section(ui, "Security", |ui| {
            ui.checkbox(
                &mut self.encrypt,
                RichText::new("Encrypt with password (AES-256)").color(TEXT()).size(13.0),
            );

            if self.encrypt {
                ui.add_space(8.0);
                ui.label(RichText::new("Password").color(MUTED()).size(12.0));
                ui.add_space(4.0);
                text_field(ui, &mut self.password, "", true);
                ui.add_space(8.0);
                ui.label(RichText::new("Confirm password").color(MUTED()).size(12.0));
                ui.add_space(4.0);
                text_field(ui, &mut self.confirm, "", true);
                hint(ui, "Keep this password safe — it can't be recovered if lost.");
            }
        });

        if let Some(err) = &self.form_error {
            ui.add_space(6.0);
            ui.label(RichText::new(err).color(Color32::from_rgb(235, 110, 110)).size(12.0));
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(10.0);

        // Buttons flash a colour on click, then fire their action once the flash
        // has lingered for `FLASH` (green = Create/start, red = Cancel/close) —
        // matching the Tag Manager settings Save/Cancel behaviour.
        let create_flashing = self.create_flash.is_some();
        let cancel_flashing = self.cancel_flash.is_some();
        let start_backup = self.create_flash.is_some_and(|t| t.elapsed() >= FLASH);
        let do_cancel = self.cancel_flash.is_some_and(|t| t.elapsed() >= FLASH);

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let create = footer_button(ui, "Create", create_flashing.then_some(FLASH_GREEN));
            if create.clicked() && self.create_flash.is_none() && !cancel_flashing {
                self.create_flash = Some(Instant::now());
            }
            ui.add_space(8.0);
            let cancel = footer_button(ui, "Cancel", cancel_flashing.then_some(FLASH_RED));
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
            if self.password != self.confirm {
                self.form_error = Some("The password and confirmation don't match.".into());
                return;
            }
            Some(self.password.clone())
        } else {
            None
        };

        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let zip_name = format!("{name}_{date}.zip");

        let prog = Arc::new(Progress::default());
        *prog.label.lock().unwrap() = "Scanning files…".to_string();

        let worker_prog = prog.clone();
        let source = self.source.clone();
        let files = self.files.clone();
        let zip_name_for_worker = zip_name.clone();
        std::thread::spawn(move || {
            // Catch any panic in the backup so it surfaces as a clean error in the
            // dialog instead of silently killing the worker (which would leave the
            // UI spinning forever) or aborting the process.
            let panic_prog = worker_prog.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_backup(source, files, zip_name_for_worker, password, &worker_prog);
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

        section(ui, "Progress", |ui| {
            let label = prog.label.lock().unwrap().clone();
            ui.label(RichText::new(label).color(MUTED()).size(12.0));
            ui.add_space(8.0);

            let total = prog.total.load(Relaxed);
            if total == 0 {
                ui.add(egui::ProgressBar::new(0.0).animate(true).desired_width(FIELD_WIDTH));
            } else {
                let cur = prog.current.load(Relaxed);
                let frac = (cur as f32 / total as f32).clamp(0.0, 1.0);
                ui.add(
                    egui::ProgressBar::new(frac)
                        .text(format!("{cur} / {total}"))
                        .desired_width(FIELD_WIDTH),
                );
            }
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(10.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let cancelling = prog.cancel.load(Relaxed);
            let btn_label = if cancelling { "Cancelling…" } else { "Cancel" };
            ui.add_enabled_ui(!cancelling, |ui| {
                if footer_button(ui, btn_label, None).clicked() {
                    prog.cancel.store(true, Relaxed);
                    prog.set_label("Cancelling…".into());
                }
            });
        });
    }
}

/// Draw the result screen. Returns `true` if the Close button was clicked.
fn done_body(ui: &mut egui::Ui, outcome: &Outcome) -> bool {
    match outcome {
        Outcome::Success(s) => {
            header(ui, "Your archive was created successfully.");
            section(ui, "Summary", |ui| {
                kv(ui, "Location", &s.zip_name);
                kv(ui, "Files written", &format!("{} / {}", s.written, s.total_selected));
                if s.encrypted {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("🔒 Password-protected (AES-256).")
                            .color(Color32::from_rgb(120, 200, 120))
                            .size(12.0),
                    );
                }
                if s.skipped_corrupt > 0 {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(format!(
                            "⚠ Skipped {} potentially corrupt image(s).",
                            s.skipped_corrupt
                        ))
                            .color(Color32::from_rgb(220, 180, 90))
                            .size(12.0),
                    );
                }
            });
        }
        Outcome::NoFiles => {
            header(ui, "Nothing was archived.");
            section(ui, "Details", |ui| {
                ui.label(RichText::new("No valid files found to back up.").color(MUTED()).size(13.0));
            });
        }
        Outcome::Cancelled => {
            header(ui, "The backup was stopped.");
            section(ui, "Details", |ui| {
                ui.label(RichText::new("Backup operation canceled.").color(MUTED()).size(13.0));
            });
        }
        Outcome::Error(e) => {
            header(ui, "Something went wrong during the backup.");
            section(ui, "Details", |ui| {
                ui.label(
                    RichText::new(e)
                        .color(Color32::from_rgb(235, 110, 110))
                        .size(12.0),
                );
            });
        }
    }

    ui.add_space(12.0);
    ui.separator();
    ui.add_space(10.0);
    let mut close = false;
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
fn header(ui: &mut egui::Ui, subtitle: &str) {
    ui.label(RichText::new(subtitle).color(MUTED()).size(11.0));
    ui.add_space(10.0);
}

/// A flat section: an uppercase muted label with controls directly below (matches
/// the Civitai popup — no bordered card).
fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.label(RichText::new(title.to_uppercase()).color(MUTED()).strong().size(11.0));
    ui.add_space(6.0);
    // Fixed width, never `available_width()` (which is infinite in an auto-sized
    // window and would leak `inf` into the persisted state).
    ui.scope(|ui| {
        ui.set_width(FIELD_WIDTH);
        add(ui);
    });
    ui.add_space(14.0);
}

/// A small muted explanatory line, shown under a control.
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(text).color(MUTED()).size(11.0));
}

/// A label : value row, used in the result summary.
fn kv(ui: &mut egui::Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{key}:")).color(MUTED()).size(12.0));
        ui.label(RichText::new(value).color(TEXT()).size(12.0));
    });
}

/// A footer button — 90×32, radius 10. Plain themed style (so it keeps the app's
/// default hover glow); pass `flash = Some(color)` to tint it for a click-flash.
/// A single-line text field styled like the Civitai popup's (rounded 10px corners,
/// roomier 10×8 padding, fills the available width).
fn text_field(ui: &mut egui::Ui, value: &mut String, hint: &str, password: bool) {
    ui.scope(|ui| {
        // The dialog rounds all widgets to 4px for the square checkboxes; round the
        // text fields back up to match Civitai.
        let r = egui::CornerRadius::same(10);
        let v = ui.visuals_mut();
        v.widgets.inactive.corner_radius = r;
        v.widgets.hovered.corner_radius = r;
        v.widgets.active.corner_radius = r;
        ui.add(
            egui::TextEdit::singleline(value)
                .password(password)
                .desired_width(f32::INFINITY)
                .margin(egui::Margin::symmetric(10, 8))
                .hint_text(hint),
        );
    });
}

fn footer_button(ui: &mut egui::Ui, label: &str, flash: Option<Color32>) -> egui::Response {
    let text = if flash.is_some() { Color32::WHITE } else { TEXT() };
    let mut btn = egui::Button::new(RichText::new(label).color(text).size(14.0))
        .corner_radius(egui::CornerRadius::same(10));
    if let Some(color) = flash {
        btn = btn.fill(color);
    }
    ui.add_sized([90.0, 32.0], btn)
}

/// The result screen's Close button — same footer style, no flash.
fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    footer_button(ui, label, None)
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
fn is_image_valid(path: &Path) -> bool {
    // HDR's `#?RGBE` signature variant trips `image`'s header reader, so validate
    // it through our own decoder instead of dropping good files from the backup.
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("hdr")) {
        return crate::image_cache::decode_hdr(path).is_some();
    }
    match image::image_dimensions(path) {
        Ok((w, h)) => w > 0 && h > 0,
        Err(_) => false,
    }
}

/// Swap a path's extension for `.txt` (its tag sidecar).
fn sidecar_txt(path: &Path) -> PathBuf {
    path.with_extension("txt")
}

/// `file` relative to `root`, using forward slashes; falls back to the file name.
fn safe_rel_path(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file);
    let s = rel
        .file_name()
        .map(|_| rel.to_string_lossy().into_owned())
        .unwrap_or_else(|| file.to_string_lossy().into_owned());
    s.replace('\\', "/")
}

fn run_backup(
    source: PathBuf,
    media: Vec<PathBuf>,
    zip_name: String,
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
        // validated to skip corrupt ones; videos and GIFs are included as-is.
        let ext = ext_lower(path);
        let no_validate = crate::is_video(path) || SKIP_VALIDATION_EXTS.contains(&ext.as_str());
        let is_still_image = crate::is_image(path) && !no_validate;

        let include = if no_validate {
            true // a video or GIF the app accepts — include as-is
        } else if is_still_image {
            is_image_valid(path) // a still image — corruption-scan it
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

    let total_selected = valid.len();
    prog.total.store(total_selected, Relaxed);

    // 2. Create <source>/backups/<name>_<date>.zip.
    let backup_dir = source.join("backups");
    if let Err(e) = std::fs::create_dir_all(&backup_dir) {
        prog.finish(Outcome::Error(format!("Failed to create backups folder: {e}")));
        return;
    }
    let zip_path = backup_dir.join(&zip_name);

    let result = write_zip(&source, &valid, &zip_path, password.as_deref(), prog);

    match result {
        Ok(WriteResult::Written(written)) => prog.finish(Outcome::Success(Summary {
            zip_name,
            written,
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
    Written(usize),
    Cancelled,
}

fn write_zip(
    source: &Path,
    valid: &[PathBuf],
    zip_path: &Path,
    password: Option<&str>,
    prog: &Progress,
) -> Result<WriteResult, String> {
    let file = std::fs::File::create(zip_path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));

    let mut written = 0usize;

    for (i, path) in valid.iter().enumerate() {
        if prog.cancel.load(Relaxed) {
            return Ok(WriteResult::Cancelled);
        }

        let display = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        prog.set_label(format!("Archiving: {display}"));

        let rel = safe_rel_path(source, path);

        let mut opts = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .compression_level(Some(6)); // "NORMAL"
        if let Some(pw) = password {
            opts = opts.with_aes_encryption(AesMode::Aes256, pw);
        }

        zip.start_file(rel, opts).map_err(|e| e.to_string())?;

        // A per-file read error is logged and skipped (the entry stays, possibly
        // empty) so one unreadable file doesn't abort the whole backup — matches
        // the Java behaviour.
        match std::fs::File::open(path) {
            Ok(mut input) => {
                if let Err(e) = std::io::copy(&mut input, &mut zip) {
                    eprintln!("Backup: failed to read {}: {e}", path.display());
                }
            }
            Err(e) => eprintln!("Backup: failed to open {}: {e}", path.display()),
        }

        written += 1;
        prog.current.store(i + 1, Relaxed);
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(WriteResult::Written(written))
}

/// A themed frame for the dialog body (matches the Civitai settings popup).
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