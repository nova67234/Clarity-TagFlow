//! Deep Scan — finds problem files in a folder and quarantines them. A Rust port
//! of terminus2's `Scan.java`, opened from the top bar's "Find Issues" button.
//!
//! Two phases run on a background thread:
//!   1. **Corruption scan** — every image is decoded; ones that fail (empty,
//!      truncated, undecodable) are moved to a `corrupted_files/` subfolder.
//!   2. **Exact-duplicate scan** (optional) — remaining images are SHA-256
//!      hashed; all but one of each identical group move to `duplicates/`.
//!
//! Videos/GIFs are listed but skipped (decode-validating a video frame-by-frame
//! is pointless here). A timestamped `scan_log_…txt` report is written into the
//! `corrupted_files/` folder.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::theme::{ACCENT1, EDGE, FIELD, MUTED, PANEL, TEXT};

const CORRUPTED_FOLDER: &str = "corrupted_files";
const DUPLICATES_FOLDER: &str = "duplicates";

/// Image extensions the corruption scan will try to decode.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "ico", "hdr"];
/// Extended formats only decodable when built with `--features avif`.
const EXTENDED_EXTS: &[&str] = &["avif", "heic", "heif", "dng", "arw", "cr2"];
/// Media we list but never decode-validate.
const SKIP_EXTS: &[&str] = &["gif", "mp4", "webm", "avi", "mov", "mkv", "m4v", "wmv", "flv"];

/// Messages from the worker thread to the UI.
enum Msg {
    Log(String),
    Progress(f32, String),
    /// A file found corrupt: (file name, reason).
    Corrupt(String, String),
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
    corrupt: Vec<String>,
    progress: f32,
    status: String,

    running: bool,
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<Msg>>,
    /// Set true once after a scan finishes so the app can refresh the browser.
    pub finished_tick: bool,
    /// Where to place the window's top-left — the Find Issues button's
    /// bottom-left, captured when the window is opened.
    anchor_pos: Option<egui::Pos2>,
}

impl Default for ScanState {
    fn default() -> Self {
        Self {
            open: false,
            input_dir: String::new(),
            scan_duplicates: true,
            log: Vec::new(),
            corrupt: Vec::new(),
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
    /// Open the window under `anchor` (the Find Issues button's bottom-left),
    /// pre-filling the folder with the currently-open one.
    pub fn open_with(&mut self, folder: Option<&Path>, anchor: Option<egui::Pos2>) {
        self.open = true;
        self.anchor_pos = anchor;
        if self.input_dir.trim().is_empty() {
            if let Some(f) = folder {
                self.input_dir = f.display().to_string();
            }
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
    if !state.open {
        return;
    }

    // Drain background messages.
    if let Some(rx) = &state.rx {
        let drained: Vec<Msg> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        for m in drained {
            match m {
                Msg::Log(l) => state.push_log(l),
                Msg::Progress(p, s) => {
                    state.progress = p;
                    state.status = s;
                }
                Msg::Corrupt(name, reason) => state.corrupt.push(format!("{name}  |  {reason}")),
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

    // A compact, fixed-size window. Shrink to fit only on tiny screens. The
    // inner content width is pinned to `content_w` so the window can't auto-grow
    // to fill a large display.
    let screen = ctx.content_rect();
    let win_w = 460.0_f32.min(screen.width() - 40.0);
    let win_h = 440.0_f32.min(screen.height() - 40.0);
    let content_w = win_w - 28.0; // minus the window's 14px inner margins
    let console_h = 110.0_f32;

    // Pop the window just under the "Find Issues" button. The button's
    // bottom-left was captured on click; clamp so the window stays on-screen.
    let anchor = state.anchor_pos.unwrap_or_else(|| {
        egui::pos2(screen.center().x - win_w / 2.0, screen.top() + 80.0)
    });
    let x = anchor.x.min(screen.right() - win_w - 10.0).max(screen.left() + 10.0);
    let y = anchor.y.min(screen.bottom() - win_h - 10.0).max(screen.top() + 10.0);

    let mut open = state.open;
    egui::Window::new("Deep Scan")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .fixed_size([win_w, win_h])
        .fixed_pos([x, y])
        .frame(window_frame())
        .show(ctx, |ui| {
            ui.set_width(content_w);
            let radius = egui::CornerRadius::same(10);
            {
                let v = ui.visuals_mut();
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.active.corner_radius = radius;
                v.widgets.noninteractive.corner_radius = radius;
            }

            ui.add_space(2.0);
            ui.label(
                egui::RichText::new("Find corrupt images and exact duplicates, then quarantine them.")
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
                if crate::svg_button(ui, folder_svg, "Choose folder to scan", 32.0, crate::theme::icon_tint(MUTED())).clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        state.input_dir = dir.display().to_string();
                    }
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
            ui.add(
                egui::ProgressBar::new(state.progress.clamp(0.0, 1.0))
                    .text(egui::RichText::new(state.status.clone()).size(12.0))
                    .corner_radius(8)
                    .desired_height(18.0),
            );
            ui.add_space(8.0);

            // Two columns: console (left) and corrupt list (right). Fixed,
            // bounded height so the window can't grow past the screen.
            ui.columns(2, |cols| {
                console_box(&mut cols[0], "Console", &state.log, console_h, "scan_console");
                console_box(
                    &mut cols[1],
                    &format!("Corrupt images ({})", state.corrupt.len()),
                    &state.corrupt,
                    console_h,
                    "scan_corrupt",
                );
            });

            ui.add_space(8.0);

            // Start / Cancel.
            ui.horizontal(|ui| {
                let gap = 10.0;
                ui.spacing_mut().item_spacing.x = gap;
                let btn_w = (ui.available_width() - gap) / 2.0;
                let size = egui::vec2(btn_w, 36.0);

                let start = egui::Button::new(
                    egui::RichText::new("Start Scan").color(egui::Color32::WHITE).strong(),
                )
                .fill(ACCENT1());
                if ui.add_enabled_ui(!state.running, |ui| ui.add_sized(size, start)).inner.clicked() {
                    start_scan(state, ctx);
                }

                let cancel = egui::Button::new(
                    egui::RichText::new("Cancel").color(egui::Color32::WHITE).strong(),
                )
                .fill(egui::Color32::from_rgb(180, 40, 40));
                if ui.add_enabled_ui(state.running, |ui| ui.add_sized(size, cancel)).inner.clicked() {
                    state.cancel.store(true, Ordering::SeqCst);
                    state.status = "Cancelling…".to_string();
                }
            });
        });
    state.open = open;
}

/// A titled, scrollable, dark console well.
fn console_box(ui: &mut egui::Ui, title: &str, lines: &[String], height: f32, salt: &str) {
    ui.label(egui::RichText::new(title).color(MUTED()).strong().size(11.0));
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
                    if lines.is_empty() {
                        ui.label(egui::RichText::new("—").color(MUTED()).monospace().size(12.0));
                    } else {
                        for l in lines {
                            ui.label(egui::RichText::new(l).color(TEXT()).monospace().size(12.0));
                        }
                    }
                });
        });
}

fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(egui::CornerRadius::same(16))
        .inner_margin(egui::Margin::same(14))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 4],
            blur: 16,
            spread: 0,
            color: egui::Color32::from_black_alpha(140),
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
    state.corrupt.clear();
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
    let corrupted_dir = dir.join(CORRUPTED_FOLDER);
    let mut valid: Vec<PathBuf> = Vec::new();
    let mut corrupt_items: Vec<(PathBuf, String)> = Vec::new();

    let total = images.len().max(1);
    for (i, p) in images.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            finish(&tx, &ctx, sum, true);
            return;
        }
        prog(i as f32 / total as f32, format!("Scanning {} / {}", i + 1, images.len()));

        match validate_image(p) {
            Ok(()) => valid.push(p.clone()),
            Err(reason) => {
                let name = file_name(p);
                let _ = tx.send(Msg::Corrupt(name.clone(), reason.clone()));
                corrupt_items.push((p.clone(), reason));
            }
        }
    }
    sum.scanned = images.len() as u32;
    sum.corrupted = corrupt_items.len() as u32;

    if !corrupt_items.is_empty() {
        log(format!("Moving {} corrupt file(s) to '{CORRUPTED_FOLDER}'…", corrupt_items.len()));
        let _ = std::fs::create_dir_all(&corrupted_dir);
        for (p, _) in &corrupt_items {
            move_into(p, &corrupted_dir, "corrupted_", &log);
        }
    } else {
        log("No corrupted images found.".into());
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

    // Write a small report alongside the quarantined files.
    write_report(&dir, &corrupt_items, dupes_moved, &log);

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

fn write_report(dir: &Path, corrupt: &[(PathBuf, String)], dupes_moved: u32, log: &impl Fn(String)) {
    let report_dir = dir.join(CORRUPTED_FOLDER);
    if std::fs::create_dir_all(&report_dir).is_err() {
        return;
    }
    let stamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let path = report_dir.join(format!("scan_log_{stamp}.txt"));
    let mut body = String::new();
    body.push_str(&format!("Deep Scan Report — {}\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    body.push_str(&format!("Scanned directory: {}\n", dir.display()));
    body.push_str("============================================================\n\n");
    body.push_str(&format!("Corrupted files moved: {}\n", corrupt.len()));
    body.push_str(&format!("Exact duplicates moved: {dupes_moved}\n\n"));
    if !corrupt.is_empty() {
        body.push_str("--- CORRUPTED FILES ---\n");
        for (p, reason) in corrupt {
            body.push_str(&format!("MOVED: {}\n  REASON: {}\n\n", file_name(p), reason));
        }
    }
    if std::fs::write(&path, body).is_ok() {
        log(format!("Log saved to: {}", path.display()));
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
/// error.
fn sha256_file(path: &Path) -> Option<String> {
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
