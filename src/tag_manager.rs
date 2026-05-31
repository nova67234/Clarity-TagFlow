//! A standalone Tag Manager panel for the right panel.
//! Mirrors the structure of `TagManagerPanel.java` with AI model selection,
//! threshold sliders, tag lists, and batch actions.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use eframe::egui;
use egui::{CornerRadius, Margin, Color32};

use crate::theme::*;

/// UI state for the tag manager.
pub struct TagManagerState {
    /// Buffer for the "add a tag" input.
    pub draft: String,

    // AI Settings
    pub ai_model: String,

    // Selection state for the Remove button
    pub selected_tags: HashSet<String>,

    // Header Status
    pub status_msg: String,
    pub status_is_error: bool,
    /// Render the status in the accent colour (e.g. the count phase of a flash).
    pub status_accent: bool,

    /// The image whose tag count the status currently reflects. Used to refresh
    /// the "Loaded N tags" status live when the user switches images.
    pub status_image: Option<PathBuf>,

    /// Whether the "Add All" batch write is awaiting confirmation.
    pub confirm_add_all: bool,

    /// The Tag Manager settings dialog (opened from the gear button).
    pub settings: crate::tag_manager_settings::TagManagerSettings,

    /// The "living" 3D particle orb shown in the AI bar.
    pub orb: crate::ai_orb::AiOrb,

    /// The "Get Models" downloader window (AI Model Manager).
    pub models: crate::ai_models::ModelManager,

    /// The currently loaded ONNX tagger, cached between Tag clicks, plus the
    /// model folder it was loaded from. `None` while a job owns it.
    pub tagger: Option<crate::tagger::Tagger>,
    pub tagger_folder: String,

    /// An in-flight background tagging job, if any.
    pub tag_job: Option<TagJob>,

    /// A transient two-stage status flash after a manual add/remove
    /// (count → "Saved" → back to the loaded-count status).
    pub flash: Option<TagFlash>,
}

/// A timed status flash: shows `label` (e.g. "Added 1 tag · 7 total"), then
/// "Saved", then expires back to the normal loaded-count status.
pub struct TagFlash {
    label: String,
    start: Instant,
}

/// Handle to a running background tagging job.
pub struct TagJob {
    rx: std::sync::mpsc::Receiver<TagJobDone>,
}

/// What a finished tagging job sends back: the (re-usable) loaded tagger, the
/// folder it belongs to, the predicted tags or an error, and the target image.
struct TagJobDone {
    tagger: Option<crate::tagger::Tagger>,
    folder: String,
    result: Result<Vec<String>, String>,
    image: PathBuf,
}

impl Default for TagManagerState {
    fn default() -> Self {
        Self {
            draft: String::new(),
            ai_model: "Select AI...".to_string(),
            selected_tags: HashSet::new(),
            status_msg: "No image selected".to_string(),
            status_is_error: true,
            status_accent: false,
            status_image: None,
            confirm_add_all: false,
            settings: crate::tag_manager_settings::TagManagerSettings::default(),
            orb: crate::ai_orb::AiOrb::default(),
            models: crate::ai_models::ModelManager::default(),
            tagger: None,
            tagger_folder: String::new(),
            tag_job: None,
            flash: None,
        }
    }
}

/// Render the tag manager. Operates on `current_tags` (the selected file's tag
/// buffer, already loaded from the sidecar by the right panel).
pub fn show(
    ui: &mut egui::Ui,
    state: &mut TagManagerState,
    current_image: Option<&Path>,
    current_tags: &mut String,
    all_images: &[PathBuf],
) {
    let mut tags = parse_tags(current_tags);

    // Auto-select the first installed model when nothing valid is chosen (e.g.
    // the "Select AI…" placeholder). `find` is a cheap in-memory catalog lookup,
    // so the disk check only runs while the selection is still the placeholder.
    if crate::ai_models::find(&state.ai_model).is_none() {
        if let Some(first) = crate::ai_models::installed_models().first() {
            state.ai_model = first.name().to_string();
        }
    }

    // Keep the header status in sync with the selected image. Recompute the
    // "Loaded N tags" count whenever the image changes (or none is selected) so
    // it stays live as the user clicks between images. Switching images also
    // cancels any in-flight add/remove flash. Transient messages set for the
    // *current* image (e.g. "thinking...") are otherwise left untouched.
    if current_image != state.status_image.as_deref() {
        state.status_image = current_image.map(Path::to_path_buf);
        state.flash = None;
        set_loaded_status(state, current_image, &tags);
    }

    // Two-stage flash after a manual add/remove: show the count for a beat,
    // then "Saved", then fall back to the loaded-count status.
    if let Some(flash) = &state.flash {
        const COUNT_SECS: f32 = 0.9; // phase 1: "Added/Removed N tags · M total"
        const SAVED_SECS: f32 = 0.8; // phase 2: "Saved"
        let t = flash.start.elapsed().as_secs_f32();
        if t < COUNT_SECS {
            // Phase 1: the count, in accent blue.
            state.status_msg = flash.label.clone();
            state.status_is_error = false;
            state.status_accent = true;
        } else if t < COUNT_SECS + SAVED_SECS {
            // Phase 2: "Saved", in green.
            state.status_msg = "Saved".to_string();
            state.status_is_error = false;
            state.status_accent = false;
        } else {
            state.flash = None;
            set_loaded_status(state, current_image, &tags);
        }
        // Keep repainting so the phases advance without needing input.
        if state.flash.is_some() {
            ui.ctx().request_repaint();
        }
    }

    // Poll a running AI tag job. When it finishes, re-cache the loaded model and
    // merge the predicted tags into the target image's sidecar.
    let mut finished: Option<TagJobDone> = None;
    if let Some(job) = &state.tag_job {
        match job.rx.try_recv() {
            Ok(done) => finished = Some(done),
            Err(std::sync::mpsc::TryRecvError::Empty) => ui.ctx().request_repaint(),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                state.tag_job = None;
                state.status_msg = "Tagging failed".to_string();
                state.status_is_error = true;
            }
        }
    }
    if let Some(done) = finished {
        state.tag_job = None;
        state.tagger = done.tagger;
        state.tagger_folder = done.folder;
        match done.result {
            Ok(new_tags) => {
                // Drop blacklisted tags (comma/newline separated in settings).
                let blacklist = parse_tags(&state.settings.blacklist);
                let mut existing = read_sidecar(&done.image);
                let mut added = 0;
                for t in new_tags {
                    let blocked = blacklist.iter().any(|b| b.eq_ignore_ascii_case(&t));
                    let dup = existing.iter().any(|e| e.eq_ignore_ascii_case(&t));
                    if !blocked && !dup {
                        existing.push(t);
                        added += 1;
                    }
                }
                write_sidecar(&done.image, &existing);
                // If the tagged image is still selected, refresh the live buffer
                // and this frame's tag list.
                if current_image == Some(done.image.as_path()) {
                    *current_tags = serialize_tags(&existing);
                    tags = existing;
                }
                state.status_msg = format!("Added {added} tags");
                state.status_is_error = false;
            }
            Err(e) => {
                state.status_msg = e;
                state.status_is_error = true;
            }
        }
    }

    // --- TOP: header bar, utility row, AI selection bar ---
    egui::Panel::top(ui.id().with("tagmgr_top"))
        .resizable(false)
        .show_separator_line(false)
        .frame(egui::Frame::NONE)
        .show_inside(ui, |ui| {
            // Header bar
            egui::Frame::new()
                .fill(FIELD())
                .corner_radius(CornerRadius::same(18))
                .inner_margin(Margin::symmetric(12, 6))
                .show(ui, |ui| {
                    // Fixed-height bar; `horizontal_centered` vertically centres the
                    // row so the title/status sit in the middle of the header.
                    ui.set_height(40.0);
                    ui.horizontal_centered(|ui| {
                        // Left: tag2 icon + title.
                        let tag_icon = egui::include_image!("../icons/tag2.svg");
                        ui.add(egui::Image::new(tag_icon).fit_to_exact_size(egui::vec2(16.0, 16.0)).tint(TEXT()));
                        ui.label(egui::RichText::new("Tag Manager").color(TEXT()).strong().size(14.0));

                        // Right: status. The 3D particle orb always sits just to
                        // the left of the status text — gently breathing while
                        // idle, spinning up while the assistant is "thinking".
                        let thinking = state.status_msg.contains("thinking");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let color = if state.status_is_error {
                                egui::Color32::from_rgb(220, 70, 70)
                            } else if state.status_accent || thinking || state.status_msg.contains("selected") {
                                ACCENT1()
                            } else {
                                egui::Color32::from_rgb(46, 160, 67)
                            };
                            // right_to_left: text first (rightmost), orb to its left.
                            ui.label(egui::RichText::new(&state.status_msg).color(color).size(12.0));
                            ui.add_space(6.0);
                            state.orb.set_state(if thinking {
                                crate::ai_orb::OrbState::Thinking
                            } else {
                                crate::ai_orb::OrbState::Idle
                            });
                            state.orb.show(ui, 30.0, None);
                        });
                    });
                });

            ui.add_space(10.0);

            // Utility row: Get Models + settings gear
            ui.horizontal(|ui| {
                let get_models_icon = egui::Image::new(egui::include_image!("../icons/extension.svg"))
                    .tint(Color32::WHITE)
                    .fit_to_exact_size(egui::vec2(16.0, 16.0));
                let get_models = ui.add_sized(
                    egui::vec2(110.0, 30.0),
                    egui::Button::image_and_text(
                        get_models_icon,
                        egui::RichText::new("Get Models").color(Color32::WHITE),
                    )
                    .corner_radius(CornerRadius::same(12)),
                );
                if get_models.clicked() {
                    state.models.toggle(); // toggle the AI Model Manager dropdown
                }
                // Dropdown drops down under the Get Models button.
                state.models.show(&get_models);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let gear = egui::include_image!("../icons/settings.svg");
                    let gear_resp = crate::svg_button(ui, gear, "Settings", 20.0, MUTED());
                    if gear_resp.clicked() {
                        state.settings.open = !state.settings.open;
                    }
                    // Settings popup drops down under the gear icon.
                    crate::tag_manager_settings::show(&gear_resp, &mut state.settings);
                });
            });

            ui.add_space(10.0);

            // AI selection bar — model dropdown + Tag button.
            egui::Frame::new()
                .fill(PANEL())
                .corner_radius(CornerRadius::same(12))
                .stroke(egui::Stroke::new(1.0, EDGE()))
                .inner_margin(Margin::symmetric(6, 6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Make controls the same height as the Tag button.
                        ui.spacing_mut().interact_size.y = 28.0;
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let tag_btn = egui::Button::new(egui::RichText::new("Tag").color(Color32::WHITE))
                                .corner_radius(CornerRadius::same(12))
                                .fill(ACCENT1());
                            if ui.add_sized(egui::vec2(56.0, 28.0), tag_btn).clicked()
                                && state.tag_job.is_none()
                            {
                                start_tag_job(state, current_image);
                            }
                            ui.add_space(6.0);
                            // Dropdown fills the remaining width to the left, and
                            // lists only models whose files are actually installed
                            // (checked on disk when the dropdown is opened).
                            egui::ComboBox::from_id_salt("ai_model_combo")
                                .width(ui.available_width())
                                .selected_text(&state.ai_model)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut state.ai_model, "Select AI...".to_string(), "Select AI...");
                                    let installed = crate::ai_models::installed_models();
                                    if installed.is_empty() {
                                        ui.label(
                                            egui::RichText::new("No models — click Get Models")
                                                .color(MUTED())
                                                .italics(),
                                        );
                                    } else {
                                        for m in installed {
                                            ui.selectable_value(&mut state.ai_model, m.name().to_string(), m.name());
                                        }
                                    }
                                });
                        });
                    });
                });

            ui.add_space(12.0);
        });

    // --- BOTTOM: manual add field + full-width action buttons (+ confirm) ---
    egui::Panel::bottom(ui.id().with("tagmgr_footer"))
        .resizable(false)
        .show_separator_line(false)
        .frame(egui::Frame::NONE)
        .show_inside(ui, |ui| {
            ui.add_space(12.0);

            ui.label(egui::RichText::new("Add Tags (manual):").color(TEXT()).strong().size(14.0));
            ui.add_space(6.0);
            egui::Frame::new()
                .fill(PANEL())
                .corner_radius(CornerRadius::same(12))
                .stroke(egui::Stroke::new(1.0, EDGE()))
                // Vertical padding so the text/caret sits inside the box instead of
                // poking out the top edge.
                .inner_margin(Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut state.draft)
                            .frame(egui::Frame::NONE)
                            .margin(Margin::ZERO)
                            .desired_width(f32::INFINITY),
                    );
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Some(path) = current_image {
                            let before = tags.len();
                            if add_tag(&mut tags, &state.draft) {
                                let added = tags.len() - before;
                                save(path, &tags, current_tags);
                                state.draft.clear();
                                resp.request_focus();
                                set_change_status(state, "Added", added, tags.len());
                            }
                        }
                    }
                });

            ui.add_space(10.0);

            // Three equal buttons spanning the panel width, matching the right
            // details panel's action-button row.
            ui.horizontal(|ui| {
                let gap = 8.0;
                ui.spacing_mut().item_spacing.x = gap;
                let btn_w = (ui.available_width() - gap * 2.0) / 3.0;
                let size = egui::vec2(btn_w, 35.0);

                let r = CornerRadius::same(12);
                ui.visuals_mut().widgets.inactive.corner_radius = r;
                ui.visuals_mut().widgets.hovered.corner_radius = r;
                ui.visuals_mut().widgets.active.corner_radius = r;

                if ui
                    .add_sized(size, egui::Button::new(egui::RichText::new("Add").color(Color32::WHITE)))
                    .clicked()
                {
                    if let Some(path) = current_image {
                        let before = tags.len();
                        if add_tag(&mut tags, &state.draft) {
                            let added = tags.len() - before;
                            save(path, &tags, current_tags);
                            state.draft.clear();
                            set_change_status(state, "Added", added, tags.len());
                        }
                    }
                }

                if ui
                    .add_sized(size, egui::Button::new(egui::RichText::new("Add All").color(Color32::WHITE)))
                    .on_hover_text("Add these tags to every loaded file")
                    .clicked()
                {
                    if state.draft.trim().is_empty() {
                        state.status_msg = "Type a tag first".to_string();
                        state.status_is_error = true;
                    } else if all_images.is_empty() {
                        state.status_msg = "No images loaded".to_string();
                        state.status_is_error = true;
                    } else {
                        state.confirm_add_all = true; // ask before touching many files
                    }
                }

                let danger_bg = egui::Color32::from_rgb(180, 40, 40);
                let remove_btn = egui::Button::new(egui::RichText::new("Remove").color(Color32::WHITE)).fill(danger_bg);
                // Last button takes the remaining width so float-division rounding
                // slack is absorbed here instead of overflowing the panel and
                // clipping the button against the frame's right edge.
                let remove_size = egui::vec2(ui.available_width(), size.y);
                if ui.add_sized(remove_size, remove_btn).clicked() && current_image.is_some() && !state.selected_tags.is_empty() {
                    let before = tags.len();
                    tags.retain(|t| !state.selected_tags.contains(t));
                    let removed = before - tags.len();
                    save(current_image.unwrap(), &tags, current_tags);
                    state.selected_tags.clear();
                    set_change_status(state, "Removed", removed, tags.len());
                }
            });

            // Inline confirmation for the multi-file "Add All" write.
            if state.confirm_add_all {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Add to all {} files?", all_images.len()))
                            .color(MUTED())
                            .size(12.0),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Confirm").clicked() {
                            let new_tags = parse_tags(&state.draft);
                            let n = add_to_all(all_images, &new_tags);
                            state.draft.clear();
                            state.confirm_add_all = false;
                            // Refresh the current file's buffer so the list reflects the write.
                            if let Some(path) = current_image {
                                *current_tags = serialize_tags(&read_sidecar(path));
                            }
                            state.status_msg = format!("Added to {n} images");
                            state.status_is_error = false;
                        }
                        if ui.button("Cancel").clicked() {
                            state.confirm_add_all = false;
                        }
                    });
                });
            }
        });

    // --- CENTER: "Current Tags:" header + tall tag list box (fills the gap) ---
    egui::CentralPanel::default()
        .frame(egui::Frame::NONE)
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Current Tags:").color(TEXT()).strong().size(14.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let copy_icon = egui::include_image!("../icons/copy.svg");
                    if crate::svg_button(ui, copy_icon, "Copy", 16.0, MUTED()).clicked() {
                        let text = serialize_tags(&tags);
                        ui.ctx().copy_text(text);
                        state.status_msg = "Copied to clipboard".to_string();
                        state.status_is_error = false;
                    }
                });
            });
            ui.add_space(6.0);

            // Tag list box: 22 rounded corners, fills the remaining vertical space.
            let list_h = ui.available_height().max(120.0);
            egui::Frame::new()
                .fill(PANEL())
                .corner_radius(CornerRadius::same(22))
                .stroke(egui::Stroke::new(1.0, EDGE()))
                .inner_margin(Margin::same(6))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .max_height(list_h - 12.0)
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            if tags.is_empty() {
                                ui.add_space(16.0);
                                ui.vertical_centered(|ui| ui.label(egui::RichText::new("No tags").color(MUTED())));
                            } else {
                                for tag in &tags {
                                    let is_selected = state.selected_tags.contains(tag);
                                    let bg_color = if is_selected {
                                        ACCENT1().gamma_multiply(0.4)
                                    } else {
                                        Color32::TRANSPARENT
                                    };
                                    let resp = egui::Frame::new()
                                        .fill(bg_color)
                                        .corner_radius(CornerRadius::same(8))
                                        .inner_margin(Margin::symmetric(10, 6))
                                        .show(ui, |ui| {
                                            ui.set_width(ui.available_width());
                                            ui.label(egui::RichText::new(tag).color(TEXT()).size(13.0))
                                        })
                                        .response;
                                    let interact = ui.interact(resp.rect, ui.id().with(tag), egui::Sense::click());
                                    if interact.clicked() {
                                        if is_selected {
                                            state.selected_tags.remove(tag);
                                        } else {
                                            state.selected_tags.insert(tag.clone());
                                        }
                                    }
                                }
                            }
                        });
                });
        });
}

/// Begin the two-stage status flash after a manual add/remove. The flash shows
/// e.g. `Added 1 tag · 7 total`, then `Saved`, then reverts to the loaded-count
/// status (driven by the state machine in `show`).
fn set_change_status(state: &mut TagManagerState, verb: &str, n: usize, total: usize) {
    let plural = if n == 1 { "" } else { "s" };
    state.flash = Some(TagFlash {
        label: format!("{verb} {n} tag{plural} · {total} total"),
        start: Instant::now(),
    });
}

/// Set the steady-state header status: the loaded-tag count, or an empty/no-image
/// notice.
fn set_loaded_status(state: &mut TagManagerState, current_image: Option<&Path>, tags: &[String]) {
    state.status_accent = false;
    if current_image.is_none() {
        state.status_msg = "No image selected".to_string();
        state.status_is_error = true;
    } else if tags.is_empty() {
        state.status_msg = "No tags".to_string();
        state.status_is_error = true;
    } else {
        state.status_msg = format!("Loaded {} tags", tags.len());
        state.status_is_error = false;
    }
}

// ---------------------------------------------------------------------------
// AI tagging
// ---------------------------------------------------------------------------

/// Kick off a background tagging job for the current image using the selected
/// model. Validates the selection and that the model files are present, then
/// spawns a worker thread; results are picked up by the poll in `show`.
fn start_tag_job(state: &mut TagManagerState, current_image: Option<&Path>) {
    let Some(image) = current_image else {
        state.status_msg = "No image selected".to_string();
        state.status_is_error = true;
        return;
    };
    let Some(info) = crate::ai_models::find(&state.ai_model) else {
        state.status_msg = "Select an AI model first".to_string();
        state.status_is_error = true;
        return;
    };
    let (Some(model_path), Some(tags_path)) = (
        crate::tagger::resolve(info.folder(), "model.onnx"),
        crate::tagger::resolve(info.folder(), info.tags_file()),
    ) else {
        state.status_msg = "Model not found — click 'Get Models'".to_string();
        state.status_is_error = true;
        return;
    };

    // Reuse the cached tagger only if it's the same model; otherwise drop it.
    let existing = if state.tagger_folder == info.folder() {
        state.tagger.take()
    } else {
        state.tagger = None;
        None
    };

    let image = image.to_path_buf();
    let threshold = state.settings.default_threshold;
    let kind = info.kind();
    let folder = info.folder().to_string();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (tagger, result) =
            crate::tagger::run_job(existing, kind, model_path, tags_path, &image, threshold);
        let _ = tx.send(TagJobDone { tagger, folder, result, image });
    });

    state.tag_job = Some(TagJob { rx });
    state.status_msg = "thinking...".to_string();
    state.status_is_error = false;
}

// ---------------------------------------------------------------------------
// Logic Helpers
// ---------------------------------------------------------------------------

/// Split a sidecar string into trimmed, de-duplicated tags (case-insensitive),
/// preserving first-seen order.
fn parse_tags(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in s.split(|c| c == ',' || c == '\n') {
        let t = raw.trim();
        if !t.is_empty() && !out.iter().any(|e| e.eq_ignore_ascii_case(t)) {
            out.push(t.to_string());
        }
    }
    out
}

/// Add `draft` to `tags` if it's non-empty. Splits commas to support batch pasting.
fn add_tag(tags: &mut Vec<String>, draft: &str) -> bool {
    if draft.trim().is_empty() { return false; }

    let mut changed = false;
    for raw in draft.split(',') {
        let t = raw.trim();
        if !t.is_empty() && !tags.iter().any(|e| e.eq_ignore_ascii_case(t)) {
            tags.push(t.to_string());
            changed = true;
        }
    }
    changed
}

/// Serialize tags back to the comma-separated sidecar form.
fn serialize_tags(tags: &[String]) -> String {
    tags.join(", ")
}

/// Apply the working tag list to the selected file: update the shared buffer and
/// write the `.txt` sidecar.
fn save(path: &Path, tags: &[String], current_tags: &mut String) {
    *current_tags = serialize_tags(tags);
    let _ = std::fs::write(crate::right_details::sidecar_txt(path), current_tags.as_bytes());
}

/// Read and parse a file's sidecar tags from disk.
fn read_sidecar(path: &Path) -> Vec<String> {
    let txt = crate::right_details::sidecar_txt(path);
    parse_tags(&std::fs::read_to_string(txt).unwrap_or_default())
}

/// Write a tag list to a file's sidecar.
fn write_sidecar(path: &Path, tags: &[String]) {
    let _ = std::fs::write(crate::right_details::sidecar_txt(path), serialize_tags(tags));
}

/// Add `new_tags` to every loaded file's sidecar, skipping files that already
/// have them all (case-insensitive). Returns the number of files changed.
fn add_to_all(all_images: &[PathBuf], new_tags: &[String]) -> usize {
    let mut count = 0;
    for img in all_images {
        let mut existing = read_sidecar(img);
        let mut changed = false;
        for t in new_tags {
            if !existing.iter().any(|e| e.eq_ignore_ascii_case(t)) {
                existing.push(t.clone());
                changed = true;
            }
        }
        if changed {
            write_sidecar(img, &existing);
            count += 1;
        }
    }
    count
}