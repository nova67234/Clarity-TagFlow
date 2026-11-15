//! A standalone Tag Manager panel for the right panel.
//! Mirrors the structure of `TagManagerPanel.java` with AI model selection,
//! threshold sliders, tag lists, and batch actions.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use eframe::egui;
use egui::{CornerRadius, Margin, Color32};

use crate::settings::{hairline, well_fill};
use crate::theme::*;

/// Amber for duplicate-tag feedback: the draft text while it matches an
/// existing tag, and the chip flash after trying to re-add one.
const DUP_TEXT_AMBER: Color32 = Color32::from_rgb(216, 162, 36);
const DUP_FLASH_AMBER: Color32 = Color32::from_rgb(240, 198, 60);
/// Chips the AI tagger just added: violet — distinct from the amber
/// duplicate flash, the green "Saved" status and the theme selection colour.
pub(crate) const AI_NEW_VIOLET: Color32 = Color32::from_rgb(155, 130, 240);
/// How long the duplicate chip flash runs (two pulses fading out).
const DUP_FLASH_SECS: f32 = 1.6;

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

    /// Whether the "Add to all images" batch write is awaiting confirmation.
    pub confirm_add_all: bool,

    /// The Tag Manager settings dialog (opened from the gear button).
    pub settings: crate::tag_manager_settings::TagManagerSettings,

    /// The "living" 3D particle orb shown in the header.
    pub orb: crate::ai_orb::AiOrb,

    /// The model downloader (AI Model Manager) — the settings popup's
    /// Models tab; ticked every frame so downloads finish regardless.
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

    /// Chips flashing amber because the user tried to re-add them (already
    /// present), with the flash start time.
    pub dup_flash: Option<(HashSet<String>, Instant)>,

    /// Tags the last AI tagging run added to the current image — tinted violet
    /// in the list so the user can tell the AI's contribution from what was
    /// already there. Cleared on image switch or the next run.
    pub ai_new_tags: HashSet<String>,

    /// When the Copy button was last clicked: tints the icon green and shows
    /// a "Copied" label beside it until the flash expires.
    pub copy_flash: Option<Instant>,

    /// The tag chip currently being dragged to a new position in the list,
    /// if any.
    pub drag_tag: Option<String>,

    /// In-place chip rename (started by right-clicking a chip): the original
    /// tag and the live edit buffer.
    pub edit_tag: Option<(String, String)>,
    /// Give the rename box keyboard focus on its first frame.
    pub edit_focus: bool,

    /// Spell-check state for the "add a tag" box (right-click suggestions).
    pub spell: crate::spellcheck::SpellcheckState,
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
            dup_flash: None,
            ai_new_tags: HashSet::new(),
            copy_flash: None,
            drag_tag: None,
            edit_tag: None,
            edit_focus: false,
            spell: crate::spellcheck::SpellcheckState::default(),
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

    // Poll model downloads every frame so they finalize, and their
    // percentages keep up, even while the settings popup is closed.
    state.models.tick(ui.ctx());

    // Ctrl+A (⌘A on macOS) selects every tag. Scoped to the pointer being over
    // the Tag Manager, and skipped while any text field owns the keyboard so
    // select-all inside the input box keeps working.
    if !tags.is_empty()
        && ui.ui_contains_pointer()
        && !ui.ctx().egui_wants_keyboard_input()
        && ui.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::A))
        })
    {
        state.selected_tags = tags.iter().cloned().collect();
    }

    // Auto-select the first installed model when nothing valid is chosen (e.g.
    // the "Select AI…" placeholder). `find` is a cheap in-memory catalog lookup,
    // so the disk check only runs while the selection is still the placeholder.
    if crate::ai_models::find(&state.ai_model).is_none()
        && let Some(first) = crate::ai_models::installed_models().first() {
            state.ai_model = first.name().to_string();
        }

    // Keep the header status in sync with the selected image. Recompute the
    // "Loaded N tags" count whenever the image changes (or none is selected) so
    // it stays live as the user clicks between images. Switching images also
    // cancels any in-flight add/remove flash. Transient messages set for the
    // *current* image (e.g. "thinking...") are otherwise left untouched.
    if current_image != state.status_image.as_deref() {
        state.status_image = current_image.map(Path::to_path_buf);
        state.flash = None;
        // A selection (or an in-progress chip drag/rename) only makes sense
        // for the image it was made on.
        state.selected_tags.clear();
        state.drag_tag = None;
        state.edit_tag = None;
        state.ai_new_tags.clear();
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
            // The image was moved or deleted while the tagger ran: writing
            // now would leave a .txt with no image behind.
            Ok(_) if !crate::archive::is_entry(&done.image) && !done.image.exists() => {
                state.status_msg = "The image was moved or deleted — tags not saved".to_string();
                state.status_is_error = true;
            }
            Ok(new_tags) => {
                // Drop blacklisted tags (comma/newline separated in settings).
                let blacklist = parse_tags(&state.settings.blacklist);
                // An unreadable sidecar is left alone rather than overwritten.
                if let Some(mut existing) = read_sidecar(&done.image) {
                    let mut fresh: HashSet<String> = HashSet::new();
                    let mut added = 0;
                    // Taggers name tags with underscores ("long_hair"): one the
                    // file already has as "long hair" isn't added again, and new
                    // ones are written the way the file writes its tags.
                    let style = existing.clone();
                    for t in new_tags {
                        let blocked = blacklist.iter().any(|b| crate::ai_tagging::same_tag(b, &t));
                        let dup = existing.iter().any(|e| crate::ai_tagging::same_tag(e, &t));
                        if !blocked && !dup {
                            let t = crate::ai_tagging::styled_like(&style, &t);
                            fresh.insert(t.clone());
                            existing.push(t);
                            added += 1;
                        }
                    }
                    match write_sidecar(&done.image, &existing) {
                        Some(text) => {
                            // If the tagged image is still selected, refresh the live
                            // buffer and this frame's tag list, and tint the run's
                            // additions so the user can tell them from the tags that
                            // were already there.
                            if current_image == Some(done.image.as_path()) {
                                *current_tags = text;
                                tags = existing;
                                state.ai_new_tags = fresh;
                            }
                            state.status_msg = format!("Added {added} tags");
                            state.status_is_error = false;
                        }
                        None => {
                            state.status_msg = "Couldn't save this image's .txt".to_string();
                            state.status_is_error = true;
                        }
                    }
                } else {
                    state.status_msg = "Couldn't read this image's .txt — left it untouched".to_string();
                    state.status_is_error = true;
                }
            }
            Err(e) => {
                state.status_msg = e;
                state.status_is_error = true;
            }
        }
    }

    // --- The view: one column, top to bottom. The tag box has the height
    // the rest leaves it. ---
    // What goes under the box is settled before the box is drawn: a click
    // in it changes the selection, and the rows under it mustn't change
    // height in the frame they were measured for.
    let sel_count = tags.iter().filter(|t| state.selected_tags.contains(t.as_str())).count();
    let confirming = state.confirm_add_all;
    let room = ui.available_height();
    // The hint under the box is shown where two rows of chips fit over it.
    let hint = sel_count == 0 && room >= ABOVE_BOX + TWO_ROWS_BOX + footer_height(false, true, confirming);
    let footer = footer_height(sel_count > 0, hint, confirming);
    let box_h = (room - ABOVE_BOX - footer).max(MIN_BOX_HEIGHT);

    let column = |ui: &mut egui::Ui| {
        // (The gaps are given in full.)
        ui.spacing_mut().item_spacing.y = 0.0;
        header(ui, state);
        ui.add_space(12.0);
        ai_bar(ui, state, current_image);
        ui.add_space(12.0);
        count_row(ui, state, &tags);
        ui.add_space(8.0);

        // The tag box. Tags render as wrapping pill chips; click one to
        // select it.
        egui::Frame::new()
            .fill(well())
            .corner_radius(CornerRadius::same(16))
            .stroke(egui::Stroke::new(1.0_f32, hairline()))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                // (The box, less the frame's margins and its outline.)
                let inner_h = box_h - 22.0;
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .max_height(inner_h)
                    // (A scroll area that scrolls is 64 tall at least, unless told.)
                    .min_scrolled_height(MIN_BOX_HEIGHT - 22.0)
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        if tags.is_empty() {
                            // A little way down the box; centred in one
                            // too short for that.
                            ui.add_space(((inner_h - 40.0) / 2.0).clamp(0.0, 24.0));
                            ui.vertical_centered(|ui| {
                                ui.label(egui::RichText::new("No tags yet").font(semibold(13.0)).color(TEXT()));
                                ui.add_space(4.0);
                                ui.label(
                                    egui::RichText::new("Add one below, or press Auto-tag")
                                        .font(medium(12.0))
                                        .color(MUTED()),
                                );
                            });
                        } else {
                            // Expire the duplicate-chip flash; keep repainting
                            // while it animates.
                            let mut flash_t = None; // 0..1 through the flash
                            if let Some((_, start)) = &state.dup_flash {
                                let t = start.elapsed().as_secs_f32() / DUP_FLASH_SECS;
                                if t >= 1.0 {
                                    state.dup_flash = None;
                                } else {
                                    flash_t = Some(t);
                                    ui.ctx().request_repaint();
                                }
                            }
                            let mut chip_rects: Vec<egui::Rect> = Vec::new();
                            let mut start_edit: Option<String> = None;
                            // Some(Some(..)) = commit rename, Some(None) = cancel.
                            let mut finish_edit: Option<Option<(String, String)>> = None;
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
                                for tag in &tags {
                                    // A chip being renamed (right-click) shows as
                                    // an inline edit box in its place.
                                    if state.edit_tag.as_ref().is_some_and(|(orig, _)| orig == tag) {
                                        let resp = {
                                            let (_, buf) = state.edit_tag.as_mut().unwrap();
                                            edit_chip(ui, buf)
                                        };
                                        chip_rects.push(resp.rect);
                                        if state.edit_focus {
                                            resp.request_focus();
                                            state.edit_focus = false;
                                        }
                                        // Enter commits; Escape / clicking away
                                        // cancels (both surrender focus).
                                        if resp.lost_focus() {
                                            let commit =
                                                ui.input(|i| i.key_pressed(egui::Key::Enter));
                                            let (orig, buf) = state.edit_tag.as_ref().unwrap();
                                            finish_edit =
                                                Some(commit.then(|| (orig.clone(), buf.clone())));
                                        }
                                        continue;
                                    }
                                    let selected = state.selected_tags.contains(tag);
                                    let flash = flash_t.filter(|_| {
                                        state.dup_flash.as_ref().is_some_and(|(set, _)| set.contains(tag))
                                    });
                                    let resp = tag_chip(
                                        ui,
                                        tag,
                                        selected,
                                        state.ai_new_tags.contains(tag),
                                        flash.map(flash_strength),
                                    );
                                    chip_rects.push(resp.rect);
                                    // Bring the already-there chip on screen as
                                    // the flash starts.
                                    if flash.is_some_and(|t| t * DUP_FLASH_SECS < 0.15) {
                                        resp.scroll_to_me(Some(egui::Align::Center));
                                    }
                                    if resp.drag_started() {
                                        state.drag_tag = Some(tag.clone());
                                    }
                                    if resp.secondary_clicked() {
                                        start_edit = Some(tag.clone());
                                    }
                                    if resp.clicked() {
                                        if selected {
                                            state.selected_tags.remove(tag);
                                        } else {
                                            state.selected_tags.insert(tag.clone());
                                        }
                                    }
                                }
                            });
                            // Order matters: finishing (e.g. focus lost to a
                            // right-click on another chip) before starting the
                            // next rename keeps the new edit alive.
                            if let Some(done) = finish_edit {
                                state.edit_tag = None;
                                if let Some((orig, new)) = done {
                                    rename_tag(state, &mut tags, &orig, new.trim(), current_image, current_tags);
                                }
                            }
                            if let Some(orig) = start_edit {
                                state.edit_tag = Some((orig.clone(), orig));
                                state.edit_focus = true;
                            }

                            // Drag-to-reorder: a floating copy of the dragged
                            // chip follows the cursor and a caret marks the gap
                            // it would land in; dropping there commits the new
                            // order to the sidecar. Releasing outside the list
                            // cancels.
                            if let Some(dragged) = state.drag_tag.clone() {
                                match tags.iter().position(|t| *t == dragged) {
                                    // The tag vanished under us (removed, or the
                                    // list changed) — abandon the drag.
                                    None => state.drag_tag = None,
                                    Some(from) => {
                                        let pointer = ui.ctx().pointer_latest_pos();
                                        let released =
                                            ui.input(|i| i.pointer.any_released());
                                        if let Some(pos) = pointer {
                                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                                            let to = insertion_index(&chip_rects, pos);
                                            drop_caret(ui, &chip_rects, to);
                                            floating_chip(ui.ctx(), &dragged, pos);
                                            if released {
                                                state.drag_tag = None;
                                                let inside = ui.clip_rect().contains(pos);
                                                // Removing shifts everything after
                                                // `from` left by one.
                                                let to = if to > from { to - 1 } else { to };
                                                if inside
                                                    && to != from
                                                    && let Some(img) = current_image
                                                {
                                                    let t = tags.remove(from);
                                                    tags.insert(to, t);
                                                    if save(state, img, &tags, current_tags) {
                                                        set_change_status(state, "Moved", 1, tags.len());
                                                    }
                                                }
                                            }
                                        } else if released {
                                            state.drag_tag = None;
                                        }
                                    }
                                }
                            }
                        }
                    });
            });

        // Under the box: the selection bar while tags are selected; else,
        // where there's room for it, what a chip does.
        if sel_count > 0 {
            ui.add_space(8.0);
            // Remove lives here, with the selection it acts on, rather than
            // as a button that sits disabled the rest of the time.
            if selection_bar(ui, state, sel_count)
                && let Some(img) = current_image
            {
                let before = tags.len();
                tags.retain(|t| !state.selected_tags.contains(t));
                let removed = before - tags.len();
                if save(state, img, &tags, current_tags) {
                    state.selected_tags.clear();
                    set_change_status(state, "Removed", removed, tags.len());
                }
            }
        } else if hint {
            ui.add_space(8.0);
            // The line keeps its place with no tags to act on, so the box
            // doesn't change size with them.
            let (line, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HINT_H), egui::Sense::hover());
            if !tags.is_empty() {
                let text = ui.painter().layout_no_wrap(CHIP_HINT.to_string(), medium(11.5), faint());
                // (Centred; from the left edge where the panel is too narrow for it.)
                let x = (line.center().x - text.size().x / 2.0).max(line.left());
                ui.painter().galley(egui::pos2(x, line.center().y - text.size().y / 2.0), text, faint());
            }
        }
        ui.add_space(12.0);

        // "Add tags", with the batch action at the row's end (a single add
        // is the box below: Enter).
        let has_draft = !state.draft.trim().is_empty();
        let can_add_all = has_draft && !all_images.is_empty();
        let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), LABEL_ROW_H), egui::Sense::hover());
        let mut row = at(ui, row);
        row.add(egui::Label::new(egui::RichText::new("Add tags").font(semibold(12.0)).color(MUTED())));
        row.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let look = if can_add_all {
                PillLook { fill: FIELD2(), ink: TEXT(), font: semibold(12.0) }
            } else {
                PillLook { fill: PANEL().lerp_to_gamma(FIELD(), 0.75), ink: faint(), font: semibold(12.0) }
            };
            let tip = if can_add_all {
                "Add these tags to every loaded file"
            } else if has_draft {
                "No images loaded"
            } else {
                "Type a tag first"
            };
            if pill(ui, "Add to all images", None, &look, LABEL_ROW_H, 12.0, can_add_all).on_hover_text(tip).clicked() {
                state.confirm_add_all = true; // ask before touching many files
            }
        });
        ui.add_space(6.0);
        draft_box(ui, state, &mut tags, current_image, current_tags);

        // Inline confirmation for the multi-file "Add to all" write.
        if confirming {
            ui.add_space(8.0);
            match confirm_bar(ui, all_images.len()) {
                Some(true) => {
                    let new_tags = parse_tags(&state.draft);
                    let (n, skipped) = add_to_all(all_images, &new_tags);
                    state.draft.clear();
                    state.confirm_add_all = false;
                    // Refresh the current file's buffer so the list reflects the write.
                    if let Some(path) = current_image
                        && let Ok(sc) = crate::sidecar::read(&crate::right_details::sidecar_txt(path))
                    {
                        *current_tags = sc.text;
                    }
                    state.status_msg = if skipped == 0 {
                        format!("Added to {n} images")
                    } else {
                        format!("Added to {n} images — {skipped} unreadable .txt files left untouched")
                    };
                    state.status_is_error = skipped > 0;
                }
                Some(false) => state.confirm_add_all = false,
                None => {}
            }
        }
    };
    if ABOVE_BOX + box_h + footer > room + 0.5 {
        // Too short for all of it (a window at its smallest, with tags
        // selected): the view scrolls rather than run past its card.
        egui::ScrollArea::vertical().id_salt("tagmgr_view").auto_shrink([false, false]).show(ui, column);
    } else {
        ui.scope(column);
    }
}

/// The Tag Manager's own faces, Manrope: Medium for text, SemiBold for labels
/// and Bold for the title and the buttons. egui picks a weight by family, so
/// each is a family of its own.
const MEDIUM: &str = "manrope_medium";
const SEMIBOLD: &str = "manrope_semibold";
const BOLD: &str = "manrope_bold";

/// Add the Tag Manager's faces to `fonts` (© The Manrope Project Authors,
/// under the SIL Open Font Licence 1.1: assets/fonts/OFL.txt), each with the
/// proportional list behind it for what Manrope lacks (CJK, emoji, symbols).
/// Laying out with an unknown family panics: whatever installs the app's
/// fonts calls this.
pub(crate) fn add_fonts(fonts: &mut egui::FontDefinitions) {
    let faces: [(&str, &'static [u8]); 3] = [
        (MEDIUM, include_bytes!("../assets/fonts/Manrope-Medium.ttf")),
        (SEMIBOLD, include_bytes!("../assets/fonts/Manrope-SemiBold.ttf")),
        (BOLD, include_bytes!("../assets/fonts/Manrope-Bold.ttf")),
    ];
    let fallbacks = fonts.families.get(&egui::FontFamily::Proportional).cloned().unwrap_or_default();
    for (name, bytes) in faces {
        fonts.font_data.insert(name.into(), egui::FontData::from_static(bytes).into());
        let mut list = vec![name.to_string()];
        list.extend(fallbacks.iter().cloned());
        fonts.families.insert(egui::FontFamily::Name(name.into()), list);
    }
}

pub(crate) fn medium(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name(MEDIUM.into()))
}

pub(crate) fn semibold(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name(SEMIBOLD.into()))
}

pub(crate) fn bold(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name(BOLD.into()))
}

/// The sunk surfaces (the AI bar, the tag box, the text box): the Settings
/// window's wells — glass under Liquid Glass, as its dropdowns are.
pub(crate) fn well() -> Color32 {
    if liquid_markers().is_some() { PANEL() } else { well_fill() }
}

/// Captions and hints: a step quieter than MUTED where small text still
/// reads at that (on the light themes MUTED is as quiet as it can be).
pub(crate) fn faint() -> Color32 {
    if is_light() { MUTED() } else { MUTED().gamma_multiply(0.88) }
}

/// The theme's accent (its blue, or Aurora's pink) deepened until a white
/// label reads on it: the fill of Auto-tag, Confirm and a selected chip.
fn accent_fill() -> Color32 {
    let c = egui::Rgba::from(selection_outline());
    let luminance = 0.2126 * c.r() + 0.7152 * c.g() + 0.0722 * c.b();
    // White on a colour of luminance L contrasts 1.05 / (L + 0.05) to 1:
    // 0.175 is 4.7 to 1.
    let k = (0.175 / luminance.max(0.001)).min(1.0);
    Color32::from(egui::Rgba::from_rgb(c.r() * k, c.g() * k, c.b() * k))
}

/// A chip's fill: lifted off the tag box — a lighter grey than it on a dark
/// theme, the cards' white on a light one (where the box is the grey), and
/// the system fill on Liquid Glass, which reads on any glass.
pub(crate) fn chip_fill(hovered: bool) -> Color32 {
    if is_light() && liquid_markers().is_none() {
        if hovered { FIELD2() } else { PANEL() }
    } else if hovered {
        FIELD2()
    } else {
        FIELD()
    }
}

/// The chips' padding around their text: 30 points tall on one line.
const CHIP_PAD: egui::Vec2 = egui::vec2(12.0, 6.0);

/// What a chip does, under the tag box; and what the text box is for.
const CHIP_HINT: &str = "Click to select · Right-click to rename · Drag to reorder";
const DRAFT_HINT: &str = "Type a tag and press Enter — commas add several";

/// The heights of the view's rows.
pub(crate) const HEADER_H: f32 = 44.0;
const AI_BAR_H: f32 = 52.0;
const COUNT_ROW_H: f32 = 24.0;
const HINT_H: f32 = 16.0;
const SELECTION_BAR_H: f32 = 44.0;
const LABEL_ROW_H: f32 = 26.0;
const DRAFT_BOX_H: f32 = 42.0;
const CONFIRM_BAR_H: f32 = 42.0;
/// The tag box with one row of chips (it's never shorter), and with two.
const MIN_BOX_HEIGHT: f32 = 52.0;
const TWO_ROWS_BOX: f32 = 88.0;

/// The height of what's over the tag box: the header, the AI bar and the
/// count row, with the gaps after them.
const ABOVE_BOX: f32 = HEADER_H + 12.0 + AI_BAR_H + 12.0 + COUNT_ROW_H + 8.0;

/// The height of what's under the tag box: the selection bar or the hint
/// line (if either), "Add tags" and its text box, and the Add-to-all
/// confirmation while it asks, with the gaps before them.
fn footer_height(selecting: bool, hint: bool, confirming: bool) -> f32 {
    let lead = if selecting {
        8.0 + SELECTION_BAR_H
    } else if hint {
        8.0 + HINT_H
    } else {
        0.0
    };
    let confirm = if confirming { 8.0 + CONFIRM_BAR_H } else { 0.0 };
    lead + 12.0 + LABEL_ROW_H + 6.0 + DRAFT_BOX_H + confirm
}

/// A child ui over `rect` that takes no room in `ui`, for the widgets of a
/// row that is laid out by hand: they sit centred on the row's height.
pub(crate) fn at(ui: &mut egui::Ui, rect: egui::Rect) -> egui::Ui {
    ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    )
}

/// How a [`pill`] is drawn.
struct PillLook {
    fill: Color32,
    ink: Color32,
    font: egui::FontId,
}

impl PillLook {
    /// White bold text on the accent: what its row is there for. Not
    /// `enabled`, it's the faded accent of the app's other accent buttons.
    fn accent(size: f32, enabled: bool) -> Self {
        if enabled {
            Self { fill: accent_fill(), ink: Color32::WHITE, font: bold(size) }
        } else {
            Self { fill: accent_fill().gamma_multiply(0.45), ink: Color32::WHITE.gamma_multiply(0.7), font: bold(size) }
        }
    }

    /// Text alone, with a faint pill under the pointer: Clear, Cancel.
    fn ghost(size: f32) -> Self {
        Self { fill: Color32::TRANSPARENT, ink: MUTED(), font: semibold(size) }
    }
}

/// The gap between a [`pill`]'s icon and its label.
const PILL_ICON_GAP: f32 = 6.0;

/// The width of a [`pill`] saying `label`, `pad` at each end (to leave room
/// for one that comes later in its row).
fn pill_width(ui: &egui::Ui, label: &str, look: &PillLook, icon: bool, pad: f32) -> f32 {
    let text = ui.painter().layout_no_wrap(label.to_string(), look.font.clone(), look.ink);
    // An icon starts 4 into the padding: its picture is smaller than its square.
    text.size().x + 2.0 * pad + if icon { 16.0 + PILL_ICON_GAP - 4.0 } else { 0.0 }
}

/// A pill-shaped button `height` tall: an optional 16-point `icon`, then
/// `label`, with `pad` at each end. It lightens under the pointer. Not
/// `enabled`, it only shows (the caller says why with a hover text).
fn pill(
    ui: &mut egui::Ui,
    label: &str,
    icon: Option<egui::ImageSource<'static>>,
    look: &PillLook,
    height: f32,
    pad: f32,
    enabled: bool,
) -> egui::Response {
    let width = pill_width(ui, label, look, icon.is_some(), pad);
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), sense);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, label));
    if ui.is_rect_visible(rect) {
        let fill = if !enabled {
            look.fill
        } else if resp.is_pointer_button_down_on() {
            look.fill.lerp_to_gamma(look.ink, 0.05)
        } else if resp.hovered() || resp.has_focus() {
            look.fill.lerp_to_gamma(look.ink, 0.12)
        } else {
            look.fill
        };
        ui.painter().rect_filled(rect, height / 2.0, fill);
        let text = ui.painter().layout_no_wrap(label.to_string(), look.font.clone(), look.ink);
        let mut x = rect.left() + pad;
        if let Some(icon) = icon {
            x -= 4.0;
            let slot = egui::Rect::from_min_size(egui::pos2(x, rect.center().y - 8.0), egui::Vec2::splat(16.0));
            egui::Image::new(icon).tint(look.ink).paint_at(ui, slot);
            x += 16.0 + PILL_ICON_GAP;
        }
        ui.painter().galley(egui::pos2(x, rect.center().y - text.size().y / 2.0), text, look.ink);
    }
    if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp }
}

/// The header: the orb, the title with the status under it, and the settings
/// button at the end.
fn header(ui: &mut egui::Ui, state: &mut TagManagerState) {
    let thinking = state.status_msg.contains("thinking");
    let color = if state.status_is_error {
        Color32::from_rgb(220, 70, 70)
    } else if state.status_accent || thinking || state.status_msg.contains("selected") {
        ACCENT1()
    } else {
        Color32::from_rgb(46, 160, 67)
    };
    // An error bursts the orb apart (red); a retry — i.e. going back to
    // Thinking — re-forms it.
    state.orb.set_state(if state.status_is_error {
        crate::ai_orb::OrbState::Error
    } else if thinking {
        crate::ai_orb::OrbState::Thinking
    } else {
        crate::ai_orb::OrbState::Idle
    });
    // Settings: its popup drops down under the button; the popup's Models
    // tab renders (and starts downloads on) the model manager.
    let gear = view_header(ui, &mut state.orb, "Tag Manager", &state.status_msg, color, "Settings", state.settings.open);
    if gear.clicked() {
        state.settings.open = !state.settings.open;
    }
    crate::tag_manager_settings::show(&gear, &mut state.settings, &mut state.models);
}

/// A right-panel view's header, [`HEADER_H`] tall: `orb`, `title` with
/// `status` (in `color`) under it, and a round button at the end, which
/// is returned: `gear_tip` says what it opens, and it stays lit while that
/// is `gear_open`. The status has one line of the room between the orb and
/// the button, so no message makes the header wider or taller: one a little
/// too long for the line is set a little smaller, and a longer one is cut
/// short with "…" (the whole message shows on hover).
pub(crate) fn view_header(
    ui: &mut egui::Ui,
    orb: &mut crate::ai_orb::AiOrb,
    title: &str,
    status: &str,
    color: Color32,
    gear_tip: &str,
    gear_open: bool,
) -> egui::Response {
    const STATUS_SIZE: f32 = 12.5;
    const ORB: f32 = 38.0;
    // The orb's cloud fills the middle of its square: the square starts this
    // far before the panel's edge, so the cloud lines up with what's below.
    const ORB_INSET: f32 = 5.0;
    const GEAR: f32 = 36.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HEADER_H), egui::Sense::hover());
    let text_left = rect.left() + ORB - ORB_INSET + 8.0;
    let text_w = (rect.right() - GEAR - 10.0 - text_left).max(0.0);

    let (title, status, title_h, status_h) = ui.fonts_mut(|f| {
        let line = |text: &str, font: egui::FontId, color: Color32, spacing: f32| {
            let mut job = egui::text::LayoutJob::simple_singleline(text.to_string(), font, color);
            job.sections[0].format.extra_letter_spacing = spacing;
            job.wrap = egui::text::TextWrapping::truncate_at_width(text_w);
            job
        };
        let title = f.layout_job(line(title, bold(17.0), TEXT(), -0.2));
        // Down to 85% of its size where that fits the message on the line
        // (the app's own longer ones, such as "The image was moved or
        // deleted — tags not saved", then show whole).
        let full = f.layout_no_wrap(status.to_string(), semibold(STATUS_SIZE), color).size().x;
        let scale = ((text_w - 2.0) / full.max(1.0)).clamp(0.85, 1.0);
        let status = f.layout_job(line(status, semibold(STATUS_SIZE * scale), color, 0.0));
        (title, status, f.row_height(&bold(17.0)), f.row_height(&semibold(STATUS_SIZE)))
    });
    // The two lines, centred on the header as one block.
    let top = rect.center().y - (title_h + status_h) / 2.0;
    ui.painter().galley(egui::pos2(text_left, top), title, TEXT());
    // (Set smaller, the status sits in the middle of its line. A label
    // given a cut-short galley shows the whole text on hover.)
    let status_at = egui::pos2(text_left, top + title_h + (status_h - status.size().y) / 2.0);
    at(ui, egui::Rect::from_min_size(status_at, status.size())).add(egui::Label::new(status));

    let orb_at = egui::pos2(rect.left() - ORB_INSET + ORB / 2.0, rect.center().y);
    orb.show(&mut at(ui, egui::Rect::from_center_size(orb_at, egui::Vec2::splat(ORB))), ORB, None);

    let gear_rect = egui::Rect::from_center_size(
        egui::pos2(rect.right() - GEAR / 2.0, rect.center().y),
        egui::Vec2::splat(GEAR),
    );
    let gear = ui
        .interact(gear_rect, ui.id().with("settings"), egui::Sense::click())
        .on_hover_text(gear_tip)
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    gear.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, gear_tip));
    if ui.is_rect_visible(gear_rect) {
        let hot = gear.hovered() || gear.has_focus() || gear_open;
        let fill = PANEL().lerp_to_gamma(FIELD(), if hot { 1.0 } else { 0.75 });
        ui.painter().circle_filled(gear_rect.center(), GEAR / 2.0, fill);
        let icon = egui::Rect::from_center_size(gear_rect.center(), egui::Vec2::splat(18.0));
        egui::Image::new(egui::include_image!("../icons/settings.svg"))
            .tint(icon_tint(if hot { TEXT() } else { MUTED() }))
            .paint_at(ui, icon);
    }
    gear
}

/// The AI bar: one sunk pill holding the model picker and, at its end, the
/// Auto-tag button.
fn ai_bar(ui: &mut egui::Ui, state: &mut TagManagerState, current_image: Option<&Path>) {
    const INSET: f32 = 6.0;
    let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), AI_BAR_H), egui::Sense::hover());
    ui.painter().rect(
        bar,
        AI_BAR_H / 2.0,
        well(),
        egui::Stroke::new(1.0_f32, hairline()),
        egui::StrokeKind::Inside,
    );
    let mut row = at(ui, bar.shrink(INSET));
    row.spacing_mut().item_spacing.x = INSET;

    // Faded while a run is under way: a click would do nothing then.
    let look = PillLook::accent(13.5, state.tag_job.is_none());
    // The button's width first: the picker has the rest of the bar.
    let button_w = pill_width(&row, "Auto-tag", &look, true, 18.0);
    let picker_w = (row.available_width() - button_w - INSET).max(0.0);
    model_picker(&mut row, state, bar, picker_w);
    let sparkle = egui::include_image!("../icons/sparkle.svg");
    if pill(&mut row, "Auto-tag", Some(sparkle), &look, AI_BAR_H - 2.0 * INSET, 18.0, state.tag_job.is_none())
        .on_hover_text("Tag this image with the chosen model")
        .clicked()
    {
        start_tag_job(state, current_image);
    }
}

/// The AI bar's model picker, `width` wide: "AI model" over the chosen
/// model's name, which is cut short where the chevron begins (the whole name
/// shows on hover). It opens the list of the models whose files are
/// installed (checked on disk when the list is opened) under the `bar`.
fn model_picker(ui: &mut egui::Ui, state: &mut TagManagerState, bar: egui::Rect, width: f32) {
    let id = egui::Id::new("tagmgr_ai_model");
    let height = ui.available_height();
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    let mut resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::ComboBox, true, &state.ai_model));
    let open = egui::Popup::is_id_open(ui.ctx(), id);
    let turn = ui.ctx().animate_bool(id.with("chevron"), open);
    if ui.is_rect_visible(rect) {
        let hot = resp.hovered() || resp.has_focus() || open;
        if hot {
            ui.painter().rect_filled(rect, height / 2.0, TEXT().gamma_multiply(0.05));
        }
        let left = rect.left() + 14.0;
        let (caption, name, caption_h, name_h) = ui.fonts_mut(|f| {
            let caption = f.layout_no_wrap("AI model".to_string(), semibold(10.5), faint());
            let mut name = egui::text::LayoutJob::simple_singleline(state.ai_model.clone(), semibold(13.0), TEXT());
            name.wrap = egui::text::TextWrapping::truncate_at_width((rect.right() - 30.0 - left).max(0.0));
            (caption, f.layout_job(name), f.row_height(&semibold(10.5)), f.row_height(&semibold(13.0)))
        });
        if name.elided {
            resp = resp.on_hover_text(&state.ai_model);
        }
        let top = rect.center().y - (caption_h + name_h) / 2.0;
        ui.painter().galley(egui::pos2(left, top), caption, faint());
        ui.painter().galley(egui::pos2(left, top + caption_h), name, TEXT());
        let chevron = egui::pos2(rect.right() - 17.0, rect.center().y);
        crate::settings::paint_chevron(ui.painter(), chevron, turn, if hot { TEXT() } else { MUTED() });
    }
    // The list hangs from the whole bar, as wide as it is (less the list's
    // own margins and outline).
    let mut anchor = resp;
    anchor.rect = bar;
    anchor.interact_rect = bar;
    crate::settings::dropdown_list(&anchor, id, (bar.width() - 14.0).max(180.0), |ui| {
        let mut pick = |ui: &mut egui::Ui, name: &str| {
            let on = state.ai_model == name;
            if crate::settings::dropdown_option(ui, name, None, None, on).clicked() {
                state.ai_model = name.to_string();
                ui.close();
            }
        };
        pick(ui, "Select AI...");
        let installed = crate::ai_models::installed_models();
        if installed.is_empty() {
            // "No models — ⚙ → Models", with real icons instead of glyphs.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let muted = |t: &str| egui::RichText::new(t).color(MUTED()).italics();
                let icon = |src: egui::ImageSource<'static>| {
                    egui::Image::new(src).tint(MUTED()).fit_to_exact_size(egui::vec2(13.0, 13.0))
                };
                ui.label(muted("No models —"));
                ui.add(icon(egui::include_image!("../icons/settings.svg")));
                ui.add(icon(egui::include_image!("../icons/arrow_right_alt.svg")));
                ui.label(muted("Models"));
            });
        } else {
            for m in installed {
                pick(ui, m.name());
            }
        }
    });
}

/// The row over the tag box: how many tags, how many of them the last AI
/// run added (the violet chips), and Copy at the row's end.
fn count_row(ui: &mut egui::Ui, state: &mut TagManagerState, tags: &[String]) {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), COUNT_ROW_H), egui::Sense::hover());

    // Copy: the list as it's saved. While the flash after a click is live
    // the icon is green and "Copied" sits beside it; both expire together.
    const COPY_FLASH_SECS: f32 = 1.5;
    let green = Color32::from_rgb(46, 160, 67);
    let flashing = match &state.copy_flash {
        Some(start) if start.elapsed().as_secs_f32() < COPY_FLASH_SECS => {
            ui.ctx().request_repaint(); // keep ticking so the flash expires
            true
        }
        Some(_) => {
            state.copy_flash = None;
            false
        }
        None => false,
    };
    let button = egui::Rect::from_min_max(egui::pos2(row.right() - 26.0, row.top()), row.max);
    let copy = ui
        .interact(button, ui.id().with("copy"), egui::Sense::click())
        .on_hover_text("Copy")
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    copy.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Copy"));
    let hot = copy.hovered() || copy.has_focus();
    if hot {
        ui.painter().rect_filled(button, 12.0, TEXT().gamma_multiply(0.08));
    }
    let tint = if flashing { green } else { icon_tint(if hot { TEXT() } else { MUTED() }) };
    let icon = egui::Rect::from_center_size(button.center(), egui::Vec2::splat(16.0));
    egui::Image::new(egui::include_image!("../icons/copy.svg")).tint(tint).paint_at(ui, icon);
    if copy.clicked() {
        ui.ctx().copy_text(serialize_tags(tags));
        state.copy_flash = Some(Instant::now());
    }
    if flashing {
        let text = ui.painter().layout_no_wrap("Copied".to_string(), semibold(12.0), green);
        let pos = egui::pos2(button.left() - 2.0 - text.size().x, row.center().y - text.size().y / 2.0);
        ui.painter().galley(pos, text, green);
    }

    let mut left = at(ui, row);
    left.spacing_mut().item_spacing.x = 6.0;
    let n = tags.len();
    let count = format!("{n} {}", if n == 1 { "tag" } else { "tags" });
    left.add(egui::Label::new(egui::RichText::new(count).font(bold(14.0)).color(TEXT())));
    let ai_new = tags.iter().filter(|t| state.ai_new_tags.contains(t.as_str())).count();
    if ai_new > 0 {
        left.add_space(3.0);
        let (dot, _) = left.allocate_exact_size(egui::vec2(6.0, 6.0), egui::Sense::hover());
        left.painter().circle_filled(dot.center(), 3.0, AI_NEW_VIOLET);
        let note = egui::RichText::new(format!("{ai_new} new from AI")).font(semibold(12.0));
        left.add(egui::Label::new(note.color(AI_NEW_VIOLET.lerp_to_gamma(TEXT(), 0.35))));
    }
}

/// The bar that takes the hint's place while tags are selected: how many, a
/// way to let go of them, and Remove at the end. True when Remove was clicked.
fn selection_bar(ui: &mut egui::Ui, state: &mut TagManagerState, count: usize) -> bool {
    const HEIGHT: f32 = SELECTION_BAR_H;
    let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HEIGHT), egui::Sense::hover());
    // It floats a little: a soft shadow under it, kept inside the bar's
    // ends (where the view scrolls, nothing shows past its edges, which
    // would cut the shadow off in a straight line).
    let shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 16,
        spread: 0,
        color: Color32::from_black_alpha(if is_light() { 28 } else { 80 }),
    };
    ui.painter().add(shadow.as_shape(bar.shrink2(egui::vec2(14.0, 0.0)), HEIGHT / 2.0));
    ui.painter().rect(bar, HEIGHT / 2.0, FIELD(), egui::Stroke::new(1.0_f32, FIELD2()), egui::StrokeKind::Inside);

    let mut row = at(ui, egui::Rect::from_min_max(bar.min + egui::vec2(16.0, 0.0), bar.max - egui::vec2(6.0, 0.0)));
    row.spacing_mut().item_spacing.x = 6.0;
    let label = egui::RichText::new(format!("{count} selected")).font(bold(13.0)).color(TEXT());
    row.add(egui::Label::new(label));
    if pill(&mut row, "Clear", None, &PillLook::ghost(12.5), 28.0, 10.0, true).clicked() {
        state.selected_tags.clear();
    }
    let mut remove = false;
    row.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let look = PillLook { fill: Color32::from_rgb(180, 40, 40), ink: Color32::WHITE, font: bold(12.5) };
        let bin = egui::include_image!("../icons/delete.svg");
        remove = pill(ui, "Remove", Some(bin), &look, 32.0, 14.0, true).clicked();
    });
    remove
}

/// The "Add tags" text box: a sunk pill, ringed in the accent colour while
/// it's typed in. Enter adds what's typed (commas add several).
fn draft_box(
    ui: &mut egui::Ui,
    state: &mut TagManagerState,
    tags: &mut Vec<String>,
    current_image: Option<&Path>,
    current_tags: &mut String,
) {
    const HEIGHT: f32 = DRAFT_BOX_H;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HEIGHT), egui::Sense::hover());
    // The pill goes under the text: its slot is taken first, filled in once
    // the field says whether it has focus (as `settings::pill_field` does).
    let bg = ui.painter().add(egui::Shape::Noop);
    // The field has the pill's whole height, so a click anywhere in it types.
    let inner = rect.shrink2(egui::vec2(16.0, 0.0));
    let mut field = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inner)
            .layout(egui::Layout::centered_and_justified(egui::Direction::TopDown)),
    );
    // Amber text while any typed segment (comma-separated, so multi-word
    // tags match whole) is already in the list.
    let draft_is_dup = state.draft.split(',').any(|raw| {
        let t = raw.trim();
        !t.is_empty() && tags.iter().any(|e| e.eq_ignore_ascii_case(t))
    });
    let out = egui::TextEdit::singleline(&mut state.draft)
        // (Named: what's drawn above the box comes and goes with the selection.)
        .id(egui::Id::new("tagmgr_draft"))
        .frame(egui::Frame::NONE)
        .margin(Margin::ZERO)
        .font(medium(13.0))
        // Explicit: the Light theme's widget text is white (for its blue buttons).
        .text_color(if draft_is_dup { DUP_TEXT_AMBER } else { TEXT() })
        .hint_text(egui::RichText::new(DRAFT_HINT).color(faint()))
        .vertical_align(egui::Align::Center)
        .desired_width(inner.width())
        .show(&mut field);
    // Live spell-check: red squiggles + right-click suggestions (same
    // treatment as the generator prompt boxes).
    crate::spellcheck::attach(&field, &out, &mut state.draft, &mut state.spell);
    let resp = &out.response.response;
    let ring = if resp.has_focus() {
        selection_outline()
    } else {
        TEXT().gamma_multiply(if state.draft.is_empty() { 0.1 } else { 0.2 })
    };
    ui.painter().set(
        bg,
        egui::epaint::RectShape::new(rect, HEIGHT / 2.0, well(), egui::Stroke::new(1.0_f32, ring), egui::StrokeKind::Inside),
    );
    if resp.lost_focus()
        && ui.input(|i| i.key_pressed(egui::Key::Enter))
        && let Some(path) = current_image
        && commit_draft(state, tags, path, current_tags)
    {
        resp.request_focus();
    }
}

/// The bar that asks before "Add to all images" writes to `files` files:
/// `Some(true)` on Confirm, `Some(false)` on Cancel.
fn confirm_bar(ui: &mut egui::Ui, files: usize) -> Option<bool> {
    const HEIGHT: f32 = CONFIRM_BAR_H;
    let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), HEIGHT), egui::Sense::hover());
    ui.painter().rect(
        bar,
        HEIGHT / 2.0,
        PANEL().lerp_to_gamma(FIELD(), 0.5),
        egui::Stroke::new(1.0_f32, hairline()),
        egui::StrokeKind::Inside,
    );
    let mut row = at(ui, egui::Rect::from_min_max(bar.min + egui::vec2(16.0, 0.0), bar.max - egui::vec2(6.0, 0.0)));
    row.spacing_mut().item_spacing.x = 6.0;
    let question = egui::RichText::new(format!("Add to all {files} files?")).font(semibold(12.5)).color(TEXT());
    row.add(egui::Label::new(question));
    let mut answer = None;
    row.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        if pill(ui, "Confirm", None, &PillLook::accent(12.5, true), 30.0, 14.0, true).clicked() {
            answer = Some(true);
        }
        if pill(ui, "Cancel", None, &PillLook::ghost(12.5), 28.0, 10.0, true).clicked() {
            answer = Some(false);
        }
    });
    answer
}

/// Turn flash progress (0..1) into blend strength: two amber pulses fading out.
fn flash_strength(t: f32) -> f32 {
    let fade = 1.0 - t;
    let pulse = 0.6 + 0.4 * (t * std::f32::consts::TAU * 2.0).cos();
    (fade * pulse).clamp(0.0, 1.0)
}

/// A tag chip: a pill lifted off the tag box (a step lighter under the
/// pointer); the accent colour with white text while selected. It's the same
/// size either way: a tick would widen it, and in a wrapped list that sends
/// the chips after it to other rows at every click. `flash` (0..1) blends
/// the chip toward amber — the "already added" pulse. `ai_new` marks a tag
/// the last AI run added: a violet outline and a faint violet wash
/// (selection style still wins while selected). Caption-length tags wrap
/// into multi-line chips at the panel width rather than clipping. Returns
/// the click response; the caller toggles.
fn tag_chip(ui: &mut egui::Ui, text: &str, selected: bool, ai_new: bool, flash: Option<f32>) -> egui::Response {
    let font = medium(13.0);
    let pad = CHIP_PAD;
    let wrap_w = (ui.max_rect().width() - pad.x * 2.0).max(40.0);
    let (galley, row_h) = ui.fonts_mut(|f| {
        (
            f.layout(text.to_string(), font.clone(), Color32::PLACEHOLDER, wrap_w),
            f.row_height(&font),
        )
    });
    // click = toggle selection; drag = reorder (handled by the caller).
    let (rect, resp) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::click_and_drag());
    if ui.is_rect_visible(rect) {
        let (mut fill, mut ink, mut edge) = if selected {
            (accent_fill(), Color32::WHITE, Color32::TRANSPARENT)
        } else {
            (chip_fill(resp.hovered()), TEXT(), EDGE())
        };
        if ai_new && !selected {
            fill = fill.lerp_to_gamma(AI_NEW_VIOLET, 0.12);
            ink = ink.lerp_to_gamma(AI_NEW_VIOLET, 0.2);
            edge = AI_NEW_VIOLET.gamma_multiply(0.6);
        }
        if let Some(k) = flash {
            fill = blend(fill, DUP_FLASH_AMBER, k);
            // Dark ink at peak so the label stays readable on amber.
            ink = blend(ink, Color32::from_rgb(64, 48, 12), k);
        }
        // Cap at the single-line pill radius so multi-line chips stay
        // rounded-rectangular instead of lozenge-shaped.
        let radius = CornerRadius::same((rect.height().min(row_h + pad.y * 2.0) / 2.0) as u8);
        ui.painter().rect(rect, radius, fill, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);
        ui.painter().galley(rect.min + pad, galley, ink);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// The in-place rename box a chip turns into on right-click: a chip-shaped
/// frame (selection-coloured edge) around a bare single-line editor sized to
/// its text.
fn edit_chip(ui: &mut egui::Ui, buf: &mut String) -> egui::Response {
    let font = medium(13.0);
    let pad = CHIP_PAD;
    // Size to the text, but never wider than the panel — a caption-length
    // rename scrolls inside the editor instead of clipping at the edge.
    let (text_w, row_h) = ui.fonts_mut(|f| {
        (
            f.layout_no_wrap(buf.clone(), font.clone(), Color32::PLACEHOLDER).size().x,
            f.row_height(&font),
        )
    });
    let w = (text_w.max(40.0) + 8.0).min((ui.max_rect().width() - pad.x * 2.0 - 10.0).max(40.0));
    // Allocate the chip-shaped rect FIRST, like `tag_chip` does — the wrapped
    // layout then places it (wrapping to a fresh row when it doesn't fit)
    // before anything paints, so the box can neither run off the panel's
    // right edge nor balloon past its text.
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, row_h) + pad * 2.0, egui::Sense::hover());
    let radius = CornerRadius::same((rect.height() / 2.0) as u8);
    ui.painter().rect(
        rect,
        radius,
        chip_fill(false),
        egui::Stroke::new(1.0_f32, selection_outline()),
        egui::StrokeKind::Inside,
    );
    // A non-allocating child ui for the editor — `ui.put` would allocate the
    // inner rect in the parent AGAIN, pulling the wrapped cursor back so the
    // next chip overlapped the box.
    let inner = egui::Rect::from_min_size(rect.min + pad, egui::vec2(w, row_h));
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inner)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    child.add(
        egui::TextEdit::singleline(buf)
            .frame(egui::Frame::NONE)
            .margin(Margin::ZERO)
            .font(font)
            // Explicit: the Light theme's widget text is white (for its blue buttons).
            .text_color(TEXT())
            .desired_width(w),
    )
}

/// Commit a chip rename: swap `orig` for `new` in place (same position), carry
/// any selection over, and save. Renaming onto a different existing tag
/// flashes that chip amber instead (the same "already there" cue as re-adding
/// a duplicate); an empty edit or no-op is a cancel. Case-only changes are
/// allowed through the duplicate check.
fn rename_tag(
    state: &mut TagManagerState,
    tags: &mut [String],
    orig: &str,
    new: &str,
    current_image: Option<&Path>,
    current_tags: &mut String,
) {
    let Some(i) = tags.iter().position(|t| t == orig) else { return };
    if new.is_empty() || new == orig {
        return;
    }
    if let Some(j) = tags.iter().position(|t| t.eq_ignore_ascii_case(new))
        && j != i
    {
        state.dup_flash = Some((std::iter::once(tags[j].clone()).collect(), Instant::now()));
        return;
    }
    tags[i] = new.to_string();
    if state.selected_tags.remove(orig) {
        state.selected_tags.insert(new.to_string());
    }
    if let Some(img) = current_image
        && save(state, img, tags, current_tags)
    {
        set_change_status(state, "Renamed", 1, tags.len());
    }
}

/// Where a chip dropped at `pos` should land in the laid-out list: the index
/// of the first chip in the pointer's row whose centre is past the pointer —
/// or, past every chip, the end of the list. Chips wrap in rows, so anything
/// in a row above counts as before the pointer.
fn insertion_index(rects: &[egui::Rect], pos: egui::Pos2) -> usize {
    for (i, r) in rects.iter().enumerate() {
        if pos.y < r.top() {
            return i; // the pointer's row ended (or it's above the first row)
        }
        if pos.y <= r.bottom() && pos.x < r.center().x {
            return i;
        }
    }
    rects.len()
}

/// Vertical caret marking the gap a dragged chip would drop into: before chip
/// `idx`, or after the last chip when `idx` is one past the end.
fn drop_caret(ui: &egui::Ui, rects: &[egui::Rect], idx: usize) {
    let (x, top, bottom) = if let Some(r) = rects.get(idx) {
        (r.left() - 4.0, r.top(), r.bottom())
    } else if let Some(r) = rects.last() {
        (r.right() + 4.0, r.top(), r.bottom())
    } else {
        return;
    };
    ui.painter().line_segment(
        [egui::pos2(x, top), egui::pos2(x, bottom)],
        egui::Stroke::new(2.0_f32, selection_outline()),
    );
}

/// The dragged chip's stand-in: a selection-coloured copy of the pill floating
/// under the cursor, on the tooltip layer so it rides above the list.
fn floating_chip(ctx: &egui::Context, text: &str, pos: egui::Pos2) {
    let font = medium(13.0);
    let pad = CHIP_PAD;
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        egui::Id::new("tag_drag_ghost"),
    ));
    // Wrap caption-length tags like the list chips do, so the ghost stays
    // a readable card instead of one screen-wide line.
    let wrap_w = (ctx.content_rect().width() * 0.5).max(120.0);
    let galley = painter.layout(text.to_string(), font.clone(), Color32::PLACEHOLDER, wrap_w);
    let row_h = painter
        .layout_no_wrap(" ".to_string(), font, Color32::PLACEHOLDER)
        .size()
        .y;
    let rect = egui::Rect::from_center_size(pos, galley.size() + pad * 2.0);
    let radius = CornerRadius::same((rect.height().min(row_h + pad.y * 2.0) / 2.0) as u8);
    painter.rect_filled(rect, radius, accent_fill().gamma_multiply(0.85));
    painter.galley(rect.min + pad, galley, Color32::WHITE);
}

/// Linear blend from `a` to `b` by `k` (0..1).
fn blend(a: Color32, b: Color32, k: f32) -> Color32 {
    Color32::from(egui::Rgba::from(a) * (1.0 - k) + egui::Rgba::from(b) * k)
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
        state.status_msg = "Model not found — Settings → Models".to_string();
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
    // Only tagger models reach here (the dropdown lists `installed_models()`,
    // which excludes non-taggers like the depth estimator).
    let Some(kind) = info.kind() else {
        state.status_msg = "That model can't be used for tagging".to_string();
        state.status_is_error = true;
        return;
    };
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
pub(crate) fn parse_tags(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in s.split([',', '\n']) {
        let t = raw.trim();
        if !t.is_empty() && !out.iter().any(|e| e.eq_ignore_ascii_case(t)) {
            out.push(t.to_string());
        }
    }
    out
}

/// Add `draft`'s tags (comma-split to support batch pasting). Returns whether
/// anything was added, plus the canonical spelling of every segment that was
/// already present (for the amber "already added" chip flash).
fn add_tags_reporting_dups(tags: &mut Vec<String>, draft: &str) -> (bool, Vec<String>) {
    let mut changed = false;
    let mut dups: Vec<String> = Vec::new();
    for raw in draft.split(',') {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        match tags.iter().position(|e| e.eq_ignore_ascii_case(t)) {
            Some(i) => {
                if !dups.contains(&tags[i]) {
                    dups.push(tags[i].clone());
                }
            }
            None => {
                tags.push(t.to_string());
                changed = true;
            }
        }
    }
    (changed, dups)
}

/// Shared Enter / Add-button handler: add the draft's tags, save on change,
/// and flash any chips that were already present. Returns whether the draft
/// was acted on (something added or duplicates flagged). It's cleared only
/// once its tags are saved: a refused save (the .txt changed on disk, or
/// can't be read or written) keeps it, so trying again is one Enter away
/// instead of retyping it.
fn commit_draft(
    state: &mut TagManagerState,
    tags: &mut Vec<String>,
    path: &Path,
    current_tags: &mut String,
) -> bool {
    let before = tags.len();
    let (changed, dups) = add_tags_reporting_dups(tags, &state.draft);
    let mut saved = true;
    if changed {
        let added = tags.len() - before;
        saved = save(state, path, tags, current_tags);
        if saved {
            set_change_status(state, "Added", added, tags.len());
        }
    }
    let had_dups = !dups.is_empty();
    if had_dups {
        state.dup_flash = Some((dups.into_iter().collect(), Instant::now()));
    }
    let acted = changed || had_dups;
    if acted && saved {
        state.draft.clear();
    }
    acted
}

/// Serialize tags back to the comma-separated sidecar form.
fn serialize_tags(tags: &[String]) -> String {
    tags.join(", ")
}

/// Apply the working tag list to the selected file's sidecar and update the
/// shared buffer with the result. Only what changed is edited in the file
/// (see `sidecar::apply_tags`), so captions keep their sentences. False —
/// with an error status, nothing written — when the existing sidecar can't
/// be read.
///
/// `tags` is the shown list (parsed from `current_tags`) with the user's
/// change applied, and saving it makes the file hold exactly that list. So
/// when the file no longer holds what `current_tags` showed — it was changed
/// in the gallery popup or another program, or a failed read left the buffer
/// empty — nothing is written (that would undo those changes or wipe the
/// file): the buffer is reloaded and the user asked to try again.
fn save(state: &mut TagManagerState, path: &Path, tags: &[String], current_tags: &mut String) -> bool {
    match read_sidecar_text(path) {
        Some(on_disk) if parse_tags(&on_disk) != parse_tags(current_tags) => {
            *current_tags = on_disk;
            state.status_msg = "The tags changed on disk — reloaded, try again".to_string();
            state.status_is_error = true;
            return false;
        }
        Some(_) => {}
        None => {
            state.status_msg = "Couldn't read this image's .txt — left it untouched".to_string();
            state.status_is_error = true;
            return false;
        }
    }
    match write_sidecar(path, tags) {
        Some(text) => {
            *current_tags = text;
            true
        }
        None => {
            state.status_msg = "Couldn't save this image's .txt".to_string();
            state.status_is_error = true;
            false
        }
    }
}

/// A file's sidecar text as it is on disk (`None` when it exists but can't
/// be read).
fn read_sidecar_text(path: &Path) -> Option<String> {
    let txt = crate::right_details::sidecar_txt(path);
    crate::sidecar::read(&txt).ok().map(|sc| sc.text)
}

/// Read and parse a file's sidecar tags from disk (`None` when the sidecar
/// exists but can't be read — callers must not write over it).
fn read_sidecar(path: &Path) -> Option<Vec<String>> {
    let txt = crate::right_details::sidecar_txt(path);
    crate::sidecar::read(&txt).ok().map(|sc| parse_tags(&sc.text))
}

/// Make a file's sidecar hold `tags`, editing only what changed and keeping
/// its encoding. Returns the new file text, or `None` (nothing written) when
/// the existing sidecar can't be read. An empty list never creates a file.
fn write_sidecar(path: &Path, tags: &[String]) -> Option<String> {
    let txt = crate::right_details::sidecar_txt(path);
    let sc = crate::sidecar::read(&txt).ok()?;
    let text = crate::sidecar::apply_tags(&sc.text, tags);
    let unchanged = text == sc.text && (sc.existed || text.is_empty());
    if !unchanged && crate::sidecar::write(&txt, &text, sc.encoding).is_err() {
        return None;
    }
    Some(text)
}

/// An image's tags as its .txt holds them now (`None` when the .txt exists
/// but can't be read) — for the AI chat's Tag images mode.
pub(crate) fn read_tags(path: &Path) -> Option<Vec<String>> {
    read_sidecar(path)
}

/// Write `tags` to an image's .txt the way a manual edit does (only what
/// changed, same encoding). Returns the new file text, or `None` when the
/// existing .txt can't be read (nothing written).
pub(crate) fn write_tags(path: &Path, tags: &[String]) -> Option<String> {
    write_sidecar(path, tags)
}

/// The AI chat just changed the tags of the image on show (or undid such a
/// change): tint the tags it added violet (like the AI tagger's; an undo
/// clears the tint) and flash the change in the header.
pub(crate) fn ai_changed_tags(state: &mut TagManagerState, before: &[String], after: &[String], undo: bool) {
    let added: HashSet<String> = after
        .iter()
        .filter(|t| !before.iter().any(|b| b.eq_ignore_ascii_case(t)))
        .cloned()
        .collect();
    let removed = before.iter().filter(|t| !after.iter().any(|a| a.eq_ignore_ascii_case(t))).count();
    state.selected_tags.clear();
    let changed = added.len() + removed;
    state.ai_new_tags = if undo { HashSet::new() } else { added };
    if changed > 0 {
        set_change_status(state, if undo { "Undid" } else { "AI changed" }, changed, after.len());
    }
}

/// Add `new_tags` to every loaded file's sidecar, skipping files that already
/// have them all (case-insensitive). Returns the number of files changed and
/// the number skipped because their sidecar couldn't be read.
fn add_to_all(all_images: &[PathBuf], new_tags: &[String]) -> (usize, usize) {
    let mut count = 0;
    let mut skipped = 0;
    for img in all_images {
        let Some(mut existing) = read_sidecar(img) else {
            skipped += 1;
            continue;
        };
        let mut changed = false;
        for t in new_tags {
            if !existing.iter().any(|e| e.eq_ignore_ascii_case(t)) {
                existing.push(t.clone());
                changed = true;
            }
        }
        if changed {
            match write_sidecar(img, &existing) {
                Some(_) => count += 1,
                None => skipped += 1,
            }
        }
    }
    (count, skipped)
}

#[cfg(test)]
mod tests {
    use super::{add_fonts, ai_bar, save, show, TagManagerState, BOLD, CHIP_HINT, DRAFT_HINT, MEDIUM, SEMIBOLD};
    use crate::slider::rig::Rig;
    use eframe::egui;
    use std::fs;
    use std::path::PathBuf;

    /// Where the view is drawn on the rig's screen: the column the right
    /// panel gives it, 376 points wide.
    const LEFT: f32 = 20.0;
    const TOP: f32 = 4.0;
    const WIDTH: f32 = 376.0;

    /// The Tag Manager in a column `height` tall, on a.png of two pictures
    /// in `dir` (a.txt holds its tags; b.png has none).
    struct View {
        /// The view's own faces are installed (the first frame does it,
        /// as the app does at startup: they're there from the next one).
        fonts: bool,
        tm: TagManagerState,
        tags: String,
        dir: PathBuf,
        height: f32,
        /// Draw the AI bar alone.
        bar_only: bool,
        /// What the view took of its column, last frame.
        used: egui::Rect,
    }

    fn scene(ui: &mut egui::Ui, v: &mut View) {
        if !v.fonts {
            let mut fonts = egui::FontDefinitions::default();
            add_fonts(&mut fonts);
            ui.ctx().set_fonts(fonts);
            v.fonts = true;
            return;
        }
        let column = egui::Rect::from_min_size(egui::pos2(LEFT, TOP), egui::vec2(WIDTH, v.height));
        let image = v.dir.join("a.png");
        let images = [image.clone(), v.dir.join("b.png")];
        v.used = ui
            .scope_builder(egui::UiBuilder::new().max_rect(column), |ui| {
                if v.bar_only {
                    ai_bar(ui, &mut v.tm, Some(&image));
                } else {
                    show(ui, &mut v.tm, Some(&image), &mut v.tags, &images);
                }
                ui.min_rect()
            })
            .inner;
    }

    /// The view `height` tall on a picture tagged `tags`, settled. `name`
    /// keeps the tests' folders apart.
    fn view(name: &str, height: f32, tags: &str) -> Rig<View> {
        let dir = std::env::temp_dir().join(format!("ctf_tag_manager_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.png"), "").unwrap();
        fs::write(dir.join("b.png"), "").unwrap();
        fs::write(dir.join("a.txt"), tags).unwrap();
        let view = View {
            fonts: false,
            tm: TagManagerState::default(),
            tags: tags.to_string(),
            dir,
            height,
            bar_only: false,
            used: egui::Rect::NOTHING,
        };
        let mut rig = Rig::new(view, scene);
        rig.idle();
        rig
    }

    fn done(rig: Rig<View>) {
        let _ = fs::remove_dir_all(&rig.state.dir);
    }

    /// Where `text` is painted (once).
    fn text(rig: &Rig<View>, text: &str) -> egui::Rect {
        let found = rig.texts(text);
        assert_eq!(found.len(), 1, "{text:?} is painted {} times", found.len());
        found[0]
    }

    /// Two rows of chips in the column.
    const SIX: &str = "cat, orange tabby, sitting, window sill, indoors, sunlight";

    /// The rows over and under the tag box add up to the column's height,
    /// whatever is under it: the last one ends where the card does. (The box
    /// takes what they leave; a row a point taller than the sum assumes
    /// would push the text box past the card's edge.)
    #[test]
    fn the_view_fills_its_card_and_no_more() {
        let mut rig = view("fills", 350.0, SIX);
        let fills = |rig: &Rig<View>, what: &str| {
            let used = rig.state.used;
            assert!((used.top() - TOP).abs() < 0.01 && (used.height() - 350.0).abs() < 0.01, "{what}: the view took {used:?}");
        };
        // With room for two rows of chips over it, the hint line is there.
        text(&rig, CHIP_HINT);
        fills(&rig, "with the hint");

        // Tags selected: the bar has the hint's place.
        rig.state.tm.selected_tags.insert("cat".into());
        rig.idle();
        text(&rig, "1 selected");
        assert!(rig.texts(CHIP_HINT).is_empty());
        fills(&rig, "with the selection bar");

        // Asking before Add to all: the hint gives way to the question.
        rig.state.tm.selected_tags.clear();
        rig.state.tm.draft = "whiskers".into();
        rig.state.tm.confirm_add_all = true;
        rig.idle();
        text(&rig, "Add to all 2 files?");
        assert!(rig.texts(CHIP_HINT).is_empty());
        fills(&rig, "with the confirmation");
        done(rig);
    }

    /// In a window at its smallest the view has about 300 points. Idle, that
    /// holds everything but the hint line. With tags selected it doesn't:
    /// the view scrolls inside its card rather than run past it.
    #[test]
    fn too_short_for_it_all_the_view_scrolls_inside_its_card() {
        let mut rig = view("scrolls", 300.0, SIX);
        assert!(rig.texts(CHIP_HINT).is_empty());
        text(&rig, DRAFT_HINT);
        assert!((rig.state.used.height() - 300.0).abs() < 0.01, "idle, the view took {:?}", rig.state.used);

        rig.state.tm.selected_tags.insert("cat".into());
        rig.idle();
        text(&rig, "1 selected");
        assert!((rig.state.used.height() - 300.0).abs() < 0.01, "selecting, the view took {:?}", rig.state.used);
        // Everything it paints is cut off at the column's end (give or take
        // the margin egui clips with).
        fn walk(shape: &egui::Shape, clip: egui::Rect, texts: &mut Vec<(String, egui::Rect)>) {
            match shape {
                egui::Shape::Text(t) => texts.push((t.galley.text().to_string(), clip)),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, clip, texts)),
                _ => {}
            }
        }
        let mut texts = vec![];
        rig.shapes.iter().for_each(|c| walk(&c.shape, c.clip_rect, &mut texts));
        assert!(texts.iter().any(|(t, _)| t == "Add tags"));
        for (t, clip) in &texts {
            assert!(clip.bottom() <= TOP + 300.0 + 4.0, "{t:?} may be painted down to {}", clip.bottom());
        }
        done(rig);
    }

    /// A selected chip is the size it was: no other chip moves when one is
    /// clicked, and what's under the box stays where it was.
    #[test]
    fn selecting_a_tag_moves_no_chip() {
        let mut rig = view("selecting", 350.0, SIX);
        let chips = |rig: &Rig<View>| SIX.split(", ").map(|t| text(rig, t)).collect::<Vec<_>>();
        let before = chips(&rig);
        let draft = text(&rig, DRAFT_HINT);
        assert!(before[5].top() > before[0].bottom(), "two rows of chips: {before:?}");

        rig.click(before[1].center());
        rig.idle();
        assert!(rig.state.tm.selected_tags.contains("orange tabby"));
        text(&rig, "1 selected");
        assert_eq!(chips(&rig), before);
        assert_eq!(text(&rig, DRAFT_HINT), draft);
        done(rig);
    }

    /// The status has one line of the header, between the orb and the
    /// settings button, and nothing under the header moves for it. The app's
    /// own longer messages show whole there (set a little smaller); a longer
    /// one still is cut short (it shows whole on hover).
    #[test]
    fn a_long_status_keeps_to_one_line_of_the_header() {
        let mut rig = view("status", 350.0, SIX);
        let short = text(&rig, "Loaded 6 tags");
        let bar = text(&rig, "AI model");
        // Whether the status is painted cut short.
        fn cut(rig: &Rig<View>, status: &str) -> bool {
            fn find(shape: &egui::Shape, text: &str) -> Option<bool> {
                match shape {
                    egui::Shape::Text(t) if t.galley.text() == text => Some(t.galley.elided),
                    egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| find(s, text)),
                    _ => None,
                }
            }
            rig.shapes.iter().find_map(|c| find(&c.shape, status)).expect("the status is painted")
        }
        let words = "Load model: Load model from /home/someone/.local/share/Clarity TagFlow/tools/pixai-tagger-v0.9/model.onnx failed: Protobuf parsing failed.";
        let unbroken = "x".repeat(300);
        for (status, whole) in [
            ("The image was moved or deleted — tags not saved", true),
            ("Couldn't read this image's .txt — left it untouched", true),
            ("The tags changed on disk — reloaded, try again", true),
            (words, false),
            (unbroken.as_str(), false),
        ] {
            rig.state.tm.status_msg = status.to_string();
            rig.state.tm.status_is_error = true;
            rig.idle();
            let line = text(&rig, status);
            // (The settings button is the last 36 points; 10 are kept clear before it.)
            assert!(line.right() <= LEFT + WIDTH - 46.0, "{status:?} runs to {}", line.right());
            assert!(line.height() <= short.height() + 2.0, "{status:?} is {} tall (one line is {})", line.height(), short.height());
            assert_eq!(cut(&rig, status), !whole, "{status:?}");
            assert_eq!(text(&rig, "AI model"), bar);
        }
        done(rig);
    }

    /// A model's name is cut short where the picker's chevron begins: it
    /// never runs under the chevron or the Auto-tag button.
    #[test]
    fn a_long_model_name_stops_before_the_chevron() {
        let mut rig = view("model", 350.0, SIX);
        rig.state.bar_only = true;
        let name = "wd-eva02-large-tagger-v3-with-a-name-that-goes-on-for-quite-a-while-longer-than-the-bar";
        rig.state.tm.ai_model = name.to_string();
        rig.idle();
        let (name, button) = (text(&rig, name), text(&rig, "Auto-tag"));
        // Between the name's end and the button's label: the chevron's 30
        // points, the gap between picker and button, and the button's icon.
        assert!(name.right() <= button.left() - 60.0, "the name runs to {}, the button's label starts at {}", name.right(), button.left());
        assert!(name.right() > button.left() - 100.0, "the name has the room there is: it ends at {}", name.right());
        done(rig);
    }

    /// Remove (in the selection bar) takes the selected tags out, in the
    /// list and in the picture's .txt.
    #[test]
    fn remove_takes_the_selected_tags_out_of_the_file() {
        let mut rig = view("remove", 350.0, "cat, orange tabby, sitting");
        rig.state.tm.selected_tags.insert("orange tabby".into());
        rig.idle();
        rig.click(text(&rig, "Remove").center());
        rig.idle();
        assert_eq!(fs::read_to_string(rig.state.dir.join("a.txt")).unwrap(), "cat, sitting");
        assert_eq!(rig.state.tags, "cat, sitting");
        assert!(rig.state.tm.selected_tags.is_empty());
        assert!(rig.texts("orange tabby").is_empty() && rig.texts("1 selected").is_empty());
        done(rig);
    }

    /// Clear (in the selection bar) lets go of the selection and changes
    /// nothing else.
    #[test]
    fn clear_lets_go_of_the_selection() {
        let mut rig = view("clear", 350.0, "cat, orange tabby, sitting");
        rig.state.tm.selected_tags.insert("orange tabby".into());
        rig.idle();
        rig.click(text(&rig, "Clear").center());
        rig.idle();
        assert!(rig.state.tm.selected_tags.is_empty());
        assert_eq!(fs::read_to_string(rig.state.dir.join("a.txt")).unwrap(), "cat, orange tabby, sitting");
        text(&rig, "orange tabby");
        done(rig);
    }

    /// Enter in the text box adds what's typed, split at the commas, less
    /// the tags already there.
    #[test]
    fn enter_adds_the_typed_tags() {
        let mut rig = view("enter", 350.0, "cat");
        rig.click(text(&rig, DRAFT_HINT).center());
        rig.idle();
        rig.type_text("orange tabby, cat, sitting");
        rig.key(egui::Key::Enter);
        rig.idle();
        assert_eq!(fs::read_to_string(rig.state.dir.join("a.txt")).unwrap(), "cat, orange tabby, sitting");
        assert_eq!(rig.state.tm.draft, "");
        text(&rig, "orange tabby");
        text(&rig, "sitting");
        done(rig);
    }

    /// "Add to all images" writes to every loaded picture, so it asks first:
    /// nothing is written until Confirm, and Cancel takes the question away.
    #[test]
    fn add_to_all_asks_first() {
        let mut rig = view("add_to_all", 350.0, "cat");
        let (a, b) = (rig.state.dir.join("a.txt"), rig.state.dir.join("b.txt"));
        // With nothing typed the button only shows.
        rig.click(text(&rig, "Add to all images").center());
        rig.idle();
        assert!(!rig.state.tm.confirm_add_all);

        rig.state.tm.draft = "sitting".into();
        rig.idle();
        rig.click(text(&rig, "Add to all images").center());
        rig.idle();
        text(&rig, "Add to all 2 files?");
        rig.click(text(&rig, "Cancel").center());
        rig.idle();
        assert!(rig.texts("Add to all 2 files?").is_empty());
        assert_eq!(fs::read_to_string(&a).unwrap(), "cat");
        assert!(!b.exists());

        rig.click(text(&rig, "Add to all images").center());
        rig.idle();
        rig.click(text(&rig, "Confirm").center());
        rig.idle();
        assert_eq!(fs::read_to_string(&a).unwrap(), "cat, sitting");
        assert_eq!(fs::read_to_string(&b).unwrap(), "sitting");
        assert_eq!(rig.state.tags, "cat, sitting");
        assert_eq!(rig.state.tm.draft, "");
        assert!(rig.texts("Add to all 2 files?").is_empty());
        done(rig);
    }

    /// Each weight is a family of its own with the app's other fonts behind
    /// it, for the characters Manrope hasn't got. (Text laid out in a family
    /// that isn't there panics.)
    #[test]
    fn every_weight_has_the_apps_fonts_behind_it() {
        let mut fonts = egui::FontDefinitions::default();
        let proportional = fonts.families[&egui::FontFamily::Proportional].clone();
        add_fonts(&mut fonts);
        for name in [MEDIUM, SEMIBOLD, BOLD] {
            let family = &fonts.families[&egui::FontFamily::Name(name.into())];
            assert_eq!(family[0], name);
            assert_eq!(family[1..], proportional[..]);
            assert!(fonts.font_data.contains_key(name));
        }
    }

    /// A Tag Manager change is written only over the text it was made from:
    /// when the file changed underneath (another view, another program, or
    /// a failed read that left the buffer empty), nothing is written and the
    /// buffer is reloaded instead.
    #[test]
    fn save_refuses_a_stale_buffer() {
        let dir = std::env::temp_dir().join(format!("ctf_tag_manager_save_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let img = dir.join("a.png");
        let txt = dir.join("a.txt");
        fs::write(&txt, "a, b, c").unwrap();
        let mut state = TagManagerState::default();

        // The buffer is empty (as after a failed read): no write.
        let mut buffer = String::new();
        assert!(!save(&mut state, &img, &["1girl".to_string()], &mut buffer));
        assert_eq!(fs::read_to_string(&txt).unwrap(), "a, b, c");
        assert_eq!(buffer, "a, b, c", "reloaded from disk");

        // From the current text, the change is written.
        let tags: Vec<String> = ["a", "b", "c", "1girl"].iter().map(|t| t.to_string()).collect();
        assert!(save(&mut state, &img, &tags, &mut buffer));
        assert_eq!(fs::read_to_string(&txt).unwrap(), "a, b, c, 1girl");
        assert_eq!(buffer, "a, b, c, 1girl");
        let _ = fs::remove_dir_all(&dir);
    }
}
