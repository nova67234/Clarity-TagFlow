//! The right details panel — a side panel for viewing and editing image sidecar metadata.
//! Currently tailored exclusively to support `.txt` sidecars.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Instant, SystemTime};

use eframe::egui;

use crate::theme::*;
use crate::card_frame;
use crate::tag_manager::TagManagerState;

/// Width of the right details panel.
pub const PANEL_WIDTH: f32 = 420.0;

/// Which view the right panel is showing, chosen from the menu dropdown.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum RightView {
    /// The image details + tag-editing actions (the original view).
    #[default]
    Details,
    /// The Tag Manager settings UI (replaces the details view).
    TagManager,
}

/// Read-only file metadata shown in the bottom "Image Details" card. A Rust
/// port of the fields surfaced by terminus2's `ImageDetails` Swing panel.
pub struct ImageMeta {
    name: String,
    file_type: String,
    dimensions: String,
    size: String,
    date: String,
    /// A small palette of dominant colors extracted from the image.
    colors: Vec<egui::Color32>,
}

impl Default for ImageMeta {
    fn default() -> Self {
        Self {
            name: "---".into(),
            file_type: "---".into(),
            dimensions: "---".into(),
            size: "---".into(),
            date: "---".into(),
            colors: Vec::new(),
        }
    }
}

/// Maintains the UI state for the right panel, such as the loaded text buffer
/// and whether the user is currently in edit mode.
pub struct RightPanelState {
    pub current_tags: String,
    pub is_editing: bool,

    /// Which view the panel currently shows (Details vs Tag Manager).
    pub view: RightView,

    // --- Delete Confirmation State ---
    pub show_delete_confirm: bool,
    pub skip_delete_confirm: bool,

    last_path: Option<PathBuf>,
    /// Cached metadata for the selected image, recomputed only on selection change.
    meta: ImageMeta,
    /// When edit mode was last entered, used to play a brief "ready to edit"
    /// flash on the tag box. `None` once the flash has finished / never started.
    edit_flash_start: Option<Instant>,
    /// When Copy was last clicked, plus whether it actually copied text (`true`)
    /// or there was nothing to copy (`false`). Drives the Copy button's flash.
    copy_flash: Option<(Instant, bool)>,
    /// When Save was last clicked, plus whether the write succeeded (`true`) or
    /// failed (`false`). Drives the Edit/Save button's green/red flash.
    save_flash: Option<(Instant, bool)>,
    /// Receiver to listen for background metadata loading
    meta_rx: Option<mpsc::Receiver<ImageMeta>>,
    /// Bumped on every new metadata load. A background load reads this and
    /// aborts *before* its heavy full-resolution decode if the selection has
    /// already moved on, so fast navigation can't pile up redundant decodes
    /// that starve the centre viewer and make it flicker.
    meta_gen: Arc<AtomicU64>,
}

impl Default for RightPanelState {
    fn default() -> Self {
        Self {
            current_tags: String::new(),
            is_editing: false,
            view: RightView::default(),
            show_delete_confirm: false,
            skip_delete_confirm: false,
            last_path: None,
            meta: ImageMeta::default(),
            edit_flash_start: None,
            copy_flash: None,
            save_flash: None,
            meta_rx: None,
            meta_gen: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Actions that the right panel can request the main app to perform
/// (since the panel itself doesn't own the image list).
pub enum RightPanelAction {
    None,
    /// Request to delete the currently selected image and its sidecar.
    DeleteCurrent,
    /// Request to move the currently selected image and its sidecar.
    MoveCurrent,
}

/// Render the right details panel.
pub fn show(
    ui: &mut egui::Ui,
    state: &mut RightPanelState,
    current_image: Option<&Path>,
    confirm_before_delete: &mut bool,
    tag_manager: &mut TagManagerState,
    all_images: &[PathBuf],
) -> RightPanelAction {
    // 0. Non-blocking check to see if our background thread finished calculating
    if let Some(rx) = &state.meta_rx {
        if let Ok(meta) = rx.try_recv() {
            state.meta = meta;
            state.meta_rx = None; // Finished!
        }
    }

    // 1. Sync the text box state when the selected image changes.
    if state.last_path.as_deref() != current_image {
        state.last_path = current_image.map(|p| p.to_path_buf());
        state.is_editing = false;
        state.show_delete_confirm = false; // Reset delete prompt when navigating away
        state.edit_flash_start = None;
        state.copy_flash = None;
        state.save_flash = None;
        state.meta_rx = None; // Cancel any pending task

        if let Some(path) = current_image {
            let txt_path = sidecar_txt(path);
            state.current_tags = std::fs::read_to_string(&txt_path).unwrap_or_default();

            // Set temporary loading state for the card
            state.meta = ImageMeta {
                name: "Loading...".into(), file_type: "...".into(),
                dimensions: "...".into(), size: "...".into(),
                date: "...".into(), colors: vec![]
            };

            // Spawn a background thread to calculate metadata and heavy colors
            let (tx, rx) = mpsc::channel();
            state.meta_rx = Some(rx);

            // This load's generation. A newer selection bumps the shared counter,
            // letting this thread bail before the expensive decode.
            let generation = state.meta_gen.fetch_add(1, Ordering::SeqCst) + 1;
            let meta_gen = Arc::clone(&state.meta_gen);

            let path_clone = path.to_path_buf();
            let ctx = ui.ctx().clone();

            thread::spawn(move || {
                // Skip the heavy decode entirely if the selection already moved on.
                if meta_gen.load(Ordering::SeqCst) != generation {
                    return;
                }
                let meta = load_meta(&path_clone);
                // Deliver + repaint only if this is still the current selection.
                if meta_gen.load(Ordering::SeqCst) == generation && tx.send(meta).is_ok() {
                    ctx.request_repaint();
                }
            });

        } else {
            state.current_tags.clear();
            state.meta = ImageMeta::default();
        }
    }

    let mut action = RightPanelAction::None;

    // 2. Render the actual panel.
    egui::Panel::right("right_details")
        .resizable(false)
        .exact_size(PANEL_WIDTH)
        .show_separator_line(false)
        // Trim the top margin so the card rises up close to the top bar
        // (left/right/bottom keep the standard 10px breathing room).
        .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            // Trim the card's bottom padding so the Image Details box can sit
            // close to the bottom edge of the panel (other sides keep 12).
            card_frame(22)
                .inner_margin(egui::Margin { left: 12, right: 12, top: 12, bottom: 12 })
                .show(ui, |ui| {
                    // Force the inner UI to claim the entire vertical height of the panel
                    let height = ui.available_height();
                    ui.set_min_height(height);

                    // Menu icon pinned to the card's top-right corner.
                    // Using ui.put allows us to place it exactly on the right edge
                    // without pushing down the centered text below it.
                    let anchor = ui.cursor().min;
                    let width = ui.available_width();
                    let btn_size = 24.0;

                    let gear_rect = egui::Rect::from_min_size(
                        egui::pos2(anchor.x + width - btn_size, anchor.y),
                        egui::vec2(btn_size, btn_size),
                    );

                    let menu_icon = egui::Image::new(egui::include_image!("../icons/menu.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(MUTED);

                    let menu_resp = ui.put(
                        gear_rect,
                        egui::Button::image(menu_icon).frame(false)
                    ).on_hover_text("Menu");

                    // The fully functional Dropdown
                    egui::Popup::menu(&menu_resp)
                        .align(egui::RectAlign::BOTTOM_END)
                        .show(|ui| {
                            ui.set_min_width(160.0);

                            // Tweak visuals slightly so the menu looks good in dark mode
                            let radius = egui::CornerRadius::same(6);
                            ui.visuals_mut().widgets.inactive.corner_radius = radius;
                            ui.visuals_mut().widgets.hovered.corner_radius = radius;

                            if ui
                                .selectable_label(state.view == RightView::Details, "Details & Actions")
                                .clicked()
                            {
                                state.view = RightView::Details;
                                ui.close(); // Fixed deprecation
                            }
                            if ui
                                .selectable_label(state.view == RightView::TagManager, "Tag Manager")
                                .clicked()
                            {
                                state.view = RightView::TagManager;
                                ui.close(); // Fixed deprecation
                            }
                        });

                    // --- Swap Views ---
                    // The Tag Manager view completely replaces the Details & Actions UI,
                    // but stays constrained perfectly within the 420px width and 22px rounded box.
                    if state.view == RightView::TagManager {
                        crate::tag_manager::show(ui, tag_manager, current_image, &mut state.current_tags, all_images);
                    } else if let Some(img_path) = current_image {
                        // --- FOOTER SECTION (Strictly Bottom Anchored) ---
                        egui::Panel::bottom("right_footer")
                            .resizable(false)
                            .show_separator_line(false)
                            .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
                            .show_inside(ui, |ui| {

                                ui.add_space(12.0);

                                // 1. Action Buttons — four equal, fixed-size buttons that
                                //    span the full tag-box width edge to edge. Fixed widths
                                //    mean toggling Edit Text/Save never resizes or shifts them.
                                ui.horizontal(|ui| {
                                    let gap = 8.0;
                                    ui.spacing_mut().item_spacing.x = gap;
                                    let btn_w = (ui.available_width() - gap * 3.0) / 4.0;
                                    let size = egui::vec2(btn_w, 35.0);
                                    let label = |t: &str| egui::RichText::new(t).size(15.0);

                                    // Rounder corners than the default (8) for all the buttons.
                                    let r = egui::CornerRadius::same(12);
                                    ui.visuals_mut().widgets.inactive.corner_radius = r;
                                    ui.visuals_mut().widgets.hovered.corner_radius = r;
                                    ui.visuals_mut().widgets.active.corner_radius = r;

                                    // Copy flashes green when it copies tags, or amber
                                    // ("warning") when there's nothing to copy — then fades back.
                                    let mut copy_btn = egui::Button::new(label("Copy"));
                                    if let Some((start, ok)) = state.copy_flash {
                                        let elapsed = start.elapsed().as_secs_f32();
                                        if elapsed < FLASH_SECS {
                                            let intensity = 1.0 - elapsed / FLASH_SECS;
                                            let target = if ok { FLASH_GREEN } else { FLASH_AMBER };
                                            let base = ui.visuals().widgets.inactive.weak_bg_fill;
                                            copy_btn = copy_btn.fill(lerp_color(base, target, intensity));
                                            ui.ctx().request_repaint(); // animate the fade
                                        }
                                    }

                                    if ui
                                        .add_sized(size, copy_btn)
                                        .on_hover_text("Copy tags to clipboard")
                                        .clicked()
                                    {
                                        let ok = !state.current_tags.trim().is_empty();
                                        if ok {
                                            ui.ctx().copy_text(state.current_tags.clone());
                                        }
                                        state.copy_flash = Some((Instant::now(), ok));
                                    }

                                    // Edit/Save slot: flashes green when a save succeeds,
                                    // red when the write fails (stays in edit mode to retry).
                                    let mut edit_btn =
                                        egui::Button::new(label(if state.is_editing { "Save" } else { "Edit Text" }));
                                    if let Some((start, ok)) = state.save_flash {
                                        let elapsed = start.elapsed().as_secs_f32();
                                        if elapsed < FLASH_SECS {
                                            let intensity = 1.0 - elapsed / FLASH_SECS;
                                            let target = if ok { FLASH_GREEN } else { FLASH_RED };
                                            let base = ui.visuals().widgets.inactive.weak_bg_fill;
                                            edit_btn = edit_btn.fill(lerp_color(base, target, intensity));
                                            ui.ctx().request_repaint();
                                        }
                                    }

                                    if ui.add_sized(size, edit_btn).clicked() {
                                        if state.is_editing {
                                            let txt_path = sidecar_txt(img_path);
                                            let ok = std::fs::write(&txt_path, &state.current_tags).is_ok();
                                            if ok {
                                                state.is_editing = false; // saved — back to view mode
                                            }
                                            state.save_flash = Some((Instant::now(), ok));
                                        } else {
                                            state.is_editing = true;
                                            state.edit_flash_start = Some(Instant::now());
                                            state.save_flash = None; // clear any stale save flash
                                        }
                                    }

                                    if ui.add_sized(size, egui::Button::new(label("Move"))).clicked() {
                                        action = RightPanelAction::MoveCurrent;
                                    }

                                    let danger_bg = egui::Color32::from_rgb(180, 40, 40);
                                    let delete_btn =
                                        egui::Button::new(label("Delete").color(egui::Color32::WHITE))
                                            .fill(danger_bg);

                                    if ui.add_sized(size, delete_btn).clicked() {
                                        // Confirm first, unless confirmations are disabled.
                                        if *confirm_before_delete {
                                            state.show_delete_confirm = true;
                                            state.skip_delete_confirm = false; // fresh checkbox
                                        } else {
                                            action = RightPanelAction::DeleteCurrent;
                                        }
                                    }
                                });

                                ui.add_space(10.0);

                                // 2. Image Details Card (Very Bottom) — no trailing
                                //    space so it sits flush near the panel's bottom edge.
                                image_details_section(ui, &state.meta);
                            });

                        // --- MAIN CONTENT SECTION (Stretches to fill middle gap) ---
                        egui::CentralPanel::default()
                            .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
                            .show_inside(ui, |ui| {

                                // 1. Header
                                ui.vertical_centered(|ui| {
                                    ui.add_space(4.0);
                                    ui.heading(egui::RichText::new("Details & Actions").color(TEXT).strong());
                                    ui.add_space(8.0);
                                });

                                // 2. Tags Label
                                ui.horizontal(|ui| {
                                    // Tight gap between the icon and the label.
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    let icon = egui::include_image!("../icons/tag.svg");
                                    ui.add(
                                        egui::Image::new(icon)
                                            .fit_to_exact_size(egui::vec2(18.0, 18.0))
                                            .tint(TEXT),
                                    );
                                    ui.label(egui::RichText::new("Tags:").color(TEXT).strong());
                                });
                                ui.add_space(4.0);

                                // 3. Tags Text Area
                                let mut display_text = state.current_tags.clone();

                                ui.scope(|ui| {
                                    let radius = egui::CornerRadius::same(22);
                                    // `noninteractive` covers the view-mode box (interactive(false)),
                                    // the others cover edit mode — so both share the rounded edges.
                                    ui.visuals_mut().widgets.noninteractive.corner_radius = radius;
                                    ui.visuals_mut().widgets.inactive.corner_radius = radius;
                                    ui.visuals_mut().widgets.hovered.corner_radius = radius;
                                    ui.visuals_mut().widgets.active.corner_radius = radius;

                                    // "Ready to edit" flash: for a brief moment after entering
                                    // edit mode, pulse the box background toward the accent so the
                                    // user notices the box is now editable.
                                    if let Some(start) = state.edit_flash_start {
                                        let elapsed = start.elapsed().as_secs_f32();
                                        if elapsed < EDIT_FLASH_SECS {
                                            let t = elapsed / EDIT_FLASH_SECS;          // 0..1
                                            let envelope = 1.0 - t;                     // overall fade-out
                                            let osc = (t * std::f32::consts::PI * 2.0).sin().abs(); // two pulses
                                            let intensity = (envelope * osc).clamp(0.0, 1.0);
                                            ui.visuals_mut().extreme_bg_color =
                                                lerp_color(FIELD, ACCENT1, intensity * 0.55);
                                            ui.ctx().request_repaint(); // keep the animation smooth
                                        }
                                    }

                                    let mut text_edit = egui::TextEdit::multiline(&mut display_text)
                                        .desired_width(f32::INFINITY)
                                        .font(egui::TextStyle::Monospace)
                                        .margin(egui::Margin::same(12))
                                        // Only selectable/editable in edit mode — outside of it
                                        // the box just displays the tags (no clicking, no cursor,
                                        // no text selection). Click "Edit Text" to interact.
                                        .interactive(state.is_editing);

                                    if !state.is_editing {
                                        text_edit = text_edit.text_color(TEXT.gamma_multiply(0.8));
                                    }

                                    ui.add_sized(ui.available_size(), text_edit);
                                });

                                if state.is_editing {
                                    state.current_tags = display_text;
                                }
                            });

                    } else {
                        // --- Empty State ---
                        ui.vertical_centered(|ui| {
                            ui.add_space(4.0);
                            ui.heading(egui::RichText::new("Details & Actions").color(TEXT).strong());
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("No image selected").color(MUTED).size(13.0));
                        });
                    }
                });
        });

    // --- 3. Delete Confirmation UI (Smaller, Centered Modal) ---
    if state.show_delete_confirm {
        let mut close_dialog = false;
        let mut confirm_delete = false;

        egui::Window::new("Confirm Delete")
            .title_bar(false) // No title bar to keep the UI small and clean
            .resizable(false)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO) // Anchor directly in the middle of the screen
            .frame(card_frame(22)) // match the rest of the UI (PANEL fill, radius 22, shadow)
            .show(ui.ctx(), |ui| {
                ui.set_max_width(260.0); // Stop it from stretching too wide

                // Wrap inner elements in vertical_centered so text and buttons line up perfectly
                ui.vertical_centered(|ui| {
                    let warn = egui::include_image!("../icons/warning.svg");
                    ui.add(
                        egui::Image::new(warn)
                            .fit_to_exact_size(egui::vec2(32.0, 32.0))
                            .tint(egui::Color32::from_rgb(220, 160, 50)),
                    );
                    ui.add_space(8.0);

                    ui.label(
                        egui::RichText::new("Are you sure you want to delete this file?")
                            .size(15.0)
                            .strong()
                            .color(TEXT)
                    );

                    ui.add_space(16.0);

                    // Scope for checkbox to ensure it acts as a true square
                    // with slightly rounded 4px edges (instead of inheriting 12px pill curves)
                    ui.scope(|ui| {
                        let r = egui::CornerRadius::same(4);
                        ui.visuals_mut().widgets.inactive.corner_radius = r;
                        ui.visuals_mut().widgets.hovered.corner_radius = r;
                        ui.visuals_mut().widgets.active.corner_radius = r;
                        ui.visuals_mut().widgets.noninteractive.corner_radius = r;

                        ui.checkbox(&mut state.skip_delete_confirm, "Don't ask again");
                    });

                    ui.add_space(20.0);

                    ui.horizontal(|ui| {
                        let btn_w = 80.0;
                        let gap = 12.0;
                        let total_w = btn_w * 2.0 + gap;

                        // Push the buttons perfectly to the middle of the layout
                        ui.add_space((ui.available_width() - total_w) / 2.0);
                        ui.spacing_mut().item_spacing.x = gap;

                        // Add matching soft-corners to the inner action buttons
                        let r = egui::CornerRadius::same(8);
                        ui.visuals_mut().widgets.inactive.corner_radius = r;
                        ui.visuals_mut().widgets.hovered.corner_radius = r;
                        ui.visuals_mut().widgets.active.corner_radius = r;

                        if ui.add_sized(egui::vec2(btn_w, 30.0), egui::Button::new("Cancel")).clicked() {
                            close_dialog = true;
                        }

                        let danger_bg = egui::Color32::from_rgb(180, 40, 40);
                        let del_btn = egui::Button::new(
                            egui::RichText::new("Delete").color(egui::Color32::WHITE)
                        ).fill(danger_bg);

                        if ui.add_sized(egui::vec2(btn_w, 30.0), del_btn).clicked() {
                            confirm_delete = true;
                        }
                    });
                });
            });

        // Resolve actions after the UI block concludes
        if confirm_delete {
            action = RightPanelAction::DeleteCurrent;
            state.show_delete_confirm = false;
            // "Don't ask again" disables (and persists, via Settings) future prompts.
            if state.skip_delete_confirm {
                *confirm_before_delete = false;
            }
        } else if close_dialog {
            state.show_delete_confirm = false;
        }
    }

    action
}

// ---------------------------------------------------------------------------
// Image Details card
// ---------------------------------------------------------------------------

const DETAIL_LABEL_W: f32 = 110.0;
const DETAIL_ROW_VPAD: f32 = 3.0;

fn image_details_section(ui: &mut egui::Ui, meta: &ImageMeta) {
    ui.horizontal(|ui| {
        // Tight gap between the icon and the heading text.
        ui.spacing_mut().item_spacing.x = 4.0;
        let icon = egui::include_image!("../icons/image.svg");
        ui.add(
            egui::Image::new(icon)
                .fit_to_exact_size(egui::vec2(18.0, 18.0))
                .tint(TEXT),
        );
        ui.label(egui::RichText::new("Image Info").color(TEXT).strong().size(15.0));
    });
    ui.add_space(8.0);

    let frame = egui::Frame::new()
        .fill(FIELD)
        .corner_radius(egui::CornerRadius::same(22)) // match the tag box
        .inner_margin(egui::Margin::symmetric(16, 12))
        // Use the same soft light edge the tag box gets from egui's default field
        // border (the gentle highlight), instead of the very faint EDGE — so both
        // inset boxes look identical.
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke);

    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        detail_row(ui, "File Name:", &meta.name);
        detail_row(ui, "File Type:", &meta.file_type);
        detail_row(ui, "Dimensions:", &meta.dimensions);
        detail_row(ui, "File Size:", &meta.size);
        detail_row(ui, "Date Modified:", &meta.date);
        detail_color_row(ui, "Colors:", &meta.colors); // NEW COLOR ROW!
    });
}

fn detail_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.add_space(DETAIL_ROW_VPAD);
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(DETAIL_LABEL_W, 18.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(egui::RichText::new(label).color(TEXT).strong());
            },
        );

        let unknown = value == "---" || value == "Loading...";
        let color = if unknown { MUTED } else { TEXT };
        let resp = ui.add(egui::Label::new(egui::RichText::new(value).color(color)).truncate());
        if !unknown {
            resp.on_hover_text(value);
        }
    });
    ui.add_space(DETAIL_ROW_VPAD);
}

/// A specialized row that draws small rounded color squares instead of text.
fn detail_color_row(ui: &mut egui::Ui, label: &str, colors: &[egui::Color32]) {
    ui.add_space(DETAIL_ROW_VPAD);
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(DETAIL_LABEL_W, 18.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(egui::RichText::new(label).color(TEXT).strong());
            },
        );

        if colors.is_empty() {
            ui.label(egui::RichText::new("---").color(MUTED));
        } else {
            // CHANGED HERE: Use horizontal_wrapped so colors wrap to the next line if there are many!
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.spacing_mut().item_spacing.y = 6.0; // Add vertical spacing for wrapped rows
                for &color in colors {
                    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                    let corner_radius = egui::CornerRadius::same(4);
                    ui.painter().rect_filled(rect, corner_radius, color);
                    ui.painter().rect_stroke(
                        rect,
                        corner_radius,
                        egui::Stroke::new(1.0, EDGE),
                        egui::StrokeKind::Inside,
                    );
                }
            });
        }
    });
    ui.add_space(DETAIL_ROW_VPAD);
}

/// Read the metadata shown in the details card, including extracting dominant colors.
fn load_meta(path: &Path) -> ImageMeta {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("---")
        .to_string();

    let file_type = path
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| !e.is_empty())
        .map(|e| e.to_uppercase())
        .unwrap_or_else(|| "(unknown)".to_string());

    let (size, date) = match std::fs::metadata(path) {
        Ok(md) => {
            let size = human_bytes(md.len());
            let date = md
                .modified()
                .ok()
                .map(format_time)
                .unwrap_or_else(|| "---".to_string());
            (size, date)
        }
        Err(_) => ("---".to_string(), "---".to_string()),
    };

    // AVIF/HEIC/HEIF can't be read by the `image` crate's header reader or
    // `open()`, so decode them once via our pure-Rust path and reuse the result
    // for both the dimensions and the colour palette below.
    #[cfg(feature = "avif")]
    let avif_img: Option<image::DynamicImage> = {
        let is_avif = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "avif" | "heic" | "heif" | "dng" | "arw"))
            .unwrap_or(false);
        if is_avif {
            crate::avif::decode_avif(path).map(image::DynamicImage::ImageRgba8)
        } else {
            None
        }
    };
    #[cfg(not(feature = "avif"))]
    let avif_img: Option<image::DynamicImage> = None;

    let dimensions = if let Some(img) = &avif_img {
        format!("{} x {}", img.width(), img.height())
    } else {
        // Use orientation-aware dimensions so a portrait photo (EXIF orientation
        // 6/8) reports the size the user sees, matching the rotated display.
        crate::image_cache::oriented_dimensions(path)
            .map(|(w, h)| format!("{} x {}", w, h))
            .unwrap_or_else(|| "---".to_string())
    };

    // --- COLOR EXTRACTION V3 (Vibrancy & Saturation Weighted) ---
    let mut palette = Vec::new();

    let loaded = match avif_img {
        Some(img) => Some(img),
        None => image::open(path).ok(),
    };
    if let Some(img) = loaded {
        // Bumped to 128x128. We use 'Nearest' filter so thin neon light streaks
        // stay sharp and aren't blurred/blended into the dark background!
        let thumb = img.resize_exact(128, 128, image::imageops::FilterType::Nearest).to_rgba8();

        let mut buckets: std::collections::HashMap<[u8; 3], (u32, u32, u32, u32)> = std::collections::HashMap::new();

        // 1. Group pixels into buckets
        for pixel in thumb.pixels() {
            if pixel[3] < 128 { continue; } // Ignore transparent

            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // Ignore almost pure black and pure white so they don't eat up palette slots
            if r < 15 && g < 15 && b < 15 { continue; }
            if r > 240 && g > 240 && b > 240 { continue; }

            // Quantize by dividing by 16 for slightly better color fidelity
            let bucket_key = [r / 16, g / 16, b / 16];

            let entry = buckets.entry(bucket_key).or_insert((0, 0, 0, 0));
            entry.0 += 1;
            entry.1 += r as u32;
            entry.2 += g as u32;
            entry.3 += b as u32;
        }

        // 2. Score buckets based on a mix of FREQUENCY and COLORFULNESS
        let mut scored_buckets: Vec<_> = buckets.into_values().map(|(count, sr, sg, sb)| {
            let avg_r = (sr / count) as u8;
            let avg_g = (sg / count) as u8;
            let avg_b = (sb / count) as u8;

            // Calculate Chroma (how colorful/saturated the color is)
            let c_max = avg_r.max(avg_g).max(avg_b);
            let c_min = avg_r.min(avg_g).min(avg_b);
            let chroma = (c_max - c_min) as f32; // 0 to 255

            // VIBRANCY MULTIPLIER: Boost the "count" artificially if the color is highly saturated.
            // An exponentially weighted curve means a neon color gets up to a
            // massive multiplier compared to a dull gray!
            let weight = 1.0 + (chroma * chroma / 1000.0);

            // We also take the square root of the count so massive walls of background
            // color don't completely drown out smaller details
            let score = (count as f32).sqrt() * weight;

            (score, avg_r, avg_g, avg_b)
        }).collect();

        // Sort by our new score (highest score first)
        scored_buckets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut distinct_colors: Vec<[u8; 3]> = Vec::new();

        // 3. Extract TRUE average color, filtering out similar ones
        for (_score, avg_r, avg_g, avg_b) in scored_buckets {
            let mut is_distinct = true;

            for &c in &distinct_colors {
                let dr = avg_r as i32 - c[0] as i32;
                let dg = avg_g as i32 - c[1] as i32;
                let db = avg_b as i32 - c[2] as i32;

                // Perceptual distance formula (greens and reds matter more to human eyes than blue)
                let dist_sq = (dr * dr * 3) + (dg * dg * 4) + (db * db * 2);

                // Increased threshold to force maximum variety!
                // It will now actively refuse to pick 5 shades of the same dark blue.
                if dist_sq < 4000 {
                    is_distinct = false;
                    break;
                }
            }

            if is_distinct {
                distinct_colors.push([avg_r, avg_g, avg_b]);
                if distinct_colors.len() == 16 {
                    break;
                }
            }
        }

        // 4. Group final palette into a gorgeous rainbow layout (Hue, then Luminance)
        distinct_colors.sort_by_key(|c| {
            let r = c[0] as f32;
            let g = c[1] as f32;
            let b = c[2] as f32;

            // Rough Hue calculation
            let hue = if r >= g && r >= b { (g - b) / (r - g.min(b)).max(1.0) }
            else if g >= r && g >= b { 2.0 + (b - r) / (g - r.min(b)).max(1.0) }
            else { 4.0 + (r - g) / (b - r.min(g)).max(1.0) };

            let wrapped_hue = ((hue * 60.0) as i32).rem_euclid(360);
            let luminance = (r * 0.299 + g * 0.587 + b * 0.114) as i32;

            // Sort by Hue group, then brightness
            (wrapped_hue / 30, luminance)
        });

        palette = distinct_colors.into_iter().map(|c| {
            egui::Color32::from_rgb(c[0], c[1], c[2])
        }).collect();
    }

    ImageMeta { name, file_type, dimensions, size, date, colors: palette }
}

fn human_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{} B", bytes);
    }
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        return format!("{:.1} KB", kb);
    }
    let mb = kb / 1024.0;
    if mb < 1024.0 {
        return format!("{:.1} MB", mb);
    }
    let gb = mb / 1024.0;
    format!("{:.1} GB", gb)
}

pub(crate) fn sidecar_txt(img_path: &Path) -> PathBuf {
    img_path.with_extension("txt")
}

fn format_time(t: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Duration (seconds) of the "ready to edit" flash on the tag box.
const EDIT_FLASH_SECS: f32 = 0.8;

/// Duration (seconds) of the button result-flashes (Copy / Save).
const FLASH_SECS: f32 = 1.0;
/// Button flash colors: green = success, amber = nothing to do, red = failure.
const FLASH_GREEN: egui::Color32 = egui::Color32::from_rgb(46, 160, 67);
const FLASH_AMBER: egui::Color32 = egui::Color32::from_rgb(200, 145, 40);
const FLASH_RED: egui::Color32 = egui::Color32::from_rgb(200, 55, 55);

/// Linearly interpolate between two colors in sRGB component space.
fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
    )
}