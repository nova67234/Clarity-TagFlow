//! Tag Manager settings — a Rust port of terminus2's `TagManagerSettings` dialog,
//! opened from the Tag Manager panel's gear button.
//!
//! In-memory only for now: the values are held here but not yet persisted across
//! runs or wired into tagging behaviour (there's no AI tagger backend yet).

use std::time::{Duration, Instant};

use eframe::egui;

use crate::theme::*;

/// Separator choices, mirroring the Java combo box.
const SEPARATOR_LABELS: [&str; 3] = [", (Comma)", "(Space)", "\\n (Newline)"];

/// Green flash shown on the Save button before the popup closes.
const SAVED_GREEN: egui::Color32 = egui::Color32::from_rgb(46, 160, 67);
/// Red flash shown on the Cancel button before the popup closes.
const CANCEL_RED: egui::Color32 = egui::Color32::from_rgb(200, 55, 55);
/// How long a button flash lingers before the popup closes.
const SAVE_FLASH: Duration = Duration::from_millis(450);

/// Tag-manager preferences plus the dialog's open state. Defaults match the Java.
pub struct TagManagerSettings {
    /// Whether the settings dialog is currently shown.
    pub open: bool,
    pub auto_save: bool,
    pub autocomplete: bool,
    pub auto_tag_append: bool,
    pub auto_tag_overwrite: bool,
    /// Index into [`SEPARATOR_LABELS`] (0 = comma, 1 = space, 2 = newline).
    pub separator_idx: usize,
    pub default_threshold: f32,
    pub blacklist: String,
    /// When Save was clicked — drives the green flash before the popup closes.
    pub save_flash: Option<Instant>,
    /// When Cancel was clicked — drives the red flash before the popup closes.
    pub cancel_flash: Option<Instant>,
}

impl Default for TagManagerSettings {
    fn default() -> Self {
        Self {
            open: false,
            auto_save: true,
            autocomplete: true,
            auto_tag_append: false,
            auto_tag_overwrite: false,
            separator_idx: 0,
            default_threshold: 0.35,
            blacklist: "sensitive, nsfw".into(),
            save_flash: None,
            cancel_flash: None,
        }
    }
}

/// Render the settings as a popup dropping down under the gear `anchor`. Save,
/// Cancel, or Escape close it (values apply live — no persistence yet, so
/// there's nothing to revert).
pub fn show(anchor: &egui::Response, s: &mut TagManagerSettings) {
    if !s.open {
        return;
    }

    let mut open = true;
    let mut close = false;

    // Button flashes: once Save/Cancel is clicked we keep the popup open showing
    // a coloured button, then close after `SAVE_FLASH`.
    let save_flashing = s.save_flash.is_some();
    let cancel_flashing = s.cancel_flash.is_some();
    let flash_done = s.save_flash.is_some_and(|t| t.elapsed() >= SAVE_FLASH)
        || s.cancel_flash.is_some_and(|t| t.elapsed() >= SAVE_FLASH);

    egui::Popup::from_response(anchor)
        .open_bool(&mut open)
        .align(egui::RectAlign::BOTTOM_END)
        .width(320.0)
        .gap(6.0)
        .frame(crate::card_frame(22))
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .show(|ui| {
            // The global theme rounds checkboxes into pills; square them off.
            let sq = egui::CornerRadius::same(4);
            ui.visuals_mut().widgets.inactive.corner_radius = sq;
            ui.visuals_mut().widgets.hovered.corner_radius = sq;
            ui.visuals_mut().widgets.active.corner_radius = sq;

            section(ui, "General Behavior", |ui| {
                ui.checkbox(&mut s.auto_save, rich("Auto-save sidecar files after AI tagging"));
                ui.add_space(6.0);
                ui.checkbox(&mut s.autocomplete, rich("Enable tag autocomplete while typing"));

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // Mutually exclusive: enabling one clears the other (matches the Java).
                if ui
                    .checkbox(&mut s.auto_tag_append, rich("Auto tag all images (Keep existing tags)"))
                    .changed()
                    && s.auto_tag_append
                {
                    s.auto_tag_overwrite = false;
                }
                ui.add_space(6.0);
                if ui
                    .checkbox(&mut s.auto_tag_overwrite, rich("Auto tag all images (Remove old tags)"))
                    .changed()
                    && s.auto_tag_overwrite
                {
                    s.auto_tag_append = false;
                }

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label(rich("Tag Separator:"));
                    egui::ComboBox::from_id_salt("tms_separator")
                        .selected_text(SEPARATOR_LABELS[s.separator_idx])
                        .show_ui(ui, |ui| {
                            for (i, label) in SEPARATOR_LABELS.iter().enumerate() {
                                ui.selectable_value(&mut s.separator_idx, i, *label);
                            }
                        });
                });
            });

            section(ui, "AI Tagger Defaults", |ui| {
                ui.horizontal(|ui| {
                    ui.label(rich("Default Confidence Threshold:"));
                    ui.add(
                        egui::DragValue::new(&mut s.default_threshold)
                            .range(0.1..=1.0)
                            .speed(0.01)
                            .fixed_decimals(2),
                    );
                });
            });

            section(ui, "Global Tag Blacklist", |ui| {
                ui.label(egui::RichText::new("Comma or newline separated").color(MUTED()).size(12.0));
                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::multiline(&mut s.blacklist)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3),
                );
            });

            ui.add_space(10.0);

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                let r = egui::CornerRadius::same(10);
                ui.visuals_mut().widgets.inactive.corner_radius = r;
                ui.visuals_mut().widgets.hovered.corner_radius = r;
                ui.visuals_mut().widgets.active.corner_radius = r;

                // Both buttons share the same base style and flash a colour once
                // clicked (green = Save, red = Cancel) before the popup closes.
                let save = if save_flashing {
                    egui::Button::new(egui::RichText::new("Save").color(egui::Color32::WHITE)).fill(SAVED_GREEN)
                } else {
                    egui::Button::new(rich("Save"))
                };
                if ui.add_sized(egui::vec2(90.0, 32.0), save).clicked() && s.save_flash.is_none() {
                    s.save_flash = Some(Instant::now());
                }

                let cancel = if cancel_flashing {
                    egui::Button::new(egui::RichText::new("Cancel").color(egui::Color32::WHITE)).fill(CANCEL_RED)
                } else {
                    egui::Button::new(rich("Cancel"))
                };
                if ui.add_sized(egui::vec2(90.0, 32.0), cancel).clicked() && s.cancel_flash.is_none() {
                    s.cancel_flash = Some(Instant::now());
                }
            });

            if save_flashing || cancel_flashing {
                ui.ctx().request_repaint(); // keep the flash animating, then close
            }
        });

    // Close once the green flash has been shown long enough.
    if flash_done {
        close = true;
    }
    let staying_open = open && !close;
    if !staying_open {
        // reset so the next open starts clean
        s.save_flash = None;
        s.cancel_flash = None;
    }
    s.open = staying_open;
}

fn rich(text: &str) -> egui::RichText {
    egui::RichText::new(text).color(TEXT())
}

/// A titled rounded group card, matching the app's settings style.
fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(6.0);
    ui.label(egui::RichText::new(title).color(MUTED()).strong().size(12.0));
    ui.add_space(4.0);

    egui::Frame::new()
        .fill(FIELD())
        .corner_radius(egui::CornerRadius::same(22))
        .inner_margin(egui::Margin::symmetric(14, 12))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
    ui.add_space(2.0);
}
