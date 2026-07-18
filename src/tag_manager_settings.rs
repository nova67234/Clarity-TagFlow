//! Tag Manager settings — a Rust port of terminus2's `TagManagerSettings` dialog,
//! opened from the Tag Manager panel's gear button.
//!
//! Two tabs behind a segmented control: **Settings** (preferences, apply live
//! and auto-save — the struct is persisted inside `Settings`, see `main.rs`)
//! and **Get Models** (the AI Model Manager catalog, `ai_models.rs`). There are
//! no Save/Cancel buttons — click anywhere outside the popup (or press Escape)
//! to close it.

use eframe::egui;

use crate::theme::*;

/// Separator choices, mirroring the Java combo box (comma / space / newline).
const SEPARATOR_LABELS: [&str; 3] = ["Comma", "Space", "Newline"];

/// Segmented-control tabs across the top of the popup.
const TABS: [&str; 2] = ["Settings", "Get Models"];

/// Tag-manager preferences plus the dialog's open state. Defaults match the
/// Java. Persisted across runs as `Settings::tag_manager`.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TagManagerSettings {
    /// Whether the settings dialog is currently shown (never persisted).
    #[serde(skip)]
    pub open: bool,
    /// Active tab index into [`TABS`] (never persisted).
    #[serde(skip)]
    pub tab: usize,
    pub auto_save: bool,
    pub autocomplete: bool,
    pub auto_tag_append: bool,
    pub auto_tag_overwrite: bool,
    /// Index into [`SEPARATOR_LABELS`] (0 = comma, 1 = space, 2 = newline).
    pub separator_idx: usize,
    pub default_threshold: f32,
    pub blacklist: String,
}

impl Default for TagManagerSettings {
    fn default() -> Self {
        Self {
            open: false,
            tab: 0,
            auto_save: true,
            autocomplete: true,
            auto_tag_append: false,
            auto_tag_overwrite: false,
            separator_idx: 0,
            default_threshold: 0.35,
            blacklist: "sensitive, nsfw".into(),
        }
    }
}

/// Render the settings as a popup dropping down under the gear `anchor`.
/// Values apply live; clicking outside or pressing Escape closes the popup.
/// `models` backs the Get Models tab (downloads keep running after close —
/// the Tag Manager ticks it every frame).
pub fn show(
    anchor: &egui::Response,
    s: &mut TagManagerSettings,
    models: &mut crate::ai_models::ModelManager,
) {
    if !s.open {
        return;
    }

    let mut open = true;
    let mut esc = false;

    egui::Popup::from_response(anchor)
        .open_bool(&mut open)
        .align(egui::RectAlign::BOTTOM_END)
        .width(380.0)
        .gap(6.0)
        .frame(crate::card_frame(22))
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            esc = ui.input(|i| i.key_pressed(egui::Key::Escape));

            segmented_control(ui, &mut s.tab);
            ui.add_space(12.0);

            if s.tab == 1 {
                models.ui(ui);
            } else {
                settings_tab(ui, s);
            }
        });

    s.open = open && !esc;
}

/// An Apple-style segmented control: a full-width capsule track with equal
/// segments, the active one raised as an accent-tinted pill.
fn segmented_control(ui: &mut egui::Ui, current: &mut usize) {
    let height = 30.0;
    let (track, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    ui.painter().rect(
        track,
        egui::CornerRadius::same(10),
        FIELD(),
        egui::Stroke::new(1.0, EDGE()),
        egui::StrokeKind::Inside,
    );

    let pad = 3.0;
    let seg_w = (track.width() - pad * 2.0) / TABS.len() as f32;
    for (i, label) in TABS.iter().enumerate() {
        let seg = egui::Rect::from_min_size(
            egui::pos2(track.min.x + pad + seg_w * i as f32, track.min.y + pad),
            egui::vec2(seg_w, height - pad * 2.0),
        );
        let resp = ui.interact(seg, ui.id().with(("tm_settings_tab", i)), egui::Sense::click());
        let selected = *current == i;
        if selected {
            ui.painter().rect_filled(
                seg,
                egui::CornerRadius::same(8),
                ACCENT1().gamma_multiply(0.30),
            );
        } else if resp.hovered() {
            ui.painter().rect_filled(
                seg,
                egui::CornerRadius::same(8),
                TEXT().gamma_multiply(0.06),
            );
        }
        ui.painter().text(
            seg.center(),
            egui::Align2::CENTER_CENTER,
            *label,
            egui::FontId::proportional(12.0),
            if selected { TEXT() } else { MUTED() },
        );
        if resp.clicked() {
            *current = i;
        }
    }
}

/// The preferences tab: Apple-style cards of label-left/switch-right rows
/// (the shared helpers from settings.rs).
fn settings_tab(ui: &mut egui::Ui, s: &mut TagManagerSettings) {
    use crate::settings::{row, row_sep, section, switch};

    section(ui, "General", |ui| {
        row(
            ui,
            "Auto-save sidecar files",
            Some("Write the .txt automatically after AI tagging."),
            |ui| {
                switch(ui, &mut s.auto_save);
            },
        );
        row_sep(ui);
        row(
            ui,
            "Tag autocomplete",
            Some("Suggest tags while typing."),
            |ui| {
                switch(ui, &mut s.autocomplete);
            },
        );
        row_sep(ui);
        // The two auto-tag modes are mutually exclusive: enabling one clears
        // the other (matches the Java).
        row(
            ui,
            "Auto tag all — keep tags",
            Some("Tag every image, keeping its existing tags."),
            |ui| {
                if switch(ui, &mut s.auto_tag_append).changed() && s.auto_tag_append {
                    s.auto_tag_overwrite = false;
                }
            },
        );
        row_sep(ui);
        row(
            ui,
            "Auto tag all — overwrite",
            Some("Tag every image, removing its old tags first."),
            |ui| {
                if switch(ui, &mut s.auto_tag_overwrite).changed() && s.auto_tag_overwrite {
                    s.auto_tag_append = false;
                }
            },
        );
        row_sep(ui);
        row(ui, "Tag separator", None, |ui| {
            // Inline pills instead of a combo — a combo's dropdown would count
            // as a click outside this popup and close it. The right-to-left
            // control layout lays children rightmost-first, so iterate reversed
            // to keep Comma · Space · Newline reading left to right.
            ui.spacing_mut().item_spacing.x = 4.0;
            for (i, label) in SEPARATOR_LABELS.iter().enumerate().rev() {
                ui.selectable_value(&mut s.separator_idx, i, *label);
            }
        });
    });

    section(ui, "AI tagger", |ui| {
        row(
            ui,
            "Confidence threshold",
            Some("Minimum score for a predicted tag to be added."),
            |ui| {
                ui.add(
                    egui::DragValue::new(&mut s.default_threshold)
                        .range(0.1..=1.0)
                        .speed(0.01)
                        .fixed_decimals(2),
                );
            },
        );
    });

    section(ui, "Tag blacklist", |ui| {
        ui.add(
            egui::TextEdit::multiline(&mut s.blacklist)
                .desired_width(f32::INFINITY)
                .desired_rows(3),
        );
        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("Tags the AI never adds — comma or newline separated.")
                .color(MUTED())
                .size(10.5),
        );
    });
}
