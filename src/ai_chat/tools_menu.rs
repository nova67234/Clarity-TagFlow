//! The input card's Tools menu: the chat's extra abilities as a grid of
//! cards, two to a row — each with its name, its switch and a few words on
//! what it is — over one panel describing the card the pointer is on. (The
//! menu used to list every description under its switch, which made it taller
//! than most windows.)

use std::sync::Arc;

use eframe::egui;

use super::{ChatMode, icon_button, mode_on, set_mode};
use crate::llm::LlmState;
use crate::settings::Settings;
use crate::theme::*;

/// The menu's content width (inside its frame): two cards wide enough for
/// the longest name ("Auto-speak replies") and its switch on one line.
const INNER_W: f32 = 400.0;
/// The frame's margin around the content.
const MARGIN: i8 = 14;
/// Space around the grid and the panel, inside the well holding them.
const WELL_PAD: i8 = 10;
/// Between the cards (both ways).
const GAP: f32 = 8.0;
const CARD_H: f32 = 64.0;
const CARD_PAD_X: f32 = 12.0;
const CARD_PAD_Y: f32 = 11.0;
/// [`crate::settings::switch`]'s size.
const SWITCH: egui::Vec2 = egui::vec2(40.0, 22.0);

/// One of the menu's tools, in the grid's order (row by row).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tool {
    Roleplay,
    AutoSpeak,
    Krea,
    TagImages,
    DetailedVision,
    WebResearch,
}

impl Tool {
    const ALL: [Tool; 6] =
        [Tool::Roleplay, Tool::AutoSpeak, Tool::Krea, Tool::TagImages, Tool::DetailedVision, Tool::WebResearch];

    fn name(self) -> &'static str {
        match self {
            Tool::Roleplay => "Role playing",
            Tool::AutoSpeak => "Auto-speak replies",
            Tool::Krea => "Krea 2 prompts",
            Tool::TagImages => "Tag images",
            Tool::DetailedVision => "Detailed vision",
            Tool::WebResearch => "Web research",
        }
    }

    /// For the card's id.
    fn key(self) -> &'static str {
        match self {
            Tool::Roleplay => "roleplay",
            Tool::AutoSpeak => "speak",
            Tool::Krea => "krea",
            Tool::TagImages => "tag",
            Tool::DetailedVision => "vision",
            Tool::WebResearch => "web",
        }
    }
}

/// The colour of the line under a card's name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Ink {
    Muted,
    /// Held on by another tool.
    Accent,
    /// Can't work yet.
    Warn,
}

/// The pill beside the name in the panel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Badge {
    On,
    Off,
    /// Held on by another tool.
    AutoOn,
}

/// What a tool's card and the panel show.
pub(super) struct Shown {
    pub on: bool,
    /// The switch can be flipped: Krea 2 prompts holds Detailed vision on,
    /// and Tag images holds Web research off.
    pub usable: bool,
    /// The few words under the name.
    pub line: String,
    pub ink: Ink,
    /// The panel's text.
    pub about: String,
    /// What's missing before it can work, in red under the text.
    pub problem: Option<String>,
    pub badge: Badge,
}

impl Shown {
    fn plain(on: bool, line: &str, about: &str) -> Shown {
        Shown {
            on,
            usable: true,
            line: line.to_string(),
            ink: Ink::Muted,
            about: about.to_string(),
            problem: None,
            badge: if on { Badge::On } else { Badge::Off },
        }
    }
}

fn shown(tool: Tool, llm: &LlmState, settings: &Settings) -> Shown {
    match tool {
        Tool::Roleplay => Shown::plain(
            mode_on(llm, settings, ChatMode::Roleplay),
            "Persona & memory",
            "The AI plays a character with its own name, persona and memories, kept apart from its main \
             memory. Set it up with the gear above.",
        ),
        Tool::AutoSpeak => Shown::plain(
            settings.ai_auto_speak,
            "Read replies aloud",
            "Read every reply aloud the moment it finishes — click any reply's listen icon to stop.",
        ),
        Tool::Krea => Shown::plain(
            mode_on(llm, settings, ChatMode::Krea),
            "Prompt generator",
            "Turn a picture — or an idea in words — into a detailed prompt for Krea 2. Ask for changes \
             (\"make it night\") to rewrite it. Reads pictures with Detailed vision.",
        ),
        Tool::TagImages => Shown::plain(
            mode_on(llm, settings, ChatMode::TagImages),
            "Folder Tag Manager",
            "Tag the open folder with the AI: the image on the left, the Tag Manager on the right. The AI \
             sees each image's tags and can change them, with Undo.",
        ),
        Tool::DetailedVision => {
            // Held on while the Krea 2 writer is (it relies on it).
            let held = settings.ai_krea_prompts;
            let mut s = Shown::plain(
                settings.ai_detailed_vision || held,
                "Close-up crop analysis",
                "Show the AI the newest picture whole plus a few close-up sections, so small details (eyes, \
                 hands, text, jewelry) aren't missed. Replies start slower. Always on with Krea 2 prompts.",
            );
            if held {
                s.usable = false;
                s.line = "Auto-on with Krea 2".to_string();
                s.ink = Ink::Accent;
                s.badge = Badge::AutoOn;
            }
            s
        }
        Tool::WebResearch => super::web_research::tool_card(llm, settings),
    }
}

/// Flip `tool`'s switch.
fn set(tool: Tool, llm: &mut LlmState, settings: &mut Settings, on: bool) {
    match tool {
        Tool::Roleplay => set_mode(llm, settings, ChatMode::Roleplay, on),
        Tool::Krea => set_mode(llm, settings, ChatMode::Krea, on),
        Tool::TagImages => set_mode(llm, settings, ChatMode::TagImages, on),
        Tool::AutoSpeak => {
            settings.ai_auto_speak = on;
            llm.auto_speak = on;
        }
        Tool::DetailedVision => {
            settings.ai_detailed_vision = on;
            llm.detailed_vision = on;
        }
        Tool::WebResearch => {
            settings.ai_web_research = on;
            llm.web_research = on;
        }
    }
}

/// The menu, while `llm.tools_open`: over the input card, its left edge on
/// the Tools button's (`button`). Clicking outside closes it, and so does its
/// gear (which opens AI Settings instead); flipping a switch doesn't.
pub(super) fn popup(button: &egui::Response, llm: &mut LlmState, settings: &mut Settings) {
    if !llm.tools_open {
        return;
    }
    // Docked to the card's top edge like the model picker: the button sits
    // in the card's bottom row, so a gap from it alone would cover the draft.
    // (The card's rect is last frame's; before there is one, the button
    // stands in.)
    let dock = llm.input_rect.map_or(button.rect, |(card, _)| {
        egui::Rect::from_x_y_ranges(button.rect.x_range(), card.top()..=card.top())
    });
    let mut open = true;
    egui::Popup::from_response(button)
        .anchor(dock)
        .open_bool(&mut open)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .align(egui::RectAlign::TOP_START) // opens upward (the card sits at the bottom)
        .width(INNER_W + 2.0 * MARGIN as f32 + 2.0)
        .gap(10.0)
        // The model picker's popover card: see-through glass in the Glass
        // themes.
        .frame(
            egui::Frame::new()
                .fill(super::picker_fill())
                .corner_radius(egui::CornerRadius::same(22))
                .inner_margin(egui::Margin::same(MARGIN))
                .stroke(egui::Stroke::new(1.0_f32, EDGE()))
                .shadow(egui::epaint::Shadow {
                    offset: [0, 16],
                    blur: 42,
                    spread: 0,
                    color: egui::Color32::from_black_alpha(165),
                }),
        )
        .show(|ui| contents(ui, llm, settings));
    llm.tools_open &= open;
}

fn contents(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut Settings) {
    // Both, or the width is only a minimum (see notes/ui-style-guide.md).
    ui.set_width(INNER_W);
    ui.set_max_width(INNER_W);

    // The header: its gear opens AI Settings — on the Role play tab while
    // role play is on.
    ui.horizontal(|ui| {
        ui.add_space(4.0);
        ui.label(egui::RichText::new("TOOLS").color(MUTED()).size(11.5).extra_letter_spacing(1.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let tip = if llm.roleplay.enabled { "Role play settings and memories" } else { "AI settings" };
            if icon_button(ui, egui::include_image!("../../icons/settings.svg"), 16.0, tip, true).clicked() {
                if llm.roleplay.enabled {
                    llm.ai_settings_tab = crate::ai_settings::AiTab::Roleplay;
                }
                llm.gen_settings_open = true;
                llm.tools_open = false;
            }
        });
    });
    ui.add_space(6.0);

    // A recessed well holding the cards and, under a hairline, the panel.
    egui::Frame::new()
        .fill(TEXT().gamma_multiply(0.035))
        .corner_radius(egui::CornerRadius::same(18))
        .inner_margin(egui::Margin::same(WELL_PAD))
        .stroke(egui::Stroke::new(1.0_f32, super::picker_hairline()))
        .show(ui, |ui| {
            let inspected = grid(ui, llm, settings);
            ui.add_space(10.0);
            let (line, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
            ui.painter().rect_filled(line, 0.0, super::picker_hairline());
            ui.add_space(10.0);
            panel(ui, llm, settings, inspected);
        });
}

/// The cards. Returns the tool the panel describes: the card under the
/// pointer, else the one it last was on (Role playing at first).
fn grid(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut Settings) -> Tool {
    let w = ui.available_width();
    let card_w = (w - GAP) / 2.0;
    let rows = Tool::ALL.len().div_ceil(2) as f32;
    let (_, area) = ui.allocate_space(egui::vec2(w, rows * CARD_H + (rows - 1.0) * GAP));
    let rect_of = |i: usize| {
        let (col, row) = ((i % 2) as f32, (i / 2) as f32);
        egui::Rect::from_min_size(
            area.min + egui::vec2(col * (card_w + GAP), row * (CARD_H + GAP)),
            egui::vec2(card_w, CARD_H),
        )
    };

    // The cards sense first, so their switches (added after) sit on top and
    // take the clicks. A card's own click only points the panel at it: a
    // stray click mustn't start a new chat (the modes do).
    let cards: Vec<egui::Response> = Tool::ALL
        .iter()
        .enumerate()
        .map(|(i, t)| ui.interact(rect_of(i), egui::Id::new("ai_tools_card").with(t.key()), egui::Sense::click()))
        .collect();
    let keep_id = egui::Id::new("ai_tools_inspected");
    let mut inspected = ui.data(|d| d.get_temp::<Tool>(keep_id)).unwrap_or(Tool::Roleplay);
    let pointed = cards.iter().position(|c| c.contains_pointer());
    if let Some(i) = pointed {
        inspected = Tool::ALL[i];
    }

    for (i, &tool) in Tool::ALL.iter().enumerate() {
        let rect = rect_of(i);
        let s = shown(tool, llm, settings);
        let id = cards[i].id;
        let t_on = ui.ctx().animate_bool_with_time(id.with("on"), s.on, 0.15);
        let t_in = ui.ctx().animate_bool_with_time(id.with("in"), tool == inspected, 0.12);

        // Off: a faint card; on: tinted blue like the model picker's chosen
        // model. The one the panel describes is lifted, with a blue rim.
        let fill = TEXT()
            .gamma_multiply(0.045)
            .lerp_to_gamma(TEXT().gamma_multiply(0.10), t_in)
            .lerp_to_gamma(super::THINK_BLUE.gamma_multiply(0.15 + 0.06 * t_in), t_on);
        let rim = TEXT()
            .gamma_multiply(0.09)
            .lerp_to_gamma(super::picker_badge_ink().gamma_multiply(0.42), t_in)
            .lerp_to_gamma(super::picker_blue().gamma_multiply(0.38 + 0.4 * t_in), t_on);
        let radius = egui::CornerRadius::same(12);
        let p = ui.painter();
        p.rect_filled(rect, radius, fill);
        p.rect_stroke(rect, radius, egui::Stroke::new(1.0_f32, rim), egui::StrokeKind::Inside);

        // The switch, top right.
        let sw = egui::Rect::from_min_size(
            egui::pos2(rect.right() - CARD_PAD_X - SWITCH.x, rect.top() + CARD_PAD_Y),
            SWITCH,
        );
        // In a child that doesn't move the grid's cursor (a scope would
        // put it back up to the switch, under the grid's last row).
        let mut on = s.on;
        let mut child = ui.new_child(egui::UiBuilder::new().id_salt(("ai_tools_switch", tool.key())).max_rect(sw));
        if !s.usable {
            child.disable();
        }
        let resp = crate::settings::switch(&mut child, &mut on);
        resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, s.usable, on, tool.name()));
        if resp.changed() {
            set(tool, llm, settings, on);
            inspected = tool;
        } else if resp.has_focus() && pointed.is_none() {
            inspected = tool; // reached with the keyboard
        }

        // The name beside it, the line under both.
        let x = rect.left() + CARD_PAD_X;
        let name_w = sw.left() - 8.0 - x;
        let name = one_line(ui, tool.name(), egui::FontId::proportional(13.5), TEXT(), name_w);
        ui.painter().galley(egui::pos2(x, sw.center().y - name.size().y / 2.0), name, TEXT());
        let ink = match s.ink {
            Ink::Muted => MUTED(),
            Ink::Accent => super::picker_badge_ink(),
            Ink::Warn => warn_ink(),
        };
        let line = one_line(ui, &s.line, egui::FontId::proportional(11.5), ink, rect.width() - 2.0 * CARD_PAD_X);
        ui.painter().galley(egui::pos2(x, rect.bottom() - CARD_PAD_Y - line.size().y), line, ink);
    }

    ui.data_mut(|d| d.insert_temp(keep_id, inspected));
    inspected
}

/// The panel under the cards: the inspected tool's name, whether it's on,
/// and what it does.
fn panel(ui: &mut egui::Ui, llm: &LlmState, settings: &Settings, tool: Tool) {
    let s = shown(tool, llm, settings);
    let dark = !is_light();
    egui::Frame::new()
        .fill(if dark { egui::Color32::from_black_alpha(64) } else { egui::Color32::from_black_alpha(10) })
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(12, 10))
        .stroke(egui::Stroke::new(1.0_f32, super::picker_hairline()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let w = ui.available_width();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tool.name()).color(TEXT()).font(crate::settings::bold(13.5)));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| badge(ui, s.badge));
            });
            ui.add_space(3.0);

            // As tall as the longest text, so the menu — which grows upward
            // from the input card — never moves under the pointer as it goes
            // from card to card.
            let gap = ui.spacing().item_spacing.y;
            let tallest = Tool::ALL
                .into_iter()
                .map(|t| {
                    let s = shown(t, llm, settings);
                    let mut h = ui.fonts_mut(|f| f.layout(s.about, about_font(), TEXT(), w)).size().y;
                    if let Some(p) = s.problem {
                        h += gap + ui.fonts_mut(|f| f.layout(p, problem_font(), TEXT(), w)).size().y;
                    }
                    h
                })
                .fold(0.0_f32, f32::max);
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.set_min_height(tallest);
                ui.label(egui::RichText::new(&s.about).font(about_font()).color(TEXT().gamma_multiply(0.8)));
                if let Some(p) = &s.problem {
                    ui.label(egui::RichText::new(p).font(problem_font()).color(warn_ink()));
                }
            });
        });
}

fn about_font() -> egui::FontId {
    egui::FontId::proportional(12.0)
}

fn problem_font() -> egui::FontId {
    egui::FontId::proportional(11.5)
}

/// Red that reads on the menu (lighter on the dark themes).
fn warn_ink() -> egui::Color32 {
    if is_light() { egui::Color32::from_rgb(200, 50, 50) } else { egui::Color32::from_rgb(235, 110, 110) }
}

/// `text` on one line, cut short with "…" past `max_w`.
fn one_line(ui: &egui::Ui, text: &str, font: egui::FontId, ink: egui::Color32, max_w: f32) -> Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_string(), font, ink);
    job.wrap = egui::text::TextWrapping::truncate_at_width(max_w);
    ui.fonts_mut(|f| f.layout_job(job))
}

/// The ON / OFF / AUTO-ON pill.
fn badge(ui: &mut egui::Ui, b: Badge) {
    const GREEN: egui::Color32 = egui::Color32::from_rgb(46, 160, 67);
    let (text, fill, rim, ink) = match b {
        Badge::On => ("ON", GREEN.gamma_multiply(0.2), GREEN.gamma_multiply(0.4), super::picker_ink(GREEN, 0.55)),
        Badge::Off => ("OFF", TEXT().gamma_multiply(0.08), TEXT().gamma_multiply(0.1), MUTED()),
        Badge::AutoOn => (
            "AUTO-ON",
            super::THINK_BLUE.gamma_multiply(0.2),
            super::picker_blue().gamma_multiply(0.4),
            super::picker_badge_ink(),
        ),
    };
    let mut job = egui::text::LayoutJob::default();
    job.append(
        text,
        0.0,
        egui::TextFormat {
            font_id: crate::settings::bold(9.5),
            color: ink,
            extra_letter_spacing: 0.8,
            ..Default::default()
        },
    );
    let galley = ui.fonts_mut(|f| f.layout_job(job));
    let (rect, _) = ui.allocate_exact_size(galley.size() + egui::vec2(16.0, 6.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, egui::CornerRadius::same(9), fill);
    p.rect_stroke(rect, egui::CornerRadius::same(9), egui::Stroke::new(1.0_f32, rim), egui::StrokeKind::Inside);
    p.galley(rect.center() - galley.size() / 2.0, galley, ink);
}
