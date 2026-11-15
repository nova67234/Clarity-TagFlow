//! Tag Manager settings — the pop-up under the Tag Manager's gear button (it
//! began as a Rust port of terminus2's `TagManagerSettings` dialog).
//!
//! Two tabs behind a segmented control: **Settings** (Auto-tag's preferences;
//! they apply as they change and are saved with the app's settings — the
//! struct is persisted inside `Settings`, see `main.rs`) and **Models** (the
//! models to download, `ai_models.rs`). There are no Save/Cancel buttons — a
//! click anywhere outside the pop-up, or Escape, closes it.
//!
//! It looks as the Settings window does — spaced-out capitals over rounded
//! cards, sunk wells for what's typed, the app's slider — set in the Tag
//! Manager's faces.

use eframe::egui;
use egui::Color32;

use crate::settings::hairline;
use crate::slider::{PillSlider, THUMB_PAD};
use crate::tag_manager::{bold, chip_fill, medium, semibold, well};
use crate::theme::*;

/// Segmented-control tabs across the top of the popup.
const TABS: [&str; 2] = ["Settings", "Models"];

/// The pop-up: this wide, this far under the gear, its corners this round.
const WIDTH: f32 = 380.0;
const GAP: f32 = 6.0;
const RADIUS: u8 = 22;
/// What its frame puts round the content: an edge line, then the padding.
const BORDER: f32 = 1.0;
const PADDING: f32 = 12.0;
/// The tabs' height, and the gap under them.
const TABS_H: f32 = 36.0;
const TABS_GAP: f32 = 14.0;

/// What's under the tabs is as tall as fits under the gear, between these
/// (then it scrolls): at most, the pop-up is 520 points tall.
const BODY_MIN: f32 = 120.0;
const BODY_MAX: f32 = 451.0;
/// Scrolling, it runs on to the pop-up's bottom edge, short of it by this:
/// a card cut off there then stays inside the rounded corners. At its end,
/// the last card stops the usual padding above the edge.
const BODY_FOOT: f32 = 5.0;
/// Its scroll bar is in the pop-up's right padding, and stops this far
/// above the body's end, where the bottom corner curves in.
const GUTTER: f32 = 12.0;
const BAR_FOOT: f32 = 8.0;
/// Kept clear between the pop-up and the window's bottom edge.
const WINDOW_MARGIN: f32 = 8.0;

/// The cards' corners (the Settings window's).
pub(crate) const CARD_RADIUS: u8 = 18;

/// The Confidence value's pill.
const PILL_W: f32 = 52.0;
const PILL_H: f32 = 26.0;
/// The scores Auto-tag can be asked for.
const CONFIDENCE: std::ops::RangeInclusive<f32> = 0.1..=1.0;
/// The most lines the blacklist's box shows at once.
const BLACKLIST_LINES: usize = 6;

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
    // The next five came over from the Java dialog but nothing acts on them
    // yet, so their switches are hidden (they showed, and did nothing). Kept
    // so saved settings still load and the switches can come back with the
    // features.
    pub auto_save: bool,
    pub autocomplete: bool,
    pub auto_tag_append: bool,
    pub auto_tag_overwrite: bool,
    /// 0 = comma, 1 = space, 2 = newline.
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
/// `models` backs the Models tab (downloads keep running after close — the
/// Tag Manager ticks it every frame).
pub fn show(
    anchor: &egui::Response,
    s: &mut TagManagerSettings,
    models: &mut crate::ai_models::ModelManager,
) {
    if !s.open {
        return;
    }

    // (The pop-up closes on Escape by itself.)
    let mut open = true;
    let mut frame = crate::float_card_frame(RADIUS);
    frame.inner_margin.bottom = BODY_FOOT as i8;

    egui::Popup::from_response(anchor)
        .open_bool(&mut open)
        .align(egui::RectAlign::BOTTOM_END)
        .width(WIDTH)
        .gap(GAP)
        .frame(frame)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            segmented_control(ui, &mut s.tab);
            ui.add_space(TABS_GAP);

            // Scope widget ids by tab so same-position widgets in the two tabs
            // don't share auto-ids (nor the tabs where they're scrolled to).
            ui.push_id(s.tab, |ui| {
                body(ui, anchor, s.tab, |ui| {
                    if s.tab == 1 {
                        models.ui(ui);
                    } else {
                        settings_tab(ui, s);
                    }
                });
            });
        });

    s.open = open;
}

/// The tabs: a sunk capsule with a segment each, the chosen one a raised
/// pill — it slides across when the other is picked.
fn segmented_control(ui: &mut egui::Ui, current: &mut usize) {
    const INSET: f32 = 3.0;
    let (track, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), TABS_H), egui::Sense::hover());
    let width = (track.width() - INSET * 2.0) / TABS.len() as f32;
    // The segment at `place` along the track (0.5: halfway from the first
    // tab to the second, where the pill is on its way across).
    let segment = |place: f32| {
        egui::Rect::from_min_size(
            egui::pos2(track.left() + INSET + width * place, track.top() + INSET),
            egui::vec2(width, TABS_H - INSET * 2.0),
        )
    };
    let mut hovered = None;
    for (i, label) in TABS.iter().enumerate() {
        let resp = ui.interact(segment(i as f32), ui.id().with(("tm_settings_tab", i)), egui::Sense::click());
        resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::RadioButton, true, *current == i, *label));
        if resp.clicked() {
            *current = i;
        }
        if resp.hovered() || resp.has_focus() {
            hovered = Some(i);
            if *current != i {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
        }
    }
    if !ui.is_rect_visible(track) {
        return;
    }

    let line = egui::Stroke::new(1.0_f32, hairline());
    let p = ui.painter();
    p.rect(track, TABS_H / 2.0, well(), line, egui::StrokeKind::Inside);
    let place = ui.ctx().animate_value_with_time(ui.id().with("tm_settings_thumb"), *current as f32, 0.16);
    let thumb = segment(place);
    p.rect(thumb, thumb.height() / 2.0, chip_fill(false), line, egui::StrokeKind::Inside);
    for (i, label) in TABS.iter().enumerate() {
        let ink = if *current == i || hovered == Some(i) { TEXT() } else { MUTED() };
        p.text(segment(i as f32).center(), egui::Align2::CENTER_CENTER, *label, semibold(12.5), ink);
    }
}

/// A group's heading: small bold capitals, spaced out, over its card.
pub(crate) fn caps_label(ui: &mut egui::Ui, text: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 14.0), egui::Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_uppercase(), bold(10.0), MUTED());
    job.sections[0].format.extra_letter_spacing = 1.1;
    let galley = ui.fonts_mut(|f| f.layout_job(job));
    // In from the card's edge by a little, as the card's corner is.
    let at = egui::pos2(rect.left() + 4.0, rect.center().y - galley.size().y / 2.0);
    ui.painter().galley(at, galley, MUTED());
}

/// The cards' fill: the pop-up's colour lifted a little toward the text
/// colour, as the Settings window's cards are — and, as there, glass under
/// Liquid Glass (PANEL is what it frosts).
pub(crate) fn card_fill() -> Color32 {
    if liquid_markers().is_some() { PANEL() } else { FLOAT().lerp_to_gamma(TEXT(), 0.035) }
}

/// A card for `add`'s rows.
fn card(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(card_fill())
        .stroke(egui::Stroke::new(1.0_f32, hairline()))
        .corner_radius(egui::CornerRadius::same(CARD_RADIUS))
        .inner_margin(egui::Margin::same(16))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 0.0;
            add(ui);
        });
}

/// The preferences tab: one card, a setting to a row.
fn settings_tab(ui: &mut egui::Ui, s: &mut TagManagerSettings) {
    caps_label(ui, "Auto-tag");
    ui.add_space(8.0);
    card(ui, |ui| {
        // Confidence: its value in a pill at the row's end (a click there
        // types one), and the app's slider under the row.
        let row = titled(ui, "Confidence", "Tags scoring below this are left out.", PILL_W + 12.0);
        let pill = egui::Rect::from_center_size(
            egui::pos2(row.right() - PILL_W / 2.0, row.center().y),
            egui::vec2(PILL_W, PILL_H),
        );
        value_pill(ui, pill, &mut s.default_threshold);
        ui.add_space(6.0);
        // The rail runs from the card's one margin to the other, in line
        // with the text: the slider starts and ends the room it keeps for
        // its thumb outside them.
        let size = egui::vec2(ui.available_width(), crate::slider::HEIGHT);
        let (rail, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let mut wide = ui.new_child(egui::UiBuilder::new().max_rect(rail.expand2(egui::vec2(THUMB_PAD, 0.0))));
        wide.spacing_mut().slider_width = rail.width() + 2.0 * THUMB_PAD;
        let usual = TagManagerSettings::default().default_threshold;
        wide.add(
            PillSlider::new(&mut s.default_threshold, CONFIDENCE)
                .step_by(0.01)
                .default_value(usual as f64)
                .show_value(false),
        );

        ui.add_space(10.0);
        let (rule, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
        ui.painter().hline(rule.x_range(), rule.center().y, egui::Stroke::new(1.0_f32, hairline()));
        ui.add_space(14.0);

        titled(ui, "Blacklist", "Tags Auto-tag never adds. Separate them with commas.", 0.0);
        ui.add_space(10.0);
        blacklist_box(ui, &mut s.blacklist);
    });
}

/// A setting's name over a line about it (which wraps where it must), with
/// `keep` points left free at the row's end for its control. Where the row
/// is.
fn titled(ui: &mut egui::Ui, title: &str, caption: &str, keep: f32) -> egui::Rect {
    let room = (ui.available_width() - keep).max(0.0);
    let (title, caption, caption_line) = ui.fonts_mut(|f| {
        let mut title = egui::text::LayoutJob::simple_singleline(title.to_string(), semibold(13.5), TEXT());
        title.wrap = egui::text::TextWrapping::truncate_at_width(room);
        let caption = egui::text::LayoutJob::simple(caption.to_string(), medium(12.0), MUTED(), room);
        (f.layout_job(title), f.layout_job(caption), f.row_height(&medium(12.0)))
    });
    // The name's line is 18 points and each of the caption's 16, 2 apart.
    let lines = (caption.size().y / caption_line).round().max(1.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 20.0 + 16.0 * lines), egui::Sense::hover());
    let p = ui.painter();
    p.galley(egui::pos2(rect.left(), rect.top() + (18.0 - title.size().y) / 2.0), title, TEXT());
    p.galley(egui::pos2(rect.left(), rect.top() + 20.0 + (16.0 * lines - caption.size().y) / 2.0), caption, MUTED());
    rect
}

/// The Confidence value while it's being typed: the text so far, and
/// whether this is the field's first frame (focus it, select it all).
#[derive(Clone)]
struct Typing {
    text: String,
    fresh: bool,
}

fn typing_id() -> egui::Id {
    egui::Id::new("tagmgr_confidence_typing")
}

/// What was typed for Confidence, as a score Auto-tag can be asked for:
/// "0.4", ".4" or "0,4" — in range, to two places.
fn typed_confidence(text: &str) -> Option<f32> {
    let v: f32 = text.trim().replace(',', ".").parse().ok()?;
    v.is_finite().then(|| (v.clamp(*CONFIDENCE.start(), *CONFIDENCE.end()) * 100.0).round() / 100.0)
}

/// The Confidence value at `rect`: a small sunk pill showing the number.
/// As on the app's sliders, a click turns it into a text field — Enter, or
/// a click elsewhere, takes what was typed; Escape leaves the value alone.
fn value_pill(ui: &mut egui::Ui, rect: egui::Rect, value: &mut f32) {
    let id = typing_id();
    let shown = format!("{value:.2}");
    let font = semibold(12.5);
    let radius = rect.height() / 2.0;
    if let Some(mut typing) = ui.data(|d| d.get_temp::<Typing>(id)) {
        let ring = egui::Stroke::new(1.0_f32, selection_outline());
        ui.painter().rect(rect, radius, well(), ring, egui::StrokeKind::Inside);
        // The field is in a child Ui over the pill, outside the card's
        // layout: nothing moves while it's open.
        let mut field_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(rect.shrink2(egui::vec2(8.0, 0.0)))
                .layout(egui::Layout::centered_and_justified(egui::Direction::TopDown)),
        );
        let mut out = egui::TextEdit::singleline(&mut typing.text)
            .id(id.with("field"))
            .frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .font(font)
            .text_color(TEXT())
            .horizontal_align(egui::Align::Center)
            .vertical_align(egui::Align::Center)
            .desired_width(f32::INFINITY)
            .show(&mut field_ui);
        let field = out.response.response;
        if typing.fresh {
            // Start with the number selected, ready to type over.
            field.request_focus();
            let all = egui::text::CCursorRange::two(
                egui::text::CCursor::new(0),
                egui::text::CCursor::new(typing.text.chars().count()),
            );
            out.state.cursor.set_char_range(Some(all));
            out.state.store(ui.ctx(), field.id);
        }
        let (escape, enter) = ui.input(|i| (i.key_pressed(egui::Key::Escape), i.key_pressed(egui::Key::Enter)));
        // Neither focused nor just left: the pop-up closed mid-typing and
        // is back — what was typed then is dropped.
        let stale = !typing.fresh && !field.has_focus() && !field.lost_focus();
        if escape {
            // Escape gives up the typing, and that's all it does: taken out
            // of the frame's input, it doesn't close the pop-up as well.
            ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape));
        }
        if escape || stale {
            ui.data_mut(|d| d.remove::<Typing>(id));
        } else if enter || field.lost_focus() {
            // Only what was typed: opened and left as it was, the value
            // stays as it is.
            if typing.text.trim() != shown
                && let Some(v) = typed_confidence(&typing.text)
            {
                *value = v;
            }
            ui.data_mut(|d| d.remove::<Typing>(id));
        } else {
            typing.fresh = false;
            ui.data_mut(|d| d.insert_temp(id, typing));
        }
    } else {
        // The field's child Ui takes an auto id: skip one without it, so
        // what comes after keeps its ids while the field is open.
        ui.skip_ahead_auto_ids(1);
        let pill = ui
            .interact(rect, id.with("pill"), egui::Sense::click())
            .on_hover_cursor(egui::CursorIcon::Text)
            .on_hover_text("Click to type a value");
        pill.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, format!("Confidence {shown}")));
        if pill.clicked() {
            ui.data_mut(|d| d.insert_temp(id, Typing { text: shown.clone(), fresh: true }));
        }
        let hot = pill.hovered() || pill.has_focus();
        let edge = if hot { TEXT().gamma_multiply(0.18) } else { hairline() };
        let p = ui.painter();
        p.rect(rect, radius, well(), egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);
        p.text(rect.center(), egui::Align2::CENTER_CENTER, shown, font, TEXT());
    }
}

/// The blacklist's text box: a sunk well two lines tall that grows with
/// what's typed — up to [`BLACKLIST_LINES`], then it scrolls (a long list
/// made this pop-up taller than the window).
fn blacklist_box(ui: &mut egui::Ui, text: &mut String) {
    // How tall two lines are, and the most it shows, as the text lays them
    // out (line by line, to the pixel).
    let (least, most) = ui.fonts_mut(|f| {
        let mut tall = |lines: usize| {
            f.layout_no_wrap("\n".repeat(lines - 1), medium(13.0), Color32::PLACEHOLDER).size().y
        };
        (tall(2), tall(BLACKLIST_LINES))
    });
    // The well's slot first, filled in once its size and focus are known.
    let bg = ui.painter().add(egui::Shape::Noop);
    let out = egui::Frame::NONE.inner_margin(egui::Margin::symmetric(12, 10)).show(ui, |ui| {
        egui::ScrollArea::vertical()
            .id_salt("tag_blacklist")
            .max_height(most)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                // Keep scrolling while a text selection is dragged past the edge.
                crate::drag_select_autoscroll(ui);
                // (The field's own two lines are a hair short of two lines
                // of text: without this the box grew by it at the second.)
                ui.set_min_height(least);
                ui.add(
                    egui::TextEdit::multiline(text)
                        .frame(egui::Frame::NONE)
                        .margin(egui::Margin::ZERO)
                        .font(medium(13.0))
                        // Explicit: the light themes' widget text is white.
                        .text_color(TEXT())
                        .desired_width(f32::INFINITY)
                        .desired_rows(2),
                )
            })
            .inner
    });
    let edge = if out.inner.has_focus() { selection_outline() } else { hairline() };
    ui.painter().set(
        bg,
        egui::epaint::RectShape::new(
            out.response.rect,
            12.0,
            well(),
            egui::Stroke::new(1.0_f32, edge),
            egui::StrokeKind::Inside,
        ),
    );
}

/// What's under the tabs, scrolling where there isn't room for all of it:
/// the pop-up is no taller than fits between the gear and the window's
/// bottom edge, so it stays under its button whichever tab it's on and
/// however much the tab holds (the Models list always scrolls; Settings
/// does in a short window, or with a long blacklist). The scroll bar is in
/// the pop-up's right padding.
fn body(ui: &mut egui::Ui, anchor: &egui::Response, tab: usize, add: impl FnOnce(&mut egui::Ui)) {
    let above = BORDER + PADDING + TABS_H + TABS_GAP;
    let below = BODY_FOOT + BORDER;
    let room = ui.ctx().content_rect().bottom() - WINDOW_MARGIN - (anchor.rect.bottom() + GAP);
    let most = (room - above - below).clamp(BODY_MIN, BODY_MAX);

    let mut scroll_ui = crate::edge_scroll_ui(ui, GUTTER);
    // The height is the body's to say: a pop-up offers what it had room
    // for last frame (the other tab's height, coming from there).
    scroll_ui.set_max_height(most);
    let mut bar = scroll_ui.max_rect();
    bar.max.y -= BAR_FOOT;
    let mut scroll = egui::ScrollArea::vertical()
        .id_salt("tagmgr_settings_body")
        .max_height(most)
        .auto_shrink([false, true])
        .scroll_bar_rect(bar);
    // Closed and opened again, a tab is where it was left — but each run
    // of the app starts it at its top (where it was scrolled to is saved
    // with the app's settings; that it has been shown is not).
    let shown = shown_id(tab);
    if !ui.data(|d| d.get_temp::<bool>(shown).unwrap_or(false)) {
        ui.data_mut(|d| d.insert_temp(shown, true));
        scroll = scroll.vertical_scroll_offset(0.0);
    }
    scroll.show(&mut scroll_ui, |ui| {
        ui.set_width((ui.available_width() - GUTTER).max(0.0));
        add(ui);
        // The pop-up's padding under the last card, less what the body
        // already stops short of the edge by.
        ui.add_space(PADDING - BODY_FOOT);
    });
    crate::edge_scroll_done(ui, &scroll_ui, GUTTER);
}

/// Where it's noted, for this run of the app, that `tab` has been shown.
fn shown_id(tab: usize) -> egui::Id {
    egui::Id::new(("tagmgr_settings_shown", tab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slider::rig::Rig;
    use egui::{Event, Key};

    /// The pop-up, open under a stand-in for the gear at the top right of
    /// the rig's screen (640 × 360).
    struct Pop {
        /// The Tag Manager's faces are installed (the first frame does it).
        fonts: bool,
        settings: TagManagerSettings,
        models: crate::ai_models::ModelManager,
        /// The pop-up's frame, edge to edge.
        frame: egui::Rect,
        /// Where the gear is.
        gear: egui::Rect,
    }

    /// The gear: 36 points across, and high enough that either tab has
    /// room under it.
    const GEAR: egui::Rect = egui::Rect::from_min_max(egui::pos2(584.0, -32.0), egui::pos2(620.0, 4.0));
    const SCREEN_H: f32 = 360.0;

    fn scene(ui: &mut egui::Ui, p: &mut Pop) {
        if !p.fonts {
            let mut fonts = egui::FontDefinitions::default();
            crate::tag_manager::add_fonts(&mut fonts);
            ui.ctx().set_fonts(fonts);
            p.fonts = true;
            return;
        }
        p.models.tick(ui.ctx());
        let gear = ui.interact(p.gear, ui.id().with("gear"), egui::Sense::click());
        show(&gear, &mut p.settings, &mut p.models);
        let id = egui::Popup::default_response_id(&gear);
        p.frame = ui.ctx().memory(|m| m.area_rect(id)).unwrap_or(egui::Rect::NOTHING);
    }

    /// The pop-up open on `tab`, settled.
    fn pop(tab: usize) -> Rig<Pop> {
        let settings = TagManagerSettings { open: true, tab, ..Default::default() };
        let models = crate::ai_models::ModelManager::offline();
        let mut rig = Rig::new(Pop { fonts: false, settings, models, frame: egui::Rect::NOTHING, gear: GEAR }, scene);
        rig.idle();
        rig
    }

    /// Where `text` is painted (once).
    fn text(rig: &Rig<Pop>, text: &str) -> egui::Rect {
        let found = rig.texts(text);
        assert_eq!(found.len(), 1, "{text:?} is painted {} times", found.len());
        found[0]
    }

    /// Every text the last frame painted, with the bottom of what it was
    /// clipped to.
    fn painted(rig: &Rig<Pop>) -> Vec<(String, f32)> {
        fn find(shape: &egui::Shape, clip: f32, found: &mut Vec<(String, f32)>) {
            match shape {
                egui::Shape::Text(t) => found.push((t.galley.text().to_string(), clip)),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, clip, found)),
                _ => {}
            }
        }
        let mut found = vec![];
        rig.shapes.iter().for_each(|c| find(&c.shape, c.clip_rect.bottom(), &mut found));
        found
    }

    /// The wheel turned a long way down, over the pop-up; then the pointer
    /// off it, and settled.
    fn scroll_to_the_end(rig: &mut Rig<Pop>) {
        rig.frame(vec![Event::PointerMoved(rig.state.frame.center())]);
        for _ in 0..30 {
            rig.frame(vec![Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -120.0),
                phase: egui::TouchPhase::Move,
                modifiers: egui::Modifiers::NONE,
            }]);
        }
        rig.frame(vec![Event::PointerMoved(egui::pos2(60.0, 300.0))]);
        rig.idle();
    }

    /// The pop-up says each thing once.
    fn says_nothing_twice(rig: &Rig<Pop>) {
        // (Without the app's image loaders, the rig paints a "⚠" for each icon.)
        let mut said: Vec<String> = painted(rig).into_iter().map(|(text, _)| text).filter(|t| t != "⚠").collect();
        let all = said.len();
        said.sort();
        said.dedup();
        assert_eq!(said.len(), all, "a text is there twice: {said:?}");
    }

    /// The pop-up hangs under the gear, its right edge the gear's, as wide
    /// on one tab as on the other; on Models it's as tall as fits over the
    /// window's bottom edge. A failed download's message changes none of it
    /// (the old pop-up grew as wide as the message).
    #[test]
    fn the_pop_up_keeps_its_place_and_its_width() {
        let mut rig = pop(0);
        let hangs = |frame: egui::Rect, tab: &str| {
            assert!((frame.right() - GEAR.right()).abs() < 0.5, "{tab}: {frame:?}");
            assert!((frame.top() - (GEAR.bottom() + GAP)).abs() < 0.5, "{tab}: {frame:?}");
            assert!((frame.width() - WIDTH).abs() < 0.5, "{tab}: {} wide", frame.width());
        };
        hangs(rig.state.frame, "Settings");
        assert!(rig.state.frame.bottom() < SCREEN_H, "the test's screen has room for it");

        rig.state.settings.tab = 1;
        rig.idle();
        let models = rig.state.frame;
        hangs(models, "Models");
        assert!((models.bottom() - (SCREEN_H - WINDOW_MARGIN)).abs() < 0.5, "{models:?}");
        // Opened on Models (the tab it was last on), it's the same pop-up.
        let reopened = pop(1).state.frame;
        hangs(reopened, "Models, opened there");
        assert!((reopened.bottom() - models.bottom()).abs() < 0.5, "{reopened:?}");

        let message = "model.onnx: https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx: connection reset. ";
        rig.state.models.fail("joytag", &message.repeat(4));
        rig.idle();
        assert_eq!(rig.state.frame, models);
        assert!(painted(&rig).iter().all(|(text, _)| !text.contains("connection reset")));
    }

    /// Each tab shows its own, and nothing twice.
    #[test]
    fn the_tabs_switch_what_is_shown() {
        let mut rig = pop(0);
        for shown in ["Settings", "Models", "AUTO-TAG", "Confidence", "Blacklist", "0.35", "sensitive, nsfw"] {
            text(&rig, shown);
        }
        assert!(rig.texts("TAGGERS").is_empty());
        says_nothing_twice(&rig);

        rig.click(text(&rig, "Models").center());
        rig.idle();
        assert_eq!(rig.state.settings.tab, 1);
        for shown in ["Settings", "Models", "TAGGERS", "PixAI Tagger v0.9Recommended", "JoyTag"] {
            text(&rig, shown);
        }
        assert!(rig.texts("AUTO-TAG").is_empty() && rig.texts("Confidence").is_empty());
        says_nothing_twice(&rig);

        rig.click(text(&rig, "Settings").center());
        rig.idle();
        assert_eq!(rig.state.settings.tab, 0);
        text(&rig, "AUTO-TAG");
        assert!(rig.state.settings.open, "the tabs' clicks are inside the pop-up");
    }

    /// The slider sets the confidence, and the pill shows it: its rail runs
    /// from 0.10 at the card's one margin to 1.00 at the other.
    #[test]
    fn the_slider_sets_the_confidence_the_pill_shows() {
        let mut rig = pop(0);
        // The card's margins: the pop-up's edge and padding, then its own.
        let frame = rig.state.frame;
        let (left, right) = (frame.left() + BORDER + PADDING + 17.0, frame.right() - BORDER - PADDING - 17.0);
        let y = text(&rig, "Tags scoring below this are left out.").bottom() + 20.0;
        let at = |along: f32| egui::pos2(left + along * (right - left), y);
        for (along, value, shown) in [(0.5, 0.55, "0.55"), (0.0, 0.1, "0.10"), (1.0, 1.0, "1.00"), (0.8, 0.82, "0.82")] {
            rig.click(at(along));
            rig.idle();
            assert_eq!(rig.state.settings.default_threshold, value, "a click {along} along the rail");
            text(&rig, shown);
        }
        // As on the app's other sliders, a double-click goes back to the
        // usual value.
        rig.double_click(at(0.6));
        rig.idle();
        assert_eq!(rig.state.settings.default_threshold, 0.35);
        text(&rig, "0.35");
        assert!(rig.state.settings.open);
    }

    /// A click on the pill types a confidence: Enter takes it (put in
    /// range), Escape drops it — and only it: the pop-up stays open.
    #[test]
    fn a_typed_confidence_is_taken_on_enter_and_dropped_on_escape() {
        let mut rig = pop(0);
        let pill = text(&rig, "0.35").center();
        let type_in = |rig: &mut Rig<Pop>, typed: &str, end: Key| {
            rig.click(pill);
            rig.idle();
            rig.type_text(typed);
            rig.key(end);
            rig.idle();
            rig.state.settings.default_threshold
        };
        assert_eq!(type_in(&mut rig, "0.6", Key::Enter), 0.6);
        text(&rig, "0.60");
        assert_eq!(type_in(&mut rig, "0,4", Key::Enter), 0.4, "a comma for the point");
        assert_eq!(type_in(&mut rig, "7", Key::Enter), 1.0, "no higher than a score goes");
        assert_eq!(type_in(&mut rig, "0", Key::Enter), 0.1, "and no lower than the slider");
        assert_eq!(type_in(&mut rig, "lots", Key::Enter), 0.1, "not a number: as it was");
        assert!(rig.state.settings.open);

        assert_eq!(type_in(&mut rig, "0.9", Key::Escape), 0.1, "Escape drops what was typed");
        assert!(rig.state.settings.open, "and leaves the pop-up open");
        text(&rig, "0.10");
        rig.key(Key::Escape);
        rig.idle();
        assert!(!rig.state.settings.open, "Escape again closes it");
    }

    #[test]
    fn a_typed_confidence_is_a_score_in_range() {
        for (typed, score) in [("0.4", Some(0.4)), (" .4 ", Some(0.4)), ("0,4", Some(0.4)), ("0.456", Some(0.46))] {
            assert_eq!(typed_confidence(typed), score, "{typed:?}");
        }
        for (typed, score) in [("35", Some(1.0)), ("-1", Some(0.1)), ("", None), ("0.4.1", None), ("NaN", None), ("inf", None)] {
            assert_eq!(typed_confidence(typed), score, "{typed:?}");
        }
    }

    /// What's typed in the box is the blacklist.
    #[test]
    fn what_is_typed_in_the_box_is_the_blacklist() {
        let mut rig = pop(0);
        rig.state.settings.blacklist.clear();
        rig.idle();
        let caption = text(&rig, "Tags Auto-tag never adds. Separate them with commas.");
        rig.click(egui::pos2(caption.center().x, caption.bottom() + 40.0));
        rig.idle();
        rig.type_text("watermark, signature");
        rig.idle();
        assert_eq!(rig.state.settings.blacklist, "watermark, signature");
        text(&rig, "watermark, signature");
        assert!(rig.state.settings.open);
    }

    /// The blacklist's box is two lines tall, grows with its text up to six
    /// and then scrolls: the pop-up doesn't grow with every line (a long
    /// list made it taller than the window).
    #[test]
    fn the_blacklist_box_grows_to_six_lines_then_scrolls() {
        struct Field {
            fonts: bool,
            text: String,
            height: f32,
            line: f32,
        }
        fn scene(ui: &mut egui::Ui, f: &mut Field) {
            if !f.fonts {
                let mut fonts = egui::FontDefinitions::default();
                crate::tag_manager::add_fonts(&mut fonts);
                ui.ctx().set_fonts(fonts);
                f.fonts = true;
                return;
            }
            ui.set_width(320.0);
            blacklist_box(ui, &mut f.text);
            f.height = ui.min_rect().height();
            // A line of the text, as lines of it are laid out.
            let tall = |lines: &str| ui.fonts_mut(|f| f.layout_no_wrap(lines.into(), medium(13.0), TEXT()).size().y);
            f.line = tall("a\nb") - tall("a");
        }
        let mut rig = Rig::new(Field { fonts: false, text: String::new(), height: 0.0, line: 0.0 }, scene);
        rig.idle();
        let line = rig.state.line;
        let mut height = |lines: usize| {
            rig.state.text = (0..lines).map(|i| format!("tag_{i}")).collect::<Vec<_>>().join(",\n");
            rig.idle();
            rig.state.height
        };
        let two = height(2);
        // The well's padding over and under its lines.
        assert!((two - (2.0 * line + 20.0)).abs() < 1.0, "{two}, with lines of {line}");
        assert_eq!(height(0), two, "two lines tall while empty");
        assert_eq!(height(1), two);
        assert!((height(3) - (two + line)).abs() < 1.0, "a third line makes it a line taller");
        let most = height(6);
        assert!((most - (two + 4.0 * line)).abs() < 1.0, "{most}: six lines");
        assert_eq!(height(7), most, "a seventh scrolls");
        assert_eq!(height(40), most);
    }

    /// The Models list scrolls inside the pop-up: it starts under the tabs
    /// and is cut off just inside the bottom edge; scrolled to its end, the
    /// last card stops the pop-up's padding above that edge.
    #[test]
    fn the_models_list_scrolls_inside_the_pop_up() {
        let mut rig = pop(1);
        let frame = rig.state.frame;
        text(&rig, "TAGGERS");
        assert!(rig.texts("TOOLS").is_empty(), "further down than the pop-up is tall");
        let inside = |rig: &Rig<Pop>| {
            for (said, clip) in painted(rig) {
                if !TABS.contains(&said.as_str()) {
                    assert!(clip <= frame.bottom() - 2.5, "{said:?} can paint down to {clip}: {frame:?}");
                }
            }
        };
        inside(&rig);

        scroll_to_the_end(&mut rig);
        assert_eq!(rig.state.frame, frame, "the pop-up stays as it is");
        assert!(rig.texts("TAGGERS").is_empty(), "scrolled away");
        inside(&rig);
        // The last row's caption sits 9 points over the row's end, and the
        // card's outline is 1 more.
        let last = text(&rig, "148 MB · speech to text for the chat");
        let card_ends = frame.bottom() - BORDER - PADDING;
        assert!((card_ends - 10.0 - last.bottom()).abs() < 3.0, "{last:?}: the card ends at {card_ends}");

        // Closed and opened again, it's where it was left…
        rig.state.settings.open = false;
        rig.idle();
        assert!(rig.texts("Settings").is_empty(), "closed");
        rig.state.settings.open = true;
        rig.idle();
        assert_eq!(text(&rig, "148 MB · speech to text for the chat"), last);
        // …but a new run of the app, which keeps where the list was
        // scrolled to and not that it has been shown, starts at its top.
        rig.ctx.data_mut(|d| d.remove_temp::<bool>(shown_id(1)));
        rig.idle();
        text(&rig, "TAGGERS");
        assert!(rig.texts("148 MB · speech to text for the chat").is_empty());
    }

    /// Short of room under the gear for all of Settings — a long blacklist,
    /// a short window — the tab scrolls inside the pop-up, which stays under
    /// its button (it used to move to wherever it fitted, the tabs with it).
    #[test]
    fn short_of_room_settings_scroll_under_the_gear() {
        let mut rig = pop(0);
        let whole = rig.state.frame;
        let fits = SCREEN_H - WINDOW_MARGIN;
        assert!(whole.bottom() < fits - 10.0, "all of it fits to begin with: {whole:?}");

        rig.state.settings.blacklist = (0..40).map(|i| format!("tag_{i}")).collect::<Vec<_>>().join(",\n");
        rig.idle();
        let long = rig.state.frame;
        assert_eq!((long.left(), long.top(), long.right()), (whole.left(), whole.top(), whole.right()));
        assert!((long.bottom() - fits).abs() < 0.5, "{long:?}: down to {fits} and no further");
        text(&rig, "AUTO-TAG");

        // The gear lower in the window: less room under it.
        let mut rig = pop(0);
        rig.state.gear = GEAR.translate(egui::vec2(0.0, 80.0));
        rig.idle();
        let short = rig.state.frame;
        assert!((short.top() - (GEAR.bottom() + 80.0 + GAP)).abs() < 0.5, "{short:?}");
        assert!((short.right() - GEAR.right()).abs() < 0.5 && (short.width() - WIDTH).abs() < 0.5, "{short:?}");
        assert!((short.bottom() - fits).abs() < 0.5, "{short:?}");
        // The blacklist is past the pop-up's end; the wheel brings it in.
        let shows = |rig: &Rig<Pop>| rig.texts("sensitive, nsfw").iter().any(|at| short.contains_rect(*at));
        assert!(!shows(&rig));
        scroll_to_the_end(&mut rig);
        assert!(shows(&rig), "{:?} in {short:?}", rig.texts("sensitive, nsfw"));
        assert_eq!(rig.state.frame, short);
    }

    /// Escape, or a click outside the pop-up, closes it.
    #[test]
    fn escape_or_a_click_outside_closes_it() {
        let mut rig = pop(0);
        rig.key(Key::Escape);
        rig.idle();
        assert!(!rig.state.settings.open);

        rig.state.settings.open = true;
        rig.idle();
        text(&rig, "AUTO-TAG");
        rig.click(text(&rig, "Blacklist").center());
        rig.idle();
        assert!(rig.state.settings.open, "a click inside leaves it open");
        rig.click(egui::pos2(60.0, 300.0));
        rig.idle();
        assert!(!rig.state.settings.open);
    }
}
