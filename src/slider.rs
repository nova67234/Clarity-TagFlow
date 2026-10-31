//! The app's slider: a value pill on the left — click it to type an exact
//! number — and a periwinkle track: an active fill with a soft glow, a dot per
//! stop on sliders with ten stops or fewer, a notch at the default value (a
//! drag close to it lands on it, Shift drags past it, a double-click goes back
//! to it), and a round white thumb that grows a halo while it's hovered,
//! dragged or focused.
//!
//! A drop-in for `egui::Slider`:
//! `ui.add(PillSlider::new(&mut value, 0.0..=2.0).step_by(0.05).default_value(1.0))`.
//! As with egui's, the mouse doesn't take keyboard focus — Tab does, and then
//! ←/→ step the value — a drag picks round numbers, and screen readers can set
//! and step it. The value stored is always the one the pill shows. The track
//! is `ui.spacing().slider_width` wide, or narrower when the row has less room;
//! the pill is as wide as the widest value the slider can show, so nothing
//! moves as the value changes.

use std::ops::RangeInclusive;

use eframe::egui::{
    self, Align, Align2, Color32, Direction, EventFilter, FontId, Key, Layout, Pos2, Rect, Response, Sense, Stroke,
    StrokeKind, Ui, UiBuilder, Vec2, Widget, WidgetInfo,
    accesskit::{Action, ActionData},
    emath::{self, Numeric},
};

use crate::theme::{TEXT, is_light};

/// Periwinkle (#5b8def), and the lighter tone it takes while hot.
const PERIWINKLE: Color32 = Color32::from_rgb(91, 141, 239);
const PERIWINKLE_LIT: Color32 = Color32::from_rgb(122, 172, 255);

/// The value pill's height, its text size, and the gap before the track.
const PILL_H: f32 = 24.0;
const PILL_TEXT: f32 = 13.0;
const GAP: f32 = 12.0;
/// The slider's height (the track's hit area), and the rail's thickness.
/// Taller than a label: a row with a label beside the slider sets this as
/// its minimum height so the label centres on it.
pub const HEIGHT: f32 = 26.0;
const RAIL_H: f32 = 6.0;
/// Room at each end of the rail so the thumb never overhangs the hit area.
const THUMB_PAD: f32 = 8.0;
/// The narrowest the track gets in a row without room for all of it.
const MIN_TRACK: f32 = 60.0;
/// A drag this close to the default's notch (a share of the track) lands on
/// the default.
const PULL: f32 = 0.025;
/// Sliders with this many stops or fewer show a dot at each.
const MAX_DOTS: usize = 10;

/// A slider in the app's periwinkle style; see the module docs.
pub struct PillSlider<'a> {
    get_set: Box<dyn 'a + FnMut(Option<f64>) -> f64>,
    min: f64,
    max: f64,
    integral: bool,
    step: Option<f64>,
    default: Option<f64>,
    suffix: String,
    decimals: Option<usize>,
    show_value: bool,
}

impl<'a> PillSlider<'a> {
    pub fn new<N: Numeric>(value: &'a mut N, range: RangeInclusive<N>) -> Self {
        Self {
            get_set: Box::new(move |v| {
                if let Some(v) = v {
                    *value = N::from_f64(v);
                }
                value.to_f64()
            }),
            min: range.start().to_f64(),
            max: range.end().to_f64(),
            integral: N::INTEGRAL,
            step: N::INTEGRAL.then_some(1.0),
            default: None,
            suffix: String::new(),
            decimals: None,
            show_value: true,
        }
    }

    /// Values land on multiples of `step` counted from the minimum.
    pub fn step_by(mut self, step: f64) -> Self {
        self.step = (step > 0.0).then_some(step);
        self
    }

    /// The value the notch marks: drags close to it land on it and a
    /// double-click on the track resets to it.
    pub fn default_value(mut self, value: f64) -> Self {
        self.default = Some(value);
        self
    }

    /// Text after the number in the pill (" px").
    pub fn suffix(mut self, suffix: impl ToString) -> Self {
        self.suffix = suffix.to_string();
        self
    }

    /// Decimal places shown — and kept: values are rounded to them.
    pub fn fixed_decimals(mut self, decimals: usize) -> Self {
        self.decimals = Some(decimals);
        self
    }

    /// Without the pill: just the track.
    pub fn show_value(mut self, show: bool) -> Self {
        self.show_value = show;
        self
    }

    /// Decimal places shown: as set, else none for whole numbers, else as
    /// many as the step has (two without a step).
    fn decimals(&self) -> usize {
        self.decimals.unwrap_or_else(|| match self.step {
            _ if self.integral => 0,
            Some(step) => step_decimals(step),
            None => 2,
        })
    }
}

/// `v` put on a slider from `min` to `max` (either way round): within range,
/// on a step, and rounded to the decimals the pill shows.
fn snap_to(v: f64, min: f64, max: f64, step: Option<f64>, decimals: usize) -> f64 {
    let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
    let mut v = v.clamp(lo, hi);
    if let Some(step) = step {
        v = (min + ((v - min) / step).round() * step).clamp(lo, hi);
    }
    emath::round_to_decimals(v, decimals).clamp(lo, hi)
}

/// How many decimal places `step` needs (0.05 → 2, 0.5 → 1, 10 → 0).
fn step_decimals(step: f64) -> usize {
    let mut scaled = step;
    let mut places = 0;
    while places < 6 && (scaled - scaled.round()).abs() > 1e-9 * scaled.abs().max(1.0) {
        scaled *= 10.0;
        places += 1;
    }
    places
}

/// What was typed in the pill, as a number: the suffix, spaces and thousands
/// separators dropped. A comma is the decimal point when it's the only
/// separator on a slider with decimals ("2,5"); otherwise it separates
/// thousands ("50,000", "1,234.5").
fn parse_typed(text: &str, suffix: &str, integral: bool) -> Option<f64> {
    let text = text.trim();
    let text = text.strip_suffix(suffix.trim()).unwrap_or(text);
    let mut number: String = text.chars().filter(|c| !c.is_whitespace() && !matches!(c, '_' | '\'')).collect();
    if integral || number.contains('.') || number.matches(',').count() > 1 {
        number.retain(|c| c != ',');
    } else {
        number = number.replace(',', ".");
    }
    number.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// The pill while it's a text field: what's typed, the number it showed when
/// opened (left as it was, the value isn't touched), and whether this is its
/// first frame (focus it, select it all).
#[derive(Clone, Default)]
struct Edit {
    text: String,
    shown: String,
    fresh: bool,
}

/// The colours that follow the theme (the periwinkle doesn't).
struct Palette {
    rail: Color32,
    rail_edge: Color32,
    notch: Color32,
    dot: Color32,
    pill: Color32,
    pill_hot: Color32,
    pill_edge: Color32,
    pill_text: Color32,
    shadow: Color32,
}

fn palette() -> Palette {
    if is_light() {
        Palette {
            rail: Color32::from_rgb(208, 214, 226),
            rail_edge: Color32::from_black_alpha(30),
            notch: Color32::from_black_alpha(80),
            dot: Color32::from_black_alpha(70),
            pill: Color32::from_rgb(234, 238, 246),
            pill_hot: Color32::from_rgb(220, 231, 252),
            pill_edge: Color32::from_black_alpha(26),
            pill_text: TEXT(),
            shadow: Color32::from_black_alpha(60),
        }
    } else {
        Palette {
            rail: Color32::from_rgb(56, 60, 70),
            rail_edge: Color32::from_black_alpha(90),
            notch: Color32::from_white_alpha(64),
            dot: Color32::from_white_alpha(55),
            pill: Color32::from_rgb(58, 62, 72),
            pill_hot: Color32::from_rgb(48, 64, 96),
            pill_edge: Color32::from_white_alpha(20),
            pill_text: Color32::from_rgb(240, 244, 252),
            shadow: Color32::from_black_alpha(120),
        }
    }
}

impl Widget for PillSlider<'_> {
    fn ui(mut self, ui: &mut Ui) -> Response {
        let (min, max, step, integral) = (self.min, self.max, self.step, self.integral);
        // Signed, so a reversed range (10..=0) runs from 10 on the left.
        let span = if (max - min).abs() > f64::EPSILON { max - min } else { f64::EPSILON };
        let decimals = self.decimals();
        let snap = |v: f64| snap_to(v, min, max, step, decimals);
        let default = self.default.map(snap);
        let suffix = std::mem::take(&mut self.suffix);
        let show = |v: f64| format!("{v:.decimals$}{suffix}");
        let font = FontId::proportional(PILL_TEXT);
        let colors = palette();

        // The pill fits the widest value the slider can show (digits counted
        // as zeros), so it keeps its width as the value changes.
        let pill_w = if self.show_value {
            let widest = [Some(min), Some(max), default]
                .into_iter()
                .flatten()
                .map(|v| {
                    let sample: String = show(v).chars().map(|c| if c.is_ascii_digit() { '0' } else { c }).collect();
                    ui.painter().layout_no_wrap(sample, font.clone(), Color32::WHITE).size().x
                })
                .fold(0.0_f32, f32::max);
            (widest + 24.0).max(48.0)
        } else {
            0.0
        };
        let gap = if self.show_value { GAP } else { 0.0 };
        // The track: slider_width, or what the row has left beside the pill.
        let track_w = ui.spacing().slider_width.min(ui.available_width() - pill_w - gap).max(MIN_TRACK);
        let (id, rect) = ui.allocate_space(Vec2::new(pill_w + gap + track_w, HEIGHT));
        let pill_rect = Rect::from_center_size(
            Pos2::new(rect.left() + pill_w / 2.0, rect.center().y),
            Vec2::new(pill_w, PILL_H),
        );
        let track_rect = Rect::from_min_max(Pos2::new(rect.right() - track_w, rect.top()), rect.max);

        let mut value = (self.get_set)(None);
        let mut changed = false;
        // Puts `v` on the slider, noting a change the stored type can hold
        // (an f32 doesn't change for float noise below its precision).
        let mut set = |v: f64, value: &mut f64| {
            let stored = (self.get_set)(Some(v));
            if stored != *value {
                *value = stored;
                changed = true;
            }
        };

        // --- The pill: shows the value; a click turns it into a text field. ---
        let edit_id = id.with("edit");
        let editing = if self.show_value { ui.data(|d| d.get_temp::<Edit>(edit_id)) } else { None };
        let mut pill_response = None;
        if let Some(mut edit) = editing {
            let painter = ui.painter();
            painter.rect_filled(pill_rect, PILL_H / 2.0, colors.pill_hot);
            painter.rect_stroke(pill_rect, PILL_H / 2.0, Stroke::new(1.0_f32, PERIWINKLE), StrokeKind::Inside);
            // The field sits in a child Ui over the pill, outside the parent's
            // layout, so nothing after the slider moves while it's open.
            let field = {
                let mut field_ui = ui.new_child(
                    UiBuilder::new()
                        .max_rect(pill_rect.shrink2(Vec2::new(8.0, 0.0)))
                        .layout(Layout::centered_and_justified(Direction::TopDown)),
                );
                let mut out = egui::TextEdit::singleline(&mut edit.text)
                    .id(id.with("field"))
                    .frame(egui::Frame::NONE)
                    .font(font.clone())
                    .text_color(colors.pill_text)
                    .horizontal_align(Align::Center)
                    .vertical_align(Align::Center)
                    .desired_width(f32::INFINITY)
                    .show(&mut field_ui);
                let field = out.response.response;
                if edit.fresh {
                    // Start with the number selected, ready to type over.
                    field.request_focus();
                    let all = egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(edit.text.chars().count()),
                    );
                    out.state.cursor.set_char_range(Some(all));
                    out.state.store(ui.ctx(), field.id);
                }
                field
            };
            let escape = ui.input(|i| i.key_pressed(Key::Escape));
            let enter = ui.input(|i| i.key_pressed(Key::Enter));
            // Neither focused nor just left: the field went away mid-edit
            // (Settings closed) and is back — drop the stale text.
            let stale = !edit.fresh && !field.has_focus() && !field.lost_focus();
            if escape || stale {
                ui.data_mut(|d| d.remove::<Edit>(edit_id));
            } else if enter || field.lost_focus() {
                // Only what was typed: opening the pill and leaving it as it
                // was keeps the value, unrounded or not.
                if edit.text.trim() != edit.shown
                    && let Some(v) = parse_typed(&edit.text, &suffix, integral)
                {
                    set(snap(v), &mut value);
                }
                ui.data_mut(|d| d.remove::<Edit>(edit_id));
            } else {
                edit.fresh = false;
                ui.data_mut(|d| d.insert_temp(edit_id, edit));
            }
        } else {
            // The field's child Ui takes an auto id; skip one without it so
            // the widgets after this slider keep theirs while it's open.
            ui.skip_ahead_auto_ids(1);
            if self.show_value {
                let pill = ui.interact(pill_rect, id.with("pill"), Sense::click()).on_hover_text("Click to type a value");
                if pill.clicked() {
                    let shown = format!("{value:.decimals$}");
                    ui.data_mut(|d| d.insert_temp(edit_id, Edit { text: shown.clone(), shown, fresh: true }));
                }
                pill_response = Some(pill);
            }
        }

        // --- The track: press or drag to set, double-click for the default,
        // arrow keys to step once Tab has given it focus. ---
        let mut response = ui.interact(track_rect, id.with("track"), Sense::click_and_drag());
        let left = track_rect.left() + THUMB_PAD;
        let usable_w = (track_rect.width() - THUMB_PAD * 2.0).max(1.0);
        let center_y = track_rect.center().y;
        let x_of = |v: f64| left + ((v - min) / span).clamp(0.0, 1.0) as f32 * usable_w;
        let value_at = |x: f32| min + ((x - left) / usable_w).clamp(0.0, 1.0) as f64 * span;
        if response.double_clicked()
            && let Some(default) = default
        {
            set(default, &mut value);
        } else if let Some(pointer) = response.interact_pointer_pos() {
            // As egui's slider does: the roundest value within the pointer's
            // reach (a physical pixel either side).
            let aim = ui.input(|i| i.aim_radius());
            let mut v = emath::smart_aim::best_in_range_f64(value_at(pointer.x - aim), value_at(pointer.x + aim));
            // Close to the notch lands on the default — but on a stepped
            // slider never so close that the stops beside it can't be reached.
            if let Some(default) = default
                && !ui.input(|i| i.modifiers.shift)
            {
                let pull = match step {
                    Some(step) => (PULL * usable_w).min(1.5 * (step / span.abs()) as f32 * usable_w - 1.0),
                    None => PULL * usable_w,
                };
                if (pointer.x - x_of(default)).abs() <= pull {
                    v = default;
                }
            }
            set(snap(v), &mut value);
        }
        let mut steps = 0.0;
        if response.has_focus() {
            // Left/right move the thumb instead of moving focus.
            ui.memory_mut(|m| {
                m.set_focus_lock_filter(response.id, EventFilter { horizontal_arrows: true, ..Default::default() });
            });
            let right = ui.input(|i| i.num_presses(Key::ArrowRight) as f64 - i.num_presses(Key::ArrowLeft) as f64);
            steps += right * span.signum();
        }
        // Screen readers: increment, decrement, set.
        let (more, less, set_to) = ui.input(|i| {
            let set_to = i
                .accesskit_action_requests(response.id, Action::SetValue)
                .filter_map(|r| match r.data {
                    Some(ActionData::NumericValue(v)) => Some(v),
                    _ => None,
                })
                .last();
            (
                i.num_accesskit_action_requests(response.id, Action::Increment),
                i.num_accesskit_action_requests(response.id, Action::Decrement),
                set_to,
            )
        });
        if ui.is_enabled() {
            steps += more as f64 - less as f64;
            if steps != 0.0 {
                // A step; without steps, a hundredth of the range — at least
                // one shown decimal, so the rounding can't swallow it.
                let unit = step.unwrap_or_else(|| (span.abs() / 100.0).max(10f64.powi(-(decimals as i32))));
                set(snap(value + steps * unit), &mut value);
            }
            if let Some(v) = set_to.filter(|v| v.is_finite()) {
                set(snap(v), &mut value);
            }
        }

        // --- Paint. ---
        let hot = response.hovered() || response.dragged() || response.has_focus();
        let hot_t = ui.ctx().animate_bool(id.with("hot"), hot);
        let drag_t = ui.ctx().animate_bool(id.with("drag"), response.dragged());
        let painter = ui.painter();

        if let Some(pill) = &pill_response {
            let pill_t = if pill.hovered() || pill.has_focus() { 1.0 } else { hot_t };
            painter.rect_filled(pill_rect, PILL_H / 2.0, lerp_color(colors.pill, colors.pill_hot, pill_t * 0.65));
            painter.rect_stroke(
                pill_rect,
                PILL_H / 2.0,
                Stroke::new(1.0_f32, lerp_color(colors.pill_edge, PERIWINKLE, pill_t)),
                StrokeKind::Inside,
            );
            painter.text(pill_rect.center(), Align2::CENTER_CENTER, show(value), font, colors.pill_text);
        }

        let thumb_x = x_of(value);
        let thumb = Pos2::new(thumb_x, center_y);
        let rail = Rect::from_min_max(
            Pos2::new(left, center_y - RAIL_H / 2.0),
            Pos2::new(left + usable_w, center_y + RAIL_H / 2.0),
        );

        // The recessed rail.
        painter.rect_filled(rail, RAIL_H / 2.0, colors.rail);
        painter.rect_stroke(rail, RAIL_H / 2.0, Stroke::new(1.0_f32, colors.rail_edge), StrokeKind::Inside);

        // The default's notch.
        if let Some(default) = default {
            let x = x_of(default);
            painter.line_segment(
                [Pos2::new(x, rail.top() - 3.0), Pos2::new(x, rail.bottom() + 3.0)],
                Stroke::new(1.5_f32, colors.notch),
            );
        }

        // The filled part: a soft bloom, the fill, a specular line on top.
        if thumb_x > left + 1.0 {
            let fill = Rect::from_min_max(rail.min, Pos2::new(thumb_x, rail.bottom()));
            let bloom = (32.0 + hot_t * 36.0) as u8;
            painter.rect_filled(
                fill.expand(2.5),
                (RAIL_H + 5.0) / 2.0,
                Color32::from_rgba_unmultiplied(91, 141, 239, bloom),
            );
            painter.rect_filled(fill, RAIL_H / 2.0, lerp_color(PERIWINKLE, PERIWINKLE_LIT, hot_t));
            painter.line_segment(
                [Pos2::new(fill.left() + 2.0, fill.top() + 1.0), Pos2::new(fill.right() - 1.0, fill.top() + 1.0)],
                Stroke::new(1.0_f32, Color32::from_white_alpha(75)),
            );
        }

        // A dot at each stop, on sliders with few.
        if let Some(step) = step {
            let stops = (span.abs() / step).round() as usize;
            if (1..=MAX_DOTS).contains(&stops) {
                for i in 0..=stops {
                    let x = left + i as f32 / stops as f32 * usable_w;
                    let color = if x <= thumb_x + 0.5 { Color32::from_white_alpha(180) } else { colors.dot };
                    painter.circle_filled(Pos2::new(x, center_y), 1.6, color);
                }
            }
        }

        // The thumb: a halo while hot, a drop shadow, the white disc, a
        // periwinkle rim and a centre dot.
        if hot_t > 0.01 {
            painter.circle_filled(
                thumb,
                8.0 + hot_t * 4.5 + drag_t * 1.5,
                Color32::from_rgba_unmultiplied(91, 141, 239, (hot_t * 68.0) as u8),
            );
        }
        // At most THUMB_PAD across from its centre, so it never overhangs.
        let thumb_r = 7.0 + hot_t * 0.75 + drag_t * 0.25;
        painter.circle_filled(thumb + Vec2::new(0.0, 2.0), thumb_r, colors.shadow);
        painter.circle_filled(thumb, thumb_r, Color32::from_rgb(246, 249, 255));
        painter.circle_stroke(
            thumb,
            thumb_r - 0.8,
            Stroke::new(1.6_f32, lerp_color(PERIWINKLE, Color32::from_rgb(140, 188, 255), hot_t)),
        );
        painter.circle_filled(thumb, 2.2, Color32::from_rgba_unmultiplied(72, 122, 230, (140.0 + hot_t * 95.0) as u8));

        if let Some(pill) = pill_response {
            response |= pill;
        }
        // The whole widget, pill or text field included, so callers can tell
        // clicks on it from clicks around it.
        response.rect = rect;
        if changed {
            response.mark_changed();
        }
        response.widget_info(|| WidgetInfo::slider(ui.is_enabled(), value, ""));
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        ui.ctx().accesskit_node_builder(response.id, |node| {
            node.set_min_numeric_value(lo);
            node.set_max_numeric_value(hi);
            if let Some(step) = step {
                node.set_numeric_value_step(step);
            }
            node.add_action(Action::SetValue);
            if value < hi {
                node.add_action(Action::Increment);
            }
            if value > lo {
                node.add_action(Action::Decrement);
            }
        });
        response
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let [ar, ag, ab, aa] = a.to_srgba_unmultiplied();
    let [br, bg, bb, ba] = b.to_srgba_unmultiplied();
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(mix(ar, br), mix(ag, bg), mix(ab, bb), mix(aa, ba))
}

/// A headless egui for tests here and in the screens that use the slider,
/// shown frame by frame and driven with raw input.
#[cfg(test)]
pub(crate) mod rig {
    use eframe::egui::{self, Event, Key, Modifiers, PointerButton, Pos2, Rect, Ui, Vec2};

    use super::THUMB_PAD;

    /// `scene` drawn into a 640 × 360 screen each frame, keeping what it
    /// needs (values, where things landed) in `state`.
    pub(crate) struct Rig<S> {
        pub ctx: egui::Context,
        pub state: S,
        /// Shift held for the next frames.
        pub shift: bool,
        /// The last frame asked for another one straight away.
        pub repaint: bool,
        /// What the last frame painted.
        pub shapes: Vec<egui::epaint::ClippedShape>,
        scene: fn(&mut Ui, &mut S),
        time: f64,
    }

    impl<S> Rig<S> {
        pub fn new(state: S, scene: fn(&mut Ui, &mut S)) -> Self {
            let ctx = egui::Context::default();
            let mut rig = Rig { ctx, state, shift: false, repaint: false, shapes: vec![], scene, time: 0.0 };
            rig.frame(vec![]);
            rig
        }

        /// Where the last frame painted `text`, each time it did.
        pub fn texts(&self, text: &str) -> Vec<Rect> {
            fn find(shape: &egui::Shape, text: &str, found: &mut Vec<Rect>) {
                match shape {
                    egui::Shape::Text(t) if t.galley.text() == text => found.push(t.visual_bounding_rect()),
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, text, found)),
                    _ => {}
                }
            }
            let mut found = vec![];
            self.shapes.iter().for_each(|c| find(&c.shape, text, &mut found));
            found
        }

        pub fn frame(&mut self, events: Vec<Event>) {
            self.time += 1.0 / 60.0;
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 360.0))),
                time: Some(self.time),
                modifiers: if self.shift { Modifiers::SHIFT } else { Modifiers::NONE },
                events,
                ..Default::default()
            };
            let (scene, state) = (self.scene, &mut self.state);
            let out = self.ctx.run_ui(input, |ui| scene(ui, state));
            self.repaint = out.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|v| v.repaint_delay.is_zero());
            self.shapes = out.shapes;
        }

        /// Settle: a few frames with nothing new, then a pause long enough
        /// that the next click isn't taken as a double-click.
        pub fn idle(&mut self) {
            for _ in 0..3 {
                self.frame(vec![]);
            }
            self.time += 1.0;
            self.frame(vec![]);
        }

        pub fn press(&mut self, pos: Pos2, pressed: bool) {
            self.frame(vec![
                Event::PointerMoved(pos),
                Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE },
            ]);
        }

        pub fn click(&mut self, pos: Pos2) {
            self.press(pos, true);
            self.press(pos, false);
        }

        pub fn double_click(&mut self, pos: Pos2) {
            self.click(pos);
            self.click(pos);
        }

        pub fn key(&mut self, key: Key) {
            self.frame(vec![Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }]);
        }

        pub fn type_text(&mut self, text: &str) {
            self.frame(vec![Event::Text(text.into())]);
        }

        /// Click a slider's pill, type, finish with `end`, settle.
        pub fn type_in_pill(&mut self, slider: Rect, text: &str, end: Key) {
            self.click(pill(slider));
            self.idle();
            self.type_text(text);
            self.key(end);
            self.idle();
        }
    }

    /// The middle of the pill of the slider at `slider`.
    pub fn pill(slider: Rect) -> Pos2 {
        Pos2::new(slider.left() + 20.0, slider.center().y)
    }

    /// The point `t` (0 to 1) along the track of the slider at `slider`,
    /// whose track is `track_w` wide.
    pub fn track(slider: Rect, track_w: f32, t: f32) -> Pos2 {
        let left = slider.right() - track_w + THUMB_PAD;
        let right = slider.right() - THUMB_PAD;
        Pos2::new(left + t * (right - left), slider.center().y)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use egui::{Event, Id, accesskit};

    use super::rig::{Rig, pill, track};
    use super::*;

    /// Sliders' values and where the last frame put them.
    struct Sliders {
        values: Vec<f32>,
        whole: i32,
        rects: Vec<Rect>,
        ids: Vec<Id>,
        /// A slider reported a change in the last frame, or ever.
        changed: bool,
        ever_changed: bool,
        /// Frames run without the sliders (as when Settings is closed).
        hidden: bool,
        /// Whatever the scene put beside the sliders.
        beside: Rect,
    }

    impl Default for Sliders {
        fn default() -> Self {
            Sliders {
                values: vec![],
                whole: 0,
                rects: vec![],
                ids: vec![],
                changed: false,
                ever_changed: false,
                hidden: false,
                beside: Rect::NOTHING,
            }
        }
    }

    impl Sliders {
        fn with(values: &[f32]) -> Self {
            Sliders { values: values.to_vec(), ..Default::default() }
        }

        fn note(&mut self, i: usize, r: &Response) {
            if self.rects.len() <= i {
                self.rects.resize(i + 1, Rect::NOTHING);
                self.ids.resize(i + 1, Id::NULL);
            }
            (self.rects[i], self.ids[i]) = (r.rect, r.id);
            self.changed |= r.changed();
            self.ever_changed |= r.changed();
        }
    }

    /// 0 to 10 in whole steps, default 5, a 120-point track.
    fn whole_steps(ui: &mut Ui, s: &mut Sliders) {
        s.changed = false;
        if s.hidden {
            return;
        }
        ui.spacing_mut().slider_width = 120.0;
        let r = ui.add(PillSlider::new(&mut s.values[0], 0.0..=10.0).step_by(1.0).default_value(5.0));
        s.note(0, &r);
    }

    /// 0 to 10 without steps, default 5, a 120-point track.
    fn no_steps(ui: &mut Ui, s: &mut Sliders) {
        s.changed = false;
        ui.spacing_mut().slider_width = 120.0;
        let r = ui.add(PillSlider::new(&mut s.values[0], 0.0..=10.0).default_value(5.0));
        s.note(0, &r);
    }

    /// Two whole-step sliders side by side, then a label.
    fn side_by_side(ui: &mut Ui, s: &mut Sliders) {
        s.changed = false;
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = 120.0;
            for i in 0..2 {
                let r = ui.add(PillSlider::new(&mut s.values[i], 0.0..=10.0).step_by(1.0));
                s.note(i, &r);
            }
            s.beside = ui.label("after").rect;
        });
    }

    /// Like Pixal3D's steps: whole numbers 1 to 50, default 12, a 90-point track.
    fn steps_1_to_50(ui: &mut Ui, s: &mut Sliders) {
        ui.spacing_mut().slider_width = 90.0;
        let r = ui.add(PillSlider::new(&mut s.whole, 1..=50).default_value(12.0));
        s.note(0, &r);
    }

    /// 10 down to 0, in whole steps.
    fn reversed(ui: &mut Ui, s: &mut Sliders) {
        ui.spacing_mut().slider_width = 120.0;
        let r = ui.add(PillSlider::new(&mut s.values[0], 10.0..=0.0).step_by(1.0));
        s.note(0, &r);
    }

    /// A 220-point row: a label, then a slider that would like 200 points of track.
    fn narrow_row(ui: &mut Ui, s: &mut Sliders) {
        let row = Rect::from_min_size(ui.cursor().min, Vec2::new(220.0, 40.0));
        ui.scope_builder(UiBuilder::new().max_rect(row).layout(Layout::left_to_right(Align::Center)), |ui| {
            ui.label("Weight");
            ui.spacing_mut().slider_width = 200.0;
            let r = ui.add(PillSlider::new(&mut s.values[0], 0.0..=2.0).default_value(1.0));
            s.note(0, &r);
        });
        s.beside = row;
    }

    #[test]
    fn typing_in_the_pill_sets_the_value() {
        let mut rig = Rig::new(Sliders::with(&[1.0]), whole_steps);
        let slider = rig.state.rects[0];
        let mut typed = |text: &str, end: Key| {
            rig.type_in_pill(slider, text, end);
            rig.state.values[0]
        };
        assert_eq!(typed("7", Key::Enter), 7.0, "typed over the selected value");
        assert_eq!(typed("42", Key::Enter), 10.0, "kept in range");
        assert_eq!(typed("2,6", Key::Enter), 3.0, "a comma works as the decimal point; whole steps");
        assert_eq!(typed("abc", Key::Enter), 3.0, "not a number: unchanged");
        assert_eq!(typed("9", Key::Escape), 3.0, "Escape leaves it alone");
        assert_eq!(typed("4", Key::Tab), 4.0, "leaving the field keeps the number");

        // Hidden mid-edit (Settings closed), then shown again after the value
        // changed elsewhere: the old text is dropped, and an Enter pressed
        // later (in another field, say) doesn't bring it back.
        rig.click(pill(slider));
        rig.idle();
        rig.type_text("8");
        rig.state.hidden = true;
        rig.idle();
        rig.state.values[0] = 6.0;
        rig.state.hidden = false;
        rig.idle();
        rig.key(Key::Enter);
        rig.idle();
        assert_eq!(rig.state.values[0], 6.0);
    }

    #[test]
    fn opening_the_pill_and_leaving_it_changes_nothing() {
        // A value with more decimals than the pill shows (set by an older
        // build, say) stays as it is until something is typed.
        let mut rig = Rig::new(Sliders::with(&[2.3456]), no_steps);
        rig.click(pill(rig.state.rects[0]));
        rig.idle();
        rig.click(Pos2::new(600.0, 340.0));
        rig.idle();
        assert_eq!(rig.state.values[0], 2.3456);
        assert!(!rig.state.ever_changed, "no change reported");
        rig.type_in_pill(rig.state.rects[0], "2.5", Key::Enter);
        assert_eq!(rig.state.values[0], 2.5);
    }

    #[test]
    fn the_track_sets_snaps_and_resets() {
        let mut rig = Rig::new(Sliders::with(&[1.0]), whole_steps);
        let at = |rig: &Rig<Sliders>, v: f32| track(rig.state.rects[0], 120.0, v / 10.0);
        rig.click(at(&rig, 8.0));
        rig.idle();
        assert_eq!(rig.state.values[0], 8.0, "a press sets the value where it lands");
        rig.double_click(at(&rig, 2.0));
        rig.idle();
        assert_eq!(rig.state.values[0], 5.0, "a double-click goes back to the default");

        let mut rig = Rig::new(Sliders::with(&[0.0]), no_steps);
        let drag = |rig: &mut Rig<Sliders>, to: f32| {
            let (from, to) = (at(rig, 9.0), at(rig, to));
            rig.press(from, true);
            for t in [0.5, 0.9, 1.0] {
                rig.frame(vec![Event::PointerMoved(from + (to - from) * t)]);
            }
            assert!(rig.state.changed, "a drag reports the change as it happens");
            rig.press(to, false);
            rig.idle();
            rig.state.values[0]
        };
        assert_eq!(drag(&mut rig, 5.2), 5.0, "a drag ending near the default lands on it");
        rig.shift = true;
        assert_eq!(drag(&mut rig, 5.2), 5.2, "Shift drags past it");
        rig.shift = false;
        assert_eq!(drag(&mut rig, 6.0), 6.0, "away from it, no pull");
        // Anywhere along it, a drag picks the roundest value within a pixel of
        // the pointer, as egui's slider does — here, where a pixel is about a
        // tenth, one with a single decimal.
        for v in [1.23, 3.77, 6.18, 8.031, 9.9] {
            let got = drag(&mut rig, v);
            assert!((got - v).abs() < 0.1, "{got} for {v}");
            assert_eq!(format!("{got:.1}").parse::<f32>().unwrap(), got, "{got} for {v}");
        }
    }

    #[test]
    fn the_default_never_hides_the_stops_beside_it() {
        // Every whole-point pointer position along the track, dragged across:
        // all the stops near the default turn up, the default among them.
        let mut rig = Rig::new(Sliders::default(), steps_1_to_50);
        let slider = rig.state.rects[0];
        let y = slider.center().y;
        let (first, last) = ((slider.right() - 90.0).floor() as i32, slider.right().ceil() as i32);
        rig.press(Pos2::new(first as f32, y), true);
        let mut seen = BTreeSet::new();
        for x in first..=last {
            rig.frame(vec![Event::PointerMoved(Pos2::new(x as f32, y))]);
            seen.insert(rig.state.whole);
        }
        rig.press(Pos2::new(last as f32, y), false);
        let missing: Vec<i32> = (6..=18).filter(|v| !seen.contains(v)).collect();
        assert!(missing.is_empty(), "never reached: {missing:?}");
    }

    #[test]
    fn keyboard_focus_comes_from_tab_not_the_mouse() {
        let mut rig = Rig::new(Sliders::with(&[4.0, 4.0]), side_by_side);
        let focused = |rig: &Rig<Sliders>| rig.ctx.memory(|m| m.focused());
        rig.key(Key::ArrowRight);
        rig.idle();
        assert_eq!(rig.state.values, [4.0, 4.0], "nothing focused: the arrows aren't the sliders'");

        // A click on the track sets the value but leaves the keyboard alone
        // (←/→ go on browsing images).
        rig.click(track(rig.state.rects[0], 120.0, 0.6));
        rig.idle();
        assert_eq!((rig.state.values[0], focused(&rig)), (6.0, None));
        rig.key(Key::ArrowRight);
        rig.idle();
        assert_eq!(rig.state.values, [6.0, 4.0]);

        // Tab: the first slider's pill, then its track. Now ←/→ step it, and
        // focus stays put rather than moving on to the next slider.
        rig.key(Key::Tab);
        rig.key(Key::Tab);
        rig.idle();
        assert_eq!(focused(&rig), Some(rig.state.ids[0]));
        for key in [Key::ArrowRight, Key::ArrowRight, Key::ArrowLeft, Key::ArrowRight] {
            rig.key(key);
        }
        rig.idle();
        assert_eq!(rig.state.values, [8.0, 4.0]);
        assert_eq!(focused(&rig), Some(rig.state.ids[0]));
    }

    #[test]
    fn editing_a_pill_moves_nothing_else() {
        let mut rig = Rig::new(Sliders::with(&[1.0, 1.0]), side_by_side);
        let (second, after) = (rig.state.rects[1], rig.state.beside);
        rig.click(pill(rig.state.rects[0]));
        rig.idle();
        assert_eq!((rig.state.rects[1], rig.state.beside), (second, after), "what follows stays put");
        // Typed into the first, then straight on to the second's pill: the
        // first keeps what was typed, the second opens.
        rig.type_text("7");
        rig.type_in_pill(second, "3", Key::Enter);
        assert_eq!(rig.state.values, [7.0, 3.0]);
    }

    #[test]
    fn the_track_gives_way_in_a_narrow_row() {
        let rig = Rig::new(Sliders::with(&[1.0]), narrow_row);
        let (slider, row) = (rig.state.rects[0], rig.state.beside);
        assert!(slider.right() <= row.right() + 0.5, "{slider:?} spills out of {row:?}");
    }

    #[test]
    fn a_reversed_range_runs_from_its_start_on_the_left() {
        let mut rig = Rig::new(Sliders::with(&[5.0]), reversed);
        rig.click(track(rig.state.rects[0], 120.0, 0.2));
        rig.idle();
        assert_eq!(rig.state.values[0], 8.0);
        rig.type_in_pill(rig.state.rects[0], "12", Key::Enter);
        assert_eq!(rig.state.values[0], 10.0);
    }

    #[test]
    fn screen_readers_can_set_and_step_it() {
        let mut rig = Rig::new(Sliders::with(&[4.0]), whole_steps);
        let ask = |rig: &Rig<Sliders>, action: Action, data: Option<ActionData>| {
            Event::AccessKitActionRequest(accesskit::ActionRequest {
                action,
                target_tree: accesskit::TreeId::ROOT,
                target_node: rig.state.ids[0].accesskit_id(),
                data,
            })
        };
        rig.frame(vec![ask(&rig, Action::SetValue, Some(ActionData::NumericValue(7.4)))]);
        assert_eq!(rig.state.values[0], 7.0);
        rig.frame(vec![ask(&rig, Action::Increment, None)]);
        assert_eq!(rig.state.values[0], 8.0);
        rig.frame(vec![ask(&rig, Action::Decrement, None), ask(&rig, Action::Decrement, None)]);
        assert_eq!(rig.state.values[0], 6.0);
    }

    #[test]
    fn steps_set_the_decimals() {
        assert_eq!(step_decimals(0.05), 2);
        assert_eq!(step_decimals(0.01), 2);
        assert_eq!(step_decimals(0.5), 1);
        assert_eq!(step_decimals(10.0), 0);
        assert_eq!(step_decimals(64.0), 0);
        let mut v = 1.0_f32;
        assert_eq!(PillSlider::new(&mut v, 0.0..=2.0).step_by(0.05).decimals(), 2);
        assert_eq!(PillSlider::new(&mut v, 0.0..=2.0).decimals(), 2);
        assert_eq!(PillSlider::new(&mut v, 0.0..=12.0).fixed_decimals(1).decimals(), 1);
        let mut n = 3_usize;
        assert_eq!(PillSlider::new(&mut n, 0..=3).decimals(), 0);
    }

    #[test]
    fn values_land_in_range_on_steps_and_on_the_shown_decimals() {
        assert_eq!(snap_to(0.954, 0.05, 1.0, Some(0.01), 2), 0.95);
        assert_eq!(snap_to(7.0, 0.05, 1.0, Some(0.01), 2), 1.0);
        assert_eq!(snap_to(-1.0, 0.05, 1.0, Some(0.01), 2), 0.05);
        assert_eq!(snap_to(0.3, 0.0, 2.0, Some(0.05), 2), 0.3, "no float noise (0.30000000000000004)");
        assert_eq!(snap_to(3000.0, 256.0, 8192.0, Some(64.0), 0), 3008.0);
        assert_eq!(snap_to(9000.0, 256.0, 8192.0, Some(64.0), 0), 8192.0);
        assert_eq!(snap_to(0.8346, 0.0, 2.0, None, 2), 0.83);
        assert_eq!(snap_to(4.3644, 1.0, 30.0, None, 1), 4.4);
        assert_eq!(snap_to(7.3, 10.0, 0.0, Some(1.0), 0), 7.0, "a reversed range");
        assert_eq!(snap_to(-3.0, 10.0, 0.0, Some(1.0), 0), 0.0);
    }

    #[test]
    fn typed_numbers() {
        assert_eq!(parse_typed(" 2,5 ", "", false), Some(2.5));
        assert_eq!(parse_typed("50,000", "", true), Some(50_000.0));
        assert_eq!(parse_typed("50 000", "", true), Some(50_000.0));
        assert_eq!(parse_typed("1,234.5", "", false), Some(1234.5));
        assert_eq!(parse_typed("1,234,567", "", false), Some(1_234_567.0));
        assert_eq!(parse_typed("300 px", " px", false), Some(300.0));
        assert_eq!(parse_typed("300px", " px", false), Some(300.0));
        assert_eq!(parse_typed("-0.5", "", false), Some(-0.5));
        assert_eq!(parse_typed("abc", "", false), None);
        assert_eq!(parse_typed("inf", "", false), None);
        assert_eq!(parse_typed("", "", false), None);
    }
}
