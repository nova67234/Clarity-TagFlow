//! The switch in the Liquid Glass theme: a knob of real glass on a track of
//! its own height — dark and recessed when off, sky to indigo with a soft glow
//! when on. The knob is frosted white glass at rest (the track tints it a
//! little when on) and turns into a clear lens over the track when the
//! pointer is on it. Where the glass can't be drawn (an old GPU, see
//! src/liquid_glass.rs) the knob is a white pill. Same 40 x 22 points as the
//! other themes' switch, so it fits wherever that one does.

use std::f32::consts::{FRAC_PI_2, PI};

use eframe::egui::{self, Color32, CornerRadius, Pos2, Rect, Stroke, StrokeKind, epaint::RectShape, pos2, vec2};

use crate::theme::is_light;

const SKY: Color32 = Color32::from_rgb(56, 189, 248);
const INDIGO: Color32 = Color32::from_rgb(99, 102, 241);
/// The knob at rest, and how much wider it gets under the pointer.
const KNOB: egui::Vec2 = egui::vec2(22.0, 20.0);
const KNOB_HOVER_GROW: f32 = 3.0;

/// Draw the switch in `rect` (the switch's 40 x 22): `t` 0 = off .. 1 = on,
/// `hover` 0..1 (both animated by the caller).
pub(super) fn paint(p: &egui::Painter, rect: Rect, t: f32, hover: f32) {
    track(p, track_rect(rect), t);
    knob(p, knob_rect(rect, t, hover), t, hover);
}

/// The track: the knob's height, its ends where the knob's are at either end.
fn track_rect(rect: Rect) -> Rect {
    let inset = (rect.height() - KNOB.y) / 2.0;
    Rect::from_center_size(rect.center(), vec2(rect.width() - 2.0 * inset, KNOB.y))
}

/// The knob at `t` (sliding from the left end to the right), `hover` wider.
fn knob_rect(rect: Rect, t: f32, hover: f32) -> Rect {
    let inset = (rect.height() - KNOB.y) / 2.0;
    let half = (KNOB.x + KNOB_HOVER_GROW * hover) / 2.0;
    let x = egui::lerp((rect.left() + inset + half)..=(rect.right() - inset - half), t);
    Rect::from_center_size(pos2(x, rect.center().y), vec2(2.0 * half, KNOB.y))
}

fn track(p: &egui::Painter, rect: Rect, t: f32) {
    let light = is_light();
    let r = rect.height() / 2.0;
    if t > 0.0 {
        // A tight glow: a wide one bled past the track's edge and made the
        // whole switch look out of focus.
        blur(p, rect, SKY.gamma_multiply(0.32 * t), 4.0);
    }
    // Off: dark and recessed (shaded along the top). Kept clear of the Liquid
    // Glass marker colours, or the track would turn into a glass panel.
    let off = if light {
        Color32::from_rgba_unmultiplied(30, 34, 45, 46)
    } else {
        Color32::from_rgba_unmultiplied(20, 24, 33, 170)
    };
    p.rect_filled(rect, r, off);
    let shade = if light { 40.0 } else { 110.0 } * (1.0 - t);
    half_rim(p, rect, true, Color32::from_black_alpha(shade as u8), 2.0);
    // On: sky to indigo, fading in over the off colour, with a sheen on top.
    if t > 0.0 {
        gradient(p, rect, SKY.gamma_multiply(t), INDIGO.gamma_multiply(t));
        half_rim(p, rect, true, Color32::from_white_alpha((70.0 * t) as u8), 1.0);
    }
    let edge = if light { Color32::from_black_alpha(30) } else { Color32::from_white_alpha(36) };
    let lit = Color32::from_rgba_unmultiplied(186, 230, 253, 140);
    p.rect_stroke(rect, r, Stroke::new(1.0_f32, mix(edge, lit, t)), StrokeKind::Inside);
}

fn knob(p: &egui::Painter, rect: Rect, t: f32, hover: f32) {
    let light = is_light();
    let r = rect.height() / 2.0;
    if crate::liquid_glass::active() {
        let style = crate::liquid_glass::Style {
            blur: 1.2 - 0.8 * hover,
            bezel: 5.5,
            refract: 3.3 + 2.8 * hover,
            tint: [1.0, 1.0, 1.0, knob_frost(t, hover)],
            vibrancy: 0.2,
            spec: 1.1,
            // Deep enough to lift it off a track of its own height.
            shadow: [0.0, 2.25, 6.5, if light { 0.31 } else { 0.52 }],
        };
        // Fades with what it's on (a window fading in, a disabled row).
        let cb = crate::liquid_glass::callback(rect, CornerRadius::from(r), style, p.opacity());
        p.add(egui::Shape::Callback(cb));
    } else {
        blur(p, rect.translate(vec2(0.0, 1.75)), Color32::from_black_alpha(110), 5.5);
        p.rect_filled(rect, r, Color32::WHITE);
    }
}

/// How much white the glass knob has: frosted at rest, a little clearer when
/// on (the track tints it), a clear lens under the pointer — clearest over
/// the coloured track: off, a clear lens over the dark track reads grey.
fn knob_frost(t: f32, hover: f32) -> f32 {
    (0.93 - 0.08 * t - 0.68 * hover * (0.35 + 0.65 * t)).max(0.12)
}

fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    Color32::from(egui::Rgba::from(a) * (1.0 - t) + egui::Rgba::from(b) * t)
}

/// A soft blurred capsule (the glow).
fn blur(p: &egui::Painter, rect: Rect, color: Color32, width: f32) {
    p.add(RectShape::filled(rect, CornerRadius::from(rect.height() / 2.0), color).with_blur_width(width));
}

/// The upper (`top`) or lower half of the capsule's rim only.
fn half_rim(p: &egui::Painter, rect: Rect, top: bool, color: Color32, width: f32) {
    let half = if top {
        Rect::from_min_max(rect.min - vec2(1.0, 1.0), pos2(rect.right() + 1.0, rect.center().y))
    } else {
        Rect::from_min_max(pos2(rect.left() - 1.0, rect.center().y), rect.max + vec2(1.0, 1.0))
    };
    p.with_clip_rect(half.intersect(p.clip_rect())).rect_stroke(
        rect,
        rect.height() / 2.0,
        Stroke::new(width, color),
        StrokeKind::Inside,
    );
}

/// The capsule filled left to right from `left` to `right`.
fn gradient(p: &egui::Painter, rect: Rect, left: Color32, right: Color32) {
    let r = rect.height() / 2.0;
    let c = rect.center();
    let at = |x: f32| mix(left, right, ((x - rect.left()) / rect.width()).clamp(0.0, 1.0));
    let n = 18;
    let mut edge: Vec<Pos2> = Vec::with_capacity(2 * (n + 1));
    for i in 0..=n {
        let a = -FRAC_PI_2 + PI * i as f32 / n as f32;
        edge.push(pos2(rect.right() - r + r * a.cos(), c.y + r * a.sin()));
    }
    for i in 0..=n {
        let a = FRAC_PI_2 + PI * i as f32 / n as f32;
        edge.push(pos2(rect.left() + r + r * a.cos(), c.y + r * a.sin()));
    }
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(c, at(c.x));
    for q in &edge {
        mesh.colored_vertex(*q, at(q.x));
    }
    let m = edge.len() as u32;
    for i in 0..m {
        mesh.add_triangle(0, 1 + i, 1 + (i + 1) % m);
    }
    p.add(egui::Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn switch() -> Rect {
        Rect::from_min_size(pos2(100.0, 50.0), vec2(40.0, 22.0))
    }

    #[test]
    fn the_track_is_the_knobs_height_and_ends_where_it_does() {
        let (s, track) = (switch(), track_rect(switch()));
        assert_eq!(track.height(), KNOB.y);
        assert!(s.contains_rect(track));
        assert_eq!(knob_rect(s, 0.0, 0.0).left(), track.left());
        assert_eq!(knob_rect(s, 1.0, 0.0).right(), track.right());
        for t in [0.0, 1.0] {
            assert_eq!(knob_rect(s, t, 0.0).height(), track.height());
        }
    }

    #[test]
    fn the_knob_stays_inside_the_switch() {
        let s = switch();
        for t in [0.0, 0.3, 0.7, 1.0] {
            for hover in [0.0, 0.5, 1.0] {
                let k = knob_rect(s, t, hover);
                assert!(s.contains_rect(k), "t {t} hover {hover}: {k:?}");
            }
        }
    }

    #[test]
    fn the_knob_is_frosted_at_rest_and_clearest_hovered_on() {
        assert!(knob_frost(0.0, 0.0) > 0.9);
        assert!(knob_frost(1.0, 0.0) < knob_frost(0.0, 0.0));
        assert!(knob_frost(0.0, 1.0) > knob_frost(1.0, 1.0));
        assert!(knob_frost(1.0, 1.0) >= 0.12);
    }
}
