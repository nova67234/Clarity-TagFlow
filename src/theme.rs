//! App colour themes (palettes) and the egui visuals derived from them.
//!
//! The palette used to live as inline `const Color32`s in `main.rs`. To support a
//! user-switchable theme at runtime, the colours are now resolved through accessor
//! functions backed by a global atomic flag. The accessors keep the old
//! SCREAMING_CASE names, so call sites only gained a `()` (e.g. `TEXT` became
//! `TEXT()`). `set` flips the active palette; `apply` pushes the matching
//! `egui::Visuals` into the context.
//!
//! Themes:
//! - **Dark** — reproduces the original `apply_theme` exactly.
//! - **Light** — light surfaces; buttons recoloured to the accent blue.
//! - **Space** — a dark theme whose gutters are transparent so an animated
//!   starfield (painted by [`paint_background`] on the bottom layer) shows through
//!   behind every panel and the image. Cards stay opaque so text stays readable.

use eframe::egui::{self, Color32, CornerRadius};
use std::sync::atomic::{AtomicU8, Ordering};

/// The available app themes. Persisted in `Settings` (defaults to `Dark`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum Theme {
    #[default]
    Dark,
    Light,
    Space,
}

/// A full set of named UI colours. `is_dark` selects the egui base visuals;
/// `starfield` enables the animated space background.
/// `field2`/`accent2` round out the palette for future use (not all are wired up
/// to a widget yet), so allow them to be unread.
#[allow(dead_code)]
struct Palette {
    bg: Color32,
    panel: Color32,
    field: Color32,
    field2: Color32,
    text: Color32,
    muted: Color32,
    accent1: Color32,
    accent2: Color32,
    edge: Color32,
    is_dark: bool,
    starfield: bool,
}

/// The original dark palette — identical values to the previous inline consts.
static DARK: Palette = Palette {
    bg: Color32::from_rgb(24, 24, 26),
    panel: Color32::from_rgb(32, 32, 34),
    field: Color32::from_rgb(45, 47, 50),
    field2: Color32::from_rgb(58, 60, 64),
    text: Color32::from_rgb(235, 235, 235),
    muted: Color32::from_rgb(170, 170, 170),
    accent1: Color32::from_rgb(64, 140, 255),
    accent2: Color32::from_rgb(90, 200, 245),
    // Faint dark edge around rounded panels — EXACTLY the original value.
    edge: Color32::from_rgba_premultiplied(18, 18, 18, 20),
    is_dark: true,
    starfield: false,
};

/// A soft light palette: off-white surfaces, dark ink text, deeper accents.
static LIGHT: Palette = Palette {
    bg: Color32::from_rgb(244, 245, 247),
    panel: Color32::from_rgb(255, 255, 255),
    field: Color32::from_rgb(238, 240, 243),
    field2: Color32::from_rgb(226, 229, 234),
    text: Color32::from_rgb(28, 29, 32),
    muted: Color32::from_rgb(110, 115, 122),
    accent1: Color32::from_rgb(28, 110, 235),
    accent2: Color32::from_rgb(20, 140, 200),
    // Faint dark edge around rounded panels (subtle on light surfaces).
    edge: Color32::from_rgba_premultiplied(0, 0, 0, 26),
    is_dark: false,
    starfield: false,
};

/// A space palette: identical to the Dark theme's panels (same colours and faint
/// dark edge, so cards look exactly like dark mode), but with transparent gutters
/// so the animated starfield shows through behind everything. `bg` is fully
/// transparent on purpose — see [`paint_background`].
static SPACE: Palette = Palette {
    bg: Color32::TRANSPARENT,
    panel: Color32::from_rgb(32, 32, 34),
    field: Color32::from_rgb(45, 47, 50),
    field2: Color32::from_rgb(58, 60, 64),
    text: Color32::from_rgb(235, 235, 235),
    muted: Color32::from_rgb(170, 170, 170),
    accent1: Color32::from_rgb(64, 140, 255),
    accent2: Color32::from_rgb(90, 200, 245),
    // Same faint dark edge as Dark — no glow outline.
    edge: Color32::from_rgba_premultiplied(18, 18, 18, 20),
    is_dark: true,
    starfield: true,
};

/// 0 = Dark, 1 = Light, 2 = Space.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

/// Switch the active palette. Call [`apply`] afterwards to push the new visuals.
pub fn set(theme: Theme) {
    let v = match theme {
        Theme::Dark => 0,
        Theme::Light => 1,
        Theme::Space => 2,
    };
    ACTIVE.store(v, Ordering::Relaxed);
}

/// The currently active theme. (Provided for completeness — the app drives theme
/// changes from `Settings`, so this isn't called internally yet.)
#[allow(dead_code)]
pub fn current() -> Theme {
    match ACTIVE.load(Ordering::Relaxed) {
        1 => Theme::Light,
        2 => Theme::Space,
        _ => Theme::Dark,
    }
}

fn palette() -> &'static Palette {
    match ACTIVE.load(Ordering::Relaxed) {
        1 => &LIGHT,
        2 => &SPACE,
        _ => &DARK,
    }
}

#[allow(non_snake_case)]
pub fn BG() -> Color32 {
    palette().bg
}

#[allow(non_snake_case)]
pub fn PANEL() -> Color32 {
    palette().panel
}

#[allow(non_snake_case)]
pub fn FIELD() -> Color32 {
    palette().field
}

#[allow(non_snake_case, dead_code)]
pub fn FIELD2() -> Color32 {
    palette().field2
}

#[allow(non_snake_case)]
pub fn TEXT() -> Color32 {
    palette().text
}

#[allow(non_snake_case)]
pub fn MUTED() -> Color32 {
    palette().muted
}

#[allow(non_snake_case)]
pub fn ACCENT1() -> Color32 {
    palette().accent1
}

#[allow(non_snake_case, dead_code)]
pub fn ACCENT2() -> Color32 {
    palette().accent2
}

#[allow(non_snake_case)]
pub fn EDGE() -> Color32 {
    palette().edge
}

/// Push the active palette into egui's global visuals. The dark/space branch
/// mirrors the original `apply_theme`; the light branch additionally recolours
/// buttons to the accent blue.
pub fn apply(ctx: &egui::Context) {
    let p = palette();
    let mut v = if p.is_dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    v.panel_fill = p.bg;
    v.window_fill = p.panel;
    v.extreme_bg_color = p.field; // text-edit background
    v.override_text_color = Some(p.text);
    v.selection.bg_fill = p.accent1.gamma_multiply(0.45);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(8);
    }

    // Light mode: paint the otherwise-grey buttons in the accent blue with white
    // labels. Buttons use `weak_bg_fill` (and their text follows
    // `override_text_color`), whereas checkboxes/sliders use `bg_fill` — so this
    // recolours buttons without turning every widget solid blue. Dark/Space keep
    // egui's default grey buttons untouched.
    if !p.is_dark {
        let idle = p.accent1;
        let hover = Color32::from_rgb(56, 132, 245); // a touch brighter
        let down = Color32::from_rgb(20, 92, 200); // pressed, darker
        let white = Color32::WHITE;

        v.widgets.inactive.weak_bg_fill = idle;
        v.widgets.inactive.bg_fill = idle;
        v.widgets.inactive.fg_stroke.color = white;
        v.widgets.inactive.bg_stroke = egui::Stroke::NONE;

        v.widgets.hovered.weak_bg_fill = hover;
        v.widgets.hovered.bg_fill = hover;
        v.widgets.hovered.fg_stroke.color = white;
        v.widgets.hovered.bg_stroke = egui::Stroke::NONE;

        v.widgets.active.weak_bg_fill = down;
        v.widgets.active.bg_fill = down;
        v.widgets.active.fg_stroke.color = white;
        v.widgets.active.bg_stroke = egui::Stroke::NONE;

        // `override_text_color` also governs Button label colour, so a global dark
        // override would fight the white-on-blue. Drop it for light mode and pin
        // the default (non-interactive) text colour to the dark ink instead, so
        // ordinary labels stay readable while button text can be white.
        v.override_text_color = None;
        v.widgets.noninteractive.fg_stroke.color = p.text;
    }

    ctx.set_visuals(v);
}

/// Cheap integer hash → `[0, 1)`. Used to give each star a stable random
/// position / phase / size from its index, so the field doesn't jump each frame.
fn hash01(i: u32, salt: u32) -> f32 {
    let mut h = i
        .wrapping_mul(0x2c1b_3c6d)
        .wrapping_add(salt.wrapping_mul(0x2745_7d83))
        .wrapping_add(0x9e37_79b9);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

/// Paint the active theme's full-window background. For [`Theme::Space`] this is
/// a deep-space base plus a twinkling starfield, drawn on the bottom-most layer
/// so every transparent-gutter region (and the image area) shows it through. For
/// other themes it's a no-op. Call once per frame near the top of `update`.
pub fn paint_background(ctx: &egui::Context) {
    if !palette().starfield {
        return;
    }

    let rect = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());

    // Deep-space base — covers the framebuffer clear colour so transparent panels
    // reveal this rather than whatever eframe cleared to.
    painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 9, 16));

    let t = ctx.input(|i| i.time) as f32;
    // Scale star count to the window area (≈1 star per 5,500 px²), clamped.
    let area = (rect.width() * rect.height()).max(1.0);
    let count = ((area / 5_500.0) as u32).clamp(120, 600);

    for i in 0..count {
        let x = rect.left() + hash01(i, 1) * rect.width();
        let y = rect.top() + hash01(i, 2) * rect.height();
        let radius = 0.5 + hash01(i, 3) * 1.5;
        let speed = 0.6 + hash01(i, 4) * 2.6; // twinkle rate
        let phase = hash01(i, 5) * std::f32::consts::TAU;

        // Twinkle: brightness oscillates 0..1.
        let tw = 0.5 + 0.5 * (t * speed + phase).sin();
        let alpha = (35.0 + tw * 205.0).clamp(0.0, 255.0) as u8;

        // Mostly white, a sprinkle of blue and warm stars.
        let tint = hash01(i, 6);
        let col = if tint > 0.90 {
            Color32::from_rgba_unmultiplied(150, 195, 255, alpha) // blue
        } else if tint > 0.82 {
            Color32::from_rgba_unmultiplied(255, 225, 190, alpha) // warm
        } else {
            Color32::from_rgba_unmultiplied(255, 255, 255, alpha) // white
        };
        painter.circle_filled(egui::pos2(x, y), radius, col);
    }

    // Keep the twinkle animating (~30 fps) without spinning the CPU flat-out.
    ctx.request_repaint_after(std::time::Duration::from_millis(33));
}
