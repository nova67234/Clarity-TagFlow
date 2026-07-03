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
//! - **Aurora** — the light-mode counterpart to Space: light surfaces with
//!   transparent gutters revealing a soft, slowly-drifting pastel aurora glow.
//! - **Glass** — a dark theme with *translucent* panels (true frosted glass), so
//!   the background bleeds through the cards themselves. Unlike Space/Aurora, the
//!   background is user-configurable: a colour (picker) plus an optional animated
//!   [`Backdrop`], both pushed in via [`set_glass_config`] and painted by
//!   [`paint_background`]. The panel colours are fixed, so changing the background
//!   recolours the gutters/backdrop without changing the glass tint itself.

use eframe::egui::{self, Color32, CornerRadius};
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

/// The available app themes. Persisted in `Settings` (defaults to `Dark`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum Theme {
    #[default]
    Dark,
    Light,
    Space,
    Aurora,
    Glass,
}

/// The background style painted behind the [`Theme::Glass`] panels. Persisted in
/// `Settings`. The chosen colour is painted as a flat base; `Starfield`/`Aurora`
/// additionally animate the same effects the Space/Aurora themes use, over it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum Backdrop {
    /// Just the flat background colour.
    #[default]
    Solid,
    /// Twinkling starfield over the background colour.
    Starfield,
    /// Drifting pastel aurora blobs over the background colour.
    Aurora,
}

/// Build a translucent [`Color32`] from straight (un-premultiplied) RGBA. egui
/// stores premultiplied colours and its `from_rgba_unmultiplied` isn't `const`,
/// so we premultiply here to keep the palette statics `const`.
const fn glass(r: u8, g: u8, b: u8, a: u8) -> Color32 {
    Color32::from_rgba_premultiplied(
        (r as u16 * a as u16 / 255) as u8,
        (g as u16 * a as u16 / 255) as u8,
        (b as u16 * a as u16 / 255) as u8,
        a,
    )
}

/// A full set of named UI colours. `is_dark` selects the egui base visuals;
/// `starfield` enables the animated space background; `aurora` enables the
/// animated light aurora background.
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
    aurora: bool,
    /// Translucent panels with a user-configurable background (the Glass theme).
    /// When set, [`paint_background`] paints the configured colour + backdrop.
    glass: bool,
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
    aurora: false,
    glass: false,
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
    aurora: false,
    glass: false,
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
    aurora: false,
    glass: false,
};

/// An aurora palette: the light-mode counterpart to Space. Identical to the Light
/// theme's cards (opaque white, so text stays readable), but with transparent
/// gutters so the animated aurora glow shows through behind everything. `bg` is
/// fully transparent on purpose — see [`paint_background`].
static AURORA: Palette = Palette {
    bg: Color32::TRANSPARENT,
    panel: Color32::from_rgb(255, 255, 255),
    field: Color32::from_rgb(238, 240, 243),
    field2: Color32::from_rgb(226, 229, 234),
    text: Color32::from_rgb(28, 29, 32),
    muted: Color32::from_rgb(110, 115, 122),
    accent1: Color32::from_rgb(28, 110, 235),
    accent2: Color32::from_rgb(20, 140, 200),
    edge: Color32::from_rgba_premultiplied(0, 0, 0, 26),
    is_dark: false,
    starfield: false,
    aurora: true,
    glass: false,
};

/// A glass palette: dark, but the panels are *translucent* (~70% opaque) so the
/// background bleeds through the cards themselves. The background colour/backdrop
/// is user-set (see [`set_glass_config`]); `bg` is transparent so the gutters
/// reveal it fully, and the panel/field fills are semi-transparent so it shows
/// through them too. Text/accents match Dark for readability.
static GLASS: Palette = Palette {
    bg: Color32::TRANSPARENT,
    panel: glass(42, 44, 50, 180),
    field: glass(60, 62, 68, 195),
    field2: glass(72, 74, 80, 195),
    text: Color32::from_rgb(235, 235, 235),
    muted: Color32::from_rgb(178, 181, 187),
    accent1: Color32::from_rgb(64, 140, 255),
    accent2: Color32::from_rgb(90, 200, 245),
    // No panel outline — the translucent fill carries the glass look on its own.
    edge: Color32::TRANSPARENT,
    is_dark: true,
    starfield: false,
    aurora: false,
    glass: true,
};

/// The light-mode Glass palette: the same frosted-glass treatment, but the panels
/// are translucent *white* and the ink is dark grey — text, muted labels, and
/// (via [`icon_tint`]) the SVG icons. Accents match the Light theme so buttons
/// stay readable white-on-blue. Selected from the Appearance tab's "Glass panels"
/// switch; the dark variant above is untouched.
static GLASS_LIGHT: Palette = Palette {
    bg: Color32::TRANSPARENT,
    panel: glass(246, 247, 250, 185),
    field: glass(233, 235, 240, 205),
    field2: glass(222, 225, 231, 205),
    text: Color32::from_rgb(55, 58, 64),
    muted: Color32::from_rgb(108, 113, 121),
    accent1: Color32::from_rgb(28, 110, 235),
    accent2: Color32::from_rgb(20, 140, 200),
    // No panel outline — the translucent fill carries the glass look on its own.
    edge: Color32::TRANSPARENT,
    is_dark: false,
    starfield: false,
    aurora: false,
    glass: true,
};

/// 0 = Dark, 1 = Light, 2 = Space, 3 = Aurora, 4 = Glass.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

/// Whether the Glass theme uses its light palette ([`GLASS_LIGHT`]) instead of
/// the dark one. Pushed each frame with [`set_glass_config`].
static GLASS_LIGHT_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The Glass theme's background, packed for atomic access: the low 24 bits are
/// the RGB colour; the next 8 bits are the [`Backdrop`] discriminant. Updated by
/// [`set_glass_config`] each frame from the persisted settings.
static GLASS_CONFIG: AtomicU32 = AtomicU32::new(0);

/// Switch the active palette. Call [`apply`] afterwards to push the new visuals.
pub fn set(theme: Theme) {
    let v = match theme {
        Theme::Dark => 0,
        Theme::Light => 1,
        Theme::Space => 2,
        Theme::Aurora => 3,
        Theme::Glass => 4,
    };
    ACTIVE.store(v, Ordering::Relaxed);
}

/// Set the Glass theme's background — a flat `rgb` colour plus an animated
/// `backdrop` painted over it — and whether the panels use the light palette.
/// Cheap (atomic stores) so callers push it every frame from the persisted
/// settings, letting the colour picker update live.
pub fn set_glass_config(rgb: [u8; 3], backdrop: Backdrop, light: bool) {
    let b = match backdrop {
        Backdrop::Solid => 0u32,
        Backdrop::Starfield => 1,
        Backdrop::Aurora => 2,
    };
    let packed = (b << 24) | ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | rgb[2] as u32;
    GLASS_CONFIG.store(packed, Ordering::Relaxed);
    GLASS_LIGHT_ON.store(light, Ordering::Relaxed);
}

/// Unpack the current Glass background into its colour and backdrop.
fn glass_config() -> (Color32, Backdrop) {
    let p = GLASS_CONFIG.load(Ordering::Relaxed);
    let color = Color32::from_rgb((p >> 16) as u8, (p >> 8) as u8, p as u8);
    let backdrop = match (p >> 24) & 0xff {
        1 => Backdrop::Starfield,
        2 => Backdrop::Aurora,
        _ => Backdrop::Solid,
    };
    (color, backdrop)
}

/// The currently active theme. (Provided for completeness — the app drives theme
/// changes from `Settings`, so this isn't called internally yet.)
#[allow(dead_code)]
pub fn current() -> Theme {
    match ACTIVE.load(Ordering::Relaxed) {
        1 => Theme::Light,
        2 => Theme::Space,
        3 => Theme::Aurora,
        4 => Theme::Glass,
        _ => Theme::Dark,
    }
}

/// True for the light-surface themes (Light and Aurora). Lets call sites pick
/// light-vs-dark styling (e.g. a console background) without enumerating themes.
pub fn is_light() -> bool {
    !palette().is_dark
}

/// Tint for icon buttons (e.g. the folder icon): a soft pink under Aurora so the
/// icons match its warm glow, dark grey under light Glass so the SVGs read on the
/// white frosted panels, otherwise the caller's normal colour `fallback`.
pub fn icon_tint(fallback: Color32) -> Color32 {
    let p = palette();
    if p.aurora {
        Color32::from_rgb(235, 130, 175) // matches the Aurora pink buttons
    } else if p.glass && !p.is_dark {
        Color32::from_rgb(75, 78, 85) // dark-grey icons on light glass
    } else {
        fallback
    }
}

/// Colour for the selected-tile outline in the browser: the theme's accent blue
/// everywhere (light modes use their deeper accent so it reads on white panels),
/// except Aurora's pink so it matches that theme's warm glow.
pub fn selection_outline() -> Color32 {
    let p = palette();
    if p.aurora {
        Color32::from_rgb(235, 130, 175)
    } else {
        p.accent1
    }
}

fn palette() -> &'static Palette {
    match ACTIVE.load(Ordering::Relaxed) {
        1 => &LIGHT,
        2 => &SPACE,
        3 => &AURORA,
        4 => {
            if GLASS_LIGHT_ON.load(Ordering::Relaxed) {
                &GLASS_LIGHT
            } else {
                &GLASS
            }
        }
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
    // Selection highlight follows the theme accent, but Aurora uses its pink so
    // selected items (tabs, menu entries, text selection) match the pink buttons.
    v.selection.bg_fill = if p.aurora {
        Color32::from_rgb(235, 130, 175).gamma_multiply(0.55)
    } else {
        p.accent1.gamma_multiply(0.45)
    };
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(8);
    }

    // Light glass gets its own QUIET widget styling instead of the Light theme's
    // accent-blue: panel-toned grey fills with a grey outline and dark-ink text.
    // - toggles (radio/checkbox) render grey, not blue
    // - widget/menu text is dark ink, not white (white-on-light was unreadable)
    // - the slider rail/knob match the panel tone, ringed grey ("outer layer")
    if p.glass && !p.is_dark {
        let ink = p.text;
        let outline = egui::Stroke::new(1.0, Color32::from_rgba_unmultiplied(0, 0, 0, 40));
        let tune = |w: &mut egui::style::WidgetVisuals, fill: Color32, weak: Color32| {
            w.bg_fill = fill;
            w.weak_bg_fill = weak;
            w.fg_stroke.color = ink;
            w.bg_stroke = outline;
        };
        tune(&mut v.widgets.inactive, Color32::from_rgb(226, 229, 234), Color32::from_rgb(214, 218, 224));
        tune(&mut v.widgets.hovered, Color32::from_rgb(214, 218, 224), Color32::from_rgb(202, 206, 213));
        tune(&mut v.widgets.active, Color32::from_rgb(202, 206, 213), Color32::from_rgb(190, 194, 201));
        tune(&mut v.widgets.open, Color32::from_rgb(214, 218, 224), Color32::from_rgb(202, 206, 213));

        // Same override handling as the Light branch below: ordinary labels pin
        // to the dark ink, while widget text follows the fg_stroke set above.
        v.override_text_color = None;
        v.widgets.noninteractive.fg_stroke.color = ink;

        ctx.set_visuals(v);
        return;
    }

    // Light mode: paint the otherwise-grey buttons in the accent blue with white
    // labels. Buttons use `weak_bg_fill` (and their text follows
    // `override_text_color`), whereas checkboxes/sliders use `bg_fill` — so this
    // recolours buttons without turning every widget solid blue. Dark/Space keep
    // egui's default grey buttons untouched.
    if !p.is_dark {
        // Aurora gets soft-pink buttons to match its warm pastel glow; plain Light
        // keeps the accent blue.
        let (idle, hover, down) = if p.aurora {
            (
                Color32::from_rgb(235, 130, 175), // light pink
                Color32::from_rgb(245, 150, 190), // brighter on hover
                Color32::from_rgb(210, 100, 150), // darker when pressed
            )
        } else {
            (
                p.accent1,
                Color32::from_rgb(56, 132, 245), // a touch brighter
                Color32::from_rgb(20, 92, 200),  // pressed, darker
            )
        };
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

/// Paint the active theme's full-window background, drawn on the bottom-most
/// layer so every transparent-gutter region (and the image area) shows it
/// through. [`Theme::Space`] gets a twinkling starfield; [`Theme::Aurora`] gets a
/// drifting pastel glow; [`Theme::Glass`] gets the user's chosen colour plus an
/// optional backdrop. Other themes are a no-op. Call once per frame near the top
/// of `update`.
pub fn paint_background(ctx: &egui::Context) {
    let p = palette();
    if p.glass {
        paint_glass(ctx);
        return;
    }
    if p.aurora {
        let rect = ctx.content_rect();
        let painter = ctx.layer_painter(egui::LayerId::background());
        // A clearly-tinted soft-blue base; the visible gutters carry the look.
        painter.rect_filled(rect, 0.0, Color32::from_rgb(212, 222, 242));
        draw_aurora_blobs(ctx, &painter, rect);
        return;
    }
    if !p.starfield {
        return;
    }
    let rect = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());
    // Deep-space base — covers the framebuffer clear colour so transparent panels
    // reveal this rather than whatever eframe cleared to.
    painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 9, 16));
    draw_starfield(ctx, &painter, rect);
}

/// Paint the Glass theme's background: the user's flat colour, plus the chosen
/// animated [`Backdrop`] over it. The translucent panels then let this show
/// through. See [`set_glass_config`].
fn paint_glass(ctx: &egui::Context) {
    let (color, backdrop) = glass_config();
    let rect = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());
    painter.rect_filled(rect, 0.0, color);
    match backdrop {
        Backdrop::Solid => {}
        Backdrop::Starfield => draw_starfield(ctx, &painter, rect),
        Backdrop::Aurora => draw_aurora_blobs(ctx, &painter, rect),
    }
}

/// Draw the twinkling starfield over whatever base `painter` already has. Shared
/// by the Space theme and the Glass theme's Starfield backdrop.
fn draw_starfield(ctx: &egui::Context, painter: &egui::Painter, rect: egui::Rect) {
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

    draw_shooting_star(t, painter, rect);

    // Keep the twinkle animating (~30 fps) without spinning the CPU flat-out.
    ctx.request_repaint_after(std::time::Duration::from_millis(33));
}

/// Occasionally streak a shooting star across the field. Stateless: time is split
/// into fixed-length epochs, and ~most epochs spawn one meteor whose start,
/// direction and timing are hashed from the epoch index — so it animates smoothly
/// and stays deterministic without any per-frame state. Roughly one every few
/// seconds.
fn draw_shooting_star(t: f32, painter: &egui::Painter, rect: egui::Rect) {
    const PERIOD: f32 = 7.0; // seconds per epoch (a meteor "slot")
    const DUR: f32 = 0.85; // how long a streak is visible
    let epoch = (t / PERIOD).floor();
    let seed = epoch.max(0.0) as u32;

    // Skip some epochs so the timing feels irregular rather than metronomic.
    if hash01(seed, 20) > 0.7 {
        return;
    }
    // Start the streak at a hashed offset within the epoch (not at its boundary).
    let local = t - epoch * PERIOD - hash01(seed, 21) * (PERIOD - DUR);
    if !(0.0..=DUR).contains(&local) {
        return;
    }
    let p = local / DUR; // 0..1 progress along the streak
    let env = (p * std::f32::consts::PI).sin(); // brightness fades in then out

    // Direction: mostly downward, biased left or right; normalised.
    let dir_x = if hash01(seed, 22) < 0.5 { -1.0 } else { 1.0 } * (0.45 + hash01(seed, 23) * 0.6);
    let dir_y = 0.5 + hash01(seed, 24) * 0.5;
    let dlen = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1e-3);
    let (ux, uy) = (dir_x / dlen, dir_y / dlen);

    // Start on the side opposite the travel direction (and in the upper area,
    // since it moves downward) so the streak crosses the visible field instead of
    // immediately flying off an edge.
    let span = rect.width().max(rect.height());
    let frac = hash01(seed, 25);
    let sx = if ux >= 0.0 {
        rect.left() + frac * rect.width() * 0.45 // moving right → start left
    } else {
        rect.left() + rect.width() * (0.55 + frac * 0.45) // moving left → start right
    };
    let sy = rect.top() + hash01(seed, 26) * rect.height() * 0.4;
    let hx = sx + ux * span * 0.6 * p;
    let hy = sy + uy * span * 0.6 * p;

    // Tail: short segments trailing the head, fading toward the tail end.
    let tail = (span * 0.10).clamp(60.0, 200.0);
    const SEG: usize = 10;
    for k in 0..SEG {
        let f0 = k as f32 / SEG as f32;
        let f1 = (k + 1) as f32 / SEG as f32;
        let a = egui::pos2(hx - ux * tail * f0, hy - uy * tail * f0);
        let b = egui::pos2(hx - ux * tail * f1, hy - uy * tail * f1);
        let alpha = (env * (1.0 - f0) * 230.0).clamp(0.0, 255.0) as u8;
        let w = 2.0 * (1.0 - f0) + 0.4;
        painter.line_segment([a, b], egui::Stroke::new(w, Color32::from_rgba_unmultiplied(255, 255, 255, alpha)));
    }
    // Bright head.
    let ha = (env * 255.0).clamp(0.0, 255.0) as u8;
    painter.circle_filled(egui::pos2(hx, hy), 1.8, Color32::from_rgba_unmultiplied(255, 255, 255, ha));
}

/// Draw the drifting pastel aurora blobs over whatever base `painter` already
/// has. Shared by the Aurora theme and the Glass theme's Aurora backdrop.
fn draw_aurora_blobs(ctx: &egui::Context, painter: &egui::Painter, rect: egui::Rect) {
    let t = ctx.input(|i| i.time) as f32;

    // Saturated pastel blobs, each drifting on its own slow ellipse. Anchored
    // toward the window edges/corners so their strong cores land in the visible
    // gutter strips rather than hiding behind the centre panel.
    const BLOBS: &[(u8, u8, u8)] = &[
        (150, 140, 255), // lavender
        (95, 190, 255),  // sky blue
        (110, 230, 190), // mint
        (255, 185, 120), // peach
        (255, 140, 190), // rose
        (130, 175, 255), // pale blue
    ];

    let span = rect.width().min(rect.height()).max(1.0);
    let radius = span * 0.5;

    for (i, &(r, g, b)) in BLOBS.iter().enumerate() {
        let i = i as u32;
        // Bias anchors toward the edges: map [0,1) to the outer thirds so cores
        // sit near the window border where the gutters are.
        let hx = hash01(i, 11);
        let hy = hash01(i, 12);
        let edge = |h: f32| if h < 0.5 { 0.05 + h * 0.4 } else { 0.55 + (h - 0.5) * 0.4 };
        let bx = rect.left() + edge(hx) * rect.width();
        let by = rect.top() + edge(hy) * rect.height();
        let speed = 0.04 + 0.05 * hash01(i, 13); // very slow
        let phase = hash01(i, 14) * std::f32::consts::TAU;
        let drift = span * 0.12;
        let cx = bx + (t * speed + phase).cos() * drift;
        let cy = by + (t * speed * 0.8 + phase).sin() * drift;

        soft_blob(painter, egui::pos2(cx, cy), radius, Color32::from_rgb(r, g, b));
    }

    // Animate gently — aurora drifts much slower than stars twinkle.
    ctx.request_repaint_after(std::time::Duration::from_millis(50));
}

/// Approximate a Gaussian-blurred blob by stacking concentric translucent
/// circles (egui has no real blur). Many rings at a modest per-ring alpha build a
/// soft falloff that's visible but still gentle.
fn soft_blob(painter: &egui::Painter, center: egui::Pos2, radius: f32, color: Color32) {
    const RINGS: usize = 24;
    for k in 0..RINGS {
        // Outer rings are larger and fainter; inner rings smaller and stronger,
        // building up a soft falloff toward the centre.
        let f = 1.0 - k as f32 / RINGS as f32; // 1.0 (outer) .. ~0.04 (inner)
        let r = radius * (0.12 + 0.88 * f);
        let alpha = 22.0_f32; // per-ring; accumulates to a clearly-visible centre
        let col = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha as u8);
        painter.circle_filled(center, r, col);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glass_config_round_trips() {
        for backdrop in [Backdrop::Solid, Backdrop::Starfield, Backdrop::Aurora] {
            for light in [false, true] {
                set_glass_config([10, 200, 255], backdrop, light);
                let (c, b) = glass_config();
                assert_eq!((c.r(), c.g(), c.b()), (10, 200, 255));
                assert_eq!(b, backdrop);
                assert_eq!(GLASS_LIGHT_ON.load(Ordering::Relaxed), light);
            }
        }
    }

    #[test]
    fn glass_panels_are_translucent() {
        // The whole point of the Glass theme: panels let the background through.
        assert!(GLASS.panel.a() < 255, "glass panel must be translucent");
        assert!(GLASS.field.a() < 255, "glass field must be translucent");
        assert_eq!(GLASS.bg, Color32::TRANSPARENT, "gutters reveal the backdrop");
    }
}
