//! Liquid Glass (Settings → Appearance → Colour theme): panels, windows,
//! popups, menus and tooltips as real glass — what's behind them shows through
//! frosted, and their curved rims bend it, catch the light and cast a soft
//! shadow.
//!
//! **How.** Under the theme the panel and floating-surface colours
//! ([`crate::theme::PANEL`], [`crate::theme::FLOAT`]) are markers. At the end
//! of each frame [`glassify`] finds every rectangle painted with one and puts a
//! paint callback in its place (its frame's own shadow goes: the glass casts
//! its own). The callback runs on the GPU when egui reaches that point of the
//! frame — everything behind the surface is drawn by then — and:
//!
//! 1. copies the screen around the surface into a texture and builds its
//!    mipmaps (the frost is a blurred mip level, smoothed over a few taps);
//! 2. draws the surface from it: inside a rounded rectangle, the backdrop
//!    frosted and tinted, its colours lifted a little; on the rim, which is
//!    rounded like a lens, the backdrop shifted inward, most at the edge (so
//!    what passes under an edge is magnified as it nears it), a little less
//!    frosted and with a hint of colour fringing; a thin bright edge and a
//!    sheen across the rim, brightest facing the light (top left), fainter on
//!    the far side; outside, a soft drop shadow.
//!
//! The content drawn after the surface (its text, its buttons) lands on top
//! as usual, and a surface painted later (a menu over a panel) sees the
//! earlier glass behind it. If the GPU can't run the shader (OpenGL older
//! than 3.1, say) or copy the window (an anti-aliased one), [`glassify`]
//! stops swapping and the markers show as plain translucent panels, like the
//! Glass theme's — the chat's composer then hides what's scrolled under it,
//! as there.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui::{self, Color32, CornerRadius, Rect, Shape};
use glow::HasContext as _;

/// What a glass surface is: a panel or card in the window, or something
/// floating over it (a window, popup, menu, tooltip).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Panel,
    Float,
}

/// How one kind of glass looks. Lengths are in points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// How far the backdrop is blurred.
    pub blur: f32,
    /// The curved rim's width, and how far at most it shifts what's behind.
    pub bezel: f32,
    pub refract: f32,
    /// Laid over the backdrop: straight RGB (0..1) and how strongly.
    pub tint: [f32; 4],
    /// How much the backdrop's colours are lifted (0 = as they are).
    pub vibrancy: f32,
    /// The rim light's strength.
    pub spec: f32,
    /// The drop shadow: offset right and down, blur, opacity.
    pub shadow: [f32; 4],
}

impl Style {
    /// The glass of `kind`, light or dark.
    pub fn of(kind: Kind, light: bool) -> Style {
        match (kind, light) {
            (Kind::Panel, false) => Style {
                blur: 14.0,
                bezel: 24.0,
                refract: 13.0,
                tint: [0.06, 0.065, 0.08, 0.48],
                vibrancy: 0.35,
                spec: 0.55,
                shadow: [0.0, 8.0, 26.0, 0.30],
            },
            (Kind::Float, false) => Style {
                blur: 18.0,
                bezel: 22.0,
                refract: 12.0,
                tint: [0.07, 0.075, 0.09, 0.56],
                vibrancy: 0.35,
                spec: 0.6,
                shadow: [0.0, 12.0, 34.0, 0.42],
            },
            (Kind::Panel, true) => Style {
                blur: 14.0,
                bezel: 24.0,
                refract: 13.0,
                tint: [0.97, 0.975, 0.99, 0.52],
                vibrancy: 0.25,
                spec: 0.8,
                shadow: [0.0, 8.0, 26.0, 0.14],
            },
            (Kind::Float, true) => Style {
                blur: 18.0,
                bezel: 22.0,
                refract: 12.0,
                tint: [0.97, 0.975, 0.99, 0.6],
                vibrancy: 0.25,
                spec: 0.85,
                shadow: [0.0, 12.0, 34.0, 0.2],
            },
        }
    }

    /// How far past the surface its callback reaches: its shadow.
    fn margin(&self) -> f32 {
        let [dx, dy, blur, _] = self.shadow;
        dx.abs().max(dy.abs()) + blur + 2.0
    }
}

/// The colours that mark glass this frame (see [`crate::theme::liquid_markers`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Markers {
    pub panel: Color32,
    pub float: Color32,
    pub light: bool,
}

impl Markers {
    /// Which glass `fill` marks, and how opaque it is: a surface fading in
    /// or out (a window opening, say) is painted in its marker scaled down,
    /// and its glass fades with it.
    fn kind(&self, fill: Color32) -> Option<(Kind, f32)> {
        // Very faint, both may fit: the nearer one.
        match (faded(fill, self.panel), faded(fill, self.float)) {
            (Some((_, near)), Some((k, far))) if far < near => Some((Kind::Float, k)),
            (Some((k, _)), _) => Some((Kind::Panel, k)),
            (None, Some((k, _))) => Some((Kind::Float, k)),
            (None, None) => None,
        }
    }
}

/// How opaque `fill` is as `marker` faded (`marker`, premultiplied, scaled
/// down — as egui paints a fading surface), and how far off it is (in
/// colour levels): `None` if it isn't that. Fainter than about 30%, it
/// can't be told from a faint white or black overlay (light glass is nearly
/// white), so it isn't glass: a surface fading in shows plainly for a frame
/// or two first.
fn faded(fill: Color32, marker: Color32) -> Option<(f32, f32)> {
    let k = fill.a() as f32 / marker.a().max(1) as f32;
    // (A little under 30%, for the rounding of a faded alpha.)
    if !(0.28..=1.0).contains(&k) {
        return None;
    }
    let off = [(fill.r(), marker.r()), (fill.g(), marker.g()), (fill.b(), marker.b())]
        .iter()
        .map(|&(f, m)| (f as f32 - m as f32 * k).abs())
        .fold(0.0, f32::max);
    (off <= 1.5).then_some((k, off))
}

/// The glass can't be drawn on this GPU (the shader wouldn't build, or the
/// window can't be copied): no more swapping.
static FAILED: AtomicBool = AtomicBool::new(false);

/// Whether Liquid Glass is on and drawing real glass — not fallen back to
/// plain panels on a GPU that can't.
pub fn active() -> bool {
    crate::theme::liquid_markers().is_some() && !FAILED.load(Ordering::Relaxed)
}

/// Fall back to plain panels for good, saying why.
fn give_up(why: &str) {
    eprintln!("[liquid glass] {why}: plain panels instead");
    FAILED.store(true, Ordering::Relaxed);
}

/// Put real glass in place of everything painted this frame with the Liquid
/// Glass theme's panel and floating-surface colours. Call once a frame,
/// after all the UI has been drawn. Does nothing under the other themes.
pub fn glassify(ctx: &egui::Context) {
    if let Some(markers) = crate::theme::liquid_markers()
        && !FAILED.load(Ordering::Relaxed)
    {
        glassify_with(ctx, &markers);
    }
}

/// Runs [`glassify`] when dropped: hold one through the frame's UI code, so
/// it runs last on every way out of it.
#[must_use = "glassify runs when this is dropped"]
pub struct AtFrameEnd(egui::Context);

/// [`glassify`] once the frame's UI is done: when the returned guard drops.
pub fn at_frame_end(ctx: &egui::Context) -> AtFrameEnd {
    AtFrameEnd(ctx.clone())
}

impl Drop for AtFrameEnd {
    fn drop(&mut self) {
        glassify(&self.0);
    }
}

/// [`glassify`] with these markers.
fn glassify_with(ctx: &egui::Context, markers: &Markers) {
    let mut layers: Vec<egui::LayerId> = ctx.memory(|m| m.layer_ids().collect());
    if !layers.contains(&egui::LayerId::background()) {
        layers.insert(0, egui::LayerId::background());
    }
    ctx.graphics_mut(|g| {
        for layer in layers {
            if let Some(list) = g.get_mut(layer) {
                let mut placed = Vec::new();
                for i in 0..list.next_idx().0 {
                    list.mutate_shape(egui::layers::ShapeIdx(i), |s| {
                        let clip = s.clip_rect;
                        swap(&mut s.shape, markers, &mut placed, clip);
                    });
                }
            }
        }
    });
}

/// A marked rectangle inside glass already placed on its layer (a window's
/// title bar, a card on a panel) isn't glass on glass: it's this quiet tint.
fn nested_tint(light: bool, opacity: f32) -> Color32 {
    let tint = if light { Color32::from_white_alpha(90) } else { Color32::from_white_alpha(14) };
    tint.gamma_multiply(opacity)
}

/// Glass in place of `shape` if it's a marked rectangle — or holds one (a
/// frame's shadow and fill come together, and then the shadow goes too).
/// `placed`: the glass already on this layer; `clip`: where the shape shows.
/// Whether it did.
fn swap(shape: &mut Shape, markers: &Markers, placed: &mut Vec<Rect>, clip: Rect) -> bool {
    match shape {
        Shape::Rect(r) if r.blur_width <= 0.0 && r.angle == 0.0 => match markers.kind(r.fill) {
            // On glass by the part that shows: a card scrolled partly out of
            // sight past its panel's edge is still on the panel.
            Some((_, opacity)) if placed.iter().any(|p| p.expand(0.5).contains_rect(r.rect.intersect(clip))) => {
                r.fill = nested_tint(markers.light, opacity);
                false
            }
            Some((kind, opacity)) => {
                placed.push(r.rect);
                *shape = Shape::Callback(callback(r.rect, r.corner_radius, Style::of(kind, markers.light), opacity));
                true
            }
            None => false,
        },
        Shape::Vec(shapes) => {
            let mut glass = false;
            for s in shapes.iter_mut() {
                glass |= swap(s, markers, placed, clip);
            }
            if glass {
                shapes.retain(|s| !matches!(s, Shape::Rect(r) if r.blur_width > 0.0));
            }
            glass
        }
        _ => false,
    }
}

/// The paint callback drawing a glass surface over `rect` (points), with
/// `corners` rounded, in `style`, `opacity` of it (1 = all of it).
pub fn callback(rect: Rect, corners: CornerRadius, style: Style, opacity: f32) -> egui::PaintCallback {
    let margin = style.margin();
    egui::PaintCallback {
        rect: rect.expand(margin),
        callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
            // The layer may have been moved or scaled since (`to_global`): the
            // surface is where the callback ended up, less its margin.
            let scale = info.viewport.width() / (rect.width() + 2.0 * margin).max(1.0);
            let surface = info.viewport.shrink(margin * scale);
            paint(painter.gl(), &info, surface, corners, scale, &style, opacity);
        })),
    }
}

/// The GPU side, built on first use.
struct Gpu {
    program: glow::Program,
    vao: glow::VertexArray,
    fbo: glow::Framebuffer,
    /// Capture textures by size (powers of two), reused frame to frame.
    textures: Vec<([i32; 2], glow::Texture)>,
    u: Uniforms,
    /// The first copy of the window worked (see [`paint`]).
    checked: bool,
}

struct Uniforms {
    backdrop: Option<glow::UniformLocation>,
    cap: Option<glow::UniformLocation>,
    tex: Option<glow::UniformLocation>,
    lod: Option<glow::UniformLocation>,
    rect: Option<glow::UniformLocation>,
    radii: Option<glow::UniformLocation>,
    bezel: Option<glow::UniformLocation>,
    refract: Option<glow::UniformLocation>,
    tint: Option<glow::UniformLocation>,
    vibrancy: Option<glow::UniformLocation>,
    spec: Option<glow::UniformLocation>,
    shadow: Option<glow::UniformLocation>,
    px: Option<glow::UniformLocation>,
    opacity: Option<glow::UniformLocation>,
}

static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

/// Textures kept for reuse, at most (by size; the oldest goes).
const MAX_TEXTURES: usize = 6;
/// Largest capture texture side.
const MAX_SIDE: i32 = 2048;

fn paint(
    gl: &glow::Context,
    info: &egui::PaintCallbackInfo,
    surface: Rect,
    corners: CornerRadius,
    scale: f32,
    style: &Style,
    opacity: f32,
) {
    // Given up earlier this frame (the surfaces after the first are queued).
    if FAILED.load(Ordering::Relaxed) {
        return;
    }
    let mut guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        match unsafe { Gpu::new(gl) } {
            Ok(gpu) => *guard = Some(gpu),
            Err(e) => {
                give_up(&format!("the glass shader isn't available on this GPU ({e})"));
                return;
            }
        }
    }
    let Some(gpu) = guard.as_mut() else { return };
    let ppp = info.pixels_per_point;
    let [sw, sh] = info.screen_size_px.map(|v| v as f32);
    let px = ppp * scale;

    // The surface in window pixels, y from the bottom (as OpenGL counts).
    let left = surface.left() * ppp;
    let right = surface.right() * ppp;
    let bottom = sh - surface.bottom() * ppp;
    let top = sh - surface.top() * ppp;
    if right - left < 1.0 || top - bottom < 1.0 {
        return;
    }
    let (hw, hh) = ((right - left) * 0.5, (top - bottom) * 0.5);
    let small = hw.min(hh);
    // A small surface gets a narrower rim (it'd swallow a tooltip whole).
    let bezel = (style.bezel * px).min(small * 0.8).max(1.0);
    let refract = style.refract * px * (bezel / (style.bezel * px).max(1.0));
    let blur = style.blur * px;

    // What's captured: the surface, plus what its rim and frost reach.
    let reach = blur * 1.5 + 4.0;
    let x0 = (left - reach).floor().max(0.0) as i32;
    let x1 = (right + reach).ceil().min(sw) as i32;
    let y0 = (bottom - reach).floor().max(0.0) as i32;
    let y1 = (top + reach).ceil().min(sh) as i32;
    let (cw, ch) = (x1 - x0, y1 - y0);
    if cw <= 0 || ch <= 0 {
        return;
    }
    // At half size: the frost hides it, and it's a quarter of the copying.
    let side = |n: i32| ((n as u32 / 2).max(8).next_power_of_two() as i32).min(MAX_SIDE);
    let size = [side(cw), side(ch)];
    let lod = (blur * size[0] as f32 / cw as f32).max(1.0).log2();

    let radius = |r: u8| (r as f32 * px).min(small);
    // Top-right, bottom-right, top-left, bottom-left, with y up.
    let radii = [radius(corners.ne), radius(corners.se), radius(corners.nw), radius(corners.sw)];
    let [sx, sy, sblur, sopacity] = style.shadow;

    unsafe {
        // The first time, check the window can be copied: not if it's
        // anti-aliased (multisampled — some drivers force it), which can't be
        // copied at a smaller size; nor on any other error. The glass would
        // show garbage.
        let first = !gpu.checked;
        if first {
            if gl.get_parameter_i32(glow::SAMPLE_BUFFERS) > 0 {
                give_up("the window is anti-aliased (multisampled), so it can't be copied");
                return;
            }
            // Errors from before, so any after the copy are its own.
            for _ in 0..16 {
                if gl.get_error() == glow::NO_ERROR {
                    break;
                }
            }
        }
        let Some(texture) = gpu.texture(gl, size) else { return };
        let read = gl.get_parameter_framebuffer(glow::READ_FRAMEBUFFER_BINDING);
        let draw = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
        // Copy what's behind (everything egui drew so far) into the texture.
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, draw);
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(gpu.fbo));
        gl.framebuffer_texture_2d(glow::DRAW_FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(texture), 0);
        gl.disable(glow::SCISSOR_TEST);
        gl.blit_framebuffer(x0, y0, x1, y1, 0, 0, size[0], size[1], glow::COLOR_BUFFER_BIT, glow::LINEAR);
        let copied = !first || gl.get_error() == glow::NO_ERROR;
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, draw);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, read);
        gl.enable(glow::SCISSOR_TEST);
        if !copied {
            give_up("copying the window didn't work on this GPU");
            return;
        }
        gpu.checked = true;
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.generate_mipmap(glow::TEXTURE_2D);

        // Then the glass, over the callback's area (the surface and its shadow).
        gl.use_program(Some(gpu.program));
        gl.bind_vertex_array(Some(gpu.vao));
        let u = &gpu.u;
        gl.uniform_1_i32(u.backdrop.as_ref(), 0);
        gl.uniform_4_f32(u.cap.as_ref(), x0 as f32, y0 as f32, cw as f32, ch as f32);
        gl.uniform_2_f32(u.tex.as_ref(), size[0] as f32, size[1] as f32);
        gl.uniform_1_f32(u.lod.as_ref(), lod);
        gl.uniform_4_f32(u.rect.as_ref(), (left + right) * 0.5, (bottom + top) * 0.5, hw, hh);
        gl.uniform_4_f32(u.radii.as_ref(), radii[0], radii[1], radii[2], radii[3]);
        gl.uniform_1_f32(u.bezel.as_ref(), bezel);
        gl.uniform_1_f32(u.refract.as_ref(), refract);
        let [r, g, b, a] = style.tint;
        gl.uniform_4_f32(u.tint.as_ref(), r, g, b, a);
        gl.uniform_1_f32(u.vibrancy.as_ref(), style.vibrancy);
        gl.uniform_1_f32(u.spec.as_ref(), style.spec);
        gl.uniform_4_f32(u.shadow.as_ref(), sx * px, -sy * px, sblur * px, sopacity);
        gl.uniform_1_f32(u.px.as_ref(), px);
        gl.uniform_1_f32(u.opacity.as_ref(), opacity);
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
        gl.bind_vertex_array(None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.use_program(None);
    }
}

impl Gpu {
    unsafe fn new(gl: &glow::Context) -> Result<Gpu, String> {
        let version = egui_glow::ShaderVersion::get(gl);
        let header = match version {
            egui_glow::ShaderVersion::Gl140 => "#version 140\n",
            egui_glow::ShaderVersion::Es300 => "#version 300 es\nprecision highp float;\n",
            other => return Err(format!("needs OpenGL 3 or OpenGL ES 3 shaders, not {other:?}")),
        };
        unsafe {
            let program = gl.create_program()?;
            let mut shaders = Vec::new();
            for (kind, source) in [(glow::VERTEX_SHADER, VERTEX), (glow::FRAGMENT_SHADER, FRAGMENT)] {
                let shader = gl.create_shader(kind)?;
                gl.shader_source(shader, &format!("{header}{source}"));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    let log = gl.get_shader_info_log(shader);
                    gl.delete_shader(shader);
                    gl.delete_program(program);
                    return Err(log);
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }
            gl.link_program(program);
            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(log);
            }
            let at = |name: &str| gl.get_uniform_location(program, name);
            let u = Uniforms {
                backdrop: at("u_backdrop"),
                cap: at("u_cap"),
                tex: at("u_tex"),
                lod: at("u_lod"),
                rect: at("u_rect"),
                radii: at("u_radii"),
                bezel: at("u_bezel"),
                refract: at("u_refract"),
                tint: at("u_tint"),
                vibrancy: at("u_vibrancy"),
                spec: at("u_spec"),
                shadow: at("u_shadow"),
                px: at("u_px"),
                opacity: at("u_opacity"),
            };
            let vao = gl.create_vertex_array()?;
            let fbo = gl.create_framebuffer()?;
            Ok(Gpu { program, vao, fbo, textures: Vec::new(), u, checked: false })
        }
    }

    /// A capture texture of `size` (with room for its mipmaps), if the GPU
    /// can make one.
    unsafe fn texture(&mut self, gl: &glow::Context, size: [i32; 2]) -> Option<glow::Texture> {
        if let Some(i) = self.textures.iter().position(|(s, _)| *s == size) {
            let entry = self.textures.remove(i);
            self.textures.push(entry);
            return Some(entry.1);
        }
        unsafe {
            let texture = gl.create_texture().ok()?;
            if self.textures.len() >= MAX_TEXTURES {
                let (_, old) = self.textures.remove(0);
                gl.delete_texture(old);
            }
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                size[0],
                size[1],
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.bind_texture(glow::TEXTURE_2D, None);
            self.textures.push((size, texture));
            Some(texture)
        }
    }
}

/// One triangle covering the viewport (the callback's area).
const VERTEX: &str = r#"
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
"#;

/// The glass. Window pixels throughout, y from the bottom.
const FRAGMENT: &str = r#"
uniform sampler2D u_backdrop; // what's behind: the captured region
uniform vec4 u_cap;           // that region: x, y, width, height
uniform vec2 u_tex;           // its texture's size, texels
uniform float u_lod;          // the frost's mip level
uniform vec4 u_rect;          // the surface: centre x, y, half width, half height
uniform vec4 u_radii;         // corners: top right, bottom right, top left, bottom left
uniform float u_bezel;        // the curved rim's width
uniform float u_refract;      // how far the rim shifts what's behind, at most
uniform vec4 u_tint;          // straight RGB, strength
uniform float u_vibrancy;
uniform float u_spec;
uniform vec4 u_shadow;        // offset x, y; blur; opacity
uniform float u_px;           // pixels per point
uniform float u_opacity;      // all of it, or fading in or out
out vec4 o_color;

// Signed distance to the rounded rectangle (negative inside).
float box_sdf(vec2 p) {
    vec4 r = u_radii;
    r.xy = (p.x > 0.0) ? r.xy : r.zw;
    r.x = (p.y > 0.0) ? r.x : r.y;
    vec2 q = abs(p) - u_rect.zw + r.x;
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}

// The backdrop at window position `at`, frosted: mip level `lod`, smoothed
// over a hexagon of taps.
vec3 frosted(vec2 at, float lod) {
    vec2 uv = (at - u_cap.xy) / u_cap.zw;
    vec2 s = exp2(lod) / u_tex;
    vec3 c = textureLod(u_backdrop, uv, lod).rgb * 0.28;
    c += textureLod(u_backdrop, uv + vec2(1.0, 0.0) * s, lod).rgb * 0.12;
    c += textureLod(u_backdrop, uv + vec2(0.5, 0.866) * s, lod).rgb * 0.12;
    c += textureLod(u_backdrop, uv + vec2(-0.5, 0.866) * s, lod).rgb * 0.12;
    c += textureLod(u_backdrop, uv + vec2(-1.0, 0.0) * s, lod).rgb * 0.12;
    c += textureLod(u_backdrop, uv + vec2(-0.5, -0.866) * s, lod).rgb * 0.12;
    c += textureLod(u_backdrop, uv + vec2(0.5, -0.866) * s, lod).rgb * 0.12;
    return c;
}

// How far the rim shifts what's behind, as a share of `u_refract`, at `x`
// across it (0 at the outer edge, 1 where the top is flat). The rim is
// rounded like a lens, its slope easing off across its whole width, so what
// passes under an edge is magnified as it nears it: about twice halfway
// across (with the shift at about half the rim's width), more at the edge.
float bend(float x) {
    float u = 1.0 - x;
    return u * u;
}

void main() {
    vec2 frag = gl_FragCoord.xy;
    vec2 p = frag - u_rect.xy;
    float d = box_sdf(p);
    float cover = clamp(0.5 - d, 0.0, 1.0);
    float ds = box_sdf(p - u_shadow.xy);
    float shadow = u_shadow.w * (1.0 - smoothstep(-u_shadow.z * 0.4, u_shadow.z, ds)) * (1.0 - cover);
    if (cover <= 0.0) {
        o_color = vec4(0.0, 0.0, 0.0, shadow * u_opacity);
        return;
    }

    // The surface's outward normal, and how far in from the edge this is.
    vec2 e = vec2(0.5, 0.0);
    vec2 g = vec2(box_sdf(p + e.xy) - box_sdf(p - e.xy), box_sdf(p + e.yx) - box_sdf(p - e.yx));
    vec2 n = g / max(length(g), 1e-4);
    float x = clamp(max(-d, 0.0) / u_bezel, 0.0, 1.0);
    float rim = 1.0 - x;

    // A convex rim gathers the light inward: it shows what's further in,
    // magnified — and, like the polished edge of a lens, clearer toward the
    // edge than the frosted middle.
    float shift = u_refract * bend(x);
    vec2 at = frag - n * shift;
    float clear = pow(rim, 1.5);
    float lod = mix(u_lod, min(u_lod, 1.0), clear);
    vec3 c = frosted(at, lod);
    // A hint of colour fringing where the rim bends most.
    float fringe = rim * rim;
    if (fringe > 0.002) {
        vec2 ur = (at - n * shift * 0.12 - u_cap.xy) / u_cap.zw;
        vec2 ub = (at + n * shift * 0.12 - u_cap.xy) / u_cap.zw;
        c.r = mix(c.r, textureLod(u_backdrop, ur, lod).r, fringe * 0.6);
        c.b = mix(c.b, textureLod(u_backdrop, ub, lod).b, fringe * 0.6);
    }

    // Colours lifted a little, then the tint.
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = clamp(mix(vec3(l), c, 1.0 + u_vibrancy), 0.0, 1.0);
    c = mix(c, u_tint.rgb, u_tint.a * (1.0 - 0.35 * rim * rim));

    // The light comes from the top left: a thin bright edge all round,
    // brightest facing it, and a sheen across the curved rim — fainter on
    // the far side, which is also a touch darker.
    vec2 light = normalize(vec2(-0.55, 0.83));
    float facing = dot(n, light);
    float toward = max(facing, 0.0);
    float away = max(-facing, 0.0);
    float edge = 1.0 - smoothstep(0.0, 1.6 * u_px, max(-d, 0.0));
    float sheen = rim * rim * rim;
    float t3 = toward * toward * toward;
    float a3 = away * away * away;
    float spec = edge * (0.14 + 0.80 * t3 + 0.50 * a3)
        + sheen * (0.50 * t3 + 0.22 * a3);
    c = c * (1.0 - 0.10 * sheen * away) + vec3(spec * u_spec);

    c = clamp(c, 0.0, 1.0);
    o_color = (vec4(c * cover, cover) + vec4(0.0, 0.0, 0.0, shadow)) * u_opacity;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{pos2, vec2, Stroke};

    const MARKERS: Markers = Markers {
        panel: Color32::from_rgba_premultiplied(18, 20, 25, 171),
        float: Color32::from_rgba_premultiplied(25, 26, 34, 182),
        light: false,
    };

    fn rect(fill: Color32) -> Shape {
        Shape::rect_filled(Rect::from_min_size(pos2(10.0, 20.0), vec2(200.0, 100.0)), 22, fill)
    }

    /// Only the marked fills become glass; a frame's shadow goes with it,
    /// and an unmarked frame keeps its own.
    #[test]
    fn marked_surfaces_become_glass() {
        let mut panel = rect(MARKERS.panel);
        assert!(swap(&mut panel, &MARKERS, &mut Vec::new(), Rect::EVERYTHING));
        let Shape::Callback(cb) = &panel else { panic!("not glass: {panel:?}") };
        let margin = Style::of(Kind::Panel, false).margin();
        assert_eq!(cb.rect, Rect::from_min_size(pos2(10.0, 20.0), vec2(200.0, 100.0)).expand(margin));

        let mut plain = rect(Color32::from_rgb(30, 30, 30));
        assert!(!swap(&mut plain, &MARKERS, &mut Vec::new(), Rect::EVERYTHING));
        assert!(matches!(plain, Shape::Rect(_)));

        let shadow = egui::epaint::Shadow { offset: [0, 4], blur: 14, spread: 0, color: Color32::from_black_alpha(110) };
        let framed = |fill| {
            let r = Rect::from_min_size(pos2(0.0, 0.0), vec2(80.0, 40.0));
            Shape::Vec(vec![
                Shape::from(shadow.as_shape(r, 12)),
                Shape::Rect(egui::epaint::RectShape::new(r, 12, fill, Stroke::new(1.0_f32, Color32::BLACK), egui::StrokeKind::Inside)),
            ])
        };
        let mut window = framed(MARKERS.float);
        assert!(swap(&mut window, &MARKERS, &mut Vec::new(), Rect::EVERYTHING));
        let Shape::Vec(v) = &window else { panic!() };
        assert_eq!(v.len(), 1, "the frame's shadow went: {v:?}");
        assert!(matches!(v[0], Shape::Callback(_)));

        let mut card = framed(Color32::from_rgb(40, 40, 40));
        assert!(!swap(&mut card, &MARKERS, &mut Vec::new(), Rect::EVERYTHING));
        let Shape::Vec(v) = &card else { panic!() };
        assert_eq!(v.len(), 2, "an ordinary card keeps its shadow");
    }

    /// Blurred rectangles (shadows, glows) and rotated ones are never glass,
    /// even in the marker colour.
    #[test]
    fn shadows_and_rotated_rects_stay() {
        let mut glow = Shape::Rect(
            egui::epaint::RectShape::filled(Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)), 2, MARKERS.panel)
                .with_blur_width(6.0),
        );
        assert!(!swap(&mut glow, &MARKERS, &mut Vec::new(), Rect::EVERYTHING));
        let mut turned = Shape::Rect(
            egui::epaint::RectShape::filled(Rect::from_min_size(pos2(0.0, 0.0), vec2(10.0, 10.0)), 2, MARKERS.panel)
                .with_angle(0.3),
        );
        assert!(!swap(&mut turned, &MARKERS, &mut Vec::new(), Rect::EVERYTHING));
    }

    /// A surface fading in or out is still glass — of the right kind,
    /// whichever theme's markers, down to about a third of its opacity.
    #[test]
    fn faded_markers_are_still_glass() {
        for light in [false, true] {
            let m = crate::theme::liquid_markers_for(light);
            for pct in 30..=100 {
                let k = pct as f32 / 100.0;
                let (panel, float) = (m.panel.gamma_multiply(k), m.float.gamma_multiply(k));
                let is = |found: Option<(Kind, f32)>, kind: Kind| {
                    found.is_some_and(|(found, o)| (o - k).abs() < 0.02 && found == kind)
                };
                assert!(is(m.kind(panel), Kind::Panel), "{light} {k}");
                assert!(is(m.kind(float), Kind::Float), "{light} {k}");
            }
            assert_eq!(m.kind(m.panel.gamma_multiply(0.2)), None, "too faint to tell");
            assert_eq!(m.kind(m.panel.lerp_to_gamma(Color32::WHITE, 0.07)), None, "a highlight over it");
            assert_eq!(m.kind(Color32::TRANSPARENT), None);
        }
    }

    /// Faint white and black overlays (veils, highlights, dimming) are never
    /// glass — light glass is nearly white, dark glass nearly black.
    #[test]
    fn faint_overlays_are_not_glass() {
        for light in [false, true] {
            let m = crate::theme::liquid_markers_for(light);
            for a in 0..=255 {
                assert_eq!(m.kind(Color32::from_white_alpha(a)), None, "{light} white {a}");
                assert_eq!(m.kind(Color32::from_black_alpha(a)), None, "{light} black {a}");
            }
        }
    }

    /// A card on a glass panel is a quiet tint, not glass on glass — also
    /// while it's scrolled partly out of sight past the panel's edge.
    #[test]
    fn cards_on_glass_are_tints() {
        let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
        let card = Rect::from_min_max(pos2(10.0, 80.0), pos2(90.0, 140.0));
        let mut shape = Shape::rect_filled(card, 8, MARKERS.panel);
        assert!(!swap(&mut shape, &MARKERS, &mut vec![panel], panel.shrink(4.0)));
        assert!(matches!(&shape, Shape::Rect(r) if r.fill == nested_tint(false, 1.0)), "{shape:?}");
        // Where it shows off the panel, it's glass of its own.
        let mut shape = Shape::rect_filled(card, 8, MARKERS.panel);
        assert!(swap(&mut shape, &MARKERS, &mut vec![panel], Rect::EVERYTHING));
    }

    /// Floating surfaces are a little more frosted and cast a deeper shadow
    /// than panels; light glass is pale, dark glass dark.
    #[test]
    fn the_kinds_of_glass() {
        for light in [false, true] {
            let (panel, float) = (Style::of(Kind::Panel, light), Style::of(Kind::Float, light));
            assert!(float.blur > panel.blur && float.shadow[1] > panel.shadow[1]);
            assert!(float.margin() > panel.margin());
            let tint = panel.tint[0] + panel.tint[1] + panel.tint[2];
            assert_eq!(tint > 1.5, light);
        }
    }

    /// The whole frame goes through: the panels' layer and a window's — the
    /// marked shapes only.
    #[test]
    fn glassify_walks_every_layer() {
        let ctx = egui::Context::default();
        let mut found = (0, 0, 0);
        // (A window shows from its second frame: the first only sizes it.)
        for _ in 0..3 {
            frame(&ctx, &mut found);
        }
        assert_eq!(found, (1, 1, 1), "the marked panel and the window, not the plain rectangle");
    }

    fn frame(ctx: &egui::Context, found: &mut (usize, usize, usize)) {
        let _ = ctx.run_ui(Default::default(), |ui| {
            let r = Rect::from_min_size(pos2(0.0, 0.0), vec2(50.0, 50.0));
            ui.painter().rect_filled(r, 8, MARKERS.panel);
            ui.painter().rect_filled(r, 8, Color32::from_rgb(9, 9, 9));
            egui::Window::new("w").frame(egui::Frame::new().fill(MARKERS.float)).show(ui.ctx(), |ui| {
                ui.label("hi");
            });
            glassify_with(ui.ctx(), &MARKERS);
            let glass = |s: &&egui::epaint::ClippedShape| match &s.shape {
                Shape::Callback(_) => true,
                Shape::Vec(v) => v.iter().any(|s| matches!(s, Shape::Callback(_))),
                _ => false,
            };
            let here = ui.layer_id();
            let windows: Vec<_> = ui.ctx().memory(|m| m.layer_ids().filter(|l| *l != here).collect());
            ui.ctx().graphics(|g| {
                let panels = g.get(here).map_or(0, |l| l.all_entries().filter(glass).count());
                let others = g.get(here).map_or(0, |l| l.all_entries().filter(|s| matches!(s.shape, Shape::Rect(_))).count());
                let windows = windows.iter().map(|l| g.get(*l).map_or(0, |l| l.all_entries().filter(glass).count())).sum();
                *found = (panels, others, windows);
            });
        });
    }
}
