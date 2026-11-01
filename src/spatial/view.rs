//! Drawing a [`Scene`]: for each pixel on screen, the nearest surface of the
//! picture that lands there as the view moves — a march along the line of
//! sight through the depth, near to far (parallax occlusion mapping) — shown
//! over the rebuilt background wherever the picture is torn at a depth edge.
//! Per pixel, so people and things come apart cleanly at their edges instead
//! of stretching.
//!
//! The picture itself is the viewer's texture of it (already on the GPU);
//! the scene's maps and background go up once per scene. All are
//! gamma-encoded like the window, so what's sampled is drawn as it is. Where
//! the shader can't be built, [`available`] turns false and the viewer falls
//! back to its plain mesh.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui;
use glow::HasContext as _;

use super::Scene;

/// How the scene is seen this frame.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    /// How far a surface moves per unit of depth away from the focus, as a
    /// share of the picture's width and height.
    pub shift: egui::Vec2,
    /// The picture shown this much bigger than its rect, so the edges never
    /// move into view.
    pub zoom: f32,
    /// The picture's corner radius (points), as the viewer rounds it.
    pub radius: f32,
}

/// The shader couldn't be built on this GPU: the viewer draws its mesh.
static FAILED: AtomicBool = AtomicBool::new(false);

/// Whether scenes can be drawn here (the shader built, or hasn't been tried).
pub fn available() -> bool {
    !FAILED.load(Ordering::Relaxed)
}

/// Draw `scene` of the picture in `texture` into `rect` (points), seen as
/// `look`.
pub fn callback(rect: egui::Rect, scene: Arc<Scene>, texture: egui::TextureId, look: Look) -> egui::PaintCallback {
    egui::PaintCallback {
        rect,
        callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
            // Gone (the viewer let it go this frame): nothing to draw.
            if let Some(picture) = painter.texture(texture) {
                paint(painter.gl(), &info, &scene, picture, look);
            }
        })),
    }
}

/// The program, and the textures of the scene last drawn.
struct Gpu {
    program: glow::Program,
    vao: glow::VertexArray,
    u: Uniforms,
    /// The scene's id, and its maps and background.
    scene: Option<(u64, [glow::Texture; 2])>,
}

struct Uniforms {
    color: Option<glow::UniformLocation>,
    maps: Option<glow::UniformLocation>,
    back: Option<glow::UniformLocation>,
    rect: Option<glow::UniformLocation>,
    shift: Option<glow::UniformLocation>,
    focus: Option<glow::UniformLocation>,
    zoom: Option<glow::UniformLocation>,
    radius: Option<glow::UniformLocation>,
    steps: Option<glow::UniformLocation>,
}

static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

fn paint(gl: &glow::Context, info: &egui::PaintCallbackInfo, scene: &Scene, picture: glow::Texture, look: Look) {
    if FAILED.load(Ordering::Relaxed) {
        return;
    }
    let mut guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        match unsafe { Gpu::new(gl) } {
            Ok(gpu) => *guard = Some(gpu),
            Err(e) => {
                eprintln!("[spatial scene] the shader isn't available on this GPU ({e}): plain mesh instead");
                FAILED.store(true, Ordering::Relaxed);
                return;
            }
        }
    }
    let Some(gpu) = guard.as_mut() else { return };
    let vp = info.viewport_in_pixels();
    if vp.width_px <= 0 || vp.height_px <= 0 {
        return;
    }
    // How far the farthest surface moves, in pixels: march about a pixel at
    // a time along the line of sight.
    let travel = (look.shift.x * vp.width_px as f32).hypot(look.shift.y * vp.height_px as f32);
    let steps = (travel * 1.25).clamp(8.0, MAX_STEPS as f32);
    unsafe {
        let Some([maps, back]) = gpu.textures(gl, scene) else { return };
        gl.use_program(Some(gpu.program));
        gl.bind_vertex_array(Some(gpu.vao));
        for (unit, texture) in [picture, maps, back].iter().enumerate() {
            gl.active_texture(glow::TEXTURE0 + unit as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(*texture));
        }
        let u = &gpu.u;
        gl.uniform_1_i32(u.color.as_ref(), 0);
        gl.uniform_1_i32(u.maps.as_ref(), 1);
        gl.uniform_1_i32(u.back.as_ref(), 2);
        gl.uniform_4_f32(
            u.rect.as_ref(),
            vp.left_px as f32,
            vp.from_bottom_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
        );
        gl.uniform_2_f32(u.shift.as_ref(), look.shift.x, look.shift.y);
        gl.uniform_1_f32(u.focus.as_ref(), scene.focus);
        gl.uniform_1_f32(u.zoom.as_ref(), look.zoom.max(1.0));
        gl.uniform_1_f32(u.radius.as_ref(), look.radius * info.pixels_per_point);
        gl.uniform_1_f32(u.steps.as_ref(), steps);
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
        for unit in (0..3).rev() {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        gl.bind_vertex_array(None);
        gl.use_program(None);
    }
}

impl Gpu {
    unsafe fn new(gl: &glow::Context) -> Result<Gpu, String> {
        let header = match egui_glow::ShaderVersion::get(gl) {
            egui_glow::ShaderVersion::Gl140 => "#version 140\n",
            egui_glow::ShaderVersion::Es300 => "#version 300 es\nprecision highp float;\n",
            other => return Err(format!("needs OpenGL 3 or OpenGL ES 3 shaders, not {other:?}")),
        };
        unsafe {
            let program = gl.create_program()?;
            let mut shaders = Vec::new();
            for (kind, source) in [(glow::VERTEX_SHADER, VERTEX), (glow::FRAGMENT_SHADER, FRAGMENT)] {
                let shader = gl.create_shader(kind)?;
                gl.shader_source(shader, &format!("{header}#define MAX_STEPS {MAX_STEPS}\n{source}"));
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
                color: at("u_color"),
                maps: at("u_maps"),
                back: at("u_back"),
                rect: at("u_rect"),
                shift: at("u_shift"),
                focus: at("u_focus"),
                zoom: at("u_zoom"),
                radius: at("u_radius"),
                steps: at("u_steps"),
            };
            let vao = gl.create_vertex_array()?;
            Ok(Gpu { program, vao, u, scene: None })
        }
    }

    /// `scene`'s maps and background, uploaded the first time it's drawn
    /// (the last scene's go then).
    unsafe fn textures(&mut self, gl: &glow::Context, scene: &Scene) -> Option<[glow::Texture; 2]> {
        if let Some((id, t)) = self.scene
            && id == scene.id
        {
            return Some(t);
        }
        unsafe {
            if let Some((_, old)) = self.scene.take() {
                for t in old {
                    gl.delete_texture(t);
                }
            }
            let maps = upload(gl, scene.w, scene.h, &scene.maps)?;
            let Some(back) = upload(gl, scene.w, scene.h, &scene.back) else {
                gl.delete_texture(maps);
                return None;
            };
            self.scene = Some((scene.id, [maps, back]));
            Some([maps, back])
        }
    }
}

/// An RGBA8 texture of `w` × `h` from `data`, smoothly filtered.
unsafe fn upload(gl: &glow::Context, w: u32, h: u32, data: &[u8]) -> Option<glow::Texture> {
    unsafe {
        let texture = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            w as i32,
            h as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(data)),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);
        Some(texture)
    }
}

/// Most steps a line of sight is marched in (about one per pixel moved).
const MAX_STEPS: u32 = 96;

/// One triangle covering the viewport (the picture's rect).
const VERTEX: &str = r#"
void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
"#;

/// The scene, per pixel. Depth is 0 far .. 1 near; a surface at depth `t`
/// shows `(t - focus) * shift` from where it is in the picture.
const FRAGMENT: &str = r#"
uniform sampler2D u_color;  // the picture (the viewer's texture of it)
uniform sampler2D u_maps;   // r: depth, g: how much the picture shows, b: the background's depth
uniform sampler2D u_back;   // the background, rebuilt behind things in front
uniform vec4 u_rect;        // the picture on screen: left, bottom (window px, y up), width, height
uniform vec2 u_shift;       // a surface's move per unit of depth from the focus (share of the picture)
uniform float u_focus;      // the depth that stays still
uniform float u_zoom;       // the picture this much bigger than its rect
uniform float u_radius;     // corner radius, px
uniform float u_steps;      // steps along a line of sight
out vec4 o_color;

// Where in the picture the nearest surface along the line of sight through
// `uv` is, marching from near to far: the first depth at which the surface
// there is no farther than the line, refined between the last two steps.
// The two layers' depths are in different channels.
vec2 nearest_front(vec2 uv) {
    float prev_t = 1.0;
    float prev_h = texture(u_maps, uv - (1.0 - u_focus) * u_shift).r;
    if (prev_h >= prev_t) return uv - (1.0 - u_focus) * u_shift;
    for (int i = 1; i <= MAX_STEPS; i++) {
        if (float(i) > u_steps) break;
        float t = 1.0 - float(i) / u_steps;
        vec2 at = uv - (t - u_focus) * u_shift;
        float h = texture(u_maps, at).r;
        if (h >= t) {
            float k = (prev_t - prev_h) / max((prev_t - prev_h) + (h - t), 1e-5);
            return uv - (mix(prev_t, t, k) - u_focus) * u_shift;
        }
        prev_t = t;
        prev_h = h;
    }
    return uv + u_focus * u_shift;
}

vec2 nearest_back(vec2 uv) {
    float prev_t = 1.0;
    float prev_h = texture(u_maps, uv - (1.0 - u_focus) * u_shift).b;
    if (prev_h >= prev_t) return uv - (1.0 - u_focus) * u_shift;
    for (int i = 1; i <= MAX_STEPS; i++) {
        if (float(i) > u_steps) break;
        float t = 1.0 - float(i) / u_steps;
        vec2 at = uv - (t - u_focus) * u_shift;
        float h = texture(u_maps, at).b;
        if (h >= t) {
            float k = (prev_t - prev_h) / max((prev_t - prev_h) + (h - t), 1e-5);
            return uv - (mix(prev_t, t, k) - u_focus) * u_shift;
        }
        prev_t = t;
        prev_h = h;
    }
    return uv + u_focus * u_shift;
}

// What's seen through `base` (where in the picture it'd be unmoved): its
// colour, over the background where the picture is see-through, and its
// depth. The picture is filtered as if unmoved (`gx`, `gy`), not by the
// jumps at depth edges, which would blur them where it has mipmaps.
vec4 seen(vec2 base, vec2 gx, vec2 gy) {
    vec2 at = nearest_front(base);
    vec4 m = texture(u_maps, at);
    vec3 c = textureGrad(u_color, at, gx, gy).rgb;
    if (m.g < 0.999) {
        vec3 behind = texture(u_back, nearest_back(base)).rgb;
        c = mix(behind, c, m.g);
    }
    return vec4(c, m.r);
}

void main() {
    vec2 px = gl_FragCoord.xy - u_rect.xy;
    // Rounded corners, as the viewer draws the picture.
    vec2 half_size = u_rect.zw * 0.5;
    vec2 q = abs(px - half_size) - half_size + u_radius;
    float sd = min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - u_radius;
    float cover = clamp(0.5 - sd, 0.0, 1.0);

    vec2 uv = vec2(px.x / u_rect.z, 1.0 - px.y / u_rect.w);
    vec2 base = 0.5 + (uv - 0.5) / u_zoom;
    vec2 gx = dFdx(base);
    vec2 gy = dFdy(base);
    vec4 s = seen(base, gx, gy);
    vec3 c = s.rgb;
    // Where the depth seen jumps from one pixel to the next — the outline
    // of something in front — four more looks across the pixel (a rotated
    // grid) smooth the outline's stair steps.
    if (fwidth(s.a) > 0.04) {
        c += seen(base - 0.375 * gx - 0.125 * gy, gx, gy).rgb;
        c += seen(base + 0.125 * gx - 0.375 * gy, gx, gy).rgb;
        c += seen(base + 0.375 * gx + 0.125 * gy, gx, gy).rgb;
        c += seen(base - 0.125 * gx + 0.375 * gy, gx, gy).rgb;
        c *= 0.2;
    }
    // Outside the rounded corners `cover` is 0: nothing drawn (no discard,
    // so the derivatives above hold everywhere).
    o_color = vec4(c * cover, cover);
}
"#;
