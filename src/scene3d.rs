//! Interactive 3D GLB viewer for the centre panel (Pixal3D output), Linux/Windows
//! only (compiled out on macOS alongside Pixal3D itself).
//!
//! Renders a loaded `.glb` model with PBR materials + lighting into eframe's
//! existing glow (OpenGL) context, inside an egui paint callback. Drag to orbit,
//! scroll to zoom. Built on `three-d` 0.19 (which pins the same glow 0.17 /
//! egui_glow 0.34 that eframe 0.34 uses, so the GL context type unifies).
//!
//! Loading is one-shot per file: the GLB bytes are read on the UI thread when the
//! path changes, then parsed + uploaded to the GPU inside the callback (the only
//! place a `three_d::Context` is available). The model is static after that, so
//! we only repaint while the user is interacting.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use eframe::egui;
use three_d::*;

use crate::theme::MUTED;

/// UI-facing handle. Owns the shared render state and tracks which file is loaded.
pub struct Scene3D {
    inner: Arc<Mutex<Inner>>,
    /// Path currently loaded (or attempted), so we only read+upload on change.
    loaded_path: Option<PathBuf>,
    /// Set if reading the GLB file failed (shown instead of the viewport).
    load_err: Option<String>,
}

/// Shared, callback-accessible state. Must be `Send + Sync + 'static` because the
/// `egui_glow::CallbackFn` requires it.
struct Inner {
    // Orbit camera, driven by the UI.
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
    /// GLB bytes waiting to be parsed + uploaded (consumed in the callback).
    pending: Option<Vec<u8>>,
    /// Once a model is uploaded, frame the camera to its bounds (one-shot).
    needs_framing: bool,
    /// GPU-side resources, built lazily once the GL context is available.
    gpu: Option<Gpu>,
    /// Background clear colour for the viewport (RGBA, linear-ish).
    bg: [f32; 4],
}

struct Gpu {
    ctx: Context,
    /// Raw glow context, kept so we can reset GL state after rendering.
    gl: Arc<glow::Context>,
    model: Option<Model<PhysicalMaterial>>,
    ambient: AmbientLight,
    key: DirectionalLight,
    fill: DirectionalLight,
}

impl Default for Scene3D {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene3D {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                yaw: 0.7,
                pitch: 0.4,
                distance: 3.0,
                target: vec3(0.0, 0.0, 0.0),
                pending: None,
                needs_framing: false,
                gpu: None,
                bg: [0.0; 4],
            })),
            loaded_path: None,
            load_err: None,
        }
    }

    /// Render the viewer for `glb_path` (the most recently generated model, if
    /// any). Call every frame while the Pixal3D view is active.
    pub fn show(&mut self, ui: &mut egui::Ui, glb_path: Option<&Path>) {
        // Load (read bytes) when the target file changes. Done on the UI thread;
        // parsing + GPU upload happen later in the callback.
        if glb_path.map(Path::to_path_buf) != self.loaded_path {
            self.loaded_path = glb_path.map(Path::to_path_buf);
            self.load_err = None;
            if let Some(p) = glb_path {
                match std::fs::read(p) {
                    Ok(bytes) => {
                        let mut inner = self.inner.lock().unwrap();
                        inner.pending = Some(bytes);
                        inner.needs_framing = true;
                    }
                    Err(e) => self.load_err = Some(format!("Couldn't read model: {e}")),
                }
            }
        }

        if glb_path.is_none() {
            placeholder(ui, "Generate a 3D model to view it here.");
            return;
        }
        if let Some(err) = &self.load_err {
            placeholder(ui, err);
            return;
        }

        // Claim the whole panel and let it orbit / zoom.
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

        let mut interacting = false;
        {
            let mut inner = self.inner.lock().unwrap();
            inner.bg = clear_color();
            if response.dragged() {
                let d = response.drag_delta();
                inner.yaw -= d.x * 0.01;
                inner.pitch = (inner.pitch + d.y * 0.01).clamp(-1.54, 1.54);
                interacting = true;
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    inner.distance = (inner.distance * (1.0 - scroll * 0.0015)).clamp(0.05, 1.0e5);
                    interacting = true;
                }
            }
            // First frame after a load: run the callback to build + frame the model.
            if inner.pending.is_some() || inner.needs_framing {
                interacting = true;
            }
        }

        let inner = Arc::clone(&self.inner);
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                inner.lock().unwrap().render(painter.gl().clone(), &info);
            })),
        };
        ui.painter().add(callback);

        // The model is static, so only keep repainting while interacting (or
        // settling a freshly loaded model).
        if interacting {
            ui.ctx().request_repaint();
        }
    }
}

impl Inner {
    fn render(&mut self, gl: Arc<glow::Context>, info: &egui::PaintCallbackInfo) {
        // Lazily build the three-d context + lights from the live glow context.
        if self.gpu.is_none() {
            let ctx = match Context::from_gl_context(gl.clone()) {
                Ok(c) => c,
                Err(_) => return,
            };
            let ambient = AmbientLight::new(&ctx, 0.5, Srgba::WHITE);
            let key = DirectionalLight::new(&ctx, 2.4, Srgba::WHITE, vec3(-0.5, -1.0, -0.7));
            let fill = DirectionalLight::new(&ctx, 1.0, Srgba::WHITE, vec3(0.6, -0.3, 0.5));
            self.gpu = Some(Gpu { ctx, gl, model: None, ambient, key, fill });
        }
        let gpu = self.gpu.as_mut().unwrap();

        // Parse + upload any pending GLB (PBR materials are applied automatically).
        if let Some(bytes) = self.pending.take() {
            match three_d_asset::io::deserialize::<CpuModel>("model.glb", bytes)
                .map_err(|e| e.to_string())
                .and_then(|cpu| Model::<PhysicalMaterial>::new(&gpu.ctx, &cpu).map_err(|e| e.to_string()))
            {
                Ok(model) => gpu.model = Some(model),
                Err(e) => {
                    eprintln!("[scene3d] GLB load failed: {e}");
                    gpu.model = None;
                }
            }
        }

        // Frame the camera to the model bounds, once.
        if self.needs_framing {
            if let Some(model) = &gpu.model {
                let mut aabb = AxisAlignedBoundingBox::EMPTY;
                for part in model {
                    aabb.expand_with_aabb(part.aabb());
                }
                if !aabb.is_empty() {
                    self.target = aabb.center();
                    let radius = (aabb.size().magnitude() * 0.5).max(1.0e-3);
                    // Pull back so the bounding sphere fits a 45° vertical FOV, with margin.
                    self.distance = radius / (std::f32::consts::FRAC_PI_8).tan() * 1.1;
                }
                self.needs_framing = false;
            }
        }

        // egui gives a top-left rect; glow/three-d want bottom-left origin, which
        // `from_bottom_px` already provides.
        let vp = info.viewport_in_pixels();
        let viewport = Viewport {
            x: vp.left_px,
            y: vp.from_bottom_px,
            width: vp.width_px.max(0) as u32,
            height: vp.height_px.max(0) as u32,
        };
        if viewport.width == 0 || viewport.height == 0 {
            return;
        }

        // Orbit camera from spherical coords (three-d's Camera has no setters, so
        // it's rebuilt each frame — also needed because the viewport changes).
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let eye = vec3(
            self.target.x + self.distance * cp * sy,
            self.target.y + self.distance * sp,
            self.target.z + self.distance * cp * cy,
        );
        let z_near = (self.distance * 0.01).max(1.0e-3);
        let z_far = self.distance * 10.0 + 100.0;
        let camera = Camera::new_perspective(
            viewport,
            eye,
            self.target,
            vec3(0.0, 1.0, 0.0),
            degrees(45.0),
            z_near,
            z_far,
        );

        // Render only our panel region — never wipe the rest of egui's framebuffer.
        let [sw, sh] = info.screen_size_px;
        let sb = ScissorBox { x: viewport.x, y: viewport.y, width: viewport.width, height: viewport.height };
        let [r, g, b, a] = self.bg;
        let screen = RenderTarget::screen(&gpu.ctx, sw, sh);
        screen.clear_partially(sb, ClearState::color_and_depth(r, g, b, a, 1.0));
        if let Some(model) = &gpu.model {
            screen.render_partially(sb, &camera, model, &[&gpu.ambient, &gpu.key, &gpu.fill]);
        }

        // Defensive: clear state three-d may leave set (scissor/program/VAO) so it
        // can't disturb egui's subsequent draws.
        use glow::HasContext as _;
        unsafe {
            gpu.gl.disable(glow::SCISSOR_TEST);
            gpu.gl.bind_vertex_array(None);
            gpu.gl.use_program(None);
        }
    }
}

/// Centred muted message shown when there's no model (or a load error).
fn placeholder(ui: &mut egui::Ui, msg: &str) {
    ui.centered_and_justified(|ui| {
        ui.label(egui::RichText::new(msg).size(15.0).color(MUTED()));
    });
}

/// Viewport clear colour, matched loosely to the active theme.
fn clear_color() -> [f32; 4] {
    if crate::theme::is_light() {
        [0.93, 0.93, 0.95, 1.0]
    } else {
        [0.07, 0.07, 0.09, 1.0]
    }
}
