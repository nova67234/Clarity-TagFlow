//! A small "living" AI presence — a sphere of glowing particles that gently
//! breathes when idle and accelerates / brightens / ripples when the assistant
//! is thinking or talking.
//!
//! This is a Rust/egui port of terminus2's `AiOrb.java`, upgraded from the flat
//! ring to a **3D particle sphere**: points are distributed over a unit sphere
//! (Fibonacci lattice), rotated each frame, projected with perspective, and
//! drawn back-to-front so nearer particles are larger and brighter.
//!
//! On top of the IDLE / THINKING / TALKING energy model it adds three reactions:
//!   • **Error** — the orb flushes red, flashes on impact, and its particles blow
//!     apart (each gets an outward velocity); they drift and hang scattered.
//!   • **Recovery** — leaving the error state springs every particle back to its
//!     place, re-forming the orb.
//!   • **Long thinking** — after a few seconds of work the sphere slowly morphs
//!     through other shapes (cube → torus → helix → …) so a long wait still feels
//!     alive, then settles back to the sphere when the work finishes.
//!
//! egui is immediate-mode, so each [`AiOrb::show`] call advances the animation by
//! the real frame delta (scaled to a 60 fps feel) and repaints while visible.

use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Ui, Vec2};

/// Distinct target shapes the orb morphs between during a long wait.
const NUM_SHAPES: usize = 4;
/// Seconds of continuous thinking before the shape-morphing kicks in.
const MORPH_START: f32 = 3.5;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OrbState {
    Idle,
    Thinking,
    /// Ported from the Java API for parity; not yet driven by any UI.
    #[allow(dead_code)]
    Talking,
    /// Something failed — the orb turns red and bursts apart until it leaves
    /// this state (e.g. on a retry), which re-forms it.
    Error,
}

pub struct AiOrb {
    state: OrbState,

    // Animation accumulators, advanced per-frame in `show`.
    phase: f32,  // sphere rotation about the vertical axis
    breath: f32, // breathing / shimmer input
    energy: f32, // 0..1, eased toward the state's target for smooth transitions
    burst: f32,  // 0..1, click easter-egg; decays each frame

    error: f32,      // 0..1, eased toward 1 in the Error state (redness + broken spring)
    impact: f32,     // 0..1, a quick flash/shock at the moment of failure; decays
    think_time: f32, // seconds spent thinking, drives the long-wait morph
    morph: f32,      // shape-morph phase; integer part = shape index, frac = blend

    /// Base particle positions on the unit sphere (Fibonacci lattice).
    particles: Vec<[f32; 3]>,
    /// Per-particle displacement from the base position (the explosion offset).
    disp: Vec<[f32; 3]>,
    /// Per-particle velocity, integrated each frame (spring + explosion).
    vel: Vec<[f32; 3]>,
}

impl Default for AiOrb {
    fn default() -> Self {
        Self::new(56)
    }
}

impl AiOrb {
    /// Build an orb with `particle_count` points spread evenly over a sphere.
    pub fn new(particle_count: usize) -> Self {
        let n = particle_count.max(8);
        let mut particles = Vec::with_capacity(n);
        // Golden-angle increment gives an even, swirl-free distribution.
        let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..n {
            let y = 1.0 - (i as f32 / (n - 1) as f32) * 2.0;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f32;
            particles.push([theta.cos() * r, y, theta.sin() * r]);
        }
        Self {
            state: OrbState::Idle,
            phase: 0.0,
            breath: 0.0,
            energy: 0.0,
            burst: 0.0,
            error: 0.0,
            impact: 0.0,
            think_time: 0.0,
            morph: 0.0,
            disp: vec![[0.0; 3]; n],
            vel: vec![[0.0; 3]; n],
            particles,
        }
    }

    /// Set the orb's state. Entering [`OrbState::Error`] detonates the particles
    /// once; leaving it lets them spring back.
    pub fn set_state(&mut self, s: OrbState) {
        if s == OrbState::Error && self.state != OrbState::Error {
            self.explode();
        }
        self.state = s;
    }

    /// Convenience: working => THINKING, otherwise IDLE.
    #[allow(dead_code)]
    pub fn set_active(&mut self, working: bool) {
        self.set_state(if working { OrbState::Thinking } else { OrbState::Idle });
    }

    /// Give every particle an outward velocity (plus jitter) and flag an impact
    /// flash — the visual "explosion".
    fn explode(&mut self) {
        self.impact = 1.0;
        self.think_time = 0.0;
        self.morph = 0.0;
        for i in 0..self.particles.len() {
            let p = self.particles[i];
            let j = i as u32;
            let jit = |k: u32| (hash01(j.wrapping_mul(13).wrapping_add(k)) - 0.5) * 0.7;
            let speed = 0.06 + hash01(j.wrapping_mul(7).wrapping_add(1)) * 0.10;
            self.vel[i] = [
                (p[0] + jit(1)) * speed,
                (p[1] + jit(2)) * speed,
                (p[2] + jit(3)) * speed,
            ];
        }
    }

    /// Allocate a `size`×`size` square at the cursor, advance the animation by the
    /// current frame delta, and paint the orb. `color` overrides the accent.
    pub fn show(&mut self, ui: &mut Ui, size: f32, color: Option<Color32>) -> egui::Response {
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
        if !ui.is_rect_visible(rect) {
            return resp; // off-screen: skip work and let it idle (no repaint)
        }

        // Easter egg: clicking the orb sets off a burst.
        if resp.clicked() {
            self.burst = 1.0;
        }

        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let f = dt * 60.0;
        let target = match self.state {
            OrbState::Idle => 0.0,
            OrbState::Thinking => 0.7,
            OrbState::Talking => 1.0,
            OrbState::Error => 0.3,
        };
        self.energy += (target - self.energy) * (0.08 * f).min(1.0);
        self.burst = (self.burst - dt * 1.8).max(0.0);

        // Redness eases in/out of the error state; the impact flash decays fast.
        let err_target = if self.state == OrbState::Error { 1.0 } else { 0.0 };
        self.error += (err_target - self.error) * (0.06 * f).min(1.0);
        self.impact = (self.impact - dt * 2.0).max(0.0);

        // Long-wait morph: accumulate thinking time, decay it otherwise.
        if self.state == OrbState::Thinking {
            self.think_time += dt;
        } else {
            self.think_time = (self.think_time - dt * 1.5).max(0.0);
        }
        let morphing = self.think_time > MORPH_START && self.error < 0.25;
        if morphing {
            self.morph += dt * 0.22; // ~one shape every 4.5s
        } else {
            self.morph = (self.morph - dt * 1.5).max(0.0); // settle back to the sphere
        }

        // Per-particle physics: a spring back to the base position (strong when
        // whole, near-zero while broken so the pieces hang scattered) plus the
        // explosion velocity, integrated with damping.
        let spring_k = 0.16 * (1.0 - self.error);
        let damp = 0.88_f32.powf(f);
        for i in 0..self.particles.len() {
            for ax in 0..3 {
                self.vel[i][ax] += -self.disp[i][ax] * spring_k * f;
                self.vel[i][ax] *= damp;
                self.disp[i][ax] += self.vel[i][ax] * f;
            }
        }

        // Spin faster while active; kick harder on a burst or an impact.
        self.phase += (0.010 + self.energy * 0.085 + self.burst * 0.30 + self.impact * 0.20) * f;
        self.breath += (0.040 + self.energy * 0.050) * f;

        self.paint(ui, rect, color);
        ui.ctx().request_repaint();
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    /// The particle's target position this frame: its sphere point morphed toward
    /// the current long-wait shape (identity when `morph` is ~0).
    fn morphed_base(&self, i: usize) -> [f32; 3] {
        let sphere = self.particles[i];
        if self.morph < 1e-3 {
            return sphere;
        }
        let n = self.particles.len();
        let cur = self.morph.floor() as usize;
        let frac = smoothstep(self.morph.fract());
        let a = shape_pos(cur % NUM_SHAPES, i, n, sphere);
        let b = shape_pos((cur + 1) % NUM_SHAPES, i, n, sphere);
        lerp3(a, b, frac)
    }

    fn paint(&self, ui: &Ui, rect: Rect, color: Option<Color32>) {
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let max_r = (rect.width().min(rect.height()) / 2.0) - 1.0;
        if max_r <= 1.0 {
            return;
        }

        // Accent normally; blends to red as the error grows.
        let accent = color.unwrap_or_else(|| crate::theme::icon_tint(crate::theme::ACCENT1()));
        let base = lerp_color(accent, Color32::from_rgb(235, 55, 48), self.error);

        let light = crate::theme::is_light();
        let alpha_scale = if light { 1.7 } else { 1.0 };
        let tip = if light {
            Color32::from_rgb(
                (base.r() as f32 * 0.45) as u8,
                (base.g() as f32 * 0.45) as u8,
                (base.b() as f32 * 0.45) as u8,
            )
        } else {
            Color32::WHITE
        };
        let mix = |t: f32, a: f32| -> Color32 {
            let t = t.clamp(0.0, 1.0);
            let lerp = |c: u8, to: u8| (c as f32 + (to as f32 - c as f32) * t) as u8;
            let a = (a * alpha_scale).clamp(0.0, 255.0) as u8;
            Color32::from_rgba_unmultiplied(
                lerp(base.r(), tip.r()),
                lerp(base.g(), tip.g()),
                lerp(base.b(), tip.b()),
                a,
            )
        };

        let boost = self.burst;
        let flash = self.impact;

        // Breathing sphere radius; swells on a burst or an impact.
        let breath_amp = 0.045 + self.energy * 0.11;
        let ring_r = max_r * (0.60 + self.breath.sin() * breath_amp) * (1.0 + 0.45 * boost + 0.18 * flash);

        // Soft halo behind everything (brighter, redder on impact).
        let halo_r = ring_r * (1.0 + 0.25 * self.energy + 0.20 * boost + 0.45 * flash);
        painter.circle_filled(
            center,
            halo_r,
            mix(0.0, 14.0 + 26.0 * self.energy + 70.0 * boost + 95.0 * flash),
        );

        // 3D transform: rotate about Y by `phase`, tilt about X.
        let (sy, cy) = self.phase.sin_cos();
        let tilt = 0.45 + (self.breath * 0.5).sin() * 0.08 * (0.3 + self.energy);
        let (st, ct) = tilt.sin_cos();

        let focal = 2.4;
        let dot_base = (max_r * (0.075 + self.energy * 0.03)).max(0.8) * (1.0 + 0.5 * boost);
        let n = self.particles.len() as f32;

        let mut pts: Vec<(f32, Pos2, f32, f32)> = Vec::with_capacity(self.particles.len());
        for i in 0..self.particles.len() {
            let mb = self.morphed_base(i);
            let d = self.disp[i];
            let (x0, y0, z0) = (mb[0] + d[0], mb[1] + d[1], mb[2] + d[2]);

            // Radial ripple that only appears with energy (thinking/talking).
            let a = self.phase + std::f32::consts::TAU * i as f32 / n;
            let scale = 1.0 + (a * 3.0 + self.breath * 2.0).sin() * 0.07 * self.energy;

            // Rotate about Y, then tilt about X.
            let x1 = x0 * cy + z0 * sy;
            let z1 = -x0 * sy + z0 * cy;
            let y2 = y0 * ct - z1 * st;
            let z2 = y0 * st + z1 * ct;
            let x2 = x1;

            // Perspective divide (clamped so a far-flung exploded particle can't
            // cross the focal plane and flip).
            let denom = (focal - z2 * scale).max(0.35);
            let persp = focal / denom;
            let pos = Pos2::new(
                center.x + x2 * scale * ring_r * persp,
                center.y + y2 * scale * ring_r * persp,
            );

            let depth = (z2 + 1.0) * 0.5;
            let shimmer = 0.55 + 0.45 * (a * 2.0 + self.breath).sin();
            let depth_fade = 0.35 + 0.65 * depth;
            let alpha = (110.0 + 140.0 * self.energy + 120.0 * boost + 60.0 * flash) * shimmer * depth_fade;
            pts.push((z2, pos, dot_base * persp, alpha));
        }

        // Draw far particles first so nearer ones overdraw them.
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (z, pos, radius, alpha) in pts {
            let depth = (z + 1.0) * 0.5;
            let white_t = (depth * 0.85 + boost * 0.6 + flash * 0.5).clamp(0.0, 1.0);
            painter.circle_filled(pos, radius, mix(white_t, alpha));
        }

        // Glowing core (fades out while the orb is blown apart).
        let core_t = (1.0 - self.error).max(0.0);
        let core_r = max_r
            * (0.10 + self.energy * 0.14 + 0.10 * boost)
            * (0.85 + 0.15 * self.breath.sin())
            * core_t;
        painter.circle_filled(
            center,
            core_r,
            mix(0.25 + 0.6 * boost, (45.0 + 90.0 * self.energy + 120.0 * boost) * core_t),
        );

        // Impact shockwave at the moment of failure.
        if flash > 0.01 {
            let wave_r = max_r * (0.40 + (1.0 - flash) * 1.45);
            painter.circle_stroke(
                center,
                wave_r,
                egui::Stroke::new(2.0 + 3.0 * flash, mix(0.8, 220.0 * flash)),
            );
        }

        // Click easter-egg shockwave.
        if boost > 0.01 {
            let wave_r = max_r * (0.55 + (1.0 - boost) * 1.15);
            painter.circle_stroke(
                center,
                wave_r,
                egui::Stroke::new(1.5 + 2.0 * boost, mix(0.5, 200.0 * boost)),
            );
        }
    }
}

/// A particle's position in shape `shape` (0 = sphere). Each maps the same index
/// to a point so the morph is a smooth per-particle interpolation.
fn shape_pos(shape: usize, i: usize, n: usize, sphere: [f32; 3]) -> [f32; 3] {
    match shape {
        // Cube: push the sphere point out to the cube's surface.
        1 => {
            let m = sphere[0].abs().max(sphere[1].abs()).max(sphere[2].abs()).max(1e-3);
            [sphere[0] / m, sphere[1] / m, sphere[2] / m]
        }
        // Torus.
        2 => {
            let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
            let ring = std::f32::consts::TAU * i as f32 / n.max(1) as f32;
            let tube = golden * i as f32;
            let (rr, tr) = (0.72, 0.32);
            let (sr, cr) = ring.sin_cos();
            let (stb, ctb) = tube.sin_cos();
            [(rr + tr * ctb) * cr, tr * stb, (rr + tr * ctb) * sr]
        }
        // Double helix-ish coil.
        3 => {
            let t = i as f32 / (n.max(2) - 1) as f32; // 0..1
            let ang = t * std::f32::consts::TAU * 3.5;
            let (sa, ca) = ang.sin_cos();
            let r = 0.55;
            [ca * r, (t - 0.5) * 1.9, sa * r]
        }
        _ => sphere,
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()))
}

/// Cheap deterministic hash → [0,1), for per-particle explosion jitter.
fn hash01(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    x = (x >> 22) ^ x;
    (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}
