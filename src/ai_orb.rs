//! A small "living" AI presence — a sphere of glowing particles that gently
//! breathes when idle and accelerates / brightens / ripples when the assistant
//! is thinking or talking.
//!
//! This is a Rust/egui port of terminus2's `AiOrb.java`, upgraded from the flat
//! ring to a **3D particle sphere**: points are distributed over a unit sphere
//! (Fibonacci lattice), rotated each frame, projected with perspective, and
//! drawn back-to-front so nearer particles are larger and brighter. The same
//! IDLE / THINKING / TALKING energy model drives the motion and glow.
//!
//! egui is immediate-mode, so there is no shared ticker as in Swing: each
//! [`AiOrb::show`] call advances the animation by the real frame delta (scaled
//! to a 60 fps feel) and requests a repaint while the orb is visible. Orbs that
//! are scrolled off-screen stop animating, so many on screen stay cheap.

use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Ui, Vec2};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OrbState {
    Idle,
    Thinking,
    /// Ported from the Java API for parity; not yet driven by any UI.
    #[allow(dead_code)]
    Talking,
}

pub struct AiOrb {
    state: OrbState,

    // Animation accumulators, advanced per-frame in `show`.
    phase: f32,  // sphere rotation about the vertical axis
    breath: f32, // breathing / shimmer input
    energy: f32, // 0..1, eased toward the state's target for smooth transitions
    burst: f32,  // 0..1, click easter-egg; decays each frame

    /// Base particle positions on the unit sphere (Fibonacci lattice).
    particles: Vec<[f32; 3]>,
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
            // y walks linearly from +1 (top) to -1 (bottom); the ring radius at
            // each height is sqrt(1 - y^2), so points hug the sphere surface.
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
            particles,
        }
    }

    pub fn set_state(&mut self, s: OrbState) {
        self.state = s;
    }

    /// Convenience: working => THINKING, otherwise IDLE.
    #[allow(dead_code)]
    pub fn set_active(&mut self, working: bool) {
        self.state = if working { OrbState::Thinking } else { OrbState::Idle };
    }

    /// Allocate a `size`×`size` square at the cursor, advance the animation by
    /// the current frame delta, and paint the orb. `color` overrides the theme
    /// accent when `Some`.
    pub fn show(&mut self, ui: &mut Ui, size: f32, color: Option<Color32>) -> egui::Response {
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
        if !ui.is_rect_visible(rect) {
            return resp; // off-screen: skip work and let it idle (no repaint)
        }

        // Easter egg: clicking the orb sets off a burst — it expands, whitens,
        // spins up, and rings out a shockwave before settling back.
        if resp.clicked() {
            self.burst = 1.0;
        }

        // Advance using the real frame delta scaled to a 60 fps feel, so the
        // motion matches the original Swing ticker regardless of frame rate.
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let f = dt * 60.0;
        let target = match self.state {
            OrbState::Idle => 0.0,
            OrbState::Thinking => 0.7,
            OrbState::Talking => 1.0,
        };
        self.energy += (target - self.energy) * (0.08 * f).min(1.0);
        self.burst = (self.burst - dt * 1.8).max(0.0); // ~0.55s decay
        // Spin faster while active, and kick harder mid-burst.
        self.phase += (0.010 + self.energy * 0.085 + self.burst * 0.30) * f;
        self.breath += (0.040 + self.energy * 0.050) * f;

        self.paint(ui, rect, color);

        // Keep animating while visible — the orb breathes even when idle.
        ui.ctx().request_repaint();
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    fn paint(&self, ui: &Ui, rect: Rect, color: Option<Color32>) {
        let painter = ui.painter_at(rect);
        let center = rect.center();
        let max_r = (rect.width().min(rect.height()) / 2.0) - 1.0;
        if max_r <= 1.0 {
            return;
        }

        // Default to the theme accent — Aurora tints it pink so the orb matches.
        let base = color.unwrap_or_else(|| crate::theme::icon_tint(crate::theme::ACCENT1()));
        // On dark themes near particles brighten toward white (the classic glow);
        // on light themes that would vanish on the white panel, so they deepen
        // toward a dark version of the base colour instead, and alpha is boosted
        // so the orb stays clearly visible.
        let light = crate::theme::is_light();
        let alpha_scale = if light { 1.7 } else { 1.0 };
        // The colour each particle blends toward as `t -> 1`.
        let tip = if light {
            // A darker, saturated version of the base.
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

        let boost = self.burst; // 0..1, click easter-egg

        // Breathing sphere radius: subtle pulse when idle, larger when active,
        // and it swells on a click burst.
        let breath_amp = 0.045 + self.energy * 0.11;
        let ring_r = max_r * (0.60 + self.breath.sin() * breath_amp) * (1.0 + 0.45 * boost);

        // Soft blue halo behind everything (grows + brightens with energy/burst).
        let halo_r = ring_r * (1.0 + 0.25 * self.energy + 0.20 * boost);
        painter.circle_filled(center, halo_r, mix(0.0, 14.0 + 26.0 * self.energy + 70.0 * boost));

        // --- 3D transform: rotate about Y by `phase`, tilt about X so we see
        //     the sphere from slightly above; the tilt wobbles a touch with
        //     energy for a livelier feel. ---
        let (sy, cy) = self.phase.sin_cos();
        let tilt = 0.45 + (self.breath * 0.5).sin() * 0.08 * (0.3 + self.energy);
        let (st, ct) = tilt.sin_cos();

        // Perspective: camera sits `focal` sphere-radii away on +Z. dot_base is
        // the on-screen particle radius before the per-point perspective scale.
        let focal = 2.4;
        let dot_base = (max_r * (0.075 + self.energy * 0.03)).max(0.8) * (1.0 + 0.5 * boost);
        let n = self.particles.len() as f32;

        // Project every particle, keeping depth for back-to-front sorting.
        // Tuple: (depth_z, screen_pos, radius, alpha).
        let mut pts: Vec<(f32, Pos2, f32, f32)> = Vec::with_capacity(self.particles.len());
        for (i, p) in self.particles.iter().enumerate() {
            let [x0, y0, z0] = *p;

            // Radial ripple that only appears with energy (thinking/talking).
            let a = self.phase + std::f32::consts::TAU * i as f32 / n;
            let scale = 1.0 + (a * 3.0 + self.breath * 2.0).sin() * 0.07 * self.energy;

            // Rotate about Y, then tilt about X.
            let x1 = x0 * cy + z0 * sy;
            let z1 = -x0 * sy + z0 * cy;
            let y2 = y0 * ct - z1 * st;
            let z2 = y0 * st + z1 * ct;
            let x2 = x1;

            // Perspective divide (z2*scale stays within ±~1.05, so the
            // denominator never approaches zero for focal = 2.4).
            let persp = focal / (focal - z2 * scale);
            let pos = Pos2::new(
                center.x + x2 * scale * ring_r * persp,
                center.y + y2 * scale * ring_r * persp,
            );

            // Depth shading: 0 at the back, 1 at the front.
            let depth = (z2 + 1.0) * 0.5;
            let shimmer = 0.55 + 0.45 * (a * 2.0 + self.breath).sin();
            let depth_fade = 0.35 + 0.65 * depth;
            let alpha = (110.0 + 140.0 * self.energy + 120.0 * boost) * shimmer * depth_fade;
            pts.push((z2, pos, dot_base * persp, alpha));
        }

        // Draw far particles first so nearer ones overdraw them.
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (z, pos, radius, alpha) in pts {
            // Near particles read white, far ones blue; a burst whitens all.
            let depth = (z + 1.0) * 0.5;
            let white_t = (depth * 0.85 + boost * 0.6).clamp(0.0, 1.0);
            painter.circle_filled(pos, radius, mix(white_t, alpha));
        }

        // Glowing core (pulses brighter as it "talks"; whiter on a burst).
        let core_r = max_r * (0.10 + self.energy * 0.14 + 0.10 * boost) * (0.85 + 0.15 * self.breath.sin());
        painter.circle_filled(center, core_r, mix(0.25 + 0.6 * boost, 45.0 + 90.0 * self.energy + 120.0 * boost));

        // Click easter-egg: an expanding shockwave ring that fades as it grows.
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
