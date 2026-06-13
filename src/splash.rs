//! Startup splash — "Clarity TagFlow" handwrites itself in a cursive script
//! over the theme background, holds for a beat, then fades into the app.
//!
//! The write-on is a left-to-right clip reveal of the laid-out cursive text
//! (which reads as handwriting because script glyphs connect), with a small
//! accent "pen" dot riding the reveal edge. Any click or key press skips it.
//! The cursive font family is registered in `install_fallback_fonts`.

use eframe::egui;

use crate::theme::{ACCENT1, BG, TEXT};

/// Seconds the text spends writing itself on.
const WRITE_SECS: f32 = 2.2;
/// Seconds the finished wordmark holds before fading.
const HOLD_SECS: f32 = 0.7;
/// Seconds the whole overlay takes to fade out.
const FADE_SECS: f32 = 0.6;

#[derive(Default)]
pub struct Splash {
    /// egui time when the splash first drew (the animation clock's zero).
    start: Option<f64>,
    done: bool,
}

impl Splash {
    /// True while the splash should be the ONLY thing rendered (the write-on
    /// and hold phases). The app skips building its panels entirely then, so
    /// launching shows the wordmark before the UI exists; the fade-out phase
    /// then plays over the real UI as it appears.
    pub fn covers_ui(&self, ctx: &egui::Context) -> bool {
        if self.done {
            return false;
        }
        match self.start {
            None => true, // first frame — the clock hasn't started yet
            Some(s) => ((ctx.input(|i| i.time) - s) as f32) < WRITE_SECS + HOLD_SECS,
        }
    }

    /// Draw the splash overlay. Call LAST in the frame so it covers the UI.
    pub fn show(&mut self, ctx: &egui::Context) {
        if self.done {
            return;
        }
        let now = ctx.input(|i| i.time);
        let start = *self.start.get_or_insert(now);
        let t = (now - start) as f32;

        // Any click or key press skips the intro.
        let skip = ctx.input(|i| {
            i.pointer.any_pressed()
                || i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. }))
        });
        if skip || t >= WRITE_SECS + HOLD_SECS + FADE_SECS {
            self.done = true;
            ctx.request_repaint();
            return;
        }

        let screen = ctx.content_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("startup_splash"),
        ));

        // Everything fades together at the end.
        let alpha = if t > WRITE_SECS + HOLD_SECS {
            (1.0 - (t - WRITE_SECS - HOLD_SECS) / FADE_SECS).clamp(0.0, 1.0)
        } else {
            1.0
        };

        painter.rect_filled(screen, 0, BG().gamma_multiply(alpha));

        // The wordmark, centred, in the cursive family (falls back to the
        // regular face when no script font is installed).
        let font = egui::FontId::new(64.0, egui::FontFamily::Name("cursive".into()));
        let color = TEXT().gamma_multiply(alpha);
        let galley = painter.layout_no_wrap("Clarity TagFlow".into(), font, color);
        let size = galley.size();
        let pos = screen.center() - size * 0.5;

        // Left-to-right write-on (smoothstepped, like a hand easing into and
        // out of a word). The clip reveal makes connected script glyphs appear
        // stroke by stroke.
        let wt = (t / WRITE_SECS).clamp(0.0, 1.0);
        let eased = wt * wt * (3.0 - 2.0 * wt);
        let reveal = size.x * eased;
        let clip = egui::Rect::from_min_size(pos, egui::vec2(reveal + 2.0, size.y));
        painter.with_clip_rect(clip).galley(pos, galley, color);

        // The "pen": an accent dot riding the reveal edge while writing.
        if wt < 1.0 {
            let tip = egui::pos2(pos.x + reveal, pos.y + size.y * 0.62);
            painter.circle_filled(tip, 3.5, ACCENT1().gamma_multiply(alpha));
        }

        ctx.request_repaint(); // keep the animation smooth
    }
}
