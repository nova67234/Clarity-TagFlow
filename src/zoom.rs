// Zoom + pan for the centre image viewer — a port of terminus2's ImageCanvas
// (ViewerPanel.java). The view keeps a `scale` (pixels per image pixel, so 1.0 is
// a true 1:1 pixel view), a `pan` offset, and a `fit_mode` flag. In fit mode the
// image is scaled to fill the viewport and re-fits on resize; once the user zooms
// or pans we leave fit mode and honour the explicit scale/offset.
//
// Controls (cross-platform — Windows, macOS, Linux):
//   • Ctrl + mouse wheel, or trackpad pinch   → zoom toward the cursor
//   • Double-click                            → toggle 100% ↔ fit
//   • Drag (when zoomed in)                   → pan
//   • Ctrl +/-                                → zoom in/out (toward centre)
//   • Ctrl 0                                  → fit to view
//   • Ctrl 1                                  → 100% (1:1)
//
// macOS reports Cmd as `command`; we accept it alongside Ctrl so the shortcuts
// feel native there.

use eframe::egui::{self, Color32, CornerRadius, Key, Pos2, Rect, Sense, Stroke, Vec2};
use std::path::{Path, PathBuf};

// Matches the Java ImageCanvas limits.
const MIN_SCALE: f32 = 0.05;
const MAX_SCALE: f32 = 12.0;
/// Per keystroke zoom factor (Java used 1.10 for Ctrl +/-).
const KEY_STEP: f32 = 1.10;
/// How long the "x%" badge stays up after a zoom change, in seconds.
const OVERLAY_SECS: f64 = 1.1;

/// What the viewer is asking the app to do after a frame, driven by the
/// right-click context menu / crop tool.
pub enum ViewerAction {
    None,
    /// Toggle the current image's favorite (heart) state.
    ToggleFavorite,
    /// Save a cropped copy of the current image (the original is kept). The rect
    /// is given as fractions (0..1) of the full image's width/height.
    Crop(CropFraction),
}

/// A crop region as fractions (0..1) of the source image's width/height.
pub struct CropFraction {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

pub struct ZoomState {
    /// Pixels per image pixel. 1.0 == 100% (1:1).
    scale: f32,
    /// Offset of the image centre from the viewport centre, in screen pixels.
    pan: Vec2,
    /// When set, the image is fit to the viewport and re-fits on every resize.
    fit_mode: bool,
    /// The path the current scale/pan applies to. Selecting a different image
    /// resets the view back to fit.
    current: Option<PathBuf>,
    /// Time (egui input clock) until which the zoom-percent badge is shown.
    overlay_until: f64,
    /// Percent value captured for the badge.
    overlay_percent: i32,
    /// Crop tool: true while the user is dragging out a crop region.
    crop_mode: bool,
    /// Anchor of the in-progress crop drag (screen coords).
    crop_start: Option<Pos2>,
    /// Current crop selection rectangle (screen coords).
    crop_rect: Option<Rect>,
}

/// The scale at which the image exactly fits inside the viewport (the larger of
/// the two axes touches the edge), clamped to the allowed zoom range.
fn fit_scale(viewport: Rect, img: Vec2) -> f32 {
    let s = (viewport.width() / img.x).min(viewport.height() / img.y);
    s.clamp(MIN_SCALE, MAX_SCALE)
}

/// Clamp a screen point into `rect`.
fn clamp_pos(p: Pos2, rect: Rect) -> Pos2 {
    Pos2::new(p.x.clamp(rect.min.x, rect.max.x), p.y.clamp(rect.min.y, rect.max.y))
}

impl Default for ZoomState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            pan: Vec2::ZERO,
            fit_mode: true,
            current: None,
            overlay_until: 0.0,
            overlay_percent: 100,
            crop_mode: false,
            crop_start: None,
            crop_rect: None,
        }
    }
}

impl ZoomState {
    /// Draw `tex` (the full-resolution image for `path`) into the viewer, handling
    /// zoom and pan. Replaces a plain fitted draw for static images. `is_favorite`
    /// labels the right-click menu's favorite entry. Returns any action the user
    /// requested (favorite / crop) for the app to carry out.
    #[must_use]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        tex: &egui::TextureHandle,
        path: &Path,
        is_favorite: bool,
    ) -> ViewerAction {
        // Reset to a fresh fit (and cancel any crop) whenever the selection changes.
        if self.current.as_deref() != Some(path) {
            self.current = Some(path.to_path_buf());
            self.scale = 1.0;
            self.pan = Vec2::ZERO;
            self.fit_mode = true;
            self.crop_mode = false;
            self.crop_start = None;
            self.crop_rect = None;
        }

        let viewport = ui.available_rect_before_wrap();
        let img = tex.size_vec2();
        if img.x < 1.0 || img.y < 1.0 || viewport.width() < 1.0 || viewport.height() < 1.0 {
            return ViewerAction::None;
        }

        // In fit mode the scale tracks the viewport (handles window resizes).
        let fit = fit_scale(viewport, img);
        if self.fit_mode {
            self.scale = fit;
            self.pan = Vec2::ZERO;
        }

        let resp = ui.interact(viewport, ui.id().with("zoom_viewer"), Sense::click_and_drag());
        let now = ui.input(|i| i.time);
        let pointer = resp.hover_pos();

        // --- Crop tool: drag out a selection, then hand the fractional rect back ---
        if self.crop_mode {
            let action = self.handle_crop(ui, &resp, viewport, img);
            self.paint_image(ui, viewport, img, tex);
            self.paint_crop_overlay(ui, viewport);
            return action;
        }

        // --- Ctrl + wheel / trackpad pinch: zoom toward the cursor ---
        // egui folds Ctrl+scroll and pinch gestures into zoom_delta(), so this
        // single path covers a mouse on Windows/Linux and a trackpad on macOS.
        let zoom_delta = ui.input(|i| i.zoom_delta());
        if resp.hovered() && (zoom_delta - 1.0).abs() > 1e-4 {
            let anchor = pointer.unwrap_or_else(|| viewport.center());
            self.zoom_at(viewport, img, self.scale * zoom_delta, anchor);
            self.flash(now);
        }

        // --- Keyboard shortcuts ---
        let (key_in, key_out, key_fit, key_one) = ui.input(|i| {
            let ctrl = i.modifiers.ctrl || i.modifiers.command;
            (
                ctrl && (i.key_pressed(Key::Plus) || i.key_pressed(Key::Equals)),
                ctrl && i.key_pressed(Key::Minus),
                ctrl && i.key_pressed(Key::Num0),
                ctrl && i.key_pressed(Key::Num1),
            )
        });
        if key_in {
            self.zoom_at(viewport, img, self.scale * KEY_STEP, viewport.center());
            self.flash(now);
        }
        if key_out {
            self.zoom_at(viewport, img, self.scale / KEY_STEP, viewport.center());
            self.flash(now);
        }
        if key_fit {
            self.fit_to_view();
            self.overlay_percent = (fit * 100.0).round() as i32;
            self.flash(now);
        }
        if key_one {
            self.zoom_at(viewport, img, 1.0, viewport.center());
            self.flash(now);
        }

        // --- Double-click toggles 100% (at the cursor) and fit ---
        if resp.double_clicked() {
            if (self.scale - 1.0).abs() < 0.01 {
                self.fit_to_view();
                self.overlay_percent = (fit * 100.0).round() as i32;
            } else {
                let anchor = pointer.unwrap_or_else(|| viewport.center());
                self.zoom_at(viewport, img, 1.0, anchor);
            }
            self.flash(now);
        }

        // --- Drag to pan (only meaningful once the image overflows the viewport) ---
        let pannable = !self.fit_mode
            && (img.x * self.scale > viewport.width() || img.y * self.scale > viewport.height());
        if pannable {
            if resp.dragged() {
                self.pan += resp.drag_delta();
                self.clamp_pan(viewport, img);
            }
            ui.ctx().set_cursor_icon(if resp.dragged() {
                egui::CursorIcon::Grabbing
            } else if resp.hovered() {
                egui::CursorIcon::Grab
            } else {
                egui::CursorIcon::Default
            });
        }

        // --- Right-click menu: favorite the image, or start a crop ---
        let mut want_favorite = false;
        let mut want_crop = false;
        resp.context_menu(|ui| {
            let fav_label = if is_favorite { "Remove favorite" } else { "Favorite" };
            if ui.button(fav_label).clicked() {
                want_favorite = true;
                ui.close();
            }
            if ui.button("Crop image…").clicked() {
                want_crop = true;
                ui.close();
            }
        });
        let mut action = ViewerAction::None;
        if want_favorite {
            action = ViewerAction::ToggleFavorite;
        } else if want_crop {
            // Enter crop mode; the actual crop is emitted once the drag finishes.
            self.crop_mode = true;
            self.crop_start = None;
            self.crop_rect = None;
        }

        // --- Paint ---
        self.paint_image(ui, viewport, img, tex);

        // --- Zoom-percent badge (fades after a moment) ---
        if now < self.overlay_until {
            self.paint_overlay(ui, viewport);
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
        }

        action
    }

    /// Paint the image: one rounded, textured rect at the current scale/pan,
    /// clipped to the viewport. When zoomed in the image overflows and its rounded
    /// corners fall outside the clip, so the visible edges read as straight; when
    /// fit or zoomed out the whole image shows with rounded corners.
    fn paint_image(&self, ui: &egui::Ui, viewport: Rect, img: Vec2, tex: &egui::TextureHandle) {
        let rect = self.image_rect(viewport, img);
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let shape = egui::epaint::RectShape::filled(rect, CornerRadius::same(22), Color32::WHITE)
            .with_texture(tex.id(), uv);
        ui.painter().with_clip_rect(viewport).add(shape);
    }

    /// Crop-mode interaction: drag to select a region (clamped to the on-screen
    /// image), Esc or right-click cancels. On release, returns the selection as a
    /// fraction (0..1) of the full image so the app can crop the original.
    fn handle_crop(
        &mut self,
        ui: &egui::Ui,
        resp: &egui::Response,
        viewport: Rect,
        img: Vec2,
    ) -> ViewerAction {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);

        // Esc or a right-click cancels the crop.
        if ui.input(|i| i.key_pressed(Key::Escape)) || resp.secondary_clicked() {
            self.crop_mode = false;
            self.crop_start = None;
            self.crop_rect = None;
            return ViewerAction::None;
        }

        // The on-screen image rectangle, clamped to what's actually visible.
        let irect = self.image_rect(viewport, img).intersect(viewport);

        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                let c = clamp_pos(p, irect);
                self.crop_start = Some(c);
                self.crop_rect = Some(Rect::from_min_max(c, c));
            }
        }
        if resp.dragged() {
            if let (Some(start), Some(p)) = (self.crop_start, resp.interact_pointer_pos()) {
                let c = clamp_pos(p, irect);
                self.crop_rect = Some(Rect::from_two_pos(start, c));
            }
        }
        if resp.drag_stopped() {
            let result = self.crop_rect.and_then(|cr| {
                if cr.width() < 4.0 || cr.height() < 4.0 || irect.width() <= 0.0 || irect.height() <= 0.0 {
                    return None;
                }
                let x = ((cr.min.x - irect.min.x) / irect.width()).clamp(0.0, 1.0);
                let y = ((cr.min.y - irect.min.y) / irect.height()).clamp(0.0, 1.0);
                let w = (cr.width() / irect.width()).clamp(0.0, 1.0 - x);
                let h = (cr.height() / irect.height()).clamp(0.0, 1.0 - y);
                (w > 0.0 && h > 0.0).then_some(CropFraction { x, y, w, h })
            });
            self.crop_mode = false;
            self.crop_start = None;
            self.crop_rect = None;
            if let Some(frac) = result {
                return ViewerAction::Crop(frac);
            }
        }
        ViewerAction::None
    }

    /// Dim everything outside the crop selection and outline it; show a hint until
    /// the user starts dragging.
    fn paint_crop_overlay(&self, ui: &egui::Ui, viewport: Rect) {
        let painter = ui.painter().with_clip_rect(viewport);
        let dim = Color32::from_black_alpha(120);

        match self.crop_rect {
            Some(cr) if cr.width() >= 1.0 && cr.height() >= 1.0 => {
                // Four rects around the selection, leaving the selection bright.
                let top = Rect::from_min_max(viewport.min, Pos2::new(viewport.max.x, cr.min.y));
                let bottom = Rect::from_min_max(Pos2::new(viewport.min.x, cr.max.y), viewport.max);
                let left = Rect::from_min_max(Pos2::new(viewport.min.x, cr.min.y), Pos2::new(cr.min.x, cr.max.y));
                let right = Rect::from_min_max(Pos2::new(cr.max.x, cr.min.y), Pos2::new(viewport.max.x, cr.max.y));
                for r in [top, bottom, left, right] {
                    painter.rect_filled(r, CornerRadius::ZERO, dim);
                }
                painter.rect_stroke(
                    cr,
                    CornerRadius::ZERO,
                    Stroke::new(1.5, Color32::from_gray(240)),
                    egui::StrokeKind::Inside,
                );
            }
            _ => {
                // No selection yet — dim the whole viewport and prompt.
                painter.rect_filled(viewport, CornerRadius::same(22), dim);
            }
        }

        // Hint banner at the top of the viewer.
        let hint = "Drag to select a crop area  •  Esc or right-click to cancel";
        let font = egui::FontId::proportional(13.0);
        let galley = painter.layout_no_wrap(hint.to_owned(), font, Color32::from_gray(235));
        let pad = egui::vec2(12.0, 7.0);
        let center = Pos2::new(viewport.center().x, viewport.min.y + galley.size().y * 0.5 + 16.0);
        let bg = Rect::from_center_size(center, galley.size() + pad * 2.0);
        painter.rect_filled(bg, CornerRadius::same(8), Color32::from_black_alpha(180));
        painter.galley(bg.min + pad, galley, Color32::from_gray(235));
    }

    /// The on-screen rectangle the image occupies for the current scale/pan.
    fn image_rect(&self, viewport: Rect, img: Vec2) -> Rect {
        Rect::from_center_size(viewport.center() + self.pan, img * self.scale)
    }

    /// Set a new scale while keeping the image point under `anchor` fixed on screen
    /// (the classic zoom-toward-cursor behaviour). Mirrors setScaleAtMouse().
    fn zoom_at(&mut self, viewport: Rect, img: Vec2, new_scale: f32, anchor: Pos2) {
        let new_scale = new_scale.clamp(MIN_SCALE, MAX_SCALE);
        let rect0 = self.image_rect(viewport, img);
        // Image-space coordinate (in image pixels) currently under the anchor.
        let img_pt = (anchor - rect0.min) / self.scale;
        self.fit_mode = false;
        self.scale = new_scale;
        // Place the image so that img_pt lands back under the anchor.
        let new_min = anchor - img_pt * self.scale;
        let new_center = new_min + (img * self.scale) * 0.5;
        self.pan = new_center - viewport.center();
        self.clamp_pan(viewport, img);
    }

    fn fit_to_view(&mut self) {
        self.fit_mode = true;
        self.pan = Vec2::ZERO;
        // scale is recomputed from the viewport on the next show().
    }

    /// Keep the image from being dragged past its edges. Centres on any axis where
    /// the scaled image is smaller than the viewport. Mirrors clampPan().
    fn clamp_pan(&mut self, viewport: Rect, img: Vec2) {
        let scaled = img * self.scale;
        let vp = viewport.size();
        if scaled.x <= vp.x {
            self.pan.x = 0.0;
        } else {
            let m = (scaled.x - vp.x) * 0.5;
            self.pan.x = self.pan.x.clamp(-m, m);
        }
        if scaled.y <= vp.y {
            self.pan.y = 0.0;
        } else {
            let m = (scaled.y - vp.y) * 0.5;
            self.pan.y = self.pan.y.clamp(-m, m);
        }
    }

    /// Capture the current zoom percent and show the badge for a short while.
    fn flash(&mut self, now: f64) {
        self.overlay_percent = (self.scale * 100.0).round() as i32;
        self.overlay_until = now + OVERLAY_SECS;
    }

    fn paint_overlay(&self, ui: &egui::Ui, viewport: Rect) {
        let text = format!("{}%", self.overlay_percent);
        let font = egui::FontId::proportional(13.0);
        let painter = ui.painter().with_clip_rect(viewport);
        let galley = painter.layout_no_wrap(text, font, Color32::from_gray(235));
        let pad = egui::vec2(10.0, 6.0);
        let size = galley.size() + pad * 2.0;
        // Bottom-centre of the viewport, a little above the edge.
        let center = Pos2::new(viewport.center().x, viewport.max.y - size.y * 0.5 - 14.0);
        let bg = Rect::from_center_size(center, size);
        painter.rect_filled(bg, CornerRadius::same(8), Color32::from_black_alpha(170));
        painter.galley(bg.min + pad, galley, Color32::from_gray(235));
    }
}
