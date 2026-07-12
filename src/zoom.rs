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
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

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
    /// Copy the current image's pixels to the system clipboard.
    CopyImage,
    /// Remove the background, saving a transparent-PNG cutout beside the original.
    RemoveBackground,
}

/// A crop region as fractions (0..1) of the source image's width/height.
pub struct CropFraction {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

// --- Spatial Scene (Apple-style depth parallax) ---------------------------

/// Model folder for the depth estimator (see the catalog in `ai_models.rs`).
const DEPTH_FOLDER: &str = "depth-anything-v2-base-onnx";
/// Shown when the depth model hasn't been downloaded yet.
const MODEL_MISSING_MSG: &str =
    "Depth model not installed — open the AI panel → Get Models → Depth (~27 MB).";
/// Max screen-space shift (px) of the most-displaced depth layer. Kept small so
/// the single-mesh warp reads as gentle depth rather than a stretchy rubber sheet
/// (this mesh warps the texture; it can't move rigid layers like Apple's does).
const PARALLAX_PX: f32 = 10.0;
/// Parallax mesh grid resolution (cells); vertices are `(GX+1)*(GY+1)`.
const GX: usize = 96;
const GY: usize = 64;
/// Box-blur passes over the coarse depth-layer grid. Spreads the displacement
/// gradient across many cells so the subject moves more as a unit and silhouettes
/// shear softly instead of stretching — the main lever against the "stretchy" look.
const BLUR_PASSES: usize = 6;

/// State for the Spatial Scene depth-parallax viewer mode. The depth map for the
/// current image is computed on a background thread, cached per path, and turned
/// into a per-vertex displaced mesh that follows the mouse (with an idle sway).
pub struct SpatialScene {
    /// Whether the parallax mode is currently engaged.
    pub active: bool,
    /// Per-image depth cache so re-entry / revisits are instant.
    cache: HashMap<PathBuf, Arc<crate::depth::DepthMap>>,
    /// In-flight depth job result for the current image.
    rx: Option<Receiver<Result<crate::depth::DepthMap, String>>>,
    /// Generation counter — bumped when the selection changes so a stale
    /// finishing thread is ignored (mirrors `right_details.rs`).
    generation: Arc<AtomicUsize>,
    /// Last depth/inference error, shown as a banner with a flat-image fallback.
    error: Option<String>,
    /// In-flight depth-model download (kicked on first use when the model isn't
    /// installed). While set, the viewer shows download progress instead of depth.
    download: Option<crate::ai_models::DownloadHandle>,
    /// Transient notice over the normal viewer (text, time-to-hide), e.g. as a
    /// fallback if the model can't be located or downloaded.
    notice: Option<(String, f64)>,
    /// Spinner shown while a depth map is being generated.
    orb: crate::ai_orb::AiOrb,
    /// Exp-smoothed parallax offset in [-1,1] per axis (mouse or idle sway).
    smoothed: Vec2,
    /// Cached per-vertex depth "layer" `(d-0.5)` for the current image, so the
    /// per-frame mesh rebuild doesn't re-sample the depth map 6k× a frame.
    layers: Vec<f32>,
    layers_key: Option<PathBuf>,
}

impl Default for SpatialScene {
    fn default() -> Self {
        Self {
            active: false,
            cache: HashMap::new(),
            rx: None,
            generation: Arc::new(AtomicUsize::new(0)),
            error: None,
            notice: None,
            orb: crate::ai_orb::AiOrb::default(),
            smoothed: Vec2::ZERO,
            download: None,
            layers: Vec::new(),
            layers_key: None,
        }
    }
}

/// A slow Lissajous drift used when the mouse is idle, so the scene keeps a
/// gentle "living photo" motion.
fn idle_sway(now: f64) -> Vec2 {
    let t = now as f32;
    Vec2::new((t * 0.45).sin() * 0.35, (t * 0.32).cos() * 0.25)
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
    /// Apple-style depth-parallax "Spatial Scene" mode.
    spatial: SpatialScene,
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
            spatial: SpatialScene::default(),
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
            // Spatial Scene: drop any in-flight depth job for the previous image
            // and clear its per-image render state. The mode itself persists; the
            // new image's depth is recomputed (or served instantly from cache).
            self.spatial.generation.fetch_add(1, Ordering::SeqCst);
            self.spatial.rx = None;
            self.spatial.error = None;
            self.spatial.layers.clear();
            self.spatial.layers_key = None;
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

        // --- Spatial Scene: depth-parallax mode takes over the viewer entirely
        //     (no zoom/pan/crop while engaged). ---
        if self.spatial.active {
            return self.show_spatial(ui, &resp, viewport, img, tex, path, is_favorite, now, pointer);
        }

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

        // --- Right-click menu: favorite the image, copy it, crop, or go spatial ---
        let mut want_favorite = false;
        let mut want_crop = false;
        let mut want_copy = false;
        let mut want_spatial = false;
        let mut want_bgremove = false;
        egui::Popup::context_menu(&resp)
            .frame(egui::Frame::menu(&resp.ctx.global_style()).corner_radius(CornerRadius::same(22)))
            .show(|ui| {
            let (fav_icon, fav_label) = if is_favorite {
                (egui::include_image!("../icons/heart_minus.svg"), "Remove favorite")
            } else {
                (egui::include_image!("../icons/heart_plus.svg"), "Favorite")
            };
            if menu_item(ui, fav_icon, fav_label) {
                want_favorite = true;
                ui.close();
            }
            if menu_item(ui, egui::include_image!("../icons/copy.svg"), "Copy image") {
                want_copy = true;
                ui.close();
            }
            if menu_item(ui, egui::include_image!("../icons/crop.svg"), "Crop image…") {
                want_crop = true;
                ui.close();
            }
            if menu_item(ui, egui::include_image!("../icons/background_remove.svg"), "Remove Background") {
                want_bgremove = true;
                ui.close();
            }
            if menu_item(ui, egui::include_image!("../icons/spatial_scene.svg"), "Spatial Scene") {
                want_spatial = true;
                ui.close();
            }
            // Region detection overlay (faces / hands / people / feet boxes). The
            // state is a process-wide singleton (src/detect.rs), so the toggle and
            // the per-image result cache are shared with the gallery-detail popup.
            let detect_label = if crate::detect::enabled() { "Hide Regions" } else { "Detect Regions" };
            if menu_item(ui, egui::include_image!("../icons/frame_inspect.svg"), detect_label) {
                crate::detect::toggle();
                ui.close();
            }
            // Age overlay — estimates an age for each detected face (InsightFace
            // genderage). Independent of the regions toggle.
            let age_label = if crate::detect::age_enabled() { "Hide Age" } else { "Detect Age" };
            if menu_item(ui, egui::include_image!("../icons/age.svg"), age_label) {
                crate::detect::toggle_age();
                ui.close();
            }
        });
        let mut action = ViewerAction::None;
        if want_favorite {
            action = ViewerAction::ToggleFavorite;
        } else if want_copy {
            action = ViewerAction::CopyImage;
        } else if want_bgremove {
            action = ViewerAction::RemoveBackground;
        } else if want_crop {
            // Enter crop mode; the actual crop is emitted once the drag finishes.
            self.crop_mode = true;
            self.crop_start = None;
            self.crop_rect = None;
        } else if want_spatial {
            self.enter_spatial(now);
        }

        // --- Paint ---
        self.paint_image(ui, viewport, img, tex);

        // --- Region-detection / age overlays (labelled boxes) ---
        if crate::detect::enabled() || crate::detect::age_enabled() {
            self.paint_detections(ui, viewport, img, path);
        }

        // --- Zoom-percent badge (fades after a moment) ---
        if now < self.overlay_until {
            self.paint_overlay(ui, viewport);
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
        }

        // --- Transient Spatial Scene notice (e.g. depth model not installed) ---
        if let Some((msg, until)) = self.spatial.notice.clone() {
            if now < until {
                self.paint_banner(ui, viewport, &msg);
                ui.ctx().request_repaint();
            } else {
                self.spatial.notice = None;
            }
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

    /// Paint the detection overlays — regions (faces / hands / people / feet)
    /// and/or per-face age labels (see `src/detect.rs`) — as labelled, coloured
    /// boxes mapped through the current zoom/pan so they track the image
    /// exactly. While models download or inference runs, small status chips
    /// stack at the top instead.
    fn paint_detections(&self, ui: &egui::Ui, viewport: Rect, img: Vec2, path: &Path) {
        let mut chip_y = viewport.min.y + 14.0;
        if crate::detect::enabled() {
            let (dets, status) = crate::detect::overlay(path, ui.ctx());
            self.paint_detection_layer(ui, viewport, img, dets, status, &mut chip_y);
        }
        if crate::detect::age_enabled() {
            let (dets, status) = crate::detect::age_overlay(path, ui.ctx());
            self.paint_detection_layer(ui, viewport, img, dets, status, &mut chip_y);
        }
    }

    /// Draw one overlay's boxes, or its status chip at `chip_y` (advanced so a
    /// second overlay's chip stacks below the first).
    fn paint_detection_layer(
        &self,
        ui: &egui::Ui,
        viewport: Rect,
        img: Vec2,
        dets: Option<std::sync::Arc<Vec<crate::detect::Detection>>>,
        status: Option<String>,
        chip_y: &mut f32,
    ) {
        let painter = ui.painter().with_clip_rect(viewport);

        if let Some(dets) = dets {
            let rect = self.image_rect(viewport, img);
            for d in dets.iter() {
                let r = Rect::from_min_max(
                    Pos2::new(
                        rect.min.x + d.rect[0] * rect.width(),
                        rect.min.y + d.rect[1] * rect.height(),
                    ),
                    Pos2::new(
                        rect.min.x + d.rect[2] * rect.width(),
                        rect.min.y + d.rect[3] * rect.height(),
                    ),
                );
                painter.rect_stroke(
                    r,
                    CornerRadius::same(3),
                    Stroke::new(2.0, d.color),
                    egui::StrokeKind::Outside,
                );
                // Label chip pinned to the box's top-left (falls inside the box
                // when the box touches the top of the viewport).
                let text = format!("{} {:.0}%", d.label, d.conf * 100.0);
                let font = egui::FontId::proportional(11.0);
                let galley = painter.layout_no_wrap(text, font, Color32::WHITE);
                let pad = Vec2::new(5.0, 2.0);
                let above = r.min.y - galley.size().y - pad.y * 2.0;
                let label_y = if above < viewport.min.y { r.min.y } else { above };
                let chip = Rect::from_min_size(
                    Pos2::new(r.min.x, label_y),
                    galley.size() + pad * 2.0,
                );
                painter.rect_filled(chip, CornerRadius::same(4), d.color.gamma_multiply(0.92));
                painter.galley(chip.min + pad, galley, Color32::WHITE);
            }
        } else if let Some(status) = status {
            // Status chip (downloading / detecting / error) at the top centre.
            let font = egui::FontId::proportional(12.5);
            let galley = painter.layout_no_wrap(status, font, Color32::from_gray(235));
            let pad = Vec2::new(10.0, 6.0);
            let size = galley.size() + pad * 2.0;
            let chip = Rect::from_min_size(
                Pos2::new(viewport.center().x - size.x * 0.5, *chip_y),
                size,
            );
            painter.rect_filled(chip, CornerRadius::same(10), Color32::from_black_alpha(160));
            painter.galley(chip.min + pad, galley, Color32::from_gray(235));
            *chip_y += size.y + 6.0;
        }
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

        if resp.drag_started()
            && let Some(p) = resp.interact_pointer_pos() {
                let c = clamp_pos(p, irect);
                self.crop_start = Some(c);
                self.crop_rect = Some(Rect::from_min_max(c, c));
            }
        if resp.dragged()
            && let (Some(start), Some(p)) = (self.crop_start, resp.interact_pointer_pos()) {
                let c = clamp_pos(p, irect);
                self.crop_rect = Some(Rect::from_two_pos(start, c));
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

    // --- Spatial Scene (depth parallax) ----------------------------------

    /// Engage Spatial Scene for the current image. If the depth model isn't
    /// installed yet, this auto-starts its download (shown with progress) so the
    /// user never hits a dead-end; the depth job kicks off automatically once the
    /// model lands. The depth job itself is started lazily by `show_spatial`.
    fn enter_spatial(&mut self, now: f64) {
        self.spatial.active = true;
        self.spatial.error = None;
        self.crop_mode = false;
        self.fit_to_view();

        if crate::tagger::resolve(DEPTH_FOLDER, "model.onnx").is_none() {
            self.spatial.download = crate::ai_models::start_model_download(DEPTH_FOLDER);
            if self.spatial.download.is_none() {
                // Catalog somehow lacks the entry — fall back to a notice.
                self.spatial.notice = Some((MODEL_MISSING_MSG.to_string(), now + 6.0));
                self.spatial.active = false;
            }
        }
    }

    /// Drive the Spatial Scene mode for one frame: handle exit, poll/kick the
    /// depth job, and paint either the parallax mesh (ready) or a flat image with
    /// a spinner/error banner. Always repaints so parallax + idle sway stay live.
    #[allow(clippy::too_many_arguments)]
    fn show_spatial(
        &mut self,
        ui: &mut egui::Ui,
        resp: &egui::Response,
        viewport: Rect,
        img: Vec2,
        tex: &egui::TextureHandle,
        path: &Path,
        is_favorite: bool,
        now: f64,
        pointer: Option<Pos2>,
    ) -> ViewerAction {
        // Esc leaves the mode.
        if ui.input(|i| i.key_pressed(Key::Escape)) {
            self.spatial.active = false;
            return ViewerAction::None;
        }

        // Right-click menu: exit, plus the usual favorite / copy.
        let mut want_exit = false;
        let mut want_favorite = false;
        let mut want_copy = false;
        egui::Popup::context_menu(resp)
            .frame(egui::Frame::menu(&resp.ctx.global_style()).corner_radius(CornerRadius::same(22)))
            .show(|ui| {
            if menu_item(ui, egui::include_image!("../icons/spatial_scene.svg"), "Exit Spatial Scene") {
                want_exit = true;
                ui.close();
            }
            let (fav_icon, fav_label) = if is_favorite {
                (egui::include_image!("../icons/heart_minus.svg"), "Remove favorite")
            } else {
                (egui::include_image!("../icons/heart_plus.svg"), "Favorite")
            };
            if menu_item(ui, fav_icon, fav_label) {
                want_favorite = true;
                ui.close();
            }
            if menu_item(ui, egui::include_image!("../icons/copy.svg"), "Copy image") {
                want_copy = true;
                ui.close();
            }
        });
        if want_exit {
            self.spatial.active = false;
            return ViewerAction::None;
        }

        // Keep animating for smooth parallax / idle sway / the loading spinner.
        ui.ctx().request_repaint();

        // --- Model download phase (first use): show progress until it lands. ---
        let dl = self.spatial.download.as_ref().map(|d| (d.done(), d.ok(), d.pct(), d.error()));
        if let Some((done, ok, pct, err)) = dl {
            if done {
                self.spatial.download = None;
                if !ok {
                    self.spatial.error =
                        Some(format!("model download failed — {}", err.unwrap_or_else(|| "unknown".into())));
                }
                // On success, fall through — the depth job starts below now that
                // the model is on disk.
            } else {
                self.paint_image(ui, viewport, img, tex);
                self.paint_download(ui, viewport, pct);
                return if want_favorite {
                    ViewerAction::ToggleFavorite
                } else if want_copy {
                    ViewerAction::CopyImage
                } else {
                    ViewerAction::None
                };
            }
        }

        // Poll the in-flight depth job.
        if let Some(rx) = &self.spatial.rx {
            match rx.try_recv() {
                Ok(Ok(map)) => {
                    self.spatial.cache.insert(path.to_path_buf(), Arc::new(map));
                    self.spatial.rx = None;
                }
                Ok(Err(e)) => {
                    self.spatial.error = Some(e);
                    self.spatial.rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.spatial.rx = None;
                    if self.spatial.error.is_none() {
                        self.spatial.error = Some("Depth job ended unexpectedly".to_string());
                    }
                }
            }
        }

        // Serve from cache, or start a job if we have neither a result nor an error.
        let depth = self.spatial.cache.get(path).cloned();
        if depth.is_none() && self.spatial.rx.is_none() && self.spatial.error.is_none() {
            self.spawn_depth_job(ui, path);
        }

        match depth {
            Some(map) => {
                let m = self.spatial_offset(ui, viewport, img, pointer, now);
                self.paint_spatial(ui, viewport, img, tex, map.as_ref(), m);
            }
            None => {
                // Still generating, or it failed — show the flat image underneath.
                self.paint_image(ui, viewport, img, tex);
                if let Some(err) = self.spatial.error.clone() {
                    self.paint_banner(ui, viewport, &format!("Depth unavailable — {err}"));
                } else {
                    self.paint_loading(ui, viewport);
                }
            }
        }

        if want_favorite {
            ViewerAction::ToggleFavorite
        } else if want_copy {
            ViewerAction::CopyImage
        } else {
            ViewerAction::None
        }
    }

    /// Spawn the background depth-estimation thread for `path` (generation-guarded
    /// so a stale finish is ignored). Mirrors the tag-job pattern.
    fn spawn_depth_job(&mut self, ui: &egui::Ui, path: &Path) {
        let Some(model) = crate::tagger::resolve(DEPTH_FOLDER, "model.onnx") else {
            // Model was removed after entering — back out with a notice.
            self.spatial.active = false;
            self.spatial.notice =
                Some((MODEL_MISSING_MSG.to_string(), ui.input(|i| i.time) + 6.0));
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        self.spatial.rx = Some(rx);
        let generation = self.spatial.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let generation_handle = Arc::clone(&self.spatial.generation);
        let img_path = path.to_path_buf();
        let ctx = ui.ctx().clone();
        std::thread::spawn(move || {
            if generation_handle.load(Ordering::SeqCst) != generation {
                return; // selection already moved on
            }
            let result = crate::depth::run_depth_job(model, img_path);
            if generation_handle.load(Ordering::SeqCst) == generation && tx.send(result).is_ok() {
                ctx.request_repaint();
            }
        });
    }

    /// Compute the smoothed parallax offset in [-1,1] per axis. While the mouse is
    /// over the image it follows the cursor (relative to the image centre); when
    /// the pointer leaves the image it eases into a gentle idle sway. (It does not
    /// drift while the cursor rests on the image — that read as buggy.)
    fn spatial_offset(
        &mut self,
        ui: &egui::Ui,
        viewport: Rect,
        img: Vec2,
        pointer: Option<Pos2>,
        now: f64,
    ) -> Vec2 {
        let rect = self.image_rect(viewport, img);
        let center = rect.center();
        let target = match pointer {
            Some(p) => Vec2::new(
                ((p.x - center.x) / (rect.width() * 0.5)).clamp(-1.0, 1.0),
                ((p.y - center.y) / (rect.height() * 0.5)).clamp(-1.0, 1.0),
            ),
            None => idle_sway(now),
        };

        // Frame-rate-aware exponential smoothing toward the target.
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let f = (dt * 60.0).min(1.0);
        self.spatial.smoothed += (target - self.spatial.smoothed) * (0.12 * f);
        self.spatial.smoothed
    }

    /// (Re)build the per-vertex depth-layer cache `(d - 0.5)` for the current
    /// image, pinning the outer border ring to 0 so the image edges stay anchored
    /// (no torn edges), then box-blurring the grid to soften silhouettes.
    fn rebuild_layers(&mut self, depth: &crate::depth::DepthMap) {
        let cols = GX + 1;
        let mut layers = vec![0.0f32; cols * (GY + 1)];
        for j in 0..=GY {
            let v = j as f32 / GY as f32;
            for i in 0..=GX {
                let u = i as f32 / GX as f32;
                layers[j * cols + i] = if i == 0 || i == GX || j == 0 || j == GY {
                    0.0
                } else {
                    depth.sample(u, v) - 0.5
                };
            }
        }
        // Soften depth transitions so hard silhouettes shear gently rather than
        // tearing. The border ring stays pinned (its neighbours pull toward 0).
        for _ in 0..BLUR_PASSES {
            let src = layers.clone();
            for j in 1..GY {
                for i in 1..GX {
                    let mut s = 0.0;
                    for dj in -1i32..=1 {
                        for di in -1i32..=1 {
                            let x = (i as i32 + di) as usize;
                            let y = (j as i32 + dj) as usize;
                            s += src[y * cols + x];
                        }
                    }
                    layers[j * cols + i] = s / 9.0;
                }
            }
        }
        self.spatial.layers = layers;
        self.spatial.layers_key = self.current.clone();
    }

    /// Paint the image as a depth-displaced parallax mesh. Near pixels (`layer`
    /// > 0) and far pixels (`layer` < 0) shift in opposite directions about the
    /// >    mid-plane, anchored so the scene doesn't drift, producing the "peek
    /// >    around" depth effect as `m` (the smoothed mouse/sway offset) changes.
    fn paint_spatial(
        &mut self,
        ui: &egui::Ui,
        viewport: Rect,
        img: Vec2,
        tex: &egui::TextureHandle,
        depth: &crate::depth::DepthMap,
        m: Vec2,
    ) {
        let cols = GX + 1;
        if self.spatial.layers_key.as_deref() != self.current.as_deref()
            || self.spatial.layers.len() != cols * (GY + 1)
        {
            self.rebuild_layers(depth);
        }

        let rect = self.image_rect(viewport, img);
        let mut mesh = egui::epaint::Mesh::with_texture(tex.id());
        for j in 0..=GY {
            let v = j as f32 / GY as f32;
            for i in 0..=GX {
                let u = i as f32 / GX as f32;
                let base = Pos2::new(rect.min.x + u * rect.width(), rect.min.y + v * rect.height());
                let layer = self.spatial.layers[j * cols + i];
                let disp = Vec2::new(
                    -m.x * layer * 2.0 * PARALLAX_PX,
                    -m.y * layer * 2.0 * PARALLAX_PX,
                );
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: base + disp,
                    uv: Pos2::new(u, v),
                    color: Color32::WHITE,
                });
            }
        }
        for j in 0..GY {
            for i in 0..GX {
                let a = (j * cols + i) as u32;
                let b = a + 1;
                let c = a + cols as u32;
                let d = c + 1;
                mesh.indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        ui.painter().with_clip_rect(viewport).add(egui::Shape::mesh(mesh));
    }

    /// Dim the viewport and show the breathing orb + "Generating depth…" caption
    /// while the depth map is computed.
    fn paint_loading(&mut self, ui: &mut egui::Ui, viewport: Rect) {
        ui.painter()
            .with_clip_rect(viewport)
            .rect_filled(viewport, CornerRadius::same(22), Color32::from_black_alpha(90));

        let orb_size = 72.0;
        let orb_rect = Rect::from_center_size(viewport.center(), Vec2::splat(orb_size));
        {
            let orb = &mut self.spatial.orb;
            orb.set_state(crate::ai_orb::OrbState::Thinking);
            ui.scope_builder(egui::UiBuilder::new().max_rect(orb_rect), |ui| {
                orb.show(ui, orb_size, None);
            });
        }

        let painter = ui.painter().with_clip_rect(viewport);
        let font = egui::FontId::proportional(13.0);
        let galley = painter.layout_no_wrap("Generating depth…".to_owned(), font, Color32::from_gray(235));
        let pad = egui::vec2(10.0, 6.0);
        let center = Pos2::new(viewport.center().x, viewport.center().y + orb_size * 0.5 + 18.0);
        let bg = Rect::from_center_size(center, galley.size() + pad * 2.0);
        painter.rect_filled(bg, CornerRadius::same(8), Color32::from_black_alpha(170));
        painter.galley(bg.min + pad, galley, Color32::from_gray(235));
    }

    /// While the depth model downloads on first use: dim the viewport, show the
    /// orb, a caption with the percentage, and a progress bar.
    fn paint_download(&mut self, ui: &mut egui::Ui, viewport: Rect, pct: u32) {
        ui.painter()
            .with_clip_rect(viewport)
            .rect_filled(viewport, CornerRadius::same(22), Color32::from_black_alpha(110));

        let orb_size = 72.0;
        let orb_center = Pos2::new(viewport.center().x, viewport.center().y - 18.0);
        let orb_rect = Rect::from_center_size(orb_center, Vec2::splat(orb_size));
        {
            let orb = &mut self.spatial.orb;
            orb.set_state(crate::ai_orb::OrbState::Thinking);
            ui.scope_builder(egui::UiBuilder::new().max_rect(orb_rect), |ui| {
                orb.show(ui, orb_size, None);
            });
        }

        let painter = ui.painter().with_clip_rect(viewport);
        let font = egui::FontId::proportional(13.0);
        let caption = format!("Downloading depth model…  {pct}%");
        let galley = painter.layout_no_wrap(caption, font, Color32::from_gray(235));
        let cap_y = orb_center.y + orb_size * 0.5 + 4.0;
        let pad = egui::vec2(10.0, 6.0);
        let bg = Rect::from_center_size(Pos2::new(viewport.center().x, cap_y), galley.size() + pad * 2.0);
        painter.rect_filled(bg, CornerRadius::same(8), Color32::from_black_alpha(170));
        painter.galley(bg.min + pad, galley, Color32::from_gray(235));

        // Progress bar under the caption.
        let (bar_w, bar_h) = (240.0, 6.0);
        let track = Rect::from_center_size(
            Pos2::new(viewport.center().x, bg.max.y + 12.0),
            Vec2::new(bar_w, bar_h),
        );
        painter.rect_filled(track, CornerRadius::same(3), Color32::from_white_alpha(38));
        let fill_w = bar_w * (pct as f32 / 100.0).clamp(0.0, 1.0);
        let fill = Rect::from_min_size(track.min, Vec2::new(fill_w, bar_h));
        painter.rect_filled(fill, CornerRadius::same(3), Color32::from_rgb(96, 165, 250));
    }

    /// A small top-of-viewport banner (shared by the model-missing notice and the
    /// depth-failure fallback).
    fn paint_banner(&self, ui: &egui::Ui, viewport: Rect, text: &str) {
        let painter = ui.painter().with_clip_rect(viewport);
        let font = egui::FontId::proportional(13.0);
        let galley = painter.layout_no_wrap(text.to_owned(), font, Color32::from_gray(235));
        let pad = egui::vec2(12.0, 7.0);
        let center = Pos2::new(viewport.center().x, viewport.min.y + galley.size().y * 0.5 + 16.0);
        let bg = Rect::from_center_size(center, galley.size() + pad * 2.0);
        painter.rect_filled(bg, CornerRadius::same(8), Color32::from_black_alpha(190));
        painter.galley(bg.min + pad, galley, Color32::from_gray(235));
    }
}

/// A right-click context-menu entry with a leading SVG icon (tinted to the menu's
/// text colour so it stays visible in light and dark themes). Returns true when
/// clicked.
fn menu_item(ui: &mut egui::Ui, icon: egui::ImageSource<'_>, label: &str) -> bool {
    let image = egui::Image::new(icon)
        .fit_to_exact_size(egui::vec2(16.0, 16.0))
        .tint(ui.visuals().text_color());
    ui.add(egui::Button::image_and_text(image, label)).clicked()
}
