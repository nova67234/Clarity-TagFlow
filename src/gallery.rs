//! Gallery view — a full-window masonry grid of the open folder's images, shown
//! instead of the three panels when the "Gallery" layout is picked in the
//! Appearance settings. Just the images themselves (no overlays or badges),
//! packed into balanced columns of varying heights. Clicking a tile selects that
//! image and switches back to the Panels layout so it opens in the viewer.
//!
//! Like the left browser, decoding is lazy: only tiles inside the viewport request
//! their thumbnail. Tile heights follow each image's aspect ratio once known, so
//! the layout settles as thumbnails load.

use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{CornerRadius, Margin, Stroke};

use crate::image_cache::{Cached, ImageCache};
use crate::theme::*;

/// Gap between tiles, both within a column and between columns.
const GAP: f32 = 12.0;
/// Rounded-corner radius of each tile.
const CORNER: u8 = 10;
/// Aspect (h/w) assumed for a tile whose thumbnail hasn't decoded yet.
const DEFAULT_ASPECT: f32 = 1.0;

/// Render the gallery into a central panel filling the area below the top bar.
/// Returns the index of a clicked image (to open in the detail popup), if any.
#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    images: &[PathBuf],
    filtered: &[usize],
    selected: Option<usize>,
    thumbs: &mut ImageCache,
    video_thumbs: &mut crate::video::VideoThumbs,
    favorites: &mut crate::favorites::Favorites,
    // Target tile/column width, driven by the Thumbnail-size setting.
    target_col_w: f32,
) -> Option<usize> {
    let mut clicked = None;
    egui::CentralPanel::default()
        .frame(
            egui::Frame::new()
                .fill(BG())
                .inner_margin(Margin { left: 12, right: 12, top: 4, bottom: 12 }),
        )
        .show_inside(ui, |ui| {
            if filtered.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Open a folder to see the gallery")
                            .size(18.0)
                            .color(MUTED()),
                    );
                });
                return;
            }

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(GAP, GAP);
                    let avail = ui.available_width();
                    let target = target_col_w.clamp(120.0, 400.0);
                    let ncols = (((avail + GAP) / (target + GAP)).floor() as usize).max(1);
                    let col_w = (avail - GAP * (ncols as f32 - 1.0)) / ncols as f32;

                    // Masonry: place each image into the currently-shortest column.
                    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); ncols];
                    let mut heights = vec![0.0f32; ncols];
                    for &i in filtered {
                        let a = aspect_of(thumbs, video_thumbs, &images[i]);
                        let c = shortest(&heights);
                        columns[c].push(i);
                        heights[c] += col_w * a + GAP;
                    }

                    ui.columns(ncols, |cols| {
                        for (ci, col_ui) in cols.iter_mut().enumerate() {
                            col_ui.spacing_mut().item_spacing.y = GAP;
                            for &i in &columns[ci] {
                                let path = &images[i];
                                let a = aspect_of(thumbs, video_thumbs, path);
                                let w = col_ui.available_width();
                                let (rect, resp) = col_ui
                                    .allocate_exact_size(egui::vec2(w, w * a), egui::Sense::click());

                                // Only decode/paint tiles actually in view (lazy).
                                if col_ui.is_rect_visible(rect) {
                                    let is_fav = favorites.is_favorite(path);
                                    tile(col_ui, thumbs, video_thumbs, path, rect, selected == Some(i), is_fav);
                                } else {
                                    col_ui
                                        .painter()
                                        .rect_filled(rect, CornerRadius::same(CORNER), FIELD());
                                }
                                if resp.hovered() {
                                    col_ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                                if resp.clicked() {
                                    // Open the detail popup for this image.
                                    clicked = Some(i);
                                }
                            }
                        }
                    });
                });
        });
    clicked
}

/// Aspect ratio (h/w) of an image's thumbnail, or a square default if unknown.
fn aspect_of(thumbs: &ImageCache, video_thumbs: &crate::video::VideoThumbs, path: &Path) -> f32 {
    if crate::is_video(path) {
        video_thumbs.aspect(path)
    } else {
        thumbs.aspect(path)
    }
    .unwrap_or(DEFAULT_ASPECT)
}

/// Index of the shortest column so far.
fn shortest(heights: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..heights.len() {
        if heights[i] < heights[best] {
            best = i;
        }
    }
    best
}

/// Paint one image tile (thumbnail, video poster, or a placeholder) into `rect`.
fn tile(
    ui: &mut egui::Ui,
    thumbs: &mut ImageCache,
    video_thumbs: &mut crate::video::VideoThumbs,
    path: &Path,
    rect: egui::Rect,
    selected: bool,
    is_fav: bool,
) {
    let radius = CornerRadius::same(CORNER);
    if crate::is_video(path) {
        match video_thumbs.request(path, ui.ctx()) {
            Some(tex) => {
                egui::Image::from_texture(&tex).corner_radius(radius).paint_at(ui, rect);
            }
            None => {
                ui.painter().rect_filled(rect, radius, FIELD());
                let s = (rect.width().min(rect.height()) * 0.35).clamp(24.0, 64.0);
                let icon = egui::Rect::from_center_size(rect.center(), egui::vec2(s, s));
                egui::Image::new(egui::include_image!("../icons/video.svg"))
                    .tint(MUTED())
                    .paint_at(ui, icon);
            }
        }
    } else {
        let now = ui.input(|i| i.time);
        match thumbs.request(path, now) {
            Cached::Ready(tex) | Cached::Animated(tex) => {
                egui::Image::from_texture(&tex).corner_radius(radius).paint_at(ui, rect);
            }
            Cached::Loading => {
                ui.painter().rect_filled(rect, radius, FIELD());
                let sp = egui::Rect::from_center_size(rect.center(), egui::vec2(26.0, 26.0));
                egui::Spinner::new().color(MUTED()).paint_at(ui, sp);
            }
            Cached::Failed => {
                ui.painter().rect_filled(rect, radius, FIELD());
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Failed",
                    egui::FontId::proportional(12.0),
                    MUTED(),
                );
            }
        }
    }
    // A heart in the top-right corner marks favorited files.
    if is_fav {
        heart_badge(ui, rect);
    }

    if selected {
        ui.painter().rect_stroke(
            rect,
            radius,
            Stroke::new(2.0, ACCENT1()),
            egui::StrokeKind::Inside,
        );
    }
}

/// Draw the favorite heart in a tile's top-right corner (the heart SVG's own pink,
/// no backdrop), with a gentle heartbeat pulse — matching the browser tiles.
fn heart_badge(ui: &egui::Ui, rect: egui::Rect) {
    let base = (rect.width().min(rect.height()) * 0.12).clamp(16.0, 26.0);
    let center = egui::pos2(rect.max.x - base, rect.min.y + base);

    // t in [0,1) over 2s; scale eases 1.0↔1.2 smoothly.
    let t = (ui.input(|i| i.time) % 2.0) / 2.0;
    let scale = 1.0 + 0.1 * (1.0 - (t * std::f64::consts::TAU).cos()) as f32;
    let size = base * scale;

    let badge = egui::Rect::from_center_size(center, egui::vec2(size, size));
    egui::Image::new(egui::include_image!("../icons/heart.svg")).paint_at(ui, badge);
    ui.ctx().request_repaint();
}
