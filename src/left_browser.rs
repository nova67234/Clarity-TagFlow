//! The left browser panel — a search/filter bar above a scrollable thumbnail
//! list. Ported from terminus2's `LeftBrowserPanel`, without the right details
//! panel.
//!
//! Thumbnails mirror the Java look: each tile is just the image itself (no card
//! box, no filename), scaled to fit a target box while keeping its aspect ratio,
//! centred, with smooth rounded corners and a selection outline. Because tiles
//! have varying heights, the list is virtualized by hand via
//! [`egui::ScrollArea::show_viewport`]: only tiles near the viewport are laid
//! out, drawn, and have their thumbnails decoded — that's the lazy loading.

use std::path::{Path, PathBuf};

use eframe::egui;
use egui::{CornerRadius, Margin, Stroke};

use crate::image_cache::{Cached, ImageCache};
use crate::theme::*;
use crate::{card_frame, svg_button}; // Note: file_name removed from imports

/// Vertical gap between tiles.
const ROW_GAP: f32 = 12.0;
/// Rounded-corner radius of the image (and selection outline).
const CORNER: u8 = 12;
/// Pixels of look-ahead above/below the viewport to decode tiles early.
const PREFETCH_PX: f32 = 600.0;
/// Aspect ratio (h/w) assumed for a tile whose image hasn't decoded yet.
const DEFAULT_ASPECT: f32 = 1.0;

/// Render the left browser panel.
///
/// Returns `true` if the search query changed this frame so the main app
/// knows to re-filter the list.
pub fn show(
    ui: &mut egui::Ui,
    images: &[PathBuf],
    filtered: &[usize], // NEW: Pass the cached list in
    search: &mut String,
    selected: &mut Option<usize>,
    thumbs: &mut ImageCache,
    video_thumbs: &mut crate::video::VideoThumbs,
    thumb_max_h: f32,
) -> bool {
    let mut search_changed = false;

    egui::Panel::left("browser")
        .resizable(false)
        .exact_size(290.0)
        .show_separator_line(false)
        // Trim the top margin so the card rises up close to the top bar,
        // matching the right details panel.
        .frame(egui::Frame::new().fill(BG()).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            card_frame(22).show(ui, |ui| {
                // Always fill the full panel height, even with no/few images — otherwise
                // the card shrinks to its content and looks tiny when the list is empty.
                ui.set_min_height(ui.available_height());

                // --- Filter bar: search fills the row, gear embedded inside ---
                let search_frame = egui::Frame::default()
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(ui.visuals().widgets.inactive.bg_stroke)
                    .corner_radius(CornerRadius::same(16))
                    .inner_margin(Margin::symmetric(10, 3));

                search_frame.show(ui, |ui| {
                    // 1. WRAP in horizontal to restrict the height to a single line!
                    ui.horizontal(|ui| {
                        // 2. Lay out from right-to-left inside that single line
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {

                            // Draw the button on the far right
                            let settings_svg = egui::include_image!("../icons/settings.svg");
                            svg_button(ui, settings_svg, "Filter Settings", 20.0, MUTED());

                            // The text field consumes exactly the remaining space to the left
                            let resp = ui.add(
                                egui::TextEdit::singleline(search)
                                    .hint_text("Search tag…")
                                    .frame(egui::Frame::NONE)
                                    .margin(Margin::same(0))
                                    .desired_width(f32::INFINITY),
                            );

                            // NEW: Notify the main loop if the user typed something
                            if resp.changed() {
                                search_changed = true;
                            }
                        });
                    });
                });

                ui.add_space(6.0);

                if filtered.is_empty() {
                    ui.add_space(20.0);
                    ui.vertical_centered(|ui| {
                        let msg = if images.is_empty() {
                            "No images yet"
                        } else {
                            "No matches"
                        };
                        ui.label(egui::RichText::new(msg).color(MUTED()).size(13.0));
                    });
                    return;
                }

                thumbnail_list(ui, images, filtered, selected, thumbs, video_thumbs, thumb_max_h);
            });
        });

    search_changed
}

/// The scrollable, hand-virtualized column of bare-image tiles.
fn thumbnail_list(
    ui: &mut egui::Ui,
    images: &[PathBuf],
    filtered: &[usize],
    selected: &mut Option<usize>,
    thumbs: &mut ImageCache,
    video_thumbs: &mut crate::video::VideoThumbs,
    thumb_max_h: f32,
) {
    // When the displayed set changes (a new folder is loaded, or the filter
    // changes) snap the scroll back to the top, instead of keeping the stale
    // offset egui persists from the previous list.
    let signature = list_signature(images, filtered);
    let sig_id = ui.id().with("thumb_list_signature");
    let list_changed = ui.memory_mut(|m| {
        let prev = m.data.get_temp::<u64>(sig_id);
        m.data.insert_temp(sig_id, signature);
        prev != Some(signature)
    });

    // Track the selected *image* (by path, not index) so we keep it in view when
    // the user navigates. Tracking by path also means deleting an image — which
    // keeps the same index but shifts the next image into it — counts as a
    // selection change, so the list follows it instead of snapping to the top.
    let selected_path = (*selected).and_then(|i| images.get(i).cloned());
    let sel_id = ui.id().with("thumb_selected_path");
    let selection_changed = ui.memory_mut(|m| {
        let prev: Option<Option<PathBuf>> = m.data.get_temp(sel_id);
        m.data.insert_temp(sel_id, selected_path.clone());
        prev != Some(selected_path.clone())
    });

    let mut scroll_area = egui::ScrollArea::vertical().auto_shrink([false, false]);
    // A list change snaps to the top — but if the selection also changed (e.g. a
    // new folder selects its first image) let the scroll-to-selected below win.
    if list_changed && !selection_changed {
        scroll_area = scroll_area.vertical_scroll_offset(0.0);
    }

    scroll_area
        .show_viewport(ui, |ui, viewport| {
            let avail_w = ui.available_width();

            // Size every tile up front (cheap arithmetic) so we know the total
            // content height and each tile's vertical offset.
            let mut sizes: Vec<(f32, f32)> = Vec::with_capacity(filtered.len());
            let mut offsets: Vec<f32> = Vec::with_capacity(filtered.len());
            let mut acc = 0.0;
            for &i in filtered {
                // Videos carry their aspect in the video-thumb cache (known once
                // the poster decodes); images carry it in the image cache. Until
                // either resolves we fall back to a square, then reflow.
                let aspect = if crate::is_video(&images[i]) {
                    video_thumbs.aspect(&images[i]).unwrap_or(DEFAULT_ASPECT)
                } else {
                    thumbs.aspect(&images[i]).unwrap_or(DEFAULT_ASPECT)
                };
                let size = fit(avail_w, thumb_max_h, aspect);
                offsets.push(acc);
                sizes.push(size);
                acc += size.1 + ROW_GAP;
            }
            let total = (acc - ROW_GAP).max(0.0);
            ui.set_min_height(total); // reserve the full scroll range

            let top = ui.min_rect().top();
            let left = ui.min_rect().left();
            let visible_min = viewport.min.y - PREFETCH_PX;
            let visible_max = viewport.max.y + PREFETCH_PX;

            // If the selection changed this frame, scroll just enough to keep
            // the selected tile within the viewport.
            if selection_changed {
                if let Some(sel) = *selected {
                    if let Some(row) = filtered.iter().position(|&i| i == sel) {
                        let (tw, th) = sizes[row];
                        let rect = egui::Rect::from_min_size(
                            egui::pos2(left + (avail_w - tw) * 0.5, top + offsets[row]),
                            egui::vec2(tw, th),
                        );
                        ui.scroll_to_rect(rect, None);
                    }
                }
            }

            for (row, &i) in filtered.iter().enumerate() {
                let y = offsets[row];
                let (tw, th) = sizes[row];
                if y + th < visible_min || y > visible_max {
                    continue; // off-screen: don't lay out, draw, or decode it
                }

                // Centre the tile horizontally and place it at its scroll offset.
                let rect = egui::Rect::from_min_size(
                    egui::pos2(left + (avail_w - tw) * 0.5, top + y),
                    egui::vec2(tw, th),
                );

                let id = ui.id().with(("thumb", i));

                // Interact strictly for clicks, hover text removed.
                let resp = ui.interact(rect, id, egui::Sense::click());

                draw_tile(ui, thumbs, video_thumbs, &images[i], rect, *selected == Some(i));

                if resp.clicked() {
                    *selected = Some(i);
                }
            }
        });
}

/// A cheap fingerprint of the currently displayed list. Changes when a new
/// folder is loaded or the filter changes, which is our cue to reset the
/// scroll position to the top.
fn list_signature(images: &[PathBuf], filtered: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    images.len().hash(&mut h);
    images.first().hash(&mut h);
    images.last().hash(&mut h);
    filtered.len().hash(&mut h);
    filtered.first().hash(&mut h);
    filtered.last().hash(&mut h);
    h.finish()
}

/// Paint a single bare-image tile (image, or a placeholder while it loads),
/// with rounded corners and a selection outline.
fn draw_tile(
    ui: &egui::Ui,
    thumbs: &mut ImageCache,
    video_thumbs: &mut crate::video::VideoThumbs,
    path: &Path,
    rect: egui::Rect,
    is_selected: bool,
) {
    let radius = CornerRadius::same(CORNER);

    if crate::is_video(path) {
        // Show the decoded poster frame; until it's ready (or if VLC isn't
        // available) fall back to a placeholder tile with a video glyph.
        match video_thumbs.request(path, ui.ctx()) {
            Some(tex) => {
                egui::Image::from_texture(&tex)
                    .corner_radius(radius)
                    .paint_at(ui, rect);
            }
            None => {
                ui.painter().rect_filled(rect, radius, FIELD());
                let s = (rect.width().min(rect.height()) * 0.4).clamp(24.0, 72.0);
                let icon_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(s, s));
                egui::Image::new(egui::include_image!("../icons/video.svg"))
                    .tint(MUTED())
                    .paint_at(ui, icon_rect);
            }
        }
        video_badge(ui, rect);
    } else {
        let now = ui.input(|i| i.time);
        match thumbs.request(path, now) {
            // Thumbnails are still images (this cache doesn't animate), but handle the
            // animated variant too so a frame still draws if that ever changes.
            Cached::Ready(tex) | Cached::Animated(tex) => {
                egui::Image::from_texture(&tex)
                    .corner_radius(radius)
                    .paint_at(ui, rect);
            }
            Cached::Loading => {
                ui.painter().rect_filled(rect, radius, FIELD());
                let spinner = egui::Rect::from_center_size(rect.center(), egui::vec2(28.0, 28.0));
                egui::Spinner::new().color(MUTED()).paint_at(ui, spinner);
            }
            Cached::Failed => {
                ui.painter().rect_filled(rect, radius, FIELD());
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Failed",
                    egui::FontId::proportional(13.0),
                    MUTED(),
                );
            }
        }

        // A small "GIF" badge marks animated files (which play in the centre viewer).
        if is_gif(path) {
            gif_badge(ui, rect);
        }
    }

    if is_selected {
        ui.painter().rect_stroke(
            rect,
            radius,
            Stroke::new(3.0, egui::Color32::GRAY),
            egui::StrokeKind::Inside,
        );
    }
}

/// True if `path` has a `.gif` extension.
fn is_gif(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gif"))
        .unwrap_or(false)
}

/// Draw a small "GIF" badge (SVG) in the tile's top-left corner, on a dark
/// rounded backdrop so it reads on any image.
fn gif_badge(ui: &egui::Ui, rect: egui::Rect) {
    // Square so the GIF wordmark in the (square) SVG keeps its proportions.
    let badge = egui::Rect::from_min_size(rect.min + egui::vec2(6.0, 6.0), egui::vec2(24.0, 24.0));
    ui.painter()
        .rect_filled(badge, CornerRadius::same(6), egui::Color32::from_black_alpha(150));
    let icon = egui::include_image!("../icons/gif.svg");
    egui::Image::new(icon)
        .tint(egui::Color32::WHITE)
        .paint_at(ui, badge);
}

/// Draw a small "video" badge (SVG) in the tile's top-left corner, on a dark
/// rounded backdrop — same treatment as the GIF badge.
fn video_badge(ui: &egui::Ui, rect: egui::Rect) {
    let badge = egui::Rect::from_min_size(rect.min + egui::vec2(6.0, 6.0), egui::vec2(24.0, 24.0));
    ui.painter()
        .rect_filled(badge, CornerRadius::same(6), egui::Color32::from_black_alpha(150));
    let icon = egui::include_image!("../icons/video.svg");
    egui::Image::new(icon)
        .tint(egui::Color32::WHITE)
        .paint_at(ui, badge.shrink(3.0));
}

/// Scale a `(w=1, h=aspect)` image to fit inside `max_w × max_h`, preserving
/// aspect ratio. Returns the resulting `(width, height)` in points.
fn fit(max_w: f32, max_h: f32, aspect: f32) -> (f32, f32) {
    let h_at_full_width = max_w * aspect;
    if h_at_full_width <= max_h {
        (max_w, h_at_full_width) // limited by width (landscape / square)
    } else {
        (max_h / aspect, max_h) // limited by height (portrait)
    }
}