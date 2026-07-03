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
    video_previews: &mut crate::video::VideoPreviews,
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

            // Start at the very top whenever a different image list arrives (new
            // folder, search/filter change). egui remembers the scroll offset per
            // widget id — across folder changes and even app restarts — so without
            // this the gallery opens wherever the previous list was left.
            let sig = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                filtered.len().hash(&mut h);
                if let Some(&f) = filtered.first() {
                    images[f].hash(&mut h);
                }
                if let Some(&l) = filtered.last() {
                    images[l].hash(&mut h);
                }
                h.finish()
            };
            let sig_id = egui::Id::new("gallery_list_sig");
            let list_changed = ui.data_mut(|d| {
                let prev = d.get_temp::<u64>(sig_id);
                d.insert_temp(sig_id, sig);
                prev != Some(sig)
            });

            let mut scroll = egui::ScrollArea::vertical().auto_shrink([false, false]);
            if list_changed {
                scroll = scroll.vertical_scroll_offset(0.0);
            }
            scroll.show(ui, |ui| {
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
                                    tile(col_ui, thumbs, video_thumbs, video_previews, path, rect, selected == Some(i), is_fav);
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

/// The floating search/filter pill shown over the Gallery in the bottom-right
/// corner: a frameless "Search tag…" field plus a settings gear that opens the
/// gallery filters (media type + thumbnail size). Movable (drag the pill
/// background), but unlike the other popups its position is remembered as an
/// OFFSET FROM THE BOTTOM-RIGHT CORNER, not an absolute point — so it always
/// starts in the corner and stays put relative to it when the app window is
/// resized. Returns `true` when the search text or media filter changed so the
/// app re-filters.
pub fn search_pill(
    ctx: &egui::Context,
    search: &mut String,
    settings: &mut crate::settings::Settings,
) -> bool {
    use crate::left_panel_settings::MediaFilter;

    let mut changed = false;
    let pill_w = 320.0;
    let pill_h = 46.0;
    let screen = ctx.content_rect();

    // Gap between the pill's bottom-right corner and the screen's. Persisted
    // (egui memory) when the user drags the pill; clamped so a shrunken window
    // can never strand it off-screen.
    let off_id = egui::Id::new("gallery_search_pill_offset");
    let default_off = egui::vec2(18.0, 18.0);
    let off = if crate::movable_popups() {
        ctx.data_mut(|d| *d.get_persisted_mut_or(off_id, default_off))
    } else {
        default_off
    };
    let max_off = (screen.size() - egui::vec2(pill_w, pill_h)).max(egui::Vec2::ZERO);
    let off = off.clamp(egui::Vec2::ZERO, max_off);

    // Last frame's measured pill size (egui auto-sizes the window); the
    // constants are only the first-frame estimate.
    let size_id = egui::Id::new("gallery_search_pill_size");
    let size = ctx
        .data(|d| d.get_temp::<egui::Vec2>(size_id))
        .unwrap_or(egui::vec2(pill_w, pill_h));

    // Only assert the position when the target really moved (first frame,
    // app-window resize, drag, clamp), and with a 1px tolerance. Re-asserting
    // every frame — or chasing sub-pixel rounding — makes egui treat the window
    // as perpetually moving and repaint forever; under a gallery full of
    // thumbnails/video tiles that runaway loop can exhaust the whole machine.
    let desired = screen.max - size - off;
    let anchor_id = egui::Id::new("gallery_search_pill_anchor");
    let last = ctx.data(|d| d.get_temp::<egui::Pos2>(anchor_id));
    let assert_pos =
        last.is_none_or(|l| (l.x - desired.x).abs() > 1.0 || (l.y - desired.y).abs() > 1.0);
    if assert_pos {
        ctx.data_mut(|d| d.insert_temp(anchor_id, desired));
    }

    let mut win = egui::Window::new("")
        .id(egui::Id::new("gallery_search_pill"))
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .movable(crate::movable_popups());
    if assert_pos {
        win = win.current_pos(desired);
    }
    let resp = win
        .frame(
            egui::Frame::new()
                .fill(PANEL())
                // Half the pill height → fully rounded ends, like the mock.
                .corner_radius(CornerRadius::same((pill_h / 2.0) as u8))
                .inner_margin(Margin::symmetric(16, 8))
                .stroke(Stroke::new(1.0, EDGE()))
                .shadow(egui::epaint::Shadow {
                    offset: [0, 4],
                    blur: 16,
                    spread: 0,
                    color: egui::Color32::from_black_alpha(120),
                }),
        )
        .show(ctx, |ui| {
            ui.set_width(pill_w - 32.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                // Gear first in a right-to-left layout so it pins to the right
                // end; the search field then fills the rest.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let gear = egui::include_image!("../icons/settings.svg");
                    let gear_resp = crate::svg_button(
                        ui,
                        gear,
                        "Gallery filters",
                        18.0,
                        crate::theme::icon_tint(MUTED()),
                    );
                    // The filters popup opens UPWARD (the pill sits at the
                    // bottom of the screen).
                    egui::Popup::menu(&gear_resp)
                        .align(egui::RectAlign::TOP_END)
                        .show(|ui| {
                            ui.set_min_width(210.0);
                            let r = CornerRadius::same(8);
                            ui.visuals_mut().widgets.inactive.corner_radius = r;
                            ui.visuals_mut().widgets.hovered.corner_radius = r;
                            ui.visuals_mut().widgets.active.corner_radius = r;

                            ui.label(
                                egui::RichText::new("SHOW").color(MUTED()).strong().size(10.5),
                            );
                            ui.add_space(2.0);
                            for opt in MediaFilter::OPTIONS {
                                if ui
                                    .selectable_label(settings.media_filter == opt, opt.label())
                                    .clicked()
                                {
                                    if settings.media_filter != opt {
                                        settings.media_filter = opt;
                                        changed = true;
                                    }
                                    ui.close();
                                }
                            }

                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new("THUMBNAIL SIZE")
                                    .color(MUTED())
                                    .strong()
                                    .size(10.5),
                            );
                            ui.add_space(2.0);
                            ui.spacing_mut().slider_width = ui.available_width() - 8.0;
                            ui.add(
                                egui::Slider::new(&mut settings.thumbnail_size, 120.0..=400.0)
                                    .show_value(false),
                            );
                        });

                    // Search field fills the remaining pill width.
                    let resp = ui.add(
                        egui::TextEdit::singleline(search)
                            .hint_text(egui::RichText::new("Search tag…").color(MUTED()))
                            .frame(egui::Frame::NONE)
                            .desired_width(ui.available_width())
                            .margin(Margin::symmetric(4, 4)),
                    );
                    if resp.changed() {
                        changed = true;
                    }
                });
            });
        });

    if let Some(resp) = resp {
        ctx.data_mut(|d| d.insert_temp(size_id, resp.response.rect.size()));
        // While the pill is being dragged egui moves the window itself; record
        // the landing spot as a new corner offset so the next frame's
        // `current_pos` (and future resizes/restarts) keep it there relative to
        // the corner.
        if resp.response.dragged() {
            let new_off = (screen.max - resp.response.rect.max).clamp(egui::Vec2::ZERO, max_off);
            ctx.data_mut(|d| d.insert_persisted(off_id, new_off));
        }
    }

    changed
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
    video_previews: &mut crate::video::VideoPreviews,
    path: &Path,
    rect: egui::Rect,
    selected: bool,
    is_fav: bool,
) {
    let radius = CornerRadius::same(CORNER);
    if crate::is_video(path) {
        // A live preview frame (the "Video thumbnail play" setting) wins over the
        // static poster when one is playing for this tile.
        let live = video_previews.frame(path, ui.ctx());
        match live.or_else(|| video_thumbs.request(path, ui.ctx())) {
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
        // A "video" badge in the top-left corner, matching the left browser.
        crate::left_browser::video_badge(ui, rect);
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
        // A "GIF" badge marks animated files, matching the left browser.
        if crate::left_browser::is_gif(path) {
            crate::left_browser::gif_badge(ui, rect);
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
    // Rasterize once + GPU-scale (see `left_browser::paint_pulsing_heart`) instead
    // of re-rasterizing the SVG every frame as the pulse size changes.
    let icon = egui::include_image!("../icons/heart.svg");
    crate::left_browser::paint_pulsing_heart(ui, icon, badge, base);
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
}
