//! Detail popup for the Gallery view. Clicking a tile opens this modal: the full
//! image on the left, and on the right the SAME "Details & Actions" layout as the
//! right panel — header, the Tags/Metadata box, the action buttons (Copy · Edit
//! Text · Move · Delete) and the Image Info card — just without the tab switcher.

use std::path::{Path, PathBuf};

use eframe::egui;

use crate::image_cache::{Cached, ImageCache};
use crate::right_details::{self, ImageMeta};
use crate::theme::*;

/// Which view the popup's right side shows (switchable via the menu).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum PopupView {
    #[default]
    Details,
    Civitai,
}

/// State for the gallery detail popup. Lives on `ViewerApp`.
pub struct DetailPopup {
    pub open: bool,
    view: PopupView,
    index: Option<usize>,
    path: Option<PathBuf>,
    /// Sidecar `.txt` tags (editable).
    tags: String,
    /// Embedded SD generation metadata, if any (read-only, formatted for display).
    metadata: Option<String>,
    /// The raw (unformatted) metadata, handed to the Civitai panel so its
    /// `Hashes:` / `TI hashes:` blocks survive (needed to list embeddings).
    metadata_raw: Option<String>,
    showing_meta: bool,
    editing: bool,
    meta: ImageMeta,
    /// Artist / character tag names (for orange/green colouring), looked up the
    /// same way the right panel does.
    artist: std::collections::HashSet<String>,
    character: std::collections::HashSet<String>,
    /// The image viewer's own zoom/pan + right-click menu state (independent of
    /// the centre viewer's).
    zoom: crate::zoom::ZoomState,
}

impl Default for DetailPopup {
    fn default() -> Self {
        Self {
            open: false,
            view: PopupView::Details,
            index: None,
            path: None,
            tags: String::new(),
            metadata: None,
            metadata_raw: None,
            showing_meta: false,
            editing: false,
            meta: ImageMeta::default(),
            artist: std::collections::HashSet::new(),
            character: std::collections::HashSet::new(),
            zoom: crate::zoom::ZoomState::default(),
        }
    }
}

impl DetailPopup {
    /// Open the popup for `path` (the image at `index` in the folder list),
    /// loading its tags, metadata and details once.
    pub fn open_for(&mut self, index: usize, path: &Path) {
        self.open = true;
        self.view = PopupView::Details;
        self.index = Some(index);
        self.path = Some(path.to_path_buf());
        let txt = right_details::sidecar_txt(path);
        self.tags = std::fs::read_to_string(&txt).unwrap_or_default();
        let (disp, raw) = crate::sd_metadata::read_both(path);
        self.metadata = disp;
        self.metadata_raw = raw;
        // Show metadata first only when there are no tags but there is metadata.
        self.showing_meta = self.tags.trim().is_empty() && self.metadata.is_some();
        self.editing = false;
        self.meta = right_details::load_meta(path);
        let mut cache = None;
        let roles = right_details::lookup_tag_roles(&mut cache, path);
        self.artist = roles.artist;
        self.character = roles.character;
    }
}

/// What the popup asks the app to do after it closes.
pub enum DetailAction {
    None,
    /// Move this image to a folder the user picks.
    Move(usize),
    /// Delete this image.
    Delete(usize),
    /// A viewer right-click action (favorite / crop / copy / remove-bg) for `path`,
    /// applied by the app exactly like the centre viewer's.
    Viewer(crate::zoom::ViewerAction, PathBuf),
}

/// Render the popup when open. Returns an action for the app to apply.
pub fn show(
    ctx: &egui::Context,
    popup: &mut DetailPopup,
    viewer: &mut ImageCache,
    civitai: &mut crate::civitai::CivitaiState,
    favorites: &mut crate::favorites::Favorites,
) -> DetailAction {
    if !popup.open {
        return DetailAction::None;
    }
    let Some(path) = popup.path.clone() else {
        popup.open = false;
        return DetailAction::None;
    };
    let index = popup.index;

    let mut action = DetailAction::None;
    let mut want_close = false;

    let screen = ctx.content_rect();
    let win_w = (screen.width() * 0.85).min(1150.0).max(480.0);
    let win_h = (screen.height() * 0.85).min(780.0).max(360.0);

    use crate::PopupPlacement;
    egui::Window::new("gallery_detail")
        .id(egui::Id::new("gallery_detail_popup"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .placed_centered(ctx)
        .fixed_size([win_w, win_h])
        .frame(window_frame())
        .show(ctx, |ui| {
            // Rounder widgets + dark input wells, matching the right panel feel.
            let radius = egui::CornerRadius::same(12);
            {
                let v = ui.visuals_mut();
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.active.corner_radius = radius;
            }

            let gap = 16.0;
            let total_w = ui.available_width();
            let col_h = ui.available_height();
            let img_w = ((total_w - gap) * 0.6).floor();
            let det_w = total_w - gap - img_w;

            ui.horizontal_top(|ui| {
                // LEFT — full image, with zoom/pan + the right-click menu (same as
                // the centre viewer). Bounded to its column so zoom uses that area.
                ui.allocate_ui_with_layout(
                    egui::vec2(img_w, col_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_min_size(egui::vec2(img_w, col_h));
                        let is_fav = favorites.is_favorite(&path);
                        let va = draw_image(ui, &mut popup.zoom, viewer, &path, is_fav);
                        if !matches!(va, crate::zoom::ViewerAction::None) {
                            action = DetailAction::Viewer(va, path.clone());
                        }
                    },
                );

                ui.add_space(gap);

                // RIGHT — the right-panel "Details & Actions" layout.
                ui.allocate_ui_with_layout(
                    egui::vec2(det_w, col_h),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_min_height(col_h);
                        ui.set_width(det_w);
                        right_column(ui, popup, civitai, &path, index, &mut action, &mut want_close);
                    },
                );
            });
        });

    if want_close {
        popup.open = false;
    }
    action
}

/// The right-hand column — a menu switcher (Details ⇄ Civitai) over the chosen view.
fn right_column(
    ui: &mut egui::Ui,
    popup: &mut DetailPopup,
    civitai: &mut crate::civitai::CivitaiState,
    path: &Path,
    index: Option<usize>,
    action: &mut DetailAction,
    want_close: &mut bool,
) {
    // Top bar: the view's title (Details only — the Civitai view draws its own
    // header) plus the menu (☰, like the right panel) and a close (✕).
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add(egui::Button::image(
                    egui::Image::new(egui::include_image!("../icons/close.svg"))
                        .fit_to_exact_size(egui::vec2(24.0, 24.0))
                        .tint(MUTED()),
                ).frame(false))
                .on_hover_text("Close")
                .clicked()
            {
                *want_close = true;
            }
            let menu_icon = egui::Image::new(egui::include_image!("../icons/menu.svg"))
                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                .tint(MUTED());
            let menu_resp = ui
                .add(egui::Button::image(menu_icon).frame(false))
                .on_hover_text("Switch view");
            egui::Popup::menu(&menu_resp)
                .align(egui::RectAlign::BOTTOM_END)
                .show(|ui| {
                    ui.set_min_width(160.0);
                    if ui
                        .selectable_label(popup.view == PopupView::Details, "Details & Actions")
                        .clicked()
                    {
                        popup.view = PopupView::Details;
                        ui.close();
                    }
                    if ui
                        .selectable_label(popup.view == PopupView::Civitai, "Civitai Resources")
                        .clicked()
                    {
                        popup.view = PopupView::Civitai;
                        ui.close();
                    }
                });
            if popup.view == PopupView::Details {
                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                    ui.set_width(ui.available_width());
                    ui.heading(egui::RichText::new("Details & Actions").color(TEXT()).strong());
                });
            }
        });
    });
    ui.add_space(8.0);

    match popup.view {
        PopupView::Details => details_content(ui, popup, path, index, action, want_close),
        PopupView::Civitai => {
            let meta = popup.metadata_raw.clone();
            crate::civitai::show(ui, civitai, Some(path), meta.as_deref());
        }
    }
}

/// The "Details & Actions" content: bottom-anchored buttons + Image Info, with the
/// Tags/Metadata box filling the gap above — exactly like the right panel.
fn details_content(
    ui: &mut egui::Ui,
    popup: &mut DetailPopup,
    path: &Path,
    index: Option<usize>,
    action: &mut DetailAction,
    want_close: &mut bool,
) {
    egui::Panel::bottom("gallery_detail_footer")
        .resizable(false)
        .show_separator_line(false)
        .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            ui.add_space(6.0);
            actions_row(ui, popup, path, index, action, want_close);
            ui.add_space(10.0);
            right_details::image_details_section(ui, &popup.meta);
        });

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/tag.svg"))
                        .fit_to_exact_size(egui::vec2(18.0, 18.0))
                        .tint(TEXT()),
                );
                let title = if popup.showing_meta { "Metadata:" } else { "Tags:" };
                ui.label(egui::RichText::new(title).color(TEXT()).strong());

                let has_tags = !popup.tags.trim().is_empty();
                let has_meta = popup.metadata.is_some();
                if has_tags && has_meta {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Button shows the view you'd switch *to*.
                        let label = if popup.showing_meta { "Tags" } else { "Metadata" };
                        if ui.button(egui::RichText::new(label).size(12.0)).clicked() {
                            popup.showing_meta = !popup.showing_meta;
                            popup.editing = false;
                        }
                    });
                }
            });
            ui.add_space(6.0);
            tags_box_fill(ui, popup);
        });
}

/// The four action buttons (Copy · Edit Text · Move · Delete) — same as the panel.
fn actions_row(
    ui: &mut egui::Ui,
    popup: &mut DetailPopup,
    path: &Path,
    index: Option<usize>,
    action: &mut DetailAction,
    want_close: &mut bool,
) {
    ui.horizontal(|ui| {
        let gap = 8.0;
        ui.spacing_mut().item_spacing.x = gap;
        let bw = (ui.available_width() - gap * 3.0) / 4.0;
        let size = egui::vec2(bw, 35.0);
        let label = |t: &str| egui::RichText::new(t).size(15.0);

        if ui.add_sized(size, egui::Button::new(label("Copy"))).clicked() {
            let text = if popup.showing_meta {
                popup.metadata.clone().unwrap_or_default()
            } else {
                popup.tags.clone()
            };
            if !text.trim().is_empty() {
                ui.ctx().copy_text(text);
            }
        }

        let edit_label = if popup.editing { "Save" } else { "Edit Text" };
        if ui.add_sized(size, egui::Button::new(label(edit_label))).clicked() {
            if popup.editing {
                let txt = right_details::sidecar_txt(path);
                let _ = std::fs::write(&txt, &popup.tags);
                popup.editing = false;
            } else {
                popup.showing_meta = false; // edit the tags, not the read-only metadata
                popup.editing = true;
            }
        }

        if ui.add_sized(size, egui::Button::new(label("Move"))).clicked() {
            if let Some(i) = index {
                *action = DetailAction::Move(i);
                *want_close = true;
            }
        }

        let del = egui::Button::new(label("Delete").color(egui::Color32::WHITE))
            .fill(egui::Color32::from_rgb(180, 40, 40));
        if ui.add_sized(size, del).clicked() {
            if let Some(i) = index {
                *action = DetailAction::Delete(i);
                *want_close = true;
            }
        }
    });
}

/// The tags / metadata box, filling the remaining central height — mirrors the
/// right panel: a frameless monospace `TextEdit` (interactive only in edit mode),
/// with artist (orange) / character (green) colouring.
fn tags_box_fill(ui: &mut egui::Ui, popup: &mut DetailPopup) {
    let showing_meta = popup.showing_meta && popup.metadata.is_some();
    let editable = popup.editing && !showing_meta;
    let mut display_text = if showing_meta {
        popup.metadata.clone().unwrap_or_default()
    } else {
        popup.tags.clone()
    };

    let artist = popup.artist.clone();
    let character = popup.character.clone();
    let highlight_roles = !showing_meta && !(artist.is_empty() && character.is_empty());
    let role_color = if editable { TEXT() } else { TEXT().gamma_multiply(0.8) };

    // Lock the box height to the remaining space so it never grows with the text.
    let box_outer_h = ui.available_height();
    let inner_h = (box_outer_h - 24.0).max(0.0); // minus the 12px margins

    egui::Frame::new()
        .fill(FIELD())
        .corner_radius(egui::CornerRadius::same(22))
        .inner_margin(egui::Margin::same(12))
        .show(ui, |ui| {
            ui.set_height(inner_h);
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical()
                .id_salt("gallery_detail_textbox")
                .auto_shrink([false, false])
                .max_height(inner_h)
                .show(ui, |ui| {
                    let mut text_edit = egui::TextEdit::multiline(&mut display_text)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace)
                        .frame(egui::Frame::NONE)
                        .interactive(editable);
                    if !editable {
                        text_edit = text_edit.text_color(TEXT().gamma_multiply(0.8));
                    }
                    let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                        right_details::highlight_tags(ui, buf.as_str(), &artist, &character, role_color, wrap)
                    };
                    if highlight_roles {
                        text_edit = text_edit.layouter(&mut layouter);
                    }
                    ui.add(text_edit);
                });
        });

    if editable {
        popup.tags = display_text;
    }
}

/// Draw the image in the column. For a ready static image this delegates to the
/// shared zoom viewer (zoom/pan + right-click menu) and returns its action; videos
/// and not-yet-loaded images show a placeholder.
fn draw_image(
    ui: &mut egui::Ui,
    zoom: &mut crate::zoom::ZoomState,
    viewer: &mut ImageCache,
    path: &Path,
    is_fav: bool,
) -> crate::zoom::ViewerAction {
    let rect = ui.available_rect_before_wrap();

    if crate::is_video(path) {
        ui.painter().rect_filled(rect, egui::CornerRadius::same(22), FIELD());
        let s = (rect.width().min(rect.height()) * 0.25).clamp(32.0, 96.0);
        let icon = egui::Rect::from_center_size(rect.center(), egui::vec2(s, s));
        egui::Image::new(egui::include_image!("../icons/video.svg"))
            .tint(MUTED())
            .paint_at(ui, icon);
        return crate::zoom::ViewerAction::None;
    }

    let now = ui.input(|i| i.time);
    match viewer.request(path, now) {
        Cached::Ready(tex) | Cached::Animated(tex) => zoom.show(ui, &tex, path, is_fav),
        Cached::Loading => {
            let sp = egui::Rect::from_center_size(rect.center(), egui::vec2(36.0, 36.0));
            egui::Spinner::new().color(MUTED()).paint_at(ui, sp);
            crate::zoom::ViewerAction::None
        }
        Cached::Failed => {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Couldn't load image",
                egui::FontId::proportional(14.0),
                MUTED(),
            );
            crate::zoom::ViewerAction::None
        }
    }
}

fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(egui::CornerRadius::same(22))
        .inner_margin(egui::Margin::same(18))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 6],
            blur: 24,
            spread: 0,
            color: egui::Color32::from_black_alpha(160),
        })
}
