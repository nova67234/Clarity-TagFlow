//! Detail popup for the Gallery view. Clicking a tile opens this modal: the full
//! image on the left, and on the right the SAME "Details & Actions" layout as the
//! right panel — header, the Tags/Metadata box, the action buttons (Copy · Edit
//! Text · Move · Delete) and the Image Info card — just without the tab switcher.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Instant, SystemTime};

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

/// What the popup's background loader streams back: the embedded SD metadata
/// first (it only needs a file read), then the details-card info (a full
/// decode + colour extraction).
enum DetailMsg {
    /// `(display, raw)` from `sd_metadata::read_both`.
    Metadata(Option<String>, Option<String>),
    Info(ImageMeta),
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
    /// True once the background `read_both` has delivered (the Civitai view
    /// waits for it — starting its lookup with `None` would cache "no
    /// resources" for the image).
    meta_loaded: bool,
    showing_meta: bool,
    editing: bool,
    /// Copy-button result flash: (start, success) — green on copy, amber when
    /// there was nothing to copy. Same feedback as the right panel.
    copy_flash: Option<(Instant, bool)>,
    /// Edit/Save-button result flash: green = saved, red = write failed.
    save_flash: Option<(Instant, bool)>,
    /// When edit mode was entered — pulses the tag box toward the accent.
    edit_flash_start: Option<Instant>,
    /// Delete-confirmation dialog state (mirrors the right panel's).
    show_delete_confirm: bool,
    skip_delete_confirm: bool,
    meta: ImageMeta,
    /// Receiver for the background loader (SD-metadata read, then the full
    /// decode + colour palette — all off the UI thread, like the right panel:
    /// opening the popup must never block on a multi-megapixel file).
    meta_rx: Option<mpsc::Receiver<DetailMsg>>,
    /// Bumped per load so a stale background decode can bail early when the
    /// popup is quickly reopened on another image.
    meta_gen: Arc<AtomicU64>,
    /// Cached parse of the shared tag_roles.json (mtime + md5->roles map),
    /// reloaded only when the file changes — so reopening the popup doesn't
    /// re-parse it every time.
    roles_cache: Option<(
        Option<SystemTime>,
        std::collections::HashMap<String, right_details::TagRoles>,
    )>,
    /// Artist / character tag names (for orange/green colouring), looked up the
    /// same way the right panel does.
    artist: std::collections::HashSet<String>,
    character: std::collections::HashSet<String>,
    /// The image viewer's own zoom/pan + right-click menu state (independent of
    /// the centre viewer's).
    zoom: crate::zoom::ZoomState,
    /// In-popup video playback (its own player, independent of the centre
    /// viewer's), plus the clip it was started for.
    video_player: Option<crate::video::VideoPlayer>,
    video_path: Option<PathBuf>,
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
            meta_loaded: false,
            showing_meta: false,
            editing: false,
            copy_flash: None,
            save_flash: None,
            edit_flash_start: None,
            show_delete_confirm: false,
            skip_delete_confirm: false,
            meta: ImageMeta::default(),
            meta_rx: None,
            meta_gen: Arc::new(AtomicU64::new(0)),
            roles_cache: None,
            artist: std::collections::HashSet::new(),
            character: std::collections::HashSet::new(),
            zoom: crate::zoom::ZoomState::default(),
            video_player: None,
            video_path: None,
        }
    }
}

impl DetailPopup {
    /// Open the popup for `path` (the image at `index` in the folder list),
    /// loading its tags, metadata and details once. The details-card metadata
    /// (full decode + colour palette) loads on a background thread; everything
    /// read here synchronously is cheap (sidecar txt, embedded text chunks,
    /// the mtime-cached roles map).
    pub fn open_for(&mut self, index: usize, path: &Path, ctx: &egui::Context) {
        self.open = true;
        // `self.view` is intentionally left alone: the popup reopens on whichever
        // view (Details / Civitai) was selected last.
        self.index = Some(index);
        self.path = Some(path.to_path_buf());
        let txt = right_details::sidecar_txt(path);
        self.tags = std::fs::read_to_string(&txt).unwrap_or_default();
        self.metadata = None;
        self.metadata_raw = None;
        self.meta_loaded = false;
        self.showing_meta = false;
        self.editing = false;
        self.copy_flash = None;
        self.save_flash = None;
        self.edit_flash_start = None;
        self.show_delete_confirm = false;
        // Stop any clip from the previous popup content; if `path` is a video it
        // restarts fresh in `draw_video` (also re-points it on arrow-key paging).
        self.video_player = None;
        self.video_path = None;

        // Everything that touches the image file goes to a background thread:
        // the SD-metadata read first (a hi-res file is read in full and scanned,
        // which froze the popup), then the details-card load (full decode +
        // dominant colours) — same pattern as the right panel. The card shows
        // "Loading..." until its part lands.
        self.meta = ImageMeta::loading();
        let (tx, rx) = mpsc::channel();
        self.meta_rx = Some(rx);
        let generation = self.meta_gen.fetch_add(1, Ordering::SeqCst) + 1;
        let meta_gen = Arc::clone(&self.meta_gen);
        let path_clone = path.to_path_buf();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            // Skip everything if the popup already moved on.
            if meta_gen.load(Ordering::SeqCst) != generation {
                return;
            }
            let (disp, raw) = crate::sd_metadata::read_both(&path_clone);
            if meta_gen.load(Ordering::SeqCst) != generation
                || tx.send(DetailMsg::Metadata(disp, raw)).is_err()
            {
                return;
            }
            ctx.request_repaint();

            let meta = right_details::load_meta(&path_clone);
            // Deliver + repaint only if this is still the current image.
            if meta_gen.load(Ordering::SeqCst) == generation
                && tx.send(DetailMsg::Info(meta)).is_ok()
            {
                ctx.request_repaint();
            }
        });

        let roles = right_details::lookup_tag_roles(&mut self.roles_cache, path);
        self.artist = roles.artist;
        self.character = roles.character;
    }

    /// The clip currently playing in the popup, if any — the poster cache skips
    /// it so the file isn't decoded twice at once (like the centre player).
    pub fn playing_video(&self) -> Option<&Path> {
        self.video_player.as_ref().and(self.video_path.as_deref())
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
    confirm_before_delete: &mut bool,
) -> DetailAction {
    if !popup.open {
        // Release a playing clip once the popup is gone (stops VLC, frees the file).
        popup.video_player = None;
        popup.video_path = None;
        return DetailAction::None;
    }
    // Non-blocking drain of the background loader.
    if let Some(rx) = &popup.meta_rx {
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(DetailMsg::Metadata(disp, raw)) => {
                    popup.metadata = disp;
                    popup.metadata_raw = raw;
                    popup.meta_loaded = true;
                    // Open straight to the metadata view when there are no tags
                    // but there is metadata — deferred to its arrival (and
                    // skipped if the user already started editing the tags).
                    if !popup.editing && popup.metadata.is_some() && popup.tags.trim().is_empty() {
                        popup.showing_meta = true;
                    }
                }
                Ok(DetailMsg::Info(meta)) => popup.meta = meta,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            popup.meta_rx = None;
        }
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
            // Only the top strip drags the popup — not stray drags on the body.
            crate::popup_drag_strip(ui, 30.0);
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
                        if crate::is_video(&path) {
                            draw_video(ui, popup, &path);
                        } else {
                            let is_fav = favorites.is_favorite(&path);
                            let va = draw_image(ui, &mut popup.zoom, viewer, &path, is_fav);
                            if !matches!(va, crate::zoom::ViewerAction::None) {
                                action = DetailAction::Viewer(va, path.clone());
                            }
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
                        right_column(
                            ui,
                            popup,
                            civitai,
                            &path,
                            index,
                            *confirm_before_delete,
                            &mut action,
                            &mut want_close,
                        );
                    },
                );
            });
        });

    // Delete confirmation — the same centered modal as the right panel's.
    if popup.show_delete_confirm {
        match right_details::delete_confirm_dialog(
            ctx,
            "gallery_detail_confirm_delete",
            &mut popup.skip_delete_confirm,
        ) {
            Some(true) => {
                popup.show_delete_confirm = false;
                // "Don't ask again" disables (and persists, via Settings) future prompts.
                if popup.skip_delete_confirm {
                    *confirm_before_delete = false;
                }
                if let Some(i) = index {
                    action = DetailAction::Delete(i);
                    want_close = true;
                }
            }
            Some(false) => popup.show_delete_confirm = false,
            None => {}
        }
    }

    if want_close {
        popup.open = false;
        // Stop playback right away (don't wait a frame for the early return).
        popup.video_player = None;
        popup.video_path = None;
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
    confirm_before_delete: bool,
    action: &mut DetailAction,
    want_close: &mut bool,
) {
    // Top bar: the view's title (Details only — the Civitai view draws its own
    // header) plus the menu (☰, like the right panel) and a close (✕).
    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // click_and_drag so a click that slips a pixel is swallowed by the
            // button instead of falling through and dragging the popup.
            if ui
                .add(egui::Button::image(
                    egui::Image::new(egui::include_image!("../icons/close.svg"))
                        .fit_to_exact_size(egui::vec2(24.0, 24.0))
                        .tint(MUTED()),
                ).frame(false).sense(egui::Sense::click_and_drag()))
                .on_hover_text("Close")
                .clicked()
            {
                *want_close = true;
            }
            let menu_icon = egui::Image::new(egui::include_image!("../icons/menu.svg"))
                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                .tint(MUTED());
            let menu_resp = ui
                .add(egui::Button::image(menu_icon).frame(false).sense(egui::Sense::click_and_drag()))
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
        PopupView::Details => {
            details_content(ui, popup, path, index, confirm_before_delete, action, want_close)
        }
        PopupView::Civitai => {
            if popup.meta_loaded {
                let meta = popup.metadata_raw.clone();
                crate::civitai::show(ui, civitai, Some(path), meta.as_deref());
            } else {
                // The background metadata read hasn't landed yet; starting the
                // Civitai lookup now (with no metadata) would cache an empty
                // result for this image. It repaints in when the read finishes.
                ui.add_space(24.0);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Reading image metadata…").color(MUTED()).size(13.0),
                    );
                });
            }
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
    confirm_before_delete: bool,
    action: &mut DetailAction,
    want_close: &mut bool,
) {
    egui::Panel::bottom("gallery_detail_footer")
        .resizable(false)
        .show_separator_line(false)
        .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            ui.add_space(6.0);
            actions_row(ui, popup, path, index, confirm_before_delete, action, want_close);
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
                let title = if popup.showing_meta { "Metadata" } else { "Tags" };
                ui.label(egui::RichText::new(title).color(TEXT()).strong());

                let has_tags = !popup.tags.trim().is_empty();
                let has_meta = popup.metadata.is_some();
                if has_tags && has_meta {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Button shows the view you'd switch *to* — icon + label,
                        // same as the right panel's switch.
                        let to = if popup.showing_meta { "Tags" } else { "Metadata" };
                        let switch_icon = egui::include_image!("../icons/window_switch.svg");
                        let btn = egui::Button::image_and_text(
                            egui::Image::new(switch_icon)
                                .fit_to_exact_size(egui::vec2(16.0, 16.0))
                                .tint(TEXT()),
                            egui::RichText::new(to).size(13.0),
                        );
                        if ui
                            .add(btn)
                            .on_hover_text(
                                "Switch between .txt tags and embedded generation metadata",
                            )
                            .clicked()
                        {
                            popup.showing_meta = !popup.showing_meta;
                            popup.editing = false; // leave edit mode on switch
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
    confirm_before_delete: bool,
    action: &mut DetailAction,
    want_close: &mut bool,
) {
    ui.horizontal(|ui| {
        let gap = 8.0;
        ui.spacing_mut().item_spacing.x = gap;
        let bw = (ui.available_width() - gap * 3.0) / 4.0;
        let size = egui::vec2(bw, 35.0);
        let label = |t: &str| egui::RichText::new(t).size(15.0);

        // Copy flashes green when it copies, amber when there's nothing to copy.
        let mut copy_btn = egui::Button::new(label("Copy"));
        if let Some(fill) = right_details::flash_fill(
            ui,
            popup.copy_flash,
            right_details::FLASH_GREEN,
            right_details::FLASH_AMBER,
        ) {
            copy_btn = copy_btn.fill(fill);
        }
        if ui
            .add_sized(size, copy_btn)
            .on_hover_text("Copy tags to clipboard")
            .clicked()
        {
            let text = if popup.showing_meta {
                popup.metadata.clone().unwrap_or_default()
            } else {
                popup.tags.clone()
            };
            let ok = !text.trim().is_empty();
            if ok {
                ui.ctx().copy_text(text);
            }
            popup.copy_flash = Some((Instant::now(), ok));
        }

        // Edit/Save: flashes green when a save succeeds, red when the write
        // fails (stays in edit mode to retry) — same as the right panel.
        let mut edit_btn =
            egui::Button::new(label(if popup.editing { "Save" } else { "Edit Text" }));
        if let Some(fill) = right_details::flash_fill(
            ui,
            popup.save_flash,
            right_details::FLASH_GREEN,
            right_details::FLASH_RED,
        ) {
            edit_btn = edit_btn.fill(fill);
        }
        if ui.add_sized(size, edit_btn).clicked() {
            if popup.showing_meta {
                // The metadata view is read-only. Clicking Edit Text here drops
                // to the .txt tags view, creating the sidecar file if it doesn't
                // exist yet, and enters edit mode.
                let txt = right_details::sidecar_txt(path);
                if !txt.exists() {
                    let _ = std::fs::write(&txt, &popup.tags);
                }
                popup.showing_meta = false;
                popup.editing = true;
                popup.edit_flash_start = Some(Instant::now());
                popup.save_flash = None;
            } else if popup.editing {
                let txt = right_details::sidecar_txt(path);
                let ok = std::fs::write(&txt, &popup.tags).is_ok();
                if ok {
                    popup.editing = false; // saved — back to view mode
                }
                popup.save_flash = Some((Instant::now(), ok));
            } else {
                popup.editing = true;
                popup.edit_flash_start = Some(Instant::now());
                popup.save_flash = None; // clear any stale save flash
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
                // Confirm first, unless confirmations are disabled.
                if confirm_before_delete {
                    popup.show_delete_confirm = true;
                    popup.skip_delete_confirm = false; // fresh checkbox
                } else {
                    *action = DetailAction::Delete(i);
                    *want_close = true;
                }
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

    // Box background, with the "ready to edit" flash pulsing it toward the
    // accent just after entering edit mode — same as the right panel.
    let box_fill = right_details::edit_flash_fill(ui, popup.edit_flash_start);

    egui::Frame::new()
        .fill(box_fill)
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
                    let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                        right_details::highlight_tags(ui, buf.as_str(), &artist, &character, role_color, wrap)
                    };
                    if editable {
                        let mut text_edit = egui::TextEdit::multiline(&mut display_text)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace)
                            .frame(egui::Frame::NONE)
                            // Fill the whole box so clicking anywhere inside it (not
                            // just on the first lines) focuses the editor.
                            .min_size(egui::vec2(0.0, inner_h));
                        if highlight_roles {
                            text_edit = text_edit.layouter(&mut layouter);
                        }
                        ui.add(text_edit);
                    } else {
                        // Display mode: an immutable `&str` buffer ignores every edit,
                        // so the text can be highlighted and copied but never changed.
                        let meta_color = TEXT().gamma_multiply(0.8);
                        // In the metadata view, colour the app stamp ("Clarity
                        // TagFlow" green, the version blue) — same as the right panel.
                        let stamp_meta = showing_meta && display_text.contains("Clarity TagFlow");
                        let mut stamp_layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                            right_details::highlight_app_stamp(ui, buf.as_str(), meta_color, wrap)
                        };
                        let mut read_only = display_text.as_str();
                        let mut text_edit = egui::TextEdit::multiline(&mut read_only)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace)
                            .frame(egui::Frame::NONE)
                            .text_color(meta_color);
                        if highlight_roles {
                            text_edit = text_edit.layouter(&mut layouter);
                        } else if stamp_meta {
                            text_edit = text_edit.layouter(&mut stamp_layouter);
                        }
                        ui.add(text_edit);
                    }
                });
        });

    if editable {
        popup.tags = display_text;
    }
}

/// Play the video in the popup's left column via libVLC — the same lifecycle as
/// the centre viewer: (re)start the player when the clip changes, then pull
/// decoded frames; show the install/unsupported notice when no player can run.
fn draw_video(ui: &mut egui::Ui, popup: &mut DetailPopup, path: &Path) {
    let support = crate::video::support();
    if matches!(support, crate::video::VideoSupport::Available) {
        if popup.video_path.as_deref() != Some(path) {
            popup.video_path = Some(path.to_path_buf());
            popup.video_player = crate::video::VideoPlayer::start(path, ui.ctx());
        }
        if let Some(player) = &mut popup.video_player {
            match player.frame(ui.ctx()) {
                Some(tex) => crate::show_fitted(ui, &tex, false),
                None => {
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Spinner::new().size(48.0).color(MUTED()));
                    });
                }
            }
            // Keep pulling frames, capped to ~60 Hz like the centre viewer (new
            // frames also wake us instantly via the player's display callback).
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(16));
            return;
        }
    } else {
        // No runtime (or unsupported build): drop any stale player so that, once
        // VLC is installed, reopening the clip restarts it.
        popup.video_player = None;
        popup.video_path = None;
    }
    crate::video_notice(ui, path, support);
}

/// Draw the image in the column. For a ready static image this delegates to the
/// shared zoom viewer (zoom/pan + right-click menu) and returns its action;
/// not-yet-loaded images show a placeholder.
fn draw_image(
    ui: &mut egui::Ui,
    zoom: &mut crate::zoom::ZoomState,
    viewer: &mut ImageCache,
    path: &Path,
    is_fav: bool,
) -> crate::zoom::ViewerAction {
    let rect = ui.available_rect_before_wrap();

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
