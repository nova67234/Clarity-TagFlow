//! The right details panel — a side panel for viewing and editing image sidecar metadata.
//! Currently tailored exclusively to support `.txt` sidecars.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Instant, SystemTime};

use eframe::egui;

use crate::theme::*;
use crate::card_frame;
use crate::tag_manager::TagManagerState;

/// Width of the right details panel.
pub const PANEL_WIDTH: f32 = 420.0;

/// Which view the right panel is showing, chosen from the menu dropdown.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum RightView {
    /// The image details + tag-editing actions (the original view).
    #[default]
    Details,
    /// The Tag Manager settings UI (replaces the details view).
    TagManager,
    /// The Gelbooru downloader UI (replaces the details view).
    Downloader,
    /// The Civitai resource-info UI (replaces the details view).
    Civitai,
    /// The Pixal3D image->3D requirement setup UI. Linux/Windows only — the
    /// variant doesn't exist on macOS so the whole feature is compiled out.
    #[cfg(not(target_os = "macos"))]
    Pixal3D,
    /// The Flux text-to-image generation (ComfyUI) view. NVIDIA-only, like Pixal3D.
    #[cfg(not(target_os = "macos"))]
    Generate,
    /// The Z-Image Turbo generation view (same ComfyUI backend).
    #[cfg(not(target_os = "macos"))]
    ZImage,
    /// The LTX-Video text/image-to-video generation view (same ComfyUI backend).
    #[cfg(not(target_os = "macos"))]
    Ltx,
    /// The Wan 2.2 text/image-to-video generation view (same ComfyUI backend).
    #[cfg(not(target_os = "macos"))]
    Wan,
    /// The SDXL Base 1.0 text-to-image generation view (same ComfyUI backend).
    #[cfg(not(target_os = "macos"))]
    Sdxl,
    /// The Anima Base v1.0 text-to-image generation view (same ComfyUI backend).
    #[cfg(not(target_os = "macos"))]
    Anima,
    /// The Krea 2 Turbo text-to-image generation view (same ComfyUI backend).
    #[cfg(not(target_os = "macos"))]
    Krea2,
}

/// Map a generation family (chosen from a Generate header dropdown) to the
/// right-panel view that hosts it.
#[cfg(not(target_os = "macos"))]
fn family_view(family: crate::generate::GenFamily) -> RightView {
    match family {
        crate::generate::GenFamily::Sdxl => RightView::Sdxl,
        crate::generate::GenFamily::Anima => RightView::Anima,
        crate::generate::GenFamily::Krea2 => RightView::Krea2,
        crate::generate::GenFamily::Flux => RightView::Generate,
        crate::generate::GenFamily::ZImage => RightView::ZImage,
        crate::generate::GenFamily::Ltx => RightView::Ltx,
        crate::generate::GenFamily::Wan => RightView::Wan,
    }
}

/// Read-only file metadata shown in the bottom "Image Details" card. A Rust
/// port of the fields surfaced by terminus2's `ImageDetails` Swing panel.
pub struct ImageMeta {
    name: String,
    file_type: String,
    dimensions: String,
    size: String,
    date: String,
    /// A small palette of dominant colors extracted from the image.
    colors: Vec<egui::Color32>,
    /// True when the selected file is a video. The details card then shows a
    /// "Video Info" view (video icon + duration / codec) instead of "Image Info".
    is_video: bool,
    /// True when the selected file is an animated GIF. The card then shows a
    /// "GIF Info" view (gif icon + frame count / duration).
    is_gif: bool,
    /// Playback length, e.g. "12:34" (videos / animated GIFs; "---" otherwise).
    duration: String,
    /// Video codec, e.g. "H.264 (avc1)" (videos only; "---" otherwise).
    codec: String,
    /// Animation frame count, e.g. "48 frames" (GIFs only; "---" otherwise).
    frames: String,
}

impl Default for ImageMeta {
    fn default() -> Self {
        Self {
            name: "---".into(),
            file_type: "---".into(),
            dimensions: "---".into(),
            size: "---".into(),
            date: "---".into(),
            colors: Vec::new(),
            is_video: false,
            is_gif: false,
            duration: "---".into(),
            codec: "---".into(),
            frames: "---".into(),
        }
    }
}

impl ImageMeta {
    /// The placeholder shown in the details card while a background `load_meta`
    /// (full decode + colour extraction) is still running.
    pub(crate) fn loading() -> Self {
        Self {
            name: "Loading...".into(),
            file_type: "...".into(),
            dimensions: "...".into(),
            size: "...".into(),
            date: "...".into(),
            colors: vec![],
            ..Self::default()
        }
    }
}

/// Artist (username) + character tag names for the current image, looked up from
/// the downloader's shared `tag_roles.json` (keyed by md5). Used to colour those
/// tags in the tag box.
#[derive(Default, Clone)]
pub(crate) struct TagRoles {
    pub(crate) artist: std::collections::HashSet<String>,
    pub(crate) character: std::collections::HashSet<String>,
}

/// Maintains the UI state for the right panel, such as the loaded text buffer
/// and whether the user is currently in edit mode.
pub struct RightPanelState {
    pub current_tags: String,
    /// Artist/character tags for the current image (looked up by md5 from the
    /// shared tag_roles.json), so the tag box can colour artist orange / char green.
    tag_roles: TagRoles,
    /// Cached parse of the shared tag_roles.json (its mtime + md5->roles map),
    /// reloaded only when the file changes — so selection changes don't re-parse it.
    roles_cache: Option<(Option<SystemTime>, std::collections::HashMap<String, TagRoles>)>,
    /// Embedded Stable-Diffusion generation metadata for the selected image,
    /// or `None` when the image carries none. Read by `crate::sd_metadata`.
    pub sd_metadata: Option<String>,
    /// The *raw* (unformatted) embedded metadata for the selected image — handed to
    /// the Civitai panel so its `Hashes:` / `TI hashes:` blocks survive intact
    /// (formatting them away hid embeddings). `None` when the image carries none.
    pub sd_metadata_raw: Option<String>,
    /// When true, the tag box shows the (read-only) SD metadata instead of tags.
    pub showing_meta: bool,
    /// State for the Gelbooru downloader view.
    pub downloader: crate::download::DownloaderState,
    /// State for the Civitai resource-info view.
    pub civitai: crate::civitai::CivitaiState,
    /// State for the Pixal3D requirement-setup view (Linux/Windows only).
    #[cfg(not(target_os = "macos"))]
    pub pixal3d: crate::pixal3d::Pixal3DState,
    /// State for the Flux/ComfyUI generation view (Linux/Windows only).
    #[cfg(not(target_os = "macos"))]
    pub generate: crate::generate::GenerateState,
    /// State for the Z-Image generation view (same backend, different model).
    #[cfg(not(target_os = "macos"))]
    pub zimage: crate::generate::GenerateState,
    /// State for the LTX-Video generation view (same backend, video output).
    #[cfg(not(target_os = "macos"))]
    pub ltx: crate::generate::GenerateState,
    /// State for the Wan 2.2 generation view (same backend, video output).
    #[cfg(not(target_os = "macos"))]
    pub wan: crate::generate::GenerateState,
    /// State for the SDXL Base 1.0 generation view (same backend, image output).
    #[cfg(not(target_os = "macos"))]
    pub sdxl: crate::generate::GenerateState,
    /// State for the Anima Base v1.0 generation view (same backend, image output).
    #[cfg(not(target_os = "macos"))]
    pub anima: crate::generate::GenerateState,
    /// State for the Krea 2 Turbo generation view.
    #[cfg(not(target_os = "macos"))]
    pub krea2: crate::generate::GenerateState,
    pub is_editing: bool,

    /// Which view the panel currently shows (Details vs Tag Manager).
    pub view: RightView,

    // --- Delete Confirmation State ---
    pub show_delete_confirm: bool,
    pub skip_delete_confirm: bool,

    last_path: Option<PathBuf>,
    /// Cached metadata for the selected image, recomputed only on selection change.
    meta: ImageMeta,
    /// When edit mode was last entered, used to play a brief "ready to edit"
    /// flash on the tag box. `None` once the flash has finished / never started.
    edit_flash_start: Option<Instant>,
    /// When Copy was last clicked, plus whether it actually copied text (`true`)
    /// or there was nothing to copy (`false`). Drives the Copy button's flash.
    copy_flash: Option<(Instant, bool)>,
    /// When Save was last clicked, plus whether the write succeeded (`true`) or
    /// failed (`false`). Drives the Edit/Save button's green/red flash.
    save_flash: Option<(Instant, bool)>,
    /// Receiver to listen for background metadata loading
    meta_rx: Option<mpsc::Receiver<ImageMeta>>,
    /// Bumped on every new metadata load. A background load reads this and
    /// aborts *before* its heavy full-resolution decode if the selection has
    /// already moved on, so fast navigation can't pile up redundant decodes
    /// that starve the centre viewer and make it flicker.
    meta_gen: Arc<AtomicU64>,
}

impl Default for RightPanelState {
    fn default() -> Self {
        Self {
            current_tags: String::new(),
            tag_roles: TagRoles::default(),
            roles_cache: None,
            sd_metadata: None,
            sd_metadata_raw: None,
            showing_meta: false,
            downloader: crate::download::DownloaderState::default(),
            civitai: crate::civitai::CivitaiState::default(),
            #[cfg(not(target_os = "macos"))]
            pixal3d: crate::pixal3d::Pixal3DState::default(),
            #[cfg(not(target_os = "macos"))]
            generate: crate::generate::GenerateState::default(),
            #[cfg(not(target_os = "macos"))]
            zimage: crate::generate::GenerateState::new(crate::generate::GenFamily::ZImage),
            #[cfg(not(target_os = "macos"))]
            ltx: crate::generate::GenerateState::new(crate::generate::GenFamily::Ltx),
            #[cfg(not(target_os = "macos"))]
            wan: crate::generate::GenerateState::new(crate::generate::GenFamily::Wan),
            #[cfg(not(target_os = "macos"))]
            sdxl: crate::generate::GenerateState::new(crate::generate::GenFamily::Sdxl),
            #[cfg(not(target_os = "macos"))]
            anima: crate::generate::GenerateState::new(crate::generate::GenFamily::Anima),
            #[cfg(not(target_os = "macos"))]
            krea2: crate::generate::GenerateState::new(crate::generate::GenFamily::Krea2),
            is_editing: false,
            view: RightView::default(),
            show_delete_confirm: false,
            skip_delete_confirm: false,
            last_path: None,
            meta: ImageMeta::default(),
            edit_flash_start: None,
            copy_flash: None,
            save_flash: None,
            meta_rx: None,
            meta_gen: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Actions that the right panel can request the main app to perform
/// (since the panel itself doesn't own the image list).
pub enum RightPanelAction {
    None,
    /// Request to delete the currently selected image and its sidecar.
    DeleteCurrent,
    /// Request to move the currently selected image and its sidecar.
    MoveCurrent,
}

/// Render the right details panel.
pub fn show(
    ui: &mut egui::Ui,
    state: &mut RightPanelState,
    current_image: Option<&Path>,
    confirm_before_delete: &mut bool,
    tag_manager: &mut TagManagerState,
    all_images: &[PathBuf],
) -> RightPanelAction {
    // 0. Non-blocking check to see if our background thread finished calculating
    if let Some(rx) = &state.meta_rx {
        if let Ok(meta) = rx.try_recv() {
            state.meta = meta;
            state.meta_rx = None; // Finished!
        }
    }

    // 1. Sync the text box state when the selected image changes.
    if state.last_path.as_deref() != current_image {
        state.last_path = current_image.map(|p| p.to_path_buf());
        state.is_editing = false;
        state.show_delete_confirm = false; // Reset delete prompt when navigating away
        state.edit_flash_start = None;
        state.copy_flash = None;
        state.save_flash = None;
        state.meta_rx = None; // Cancel any pending task

        if let Some(path) = current_image {
            let txt_path = sidecar_txt(path);
            state.current_tags = std::fs::read_to_string(&txt_path).unwrap_or_default();
            // Look up this image's artist/character roles (by md5) from the shared
            // tag_roles.json, so the tag box can colour those tags.
            state.tag_roles = lookup_tag_roles(&mut state.roles_cache, path);
            // Read embedded SD generation metadata (PNG text chunks / EXIF
            // UserComment) once — both the formatted display text and the raw
            // string the Civitai panel parses. Cheap relative to the decode below.
            let (disp, raw) = crate::sd_metadata::read_both(path);
            state.sd_metadata = disp;
            state.sd_metadata_raw = raw;
            // Default to the .txt tags, but when the image has no tags yet *and*
            // carries embedded generation metadata, open straight to the metadata
            // view so it isn't hidden behind a manual switch. With both present,
            // tags win and the user switches manually.
            state.showing_meta = state.current_tags.trim().is_empty() && state.sd_metadata.is_some();

            // Set temporary loading state for the card
            state.meta = ImageMeta::loading();

            // Spawn a background thread to calculate metadata and heavy colors
            let (tx, rx) = mpsc::channel();
            state.meta_rx = Some(rx);

            // This load's generation. A newer selection bumps the shared counter,
            // letting this thread bail before the expensive decode.
            let generation = state.meta_gen.fetch_add(1, Ordering::SeqCst) + 1;
            let meta_gen = Arc::clone(&state.meta_gen);

            let path_clone = path.to_path_buf();
            let ctx = ui.ctx().clone();

            thread::spawn(move || {
                // Skip the heavy decode entirely if the selection already moved on.
                if meta_gen.load(Ordering::SeqCst) != generation {
                    return;
                }
                let meta = load_meta(&path_clone);
                // Deliver + repaint only if this is still the current selection.
                if meta_gen.load(Ordering::SeqCst) == generation && tx.send(meta).is_ok() {
                    ctx.request_repaint();
                }
            });

        } else {
            state.current_tags.clear();
            state.tag_roles = TagRoles::default();
            state.sd_metadata = None;
            state.sd_metadata_raw = None;
            state.showing_meta = false;
            state.meta = ImageMeta::default();
        }
    }

    let mut action = RightPanelAction::None;

    // 2. Render the actual panel.
    egui::Panel::right("right_details")
        .resizable(false)
        .exact_size(PANEL_WIDTH)
        .show_separator_line(false)
        // Trim the top margin so the card rises up close to the top bar
        // (left/right/bottom keep the standard 10px breathing room).
        .frame(egui::Frame::new().fill(BG()).inner_margin(egui::Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            // Trim the card's bottom padding so the Image Details box can sit
            // close to the bottom edge of the panel (other sides keep 12).
            card_frame(22)
                .inner_margin(egui::Margin { left: 12, right: 12, top: 12, bottom: 12 })
                .show(ui, |ui| {
                    // Force the inner UI to claim the entire vertical height of the panel
                    let height = ui.available_height();
                    ui.set_min_height(height);

                    // Menu icon pinned to the card's top-right corner.
                    // Using ui.put allows us to place it exactly on the right edge
                    // without pushing down the centered text below it.
                    let anchor = ui.cursor().min;
                    let width = ui.available_width();
                    let btn_size = 24.0;

                    let gear_rect = egui::Rect::from_min_size(
                        egui::pos2(anchor.x + width - btn_size, anchor.y),
                        egui::vec2(btn_size, btn_size),
                    );

                    let menu_icon = egui::Image::new(egui::include_image!("../icons/menu.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(MUTED());

                    let menu_resp = ui.put(
                        gear_rect,
                        egui::Button::image(menu_icon).frame(false)
                    ).on_hover_text("Menu");

                    // The fully functional Dropdown
                    egui::Popup::menu(&menu_resp)
                        .align(egui::RectAlign::BOTTOM_END)
                        .frame(
                            egui::Frame::new()
                                .fill(PANEL())
                                .corner_radius(egui::CornerRadius::same(22))
                                .inner_margin(egui::Margin::same(12))
                                .stroke(egui::Stroke::new(1.0, EDGE())),
                        )
                        .show(|ui| {
                            ui.set_min_width(160.0);

                            // Tweak visuals slightly so the menu looks good in dark mode
                            let radius = egui::CornerRadius::same(6);
                            ui.visuals_mut().widgets.inactive.corner_radius = radius;
                            ui.visuals_mut().widgets.hovered.corner_radius = radius;
                            ui.visuals_mut().widgets.active.corner_radius = radius;

                            if ui
                                .selectable_label(state.view == RightView::Details, "Details & Actions")
                                .clicked()
                            {
                                state.view = RightView::Details;
                                ui.close(); // Fixed deprecation
                            }
                            if ui
                                .selectable_label(state.view == RightView::Civitai, "Civitai Resources")
                                .clicked()
                            {
                                state.view = RightView::Civitai;
                                ui.close();
                            }
                            if ui
                                .selectable_label(state.view == RightView::TagManager, "Tag Manager")
                                .clicked()
                            {
                                state.view = RightView::TagManager;
                                ui.close(); // Fixed deprecation
                            }
                            if ui
                                .selectable_label(state.view == RightView::Downloader, "Downloader")
                                .clicked()
                            {
                                state.view = RightView::Downloader;
                                ui.close();
                            }
                            // Pixal3D — Linux/Windows only (hidden on macOS).
                            #[cfg(not(target_os = "macos"))]
                            if ui
                                .selectable_label(state.view == RightView::Pixal3D, "Pixal3D")
                                .clicked()
                            {
                                state.view = RightView::Pixal3D;
                                ui.close();
                            }
                            // One "Text to Image" entry for the four image models
                            // (SDXL / Anima / Flux / Z-Image). It opens SDXL first;
                            // the view's header dropdown switches between them. Stays
                            // highlighted while any of those models is showing.
                            #[cfg(not(target_os = "macos"))]
                            {
                                let t2i_active = matches!(
                                    state.view,
                                    RightView::Sdxl
                                        | RightView::Anima
                                        | RightView::Krea2
                                        | RightView::Generate
                                        | RightView::ZImage
                                );
                                if ui.selectable_label(t2i_active, "Text to Image").clicked() {
                                    if !t2i_active {
                                        state.view = RightView::Sdxl;
                                    }
                                    ui.close();
                                }
                            }
                            // One "Text to Video" entry for the two video Directors
                            // (LTX / Wan). It opens LTX first; the view's header
                            // dropdown switches between them.
                            #[cfg(not(target_os = "macos"))]
                            {
                                let t2v_active = matches!(state.view, RightView::Ltx | RightView::Wan);
                                if ui.selectable_label(t2v_active, "Text to Video").clicked() {
                                    if !t2v_active {
                                        state.view = RightView::Ltx;
                                    }
                                    ui.close();
                                }
                            }
                        });

                    // --- Swap Views ---
                    // The Tag Manager view completely replaces the Details & Actions UI,
                    // but stays constrained perfectly within the 420px width and 22px rounded box.
                    // Pixal3D is Linux/Windows only; on macOS the variant/field don't
                    // exist, so gate the check behind a platform-safe boolean.
                    #[cfg(not(target_os = "macos"))]
                    let show_pixal3d = state.view == RightView::Pixal3D;
                    #[cfg(target_os = "macos")]
                    let show_pixal3d = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_generate = state.view == RightView::Generate;
                    #[cfg(target_os = "macos")]
                    let show_generate = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_zimage = state.view == RightView::ZImage;
                    #[cfg(target_os = "macos")]
                    let show_zimage = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_ltx = state.view == RightView::Ltx;
                    #[cfg(target_os = "macos")]
                    let show_ltx = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_wan = state.view == RightView::Wan;
                    #[cfg(target_os = "macos")]
                    let show_wan = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_sdxl = state.view == RightView::Sdxl;
                    #[cfg(target_os = "macos")]
                    let show_sdxl = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_anima = state.view == RightView::Anima;
                    #[cfg(target_os = "macos")]
                    let show_anima = false;

                    #[cfg(not(target_os = "macos"))]
                    let show_krea2 = state.view == RightView::Krea2;
                    #[cfg(target_os = "macos")]
                    let show_krea2 = false;

                    if show_pixal3d {
                        #[cfg(not(target_os = "macos"))]
                        crate::pixal3d::show(ui, &mut state.pixal3d, current_image);
                    } else if show_generate {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.generate, None);
                            if let Some(f) = state.generate.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if show_zimage {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.zimage, None);
                            if let Some(f) = state.zimage.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if show_ltx {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.ltx, current_image);
                            if let Some(f) = state.ltx.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if show_wan {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.wan, current_image);
                            if let Some(f) = state.wan.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if show_sdxl {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.sdxl, None);
                            if let Some(f) = state.sdxl.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if show_anima {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.anima, None);
                            if let Some(f) = state.anima.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if show_krea2 {
                        #[cfg(not(target_os = "macos"))]
                        {
                            crate::generate::show(ui, &mut state.krea2, None);
                            if let Some(f) = state.krea2.family_switch.take() {
                                state.view = family_view(f);
                            }
                        }
                    } else if state.view == RightView::TagManager {
                        crate::tag_manager::show(ui, tag_manager, current_image, &mut state.current_tags, all_images);
                    } else if state.view == RightView::Downloader {
                        crate::download::show(ui, &mut state.downloader);
                    } else if state.view == RightView::Civitai {
                        crate::civitai::show(ui, &mut state.civitai, current_image, state.sd_metadata_raw.as_deref());
                    } else if let Some(img_path) = current_image {
                        // --- FOOTER SECTION (Strictly Bottom Anchored) ---
                        egui::Panel::bottom("right_footer")
                            .resizable(false)
                            .show_separator_line(false)
                            .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
                            .show_inside(ui, |ui| {

                                ui.add_space(12.0);

                                // 1. Action Buttons — four equal, fixed-size buttons that
                                //    span the full tag-box width edge to edge. Fixed widths
                                //    mean toggling Edit Text/Save never resizes or shifts them.
                                ui.horizontal(|ui| {
                                    let gap = 8.0;
                                    ui.spacing_mut().item_spacing.x = gap;
                                    let btn_w = (ui.available_width() - gap * 3.0) / 4.0;
                                    let size = egui::vec2(btn_w, 35.0);
                                    let label = |t: &str| egui::RichText::new(t).size(15.0);

                                    // Rounder corners than the default (8) for all the buttons.
                                    let r = egui::CornerRadius::same(12);
                                    ui.visuals_mut().widgets.inactive.corner_radius = r;
                                    ui.visuals_mut().widgets.hovered.corner_radius = r;
                                    ui.visuals_mut().widgets.active.corner_radius = r;

                                    // Copy flashes green when it copies tags, or amber
                                    // ("warning") when there's nothing to copy — then fades back.
                                    let mut copy_btn = egui::Button::new(label("Copy"));
                                    if let Some(fill) = flash_fill(ui, state.copy_flash, FLASH_GREEN, FLASH_AMBER) {
                                        copy_btn = copy_btn.fill(fill);
                                    }

                                    if ui
                                        .add_sized(size, copy_btn)
                                        .on_hover_text("Copy tags to clipboard")
                                        .clicked()
                                    {
                                        // Copy whichever text is on screen — tags or
                                        // the embedded generation metadata.
                                        let to_copy = if state.showing_meta {
                                            state.sd_metadata.clone().unwrap_or_default()
                                        } else {
                                            state.current_tags.clone()
                                        };
                                        let ok = !to_copy.trim().is_empty();
                                        if ok {
                                            ui.ctx().copy_text(to_copy);
                                        }
                                        state.copy_flash = Some((Instant::now(), ok));
                                    }

                                    // Edit/Save slot: flashes green when a save succeeds,
                                    // red when the write fails (stays in edit mode to retry).
                                    let mut edit_btn =
                                        egui::Button::new(label(if state.is_editing { "Save" } else { "Edit Text" }));
                                    if let Some(fill) = flash_fill(ui, state.save_flash, FLASH_GREEN, FLASH_RED) {
                                        edit_btn = edit_btn.fill(fill);
                                    }

                                    if ui.add_sized(size, edit_btn).clicked() {
                                        if state.showing_meta {
                                            // The metadata view is read-only. Clicking
                                            // Edit Text here drops to the .txt tags view,
                                            // creating the sidecar file if it doesn't
                                            // exist yet, and enters edit mode so tags can
                                            // be added to a metadata-only image.
                                            let txt_path = sidecar_txt(img_path);
                                            if !txt_path.exists() {
                                                let _ = std::fs::write(&txt_path, &state.current_tags);
                                            }
                                            state.showing_meta = false;
                                            state.is_editing = true;
                                            state.edit_flash_start = Some(Instant::now());
                                            state.save_flash = None;
                                        } else if state.is_editing {
                                            let txt_path = sidecar_txt(img_path);
                                            let ok = std::fs::write(&txt_path, &state.current_tags).is_ok();
                                            if ok {
                                                state.is_editing = false; // saved — back to view mode
                                            }
                                            state.save_flash = Some((Instant::now(), ok));
                                        } else {
                                            state.is_editing = true;
                                            state.edit_flash_start = Some(Instant::now());
                                            state.save_flash = None; // clear any stale save flash
                                        }
                                    }

                                    if ui.add_sized(size, egui::Button::new(label("Move"))).clicked() {
                                        action = RightPanelAction::MoveCurrent;
                                    }

                                    let danger_bg = egui::Color32::from_rgb(180, 40, 40);
                                    let delete_btn =
                                        egui::Button::new(label("Delete").color(egui::Color32::WHITE))
                                            .fill(danger_bg);

                                    if ui.add_sized(size, delete_btn).clicked() {
                                        // Confirm first, unless confirmations are disabled.
                                        if *confirm_before_delete {
                                            state.show_delete_confirm = true;
                                            state.skip_delete_confirm = false; // fresh checkbox
                                        } else {
                                            action = RightPanelAction::DeleteCurrent;
                                        }
                                    }
                                });

                                ui.add_space(10.0);

                                // 2. Image Details Card (Very Bottom) — no trailing
                                //    space so it sits flush near the panel's bottom edge.
                                image_details_section(ui, &state.meta);
                            });

                        // --- MAIN CONTENT SECTION (Stretches to fill middle gap) ---
                        egui::CentralPanel::default()
                            .frame(egui::Frame::NONE.inner_margin(egui::Margin::ZERO))
                            .show_inside(ui, |ui| {

                                // 1. Header
                                ui.vertical_centered(|ui| {
                                    ui.add_space(4.0);
                                    ui.heading(egui::RichText::new("Details & Actions").color(TEXT()).strong());
                                    ui.add_space(8.0);
                                });

                                // 2. Tags / Metadata Label + switch button. When the
                                //    image carries embedded SD generation metadata, a
                                //    small button (right-aligned) toggles the box
                                //    between the .txt tags and the read-only metadata
                                //    view, mirroring terminus2's switch.
                                ui.horizontal(|ui| {
                                    // Tight gap between the icon and the label.
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    let icon = egui::include_image!("../icons/tag.svg");
                                    ui.add(
                                        egui::Image::new(icon)
                                            .fit_to_exact_size(egui::vec2(18.0, 18.0))
                                            .tint(TEXT()),
                                    );
                                    let title = if state.showing_meta { "Metadata" } else { "Tags" };
                                    ui.label(egui::RichText::new(title).color(TEXT()).strong());

                                    // Only offer the switch when there are two views to
                                    // flip between — i.e. the image has both .txt tags
                                    // and embedded metadata. With only one (or neither),
                                    // the box already shows the right thing and there's
                                    // nothing to switch to, so the button is hidden.
                                    let has_tags = !state.current_tags.trim().is_empty();
                                    let has_meta = state.sd_metadata.is_some();
                                    if has_tags && has_meta {
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                let to = if state.showing_meta { "Tags" } else { "Metadata" };
                                                let switch_icon =
                                                    egui::include_image!("../icons/window_switch.svg");
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
                                                    state.showing_meta = !state.showing_meta;
                                                    state.is_editing = false; // leave edit mode on switch
                                                }
                                            },
                                        );
                                    }
                                });
                                ui.add_space(4.0);

                                // 3. Tags / Metadata Text Area. The metadata view is
                                //    always read-only (no editing, no save-back).
                                let showing_meta = state.showing_meta && state.sd_metadata.is_some();
                                let editable = state.is_editing && !showing_meta;
                                let mut display_text = if showing_meta {
                                    state.sd_metadata.clone().unwrap_or_default()
                                } else {
                                    state.current_tags.clone()
                                };

                                // Artist/character colouring for the tag view (not
                                // the metadata view). Cloned for the layouter closure.
                                let artist_set = state.tag_roles.artist.clone();
                                let character_set = state.tag_roles.character.clone();
                                let highlight_roles = !showing_meta
                                    && !(artist_set.is_empty() && character_set.is_empty());
                                let role_color = if editable { TEXT() } else { TEXT().gamma_multiply(0.8) };

                                // The box is a FIXED-size rounded frame that always
                                // fills the remaining panel height — it never resizes
                                // and never moves. Long text (e.g. SD metadata) scrolls
                                // *inside* it via the ScrollArea, while the box and its
                                // rounded corners stay put. The TextEdit is frameless
                                // (`.frame(false)`) so it paints no background of its
                                // own that could scroll with the text — the Frame below
                                // is the only box.
                                let radius = egui::CornerRadius::same(22);

                                // Box background, with the "ready to edit" flash pulsing
                                // it toward the accent just after entering edit mode.
                                let box_fill = edit_flash_fill(ui, state.edit_flash_start);

                                // Lock the box height to the remaining space *before*
                                // building the frame, so its size is independent of the
                                // text it holds.
                                let box_outer_h = ui.available_height();
                                let inner_h = (box_outer_h - 24.0).max(0.0); // minus the 12px margins

                                egui::Frame::new()
                                    .fill(box_fill)
                                    .corner_radius(radius)
                                    .inner_margin(egui::Margin::same(12))
                                    .show(ui, |ui| {
                                        ui.set_height(inner_h); // fixed — never grows with text
                                        ui.set_width(ui.available_width());

                                        egui::ScrollArea::vertical()
                                            .auto_shrink([false, false])
                                            .max_height(inner_h)
                                            .show(ui, |ui| {
                                                // Colour artist (orange) / character
                                                // (green) tags via a custom layouter.
                                                let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                                                    highlight_tags(ui, buf.as_str(), &artist_set, &character_set, role_color, wrap)
                                                };

                                                if editable {
                                                    let mut text_edit = egui::TextEdit::multiline(&mut display_text)
                                                        .desired_width(f32::INFINITY)
                                                        .font(egui::TextStyle::Monospace)
                                                        .frame(egui::Frame::NONE) // the Frame above is the box
                                                        // Fill the whole box so clicking
                                                        // anywhere inside it (not just on
                                                        // the first lines) focuses the
                                                        // editor and places the caret.
                                                        .min_size(egui::vec2(0.0, inner_h));
                                                    if highlight_roles {
                                                        text_edit = text_edit.layouter(&mut layouter);
                                                    }
                                                    ui.add(text_edit);
                                                } else {
                                                    // Display mode: an immutable `&str`
                                                    // buffer ignores every edit, so the
                                                    // text can be highlighted and copied
                                                    // but never changed.
                                                    let meta_color = TEXT().gamma_multiply(0.8);
                                                    // In the metadata view, colour the app
                                                    // stamp ("Clarity TagFlow" green, the
                                                    // version blue) so images made with
                                                    // this app stand out.
                                                    let stamp_meta = showing_meta
                                                        && display_text.contains("Clarity TagFlow");
                                                    let mut stamp_layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap: f32| {
                                                        highlight_app_stamp(ui, buf.as_str(), meta_color, wrap)
                                                    };
                                                    let mut read_only = display_text.as_str();
                                                    let mut text_edit = egui::TextEdit::multiline(&mut read_only)
                                                        .desired_width(f32::INFINITY)
                                                        .font(egui::TextStyle::Monospace)
                                                        .frame(egui::Frame::NONE) // the Frame above is the box
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
                                    state.current_tags = display_text;
                                }
                            });

                    } else {
                        // --- Empty State ---
                        ui.vertical_centered(|ui| {
                            ui.add_space(4.0);
                            ui.heading(egui::RichText::new("Details & Actions").color(TEXT()).strong());
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("No image selected").color(MUTED()).size(13.0));
                        });
                    }
                });
        });

    // --- 3. Delete Confirmation UI (Smaller, Centered Modal) ---
    if state.show_delete_confirm {
        match delete_confirm_dialog(ui.ctx(), "right_panel_confirm_delete", &mut state.skip_delete_confirm) {
            Some(true) => {
                action = RightPanelAction::DeleteCurrent;
                state.show_delete_confirm = false;
                // "Don't ask again" disables (and persists, via Settings) future prompts.
                if state.skip_delete_confirm {
                    *confirm_before_delete = false;
                }
            }
            Some(false) => state.show_delete_confirm = false,
            None => {}
        }
    }

    action
}

// ---------------------------------------------------------------------------
// Image Details card
// ---------------------------------------------------------------------------

const DETAIL_LABEL_W: f32 = 110.0;
const DETAIL_ROW_VPAD: f32 = 3.0;

pub(crate) fn image_details_section(ui: &mut egui::Ui, meta: &ImageMeta) {
    // Swap the heading + icon to match the selection: "Video Info" (video icon)
    // for videos, "GIF Info" (gif icon) for animated GIFs, else "Image Info".
    ui.horizontal(|ui| {
        // Tight gap between the icon and the heading text.
        ui.spacing_mut().item_spacing.x = 4.0;
        let icon = if meta.is_video {
            egui::include_image!("../icons/video_info.svg")
        } else if meta.is_gif {
            egui::include_image!("../icons/gif_info.svg")
        } else {
            egui::include_image!("../icons/image.svg")
        };
        ui.add(
            egui::Image::new(icon)
                .fit_to_exact_size(egui::vec2(18.0, 18.0))
                .tint(TEXT()),
        );
        let heading = if meta.is_video {
            "Video Info"
        } else if meta.is_gif {
            "GIF Info"
        } else {
            "Image Info"
        };
        ui.label(egui::RichText::new(heading).color(TEXT()).strong().size(15.0));
    });
    ui.add_space(8.0);

    let frame = egui::Frame::new()
        .fill(FIELD())
        .corner_radius(egui::CornerRadius::same(22)) // match the tag box
        .inner_margin(egui::Margin::symmetric(16, 12))
        // On the dark themes, borrow the same soft light edge the tag box gets
        // from egui's default field border (a gentle highlight). On the light
        // themes that default stroke is a plainly visible grey outline — drop it
        // there; the field tint separates the card on its own.
        .stroke(if crate::theme::is_light() {
            egui::Stroke::NONE
        } else {
            ui.visuals().widgets.noninteractive.bg_stroke
        });

    frame.show(ui, |ui| {
        ui.set_width(ui.available_width());
        detail_row(ui, "File Name:", &meta.name);
        detail_row(ui, "File Type:", &meta.file_type);
        if meta.is_video {
            // Video-specific facts that are actually useful to see at a glance.
            detail_row(ui, "Resolution:", &meta.dimensions);
            detail_row(ui, "Duration:", &meta.duration);
            detail_row(ui, "Codec:", &meta.codec);
            detail_row(ui, "File Size:", &meta.size);
            detail_row(ui, "Date Modified:", &meta.date);
        } else if meta.is_gif {
            // GIF-specific facts: how many frames and how long it plays. The
            // colour palette is still useful, so keep it too.
            detail_row(ui, "Dimensions:", &meta.dimensions);
            detail_row(ui, "Frames:", &meta.frames);
            detail_row(ui, "Duration:", &meta.duration);
            detail_row(ui, "File Size:", &meta.size);
            detail_row(ui, "Date Modified:", &meta.date);
            detail_color_row(ui, "Colors:", &meta.colors);
        } else {
            detail_row(ui, "Dimensions:", &meta.dimensions);
            detail_row(ui, "File Size:", &meta.size);
            detail_row(ui, "Date Modified:", &meta.date);
            detail_color_row(ui, "Colors:", &meta.colors); // NEW COLOR ROW!
        }
    });
}

fn detail_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.add_space(DETAIL_ROW_VPAD);
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(DETAIL_LABEL_W, 18.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(egui::RichText::new(label).color(TEXT()).strong());
            },
        );

        let unknown = value == "---" || value == "Loading...";
        let color = if unknown { MUTED() } else { TEXT() };
        // A truncated `Label` already shows the full text as a tooltip on hover, so
        // don't add a second `on_hover_text` — that produced two stacked tooltips.
        ui.add(egui::Label::new(egui::RichText::new(value).color(color)).truncate());
    });
    ui.add_space(DETAIL_ROW_VPAD);
}

/// A specialized row that draws small rounded color squares instead of text.
fn detail_color_row(ui: &mut egui::Ui, label: &str, colors: &[egui::Color32]) {
    ui.add_space(DETAIL_ROW_VPAD);
    ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(DETAIL_LABEL_W, 18.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(egui::RichText::new(label).color(TEXT()).strong());
            },
        );

        if colors.is_empty() {
            ui.label(egui::RichText::new("---").color(MUTED()));
        } else {
            // CHANGED HERE: Use horizontal_wrapped so colors wrap to the next line if there are many!
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.spacing_mut().item_spacing.y = 6.0; // Add vertical spacing for wrapped rows
                for &color in colors {
                    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                    let corner_radius = egui::CornerRadius::same(4);
                    ui.painter().rect_filled(rect, corner_radius, color);
                    ui.painter().rect_stroke(
                        rect,
                        corner_radius,
                        egui::Stroke::new(1.0, EDGE()),
                        egui::StrokeKind::Inside,
                    );
                }
            });
        }
    });
    ui.add_space(DETAIL_ROW_VPAD);
}

/// Read the metadata shown in the details card, including extracting dominant colors.
pub(crate) fn load_meta(path: &Path) -> ImageMeta {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("---")
        .to_string();

    let file_type = path
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| !e.is_empty())
        .map(|e| e.to_uppercase())
        .unwrap_or_else(|| "(unknown)".to_string());

    let (size, date) = match std::fs::metadata(path) {
        Ok(md) => {
            let size = human_bytes(md.len());
            let date = md
                .modified()
                .ok()
                .map(format_time)
                .unwrap_or_else(|| "---".to_string());
            (size, date)
        }
        Err(_) => ("---".to_string(), "---".to_string()),
    };

    // Videos: skip the image decode / colour extraction entirely (it can't read a
    // video and would waste time). Read lightweight container facts instead and
    // return a "Video Info" meta.
    if crate::is_video(path) {
        let (dimensions, duration, codec) = read_video_meta(path);
        return ImageMeta {
            name,
            file_type,
            dimensions,
            size,
            date,
            colors: Vec::new(),
            is_video: true,
            is_gif: false,
            duration,
            codec,
            frames: "---".into(),
        };
    }

    // AVIF/HEIC/HEIF/raw can't be read by the `image` crate's header reader or
    // `open()` at all; HDR can but must be tone-mapped to look right. Decode either
    // once via our own path (HDR is always available; the heavy formats only with
    // the `avif` feature) and reuse the result for both the dimensions and the
    // colour palette below.
    let predecoded: Option<image::DynamicImage> = {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "hdr" {
            crate::image_cache::decode_hdr(path).map(image::DynamicImage::ImageRgba8)
        } else if matches!(ext.as_str(), "tif" | "tiff") && image::image_dimensions(path).is_err() {
            // A raw/JPEG-compressed TIFF the `image` crate can't read (rendered
            // image JPEG-compressed in IFD0, raw CFA in a sub-IFD). Recover the
            // embedded camera JPEG so the dimensions and colour palette still show.
            // Normal TIFFs pass `image_dimensions` and skip this (they use the fast
            // header/`open` paths below).
            std::fs::read(path)
                .ok()
                .and_then(|b| crate::raw_preview::largest_embedded_jpeg(&b))
                .map(image::DynamicImage::ImageRgba8)
        } else {
            #[cfg(feature = "avif")]
            {
                if crate::is_extended_extension(&ext) {
                    crate::avif::decode_avif(path).map(image::DynamicImage::ImageRgba8)
                } else {
                    None
                }
            }
            #[cfg(not(feature = "avif"))]
            {
                None
            }
        }
    };

    let dimensions = if let Some(img) = &predecoded {
        format!("{} x {}", img.width(), img.height())
    } else {
        // Use orientation-aware dimensions so a portrait photo (EXIF orientation
        // 6/8) reports the size the user sees, matching the rotated display.
        crate::image_cache::oriented_dimensions(path)
            .map(|(w, h)| format!("{} x {}", w, h))
            .unwrap_or_else(|| "---".to_string())
    };

    // --- COLOR EXTRACTION V3 (Vibrancy & Saturation Weighted) ---
    let mut palette = Vec::new();

    let loaded = match predecoded {
        Some(img) => Some(img),
        None => image::open(path).ok(),
    };
    if let Some(img) = loaded {
        // Bumped to 128x128. We use 'Nearest' filter so thin neon light streaks
        // stay sharp and aren't blurred/blended into the dark background!
        let thumb = img.resize_exact(128, 128, image::imageops::FilterType::Nearest).to_rgba8();

        let mut buckets: std::collections::HashMap<[u8; 3], (u32, u32, u32, u32)> = std::collections::HashMap::new();

        // 1. Group pixels into buckets
        for pixel in thumb.pixels() {
            if pixel[3] < 128 { continue; } // Ignore transparent

            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // Ignore almost pure black and pure white so they don't eat up palette slots
            if r < 15 && g < 15 && b < 15 { continue; }
            if r > 240 && g > 240 && b > 240 { continue; }

            // Quantize by dividing by 16 for slightly better color fidelity
            let bucket_key = [r / 16, g / 16, b / 16];

            let entry = buckets.entry(bucket_key).or_insert((0, 0, 0, 0));
            entry.0 += 1;
            entry.1 += r as u32;
            entry.2 += g as u32;
            entry.3 += b as u32;
        }

        // 2. Score buckets based on a mix of FREQUENCY and COLORFULNESS
        let mut scored_buckets: Vec<_> = buckets.into_values().map(|(count, sr, sg, sb)| {
            let avg_r = (sr / count) as u8;
            let avg_g = (sg / count) as u8;
            let avg_b = (sb / count) as u8;

            // Calculate Chroma (how colorful/saturated the color is)
            let c_max = avg_r.max(avg_g).max(avg_b);
            let c_min = avg_r.min(avg_g).min(avg_b);
            let chroma = (c_max - c_min) as f32; // 0 to 255

            // VIBRANCY MULTIPLIER: Boost the "count" artificially if the color is highly saturated.
            // An exponentially weighted curve means a neon color gets up to a
            // massive multiplier compared to a dull gray!
            let weight = 1.0 + (chroma * chroma / 1000.0);

            // We also take the square root of the count so massive walls of background
            // color don't completely drown out smaller details
            let score = (count as f32).sqrt() * weight;

            (score, avg_r, avg_g, avg_b)
        }).collect();

        // Sort by our new score (highest score first)
        scored_buckets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut distinct_colors: Vec<[u8; 3]> = Vec::new();

        // 3. Extract TRUE average color, filtering out similar ones
        for (_score, avg_r, avg_g, avg_b) in scored_buckets {
            let mut is_distinct = true;

            for &c in &distinct_colors {
                let dr = avg_r as i32 - c[0] as i32;
                let dg = avg_g as i32 - c[1] as i32;
                let db = avg_b as i32 - c[2] as i32;

                // Perceptual distance formula (greens and reds matter more to human eyes than blue)
                let dist_sq = (dr * dr * 3) + (dg * dg * 4) + (db * db * 2);

                // Increased threshold to force maximum variety!
                // It will now actively refuse to pick 5 shades of the same dark blue.
                if dist_sq < 4000 {
                    is_distinct = false;
                    break;
                }
            }

            if is_distinct {
                distinct_colors.push([avg_r, avg_g, avg_b]);
                if distinct_colors.len() == 16 {
                    break;
                }
            }
        }

        // 4. Group final palette into a gorgeous rainbow layout (Hue, then Luminance)
        distinct_colors.sort_by_key(|c| {
            let r = c[0] as f32;
            let g = c[1] as f32;
            let b = c[2] as f32;

            // Rough Hue calculation
            let hue = if r >= g && r >= b { (g - b) / (r - g.min(b)).max(1.0) }
            else if g >= r && g >= b { 2.0 + (b - r) / (g - r.min(b)).max(1.0) }
            else { 4.0 + (r - g) / (b - r.min(g)).max(1.0) };

            let wrapped_hue = ((hue * 60.0) as i32).rem_euclid(360);
            let luminance = (r * 0.299 + g * 0.587 + b * 0.114) as i32;

            // Sort by Hue group, then brightness
            (wrapped_hue / 30, luminance)
        });

        palette = distinct_colors.into_iter().map(|c| {
            egui::Color32::from_rgb(c[0], c[1], c[2])
        }).collect();
    }

    // Animated GIFs get their frame count + play time read cheaply from the
    // container (no pixel decode); the dimensions and palette above still apply.
    let (is_gif, gif_frames, gif_duration) = read_gif_meta(path);

    ImageMeta {
        name,
        file_type,
        dimensions,
        size,
        date,
        colors: palette,
        is_video: false,
        is_gif,
        duration: gif_duration,
        codec: "---".into(),
        frames: gif_frames,
    }
}

/// Read animated-GIF facts: `(is_animated_gif, "N frames", "M:SS")`. A
/// single-frame (static) GIF reports `is_gif = false` so it stays in the normal
/// "Image Info" view. Non-GIFs return all-default.
fn read_gif_meta(path: &Path) -> (bool, String, String) {
    let is_gif_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gif"))
        .unwrap_or(false);
    if !is_gif_ext {
        return (false, "---".into(), "---".into());
    }
    match crate::gif_info::probe(path) {
        // Only treat it as a GIF "animation" when it has more than one frame.
        Some(info) if info.frames > 1 => {
            let frames = format!("{} frames", info.frames);
            let duration = info.duration_secs.map(format_duration).unwrap_or_else(|| "---".into());
            (true, frames, duration)
        }
        _ => (false, "---".into(), "---".into()),
    }
}

/// Read useful video facts — resolution, duration, codec — without external
/// tools. Parses the ISO base-media (MP4 / MOV / M4V) box structure, reading only
/// the small `moov` header rather than the whole (often huge) file. Containers we
/// can't parse (MKV / WebM / AVI / …) return "---" for the unknown fields, so the
/// card still shows the file name, type, size and date.
fn read_video_meta(path: &Path) -> (String, String, String) {
    let unknown = || ("---".to_string(), "---".to_string(), "---".to_string());
    match crate::mp4::probe(path) {
        Some(info) => {
            let resolution = match (info.width, info.height) {
                (Some(w), Some(h)) if w > 0 && h > 0 => format!("{w} x {h}"),
                _ => "---".to_string(),
            };
            let duration = info.duration_secs.map(format_duration).unwrap_or_else(|| "---".into());
            let codec = info.codec.unwrap_or_else(|| "---".into());
            (resolution, duration, codec)
        }
        None => unknown(),
    }
}

/// Format a duration in seconds as `H:MM:SS` (or `M:SS` under an hour).
fn format_duration(secs: f64) -> String {
    let total = secs.round().max(0.0) as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

fn human_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{} B", bytes);
    }
    let kb = bytes as f64 / 1024.0;
    if kb < 1024.0 {
        return format!("{:.1} KB", kb);
    }
    let mb = kb / 1024.0;
    if mb < 1024.0 {
        return format!("{:.1} MB", mb);
    }
    let gb = mb / 1024.0;
    format!("{:.1} GB", gb)
}

pub(crate) fn sidecar_txt(img_path: &Path) -> PathBuf {
    img_path.with_extension("txt")
}

/// Tag-role highlight colours (Danbooru-style): artist orange, character green.
const ARTIST_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 150, 50);
const CHARACTER_COLOR: egui::Color32 = egui::Color32::from_rgb(80, 200, 120);

/// Look up an image's artist/character roles by its md5 (the downloaded file's
/// stem) from the shared `tag_roles.json`. The parsed map is cached and only
/// re-read when the file's mtime changes, so rapid selection changes don't
/// re-parse it. Returns empty roles for images not in the map.
pub(crate) fn lookup_tag_roles(
    cache: &mut Option<(Option<SystemTime>, std::collections::HashMap<String, TagRoles>)>,
    img_path: &Path,
) -> TagRoles {
    let path = crate::download::tag_roles_path();
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    // (Re)load the map when the cache is empty or the file changed on disk.
    let stale = cache.as_ref().map(|(m, _)| *m != mtime).unwrap_or(true);
    if stale {
        let map = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.as_object().cloned())
            .map(|obj| {
                let set = |entry: &serde_json::Value, key: &str| -> std::collections::HashSet<String> {
                    entry
                        .get(key)
                        .and_then(|x| x.as_array())
                        .map(|arr| arr.iter().filter_map(|e| e.as_str().map(str::to_string)).collect())
                        .unwrap_or_default()
                };
                obj.into_iter()
                    .map(|(md5, entry)| {
                        (md5, TagRoles { artist: set(&entry, "artist"), character: set(&entry, "character") })
                    })
                    .collect()
            })
            .unwrap_or_default();
        *cache = Some((mtime, map));
    }
    let key = img_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    cache
        .as_ref()
        .and_then(|(_, map)| map.get(key))
        .cloned()
        .unwrap_or_default()
}

/// Build a colour-highlighted layout for the (comma-separated) tag text: artist
/// tags orange, character tags green, everything else `default_color`.
pub(crate) fn highlight_tags(
    ui: &egui::Ui,
    text: &str,
    artist: &std::collections::HashSet<String>,
    character: &std::collections::HashSet<String>,
    default_color: egui::Color32,
    wrap_width: f32,
) -> std::sync::Arc<egui::Galley> {
    let font_id = egui::TextStyle::Monospace.resolve(ui.style());
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = wrap_width;
    // Keep the commas with their token so spacing/positions are preserved exactly.
    for piece in text.split_inclusive(',') {
        let name = piece.trim().trim_end_matches(',').trim();
        let color = if artist.contains(name) {
            ARTIST_COLOR
        } else if character.contains(name) {
            CHARACTER_COLOR
        } else {
            default_color
        };
        job.append(
            piece,
            0.0,
            egui::TextFormat { font_id: font_id.clone(), color, ..Default::default() },
        );
    }
    ui.fonts_mut(|f| f.layout_job(job))
}

/// Colour of the "Clarity TagFlow" app name in the metadata view (green) and of
/// the version number after it (blue) — the stamp generated images carry, so
/// it's obvious at a glance an image was made with this app.
const STAMP_NAME_COLOR: egui::Color32 = egui::Color32::from_rgb(46, 160, 67);
const STAMP_VERSION_COLOR: egui::Color32 = egui::Color32::from_rgb(83, 156, 255);

/// Layouter for the metadata view: paints every "Clarity TagFlow" occurrence
/// green and a following "vX.Y.Z" version token blue; everything else keeps
/// `default_color`. Positions/wrapping are unchanged (text is appended verbatim),
/// so selection and copying still line up exactly.
pub(crate) fn highlight_app_stamp(
    ui: &egui::Ui,
    text: &str,
    default_color: egui::Color32,
    wrap_width: f32,
) -> std::sync::Arc<egui::Galley> {
    const APP: &str = "Clarity TagFlow";
    let font_id = egui::TextStyle::Monospace.resolve(ui.style());
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = wrap_width;
    let fmt = |color: egui::Color32| egui::TextFormat {
        font_id: font_id.clone(),
        color,
        ..Default::default()
    };

    let mut rest = text;
    while let Some(pos) = rest.find(APP) {
        job.append(&rest[..pos], 0.0, fmt(default_color));
        job.append(APP, 0.0, fmt(STAMP_NAME_COLOR));
        rest = &rest[pos + APP.len()..];
        // A " vX.Y…" version token right after the name turns blue.
        if let Some(stripped) = rest.strip_prefix(' ') {
            let is_version = stripped.starts_with('v')
                && stripped[1..].starts_with(|c: char| c.is_ascii_digit());
            if is_version {
                let end = stripped
                    .find(char::is_whitespace)
                    .unwrap_or(stripped.len());
                job.append(" ", 0.0, fmt(default_color));
                job.append(&stripped[..end], 0.0, fmt(STAMP_VERSION_COLOR));
                rest = &stripped[end..];
            }
        }
    }
    job.append(rest, 0.0, fmt(default_color));
    ui.fonts_mut(|f| f.layout_job(job))
}

fn format_time(t: SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = t.into();
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Duration (seconds) of the "ready to edit" flash on the tag box.
const EDIT_FLASH_SECS: f32 = 0.8;

/// Duration (seconds) of the button result-flashes (Copy / Save).
const FLASH_SECS: f32 = 1.0;
/// Button flash colors: green = success, amber = nothing to do, red = failure.
pub(crate) const FLASH_GREEN: egui::Color32 = egui::Color32::from_rgb(46, 160, 67);
pub(crate) const FLASH_AMBER: egui::Color32 = egui::Color32::from_rgb(200, 145, 40);
pub(crate) const FLASH_RED: egui::Color32 = egui::Color32::from_rgb(200, 55, 55);

/// Fill for a button result-flash, fading from `ok`/`fail` back to the normal
/// button colour over `FLASH_SECS`. `None` once the flash has expired (or there
/// is no flash) — leave the button's default fill alone then. Shared by the
/// right panel and the gallery detail popup.
pub(crate) fn flash_fill(
    ui: &egui::Ui,
    flash: Option<(Instant, bool)>,
    ok: egui::Color32,
    fail: egui::Color32,
) -> Option<egui::Color32> {
    let (start, was_ok) = flash?;
    let elapsed = start.elapsed().as_secs_f32();
    if elapsed >= FLASH_SECS {
        return None;
    }
    let intensity = 1.0 - elapsed / FLASH_SECS;
    let target = if was_ok { ok } else { fail };
    let base = ui.visuals().widgets.inactive.weak_bg_fill;
    ui.ctx().request_repaint(); // animate the fade
    Some(lerp_color(base, target, intensity))
}

/// The tag box's fill, with the "ready to edit" flash pulsing it toward the
/// accent just after entering edit mode. Shared by the right panel and the
/// gallery detail popup.
pub(crate) fn edit_flash_fill(ui: &egui::Ui, start: Option<Instant>) -> egui::Color32 {
    let mut fill = FIELD();
    if let Some(start) = start {
        let elapsed = start.elapsed().as_secs_f32();
        if elapsed < EDIT_FLASH_SECS {
            let t = elapsed / EDIT_FLASH_SECS;          // 0..1
            let envelope = 1.0 - t;                     // overall fade-out
            let osc = (t * std::f32::consts::PI * 2.0).sin().abs(); // two pulses
            let intensity = (envelope * osc).clamp(0.0, 1.0);
            fill = lerp_color(FIELD(), ACCENT1(), intensity * 0.55);
            ui.ctx().request_repaint(); // keep the animation smooth
        }
    }
    fill
}

/// The delete-confirmation modal (warning icon, "Don't ask again" checkbox,
/// Cancel / Delete) — shared by the right panel and the gallery detail popup.
/// Returns `Some(true)` when Delete is clicked, `Some(false)` on Cancel, and
/// `None` while the dialog stays open.
pub(crate) fn delete_confirm_dialog(
    ctx: &egui::Context,
    id: &str,
    skip_confirm: &mut bool,
) -> Option<bool> {
    let mut result = None;

    egui::Window::new("Confirm Delete")
        .id(egui::Id::new(id))
        .title_bar(false) // No title bar to keep the UI small and clean
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO) // Anchor directly in the middle of the screen
        .frame(card_frame(22)) // match the rest of the UI (PANEL() fill, radius 22, shadow)
        .show(ctx, |ui| {
            ui.set_max_width(260.0); // Stop it from stretching too wide

            // Wrap inner elements in vertical_centered so text and buttons line up perfectly
            ui.vertical_centered(|ui| {
                let warn = egui::include_image!("../icons/warning.svg");
                ui.add(
                    egui::Image::new(warn)
                        .fit_to_exact_size(egui::vec2(32.0, 32.0))
                        .tint(egui::Color32::from_rgb(220, 160, 50)),
                );
                ui.add_space(8.0);

                ui.label(
                    egui::RichText::new("Are you sure you want to delete this file?")
                        .size(15.0)
                        .strong()
                        .color(TEXT())
                );

                ui.add_space(16.0);

                // Scope for checkbox to ensure it acts as a true square
                // with slightly rounded 4px edges (instead of inheriting 12px pill curves)
                ui.scope(|ui| {
                    let r = egui::CornerRadius::same(4);
                    ui.visuals_mut().widgets.inactive.corner_radius = r;
                    ui.visuals_mut().widgets.hovered.corner_radius = r;
                    ui.visuals_mut().widgets.active.corner_radius = r;
                    ui.visuals_mut().widgets.noninteractive.corner_radius = r;

                    ui.checkbox(skip_confirm, "Don't ask again");
                });

                ui.add_space(20.0);

                ui.horizontal(|ui| {
                    let btn_w = 80.0;
                    let gap = 12.0;
                    let total_w = btn_w * 2.0 + gap;

                    // Push the buttons perfectly to the middle of the layout
                    ui.add_space((ui.available_width() - total_w) / 2.0);
                    ui.spacing_mut().item_spacing.x = gap;

                    // Add matching soft-corners to the inner action buttons
                    let r = egui::CornerRadius::same(8);
                    ui.visuals_mut().widgets.inactive.corner_radius = r;
                    ui.visuals_mut().widgets.hovered.corner_radius = r;
                    ui.visuals_mut().widgets.active.corner_radius = r;

                    if ui.add_sized(egui::vec2(btn_w, 30.0), egui::Button::new("Cancel")).clicked() {
                        result = Some(false);
                    }

                    let danger_bg = egui::Color32::from_rgb(180, 40, 40);
                    let del_btn = egui::Button::new(
                        egui::RichText::new("Delete").color(egui::Color32::WHITE)
                    ).fill(danger_bg);

                    if ui.add_sized(egui::vec2(btn_w, 30.0), del_btn).clicked() {
                        result = Some(true);
                    }
                });
            });
        });

    result
}

/// Linearly interpolate between two colors in sRGB component space.
fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    egui::Color32::from_rgb(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
    )
}