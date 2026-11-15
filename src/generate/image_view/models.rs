//! The Model picker's list as picture tiles, like the left panel's: one
//! rounded tile per model down a column, the chosen one ringed in the
//! selection colour, the scroll bar out in the margin. Each tile has the
//! model's name on it in blue and, top left, what it's for ("Flux",
//! "Krea 2"). Favorites (right-click) come first, with the left panel's
//! pulsing heart top right.
//!
//! A tile's picture is, first, one the user set (right-click) or the
//! model's own preview on Civitai (looked up by the file's hash, for a
//! model on this PC); else the first image made with it here; else a dark
//! tile in its family's colour.

use super::*;

/// The gap between tiles and their corners: the left panel's.
pub(super) const TILE_GAP: f32 = 12.0;
pub(super) const CORNER: u8 = 12;
/// A tile's height for its width.
pub(super) const TILE_ASPECT: f32 = 0.56;
/// The list's margin, which the scroll bar rides.
pub(super) const MARGIN: i8 = 12;
pub(super) const CAPTION_H: f32 = 18.0;
pub(super) const CAPTION_GAP: f32 = 8.0;
pub(super) const FOOTER_H: f32 = 32.0;
const BADGE_H: f32 = 20.0;
/// The names' blue: the dark themes' accent, a step lighter, for it sits on
/// a picture's dark foot in every theme.
pub(super) const NAME_BLUE: Color32 = Color32::from_rgb(110, 170, 255);
/// The longest side a tile's picture is kept at (a tile is ~350 wide, at up
/// to twice that in pixels).
const PICTURE_EDGE: u32 = 768;
/// The loading ring's radius (room for "100%" inside).
const RING_R: f32 = 18.0;

/// What a tile picks.
#[derive(Clone, PartialEq, Debug)]
pub(in crate::generate) enum Pick {
    Builtin(GenModel),
    Checkpoint(String),
}

impl Pick {
    /// Its pictures' name in the thumbnails folder.
    pub(in crate::generate) fn key(&self) -> String {
        match self {
            Pick::Builtin(m) => format!("builtin-{m:?}"),
            Pick::Checkpoint(file) => format!("ckpt-{}", ckpt_display_name(file)),
        }
    }
}

/// The model the view has now.
pub(in crate::generate) fn current(state: &GenerateState) -> Pick {
    match &state.checkpoint {
        Some(file) => Pick::Checkpoint(file.clone()),
        None => Pick::Builtin(state.model),
    }
}

/// The tile a generation's first image is for: the model picked — not for
/// video, nor for an imported workflow (which picks its own model).
pub(in crate::generate) fn run_key(state: &GenerateState) -> Option<String> {
    (!state.family.is_video() && state.imported_workflow.is_none()).then(|| current(state).key())
}

/// A tile's picture as last loaded, and the file it came from as it was
/// then (a new file, or a changed one, is loaded again).
#[derive(Default)]
pub(in crate::generate) struct Thumb {
    from: Option<(PathBuf, Option<std::time::SystemTime>)>,
    pub(in crate::generate) tex: Option<egui::TextureHandle>,
}

/// Where the tiles' pictures are kept: `<key>.img` (Civitai's preview, or
/// one the user set), `<key>.none` (Civitai has none, or the user took the
/// picture away) and `<key>.gen.jpg` (the first image made with it).
fn thumbs_dir() -> PathBuf {
    #[cfg(test)]
    if let Some(dir) = THUMBS_DIR_FOR_TEST.with_borrow(|dir| dir.clone()) {
        return dir;
    }
    comfy_base().join("model_thumbs")
}

/// The picture a tile shows, if it has one.
fn picture(key: &str) -> Option<PathBuf> {
    let dir = thumbs_dir();
    [format!("{key}.img"), format!("{key}.gen.jpg")].into_iter().map(|name| dir.join(name)).find(|p| p.is_file())
}

/// The tile's picture, loaded once (again when its file changes) — a few a
/// frame, so the list opens at once and they come in as it's shown.
fn texture(ui: &egui::Ui, thumbs: &mut std::collections::HashMap<String, Thumb>, key: &str) -> Option<egui::TextureHandle> {
    let file = picture(key).map(|p| {
        let stamp = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
        (p, stamp)
    });
    let slot = thumbs.entry(key.to_string()).or_default();
    if slot.from != file {
        match &file {
            None => *slot = Thumb::default(),
            // (The old one stays up until the new one's in.)
            Some(_) if !take_thumb_decode_budget(ui.ctx()) => {}
            Some((path, _)) => {
                slot.tex = std::fs::read(path)
                    .ok()
                    .and_then(|bytes| image::load_from_memory(&bytes).ok())
                    .map(|picture| {
                        let picture = shrunk(picture).to_rgba8();
                        let size = [picture.width() as usize, picture.height() as usize];
                        let image = egui::ColorImage::from_rgba_unmultiplied(size, picture.as_raw());
                        ui.ctx().load_texture(format!("model_thumb_{key}"), image, Default::default())
                    });
                slot.from = file;
            }
        }
    }
    slot.tex.clone()
}

/// `picture` no larger than a tile needs (a smaller one is kept as it is).
fn shrunk(picture: image::DynamicImage) -> image::DynamicImage {
    if picture.width().max(picture.height()) > PICTURE_EDGE {
        picture.thumbnail(PICTURE_EDGE, PICTURE_EDGE)
    } else {
        picture
    }
}

/// One row of the list.
enum Row {
    Caption(&'static str),
    Tile(Tile),
}

impl Row {
    fn height(&self, tile_h: f32) -> f32 {
        match self {
            Row::Caption(_) => CAPTION_H,
            Row::Tile(_) => tile_h,
        }
    }
}

struct Tile {
    pick: Pick,
    name: String,
    note: Option<String>,
    /// What it's for, top left.
    badge: Option<&'static str>,
    /// A favorite: listed first, with a heart.
    fave: bool,
    /// Its family's colour, for a tile without a picture.
    tint: Color32,
}

/// The family's colour (as the LoRA picker marks it).
fn family_color(family: GenFamily) -> Color32 {
    match family {
        GenFamily::Flux => LoraBase::Flux,
        GenFamily::ZImage => LoraBase::ZImage,
        GenFamily::Sdxl => LoraBase::Sdxl,
        GenFamily::Anima => LoraBase::Anima,
        GenFamily::Krea2 => LoraBase::Krea2,
        GenFamily::Ltx => LoraBase::Ltx,
        GenFamily::Wan => LoraBase::Wan,
    }
    .dot_color()
}

/// The family's built-in models, then its installed checkpoints (this
/// family's, and others' when asked for), and whether there are others.
fn rows(state: &GenerateState) -> (Vec<Row>, bool) {
    let family = state.family;
    let mut rows = Vec::new();
    for &m in GenModel::all_for(family) {
        let missing_on_host = match (&state.remote_host, &state.remote_catalog) {
            (Some(host), Some(cat)) if !cat.ready.contains(&m) => Some(format!("not set up on {host}")),
            _ => None,
        };
        let (name, size) = short_label(m.label(), family.title());
        rows.push(Row::Tile(Tile {
            pick: Pick::Builtin(m),
            name,
            note: missing_on_host.or(size),
            badge: Some(family.title()),
            fave: false,
            tint: family_color(family),
        }));
    }
    if !family.uses_checkpoint_picker() {
        return (faves_first(state, rows), false);
    }
    // (The built-in ones are above already.)
    let builtin = family.builtin_ckpt_files();
    let installed = state.checkpoints.iter().filter(|c| !builtin.contains(&c.file.as_str()));
    let (ours, others): (Vec<_>, Vec<_>) = installed.partition(|c| c.base.matches(family));
    let shown_others = if state.show_other_ckpts { others.len() } else { 0 };
    let shown: Vec<_> = ours.iter().chain(others.iter().take(shown_others)).collect();
    if !shown.is_empty() {
        rows.push(Row::Caption("Installed checkpoints"));
    }
    for c in shown {
        let known = c.base != LoraBase::Unknown;
        rows.push(Row::Tile(Tile {
            pick: Pick::Checkpoint(c.file.clone()),
            name: ckpt_display_name(&c.file).to_string(),
            note: None,
            badge: known.then(|| c.base.label()),
            fave: false,
            tint: if known { c.base.dot_color() } else { family_color(family) },
        }));
    }
    (faves_first(state, rows), !others.is_empty())
}

/// `rows` with the favorites taken out to the top, under their own caption
/// (and the rest's first group under one too, so the two read apart).
fn faves_first(state: &GenerateState, rows: Vec<Row>) -> Vec<Row> {
    let (faves, rest): (Vec<Row>, Vec<Row>) = rows
        .into_iter()
        .map(|mut row| {
            if let Row::Tile(tile) = &mut row {
                tile.fave = is_fave(state, &fave_id("model", &tile.pick.key()));
            }
            row
        })
        .partition(|row| matches!(row, Row::Tile(tile) if tile.fave));
    if faves.is_empty() {
        return rest;
    }
    let mut out = vec![Row::Caption("Favorites")];
    out.extend(faves);
    if matches!(rest.first(), Some(Row::Tile(_))) {
        out.push(Row::Caption("Built-in"));
    }
    out.extend(rest);
    // (A caption whose tiles all went up goes too.)
    let mut tidy: Vec<Row> = Vec::new();
    for row in out {
        if matches!(row, Row::Caption(_)) && matches!(tidy.last(), Some(Row::Caption(_))) {
            tidy.pop();
        }
        tidy.push(row);
    }
    if matches!(tidy.last(), Some(Row::Caption(_))) {
        tidy.pop();
    }
    tidy
}

// --- Favorites: models', LoRAs' and embeddings', kept together. ---

/// A favorite's id: what it is and its name (`model:builtin-DevQ4`,
/// `lora:detail.safetensors`).
pub(in crate::generate) fn fave_id(kind: &str, name: &str) -> String {
    format!("{kind}:{name}")
}

/// The favorites file: one id a line, beside the models' pictures.
fn faves_file() -> PathBuf {
    thumbs_dir().join("favorites.txt")
}

fn read_faves() -> Vec<String> {
    std::fs::read_to_string(faves_file())
        .map(|text| text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect())
        .unwrap_or_default()
}

/// Read the favorites in (each time a list opens: another tab may have
/// changed them).
pub(in crate::generate) fn load_faves(state: &mut GenerateState) {
    state.faves = Some(read_faves());
}

pub(in crate::generate) fn is_fave(state: &GenerateState, id: &str) -> bool {
    state.faves.as_ref().is_some_and(|faves| faves.iter().any(|f| f == id))
}

/// Make `id` a favorite, or not. (From the file as it is now, so another
/// tab's change since isn't lost.)
pub(in crate::generate) fn toggle_fave(state: &mut GenerateState, id: &str) {
    let mut faves = read_faves();
    match faves.iter().position(|f| f == id) {
        Some(i) => {
            faves.remove(i);
        }
        None => faves.push(id.to_string()),
    }
    if std::fs::create_dir_all(thumbs_dir()).is_ok() {
        let _ = std::fs::write(faves_file(), faves.join("\n") + "\n");
    }
    state.faves = Some(faves);
}

/// The left panel's heart, pulsing gently, centred on `center`.
pub(super) fn heart(ui: &egui::Ui, center: egui::Pos2) {
    const BASE: f32 = 18.0;
    let t = (ui.input(|i| i.time) % 2.0) / 2.0;
    let size = BASE * (1.0 + 0.1 * (1.0 - (t * std::f64::consts::TAU).cos()) as f32);
    let badge = egui::Rect::from_center_size(center, egui::vec2(size, size));
    crate::left_browser::paint_pulsing_heart(ui, egui::include_image!("../../../icons/heart.svg"), badge, BASE);
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
}

/// Where a tile's heart goes: its top right corner.
pub(super) fn heart_at(rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(rect.right() - 18.0, rect.top() + 18.0)
}

/// The favorite item of a right-click menu; true when it's clicked.
pub(super) fn fave_item(ui: &mut egui::Ui, fave: bool) -> bool {
    let (icon, label) = if fave {
        (egui::include_image!("../../../icons/heart_minus.svg"), "Remove Favorite")
    } else {
        (egui::include_image!("../../../icons/heart_plus.svg"), "Favorite")
    };
    crate::zoom::menu_item(ui, icon, label)
}

/// The open list, hung from the whole bar (`anchor`). It closes on a pick,
/// on Escape, and on a click outside it and the bar — but not on one in a
/// tile's right-click menu. `opening`: it opens this frame (at the chosen
/// model).
pub(super) fn list(anchor: &egui::Response, id: egui::Id, state: &mut GenerateState, opening: bool) {
    let ctx = anchor.ctx.clone();
    let bar = anchor.rect;
    let margin = f32::from(MARGIN);
    // As far down as the window goes.
    let room = ctx.content_rect().bottom() - bar.bottom() - 6.0 - 12.0 - 2.0 * margin;
    // (egui's one pop-up is a tile's menu, if one was open as this began.)
    let menu_was_open = egui::Popup::is_any_open(&ctx);
    let mut open = true;
    let mut picked = None;
    if opening || state.faves.is_none() {
        load_faves(state);
    }
    let shown = egui::Popup::from_response(anchor)
        .id(id)
        .open_bool(&mut open)
        .align(egui::RectAlign::BOTTOM_START)
        .gap(6.0)
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .frame(crate::settings::dropdown_frame().inner_margin(MARGIN).corner_radius(22))
        .show(|ui| {
            let width = bar.width() - 2.0 * margin;
            ui.set_width(width);
            THUMB_DECODE_BUDGET.store(2, Ordering::Relaxed);
            picked = tiles(ui, state, width, room.max(160.0), opening);
        });
    let was_at = ctx.data(|d| d.get_temp::<egui::Rect>(id.with("rect")));
    let outside = shown.is_some_and(|shown| {
        let rect = shown.response.rect;
        ctx.data_mut(|d| d.insert_temp(id.with("rect"), rect));
        // (Grown or shrunk — other families shown — the frame's done again at
        // its new size rather than shown a frame askew.)
        if was_at.is_some_and(|was| (was.size() - rect.size()).length() > 0.5) {
            ctx.request_discard("the list changed size");
            // (And again until it settles: what's in it can lag a frame too.)
            ctx.request_repaint();
        }
        ctx.input(|i| {
            i.pointer.any_click()
                && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p) && !bar.contains(p))
        })
    });
    if let Some(pick) = picked {
        choose(state, pick);
        open = false;
    }
    if !open || (outside && !menu_was_open) {
        state.model_list_open = false;
    }
}

/// The tiles (scrolling past `max_h`) and, when there are other families'
/// checkpoints, the switch that shows them. The tile clicked, if one was.
fn tiles(ui: &mut egui::Ui, state: &mut GenerateState, width: f32, max_h: f32, opening: bool) -> Option<Pick> {
    let (rows, others) = rows(state);
    let tile_h = (width * TILE_ASPECT).round();
    let chosen = current(state);
    let mut content_h = 0.0;
    let mut chosen_at = 0.0;
    for (i, row) in rows.iter().enumerate() {
        content_h += gap_before(&rows, i);
        if matches!(row, Row::Tile(tile) if tile.pick == chosen) {
            chosen_at = content_h;
        }
        content_h += row.height(tile_h);
    }
    let footer_h = if others { TILE_GAP + 1.0 + 8.0 + FOOTER_H } else { 0.0 };
    // At least one tile, however short the window.
    let scroll_h = content_h.min((max_h - footer_h).max(tile_h));

    let mut picked = None;
    let gutter = f32::from(MARGIN);
    let mut wide = crate::edge_scroll_ui(ui, gutter);
    let mut scroll = egui::ScrollArea::vertical();
    if opening {
        // It opens with the chosen model in sight — from the top (the
        // favorites) when it's in sight there — not where it was last left
        // (that's kept, even between runs of the app).
        scroll = scroll.vertical_scroll_offset((chosen_at + tile_h - scroll_h).max(0.0));
    }
    scroll
        .id_salt(("t2i_model_tiles", state.family.title()))
        // (Its height is settled here, not by the pop-up's first sizing.)
        .min_scrolled_height(scroll_h)
        .max_height(scroll_h)
        .auto_shrink([false, true])
        .show(&mut wide, |ui| {
            ui.set_width(width);
            ui.spacing_mut().item_spacing.y = 0.0;
            for (i, row) in rows.iter().enumerate() {
                ui.add_space(gap_before(&rows, i));
                match row {
                    Row::Caption(text) => caption(ui, text),
                    Row::Tile(tile) => {
                        if tile_ui(ui, state, tile, tile_h) {
                            picked = Some(tile.pick.clone());
                        }
                    }
                }
            }
        });
    crate::edge_scroll_done(ui, &wide, gutter);

    if others {
        ui.add_space(TILE_GAP);
        let (line, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
        ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0_f32, hairline()));
        ui.add_space(8.0);
        let (row, _) = ui.allocate_exact_size(egui::vec2(width, FOOTER_H), egui::Sense::hover());
        let mut row_ui = at(ui, row);
        row_ui.horizontal_centered(|ui| {
            ui.add_space(4.0);
            ui.label(RichText::new("Show other-family checkpoints").font(medium(12.5)).color(TEXT()));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                crate::settings::switch(ui, &mut state.show_other_ckpts);
            });
        });
    }
    picked
}

/// The space above row `i`: a caption sits closer to its tiles.
fn gap_before(rows: &[Row], i: usize) -> f32 {
    match i {
        0 => 0.0,
        _ if matches!(rows[i - 1], Row::Caption(_)) => CAPTION_GAP,
        _ => TILE_GAP,
    }
}

pub(super) fn caption(ui: &mut egui::Ui, text: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), CAPTION_H), egui::Sense::hover());
    ui.painter().text(
        egui::pos2(rect.left() + 2.0, rect.bottom()),
        egui::Align2::LEFT_BOTTOM,
        text,
        medium(12.0),
        MUTED(),
    );
}

/// Make `pick` the model.
fn choose(state: &mut GenerateState, pick: Pick) {
    match pick {
        Pick::Builtin(m) => {
            state.model = m;
            state.checkpoint = None;
        }
        // A checkpoint runs through the family's default model's set-up.
        Pick::Checkpoint(file) => {
            state.model = state.family.default_model();
            state.checkpoint = Some(file);
        }
    }
    state.steps = state.model.default_steps();
    state.cfg = state.model.default_cfg();
}

/// One tile; true when it's clicked.
fn tile_ui(ui: &mut egui::Ui, state: &mut GenerateState, tile: &Tile, height: f32) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let key = tile.pick.key();
    if ui.is_rect_visible(rect) {
        let radius = CornerRadius::same(CORNER);
        let waiting = match texture(ui, &mut state.model_thumbs, &key) {
            Some(tex) => {
                egui::Image::from_texture(&tex)
                    .uv(cover_uv(tex.size_vec2(), rect.size()))
                    .corner_radius(radius)
                    .paint_at(ui, rect);
                None
            }
            None => {
                let waiting = loading(&key);
                let sparkle = egui::include_image!("../../../icons/sparkle.svg");
                blank(ui, rect, tile.tint, waiting.is_none().then_some(sparkle));
                waiting
            }
        };
        if let Some(waiting) = waiting {
            ring(ui, glyph_at(rect), waiting);
        }
        foot(ui, rect);
        let selected = current(state) == tile.pick;
        if resp.hovered() && !selected {
            ui.painter().rect_filled(rect, radius, Color32::from_white_alpha(12));
        }
        words(ui, rect, tile);
        if let Some(badge) = tile.badge {
            corner_badge(ui, rect, badge);
        }
        if tile.fave {
            heart(ui, heart_at(rect));
        }
        if selected {
            ui.painter().rect_stroke(
                rect,
                radius,
                egui::Stroke::new(3.0_f32, crate::theme::selection_outline()),
                egui::StrokeKind::Inside,
            );
        }
    }
    if menu(&resp, &key, tile.fave) {
        toggle_fave(state, &fave_id("model", &key));
    }
    resp.clicked()
}

/// The part of a `size` picture that fills a `tile` without stretching it:
/// its middle across, and nearer its top than its bottom (where faces are).
pub(super) fn cover_uv(size: egui::Vec2, tile: egui::Vec2) -> egui::Rect {
    let (picture, wanted) = (size.x / size.y.max(1.0), tile.x / tile.y.max(1.0));
    if picture > wanted {
        let w = wanted / picture;
        egui::Rect::from_min_max(egui::pos2((1.0 - w) / 2.0, 0.0), egui::pos2((1.0 + w) / 2.0, 1.0))
    } else {
        let h = picture / wanted;
        let top = (1.0 - h) * 0.3;
        egui::Rect::from_min_max(egui::pos2(0.0, top), egui::pos2(1.0, top + h))
    }
}

/// A tile without a picture: dark, in its family's colour, with its glyph
/// (when there's no ring there instead).
pub(super) fn blank(ui: &egui::Ui, rect: egui::Rect, tint: Color32, glyph: Option<egui::ImageSource<'_>>) {
    let base = Color32::from_rgb(22, 22, 26);
    ui.painter().rect_filled(rect, CornerRadius::same(CORNER), lerp_color(base, tint, 0.28));
    if let Some(glyph) = glyph {
        let icon = egui::Rect::from_center_size(glyph_at(rect), egui::vec2(30.0, 30.0));
        egui::Image::new(glyph).tint(Color32::from_white_alpha(70)).paint_at(ui, icon);
    }
}

/// Where a blank tile's glyph (or ring) sits: a little over its middle, clear
/// of the words at its foot.
pub(super) fn glyph_at(rect: egui::Rect) -> egui::Pos2 {
    rect.center() - egui::vec2(0.0, 14.0)
}

/// The ring on a tile whose picture is on its way (as the downloads' ring):
/// a faint track while its turn's to come, filling with the per cent while
/// its file is read, turning while its preview downloads.
pub(super) fn ring(ui: &egui::Ui, center: egui::Pos2, loading: Loading) {
    let painter = ui.painter();
    painter.circle_stroke(center, RING_R, egui::Stroke::new(3.0_f32, Color32::from_white_alpha(40)));
    let (start, sweep) = match loading {
        Loading::Waiting => return,
        Loading::Reading(part) => {
            let percent = format!("{}%", (part.clamp(0.0, 1.0) * 100.0) as u32);
            painter.text(center, egui::Align2::CENTER_CENTER, percent, semibold(10.5), Color32::WHITE);
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            (-std::f32::consts::FRAC_PI_2, part.clamp(0.0, 1.0) * std::f32::consts::TAU)
        }
        Loading::Downloading => {
            ui.ctx().request_repaint(); // (It turns.)
            let t = ui.input(|i| i.time) as f32;
            ((t * 3.0) % std::f32::consts::TAU, std::f32::consts::PI * 0.6)
        }
    };
    if sweep < 0.01 {
        return;
    }
    let points: Vec<egui::Pos2> = (0..=32)
        .map(|k| {
            let a = start + sweep * (k as f32 / 32.0);
            center + RING_R * egui::vec2(a.cos(), a.sin())
        })
        .collect();
    painter.add(egui::Shape::line(points, egui::Stroke::new(3.0_f32, NAME_BLUE)));
}

/// The dark foot the words sit on, fading up into the picture.
pub(super) fn foot(ui: &egui::Ui, rect: egui::Rect) {
    let id = egui::Id::new("model_tile_foot");
    let tex = ui.ctx().data_mut(|d| d.get_temp::<egui::TextureHandle>(id)).unwrap_or_else(|| {
        let steps = 64;
        let rgba: Vec<u8> = (0..steps)
            .flat_map(|y| {
                let t = y as f32 / (steps - 1) as f32;
                [0, 0, 0, (t.powf(1.6) * 200.0) as u8]
            })
            .collect();
        let image = egui::ColorImage::from_rgba_unmultiplied([1, steps], &rgba);
        let tex = ui.ctx().load_texture("model_tile_foot", image, egui::TextureOptions::LINEAR);
        ui.ctx().data_mut(|d| d.insert_temp(id, tex.clone()));
        tex
    });
    let foot = egui::Rect::from_min_max(egui::pos2(rect.left(), rect.bottom() - rect.height() * 0.62), rect.max);
    let corners = CornerRadius { nw: 0, ne: 0, sw: CORNER, se: CORNER };
    let full = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
    ui.painter().add(egui::epaint::RectShape::filled(foot, corners, Color32::WHITE).with_texture(tex.id(), full));
}

/// One line of `text`, cut short with … where it would pass `width`.
pub(super) fn line(ui: &egui::Ui, text: &str, font: egui::FontId, color: Color32, width: f32) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_string(), font, color);
    job.wrap = egui::text::TextWrapping::truncate_at_width(width);
    ui.fonts_mut(|f| f.layout_job(job))
}

/// The name (blue) and its note, at the foot of the tile.
fn words(ui: &egui::Ui, rect: egui::Rect, tile: &Tile) {
    let pad = 12.0;
    let width = rect.width() - 2.0 * pad;
    let mut bottom = rect.bottom() - pad;
    if let Some(note) = &tile.note {
        let galley = line(ui, note, medium(11.5), Color32::from_white_alpha(200), width);
        bottom -= galley.size().y;
        ui.painter().galley(egui::pos2(rect.left() + pad, bottom), galley, Color32::WHITE);
        bottom -= 1.0;
    }
    let galley = line(ui, &tile.name, semibold(15.0), NAME_BLUE, width);
    ui.painter().galley(egui::pos2(rect.left() + pad, bottom - galley.size().y), galley, NAME_BLUE);
}

/// What it's for, in the tile's top left corner (where the left panel puts
/// its badges, on their dark backdrop; the top right is the heart's).
pub(super) fn corner_badge(ui: &egui::Ui, rect: egui::Rect, text: &str) {
    let galley = line(ui, text, semibold(11.0), Color32::WHITE, rect.width() / 2.0);
    let size = egui::vec2(galley.size().x + 14.0, BADGE_H);
    let badge = egui::Rect::from_min_size(egui::pos2(rect.left() + 8.0, rect.top() + 8.0), size);
    ui.painter().rect_filled(badge, CornerRadius::same(6), Color32::from_black_alpha(150));
    let at = badge.center() - galley.size() / 2.0;
    ui.painter().galley(at, galley, Color32::WHITE);
}

/// The tile's right-click menu: make it a favorite (or not), set its
/// picture, or take it away. True when the favorite item is clicked.
fn menu(resp: &egui::Response, key: &str, fave: bool) -> bool {
    let mut set = false;
    let mut clear = false;
    let mut faved = false;
    egui::Popup::context_menu(resp).frame(crate::zoom::menu_frame()).show(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        if fave_item(ui, fave) {
            faved = true;
            ui.close();
        }
        crate::zoom::menu_sep(ui);
        if crate::zoom::menu_item(ui, egui::include_image!("../../../icons/image.svg"), "Set Thumbnail…") {
            set = true;
            ui.close();
        }
        if picture(key).is_some()
            && crate::zoom::menu_item(ui, egui::include_image!("../../../icons/delete.svg"), "Remove Thumbnail")
        {
            clear = true;
            ui.close();
        }
    });
    if set
        && let Some(file) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "gif", "bmp"])
            .pick_file()
    {
        set_picture(key, &file);
    }
    if clear {
        clear_picture(key);
    }
    faved
}

/// Make `file` the tile's picture (if it's an image this can show).
fn set_picture(key: &str, file: &Path) {
    let dir = thumbs_dir();
    if let Ok(bytes) = std::fs::read(file)
        && crate::civitai::decode_thumb(&bytes, 16).is_some()
        && std::fs::create_dir_all(&dir).is_ok()
    {
        let _ = std::fs::write(dir.join(format!("{key}.img")), &bytes);
        let _ = std::fs::remove_file(dir.join(format!("{key}.none")));
    }
}

/// Take the tile's picture away — and keep Civitai's from coming back.
fn clear_picture(key: &str) {
    let dir = thumbs_dir();
    let _ = std::fs::remove_file(dir.join(format!("{key}.img")));
    let _ = std::fs::remove_file(dir.join(format!("{key}.gen.jpg")));
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = std::fs::write(dir.join(format!("{key}.none")), b"");
    }
}

/// Keep the first image made with a model as its tile's picture, while it
/// has none of its own making yet (shrunk, on a worker).
pub(in crate::generate) fn keep_first_image(key: &str, image: &Path) {
    static KEEPING: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let dir = thumbs_dir();
    let out = dir.join(format!("{key}.gen.jpg"));
    if out.exists() || dir.join(format!("{key}.none")).exists() {
        return;
    }
    {
        let mut keeping = KEEPING.lock().unwrap_or_else(|e| e.into_inner());
        if keeping.iter().any(|k| k == key) {
            return; // (A run's second image.)
        }
        keeping.push(key.to_string());
    }
    let (key, image) = (key.to_string(), image.to_path_buf());
    let job = move || {
        if let Ok(picture) = image::open(&image)
            && std::fs::create_dir_all(&dir).is_ok()
        {
            let small = shrunk(picture).to_rgb8();
            // (Written whole, then named: a half-written one is never loaded.)
            let part = dir.join(format!("{key}.gen.part"));
            if small.save_with_format(&part, image::ImageFormat::Jpeg).is_ok() {
                let _ = std::fs::rename(&part, &out);
            }
        }
        KEEPING.lock().unwrap_or_else(|e| e.into_inner()).retain(|k| k != &key);
    };
    // (A test waits for it.)
    #[cfg(test)]
    job();
    #[cfg(not(test))]
    std::thread::spawn(job);
}

/// The list opened: look up Civitai's previews for the models on this PC
/// that have no picture yet (a worker hashes each file once; it stops when
/// the list closes, see [`closed`]).
pub(super) fn opened(state: &GenerateState, ctx: &egui::Context) {
    if state.remote_host.is_some() {
        return; // (The files are on the other PC.)
    }
    let family = state.family;
    let models = GenModel::all_for(family);
    let builtin = family.builtin_ckpt_files();
    let (rows, _) = rows(state);
    let wanted: Vec<String> = rows
        .iter()
        .filter_map(|row| match row {
            Row::Tile(tile) => Some(tile.pick.key()),
            Row::Caption(_) => None,
        })
        .collect();
    let dir = thumbs_dir();
    let todo: Vec<(String, PathBuf)> = scan_checkpoints()
        .into_iter()
        .filter_map(|(file, path)| {
            // A family's one built-in model is its one built-in file.
            let pick = match (models, builtin) {
                ([m], [b]) if *b == file => Pick::Builtin(*m),
                _ if builtin.contains(&file.as_str()) => return None,
                _ => Pick::Checkpoint(file),
            };
            Some((pick.key(), path))
        })
        .filter(|(key, _)| {
            wanted.contains(key)
                && !dir.join(format!("{key}.img")).exists()
                && !dir.join(format!("{key}.none")).exists()
        })
        .collect();
    #[cfg(test)]
    {
        let _ = ctx;
        FETCH_FOR_TEST.set(Some(todo.into_iter().map(|(key, _)| key).collect()));
    }
    #[cfg(not(test))]
    fetch(todo, ctx.clone());
}

/// The list closed: the look-up stops after the file it's on.
pub(super) fn closed() {
    set_cancelled(true);
}

#[cfg(not(test))]
static FETCH_CANCEL: AtomicBool = AtomicBool::new(false);
#[cfg(not(test))]
static FETCH_RUNNING: AtomicBool = AtomicBool::new(false);

#[cfg(not(test))]
fn cancelled() -> bool {
    FETCH_CANCEL.load(Ordering::SeqCst)
}
#[cfg(not(test))]
fn set_cancelled(stop: bool) {
    FETCH_CANCEL.store(stop, Ordering::SeqCst);
}
// (A test's own, so tests closing lists alongside it don't stop it.)
#[cfg(test)]
fn cancelled() -> bool {
    CANCEL_FOR_TEST.get()
}
#[cfg(test)]
fn set_cancelled(stop: bool) {
    CANCEL_FOR_TEST.set(stop);
}

/// Where the look-up has got to, for the tiles' rings.
#[derive(Default)]
struct Fetching {
    /// Still to come, in turn.
    waiting: Vec<String>,
    /// The one it's on: how far through reading its file (`None`: read,
    /// and its preview is downloading).
    now: Option<(String, Option<f32>)>,
}

#[cfg(not(test))]
static FETCHING: std::sync::Mutex<Fetching> = std::sync::Mutex::new(Fetching { waiting: Vec::new(), now: None });

#[cfg(not(test))]
fn fetching<R>(f: impl FnOnce(&mut Fetching) -> R) -> R {
    f(&mut FETCHING.lock().unwrap_or_else(|e| e.into_inner()))
}
#[cfg(test)]
fn fetching<R>(f: impl FnOnce(&mut Fetching) -> R) -> R {
    FETCHING_FOR_TEST.with_borrow_mut(f)
}

/// What a tile's picture is waiting on, if anything.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(in crate::generate) enum Loading {
    /// Its turn's to come.
    Waiting,
    /// Its file is being read (this far through) to look it up by.
    Reading(f32),
    /// Its preview is downloading.
    Downloading,
}

pub(in crate::generate) fn loading(key: &str) -> Option<Loading> {
    fetching(|f| match &f.now {
        Some((now, Some(part))) if now == key => Some(Loading::Reading(*part)),
        Some((now, None)) if now == key => Some(Loading::Downloading),
        _ => f.waiting.iter().any(|k| k == key).then_some(Loading::Waiting),
    })
}

/// The look-up, on a worker. One at a time.
#[cfg(not(test))]
fn fetch(todo: Vec<(String, PathBuf)>, ctx: egui::Context) {
    set_cancelled(false);
    if todo.is_empty() || FETCH_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        struct Running;
        impl Drop for Running {
            fn drop(&mut self) {
                FETCH_RUNNING.store(false, Ordering::SeqCst);
            }
        }
        let _running = Running;
        let lookup = |sha: &str| crate::civitai::preview_media_by_hash(sha).map(|found| found.map(|(still, _)| still));
        work(todo, &thumbs_dir(), &lookup, &|| ctx.request_repaint());
    });
}

/// Civitai's preview for a SHA-256: the picture's bytes, `None` if Civitai
/// has none.
type Lookup<'a> = &'a dyn Fn(&str) -> Result<Option<Vec<u8>>, crate::civitai::PreviewFail>;

/// The look-up's work: each file's SHA-256 (the session keeps it: checkpoints
/// are gigabytes) → its preview, kept as `<key>.img` (or `<key>.none` when
/// Civitai has none). It says where it's got to as it goes, for the rings.
pub(in crate::generate) fn work(todo: Vec<(String, PathBuf)>, dir: &Path, lookup: Lookup, repaint: &dyn Fn()) {
    let _ = std::fs::create_dir_all(dir);
    fetching(|f| {
        f.waiting = todo.iter().map(|(key, _)| key.clone()).collect();
        f.now = None;
    });
    repaint();
    let mut failed_in_a_row = 0;
    for (key, path) in &todo {
        if cancelled() {
            break;
        }
        let now = |part: Option<f32>| {
            fetching(|f| {
                f.waiting.retain(|k| k != key);
                f.now = Some((key.clone(), part));
            });
            repaint();
        };
        now(Some(0.0));
        // (Said each whole per cent: not each block of a 17 GB file.)
        let mut shown = 0;
        let sha = cached_sha256_with(path, |part| {
            let percent = (part * 100.0) as u32;
            if percent > shown {
                shown = percent;
                now(Some(part));
            }
        });
        let Some(sha) = sha else { continue };
        now(None);
        match lookup(&sha) {
            Ok(Some(still)) => {
                let _ = std::fs::write(dir.join(format!("{key}.img")), still);
                failed_in_a_row = 0;
            }
            Ok(None) => {
                let _ = std::fs::write(dir.join(format!("{key}.none")), b"");
                failed_in_a_row = 0;
            }
            Err(crate::civitai::PreviewFail::Offline) => break,
            Err(crate::civitai::PreviewFail::Item) => {
                failed_in_a_row += 1;
                if failed_in_a_row >= MAX_ITEM_FAILURES_IN_A_ROW {
                    break;
                }
            }
        }
    }
    // (Stopped or done, no ring is left turning.)
    fetching(|f| *f = Fetching::default());
    repaint();
}

#[cfg(test)]
thread_local! {
    /// Where a test keeps the tiles' pictures.
    pub(in crate::generate) static THUMBS_DIR_FOR_TEST: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
    /// What the last opening would have looked up on Civitai.
    pub(in crate::generate) static FETCH_FOR_TEST: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
    /// The look-up's progress, in a test.
    static FETCHING_FOR_TEST: std::cell::RefCell<Fetching> = std::cell::RefCell::new(Fetching::default());
    /// The look-up is to stop, in a test.
    pub(in crate::generate) static CANCEL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set where a test's look-up has got to: `waiting` in turn, and the one
/// it's `now` on.
#[cfg(test)]
pub(in crate::generate) fn set_fetching_for_test(waiting: &[&str], now: Option<(&str, Option<f32>)>) {
    fetching(|f| {
        f.waiting = waiting.iter().map(|k| k.to_string()).collect();
        f.now = now.map(|(key, part)| (key.to_string(), part));
    });
}
