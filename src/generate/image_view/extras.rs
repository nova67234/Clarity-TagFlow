//! The text box's LoRAs | Embeddings pill (between its + and send, in the
//! Family | Model bar's style) and the lists it opens over the text box:
//! picture tiles as in the Model list, each keeping its picture's shape (as
//! the left panel's do). A click turns one on — ringed, its weight slider
//! along its foot under its name; its ⓘ fills the list with its details; and
//! favorites (right-click) come first, with the left panel's heart.

use super::models::{
    CAPTION_GAP, CAPTION_H, CORNER, FOOTER_H, Loading, MARGIN, NAME_BLUE, TILE_ASPECT, TILE_GAP, blank, caption,
    corner_badge, cover_uv, fave_id, fave_item, foot, glyph_at, heart, is_fave, line, load_faves, ring, toggle_fave,
};
use super::*;

/// Which list.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(in crate::generate) enum Extra {
    Loras,
    Embeddings,
}

impl Extra {
    fn title(self) -> &'static str {
        match self {
            Extra::Loras => "LoRAs",
            Extra::Embeddings => "Embeddings",
        }
    }

    /// What it lists, in a sentence.
    fn plural(self) -> &'static str {
        match self {
            Extra::Loras => "LoRAs",
            Extra::Embeddings => "embeddings",
        }
    }

    /// Its favorites' kind.
    fn kind(self) -> &'static str {
        match self {
            Extra::Loras => "lora",
            Extra::Embeddings => "embedding",
        }
    }

    /// The caption over the rest, under the favorites.
    fn all(self) -> &'static str {
        match self {
            Extra::Loras => "All LoRAs",
            Extra::Embeddings => "All embeddings",
        }
    }

    fn glyph(self) -> egui::ImageSource<'static> {
        match self {
            Extra::Loras => egui::include_image!("../../../icons/lora.svg"),
            Extra::Embeddings => egui::include_image!("../../../icons/extension.svg"),
        }
    }
}

/// The pill's height (the text box's row of buttons).
pub(super) const PILL_H: f32 = 44.0;
/// A tile is at most this tall, and at least this, for its width (its
/// picture is cropped past them).
const TALLEST: f32 = 1.25;
const WIDEST: f32 = 0.5;
/// The ⓘ's radius.
const INFO_R: f32 = 14.0;
/// The details' top row (back, name, favorite).
const BACK_H: f32 = 36.0;
/// An empty list's message.
const EMPTY_H: f32 = 72.0;
/// The positive / negative chips' height.
const CHIP_H: f32 = 20.0;

/// The pill in `rect`. Its lists hang over `prompt` (the text box), as wide.
pub(super) fn pill(ui: &mut egui::Ui, rect: egui::Rect, prompt: egui::Rect, state: &mut GenerateState) {
    ui.painter().rect(
        rect,
        rect.height() / 2.0,
        TEXT().gamma_multiply(0.03),
        egui::Stroke::new(1.0_f32, hairline()),
        egui::StrokeKind::Inside,
    );
    let inner = rect.shrink(3.0);
    let half = (inner.width() - 5.0) / 2.0;
    let left = egui::Rect::from_min_size(inner.min, egui::vec2(half, inner.height()));
    let right = egui::Rect::from_min_max(egui::pos2(left.right() + 5.0, inner.top()), inner.max);
    let x = left.right() + 2.5;
    ui.painter().line_segment(
        [egui::pos2(x, inner.top() + 8.0), egui::pos2(x, inner.bottom() - 8.0)],
        egui::Stroke::new(1.0_f32, hairline()),
    );

    let family = state.family.title();
    let open = state.extras_open;
    let mut anchor = None;
    for (kind, half) in [(Extra::Loras, left), (Extra::Embeddings, right)] {
        let id = egui::Id::new(("t2i_extras", kind.kind(), family));
        let resp = picker(ui, half, id, kind.title(), &value(state, kind), None, open == Some(kind));
        if resp.clicked() {
            state.extras_open = if open == Some(kind) { None } else { Some(kind) };
            state.extras_info = None;
        }
        if anchor.is_none() {
            anchor = Some(resp);
        }
    }

    let open = state.extras_open;
    let opening = open.is_some() && open != state.extras_was_open;
    if open != state.extras_was_open {
        if let Some(was) = state.extras_was_open {
            stop(was);
        }
        if let Some(now) = open {
            start(state, ui.ctx(), now);
        }
    }
    state.extras_was_open = open;
    if let (Some(kind), Some(mut anchor)) = (open, anchor) {
        // (The list hangs from the text box's width, over it.)
        anchor.rect = egui::Rect::from_x_y_ranges(prompt.x_range(), rect.y_range());
        anchor.interact_rect = anchor.rect;
        list(&anchor, rect, state, kind, opening);
    }
}

/// What the pill says of a list: how many are on.
fn value(state: &GenerateState, kind: Extra) -> String {
    let on = match kind {
        Extra::Loras => state.loras.iter().filter(|l| l.selected).count(),
        Extra::Embeddings => state.embeddings.iter().filter(|e| e.selected).count(),
    };
    match on {
        0 => "None".to_string(),
        n => format!("{n} on"),
    }
}

/// A list opens: what's installed is looked for again (a new download is
/// there without a restart) and the previews looked up, and the favorites
/// read in.
fn start(state: &mut GenerateState, ctx: &egui::Context, kind: Extra) {
    load_faves(state);
    // (A test sets out its own LoRAs, and looks nothing up.)
    #[cfg(not(test))]
    match kind {
        Extra::Loras => refresh_loras(state, ctx),
        Extra::Embeddings => refresh_embeddings(state, ctx),
    }
    #[cfg(test)]
    let _ = (ctx, kind);
}

/// A list closes: its previews' look-up stops.
fn stop(kind: Extra) {
    #[cfg(not(test))]
    match kind {
        Extra::Loras => cancel_lora_thumb_fetch(),
        Extra::Embeddings => cancel_embedding_thumb_fetch(),
    }
    #[cfg(test)]
    let _ = kind;
}

/// The open list, over the text box (`anchor`, at the pill's row). It closes
/// on Escape and on a click outside it and the pill — not on one in a tile's
/// menu or a delete's confirmation.
fn list(anchor: &egui::Response, pill: egui::Rect, state: &mut GenerateState, kind: Extra, opening: bool) {
    let ctx = anchor.ctx.clone();
    let margin = f32::from(MARGIN);
    // As far up as the window goes.
    let room = (anchor.rect.top() - ctx.content_rect().top() - 6.0 - 12.0 - 2.0 * margin).max(160.0);
    let width = anchor.rect.width() - 2.0 * margin;
    let menu_was_open = egui::Popup::is_any_open(&ctx)
        || state.lora_delete_confirm.is_some()
        || state.embedding_delete_confirm.is_some();
    let id = egui::Id::new(("t2i_extras_list", kind.kind(), state.family.title()));
    let mut open = true;
    let shown = egui::Popup::from_response(anchor)
        .id(id)
        .open_bool(&mut open)
        .align(egui::RectAlign::TOP_START)
        .gap(6.0)
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .frame(crate::settings::dropdown_frame().inner_margin(MARGIN).corner_radius(22))
        .show(|ui| {
            ui.set_width(width);
            THUMB_DECODE_BUDGET.store(3, Ordering::Relaxed);
            apply_sniffs(state);
            pick_up_previews(ui, state, kind);
            match state.extras_info.clone() {
                Some(file) => details(ui, state, kind, &file, width, room),
                None => tiles(ui, state, kind, width, room, opening),
            }
        });
    // (Where it was last frame counts too: a click that changes what it
    // shows — back from the details — changes its size under the pointer.)
    let was_at = ctx.data(|d| d.get_temp::<egui::Rect>(id.with("rect")));
    let outside = shown.is_some_and(|shown| {
        let rect = shown.response.rect;
        ctx.data_mut(|d| d.insert_temp(id.with("rect"), rect));
        // It hangs up from the pill, placed by its last size: grown or
        // shrunk (the details, back, other families shown), the frame's
        // done again at its new size rather than shown a frame askew.
        if was_at.is_some_and(|was| (was.size() - rect.size()).length() > 0.5) {
            ctx.request_discard("the list changed size");
            // (And again until it settles: what's in it can lag a frame too.)
            ctx.request_repaint();
        }
        let inside = |p: egui::Pos2| rect.contains(p) || was_at.is_some_and(|r| r.contains(p)) || pill.contains(p);
        ctx.input(|i| i.pointer.any_click() && i.pointer.interact_pos().is_some_and(|p| !inside(p)))
    });
    if !open || (outside && !menu_was_open) {
        state.extras_open = None;
        state.extras_info = None;
    }
}

/// The families sniffed since the last frame, on their entries.
fn apply_sniffs(state: &mut GenerateState) {
    let models = comfy_base().join("ComfyUI").join("models");
    if state.loras.iter().any(|l| !l.sniffed) {
        let dir = models.join("loras");
        let cache = SNIFF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        for l in state.loras.iter_mut().filter(|l| !l.sniffed) {
            if let Some(base) = cache.get(&dir.join(&l.file)) {
                l.base = *base;
                l.sniffed = true;
            }
        }
    }
    if state.embeddings.iter().any(|e| !e.sniffed) {
        let dir = models.join("embeddings");
        let cache = SNIFF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        for e in state.embeddings.iter_mut().filter(|e| !e.sniffed) {
            if let Some(base) = cache.get(&dir.join(&e.file)) {
                e.base = *base;
                e.sniffed = true;
            }
        }
    }
}

/// The previews that have come in since the last frame, on their entries.
fn pick_up_previews(ui: &egui::Ui, state: &mut GenerateState, kind: Extra) {
    match kind {
        Extra::Loras => {
            let dir = lora_thumbs_dir();
            for l in &mut state.loras {
                pick_up_lora_preview(ui, l, &dir);
            }
        }
        Extra::Embeddings => {
            let dir = embedding_thumbs_dir();
            for e in &mut state.embeddings {
                pick_up_embedding_preview(ui, e, &dir);
            }
        }
    }
}

/// One LoRA or embedding, as the list has it.
#[derive(Clone)]
struct Item {
    /// Where it is in its list.
    i: usize,
    file: String,
    name: String,
    base: LoraBase,
    sniffed: bool,
    selected: bool,
    /// Made for another family.
    off: bool,
    fave: bool,
    /// Civitai has no preview of it.
    missing: bool,
}

fn items(state: &GenerateState, kind: Extra) -> Vec<Item> {
    let family = state.family;
    let item = |i, file: &str, name: &str, base: LoraBase, sniffed, selected, missing| Item {
        i,
        file: file.to_string(),
        name: name.to_string(),
        base,
        sniffed,
        selected,
        off: !base.matches(family),
        fave: is_fave(state, &fave_id(kind.kind(), file)),
        missing,
    };
    match kind {
        Extra::Loras => state
            .loras
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let name = l.file.trim_end_matches(".safetensors");
                item(i, &l.file, name, l.base, l.sniffed, l.selected, l.thumb_missing)
            })
            .collect(),
        Extra::Embeddings => state
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| item(i, &e.file, embed_stem(&e.file), e.base, e.sniffed, e.selected, e.thumb_missing))
            .collect(),
    }
}

fn show_other(state: &mut GenerateState, kind: Extra) -> &mut bool {
    match kind {
        Extra::Loras => &mut state.show_other_loras,
        Extra::Embeddings => &mut state.show_other_embeddings,
    }
}

/// One row of a list.
enum Row {
    Caption(&'static str),
    Tile(Item),
}

/// The list's rows — the favorites first, then this family's, then (when
/// asked for) other families' — and how many are other families'.
fn rows(state: &mut GenerateState, kind: Extra) -> (Vec<Row>, usize) {
    let all = items(state, kind);
    let others_shown = *show_other(state, kind);
    let hidden = all.iter().filter(|it| it.off).count();
    let shown = all.into_iter().filter(|it| !it.off || others_shown);
    let (faves, rest): (Vec<Item>, Vec<Item>) = shown.partition(|it| it.fave);
    let (ours, others): (Vec<Item>, Vec<Item>) = rest.into_iter().partition(|it| !it.off);
    let mut rows = Vec::new();
    if !faves.is_empty() {
        rows.push(Row::Caption("Favorites"));
        rows.extend(faves.into_iter().map(Row::Tile));
        if !ours.is_empty() {
            rows.push(Row::Caption(kind.all()));
        }
    }
    rows.extend(ours.into_iter().map(Row::Tile));
    if !others.is_empty() {
        rows.push(Row::Caption("For other models"));
        rows.extend(others.into_iter().map(Row::Tile));
    }
    (rows, hidden)
}

/// The space above row `i`: a caption sits closer to its tiles.
fn gap_before(rows: &[Row], i: usize) -> f32 {
    match i {
        0 => 0.0,
        _ if matches!(rows[i - 1], Row::Caption(_)) => CAPTION_GAP,
        _ => TILE_GAP,
    }
}

/// What to show of entry `i` now: its preview (an animation's frame now).
fn picture(ui: &egui::Ui, state: &GenerateState, kind: Extra, i: usize) -> Option<egui::TextureHandle> {
    match kind {
        Extra::Loras => {
            let l = &state.loras[i];
            match &l.anim {
                Some(frames) => Some(anim_frame(ui, frames)),
                None => l.thumb.clone(),
            }
        }
        Extra::Embeddings => state.embeddings[i].thumb.clone(),
    }
}

/// The height `picture` takes `width` wide: its own shape, kept between
/// [`WIDEST`] and [`TALLEST`] (a model list's shape without one).
fn height_for(picture: Option<&egui::TextureHandle>, width: f32) -> f32 {
    let aspect = picture.map_or(TILE_ASPECT, |tex| {
        let [w, h] = tex.size();
        (h as f32 / w.max(1) as f32).clamp(WIDEST, TALLEST)
    });
    (width * aspect).round()
}

/// The previews' look-up is going.
fn fetching(kind: Extra) -> bool {
    match kind {
        Extra::Loras => lora_thumbs_fetching(),
        Extra::Embeddings => embeddings_fetching(),
    }
}

/// The tiles (scrolling past `max_h`) and, when other families' are about,
/// the switch that shows them.
fn tiles(ui: &mut egui::Ui, state: &mut GenerateState, kind: Extra, width: f32, max_h: f32, opening: bool) {
    let (rows, hidden) = rows(state, kind);
    let heights: Vec<f32> = rows
        .iter()
        .map(|row| match row {
            Row::Caption(_) => CAPTION_H,
            Row::Tile(it) => height_for(picture(ui, state, kind, it.i).as_ref(), width),
        })
        .collect();
    let mut content_h = 0.0;
    for (i, h) in heights.iter().enumerate() {
        content_h += gap_before(&rows, i) + h;
    }
    let footer_h = if hidden > 0 { TILE_GAP + 1.0 + 8.0 + FOOTER_H } else { 0.0 };

    if rows.is_empty() {
        empty(ui, state, kind, width);
    } else {
        // At least a tile, however short the window.
        let scroll_h = content_h.min((max_h - footer_h).max(heights.iter().cloned().fold(0.0, f32::max).min(content_h)));
        let gutter = f32::from(MARGIN);
        let mut wide = crate::edge_scroll_ui(ui, gutter);
        let mut scroll = egui::ScrollArea::vertical();
        if opening {
            // (It opens at the top, not where it was left last time.)
            scroll = scroll.vertical_scroll_offset(0.0);
        }
        scroll
            .id_salt(("t2i_extras_tiles", kind.kind(), state.family.title()))
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
                        Row::Tile(it) => tile(ui, state, kind, it, width, heights[i]),
                    }
                }
            });
        crate::edge_scroll_done(ui, &wide, gutter);
    }

    if hidden > 0 {
        ui.add_space(TILE_GAP);
        let (line, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
        ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0_f32, hairline()));
        ui.add_space(8.0);
        let (row, _) = ui.allocate_exact_size(egui::vec2(width, FOOTER_H), egui::Sense::hover());
        let mut row_ui = at(ui, row);
        row_ui.horizontal_centered(|ui| {
            ui.add_space(4.0);
            let text = format!("Show other-family {} ({hidden})", kind.plural());
            ui.label(RichText::new(text).font(medium(12.5)).color(TEXT()));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                crate::settings::switch(ui, show_other(state, kind));
            });
        });
    }
}

/// A list with nothing to show: where the files go, or that this family
/// has none.
fn empty(ui: &mut egui::Ui, state: &GenerateState, kind: Extra, width: f32) {
    let none_at_all = match kind {
        Extra::Loras => state.loras.is_empty(),
        Extra::Embeddings => state.embeddings.is_empty(),
    };
    let text = match (kind, none_at_all) {
        (Extra::Loras, true) => "No LoRAs yet: put .safetensors files in models/loras.".to_string(),
        (Extra::Embeddings, true) => "No embeddings yet: put them in models/embeddings.".to_string(),
        (_, false) => format!("No {} {} here.", state.family.title(), kind.title()),
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, EMPTY_H), egui::Sense::hover());
    let galley = ui.fonts_mut(|f| f.layout(text, medium(12.5), MUTED(), width - 24.0));
    ui.painter().galley(rect.center() - galley.size() / 2.0, galley, MUTED());
}

/// The tile's family colour (for one without a picture).
fn tint(it: &Item) -> Color32 {
    if it.base == LoraBase::Unknown { ACCENT1() } else { it.base.dot_color() }
}

/// One tile, `height` tall.
fn tile(ui: &mut egui::Ui, state: &mut GenerateState, kind: Extra, it: &Item, width: f32, height: f32) {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let id = egui::Id::new(("t2i_extra_tile", kind.kind(), &it.file));
    let radius = CornerRadius::same(CORNER);
    let mut info = false;
    if ui.is_rect_visible(rect) {
        match picture(ui, state, kind, it.i) {
            Some(tex) => {
                egui::Image::from_texture(&tex)
                    .uv(cover_uv(tex.size_vec2(), rect.size()))
                    .corner_radius(radius)
                    .paint_at(ui, rect);
            }
            None => {
                // While its preview's being looked for, a ring turns there.
                let looking = !it.missing && fetching(kind);
                blank(ui, rect, tint(it), (!looking).then_some(kind.glyph()));
                if looking {
                    ring(ui, glyph_at(rect), Loading::Downloading);
                }
            }
        }
        foot(ui, rect);
        if resp.hovered() && !it.selected {
            ui.painter().rect_filled(rect, radius, Color32::from_white_alpha(12));
        }
        if it.sniffed && it.base != LoraBase::Unknown {
            corner_badge(ui, rect, it.base.label());
        }
    }

    // ⓘ, top right; the heart beside it.
    let info_rect = egui::Rect::from_center_size(
        egui::pos2(rect.right() - 8.0 - INFO_R, rect.top() + 8.0 + INFO_R),
        egui::vec2(2.0 * INFO_R, 2.0 * INFO_R),
    );
    let info_resp = ui
        .interact(info_rect, id.with("info"), egui::Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text("Details");
    info |= info_resp.clicked();
    if ui.is_rect_visible(rect) {
        let fill = Color32::from_black_alpha(if info_resp.hovered() { 210 } else { 150 });
        ui.painter().circle_filled(info_rect.center(), INFO_R, fill);
        egui::Image::new(egui::include_image!("../../../icons/info.svg"))
            .tint(Color32::WHITE)
            .paint_at(ui, egui::Rect::from_center_size(info_rect.center(), egui::vec2(16.0, 16.0)));
        if it.fave {
            heart(ui, egui::pos2(info_rect.left() - 6.0 - 9.0, info_rect.center().y));
        }
    }

    // Its name at its foot; on, its weight under it (and, for an embedding,
    // which prompt it goes in, beside the name).
    let pad = 12.0;
    let mut name_bottom = rect.bottom() - pad;
    let mut name_room = rect.width() - 2.0 * pad;
    if it.selected {
        let slider = egui::Rect::from_min_max(
            egui::pos2(rect.left() + pad, rect.bottom() - 10.0 - crate::slider::HEIGHT),
            egui::pos2(rect.right() - pad, rect.bottom() - 10.0),
        );
        let mut slider_ui = at(ui, slider);
        slider_ui.spacing_mut().slider_width = f32::INFINITY;
        slider_ui.add(PillSlider::new(strength(state, kind, it.i), 0.0..=2.0).default_value(DEFAULT_STRENGTH as f64));
        name_bottom = slider.top() - 6.0;
        if kind == Extra::Embeddings {
            let used = chips(ui, state, it.i, egui::pos2(rect.right() - pad, name_bottom), id);
            name_room -= used + 8.0;
        }
    }
    if ui.is_rect_visible(rect) {
        let name = line(ui, &it.name, semibold(15.0), NAME_BLUE, name_room);
        ui.painter().galley(egui::pos2(rect.left() + pad, name_bottom - name.size().y), name, NAME_BLUE);
        if it.selected {
            ui.painter().rect_stroke(
                rect,
                radius,
                egui::Stroke::new(3.0_f32, crate::theme::selection_outline()),
                egui::StrokeKind::Inside,
            );
        }
    }

    if menu(state, kind, it, &resp) {
        return;
    }
    if info {
        state.extras_info = Some(it.file.clone());
    } else if resp.clicked() {
        let on = selected(state, kind, it.i);
        *on = !*on;
    }
}

fn selected(state: &mut GenerateState, kind: Extra, i: usize) -> &mut bool {
    match kind {
        Extra::Loras => &mut state.loras[i].selected,
        Extra::Embeddings => &mut state.embeddings[i].selected,
    }
}

fn strength(state: &mut GenerateState, kind: Extra, i: usize) -> &mut f32 {
    match kind {
        Extra::Loras => &mut state.loras[i].strength,
        Extra::Embeddings => &mut state.embeddings[i].strength,
    }
}

/// An embedding's "Positive" / "Negative" chips, ending at `right_bottom`:
/// which prompt it goes in. The width they take.
fn chips(ui: &mut egui::Ui, state: &mut GenerateState, i: usize, right_bottom: egui::Pos2, id: egui::Id) -> f32 {
    let negative = state.embeddings[i].negative;
    let mut right = right_bottom.x;
    for (text, on, color, is_negative) in [
        ("Negative", negative, Color32::from_rgb(200, 80, 80), true),
        ("Positive", !negative, Color32::from_rgb(70, 160, 90), false),
    ] {
        let galley = line(ui, text, semibold(11.0), Color32::WHITE, 200.0);
        let size = egui::vec2(galley.size().x + 16.0, CHIP_H);
        let chip = egui::Rect::from_min_size(egui::pos2(right - size.x, right_bottom.y - size.y), size);
        let resp = ui
            .interact(chip, id.with(text), egui::Sense::click())
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        if resp.clicked() {
            state.embeddings[i].negative = is_negative;
        }
        let fill = if on { color } else { Color32::from_black_alpha(150) };
        ui.painter().rect_filled(chip, CornerRadius::same(10), fill);
        let ink = if on { Color32::WHITE } else { Color32::from_white_alpha(170) };
        ui.painter().galley(chip.center() - galley.size() / 2.0, galley, ink);
        right = chip.left() - 4.0;
    }
    right_bottom.x - right - 4.0
}

/// A tile's right-click menu: favorite (or not), its picture, and delete
/// (on this PC). True when it's deleting.
fn menu(state: &mut GenerateState, kind: Extra, it: &Item, resp: &egui::Response) -> bool {
    let (mut faved, mut set, mut delete) = (false, false, false);
    egui::Popup::context_menu(resp).frame(crate::zoom::menu_frame()).show(|ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        if fave_item(ui, it.fave) {
            faved = true;
            ui.close();
        }
        crate::zoom::menu_sep(ui);
        if crate::zoom::menu_item(ui, egui::include_image!("../../../icons/image.svg"), "Set Thumbnail…") {
            set = true;
            ui.close();
        }
        // (Another PC's is its owner's to delete.)
        if picker_host().is_none() {
            crate::zoom::menu_sep(ui);
            let label = match kind {
                Extra::Loras => "Delete LoRA…",
                Extra::Embeddings => "Delete Embedding…",
            };
            if crate::zoom::menu_item(ui, egui::include_image!("../../../icons/delete.svg"), label) {
                delete = true;
                ui.close();
            }
        }
    });
    if faved {
        toggle_fave(state, &fave_id(kind.kind(), &it.file));
    }
    if set
        && let Some(src) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "gif", "bmp"])
            .pick_file()
    {
        match kind {
            Extra::Loras => set_lora_thumbnail(&mut state.loras[it.i], &lora_thumbs_dir(), &src),
            Extra::Embeddings => set_embedding_thumbnail(&mut state.embeddings[it.i], &embedding_thumbs_dir(), &src),
        }
    }
    if delete {
        match kind {
            Extra::Loras => state.lora_delete_confirm = Some(it.file.clone()),
            Extra::Embeddings => state.embedding_delete_confirm = Some(it.file.clone()),
        }
    }
    delete
}

/// The list filled with one's details: back, its name and favorite on top;
/// its picture; on or off, its weight (and for an embedding, which prompt);
/// then what its file says (a LoRA's trigger words and training facts).
fn details(ui: &mut egui::Ui, state: &mut GenerateState, kind: Extra, file: &str, width: f32, room: f32) {
    let Some(it) = items(state, kind).into_iter().find(|it| it.file == file) else {
        // (It's gone: deleted, or the list was looked over again.)
        state.extras_info = None;
        return;
    };
    let id = egui::Id::new(("t2i_extra_details", kind.kind(), file));

    // Back, the name, the favorite.
    let (head, _) = ui.allocate_exact_size(egui::vec2(width, BACK_H), egui::Sense::hover());
    let back = egui::Rect::from_min_size(head.min, egui::vec2(BACK_H, BACK_H));
    let back_resp = ui
        .interact(back, id.with("back"), egui::Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(format!("Back to the {}", kind.title()));
    if back_resp.hovered() {
        ui.painter().circle_filled(back.center(), BACK_H / 2.0, TEXT().gamma_multiply(0.06));
    }
    let c = back.center();
    ui.painter().add(egui::Shape::line(
        vec![c + egui::vec2(3.0, -6.0), c + egui::vec2(-3.0, 0.0), c + egui::vec2(3.0, 6.0)],
        egui::Stroke::new(2.0_f32, TEXT()),
    ));
    let fave = egui::Rect::from_min_size(egui::pos2(head.right() - BACK_H, head.top()), egui::vec2(BACK_H, BACK_H));
    let fave_resp = ui
        .interact(fave, id.with("fave"), egui::Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(if it.fave { "Remove Favorite" } else { "Favorite" });
    if it.fave {
        heart(ui, fave.center());
    } else {
        let tint = if fave_resp.hovered() { TEXT() } else { MUTED() };
        egui::Image::new(egui::include_image!("../../../icons/heart_plus.svg"))
            .tint(icon_tint(tint))
            .paint_at(ui, egui::Rect::from_center_size(fave.center(), egui::vec2(20.0, 20.0)));
    }
    let title = line(ui, &it.name, semibold(16.0), TEXT(), (fave.left() - back.right() - 12.0).max(0.0));
    ui.painter().galley(egui::pos2(back.right() + 6.0, head.center().y - title.size().y / 2.0), title, TEXT());
    if back_resp.clicked() {
        state.extras_info = None;
        return;
    }
    if fave_resp.clicked() {
        toggle_fave(state, &fave_id(kind.kind(), file));
    }
    ui.add_space(10.0);

    // As tall as what's in it was last frame (as tall as there's room for,
    // the first): the list hangs up from the pill, so it's given the height
    // it's to have rather than left to grow into what it had.
    let max_h = (room - BACK_H - 10.0).max(120.0);
    let measured = id.with("content_h");
    let wanted = ui.ctx().data(|d| d.get_temp::<f32>(measured)).unwrap_or(max_h).min(max_h);
    let shown = egui::ScrollArea::vertical()
        .id_salt(("t2i_extra_details_scroll", kind.kind()))
        .min_scrolled_height(wanted)
        .max_height(max_h)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.set_width(width);
            ui.spacing_mut().item_spacing.y = 6.0;
            // Its picture, its shape.
            let shown = picture(ui, state, kind, it.i);
            // (Not so tall the details go out of sight: past that it's cropped.)
            let tall = height_for(shown.as_ref(), width).min((room * 0.42).max(width * WIDEST));
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, tall), egui::Sense::hover());
            match shown {
                Some(tex) => {
                    egui::Image::from_texture(&tex)
                        .uv(cover_uv(tex.size_vec2(), rect.size()))
                        .corner_radius(CORNER)
                        .paint_at(ui, rect);
                }
                None => blank(ui, rect, tint(&it), Some(kind.glyph())),
            }
            if it.sniffed && it.base != LoraBase::Unknown {
                corner_badge(ui, rect, it.base.label());
            }
            ui.add_space(8.0);

            // On or off, and how strong.
            let use_it = match kind {
                Extra::Loras => "Use this LoRA",
                Extra::Embeddings => "Use this embedding",
            };
            setting_row(ui, use_it, |ui| {
                crate::settings::switch(ui, selected(state, kind, it.i));
            });
            if *selected(state, kind, it.i) {
                slider_row(
                    ui,
                    Some("Weight"),
                    PillSlider::new(strength(state, kind, it.i), 0.0..=2.0).default_value(DEFAULT_STRENGTH as f64),
                );
                if kind == Extra::Embeddings {
                    setting_row(ui, "Goes in", |ui| {
                        let negative = &mut state.embeddings[it.i].negative;
                        for (text, is_negative, color) in [
                            ("Negative", true, Color32::from_rgb(200, 80, 80)),
                            ("Positive", false, Color32::from_rgb(70, 160, 90)),
                        ] {
                            let chip = polarity_pill(ui, text, *negative == is_negative, color);
                            if ui
                                .interact(chip, id.with(text), egui::Sense::click())
                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                .clicked()
                            {
                                *negative = is_negative;
                            }
                        }
                    });
                }
            }
            ui.add_space(4.0);
            let (rule, _) = ui.allocate_exact_size(egui::vec2(width, 1.0), egui::Sense::hover());
            ui.painter().hline(rule.x_range(), rule.center().y, egui::Stroke::new(1.0_f32, hairline()));
            ui.add_space(4.0);

            // What its file says. (Its text can be selected, and copied.)
            ui.style_mut().interaction.selectable_labels = true;
            match kind {
                Extra::Loras => lora_facts(ui, state, it.i, id),
                Extra::Embeddings => {
                    let token = format!("embedding:{}", it.name);
                    copy_heading(ui, "Use it by name", &token, id);
                    ui.label(RichText::new(&token).font(medium(12.5)).color(MUTED()));
                }
            }
            let made_for = if it.sniffed && it.base != LoraBase::Unknown { it.base.label() } else { "Not known" };
            crate::settings::label_value_row(ui, "Made for:", made_for, 11.5);
            crate::settings::label_value_row(ui, "File:", &it.file, 11.5);
        });
    ui.ctx().data_mut(|d| d.insert_temp(measured, shown.content_size.y));
}

/// A row of the details: `label`, and its control on the right.
fn setting_row(ui: &mut egui::Ui, label: &str, control: impl FnOnce(&mut egui::Ui)) {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 32.0), egui::Sense::hover());
    let mut row_ui = at(ui, row);
    row_ui.horizontal_centered(|ui| {
        ui.label(RichText::new(label).font(medium(12.5)).color(TEXT()));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            control(ui);
        });
    });
}

/// A heading with a copy button that puts `text` on the clipboard.
fn copy_heading(ui: &mut egui::Ui, heading: &str, text: &str, id: egui::Id) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(heading).font(semibold(12.0)).color(TEXT()));
        let copy = ui
            .add(
                egui::Image::new(egui::include_image!("../../../icons/copy.svg"))
                    .fit_to_exact_size(egui::vec2(13.0, 13.0))
                    .tint(icon_tint(MUTED()))
                    .sense(egui::Sense::click()),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("Copy");
        if copy.clicked() {
            ui.ctx().copy_text(text.to_string());
            ui.ctx().data_mut(|d| d.insert_temp(id.with("copied"), ui.input(|i| i.time)));
        }
    });
}

/// A LoRA's trigger words and training facts, read from its file the first
/// time (from the other PC's, for its LoRAs).
fn lora_facts(ui: &mut egui::Ui, state: &mut GenerateState, i: usize, id: egui::Id) {
    let l = &mut state.loras[i];
    if l.info.is_none() {
        l.info = match picker_host() {
            Some(_) => remote_lora_info_cached(&l.file, ui.ctx()),
            None => Some(read_lora_info(&comfy_base().join("ComfyUI").join("models").join("loras").join(&l.file))),
        };
    }
    let Some(info) = l.info.clone() else {
        ui.label(RichText::new("Loading…").font(medium(12.0)).color(MUTED()));
        return;
    };
    if info.triggers.is_empty() && info.facts.is_empty() {
        ui.label(RichText::new("No details in this file.").font(medium(12.0)).color(MUTED()));
    }
    if !info.triggers.is_empty() {
        let words = info.triggers.join(", ");
        copy_heading(ui, "Trigger words", &words, id);
        ui.label(RichText::new(words).font(medium(12.5)).color(MUTED()));
        ui.add_space(4.0);
    }
    for (label, value) in &info.facts {
        crate::settings::label_value_row(ui, &format!("{label}:"), value, 11.5);
    }
}
