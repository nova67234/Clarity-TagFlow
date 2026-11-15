//! Text to Image: the view of the image families (SDXL, Anima, Krea 2, Flux
//! and Z-Image), one column from the top:
//!
//! - the header: the orb, the title over the status, and the setup button;
//! - the model bar: the family and the model, each a list;
//! - the prompt box, which has the height the rest leaves it, with the "+"
//!   menu and send in its bottom corners;
//! - the settings bar: the image's shape and size, the steps, the guidance
//!   and the seed, each opening over the bar, one at a time;
//! - how many images this session made, and the log.
//!
//! The video Directors keep the layout in [`super::show_inner`].

use super::*;
use crate::settings::hairline;
use crate::tag_manager::{HEADER_H, at, bold, chip_fill, faint, medium, semibold, view_header, well};

/// The settings of the bar under the prompt box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Setting {
    Image,
    Steps,
    Guidance,
    Seed,
}

const SETTINGS: [Setting; 4] = [Setting::Image, Setting::Steps, Setting::Guidance, Setting::Seed];

/// The families, as the family list has them.
const FAMILIES: [GenFamily; 5] =
    [GenFamily::Sdxl, GenFamily::Anima, GenFamily::Krea2, GenFamily::Flux, GenFamily::ZImage];

/// The heights of the view's rows, and the gap between them.
const BAR_H: f32 = 52.0;
const GAP: f32 = 12.0;
const SESSION_GAP: f32 = 10.0;
const SESSION_H: f32 = 16.0;
const LOG_ROW_H: f32 = 24.0;
const CONSOLE_GAP: f32 = 8.0;
const CONSOLE_H: f32 = 140.0;
/// The prompt box is never shorter: three lines over its buttons.
const PROMPT_MIN_H: f32 = 148.0;
/// The prompt box's row of buttons.
const FOOTER_H: f32 = extras::PILL_H;
/// What an import says, over the row of buttons.
const NOTE_H: f32 = 16.0;

/// What a bar holds sits this far in from its edge: 40 tall.
const INSET: f32 = 6.0;

/// An open setting, over the bar's segments: where its first row starts
/// under the bar's top edge, a row's height, the shapes' height, and what's
/// under its last row (space, the hairline, space).
const EDITOR_TOP: f32 = 12.0;
const EDITOR_ROW_H: f32 = 32.0;
const SHAPES_H: f32 = 62.0;
const SHAPES_GAP: f32 = 8.0;
const EDITOR_FOOT: f32 = 10.0 + 1.0 + 5.0;

/// How much taller the settings bar is with `setting` open.
fn editor_height(setting: Setting) -> f32 {
    let rows = match setting {
        Setting::Image => SHAPES_H + SHAPES_GAP + EDITOR_ROW_H,
        Setting::Steps | Setting::Guidance | Setting::Seed => EDITOR_ROW_H,
    };
    EDITOR_TOP - INSET + rows + EDITOR_FOOT
}

/// The view, in what's left of the right panel's card.
pub(super) fn show(ui: &mut egui::Ui, state: &mut GenerateState) {
    sync_remote(state, ui.ctx());
    drain_runner(state, ui.ctx());
    popups(ui.ctx(), state);

    // What's under the prompt box is settled before any of it is drawn: a
    // click in one row mustn't change another's height in the frame they
    // were measured for.
    let open = state.open_setting;
    let session = !state.gen_images.is_empty();
    let log = state.show_log;
    let above = HEADER_H + GAP + BAR_H + GAP;
    let below = GAP
        + BAR_H
        + open.map_or(0.0, editor_height)
        + if session { SESSION_GAP + SESSION_H } else { 0.0 }
        + GAP
        + LOG_ROW_H
        + if log { CONSOLE_GAP + CONSOLE_H } else { 0.0 };
    let room = ui.available_height();
    let prompt_h = (room - above - below).max(PROMPT_MIN_H);
    let family = state.family.title();

    let mut column = |ui: &mut egui::Ui| {
        // (The gaps are given in full.)
        ui.spacing_mut().item_spacing.y = 0.0;
        header(ui, state);
        ui.add_space(GAP);
        model_bar(ui, state);
        ui.add_space(GAP);
        prompt_box(ui, state, prompt_h);
        ui.add_space(GAP);
        settings_bar(ui, state, open);
        if session {
            ui.add_space(SESSION_GAP);
            session_line(ui, state);
        }
        ui.add_space(GAP);
        log_row(ui, state);
        if log {
            ui.add_space(CONSOLE_GAP);
            console(ui, state);
        }
    };
    if above + prompt_h + below > room + 0.5 {
        // Too short for all of it (a small window with a setting and the log
        // open): the view scrolls rather than run past its card. The scroll
        // bar goes in the card's margin, so it rides the panel's edge
        // instead of the ends of the bars.
        const SCROLL_GUTTER: f32 = 12.0;
        let mut wide = crate::edge_scroll_ui(ui, SCROLL_GUTTER);
        egui::ScrollArea::vertical().id_salt(("t2i_view", family)).auto_shrink([false, false]).show(&mut wide, |ui| {
            ui.set_max_width(ui.available_width() - SCROLL_GUTTER);
            column(ui);
        });
        crate::edge_scroll_done(ui, &wide, SCROLL_GUTTER);
    } else {
        ui.scope(column);
    }
}

/// The status line and its colour. Before generation is set up on this PC
/// (`needs_setup`) the app's "Ready" would send a first prompt nowhere: the
/// line says what is needed instead (the setup button has it).
fn shown_status(state: &GenerateState, needs_setup: bool) -> (String, Color32) {
    if state.status_err {
        (state.status.clone(), RED)
    } else if state.running {
        (state.status.clone(), ACCENT1())
    } else if needs_setup && state.remote_host.is_none() && state.status == "Ready" {
        ("Setup needed".to_string(), ACCENT1())
    } else {
        (state.status.clone(), GREEN)
    }
}

/// Generation can't run until something behind the setup button is seen to:
/// it isn't set up on this PC, or on the Link host this tab runs on.
fn needs_setup(state: &GenerateState) -> bool {
    match (&state.remote_host, &state.remote_catalog) {
        (None, _) => !is_installed(),
        (Some(_), Some(catalog)) => !catalog.installed,
        (Some(_), None) => catalog_error().is_some(),
    }
}

/// The chosen model is gated and there's no Hugging Face token to download
/// it with: the setup card has the box for one.
fn needs_token(state: &GenerateState) -> bool {
    state.remote_host.is_none() && state.model.gated() && state.hf_token.trim().is_empty()
}

/// The header: the orb, the title with the status under it, and the setup
/// button, which carries a dot while its card has something to see to
/// ([`needs_setup`], [`needs_token`]).
fn header(ui: &mut egui::Ui, state: &mut GenerateState) {
    let needs_setup = needs_setup(state);
    let (status, color) = shown_status(state, needs_setup);
    state.orb.set_state(if state.status_err {
        crate::ai_orb::OrbState::Error
    } else if state.running {
        crate::ai_orb::OrbState::Thinking
    } else {
        crate::ai_orb::OrbState::Idle
    });
    let gear = view_header(ui, &mut state.orb, "Text to Image", &status, color, "Setup", state.show_setup);
    if (needs_setup || needs_token(state)) && ui.is_rect_visible(gear.rect) {
        // On the button's shoulder, ringed in the panel's colour so it reads
        // as a dot on the button rather than a bite out of it.
        let at = gear.rect.center() + egui::vec2(12.5, -12.5);
        ui.painter().circle_filled(at, 5.0, PANEL());
        ui.painter().circle_filled(at, 3.5, selection_outline());
    }
    if gear.clicked() {
        state.show_setup = !state.show_setup;
    }
    setup_card(&gear, state);
}

/// The setup card's width.
const SETUP_W: f32 = 248.0;

/// The card under the setup button: on this PC, Setup Requirements and
/// whether it has run (and the Hugging Face token a gated model needs); on a
/// Link host, where the job runs and what that PC has.
fn setup_card(anchor: &egui::Response, state: &mut GenerateState) {
    if !state.show_setup {
        return;
    }
    // (The card closes on Escape by itself.)
    let mut open = true;
    let mut run = false;
    egui::Popup::from_response(anchor)
        .open_bool(&mut open)
        .align(egui::RectAlign::BOTTOM_END)
        .width(SETUP_W)
        .gap(6.0)
        .frame(crate::float_card_frame(18).inner_margin(Margin::same(14)))
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.set_width(SETUP_W - 28.0);
            match state.remote_host.clone() {
                Some(host) => remote_card(ui, state, &host),
                None => run = local_card(ui, state),
            }
        });
    state.show_setup = open && !run;
    if run {
        start_setup(state, &anchor.ctx);
    }
}

/// The setup card on this PC. True when Setup Requirements was clicked.
fn local_card(ui: &mut egui::Ui, state: &mut GenerateState) -> bool {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 20.0), egui::Sense::hover());
    let mut line = at(ui, row);
    line.add(egui::Label::new(RichText::new("Requirements").font(semibold(13.0)).color(TEXT())));
    line.with_layout(Layout::right_to_left(Align::Center), |ui| {
        let note = if is_installed() {
            RichText::new("Installed").font(semibold(12.0)).color(GREEN)
        } else {
            RichText::new("NVIDIA GPU required").font(medium(11.5)).color(MUTED())
        };
        ui.add(egui::Label::new(note));
    });
    ui.add_space(12.0);

    // Faded while a job runs: a click would start a second one.
    let run = wide_button(ui, "Setup Requirements", !state.running)
        .on_hover_text(if state.running {
            "Wait for the running job to finish"
        } else {
            "Download and install what generation needs"
        })
        .clicked();

    if state.model.gated() {
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.add(egui::Label::new(RichText::new("HF token (dev is gated)").font(medium(11.5)).color(MUTED())));
            ui.add(
                egui::Image::new(egui::include_image!("../../icons/info.svg"))
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(icon_tint(MUTED())),
            )
            .on_hover_ui(|ui| {
                ui.set_max_width(260.0);
                ui.label("Flux dev is a gated model: accept its license on Hugging Face and paste a token.");
                crate::arrow_link(ui, "Manage tokens", "https://huggingface.co/settings/tokens", None);
            });
        });
        ui.add_space(6.0);
        let field = egui::TextEdit::singleline(&mut state.hf_token).password(true);
        // Kept as it's typed rather than when the box loses focus: the card
        // can close (its button does it) without being drawn again, and the
        // box would never see its focus go.
        if crate::settings::pill_field(ui, f32::INFINITY, field).changed() {
            keep_token(&state.hf_token);
        }
    }
    run
}

/// Keep the Hugging Face token for the next launch (encrypted, with the
/// models).
fn keep_token(token: &str) {
    #[cfg(not(test))]
    crate::pixal3d::save_hf_token(token);
    // (A test mustn't write over this machine's own token: it sees what
    // would have been kept instead.)
    #[cfg(test)]
    KEPT_TOKEN_FOR_TEST.set(Some(token.to_string()));
}

#[cfg(test)]
thread_local! {
    /// What [`keep_token`] was last given, in a test.
    static KEPT_TOKEN_FOR_TEST: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// The setup card while a Link host runs this tab's jobs.
fn remote_card(ui: &mut egui::Ui, state: &GenerateState, host: &str) {
    ui.add(
        egui::Label::new(RichText::new(format!("Runs on {host}")).font(semibold(13.0)).color(TEXT())).truncate(),
    )
    .on_hover_text("Settings → Link: this PC uses that PC's GPU and models");
    ui.add_space(4.0);
    let note = match &state.remote_catalog {
        None if catalog_error().is_some() => "Couldn't get its models",
        None => "Checking its models…",
        Some(c) if !c.installed => "Generation isn't set up there",
        Some(_) => "Its GPU and models",
    };
    ui.add(egui::Label::new(RichText::new(note).font(medium(11.5)).color(MUTED())).wrap());
}

/// A pill-shaped button as wide as its column, 36 tall. Not `enabled`, it
/// only shows.
fn wide_button(ui: &mut egui::Ui, label: &str, enabled: bool) -> egui::Response {
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 36.0), sense);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, label));
    if ui.is_rect_visible(rect) {
        let ink = if enabled { TEXT() } else { faint() };
        let fill = if !enabled {
            FIELD().gamma_multiply(0.6)
        } else if resp.is_pointer_button_down_on() {
            FIELD2().lerp_to_gamma(ink, 0.05)
        } else if resp.hovered() || resp.has_focus() {
            FIELD2().lerp_to_gamma(ink, 0.12)
        } else {
            FIELD2()
        };
        ui.painter().rect_filled(rect, 18.0, fill);
        let text = ui.painter().layout_no_wrap(label.to_string(), semibold(12.5), ink);
        ui.painter().galley(rect.center() - text.size() / 2.0, text, ink);
    }
    if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp }
}

/// The family picker's width in the model bar, and its list's.
const FAMILY_W: f32 = 124.0;
const FAMILY_LIST_W: f32 = 172.0;

/// The model bar: one sunk pill holding the family and, in the rest of it,
/// the model. Their lists hang from the bar, each at its own end.
fn model_bar(ui: &mut egui::Ui, state: &mut GenerateState) {
    let (bar, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), BAR_H), egui::Sense::hover());
    ui.painter().rect(bar, BAR_H / 2.0, well(), egui::Stroke::new(1.0_f32, hairline()), egui::StrokeKind::Inside);
    let inner = bar.shrink(INSET);
    let family = egui::Rect::from_min_size(inner.min, egui::vec2(FAMILY_W.min(inner.width() * 0.45), inner.height()));
    let model = egui::Rect::from_min_max(egui::pos2(family.right() + 5.0, inner.top()), inner.max);
    let x = family.right() + 2.5;
    ui.painter().line_segment(
        [egui::pos2(x, inner.top() + 9.0), egui::pos2(x, inner.bottom() - 9.0)],
        egui::Stroke::new(1.0_f32, hairline()),
    );
    // A list hangs from the whole bar, not from its own picker.
    let from_bar = |resp: egui::Response| {
        let mut anchor = resp;
        anchor.rect = bar;
        anchor.interact_rect = bar;
        anchor
    };

    let id = egui::Id::new(("t2i_family", state.family.title()));
    let open = egui::Popup::is_id_open(ui.ctx(), id);
    let resp = picker(ui, family, id, "Family", state.family.title(), None, open);
    egui::Popup::menu(&from_bar(resp))
        .id(id)
        .align(egui::RectAlign::BOTTOM_START)
        .gap(6.0)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .frame(crate::settings::dropdown_frame())
        .show(|ui| {
            ui.set_width(FAMILY_LIST_W);
            ui.spacing_mut().item_spacing.y = 3.0;
            for family in FAMILIES {
                let chosen = state.family == family;
                if crate::settings::dropdown_option(ui, family.title(), None, None, chosen).clicked() {
                    if !chosen {
                        // (The right panel shows that family's own tab.)
                        state.family_switch = Some(family);
                    }
                    ui.close();
                }
            }
        });

    let id = egui::Id::new(("t2i_model", state.family.title()));
    let (name, note) = model_text(state);
    let resp = picker(ui, model, id, "Model", &name, note.as_deref(), state.model_list_open);
    if resp.clicked() {
        state.model_list_open = !state.model_list_open;
    }
    // The installed checkpoints are looked for once each time the list
    // opens, so a new download is there without a restart.
    // (Civitai's previews are looked up while it's open.)
    let open = state.model_list_open;
    let opening = open && !state.ckpt_menu_was_open;
    if opening {
        if state.family.uses_checkpoint_picker() {
            refresh_checkpoints(state);
        }
        models::opened(state, ui.ctx());
    } else if !open && state.ckpt_menu_was_open {
        models::closed();
    }
    state.ckpt_menu_was_open = open;
    if open {
        models::list(&from_bar(resp), id, state, opening);
    }
}

/// The chosen model as the bar shows it: its name and, for a built-in one,
/// a note (its size, or what it asks of the GPU).
fn model_text(state: &GenerateState) -> (String, Option<String>) {
    match &state.checkpoint {
        Some(file) => (ckpt_display_name(file).to_string(), None),
        None => short_label(state.model.label(), state.family.title()),
    }
}

/// A built-in model's label without what the bar says beside it: in SDXL's
/// bar "SDXL Base 1.0 (~6.9 GB)" is "Base 1.0", with the note "6.9 GB".
pub(super) fn short_label(label: &str, family: &str) -> (String, Option<String>) {
    // (The family's name as a word of its own: "SDXL ", not "SDXLent".)
    let rest = label
        .strip_prefix(family)
        .and_then(|rest| rest.strip_prefix(' '))
        .map(str::trim_start)
        .filter(|rest| !rest.is_empty())
        .unwrap_or(label);
    match rest.strip_suffix(')').and_then(|rest| rest.rsplit_once(" (")) {
        Some((name, note)) if !name.is_empty() => {
            (name.to_string(), Some(note.trim_start_matches('~').to_string()))
        }
        _ => (rest.to_string(), None),
    }
}

/// One of the model bar's pickers, over `rect`: `caption` over `value` and,
/// where there's room for all of it after that, a quieter `note`. A value
/// too long for the picker is cut short where the chevron begins; whatever
/// isn't shown is on hover. The chevron turns while the list `id` is open.
fn picker(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
    caption: &str,
    value: &str,
    note: Option<&str>,
    open: bool,
) -> egui::Response {
    let mut resp =
        ui.interact(rect, id.with("picker"), egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::ComboBox, true, format!("{caption}: {value}")));
    let turn = ui.ctx().animate_bool(id.with("chevron"), open);
    if ui.is_rect_visible(rect) {
        let hot = resp.hovered() || resp.has_focus() || open;
        if hot {
            ui.painter().rect_filled(rect, rect.height() / 2.0, TEXT().gamma_multiply(0.05));
        }
        let left = rect.left() + 14.0;
        let room = (rect.right() - 30.0 - left).max(0.0);
        let line = |note: Option<&str>| {
            let mut job = egui::text::LayoutJob::simple_singleline(value.to_string(), semibold(13.0), TEXT());
            if let Some(note) = note {
                let quiet = egui::TextFormat { font_id: medium(13.0), color: faint(), ..Default::default() };
                job.append(&format!(" · {note}"), 0.0, quiet);
            }
            job.wrap = egui::text::TextWrapping::truncate_at_width(room);
            job
        };
        let (caption, text, whole, caption_h, text_h) = ui.fonts_mut(|f| {
            let caption = f.layout_no_wrap(caption.to_string(), semibold(10.5), faint());
            let mut text = f.layout_job(line(note));
            let mut whole = !text.elided;
            if !whole && note.is_some() {
                // Half a note says less than none: the value alone.
                text = f.layout_job(line(None));
                whole = false;
            }
            (caption, text, whole, f.row_height(&semibold(10.5)), f.row_height(&semibold(13.0)))
        });
        if !whole {
            resp = resp.on_hover_text(match note {
                Some(note) => format!("{value} · {note}"),
                None => value.to_string(),
            });
        }
        let top = rect.center().y - (caption_h + text_h) / 2.0;
        ui.painter().galley(egui::pos2(left, top), caption, faint());
        ui.painter().galley(egui::pos2(left, top + caption_h), text, TEXT());
        let chevron = egui::pos2(rect.right() - 17.0, rect.center().y);
        crate::settings::paint_chevron(ui.painter(), chevron, turn, if hot { TEXT() } else { MUTED() });
    }
    resp
}

/// The prompt box, `height` tall: the prompt (long text scrolls inside)
/// over its row of buttons — the "+" menu in the bottom-left corner, send
/// (stop, while a job runs) in the bottom-right one, and between them the
/// LoRAs | Embeddings pill; what an import says sits over the row. A file
/// dropped on the box is imported.
fn prompt_box(ui: &mut egui::Ui, state: &mut GenerateState, height: f32) {
    const RADIUS: f32 = 22.0;
    let drag_over = dragging_over_prompt(ui, state);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    ui.painter().rect(rect, RADIUS, well(), egui::Stroke::new(1.0_f32, hairline()), egui::StrokeKind::Inside);

    // The buttons' row first: the text has what's over it (and what an
    // import says, over the row, when it says something).
    let footer = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 12.0, rect.bottom() - 12.0 - FOOTER_H),
        egui::pos2(rect.right() - 12.0, rect.bottom() - 12.0),
    );
    let note = import_note(state, drag_over);
    let note_rect = note.as_ref().map(|_| {
        egui::Rect::from_min_max(
            egui::pos2(rect.left() + 18.0, footer.top() - 6.0 - NOTE_H),
            egui::pos2(rect.right() - 18.0, footer.top() - 6.0),
        )
    });
    // (The text runs to the box's edge less the scroll bar's room, so the
    // bar rides the edge rather than the words.)
    let text = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 18.0, rect.top() + 14.0),
        egui::pos2(rect.right() - 8.0, note_rect.map_or(footer.top() - 8.0, |r| r.top() - 4.0)),
    );
    let mut text_ui = ui.new_child(egui::UiBuilder::new().max_rect(text).layout(Layout::top_down(Align::Min)));
    egui::ScrollArea::vertical()
        .id_salt(("t2i_prompt", state.family.title()))
        .max_height(text.height())
        .auto_shrink([false, false])
        .show(&mut text_ui, |ui| {
            // Keep scrolling while a text selection is dragged past the edge.
            crate::drag_select_autoscroll(ui);
            let font = medium(15.0);
            // Enough rows to fill the box, so a click anywhere in it lands
            // on the text (the row count is the only lever on the height).
            let row_h = ui.fonts_mut(|f| f.row_height(&font));
            let rows = ((text.height() / row_h).floor() as usize).max(2);
            // (The hint is set in the box's face too: left to itself it takes
            // the app's body text.)
            let hint = RichText::new("a cinematic photo of…").font(font.clone()).color(faint());
            let out = egui::TextEdit::multiline(&mut state.prompt)
                .font(font)
                // Explicit: the Light theme's widget text is white (for its
                // blue buttons).
                .text_color(TEXT())
                .hint_text(hint)
                .frame(egui::Frame::NONE)
                .margin(Margin::ZERO)
                .desired_width(ui.available_width() - 10.0)
                .desired_rows(rows)
                .show(ui);
            // Live spell-check: red squiggles + right-click suggestions.
            crate::spellcheck::attach(ui, &out, &mut state.prompt, &mut state.spell);
        });

    // The "+" and send in the corners, the LoRAs | Embeddings pill between.
    let button = egui::vec2(28.0, 28.0);
    let plus = egui::Rect::from_center_size(egui::pos2(footer.left() + 14.0, footer.center().y), button);
    let send = egui::Rect::from_center_size(egui::pos2(footer.right() - 14.0, footer.center().y), button);
    add_menu(&mut at(ui, plus), state, false);
    if send_or_stop(&mut at(ui, send), state, None) == SendClick::NotSetUp {
        // Its hover says to run Setup Requirements first: a click shows
        // where that is.
        state.show_setup = true;
    }
    let pill = egui::Rect::from_min_max(egui::pos2(plus.right() + 10.0, footer.top()), egui::pos2(send.left() - 10.0, footer.bottom()));
    extras::pill(ui, pill, rect, state);
    if let (Some((note, color, why)), Some(note_rect)) = (note, note_rect) {
        let label = at(ui, note_rect).add(egui::Label::new(RichText::new(note).font(medium(11.5)).color(color)).truncate());
        if let Some(why) = why {
            label.on_hover_text(why);
        }
    }

    prompt_drop(ui, state, rect, drag_over);
}

/// What the prompt box says beside its "+": that a file over it will be
/// imported, how an import's custom nodes are coming along, or that the
/// imported workflow is the one Generate runs. With why, where that helps.
fn import_note(state: &GenerateState, drag_over: bool) -> Option<(String, Color32, Option<&'static str>)> {
    if drag_over {
        Some(("Drop to import prompt + install its tools".to_string(), DROP_BLUE, None))
    } else if state.running {
        // While installing or generating, the live status.
        state.nodes_status.clone().map(|status| (status, DROP_BLUE, None))
    } else if state.imported_workflow.is_some() {
        Some((
            "⚡ Custom nodes active — using imported workflow".to_string(),
            GREEN,
            Some(
                "Generate runs the dropped image's own ComfyUI graph (custom nodes and all).\nClear it from the + menu to use the built-in pipeline.",
            ),
        ))
    } else {
        state.nodes_status.clone().map(|status| (status, GREEN, None))
    }
}

/// The settings bar: one sunk pill of four segments — the image's shape
/// and size, the steps, the guidance and the seed — each with its value
/// under its name. A click on one opens it over the segments (`open`: the
/// one that is), and the bar is that much taller.
fn settings_bar(ui: &mut egui::Ui, state: &mut GenerateState, open: Option<Setting>) {
    let height = BAR_H + open.map_or(0.0, editor_height);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    ui.painter().rect(rect, BAR_H / 2.0, well(), egui::Stroke::new(1.0_f32, hairline()), egui::StrokeKind::Inside);
    let row = egui::Rect::from_min_max(
        egui::pos2(rect.left() + INSET, rect.bottom() - BAR_H + INSET),
        egui::pos2(rect.right() - INSET, rect.bottom() - INSET),
    );
    if let Some(setting) = open {
        let y = row.top() - 5.5;
        ui.painter().line_segment(
            [egui::pos2(rect.left() + 16.0, y), egui::pos2(rect.right() - 16.0, y)],
            egui::Stroke::new(1.0_f32, hairline()),
        );
        let editor = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 14.0, rect.top() + EDITOR_TOP),
            egui::pos2(rect.right() - 14.0, row.top() - EDITOR_FOOT),
        );
        let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(editor).layout(Layout::top_down(Align::Min)));
        ui.spacing_mut().item_spacing.y = 0.0;
        match setting {
            Setting::Image => image_editor(&mut ui, state),
            Setting::Steps => {
                let default = state.model.default_steps() as f64;
                slider_row(&mut ui, None, PillSlider::new(&mut state.steps, 1..=50).default_value(default));
            }
            Setting::Guidance => {
                let default = state.model.default_cfg() as f64;
                slider_row(&mut ui, None, PillSlider::new(&mut state.cfg, 1.0..=8.0).default_value(default));
            }
            Setting::Seed => seed_editor(&mut ui, state),
        }
    }

    // The segments: the first is wider, for the image's size.
    const WEIGHTS: [f32; 4] = [1.7, 1.0, 1.0, 1.0];
    // Between two: a hairline with 2 either side.
    const BETWEEN: f32 = 5.0;
    let unit = (row.width() - BETWEEN * 3.0) / WEIGHTS.iter().sum::<f32>();
    let mut left = row.left();
    for (i, setting) in SETTINGS.into_iter().enumerate() {
        let segment_rect = egui::Rect::from_min_size(egui::pos2(left, row.top()), egui::vec2(unit * WEIGHTS[i], row.height()));
        // (No line beside the open one: its pill is the edge.)
        if i > 0 && open != Some(setting) && open != Some(SETTINGS[i - 1]) {
            let x = left - BETWEEN / 2.0;
            ui.painter().line_segment(
                [egui::pos2(x, row.top() + 10.0), egui::pos2(x, row.bottom() - 10.0)],
                egui::Stroke::new(1.0_f32, hairline()),
            );
        }
        if segment(ui, state, segment_rect, setting, open == Some(setting)).clicked() {
            state.open_setting = if open == Some(setting) { None } else { Some(setting) };
        }
        left = segment_rect.right() + BETWEEN;
    }
}

/// The three shapes, as the editor's tiles and the Image segment's icon
/// have them: (icon, name), in [`aspect_idx`]'s order.
fn shapes() -> [(egui::ImageSource<'static>, &'static str); 3] {
    [
        (egui::include_image!("../../icons/square.svg"), "Square"),
        (egui::include_image!("../../icons/landscape.svg"), "Landscape"),
        (egui::include_image!("../../icons/portrait.svg"), "Portrait"),
    ]
}

/// The guidance as a segment shows it: to the two decimals the slider keeps,
/// less a last zero ("7.0", "3.5", "7.25").
fn guidance_text(cfg: f32) -> String {
    let text = format!("{cfg:.2}");
    text.strip_suffix('0').map(str::to_string).unwrap_or(text)
}

/// One segment of the settings bar over `rect`: the setting's name over its
/// value (cut short, if it must be, with the whole of it on hover), on a
/// raised pill while it's the `open` one.
fn segment(ui: &mut egui::Ui, state: &GenerateState, rect: egui::Rect, setting: Setting, open: bool) -> egui::Response {
    let (name, value, tip) = match setting {
        Setting::Image => ("Image", format!("{} × {}", state.width, state.height), "Shape and size"),
        Setting::Steps => ("Steps", state.steps.to_string(), "Steps"),
        Setting::Guidance => ("Guidance", guidance_text(state.cfg), "Guidance (CFG)"),
        Setting::Seed if state.randomize_seed => ("Seed", "Random".to_string(), "Seed"),
        Setting::Seed => ("Seed", state.seed.to_string(), "Seed"),
    };
    let id = egui::Id::new(("t2i_setting", state.family.title(), name));
    let mut resp = ui.interact(rect, id, egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    resp.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Button, true, open, format!("{tip}: {value}"))
    });
    if !ui.is_rect_visible(rect) {
        return resp;
    }
    let hot = resp.hovered() || resp.has_focus();
    if open {
        ui.painter().rect_filled(rect, rect.height() / 2.0, chip_fill(hot));
    } else if hot {
        ui.painter().rect_filled(rect, rect.height() / 2.0, TEXT().gamma_multiply(0.05));
    }
    // The image's value starts with its shape.
    let icon = (setting == Setting::Image).then(|| shapes()[aspect_idx(state.width, state.height)].0.clone());
    let (icon_w, icon_gap) = if icon.is_some() { (13.0, 5.0) } else { (0.0, 0.0) };
    let room = (rect.width() - 12.0 - icon_w - icon_gap).max(0.0);
    let name_ink = if open { MUTED() } else { faint() };
    let (name, text, name_h, text_h) = ui.fonts_mut(|f| {
        let name = f.layout_no_wrap(name.to_string(), semibold(10.5), name_ink);
        let mut job = egui::text::LayoutJob::simple_singleline(value.clone(), bold(13.0), TEXT());
        job.wrap = egui::text::TextWrapping::truncate_at_width(room);
        (name, f.layout_job(job), f.row_height(&semibold(10.5)), f.row_height(&bold(13.0)))
    });
    if text.elided {
        resp = resp.on_hover_text(format!("{tip}: {value}"));
    } else if setting != Setting::Steps && setting != Setting::Seed {
        resp = resp.on_hover_text(tip);
    }
    let top = rect.center().y - (name_h + text_h) / 2.0;
    ui.painter().galley(egui::pos2(rect.center().x - name.size().x / 2.0, top), name, name_ink);
    let mut x = rect.center().x - (icon_w + icon_gap + text.size().x) / 2.0;
    let mid = top + name_h + text_h / 2.0;
    if let Some(icon) = icon {
        let slot = egui::Rect::from_min_size(egui::pos2(x, mid - icon_w / 2.0), egui::Vec2::splat(icon_w));
        egui::Image::new(icon).tint(icon_tint(MUTED())).paint_at(ui, slot);
        x += icon_w + icon_gap;
    }
    ui.painter().galley(egui::pos2(x, mid - text.size().y / 2.0), text, TEXT());
    resp
}

/// The Image setting, open: the three shapes, then the size they scale to.
fn image_editor(ui: &mut egui::Ui, state: &mut GenerateState) {
    const BETWEEN: f32 = 6.0;
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), SHAPES_H), egui::Sense::hover());
    let width = (row.width() - BETWEEN * 2.0) / 3.0;
    let active = aspect_idx(state.width, state.height);
    for (i, (icon, name)) in shapes().into_iter().enumerate() {
        let tile = egui::Rect::from_min_size(
            egui::pos2(row.left() + (width + BETWEEN) * i as f32, row.top()),
            egui::vec2(width, SHAPES_H),
        );
        let id = egui::Id::new(("t2i_shape", state.family.title(), name));
        // (What the shape comes to at this size is on its hover.)
        let (w, h) = aspect_dims(state.family, i, state.size);
        if shape_tile(ui, tile, id, icon, name, i == active).on_hover_text(format!("{w} × {h}")).clicked() {
            state.width = w;
            state.height = h;
        }
    }
    ui.add_space(SHAPES_GAP);

    // One size scales all three shapes: the chosen one follows the slider
    // (snapped to the family's grid).
    let mut size = state.size;
    slider_row(ui, Some("Size"), PillSlider::new(&mut size, 512..=2048).step_by(64.0).default_value(DEFAULT_SIZE as f64));
    if size != state.size {
        state.size = size;
        let (w, h) = aspect_dims(state.family, aspect_idx(state.width, state.height), size);
        state.width = w;
        state.height = h;
    }
}

/// One of the Image setting's shapes over `rect`: its icon over its name,
/// on a raised tile while it's the `chosen` one.
fn shape_tile(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
    icon: egui::ImageSource<'static>,
    name: &str,
    chosen: bool,
) -> egui::Response {
    let resp = ui.interact(rect, id, egui::Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::RadioButton, true, chosen, name));
    if ui.is_rect_visible(rect) {
        let hot = resp.hovered() || resp.has_focus();
        if chosen {
            ui.painter().rect_filled(rect, 16.0, chip_fill(hot));
        } else if hot {
            ui.painter().rect_filled(rect, 16.0, TEXT().gamma_multiply(0.05));
        }
        let ink = if chosen || hot { TEXT() } else { MUTED() };
        let text = ui.painter().layout_no_wrap(name.to_string(), semibold(12.5), ink);
        // The icon and the name, as one block in the middle of the tile.
        let top = rect.center().y - (18.0 + 6.0 + text.size().y) / 2.0;
        let slot = egui::Rect::from_min_size(egui::pos2(rect.center().x - 9.0, top), egui::Vec2::splat(18.0));
        egui::Image::new(icon).tint(icon_tint(ink)).paint_at(ui, slot);
        ui.painter().galley(egui::pos2(rect.center().x - text.size().x / 2.0, top + 24.0), text, ink);
    }
    resp
}

/// A row of an open setting holding `slider` — its pill, then its track
/// across what's left — after `label`, if it has one.
fn slider_row(ui: &mut egui::Ui, label: Option<&str>, slider: PillSlider<'_>) {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), EDITOR_ROW_H), egui::Sense::hover());
    let mut ui = at(ui, row);
    ui.spacing_mut().item_spacing.x = 12.0;
    if let Some(label) = label {
        ui.add(egui::Label::new(RichText::new(label).font(medium(12.5)).color(MUTED())));
    }
    // (The track takes the row: the slider keeps it to what's there.)
    ui.spacing_mut().slider_width = f32::INFINITY;
    ui.add(slider);
}

/// The Seed setting, open: a new seed each run or, with that off, the one
/// in the number box.
fn seed_editor(ui: &mut egui::Ui, state: &mut GenerateState) {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), EDITOR_ROW_H), egui::Sense::hover());
    let mut ui = at(ui, row);
    ui.spacing_mut().item_spacing.x = 12.0;
    ui.add(egui::Label::new(RichText::new("Random").font(medium(12.5)).color(MUTED())));
    crate::settings::switch(&mut ui, &mut state.randomize_seed)
        .on_hover_text("A new seed for each image");
    if !state.randomize_seed {
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| seed_field(ui, &mut state.seed));
    }
}

/// The seed's number box, in the look of the sliders' value pills: click it
/// to type, or drag across it.
fn seed_field(ui: &mut egui::Ui, seed: &mut i64) {
    const SIZE: egui::Vec2 = egui::vec2(150.0, 24.0);
    let (fill, edge, ink) = crate::slider::pill_colors();
    ui.allocate_ui_with_layout(SIZE, Layout::centered_and_justified(egui::Direction::TopDown), |ui| {
        let style = ui.style_mut();
        style.spacing.interact_size = SIZE;
        style.spacing.button_padding = egui::vec2(10.0, 0.0);
        style.override_font_id = Some(egui::FontId::proportional(13.0));
        let visuals = &mut style.visuals;
        visuals.text_edit_bg_color = Some(fill);
        visuals.selection.stroke = egui::Stroke::new(1.0_f32, selection_outline());
        // Explicit: the Light theme's widget text is white (for its blue
        // buttons).
        visuals.override_text_color = Some(ink);
        for widget in [&mut visuals.widgets.inactive, &mut visuals.widgets.hovered, &mut visuals.widgets.active] {
            widget.bg_fill = fill;
            widget.weak_bg_fill = fill;
            widget.bg_stroke = egui::Stroke::new(1.0_f32, edge);
            widget.fg_stroke.color = ink;
            widget.corner_radius = CornerRadius::same(12);
            widget.expansion = 0.0;
        }
        ui.add(egui::DragValue::new(seed)).on_hover_text("The seed every image starts from")
    });
}

/// How many images this session made, and where they are: in the left
/// browser and the centre viewer while this view is open.
fn session_line(ui: &mut egui::Ui, state: &GenerateState) {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), SESSION_H), egui::Sense::hover());
    let text = format!("{} image(s) this session — browse them on the left.", state.gen_images.len());
    at(ui, row).add(egui::Label::new(RichText::new(text).font(medium(11.5)).color(MUTED())).truncate());
}

/// The row under the settings: "Log", which opens the console under it,
/// and at the row's end the icon that copies the log.
fn log_row(ui: &mut egui::Ui, state: &mut GenerateState) {
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), LOG_ROW_H), egui::Sense::hover());

    // (The arrow's picture is smaller than its square: the pill starts
    // before the row, so the arrow lines up with what's over it.)
    let label = ui.painter().layout_no_wrap("Log".to_string(), semibold(12.0), MUTED());
    let pill = egui::Rect::from_min_size(
        egui::pos2(row.left() - 4.0, row.top()),
        egui::vec2(2.0 + 16.0 + 2.0 + label.size().x + 10.0, LOG_ROW_H),
    );
    let toggle = ui
        .interact(pill, ui.id().with(("t2i_log", state.family.title())), egui::Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    toggle.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::CollapsingHeader, true, state.show_log, "Log"));
    let hot = toggle.hovered() || toggle.has_focus();
    if hot {
        ui.painter().rect_filled(pill, LOG_ROW_H / 2.0, TEXT().gamma_multiply(0.06));
    }
    let ink = if hot { TEXT() } else { MUTED() };
    let arrow = if state.show_log {
        egui::include_image!("../../icons/arrow_drop_down.svg")
    } else {
        egui::include_image!("../../icons/arrow_right.svg")
    };
    let slot = egui::Rect::from_min_size(egui::pos2(pill.left() + 2.0, pill.center().y - 8.0), egui::Vec2::splat(16.0));
    egui::Image::new(arrow).tint(icon_tint(ink)).paint_at(ui, slot);
    let label = ui.painter().layout_no_wrap("Log".to_string(), semibold(12.0), ink);
    ui.painter().galley(egui::pos2(slot.right() + 2.0, pill.center().y - label.size().y / 2.0), label, ink);
    if toggle.clicked() {
        state.show_log = !state.show_log;
    }

    // Copy: the icon is green (copied) or red (it failed) for a moment
    // after a click. With nothing logged it only shows.
    const COPY_FLASH_SECS: f32 = 0.9;
    let flash = match state.copy_flash {
        Some((when, ok)) if when.elapsed().as_secs_f32() < COPY_FLASH_SECS => {
            ui.ctx().request_repaint(); // so the flash ends by itself
            Some(ok)
        }
        _ => None,
    };
    let enabled = !state.log.is_empty();
    let button = egui::Rect::from_min_max(egui::pos2(row.right() - 26.0, row.top()), row.max);
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let copy = ui.interact(button, ui.id().with(("t2i_copy_log", state.family.title())), sense).on_hover_text(match flash {
        Some(true) => "Copied!",
        Some(false) => "Copy failed",
        None => "Copy the log",
    });
    copy.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, "Copy the log"));
    let hot = enabled && (copy.hovered() || copy.has_focus());
    if hot {
        ui.painter().rect_filled(button, 12.0, TEXT().gamma_multiply(0.08));
    }
    let tint = match flash {
        Some(true) => GREEN,
        Some(false) => RED,
        None if !enabled => icon_tint(MUTED()).gamma_multiply(0.45),
        None => icon_tint(if hot { TEXT() } else { MUTED() }),
    };
    let icon = egui::Rect::from_center_size(button.center(), egui::Vec2::splat(16.0));
    egui::Image::new(egui::include_image!("../../icons/copy.svg")).tint(tint).paint_at(ui, icon);
    if copy.clicked() {
        copy_log(state);
    }
}

/// The log's console: a well that holds its height whatever is logged,
/// following the last line as lines come.
fn console(ui: &mut egui::Ui, state: &GenerateState) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), CONSOLE_H), egui::Sense::hover());
    ui.painter().rect(
        rect,
        16.0,
        crate::theme::console_bg(),
        egui::Stroke::new(1.0_f32, hairline()),
        egui::StrokeKind::Inside,
    );
    let inner = rect.shrink(10.0);
    let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(Layout::top_down(Align::Min)));
    egui::ScrollArea::vertical()
        .id_salt(("t2i_console", state.family.title()))
        .max_height(inner.height())
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(&mut ui, |ui| {
            // Keep scrolling while a text selection is dragged past the edge.
            crate::drag_select_autoscroll(ui);
            if state.log.is_empty() {
                ui.label(RichText::new("Log output will appear here.").color(MUTED()).monospace().size(12.0));
            } else {
                for line in &state.log {
                    ui.label(RichText::new(line).color(TEXT()).monospace().size(12.0));
                }
            }
        });
}

pub(super) mod extras;
pub(super) mod models;

#[cfg(test)]
mod tests;
