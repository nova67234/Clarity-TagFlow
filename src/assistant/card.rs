//! The work assistant on screen: its pill in the top bar, the chat card that
//! drops from it, and the banner a reply shows as while the card is closed —
//! a notification, to be read and left, or clicked to carry on.

use eframe::egui;
use egui::{Align, Color32, Layout, Sense};

use crate::ai_orb::{AiOrb, OrbState};
use crate::llm::{ChatRole, LlmState};
use crate::theme::*;

/// The pill's size in the top bar.
const PILL: egui::Vec2 = egui::vec2(206.0, 40.0);
/// The card's width, and the banner's.
const CARD_W: f32 = 430.0;
const BANNER_W: f32 = 340.0;
/// How long a banner stays, and how long it takes to fade at the end.
const BANNER_SECS: f64 = 7.0;
const BANNER_FADE: f64 = 0.6;
/// The newest messages the card shows (the conversation itself is whole, in
/// AI Chat's list).
const SHOWN: usize = 40;

/// What the user can ask with a click when the conversation is empty.
const LOOK_FIRST: &str = "What do you think of this image?";
const RECREATE_FIRST: &str = "Recreate this image";
const OPEN_FIRST: &str = "Open Text to Image";

/// The assistant's pill, card and banner, between frames.
#[derive(Default)]
pub struct AssistantUi {
    /// The card is open.
    pub open: bool,
    draft: String,
    /// An image attached to the message being written (the send box's +).
    attached: Option<std::path::PathBuf>,
    /// The app's orb, as AI Chat's thinking row has it at this size (more
    /// particles than its 56 run together into a blob here).
    orb: AiOrb,
    /// Where the pill was drawn (the card hangs under it).
    pill: Option<egui::Rect>,
    /// The conversation as last looked at: its id and length, so a reply
    /// that has come in since is known for new.
    seen: (u64, usize),
    /// A reply is shown as a banner: its place in the conversation, and
    /// since when.
    banner: Option<(usize, f64)>,
    /// A reply came in while the card was closed (the pill's dot).
    unread: bool,
    /// The card was just opened: its text box takes the keyboard.
    focus: bool,
    markdown: egui_commonmark::CommonMarkCache,
}

/// What the card asks of the app.
#[derive(Debug, PartialEq)]
pub enum Ask {
    /// Send this to the assistant.
    Send(String),
    /// Send this with the selected image shown: it asks the assistant to
    /// look, in so many words (the card's own asks).
    Show(String),
    /// Send this with an image the user attached (the send box's +).
    With(String, std::path::PathBuf),
    /// Stop the reply that's coming.
    Stop,
    /// Stop the reply being read aloud (Tools → Auto-speak replies).
    Quiet,
}

/// What a reply still coming is doing, while its words aren't there to
/// read: the line the pill and the card show, and — while it thinks — the
/// end of its thought, so the wait is seen to be going somewhere.
#[derive(Debug, Default, PartialEq)]
struct Doing {
    what: String,
    thought: Option<String>,
}

/// The most of a thought's last line the card shows.
const THOUGHT_CHARS: usize = 96;

/// What the reply that's coming is doing (None: none is). The worker's own
/// line until the first words — loading the model, reading the message —
/// then by what has come: a thought (hidden, but for its last line), words
/// to read, or the prompt of an image, which is the app's to read.
fn doing(llm: &LlmState) -> Option<Doing> {
    if !llm.assistant_busy() {
        return None;
    }
    let raw = llm.assistant_chat().and_then(|c| c.msgs.last()).filter(|m| m.role == ChatRole::Model).map_or("", |m| m.text.as_str());
    let (thought, said) = crate::llm::split_channels(raw);
    // (Counted from when the reply was asked for, then from its first
    // word: `gen_started`.)
    let secs = llm.gen_started.map(|t| t.elapsed().as_secs()).filter(|&s| s >= 2);
    let timed = |what: &str| match secs {
        Some(secs) => format!("{what} {secs} s"),
        None => what.to_string(),
    };
    let dressing = |c: char| c.is_whitespace() || matches!(c, '`' | '*' | '_');
    Some(if said.lines().any(|line| line.trim_start_matches(dressing).starts_with(crate::assistant::GENERATE_TAG)) {
        Doing { what: timed("Writing the prompt…"), thought: None }
    } else if !said.trim().is_empty() {
        Doing { what: "Writing…".to_string(), thought: None }
    } else if !thought.trim().is_empty() {
        let line = thought.lines().rev().map(str::trim).find(|line| !line.is_empty()).unwrap_or_default();
        let skip = line.chars().count().saturating_sub(THOUGHT_CHARS);
        let tail: String = line.chars().skip(skip).collect();
        Doing { what: timed("Thinking…"), thought: Some(if skip > 0 { format!("…{}", tail.trim_start()) } else { tail }) }
    } else {
        // Before the first word: what the worker says it's doing (loading
        // the model, reading the image…) — until it says, that the image is
        // on its way to it, or just that it's thinking.
        let status = llm.status.trim();
        let what = match (status.is_empty() || status == "Starting…", llm.assistant.looking) {
            (true, true) => "Looking at the image…",
            (true, false) => "Thinking…",
            (false, _) => status,
        };
        Doing { what: timed(what), thought: None }
    })
}

/// What the pill says: what the assistant is doing, or just that it's there.
fn pill_text(llm: &LlmState) -> String {
    doing(llm).map_or_else(|| llm.assistant_name().to_string(), |doing| doing.what)
}

/// The pill in the top bar: the orb — alive while a reply is coming — and a
/// line. A click opens the card, or closes it.
pub fn pill(ui: &mut egui::Ui, state: &mut AssistantUi, llm: &LlmState) {
    let (rect, resp) = ui.allocate_exact_size(PILL, Sense::click());
    state.pill = Some(rect);
    let busy = llm.assistant_busy();
    state.orb.set_state(if busy { OrbState::Thinking } else { OrbState::Idle });
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Work assistant"));
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    let p = ui.painter();
    let fill = if resp.hovered() || state.open { FIELD2() } else { FIELD() };
    let edge = if state.open { ACCENT1() } else { EDGE() };
    p.rect(rect, PILL.y / 2.0, fill, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);

    let inner = rect.shrink2(egui::vec2(8.0, 0.0));
    let mut row = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)));
    row.spacing_mut().item_spacing.x = 8.0;
    let orb = state.orb.show(&mut row, 28.0, None);
    let room = (inner.right() - row.cursor().left() - if state.unread { 16.0 } else { 4.0 }).max(0.0);
    row.allocate_ui_with_layout(egui::vec2(room, PILL.y), Layout::left_to_right(Align::Center), |ui| {
        let ink = if busy { MUTED() } else { TEXT() };
        ui.add(egui::Label::new(egui::RichText::new(pill_text(llm)).color(ink).size(13.5)).truncate().selectable(false));
    });
    if state.unread {
        ui.painter().circle_filled(egui::pos2(rect.right() - 14.0, rect.center().y), 4.0, ACCENT1());
    }
    if resp.clicked() || orb.clicked() {
        state.open = !state.open;
        state.focus = state.open;
    }
    if busy {
        ui.ctx().request_repaint();
    }
}

/// One message as the card shows it.
struct Shown {
    user: bool,
    text: String,
    image: Option<std::path::PathBuf>,
    did: Vec<String>,
    /// Role play: the goals the character set or met with this reply, and
    /// how it grew (traits, its persona) — noted under it, as in AI Chat.
    goals: Vec<crate::roleplay::GoalEvent>,
    growth: Vec<crate::roleplay::GrowthEvent>,
}

/// The newest messages of the assistant's conversation, as they read now:
/// a reply still coming without its thought and its control lines.
fn conversation(llm: &LlmState) -> Vec<Shown> {
    let Some(chat) = llm.assistant_chat() else { return Vec::new() };
    let streaming = llm.assistant_busy();
    let last = chat.msgs.len().saturating_sub(1);
    chat.msgs
        .iter()
        .enumerate()
        .skip(chat.msgs.len().saturating_sub(SHOWN))
        .map(|(i, m)| {
            let user = m.role == ChatRole::User;
            let text = if !user && streaming && i == last {
                let said = crate::llm::split_channels(&m.text).1;
                // (In character, the diary lines it ends with are the
                // app's too.)
                let said = if llm.gen_roleplay { crate::roleplay::strip_memory_lines(&said) } else { said };
                crate::assistant::strip(&said)
            } else {
                m.text.clone()
            };
            Shown { user, text, image: m.image.clone(), did: m.did.clone(), goals: m.goals.clone(), growth: m.growth.clone() }
        })
        .collect()
}

/// The card and the banner, hung under the pill. `here` is where the user
/// is, in a few words (the card's second line: what the assistant knows).
/// What the user asked of it, if anything.
pub fn show(
    ctx: &egui::Context,
    state: &mut AssistantUi,
    llm: &mut LlmState,
    settings: &mut crate::settings::Settings,
    here: &str,
    can_generate: bool,
) -> Option<Ask> {
    let pill = state.pill?;
    let busy = llm.assistant_busy();
    // A reply that has come in since the conversation was last looked at.
    if let Some(chat) = llm.assistant_chat()
        && !busy
        && state.seen != (chat.id, chat.msgs.len())
    {
        state.seen = (chat.id, chat.msgs.len());
        let reply = chat.msgs.last().filter(|m| m.role == ChatRole::Model && (!m.text.trim().is_empty() || !m.did.is_empty()));
        if reply.is_some() {
            state.banner = Some((chat.msgs.len() - 1, ctx.input(|i| i.time)));
            state.unread = true;
        }
    }
    if state.open {
        state.banner = None;
        state.unread = false;
    }

    let opened = ctx.animate_bool_with_time(egui::Id::new("assistant_card_open"), state.open, 0.16);
    let mut ask = None;
    if opened > 0.0 {
        ask = card(ctx, state, llm, settings, pill, here, can_generate, opened);
    } else if let Some((index, since)) = state.banner {
        let age = ctx.input(|i| i.time) - since;
        if age >= BANNER_SECS {
            state.banner = None;
        } else {
            let fade = ((BANNER_SECS - age) / BANNER_FADE).clamp(0.0, 1.0) as f32;
            if banner(ctx, llm, pill, index, fade) {
                state.open = true;
                state.focus = true;
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    ask
}

/// The frame both hang in: the app's card, rounder, with a soft shadow under
/// it — a notification's.
fn frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .stroke(egui::Stroke::new(1.0_f32, EDGE()))
        .corner_radius(22)
        .inner_margin(egui::Margin::same(14))
        .shadow(egui::epaint::Shadow { offset: [0, 10], blur: 28, spread: 0, color: Color32::from_black_alpha(110) })
}

/// Where something `width` wide hangs under the pill: centred on it, inside
/// the window.
fn under(ctx: &egui::Context, pill: egui::Rect, width: f32, drop: f32) -> egui::Pos2 {
    let screen = ctx.content_rect();
    let left = (pill.center().x - width / 2.0).min(screen.right() - width - 12.0).max(screen.left() + 12.0);
    egui::pos2(left, pill.bottom() + 12.0 + drop)
}

/// A reply as a banner under the pill: the orb's dot, what it said (its
/// first lines) and what it did. True when it's clicked (the card opens).
fn banner(ctx: &egui::Context, llm: &LlmState, pill: egui::Rect, index: usize, fade: f32) -> bool {
    let Some(m) = llm.assistant_chat().and_then(|c| c.msgs.get(index)) else { return false };
    let mut said: String = m.text.split_whitespace().collect::<Vec<_>>().join(" ");
    if said.chars().count() > 150 {
        said = format!("{}…", said.chars().take(150).collect::<String>().trim_end());
    }
    let mut clicked = false;
    egui::Area::new(egui::Id::new("assistant_banner"))
        .order(egui::Order::Foreground)
        .fixed_pos(under(ctx, pill, BANNER_W, 0.0))
        .show(ctx, |ui| {
            ui.set_opacity(fade);
            let resp = frame()
                .show(ui, |ui| {
                    ui.set_width(BANNER_W - 28.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(llm.assistant_name()).color(TEXT()).strong().size(12.5));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.label(egui::RichText::new("now").color(MUTED()).size(11.5));
                        });
                    });
                    if !said.is_empty() {
                        ui.add(egui::Label::new(egui::RichText::new(said).color(TEXT()).size(13.0)).wrap().selectable(false));
                    }
                    for did in &m.did {
                        did_line(ui, did);
                    }
                })
                .response;
            let resp = ui.interact(resp.rect, ui.id().with("open"), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
            clicked = resp.clicked();
        });
    clicked
}

/// What a reply did, as a small quiet line after a tick.
fn did_line(ui: &mut egui::Ui, did: &str) {
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add(
            egui::Image::new(egui::include_image!("../../icons/check_circle.svg"))
                .fit_to_exact_size(egui::vec2(13.0, 13.0))
                .tint(crate::tag_manager::AI_NEW_VIOLET),
        );
        ui.add(egui::Label::new(egui::RichText::new(did).color(MUTED()).size(11.5)).wrap());
    });
}

/// The chat card: who it is and where you are, the conversation, and the
/// box to write in.
#[allow(clippy::too_many_arguments)]
fn card(
    ctx: &egui::Context,
    state: &mut AssistantUi,
    llm: &mut LlmState,
    settings: &mut crate::settings::Settings,
    pill: egui::Rect,
    here: &str,
    can_generate: bool,
    opened: f32,
) -> Option<Ask> {
    let screen = ctx.content_rect();
    let width = CARD_W.min(screen.width() - 24.0);
    let mut ask = None;
    let msgs = conversation(llm);
    let busy = llm.assistant_busy();
    egui::Area::new(egui::Id::new("assistant_card"))
        .order(egui::Order::Foreground)
        .fixed_pos(under(ctx, pill, width, (opened - 1.0) * 10.0))
        .show(ctx, |ui| {
            ui.set_opacity(opened);
            frame().show(ui, |ui| {
                ui.set_width(width - 28.0);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 1.0;
                        // (Its character's name, in role play.)
                        ui.label(egui::RichText::new(llm.assistant_name()).color(TEXT()).strong().size(14.5));
                        ui.add(egui::Label::new(egui::RichText::new(here).color(MUTED()).size(11.5)).truncate());
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if crate::svg_button(ui, egui::include_image!("../../icons/close.svg"), "Close", 26.0, MUTED()).clicked() {
                            state.open = false;
                        }
                    });
                });
                ui.add_space(8.0);

                let most = (screen.bottom() - pill.bottom() - 190.0).clamp(120.0, 380.0);
                egui::ScrollArea::vertical().max_height(most).auto_shrink([false, true]).stick_to_bottom(true).show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;
                    if msgs.is_empty() {
                        ask = first_asks(ui, can_generate).or(ask.take());
                    }
                    let last = msgs.len().saturating_sub(1);
                    let coming = doing(llm);
                    for (i, m) in msgs.iter().enumerate() {
                        if m.user {
                            user_message(ui, llm, m);
                        } else {
                            reply(ui, &mut state.markdown, m, coming.as_ref().filter(|_| i == last));
                        }
                    }
                    // "Which model should I use?": the ones to pick from,
                    // under the question — a click answers it.
                    if llm.assistant.model_question && !busy && msgs.last().is_some_and(|m| !m.user) {
                        ask = model_asks(ui, &llm.assistant.scene.models).or(ask.take());
                    }
                    // Being shown the image: the reply that asked is gone,
                    // the one that answers hasn't begun.
                    if let Some(coming) = coming.as_ref().filter(|_| msgs.last().is_some_and(|m| m.user)) {
                        waiting(ui, coming);
                    }
                });
                if let Some(why) = &llm.run_err {
                    ui.add_space(4.0);
                    ui.add(egui::Label::new(egui::RichText::new(why).color(Color32::from_rgb(230, 90, 90)).size(11.5)).wrap());
                }
                ui.add_space(10.0);
                ask = composer(ui, state, llm, settings, busy).or(ask.take());
            });
        });
    if state.open && ctx.input(|i| i.key_pressed(egui::Key::Escape)) && ctx.memory(|m| m.focused().is_none()) {
        state.open = false;
    }
    ask
}

/// An empty conversation: what the assistant is for, and a few things to ask
/// with a click.
fn first_asks(ui: &mut egui::Ui, can_generate: bool) -> Option<Ask> {
    let mut ask = None;
    ui.add(
        egui::Label::new(
            egui::RichText::new(
                "I can see which view you're in and which image is selected — by name. I only look at an image \
                 when you ask me to.",
            )
            .color(MUTED())
            .size(12.5),
        )
        .wrap(),
    );
    ui.add_space(2.0);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        let asks: &[&str] = if can_generate { &[LOOK_FIRST, RECREATE_FIRST, OPEN_FIRST] } else { &[LOOK_FIRST] };
        for text in asks {
            if crate::scan::pill(ui, text, crate::scan::Pill::Outline, None, true).clicked() {
                // (The two about the image ask it to look.)
                ask = Some(if *text == OPEN_FIRST { Ask::Send(text.to_string()) } else { Ask::Show(text.to_string()) });
            }
        }
    });
    ask
}

/// The models the assistant asks the user to pick from, to answer with a
/// click (typing a name answers the same).
fn model_asks(ui: &mut egui::Ui, models: &[&'static str]) -> Option<Ask> {
    let mut ask = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        for model in models {
            if crate::scan::pill(ui, model, crate::scan::Pill::Outline, None, true).clicked() {
                ask = Some(Ask::Send(model.to_string()));
            }
        }
    });
    ask
}

/// The user's message: on the right, in the accent — under the image it was
/// sent with, once the assistant asked to look.
fn user_message(ui: &mut egui::Ui, llm: &mut LlmState, m: &Shown) {
    ui.with_layout(Layout::top_down(Align::Max), |ui| {
        if let Some(image) = &m.image {
            match llm.media_thumb(ui.ctx(), image) {
                Some(tex) => {
                    let size = tex.size_vec2();
                    let scale = (132.0 / size.x.max(size.y)).min(1.0);
                    ui.add(egui::Image::from_texture(&tex).fit_to_exact_size(size * scale).corner_radius(10))
                        .on_hover_text("Shown to the assistant with this message");
                }
                None => {
                    ui.label(egui::RichText::new(crate::ai_tagging::file_name(image)).color(MUTED()).size(11.5));
                }
            }
        }
        let room = ui.available_width() * 0.82;
        egui::Frame::new().fill(ACCENT1()).corner_radius(16).inner_margin(egui::Margin::symmetric(12, 8)).show(ui, |ui| {
            ui.set_max_width(room);
            ui.add(egui::Label::new(egui::RichText::new(&m.text).color(Color32::WHITE).size(13.5)).wrap());
        });
    });
}

/// The assistant's reply: its words, then what it did. While it's coming
/// and has said nothing yet, what it's busy with.
fn reply(ui: &mut egui::Ui, cache: &mut egui_commonmark::CommonMarkCache, m: &Shown, coming: Option<&Doing>) {
    if m.text.trim().is_empty() && m.did.is_empty() && m.goals.is_empty() && m.growth.is_empty() {
        match coming {
            Some(coming) => waiting(ui, coming),
            None => {
                ui.label(egui::RichText::new("No answer.").color(MUTED()).italics().size(12.5));
            }
        }
        return;
    }
    if !m.text.trim().is_empty() {
        ui.scope(|ui| {
            let style = ui.style_mut();
            if let Some(body) = style.text_styles.get_mut(&egui::TextStyle::Body) {
                body.size = 13.5;
            }
            style.visuals.override_text_color = Some(TEXT());
            egui_commonmark::CommonMarkViewer::new().keep_line_breaks(true).show(ui, cache, &m.text);
        });
    }
    for did in &m.did {
        did_line(ui, did);
    }
    crate::ai_chat::goal_note(ui, &m.goals);
    crate::ai_chat::growth_note(ui, &m.growth);
    // Its words are there, and more is coming that isn't for reading: the
    // prompt of the image it's about to make.
    if let Some(coming) = coming.filter(|c| c.what.starts_with("Writing the prompt")) {
        waiting(ui, coming);
    }
}

/// A spinner and what's being waited for — with the end of the thought
/// under it, while there is one.
fn waiting(ui: &mut egui::Ui, doing: &Doing) {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(13.0).color(MUTED()));
            ui.label(egui::RichText::new(&doing.what).color(MUTED()).size(12.5));
        });
        if let Some(thought) = &doing.thought {
            let ink = MUTED().gamma_multiply(0.7);
            ui.add(egui::Label::new(egui::RichText::new(thought).color(ink).italics().size(11.5)).truncate().selectable(false));
        }
    });
    ui.ctx().request_repaint();
}

/// The send box's height: a pill.
const BOX_H: f32 = 40.0;

/// The send box: one pill holding what AI Chat's input card has but its
/// Chats list (the assistant has the one conversation) — attach (+), Tools,
/// the text, the model, and send. Enter sends; while a reply is coming, or
/// being read aloud, the button stops it.
fn composer(
    ui: &mut egui::Ui,
    state: &mut AssistantUi,
    llm: &mut LlmState,
    settings: &mut crate::settings::Settings,
    busy: bool,
) -> Option<Ask> {
    let mut ask = None;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), BOX_H), Sense::hover());
    // The pill goes under what's in it: its slot first, filled in once the
    // text says whether it has the keyboard (as settings::pill_field does).
    let bg = ui.painter().add(egui::Shape::Noop);
    let inner = egui::Rect::from_min_max(rect.min + egui::vec2(6.0, 0.0), rect.max - egui::vec2(5.0, 0.0));
    let mut row = ui.new_child(egui::UiBuilder::new().max_rect(inner).layout(Layout::left_to_right(Align::Center)));
    row.spacing_mut().item_spacing.x = 2.0;

    // Attach: an image for the assistant to see with the message. Once
    // there is one, it shows here — a click takes it off again.
    match state.attached.clone() {
        Some(path) => {
            let (slot, resp) = row.allocate_exact_size(egui::vec2(28.0, 28.0), Sense::click());
            let resp = resp
                .on_hover_text(format!("{} — click to remove", crate::ai_tagging::file_name(&path)))
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Remove the attachment"));
            let at = egui::Rect::from_center_size(slot.center(), egui::vec2(26.0, 26.0));
            match llm.media_thumb(row.ctx(), &path) {
                Some(tex) => {
                    // Cut square from its middle, to fill the round slot.
                    let size = tex.size_vec2();
                    let (u, v) = ((1.0 - (size.y / size.x).min(1.0)) / 2.0, (1.0 - (size.x / size.y).min(1.0)) / 2.0);
                    egui::Image::from_texture(&tex)
                        .uv(egui::Rect::from_min_max(egui::pos2(u, v), egui::pos2(1.0 - u, 1.0 - v)))
                        .corner_radius(13)
                        .paint_at(&row, at);
                }
                None => {
                    row.painter().circle_filled(at.center(), 13.0, FIELD2());
                }
            }
            if resp.hovered() {
                row.painter().circle_filled(at.center(), 13.0, Color32::from_black_alpha(150));
                egui::Image::new(egui::include_image!("../../icons/close.svg"))
                    .tint(Color32::WHITE)
                    .paint_at(&row, egui::Rect::from_center_size(at.center(), egui::vec2(12.0, 12.0)));
            }
            if resp.clicked() {
                state.attached = None;
            }
        }
        None => {
            let add = egui::include_image!("../../icons/add.svg");
            if crate::ai_chat::icon_button(&mut row, add, 19.0, "Attach an image for the assistant to see", !busy).clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("Images", crate::ai_chat::DROP_EXTS).pick_file()
            {
                state.attached = Some(path);
                state.focus = true;
            }
        }
    }
    // Tools: role play, reading replies aloud, close-ups — the ones of AI
    // Chat's menu that are the assistant's too.
    let tools = crate::ai_chat::icon_button(&mut row, egui::include_image!("../../icons/tools.svg"), 17.0, "Tools", true);
    if tools.clicked() {
        llm.tools_open = !llm.tools_open;
    }
    crate::ai_chat::assistant_tools(&tools, rect, llm, settings);
    row.add_space(4.0);

    // From the right: send (or stop), the model, and the text in between.
    let speaking = !busy && llm.speaking_aloud();
    let mut has_focus = false;
    row.with_layout(Layout::right_to_left(Align::Center), |row| {
        row.spacing_mut().item_spacing.x = 6.0;
        let (icon, tip) = if busy {
            (egui::include_image!("../../icons/stop.svg"), "Stop")
        } else if speaking {
            (egui::include_image!("../../icons/stop.svg"), "Stop reading aloud")
        } else {
            (egui::include_image!("../../icons/send.svg"), "Send")
        };
        let ready = busy || speaking || !state.draft.trim().is_empty() || state.attached.is_some();
        let (at, resp) = row.allocate_exact_size(egui::vec2(30.0, 30.0), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ready, tip));
        let resp = resp.on_hover_text(tip);
        let fill = if !ready { FIELD2() } else if resp.hovered() { ACCENT1().gamma_multiply(0.85) } else { ACCENT1() };
        row.painter().circle_filled(at.center(), 15.0, fill);
        let tint = if ready { Color32::WHITE } else { MUTED() };
        egui::Image::new(icon).tint(tint).paint_at(row, egui::Rect::from_center_size(at.center(), egui::vec2(14.0, 14.0)));

        crate::ai_chat::assistant_model_selector(row, rect, llm, settings);

        let hint = if llm.roleplay.enabled { format!("Message {}", llm.assistant_name()) } else { "Ask the assistant".to_string() };
        let field = row
            .with_layout(Layout::left_to_right(Align::Center), |row| {
                row.add(
                    egui::TextEdit::singleline(&mut state.draft)
                        .hint_text(hint)
                        .frame(egui::Frame::NONE)
                        .margin(egui::Margin::ZERO)
                        .font(egui::FontId::proportional(13.0))
                        .text_color(TEXT())
                        .vertical_align(Align::Center)
                        .desired_width(row.available_width()),
                )
            })
            .inner;
        if std::mem::take(&mut state.focus) {
            field.request_focus();
        }
        has_focus = field.has_focus();
        let entered = field.lost_focus() && row.input(|i| i.key_pressed(egui::Key::Enter));
        let text = state.draft.trim();
        if busy {
            if resp.clicked() {
                ask = Some(Ask::Stop);
            }
        } else if (resp.clicked() || entered) && (!text.is_empty() || state.attached.is_some()) {
            let text = std::mem::take(&mut state.draft).trim().to_string();
            ask = Some(match state.attached.take() {
                Some(image) => Ask::With(text, image),
                None => Ask::Send(text),
            });
            field.request_focus();
        } else if resp.clicked() && speaking {
            ask = Some(Ask::Quiet);
        } else if entered {
            field.request_focus();
        }
    });
    let ring = if has_focus { crate::theme::selection_outline() } else { EDGE() };
    ui.painter().set(
        bg,
        egui::epaint::RectShape::new(
            rect,
            BOX_H / 2.0,
            crate::settings::well_fill(),
            egui::Stroke::new(1.0_f32, ring),
            egui::StrokeKind::Inside,
        ),
    );
    ask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Chat, ChatMsg};
    use crate::slider::rig::Rig;

    /// The pill, card and banner over an app, and what they asked of it.
    struct Over {
        ui: AssistantUi,
        llm: LlmState,
        settings: crate::settings::Settings,
        can_generate: bool,
        asks: Vec<Ask>,
        /// The app's fonts are in (the Tools menu sets names in its bold).
        fonts: bool,
    }

    fn scene(ui: &mut egui::Ui, s: &mut Over) {
        if !s.fonts {
            // (main.rs binds "bold" to its bold face; any face does here.)
            let mut fonts = egui::FontDefinitions::default();
            let plain = fonts.families[&egui::FontFamily::Proportional].clone();
            fonts.families.insert(egui::FontFamily::Name("bold".into()), plain);
            ui.ctx().set_fonts(fonts);
            s.fonts = true;
            return;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.set_height(56.0);
            ui.add_space(300.0);
            pill(ui, &mut s.ui, &s.llm);
        });
        s.asks.extend(show(ui.ctx(), &mut s.ui, &mut s.llm, &mut s.settings, "Gallery · fox.png", s.can_generate));
    }

    fn msg(role: ChatRole, text: &str, did: &[&str]) -> ChatMsg {
        ChatMsg {
            role,
            text: text.to_string(),
            thinking: None,
            image: None,
            doc: None,
            tag_image: None,
            tag_change: None,
            goals: vec![],
            growth: vec![],
            did: did.iter().map(|d| d.to_string()).collect(),
            character: None,
            remember: false,
            picture_memory: false,
            shared_with: vec![],
            research: None,
            krea_prompt: false,
        }
    }

    /// The app with the assistant's conversation as `msgs` (none: no
    /// conversation yet).
    fn over(msgs: Vec<ChatMsg>) -> Rig<Over> {
        let mut llm = LlmState::default();
        if !msgs.is_empty() {
            let mut chat = Chat::new(77);
            chat.msgs = msgs;
            llm.chats.push(chat);
            llm.assistant.chat = Some(77);
        }
        let over =
            Over { ui: AssistantUi::default(), llm, settings: Default::default(), can_generate: true, asks: vec![], fonts: false };
        let mut rig = Rig::new(over, scene);
        rig.screen = egui::vec2(1200.0, 800.0);
        rig.idle();
        rig
    }

    fn one(rig: &Rig<Over>, text: &str) -> egui::Rect {
        let found = rig.texts(text);
        assert_eq!(found.len(), 1, "{text:?} is on screen once");
        found[0]
    }

    /// Where the send box's buttons are, going by its hint (the box is one
    /// pill: attach and Tools before the text, send at its far end).
    struct SendBox {
        attach: egui::Pos2,
        tools: egui::Pos2,
        send: egui::Pos2,
    }

    fn send_box(rig: &Rig<Over>) -> SendBox {
        let hint = rig.texts("Ask the assistant").into_iter().chain(rig.texts("Message Mira")).next().expect("the send box's hint");
        let (left, y) = (hint.left() - 70.0, hint.center().y);
        SendBox {
            attach: egui::pos2(left + 20.0, y),
            tools: egui::pos2(left + 50.0, y),
            send: egui::pos2(left + (CARD_W - 28.0) - 20.0, y),
        }
    }

    /// Where the pill is: its line of text (the topmost "Assistant" — the
    /// card and the banner are titled so too).
    fn pill_at(rig: &Rig<Over>) -> egui::Pos2 {
        let named = rig.texts("Assistant");
        named.iter().map(|r| r.center()).min_by(|a, b| a.y.total_cmp(&b.y)).expect("the pill is in the top bar")
    }

    /// The pill opens the card under it — and closes it. With nothing said
    /// yet the card offers things to ask; a click on one asks it.
    #[test]
    fn the_pill_opens_the_card() {
        let mut rig = over(vec![]);
        let pill = pill_at(&rig);
        assert!(rig.texts("Ask the assistant").is_empty() && !rig.state.ui.open);
        rig.click(pill);
        rig.idle();
        assert!(rig.state.ui.open);
        // (The card's own title, and the pill's.)
        assert_eq!(rig.texts("Assistant").len(), 2);
        let hint = one(&rig, "Ask the assistant");
        assert!(hint.top() > pill.y + 20.0, "under the pill: {hint:?}");
        assert!((hint.center().x - pill.x).abs() < CARD_W / 2.0, "hung from it: {hint:?}");
        assert_eq!(rig.texts("Gallery · fox.png").len(), 1, "where the user is");
        for ask in [LOOK_FIRST, RECREATE_FIRST, OPEN_FIRST] {
            one(&rig, ask);
        }
        let recreate = one(&rig, RECREATE_FIRST).center();
        rig.click(recreate);
        // (About the image: it goes with the image shown.)
        assert_eq!(rig.state.asks, [Ask::Show(RECREATE_FIRST.to_string())]);
        let (look, open) = (one(&rig, LOOK_FIRST).center(), one(&rig, OPEN_FIRST).center());
        rig.click(look);
        rig.click(open);
        assert_eq!(rig.state.asks[1..], [Ask::Show(LOOK_FIRST.to_string()), Ask::Send(OPEN_FIRST.to_string())]);
        rig.state.asks.truncate(1);

        // Where nothing can be generated, nothing about it is offered.
        rig.state.can_generate = false;
        rig.idle();
        one(&rig, LOOK_FIRST);
        assert!(rig.texts(RECREATE_FIRST).is_empty() && rig.texts(OPEN_FIRST).is_empty());

        let pill = pill_at(&rig);
        rig.click(pill);
        rig.idle();
        assert!(!rig.state.ui.open && rig.texts("Ask the assistant").is_empty());
    }

    /// What's typed in the card goes on Enter, and the box is ready for the
    /// next; nothing typed sends nothing.
    #[test]
    fn enter_sends_what_is_typed() {
        let mut rig = over(vec![]);
        let pill = pill_at(&rig);
        rig.click(pill);
        rig.idle();
        rig.key(egui::Key::Enter);
        rig.idle();
        assert!(rig.state.asks.is_empty());
        // (The box has the keyboard from the moment the card opens.)
        rig.type_text("open the gallery");
        rig.key(egui::Key::Enter);
        rig.idle();
        assert_eq!(rig.state.asks, [Ask::Send("open the gallery".to_string())]);
        one(&rig, "Ask the assistant");
        rig.type_text("and back");
        rig.key(egui::Key::Enter);
        assert_eq!(rig.state.asks.len(), 2);
    }

    /// While a reply is coming and has nothing to read yet, the pill and
    /// the card say what it's doing: the worker's line until the first
    /// words, then that it's thinking — for how long, with the end of the
    /// thought under it — or writing an image's prompt.
    #[test]
    fn a_coming_reply_says_what_it_is_doing() {
        let mut rig = over(vec![msg(ChatRole::User, "Recreate it at night", &[]), msg(ChatRole::Model, "", &[])]);
        assert_eq!(doing(&rig.state.llm), None);
        rig.state.llm.running = true;
        rig.state.llm.gen_assistant = true;
        let coming = |rig: &mut Rig<Over>, raw: &str| {
            rig.state.llm.chats[1].msgs[1].text = raw.to_string();
            doing(&rig.state.llm).unwrap()
        };
        let plain = |what: &str| Doing { what: what.to_string(), thought: None };
        // Nothing has come: the worker's line (none yet: thinking — or,
        // shown the image, looking at it).
        assert_eq!(coming(&mut rig, ""), plain("Thinking…"));
        rig.state.llm.status = "Starting…".into();
        assert_eq!(coming(&mut rig, ""), plain("Thinking…"));
        rig.state.llm.assistant.looking = true;
        assert_eq!(coming(&mut rig, ""), plain("Looking at the image…"));
        rig.state.llm.status = "Reading the image…".into();
        assert_eq!(coming(&mut rig, ""), plain("Reading the image…"), "what the worker says, once it does");
        rig.state.llm.assistant.looking = false;
        rig.state.llm.status = "Loading the model (first time takes a minute)…".into();
        assert_eq!(coming(&mut rig, ""), plain("Loading the model (first time takes a minute)…"));
        rig.state.llm.gen_started = Some(std::time::Instant::now() - std::time::Duration::from_secs(40));
        assert_eq!(coming(&mut rig, "").what, "Loading the model (first time takes a minute)… 40 s");
        rig.state.llm.gen_started = None;

        // A thought: its last line — the end of it, when it's long.
        let thinking = coming(&mut rig, "<|channel>thought\nThe user wants night.\nA streetlamp outside, then.\n\n");
        assert_eq!(thinking, Doing { what: "Thinking…".into(), thought: Some("A streetlamp outside, then.".into()) });
        let long = format!("<think>{}the very end", "word ".repeat(40));
        let tail = coming(&mut rig, &long).thought.unwrap();
        assert!(tail.starts_with('…') && tail.ends_with("word the very end") && tail.chars().count() <= THOUGHT_CHARS + 1, "{tail}");
        // Timed from its first word, once that's worth saying.
        rig.state.llm.gen_started = Some(std::time::Instant::now() - std::time::Duration::from_secs(12));
        assert_eq!(coming(&mut rig, "<|channel>thought\nHm.").what, "Thinking… 12 s");
        rig.state.llm.gen_started = Some(std::time::Instant::now());
        assert_eq!(coming(&mut rig, "<|channel>thought\nHm.").what, "Thinking…");

        // Words, and the prompt after them (however it's dressed).
        assert_eq!(coming(&mut rig, "<|channel>thought\nHm.<channel|>On it."), plain("Writing…"));
        assert_eq!(coming(&mut rig, "<|channel>thought\nHm.<channel|>On it.\n`GENERATE: a cat"), plain("Writing the prompt…"));
        assert_eq!(coming(&mut rig, "On it. Not GENERATE: yet"), plain("Writing…"));

        // In the card: the line, and the thought under it; neither once
        // the reply has ended.
        rig.state.llm.gen_started = None;
        rig.state.llm.chats[1].msgs[1].text = "<|channel>thought\nA streetlamp outside, then.".into();
        rig.idle();
        let pill = one(&rig, "Thinking…").center();
        rig.click(pill);
        rig.idle();
        let lines = rig.texts("Thinking…");
        assert_eq!(lines.len(), 2, "in the pill and in the card");
        let thought = one(&rig, "A streetlamp outside, then.");
        assert!(lines.iter().any(|line| thought.top() > line.bottom() - 2.0 && thought.top() < line.bottom() + 8.0), "{lines:?} {thought:?}");
        // Words to read need no line under them: the pill's is enough.
        rig.state.llm.chats[1].msgs[1].text = "<|channel>thought\nHm.<channel|>On it.".into();
        rig.idle();
        one(&rig, "On it.");
        assert_eq!(rig.texts("Writing…").len(), 1);
        rig.state.llm.running = false;
        rig.state.llm.chats[1].msgs[1].text = "Done.".into();
        rig.idle();
        assert!(rig.texts("Thinking…").is_empty() && rig.texts("A streetlamp outside, then.").is_empty());
    }

    /// The send box is one pill with what AI Chat's input has, but for its
    /// Chats: Tools opens the assistant's menu — role play, reading aloud,
    /// close-ups, and none of the chat's own modes — under the box.
    #[test]
    fn the_send_box_has_the_assistants_tools() {
        let mut rig = over(vec![]);
        // (Room under the card, as there is under the top bar: with none,
        // the menu goes over it instead.)
        rig.screen = egui::vec2(1200.0, 1500.0);
        rig.idle();
        let pill = pill_at(&rig);
        rig.click(pill);
        rig.idle();
        assert!(rig.texts("Role playing").is_empty() && !rig.state.llm.tools_open);
        let tools = send_box(&rig).tools;
        rig.click(tools);
        rig.idle();
        assert!(rig.state.llm.tools_open);
        let hint = one(&rig, "Ask the assistant");
        // (Each is named on its card, and the one the panel is about there too.)
        for tool in ["Role playing", "Auto-speak replies", "Detailed vision"] {
            let named = rig.texts(tool);
            assert!(!named.is_empty() && named.iter().all(|r| r.top() > hint.bottom()), "{tool}: under the box, {named:?} {hint:?}");
        }
        for tool in ["Krea 2 prompts", "Tag images", "Web research"] {
            assert!(rig.texts(tool).is_empty(), "{tool} is the chat's");
        }
        one(&rig, "Answers as your character");

        // Its switches are the app's own: Detailed vision's, here.
        assert!(!rig.state.settings.ai_detailed_vision);
        let vision = rig.texts("Detailed vision").into_iter().min_by(|a, b| a.top().total_cmp(&b.top())).unwrap();
        rig.click(egui::pos2(vision.left() + 142.0, vision.center().y));
        assert!(rig.state.settings.ai_detailed_vision && rig.state.llm.detailed_vision);
        assert!(rig.state.llm.tools_open, "a switch doesn't close it");
        // A click outside does.
        rig.click(egui::pos2(40.0, 700.0));
        rig.idle();
        assert!(!rig.state.llm.tools_open && rig.texts("Role playing").is_empty());
    }

    /// With Role playing on the card is the character's: its name on the
    /// pill and the card, and in the send box's hint. The conversation with
    /// the assistant isn't theirs: the card starts empty.
    #[test]
    fn in_role_play_the_card_is_the_characters() {
        let mut rig = over(vec![msg(ChatRole::User, "hello", &[]), msg(ChatRole::Model, "Hi!", &[])]);
        let pill = pill_at(&rig);
        rig.click(pill);
        rig.idle();
        one(&rig, "Hi!");
        assert_eq!(rig.texts("Assistant").len(), 2);
        rig.state.llm.roleplay.ai_name = "Mira".into();
        rig.state.llm.roleplay.enabled = true;
        rig.idle();
        assert!(rig.texts("Assistant").is_empty() && rig.texts("Mira").len() == 2, "on the pill and the card");
        one(&rig, "Message Mira");
        assert!(rig.texts("Hi!").is_empty() && rig.texts("hello").is_empty());
        one(&rig, LOOK_FIRST);
        rig.type_text("there you are");
        rig.key(egui::Key::Enter);
        assert_eq!(rig.state.asks, [Ask::Send("there you are".to_string())]);

        // Back with the assistant, its conversation is there again.
        rig.state.llm.roleplay.enabled = false;
        rig.idle();
        one(&rig, "Hi!");
        one(&rig, "Ask the assistant");

        // A reply the character is still writing (asked while Role playing
        // was on) shows without the diary line it ends with, and without
        // the app's.
        rig.state.llm.chats[1].msgs.push(msg(ChatRole::User, "come and see", &[]));
        rig.state.llm.chats[1].msgs.push(msg(ChatRole::Model, "On my way!\nMEMORY: Andrew likes foxes\nOPEN: gall", &[]));
        rig.state.llm.running = true;
        rig.state.llm.gen_assistant = true;
        rig.state.llm.gen_roleplay = true;
        rig.idle();
        one(&rig, "On my way!");
        assert!(rig.texts("MEMORY: Andrew likes foxes").is_empty() && rig.texts("OPEN: gall").is_empty());
    }

    /// An image attached with the + goes with the next message — with no
    /// words, too — and a click on it in the box takes it off again.
    #[test]
    fn an_attached_image_goes_with_what_is_sent() {
        let mut rig = over(vec![]);
        let pill = pill_at(&rig);
        rig.click(pill);
        rig.idle();
        let image = std::path::PathBuf::from("/pics/cat.png");
        rig.state.ui.attached = Some(image.clone());
        rig.idle();
        rig.type_text("what is it?");
        rig.key(egui::Key::Enter);
        assert_eq!(rig.state.asks, [Ask::With("what is it?".to_string(), image.clone())]);
        assert!(rig.state.ui.attached.is_none(), "sent with it");

        // Alone: the send button, with nothing written.
        rig.state.ui.attached = Some(image.clone());
        rig.idle();
        let send = send_box(&rig).send;
        rig.click(send);
        assert_eq!(rig.state.asks[1], Ask::With(String::new(), image.clone()));
        rig.click(send);
        assert_eq!(rig.state.asks.len(), 2, "nothing left to send");

        // Taken off again: where the + was.
        rig.state.ui.attached = Some(image);
        rig.idle();
        let attach = send_box(&rig).attach;
        rig.click(attach);
        rig.idle();
        assert!(rig.state.ui.attached.is_none());
        rig.click(send);
        assert_eq!(rig.state.asks.len(), 2);
    }

    /// "Which model should I use?": the models to pick from are under the
    /// question, and a click answers with that model's name. Not while a
    /// reply is coming, and not once it's answered.
    #[test]
    fn the_models_to_pick_from_are_under_the_question() {
        let mut rig = over(vec![
            msg(ChatRole::User, "make me a fox", &[]),
            msg(ChatRole::Model, "A fox in fresh snow.\n\nWhich model should I use?", &[]),
        ]);
        rig.state.llm.assistant.scene.models = vec!["SDXL", "Krea 2", "Z-Image"];
        rig.state.llm.assistant.model_question = true;
        let pill = pill_at(&rig);
        rig.click(pill);
        rig.idle();
        let question = one(&rig, "Which model should I use?");
        let (sdxl, krea, z) = (one(&rig, "SDXL"), one(&rig, "Krea 2"), one(&rig, "Z-Image"));
        assert!(sdxl.top() > question.bottom() && sdxl.right() < krea.left() && krea.right() < z.left());
        rig.click(krea.center());
        assert_eq!(rig.state.asks, [Ask::Send("Krea 2".to_string())]);

        rig.state.llm.running = true;
        rig.state.llm.gen_assistant = true;
        rig.idle();
        assert!(rig.texts("SDXL").is_empty(), "a reply is coming");
        rig.state.llm.running = false;
        rig.state.llm.assistant.model_question = false;
        rig.idle();
        assert!(rig.texts("SDXL").is_empty() && rig.texts("Z-Image").is_empty(), "answered");
        one(&rig, "Which model should I use?");
    }

    /// The conversation: the user's messages on the right, the replies on
    /// the left with what they did under them. A reply still coming shows
    /// without its thought or its control lines, the pill says what it's
    /// busy with, and the button stops it.
    #[test]
    fn the_card_shows_the_conversation() {
        let mut rig = over(vec![
            msg(ChatRole::User, "What do you think of it?", &[]),
            msg(ChatRole::Model, "Calm and warm.", &["Liked it — the warm light"]),
            msg(ChatRole::User, "Recreate it at night", &[]),
            msg(ChatRole::Model, "<|channel>thought\nNight, a lamp.<channel|>On it.\nGENERATE: a cat at ni", &[]),
        ]);
        rig.state.llm.running = true;
        rig.state.llm.gen_assistant = true;
        rig.state.llm.status = "Thinking…".into();
        rig.idle();
        // (Its words are there, and the image's prompt is coming.)
        let pill = one(&rig, "Writing the prompt…").center();
        rig.click(pill);
        rig.idle();
        let (asked, said, did) = (one(&rig, "What do you think of it?"), one(&rig, "Calm and warm."), one(&rig, "Liked it — the warm light"));
        assert!(asked.left() > said.left() + 60.0, "the user's on the right: {asked:?} {said:?}");
        assert!(said.top() > asked.bottom() && did.top() > said.bottom() - 2.0);
        let coming = one(&rig, "On it.");
        assert!(coming.top() > one(&rig, "Recreate it at night").bottom());
        assert!(rig.texts("Night, a lamp.").is_empty(), "its thought stays its own");
        assert!(rig.texts("GENERATE: a cat at ni").is_empty(), "and its control lines the app's");
        let writing = rig.texts("Writing the prompt…");
        assert_eq!(writing.len(), 2, "in the pill, and under what it said");
        assert!(writing.iter().any(|r| r.top() > coming.bottom() - 2.0));
        // Stop, where Send was.
        let button = send_box(&rig).send;
        rig.click(button);
        assert_eq!(rig.state.asks, [Ask::Stop]);

        // Ended: what it wrote as it stands, the pill at rest.
        rig.state.llm.running = false;
        rig.state.llm.chats[1].msgs[3] = msg(ChatRole::Model, "On it.", &["Started generating in Text to Image"]);
        rig.idle();
        one(&rig, "Started generating in Text to Image");
        assert!(rig.texts("Writing…").is_empty() && rig.texts("Assistant").len() == 2);
        assert!(rig.state.ui.banner.is_none() && !rig.state.ui.unread, "it was read as it came");
    }

    /// With the card closed a reply shows as a banner under the pill — what
    /// it said and what it did — for a few seconds; a click on it opens the
    /// card. What was already there when the app started is no news.
    #[test]
    fn a_reply_shows_as_a_banner_while_the_card_is_closed() {
        let mut rig = over(vec![msg(ChatRole::User, "Recreate it at night", &[])]);
        assert!(rig.state.ui.banner.is_none(), "a message of the user's is no news");
        rig.state.llm.chats[1].msgs.push(msg(ChatRole::Model, "On it — at night.", &["Opened Text to Image"]));
        // (Something new on screen is measured in its first frame, and
        // painted from the second.)
        rig.frame(vec![]);
        rig.frame(vec![]);
        let said = one(&rig, "On it — at night.");
        one(&rig, "Opened Text to Image");
        assert!(rig.state.ui.unread && !rig.state.ui.open);
        assert!(rig.texts("Ask the assistant").is_empty(), "a banner, not the card");
        assert!(said.top() > pill_at(&rig).y + 20.0);

        // It goes by itself.
        for _ in 0..(BANNER_SECS as usize + 2) {
            rig.idle();
        }
        assert!(rig.texts("On it — at night.").is_empty() && rig.state.ui.banner.is_none());
        assert!(rig.state.ui.unread, "the pill still says there's something new");

        // The next one, clicked: the card, with all of it.
        rig.state.llm.chats[1].msgs.push(msg(ChatRole::User, "thanks", &[]));
        rig.state.llm.chats[1].msgs.push(msg(ChatRole::Model, "Any time.", &[]));
        rig.frame(vec![]);
        rig.frame(vec![]);
        let banner = one(&rig, "Any time.").center();
        rig.click(banner);
        rig.idle();
        assert!(rig.state.ui.open && !rig.state.ui.unread && rig.state.ui.banner.is_none());
        one(&rig, "Ask the assistant");
        one(&rig, "On it — at night.");
    }
}
