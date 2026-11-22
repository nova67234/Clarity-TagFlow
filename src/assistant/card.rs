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
const CARD_W: f32 = 400.0;
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
pub struct AssistantUi {
    /// The card is open.
    pub open: bool,
    draft: String,
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

impl Default for AssistantUi {
    fn default() -> Self {
        AssistantUi {
            open: false,
            draft: String::new(),
            orb: AiOrb::new(260),
            pill: None,
            seen: (0, 0),
            banner: None,
            unread: false,
            focus: false,
            markdown: Default::default(),
        }
    }
}

/// What the card asks of the app.
#[derive(Debug, PartialEq)]
pub enum Ask {
    /// Send this to the assistant.
    Send(String),
    /// Stop the reply that's coming.
    Stop,
}

/// What the pill says: what the assistant is doing, or just that it's there.
fn pill_text(llm: &LlmState) -> String {
    if !llm.assistant_busy() {
        "Assistant".to_string()
    } else if llm.assistant.looking {
        "Looking at the image…".to_string()
    } else if llm.status.trim().is_empty() {
        "Thinking…".to_string()
    } else {
        llm.status.clone()
    }
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
                crate::assistant::strip(&crate::llm::split_channels(&m.text).1)
            } else {
                m.text.clone()
            };
            Shown { user, text, image: m.image.clone(), did: m.did.clone() }
        })
        .collect()
}

/// The card and the banner, hung under the pill. `here` is where the user
/// is, in a few words (the card's second line: what the assistant knows).
/// What the user asked of it, if anything.
pub fn show(ctx: &egui::Context, state: &mut AssistantUi, llm: &mut LlmState, here: &str, can_generate: bool) -> Option<Ask> {
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
        ask = card(ctx, state, llm, pill, here, can_generate, opened);
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
                        ui.label(egui::RichText::new("Assistant").color(TEXT()).strong().size(12.5));
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
fn card(
    ctx: &egui::Context,
    state: &mut AssistantUi,
    llm: &mut LlmState,
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
                        ui.label(egui::RichText::new("Assistant").color(TEXT()).strong().size(14.5));
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
                    for (i, m) in msgs.iter().enumerate() {
                        if m.user {
                            user_message(ui, llm, m);
                        } else {
                            reply(ui, &mut state.markdown, m, busy && i == last, &llm.status, llm.assistant.looking);
                        }
                    }
                    // Being shown the image: the reply that asked is gone,
                    // the one that answers hasn't begun.
                    if busy && msgs.last().is_some_and(|m| m.user) {
                        waiting(ui, if llm.assistant.looking { "Looking at the image…" } else { &llm.status });
                    }
                });
                if let Some(why) = &llm.run_err {
                    ui.add_space(4.0);
                    ui.add(egui::Label::new(egui::RichText::new(why).color(Color32::from_rgb(230, 90, 90)).size(11.5)).wrap());
                }
                ui.add_space(10.0);
                ask = composer(ui, state, busy).or(ask.take());
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
                ask = Some(Ask::Send(text.to_string()));
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
fn reply(ui: &mut egui::Ui, cache: &mut egui_commonmark::CommonMarkCache, m: &Shown, coming: bool, status: &str, looking: bool) {
    if m.text.trim().is_empty() && m.did.is_empty() {
        if coming {
            waiting(ui, if looking { "Looking at the image…" } else { status });
        } else {
            ui.label(egui::RichText::new("No answer.").color(MUTED()).italics().size(12.5));
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
}

/// A spinner and what's being waited for.
fn waiting(ui: &mut egui::Ui, what: &str) {
    ui.horizontal(|ui| {
        ui.add(egui::Spinner::new().size(13.0).color(MUTED()));
        let what = if what.trim().is_empty() { "Thinking…" } else { what };
        ui.label(egui::RichText::new(what).color(MUTED()).size(12.5));
    });
    ui.ctx().request_repaint();
}

/// The box to write in, and the button after it: send — on Enter too — or,
/// while a reply is coming, stop.
fn composer(ui: &mut egui::Ui, state: &mut AssistantUi, busy: bool) -> Option<Ask> {
    let mut ask = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        let width = ui.available_width() - 40.0;
        let field = ui
            .allocate_ui_with_layout(egui::vec2(width, 32.0), Layout::left_to_right(Align::Center), |ui| {
                crate::settings::pill_field(ui, width, egui::TextEdit::singleline(&mut state.draft).hint_text("Ask the assistant"))
            })
            .inner;
        if std::mem::take(&mut state.focus) {
            field.request_focus();
        }
        let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let (icon, tip) = if busy {
            (egui::include_image!("../../icons/stop.svg"), "Stop")
        } else {
            (egui::include_image!("../../icons/send.svg"), "Send")
        };
        let ready = busy || !state.draft.trim().is_empty();
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ready, tip));
        let resp = resp.on_hover_text(tip);
        let fill = if !ready { FIELD2() } else if resp.hovered() { ACCENT1().gamma_multiply(0.85) } else { ACCENT1() };
        ui.painter().circle_filled(rect.center(), 16.0, fill);
        let tint = if ready { Color32::WHITE } else { MUTED() };
        egui::Image::new(icon).tint(tint).paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(15.0, 15.0)));
        if busy {
            if resp.clicked() {
                ask = Some(Ask::Stop);
            }
        } else if (resp.clicked() || entered) && !state.draft.trim().is_empty() {
            ask = Some(Ask::Send(std::mem::take(&mut state.draft)));
            field.request_focus();
        } else if entered {
            field.request_focus();
        }
    });
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
        can_generate: bool,
        asks: Vec<Ask>,
    }

    fn scene(ui: &mut egui::Ui, s: &mut Over) {
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.set_height(56.0);
            ui.add_space(300.0);
            pill(ui, &mut s.ui, &s.llm);
        });
        s.asks.extend(show(ui.ctx(), &mut s.ui, &mut s.llm, "Gallery · fox.png", s.can_generate));
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
        let mut rig = Rig::new(Over { ui: AssistantUi::default(), llm, can_generate: true, asks: vec![] }, scene);
        rig.screen = egui::vec2(1200.0, 800.0);
        rig.idle();
        rig
    }

    fn one(rig: &Rig<Over>, text: &str) -> egui::Rect {
        let found = rig.texts(text);
        assert_eq!(found.len(), 1, "{text:?} is on screen once");
        found[0]
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
        assert_eq!(rig.state.asks, [Ask::Send(RECREATE_FIRST.to_string())]);

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
        rig.state.llm.status = "Writing…".into();
        rig.idle();
        assert_eq!(one(&rig, "Writing…").center().y, rig.texts("Writing…")[0].center().y);
        let pill = one(&rig, "Writing…").center();
        rig.click(pill);
        rig.idle();
        let (asked, said, did) = (one(&rig, "What do you think of it?"), one(&rig, "Calm and warm."), one(&rig, "Liked it — the warm light"));
        assert!(asked.left() > said.left() + 60.0, "the user's on the right: {asked:?} {said:?}");
        assert!(said.top() > asked.bottom() && did.top() > said.bottom() - 2.0);
        let coming = one(&rig, "On it.");
        assert!(coming.top() > one(&rig, "Recreate it at night").bottom());
        assert!(rig.texts("Night, a lamp.").is_empty(), "its thought stays its own");
        assert!(rig.texts("GENERATE: a cat at ni").is_empty(), "and its control lines the app's");
        // Stop, where Send was.
        let hint = one(&rig, "Ask the assistant");
        let button = egui::pos2(hint.left() - 14.0 + (CARD_W - 28.0 - 40.0) + 8.0 + 16.0, hint.center().y);
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
