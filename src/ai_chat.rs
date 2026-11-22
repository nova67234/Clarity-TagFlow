//! AI Chat — the full-window view shown instead of the three panels while
//! Settings → AI Model → "Activate AI Chat" is on (the top bar stays).
//!
//! Layout: just the conversation, centred, with the input card at the bottom
//! — no side panels. A workflow from the input card's Tools menu brings its
//! own: Tag images puts the image on the left and the Tag Manager on the
//! right (src/main.rs). The session's chats are behind the input card's
//! chats button, and AI Settings behind the Tools menu's gear.
//!
//! All chat state (conversations, streaming reply, draft) lives on
//! `LlmState` (src/llm.rs); this module only draws it. Web research's parts
//! (the sources under a reply, the notes viewer) are in
//! src/ai_chat/web_research.rs.

mod markdown;
mod tools_menu;
mod web_research;

use eframe::egui;
use egui::Margin;

use crate::llm::{ChatMsg, ChatRole, LlmState};
use crate::theme::*;

pub use web_research::NotesView;

/// Drag-and-drop accent — the blue the input card turns while a file is
/// dragged over it (matches the generator prompt box's import blue).
const DROP_BLUE: egui::Color32 = egui::Color32::from_rgb(56, 132, 255);

/// Image types the input card accepts from a drop (matches the + picker).
/// Videos are accepted too, via `crate::is_video` — the model worker samples
/// them into a handful of timestamped frames.
const DROP_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"];

/// Video types offered by the + picker (matches `main::VIDEO_EXTENSIONS`).
const VIDEO_PICK_EXTS: &[&str] = &["mp4", "mov", "mkv", "webm", "avi", "m4v", "wmv", "flv"];

/// True when the AI Chat's input card (visible this or last frame) claims a
/// file dropped at the pointer, so the gallery doesn't also add it — the same
/// freshness-checked-rect scheme as `generate::generator_claims_drop`. A drop
/// with no known pointer position (common on Windows mid-drag) is claimed
/// whenever the card is live, since it was showing the highlight.
pub fn claims_drop(llm: &crate::llm::LlmState, ctx: &egui::Context) -> bool {
    // Tag images sends the image on the left instead; a drop goes to the
    // browser (opening a folder to tag, say).
    if llm.tag_images && llm.tag_target.is_some() {
        return false;
    }
    let Some((rect, t)) = llm.input_rect else {
        return false;
    };
    if (ctx.input(|i| i.time) - t).abs() >= 0.5 {
        return false;
    }
    let pos = ctx.input(|i| i.pointer.interact_pos().or(i.pointer.latest_pos()));
    pos.is_none_or(|p| rect.contains(p))
}

pub fn show(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut crate::settings::Settings) {
    llm.poll(ui.ctx());
    // Where this view starts: under the top bar (a panel inside `ui`).
    let view_top = ui.available_rect_before_wrap().top();

    // Not usable yet — say why, centred, instead of an empty chat. With a
    // way out: this view replaces the whole browser, and Settings only shows
    // the "Activate AI Chat" switch for an installed model (picking one that
    // isn't, or a download of many GB, would otherwise strand the user here).
    // A Link host answering for this PC makes it usable too.
    if !llm.usable() {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.4);
            let msg = if !crate::llm::BUILT_WITH_LLM {
                "This build was compiled without the AI feature.".to_string()
            } else if let Some(dl) = &llm.download {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(150));
                format!("Downloading {}… {}%", llm.model.label(), dl.pct())
            } else {
                format!(
                    "{} isn't downloaded yet — download it in Settings → AI Model, \
                     or pick a model you have there.",
                    llm.model.label()
                )
            };
            ui.label(egui::RichText::new(msg).color(MUTED()).size(14.0));
            ui.add_space(10.0);
            if crate::llm::BUILT_WITH_LLM && ui.button("Open AI Model settings").clicked() {
                settings.open = true;
                settings.tab = crate::settings::SettingsTab::AiModel;
            }
            if ui.button("Turn off AI Chat").on_hover_text("Back to the browser").clicked() {
                settings.ai_chat = false;
                llm.chat_off(settings.ai_assistant);
            }
        });
        return;
    }

    // No side panels of its own: with no workflow on in the Tools menu it's
    // only the chat. Tag images brings the image and the Tag Manager
    // (main.rs draws those around this view); role play's character and
    // memories live in AI Settings; the chats are behind the chats button.

    // --- The conversation + input pill, in a centred column. ---
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(BG()).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            conversation(ui, llm, settings);
        });

    // The gear's AI Settings window and the memory map it opens (kept under
    // the top bar, where this view starts).
    crate::ai_settings::window(ui.ctx(), llm, settings);
    crate::memory_map::window(ui.ctx(), llm, view_top);
    // A web lookup's notes, opened from under its reply.
    web_research::notes_window(ui.ctx(), llm);

    // Click-to-enlarge overlay for message attachments, above everything.
    lightbox_ui(ui.ctx(), llm);
}

/// A clicked chat attachment, enlarged over the whole app: images decode
/// hi-res on a worker thread; videos actually play (audio on, looping per
/// the user's Settings) through the embedded VLC player. Esc, the ✕, or a
/// click on the backdrop closes it — closing drops the player, which stops
/// playback.
pub struct Lightbox {
    path: std::path::PathBuf,
    /// Hi-res still arriving from the decode thread (images only).
    rx: Option<std::sync::mpsc::Receiver<Option<egui::ColorImage>>>,
    tex: Option<egui::TextureHandle>,
    player: Option<crate::video::VideoPlayer>,
}

impl Lightbox {
    fn open(path: &std::path::Path, ctx: &egui::Context) -> Self {
        let mut lb = Lightbox { path: path.to_path_buf(), rx: None, tex: None, player: None };
        if crate::is_video(path) {
            lb.player = crate::video::VideoPlayer::start(path, ctx);
        }
        if lb.player.is_none() {
            // Off-thread hi-res decode; until it lands (or if it fails — e.g.
            // a video in a VLC-less build) the cached message thumbnail shows.
            let (tx, rx) = std::sync::mpsc::channel();
            lb.rx = Some(rx);
            let p = path.to_path_buf();
            let repaint = ctx.clone();
            std::thread::spawn(move || {
                let img = crate::image_cache::open_oriented(&p).ok().map(|img| {
                    let t = img.thumbnail(2048, 2048).to_rgba8();
                    egui::ColorImage::from_rgba_unmultiplied(
                        [t.width() as usize, t.height() as usize],
                        t.as_raw(),
                    )
                });
                let _ = tx.send(img);
                repaint.request_repaint();
            });
        }
        lb
    }
}

fn lightbox_ui(ctx: &egui::Context, llm: &mut LlmState) {
    let Some(mut lb) = llm.lightbox.take() else { return };
    if let Some(rx) = &lb.rx
        && let Ok(res) = rx.try_recv() {
        lb.rx = None;
        lb.tex =
            res.map(|img| ctx.load_texture("ai_chat_lightbox", img, egui::TextureOptions::LINEAR));
    }

    let screen = ctx.content_rect();
    let mut close = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    egui::Area::new(egui::Id::new("ai_chat_lightbox"))
        .order(egui::Order::Foreground)
        .fixed_pos(screen.min)
        .show(ctx, |ui| {
            // Dimmed backdrop over the whole app; clicking it closes.
            let bg = ui.interact(screen, egui::Id::new("ai_lightbox_bg"), egui::Sense::click());
            ui.painter().rect_filled(screen, 0.0, egui::Color32::from_black_alpha(180));

            // The media: the playing video's frame, the hi-res still, or —
            // while the still decodes — the message thumbnail as a stand-in.
            let tex = if let Some(p) = &mut lb.player { p.frame(ctx) } else { lb.tex.clone() }
                .or_else(|| llm.msg_thumbs.get(&lb.path).cloned().flatten());
            match tex {
                Some(tex) => {
                    let size = tex.size_vec2();
                    let max = screen.size() * 0.88;
                    let scale = (max.x / size.x).min(max.y / size.y).min(6.0);
                    let rect = egui::Rect::from_center_size(screen.center(), size * scale);
                    ui.put(
                        rect,
                        egui::Image::new(&tex)
                            .fit_to_exact_size(size * scale)
                            .corner_radius(egui::CornerRadius::same(10)),
                    );
                    // Swallow clicks on the media so they don't hit the
                    // backdrop's click-to-close.
                    let _ = ui.interact(rect, egui::Id::new("ai_lightbox_media"), egui::Sense::click());
                }
                None => {
                    ui.put(
                        egui::Rect::from_center_size(screen.center(), egui::vec2(40.0, 40.0)),
                        egui::Spinner::new().size(32.0),
                    );
                }
            }

            // ✕ pinned top-right, inking up on hover.
            let crect = egui::Rect::from_center_size(
                egui::pos2(screen.right() - 26.0, screen.top() + 26.0),
                egui::vec2(28.0, 28.0),
            );
            let cresp = ui
                .interact(crect, egui::Id::new("ai_lightbox_close"), egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            let ink = if cresp.hovered() { egui::Color32::WHITE } else { egui::Color32::from_gray(190) };
            egui::Image::new(egui::include_image!("../icons/x.svg")).tint(ink).paint_at(
                ui,
                egui::Rect::from_center_size(crect.center(), egui::vec2(18.0, 18.0)),
            );

            if bg.clicked() || cresp.clicked() {
                close = true;
            }
        });

    if !close {
        llm.lightbox = Some(lb);
    }
}

/// The chats popup (the input card's chats button): a "CHATS" header with a
/// new-chat button, then one row per conversation this session — click one
/// to switch to it, ✕ to delete it.
fn chat_list(ui: &mut egui::Ui, llm: &mut LlmState) {
    // The popup's width: the rows' left-aligned titles would otherwise grow
    // it to the whole window.
    ui.set_max_width(260.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("CHATS").color(MUTED()).strong().size(11.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if icon_button(ui, egui::include_image!("../icons/add.svg"), 18.0, "New chat", true)
                .clicked()
            {
                llm.new_chat();
                llm.chats_open = false;
            }
        });
    });
    ui.add_space(5.0);

    egui::ScrollArea::vertical().id_salt("ai_chat_list").max_height(340.0).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        // Title buttons all the same width, the delete icons in a column.
        let title_w = ui.available_width() - 30.0;
        let mut delete: Option<usize> = None;
        for i in 0..llm.chats.len() {
            let selected = i == llm.active_chat;
            let title = llm.chats[i].title();
            ui.horizontal(|ui| {
                // The title on the left (the empty right text takes the rest),
                // cut short at the button's edge: a chat that starts with a
                // PDF is named after the file, and a long name stretched the
                // popup (its delete icon with it).
                let mut btn = egui::Button::new(egui::RichText::new(title.as_str()).color(TEXT()).size(13.0))
                    .right_text("")
                    .truncate()
                    .corner_radius(egui::CornerRadius::same(18));
                btn = if selected {
                    btn.fill(ACCENT1().gamma_multiply(0.15))
                } else {
                    btn.fill(egui::Color32::TRANSPARENT)
                };
                let mut row = ui.add_sized(egui::vec2(title_w, 30.0), btn);
                // Cut short (it wanted more room than it has): the whole
                // title on hover.
                if row.intrinsic_size().is_some_and(|full| full.x > row.rect.width() + 0.5) {
                    row = row.on_hover_text(title.as_str());
                }
                if row.clicked() {
                    llm.select_chat(i);
                    llm.chats_open = false;
                }
                if icon_button_tinted(ui, egui::include_image!("../icons/x.svg"), 14.0, "Delete this chat", true, icon_tint(MUTED()))
                    .clicked()
                {
                    delete = Some(i);
                }
            });
        }
        if let Some(i) = delete {
            llm.delete_chat(i);
        }
    });
}

/// The Tools menu's conversation modes — one at a time.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatMode {
    Roleplay,
    Krea,
    TagImages,
}

/// Whether a conversation mode is on.
pub(crate) fn mode_on(llm: &LlmState, settings: &crate::settings::Settings, mode: ChatMode) -> bool {
    match mode {
        ChatMode::Roleplay => llm.roleplay.enabled,
        ChatMode::Krea => settings.ai_krea_prompts,
        ChatMode::TagImages => settings.ai_tag_images,
    }
}

/// Turn a conversation mode on or off. Turning one on switches the others
/// off, and either way a fresh chat starts: each mode changes the whole
/// conversation (its brief, persona or memories), so it never switches
/// mid-chat. (`new_chat` reuses the current chat if it's still empty.)
pub(crate) fn set_mode(llm: &mut LlmState, settings: &mut crate::settings::Settings, mode: ChatMode, on: bool) {
    if on {
        if mode != ChatMode::Roleplay && llm.roleplay.enabled {
            llm.roleplay.enabled = false;
            llm.roleplay.save();
        }
        if mode != ChatMode::Krea {
            settings.ai_krea_prompts = false;
            llm.krea_prompts = false;
        }
        if mode != ChatMode::TagImages {
            settings.ai_tag_images = false;
            llm.tag_images = false;
        }
    }
    match mode {
        ChatMode::Roleplay => {
            llm.roleplay.enabled = on;
            llm.roleplay.save();
        }
        ChatMode::Krea => {
            settings.ai_krea_prompts = on;
            llm.krea_prompts = on;
        }
        ChatMode::TagImages => {
            settings.ai_tag_images = on;
            llm.tag_images = on;
        }
    }
    llm.new_chat();
}

/// The conversation column: scrollable bubbles + the bottom input pill, both
/// centred on the same column (like the sketch's two guide lines).
fn conversation(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut crate::settings::Settings) {
    let avail_w = ui.available_width();
    // The centred content column both the messages and the pill live in.
    // Floor at 0: a squeezed panel can leave avail_w under 16, and a negative
    // set_width panics.
    let col_w = (avail_w * 0.58).clamp(420.0, 880.0).min((avail_w - 16.0).max(0.0));
    let pad = ((avail_w - col_w) / 2.0).max(0.0);

    // The conversation area under the input card. The reservation is
    // CONSTANT while typing — the input card floats on its own layer and
    // grows upward OVER the messages, so the chat never reflows as lines are
    // added. Only discrete actions (attaching an image, an error line)
    // change the reservation.
    let panel = ui.max_rect();
    let row_h = 20.0;
    let mut input_h = 104.0;
    if llm.draft_image.is_some() {
        // A 64px preview once the thumb/poster exists, a one-line chip until
        // a video's poster capture lands.
        input_h += if llm.draft_thumb.is_some() { 78.0 } else { 30.0 };
    }
    if llm.run_err.is_some() {
        input_h += 22.0;
    }
    if llm.detail_reason.is_some() && !llm.detail_notice_done {
        input_h += 36.0;
    }
    // Liquid Glass: the messages run on under the card, which shows them
    // frosted through its glass as they scroll by (the list ends with room
    // for the card, so the last message still rests above it).
    let under_glass = crate::liquid_glass::active();
    let shown_h = (ui.available_height() - input_h).max(80.0);
    let msgs_h = if under_glass { ui.available_height().max(80.0) } else { shown_h };

    egui::ScrollArea::vertical()
        .id_salt("ai_chat_msgs")
        .auto_shrink([false, false])
        .max_height(msgs_h)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            // Keep scrolling while a text selection is dragged past the edge
            // (under Liquid Glass, past the card's top: it's over the rest).
            let mut shown = ui.clip_rect();
            if under_glass {
                shown.max.y -= input_h;
            }
            crate::drag_select_autoscroll_in(ui, shown);
            ui.add_space(12.0);
            let chat = &llm.chats[llm.active_chat];
            if chat.msgs.is_empty() {
                ui.add_space(shown_h * 0.42);
                let hint = if llm.tag_images && llm.tag_target.is_some() {
                    "Ask the AI to tag this image, fix its tags or describe it — changes save right \
                     away, with Undo under each reply"
                } else if llm.tag_images {
                    "Open a folder to tag its images with the AI"
                } else if llm.krea_prompts {
                    "Attach a picture with + or describe an idea — get a Krea 2 prompt"
                } else {
                    "Ask AI anything — attach an image or a PDF with +, or pick a workflow in Tools"
                };
                ui.vertical_centered(|ui| {
                    ui.set_max_width(col_w);
                    ui.label(egui::RichText::new(hint).color(MUTED()).size(14.5));
                });
            }
            ui.horizontal(|ui| {
                ui.add_space(pad);
                ui.vertical(|ui| {
                    ui.set_width(col_w);
                    let n = llm.chats[llm.active_chat].msgs.len();
                    // Only the chat the reply streams into shows it live —
                    // another chat's last reply is finished text.
                    let live_chat = llm.streams_into(llm.chats[llm.active_chat].id);
                    let chat_id = llm.chats[llm.active_chat].id;
                    for i in 0..n {
                        let msg = &llm.chats[llm.active_chat].msgs[i];
                        let streaming = live_chat && i + 1 == n && msg.role == ChatRole::Model;
                        // A message out of sight isn't laid out again: its
                        // room is kept. (Every one was, every frame, and a
                        // long chat crawled while a reply came in.) Not the
                        // last one — it's what changes — nor one being
                        // edited or read out.
                        let at = Some((chat_id, i));
                        let settled = i + 1 < n && llm.editing != at && llm.speak_target != at;
                        // (Its picture comes in a moment after it does.)
                        let picture = msg.image.as_ref().and_then(|path| llm.msg_thumbs.get(path));
                        let picture = picture.map(|thumb| thumb.as_ref().map(|t| t.size()));
                        let key = message_key(msg, picture, col_w, ui.ctx().pixels_per_point());
                        show_or_keep_room(ui, ("ai_msg", chat_id, i), key, col_w, settled, |ui| {
                            message(ui, llm, i, streaming);
                        });
                        ui.add_space(12.0);
                    }
                    // A retry or edit clicked inside the list is applied
                    // here, after the loop, so it can safely truncate the
                    // message list.
                    if let Some(i) = llm.pending_retry.take() {
                        llm.retry(i, ui.ctx());
                    }
                    if let Some(i) = llm.pending_edit.take() {
                        let text = std::mem::take(&mut llm.edit_draft);
                        llm.editing = None;
                        llm.apply_edit(i, text, ui.ctx());
                    }
                });
            });
            ui.add_space(if under_glass { 6.0 + input_h } else { 6.0 });
        });

    // --- Bottom input card ---
    // the text area beneath it (capped at 8 visible lines, scrolling
    // internally beyond that), and a bottom row with + / send. The card lives
    // on its OWN floating layer, anchored to the panel's bottom and centred
    // on the conversation column — growing text extends it upward over the
    // chat without reflowing the messages underneath.
    ensure_draft_thumb(ui.ctx(), llm);

    // Drag-and-drop onto the card: while an image is dragged over it the card
    // expands to its full height and turns blue with a centred attach icon
    // (like the text-to-image prompt box); dropping attaches the file. The
    // hover/drop checks use last frame's card rect; an unknown pointer
    // position mid-drag (common on Windows) counts as over the card.
    // (Not in Tag images: the image on the left is what gets sent.)
    let dragging_files =
        ui.input(|i| !i.raw.hovered_files.is_empty()) && !(llm.tag_images && llm.tag_target.is_some());
    if dragging_files {
        ui.ctx().request_repaint();
    }
    let hover_pos = ui.input(|i| i.pointer.hover_pos());
    let drag_over_card = dragging_files
        && llm
        .input_rect
        .is_none_or(|(r, _)| hover_pos.is_none_or(|p| r.contains(p)));
    let dropped: Vec<std::path::PathBuf> = ui.input(|i| {
        i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
    });
    if !dropped.is_empty() && claims_drop(llm, ui.ctx()) {
        // A PDF takes the draft's PDF slot (and starts being read); a
        // picture or video its picture slot. Not in Tag images.
        let tagging = llm.tag_images && llm.tag_target.is_some();
        if let Some(p) = dropped.iter().find(|p| crate::pdf::is_pdf(p)).filter(|_| !tagging) {
            attach_doc(llm, p.clone(), ui.ctx());
        }
        if let Some(p) = dropped.into_iter().find(|p| {
            crate::is_video(p)
                || p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| DROP_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        }) {
            llm.draft_image = Some(p);
            llm.run_err = None;
        }
    }

    let screen = ui.ctx().content_rect();
    let off_x = panel.center().x - screen.center().x;
    let area = egui::Area::new(egui::Id::new("ai_chat_input_card"))
        .order(egui::Order::Foreground)
        .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(off_x, -14.0))
        .show(ui.ctx(), |ui| {
            // Glass themes: repaint the window backdrop (pixel-aligned) as an
            // opaque patch under the card, so the card can share the other
            // panels' translucent PANEL fill — matching the top bar and chat
            // list exactly — without messages scrolled beneath showing through.
            // Uses last frame's rect (stored below); the one-frame lag while the
            // card grows is invisible. No-op outside Glass, and in Liquid
            // Glass, where the card is real glass over the messages.
            if let Some((r, _)) = llm.input_rect {
                crate::theme::paint_glass_patch(ui.ctx(), ui.painter(), r);
            }
            // While a file is dragged over: the expanded blue drop zone replaces
            // the whole card.
            if drag_over_card {
                let drop_h = 8.0 * row_h + 44.0;
                egui::Frame::new()
                    .fill(DROP_BLUE.gamma_multiply(0.18))
                    .stroke(egui::Stroke::new(1.5_f32, DROP_BLUE))
                    .corner_radius(egui::CornerRadius::same(24))
                    .inner_margin(egui::Margin::symmetric(12, 10))
                    .show(ui, |ui| {
                        ui.set_width(col_w - 24.0);
                        ui.set_height(drop_h);
                        ui.vertical_centered(|ui| {
                            ui.add_space(drop_h / 2.0 - 34.0);
                            ui.add(
                                egui::Image::new(egui::include_image!("../icons/attach_file.svg"))
                                    .fit_to_exact_size(egui::vec2(38.0, 38.0))
                                    .tint(DROP_BLUE),
                            );
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("Add files here").color(DROP_BLUE).size(13.5));
                        });
                    });
                return;
            }

            if let Some(e) = llm.run_err.clone() {
                ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
                ui.add_space(2.0);
            }
            // Image detail was lowered for this GPU: a quiet one-liner, shown
            // once per session (until the next message is sent).
            if let Some(reason) = llm.detail_reason.clone().filter(|_| !llm.detail_notice_done) {
                ui.scope(|ui| {
                    ui.set_max_width(col_w);
                    ui.label(egui::RichText::new(reason).color(MUTED()).size(11.5));
                });
                ui.add_space(2.0);
            }
            // The same translucent PANEL fill as the top bar and the side cards —
            // the glass-patch underlay painted above keeps messages from showing
            // through it. A soft shadow (same family as card_frame's) lifts the
            // card visually above the chat.
            egui::Frame::new()
                .fill(PANEL())
                .stroke(egui::Stroke::new(1.0_f32, EDGE()))
                .corner_radius(egui::CornerRadius::same(24))
                .inner_margin(egui::Margin::symmetric(12, 10))
                .shadow(egui::epaint::Shadow {
                    offset: [0, 4],
                    blur: 14,
                    spread: 0,
                    color: egui::Color32::from_black_alpha(110),
                })
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.set_width(col_w - 24.0);

                        // The PDFs the AI reads with every message of this
                        // chat, each with a ✕ to put it away.
                        docs_in_use(ui, llm);

                        // Attached-media preview (rounded, with a remove ✕): image
                        // thumbnail, or a video's poster still with a play badge —
                        // a filename chip while the poster is still being captured.
                        if let Some(path) = llm.draft_image.clone() {
                            let tex = llm.draft_thumb.as_ref().map(|(_, t)| t.clone());
                            ui.horizontal(|ui| {
                                if let Some(tex) = tex {
                                    let size = tex.size_vec2();
                                    let scale = (64.0 / size.y).min(64.0 / size.x);
                                    let resp = ui.add(
                                        egui::Image::new(&tex)
                                            .fit_to_exact_size(size * scale)
                                            .corner_radius(egui::CornerRadius::same(10)),
                                    );
                                    if crate::is_video(&path) {
                                        paint_play_badge(ui, resp.rect);
                                    }
                                } else {
                                    // The name leaves room for the ✕ after it.
                                    let x_w = ui
                                        .fonts_mut(|f| f.layout_no_wrap("✕".to_string(), egui::FontId::proportional(11.0), MUTED()))
                                        .size()
                                        .x
                                        + 2.0 * ui.spacing().button_padding.x
                                        + ui.spacing().item_spacing.x;
                                    media_chip(ui, &path, icon_tint(MUTED()), x_w);
                                }
                                if ui
                                    .add(egui::Button::new(egui::RichText::new("✕").color(MUTED()).size(11.0)).frame(false))
                                    .on_hover_text("Remove the attachment")
                                    .clicked()
                                {
                                    llm.draft_image = None;
                                    llm.draft_thumb = None;
                                }
                            });
                            ui.add_space(6.0);
                        }
                        // An attached PDF: its first page, name and size —
                        // or how reading it is going — with a remove ✕.
                        if let Some(path) = llm.draft_doc.clone() {
                            let state = crate::pdf::request(&path, ui.ctx());
                            ui.horizontal(|ui| {
                                doc_card(ui, llm, &path, &state, false);
                                if remove_button(ui, 16.0, "Remove the PDF").clicked() {
                                    llm.draft_doc = None;
                                    llm.run_err = None;
                                }
                            });
                            ui.add_space(6.0);
                        }

                        // Draft text: grows with its content (the card extends upward
                        // because it's bottom-anchored), scrolls inside past 8 lines.
                        // Enter sends; Shift+Enter makes a newline (the editor's
                        // return key, so a plain Enter never inserts one — not
                        // even mid-text, where trimming couldn't remove it).
                        let mut send_now = false;
                        egui::ScrollArea::vertical()
                            .id_salt("ai_chat_draft")
                            .max_height(8.0 * row_h)
                            .auto_shrink([false, true])
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                let out = egui::TextEdit::multiline(&mut llm.draft)
                                    .desired_rows(1)
                                    .frame(egui::Frame::NONE)
                                    .font(egui::FontId::proportional(15.0))
                                    .hint_text("Ask the AI")
                                    .desired_width(f32::INFINITY)
                                    .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter))
                                    .show(ui);
                                // Live spell-check: red squiggles + right-click
                                // suggestions, same as the generator prompt boxes.
                                crate::spellcheck::attach(ui, &out, &mut llm.draft, &mut llm.spell);
                                let resp = &out.response.response;
                                let enter = resp.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                                if enter {
                                    send_now = true;
                                }
                            });

                        // Bottom row: attach on the left, streaming status in the
                        // middle, send on the right.
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            // Tag images sends the image on the left with every
                            // message, so attaching another one is switched off.
                            let tagging = llm.tag_images && llm.tag_target.is_some();
                            let attach_tip = if tagging {
                                "Tag images sends the image on the left with every message"
                            } else {
                                "Attach an image, a video or a PDF"
                            };
                            if icon_button(ui, egui::include_image!("../icons/add.svg"), 20.0, attach_tip, !llm.running && !tagging)
                                .clicked()
                                && !tagging
                                && let Some(path) = rfd::FileDialog::new()
                                .add_filter("Images, videos and PDFs", &[DROP_EXTS, VIDEO_PICK_EXTS, &["pdf"]].concat())
                                .add_filter("Images", DROP_EXTS)
                                .add_filter("Videos", VIDEO_PICK_EXTS)
                                .add_filter("PDFs", &["pdf"])
                                .pick_file()
                            {
                                if crate::pdf::is_pdf(&path) {
                                    attach_doc(llm, path, ui.ctx());
                                } else {
                                    llm.draft_image = Some(path);
                                    llm.run_err = None;
                                }
                            }
                            // Tools menu: extra chat abilities with on/off toggles.
                            let tools_resp = icon_button(ui, egui::include_image!("../icons/tools.svg"), 18.0, "Tools", true);
                            if tools_resp.clicked() {
                                llm.tools_open = !llm.tools_open;
                                llm.chats_open = false;
                            }
                            // Chats: this session's conversations, and a new one
                            // (the view has no chat list panel).
                            let chats_resp = icon_button(ui, egui::include_image!("../icons/chats.svg"), 17.0, "Chats", true);
                            if chats_resp.clicked() {
                                llm.chats_open = !llm.chats_open;
                                llm.tools_open = false;
                            }
                            if llm.chats_open {
                                let mut open = true;
                                egui::Popup::from_response(&chats_resp)
                                    .open_bool(&mut open)
                                    .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                                    .align(egui::RectAlign::TOP_START) // opens upward, like Tools
                                    .width(260.0)
                                    .gap(8.0)
                                    .frame(crate::float_card_frame(14))
                                    .show(|ui| chat_list(ui, llm));
                                // Closed by a click outside, or by picking a chat.
                                llm.chats_open &= open;
                            }
                            // Web research is on: its globe (a click turns it off).
                            web_research::composer_globe(ui, llm, settings);
                            // Remember across chats: the AI is writing down
                            // what a chat brought (sending anything stops it).
                            if !llm.running && llm.memory_activity() == crate::llm::MemoryActivity::Updating {
                                ui.add_space(2.0);
                                ui.add(egui::Spinner::new().size(14.0).color(MUTED()));
                                ui.label(egui::RichText::new("Saving to memory…").color(MUTED()).size(11.5))
                                    .on_hover_text(
                                        "The AI is writing down what your chat brought (AI Settings → Memory). \
                                         Sending a message stops it; it picks up again later.",
                                    );
                            } else if let Some(crate::memory::LastPass::Saved { at, added, notes, .. }) = llm.memory_last()
                                && (*added > 0 || *notes)
                                && crate::memory::now().saturating_sub(*at) < 8
                            {
                                // Just saved something: say so for a few seconds.
                                ui.add_space(2.0);
                                let what = match (*added, *notes) {
                                    (0, _) => "its notes about itself".to_string(),
                                    (1, _) => "1 memory".to_string(),
                                    (n, _) => format!("{n} memories"),
                                };
                                ui.label(egui::RichText::new("Saved to memory").color(MUTED()).size(11.5))
                                    .on_hover_text(format!("The AI just saved {what} (AI Settings → Memory)."));
                                ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
                            }
                            tools_menu::popup(&tools_resp, llm, settings);
                            // A PDF still being read (or that can't be) holds
                            // the message back; its card says which.
                            let doc_ready = llm.draft_doc.as_ref().is_none_or(|d| {
                                matches!(crate::pdf::request(d, ui.ctx()), crate::pdf::State::Ready(_))
                            });
                            let can_send = !llm.running
                                && doc_ready
                                && (!llm.draft.trim().is_empty() || llm.draft_image.is_some() || llm.draft_doc.is_some());
                            // While a reply streams the send icon becomes a stop
                            // button, same as the text-to-image view. With an
                            // empty draft only the mic shows; once there's
                            // something to send the blue send icon takes the
                            // right edge and the mic slides one slot left.
                            let send_clicked = ui
                                .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    let clicked = if llm.running {
                                        if icon_button(ui, egui::include_image!("../icons/stop.svg"), 16.0, "Stop generating", true)
                                            .clicked()
                                        {
                                            llm.stop_generation();
                                        }
                                        false
                                    } else if can_send {
                                        icon_button_tinted(ui, egui::include_image!("../icons/send.svg"), 19.0, "Send", true, DROP_BLUE)
                                            .clicked()
                                    } else {
                                        false
                                    };
                                    mic_button(ui, llm);
                                    chat_model_selector(ui, llm, settings);
                                    clicked
                                })
                                .inner;

                            if (send_now || send_clicked) && can_send {
                                llm.send_draft(ui.ctx());
                            }
                        });
                    });
                });
        });

    // Publish the card's rect for next frame's drag-highlight and for the
    // gallery's drop handler to check (claims_drop).
    llm.input_rect = Some((area.response.rect, ui.ctx().input(|i| i.time)));
}

/// What a message's height hangs on, for [`show_or_keep_room`]: what it
/// says and holds, its `picture` as far as that has loaded (not yet, not
/// readable, or its size), and how wide (and at what zoom) it's set.
fn message_key(msg: &ChatMsg, picture: Option<Option<[usize; 2]>>, width: f32, zoom: f32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (msg.role == ChatRole::User).hash(&mut hasher);
    msg.text.hash(&mut hasher);
    msg.thinking.hash(&mut hasher);
    msg.image.hash(&mut hasher);
    msg.doc.hash(&mut hasher);
    msg.tag_image.hash(&mut hasher);
    msg.tag_change.is_some().hash(&mut hasher);
    msg.goals.len().hash(&mut hasher);
    msg.growth.len().hash(&mut hasher);
    msg.did.len().hash(&mut hasher);
    msg.character.hash(&mut hasher);
    msg.research.is_some().hash(&mut hasher);
    msg.krea_prompt.hash(&mut hasher);
    picture.hash(&mut hasher);
    width.to_bits().hash(&mut hasher);
    zoom.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// Draw `add` — or, where it's out of sight, only keep the room it took the
/// last time it was drawn (`width` wide), without laying it out again. It's
/// drawn whenever it would be in sight, when what its height hangs on (`key`)
/// has changed since it was measured, and when it isn't `settled`. True when
/// it was drawn.
///
/// What's drawn has ids of its own (by `id`), so they don't shift as the
/// things before it go in and out of sight.
fn show_or_keep_room(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash,
    key: u64,
    width: f32,
    settled: bool,
    add: impl FnOnce(&mut egui::Ui),
) -> bool {
    let id = egui::Id::new(id);
    let measured = ui.data(|d| d.get_temp::<(u64, f32)>(id)).filter(|(at, _)| *at == key).map(|(_, height)| height);
    if let Some(height) = measured.filter(|_| settled) {
        let room = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(width, height));
        if !ui.is_rect_visible(room) {
            ui.allocate_space(room.size());
            return false;
        }
    }
    let height = ui.push_id(id, add).response.rect.height();
    ui.data_mut(|d| d.insert_temp(id, (key, height)));
    true
}

/// The button under a Krea 2 prompt.
const TO_IMAGE: &str = "Use in Text to Image";

/// One conversation entry. User messages sit right in an accent-tinted
/// bubble (corner radius 22) that hugs the text; the model's replies render
/// with no bubble at all — just modern formatted markdown (real bold, bullet
/// lists, headings, code blocks) straight on the background, like current
/// assistant UIs. A streaming reply shows a ▌ cursor.
fn message(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, streaming: bool) {
    let (role, text, stored_thinking, image, doc, tag_image, tag_change, goals, growth, did, krea_prompt) = {
        let m = &llm.chats[llm.active_chat].msgs[index];
        (
            m.role,
            m.text.clone(),
            m.thinking.clone(),
            m.image.clone(),
            m.doc.clone(),
            m.tag_image.clone(),
            m.tag_change.clone(),
            m.goals.clone(),
            m.growth.clone(),
            m.did.clone(),
            m.krea_prompt,
        )
    };

    if role == ChatRole::User {
        let max_w = ui.available_width() * 0.82;
        let chat_id = llm.chats[llm.active_chat].id;
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            // Tag images: which image this message was about.
            if let Some(path) = &tag_image {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.add(
                        egui::Image::new(egui::include_image!("../icons/tag.svg"))
                            .fit_to_exact_size(egui::vec2(12.0, 12.0))
                            .tint(chat_icon_tint(MUTED())),
                    );
                    // Cut short at the column's edge (the hover has the whole
                    // path): a name wider than the column — a narrow one, with
                    // the picture beside it — slid the conversation off to
                    // the left.
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(crate::ai_tagging::file_name(path))
                                .color(chat_text_override().unwrap_or(MUTED()))
                                .size(11.5),
                        )
                        .truncate()
                        .show_tooltip_when_elided(false),
                    )
                    .on_hover_text(path.display().to_string());
                });
                ui.add_space(2.0);
            }
            // The attached image shows as its own rounded thumbnail, separate
            // from (and above) the text bubble.
            if let Some(path) = &image {
                match msg_thumb(ui.ctx(), llm, path) {
                    Some(tex) => {
                        let size = tex.size_vec2();
                        let scale = (260.0 / size.y).min(400.0 / size.x).min(1.0);
                        let resp = ui
                            .add(
                                egui::Image::new(&tex)
                                    .fit_to_exact_size(size * scale)
                                    .corner_radius(egui::CornerRadius::same(12))
                                    .sense(egui::Sense::click()),
                            )
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .on_hover_text(if crate::is_video(path) {
                                "Click to play"
                            } else {
                                "Click to enlarge"
                            });
                        // A clip's poster gets a play badge — the lightbox
                        // (click) is where it actually plays.
                        if crate::is_video(path) {
                            paint_play_badge(ui, resp.rect);
                        }
                        if resp.clicked() {
                            llm.lightbox = Some(Lightbox::open(path, ui.ctx()));
                        }
                    }
                    None => {
                        media_chip(ui, path, chat_icon_tint(MUTED()), 0.0);
                    }
                }
                if !text.is_empty() {
                    ui.add_space(4.0);
                }
            }
            // The attached PDF: its card opens it.
            if let Some(path) = &doc {
                if image.is_some() {
                    ui.add_space(4.0);
                }
                let state = crate::pdf::request(path, ui.ctx());
                let resp = doc_card(ui, llm, path, &state, true);
                if resp.clicked() {
                    crate::scan::open_in_default(path);
                }
                if !text.is_empty() {
                    ui.add_space(4.0);
                }
            }

            // Editing: the bubble becomes an in-place editor. Enter (or the
            // send icon) resends from here — everything after this message
            // is discarded and the reply regenerates. Esc cancels.
            if llm.editing == Some((chat_id, index)) {
                egui::Frame::new()
                    .fill(ui.visuals().extreme_bg_color)
                    .stroke(egui::Stroke::new(1.0_f32, EDGE()))
                    .corner_radius(egui::CornerRadius::same(16))
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.set_width(max_w);
                        let resp = ui.add(
                            egui::TextEdit::multiline(&mut llm.edit_draft)
                                .desired_rows(2)
                                .frame(egui::Frame::NONE)
                                .font(egui::FontId::proportional(14.0))
                                .desired_width(f32::INFINITY)
                                .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter)),
                        );
                        let enter = resp.has_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                        let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
                        ui.horizontal(|ui| {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let can_send = !llm.running
                                    && (!llm.edit_draft.trim().is_empty() || image.is_some() || doc.is_some());
                                let send = msg_icon_button(
                                    ui,
                                    egui::include_image!("../icons/send.svg"),
                                    16.0,
                                    "Send — the conversation continues from here",
                                    can_send,
                                )
                                    .clicked();
                                let cancel = msg_icon_button(
                                    ui,
                                    egui::include_image!("../icons/x.svg"),
                                    14.0,
                                    "Cancel editing",
                                    true,
                                )
                                    .clicked();
                                if (send || enter) && can_send {
                                    llm.pending_edit = Some(index);
                                }
                                if cancel || escape {
                                    llm.editing = None;
                                    llm.edit_draft.clear();
                                }
                            });
                        });
                    });
                return;
            }

            if !text.is_empty() {
                // iMessage-style bubble: SOLID accent fill with white text.
                // The old translucent accent (28%) washed out over the Glass
                // themes' animated backdrops and read as "behind" the chat.
                // A custom chat text colour (Appearance tab) wins over white.
                let bubble_text = chat_text_override().unwrap_or(egui::Color32::WHITE);
                egui::Frame::new()
                    .fill(ACCENT1())
                    .corner_radius(egui::CornerRadius::same(18))
                    .inner_margin(egui::Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        // Hug short messages: cap the bubble at the text's own
                        // measured width (the widest line for multi-line
                        // drafts), up to 82% of the column.
                        let est = ui
                            .painter()
                            .layout_no_wrap(text.clone(), egui::FontId::proportional(14.0), bubble_text)
                            .size()
                            .x
                            + 12.0;
                        ui.set_max_width(est.min(max_w));
                        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                            // Emoji-aware label: color Twemoji instead of
                            // monochrome font glyphs.
                            crate::emoji::label(ui, &text, bubble_text, 14.0, false);
                        });
                    });
            }

            // Edit action under the bubble (mirrors the reply action row).
            ui.add_space(2.0);
            let can_edit = !llm.running;
            if msg_icon_button(ui, egui::include_image!("../icons/edit.svg"), 14.0, "Edit this message", can_edit)
                .clicked()
                && can_edit
            {
                llm.editing = Some((chat_id, index));
                llm.edit_draft.clone_from(&text);
            }
        });
        return;
    }

    // Model reply. An album picture the AI chose to share shows above the
    // text, on its side of the column.
    if let Some(path) = &image
        && let Some(tex) = msg_thumb(ui.ctx(), llm, path) {
        let size = tex.size_vec2();
        let scale = (260.0 / size.y).min(400.0 / size.x).min(1.0);
        let resp = ui
            .add(
                egui::Image::new(&tex)
                    .fit_to_exact_size(size * scale)
                    .corner_radius(egui::CornerRadius::same(12))
                    .sense(egui::Sense::click()),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(if crate::is_video(path) { "Click to play" } else { "Click to enlarge" });
        if crate::is_video(path) {
            paint_play_badge(ui, resp.rect);
        }
        if resp.clicked() {
            llm.lightbox = Some(Lightbox::open(path, ui.ctx()));
        }
        ui.add_space(4.0);
    }

    // Web research: what was looked up for this reply (shared, not copied).
    let research = llm.chats[llm.active_chat].msgs[index].research.clone();

    // The thought channel (the 31B reasons before answering):
    // split live while the reply streams; a finished message keeps its
    // thoughts on the `thinking` field (llm.rs moves them there on Done).
    let (thinking, body) = if streaming {
        crate::llm::split_channels(&text)
    } else {
        (stored_thinking.unwrap_or_default(), text.clone())
    };

    // Nothing visible streamed yet (empty, or only channel tags so far) → a
    // spinner with the worker status ("Loading the model…", "Thinking…")
    // where the reply will appear.
    if streaming && thinking.is_empty() && body.is_empty() {
        ui.horizontal(|ui| {
            // Muted like a caption normally; the chat text-colour override
            // (or the Aurora-glow black default) wins.
            let col = chat_text_override().or_else(chat_aurora_ink).unwrap_or(MUTED());
            ui.add(egui::Spinner::new().size(12.0).color(col));
            ui.label(egui::RichText::new(&llm.status).color(col).size(12.5));
        });
        // The sources are in while the answer is still being written.
        if let Some(report) = &research {
            ui.add_space(4.0);
            web_research::card(ui, llm, report, true);
        }
        return;
    }
    let mut shown = if streaming && research.is_some() {
        // A researched reply never carries out memory, picture or tag lines
        // (llm/web.rs takes them out when it ends) — never shown either.
        crate::ai_tagging::strip_lines(&crate::roleplay::strip_memory_lines(&body))
    } else if streaming && llm.gen_roleplay {
        // Diary lines are extracted for real when the reply finishes; hide
        // them while it streams so they never flash up. (Whether it's a
        // role-play reply was settled when it was asked.)
        crate::roleplay::strip_memory_lines(&body)
    } else if streaming && llm.gen_assistant {
        // The work assistant's control lines, the same way.
        crate::assistant::strip(&body)
    } else if streaming
        && (llm.tag_images
            || index.checked_sub(1).and_then(|i| llm.chats[llm.active_chat].msgs.get(i)).is_some_and(|m| m.tag_image.is_some()))
    {
        // Same for Tag images' tag-edit lines (also when the mode was
        // switched off while the reply streams).
        crate::ai_tagging::strip_lines(&body)
    } else {
        body.clone()
    };
    // Its citations ("[2]") link to the sources; nothing else links out
    // (worked out once, when the reply ends).
    if let Some(report) = &research {
        shown = match &report.display {
            Some(display) if !streaming => display.clone(),
            _ => crate::research::display_answer(&shown, &report.sources),
        };
    }
    if streaming && !shown.trim().is_empty() {
        markdown::push_cursor(&mut shown);
    }

    // Reply text renders bare on the background (no bubble/card) — only the
    // thoughts well and code blocks get their own surfaces.
    if !thinking.is_empty() {
        thinking_row(ui, llm, index, &thinking, streaming && body.is_empty());
        if !shown.trim().is_empty() {
            ui.add_space(4.0);
        }
    }
    if !shown.trim().is_empty() {
        chat_markdown(ui, &mut llm.md_cache, &shown, false);
    } else if !streaming && !thinking.is_empty() {
        // Ended (stopped, failed, or the model said nothing out loud) while
        // still thinking: no answer, only the thought (in the Thoughts row
        // above) — say so rather than leave a blank.
        ui.add_space(2.0);
        let col = chat_text_override().or_else(chat_aurora_ink).unwrap_or(MUTED());
        ui.label(egui::RichText::new("Ended while still thinking — no answer").color(col).size(12.5).italics());
    }
    let has_text = !text.trim().is_empty();

    // Web research: the sources the reply was written from, and its notes.
    if let Some(report) = &research {
        ui.add_space(6.0);
        web_research::card(ui, llm, report, streaming);
    }

    // Tag images: what this reply changed in the image's tags, with Undo.
    if let Some(change) = &tag_change {
        ui.add_space(4.0);
        tag_change_note(ui, llm, index, change);
    }

    // Role play: goals this reply set, achieved or failed.
    if !goals.is_empty() {
        ui.add_space(4.0);
        goal_note(ui, &goals);
    }
    // ...and what it changed about the character itself.
    if !growth.is_empty() {
        ui.add_space(4.0);
        growth_note(ui, &growth);
    }
    // The work assistant: what this reply did in the app (it's the same
    // conversation here as in its card).
    for did in &did {
        ui.add_space(4.0);
        quiet_note(ui, egui::include_image!("../icons/check_circle.svg"), crate::tag_manager::AI_NEW_VIOLET, did);
    }

    // Action row under every finished reply: copy / regenerate / listen
    // (copy and listen only when there's text to copy or read).
    // (Text to Image isn't in the macOS build.)
    let to_image = krea_prompt && has_text && cfg!(not(target_os = "macos"));
    if !streaming {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            // A Krea 2 prompt's own button is at the row's far end — the
            // bottom right corner under the prompt, apart from the icons,
            // which keep clear of it.
            let room = if to_image { crate::scan::pill_width(ui, TO_IMAGE, crate::scan::Pill::Outline) + 12.0 } else { 0.0 };
            let icons = egui::vec2((ui.available_width() - room).max(0.0), 0.0);
            ui.allocate_ui_with_layout(icons, egui::Layout::left_to_right(egui::Align::Center), |ui| {
                if has_text
                    && msg_icon_button(ui, egui::include_image!("../icons/copy.svg"), 15.0, "Copy message", true)
                        .clicked()
                {
                    // A researched reply comes with its sources.
                    let copied = match &research {
                        Some(report) => web_research::copy_text(&text, report),
                        None => text.clone(),
                    };
                    ui.ctx().copy_text(copied);
                }
                let can_retry = !llm.running;
                if msg_icon_button(ui, egui::include_image!("../icons/retry.svg"), 15.0, "Regenerate this reply", can_retry)
                    .clicked()
                    && can_retry
                {
                    // Applied after the message list finishes drawing (this
                    // mutates the list being iterated).
                    llm.pending_retry = Some(index);
                }
                let speaking = llm.any_speaking();
                let tip = if speaking {
                    "Stop reading"
                } else if llm.voice.installed {
                    "Read aloud (OmniVoice)"
                } else {
                    "Read aloud"
                };
                let chat_id = llm.chats[llm.active_chat].id;
                if has_text && msg_icon_button(ui, egui::include_image!("../icons/listen.svg"), 15.0, tip, true).clicked() {
                    llm.speak_target = Some((chat_id, index));
                    llm.listen(&text, ui.ctx());
                }
                // Voice status belongs to the reply being read, not every reply.
                if llm.speak_target == Some((chat_id, index)) {
                    // Speech on its way with nothing playing yet (the model
                    // loading on first use, or the next sentence synthesizing).
                    if llm.voice.waiting() {
                        ui.add(egui::Spinner::new().size(14.0).color(MUTED()));
                        if let Some(status) = llm.voice.status(ui.ctx()) {
                            ui.add(egui::Label::new(egui::RichText::new(status).color(MUTED()).size(11.0)).truncate());
                        }
                    }
                    if let Some(e) = &llm.voice.last_err {
                        // One line, like the status above; the rest on hover. The
                        // engine's errors run long (PyTorch's out-of-memory one
                        // is some 400 characters) and went off the window's edge.
                        ui.add(
                            egui::Label::new(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(11.0))
                                .truncate(),
                        );
                    }
                }
            });
            if to_image {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let tip = "Put this prompt in Text to Image, on Krea 2, and open it. That leaves the chat \
                               and frees its model, so Krea 2 has the graphics card.";
                    if crate::scan::pill(ui, TO_IMAGE, crate::scan::Pill::Outline, None, true).on_hover_text(tip).clicked() {
                        llm.to_image = Some(text.clone());
                    }
                });
            }
        });
    }
}

/// Role play: one quiet line per goal a reply set, achieved or failed (the
/// character's goals, src/roleplay.rs) — small, so the story stays in front.
fn goal_note(ui: &mut egui::Ui, events: &[crate::roleplay::GoalEvent]) {
    use crate::roleplay::GoalStatus;
    for e in events {
        let (icon, tint, what) = match e.status {
            GoalStatus::Active => (egui::include_image!("../icons/flag.svg"), ACCENT1(), "New goal"),
            GoalStatus::Achieved => {
                (egui::include_image!("../icons/check_circle.svg"), egui::Color32::from_rgb(46, 160, 67), "Goal achieved")
            }
            GoalStatus::Failed => (egui::include_image!("../icons/x.svg"), MUTED(), "Goal failed"),
        };
        quiet_note(ui, icon, tint, &format!("{what}: {}", e.text));
    }
}

/// Role play: one quiet line per change a reply made to the character — a
/// trait gained or dropped, a part of its persona added, rewritten or
/// removed (src/roleplay.rs). The persona's can be undone in AI Settings →
/// Role play, where the traits are too.
fn growth_note(ui: &mut egui::Ui, events: &[crate::roleplay::GrowthEvent]) {
    for e in events {
        let (icon, tint, what) = growth_line(e.kind);
        quiet_note(ui, icon, tint, &format!("{what}: {}", e.text));
    }
}

/// A growth note's icon, its tint and what it says happened.
fn growth_line(kind: crate::roleplay::GrowthKind) -> (egui::ImageSource<'static>, egui::Color32, &'static str) {
    use crate::roleplay::GrowthKind;
    match kind {
        GrowthKind::TraitGained => (egui::include_image!("../icons/sparkle.svg"), ACCENT1(), "New trait"),
        GrowthKind::TraitDropped => (egui::include_image!("../icons/x.svg"), MUTED(), "Trait dropped"),
        GrowthKind::PersonaAdded => (egui::include_image!("../icons/edit.svg"), ACCENT1(), "Added to its persona"),
        GrowthKind::PersonaChanged => (egui::include_image!("../icons/edit.svg"), ACCENT1(), "Persona rewritten"),
        GrowthKind::PersonaRemoved => (egui::include_image!("../icons/x.svg"), MUTED(), "Taken out of its persona"),
    }
}

/// One small muted line under a reply, after its icon.
fn quiet_note(ui: &mut egui::Ui, icon: egui::ImageSource<'_>, tint: egui::Color32, text: &str) {
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add(egui::Image::new(icon).fit_to_exact_size(egui::vec2(13.0, 13.0)).tint(tint));
        let w = ui.available_width().min(560.0);
        ui.scope(|ui| {
            ui.set_max_width(w);
            ui.add(egui::Label::new(egui::RichText::new(text).color(MUTED()).size(11.5)).wrap());
        });
    });
}

/// Tag images: the note under a reply that changed tags — which image, the
/// tags it added (violet, like the Tag Manager's AI tint) and removed, and an
/// Undo button (main.rs carries the undo out).
fn tag_change_note(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, change: &crate::ai_tagging::TagChange) {
    let violet = crate::tag_manager::AI_NEW_VIOLET;
    // A deeper violet for text on the light themes' pale card.
    let violet_ink = if crate::theme::is_light() { egui::Color32::from_rgb(104, 72, 200) } else { violet };
    let chat_id = llm.chats[llm.active_chat].id;
    let file = crate::ai_tagging::file_name(&change.image);
    let (added, removed) = change.diff();
    // A new order or spelling is a change too, with nothing to list.
    let changed = change.changed();
    let ink = chat_text();
    egui::Frame::new()
        .fill(violet.gamma_multiply(0.07))
        .stroke(egui::Stroke::new(1.0_f32, violet.gamma_multiply(0.35)))
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width().min(560.0));
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/tag.svg"))
                        .fit_to_exact_size(egui::vec2(14.0, 14.0))
                        .tint(violet_ink),
                );
                let (title, col) = if let Some(e) = &change.error {
                    let title = if file.is_empty() {
                        format!("Couldn't change any tags: {e}")
                    } else {
                        format!("Couldn't change the tags on {file}: {e}")
                    };
                    (title, egui::Color32::from_rgb(210, 70, 70))
                } else if !changed {
                    (format!("No tag changes needed on {file}"), MUTED())
                } else if change.undone {
                    (format!("Undid the tag changes on {file}"), MUTED())
                } else {
                    (format!("Changed the tags on {file}"), ink)
                };
                ui.add(egui::Label::new(egui::RichText::new(title).color(col).size(12.5)).wrap());
                if changed && change.error.is_none() && !change.undone {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can_undo = !llm.running;
                        let undo = ui
                            .add_enabled(can_undo, egui::Button::new(egui::RichText::new("Undo").size(12.0)))
                            .on_hover_text("Put this image's tags back the way they were");
                        if undo.clicked() {
                            llm.tag_requests.push(crate::ai_tagging::TagRequest::Undo { chat_id, index });
                        }
                    });
                }
            });
            if changed {
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
                    const SHOWN: usize = 16;
                    let dim = change.undone;
                    if added.is_empty() && removed.is_empty() {
                        let what = if change.respelled() { "Same tags, new spelling" } else { "Same tags, new order" };
                        ui.label(egui::RichText::new(what).color(MUTED()).size(12.0));
                    }
                    for t in added.iter().take(SHOWN) {
                        let c = if dim { MUTED() } else { violet_ink };
                        ui.label(egui::RichText::new(format!("+ {t}")).color(c).size(12.0));
                    }
                    for t in removed.iter().take(SHOWN.saturating_sub(added.len())) {
                        ui.label(egui::RichText::new(format!("− {t}")).color(MUTED()).strikethrough().size(12.0));
                    }
                    let more = (added.len() + removed.len()).saturating_sub(SHOWN);
                    if more > 0 {
                        ui.label(egui::RichText::new(format!("and {more} more")).color(MUTED()).size(12.0));
                    }
                });
            }
            // A failed Undo leaves the change in place; say why, and Undo
            // stays to try again.
            if let Some(e) = change.undo_error.as_ref().filter(|_| changed && !change.undone) {
                ui.add_space(4.0);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!("Couldn't undo: {e}"))
                            .color(egui::Color32::from_rgb(210, 70, 70))
                            .size(12.0),
                    )
                    .wrap(),
                );
            }
        });
}

/// Markdown in the conversation: model replies, and (with `thought`) the
/// model's thoughts in the "Thoughts" box. Both share the code-surface and ink
/// rules; thoughts use smaller, muted body text so they stay quieter than the
/// answer, while bold and headings keep the normal chat ink (and the bold
/// font weight) so they still stand out.
fn chat_markdown(
    ui: &mut egui::Ui,
    cache: &mut egui_commonmark::CommonMarkCache,
    text: &str,
    thought: bool,
) {
    ui.scope(|ui| {
        let s = ui.style_mut();
        // Slightly larger body text for replies than egui's default; thoughts
        // keep the old 12.5 caption size, with headings and code scaled down
        // to match.
        if let Some(body) = s.text_styles.get_mut(&egui::TextStyle::Body) {
            body.size = if thought { 12.5 } else { 14.5 };
        }
        if thought {
            if let Some(h) = s.text_styles.get_mut(&egui::TextStyle::Heading) {
                h.size = 14.5;
            }
            if let Some(m) = s.text_styles.get_mut(&egui::TextStyle::Monospace) {
                m.size = 11.5;
            }
        }
        // Code gets SOLID, neutral surfaces (never translucent — over
        // the animated Glass backdrops code was unreadable): a quiet
        // near-black editor tone under light text, a soft grey under
        // dark ink, with a matching chip for inline `code`. Keyed off
        // the EFFECTIVE chat ink (theme, Aurora-glow default, or the
        // user's override) so a black-text setup never lands on dark
        // chips it can't be read on. The viewer takes the block fill
        // from `extreme_bg_color`, the chip fill from `code_bg_color`,
        // and the radius/stroke from the noninteractive widget style.
        let ink = chat_text();
        let dark_ink = (ink.r() as u16 + ink.g() as u16 + ink.b() as u16) < 384;
        let use_light_code_bg = crate::theme::is_light() || dark_ink;

        // Multi-line code blocks keep their solid background (these DO get rounded corners)
        let code_bg = if use_light_code_bg {
            egui::Color32::from_rgb(238, 240, 245)
        } else {
            egui::Color32::from_rgb(24, 26, 31)
        };

        // Inline `code` chips: a see-through grey veil — the backdrop
        // stays visible through the rounded chip. The vendored viewer
        // pairs it with light-grey text (it keys the code text colour
        // off the chip's brightness, see misc.rs).
        let inline_bg = egui::Color32::from_rgba_unmultiplied(128, 132, 140, 90);

        let v = ui.visuals_mut();
        v.extreme_bg_color = code_bg;
        v.code_bg_color = inline_bg;

        v.dark_mode = !use_light_code_bg;

        // Keep the mouse text selection highlight soft as well
        v.selection.bg_fill = ACCENT1().gamma_multiply(0.20);
        v.selection.stroke.color = TEXT();

        v.widgets.noninteractive.corner_radius = egui::CornerRadius::same(12);
        v.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
        // Reply text follows the Appearance tab's chat text-colour override
        // (chat_text() falls back to the theme's normal ink without one).
        v.override_text_color = Some(if thought {
            chat_text_override().or_else(chat_aurora_ink).unwrap_or(MUTED())
        } else {
            chat_text()
        });
        // Bold/heading text resolves through strong_text_color() — the
        // widgets.active colour — which BYPASSES override_text_color
        // (egui RichText::get_text_color), so under a dark ink it stayed
        // near-white and washed out over the bright backdrop. Pin it to
        // the ink. Links get a deep readable blue when the ink is dark.
        v.widgets.active.fg_stroke.color = chat_text();
        if dark_ink {
            v.hyperlink_color = egui::Color32::from_rgb(28, 110, 235);
        }
        // A table wider than the column scrolls sideways in its own strip
        // (the vendored viewer): its bar gets a line of its own under the
        // table, the handle showing without the pointer on it, so the table
        // reads as one that scrolls and its last row stays clear of the bar.
        {
            let scroll = &mut ui.style_mut().spacing.scroll;
            scroll.floating_allocated_width = 6.0;
            scroll.dormant_handle_opacity = 0.5;
        }
        // (Prices that would be read as maths, maths written `\( … \)`.)
        let text = markdown::prepare(text);
        egui_commonmark::CommonMarkViewer::new()
            .render_math_fn(Some(&render_math))
            // A line break in a reply is one on screen, as in chat apps (a
            // poem's lines, an address, a character's lines in role-play).
            .keep_line_breaks(true)
            .show(ui, cache, &text);
    });
}

/// The Thinking row's label: whose thoughts — a role-play character's when
/// it thinks in character, "Mira's thoughts" — and, while they stream, how
/// long so far.
fn thoughts_label(character: Option<&str>, live: bool, secs: Option<u64>) -> String {
    let label = match (character, live, secs) {
        (Some(c), true, Some(s)) => format!("{c} is thinking… {s} s"),
        (Some(c), true, None) => format!("{c} is thinking…"),
        (Some(c), false, _) => format!("{c}'s thoughts"),
        (None, true, Some(s)) => format!("Thinking… {s} s"),
        (None, true, None) => "Thinking…".to_string(),
        (None, false, _) => "Thoughts".to_string(),
    };
    // An unnamed character is "the AI": "The AI's thoughts".
    let mut chars = label.chars();
    chars.next().map(|f| f.to_uppercase().chain(chars).collect()).unwrap_or_default()
}

/// Where a reply's Thoughts row keeps whether it's open.
fn thoughts_open_id(chat_id: u64, index: usize) -> egui::Id {
    egui::Id::new(("ai_thinking_open", chat_id, index))
}

/// Show message `index` of chat `chat_id` with its Thoughts row open: a
/// reply that was only a thought (src/llm.rs), so what the model did write
/// is in sight under the line saying it gave no answer.
pub(crate) fn open_thoughts(ctx: &egui::Context, chat_id: u64, index: usize) {
    ctx.data_mut(|d| d.insert_temp(thoughts_open_id(chat_id, index), true));
}

/// The collapsible "Thinking" row above a reply: the little AI orb, a muted
/// label (pulsing softly while the thought still streams) and a chevron that
/// rotates as the section opens. Clicking anywhere on it (orb included)
/// fades in the model's thought channel underneath — an indented quote block
/// with an orb-accent rule, rather than a boxed card.
fn thinking_row(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, thinking: &str, live: bool) {
    let chat_id = llm.chats[llm.active_chat].id;
    let open_id = thoughts_open_id(chat_id, index);
    let mut open = ui.data_mut(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
    let mut toggled = false;
    // Drives both the chevron rotation and the body fade, so open/close
    // sweeps instead of snapping.
    let anim = ui.ctx().animate_bool(open_id.with("anim"), open);

    ui.horizontal(|ui| {
        let orb = llm.think_orbs.entry((chat_id, index)).or_default();
        orb.set_state(if live {
            crate::ai_orb::OrbState::Thinking
        } else {
            crate::ai_orb::OrbState::Idle
        });
        if orb.show(ui, 28.0, None).clicked() {
            toggled = true;
        }

        // While the thought streams, a seconds counter shows it hasn't
        // stalled (Max can think for minutes on a partly-CPU 31B). In role
        // play with thoughts in character they're the character's own.
        let character = llm.chats[llm.active_chat].msgs.get(index).and_then(|m| m.character.clone());
        let secs = llm.gen_started.filter(|_| live).map(|t| t.elapsed().as_secs());
        let label = thoughts_label(character.as_deref(), live, secs);
        let tip = match &character {
            Some(c) => format!("See what {c} thought before answering"),
            None => "See what the model thought before answering".to_string(),
        };
        // Muted like a caption normally; the chat text-colour override
        // (or the Aurora-glow black default) wins. While the thought streams
        // the label breathes, matching the orb's hot state.
        let mut col = chat_text_override().or_else(chat_aurora_ink).unwrap_or(MUTED());
        if live {
            let pulse = 0.55 + 0.45 * ((ui.input(|i| i.time) * 2.6).sin() as f32 * 0.5 + 0.5);
            col = col.gamma_multiply(pulse);
        }
        let galley = ui.painter().layout_no_wrap(
            label.to_string(),
            egui::FontId::proportional(12.5),
            col,
        );
        let size = egui::vec2(galley.size().x + 8.0 + 16.0, 22.0);
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
        let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tip);
        if resp.clicked() {
            toggled = true;
        }
        if ui.is_rect_visible(rect) {
            // A soft pill behind the label on hover, so the row reads as a
            // button without being dressed as one.
            if resp.hovered() {
                ui.painter().rect_filled(
                    rect.expand2(egui::vec2(6.0, 1.0)),
                    egui::CornerRadius::same(11),
                    FIELD().gamma_multiply(0.6),
                );
            }
            let text_pos = egui::pos2(rect.left(), rect.center().y - galley.size().y / 2.0);
            ui.painter().galley(text_pos, galley, col);
            // One chevron, rotated by the open animation (→ swings down to ↓).
            egui::Image::new(egui::include_image!("../icons/arrow_right.svg"))
                .tint(chat_icon_tint(MUTED()))
                .rotate(anim * std::f32::consts::FRAC_PI_2, egui::Vec2::splat(0.5))
                .paint_at(
                    ui,
                    egui::Rect::from_center_size(
                        egui::pos2(rect.right() - 8.0, rect.center().y),
                        egui::vec2(14.0, 14.0),
                    ),
                );
        }
    });
    if toggled {
        open = !open;
        ui.data_mut(|d| d.insert_temp(open_id, open));
    }

    if anim > 0.0 {
        ui.add_space(2.0);
        ui.scope(|ui| {
            ui.set_opacity(anim);
            // Glass keeps a whisper-of-white veil so the thoughts stay
            // readable over a bright backdrop; every other theme runs the
            // quote bare — the accent rule alone marks it.
            let fill = if crate::theme::is_glass() {
                egui::Color32::from_white_alpha(60)
            } else {
                egui::Color32::TRANSPARENT
            };
            let inner = egui::Frame::new()
                .fill(fill)
                .corner_radius(egui::CornerRadius::same(12))
                .inner_margin(egui::Margin { left: 24, right: 12, top: 6, bottom: 6 })
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    // Markdown like the replies (lists, bold, code blocks with
                    // highlighting), in the quieter thought style. Its own id
                    // so it never clashes with the reply's viewer below.
                    ui.push_id(("thought_md", chat_id, index), |ui| {
                        chat_markdown(ui, &mut llm.md_cache, thinking, true);
                    });
                });
            // The quote rule: a rounded orb-accent bar down the left edge.
            let r = inner.response.rect;
            let bar = egui::Rect::from_min_max(
                egui::pos2(r.left() + 9.0, r.top() + 5.0),
                egui::pos2(r.left() + 12.0, r.bottom() - 5.0),
            );
            ui.painter().rect_filled(
                bar,
                egui::CornerRadius::same(2),
                crate::theme::orb_color().gamma_multiply(0.55),
            );
        });
    }
}

/// The cached chat-display texture for a message's attached image — decoded
/// once (downscaled), `None` cached for unreadable files so they aren't
/// retried every frame (those fall back to a filename chip).
fn msg_thumb(
    ctx: &egui::Context,
    llm: &mut LlmState,
    path: &std::path::Path,
) -> Option<egui::TextureHandle> {
    // Image still or video poster, decoded off-thread; the chip shows while
    // it loads.
    llm.media_thumb(ctx, path)
}

/// Keep `draft_thumb` in sync with `draft_image`: decode the attached image
/// once (downscaled) into a GPU texture for the in-card preview. An
/// unreadable file clears the attachment with a readable error instead of
/// failing later inside the model worker.
fn ensure_draft_thumb(ctx: &egui::Context, llm: &mut LlmState) {
    let Some(path) = llm.draft_image.clone() else {
        llm.draft_thumb = None;
        return;
    };
    if llm.draft_thumb.as_ref().is_some_and(|(p, _)| *p == path) {
        return;
    }
    // Decoded on a background thread (a big photo or a video's poster
    // capture would stall the UI); a filename chip shows until it lands. The
    // same texture then serves the sent message. For a video the worker
    // samples the actual frames at send time.
    llm.draft_thumb = llm.media_thumb(ctx, &path).map(|t| (path.clone(), t));
    if !crate::is_video(&path) && llm.thumb_failed(&path) {
        llm.draft_image = None;
        llm.draft_thumb = None;
        llm.run_err = Some("Couldn't read that image.".to_string());
    }
}

/// A play badge painted over a video attachment's poster still, so a clip
/// doesn't pass for a photo — the thumbnail never plays, the frames go to
/// the model instead.
fn paint_play_badge(ui: &mut egui::Ui, rect: egui::Rect) {
    let c = rect.center();
    ui.painter().circle_filled(c, 15.0, egui::Color32::from_black_alpha(150));
    egui::Image::new(egui::include_image!("../icons/playbutton.svg"))
        .tint(egui::Color32::WHITE)
        .paint_at(ui, egui::Rect::from_center_size(c + egui::vec2(1.0, 0.0), egui::vec2(16.0, 16.0)));
}

/// Filename chip for an attachment with no thumbnail (an unreadable image, or
/// a video whose poster is still being captured): a small type icon + name.
/// Attach `path` (a PDF) to the draft and start reading it.
fn attach_doc(llm: &mut LlmState, path: std::path::PathBuf, ctx: &egui::Context) {
    crate::pdf::request(&path, ctx);
    llm.draft_doc = Some(path);
    llm.run_err = None;
}

/// A PDF attachment (`path`, read as far as `state` says): a small picture
/// of its first page beside its name and size — or how reading it is going,
/// or why it can't be read. Measured and painted as one piece, so it sits
/// right in a right-aligned message as well as in the message box. In a
/// sent message (`in_message`) it opens the PDF when clicked.
fn doc_card(
    ui: &mut egui::Ui,
    llm: &mut LlmState,
    path: &std::path::Path,
    state: &crate::pdf::State,
    in_message: bool,
) -> egui::Response {
    const PAD: f32 = 7.0;
    const THUMB: egui::Vec2 = egui::vec2(36.0, 48.0);
    const SPIN: f32 = 16.0;
    let name = crate::ai_tagging::file_name(path);
    let short: String = if name.chars().count() > 34 {
        format!("{}…", name.chars().take(33).collect::<String>())
    } else {
        name.clone()
    };
    let thumb = match state {
        crate::pdf::State::Ready(doc) => doc_thumb(ui.ctx(), llm, path, doc),
        _ => None,
    };
    let (status, status_colour) = match state {
        crate::pdf::State::Reading => ("Reading…".to_string(), MUTED()),
        crate::pdf::State::Ready(doc) => (format!("PDF · {}", doc.describe()), MUTED()),
        crate::pdf::State::Failed(why) => (format!("Couldn't read it: {why}"), egui::Color32::from_rgb(210, 70, 70)),
    };
    let spin = if matches!(state, crate::pdf::State::Reading) { SPIN } else { 0.0 };
    let name_colour = if in_message { chat_text_override().unwrap_or(chat_text()) } else { TEXT() };
    let icon_colour = if in_message { chat_icon_tint(MUTED()) } else { icon_tint(MUTED()) };
    let painter = ui.painter();
    let name_g = painter.layout(short, egui::FontId::proportional(13.0), name_colour, 250.0);
    let status_g = painter.layout(status, egui::FontId::proportional(11.5), status_colour, 250.0 - spin);
    let text = egui::vec2(
        name_g.size().x.max(status_g.size().x + spin),
        name_g.size().y + 2.0 + status_g.size().y,
    );
    let size = egui::vec2(PAD + THUMB.x + 9.0 + text.x + PAD + 3.0, PAD * 2.0 + THUMB.y.max(text.y));
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter();
    painter.rect(
        rect,
        egui::CornerRadius::same(12),
        FIELD(),
        egui::Stroke::new(1.0_f32, EDGE()),
        egui::StrokeKind::Inside,
    );
    // The first page — or the PDF icon until it's drawn.
    let slot = egui::Rect::from_min_size(egui::pos2(rect.left() + PAD, rect.center().y - THUMB.y / 2.0), THUMB);
    match &thumb {
        Some(tex) => {
            let scale = (THUMB.x / tex.size_vec2().x).min(THUMB.y / tex.size_vec2().y);
            let fit = egui::Rect::from_center_size(slot.center(), tex.size_vec2() * scale);
            egui::Image::new(tex).corner_radius(egui::CornerRadius::same(3)).paint_at(ui, fit);
        }
        None => {
            let icon = egui::Rect::from_center_size(slot.center(), egui::vec2(24.0, 24.0));
            egui::Image::new(egui::include_image!("../icons/pdf.svg")).tint(icon_colour).paint_at(ui, icon);
        }
    }
    let at = egui::pos2(slot.right() + 9.0, rect.center().y - text.y / 2.0);
    let status_at = at + egui::vec2(spin, name_g.size().y + 2.0);
    if spin > 0.0 {
        let s = egui::Rect::from_min_size(at + egui::vec2(0.0, name_g.size().y + 3.0), egui::vec2(11.0, 11.0));
        egui::Spinner::new().size(11.0).color(MUTED()).paint_at(ui, s);
        ui.ctx().request_repaint();
    }
    let painter = ui.painter();
    painter.galley(at, name_g, name_colour);
    painter.galley(status_at, status_g, status_colour);
    if in_message {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(format!("Open {name}"))
    } else {
        resp.on_hover_text(name)
    }
}

/// A small ✕ (the app's x icon) that takes something away; `size` across.
fn remove_button(ui: &mut egui::Ui, size: f32, tip: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size + 4.0, size + 4.0), egui::Sense::click());
    let colour = if resp.hovered() { TEXT() } else { MUTED() };
    egui::Image::new(egui::include_image!("../icons/x.svg"))
        .tint(icon_tint(colour))
        .paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(size, size)));
    resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tip)
}

/// The texture of a read PDF's first page (cached with the chat's other
/// attachment pictures).
fn doc_thumb(
    ctx: &egui::Context,
    llm: &mut LlmState,
    path: &std::path::Path,
    doc: &crate::pdf::Doc,
) -> Option<egui::TextureHandle> {
    if let Some(t) = llm.msg_thumbs.get(path) {
        return t.clone();
    }
    let tex = doc
        .preview
        .clone()
        .map(|img| ctx.load_texture(format!("ai-chat-pdf-{}", path.display()), img, Default::default()));
    llm.msg_thumbs.insert(path.to_path_buf(), tex.clone());
    tex
}

/// The PDFs the AI reads with every message of the active chat (but the
/// one in the draft), as small chips, each with a ✕ that puts it away.
fn docs_in_use(ui: &mut egui::Ui, llm: &mut LlmState) {
    if llm.tag_images && llm.tag_target.is_some() {
        return;
    }
    let docs: Vec<std::path::PathBuf> = llm.chats[llm.active_chat]
        .docs()
        .into_iter()
        .filter(|d| llm.draft_doc.as_ref() != Some(d))
        .collect();
    if docs.is_empty() {
        return;
    }
    let mut put_away = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
        ui.label(egui::RichText::new("Reading").color(MUTED()).size(11.5))
            .on_hover_text("The AI reads these PDFs with every message in this chat.");
        for d in &docs {
            egui::Frame::new()
                .fill(FIELD())
                .stroke(egui::Stroke::new(1.0_f32, EDGE()))
                .corner_radius(egui::CornerRadius::same(10))
                .inner_margin(egui::Margin::symmetric(7, 2))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(13.0, 13.0), egui::Sense::hover());
                    egui::Image::new(egui::include_image!("../icons/pdf.svg")).tint(icon_tint(MUTED())).paint_at(ui, rect);
                    let name = crate::ai_tagging::file_name(d);
                    let short: String = if name.chars().count() > 26 {
                        format!("{}…", name.chars().take(25).collect::<String>())
                    } else {
                        name.clone()
                    };
                    ui.label(egui::RichText::new(short).color(TEXT()).size(11.5)).on_hover_text(name);
                    if remove_button(ui, 12.0, "Put it away: the AI stops reading this PDF").clicked() {
                        put_away = Some(d.clone());
                    }
                });
        }
    });
    if let Some(d) = put_away {
        llm.chats[llm.active_chat].docs_off.push(d);
    }
    ui.add_space(6.0);
}

/// An attachment without a preview (a clip whose poster isn't ready, a file
/// that can't be read any more): its icon and name on one line. `reserve` is
/// the room to keep free after the name, for what the caller puts there.
fn media_chip(ui: &mut egui::Ui, path: &std::path::Path, tint: egui::Color32, reserve: f32) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    ui.horizontal(|ui| {
        let icon = if crate::is_video(path) {
            egui::Image::new(egui::include_image!("../icons/video.svg"))
        } else {
            egui::Image::new(egui::include_image!("../icons/image.svg"))
        };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
        icon.tint(tint).paint_at(ui, rect);
        // Cut short at the edge of the message box / the column (the whole
        // name on hover): a long one stretched the box past the window, or
        // slid the conversation off to the left.
        let room = (ui.available_width() - reserve).max(0.0);
        ui.scope(|ui| {
            ui.set_max_width(room);
            ui.add(egui::Label::new(egui::RichText::new(name).color(MUTED()).size(11.5)).truncate());
        });
    });
}

/// Render chat maths (`$...$` / `$$...$$`) as plain Unicode text (see
/// [`markdown::tex_to_unicode`]): there's no real maths engine, but what
/// local models write mostly reads well that way.
fn render_math(ui: &mut egui::Ui, tex: &str, inline: bool) {
    let size = if inline { 14.5 } else { 15.5 };
    // (What the fonts here haven't got is written another way.)
    let font = egui::FontId::proportional(size);
    let shown = markdown::tex_to_unicode(tex, &mut |c| ui.fonts_mut(|f| f.has_glyph(&font, c)));
    let rt = egui::RichText::new(shown).italics().color(chat_text()).size(size);
    if inline {
        ui.label(rt);
    } else {
        // Display maths gets its own centred line, roughly like real LaTeX.
        ui.vertical_centered(|ui| {
            ui.label(rt);
        });
    }
}

/// Tint for icons in the conversation (messages) area: the Appearance tab's
/// chat icon-colour override when set, otherwise the app-wide [`icon_tint`].
/// Only the message column uses this — the chats side panel and the composer
/// (message box) keep the theme's normal tint.
fn chat_icon_tint(fallback: egui::Color32) -> egui::Color32 {
    chat_icon_override()
        .or_else(chat_aurora_ink)
        .unwrap_or_else(|| icon_tint(fallback))
}

/// A bare icon button on a 28px click target (same look as the text-to-image
/// view's send button): the SVG paints dead-centre, tinted to the theme's
/// text colour, dimmed when disabled. Used by the side panel and the composer;
/// the messages area uses [`msg_icon_button`] so the icon-colour override
/// applies there only.
fn icon_button(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'_>,
    icon_size: f32,
    tip: &str,
    enabled: bool,
) -> egui::Response {
    icon_button_tinted(ui, icon, icon_size, tip, enabled, icon_tint(TEXT()))
}

/// [`icon_button`] for the conversation area: honours the Appearance tab's
/// chat icon-colour override.
fn msg_icon_button(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'_>,
    icon_size: f32,
    tip: &str,
    enabled: bool,
) -> egui::Response {
    icon_button_tinted(ui, icon, icon_size, tip, enabled, chat_icon_tint(TEXT()))
}

/// The model picker next to the mic. Closed, it blends into the composer
/// bar: no pill or border, just the model's name — with its thinking level
/// after a muted dot when it thinks ("Gemma 4 31B · Max") — and a small
/// arrow. It opens a popover docked 10 px above the input card (so it never
/// covers the draft) listing ONLY the installed models: a card each with a
/// tagline (and, for the models that think, what kind of thinker), the
/// active one tinted with a check. Under a divider, a "Thinking level" row
/// unfolds the cube slider in a recessed well. Hidden while nothing is
/// installed yet (Settings → AI Model handles first-time setup).
fn chat_model_selector(
    ui: &mut egui::Ui,
    llm: &mut LlmState,
    settings: &mut crate::settings::Settings,
) {
    // Linked to a host that runs the chat: its model, shown (the host picks
    // it); the thinking level still applies.
    let remote = crate::link::remote_chat().map(|(_, host)| host);
    let installed: Vec<crate::llm::GemmaModel> = if remote.is_some() {
        Vec::new()
    } else {
        crate::llm::GemmaModel::ALL.into_iter().filter(|m| m.installed()).collect()
    };
    if installed.is_empty() && remote.is_none() {
        return;
    }
    let (label, thinks) = match &remote {
        Some(_) => (
            crate::link::remote_chat_model().unwrap_or_else(|| "Host model".to_string()),
            crate::link::remote_chat_thinks(),
        ),
        None => (llm.model.label().to_string(), llm.model.thinks()),
    };
    let menu_id = ui.id().with("chat_model_menu");
    let open = egui::Popup::is_id_open(ui.ctx(), menu_id);

    // "Gemma 4 31B" + a muted " · Medium" when the model thinks (the dot
    // keeps "Qwen3-VL 30B Thinking · Off" from reading "Thinking Off").
    let mut job = egui::text::LayoutJob::default();
    job.append(&label, 0.0, egui::TextFormat::simple(egui::FontId::proportional(12.5), TEXT()));
    if thinks {
        job.append(
            &format!("· {}", settings.ai_gen.thinking.label()),
            5.0,
            egui::TextFormat::simple(egui::FontId::proportional(12.5), MUTED()),
        );
    }
    job.wrap = egui::text::TextWrapping::truncate_at_width(200.0);
    let galley = ui.fonts_mut(|f| f.layout_job(job));
    let (arrow, gap) = (14.0, 4.0);
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(galley.size().x + gap + arrow, 28.0), egui::Sense::click());
    let tip = match &remote {
        Some(host) => format!("Runs on {host} (Settings → Link) — thinking level"),
        None => "Model and thinking level".to_string(),
    };
    let resp = resp.on_hover_text(tip).on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let text_pos = egui::pos2(rect.left(), rect.center().y - galley.size().y / 2.0);
        ui.painter().galley(text_pos, galley, TEXT());
        let arrow_src = if open {
            egui::include_image!("../icons/arrow_up.svg")
        } else {
            egui::include_image!("../icons/arrow_down.svg")
        };
        let arrow_rect = egui::Rect::from_center_size(
            egui::pos2(rect.right() - arrow / 2.0, rect.center().y),
            egui::vec2(arrow, arrow),
        );
        egui::Image::new(arrow_src).tint(icon_tint(MUTED())).paint_at(ui, arrow_rect);
    }

    // Docked to the input card's top edge, not to this button: the button
    // sits in the card's bottom row, so a gap from it alone would cover the
    // draft. Right edges line up with the button. (The card's rect is last
    // frame's; before there is one, the button stands in.)
    let dock = llm.input_rect.map_or(resp.rect, |(card, _)| {
        egui::Rect::from_x_y_ranges(resp.rect.x_range(), card.top()..=card.top())
    });
    egui::Popup::menu(&resp)
        .id(menu_id)
        .anchor(dock)
        .align(egui::RectAlign::TOP_END)
        .gap(10.0)
        // Dragging the thinking slider mustn't close the menu; picking a
        // model closes it explicitly.
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        // The popover paints its own card (below); in the Glass themes it's
        // see-through glass, like the Settings window.
        .frame(egui::Frame::NONE)
        .show(|ui| {
            egui::Frame::new()
                .fill(picker_fill())
                .corner_radius(egui::CornerRadius::same(22))
                .inner_margin(egui::Margin::same(8))
                .stroke(egui::Stroke::new(1.0_f32, EDGE()))
                .shadow(egui::epaint::Shadow {
                    offset: [0, 16],
                    blur: 42,
                    spread: 0,
                    color: egui::Color32::from_black_alpha(165),
                })
                .show(ui, |ui| {
                    ui.set_width(292.0);
                    ui.spacing_mut().item_spacing.y = 3.0;

                    // What the list is: this PC's models, or the linked PC's.
                    let header = if remote.is_some() { "RUNS ON ANOTHER PC" } else { "INSTALLED MODELS" };
                    ui.horizontal(|ui| {
                        ui.add_space(8.0);
                        ui.label(egui::RichText::new(header).color(MUTED().gamma_multiply(0.85)).strong().size(9.5));
                    });
                    ui.add_space(2.0);

                    if let Some(host) = &remote {
                        ui.horizontal(|ui| {
                            ui.add_space(8.0);
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(format!("{label} on {host}")).color(TEXT()).strong().size(13.0));
                                ui.label(
                                    egui::RichText::new("Its model is picked on that PC. Settings → Link to run on this one.")
                                        .color(MUTED())
                                        .size(10.5),
                                );
                            });
                        });
                        ui.add_space(4.0);
                    }

                    for m in installed {
                        if model_row(ui, m, llm.model == m) {
                            settings.ai_gemma_model = m;
                            llm.set_model(m);
                            ui.close();
                        }
                    }

                    ui.add_space(3.0);
                    ui.separator();
                    ui.add_space(3.0);

                    // Thinking level — expands in place (no second popup): a row with
                    // the current level and a chevron that turns down to reveal the
                    // cube slider in a recessed well underneath.
                    let open_id = ui.id().with("thinking_expanded");
                    let mut expanded = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(false);
                    if thinking_row_header(ui, thinks, settings.ai_gen.thinking, expanded) {
                        expanded = !expanded;
                        ui.data_mut(|d| d.insert_temp(open_id, expanded));
                    }
                    let anim = ui.ctx().animate_bool(open_id.with("anim"), expanded);
                    if anim > 0.0 {
                        ui.scope(|ui| {
                            ui.set_opacity(anim);
                            ui.add_space(4.0 * anim);
                            egui::Frame::new()
                                .fill(picker_well_fill())
                                .corner_radius(egui::CornerRadius::same(15))
                                .inner_margin(egui::Margin::symmetric(12, 10))
                                .stroke(egui::Stroke::new(1.0_f32, picker_hairline()))
                                .show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.add_enabled_ui(thinks, |ui| {
                                        let mut level = settings.ai_gen.thinking;
                                        thinking_slider(ui, &mut level);
                                        if level != settings.ai_gen.thinking {
                                            settings.ai_gen.thinking = level;
                                            llm.params.thinking = level;
                                        }
                                    });
                                    // A hairline between the slider and what the level means.
                                    ui.add_space(6.0);
                                    let (line, _) =
                                        ui.allocate_exact_size(egui::vec2(ui.available_width(), 1.0), egui::Sense::hover());
                                    ui.painter().rect_filled(line, 0.0, picker_hairline());
                                    ui.add_space(5.0);
                                    // As tall as the longest level's hint (Max's runs to
                                    // three lines, most take one), so the slider doesn't
                                    // jump under the pointer as the level changes — the
                                    // popover grows upward from the card. Left-aligned:
                                    // the menu's justified layout would stretch the
                                    // gaps between words.
                                    let w = ui.available_width();
                                    let tallest = crate::llm::ThinkLevel::ALL
                                        .into_iter()
                                        .map(|l| {
                                            // settings::hint's text size outside the Settings window.
                                            let font = egui::FontId::proportional(11.0);
                                            ui.fonts_mut(|f| f.layout(thinking_hint(thinks, l), font, MUTED(), w)).size().y
                                        })
                                        .fold(0.0_f32, f32::max);
                                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                                        ui.set_min_height(tallest + 2.0); // + hint's space above
                                        crate::settings::hint(ui, &thinking_hint(thinks, settings.ai_gen.thinking));
                                    });
                                });
                        });
                    }
                });
        });
}

/// The picker's "Thinking level" row: a wireframe cube in the level's
/// colour, the label, the level in a pill tinted the same, and a chevron
/// that points right when folded and down when open — all grey while the
/// level is Off. True when clicked.
fn thinking_row_header(ui: &mut egui::Ui, thinks: bool, level: crate::llm::ThinkLevel, expanded: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 36.0), egui::Sense::click());
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        if expanded || resp.hovered() {
            let shade = if resp.hovered() { 0.06 } else { 0.03 };
            p.rect_filled(rect, egui::CornerRadius::same(13), TEXT().gamma_multiply(shade));
        }

        let on = thinks && level != crate::llm::ThinkLevel::Off;
        let col = if on { think_color(level.index() as f32 / 4.0) } else { MUTED() };
        paint_wire_cube(p, egui::pos2(rect.left() + 19.0, rect.center().y), 6.8, col);
        p.text(
            egui::pos2(rect.left() + 34.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            "Thinking level",
            egui::FontId::proportional(13.0),
            TEXT(),
        );

        // The level ("Max", "Medium", "Off", …) in a pill.
        let value = if thinks { level.label() } else { "Off" };
        let ink = if on { picker_ink(col, 0.45) } else { MUTED() };
        let text = p.layout_no_wrap(value.to_string(), egui::FontId::proportional(11.0), ink);
        let w = text.size().x + 16.0;
        let pill = egui::Rect::from_center_size(
            egui::pos2(rect.right() - 28.0 - w / 2.0, rect.center().y),
            egui::vec2(w, 20.0),
        );
        p.rect_filled(pill, egui::CornerRadius::same(10), col.gamma_multiply(0.18));
        p.rect_stroke(
            pill,
            egui::CornerRadius::same(10),
            egui::Stroke::new(1.0_f32, col.gamma_multiply(0.40)),
            egui::StrokeKind::Inside,
        );
        p.galley(pill.center() - text.size() / 2.0, text, ink);

        let turn = ui.ctx().animate_bool(ui.id().with("thinking_chevron"), expanded);
        egui::Image::new(egui::include_image!("../icons/arrow_right.svg"))
            .tint(icon_tint(MUTED()))
            .rotate(turn * std::f32::consts::FRAC_PI_2, egui::Vec2::splat(0.5))
            .paint_at(ui, egui::Rect::from_center_size(egui::pos2(rect.right() - 14.0, rect.center().y), egui::vec2(16.0, 16.0)));
    }
    resp.clicked()
}

/// The Thinking level row's icon: an isometric cube in outline — a hexagon
/// with the three edges that meet at the near corner.
fn paint_wire_cube(painter: &egui::Painter, c: egui::Pos2, r: f32, col: egui::Color32) {
    let (w, h) = (r * 0.866, r * 0.5);
    let top = egui::pos2(c.x, c.y - r);
    let tr = egui::pos2(c.x + w, c.y - h);
    let br = egui::pos2(c.x + w, c.y + h);
    let bot = egui::pos2(c.x, c.y + r);
    let bl = egui::pos2(c.x - w, c.y + h);
    let tl = egui::pos2(c.x - w, c.y - h);
    let stroke = egui::Stroke::new(1.5_f32, col);
    painter.add(egui::Shape::closed_line(vec![top, tr, br, bot, bl, tl], stroke));
    painter.line_segment([tl, c], stroke);
    painter.line_segment([tr, c], stroke);
    painter.line_segment([bot, c], stroke);
}

/// What the thinking level means, under the slider.
fn thinking_hint(thinks: bool, level: crate::llm::ThinkLevel) -> String {
    use crate::llm::ThinkLevel as L;
    if !thinks {
        return "This model answers directly — thinking applies to Gemma 4 31B, \
                Qwen3-VL 30B Thinking and DeepSeek R1."
            .to_string();
    }
    match level {
        L::Off => "Answers right away — fastest, but weaker on tricky questions.",
        L::Low => "A quick think before answering.",
        L::Medium => "Balanced — the default.",
        L::High => "Thinks it through. Slower replies.",
        L::Max => "Thinks the longest before answering: the slowest replies, and long chats \
                   keep fewer old messages in memory.",
    }
    .to_string()
}

/// One model in the picker: a rounded card — tinted, with an edge, for the
/// active one; faintly on hover — holding the name (followed, for the models
/// that think, by a badge saying what kind of thinker), the tagline beneath,
/// and a check at the right end on the active one. True when clicked.
fn model_row(ui: &mut egui::Ui, m: crate::llm::GemmaModel, active: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 48.0), egui::Sense::click());
    let resp = resp.on_hover_text(m.hint()).on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        let radius = egui::CornerRadius::same(15);
        if active {
            p.rect_filled(rect, radius, THINK_BLUE.gamma_multiply(0.15));
            p.rect_stroke(
                rect,
                radius,
                egui::Stroke::new(1.0_f32, picker_blue().gamma_multiply(0.38)),
                egui::StrokeKind::Inside,
            );
        } else if resp.hovered() {
            p.rect_filled(rect, radius, TEXT().gamma_multiply(0.05));
        }

        let x = rect.left() + 13.0;
        let title_y = rect.top() + 16.0;
        let title_ink = if active { TEXT() } else { TEXT().gamma_multiply(0.92) };
        let title = p.layout_no_wrap(m.label().to_string(), egui::FontId::proportional(13.5), title_ink);
        let title_w = title.size().x;
        p.galley(egui::pos2(x, title_y - title.size().y / 2.0), title, title_ink);

        if let Some(badge) = thinker_badge(m) {
            let ink = picker_badge_ink();
            let text = p.layout_no_wrap(badge.to_string(), egui::FontId::proportional(9.5), ink);
            let pill = egui::Rect::from_min_size(
                egui::pos2(x + title_w + 7.0, title_y - 8.5),
                egui::vec2(text.size().x + 12.0, 17.0),
            );
            p.rect_filled(pill, egui::CornerRadius::same(9), THINK_BLUE.gamma_multiply(0.18));
            p.rect_stroke(
                pill,
                egui::CornerRadius::same(9),
                egui::Stroke::new(1.0_f32, picker_blue().gamma_multiply(0.30)),
                egui::StrokeKind::Inside,
            );
            p.galley(pill.center() - text.size() / 2.0, text, ink);
        }

        p.text(
            egui::pos2(x, rect.top() + 33.5),
            egui::Align2::LEFT_CENTER,
            m.tagline(),
            egui::FontId::proportional(11.5),
            MUTED(),
        );

        if active {
            // The icon's tick fills only half its box: 20 points draws a
            // ~10-point check (at 15 it was a faint 7).
            egui::Image::new(egui::include_image!("../icons/check.svg"))
                .tint(picker_blue())
                .paint_at(ui, egui::Rect::from_center_size(egui::pos2(rect.right() - 18.0, rect.center().y), egui::vec2(20.0, 20.0)));
        }
    }
    resp.clicked()
}

/// The badge after a thinking model's name in the picker: how it's built,
/// and that it thinks. None for the models that answer straight away.
fn thinker_badge(m: crate::llm::GemmaModel) -> Option<&'static str> {
    use crate::llm::GemmaModel as G;
    if !m.thinks() {
        return None;
    }
    Some(match m {
        G::D31B => "Dense · Thinks",
        G::Qwen30B => "MoE · Thinks",
        G::DeepSeek32B | G::DeepSeek8B => "Thinks · Text only",
        _ => "Thinks",
    })
}

/// The picker popover's fill: in the Glass themes the Settings window's
/// see-through glass; otherwise a step darker than the card in dark themes,
/// the card's own white in light ones (darkened, it turns a muddy grey).
fn picker_fill() -> egui::Color32 {
    if is_glass() {
        FLOAT()
    } else if is_light() {
        PANEL()
    } else {
        PANEL().lerp_to_gamma(egui::Color32::BLACK, 0.14)
    }
}

/// The thinking slider's recessed well: sunk darker in dark themes, the
/// field grey in light ones.
fn picker_well_fill() -> egui::Color32 {
    if is_light() { FIELD() } else { FIELD().lerp_to_gamma(egui::Color32::BLACK, 0.25) }
}

/// The picker's hairlines: the well's edge and its divider (the slider's
/// rail is the same ink, a little stronger). Ink-based, because the theme's
/// EDGE is a faint dark line that vanishes on these dark fills — and is
/// fully clear in the Glass themes.
fn picker_hairline() -> egui::Color32 {
    TEXT().gamma_multiply(0.08)
}

/// The picker's blue: the active model's edge and check, the badges' edges
/// — deeper on light themes, where the pale blue washes out.
fn picker_blue() -> egui::Color32 {
    if is_light() { egui::Color32::from_rgb(37, 99, 235) } else { egui::Color32::from_rgb(96, 165, 250) }
}

/// The text on the picker's thinker badges.
fn picker_badge_ink() -> egui::Color32 {
    if is_light() { egui::Color32::from_rgb(29, 78, 216) } else { egui::Color32::from_rgb(147, 197, 253) }
}

/// `col` as text in the picker: lifted toward white by `lift` on dark
/// themes; deepened on light ones, where a lifted tint would wash out.
fn picker_ink(col: egui::Color32, lift: f32) -> egui::Color32 {
    if is_light() {
        col.lerp_to_gamma(egui::Color32::BLACK, 0.2)
    } else {
        col.lerp_to_gamma(egui::Color32::WHITE, lift)
    }
}

/// The input card's mic: local Whisper dictation (src/dictate.rs). Idle it's
/// a plain icon; recording turns it red and pulsing; transcription swaps in a
/// spinner; the first-ever click downloads the speech model with progress in
/// the tooltip. Finished transcripts are appended to the draft here too.
fn mic_button(ui: &mut egui::Ui, llm: &mut LlmState) {
    if let Some(text) = llm.dictate.poll(ui.ctx()) {
        if !llm.draft.is_empty() && !llm.draft.ends_with(char::is_whitespace) {
            llm.draft.push(' ');
        }
        llm.draft.push_str(&text);
    }
    let mic = egui::include_image!("../icons/mic.svg");
    match llm.dictate.phase() {
        crate::dictate::Phase::Downloading(pct) => {
            icon_button(ui, mic, 18.0, &format!("Downloading the speech model… {pct}%"), false);
        }
        crate::dictate::Phase::Transcribing => {
            ui.add_sized(egui::vec2(28.0, 28.0), egui::Spinner::new().size(16.0))
                .on_hover_text("Transcribing…");
        }
        crate::dictate::Phase::Recording => {
            let red = egui::Color32::from_rgb(255, 69, 58);
            let pulse = ((ui.input(|i| i.time) * 3.0).sin() as f32 * 0.5 + 0.5) * 0.5 + 0.5;
            let secs = llm.dictate.seconds();
            let tip = format!("Stop dictation ({}:{:02})", secs / 60, secs % 60);
            if icon_button_tinted(ui, mic, 18.0, &tip, true, red.gamma_multiply(pulse)).clicked() {
                llm.dictate.toggle(ui.ctx());
            }
        }
        crate::dictate::Phase::Idle => {
            let tip = match &llm.dictate.err {
                Some(e) => format!("Dictate — last attempt failed: {e}"),
                None if !crate::dictate::installed() => {
                    "Dictate — downloads the speech model on first use (~148 MB)".to_string()
                }
                None => "Dictate — speech to text".to_string(),
            };
            if icon_button(ui, mic, 18.0, &tip, true).clicked() {
                llm.dictate.toggle(ui.ctx());
            }
        }
    }
}

/// Thinking-slider colours: blue at the low end blending into purple at Max.
const THINK_BLUE: egui::Color32 = egui::Color32::from_rgb(56, 132, 255);
const THINK_PURPLE: egui::Color32 = egui::Color32::from_rgb(172, 82, 250);

/// Blue → purple at `t` (0..=1).
fn think_color(t: f32) -> egui::Color32 {
    let l = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    egui::Color32::from_rgb(
        l(THINK_BLUE.r(), THINK_PURPLE.r()),
        l(THINK_BLUE.g(), THINK_PURPLE.g()),
        l(THINK_BLUE.b(), THINK_PURPLE.b()),
    )
}

/// Cubes in the pile per level (Off … Max): a lone cube for Low, a full row
/// for Medium, a pyramid for Max.
const THINK_CUBES: [usize; 5] = [0, 1, 3, 5, 6];
/// Pile slots in grid units (x across, row up), in draw order — lower rows
/// first so upper cubes overlap the ones they sit on.
const THINK_SLOTS: [(f32, f32); 6] =
    [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-0.5, 1.0), (0.5, 1.0), (0.0, 2.0)];

/// One isometric cube with its top-face centre-bottom at `c`, edge `a`:
/// lit top, mid left face, shaded right face.
fn paint_cube(painter: &egui::Painter, c: egui::Pos2, a: f32, col: egui::Color32, alpha: f32) {
    let w = a * 0.87;
    let p = |x: f32, y: f32| egui::pos2(c.x + x, c.y + y);
    let tint = |col: egui::Color32| col.gamma_multiply(alpha);
    let top = col.lerp_to_gamma(egui::Color32::WHITE, 0.35);
    let right = col.lerp_to_gamma(egui::Color32::BLACK, 0.35);
    let edge = egui::Stroke::new(0.8_f32, tint(col.lerp_to_gamma(egui::Color32::WHITE, 0.55)));
    painter.add(egui::Shape::convex_polygon(
        vec![p(0.0, -a), p(w, -a / 2.0), p(0.0, 0.0), p(-w, -a / 2.0)],
        tint(top),
        edge,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![p(-w, -a / 2.0), p(0.0, 0.0), p(0.0, a), p(-w, a / 2.0)],
        tint(col),
        edge,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![p(0.0, 0.0), p(w, -a / 2.0), p(w, a / 2.0), p(0.0, a)],
        tint(right),
        edge,
    ));
}

/// The model picker's thinking slider: a pile of cubes that grows with the
/// level (glowing at Max), the level's name with its allowance of thinking
/// tokens in a chip ("4,096 tok"), and a 5-stop track whose fill blends
/// blue → purple, named under each stop. Click or drag anywhere on the
/// track — or click a stop's name — to set it.
fn thinking_slider(ui: &mut egui::Ui, level: &mut crate::llm::ThinkLevel) -> egui::Response {
    use crate::llm::ThinkLevel;
    let enabled = ui.is_enabled();
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 94.0), egui::Sense::hover());
    let track = egui::Rect::from_min_max(
        egui::pos2(rect.left() + 8.0, rect.bottom() - 38.0),
        egui::pos2(rect.right() - 8.0, rect.bottom() - 16.0),
    );
    // The track plus the stop names under it.
    let hit = egui::Rect::from_min_max(
        egui::pos2(track.left() - 4.0, track.top() - 8.0),
        egui::pos2(track.right() + 4.0, rect.bottom()),
    );
    let resp = ui
        .interact(hit, ui.id().with("think_track"), egui::Sense::click_and_drag())
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    if let Some(pos) = resp.interact_pointer_pos()
        && (resp.dragged() || resp.clicked() || resp.is_pointer_button_down_on())
    {
        let t = ((pos.x - track.left()) / track.width()).clamp(0.0, 1.0);
        *level = ThinkLevel::ALL[(t * 4.0).round() as usize];
    }

    let idx = level.index();
    // Glide the colour and knob between stops instead of snapping.
    let t = ui.ctx().animate_value_with_time(ui.id().with("think_t"), idx as f32 / 4.0, 0.18);
    let dim = if enabled { 1.0 } else { 0.4 };
    let col = think_color(t).gamma_multiply(dim);
    let painter = ui.painter_at(rect.expand(6.0));
    let time = ui.input(|i| i.time);
    let is_max = *level == ThinkLevel::Max && enabled;

    // --- The cube pile, left of the level name. ---
    let a = 9.0;
    let base = egui::pos2(rect.left() + 38.0, rect.top() + 42.0);
    if is_max {
        // A soft purple glow that breathes while Max is selected (radius 32
        // keeps it inside the painter's clip — no flat top).
        let pulse = 0.65 + 0.35 * ((time * 2.4).sin() as f32 * 0.5 + 0.5);
        soft_glow(&painter, base + egui::vec2(0.0, -14.0), 32.0, THINK_PURPLE.gamma_multiply(0.42 * pulse));
        // ~30 fps is plenty for a slow pulse (same cap as the heart badges).
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
    }
    let count = THINK_CUBES[idx];
    if count == 0 {
        // Off: a faint outline where the pile would be.
        let c = base;
        let w = a * 0.87;
        let ghost = egui::Stroke::new(1.0_f32, MUTED().gamma_multiply(0.6 * dim));
        painter.add(egui::Shape::closed_line(
            vec![
                c + egui::vec2(0.0, -a), c + egui::vec2(w, -a / 2.0), c + egui::vec2(w, a / 2.0),
                c + egui::vec2(0.0, a), c + egui::vec2(-w, a / 2.0), c + egui::vec2(-w, -a / 2.0),
            ],
            ghost,
        ));
    }
    for (i, (sx, row)) in THINK_SLOTS.iter().enumerate() {
        // Each cube pops in/out as the level changes.
        let alpha = ui.ctx().animate_bool_with_time(
            ui.id().with(("think_cube", i)),
            i < count,
            0.15,
        );
        if alpha <= 0.0 {
            continue;
        }
        let c = base + egui::vec2(sx * a * 1.74, -row * a * 1.5 + (1.0 - alpha) * 4.0);
        // Cubes higher in the pile lean further toward purple.
        let cube_col = think_color((t + row * 0.12).min(1.0)).gamma_multiply(dim);
        paint_cube(&painter, c, a, cube_col, alpha);
    }

    // --- Level name, its thinking-token allowance, a short caption. ---
    let text_x = rect.left() + 82.0;
    painter.text(
        egui::pos2(text_x, rect.top() + 14.0),
        egui::Align2::LEFT_CENTER,
        level.label(),
        egui::FontId::proportional(15.0),
        if enabled { picker_ink(think_color(t), 0.2) } else { MUTED() },
    );
    // The real allowance (ThinkLevel::budget), so the chip can't drift
    // from what the model is given.
    let chip_ink = picker_ink(think_color(t), 0.45).gamma_multiply(dim);
    let chip_text = painter.layout_no_wrap(
        format!("{} tok", thousands(level.budget())),
        egui::FontId::monospace(9.5),
        chip_ink,
    );
    let chip = egui::Rect::from_min_size(
        egui::pos2(rect.right() - chip_text.size().x - 18.0, rect.top() + 5.0),
        egui::vec2(chip_text.size().x + 14.0, 18.0),
    );
    painter.rect_filled(chip, egui::CornerRadius::same(9), col.gamma_multiply(0.14));
    painter.rect_stroke(
        chip,
        egui::CornerRadius::same(9),
        egui::Stroke::new(1.0_f32, col.gamma_multiply(0.32)),
        egui::StrokeKind::Inside,
    );
    painter.galley(chip.center() - chip_text.size() / 2.0, chip_text, chip_ink);
    painter.text(
        egui::pos2(text_x, rect.top() + 32.0),
        egui::Align2::LEFT_CENTER,
        if *level == ThinkLevel::Off { "no thinking" } else { "thinking before it answers" },
        egui::FontId::proportional(11.0),
        MUTED().gamma_multiply(dim),
    );

    // --- The track: rail, gradient fill to the knob, stop dots and names. ---
    let cy = track.center().y;
    let (x0, x1) = (track.left(), track.right());
    let kx = x0 + (x1 - x0) * t;
    painter.line_segment(
        [egui::pos2(x0, cy), egui::pos2(x1, cy)],
        egui::Stroke::new(5.5_f32, TEXT().gamma_multiply(0.12 * dim)),
    );
    if kx > x0 + 0.5 {
        let mut mesh = egui::Mesh::default();
        let (h, left_col) = (2.75, THINK_BLUE.gamma_multiply(dim));
        mesh.colored_vertex(egui::pos2(x0, cy - h), left_col);
        mesh.colored_vertex(egui::pos2(kx, cy - h), col);
        mesh.colored_vertex(egui::pos2(kx, cy + h), col);
        mesh.colored_vertex(egui::pos2(x0, cy + h), left_col);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(mesh);
        painter.circle_filled(egui::pos2(x0, cy), h, left_col);
    }
    let names = ["Off", "Low", "Med", "High", "Max"];
    for (i, name) in names.into_iter().enumerate() {
        let sx = x0 + (x1 - x0) * i as f32 / 4.0;
        let on = (i as f32 / 4.0) <= t + 0.001;
        painter.circle_filled(
            egui::pos2(sx, cy),
            2.0,
            if on { egui::Color32::WHITE.gamma_multiply(0.85 * dim) } else { MUTED().gamma_multiply(0.65 * dim) },
        );
        // The end names sit flush with the track's ends.
        let (x, align) = match i {
            0 => (x0 - 2.0, egui::Align2::LEFT_CENTER),
            4 => (x1 + 2.0, egui::Align2::RIGHT_CENTER),
            _ => (sx, egui::Align2::CENTER_CENTER),
        };
        let ink = if i == idx {
            picker_ink(think_color(t), 0.3).gamma_multiply(dim)
        } else {
            MUTED().gamma_multiply(0.75 * dim)
        };
        painter.text(egui::pos2(x, rect.bottom() - 6.0), align, name, egui::FontId::proportional(9.5), ink);
    }
    if is_max {
        let pulse = 0.6 + 0.4 * ((time * 2.4).sin() as f32 * 0.5 + 0.5);
        painter.circle_filled(egui::pos2(kx, cy), 12.5, THINK_PURPLE.gamma_multiply(0.25 * pulse));
    }
    painter.circle(
        egui::pos2(kx, cy),
        if resp.hovered() || resp.dragged() { 8.5 } else { 7.5 },
        col,
        egui::Stroke::new(2.0_f32, egui::Color32::WHITE.gamma_multiply(0.92 * dim)),
    );
    // Sparkles: four at Max, two at High. They animate, so keep the
    // slider redrawing (~30 fps) while they show.
    let sparkles = match *level {
        ThinkLevel::Max => 4,
        ThinkLevel::High => 2,
        _ => 0,
    };
    if sparkles > 0 && enabled {
        knob_sparkles(ui.painter(), egui::pos2(kx, cy), time, sparkles);
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
    }
    resp
}

/// 4096 → "4,096".
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Little four-point stars twinkling slowly around the knob (`count` of
/// them: 4 at Max, 2 at High), each on its own phase so they never flash
/// together.
fn knob_sparkles(painter: &egui::Painter, c: egui::Pos2, time: f64, count: usize) {
    let t = time as f32;
    // (angle around the knob, distance, phase, peak size)
    const SPARKS: [(f32, f32, f32, f32); 4] =
        [(-2.2, 13.0, 0.0, 4.2), (-0.6, 14.0, 1.7, 3.4), (0.9, 12.0, 3.3, 3.8), (2.6, 13.5, 4.8, 3.0)];
    let tint = egui::Color32::from_rgb(236, 222, 255); // purple-white
    // High keeps the two that sit apart (first and third), so it reads as
    // a lighter version of Max.
    let picked: &[usize] = if count >= 4 { &[0, 1, 2, 3] } else { &[0, 2] };
    for &(angle, dist, phase, size) in picked.iter().map(|&i| &SPARKS[i]) {
        // Mostly dark, briefly bright: a slow ~4 s twinkle per sparkle.
        let s = ((t * 1.5 + phase).sin() * 0.5 + 0.5).powi(4);
        if s < 0.02 {
            continue;
        }
        // A gentle drift around the knob so they don't sit perfectly still.
        let a = angle + t * 0.12;
        let p = c + egui::vec2(a.cos(), a.sin()) * dist;
        let r = size * (0.55 + 0.45 * s);
        let w = r * 0.24;
        let col = tint.gamma_multiply(s);
        // A four-point star as two thin diamonds (each convex).
        for (dx, dy) in [(0.0, 1.0), (1.0, 0.0)] {
            let (ax, ay) = (dx * r, dy * r);
            let (bx, by) = (dy * w, dx * w);
            painter.add(egui::Shape::convex_polygon(
                vec![
                    p + egui::vec2(-ax, -ay),
                    p + egui::vec2(bx, by),
                    p + egui::vec2(ax, ay),
                    p + egui::vec2(-bx, -by),
                ],
                col,
                egui::Stroke::NONE,
            ));
        }
        painter.circle_filled(p, w * 0.9, egui::Color32::WHITE.gamma_multiply(s));
    }
}

/// A radial glow: `col` at the centre, half strength at 45% of the radius,
/// fully clear at the rim, so there are no hard ring edges.
fn soft_glow(painter: &egui::Painter, c: egui::Pos2, r: f32, col: egui::Color32) {
    let n = 48u32;
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(c, col);
    let mid = col.gamma_multiply(0.5);
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let d = egui::vec2(a.cos(), a.sin());
        mesh.colored_vertex(c + d * (r * 0.45), mid);
        mesh.colored_vertex(c + d * r, egui::Color32::TRANSPARENT);
    }
    for i in 0..n {
        let (m0, o0) = (1 + 2 * i, 2 + 2 * i);
        let (m1, o1) = (1 + 2 * ((i + 1) % n), 2 + 2 * ((i + 1) % n));
        mesh.add_triangle(0, m0, m1);
        mesh.add_triangle(m0, o0, o1);
        mesh.add_triangle(m0, o1, m1);
    }
    painter.add(mesh);
}

fn icon_button_tinted(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'_>,
    icon_size: f32,
    tip: &str,
    enabled: bool,
    tint: egui::Color32,
) -> egui::Response {
    // A disabled button only senses hover (for its tooltip) — it must not
    // report clicks while it looks greyed out.
    let sense = if enabled { egui::Sense::click() } else { egui::Sense::hover() };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(28.0, 28.0), sense);
    let resp = resp.on_hover_text(tip);
    let resp = if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp };
    if ui.is_rect_visible(rect) {
        let tint = if enabled { tint } else { tint.gamma_multiply(0.45) };
        egui::Image::new(icon)
            .tint(tint)
            .paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(icon_size, icon_size)));
    }
    resp
}
#[cfg(test)]
mod tests {
    /// The Thinking row names a role-play character's thoughts as its own
    /// ("Mira's thoughts"); other replies keep the plain labels.
    #[test]
    fn thoughts_are_labelled_with_the_character() {
        assert_eq!(thoughts_label(Some("Mira"), true, Some(12)), "Mira is thinking… 12 s");
        assert_eq!(thoughts_label(Some("Mira"), true, None), "Mira is thinking…");
        assert_eq!(thoughts_label(Some("Mira"), false, Some(3)), "Mira's thoughts");
        assert_eq!(thoughts_label(Some("the AI"), false, None), "The AI's thoughts");
        assert_eq!(thoughts_label(None, true, Some(4)), "Thinking… 4 s");
        assert_eq!(thoughts_label(None, true, None), "Thinking…");
        assert_eq!(thoughts_label(None, false, None), "Thoughts");
    }

    use super::thoughts_label;
    use super::thousands;

    /// The thinking chip's allowance reads "4,096 tok", not "4096 tok".
    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(256), "256");
        assert_eq!(thousands(1024), "1,024");
        assert_eq!(thousands(4096), "4,096");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    use eframe::egui;

    /// A table wider than the reply column scrolls sideways in a strip of
    /// its own: the reply keeps to the column, and the text after the table
    /// wraps there. (The table widened the column, so what followed ran on
    /// in one line, off the window.) A table that fits is drawn straight
    /// into the reply, as it always was.
    #[test]
    fn a_wide_table_keeps_the_reply_in_its_column() {
        use crate::slider::rig::Rig;
        struct Reply {
            text: String,
            cache: egui_commonmark::CommonMarkCache,
            used: egui::Rect,
        }
        fn scene(ui: &mut egui::Ui, s: &mut Reply) {
            ui.set_width(400.0);
            super::chat_markdown(ui, &mut s.cache, &s.text, false);
            s.used = ui.min_rect();
        }
        // How wide the part of the screen is that the table's first cell may draw in.
        fn clip_width(rig: &Rig<Reply>) -> f32 {
            let cell = rig.shapes.iter().find(|c| matches!(&c.shape, egui::Shape::Text(t) if t.galley.text() == "Date"));
            cell.expect("the table's first cell").clip_rect.width()
        }
        let after = "After the table the reply goes on with a paragraph of ordinary text that is long enough to \
                     need a few lines of the column.";
        let small = format!("| Date | Paid in |\n|---|---|\n| 2026-11-11 | 1,234.56 |\n\n{after}");
        let mut rig = Rig::new(Reply { text: small, cache: Default::default(), used: egui::Rect::NOTHING }, scene);
        rig.idle();
        let screen = clip_width(&rig);
        assert!(screen > 600.0, "a table that fits is in no strip: its cells draw in {screen}");

        let row = "| 2026-11-11 | A fairly long description of the item in this row | 1,234.56 | 7,890.12 | \
                   a note about it | and one more |\n";
        rig.state.text = format!(
            "| Date | Description | Paid in | Paid out | Notes | More |\n|---|---|---|---|---|---|\n{}\n{after}",
            row.repeat(3)
        );
        // Frame by frame the reply never spills, from the first one the
        // table is too wide in; then the table is in its strip.
        for _ in 0..4 {
            rig.frame(vec![]);
            assert!(rig.state.used.width() <= 400.5, "the reply is {} wide", rig.state.used.width());
        }
        let used = rig.state.used;
        assert!(clip_width(&rig) < 410.0, "the table's cells draw in {}", clip_width(&rig));
        let para = rig.texts(after);
        assert_eq!(para.len(), 1);
        assert!(para[0].right() <= used.left() + 400.5, "{:?} runs out of {used:?}", para[0]);
        assert!(para[0].height() > 30.0, "{:?} is on one line", para[0]);
    }

    /// An attachment's name is cut short where its row ends, less the room
    /// asked for after it (a draft's ✕). (A long file name stretched the
    /// message box past the window.)
    #[test]
    fn a_long_attachment_name_keeps_to_its_row() {
        use crate::slider::rig::Rig;
        struct Chip {
            name: String,
            reserve: f32,
            used: egui::Rect,
        }
        fn scene(ui: &mut egui::Ui, s: &mut Chip) {
            ui.set_width(300.0);
            super::media_chip(ui, std::path::Path::new(&s.name), egui::Color32::GRAY, s.reserve);
            s.used = ui.min_rect();
        }
        let name = format!("{}.mp4", "a_clip_with_a_very_long_file_name_".repeat(6));
        let mut rig = Rig::new(Chip { name: name.clone(), reserve: 0.0, used: egui::Rect::NOTHING }, scene);
        for reserve in [0.0, 30.0] {
            rig.state.reserve = reserve;
            rig.frame(vec![]);
            let (used, shown) = (rig.state.used, rig.texts(&name)[0]);
            assert!(used.width() <= 300.5, "the row grew to {}", used.width());
            assert!(shown.right() <= used.left() + 300.5 - reserve, "{shown:?} with {reserve} kept free");
        }
    }

    // --- What a reply's markdown makes of what models write. ---

    struct Reply {
        text: String,
        cache: egui_commonmark::CommonMarkCache,
    }

    /// `text` as a reply, in a 420-wide column.
    fn reply(text: &str) -> crate::slider::rig::Rig<Reply> {
        fn scene(ui: &mut egui::Ui, s: &mut Reply) {
            ui.set_width(420.0);
            super::chat_markdown(ui, &mut s.cache, &s.text, false);
        }
        let mut rig = crate::slider::rig::Rig::new(Reply { text: text.to_string(), cache: Default::default() }, scene);
        rig.screen = egui::vec2(640.0, 900.0);
        rig.idle();
        rig
    }

    /// Each run of text the last frame painted: what it says, where, and how
    /// its first part is set.
    fn painted<S>(rig: &crate::slider::rig::Rig<S>) -> Vec<(String, egui::Rect, egui::TextFormat)> {
        fn find(shape: &egui::Shape, found: &mut Vec<(String, egui::Rect, egui::TextFormat)>) {
            match shape {
                egui::Shape::Text(t) if !t.galley.text().trim().is_empty() => {
                    let format = t.galley.job.sections.first().map(|s| s.format.clone()).unwrap_or_default();
                    found.push((t.galley.text().to_string(), t.visual_bounding_rect(), format));
                }
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, found)),
                _ => {}
            }
        }
        let mut found = vec![];
        rig.shapes.iter().for_each(|c| find(&c.shape, &mut found));
        found
    }

    /// The one run of text that says `text`.
    fn run<S>(rig: &crate::slider::rig::Rig<S>, text: &str) -> (egui::Rect, egui::TextFormat) {
        let found: Vec<_> = painted(rig).into_iter().filter(|(said, _, _)| said == text).collect();
        assert_eq!(found.len(), 1, "{text:?} is painted {} times in {:?}", found.len(), words(rig));
        (found[0].1, found[0].2.clone())
    }

    /// Everything painted, end to end.
    fn words<S>(rig: &crate::slider::rig::Rig<S>) -> String {
        painted(rig).into_iter().map(|(said, _, _)| said).collect()
    }

    /// Prices keep their dollar signs, and maths reads as maths however it's
    /// marked. ("$5-$10" was shown "5-10"; `\( x^2 \)` as "( x^2 )";
    /// `\frac{a}{b}` as "\fracab".)
    #[test]
    fn maths_in_a_reply_reads_as_maths() {
        let rig = reply("It costs $5-$10 per month.");
        assert_eq!(words(&rig), "It costs $5-$10 per month.");
        let rig = reply(r"Inline \( x^2 \), then $\frac{a}{b}$ and $90^\circ$.");
        for maths in ["x²", "a/b", "90°"] {
            assert!(run(&rig, maths).1.italics, "{maths} is set as maths");
        }
        assert!(!words(&rig).contains(['\\', '(', '$']), "{}", words(&rig));
        let rig = reply("So:\n\n\\[ E = mc^2 \\]");
        let (formula, _) = run(&rig, "E = mc²");
        assert!((formula.center().x - 210.0).abs() < 2.0, "{formula:?} is centred in the column");
        // What the fonts here haven't got is written another way (no box).
        let rig = reply(r"$y_i$");
        let has = rig.ctx.fonts_mut(|f| f.has_glyph(&egui::FontId::proportional(14.5), 'ᵢ'));
        assert_eq!(words(&rig), if has { "yᵢ" } else { "y_i" });
    }

    /// The HTML tags models write in their markdown do what they say, where
    /// they were shown as text: `<br>` (in a table cell too), `<sub>`,
    /// `<sup>`, `<u>`… A tag with no other half is still text.
    #[test]
    fn html_tags_in_a_reply_do_what_they_say() {
        let rig = reply("Water is H<sub>2</sub>O, and E = mc<sup>3</sup>.");
        assert_eq!(words(&rig), "Water is H2O, and E = mc3.");
        let (body, (lowered, low), (raised, high)) = (run(&rig, "Water is H").0, run(&rig, "2"), run(&rig, "3"));
        assert!(low.font_id.size < 12.0 && high.font_id.size < 12.0, "small: {} {}", low.font_id.size, high.font_id.size);
        assert!(raised.top() < lowered.top(), "{raised:?} is over {lowered:?}");
        assert!(lowered.bottom() <= body.bottom() + 1.0 && raised.top() >= body.top() - 3.0);

        let rig = reply("one<br>two<br/>three<BR>four");
        let tops: Vec<f32> = ["one", "two", "three", "four"].iter().map(|line| run(&rig, line).0.top()).collect();
        assert!(tops.windows(2).all(|pair| pair[1] > pair[0] + 10.0), "{tops:?}");

        let rig = reply("| Plan | Notes |\n|---|---|\n| Basic | one<br>two |\n| Pro | three |");
        let (one, two, three) = (run(&rig, "one").0, run(&rig, "two").0, run(&rig, "three").0);
        assert!(two.top() > one.bottom() - 2.0 && three.top() > two.bottom(), "{one:?} {two:?} {three:?}");
        // (To a few pixels: letters start a little in from where they're put.)
        assert!((one.left() - two.left()).abs() < 2.5 && (one.left() - three.left()).abs() < 2.5, "{one:?} {two:?} {three:?}");
        assert!(!words(&rig).contains('<'), "{}", words(&rig));

        let rig = reply("An <i>italic</i>, <u>under</u>, <s>gone</s>, <kbd>Ctrl</kbd>, <b>bold</b>.");
        assert_eq!(words(&rig), "An italic, under, gone, Ctrl, bold.");
        assert!(run(&rig, "italic").1.italics);
        assert!(run(&rig, "under").1.underline != egui::Stroke::NONE);
        assert!(run(&rig, "gone").1.strikethrough != egui::Stroke::NONE);
        assert_eq!(run(&rig, "Ctrl").1.font_id.family, egui::FontFamily::Monospace);
        let between: Vec<_> = painted(&rig).into_iter().filter(|(said, _, _)| said == ", ").collect();
        assert_eq!(between.len(), 4);
        for (_, _, format) in between {
            assert!(!format.italics && format.underline == egui::Stroke::NONE && format.strikethrough == egui::Stroke::NONE);
            assert_eq!(format.font_id.family, egui::FontFamily::Proportional, "each is off again after it");
        }

        // A tag on its own — in a sentence about HTML, or never closed in
        // its paragraph — is text, and styles nothing after it.
        let rig = reply("The <b> tag is bold, <i>this never closes.\n\nA new paragraph.");
        assert_eq!(words(&rig), "The <b> tag is bold, <i>this never closes.A new paragraph.");
        assert!(!run(&rig, "A new paragraph.").1.italics);
        // One this doesn't know is as written.
        let rig = reply("Replace <your-name> here, <span>as is</span>.");
        assert_eq!(words(&rig), "Replace <your-name> here, <span>as is</span>.");
    }

    /// A line break in a reply is one on screen (a poem, an address) — in a
    /// list item starting under the item's text.
    #[test]
    fn line_breaks_in_a_reply_are_kept() {
        let rig = reply("Roses are red,\nViolets are blue,\nSugar is sweet.");
        let lines: Vec<egui::Rect> =
            ["Roses are red,", "Violets are blue,", "Sugar is sweet."].iter().map(|line| run(&rig, line).0).collect();
        for pair in lines.windows(2) {
            assert!(pair[1].top() > pair[0].bottom() - 2.0, "{pair:?}");
            assert!((pair[1].left() - pair[0].left()).abs() < 2.5, "{pair:?}");
        }
        let rig = reply("- first line\n  second line\n- next\n\n> quoted\n> again");
        let (first, second, next) = (run(&rig, "first line").0, run(&rig, "second line").0, run(&rig, "next").0);
        assert!(second.top() > first.bottom() - 2.0 && next.top() > second.bottom() - 2.0);
        assert!((second.left() - first.left()).abs() < 2.5, "{second:?} starts under {first:?}");
        assert!(run(&rig, "again").0.top() > run(&rig, "quoted").0.bottom() - 2.0);
        // A paragraph's words still wrap where the column ends.
        let long = "A paragraph of ordinary text, with no line breaks of its own. ".repeat(6);
        let (wrapped, _) = run(&reply(&long), long.trim_end());
        assert!(wrapped.height() > 30.0 && wrapped.right() < 640.0, "{wrapped:?}");
    }

    /// What a list item holds besides its first line is under that line: a
    /// step's code block (it sat to the right of the step's text), its
    /// second paragraph (it ran straight on from the first), and what comes
    /// after either.
    #[test]
    fn a_list_items_blocks_are_under_its_text() {
        let rig = reply("1. Install it:\n   ```sh\n   pip install x\n   ```\n   and then more\n2. Run it.");
        let (step, code, more, next) = (
            run(&rig, "Install it:").0,
            run(&rig, "pip install x").0,
            run(&rig, "and then more").0,
            run(&rig, "Run it.").0,
        );
        assert!(code.top() > step.bottom() && more.top() > code.bottom() && next.top() > more.bottom(), "{step:?} {code:?} {more:?}");
        assert!(code.left() < step.left() + 8.0 && (more.left() - step.left()).abs() < 2.5, "{step:?} {code:?} {more:?}");

        let rig = reply("- one\n\n  second paragraph\n\n  ```\n  code\n  ```\n\n  third paragraph\n\n- two");
        let tops: Vec<egui::Rect> =
            ["one", "second paragraph", "code", "third paragraph", "two"].iter().map(|said| run(&rig, said).0).collect();
        for pair in tops.windows(2) {
            assert!(pair[1].top() > pair[0].bottom() - 2.0, "{pair:?}");
        }
        assert!((tops[1].left() - tops[0].left()).abs() < 2.5 && (tops[3].left() - tops[0].left()).abs() < 2.5, "{tops:?}");

        // After a list inside it, too.
        let rig = reply("- a\n\n  - b\n\n  after b\n- c");
        let (b, after, c) = (run(&rig, "b").0, run(&rig, "after b").0, run(&rig, "c").0);
        assert!(after.top() > b.bottom() && c.top() > after.bottom(), "{b:?} {after:?} {c:?}");

        // An item that is only its text, a task or a code block is as it was.
        let rig = reply("- plain `code` [link](https://example.com) end\n- [ ] task\n- ```\n  only\n  ```");
        let (plain, end) = (run(&rig, "plain ").0, run(&rig, " end").0);
        assert!((plain.center().y - end.center().y).abs() < 4.0, "{plain:?} {end:?}");
        assert!(run(&rig, "task").0.top() > plain.bottom() - 2.0 && run(&rig, "only").0.top() > run(&rig, "task").0.bottom() - 2.0);
        let gap = run(&rig, "only").0.top() - run(&rig, "task").0.bottom();
        assert!(gap < 14.0, "no empty line over the code: {gap}");
    }

    /// Only web links open. (What opens a clicked link hands anything else —
    /// a `file:` link, a path — to the system.) The rest are their text.
    #[test]
    fn only_web_links_in_a_reply_open() {
        let mut rig = reply(
            "[web](https://example.com/a)\n\n[file](file:///etc/passwd)\n\n[path](C:\\Windows\\notepad.exe)\n\n\
             [bare](www.example.com/b)\n\n[page](notes.html)\n\n[mail](mailto:a@example.com)\n\n[script](javascript:alert(1))\n\n\
             <https://example.com/c>",
        );
        for link in ["web", "file", "path", "bare", "page", "mail", "script", "https://example.com/c"] {
            let at = run(&rig, link).0.center();
            rig.click(at);
            rig.idle();
        }
        assert_eq!(
            rig.opened,
            ["https://example.com/a", "https://www.example.com/b", "mailto:a@example.com", "https://example.com/c"]
        );
        // What opens looks like a link; what doesn't, like the words round it.
        let link = run(&rig, "web").1.color;
        assert_eq!(run(&rig, "mail").1.color, link);
        for text in ["file", "path", "page", "script"] {
            assert_ne!(run(&rig, text).1.color, link, "{text}");
        }
    }

    /// A picture that can't be shown says what it is, where it was a lone ⚠:
    /// one on the web is a link to it (nothing is fetched).
    #[test]
    fn a_picture_in_a_reply_says_what_it_is() {
        let mut rig = reply("Here: ![a cat](https://example.com/cat.png)\n\n![](https://example.com/x.png)\n\n![chart](file:///nowhere/chart.png)");
        // (The file is looked for off this thread.)
        for _ in 0..500 {
            if words(&rig).ends_with("chart") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            rig.frame(vec![]);
        }
        assert_eq!(words(&rig), "Here: a cathttps://example.com/x.pngchart");
        for text in ["a cat", "https://example.com/x.png", "chart"] {
            let at = run(&rig, text).0.center();
            rig.click(at);
            rig.idle();
        }
        assert_eq!(rig.opened, ["https://example.com/cat.png", "https://example.com/x.png"]);
    }

    /// The colour of the code that says `word`, in the block painted.
    fn code_color(rig: &crate::slider::rig::Rig<Reply>, word: &str) -> egui::Color32 {
        fn find(shape: &egui::Shape, word: &str, found: &mut Vec<egui::Color32>) {
            match shape {
                egui::Shape::Text(t) => {
                    if let Some(at) = t.galley.text().find(word) {
                        let section = t.galley.job.sections.iter().find(|s| s.byte_range.contains(&at));
                        found.extend(section.map(|s| s.format.color));
                    }
                }
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, word, found)),
                _ => {}
            }
        }
        let mut found = vec![];
        rig.shapes.iter().for_each(|c| find(&c.shape, word, &mut found));
        assert_eq!(found.len(), 1, "{word:?} in the code");
        found[0]
    }

    /// Code is coloured in the languages replies are written in, however the
    /// block names them. (Only C, Python, Rust and TOML were, in lower case.)
    #[test]
    fn code_in_a_reply_is_coloured() {
        let rig = reply("```JavaScript\n// a note\nconst total = \"a // b\"; /* aside */ let next = 5;\n```");
        let (keyword, name, text, comment) =
            (code_color(&rig, "const"), code_color(&rig, "total"), code_color(&rig, "a // b"), code_color(&rig, "a note"));
        assert_eq!(code_color(&rig, "let"), keyword);
        assert_eq!(code_color(&rig, "next"), name);
        assert_eq!(code_color(&rig, "aside"), comment);
        let all = [keyword, name, text, comment];
        for (i, a) in all.iter().enumerate() {
            assert!(all[i + 1..].iter().all(|b| a != b), "{all:?}");
        }
        // The same colours whichever highlighter has the language.
        let rig = reply("```Python\n# a note\ndef total():\n    return \"text\"\n```");
        assert_eq!(
            (code_color(&rig, "def"), code_color(&rig, "total"), code_color(&rig, "text"), code_color(&rig, "a note")),
            (keyword, name, text, comment)
        );
        for (block, words, plain) in [
            ("```sql\nselect name FROM users -- all of them\n```", vec!["select", "FROM"], "users"),
            ("```bash\nif [ -f x ]; then echo $# found; fi # done\n```", vec!["if", "then", "echo", "fi"], "found"),
            ("```json\n{\"on\": true, \"n\": null}\n```", vec!["true", "null"], "n"),
            ("```ts title=\"a.ts\"\ninterface A { readonly x: number }\n```", vec!["interface", "readonly", "number"], "x"),
            ("```go\nfunc main() { defer f() }\n```", vec!["func", "defer"], "main"),
            ("```c#\npublic class A { }\n```", vec!["public", "class"], "A"),
        ] {
            let rig = reply(block);
            for word in words {
                assert_eq!(code_color(&rig, word), keyword, "{word} in {block}");
            }
            assert_ne!(code_color(&rig, plain), keyword, "{plain} in {block}");
        }
        // A comment mark inside other things isn't one.
        let rig = reply("```yaml\nurl: http://example.com/x # the site\ncount: $#\n```");
        assert_ne!(code_color(&rig, "example"), comment);
        assert_eq!(code_color(&rig, "the site"), comment);
        let rig = reply("```sql\nselect 'it''s -- not a comment' -- but this is\n```");
        assert_eq!(code_color(&rig, "not a comment"), text);
        assert_eq!(code_color(&rig, "but this is"), comment);
        // A language neither knows is plain: one colour.
        let rig = reply("```brainfuck\nconst x = \"a\" // b\n```");
        assert_eq!(code_color(&rig, "const"), code_color(&rig, "// b"));
    }

    /// A message out of sight isn't laid out again: its room is kept. One
    /// that would be in sight always is, and one whose key has changed, and
    /// one that isn't settled (the last).
    #[test]
    fn messages_out_of_sight_keep_their_room() {
        use crate::slider::rig::Rig;
        struct Chat {
            keys: Vec<u64>,
            cache: egui_commonmark::CommonMarkCache,
            drawn: Vec<usize>,
            height: f32,
        }
        fn scene(ui: &mut egui::Ui, s: &mut Chat) {
            s.drawn.clear();
            egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                ui.set_width(400.0);
                let n = s.keys.len();
                for i in 0..n {
                    let (key, cache, drawn) = (s.keys[i], &mut s.cache, &mut s.drawn);
                    super::show_or_keep_room(ui, ("test_msg", i), key, 400.0, i + 1 < n, |ui| {
                        drawn.push(i);
                        let text = format!("Message {i}\n\n- a point\n- another\n\n`code` and **bold** words.");
                        super::chat_markdown(ui, cache, &text, false);
                    });
                    ui.add_space(12.0);
                }
                s.height = ui.min_rect().height();
            });
        }
        let chat = Chat { keys: vec![7; 40], cache: Default::default(), drawn: vec![], height: 0.0 };
        let mut rig = Rig::new(chat, scene);
        assert_eq!(rig.state.drawn, (0..40).collect::<Vec<_>>(), "all of them, the first time");
        let height = rig.state.height;
        rig.frame(vec![]);
        // What's in sight (the top few), and the last.
        let top = rig.state.drawn.clone();
        assert!(top.len() < 8 && top[0] == 0 && top.last() == Some(&39), "{top:?}");
        assert!(top[..top.len() - 1].windows(2).all(|w| w[1] == w[0] + 1), "{top:?}");
        assert!((rig.state.height - height).abs() < 0.5, "{} → {}", height, rig.state.height);
        assert_eq!(rig.texts("Message 0").len(), 1);

        // Scrolled on: what's in sight there — drawn, and where it should be.
        let wheel = |rig: &mut Rig<Chat>, dy: f32| {
            let at = egui::pos2(200.0, 150.0);
            rig.frame(vec![
                egui::Event::PointerMoved(at),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, dy),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
            rig.idle();
        };
        for _ in 0..6 {
            wheel(&mut rig, -150.0);
        }
        // (Until the scrolling has come to rest.)
        for _ in 0..40 {
            let before = rig.state.drawn.clone();
            let at = rig.texts(&format!("Message {}", before[1]));
            rig.idle();
            if rig.state.drawn == before && rig.texts(&format!("Message {}", before[1])) == at {
                break;
            }
        }
        let middle = rig.state.drawn.clone();
        assert!(middle[0] > top[top.len() - 2] && middle.len() < 8, "{middle:?}");
        assert!((rig.state.height - height).abs() < 0.5);
        // (The first of them may start over the top.)
        for i in &middle[1..middle.len() - 1] {
            let shown = rig.texts(&format!("Message {i}"));
            assert_eq!(shown.len(), 1, "message {i}");
        }
        // Each is where it would be with all of them drawn.
        let at = rig.texts(&format!("Message {}", middle[1]))[0];
        rig.state.keys = vec![8; 40];
        rig.frame(vec![]);
        assert_eq!(rig.state.drawn.len(), 40, "its key changed: every one is measured again");
        assert_eq!(rig.texts(&format!("Message {}", middle[1]))[0], at);
        rig.frame(vec![]);
        assert_eq!(rig.state.drawn, middle);

        // One whose key changes out of sight is drawn once more, then left.
        rig.state.keys[0] = 9;
        rig.frame(vec![]);
        assert_eq!(rig.state.drawn[0], 0);
        rig.frame(vec![]);
        assert_eq!(rig.state.drawn, middle);
    }

    /// A Krea 2 prompt has its own button, at the bottom right corner under
    /// it — apart from the copy, retry and listen icons — which hands the
    /// prompt over for Text to Image. Other replies haven't.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_krea_prompt_can_be_taken_to_text_to_image() {
        use crate::llm::{ChatMsg, ChatRole, LlmState};
        const PROMPT: &str = "A weathered lighthouse keeper in a navy wool sweater reads a creased letter by \
                              the amber glow of a brass oil lamp, rain streaking the small window behind him.";
        fn scene(ui: &mut egui::Ui, llm: &mut LlmState) {
            ui.set_width(600.0);
            let n = llm.chats[llm.active_chat].msgs.len();
            for i in 0..n {
                super::message(ui, llm, i, false);
            }
        }
        let reply = |text: &str, krea_prompt: bool| ChatMsg {
            role: ChatRole::Model,
            text: text.to_string(),
            thinking: None,
            image: None,
            doc: None,
            tag_image: None,
            tag_change: None,
            goals: vec![],
            growth: vec![],
            did: vec![],
            character: None,
            remember: false,
            picture_memory: false,
            shared_with: vec![],
            research: None,
            krea_prompt,
        };
        let mut llm = LlmState::default();
        llm.chats[0].msgs.push(reply("Hi! Send me a picture or an idea.", false));
        llm.chats[0].msgs.push(reply(PROMPT, true));
        let mut rig = crate::slider::rig::Rig::new(llm, scene);
        rig.screen = egui::vec2(640.0, 700.0);
        rig.idle();

        let buttons = rig.texts(super::TO_IMAGE);
        assert_eq!(buttons.len(), 1, "under the prompt only");
        let button = buttons[0];
        // (The prompt wraps over several lines: one run of text.)
        let prompt = run(&rig, PROMPT).0;
        assert!(button.top() > prompt.bottom(), "{button:?} is under {prompt:?}");
        assert!(button.top() < prompt.bottom() + 24.0, "right under it: {button:?} {prompt:?}");
        assert!(button.right() > 560.0 && button.right() <= 600.0, "at the right end of the column: {button:?}");
        assert!(button.left() > 380.0, "apart from the icons on the left: {button:?}");
        assert!(rig.state.to_image.is_none());

        rig.click(button.center());
        assert_eq!(rig.state.to_image.as_deref(), Some(PROMPT));
    }

    /// Role play: what a reply changed about the character is noted under
    /// it, a quiet line each, like its goals.
    #[test]
    fn a_reply_that_changed_the_character_says_so() {
        use crate::llm::{ChatMsg, ChatRole, LlmState};
        use crate::roleplay::{GoalEvent, GoalStatus, GrowthEvent, GrowthKind};
        fn scene(ui: &mut egui::Ui, llm: &mut LlmState) {
            ui.set_width(600.0);
            super::message(ui, llm, 0, false);
        }
        let change = |kind, text: &str| GrowthEvent { kind, text: text.to_string() };
        let mut llm = LlmState::default();
        llm.chats[0].msgs.push(ChatMsg {
            role: ChatRole::Model,
            text: "I did it, Andrew!".to_string(),
            thinking: None,
            image: None,
            doc: None,
            tag_image: None,
            tag_change: None,
            goals: vec![GoalEvent { status: GoalStatus::Achieved, text: "Swim the bay".into() }],
            growth: vec![
                change(GrowthKind::TraitGained, "braver than she knew"),
                change(GrowthKind::TraitDropped, "shy"),
                change(GrowthKind::PersonaAdded, "She crossed the bay on her own."),
                change(GrowthKind::PersonaChanged, "Mira is a sailor."),
                change(GrowthKind::PersonaRemoved, "She has never left the harbour."),
            ],
            did: vec![],
            character: None,
            remember: false,
            picture_memory: false,
            shared_with: vec![],
            research: None,
            krea_prompt: false,
        });
        let mut rig = crate::slider::rig::Rig::new(llm, scene);
        rig.idle();
        let lines = [
            "I did it, Andrew!",
            "Goal achieved: Swim the bay",
            "New trait: braver than she knew",
            "Trait dropped: shy",
            "Added to its persona: She crossed the bay on her own.",
            "Persona rewritten: Mira is a sailor.",
            "Taken out of its persona: She has never left the harbour.",
        ];
        let tops: Vec<f32> = lines.iter().map(|line| run(&rig, line).0.top()).collect();
        assert!(tops.windows(2).all(|pair| pair[1] > pair[0] + 8.0), "one under the other: {tops:?}");
        // Small and quiet, so the story stays in front.
        let (reply, note) = (run(&rig, lines[0]).1, run(&rig, lines[2]).1);
        assert!(note.font_id.size < reply.font_id.size && note.color != reply.color);
    }

    /// A message's key is what its height hangs on.
    #[test]
    fn a_messages_key_follows_what_it_says() {
        use crate::llm::{ChatMsg, ChatRole};
        let msg = |text: &str| ChatMsg {
            role: ChatRole::Model,
            text: text.to_string(),
            thinking: None,
            image: None,
            doc: None,
            tag_image: None,
            tag_change: None,
            goals: vec![],
            growth: vec![],
            did: vec![],
            character: None,
            remember: false,
            picture_memory: false,
            shared_with: vec![],
            research: None,
            krea_prompt: false,
        };
        let key = |m: &ChatMsg| super::message_key(m, None, 600.0, 1.0);
        assert_eq!(key(&msg("a reply")), key(&msg("a reply")));
        assert_ne!(key(&msg("a reply")), key(&msg("a reply, longer")));
        let mut thought = msg("a reply");
        thought.thinking = Some("hm".into());
        assert_ne!(key(&thought), key(&msg("a reply")));
        let mut user = msg("a reply");
        user.role = ChatRole::User;
        assert_ne!(key(&user), key(&msg("a reply")));
        assert_ne!(super::message_key(&msg("a reply"), None, 500.0, 1.0), key(&msg("a reply")));
        assert_ne!(super::message_key(&msg("a reply"), None, 600.0, 1.25), key(&msg("a reply")));
        // Its picture loading changes its height.
        assert_ne!(super::message_key(&msg("a reply"), Some(Some([512, 300])), 600.0, 1.0), key(&msg("a reply")));
        assert_ne!(super::message_key(&msg("a reply"), Some(None), 600.0, 1.0), key(&msg("a reply")));
    }
}
