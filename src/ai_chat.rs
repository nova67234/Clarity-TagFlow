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

mod web_research;

use eframe::egui;
use egui::Margin;

use crate::llm::{ChatRole, LlmState};
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
                llm.unload();
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

    // The gear's AI Settings window and the memory map it opens.
    crate::ai_settings::window(ui.ctx(), llm, settings);
    crate::memory_map::window(ui.ctx(), llm);
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
                // The title on the left (the empty right text takes the rest).
                let mut btn = egui::Button::new(egui::RichText::new(title).color(TEXT()).size(13.0))
                    .right_text("")
                    .corner_radius(egui::CornerRadius::same(18));
                btn = if selected {
                    btn.fill(ACCENT1().gamma_multiply(0.15))
                } else {
                    btn.fill(egui::Color32::TRANSPARENT)
                };
                if ui.add_sized(egui::vec2(title_w, 30.0), btn).clicked() {
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

/// One conversation mode's row in the Tools menu: label and switch, with a
/// muted hint beneath.
fn mode_row(
    ui: &mut egui::Ui,
    llm: &mut LlmState,
    settings: &mut crate::settings::Settings,
    mode: ChatMode,
    label: &str,
    hint: &str,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(TEXT()).size(13.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut on = mode_on(llm, settings, mode);
            if crate::settings::switch(ui, &mut on).changed() {
                set_mode(llm, settings, mode, on);
            }
        });
    });
    crate::settings::hint(ui, hint);
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
                    "Ask AI anything — attach an image with +, or pick a workflow in Tools"
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
                    for i in 0..n {
                        let streaming = {
                            let msg = &llm.chats[llm.active_chat].msgs[i];
                            live_chat && i + 1 == n && msg.role == ChatRole::Model
                        };
                        message(ui, llm, i, streaming);
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
    if !dropped.is_empty() && claims_drop(llm, ui.ctx())
        && let Some(p) = dropped.into_iter().find(|p| {
        crate::is_video(p)
            || p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| DROP_EXTS.contains(&e.to_ascii_lowercase().as_str()))
    }) {
        llm.draft_image = Some(p);
        llm.run_err = None;
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
                    .stroke(egui::Stroke::new(1.5, DROP_BLUE))
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
                .stroke(egui::Stroke::new(1.0, EDGE()))
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
                                    media_chip(ui, &path, icon_tint(MUTED()));
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
                                "Attach an image or video"
                            };
                            if icon_button(ui, egui::include_image!("../icons/add.svg"), 20.0, attach_tip, !llm.running && !tagging)
                                .clicked()
                                && !tagging
                                && let Some(path) = rfd::FileDialog::new()
                                .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"])
                                .add_filter("Videos", VIDEO_PICK_EXTS)
                                .pick_file()
                            {
                                llm.draft_image = Some(path);
                                llm.run_err = None;
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
                            if llm.tools_open {
                                let mut open = llm.tools_open;
                                egui::Popup::from_response(&tools_resp)
                                    .open_bool(&mut open)
                                    // Toggling a switch shouldn't dismiss the
                                    // popup — only clicking outside it does.
                                    .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                                    .align(egui::RectAlign::TOP_START) // opens upward (card sits at the bottom)
                                    .width(280.0)
                                    .gap(8.0)
                                    .frame(crate::float_card_frame(14))
                                    .show(|ui| {
                                        // Apple-style settings card: label + macOS
                                        // switch per row, muted hint beneath, faint
                                        // separator between rows (src/settings.rs).
                                        // The header's gear opens AI Settings — on the
                                        // Role play tab while role play is on.
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("TOOLS").color(MUTED()).strong().size(11.0));
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                let tip = if llm.roleplay.enabled {
                                                    "Role play settings and memories"
                                                } else {
                                                    "AI settings"
                                                };
                                                if icon_button(ui, egui::include_image!("../icons/settings.svg"), 16.0, tip, true)
                                                    .clicked()
                                                {
                                                    if llm.roleplay.enabled {
                                                        llm.ai_settings_tab = crate::ai_settings::AiTab::Roleplay;
                                                    }
                                                    llm.gen_settings_open = true;
                                                    llm.tools_open = false;
                                                }
                                            });
                                        });
                                        ui.add_space(5.0);
                                        crate::settings::card(ui, |ui| {
                                            mode_row(
                                                ui,
                                                llm,
                                                settings,
                                                ChatMode::Roleplay,
                                                "Role playing",
                                                "The AI plays a character with its own name, \
                                                 persona and memories, kept apart from its main \
                                                 memory. Set it up with the gear above.",
                                            );
                                            crate::settings::row_sep(ui);
                                            ui.horizontal(|ui| {
                                                ui.label(egui::RichText::new("Auto-speak replies").color(TEXT()).size(13.0));
                                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                    let mut on = settings.ai_auto_speak;
                                                    if crate::settings::switch(ui, &mut on).changed() {
                                                        settings.ai_auto_speak = on;
                                                        llm.auto_speak = on;
                                                    }
                                                });
                                            });
                                            crate::settings::hint(
                                                ui,
                                                "Read every reply aloud the moment it \
                                                 finishes — click any reply's listen \
                                                 icon to stop.",
                                            );
                                            crate::settings::row_sep(ui);
                                            mode_row(
                                                ui,
                                                llm,
                                                settings,
                                                ChatMode::Krea,
                                                "Krea 2 prompts",
                                                "Turn a picture — or an idea in words — \
                                                 into a detailed prompt for Krea 2. Ask for \
                                                 changes (\"make it night\") to rewrite it. \
                                                 Reads pictures with Detailed vision.",
                                            );
                                            crate::settings::row_sep(ui);
                                            mode_row(
                                                ui,
                                                llm,
                                                settings,
                                                ChatMode::TagImages,
                                                "Tag images",
                                                "Tag the open folder with the AI: the image on \
                                                 the left, the Tag Manager on the right. The AI \
                                                 sees each image's tags and can change them, \
                                                 with Undo.",
                                            );
                                            crate::settings::row_sep(ui);
                                            ui.horizontal(|ui| {
                                                ui.label(egui::RichText::new("Detailed vision").color(TEXT()).size(13.0));
                                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                    // Forced on while the Krea 2
                                                    // writer is (it relies on it).
                                                    let forced = settings.ai_krea_prompts;
                                                    let mut on = settings.ai_detailed_vision || forced;
                                                    let resp = ui.add_enabled_ui(!forced, |ui| {
                                                        crate::settings::switch(ui, &mut on)
                                                    });
                                                    if resp.inner.changed() {
                                                        settings.ai_detailed_vision = on;
                                                        llm.detailed_vision = on;
                                                    }
                                                });
                                            });
                                            crate::settings::hint(
                                                ui,
                                                "Show the AI the newest picture whole plus \
                                                 a few close-up sections, so small details \
                                                 (eyes, hands, text, jewelry) aren't missed. \
                                                 Replies start slower. Always on with Krea 2 \
                                                 prompts.",
                                            );
                                            crate::settings::row_sep(ui);
                                            web_research::tools_row(ui, llm, settings);
                                        });
                                    });
                                // Closed by a click outside, or by the gear
                                // (which opens AI Settings instead).
                                llm.tools_open &= open;
                            }
                            let can_send = !llm.running
                                && (!llm.draft.trim().is_empty() || llm.draft_image.is_some());
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

/// One conversation entry. User messages sit right in an accent-tinted
/// bubble (corner radius 22) that hugs the text; the model's replies render
/// with no bubble at all — just modern formatted markdown (real bold, bullet
/// lists, headings, code blocks) straight on the background, like current
/// assistant UIs. A streaming reply shows a ▌ cursor.
fn message(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, streaming: bool) {
    let (role, text, stored_thinking, image, tag_image, tag_change) = {
        let m = &llm.chats[llm.active_chat].msgs[index];
        (m.role, m.text.clone(), m.thinking.clone(), m.image.clone(), m.tag_image.clone(), m.tag_change.clone())
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
                    ui.label(
                        egui::RichText::new(crate::ai_tagging::file_name(path))
                            .color(chat_text_override().unwrap_or(MUTED()))
                            .size(11.5),
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
                        media_chip(ui, path, chat_icon_tint(MUTED()));
                    }
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
                    .stroke(egui::Stroke::new(1.0, EDGE()))
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
                                    && (!llm.edit_draft.trim().is_empty() || image.is_some());
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
        shown.push('▌');
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
        // Ended (stopped, or failed) while still thinking: no answer, only
        // the thought (in the Thoughts row above) — say so rather than
        // leave a blank.
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

    // Action row under every finished reply: copy / regenerate / listen
    // (copy and listen only when there's text to copy or read).
    if !streaming {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
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
                // First OmniVoice use loads the model — show that it's coming.
                if llm.voice.loading || llm.voice.pending > 0 {
                    ui.add(egui::Spinner::new().size(14.0).color(MUTED()));
                }
                if let Some(e) = &llm.voice.last_err {
                    ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(11.0));
                }
            }
        });
    }
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
        egui_commonmark::CommonMarkViewer::new()
            .render_math_fn(Some(&render_math))
            .show(ui, cache, text);
    });
}

/// The collapsible "Thinking" row above a reply: the little AI orb, a muted
/// label (pulsing softly while the thought still streams) and a chevron that
/// rotates as the section opens. Clicking anywhere on it (orb included)
/// fades in the model's thought channel underneath — an indented quote block
/// with an orb-accent rule, rather than a boxed card.
fn thinking_row(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, thinking: &str, live: bool) {
    let chat_id = llm.chats[llm.active_chat].id;
    let open_id = egui::Id::new(("ai_thinking_open", chat_id, index));
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
        // stalled (Max can think for minutes on a partly-CPU 31B).
        let label = match llm.gen_started.filter(|_| live) {
            Some(t) => format!("Thinking… {} s", t.elapsed().as_secs()),
            None if live => "Thinking…".to_string(),
            None => "Thoughts".to_string(),
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
        let resp = resp
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("See what the model thought before answering");
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
fn media_chip(ui: &mut egui::Ui, path: &std::path::Path, tint: egui::Color32) {
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
        ui.label(egui::RichText::new(name).color(MUTED()).size(11.5));
    });
}

/// Render chat math (`$...$` / `$$...$$`) via a minimal TeX → Unicode
/// fallback — there's no real math engine, but local models mostly emit
/// simple symbols (`$\pi$`, `$r^2$`), which used to show as raw TeX.
/// Unmapped commands pass through as-is, which still beats raw `$…$`.
fn render_math(ui: &mut egui::Ui, tex: &str, inline: bool) {
    let rt = egui::RichText::new(tex_to_unicode(tex))
        .italics()
        .color(chat_text())
        .size(if inline { 14.5 } else { 15.5 });
    if inline {
        ui.label(rt);
    } else {
        // Display math gets its own centred line, roughly like real LaTeX.
        ui.vertical_centered(|ui| {
            ui.label(rt);
        });
    }
}

/// Map the TeX commands local models actually emit to Unicode. Entries whose
/// name prefixes another (`\int`/`\in`, `\infty`/`\in`, `\leq`/`\le`) are
/// ordered longest-first because replacement is sequential.
fn tex_to_unicode(tex: &str) -> String {
    const MAP: &[(&str, &str)] = &[
        ("\\varepsilon", "ε"), ("\\epsilon", "ε"), ("\\varphi", "φ"), ("\\upsilon", "υ"),
        ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
        ("\\zeta", "ζ"), ("\\eta", "η"), ("\\theta", "θ"), ("\\iota", "ι"),
        ("\\kappa", "κ"), ("\\lambda", "λ"), ("\\mu", "μ"), ("\\nu", "ν"),
        ("\\xi", "ξ"), ("\\rho", "ρ"), ("\\sigma", "σ"), ("\\tau", "τ"),
        ("\\phi", "φ"), ("\\chi", "χ"), ("\\psi", "ψ"), ("\\omega", "ω"),
        ("\\Gamma", "Γ"), ("\\Delta", "Δ"), ("\\Theta", "Θ"), ("\\Lambda", "Λ"),
        ("\\Xi", "Ξ"), ("\\Pi", "Π"), ("\\Sigma", "Σ"), ("\\Phi", "Φ"),
        ("\\Psi", "Ψ"), ("\\Omega", "Ω"),
        ("\\times", "×"), ("\\cdot", "·"), ("\\div", "÷"), ("\\pm", "±"), ("\\mp", "∓"),
        ("\\leq", "≤"), ("\\geq", "≥"), ("\\neq", "≠"), ("\\approx", "≈"),
        ("\\infty", "∞"), ("\\int", "∫"), ("\\notin", "∉"),
        ("\\sqrt", "√"), ("\\sum", "∑"), ("\\prod", "∏"), ("\\partial", "∂"),
        ("\\rightarrow", "→"), ("\\leftarrow", "←"), ("\\Rightarrow", "⇒"),
        ("\\Leftarrow", "⇐"), ("\\to", "→"),
        ("\\subset", "⊂"), ("\\cup", "∪"), ("\\cap", "∩"),
        ("\\forall", "∀"), ("\\exists", "∃"),
        ("\\degree", "°"), ("\\circ", "∘"), ("\\ldots", "…"), ("\\dots", "…"),
        ("\\left", ""), ("\\right", ""),
        // Short forms LAST — they prefix longer commands above (\le → \left,
        // \leftarrow; \in → \int, \infty), which must be consumed first.
        ("\\le", "≤"), ("\\ge", "≥"), ("\\ne", "≠"), ("\\in", "∈"),
        ("\\pi", "π"),
        ("^2", "²"), ("^3", "³"),
    ];
    let mut s = tex.to_string();
    for (from, to) in MAP {
        s = s.replace(from, to);
    }
    // Drop TeX grouping braces — for the simple math this handles, they only
    // add noise (`{area:.2f}`-style text lives in code spans, not math).
    s.retain(|c| c != '{' && c != '}');
    s.trim().to_string()
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
        return "This model answers directly — thinking applies to Gemma 4 31B and \
                Qwen3-VL 30B Thinking."
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
    let edge = egui::Stroke::new(0.8, tint(col.lerp_to_gamma(egui::Color32::WHITE, 0.55)));
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
        let ghost = egui::Stroke::new(1.0, MUTED().gamma_multiply(0.6 * dim));
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
}
