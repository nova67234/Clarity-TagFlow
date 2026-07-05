//! AI Chat — the full-window view shown instead of the three panels while
//! Settings → AI Model → "Activate AI Chat" is on (the top bar stays).
//!
//! Layout (from the design sketches): the chat list lives in a left card
//! panel built exactly like the image browser's (same `Panel::left` +
//! `card_frame`), just with conversations instead of thumbnails. The
//! conversation and the input pill sit in a centred column, Gemini-style —
//! bubbles hug their text, the tall rounded pill has a `+` (attach image,
//! add.svg) on the left and the text-to-image send icon (send.svg) on the
//! right.
//!
//! All chat state (conversations, streaming reply, draft) lives on
//! `LlmState` (src/llm.rs); this module only draws it.

use eframe::egui;
use egui::Margin;

use crate::card_frame;
use crate::llm::{ChatRole, LlmState};
use crate::theme::*;

/// Width of the chats side panel — same as the Details & Actions panel's.
const PANEL_W: f32 = crate::right_details::PANEL_WIDTH;

/// Drag-and-drop accent — the blue the input card turns while a file is
/// dragged over it (matches the generator prompt box's import blue).
const DROP_BLUE: egui::Color32 = egui::Color32::from_rgb(56, 132, 255);

/// Image types the input card accepts from a drop (matches the + picker).
const DROP_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"];

/// True when the AI Chat's input card (visible this or last frame) claims a
/// file dropped at the pointer, so the gallery doesn't also add it — the same
/// freshness-checked-rect scheme as `generate::generator_claims_drop`. A drop
/// with no known pointer position (common on Windows mid-drag) is claimed
/// whenever the card is live, since it was showing the highlight.
pub fn claims_drop(llm: &crate::llm::LlmState, ctx: &egui::Context) -> bool {
    let Some((rect, t)) = llm.input_rect else {
        return false;
    };
    if (ctx.input(|i| i.time) - t).abs() >= 0.5 {
        return false;
    }
    let pos = ctx.input(|i| i.pointer.interact_pos().or(i.pointer.latest_pos()));
    pos.map_or(true, |p| rect.contains(p))
}

pub fn show(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut crate::settings::Settings) {
    llm.poll(ui.ctx());

    // Not usable yet — say why, centred, instead of an empty chat.
    if !crate::llm::BUILT_WITH_LLM || !llm.installed {
        ui.centered_and_justified(|ui| {
            let msg = if !crate::llm::BUILT_WITH_LLM {
                "This build was compiled without the AI feature."
            } else {
                "The AI model isn't set up yet — go to Settings → AI Model \
                 and press \"Set up everything\"."
            };
            ui.label(egui::RichText::new(msg).color(MUTED()).size(14.0));
        });
        return;
    }

    // --- Left: the chat list, in the image browser's card panel style. ---
    egui::Panel::left("ai_chat_list")
        .resizable(false)
        .exact_size(PANEL_W)
        .show_separator_line(false)
        .frame(egui::Frame::new().fill(BG()).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            card_frame(22).show(ui, |ui| {
                // Fill the panel's full height, like the browser card.
                ui.set_min_height(ui.available_height());
                ui.set_width(ui.available_width());
                if llm.roleplay.enabled {
                    // Role play: chats get the top, the character + shared
                    // memory diary take the rest.
                    let h = ui.available_height();
                    egui::ScrollArea::vertical()
                        .id_salt("ai_chat_list_outer")
                        .max_height(h * 0.32)
                        .auto_shrink([false, true])
                        .show(ui, |ui| chat_list(ui, llm, settings));
                    ui.add_space(6.0);
                    ui.separator();
                    crate::roleplay::panel_ui(ui, &mut llm.roleplay);
                } else {
                    chat_list(ui, llm, settings);
                }
            });
        });

    // --- Centre: the conversation + input pill, in a centred column. ---
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(BG()).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            conversation(ui, llm, settings);
        });
}

/// The chat list card: a "Chats" header with a new-chat button and the
/// AI-settings gear (sampling knobs), then one row per conversation ("tabs
/// like switch between chats").
fn chat_list(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut crate::settings::Settings) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Chats").color(TEXT()).strong().size(14.5));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(2.0);
            if icon_button(ui, egui::include_image!("../icons/add.svg"), 18.0, "New chat", true)
                .clicked()
            {
                llm.new_chat();
            }
            let gear = icon_button(
                ui,
                egui::include_image!("../icons/settings.svg"),
                16.0,
                "AI settings (temperature, reply length…)",
                true,
            );
            if gear.clicked() {
                llm.gen_settings_open = !llm.gen_settings_open;
            }
            if llm.gen_settings_open {
                let mut open = llm.gen_settings_open;
                egui::Popup::from_response(&gear)
                    .open_bool(&mut open)
                    .align(egui::RectAlign::BOTTOM_START)
                    .width(250.0)
                    .gap(8.0)
                    .frame(crate::card_frame(14))
                    .show(|ui| gen_settings_ui(ui, settings));
                llm.gen_settings_open = open;
            }
        });
    });
    ui.add_space(8.0);

    egui::ScrollArea::vertical().id_salt("ai_chat_list").show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        let mut delete: Option<usize> = None;
        for i in 0..llm.chats.len() {
            let selected = i == llm.active_chat;
            let title = llm.chats[i].title();
            ui.horizontal(|ui| {
                let text_color = if selected { egui::Color32::WHITE } else { TEXT() };
                let mut btn = egui::Button::new(
                    egui::RichText::new(title).color(text_color).size(13.0),
                )
                .corner_radius(egui::CornerRadius::same(10))
                .min_size(egui::vec2(ui.available_width() - 26.0, 30.0));
                btn = if selected {
                    btn.fill(ACCENT1())
                } else {
                    btn.fill(egui::Color32::TRANSPARENT)
                };
                if ui.add(btn).clicked() {
                    llm.active_chat = i;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("✕").color(MUTED()).size(11.0)).frame(false))
                    .on_hover_text("Delete this chat")
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

/// The gear popup: sampling knobs for how the AI replies. Edits land in the
/// persisted settings (main.rs mirrors them onto `LlmState` each frame) and
/// apply from the next message — no model reload.
fn gen_settings_ui(ui: &mut egui::Ui, settings: &mut crate::settings::Settings) {
    let p = &mut settings.ai_gen;
    ui.label(egui::RichText::new("AI SETTINGS").color(MUTED()).strong().size(10.5));
    ui.add_space(6.0);

    let knob = |ui: &mut egui::Ui, name: &str, hint: &str, slider: egui::Slider<'_>| {
        ui.label(egui::RichText::new(name).color(TEXT()).size(12.5));
        ui.spacing_mut().slider_width = 170.0;
        ui.add(slider);
        ui.label(egui::RichText::new(hint).color(MUTED()).size(10.5));
        ui.add_space(8.0);
    };

    knob(
        ui,
        "Temperature",
        "Randomness of the replies: 0 is focused and repeatable, \
         1 is Gemma's recommended default, 2 gets wild.",
        egui::Slider::new(&mut p.temperature, 0.0..=2.0).step_by(0.05).fixed_decimals(2),
    );
    knob(
        ui,
        "Top-K",
        "Pick each word from only the K most likely candidates. \
         Lower is safer, higher is more varied. Default 64.",
        egui::Slider::new(&mut p.top_k, 1..=128),
    );
    knob(
        ui,
        "Top-P",
        "Or by probability: drop unlikely words past this cumulative \
         share. Lower is safer. Default 0.95.",
        egui::Slider::new(&mut p.top_p, 0.05..=1.0).step_by(0.01).fixed_decimals(2),
    );
    knob(
        ui,
        "Max reply length",
        "The longest answer the AI may write, in tokens (≈ ¾ of a \
         word each). Default 3072.",
        egui::Slider::new(&mut p.max_tokens, 256..=8192).step_by(64.0),
    );

    if *p != crate::llm::GenParams::default()
        && ui
            .button(egui::RichText::new("Reset to defaults").size(11.5))
            .clicked()
    {
        *p = crate::llm::GenParams::default();
    }
}

/// The conversation column: scrollable bubbles + the bottom input pill, both
/// centred on the same column (like the sketch's two guide lines).
fn conversation(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut crate::settings::Settings) {
    let avail_w = ui.available_width();
    // The centred content column both the messages and the pill live in.
    let col_w = (avail_w * 0.58).clamp(420.0, 880.0).min(avail_w - 16.0);
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
        input_h += 78.0;
    }
    if llm.run_err.is_some() {
        input_h += 22.0;
    }
    let msgs_h = (ui.available_height() - input_h).max(80.0);

    egui::ScrollArea::vertical()
        .id_salt("ai_chat_msgs")
        .auto_shrink([false, false])
        .max_height(msgs_h)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.add_space(12.0);
            let chat = &llm.chats[llm.active_chat];
            if chat.msgs.is_empty() {
                ui.add_space(msgs_h * 0.42);
                ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new("Ask Gemma anything — attach an image with +")
                            .color(MUTED())
                            .size(14.5),
                    );
                });
            }
            ui.horizontal(|ui| {
                ui.add_space(pad);
                ui.vertical(|ui| {
                    ui.set_width(col_w);
                    let n = llm.chats[llm.active_chat].msgs.len();
                    for i in 0..n {
                        let streaming = {
                            let msg = &llm.chats[llm.active_chat].msgs[i];
                            llm.running && i + 1 == n && msg.role == ChatRole::Model
                        };
                        message(ui, llm, i, streaming);
                        ui.add_space(12.0);
                    }
                    // A retry clicked inside the list is applied here, after
                    // the loop, so it can safely truncate the message list.
                    if let Some(i) = llm.pending_retry.take() {
                        llm.retry(i, ui.ctx());
                    }
                });
            });
            ui.add_space(6.0);
        });

    // --- Bottom input card ---
    // Gemini-style: attached-image thumbnail at the top-left inside the card,
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
    let dragging_files = ui.input(|i| !i.raw.hovered_files.is_empty());
    if dragging_files {
        ui.ctx().request_repaint();
    }
    let hover_pos = ui.input(|i| i.pointer.hover_pos());
    let drag_over_card = dragging_files
        && llm
            .input_rect
            .map_or(true, |(r, _)| hover_pos.map_or(true, |p| r.contains(p)));
    let dropped: Vec<std::path::PathBuf> = ui.input(|i| {
        i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()
    });
    if !dropped.is_empty() && claims_drop(llm, ui.ctx()) {
        if let Some(p) = dropped.into_iter().find(|p| {
            p.extension()
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
        egui::Frame::new()
            .fill(ui.visuals().extreme_bg_color)
            .stroke(egui::Stroke::new(1.0, EDGE()))
            .corner_radius(egui::CornerRadius::same(24))
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.vertical(|ui| {
                ui.set_width(col_w - 24.0);

                // Attached-image thumbnail (rounded, with a remove ✕).
                if llm.draft_image.is_some() {
                    let tex = llm.draft_thumb.as_ref().map(|(_, t)| t.clone());
                    if let Some(tex) = tex {
                        ui.horizontal(|ui| {
                            let size = tex.size_vec2();
                            let scale = (64.0 / size.y).min(64.0 / size.x);
                            ui.add(
                                egui::Image::new(&tex)
                                    .fit_to_exact_size(size * scale)
                                    .corner_radius(egui::CornerRadius::same(10)),
                            );
                            if ui
                                .add(egui::Button::new(egui::RichText::new("✕").color(MUTED()).size(11.0)).frame(false))
                                .on_hover_text("Remove the image")
                                .clicked()
                            {
                                llm.draft_image = None;
                                llm.draft_thumb = None;
                            }
                        });
                        ui.add_space(6.0);
                    }
                }

                // Draft text: grows with its content (the card extends upward
                // because it's bottom-anchored), scrolls inside past 8 lines.
                // Enter sends; Shift+Enter makes a newline.
                let mut send_now = false;
                egui::ScrollArea::vertical()
                    .id_salt("ai_chat_draft")
                    .max_height(8.0 * row_h)
                    .auto_shrink([false, true])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        let edit = egui::TextEdit::multiline(&mut llm.draft)
                            .desired_rows(1)
                            .frame(egui::Frame::NONE)
                            .font(egui::FontId::proportional(15.0))
                            .hint_text("Ask Gemma")
                            .desired_width(f32::INFINITY);
                        let resp = ui.add(edit);
                        let enter = resp.has_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                        if enter {
                            // Drop the newline the Enter keystroke inserted.
                            while llm.draft.ends_with('\n') {
                                llm.draft.pop();
                            }
                            send_now = true;
                        }
                    });

                // Bottom row: attach on the left, streaming status in the
                // middle, send on the right.
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if icon_button(ui, egui::include_image!("../icons/add.svg"), 20.0, "Attach an image", !llm.running)
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"])
                            .pick_file()
                        {
                            llm.draft_image = Some(path);
                        }
                    }
                    // Tools menu: extra chat abilities with on/off toggles.
                    let tools_resp = icon_button(ui, egui::include_image!("../icons/tools.svg"), 18.0, "Tools", true);
                    if tools_resp.clicked() {
                        llm.tools_open = !llm.tools_open;
                    }
                    if llm.tools_open {
                        let mut open = llm.tools_open;
                        egui::Popup::from_response(&tools_resp)
                            .open_bool(&mut open)
                            .align(egui::RectAlign::TOP_START) // opens upward (card sits at the bottom)
                            .width(240.0)
                            .gap(8.0)
                            .frame(crate::card_frame(14))
                            .show(|ui| {
                                ui.label(egui::RichText::new("TOOLS").color(MUTED()).strong().size(10.5));
                                ui.add_space(4.0);
                                let mut on = llm.roleplay.enabled;
                                let resp = ui.checkbox(&mut on, egui::RichText::new("Role playing").color(TEXT()));
                                if resp.changed() {
                                    llm.roleplay.enabled = on;
                                    llm.roleplay.save();
                                }
                                ui.label(
                                    egui::RichText::new(
                                        "Give the AI a persona and a shared memory \
                                         diary (left panel). It remembers facts, \
                                         permissions and the story as it unfolds.",
                                    )
                                    .color(MUTED())
                                    .size(10.5),
                                );
                                ui.add_space(6.0);
                                let mut speak = settings.ai_auto_speak;
                                let resp = ui.checkbox(&mut speak, egui::RichText::new("Auto-speak replies").color(TEXT()));
                                if resp.changed() {
                                    settings.ai_auto_speak = speak;
                                    llm.auto_speak = speak;
                                }
                                ui.label(
                                    egui::RichText::new(
                                        "Read every reply aloud the moment it \
                                         finishes — click any reply's listen \
                                         icon to stop.",
                                    )
                                    .color(MUTED())
                                    .size(10.5),
                                );
                            });
                        llm.tools_open = open;
                    }
                    if llm.running {
                        ui.add_space(4.0);
                        ui.add(egui::Spinner::new().size(12.0).color(MUTED()));
                        ui.label(egui::RichText::new(&llm.status).color(MUTED()).size(11.5));
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(120));
                    }
                    let can_send = !llm.running
                        && (!llm.draft.trim().is_empty() || llm.draft_image.is_some());
                    let send_clicked = ui
                        .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            icon_button(ui, egui::include_image!("../icons/send.svg"), 19.0, "Send", can_send)
                        })
                        .inner
                        .clicked();

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
    let (role, text, stored_thinking, image) = {
        let m = &llm.chats[llm.active_chat].msgs[index];
        (m.role, m.text.clone(), m.thinking.clone(), m.image.clone())
    };

    if role == ChatRole::User {
        let max_w = ui.available_width() * 0.82;
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            // The attached image shows as its own rounded thumbnail, separate
            // from (and above) the text bubble.
            if let Some(path) = &image {
                match msg_thumb(ui.ctx(), llm, path) {
                    Some(tex) => {
                        let size = tex.size_vec2();
                        let scale = (180.0 / size.y).min(280.0 / size.x).min(1.0);
                        ui.add(
                            egui::Image::new(&tex)
                                .fit_to_exact_size(size * scale)
                                .corner_radius(egui::CornerRadius::same(12)),
                        );
                    }
                    None => {
                        let name = path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        ui.label(egui::RichText::new(format!("🖼 {name}")).color(MUTED()).size(11.5));
                    }
                }
                if !text.is_empty() {
                    ui.add_space(4.0);
                }
            }
            if !text.is_empty() {
                egui::Frame::new()
                    .fill(ACCENT1().gamma_multiply(0.28))
                    .corner_radius(egui::CornerRadius::same(22))
                    .inner_margin(egui::Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        // Hug short messages: cap the bubble at the text's own
                        // measured width (the widest line for multi-line
                        // drafts), up to 82% of the column.
                        let est = ui
                            .painter()
                            .layout_no_wrap(text.clone(), egui::FontId::proportional(14.0), TEXT())
                            .size()
                            .x
                            + 12.0;
                        ui.set_max_width(est.min(max_w));
                        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                            // Emoji-aware label: color Twemoji instead of
                            // monochrome font glyphs.
                            crate::emoji::label(ui, &text, TEXT(), 14.0, false);
                        });
                    });
            }
        });
        return;
    }

    // Model reply. An album picture the AI chose to share shows above the
    // text, on its side of the column.
    if let Some(path) = &image {
        if let Some(tex) = msg_thumb(ui.ctx(), llm, path) {
            let size = tex.size_vec2();
            let scale = (180.0 / size.y).min(280.0 / size.x).min(1.0);
            ui.add(
                egui::Image::new(&tex)
                    .fit_to_exact_size(size * scale)
                    .corner_radius(egui::CornerRadius::same(12)),
            );
            ui.add_space(4.0);
        }
    }

    // The thought channel (the 26B/31B variants reason before answering):
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
            ui.add(egui::Spinner::new().size(12.0).color(MUTED()));
            ui.label(egui::RichText::new(&llm.status).color(MUTED()).size(12.5));
        });
        return;
    }
    if !thinking.is_empty() {
        thinking_row(ui, llm, index, &thinking, streaming && body.is_empty());
        ui.add_space(4.0);
    }

    let mut shown = if streaming && llm.roleplay.enabled {
        // Diary lines are extracted for real when the reply finishes; hide
        // them while it streams so they never flash up.
        crate::roleplay::strip_memory_lines(&body)
    } else {
        body.clone()
    };
    if streaming && !body.is_empty() {
        shown.push('▌');
    }
    if !shown.trim().is_empty() {
        ui.scope(|ui| {
            // Slightly larger body text for replies than egui's default.
            if let Some(body) = ui.style_mut().text_styles.get_mut(&egui::TextStyle::Body) {
                body.size = 14.5;
            }
            // Code blocks: same frosted colour as the top bar's card (PANEL()),
            // no outline, and rounded to 22 — the viewer takes its fill from
            // `extreme_bg_color` and the radius/stroke from the noninteractive
            // widget style.
            let v = ui.visuals_mut();
            v.extreme_bg_color = PANEL();
            v.widgets.noninteractive.corner_radius = egui::CornerRadius::same(22);
            v.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
            egui_commonmark::CommonMarkViewer::new().show(ui, &mut llm.md_cache, &shown);
        });
    }

    // Action row under every finished reply: copy / regenerate / listen.
    if !streaming {
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            if icon_button(ui, egui::include_image!("../icons/copy.svg"), 15.0, "Copy message", true)
                .clicked()
            {
                ui.ctx().copy_text(text.clone());
            }
            let can_retry = !llm.running;
            if icon_button(ui, egui::include_image!("../icons/retry.svg"), 15.0, "Regenerate this reply", can_retry)
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
            if icon_button(ui, egui::include_image!("../icons/listen.svg"), 15.0, tip, true).clicked() {
                llm.listen(&text, ui.ctx());
            }
            // First OmniVoice use loads the model — show that it's coming.
            if llm.voice.loading || llm.voice.pending > 0 {
                ui.add(egui::Spinner::new().size(11.0).color(MUTED()));
            }
            if let Some(e) = &llm.voice.last_err {
                ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(11.0));
            }
        });
    }
}

/// The collapsible "Thinking" row above a reply: the little AI orb, a muted
/// label and a drop-down arrow. Clicking anywhere on it (orb included)
/// reveals the model's thought channel underneath, muted in a soft card. The
/// orb runs hot while the thought still streams (`live`), then settles to
/// its idle breathing.
fn thinking_row(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, thinking: &str, live: bool) {
    let chat_id = llm.chats[llm.active_chat].id;
    let open_id = egui::Id::new(("ai_thinking_open", chat_id, index));
    let mut open = ui.data_mut(|d| d.get_temp::<bool>(open_id).unwrap_or(false));
    let mut toggled = false;

    ui.horizontal(|ui| {
        let orb = llm.think_orbs.entry((chat_id, index)).or_default();
        orb.set_state(if live {
            crate::ai_orb::OrbState::Thinking
        } else {
            crate::ai_orb::OrbState::Idle
        });
        if orb.show(ui, 20.0, None).clicked() {
            toggled = true;
        }

        let label = if live { "Thinking…" } else { "Thoughts" };
        let galley = ui.painter().layout_no_wrap(
            label.to_string(),
            egui::FontId::proportional(12.5),
            MUTED(),
        );
        let size = egui::vec2(galley.size().x + 4.0 + 16.0, 20.0);
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
        let resp = resp
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("See what the model thought before answering");
        if resp.clicked() {
            toggled = true;
        }
        if ui.is_rect_visible(rect) {
            let text_pos = egui::pos2(rect.left(), rect.center().y - galley.size().y / 2.0);
            ui.painter().galley(text_pos, galley, MUTED());
            let arrow = if open {
                egui::include_image!("../icons/arrow_drop_down.svg")
            } else {
                egui::include_image!("../icons/arrow_right.svg")
            };
            egui::Image::new(arrow).tint(icon_tint(MUTED())).paint_at(
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

    if open {
        ui.add_space(2.0);
        egui::Frame::new()
            .fill(PANEL().gamma_multiply(0.55))
            .corner_radius(egui::CornerRadius::same(14))
            .inner_margin(egui::Margin::symmetric(12, 9))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(egui::RichText::new(thinking).color(MUTED()).size(12.5));
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
    if let Some(t) = llm.msg_thumbs.get(path) {
        return t.clone();
    }
    let tex = image::open(path).ok().map(|img| {
        let t = img.thumbnail(512, 512).to_rgba8();
        let size = [t.width() as usize, t.height() as usize];
        ctx.load_texture(
            format!("ai-chat-img-{}", path.display()),
            egui::ColorImage::from_rgba_unmultiplied(size, t.as_raw()),
            Default::default(),
        )
    });
    llm.msg_thumbs.insert(path.to_path_buf(), tex.clone());
    tex
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
    match image::open(&path) {
        Ok(img) => {
            let t = img.thumbnail(128, 128).to_rgba8();
            let size = [t.width() as usize, t.height() as usize];
            let tex = ctx.load_texture(
                "ai-chat-draft-thumb",
                egui::ColorImage::from_rgba_unmultiplied(size, t.as_raw()),
                Default::default(),
            );
            llm.draft_thumb = Some((path, tex));
        }
        Err(e) => {
            llm.draft_image = None;
            llm.draft_thumb = None;
            llm.run_err = Some(format!("Couldn't read that image: {e}"));
        }
    }
}

/// A bare icon button on a 28px click target (same look as the text-to-image
/// view's send button): the SVG paints dead-centre, tinted to the theme's
/// text colour, dimmed when disabled.
fn icon_button(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'_>,
    icon_size: f32,
    tip: &str,
    enabled: bool,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::click());
    let resp = resp.on_hover_text(tip);
    let resp = if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp };
    if ui.is_rect_visible(rect) {
        let tint = icon_tint(TEXT());
        let tint = if enabled { tint } else { tint.gamma_multiply(0.45) };
        egui::Image::new(icon)
            .tint(tint)
            .paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(icon_size, icon_size)));
    }
    resp
}
