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

/// Width of the chats side panel (a slimmer sibling of the 290px browser).
const PANEL_W: f32 = 250.0;

pub fn show(ui: &mut egui::Ui, llm: &mut LlmState) {
    llm.poll();

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
                chat_list(ui, llm);
            });
        });

    // --- Centre: the conversation + input pill, in a centred column. ---
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(BG()).inner_margin(Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            conversation(ui, llm);
        });
}

/// The chat list card: a "Chats" header with a new-chat button, then one row
/// per conversation ("tabs like switch between chats").
fn chat_list(ui: &mut egui::Ui, llm: &mut LlmState) {
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

/// The conversation column: scrollable bubbles + the bottom input pill, both
/// centred on the same column (like the sketch's two guide lines).
fn conversation(ui: &mut egui::Ui, llm: &mut LlmState) {
    let avail_w = ui.available_width();
    // The centred content column both the messages and the pill live in.
    let col_w = (avail_w * 0.58).clamp(420.0, 880.0).min(avail_w - 16.0);
    let pad = ((avail_w - col_w) / 2.0).max(0.0);

    // Room reserved under the messages for the pill (and its extra lines).
    let mut input_h = 96.0;
    if llm.draft_image.is_some() {
        input_h += 24.0;
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
                });
            });
            ui.add_space(6.0);
        });

    // --- Bottom input pill (centred on the same column) ---
    ui.add_space(8.0);
    if let Some(e) = llm.run_err.clone() {
        ui.horizontal(|ui| {
            ui.add_space(pad + 14.0);
            ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
        });
        ui.add_space(2.0);
    }
    // Attached-image chip, shown above the pill until sent or removed.
    if let Some(img) = llm.draft_image.clone() {
        ui.horizontal(|ui| {
            ui.add_space(pad + 14.0);
            let name = img.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            ui.label(egui::RichText::new(format!("🖼 {name}")).color(MUTED()).size(12.0));
            if ui.small_button("✕").on_hover_text("Remove the image").clicked() {
                llm.draft_image = None;
            }
        });
        ui.add_space(2.0);
    }

    ui.horizontal(|ui| {
        ui.add_space(pad);
        egui::Frame::new()
            .fill(ui.visuals().extreme_bg_color)
            .stroke(egui::Stroke::new(1.0, EDGE()))
            .corner_radius(egui::CornerRadius::same(255))
            .inner_margin(egui::Margin::symmetric(14, 12))
            .show(ui, |ui| {
                ui.set_width(col_w - 28.0);
                ui.horizontal(|ui| {
                    // "+" — attach an image (the vision input).
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
                    ui.add_space(4.0);

                    // Draft text. Enter sends; Shift+Enter makes a newline.
                    let send_now = {
                        let edit = egui::TextEdit::multiline(&mut llm.draft)
                            .desired_rows(1)
                            .frame(egui::Frame::NONE)
                            .font(egui::FontId::proportional(15.0))
                            .hint_text("Ask Gemma")
                            .desired_width(ui.available_width() - 40.0);
                        let resp = ui.add(edit);
                        let enter = resp.has_focus()
                            && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                        if enter {
                            // Drop the newline the Enter keystroke inserted.
                            while llm.draft.ends_with('\n') {
                                llm.draft.pop();
                            }
                        }
                        enter
                    };

                    // Send — the text-to-image view's send icon.
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

    // While a reply streams in, keep painting and show the worker status.
    if llm.running {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.add_space(pad + 14.0);
            ui.add(egui::Spinner::new().size(12.0).color(MUTED()));
            ui.label(egui::RichText::new(&llm.status).color(MUTED()).size(11.5));
        });
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(120));
    }
}

/// One conversation entry. User messages sit right in an accent-tinted
/// bubble (corner radius 22) that hugs the text; the model's replies render
/// with no bubble at all — just modern formatted markdown (real bold, bullet
/// lists, headings, code blocks) straight on the background, like current
/// assistant UIs. A streaming reply shows a ▌ cursor.
fn message(ui: &mut egui::Ui, llm: &mut LlmState, index: usize, streaming: bool) {
    let (role, text, image) = {
        let m = &llm.chats[llm.active_chat].msgs[index];
        (m.role, m.text.clone(), m.image.clone())
    };

    if role == ChatRole::User {
        ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
            egui::Frame::new()
                .fill(ACCENT1().gamma_multiply(0.28))
                .corner_radius(egui::CornerRadius::same(22))
                .inner_margin(egui::Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_max_width(ui.available_width() * 0.82);
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        if let Some(img) = &image {
                            let name = img
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                            ui.label(egui::RichText::new(format!("🖼 {name}")).color(MUTED()).size(11.5));
                        }
                        if !text.is_empty() {
                            // Emoji-aware label: color Twemoji instead of
                            // monochrome font glyphs.
                            crate::emoji::label(ui, &text, TEXT(), 14.0, false);
                        }
                    });
                });
        });
        return;
    }

    // Model reply. Nothing streamed yet → a spinner with the worker status
    // ("Loading the model…", "Thinking…") where the reply will appear.
    if streaming && text.is_empty() {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0).color(MUTED()));
            ui.label(egui::RichText::new(&llm.status).color(MUTED()).size(12.5));
        });
        return;
    }
    let mut shown = text;
    if streaming {
        shown.push('▌');
    }
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
