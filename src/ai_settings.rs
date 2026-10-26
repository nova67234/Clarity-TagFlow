//! AI Settings: the window behind the gear in the AI Chat's Tools menu. Built
//! from the main Settings window's parts (src/settings.rs) so the two look the
//! same: a sidebar with the title and tabs, sections of rounded cards with
//! label/description rows and macOS-style switches on the right.
//!
//! * **Replies:** the sampling knobs (temperature, top-k, top-p, reply length).
//! * **Memory:** remembering across chats, the 3D memory map
//!   (src/memory_map.rs), the AI's own notes, and forgetting.
//! * **Role play:** the character (names, persona) and its memories — a
//!   store of their own (`MemoryKind::Roleplay`), kept apart from the main
//!   memory, with the same map. The Tools menu's gear opens this tab while
//!   role play is on.

use eframe::egui;

use crate::llm::LlmState;
use crate::settings::{card, row, row_sep, section, switch, Settings};
use crate::theme::*;

/// Which tab the AI Settings window shows (not persisted).
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash, Debug)]
pub enum AiTab {
    #[default]
    Replies,
    Memory,
    Roleplay,
}

/// Draw the AI Settings window when it's open (`llm.gen_settings_open`).
pub fn window(ctx: &egui::Context, llm: &mut LlmState, settings: &mut Settings) {
    if !llm.gen_settings_open {
        return;
    }
    let mut want_close = ctx.input(|i| i.key_pressed(egui::Key::Escape)) && !llm.memory_map.open;
    // Same fixed size rule as the main Settings window (kept on the screen
    // at large interface sizes).
    let (win_w, win_h) = crate::settings::settings_window_size(ctx.content_rect());
    egui::Window::new("AI Settings")
        .id(egui::Id::new("ai_settings_window"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(crate::settings::window_frame())
        .fixed_size(egui::vec2(win_w, win_h))
        .show(ctx, |ui| {
            let radius = egui::CornerRadius::same(10);
            let visuals = ui.visuals_mut();
            visuals.widgets.noninteractive.corner_radius = radius;
            visuals.widgets.inactive.corner_radius = radius;
            visuals.widgets.hovered.corner_radius = radius;
            visuals.widgets.active.corner_radius = radius;
            visuals.widgets.open.corner_radius = radius;

            let row_resp = ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;

                // --- Sidebar: icon + title, underline, tabs ---
                ui.vertical(|ui| {
                    ui.set_width(136.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        ui.add(
                            egui::Image::new(egui::include_image!("../icons/ai-brain.svg"))
                                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                .tint(TEXT()),
                        );
                        ui.heading(egui::RichText::new("AI Settings").color(TEXT()).strong().size(17.0));
                    });
                    ui.add_space(8.0);
                    let w = ui.available_width() - 10.0;
                    let (line, _) = ui.allocate_exact_size(egui::vec2(w, 1.0), egui::Sense::hover());
                    ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0_f32, EDGE()));
                    ui.add_space(10.0);
                    ui.spacing_mut().item_spacing.y = 4.0;
                    for (tab, label) in
                        [(AiTab::Replies, "Replies"), (AiTab::Memory, "Memory"), (AiTab::Roleplay, "Role play")]
                    {
                        if crate::settings::side_tab(ui, llm.ai_settings_tab == tab, label, false) {
                            llm.ai_settings_tab = tab;
                        }
                    }
                });
                ui.add_space(16.0);

                // --- Content: close button, then the tab's body ---
                ui.vertical(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        if ui
                            .add(
                                egui::Button::image(
                                    egui::Image::new(egui::include_image!("../icons/close.svg"))
                                        .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                        .tint(TEXT()),
                                )
                                .frame(false),
                            )
                            .on_hover_text("Close")
                            .clicked()
                        {
                            want_close = true;
                        }
                    });
                    ui.add_space(2.0);
                    const SCROLL_GUTTER: f32 = 12.0;
                    let mut scroll_ui = crate::edge_scroll_ui(ui, SCROLL_GUTTER);
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(&mut scroll_ui, |ui| {
                            ui.set_width(ui.available_width() - SCROLL_GUTTER);
                            // Scope ids by tab (see the main Settings window).
                            ui.push_id(llm.ai_settings_tab, |ui| match llm.ai_settings_tab {
                                AiTab::Replies => replies_tab(ui, settings),
                                AiTab::Memory => memory_tab(ui, llm, settings),
                                AiTab::Roleplay => roleplay_tab(ui, llm, settings),
                            });
                        });
                    crate::edge_scroll_done(ui, &scroll_ui, SCROLL_GUTTER);
                });
            });

            // The sidebar/content divider, full height.
            let rect = row_resp.response.rect;
            ui.painter().vline(rect.left() + 144.0, rect.y_range(), egui::Stroke::new(1.0_f32, EDGE()));
        });
    if want_close {
        llm.gen_settings_open = false;
    }
}

/// A slider sized to sit in a settings row's control slot.
fn row_slider(ui: &mut egui::Ui, slider: egui::Slider<'_>) {
    ui.spacing_mut().slider_width = 110.0;
    ui.add(slider);
}

/// Replies: the sampling knobs. Edits land in `settings.ai_gen` and apply from
/// the next message (main.rs mirrors them onto `LlmState` each frame).
fn replies_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    let p = &mut settings.ai_gen;
    section(ui, "Replies", |ui| {
        row(
            ui,
            "Temperature",
            Some("Randomness: 0 is focused and repeatable, 1 is Gemma's recommended default, 2 gets wild."),
            |ui| row_slider(ui, egui::Slider::new(&mut p.temperature, 0.0..=2.0).step_by(0.05).fixed_decimals(2)),
        );
        row_sep(ui);
        row(
            ui,
            "Top-K",
            Some("Pick each word from only the K most likely candidates. Lower is safer. Default 64."),
            |ui| row_slider(ui, egui::Slider::new(&mut p.top_k, 1..=128)),
        );
        row_sep(ui);
        row(
            ui,
            "Top-P",
            Some("Or by probability: drop unlikely words past this share. Lower is safer. Default 0.95."),
            |ui| row_slider(ui, egui::Slider::new(&mut p.top_p, 0.05..=1.0).step_by(0.01).fixed_decimals(2)),
        );
        row_sep(ui);
        row(
            ui,
            "Max reply length",
            Some("The longest answer the AI may write, in tokens (about ¾ of a word each). Default 3072."),
            |ui| row_slider(ui, egui::Slider::new(&mut p.max_tokens, 256..=8192).step_by(64.0)),
        );
    });
    // The thinking level lives in GenParams too, but it has its own control
    // (the chat's model picker) — resetting the sampling knobs leaves it alone.
    let defaults = crate::llm::GenParams { thinking: p.thinking, ..Default::default() };
    ui.horizontal(|ui| {
        if ui
            .add_enabled(*p != defaults, egui::Button::new(egui::RichText::new("Reset to defaults").size(12.5)))
            .clicked()
        {
            *p = defaults;
        }
        ui.label(egui::RichText::new("Thinking is set in the chat's model picker.").color(MUTED()).size(11.0));
    });
}

/// Memory: remembering across chats, the map, the AI's notes, forgetting.
fn memory_tab(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut Settings) {
    if llm.memory.load_failed {
        section(ui, "Memories couldn't be read", |ui| {
            ui.label(
                egui::RichText::new(
                    "They were probably saved on another computer or user account. They're left \
                     untouched, and nothing new is saved until you start fresh. Starting fresh keeps \
                     the old folder as ai_memory.bak.",
                )
                .color(TEXT())
                .size(12.5),
            );
            ui.add_space(6.0);
            if ui.button(egui::RichText::new("Start fresh").size(12.5)).clicked() {
                llm.memory.start_fresh();
                llm.reset_memory_queue();
            }
        });
    }

    let (n, topics) = (llm.memory.count(), llm.memory.topics.len());
    section(ui, "Memory", |ui| {
        row(
            ui,
            "Remember across chats",
            // Only Windows really encrypts (DPAPI); src/secret.rs merely
            // obfuscates on other systems, so don't promise it there.
            Some(if cfg!(windows) {
                "After each chat the AI writes down what matters, by itself, and remembers it in new \
                 chats. Stored encrypted on this computer."
            } else {
                "After each chat the AI writes down what matters, by itself, and remembers it in new \
                 chats. Stored on this computer."
            }),
            |ui| {
                switch(ui, &mut settings.ai_memory_enabled);
            },
        );
        // What it's doing, while it's on (the switch takes effect next frame).
        if settings.ai_memory_enabled {
            use crate::llm::MemoryActivity as A;
            let status = match llm.memory_activity() {
                A::Updating => "Updating memories now…".to_string(),
                A::Waiting(1) => "1 chat waiting to be read.".to_string(),
                A::Waiting(n) => format!("{n} chats waiting to be read."),
                A::UpToDate | A::Off => "Up to date.".to_string(),
                A::Unreadable => String::new(),
            };
            if !status.is_empty() {
                crate::settings::hint(
                    ui,
                    &format!(
                        "{status} A chat is read once you leave it or it has been quiet for 3 minutes, \
                         while the model is loaded. Role play and Tag images never are; of Krea 2 \
                         prompts, only the pictures (they still use what's remembered)."
                    ),
                );
            }
        }
        row_sep(ui);
        let stats = if n == 0 {
            "Nothing remembered yet.".to_string()
        } else {
            let updated = llm
                .memory
                .last_updated()
                .map(|t| format!(", updated {}", crate::memory::format_date(t)))
                .unwrap_or_default();
            format!(
                "{n} {} in {topics} {}{updated}.",
                if n == 1 { "memory" } else { "memories" },
                if topics == 1 { "topic" } else { "topics" },
            )
        };
        row(ui, "Memory map", Some(&stats), |ui| {
            if ui
                .button(egui::RichText::new("Open").size(12.5))
                .on_hover_text("See every memory as a 3D brain")
                .clicked()
            {
                llm.memory_map.show(crate::memory::MemoryKind::Main);
                llm.gen_settings_open = false;
            }
        });
    });

    album_section(ui, llm);

    section(ui, "The AI's notes", |ui| {
        ui.label(egui::RichText::new("Persona").color(TEXT()).size(13.0));
        ui.label(
            egui::RichText::new(
                "How the AI sees itself and its voice. It updates these notes after chats; you can edit them.",
            )
            .color(MUTED())
            .size(10.5),
        );
        ui.add_space(4.0);
        // Typed into a draft and saved when the box loses focus, so the
        // encrypted file isn't rewritten on every keystroke.
        let id = ui.id().with("persona_draft");
        let mut draft: String =
            ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| llm.memory.header.persona.clone());
        let resp = ui.add_enabled(
            !llm.memory.load_failed,
            egui::TextEdit::multiline(&mut draft)
                .hint_text("Nothing yet — the AI writes this after your first chats.")
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        if resp.lost_focus() {
            llm.memory.set_persona(draft.trim());
            ui.data_mut(|d| d.remove::<String>(id));
        } else if resp.has_focus() {
            ui.data_mut(|d| d.insert_temp(id, draft));
        }
        row_sep(ui);
        ui.label(egui::RichText::new("Last chat summary").color(TEXT()).size(13.0));
        let summary = if llm.memory.header.summary.trim().is_empty() {
            "No summary yet. It's written after a chat ends.".to_string()
        } else {
            llm.memory.header.summary.clone()
        };
        ui.label(egui::RichText::new(summary).color(MUTED()).size(12.0));
    });

    // Forgetting: a red button that asks once before deleting everything.
    let confirm_id = ui.id().with("confirm_clear");
    let confirming = ui.data(|d| d.get_temp::<bool>(confirm_id)).unwrap_or(false);
    card(ui, |ui| {
        row(
            ui,
            "Clear all memories",
            Some("Deletes every memory, the kept pictures, the summary and the AI's notes from this computer."),
            |ui| {
                let enabled = !llm.memory.load_failed && (n > 0 || !llm.memory.is_empty());
                if confirming {
                    let yes = egui::Button::new(egui::RichText::new("Delete all").color(egui::Color32::WHITE).size(12.5))
                        .fill(egui::Color32::from_rgb(180, 40, 40));
                    if ui.add(yes).clicked() {
                        llm.memory.clear_all();
                        crate::album::clear(crate::memory::MemoryKind::Main);
                        // Nor does anything waiting to be read come back.
                        llm.reset_memory_queue();
                        ui.data_mut(|d| d.remove::<bool>(confirm_id));
                    }
                    if ui.button(egui::RichText::new("Cancel").size(12.5)).clicked() {
                        ui.data_mut(|d| d.remove::<bool>(confirm_id));
                    }
                } else if ui.add_enabled(enabled, egui::Button::new(egui::RichText::new("Clear…").size(12.5))).clicked() {
                    ui.data_mut(|d| d.insert_temp(confirm_id, true));
                }
            },
        );
    });
}

/// One picture of the album: where its memory is, and what it says.
struct Kept {
    topic: usize,
    id: u64,
    title: String,
    text: String,
    image: String,
    created: i64,
}

/// The pictures the AI kept from chats (src/album.rs): a grid of them,
/// newest first; clicking one opens it under the grid with what it shows,
/// why the AI kept it and Delete (which forgets its memory too).
fn album_section(ui: &mut egui::Ui, llm: &mut LlmState) {
    use crate::memory::{MemoryKind, PICTURES_TOPIC};
    const THUMB: f32 = 84.0;
    let mut kept: Vec<Kept> = llm
        .memory
        .find_topic(PICTURES_TOPIC)
        .map(|ti| {
            llm.memory.topics[ti]
                .memories
                .iter()
                .filter_map(|m| {
                    Some(Kept {
                        topic: ti,
                        id: m.id,
                        title: m.title.clone(),
                        text: m.text.clone(),
                        image: m.image.clone()?,
                        created: m.created,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    kept.sort_by_key(|k| std::cmp::Reverse(k.created));

    let open_id = ui.id().with("album_open");
    let mut open: Option<u64> = ui.data(|d| d.get_temp(open_id)).filter(|id| kept.iter().any(|k| k.id == *id));
    let mut delete: Option<usize> = None;
    section(ui, "Album", |ui| {
        if kept.is_empty() {
            crate::settings::hint(
                ui,
                "Nothing kept yet. When you share pictures, in a chat or in Krea 2 prompts, the AI keeps \
                 the few that matter to you — what they show and why — and gets to know what you love.",
            );
            return;
        }
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
            for k in &kept {
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(THUMB, THUMB), egui::Sense::click());
                let resp = resp.on_hover_text(&k.title).on_hover_cursor(egui::CursorIcon::PointingHand);
                let tex = crate::album::load(MemoryKind::Main, &k.image).and_then(|p| llm.media_thumb(ui.ctx(), &p));
                let radius = egui::CornerRadius::same(12);
                match tex {
                    // Filled, not fitted: the middle of the picture.
                    Some(tex) => {
                        let size = tex.size_vec2();
                        let s = (size.x.min(size.y) / size.x.max(1.0), size.x.min(size.y) / size.y.max(1.0));
                        let uv = egui::Rect::from_center_size(egui::pos2(0.5, 0.5), egui::vec2(s.0, s.1));
                        egui::Image::new(&tex).uv(uv).corner_radius(radius).paint_at(ui, rect);
                    }
                    None => {
                        ui.painter().rect_filled(rect, radius, TEXT().gamma_multiply(0.06));
                    }
                }
                let ring = if open == Some(k.id) {
                    egui::Stroke::new(2.0, crate::theme::selection_outline())
                } else if resp.hovered() {
                    egui::Stroke::new(1.0, TEXT().gamma_multiply(0.4))
                } else {
                    egui::Stroke::new(1.0, TEXT().gamma_multiply(0.08))
                };
                ui.painter().rect_stroke(rect, radius, ring, egui::StrokeKind::Inside);
                if resp.clicked() {
                    open = if open == Some(k.id) { None } else { Some(k.id) };
                }
            }
        });

        let Some((i, k)) = kept.iter().enumerate().find(|(_, k)| open == Some(k.id)) else { return };
        row_sep(ui);
        if let Some(tex) = crate::album::load(MemoryKind::Main, &k.image).and_then(|p| llm.media_thumb(ui.ctx(), &p)) {
            let size = tex.size_vec2();
            let scale = (ui.available_width() / size.x).min(240.0 / size.y).min(1.0);
            ui.add(egui::Image::new(&tex).fit_to_exact_size(size * scale).corner_radius(egui::CornerRadius::same(12)));
            ui.add_space(8.0);
        }
        ui.label(egui::RichText::new(&k.title).color(TEXT()).font(crate::settings::bold(13.5)));
        ui.add_space(2.0);
        ui.label(egui::RichText::new(&k.text).color(MUTED()).size(12.0));
        // Why it was kept, in the AI's words (read once, then kept here).
        let why_id = open_id.with(("why", k.id));
        let why: Option<String> = ui
            .data(|d| d.get_temp::<Option<String>>(why_id))
            .unwrap_or_else(|| {
                let why = crate::album::why(MemoryKind::Main, &k.image).filter(|w| !w.trim().is_empty());
                ui.data_mut(|d| d.insert_temp(why_id, why.clone()));
                why
            });
        if let Some(why) = why {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(format!("Why I kept it: {why}")).color(TEXT()).italics().size(12.0));
        }
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("Kept {}", crate::memory::format_date(k.created)))
                    .color(MUTED())
                    .size(11.0),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(egui::RichText::new("Delete").size(12.0))
                    .on_hover_text("Delete this picture and what the AI remembers about it")
                    .clicked()
                {
                    delete = Some(i);
                }
            });
        });
    });
    if let Some(i) = delete {
        let k = &kept[i];
        llm.memory.delete(k.topic, k.id);
        crate::album::remove(MemoryKind::Main, &k.image);
        open = None;
    }
    ui.data_mut(|d| match open {
        Some(id) => {
            d.insert_temp(open_id, id);
        }
        None => d.remove::<u64>(open_id),
    });
}

/// A small label above a text box (the Role play tab's character fields).
fn field_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(TEXT()).size(13.0));
    ui.add_space(2.0);
}

/// Role play: the character (names, persona) and its memories, which live in
/// a store of their own — the same format and map as the main memory, never
/// mixed with it.
fn roleplay_tab(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut Settings) {
    if llm.roleplay.load_failed {
        section(ui, "The character couldn't be read", |ui| {
            ui.label(
                egui::RichText::new(
                    "It was probably saved on another computer or user account. It's left untouched, \
                     and nothing is saved over it until you start a new one. Starting a new one keeps \
                     the old file as ai_chat_roleplay.json.bak.",
                )
                .color(TEXT())
                .size(12.5),
            );
            ui.add_space(6.0);
            if ui.button(egui::RichText::new("Start a new character").size(12.5)).clicked() {
                llm.roleplay.start_new_diary();
            }
        });
    }
    if llm.rp_memory.load_failed {
        section(ui, "Role play memories couldn't be read", |ui| {
            ui.label(
                egui::RichText::new(
                    "They were probably saved on another computer or user account. They're left \
                     untouched, and nothing new is saved until you start fresh. Starting fresh keeps \
                     the old folder as ai_memory_roleplay.bak.",
                )
                .color(TEXT())
                .size(12.5),
            );
            ui.add_space(6.0);
            if ui.button(egui::RichText::new("Start fresh").size(12.5)).clicked() {
                llm.rp_memory.start_fresh();
            }
        });
    }

    section(ui, "Role play", |ui| {
        row(
            ui,
            "Role playing",
            Some("The AI plays a character with its own name, persona and memories. Switching starts a new chat."),
            |ui| {
                let mut on = llm.roleplay.enabled;
                if switch(ui, &mut on).changed() {
                    crate::ai_chat::set_mode(llm, settings, crate::ai_chat::ChatMode::Roleplay, on);
                }
            },
        );
    });

    section(ui, "Character", |ui| {
        let rp = &mut llm.roleplay;
        let mut changed = false;
        ui.add_enabled_ui(!rp.load_failed, |ui| {
            ui.horizontal(|ui| {
                let w = ((ui.available_width() - 12.0) / 2.0).max(60.0);
                ui.vertical(|ui| {
                    ui.set_width(w);
                    field_label(ui, "AI's name");
                    changed |= ui
                        .add(egui::TextEdit::singleline(&mut rp.ai_name).hint_text("e.g. Mira").desired_width(w))
                        .lost_focus();
                });
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.set_width(w);
                    field_label(ui, "Your name");
                    changed |= ui
                        .add(egui::TextEdit::singleline(&mut rp.user_name).hint_text("What it calls you").desired_width(w))
                        .lost_focus();
                });
            });
            row_sep(ui);
            field_label(ui, "Persona");
            changed |= ui
                .add(
                    egui::TextEdit::multiline(&mut rp.persona)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY)
                        .hint_text("Who is the AI? Looks, personality, the world you're in…"),
                )
                .lost_focus();
        });
        if changed {
            rp.save();
        }
    });

    let (n, topics) = (llm.rp_memory.count(), llm.rp_memory.topics.len());
    section(ui, "Memories", |ui| {
        let stats = if n == 0 {
            "Nothing remembered yet — the AI writes memories as the story unfolds.".to_string()
        } else {
            let updated = llm
                .rp_memory
                .last_updated()
                .map(|t| format!(", updated {}", crate::memory::format_date(t)))
                .unwrap_or_default();
            format!(
                "{n} {} in {topics} {}{updated}.",
                if n == 1 { "memory" } else { "memories" },
                if topics == 1 { "topic" } else { "topics" },
            )
        };
        row(ui, "Memory map", Some(&stats), |ui| {
            if ui
                .button(egui::RichText::new("Open").size(12.5))
                .on_hover_text("See the character's memories as a 3D brain")
                .clicked()
            {
                llm.memory_map.show(crate::memory::MemoryKind::Roleplay);
                llm.gen_settings_open = false;
            }
        });
        row_sep(ui);
        ui.label(
            egui::RichText::new(
                "Kept apart from the AI's main memory, so the story never comes up in normal chats. \
                 Add, edit or forget memories in the map.",
            )
            .color(MUTED())
            .size(10.5),
        );
    });

    // Forgetting: asks once before deleting everything.
    let confirm_id = ui.id().with("confirm_clear_rp");
    let confirming = ui.data(|d| d.get_temp::<bool>(confirm_id)).unwrap_or(false);
    card(ui, |ui| {
        row(
            ui,
            "Clear role play memories",
            Some("Deletes every role play memory and the pictures the AI kept, from this computer."),
            |ui| {
                let enabled = !llm.rp_memory.load_failed && n > 0;
                if confirming {
                    let yes = egui::Button::new(egui::RichText::new("Delete all").color(egui::Color32::WHITE).size(12.5))
                        .fill(egui::Color32::from_rgb(180, 40, 40));
                    if ui.add(yes).clicked() {
                        llm.rp_memory.clear_all();
                        crate::roleplay::clear_album();
                        ui.data_mut(|d| d.remove::<bool>(confirm_id));
                    }
                    if ui.button(egui::RichText::new("Cancel").size(12.5)).clicked() {
                        ui.data_mut(|d| d.remove::<bool>(confirm_id));
                    }
                } else if ui.add_enabled(enabled, egui::Button::new(egui::RichText::new("Clear…").size(12.5))).clicked() {
                    ui.data_mut(|d| d.insert_temp(confirm_id, true));
                }
            },
        );
    });
}
