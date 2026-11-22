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
//! * **Web research:** the search service (and its key or address), how far
//!   a lookup goes, and where its notes are saved (src/research.rs).

use eframe::egui;

use crate::llm::LlmState;
use crate::settings::{card, row, row_sep, section, switch, Settings};
use crate::slider::PillSlider;
use crate::theme::*;

/// Which tab the AI Settings window shows (not persisted).
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash, Debug)]
pub enum AiTab {
    #[default]
    Replies,
    Memory,
    Roleplay,
    Research,
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
    // Under Liquid Glass it takes the main Settings window's look.
    let liquid = crate::theme::liquid_markers().is_some();
    if liquid {
        crate::settings::begin_main_look();
    }
    egui::Window::new("AI Settings")
        .id(egui::Id::new("ai_settings_window"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(crate::settings::fresh_window_frame())
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
                    for (tab, label) in [
                        (AiTab::Replies, "Replies"),
                        (AiTab::Memory, "Memory"),
                        (AiTab::Roleplay, "Role play"),
                        (AiTab::Research, "Web research"),
                    ] {
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
                            ui.set_width((ui.available_width() - SCROLL_GUTTER).max(0.0));
                            // Scope ids by tab (see the main Settings window).
                            ui.push_id(llm.ai_settings_tab, |ui| match llm.ai_settings_tab {
                                AiTab::Replies => replies_tab(ui, settings),
                                AiTab::Memory => memory_tab(ui, llm, settings),
                                AiTab::Roleplay => roleplay_tab(ui, llm, settings),
                                AiTab::Research => research_tab(ui, llm),
                            });
                        });
                    crate::edge_scroll_done(ui, &scroll_ui, SCROLL_GUTTER);
                });
            });

            // The sidebar/content divider, full height.
            let rect = row_resp.response.rect;
            ui.painter().vline(rect.left() + 144.0, rect.y_range(), egui::Stroke::new(1.0_f32, EDGE()));
        });
    if liquid {
        crate::settings::end_main_look();
    }
    if want_close {
        llm.gen_settings_open = false;
    }
}

/// A slider sized to sit in a settings row's control slot.
fn row_slider(ui: &mut egui::Ui, slider: PillSlider<'_>) {
    ui.spacing_mut().slider_width = 110.0;
    ui.add(slider);
}

/// Replies: the sampling knobs. Edits land in `settings.ai_gen` and apply from
/// the next message (main.rs mirrors them onto `LlmState` each frame).
fn replies_tab(ui: &mut egui::Ui, settings: &mut Settings) {
    let p = &mut settings.ai_gen;
    // Each knob's notch sits at its default.
    let d = crate::llm::GenParams::default();
    section(ui, "Replies", |ui| {
        row(
            ui,
            "Temperature",
            Some("Randomness: 0 is focused and repeatable, 1 is Gemma's recommended default, 2 gets wild."),
            |ui| {
                row_slider(
                    ui,
                    PillSlider::new(&mut p.temperature, 0.0..=2.0)
                        .step_by(0.05)
                        .fixed_decimals(2)
                        .default_value(d.temperature as f64),
                )
            },
        );
        row_sep(ui);
        row(
            ui,
            "Top-K",
            Some("Pick each word from only the K most likely candidates. Lower is safer. Default 64."),
            |ui| row_slider(ui, PillSlider::new(&mut p.top_k, 1..=128).default_value(d.top_k as f64)),
        );
        row_sep(ui);
        row(
            ui,
            "Top-P",
            Some("Or by probability: drop unlikely words past this share. Lower is safer. Default 0.95."),
            |ui| {
                row_slider(
                    ui,
                    PillSlider::new(&mut p.top_p, 0.05..=1.0)
                        .step_by(0.01)
                        .fixed_decimals(2)
                        .default_value(d.top_p as f64),
                )
            },
        );
        row_sep(ui);
        row(
            ui,
            "Max reply length",
            Some("The longest answer the AI may write, in tokens (about ¾ of a word each). Default 3072."),
            |ui| {
                row_slider(
                    ui,
                    PillSlider::new(&mut p.max_tokens, 256..=8192).step_by(64.0).default_value(d.max_tokens as f64),
                )
            },
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
                 chats — only what's said while this is on. Stored encrypted on this computer."
            } else {
                "After each chat the AI writes down what matters, by itself, and remembers it in new \
                 chats — only what's said while this is on. Stored on this computer."
            }),
            |ui| {
                switch(ui, &mut settings.ai_memory_enabled);
            },
        );
        // What it's doing, while it's on (the switch takes effect next frame).
        if settings.ai_memory_enabled {
            memory_status(ui, llm);
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
        let resp = ui
            .add_enabled_ui(!llm.memory.load_failed, |ui| {
                crate::settings::well_box(
                    ui,
                    egui::TextEdit::multiline(&mut draft)
                        .hint_text("Nothing yet — the AI writes this after your first chats.")
                        .desired_rows(3),
                )
            })
            .inner;
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

/// Under "Remember across chats": what the memory is doing, with Save now
/// while chats wait to be saved, and what the last save did (or why it
/// didn't work).
fn memory_status(ui: &mut egui::Ui, llm: &mut LlmState) {
    use crate::llm::MemoryActivity as A;
    use crate::memory::{LastPass, Retry};
    let activity = llm.memory_activity();
    if activity == A::Unreadable {
        return;
    }
    let saving = llm.saving_memory_now();
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let status = match activity {
            A::Updating => "Saving to memory now…".to_string(),
            // Save now waits a little (after a failed try, or while another
            // PC uses this one's AI).
            A::Waiting(_) if saving => "Saving to memory soon…".to_string(),
            A::Waiting(1) => "1 chat isn't saved to memory yet.".to_string(),
            A::Waiting(n) => format!("{n} chats aren't saved to memory yet."),
            _ => "Everything is saved.".to_string(),
        };
        ui.label(egui::RichText::new(status).color(TEXT()).size(12.5));
        if matches!(activity, A::Waiting(_)) && !saving {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let tip = if crate::link::remote_chat().is_some() {
                    "Save what's waiting now, on the PC your chats run on (while it has its model loaded)."
                } else {
                    "Save what's waiting now instead of after a quiet moment. The model loads first if it \
                     isn't loaded."
                };
                let save = ui
                    .add_enabled(llm.usable(), egui::Button::new(egui::RichText::new("Save now").size(12.5)))
                    .on_hover_text(tip)
                    .on_disabled_hover_text("Set up the model first (Settings → AI Model).");
                if save.clicked() {
                    llm.save_memory_now();
                }
            });
        }
    });
    if let Some(loading) = llm.memory_loading() {
        crate::settings::hint(ui, loading);
    }
    if let (A::Waiting(_), Some(why)) = (activity, llm.memory_save_blocked()) {
        crate::settings::hint(ui, &format!("Not everything could be saved now: {why}."));
    }
    if let Some(last) = llm.memory_last() {
        let line = match last {
            LastPass::Saved { at, added, forgotten, notes, summary } => {
                let mut what: Vec<String> = Vec::new();
                match added {
                    0 => {}
                    1 => what.push("1 new or updated memory".to_string()),
                    n => what.push(format!("{n} new or updated memories")),
                }
                if *forgotten > 0 {
                    what.push(format!("forgot {forgotten}"));
                }
                if *notes {
                    what.push("its notes about itself".to_string());
                }
                if *summary {
                    what.push("a summary of the chat".to_string());
                }
                if what.is_empty() {
                    what.push("nothing new worth keeping".to_string());
                }
                format!("Last saved at {}: {}.", clock(*at), what.join(", "))
            }
            LastPass::Failed { at, why, retry } => {
                let next = match retry {
                    Retry::Soon => "It's tried again in a minute — nothing is lost.",
                    Retry::NextRun => {
                        "It's kept and tried again when the app next starts (or with Save now) — nothing \
                         is lost."
                    }
                    Retry::GaveUp => {
                        "It didn't work any time it was tried, so that part of a chat was left out — tell \
                         the AI again if it mattered."
                    }
                };
                format!("The last save, at {}, didn't work ({why}). {next}", clock(*at))
            }
        };
        crate::settings::hint(ui, &line);
    }
    crate::settings::hint(
        ui,
        "A chat is saved once you leave it, or after 3 quiet minutes while the model is loaded — and \
         until then, new chats already know what you said in it. Role play and Tag images are never \
         saved; of Krea 2 prompts, only the pictures (they still use what's remembered).",
    );
}

/// `secs` (unix) as a time of day: "9:41 PM".
fn clock(secs: i64) -> String {
    use chrono::TimeZone as _;
    chrono::Local.timestamp_opt(secs, 0).single().map(|t| t.format("%-I:%M %p").to_string()).unwrap_or_default()
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
                    egui::Stroke::new(2.0_f32, crate::theme::selection_outline())
                } else if resp.hovered() {
                    egui::Stroke::new(1.0_f32, TEXT().gamma_multiply(0.4))
                } else {
                    egui::Stroke::new(1.0_f32, TEXT().gamma_multiply(0.08))
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
        row_sep(ui);
        let rp = &mut llm.roleplay;
        ui.add_enabled_ui(!rp.load_failed, |ui| {
            row(
                ui,
                "Date and time",
                Some(
                    "The character is told today's date and time on this computer with each message. \
                     Turn it off for a story set in another time or world.",
                ),
                |ui| {
                    if switch(ui, &mut rp.real_time).changed() {
                        rp.save();
                    }
                },
            );
            row_sep(ui);
            row(
                ui,
                "Thoughts in character",
                Some(
                    "With a model that thinks before answering (Gemma 4 31B, Qwen3-VL 30B Thinking), \
                     its thoughts are the character's inner voice — first person, in character — \
                     instead of notes about the chat.",
                ),
                |ui| {
                    if switch(ui, &mut rp.thoughts_in_character).changed() {
                        rp.save();
                    }
                },
            );
        });
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
                    changed |=
                        crate::settings::pill_field(ui, w, egui::TextEdit::singleline(&mut rp.ai_name).hint_text("e.g. Mira"))
                            .lost_focus();
                });
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.set_width(w);
                    field_label(ui, "Your name");
                    changed |= crate::settings::pill_field(
                        ui,
                        w,
                        egui::TextEdit::singleline(&mut rp.user_name).hint_text("What it calls you"),
                    )
                    .lost_focus();
                });
            });
            row_sep(ui);
            field_label(ui, "Persona");
            changed |= crate::settings::well_box(
                ui,
                egui::TextEdit::multiline(&mut rp.persona)
                    .desired_rows(4)
                    .hint_text("Who is the AI? Looks, personality, the world you're in…"),
            )
            .lost_focus();
            changed |= persona_undo(ui, rp);
            row_sep(ui);
            field_label(ui, "Traits");
            changed |= traits_editor(ui, rp);
            row_sep(ui);
            row(
                ui,
                "Grows on its own",
                Some(
                    "As the story changes the character, it gains and drops traits and brings its \
                     persona up to date itself — a part at a time. A note under the reply says when, \
                     and a change to the persona can be undone here.",
                ),
                |ui| {
                    changed |= switch(ui, &mut rp.own_growth).changed();
                },
            );
        });
        if changed {
            rp.save();
        }
    });

    goals_section(ui, &mut llm.roleplay);

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
            Some("Deletes every role play memory, the character's goals and the pictures the AI kept, from this computer."),
            |ui| {
                let enabled = !llm.rp_memory.load_failed && (n > 0 || !llm.roleplay.goals.is_empty());
                if confirming {
                    let yes = egui::Button::new(egui::RichText::new("Delete all").color(egui::Color32::WHITE).size(12.5))
                        .fill(egui::Color32::from_rgb(180, 40, 40));
                    if ui.add(yes).clicked() {
                        llm.rp_memory.clear_all();
                        crate::roleplay::clear_album();
                        if !llm.roleplay.goals.is_empty() {
                            llm.roleplay.goals.clear();
                            llm.roleplay.save();
                        }
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

/// Under the persona, once the character has changed it: when it last did,
/// and Undo — back to the persona as it was before that change (shown on
/// hover), one change at a time. True when undone.
fn persona_undo(ui: &mut egui::Ui, rp: &mut crate::roleplay::RoleplayState) -> bool {
    let Some(before) = rp.persona_history.last() else { return false };
    let mut undone = false;
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let when = crate::memory::format_date(before.replaced);
        let was = if before.text.trim().is_empty() { "(empty)" } else { before.text.trim() };
        ui.label(egui::RichText::new(format!("{} changed this on {when}.", rp.ai_display())).color(MUTED()).size(11.5));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            undone = ui
                .button(egui::RichText::new("Undo").size(12.5))
                .on_hover_text(format!("Back to how it read before:\n\n{was}"))
                .clicked();
        });
    });
    undone && rp.undo_persona()
}

/// What the traits editor is doing between frames: the trait being edited
/// (`NEW_TRAIT` for one being added) and the text typed so far.
#[derive(Clone)]
struct TraitDraft {
    at: usize,
    text: String,
    /// The box has been given the keyboard.
    focused: bool,
}

const NEW_TRAIT: usize = usize::MAX;

/// Role play → Character → Traits: one pill a trait — the character's own
/// with a violet dot, like the Tag Manager's AI tags. Click one to rewrite
/// it (emptied, it goes), its × to remove it, the last pill to add one. True
/// when the traits changed.
fn traits_editor(ui: &mut egui::Ui, rp: &mut crate::roleplay::RoleplayState) -> bool {
    use crate::roleplay::MAX_TRAITS;
    let id = ui.id().with("rp_traits");
    let mut draft: Option<TraitDraft> = ui.data(|d| d.get_temp(id));
    // (A trait the character dropped meanwhile isn't there to edit.)
    if draft.as_ref().is_some_and(|d| d.at != NEW_TRAIT && d.at >= rp.traits.len()) {
        draft = None;
    }
    let who = rp.ai_display().to_string();
    let mut changed = false;
    // What the pills ask for, carried out once they're all drawn.
    let (mut edit, mut remove, mut commit) = (None, None, None);
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
        for (i, t) in rp.traits.iter().enumerate() {
            match &mut draft {
                Some(d) if d.at == i => commit = trait_field(ui, d).or(commit),
                _ => match trait_pill(ui, t, &who) {
                    Some(TraitAsk::Edit) => edit = Some(i),
                    Some(TraitAsk::Remove) => remove = Some(i),
                    None => {}
                },
            }
        }
        // The last pill adds one — or is the box it's being typed in.
        if let Some(d) = draft.as_mut().filter(|d| d.at == NEW_TRAIT) {
            commit = trait_field(ui, d).or(commit);
        } else if rp.traits.len() < MAX_TRAITS && add_pill(ui, rp.traits.is_empty()).clicked() {
            edit = Some(NEW_TRAIT);
        }
    });
    if rp.traits.is_empty() && draft.is_none() && edit.is_none() {
        ui.add_space(2.0);
        let hint = if rp.own_growth {
            "Who the character is, a few words each — add some, or let it find its own as the story goes."
        } else {
            "Who the character is, a few words each."
        };
        ui.label(egui::RichText::new(hint).color(MUTED()).size(11.5));
    }
    if let Some(done) = commit {
        if let (Some(d), TraitDone::Keep) = (&draft, done) {
            if d.at == NEW_TRAIT {
                changed |= rp.add_trait(&d.text, false, crate::memory::now()).is_some();
            } else {
                let before = rp.traits.clone();
                rp.set_trait(d.at, &d.text);
                changed |= rp.traits != before;
            }
        }
        draft = None;
    }
    if let Some(i) = remove {
        rp.traits.remove(i);
        changed = true;
        // (The one being edited keeps its place.)
        draft = draft.and_then(|d| match d.at {
            at if at == i => None,
            at if at != NEW_TRAIT && at > i => Some(TraitDraft { at: at - 1, ..d }),
            _ => Some(d),
        });
    }
    if let Some(at) = edit {
        let text = rp.traits.get(at).map(|t| t.text.clone()).unwrap_or_default();
        draft = Some(TraitDraft { at, text, focused: false });
    }
    ui.data_mut(|d| match draft {
        Some(draft) => {
            d.insert_temp(id, draft);
        }
        None => d.remove::<TraitDraft>(id),
    });
    changed
}

/// What a click on a trait's pill asks for.
enum TraitAsk {
    Edit,
    Remove,
}

/// How editing a trait ended.
#[derive(Clone, Copy)]
enum TraitDone {
    /// Enter, Tab or a click elsewhere: what's typed is the trait.
    Keep,
    /// Escape: as it was.
    Cancel,
}

/// The height of a trait's pill (a text box's, so nothing moves when one
/// turns into the other).
const TRAIT_H: f32 = 32.0;
/// The narrowest a trait's text box gets.
const TRAIT_FIELD_MIN: f32 = 140.0;

/// One trait as a pill: its words (cut short where the row is narrower), and
/// a × at its end.
fn trait_pill(ui: &mut egui::Ui, t: &crate::roleplay::Trait, who: &str) -> Option<TraitAsk> {
    let (pad, dot, close) = (12.0, if t.by_ai { 12.0 } else { 0.0 }, 22.0);
    let room = (ui.available_width().max(160.0) - pad - dot - close - 4.0).max(40.0);
    let mut job = egui::text::LayoutJob::simple_singleline(t.text.clone(), egui::FontId::proportional(12.5), TEXT());
    job.wrap = egui::text::TextWrapping::truncate_at_width(room);
    let galley = ui.painter().layout_job(job);
    let size = egui::vec2(pad + dot + galley.size().x + close + 4.0, TRAIT_H);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let x_rect = egui::Rect::from_center_size(egui::pos2(rect.right() - 4.0 - close / 2.0, rect.center().y), egui::vec2(close, close));
    let x = ui.interact(x_rect, resp.id.with("remove"), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &t.text));
    x.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, format!("Remove the trait {}", t.text)));
    let since = crate::memory::format_date(t.added);
    let whose = if t.by_ai { format!("{who}'s own, since {since}.") } else { "Yours.".to_string() };
    let resp = resp.on_hover_text(format!("{whose} Click to rewrite it.")).on_hover_cursor(egui::CursorIcon::PointingHand);
    let x = x.on_hover_text("Remove this trait").on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        let hot = resp.hovered() && !x.hovered();
        let fill = if hot { crate::settings::well_fill().lerp_to_gamma(TEXT(), 0.06) } else { crate::settings::well_fill() };
        p.rect(rect, TRAIT_H / 2.0, fill, egui::Stroke::new(1.0_f32, EDGE()), egui::StrokeKind::Inside);
        if t.by_ai {
            p.circle_filled(egui::pos2(rect.left() + pad + 3.0, rect.center().y), 3.0, crate::tag_manager::AI_NEW_VIOLET);
        }
        let at = egui::pos2(rect.left() + pad + dot, rect.center().y - galley.size().y / 2.0);
        p.galley(at, galley, TEXT());
        if x.hovered() {
            p.circle_filled(x_rect.center(), close / 2.0, TEXT().gamma_multiply(0.08));
        }
        let tint = crate::theme::icon_tint(if x.hovered() { TEXT() } else { MUTED() });
        let icon = egui::Rect::from_center_size(x_rect.center(), egui::vec2(11.0, 11.0));
        egui::Image::new(egui::include_image!("../icons/close.svg")).tint(tint).paint_at(ui, icon);
    }
    if x.clicked() {
        Some(TraitAsk::Remove)
    } else if resp.clicked() {
        Some(TraitAsk::Edit)
    } else {
        None
    }
}

/// The pill that adds a trait: a dashed-looking, quieter one at the end of
/// the row ("Add a trait" while there are none, a plus after).
fn add_pill(ui: &mut egui::Ui, first: bool) -> egui::Response {
    let label = if first { "+  Add a trait" } else { "+  Add" };
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(12.5), MUTED());
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(galley.size().x + 28.0, TRAIT_H), egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Add a trait"));
    let resp = resp.on_hover_text("Add a trait: who the character is, in a few words").on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let ink = if resp.hovered() { TEXT() } else { MUTED() };
        let edge = if resp.hovered() { MUTED() } else { EDGE() };
        ui.painter().rect_stroke(rect, TRAIT_H / 2.0, egui::Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);
        ui.painter().galley_with_override_text_color(rect.center() - galley.size() / 2.0, galley, ink);
    }
    resp
}

/// The text box a trait is typed in (in place of its pill). How it ended,
/// once it has: on Enter, Tab or a click elsewhere what's typed is kept; on
/// Escape it's as it was.
fn trait_field(ui: &mut egui::Ui, draft: &mut TraitDraft) -> Option<TraitDone> {
    let typed = ui.painter().layout_no_wrap(draft.text.clone(), egui::FontId::proportional(13.0), TEXT()).size().x;
    let width = (typed + 44.0).clamp(TRAIT_FIELD_MIN, ui.available_width().max(TRAIT_FIELD_MIN));
    // (In a ui of its own: a pill field leaves a row's cursor inside its
    // own right end, so the next pill sat on it.)
    let edit =
        egui::TextEdit::singleline(&mut draft.text).hint_text("A few words").char_limit(crate::roleplay::MAX_TRAIT_CHARS);
    let resp = ui
        .allocate_ui_with_layout(egui::vec2(width, TRAIT_H), egui::Layout::left_to_right(egui::Align::Center), |ui| {
            crate::settings::pill_field(ui, width, edit)
        })
        .inner;
    if !draft.focused {
        resp.request_focus();
        draft.focused = true;
        return None;
    }
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        return Some(TraitDone::Cancel);
    }
    resp.lost_focus().then_some(TraitDone::Keep)
}

/// Role play → Goals: the switch for the character's own goals, and the
/// goals themselves — the ones it's working on, then the finished ones,
/// newest first — each of which can be forgotten.
fn goals_section(ui: &mut egui::Ui, rp: &mut crate::roleplay::RoleplayState) {
    use crate::roleplay::GoalStatus;
    section(ui, "Goals", |ui| {
        let mut changed = false;
        ui.add_enabled_ui(!rp.load_failed, |ui| {
            row(
                ui,
                "Own goals",
                Some(
                    "The character sets goals of its own as the story unfolds — up to three at a \
                     time — works toward them, and marks each one achieved or failed. A note under \
                     the reply says when.",
                ),
                |ui| {
                    changed |= switch(ui, &mut rp.own_goals).changed();
                },
            );
            if rp.goals.is_empty() {
                if rp.own_goals {
                    row_sep(ui);
                    ui.label(
                        egui::RichText::new("No goals yet — the character sets them as the story gives it reasons.")
                            .color(MUTED())
                            .size(11.5),
                    );
                }
                return;
            }
            row_sep(ui);
            // Active first, in the order set; then the finished, newest first.
            let mut order: Vec<usize> = (0..rp.goals.len()).collect();
            order.sort_by_key(|&i| {
                let g = &rp.goals[i];
                match g.status {
                    GoalStatus::Active => (0, g.set),
                    _ => (1, -g.ended.unwrap_or(g.set)),
                }
            });
            let mut forget = None;
            for i in order {
                if goal_row(ui, &rp.goals[i]) {
                    forget = Some(i);
                }
            }
            if let Some(i) = forget {
                rp.goals.remove(i);
                changed = true;
            }
            if rp.goals.iter().any(|g| g.status != GoalStatus::Active) {
                ui.add_space(2.0);
                if ui
                    .button(egui::RichText::new("Clear finished").size(12.5))
                    .on_hover_text("Forget the achieved and failed goals")
                    .clicked()
                {
                    rp.goals.retain(|g| g.status == GoalStatus::Active);
                    changed = true;
                }
            }
        });
        if changed {
            rp.save();
        }
    });
}

/// One goal: its status icon, its words, and since when — or when and how it
/// ended. True when its forget button was clicked.
fn goal_row(ui: &mut egui::Ui, g: &crate::roleplay::Goal) -> bool {
    use crate::roleplay::GoalStatus;
    let date = |t: i64| crate::memory::format_date(t);
    let (icon, tint, when) = match g.status {
        GoalStatus::Active => (
            egui::include_image!("../icons/flag.svg"),
            ACCENT1(),
            format!("Working on it since {}", date(g.set)),
        ),
        GoalStatus::Achieved => (
            egui::include_image!("../icons/check_circle.svg"),
            egui::Color32::from_rgb(46, 160, 67),
            format!("Achieved {}", date(g.ended.unwrap_or(g.set))),
        ),
        GoalStatus::Failed => (
            egui::include_image!("../icons/x.svg"),
            MUTED(),
            format!("Failed {}", date(g.ended.unwrap_or(g.set))),
        ),
    };
    let meta = if g.outcome.is_empty() { when } else { format!("{when} — {}", g.outcome) };
    let mut forget = false;
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        ui.add(egui::Image::new(icon).fit_to_exact_size(egui::vec2(15.0, 15.0)).tint(tint));
        let w = (ui.available_width() - 32.0).max(80.0);
        ui.vertical(|ui| {
            ui.set_width(w);
            let ink = if g.status == GoalStatus::Failed { MUTED() } else { TEXT() };
            ui.add(egui::Label::new(egui::RichText::new(&g.text).color(ink).size(12.5)).wrap());
            ui.add(egui::Label::new(egui::RichText::new(meta).color(MUTED()).size(11.0)).wrap());
        });
        forget = crate::settings::clear_button(ui, "Forget this goal").clicked();
    });
    ui.add_space(4.0);
    forget
}

/// A one-line text box in a row's control slot that edits through a draft:
/// what's typed is handed to `commit` when the box loses focus (Enter, Tab
/// or a click elsewhere), so a half-typed value is never used or saved.
fn draft_field(
    ui: &mut egui::Ui,
    salt: &str,
    shown: &str,
    hint: &str,
    password: bool,
    commit: impl FnOnce(&str),
) -> egui::Response {
    let id = ui.id().with(("draft_field", salt));
    let mut draft: String = ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| shown.to_string());
    let resp = crate::settings::pill_field(
        ui,
        172.0,
        egui::TextEdit::singleline(&mut draft).hint_text(hint).password(password).char_limit(512),
    );
    if resp.lost_focus() {
        commit(draft.trim());
        ui.data_mut(|d| d.remove::<String>(id));
    } else if resp.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, draft));
    }
    resp
}

/// Web research: which service the chat searches (and its key or address),
/// how far a lookup goes, and where the notes go. Saved, to a file of their
/// own, as soon as something changes.
fn research_tab(ui: &mut egui::Ui, llm: &mut LlmState) {
    use crate::research::{Depth, Provider};
    let before = llm.research.clone();
    let r = &mut llm.research;

    section(ui, "Search with", |ui| {
        let about = match r.provider {
            Provider::Wikipedia => "Wikipedia's own search, articles read in full. Free, nothing to set up.",
            Provider::Tavily => "A web search made for AI. A free key covers 1,000 searches a month.",
            Provider::Brave => "Brave's independent web search. Needs a key from Brave's API site.",
            Provider::Searxng => "Your own SearXNG server: a private metasearch of many engines.",
        };
        row(ui, "Service", Some(about), |ui| {
            crate::settings::pill_dropdown(
                ui,
                "research_provider",
                &mut r.provider,
                &[
                    (Provider::Wikipedia, "Wikipedia", None, Some("Free, no setup")),
                    (Provider::Tavily, "Tavily", None, Some("Free key")),
                    (Provider::Brave, "Brave Search", None, Some("Needs a key")),
                    (Provider::Searxng, "SearXNG", None, Some("Your server")),
                ],
            );
        });
        row_sep(ui);
        let stored = if cfg!(windows) { "Stored encrypted on this computer." } else { "Stored on this computer." };
        match r.provider {
            Provider::Wikipedia => {
                let lang = r.wiki_lang.clone();
                row(ui, "Language", Some("Which Wikipedia to search: en, de, fr, es, ja…"), |ui| {
                    draft_field(ui, "wiki_lang", &lang, "en", false, |v| {
                        r.wiki_lang = if v.is_empty() { "en".to_string() } else { v.to_ascii_lowercase() };
                    });
                });
            }
            Provider::Tavily | Provider::Brave => {
                let (name, url) = if r.provider == Provider::Tavily {
                    ("Tavily", "https://app.tavily.com/home")
                } else {
                    ("Brave Search", "https://api-dashboard.search.brave.com/")
                };
                let has = r.has_key(r.provider);
                let hint = if has { "Key saved" } else { "Paste your key" };
                let sub = if has { format!("{stored} Paste a new one to replace it.") } else { stored.to_string() };
                let provider = r.provider;
                row(ui, "API key", Some(sub.as_str()), |ui| {
                    // Never shown: a saved key stays saved until another is
                    // pasted (or it's removed below).
                    draft_field(ui, &format!("key_{name}"), "", hint, true, |v| {
                        if !v.is_empty() {
                            match provider {
                                Provider::Tavily => r.set_tavily_key(v),
                                _ => r.set_brave_key(v),
                            }
                        }
                    });
                });
                ui.horizontal(|ui| {
                    crate::arrow_link(ui, &format!("Get a {name} key"), url, Some(11.0));
                    if has {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button(egui::RichText::new("Remove key").size(11.5)).clicked() {
                                match provider {
                                    Provider::Tavily => r.set_tavily_key(""),
                                    _ => r.set_brave_key(""),
                                }
                            }
                        });
                    }
                });
            }
            Provider::Searxng => {
                let addr = r.searxng_url.clone();
                row(
                    ui,
                    "Server address",
                    Some("Its JSON results must be on (search: formats: json in its settings.yml)."),
                    |ui| {
                        draft_field(ui, "searxng_url", &addr, "http://localhost:8888", false, |v| {
                            r.searxng_url = v.to_string();
                        });
                    },
                );
            }
        }
        // What's still missing (a key is asked for right above).
        let why = match r.provider {
            Provider::Tavily | Provider::Brave if !r.has_key(r.provider) => {
                Some("paste a key above to start searching".to_string())
            }
            _ => r.missing(),
        };
        if let Some(why) = why {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(format!("Not ready: {why}.")).color(egui::Color32::from_rgb(210, 70, 70)).size(11.5));
        }
        row_sep(ui);
        row(
            ui,
            "Depth",
            Some(match r.depth {
                Depth::Quick => "One round of searches, a few pages read. Quickest.",
                Depth::Thorough => "Up to three rounds: after each, the AI looks again for what's still missing.",
            }),
            |ui| {
                crate::settings::pill_dropdown(
                    ui,
                    "research_depth",
                    &mut r.depth,
                    &[
                        (Depth::Quick, "Quick", None, Some("One round of searches")),
                        (Depth::Thorough, "Thorough", None, Some("Up to three rounds")),
                    ],
                );
            },
        );
    });

    section(ui, "Notes", |ui| {
        let folder = r.notes_folder();
        row(ui, "Notes folder", Some(&folder.display().to_string()), |ui| {
            if ui.button(egui::RichText::new("Change…").size(12.5)).clicked()
                && let Some(dir) = rfd::FileDialog::new().set_directory(&folder).pick_folder()
            {
                r.notes_dir = dir.to_string_lossy().into_owned();
            }
            ui.add_space(6.0);
            if ui.button(egui::RichText::new("Open").size(12.5)).on_hover_text("Open the folder").clicked() {
                let _ = std::fs::create_dir_all(&folder);
                crate::scan::open_in_default(&folder);
            }
        });
        crate::settings::hint(
            ui,
            "Every lookup is saved here as a Markdown file: the question, the searches, the answer, \
             and the exact text the AI was given from each page. Open one from under its reply.",
        );
        if !r.notes_dir.trim().is_empty() && ui.button(egui::RichText::new("Use the default folder").size(11.5)).clicked() {
            r.notes_dir.clear();
        }
    });

    section(ui, "Privacy and safety", |ui| {
        for line in [
            "Only the searches the AI writes go to the service you picked; the pages it reads are \
             downloaded straight from this computer. Nothing else is sent.",
            "Web pages are treated as untrusted. Nothing they say can reach this computer or your home \
             network, text they hide from readers is dropped, and a reply written from them can't save \
             memories, keep pictures or change tags — it links only to the pages it read.",
            "Off in Tag images. Notes are plain files: edit or delete them whenever you like.",
        ] {
            ui.label(egui::RichText::new(line).color(TEXT()).size(12.0));
            ui.add_space(4.0);
        }
    });

    if llm.research != before {
        llm.research.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roleplay::{GrowthLine, RoleplayState, Trait};
    use crate::slider::rig::Rig;

    /// The character under edit, and how often the editors said it changed
    /// (when it's saved).
    struct Editing {
        rp: RoleplayState,
        saves: usize,
    }

    fn scene(ui: &mut egui::Ui, s: &mut Editing) {
        ui.set_width(520.0);
        if persona_undo(ui, &mut s.rp) {
            s.saves += 1;
        }
        if traits_editor(ui, &mut s.rp) {
            s.saves += 1;
        }
    }

    fn editing(traits: &[(&str, bool)]) -> Rig<Editing> {
        let traits = traits.iter().map(|&(text, by_ai)| Trait { text: text.into(), by_ai, added: 1 }).collect();
        let rp = RoleplayState { ai_name: "Mira".into(), traits, ..Default::default() };
        let mut rig = Rig::new(Editing { rp, saves: 0 }, scene);
        rig.idle();
        rig
    }

    fn traits(rig: &Rig<Editing>) -> Vec<(&str, bool)> {
        rig.state.rp.traits.iter().map(|t| (t.text.as_str(), t.by_ai)).collect()
    }

    /// Where a trait's pill has its words, and its × after them.
    fn pill(rig: &Rig<Editing>, text: &str) -> (egui::Pos2, egui::Pos2) {
        let words = rig.texts(text);
        assert_eq!(words.len(), 1, "{text:?} has a pill");
        (words[0].center(), egui::pos2(words[0].right() + 11.0, words[0].center().y))
    }

    fn select_all(rig: &mut Rig<Editing>) {
        rig.frame(vec![egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        }]);
    }

    /// The traits are pills: a click on one rewrites it (it's the user's
    /// from then on), its × removes it, the last pill adds one.
    #[test]
    fn traits_are_edited_as_pills() {
        let mut rig = editing(&[("shy", false), ("loves the rain", true)]);
        assert_eq!(rig.texts("+  Add").len(), 1);

        // Rewritten: the pill turns into a box holding its words.
        let (words, _) = pill(&rig, "loves the rain");
        rig.click(words);
        rig.idle();
        assert_eq!(rig.state.saves, 0, "nothing changes until it's typed");
        select_all(&mut rig);
        rig.type_text("loves a storm");
        assert_eq!(traits(&rig), [("shy", false), ("loves the rain", true)], "not while it's being typed");
        rig.key(egui::Key::Enter);
        rig.idle();
        assert_eq!(traits(&rig), [("shy", false), ("loves a storm", false)]);
        assert_eq!(rig.state.saves, 1);
        assert_eq!(rig.texts("loves a storm").len(), 1, "a pill again");

        // Escape leaves it as it was.
        let (words, _) = pill(&rig, "shy");
        rig.click(words);
        rig.idle();
        select_all(&mut rig);
        rig.type_text("bold");
        // (The box has its room: the next pill starts after it — past the
        // box's end, the gap and the pill's own edge.)
        let (typed, next) = (rig.texts("bold")[0], rig.texts("loves a storm")[0]);
        assert!(next.left() >= typed.left() - 14.0 + TRAIT_FIELD_MIN + 6.0 + 12.0 - 1.0, "{typed:?} then {next:?}");
        rig.key(egui::Key::Escape);
        rig.idle();
        assert_eq!(traits(&rig), [("shy", false), ("loves a storm", false)]);
        assert_eq!(rig.state.saves, 1);

        // Added with the last pill; one it already has isn't added twice.
        for (typed, saves) in [("quick to laugh", 2), ("SHY", 2)] {
            let add = rig.texts("+  Add")[0].center();
            rig.click(add);
            rig.idle();
            rig.type_text(typed);
            rig.key(egui::Key::Enter);
            rig.idle();
            assert_eq!(rig.state.saves, saves, "{typed}");
        }
        assert_eq!(traits(&rig), [("shy", false), ("loves a storm", false), ("quick to laugh", false)]);

        // Removed by its ×; emptied, a trait goes too.
        let (_, x) = pill(&rig, "loves a storm");
        rig.click(x);
        rig.idle();
        assert_eq!(traits(&rig), [("shy", false), ("quick to laugh", false)]);
        let (words, _) = pill(&rig, "shy");
        rig.click(words);
        rig.idle();
        select_all(&mut rig);
        rig.key(egui::Key::Backspace);
        rig.key(egui::Key::Enter);
        rig.idle();
        assert_eq!(traits(&rig), [("quick to laugh", false)]);
        assert_eq!(rig.state.saves, 4);
    }

    /// With none, the first pill says what it's for; at the most a
    /// character keeps, there's no pill to add another.
    #[test]
    fn the_add_pill_comes_and_goes() {
        let rig = editing(&[]);
        assert_eq!(rig.texts("+  Add a trait").len(), 1);
        let full: Vec<String> = (0..crate::roleplay::MAX_TRAITS).map(|i| format!("trait {i}")).collect();
        let full: Vec<(&str, bool)> = full.iter().map(|t| (t.as_str(), false)).collect();
        let rig = editing(&full);
        assert!(rig.texts("+  Add").is_empty() && rig.texts("+  Add a trait").is_empty());
        assert_eq!(rig.texts("trait 15").len(), 1);
    }

    /// Once the character has changed its persona, the line under it says
    /// when — and Undo takes it back, a change at a time.
    #[test]
    fn a_change_to_the_persona_can_be_undone() {
        let mut rig = editing(&[]);
        assert!(rig.texts("Undo").is_empty(), "nothing to undo yet");
        rig.state.rp.persona = "Mira is a shy sailor. She has never left the harbour.".into();
        rig.state.rp.apply_growth(&[GrowthLine::PersonaGone("[P2]".into())], 1_795_000_000);
        rig.state.rp.apply_growth(&[GrowthLine::PersonaAdd("She swam the bay.".into())], 1_795_000_100);
        rig.idle();
        let said = format!("Mira changed this on {}.", crate::memory::format_date(1_795_000_100));
        assert_eq!(rig.texts(&said).len(), 1);

        let undo = rig.texts("Undo")[0].center();
        rig.click(undo);
        rig.idle();
        assert_eq!(rig.state.rp.persona, "Mira is a shy sailor.");
        assert_eq!(rig.state.saves, 1);
        let undo = rig.texts("Undo")[0].center();
        rig.click(undo);
        rig.idle();
        assert_eq!(rig.state.rp.persona, "Mira is a shy sailor. She has never left the harbour.");
        assert!(rig.texts("Undo").is_empty() && rig.state.saves == 2);
    }
}
