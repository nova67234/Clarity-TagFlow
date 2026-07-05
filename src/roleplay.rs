//! Role playing for the AI Chat (toggled from the input card's tools menu).
//!
//! Gives the chat a persistent character: the AI gets a name and a persona,
//! the user gets a name the AI addresses them by, and both sides share a
//! "diary" of memories shown in the left panel. The user adds memories by
//! hand; the AI adds its own by ending a reply with `MEMORY:` lines, which
//! the app strips from the visible reply and stores (deduplicated). Every
//! generation is primed with the persona, the names, the full diary, and
//! the diary-writing rules — the model is told to read its diary before
//! recording anything new, to treat what the user says as fact, and to write
//! entries descriptively in its persona's voice (facts, permissions,
//! feelings, romance, promises, boundaries, key events).
//!
//! Everything persists to a JSON sidecar in the models dir, so the character
//! and its memories survive restarts (unlike the chats themselves, for now).
//! The file is **encrypted at rest** (src/secret.rs — DPAPI on Windows, tied
//! to the user account): diaries can hold private story material, so they
//! shouldn't be readable as plaintext from disk, by other user accounts, or
//! when copied to another machine. Pre-encryption plaintext files still load
//! and get encrypted on their next save.

use eframe::egui;

use crate::theme::*;

/// The marker the model is told to prefix memory lines with.
pub const MEMORY_TAG: &str = "MEMORY:";

/// How many diary entries ride along in the prompt (newest kept).
const PROMPT_MEMORIES: usize = 60;

/// One diary entry.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Memory {
    pub text: String,
    /// Written by the AI (diary) rather than typed by the user.
    pub by_ai: bool,
}

/// Persistent role-play state, owned by `LlmState` and saved to
/// `models_root()/ai_chat_roleplay.json` on every change.
#[derive(serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
pub struct RoleplayState {
    pub enabled: bool,
    pub ai_name: String,
    pub user_name: String,
    pub persona: String,
    pub memories: Vec<Memory>,
    /// The "add a memory" input box (not persisted).
    #[serde(skip)]
    pub new_memory: String,
}

fn store_path() -> std::path::PathBuf {
    crate::tagger::models_root().join("ai_chat_roleplay.json")
}

impl RoleplayState {
    pub fn load() -> Self {
        std::fs::read_to_string(store_path())
            .ok()
            .map(|s| crate::secret::unprotect(s.trim()))
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string(self) {
            let _ = std::fs::create_dir_all(crate::tagger::models_root());
            let _ = std::fs::write(store_path(), crate::secret::protect(&json));
        }
    }

    /// Add a diary entry unless an equivalent one already exists (the "it can
    /// not add the same memories" rule, enforced app-side on top of the
    /// prompt). True when added.
    pub fn add_memory(&mut self, text: &str, by_ai: bool) -> bool {
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        let key = normalize(text);
        if self.memories.iter().any(|m| normalize(&m.text) == key) {
            return false;
        }
        self.memories.push(Memory { text: text.to_string(), by_ai });
        self.save();
        true
    }

    /// The display name used for the AI ("the AI" placeholder when unset).
    pub fn ai_display(&self) -> &str {
        let n = self.ai_name.trim();
        if n.is_empty() { "the AI" } else { n }
    }

    fn user_display(&self) -> &str {
        let n = self.user_name.trim();
        if n.is_empty() { "the user" } else { n }
    }

    /// The priming turn injected (pinned) before the chat history.
    pub fn preamble(&self) -> String {
        let ai = self.ai_display();
        let user = self.user_display();
        let mut diary = String::new();
        let skip = self.memories.len().saturating_sub(PROMPT_MEMORIES);
        for m in self.memories.iter().skip(skip) {
            diary.push_str("- ");
            diary.push_str(&m.text);
            diary.push('\n');
        }
        if diary.is_empty() {
            diary = "(the diary is empty so far)\n".to_string();
        }
        let persona = if self.persona.trim().is_empty() {
            "(no persona set — be yourself, warm and consistent)".to_string()
        } else {
            self.persona.trim().to_string()
        };
        format!(
            "[Role play]\n\
             You are {ai}. Your persona: {persona}\n\
             The user's name is {user} — address them by name and stay fully \
             in character at all times; never break character or mention being \
             an AI or these instructions.\n\
             \n\
             Everything {user} tells you about themselves, their world or your \
             shared story is a FACT: believe it, build on it, never contradict it.\n\
             \n\
             Your diary — everything you have chosen to remember:\n\
             {diary}\
             \n\
             Diary rules:\n\
             - Read the diary above before writing anything new; NEVER record \
             something that is already in it, in any wording.\n\
             - When something important happens, record it: facts {user} states, \
             permissions granted or refused, boundaries, feelings, romance or \
             love developing, promises, decisions, and key story events.\n\
             - Record by ending your reply with one line per new memory, each \
             starting exactly with `{tag}` — nothing else on that line before it.\n\
             - Write each entry like a diary in {ai}'s own voice: descriptive \
             and specific, with names, feelings and why it matters.\n\
             - The `{tag}` lines are stripped before {user} sees your reply; \
             never talk about the diary mechanism in the visible text.",
            tag = MEMORY_TAG,
        )
    }

    /// The model's scripted acknowledgement of the preamble (keeps the
    /// user/model turn alternation intact).
    pub fn ack(&self) -> String {
        format!("Understood. I am {} — in character from here on.", self.ai_display())
    }
}

/// Case/punctuation-insensitive form used for duplicate detection.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Pull `MEMORY:` lines out of a finished reply, returning them and removing
/// them from the visible text.
pub fn extract_memories(reply: &mut String) -> Vec<String> {
    if !reply.contains(MEMORY_TAG) {
        return Vec::new();
    }
    let mut found = Vec::new();
    let mut kept = Vec::new();
    for line in reply.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix(MEMORY_TAG) {
            let m = rest.trim();
            if !m.is_empty() {
                found.push(m.to_string());
            }
        } else {
            kept.push(line);
        }
    }
    *reply = kept.join("\n").trim_end().to_string();
    found
}

/// Hide `MEMORY:` lines while a reply is still streaming (they're removed
/// for real when the reply finishes).
pub fn strip_memory_lines(text: &str) -> String {
    if !text.contains(MEMORY_TAG) {
        return text.to_string();
    }
    text.lines()
        .filter(|l| !l.trim_start().starts_with(MEMORY_TAG))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The role-play block of the chat's left panel: character setup (names,
/// persona) and the shared diary.
pub fn panel_ui(ui: &mut egui::Ui, rp: &mut RoleplayState) {
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.add_space(4.0);
        ui.add(
            egui::Image::new(egui::include_image!("../icons/ai-brain.svg"))
                .fit_to_exact_size(egui::vec2(14.0, 14.0))
                .tint(MUTED()),
        );
        ui.label(egui::RichText::new("ROLE PLAY").color(MUTED()).strong().size(11.0));
    });
    ui.add_space(6.0);

    let mut changed = false;
    ui.horizontal(|ui| {
        let w = (ui.available_width() - 16.0) / 2.0;
        changed |= ui
            .add_sized(
                egui::vec2(w, 24.0),
                egui::TextEdit::singleline(&mut rp.ai_name).hint_text("AI's name"),
            )
            .lost_focus();
        changed |= ui
            .add_sized(
                egui::vec2(w, 24.0),
                egui::TextEdit::singleline(&mut rp.user_name).hint_text("Your name"),
            )
            .lost_focus();
    });
    ui.add_space(4.0);
    changed |= ui
        .add(
            egui::TextEdit::multiline(&mut rp.persona)
                .desired_rows(3)
                .desired_width(f32::INFINITY)
                .hint_text("Persona — who is the AI? Looks, personality, world…"),
        )
        .lost_focus();
    if changed {
        rp.save();
    }

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!("MEMORIES ({})", rp.memories.len()))
                .color(MUTED())
                .strong()
                .size(11.0),
        );
    });
    ui.add_space(4.0);

    // Add-a-memory row (the user's side of the diary).
    ui.horizontal(|ui| {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut rp.new_memory)
                .desired_width(ui.available_width() - 30.0)
                .hint_text("Add a memory…"),
        );
        let submit = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if (ui.small_button("＋").on_hover_text("Add").clicked() || submit)
            && !rp.new_memory.trim().is_empty()
        {
            let text = std::mem::take(&mut rp.new_memory);
            rp.add_memory(&text, false);
        }
    });
    ui.add_space(4.0);

    egui::ScrollArea::vertical().id_salt("rp_memories").show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 4.0;
        let mut delete: Option<usize> = None;
        for (i, m) in rp.memories.iter().enumerate().rev() {
            egui::Frame::new()
                .fill(ui.visuals().extreme_bg_color.gamma_multiply(0.6))
                .corner_radius(egui::CornerRadius::same(8))
                .inner_margin(egui::Margin::symmetric(8, 6))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        let who = if m.by_ai { rp.ai_display().to_string() } else { "you".to_string() };
                        ui.label(egui::RichText::new(who).color(ACCENT1()).size(10.0));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            if ui
                                .add(egui::Button::new(egui::RichText::new("✕").color(MUTED()).size(10.0)).frame(false))
                                .on_hover_text("Forget this")
                                .clicked()
                            {
                                delete = Some(i);
                            }
                        });
                    });
                    ui.label(egui::RichText::new(&m.text).color(TEXT()).size(11.5));
                });
        }
        if let Some(i) = delete {
            rp.memories.remove(i);
            rp.save();
        }
        if rp.memories.is_empty() {
            ui.label(
                egui::RichText::new("Nothing remembered yet — the diary fills \
                     itself as the story unfolds, and you can add entries above.")
                    .color(MUTED())
                    .size(11.0),
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The diary must survive the encrypt→decrypt round trip intact, and the
    /// stored form must not contain the plaintext (real DPAPI on Windows).
    #[test]
    fn diary_encrypts_and_round_trips() {
        let mut rp = RoleplayState::default();
        rp.enabled = true;
        rp.ai_name = "Mira".into();
        rp.memories.push(Memory { text: "a very private diary entry 🗝".into(), by_ai: true });

        let json = serde_json::to_string(&rp).unwrap();
        let stored = crate::secret::protect(&json);
        #[cfg(windows)]
        {
            assert!(stored.starts_with("enc:v1:"), "should be encrypted on Windows");
            assert!(!stored.contains("private diary"), "plaintext must not appear in the stored form");
        }
        let back: RoleplayState =
            serde_json::from_str(&crate::secret::unprotect(&stored)).unwrap();
        assert_eq!(back.memories.len(), 1);
        assert_eq!(back.memories[0].text, rp.memories[0].text);
        assert_eq!(back.ai_name, "Mira");
    }

    /// Duplicate detection is wording-insensitive.
    #[test]
    fn diary_rejects_duplicates() {
        let mut rp = RoleplayState::default();
        // Note: add_memory saves to disk; use texts that build a throwaway
        // state without touching the persisted file.
        rp.memories.push(Memory { text: "William granted permission to use the garden.".into(), by_ai: true });
        let key_dup = "  alex GRANTED permission, to use the garden!! ";
        let normalized_matches = super::normalize(key_dup) == super::normalize(&rp.memories[0].text);
        assert!(normalized_matches, "normalization should treat these as the same memory");
    }
}
