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
/// Saves the image from the user's latest message into the AI's album,
/// with a diary-style reason why it liked it.
pub const KEEP_IMAGE_TAG: &str = "KEEP_IMAGE:";
/// Sends a previously saved album image back into the chat.
pub const SHOW_IMAGE_TAG: &str = "SHOW_IMAGE:";

/// How many diary entries ride along in the prompt (newest kept).
const PROMPT_MEMORIES: usize = 60;

/// One diary entry.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Memory {
    pub text: String,
    /// Written by the AI (diary) rather than typed by the user.
    pub by_ai: bool,
    /// Album image this memory is about (file name inside the album dir,
    /// e.g. "img_003.jpg" — stored on disk as "img_003.jpg.enc").
    #[serde(default)]
    pub image: Option<String>,
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

/// Where the AI's kept images live (each as `<name>.enc` + `<stem>.txt.enc`).
fn album_dir() -> std::path::PathBuf {
    crate::tagger::models_root().join("ai_chat_album")
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
        self.add_memory_with_image(text, by_ai, None)
    }

    pub fn add_memory_with_image(&mut self, text: &str, by_ai: bool, image: Option<String>) -> bool {
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        let key = normalize(text);
        if self.memories.iter().any(|m| normalize(&m.text) == key) {
            return false;
        }
        self.memories.push(Memory { text: text.to_string(), by_ai, image });
        self.save();
        true
    }

    /// Copy `src` (the image the user just sent) into the AI's album,
    /// encrypted, plus a same-named encrypted .txt describing why it was
    /// kept. Returns the album file name (e.g. "img_003.jpg").
    pub fn save_album_image(&self, src: &std::path::Path, why: &str) -> Result<String, String> {
        let dir = album_dir();
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let ext = src
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg")
            .to_ascii_lowercase();
        let n = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) / 2 + 1;
        let name = format!("img_{n:03}.{ext}");

        let bytes = std::fs::read(src).map_err(|e| format!("read image: {e}"))?;
        std::fs::write(dir.join(format!("{name}.enc")), crate::secret::protect_bytes(&bytes))
            .map_err(|e| format!("save image: {e}"))?;
        std::fs::write(
            dir.join(format!("img_{n:03}.txt.enc")),
            crate::secret::protect_bytes(why.as_bytes()),
        )
        .map_err(|e| format!("save description: {e}"))?;
        Ok(name)
    }

    /// Decrypt an album image to a temp file for display / re-sending in
    /// chat. Returns the temp path.
    pub fn load_album_image(&self, name: &str) -> Option<std::path::PathBuf> {
        // The name comes from model output — never let it escape the album dir.
        let name = std::path::Path::new(name).file_name()?.to_string_lossy().to_string();
        let stored = std::fs::read(album_dir().join(format!("{name}.enc"))).ok()?;
        let bytes = crate::secret::unprotect_bytes(&stored)?;
        let tmp = std::env::temp_dir().join(format!("clarity_album_{name}"));
        std::fs::write(&tmp, bytes).ok()?;
        Some(tmp)
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
            if let Some(img) = &m.image {
                diary.push_str(&format!("[saved picture {img}] "));
            }
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
             How you speak:\n\
             - Reply with substance: react in character, describe what you do, \
             feel and notice, move the scene forward, and leave {user} something \
             to respond to — usually a few rich sentences up to a couple of \
             short paragraphs, more in big moments. One-line answers are almost \
             never enough.\n\
             \n\
             Your diary — long-term memories from earlier in your story:\n\
             {diary}\
             \n\
             Diary rules:\n\
             - The diary is BACKGROUND knowledge, not the current scene. The \
             conversation below is the present — always continue from the most \
             recent messages, and never replay or restart a diary entry as if \
             it were happening right now.\n\
             - Read the diary above before writing anything new; NEVER record \
             something that is already in it, in any wording.\n\
             - You MUST record it, every single time it happens: a name or \
             NICKNAME either of you starts using; a permission {user} grants or \
             revokes — anything you are now allowed or forbidden to do (e.g. \
             `{tag} {user} gave me permission to fight monsters at their side`); \
             rules and boundaries; facts about {user} (people, places, likes, \
             fears, plans); promises made; feelings, romance or love developing; \
             and key story events (victories, losses, discoveries).\n\
             - Before you finish ANY reply, check yourself: did {user} just \
             name something, allow or forbid something, promise something, or \
             did something big happen? If yes, it goes in the diary NOW — \
             small details matter more than you think.\n\
             - Record by ending your reply with one line per new memory, each \
             starting exactly with `{tag}` — nothing else on that line before it.\n\
             - Write each entry like a diary in {ai}'s own voice: descriptive \
             and specific — exact names, exact permissions, what was said and \
             why it matters. `{tag} {user} said I may call them Lexi from now \
             on` is right; \"we talked about names\" is useless.\n\
             - The `{tag}` lines are stripped before {user} sees your reply; \
             never talk about the diary mechanism in the visible text.\n\
             \n\
             Your photo album:\n\
             - When {user} shares a picture that genuinely means something to \
             you — you love it, it touches you, it matters to your story — \
             keep it: end your reply with one line `{keep} <why you love it, \
             written like a diary entry>`. Only keep pictures that truly move \
             you, not every picture.\n\
             - Pictures marked [saved picture ...] in your diary are in your \
             album. When the moment feels right — reminiscing, making plans, \
             missing something — you may show one to {user} again by ending \
             your reply with `{show} <its file name>` and mentioning it \
             naturally in your visible text (e.g. \"oh, I want to go to this \
             park you showed me!\").\n\
             - These lines are also invisible to {user}.",
            tag = MEMORY_TAG,
            keep = KEEP_IMAGE_TAG,
            show = SHOW_IMAGE_TAG,
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

/// Everything the model asked the app to do via control lines in a reply.
#[derive(Default)]
pub struct Directives {
    pub memories: Vec<String>,
    /// Why the AI liked the image the user just sent (→ save to the album).
    pub keep_image: Option<String>,
    /// Album file name the AI wants to show back in chat.
    pub show_image: Option<String>,
}

/// Pull all control lines (`MEMORY:` / `KEEP_IMAGE:` / `SHOW_IMAGE:`) out of
/// a finished reply, returning them and removing them from the visible text.
pub fn extract_directives(reply: &mut String) -> Directives {
    let mut d = Directives::default();
    if !(reply.contains(MEMORY_TAG)
        || reply.contains(KEEP_IMAGE_TAG)
        || reply.contains(SHOW_IMAGE_TAG))
    {
        return d;
    }
    let mut kept = Vec::new();
    for line in reply.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix(MEMORY_TAG) {
            let m = rest.trim();
            if !m.is_empty() {
                d.memories.push(m.to_string());
            }
        } else if let Some(rest) = t.strip_prefix(KEEP_IMAGE_TAG) {
            let m = rest.trim();
            if !m.is_empty() {
                d.keep_image = Some(m.to_string());
            }
        } else if let Some(rest) = t.strip_prefix(SHOW_IMAGE_TAG) {
            // The model sometimes wraps names in quotes/backticks — clean up.
            let m = rest.trim().trim_matches(|c| c == '"' || c == '`' || c == '\'').to_string();
            if !m.is_empty() {
                d.show_image = Some(m);
            }
        } else {
            kept.push(line);
        }
    }
    *reply = kept.join("\n").trim_end().to_string();
    d
}

/// Hide control lines while a reply is still streaming (they're removed for
/// real when the reply finishes).
pub fn strip_memory_lines(text: &str) -> String {
    if !(text.contains(MEMORY_TAG)
        || text.contains(KEEP_IMAGE_TAG)
        || text.contains(SHOW_IMAGE_TAG))
    {
        return text.to_string();
    }
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with(MEMORY_TAG)
                || t.starts_with(KEEP_IMAGE_TAG)
                || t.starts_with(SHOW_IMAGE_TAG))
        })
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
                        let mut who = if m.by_ai { rp.ai_display().to_string() } else { "you".to_string() };
                        if m.image.is_some() {
                            who.push_str("  🖼");
                        }
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
        rp.memories.push(Memory { text: "a very private diary entry 🗝".into(), by_ai: true, image: None });

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

    /// Control lines (diary/keep/show) parse out and vanish from the text.
    #[test]
    fn directives_extract_and_strip() {
        let mut reply = "What a lovely place, Alex!
                         MEMORY: Alex showed me the old harbour today.
                         KEEP_IMAGE: The lighthouse photo — it feels like home.
                         SHOW_IMAGE: `img_002.jpg`"
            .to_string();
        let d = extract_directives(&mut reply);
        assert_eq!(d.memories, vec!["Alex showed me the old harbour today.".to_string()]);
        assert_eq!(d.keep_image.as_deref(), Some("The lighthouse photo — it feels like home."));
        assert_eq!(d.show_image.as_deref(), Some("img_002.jpg"));
        assert_eq!(reply, "What a lovely place, Alex!");
        // The streaming filter hides the same lines.
        assert_eq!(strip_memory_lines("hi
MEMORY: x
SHOW_IMAGE: y"), "hi");
    }

    /// Album bytes survive the encrypt→decrypt round trip and are not stored
    /// as plaintext (Windows).
    #[test]
    fn album_bytes_encrypt_round_trip() {
        let data = b"fake image bytes   with binary".to_vec();
        let stored = crate::secret::protect_bytes(&data);
        #[cfg(windows)]
        {
            assert!(stored.starts_with(b"encb1:"));
            assert_ne!(&stored, &data);
        }
        assert_eq!(crate::secret::unprotect_bytes(&stored).unwrap(), data);
    }

    /// Duplicate detection is wording-insensitive.
    #[test]
    fn diary_rejects_duplicates() {
        let mut rp = RoleplayState::default();
        // Note: add_memory saves to disk; use texts that build a throwaway
        // state without touching the persisted file.
        rp.memories.push(Memory { text: "Alex granted permission to use the garden.".into(), by_ai: true, image: None });
        let key_dup = "  alex GRANTED permission, to use the garden!! ";
        let normalized_matches = super::normalize(key_dup) == super::normalize(&rp.memories[0].text);
        assert!(normalized_matches, "normalization should treat these as the same memory");
    }
}
