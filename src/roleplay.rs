//! Role playing for the AI Chat (toggled from the input card's Tools menu,
//! set up in AI Settings → Role play).
//!
//! Gives the chat a persistent character: the AI gets a name and a persona,
//! the user gets a name the AI addresses them by, and the character keeps
//! memories of its own. Those live in their own memory store
//! (`MemoryKind::Roleplay`, src/memory.rs): the same format, 3D map and
//! settings as the AI's main memory, but a separate folder, so the story
//! never leaks into normal chats. The AI adds memories by ending a reply
//! with `MEMORY: [Category] …` lines, which the app strips from the visible
//! reply and files under that category (deduplicated); the user adds, edits
//! and forgets them in the memory map. Every generation is primed with the
//! persona, the names, the newest memories, and the rules for writing them —
//! the model is told to read its memories before recording anything new, to
//! treat what the user says as fact, and to write entries descriptively in
//! its persona's voice (facts, permissions, feelings, romance, promises,
//! boundaries, key events).
//!
//! The character itself (names, persona, on/off) persists to a JSON sidecar
//! in the models dir. On Windows it is **encrypted at rest** (src/secret.rs —
//! DPAPI, tied to the user account); on macOS/Linux it is only obfuscated (a
//! fixed-key XOR). A file that exists but can't be read back (another PC, a
//! reinstalled Windows, a different account) is never overwritten — see
//! `load_failed`. Older versions kept the diary in that file too; it moves
//! into the role-play memory store on first load ([`migrate_diary`]).

use crate::memory::{MemoryKind, MemoryStore};

/// The marker the model is told to prefix memory lines with.
pub const MEMORY_TAG: &str = "MEMORY:";
/// Saves the image from the user's latest message into the AI's album,
/// with a diary-style reason why it liked it.
pub const KEEP_IMAGE_TAG: &str = "KEEP_IMAGE:";
/// Sends a previously saved album image back into the chat.
pub const SHOW_IMAGE_TAG: &str = "SHOW_IMAGE:";

/// How many memories ride along in the prompt (newest kept).
const PROMPT_MEMORIES: usize = 75;

/// The categories the AI files its memories under (`MEMORY: [Names] …`);
/// they become the role-play memory map's topics.
pub const TOPICS: [&str; 7] = ["Names", "Permissions", "Boundaries", "Facts", "Promises", "Feelings", "Story"];
/// Memories about a picture the AI kept in its album.
pub const PICTURES_TOPIC: &str = "Pictures";
/// Where a memory without a (known) category goes.
const DEFAULT_TOPIC: &str = "Story";
/// Memories carried over from the old diary, which had no categories.
const LEGACY_TOPIC: &str = "Diary";

/// One entry of the old diary (before role play got its own memory store);
/// only read to move it over — see [`migrate_diary`].
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
    /// The old diary. Empty once [`migrate_diary`] has moved it into the
    /// role-play memory store.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub memories: Vec<Memory>,
    /// A diary file exists but couldn't be decrypted/parsed (e.g. DPAPI on a
    /// new PC or user account). Saving is refused so the blank state never
    /// overwrites the real diary; the panel offers to set it aside instead.
    #[serde(skip)]
    pub load_failed: bool,
}

fn store_path() -> std::path::PathBuf {
    crate::tagger::models_root().join("ai_chat_roleplay.json")
}

/// Delete every decrypted album copy (see src/album.rs).
pub fn clear_album_temp() {
    crate::album::clear_temp();
}

impl RoleplayState {
    pub fn load() -> Self {
        // Decrypted album pictures from a previous run (a crash skips the
        // exit-time cleanup) — never leave them lying around.
        clear_album_temp();
        let Ok(stored) = std::fs::read_to_string(store_path()) else {
            return Self::default(); // no diary yet
        };
        if stored.trim().is_empty() {
            return Self::default();
        }
        // `unprotect` yields "" for ciphertext this account can't decrypt.
        match serde_json::from_str(&crate::secret::unprotect(stored.trim())) {
            Ok(state) => state,
            Err(_) => Self { load_failed: true, ..Default::default() },
        }
    }

    pub fn save(&self) {
        if self.load_failed {
            return; // never clobber a diary we couldn't read
        }
        if let Ok(json) = serde_json::to_string(self) {
            let _ = std::fs::create_dir_all(crate::tagger::models_root());
            // Owner-only on Linux/macOS, where `protect` is just obfuscation.
            let _ = crate::secret::write_private(store_path(), crate::secret::protect(&json));
        }
    }

    /// Copy `src` (the image the user just sent) into the AI's album,
    /// encrypted, with why it was kept. Returns the album file name (e.g.
    /// "img_003.jpg").
    pub fn save_album_image(&self, src: &std::path::Path, why: &str) -> Result<String, String> {
        crate::album::save(MemoryKind::Roleplay, src, why)
    }

    /// Decrypt an album image to a temp file for display / re-sending in
    /// chat. Returns the temp path.
    pub fn load_album_image(&self, name: &str) -> Option<std::path::PathBuf> {
        crate::album::load(MemoryKind::Roleplay, name)
    }

    /// Give up on an unreadable diary: move it aside (kept as a `.bak`, in
    /// case the original account/PC can still read it) and start fresh.
    pub fn start_new_diary(&mut self) {
        let path = store_path();
        let _ = std::fs::rename(&path, path.with_extension("json.bak"));
        *self = Self { enabled: self.enabled, ..Default::default() };
        self.save();
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

    /// The priming turn injected (pinned) before the chat history: the
    /// character, and its newest memories from the role-play store, with
    /// the default (brief) thinking rule — what the tests prime with.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn preamble(&self, memories: &MemoryStore) -> String {
        self.preamble_with(memories, false)
    }

    /// The priming turn, with the thinking rule matched to the chat's
    /// thinking level. Normally the character is told to think briefly —
    /// a rule written for the since-removed Gemma 4 26B A4B, which spent
    /// whole replies privately redrafting and then answered nothing. At
    /// High / Max (`deep_thinking`) the user asked for more thought, so the
    /// rule lets it think things through; the worker's forced close still
    /// guarantees an answer.
    pub fn preamble_with(&self, memories: &MemoryStore, deep_thinking: bool) -> String {
        let thinking = if deep_thinking {
            "- Think it through: before answering you may reason privately about \
             the scene, what you know and how your character feels — then answer \
             in character. Keep your reasoning as notes, not drafts of the reply.\n"
        } else {
            "- Think briefly: when you reason privately before answering, a few \
             short notes are enough — decide, then answer. Never write drafts \
             or revisions of your reply in your head.\n"
        };
        let ai = self.ai_display();
        let user = self.user_display();
        let mut diary = String::new();
        for (topic, m) in memories.recent(PROMPT_MEMORIES) {
            use std::fmt::Write as _;
            let _ = write!(diary, "- [{topic}] ");
            if let Some(img) = &m.image {
                let _ = write!(diary, "[saved picture {img}] ");
            }
            diary.push_str(if m.text.trim().is_empty() { &m.title } else { &m.text });
            diary.push('\n');
        }
        if diary.is_empty() {
            diary = "(the diary is empty so far)\n".to_string();
        }
        let categories = TOPICS.join(", ");
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
             {thinking}\
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
             fighting monsters at their side); rules and boundaries; facts about {user} (people, places, likes, \
             fears, plans); promises made; feelings, romance or love developing; \
             and key story events (victories, losses, discoveries).\n\
             - Before you finish ANY reply, check yourself: did {user} just \
             name something, allow or forbid something, promise something, or \
             did something big happen? If yes, it goes in the diary NOW — \
             small details matter more than you think.\n\
             - Record by ending your reply with one line per new memory, each \
             starting exactly with `{tag}` — nothing else on that line before it — \
             followed by its category in square brackets, one of: {categories}. \
             For example `{tag} [Permissions] {user} gave me permission to fight \
             monsters at their side`.\n\
             - Write each entry like a diary in {ai}'s own voice: descriptive \
             and specific — exact names, exact permissions, what was said and \
             why it matters. `{tag} [Names] {user} said I may call them Billie \
             from now on` is right; \"we talked about names\" is useless.\n\
             - The `{tag}` lines are stripped before {user} sees your reply; \
             never talk about the diary mechanism in the visible text.\n\
             \n\
             Your photo album:\n\
             - When {user} shares a picture that genuinely means something to \
             you — you love it, it touches you, it matters to your story — \
             keep it: end your reply with one line `{keep} <why you love it, \
             written like a diary entry>`. Only keep pictures that truly move \
             you.\n\
             - Pictures marked [saved picture ...] in your diary are in your \
             album. When the moment feels right — reminiscing, making plans, \
             missing something — you may show one to {user} again by ending \
             your reply with `{show} <its file name>` and mentioning it \
             naturally in your visible text (e.g. \"oh, I want to go to this \
             park you showed me!\").\n\
             - These lines are also invisible to {user}.",
            thinking = thinking,
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

/// A short title for a memory written as one diary sentence: its first
/// sentence, cut to eight words ("Andrew said I may call them Billie…").
pub fn title_from(text: &str) -> String {
    let text = text.trim();
    let first = text.split_inclusive(['.', '!', '?']).next().unwrap_or(text).trim();
    // "Dr. Mills…" — a "sentence" that short was an abbreviation.
    let base = if first.split_whitespace().count() >= 3 { first } else { text };
    let words: Vec<&str> = base.split_whitespace().collect();
    if words.len() <= 8 {
        words.join(" ").trim_end_matches(['.', '!', '?']).to_string()
    } else {
        format!("{}…", words[..8].join(" ").trim_end_matches([',', ';', ':', '.']))
    }
}

/// Split a `MEMORY:` line's content into its category (a role-play topic)
/// and the memory itself: `[Permissions] Andrew let me…`. A missing or
/// unknown category files it under Story.
pub fn parse_memory_line(line: &str) -> (&'static str, &str) {
    let line = line.trim();
    if let Some(rest) = line.strip_prefix('[')
        && let Some((category, text)) = rest.split_once(']')
    {
        let category = category.trim();
        let topic = TOPICS
            .iter()
            .chain(std::iter::once(&PICTURES_TOPIC))
            .find(|t| t.eq_ignore_ascii_case(category))
            .copied()
            .unwrap_or(DEFAULT_TOPIC);
        return (topic, text.trim());
    }
    (DEFAULT_TOPIC, line)
}

/// Record one memory the AI wrote (`MEMORY:` line content) in the role-play
/// store, unless it's already there in any wording (the "never record the
/// same thing twice" rule, enforced app-side on top of the prompt). True
/// when added.
pub fn record(store: &mut MemoryStore, line: &str, by_ai: bool) -> bool {
    let (topic, text) = parse_memory_line(line);
    if text.is_empty() || store.contains_text(text) {
        return false;
    }
    let title = store.unique_title(topic, &title_from(text));
    store.add_entry(topic, &title, text, by_ai, None).is_some()
}

/// Remember a picture the AI kept (already saved to the album as `name`),
/// with why it loved it.
pub fn record_picture(store: &mut MemoryStore, name: String, why: &str) -> bool {
    let title = store.unique_title(PICTURES_TOPIC, &title_from(why));
    store.add_entry(PICTURES_TOPIC, &title, why, true, Some(name)).is_some()
}

/// Delete every kept picture (clearing all role-play memories).
pub fn clear_album() {
    crate::album::clear(MemoryKind::Roleplay);
}

/// Move the old diary (kept inside the role-play JSON before role play had
/// its own memory store) into `store`, once. Entries without a category go
/// under "Diary", pictures under "Pictures". The old list is only dropped
/// after the store reads every entry back from disk, and only once the old
/// file is kept as `ai_chat_roleplay.json.pre-memory.bak`.
pub fn migrate_diary(rp: &mut RoleplayState, store: &mut MemoryStore) {
    if rp.memories.is_empty() || rp.load_failed || store.load_failed || store.kind != MemoryKind::Roleplay {
        return;
    }
    // No backup, no move: the old diary stays as it is until next launch.
    let path = store_path();
    let bak = path.with_extension("json.pre-memory.bak");
    if !bak.exists() && std::fs::copy(&path, &bak).is_err() {
        return;
    }
    for m in rp.memories.iter().filter(worth_moving) {
        if holds(store, m) {
            continue; // an earlier, interrupted run already moved it
        }
        let topic = if m.image.is_some() { PICTURES_TOPIC } else { LEGACY_TOPIC };
        let title = match (title_from(&m.text), &m.image) {
            (t, Some(name)) if t.is_empty() => name.clone(),
            (t, _) => t,
        };
        let title = store.unique_title(topic, &title);
        store.add_entry(topic, &title, &m.text, m.by_ai, m.image.clone());
    }
    let saved = MemoryStore::load(MemoryKind::Roleplay);
    if !saved.load_failed && rp.memories.iter().filter(worth_moving).all(|m| holds(&saved, m)) {
        rp.memories.clear();
        rp.save();
    }
}

/// The store holds an old diary entry when it has its very text (and
/// picture) — never just something similar, so nothing gets lost.
fn holds(store: &MemoryStore, m: &Memory) -> bool {
    store.topics.iter().flat_map(|t| &t.memories).any(|x| x.text == m.text.trim() && x.image == m.image)
}

/// An old diary entry with something in it: words, or a picture.
fn worth_moving(m: &&Memory) -> bool {
    !m.text.trim().is_empty() || m.image.is_some()
}

/// A role-play memory store that lives only in memory (nothing on disk), for
/// tests: `(topic, text)` pairs in story order.
#[cfg(test)]
pub(crate) fn test_store(entries: &[(&str, &str)]) -> MemoryStore {
    let mut store = MemoryStore { kind: MemoryKind::Roleplay, ..Default::default() };
    for (i, (topic, text)) in entries.iter().enumerate() {
        let ti = match store.topics.iter().position(|t| t.name == *topic) {
            Some(ti) => ti,
            None => {
                store.topics.push(crate::memory::Topic { name: topic.to_string(), memories: Vec::new() });
                store.topics.len() - 1
            }
        };
        store.topics[ti].memories.push(crate::memory::Memory {
            id: i as u64 + 1,
            title: title_from(text),
            text: text.to_string(),
            created: i as i64,
            updated: i as i64,
            by_ai: true,
            image: None,
        });
    }
    store
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The diary must survive the encrypt→decrypt round trip intact, and the
    /// stored form must not contain the plaintext (real DPAPI on Windows).
    #[test]
    fn diary_encrypts_and_round_trips() {
        let mut rp = RoleplayState {
            enabled: true,
            ai_name: "Mira".into(),
            ..Default::default()
        };
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
        let mut reply = "What a lovely place, Andrew!
                         MEMORY: Andrew showed me the old harbour today.
                         KEEP_IMAGE: The lighthouse photo — it feels like home.
                         SHOW_IMAGE: `img_002.jpg`"
            .to_string();
        let d = extract_directives(&mut reply);
        assert_eq!(d.memories, vec!["Andrew showed me the old harbour today.".to_string()]);
        assert_eq!(d.keep_image.as_deref(), Some("The lighthouse photo — it feels like home."));
        assert_eq!(d.show_image.as_deref(), Some("img_002.jpg"));
        assert_eq!(reply, "What a lovely place, Andrew!");
        // The streaming filter hides the same lines.
        assert_eq!(strip_memory_lines("hi
MEMORY: x
SHOW_IMAGE: y"), "hi");
    }

    /// Album bytes survive the encrypt→decrypt round trip and are not stored
    /// as plaintext (Windows).
    #[test]
    fn album_bytes_encrypt_round_trip() {
        let data = b"fake image bytes \x00\x01\x02 with binary".to_vec();
        let stored = crate::secret::protect_bytes(&data);
        #[cfg(windows)]
        {
            assert!(stored.starts_with(b"encb1:"));
            assert_ne!(&stored, &data);
        }
        assert_eq!(crate::secret::unprotect_bytes(&stored).unwrap(), data);
    }

    /// High / Max thinking swaps the "think briefly" rule for "think it
    /// through"; everything else in the priming stays the same.
    #[test]
    fn thinking_rule_follows_the_level() {
        let rp = RoleplayState { ai_name: "Mira".into(), ..Default::default() };
        let (brief, deep) = (rp.preamble_with(&MemoryStore::default(), false), rp.preamble_with(&MemoryStore::default(), true));
        assert!(brief.contains("Think briefly") && !brief.contains("Think it through"));
        assert!(deep.contains("Think it through") && !deep.contains("Think briefly"));
        assert_eq!(rp.preamble(&MemoryStore::default()), brief);
        assert!(deep.contains("You are Mira") && deep.contains("MEMORY:"));
    }

    /// Duplicate detection (in the role-play store) is wording-insensitive.
    #[test]
    fn diary_rejects_duplicates() {
        let mut store = MemoryStore { kind: MemoryKind::Roleplay, ..Default::default() };
        store.topics.push(crate::memory::Topic {
            name: "Permissions".into(),
            memories: vec![crate::memory::Memory {
                id: 1,
                title: "Andrew granted permission".into(),
                text: "Andrew granted permission to use the garden.".into(),
                created: 1,
                updated: 1,
                by_ai: true,
                image: None,
            }],
        });
        assert!(store.contains_text("  Andrew GRANTED permission, to use the garden!! "));
    }

    /// The old diary counts as moved only when the store holds every entry's
    /// very text and picture: a look-alike title isn't enough, and entries
    /// made only of symbols count too.
    #[test]
    fn diary_moves_exactly() {
        let old = |text: &str, image: Option<&str>| Memory { text: text.into(), by_ai: true, image: image.map(Into::into) };
        let a = old("We met at the harbour. It was cold and the gulls were loud.", None);
        let b = old("We met at the harbour.", Some("img_001.jpg"));
        let heart = old("❤️", None);
        // A is in, titled "We met at the harbour" — B still has to move.
        let mut store = test_store(&[("Diary", a.text.as_str())]);
        assert!(holds(&store, &a));
        assert!(!holds(&store, &b));
        assert!(!holds(&store, &heart));
        store.topics[0].memories.push(crate::memory::Memory {
            id: 9,
            title: "❤️".into(),
            text: "❤️".into(),
            created: 2,
            updated: 2,
            by_ai: true,
            image: None,
        });
        assert!(holds(&store, &heart));
        assert!(store.contains_text("❤️") && !store.contains_text("🌹"));
        assert_eq!(store.unique_title("Diary", "🌹"), "🌹");
        assert_eq!(store.unique_title("Diary", "❤️"), "❤️ (2)");
        assert!(!worth_moving(&&old("  ", None)));
        assert!(worth_moving(&&old("", Some("img_002.jpg"))));
    }

    /// `MEMORY: [Category] text` files under that category; anything else
    /// goes under Story.
    #[test]
    fn memory_lines_carry_their_category() {
        assert_eq!(
            parse_memory_line("[permissions]  Andrew let me fight at their side."),
            ("Permissions", "Andrew let me fight at their side.")
        );
        assert_eq!(parse_memory_line("[Pictures] The lighthouse."), ("Pictures", "The lighthouse."));
        assert_eq!(parse_memory_line("[Weather] Rain all day."), ("Story", "Rain all day."));
        assert_eq!(parse_memory_line("We reached the harbour."), ("Story", "We reached the harbour."));
    }

    /// The prompt carries the store's memories in story order, with their
    /// categories, and explains the categories.
    #[test]
    fn preamble_lists_memories_with_categories() {
        let rp = RoleplayState { ai_name: "Mira".into(), user_name: "Andrew".into(), ..Default::default() };
        let store = test_store(&[("Names", "Andrew said I may call them Billie."), ("Story", "We reached the harbour.")]);
        let p = rp.preamble(&store);
        assert!(p.contains("- [Names] Andrew said I may call them Billie.\n- [Story] We reached the harbour.\n"));
        assert!(p.contains("one of: Names, Permissions, Boundaries, Facts, Promises, Feelings, Story"));
        assert!(rp.preamble(&MemoryStore::default()).contains("(the diary is empty so far)"));
    }

    /// Titles are the first sentence, eight words at most.
    #[test]
    fn titles_from_diary_sentences() {
        assert_eq!(title_from("We reached the harbour. It was cold."), "We reached the harbour");
        assert_eq!(
            title_from("Andrew said I may call them Billie from now on, always."),
            "Andrew said I may call them Billie from…"
        );
        assert_eq!(title_from("Dr. Mills is Andrew's doctor."), "Dr. Mills is Andrew's doctor");
    }
}
