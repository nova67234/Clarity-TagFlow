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
//! The character also has goals of its own: it sets them itself as the story
//! gives it reasons (`GOAL: …`, at most three at a time), works toward them in
//! character, and ends each one as achieved or failed (`GOAL_DONE: [G1] …` /
//! `GOAL_FAILED: [G1] …`, by the numbers its prompt lists them with). They're
//! kept with the character, shown in AI Settings → Role play, and noted under
//! the reply that set or ended them. Both they and the real date and time
//! (llm/clock.rs) can be switched off per character.
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
/// A goal the character sets itself.
pub const GOAL_TAG: &str = "GOAL:";
/// A goal that came true: `GOAL_DONE: [G1] how it happened`.
pub const GOAL_DONE_TAG: &str = "GOAL_DONE:";
/// A goal that can't come true any more: `GOAL_FAILED: [G1] why`.
pub const GOAL_FAILED_TAG: &str = "GOAL_FAILED:";
/// Every control-line prefix (hidden from the visible reply).
const TAGS: [&str; 6] = [MEMORY_TAG, KEEP_IMAGE_TAG, SHOW_IMAGE_TAG, GOAL_TAG, GOAL_DONE_TAG, GOAL_FAILED_TAG];

/// Goals the character pursues at once.
pub const MAX_ACTIVE_GOALS: usize = 3;
/// Finished goals kept (the oldest go first).
const MAX_FINISHED_GOALS: usize = 50;
/// Finished goals the prompt reminds the character of, newest first.
const PROMPT_FINISHED_GOALS: usize = 5;
/// A goal's length cap (a runaway line isn't a goal).
const MAX_GOAL_CHARS: usize = 240;

/// Where a goal stands.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GoalStatus {
    Active,
    Achieved,
    Failed,
}

/// A goal the character set itself.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct Goal {
    /// What it wants, in its own words.
    pub text: String,
    pub status: GoalStatus,
    /// When it was set (Unix seconds).
    pub set: i64,
    /// When it was achieved or failed.
    #[serde(default)]
    pub ended: Option<i64>,
    /// How it came true, or why it couldn't (the character's words; may be
    /// empty).
    #[serde(default)]
    pub outcome: String,
}

/// What one reply did to the goals, for the note under it: a goal set
/// (`Active`), achieved or failed.
#[derive(Clone, Debug, PartialEq)]
pub struct GoalEvent {
    pub status: GoalStatus,
    pub text: String,
}

/// A goal line from a reply, still unresolved.
#[derive(Clone, Debug, PartialEq)]
pub enum GoalLine {
    New(String),
    Done(String),
    Failed(String),
}

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
/// `models_root()/ai_chat_roleplay.json` on every change. (A field missing
/// from an older file takes its value from `Default` — the switches below
/// are on there.)
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RoleplayState {
    pub enabled: bool,
    pub ai_name: String,
    pub user_name: String,
    pub persona: String,
    /// The character sets goals of its own (see the module docs).
    pub own_goals: bool,
    /// Its goals, active and finished, in the order they were set.
    pub goals: Vec<Goal>,
    /// The character is told the date and time on this PC with every
    /// message (llm/clock.rs). Off for stories set in another time or world.
    pub real_time: bool,
    /// A model that thinks before answering thinks as the character — its
    /// inner voice, first person — rather than in notes about the chat.
    pub thoughts_in_character: bool,
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

impl Default for RoleplayState {
    fn default() -> Self {
        Self {
            enabled: false,
            ai_name: String::new(),
            user_name: String::new(),
            persona: String::new(),
            own_goals: true,
            goals: Vec::new(),
            real_time: true,
            thoughts_in_character: true,
            memories: Vec::new(),
            load_failed: false,
        }
    }
}

fn store_path() -> std::path::PathBuf {
    if cfg!(test) {
        // Tests never read or overwrite the real character: each test thread
        // has a file of its own in the temp folder.
        return std::env::temp_dir().join(
            format!("ctf_roleplay_test_{}_{:?}.json", std::process::id(), std::thread::current().id())
                .replace(['(', ')'], ""),
        );
    }
    crate::tagger::models_root().join("ai_chat_roleplay.json")
}

/// This test thread's character file (see `store_path`), to read back or
/// remove.
#[cfg(test)]
pub(crate) fn test_store_path() -> std::path::PathBuf {
    store_path()
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
    /// guarantees an answer. With thoughts in character (the default) either
    /// rule has it think as the character — its inner voice, first person —
    /// instead of in notes about "the user".
    pub fn preamble_with(&self, memories: &MemoryStore, deep_thinking: bool) -> String {
        let ai = self.ai_display();
        let user = self.user_display();
        let thinking = self.thinking_rule(deep_thinking);
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
             {goals}\
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
            goals = self.goals_prompt(),
            tag = MEMORY_TAG,
            keep = KEEP_IMAGE_TAG,
            show = SHOW_IMAGE_TAG,
        )
    }

    /// The priming's goals block (empty with own goals off): the active
    /// goals numbered [G1].. in the order they were set — the numbers the
    /// character ends them by — the last few finished ones, and the rules.
    fn goals_prompt(&self) -> String {
        use std::fmt::Write as _;
        if !self.own_goals {
            return String::new();
        }
        let (ai, user) = (self.ai_display(), self.user_display());
        let mut list = String::new();
        for (n, g) in self.active_goals().enumerate() {
            let _ = writeln!(list, "[G{}] {}", n + 1, g.text);
        }
        if list.is_empty() {
            list = "(none yet — set one when the story gives you a reason)\n".to_string();
        }
        let mut finished: Vec<&Goal> = self.goals.iter().filter(|g| g.status != GoalStatus::Active).collect();
        finished.sort_by_key(|g| std::cmp::Reverse(g.ended.unwrap_or(g.set)));
        if !finished.is_empty() {
            list.push_str("Finished recently:\n");
            for g in finished.into_iter().take(PROMPT_FINISHED_GOALS) {
                let how = if g.status == GoalStatus::Achieved { "achieved" } else { "failed" };
                let _ = writeln!(list, "- {how}: {}", g.text);
            }
        }
        format!(
            "Your goals — things YOU want in your story with {user}, chosen by you:\n\
             {list}\
             \n\
             Goal rules:\n\
             - Want things, as {ai} would: something to do, find, learn, make, \
             share or earn in your story with {user} — specific, and possible in \
             this story. Keep one to {max} goals at a time; when you have fewer \
             and the story gives you a reason, set one by ending your reply with \
             `{goal} <the goal>`.\n\
             - Work toward your goals in character: take steps, steer scenes \
             toward them, ask for what you want — naturally, never forcing it; \
             {user}'s choices always come first.\n\
             - When a goal truly comes true in the story, end your reply with \
             `{done} [G1] <how it happened>`; when it can't happen any more or \
             no longer fits the story, `{failed} [G1] <why>` — with that goal's \
             number. Never mark a goal done before it has really happened.\n\
             - These lines are invisible to {user} too; never mention goals or \
             their numbers in the visible text.\n\
             \n",
            max = MAX_ACTIVE_GOALS,
            goal = GOAL_TAG,
            done = GOAL_DONE_TAG,
            failed = GOAL_FAILED_TAG,
        )
    }

    /// The goals still being pursued, in the order they were set.
    pub fn active_goals(&self) -> impl Iterator<Item = &Goal> {
        self.goals.iter().filter(|g| g.status == GoalStatus::Active)
    }

    /// Act on a finished reply's goal lines: the endings first, by the
    /// numbers the reply was primed with (the active goals in order, before
    /// anything changed), then the new goals — no repeats, at most
    /// MAX_ACTIVE_GOALS at a time. What changed comes back for the reply's
    /// note; the caller saves.
    pub fn apply_goals(&mut self, lines: &[GoalLine], now: i64) -> Vec<GoalEvent> {
        let primed: Vec<usize> = (0..self.goals.len()).filter(|&i| self.goals[i].status == GoalStatus::Active).collect();
        let mut events = Vec::new();
        for line in lines {
            let (status, content) = match line {
                GoalLine::Done(c) => (GoalStatus::Achieved, c),
                GoalLine::Failed(c) => (GoalStatus::Failed, c),
                GoalLine::New(_) => continue,
            };
            let Some((i, outcome)) = goal_ref(content, &primed, &self.goals) else { continue };
            let g = &mut self.goals[i];
            g.status = status;
            g.ended = Some(now);
            g.outcome = outcome;
            events.push(GoalEvent { status, text: g.text.clone() });
        }
        for line in lines {
            let GoalLine::New(raw) = line else { continue };
            let text = clean_goal(raw);
            let repeat = self.active_goals().any(|g| same_goal(&g.text, &text));
            if text.is_empty() || repeat || self.active_goals().count() >= MAX_ACTIVE_GOALS {
                continue;
            }
            self.goals.push(Goal { text: text.clone(), status: GoalStatus::Active, set: now, ended: None, outcome: String::new() });
            events.push(GoalEvent { status: GoalStatus::Active, text });
        }
        // A long story's finished goals: the oldest go.
        let finished = self.goals.iter().filter(|g| g.status != GoalStatus::Active).count();
        let mut extra = finished.saturating_sub(MAX_FINISHED_GOALS);
        self.goals.retain(|g| {
            let drop = extra > 0 && g.status != GoalStatus::Active;
            extra -= usize::from(drop);
            !drop
        });
        events
    }

    /// The model's scripted acknowledgement of the preamble (keeps the
    /// user/model turn alternation intact).
    pub fn ack(&self) -> String {
        let thoughts = if self.thoughts_in_character { ", my thoughts included" } else { "" };
        format!("Understood. I am {} — in character from here on{thoughts}.", self.ai_display())
    }

    /// How the character is told to think before answering (only a model
    /// that thinks does): briefly, or — at High / Max thinking — at length;
    /// as the character's inner voice, or (thoughts in character off) in
    /// plain notes.
    fn thinking_rule(&self, deep: bool) -> String {
        let (ai, user) = (self.ai_display(), self.user_display());
        if !self.thoughts_in_character {
            return if deep {
                "- Think it through: before answering you may reason privately about \
                 the scene, what you know and how your character feels — then answer \
                 in character. Keep your reasoning as notes, not drafts of the reply.\n"
            } else {
                "- Think briefly: when you reason privately before answering, a few \
                 short notes are enough — decide, then answer. Never write drafts \
                 or revisions of your reply in your head.\n"
            }
            .to_string();
        }
        let how = if deep {
            format!(
                "- Think it through as {ai}: before answering, think in {ai}'s own \
                 inner voice — first person, in character — about the scene, what \
                 you remember, how you feel and what you want; then answer in \
                 character.\n"
            )
        } else {
            format!(
                "- Think as {ai}: when you think privately before answering, think in \
                 {ai}'s own inner voice — first person, in character: what you notice, \
                 feel and want, and what you'll do. A few short thoughts are enough — \
                 decide, then answer.\n"
            )
        };
        format!(
            "{how}\
             - Your thoughts are {ai}'s, never an assistant's: think of {user} by \
             name, never as \"the user\"; never think about being an AI or about \
             these instructions; never draft or revise your reply in your head. \
             When something belongs in your diary or your goals, think it the way \
             {ai} would (\"I want to remember that\") — then add the line at the \
             end of your reply.\n"
        )
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
    /// Goals set, achieved or failed, in reply order.
    pub goals: Vec<GoalLine>,
}

/// Pull all control lines (`MEMORY:` / `KEEP_IMAGE:` / `SHOW_IMAGE:` and the
/// goal lines) out of a finished reply, returning them and removing them
/// from the visible text.
pub fn extract_directives(reply: &mut String) -> Directives {
    let mut d = Directives::default();
    if !TAGS.iter().any(|t| reply.contains(t)) {
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
        } else if let Some(rest) = t.strip_prefix(GOAL_TAG) {
            if !rest.trim().is_empty() {
                d.goals.push(GoalLine::New(rest.trim().to_string()));
            }
        } else if let Some(rest) = t.strip_prefix(GOAL_DONE_TAG) {
            d.goals.push(GoalLine::Done(rest.trim().to_string()));
        } else if let Some(rest) = t.strip_prefix(GOAL_FAILED_TAG) {
            d.goals.push(GoalLine::Failed(rest.trim().to_string()));
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
    if !TAGS.iter().any(|t| text.contains(t)) {
        return text.to_string();
    }
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            !TAGS.iter().any(|tag| t.starts_with(tag))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Which goal a `GOAL_DONE:` / `GOAL_FAILED:` line ends, and the rest of the
/// line (how it came true, or why not): by number — `[G2]`, `G2`, `#2` —
/// among the goals the reply was primed with (`primed`, the active ones in
/// order); else by its words; else the only one there is. Never one that
/// already ended.
fn goal_ref(content: &str, primed: &[usize], goals: &[Goal]) -> Option<(usize, String)> {
    let open = |i: &usize| goals[*i].status == GoalStatus::Active;
    let content = content.trim();
    let s = content.strip_prefix('[').unwrap_or(content);
    let s = s.strip_prefix(['G', 'g', '#']).unwrap_or(s);
    let digits = s.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 {
        let n: usize = s[..digits].parse().ok()?;
        let i = *primed.get(n.checked_sub(1)?)?;
        let rest = s[digits..].trim_start_matches([']', '.', ':', ')']);
        return open(&i).then(|| (i, outcome_of(rest)));
    }
    let live: Vec<usize> = primed.iter().copied().filter(open).collect();
    let best = live
        .iter()
        .map(|&i| (i, overlap(&goals[i].text, content)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let outcome = |i: usize| if same_goal(&goals[i].text, content) { String::new() } else { outcome_of(content) };
    match best {
        Some((i, score)) if score >= 0.5 => Some((i, outcome(i))),
        _ if live.len() == 1 => Some((live[0], outcome(live[0]))),
        _ => None,
    }
}

/// How a goal ended, as the line says it: leading separators off, capped.
fn outcome_of(rest: &str) -> String {
    cap(rest.trim_start_matches(|c: char| c.is_whitespace() || matches!(c, '-' | '—' | '–' | ':')).trim())
}

/// A new goal as written, tidied: no quotes or self-given number, one line,
/// capped.
fn clean_goal(raw: &str) -> String {
    let mut t = raw.trim().trim_matches(|c| matches!(c, '"' | '`' | '“' | '”')).trim();
    if let Some(rest) = t.strip_prefix('[')
        && let Some((tag, after)) = rest.split_once(']')
        && tag.trim_start_matches(['G', 'g']).chars().all(|c| c.is_ascii_digit())
    {
        t = after.trim();
    }
    cap(&t.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn cap(s: &str) -> String {
    if s.chars().count() <= MAX_GOAL_CHARS {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(MAX_GOAL_CHARS).collect::<String>().trim_end())
    }
}

/// The same goal in other words: the same words, or one wording inside the
/// other.
fn same_goal(a: &str, b: &str) -> bool {
    let (na, nb) = (crate::memory::normalize(a), crate::memory::normalize(b));
    crate::memory::same_words(a, b) || (na.len() >= 12 && nb.len() >= 12 && (na.contains(&nb) || nb.contains(&na)))
}

/// How much of a goal's wording a line repeats: the share of its words
/// (three letters or more, not the commonest ones) that the line has too.
fn overlap(goal: &str, line: &str) -> f32 {
    const COMMON: [&str; 16] = [
        "the", "and", "for", "with", "her", "his", "him", "them", "their", "our", "you", "your", "from", "into",
        "that", "this",
    ];
    let words = |s: &str| -> std::collections::HashSet<String> {
        crate::memory::normalize(s)
            .split_whitespace()
            .filter(|w| w.chars().count() >= 3 && !COMMON.contains(w))
            .map(String::from)
            .collect()
    };
    let (g, l) = (words(goal), words(line));
    if g.is_empty() {
        return 0.0;
    }
    g.intersection(&l).count() as f32 / g.len() as f32
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

    /// High / Max thinking swaps the brief thinking rule for the longer one;
    /// in character (the default) both have it think as the character, in
    /// its own voice — off, they ask for plain notes. Everything else in the
    /// priming stays the same.
    #[test]
    fn thinking_rule_follows_the_level_and_the_voice() {
        let mut rp = RoleplayState { ai_name: "Mira".into(), user_name: "Andrew".into(), ..Default::default() };
        let primed = |rp: &RoleplayState| {
            (rp.preamble_with(&MemoryStore::default(), false), rp.preamble_with(&MemoryStore::default(), true))
        };
        let (brief, deep) = primed(&rp);
        assert!(brief.contains("- Think as Mira:") && brief.contains("A few short thoughts are enough"));
        assert!(deep.contains("- Think it through as Mira:") && !deep.contains("A few short thoughts"));
        for p in [&brief, &deep] {
            assert!(p.contains("Mira's own inner voice — first person"), "{p}");
            assert!(p.contains("think of Andrew by name, never as \"the user\""));
            assert!(!p.contains("Think briefly") && !p.contains("as notes"));
        }
        assert_eq!(rp.preamble(&MemoryStore::default()), brief);
        assert!(deep.contains("You are Mira") && deep.contains("MEMORY:"));
        assert_eq!(rp.ack(), "Understood. I am Mira — in character from here on, my thoughts included.");

        rp.thoughts_in_character = false;
        let (brief, deep) = primed(&rp);
        assert!(brief.contains("Think briefly") && !brief.contains("Think it through"));
        assert!(deep.contains("Think it through:") && !deep.contains("Think briefly"));
        assert!(!brief.contains("inner voice") && !deep.contains("inner voice"));
        assert_eq!(rp.ack(), "Understood. I am Mira — in character from here on.");
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

    fn goal(text: &str, status: GoalStatus, set: i64) -> Goal {
        let ended = (status != GoalStatus::Active).then_some(set + 1);
        Goal { text: text.into(), status, set, ended, outcome: String::new() }
    }

    fn character(goals: Vec<Goal>) -> RoleplayState {
        RoleplayState { ai_name: "Mira".into(), user_name: "Andrew".into(), goals, ..Default::default() }
    }

    /// Goal lines come out of the reply with the rest, and stay hidden
    /// while it streams.
    #[test]
    fn goal_lines_extract_and_strip() {
        let mut reply = "We made it to the top!\n\
                         GOAL_DONE: [G1] We climbed the lighthouse at sunset.\n\
                         GOAL_FAILED: [G2] The festival was cancelled.\n\
                         GOAL: Learn to sail with Andrew\n\
                         GOAL:   "
            .to_string();
        let d = extract_directives(&mut reply);
        assert_eq!(reply, "We made it to the top!");
        assert_eq!(
            d.goals,
            vec![
                GoalLine::Done("[G1] We climbed the lighthouse at sunset.".into()),
                GoalLine::Failed("[G2] The festival was cancelled.".into()),
                GoalLine::New("Learn to sail with Andrew".into()),
            ]
        );
        assert_eq!(strip_memory_lines("hi\nGOAL: x\nGOAL_DONE: [G1]\nGOAL_FAILED: [G2] y"), "hi");
    }

    /// New goals: no repeats in other words, at most three at a time.
    #[test]
    fn goals_are_set_without_repeats_and_capped() {
        let mut rp = character(vec![]);
        let lines = |v: &[&str]| v.iter().map(|t| GoalLine::New(t.to_string())).collect::<Vec<_>>();
        let events = rp.apply_goals(&lines(&["\"Visit the lighthouse with Andrew\"", "visit the LIGHTHOUSE with Andrew!", "[G2] Learn to sail"]), 100);
        assert_eq!(
            events,
            vec![
                GoalEvent { status: GoalStatus::Active, text: "Visit the lighthouse with Andrew".into() },
                GoalEvent { status: GoalStatus::Active, text: "Learn to sail".into() },
            ]
        );
        let events = rp.apply_goals(&lines(&["Bake bread together", "Find the lost compass"]), 200);
        assert_eq!(events.len(), 1, "a fourth goal waits");
        assert_eq!(rp.active_goals().count(), MAX_ACTIVE_GOALS);
        assert!(rp.goals.iter().all(|g| g.status == GoalStatus::Active && g.ended.is_none()));
    }

    /// Endings go by the numbers the reply was primed with — even after an
    /// earlier line in the same reply ended another goal — else by the
    /// goal's words, else to the only goal there is; never twice.
    #[test]
    fn goals_end_by_number_words_or_the_only_one() {
        let primed = || {
            character(vec![
                goal("Visit the lighthouse with Andrew", GoalStatus::Active, 1),
                goal("Old wish", GoalStatus::Achieved, 2),
                goal("Learn to sail with Andrew", GoalStatus::Active, 3),
                goal("Bake bread together", GoalStatus::Active, 4),
            ])
        };
        let mut rp = primed();
        let events = rp.apply_goals(
            &[
                GoalLine::Done("[G1] — we climbed it at sunset".into()),
                GoalLine::Failed("[G3] the oven broke for good".into()),
                GoalLine::Done("[G1] again".into()),
                GoalLine::Done("G9 nothing there".into()),
            ],
            50,
        );
        assert_eq!(events.len(), 2, "{events:?}");
        assert_eq!(rp.goals[0].status, GoalStatus::Achieved);
        assert_eq!(rp.goals[0].outcome, "we climbed it at sunset");
        assert_eq!(rp.goals[0].ended, Some(50));
        assert_eq!(rp.goals[3].status, GoalStatus::Failed);
        assert_eq!(rp.goals[2].status, GoalStatus::Active);

        // No number: the goal whose words it shares most ("sail" and
        // "andrew" of learn/sail/andrew), the whole line as the outcome.
        let mut rp = primed();
        rp.apply_goals(&[GoalLine::Done("Learned to sail at last: Andrew was so proud".into())], 60);
        assert_eq!(rp.goals[2].status, GoalStatus::Achieved);
        assert_eq!(rp.goals[0].status, GoalStatus::Active);

        // Words that fit none, with several goals: nothing ends.
        let mut rp = primed();
        assert!(rp.apply_goals(&[GoalLine::Done("It happened!".into())], 70).is_empty());
        // ...but with one goal left, it's that one.
        let mut rp = character(vec![goal("Visit the lighthouse with Andrew", GoalStatus::Active, 1)]);
        let events = rp.apply_goals(&[GoalLine::Done("We finally went there today".into())], 80);
        assert_eq!(events, vec![GoalEvent { status: GoalStatus::Achieved, text: "Visit the lighthouse with Andrew".into() }]);
        assert_eq!(rp.goals[0].outcome, "We finally went there today");
        // A goal restated is no outcome.
        let mut rp = character(vec![goal("Visit the lighthouse with Andrew", GoalStatus::Active, 1)]);
        rp.apply_goals(&[GoalLine::Done("Visit the lighthouse with Andrew.".into())], 80);
        assert_eq!(rp.goals[0].outcome, "");
    }

    /// A long story keeps its newest finished goals and every active one.
    #[test]
    fn finished_goals_are_capped() {
        let mut goals: Vec<Goal> =
            (0..MAX_FINISHED_GOALS as i64 + 5).map(|i| goal(&format!("Goal number {i}"), GoalStatus::Achieved, i)).collect();
        goals.push(goal("Still going", GoalStatus::Active, 999));
        let mut rp = character(goals);
        rp.apply_goals(&[GoalLine::New("One more".into())], 1000);
        assert_eq!(rp.goals.iter().filter(|g| g.status != GoalStatus::Active).count(), MAX_FINISHED_GOALS);
        assert_eq!(rp.goals[0].text, "Goal number 5", "the oldest went");
        assert_eq!(rp.active_goals().count(), 2);
    }

    /// The priming lists the active goals by number, the last finished ones
    /// and the rules — and nothing about goals with own goals off.
    #[test]
    fn preamble_carries_goals_only_when_on() {
        let mut rp = character(vec![
            goal("Visit the lighthouse with Andrew", GoalStatus::Active, 1),
            goal("Learn to sail", GoalStatus::Achieved, 2),
            goal("Bake bread together", GoalStatus::Active, 3),
        ]);
        let p = rp.preamble(&MemoryStore::default());
        assert!(p.contains("[G1] Visit the lighthouse with Andrew\n[G2] Bake bread together\n"), "{p}");
        assert!(p.contains("- achieved: Learn to sail\n"));
        assert!(p.contains("GOAL_DONE: [G1]") && p.contains("GOAL_FAILED: [G1]") && p.contains("`GOAL: <the goal>`"));
        assert!(p.contains("Your photo album:"), "the album rules still follow");
        let empty = character(vec![]).preamble(&MemoryStore::default());
        assert!(empty.contains("(none yet"));
        rp.own_goals = false;
        let off = rp.preamble(&MemoryStore::default());
        assert!(!off.contains("GOAL") && !off.contains("Visit the lighthouse"), "{off}");
    }

    /// A character saved before goals (and the time switch) existed reads
    /// back with both on; goals survive the round trip.
    #[test]
    fn older_characters_get_goals_and_the_clock() {
        let old: RoleplayState = serde_json::from_str(r#"{"enabled":true,"ai_name":"Mira","persona":"a sailor"}"#).unwrap();
        assert!(old.own_goals && old.real_time && old.thoughts_in_character && old.goals.is_empty());
        let mut rp = character(vec![goal("Visit the lighthouse", GoalStatus::Failed, 7)]);
        rp.goals[0].outcome = "storm season".into();
        rp.real_time = false;
        let back: RoleplayState = serde_json::from_str(&serde_json::to_string(&rp).unwrap()).unwrap();
        assert_eq!(back.goals, rp.goals);
        assert!(!back.real_time);
    }
}
