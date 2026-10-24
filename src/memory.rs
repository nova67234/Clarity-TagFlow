//! The AI's long-term memory across chats (the chat list's gear → AI Settings
//! → Memory, and the 3D memory map in src/memory_map.rs).
//!
//! The layout follows the design from the "Creating 3D visualizations in chat"
//! session:
//!
//! * a small **header** a new chat can load cheaply: the AI's own persona
//!   (voice and self-notes), a rolling summary of the last chat, and the topic
//!   **index** (names only, no content);
//! * **one file per topic** (a project, a person, a preference area…) holding
//!   that topic's memories, read only when a chat actually needs them.
//!
//! The background update pass after a chat ends and the on-demand lookup
//! during a chat are separate work (see the "AI Memory Across Chats" spec);
//! this module is the store they share with the memory map and the settings
//! tab.
//!
//! Files live in `models_root()/ai_memory/`: `header.json` plus
//! `topics/<slug>.json`, each **encrypted at rest** with src/secret.rs like the
//! role-play diary. A file that exists but can't be read (say the folder was
//! copied from another computer or user account) marks the store
//! `load_failed`, and nothing is ever written over it until the user chooses
//! to start fresh.

use std::path::PathBuf;

/// One remembered thing.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct Memory {
    pub id: u64,
    /// A short headline ("Sister lives in Lisbon").
    pub title: String,
    /// The detail behind it, in a sentence or two.
    pub text: String,
    /// Unix seconds.
    pub created: i64,
    pub updated: i64,
}

/// A topic's memories (one file on disk).
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Topic {
    pub name: String,
    pub memories: Vec<Memory>,
}

/// One entry of the topic index: enough to list and find a topic without
/// reading its file.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct IndexEntry {
    pub name: String,
    /// File stem under `topics/`: an opaque `topic-N`, so the file names
    /// never give away what the encrypted topics are about.
    pub slug: String,
    pub count: usize,
    pub updated: i64,
}

/// The always-loaded part of the memory.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Header {
    /// The AI's own persona: how it describes itself and its voice. Written
    /// by the AI in the background pass; the user can edit it in settings.
    pub persona: String,
    /// A short recap of the most recent chat(s), read when a new chat starts.
    pub summary: String,
    pub summary_updated: i64,
    /// The topic index (names only — content stays in the topic files).
    pub index: Vec<IndexEntry>,
    /// Next memory id to hand out.
    pub next_id: u64,
}

/// The whole memory as the UI sees it: the header plus every topic.
#[derive(Default)]
pub struct MemoryStore {
    pub header: Header,
    /// Every topic, in index order.
    pub topics: Vec<Topic>,
    /// A file exists but couldn't be read: show it, never overwrite it.
    pub load_failed: bool,
    /// Bumped on every change, so views (the memory map) can rebuild.
    pub revision: u64,
}

fn dir() -> PathBuf {
    crate::tagger::models_root().join("ai_memory")
}

fn topics_dir() -> PathBuf {
    dir().join("topics")
}

fn header_path() -> PathBuf {
    dir().join("header.json")
}

fn topic_path(slug: &str) -> PathBuf {
    topics_dir().join(format!("{slug}.json"))
}

pub fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Case/punctuation-insensitive form, for matching topic names and spotting
/// duplicate memories.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The file stem for a new topic: an opaque `topic-N` (the first one `taken`
/// says is free). A topic's name is as private as its memories, so it never
/// becomes a file name.
fn next_slug(taken: impl Fn(&str) -> bool) -> String {
    let mut n = 1u32;
    loop {
        let slug = format!("topic-{n}");
        if !taken(&slug) {
            return slug;
        }
        n += 1;
    }
}

/// Read and decrypt one JSON file. `Ok(None)` = no file (or empty);
/// `Err(())` = a file exists but can't be read.
fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> Result<Option<T>, ()> {
    let stored = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        // Locked, no permission, not text…: something is there that can't be
        // read, so it must never be taken for "empty" and overwritten.
        Err(_) => return Err(()),
    };
    if stored.trim().is_empty() {
        return Ok(None);
    }
    // `unprotect` yields "" for ciphertext this account can't decrypt.
    serde_json::from_str(&crate::secret::unprotect(stored.trim())).map(Some).map_err(|_| ())
}

/// Encrypt and save one JSON file. It goes to a temp file first and is then
/// renamed over the old one, so a crash mid-save (or a chat reading a topic at
/// that moment) never sees half a file: a torn file would make the whole store
/// unreadable, and therefore read-only.
fn write_json<T: serde::Serialize>(path: &std::path::Path, value: &T) {
    let Ok(json) = serde_json::to_string(value) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let stored = crate::secret::protect(&json);
    // `protect()` hands the plaintext back when encryption fails; memories
    // are never written to disk unencrypted.
    if stored == json {
        eprintln!("[memory] encryption failed, not saving {}", path.display());
        return;
    }
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, stored).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

impl MemoryStore {
    /// Load the header and every topic it indexes.
    pub fn load() -> Self {
        let mut store = MemoryStore::default();
        match read_json::<Header>(&header_path()) {
            Ok(Some(h)) => store.header = h,
            Ok(None) => return store, // nothing remembered yet
            Err(()) => {
                store.load_failed = true;
                return store;
            }
        }
        let index = store.header.index.clone();
        for entry in &index {
            match Self::load_topic_file(&entry.slug) {
                Ok(Some(t)) => store.topics.push(t),
                Ok(None) => store.topics.push(Topic { name: entry.name.clone(), memories: Vec::new() }),
                Err(()) => {
                    store.load_failed = true;
                    store.topics.push(Topic { name: entry.name.clone(), memories: Vec::new() });
                }
            }
        }
        store
    }

    /// Read one topic's file by its slug, without loading anything else —
    /// the cheap lookup a chat uses when it needs a topic's memories.
    pub fn load_topic_file(slug: &str) -> Result<Option<Topic>, ()> {
        read_json::<Topic>(&topic_path(slug))
    }

    /// Total memories across all topics.
    pub fn count(&self) -> usize {
        self.topics.iter().map(|t| t.memories.len()).sum()
    }

    /// The newest change to any memory or the summary.
    pub fn last_updated(&self) -> Option<i64> {
        let latest = self
            .header
            .index
            .iter()
            .map(|e| e.updated)
            .chain(std::iter::once(self.header.summary_updated))
            .max()
            .unwrap_or(0);
        (latest > 0).then_some(latest)
    }

    fn touch(&mut self) {
        self.revision += 1;
    }

    fn save_header(&self) {
        if !self.load_failed {
            write_json(&header_path(), &self.header);
        }
    }

    fn save_topic(&self, i: usize) {
        if self.load_failed {
            return;
        }
        if let (Some(t), Some(e)) = (self.topics.get(i), self.header.index.get(i)) {
            write_json(&topic_path(&e.slug), t);
        }
    }

    /// Bring index entry `i` in line with its topic (count + updated).
    fn reindex(&mut self, i: usize) {
        if let (Some(t), Some(e)) = (self.topics.get(i), self.header.index.get_mut(i)) {
            e.name.clone_from(&t.name);
            e.count = t.memories.len();
            e.updated = t.memories.iter().map(|m| m.updated).max().unwrap_or(e.updated);
        }
    }

    /// Index of the topic called `name` (case- and punctuation-insensitive).
    pub fn find_topic(&self, name: &str) -> Option<usize> {
        let key = normalize(name);
        self.topics.iter().position(|t| normalize(&t.name) == key)
    }

    /// The topic called `name`, created (with its own file) if it's new.
    fn topic_mut_or_create(&mut self, name: &str) -> usize {
        if let Some(i) = self.find_topic(name) {
            return i;
        }
        let slug = next_slug(|s| self.header.index.iter().any(|e| e.slug == s) || topic_path(s).exists());
        self.topics.push(Topic { name: name.trim().to_string(), memories: Vec::new() });
        self.header.index.push(IndexEntry {
            name: name.trim().to_string(),
            slug,
            count: 0,
            updated: now(),
        });
        self.topics.len() - 1
    }

    /// Remember something under `topic`. A memory with the same title in that
    /// topic is updated in place instead of duplicated. Returns the topic
    /// index and the memory id, or `None` for empty input or an unreadable
    /// store.
    pub fn add(&mut self, topic: &str, title: &str, text: &str) -> Option<(usize, u64)> {
        let (topic, title, text) = (topic.trim(), title.trim(), text.trim());
        if self.load_failed || topic.is_empty() || (title.is_empty() && text.is_empty()) {
            return None;
        }
        let ti = self.topic_mut_or_create(topic);
        let key = normalize(title);
        let t = now();
        let id = if let Some(m) = self.topics[ti].memories.iter_mut().find(|m| normalize(&m.title) == key) {
            m.text = text.to_string();
            m.updated = t;
            m.id
        } else {
            self.header.next_id += 1;
            let id = self.header.next_id;
            self.topics[ti].memories.push(Memory {
                id,
                title: title.to_string(),
                text: text.to_string(),
                created: t,
                updated: t,
            });
            id
        };
        self.reindex(ti);
        self.save_topic(ti);
        self.save_header();
        self.touch();
        Some((ti, id))
    }

    /// Change a memory's title and text.
    pub fn update(&mut self, topic: usize, id: u64, title: &str, text: &str) {
        if self.load_failed {
            return;
        }
        let Some(m) = self.topics.get_mut(topic).and_then(|t| t.memories.iter_mut().find(|m| m.id == id)) else {
            return;
        };
        m.title = title.trim().to_string();
        m.text = text.trim().to_string();
        m.updated = now();
        self.reindex(topic);
        self.save_topic(topic);
        self.save_header();
        self.touch();
    }

    /// Forget one memory. A topic left empty is removed with its file.
    pub fn delete(&mut self, topic: usize, id: u64) {
        if self.load_failed {
            return;
        }
        let Some(t) = self.topics.get_mut(topic) else { return };
        t.memories.retain(|m| m.id != id);
        if t.memories.is_empty() {
            self.delete_topic(topic);
            return;
        }
        self.reindex(topic);
        self.save_topic(topic);
        self.save_header();
        self.touch();
    }

    /// Forget a whole topic.
    pub fn delete_topic(&mut self, topic: usize) {
        if self.load_failed || topic >= self.topics.len() {
            return;
        }
        self.topics.remove(topic);
        let entry = self.header.index.remove(topic);
        let _ = std::fs::remove_file(topic_path(&entry.slug));
        self.save_header();
        self.touch();
    }

    /// Change the AI's persona notes (settings tab).
    pub fn set_persona(&mut self, persona: &str) {
        if self.load_failed || self.header.persona == persona {
            return;
        }
        self.header.persona = persona.to_string();
        self.save_header();
        self.touch();
    }

    /// Forget everything: every topic, the summary and the persona notes.
    pub fn clear_all(&mut self) {
        if self.load_failed {
            return;
        }
        let _ = std::fs::remove_dir_all(dir());
        let next_id = self.header.next_id;
        self.header = Header { next_id, ..Default::default() };
        self.topics.clear();
        self.touch();
    }

    /// Give up on an unreadable memory: move the folder aside (kept as
    /// `ai_memory.bak`, replacing an older backup) and start empty.
    pub fn start_fresh(&mut self) {
        let bak = dir().with_file_name("ai_memory.bak");
        let _ = std::fs::remove_dir_all(&bak);
        let _ = std::fs::rename(dir(), &bak);
        *self = MemoryStore { revision: self.revision + 1, ..Default::default() };
    }
}

/// "Oct 24, 2026" for a unix-seconds timestamp (local time).
pub fn format_date(secs: i64) -> String {
    use chrono::TimeZone as _;
    match chrono::Local.timestamp_opt(secs, 0).single() {
        Some(dt) => dt.format("%b %-d, %Y").to_string(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Topic files get opaque names, skipping ones already taken.
    #[test]
    fn topic_files_get_opaque_names() {
        assert_eq!(next_slug(|_| false), "topic-1");
        assert_eq!(next_slug(|s| s == "topic-1" || s == "topic-2"), "topic-3");
    }

    /// Only a missing file counts as empty. Anything there that can't be read
    /// is an error, so the store goes read-only instead of being overwritten.
    #[test]
    fn unreadable_files_are_errors() {
        let dir = std::env::temp_dir().join(format!("ctf_memory_unreadable_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("a-folder.json")).unwrap();
        std::fs::write(dir.join("not-text.json"), [0xff, 0xfe, 0x00]).unwrap();
        assert_eq!(read_json::<Topic>(&dir.join("missing.json")).map(|t| t.is_none()), Ok(true));
        assert!(read_json::<Topic>(&dir.join("a-folder.json")).is_err());
        assert!(read_json::<Topic>(&dir.join("not-text.json")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Topic lookup ignores case, spacing and punctuation (no disk access).
    #[test]
    fn topic_lookup_is_forgiving() {
        let mut s = MemoryStore::default();
        s.topics.push(Topic { name: "Clarity TagFlow".into(), memories: Vec::new() });
        assert_eq!(s.find_topic("  clarity   tagflow! "), Some(0));
        assert_eq!(s.find_topic("Clarity"), None);
    }

    /// Saves round-trip through the encryption and leave no temp file behind.
    #[test]
    fn saves_round_trip_without_leftovers() {
        let dir = std::env::temp_dir().join(format!("ctf_memory_test_{}", std::process::id()));
        let path = dir.join("topics").join("work.json");
        let topic = Topic {
            name: "Work".into(),
            memories: vec![Memory { id: 7, title: "Title".into(), text: "Some detail".into(), created: 1, updated: 2 }],
        };
        write_json(&path, &topic);
        write_json(&path, &topic); // replacing an existing file works too
        let back: Topic = read_json(&path).expect("readable").expect("present");
        assert_eq!(back.memories, topic.memories);
        assert!(!path.with_extension("json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An unreadable store refuses every write, so nothing overwrites it.
    #[test]
    fn unreadable_store_is_read_only() {
        let mut s = MemoryStore { load_failed: true, ..Default::default() };
        assert_eq!(s.add("Work", "Title", "Text"), None);
        s.set_persona("changed");
        assert!(s.header.persona.is_empty());
        assert_eq!(s.revision, 0);
    }
}
