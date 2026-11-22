//! Favorites ("hearts") — a Rust port of terminus2's `HeartManager`.
//!
//! Favorites are tracked by a fast *content* hash (file size + first/last 4 KiB,
//! see [`crate::scan::fast_content_hash`]) rather than by path, so hearting an
//! image survives moving or renaming it. The set of hashes is persisted as a JSON
//! array in `hearted.json` under the per-user config dir (the same location the
//! Gelbooru downloader uses), and saved synchronously on every toggle — the set
//! is tiny, so there's no need for the background save thread the Java version had.
//!
//! The work assistant's own opinion of an image it was shown (src/assistant.rs)
//! is kept the same way, by the same hash, apart from the user's hearts:
//! `ai_opinions.json`, hash → liked or not. A tile shows it as a small AI mark.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// In-memory favorites set plus a per-session path→hash cache so a given file's
/// content hash is only computed once (it's recomputed lazily on first use, e.g.
/// when a thumbnail scrolls into view).
#[derive(Default)]
pub struct Favorites {
    hashes: HashSet<String>,
    cache: HashMap<PathBuf, String>,
    /// What the work assistant thought of the images it was shown: liked
    /// (true) or not.
    ai: HashMap<String, bool>,
    /// Where `ai` is kept (a test's own file).
    ai_store: Option<PathBuf>,
}

impl Favorites {
    /// Load the persisted favorites (an empty set if the file is missing/corrupt).
    pub fn load() -> Self {
        let hashes = std::fs::read(store_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<HashSet<String>>(&bytes).ok())
            .unwrap_or_default();
        let ai = std::fs::read(ai_store_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<HashMap<String, bool>>(&bytes).ok())
            .unwrap_or_default();
        Self { hashes, cache: HashMap::new(), ai, ai_store: Some(ai_store_path()) }
    }

    /// What the work assistant thought of `path`'s content, if it has said:
    /// liked (true) or not.
    pub fn ai_opinion(&mut self, path: &Path) -> Option<bool> {
        if self.ai.is_empty() {
            return None;
        }
        let hash = self.hash_of(path)?;
        self.ai.get(&hash).copied()
    }

    /// The work assistant liked (or didn't like) the image at `path`: kept,
    /// in place of what it thought of it before. False when the file can't
    /// be read.
    pub fn set_ai_opinion(&mut self, path: &Path, like: bool) -> bool {
        let Some(hash) = self.hash_of(path) else { return false };
        if self.ai.insert(hash, like) != Some(like)
            && let Some(store) = &self.ai_store
        {
            if let Some(dir) = store.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(json) = serde_json::to_vec_pretty(&self.ai) {
                let _ = std::fs::write(store, json);
            }
        }
        true
    }

    /// Whether `path`'s content is currently favorited. Takes `&mut self` because
    /// it may populate the path→hash cache; it never touches disk after the first
    /// look-up for a given path.
    pub fn is_favorite(&mut self, path: &Path) -> bool {
        match self.hash_of(path) {
            Some(h) => self.hashes.contains(&h),
            None => false,
        }
    }

    /// Toggle `path`'s favorite state, persist, and return the new state.
    pub fn toggle(&mut self, path: &Path) -> bool {
        let Some(h) = self.hash_of(path) else { return false };
        let now_favorite = if self.hashes.contains(&h) {
            self.hashes.remove(&h);
            false
        } else {
            self.hashes.insert(h);
            true
        };
        self.save();
        now_favorite
    }

    /// Drop `path`'s cached hash. Call when the file there is deleted or moved,
    /// or a new one is written there (a crop or cutout can reuse a deleted
    /// one's name): otherwise the next look-up reuses the old file's hash, and
    /// the new file shows — and toggles — the old file's heart. (Look-ups run
    /// per tile per frame, so they can't afford to stat the file.)
    pub fn forget(&mut self, path: &Path) {
        self.cache.remove(path);
    }

    /// Content hash for `path`, computed once and then cached for the session.
    fn hash_of(&mut self, path: &Path) -> Option<String> {
        if let Some(h) = self.cache.get(path) {
            return Some(h.clone());
        }
        let h = crate::scan::fast_content_hash(path)?;
        self.cache.insert(path.to_path_buf(), h.clone());
        Some(h)
    }

    /// Write the hash set to disk (best-effort; errors are ignored, matching the
    /// rest of the app's "favorites are nice-to-have" persistence).
    fn save(&self) {
        let path = store_path();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_vec_pretty(&self.hashes) {
            let _ = std::fs::write(path, json);
        }
    }
}

/// `…/Clarity TagFlow/hearted.json` under the per-user config dir (mirrors the
/// path scheme in `download.rs`).
fn store_path() -> PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow"))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hearted.json")
}

/// `…/Clarity TagFlow/ai_opinions.json`, beside the hearts.
fn ai_store_path() -> PathBuf {
    store_path().with_file_name("ai_opinions.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    // The assistant's opinion goes with the image's content (a renamed file
    // keeps it), is kept on disk, and a later one replaces it. Apart from the
    // user's hearts.
    #[test]
    fn the_assistants_opinion_is_kept_by_content() {
        let dir = std::env::temp_dir().join(format!("ctf_ai_opinions_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b, store) = (dir.join("a.png"), dir.join("b.png"), dir.join("ai_opinions.json"));
        std::fs::write(&a, b"image a").unwrap();
        std::fs::write(&b, b"image b").unwrap();
        let mut fav = Favorites { ai_store: Some(store.clone()), ..Default::default() };
        assert_eq!(fav.ai_opinion(&a), None);
        assert!(fav.set_ai_opinion(&a, true) && fav.set_ai_opinion(&b, false));
        assert_eq!((fav.ai_opinion(&a), fav.ai_opinion(&b)), (Some(true), Some(false)));
        assert!(!fav.is_favorite(&a), "not a heart of the user's");
        assert!(!fav.set_ai_opinion(&dir.join("gone.png"), true));

        // Read back from disk; renamed, the file keeps its mark.
        let kept: HashMap<String, bool> = serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
        let mut later = Favorites { ai: kept, ai_store: Some(store.clone()), ..Default::default() };
        let moved = dir.join("renamed.png");
        std::fs::rename(&a, &moved).unwrap();
        assert_eq!(later.ai_opinion(&moved), Some(true));
        // It changed its mind.
        assert!(later.set_ai_opinion(&moved, false));
        assert_eq!(later.ai_opinion(&moved), Some(false));
        let kept: HashMap<String, bool> = serde_json::from_slice(&std::fs::read(&store).unwrap()).unwrap();
        assert!(kept.values().all(|liked| !liked) && kept.len() == 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Crop, heart the crop, delete it, crop again under the same name: once
    // the path is forgotten the new file is judged by its own content.
    // (Never calls `toggle`, which would write the real hearted.json.)
    #[test]
    fn forget_rehashes_a_replaced_file() {
        let dir = std::env::temp_dir().join(format!("ctf_favorites_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x_crop.png");
        std::fs::write(&p, b"first crop").unwrap();
        let mut fav = Favorites::default();
        let first = fav.hash_of(&p).unwrap();
        fav.hashes.insert(first);
        assert!(fav.is_favorite(&p));

        std::fs::write(&p, b"second crop").unwrap();
        fav.forget(&p);
        assert!(!fav.is_favorite(&p), "the new file inherited the old heart");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
