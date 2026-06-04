//! Favorites ("hearts") — a Rust port of terminus2's `HeartManager`.
//!
//! Favorites are tracked by a fast *content* hash (file size + first/last 4 KiB,
//! see [`crate::scan::fast_content_hash`]) rather than by path, so hearting an
//! image survives moving or renaming it. The set of hashes is persisted as a JSON
//! array in `hearted.json` under the per-user config dir (the same location the
//! Gelbooru downloader uses), and saved synchronously on every toggle — the set
//! is tiny, so there's no need for the background save thread the Java version had.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// In-memory favorites set plus a per-session path→hash cache so a given file's
/// content hash is only computed once (it's recomputed lazily on first use, e.g.
/// when a thumbnail scrolls into view).
#[derive(Default)]
pub struct Favorites {
    hashes: HashSet<String>,
    cache: HashMap<PathBuf, String>,
}

impl Favorites {
    /// Load the persisted favorites (an empty set if the file is missing/corrupt).
    pub fn load() -> Self {
        let hashes = std::fs::read(store_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice::<HashSet<String>>(&bytes).ok())
            .unwrap_or_default();
        Self { hashes, cache: HashMap::new() }
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
