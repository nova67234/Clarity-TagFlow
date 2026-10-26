//! The AI's photo albums: pictures it kept from a chat, each **encrypted at
//! rest** (src/secret.rs) as `<name>.enc`, with why it kept them beside it as
//! `<stem>.txt.enc`. The memory about a picture names it (`Memory::image`);
//! forgetting that memory deletes the picture.
//!
//! Two albums, one per memory store (see [`MemoryKind`]): role play's (its
//! character's, src/roleplay.rs) and the main memory's (Remember across
//! chats, src/llm/memory_pass.rs). Showing a picture needs a plain copy (the
//! chat and the model read files): those go to a temp folder that is wiped
//! at startup and at exit.

use std::path::{Path, PathBuf};

use crate::memory::MemoryKind;

/// Where an album's pictures live.
fn dir(kind: MemoryKind) -> PathBuf {
    crate::memory::root().join(match kind {
        MemoryKind::Main => "ai_memory_album",
        MemoryKind::Roleplay => "ai_chat_album",
    })
}

/// Decrypted copies of the pictures being shown.
fn temp_dir(kind: MemoryKind) -> PathBuf {
    // A test's go beside its scratch memories, never among the app's.
    #[cfg(test)]
    let base = crate::memory::root().join("clarity_album");
    #[cfg(not(test))]
    let base = std::env::temp_dir().join("clarity_album");
    match kind {
        MemoryKind::Main => base.join("main"),
        MemoryKind::Roleplay => base,
    }
}

/// A picture's name as an album file: just the file name, so a name from
/// model output can never reach outside the album.
fn safe_name(name: &str) -> Option<String> {
    let name = Path::new(name).file_name()?.to_string_lossy().to_string();
    (!name.is_empty()).then_some(name)
}

/// Copy `src` into the album, encrypted, with `why` it was kept. Returns its
/// name in the album (e.g. "img_003.jpg").
pub fn save(kind: MemoryKind, src: &Path, why: &str) -> Result<String, String> {
    let dir = dir(kind);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_ascii_lowercase();
    // One past the highest number in use — counting files would reuse a
    // number (and overwrite a kept picture) once any had been forgotten.
    let n = std::fs::read_dir(&dir)
        .map(|d| {
            d.filter_map(|e| {
                let name = e.ok()?.file_name().to_string_lossy().to_string();
                name.strip_prefix("img_")?.split('.').next()?.parse::<u32>().ok()
            })
            .max()
            .unwrap_or(0)
        })
        .unwrap_or(0)
        + 1;
    let name = format!("img_{n:03}.{ext}");

    let bytes = std::fs::read(src).map_err(|e| format!("read image: {e}"))?;
    crate::secret::write_private(dir.join(format!("{name}.enc")), crate::secret::protect_bytes(&bytes))
        .map_err(|e| format!("save image: {e}"))?;
    crate::secret::write_private(dir.join(format!("img_{n:03}.txt.enc")), crate::secret::protect_bytes(why.as_bytes()))
        .map_err(|e| format!("save description: {e}"))?;
    Ok(name)
}

/// A plain copy of picture `name` to show or send, decrypted on first use.
pub fn load(kind: MemoryKind, name: &str) -> Option<PathBuf> {
    let name = safe_name(name)?;
    let tmp = temp_dir(kind).join(&name);
    if tmp.is_file() {
        return Some(tmp);
    }
    let stored = std::fs::read(dir(kind).join(format!("{name}.enc"))).ok()?;
    let bytes = crate::secret::unprotect_bytes(&stored)?;
    std::fs::create_dir_all(temp_dir(kind)).ok()?;
    // Owner-only on Linux/macOS (a shared /tmp).
    crate::secret::write_private(&tmp, bytes).ok()?;
    Some(tmp)
}

/// Why picture `name` was kept, as written when it was.
pub fn why(kind: MemoryKind, name: &str) -> Option<String> {
    let name = safe_name(name)?;
    let stem = name.split('.').next().unwrap_or(&name);
    let stored = std::fs::read(dir(kind).join(format!("{stem}.txt.enc"))).ok()?;
    String::from_utf8(crate::secret::unprotect_bytes(&stored)?).ok()
}

/// Delete picture `name`, its note and any plain copy.
pub fn remove(kind: MemoryKind, name: &str) {
    let Some(name) = safe_name(name) else { return };
    let stem = name.split('.').next().unwrap_or(&name).to_string();
    let dir = dir(kind);
    let _ = std::fs::remove_file(dir.join(format!("{name}.enc")));
    let _ = std::fs::remove_file(dir.join(format!("{stem}.txt.enc")));
    let _ = std::fs::remove_file(temp_dir(kind).join(&name));
}

/// Delete the whole album (all its memories were cleared).
pub fn clear(kind: MemoryKind) {
    let _ = std::fs::remove_dir_all(dir(kind));
    let _ = std::fs::remove_dir_all(temp_dir(kind));
}

/// Delete every plain copy, of both albums.
pub fn clear_temp() {
    let _ = std::fs::remove_dir_all(temp_dir(MemoryKind::Roleplay));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryKind::Main;

    /// A kept picture comes back the same, with why, from its own album
    /// only; removing it leaves nothing behind.
    #[test]
    fn keeping_a_picture() {
        let _root = crate::memory::scratch_root("album");
        let src = crate::memory::root().join("dog.PNG");
        std::fs::write(&src, b"\x89PNG not really").unwrap();
        let name = save(Main, &src, "Their dog Max, asleep.").unwrap();
        assert_eq!(name, "img_001.png");
        assert_eq!(save(Main, &src, "again").unwrap(), "img_002.png");
        let stored = std::fs::read(dir(Main).join("img_001.png.enc")).unwrap();
        assert_ne!(stored, b"\x89PNG not really", "stored encrypted");
        assert_eq!(std::fs::read(load(Main, &name).unwrap()).unwrap(), b"\x89PNG not really");
        assert_eq!(why(Main, &name).as_deref(), Some("Their dog Max, asleep."));
        assert!(load(MemoryKind::Roleplay, &name).is_none(), "albums are apart");
        assert!(load(Main, "../../secret.png").is_none());
        remove(Main, &name);
        assert!(load(Main, &name).is_none() && why(Main, &name).is_none());
        clear(Main);
        assert!(!dir(Main).exists());
    }
}
