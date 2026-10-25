//! Reading and writing `.txt` tag sidecars without ever losing what's in them.
//!
//! Two rules:
//!
//! * **Read anything.** Older tag tools save sidecars as ANSI (Windows-1252)
//!   or UTF-16, not UTF-8. Those decode here (BOM-sniffed UTF-16, UTF-8, then
//!   a Windows-1252 fallback) instead of failing — a failed read used to show
//!   an empty tag box that then saved over the real tags. A sidecar that can't
//!   be *read* at all (permissions, I/O) is an error the callers must not save
//!   over.
//! * **Change only what changed.** [`apply_tags`] edits the original text in
//!   place — appending added tags, cutting removed ones, renaming one in situ —
//!   so a natural-language caption (Flux / JoyCaption training sets) isn't
//!   chopped apart at every comma just because one tag was added.

use std::path::Path;

/// How a sidecar's bytes were encoded — written back the same way, except
/// Windows-1252, which is upgraded to UTF-8 (identical for plain ASCII tags).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Encoding {
    Utf8,
    Utf8Bom,
    Utf16Le,
    Utf16Be,
    Windows1252,
}

/// A sidecar's text plus how it was stored.
#[derive(Clone, Debug)]
pub struct Sidecar {
    pub text: String,
    pub encoding: Encoding,
    /// The file existed (a missing sidecar reads as empty text).
    pub existed: bool,
}

/// Read the sidecar `txt` (through the archive layer, so sidecars inside a
/// mounted zip read too). Missing → empty; unreadable → `Err`.
pub fn read(txt: &Path) -> Result<Sidecar, String> {
    let bytes = match crate::archive::read(txt) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Sidecar { text: String::new(), encoding: Encoding::Utf8, existed: false });
        }
        Err(e) => return Err(format!("Couldn't read {}: {e}", file_name(txt))),
    };
    let (text, encoding) = decode(&bytes);
    Ok(Sidecar { text, encoding, existed: true })
}

/// Read just the text (empty for a missing or unreadable sidecar) — for
/// display-only uses like search.
pub fn read_text_lossy(txt: &Path) -> String {
    read(txt).map(|s| s.text).unwrap_or_default()
}

/// Write `text` to `txt` in `encoding` (Windows-1252 is written as UTF-8).
/// The file is replaced in one step (see [`replace_file`]), never truncated
/// in place.
pub fn write(txt: &Path, text: &str, encoding: Encoding) -> std::io::Result<()> {
    let bytes: Vec<u8> = match encoding {
        Encoding::Utf8 | Encoding::Windows1252 => text.as_bytes().to_vec(),
        Encoding::Utf8Bom => [&[0xEF, 0xBB, 0xBF][..], text.as_bytes()].concat(),
        Encoding::Utf16Le => {
            let mut b = vec![0xFF, 0xFE];
            b.extend(text.encode_utf16().flat_map(|u| u.to_le_bytes()));
            b
        }
        Encoding::Utf16Be => {
            let mut b = vec![0xFE, 0xFF];
            b.extend(text.encode_utf16().flat_map(|u| u.to_be_bytes()));
            b
        }
    };
    replace_file(txt, &bytes)
}

/// Replace `path`'s contents with `bytes`: they go to a temporary file in
/// the same folder, which is then renamed over `path`. A crash mid-write
/// leaves the old file (plus a stray `.tmp`), where truncating in place left
/// an empty or half-written `.txt`. On failure the temporary file is removed
/// and `path` is untouched.
///
/// An existing file is written in place, the old way, where a replace can't
/// work: it has other hard links (a rename would split it from them, and
/// they'd keep the old tags), the folder allows writing the file but not
/// creating one next to it, or the rename is refused (on Windows, another
/// program holding the `.txt` open without delete sharing).
fn replace_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    // A symlinked sidecar: replace the file it points at and keep the link
    // (renaming over the link would swap it for a plain file).
    let path = match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => std::fs::canonicalize(path)?,
        _ => path.to_path_buf(),
    };
    let existing = std::fs::metadata(&path).ok();
    // A read-only sidecar stays as it is, as with the in-place write (a
    // rename would replace it anyway).
    if existing.as_ref().is_some_and(|m| m.permissions().readonly()) {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "the file is read-only"));
    }
    let is_file = existing.as_ref().is_some_and(std::fs::Metadata::is_file);
    #[cfg(unix)]
    if is_file && existing.as_ref().is_some_and(|m| std::os::unix::fs::MetadataExt::nlink(m) > 1) {
        return write_in_place(&path, bytes);
    }
    let (tmp, mut file) = match create_temp(&path, existing.as_ref()) {
        Ok(created) => created,
        Err(_) if is_file => return write_in_place(&path, bytes),
        Err(e) => return Err(e),
    };
    let written = file.write_all(bytes);
    drop(file);
    if let Err(e) = written {
        // Out of space, most likely — writing in place would fail the same
        // way, after cutting the old file short. Leave it as it is.
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    match rename_over(&tmp, &path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            if is_file { write_in_place(&path, bytes) } else { Err(e) }
        }
    }
}

/// Overwrite an existing file's contents where it is: the new bytes first,
/// then the file cut to their length, so it's never left empty.
fn write_in_place(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.write_all(bytes)?;
    file.set_len(bytes.len() as u64)
}

/// A new, empty file next to `path` (the same folder, so the rename stays on
/// one volume) under a name no other write is using, carrying over what the
/// in-place write kept: the Unix permission bits, or the Windows hidden /
/// system attributes.
fn create_temp(path: &Path, existing: Option<&std::fs::Metadata>) -> std::io::Result<(std::path::PathBuf, std::fs::File)> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = file_name(path);
    let mut tries = 0;
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_file_name(format!(".{name}.{}-{n}.tmp", std::process::id()));
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(windows)]
        if let Some(m) = existing {
            use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
            const HIDDEN_SYSTEM: u32 = 0x2 | 0x4;
            opts.attributes(m.file_attributes() & HIDDEN_SYSTEM);
        }
        match opts.open(&tmp) {
            Ok(file) => {
                if cfg!(unix) && let Some(m) = existing {
                    let _ = file.set_permissions(m.permissions());
                }
                return Ok((tmp, file));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && tries < 100 => tries += 1,
            Err(e) => return Err(e),
        }
    }
}

/// `std::fs::rename`, retried for a moment on Windows: an antivirus scan or
/// the search indexer can hold the just-written file open briefly, failing
/// the rename with "access denied" / a sharing violation.
fn rename_over(from: &Path, to: &Path) -> std::io::Result<()> {
    let mut tries = 0;
    loop {
        match std::fs::rename(from, to) {
            // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
            Err(e) if cfg!(windows) && tries < 5 && matches!(e.raw_os_error(), Some(5 | 32 | 33)) => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            result => return result,
        }
    }
}

/// Decode sidecar bytes: UTF-16 by BOM, UTF-8 (with or without BOM), else
/// Windows-1252 — which maps every byte, so decoding never fails.
pub fn decode(bytes: &[u8]) -> (String, Encoding) {
    let utf16 = |body: &[u8], le: bool| {
        let units = body
            .chunks_exact(2)
            .map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) });
        char::decode_utf16(units).map(|c| c.unwrap_or('\u{FFFD}')).collect::<String>()
    };
    if let Some(body) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return (utf16(body, true), Encoding::Utf16Le);
    }
    if let Some(body) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return (utf16(body, false), Encoding::Utf16Be);
    }
    if let Some(body) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF])
        && let Ok(s) = std::str::from_utf8(body)
    {
        return (s.to_string(), Encoding::Utf8Bom);
    }
    if let Ok(s) = std::str::from_utf8(bytes) {
        return (s.to_string(), Encoding::Utf8);
    }
    (bytes.iter().map(|&b| cp1252(b)).collect(), Encoding::Windows1252)
}

/// One Windows-1252 byte as a char (0x80–0x9F hold the typographic extras;
/// everything else matches Latin-1).
fn cp1252(b: u8) -> char {
    const HIGH: [char; 32] = [
        '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8D}', 'Ž', '\u{8F}',
        '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9D}', 'ž', 'Ÿ',
    ];
    match b {
        0x80..=0x9F => HIGH[(b - 0x80) as usize],
        _ => b as char,
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// The comma/newline-separated segments of `text` with their byte spans
/// (trimmed — the span covers just the segment's words).
fn segments(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ','))) {
        if c == ',' || c == '\n' {
            let raw = &text[start..i];
            let lead = raw.len() - raw.trim_start().len();
            let t = raw.trim();
            if !t.is_empty() {
                out.push((start + lead, start + lead + t.len()));
            }
            start = i + c.len_utf8();
        }
    }
    out
}

/// Rewrite `old` so its tags become `new_tags`, touching as little of the
/// original text as possible: a one-for-one swap renames the segment in
/// place, removed tags are cut out with their separator, and added tags are
/// appended. Only a pure reorder falls back to a clean `a, b, c` rewrite.
pub fn apply_tags(old: &str, new_tags: &[String]) -> String {
    let segs = segments(old);
    let seg_text = |&(a, b): &(usize, usize)| &old[a..b];
    let has = |list: &[String], t: &str| list.iter().any(|x| x.eq_ignore_ascii_case(t));
    let old_tags: Vec<&str> = segs.iter().map(seg_text).collect();
    let removed: Vec<usize> =
        (0..segs.len()).filter(|&i| !has(new_tags, old_tags[i])).collect();
    let added: Vec<&String> = new_tags
        .iter()
        .filter(|t| !old_tags.iter().any(|o| o.eq_ignore_ascii_case(t)))
        .collect();

    if removed.is_empty() && added.is_empty() {
        // Same set: keep the file unless the order (or case) was changed.
        let mut seen: Vec<&str> = Vec::new();
        for t in &old_tags {
            if !seen.iter().any(|s| s.eq_ignore_ascii_case(t)) {
                seen.push(t);
            }
        }
        let same = seen.len() == new_tags.len() && seen.iter().zip(new_tags).all(|(a, b)| *a == b.as_str());
        return if same { old.to_string() } else { new_tags.join(", ") };
    }

    // A rename: one tag out, one in — swap the words in place.
    if removed.len() == 1 && added.len() == 1 {
        let (a, b) = segs[removed[0]];
        return format!("{}{}{}", &old[..a], added[0], &old[b..]);
    }

    // Cut removed segments, each with one separator: the one after it, or —
    // for removed segments after the last kept one — the one before it, from
    // the end of the last kept segment. The ranges are worked out on the
    // original text and merged where they meet (several trailing tags removed
    // at once share one cut), then cut back to front so earlier spans stay
    // valid.
    let mut text = old.to_string();
    let last_kept = (0..segs.len()).rev().find(|i| !removed.contains(i));
    let mut cuts: Vec<(usize, usize)> = removed
        .iter()
        .map(|&i| {
            let (a, b) = segs[i];
            match last_kept {
                // Up to the next segment: drops "tag, " cleanly.
                Some(k) if i < k => (a, segs[i + 1].0),
                // After the last kept tag: drop the separator before it.
                Some(k) => (segs[k].1, b),
                // Every tag goes: the words and what separates them.
                None => (a, segs.get(i + 1).map_or(b, |s| s.0)),
            }
        })
        .collect();
    cuts.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in cuts {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    for &(a, b) in merged.iter().rev() {
        text.replace_range(a..b, "");
    }
    for t in added {
        let base = text.trim_end();
        let tail = &text[base.len()..];
        text = if base.is_empty() {
            format!("{t}{tail}")
        } else if base.ends_with(',') {
            format!("{base} {t}{tail}")
        } else {
            format!("{base}, {t}{tail}")
        };
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn decodes_ansi_and_utf16() {
        // "café" in Windows-1252 (é = 0xE9) — invalid UTF-8.
        let (s, e) = decode(&[b'c', b'a', b'f', 0xE9, b',', b' ', 0x93, b'q', 0x94]);
        assert_eq!((s.as_str(), e), ("café, “q”", Encoding::Windows1252));
        let le: Vec<u8> = [0xFF, 0xFE].into_iter().chain("1girl, 猫".encode_utf16().flat_map(u16::to_le_bytes)).collect();
        assert_eq!(decode(&le), ("1girl, 猫".to_string(), Encoding::Utf16Le));
        assert_eq!(decode("plain, tags".as_bytes()).1, Encoding::Utf8);
        assert_eq!(decode(b"\xEF\xBB\xBFbom").0, "bom");
    }

    #[test]
    fn utf16_round_trips() {
        let dir = std::env::temp_dir().join("clarity_sidecar_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.txt");
        write(&p, "1girl, 猫", Encoding::Utf16Le).unwrap();
        let s = read(&p).unwrap();
        assert_eq!((s.text.as_str(), s.encoding, s.existed), ("1girl, 猫", Encoding::Utf16Le, true));
        let missing = read(&dir.join("none.txt")).unwrap();
        assert!(!missing.existed && missing.text.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fresh, empty scratch folder for one test.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("clarity_sidecar_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Temporary files a write left behind in `dir`.
    fn leftovers(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect()
    }

    #[test]
    fn write_replaces_the_whole_file_in_its_encoding() {
        let dir = scratch("replace");
        let p = dir.join("a.txt");
        std::fs::write(&p, "a much longer original caption, with, many, tags").unwrap();
        write(&p, "1girl, 猫", Encoding::Utf16Le).unwrap();
        let utf16: Vec<u8> = [0xFF, 0xFE].into_iter().chain("1girl, 猫".encode_utf16().flat_map(u16::to_le_bytes)).collect();
        assert_eq!(std::fs::read(&p).unwrap(), utf16);
        write(&p, "bom", Encoding::Utf8Bom).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"\xEF\xBB\xBFbom");
        write(&dir.join("new.txt"), "fresh", Encoding::Windows1252).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("new.txt")).unwrap(), "fresh");
        assert!(leftovers(&dir).is_empty(), "{:?}", leftovers(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_write_leaves_the_old_file_and_no_temp() {
        let dir = scratch("fail");
        // A folder in the way: the final rename fails.
        std::fs::create_dir_all(dir.join("taken.txt")).unwrap();
        assert!(write(&dir.join("taken.txt"), "x", Encoding::Utf8).is_err());
        assert!(dir.join("taken.txt").is_dir());
        // A read-only sidecar is left alone, as the in-place write did.
        let ro = dir.join("ro.txt");
        std::fs::write(&ro, "keep").unwrap();
        let original = std::fs::metadata(&ro).unwrap().permissions();
        let mut read_only = original.clone();
        read_only.set_readonly(true);
        std::fs::set_permissions(&ro, read_only).unwrap();
        assert!(write(&ro, "x", Encoding::Utf8).is_err());
        assert_eq!(std::fs::read_to_string(&ro).unwrap(), "keep");
        std::fs::set_permissions(&ro, original).unwrap();
        assert!(leftovers(&dir).is_empty(), "{:?}", leftovers(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn write_keeps_a_symlink_and_permissions() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = scratch("link");
        std::fs::write(dir.join("shared.txt"), "old").unwrap();
        std::fs::set_permissions(dir.join("shared.txt"), std::fs::Permissions::from_mode(0o640)).unwrap();
        std::os::unix::fs::symlink(dir.join("shared.txt"), dir.join("a.txt")).unwrap();
        write(&dir.join("a.txt"), "new", Encoding::Utf8).unwrap();
        assert!(std::fs::symlink_metadata(dir.join("a.txt")).unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read_to_string(dir.join("shared.txt")).unwrap(), "new");
        let mode = std::fs::metadata(dir.join("shared.txt")).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o640);
        assert!(leftovers(&dir).is_empty(), "{:?}", leftovers(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A sidecar with another hard link is written in place: both names keep
    /// showing the same (new) tags.
    #[cfg(unix)]
    #[test]
    fn write_keeps_hard_links_together() {
        let dir = scratch("hardlink");
        std::fs::write(dir.join("a.txt"), "a much longer old caption").unwrap();
        std::fs::hard_link(dir.join("a.txt"), dir.join("b.txt")).unwrap();
        write(&dir.join("a.txt"), "new", Encoding::Utf8).unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("a.txt")).unwrap(), "new");
        assert_eq!(std::fs::read_to_string(dir.join("b.txt")).unwrap(), "new");
        assert!(leftovers(&dir).is_empty(), "{:?}", leftovers(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn adding_a_tag_keeps_a_caption_intact() {
        let caption = "A woman sits on a boat, her feet resting on the side, at golden hour.\n";
        let tags = super::segments(caption).iter().map(|&(a, b)| caption[a..b].to_string()).collect::<Vec<_>>();
        let mut new = tags.clone();
        new.push("lake".into());
        assert_eq!(
            apply_tags(caption, &new),
            "A woman sits on a boat, her feet resting on the side, at golden hour., lake\n"
        );
    }

    #[test]
    fn remove_rename_and_unchanged() {
        let old = "1girl, solo,  red hair\nsmile";
        assert_eq!(apply_tags(old, &v(&["1girl", "red hair", "smile"])), "1girl, red hair\nsmile");
        assert_eq!(apply_tags(old, &v(&["1girl", "solo", "red hair"])), "1girl, solo,  red hair");
        assert_eq!(apply_tags(old, &v(&["1girl", "solo", "blue hair", "smile"])), "1girl, solo,  blue hair\nsmile");
        assert_eq!(apply_tags(old, &v(&["1girl", "solo", "red hair", "smile"])), old);
        assert_eq!(apply_tags(old, &v(&["solo", "1girl", "red hair", "smile"])), "solo, 1girl, red hair, smile");
        assert_eq!(apply_tags("", &v(&["a", "b"])), "a, b");
    }

    /// Several tags removed at once — at the end, in a row, or all of them —
    /// with additions in the same write (the AI chat's edits and undos do
    /// this) cut cleanly instead of overlapping (which used to panic).
    #[test]
    fn several_removals_at_once() {
        let old = "a, b, c, d";
        assert_eq!(apply_tags(old, &v(&["a", "b"])), "a, b");
        assert_eq!(apply_tags(old, &v(&["a", "c"])), "a, c");
        assert_eq!(apply_tags(old, &v(&["d"])), "d");
        assert_eq!(apply_tags(old, &v(&["x"])), "x");
        assert_eq!(apply_tags("a, b, c\n", &v(&["a"])), "a\n");
        let tagged = "scenery, sunset, mountain, lake, orange sky, reflection, no humans, outdoors, golden hour, sun";
        assert_eq!(
            apply_tags(
                tagged,
                &v(&["scenery", "sunset", "mountain", "lake", "orange sky", "reflection", "no humans", "outdoors", "cloud"])
            ),
            "scenery, sunset, mountain, lake, orange sky, reflection, no humans, outdoors, cloud"
        );
    }
}
