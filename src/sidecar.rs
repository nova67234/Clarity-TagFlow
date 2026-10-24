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
    std::fs::write(txt, bytes)
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

    // Cut removed segments (each with the separator after it, or before it
    // for the last one), back to front so earlier spans stay valid.
    let mut text = old.to_string();
    for &i in removed.iter().rev() {
        let (a, b) = segs[i];
        let (cut_a, cut_b) = match segs.get(i + 1) {
            // Up to the next segment: drops "tag, " cleanly.
            Some(&(next, _)) => (a, next),
            // Last segment: drop the separator before it instead.
            None => (
                old[..a].trim_end().trim_end_matches([',', '\n']).trim_end().len(),
                b,
            ),
        };
        text.replace_range(cut_a..cut_b, "");
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
}
