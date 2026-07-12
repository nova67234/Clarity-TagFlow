//! Live spell-check for the Generate (Flux) and Z-Image prompt boxes — a port of
//! terminus2's SpellChecker.java. Unknown words get a red wavy underline; a
//! right-click on one offers the closest dictionary words (Levenshtein distance)
//! plus "Add to dictionary". The Java original downloaded its 10k-word English
//! list at startup; here the same list is embedded at build time, and words the
//! user adds are persisted one-per-line in the config dir.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

use eframe::egui;
use egui::text::CCursor;
use egui::text_edit::TextEditOutput;
use egui::{Color32, Pos2, Shape, Stroke};

/// google-10000-english-no-swears.txt — the list SpellChecker.java fetched.
const WORD_LIST: &str = include_str!("../assets/google-10000-english-no-swears.txt");

/// Same red as the Java squiggle painter.
const SQUIGGLE: Color32 = Color32::from_rgb(239, 68, 68);

/// Per-text-field state: the word under the last right-click, driving the menu.
#[derive(Default)]
pub struct SpellcheckState {
    menu: Option<MenuTarget>,
}

struct MenuTarget {
    byte_range: (usize, usize),
    word: String,
    suggestions: Vec<String>,
}

/// Run the spell-check over a just-shown `TextEdit`: paint squiggles under
/// unknown words and handle the right-click suggestion menu (which may rewrite
/// `text`). Call immediately after `TextEdit::show`, from the same `Ui`.
pub fn attach(ui: &egui::Ui, out: &TextEditOutput, text: &mut String, state: &mut SpellcheckState) {
    let words = misspelled_words(text);
    let resp = &out.response.response;

    // --- Squiggles (clipped like the text itself, so they scroll with it). ---
    if !words.is_empty() {
        let painter = ui.painter().with_clip_rect(out.text_clip_rect);
        for w in &words {
            // One segment per visual row: walk the word's cursor rects and break
            // wherever the row (bottom y) changes — long words can wrap mid-word.
            let mut seg = out.galley.pos_from_cursor(CCursor::new(w.char_start));
            let mut prev_x = seg.left();
            for i in (w.char_start + 1)..=w.char_end {
                let r = out.galley.pos_from_cursor(CCursor::new(i));
                if (r.bottom() - seg.bottom()).abs() > 0.5 {
                    squiggle(&painter, out.galley_pos, seg.left(), prev_x, seg.bottom());
                    seg = r;
                }
                prev_x = r.left();
            }
            squiggle(&painter, out.galley_pos, seg.left(), prev_x, seg.bottom());
        }
    }

    // --- Right-click: capture the misspelled word under the pointer (if any). ---
    if resp.secondary_clicked() {
        state.menu = resp.interact_pointer_pos().and_then(|pos| {
            let cc = out.galley.cursor_from_pos(pos - out.galley_pos);
            words
                .iter()
                .find(|w| cc.index >= w.char_start && cc.index <= w.char_end)
                .map(|w| {
                    let word = text[w.byte_start..w.byte_end].to_string();
                    MenuTarget {
                        byte_range: (w.byte_start, w.byte_end),
                        suggestions: suggestions(&word.to_lowercase()),
                        word,
                    }
                })
        });
    }

    // Only attach the menu when the click landed on a misspelled word — a plain
    // right-click elsewhere in the prompt shows nothing (matches the Java UX).
    let mut replace: Option<((usize, usize), String)> = None;
    if let Some(menu) = &state.menu {
        resp.context_menu(|ui| {
            if menu.suggestions.is_empty() {
                ui.add_enabled(false, egui::Button::new("No suggestions"));
            } else {
                for s in &menu.suggestions {
                    let cased = match_case(&menu.word, s);
                    if ui.button(&cased).clicked() {
                        replace = Some((menu.byte_range, cased));
                        ui.close();
                    }
                }
            }
            ui.separator();
            if ui.button("Add to dictionary").clicked() {
                add_to_dictionary(&menu.word);
                ui.close();
            }
        });
    }
    if let Some(((start, end), with)) = replace {
        if end <= text.len() && text.is_char_boundary(start) && text.is_char_boundary(end) {
            text.replace_range(start..end, &with);
        }
        state.menu = None;
    }
}

/// A run of alphabetic chars not in the dictionary, with both index spaces:
/// chars for galley cursors, bytes for `String::replace_range`.
struct Misspelled {
    char_start: usize,
    char_end: usize,
    byte_start: usize,
    byte_end: usize,
}

/// Scan `text` for unknown words. A word is a run of alphabetic chars, but only
/// pure-ASCII words are checked — the dictionary is English, so flagging the
/// fragments of accented words ("café") would squiggle them forever (the Java
/// `\b[a-zA-Z]+\b` regex had exactly that bug). Single letters are never
/// flagged — they're mostly the orphaned tails of contractions ("don't").
fn misspelled_words(text: &str) -> Vec<Misspelled> {
    let mut out = Vec::new();
    let mut word: Option<(usize, usize)> = None; // (char_start, byte_start)
    let mut char_idx = 0;
    for (byte_idx, ch) in text.char_indices() {
        if ch.is_alphabetic() {
            word.get_or_insert((char_idx, byte_idx));
        } else if let Some((cs, bs)) = word.take() {
            push_if_unknown(&mut out, text, cs, char_idx, bs, byte_idx);
        }
        char_idx += 1;
    }
    if let Some((cs, bs)) = word {
        push_if_unknown(&mut out, text, cs, char_idx, bs, text.len());
    }
    out
}

fn push_if_unknown(out: &mut Vec<Misspelled>, text: &str, cs: usize, ce: usize, bs: usize, be: usize) {
    let word = &text[bs..be];
    let ascii = word.chars().all(|c| c.is_ascii_alphabetic());
    if ascii && word.len() >= 2 && !is_known(&word.to_ascii_lowercase()) {
        out.push(Misspelled { char_start: cs, char_end: ce, byte_start: bs, byte_end: be });
    }
}

fn is_known(word_lower: &str) -> bool {
    base_dictionary().contains(word_lower)
        || custom_words().read().is_ok_and(|set| set.contains(word_lower))
}

/// The zigzag underline: 2px rise/fall every 2px, like the Java painter.
fn squiggle(painter: &egui::Painter, galley_pos: Pos2, x1: f32, x2: f32, bottom: f32) {
    if x2 - x1 < 1.0 {
        return;
    }
    let y = galley_pos.y + bottom - 2.0;
    let mut points = Vec::with_capacity(((x2 - x1) / 2.0) as usize + 2);
    let mut x = galley_pos.x + x1;
    let end = galley_pos.x + x2;
    let mut up = true;
    while x < end + 2.0 {
        points.push(Pos2::new(x.min(end), if up { y } else { y + 2.0 }));
        x += 2.0;
        up = !up;
    }
    painter.add(Shape::line(points, Stroke::new(1.0, SQUIGGLE)));
}

/// Closest dictionary words by Levenshtein distance — same filters as the Java
/// version: length within ±2, matching first letter (for words > 2 chars),
/// distance ≤ 3, best 5.
fn suggestions(misspelled: &str) -> Vec<String> {
    let n = misspelled.chars().count();
    let (min_len, max_len) = (n.saturating_sub(2).max(1), n + 2);
    let first = misspelled.chars().next();
    let mut scored: Vec<(usize, String)> = Vec::new();
    let mut consider = |w: &str| {
        let wn = w.chars().count();
        if wn < min_len || wn > max_len {
            return;
        }
        if n > 2 && w.chars().next() != first {
            return;
        }
        let d = levenshtein(misspelled, w);
        if d <= 3 {
            scored.push((d, w.to_string()));
        }
    };
    for w in base_dictionary() {
        consider(w);
    }
    if let Ok(custom) = custom_words().read() {
        for w in custom.iter() {
            consider(w);
        }
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.truncate(5);
    scored.into_iter().map(|(_, w)| w).collect()
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut costs: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        costs[0] = i + 1;
        let mut nw = i;
        for (j, &cb) in b.iter().enumerate() {
            let cj = (1 + costs[j + 1].min(costs[j])).min(if ca == cb { nw } else { nw + 1 });
            nw = costs[j + 1];
            costs[j + 1] = cj;
        }
    }
    costs[b.len()]
}

/// Recase `suggestion` to match how the user typed `original`:
/// "Cinematc" → "Cinematic", "CINEMATC" → "CINEMATIC".
fn match_case(original: &str, suggestion: &str) -> String {
    let mut chars = original.chars();
    match (chars.next(), chars.next()) {
        (Some(c0), Some(c1)) if c0.is_uppercase() && c1.is_uppercase() => suggestion.to_uppercase(),
        (Some(c0), _) if c0.is_uppercase() => {
            let mut s = suggestion.chars();
            match s.next() {
                Some(f) => f.to_uppercase().chain(s).collect(),
                None => String::new(),
            }
        }
        _ => suggestion.to_string(),
    }
}

// --- Dictionary storage. ---

fn base_dictionary() -> &'static HashSet<&'static str> {
    static DICT: OnceLock<HashSet<&'static str>> = OnceLock::new();
    DICT.get_or_init(|| {
        WORD_LIST
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect()
    })
}

/// Words added via "Add to dictionary", persisted across runs.
fn custom_words() -> &'static RwLock<HashSet<String>> {
    static CUSTOM: OnceLock<RwLock<HashSet<String>>> = OnceLock::new();
    CUSTOM.get_or_init(|| {
        let set = std::fs::read_to_string(custom_words_path())
            .map(|s| {
                s.lines()
                    .map(|l| l.trim().to_lowercase())
                    .filter(|l| !l.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        RwLock::new(set)
    })
}

fn custom_words_path() -> PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow"))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("spellcheck_custom_words.txt")
}

fn add_to_dictionary(word: &str) {
    let Ok(mut set) = custom_words().write() else {
        return;
    };
    if set.insert(word.to_lowercase()) {
        let path = custom_words_path();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut lines: Vec<&str> = set.iter().map(String::as_str).collect();
        lines.sort_unstable();
        let _ = std::fs::write(path, lines.join("\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scans_unknown_words_with_both_index_spaces() {
        // "café" exercises two things: char/byte indices diverging, and accented
        // words being exempt from the English dictionary check.
        let text = "café zzxqv ok";
        let words = misspelled_words(text);
        assert_eq!(words.len(), 1);
        let w = &words[0];
        assert_eq!(&text[w.byte_start..w.byte_end], "zzxqv");
        assert_eq!((w.char_start, w.char_end), (5, 10));
    }

    #[test]
    fn known_and_single_letter_words_pass() {
        assert!(misspelled_words("the quick brown fox").is_empty());
        // orphaned contraction tail: "don't" → "don" + "t"; "t" must not be flagged
        assert!(misspelled_words("don't").is_empty());
    }

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("same", "same"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn suggestions_find_close_words() {
        // "worde" → "word"/"words" should rank near the top (distance 1)
        let s = suggestions("worde");
        assert!(s.iter().any(|w| w == "word" || w == "words"), "{s:?}");
        assert!(s.len() <= 5);
    }

    #[test]
    fn case_matching() {
        assert_eq!(match_case("Cinematc", "cinematic"), "Cinematic");
        assert_eq!(match_case("CINEMATC", "cinematic"), "CINEMATIC");
        assert_eq!(match_case("cinematc", "cinematic"), "cinematic");
    }
}
