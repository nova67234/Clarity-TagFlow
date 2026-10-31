//! Picking what to show the AI from the pages read: each page is cut into
//! passages of a few sentences, every passage is scored against the
//! question and the searches (BM25, the classic search-engine ranking), and
//! the best ones are kept within a size budget — a few per page at most, so
//! one long page can't crowd out the rest.

use std::collections::{HashMap, HashSet};

/// One page's text, with where it came in the search results (0 = first).
pub(super) struct Doc<'a> {
    pub text: &'a str,
    pub rank: usize,
}

/// Target passage size, in characters.
const CHUNK: usize = 700;
/// Passages shorter than this are dropped (menus, captions, "Share").
const MIN_CHUNK: usize = 40;

/// Words that say nothing about what's asked.
const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "before", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "new", "no", "not", "now", "of", "on",
    "one", "or", "our", "out", "she", "so", "some", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "up", "us", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your", "tell", "please", "explain", "find", "search", "show", "give", "know",
    "latest", "web", "look", "much", "many",
];

/// The words of `text` worth matching: lowercase, letters and digits, no
/// stopwords. Runs of CJK characters (written without spaces) become
/// overlapping character pairs.
pub(super) fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let lower = word.to_lowercase();
        if lower.chars().any(is_cjk) {
            let chars: Vec<char> = lower.chars().collect();
            if chars.len() == 1 {
                out.push(lower);
            }
            for pair in chars.windows(2) {
                out.push(pair.iter().collect());
            }
            continue;
        }
        if lower.chars().count() < 2 || STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        out.push(lower);
    }
    out
}

fn is_cjk(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{ac00}'..='\u{d7af}'
        | '\u{f900}'..='\u{faff}')
}

/// `text` cut into passages of about [`CHUNK`] characters, at line breaks
/// where it can and sentence ends where a line is too long. Each with its
/// position in the page.
pub(super) fn chunks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        let t = cur.trim();
        if t.chars().count() >= MIN_CHUNK && mostly_words(t) {
            out.push(t.to_string());
        }
        cur.clear();
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            if cur.len() >= CHUNK / 3 {
                flush(&mut cur, &mut out);
            }
            continue;
        }
        for piece in split_long(line) {
            if !cur.is_empty() && cur.len() + piece.len() > CHUNK {
                flush(&mut cur, &mut out);
            }
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(piece);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// A long line in sentence-end pieces of at most about [`CHUNK`] characters
/// (cut hard where a "sentence" runs on).
fn split_long(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = line;
    while rest.len() > CHUNK + CHUNK / 3 {
        let window_end = floor_char(rest, CHUNK + CHUNK / 3);
        let window = &rest[..window_end];
        let cut = ['.', '?', '!', '。', '？', '！', ';']
            .iter()
            .filter_map(|p| window.rfind(*p).map(|i| i + p.len_utf8()))
            .filter(|&i| i >= CHUNK / 2)
            .max()
            .or_else(|| window.rfind(' ').filter(|&i| i >= CHUNK / 2))
            .unwrap_or(floor_char(rest, CHUNK));
        out.push(rest[..cut].trim());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// The largest char boundary at or below `i`.
fn floor_char(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// At least half letters, digits and spaces (not a row of symbols).
fn mostly_words(s: &str) -> bool {
    let total = s.chars().count();
    let wordy = s.chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).count();
    wordy * 2 >= total
}

/// The best passages of `docs` for `query_terms` within `budget` characters,
/// at most `per_doc` from one page: for each doc, its picked passages in
/// page order (empty when none was picked).
pub(super) fn pick(docs: &[Doc<'_>], query_terms: &[String], budget: usize, per_doc: usize) -> Vec<Vec<String>> {
    let all: Vec<Vec<String>> = docs.iter().map(|d| chunks(d.text)).collect();
    let mut picked: Vec<Vec<(usize, String)>> = vec![Vec::new(); docs.len()];
    let wanted: HashSet<&str> = query_terms.iter().map(String::as_str).collect();

    // Score every passage (BM25 over all passages of all pages).
    let mut df: HashMap<&str, usize> = HashMap::new();
    let tokenized: Vec<Vec<Vec<String>>> = all.iter().map(|cs| cs.iter().map(|c| terms(c)).collect()).collect();
    let mut n = 0usize;
    let mut total_len = 0usize;
    for doc in &tokenized {
        for toks in doc {
            n += 1;
            total_len += toks.len();
            let seen: HashSet<&str> = toks.iter().map(String::as_str).filter(|t| wanted.contains(t)).collect();
            for t in seen {
                *df.entry(t).or_insert(0) += 1;
            }
        }
    }
    let avg = (total_len as f64 / n.max(1) as f64).max(1.0);
    let (k1, b) = (1.2, 0.75);
    let mut scored: Vec<(f64, usize, usize)> = Vec::new(); // (score, doc, chunk)
    for (di, doc) in tokenized.iter().enumerate() {
        for (ci, toks) in doc.iter().enumerate() {
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for t in toks.iter().filter(|t| wanted.contains(t.as_str())) {
                *tf.entry(t.as_str()).or_insert(0) += 1;
            }
            let len = toks.len() as f64;
            let mut s = 0.0;
            for (t, f) in tf {
                let d = *df.get(t).unwrap_or(&0) as f64;
                let idf = (1.0 + (n as f64 - d + 0.5) / (d + 0.5)).ln();
                let f = f as f64;
                s += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * len / avg));
            }
            // A little for earlier search results and for a page's opening.
            s += 0.15 / (1.0 + docs[di].rank as f64) + if ci == 0 { 0.1 } else { 0.0 };
            scored.push((s, di, ci));
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    // Every page's best passage first (in search order), then the rest by score.
    let mut used = 0usize;
    let mut taken: HashSet<(usize, usize)> = HashSet::new();
    let mut take = |di: usize, ci: usize, picked: &mut Vec<Vec<(usize, String)>>, used: &mut usize| -> bool {
        let text = &all[di][ci];
        if taken.contains(&(di, ci)) || picked[di].len() >= per_doc || *used + text.len() > budget {
            return false;
        }
        taken.insert((di, ci));
        *used += text.len();
        picked[di].push((ci, text.clone()));
        true
    };
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_by_key(|&d| docs[d].rank);
    for d in order {
        if let Some(&(_, di, ci)) = scored.iter().find(|(_, di, _)| *di == d) {
            take(di, ci, &mut picked, &mut used);
        }
    }
    for &(_, di, ci) in &scored {
        take(di, ci, &mut picked, &mut used);
    }
    picked
        .into_iter()
        .map(|mut p| {
            p.sort_by_key(|(ci, _)| *ci);
            p.into_iter().map(|(_, t)| t).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_drop_stopwords_and_pair_cjk() {
        assert_eq!(terms("What is the population of Tokyo?"), vec!["population", "tokyo"]);
        assert_eq!(terms("東京の人口"), vec!["東京", "京の", "の人", "人口"]);
        assert_eq!(terms("GPT-5 vs. Rust 1.80"), vec!["gpt", "vs", "rust", "80"]);
    }

    #[test]
    fn chunks_respect_size_and_drop_junk() {
        let para = "This is a sentence about rust. ".repeat(60);
        let text = format!("Menu\nHome | About\n\n{para}\n\n© | | | | | | | | | | | | | | | | | | | | | | | |");
        let cs = chunks(&text);
        assert!(!cs.is_empty());
        for c in &cs {
            assert!(c.len() <= CHUNK + CHUNK / 3 + 10, "chunk of {}", c.len());
            assert!(c.chars().count() >= MIN_CHUNK);
        }
        assert!(!cs.iter().any(|c| c.starts_with('©')));
        // No text, no chunks; a run-on line without spaces is still cut.
        assert!(chunks("").is_empty());
        let runon = "x".repeat(5000);
        assert!(chunks(&runon).iter().all(|c| c.len() <= CHUNK + CHUNK / 3));
        // Multi-byte text is cut on character boundaries.
        let jp = "東京は日本の首都です。".repeat(200);
        let _ = chunks(&jp);
    }

    #[test]
    fn relevant_passages_win_and_budget_holds() {
        let filler = "Cooking pasta needs salted water and patience. ".repeat(20);
        let relevant = "The Eiffel Tower is 330 metres tall and was finished in 1889. ".repeat(3);
        let page_a = format!("{filler}\n\n{relevant}\n\n{filler}");
        let page_b = format!("{filler}\n\n{filler}");
        let docs = [Doc { text: &page_b, rank: 0 }, Doc { text: &page_a, rank: 1 }];
        let q = terms("How tall is the Eiffel Tower?");
        let picked = pick(&docs, &q, 1200, 3);
        assert!(picked[1].iter().any(|p| p.contains("Eiffel")), "{picked:?}");
        let used: usize = picked.iter().flatten().map(|p| p.len()).sum();
        assert!(used <= 1200);
        // At most `per_doc` from one page.
        let picked = pick(&docs, &q, 100_000, 1);
        assert!(picked.iter().all(|p| p.len() <= 1));
    }

    #[test]
    fn nothing_matching_still_gives_each_page_its_opening() {
        let a = "Alpha page opening paragraph with plenty of words in it.\n\nMore alpha text here and there.";
        let b = "Beta page opening paragraph with plenty of words in it too.";
        let docs = [Doc { text: a, rank: 0 }, Doc { text: b, rank: 1 }];
        let picked = pick(&docs, &terms("zzz qqq"), 10_000, 2);
        assert!(picked[0][0].starts_with("Alpha"));
        assert!(picked[1][0].starts_with("Beta"));
    }
}
