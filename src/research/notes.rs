//! What a lookup leaves behind: the answer made safe to show, and the notes
//! file (Markdown) the user can open and review — the question, the
//! searches, the answer with its citations linked, and every source with the
//! exact excerpts the AI was given.
//!
//! Everything from the web is treated as hostile here too: in the notes,
//! excerpts sit in fenced blocks (shown as written, never rendered) and
//! titles are escaped, so a page can't plant a remote image (a tracking
//! pixel when the notes are opened), raw HTML or a disguised link.

use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::net::{check_url, normalized};
use super::{Outcome, Report, Source};

/// The Markdown extensions the chat renders with (egui_commonmark's), and
/// `$…$` math, which it adds where math is drawn (the chat; the notes viewer
/// has none). Text is made safe under both readings: one can see a link
/// where the other sees math or code.
fn options(math: bool) -> Options {
    let o = Options::ENABLE_TABLES
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_DEFINITION_LIST;
    if math { o | Options::ENABLE_MATH } else { o }
}

/// `dest` is one of the sources (an http(s) URL the research read).
fn is_source(dest: &str, sources: &[Source]) -> bool {
    let Ok(u) = check_url(dest) else { return false };
    let key = normalized(&u);
    sources.iter().any(|s| check_url(&s.url).is_ok_and(|v| normalized(&v) == key))
}

/// The AI's answer made safe to show and save: images become their alt text,
/// links stay only when they point at one of `sources` (others become their
/// text), raw HTML is escaped so it shows as text, and a web address written
/// out that isn't a source becomes code (Markdown apps that link bare
/// addresses won't). Code is left alone. Done until nothing changes, with
/// math read and not read.
pub(crate) fn clean_answer(text: &str, sources: &[Source]) -> String {
    let mut out = text.to_string();
    for _ in 0..8 {
        let mut changed = false;
        for math in [true, false] {
            if let Some(next) = clean_once(&out, sources, math) {
                out = next;
                changed = true;
            }
        }
        if !changed {
            return out;
        }
    }
    // Never settles (it always does): all of it as plain text.
    fenced(&out)
}

/// One pass of [`clean_answer`], reading `text` with or without math; `None`
/// when there's nothing to change.
fn clean_once(text: &str, sources: &[Source], math: bool) -> Option<String> {
    let mut edits: Vec<(Range<usize>, String)> = Vec::new();
    let mut code_block = 0usize;
    let mut events = Parser::new_ext(text, options(math)).into_offset_iter().peekable();
    while let Some((event, range)) = events.next() {
        match event {
            Event::Start(Tag::CodeBlock(_)) => code_block += 1,
            Event::End(TagEnd::CodeBlock) => code_block = code_block.saturating_sub(1),
            Event::Text(_) if code_block == 0 => bare_addresses(text, range, sources, &mut edits),
            Event::Start(Tag::Image { .. }) => {
                // The alt text, as plain (escaped) text.
                let mut alt = String::new();
                let mut depth = 1;
                for (e, _) in events.by_ref() {
                    match e {
                        Event::Start(Tag::Image { .. }) => depth += 1,
                        Event::End(TagEnd::Image) => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        Event::Text(t) | Event::Code(t) => alt.push_str(&t),
                        _ => {}
                    }
                }
                edits.push((range, escape_inline(&alt)));
            }
            Event::Start(Tag::Link { dest_url, .. }) if !is_source(&dest_url, sources) => {
                // Not one of the sources: keep what the link says, as text.
                let mut inner: Option<Range<usize>> = None;
                let mut depth = 1;
                for (e, r) in events.by_ref() {
                    match e {
                        Event::Start(Tag::Link { .. }) => depth += 1,
                        Event::End(TagEnd::Link) => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    inner = Some(match inner {
                        Some(i) => i.start.min(r.start)..i.end.max(r.end),
                        None => r,
                    });
                }
                let shown = match inner {
                    // An autolink's text is its URL: as code, not a link.
                    Some(i) if text[range.clone()].starts_with('<') => code_span(&text[i]),
                    Some(i) => clean_answer(&text[i], sources),
                    None => String::new(),
                };
                edits.push((range, shown));
            }
            Event::Html(h) | Event::InlineHtml(h) => {
                edits.push((range, h.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")));
            }
            _ => {}
        }
    }
    if edits.is_empty() {
        return None;
    }
    Some(apply(text, edits))
}

/// Web addresses written out in `text[range]` (`https://…`, `www.…`) that
/// aren't sources, as code edits.
fn bare_addresses(text: &str, range: Range<usize>, sources: &[Source], edits: &mut Vec<(Range<usize>, String)>) {
    let raw = &text[range.clone()];
    let lower = raw.to_ascii_lowercase();
    let mut from = 0;
    while let Some(found) = ["http://", "https://", "www."]
        .iter()
        .filter_map(|p| lower[from..].find(p).map(|i| from + i))
        .min()
    {
        // To the next space or quote, less trailing punctuation.
        let rest = &raw[found..];
        let len = rest.find(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '"' | '`')).unwrap_or(rest.len());
        let addr = rest[..len].trim_end_matches(['.', ',', ';', ':', '!', '?', ')', ']', '}', '\'', '*', '_', '~']);
        from = found + len.max(1);
        if addr.len() <= 4 {
            continue;
        }
        let full = if addr.to_ascii_lowercase().starts_with("www.") { format!("http://{addr}") } else { addr.to_string() };
        if is_source(&full, sources) {
            continue;
        }
        let at = range.start + found;
        edits.push((at..at + addr.len(), code_span(addr)));
    }
}

/// [`clean_answer`], with each citation like `[2]` (outside code and links)
/// turned into a link to that source.
pub(crate) fn display_answer(text: &str, sources: &[Source]) -> String {
    let clean = clean_answer(text, sources);
    if sources.is_empty() {
        return clean;
    }
    // Where citations mustn't be touched: code, math, links, raw HTML.
    let mut off_limits: Vec<Range<usize>> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for (event, range) in Parser::new_ext(&clean, options(true)).into_offset_iter() {
        match event {
            Event::Code(_)
            | Event::Html(_)
            | Event::InlineHtml(_)
            | Event::InlineMath(_)
            | Event::DisplayMath(_) => off_limits.push(range),
            Event::Start(Tag::CodeBlock(_) | Tag::Link { .. } | Tag::Image { .. }) => open.push(range.start),
            Event::End(TagEnd::CodeBlock | TagEnd::Link | TagEnd::Image) => {
                if let Some(start) = open.pop() {
                    off_limits.push(start..range.end);
                }
            }
            _ => {}
        }
    }
    let blocked = |at: usize| off_limits.iter().any(|r| r.contains(&at));
    let bytes = clean.as_bytes();
    let mut edits = Vec::new();
    let mut i = 0;
    while let Some(n) = memchr::memchr(b'[', &bytes[i..]) {
        let at = i + n;
        i = at + 1;
        let digits = bytes[at + 1..].iter().take(3).take_while(|b| b.is_ascii_digit()).count();
        if digits == 0 || bytes.get(at + 1 + digits) != Some(&b']') {
            continue;
        }
        let end = at + digits + 2;
        if matches!(bytes.get(end), Some(b'(' | b':')) || blocked(at) || (at > 0 && bytes[at - 1] == b'\\') {
            continue;
        }
        let Ok(n) = clean[at + 1..at + 1 + digits].parse::<usize>() else { continue };
        if let Some(s) = sources.iter().find(|s| s.n == n)
            && let Ok(u) = check_url(&s.url)
        {
            edits.push((at..end, format!("[[{n}]](<{u}>)")));
        }
    }
    // Once more through the cleaner: the new links are to sources, so they
    // stay, and nothing they ended up next to can make another.
    clean_answer(&apply(&clean, edits), sources)
}

/// `text` with the (non-overlapping) `edits` applied.
fn apply(text: &str, mut edits: Vec<(Range<usize>, String)>) -> String {
    edits.sort_by_key(|(r, _)| r.start);
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for (r, with) in edits {
        if r.start < at {
            continue; // inside an edit already made
        }
        out.push_str(&text[at..r.start]);
        out.push_str(&with);
        at = r.end;
    }
    out.push_str(&text[at..]);
    out
}

/// Text escaped for Markdown so it reads as itself: no links, images, HTML,
/// entities or code spans can start in it. Line breaks become spaces.
pub(crate) fn escape_inline(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::with_capacity(flat.len());
    let mut chars = flat.chars().peekable();
    while let Some(c) = chars.next() {
        let next = chars.peek().copied();
        let escape = match c {
            '\\' | '`' | '<' | '>' | '[' | ']' | '*' | '_' | '|' => true,
            '!' => next == Some('['),
            '&' => next.is_some_and(|n| n.is_ascii_alphanumeric() || n == '#'),
            _ => false,
        };
        if escape {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// `s` as an inline code span whose fence can't be closed from inside.
fn code_span(s: &str) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let longest = longest_run(&flat, '`');
    let fence = "`".repeat(longest + 1);
    format!("{fence} {flat} {fence}")
}

fn longest_run(s: &str, c: char) -> usize {
    let mut best = 0;
    let mut run = 0;
    for ch in s.chars() {
        if ch == c {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    best
}

/// `s` as a fenced block shown exactly as written (a fence longer than any
/// run of tildes inside, so the text can't end it early).
fn fenced(s: &str) -> String {
    let fence = "~".repeat((longest_run(s, '~') + 1).max(4));
    format!("{fence}text\n{}\n{fence}", s.trim_end())
}

/// The notes file's Markdown.
pub(crate) fn render(report: &Report, answer: &str) -> String {
    let mut md = String::new();
    md.push_str("# Web research notes\n\n");
    md.push_str(&format!("**Question:** {}\n\n", escape_inline(&report.question)));
    md.push_str(&format!("- **When:** {}\n", escape_inline(&report.when)));
    md.push_str(&format!("- **Searched with:** {}\n", escape_inline(&report.service)));
    md.push_str(&format!("- **Depth:** {}\n", report.depth.label()));
    if !report.queries.is_empty() {
        let qs: Vec<String> = report.queries.iter().map(|q| code_span(q)).collect();
        md.push_str(&format!("- **Searches:** {}\n", qs.join(" · ")));
    }
    if let Some(model) = &report.model {
        md.push_str(&format!("- **Answered by:** {}\n", escape_inline(model)));
    }
    match &report.outcome {
        Outcome::Searched => {}
        Outcome::NotNeeded => md.push_str("- **Result:** no web search was needed\n"),
        Outcome::Failed(why) => md.push_str(&format!("- **Result:** web research didn't work — {}\n", escape_inline(why))),
        Outcome::Stopped => md.push_str("- **Result:** stopped before it finished\n"),
    }
    md.push_str("\n## Answer\n\n");
    let shown = display_answer(answer, &report.sources);
    md.push_str(if shown.trim().is_empty() { "*(no answer)*" } else { shown.trim() });
    md.push('\n');
    if let Some(why) = &report.cut_short {
        md.push_str(&format!("\n*(This answer is incomplete: it {}.)*\n", escape_inline(why)));
    }
    if !report.sources.is_empty() {
        md.push_str("\n## Sources\n\nThe exact text the AI was given from each page.\n");
        for s in &report.sources {
            md.push_str(&format!("\n### [{}] {}\n\n", s.n, escape_inline(&s.title)));
            match check_url(&s.url) {
                Ok(u) => md.push_str(&format!("{} · <{u}>\n\n", escape_inline(&s.site))),
                Err(_) => md.push_str(&format!("{}\n\n", code_span(&s.url))),
            }
            md.push_str(&fenced(&s.excerpts.join("\n\n…\n\n")));
            md.push('\n');
        }
    }
    if !report.skipped.is_empty() {
        md.push_str("\n## Pages that couldn't be used\n\n");
        for s in &report.skipped {
            md.push_str(&format!("- {} — {}\n", code_span(&s.url), escape_inline(&s.why)));
        }
    }
    md.push_str(
        "\n---\n\n*Saved by Clarity TagFlow's web research. Web pages can be wrong, out of date or \
         written to mislead — check the sources before relying on this.*\n",
    );
    md
}

/// "2026-10-25 14.32 how tall is the eiffel tower.md": the time, then the
/// question's first words (letters and digits only — nothing that could
/// name another folder, a device or a hidden file).
pub(crate) fn file_name(question: &str, stamp: &str) -> String {
    let words: String = question.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
    let mut slug = words.split_whitespace().take(8).collect::<Vec<_>>().join(" ").to_lowercase();
    if let Some((cut, _)) = slug.char_indices().nth(50) {
        slug.truncate(cut);
    }
    let slug = slug.trim();
    let slug = if slug.is_empty() { "research" } else { slug };
    format!("{stamp} {slug}.md")
}

/// Write the notes into `dir` (created if needed) under a fresh name — an
/// existing file (or link) of that name is never written through: a number
/// is added instead.
pub(crate) fn write(dir: &Path, report: &Report, answer: &str, stamp: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let content = render(report, answer);
    let base = file_name(&report.question, stamp);
    let stem = base.trim_end_matches(".md");
    for n in 1..=99 {
        let name = if n == 1 { base.clone() } else { format!("{stem} ({n}).md") };
        let path = dir.join(name);
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut f) => {
                f.write_all(content.as_bytes())?;
                return Ok(path);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "too many notes with this name"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::research::{Depth, Skipped};

    fn sources() -> Vec<Source> {
        vec![
            Source { n: 1, title: "Eiffel Tower".into(), url: "https://en.wikipedia.org/wiki/Eiffel_Tower".into(), site: "en.wikipedia.org".into(), excerpts: vec!["It is 330 m tall.".into()] },
            Source { n: 2, title: "Paris (city)".into(), url: "https://example.com/paris_(city)".into(), site: "example.com".into(), excerpts: vec!["Paris.".into()] },
        ]
    }

    /// Where `md` links to, read with and without math; panics on an image
    /// or raw HTML in it.
    fn link_targets(md: &str) -> Vec<String> {
        let mut out = Vec::new();
        for math in [true, false] {
            for ev in Parser::new_ext(md, options(math)) {
                match ev {
                    Event::Start(Tag::Link { dest_url, .. }) => out.push(dest_url.to_string()),
                    Event::Start(Tag::Image { .. }) | Event::Html(_) | Event::InlineHtml(_) => {
                        panic!("{ev:?} in {md:?}")
                    }
                    _ => {}
                }
            }
        }
        out
    }

    fn only_sources(md: &str) {
        for t in link_targets(md) {
            assert!(
                t == "https://en.wikipedia.org/wiki/Eiffel_Tower" || t == "https://example.com/paris_(city)",
                "{t} in {md:?}"
            );
        }
    }

    #[test]
    fn math_and_nesting_cant_smuggle_a_link() {
        let s = sources();
        for a in [
            // Math to one reading, code to the other.
            "$`$ [x](https://evil.example/a) `",
            "$[x](https://evil.example/b)$",
            "$$\n[x](https://evil.example/c)\n$$",
            // A link whose removal leaves another.
            "[[x](https://evil.example/d)](https://evil.example/e)",
            "[a [1]](https://evil.example/f)",
            "![[x](https://evil.example/g)](https://evil.example/h.png)",
        ] {
            let c = clean_answer(a, &s);
            only_sources(&c);
            only_sources(&display_answer(a, &s));
            assert!(link_targets(&c).is_empty(), "{a:?} → {c:?}");
        }
        // However deep the nesting, it ends — with no link.
        let deep = format!("{}x{}", "[".repeat(40), "](https://evil.example/z)".repeat(40));
        let c = clean_answer(&deep, &s);
        assert!(link_targets(&c).is_empty(), "{c}");
    }

    /// Random Markdown-ish text: links, images, code, math, HTML, entities,
    /// brackets and addresses thrown together.
    fn noise(seed: &mut u64, pieces: usize) -> String {
        const BITS: &[&str] = &[
            "[", "]", "(", ")", "<", ">", "!", "`", "``", "$", "$$", "*", "_", "~", "\\", "&", "&amp;", "&#60;", "#",
            "|", ":", "-", "\n", "\n\n", "    ", "\t", "x", "word ", "[1]", "[2]", "[9]", "https://evil.example/p",
            "https://en.wikipedia.org/wiki/Eiffel_Tower", "www.evil.example", "javascript:alert(1)", "file:///C:/x",
            "<img src=x>", "<a href=\"https://evil.example\">", "</a>", "<script>", "![", "](", "]: ", "\"", "'",
            "é", "日本", "🗼", "\u{200b}", "\u{202e}", "```", "~~~", "> ", "1. ", "- ", "[^1]", "[^1]: n", ": ",
        ];
        let mut out = String::new();
        for _ in 0..pieces {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            out.push_str(BITS[(*seed % BITS.len() as u64) as usize]);
        }
        out
    }

    #[test]
    fn random_answers_never_link_elsewhere() {
        let s = sources();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..3000 {
            let a = noise(&mut seed, 1 + round % 90);
            let c = clean_answer(&a, &s);
            only_sources(&c);
            let d = display_answer(&a, &s);
            only_sources(&d);
            // Settled: cleaning the clean text changes nothing.
            assert_eq!(clean_answer(&c, &s), c, "{a:?}");
        }
    }

    #[test]
    fn written_out_addresses_that_arent_sources_become_code() {
        let s = sources();
        let c = clean_answer(
            "See https://evil.example/phish, or www.evil.example. The page \
             https://en.wikipedia.org/wiki/Eiffel_Tower is fine.\n\n```\nhttps://in.code.example\n```",
            &s,
        );
        assert!(c.contains("` https://evil.example/phish `,"), "{c}");
        assert!(c.contains("` www.evil.example `."), "{c}");
        assert!(c.contains("page https://en.wikipedia.org/wiki/Eiffel_Tower is"), "{c}");
        assert!(c.contains("\nhttps://in.code.example\n"), "{c}");
        // Settled: cleaning again changes nothing.
        assert_eq!(clean_answer(&c, &s), c);
    }

    #[test]
    fn answers_lose_images_html_and_foreign_links() {
        let s = sources();
        let a = "See ![logo](https://evil.example/pixel.png) and <img src=\"https://evil.example/x\"> \
                 [click](https://evil.example/phish) [ok](https://en.wikipedia.org/wiki/Eiffel_Tower) \
                 [run](file:///C:/x.exe) <https://evil.example/auto> [js](javascript:alert(1)).";
        let c = clean_answer(a, &s);
        assert!(!c.contains("evil.example/pixel"), "{c}");
        assert!(!c.contains("<img"), "{c}");
        assert!(c.contains("&lt;img"), "{c}");
        assert!(!c.contains("](https://evil"), "{c}");
        assert!(!c.contains("file:"), "{c}");
        assert!(!c.contains("javascript:"), "{c}");
        assert!(c.contains("[ok](https://en.wikipedia.org/wiki/Eiffel_Tower)"), "{c}");
        assert!(c.contains("click"), "{c}");
        assert!(c.contains("`` https://evil.example/auto ``") || c.contains("` https://evil.example/auto `"), "{c}");
        // Rendered, nothing links anywhere but the source.
        for t in link_targets(&c) {
            assert_eq!(t, "https://en.wikipedia.org/wiki/Eiffel_Tower");
        }
        // Code keeps its characters.
        let code = "Use `<b>` and:\n\n```html\n<img src=x>\n```\n";
        assert_eq!(clean_answer(code, &s), code);
        // Reference-style links to elsewhere lose their link too.
        let r = clean_answer("[here][x]\n\n[x]: https://evil.example/\n", &s);
        assert!(link_targets(&r).is_empty(), "{r}");
    }

    #[test]
    fn citations_link_to_their_sources_only_where_safe() {
        let s = sources();
        let d = display_answer("It's 330 m [1], in Paris [2][1]. Not [3], not `[1]`, not \\[2\\].", &s);
        assert!(d.contains("[[1]](<https://en.wikipedia.org/wiki/Eiffel_Tower>)"), "{d}");
        assert!(d.contains("[[2]](<https://example.com/paris_(city)>)"), "{d}");
        assert!(d.contains("[3]") && d.contains("`[1]`"), "{d}");
        assert_eq!(link_targets(&d).len(), 6, "{d}"); // 3 links, read two ways
        only_sources(&d);
        // Idempotent: shown twice, linked once.
        assert_eq!(display_answer(&d, &s).matches("](<").count(), 3);
    }

    #[test]
    fn inline_escaping_defuses_markup() {
        let e = escape_inline("![x](http://e/p.png) <img src=y> [a](b) `c` &lt; \\ *b*\nnext");
        let rendered: Vec<Event> = Parser::new_ext(&e, options(true)).chain(Parser::new_ext(&e, options(false))).collect();
        assert!(!rendered.iter().any(|ev| matches!(
            ev,
            Event::Start(Tag::Image { .. } | Tag::Link { .. } | Tag::Emphasis) | Event::Html(_) | Event::InlineHtml(_) | Event::Code(_)
        )), "{e}");
        assert!(!e.contains('\n'));
    }

    #[test]
    fn notes_keep_web_text_inert() {
        let mut s = sources();
        s[0].title = "Tower <img src=https://evil.example/t> ![p](https://evil.example/q)".into();
        s[0].excerpts = vec!["~~~~\n# Fake heading\n<script>alert(1)</script>\n![x](https://evil.example/z)\n~~~~~~".into()];
        let report = Report {
            question: "How tall is the [Eiffel](https://evil.example) tower?".into(),
            when: "25 October 2026, 14:32".into(),
            service: "Wikipedia (en)".into(),
            depth: Depth::Quick,
            queries: vec!["eiffel tower `height`".into()],
            sources: s,
            skipped: vec![Skipped { url: "http://192.168.1.1/x".into(), why: "points at this PC or the local network".into() }],
            outcome: Outcome::Searched,
            model: Some("Gemma 4 31B".into()),
            cut_short: None,
            display: None,
            notes: None,
        };
        let md = render(&report, "It's 330 m [1]. ![t](https://evil.example/a)");
        only_sources(&md);
        // The excerpt's fake heading stays inside the fence.
        assert!(!Parser::new_ext(&md, options(true)).any(|e| matches!(e, Event::Text(t) if &*t == "Fake heading")), "{md}");
        assert!(md.contains("## Sources") && md.contains("## Pages that couldn't be used"));
    }

    #[test]
    fn file_names_are_tame() {
        assert_eq!(file_name("How tall is the Eiffel Tower?", "2026-10-25 14.32"), "2026-10-25 14.32 how tall is the eiffel tower.md");
        for q in ["../../etc/passwd", "..\\..\\Windows\\System32", "CON", "a:b*c?d\"e<f>g|h", "   ", "\u{202e}gnp.exe", "....", "x\0y"] {
            let n = file_name(q, "2026-10-25 14.32");
            assert!(n.starts_with("2026-10-25 14.32 ") && n.ends_with(".md"), "{n}");
            let slug = &n["2026-10-25 14.32 ".len()..n.len() - 3];
            assert!(slug.chars().all(|c| c.is_alphanumeric() || c == ' '), "{n}");
            assert!(!slug.is_empty());
        }
        let long = file_name(&"word ".repeat(100), "t");
        assert!(long.chars().count() < 70);
    }

    #[test]
    fn notes_never_overwrite() {
        let dir = std::env::temp_dir().join(format!("ctf_notes_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let report = Report { question: "Same question".into(), ..Report::default() };
        let a = write(&dir, &report, "one", "2026-01-01 10.00").unwrap();
        let b = write(&dir, &report, "two", "2026-01-01 10.00").unwrap();
        assert_ne!(a, b);
        assert!(b.file_name().unwrap().to_string_lossy().ends_with(" (2).md"));
        assert!(std::fs::read_to_string(&a).unwrap().contains("one"));
        assert!(std::fs::read_to_string(&b).unwrap().contains("two"));
        #[cfg(unix)]
        {
            // A link planted where the next note would go isn't written through.
            let target = dir.join("target.txt");
            std::fs::write(&target, "keep").unwrap();
            std::os::unix::fs::symlink(&target, dir.join("2026-01-01 10.00 same question (3).md")).unwrap();
            let c = write(&dir, &report, "three", "2026-01-01 10.00").unwrap();
            assert!(c.file_name().unwrap().to_string_lossy().ends_with(" (4).md"));
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
