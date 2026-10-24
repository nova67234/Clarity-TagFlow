//! Tag images with the AI chat (the Tools menu's "Tag images" mode).
//!
//! The chat becomes a tagging workstation: the image on the left with
//! previous/next, the conversation in the middle, the Tag Manager on the
//! right (src/main.rs lays the three out; with no image selected only the
//! chat shows). Every message the user sends is about the image on show at
//! that moment: the image goes along as the vision input, with its file name
//! and its current tags, read fresh from its .txt so edits made by hand count.
//!
//! The AI changes tags by ending its reply with control lines —
//! `TAGS_ADD:`, `TAGS_REMOVE:`, `TAGS_SET:` — which never show in the chat
//! (like role play's `MEMORY:` lines). When the reply finishes the app applies
//! them to that image's .txt straight away, the Tag Manager tints what the AI
//! added violet (like the AI tagger's tags), and a note under the reply lists
//! the change with an Undo button.

use std::path::{Path, PathBuf};

use eframe::egui;

use crate::theme::*;

pub const ADD_MARKER: &str = "TAGS_ADD:";
pub const REMOVE_MARKER: &str = "TAGS_REMOVE:";
pub const SET_MARKER: &str = "TAGS_SET:";

/// The priming turn pinned in front of the chat history (like the Krea 2
/// brief and role play's preamble).
pub const BRIEF: &str = "[Image tagging]\n\
You are my image-tagging assistant. With each of my messages you get the \
image I'm working on, its file name and its current tags: the comma-separated \
list in the image's .txt file, used to train image models. You may also get \
zoomed-in sections of the same image; use them for small details only.\n\
\n\
- Answer my questions about the image and its tags normally.\n\
- When I ask you to tag the image, or to add, remove, fix, clean up, reorder \
or rewrite tags, make the change yourself: end your reply with control lines, \
each on its own line and starting exactly with its marker:\n\
TAGS_ADD: tag one, tag two\n\
TAGS_REMOVE: tag three, tag four\n\
TAGS_SET: every tag, in the new order (only to rewrite or reorder the whole list)\n\
- Above the control lines, say in a sentence or two what you changed. The \
control lines are hidden from me: the app applies them to the .txt file at \
once, and I can undo.\n\
- Only change tags when I ask, or after I agree to changes you suggested.\n\
- Match the existing tags' style: the same separator inside tags (spaces or \
underscores), the same letter case, and the same kind of vocabulary \
(booru-style tags or short phrases). With no tags yet, use lowercase \
booru-style tags with spaces.\n\
- Tag what is clearly visible: the subject, hair, eyes, clothing, \
accessories, expression, pose, framing, background, lighting and art style. \
Never guess who a real person is, and don't invent details you can't see.";

/// The model's scripted acknowledgement of [`BRIEF`].
pub const ACK: &str = "Understood. I'll change tags only with TAGS_ADD, TAGS_REMOVE or TAGS_SET \
lines at the end of my reply.";

/// The tag edits the AI asked for at the end of one reply.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct TagEdits {
    pub add: Vec<String>,
    pub remove: Vec<String>,
    /// The complete new list (replaces everything; removes and adds then
    /// apply on top).
    pub set: Option<Vec<String>>,
}

impl TagEdits {
    pub fn is_empty(&self) -> bool {
        self.add.is_empty() && self.remove.is_empty() && self.set.is_none()
    }
}

/// A tag change the app made for one reply, kept on that message for its
/// note and Undo button.
#[derive(Clone, Debug, Default)]
pub struct TagChange {
    pub image: PathBuf,
    pub before: Vec<String>,
    pub after: Vec<String>,
    /// The .txt's text before (`None`: there was none) and after.
    pub before_text: Option<String>,
    pub after_text: String,
    /// The .txt's exact bytes before, so Undo puts back the very same file
    /// (encoding, BOM and line endings included) when nothing changed since.
    pub before_bytes: Option<Vec<u8>>,
    pub undone: bool,
    /// Why nothing was written, if so.
    pub error: Option<String>,
    /// Why the last Undo didn't work; it can be tried again.
    pub undo_error: Option<String>,
}

impl TagChange {
    /// A change that wasn't made, and why.
    pub fn failed(image: PathBuf, why: &str) -> Self {
        TagChange { image, error: Some(why.to_string()), ..Default::default() }
    }

    /// The edit changed the file — a new order or spelling counts, even
    /// with no tag added or removed.
    pub fn changed(&self) -> bool {
        self.error.is_none() && self.before_text.as_deref().unwrap_or("") != self.after_text
    }

    /// Tags the change added and removed.
    pub fn diff(&self) -> (Vec<&String>, Vec<&String>) {
        let added = self.after.iter().filter(|t| !self.before.iter().any(|b| same_tag(b, t))).collect();
        let removed = self.before.iter().filter(|t| !self.after.iter().any(|a| same_tag(a, t))).collect();
        (added, removed)
    }

    /// The same tags in the same order, only spelled differently.
    pub fn respelled(&self) -> bool {
        self.before.len() == self.after.len() && self.before.iter().zip(&self.after).all(|(a, b)| same_tag(a, b))
    }
}

/// What to send again once the tag changes of the replies it replaces are
/// taken back (see [`TagRequest::Rewind`]).
pub enum Resend {
    /// Regenerate the reply at this index.
    Retry(usize),
    /// Resend the user message at this index with new words.
    Edit(usize, String),
}

/// Work for main.rs, which owns the image list, the Tag Manager and the
/// right panel — the chat only asks.
pub enum TagRequest {
    /// Apply a finished reply's edits to `image` (the image the user's
    /// message was about); `index` is the reply's position in the chat.
    Apply { chat_id: u64, index: usize, image: PathBuf, edits: TagEdits },
    /// The Undo button under that reply.
    Undo { chat_id: u64, index: usize },
    /// Regenerate or edit-and-resend is about to replace the replies from
    /// `from` on: undo their tag changes (newest first), then resend.
    Rewind { chat_id: u64, from: usize, then: Resend },
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Add,
    Remove,
    Set,
}

/// A control line's kind and list, tolerating stray Markdown around it
/// (`**TAGS_ADD:** …`, backticks, a bullet).
fn control_line(line: &str) -> Option<(Kind, &str)> {
    let t = line.trim().trim_start_matches(['*', '`', '-', '>', ' ']);
    for (marker, kind) in [(ADD_MARKER, Kind::Add), (REMOVE_MARKER, Kind::Remove), (SET_MARKER, Kind::Set)] {
        if let Some(rest) = t.strip_prefix(marker) {
            return Some((kind, rest.trim().trim_matches(['*', '`']).trim()));
        }
    }
    None
}

/// A line still on its way to becoming a control line while a reply streams
/// ("TAGS_A…").
fn partial_marker(line: &str) -> bool {
    let t = line.trim().trim_start_matches(['*', '`', '-', '>', ' ']);
    !t.is_empty() && [ADD_MARKER, REMOVE_MARKER, SET_MARKER].iter().any(|m| m.starts_with(t))
}

/// A bullet or number in front of a list item ("- red eyes", "2. smile"):
/// the item, and whether there was one.
fn strip_bullet(t: &str) -> (&str, bool) {
    if let Some(rest) = t.strip_prefix(['-', '*', '•']).filter(|r| r.starts_with(' ')) {
        return (rest.trim(), true);
    }
    let digits = t.chars().take_while(char::is_ascii_digit).count();
    if (1..=3).contains(&digits)
        && let Some(rest) = t[digits..].strip_prefix(['.', ')'])
        && rest.starts_with(' ')
    {
        return (rest.trim(), true);
    }
    (t, false)
}

/// A line that carries on a control line's list: the list itself under an
/// empty marker ("TAGS_SET:" then "1girl, solo", or one tag per bullet), or
/// — `wrapped` — the rest of a list that ended in a comma. Never prose: no
/// sentence endings, every item a few words at most, and a lone item only as
/// a bullet or a wrapped list's tail. (Missing a list just leaves it showing;
/// taking a sentence for tags would rewrite the image's tags.)
fn list_line(line: &str, wrapped: bool) -> Option<&str> {
    let raw = line.trim();
    if raw.is_empty() || raw.starts_with("```") || control_line(line).is_some() || partial_marker(line) {
        return None;
    }
    let (t, bullet) = strip_bullet(raw);
    // A full stop may end a bullet, or a wrapped list's last line.
    let t = if bullet || wrapped { t.trim_end_matches('.').trim_end() } else { t };
    if t.is_empty() || t.ends_with(['.', '!', '?', ':', ';']) {
        return None;
    }
    let items: Vec<&str> = t.split(',').map(str::trim).filter(|i| !i.is_empty()).collect();
    let words = if wrapped { 8 } else { 5 };
    let short = !items.is_empty() && items.iter().all(|i| i.split_whitespace().count() <= words);
    (short && (items.len() > 1 || bullet || wrapped)).then_some(t)
}

/// "(none)", "n/a", "(unchanged)"… — what a model writes when it has
/// nothing to list. A whole item in brackets is a remark, not a tag.
fn placeholder(t: &str) -> bool {
    if t.len() > 2 && t.starts_with('(') && t.ends_with(')') {
        return true;
    }
    let t = t.trim().to_lowercase();
    matches!(t.as_str(), "none" | "nothing" | "n/a" | "unchanged" | "no change" | "no changes" | "-" | "—" | "–")
}

/// The tags of a control line: comma-separated, with brackets around the
/// whole list (`["red eyes", "smile"]`), quotes, stray full stops and
/// placeholders like "(none)" trimmed, and duplicates dropped.
fn split_list(s: &str) -> Vec<String> {
    let mut s = s.trim();
    if s.len() >= 2 && s.starts_with('[') && s.ends_with(']') {
        s = &s[1..s.len() - 1];
    }
    let mut out: Vec<String> = Vec::new();
    for raw in s.split(',') {
        let mut t = raw.trim().trim_matches(['"', '\'', '`']).trim();
        // "smile." — a sentence's full stop, not part of the tag (but leave
        // emoticons like ":." alone).
        if t.ends_with('.') && t.chars().any(char::is_alphanumeric) {
            t = t.trim_end_matches('.').trim_end();
        }
        if !t.is_empty() && !placeholder(t) && !out.iter().any(|e| same_tag(e, t)) {
            out.push(t.to_string());
        }
    }
    out
}

/// Walk a reply: the lines that stay visible, and the edits its control
/// lines ask for (with their lists, even when those run onto the next lines).
fn parse(reply: &str) -> (Vec<&str>, TagEdits) {
    let lines: Vec<&str> = reply.lines().collect();
    let mut kept = Vec::new();
    let mut edits = TagEdits::default();
    let mut i = 0;
    while i < lines.len() {
        let Some((kind, rest)) = control_line(lines[i]) else {
            kept.push(lines[i]);
            i += 1;
            continue;
        };
        i += 1;
        // The list under an empty marker, or the rest of one wrapped after a
        // trailing comma, from the lines right below.
        let below = rest.is_empty();
        let mut list = rest.to_string();
        while i < lines.len() {
            let wrapped = list.trim_end().ends_with(',');
            if !below && !wrapped {
                break;
            }
            let Some(more) = list_line(lines[i], wrapped) else { break };
            if !list.is_empty() {
                list.push_str(", ");
            }
            list.push_str(more);
            i += 1;
        }
        // A list still ending in a comma lost its tail somewhere.
        let cut = list.trim_end().ends_with(',');
        let tags = split_list(&list);
        match kind {
            Kind::Add => edits.add.extend(tags),
            Kind::Remove => edits.remove.extend(tags),
            // An empty or cut-short TAGS_SET is a slip: as the whole new list
            // it would drop every tag it misses.
            Kind::Set if !tags.is_empty() && !cut => edits.set = Some(tags),
            Kind::Set => {}
        }
    }
    (kept, edits)
}

/// Pull the control lines out of a finished reply, returning the edits and
/// leaving only the visible text.
pub fn extract(reply: &mut String) -> TagEdits {
    if !reply.contains("TAGS_") {
        return TagEdits::default();
    }
    let (visible, edits) = {
        let (mut kept, edits) = parse(reply);
        drop_empty_fences(&mut kept, false);
        (kept.join("\n").trim_end().to_string(), edits)
    };
    *reply = visible;
    edits
}

/// Drop the code blocks the control lines leave empty (when the model fenced
/// them), so no empty box shows in the chat. `open_end`: also drop a block
/// still open at the end with nothing in it yet (mid-stream).
fn drop_empty_fences(lines: &mut Vec<&str>, open_end: bool) {
    let is_fence = |l: &str| l.trim_start().starts_with("```");
    let mut i = 0;
    while i < lines.len() {
        if !is_fence(lines[i]) {
            i += 1;
            continue;
        }
        match (i + 1..lines.len()).find(|&k| is_fence(lines[k])) {
            Some(k) if lines[i + 1..k].iter().all(|l| l.trim().is_empty()) => {
                lines.drain(i..=k);
            }
            Some(k) => i = k + 1,
            None => {
                if open_end && lines[i + 1..].iter().all(|l| l.trim().is_empty()) {
                    lines.truncate(i);
                }
                break;
            }
        }
    }
}

/// Hide control lines while a reply is still streaming (they're removed for
/// real when it finishes) — including a last line that's still on its way to
/// becoming one ("TAGS_A…").
pub fn strip_lines(text: &str) -> String {
    let last = text.rsplit('\n').next().unwrap_or("");
    let last_partial = !text.ends_with('\n') && partial_marker(last) && control_line(last).is_none();
    if !text.contains("TAGS_") && !last_partial {
        return text.to_string();
    }
    let (mut lines, _) = parse(text);
    if last_partial && lines.last() == Some(&last) {
        lines.pop();
    }
    drop_empty_fences(&mut lines, true);
    lines.join("\n")
}

/// Two tags are the same when they only differ in case, or in spaces versus
/// underscores ("Long_Hair" = "long hair").
pub fn same_tag(a: &str, b: &str) -> bool {
    fn key(s: &str) -> String {
        s.to_lowercase().replace('_', " ").split_whitespace().collect::<Vec<_>>().join(" ")
    }
    key(a) == key(b)
}

/// Write `tag` the way `existing` writes theirs: underscores or spaces
/// between words, and all lowercase when they all are.
fn styled_like(existing: &[String], tag: &str) -> String {
    let words = |t: &&String| t.trim().contains([' ', '_']);
    let underscores = existing.iter().filter(words).filter(|t| t.contains('_') && !t.contains(' ')).count();
    let spaces = existing.iter().filter(words).filter(|t| t.contains(' ') && !t.contains('_')).count();
    let mut out = tag.trim().to_string();
    if underscores > spaces {
        out = out.split_whitespace().collect::<Vec<_>>().join("_");
    } else if spaces > underscores && out.chars().all(|c| c.is_alphanumeric() || c == '_') {
        // Only plain word_word tags: "^_^" keeps its underscore.
        out = out.replace('_', " ");
    }
    if !existing.is_empty() && existing.iter().all(|t| !t.chars().any(char::is_uppercase)) {
        out = out.to_lowercase();
    }
    out
}

/// The tag list after `edits`: the full new list if one was given (in the
/// existing style, so it can also fix spellings), then the removals, then the
/// additions (appended in order, in the existing style, skipping ones already
/// there). Without a new list, the existing tags stay exactly as they are,
/// duplicates included.
pub fn apply(before: &[String], edits: &TagEdits) -> Vec<String> {
    let mut tags: Vec<String> = match &edits.set {
        Some(list) => {
            let mut out: Vec<String> = Vec::new();
            for t in list {
                let t = styled_like(before, t);
                if !t.is_empty() && !out.iter().any(|e| same_tag(e, &t)) {
                    out.push(t);
                }
            }
            out
        }
        None => before.to_vec(),
    };
    tags.retain(|t| !edits.remove.iter().any(|r| same_tag(t, r)));
    for a in &edits.add {
        let a = styled_like(before, a);
        if !a.is_empty() && !tags.iter().any(|t| same_tag(t, &a)) {
            tags.push(a);
        }
    }
    tags
}

/// The tag list after undoing `change`, given the tags the image has now:
/// exactly what it had before when nothing else changed since, otherwise
/// just the AI's own change taken back — its additions removed, its
/// removals put back, the old order restored — so the user's later edits
/// (tags they added, their spelling) survive.
pub fn undo(current: &[String], change: &TagChange) -> Vec<String> {
    let unchanged = current.len() == change.after.len()
        && current.iter().zip(&change.after).all(|(a, b)| same_tag(a, b));
    if unchanged {
        return change.before.clone();
    }
    let (added, removed) = change.diff();
    let kept: Vec<&String> = current.iter().filter(|t| !added.iter().any(|a| same_tag(a, t))).collect();
    // The tags it had before, in their old order — each matched to one tag
    // it has now (the same spelling first, so "long hair" and "long_hair"
    // both survive)…
    let mut used = vec![false; kept.len()];
    let mut out: Vec<String> = Vec::new();
    for b in &change.before {
        let exact = (0..kept.len()).find(|&i| !used[i] && kept[i] == b);
        match exact.or_else(|| (0..kept.len()).find(|&i| !used[i] && same_tag(kept[i], b))) {
            Some(i) => {
                used[i] = true;
                out.push(kept[i].clone());
            }
            None if removed.iter().any(|r| same_tag(r, b)) => out.push(b.clone()),
            None => {} // the user removed it since
        }
    }
    // …then the ones the user added since.
    out.extend(kept.iter().zip(&used).filter(|(_, u)| !**u).map(|(k, _)| (*k).clone()));
    out
}

/// The image's file name, for the model and the chat.
pub fn file_name(image: &Path) -> String {
    image.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// What goes in front of the newest message (hidden from the chat): which
/// image it's about and its tags right now. `None` = the .txt exists but
/// can't be read.
pub fn context(image: &Path, tags: Option<&[String]>) -> String {
    let tags = match tags {
        Some([]) => "(none yet)".to_string(),
        Some(t) => t.join(", "),
        None => "(its .txt file can't be read, so its tags can't be changed)".to_string(),
    };
    format!("[Image: {}]\nCurrent tags: {tags}", file_name(image))
}

/// What goes in front of an older message: just which image it was about
/// (its tags then may be out of date).
pub fn label(image: &Path) -> String {
    format!("[Image: {}]", file_name(image))
}

/// What the image panel shows; main.rs resolves the texture from its image
/// caches.
pub struct PanelImage<'a> {
    pub path: &'a Path,
    /// 1-based position in the browser's (filtered) list, if it's in it.
    pub position: Option<usize>,
    pub total: usize,
    pub tex: Option<&'a egui::TextureHandle>,
    /// Still decoding (a thumbnail or spinner shows meanwhile).
    pub loading: bool,
    /// The image couldn't be decoded.
    pub failed: bool,
}

/// A round previous/next button with a chevron.
fn nav_button(ui: &mut egui::Ui, next: bool, enabled: bool, tip: &str) -> bool {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::click());
    let resp = resp.on_hover_text(tip);
    let hovered = enabled && resp.hovered();
    let painter = ui.painter();
    if hovered {
        painter.circle_filled(rect.center(), 14.0, TEXT().gamma_multiply(0.10));
    }
    let col = if !enabled {
        MUTED().gamma_multiply(0.45)
    } else if hovered {
        TEXT()
    } else {
        MUTED()
    };
    let c = rect.center();
    let s = if next { 1.0 } else { -1.0 };
    let stroke = egui::Stroke::new(2.0_f32, col);
    painter.line_segment([c + egui::vec2(-3.0 * s, -6.0), c + egui::vec2(3.0 * s, 0.0)], stroke);
    painter.line_segment([c + egui::vec2(3.0 * s, 0.0), c + egui::vec2(-3.0 * s, 6.0)], stroke);
    if enabled && resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    enabled && resp.clicked()
}

/// The tagging view's left panel: previous/next with the position, the file
/// name, and the image fitted below. Returns -1 / +1 when the user asks for
/// the previous / next image.
pub fn image_panel(ui: &mut egui::Ui, img: PanelImage<'_>, width: f32) -> i32 {
    let mut step = 0;
    egui::Panel::left("ai_tag_image")
        .resizable(false)
        .exact_size(width)
        .show_separator_line(false)
        .frame(egui::Frame::new().fill(BG()).inner_margin(egui::Margin { left: 10, right: 10, top: 0, bottom: 10 }))
        .show_inside(ui, |ui| {
            crate::card_frame(22).show(ui, |ui| {
                ui.set_min_height(ui.available_height());
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    let first = img.position == Some(1);
                    let last = img.position == Some(img.total);
                    if nav_button(ui, false, img.position.is_some() && !first, "Previous image (← when you're not typing)") {
                        step = -1;
                    }
                    let pos = match img.position {
                        Some(p) => format!("{p} of {}", img.total),
                        None => format!("— of {}", img.total),
                    };
                    ui.label(egui::RichText::new(pos).color(TEXT()).size(13.0));
                    if nav_button(ui, true, img.position.is_some() && !last, "Next image (→ when you're not typing)") {
                        step = 1;
                    }
                    ui.add_space(6.0);
                    ui.add(
                        egui::Label::new(egui::RichText::new(file_name(img.path)).color(MUTED()).size(12.0))
                            .truncate(),
                    )
                    .on_hover_text(img.path.display().to_string());
                });
                ui.add_space(8.0);
                match img.tex {
                    Some(tex) => crate::show_fitted(ui, tex, img.loading),
                    None if img.failed => {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("Couldn't load this image").color(MUTED()));
                        });
                    }
                    None => {
                        ui.centered_and_justified(|ui| {
                            ui.add(egui::Spinner::new().size(36.0).color(MUTED()));
                        });
                    }
                }
            });
        });
    step
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(tags: &[&str]) -> Vec<String> {
        tags.iter().map(|t| t.to_string()).collect()
    }

    /// Control lines come out (with stray Markdown), the visible text stays.
    #[test]
    fn edits_extract_and_strip() {
        let mut reply = "Added the eye colour and dropped the wrong hat.\n\
                         **TAGS_ADD:** red eyes, smile.\n\
                         TAGS_REMOVE: `hat`\n"
            .to_string();
        let e = extract(&mut reply);
        assert_eq!(e.add, v(&["red eyes", "smile"]));
        assert_eq!(e.remove, v(&["hat"]));
        assert_eq!(e.set, None);
        assert_eq!(reply, "Added the eye colour and dropped the wrong hat.");
        // While streaming, a line on its way to becoming a marker is hidden.
        assert_eq!(strip_lines("Done.\nTAGS_ADD: a\nTAGS_RE"), "Done.");
        assert_eq!(strip_lines("Two tags."), "Two tags.");
    }

    /// Control lines the model put in a code block leave no empty block
    /// behind; a real code block stays.
    #[test]
    fn fenced_control_lines_leave_no_empty_block() {
        let mut reply = "Done.\n```\nTAGS_ADD: sun\nTAGS_REMOVE: cloud\n```\n".to_string();
        let e = extract(&mut reply);
        assert_eq!(e.add, v(&["sun"]));
        assert_eq!(e.remove, v(&["cloud"]));
        assert_eq!(reply, "Done.");
        let mut reply = "Like this:\n```text\nlong hair\n```\nTAGS_ADD: long hair".to_string();
        assert_eq!(extract(&mut reply).add, v(&["long hair"]));
        assert_eq!(reply, "Like this:\n```text\nlong hair\n```");
        // Mid-stream, before the block is closed (or the line complete).
        assert_eq!(strip_lines("Done.\n```\nTAGS_ADD: sun"), "Done.");
        assert_eq!(strip_lines("Done.\n```text\nTAGS_A"), "Done.");
        assert_eq!(strip_lines("Done.\n```text\nTAGS_ADD: sun\n```"), "Done.");
    }

    /// Lists that run onto the next lines are read whole; an empty TAGS_SET
    /// never wipes the tags; brackets and placeholders don't become tags.
    #[test]
    fn lists_over_several_lines() {
        let mut reply = "New order.\n**TAGS_SET:**\n1girl, solo\nlong hair, smile".to_string();
        assert_eq!(extract(&mut reply).set, Some(v(&["1girl", "solo", "long hair", "smile"])));
        assert_eq!(reply, "New order.");
        let mut reply = "Reordered.\nTAGS_SET: solo, 1girl, long hair,\nblue eyes, smile.".to_string();
        assert_eq!(extract(&mut reply).set, Some(v(&["solo", "1girl", "long hair", "blue eyes", "smile"])));
        assert_eq!(reply, "Reordered.");
        // A TAGS_SET whose wrapped tail went missing isn't trusted.
        let mut reply = "TAGS_SET: solo, 1girl,\nWhat do you think?".to_string();
        assert!(extract(&mut reply).is_empty());
        assert_eq!(reply, "What do you think?");
        let mut reply = "Added two.\nTAGS_ADD:\n- red eyes\n- smile\nAnything else?".to_string();
        assert_eq!(extract(&mut reply).add, v(&["red eyes", "smile"]));
        assert_eq!(reply, "Added two.\nAnything else?");
        let mut reply = "TAGS_ADD:\n1. red eyes\n2) smile".to_string();
        assert_eq!(extract(&mut reply).add, v(&["red eyes", "smile"]));
        let mut reply = "TAGS_ADD:\n[\"red eyes\", \"smile\"]".to_string();
        assert_eq!(extract(&mut reply).add, v(&["red eyes", "smile"]));
        // Prose after an empty marker stays prose — the unused markers a
        // model leaves empty never turn the next sentence into tags.
        let mut reply = "I added the eye colour.\nTAGS_ADD: red eyes\nTAGS_REMOVE:\nTAGS_SET:\n\n\
                         If you want, I can tag the background too."
            .to_string();
        let e = extract(&mut reply);
        assert_eq!((e.add, e.remove, e.set), (v(&["red eyes"]), vec![], None));
        assert_eq!(reply, "I added the eye colour.\n\nIf you want, I can tag the background too.");
        for prose in [
            "That is the whole list in a new order.",
            "No changes needed",
            "Hope this helps",
            "**Note**",
            "Sure, done.",
            "I added red eyes and a smile, since both are clearly visible",
        ] {
            let mut reply = format!("TAGS_SET:\n{prose}");
            assert!(extract(&mut reply).is_empty(), "{prose}");
            assert_eq!(reply, prose);
        }
        let mut reply = "TAGS_ADD:\nred eyes, smile\nI added both, since they're clearly visible.".to_string();
        assert_eq!(extract(&mut reply).add, v(&["red eyes", "smile"]));
        assert_eq!(reply, "I added both, since they're clearly visible.");
        let mut reply = "TAGS_ADD: [\"red eyes\", \"smile\"]\nTAGS_REMOVE: (none)".to_string();
        let e = extract(&mut reply);
        assert_eq!(e.add, v(&["red eyes", "smile"]));
        assert!(e.remove.is_empty());
        for nothing in ["TAGS_SET: (unchanged)", "TAGS_SET: none", "TAGS_SET: no changes", "TAGS_ADD: n/a"] {
            let mut reply = nothing.to_string();
            assert!(extract(&mut reply).is_empty(), "{nothing}");
        }
        // A tag with a bracketed part is still a tag.
        let mut reply = "TAGS_ADD: hatsune miku (cosplay)".to_string();
        assert_eq!(extract(&mut reply).add, v(&["hatsune miku (cosplay)"]));
        // Mid-stream: the line above an unfinished marker stays.
        assert_eq!(strip_lines("Added red eyes.\nTAGS_ADD:"), "Added red eyes.");
        assert_eq!(strip_lines("Added red eyes.\nTAGS_ADD:\nred eyes, sm"), "Added red eyes.");
    }

    /// A new order or spelling counts as a change (with Undo), though no tag
    /// was added or removed; duplicates the user had are left alone.
    #[test]
    fn reorders_are_changes() {
        let before = v(&["1girl", "Solo", "long hair", "smile"]);
        let e = TagEdits { set: Some(v(&["Solo", "1girl", "long_hair", "smile"])), ..Default::default() };
        let after = apply(&before, &e);
        assert_eq!(after, v(&["Solo", "1girl", "long hair", "smile"]));
        // TAGS_SET can fix a spelling, too.
        let caps = v(&["Blue Eyes", "smile"]);
        let fixed = apply(&caps, &TagEdits { set: Some(v(&["blue eyes", "smile"])), ..Default::default() });
        assert_eq!(fixed, v(&["blue eyes", "smile"]));
        let respelled = TagChange { before: caps, after: fixed, ..Default::default() };
        assert!(respelled.respelled());
        let change = TagChange {
            before: before.clone(),
            after,
            before_text: Some("1girl, Solo\nlong hair, smile\n".into()),
            after_text: "Solo, 1girl, long hair, smile".into(),
            ..Default::default()
        };
        assert!(change.changed());
        let (added, removed) = change.diff();
        assert!(added.is_empty() && removed.is_empty());
        assert!(!change.respelled());
        let dupes = v(&["long hair", "smile", "long_hair"]);
        let e = TagEdits { add: v(&["red eyes"]), ..Default::default() };
        assert_eq!(apply(&dupes, &e), v(&["long hair", "smile", "long_hair", "red eyes"]));
        let none = TagChange { before_text: Some("a, b".into()), after_text: "a, b".into(), ..Default::default() };
        assert!(!none.changed());
    }

    /// Edits follow the existing style and never duplicate.
    #[test]
    fn edits_apply_in_the_existing_style() {
        let before = v(&["1girl", "long_hair", "blue_eyes", "hat"]);
        let e = TagEdits { add: v(&["red eyes", "Long Hair"]), remove: v(&["blue eyes", "HAT"]), set: None };
        assert_eq!(apply(&before, &e), v(&["1girl", "long_hair", "red_eyes"]));
        let spaced = v(&["long hair", "smile"]);
        let e = TagEdits { add: v(&["open_mouth", "^_^"]), ..Default::default() };
        assert_eq!(apply(&spaced, &e), v(&["long hair", "smile", "open mouth", "^_^"]));
        let e = TagEdits { set: Some(v(&["smile", "long hair"])), ..Default::default() };
        assert_eq!(apply(&spaced, &e), v(&["smile", "long hair"]));
    }

    /// Undo restores the old list, or — after later hand edits — only takes
    /// back what the AI did.
    #[test]
    fn undo_takes_back_only_the_ai_change() {
        let change = TagChange {
            image: PathBuf::from("a.png"),
            before: v(&["cat", "hat"]),
            after: v(&["cat", "red eyes"]),
            ..Default::default()
        };
        assert_eq!(undo(&v(&["cat", "red eyes"]), &change), v(&["cat", "hat"]));
        assert_eq!(undo(&v(&["cat", "red eyes", "outdoors"]), &change), v(&["cat", "hat", "outdoors"]));
        // A new order, then a tag the user added: the old order comes back
        // and their tag stays; a tag the user removed since stays removed.
        let reorder = TagChange {
            image: PathBuf::from("a.png"),
            before: v(&["1girl", "solo", "smile"]),
            after: v(&["solo", "1girl", "smile"]),
            ..Default::default()
        };
        assert_eq!(undo(&v(&["solo", "1girl", "smile", "outdoors"]), &reorder), v(&["1girl", "solo", "smile", "outdoors"]));
        assert_eq!(undo(&v(&["solo", "1girl"]), &reorder), v(&["1girl", "solo"]));
        // Two spellings of one tag the user had both survive.
        let dupes = v(&["long hair", "smile", "long_hair"]);
        let after = apply(&dupes, &TagEdits { add: v(&["red eyes"]), ..Default::default() });
        let change = TagChange { before: dupes.clone(), after, ..Default::default() };
        let now = v(&["long hair", "smile", "long_hair", "red eyes", "outdoors"]);
        assert_eq!(undo(&now, &change), v(&["long hair", "smile", "long_hair", "outdoors"]));
    }

    /// The context line names the image and its tags.
    #[test]
    fn context_lines() {
        let p = Path::new("/pics/cat 01.png");
        let tags = v(&["cat", "hat"]);
        assert_eq!(context(p, Some(&tags[..])), "[Image: cat 01.png]\nCurrent tags: cat, hat");
        assert!(context(p, Some(&[][..])).ends_with("(none yet)"));
        assert!(context(p, None).contains("can't be read"));
        assert_eq!(label(p), "[Image: cat 01.png]");
    }
}
