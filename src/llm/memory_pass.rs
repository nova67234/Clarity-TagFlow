//! Remembering by itself (AI Settings → Memory → Remember across chats).
//!
//! **Reading.** Every reply in an ordinary chat starts with a memory note
//! (`MemoryStore::chat_brief` in src/memory.rs): the AI's notes about itself,
//! the last chat's summary, the topics it remembers things about and the
//! memories most related to what's being said — pinned before the history
//! the way role play's preamble is, and picked afresh for every reply. The
//! worker re-reads the whole prompt every turn anyway, so a note that
//! changes from reply to reply costs nothing extra.
//!
//! **Writing.** A background pass reads what's new in a chat and writes down
//! memories, a summary and the AI's own notes (`pass_head`, `parse_pass`,
//! `apply_pass`). It runs on a model that's already loaded — it never loads
//! one — once the user has left the chat (picked another one, started a new
//! one), the chat has been quiet for three minutes, or AI Chat was switched
//! off; and only while AI Chat is on (or just switched off), nothing has
//! been sent or answered for 20 seconds and nothing is being typed. Sending
//! anything stops it at once — the worker reads its prompt in small steps,
//! so even a model partly on the CPU lets go within one — and it runs again
//! later. So does anything else that needs the worker: a Link device's
//! question, unloading, another model.
//!
//! **Where.** On this PC's loaded model, or on the Link host the chat runs
//! on — but only messages already sent to that host go there (each message
//! records the hosts it went to, and a part never mixes them): what stayed
//! on this PC stays here. A host reads a device's pass in steps too, never
//! loads its model for one, and stops it for anything else, like its own.
//!
//! **Never lost, never brought back.** What no pass has read when the app
//! closes is kept for the next run, encrypted beside the memories
//! (`PendingChat`), with any message that looks like it holds a secret left
//! out. A pass is thrown away when the user changed the memory while it ran
//! (it worked from the old memory, and would write back what they just
//! removed). Switching memory off, Clear all and Start fresh drop everything
//! waiting. Role play, Krea 2 prompts and Tag images pin briefs of their
//! own, and what's said in them is never read. Krea 2 prompts still recall:
//! its brief comes with the memory note, so prompts can follow the user's
//! taste.
//!
//! **Pictures.** Images sent in a chat (and, in Krea 2 prompts, the images
//! alone, with the prompt written for each to say what it shows) reach a
//! pass numbered — `[image 1]`… — and it may keep a couple in the album
//! (src/album.rs): a copy, encrypted, under a memory in the Pictures topic
//! saying what it shows, with why it was kept beside it. Pictures of the
//! same thing again and again become a memory too, and grow the AI's notes
//! about itself. The pass never sees the pictures, only what was said about
//! them; the copies are made here, on this PC, even for a pass that ran on
//! a Link host.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;

use super::{Chat, ChatRole, Cmd, CmdMsg, GenParams, LlmState, Msg, RemoteFlags, RemotePass, ThinkLevel};
use crate::memory::{self, MemoryStore, PendingChat};

/// A chat is read once it has been quiet this long (or left).
const IDLE: Duration = Duration::from_secs(180);
/// A pass starts only once nothing was sent or answered for this long.
const QUIET: Duration = Duration::from_secs(20);
/// With AI Chat switched off, the passes still due get this long; then the
/// model is freed.
const OFF_GRACE: Duration = Duration::from_secs(90);
/// A Link host keeps its model loaded for ten minutes after the last
/// question: passes go there only while it most likely still has it (it
/// says so when it hasn't, and never loads it for a pass).
const HOST_WARM: Duration = Duration::from_secs(8 * 60);
/// After a pass failed, the next one waits this long; after something else
/// needed the worker (and stopped it), this long.
const RETRY_AFTER_FAILURE: Duration = Duration::from_secs(60);
const RETRY_AFTER_STOP: Duration = Duration::from_secs(30);
/// A part of a chat whose pass failed this many times is given up on.
const MAX_TRIES: u8 = 2;
/// Chat parts kept for the next run, at most (the newest).
const MAX_PENDING: usize = 8;
/// A pass's answer, at most (tokens): room for twelve memories, the summary
/// and the notes. It ends as soon as its JSON is complete anyway.
const PASS_REPLY: usize = 1536;
/// The model's side of the pinned memory note.
const NOTE_ACK: &str = "Understood.";
/// Ends the memory note in a chat that recalls but isn't saved (Krea 2
/// prompts).
const RECALL_ONLY: &str = "\nRight now you're writing image prompts: use what you remember (like the \
                           user's taste in subjects, styles or colours) only where it fits the prompt \
                           asked for. Of what's said here, only the pictures may go into your memory.\n";

/// What the memory is doing, for the chat and AI Settings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryActivity {
    /// Remember across chats is off.
    Off,
    /// The memory can't be read on this computer (it's left as it is).
    Unreadable,
    /// A pass is reading a chat now.
    Updating,
    /// This many chats (or parts of long ones) are waiting for a pass.
    Waiting(usize),
    /// Nothing waiting.
    UpToDate,
}

/// The memory passes' state (a field of `LlmState`).
#[derive(Default)]
pub(crate) struct PassState {
    /// Mirrors the setting: `None` until main.rs first sets it.
    enabled: Option<bool>,
    /// AI Chat is on (main.rs mirrors it): passes start only then, or in the
    /// grace after it's switched off.
    chat_on: bool,
    /// The pass in flight.
    job: Option<Job>,
    /// Chat parts the last run left unread (read once memory is first on).
    pending: Vec<PendingChat>,
    pending_loaded: bool,
    /// The chat (by this session's id) whose pass wrote the summary there is
    /// now: its notes leave the summary out, its next pass extends it.
    summary_chat: Option<u64>,
    /// When something was last sent or answered.
    last_activity: Option<Instant>,
    /// AI Chat was switched off: passes still due may run until then.
    off_until: Option<Instant>,
    /// No pass starts before then: the last one failed, or something else
    /// needed the worker.
    retry_at: Option<Instant>,
    /// When the Link host said its model isn't loaded: no pass goes there
    /// until the chat has been used again.
    host_cold: Option<Instant>,
}

/// What a pass reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    /// A chat (by id), up to this many messages.
    Chat { id: u64, upto: usize },
    /// One of the parts the last run left.
    Pending(usize),
}

/// A pass in flight.
struct Job {
    target: Target,
    /// `memory.revision` when it started.
    revision: u64,
    cancel: Arc<AtomicBool>,
    /// Asked of a Link host: set when it said its model isn't loaded.
    cold: Option<Arc<AtomicBool>>,
    rx: Receiver<Msg>,
    answer: String,
    /// The pictures its lines number, for the album.
    pictures: Vec<std::path::PathBuf>,
}

/// The next part to read.
struct Next {
    target: Target,
    lines: Vec<String>,
    touched: Vec<String>,
    extends_summary: bool,
    /// The pictures its lines number (`[image 1]` is the first).
    pictures: Vec<std::path::PathBuf>,
    /// It goes to the Link host (it has been sent there before); to this
    /// PC's worker otherwise.
    on_host: bool,
}

/// Where a pass can run right now: on this PC's loaded worker, and on the
/// Link host the chat runs on (by its key), if any.
struct Places {
    local: bool,
    host: Option<String>,
}

impl Places {
    /// Where a part that has been sent to `shared` (Link hosts, by key) can
    /// run: on the host when it's one of them (`Some(true)`), else here
    /// (`Some(false)`) — or nowhere now.
    fn for_part(&self, shared: &[String]) -> Option<bool> {
        if self.host.as_ref().is_some_and(|h| shared.contains(h)) {
            return Some(true);
        }
        self.local.then_some(false)
    }
}

/// A Link host's identity: its key, in hex.
pub(super) fn host_id(peer: &crate::link::Peer) -> String {
    peer.key.iter().map(|b| format!("{b:02x}")).collect()
}

/// Print what the memory does to stderr (`CTF_MEMORY_DEBUG=1`), for tuning
/// the prompts on real models.
fn debug() -> bool {
    std::env::var_os("CTF_MEMORY_DEBUG").is_some()
}

/// Lines in one pass, at most (a Link host takes 400).
const PASS_MAX_LINES: usize = 300;

/// The part of a chat a pass reads.
#[derive(Debug, PartialEq)]
struct Part {
    /// The messages, as a pass reads them (see `memory::chat_line`).
    lines: Vec<String>,
    /// The message count it covers.
    upto: usize,
    /// The Link hosts (by key) every one of them went to.
    hosts: Vec<String>,
    /// The pictures its lines number (`[image 1]` is the first).
    pictures: Vec<std::path::PathBuf>,
}

/// The same hosts, in any order.
fn same_hosts(a: &[String], b: &[String]) -> bool {
    a.len() == b.len() && a.iter().all(|h| b.contains(h))
}

/// A message a pass may read.
fn readable(m: &super::ChatMsg) -> bool {
    (m.remember || m.picture_memory) && (!m.text.trim().is_empty() || m.image.is_some())
}

/// The part of `chat` from message `from` on that a pass reads: up to the
/// newest finished reply, or as much as fits one pass, or up to where the
/// messages went somewhere else (a Link host, or only here) — the rest is
/// the next part. `None` when there's nothing from `from` on. The lines may
/// be empty: nothing there may be read (role play, say), and the part only
/// needs marking as read. With `streaming`, the chat's last message is a
/// reply still being written, so it waits.
fn pass_part(chat: &Chat, from: usize, streaming: bool) -> Option<Part> {
    let msgs = if streaming { &chat.msgs[..chat.msgs.len().saturating_sub(1)] } else { &chat.msgs[..] };
    let end = msgs.iter().rposition(|m| m.role == ChatRole::Model)? + 1;
    if from >= end {
        return None;
    }
    let mut part = Part { lines: Vec::new(), upto: end, hosts: Vec::new(), pictures: Vec::new() };
    let mut used = 0;
    for (i, m) in msgs.iter().enumerate().take(end).skip(from) {
        if !readable(m) {
            continue;
        }
        let user = m.role == ChatRole::User;
        // Images are numbered (videos aren't kept); a Krea 2 message is read
        // for its picture only: the image, then the prompt saying what it
        // shows (for an image in an earlier part, nothing).
        let picture = m.image.as_deref().filter(|p| !crate::is_video(p));
        let n = part.pictures.len() + 1;
        let label = match (picture, &m.image) {
            (Some(_), _) => Some(format!("image {n}")),
            (None, Some(p)) => Some(attachment_kind(p).to_string()),
            (None, None) => None,
        };
        let line = if m.remember {
            memory::chat_line(user, &m.text, label.as_deref())
        } else if user {
            memory::chat_line(true, "", label.as_deref())
        } else if !part.pictures.is_empty() {
            memory::picture_line(part.pictures.len(), &m.text)
        } else {
            continue;
        };
        let cost = memory::approx_tokens(&line) + 1;
        if part.lines.is_empty() {
            part.hosts = m.shared_with.clone();
        } else if !same_hosts(&part.hosts, &m.shared_with)
            || used + cost > memory::PASS_CHAT
            || part.lines.len() >= PASS_MAX_LINES
        {
            part.upto = i;
            break;
        }
        used += cost;
        part.lines.push(line);
        if let Some(p) = picture {
            part.pictures.push(p.to_path_buf());
        }
    }
    Some(part)
}

/// How an attachment shows in a pass: `[image]` or `[video]`.
fn attachment_kind(path: &std::path::Path) -> &'static str {
    if crate::is_video(path) { "video" } else { "image" }
}

/// The topics a part touched, and the album's when it has pictures: the
/// pass sees what's kept already (not to keep it twice, and to notice the
/// same thing coming back).
fn with_pictures(mut touched: Vec<String>, part: &Part) -> Vec<String> {
    let pictures = crate::memory::PICTURES_TOPIC;
    if !part.pictures.is_empty() && !touched.iter().any(|t| t.eq_ignore_ascii_case(pictures)) {
        touched.push(pictures.to_string());
    }
    touched
}

/// The first message of `chat` no pass has read yet that a pass may read.
fn first_unread(chat: &Chat) -> Option<&super::ChatMsg> {
    let end = chat.msgs.iter().rposition(|m| m.role == ChatRole::Model)? + 1;
    chat.msgs.iter().take(end).skip(chat.memory_synced).find(|m| readable(m))
}

/// Chats waiting for a pass: the parts the last run left, and the open
/// chats with something unread. `streaming`: the chat a reply is being
/// written into.
fn waiting(pass: &PassState, chats: &[Chat], streaming: Option<u64>) -> usize {
    pass.pending.len() + chats.iter().filter(|c| has_unread(c, streaming == Some(c.id))).count()
}

/// Something waiting for a pass can run in `places`.
fn can_run_any(pass: &PassState, chats: &[Chat], places: &Places) -> bool {
    pass.pending.iter().any(|p| places.for_part(p.host.as_slice()).is_some())
        || chats.iter().filter_map(first_unread).any(|m| places.for_part(&m.shared_with).is_some())
}

/// `chat` has messages no pass has read yet that a pass may read.
fn has_unread(chat: &Chat, streaming: bool) -> bool {
    let msgs = if streaming { &chat.msgs[..chat.msgs.len().saturating_sub(1)] } else { &chat.msgs[..] };
    let Some(end) = msgs.iter().rposition(|m| m.role == ChatRole::Model).map(|i| i + 1) else {
        return false;
    };
    msgs.iter().take(end).skip(chat.memory_synced).any(readable)
}

impl PassState {
    /// Stop the pass in flight. Nothing it read is marked as read, so it
    /// runs again later.
    fn cancel(&mut self) {
        if let Some(job) = self.job.take() {
            job.cancel.store(true, Ordering::SeqCst);
            super::forget_pass(&job.cancel);
        }
    }
}

/// The memory passes' bookkeeping, over what they work on: borrowed from
/// `LlmState` (tests build one by hand).
struct Books<'a> {
    pass: &'a mut PassState,
    memory: &'a mut MemoryStore,
    chats: &'a mut [Chat],
    /// The chat on screen, by id.
    active: Option<u64>,
}

impl Books<'_> {
    /// Take in what the pass in flight has written; once its answer is
    /// complete (or it ended), act on it.
    fn collect(&mut self) {
        let Some(job) = self.pass.job.as_mut() else { return };
        let mut ended: Option<Result<(), String>> = None;
        loop {
            match job.rx.try_recv() {
                Ok(Msg::Token(t)) => job.answer.push_str(&t),
                Ok(Msg::Done(r)) => {
                    ended = Some(r);
                    break;
                }
                Ok(_) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    ended = Some(Err("the AI stopped".to_string()));
                    break;
                }
            }
        }
        let out = memory::parse_pass(&super::split_channels(&job.answer).1);
        if out.is_none() && ended.is_none() {
            return; // still writing
        }
        let job = self.pass.job.take().expect("checked above");
        // Stopped for something else (a question, a model swap) — unless
        // the answer was complete by then. Either way, nothing more is
        // needed from it.
        let stopped = job.cancel.swap(true, Ordering::SeqCst);
        let cold = job.cold.as_ref().is_some_and(|c| c.load(Ordering::SeqCst));
        super::forget_pass(&job.cancel);
        if debug() {
            eprintln!("[memory] pass answer: {}", job.answer);
        }
        let now = Instant::now();
        match out {
            Some(out) if self.memory.revision == job.revision => {
                self.finish(job.target, &out, &job.pictures);
                self.pass.retry_at = None;
            }
            // The memory changed while the pass ran: it read the old one, so
            // it goes again (from the memory as it is now).
            Some(_) => {}
            // The host has no model loaded: not before the chat is used again.
            None if cold => self.pass.host_cold = Some(now),
            None if stopped => self.pass.retry_at = Some(now + RETRY_AFTER_STOP),
            None => {
                let why = match ended {
                    Some(Err(e)) => e,
                    _ => "no memory update in the answer".to_string(),
                };
                self.failed(job.target, &why);
                self.pass.retry_at = Some(now + RETRY_AFTER_FAILURE);
            }
        }
    }

    /// Write down what a pass decided and mark what it read as read.
    fn finish(&mut self, target: Target, out: &memory::PassOutput, pictures: &[std::path::PathBuf]) {
        let applied = self.memory.apply_pass(out);
        let kept = self.keep_pictures(&out.keep, pictures);
        if debug() {
            eprintln!("[memory] applied: {applied:?}, {kept} pictures kept");
        }
        match target {
            Target::Chat { id, upto } => {
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    c.memory_synced = c.memory_synced.max(upto.min(c.msgs.len()));
                    c.memory_tries = 0;
                }
                if applied.summary {
                    self.pass.summary_chat = Some(id);
                }
            }
            Target::Pending(i) => {
                if i < self.pass.pending.len() {
                    self.pass.pending.remove(i);
                }
                self.memory.save_pending(&self.pass.pending);
                if applied.summary {
                    self.pass.summary_chat = None;
                }
            }
        }
    }

    /// Keep the pictures a pass chose (at most `PASS_MAX_KEEP`) in the
    /// album: a copy, encrypted, under a memory saying what it shows. Not
    /// one that's gone, isn't an image, looks like it holds a secret, or
    /// whose title the album has already. How many were kept.
    fn keep_pictures(&mut self, keep: &[memory::KeepPicture], pictures: &[std::path::PathBuf]) -> usize {
        use crate::memory::{MemoryKind, PICTURES_TOPIC};
        let mut kept = 0;
        let mut used: Vec<usize> = Vec::new();
        for k in keep.iter().filter(|k| k.keepable()).take(memory::PASS_MAX_KEEP) {
            let Some(src) = pictures.get(k.image - 1) else { continue };
            if used.contains(&k.image) || !src.is_file() || crate::is_video(src) {
                continue;
            }
            let description = memory::cut(k.description.trim(), 400);
            let title = if k.title.trim().is_empty() {
                crate::roleplay::title_from(&description)
            } else {
                memory::cut(k.title.trim(), 80)
            };
            let known = self
                .memory
                .find_topic(PICTURES_TOPIC)
                .is_some_and(|ti| self.memory.topics[ti].memories.iter().any(|m| memory::same_words(&m.title, &title)));
            if known {
                continue;
            }
            let why = memory::cut(k.why.trim(), 400);
            match crate::album::save(MemoryKind::Main, src, &why) {
                Ok(name) => {
                    if self.memory.add_entry(PICTURES_TOPIC, &title, &description, true, Some(name.clone())).is_some() {
                        used.push(k.image);
                        kept += 1;
                    } else {
                        crate::album::remove(MemoryKind::Main, &name);
                    }
                }
                Err(e) => eprintln!("[memory] couldn't keep a picture: {e}"),
            }
        }
        kept
    }

    /// A pass failed: it's tried once more, then that part is given up on
    /// (so a chat can't retry forever).
    fn failed(&mut self, target: Target, why: &str) {
        eprintln!("[memory] a memory update failed: {why}");
        match target {
            Target::Chat { id, upto } => {
                if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
                    c.memory_tries += 1;
                    if c.memory_tries >= MAX_TRIES {
                        c.memory_synced = c.memory_synced.max(upto.min(c.msgs.len()));
                        c.memory_tries = 0;
                    }
                }
            }
            Target::Pending(i) => {
                let Some(p) = self.pass.pending.get_mut(i) else { return };
                p.tries += 1;
                if p.tries >= MAX_TRIES {
                    self.pass.pending.remove(i);
                    self.memory.save_pending(&self.pass.pending);
                }
            }
        }
    }

    /// The next part to read that can run in `places`, if one is due: what
    /// the last run left first; then, of the chats left behind or quiet for
    /// `IDLE` (all of them once AI Chat is off), the one waiting longest.
    /// Parts with nothing to read are marked as read on the way. With
    /// nothing due, says when the next part falls due, if one will.
    fn next(&mut self, now: Instant, off: bool, places: &Places) -> Result<Next, Option<Duration>> {
        for (i, p) in self.pass.pending.iter().enumerate() {
            if let Some(on_host) = places.for_part(p.host.as_slice()) {
                return Ok(Next {
                    target: Target::Pending(i),
                    lines: p.lines.clone(),
                    touched: p.touched.clone(),
                    extends_summary: p.extends_summary,
                    pictures: p.pictures.clone(),
                    on_host,
                });
            }
        }
        let mut best: Option<(usize, bool)> = None;
        let mut soonest: Option<Duration> = None;
        for i in 0..self.chats.len() {
            let c = &self.chats[i];
            let Some(first) = first_unread(c) else {
                // Nothing there a pass may read (role play, say): marked read.
                if let Some(end) = c.msgs.iter().rposition(|m| m.role == ChatRole::Model).map(|e| e + 1)
                    && c.memory_synced < end
                {
                    self.chats[i].memory_synced = end;
                }
                continue;
            };
            // Its next part goes where its first message went.
            let Some(on_host) = places.for_part(&first.shared_with) else { continue };
            let since = c.changed.map_or(IDLE, |t| now.saturating_duration_since(t));
            if !off && Some(c.id) == self.active && since < IDLE {
                soonest = Some(soonest.map_or(IDLE - since, |s| s.min(IDLE - since)));
                continue;
            }
            if best.is_none_or(|(b, _)| c.changed < self.chats[b].changed) {
                best = Some((i, on_host));
            }
        }
        let Some((i, on_host)) = best else { return Err(soonest) };
        let c = &self.chats[i];
        // Something unread there, so the part has lines.
        let Some(part) = pass_part(c, c.memory_synced, false) else { return Err(soonest) };
        Ok(Next {
            target: Target::Chat { id: c.id, upto: part.upto },
            touched: with_pictures(self.memory.touched_topics(&part.lines.join("\n"), &c.memory_topics), &part),
            lines: part.lines,
            extends_summary: self.pass.summary_chat == Some(c.id),
            pictures: part.pictures,
            on_host,
        })
    }

    /// Stop the pass in flight and drop everything waiting for one — what
    /// the last run left, and every open chat's unread part.
    fn reset(&mut self) {
        self.pass.cancel();
        self.pass.pending.clear();
        self.pass.pending_loaded = true;
        self.pass.retry_at = None;
        self.memory.save_pending(&[]);
        self.pass.summary_chat = None;
        for c in self.chats.iter_mut() {
            c.memory_synced = c.msgs.len();
            c.memory_topics.clear();
            c.memory_tries = 0;
        }
    }

    /// Keep what no pass has read yet for the next run, encrypted beside the
    /// memories: what the last run left, then every chat's unread parts —
    /// the newest `MAX_PENDING` of them, without any message that looks like
    /// it holds a secret.
    fn keep_for_later(&mut self, streaming: Option<u64>) {
        self.pass.cancel();
        let mut keep = std::mem::take(&mut self.pass.pending);
        for c in self.chats.iter() {
            let extends = self.pass.summary_chat == Some(c.id);
            let mut from = c.memory_synced;
            let mut first = true;
            while let Some(part) = pass_part(c, from, streaming == Some(c.id)) {
                if !part.lines.is_empty() {
                    let touched =
                        with_pictures(self.memory.touched_topics(&part.lines.join("\n"), &c.memory_topics), &part);
                    keep.push(PendingChat {
                        lines: part.lines.iter().map(|l| memory::redact_line(l)).collect(),
                        touched,
                        extends_summary: extends || !first,
                        host: part.hosts.last().cloned(),
                        pictures: part.pictures.clone(),
                        tries: 0,
                    });
                    first = false;
                }
                if part.upto <= from {
                    break;
                }
                from = part.upto;
            }
        }
        let skip = keep.len().saturating_sub(MAX_PENDING);
        self.memory.save_pending(&keep[skip..]);
        if debug() {
            eprintln!("[memory] kept {} chat parts for the next run", keep.len() - skip);
        }
    }
}

impl LlmState {
    /// The passes' bookkeeping over this chat state.
    fn books(&mut self) -> Books<'_> {
        let active = self.chats.get(self.active_chat).map(|c| c.id);
        Books { pass: &mut self.mem_pass, memory: &mut self.memory, chats: &mut self.chats, active }
    }

    /// Mirror the "Remember across chats" setting (main.rs, every frame). A
    /// change starts afresh: nothing said before memory came on is read,
    /// nor anything waiting when it went off.
    pub fn set_memory_enabled(&mut self, on: bool) {
        match self.mem_pass.enabled.replace(on) {
            Some(was) if was != on => self.reset_memory_queue(),
            // Off at launch: nothing an older run left is kept either.
            None if !on => self.memory.save_pending(&[]),
            _ => {}
        }
    }

    /// Mirror the "Activate AI Chat" setting (main.rs, every frame): memory
    /// passes start only while it's on (and in the grace after it's
    /// switched off), never because some other view polls the chat state.
    pub fn set_ai_chat_on(&mut self, on: bool) {
        self.mem_pass.chat_on = on;
    }

    /// Memory is on, and its store can be read and written.
    pub(super) fn memory_on(&self) -> bool {
        self.mem_pass.enabled == Some(true) && !self.memory.load_failed
    }

    /// What's said now may go into memory: it's on, and no conversation mode
    /// with a brief of its own is.
    pub(super) fn memory_wanted(&self) -> bool {
        self.memory_recalled() && !self.krea_prompts
    }

    /// Pictures sent now may go into the album (but not what's said): in
    /// Krea 2 prompts, with the memory on.
    pub(super) fn pictures_wanted(&self) -> bool {
        self.memory_recalled() && self.krea_prompts
    }

    /// Replies now may use the memory: it's on, and neither role play (it
    /// has its own diary) nor Tag images is. Krea 2 prompts recall, and
    /// save only their pictures.
    pub(super) fn memory_recalled(&self) -> bool {
        self.memory_on() && !self.roleplay.enabled && !self.tag_images
    }

    /// Stop the pass in flight and drop everything waiting for one, so
    /// nothing from before comes back: for memory switched on or off, Clear
    /// all and Start fresh.
    pub fn reset_memory_queue(&mut self) {
        self.books().reset();
    }

    /// What the memory is doing.
    pub fn memory_activity(&self) -> MemoryActivity {
        if self.mem_pass.enabled != Some(true) {
            return MemoryActivity::Off;
        }
        if self.memory.load_failed {
            return MemoryActivity::Unreadable;
        }
        if self.mem_pass.job.is_some() {
            return MemoryActivity::Updating;
        }
        match waiting(&self.mem_pass, &self.chats, self.gen_chat.filter(|_| self.running)) {
            0 => MemoryActivity::UpToDate,
            n => MemoryActivity::Waiting(n),
        }
    }

    /// Something has to drive the memory (or the reply) while AI Chat is
    /// hidden: main.rs keeps polling then.
    pub fn needs_poll(&self) -> bool {
        self.running || self.unload_when_done || self.mem_pass.job.is_some()
    }

    /// The memory note for the next reply in the active chat, when there is
    /// one: the user's turn and the model's acknowledgement, to pin before
    /// the history. Remembers which topics it showed (their stored titles go
    /// to the chat's pass, when what's said now is saved).
    pub(super) fn memory_note(&mut self) -> Option<[CmdMsg; 2]> {
        if !self.memory_recalled() {
            return None;
        }
        let saved = self.memory_wanted();
        // What's being said: the newest message counts most. Long ones count
        // by their start (the matching runs on the UI thread).
        const QUERY_CHARS: usize = 2000;
        let chat = &self.chats[self.active_chat];
        let mut said = chat.msgs.iter().rev().filter(|m| m.role == ChatRole::User).map(|m| m.text.as_str());
        let (last, before) = (said.next().unwrap_or(""), said.next().unwrap_or(""));
        let reply = chat
            .msgs
            .iter()
            .rev()
            .find(|m| m.role == ChatRole::Model && !m.text.trim().is_empty())
            .map_or("", |m| m.text.as_str());
        let query = [
            (memory::head_chars(last, QUERY_CHARS), 1.0),
            (memory::head_chars(before, QUERY_CHARS), 0.5),
            (memory::head_chars(reply, QUERY_CHARS), 0.3),
        ];
        let this_chat = self.mem_pass.summary_chat == Some(chat.id);
        let (mut note, shown) = self.memory.chat_brief(&query, this_chat)?;
        if !saved {
            note.push_str(RECALL_ONLY);
        }
        if debug() {
            eprintln!("[memory] note, ~{} tokens:\n{note}", memory::approx_tokens(&note));
        }
        let chat = &mut self.chats[self.active_chat];
        for topic in shown.into_iter().filter(|_| saved) {
            if chat.memory_topics.len() < 24 && !chat.memory_topics.contains(&topic) {
                chat.memory_topics.push(topic);
            }
        }
        Some([
            CmdMsg { user: true, text: note, image: None },
            CmdMsg { user: false, text: NOTE_ACK.to_string(), image: None },
        ])
    }

    /// A message was sent, or a reply finished, in chat `id`.
    pub(super) fn chat_changed(&mut self, id: u64) {
        let now = Instant::now();
        self.mem_pass.last_activity = Some(now);
        if let Some(c) = self.chats.iter_mut().find(|c| c.id == id) {
            c.changed = Some(now);
        }
    }

    /// Stop the pass in flight (it runs again later).
    pub(super) fn cancel_pass(&mut self) {
        self.mem_pass.cancel();
    }

    /// Stop the pass in flight if it reads chat `id` (deleted, or cut back).
    pub(super) fn cancel_pass_for(&mut self, id: u64) {
        let reads = |j: &Job| matches!(j.target, Target::Chat { id: c, .. } if c == id);
        if self.mem_pass.job.as_ref().is_some_and(reads) {
            self.mem_pass.cancel();
        }
    }

    /// AI Chat was switched off: the model goes once the reply in flight and
    /// the passes still due are done (within `OFF_GRACE`).
    pub(super) fn begin_unload(&mut self) {
        self.unload_when_done = true;
        self.mem_pass.off_until = Some(Instant::now() + OFF_GRACE);
        self.finish_unload_if_idle();
    }

    /// AI Chat came back on before the model went.
    pub(super) fn keep_loaded(&mut self) {
        self.unload_when_done = false;
        self.mem_pass.off_until = None;
    }

    /// Free the model now if AI Chat is off and nothing needs it any more.
    pub(super) fn finish_unload_if_idle(&mut self) {
        if !self.unload_when_done || self.running || self.memory_work_left() {
            return;
        }
        self.unload_when_done = false;
        self.mem_pass.off_until = None;
        super::release_shared_worker();
    }

    /// With AI Chat off: a pass is running, or one is due that can run now.
    fn memory_work_left(&self) -> bool {
        let now = Instant::now();
        if !self.memory_on() || self.mem_pass.off_until.is_none_or(|t| now >= t) {
            return false;
        }
        if self.mem_pass.job.is_some() {
            return true;
        }
        // One that just failed or was stopped doesn't hold the model.
        if self.mem_pass.retry_at.is_some_and(|t| now < t) {
            return false;
        }
        can_run_any(&self.mem_pass, &self.chats, &self.places())
    }

    /// Where a pass could run right now without loading a model or making
    /// anyone wait: on the loaded worker, when it's the chat's model and no
    /// Link device is using it; on the Link host the chat runs on, while it
    /// most likely still has its model loaded and hasn't said otherwise since
    /// the chat was last used.
    fn places(&self) -> Places {
        let local = super::worker_matches(self.model, self.image_detail) && !super::link_busy();
        let warm = self.mem_pass.last_activity.is_some_and(|t| {
            t.elapsed() < HOST_WARM && self.mem_pass.host_cold.is_none_or(|cold| t > cold)
        });
        let host = crate::link::remote_chat().filter(|_| warm).map(|(peer, _)| host_id(&peer));
        Places { local, host }
    }

    /// Every frame, from `poll`: take in the pass's answer, and start the
    /// next pass when one is due.
    pub(super) fn tick_memory(&mut self, ctx: &egui::Context) {
        if !self.memory_on() {
            self.mem_pass.cancel();
            return;
        }
        if !self.mem_pass.pending_loaded {
            self.mem_pass.pending_loaded = true;
            self.mem_pass.pending = self.memory.load_pending();
        }
        if self.unload_when_done
            && let Some(until) = self.mem_pass.off_until
        {
            let now = Instant::now();
            if now >= until {
                // Out of time: the rest waits (for next time, or the next run).
                self.mem_pass.cancel();
                return;
            }
            ctx.request_repaint_after(until - now);
        }
        self.books().collect();
        if self.mem_pass.job.is_none() {
            self.start_pass(ctx);
        }
    }

    /// Start the next pass, when one is due and can run without making
    /// anyone wait.
    fn start_pass(&mut self, ctx: &egui::Context) {
        if self.running {
            return;
        }
        let now = Instant::now();
        let off = self.unload_when_done;
        if !off {
            // Only with the chat on, and a message being written (or edited)
            // goes first.
            let on_screen = self.chats.get(self.active_chat).map(|c| c.id);
            if !self.mem_pass.chat_on
                || !self.draft.trim().is_empty()
                || self.draft_image.is_some()
                || self.editing.is_some_and(|(id, _)| Some(id) == on_screen)
            {
                return;
            }
            if let Some(t) = self.mem_pass.last_activity
                && now < t + QUIET
            {
                ctx.request_repaint_after(t + QUIET - now);
                return;
            }
        }
        if let Some(t) = self.mem_pass.retry_at
            && now < t
        {
            ctx.request_repaint_after(t - now);
            return;
        }
        let places = self.places();
        if !places.local && places.host.is_none() {
            // A Link device's question just ended: look again shortly.
            if super::link_busy() {
                ctx.request_repaint_after(Duration::from_secs(5));
            }
            return;
        }
        let next = match self.books().next(now, off, &places) {
            Ok(next) => next,
            Err(again) => {
                if let Some(d) = again {
                    ctx.request_repaint_after(d);
                }
                return;
            }
        };
        let head = self.memory.pass_head(&next.touched, next.extends_summary, next.pictures.len());
        let params = GenParams {
            temperature: 0.2,
            top_k: self.params.top_k,
            top_p: self.params.top_p,
            max_tokens: PASS_REPLY,
            thinking: ThinkLevel::Off,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut cold = None;
        let sent = if next.on_host {
            // On the Link host, as a question of its own (in parts, for a
            // host that knows passes).
            let Some((peer, host)) = crate::link::remote_chat() else { return };
            let question = vec![CmdMsg { user: true, text: memory::assemble_pass(&head, &next.lines), image: None }];
            let flags = RemoteFlags { cancel: cancel.clone(), cut_off: Arc::default() };
            let is_cold = Arc::new(AtomicBool::new(false));
            cold = Some(is_cold.clone());
            let pass = RemotePass { flags, head, lines: next.lines, cold: is_cold };
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .name("ai-memory-pass".into())
                .spawn(move || super::remote_ask(&peer, &host, question, params, 0, false, Some(pass), &tx, &ctx))
                .is_ok()
        } else {
            // Marked before it's queued: a question arriving meanwhile must
            // be able to stop it.
            super::note_pass(&cancel);
            let cmd = Cmd::Pass { head, lines: next.lines, params, reply: tx, cancel: cancel.clone() };
            let queued = super::submit_loaded(self.model, self.image_detail, cmd).is_ok();
            if !queued {
                super::forget_pass(&cancel);
            }
            queued
        };
        if sent {
            if debug() {
                eprintln!("[memory] pass started: {:?}, on the {}", next.target, if next.on_host { "host" } else { "worker" });
            }
            self.mem_pass.job = Some(Job {
                target: next.target,
                revision: self.memory.revision,
                cancel,
                cold,
                rx,
                answer: String::new(),
                pictures: next.pictures,
            });
        }
    }

    /// The app is closing: what no pass has read yet is kept for the next
    /// run (encrypted, beside the memories).
    pub fn save_memory_for_later(&mut self) {
        if !self.memory_on() {
            return;
        }
        if !self.mem_pass.pending_loaded {
            self.mem_pass.pending_loaded = true;
            self.mem_pass.pending = self.memory.load_pending();
        }
        let streaming = self.gen_chat.filter(|_| self.running);
        self.books().keep_for_later(streaming);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMsg;
    use crate::memory::MemoryKind;

    fn msg(user: bool, text: &str, remember: bool) -> ChatMsg {
        ChatMsg {
            role: if user { ChatRole::User } else { ChatRole::Model },
            text: text.to_string(),
            thinking: None,
            image: None,
            tag_image: None,
            tag_change: None,
            remember,
            picture_memory: false,
            shared_with: Vec::new(),
        }
    }

    /// A chat of `(user, text, remember)` messages.
    fn chat(id: u64, msgs: &[(bool, &str, bool)]) -> Chat {
        let mut c = Chat::new(id);
        c.msgs = msgs.iter().map(|(u, t, r)| msg(*u, t, *r)).collect();
        c
    }

    const ANSWER: &str = "{\"summary\": \"We talked about her sister.\", \"persona\": \"\", \"remember\": \
                          [{\"topic\": \"Family\", \"title\": \"Sister lives in Lisbon\", \"text\": \"Her sister \
                          Ana lives in Lisbon.\"}], \"forget\": []}";

    /// A pass in flight for `target`, fed through the returned sender (and,
    /// for a host's, its "not loaded" flag).
    fn job(pass: &mut PassState, memory: &MemoryStore, target: Target) -> (std::sync::mpsc::Sender<Msg>, Arc<AtomicBool>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cold = Arc::new(AtomicBool::new(false));
        pass.job = Some(Job {
            target,
            revision: memory.revision,
            cancel: Arc::new(AtomicBool::new(false)),
            cold: Some(cold.clone()),
            rx,
            answer: String::new(),
            pictures: Vec::new(),
        });
        (tx, cold)
    }

    const HERE: Places = Places { local: true, host: None };

    /// A pass reads up to the newest finished reply, skips what may not be
    /// read, and a long chat goes in parts.
    #[test]
    fn the_part_a_pass_reads() {
        let c = chat(
            7,
            &[
                (true, "My sister lives in Lisbon", true),
                (false, "Nice! Do you visit her?", true),
                (true, "*waves wand*", false),
                (false, "*the castle doors open*", false),
                (true, "Back to normal: I'm a nurse", true),
            ],
        );
        let part = pass_part(&c, 0, false).unwrap();
        assert_eq!(part.lines, vec!["User: My sister lives in Lisbon", "You: Nice! Do you visit her?"]);
        assert_eq!(part.upto, 4, "the unanswered message waits for its reply");
        assert!(part.hosts.is_empty());
        assert!(has_unread(&c, false));
        assert_eq!(pass_part(&c, 4, false), None);
        let part = pass_part(&c, 2, false).unwrap();
        assert!(part.lines.is_empty() && part.upto == 4, "role play only: nothing to read, just marking");
        let mut read = chat(7, &[(true, "a", true), (false, "b", true), (true, "c", false), (false, "d", false)]);
        read.memory_synced = 2;
        assert!(!has_unread(&read, false));
        let streaming = chat(8, &[(true, "Hi", true), (false, "Hel", true)]);
        assert_eq!(pass_part(&streaming, 0, true), None, "the reply being written waits");
        assert!(!has_unread(&streaming, true));
        let texts: Vec<String> = (0..80).map(|i| format!("message {i} {}", "word ".repeat(60))).collect();
        let long: Vec<(bool, &str, bool)> = texts.iter().enumerate().map(|(i, t)| (i % 2 == 0, t.as_str(), true)).collect();
        let c = chat(9, &long);
        let (mut from, mut parts, mut seen) = (0, 0, 0);
        while let Some(part) = pass_part(&c, from, false) {
            let cost: usize = part.lines.iter().map(|l| memory::approx_tokens(l) + 1).sum();
            assert!(cost <= memory::PASS_CHAT);
            seen += part.lines.len();
            parts += 1;
            from = part.upto;
        }
        assert!(parts > 1, "a long chat goes in parts");
        assert_eq!(seen, 80, "none skipped");
    }

    /// A finished pass is written down, and what it read is marked read —
    /// even when the answer is complete before the model ends its turn.
    #[test]
    fn a_finished_pass_is_written_down() {
        let _root = memory::scratch_root("pass_finish");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let mut chats = vec![chat(7, &[(true, "My sister Ana lives in Lisbon", true), (false, "Lovely!", true)])];
        let (tx, _) = job(&mut pass, &memory, Target::Chat { id: 7, upto: 2 });
        let (a, b) = ANSWER.split_at(60);
        tx.send(Msg::Token(a.to_string())).unwrap();
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: Some(7) };
        books.collect();
        assert!(books.pass.job.is_some(), "still writing");
        tx.send(Msg::Token(b.to_string())).unwrap();
        books.collect();
        assert!(books.pass.job.is_none());
        assert_eq!(books.chats[0].memory_synced, 2);
        assert_eq!(books.pass.summary_chat, Some(7));
        assert_eq!(books.memory.header.summary, "We talked about her sister.");
        assert_eq!(books.memory.count(), 1);
        assert_eq!(books.pass.retry_at, None);
        assert_eq!(MemoryStore::load(MemoryKind::Main).count(), 1, "on disk");
    }

    /// Pictures are numbered as a pass reads them; a Krea 2 message is read
    /// for its picture only (the prompt saying what it shows, never the
    /// words typed), and a video isn't numbered.
    #[test]
    fn pictures_in_a_pass() {
        let mut c = chat(1, &[(true, "My dog Max!", true), (false, "What a good boy.", true)]);
        c.msgs[0].image = Some("C:/pics/max.jpg".into());
        let mut krea = msg(true, "make it moody", false);
        krea.image = Some("C:/pics/cat.png".into());
        krea.picture_memory = true;
        let mut prompt = msg(false, "A grey tabby cat curled on a sunlit windowsill.", false);
        prompt.picture_memory = true;
        let mut clip = msg(true, "and a clip", true);
        clip.image = Some("C:/pics/run.mp4".into());
        let ignored = msg(true, "a Krea 2 idea in words", false);
        c.msgs.extend([krea, prompt, ignored, clip, msg(false, "Fun!", true)]);
        let part = pass_part(&c, 0, false).unwrap();
        assert_eq!(
            part.lines,
            vec![
                "User: [image 1] My dog Max!",
                "You: What a good boy.",
                "User: [image 2]",
                "You (the Krea 2 prompt you wrote for image 2, only to know what it shows): A grey tabby cat curled \
                 on a sunlit windowsill.",
                "User: [video] and a clip",
                "You: Fun!",
            ]
        );
        assert_eq!(part.pictures, vec![std::path::PathBuf::from("C:/pics/max.jpg"), "C:/pics/cat.png".into()]);
        assert_eq!(with_pictures(vec!["Pets".into()], &part), vec!["Pets".to_string(), "Pictures".into()]);
        // A Krea 2 prompt whose picture is in an earlier part says nothing.
        let later = pass_part(&c, 3, false).unwrap();
        assert_eq!(later.lines[0], "User: [video] and a clip");
        assert!(later.pictures.is_empty());
    }

    /// A pass's chosen pictures go into the album, encrypted, under a memory
    /// in the Pictures topic — not a picture that's gone, a number that isn't
    /// there, one that looks like a secret, or one already kept.
    #[test]
    fn a_pass_keeps_pictures() {
        let _root = memory::scratch_root("pass_album");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let dog = memory::root().join("max.jpg");
        std::fs::write(&dog, b"jpeg bytes").unwrap();
        let mut chats = vec![chat(7, &[(true, "My dog Max!", true), (false, "Adorable.", true)])];
        chats[0].msgs[0].image = Some(dog.clone());
        let pictures = vec![dog.clone(), memory::root().join("gone.jpg")];
        let keep = |image: usize, title: &str, description: &str| memory::KeepPicture {
            image,
            title: title.into(),
            description: description.into(),
            why: "They love him so much.".into(),
        };
        let out = memory::PassOutput {
            keep: vec![
                keep(1, "Max the dog", "Their golden retriever Max asleep on a blue rug."),
                keep(2, "Gone", "A picture that isn't there any more."),
            ],
            ..Default::default()
        };
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: Some(7) };
        books.finish(Target::Chat { id: 7, upto: 2 }, &out, &pictures);
        let ti = books.memory.find_topic("Pictures").expect("the album's topic");
        let m = &books.memory.topics[ti].memories[0];
        assert_eq!(books.memory.topics[ti].memories.len(), 1);
        assert_eq!((m.title.as_str(), m.text.as_str()), ("Max the dog", "Their golden retriever Max asleep on a blue rug."));
        let name = m.image.clone().expect("names its picture");
        assert_eq!(std::fs::read(crate::album::load(MemoryKind::Main, &name).unwrap()).unwrap(), b"jpeg bytes");
        assert_eq!(crate::album::why(MemoryKind::Main, &name).as_deref(), Some("They love him so much."));
        // Kept already, out of range, a secret: nothing more.
        let again = memory::PassOutput {
            keep: vec![
                keep(1, "Max the dog", "Max again."),
                keep(9, "Nine", "No ninth picture."),
                keep(1, "Wifi card", "The wifi password is hunter2-xyz on a card."),
            ],
            ..Default::default()
        };
        assert_eq!(books.keep_pictures(&again.keep, &pictures), 0);
        assert_eq!(books.memory.topics[ti].memories.len(), 1);
        crate::album::clear(MemoryKind::Main);
    }

    /// The user changed the memory while the pass ran: its answer is thrown
    /// away and the chat stays unread. A pass stopped for something else
    /// doesn't count as a try but waits a little; a failing one waits
    /// longer, is tried once more, then given up on; a host without its
    /// model loaded gets no more passes until the chat is used again.
    #[test]
    fn passes_that_dont_count() {
        let _root = memory::scratch_root("pass_discard");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let mut chats = vec![chat(7, &[(true, "My sister Ana lives in Lisbon", true), (false, "Lovely!", true)])];
        let target = Target::Chat { id: 7, upto: 2 };
        let (tx, _) = job(&mut pass, &memory, target);
        memory.add("Family", "Brother", "Has a brother."); // edited in the map meanwhile
        tx.send(Msg::Token(ANSWER.to_string())).unwrap();
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: Some(7) };
        books.collect();
        assert!(books.pass.job.is_none());
        assert_eq!(books.memory.count(), 1, "nothing written");
        assert_eq!((books.chats[0].memory_synced, books.chats[0].memory_tries), (0, 0));
        assert_eq!(books.pass.retry_at, None, "goes again straight away");
        // Stopped (a question came): no try counted, a short wait.
        let (tx, _) = job(books.pass, books.memory, target);
        books.pass.job.as_ref().unwrap().cancel.store(true, Ordering::SeqCst);
        tx.send(Msg::Done(Ok(()))).unwrap();
        books.collect();
        assert_eq!((books.chats[0].memory_synced, books.chats[0].memory_tries), (0, 0));
        let wait = books.pass.retry_at.expect("waits").saturating_duration_since(Instant::now());
        assert!(wait > RETRY_AFTER_STOP - Duration::from_secs(5) && wait <= RETRY_AFTER_STOP);
        // Fails twice: a longer wait each time, then given up on.
        for tries in [1, 0] {
            let (tx, _) = job(books.pass, books.memory, target);
            tx.send(Msg::Token("Sorry, I can't do that.".into())).unwrap();
            tx.send(Msg::Done(Ok(()))).unwrap();
            books.collect();
            assert_eq!(books.chats[0].memory_tries, tries);
            let wait = books.pass.retry_at.expect("waits").saturating_duration_since(Instant::now());
            assert!(wait > RETRY_AFTER_STOP);
        }
        assert_eq!(books.chats[0].memory_synced, 2);
        // The worker going away mid-pass is a failure too.
        let (tx, _) = job(books.pass, books.memory, target);
        drop(tx);
        books.collect();
        assert!(books.pass.job.is_none());
        // The host said its model isn't loaded: not counted, the host is
        // marked cold.
        books.chats[0].memory_synced = 0;
        books.chats[0].memory_tries = 0;
        books.pass.retry_at = None;
        let (tx, cold) = job(books.pass, books.memory, target);
        books.pass.job.as_ref().unwrap().cancel.store(true, Ordering::SeqCst);
        cold.store(true, Ordering::SeqCst);
        tx.send(Msg::Done(Ok(()))).unwrap();
        books.collect();
        assert!(books.pass.host_cold.is_some());
        assert_eq!((books.chats[0].memory_tries, books.pass.retry_at), (0, None));
    }

    /// What's due: the last run's parts first; a chat left behind at once,
    /// the one on screen after three quiet minutes, every chat once AI Chat
    /// is off; parts with nothing to read are just marked.
    #[test]
    fn what_is_due() {
        let _root = memory::scratch_root("pass_due");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let now = Instant::now();
        let mut chats = vec![
            chat(1, &[(true, "I'm a nurse", true), (false, "Nice.", true)]),
            chat(2, &[(true, "*waves wand*", false), (false, "*poof*", false)]),
        ];
        chats[0].changed = Some(now);
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: Some(1) };
        let again = books.next(now, false, &HERE).err().flatten().expect("due later");
        assert!(again <= IDLE && again > IDLE - Duration::from_secs(5));
        assert_eq!(books.chats[1].memory_synced, 2, "nothing to read there: marked");
        let due = |b: &mut Books, now: Instant, off: bool| b.next(now, off, &HERE).ok().map(|n| n.target);
        assert_eq!(due(&mut books, now + IDLE, false), Some(Target::Chat { id: 1, upto: 2 }), "quiet long enough");
        assert_eq!(due(&mut books, now, true), Some(Target::Chat { id: 1, upto: 2 }), "AI Chat off");
        books.active = Some(2);
        assert_eq!(due(&mut books, now, false), Some(Target::Chat { id: 1, upto: 2 }), "left behind");
        books.pass.pending.push(PendingChat { lines: vec!["User: hi".into()], extends_summary: true, ..Default::default() });
        let n = books.next(now, false, &HERE).ok().unwrap();
        assert!(n.target == Target::Pending(0) && n.extends_summary && n.lines == vec!["User: hi"] && !n.on_host);
        // Its pass done, it's gone — from the file too.
        books.memory.save_pending(&books.pass.pending);
        let (tx, _) = job(books.pass, books.memory, Target::Pending(0));
        tx.send(Msg::Token(ANSWER.to_string())).unwrap();
        books.collect();
        assert!(books.pass.pending.is_empty() && books.memory.load_pending().is_empty());
        assert_eq!(books.pass.summary_chat, None);
        assert_eq!(waiting(books.pass, books.chats, None), 1, "chat 1 still waits");
    }

    /// A message goes to a Link host only if it was sent there; what stayed
    /// on this PC is read on this PC or not at all, and a part never mixes
    /// the two.
    #[test]
    fn chats_go_only_where_they_went() {
        let _root = memory::scratch_root("pass_where");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let now = Instant::now();
        let mut chats = vec![
            chat(1, &[(true, "Asked here", true), (false, "Answered here", true)]),
            chat(2, &[(true, "Asked the tower", true), (false, "Answered there", true)]),
        ];
        for m in &mut chats[1].msgs {
            m.shared_with.push("aa11".into());
        }
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: None };
        let host_only = Places { local: false, host: Some("aa11".into()) };
        let other_host = Places { local: false, host: Some("bb22".into()) };
        let both = Places { local: true, host: Some("aa11".into()) };
        let n = books.next(now, false, &host_only).ok().unwrap();
        assert_eq!((n.target, n.on_host), (Target::Chat { id: 2, upto: 2 }, true));
        assert!(books.next(now, false, &other_host).is_err(), "never to another host");
        assert!(can_run_any(books.pass, books.chats, &host_only));
        assert!(!can_run_any(books.pass, books.chats, &other_host));
        // With the worker loaded here, chat 1 can be read too (here).
        books.chats[1].memory_synced = 2;
        let n = books.next(now, false, &both).ok().unwrap();
        assert_eq!((n.target, n.on_host), (Target::Chat { id: 1, upto: 2 }, false));
        // The last run's parts: only where they went, or here.
        books.chats[0].memory_synced = 2;
        books.pass.pending.push(PendingChat { lines: vec!["User: local".into()], ..Default::default() });
        books.pass.pending.push(PendingChat { lines: vec!["User: tower".into()], host: Some("aa11".into()), ..Default::default() });
        let n = books.next(now, false, &host_only).ok().unwrap();
        assert_eq!((n.target, n.on_host), (Target::Pending(1), true));
        let n = books.next(now, false, &HERE).ok().unwrap();
        assert_eq!((n.target, n.on_host), (Target::Pending(0), false));
        // A chat that ran on the tower, then here: two parts, each going
        // only where its messages went.
        let mut c = chat(3, &[(true, "On the tower", true), (false, "Yes", true), (true, "Here now", true), (false, "Ok", true)]);
        c.msgs[0].shared_with.push("aa11".into());
        c.msgs[1].shared_with.push("aa11".into());
        let first = pass_part(&c, 0, false).unwrap();
        assert_eq!((first.upto, first.hosts.clone()), (2, vec!["aa11".to_string()]));
        let second = pass_part(&c, first.upto, false).unwrap();
        assert_eq!((second.lines.len(), second.upto, second.hosts.is_empty()), (2, 4, true));
    }

    /// Closing keeps every unread part, in parts a pass can take, without
    /// secrets, and where each may go; switching memory off (or clearing
    /// it) drops everything waiting.
    #[test]
    fn kept_for_later_and_reset() {
        let _root = memory::scratch_root("pass_keep");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let texts: Vec<String> = (0..40).map(|i| format!("message {i} {}", "word ".repeat(60))).collect();
        let long: Vec<(bool, &str, bool)> = texts.iter().enumerate().map(|(i, t)| (i % 2 == 0, t.as_str(), true)).collect();
        let mut chats = vec![
            chat(1, &long),
            chat(
                2,
                &[
                    (true, "I'm a nurse", true),
                    (false, "Nice.", true),
                    (true, "my wifi password is hunter2-xyz", true),
                    (false, "Noted, I won't keep it.", true),
                    (true, "Still typ", true),
                ],
            ),
        ];
        for m in &mut chats[1].msgs {
            m.shared_with.push("aa11".into());
        }
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: Some(2) };
        books.pass.summary_chat = Some(2);
        books.keep_for_later(None);
        let kept = books.memory.load_pending();
        assert!(kept.len() >= 3);
        assert!(!kept[0].extends_summary && kept[1].extends_summary, "a long chat's later parts extend its summary");
        assert_eq!(kept[0].host, None);
        let last = kept.last().unwrap();
        assert_eq!(last.lines[..2], ["User: I'm a nurse", "You: Nice."]);
        assert!(last.lines[2].starts_with("User: [left out"), "no password kept");
        assert!(last.extends_summary && last.host.as_deref() == Some("aa11"));
        assert_eq!(kept.iter().map(|p| p.lines.len()).sum::<usize>(), 44);
        books.reset();
        assert!(books.memory.load_pending().is_empty());
        assert!(books.chats.iter().all(|c| c.memory_synced == c.msgs.len()));
        assert_eq!(waiting(books.pass, books.chats, None), 0);
        assert_eq!(books.pass.summary_chat, None);
    }
}
