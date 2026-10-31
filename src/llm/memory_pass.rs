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
//! one by itself — once the user has left the chat (picked another one,
//! started a new one), the chat has been quiet for three minutes, or AI Chat
//! was switched off; and only while AI Chat is on (or just switched off),
//! nothing has been sent or answered for 20 seconds and nothing is being
//! typed. Sending anything stops it at once — the worker reads its prompt in
//! small steps, so even a model partly on the CPU lets go within one — and
//! it runs again later. So does anything else that needs the worker: a Link
//! device's question, unloading, another model.
//!
//! **Where.** On this PC's loaded model, or on the Link host the chat runs
//! on — but only messages already sent to that host go there (each message
//! records the hosts it went to, and a part never mixes them): what stayed
//! on this PC stays here. A host reads a device's pass in steps too, never
//! loads its model for one, and stops it for anything else, like its own.
//!
//! **Known at once.** Until a pass has read it, what was said lately in the
//! other chats (and before the app last closed) rides along in the memory
//! note itself — its newest lines, secrets left out, the AI's own kept short
//! (they're read with every reply) — so a new chat knows what the user just
//! told the AI elsewhere ("your name is Nova"), not only after a pass. **Save
//! now** (AI Settings → Memory, while AI Chat is on) runs the passes due at
//! once instead of waiting for quiet, loading this chat's model if it isn't
//! — never for a chat that runs on a Link host, whose passes go there while
//! its model is loaded.
//!
//! **Never lost, never brought back.** What no pass has read when the app
//! closes is kept for the next run, encrypted beside the memories
//! (`PendingChat`), with any message that looks like it holds a secret left
//! out. A part of a chat whose pass failed twice (an answer that can't be
//! read even mended, see `memory::parse_pass`) is set aside the same way,
//! and the rest of the chat goes on; it gets two more tries each run (or
//! with Save now), leaves the summary to the newer chats read since, goes
//! back to its chat if that's edited over it, goes with the chat if it's
//! deleted, and only after `GIVE_UP` failed passes is it left out — AI
//! Settings says what the last pass did. A pass is thrown away when the
//! user changed the memory while it ran (it worked from the old memory, and
//! would write back what they just removed). Switching memory off, Clear
//! all and Start fresh drop everything waiting. Role play, Krea 2 prompts
//! and Tag images pin briefs of their own, and what's said in them is never
//! read. Krea 2 prompts still recall: its brief comes with the memory note,
//! so prompts can follow the user's taste.
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
use crate::memory::{self, LastPass, MemoryStore, PendingChat, Retry};

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
/// A part of a chat whose pass failed this many times in a run is set aside
/// for the next run (or Save now).
const MAX_TRIES: u8 = 2;
/// A part whose passes failed this many times, over runs, is left out.
const GIVE_UP: u8 = 6;
/// The memory note's lines from the latest chats no pass has read yet, at
/// most (approximate tokens) — every reply reads them — and one of the
/// user's, or of the AI's, at most (characters).
const RECENT_NOTE: usize = 250;
const RECENT_LINE: usize = 300;
const RECENT_REPLY: usize = 160;
/// Ends a pass's instructions for a part set aside (read after newer ones).
const OLDER_PART: &str = "This part of the chat is older than what you remember above (it couldn't be read \
                          before): where the two differ, what you remember is newer, so keep it, and only add \
                          what's missing.\n";
/// Why Save now couldn't save everything (AI Settings).
const HOST_COLD: &str = "the PC your chats run on doesn't have its model loaded right now — it's saved later, \
                         on its own";
const LOCAL_ONLY: &str = "part of it was said while your chats ran on this PC, so it's saved here once this \
                          PC's model is in use again";
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
    /// The user asked to save now: the passes due run without waiting for
    /// quiet, loading the model if needed, until none is left (or something
    /// is sent, or AI Chat is switched off).
    save_now: bool,
    /// Why the last Save now couldn't save everything, if it couldn't.
    save_blocked: Option<&'static str>,
    /// What the last pass did.
    last: Option<LastPass>,
    /// The channel of a Save now that was loading the model when a question
    /// stopped it: the load goes on for the question, and its progress shows
    /// in the chat — with the status it showed last.
    loader: Option<(Receiver<Msg>, String)>,
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
    /// The memories whose details it was shown whole (see
    /// `MemoryStore::pass_seen`): only theirs does it rewrite.
    seen: Vec<u64>,
    /// Save now queued it to load the model, and the load isn't done: its
    /// progress so far.
    loading: bool,
    progress: Option<String>,
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
    /// A part set aside: older than what's remembered now.
    older: bool,
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
    let seen = if streaming { chat.msgs.len().saturating_sub(1) } else { chat.msgs.len() };
    part_of(chat, from, seen)
}

/// [`pass_part`] over the chat's first `seen` messages.
fn part_of(chat: &Chat, from: usize, seen: usize) -> Option<Part> {
    let msgs = &chat.msgs[..seen.min(chat.msgs.len())];
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

/// Something waiting for a pass — and not set aside this run — can run in
/// `places`.
fn can_run_any(pass: &PassState, chats: &[Chat], places: &Places) -> bool {
    pass.pending.iter().any(|p| p.tries < MAX_TRIES && places.for_part(p.host.as_slice()).is_some())
        || chats
            .iter()
            .filter(|c| c.memory_tries < MAX_TRIES)
            .filter_map(first_unread)
            .any(|m| places.for_part(&m.shared_with).is_some())
}

/// Something waiting for a pass — and not set aside this run — can't run in
/// `places` (it went to, or stayed on, a PC that isn't answering).
fn stuck(pass: &PassState, chats: &[Chat], places: &Places) -> bool {
    pass.pending.iter().any(|p| p.tries < MAX_TRIES && places.for_part(p.host.as_slice()).is_none())
        || chats
            .iter()
            .filter(|c| c.memory_tries < MAX_TRIES)
            .filter_map(first_unread)
            .any(|m| places.for_part(&m.shared_with).is_none())
}

/// What Save now does when no pass can run where they usually do (this
/// PC's loaded model, a Link host with its model loaded).
#[derive(Debug, PartialEq)]
enum NoPlace {
    /// Load this chat's model for it.
    Load,
    /// Look again shortly: a Link device is using this PC's model.
    Wait,
    /// Not now — and why, when there's something to say.
    Stop(Option<&'static str>),
}

/// `remote`: the chat runs on a Link host (whose model isn't loaded, or the
/// host would be a place); `busy`: a Link device is using this PC's model;
/// `can_load`: this chat's model can load here.
fn no_place(remote: bool, busy: bool, can_load: bool) -> NoPlace {
    if remote {
        // Never this PC's model for a chat that runs on another PC: it may
        // be one this PC can't hold.
        NoPlace::Stop(Some(HOST_COLD))
    } else if busy {
        NoPlace::Wait
    } else if can_load {
        NoPlace::Load
    } else {
        NoPlace::Stop(None)
    }
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
                Ok(Msg::Token(t)) => {
                    job.loading = false;
                    job.answer.push_str(&t);
                }
                // Save now loading the model: how far it is (AI Settings).
                Ok(Msg::Status(s)) if job.loading => {
                    if s == "Ready" {
                        job.loading = false;
                        job.progress = None;
                    } else {
                        job.progress = Some(s);
                    }
                }
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
        let visible = super::split_channels(&job.answer).1;
        let mut out = memory::parse_pass(&visible);
        if out.is_none() && ended.is_none() {
            return; // still writing
        }
        let target = job.target;
        let cancelled = job.cancel.load(Ordering::SeqCst);
        // Ended before its JSON did: what's complete in it still counts — on
        // the part's last try, or when it holds a memory (for a summary
        // alone, the rest of the part isn't given up yet).
        if out.is_none() && ended.as_ref().is_some_and(|r| r.is_ok()) && !cancelled {
            let last_try = self.last_try(target);
            out = memory::salvage_pass(&visible).filter(|o| last_try || o.has_entries());
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
                self.finish(job.target, &out, &job.pictures, &job.seen);
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
                    _ => "the AI's answer had no memory update in it".to_string(),
                };
                self.failed(job.target, &why);
                self.pass.retry_at = Some(now + RETRY_AFTER_FAILURE);
            }
        }
    }

    /// Write down what a pass decided and mark what it read as read. `seen`:
    /// the memories whose details it was shown whole.
    fn finish(&mut self, target: Target, out: &memory::PassOutput, pictures: &[std::path::PathBuf], seen: &[u64]) {
        // A part set aside is read after newer chats: the summary stays
        // theirs.
        let aside = matches!(target, Target::Pending(i) if self.pass.pending.get(i).is_some_and(|p| p.aside));
        let applied = if aside {
            self.memory.apply_pass(&memory::PassOutput { summary: String::new(), ..out.clone() }, seen)
        } else {
            self.memory.apply_pass(out, seen)
        };
        let kept = self.keep_pictures(&out.keep, pictures);
        self.pass.save_blocked = None;
        self.pass.last = Some(LastPass::Saved {
            at: memory::now(),
            added: applied.added + kept,
            forgotten: applied.forgotten,
            notes: applied.persona,
            summary: applied.summary,
        });
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

    /// A pass failed: it's tried once more in a minute. After that a chat's
    /// part is set aside — kept for the next run (or Save now) with what the
    /// last run left, while the rest of the chat goes on — and a part set
    /// aside before waits for the next run; one that failed `GIVE_UP` times,
    /// over runs, is left out. Never dropped unread without AI Settings
    /// saying so.
    fn failed(&mut self, target: Target, why: &str) {
        eprintln!("[memory] a memory update failed: {why}");
        let retry = match target {
            Target::Chat { id, upto } => self.chat_failed(id, upto),
            Target::Pending(i) => self.pending_failed(i),
        };
        self.pass.last = Some(LastPass::Failed { at: memory::now(), why: why.to_string(), retry });
    }

    /// A pass over chat `id` up to message `upto` failed (see `failed`).
    fn chat_failed(&mut self, id: u64, upto: usize) -> Retry {
        let Some(c) = self.chats.iter_mut().find(|c| c.id == id) else { return Retry::Soon };
        c.memory_tries = c.memory_tries.saturating_add(1);
        if c.memory_tries < MAX_TRIES {
            return Retry::Soon;
        }
        // (Without the part, the whole chat waits for the next run.)
        let from = c.memory_synced;
        let Some(part) = part_of(c, from, upto) else { return Retry::NextRun };
        c.memory_tries = 0;
        c.memory_synced = from.max(part.upto);
        if !part.lines.is_empty() {
            let touched = with_pictures(self.memory.touched_topics(&part.lines.join("\n"), &c.memory_topics), &part);
            self.pass.pending.push(PendingChat {
                lines: part.lines.iter().map(|l| memory::redact_line(l)).collect(),
                touched,
                extends_summary: false,
                host: part.hosts.last().cloned(),
                pictures: part.pictures,
                tries: MAX_TRIES,
                failures: MAX_TRIES,
                aside: true,
                chat: Some((id, from..part.upto)),
            });
            self.memory.save_pending(&self.pass.pending);
        }
        Retry::NextRun
    }

    /// The pass over `target` is its last before the part would be left out:
    /// a cut-off answer then counts even for its summary alone (before that,
    /// a chat's part is set aside and goes again). Not for a part set aside,
    /// whose summary doesn't count: it's left out, and AI Settings says so.
    fn last_try(&self, target: Target) -> bool {
        match target {
            Target::Chat { .. } => false,
            Target::Pending(i) => {
                self.pass.pending.get(i).is_some_and(|p| !p.aside && p.failures.saturating_add(1) >= GIVE_UP)
            }
        }
    }

    /// Parts set aside this run go back to their chat when it was cut back
    /// over them (an edit, a retry: what they read may be gone or changed,
    /// so the chat reads it again as it is now), and go with a deleted chat.
    fn reclaim(&mut self) {
        // The part being read stays put (`cancel_pass_for` stops it if its
        // chat goes), and keeps its place among the rest.
        let reading = match self.pass.job.as_ref().map(|j| j.target) {
            Some(Target::Pending(i)) => Some(i),
            _ => None,
        };
        let chats = &mut *self.chats;
        let (mut at, mut gone_before, mut changed) = (0, 0, false);
        self.pass.pending.retain(|p| {
            let i = at;
            at += 1;
            let keep = match &p.chat {
                _ if Some(i) == reading => true,
                None => true,
                Some((id, span)) => match chats.iter_mut().find(|c| c.id == *id) {
                    None => false,
                    Some(c) if c.memory_synced < span.end => {
                        c.memory_synced = c.memory_synced.min(span.start);
                        c.memory_tries = 0;
                        false
                    }
                    Some(_) => true,
                },
            };
            if !keep {
                changed = true;
                gone_before += usize::from(reading.is_some_and(|r| i < r));
            }
            keep
        });
        if let (Some(r), Some(job)) = (reading, self.pass.job.as_mut()) {
            job.target = Target::Pending(r - gone_before);
        }
        if changed {
            self.memory.save_pending(&self.pass.pending);
        }
    }

    /// A pass over the kept part `i` failed (see `failed`).
    fn pending_failed(&mut self, i: usize) -> Retry {
        let Some(p) = self.pass.pending.get_mut(i) else { return Retry::Soon };
        p.tries = p.tries.saturating_add(1);
        p.failures = p.failures.saturating_add(1);
        let (gave_up, set_aside) = (p.failures >= GIVE_UP, p.tries >= MAX_TRIES);
        if gave_up {
            self.pass.pending.remove(i);
        }
        self.memory.save_pending(&self.pass.pending);
        if gave_up {
            Retry::GaveUp
        } else if set_aside {
            Retry::NextRun
        } else {
            Retry::Soon
        }
    }

    /// The next part to read that can run in `places`, if one is due: what
    /// the last run left first; then, of the chats left behind or quiet for
    /// `IDLE` (all of them once AI Chat is off), the one waiting longest.
    /// Parts with nothing to read are marked as read on the way. With
    /// nothing due, says when the next part falls due, if one will.
    fn next(&mut self, now: Instant, off: bool, places: &Places) -> Result<Next, Option<Duration>> {
        for (i, p) in self.pass.pending.iter().enumerate() {
            if p.tries >= MAX_TRIES {
                continue; // waits for the next run
            }
            if let Some(on_host) = places.for_part(p.host.as_slice()) {
                return Ok(Next {
                    target: Target::Pending(i),
                    lines: p.lines.clone(),
                    touched: p.touched.clone(),
                    extends_summary: p.extends_summary,
                    pictures: p.pictures.clone(),
                    on_host,
                    older: p.aside,
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
            if c.memory_tries >= MAX_TRIES {
                continue; // waits for the next run
            }
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
            older: false,
        })
    }

    /// Stop the pass in flight and drop everything waiting for one — what
    /// the last run left, and every open chat's unread part.
    fn reset(&mut self) {
        self.pass.cancel();
        self.pass.pending.clear();
        self.pass.pending_loaded = true;
        self.pass.retry_at = None;
        self.pass.save_now = false;
        self.pass.save_blocked = None;
        self.pass.last = None;
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
        // (Not a part set aside from a chat deleted or edited since.)
        self.reclaim();
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
                        ..Default::default()
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
        if !on {
            // A Save now ends with AI Chat: nothing may load the model then.
            self.mem_pass.save_now = false;
        }
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

    /// Save now (AI Settings → Memory, while AI Chat is on): the passes due
    /// run at once — the chat on screen included, without waiting for quiet,
    /// and loading the model if it isn't (see `no_place`) — and parts set
    /// aside get another go.
    pub fn save_memory_now(&mut self) {
        if !self.memory_on() || !self.mem_pass.chat_on {
            return;
        }
        self.mem_pass.save_now = true;
        self.mem_pass.save_blocked = None;
        self.mem_pass.retry_at = None;
        self.mem_pass.host_cold = None;
        for c in self.chats.iter_mut() {
            c.memory_tries = 0;
        }
        for p in self.mem_pass.pending.iter_mut() {
            p.tries = 0;
        }
    }

    /// Save now was asked for and is still going.
    pub fn saving_memory_now(&self) -> bool {
        self.mem_pass.save_now
    }

    /// What the last pass did, if one ran this session.
    pub fn memory_last(&self) -> Option<&LastPass> {
        self.mem_pass.last.as_ref()
    }

    /// Save now is loading the model first: how far it is.
    pub fn memory_loading(&self) -> Option<&str> {
        self.mem_pass.job.as_ref().filter(|j| j.loading).and_then(|j| j.progress.as_deref())
    }

    /// Why the last Save now couldn't save everything, if it couldn't.
    pub fn memory_save_blocked(&self) -> Option<&'static str> {
        self.mem_pass.save_blocked
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

    /// What was said lately in the other chats — and before the app last
    /// closed — that no pass has read yet, for the memory note: the newest
    /// lines within `RECENT_NOTE` tokens, oldest first, without what looks
    /// like a secret or a Krea 2 picture's prompt. (The chat on screen has
    /// its own history.) When the reply comes from a Link host (`host`, by
    /// key), only lines already sent there: what stayed on this PC stays
    /// here, as with the passes.
    fn unsaved_lines(&mut self, host: Option<&str>) -> Vec<String> {
        if !self.mem_pass.pending_loaded {
            self.mem_pass.pending_loaded = true;
            self.mem_pass.pending = self.memory.load_pending();
        }
        let active = self.chats.get(self.active_chat).map(|c| c.id);
        let streaming = self.gen_chat.filter(|_| self.running);
        // A line that says something: not a Krea 2 picture's prompt, not a
        // bare "[image 2]" or "[left out: …]".
        let says_something = |line: &str| {
            let text = line.split_once(": ").map_or(line, |(_, t)| t).trim();
            let bare = text.starts_with('[') && text.ends_with(']') && !text[1..text.len() - 1].contains('[');
            !line.starts_with("You (") && !text.is_empty() && !bare
        };
        let may_go = |shared: &[String]| host.is_none_or(|h| shared.iter().any(|s| s == h));
        // The user's own words matter most: the AI's are kept shorter.
        let shorten = |line: &str| memory::cut(line, if line.starts_with("User:") { RECENT_LINE } else { RECENT_REPLY });
        let mut lines: Vec<String> = self
            .mem_pass
            .pending
            .iter()
            // (A part set aside from the chat on screen is in its history.)
            .filter(|p| may_go(p.host.as_slice()) && p.chat.as_ref().is_none_or(|(c, _)| Some(*c) != active))
            .flat_map(|p| p.lines.iter())
            .filter(|l| says_something(l))
            .map(|l| shorten(l))
            .collect();
        let mut others: Vec<&Chat> = self.chats.iter().filter(|c| Some(c.id) != active).collect();
        others.sort_by_key(|c| c.changed);
        for c in others {
            let end = if streaming == Some(c.id) { c.msgs.len().saturating_sub(1) } else { c.msgs.len() };
            for m in c.msgs[..end].iter().skip(c.memory_synced) {
                if m.remember && !m.text.trim().is_empty() && may_go(&m.shared_with) {
                    // Secrets out first, then cut (a cut could hide one).
                    let line = memory::redact_line(&memory::chat_line(m.role == ChatRole::User, &m.text, None));
                    if says_something(&line) {
                        lines.push(shorten(&line));
                    }
                }
            }
        }
        // The newest that fit.
        let mut used = 0;
        let mut keep: Vec<String> = Vec::new();
        for line in lines.into_iter().rev() {
            let cost = memory::approx_tokens(&line) + 1;
            if used + cost > RECENT_NOTE {
                break;
            }
            used += cost;
            keep.push(line);
        }
        keep.reverse();
        keep
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
        let host = crate::link::remote_chat().map(|(peer, _)| host_id(&peer));
        let recent = if saved { self.unsaved_lines(host.as_deref()) } else { Vec::new() };
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
        let (mut note, shown) = self.memory.chat_brief(&query, this_chat, &recent)?;
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

    /// Stop the pass in flight (it runs again later) — something was sent,
    /// so a Save now is over too: the rest runs at the usual times.
    pub(super) fn cancel_pass(&mut self) {
        // A Save now loading the model: the question waits for the load
        // anyway, and the chat shows where it is, and how it goes on
        // (`forward_load_progress`).
        if let Some(job) = self.mem_pass.job.as_mut().filter(|j| j.loading) {
            let (_, gone) = std::sync::mpsc::channel();
            let rx = std::mem::replace(&mut job.rx, gone);
            if let Some(progress) = job.progress.clone() {
                self.status = progress;
            }
            self.mem_pass.loader = Some((rx, self.status.clone()));
        }
        self.mem_pass.cancel();
        self.mem_pass.save_now = false;
    }

    /// While a question waits for the model a stopped Save now was loading,
    /// the chat says how far the load is (the worker tells only whoever
    /// asked first: that pass) — until the question's own progress takes
    /// over (another model loading for it, say).
    fn forward_load_progress(&mut self) {
        let Some((rx, shown)) = &self.mem_pass.loader else { return };
        if self.status != *shown {
            self.mem_pass.loader = None;
            return;
        }
        let (mut latest, mut over) = (None, false);
        loop {
            match rx.try_recv() {
                Ok(Msg::Status(s)) if s == "Ready" => {
                    over = true;
                    break;
                }
                Ok(Msg::Status(s)) => latest = Some(s),
                Ok(Msg::Done(_) | Msg::Token(_)) | Err(TryRecvError::Disconnected) => {
                    over = true;
                    break;
                }
                Ok(_) => {}
                Err(TryRecvError::Empty) => break,
            }
        }
        if over {
            self.mem_pass.loader = None;
        } else if let Some(s) = latest
            && self.running
            && !super::GEN_CANCEL.load(Ordering::SeqCst)
            && crate::link::remote_chat().is_none()
        {
            self.status = s.clone();
            if let Some((_, shown)) = &mut self.mem_pass.loader {
                *shown = s;
            }
        }
    }

    /// Stop the pass in flight if it reads chat `id` (deleted, or cut back)
    /// — or a part set aside from it.
    pub(super) fn cancel_pass_for(&mut self, id: u64) {
        let pending = &self.mem_pass.pending;
        let reads = |j: &Job| match j.target {
            Target::Chat { id: c, .. } => c == id,
            Target::Pending(i) => pending.get(i).is_some_and(|p| p.chat.as_ref().is_some_and(|(c, _)| *c == id)),
        };
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
        self.forward_load_progress();
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
        self.books().reclaim();
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
        // Save now (only while AI Chat is on): no waiting for quiet, or for
        // a model to be loaded.
        let asked = self.mem_pass.save_now && self.mem_pass.chat_on;
        if !off && !asked {
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
        let mut places = self.places();
        // Asked to save now with nowhere to run: this chat's model may load
        // for it (never for a pass that just happens to be due).
        let mut load = false;
        if asked && !places.local && places.host.is_none() {
            let remote = crate::link::remote_chat().is_some();
            match no_place(remote, super::link_busy(), super::BUILT_WITH_LLM && self.installed) {
                NoPlace::Load => {
                    load = true;
                    places.local = true;
                }
                NoPlace::Wait => {
                    ctx.request_repaint_after(Duration::from_secs(5));
                    return;
                }
                NoPlace::Stop(why) => {
                    self.mem_pass.save_now = false;
                    self.mem_pass.save_blocked = why;
                    return;
                }
            }
        }
        if !places.local && places.host.is_none() {
            // A Link device's question just ended: look again shortly.
            if super::link_busy() {
                ctx.request_repaint_after(Duration::from_secs(5));
            }
            return;
        }
        let next = match self.books().next(now, off || asked, &places) {
            Ok(next) => next,
            Err(again) => {
                if asked {
                    // Nothing left that can be saved now — but what went to,
                    // or stayed on, a PC that isn't answering still waits.
                    self.mem_pass.save_now = false;
                    if stuck(&self.mem_pass, &self.chats, &places) {
                        self.mem_pass.save_blocked = Some(LOCAL_ONLY);
                    }
                }
                if let Some(d) = again {
                    ctx.request_repaint_after(d);
                }
                return;
            }
        };
        let mut head = self.memory.pass_head(&next.touched, next.extends_summary, next.pictures.len());
        let seen = self.memory.pass_seen(&next.touched);
        if next.older {
            head.push_str(OLDER_PART);
        }
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
        } else if load {
            // Save now, with the model to load first: queued like a question
            // (which loads it), then marked so a real question can stop it.
            let cmd = Cmd::Pass { head, lines: next.lines, params, reply: tx, cancel: cancel.clone() };
            let queued = super::submit(ctx, self.model, self.image_detail, cmd).is_ok();
            if queued {
                super::note_pass(&cancel);
                // A Link device's question that came in meanwhile goes first.
                if super::link_busy() {
                    cancel.store(true, Ordering::SeqCst);
                }
            } else {
                self.mem_pass.save_now = false;
            }
            queued
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
            self.mem_pass.save_blocked = None;
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
                seen,
                loading: load,
                progress: None,
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
            research: None,
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
            seen: Vec::new(),
            loading: false,
            progress: None,
        });
        (tx, cold)
    }

    const HERE: Places = Places { local: true, host: None };

    /// Right after the user tells the AI something, a new chat's memory note
    /// already knows it — before any pass — and so does the first chat of the
    /// next run, from what the last one left.
    #[test]
    fn a_new_chat_knows_what_was_just_said() {
        let _root = memory::scratch_root("repro_bridge");
        let mut llm = LlmState::default();
        llm.roleplay.enabled = false;
        llm.set_memory_enabled(true);
        llm.set_ai_chat_on(true);
        llm.chats[0].msgs.push(msg(true, "From now on your name is Nova. I love cats and sushi.", true));
        llm.chats[0].msgs.push(msg(false, "Nova it is! Cats and sushi, noted.", true));
        llm.new_chat();
        let active = llm.active_chat;
        llm.chats[active].msgs.push(msg(true, "What's your name, and what do I like?", true));
        let note = llm.memory_note().expect("a memory note for the new chat");
        assert!(note[0].text.contains("Nova") && note[0].text.contains("sushi"), "{}", note[0].text);
        assert!(note[0].text.contains("not sorted into your memory yet"), "{}", note[0].text);
        // The chat on screen isn't repeated to itself.
        assert!(!note[0].text.contains("What's your name"), "{}", note[0].text);
        // Once a pass has read that chat, its lines leave the note (the
        // memories take over).
        let len = llm.chats[0].msgs.len();
        llm.chats[0].memory_synced = len;
        assert!(llm.memory_note().is_none(), "nothing remembered, nothing waiting");
        // After a restart: what the last run left counts the same way.
        llm.mem_pass.pending = vec![PendingChat {
            lines: vec!["User: Call me Sam, I'm a nurse.".into(), "You: Hi Sam!".into()],
            ..Default::default()
        }];
        let note = llm.memory_note().expect("the last run's chat");
        assert!(note[0].text.contains("Call me Sam"), "{}", note[0].text);
    }

    /// The note's latest lines: only what may be remembered, no secrets, no
    /// Krea 2 prompts or bare picture lines, the newest within a budget.
    #[test]
    fn the_latest_lines_in_the_note() {
        let _root = memory::scratch_root("latest_lines");
        let mut llm = LlmState::default();
        llm.roleplay.enabled = false;
        llm.set_memory_enabled(true);
        let mut c = chat(5, &[
            (true, "*waves wand* (role play)", false),
            (false, "*poof*", false),
            (true, "my wifi password is hunter2-xyz", true),
            (false, "I won't keep that.", true),
            (true, "I'm vegetarian.", true),
            (false, "Noted!", true),
        ]);
        c.msgs[2].remember = true;
        llm.chats = vec![c];
        llm.mem_pass.pending = vec![PendingChat {
            lines: vec![
                "User: [image 1]".into(),
                "You (the Krea 2 prompt you wrote for image 1, only to know what it shows): A cat.".into(),
                "User: [left out: it looked like it held a password or another secret]".into(),
                "User: I live in Porto.".into(),
            ],
            ..Default::default()
        }];
        llm.mem_pass.pending_loaded = true;
        llm.new_chat();
        let lines = llm.unsaved_lines(None);
        assert_eq!(
            lines,
            vec![
                "User: I live in Porto.".to_string(),
                "You: I won't keep that.".into(),
                "User: I'm vegetarian.".into(),
                "You: Noted!".into(),
            ],
            "the role play, the password and the pictures stay out"
        );
        // However much was said, only the newest fit.
        let long: Vec<(bool, String, bool)> = (0..200).map(|i| (i % 2 == 0, format!("fact number {i} {}", "word ".repeat(30)), true)).collect();
        let refs: Vec<(bool, &str, bool)> = long.iter().map(|(u, t, r)| (*u, t.as_str(), *r)).collect();
        let mut older = chat(9, &refs);
        let then = Instant::now();
        older.changed = Some(then);
        llm.chats.insert(0, older);
        llm.active_chat = llm.chats.len() - 1; // still the new chat
        llm.chats[1].changed = Some(then + Duration::from_secs(600));
        let lines = llm.unsaved_lines(None);
        let cost: usize = lines.iter().map(|l| memory::approx_tokens(l) + 1).sum();
        assert!(cost <= RECENT_NOTE, "{cost}");
        assert!(lines.last().is_some_and(|l| l.contains("Noted!")), "the newest last: {lines:?}");
        assert!(lines.iter().all(|l| l.chars().count() <= RECENT_LINE + 10));
        // The AI's own lines are kept shorter than the user's.
        assert!(lines.iter().any(|l| l.starts_with("User: fact") && l.chars().count() > RECENT_REPLY + 10));
        assert!(lines.iter().filter(|l| l.starts_with("You:")).all(|l| l.chars().count() <= RECENT_REPLY + 10));
        // Answered by a Link host: only what went there.
        llm.chats.remove(0);
        llm.active_chat = llm.chats.len() - 1;
        llm.chats[0].msgs[4].shared_with.push("aa11".into());
        llm.chats[0].msgs[5].shared_with.push("aa11".into());
        llm.mem_pass.pending[0].host = Some("bb22".into());
        assert_eq!(llm.unsaved_lines(Some("aa11")), vec!["User: I'm vegetarian.".to_string(), "You: Noted!".into()]);
    }

    /// Save now: a chat on screen is due at once, failed parts get another
    /// go, and sending something — or switching AI Chat off — ends it.
    #[test]
    fn save_now() {
        let _root = memory::scratch_root("save_now");
        let mut llm = LlmState::default();
        llm.roleplay.enabled = false;
        llm.set_memory_enabled(true);
        llm.set_ai_chat_on(true);
        llm.chats = vec![chat(1, &[(true, "I'm a nurse", true), (false, "Nice.", true)])];
        llm.chats[0].changed = Some(Instant::now());
        llm.chats[0].memory_tries = MAX_TRIES;
        llm.mem_pass.pending_loaded = true;
        llm.mem_pass.retry_at = Some(Instant::now() + Duration::from_secs(60));
        let now = Instant::now();
        assert!(llm.books().next(now, false, &HERE).is_err(), "on screen, quiet too short, failed twice");
        llm.save_memory_now();
        assert!(llm.saving_memory_now());
        assert_eq!((llm.chats[0].memory_tries, llm.mem_pass.retry_at), (0, None));
        let n = llm.books().next(now, true, &HERE).expect("due now");
        assert_eq!(n.target, Target::Chat { id: 1, upto: 2 });
        llm.cancel_pass();
        assert!(!llm.saving_memory_now(), "something was sent: back to the usual times");
        // AI Chat switched off: Save now ends (nothing may load the model
        // then), and can't start until it's back on.
        llm.save_memory_now();
        llm.set_ai_chat_on(false);
        assert!(!llm.saving_memory_now());
        llm.save_memory_now();
        assert!(!llm.saving_memory_now());
        llm.set_ai_chat_on(true);
        llm.set_memory_enabled(false);
        llm.save_memory_now();
        assert!(!llm.saving_memory_now(), "nothing to save with memory off");
    }

    /// With nowhere to run, Save now loads this chat's model — never for a
    /// chat on a Link host, and not while a Link device uses this PC's.
    #[test]
    fn where_save_now_runs() {
        assert_eq!(no_place(false, false, true), NoPlace::Load);
        assert_eq!(no_place(false, true, true), NoPlace::Wait);
        assert_eq!(no_place(false, false, false), NoPlace::Stop(None));
        assert_eq!(no_place(true, false, true), NoPlace::Stop(Some(HOST_COLD)), "the host's model, not this PC's");
        assert_eq!(no_place(true, true, true), NoPlace::Stop(Some(HOST_COLD)));
        // What stayed on this PC while the chat runs on a host waits for this
        // PC's model; what went to the host runs there.
        let host = Places { local: false, host: Some("aa11".into()) };
        let mut pass = PassState::default();
        let mut chats = vec![chat(1, &[(true, "I'm a nurse", true), (false, "Nice.", true)])];
        assert!(stuck(&pass, &chats, &host) && !can_run_any(&pass, &chats, &host));
        for m in chats[0].msgs.iter_mut() {
            m.shared_with.push("aa11".into());
        }
        assert!(!stuck(&pass, &chats, &host) && can_run_any(&pass, &chats, &host));
        // A part set aside this run counts for neither.
        pass.pending.push(PendingChat { lines: vec!["User: hi".into()], tries: MAX_TRIES, ..Default::default() });
        chats.clear();
        assert!(!stuck(&pass, &chats, &host) && !can_run_any(&pass, &chats, &HERE));
    }

    /// What the last pass did is kept for AI Settings.
    #[test]
    fn the_last_pass_is_reported() {
        let _root = memory::scratch_root("last_pass");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let mut chats = vec![chat(7, &[(true, "My sister Ana lives in Lisbon", true), (false, "Lovely!", true)])];
        let (tx, _) = job(&mut pass, &memory, Target::Chat { id: 7, upto: 2 });
        tx.send(Msg::Token(ANSWER.to_string())).unwrap();
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: None };
        books.collect();
        assert!(matches!(
            books.pass.last,
            Some(LastPass::Saved { added: 1, forgotten: 0, notes: false, summary: true, .. })
        ));
    }

    /// An answer that ran out of room still counts for what's complete in
    /// it — but not one that was stopped (it goes again).
    #[test]
    fn a_cut_off_answer_counts_for_what_is_complete() {
        let _root = memory::scratch_root("cut_off");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let mut chats = vec![chat(7, &[(true, "I love cats and sushi, and I hike", true), (false, "Fun!", true)])];
        let cut = "{\"summary\": \"They told me what they love.\", \"remember\": [{\"topic\": \"Likes\", \"title\": \
                   \"Loves cats\", \"text\": \"The user loves cats.\"}, {\"topic\": \"Likes\", \"title\": \"Loves sushi\", \
                   \"text\": \"The user loves su";
        let target = Target::Chat { id: 7, upto: 2 };
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: None };
        // Stopped mid-way: nothing written, read again later.
        let (tx, _) = job(books.pass, books.memory, target);
        tx.send(Msg::Token(cut.into())).unwrap();
        books.pass.job.as_ref().unwrap().cancel.store(true, Ordering::SeqCst);
        tx.send(Msg::Done(Ok(()))).unwrap();
        books.collect();
        assert_eq!((books.memory.count(), books.chats[0].memory_synced), (0, 0));
        // Ran out of room: the complete memory and the summary are kept.
        books.pass.retry_at = None;
        let (tx, _) = job(books.pass, books.memory, target);
        tx.send(Msg::Token(cut.into())).unwrap();
        tx.send(Msg::Done(Ok(()))).unwrap();
        books.collect();
        assert_eq!(books.memory.count(), 1);
        assert_eq!(books.memory.header.summary, "They told me what they love.");
        assert_eq!(books.chats[0].memory_synced, 2);
        // Cut inside its first memory: a summary alone isn't kept — the part
        // goes again, then is set aside (later tries may get the memories
        // too); and set aside, its summary never counts, so on its very last
        // try it's left out, and AI Settings says so.
        books.chats[0].msgs.push(msg(true, "And I'm a nurse, called Sam.", true));
        books.chats[0].msgs.push(msg(false, "Nice to meet you, Sam!", true));
        fn cut_again(books: &mut Books, target: Target) {
            let early = "{\"summary\": \"Sam told me about work.\", \"remember\": [{\"topic\": \"Work\", \"title\": \"Is a nu";
            books.pass.retry_at = None;
            let (tx, _) = job(books.pass, books.memory, target);
            tx.send(Msg::Token(early.into())).unwrap();
            tx.send(Msg::Done(Ok(()))).unwrap();
            books.collect();
        }
        for retry in [Retry::Soon, Retry::NextRun] {
            cut_again(&mut books, Target::Chat { id: 7, upto: 4 });
            assert!(matches!(&books.pass.last, Some(LastPass::Failed { retry: r, .. }) if *r == retry));
        }
        assert_eq!((books.chats[0].memory_synced, books.pass.pending.len()), (4, 1));
        books.pass.pending[0].tries = 0;
        books.pass.pending[0].failures = GIVE_UP - 1;
        cut_again(&mut books, Target::Pending(0));
        assert!(books.pass.pending.is_empty());
        assert!(matches!(&books.pass.last, Some(LastPass::Failed { retry: Retry::GaveUp, .. })));
        assert_eq!(books.memory.header.summary, "They told me what they love.", "the newer summary stays");
        // A part the last run left (not set aside) does count for its
        // summary alone on its last try.
        books.pass.pending.push(PendingChat {
            lines: vec!["User: I'm a nurse.".into()],
            failures: GIVE_UP - 1,
            ..Default::default()
        });
        cut_again(&mut books, Target::Pending(0));
        assert!(books.pass.pending.is_empty());
        assert_eq!(books.memory.header.summary, "Sam told me about work.");
    }

    /// A part set aside goes back to its chat when that's edited over it,
    /// goes with it when it's deleted, never pushes another part out, and
    /// isn't repeated to its own chat.
    #[test]
    fn set_aside_parts_follow_their_chat() {
        let _root = memory::scratch_root("aside_follow");
        let mut llm = LlmState::default();
        llm.roleplay.enabled = false;
        llm.set_memory_enabled(true);
        llm.chats = vec![chat(7, &[(true, "My sister lives in Lisbon.", true), (false, "Lovely!", true)])];
        llm.active_chat = 0;
        llm.mem_pass.pending_loaded = true;
        // Eight parts wait already (on a PC whose chats run on a Link host).
        llm.mem_pass.pending =
            (0..MAX_PENDING).map(|i| PendingChat { lines: vec![format!("User: part {i}")], ..Default::default() }).collect();
        fn fail_twice(llm: &mut LlmState, id: u64) {
            for _ in 0..2 {
                let (tx, _) = job(&mut llm.mem_pass, &llm.memory, Target::Chat { id, upto: 2 });
                tx.send(Msg::Token("Sorry, I can't do that.".into())).unwrap();
                tx.send(Msg::Done(Ok(()))).unwrap();
                llm.books().collect();
            }
        }
        let says = |llm: &mut LlmState, what: &str| llm.unsaved_lines(None).iter().any(|l| l.contains(what));
        fail_twice(&mut llm, 7);
        assert_eq!(llm.mem_pass.pending.len(), MAX_PENDING + 1, "nothing pushed out");
        assert!(!says(&mut llm, "Lisbon"), "not repeated to its own chat");
        llm.new_chat();
        assert!(says(&mut llm, "Lisbon"), "told to the others");
        // Edited over it: back to the chat, which reads it as it is now.
        let c = llm.chats.iter_mut().find(|c| c.id == 7).unwrap();
        c.msgs[0].text = "My sister lives in Porto.".into();
        c.memory_synced = 0;
        llm.books().reclaim();
        assert_eq!(llm.mem_pass.pending.len(), MAX_PENDING);
        assert!(!says(&mut llm, "Lisbon") && says(&mut llm, "Porto"));
        // Deleted: it goes too, never to be read.
        fail_twice(&mut llm, 7);
        assert_eq!(llm.mem_pass.pending.len(), MAX_PENDING + 1);
        let i = llm.chats.iter().position(|c| c.id == 7).unwrap();
        llm.delete_chat(i);
        llm.books().reclaim();
        assert_eq!(llm.mem_pass.pending.len(), MAX_PENDING);
        assert!(!says(&mut llm, "Porto"));
        // Deleted while a part after it is being read: it goes all the same,
        // and the part being read keeps its place.
        llm.chats.push(chat(8, &[(true, "I play the cello.", true), (false, "Lovely!", true)]));
        fail_twice(&mut llm, 8);
        llm.mem_pass.pending.push(PendingChat { lines: vec!["User: the last one".into()], ..Default::default() });
        let reading = llm.mem_pass.pending.len() - 1;
        let _job = job(&mut llm.mem_pass, &llm.memory, Target::Pending(reading));
        let i = llm.chats.iter().position(|c| c.id == 8).unwrap();
        llm.delete_chat(i);
        llm.books().reclaim();
        let now_at = reading - 1;
        assert_eq!(llm.mem_pass.job.as_ref().map(|j| j.target), Some(Target::Pending(now_at)));
        assert_eq!(llm.mem_pass.pending[now_at].lines[0], "User: the last one");
        assert!(!says(&mut llm, "cello"));
        // Deleted, and the app closed before it was noticed: not kept.
        llm.chats.push(chat(9, &[(true, "I live in Oslo.", true), (false, "Nice!", true)]));
        fail_twice(&mut llm, 9);
        llm.chats.retain(|c| c.id != 9);
        llm.save_memory_for_later();
        assert!(!llm.memory.load_pending().iter().any(|p| p.lines.iter().any(|l| l.contains("Oslo"))));
    }

    /// A part set aside goes again in the next run; one that failed in run
    /// after run is left out, and AI Settings says so.
    #[test]
    fn a_part_that_always_fails_is_left_out_in_the_end() {
        let _root = memory::scratch_root("give_up");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let part = PendingChat { lines: vec!["User: Call me Sam.".into()], failures: GIVE_UP - 2, ..Default::default() };
        pass.pending = vec![part];
        let mut chats: Vec<Chat> = Vec::new();
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: None };
        for (left, retry) in [(1, Retry::Soon), (0, Retry::GaveUp)] {
            books.pass.retry_at = None;
            let (tx, _) = job(books.pass, books.memory, Target::Pending(0));
            tx.send(Msg::Token("Sorry, I can't do that.".into())).unwrap();
            tx.send(Msg::Done(Ok(()))).unwrap();
            books.collect();
            assert_eq!(books.pass.pending.len(), left);
            assert!(matches!(&books.pass.last, Some(LastPass::Failed { retry: r, .. }) if *r == retry));
        }
        assert!(books.memory.load_pending().is_empty(), "gone from disk too");
        // Its count of failures is kept between runs.
        let kept = PendingChat { lines: vec!["User: hi".into()], failures: 3, tries: 1, ..Default::default() };
        books.memory.save_pending(std::slice::from_ref(&kept));
        assert_eq!(books.memory.load_pending(), vec![PendingChat { tries: 0, ..kept }]);
    }

    /// Save now loading the model: AI Settings shows how far it is, and a
    /// question that stops the pass sees the rest of the load in the chat.
    #[test]
    fn the_load_shows_its_progress() {
        let _root = memory::scratch_root("load_progress");
        let mut llm = LlmState::default();
        llm.set_memory_enabled(true);
        let (tx, _) = job(&mut llm.mem_pass, &llm.memory, Target::Chat { id: 1, upto: 2 });
        llm.mem_pass.job.as_mut().unwrap().loading = true;
        tx.send(Msg::Status("Loading the model (first time takes a minute)…".into())).unwrap();
        llm.books().collect();
        assert_eq!(llm.memory_loading(), Some("Loading the model (first time takes a minute)…"));
        // A question comes: the pass stops, the load goes on for it — and the
        // chat shows where it is at once, then how it goes on.
        llm.running = true;
        llm.status = "Starting…".into();
        llm.cancel_pass();
        assert!(llm.mem_pass.job.is_none() && llm.mem_pass.loader.is_some());
        assert_eq!(llm.status, "Loading the model (first time takes a minute)…");
        tx.send(Msg::Status("Loading the vision projector…".into())).unwrap();
        llm.forward_load_progress();
        assert_eq!(llm.status, "Loading the vision projector…");
        // Loaded: the question's own progress takes over.
        tx.send(Msg::Status("Ready".into())).unwrap();
        llm.forward_load_progress();
        assert!(llm.mem_pass.loader.is_none());
        llm.status = "Thinking…".into();
        llm.forward_load_progress();
        assert_eq!(llm.status, "Thinking…");
        // The question's own progress came first (another model loading for
        // it): the old load says nothing more.
        let (tx, _) = job(&mut llm.mem_pass, &llm.memory, Target::Chat { id: 1, upto: 2 });
        llm.mem_pass.job.as_mut().unwrap().loading = true;
        llm.status = "Starting…".into();
        llm.cancel_pass();
        llm.status = "Loading the model (first time takes a minute)…".into();
        tx.send(Msg::Status("Loading the vision projector…".into())).unwrap();
        llm.forward_load_progress();
        assert_eq!(llm.status, "Loading the model (first time takes a minute)…");
        assert!(llm.mem_pass.loader.is_none());
        // A pass that loads nothing shows no progress.
        let (tx, _) = job(&mut llm.mem_pass, &llm.memory, Target::Chat { id: 1, upto: 2 });
        tx.send(Msg::Status("Thinking…".into())).unwrap();
        llm.books().collect();
        assert_eq!(llm.memory_loading(), None);
        llm.cancel_pass();
        assert!(llm.mem_pass.loader.is_none());
    }

    /// A stand-in for the loaded worker: what's queued on it arrives on the
    /// receiver. Gone again when dropped.
    struct FakeWorker(std::sync::mpsc::Receiver<Cmd>);

    impl FakeWorker {
        fn install(model: super::super::GemmaModel, detail: super::super::ImageDetail) -> FakeWorker {
            let (tx, rx) = std::sync::mpsc::channel();
            *super::super::SHARED.lock().unwrap() = Some(super::super::SharedWorker { tx, model, detail, thread: None });
            FakeWorker(rx)
        }
    }

    impl Drop for FakeWorker {
        fn drop(&mut self) {
            *super::super::SHARED.lock().unwrap() = None;
        }
    }

    /// A whole test session, with a stand-in worker: what the user tells the
    /// AI isn't saved while they're still chatting — but a new chat knows it
    /// at once, and Save now sends the pass straight away. The stand-in
    /// answers the way models slip (a name in unescaped quotes), and the
    /// memories and the AI's notes are written, on disk too.
    #[test]
    fn a_test_session_saves_its_memories() {
        let _root = memory::scratch_root("session");
        let mut llm = LlmState::default();
        llm.roleplay.enabled = false;
        llm.set_memory_enabled(true);
        llm.set_ai_chat_on(true);
        let worker = FakeWorker::install(llm.model, llm.image_detail);
        let ctx = egui::Context::default();
        let id = llm.chats[0].id;
        llm.chats[0].msgs.push(msg(true, "Your name is Nova now. I love cats, sushi and hiking.", true));
        llm.chats[0].msgs.push(msg(false, "Nova it is! Cats, sushi and hiking — lovely.", true));
        llm.chat_changed(id);
        llm.tick_memory(&ctx);
        assert!(worker.0.try_recv().is_err(), "not while the user is still chatting");
        assert_eq!(llm.memory_activity(), MemoryActivity::Waiting(1));
        // A new chat knows anyway.
        llm.new_chat();
        let note = llm.memory_note().unwrap();
        assert!(note[0].text.contains("Your name is Nova now."));
        // Save now: the pass goes at once.
        llm.save_memory_now();
        llm.tick_memory(&ctx);
        let Ok(Cmd::Pass { lines, reply, .. }) = worker.0.try_recv() else { panic!("no pass was queued") };
        assert_eq!(lines[0], "User: Your name is Nova now. I love cats, sushi and hiking.");
        assert_eq!(llm.memory_activity(), MemoryActivity::Updating);
        let answer = "{\"summary\": \"The user named me \"Nova\" and told me what they love.\", \
                      \"persona\": \"My name is \"Nova\", a name the user gave me.\", \"remember\": [\
                      {\"topic\": \"Likes\", \"title\": \"Loves cats\", \"text\": \"The user loves cats.\"}, \
                      {\"topic\": \"Likes\", \"title\": \"Loves sushi\", \"text\": \"The user loves sushi.\"}, \
                      {\"topic\": \"Hobbies\", \"title\": \"Hikes\", \"text\": \"The user goes hiking.\"}], \
                      \"forget\": []}";
        reply.send(Msg::Token(answer.into())).unwrap();
        reply.send(Msg::Done(Ok(()))).unwrap();
        llm.tick_memory(&ctx);
        assert_eq!(llm.memory.count(), 3);
        assert_eq!(llm.memory.header.persona, "My name is \"Nova\", a name the user gave me.");
        assert!(matches!(llm.memory_last(), Some(LastPass::Saved { added: 3, notes: true, summary: true, .. })));
        assert_eq!(llm.chats[0].memory_synced, 2);
        llm.tick_memory(&ctx);
        assert!(!llm.saving_memory_now(), "nothing left to save");
        assert_eq!(llm.memory_activity(), MemoryActivity::UpToDate);
        let on_disk = MemoryStore::load(MemoryKind::Main);
        assert_eq!((on_disk.count(), on_disk.header.persona.contains("Nova")), (3, true));
        // And the new chat now knows it from memory.
        let note = llm.memory_note().unwrap();
        assert!(note[0].text.contains("My name is \"Nova\"") && note[0].text.contains("Loves sushi"), "{}", note[0].text);
        assert!(!note[0].text.contains("not sorted into your memory yet"));
        drop(worker);
    }

    /// A part whose pass failed twice is set aside, never dropped: the rest
    /// of the chat goes on without it, it's kept when the app closes, and
    /// the next run tries it first.
    #[test]
    fn a_failed_part_is_kept_for_the_next_run() {
        let _root = memory::scratch_root("repro_failed");
        let mut memory = MemoryStore::load(MemoryKind::Main);
        let mut pass = PassState { enabled: Some(true), pending_loaded: true, ..Default::default() };
        let mut chats = vec![chat(7, &[(true, "Your name is Nova. I love sushi.", true), (false, "Got it!", true)])];
        let target = Target::Chat { id: 7, upto: 2 };
        let mut books = Books { pass: &mut pass, memory: &mut memory, chats: &mut chats, active: None };
        for _ in 0..2 {
            let (tx, _) = job(books.pass, books.memory, target);
            tx.send(Msg::Token("Sorry, I can't do that.".into())).unwrap();
            tx.send(Msg::Done(Ok(()))).unwrap();
            books.collect();
        }
        assert_eq!(books.pass.pending.len(), 1, "set aside");
        assert_eq!(books.pass.pending[0].lines[0], "User: Your name is Nova. I love sushi.");
        // What's said next in that chat is read without waiting for it.
        books.chats[0].msgs.push(msg(true, "I also like ramen.", true));
        books.chats[0].msgs.push(msg(false, "Noted!", true));
        let n = books.next(Instant::now(), true, &HERE).expect("the rest of the chat");
        assert_eq!((n.target, n.lines[0].as_str()), (Target::Chat { id: 7, upto: 4 }, "User: I also like ramen."));
        // Closing the app keeps it, with the unread rest; the next run tries
        // it first.
        books.keep_for_later(None);
        let kept = books.memory.load_pending();
        assert_eq!(kept.len(), 2);
        assert_eq!(
            (kept[0].lines[0].as_str(), kept[0].tries, kept[0].failures),
            ("User: Your name is Nova. I love sushi.", 0, MAX_TRIES)
        );
        books.pass.pending = kept;
        let n = books.next(Instant::now(), true, &HERE).expect("tried again");
        assert_eq!(n.target, Target::Pending(0));
        assert!(n.older && !n.extends_summary, "told it's older than what's remembered");
    }

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
        books.finish(Target::Chat { id: 7, upto: 2 }, &out, &pictures, &[]);
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
    /// longer, is tried once more, then set aside for the next run; a host
    /// without its model loaded gets no more passes until the chat is used
    /// again.
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
        // Fails twice: a longer wait each time; then the part is set aside
        // for the next run (or Save now) — never dropped — and AI Settings
        // says why.
        for (tries, retry) in [(1, Retry::Soon), (0, Retry::NextRun)] {
            let (tx, _) = job(books.pass, books.memory, target);
            tx.send(Msg::Token("Sorry, I can't do that.".into())).unwrap();
            tx.send(Msg::Done(Ok(()))).unwrap();
            books.collect();
            assert_eq!(books.chats[0].memory_tries, tries);
            assert!(matches!(&books.pass.last, Some(LastPass::Failed { why, retry: r, .. })
                if why.contains("had no memory update") && *r == retry));
            let wait = books.pass.retry_at.expect("waits").saturating_duration_since(Instant::now());
            assert!(wait > RETRY_AFTER_STOP);
        }
        assert_eq!(books.chats[0].memory_synced, 2, "the chat goes on");
        assert_eq!(books.pass.pending.len(), 1, "what it read is set aside, not dropped");
        assert_eq!(books.pass.pending[0].lines, vec!["User: My sister Ana lives in Lisbon", "You: Lovely!"]);
        assert_eq!((books.pass.pending[0].tries, books.pass.pending[0].failures), (MAX_TRIES, MAX_TRIES));
        assert_eq!(books.pass.pending[0].chat, Some((7, 0..2)));
        assert!(books.pass.pending[0].aside && !books.pass.pending[0].extends_summary);
        let on_disk = books.pass.pending.iter().map(|p| PendingChat { tries: 0, chat: None, ..p.clone() });
        assert_eq!(books.memory.load_pending(), on_disk.collect::<Vec<_>>());
        let later = Instant::now() + Duration::from_secs(3600);
        assert!(books.next(later, true, &HERE).is_err(), "not tried again this run");
        assert!(!can_run_any(books.pass, books.chats, &HERE), "nothing holds the model for it");
        assert_eq!(waiting(books.pass, books.chats, None), 1, "still waiting");
        books.pass.pending.clear();
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
