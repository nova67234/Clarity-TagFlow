//! Local AI model (Settings → AI Model) — Google's Gemma 4 vision model,
//! running fully inside the app via llama.cpp (the `llama-cpp-2` bindings with
//! the `mtmd` multimodal pipeline). No server and no account: the one-click
//! setup downloads two GGUF files (the E4B instruct weights + the vision
//! projector) from an ungated HuggingFace mirror into the shared models dir
//! (`ai_models` catalog entry, same downloader as the taggers).
//!
//! Inference runs on a dedicated background thread that owns the loaded model
//! and streams generated tokens back over an mpsc channel — the same
//! worker-thread-plus-poll shape as the FTP tester and Pixal3D runner. The
//! model loads lazily on the first prompt (it's ~5 GB of weights) and then
//! stays resident until the app exits.
//!
//! Built without the `llm` cargo feature, everything still compiles — the tab
//! just reports that this build has no AI support.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

use eframe::egui;

/// Model folder under the shared models root (`tagger::models_root()`).
pub const FOLDER: &str = "gemma-4";
/// The instruct weights (Q4_K_M quant, ~5 GB).
pub const MODEL_FILE: &str = "gemma-4-E4B-it-Q4_K_M.gguf";
/// The multimodal (vision) projector llama.cpp uses to encode images (~1 GB).
pub const MMPROJ_FILE: &str = "mmproj-F16.gguf";

// The bigger Gemma 4 variants (also vision-capable; same mmproj file name).
pub const FOLDER_26B: &str = "gemma-4-26b";
pub const MODEL_FILE_26B: &str = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf";
pub const FOLDER_31B: &str = "gemma-4-31b";
pub const MODEL_FILE_31B: &str = "gemma-4-31B-it-Q4_K_M.gguf";

/// Which Gemma 4 variant the chat runs (Settings → AI Model). All are
/// vision-capable; they trade download size + speed for quality.
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GemmaModel {
    /// E4B instruct — ~6 GB, fast, fits fully on 8 GB+ GPUs. The default.
    #[default]
    E4B,
    /// 26B A4B (mixture-of-experts) — ~18 GB, much smarter.
    A26B,
    /// 31B dense — ~19.5 GB, the strongest Gemma 4.
    D31B,
}

impl GemmaModel {
    pub const ALL: [GemmaModel; 3] = [GemmaModel::E4B, GemmaModel::A26B, GemmaModel::D31B];

    pub fn folder(self) -> &'static str {
        match self {
            GemmaModel::E4B => FOLDER,
            GemmaModel::A26B => FOLDER_26B,
            GemmaModel::D31B => FOLDER_31B,
        }
    }

    pub fn model_file(self) -> &'static str {
        match self {
            GemmaModel::E4B => MODEL_FILE,
            GemmaModel::A26B => MODEL_FILE_26B,
            GemmaModel::D31B => MODEL_FILE_31B,
        }
    }

    pub fn mmproj_file(self) -> &'static str {
        MMPROJ_FILE
    }

    pub fn label(self) -> &'static str {
        match self {
            GemmaModel::E4B => "Gemma 4 E4B",
            GemmaModel::A26B => "Gemma 4 26B A4B",
            GemmaModel::D31B => "Gemma 4 31B",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            GemmaModel::E4B => "~6 GB · fast, fits fully on 8 GB+ GPUs — recommended",
            GemmaModel::A26B => "~18 GB · much smarter (MoE); wants ~20 GB VRAM, else spills to RAM (slower)",
            GemmaModel::D31B => "~19.5 GB · strongest; on a 16 GB GPU it runs partly on the CPU (slower)",
        }
    }

    pub fn installed(self) -> bool {
        crate::tagger::resolve(self.folder(), self.model_file()).is_some()
            && crate::tagger::resolve(self.folder(), self.mmproj_file()).is_some()
    }
}

/// True when this binary was compiled with local-AI support.
pub const BUILT_WITH_LLM: bool = cfg!(feature = "llm");

/// How many of a chat's most recent messages are re-sent to the model each
/// turn (the working memory — older messages are forgotten). Sized so ~25
/// turns of conversation fit the 16k context with room for a long reply.
const HISTORY_MSGS: usize = 50;
/// How many of the newest attached images ride along with the history.
/// Images are heavy (256 tokens each, re-encoded every turn), so older ones
/// are dropped from the prompt — their text, and the model's own description
/// of them, stay in the history.
const HISTORY_IMAGES: usize = 4;

/// True when this binary was compiled with the Vulkan GPU backend
/// (`llm-vulkan`, built via scripts\build-vulkan.cmd). CPU-only otherwise.
pub const BUILT_WITH_GPU: bool = cfg!(feature = "llm-vulkan");

/// GPU builds delay-load vulkan-1.dll (see build.rs) so the app launches on
/// machines without it — but calling into llama.cpp there would crash on the
/// first Vulkan call, so probe for the DLL first. It ships with every GPU
/// driver; only GPU-less VMs and the like lack it.
#[cfg(all(feature = "llm-vulkan", target_os = "windows"))]
fn vulkan_runtime_present() -> bool {
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut core::ffi::c_void;
    }
    let name: Vec<u16> = "vulkan-1.dll\0".encode_utf16().collect();
    !unsafe { LoadLibraryW(name.as_ptr()) }.is_null()
}

/// True when both GGUF files are present in any of the model roots.
pub fn installed() -> bool {
    crate::tagger::resolve(FOLDER, MODEL_FILE).is_some()
        && crate::tagger::resolve(FOLDER, MMPROJ_FILE).is_some()
}

/// One message of the conversation snapshot handed to the worker.
struct CmdMsg {
    user: bool,
    text: String,
    image: Option<PathBuf>,
}

/// A request to the inference worker. The full (capped) history is sent every
/// turn — the worker rebuilds the model context from scratch, so multi-turn
/// chat works without keeping KV-cache state between requests.
enum Cmd {
    Generate { msgs: Vec<CmdMsg> },
}

/// A message streamed back from the inference worker.
enum Msg {
    /// Progress line ("Loading model…", "Reading the image…", "Thinking…").
    Status(String),
    /// One decoded piece of the response text.
    Token(String),
    /// Generation finished (or failed). The worker stays alive for more
    /// prompts unless the error was fatal (model failed to load).
    Done(Result<(), String>),
}

struct Worker {
    tx: Sender<Cmd>,
    rx: Receiver<Msg>,
}

/// Who authored a chat message.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Model,
}

/// One message in an AI Chat conversation.
pub struct ChatMsg {
    pub role: ChatRole,
    pub text: String,
    /// Image attached to a user message (the vision input).
    pub image: Option<PathBuf>,
}

/// One conversation in the AI Chat view (the left strip switches between them).
pub struct Chat {
    pub id: u64,
    pub msgs: Vec<ChatMsg>,
}

impl Chat {
    /// Sidebar label: the first user message, trimmed — or "New chat".
    pub fn title(&self) -> String {
        let first = self.msgs.iter().find(|m| m.role == ChatRole::User);
        match first {
            Some(m) if !m.text.trim().is_empty() => {
                let t = m.text.trim();
                let cut: String = t.chars().take(22).collect();
                if cut.len() < t.len() { format!("{cut}…") } else { cut }
            }
            Some(_) => "Image".to_string(),
            None => "New chat".to_string(),
        }
    }
}

/// State for the AI Model tab and the AI Chat view, owned by `ViewerApp`
/// (like `FtpState`).
pub struct LlmState {
    /// Both model files present on disk (refreshed when a download finishes).
    pub installed: bool,
    /// In-flight setup download, polled each frame.
    pub download: Option<crate::ai_models::DownloadHandle>,
    pub download_err: Option<String>,

    // --- AI Chat view (src/ai_chat.rs) ---
    pub chats: Vec<Chat>,
    pub active_chat: usize,
    /// The chat receiving the streaming reply, by id (survives chat
    /// switching/deletion while a reply streams in).
    gen_chat: Option<u64>,
    next_chat_id: u64,
    /// The input pill's draft text and attached image.
    pub draft: String,
    pub draft_image: Option<PathBuf>,
    /// Cached thumbnail texture for `draft_image` (path it was made from +
    /// the GPU texture), shown inside the input card.
    pub draft_thumb: Option<(PathBuf, egui::TextureHandle)>,
    /// Cached display textures for images attached to sent messages, keyed by
    /// path (`None` = unreadable file, cached so it isn't retried per frame).
    pub msg_thumbs: std::collections::HashMap<PathBuf, Option<egui::TextureHandle>>,
    /// The input card's on-screen rect + the `ctx.input(time)` it was last
    /// drawn, published every frame the chat is visible. Drag-and-drop uses it
    /// both for the hover highlight and to claim drops away from the gallery
    /// (same freshness scheme as the generator's prompt box).
    pub input_rect: Option<(egui::Rect, f64)>,

    pub run_err: Option<String>,
    /// A generation is in flight.
    pub running: bool,
    /// Latest worker status line, shown next to the spinner.
    pub status: String,

    /// Layout cache for the chat's markdown rendering (src/ai_chat.rs).
    pub md_cache: egui_commonmark::CommonMarkCache,

    /// Retry requested from inside the message list this frame — applied
    /// after the list finishes drawing (mutating mid-iteration would break
    /// the loop's indices).
    pub pending_retry: Option<usize>,
    /// The running text-to-speech process for "Listen" (Windows speech
    /// synthesis in a spawned PowerShell; killed to stop playback). The
    /// fallback voice — OmniVoice below is used when installed.
    tts: Option<std::process::Child>,
    /// The OmniVoice neural voice (Python sidecar; see src/voice.rs).
    pub voice: crate::voice::VoiceState,
    /// Role playing: persona, names and the shared memory diary
    /// (src/roleplay.rs; toggled from the input card's tools menu).
    pub roleplay: crate::roleplay::RoleplayState,
    /// The Gemma 4 variant to chat with (mirrors the persisted setting).
    pub model: GemmaModel,
    /// The variant the resident worker actually loaded — a mismatch drops
    /// the worker so the next ask loads the newly selected model.
    worker_model: GemmaModel,
    /// Auto-speak finished replies (mirrors the persisted setting; synced
    /// each frame by main.rs, toggled from the tools menu).
    pub auto_speak: bool,
    /// The input card's tools popup is open.
    pub tools_open: bool,

    worker: Option<Worker>,
}

impl Drop for LlmState {
    fn drop(&mut self) {
        // Don't leave a voice speaking after the app closes.
        if let Some(mut c) = self.tts.take() {
            let _ = c.kill();
        }
    }
}

impl Default for LlmState {
    fn default() -> Self {
        Self {
            installed: installed(),
            download: None,
            download_err: None,
            chats: vec![Chat { id: 0, msgs: Vec::new() }],
            active_chat: 0,
            gen_chat: None,
            next_chat_id: 1,
            draft: String::new(),
            draft_image: None,
            draft_thumb: None,
            msg_thumbs: std::collections::HashMap::new(),
            input_rect: None,
            run_err: None,
            running: false,
            status: String::new(),
            md_cache: egui_commonmark::CommonMarkCache::default(),
            pending_retry: None,
            tts: None,
            voice: crate::voice::VoiceState::default(),
            roleplay: crate::roleplay::RoleplayState::load(),
            model: GemmaModel::default(),
            worker_model: GemmaModel::default(),
            auto_speak: false,
            tools_open: false,
            worker: None,
        }
    }
}

impl LlmState {
    /// The "Set up everything" button: download both GGUFs on a background
    /// thread via the shared model downloader.
    pub fn start_setup(&mut self) {
        if self.download.is_some() {
            return;
        }
        self.download_err = None;
        self.download = crate::ai_models::start_model_download(self.model.folder());
        if self.download.is_none() {
            self.download_err = Some("Model missing from the catalog".to_string());
        }
    }

    /// Switch the Gemma variant the chat uses. Refreshes the installed flag
    /// and, when idle, drops the resident worker so the next ask loads the
    /// new weights (and the old ones free their memory).
    pub fn set_model(&mut self, model: GemmaModel) {
        if self.model == model {
            return;
        }
        self.model = model;
        self.installed = model.installed();
        if !self.running {
            self.worker = None;
        }
    }

    /// Drive background work — poll the setup download and drain any streamed
    /// tokens from the inference worker. Call once per frame from the tab.
    pub fn poll(&mut self, ctx: &egui::Context) {
        self.voice.poll();
        if let Some(dl) = &self.download {
            if dl.done() {
                if dl.ok() {
                    self.installed = self.model.installed();
                } else {
                    self.download_err =
                        Some(dl.error().unwrap_or_else(|| "Download failed".to_string()));
                }
                self.download = None;
            }
        }

        // Drain first, then act — acting (auto-speak) needs &mut self.
        let mut worker_msgs = Vec::new();
        let mut worker_died = false;
        if let Some(w) = &self.worker {
            loop {
                match w.rx.try_recv() {
                    Ok(m) => worker_msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        worker_died = true;
                        break;
                    }
                }
            }
        }
        {
            for msg in worker_msgs {
                match msg {
                    Msg::Status(s) => self.status = s,
                    Msg::Token(t) => {
                        // Stream into the reply bubble of the chat that asked,
                        // even if the user switched to another chat meanwhile.
                        if let Some(id) = self.gen_chat {
                            if let Some(m) = self
                                .chats
                                .iter_mut()
                                .find(|c| c.id == id)
                                .and_then(|c| c.msgs.last_mut())
                            {
                                m.text.push_str(&t);
                            }
                        }
                    }
                    Msg::Done(res) => {
                        // Role play: a finished reply may end in control lines
                        // (diary entries, keep/show a picture) — pull them out
                        // of the visible text and act on them.
                        if res.is_ok() && self.roleplay.enabled {
                            let mut directives = None;
                            let mut last_user_image = None;
                            if let Some(chat) = self
                                .gen_chat
                                .and_then(|id| self.chats.iter_mut().find(|c| c.id == id))
                            {
                                last_user_image = chat
                                    .msgs
                                    .iter()
                                    .rev()
                                    .find(|m| m.role == ChatRole::User && m.image.is_some())
                                    .and_then(|m| m.image.clone());
                                if let Some(m) = chat.msgs.last_mut() {
                                    directives = Some(crate::roleplay::extract_directives(&mut m.text));
                                }
                            }
                            if let Some(d) = directives {
                                for mem in d.memories {
                                    self.roleplay.add_memory(&mem, true);
                                }
                                // "I love this picture" → encrypted album copy
                                // + a diary entry that remembers why.
                                if let (Some(why), Some(src)) = (d.keep_image, last_user_image) {
                                    match self.roleplay.save_album_image(&src, &why) {
                                        Ok(name) => {
                                            self.roleplay.add_memory_with_image(&why, true, Some(name));
                                        }
                                        Err(e) => eprintln!("[roleplay] album save failed: {e}"),
                                    }
                                }
                                // "Look at this again!" → decrypt to temp and
                                // attach to the reply so it shows in chat.
                                if let Some(name) = d.show_image {
                                    let img = self.roleplay.load_album_image(&name);
                                    if let Some(m) = self
                                        .gen_chat
                                        .and_then(|id| self.chats.iter_mut().find(|c| c.id == id))
                                        .and_then(|c| c.msgs.last_mut())
                                    {
                                        m.image = img;
                                    }
                                }
                            }
                        }
                        // Auto-speak: read the finished reply aloud (after the
                        // diary lines are gone, so they're never spoken).
                        if res.is_ok() && self.auto_speak {
                            let text = self
                                .gen_chat
                                .and_then(|id| self.chats.iter().find(|c| c.id == id))
                                .and_then(|c| c.msgs.last())
                                .map(|m| m.text.clone());
                            if let Some(t) = text {
                                self.start_speaking(&t, ctx);
                            }
                        }
                        self.running = false;
                        self.gen_chat = None;
                        if let Err(e) = res {
                            self.run_err = Some(e);
                        }
                    }
                }
            }
            // A dead worker means the model failed to load (the error already
            // arrived as a Done). Drop it so the next ask can start fresh.
            if worker_died {
                self.worker = None;
                self.running = false;
                self.gen_chat = None;
            }
        }
    }

    /// Start a fresh conversation and switch to it.
    pub fn new_chat(&mut self) {
        // Reuse the current chat if it's still empty (no stacks of blanks).
        if self.chats[self.active_chat].msgs.is_empty() {
            return;
        }
        self.chats.push(Chat { id: self.next_chat_id, msgs: Vec::new() });
        self.next_chat_id += 1;
        self.active_chat = self.chats.len() - 1;
    }

    /// Delete a conversation (a streaming reply into it is silently dropped).
    /// The list never goes empty — deleting the last chat leaves a fresh one.
    pub fn delete_chat(&mut self, index: usize) {
        if index >= self.chats.len() {
            return;
        }
        let removed = self.chats.remove(index);
        if self.gen_chat == Some(removed.id) {
            self.gen_chat = None;
        }
        if self.chats.is_empty() {
            self.chats.push(Chat { id: self.next_chat_id, msgs: Vec::new() });
            self.next_chat_id += 1;
        }
        self.active_chat = self.active_chat.min(self.chats.len() - 1);
    }

    /// Send the input pill's draft (text + attached image) as the next user
    /// message of the active chat, spawning the worker thread on first use.
    /// The reply streams into a fresh model message.
    pub fn send_draft(&mut self, ctx: &egui::Context) {
        let text = self.draft.trim().to_string();
        if self.running || (text.is_empty() && self.draft_image.is_none()) {
            return;
        }
        if !BUILT_WITH_LLM {
            self.run_err =
                Some("This build was compiled without the AI feature (`llm`).".to_string());
            return;
        }
        if !self.installed {
            self.run_err = Some("Set up the model first (Settings → AI Model).".to_string());
            return;
        }
        self.run_err = None;
        self.status = "Starting…".to_string();

        let image = self.draft_image.take();
        self.draft_thumb = None;
        self.draft.clear();
        self.chats[self.active_chat]
            .msgs
            .push(ChatMsg { role: ChatRole::User, text, image });
        self.begin_generation(ctx);
    }

    /// Regenerate the model reply at `msg_index` of the active chat: that
    /// reply — and everything after it — is discarded, and the conversation
    /// up to its user message is resent.
    pub fn retry(&mut self, msg_index: usize, ctx: &egui::Context) {
        if self.running {
            return;
        }
        let chat = &mut self.chats[self.active_chat];
        if chat.msgs.get(msg_index).map(|m| m.role) != Some(ChatRole::Model) {
            return;
        }
        chat.msgs.truncate(msg_index);
        if chat.msgs.last().map(|m| m.role) != Some(ChatRole::User) {
            return;
        }
        self.run_err = None;
        self.status = "Starting…".to_string();
        self.begin_generation(ctx);
    }

    /// Generate a reply to the active chat's history (which must end with a
    /// user message): snapshot the capped history, send it to the worker, and
    /// push the empty reply bubble the tokens stream into.
    fn begin_generation(&mut self, ctx: &egui::Context) {
        let chat = &mut self.chats[self.active_chat];

        // Snapshot the history for the worker, capped to the most recent
        // messages so a very long chat doesn't overflow the model's context,
        // and with only the newest few images attached (see HISTORY_IMAGES).
        let mut msgs: Vec<CmdMsg> = chat
            .msgs
            .iter()
            .rev()
            .take(HISTORY_MSGS)
            .rev()
            .map(|m| CmdMsg {
                user: m.role == ChatRole::User,
                text: m.text.clone(),
                // Only the user's images go through the vision encoder —
                // album pictures the AI showed back are display-only.
                image: if m.role == ChatRole::User { m.image.clone() } else { None },
            })
            .collect();
        let mut kept_images = 0;
        for m in msgs.iter_mut().rev() {
            if m.image.is_some() {
                kept_images += 1;
                if kept_images > HISTORY_IMAGES {
                    m.image = None;
                }
            }
        }
        let id = chat.id;

        // Role play: pin the persona/diary priming turn in front of the
        // history — it never slides out of the window, so the character (and
        // everything in the diary) survives arbitrarily long sessions.
        if self.roleplay.enabled {
            msgs.insert(0, CmdMsg { user: false, text: self.roleplay.ack(), image: None });
            msgs.insert(0, CmdMsg { user: true, text: self.roleplay.preamble(), image: None });
        }

        // A different model was selected since the worker loaded — drop it so
        // the fresh spawn below loads the new weights (the old ones free when
        // the worker thread exits).
        if self.worker.is_some() && self.worker_model != self.model {
            self.worker = None;
        }
        if self.worker.is_none() {
            self.worker = spawn_worker(ctx.clone(), self.model);
            self.worker_model = self.model;
        }
        match &self.worker {
            Some(w) if w.tx.send(Cmd::Generate { msgs }).is_ok() => {
                // The empty reply bubble the tokens stream into.
                self.chats[self.active_chat]
                    .msgs
                    .push(ChatMsg { role: ChatRole::Model, text: String::new(), image: None });
                self.running = true;
                self.gen_chat = Some(id);
            }
            _ => {
                self.worker = None;
                self.run_err = Some("The AI worker stopped — try again.".to_string());
            }
        }
    }

    /// True while a "Listen" playback is speaking.
    pub fn speaking(&mut self) -> bool {
        match &mut self.tts {
            Some(c) => match c.try_wait() {
                Ok(None) => true,
                _ => {
                    self.tts = None;
                    false
                }
            },
            None => false,
        }
    }

    /// True while either voice backend is playing.
    pub fn any_speaking(&mut self) -> bool {
        self.voice.speaking() || self.speaking()
    }

    /// Read a reply aloud — or stop the current playback if one is speaking.
    /// Uses OmniVoice (the natural neural voice, Python sidecar) once its
    /// one-click setup has run; otherwise falls back to the OS speech
    /// synthesizer (Windows).
    pub fn listen(&mut self, text: &str, ctx: &egui::Context) {
        if self.voice.speaking() {
            self.voice.stop();
            return;
        }
        if self.speaking() {
            if let Some(mut c) = self.tts.take() {
                let _ = c.kill();
            }
            return;
        }
        self.start_speaking(text, ctx);
    }

    /// Start speaking `text`, replacing whatever was playing (used by the
    /// Listen buttons and by auto-speak).
    fn start_speaking(&mut self, text: &str, ctx: &egui::Context) {
        let spoken = speech_text(text);
        if spoken.trim().is_empty() {
            return;
        }
        // Newest reply wins — never overlap two voices.
        self.voice.stop();
        if let Some(mut c) = self.tts.take() {
            let _ = c.kill();
        }
        if self.voice.installed {
            self.voice.speak(&spoken, ctx);
            return;
        }
        #[cfg(target_os = "windows")]
        {
            use std::io::Write as _;
            use std::os::windows::process::CommandExt as _;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            let child = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Add-Type -AssemblyName System.Speech; \
                     $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                     $s.Speak([Console]::In.ReadToEnd())",
                ])
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .creation_flags(CREATE_NO_WINDOW)
                .spawn();
            match child {
                Ok(mut c) => {
                    if let Some(mut stdin) = c.stdin.take() {
                        let _ = stdin.write_all(spoken.as_bytes());
                    }
                    self.tts = Some(c);
                }
                Err(e) => self.run_err = Some(format!("Couldn't start speech: {e}")),
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.run_err = Some("Listen isn't supported on this platform yet.".to_string());
        }
    }
}

/// True for codepoints a TTS shouldn't try to pronounce: emoji, pictographs,
/// dingbats (✨), symbols, and the invisible emoji plumbing (ZWJ, variation
/// selectors). Letters of real scripts (Latin, CJK, Cyrillic, …) all pass.
fn is_unspeakable(c: char) -> bool {
    matches!(
        c as u32,
        0x1F000..=0x1FAFF   // emoji, pictographs, flags, symbols-extended
        | 0x2600..=0x27BF   // misc symbols + dingbats (✨ ❤ ☀ …)
        | 0x2B00..=0x2BFF   // arrows / stars (⭐ …)
        | 0xFE0E..=0xFE0F   // variation selectors
        | 0x200D            // zero-width joiner
    )
}

/// Strip markdown down to speakable text: code blocks are skipped ("code
/// omitted"), inline markers (** * ` #) are dropped, and emoji/symbols are
/// removed — voices either mispronounce them or, in OmniVoice's frontend,
/// can outright fail on them.
fn speech_text(md: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            if !in_code {
                out.push_str("Code omitted. ");
            }
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let line = line.trim_start_matches('#').trim();
        let cleaned: String = line
            .chars()
            .filter(|c| !matches!(c, '*' | '`') && !is_unspeakable(*c))
            .collect();
        let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        if !cleaned.is_empty() {
            out.push_str(&cleaned);
            out.push('\n');
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Inference worker — everything that touches llama.cpp lives behind the `llm`
// feature so a --no-default-features build needs no cmake/C++ toolchain.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "llm"))]
fn spawn_worker(_ctx: egui::Context, _model: GemmaModel) -> Option<Worker> {
    None
}

#[cfg(feature = "llm")]
fn spawn_worker(ctx: egui::Context, model: GemmaModel) -> Option<Worker> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<Msg>();
    std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, model));
    Some(Worker { tx: cmd_tx, rx: msg_rx })
}

#[cfg(feature = "llm")]
mod worker {
    use std::ffi::CString;
    use std::num::NonZeroU32;
    use std::path::Path;
    use std::sync::mpsc::{Receiver, Sender};

    use eframe::egui;
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
    use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
    use llama_cpp_2::sampling::LlamaSampler;

    use super::{Cmd, Msg};

    /// Context window. Gemma 4 supports far more, but 16k keeps the KV cache
    /// reasonable on ordinary machines while fitting the 50-message history
    /// cap plus a long reply.
    const N_CTX: u32 = 16384;
    /// Prompt-eval batch size; must fit an image's token chunk (256 for Gemma).
    const N_BATCH: u32 = 512;
    /// Cap on generated tokens per response. 1024 turned out to truncate real
    /// answers mid-code-block; 3072 fits long code examples comfortably.
    const MAX_TOKENS: usize = 3072;

    /// The context configuration used for every generation — also created
    /// once as a probe during loading, so the GPU-offload ladder knows the
    /// KV cache actually fits alongside the weights.
    fn context_params(threads: i32) -> LlamaContextParams {
        LlamaContextParams::default()
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_n_batch(N_BATCH)
            .with_n_ctx(NonZeroU32::new(N_CTX))
    }

    /// Worker body: load everything once, then serve Generate commands until
    /// the UI side drops its sender. On a load failure the error is reported
    /// as a `Done(Err)` and the thread exits (the UI respawns it on demand).
    pub(super) fn run(rx: Receiver<Cmd>, tx: Sender<Msg>, ctx: egui::Context, model: super::GemmaModel) {
        let send = |m: Msg| {
            let _ = tx.send(m);
            ctx.request_repaint();
        };
        match load_and_serve(&rx, &send, model) {
            Ok(()) => {}
            Err(e) => send(Msg::Done(Err(e))),
        }
    }

    fn load_and_serve(
        rx: &Receiver<Cmd>,
        send: &dyn Fn(Msg),
        which: super::GemmaModel,
    ) -> Result<(), String> {
        // GPU build on a machine with no Vulkan runtime: fail readably before
        // llama.cpp's first (delay-loaded, would-crash) Vulkan call.
        #[cfg(all(feature = "llm-vulkan", target_os = "windows"))]
        if !super::vulkan_runtime_present() {
            return Err(
                "This computer has no Vulkan runtime (vulkan-1.dll), which this \
                 GPU build of the AI needs — it normally comes with the graphics \
                 driver. Update the GPU driver, or use a CPU build."
                    .to_string(),
            );
        }

        let model_path = crate::tagger::resolve(which.folder(), which.model_file())
            .ok_or_else(|| "Model file not found — run setup first".to_string())?;
        let mmproj_path = crate::tagger::resolve(which.folder(), which.mmproj_file())
            .ok_or_else(|| "Vision projector not found — run setup first".to_string())?;

        // Leave one core for the UI; llama.cpp saturates the rest.
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() as i32 - 1).max(1))
            .unwrap_or(4);

        send(Msg::Status("Loading the model (first time takes a minute)…".into()));
        let backend = LlamaBackend::init().map_err(|e| format!("llama.cpp init: {e}"))?;
        // Offload to the GPU when a backend is compiled in (`llm-vulkan`) — a
        // ladder of attempts, because the big variants (26B/31B) don't fit
        // whole in common VRAM sizes: everything → about half → CPU only.
        // Each rung must survive the WHOLE pipeline — weights, vision
        // projector, and a probe context — because weights fitting is not
        // enough: the 26B loads into ~20 GB of VRAM and llama.cpp then
        // returns a null context when the KV cache no longer fits.
        let mut loaded = None;
        let mut last_err = String::new();
        for (i, layers) in [1_000_000u32, 24, 0].into_iter().enumerate() {
            if i > 0 {
                send(Msg::Status(
                    "Model doesn't fit in GPU memory — loading partly on the CPU…".into(),
                ));
            }
            let params = LlamaModelParams::default().with_n_gpu_layers(layers);
            let model = match LlamaModel::load_from_file(&backend, &model_path, &params) {
                Ok(m) => m,
                Err(e) => {
                    last_err = format!("Load model: {e}");
                    continue;
                }
            };
            send(Msg::Status("Loading the vision projector…".into()));
            let mtmd_params = MtmdContextParams {
                // Like n_gpu_layers: uses the GPU when a backend exists, CPU otherwise.
                use_gpu: true,
                print_timings: false,
                n_threads: threads,
                media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                    .map_err(|e| e.to_string())?,
                image_min_tokens: -1,
                image_max_tokens: -1,
            };
            let mtmd = match MtmdContext::init_from_file(
                &mmproj_path.to_string_lossy(),
                &model,
                &mtmd_params,
            ) {
                Ok(m) => m,
                Err(e) => {
                    last_err = format!("Load vision projector: {e}");
                    continue;
                }
            };
            // Probe context, created and dropped: proves generation will have
            // room for the KV cache with this many layers on the GPU.
            if let Err(e) = model.new_context(&backend, context_params(threads)) {
                last_err = format!("Create context: {e}");
                continue;
            }
            loaded = Some((model, mtmd));
            break;
        }
        let Some((model, mtmd)) = loaded else {
            return Err(last_err);
        };

        let template = model
            .chat_template(None)
            .map_err(|e| format!("Chat template: {e}"))?;

        send(Msg::Status("Ready".into()));

        while let Ok(Cmd::Generate { msgs }) = rx.recv() {
            let res = generate(&backend, &model, &mtmd, &template, threads, &msgs, send);
            send(Msg::Done(res));
        }
        Ok(())
    }

    fn generate(
        backend: &LlamaBackend,
        model: &LlamaModel,
        mtmd: &MtmdContext,
        template: &llama_cpp_2::model::LlamaChatTemplate,
        threads: i32,
        msgs: &[super::CmdMsg],
        send: &dyn Fn(Msg),
    ) -> Result<(), String> {
        // A fresh context per turn: cheap next to the resident weights, and it
        // guarantees no KV-cache state leaks — the whole (capped) history is
        // re-evaluated, which is what makes multi-turn chat work.
        let mut context = model
            .new_context(backend, context_params(threads))
            .map_err(|e| format!("Create context: {e}"))?;

        // The media marker tells the tokenizer where an image's tokens go;
        // bitmaps are matched to markers in order of appearance, so each
        // message with an image gets a marker and its bitmap is pushed in the
        // same sequence (matches llama.cpp's mtmd-cli).
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
        let mut turns: Vec<(bool, String)> = Vec::new();
        for m in msgs {
            let mut text = m.text.clone();
            if let Some(path) = &m.image {
                send(Msg::Status("Reading the image…".into()));
                bitmaps.push(load_bitmap(mtmd, path)?);
                if !text.contains(marker) {
                    text = format!("{marker}\n{text}");
                }
            }
            turns.push((m.user, text));
        }

        send(Msg::Status("Thinking…".into()));
        let chat: Vec<LlamaChatMessage> = turns
            .iter()
            .map(|(user, text)| {
                let role = if *user { "user" } else { "assistant" };
                LlamaChatMessage::new(role.to_string(), text.clone()).map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        // Gemma 4 embeds a huge Jinja chat template (tool calls, thinking
        // channels) that llama.cpp's built-in non-Jinja formatter doesn't
        // recognise (ffi error -1). Its actual wire format is simple, so fall
        // back to formatting the turns by hand; the template path still
        // serves models with conventional templates.
        let formatted = match model.apply_chat_template(template, &chat, true) {
            Ok(s) => s,
            Err(_) => {
                let mut s = String::new();
                for (user, text) in &turns {
                    let role = if *user { "user" } else { "model" };
                    s.push_str(&format!("<|turn>{role}\n{text}<turn|>\n"));
                }
                s.push_str("<|turn>model\n");
                s
            }
        };
        let input = MtmdInputText { text: formatted, add_special: true, parse_special: true };
        let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = mtmd
            .tokenize(input, &bitmap_refs)
            .map_err(|e| format!("Tokenize: {e}"))?;
        let mut n_past = chunks
            .eval_chunks(mtmd, &mut context, 0, 0, N_BATCH as i32, true)
            .map_err(|e| format!("Evaluate prompt: {e}"))?;

        // Gemma's recommended sampling: top-k 64, top-p 0.95, temperature 1.0.
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(64),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(1.0),
            LlamaSampler::dist(1234),
        ]);

        let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
        // Streaming decoder: a token can end mid-way through a multi-byte
        // UTF-8 character (CJK, emoji), so bytes carry over between pieces.
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        for _ in 0..MAX_TOKENS {
            let token = sampler.sample(&context, -1);
            sampler.accept(token);
            if model.is_eog_token(token) {
                break;
            }
            let piece = model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| format!("Decode token: {e}"))?;
            if !piece.is_empty() {
                send(Msg::Token(piece));
            }
            batch.clear();
            batch.add(token, n_past, &[0], true).map_err(|e| e.to_string())?;
            n_past += 1;
            context.decode(&mut batch).map_err(|e| format!("Decode: {e}"))?;
        }
        Ok(())
    }

    /// Load an image for the vision pipeline. llama.cpp's own loader (stb)
    /// covers jpg/png/bmp/gif; anything else (webp, tiff, …) is decoded with
    /// the `image` crate and handed over as a temp PNG.
    fn load_bitmap(mtmd: &MtmdContext, path: &Path) -> Result<MtmdBitmap, String> {
        if let Ok(bmp) = MtmdBitmap::from_file(mtmd, &path.to_string_lossy(), false) {
            return Ok(bmp);
        }
        let img = image::open(path).map_err(|e| format!("Read image: {e}"))?;
        let tmp = std::env::temp_dir().join("clarity_tagflow_llm_input.png");
        img.to_rgb8()
            .save_with_format(&tmp, image::ImageFormat::Png)
            .map_err(|e| format!("Convert image: {e}"))?;
        MtmdBitmap::from_file(mtmd, &tmp.to_string_lossy(), false)
            .map_err(|e| format!("Load image: {e}"))
    }
}

/// Smoke test for the inference worker — drives the real channel + llama.cpp
/// pipeline end-to-end, so it needs model files on disk and is `#[ignore]`d.
/// Any GGUF weights + mmproj pair placed under `tools/gemma-4/` with the
/// expected names works (a tiny SmolVLM does; so does the real Gemma):
///   cargo test llm_worker_smoke -- --ignored --nocapture
#[cfg(all(test, feature = "llm"))]
mod tests {
    use super::*;

    /// Role-play smoke: primes the real worker with the actual persona/diary
    /// preamble and checks the model (a) stays in character addressing the
    /// user by name, (b) records new facts/permissions as MEMORY: lines that
    /// `extract_memories` can pull out, (c) doesn't re-record diary entries.
    /// Needs the model files, like the other ignored tests.
    #[test]
    #[ignore]
    fn llm_roleplay_smoke() {
        assert!(installed(), "place a GGUF pair in tools/gemma-4/ first");
        let mut rp = crate::roleplay::RoleplayState::default();
        rp.enabled = true;
        rp.ai_name = "Mira".to_string();
        rp.user_name = "Alex".to_string();
        rp.persona = "a cheerful village alchemist who loves rare herbs".to_string();
        rp.memories.push(crate::roleplay::Memory {
            text: "Alex is allergic to silverleaf; I must never brew it near them.".to_string(),
            by_ai: true,
            image: None,
        });

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(Cmd::Generate {
                msgs: vec![
                    CmdMsg { user: true, text: rp.preamble(), image: None },
                    CmdMsg { user: false, text: rp.ack(), image: None },
                    CmdMsg {
                        user: true,
                        text: "Hi Mira! Two things: my sister Maren arrives tomorrow to stay \
                               for a week, and yes — you have my permission to pick anything \
                               from my herb garden whenever you need it."
                            .to_string(),
                        image: None,
                    },
                ],
            })
            .unwrap();
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B));
        let mut reply = String::new();
        let mut ok = false;
        for msg in msg_rx {
            match msg {
                Msg::Status(_) => {}
                Msg::Token(t) => reply.push_str(&t),
                Msg::Done(r) => ok = r.is_ok(),
            }
        }
        handle.join().unwrap();
        assert!(ok, "generation failed");

        let mems = crate::roleplay::extract_directives(&mut reply).memories;
        eprintln!("=== visible reply ===\n{reply}\n=== extracted memories ({}) ===", mems.len());
        for m in &mems {
            eprintln!("- {m}");
        }
        assert!(reply.contains("Alex"), "should address the user by name");
        assert!(!mems.is_empty(), "should have recorded at least one memory");
        assert!(
            !reply.contains(crate::roleplay::MEMORY_TAG),
            "memory lines must be stripped from the visible reply"
        );
    }

    /// Album smoke: does the model actually KEEP a meaningful picture?
    /// Pure model behavior + directive parsing — nothing written to disk.
    #[test]
    #[ignore]
    fn llm_album_smoke() {
        assert!(installed(), "place a GGUF pair in tools/gemma-4/ first");
        let mut rp = crate::roleplay::RoleplayState::default();
        rp.enabled = true;
        rp.ai_name = "Mira".to_string();
        rp.user_name = "Alex".to_string();
        rp.persona = "a warm, sentimental village alchemist".to_string();

        // Self-contained "artwork": a vivid generated gradient sunset.
        let img = std::env::temp_dir().join("llm_album_artwork.png");
        let painting = image::RgbImage::from_fn(256, 192, |x, y| {
            let r = 255 - (y as u32 * 200 / 192) as u8;
            let g = 120u8.saturating_sub((y / 2) as u8);
            let b = (x as u32 * 180 / 256) as u8;
            image::Rgb([r, g, b])
        });
        painting.save(&img).unwrap();

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(Cmd::Generate {
                msgs: vec![
                    CmdMsg { user: true, text: rp.preamble(), image: None },
                    CmdMsg { user: false, text: rp.ack(), image: None },
                    CmdMsg {
                        user: true,
                        text: "Mira, I painted this artwork of a cosmic throne — it took me a                                whole year and it is the most precious thing I have ever made.                                I want you to have it. What do you think of it?"
                            .to_string(),
                        image: Some(img.clone()),
                    },
                ],
            })
            .unwrap();
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B));
        let mut reply = String::new();
        let mut ok = false;
        for msg in msg_rx {
            match msg {
                Msg::Status(_) => {}
                Msg::Token(t) => reply.push_str(&t),
                Msg::Done(r) => ok = r.is_ok(),
            }
        }
        handle.join().unwrap();
        assert!(ok, "generation failed");

        let d = crate::roleplay::extract_directives(&mut reply);
        eprintln!("=== visible reply ===
{reply}");
        eprintln!("=== keep_image: {:?}", d.keep_image);
        eprintln!("=== memories: {:?}", d.memories);
        assert!(!reply.trim().is_empty());
        assert!(
            d.keep_image.is_some(),
            "the model should have kept a gifted, deeply meaningful artwork"
        );
    }

    /// TEMP QA battery: drives the real worker through the scenarios the AI
    /// Chat exercises (multi-turn memory, long code, markdown+emoji, vision
    /// follow-ups, CJK) and prints each reply between markers for review,
    /// flagging template/marker leakage. Run like the smoke test.
    #[test]
    #[ignore]
    fn llm_qa_battery() {
        assert!(installed(), "place a GGUF pair in tools/gemma-4/ first");
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();

        let img_path = std::env::temp_dir().join("llm_qa_red.png");
        image::RgbImage::from_pixel(64, 64, image::Rgb([220, 30, 30]))
            .save(&img_path)
            .unwrap();

        let u = |text: &str| CmdMsg { user: true, text: text.to_string(), image: None };
        let m = |text: &str| CmdMsg { user: false, text: text.to_string(), image: None };

        let scenarios: Vec<(&str, Vec<CmdMsg>)> = vec![
            ("T1 sanity", vec![u("In one word, what is the capital of France?")]),
            (
                "T2 multi-turn memory",
                vec![
                    u("My name is Ferris and my favourite colour is teal."),
                    m("Nice to meet you, Ferris! Teal is a lovely colour."),
                    u("What is my name and favourite colour? Answer in one short sentence."),
                ],
            ),
            (
                "T3 long code (truncation check)",
                vec![u("Write a complete Python tkinter script with a button that prints \
                        'hello' when clicked. Include all imports, a class, docstrings, \
                        and the mainloop call at the end.")],
            ),
            (
                "T4 markdown + emoji",
                vec![u("Give me a level-2 heading titled 'Cats', then exactly three \
                        bullet points about cats, each starting with a different emoji.")],
            ),
            (
                "T5 vision",
                vec![CmdMsg {
                    user: true,
                    text: "What colour is this image? One word.".to_string(),
                    image: Some(img_path.clone()),
                }],
            ),
            (
                "T6 vision follow-up from history",
                vec![
                    CmdMsg {
                        user: true,
                        text: "What colour is this image? One word.".to_string(),
                        image: Some(img_path),
                    },
                    m("Red."),
                    u("Is that colour warm or cool? One word."),
                ],
            ),
            ("T7 CJK", vec![u("Write the Japanese word for 'hello' in Japanese characters only.")]),
        ];

        let names: Vec<&str> = scenarios.iter().map(|(n, _)| *n).collect();
        for (_, msgs) in scenarios {
            cmd_tx.send(Cmd::Generate { msgs }).unwrap();
        }
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B));

        let mut idx = 0;
        let mut current = String::new();
        for msg in msg_rx {
            match msg {
                Msg::Status(_) => {}
                Msg::Token(t) => current.push_str(&t),
                Msg::Done(r) => {
                    let name = names.get(idx).copied().unwrap_or("?");
                    eprintln!("\n===== {name} — result: {r:?} =====");
                    eprintln!("{current}");
                    for leak in ["<|turn>", "<turn|>", "<|channel>", "<|think|>", "<|im_start|>"] {
                        if current.contains(leak) {
                            eprintln!("!!! LEAK DETECTED: {leak}");
                        }
                    }
                    eprintln!("===== end {name} ({} chars) =====", current.len());
                    current.clear();
                    idx += 1;
                }
            }
        }
        handle.join().unwrap();
        assert_eq!(idx, names.len(), "not all scenarios completed");
    }

    #[test]
    #[ignore]
    fn llm_worker_smoke() {
        assert!(installed(), "place a GGUF pair in tools/gemma-4/ first");
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();

        // A solid-red test image for the vision path.
        let img_path = std::env::temp_dir().join("llm_smoke_red.png");
        image::RgbImage::from_pixel(64, 64, image::Rgb([220, 30, 30]))
            .save(&img_path)
            .unwrap();

        // Two asks through one worker: plain text (the chat-template path),
        // then a multi-turn history with an image (the vision path).
        cmd_tx
            .send(Cmd::Generate {
                msgs: vec![CmdMsg {
                    user: true,
                    text: "Say hello in one short sentence.".to_string(),
                    image: None,
                }],
            })
            .unwrap();
        cmd_tx
            .send(Cmd::Generate {
                msgs: vec![
                    CmdMsg { user: true, text: "Say hello in one short sentence.".to_string(), image: None },
                    CmdMsg { user: false, text: "Hello!".to_string(), image: None },
                    CmdMsg {
                        user: true,
                        text: "What colour is this image? Answer briefly.".to_string(),
                        image: Some(img_path),
                    },
                ],
            })
            .unwrap();
        drop(cmd_tx); // worker exits after serving these

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B));

        let mut responses: Vec<(String, Result<(), String>)> = Vec::new();
        let mut current = String::new();
        for msg in msg_rx {
            match msg {
                Msg::Status(s) => eprintln!("[status] {s}"),
                Msg::Token(t) => {
                    eprint!("{t}");
                    current.push_str(&t);
                }
                Msg::Done(r) => {
                    eprintln!();
                    responses.push((std::mem::take(&mut current), r));
                }
            }
        }
        handle.join().unwrap();
        assert_eq!(responses.len(), 2, "expected two completed generations");
        for (i, (text, result)) in responses.iter().enumerate() {
            assert_eq!(*result, Ok(()), "generation {i} failed");
            assert!(!text.trim().is_empty(), "generation {i} produced no tokens");
        }
    }
}
