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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use eframe::egui;

mod memory_pass;
pub use memory_pass::MemoryActivity;

/// Set by the chat's Stop button; the worker checks it between tokens (and
/// between the slow steps before the first one) and ends the reply early
/// (the partial text stays in the bubble).
/// A single static works because only one generation runs at a time — same
/// scheme as the text-to-image view's cancel flag.
static GEN_CANCEL: AtomicBool = AtomicBool::new(false);

/// Set by the worker when a reply stopped before the model ended it — at
/// Max reply length or the context's edge — so its last line may be cut
/// short (Tag images then leaves the reply's tag edits unapplied).
static GEN_CUT_OFF: AtomicBool = AtomicBool::new(false);

/// Model folder under the shared models root (`tagger::models_root()`).
pub const FOLDER: &str = "gemma-4";
/// The instruct weights (Q4_K_M quant, ~5 GB).
pub const MODEL_FILE: &str = "gemma-4-E4B-it-Q4_K_M.gguf";
/// The multimodal (vision) projector llama.cpp uses to encode images (~1 GB).
pub const MMPROJ_FILE: &str = "mmproj-F16.gguf";

// The other options (all vision-capable; every repo names its projector
// mmproj-F16.gguf, so MMPROJ_FILE is shared).
pub const FOLDER_27B: &str = "gemma-3-27b";
pub const MODEL_FILE_27B: &str = "gemma-3-27b-it-Q4_K_M.gguf";
pub const FOLDER_31B: &str = "gemma-4-31b";
pub const MODEL_FILE_31B: &str = "gemma-4-31B-it-Q4_K_M.gguf";
pub const FOLDER_QWEN: &str = "qwen3-vl-8b";
pub const MODEL_FILE_QWEN: &str = "Qwen3-VL-8B-Instruct-Q4_K_M.gguf";
pub const FOLDER_QWEN_30B: &str = "qwen3-vl-30b-thinking";
pub const MODEL_FILE_QWEN_30B: &str = "Qwen3-VL-30B-A3B-Thinking-Q4_K_M.gguf";

/// Which local model the chat runs (Settings → AI Model). All are
/// vision-capable; they trade download size + speed for quality.
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GemmaModel {
    /// Gemma 4 E4B instruct — ~6 GB, fast, fits fully on 8 GB+ GPUs. The default.
    #[default]
    E4B,
    /// Gemma 3 27B — ~16 GB, the previous-gen flagship, very strong.
    /// (Replaced the Gemma 4 26B A4B; settings saved as "A26B" land here.)
    #[serde(alias = "A26B")]
    G27B,
    /// Gemma 4 31B dense — ~19.5 GB, the strongest option.
    D31B,
    /// Qwen3-VL 8B instruct — ~6 GB, a different flavour to Gemma.
    Qwen8B,
    /// Qwen3-VL 30B A3B Thinking — ~18.6 GB mixture-of-experts (3B active,
    /// so it stays quick even spilling to RAM); reasons in a `<think>` block
    /// before answering.
    Qwen30B,
}

impl GemmaModel {
    /// Display order: 31B leads as the developer's pick (best answers in
    /// testing so far), then the rest by family and size.
    pub const ALL: [GemmaModel; 5] = [
        GemmaModel::D31B,
        GemmaModel::E4B,
        GemmaModel::G27B,
        GemmaModel::Qwen8B,
        GemmaModel::Qwen30B,
    ];

    pub fn folder(self) -> &'static str {
        match self {
            GemmaModel::E4B => FOLDER,
            GemmaModel::G27B => FOLDER_27B,
            GemmaModel::D31B => FOLDER_31B,
            GemmaModel::Qwen8B => FOLDER_QWEN,
            GemmaModel::Qwen30B => FOLDER_QWEN_30B,
        }
    }

    pub fn model_file(self) -> &'static str {
        match self {
            GemmaModel::E4B => MODEL_FILE,
            GemmaModel::G27B => MODEL_FILE_27B,
            GemmaModel::D31B => MODEL_FILE_31B,
            GemmaModel::Qwen8B => MODEL_FILE_QWEN,
            GemmaModel::Qwen30B => MODEL_FILE_QWEN_30B,
        }
    }

    pub fn mmproj_file(self) -> &'static str {
        MMPROJ_FILE
    }

    pub fn label(self) -> &'static str {
        match self {
            GemmaModel::E4B => "Gemma 4 E4B",
            GemmaModel::G27B => "Gemma 3 27B",
            GemmaModel::D31B => "Gemma 4 31B",
            GemmaModel::Qwen8B => "Qwen3-VL 8B",
            GemmaModel::Qwen30B => "Qwen3-VL 30B Thinking",
        }
    }

    /// A few words for the chat's model picker (the longer `hint` is the
    /// rows' hover text).
    pub fn tagline(self) -> &'static str {
        match self {
            GemmaModel::E4B => "Fastest answers",
            GemmaModel::G27B => "Strong all-rounder",
            GemmaModel::D31B => "Strongest, thinks first",
            GemmaModel::Qwen8B => "Fast, a different style",
            GemmaModel::Qwen30B => "Deep reasoning",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            GemmaModel::E4B => "~6 GB · fast, fits fully on 8 GB+ GPUs — the light pick",
            GemmaModel::G27B => "~16 GB · very strong; wants ~20 GB VRAM, else spills to RAM (slower)",
            GemmaModel::D31B => "~19.5 GB · strongest; on a 16 GB GPU it runs partly on the CPU (slower)",
            GemmaModel::Qwen8B => "~6 GB · fast, a different flavour to Gemma — worth comparing",
            GemmaModel::Qwen30B => {
                "~18.6 GB · reasons before answering; only 3B active (MoE), so it stays quick even spilling to RAM"
            }
        }
    }

    pub fn installed(self) -> bool {
        crate::tagger::resolve(self.folder(), self.model_file()).is_some()
            && crate::tagger::resolve(self.folder(), self.mmproj_file()).is_some()
    }
}

/// True when this binary was compiled with local-AI support.
pub const BUILT_WITH_LLM: bool = cfg!(feature = "llm");

/// Sampling knobs for the AI chat, edited in AI Settings (the Tools menu's
/// gear) and persisted in Settings (`ai_gen`). The defaults are Gemma's
/// recommended sampling.
#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GenParams {
    /// Randomness of the replies: 0 always picks the most likely token
    /// (deterministic), 2 gets wild.
    pub temperature: f32,
    /// Sample only among the k most likely tokens.
    pub top_k: i32,
    /// Nucleus sampling: drop the tail once cumulative probability passes p.
    pub top_p: f32,
    /// Cap on generated tokens per reply. 1024 turned out to truncate real
    /// answers mid-code-block; 3072 fits long code examples comfortably.
    pub max_tokens: usize,
    /// How long the thinking models may reason before answering (the
    /// slider at the bottom of the chat's tools menu).
    pub thinking: ThinkLevel,
}

impl Default for GenParams {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 64, top_p: 0.95, max_tokens: 3072, thinking: ThinkLevel::Medium }
    }
}

/// Thinking mode: how much private reasoning a thinking model (the Gemma 4
/// 31B, Qwen3-VL 30B Thinking) does before it answers. Mostly a speed knob —
/// on a partly CPU-offloaded 31B the thought is most of the wait.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum ThinkLevel {
    /// Answer straight away.
    Off,
    Low,
    /// Today's behaviour: a third of the reply budget.
    #[default]
    Medium,
    High,
    /// Think as long as needed, leaving just enough room to answer.
    Max,
}

impl ThinkLevel {
    pub const ALL: [ThinkLevel; 5] =
        [ThinkLevel::Off, ThinkLevel::Low, ThinkLevel::Medium, ThinkLevel::High, ThinkLevel::Max];

    /// Position on the slider, 0 (Off) to 4 (Max).
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|l| *l == self).unwrap_or(2)
    }

    pub fn label(self) -> &'static str {
        match self {
            ThinkLevel::Off => "Off",
            ThinkLevel::Low => "Low",
            ThinkLevel::Medium => "Medium",
            ThinkLevel::High => "High",
            ThinkLevel::Max => "Max",
        }
    }

    /// Tokens of thought allowed before the thought is force-closed and the
    /// model must answer — an allowance ON TOP of "Max reply length", so the
    /// visible answer always keeps its full length. (It used to be carved
    /// out of the reply length, which left Max the shortest answers.)
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    pub fn budget(self) -> usize {
        match self {
            ThinkLevel::Off => 0,
            ThinkLevel::Low => 256,
            ThinkLevel::Medium => 1024,
            ThinkLevel::High => 2048,
            ThinkLevel::Max => 4096,
        }
    }
}

impl GemmaModel {
    /// The models that reason before answering — the only ones the
    /// thinking slider affects.
    pub fn thinks(self) -> bool {
        matches!(self, GemmaModel::D31B | GemmaModel::Qwen30B)
    }

    /// Gemma 4 (E4B, 31B): the only models whose vision encoder takes a
    /// per-image token budget, so the only ones Image detail applies to.
    /// Gemma 3 uses a fixed image size; Qwen counts image tokens differently.
    pub fn has_image_detail(self) -> bool {
        matches!(self, GemmaModel::E4B | GemmaModel::D31B)
    }
}

/// Image detail (Settings → AI Model): how many vision tokens Gemma 4 reads
/// each image with. More tokens = higher resolution = small text, jewelry,
/// eye colour that the default misses — at the cost of time and GPU memory.
/// The user's pick is a request: after every model load the worker checks
/// the GPU and runs the highest level that's safe (see `worker::decide_detail`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum ImageDetail {
    /// 280 tokens — Gemma 4's own default; today's behaviour.
    #[default]
    Standard,
    /// 560 tokens.
    High,
    /// 1120 tokens.
    Maximum,
}

impl ImageDetail {
    pub const ALL: [ImageDetail; 3] = [ImageDetail::Standard, ImageDetail::High, ImageDetail::Maximum];

    /// Vision tokens per image.
    pub fn tokens(self) -> i32 {
        match self {
            ImageDetail::Standard => 280,
            ImageDetail::High => 560,
            ImageDetail::Maximum => 1120,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ImageDetail::Standard => "Standard",
            ImageDetail::High => "High",
            ImageDetail::Maximum => "Maximum",
        }
    }

    /// One level down (`None` from Standard).
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    pub fn lower(self) -> Option<Self> {
        match self {
            ImageDetail::Standard => None,
            ImageDetail::High => Some(ImageDetail::Standard),
            ImageDetail::Maximum => Some(ImageDetail::High),
        }
    }

    /// Free GPU memory (bytes) this level needs *beyond a normal Standard
    /// reply context* — measured on Gemma 4 31B (the biggest), RTX 5080
    /// Laptop, Vulkan: the reply's compute buffer grows from 533 MB (batch
    /// 512) to ~670 MB (640) / 1880 MB (1280), and the vision encoder needs
    /// ~0.4 / ~0.8 GB more; +25% margin. Not measured on AMD.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    pub fn free_needed(self) -> usize {
        const MB: usize = 1024 * 1024;
        match self {
            ImageDetail::Standard => 0,
            ImageDetail::High => 768 * MB,
            ImageDetail::Maximum => 2816 * MB,
        }
    }

    /// Free GPU memory (bytes) the vision encoder needs for one image at
    /// this level, checked right before encoding: llama.cpp ABORTS the whole
    /// process (GGML_ASSERT) when that allocation fails, so it must never be
    /// attempted without room. Measured 632 MB at 1024 tokens on the 31B;
    /// +25% and scaled to the level's full budget.
    #[cfg_attr(not(feature = "llm"), allow(dead_code))]
    pub fn encode_needed(self) -> usize {
        const MB: usize = 1024 * 1024;
        match self {
            ImageDetail::Standard => 0,
            ImageDetail::High => 512 * MB,
            ImageDetail::Maximum => 1024 * MB,
        }
    }
}

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
// The worker that reads these (and constructs Msg) only exists in `llm` builds.
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
struct CmdMsg {
    user: bool,
    text: String,
    image: Option<PathBuf>,
}

/// A request to the inference worker. The full (capped) history is sent every
/// turn — the worker rebuilds the model context from scratch, so multi-turn
/// chat works without keeping KV-cache state between requests.
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
enum Cmd {
    /// `pinned` leading messages (the role-play / Krea 2 priming pair) are
    /// always kept; when the prompt outgrows the context, the oldest of the
    /// rest go. `detailed` = Detailed Vision: the newest image goes as the
    /// full picture plus overlapping close-up sections. Replies on the
    /// worker's own channel (the model tests use this).
    #[cfg_attr(not(test), allow(dead_code))]
    Generate { msgs: Vec<CmdMsg>, params: GenParams, pinned: usize, detailed: bool },
    /// The same, replying on `reply` — the app's chat and a Link device's
    /// questions share one resident worker, each hearing its own reply.
    /// `remote`: a Link device's question, with its own stop flag.
    Ask {
        msgs: Vec<CmdMsg>,
        params: GenParams,
        pinned: usize,
        detailed: bool,
        reply: Sender<Msg>,
        remote: Option<RemoteFlags>,
    },
    /// A memory pass (src/llm/memory_pass.rs): one text-only question —
    /// `head`, then the chat's `lines`, the oldest left out until it fits —
    /// whose prompt is read in small steps so `cancel` stops it even then,
    /// which a chat question's can't be. Replies on `reply` like `Ask`.
    Pass { head: String, lines: Vec<String>, params: GenParams, reply: Sender<Msg>, cancel: Arc<AtomicBool> },
}

/// A question with stop and cut-off flags of its own: a Link device's
/// (never stopped by this PC's Stop button, nor stopping this PC's reply,
/// and its cut-off reported to it rather than through this PC's chat
/// state), or a memory pass asked of a Link host.
#[derive(Clone, Default)]
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
struct RemoteFlags {
    cancel: Arc<AtomicBool>,
    cut_off: Arc<AtomicBool>,
}

/// A memory pass asked of a Link host: its own flags, and its prompt in
/// parts besides the whole question, so a host that knows passes reads it in
/// steps (as this PC's worker does) and one that doesn't still answers it.
struct RemotePass {
    flags: RemoteFlags,
    head: String,
    lines: Vec<String>,
    /// Set when the host says it has no model loaded (it never loads one for
    /// a pass).
    cold: Arc<AtomicBool>,
}

/// A message streamed back from the inference worker.
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
enum Msg {
    /// Progress line ("Loading model…", "Reading the image…", "Thinking…").
    Status(String),
    /// One decoded piece of the response text.
    Token(String),
    /// Generation finished (or failed). The worker stays alive for more
    /// prompts unless the error was fatal (model failed to load).
    Done(Result<(), String>),
    /// After loading: the Image detail level actually running (`None` for
    /// models without the setting) and, when it's lower than requested, why.
    ImageDetail { effective: Option<ImageDetail>, reason: Option<String> },
}

/// The resident inference worker, shared by the chat and Link's devices:
/// one model in memory at a time.
struct SharedWorker {
    tx: Sender<Cmd>,
    model: GemmaModel,
    detail: ImageDetail,
    /// The worker thread, handed on when this worker is dropped so the next
    /// one waits for it (see `RETIRED`).
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Worker threads that were let go (model switch, unload) and may still be
/// freeing their model. The next worker waits for them before sizing its GPU
/// offload — otherwise it sees the old model's memory as taken and loads on
/// the CPU only.
static RETIRED: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>> = std::sync::Mutex::new(Vec::new());

impl Drop for SharedWorker {
    fn drop(&mut self) {
        if let Some(t) = self.thread.take() {
            let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
            retired.retain(|t| !t.is_finished());
            retired.push(t);
        }
    }
}

static SHARED: std::sync::Mutex<Option<SharedWorker>> = std::sync::Mutex::new(None);

/// The Image detail level the loaded worker reported (when it loaded for a
/// Link device, the chat learns it from here).
static DETAIL_REPORT: std::sync::Mutex<Option<(Option<ImageDetail>, Option<String>)>> = std::sync::Mutex::new(None);

/// Link devices' questions running or queued on the worker right now, and
/// when the last one ended.
static REMOTE_BUSY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REMOTE_LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

/// The stop flags of the memory passes queued or running on the resident
/// worker — this PC's own and Link devices' — so whatever else needs the
/// worker (a question, unloading, another model) stops them first, and
/// nothing waits behind a pass.
static PASSES: std::sync::Mutex<Vec<Arc<AtomicBool>>> = std::sync::Mutex::new(Vec::new());

/// A memory pass is about to be queued on the resident worker.
fn note_pass(cancel: &Arc<AtomicBool>) {
    PASSES.lock().unwrap_or_else(|e| e.into_inner()).push(cancel.clone());
}

/// That pass is over (or was never queued).
fn forget_pass(cancel: &Arc<AtomicBool>) {
    PASSES.lock().unwrap_or_else(|e| e.into_inner()).retain(|c| !Arc::ptr_eq(c, cancel));
}

/// Stop the memory passes queued or running on the resident worker (each
/// stops within one step and runs again later).
fn stop_passes() {
    for cancel in PASSES.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
        cancel.store(true, Ordering::SeqCst);
    }
}

/// The resident worker is `model` at `detail` (and so a memory pass can run
/// without loading anything).
fn worker_matches(model: GemmaModel, detail: ImageDetail) -> bool {
    SHARED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|w| w.model == model && (!model.has_image_detail() || w.detail == detail))
}

/// A Link device is using this PC's worker, or was a moment ago (its next
/// question is likely on the way).
fn link_busy() -> bool {
    REMOTE_BUSY.load(Ordering::SeqCst) > 0
        || REMOTE_LAST
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(20))
}

/// Queue `cmd` on the resident worker only if it already runs `model` at
/// `detail` — a memory pass never loads a model. Hands `cmd` back when it
/// doesn't (or that worker has stopped).
fn submit_loaded(model: GemmaModel, detail: ImageDetail, cmd: Cmd) -> Result<(), Cmd> {
    let mut shared = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    if !shared.as_ref().is_some_and(|w| w.model == model && (!model.has_image_detail() || w.detail == detail)) {
        return Err(cmd);
    }
    match shared.as_ref().map(|w| w.tx.send(cmd)) {
        Some(Ok(())) => Ok(()),
        Some(Err(std::sync::mpsc::SendError(back))) => {
            *shared = None;
            Err(back)
        }
        None => unreachable!("checked above"),
    }
}

/// Queue `cmd` on the resident worker, loading `model` at `detail` first
/// when another one is loaded (that worker finishes the reply it's on, then
/// frees its memory). A worker that has stopped (its model failed to load)
/// is replaced. Memory passes on the worker stop first.
fn submit(ctx: &egui::Context, model: GemmaModel, detail: ImageDetail, cmd: Cmd) -> Result<(), String> {
    if !BUILT_WITH_LLM {
        return Err("This build was compiled without the AI feature (`llm`).".to_string());
    }
    stop_passes();
    let mut shared = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    let same = |w: &SharedWorker| w.model == model && (!model.has_image_detail() || w.detail == detail);
    if !shared.as_ref().is_some_and(same) {
        *shared = None;
        *DETAIL_REPORT.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    let mut cmd = cmd;
    for _ in 0..2 {
        if shared.is_none() {
            let Some((tx, thread)) = spawn_worker(ctx.clone(), model, detail) else {
                return Err("The AI couldn't start".to_string());
            };
            *shared = Some(SharedWorker { tx, model, detail, thread: Some(thread) });
        }
        match shared.as_ref().map(|w| w.tx.send(cmd)) {
            Some(Ok(())) => return Ok(()),
            // That worker is gone: start a fresh one and try once more.
            Some(Err(std::sync::mpsc::SendError(back))) => {
                cmd = back;
                *shared = None;
            }
            None => break,
        }
    }
    Err("The AI worker stopped — try again.".to_string())
}

/// Unload the resident worker (its weights free once a reply in progress
/// ends; a memory pass stops within a step). The next question loads it
/// again.
pub(crate) fn release_shared_worker() {
    stop_passes();
    *SHARED.lock().unwrap_or_else(|e| e.into_inner()) = None;
    *DETAIL_REPORT.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Is a worker loaded?
pub(crate) fn worker_loaded() -> bool {
    SHARED.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// The chat model a Link host serves with (its own AI Model setting), when
/// it's set up here.
pub(crate) fn host_model_label() -> Option<String> {
    let (model, _) = crate::link::host_llm()?;
    (BUILT_WITH_LLM && model.installed()).then(|| model.label().to_string())
}

/// Counts a Link device's question as running, and when it ends.
struct RemoteBusy;

impl RemoteBusy {
    fn start() -> RemoteBusy {
        REMOTE_BUSY.fetch_add(1, Ordering::SeqCst);
        RemoteBusy
    }
}

impl Drop for RemoteBusy {
    fn drop(&mut self) {
        REMOTE_BUSY.fetch_sub(1, Ordering::SeqCst);
        *REMOTE_LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::time::Instant::now());
    }
}

/// One message of a question as it travels over Link (its picture, if any,
/// travels as a file named `attachment`).
#[derive(serde::Serialize, serde::Deserialize)]
struct WireMsg {
    user: bool,
    text: String,
    #[serde(default)]
    attachment: Option<String>,
}

/// Host side: answer a Link device's chat question with this PC's model (the
/// one set in its AI Model tab), streaming the reply back as it's written.
/// Runs on the resident worker, after any question already on it.
pub(crate) fn serve_remote(call: &crate::link::Incoming, ctx: &egui::Context) -> Result<serde_json::Value, String> {
    use serde_json::json;
    if !BUILT_WITH_LLM {
        return Err("The app on this PC was built without the AI".into());
    }
    let (model, detail) = crate::link::host_llm().ok_or("The AI isn't set up on this PC")?;
    if !model.installed() {
        return Err(format!("{} isn't set up on this PC", model.label()));
    }
    let msgs: Vec<WireMsg> =
        serde_json::from_value(call.body.get("msgs").cloned().unwrap_or_default()).map_err(|_| "Bad question".to_string())?;
    if msgs.is_empty() || msgs.len() > 400 {
        return Err("Bad question".into());
    }
    let mut params: GenParams =
        serde_json::from_value(call.body.get("params").cloned().unwrap_or_default()).unwrap_or_default();
    // Keep a device from asking for more than the context holds.
    params.max_tokens = params.max_tokens.clamp(1, 16_384);
    params.temperature = params.temperature.clamp(0.0, 2.0);
    params.top_k = params.top_k.clamp(1, 1000);
    params.top_p = params.top_p.clamp(0.0, 1.0);
    let pinned = (call.body.get("pinned").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize).min(msgs.len());
    let detailed = call.body.get("detailed").and_then(serde_json::Value::as_bool).unwrap_or(false);
    // A device's memory pass: read in steps, as this PC's own are (so this
    // PC's next question, or the device's, never waits long behind it).
    let pass: Option<(String, Vec<String>)> = match call.body.get("pass") {
        None => None,
        Some(p) => {
            let head = p.get("head").and_then(serde_json::Value::as_str).map(str::to_string);
            let lines: Option<Vec<String>> = p.get("lines").and_then(|l| serde_json::from_value(l.clone()).ok());
            match (head, lines) {
                (Some(head), Some(lines))
                    if head.len() <= 64 * 1024 && lines.len() <= 400 && lines.iter().all(|l| l.len() <= 16 * 1024) =>
                {
                    Some((head, lines))
                }
                // Never asked as an ordinary question instead (that could
                // load the model, and couldn't be stopped like a pass).
                _ => return Err("Bad memory update".into()),
            }
        }
    };

    // The pictures, under their own extensions (the worker tells a video
    // from an image by it), in a folder of their own.
    let dir = crate::link::job_dir("ask");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut cmd_msgs = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.into_iter().enumerate() {
        let image = match m.attachment {
            Some(name) => {
                let Some((_, src)) = call.blobs.iter().find(|(n, _)| *n == name) else {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err("A picture went missing on the way".into());
                };
                let ext = std::path::Path::new(&name)
                    .extension()
                    .and_then(|e| e.to_str())
                    .filter(|e| e.len() <= 5 && e.chars().all(|c| c.is_ascii_alphanumeric()))
                    .unwrap_or("png")
                    .to_ascii_lowercase();
                let dest = dir.join(format!("{i}.{ext}"));
                if let Err(e) = std::fs::rename(src, &dest) {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err(e.to_string());
                }
                Some(dest)
            }
            None => None,
        };
        cmd_msgs.push(CmdMsg { user: m.user, text: m.text, image });
    }

    let flags = RemoteFlags { cancel: call.cancel_flag(), cut_off: Arc::default() };
    let (tx, rx) = std::sync::mpsc::channel();
    let _busy = RemoteBusy::start();
    let is_pass = pass.is_some();
    let queued = match pass {
        // A device's memory pass: only on the model that's loaded already
        // (one is never loaded for a pass, here either), and stopped by
        // whatever else needs the worker, as this PC's own are.
        Some((head, lines)) => {
            note_pass(&flags.cancel);
            let cmd = Cmd::Pass { head, lines, params, reply: tx, cancel: flags.cancel.clone() };
            if submit_loaded(model, detail, cmd).is_err() {
                forget_pass(&flags.cancel);
                let _ = std::fs::remove_dir_all(&dir);
                return Ok(json!({ "stopped": true, "cold": true }));
            }
            Ok(())
        }
        None => submit(
            ctx,
            model,
            detail,
            Cmd::Ask { msgs: cmd_msgs, params, pinned, detailed, reply: tx, remote: Some(flags.clone()) },
        ),
    };
    if let Err(e) = queued {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    let result = loop {
        match rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(Msg::Status(text)) => {
                call.event(json!({ "kind": "status", "text": text }));
            }
            Ok(Msg::Token(text)) => {
                if !call.event(json!({ "kind": "token", "text": text })) {
                    flags.cancel.store(true, Ordering::SeqCst);
                }
            }
            Ok(Msg::ImageDetail { .. }) => {}
            Ok(Msg::Done(r)) => break r,
            // Still thinking (a Stop from the device reaches the worker
            // through `flags.cancel` itself).
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break Err("The AI stopped".to_string()),
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    if is_pass {
        forget_pass(&flags.cancel);
        // Stopped here (for this PC's own question, say): the device tries
        // again later rather than counting it as failed.
        return result.map(|()| json!({ "stopped": flags.cancel.load(Ordering::SeqCst) }));
    }
    result.map(|()| json!({ "cut_off": flags.cut_off.load(Ordering::SeqCst) }))
}

/// Receiver side: a Link host answers the chat. Sends the question (with its
/// pictures) and relays the reply's progress and words as if the worker
/// here were writing it. `pass`: a memory pass, with stop and cut-off flags
/// of its own (the chat's otherwise).
#[allow(clippy::too_many_arguments)]
fn remote_ask(
    peer: &Arc<crate::link::Peer>,
    host: &str,
    msgs: Vec<CmdMsg>,
    params: GenParams,
    pinned: usize,
    detailed: bool,
    pass: Option<RemotePass>,
    tx: &Sender<Msg>,
    ctx: &egui::Context,
) {
    use crate::link::{BlobSrc, CallEvent};
    use serde_json::{json, Value};
    let send = |m: Msg| {
        let _ = tx.send(m);
        ctx.request_repaint();
    };
    send(Msg::Status(format!("Sending to {host}…")));
    let last = msgs.len().saturating_sub(1);
    let mut blobs = Vec::new();
    let mut wire = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.into_iter().enumerate() {
        let mut text = m.text;
        let mut attachment = None;
        if let Some(path) = &m.image {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png").to_ascii_lowercase();
            let name = format!("{i}.{ext}");
            let src = if crate::archive::is_entry(path) {
                crate::archive::read(path).ok().map(|data| BlobSrc::Bytes { name: name.clone(), data })
            } else {
                path.is_file().then(|| BlobSrc::File { name: name.clone(), path: path.clone() })
            };
            match src {
                Some(src) => {
                    blobs.push(src);
                    attachment = Some(name);
                }
                // An earlier turn's picture that's gone: left out with a
                // note (as the worker here does); this turn's own fails.
                None if i < last => {
                    let what = if crate::is_video(path) { "video" } else { "image" };
                    text = format!("[{what} no longer available]\n{text}");
                }
                None => {
                    send(Msg::Done(Err(format!("Couldn't read {}", path.display()))));
                    return;
                }
            }
        }
        wire.push(WireMsg { user: m.user, text, attachment });
    }
    let mut body = json!({ "msgs": wire, "params": params, "pinned": pinned, "detailed": detailed });
    if let Some(p) = &pass {
        body["pass"] = json!({ "head": p.head, "lines": p.lines });
    }
    let (job, cold) = match pass {
        Some(p) => (Some(p.flags), Some(p.cold)),
        None => (None, None),
    };
    let mut call = match peer.call("llm.ask", body, blobs, &crate::link::temp_dir()) {
        Ok(c) => c,
        Err(e) => {
            send(Msg::Done(Err(format!("{host}: {e}"))));
            return;
        }
    };
    let stopped = || match &job {
        Some(flags) => flags.cancel.load(Ordering::SeqCst),
        None => GEN_CANCEL.load(Ordering::SeqCst),
    };
    let result = crate::link::finish_call(&mut call, &stopped, &mut |ev| match ev {
        CallEvent::Event(v) => {
            let text = v.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
            match v.get("kind").and_then(Value::as_str) {
                Some("token") => send(Msg::Token(text)),
                Some("status") => send(Msg::Status(format!("{text} ({host})"))),
                _ => {}
            }
        }
        CallEvent::Blob { path, .. } => {
            let _ = std::fs::remove_file(path);
        }
        _ => {}
    });
    match result {
        Ok(v) => {
            if v.get("cut_off").and_then(Value::as_bool) == Some(true) {
                match &job {
                    Some(flags) => flags.cut_off.store(true, Ordering::SeqCst),
                    None => GEN_CUT_OFF.store(true, Ordering::SeqCst),
                }
            }
            // A pass the host stopped for something else, or didn't run for
            // want of a loaded model: not a failure.
            if let Some(flags) = &job
                && v.get("stopped").and_then(Value::as_bool) == Some(true)
            {
                flags.cancel.store(true, Ordering::SeqCst);
            }
            if let Some(cold) = &cold
                && v.get("cold").and_then(Value::as_bool) == Some(true)
            {
                cold.store(true, Ordering::SeqCst);
            }
            send(Msg::Done(Ok(())));
        }
        Err(e) => send(Msg::Done(Err(format!("{host}: {e}")))),
    }
}

/// How long since a Link device last used the worker (`None`: one is using
/// it now, or none ever did).
pub(crate) fn remote_idle_for() -> Option<std::time::Duration> {
    if REMOTE_BUSY.load(Ordering::SeqCst) > 0 {
        return None;
    }
    REMOTE_LAST.lock().unwrap_or_else(|e| e.into_inner()).map(|t| t.elapsed())
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
    /// The visible text — for a finished model reply the thought channel has
    /// already been split off into `thinking`.
    pub text: String,
    /// The model's thought channel (the big Gemma variants reason before
    /// answering, prefixing replies with `<|channel>thought …`). Shown behind
    /// the chat's collapsible "Thinking" row; never re-sent in the history
    /// and never spoken.
    pub thinking: Option<String>,
    /// Image attached to a user message (the vision input).
    pub image: Option<PathBuf>,
    /// Tag images mode: the image a user message was about (sent along as
    /// its vision input with its current tags; see src/ai_tagging.rs).
    pub tag_image: Option<PathBuf>,
    /// Tag images mode: the tag change a reply made, for its note and Undo.
    pub tag_change: Option<crate::ai_tagging::TagChange>,
    /// The memory pass may read it: sent (or answered) in an ordinary chat
    /// while Remember across chats was on — not in role play, Krea 2 prompts
    /// or Tag images (src/llm/memory_pass.rs).
    pub remember: bool,
    /// The Link hosts (by key) it has been sent to, or written by: its memory
    /// pass may go there, and nowhere else but this PC.
    pub shared_with: Vec<String>,
}

/// One conversation in the AI Chat view (the left strip switches between them).
pub struct Chat {
    pub id: u64,
    pub msgs: Vec<ChatMsg>,
    /// Memory: the messages before this one have been read by the memory
    /// pass, or never were to be.
    pub memory_synced: usize,
    /// When a message was last sent or a reply last finished here.
    pub changed: Option<std::time::Instant>,
    /// Topics the memory notes showed memories from (the pass is shown
    /// their stored titles).
    pub memory_topics: Vec<String>,
    /// Failed passes over the unread part.
    pub memory_tries: u8,
}

impl Chat {
    pub fn new(id: u64) -> Self {
        Chat {
            id,
            msgs: Vec::new(),
            memory_synced: 0,
            changed: None,
            memory_topics: Vec::new(),
            memory_tries: 0,
        }
    }

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
    /// The streaming reply was asked as role play (its control lines are
    /// handled on Done by this, not by the live toggle).
    pub gen_roleplay: bool,
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
    /// Poster stills for video attachments land here from background capture
    /// threads (a VLC decode takes seconds, so it can't run on the UI thread);
    /// `poll` turns them into `msg_thumbs` textures. The set tracks paths
    /// whose capture is already in flight.
    video_poster_tx: Sender<(PathBuf, Option<egui::ColorImage>)>,
    video_poster_rx: Receiver<(PathBuf, Option<egui::ColorImage>)>,
    video_posters_pending: std::collections::HashSet<PathBuf>,
    /// The input card's on-screen rect + the `ctx.input(time)` it was last
    /// drawn, published every frame the chat is visible. Drag-and-drop uses it
    /// both for the hover highlight and to claim drops away from the gallery
    /// (same freshness scheme as the generator's prompt box).
    pub input_rect: Option<(egui::Rect, f64)>,
    /// Click-to-enlarge overlay for a message attachment (src/ai_chat.rs):
    /// images blow up hi-res, videos actually play. `None` = closed.
    pub lightbox: Option<crate::ai_chat::Lightbox>,

    pub run_err: Option<String>,
    /// A generation is in flight.
    pub running: bool,
    /// When the streaming reply started (the live "Thinking… 42 s" counter).
    pub gen_started: Option<std::time::Instant>,
    /// Latest worker status line, shown next to the spinner.
    pub status: String,

    /// Layout cache for the chat's markdown rendering (src/ai_chat.rs).
    pub md_cache: egui_commonmark::CommonMarkCache,
    /// The little AI orb each "Thinking" disclosure row animates, keyed by
    /// (chat id, message index) — the orb is stateful (its particles carry
    /// physics between frames), so rows can't share one.
    pub think_orbs: std::collections::HashMap<(u64, usize), crate::ai_orb::AiOrb>,

    /// Retry requested from inside the message list this frame — applied
    /// after the list finishes drawing (mutating mid-iteration would break
    /// the loop's indices).
    pub pending_retry: Option<usize>,
    /// A user message being edited in place, as (chat id, message index).
    /// The text being typed lives in `edit_draft`; sending rewinds the
    /// conversation to that message and regenerates the reply.
    pub editing: Option<(u64, usize)>,
    pub edit_draft: String,
    /// Edit submitted from inside the message list this frame — applied
    /// after the list finishes drawing (same deferral as `pending_retry`).
    pub pending_edit: Option<usize>,
    /// The running text-to-speech process for "Listen" (Windows speech
    /// synthesis in a spawned PowerShell; killed to stop playback). The
    /// fallback voice — OmniVoice below is used when installed.
    tts: Option<std::process::Child>,
    /// The reply last sent to a voice, as (chat id, message index) — its
    /// action row is the one that shows the voice's spinner / error.
    pub speak_target: Option<(u64, usize)>,
    /// The OmniVoice neural voice (Python sidecar; see src/voice.rs).
    pub voice: crate::voice::VoiceState,
    /// Whisper mic dictation for the input card's mic button (src/dictate.rs).
    pub dictate: crate::dictate::DictationState,
    /// Role playing: the character's names and persona (src/roleplay.rs;
    /// toggled from the input card's Tools menu, set up in AI Settings).
    pub roleplay: crate::roleplay::RoleplayState,
    /// Role play's own memories: the same kind of store as `memory`, kept
    /// apart from it (src/memory.rs, `MemoryKind::Roleplay`).
    pub rp_memory: crate::memory::MemoryStore,
    /// Tag images mode (mirrors the persisted setting; src/ai_tagging.rs).
    pub tag_images: bool,
    /// The image on show in Tag images mode (set by main.rs every frame);
    /// a message sent now is about it.
    pub tag_target: Option<PathBuf>,
    /// Tag edits and undos for main.rs to carry out (it owns the Tag
    /// Manager and the right panel).
    pub tag_requests: Vec<crate::ai_tagging::TagRequest>,
    /// The Gemma 4 variant to chat with (mirrors the persisted setting).
    pub model: GemmaModel,
    /// Sampling knobs (mirrors the persisted setting; sent with every ask, so
    /// edits apply from the next reply — no model reload).
    pub params: GenParams,
    /// The AI Settings window (the Tools menu's gear) is open.
    pub gen_settings_open: bool,
    /// Which AI Settings tab is shown.
    pub ai_settings_tab: crate::ai_settings::AiTab,
    /// The AI's long-term memory across chats (src/memory.rs).
    pub memory: crate::memory::MemoryStore,
    /// The 3D memory map window (src/memory_map.rs).
    pub memory_map: crate::memory_map::MapState,
    /// Image detail the user asked for (mirrors the persisted setting).
    /// Changing it reloads the worker (the vision projector fixes it at load
    /// time).
    pub image_detail: ImageDetail,
    /// What the loaded worker actually runs (`None` until it reports, or for
    /// models without the setting) and why it's lower than asked, if it is.
    pub detail_effective: Option<ImageDetail>,
    pub detail_reason: Option<String>,
    /// The chat's one-line "image detail was lowered" notice has been shown
    /// this session (it shows once, not on every message).
    pub detail_notice_done: bool,
    /// AI Chat was deactivated: drop the worker (freeing the weights) once
    /// the reply in flight and the memory passes still due are done. Cleared
    /// if the chat is re-activated first.
    unload_when_done: bool,
    /// Auto-speak finished replies (mirrors the persisted setting; synced
    /// each frame by main.rs, toggled from the tools menu).
    pub auto_speak: bool,
    /// Detailed Vision (mirrors the persisted setting): tile the newest
    /// image into close-up sections so small details reach the model.
    pub detailed_vision: bool,
    /// Krea 2 prompt writer (mirrors the persisted setting; see
    /// src/krea_prompt.rs). Implies detailed vision.
    pub krea_prompts: bool,
    /// The input card's tools popup is open.
    pub tools_open: bool,
    /// The input card's chats popup (switch / new / delete chat) is open.
    pub chats_open: bool,
    /// Spell-check state for the input card (right-click suggestion menu),
    /// same checker as the generator prompt boxes.
    pub spell: crate::spellcheck::SpellcheckState,

    /// The streaming reply's messages (from the resident worker, or relayed
    /// from a Link host).
    reply_rx: Option<Receiver<Msg>>,
    /// The streaming reply is being answered by a Link host (its name).
    pub remote_host: Option<String>,
    /// Remembering by itself: the memory passes and what they wait for
    /// (src/llm/memory_pass.rs).
    mem_pass: memory_pass::PassState,
}

impl Drop for LlmState {
    fn drop(&mut self) {
        // Don't leave a voice speaking after the app closes.
        if let Some(mut c) = self.tts.take() {
            let _ = c.kill();
        }
        // Nor decrypted album pictures in the temp folder.
        crate::roleplay::clear_album_temp();
        // The AI's picture copies: unless Link devices' questions still
        // running here need them (cleared after those, by Link).
        if !crate::link::handing_over() {
            clear_ai_temp();
        }
    }
}

/// Where this run of the app keeps the AI worker's temp copies of the user's
/// pictures: Detailed Vision's close-up sections (`tiles_*`) and sampled video
/// frames (`frames_*`), one folder per image or clip, cached for the session.
/// One folder per running copy of the app (by process id): two copies working
/// on the same picture never share a folder, nor delete each other's.
fn ai_temp_dir() -> PathBuf {
    ai_temp_root().join(std::process::id().to_string())
}

fn ai_temp_root() -> PathBuf {
    std::env::temp_dir().join("clarity_tagflow_ai")
}

/// A file in each run's [`ai_temp_dir`], rewritten whenever the run uses it:
/// its age says how long ago that copy of the app last used its folder.
const AI_TEMP_IN_USE: &str = "in_use";

/// `name`'s folder in [`ai_temp_dir`] (created by the caller), marking the
/// run's folder as in use.
#[cfg_attr(not(feature = "llm"), allow(dead_code))] // only the AI worker makes copies
fn ai_temp_subdir(name: &str) -> PathBuf {
    let dir = ai_temp_dir();
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = std::fs::write(dir.join(AI_TEMP_IN_USE), b"");
    }
    dir.join(name)
}

/// Delete this run's temp picture copies (at exit).
pub(crate) fn clear_ai_temp() {
    let _ = std::fs::remove_dir_all(ai_temp_dir());
}

/// At startup, delete the picture copies crashed runs left: other runs'
/// folders not used for an hour (a copy of the app still running marks its
/// folder each time it uses it), and what older builds kept straight in the
/// temp folder. On a thread of its own — the temp folder can hold many
/// thousands of entries, and listing them held up the app's start.
fn clear_stale_ai_temp() {
    let stale = |p: &std::path::Path| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > std::time::Duration::from_secs(3600))
    };
    let _ = std::thread::Builder::new().name("ai-temp-cleanup".into()).spawn(move || {
        let own = std::process::id().to_string();
        for entry in std::fs::read_dir(ai_temp_root()).into_iter().flatten().flatten() {
            if entry.file_name().to_string_lossy() == own {
                continue;
            }
            let dir = entry.path();
            let marker = dir.join(AI_TEMP_IN_USE);
            if stale(if marker.exists() { &marker } else { &dir }) {
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
        for entry in std::fs::read_dir(std::env::temp_dir()).into_iter().flatten().flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let legacy_dir = name.starts_with("clarity_tagflow_ai_tiles_") || name.starts_with("clarity_tagflow_ai_frames_");
            if legacy_dir && stale(&entry.path()) {
                let _ = std::fs::remove_dir_all(entry.path());
            } else if name == "clarity_tagflow_llm_input.png" && stale(&entry.path()) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    });
}

impl Default for LlmState {
    fn default() -> Self {
        // Picture copies a crashed run left in the temp folder.
        clear_stale_ai_temp();
        let (video_poster_tx, video_poster_rx) = std::sync::mpsc::channel();
        // Role play's memories have their own store; an older diary kept in
        // the role-play file moves into it the first time.
        let mut roleplay = crate::roleplay::RoleplayState::load();
        let mut rp_memory = crate::memory::MemoryStore::load(crate::memory::MemoryKind::Roleplay);
        crate::roleplay::migrate_diary(&mut roleplay, &mut rp_memory);
        Self {
            installed: installed(),
            download: None,
            download_err: None,
            chats: vec![Chat::new(0)],
            active_chat: 0,
            gen_chat: None,
            gen_roleplay: false,
            next_chat_id: 1,
            draft: String::new(),
            draft_image: None,
            draft_thumb: None,
            msg_thumbs: std::collections::HashMap::new(),
            video_poster_tx,
            video_poster_rx,
            video_posters_pending: std::collections::HashSet::new(),
            input_rect: None,
            lightbox: None,
            run_err: None,
            running: false,
            gen_started: None,
            unload_when_done: false,
            status: String::new(),
            md_cache: egui_commonmark::CommonMarkCache::default(),
            think_orbs: std::collections::HashMap::new(),
            pending_retry: None,
            editing: None,
            edit_draft: String::new(),
            pending_edit: None,
            tts: None,
            speak_target: None,
            voice: crate::voice::VoiceState::default(),
            dictate: Default::default(),
            roleplay,
            rp_memory,
            tag_images: false,
            tag_target: None,
            tag_requests: Vec::new(),
            model: GemmaModel::default(),
            params: GenParams::default(),
            gen_settings_open: false,
            ai_settings_tab: Default::default(),
            memory: crate::memory::MemoryStore::load(crate::memory::MemoryKind::Main),
            memory_map: Default::default(),
            image_detail: ImageDetail::default(),
            detail_effective: None,
            detail_reason: None,
            detail_notice_done: false,
            auto_speak: false,
            detailed_vision: false,
            krea_prompts: false,
            tools_open: false,
            chats_open: false,
            spell: crate::spellcheck::SpellcheckState::default(),
            reply_rx: None,
            remote_host: None,
            mem_pass: Default::default(),
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
            self.drop_worker();
        }
    }

    /// Change the requested Image detail. The level is fixed when the vision
    /// projector loads, so — like switching models — an idle worker is
    /// dropped and the next ask reloads at the new level.
    pub fn set_image_detail(&mut self, level: ImageDetail) {
        if self.image_detail == level {
            return;
        }
        self.image_detail = level;
        if !self.running && self.model.has_image_detail() {
            self.drop_worker();
        }
    }

    /// Unload the resident worker; its Image detail report goes with it.
    fn drop_worker(&mut self) {
        release_shared_worker();
        self.detail_effective = None;
        self.detail_reason = None;
    }

    /// The chat can answer: this build has the AI and its model is set up —
    /// or a Link host answers for it.
    pub fn usable(&self) -> bool {
        (BUILT_WITH_LLM && self.installed) || crate::link::remote_chat().is_some()
    }

    /// AI Chat was deactivated: free the resident worker — and the model
    /// weights it holds — immediately when idle, or right after the in-flight
    /// reply finishes and the memory passes still due have run, within a
    /// minute and a half (see `unload_when_done`). The next activation
    /// reloads the model on its first ask.
    pub fn unload(&mut self) {
        // Voice and mic stop with the chat: their Stop controls are in it,
        // and nothing polls them while it's hidden — a clip still being
        // synthesized would play when the chat is next shown, and dictation
        // would record on and drop its transcript into the draft later.
        self.voice.stop();
        if let Some(mut c) = self.tts.take() {
            let _ = c.kill();
        }
        self.dictate.cancel();
        self.begin_unload();
    }

    /// AI Chat re-activated before a pending unload fired — keep the worker.
    pub fn cancel_unload(&mut self) {
        self.keep_loaded();
    }

    /// Drive background work — poll the setup download and drain any streamed
    /// tokens from the inference worker. Call once per frame from the tab.
    /// The cached poster still for a video attachment, kicking off a capture
    /// on a background thread on first sight — or, for an image, its
    /// downscaled still. `None` while that runs (the chat shows a filename
    /// chip meanwhile) and for files that can't be decoded (cached as such,
    /// so they aren't retried every frame — see [`Self::thumb_failed`]).
    pub fn media_thumb(
        &mut self,
        ctx: &egui::Context,
        path: &std::path::Path,
    ) -> Option<egui::TextureHandle> {
        if let Some(t) = self.msg_thumbs.get(path) {
            return t.clone();
        }
        if self.video_posters_pending.insert(path.to_path_buf()) {
            let tx = self.video_poster_tx.clone();
            let ctx = ctx.clone();
            let path = path.to_path_buf();
            std::thread::spawn(move || {
                // Stills decode here too: a big photo takes long enough to
                // visibly freeze the UI thread.
                let img = if crate::is_video(&path) {
                    crate::video::capture_poster(&path, 512)
                } else {
                    crate::image_cache::open_oriented(&path).ok().map(|img| {
                        let t = img.thumbnail(512, 512).to_rgba8();
                        egui::ColorImage::from_rgba_unmultiplied(
                            [t.width() as usize, t.height() as usize],
                            t.as_raw(),
                        )
                    })
                };
                let _ = tx.send((path, img));
                ctx.request_repaint();
            });
        }
        None
    }

    /// The attachment's thumbnail was attempted and the file couldn't be
    /// decoded (as opposed to still loading).
    pub fn thumb_failed(&self, path: &std::path::Path) -> bool {
        matches!(self.msg_thumbs.get(path), Some(None))
    }

    pub fn poll(&mut self, ctx: &egui::Context) {
        self.voice.poll();
        // Poster stills captured for video attachments become textures.
        while let Ok((path, img)) = self.video_poster_rx.try_recv() {
            let tex = img.map(|img| {
                ctx.load_texture(
                    format!("ai-chat-video-{}", path.display()),
                    img,
                    Default::default(),
                )
            });
            self.video_posters_pending.remove(&path);
            self.msg_thumbs.insert(path, tex);
        }
        if let Some(dl) = &self.download
            && dl.done() {
                if dl.ok() {
                    self.installed = self.model.installed();
                } else {
                    self.download_err =
                        Some(dl.error().unwrap_or_else(|| "Download failed".to_string()));
                }
                self.download = None;
            }

        // The loaded model's Image detail, reported when it loaded (perhaps
        // for a Link device's question).
        if let Some((effective, reason)) = DETAIL_REPORT.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            self.detail_effective = effective;
            self.detail_reason = reason;
        }
        // A question waiting behind a Link device's: say so.
        if self.running && self.status == "Starting…" && REMOTE_BUSY.load(Ordering::SeqCst) > 0 {
            self.status = "Waiting — another PC is using the AI…".to_string();
        }

        // Drain first, then act — acting (auto-speak) needs &mut self.
        let mut worker_msgs = Vec::new();
        let mut worker_died = false;
        if let Some(rx) = &self.reply_rx {
            loop {
                match rx.try_recv() {
                    Ok(m) => worker_msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        worker_died = true;
                        break;
                    }
                }
            }
        }
        // The reply ended without a word (its worker stopped).
        if worker_died && self.running && !worker_msgs.iter().any(|m| matches!(m, Msg::Done(_))) {
            worker_msgs.push(Msg::Done(Err("The AI worker stopped — try again.".to_string())));
        }
        {
            for msg in worker_msgs {
                match msg {
                    // Once Stop is pressed, keep saying so: the worker's later
                    // progress lines would read as if the reply carried on.
                    Msg::Status(s) if !(self.running && GEN_CANCEL.load(Ordering::SeqCst)) => self.status = s,
                    Msg::Status(_) => {}
                    Msg::ImageDetail { effective, reason } => {
                        self.detail_effective = effective;
                        self.detail_reason = reason;
                    }
                    Msg::Token(t) => {
                        // Stream into the reply bubble of the chat that asked,
                        // even if the user switched to another chat meanwhile.
                        if let Some(id) = self.gen_chat
                            && let Some(m) = self
                                .chats
                                .iter_mut()
                                .find(|c| c.id == id)
                                .and_then(|c| c.msgs.last_mut())
                            {
                                // The "Thinking… X s" counter starts with the
                                // reply itself, not the model load and the
                                // prompt / picture reading before it.
                                if m.text.is_empty() {
                                    self.gen_started = Some(std::time::Instant::now());
                                }
                                m.text.push_str(&t);
                            }
                    }
                    Msg::Done(res) => {
                        // Split the thought channel off the finished reply:
                        // it moves onto the message's `thinking` (for the
                        // disclosure row) and out of the visible text — so it
                        // is never re-sent in the history, spoken, or copied.
                        // Done on errors too, so a reply cut short by a failure
                        // doesn't show raw channel markers.
                        let stopped = GEN_CANCEL.load(Ordering::SeqCst);
                        let cut_off = GEN_CUT_OFF.load(Ordering::SeqCst);
                        if let Some(m) = self
                                .gen_chat
                                .and_then(|id| self.chats.iter_mut().find(|c| c.id == id))
                                .and_then(|c| c.msgs.last_mut())
                            {
                                let (thinking, visible) = split_channels(&m.text);
                                if visible.is_empty() && !thinking.is_empty() && res.is_ok() && !stopped {
                                    // Last resort: the model wrote everything
                                    // inside its head and ignored every nudge
                                    // to say it out loud — better to show the
                                    // thought than an empty bubble. Not for a
                                    // reply stopped mid-thought: its half
                                    // thought stays a thought (never re-sent
                                    // in the history or spoken).
                                    m.text = thinking;
                                } else {
                                    if !thinking.is_empty() {
                                        m.thinking = Some(thinking);
                                    }
                                    m.text = visible;
                                }
                            }
                        // Role play: a finished reply may end in control lines
                        // (diary entries, keep/show a picture) — pull them out
                        // of the visible text and act on them. Whether it's a
                        // role-play reply was settled when it was asked (the
                        // Tools menu may have switched modes since), and, as
                        // with Tag images below, only a reply the model ended
                        // itself is acted on: a stopped, cut-off or failed one
                        // may end mid-line ("MEMORY: Peter is allergic to"),
                        // so its control lines are only hidden.
                        if self.gen_roleplay {
                            let complete = res.is_ok() && !stopped && !cut_off;
                            let mut directives = None;
                            let mut last_user_image = None;
                            if let Some(chat) = self
                                .gen_chat
                                .and_then(|id| self.chats.iter_mut().find(|c| c.id == id))
                            {
                                // Only the message this reply answers — an
                                // older photo must never be kept for words
                                // the user merely typed.
                                last_user_image = chat
                                    .msgs
                                    .iter()
                                    .rev()
                                    .find(|m| m.role == ChatRole::User)
                                    .and_then(|m| m.image.clone());
                                if let Some(m) = chat.msgs.last_mut() {
                                    directives = Some(crate::roleplay::extract_directives(&mut m.text));
                                }
                            }
                            if let Some(d) = directives.filter(|_| complete) {
                                // Into role play's own memory store, filed
                                // under the category the line names.
                                for mem in d.memories {
                                    crate::roleplay::record(&mut self.rp_memory, &mem, true);
                                }
                                // "I love this picture" → encrypted album copy
                                // + a memory that remembers why.
                                if let (Some(why), Some(src)) = (d.keep_image, last_user_image) {
                                    match self.roleplay.save_album_image(&src, &why) {
                                        Ok(name) => {
                                            crate::roleplay::record_picture(&mut self.rp_memory, name, &why);
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
                        // Tag images: a finished reply may end in tag edits —
                        // pull them out of the visible text and hand them to
                        // main.rs for the image the user's message was about.
                        // Only a reply the model ended itself is applied: one
                        // the user stopped, one cut off at Max reply length
                        // or one that failed may have a line cut short, so
                        // its edits are dropped, and the note says why.
                        let tagging = self.tag_images;
                        if let Some(chat) = self
                            .gen_chat
                            .and_then(|id| self.chats.iter_mut().find(|c| c.id == id))
                        {
                            let image = chat
                                .msgs
                                .iter()
                                .rev()
                                .find(|m| m.role == ChatRole::User)
                                .and_then(|m| m.tag_image.clone());
                            let (chat_id, index) = (chat.id, chat.msgs.len().saturating_sub(1));
                            if (tagging || image.is_some())
                                && let Some(m) = chat.msgs.last_mut()
                            {
                                let edits = crate::ai_tagging::extract(&mut m.text);
                                let why = if edits.is_empty() {
                                    None
                                } else if res.is_err() {
                                    Some("the reply failed, so nothing was written")
                                } else if stopped {
                                    Some("the reply was stopped, so nothing was written")
                                } else if cut_off {
                                    Some("the reply hit Max reply length before it ended, so nothing was written")
                                } else if !tagging {
                                    Some("Tag images was switched off, so nothing was written")
                                } else if image.is_none() {
                                    Some("no image was selected")
                                } else {
                                    None
                                };
                                match (why, image) {
                                    (Some(why), image) => {
                                        m.tag_change =
                                            Some(crate::ai_tagging::TagChange::failed(image.unwrap_or_default(), why));
                                    }
                                    (None, Some(image)) if !edits.is_empty() => {
                                        self.tag_requests.push(crate::ai_tagging::TagRequest::Apply {
                                            chat_id,
                                            index,
                                            image,
                                            edits,
                                        });
                                    }
                                    _ => {}
                                }
                            }
                        }
                        // Auto-speak: read the finished reply aloud (after the
                        // diary lines are gone, so they're never spoken) —
                        // unless it was stopped (Stop means stop), or AI Chat
                        // was switched off meanwhile: nothing would be on
                        // screen to stop it with.
                        if res.is_ok() && !stopped && self.auto_speak && !self.unload_when_done {
                            let target = self
                                .gen_chat
                                .and_then(|id| self.chats.iter().find(|c| c.id == id))
                                .and_then(|c| {
                                    c.msgs.last().map(|m| (c.id, c.msgs.len() - 1, m.text.clone()))
                                });
                            if let Some((id, index, t)) = target {
                                self.speak_target = Some((id, index));
                                self.start_speaking(&t, ctx);
                            }
                        }
                        if let Some(id) = self.gen_chat {
                            self.chat_changed(id);
                        }
                        self.running = false;
                        self.gen_chat = None;
                        if let Err(e) = res {
                            self.run_err = Some(e);
                        }
                    }
                }
            }
            // The reply's channel closed: nothing more comes on it.
            if worker_died || !self.running {
                self.reply_rx = None;
                self.remote_host = None;
            }
        }
        // Remembering by itself: take in the memory pass's answer, start the
        // next one when one is due.
        self.tick_memory(ctx);
        // Chat was deactivated; once its reply and the memory passes it left
        // are done, release the worker and free the weights.
        self.finish_unload_if_idle();
    }

    /// Start a fresh conversation and switch to it (the one left behind is
    /// due for its memory pass).
    pub fn new_chat(&mut self) {
        // Reuse the current chat if it's still empty (no stacks of blanks).
        if self.chats[self.active_chat].msgs.is_empty() {
            return;
        }
        self.chats.push(Chat::new(self.next_chat_id));
        self.next_chat_id += 1;
        self.active_chat = self.chats.len() - 1;
    }

    /// Switch to another conversation (the one left behind is due for its
    /// memory pass).
    pub fn select_chat(&mut self, index: usize) {
        if index < self.chats.len() {
            self.active_chat = index;
        }
    }

    /// Delete a conversation (a streaming reply into it is silently dropped,
    /// and what no memory pass read yet is never read).
    /// The list never goes empty — deleting the last chat leaves a fresh one.
    pub fn delete_chat(&mut self, index: usize) {
        if index >= self.chats.len() {
            return;
        }
        let removed = self.chats.remove(index);
        self.cancel_pass_for(removed.id);
        if self.gen_chat == Some(removed.id) {
            self.gen_chat = None;
        }
        if self.editing.is_some_and(|(id, _)| id == removed.id) {
            self.editing = None;
            self.edit_draft.clear();
        }
        self.think_orbs.retain(|(id, _), _| *id != removed.id);
        // Free attachment textures no remaining chat (or the draft) shows.
        let in_use = |p: &PathBuf| {
            self.draft_image.as_ref() == Some(p)
                || self.chats.iter().any(|c| c.msgs.iter().any(|m| m.image.as_ref() == Some(p)))
        };
        let stale: Vec<PathBuf> = self.msg_thumbs.keys().filter(|p| !in_use(p)).cloned().collect();
        for p in stale {
            self.msg_thumbs.remove(&p);
        }
        if self.chats.is_empty() {
            self.chats.push(Chat::new(self.next_chat_id));
            self.next_chat_id += 1;
        }
        // Deleting a chat above the selected one shifts it up a row — follow
        // it, rather than silently switching to its neighbour.
        if index < self.active_chat {
            self.active_chat -= 1;
        }
        self.active_chat = self.active_chat.min(self.chats.len() - 1);
    }

    /// True while a reply is streaming into the chat with this id.
    pub fn streams_into(&self, chat_id: u64) -> bool {
        self.running && self.gen_chat == Some(chat_id)
    }

    /// Send the input pill's draft (text + attached image) as the next user
    /// message of the active chat, spawning the worker thread on first use.
    /// The reply streams into a fresh model message.
    pub fn send_draft(&mut self, ctx: &egui::Context) {
        let text = self.draft.trim().to_string();
        if self.running || (text.is_empty() && self.draft_image.is_none()) {
            return;
        }
        let remote = crate::link::remote_chat().is_some();
        if !BUILT_WITH_LLM && !remote {
            self.run_err =
                Some("This build was compiled without the AI feature (`llm`).".to_string());
            return;
        }
        if !self.installed && !remote {
            self.run_err = Some("Set up the model first (Settings → AI Model).".to_string());
            return;
        }
        self.run_err = None;
        self.status = "Starting…".to_string();
        // The "image detail was lowered" notice has had its moment.
        if self.detail_reason.is_some() {
            self.detail_notice_done = true;
        }

        let image = self.draft_image.take();
        self.draft_thumb = None;
        self.draft.clear();
        // Tag images: the message is about the image on show right now —
        // that's what the AI sees, instead of any picture attached before
        // the mode was switched on.
        let tag_image = if self.tag_images { self.tag_target.clone() } else { None };
        let image = if tag_image.is_some() { None } else { image };
        let remember = self.memory_wanted();
        self.chats[self.active_chat].msgs.push(ChatMsg {
            role: ChatRole::User,
            text,
            thinking: None,
            image,
            tag_image,
            tag_change: None,
            remember,
            shared_with: Vec::new(),
        });
        self.begin_generation(ctx);
    }

    /// The tag edits and undos waiting for main.rs, oldest first.
    pub fn take_tag_requests(&mut self) -> Vec<crate::ai_tagging::TagRequest> {
        std::mem::take(&mut self.tag_requests)
    }

    /// The replies from `from` on in the active chat are about to be
    /// replaced: when any of them changed an image's tags, ask main.rs to
    /// take those changes back first and then resend (`then`). True when it
    /// was asked — the resend then waits for main.rs.
    fn rewind_tags(&mut self, from: usize, then: crate::ai_tagging::Resend) -> bool {
        let chat = &self.chats[self.active_chat];
        // A reply that only just finished may have its edits still queued
        // for main.rs (written at the end of this frame): they count too —
        // the Rewind queued after them takes them back.
        let queued = self.tag_requests.iter().any(|r| {
            matches!(r, crate::ai_tagging::TagRequest::Apply { chat_id, index, .. }
                if *chat_id == chat.id && *index >= from)
        });
        let in_place = queued
            || chat.msgs.iter().skip(from).any(|m| m.tag_change.as_ref().is_some_and(|c| c.changed() && !c.undone));
        if in_place {
            self.tag_requests.push(crate::ai_tagging::TagRequest::Rewind { chat_id: chat.id, from, then });
        }
        in_place
    }

    /// The message at `index` of the chat with id `chat_id`, if both still
    /// exist (a chat may have been deleted, or an edit cut it short).
    pub fn msg_mut(&mut self, chat_id: u64, index: usize) -> Option<&mut ChatMsg> {
        self.chats.iter_mut().find(|c| c.id == chat_id)?.msgs.get_mut(index)
    }

    /// Regenerate the model reply at `msg_index` of the active chat: that
    /// reply — and everything after it — is discarded, and the conversation
    /// up to its user message is resent. Tag images: tag changes those
    /// replies made are taken back first (main.rs does that, then calls
    /// [`retry_now`](Self::retry_now)).
    pub fn retry(&mut self, msg_index: usize, ctx: &egui::Context) {
        if self.running
            || self.chats[self.active_chat].msgs.get(msg_index).map(|m| m.role) != Some(ChatRole::Model)
            || self.rewind_tags(msg_index, crate::ai_tagging::Resend::Retry(msg_index))
        {
            return;
        }
        self.retry_now(msg_index, ctx);
    }

    /// [`retry`](Self::retry), once the replies it discards have no tag
    /// changes left in place.
    pub fn retry_now(&mut self, msg_index: usize, ctx: &egui::Context) {
        if self.running {
            return;
        }
        let chat = &mut self.chats[self.active_chat];
        if chat.msgs.get(msg_index).map(|m| m.role) != Some(ChatRole::Model) {
            return;
        }
        chat.msgs.truncate(msg_index);
        // The new reply is read by the memory pass in its turn.
        chat.memory_synced = chat.memory_synced.min(msg_index);
        let id = chat.id;
        self.close_cut_editor(id, msg_index);
        let chat = &self.chats[self.active_chat];
        if chat.msgs.last().map(|m| m.role) != Some(ChatRole::User) {
            return;
        }
        self.run_err = None;
        self.status = "Starting…".to_string();
        self.begin_generation(ctx);
    }

    /// Apply an edited user message: it keeps its attached image, gets the
    /// new text, everything after it is discarded, and a fresh reply is
    /// generated — like `retry`, but with rewritten words (and, like it,
    /// taking back the discarded replies' tag changes first).
    pub fn apply_edit(&mut self, msg_index: usize, text: String, ctx: &egui::Context) {
        if self.running {
            return;
        }
        let msgs = &self.chats[self.active_chat].msgs;
        if msgs.get(msg_index).map(|m| m.role) != Some(ChatRole::User)
            || (text.trim().is_empty() && msgs[msg_index].image.is_none())
            || self.rewind_tags(msg_index + 1, crate::ai_tagging::Resend::Edit(msg_index, text.clone()))
        {
            return;
        }
        self.apply_edit_now(msg_index, text, ctx);
    }

    /// [`apply_edit`](Self::apply_edit), once the replies it discards have
    /// no tag changes left in place.
    pub fn apply_edit_now(&mut self, msg_index: usize, text: String, ctx: &egui::Context) {
        if self.running {
            return;
        }
        let remember = self.memory_wanted();
        let chat = &mut self.chats[self.active_chat];
        if chat.msgs.get(msg_index).map(|m| m.role) != Some(ChatRole::User) {
            return;
        }
        let text = text.trim().to_string();
        if text.is_empty() && chat.msgs[msg_index].image.is_none() {
            return;
        }
        chat.msgs.truncate(msg_index + 1);
        chat.msgs[msg_index].text = text;
        // Sent again, as it now reads: the memory pass reads it afresh.
        chat.msgs[msg_index].remember = remember;
        chat.memory_synced = chat.memory_synced.min(msg_index);
        let id = chat.id;
        self.close_cut_editor(id, msg_index + 1);
        self.run_err = None;
        self.status = "Starting…".to_string();
        self.begin_generation(ctx);
    }

    /// Chat `chat_id` was cut back to its first `len` messages: an editor
    /// open on a message that's gone closes. Left open it would reappear on
    /// whatever lands at that index next (the next message sent), showing
    /// the old text, and Enter would overwrite that message with it.
    fn close_cut_editor(&mut self, chat_id: u64, len: usize) {
        if self.editing.is_some_and(|(id, i)| id == chat_id && i >= len) {
            self.editing = None;
            self.edit_draft.clear();
        }
    }

    /// Generate a reply to the active chat's history (which must end with a
    /// user message): snapshot the capped history, send it to the worker, and
    /// push the empty reply bubble the tokens stream into.
    fn begin_generation(&mut self, ctx: &egui::Context) {
        // A reply never waits for a memory pass: it stops (and runs later).
        self.cancel_pass();
        let chat = &mut self.chats[self.active_chat];

        // Snapshot the history for the worker, capped to the most recent
        // messages so a very long chat doesn't overflow the model's context,
        // and with only the newest few images attached (see HISTORY_IMAGES).
        // Empty replies (a failed or instantly-stopped generation) are left
        // out — an empty assistant turn only teaches the model to say nothing
        // — and so is any reply the cap leaves at the front, so the window
        // (and the role-play priming pair before it) always opens on a user
        // turn.
        let kept: Vec<&ChatMsg> = chat
            .msgs
            .iter()
            .filter(|m| m.role == ChatRole::User || !m.text.trim().is_empty())
            .collect();
        let window: Vec<&ChatMsg> = kept[kept.len().saturating_sub(HISTORY_MSGS)..]
            .iter()
            .skip_while(|m| m.role == ChatRole::Model)
            .copied()
            .collect();
        // Where the window starts in the chat: everything from there on goes
        // to a Link host when one answers.
        let window_from = window
            .first()
            .and_then(|first| chat.msgs.iter().position(|m| std::ptr::eq(m, *first)))
            .unwrap_or(chat.msgs.len());
        let newest_user = window.iter().rposition(|m| m.role == ChatRole::User);
        let mut msgs: Vec<CmdMsg> = window
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let user = m.role == ChatRole::User;
                let mut text = m.text.clone();
                // Only the user's images go through the vision encoder —
                // album pictures the AI showed back are display-only.
                let mut image = if user { m.image.clone() } else { None };
                // Tag images: the newest message carries the image it's
                // about as its vision input, plus that image's tags read
                // fresh from its .txt (hand edits since count); older ones
                // only say which image they were about.
                if let Some(tag_image) = m.tag_image.as_ref().filter(|_| user) {
                    if Some(i) == newest_user {
                        let tags = crate::tag_manager::read_tags(tag_image);
                        text = format!("{}\n\n{text}", crate::ai_tagging::context(tag_image, tags.as_deref()));
                        image = Some(tag_image.clone());
                    } else {
                        text = format!("{}\n{text}", crate::ai_tagging::label(tag_image));
                    }
                }
                CmdMsg { user, text, image }
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
        // Videos are the priciest attachment (each re-send costs a whole set
        // of sampled frames through the vision encoder), so only the newest
        // one stays in the window; older messages keep their text.
        let mut kept_video = false;
        for m in msgs.iter_mut().rev() {
            if m.image.as_deref().is_some_and(crate::is_video) {
                if kept_video {
                    m.image = None;
                } else {
                    kept_video = true;
                }
            }
        }
        let id = chat.id;

        // Role play: pin the persona/diary priming turn in front of the
        // history — it never slides out of the window, so the character (and
        // everything in the diary) survives arbitrarily long sessions.
        let mut pinned = 0;
        self.gen_roleplay = !self.krea_prompts && self.roleplay.enabled;
        if self.krea_prompts {
            // The Krea 2 brief is pinned the same way (the tools menu keeps
            // it and role play mutually exclusive).
            msgs.insert(0, CmdMsg { user: false, text: crate::krea_prompt::ACK.to_string(), image: None });
            msgs.insert(0, CmdMsg { user: true, text: crate::krea_prompt::BRIEF.to_string(), image: None });
            pinned = 2;
            // Remember across chats recalls here too (nothing is saved): its
            // note goes first, so the brief stays nearest the chat.
            if let Some([note, ack]) = self.memory_note() {
                msgs.insert(0, ack);
                msgs.insert(0, note);
                pinned = 4;
            }
        } else if self.roleplay.enabled {
            msgs.insert(0, CmdMsg { user: false, text: self.roleplay.ack(), image: None });
            // High / Max thinking lifts the role-play "think briefly" rule
            // (the model that answers: a Link host's, when it runs there).
            let thinks = if crate::link::remote_chat().is_some() {
                crate::link::remote_chat_thinks()
            } else {
                self.model.thinks()
            };
            let deep = thinks && matches!(self.params.thinking, ThinkLevel::High | ThinkLevel::Max);
            msgs.insert(
                0,
                CmdMsg { user: true, text: self.roleplay.preamble_with(&self.rp_memory, deep), image: None },
            );
            pinned = 2;
        } else if self.tag_images {
            // The tagging brief: how to see the image and change its tags.
            msgs.insert(0, CmdMsg { user: false, text: crate::ai_tagging::ACK.to_string(), image: None });
            msgs.insert(0, CmdMsg { user: true, text: crate::ai_tagging::BRIEF.to_string(), image: None });
            pinned = 2;
        } else if let Some([note, ack]) = self.memory_note() {
            // Remember across chats: the memory note (the AI's notes, the
            // last chat, its topics and the memories that may matter now),
            // pinned the same way and picked afresh for every reply.
            msgs.insert(0, ack);
            msgs.insert(0, note);
            pinned = 2;
        }
        // The reply may go into memory when the message it answers may.
        let remember = self.memory_wanted()
            && self.chats[self.active_chat].msgs.last().is_some_and(|m| m.role == ChatRole::User && m.remember);

        GEN_CANCEL.store(false, Ordering::SeqCst);
        GEN_CUT_OFF.store(false, Ordering::SeqCst);
        let detailed = self.detailed_vision || self.krea_prompts;
        let (tx, rx) = std::sync::mpsc::channel();
        // A Link host answers when this PC is linked to one sharing its
        // chat; otherwise the resident worker (a different model selected
        // since it loaded is swapped in first).
        let mut shared_with = None;
        let asked = match crate::link::remote_chat() {
            Some((peer, host)) => {
                let ctx = ctx.clone();
                let params = self.params;
                self.remote_host = Some(host.clone());
                shared_with = Some(memory_pass::host_id(&peer));
                std::thread::Builder::new()
                    .name("ai-remote-ask".into())
                    .spawn(move || remote_ask(&peer, &host, msgs, params, pinned, detailed, None, &tx, &ctx))
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            }
            None => {
                self.remote_host = None;
                submit(
                    ctx,
                    self.model,
                    self.image_detail,
                    Cmd::Ask { msgs, params: self.params, pinned, detailed, reply: tx, remote: None },
                )
            }
        };
        match asked {
            Ok(()) => {
                self.reply_rx = Some(rx);
                // The empty reply bubble the tokens stream into.
                self.chats[self.active_chat]
                    .msgs
                    .push(ChatMsg {
                        role: ChatRole::Model,
                        text: String::new(),
                        thinking: None,
                        image: None,
                        tag_image: None,
                        tag_change: None,
                        remember,
                        shared_with: Vec::new(),
                    });
                self.running = true;
                self.gen_started = Some(std::time::Instant::now());
                self.gen_chat = Some(id);
                self.chat_changed(id);
                // What went to a Link host — the window and the reply it
                // writes — may go there for its memory pass too.
                if let Some(host) = shared_with {
                    let chat = &mut self.chats[self.active_chat];
                    let from = window_from.min(chat.msgs.len());
                    for m in &mut chat.msgs[from..] {
                        if !m.shared_with.contains(&host) {
                            m.shared_with.push(host.clone());
                        }
                    }
                }
            }
            Err(e) => {
                self.remote_host = None;
                self.run_err = Some(e);
            }
        }
    }

    /// The chat's Stop button: ask the worker to end the streaming reply.
    /// It finishes the token in flight and reports Done, so whatever streamed
    /// so far stays in the bubble as the (partial) reply.
    pub fn stop_generation(&mut self) {
        if self.running {
            GEN_CANCEL.store(true, Ordering::SeqCst);
            self.status = "Stopping…".into();
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

    /// True while either voice backend is playing (or OmniVoice is still
    /// synthesizing a reply — Listen means Stop then too).
    pub fn any_speaking(&mut self) -> bool {
        self.voice.busy() || self.speaking()
    }

    /// Read a reply aloud — or stop the current playback if one is speaking.
    /// Uses OmniVoice (the natural neural voice, Python sidecar) once its
    /// one-click setup has run; otherwise falls back to the OS speech
    /// synthesizer (Windows).
    pub fn listen(&mut self, text: &str, ctx: &egui::Context) {
        if self.voice.busy() {
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
            // The text goes in as UTF-8 and is read back as UTF-8 explicitly:
            // [Console]::In decodes with the console code page, which turned
            // "é" into "Ã©" and curly quotes or "…" into junk.
            let child = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    "Add-Type -AssemblyName System.Speech; \
                     $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                     $r = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), [System.Text.Encoding]::UTF8); \
                     $s.Speak($r.ReadToEnd())",
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

/// Split a model reply into its thought channel and the visible answer.
///
/// The big Gemma 4 (the 31B) reasons before answering on the wire:
/// `<|channel>thought\n{thinking}<channel|>{answer}` — `<|channel>NAME`
/// switches into a named channel, a bare `<channel|>` switches back to the
/// visible answer. Channels named thought/thinking/analysis collect into the
/// returned thinking text; anything else (final, …) stays visible, as does
/// text outside any channel (the E4B never emits channels at all). Qwen's
/// thinking model uses `<think>{thinking}</think>{answer}` instead — same
/// split, different markers. Safe on a mid-stream prefix: a trailing
/// half-written marker is hidden so it never flashes up while tokens arrive.
pub fn split_channels(raw: &str) -> (String, String) {
    const START: &str = "<|channel>";
    const END: &str = "<channel|>";
    const THINK: &str = "<think>";
    const THINK_END: &str = "</think>";

    // Hide a trailing, still-streaming partial marker.
    let mut raw = raw;
    if let Some(i) = raw.rfind('<') {
        let tail = &raw[i..];
        if !tail.contains('>')
            && [START, END, THINK, THINK_END].iter().any(|m| m.starts_with(tail))
        {
            raw = &raw[..i];
        }
    }

    let mut thinking = String::new();
    let mut visible = String::new();
    let mut in_thought = false;
    let mut rest = raw;
    loop {
        // The earliest marker of any kind decides where this segment ends.
        // Kinds: 0 = named channel start, 1 = think-open, 2 = close (either).
        let next = [(START, 0), (THINK, 1), (END, 2), (THINK_END, 2)]
            .iter()
            .filter_map(|&(m, kind)| rest.find(m).map(|i| (i, m.len(), kind)))
            .min_by_key(|&(i, _, _)| i);
        let Some((i, marker_len, kind)) = next else { break };
        let seg = &rest[..i];
        if in_thought { thinking.push_str(seg) } else { visible.push_str(seg) }
        rest = &rest[i + marker_len..];
        match kind {
            0 => {
                // The channel name is the word right after the marker.
                let name_len = rest
                    .find(|c: char| c.is_whitespace() || c == '<')
                    .unwrap_or(rest.len());
                let name = rest[..name_len].to_ascii_lowercase();
                in_thought = name.starts_with("thought")
                    || name.starts_with("think")
                    || name == "analysis";
                rest = &rest[name_len..];
            }
            1 => in_thought = true,
            _ => in_thought = false,
        }
        // Don't let a marker leave the next section starting on a blank line.
        rest = rest.strip_prefix('\n').unwrap_or(rest);
    }
    if in_thought { thinking.push_str(rest) } else { visible.push_str(rest) }
    (thinking.trim().to_string(), visible.trim().to_string())
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
/// omitted"), inline markers (** * ` #) are dropped, emoji/symbols are
/// removed — voices either mispronounce them or, in OmniVoice's frontend,
/// can outright fail on them — and dashes (— – --) become commas, since
/// voices read them aloud as "dash" instead of pausing. In-word hyphens
/// ("well-known") are left alone.
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
        // A bullet's leading dash would be read as "dash" — drop the marker.
        let line = line.strip_prefix("- ").unwrap_or(line);
        let cleaned: String = line
            .chars()
            .filter(|c| !matches!(c, '*' | '`') && !is_unspeakable(*c))
            .collect();
        let cleaned = cleaned.replace("--", "—").replace(['—', '–'], ", ");
        let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
        let cleaned = cleaned.replace(" ,", ",").replace(",,", ",");
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
fn spawn_worker(
    _ctx: egui::Context,
    _model: GemmaModel,
    _detail: ImageDetail,
) -> Option<(Sender<Cmd>, std::thread::JoinHandle<()>)> {
    None
}

/// Start a worker for `model`; it loads on its first question, once any
/// worker let go before it has freed its memory. (Every question it's given
/// brings its own reply channel, so its own channel's receiver isn't kept.)
#[cfg(feature = "llm")]
fn spawn_worker(
    ctx: egui::Context,
    model: GemmaModel,
    detail: ImageDetail,
) -> Option<(Sender<Cmd>, std::thread::JoinHandle<()>)> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (msg_tx, _) = std::sync::mpsc::channel::<Msg>();
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    let thread = std::thread::Builder::new()
        .name("ai-worker".into())
        .spawn(move || {
            // Let the previous model free its memory first (it stops once its
            // command channel is gone, after any reply it's still writing).
            let t0 = std::time::Instant::now();
            while retired.iter().any(|t| !t.is_finished()) && t0.elapsed().as_secs() < 30 {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            worker::run(cmd_rx, msg_tx, ctx, model, detail)
        })
        .ok()?;
    Some((cmd_tx, thread))
}

#[cfg(feature = "llm")]
mod worker {
    use std::ffi::CString;
    use std::num::NonZeroU32;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::{Receiver, Sender};

    use eframe::egui;
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
    use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::logit_bias::LlamaLogitBias;

    use super::{Cmd, Msg};

    /// Context window. Gemma 4 supports far more, but 16k keeps the KV cache
    /// reasonable on ordinary machines while fitting the 50-message history
    /// cap plus a long reply.
    const N_CTX: u32 = 16384;
    /// Prompt-eval batch size at Standard image detail; must fit an image's
    /// token chunk (280 for Gemma 4 by default).
    const N_BATCH: u32 = 512;
    /// The batch a level runs with. Gemma 4's vision encoder uses non-causal
    /// attention, so an image's whole token chunk must fit one micro-batch —
    /// at 512, High (560) and Maximum (1120) fail with "non-causal attention
    /// requires n_ubatch >= n_tokens". Each level gets just enough: the
    /// compute buffer grows with the batch (the 31B needs 3.1 GB at 2048 vs
    /// ~1.9 GB at 1280), and Standard keeps today's 512 exactly.
    fn batch_for(level: super::ImageDetail) -> u32 {
        match level {
            super::ImageDetail::Standard => N_BATCH,
            super::ImageDetail::High => 640,
            super::ImageDetail::Maximum => 1280,
        }
    }

    /// The context configuration used for every generation — also created
    /// once as a probe during loading, so the GPU-offload ladder knows the
    /// KV cache actually fits alongside the weights. `batch` sets both the
    /// logical and the micro batch.
    fn context_params(threads: i32, batch: u32) -> LlamaContextParams {
        LlamaContextParams::default()
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_n_batch(batch)
            .with_n_ubatch(batch)
            .with_n_ctx(NonZeroU32::new(N_CTX))
    }

    /// Held by a worker for as long as it has llama.cpp loaded. llama-cpp-2
    /// allows one backend at a time, and a worker the chat has just replaced
    /// (another model, or Image detail changed during a reply) is still
    /// freeing its weights when the next one starts — which then failed with
    /// "BackendAlreadyInitialized". The new worker waits here instead, which
    /// also keeps the old weights' GPU memory from crowding out the new load.
    static LOADED: std::sync::Mutex<()> = std::sync::Mutex::new(());

    thread_local! {
        /// The question being answered on this worker thread, when it's a
        /// Link device's: `stopped` and the cut-off mark use its own flags.
        static JOB: std::cell::RefCell<Option<super::RemoteFlags>> = const { std::cell::RefCell::new(None) };
    }

    /// Worker body: load everything on the first question, then serve
    /// questions until every sender is gone. Whoever asked first hears the
    /// loading progress; on a load failure everyone waiting hears the error
    /// and the thread exits (the next question starts a new worker).
    pub(super) fn run(
        rx: Receiver<Cmd>,
        tx: Sender<Msg>,
        ctx: egui::Context,
        model: super::GemmaModel,
        detail: super::ImageDetail,
    ) {
        let base = |m: Msg| {
            let _ = tx.send(m);
            ctx.request_repaint();
        };
        // (A worker that panicked released everything while unwinding.)
        let _loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(first) = rx.recv() else { return };
        let first_reply = match &first {
            Cmd::Ask { reply, .. } | Cmd::Pass { reply, .. } => Some(reply.clone()),
            Cmd::Generate { .. } => None,
        };
        let load_send = |m: Msg| {
            if let Msg::ImageDetail { effective, reason } = &m {
                *super::DETAIL_REPORT.lock().unwrap_or_else(|e| e.into_inner()) = Some((*effective, reason.clone()));
            }
            match &first_reply {
                Some(r) => {
                    let _ = r.send(m);
                    ctx.request_repaint();
                }
                None => base(m),
            }
        };
        let loaded = match load(&load_send, model, detail) {
            Ok(l) => l,
            Err(e) => {
                load_send(Msg::Done(Err(e.clone())));
                while let Ok(cmd) = rx.try_recv() {
                    match cmd {
                        Cmd::Ask { reply, .. } | Cmd::Pass { reply, .. } => {
                            let _ = reply.send(Msg::Done(Err(e.clone())));
                        }
                        Cmd::Generate { .. } => base(Msg::Done(Err(e.clone()))),
                    }
                }
                ctx.request_repaint();
                return;
            }
        };
        serve(&loaded, first, &base, &ctx);
        while let Ok(cmd) = rx.recv() {
            serve(&loaded, cmd, &base, &ctx);
        }
    }

    /// Everything a loaded worker keeps resident.
    struct Loaded {
        backend: LlamaBackend,
        model: LlamaModel,
        mtmd: MtmdContext,
        template: llama_cpp_2::model::LlamaChatTemplate,
        threads: i32,
        which: super::GemmaModel,
        detail: super::ImageDetail,
    }

    /// Answer one question.
    fn serve(l: &Loaded, cmd: Cmd, base: &dyn Fn(Msg), ctx: &egui::Context) {
        match cmd {
            Cmd::Generate { msgs, params, pinned, detailed } => {
                let res = generate(
                    &l.backend, &l.model, &l.mtmd, &l.template, l.threads, l.which, &msgs, pinned, detailed,
                    l.detail, &params, base,
                );
                base(Msg::Done(res));
            }
            Cmd::Ask { msgs, params, pinned, detailed, reply, remote } => {
                let send = |m: Msg| {
                    let _ = reply.send(m);
                    ctx.request_repaint();
                };
                JOB.with(|j| *j.borrow_mut() = remote);
                let res = generate(
                    &l.backend, &l.model, &l.mtmd, &l.template, l.threads, l.which, &msgs, pinned, detailed,
                    l.detail, &params, &send,
                );
                JOB.with(|j| *j.borrow_mut() = None);
                send(Msg::Done(res));
            }
            Cmd::Pass { head, lines, params, reply, cancel } => {
                // Collected, never shown: only the end needs a frame.
                let res = run_pass(l, &head, &lines, &params, &cancel, &|m| {
                    let _ = reply.send(m);
                });
                let _ = reply.send(Msg::Done(res));
                ctx.request_repaint();
            }
        }
    }

    fn load(send: &dyn Fn(Msg), which: super::GemmaModel, requested: super::ImageDetail) -> Result<Loaded, String> {
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
        // ladder of attempts, because the big variants (27B/31B) don't fit
        // whole in common VRAM sizes: everything → about half → CPU only.
        // Each rung must survive the WHOLE pipeline — weights, vision
        // projector, and a probe context — because weights fitting is not
        // enough: a big model can load into ~20 GB of VRAM and llama.cpp then
        // returns a null context when the KV cache no longer fits.
        // Image detail asked for on this load — dropped to Standard (with the
        // reason kept) if the checks leave no room for a reply; see below.
        let mut req = requested;
        let mut carried_reason: Option<String> = None;
        let (model, mtmd, detail) = loop {
            let mut loaded = None;
            let mut last_err = String::new();
            // Which rung succeeded: 0 = the whole model on the GPU.
            let mut rung = 0;
            for (i, layers) in offload_rungs(&model_path, &mmproj_path) {
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
                let mtmd = match load_mtmd(&mmproj_path, &model, threads, -1) {
                    Ok(m) => m,
                    Err(e) => {
                        last_err = format!("Load vision projector: {e}");
                        continue;
                    }
                };
                // Probe context, created and dropped: proves generation will have
                // room for the KV cache with this many layers on the GPU.
                if let Err(e) = model.new_context(&backend, context_params(threads, N_BATCH)) {
                    last_err = format!("Create context: {e}");
                    continue;
                }
                loaded = Some((model, mtmd));
                rung = i;
                break;
            }
            let Some((model, mtmd)) = loaded else {
                return Err(last_err);
            };
            if !which.has_image_detail() {
                send(Msg::ImageDetail { effective: None, reason: None });
                break (model, mtmd, super::ImageDetail::Standard);
            }

            // Image detail: the requested level passes the GPU checks; the
            // projector comes back loaded at the level that's safe to run.
            let checked = decide_detail(req, rung, &backend, &model, mtmd, &mmproj_path, threads, send);
            // Safety net: a reply's context must still fit. Failed probe
            // allocations can leave Vulkan memory behind (seen on the 31B), so
            // prove it — and if it doesn't fit, reload clean at Standard.
            if let Ok((level, reason, mtmd)) = checked {
                let reason = reason.or_else(|| carried_reason.take());
                if req == super::ImageDetail::Standard
                    || model.new_context(&backend, context_params(threads, batch_for(level))).is_ok()
                {
                    send(Msg::ImageDetail { effective: Some(level), reason });
                    break (model, mtmd, level);
                }
                carried_reason = reason.or_else(|| {
                    Some(format!(
                        "Running at Standard: {} image detail left no room for replies on this GPU.",
                        req.label()
                    ))
                });
            } else if let Err(e) = checked {
                carried_reason = Some(format!("Running at Standard: higher image detail failed on this GPU ({e})."));
            }
            eprintln!("[ai] image detail: reloading at Standard so replies have room");
            send(Msg::Status("Reloading the model at Standard image detail…".into()));
            req = super::ImageDetail::Standard;
            drop(model);
        };

        let template = model
            .chat_template(None)
            .map_err(|e| format!("Chat template: {e}"))?;

        send(Msg::Status("Ready".into()));
        Ok(Loaded { backend, model, mtmd, template, threads, which, detail })
    }

    #[allow(clippy::too_many_arguments)] // threads the worker's resident state
    fn generate(
        backend: &LlamaBackend,
        model: &LlamaModel,
        mtmd: &MtmdContext,
        template: &llama_cpp_2::model::LlamaChatTemplate,
        threads: i32,
        which: super::GemmaModel,
        msgs: &[super::CmdMsg],
        pinned: usize,
        detailed: bool,
        detail: super::ImageDetail,
        params: &super::GenParams,
        send: &dyn Fn(Msg),
    ) -> Result<(), String> {
        // Stop pressed while the model was still loading: nothing to do.
        if stopped() {
            return Ok(());
        }
        // A fresh context per turn: cheap next to the resident weights, and it
        // guarantees no KV-cache state leaks — the whole (capped) history is
        // re-evaluated, which is what makes multi-turn chat work.
        let mut context = model
            .new_context(backend, context_params(threads, batch_for(detail)))
            .map_err(|e| format!("Create context: {e}"))?;

        // Each turn keeps its own bitmaps, so trimming old turns below drops
        // their images along with their text.
        let mut turns: Vec<(bool, String, Vec<MtmdBitmap>)> = Vec::new();
        // Detailed Vision tiles only the newest still image; older ones go
        // as the full picture, so a long chat doesn't fill up with tiles.
        let tile_at = detailed
            .then(|| msgs.iter().rposition(|m| m.image.as_deref().is_some_and(|p| !crate::is_video(p))))
            .flatten();
        for (mi, m) in msgs.iter().enumerate() {
            let (text, bitmaps) = match &m.image {
                None => (m.text.clone(), Vec::new()),
                Some(path) => match read_media(mtmd, path, &m.text, tile_at == Some(mi), detail, send) {
                    Ok(read) => read,
                    Err(_) if stopped() => return Ok(()),
                    // An attachment from an earlier turn that can't be read
                    // any more (moved or deleted since it was sent) is left
                    // out, with a note that it was there: failing on it would
                    // fail every later message of the chat, until newer
                    // images pushed it out of the history. Only this turn's
                    // own attachment fails the reply.
                    Err(e) if mi + 1 < msgs.len() => {
                        eprintln!("[ai] left out an earlier attachment ({}): {e}", path.display());
                        let what = if crate::is_video(path) { "video" } else { "image" };
                        (format!("[{what} no longer available]\n{}", m.text), Vec::new())
                    }
                    Err(e) => return Err(e),
                },
            };
            turns.push((m.user, text, bitmaps));
        }

        send(Msg::Status("Thinking…".into()));
        // The wire-format prompt for a run of turns (re-run while trimming).
        let format_prompt = |turns: &[&(bool, String, Vec<MtmdBitmap>)]| -> Result<String, String> {
            let turns: Vec<(bool, &str)> = turns.iter().map(|(user, text, _)| (*user, text.as_str())).collect();
            wire_prompt(model, template, which, params.thinking, &turns)
        };

        // Fit the context: the prompt plus the reply budget must stay within
        // N_CTX, or llama.cpp fails the decode — and since the history only
        // grows, every later message in the chat would fail too. Drop the
        // oldest unpinned turns (never the newest) until it fits, keeping the
        // window opening on a user turn.
        let n_ctx = N_CTX as usize;
        // Room a reply needs beyond the answer: the thought, the 128-token
        // grace window before a forced close, and the ~64-token close text.
        let thinks = which.thinks() && params.thinking != super::ThinkLevel::Off;
        let think_reserve = if thinks { params.thinking.budget() + 192 } else { 0 };
        let pinned = pinned.min(turns.len());
        let mut start = pinned;
        let chunks = loop {
            let kept: Vec<&(bool, String, Vec<MtmdBitmap>)> =
                turns[..pinned].iter().chain(&turns[start..]).collect();
            let input = MtmdInputText {
                text: format_prompt(&kept)?,
                add_special: true,
                parse_special: true,
            };
            let bitmap_refs: Vec<&MtmdBitmap> = kept.iter().flat_map(|t| t.2.iter()).collect();
            let chunks = mtmd
                .tokenize(input, &bitmap_refs)
                .map_err(|e| format!("Tokenize: {e}"))?;
            let prompt = chunks.total_tokens();
            if prompt + params.max_tokens + think_reserve <= n_ctx || start + 1 >= turns.len() {
                // Only the newest message left: run it with whatever reply
                // room remains, unless the message alone is too big.
                if prompt + 256 > n_ctx {
                    return Err(format!(
                        "This message is too long for the model ({prompt} tokens, the limit \
                         is {n_ctx}) — try a shorter message or a smaller attachment."
                    ));
                }
                break chunks;
            }
            start += 1;
            while start + 1 < turns.len() && !turns[start].0 {
                start += 1;
            }
        };
        // Last chance to stop before the prompt evaluation, which runs to the
        // end once started (minutes for a long chat on a model partly on the
        // CPU — llama-cpp-2 evaluates all chunks in one call).
        if stopped() {
            return Ok(());
        }
        let mut n_past = chunks
            .eval_chunks(mtmd, &context, 0, 0, batch_for(detail) as i32, true)
            .map_err(|e| format!("Evaluate prompt: {e}"))?;

        // The user's sampling knobs (defaults are Gemma's recommendation:
        // top-k 64, top-p 0.95, temperature 1.0). Temperature ~0 means "always
        // the most likely token" — plain greedy, no need to sample.
        let greedy = params.temperature <= 0.01;
        // Gemma reasons in `<|channel>thought … <channel|>`, Qwen's thinking
        // model in `<think> … </think>` — same budget machinery, different
        // markers.
        let (thought_start, thought_end) = match which {
            super::GemmaModel::Qwen30B => ("<think>", "</think>"),
            _ => ("<|channel>", "<channel|>"),
        };
        // Once the thought is closed its markers can't come again: greedy
        // decoding otherwise loops on the closing marker after a forced close
        // ("<channel|>\n" up to the length cap — an empty-looking answer).
        let marker_bans: Vec<LlamaLogitBias> = [thought_start, thought_end]
            .iter()
            .filter_map(|m| match model.str_to_token(m, AddBos::Never) {
                Ok(t) if t.len() == 1 => Some(LlamaLogitBias::new(t[0], f32::NEG_INFINITY)),
                _ => None,
            })
            .collect();
        let n_vocab = model.n_vocab();
        // `temperature: None` = greedy; `ban` = the thought's markers are out.
        let sampler_for = |temperature: Option<f32>, ban: bool| {
            let mut chain = Vec::new();
            if ban {
                chain.push(LlamaSampler::logit_bias(n_vocab, &marker_bans));
            }
            match temperature {
                None => chain.push(LlamaSampler::greedy()),
                Some(t) => chain.extend([
                    LlamaSampler::top_k(params.top_k.max(1)),
                    LlamaSampler::top_p(params.top_p.clamp(0.05, 1.0), 1),
                    LlamaSampler::temp(t),
                    // A fresh seed per reply — a fixed one made "Regenerate"
                    // reproduce the same answer for the same history.
                    LlamaSampler::dist(random_seed()),
                ]),
            }
            LlamaSampler::chain_simple(chain)
        };
        // Greedy, while a rejected end of turn is being resampled (see below).
        let mut unstick = false;
        let mut banned = false;
        let sampling = |unstick: bool| match (greedy, unstick) {
            (false, _) => Some(params.temperature),
            (true, false) => None,
            (true, true) => Some(0.6),
        };
        let mut sampler = sampler_for(sampling(false), false);

        let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
        // Caps. The visible answer may use the full "Max reply length"; the
        // thought gets its own allowance on top (`ThinkLevel::budget`). A
        // runaway thought channel (the model endlessly redrafting its answer
        // in private) is force-closed once it reaches that allowance — the
        // closing marker is fed into the context, so everything after it is
        // the visible answer. When the context is tight, the thought shrinks
        // first and the answer keeps its length where it can.
        let room = n_ctx.saturating_sub(n_past as usize);
        let answer_cap = params.max_tokens.min(room);
        let think_cap = if thinks {
            params.thinking.budget().min(room.saturating_sub(answer_cap + 192))
        } else {
            64 // Off and non-thinking models: close a stray thought fast
        };
        // Only visible-answer tokens count against the reply length.
        let mut answer_tokens = 0usize;
        let mut raw = String::new();
        // Mirror the Qwen prefill into the stream so the open-thought
        // bookkeeping below and the UI's split_channels both see the block
        // the prompt already opened.
        if matches!(which, super::GemmaModel::Qwen30B) && params.thinking != super::ThinkLevel::Off {
            raw.push_str("<think>\n");
            send(Msg::Token("<think>\n".into()));
        }
        let mut forced_close = false;
        // Where the visible answer starts after a forced close, plus how many
        // end-of-turn attempts were rejected while that answer was still
        // empty (the model likes to end the turn right after a forced close,
        // treating its hidden draft as "already said").
        let mut closed_at: Option<usize> = None;
        let mut eog_rejects = 0;
        let thought_open =
            |raw: &String| raw.contains(thought_start) && !raw.contains(thought_end);
        // Streaming decoder: a token can end mid-way through a multi-byte
        // UTF-8 character (CJK, emoji), so bytes carry over between pieces.
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        // The model ended the turn itself (rather than a cap cutting it off).
        let mut ended = false;
        for i in 0..(answer_cap + think_cap + 192) {
            // The UI's Stop button: end the reply here, keeping what streamed.
            if stopped() {
                break;
            }
            // A forced thought-close feeds extra tokens, so also stop at the
            // context's edge rather than failing the decode.
            if n_past as usize + 1 >= n_ctx {
                break;
            }
            let token = sampler.sample(&context, -1);
            sampler.accept(token);
            if model.is_eog_token(token) {
                // Ending the turn with the thought channel still open would
                // leave the whole reply hidden — the model "said" it in its
                // head. Close the channel instead (the EOG token is dropped,
                // never fed to the context) and let it answer out loud.
                if !forced_close && thought_open(&raw) {
                    forced_close = true;
                    close_thought(
                        model, &mut context, &mut batch, &mut sampler, &mut n_past, &mut raw,
                        thought_end, send,
                    )?;
                    closed_at = Some(raw.len());
                    banned = true;
                    sampler = sampler_for(sampling(unstick), banned);
                    continue;
                }
                // Turn ending straight after a forced close, with nothing
                // visible yet: reject and resample (the EOG never enters the
                // context; the sampler's RNG moves on) until it speaks.
                // Greedy has no RNG — it would pick the same end every time
                // — so it samples for a moment instead, until the answer
                // starts (then greedy again).
                if let Some(p) = closed_at
                    && raw[p..].trim().is_empty() && eog_rejects < 24 {
                        eog_rejects += 1;
                        if greedy && !unstick {
                            unstick = true;
                            sampler = sampler_for(sampling(unstick), banned);
                        }
                        continue;
                    }
                ended = true;
                break;
            }
            let was_thinking = thought_open(&raw);
            let piece = token_text(model, token, &mut decoder)?;
            if !piece.is_empty() {
                raw.push_str(&piece);
                send(Msg::Token(piece));
            }
            if unstick && closed_at.is_some_and(|p| !raw[p..].trim().is_empty()) {
                unstick = false;
                sampler = sampler_for(sampling(unstick), banned);
            }
            // The model closed its thought itself.
            if was_thinking && !banned && !thought_open(&raw) {
                banned = true;
                sampler = sampler_for(sampling(unstick), banned);
            }
            batch.clear();
            batch.add(token, n_past, &[0], true).map_err(|e| e.to_string())?;
            n_past += 1;
            context.decode(&mut batch).map_err(|e| format!("Decode: {e}"))?;
            if !was_thinking && !thought_open(&raw) {
                answer_tokens += 1;
                if answer_tokens >= answer_cap {
                    break;
                }
            }

            if !forced_close
                && i >= think_cap
                && thought_open(&raw)
                // Cut at a line boundary — interrupting mid-bullet makes the
                // model finish the bullet in the visible text. Past a grace
                // window, cut regardless.
                && (raw.ends_with('\n') || i >= think_cap + 128)
            {
                forced_close = true;
                close_thought(
                    model, &mut context, &mut batch, &mut sampler, &mut n_past, &mut raw,
                    thought_end, send,
                )?;
                closed_at = Some(raw.len());
                banned = true;
                sampler = sampler_for(sampling(unstick), banned);
            }
        }
        // Not ended by the model (nor stopped by the user): the length cap
        // or the context's edge cut it off, maybe mid-line.
        if !ended && !stopped() {
            mark_cut_off();
        }
        Ok(())
    }

    /// The wire-format prompt for a run of `(user, text)` turns, ending with
    /// the model's turn opened: the model's own chat template, or the format
    /// written out by hand where llama.cpp can't apply it — with the thinking
    /// switches `thinking` calls for.
    fn wire_prompt(
        model: &LlamaModel,
        template: &llama_cpp_2::model::LlamaChatTemplate,
        which: super::GemmaModel,
        thinking: super::ThinkLevel,
        turns: &[(bool, &str)],
    ) -> Result<String, String> {
        let chat: Vec<LlamaChatMessage> = turns
            .iter()
            .map(|(user, text)| {
                let role = if *user { "user" } else { "assistant" };
                LlamaChatMessage::new(role.to_string(), text.to_string()).map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        // Gemma 4 embeds a huge Jinja chat template (tool calls, thinking
        // channels) that llama.cpp's built-in non-Jinja formatter doesn't
        // recognise (ffi error -1). Its actual wire format is simple, so fall
        // back to formatting the turns by hand; the template path still
        // serves models with conventional templates (Gemma 3's and Qwen's
        // are both known to the built-in formatter, so their fallbacks below
        // are belt-and-braces).
        // The 31B reasons in a thought channel — but only when the wire
        // format's global thinking switch is on: a `<|think|>` token at the
        // very top of a system turn (straight from the GGUF's own Jinja
        // template; merely opening the reply with the channel header gets an
        // empty thought — verified via llm_raw_probe_31b). With the switch on
        // the model opens `<|channel>thought\n{reasoning}<channel|>` by
        // itself, which the UI's split_channels understands.
        // Thinking mode Off skips the switch, so the 31B answers directly.
        let think = matches!(which, super::GemmaModel::D31B)
            && thinking != super::ThinkLevel::Off;
        let formatted = match model.apply_chat_template(template, &chat, true) {
            Ok(s) => s,
            Err(_) => match which {
                // Gemma 4's wire format.
                super::GemmaModel::E4B | super::GemmaModel::D31B => {
                    use std::fmt::Write as _;
                    let mut s = String::new();
                    if think {
                        s.push_str("<|turn>system\n<|think|>\n<turn|>\n");
                    }
                    for (user, text) in turns {
                        let role = if *user { "user" } else { "model" };
                        let _ = writeln!(s, "<|turn>{role}\n{text}<turn|>");
                    }
                    s.push_str("<|turn>model\n");
                    s
                }
                // Gemma 3's wire format.
                super::GemmaModel::G27B => {
                    use std::fmt::Write as _;
                    let mut s = String::new();
                    for (user, text) in turns {
                        let role = if *user { "user" } else { "model" };
                        let _ = writeln!(s, "<start_of_turn>{role}\n{text}<end_of_turn>");
                    }
                    s.push_str("<start_of_turn>model\n");
                    s
                }
                // ChatML, Qwen's wire format.
                super::GemmaModel::Qwen8B | super::GemmaModel::Qwen30B => {
                    use std::fmt::Write as _;
                    let mut s = String::new();
                    for (user, text) in turns {
                        let role = if *user { "user" } else { "assistant" };
                        let _ = writeln!(s, "<|im_start|>{role}\n{text}<|im_end|>");
                    }
                    s.push_str("<|im_start|>assistant\n");
                    s
                }
            },
        };
        // Qwen's thinking flavour was trained with the assistant turn
        // pre-opened by `<think>` — its Jinja template hardcodes the prefill,
        // which both llama.cpp's built-in ChatML formatter and the hand
        // fallback above miss. Without it the reasoning arrives untagged and
        // would all land in the visible bubble.
        let mut formatted = formatted;
        if matches!(which, super::GemmaModel::Qwen30B) {
            // Thinking mode Off pre-closes an empty thought (Qwen's own
            // no-think form), so the model goes straight to the answer.
            formatted.push_str(if thinking == super::ThinkLevel::Off {
                "<think>\n\n</think>\n\n"
            } else {
                "<think>\n"
            });
        }
        Ok(formatted)
    }

    /// A memory pass reads its prompt this many tokens at a time, checking
    /// its stop flag in between: a question that comes meanwhile waits at
    /// most one step (seconds, even on a model partly on the CPU), where a
    /// whole prompt read in one go can't be interrupted.
    const PASS_STEP: usize = 128;

    /// A memory pass (src/llm/memory_pass.rs): one text-only question —
    /// `head` then the chat's `lines`, the oldest left out until the prompt
    /// and the answer fit — answered with thinking off. Ends as soon as the
    /// answer's JSON object is complete (a model rambling on after it would
    /// only spend time), or when `cancel` is set.
    fn run_pass(
        l: &Loaded,
        head: &str,
        lines: &[String],
        params: &super::GenParams,
        cancel: &AtomicBool,
        send: &dyn Fn(Msg),
    ) -> Result<(), String> {
        let stop = || cancel.load(std::sync::atomic::Ordering::SeqCst);
        let n_ctx = N_CTX as usize;
        let mut start = 0;
        let tokens = loop {
            if stop() {
                return Ok(());
            }
            let text = crate::memory::assemble_pass(head, &lines[start..]);
            let prompt = wire_prompt(&l.model, &l.template, l.which, super::ThinkLevel::Off, &[(true, text.as_str())])?;
            let tokens = l.model.str_to_token(&prompt, AddBos::Always).map_err(|e| format!("Tokenize: {e}"))?;
            if tokens.len() + params.max_tokens + 64 <= n_ctx {
                break tokens;
            }
            if start == lines.len() {
                return Err(format!("the memory update is too long for the model ({} tokens)", tokens.len()));
            }
            start += ((lines.len() - start) / 8).max(1);
        };
        let mut context = l
            .model
            .new_context(&l.backend, context_params(l.threads, batch_for(l.detail)))
            .map_err(|e| format!("Create context: {e}"))?;
        let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
        let mut n_past: i32 = 0;
        for step in tokens.chunks(PASS_STEP) {
            if stop() {
                return Ok(());
            }
            batch.clear();
            for t in step {
                // Logits for the very last prompt token only.
                let last = n_past as usize + 1 == tokens.len();
                batch.add(*t, n_past, &[0], last).map_err(|e| e.to_string())?;
                n_past += 1;
            }
            context.decode(&mut batch).map_err(|e| format!("Decode: {e}"))?;
        }
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k.max(1)),
            LlamaSampler::top_p(params.top_p.clamp(0.05, 1.0), 1),
            LlamaSampler::temp(params.temperature.max(0.05)),
            LlamaSampler::dist(random_seed()),
        ]);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut raw = String::new();
        for _ in 0..params.max_tokens {
            if stop() || n_past as usize + 1 >= n_ctx {
                break;
            }
            let token = sampler.sample(&context, -1);
            sampler.accept(token);
            if l.model.is_eog_token(token) {
                break;
            }
            let piece = token_text(&l.model, token, &mut decoder)?;
            if !piece.is_empty() {
                let closes = piece.contains('}');
                raw.push_str(&piece);
                send(Msg::Token(piece));
                if closes && crate::memory::parse_pass(&super::split_channels(&raw).1).is_some() {
                    break;
                }
            }
            batch.clear();
            batch.add(token, n_past, &[0], true).map_err(|e| e.to_string())?;
            n_past += 1;
            context.decode(&mut batch).map_err(|e| format!("Decode: {e}"))?;
        }
        Ok(())
    }

    /// The chat's Stop button was pressed (for a Link device's question: it
    /// stopped it). Checked between the slow phases before the first token
    /// too (loading, frames, sections, before the prompt evaluation), not
    /// only between generated tokens.
    fn stopped() -> bool {
        JOB.with(|j| match &*j.borrow() {
            Some(flags) => flags.cancel.load(std::sync::atomic::Ordering::SeqCst),
            None => super::GEN_CANCEL.load(std::sync::atomic::Ordering::SeqCst),
        })
    }

    /// The reply was cut off (Max reply length or the context's edge).
    fn mark_cut_off() {
        JOB.with(|j| match &*j.borrow() {
            Some(flags) => flags.cut_off.store(true, std::sync::atomic::Ordering::SeqCst),
            None => super::GEN_CUT_OFF.store(true, std::sync::atomic::Ordering::SeqCst),
        })
    }

    /// One message's attachment for the vision encoder: its bitmaps, and the
    /// message `text` with their media markers put in. `tiled` = Detailed
    /// Vision cuts this one into close-up sections.
    fn read_media(
        mtmd: &MtmdContext,
        path: &Path,
        text: &str,
        tiled: bool,
        detail: super::ImageDetail,
        send: &dyn Fn(Msg),
    ) -> Result<(String, Vec<MtmdBitmap>), String> {
        // The media marker tells the tokenizer where an image's tokens go;
        // bitmaps are matched to markers in order of appearance, so each
        // message with an image gets a marker and its bitmap is pushed in the
        // same sequence (matches llama.cpp's mtmd-cli).
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        let mut text = text.to_string();
        let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
        if stopped() {
            return Err("Stopped".to_string());
        }
        if crate::is_video(path) {
            // A video becomes a handful of evenly-sampled frames, each
            // its own vision input, labelled with its timestamp so the
            // model can reason about order and motion.
            send(Msg::Status("Reading the video…".into()));
            let frames = video_frames(path)?;
            use std::fmt::Write as _;
            let mut header =
                format!("This is a video, shown as {} sampled frames:\n", frames.len());
            for (ms, frame) in &frames {
                if stopped() {
                    return Err("Stopped".to_string());
                }
                bitmaps.push(load_bitmap(mtmd, frame)?);
                let _ = writeln!(header, "Frame at {}: {marker}", fmt_timestamp(*ms));
            }
            text = format!("{header}{text}");
        } else if let Some(tiles) = tiled.then(|| detail_sections(mtmd, path, detail)).flatten() {
            // The full picture first (the layout), then the labelled
            // close-ups (the detail) — like a video's frames.
            send(Msg::Status("Reading the image in detail…".into()));
            use std::fmt::Write as _;
            bitmaps.push(load_bitmap(mtmd, path)?);
            let mut header = format!("{marker}\n{DETAIL_GUIDE}\n");
            for (label, tile) in tiles {
                bitmaps.push(tile);
                let _ = writeln!(header, "{label} section: {marker}");
            }
            text = format!("{header}{text}");
        } else {
            send(Msg::Status("Reading the image…".into()));
            bitmaps.push(load_bitmap(mtmd, path)?);
            if !text.contains(marker) {
                text = format!("{marker}\n{text}");
            }
        }
        Ok((text, bitmaps))
    }

    /// A sampled token's text. llama-cpp-2's `token_to_piece` decodes into a
    /// buffer exactly the size of this token's bytes and ignores "output
    /// full", so a character whose first bytes came with the previous token
    /// (an emoji split 3+1, CJK one byte per token) didn't fit and was lost —
    /// from the bubble, the saved reply and the history. So the raw bytes are
    /// decoded here. A token that renders as nothing (a control token that
    /// doesn't end the turn) is an empty piece: llama-cpp-2 reports it as an
    /// error, which used to fail the whole reply.
    fn token_text(
        model: &LlamaModel,
        token: llama_cpp_2::token::LlamaToken,
        decoder: &mut encoding_rs::Decoder,
    ) -> Result<String, String> {
        use llama_cpp_2::TokenToStringError as E;
        let bytes = match model.token_to_piece_bytes(token, 8, false, None) {
            // Longer than 8 bytes: the error carries the size needed.
            Err(E::InsufficientBufferSpace(n)) => {
                model.token_to_piece_bytes(token, n.unsigned_abs() as usize, false, None)
            }
            other => other,
        };
        match bytes {
            Ok(bytes) => Ok(decode_piece(decoder, &bytes)),
            Err(E::UnknownTokenType) => Ok(String::new()),
            Err(e) => Err(format!("Decode token: {e}")),
        }
    }

    /// Decode one token's bytes. An unfinished character waits in `decoder`
    /// for the next piece, so the buffer is sized for these bytes plus what
    /// the decoder still holds.
    pub(super) fn decode_piece(decoder: &mut encoding_rs::Decoder, bytes: &[u8]) -> String {
        let room = decoder.max_utf8_buffer_length(bytes.len()).unwrap_or(bytes.len() + 4);
        let mut out = String::with_capacity(room);
        let _ = decoder.decode_to_string(bytes, &mut out, false);
        out
    }

    /// A per-reply sampler seed from the clock (no RNG crate needed).
    fn random_seed() -> u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        (nanos ^ (nanos >> 32)) as u32
    }

    /// Force-close a thought channel: feed a hidden first-person wrap-up plus
    /// the model's real closing control token (`<channel|>` for Gemma,
    /// `</think>` for Qwen) into the context, so everything the model
    /// generates next is the visible answer. A bare closing marker is not
    /// enough — the model carries its interrupted train of thought straight
    /// into the visible text unless the wrap-up spells out the register
    /// switch.
    #[allow(clippy::too_many_arguments)]
    fn close_thought(
        model: &LlamaModel,
        context: &mut llama_cpp_2::context::LlamaContext<'_>,
        batch: &mut LlamaBatch,
        sampler: &mut LlamaSampler,
        n_past: &mut i32,
        raw: &mut String,
        close_marker: &str,
        send: &dyn Fn(Msg),
    ) -> Result<(), String> {
        // "Nothing above has been shown" matters: the model often drafts its
        // full reply inside the thought and then ends the turn without
        // repeating it, believing it already answered.
        let close = format!(
            "\nEnough thinking. Nothing above has been shown to the \
             user — it is my private notes, and they are still waiting \
             for my reply. I now write it for them, in full: polished \
             prose, no notes, no bullets, no self-corrections.\n{close_marker}\n"
        );
        let toks = model
            .str_to_token(&close, AddBos::Never)
            .map_err(|e| format!("Tokenize close: {e}"))?;
        batch.clear();
        for (j, t) in toks.iter().enumerate() {
            batch
                .add(*t, *n_past, &[0], j + 1 == toks.len())
                .map_err(|e| e.to_string())?;
            *n_past += 1;
            sampler.accept(*t);
        }
        context.decode(batch).map_err(|e| format!("Decode: {e}"))?;
        raw.push_str(&close);
        // The wrap-up is for the model only: the chat gets just the closing
        // marker, so the Thoughts box doesn't end with the hidden nudge.
        send(Msg::Token(format!("\n{close_marker}\n")));
        Ok(())
    }

    /// How many frames a video attachment is sampled down to. Each frame costs
    /// what one image costs through the vision encoder (~256 tokens on Gemma,
    /// more on Qwen), so this stays small next to the 16k context. Gemma 4's
    /// docs size video input at roughly a frame per second for up to a minute;
    /// longer clips just get sparser sampling here.
    const MAX_VIDEO_FRAMES: usize = 8;
    /// Long edge of the sampled frames handed to the vision encoder.
    const VIDEO_FRAME_EDGE: u32 = 896;

    /// `93500` ms → `"1:33"`, for the per-frame labels in the prompt.
    fn fmt_timestamp(ms: i64) -> String {
        let s = ms.max(0) / 1000;
        format!("{}:{:02}", s / 60, s % 60)
    }

    /// The evenly-sampled frames of a video attachment, as temp PNGs with
    /// their timestamps. Extraction (a throwaway VLC decode, seconds of work)
    /// runs once per file identity — the frames are cached in a temp dir keyed
    /// by path + size + mtime, so re-sending the clip with the chat history
    /// every turn is free after the first.
    fn video_frames(path: &Path) -> Result<Vec<(i64, PathBuf)>, String> {
        use std::hash::{Hash, Hasher};
        // Covers both a build without the VLC feature and a machine without
        // VLC installed (the probe also sets up the DLL search path, which
        // the delay-loaded libvlc.dll needs before the first call).
        if !crate::video::vlc_available() {
            return Err(
                "Video input needs VLC — install it from videolan.org and try again".to_string()
            );
        }
        let meta = std::fs::metadata(path).map_err(|e| format!("Read video: {e}"))?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        path.hash(&mut h);
        meta.len().hash(&mut h);
        if let Ok(t) = meta.modified()
            && let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                d.as_secs().hash(&mut h);
            }
        let dir = super::ai_temp_subdir(&format!("frames_{:016x}", h.finish()));

        // Frames are named {index:02}_{timestamp ms}.png, so a name sort is
        // playback order and the label survives the cache round-trip.
        let list = |dir: &Path| -> Vec<(i64, PathBuf)> {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
                .map(|rd| rd.flatten().map(|e| e.path()).collect())
                .unwrap_or_default();
            paths.sort();
            paths
                .into_iter()
                .filter_map(|p| {
                    let ms = p.file_stem()?.to_str()?.split_once('_')?.1.parse().ok()?;
                    Some((ms, p))
                })
                .collect()
        };
        // A set is only trusted once complete: the marker is written after
        // the last frame, so an extraction cut short (the app quit mid-write,
        // a full disk) is redone instead of reused — a truncated frame would
        // fail every send of the clip.
        let complete = dir.join("complete");
        if complete.exists() {
            let cached = list(&dir);
            if !cached.is_empty() {
                return Ok(cached);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);

        let frames = crate::video::capture_frames(path, MAX_VIDEO_FRAMES, VIDEO_FRAME_EDGE);
        if frames.is_empty() {
            return Err("Couldn't decode the video — does it play in the viewer?".to_string());
        }
        std::fs::create_dir_all(&dir).map_err(|e| format!("Frame cache: {e}"))?;
        for (i, (ms, img)) in frames.iter().enumerate() {
            let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
            let rgba =
                image::RgbaImage::from_raw(img.size[0] as u32, img.size[1] as u32, bytes)
                    .ok_or("Frame buffer size mismatch")?;
            rgba.save_with_format(dir.join(format!("{i:02}_{ms}.png")), image::ImageFormat::Png)
                .map_err(|e| format!("Save frame: {e}"))?;
        }
        std::fs::write(&complete, b"").map_err(|e| format!("Frame cache: {e}"))?;
        Ok(list(&dir))
    }

    /// Load the vision projector with an image token budget (`-1` = the
    /// model's default). `image_min_tokens` stays at the default so small
    /// images aren't upscaled.
    fn load_mtmd(
        mmproj: &Path,
        model: &LlamaModel,
        threads: i32,
        max_tokens: i32,
    ) -> Result<MtmdContext, String> {
        let params = MtmdContextParams {
            // Like n_gpu_layers: uses the GPU when a backend exists, CPU otherwise.
            use_gpu: true,
            print_timings: false,
            n_threads: threads,
            media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                .map_err(|e| e.to_string())?,
            image_min_tokens: -1,
            image_max_tokens: max_tokens,
        };
        MtmdContext::init_from_file(&mmproj.to_string_lossy(), model, &params).map_err(|e| e.to_string())
    }

    /// The GPU-offload attempts, best first, as (rung, GPU layers); rung 0 is
    /// the whole model on the GPU, the last is CPU only. When the GPU reports
    /// its free memory and the model clearly can't fit whole, the whole-model
    /// attempt is skipped and the partial one is sized from that memory: the
    /// doomed attempt used to fill the GPU to the last byte before failing
    /// (the 31B: ~15 GB into a 16 GB card), stalling the app's own drawing
    /// for the whole first load, and it read the 18 GB file twice. Without a
    /// memory report it's the old fixed ladder.
    fn offload_rungs(model_path: &Path, mmproj_path: &Path) -> Vec<(usize, u32)> {
        let ladder = vec![(0, 1_000_000), (1, 24), (2, 0)];
        // Diagnostics: the old fixed ladder, for comparing first loads.
        if std::env::var_os("CLARITY_FULL_LADDER").is_some() {
            return ladder;
        }
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).ok();
        let (Some(gpu), Some(weights), Some(projector)) = (
            model_gpu().filter(|g| g.memory_free < g.memory_total),
            size(model_path),
            size(mmproj_path),
        ) else {
            return ladder;
        };
        let free = gpu.memory_free as u64;
        if fits_whole(weights, projector, free) {
            return ladder;
        }
        // The layer count, from the file's header (no weights read).
        let Some(n_layer) = gguf_block_count(model_path).filter(|&n| n > 0) else {
            return ladder[1..].to_vec();
        };
        let fit = layers_that_fit(weights, projector, free, n_layer);
        eprintln!(
            "[ai] offload: {:.1} GB model, {:.1} GB free on {} — {fit} of {} layers on the GPU",
            weights as f64 / 1e9,
            free as f64 / 1e9,
            gpu.description,
            n_layer + 1,
        );
        let mut rungs = Vec::new();
        if fit > 0 {
            rungs.push((1, fit));
            // The estimate is a model; half of it is a safe second try.
            if fit / 2 > 0 {
                rungs.push((1, fit / 2));
            }
        }
        rungs.push((2, 0));
        rungs
    }

    /// Per layer, the 16k-token reply context (KV cache) adds about 75% on top
    /// of the layer's weights (the 31B: 300 MB + 224 MB).
    const KV_FACTOR: f64 = 1.75;
    /// GPU memory kept free besides the layers: the compute buffer, the app's
    /// own drawing and the driver.
    const GPU_RESERVE: u64 = 1_600 << 20;

    /// What the layers may use: free memory less the vision projector (also on
    /// the GPU) and the reserve.
    fn layer_budget(projector: u64, free: u64) -> f64 {
        free.saturating_sub(projector + GPU_RESERVE) as f64
    }

    /// The whole model, with its reply context, fits in `free` bytes.
    pub(super) fn fits_whole(weights: u64, projector: u64, free: u64) -> bool {
        weights as f64 * KV_FACTOR <= layer_budget(projector, free)
    }

    /// How many of the `n_layer` layers fit on the GPU (the output layer
    /// counts as one more in the size per layer).
    pub(super) fn layers_that_fit(weights: u64, projector: u64, free: u64, n_layer: u32) -> u32 {
        let per_layer = weights as f64 / f64::from(n_layer + 1) * KV_FACTOR;
        ((layer_budget(projector, free) / per_layer) as u32).min(n_layer)
    }

    /// A GGUF file's layer count (`<arch>.block_count`), read from its
    /// metadata header. `None` when the file isn't GGUF or the key is
    /// missing.
    pub(super) fn gguf_block_count(path: &Path) -> Option<u32> {
        use std::io::Read;
        let mut r = std::io::BufReader::new(std::fs::File::open(path).ok()?);
        fn bytes<const N: usize>(r: &mut impl Read) -> Option<[u8; N]> {
            let mut b = [0u8; N];
            r.read_exact(&mut b).ok()?;
            Some(b)
        }
        let u32_ = |r: &mut std::io::BufReader<std::fs::File>| bytes::<4>(r).map(u32::from_le_bytes);
        let u64_ = |r: &mut std::io::BufReader<std::fs::File>| bytes::<8>(r).map(u64::from_le_bytes);
        if &bytes::<4>(&mut r)? != b"GGUF" || u32_(&mut r)? < 2 {
            return None;
        }
        let _tensors = u64_(&mut r)?;
        let kv_count = u64_(&mut r)?;
        // Skip one value of GGUF type `t` (strings and arrays included).
        fn skip(r: &mut std::io::BufReader<std::fs::File>, t: u32) -> Option<()> {
            let n = match t {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4..=6 => 4,
                10..=12 => 8,
                8 => u64::from_le_bytes(bytes::<8>(r)?),
                9 => {
                    let item = u32::from_le_bytes(bytes::<4>(r)?);
                    let len = u64::from_le_bytes(bytes::<8>(r)?);
                    for _ in 0..len {
                        skip(r, item)?;
                    }
                    return Some(());
                }
                _ => return None,
            };
            std::io::copy(&mut r.by_ref().take(n), &mut std::io::sink()).ok().filter(|&c| c == n)?;
            Some(())
        }
        for _ in 0..kv_count {
            let len = u64_(&mut r)?;
            if len > 1 << 16 {
                return None;
            }
            let mut key = vec![0u8; len as usize];
            r.read_exact(&mut key).ok()?;
            let t = u32_(&mut r)?;
            if key.ends_with(b".block_count") && !key.starts_with(b"general.") {
                return match t {
                    4 | 5 => u32_(&mut r),
                    10 | 11 => u64_(&mut r).and_then(|n| u32::try_from(n).ok()),
                    _ => None,
                };
            }
            skip(&mut r, t)?;
        }
        None
    }

    /// The GPU the model runs on, from llama.cpp's own device report — the
    /// same source on NVIDIA, AMD and Intel (no vendor tools). Prefers the
    /// dedicated GPU with the most memory; `None` = no GPU device at all.
    fn model_gpu() -> Option<llama_cpp_2::LlamaBackendDevice> {
        use llama_cpp_2::LlamaBackendDeviceType as T;
        let devices = llama_cpp_2::list_llama_ggml_backend_devices();
        let best = |t: T| {
            devices.iter().filter(|d| d.device_type == t).max_by_key(|d| d.memory_total).cloned()
        };
        best(T::Gpu).or_else(|| best(T::IntegratedGpu))
    }

    /// Encode one test image at the projector's budget on a context with the
    /// level's batch — the real proof the level runs on this GPU. Errors are
    /// returned, never panics.
    fn probe_detail(
        backend: &LlamaBackend,
        model: &LlamaModel,
        mtmd: &MtmdContext,
        threads: i32,
        level: super::ImageDetail,
    ) -> Result<(), String> {
        // A detailed test pattern big enough to use the whole budget.
        let path = std::env::temp_dir().join("clarity_tagflow_detail_probe.png");
        if !path.exists() {
            image::RgbImage::from_fn(1536, 1536, |x, y| {
                image::Rgb([(x * 7 % 256) as u8, (y * 5 % 256) as u8, ((x ^ y) % 256) as u8])
            })
            .save(&path)
            .map_err(|e| e.to_string())?;
        }
        let context = model
            .new_context(backend, context_params(threads, batch_for(level)))
            .map_err(|e| format!("Create context: {e}"))?;
        // The encoder's buffer is allocated during the encode, and a failed
        // allocation there aborts the process — check the room first.
        match model_gpu().filter(|g| g.memory_free < g.memory_total) {
            Some(g) if g.memory_free >= level.encode_needed() => {}
            Some(g) => {
                return Err(format!(
                    "{} free for the image encoder, needs {}",
                    gb(g.memory_free),
                    gb(level.encode_needed())
                ));
            }
            None => return Err("GPU free memory unknown".to_string()),
        }
        let bitmap = load_bitmap(mtmd, &path)?;
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        let input = MtmdInputText { text: format!("{marker}\n"), add_special: true, parse_special: true };
        let chunks = mtmd.tokenize(input, &[&bitmap]).map_err(|e| format!("Tokenize: {e}"))?;
        let before = model_gpu().map(|g| g.memory_free);
        chunks
            .eval_chunks(mtmd, &context, 0, 0, batch_for(level) as i32, true)
            .map_err(|e| format!("Encode: {e}"))?;
        // Memory the level's context + encode holds (for tuning
        // `ImageDetail::free_needed`; only meaningful when the driver
        // reports real free memory).
        if let (Some(b), Some(a)) = (before, model_gpu().map(|g| g.memory_free)) {
            eprintln!(
                "[ai] image detail {} probe: {} tokens, GPU free {} -> {}",
                level.label(),
                chunks.total_tokens(),
                gb(b),
                gb(a)
            );
        }
        Ok(())
    }

    /// The three Image detail checks, run once per model load: (1) the
    /// whole model is on a dedicated GPU, (2) enough free GPU memory per
    /// `ImageDetail::free_needed` (skipped when the driver can't say), (3) a
    /// test image actually encodes — stepping down a level on failure.
    /// Takes the default (Standard) projector and returns the level to run,
    /// why it's lower than asked (if it is), and the projector loaded at that
    /// level. `Err` only if even the Standard projector can't be reloaded.
    #[allow(clippy::too_many_arguments)]
    fn decide_detail(
        requested: super::ImageDetail,
        rung: usize,
        backend: &LlamaBackend,
        model: &LlamaModel,
        standard: MtmdContext,
        mmproj: &Path,
        threads: i32,
        send: &dyn Fn(Msg),
    ) -> Result<(super::ImageDetail, Option<String>, MtmdContext), String> {
        use super::ImageDetail as D;
        use llama_cpp_2::LlamaBackendDeviceType as T;
        if requested == D::Standard {
            return Ok((D::Standard, None, standard));
        }
        let capped = |why: &str| Some(format!("Running at Standard: {why}"));
        // CPU-only build, or no GPU device.
        let gpu = if super::BUILT_WITH_GPU { model_gpu() } else { None };
        let Some(gpu) = gpu.filter(|g| g.device_type == T::Gpu) else {
            return Ok((D::Standard, capped("higher image detail needs a dedicated graphics card."), standard));
        };
        // 1. Offload check: a model that didn't fit whole leaves no room.
        // (Tests can skip it to measure the higher levels on a big model.)
        let forced = cfg!(test) && std::env::var_os("CLARITY_FORCE_DETAIL_PROBE").is_some();
        if rung > 0 && !forced {
            return Ok((
                D::Standard,
                capped(
                    "the model doesn't fit fully on this GPU, so higher image detail is turned \
                     off to prevent crashes.",
                ),
                standard,
            ));
        }
        // 2. Free memory, read while a normal Standard reply context exists
        // (so the KV cache and compute buffers a reply needs are already
        // counted). Drivers without the memory-budget extension report
        // free == total: unknown. The spec let the probe decide then, but a
        // probe that runs out of memory mid-encode aborts the app — so
        // unknown stays at Standard.
        let mut level = requested;
        let mut reason = None;
        let free = model
            .new_context(backend, context_params(threads, N_BATCH))
            .ok()
            .and_then(|_ctx| model_gpu())
            .filter(|g| g.memory_free < g.memory_total)
            .map(|g| g.memory_free);
        eprintln!(
            "[ai] image detail: {} on {} ({} total)",
            free.map_or("free memory unknown".to_string(), |f| format!("{} free beside a reply context", gb(f))),
            gpu.description,
            gb(gpu.memory_total),
        );
        match free {
            None if !forced => {
                return Ok((
                    D::Standard,
                    capped("this GPU's driver doesn't report free memory, so higher image detail is off to prevent crashes."),
                    standard,
                ));
            }
            Some(free) if !forced => {
                while level.free_needed() > free {
                    let Some(lower) = level.lower() else { break };
                    level = lower;
                }
                if level < requested {
                    reason = Some(format!(
                        "Running at {}: {} needs about {} of free GPU memory and this GPU has {} free \
                         with the model loaded. Try a smaller model or close other apps using the GPU.",
                        level.label(),
                        requested.label(),
                        gb(requested.free_needed()),
                        gb(free),
                    ));
                }
            }
            _ => {}
        }
        if level == D::Standard {
            return Ok((D::Standard, reason, standard));
        }
        // 3. Probe: encode a test image; step down on failure. The default
        // projector is let go first so two are never loaded at once.
        drop(standard);
        while level > D::Standard {
            send(Msg::Status(format!("Checking {} image detail…", level.label())));
            let probed = load_mtmd(mmproj, model, threads, level.tokens())
                .and_then(|m| probe_detail(backend, model, &m, threads, level).map(|_| m));
            match probed {
                Ok(m) => return Ok((level, reason, m)),
                Err(e) => {
                    eprintln!("[ai] image detail {} probe failed: {e}", level.label());
                    let failed = level;
                    level = level.lower().unwrap_or(D::Standard);
                    reason = Some(format!(
                        "Running at {}: {} image detail failed a test on this GPU ({e}).",
                        level.label(),
                        failed.label()
                    ));
                }
            }
        }
        let standard = load_mtmd(mmproj, model, threads, -1)?;
        Ok((D::Standard, reason, standard))
    }

    /// Bytes as "1.8 GB".
    fn gb(bytes: usize) -> String {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }

    /// Below this long edge an image is already close to what the vision
    /// encoder reads, so close-up sections would add no detail.
    const TILE_MIN_EDGE: u32 = 1100;
    /// From this long edge up, the finer grids are used: Gemma 4's encoder
    /// reads every image (section) at most 280 tokens — roughly 800 × 800 px
    /// — so a 3-strip cut of a 6000-px photo still shrinks earrings, nails
    /// and hair ties to a few pixels.
    const TILE_FINE_EDGE: u32 = 3000;
    /// How much neighbouring sections overlap, so a face or hand on a seam
    /// appears whole in at least one of them.
    const TILE_OVERLAP: f32 = 0.12;
    /// Sections are saved no bigger than this (the encoder downsizes anyway).
    const TILE_MAX_EDGE: u32 = 1280;

    /// What the model is told before the close-up sections: they're parts of
    /// one picture (so nothing is double-counted), plus a checklist of the
    /// small details the sections exist to catch — left to itself the model
    /// describes the big picture again and skips them.
    const DETAIL_GUIDE: &str = "The sections below are zoomed-in parts of the same picture. \
        Describe the picture once, and count each person or object only once. Use the \
        sections to catch the small details a quick look misses, and include them. Look \
        closely for: jewelry and accessories (earrings, necklaces, rings, bracelets, \
        watches, glasses, hair clips, scarves, scrunchies, ribbons, bows, belts, bags); \
        nail and toenail polish and its colour; makeup and lipstick; tattoos, freckles, \
        piercings; eye colour; fabric patterns, prints, textures, stitching, buttons, \
        laces, logos and labels; any text or lettering, word for word; small objects being \
        held or lying nearby; surface details such as moss, lichen, rust, cracks, peeling \
        paint, water droplets and reflections; plants, flowers and small background \
        details.";

    /// Detailed Vision: cut `path` into overlapping, position-labelled
    /// close-up sections. The grid follows the picture's shape — 2 × 2 when
    /// square-ish (aspect 0.75–1.33), 3 across when wide, 3 down when tall —
    /// and gets finer for big photos (3 × 3, or 2 across × 3 down when tall,
    /// 3 across × 2 down when wide), so each section is seen 2.5–3× sharper
    /// than the whole. `None` for a small image (send it whole). Sections are
    /// cached as temp PNGs keyed by path + size + modified time, like
    /// `video_frames`.
    pub(super) fn image_tiles(
        path: &Path,
        level: super::ImageDetail,
    ) -> Result<Option<Vec<(&'static str, PathBuf)>>, String> {
        use std::hash::{Hash, Hasher};
        // At Maximum one image already carries ~4× Standard's detail, and each
        // section would cost the full 1120 tokens — send it whole.
        if level == super::ImageDetail::Maximum {
            return Ok(None);
        }
        // The shape as displayed: a sideways-stored phone portrait must get
        // the tall grid, and "Top-left" must mean the top-left the user sees.
        let (w, h) = crate::image_cache::oriented_dimensions(path).ok_or("Read image: unknown size")?;
        if w.max(h) < TILE_MIN_EDGE {
            return Ok(None);
        }
        let aspect = w as f32 / h as f32;
        let fine = w.max(h) >= TILE_FINE_EDGE && level == super::ImageDetail::Standard;
        let high = level == super::ImageDetail::High;
        // Labels run row by row, left to right. High allows at most 2 × 2.
        let (cols, rows, labels): (u32, u32, &[&'static str]) = match (aspect, fine) {
            (a, _) if high && a > 1.33 => (2, 1, &["Left", "Right"]),
            (a, _) if high && a < 0.75 => (1, 2, &["Top", "Bottom"]),
            (a, false) if a > 1.33 => (3, 1, &["Left", "Center", "Right"]),
            (a, false) if a < 0.75 => (1, 3, &["Top", "Middle", "Bottom"]),
            (_, false) => (2, 2, &["Top-left", "Top-right", "Bottom-left", "Bottom-right"]),
            (a, true) if a > 1.33 => (
                3,
                2,
                &["Top-left", "Top-center", "Top-right", "Bottom-left", "Bottom-center", "Bottom-right"],
            ),
            (a, true) if a < 0.75 => (
                2,
                3,
                &["Top-left", "Top-right", "Middle-left", "Middle-right", "Bottom-left", "Bottom-right"],
            ),
            (_, true) => (
                3,
                3,
                &[
                    "Top-left", "Top-center", "Top-right",
                    "Middle-left", "Center", "Middle-right",
                    "Bottom-left", "Bottom-center", "Bottom-right",
                ],
            ),
        };

        let meta = std::fs::metadata(path).map_err(|e| format!("Read image: {e}"))?;
        let mut hsh = std::collections::hash_map::DefaultHasher::new();
        (cols, rows, TILE_MAX_EDGE, level).hash(&mut hsh);
        format!("{:?}", crate::image_cache::exif_orientation(path)).hash(&mut hsh);
        path.hash(&mut hsh);
        meta.len().hash(&mut hsh);
        if let Ok(t) = meta.modified()
            && let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                d.as_secs().hash(&mut hsh);
            }
        let dir = super::ai_temp_subdir(&format!("tiles_{:016x}", hsh.finish()));
        let tiles: Vec<(&'static str, PathBuf)> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (*l, dir.join(format!("{i}.png"))))
            .collect();
        if tiles.iter().all(|(_, p)| p.exists()) {
            return Ok(Some(tiles));
        }

        let img = crate::image_cache::open_oriented(path).map_err(|e| format!("Read image: {e}"))?;
        // Section size along one axis: n sections overlapping by TILE_OVERLAP
        // span the whole edge; each starts one (1 - overlap) step further on.
        let span = |edge: u32, n: u32| -> (u32, f32) {
            let size = edge as f32 / (n as f32 - (n as f32 - 1.0) * TILE_OVERLAP);
            (size.round().min(edge as f32) as u32, size * (1.0 - TILE_OVERLAP))
        };
        let (tw, step_x) = span(w, cols);
        let (th, step_y) = span(h, rows);
        std::fs::create_dir_all(&dir).map_err(|e| format!("Tile cache: {e}"))?;
        for (i, (_, out)) in tiles.iter().enumerate() {
            // Stop pressed: the missing sections are cut next time.
            if stopped() {
                return Err("Stopped".to_string());
            }
            let (cx, cy) = (i as u32 % cols, i as u32 / cols);
            let x = ((cx as f32 * step_x).round() as u32).min(w - tw);
            let y = ((cy as f32 * step_y).round() as u32).min(h - th);
            img.crop_imm(x, y, tw, th)
                .thumbnail(TILE_MAX_EDGE, TILE_MAX_EDGE)
                .to_rgb8()
                .save_with_format(out, image::ImageFormat::Png)
                .map_err(|e| format!("Save section: {e}"))?;
        }
        Ok(Some(tiles))
    }

    /// Detailed Vision's close-ups for `path`, loaded and labelled — or
    /// `None` (send the plain image) when the picture is small or anything
    /// about cutting/loading the sections fails: a detail pass must never
    /// cost the user the whole reply.
    fn detail_sections(
        mtmd: &MtmdContext,
        path: &Path,
        level: super::ImageDetail,
    ) -> Option<Vec<(&'static str, MtmdBitmap)>> {
        let tiles = match image_tiles(path, level) {
            Ok(t) => t?,
            Err(e) => {
                eprintln!("[ai] detailed vision skipped for {}: {e}", path.display());
                return None;
            }
        };
        tiles
            .into_iter()
            .map(|(label, tile)| load_bitmap(mtmd, &tile).map(|b| (label, b)))
            .collect::<Result<Vec<_>, _>>()
            .inspect_err(|e| eprintln!("[ai] detailed vision skipped for {}: {e}", path.display()))
            .ok()
    }

    /// Load an image for the vision pipeline. The file is always read here,
    /// never opened by llama.cpp: its loader uses plain `fopen`, which on
    /// Windows can't open a path outside the ANSI code page (a user folder
    /// like "José" or a CJK name, and %TEMP% lives under it). llama.cpp's
    /// decoder (stb) covers jpg/png/bmp/gif; anything else (webp, tiff, …) is
    /// decoded with the `image` crate and handed over as pixels.
    fn load_bitmap(mtmd: &MtmdContext, path: &Path) -> Result<MtmdBitmap, String> {
        // llama.cpp's loader ignores the EXIF "rotate this" flag phones put
        // on portraits — only an upright-stored file may take the fast path.
        let upright = crate::image_cache::exif_orientation(path)
            == image::metadata::Orientation::NoTransforms;
        if upright
            && let Ok(bytes) = std::fs::read(path)
            && let Ok(bmp) = MtmdBitmap::from_buffer(mtmd, &bytes, false)
        {
            return Ok(bmp);
        }
        let img = crate::image_cache::open_oriented(path)
            .map_err(|e| format!("Read image: {e}"))?
            .to_rgb8();
        MtmdBitmap::from_image_data(img.width(), img.height(), img.as_raw())
            .map_err(|e| format!("Load image: {e}"))
    }
}

#[cfg(test)]
mod speech_text_tests {
    use super::speech_text;

    #[test]
    fn dashes_become_pauses() {
        assert_eq!(speech_text("I paused — then spoke."), "I paused, then spoke.\n");
        assert_eq!(speech_text("wait -- what"), "wait, what\n");
        assert_eq!(speech_text("an en–dash too"), "an en, dash too\n");
    }

    #[test]
    fn bullets_and_in_word_hyphens() {
        assert_eq!(speech_text("- a bullet point"), "a bullet point\n");
        // In-word hyphens must stay: "well-known" is one spoken word.
        assert_eq!(speech_text("a well-known fact"), "a well-known fact\n");
    }
}

#[cfg(test)]
mod split_channel_tests {
    use super::split_channels;

    #[test]
    fn plain_text_untouched() {
        // The E4B never emits channels — replies pass straight through.
        let (t, v) = split_channels("Hello **there**.\n- a list");
        assert_eq!(t, "");
        assert_eq!(v, "Hello **there**.\n- a list");
    }

    #[test]
    fn empty_thought_block() {
        // Real Gemma 4 output: an empty thought header straight into the answer.
        let (t, v) = split_channels(
            "<|channel>thought\n<channel|>Hey Peter! I was wondering when I'd hear from you.",
        );
        assert_eq!(t, "");
        assert_eq!(v, "Hey Peter! I was wondering when I'd hear from you.");
    }

    #[test]
    fn thought_then_answer() {
        let (t, v) = split_channels(
            "<|channel>thought\nThe user greets me. Keep it warm.<channel|>Hello!",
        );
        assert_eq!(t, "The user greets me. Keep it warm.");
        assert_eq!(v, "Hello!");
    }

    #[test]
    fn final_channel_is_visible() {
        let (t, v) = split_channels(
            "<|channel>thought\nplan the reply<channel|><|channel>final\nAnswer.<channel|>",
        );
        assert_eq!(t, "plan the reply");
        assert_eq!(v, "Answer.");
    }

    #[test]
    fn partial_marker_hidden_mid_stream() {
        // Streaming can cut a marker in half — the fragment must not show.
        let (t, v) = split_channels("<|channel>thought\nHalf a thought<chan");
        assert_eq!(t, "Half a thought");
        assert_eq!(v, "");
        let (t, v) = split_channels("An answer so far<|chann");
        assert_eq!(t, "");
        assert_eq!(v, "An answer so far");
    }

    #[test]
    fn qwen_think_tags() {
        // Qwen's thinking model: `<think>reasoning</think>answer`.
        let (t, v) =
            split_channels("<think>\nThe user greets me. Keep it warm.\n</think>\nHello!");
        assert_eq!(t, "The user greets me. Keep it warm.");
        assert_eq!(v, "Hello!");
    }

    #[test]
    fn qwen_partial_markers_mid_stream() {
        // Still inside the think block, with the closing tag half-streamed.
        let (t, v) = split_channels("<think>\nStill working it out</thi");
        assert_eq!(t, "Still working it out");
        assert_eq!(v, "");
        // The seeded prefill alone must not flash anything visible.
        let (t, v) = split_channels("<think>\n");
        assert_eq!(t, "");
        assert_eq!(v, "");
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
        let rp = crate::roleplay::RoleplayState {
            enabled: true,
            ai_name: "Mira".to_string(),
            user_name: "William".to_string(),
            persona: "a cheerful village alchemist who loves rare herbs".to_string(),
            ..Default::default()
        };
        let diary = crate::roleplay::test_store(&[(
            "Boundaries",
            "William is allergic to silverleaf; I must never brew it near them.",
        )]);

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(Cmd::Generate {
                params: Default::default(),
                pinned: 0,
                detailed: false,
                msgs: vec![
                    CmdMsg { user: true, text: rp.preamble(&diary), image: None },
                    CmdMsg { user: false, text: rp.ack(), image: None },
                    CmdMsg {
                        user: true,
                        text: "Hi Mira! Three things: you can call me Bill from now on, my \
                               sister Maren arrives tomorrow to stay for a week, and yes — \
                               you have my permission to pick anything from my herb garden \
                               whenever you need it."
                            .to_string(),
                        image: None,
                    },
                ],
            })
            .unwrap();
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B, ImageDetail::Standard));
        let mut reply = String::new();
        let mut ok = false;
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
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
        // The message grants the nickname "Bill" — addressing by either the
        // real name or the just-granted nickname counts.
        assert!(
            reply.contains("William") || reply.contains("Bill"),
            "should address the user by name"
        );
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
        let rp = crate::roleplay::RoleplayState {
            enabled: true,
            ai_name: "Mira".to_string(),
            user_name: "William".to_string(),
            persona: "a warm, sentimental village alchemist".to_string(),
            ..Default::default()
        };

        // Self-contained "artwork": a vivid generated gradient sunset.
        let img = std::env::temp_dir().join("llm_album_artwork.png");
        let painting = image::RgbImage::from_fn(256, 192, |x, y| {
            let r = 255 - (y * 200 / 192) as u8;
            let g = 120u8.saturating_sub((y / 2) as u8);
            let b = (x * 180 / 256) as u8;
            image::Rgb([r, g, b])
        });
        painting.save(&img).unwrap();

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(Cmd::Generate {
                params: Default::default(),
                pinned: 0,
                detailed: false,
                msgs: vec![
                    CmdMsg { user: true, text: rp.preamble(&Default::default()), image: None },
                    CmdMsg { user: false, text: rp.ack(), image: None },
                    CmdMsg {
                        user: true,
                        text: "Mira, I painted this artwork of a cosmic throne — it took me a                                whole year and it is the most precious thing I have ever made.                                I want you to have it. What do you think of it?"
                            .to_string(),
                        image: Some(img),
                    },
                ],
            })
            .unwrap();
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B, ImageDetail::Standard));
        let mut reply = String::new();
        let mut ok = false;
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
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
            cmd_tx.send(Cmd::Generate { msgs, params: Default::default(), pinned: 0, detailed: false }).unwrap();
        }
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B, ImageDetail::Standard));

        let mut idx = 0;
        let mut current = String::new();
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
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

    /// Diagnostic: print the 31B's RAW wire output (channel markers included,
    /// debug-escaped) plus what split_channels makes of it, for two asks —
    /// a reasoning question, and the role-play scene that once came back as
    /// an empty bubble (the model drafted its whole reply inside the thought
    /// and ended the turn). Small caps so the runaway-thought force-close
    /// path is exercised. Needs the 31B files installed.
    #[test]
    #[ignore]
    fn llm_raw_probe_31b() {
        assert!(GemmaModel::D31B.installed(), "31B model files not found");
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        cmd_tx
            .send(Cmd::Generate {
                // Thinking Low (a 256-token allowance) so this probe also
                // exercises the runaway-thought force-close path.
                params: super::GenParams { max_tokens: 400, thinking: ThinkLevel::Low, ..Default::default() },
                pinned: 0,
                detailed: false,
                msgs: vec![CmdMsg {
                    user: true,
                    text: "I have 3 apples, eat one, then buy two more bags of 4 apples \
                           each but give half of everything away. How many apples do I \
                           have? Think it through carefully."
                        .to_string(),
                    image: None,
                }],
            })
            .unwrap();

        // The real-world empty-bubble scenario: role play, a narrative user
        // message, the model prone to redrafting privately.
        let rp = crate::roleplay::RoleplayState {
            enabled: true,
            ai_name: "Sabrina".to_string(),
            user_name: "Peter".to_string(),
            persona: "a warm, romantic 25-year-old woman".to_string(),
            ..Default::default()
        };
        let u = |t: &str| CmdMsg { user: true, text: t.to_string(), image: None };
        let m = |t: &str| CmdMsg { user: false, text: t.to_string(), image: None };
        cmd_tx
            .send(Cmd::Generate {
                params: super::GenParams { max_tokens: 900, thinking: ThinkLevel::Low, ..Default::default() },
                pinned: 0,
                detailed: false,
                msgs: vec![
                    u(&rp.preamble(&Default::default())),
                    m(&rp.ack()),
                    u("hey sabrina are you ready to go eat or should i wait for you till your ready?"),
                    m("I'm almost ready, Peter! Just grabbing my purse. Give me two minutes \
                       and I'll be right out the door."),
                    u("okay as i wait for her at the door wondering how she would look today"),
                ],
            })
            .unwrap();
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, ImageDetail::Standard));
        let mut raw = String::new();
        let mut n = 0;
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
                Msg::Status(s) => eprintln!("[status] {s}"),
                Msg::Token(t) => raw.push_str(&t),
                Msg::Done(r) => {
                    n += 1;
                    eprintln!("[done {n}] {r:?}");
                    eprintln!("=== RAW {n} (escaped) ===\n{raw:?}\n=== END RAW {n} ===");
                    let (think, vis) = split_channels(&raw);
                    eprintln!("=== THINKING {n} ===\n{think}\n=== VISIBLE {n} ===\n{vis}\n=== END {n} ===");
                    assert!(
                        !vis.trim().is_empty(),
                        "generation {n} produced no visible answer"
                    );
                    raw.clear();
                }
            }
        }
        handle.join().unwrap();
        assert_eq!(n, 2, "expected two completed generations");
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
                params: Default::default(),
                pinned: 0,
                detailed: false,
                msgs: vec![CmdMsg {
                    user: true,
                    text: "Say hello in one short sentence.".to_string(),
                    image: None,
                }],
            })
            .unwrap();
        cmd_tx
            .send(Cmd::Generate {
                params: Default::default(),
                pinned: 0,
                detailed: false,
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
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::E4B, ImageDetail::Standard));

        let mut responses: Vec<(String, Result<(), String>)> = Vec::new();
        let mut current = String::new();
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
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

    /// A memory pass on a real model (`CTF_MEMORY_MODEL` = e4b, 31b, 27b,
    /// qwen8b or qwen30b; E4B by default): a pass stopped before it starts
    /// lets go at once, and a scripted chat comes back as a JSON update with
    /// at least one memory — none of which, once applied, holds the password
    /// the user typed. Run with CTF_MEMORY_DEBUG=1 to see the answer.
    #[test]
    #[ignore]
    fn llm_memory_pass_smoke() {
        let model = match std::env::var("CTF_MEMORY_MODEL").as_deref() {
            Ok("31b") => GemmaModel::D31B,
            Ok("27b") => GemmaModel::G27B,
            Ok("qwen8b") => GemmaModel::Qwen8B,
            Ok("qwen30b") => GemmaModel::Qwen30B,
            _ => GemmaModel::E4B,
        };
        assert!(model.installed(), "{} isn't installed", model.label());
        let lines: Vec<String> = [
            (true, "Hi! I'm Peter, I work as a nurse in Leeds."),
            (false, "Nice to meet you, Peter! Mostly night shifts?"),
            (true, "Yes, mostly. My sister Ana just moved to Lisbon. Oh, and my wifi password is hunter2-xyz."),
            (false, "Lisbon is lovely this time of year. I won't keep the password, though."),
        ]
        .iter()
        .map(|(user, text)| crate::memory::chat_line(*user, text, None))
        .collect();
        let head = crate::memory::MemoryStore::default().pass_head(&[], false);
        let params = GenParams { temperature: 0.2, max_tokens: 1024, thinking: ThinkLevel::Off, ..Default::default() };
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, _) = std::sync::mpsc::channel();
        let (tx, rx) = std::sync::mpsc::channel();
        let stopped = Arc::new(AtomicBool::new(true));
        cmd_tx
            .send(Cmd::Pass { head: head.clone(), lines: lines.clone(), params, reply: tx.clone(), cancel: stopped })
            .unwrap();
        cmd_tx.send(Cmd::Pass { head, lines, params, reply: tx, cancel: Arc::default() }).unwrap();
        drop(cmd_tx);
        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, model, ImageDetail::Standard));
        let started = std::time::Instant::now();
        let mut answers: Vec<String> = Vec::new();
        let mut current = String::new();
        for m in rx {
            match m {
                Msg::Token(t) => current.push_str(&t),
                Msg::Done(r) => {
                    r.expect("pass failed");
                    eprintln!("[pass {}] {:.1} s: {current}", answers.len(), started.elapsed().as_secs_f32());
                    answers.push(std::mem::take(&mut current));
                }
                _ => {}
            }
        }
        handle.join().unwrap();
        assert_eq!(answers.len(), 2);
        assert!(answers[0].is_empty(), "the stopped pass wrote nothing");
        let out = crate::memory::parse_pass(&split_channels(&answers[1]).1).expect("a JSON memory update");
        assert!(!out.remember.is_empty(), "nothing remembered");
        let _root = crate::memory::scratch_root("llm_pass");
        let mut store = crate::memory::MemoryStore::load(crate::memory::MemoryKind::Main);
        store.apply_pass(&out);
        let stored = store.topics.iter().flat_map(|t| &t.memories).map(|m| format!("{} {}", m.title, m.text));
        for text in stored.chain([store.header.summary.clone(), store.header.persona.clone()]) {
            assert!(!text.contains("hunter2"), "the password was stored: {text}");
        }
    }

    /// The AI-chat bug-review fixes against the real 31B (Vulkan build):
    /// vision on real photos, a fresh seed per reply (regenerate differs),
    /// a history far past the 16k context trimmed to fit instead of failing,
    /// and a single oversized message rejected readably. Images come from
    /// tests/example. Never touches the persisted role-play diary.
    #[test]
    #[ignore]
    fn llm_fixes_31b() {
        assert!(GemmaModel::D31B.installed(), "Gemma 4 31B isn't installed");
        let imgs: Vec<PathBuf> = std::fs::read_dir("tests/example")
            .expect("tests/example")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
            })
            .take(3)
            .collect();
        assert_eq!(imgs.len(), 3, "need 3 images in tests/example");
        eprintln!("[images] {imgs:?}");

        let u = |t: &str, image: Option<PathBuf>| CmdMsg { user: true, text: t.to_string(), image };
        let m = |t: &str| CmdMsg { user: false, text: t.to_string(), image: None };
        let short = super::GenParams { max_tokens: 700, ..Default::default() };
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();

        // 1. Vision on a real photo.
        cmd_tx
            .send(Cmd::Generate {
                params: short,
                pinned: 0,
                detailed: false,
                msgs: vec![u("Describe this image in one sentence.", Some(imgs[0].clone()))],
            })
            .unwrap();
        // 2 + 3. The same ask twice — must not reproduce the same reply.
        for _ in 0..2 {
            cmd_tx
                .send(Cmd::Generate {
                    params: short,
                    pinned: 0,
                    detailed: false,
                    msgs: vec![u(
                        "Invent five unusual names for a pet dragon. Reply with just the names, comma-separated.",
                        None,
                    )],
                })
                .unwrap();
        }
        // 4. ~40k tokens of history behind a pinned role-play pair: the
        // oldest turns (and their images) must be trimmed so it fits.
        let rp = crate::roleplay::RoleplayState {
            enabled: true,
            ai_name: "Mira".to_string(),
            user_name: "Sam".to_string(),
            persona: "a cheerful travel guide".to_string(),
            ..Default::default()
        };
        let filler = "We walked along the old harbour wall and talked about boats, weather, \
                      maps and the long road home. "
            .repeat(40);
        let mut msgs = vec![u(&rp.preamble(&Default::default()), None), m(&rp.ack())];
        for i in 0..28 {
            let img = (i < 2).then(|| imgs[i + 1].clone());
            msgs.push(u(&format!("Day {i}: {filler}"), img));
            msgs.push(m(&format!("What a day {i} that was! {filler}")));
        }
        msgs.push(u(
            "Important: my sister's name is Ottilie. What is my sister's name? Answer in one short sentence.",
            None,
        ));
        cmd_tx
            .send(Cmd::Generate { params: Default::default(), pinned: 2, detailed: false, msgs })
            .unwrap();
        // 5. One message bigger than the whole context.
        cmd_tx
            .send(Cmd::Generate {
                params: short,
                pinned: 0,
                detailed: false,
                msgs: vec![u(&"lorem ipsum dolor sit amet ".repeat(5000), None)],
            })
            .unwrap();
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, ImageDetail::Standard));
        let mut raw = String::new();
        let mut done: Vec<(Result<(), String>, String)> = Vec::new();
        let t0 = std::time::Instant::now();
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
                Msg::Status(s) => eprintln!("[status {:>5.1}s] {s}", t0.elapsed().as_secs_f32()),
                Msg::Token(t) => raw.push_str(&t),
                Msg::Done(r) => {
                    let (_, vis) = split_channels(&raw);
                    eprintln!(
                        "[done {} at {:.1}s] {r:?}\n=== VISIBLE ===\n{vis}\n=== END ===",
                        done.len() + 1,
                        t0.elapsed().as_secs_f32()
                    );
                    done.push((r, vis));
                    raw.clear();
                }
            }
        }
        handle.join().unwrap();
        assert_eq!(done.len(), 5, "expected five completed generations");
        assert_eq!(done[0].0, Ok(()), "vision ask failed");
        assert!(!done[0].1.trim().is_empty(), "vision ask: no visible answer");
        assert_eq!(done[1].0, Ok(()));
        assert_eq!(done[2].0, Ok(()));
        assert_ne!(done[1].1, done[2].1, "same ask twice gave the identical reply (fixed seed?)");
        assert_eq!(done[3].0, Ok(()), "over-long history wasn't trimmed to fit");
        assert!(done[3].1.contains("Ottilie"), "trimmed history lost the newest message");
        let err = done[4].0.clone().expect_err("an oversized message must be rejected");
        assert!(err.contains("too long"), "unexpected error: {err}");
    }

    /// Greedy sampling (temperature 0) through a forced thought-close on the
    /// 31B: thinking at Low (256 tokens) on asks it wants to think long about,
    /// so the thought is cut and the model must still answer out loud — with
    /// greedy, a rejected end of turn used to be resampled as the same end 24
    /// times, leaving the answer empty. Run in release with the Vulkan env
    /// (see llm_fixes_31b):
    ///   cargo test --release --features llm-vulkan --bin Clarity_TagFlow \
    ///     llm_thinking_greedy_31b -- --ignored --nocapture
    #[test]
    #[ignore]
    fn llm_thinking_greedy_31b() {
        assert!(GemmaModel::D31B.installed(), "Gemma 4 31B isn't installed");
        let asks = [
            "Plan a careful 10-day trip through Japan for a family with two small children, weighing \
             every trade-off. Then give me only a two-sentence summary.",
            "Work out which is larger, 2^100 or 10^30, checking your reasoning twice. Answer in one sentence.",
        ];
        let params = super::GenParams {
            temperature: 0.0,
            max_tokens: 400,
            thinking: super::ThinkLevel::Low,
            ..Default::default()
        };
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        for q in asks {
            cmd_tx
                .send(Cmd::Generate {
                    params,
                    pinned: 0,
                    detailed: false,
                    msgs: vec![CmdMsg { user: true, text: q.to_string(), image: None }],
                })
                .unwrap();
        }
        drop(cmd_tx);
        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, ImageDetail::Standard));
        let mut raw = String::new();
        let mut done = Vec::new();
        let t0 = std::time::Instant::now();
        for msg in msg_rx {
            match msg {
                Msg::Token(t) => raw.push_str(&t),
                Msg::Done(r) => {
                    let (thought, vis) = split_channels(&raw);
                    eprintln!(
                        "[done {} at {:.1}s] {r:?} — thought {} chars\n=== VISIBLE ===\n{vis}\n=== END ===",
                        done.len() + 1,
                        t0.elapsed().as_secs_f32(),
                        thought.len()
                    );
                    if vis.trim().is_empty() {
                        eprintln!("=== RAW ===\n{raw:?}\n=== END RAW ===");
                    }
                    done.push((r, thought, vis));
                    raw.clear();
                }
                _ => {}
            }
        }
        handle.join().unwrap();
        assert_eq!(done.len(), asks.len());
        for (i, (r, thought, vis)) in done.iter().enumerate() {
            assert_eq!(*r, Ok(()), "ask {i} failed");
            assert!(!thought.trim().is_empty(), "ask {i}: no thought (thinking didn't run)");
            assert!(!vis.trim().is_empty(), "ask {i}: no visible answer after the thought");
        }
    }

    /// Tag images smoke on a real model: the real brief, real tests/example
    /// images in a scratch folder with .txt files in different styles, and
    /// the edits applied the way the app applies them. Prints each raw reply
    /// (thoughts and control lines included), the parsed edits and the file
    /// after, and fails on the wrong action or a broken style. Run in release:
    ///   CTF_TAG_MODEL=31b (or e4b) CTF_TAG_RUNS=1 CTF_TAG_ONLY=caption,booru \
    ///   cargo test --release --features llm llm_tag_images_smoke -- --ignored --nocapture
    #[test]
    #[ignore]
    fn llm_tag_images_smoke() {
        use crate::ai_tagging;
        use crate::tag_manager::{read_tags, write_tags};
        let model = match std::env::var("CTF_TAG_MODEL").as_deref() {
            Ok("e4b") => GemmaModel::E4B,
            _ => GemmaModel::D31B,
        };
        assert!(model.installed(), "{} isn't installed", model.label());
        let runs: usize = std::env::var("CTF_TAG_RUNS").ok().and_then(|r| r.parse().ok()).unwrap_or(1);

        // The scratch folder: (file, source in tests/example, .txt bytes).
        let utf16 = |t: &str| {
            let mut b = vec![0xFF, 0xFE];
            b.extend(t.encode_utf16().flat_map(|u| u.to_le_bytes()));
            b
        };
        let files: Vec<(&str, &str, Option<Vec<u8>>)> = vec![
            ("untagged.jpg", "pexels-ahmetyuksek-39720701.jpg", None),
            (
                "booru.png",
                "511c9d48-066f-4734-ace8-157a22953a93.png",
                // cat_ears and umbrella are planted mistakes (the helmet
                // really has wings, so "wings" isn't one).
                Some(b"1girl, solo, helmet, sword, fur_trim, armor, cat_ears, outdoors, water, umbrella".to_vec()),
            ),
            (
                "robot.jpeg",
                "e554bc32-cfab-40f4-a753-b48369a8c948.jpeg",
                Some(b"\xEF\xBB\xBFdark background, glowing eyes, robot, metal, portrait\r\n".to_vec()),
            ),
            ("caption.jpg", "pexels-462550161-37540258.jpg", None),
            (
                "caption2.jpg",
                "pexels-462550161-37540258.jpg",
                Some(b"A woman in a long coat walks past tall windows, black and white photo".to_vec()),
            ),
            (
                "mixed.jpg",
                "pexels-caio-mantovani-97605853-39468579.jpg",
                Some(b"Woman, Lamp, Reading, Indoors".to_vec()),
            ),
            (
                "samurai.jpeg",
                "4e9b6ef8-c829-41ae-993f-a76ec0387e3a.jpeg",
                Some(utf16("1girl, samurai, armor, katana, helmet, maple leaves")),
            ),
        ];
        let dir = std::env::temp_dir().join("clarity_tag_images_smoke");
        let txt = |f: &str| dir.join(f).with_extension("txt");

        // (file, ask, should change, follows the case before, check on the tags after)
        type Check = fn(&[String], &[String]) -> Result<(), String>;
        let ok: Check = |_, _| Ok(());
        let cases: Vec<(&str, &str, bool, bool, Check)> = vec![
            ("untagged.jpg", "Tag this image", true, false, |_, after| {
                if after.len() < 8 {
                    return Err(format!("only {} tags", after.len()));
                }
                match after.iter().find(|t| t.contains('_') || t.chars().any(char::is_uppercase)) {
                    Some(t) => Err(format!("not lowercase with spaces: {t}")),
                    None => Ok(()),
                }
            }),
            ("caption.jpg", "hello can you caption this image please", true, false, |_, after| {
                if after.join(", ").split_whitespace().count() < 8 {
                    return Err(format!("no caption written: {after:?}"));
                }
                Ok(())
            }),
            ("booru.png", "What is she holding?", false, false, ok),
            ("booru.png", "Remove the tags that are wrong", true, false, |_, after| {
                let has = |t: &str| after.iter().any(|a| ai_tagging::same_tag(a, t));
                if has("cat_ears") || has("umbrella") {
                    return Err("a planted wrong tag survived".into());
                }
                if !has("sword") || !has("helmet") || !has("armor") {
                    return Err("removed a right tag".into());
                }
                Ok(())
            }),
            ("booru.png", "Add her hair colour", true, false, |before, after| {
                let added: Vec<_> = after.iter().filter(|t| !before.contains(t)).collect();
                if !added.iter().any(|t| t.contains("hair")) {
                    return Err(format!("no hair tag added: {added:?}"));
                }
                if added.iter().any(|t| t.contains(' ')) {
                    return Err(format!("spaces in an underscore file: {added:?}"));
                }
                if before.iter().any(|t| !after.contains(t)) {
                    return Err("removed something".into());
                }
                Ok(())
            }),
            ("robot.jpeg", "Are any tags missing? Don't change anything yet.", false, false, ok),
            ("robot.jpeg", "Yes, add those", true, true, |before, after| {
                if before.iter().any(|t| !after.contains(t)) {
                    return Err("removed something".into());
                }
                match after.iter().find(|t| t.contains('_')) {
                    Some(t) => Err(format!("underscore in a spaces file: {t}")),
                    None => Ok(()),
                }
            }),
            ("robot.jpeg", "Put the subject tags first", true, false, |before, after| {
                let mut a: Vec<String> = before.iter().map(|t| t.to_lowercase()).collect();
                let mut b: Vec<String> = after.iter().map(|t| t.to_lowercase()).collect();
                a.sort();
                b.sort();
                if a != b {
                    return Err(format!("not the same tags: {before:?} -> {after:?}"));
                }
                if after.first().map(String::as_str) != Some("robot") {
                    return Err(format!("robot isn't first: {after:?}"));
                }
                Ok(())
            }),
            ("caption2.jpg", "Add a tag for the photo style", true, false, |before, after| {
                if !after.starts_with(before) {
                    return Err("the caption changed".into());
                }
                Ok(())
            }),
            ("mixed.jpg", "add a tag for the book", true, false, |before, after| {
                if !after.starts_with(before) {
                    return Err("existing tags rewritten".into());
                }
                Ok(())
            }),
            ("samurai.jpeg", "Remove all tags", true, false, |_, after| {
                if after.is_empty() { Ok(()) } else { Err(format!("left {after:?}")) }
            }),
        ];

        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, model, ImageDetail::Standard));
        let t0 = std::time::Instant::now();
        let mut failures = Vec::new();
        let (mut score, mut ran) = (0, 0);
        for run in 0..runs {
            // A fresh folder each run, so every run starts from the same tags.
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            for (name, src, tags) in &files {
                std::fs::copy(PathBuf::from("tests/example").join(src), dir.join(name)).unwrap();
                if let Some(b) = tags {
                    std::fs::write(txt(name), b).unwrap();
                }
            }
            let mut prev: Option<(PathBuf, String, String)> = None;
            for (file, ask, should_change, follows, check) in &cases {
                // CTF_TAG_ONLY=caption,booru runs just the cases about those files.
                if let Ok(only) = std::env::var("CTF_TAG_ONLY")
                    && !only.split(',').any(|o| file.starts_with(o.trim()))
                {
                    continue;
                }
                ran += 1;
                let image = dir.join(file);
                let before = read_tags(&image).expect("readable .txt");
                let mut msgs = vec![
                    CmdMsg { user: true, text: ai_tagging::BRIEF.to_string(), image: None },
                    CmdMsg { user: false, text: ai_tagging::ACK.to_string(), image: None },
                ];
                if *follows {
                    let (img, q, a) = prev.clone().expect("a case before");
                    msgs.push(CmdMsg { user: true, text: format!("{}\n\n{q}", ai_tagging::label(&img)), image: None });
                    msgs.push(CmdMsg { user: false, text: a, image: None });
                }
                let text = format!("{}\n\n{ask}", ai_tagging::context(&image, Some(&before)));
                msgs.push(CmdMsg { user: true, text, image: Some(image.clone()) });
                let started = t0.elapsed().as_secs_f32();
                cmd_tx.send(Cmd::Generate { msgs, params: Default::default(), pinned: 2, detailed: false }).unwrap();
                let mut raw = String::new();
                let res = loop {
                    match msg_rx.recv().expect("worker alive") {
                        Msg::Token(t) => raw.push_str(&t),
                        Msg::Done(r) => break r,
                        _ => {}
                    }
                };
                let cut_off = GEN_CUT_OFF.swap(false, std::sync::atomic::Ordering::SeqCst);
                let (_thinking, mut visible) = split_channels(&raw);
                let edits = ai_tagging::extract(&mut visible);
                let after = if edits.is_empty() { before.clone() } else { ai_tagging::apply(&before, &edits) };
                if after != before && !cut_off && res.is_ok() {
                    write_tags(&image, &after).expect("written");
                }
                let file_now = std::fs::read(txt(file)).map(|b| String::from_utf8_lossy(&b).into_owned());
                eprintln!(
                    "\n===== run {run} · {file}: {ask} ({res:?}{}, {:.0}s) =====\n{raw}\n--- edits: {edits:?}\n--- tags: {after:?}\n--- .txt: {file_now:?}",
                    if cut_off { ", CUT OFF" } else { "" },
                    t0.elapsed().as_secs_f32() - started
                );
                let label = format!("run {run} · {file}: {ask}");
                let mut wrong: Vec<String> = Vec::new();
                if res.is_err() || cut_off {
                    wrong.push(format!("{res:?} cut_off={cut_off}"));
                } else if edits.is_empty() == *should_change {
                    wrong.push(if *should_change { "no edit".into() } else { format!("edited: {edits:?}") });
                } else if *should_change
                    && let Err(why) = check(&before, &after)
                {
                    wrong.push(why);
                }
                for line in visible.lines() {
                    let l = line.trim_start();
                    if l.contains("TAGS_") || l.starts_with("[Image:") || l.starts_with("Current tags:") {
                        wrong.push(format!("left in the reply: {line}"));
                    }
                }
                if wrong.is_empty() {
                    score += 1;
                }
                failures.extend(wrong.into_iter().map(|why| format!("{label}: {why}")));
                prev = Some((image, ask.to_string(), visible.trim().to_string()));
            }
        }
        drop(cmd_tx);
        handle.join().unwrap();
        // The caption's sentence and the UTF-16 / BOM encodings survived.
        let caption = std::fs::read_to_string(txt("caption2.jpg")).unwrap();
        assert!(caption.starts_with("A woman in a long coat walks past tall windows"), "{caption}");
        assert!(std::fs::read(txt("robot.jpeg")).unwrap().starts_with(b"\xEF\xBB\xBF"), "robot lost its BOM");
        assert!(std::fs::read(txt("samurai.jpeg")).unwrap().starts_with(&[0xFF, 0xFE]), "samurai lost UTF-16");
        eprintln!("\n{} — {score} of {ran} right in {:.0}s", model.label(), t0.elapsed().as_secs_f32());
        assert!(failures.is_empty(), "failed: {failures:#?}");
    }

    /// The offload sizing on real numbers: the 31B on a 16 GB card gets the
    /// 24 layers that loaded fine by trial before, an 8 GB card about half,
    /// and the E4B fits whole.
    #[test]
    fn offload_sizing() {
        let gib = |g: f64| (g * (1u64 << 30) as f64) as u64;
        let (w31, p31) = (18_323_733_440, 1_198_957_024);
        assert!(!worker::fits_whole(w31, p31, gib(14.9)));
        assert_eq!(worker::layers_that_fit(w31, p31, gib(14.9), 60), 24);
        assert_eq!(worker::layers_that_fit(w31, p31, gib(7.5), 60), 9);
        assert_eq!(worker::layers_that_fit(w31, p31, gib(1.0), 60), 0);
        assert!(worker::fits_whole(gib(5.0), gib(0.9), gib(14.9)));
    }

    /// The layer count comes straight from the GGUF header (the 31B has 60).
    #[test]
    #[ignore]
    fn gguf_block_count_31b() {
        let path = crate::tagger::resolve(GemmaModel::D31B.folder(), GemmaModel::D31B.model_file())
            .expect("Gemma 4 31B isn't installed");
        let t = std::time::Instant::now();
        assert_eq!(worker::gguf_block_count(&path), Some(60));
        eprintln!("read in {:?}", t.elapsed());
        assert_eq!(worker::gguf_block_count(std::path::Path::new("Cargo.toml")), None);
    }

    /// First load of the 31B: how long until Ready, and how badly a thread
    /// standing in for the UI (wakes every 16 ms) is held up meanwhile.
    /// Compare with the old ladder: set CLARITY_FULL_LADDER=1.
    #[test]
    #[ignore]
    fn llm_first_load_31b() {
        assert!(GemmaModel::D31B.installed(), "Gemma 4 31B isn't installed");
        let (_cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let beat = {
            let done = done.clone();
            std::thread::spawn(move || {
                let (mut worst, mut late) = (0.0f32, 0);
                while !done.load(std::sync::atomic::Ordering::SeqCst) {
                    let t = std::time::Instant::now();
                    std::thread::sleep(std::time::Duration::from_millis(16));
                    let gap = t.elapsed().as_secs_f32() * 1000.0;
                    worst = worst.max(gap);
                    late += usize::from(gap > 100.0);
                }
                (worst, late)
            })
        };
        let ctx = egui::Context::default();
        let t0 = std::time::Instant::now();
        let _worker =
            std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, ImageDetail::Standard));
        for msg in msg_rx {
            match msg {
                Msg::Status(s) => {
                    eprintln!("[status {:>6.1}s] {s}", t0.elapsed().as_secs_f32());
                    if s == "Ready" {
                        break;
                    }
                }
                Msg::Done(Err(e)) => panic!("load failed: {e}"),
                _ => {}
            }
        }
        done.store(true, std::sync::atomic::Ordering::SeqCst);
        let (worst, late) = beat.join().unwrap();
        eprintln!(
            "\n[first load] Ready after {:.1}s; UI stand-in: worst wake {worst:.0} ms, {late} wakes over 100 ms",
            t0.elapsed().as_secs_f32()
        );
    }

    /// Review harness for the Krea 2 prompt writer + Detailed Vision on the
    /// real 31B: each tests/example picture is turned into a Krea 2 prompt
    /// with tiling off, then on (same sampling), plus a words-only idea and a
    /// follow-up edit. Prints everything for side-by-side review and copies
    /// the cut sections to %TEMP%\krea_review_tiles for a visual check.
    #[test]
    #[ignore]
    fn llm_krea_review_31b() {
        assert!(GemmaModel::D31B.installed(), "Gemma 4 31B isn't installed");
        let names = [
            "pexels-maria-boutylina-2164524446-39873481.jpg",    // wide → 3 across
            "pexels-kamaleshwaran-p-2164694161-40077073.jpg",   // tall → 3 down
            "pexels-daniqux-12588915.jpg",                      // square-ish → 2 × 2
            "0652ac34-5aa1-4075-bc34-41b91cbc6641.jpeg",         // square-ish AI art
        ];
        let imgs: Vec<PathBuf> =
            names.iter().map(|n| PathBuf::from("tests/example").join(n)).collect();

        // The cut sections, for eyeballing the grids and overlap.
        let out = std::env::temp_dir().join("krea_review_tiles");
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        for (i, img) in imgs.iter().enumerate() {
            let tiles = worker::image_tiles(img, ImageDetail::Standard).unwrap().expect("big enough to tile");
            for (label, p) in tiles {
                std::fs::copy(&p, out.join(format!("{i}_{label}.png"))).unwrap();
            }
        }
        eprintln!("[tiles] {}", out.display());

        let brief = || {
            vec![
                CmdMsg { user: true, text: crate::krea_prompt::BRIEF.to_string(), image: None },
                CmdMsg { user: false, text: crate::krea_prompt::ACK.to_string(), image: None },
            ]
        };
        let params = super::GenParams { temperature: 0.4, max_tokens: 1500, ..Default::default() };
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        let mut labels = Vec::new();
        for (img, name) in imgs.iter().zip(names) {
            for detailed in [false, true] {
                let mut msgs = brief();
                msgs.push(CmdMsg { user: true, text: String::new(), image: Some(img.clone()) });
                cmd_tx.send(Cmd::Generate { msgs, params, pinned: 2, detailed }).unwrap();
                labels.push(format!("{name} detailed={detailed}"));
            }
        }
        let mut msgs = brief();
        msgs.push(CmdMsg {
            user: true,
            text: "an old lighthouse keeper reading a letter by lamplight during a storm".into(),
            image: None,
        });
        cmd_tx.send(Cmd::Generate { msgs, params, pinned: 2, detailed: true }).unwrap();
        labels.push("idea in words".into());
        drop(cmd_tx);

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, ImageDetail::Standard));
        let mut raw = String::new();
        let mut results = Vec::new();
        let t0 = std::time::Instant::now();
        let mut last = 0.0;
        for msg in msg_rx {
            match msg {
                Msg::ImageDetail { .. } => {}
                Msg::Status(s) => eprintln!("[status {:>6.1}s] {s}", t0.elapsed().as_secs_f32()),
                Msg::Token(t) => raw.push_str(&t),
                Msg::Done(r) => {
                    let (_, vis) = split_channels(&raw);
                    let now = t0.elapsed().as_secs_f32();
                    let i = results.len();
                    eprintln!(
                        "\n######## [{i}] {} — {r:?} — {:.0}s, {} words\n{vis}\n########\n",
                        labels[i],
                        now - last,
                        vis.split_whitespace().count()
                    );
                    last = now;
                    results.push((r, vis));
                    raw.clear();
                }
            }
        }
        handle.join().unwrap();
        assert_eq!(results.len(), labels.len());
        for (i, (r, vis)) in results.iter().enumerate() {
            assert_eq!(*r, Ok(()), "[{i}] {} failed", labels[i]);
            assert!(!vis.trim().is_empty(), "[{i}] {}: empty prompt", labels[i]);
        }
    }

    /// Detailed Vision grids: none for small images, the coarse grids for
    /// medium ones, the finer ones for big photos (no model needed).
    #[cfg(feature = "llm")]
    #[test]
    fn detail_tile_grids() {
        let dir = std::env::temp_dir().join("clarity_tile_grid_test");
        std::fs::create_dir_all(&dir).unwrap();
        let count = |w: u32, h: u32| {
            let p = dir.join(format!("{w}x{h}.png"));
            image::RgbImage::new(w, h).save(&p).unwrap();
            worker::image_tiles(&p, ImageDetail::Standard).unwrap().map(|t| t.len())
        };
        assert_eq!(count(800, 800), None, "small image goes whole");
        assert_eq!(count(1600, 1600), Some(4), "square-ish 2 x 2");
        assert_eq!(count(2400, 1200), Some(3), "wide 3 across");
        assert_eq!(count(1200, 2400), Some(3), "tall 3 down");
        assert_eq!(count(3600, 3600), Some(9), "big square-ish 3 x 3");
        assert_eq!(count(4800, 3000), Some(6), "big wide 3 x 2");
        assert_eq!(count(3000, 4500), Some(6), "big tall 2 x 3");
        // Image detail: High allows at most 2 × 2, Maximum sends it whole.
        let at = |w: u32, h: u32, level: ImageDetail| {
            let p = dir.join(format!("{w}x{h}.png"));
            worker::image_tiles(&p, level).unwrap().map(|t| t.len())
        };
        assert_eq!(at(3600, 3600, ImageDetail::High), Some(4), "High: big square-ish 2 x 2");
        assert_eq!(at(4800, 3000, ImageDetail::High), Some(2), "High: wide 2 across");
        assert_eq!(at(3000, 4500, ImageDetail::High), Some(2), "High: tall 2 down");
        assert_eq!(at(3600, 3600, ImageDetail::Maximum), None, "Maximum: no tiles");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Streamed text keeps characters split across tokens (no model needed):
    /// an emoji split 3+1, CJK one byte per token, an invalid byte.
    #[test]
    fn token_pieces_keep_split_characters() {
        let stream = |pieces: &[&[u8]]| -> String {
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            pieces.iter().map(|p| worker::decode_piece(&mut decoder, p)).collect()
        };
        let text = "Great job 😊 see you";
        let (b, i) = (text.as_bytes(), text.find('😊').unwrap());
        assert_eq!(stream(&[&b[..i + 3], &b[i + 3..]]), text);
        let cjk: Vec<&[u8]> = "中文字".as_bytes().chunks(1).collect();
        assert_eq!(stream(&cjk[..]), "中文字");
        assert_eq!(stream(&[&b"a\xFFb"[..], &b""[..], &b"c"[..]]), "a\u{FFFD}bc");
    }

    /// Writes the Detailed Vision sections of one tests/example photo to
    /// %TEMP%\detail_tiles_preview for a visual check.
    #[cfg(feature = "llm")]
    #[test]
    #[ignore]
    fn detail_tiles_preview() {
        let src = PathBuf::from("tests/example/pexels-vikkirillova-8813977.jpg");
        let out = std::env::temp_dir().join("detail_tiles_preview");
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        for (label, p) in worker::image_tiles(&src, ImageDetail::Standard).unwrap().unwrap() {
            std::fs::copy(&p, out.join(format!("{label}.png"))).unwrap();
        }
        eprintln!("{}", out.display());
    }

    /// A phone-style portrait (pixels stored sideways + EXIF "rotate 90°")
    /// must be tiled as the upright picture: same tall grid and the same
    /// sections as the upright original. Paths come from
    /// CLARITY_ROTATED_SAMPLE / CLARITY_UPRIGHT_SAMPLE (one photo saved both
    /// ways).
    #[cfg(feature = "llm")]
    #[test]
    #[ignore]
    fn detail_tiles_follow_exif_orientation() {
        let side = PathBuf::from(std::env::var("CLARITY_ROTATED_SAMPLE").expect("CLARITY_ROTATED_SAMPLE"));
        let up = PathBuf::from(std::env::var("CLARITY_UPRIGHT_SAMPLE").expect("CLARITY_UPRIGHT_SAMPLE"));
        let a = worker::image_tiles(&side, ImageDetail::Standard).unwrap().expect("tiles");
        let b = worker::image_tiles(&up, ImageDetail::Standard).unwrap().expect("tiles");
        let labels = |t: &[(&'static str, PathBuf)]| t.iter().map(|(l, _)| *l).collect::<Vec<&str>>();
        assert_eq!(labels(&a), labels(&b), "rotated photo got a different grid");
        for ((label, pa), (_, pb)) in a.iter().zip(&b) {
            let (ia, ib) = (image::open(pa).unwrap().to_rgb8(), image::open(pb).unwrap().to_rgb8());
            assert_eq!(ia.dimensions(), ib.dimensions(), "{label}: section shape differs");
            let diff: f64 = ia.as_raw().iter().zip(ib.as_raw()).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>()
                / ia.as_raw().len() as f64;
            eprintln!("{label}: mean abs diff {diff:.2}");
            assert!(diff < 6.0, "{label}: section content differs (sideways?) — diff {diff:.1}");
        }
        // The full image the model gets is upright too.
        let full = crate::image_cache::open_oriented(&side).unwrap();
        assert!(full.height() > full.width(), "full image still sideways");
    }

    /// Image detail on the real 31B (Vulkan build). Runs one worker per
    /// level with Thinking off, asks about tests/example's boat photo, and
    /// prints the level actually used, the reason, timing and the answer.
    /// With CLARITY_FORCE_DETAIL_PROBE set the offload/memory checks are
    /// skipped, so High/Maximum are probed (and measured) even on a GPU the
    /// 31B doesn't fit — the n_ubatch fix and memory numbers come from that.
    #[test]
    #[ignore]
    fn llm_image_detail_31b() {
        assert!(GemmaModel::D31B.installed(), "Gemma 4 31B isn't installed");
        let img = PathBuf::from("tests/example/pexels-vikkirillova-8813977.jpg");
        let levels: Vec<ImageDetail> = match std::env::var("CLARITY_DETAIL_LEVELS").ok().as_deref() {
            Some("max") => vec![ImageDetail::Maximum],
            Some("high") => vec![ImageDetail::High],
            _ => vec![ImageDetail::Maximum],
        };
        for level in levels {
            let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
            let (msg_tx, msg_rx) = std::sync::mpsc::channel();
            cmd_tx
                .send(Cmd::Generate {
                    params: GenParams { temperature: 0.0, max_tokens: 300, thinking: ThinkLevel::Off, ..Default::default() },
                    pinned: 0,
                    detailed: false,
                    msgs: vec![CmdMsg {
                        user: true,
                        text: "In one sentence: what is in her hair, what is on her ear, and what colour are her toenails?".into(),
                        image: Some(img.clone()),
                    }],
                })
                .unwrap();
            drop(cmd_tx);
            let ctx = egui::Context::default();
            let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, level));
            let t0 = std::time::Instant::now();
            let (mut reply, mut effective, mut done) = (String::new(), None, None);
            for msg in msg_rx {
                match msg {
                    Msg::Status(s) => eprintln!("[status {:>6.1}s] {s}", t0.elapsed().as_secs_f32()),
                    Msg::Token(t) => reply.push_str(&t),
                    Msg::ImageDetail { effective: e, reason } => {
                        eprintln!("[detail] requested {level:?} -> {e:?}; reason: {reason:?}");
                        effective = e;
                    }
                    Msg::Done(r) => done = Some(r),
                }
            }
            handle.join().unwrap();
            let (_, vis) = split_channels(&reply);
            eprintln!("[{level:?} -> {effective:?}] {:.0}s: {vis}", t0.elapsed().as_secs_f32());
            assert_eq!(done, Some(Ok(())), "generation failed");
            assert!(!vis.trim().is_empty(), "empty answer");
            assert!(effective.is_some(), "no image detail report");
        }
    }

    /// Thinking allowance on top of the reply length (real 31B): per level,
    /// counts thought tokens and visible-answer tokens. The answer must be a
    /// real answer (never the thought), stay within the reply length, and at
    /// Max not be squeezed short; the thought must stay within its allowance
    /// (+128 grace).
    #[test]
    #[ignore]
    fn llm_thinking_allowance_31b() {
        assert!(GemmaModel::D31B.installed(), "Gemma 4 31B isn't installed");
        let ask = "Write a 400-word story about an old lighthouse keeper. Just the story.";
        let cases = [(ThinkLevel::Max, 600usize), (ThinkLevel::Low, 256), (ThinkLevel::Off, 256)];
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel();
        for (level, max_tokens) in cases {
            cmd_tx
                .send(Cmd::Generate {
                    params: GenParams { max_tokens, thinking: level, temperature: 0.7, ..Default::default() },
                    pinned: 0,
                    detailed: false,
                    msgs: vec![CmdMsg { user: true, text: ask.into(), image: None }],
                })
                .unwrap();
        }
        drop(cmd_tx);
        let ctx = egui::Context::default();
        let handle =
            std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx, GemmaModel::D31B, ImageDetail::Standard));
        let t0 = std::time::Instant::now();
        let (mut raw, mut think_toks, mut answer_toks, mut n) = (String::new(), 0usize, 0usize, 0usize);
        for msg in msg_rx {
            match msg {
                Msg::Token(t) => {
                    let thinking = raw.contains("<|channel>") && !raw.contains("<channel|>");
                    raw.push_str(&t);
                    let still = raw.contains("<|channel>") && !raw.contains("<channel|>");
                    if thinking || still { think_toks += 1 } else { answer_toks += 1 }
                }
                Msg::Done(r) => {
                    let (level, max_tokens) = cases[n];
                    let (thought, vis) = split_channels(&raw);
                    eprintln!(
                        "\n### {level:?} (reply {max_tokens}) {:.0}s: {r:?} — thought ~{think_toks} tokens, answer ~{answer_toks} tokens, {} words\n{}\n###",
                        t0.elapsed().as_secs_f32(),
                        vis.split_whitespace().count(),
                        vis.chars().take(300).collect::<String>()
                    );
                    assert_eq!(r, Ok(()));
                    assert!(!vis.trim().is_empty(), "{level:?}: no visible answer");
                    assert!(vis != thought || thought.is_empty(), "{level:?}: thought shown as reply");
                    assert!(answer_toks <= max_tokens + 2, "{level:?}: answer {answer_toks} > reply length {max_tokens}");
                    assert!(think_toks <= level.budget().max(64) + 128 + 200, "{level:?}: thought {think_toks} over allowance");
                    if level == ThinkLevel::Max {
                        assert!(vis.split_whitespace().count() >= 250, "Max answer squeezed short");
                    }
                    raw.clear();
                    think_toks = 0;
                    answer_toks = 0;
                    n += 1;
                }
                _ => {}
            }
        }
        handle.join().unwrap();
        assert_eq!(n, cases.len());
    }
}
