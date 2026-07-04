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

/// True when this binary was compiled with local-AI support.
pub const BUILT_WITH_LLM: bool = cfg!(feature = "llm");

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

/// A request to the inference worker.
enum Cmd {
    Generate { prompt: String, image: Option<PathBuf> },
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

/// State for the AI Model tab, owned by `ViewerApp` (like `FtpState`).
pub struct LlmState {
    /// Both model files present on disk (refreshed when a download finishes).
    pub installed: bool,
    /// In-flight setup download, polled each frame.
    pub download: Option<crate::ai_models::DownloadHandle>,
    pub download_err: Option<String>,

    // --- Test area (Settings → AI Model → Try it) ---
    pub prompt: String,
    /// Optional image attached to the prompt (the "vision" in vision model).
    pub image: Option<PathBuf>,
    /// The streamed response so far.
    pub response: String,
    pub run_err: Option<String>,
    /// A generation is in flight.
    pub running: bool,
    /// Latest worker status line, shown next to the spinner.
    pub status: String,

    worker: Option<Worker>,
}

impl Default for LlmState {
    fn default() -> Self {
        Self {
            installed: installed(),
            download: None,
            download_err: None,
            prompt: String::new(),
            image: None,
            response: String::new(),
            run_err: None,
            running: false,
            status: String::new(),
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
        self.download = crate::ai_models::start_model_download(FOLDER);
        if self.download.is_none() {
            self.download_err = Some("Model missing from the catalog".to_string());
        }
    }

    /// Drive background work — poll the setup download and drain any streamed
    /// tokens from the inference worker. Call once per frame from the tab.
    pub fn poll(&mut self) {
        if let Some(dl) = &self.download {
            if dl.done() {
                if dl.ok() {
                    self.installed = installed();
                } else {
                    self.download_err =
                        Some(dl.error().unwrap_or_else(|| "Download failed".to_string()));
                }
                self.download = None;
            }
        }

        if let Some(w) = &self.worker {
            let mut worker_died = false;
            loop {
                match w.rx.try_recv() {
                    Ok(Msg::Status(s)) => self.status = s,
                    Ok(Msg::Token(t)) => self.response.push_str(&t),
                    Ok(Msg::Done(res)) => {
                        self.running = false;
                        if let Err(e) = res {
                            self.run_err = Some(e);
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        worker_died = true;
                        break;
                    }
                }
            }
            // A dead worker means the model failed to load (the error already
            // arrived as a Done). Drop it so the next Ask can start fresh.
            if worker_died {
                self.worker = None;
                self.running = false;
            }
        }
    }

    /// Send the current prompt (+ attached image) to the model, spawning the
    /// worker thread on first use. Tokens stream into `self.response`.
    pub fn generate(&mut self, ctx: &egui::Context) {
        if self.running || self.prompt.trim().is_empty() {
            return;
        }
        if !BUILT_WITH_LLM {
            self.run_err =
                Some("This build was compiled without the AI feature (`llm`).".to_string());
            return;
        }
        if !self.installed {
            self.run_err = Some("Set up the model first.".to_string());
            return;
        }
        self.run_err = None;
        self.response.clear();
        self.status = "Starting…".to_string();

        if self.worker.is_none() {
            self.worker = spawn_worker(ctx.clone());
        }
        let cmd = Cmd::Generate { prompt: self.prompt.clone(), image: self.image.clone() };
        match &self.worker {
            Some(w) if w.tx.send(cmd).is_ok() => self.running = true,
            _ => {
                self.worker = None;
                self.run_err = Some("The AI worker stopped — try again.".to_string());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inference worker — everything that touches llama.cpp lives behind the `llm`
// feature so a --no-default-features build needs no cmake/C++ toolchain.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "llm"))]
fn spawn_worker(_ctx: egui::Context) -> Option<Worker> {
    None
}

#[cfg(feature = "llm")]
fn spawn_worker(ctx: egui::Context) -> Option<Worker> {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (msg_tx, msg_rx) = std::sync::mpsc::channel::<Msg>();
    std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx));
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

    use super::{Cmd, Msg, FOLDER, MMPROJ_FILE, MODEL_FILE};

    /// Context window. Gemma 4 supports far more, but 4k keeps the KV cache
    /// small enough for CPU inference on ordinary machines.
    const N_CTX: u32 = 4096;
    /// Prompt-eval batch size; must fit an image's token chunk (256 for Gemma).
    const N_BATCH: u32 = 512;
    /// Cap on generated tokens per response.
    const MAX_TOKENS: usize = 1024;

    /// Worker body: load everything once, then serve Generate commands until
    /// the UI side drops its sender. On a load failure the error is reported
    /// as a `Done(Err)` and the thread exits (the UI respawns it on demand).
    pub(super) fn run(rx: Receiver<Cmd>, tx: Sender<Msg>, ctx: egui::Context) {
        let send = |m: Msg| {
            let _ = tx.send(m);
            ctx.request_repaint();
        };
        match load_and_serve(&rx, &send) {
            Ok(()) => {}
            Err(e) => send(Msg::Done(Err(e))),
        }
    }

    fn load_and_serve(rx: &Receiver<Cmd>, send: &dyn Fn(Msg)) -> Result<(), String> {
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

        let model_path = crate::tagger::resolve(FOLDER, MODEL_FILE)
            .ok_or_else(|| "Model file not found — run setup first".to_string())?;
        let mmproj_path = crate::tagger::resolve(FOLDER, MMPROJ_FILE)
            .ok_or_else(|| "Vision projector not found — run setup first".to_string())?;

        // Leave one core for the UI; llama.cpp saturates the rest.
        let threads = std::thread::available_parallelism()
            .map(|n| (n.get() as i32 - 1).max(1))
            .unwrap_or(4);

        send(Msg::Status("Loading the model (first time takes a minute)…".into()));
        let backend = LlamaBackend::init().map_err(|e| format!("llama.cpp init: {e}"))?;
        // Offload every layer to the GPU when a GPU backend is compiled in
        // (the `llm-vulkan` feature); a CPU-only build ignores this.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(1_000_000);
        let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
            .map_err(|e| format!("Load model: {e}"))?;

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
        let mtmd = MtmdContext::init_from_file(
            &mmproj_path.to_string_lossy(),
            &model,
            &mtmd_params,
        )
        .map_err(|e| format!("Load vision projector: {e}"))?;

        let template = model
            .chat_template(None)
            .map_err(|e| format!("Chat template: {e}"))?;

        send(Msg::Status("Ready".into()));

        while let Ok(Cmd::Generate { prompt, image }) = rx.recv() {
            let res = generate(&backend, &model, &mtmd, &template, threads, &prompt, image.as_deref(), send);
            send(Msg::Done(res));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate(
        backend: &LlamaBackend,
        model: &LlamaModel,
        mtmd: &MtmdContext,
        template: &llama_cpp_2::model::LlamaChatTemplate,
        threads: i32,
        prompt: &str,
        image: Option<&Path>,
        send: &dyn Fn(Msg),
    ) -> Result<(), String> {
        // A fresh context per prompt: cheap next to the resident weights, and
        // it guarantees no KV-cache state leaks between single-turn asks.
        let context_params = LlamaContextParams::default()
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_n_batch(N_BATCH)
            .with_n_ctx(NonZeroU32::new(N_CTX));
        let mut context = model
            .new_context(backend, context_params)
            .map_err(|e| format!("Create context: {e}"))?;

        // The media marker tells the tokenizer where the image's tokens go.
        // Without an explicit one, append it (matches llama.cpp's mtmd-cli).
        let marker = llama_cpp_2::mtmd::mtmd_default_marker();
        let mut prompt = prompt.to_string();
        let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
        if let Some(path) = image {
            send(Msg::Status("Reading the image…".into()));
            bitmaps.push(load_bitmap(mtmd, path)?);
            if !prompt.contains(marker) {
                prompt.push('\n');
                prompt.push_str(marker);
            }
        }

        send(Msg::Status("Thinking…".into()));
        let chat = vec![
            LlamaChatMessage::new("user".to_string(), prompt.clone()).map_err(|e| e.to_string())?,
        ];
        // Gemma 4 embeds a huge Jinja chat template (tool calls, thinking
        // channels) that llama.cpp's built-in non-Jinja formatter doesn't
        // recognise (ffi error -1). Its actual single-turn wire format is
        // simple, so fall back to formatting it by hand; the template path
        // still serves models with conventional templates.
        let formatted = match model.apply_chat_template(template, &chat, true) {
            Ok(s) => s,
            Err(_) => format!("<|turn>user\n{prompt}<turn|>\n<|turn>model\n"),
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

        // Two asks through one worker: plain text (the chat-template path)
        // and an image question (the vision path).
        cmd_tx
            .send(Cmd::Generate { prompt: "Say hello in one short sentence.".to_string(), image: None })
            .unwrap();
        cmd_tx
            .send(Cmd::Generate {
                prompt: "What colour is this image? Answer briefly.".to_string(),
                image: Some(img_path),
            })
            .unwrap();
        drop(cmd_tx); // worker exits after serving these

        let ctx = egui::Context::default();
        let handle = std::thread::spawn(move || worker::run(cmd_rx, msg_tx, ctx));

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
