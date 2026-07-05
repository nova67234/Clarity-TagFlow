//! OmniVoice — a natural neural voice for the AI chat's "Listen" buttons
//! (src/ai_chat.rs), so role-play and hours-long sessions don't sound like a
//! talking robot.
//!
//! OmniVoice (k2-fsa, Apache-2.0, 600+ languages, voice design) is
//! PyTorch-only — there is no ONNX export yet — so it runs in a managed
//! Python sidecar, exactly like Pixal3D: the one-click setup in Settings →
//! AI Model downloads a standalone Python, GPU PyTorch (cu128, with a CPU
//! fallback), the `omnivoice` package, and the model weights. At runtime a
//! persistent worker process (`voice_worker.py`, written by the setup) loads
//! the model once and then synthesizes on demand: one JSON request per stdin
//! line, one `WAV <path>` / `ERR <msg>` reply per stdout line. Rust plays
//! the wav via rodio. The worker inherits the app's kill-on-exit Job Object,
//! so it can never outlive the app.
//!
//! The voice itself comes from OmniVoice's "voice design" mode — a text
//! description like "female, warm, natural" (user-tunable in Settings), no
//! reference audio needed. If/when OmniVoice ships ONNX support, this whole
//! sidecar can be swapped for an in-process backend without touching the UI.

use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use eframe::egui;

/// Default voice-design description (Settings → AI Model lets the user tune
/// it). OmniVoice accepts a FIXED attribute vocabulary — arbitrary words like
/// "warm" are rejected: gender (male/female), age (child/teenager/young
/// adult/middle-aged/elderly), pitch (very low/low/moderate/high/very high
/// pitch), "whisper", and accents (american/british/australian/canadian/
/// chinese/indian/japanese/korean/portuguese/russian accent).
pub const DEFAULT_STYLE: &str = "female, young adult, moderate pitch";

/// Standalone Python (astral-sh/python-build-standalone), same pinned release
/// as Pixal3D's.
const PY_TAG: &str = "20260602";
const PY_VER: &str = "3.12.13";
/// PyTorch CUDA wheel index — cu128 covers RTX 50-series (sm_120).
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
/// The OmniVoice model repo on HuggingFace (auto-downloaded during setup).
const MODEL_REPO: &str = "k2-fsa/OmniVoice";

/// The synth worker. Loads the model once (GPU when available), then serves
/// one JSON request per stdin line: {"text": "...", "instruct": "..."} →
/// "WAV <path>" (or "ERR <msg>") per stdout line. "READY" once loaded.
const WORKER_PY: &str = r#"import json, os, sys, tempfile, traceback

def main():
    import torch
    from omnivoice import OmniVoice
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    import soundfile as sf
    print("READY", flush=True)
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            kwargs = {"text": req["text"]}
            if req.get("instruct"):
                kwargs["instruct"] = req["instruct"]
            try:
                audio = model.generate(**kwargs)
            except (ValueError, TypeError) as ve:
                # A rejected voice description (fixed attribute vocabulary) or
                # a bad instruct/text pairing (tokenizer TypeError) shouldn't
                # silence the reply — retry with the model's default voice.
                if "instruct" in kwargs:
                    print("WARN voice description not accepted; using the default voice", flush=True)
                    kwargs.pop("instruct")
                    audio = model.generate(**kwargs)
                else:
                    raise ve
            n += 1
            path = os.path.join(tempfile.gettempdir(), "clarity_omnivoice_%d.wav" % (n % 2))
            sf.write(path, audio[0], 24000)
            print("WAV " + path, flush=True)
        except Exception as e:
            # The text goes to stderr (visible as [voice:err] in a console
            # build) so a failing input can be captured and reproduced.
            try:
                sys.stderr.write("failing text: %r\n" % req.get("text", "")[:300])
            except Exception:
                pass
            print("ERR %s: %s" % (type(e).__name__, str(e).replace("\n", " | ")), flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERR " + traceback.format_exc().replace("\n", " | "), flush=True)
"#;

fn base_dir() -> PathBuf {
    crate::tagger::models_root().join("omnivoice")
}

fn python_exe() -> PathBuf {
    if cfg!(windows) {
        base_dir().join("python").join("python.exe")
    } else {
        base_dir().join("python").join("bin").join("python3")
    }
}

fn worker_script() -> PathBuf {
    base_dir().join("voice_worker.py")
}

/// Written at the very end of a successful setup.
fn marker() -> PathBuf {
    base_dir().join("installed.ok")
}

/// True when the one-click setup has completed on this machine.
pub fn installed() -> bool {
    python_exe().exists() && worker_script().exists() && marker().exists()
}

fn py_tarball_url() -> String {
    let triple = if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else if cfg!(target_os = "macos") {
        "aarch64-apple-darwin"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    format!(
        "https://github.com/astral-sh/python-build-standalone/releases/download/{PY_TAG}/\
         cpython-{PY_VER}+{PY_TAG}-{triple}-install_only.tar.gz"
    )
}

enum SetupMsg {
    Line(String),
    Done(bool),
}

enum EngineMsg {
    Ready,
    Wav(PathBuf),
    /// Non-fatal notice (e.g. invalid voice description) — the wav still comes.
    Warn(String),
    Err(String),
    /// Worker process ended; carries the last stderr line for diagnosis.
    Exited(Option<String>),
}

enum PlayerCmd {
    Play(PathBuf),
    Stop,
}

struct Engine {
    child: Child,
    stdin: ChildStdin,
}

/// State for the OmniVoice setup + runtime, owned by `LlmState`.
pub struct VoiceState {
    pub installed: bool,

    // --- Setup (Settings → AI Model → Natural voice) ---
    pub setting_up: bool,
    pub setup_status: String,
    pub setup_failed: bool,
    setup_rx: Option<Receiver<SetupMsg>>,

    /// The voice-design description sent with every request (synced from the
    /// persisted setting by main.rs).
    pub style: String,

    // --- Runtime (the Python sidecar) ---
    engine: Option<Engine>,
    engine_rx: Option<Receiver<EngineMsg>>,
    /// Worker is up with the model loaded.
    pub ready: bool,
    /// Worker spawned, model still loading (first Listen takes a while).
    pub loading: bool,
    /// Synth requests sent and not yet answered.
    pub pending: usize,
    pub last_err: Option<String>,

    player_tx: Option<Sender<PlayerCmd>>,
    speaking: Arc<AtomicBool>,
}

impl Default for VoiceState {
    fn default() -> Self {
        Self {
            installed: installed(),
            setting_up: false,
            setup_status: String::new(),
            setup_failed: false,
            setup_rx: None,
            style: DEFAULT_STYLE.to_string(),
            engine: None,
            engine_rx: None,
            ready: false,
            loading: false,
            pending: 0,
            last_err: None,
            player_tx: None,
            speaking: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Drop for VoiceState {
    fn drop(&mut self) {
        // The job object would reap it anyway; be tidy on clean exits.
        if let Some(mut e) = self.engine.take() {
            let _ = e.child.kill();
        }
    }
}

impl VoiceState {
    /// True while OmniVoice audio is playing.
    pub fn speaking(&self) -> bool {
        self.speaking.load(Relaxed)
    }

    /// Stop playback and discard queued requests' results.
    pub fn stop(&mut self) {
        if let Some(tx) = &self.player_tx {
            let _ = tx.send(PlayerCmd::Stop);
        }
        self.speaking.store(false, Relaxed);
    }

    /// Drain setup/engine messages. Call once per frame (from `LlmState::poll`).
    pub fn poll(&mut self) {
        if let Some(rx) = &self.setup_rx {
            loop {
                match rx.try_recv() {
                    Ok(SetupMsg::Line(l)) => self.setup_status = l,
                    Ok(SetupMsg::Done(ok)) => {
                        self.setting_up = false;
                        self.setup_failed = !ok;
                        self.installed = installed();
                        if ok {
                            self.setup_status = "Voice installed".to_string();
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.setup_rx = None;
                        self.setting_up = false;
                        break;
                    }
                }
            }
        }

        // Drain first, then act — acting needs &mut self (the player).
        let mut msgs = Vec::new();
        if let Some(rx) = &self.engine_rx {
            loop {
                match rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        msgs.push(EngineMsg::Exited(None));
                        break;
                    }
                }
            }
        }
        for m in msgs {
            match m {
                EngineMsg::Ready => {
                    self.ready = true;
                    self.loading = false;
                    self.last_err = None;
                }
                EngineMsg::Wav(path) => {
                    self.pending = self.pending.saturating_sub(1);
                    let tx = self.player();
                    let _ = tx.send(PlayerCmd::Play(path));
                }
                EngineMsg::Warn(w) => self.last_err = Some(w),
                EngineMsg::Err(e) => {
                    self.pending = self.pending.saturating_sub(1);
                    self.loading = false;
                    self.last_err = Some(e);
                }
                EngineMsg::Exited(stderr_tail) => {
                    self.engine = None;
                    self.engine_rx = None;
                    self.ready = false;
                    self.pending = 0;
                    // A crash before/at model load would otherwise be silent —
                    // surface the last thing it said on stderr.
                    if self.loading || self.last_err.is_none() {
                        let detail = stderr_tail.unwrap_or_else(|| "no output".to_string());
                        self.last_err = Some(format!("The voice worker stopped ({detail})"));
                    }
                    self.loading = false;
                }
            }
        }
    }

    /// Speak `text` with the current voice style, spawning the sidecar on
    /// first use (the model load makes the very first Listen slow — `loading`
    /// is set so the UI can say so).
    pub fn speak(&mut self, text: &str, ctx: &egui::Context) {
        self.last_err = None;
        if self.engine.is_none() {
            match self.spawn_engine(ctx) {
                Ok(()) => self.loading = true,
                Err(e) => {
                    self.last_err = Some(e);
                    return;
                }
            }
        }
        let req = serde_json::json!({ "text": text, "instruct": self.style });
        if let Some(engine) = &mut self.engine {
            let line = format!("{req}\n");
            if engine.stdin.write_all(line.as_bytes()).is_err() {
                let _ = engine.child.kill();
                self.engine = None;
                self.ready = false;
                self.loading = false;
                self.last_err = Some("The voice worker stopped — try again.".to_string());
                return;
            }
            let _ = engine.stdin.flush();
            self.pending += 1;
        }
    }

    fn spawn_engine(&mut self, ctx: &egui::Context) -> Result<(), String> {
        // Children inherit the app's kill-on-exit Job Object, so the sidecar
        // (and its worker grandchildren) die with the app, crash included.
        #[cfg(windows)]
        crate::pixal3d::ensure_kill_on_exit_job();

        // Keep the installed worker script current with this build — setup
        // wrote the version of whenever it ran, and it may have been fixed.
        let _ = std::fs::write(worker_script(), WORKER_PY);

        let mut cmd = Command::new(python_exe());
        cmd.arg("-u")
            .arg(worker_script())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(base_dir());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn().map_err(|e| format!("Couldn't start the voice worker: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let stderr = child.stderr.take().ok_or("no stderr")?;

        // stderr: keep the newest line for crash diagnosis, mirror everything
        // to the console (visible when run from an IDE in debug builds).
        let stderr_tail = Arc::new(std::sync::Mutex::new(None::<String>));
        {
            let tail = stderr_tail.clone();
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
                    if !line.trim().is_empty() {
                        eprintln!("[voice:err] {line}");
                        *tail.lock().unwrap() = Some(line);
                    }
                }
            });
        }

        let (tx, rx) = mpsc::channel();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                eprintln!("[voice] {line}");
                let msg = if line == "READY" {
                    EngineMsg::Ready
                } else if let Some(p) = line.strip_prefix("WAV ") {
                    EngineMsg::Wav(PathBuf::from(p))
                } else if let Some(w) = line.strip_prefix("WARN ") {
                    EngineMsg::Warn(w.to_string())
                } else if let Some(e) = line.strip_prefix("ERR ") {
                    EngineMsg::Err(e.to_string())
                } else {
                    continue;
                };
                let _ = tx.send(msg);
                repaint.request_repaint();
            }
            let _ = tx.send(EngineMsg::Exited(stderr_tail.lock().unwrap().clone()));
            repaint.request_repaint();
        });

        self.engine = Some(Engine { child, stdin });
        self.engine_rx = Some(rx);
        Ok(())
    }

    /// The audio playback thread (created on first use). Owns the output
    /// stream; plays one wav at a time, newest wins.
    fn player(&mut self) -> Sender<PlayerCmd> {
        if let Some(tx) = &self.player_tx {
            return tx.clone();
        }
        let (tx, rx) = mpsc::channel::<PlayerCmd>();
        spawn_player_thread(rx, self.speaking.clone());
        self.player_tx = Some(tx.clone());
        tx
    }

    /// The one-click "Set up voice" button: everything runs on a background
    /// thread, streaming one-line progress into `setup_status`.
    pub fn start_setup(&mut self, ctx: &egui::Context) {
        if self.setting_up {
            return;
        }
        self.setting_up = true;
        self.setup_failed = false;
        self.setup_status = "Starting…".to_string();
        let (tx, rx) = mpsc::channel();
        self.setup_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let ok = run_setup(&tx, &ctx);
            let _ = tx.send(SetupMsg::Done(ok));
            ctx.request_repaint();
        });
    }
}

/// Playback via rodio (only present in `llm` builds — the same feature that
/// carries the AI chat this voice speaks for).
#[cfg(feature = "llm")]
fn spawn_player_thread(rx: Receiver<PlayerCmd>, speaking: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let Ok((_stream, handle)) = rodio::OutputStream::try_default() else {
            return;
        };
        let mut sink: Option<rodio::Sink> = None;
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(PlayerCmd::Play(path)) => {
                    if let Some(s) = &sink {
                        s.stop();
                    }
                    let file = std::fs::File::open(&path);
                    let new_sink = rodio::Sink::try_new(&handle);
                    if let (Ok(file), Ok(s)) = (file, new_sink) {
                        if let Ok(dec) = rodio::Decoder::new(std::io::BufReader::new(file)) {
                            s.append(dec);
                            speaking.store(true, Relaxed);
                            sink = Some(s);
                        }
                    }
                }
                Ok(PlayerCmd::Stop) => {
                    if let Some(s) = &sink {
                        s.stop();
                    }
                    speaking.store(false, Relaxed);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if let Some(s) = &sink {
                if s.empty() {
                    speaking.store(false, Relaxed);
                }
            }
        }
    });
}

#[cfg(not(feature = "llm"))]
fn spawn_player_thread(rx: Receiver<PlayerCmd>, _speaking: Arc<AtomicBool>) {
    // No audio backend in a build without the AI feature.
    drop(rx);
}

fn run_setup(tx: &Sender<SetupMsg>, ctx: &egui::Context) -> bool {
    #[cfg(windows)]
    crate::pixal3d::ensure_kill_on_exit_job();

    let send = |line: String| {
        let _ = tx.send(SetupMsg::Line(line));
        ctx.request_repaint();
    };

    let base = base_dir();
    if let Err(e) = std::fs::create_dir_all(&base) {
        send(format!("Could not create {}: {e}", base.display()));
        return false;
    }

    // 1 — standalone Python (shared release with Pixal3D, but its own copy so
    // the two features can't break each other's environments).
    let py = python_exe();
    if !py.exists() {
        send("[1/5] Downloading Python…".to_string());
        let tarball = base.join("python.tar.gz");
        if let Err(e) = download(&py_tarball_url(), &tarball, &send) {
            send(format!("Python download failed: {e}"));
            return false;
        }
        send("[1/5] Extracting Python…".to_string());
        if !run_logged(&send, &base, "tar", &["-xzf", "python.tar.gz"]) {
            send("Failed to extract Python".to_string());
            return false;
        }
        let _ = std::fs::remove_file(&tarball);
    }
    let py = py.to_string_lossy().to_string();

    // 2 — PyTorch (GPU cu128 first — RTX 50-series included — CPU fallback).
    send("[2/5] Installing PyTorch (GPU)… this is the big one".to_string());
    let gpu_ok = run_logged(
        &send,
        &base,
        &py,
        &[
            "-m", "pip", "install",
            "torch==2.8.0+cu128", "torchaudio==2.8.0+cu128",
            "--extra-index-url", TORCH_INDEX,
        ],
    );
    if !gpu_ok {
        send("[2/5] GPU PyTorch failed — installing CPU PyTorch instead".to_string());
        if !run_logged(&send, &base, &py, &["-m", "pip", "install", "torch", "torchaudio"]) {
            send("PyTorch install failed".to_string());
            return false;
        }
    }

    // 3 — OmniVoice + the wav writer its examples use.
    send("[3/5] Installing OmniVoice…".to_string());
    if !run_logged(&send, &base, &py, &["-m", "pip", "install", "omnivoice", "soundfile"]) {
        send("OmniVoice install failed".to_string());
        return false;
    }

    // 4 — the synth worker script (+ syntax check).
    send("[4/5] Writing the voice worker…".to_string());
    if let Err(e) = std::fs::write(worker_script(), WORKER_PY) {
        send(format!("Could not write worker script: {e}"));
        return false;
    }
    if !run_logged(&send, &base, &py, &["-m", "py_compile", "voice_worker.py"]) {
        send("Worker script failed to compile".to_string());
        return false;
    }

    // 5 — the model weights, so the first Listen doesn't surprise-download.
    send("[5/5] Downloading the OmniVoice model…".to_string());
    let fetch = format!(
        "from huggingface_hub import snapshot_download; snapshot_download('{MODEL_REPO}')"
    );
    if !run_logged(&send, &base, &py, &["-c", &fetch]) {
        send("Model download failed".to_string());
        return false;
    }

    let _ = std::fs::write(marker(), b"ok");
    true
}

/// Run a command from `dir`, forwarding its merged output line-by-line to the
/// status label. True on exit code 0.
fn run_logged(send: &dyn Fn(String), dir: &std::path::Path, program: &str, args: &[&str]) -> bool {
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(dir).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            send(format!("cannot run {program}: {e}"));
            return false;
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (ltx, lrx) = mpsc::channel::<String>();
    let mut readers = Vec::new();
    for pipe in [stdout.map(|s| Box::new(s) as Box<dyn Read + Send>), stderr.map(|s| Box::new(s) as Box<dyn Read + Send>)]
        .into_iter()
        .flatten()
    {
        let ltx = ltx.clone();
        readers.push(std::thread::spawn(move || {
            for line in std::io::BufReader::new(pipe).lines().map_while(Result::ok) {
                let _ = ltx.send(line);
            }
        }));
    }
    drop(ltx);
    for line in lrx {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            send(trimmed.to_string());
        }
    }
    for r in readers {
        let _ = r.join();
    }
    matches!(child.wait(), Ok(s) if s.success())
}

/// Streaming download with percentage progress lines (same ureq/NativeTls
/// setup as the model downloader — see ai_models.rs for why NativeTls).
fn download(url: &str, dest: &std::path::Path, send: &dyn Fn(String)) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .build()
        .into();
    let resp = agent.get(url).call().map_err(|e| e.to_string())?;
    let total: u64 = resp
        .headers()
        .get("Content-Length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut reader = resp.into_body().into_reader();
    let mut out = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 1 << 16];
    let mut got: u64 = 0;
    let mut last_pct = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        got += n as u64;
        if total > 0 {
            let pct = (got * 100 / total) as u32;
            if pct >= last_pct + 5 {
                last_pct = pct;
                send(format!("Downloading… {pct}%"));
            }
        }
    }
    Ok(())
}
