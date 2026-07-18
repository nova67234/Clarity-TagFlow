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

use crate::theme::*;

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
const WORKER_PY: &str = r#"import json, os, re, sys, tempfile, traceback

# Emoji / pictographs / dingbats and their invisible plumbing — some break
# OmniVoice's text frontend outright. The app strips them too; this is the
# belt-and-braces layer.
UNSPEAKABLE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿︎️‍]"
)

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
            text = req["text"]
            stripped = UNSPEAKABLE.sub("", text)
            if stripped.strip():
                text = stripped
            kwargs = {"text": text}
            if req.get("ref_audio"):
                # Voice cloning: a reference recording + its transcript.
                kwargs["ref_audio"] = req["ref_audio"]
                if req.get("ref_text"):
                    kwargs["ref_text"] = req["ref_text"]
            elif req.get("instruct"):
                kwargs["instruct"] = req["instruct"]
            # Layered fallbacks so a bad voice never silences the reply:
            # sample -> description -> the model's default voice.
            try:
                audio = model.generate(**kwargs)
            except Exception as ve:
                if "ref_audio" in kwargs:
                    print("WARN voice sample not usable; using the voice description", flush=True)
                    kwargs.pop("ref_audio", None)
                    kwargs.pop("ref_text", None)
                    if req.get("instruct"):
                        kwargs["instruct"] = req["instruct"]
                    try:
                        audio = model.generate(**kwargs)
                    except (ValueError, TypeError) :
                        print("WARN voice description not accepted; using the default voice", flush=True)
                        kwargs.pop("instruct", None)
                        audio = model.generate(**kwargs)
                elif "instruct" in kwargs:
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

// The player thread that reads these only exists in `llm` builds.
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
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
    /// Voice cloning: reference recording + its transcript (synced from the
    /// persisted settings by main.rs). When set, it wins over `style`.
    pub ref_audio: String,
    pub ref_text: String,

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

    /// The floating always-on-top sample recorder popup.
    pub rec: Recorder,
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
            ref_audio: String::new(),
            ref_text: String::new(),
            engine: None,
            engine_rx: None,
            ready: false,
            loading: false,
            pending: 0,
            last_err: None,
            player_tx: None,
            speaking: Arc::new(AtomicBool::new(false)),
            rec: Recorder::default(),
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
        // A voice sample (cloning) wins over the description (voice design).
        let use_ref = !self.ref_audio.trim().is_empty()
            && std::path::Path::new(self.ref_audio.trim()).exists();
        let req = if use_ref {
            serde_json::json!({
                "text": text,
                "ref_audio": self.ref_audio.trim(),
                "ref_text": self.ref_text.trim(),
                // Still sent so the worker can fall back to it if the
                // sample can't be used.
                "instruct": self.style,
            })
        } else {
            serde_json::json!({ "text": text, "instruct": self.style })
        };
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
            // Setup downloaded the full model already — never let the model
            // load ping the Hub (fully offline, and no stall on flaky
            // connections).
            .env("HF_HUB_OFFLINE", "1")
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
                    if let (Ok(file), Ok(s)) = (file, new_sink)
                        && let Ok(dec) = rodio::Decoder::new(std::io::BufReader::new(file)) {
                            s.append(dec);
                            speaking.store(true, Relaxed);
                            sink = Some(s);
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
            if let Some(s) = &sink
                && s.empty() {
                    speaking.store(false, Relaxed);
                }
        }
    });
}

#[cfg(not(feature = "llm"))]
fn spawn_player_thread(rx: Receiver<PlayerCmd>, _speaking: Arc<AtomicBool>) {
    // No audio backend in a build without the AI feature.
    drop(rx);
}

// ---------------------------------------------------------------------------
// Voice-sample recorder — a small always-on-top popup (extra viewport) with a
// mic button. It records WHAT'S PLAYING (WASAPI loopback of the default
// output), so a voice heard on YouTube / in a game can be captured as a
// cloning sample without leaving the other app. Saved wavs land in
// models_root()/omnivoice/samples/ and auto-fill the Voice sample setting.
// ---------------------------------------------------------------------------

/// Recordings stop themselves at this length — cloning wants 3–10 s anyway.
const MAX_RECORD_SECS: u64 = 30;

/// State for the floating recorder popup, owned by `VoiceState`.
#[derive(Default)]
pub struct Recorder {
    /// The popup window is shown (toggled from the AI Model tab).
    pub open: bool,
    pub recording: bool,
    stop: Arc<AtomicBool>,
    started: Option<std::time::Instant>,
    rx: Option<Receiver<Result<PathBuf, String>>>,
    /// A finished recording, waiting for main.rs to adopt it as the
    /// cloning sample.
    pub saved: Option<PathBuf>,
    /// Keeps the "saved!" note visible after main.rs takes `saved`.
    pub just_saved: bool,
    pub err: Option<String>,
}

impl Recorder {
    pub fn toggle_recording(&mut self, ctx: &egui::Context) {
        if self.recording {
            self.stop.store(true, Relaxed);
        } else {
            self.start(ctx);
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        self.err = None;
        self.saved = None;
        self.just_saved = false;
        self.stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let stop = self.stop.clone();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            let res = record_loopback(&stop);
            let _ = tx.send(res);
            repaint.request_repaint();
        });
        self.recording = true;
        self.started = Some(std::time::Instant::now());
    }

    pub fn seconds(&self) -> u64 {
        self.started.map(|s| s.elapsed().as_secs()).unwrap_or(0)
    }

    pub fn poll(&mut self) {
        if let Some(rx) = &self.rx {
            match rx.try_recv() {
                Ok(Ok(path)) => {
                    self.recording = false;
                    self.rx = None;
                    self.saved = Some(path);
                    self.just_saved = true;
                }
                Ok(Err(e)) => {
                    self.recording = false;
                    self.rx = None;
                    self.err = Some(e);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.recording = false;
                    self.rx = None;
                }
            }
        }
    }
}

/// Capture the default output device (loopback) until stopped or the length
/// cap, then write a mono 16-bit wav into the samples dir.
#[cfg(feature = "llm")]
fn record_loopback(stop: &AtomicBool) -> Result<PathBuf, String> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use rodio::cpal::{self, SampleFormat};
    use std::sync::Mutex;

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("No output device to record from")?;
    let config = device
        .default_output_config()
        .map_err(|e| format!("Audio config: {e}"))?;
    if config.sample_format() != SampleFormat::F32 {
        return Err(format!("Unsupported sample format {:?}", config.sample_format()));
    }
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buf2 = buf.clone();
    // Building an INPUT stream on an OUTPUT device = WASAPI loopback: we get
    // exactly what the speakers are playing.
    let stream = device
        .build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                buf2.lock().unwrap().extend_from_slice(data);
            },
            |e| eprintln!("[recorder] stream error: {e}"),
            None,
        )
        .map_err(|e| format!("Couldn't open loopback capture: {e}"))?;
    stream.play().map_err(|e| format!("Couldn't start capture: {e}"))?;

    let started = std::time::Instant::now();
    while !stop.load(Relaxed) && started.elapsed().as_secs() < MAX_RECORD_SECS {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(stream);

    let samples = std::mem::take(&mut *buf.lock().unwrap());
    // Half a second of frames minimum, or there's nothing worth keeping.
    if samples.len() < (sample_rate as usize * channels) / 2 {
        return Err("Nothing captured — was anything playing?".to_string());
    }

    let dir = base_dir().join("samples");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let n = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) + 1;
    let path = dir.join(format!("sample_{n:03}.wav"));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec).map_err(|e| e.to_string())?;
    // Downmix to mono (cloning wants one voice, and it halves the file).
    for frame in samples.chunks_exact(channels) {
        let avg = frame.iter().sum::<f32>() / channels as f32;
        let v = (avg.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(not(feature = "llm"))]
fn record_loopback(_stop: &AtomicBool) -> Result<PathBuf, String> {
    Err("This build was compiled without the AI feature.".to_string())
}

/// Windows: ask DWM to clip the recorder window to Windows 11's rounded-corner
/// shape. The glow backend can't create transparent child viewports on Windows
/// (WGL pixel formats don't support transparency), so drawing a rounded frame
/// inside a transparent window — the approach used on other platforms — leaves
/// opaque black corners. Clipping the window itself at the OS level needs no
/// alpha channel. On Windows 10 the attribute doesn't exist and the call is a
/// harmless no-op (square corners, but no black artifacts).
#[cfg(windows)]
fn round_corners_win11() {
    use std::os::windows::ffi::OsStrExt as _;
    #[link(name = "user32")]
    unsafe extern "system" {
        fn FindWindowW(class: *const u16, title: *const u16) -> isize;
    }
    #[link(name = "dwmapi")]
    unsafe extern "system" {
        fn DwmSetWindowAttribute(
            hwnd: isize,
            attr: u32,
            value: *const std::ffi::c_void,
            size: u32,
        ) -> i32;
    }
    const DWMWA_WINDOW_CORNER_PREFERENCE: u32 = 33;
    const DWMWCP_ROUND: i32 = 2;
    let title: Vec<u16> = std::ffi::OsStr::new(RECORDER_TITLE)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        let hwnd = FindWindowW(std::ptr::null(), title.as_ptr());
        if hwnd != 0 {
            DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                (&DWMWCP_ROUND as *const i32).cast(),
                std::mem::size_of::<i32>() as u32,
            );
        }
    }
}

const RECORDER_TITLE: &str = "Record voice sample";

/// The floating recorder popup: a tiny frameless always-on-top window (its
/// own OS viewport, so it floats over other apps too). Drag anywhere to move;
/// mic toggles recording; ✕ closes.
pub fn recorder_window(ctx: &egui::Context, voice: &mut VoiceState) {
    if !voice.rec.open {
        return;
    }
    voice.rec.poll();

    ctx.show_viewport_immediate(
        egui::ViewportId::from_hash_of("voice_recorder"),
        egui::ViewportBuilder::default()
            .with_title(RECORDER_TITLE)
            .with_inner_size([236.0, 76.0])
            .with_always_on_top()
            .with_decorations(false)
            .with_resizable(false)
            // Transparency only works off-Windows (see round_corners_win11).
            .with_transparent(cfg!(not(windows))),
        |ctx, _class| {
            // An Area pinned at the origin fills the (tiny) viewport — the
            // ctx-level CentralPanel::show is deprecated in egui 0.34.
            let size = ctx.content_rect().size();
            egui::Area::new(egui::Id::new("voice_rec_root"))
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(0.0, 0.0))
                .show(ctx, |ui| {
                    // Windows: paint edge-to-edge and let DWM round the
                    // window itself (round_corners_win11) — a rounded frame
                    // here would expose black corners, since the viewport
                    // can't be transparent. Elsewhere the window IS
                    // transparent, so draw the pill shape ourselves.
                    let (radius, stroke) = if cfg!(windows) {
                        (egui::CornerRadius::ZERO, egui::Stroke::NONE)
                    } else {
                        (
                            egui::CornerRadius::same(16),
                            egui::Stroke::new(1.0, EDGE()),
                        )
                    };
                    egui::Frame::new()
                        .fill(PANEL())
                        .corner_radius(radius)
                        .inner_margin(egui::Margin::symmetric(12, 10))
                        .stroke(stroke)
                        .show(ui, |ui| {
                            ui.set_width(size.x - 24.0);
                            ui.set_height(size.y - 20.0);
                    // Drag anywhere on the pill to move the window.
                    let bg = ui.interact(
                        ui.max_rect(),
                        egui::Id::new("rec_drag"),
                        egui::Sense::drag(),
                    );
                    if bg.drag_started() {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }

                    ui.horizontal(|ui| {
                        // Mic button — red while recording.
                        let tint = if voice.rec.recording {
                            egui::Color32::from_rgb(230, 70, 70)
                        } else {
                            icon_tint(TEXT())
                        };
                        let (rect, resp) = ui.allocate_exact_size(
                            egui::vec2(34.0, 34.0),
                            egui::Sense::click(),
                        );
                        let resp = resp
                            .on_hover_text(if voice.rec.recording { "Stop" } else { "Record what's playing" })
                            .on_hover_cursor(egui::CursorIcon::PointingHand);
                        egui::Image::new(egui::include_image!("../icons/mic.svg"))
                            .tint(tint)
                            .paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(22.0, 22.0)));
                        if resp.clicked() {
                            voice.rec.toggle_recording(ui.ctx());
                        }

                        ui.vertical(|ui| {
                            let status = if voice.rec.recording {
                                format!("Recording… {}s (max {MAX_RECORD_SECS}s)", voice.rec.seconds())
                            } else if let Some(e) = &voice.rec.err {
                                e.clone()
                            } else if voice.rec.just_saved {
                                "Saved as the voice sample!".to_string()
                            } else {
                                "Play the voice, then hit the mic".to_string()
                            };
                            let color = if voice.rec.err.is_some() {
                                egui::Color32::from_rgb(210, 70, 70)
                            } else {
                                MUTED()
                            };
                            ui.label(egui::RichText::new(status).color(color).size(11.5));
                            ui.label(
                                egui::RichText::new("3–10s of one clean voice works best")
                                    .color(MUTED())
                                    .size(10.0),
                            );
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            if ui
                                .add(egui::Button::new(egui::RichText::new("✕").color(MUTED())).frame(false))
                                .on_hover_text("Close")
                                .clicked()
                            {
                                voice.rec.open = false;
                            }
                        });
                    });

                    if voice.rec.recording {
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
                    }
                        });
                });

            if ctx.input(|i| i.viewport().close_requested()) {
                voice.rec.open = false;
            }
        },
    );

    // Idempotent and cheap, so applied every frame — this also covers the OS
    // window being recreated (the viewport is destroyed whenever the popup
    // closes and rebuilt on reopen).
    #[cfg(windows)]
    round_corners_win11();
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

/// Streaming download with percentage progress lines, through the shared
/// resumable downloader (net.rs: `.part` temp, retry with backoff, Range
/// resume — and unlike the old inline version, `dest` only appears once the
/// download is complete).
fn download(url: &str, dest: &std::path::Path, send: &dyn Fn(String)) -> Result<(), String> {
    let mut last_pct = 0u64;
    crate::net::download(url, dest, "", &mut |note| match note {
        crate::net::Note::Progress { got, total } => {
            if let Some(pct) = (got * 100).checked_div(total)
                && pct >= last_pct + 5
            {
                last_pct = pct;
                send(format!("Downloading… {pct}%"));
            }
        }
        crate::net::Note::Retry { attempt, of, .. } => {
            send(format!("Connection dropped — retry {}/{}…", attempt - 1, of - 1));
        }
    })
}
