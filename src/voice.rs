//! OmniVoice — a natural neural voice for the AI chat's "Listen" buttons
//! (src/ai_chat.rs), so role-play and hours-long sessions don't sound like a
//! talking robot.
//!
//! OmniVoice (k2-fsa, Apache-2.0, 600+ languages, voice design) is
//! PyTorch-only — there is no ONNX export yet — so it runs in a managed
//! Python sidecar, exactly like Pixal3D: the one-click setup in Settings →
//! AI Model downloads a standalone Python, the PyTorch build that fits the
//! graphics card (NVIDIA → CUDA; a supported AMD Radeon on Windows → AMD's
//! ROCm build; otherwise CPU), the `omnivoice` package, and the model
//! weights. At runtime a persistent worker process (`voice_worker.py`,
//! rewritten at every start) loads the model once and then synthesizes on
//! demand: one JSON request per stdin line, one `WAV <path>` / `ERR <msg>`
//! reply per stdout line. The worker inherits the app's kill-on-exit Job
//! Object, so it can never outlive the app.
//!
//! A reply is spoken in sentence-sized pieces: the first is short, so the
//! voice starts within seconds; each next piece is requested the moment the
//! previous one is ready, so it synthesizes while that one plays (rodio
//! queues them). The worker adapts its diffusion steps so synthesis keeps up
//! with playback, keeps the reply in the voice of its first piece, and moves
//! to the CPU if the GPU fails.
//!
//! The voice itself comes from OmniVoice's "voice design" mode — a text
//! description like "female, warm, natural" (user-tunable in Settings), no
//! reference audio needed — or from a cloning sample. If/when OmniVoice
//! ships ONNX support, this whole sidecar can be swapped for an in-process
//! backend without touching the UI.

use std::collections::VecDeque;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, OnceLock};

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
/// as Pixal3D's. 3.12 is also the only Python AMD's Windows PyTorch supports.
const PY_TAG: &str = "20260602";
const PY_VER: &str = "3.12.13";
/// PyTorch CUDA wheel index — cu128 covers RTX 50-series (sm_120).
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
/// PyTorch's CPU-only wheels (a fraction of the CUDA build's size).
const CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
/// AMD's PyTorch for Windows (ROCm 7.2.1 — Radeon RX 7000/9000, Radeon PRO
/// W7000 / AI PRO, Ryzen AI Max and Ryzen AI 300 graphics), installed exactly
/// as AMD's guide says: the ROCm runtime first, then PyTorch built on it.
/// Needs Python 3.12 and AMD Software: Adrenalin Edition 26.2.2 or newer.
/// https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html
const ROCM_SDK: [&str; 4] = [
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
];
const ROCM_TORCH: [&str; 2] = [
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
];
/// What a failed ROCm attempt leaves behind (several GB), removed before the
/// CPU build goes in.
const ROCM_PACKAGES: [&str; 6] =
    ["torch", "torchaudio", "rocm", "rocm-sdk-core", "rocm-sdk-devel", "rocm-sdk-libraries-custom"];
/// The OmniVoice model repo on HuggingFace (auto-downloaded during setup).
const MODEL_REPO: &str = "k2-fsa/OmniVoice";

/// The synth worker. Loads the model once (on the GPU when PyTorch can drive
/// it, else the CPU), announces `READY {json}` (where it runs), then serves
/// one JSON request per stdin line — {"text", "instruct", "ref_audio",
/// "ref_text", "cont": continues the previous piece's reply, "more": more
/// pieces follow} — answering "WAV <path>" (or "ERR <msg>"); "WARN <msg>" for
/// a voice that fell back, "LOG <stage>" while loading.
const WORKER_PY: &str = r##"import json, os, re, sys, tempfile, time, traceback

# Emoji / pictographs / dingbats and their invisible plumbing — some break
# OmniVoice's text frontend outright. The app strips them too; this is the
# belt-and-braces layer.
UNSPEAKABLE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿︎️‍]")

# The app speaks a reply piece by piece (split at sentences), and the next
# piece synthesizes while this one plays. The diffusion steps adapt so that
# synthesis runs at about this fraction of real time: OmniVoice's default 32
# where the hardware keeps up, fewer (down to 8) where it doesn't, so speech
# keeps flowing instead of stalling — a CPU at 32 steps needs about three
# seconds per second of speech.
TARGET_RTF = 0.7
MIN_STEPS, MAX_STEPS, CPU_STEPS = 8, 32, 12

# Words that mark an exception as the GPU's (driver, memory, kernels): the
# voice then moves to the CPU instead of failing every reply.
GPU_WORDS = ("HIP", "hip", "CUDA", "cuda", "MIOpen", "rocBLAS", "hipBLAS",
             "cuBLAS", "CUBLAS", "out of memory", "OutOfMemory", "device-side")


def say(kind, text=""):
    print((kind + " " + text).rstrip(), flush=True)


def first_line(e):
    lines = str(e).strip().splitlines()
    return (lines[0] if lines else type(e).__name__)[:200]


def gpu_trouble(e):
    s = type(e).__name__ + " " + str(e)
    return any(w in s for w in GPU_WORDS)


class Voice:
    def __init__(self):
        import torch
        self.torch = torch
        if getattr(torch.version, "hip", None):
            # AMD: MIOpen (cuDNN's stand-in) searches and compiles kernels
            # for every new tensor shape — and every sentence is a new
            # length, so each Listen stalled for seconds to minutes.
            # PyTorch's own convolutions need none of that.
            torch.backends.cudnn.enabled = False
        from omnivoice import OmniVoice
        self.OmniVoice = OmniVoice
        self.note = ""
        gpu = self.find_gpu()
        if gpu:
            try:
                self.load(*gpu)
                return
            except Exception as e:
                self.gpu_failed(e)
        self.load("cpu", "CPU", "CPU")

    def find_gpu(self):
        torch = self.torch
        if not torch.cuda.is_available():
            return None
        forced = os.environ.get("CLARITY_VOICE_CPU")
        if forced:
            # The app saw this GPU crash the worker outright.
            self.note = forced
            return None
        # An integrated GPU beside a graphics card: the card (more memory).
        i = max(range(torch.cuda.device_count()),
                key=lambda d: torch.cuda.get_device_properties(d).total_memory)
        hip = getattr(torch.version, "hip", None)
        if hip:
            backend = "ROCm " + ".".join(hip.split(".")[:2])
        else:
            backend = "CUDA " + str(torch.version.cuda)
        return "cuda:%d" % i, torch.cuda.get_device_name(i), backend

    def load(self, device, name, backend):
        torch = self.torch
        self.model = None
        say("LOG", "load-gpu" if device != "cpu" else "load")
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = self.OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        self.device, self.name, self.backend = device, name, backend
        self.steps = MAX_STEPS if device != "cpu" else CPU_STEPS
        self.anchor = None
        self.clones = {}
        self.bad_style = None
        # A GPU's first run initializes its kernels and libraries: pay that
        # now, while the app shows "loading", not on the first reply.
        say("LOG", "warm-gpu" if device != "cpu" else "warm")
        self.model.generate(text="Hello there.", num_step=4)

    def gpu_failed(self, e):
        traceback.print_exc()
        self.note = "the GPU failed (%s)" % first_line(e)
        self.model = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def ready(self):
        say("READY", json.dumps({
            "device": "cpu" if self.device == "cpu" else "gpu",
            "name": self.name,
            "backend": self.backend,
            "note": self.note,
        }))

    def speak(self, req):
        text = req["text"]
        stripped = UNSPEAKABLE.sub("", text)
        if stripped.strip():
            text = stripped
        if not req.get("cont"):
            self.anchor = None  # a new reply
        try:
            return self.render(text, req)
        except Exception as e:
            if self.device == "cpu" or not gpu_trouble(e):
                raise
            self.gpu_failed(e)
            self.load("cpu", "CPU", "CPU")
            self.ready()
            return self.render(text, req)

    def render(self, text, req):
        steps = self.steps
        start = time.perf_counter()
        audio = None
        prompt = self.sample(req)
        if prompt is not None:
            try:
                audio = self.model.generate(
                    text=text, voice_clone_prompt=prompt, num_step=steps)[0]
            except Exception as e:
                if gpu_trouble(e):
                    raise
                traceback.print_exc()
                self.clones[self.sample_key] = None
                say("WARN", "voice sample not usable; using the voice description")
        if audio is None:
            audio = self.design(text, req, steps)
        self.pace(time.perf_counter() - start, len(audio) / self.model.sampling_rate, steps)
        return audio

    def sample(self, req):
        """The cloning sample, encoded once (VoiceClonePrompt), or None."""
        path = (req.get("ref_audio") or "").strip()
        if not path:
            return None
        ref_text = (req.get("ref_text") or "").strip() or None
        try:
            key = (path, os.path.getmtime(path), ref_text)
        except OSError:
            return None
        if key not in self.clones:
            prompt = None
            try:
                with self.torch.inference_mode():
                    prompt = self.model.create_voice_clone_prompt(
                        ref_audio=path, ref_text=ref_text)
            except Exception as e:
                if gpu_trouble(e):
                    raise
                sys.stderr.write("voice sample: %s\n" % first_line(e))
            self.clones = {key: prompt}
        self.sample_key = key
        if self.clones[key] is None and not req.get("cont"):
            say("WARN", "voice sample not usable; using the voice description")
        return self.clones[key]

    def design(self, text, req, steps):
        """Voice design (the description), or the default voice."""
        kw = {"text": text, "num_step": steps}
        if self.anchor is not None:
            # Later pieces of a reply: the voice its first piece came out in.
            kw["voice_clone_prompt"] = self.anchor
        style = (req.get("instruct") or "").strip()
        if style and style != self.bad_style:
            try:
                return self.anchored(self.model.generate(instruct=style, **kw)[0], text, req)
            except ValueError as e:
                if "instruct" not in str(e):
                    raise
                self.bad_style = style
                sys.stderr.write("voice description: %s\n" % first_line(e))
                say("WARN", "voice description not accepted; using the default voice")
        elif style and not req.get("cont"):
            say("WARN", "voice description not accepted; using the default voice")
        return self.anchored(self.model.generate(**kw)[0], text, req)

    def anchored(self, audio, text, req):
        """Keep a reply's first piece as the voice of its other pieces — how
        OmniVoice keeps a long text in one voice (chunk 0 is the reference
        for the rest). Without it, every piece could be a new speaker."""
        sr = self.model.sampling_rate
        if self.anchor is None and req.get("more") and len(audio) >= 1.5 * sr:
            try:
                with self.torch.inference_mode():
                    wav = self.torch.from_numpy(audio.astype("float32").reshape(1, -1))
                    self.anchor = self.model.create_voice_clone_prompt(
                        ref_audio=(wav, sr), ref_text=text)
            except Exception as e:
                if gpu_trouble(e):
                    raise
                sys.stderr.write("voice anchor: %s\n" % first_line(e))
        return audio

    def pace(self, took, secs, steps):
        if secs < 0.5:
            return
        rtf = took / secs
        want = max(MIN_STEPS, min(MAX_STEPS, int(steps * TARGET_RTF / max(rtf, 1e-3))))
        # Halfway there each time: one odd sentence mustn't swing it.
        if want < self.steps:
            self.steps = (self.steps + want) // 2
        else:
            self.steps = (self.steps + want + 1) // 2
        say("DBG", "%.1fs of speech in %.1fs at %d steps on %s; next %d"
            % (secs, took, steps, self.name, self.steps))


def main():
    voice = Voice()
    import soundfile as sf
    voice.ready()
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = {}
        try:
            req = json.loads(line)
            audio = voice.speak(req)
            n += 1
            # Four slots: the app reads each clip the moment it's announced
            # and asks for the next piece only then.
            path = os.path.join(tempfile.gettempdir(), "clarity_omnivoice_%d.wav" % (n % 4))
            sf.write(path, audio, voice.model.sampling_rate)
            say("WAV", path)
        except Exception as e:
            # The text goes to stderr (visible as [voice:err] in a console
            # build) so a failing input can be captured and reproduced.
            try:
                sys.stderr.write("failing text: %r\n" % req.get("text", "")[:300])
            except Exception:
                pass
            say("ERR", "%s: %s" % (type(e).__name__, str(e).replace("\n", " | ")))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("ERR", traceback.format_exc().replace("\n", " | "))
"##;

/// Setup's GPU check: a few real kernels (matrix multiply, convolution,
/// attention) with the voice's own Python — `torch.cuda.is_available()`
/// alone says yes on GPUs whose kernels then fail. Prints "GPU OK: <name>";
/// a wedged driver can't hang setup (the watchdog ends it after 5 minutes).
const GPU_CHECK: &str = r##"import faulthandler, sys
faulthandler.dump_traceback_later(300, exit=True)
import torch
import torch.nn.functional as F
if getattr(torch.version, "hip", None):
    torch.backends.cudnn.enabled = False
if not torch.cuda.is_available():
    sys.exit("PyTorch can't see the GPU")
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
y = (x @ x).float()
c = F.conv1d(torch.randn(1, 32, 1000, device="cuda"), torch.randn(32, 32, 7, device="cuda"), padding=3)
q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
a = F.scaled_dot_product_attention(q, q, q).float()
torch.cuda.synchronize()
if not all(bool(torch.isfinite(t).all()) for t in (y, c, a)):
    sys.exit("the GPU computed garbage")
print("GPU OK: " + torch.cuda.get_device_name(0), flush=True)
"##;

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

/// Written at the very end of a successful setup (see `SetupInfo`).
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

/// The PyTorch build the voice runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Build {
    /// NVIDIA: PyTorch's CUDA 12.8 wheels.
    Cuda,
    /// AMD Radeon on Windows: AMD's ROCm 7.2.1 wheels.
    Rocm,
    Cpu,
}

impl Build {
    fn key(self) -> &'static str {
        match self {
            Build::Cuda => "cuda",
            Build::Rocm => "rocm",
            Build::Cpu => "cpu",
        }
    }

    fn from_key(s: &str) -> Option<Self> {
        [Build::Cuda, Build::Rocm, Build::Cpu].into_iter().find(|b| b.key() == s)
    }
}

/// What setup installed, kept in `installed.ok` as key=value lines: the
/// PyTorch build, the GPU its check passed on, the GPU build it tried, and
/// why the GPU isn't used (when it isn't). Installs from before this was
/// recorded hold just "ok" — and they always tried CUDA.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SetupInfo {
    pub torch: Option<Build>,
    /// The GPU the check passed on ("" = none).
    pub gpu: String,
    tried: Option<Build>,
    /// Why the voice can't use the GPU ("" when it can, or there's none).
    pub note: String,
}

impl SetupInfo {
    fn parse(text: &str) -> Self {
        let mut info = Self::default();
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else { continue };
            let value = value.trim();
            match key.trim() {
                "torch" => info.torch = Build::from_key(value),
                "gpu" => info.gpu = value.to_string(),
                "tried" => info.tried = Build::from_key(value),
                "note" => info.note = value.to_string(),
                _ => {}
            }
        }
        info
    }

    fn render(&self) -> String {
        let one_line = |s: &str| s.replace(['\r', '\n'], " ");
        let mut out = String::from("ok\n");
        if let Some(t) = self.torch {
            out.push_str(&format!("torch={}\n", t.key()));
        }
        out.push_str(&format!("gpu={}\n", one_line(&self.gpu)));
        if let Some(t) = self.tried {
            out.push_str(&format!("tried={}\n", t.key()));
        }
        out.push_str(&format!("note={}\n", one_line(&self.note)));
        out
    }

    /// The GPU build setup last attempted (`Cpu` when it went straight for
    /// the CPU one).
    fn tried(&self) -> Build {
        match (self.tried, self.torch) {
            (Some(t), _) => t,
            (None, None) => Build::Cuda,
            (None, Some(_)) => Build::Cpu,
        }
    }
}

fn read_setup() -> SetupInfo {
    std::fs::read_to_string(marker()).map(|s| SetupInfo::parse(&s)).unwrap_or_default()
}

/// This machine's graphics adapters by name ("AMD Radeon(TM) 8060S
/// Graphics", "NVIDIA GeForce RTX 4090"…); empty when they can't be listed.
/// Windows asks WMI, which takes a second or so — keep it off the UI thread.
/// Linux learns only the vendors (sysfs); macOS needs none of this.
fn detect_adapters() -> Vec<String> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        // Output, whatever the exit code: `reg query /s` fails on the
        // access-denied subkeys after printing what it found.
        let run = |program: &str, args: &[&str]| -> String {
            Command::new(program)
                .args(args)
                .stdin(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(CREATE_NO_WINDOW)
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default()
        };
        let mut names: Vec<String> = run(
            "powershell",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ],
        )
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
        if names.is_empty() {
            // PowerShell locked down: the display adapters' driver
            // descriptions, straight from the registry
            // ("    DriverDesc    REG_SZ    AMD Radeon(TM) 8060S Graphics").
            const DISPLAY_CLASS: &str =
                r"HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
            names = run("reg", &["query", DISPLAY_CLASS, "/s", "/v", "DriverDesc"])
                .lines()
                .filter_map(|l| l.split_once("REG_SZ"))
                .map(|(_, name)| name.trim().to_string())
                .filter(|n| !n.is_empty())
                .collect();
            names.dedup();
        }
        names
    }
    #[cfg(target_os = "linux")]
    {
        let mut names: Vec<String> = std::fs::read_dir("/sys/class/drm")
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path().join("device/vendor")).ok())
            .filter_map(|v| match v.trim() {
                "0x10de" => Some("NVIDIA GPU".to_string()),
                "0x1002" => Some("AMD GPU".to_string()),
                "0x8086" => Some("Intel GPU".to_string()),
                _ => None,
            })
            .collect();
        names.sort();
        names.dedup();
        names
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Vec::new()
    }
}

/// `detect_adapters`, run once in the background for the UI: None until
/// known (a repaint follows).
fn adapters(ctx: &egui::Context) -> Option<&'static [String]> {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    static ASKED: AtomicBool = AtomicBool::new(false);
    if let Some(names) = NAMES.get() {
        return Some(names);
    }
    if !ASKED.swap(true, Relaxed) {
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = NAMES.set(detect_adapters());
            ctx.request_repaint();
        });
    }
    None
}

/// AMD GPUs that AMD's PyTorch for Windows (ROCm 7.2.1) supports, by name:
/// Radeon RX 7000 / 9000, Radeon PRO W7000 and AI PRO, and the integrated
/// Radeon 8060S / 8050S / 8040S (Ryzen AI Max) and 890M / 880M (Ryzen AI
/// 300). Older ones (RX 6000, Radeon 780M…) would download gigabytes that
/// can't run.
fn rocm_supported(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("radeon")
        && ["rx 7", "rx 9", "pro w7", "ai pro r", "8060s", "8050s", "8040s", "890m", "880m"]
            .iter()
            .any(|m| n.contains(m))
}

/// The PyTorch build for these adapters, and the adapter it's for: NVIDIA →
/// CUDA; a supported Radeon (Windows only — AMD's wheels are Windows ones)
/// → ROCm; others → CPU. Unknown hardware gets CUDA as before (that build
/// runs on the CPU too); a Mac gets the CPU build (with Apple's GPU support).
fn pick_build(adapters: &[String], windows: bool, mac: bool) -> (Build, Option<String>) {
    if mac {
        return (Build::Cpu, None);
    }
    if let Some(n) = adapters.iter().find(|n| n.to_ascii_lowercase().contains("nvidia")) {
        return (Build::Cuda, Some(n.clone()));
    }
    if windows && let Some(n) = adapters.iter().find(|n| rocm_supported(n)) {
        return (Build::Rocm, Some(n.clone()));
    }
    if adapters.is_empty() {
        return (Build::Cuda, None);
    }
    (Build::Cpu, None)
}

/// Where the running worker put the voice (its READY line).
#[derive(serde::Deserialize, Clone, Debug, Default, PartialEq)]
pub struct RunsOn {
    /// "gpu" or "cpu".
    pub device: String,
    pub name: String,
    /// "CUDA 12.8", "ROCm 7.2" or "CPU".
    pub backend: String,
    /// Why it's on the CPU although a GPU build is installed ("" otherwise).
    #[serde(default)]
    pub note: String,
}

enum SetupMsg {
    Line(String),
    Done(bool),
}

enum EngineMsg {
    /// The model is loaded; where it runs (None from a worker too old to say).
    Ready(Option<RunsOn>),
    /// A loading stage: "load", "warm" (on the CPU), "load-gpu", "warm-gpu".
    Log(String),
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
    /// Queue a clip after whatever is playing — a reply's pieces play back
    /// to back.
    Play(PathBuf),
    /// Stop, dropping everything queued.
    Stop,
}

struct Engine {
    child: Child,
    stdin: ChildStdin,
}

/// State for the OmniVoice setup + runtime, owned by `LlmState`.
pub struct VoiceState {
    pub installed: bool,
    /// What setup installed (`installed.ok`).
    pub setup: SetupInfo,

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
    /// The loading stage the worker last reported.
    stage: Option<String>,
    /// Where the running worker put the voice (None until it's loaded).
    pub runs_on: Option<RunsOn>,
    /// The worker died while on the GPU (a driver crash kills the process
    /// outright — no Python error to catch): it restarts on the CPU for the
    /// rest of this session instead of crashing at every Listen.
    gpu_crashed: bool,
    /// Synth requests sent and not yet answered.
    pub pending: usize,
    /// How many of the oldest `pending` answers to throw away unplayed —
    /// requests that were still synthesizing when Stop was pressed. The
    /// worker answers in order, so the next `discard` results are theirs.
    discard: usize,
    /// The rest of the reply being spoken: pieces not sent yet. Each goes
    /// when the previous one's audio arrives, so it synthesizes while that
    /// one plays — and Stop has nothing queued in the worker to wait out.
    queue: VecDeque<String>,
    pub last_err: Option<String>,

    player_tx: Option<Sender<PlayerCmd>>,
    speaking: Arc<AtomicBool>,

    /// The floating always-on-top sample recorder popup.
    pub rec: Recorder,
}

/// The worker's synthesized clips (`clarity_omnivoice_{0..3}.wav` in the
/// temp folder, see WORKER_PY) — spoken chat replies.
const CLIP_SLOTS: usize = 4;

/// Delete the worker's synthesized clips: at exit, and at startup in case a
/// crash skipped that.
fn clear_clips() {
    for n in 0..CLIP_SLOTS {
        let _ = std::fs::remove_file(std::env::temp_dir().join(format!("clarity_omnivoice_{n}.wav")));
    }
}

impl Default for VoiceState {
    fn default() -> Self {
        clear_clips();
        Self {
            installed: installed(),
            setup: read_setup(),
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
            stage: None,
            runs_on: None,
            gpu_crashed: false,
            pending: 0,
            discard: 0,
            queue: VecDeque::new(),
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
        clear_clips();
    }
}

impl VoiceState {
    /// True while OmniVoice audio is playing.
    pub fn speaking(&self) -> bool {
        self.speaking.load(Relaxed)
    }

    /// True while a reply is playing OR still being synthesized (including
    /// the first-use model load) — the window in which Listen means "Stop".
    pub fn busy(&self) -> bool {
        self.speaking() || self.pending > self.discard || !self.queue.is_empty()
    }

    /// Speech is on its way but nothing is playing — the model loading, or
    /// a piece synthesizing — so the chat shows its spinner. (After Stop the
    /// model may go on loading, unannounced, for the next Listen.)
    pub fn waiting(&self) -> bool {
        self.pending > self.discard && !self.speaking()
    }

    /// Stop playback, drop the reply's unsent pieces, and discard the
    /// results of requests still synthesizing — they'd play after Stop.
    pub fn stop(&mut self) {
        if let Some(tx) = &self.player_tx {
            let _ = tx.send(PlayerCmd::Stop);
        }
        self.speaking.store(false, Relaxed);
        self.discard = self.pending;
        self.queue.clear();
    }

    /// The line beside the chat's spinner: the loading stage, or — when the
    /// voice runs on the CPU and this machine has a GPU that setup hasn't
    /// tried — where to fix that.
    pub fn status(&self, ctx: &egui::Context) -> Option<String> {
        if self.loading {
            let stage = match self.stage.as_deref() {
                Some("warm-gpu") => "Warming up the GPU…",
                Some("warm") => "Warming up…",
                Some("load-gpu") => "Loading the voice onto the GPU…",
                _ => "Loading the voice…",
            };
            return Some(stage.to_string());
        }
        let on_cpu = self.runs_on.as_ref().is_some_and(|r| r.device == "cpu");
        if on_cpu && let Some(gpu) = self.gpu_upgrade(ctx) {
            return Some(format!(
                "Slow on the CPU — Settings → AI Model can move the voice to your {gpu}"
            ));
        }
        None
    }

    /// "Runs on …" for Settings: where the running worker put the voice,
    /// else what setup installed for. None when unknown (an install from
    /// before this was recorded, not started yet).
    pub fn runs_on_line(&self) -> Option<String> {
        if let Some(r) = &self.runs_on {
            return Some(if r.device == "gpu" {
                format!("Runs on the GPU: {} ({})", r.name, r.backend)
            } else if r.note.is_empty() {
                "Runs on the CPU".to_string()
            } else {
                format!("Runs on the CPU — {}", r.note)
            });
        }
        let s = &self.setup;
        let via = match s.torch? {
            Build::Cpu if s.note.is_empty() => return Some("Runs on the CPU".to_string()),
            Build::Cpu => return Some(format!("Runs on the CPU — {}", s.note)),
            Build::Rocm => "AMD ROCm",
            Build::Cuda => "NVIDIA CUDA",
        };
        Some(if s.gpu.is_empty() {
            format!("Runs on the CPU — {}", s.note)
        } else {
            format!("Runs on the GPU: {} ({via})", s.gpu)
        })
    }

    /// A GPU this machine has that setup hasn't tried yet — e.g. a Radeon
    /// on an install from before AMD support, which left the voice on the
    /// CPU. Its name, for the "Set up for this GPU" offer.
    pub fn gpu_upgrade(&self, ctx: &egui::Context) -> Option<String> {
        if !self.installed || self.setting_up {
            return None;
        }
        let (build, adapter) = pick_build(adapters(ctx)?, cfg!(windows), cfg!(target_os = "macos"));
        (build != Build::Cpu && self.setup.tried() != build)
            .then(|| adapter.unwrap_or_else(|| "GPU".to_string()))
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
                        self.setup = read_setup();
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
                EngineMsg::Ready(info) => {
                    self.ready = true;
                    self.loading = false;
                    self.stage = None;
                    // A GPU build that ended up on the CPU says why.
                    self.last_err = info
                        .as_ref()
                        .filter(|i| !i.note.is_empty())
                        .map(|i| format!("The voice runs on the CPU — {}", i.note));
                    self.runs_on = info;
                }
                EngineMsg::Log(stage) => {
                    // Also mid-session: the worker reloading on the CPU after
                    // its GPU failed.
                    self.loading = true;
                    self.stage = Some(stage);
                }
                EngineMsg::Wav(path) => {
                    self.pending = self.pending.saturating_sub(1);
                    if self.discard > 0 {
                        // Synthesized for a request that was stopped.
                        self.discard -= 1;
                    } else {
                        let tx = self.player();
                        let _ = tx.send(PlayerCmd::Play(path));
                        self.next_piece();
                    }
                }
                EngineMsg::Warn(w) => self.last_err = Some(w),
                EngineMsg::Err(e) => {
                    self.pending = self.pending.saturating_sub(1);
                    self.loading = false;
                    if self.discard > 0 {
                        self.discard -= 1;
                    } else {
                        self.last_err = Some(e);
                        // One bad piece mustn't silence the rest of the reply.
                        self.next_piece();
                    }
                }
                EngineMsg::Exited(stderr_tail) => {
                    if matches!(self.stage.as_deref(), Some("load-gpu" | "warm-gpu"))
                        || self.runs_on.as_ref().is_some_and(|r| r.device == "gpu")
                    {
                        self.gpu_crashed = true;
                    }
                    self.engine = None;
                    self.engine_rx = None;
                    self.ready = false;
                    self.pending = 0;
                    self.discard = 0;
                    self.queue.clear();
                    self.stage = None;
                    self.runs_on = None;
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

    /// Speak `text` with the current voice, spawning the sidecar on first use
    /// (the model load makes the very first Listen slow — `loading` is set so
    /// the UI can say so). The reply goes in pieces (see `pieces`): the first
    /// now, each next one as soon as the previous one's audio is ready.
    pub fn speak(&mut self, text: &str, ctx: &egui::Context) {
        self.last_err = None;
        let mut pieces = pieces(text).into_iter();
        let Some(first) = pieces.next() else { return };
        if self.engine.is_none() {
            match self.spawn_engine(ctx) {
                Ok(()) => {
                    self.loading = true;
                    self.stage = None;
                }
                Err(e) => {
                    self.last_err = Some(e);
                    return;
                }
            }
        }
        self.queue = pieces.collect();
        self.send(&first, false);
    }

    /// Send the reply's next piece, if any.
    fn next_piece(&mut self) {
        if let Some(text) = self.queue.pop_front() {
            self.send(&text, true);
        }
    }

    /// One piece's request. `cont`: it continues the reply an earlier piece
    /// started (the worker keeps that piece's voice); `more`: pieces follow.
    fn send(&mut self, text: &str, cont: bool) {
        // A voice sample (cloning) wins over the description (voice design).
        let use_ref = !self.ref_audio.trim().is_empty()
            && std::path::Path::new(self.ref_audio.trim()).exists();
        let mut req = serde_json::json!({
            "text": text,
            // Also sent with a sample, so the worker can fall back to it if
            // the sample can't be used.
            "instruct": self.style,
            "cont": cont,
            "more": !self.queue.is_empty(),
        });
        if use_ref {
            req["ref_audio"] = self.ref_audio.trim().into();
            req["ref_text"] = self.ref_text.trim().into();
        }
        let Some(engine) = &mut self.engine else {
            self.queue.clear();
            return;
        };
        let line = format!("{req}\n");
        if engine.stdin.write_all(line.as_bytes()).and_then(|()| engine.stdin.flush()).is_err() {
            let _ = engine.child.kill();
            self.engine = None;
            self.ready = false;
            self.loading = false;
            self.queue.clear();
            self.last_err = Some("The voice worker stopped — try again.".to_string());
            return;
        }
        self.pending += 1;
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
            // UTF-8 pipes both ways: on Windows they'd otherwise use the ANSI
            // code page, so "don’t" reached the voice as "donâ€™t".
            .env("PYTHONUTF8", "1")
            .current_dir(base_dir());
        // AMD: if anything still reaches MIOpen (the worker turns it off for
        // PyTorch's convolutions), take its first kernel, not a search.
        if std::env::var_os("MIOPEN_FIND_MODE").is_none() {
            cmd.env("MIOPEN_FIND_MODE", "FAST");
        }
        if self.gpu_crashed {
            cmd.env("CLARITY_VOICE_CPU", "the GPU crashed the voice worker earlier");
        }
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
                for line in lossy_lines(stderr) {
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
            for line in lossy_lines(stdout) {
                eprintln!("[voice] {line}");
                let msg = if let Some(info) = line.strip_prefix("READY") {
                    EngineMsg::Ready(serde_json::from_str(info.trim()).ok())
                } else if let Some(stage) = line.strip_prefix("LOG ") {
                    EngineMsg::Log(stage.to_string())
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

    /// End the worker now and forget its state — setup is about to replace
    /// the PyTorch it has loaded (Windows can't overwrite DLLs in use).
    fn shut_down(&mut self) {
        self.stop();
        // Its exit isn't news.
        self.engine_rx = None;
        if let Some(mut e) = self.engine.take() {
            let _ = e.child.kill();
            let _ = e.child.wait();
        }
        self.ready = false;
        self.loading = false;
        self.pending = 0;
        self.discard = 0;
        self.stage = None;
        self.runs_on = None;
    }

    /// The audio playback thread (created on first use). Owns the output
    /// stream; plays clips back to back until Stop.
    fn player(&mut self) -> Sender<PlayerCmd> {
        if let Some(tx) = &self.player_tx {
            return tx.clone();
        }
        let (tx, rx) = mpsc::channel::<PlayerCmd>();
        spawn_player_thread(rx, self.speaking.clone());
        self.player_tx = Some(tx.clone());
        tx
    }

    /// The one-click "Set up voice" button — also "Set up again" and "Set up
    /// for this GPU" on an installed voice: everything runs on a background
    /// thread, streaming one-line progress into `setup_status`. Steps already
    /// done (Python, the model) are kept.
    pub fn start_setup(&mut self, ctx: &egui::Context) {
        if self.setting_up {
            return;
        }
        // A re-run replaces PyTorch under the worker, and the voice counts as
        // not installed until it's done (Listen uses the system voice
        // meanwhile; an interrupted run is simply set up again).
        self.shut_down();
        let _ = std::fs::remove_file(marker());
        self.installed = false;
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

/// A child's output as lines — lossy, so one stray byte (a path in the ANSI
/// code page, say) can't end the reading and leave the child blocked on a
/// full pipe.
fn lossy_lines(pipe: impl Read) -> impl Iterator<Item = String> {
    std::io::BufReader::new(pipe)
        .split(b'\n')
        .map_while(Result::ok)
        .map(|l| String::from_utf8_lossy(&l).trim_end_matches('\r').to_string())
}

// --- Pieces: how a reply is split for speaking --------------------------------

/// The first piece is short (it ends at the first sentence break after
/// FIRST_MIN chars), so the voice starts within seconds; the others join
/// whole sentences to PIECE_MIN..PIECE_MAX chars, room for natural phrasing.
const FIRST_MIN: usize = 40;
const FIRST_MAX: usize = 160;
const PIECE_MIN: usize = 90;
const PIECE_MAX: usize = 260;

fn count(s: &str) -> usize {
    s.chars().count()
}

/// Chinese/Japanese script (written without spaces between words or
/// sentences).
fn is_cjk(c: char) -> bool {
    matches!(c, '\u{3000}'..='\u{30FF}' | '\u{3400}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}' | '\u{FF00}'..='\u{FFEF}')
}

/// Join two runs of text with a space, unless both sides are CJK.
fn join(cur: &mut String, part: &str) {
    if !cur.is_empty()
        && !(cur.ends_with(is_cjk) && part.starts_with(is_cjk))
    {
        cur.push(' ');
    }
    cur.push_str(part);
}

/// The pieces a reply is spoken in (see FIRST_MIN). Pieces with nothing to
/// say (only punctuation) are dropped.
fn pieces(text: &str) -> Vec<String> {
    let limits = |done: usize| if done == 0 { (FIRST_MIN, FIRST_MAX) } else { (PIECE_MIN, PIECE_MAX) };
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for sentence in sentences(text) {
        for part in split_long(&sentence, limits(out.len()).1) {
            let (_, max) = limits(out.len());
            if !cur.is_empty() && count(&cur) + 1 + count(&part) > max {
                out.push(std::mem::take(&mut cur));
            }
            join(&mut cur, &part);
            if count(&cur) >= limits(out.len()).0 {
                out.push(std::mem::take(&mut cur));
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|p| p.chars().any(char::is_alphanumeric));
    out
}

/// A text's sentences: it splits after . ! ? … (and any closing quotes or
/// brackets) when a space and a non-lowercase start follow — so "e.g. this"
/// and "3.14" stay whole — after 。！？, and at line breaks. A line with no
/// end punctuation (a heading, a list item) gets a period, so it's read with
/// a pause instead of running into the next line.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let chars: Vec<char> = line.trim().chars().collect();
        let mut start = 0;
        let mut i = 0;
        while i < chars.len() {
            let mut end = None;
            if matches!(chars[i], '。' | '！' | '？') {
                end = Some(i + 1);
            } else if matches!(chars[i], '.' | '!' | '?' | '…') {
                let mut j = i + 1;
                while j < chars.len()
                    && matches!(chars[j], '.' | '!' | '?' | '…' | '"' | '\'' | '”' | '’' | ')' | ']')
                {
                    j += 1;
                }
                let next = chars[j..].iter().find(|c| !c.is_whitespace());
                if j < chars.len() && chars[j].is_whitespace() && next.is_some_and(|c| !c.is_lowercase()) {
                    end = Some(j);
                }
            }
            match end {
                Some(e) => {
                    let s: String = chars[start..e].iter().collect();
                    if !s.trim().is_empty() {
                        out.push(s.trim().to_string());
                    }
                    start = e;
                    i = e;
                }
                None => i += 1,
            }
        }
        let mut s: String = chars[start..].iter().collect::<String>().trim().to_string();
        if s.is_empty() {
            continue;
        }
        if s.ends_with(|c: char| c.is_ascii_alphanumeric() || c == ')') {
            s.push('.');
        }
        out.push(s);
    }
    out
}

/// `s` split after every char that `mark` accepts (each mark stays with its
/// part).
fn split_after(s: &str, mark: impl Fn(char) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        cur.push(c);
        if mark(c) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A sentence cut into parts of at most `max` chars: at clause breaks (, ;
/// : and their CJK forms) where it can, else between words, else anywhere
/// (a CJK run has no spaces).
fn split_long(s: &str, max: usize) -> Vec<String> {
    if count(s) <= max {
        return vec![s.to_string()];
    }
    let mut parts = Vec::new();
    for clause in split_after(s, |c| matches!(c, ',' | ';' | ':' | '，' | '、' | '；' | '：')) {
        if count(&clause) <= max {
            parts.push(clause);
            continue;
        }
        for word in split_after(&clause, char::is_whitespace) {
            if count(&word) <= max {
                parts.push(word);
            } else {
                let chars: Vec<char> = word.chars().collect();
                parts.extend(chars.chunks(max).map(|c| c.iter().collect::<String>()));
            }
        }
    }
    // Pack the parts (spacing kept as written) into runs of at most `max`.
    let mut out = Vec::new();
    let mut cur = String::new();
    for p in parts {
        if !cur.trim().is_empty() && count(&cur) + count(&p) > max {
            out.push(cur.trim().to_string());
            cur.clear();
        }
        cur.push_str(&p);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Playback via rodio (only present in `llm` builds — the same feature that
/// carries the AI chat this voice speaks for).
#[cfg(feature = "llm")]
fn spawn_player_thread(rx: Receiver<PlayerCmd>, speaking: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        // rodio 0.22: OutputStream/Sink became DeviceSinkBuilder/Player.
        let Ok(mut device) = rodio::DeviceSinkBuilder::open_default_sink() else {
            return;
        };
        device.log_on_drop(false);
        let mut player: Option<rodio::Player> = None;
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(PlayerCmd::Play(path)) => {
                    // Read in whole now: the worker reuses its few clip files.
                    if let Ok(bytes) = std::fs::read(&path)
                        && let Ok(dec) = rodio::Decoder::new(std::io::Cursor::new(bytes))
                    {
                        player
                            .get_or_insert_with(|| rodio::Player::connect_new(device.mixer()))
                            .append(dec);
                        speaking.store(true, Relaxed);
                    }
                }
                Ok(PlayerCmd::Stop) => {
                    if let Some(p) = player.take() {
                        p.stop();
                    }
                    speaking.store(false, Relaxed);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            if let Some(p) = &player
                && p.empty() {
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
    /// Latest capture peak (f32 bits), written by the audio callback — drives
    /// the live waveform in the popup.
    level: Arc<AtomicU32>,
    /// UI-side waveform history: one bar per frame while recording.
    wave: Vec<f32>,
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
        self.level.store(0, Relaxed);
        self.wave.clear();
        let (tx, rx) = mpsc::channel();
        self.rx = Some(rx);
        let stop = self.stop.clone();
        let level = self.level.clone();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            let res = record_loopback(&stop, &level);
            // Nobody listening: the popup was closed mid-recording (see
            // `close`), so this sample must not be left behind either.
            if let Err(mpsc::SendError(Ok(path))) = tx.send(res) {
                let _ = std::fs::remove_file(path);
            }
            repaint.request_repaint();
        });
        self.recording = true;
        self.started = Some(std::time::Instant::now());
    }

    /// Close the popup. A recording still running is stopped and thrown
    /// away: nothing polls the recorder while it's closed, so it used to run
    /// on to the 30 s cap and pop up as the new voice sample (replacing the
    /// old one and clearing its transcript) the next time the popup opened.
    pub fn close(&mut self) {
        self.open = false;
        if self.recording {
            self.stop.store(true, Relaxed);
            self.recording = false;
            self.rx = None;
        }
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
fn record_loopback(stop: &AtomicBool, level: &Arc<AtomicU32>) -> Result<PathBuf, String> {
    // rodio 0.22 no longer re-exports cpal — it's a direct dependency now.
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use cpal::SampleFormat;
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
    // cpal 0.17: sample_rate() returns a plain u32 (was a tuple struct).
    let sample_rate = config.sample_rate();
    let channels = config.channels() as usize;

    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buf2 = buf.clone();
    let level2 = level.clone();
    // Building an INPUT stream on an OUTPUT device = WASAPI loopback: we get
    // exactly what the speakers are playing.
    let stream = device
        .build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Chunk peak → the popup's live waveform.
                let peak = data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                level2.store(peak.to_bits(), Relaxed);
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
fn record_loopback(_stop: &AtomicBool, _level: &Arc<AtomicU32>) -> Result<PathBuf, String> {
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
                            egui::Stroke::new(1.0_f32, EDGE()),
                        )
                    };
                    // Opaque: in the Glass theme PANEL is see-through, and
                    // this window has nothing of the app behind it.
                    egui::Frame::new()
                        .fill(crate::theme::SOLID_PANEL())
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

                    // Vertically centred so the pill doesn't look top-heavy.
                    ui.horizontal_centered(|ui| {
                        ui.spacing_mut().item_spacing.x = 10.0;
                        // Apple-style record control: a ringed red circle (with a
                        // mic glyph) that morphs into a rounded "stop" square and
                        // gains a soft pulsing halo while recording.
                        let recording = voice.rec.recording;
                        let (rect, resp) =
                            ui.allocate_exact_size(egui::vec2(38.0, 38.0), egui::Sense::click());
                        let resp = resp
                            .on_hover_text(if recording { "Stop" } else { "Record what's playing" })
                            .on_hover_cursor(egui::CursorIcon::PointingHand);
                        let center = rect.center();
                        let red = egui::Color32::from_rgb(255, 69, 58);
                        let t = ui.ctx().animate_bool(egui::Id::new("rec_morph"), recording);
                        ui.painter().circle_stroke(
                            center,
                            17.0,
                            egui::Stroke::new(1.5_f32, if recording { red } else { EDGE() }),
                        );
                        if recording {
                            let now = ui.input(|i| i.time);
                            let pulse = ((now * 2.4).sin() as f32 * 0.5 + 0.5) * 0.35 + 0.10;
                            ui.painter().circle_stroke(
                                center,
                                20.0,
                                egui::Stroke::new(2.0_f32, red.gamma_multiply(pulse)),
                            );
                        }
                        // The red core: full circle when idle, stop square when live.
                        let size = egui::lerp(28.0..=15.0, t);
                        let corner = egui::lerp(14.0..=4.0, t);
                        ui.painter().rect_filled(
                            egui::Rect::from_center_size(center, egui::vec2(size, size)),
                            egui::CornerRadius::same(corner as u8),
                            red,
                        );
                        // Mic glyph inside the idle circle, fading out on record.
                        if t < 0.99 {
                            egui::Image::new(egui::include_image!("../icons/mic.svg"))
                                .tint(egui::Color32::WHITE.gamma_multiply(1.0 - t))
                                .paint_at(ui, egui::Rect::from_center_size(center, egui::vec2(15.0, 15.0)));
                        }
                        if resp.clicked() {
                            voice.rec.toggle_recording(ui.ctx());
                        }

                        // Middle column: timer/status on top; while recording a
                        // live Voice-Memos-style waveform (driven by the real
                        // capture level) replaces the hint line.
                        ui.vertical(|ui| {
                            // A nested vertical ui anchors to the row's TOP
                            // (egui places child uis before their height is
                            // known) — pad down so the two lines sit centred
                            // beside the 38px record control.
                            ui.add_space(12.0);
                            let s = voice.rec.seconds();
                            let (line1, c1, line2): (String, egui::Color32, &str) = if recording {
                                (
                                    format!("0:{s:02} / 0:{MAX_RECORD_SECS}"),
                                    TEXT(),
                                    "Recording what's playing…",
                                )
                            } else if let Some(e) = &voice.rec.err {
                                (
                                    e.clone(),
                                    egui::Color32::from_rgb(255, 105, 97),
                                    "Play the audio, then try again",
                                )
                            } else if voice.rec.just_saved {
                                (
                                    "Saved as the voice sample".into(),
                                    egui::Color32::from_rgb(48, 209, 88),
                                    "Record again to replace it",
                                )
                            } else {
                                (
                                    "Record what's playing".into(),
                                    TEXT(),
                                    "3–10s of one clean voice is best",
                                )
                            };
                            ui.label(egui::RichText::new(line1).color(c1).size(12.5).strong());
                            ui.add_space(1.0);
                            if recording {
                                // Perceptual sqrt so quiet audio still moves the
                                // bars; newest bar on the right, scrolling left.
                                let lvl = f32::from_bits(voice.rec.level.load(Relaxed)).sqrt();
                                voice.rec.wave.push(lvl);
                                if voice.rec.wave.len() > 2048 {
                                    voice.rec.wave.drain(..1024);
                                }
                                let (wr, _) = ui
                                    .allocate_exact_size(egui::vec2(118.0, 14.0), egui::Sense::hover());
                                let (bw, gap) = (2.0, 1.5);
                                let n = (wr.width() / (bw + gap)) as usize;
                                let len = voice.rec.wave.len();
                                let p = ui.painter();
                                for i in 0..n {
                                    let v = if len + i >= n { voice.rec.wave[len + i - n] } else { 0.0 };
                                    let h = (wr.height() * v.clamp(0.06, 1.0)).max(2.0);
                                    let x = wr.left() + i as f32 * (bw + gap);
                                    let bar = egui::Rect::from_center_size(
                                        egui::pos2(x + bw / 2.0, wr.center().y),
                                        egui::vec2(bw, h),
                                    );
                                    p.rect_filled(
                                        bar,
                                        egui::CornerRadius::same(1),
                                        red.gamma_multiply(0.35 + 0.65 * v.min(1.0)),
                                    );
                                }
                            } else {
                                ui.label(egui::RichText::new(line2).color(MUTED()).size(10.0));
                            }
                        });

                        // Bare ✕ icon pinned to the pill's top-right corner,
                        // inking up on hover.
                        let pill = ui.max_rect();
                        let crect = egui::Rect::from_center_size(
                            egui::pos2(pill.right() - 7.0, pill.top() + 7.0),
                            egui::vec2(20.0, 20.0),
                        );
                        let cresp = ui
                            .interact(crect, egui::Id::new("rec_close"), egui::Sense::click())
                            .on_hover_text("Close")
                            .on_hover_cursor(egui::CursorIcon::PointingHand);
                        let ink = if cresp.hovered() { TEXT() } else { MUTED() };
                        egui::Image::new(egui::include_image!("../icons/x.svg"))
                            .tint(icon_tint(ink))
                            .paint_at(
                                ui,
                                egui::Rect::from_center_size(crect.center(), egui::vec2(15.0, 15.0)),
                            );
                        if cresp.clicked() {
                            voice.rec.close();
                        }
                    });

                    if voice.rec.recording {
                        // Fast enough for the halo pulse to read as smooth.
                        ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
                    }
                        });
                });

            if ctx.input(|i| i.viewport().close_requested()) {
                voice.rec.close();
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

    // 2 — PyTorch for this machine's graphics card.
    send("[2/5] Looking at the graphics card…".to_string());
    let adapters = detect_adapters();
    let (build, adapter) = pick_build(&adapters, cfg!(windows), cfg!(target_os = "macos"));
    let mut info = SetupInfo { tried: (build != Build::Cpu).then_some(build), ..Default::default() };
    if build == Build::Cpu
        && cfg!(windows)
        && let Some(amd) = adapters.iter().find(|n| n.to_ascii_lowercase().contains("radeon"))
    {
        info.note = format!("AMD's PyTorch for Windows doesn't support the {amd}");
    }
    let Some(torch) = install_torch(&send, &base, &py, build, adapter.as_deref(), &mut info) else {
        send("PyTorch install failed".to_string());
        return false;
    };
    info.torch = Some(torch);

    // 3 — OmniVoice + the wav writer its examples use. (PyTorch is already
    // there, so pip keeps the build step 2 chose.)
    send("[3/5] Installing OmniVoice…".to_string());
    if !pip_retry(&send, &base, &py, &["-m", "pip", "install", "omnivoice", "soundfile"]) {
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

    // 5 — the model weights, so the first Listen doesn't surprise-download
    // (a re-run finds them in place and doesn't fetch them again).
    send("[5/5] Downloading the OmniVoice model…".to_string());
    let fetch = format!(
        "from huggingface_hub import snapshot_download\n\
         try:\n    snapshot_download('{MODEL_REPO}', local_files_only=True)\n\
         except Exception:\n    snapshot_download('{MODEL_REPO}')\n"
    );
    if !run_logged(&send, &base, &py, &["-c", &fetch]) {
        send("Model download failed".to_string());
        return false;
    }

    let _ = std::fs::write(marker(), info.render());
    true
}

/// Step 2: PyTorch for `build`, falling back to the CPU build when the GPU
/// one won't install (or, AMD's, won't run). Records in `info` the GPU the
/// check passed on, or why the GPU isn't used. None when not even the CPU
/// build installs.
fn install_torch(
    send: &dyn Fn(String),
    base: &std::path::Path,
    py: &str,
    build: Build,
    adapter: Option<&str>,
    info: &mut SetupInfo,
) -> Option<Build> {
    let gpu = adapter.unwrap_or("GPU");
    match build {
        Build::Cuda => {
            // Retries matter especially here: a dropped connection would
            // otherwise read as "GPU install failed" and downgrade to CPU torch.
            send("[2/5] Installing PyTorch (GPU)… this is the big one".to_string());
            let args = [
                "-m", "pip", "install",
                "torch==2.8.0+cu128", "torchaudio==2.8.0+cu128",
                "--extra-index-url", TORCH_INDEX,
            ];
            if pip_retry(send, base, py, &args) {
                send("[2/5] Checking the GPU…".to_string());
                match gpu_check(send, base, py) {
                    Ok(name) => info.gpu = name,
                    // Kept anyway: it runs on the CPU too, and on the GPU once
                    // the driver can drive it.
                    Err(e) => {
                        info.note = format!("PyTorch can't use the {gpu} ({e}); updating its driver usually fixes it")
                    }
                }
                return Some(Build::Cuda);
            }
            info.note = "the GPU build of PyTorch didn't install".to_string();
            send("[2/5] GPU PyTorch failed — installing CPU PyTorch instead".to_string());
        }
        Build::Rocm => {
            send(format!("[2/5] Installing AMD ROCm for the {gpu}… this is the big one"));
            // --no-cache-dir as AMD's guide has it: gigabytes of wheels that
            // would otherwise stay behind in pip's cache.
            let pip = ["-m", "pip", "install", "--no-cache-dir"];
            let sdk: Vec<&str> = pip.into_iter().chain(ROCM_SDK).collect();
            let torch: Vec<&str> = pip.into_iter().chain(ROCM_TORCH).collect();
            let installed = pip_try(send, base, py, &sdk)
                .map_err(|e| ("AMD's ROCm runtime didn't install", e))
                .and_then(|()| {
                    send("[2/5] Installing PyTorch for ROCm…".to_string());
                    pip_try(send, base, py, &torch).map_err(|e| ("AMD's PyTorch didn't install", e))
                });
            let note = match installed {
                Err((what, e)) if e.contains("WinError 206") || e.contains("too long") => format!(
                    "{what}: Windows' 260-character path limit got in the way. Turn on long \
                     paths (LongPathsEnabled), then Set up again"
                ),
                Err((what, e)) => format!("{what} ({e}). Check the connection, then Set up again"),
                Ok(()) => {
                    send("[2/5] Checking the GPU…".to_string());
                    match gpu_check(send, base, py) {
                        Ok(name) => {
                            info.gpu = name;
                            return Some(Build::Rocm);
                        }
                        Err(e) => format!(
                            "AMD's PyTorch can't use the {gpu} ({e}). Update AMD Software \
                             (Adrenalin 26.2.2 or newer), then Set up again"
                        ),
                    }
                }
            };
            send(format!("[2/5] {note} — installing CPU PyTorch for now"));
            info.note = note;
            // Several GB that can't run here.
            let rm: Vec<&str> = ["-m", "pip", "uninstall", "-y"].into_iter().chain(ROCM_PACKAGES).collect();
            let _ = run_logged(send, base, py, &rm);
        }
        Build::Cpu => send("[2/5] Installing PyTorch (CPU)…".to_string()),
    }
    install_cpu_torch(send, base, py).then_some(Build::Cpu)
}

/// PyTorch's CPU build, the same 2.8.0 as the CUDA one. A Mac takes PyPI's,
/// which is the CPU build there (with Apple's GPU support); elsewhere PyPI is
/// the fallback.
fn install_cpu_torch(send: &dyn Fn(String), base: &std::path::Path, py: &str) -> bool {
    let cpu_index = [
        "-m", "pip", "install",
        "torch==2.8.0+cpu", "torchaudio==2.8.0+cpu",
        "--extra-index-url", CPU_INDEX,
    ];
    (!cfg!(target_os = "macos") && pip_retry(send, base, py, &cpu_index))
        || pip_retry(send, base, py, &["-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0"])
}

/// Run GPU_CHECK with the voice's Python: Ok(the GPU's name) or Err(the
/// last thing it said).
fn gpu_check(send: &dyn Fn(String), base: &std::path::Path, py: &str) -> Result<String, String> {
    let (ok, lines) = run_lines(send, base, py, &["-c", GPU_CHECK]);
    match lines.iter().rev().find_map(|l| l.strip_prefix("GPU OK: ")) {
        Some(name) if ok => Ok(name.trim().to_string()),
        // GPU_CHECK's watchdog fired: faulthandler's "Timeout (0:05:00)!".
        _ if lines.iter().any(|l| l.starts_with("Timeout (")) => {
            Err("the GPU didn't answer for 5 minutes".to_string())
        }
        _ => {
            let last = lines.last().map(String::as_str).unwrap_or("no output");
            Err(last.chars().take(160).collect())
        }
    }
}

/// Run a pip command with retries — PyPI/CDN connections drop mid-download
/// often enough that one attempt can't be trusted for multi-GB wheels. pip
/// caches completed downloads, so a retry only re-fetches what was cut off.
fn pip_retry(send: &dyn Fn(String), dir: &std::path::Path, py: &str, args: &[&str]) -> bool {
    pip_try(send, dir, py, args).is_ok()
}

/// `pip_retry`, with pip's last error line when every attempt failed.
fn pip_try(send: &dyn Fn(String), dir: &std::path::Path, py: &str, args: &[&str]) -> Result<(), String> {
    const ATTEMPTS: u32 = 3;
    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        let (ok, lines) = run_lines(send, dir, py, args);
        if ok {
            return Ok(());
        }
        // pip ends a failure with "ERROR: …" (a WARNING about its own version
        // may follow it).
        last = lines
            .iter()
            .rev()
            .find(|l| l.starts_with("ERROR") || l.contains("Error"))
            .or(lines.last())
            .map(|l| l.chars().take(200).collect())
            .unwrap_or_default();
        if attempt < ATTEMPTS {
            send(format!(
                "Install hiccup (usually a dropped connection) — retrying, attempt {}/{ATTEMPTS}…",
                attempt + 1
            ));
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
    }
    Err(last)
}

/// Run a command from `dir`, forwarding its merged output line-by-line to the
/// status label. True on exit code 0.
fn run_logged(send: &dyn Fn(String), dir: &std::path::Path, program: &str, args: &[&str]) -> bool {
    run_lines(send, dir, program, args).0
}

/// `run_logged`, also handing back the output's last lines.
fn run_lines(send: &dyn Fn(String), dir: &std::path::Path, program: &str, args: &[&str]) -> (bool, Vec<String>) {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .current_dir(dir)
        // Python and pip print UTF-8 (paths under a non-ASCII user name).
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
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
            return (false, vec![e.to_string()]);
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
            for line in lossy_lines(pipe) {
                let _ = ltx.send(line);
            }
        }));
    }
    drop(ltx);
    let mut lines = Vec::new();
    for line in lrx {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            send(trimmed.to_string());
            if lines.len() == 64 {
                lines.remove(0);
            }
            lines.push(trimmed.to_string());
        }
    }
    for r in readers {
        let _ = r.join();
    }
    (matches!(child.wait(), Ok(s) if s.success()), lines)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn sentences_split_at_real_ends() {
        assert_eq!(
            sentences("Hello there. How are you? I'm fine!"),
            s(&["Hello there.", "How are you?", "I'm fine!"])
        );
        // Abbreviations before lowercase, decimals, and quotes stay whole.
        assert_eq!(
            sentences("Use e.g. this one. Pi is 3.14 today."),
            s(&["Use e.g. this one.", "Pi is 3.14 today."])
        );
        assert_eq!(
            sentences("He said \"Stop!\" and left. Then he came back."),
            s(&["He said \"Stop!\" and left.", "Then he came back."])
        );
        assert_eq!(sentences("Wait... what? Yes."), s(&["Wait... what?", "Yes."]));
        assert_eq!(sentences("你好。今天天气很好！"), s(&["你好。", "今天天气很好！"]));
    }

    #[test]
    fn lines_without_end_punctuation_get_a_pause() {
        assert_eq!(
            sentences("Key points\nFirst item\nSecond item (maybe)\nDone:"),
            s(&["Key points.", "First item.", "Second item (maybe).", "Done:"])
        );
    }

    #[test]
    fn first_piece_is_short_and_nothing_is_lost() {
        let text = "Sure! Here's a quick summary of the article you shared. It covers three \
                    things: the history of the bridge, how it was built, and why it matters \
                    today. The bridge opened in 1937 after four years of work. Engineers had \
                    to deal with strong currents, fog and deep water. Today it carries over \
                    a hundred thousand vehicles a day, and it's one of the most photographed \
                    structures in the world.";
        let p = pieces(text);
        assert!(p.len() >= 3, "{p:?}");
        assert_eq!(p[0], "Sure! Here's a quick summary of the article you shared.");
        assert!(count(&p[0]) <= FIRST_MAX);
        for piece in &p[1..] {
            assert!(count(piece) <= PIECE_MAX, "{piece}");
        }
        // All but the last reach the minimum.
        for piece in &p[1..p.len() - 1] {
            assert!(count(piece) >= PIECE_MIN, "{piece}");
        }
        let words = |t: &str| t.split_whitespace().map(String::from).collect::<Vec<_>>();
        assert_eq!(words(&p.join(" ")), words(text));
    }

    #[test]
    fn short_and_empty_replies() {
        assert_eq!(pieces("Hi!"), s(&["Hi!"]));
        assert!(pieces("").is_empty());
        assert!(pieces("...\n—").is_empty());
    }

    #[test]
    fn long_sentences_are_cut_within_limits() {
        let clause = "and then the long road wound on through the hills, ";
        let long = clause.repeat(14) + "until it ended.";
        let p = pieces(&long);
        assert!(p.len() > 2, "{p:?}");
        assert!(count(&p[0]) <= FIRST_MAX);
        assert!(p.iter().all(|x| count(x) <= PIECE_MAX), "{p:?}");
        // Cut at the commas, not mid-word.
        assert!(p[..p.len() - 1].iter().all(|x| x.ends_with(',')), "{p:?}");

        // A CJK run with no punctuation at all: hard cuts, no spaces added.
        let cjk = "天".repeat(600);
        let p = pieces(&cjk);
        assert!(p.iter().all(|x| count(x) <= PIECE_MAX && !x.contains(' ')));
        assert_eq!(p.concat(), cjk);
    }

    #[test]
    fn setup_info_round_trips() {
        // Installs from before the record: just "ok" — they tried CUDA.
        let old = SetupInfo::parse("ok");
        assert_eq!(old.torch, None);
        assert_eq!(old.tried(), Build::Cuda);

        let rocm = SetupInfo {
            torch: Some(Build::Rocm),
            gpu: "AMD Radeon(TM) 8060S Graphics".into(),
            tried: Some(Build::Rocm),
            note: String::new(),
        };
        assert_eq!(SetupInfo::parse(&rocm.render()), rocm);

        let fell_back = SetupInfo {
            torch: Some(Build::Cpu),
            gpu: String::new(),
            tried: Some(Build::Rocm),
            note: "AMD's PyTorch can't use the GPU (HIP error:\ninvalid device function)".into(),
        };
        let back = SetupInfo::parse(&fell_back.render());
        assert_eq!(back.tried(), Build::Rocm);
        assert_eq!(back.note, "AMD's PyTorch can't use the GPU (HIP error: invalid device function)");

        let cpu = SetupInfo { torch: Some(Build::Cpu), ..Default::default() };
        assert_eq!(SetupInfo::parse(&cpu.render()).tried(), Build::Cpu);
    }

    #[test]
    fn build_follows_the_graphics_card() {
        let win = |names: &[&str]| pick_build(&s(names), true, false);
        assert_eq!(
            win(&["AMD Radeon(TM) 8060S Graphics"]),
            (Build::Rocm, Some("AMD Radeon(TM) 8060S Graphics".to_string()))
        );
        // NVIDIA wins over an integrated Radeon.
        assert_eq!(win(&["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 4090"]).0, Build::Cuda);
        // Radeons AMD's Windows PyTorch doesn't support, and other GPUs: CPU.
        assert_eq!(win(&["AMD Radeon RX 6800 XT"]).0, Build::Cpu);
        assert_eq!(win(&["Intel(R) Arc(TM) A770 Graphics"]).0, Build::Cpu);
        // Unknown: CUDA as before (it runs on the CPU too).
        assert_eq!(win(&[]).0, Build::Cuda);
        // AMD's wheels are Windows-only; a Mac takes the CPU build.
        assert_eq!(pick_build(&s(&["AMD GPU"]), false, false).0, Build::Cpu);
        assert_eq!(pick_build(&s(&["NVIDIA GPU"]), false, false).0, Build::Cuda);
        assert_eq!(pick_build(&[], false, true).0, Build::Cpu);
    }

    #[test]
    fn rocm_support_by_name() {
        for yes in [
            "AMD Radeon(TM) 8060S Graphics",
            "AMD Radeon 8050S",
            "AMD Radeon 890M Graphics",
            "AMD Radeon RX 7900 XTX",
            "AMD Radeon RX 9070 XT",
            "AMD Radeon AI PRO R9700",
            "AMD Radeon PRO W7900",
        ] {
            assert!(rocm_supported(yes), "{yes}");
        }
        for no in [
            "AMD Radeon RX 6800 XT",
            "AMD Radeon 780M Graphics",
            "AMD Radeon(TM) Graphics",
            "AMD Radeon RX 580",
            "NVIDIA GeForce RTX 4090",
        ] {
            assert!(!rocm_supported(no), "{no}");
        }
    }

    #[test]
    fn runs_on_reads_the_worker_then_the_record() {
        let mut v = VoiceState::default();
        v.setup = SetupInfo::parse("ok");
        assert_eq!(v.runs_on_line(), None);
        v.setup = SetupInfo {
            torch: Some(Build::Rocm),
            gpu: "AMD Radeon(TM) 8060S Graphics".into(),
            tried: Some(Build::Rocm),
            note: String::new(),
        };
        assert_eq!(
            v.runs_on_line().as_deref(),
            Some("Runs on the GPU: AMD Radeon(TM) 8060S Graphics (AMD ROCm)")
        );
        v.runs_on = Some(RunsOn {
            device: "cpu".into(),
            name: "CPU".into(),
            backend: "CPU".into(),
            note: "the GPU failed (HIP out of memory)".into(),
        });
        assert_eq!(v.runs_on_line().as_deref(), Some("Runs on the CPU — the GPU failed (HIP out of memory)"));
    }

    #[test]
    fn worker_ready_line_parses() {
        let info: RunsOn = serde_json::from_str(
            r#"{"device": "gpu", "name": "AMD Radeon(TM) 8060S Graphics", "backend": "ROCm 7.2", "note": ""}"#,
        )
        .unwrap();
        assert_eq!(info.device, "gpu");
        assert_eq!(info.backend, "ROCm 7.2");
    }

    /// A reply goes out one piece at a time: the next when the previous
    /// one's audio arrives; a failed piece doesn't end it; Stop drops the
    /// rest and discards what's still synthesizing.
    #[test]
    fn reply_pieces_flow_one_at_a_time() {
        // Any process that swallows stdin stands in for the worker.
        let Ok(mut child) = Command::new("python3")
            .args(["-c", "import sys\nfor _ in sys.stdin: pass"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
        else {
            return;
        };
        let stdin = child.stdin.take().unwrap();
        let mut v = VoiceState::default();
        v.engine = Some(Engine { child, stdin });
        let (tx, rx) = mpsc::channel();
        v.engine_rx = Some(rx);
        let ctx = egui::Context::default();

        let text = "The first sentence is long enough to go out alone. ".to_string()
            + &"Another sentence follows, with a few more words in it. ".repeat(6);
        let total = pieces(&text).len();
        assert!(total >= 3, "{total}");
        v.speak(&text, &ctx);
        assert_eq!((v.pending, v.queue.len()), (1, total - 1));
        assert!(v.busy() && v.waiting());

        tx.send(EngineMsg::Log("load".into())).unwrap();
        tx.send(EngineMsg::Ready(None)).unwrap();
        tx.send(EngineMsg::Wav("a.wav".into())).unwrap();
        v.poll();
        assert!(!v.loading);
        assert_eq!((v.pending, v.queue.len()), (1, total - 2));

        tx.send(EngineMsg::Err("boom".into())).unwrap();
        v.poll();
        assert_eq!(v.last_err.as_deref(), Some("boom"));
        assert_eq!((v.pending, v.queue.len()), (1, total - 3));

        v.stop();
        assert!(v.queue.is_empty());
        assert!(!v.busy() && !v.waiting());
        tx.send(EngineMsg::Wav("b.wav".into())).unwrap();
        v.poll();
        assert_eq!((v.pending, v.queue.len()), (0, 0));

        drop(tx);
        v.poll();
        assert!(v.engine.is_none());
    }

    /// A worker that dies on the GPU (a driver crash: no Python error to
    /// report) comes back on the CPU; one that dies on the CPU doesn't change
    /// anything.
    #[test]
    fn gpu_crash_moves_the_next_worker_to_the_cpu() {
        for (stage, crashed) in [("warm-gpu", true), ("load-gpu", true), ("load", false)] {
            let mut v = VoiceState::default();
            let (tx, rx) = mpsc::channel();
            v.engine_rx = Some(rx);
            tx.send(EngineMsg::Log(stage.into())).unwrap();
            v.poll();
            assert!(v.loading);
            drop(tx);
            v.poll();
            assert_eq!(v.gpu_crashed, crashed, "{stage}");
            assert!(!v.loading);
            assert!(v.last_err.as_deref().is_some_and(|e| e.starts_with("The voice worker stopped")));
        }
    }

    /// The embedded Python must at least parse (checked when a python3 is
    /// around; the setup's py_compile is the real gate).
    #[test]
    fn embedded_python_parses() {
        let Ok(out) = Command::new("python3").arg("--version").output() else { return };
        if !out.status.success() {
            return;
        }
        for (name, src) in [("voice_worker", WORKER_PY), ("gpu_check", GPU_CHECK)] {
            let path = std::env::temp_dir().join(format!("clarity_test_{name}_{}.py", std::process::id()));
            std::fs::write(&path, src).unwrap();
            let ok = Command::new("python3")
                .args(["-c", "import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())"])
                .arg(&path)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            let _ = std::fs::remove_file(&path);
            assert!(ok, "{name} doesn't parse");
        }
    }
}
