//! Voice dictation for the AI chat's mic button — records the default
//! microphone and transcribes it locally with Whisper (whisper.cpp via the
//! whisper-rs bindings), fully in-process like the Gemma chat model.
//!
//! Flow: first click downloads the model if it's missing (same catalog +
//! progress plumbing as every other model, see src/ai_models.rs). Once
//! installed, a click starts capturing the mic on a worker thread; the next
//! click stops it, the worker runs Whisper over the clip and the transcript
//! lands in the chat draft via [`DictationState::poll`].

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use eframe::egui;

/// Model folder + file under the shared models root (`tools/`), also
/// registered in the Get Models catalog so it can be fetched from there.
pub const FOLDER: &str = "whisper-base";
pub const MODEL_FILE: &str = "ggml-base.bin";

/// Cap a dictation at two minutes — Whisper transcribes the whole clip in
/// one go, so an unbounded recording would mean unbounded memory and wait.
const MAX_RECORD_SECS: u64 = 120;

/// Whisper's fixed input format: 16 kHz mono f32 PCM.
const WHISPER_RATE: u32 = 16_000;

pub fn installed() -> bool {
    crate::tagger::resolve(FOLDER, MODEL_FILE).is_some()
}

/// What the mic button should show this frame.
pub enum Phase {
    Idle,
    /// First-use model download, with overall percent.
    Downloading(u32),
    Recording,
    Transcribing,
}

#[derive(Default)]
pub struct DictationState {
    /// Set to stop the capture loop; the worker then transcribes and exits.
    stop: Arc<AtomicBool>,
    /// Flipped by the worker once capture ends and inference begins.
    transcribing: Arc<AtomicBool>,
    /// Yields the finished transcript (or an error). `Some` = job in flight.
    rx: Option<Receiver<Result<String, String>>>,
    /// First-use model download, polled each frame.
    download: Option<crate::ai_models::DownloadHandle>,
    /// Last failure, surfaced in the mic tooltip until the next attempt.
    pub err: Option<String>,
    started: Option<std::time::Instant>,
}

impl DictationState {
    pub fn phase(&self) -> Phase {
        if let Some(d) = &self.download {
            return Phase::Downloading(d.pct());
        }
        if self.rx.is_some() {
            if self.transcribing.load(Relaxed) {
                return Phase::Transcribing;
            }
            return Phase::Recording;
        }
        Phase::Idle
    }

    pub fn seconds(&self) -> u64 {
        self.started.map(|s| s.elapsed().as_secs()).unwrap_or(0)
    }

    /// Mic click: start recording, stop recording, or kick the model
    /// download — whichever the current phase calls for.
    pub fn toggle(&mut self, ctx: &egui::Context) {
        match self.phase() {
            Phase::Recording => self.stop.store(true, Relaxed),
            // Nothing sensible to do mid-download/-transcription.
            Phase::Downloading(_) | Phase::Transcribing => {}
            Phase::Idle => {
                self.err = None;
                if !installed() {
                    self.download = crate::ai_models::start_model_download(FOLDER);
                    return;
                }
                let Some(model) = crate::tagger::resolve(FOLDER, MODEL_FILE) else {
                    self.err = Some("Speech model not found".into());
                    return;
                };
                self.stop = Arc::new(AtomicBool::new(false));
                self.transcribing = Arc::new(AtomicBool::new(false));
                let (tx, rx) = mpsc::channel();
                self.rx = Some(rx);
                self.started = Some(std::time::Instant::now());
                let stop = self.stop.clone();
                let transcribing = self.transcribing.clone();
                let repaint = ctx.clone();
                std::thread::spawn(move || {
                    let res = record_and_transcribe(&stop, &transcribing, model);
                    let _ = tx.send(res);
                    repaint.request_repaint();
                });
            }
        }
    }

    /// Drive the state each frame; returns the finished transcript, if any.
    pub fn poll(&mut self, ctx: &egui::Context) -> Option<String> {
        if let Some(d) = &self.download
            && d.done()
        {
            if !d.ok() {
                self.err = Some(d.error().unwrap_or_else(|| "Download failed".into()));
            }
            self.download = None;
        }
        let mut out = None;
        if let Some(rx) = &self.rx {
            match rx.try_recv() {
                Ok(Ok(text)) => {
                    self.rx = None;
                    out = Some(text);
                }
                Ok(Err(e)) => {
                    self.rx = None;
                    self.err = Some(e);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => self.rx = None,
            }
        }
        // Keep the pulse/spinner/progress animating while anything runs.
        if self.rx.is_some() || self.download.is_some() {
            ctx.request_repaint();
        }
        out
    }
}

/// Worker: capture the mic until stopped (or the length cap), then run
/// Whisper over the clip.
#[cfg(feature = "llm")]
fn record_and_transcribe(
    stop: &AtomicBool,
    transcribing: &AtomicBool,
    model: std::path::PathBuf,
) -> Result<String, String> {
    let pcm = record_mic(stop)?;
    transcribing.store(true, Relaxed);
    transcribe(&model, &pcm)
}

#[cfg(not(feature = "llm"))]
fn record_and_transcribe(
    _stop: &AtomicBool,
    _transcribing: &AtomicBool,
    _model: std::path::PathBuf,
) -> Result<String, String> {
    Err("This build was compiled without the AI feature.".to_string())
}

/// Capture the default input device until stopped, returning 16 kHz mono f32
/// PCM ready for Whisper. Same cpal plumbing as the voice-sample recorder
/// (src/voice.rs), but on the microphone instead of the loopback.
#[cfg(feature = "llm")]
fn record_mic(stop: &AtomicBool) -> Result<Vec<f32>, String> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use rodio::cpal::{self, SampleFormat};
    use std::sync::Mutex;

    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No microphone found")?;
    let config = device
        .default_input_config()
        .map_err(|e| format!("Microphone config: {e}"))?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buf2 = buf.clone();
    let err_fn = |e| eprintln!("[dictate] stream error: {e}");
    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                buf2.lock().unwrap().extend_from_slice(data);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut b = buf2.lock().unwrap();
                b.extend(data.iter().map(|&v| v as f32 / i16::MAX as f32));
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let mut b = buf2.lock().unwrap();
                b.extend(data.iter().map(|&v| (v as f32 / u16::MAX as f32) * 2.0 - 1.0));
            },
            err_fn,
            None,
        ),
        f => return Err(format!("Unsupported sample format {f:?}")),
    }
    .map_err(|e| format!("Couldn't open the microphone: {e}"))?;
    stream.play().map_err(|e| format!("Couldn't start capture: {e}"))?;

    let started = std::time::Instant::now();
    while !stop.load(Relaxed) && started.elapsed().as_secs() < MAX_RECORD_SECS {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(stream);

    let samples = std::mem::take(&mut *buf.lock().unwrap());
    // Half a second of frames minimum, or there's nothing worth transcribing.
    if samples.len() < (sample_rate as usize * channels) / 2 {
        return Err("Didn't catch anything — is the microphone working?".to_string());
    }

    // Downmix to mono, then linear-resample to Whisper's 16 kHz.
    let mono: Vec<f32> = samples
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect();
    if sample_rate == WHISPER_RATE {
        return Ok(mono);
    }
    let ratio = sample_rate as f64 / WHISPER_RATE as f64;
    let out_len = (mono.len() as f64 / ratio) as usize;
    Ok((0..out_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let i0 = src as usize;
            let a = mono[i0.min(mono.len() - 1)];
            let b = *mono.get(i0 + 1).unwrap_or(&a);
            a + (b - a) * (src - i0 as f64) as f32
        })
        .collect())
}

/// Run Whisper over a 16 kHz mono clip. Loads the model per call — the base
/// model reads in well under a second, not worth caching 150 MB in RAM
/// between dictations.
#[cfg(feature = "llm")]
fn transcribe(model: &std::path::Path, pcm: &[f32]) -> Result<String, String> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    // whisper.cpp chats to stderr on every load; route it to the (unused)
    // log crate instead. Idempotent-enough — reinstalling is harmless.
    whisper_rs::install_logging_hooks();

    let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
        .map_err(|e| format!("Couldn't load the speech model: {e}"))?;
    let mut state = ctx.create_state().map_err(|e| format!("Whisper init: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("auto"));
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().min(8) as i32)
        .unwrap_or(4);
    params.set_n_threads(threads);

    state.full(params, pcm).map_err(|e| format!("Transcription failed: {e}"))?;

    let mut out = String::new();
    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else { continue };
        let Ok(text) = seg.to_str_lossy() else { continue };
        // Whisper narrates silence as bracketed stage directions —
        // "[BLANK_AUDIO]", "(music)" — drop those outright.
        let t = text.trim();
        if (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('(') && t.ends_with(')')) {
            continue;
        }
        if !out.is_empty() && !out.ends_with(char::is_whitespace) {
            out.push(' ');
        }
        out.push_str(t);
    }
    let out = out.trim().to_string();
    if out.is_empty() {
        Err("Didn't catch any speech — try again?".to_string())
    } else {
        Ok(out)
    }
}
