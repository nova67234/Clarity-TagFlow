//! Region detection overlay — draws labelled boxes over the viewer image for
//! faces, hands, and people.
//!
//! Three single-class YOLOv8 detectors (DeepGHS ONNX exports, trained on both
//! anime and photos) run through the same `ort` runtime the taggers use. The
//! models are fetched as one AI-Model-Manager catalog entry ("Region Detection",
//! folder `region-detect` — see `src/ai_models.rs`); the first "Detect Regions"
//! click auto-starts that download.
//!
//! State is a process-wide singleton so both hosts of the zoom viewer (the
//! centre panel and the gallery-detail popup) share the toggle, the per-image
//! result cache, and the background worker without threading state through
//! their call chains. Inference runs on a short-lived background thread — one
//! job at a time; results land in a path-keyed cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, OnceLock};

use eframe::egui;
use egui::Color32;
use image::{imageops, Rgb, RgbImage};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// The AI-Model-Manager catalog folder holding the detector files.
const FOLDER: &str = "region-detect";

/// One detector model: its file in [`FOLDER`], the (square) letterbox input
/// size, a confidence floor, and the mapping from the model's class index to a
/// box label + colour — `None` classes are ignored, which is how the feet
/// detector (NudeNet, 18 classes) is filtered down to just its two feet classes.
struct DetectorDef {
    file: &'static str,
    input: u32,
    thr: f32,
    class: fn(usize) -> Option<(&'static str, Color32)>,
}

const FACE_COLOR: Color32 = Color32::from_rgb(64, 140, 255); // blue
const HAND_COLOR: Color32 = Color32::from_rgb(235, 150, 45); // orange
const PERSON_COLOR: Color32 = Color32::from_rgb(46, 160, 67); // green
const FEET_COLOR: Color32 = Color32::from_rgb(180, 132, 255); // violet

const DETECTORS: [DetectorDef; 4] = [
    DetectorDef {
        file: "face.onnx",
        input: 640,
        thr: 0.35,
        class: |_| Some(("Face", FACE_COLOR)),
    },
    DetectorDef {
        file: "hand.onnx",
        input: 640,
        thr: 0.35,
        class: |_| Some(("Hand", HAND_COLOR)),
    },
    DetectorDef {
        file: "person.onnx",
        input: 640,
        thr: 0.30,
        class: |_| Some(("Person", PERSON_COLOR)),
    },
    // NudeNet 320n — only its FEET_EXPOSED (7) / FEET_COVERED (9) classes are
    // kept; everything else the model can find is discarded.
    DetectorDef {
        file: "feet.onnx",
        input: 320,
        thr: 0.30,
        class: |c| match c {
            7 | 9 => Some(("Feet", FEET_COLOR)),
            _ => None,
        },
    },
];

/// One detected region, in normalized (0..1) image coordinates.
pub struct Detection {
    pub label: String,
    pub color: Color32,
    /// `[x0, y0, x1, y1]` as fractions of the image width/height.
    pub rect: [f32; 4],
    pub conf: f32,
}

/// Whether the region overlay is currently shown (flipped by the viewer's
/// right-click "Detect Regions" item). Session-only.
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn toggle() {
    ENABLED.fetch_xor(true, Ordering::Relaxed);
}

/// Whether the age overlay is currently shown ("Detect Age"). Independent of
/// the region overlay — either or both can be on.
static AGE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn age_enabled() -> bool {
    AGE_ENABLED.load(Ordering::Relaxed)
}

pub fn toggle_age() {
    AGE_ENABLED.fetch_xor(true, Ordering::Relaxed);
}

/// Loaded ONNX sessions, kept across jobs so repeat detections are instant.
static SESSIONS: Mutex<Option<Vec<Session>>> = Mutex::new(None);

/// The age-estimation session (InsightFace genderage), loaded on first use.
static AGE_SESSION: Mutex<Option<Session>> = Mutex::new(None);

/// Age labels are drawn on the detected face box in teal.
const AGE_COLOR: Color32 = Color32::from_rgb(90, 200, 245);

/// Finished detections per image path.
type DetCache = HashMap<PathBuf, Arc<Vec<Detection>>>;
/// An in-flight detection job: the image path + its result channel.
type DetJob = Option<(PathBuf, Receiver<Result<Vec<Detection>, String>>)>;
/// A blocking detector: image path in, detections out.
type DetectFn = fn(&Path) -> Result<Vec<Detection>, String>;

#[derive(Default)]
struct Inner {
    /// Finished region results per image path.
    cache: DetCache,
    /// Finished age results per image path.
    age_cache: DetCache,
    /// The in-flight region job, one at a time.
    job: DetJob,
    /// The in-flight age job, one at a time (independent of the region job).
    age_job: DetJob,
    /// The in-flight model download, when the models aren't installed yet
    /// (one catalog entry covers both overlays' models).
    download: Option<crate::ai_models::DownloadHandle>,
    /// Last error (model load / inference / download), shown in the overlay.
    error: Option<String>,
}

fn inner() -> &'static Mutex<Inner> {
    static INNER: OnceLock<Mutex<Inner>> = OnceLock::new();
    INNER.get_or_init(|| Mutex::new(Inner::default()))
}

/// True when all the detector files are present in the model roots.
fn installed() -> bool {
    DETECTORS.iter().all(|d| crate::tagger::resolve(FOLDER, d.file).is_some())
}

/// True when the age path's models (face detector + genderage) are present.
fn installed_age() -> bool {
    crate::tagger::resolve(FOLDER, DETECTORS[0].file).is_some()
        && crate::tagger::resolve(FOLDER, "age.onnx").is_some()
}

/// Which overlay a [`drive`] call serves.
#[derive(Clone, Copy)]
enum Kind {
    Regions,
    Age,
}

/// Drive the region overlay for `path` — see [`drive`].
pub fn overlay(path: &Path, ctx: &egui::Context) -> (Option<Arc<Vec<Detection>>>, Option<String>) {
    drive(Kind::Regions, path, ctx)
}

/// Drive the age overlay for `path` — see [`drive`].
pub fn age_overlay(path: &Path, ctx: &egui::Context) -> (Option<Arc<Vec<Detection>>>, Option<String>) {
    drive(Kind::Age, path, ctx)
}

/// Returns the cached detections for `path` when ready, plus a status line
/// ("Downloading… 42%", "Detecting…", an error) while not. Starts the model
/// download / inference job as needed. Call every frame the overlay is
/// enabled; cheap when the result is cached.
fn drive(kind: Kind, path: &Path, ctx: &egui::Context) -> (Option<Arc<Vec<Detection>>>, Option<String>) {
    let mut st = match inner().lock() {
        Ok(g) => g,
        Err(_) => return (None, Some("Detection state poisoned".into())),
    };
    let Inner { cache, age_cache, job, age_job, download, error } = &mut *st;

    // Absorb finished inference jobs (both kinds, so either overlay keeps the
    // other's results fresh too).
    let absorb = |job: &mut DetJob, cache: &mut DetCache, error: &mut Option<String>| {
        if let Some((job_path, rx)) = job
            && let Ok(result) = rx.try_recv() {
                let job_path = job_path.clone();
                *job = None;
                match result {
                    Ok(dets) => {
                        *error = None;
                        cache.insert(job_path, Arc::new(dets));
                    }
                    Err(e) => *error = Some(e),
                }
            }
    };
    absorb(job, cache, error);
    absorb(age_job, age_cache, error);

    // Absorb a finished model download.
    if let Some(dl) = &download
        && dl.done() {
            if !dl.ok() {
                *error = Some(dl.error().unwrap_or_else(|| "Model download failed".into()));
            }
            *download = None;
        }

    // Route to this overlay's cache / job / models / worker.
    let (cache, job, ready, work): (&mut DetCache, &mut DetJob, bool, DetectFn) = match kind {
        Kind::Regions => (cache, job, installed(), detect_file),
        Kind::Age => (age_cache, age_job, installed_age(), detect_age_file),
    };

    if let Some(dets) = cache.get(path) {
        return (Some(Arc::clone(dets)), None);
    }
    if let Some(e) = &error {
        return (None, Some(format!("Detection failed: {e}")));
    }

    // Models missing → kick off (or report) the shared catalog download.
    if !ready {
        if download.is_none() {
            *download = crate::ai_models::start_model_download(FOLDER);
            if download.is_none() {
                *error = Some("No catalog entry for the detection models".into());
            }
        }
        let pct = download.as_ref().map(|d| d.pct()).unwrap_or(0);
        ctx.request_repaint_after(std::time::Duration::from_millis(150));
        return (None, Some(format!("Downloading detection models… {pct}%")));
    }

    // Start an inference job for this image once this overlay's worker is free.
    // (A job for a different image just finishes first; we retry next frame.)
    if job.is_none() {
        let (tx, rx) = mpsc::channel();
        *job = Some((path.to_path_buf(), rx));
        let p = path.to_path_buf();
        let c = ctx.clone();
        std::thread::spawn(move || {
            let r = work(&p);
            let _ = tx.send(r);
            c.request_repaint();
        });
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(120));
    let verb = match kind {
        Kind::Regions => "Detecting…",
        Kind::Age => "Estimating age…",
    };
    (None, Some(verb.into()))
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// Load the detector sessions into `guard` if not already loaded.
fn ensure_sessions(guard: &mut Option<Vec<Session>>) -> Result<(), String> {
    if guard.is_some() {
        return Ok(());
    }
    let mut loaded = Vec::new();
    for def in &DETECTORS {
        let model = crate::tagger::resolve(FOLDER, def.file)
            .ok_or_else(|| format!("{} missing — re-download Region Detection", def.file))?;
        let session = Session::builder()
            .map_err(|e| format!("ORT init: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ORT options: {e}"))?
            .commit_from_file(&model)
            .map_err(|e| format!("Load {}: {e}", def.file))?;
        loaded.push(session);
    }
    *guard = Some(loaded);
    Ok(())
}

/// Run one detector on `img`, returning `(class, normalized rect, conf)` boxes.
fn run_detector(
    session: &mut Session,
    def: &DetectorDef,
    img: &RgbImage,
) -> Result<Vec<(usize, [f32; 4], f32)>, String> {
    let (tensor_data, scale, dx, dy) = letterbox(img, def.input);
    let boxes = run_yolo(session, &tensor_data, def.input, def.thr)?;
    let w = img.width() as f32;
    let h = img.height() as f32;
    let mut out = Vec::new();
    for (class, bx, conf) in boxes {
        // Map the input-space box back through the letterbox to normalized
        // image coordinates.
        let un = |v: f32, off: f32| (v - off) / scale;
        let rect = [
            (un(bx[0], dx) / w).clamp(0.0, 1.0),
            (un(bx[1], dy) / h).clamp(0.0, 1.0),
            (un(bx[2], dx) / w).clamp(0.0, 1.0),
            (un(bx[3], dy) / h).clamp(0.0, 1.0),
        ];
        if rect[2] > rect[0] && rect[3] > rect[1] {
            out.push((class, rect, conf));
        }
    }
    Ok(out)
}

/// Run all the region detectors on `path`, merging their boxes.
fn detect_file(path: &Path) -> Result<Vec<Detection>, String> {
    let img = crate::tagger::load_rgb_on_white(path)?;

    let mut guard = SESSIONS.lock().map_err(|_| "Session lock poisoned".to_string())?;
    ensure_sessions(&mut guard)?;
    let sessions = guard.as_mut().expect("just loaded");

    let mut all = Vec::new();
    for (i, session) in sessions.iter_mut().enumerate() {
        let def = &DETECTORS[i];
        for (class, rect, conf) in run_detector(session, def, &img)? {
            // Skip classes this detector doesn't surface (e.g. NudeNet's
            // non-feet classes).
            let Some((label, color)) = (def.class)(class) else { continue };
            all.push(Detection { label: label.to_string(), color, rect, conf });
        }
    }
    Ok(all)
}

/// Estimate an age per detected face: face boxes from the face detector, then
/// InsightFace's `genderage` on each face crop. The label lands on the face box
/// (e.g. "Age ~24").
fn detect_age_file(path: &Path) -> Result<Vec<Detection>, String> {
    let img = crate::tagger::load_rgb_on_white(path)?;

    // Face boxes first (DETECTORS[0] is the face model).
    let faces = {
        let mut guard = SESSIONS.lock().map_err(|_| "Session lock poisoned".to_string())?;
        ensure_sessions(&mut guard)?;
        let sessions = guard.as_mut().expect("just loaded");
        run_detector(&mut sessions[0], &DETECTORS[0], &img)?
    };
    if faces.is_empty() {
        return Ok(Vec::new());
    }

    let mut guard = AGE_SESSION.lock().map_err(|_| "Age session lock poisoned".to_string())?;
    if guard.is_none() {
        let model = crate::tagger::resolve(FOLDER, "age.onnx")
            .ok_or_else(|| "age.onnx missing — re-download Region Detection".to_string())?;
        let session = Session::builder()
            .map_err(|e| format!("ORT init: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ORT options: {e}"))?
            .commit_from_file(&model)
            .map_err(|e| format!("Load age.onnx: {e}"))?;
        *guard = Some(session);
    }
    let session = guard.as_mut().expect("just loaded");
    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or_else(|| "Age model has no inputs".to_string())?;
    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or_else(|| "Age model has no outputs".to_string())?;

    let mut out = Vec::new();
    for (_, rect, conf) in faces {
        let data = crop_face_96(&img, rect);
        let tensor = Tensor::from_array(([1i64, 3, 96, 96], data))
            .map_err(|e| format!("Tensor: {e}"))?;
        let outputs = session
            .run(ort::inputs![input_name.as_str() => tensor])
            .map_err(|e| format!("Age inference: {e}"))?;
        let (_, raw) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Age output: {e}"))?;
        // genderage output: [female_score, male_score, age / 100].
        let age = (raw.get(2).copied().unwrap_or(0.0) * 100.0).round() as i32;
        if age > 0 {
            out.push(Detection {
                label: format!("Age ~{age}"),
                color: AGE_COLOR,
                rect,
                conf,
            });
        }
    }
    Ok(out)
}

/// Crop a square around the face box (1.5× its long side, InsightFace's margin
/// convention), clamped inside the image, resized to the genderage model's
/// 96×96 input. Values are raw RGB 0..255 floats in NCHW (no normalisation —
/// matching InsightFace's `blobFromImage(scale=1.0, mean=0, swapRB)`).
fn crop_face_96(img: &RgbImage, rect: [f32; 4]) -> Vec<f32> {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (bx, by) = ((rect[0] + rect[2]) * 0.5 * w, (rect[1] + rect[3]) * 0.5 * h);
    let side = ((rect[2] - rect[0]) * w).max((rect[3] - rect[1]) * h) * 1.5;
    let side = side.clamp(8.0, w.min(h));
    // Clamp the square inside the image (shift rather than shrink).
    let x0 = (bx - side * 0.5).clamp(0.0, w - side);
    let y0 = (by - side * 0.5).clamp(0.0, h - side);
    let crop = imageops::crop_imm(img, x0 as u32, y0 as u32, side as u32, side as u32).to_image();
    let resized = imageops::resize(&crop, 96, 96, imageops::FilterType::Triangle);

    let n = 96 * 96;
    let mut data = vec![0f32; 3 * n];
    for (i, px) in resized.pixels().enumerate() {
        data[i] = px[0] as f32;
        data[n + i] = px[1] as f32;
        data[2 * n + i] = px[2] as f32;
    }
    data
}

/// Aspect-preserving resize onto a grey `input`×`input` canvas (YOLO's
/// letterbox). Returns the NCHW `f32` tensor data plus the scale and paste
/// offsets needed to map boxes back to source coordinates.
fn letterbox(img: &RgbImage, input: u32) -> (Vec<f32>, f32, f32, f32) {
    let (w, h) = (img.width(), img.height());
    let scale = (input as f32 / w as f32).min(input as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let resized = imageops::resize(img, nw, nh, imageops::FilterType::Triangle);
    let dx = ((input - nw) / 2) as i64;
    let dy = ((input - nh) / 2) as i64;
    let mut canvas = RgbImage::from_pixel(input, input, Rgb([114, 114, 114]));
    imageops::replace(&mut canvas, &resized, dx, dy);

    let n = (input * input) as usize;
    let mut data = vec![0f32; 3 * n];
    for (i, px) in canvas.pixels().enumerate() {
        data[i] = px[0] as f32 / 255.0;
        data[n + i] = px[1] as f32 / 255.0;
        data[2 * n + i] = px[2] as f32 / 255.0;
    }
    (data, scale, dx as f32, dy as f32)
}

/// Run one YOLOv8-style detector on the letterboxed tensor, returning boxes in
/// input coordinates as `(class_index, [x0, y0, x1, y1], confidence)` after
/// per-class NMS.
fn run_yolo(
    session: &mut Session,
    data: &[f32],
    input: u32,
    thr: f32,
) -> Result<Vec<(usize, [f32; 4], f32)>, String> {
    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or_else(|| "Model has no inputs".to_string())?;
    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .ok_or_else(|| "Model has no outputs".to_string())?;

    let shape = [1i64, 3, input as i64, input as i64];
    let tensor =
        Tensor::from_array((shape, data.to_vec())).map_err(|e| format!("Tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_name.as_str() => tensor])
        .map_err(|e| format!("Inference: {e}"))?;
    let (out_shape, raw) = outputs[output_name.as_str()]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Output: {e}"))?;

    let dims: Vec<i64> = out_shape.iter().copied().collect();
    if dims.len() != 3 {
        return Err(format!("Unexpected YOLO output shape {dims:?}"));
    }
    // Ultralytics exports are [1, 4+nc, anchors]; handle a transposed
    // [1, anchors, 4+nc] export too (channels is the small dimension).
    let (c, n, chan_major) = if dims[1] <= dims[2] {
        (dims[1] as usize, dims[2] as usize, true)
    } else {
        (dims[2] as usize, dims[1] as usize, false)
    };
    if c < 5 {
        return Err(format!("Unexpected YOLO output shape {dims:?}"));
    }
    let at = |ch: usize, j: usize| -> f32 {
        if chan_major {
            raw[ch * n + j]
        } else {
            raw[j * c + ch]
        }
    };

    // Decode: best class per anchor, then centre/size → corners.
    let mut boxes: Vec<(usize, [f32; 4], f32)> = Vec::new();
    for j in 0..n {
        let mut conf = 0f32;
        let mut class = 0usize;
        for ch in 4..c {
            let s = at(ch, j);
            if s > conf {
                conf = s;
                class = ch - 4;
            }
        }
        if conf < thr {
            continue;
        }
        let (cx, cy, bw, bh) = (at(0, j), at(1, j), at(2, j), at(3, j));
        boxes.push((class, [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], conf));
    }

    // Greedy NMS per class, highest confidence first.
    boxes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<(usize, [f32; 4], f32)> = Vec::new();
    for (class, b, conf) in boxes {
        if kept.iter().all(|(kc, k, _)| *kc != class || iou(&b, k) < 0.45) {
            kept.push((class, b, conf));
        }
    }
    Ok(kept)
}

/// Intersection-over-union of two `[x0, y0, x1, y1]` boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end pipeline check on a real image with an obvious face + person.
    /// Skips (passes) when the models aren't installed, so CI stays green.
    #[test]
    fn detects_face_and_person_in_sample() {
        if !installed() {
            eprintln!("region-detect models not installed; skipping");
            return;
        }
        let sample = Path::new("tests/bug/2.png");
        if !sample.exists() {
            eprintln!("sample image missing; skipping");
            return;
        }
        let dets = detect_file(sample).expect("detection should run");
        let labels: Vec<&str> = dets.iter().map(|d| d.label.as_str()).collect();
        eprintln!("detections: {:?}", dets.iter().map(|d| (&d.label, d.conf)).collect::<Vec<_>>());
        assert!(labels.contains(&"Face"), "expected a face, got {labels:?}");
        assert!(labels.contains(&"Person"), "expected a person, got {labels:?}");
        for d in &dets {
            assert!(d.rect[0] < d.rect[2] && d.rect[1] < d.rect[3], "box must be well-formed");
            assert!(d.conf > 0.0 && d.conf <= 1.0);
        }
    }

    /// End-to-end age pipeline check: face crop → genderage → an "Age ~N" label
    /// on the face box. Skips (passes) when the models aren't installed.
    #[test]
    fn estimates_age_on_sample_face() {
        if !installed_age() {
            eprintln!("age models not installed; skipping");
            return;
        }
        let sample = Path::new("tests/bug/2.png");
        if !sample.exists() {
            eprintln!("sample image missing; skipping");
            return;
        }
        let dets = detect_age_file(sample).expect("age estimation should run");
        eprintln!("ages: {:?}", dets.iter().map(|d| (&d.label, d.conf)).collect::<Vec<_>>());
        assert!(!dets.is_empty(), "expected an age for the sample's face");
        for d in &dets {
            assert!(d.label.starts_with("Age ~"), "unexpected label {}", d.label);
            let n: i32 = d.label.trim_start_matches("Age ~").parse().expect("numeric age");
            assert!((1..120).contains(&n), "implausible age {n}");
        }
    }
}
