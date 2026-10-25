//! Monocular depth estimation for the Spatial Scene parallax viewer.
//!
//! Runs **Depth Anything V2** (ONNX) through the `ort` crate — the same runtime
//! the taggers use (see `src/tagger.rs`). It produces a normalized depth map
//! (`0.0` = far, `1.0` = near) that `src/zoom.rs` turns into a depth-displaced
//! parallax mesh.
//!
//! The model is fetched via the AI Model Manager (the "Depth Anything V2 Small"
//! catalog entry in `src/ai_models.rs`) into
//! `tools/depth-anything-v2-small-onnx/` (the newer onnx-community export, whose
//! weights live in an external `model.onnx_data` sidecar next to `model.onnx`).

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use eframe::egui;
use image::imageops::FilterType;
use image::RgbImage;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// Depth Anything V2 expects a square input that is a multiple of its 14-px patch
/// size; 518 = 14 × 37 is the published default.
const DSIZE: u32 = 518;

/// A normalized depth map. `data` is row-major (`w * h`), each value in
/// `0.0..=1.0` where **1.0 is nearest** the camera and **0.0 is farthest**.
pub struct DepthMap {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl DepthMap {
    /// Bilinearly sample the map at normalized `(u, v)` in `0..=1` (`u` across the
    /// width, `v` down the height). Lets a coarse render grid read the fixed-size
    /// depth map at any resolution.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        if self.w == 0 || self.h == 0 {
            return 0.5;
        }
        let fx = u.clamp(0.0, 1.0) * (self.w - 1) as f32;
        let fy = v.clamp(0.0, 1.0) * (self.h - 1) as f32;
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let at = |x: usize, y: usize| self.data[y * self.w + x];
        let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        top * (1.0 - ty) + bot * ty
    }
}

pub struct DepthEstimator {
    session: Session,
    input_name: String,
    output_name: String,
}

impl DepthEstimator {
    /// Load a Depth Anything V2 ONNX model. Input/output tensor names are read
    /// from the graph rather than hardcoded, since exports vary. Works with the
    /// external-data export too (ONNX Runtime resolves `model.onnx_data` from the
    /// model file's directory).
    pub fn load(model: &Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("ORT init: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ORT options: {e}"))?
            .commit_from_file(model)
            .map_err(|e| format!("Load model: {e}"))?;

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

        Ok(Self { session, input_name, output_name })
    }

    /// Estimate depth for the image at `path`. Decodes via the shared
    /// [`crate::tagger::load_rgb_on_white`] (so AVIF/RAW/HDR work too), runs
    /// inference, and min-max normalizes the relative inverse-depth output to
    /// `0..1` (`1.0` = nearest).
    pub fn estimate(&mut self, path: &Path) -> Result<DepthMap, String> {
        let img = crate::tagger::load_rgb_on_white(path)?;
        let data = preprocess(&img);
        let shape = [1_i64, 3, DSIZE as i64, DSIZE as i64];
        let tensor = Tensor::from_array((shape, data)).map_err(|e| format!("Tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| format!("Inference: {e}"))?;

        let (_out_shape, raw) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Output: {e}"))?;

        // We feed a square 518×518 input, so Depth Anything's output is square
        // too; derive the side from the element count (robust to whether the
        // export emits a [1,H,W] or [1,1,H,W] shape).
        let n = raw.len();
        if n == 0 {
            return Err("Empty depth output".to_string());
        }
        let side = (n as f64).sqrt().round() as usize;
        if side == 0 || side * side != n {
            return Err(format!("Unexpected depth output length {n} (not square)"));
        }

        // Depth Anything emits relative INVERSE depth (larger = nearer). Min-max
        // normalize to 0..1; this preserves polarity, so 1.0 ends up nearest.
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in raw.iter() {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return Err("Depth output had no finite values".to_string());
        }
        let span = (hi - lo).max(1e-6);
        let data: Vec<f32> = raw.iter().map(|&v| ((v - lo) / span).clamp(0.0, 1.0)).collect();

        Ok(DepthMap { w: side, h: side, data })
    }
}

/// Preprocess to Depth Anything V2's expected input: stretch-resize to 518²
/// (no padding — padding would inject false flat-depth borders), scale to
/// `0..1`, ImageNet mean/std normalize, planar RGB (NCHW `[1, 3, 518, 518]`).
fn preprocess(img: &RgbImage) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let resized = image::imageops::resize(img, DSIZE, DSIZE, FilterType::Triangle);
    let plane = (DSIZE * DSIZE) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for (i, p) in resized.pixels().enumerate() {
        data[i] = ((p[0] as f32 / 255.0) - MEAN[0]) / STD[0];
        data[i + plane] = ((p[1] as f32 / 255.0) - MEAN[1]) / STD[1];
        data[i + 2 * plane] = ((p[2] as f32 / 255.0) - MEAN[2]) / STD[2];
    }
    data
}

/// How long the worker stays up — keeping the model loaded — with nothing to
/// do. After that it ends, freeing the session; the next request reloads it.
const IDLE_UNLOAD: Duration = Duration::from_secs(120);

/// A finished estimate: the image it's for, and its depth map or an error.
pub type DepthResult = (PathBuf, Result<DepthMap, String>);

/// Depth estimation for the Spatial Scene, off the UI thread: a single worker
/// that keeps the model loaded between images and only ever runs the newest
/// request. (A thread with a fresh ~370 MB session per image used to pile up
/// when paging through images, several running at once.)
pub struct DepthWorker {
    shared: Arc<Shared>,
    tx: Sender<DepthResult>,
    rx: Receiver<DepthResult>,
}

struct Request {
    model: PathBuf,
    image: PathBuf,
    ctx: egui::Context,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    /// The newest request not started yet. A newer one replaces it, so an
    /// image paged past is never estimated.
    next: Option<Request>,
    /// Whether the worker thread is up (it ends after [`IDLE_UNLOAD`]).
    running: bool,
}

impl Default for DepthWorker {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { shared: Arc::default(), tx, rx }
    }
}

impl DepthWorker {
    /// Ask for `image`'s depth map, replacing any request not started yet;
    /// `ctx` is repainted when the result is ready.
    pub fn request(&self, model: PathBuf, image: PathBuf, ctx: &egui::Context) {
        let mut st = self.shared.state.lock().unwrap();
        st.next = Some(Request { model, image, ctx: ctx.clone() });
        if st.running {
            self.shared.wake.notify_one();
        } else {
            st.running = true;
            let shared = Arc::clone(&self.shared);
            let tx = self.tx.clone();
            std::thread::spawn(move || worker_loop(&shared, &tx));
        }
    }

    /// The next finished estimate, if any.
    pub fn try_recv(&self) -> Option<DepthResult> {
        self.rx.try_recv().ok()
    }
}

fn worker_loop(shared: &Shared, tx: &Sender<DepthResult>) {
    // The loaded model (and which file it came from), kept between requests.
    let mut loaded: Option<(PathBuf, DepthEstimator)> = None;
    while let Some(req) = next_request(shared) {
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !matches!(&loaded, Some((m, _)) if *m == req.model) {
                loaded = None; // free the old session before loading another
                loaded = Some((req.model.clone(), DepthEstimator::load(&req.model)?));
            }
            let (_, est) = loaded.as_mut().expect("loaded above");
            est.estimate(&req.image)
        }));
        // A panic (say, in a decoder) mustn't end the worker and leave the
        // viewer on "Generating depth…": report it, reload the model next time.
        let result = run.unwrap_or_else(|_| {
            loaded = None;
            Err("Depth estimation crashed".to_string())
        });
        {
            // Paging away and back may have asked for this same image again.
            let mut st = shared.state.lock().unwrap();
            if st.next.as_ref().is_some_and(|n| n.image == req.image && n.model == req.model) {
                st.next = None;
            }
        }
        if tx.send((req.image, result)).is_err() {
            return; // the viewer is gone
        }
        req.ctx.request_repaint();
    }
}

/// Wait for the next request; `None` after [`IDLE_UNLOAD`] without one (the
/// thread then ends, dropping the model).
fn next_request(shared: &Shared) -> Option<Request> {
    let mut st = shared.state.lock().unwrap();
    loop {
        if let Some(req) = st.next.take() {
            return Some(req);
        }
        let (guard, wait) = shared.wake.wait_timeout(st, IDLE_UNLOAD).unwrap();
        st = guard;
        if wait.timed_out() && st.next.is_none() {
            st.running = false;
            return None;
        }
    }
}
