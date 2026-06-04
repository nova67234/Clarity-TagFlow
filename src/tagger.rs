//! ONNX image tagging inference, via the `ort` crate (ONNX Runtime).
//!
//! A Rust port of terminus2's `WD14Tagger` / `JoyTag` / `PixaiTagger`. Each model
//! family has its own preprocessing, tag-file format, and output handling:
//!
//! * **WD14** (SmilingWolf) — aspect-preserving resize onto a white pad, BGR
//!   0–255, NHWC layout; outputs are already probabilities. Rating tags
//!   (category 9) are dropped.
//! * **JoyTag** — pad to square (white), resize 448, CLIP mean/std normalise,
//!   planar RGB NCHW; outputs are logits → sigmoid.
//! * **PixAI** — stretch to 448 (white background for alpha), `/255` then
//!   mean/std 0.5 → [-1, 1], planar RGB NCHW; reads the `prediction` output
//!   (already sigmoid). Character tags (category 4) require a higher threshold.
//!
//! Models are loaded from `tools/<folder>/` (see [`resolve`]). Loading +
//! inference are heavy (some models are ~1.3 GB), so callers run [`run_job`] on a
//! background thread.

use std::path::{Path, PathBuf};

use image::{imageops::FilterType, Rgb, RgbImage};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

const SIZE: u32 = 448;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TaggerKind {
    Wd14,
    JoyTag,
    Pixai,
}

/// The writable directory where downloaded models live. A per-user data dir
/// (e.g. `%APPDATA%\Clarity TagFlow\tools` on Windows,
/// `~/Library/Application Support/Clarity TagFlow/tools` on macOS) so it works
/// even when the app itself is installed read-only (a `.app`, Program Files).
/// Falls back to a local `tools/` dir if no data dir is available.
pub fn models_root() -> PathBuf {
    dirs::data_dir()
        .map(|d| d.join("Clarity TagFlow").join("tools"))
        .unwrap_or_else(|| PathBuf::from("tools"))
}

/// All directories searched for an installed model: the writable data dir
/// first, then a local `tools/` (portable / dev) and the terminus2 resources
/// path.
fn read_roots() -> Vec<PathBuf> {
    vec![
        models_root(),
        PathBuf::from("tools"),
        PathBuf::from("src/main/resources/tools"),
    ]
}

/// Find a model file across the searched roots, mirroring the Java
/// `resolveToolFile`.
pub fn resolve(folder: &str, file: &str) -> Option<PathBuf> {
    for base in read_roots() {
        let p = base.join(folder).join(file);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

struct TagInfo {
    name: String,
    category: i32,
}

pub struct Tagger {
    session: Session,
    input_name: String,
    output_name: String,
    output_is_logits: bool,
    tags: Vec<TagInfo>,
    kind: TaggerKind,
}

impl Tagger {
    pub fn load(kind: TaggerKind, model: &Path, tags_path: &Path) -> Result<Tagger, String> {
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

        let out_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
        if out_names.is_empty() {
            return Err("Model has no outputs".to_string());
        }

        // PixAI's graph exposes prediction/embedding/logits; pick the scores
        // output by NAME (index 0 could be the embedding → nonsense tags).
        let (output_name, output_is_logits) = if kind == TaggerKind::Pixai {
            if out_names.iter().any(|n| n == "prediction") {
                ("prediction".to_string(), false)
            } else if out_names.iter().any(|n| n == "logits") {
                ("logits".to_string(), true)
            } else {
                (out_names[0].clone(), false)
            }
        } else {
            (out_names[0].clone(), false)
        };

        let tags = match kind {
            TaggerKind::Wd14 => load_tags_csv(tags_path, 1, 2)?, // name col 1, category col 2
            TaggerKind::Pixai => load_tags_csv(tags_path, 2, 3)?, // name col 2, category col 3
            TaggerKind::JoyTag => load_tags_txt(tags_path)?,
        };

        Ok(Tagger { session, input_name, output_name, output_is_logits, tags, kind })
    }

    /// Run the model on `image_path`, returning tag names with confidence ≥
    /// `threshold`, highest-confidence first.
    pub fn predict(&mut self, image_path: &Path, threshold: f32) -> Result<Vec<String>, String> {
        let img = load_rgb_on_white(image_path)?;

        let (data, shape): (Vec<f32>, [i64; 4]) = match self.kind {
            TaggerKind::Wd14 => (preprocess_wd14(&img), [1, SIZE as i64, SIZE as i64, 3]),
            TaggerKind::JoyTag => (preprocess_joytag(&img), [1, 3, SIZE as i64, SIZE as i64]),
            TaggerKind::Pixai => (preprocess_pixai(&img), [1, 3, SIZE as i64, SIZE as i64]),
        };

        let tensor = Tensor::from_array((shape, data)).map_err(|e| format!("Tensor: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| format!("Inference: {e}"))?;

        let (_shape, raw) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Output: {e}"))?;

        let n = raw.len().min(self.tags.len());

        // Decide whether the raw values need a sigmoid. JoyTag is always logits;
        // PixAI's "logits" output too; otherwise sniff for out-of-[0,1] values.
        let mut need_sigmoid = matches!(self.kind, TaggerKind::JoyTag) || self.output_is_logits;
        if self.kind == TaggerKind::Pixai && !need_sigmoid {
            need_sigmoid = raw[..n].iter().any(|&v| v < 0.0 || v > 1.0);
        }

        let mut scored: Vec<(f32, &str)> = Vec::new();
        for i in 0..n {
            let p = if need_sigmoid { 1.0 / (1.0 + (-raw[i]).exp()) } else { raw[i] };
            let info = &self.tags[i];

            // WD14: drop rating tags (category 9).
            if self.kind == TaggerKind::Wd14 && info.category == 9 {
                continue;
            }
            // PixAI: character tags (category 4) are held to a higher bar.
            let thr = if self.kind == TaggerKind::Pixai && info.category == 4 {
                threshold.max(0.85)
            } else {
                threshold
            };
            if p >= thr {
                scored.push((p, &info.name));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().map(|(_, name)| name.to_string()).collect())
    }
}

/// Load (or reuse) a tagger and predict, returning the tagger so the caller can
/// re-cache it. Designed to be the body of a background thread.
pub fn run_job(
    existing: Option<Tagger>,
    kind: TaggerKind,
    model: PathBuf,
    tags: PathBuf,
    image: &Path,
    threshold: f32,
) -> (Option<Tagger>, Result<Vec<String>, String>) {
    let mut tagger = match existing {
        Some(t) => t,
        None => match Tagger::load(kind, &model, &tags) {
            Ok(t) => t,
            Err(e) => return (None, Err(e)),
        },
    };
    let result = tagger.predict(image, threshold);
    (Some(tagger), result)
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

/// Decode an image and flatten any alpha onto a white background (all three
/// taggers were trained with white padding / `force_background='white'`).
fn load_rgb_on_white(path: &Path) -> Result<RgbImage, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let rgba = if ext == "hdr" {
        // HDR can't be read by `image::open` directly (linear floats + strict
        // signature check); tone-map it the same way the viewer does so the tagger
        // sees the displayed image.
        crate::image_cache::decode_hdr(path).ok_or_else(|| "Read image: HDR decode failed".to_string())?
    } else if is_extended_format(&ext) {
        // AVIF/HEIC and camera raw (DNG/ARW/CR2/NEF) can't be read by
        // `image::open`; decode them through the same pure-Rust path the viewer
        // uses so these files can be AI-tagged too.
        decode_extended(path)?
    } else if matches!(ext.as_str(), "tif" | "tiff") {
        // Most TIFFs decode normally, but raw/JPEG-compressed ones (rendered image
        // JPEG-compressed in IFD0, raw CFA in a sub-IFD) make `image` bail with
        // "unknown photometric interpretation"; recover the embedded camera JPEG
        // so these files can be tagged too — the same fallback the viewer uses.
        match image::open(path) {
            Ok(img) => img.to_rgba8(),
            Err(e) => std::fs::read(path)
                .ok()
                .and_then(|b| crate::raw_preview::largest_embedded_jpeg(&b))
                .ok_or_else(|| format!("Read image: {e}"))?,
        }
    } else {
        image::open(path).map_err(|e| format!("Read image: {e}"))?.to_rgba8()
    };
    let mut out = RgbImage::new(rgba.width(), rgba.height());
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = p[3] as f32 / 255.0;
        let blend = |c: u8| (c as f32 * a + 255.0 * (1.0 - a)).round() as u8;
        out.put_pixel(x, y, Rgb([blend(p[0]), blend(p[1]), blend(p[2])]));
    }
    Ok(out)
}

/// Extended formats (AVIF/HEIC + camera raw) that need the pure-Rust decoders in
/// `crate::avif` rather than `image::open`. Always `false` in builds without the
/// `avif` feature (those files aren't recognised as images anyway).
fn is_extended_format(ext: &str) -> bool {
    #[cfg(feature = "avif")]
    {
        matches!(ext, "avif" | "heic" | "heif" | "dng" | "arw" | "cr2" | "nef")
    }
    #[cfg(not(feature = "avif"))]
    {
        let _ = ext;
        false
    }
}

/// Decode an extended-format file (see [`is_extended_format`]) to RGBA via the
/// shared `crate::avif` decoders. Only reachable with the `avif` feature.
#[cfg(feature = "avif")]
fn decode_extended(path: &Path) -> Result<image::RgbaImage, String> {
    crate::avif::decode_avif(path).ok_or_else(|| "Read image: extended-format decode failed".to_string())
}

#[cfg(not(feature = "avif"))]
fn decode_extended(_path: &Path) -> Result<image::RgbaImage, String> {
    Err("Read image: extended formats need the `avif` build feature".to_string())
}

/// WD14: aspect-preserving resize centred on a white pad; BGR 0–255, NHWC.
fn preprocess_wd14(img: &RgbImage) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let scale = (SIZE as f32 / w as f32).min(SIZE as f32 / h as f32);
    let dw = ((w as f32 * scale).round() as u32).max(1);
    let dh = ((h as f32 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(img, dw, dh, FilterType::CatmullRom);

    let mut canvas = RgbImage::from_pixel(SIZE, SIZE, Rgb([255, 255, 255]));
    let ox = ((SIZE - dw) / 2) as i64;
    let oy = ((SIZE - dh) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized, ox, oy);

    let mut data = Vec::with_capacity((SIZE * SIZE * 3) as usize);
    for p in canvas.pixels() {
        data.push(p[2] as f32); // B
        data.push(p[1] as f32); // G
        data.push(p[0] as f32); // R
    }
    data
}

/// JoyTag: pad to square (white), resize 448, CLIP mean/std normalise, planar RGB.
fn preprocess_joytag(img: &RgbImage) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    let (w, h) = (img.width(), img.height());
    let m = w.max(h);
    let mut square = RgbImage::from_pixel(m, m, Rgb([255, 255, 255]));
    image::imageops::overlay(&mut square, img, ((m - w) / 2) as i64, ((m - h) / 2) as i64);
    let resized = image::imageops::resize(&square, SIZE, SIZE, FilterType::CatmullRom);

    let plane = (SIZE * SIZE) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for (i, p) in resized.pixels().enumerate() {
        data[i] = ((p[0] as f32 / 255.0) - MEAN[0]) / STD[0];
        data[i + plane] = ((p[1] as f32 / 255.0) - MEAN[1]) / STD[1];
        data[i + 2 * plane] = ((p[2] as f32 / 255.0) - MEAN[2]) / STD[2];
    }
    data
}

/// PixAI: stretch directly to 448 (bilinear), `/255`, normalise mean/std 0.5
/// → [-1, 1]; planar RGB.
fn preprocess_pixai(img: &RgbImage) -> Vec<f32> {
    let resized = image::imageops::resize(img, SIZE, SIZE, FilterType::Triangle);
    let plane = (SIZE * SIZE) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for (i, p) in resized.pixels().enumerate() {
        data[i] = ((p[0] as f32 / 255.0) - 0.5) / 0.5;
        data[i + plane] = ((p[1] as f32 / 255.0) - 0.5) / 0.5;
        data[i + 2 * plane] = ((p[2] as f32 / 255.0) - 0.5) / 0.5;
    }
    data
}

// ---------------------------------------------------------------------------
// Tag file parsing
// ---------------------------------------------------------------------------

/// Parse a SmilingWolf/DeepGHS `selected_tags.csv`. `name_col`/`cat_col` select
/// the columns (WD14: 1/2, PixAI deepghs: 2/3). The trailing PixAI `ips` column
/// can contain commas, so we never rely on columns past `cat_col`.
fn load_tags_csv(path: &Path, name_col: usize, cat_col: usize) -> Result<Vec<TagInfo>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Read tags: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() > cat_col {
            let name = parts[name_col].trim().replace('"', "");
            let category = parts[cat_col].trim().parse::<i32>().unwrap_or(0);
            out.push(TagInfo { name, category });
        } else {
            out.push(TagInfo { name: "unknown".into(), category: 0 });
        }
    }
    Ok(out)
}

/// Parse JoyTag's `top_tags.txt` — one tag per non-blank line.
fn load_tags_txt(path: &Path) -> Result<Vec<TagInfo>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Read tags: {e}"))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| TagInfo { name: l.trim().replace('"', "").replace(',', ""), category: 0 })
        .collect())
}
