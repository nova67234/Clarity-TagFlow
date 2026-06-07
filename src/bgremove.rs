//! Background removal via **BiRefNet** (ONNX) through the `ort` crate — the same
//! runtime the taggers and depth estimator use. Produces a transparent-background
//! PNG cutout of the subject (the matte is applied as the alpha channel).
//!
//! Model: `onnx-community/BiRefNet_lite-ONNX`, fetched by the AI Model Manager
//! into `tools/birefnet-lite-onnx/`. Input is a 1024² ImageNet-normalized NCHW
//! tensor; the output is a single-channel saliency matte (subject = bright).

use std::path::{Path, PathBuf};

use image::imageops::FilterType;
use image::{GrayImage, Rgba, RgbaImage, RgbImage};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// BiRefNet's published input resolution.
const ISIZE: u32 = 1024;

/// Run BiRefNet on `image`, writing a transparent-background PNG next to it.
/// Returns the path of the written cutout. Designed to be the body of a
/// `std::thread::spawn`, mirroring [`crate::depth::run_depth_job`].
pub fn run_bgremove_job(model: PathBuf, image: PathBuf) -> Result<PathBuf, String> {
    // Decode the original at full resolution (handles AVIF/RAW/HDR too). We keep
    // these RGB pixels for the output and only feed a resized copy to the model.
    let img = crate::tagger::load_rgb_on_white(&image)?;
    let (ow, oh) = (img.width(), img.height());
    if ow == 0 || oh == 0 {
        return Err("Empty image".to_string());
    }

    let mut session = Session::builder()
        .map_err(|e| format!("ORT init: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ORT options: {e}"))?
        .commit_from_file(&model)
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

    let data = preprocess(&img);
    let shape = [1_i64, 3, ISIZE as i64, ISIZE as i64];
    let tensor = Tensor::from_array((shape, data)).map_err(|e| format!("Tensor: {e}"))?;

    let outputs = session
        .run(ort::inputs![input_name.as_str() => tensor])
        .map_err(|e| format!("Inference: {e}"))?;
    let (_out_shape, raw) = outputs[output_name.as_str()]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Output: {e}"))?;

    // The matte is square (square input). Derive its side from the element count.
    let n = raw.len();
    if n == 0 {
        return Err("Empty matte output".to_string());
    }
    let side = (n as f64).sqrt().round() as usize;
    if side == 0 || side * side != n {
        return Err(format!("Unexpected matte length {n} (not square)"));
    }

    // Some exports emit probabilities in 0..1; others emit raw logits. Sigmoid only
    // when the values fall outside 0..1, so we don't double-squash a probability.
    let (lo, hi) = raw
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let needs_sigmoid = lo < -0.001 || hi > 1.001;
    let mask: Vec<u8> = raw
        .iter()
        .map(|&v| {
            let p = if needs_sigmoid { 1.0 / (1.0 + (-v).exp()) } else { v };
            (p.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();

    // Resize the matte to the original resolution and use it as the alpha channel.
    let mask_img = GrayImage::from_raw(side as u32, side as u32, mask)
        .ok_or_else(|| "Could not build matte image".to_string())?;
    let alpha = image::imageops::resize(&mask_img, ow, oh, FilterType::Triangle);

    let mut out = RgbaImage::new(ow, oh);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let rgb = img.get_pixel(x, y);
        *px = Rgba([rgb[0], rgb[1], rgb[2], alpha.get_pixel(x, y)[0]]);
    }

    let dest = bg_destination(&image);
    out.save(&dest).map_err(|e| format!("Save: {e}"))?;
    Ok(dest)
}

/// Preprocess to BiRefNet's input: stretch-resize to 1024², scale to 0..1,
/// ImageNet mean/std normalize, planar RGB (NCHW `[1, 3, 1024, 1024]`).
fn preprocess(img: &RgbImage) -> Vec<f32> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let resized = image::imageops::resize(img, ISIZE, ISIZE, FilterType::Triangle);
    let plane = (ISIZE * ISIZE) as usize;
    let mut data = vec![0.0f32; 3 * plane];
    for (i, p) in resized.pixels().enumerate() {
        data[i] = ((p[0] as f32 / 255.0) - MEAN[0]) / STD[0];
        data[i + plane] = ((p[1] as f32 / 255.0) - MEAN[1]) / STD[1];
        data[i + 2 * plane] = ((p[2] as f32 / 255.0) - MEAN[2]) / STD[2];
    }
    data
}

/// Output path for a cutout: `<stem>_nobg.png` beside the source.
pub fn bg_destination(src: &Path) -> PathBuf {
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let dir = src.parent().map(Path::to_path_buf).unwrap_or_default();
    dir.join(format!("{stem}_nobg.png"))
}
