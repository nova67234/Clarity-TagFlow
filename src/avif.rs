//! Pure-Rust AVIF decoding, gated behind the `avif` cargo feature.
//!
//! `avif-parse` pulls the AV1 frame (an OBU byte stream) out of the AVIF/HEIF
//! container; `rav1d` (a Rust port of dav1d, no C library / nasm needed) decodes
//! it to YUV; we convert YUV → RGBA here.
//!
//! rav1d only exposes the dav1d **C ABI**, but those functions and structs are
//! `pub`, so we call them through their Rust paths (`rav1d::src::lib::*` /
//! `rav1d::include::dav1d::*`). Calling them by path — rather than re-declaring
//! `extern "C"` symbols — is what makes cargo actually link librav1d, and it
//! reuses rav1d's own struct definitions so there's no hand-written ABI to get
//! wrong.
//!
//! Scope/caveats (a "make it viewable" decoder, not colour-exact): limited-range
//! BT.601 coefficients regardless of the file's signalled matrix/range, and
//! container transforms (rotate/mirror/crop) are not applied. Fine to browse and
//! tag; not a reference renderer.

use std::ptr::NonNull;

use image::RgbaImage;

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_default_settings, dav1d_get_picture, dav1d_open,
    dav1d_picture_unref, dav1d_send_data,
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Decode an AVIF, HEIC, or HEIF file to an RGBA image. Returns `None` on any
/// failure so the caller falls back to the usual "couldn't load" placeholder.
///
/// HEIC/HEIF (HEVC-coded) goes through the pure-Rust `heic` crate; AVIF
/// (AV1-coded) goes through avif-parse + rav1d below. They share the `.heif`
/// extension, so we try the AVIF path first and fall back to HEVC if it fails.
pub fn decode_avif(path: &std::path::Path) -> Option<RgbaImage> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    if ext == "heic" {
        return decode_heic(path);
    }
    if matches!(ext.as_str(), "dng" | "arw") {
        return decode_raw(path);
    }

    let bytes = std::fs::read(path).ok()?;
    let parsed = match avif_parse::read_avif(&mut bytes.as_slice()) {
        Ok(p) => p,
        // A `.heif`/`.avif` that isn't actually AV1 — try the HEVC decoder.
        Err(_) => return decode_heic(path),
    };

    let (mut rgba, w, h) = decode_obu_to_rgba(&parsed.primary_item)?;
    if let Some(alpha_obu) = &parsed.alpha_item {
        if let Some((alpha, aw, ah)) = decode_gray(alpha_obu) {
            if aw == w && ah == h {
                for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
                    px[3] = alpha[i];
                }
            }
        }
    }
    RgbaImage::from_raw(w, h, rgba)
}

/// Decode a HEIC/HEIF (HEVC-coded) file via the pure-Rust `heic` crate.
fn decode_heic(path: &std::path::Path) -> Option<RgbaImage> {
    let bytes = std::fs::read(path).ok()?;
    let out = heic::DecoderConfig::new()
        .decode(&bytes, heic::PixelLayout::Rgba8)
        .ok()?;
    RgbaImage::from_raw(out.width, out.height, out.data)
}

/// sRGB→XYZ matrix for the D65 white point (the standard dcraw constants).
/// Used to build the camera→sRGB color matrix.
const XYZ_RGB: [[f64; 3]; 3] = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
];

/// Invert a 3×3 matrix; returns the identity if it's singular (degenerate input
/// shouldn't happen for a real camera matrix, but we never want to divide by 0).
fn invert_3x3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    let mut out = [[0.0; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    out
}

/// Build the white-preserving camera-RGB → linear-sRGB matrix from the DNG's
/// `xyz_to_cam` matrix (camera's XYZ→camera-RGB, 4×3, rows R/G/B/E). This is the
/// dcraw `cam_xyz_coeff` algorithm: form `cam_rgb = xyz_to_cam · XYZ_RGB`
/// (mapping linear-sRGB → camera), normalise each row so it sums to 1 (so a
/// neutral camera pixel maps to a neutral sRGB pixel — the step zenraw's own
/// develop skips, which is why its output has a magenta cast), then invert.
fn camera_to_srgb_matrix(xyz_to_cam: [[f32; 3]; 4]) -> [[f64; 3]; 3] {
    let mut cam_rgb = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += xyz_to_cam[i][k] as f64 * XYZ_RGB[k][j];
            }
            cam_rgb[i][j] = s;
        }
    }
    for row in cam_rgb.iter_mut() {
        let sum: f64 = row.iter().sum();
        if sum.abs() > 1e-12 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    invert_3x3(cam_rgb)
}

/// Encode a linear value [0,1] to 8-bit sRGB (standard sRGB transfer curve).
fn linear_to_srgb8(v: f64) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Read a little-endian f32 sample from a row's bytes at byte offset `o`.
fn read_f32(row: &[u8], o: usize) -> f64 {
    let b = [
        row.get(o).copied().unwrap_or(0),
        row.get(o + 1).copied().unwrap_or(0),
        row.get(o + 2).copied().unwrap_or(0),
        row.get(o + 3).copied().unwrap_or(0),
    ];
    f32::from_le_bytes(b) as f64
}

/// Decode a camera-raw file (DNG, Sony ARW, …) to display-ready sRGB.
///
/// zenraw auto-detects the format from the bytes, so a single path handles every
/// raw type its rawloader backend supports — we just dispatch any raw extension
/// here. Tries a full raw develop first ([`decode_raw_develop`]); if that fails —
/// e.g. the camera model isn't supported by the backend (DJI drones, some newer
/// Sony bodies, exotic compression all report `Unsupported`) — falls back to the
/// camera-rendered JPEG preview embedded in the file ([`decode_raw_embedded_jpeg`]).
/// The preview has correct colour and is usually full or near-full resolution, so
/// the file still displays well even when we can't develop its raw sensor data.
fn decode_raw(path: &std::path::Path) -> Option<RgbaImage> {
    let bytes = std::fs::read(path).ok()?;
    if let Some(img) = decode_raw_develop(&bytes) {
        return Some(img);
    }
    decode_raw_embedded_jpeg(&bytes)
}

/// Scan a raw file (DNG/ARW/TIFF-based) for embedded JPEG previews and decode the
/// largest one.
///
/// Every such file carries at least one camera-rendered JPEG (a small thumbnail,
/// and usually a larger preview). We byte-scan for JPEG start markers (`FF D8 FF`),
/// decode each candidate, and keep the one with the most pixels. Used as the
/// fallback when the raw develop can't run.
///
/// The previews are stored in *sensor* orientation; the rotation/flip needed to
/// view them upright lives in the container's TIFF `Orientation` tag (274), not
/// inside each preview JPEG — so we read that once and apply it (falling back to
/// the JPEG's own EXIF orientation if the container has none).
fn decode_raw_embedded_jpeg(bytes: &[u8]) -> Option<RgbaImage> {
    let container_orientation = tiff_orientation(bytes);

    let mut best: Option<RgbaImage> = None;
    let mut best_px = 0u64;
    let mut attempts = 0u32;
    let mut i = 0usize;
    while i + 3 < bytes.len() {
        if bytes[i] == 0xFF && bytes[i + 1] == 0xD8 && bytes[i + 2] == 0xFF {
            if let Some(img) = decode_jpeg_oriented(&bytes[i..], container_orientation) {
                let px = img.width() as u64 * img.height() as u64;
                if px > best_px {
                    best_px = px;
                    best = Some(img);
                }
            }
            // Count every JPEG-marker candidate (not just successful decodes) so a
            // run of false `FF D8 FF` byte coincidences can't make us scan forever.
            attempts += 1;
            if attempts >= 64 {
                break;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    best
}

/// Read the TIFF `Orientation` tag (274) from IFD0 of a DNG/TIFF container.
/// Returns the raw EXIF orientation value (1–8), or `None` if absent/malformed.
///
/// A DNG begins with a TIFF header: byte-order mark (`II`=little / `MM`=big),
/// magic `42`, then the offset to IFD0. IFD0 is a count followed by 12-byte
/// entries (tag, type, count, value); for a SHORT the value sits in the first two
/// bytes of the value field. We only need tag 274, so we do a tiny hand parse
/// rather than pulling in a full TIFF reader.
fn tiff_orientation(bytes: &[u8]) -> Option<u8> {
    if bytes.len() < 8 {
        return None;
    }
    let le = match &bytes[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16a = |b: &[u8]| -> u16 {
        if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) }
    };
    let u32a = |b: &[u8]| -> u32 {
        if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        }
    };
    if u16a(&bytes[2..4]) != 42 {
        return None;
    }
    let ifd = u32a(&bytes[4..8]) as usize;
    if ifd + 2 > bytes.len() {
        return None;
    }
    let count = u16a(&bytes[ifd..ifd + 2]) as usize;
    for e in 0..count {
        let off = ifd + 2 + e * 12;
        if off + 12 > bytes.len() {
            break;
        }
        if u16a(&bytes[off..off + 2]) == 274 {
            let v = u16a(&bytes[off + 8..off + 10]);
            if (1..=8).contains(&v) {
                return Some(v as u8);
            }
        }
    }
    None
}

/// Decode a JPEG and apply orientation. `container_orientation` is the DNG's
/// TIFF orientation tag (preferred, since DNG previews carry no EXIF of their
/// own); if it's `None` we fall back to the JPEG's own EXIF orientation. The
/// `image` crate doesn't honour either automatically, so we rotate/flip here.
fn decode_jpeg_oriented(bytes: &[u8], container_orientation: Option<u8>) -> Option<RgbaImage> {
    use image::ImageDecoder;
    let mut decoder =
        image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(bytes)).ok()?;
    let orientation = container_orientation
        .and_then(image::metadata::Orientation::from_exif)
        .or_else(|| decoder.orientation().ok())
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    Some(img.into_rgba8())
}

/// Full raw develop of a camera-raw file (DNG/ARW/…) via zenraw. Returns `None`
/// if zenraw can't decode the file (unsupported camera, exotic compression, …) so
/// the caller can fall back to the embedded preview.
///
/// zenraw 0.2's built-in `Develop` mode renders real DNGs with a strong magenta
/// cast: its camera→sRGB matrix isn't white-preserving (it normalises the
/// XYZ→camera matrix before inverting rather than making camera-white map to
/// display-white), so neutral tones come out tinted. We can't fix that through
/// its config, so we request [`OutputMode::CameraRaw`] — demosaiced, cropped,
/// oriented camera-RGB as f32 in `[0,1]`, with no white balance or colour matrix
/// — and run our own develop:
///
/// 1. **Gray-world white balance.** The as-shot WB coefficients rawloader reads
///    from older DNGs are unreliable (e.g. a Canon 350D file reports
///    `[2.25, 1.0, 1.49]`, which over-boosts red/blue into a magenta cast).
///    Rather than trust them, we scale R and B so the image's *mean* matches
///    green — a metadata-independent estimate that reproduces the camera's own
///    near-neutral average and generalises across cameras.
/// 2. **White-preserving camera→sRGB matrix** (see [`camera_to_srgb_matrix`]),
///    so a neutral (gray-world-balanced) pixel maps to neutral sRGB.
/// 3. **sRGB transfer curve.**
fn decode_raw_develop(bytes: &[u8]) -> Option<RgbaImage> {
    let cfg = zenraw::RawDecodeConfig::new().with_output(zenraw::OutputMode::CameraRaw);
    let out = zenraw::decode(bytes, &cfg, &enough::Unstoppable).ok()?;

    let info = &out.info;
    let buf = &out.pixels;
    let w = buf.width();
    let h = buf.height();
    if w == 0 || h == 0 {
        return None;
    }
    let desc = buf.descriptor();
    let channels = desc.layout().channels(); // 3 (RGB) for CameraRaw
    if channels < 3 {
        return None;
    }

    let slice = buf.as_slice();
    let bpp = channels * 4; // f32 samples

    // Pass 1 — channel means for gray-world white balance.
    let (mut mr, mut mg, mut mb, mut n) = (0f64, 0f64, 0f64, 0f64);
    for y in 0..h {
        let row = slice.row(y);
        for x in 0..w as usize {
            let o = x * bpp;
            mr += read_f32(row, o);
            mg += read_f32(row, o + 4);
            mb += read_f32(row, o + 8);
            n += 1.0;
        }
    }
    // Scale R and B so their mean equals green's (green left unchanged).
    let wb = [
        if mr > 1e-9 { mg / mr } else { 1.0 },
        1.0,
        if mb > 1e-9 { mg / mb } else { 1.0 },
    ];
    let _ = (n, info.wb_coeffs); // (means already accumulated; as-shot WB intentionally unused)

    let m = camera_to_srgb_matrix(info.color_matrix);

    // Pass 2 — apply WB, the camera→sRGB matrix, and the sRGB curve.
    let mut rgba = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for y in 0..h {
        let row = slice.row(y);
        for x in 0..w as usize {
            let o = x * bpp;
            let cr = read_f32(row, o) * wb[0];
            let cg = read_f32(row, o + 4) * wb[1];
            let cb = read_f32(row, o + 8) * wb[2];
            let sr = m[0][0] * cr + m[0][1] * cg + m[0][2] * cb;
            let sg = m[1][0] * cr + m[1][1] * cg + m[1][2] * cb;
            let sb = m[2][0] * cr + m[2][1] * cg + m[2][2] * cb;
            rgba.push(linear_to_srgb8(sr));
            rgba.push(linear_to_srgb8(sg));
            rgba.push(linear_to_srgb8(sb));
            rgba.push(255);
        }
    }

    RgbaImage::from_raw(w, h, rgba)
}

// ---------------------------------------------------------------------------
// rav1d decode → a filled Dav1dPicture
// ---------------------------------------------------------------------------

/// Owns an open decoder context + decoded picture; frees both on drop.
struct Decoded {
    ctx: Option<Dav1dContext>,
    pic: Dav1dPicture,
}

impl Drop for Decoded {
    fn drop(&mut self) {
        unsafe {
            dav1d_picture_unref(Some(NonNull::from(&mut self.pic)));
            if self.ctx.is_some() {
                let mut c = self.ctx;
                dav1d_close(Some(NonNull::from(&mut c)));
            }
        }
    }
}

/// Decode a single AV1 OBU and return the filled picture.
fn decode_obu(obu: &[u8]) -> Option<Decoded> {
    unsafe {
        // Settings: zero-init (all the Option<fn>/Option<NonNull> fields become
        // None), then let dav1d fill its defaults.
        let mut settings: Dav1dSettings = std::mem::zeroed();
        dav1d_default_settings(NonNull::from(&mut settings));
        settings.n_threads = 1; // single still image — no thread pool needed

        // Open the decoder.
        let mut ctx: Option<Dav1dContext> = None;
        let r = dav1d_open(Some(NonNull::from(&mut ctx)), Some(NonNull::from(&mut settings)));
        if r.0 != 0 || ctx.is_none() {
            return None;
        }

        // Allocate a dav1d-owned buffer and copy our OBU into it (so we don't
        // have to deal with the crate-private free-callback type).
        let mut data: Dav1dData = Default::default();
        let buf = dav1d_data_create(Some(NonNull::from(&mut data)), obu.len());
        if buf.is_null() {
            let mut c = ctx;
            dav1d_close(Some(NonNull::from(&mut c)));
            return None;
        }
        std::ptr::copy_nonoverlapping(obu.as_ptr(), buf, obu.len());

        // Send the frame, then pull the decoded picture. For a complete still a
        // single send + get suffices.
        if dav1d_send_data(ctx, Some(NonNull::from(&mut data))).0 != 0 {
            let mut c = ctx;
            dav1d_close(Some(NonNull::from(&mut c)));
            return None;
        }

        let mut pic: Dav1dPicture = std::mem::zeroed();
        if dav1d_get_picture(ctx, Some(NonNull::from(&mut pic))).0 != 0 {
            let mut c = ctx;
            dav1d_close(Some(NonNull::from(&mut c)));
            return None;
        }

        Some(Decoded { ctx, pic })
    }
}

// ---------------------------------------------------------------------------
// YUV → RGBA
// ---------------------------------------------------------------------------

/// Decode an OBU and convert YUV → packed RGBA8 (alpha = 255).
fn decode_obu_to_rgba(obu: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let dec = decode_obu(obu)?;
    let p = &dec.pic;
    let w = p.p.w.max(0) as usize;
    let h = p.p.h.max(0) as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let bpc = p.p.bpc as u32;
    let shift = bpc.saturating_sub(8);
    let layout = p.p.layout; // 0=I400, 1=I420, 2=I422, 3=I444

    let y_ptr = p.data[0].map(|p| p.as_ptr() as *const u8)?;
    let u_ptr = p.data[1].map(|p| p.as_ptr() as *const u8);
    let v_ptr = p.data[2].map(|p| p.as_ptr() as *const u8);
    let y_stride = p.stride[0];
    let c_stride = p.stride[1];

    let sample = |base: *const u8, stride: isize, x: usize, y: usize| -> u8 {
        unsafe {
            if bpc > 8 {
                let row = base.offset(stride * y as isize) as *const u16;
                (*row.add(x) >> shift) as u8
            } else {
                *base.offset(stride * y as isize).add(x)
            }
        }
    };

    let chroma_xy = |x: usize, y: usize| -> (usize, usize) {
        match layout {
            1 => (x / 2, y / 2), // I420
            2 => (x / 2, y),     // I422
            3 => (x, y),         // I444
            _ => (0, 0),         // I400
        }
    };

    let mono = layout == 0 || u_ptr.is_none() || v_ptr.is_none();
    let mut out = vec![0u8; w * h * 4];

    for y in 0..h {
        for x in 0..w {
            let yv = sample(y_ptr, y_stride, x, y) as f32;
            let (r, g, b) = if mono {
                (yv, yv, yv)
            } else {
                let (cx, cy) = chroma_xy(x, y);
                let u = sample(u_ptr.unwrap(), c_stride, cx, cy) as f32 - 128.0;
                let v = sample(v_ptr.unwrap(), c_stride, cx, cy) as f32 - 128.0;
                let yl = (yv - 16.0) * 1.164;
                (yl + 1.596 * v, yl - 0.392 * u - 0.813 * v, yl + 2.017 * u)
            };
            let i = (y * w + x) * 4;
            out[i] = r.clamp(0.0, 255.0) as u8;
            out[i + 1] = g.clamp(0.0, 255.0) as u8;
            out[i + 2] = b.clamp(0.0, 255.0) as u8;
            out[i + 3] = 255;
        }
    }
    Some((out, w as u32, h as u32))
}

/// Decode an OBU as single-channel luma — used for the alpha plane.
fn decode_gray(obu: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let dec = decode_obu(obu)?;
    let p = &dec.pic;
    let w = p.p.w.max(0) as usize;
    let h = p.p.h.max(0) as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let bpc = p.p.bpc as u32;
    let shift = bpc.saturating_sub(8);
    let y_ptr = p.data[0].map(|p| p.as_ptr() as *const u8)?;
    let y_stride = p.stride[0];

    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = unsafe {
                if bpc > 8 {
                    let row = y_ptr.offset(y_stride * y as isize) as *const u16;
                    (*row.add(x) >> shift) as u8
                } else {
                    *y_ptr.offset(y_stride * y as isize).add(x)
                }
            };
        }
    }
    Some((out, w as u32, h as u32))
}

// TEMP smoke test: decode sample AVIFs to PNG so the result can be eyeballed.
// cargo test --no-default-features --features avif avif_smoke -- --nocapture
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avif_smoke() {
        // The sample set lives in `avif/` (gitignored, local only). Skip cleanly
        // when it's absent so the test never fails on a fresh checkout / CI.
        if !std::path::Path::new("avif").is_dir() {
            eprintln!("skipping avif_smoke: no local avif/ sample folder");
            return;
        }
        let cases = [
            "avif/fox.profile0.8bpc.yuv420.avif",
            "avif/fox.profile1.8bpc.yuv444.avif",
            "avif/fox.profile2.10bpc.yuv422.avif",
            "avif/kimono.avif",
            "avif/plum-blossom-small.profile0.8bpc.yuv420.alpha-full.avif",
        ];
        for c in cases {
            if !std::path::Path::new(c).exists() {
                continue;
            }
            match decode_avif(std::path::Path::new(c)) {
                Some(img) => {
                    let out = format!("{c}.decoded.png");
                    img.save(&out).unwrap();
                    println!("OK {c} -> {}x{} -> {out}", img.width(), img.height());
                }
                None => println!("FAIL {c}"),
            }
        }
    }

    // cargo test --no-default-features --features avif heic_smoke -- --nocapture
    #[test]
    fn heic_smoke() {
        let dir = std::path::Path::new("tests/HEIC");
        if !dir.is_dir() {
            eprintln!("skipping heic_smoke: no tests/HEIC folder");
            return;
        }
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("heic"))
                != Some(true)
            {
                continue;
            }
            match decode_avif(&p) {
                Some(img) => {
                    let out = format!("{}.decoded.png", p.display());
                    img.save(&out).unwrap();
                    println!("OK {} -> {}x{} -> {out}", p.display(), img.width(), img.height());
                }
                None => println!("FAIL {}", p.display()),
            }
        }
    }

    // Decode every raw sample (DNG + ARW) to a PNG so the result can be eyeballed.
    // Skips cleanly when the local sample folders are absent.
    // cargo test --no-default-features --features avif raw_smoke -- --nocapture
    #[test]
    fn raw_smoke() {
        for (dir, ext) in [("tests/DNG", "dng"), ("tests/ARW", "arw")] {
            let dir = std::path::Path::new(dir);
            if !dir.is_dir() {
                eprintln!("skipping {ext}: no {} folder", dir.display());
                continue;
            }
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case(ext))
                    != Some(true)
                {
                    continue;
                }
                match decode_avif(&p) {
                    Some(img) => {
                        let out = format!("{}.decoded.png", p.display());
                        img.save(&out).unwrap();
                        println!("OK {} -> {}x{} -> {out}", p.display(), img.width(), img.height());
                    }
                    None => println!("FAIL {}", p.display()),
                }
            }
        }
    }
}
