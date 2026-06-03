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
    if matches!(ext.as_str(), "dng" | "arw" | "cr2" | "nef") {
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

/// Normalise each row of a 3×3 matrix so it sums to 1 (rows that sum to ~0 are
/// left alone). When the *final* camera→sRGB matrix has rows summing to 1, a
/// white-balanced neutral pixel `[n,n,n]` maps to a neutral sRGB `[n,n,n]` —
/// i.e. the matrix is white-preserving.
fn normalize_rows(mut m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    for row in m.iter_mut() {
        let sum: f64 = row.iter().sum();
        if sum.abs() > 1e-12 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    m
}

/// Build the white-preserving camera-RGB → linear-sRGB matrix from the camera's
/// `xyz_to_cam` matrix (XYZ→camera-RGB, 4×3, rows R/G/B/E; we use the top 3×3).
///
/// This mirrors a reference developer (zenraw's `compute_cam_to_output_matrix`):
///   1. normalise the rows of `xyz_to_cam`,
///   2. invert to get camera→XYZ,
///   3. left-multiply by XYZ→sRGB (the inverse of the D65 [`XYZ_RGB`] matrix),
///   4. normalise the rows of the *result* so they sum to 1.
///
/// Step 4 is the white-preservation step. An earlier version instead followed
/// dcraw's `cam_xyz_coeff` literally — normalising the rows of `xyz_to_cam·XYZ_RGB`
/// (the sRGB→cam matrix) *before* inverting. dcraw folds those per-row factors
/// into its white-balance premultipliers (`pre_mul`); we apply only the green-
/// normalised as-shot coeffs as WB, so skipping that left the final matrix with
/// rows that didn't sum to 1 — a residual colour cast (e.g. a green tint on
/// some Sony ARWs). Normalising the final matrix instead makes WB and matrix
/// self-consistent. See `avif::tests::raw_compare`.
fn camera_to_srgb_matrix(xyz_to_cam: [[f32; 3]; 4]) -> [[f64; 3]; 3] {
    // Top 3×3 of xyz_to_cam (drop the E/emerald row), normalised, then inverted.
    let xtc = [
        [xyz_to_cam[0][0] as f64, xyz_to_cam[0][1] as f64, xyz_to_cam[0][2] as f64],
        [xyz_to_cam[1][0] as f64, xyz_to_cam[1][1] as f64, xyz_to_cam[1][2] as f64],
        [xyz_to_cam[2][0] as f64, xyz_to_cam[2][1] as f64, xyz_to_cam[2][2] as f64],
    ];
    let cam_to_xyz = invert_3x3(normalize_rows(xtc));
    let xyz_to_srgb = invert_3x3(XYZ_RGB);

    // cam→sRGB = (XYZ→sRGB) · (cam→XYZ).
    let mut m = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += xyz_to_srgb[i][k] * cam_to_xyz[k][j];
            }
            m[i][j] = s;
        }
    }
    normalize_rows(m)
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
///
/// When the develop succeeds **and** the file carries an embedded preview, we
/// colour-match the full-resolution develop to that preview
/// ([`color_match_to_preview`]). Our generic develop can't reproduce the camera's
/// scene-adaptive colour science — e.g. mixed-lighting scenes (a sodium-lit
/// tunnel) come out with a hue cast the camera's own JPEG doesn't have — so we
/// fit a linear colour transform from develop→preview and apply it, keeping the
/// develop's full resolution while adopting the camera's correct colour.
fn decode_raw(path: &std::path::Path) -> Option<RgbaImage> {
    let bytes = std::fs::read(path).ok()?;
    if let Some(dev) = decode_raw_develop(&bytes) {
        if let Some(prev) = decode_raw_embedded_jpeg(&bytes) {
            if let Some(matched) = color_match_to_preview(&dev, &prev) {
                return Some(matched);
            }
        }
        return Some(dev);
    }
    decode_raw_embedded_jpeg(&bytes)
}

/// Recolour a full-resolution develop so its colours match the camera's embedded
/// JPEG preview, while keeping the develop's resolution and detail.
///
/// The preview is the camera's own (correct) rendering but is low-resolution on
/// some bodies (Sony ARW embeds only ~1.6 MP); the develop is full-resolution but
/// colour-approximate. We fit a single affine colour transform `[r,g,b,1]·M` that
/// best maps the (downscaled) develop onto the preview by least squares, then
/// apply that 4×3 `M` to every full-resolution develop pixel. A global linear map
/// can't capture tone-curve differences exactly, but it removes white-balance /
/// hue casts well, which is what the camera's colour science mainly differs by.
///
/// Returns `None` (caller keeps the plain develop) if the preview is empty or its
/// aspect ratio doesn't match the develop — the latter guards against an
/// orientation mismatch between the two, which would otherwise fit garbage.
fn color_match_to_preview(dev: &RgbaImage, prev: &RgbaImage) -> Option<RgbaImage> {
    let (dw, dh) = (dev.width(), dev.height());
    let (pw, ph) = (prev.width(), prev.height());
    if pw == 0 || ph == 0 || dw == 0 || dh == 0 {
        return None;
    }
    let (ra, rb) = (dw as f64 / dh as f64, pw as f64 / ph as f64);
    if (ra - rb).abs() > 0.02 * ra.max(rb) {
        return None;
    }

    // Fit at preview resolution: downscale the develop to match, then accumulate
    // the normal equations for the least-squares solve (Aᵀ·A and Aᵀ·b) over every
    // corresponding pixel pair, with A rows `[r,g,b,1]` and b the preview RGB.
    let small = image::imageops::resize(dev, pw, ph, image::imageops::FilterType::Triangle);
    let mut ata = [[0f64; 4]; 4];
    let mut atb = [[0f64; 3]; 4];
    for (sp, pp) in small.pixels().zip(prev.pixels()) {
        let v = [sp[0] as f64 / 255.0, sp[1] as f64 / 255.0, sp[2] as f64 / 255.0, 1.0];
        let t = [pp[0] as f64 / 255.0, pp[1] as f64 / 255.0, pp[2] as f64 / 255.0];
        for i in 0..4 {
            for j in 0..4 {
                ata[i][j] += v[i] * v[j];
            }
            for c in 0..3 {
                atb[i][c] += v[i] * t[c];
            }
        }
    }
    let m = solve4x3(ata, atb)?; // m[i][c]: contribution of channel i to output c

    let to_u8 = |x: f64| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut out = RgbaImage::new(dw, dh);
    for (o, d) in out.pixels_mut().zip(dev.pixels()) {
        let (r, g, b) = (d[0] as f64 / 255.0, d[1] as f64 / 255.0, d[2] as f64 / 255.0);
        let nr = r * m[0][0] + g * m[1][0] + b * m[2][0] + m[3][0];
        let ng = r * m[0][1] + g * m[1][1] + b * m[2][1] + m[3][1];
        let nb = r * m[0][2] + g * m[1][2] + b * m[2][2] + m[3][2];
        *o = image::Rgba([to_u8(nr), to_u8(ng), to_u8(nb), 255]);
    }
    Some(out)
}

/// Solve the 4×4 system `a · x = b` (b and x have 3 columns) by Gauss-Jordan
/// elimination with partial pivoting. Returns `None` if `a` is singular.
fn solve4x3(mut a: [[f64; 4]; 4], mut b: [[f64; 3]; 4]) -> Option<[[f64; 3]; 4]> {
    for col in 0..4 {
        // Partial pivot: move the row with the largest |value| in this column up.
        let mut piv = col;
        for r in (col + 1)..4 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        // Normalise the pivot row, then eliminate the column from all other rows.
        let d = a[col][col];
        for c in col..4 {
            a[col][c] /= d;
        }
        for c in 0..3 {
            b[col][c] /= d;
        }
        for r in 0..4 {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f != 0.0 {
                for c in col..4 {
                    a[r][c] -= f * a[col][c];
                }
                for c in 0..3 {
                    b[r][c] -= f * b[col][c];
                }
            }
        }
    }
    Some(b)
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
/// 1. **As-shot white balance.** We use the camera's recorded WB multipliers
///    (`wb_coeffs`, `[R, G, B, E]`) normalised by green, exactly as a reference
///    raw developer does. An earlier version used gray-world WB instead (forcing
///    each channel's mean to neutral) out of a concern that older WB coeffs were
///    unreliable — but gray-world strips the scene's actual colour cast (golden
///    hour goes drab, blue hour goes muddy, a warm sunset goes green), and on
///    every test raw (ARW/CR2/NEF/DNG) the as-shot coeffs match the camera's own
///    embedded preview far better. See `avif::tests::raw_compare`.
/// 2. **White-preserving camera→sRGB matrix** (see [`camera_to_srgb_matrix`]),
///    so a neutral (WB-balanced) pixel maps to neutral sRGB.
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

    // As-shot white balance: the camera's recorded multipliers normalised by
    // green (so green is left unchanged). `wb_coeffs` is `[R, G, B, E]`; the E
    // (emerald) channel can be NaN and is unused here. Fall back to no-op WB if
    // green is missing/zero so a malformed file can't divide by zero.
    let g = info.wb_coeffs[1] as f64;
    let wb = if g.abs() > 1e-9 {
        [info.wb_coeffs[0] as f64 / g, 1.0, info.wb_coeffs[2] as f64 / g]
    } else {
        [1.0, 1.0, 1.0]
    };

    let m = camera_to_srgb_matrix(info.color_matrix);

    // Apply WB, the camera→sRGB matrix, and the sRGB curve.
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

    // Compare our raw-develop output against the camera's own embedded JPEG
    // preview (the reference "correct" colours) for every raw sample. Saves both
    // as PNGs next to each source file and prints the mean RGB of each so colour
    // drift (e.g. a wrong white-balance cast) is visible numerically and by eye.
    // Skips cleanly when the local sample folders are absent.
    // cargo test --no-default-features --features avif raw_compare -- --nocapture
    #[test]
    fn raw_compare() {
        fn mean_rgb(img: &RgbaImage) -> (f64, f64, f64) {
            let (mut r, mut g, mut b, mut n) = (0f64, 0f64, 0f64, 0f64);
            for px in img.pixels() {
                r += px[0] as f64;
                g += px[1] as f64;
                b += px[2] as f64;
                n += 1.0;
            }
            (r / n, g / n, b / n)
        }
        let mut entries: Vec<std::path::PathBuf> = Vec::new();
        for (d, ext) in [("tests/DNG", "dng"), ("tests/ARW", "arw"), ("tests/CR2", "cr2"), ("tests/NEF", "nef")] {
            let dir = std::path::Path::new(d);
            if !dir.is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case(ext))
                    == Some(true)
                {
                    entries.push(p);
                }
            }
        }
        for p in entries {
            let bytes = std::fs::read(&p).unwrap();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let stem = p.with_extension("");
            let stem = stem.to_string_lossy().to_string();
            println!("\n=== {name} ===");
            let dev = decode_raw_develop(&bytes);
            let prev = decode_raw_embedded_jpeg(&bytes);
            if let Some(dev) = &dev {
                let (r, g, b) = mean_rgb(dev);
                println!("  develop  {}x{}  mean RGB = ({r:.1}, {g:.1}, {b:.1})", dev.width(), dev.height());
                dev.save(format!("{stem}.develop.png")).unwrap();
            } else {
                println!("  develop  FAILED");
            }
            if let Some(prev) = &prev {
                let (r, g, b) = mean_rgb(prev);
                println!("  preview  {}x{}  mean RGB = ({r:.1}, {g:.1}, {b:.1})", prev.width(), prev.height());
                prev.save(format!("{stem}.preview.png")).unwrap();
            } else {
                println!("  preview  FAILED");
            }
            // The colour-matched result is what the app actually displays (see
            // `decode_raw`): develop recoloured to the preview.
            if let (Some(dev), Some(prev)) = (&dev, &prev) {
                if let Some(matched) = color_match_to_preview(dev, prev) {
                    let (r, g, b) = mean_rgb(&matched);
                    println!("  matched  {}x{}  mean RGB = ({r:.1}, {g:.1}, {b:.1})", matched.width(), matched.height());
                    matched.save(format!("{stem}.matched.png")).unwrap();
                }
            }
        }
    }

    // Decode every raw sample (DNG + ARW + CR2) to a PNG so the result can be eyeballed.
    // Skips cleanly when the local sample folders are absent.
    // cargo test --no-default-features --features avif raw_smoke -- --nocapture
    #[test]
    fn raw_smoke() {
        for (dir, ext) in [("tests/DNG", "dng"), ("tests/ARW", "arw"), ("tests/CR2", "cr2"), ("tests/NEF", "nef")] {
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
