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

/// Decode an AVIF/HEIF file to an RGBA image. Returns `None` on any failure so
/// the caller falls back to the usual "couldn't load" placeholder.
pub fn decode_avif(path: &std::path::Path) -> Option<RgbaImage> {
    let bytes = std::fs::read(path).ok()?;
    let parsed = avif_parse::read_avif(&mut bytes.as_slice()).ok()?;

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
}
