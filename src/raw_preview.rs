//! Recovering a viewable image from a TIFF/DNG-style container by pulling out the
//! camera-rendered JPEG it embeds.
//!
//! Camera-raw files (DNG/ARW/CR2/NEF) and some "raw" TIFFs (e.g. Lightroom output
//! that stores the rendered image JPEG-compressed inside a TIFF, with the raw CFA
//! data in a sub-IFD) can't be decoded by the `image` crate's TIFF reader — IFD0
//! is JPEG-compressed and may even omit the PhotometricInterpretation tag, which
//! makes the reader bail with "unknown photometric interpretation". But every such
//! file carries at least one self-contained camera JPEG (a thumbnail, and usually
//! a full-resolution preview), so we byte-scan for it and decode the largest one.
//!
//! Depends only on the `image` crate, so it's always compiled (unlike the
//! `avif`-gated raw *develop* pipeline in [`crate::avif`], which reuses this as its
//! embedded-preview fallback).

use image::RgbaImage;

/// Scan a raw/TIFF container for embedded JPEG previews and decode the largest one.
///
/// We byte-scan for JPEG start markers (`FF D8 FF`), decode each candidate, and
/// keep the one with the most pixels.
///
/// The previews are stored in *sensor* orientation; the rotation/flip needed to
/// view them upright lives in the container's TIFF `Orientation` tag (274), not
/// inside each preview JPEG — so we read that once and apply it (falling back to
/// the JPEG's own EXIF orientation if the container has none).
pub fn largest_embedded_jpeg(bytes: &[u8]) -> Option<RgbaImage> {
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

/// A camera raw's thumbnail, straight from the JPEG previews it embeds: the
/// smallest one whose long side reaches `max_edge`, decoded with the JPEG's own
/// DCT downscaling (1/2, 1/4, 1/8). So a 320 px tile reads a medium preview at
/// most — never develops the raw (seconds and gigabytes for a few hundred
/// pixels) nor decodes a full 24 MP preview. Orientation as in
/// [`largest_embedded_jpeg`]. `None` when no embedded JPEG is that large (a
/// tiny one would make a blurry tile: the caller develops the raw instead).
#[cfg_attr(not(feature = "avif"), allow(dead_code))] // raws are decoded only with avif
pub fn embedded_jpeg_for(bytes: &[u8], max_edge: u32) -> Option<RgbaImage> {
    use jpeg_decoder::{CodingProcess, Decoder};
    let want = max_edge.min(u16::MAX as u32) as u16;
    // Candidates by header only (cheap): (offset, width, height). Lossless
    // JPEG is a raw's sensor data (CR2, many DNGs), not a picture.
    let mut found: Vec<(usize, u16, u16)> = Vec::new();
    let mut attempts = 0u32;
    let mut i = 0usize;
    while i + 3 < bytes.len() {
        if bytes[i] == 0xFF && bytes[i + 1] == 0xD8 && bytes[i + 2] == 0xFF {
            let mut d = Decoder::new(std::io::Cursor::new(&bytes[i..]));
            if d.read_info().is_ok()
                && let Some(info) = d.info()
                && info.coding_process != CodingProcess::Lossless
                && info.width.max(info.height) >= want
            {
                found.push((i, info.width, info.height));
            }
            attempts += 1;
            if attempts >= 64 {
                break;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    // A size found again and again is a tile of the image data (a lossy
    // DNG stores its pixels as JPEG tiles), not a preview.
    let count = |w: u16, h: u16| found.iter().filter(|&&(_, fw, fh)| (fw, fh) == (w, h)).count();
    let mut previews: Vec<(usize, u64)> = found
        .iter()
        .filter(|&&(_, w, h)| count(w, h) < 3)
        .map(|&(at, w, h)| (at, w as u64 * h as u64))
        .collect();
    // Smallest first; a candidate that won't decode gives way to the next.
    previews.sort_by_key(|&(_, px)| px);
    let container_orientation = tiff_orientation(bytes);
    previews.into_iter().find_map(|(at, _)| decode_jpeg_scaled(&bytes[at..], want, container_orientation))
}

/// Decode a JPEG at the smallest DCT scale that still reaches `edge` on one
/// side, then apply orientation (the container's, else the JPEG's own EXIF).
#[cfg_attr(not(feature = "avif"), allow(dead_code))]
fn decode_jpeg_scaled(bytes: &[u8], edge: u16, container_orientation: Option<u8>) -> Option<RgbaImage> {
    use jpeg_decoder::PixelFormat;
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    d.scale(edge, edge).ok()?;
    let pixels = d.decode().ok()?;
    let info = d.info()?;
    let (w, h) = (info.width as u32, info.height as u32);
    let mut img = match info.pixel_format {
        PixelFormat::RGB24 => image::RgbImage::from_raw(w, h, pixels).map(image::DynamicImage::ImageRgb8),
        PixelFormat::L8 => image::GrayImage::from_raw(w, h, pixels).map(image::DynamicImage::ImageLuma8),
        _ => None,
    }?;
    let orientation = container_orientation
        .or_else(|| d.exif_data().and_then(tiff_orientation))
        .and_then(image::metadata::Orientation::from_exif)
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    img.apply_orientation(orientation);
    Some(img.into_rgba8())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jpeg(w: u32, h: u32) -> Vec<u8> {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([200, 40, 40])));
        let mut out = std::io::Cursor::new(Vec::new());
        img.write_to(&mut out, image::ImageFormat::Jpeg).unwrap();
        out.into_inner()
    }

    /// A thumbnail comes from the smallest embedded preview that's big
    /// enough, decoded scaled down; with none big enough, there's none.
    #[test]
    fn thumbnail_uses_the_smallest_big_enough_preview() {
        let mut raw = b"II*\0\x08\0\0\0\0\0junk".to_vec();
        raw.extend(jpeg(64, 48));
        raw.extend(b"sensor data");
        // Image data stored as JPEG tiles (a lossy DNG) isn't a preview.
        for _ in 0..4 {
            raw.extend(jpeg(256, 256));
        }
        raw.extend(jpeg(400, 300));
        raw.extend(b"more");
        raw.extend(jpeg(1600, 1200));
        let thumb = embedded_jpeg_for(&raw, 100).unwrap();
        assert_eq!(thumb.dimensions(), (100, 75)); // the 400 px one at 1/4
        let thumb = embedded_jpeg_for(&raw, 1000).unwrap();
        assert_eq!(thumb.dimensions(), (1600, 1200)); // only the big one reaches 1000
        assert!(embedded_jpeg_for(&raw, 2000).is_none());
        assert_eq!(largest_embedded_jpeg(&raw).unwrap().dimensions(), (1600, 1200));
    }
}
