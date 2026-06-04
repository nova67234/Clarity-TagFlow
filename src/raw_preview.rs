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
