//! Minimal GIF reader — just enough to surface a GIF's frame count and total
//! animation duration in the details card, without decoding any pixels.
//!
//! It walks the GIF block structure (graphic-control extensions + image
//! descriptors), skipping all sub-block payloads, so probing is cheap even for a
//! large animation. Dimensions come from the regular image path; this only adds
//! the animation facts.

use std::io::Read;
use std::path::Path;

/// Animation facts cheaply extracted from a GIF.
pub struct GifInfo {
    /// Number of image frames (1 for a non-animated GIF).
    pub frames: u32,
    /// Total play time in seconds, summing per-frame delays. `None` if the GIF
    /// has no usable delays (e.g. a single static frame).
    pub duration_secs: Option<f64>,
}

/// Don't read GIFs larger than this into memory (GIFs this big are pathological).
const MAX_GIF: usize = 256 * 1024 * 1024; // 256 MiB

/// Probe a GIF for its frame count and duration. Returns `None` if the file
/// isn't a GIF or is malformed.
pub fn probe(path: &Path) -> Option<GifInfo> {
    let mut f = crate::archive::open(path).ok()?;
    let len = f.len().ok()? as usize;
    if len > MAX_GIF {
        return None;
    }
    let mut buf = Vec::with_capacity(len.min(MAX_GIF));
    f.take(MAX_GIF as u64).read_to_end(&mut buf).ok()?;

    // Header: "GIF87a" or "GIF89a".
    if buf.len() < 13 || &buf[0..3] != b"GIF" {
        return None;
    }

    let mut p = 6usize; // past the 6-byte signature

    // Logical Screen Descriptor (7 bytes). The packed byte tells us whether a
    // Global Color Table follows, and how big it is.
    let packed = buf[p + 4];
    p += 7;
    if packed & 0x80 != 0 {
        let gct_entries = 1usize << ((packed & 0x07) + 1);
        p += gct_entries * 3;
    }

    let mut frames: u32 = 0;
    let mut total_cs: u64 = 0; // total delay in centiseconds (1/100 s)
    // Delay (centiseconds) parsed from the most recent Graphic Control Extension,
    // applied to the next image descriptor.
    let mut pending_delay: u16 = 0;

    while p < buf.len() {
        match buf[p] {
            // Extension introducer.
            0x21 => {
                p += 1;
                if p >= buf.len() {
                    break;
                }
                let label = buf[p];
                p += 1;
                if label == 0xF9 {
                    // Graphic Control Extension: block size (4), packed, delay
                    // (2, little-endian), transparent index, then a 0 terminator.
                    if p + 1 + 4 > buf.len() {
                        break;
                    }
                    let block_size = buf[p] as usize; // always 4
                    if block_size >= 3 {
                        pending_delay = u16::from_le_bytes([buf[p + 2], buf[p + 3]]);
                    }
                    p = skip_sub_blocks(&buf, p)?;
                } else {
                    // Other extension (application / comment / plain-text): skip
                    // its sub-blocks.
                    p = skip_sub_blocks(&buf, p)?;
                }
            }
            // Image descriptor — one per frame.
            0x2C => {
                frames += 1;
                // GIF players treat a 0 (or 1) delay as "go as fast as possible"
                // and clamp it to ~100 ms. Mirror that so the reported duration
                // matches what the user actually sees.
                let eff = if pending_delay < 2 { 10 } else { pending_delay };
                total_cs += eff as u64;
                pending_delay = 0;

                // 9-byte descriptor: left, top, width, height (2 each) + packed.
                if p + 10 > buf.len() {
                    break;
                }
                let img_packed = buf[p + 9];
                p += 10;
                // Optional Local Color Table.
                if img_packed & 0x80 != 0 {
                    let lct_entries = 1usize << ((img_packed & 0x07) + 1);
                    p += lct_entries * 3;
                }
                // LZW minimum code size byte, then image-data sub-blocks.
                if p >= buf.len() {
                    break;
                }
                p += 1;
                p = skip_sub_blocks(&buf, p)?;
            }
            // Trailer — end of GIF.
            0x3B => break,
            // Unknown byte: bail rather than spin.
            _ => break,
        }
    }

    if frames == 0 {
        return None;
    }
    let duration_secs = if frames > 1 && total_cs > 0 {
        Some(total_cs as f64 / 100.0)
    } else {
        None
    };
    Some(GifInfo {
        frames,
        duration_secs,
    })
}

/// Advance `p` (positioned at a sub-block size byte) past a chain of length-
/// prefixed data sub-blocks, ending just after the 0 terminator.
fn skip_sub_blocks(buf: &[u8], mut p: usize) -> Option<usize> {
    while p < buf.len() {
        let size = buf[p] as usize;
        p += 1;
        if size == 0 {
            return Some(p); // block terminator
        }
        p = p.checked_add(size)?;
    }
    None
}
