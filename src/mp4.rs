//! Minimal ISO base-media file format reader (MP4 / MOV / M4V) — just enough to
//! surface a video's resolution, duration and codec in the details card without
//! pulling in a full demuxer.
//!
//! It reads only the small `moov` header box (seeking past the large `mdat`
//! payload), so probing a multi-gigabyte clip touches just a few KB. Containers
//! that aren't ISO-BMFF (MKV / WebM / AVI) simply return `None`.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// The video facts we can cheaply extract. Any field may be unknown.
pub struct VideoInfo {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_secs: Option<f64>,
    pub codec: Option<String>,
}

/// Largest `moov` box we'll read into memory — a guard against absurd headers.
const MAX_MOOV: u64 = 64 * 1024 * 1024;

/// Probe an MP4/MOV/M4V file. Returns `None` if it isn't ISO-BMFF or has no
/// readable `moov`.
pub fn probe(path: &Path) -> Option<VideoInfo> {
    let mut f = crate::archive::open(path).ok()?;
    let file_len = f.len().ok()?;

    let (moov_off, moov_total) = find_top_level_box(&mut f, file_len, b"moov")?;
    if !(8..=MAX_MOOV).contains(&moov_total) {
        return None;
    }
    f.seek(SeekFrom::Start(moov_off)).ok()?;
    let mut buf = vec![0u8; moov_total as usize];
    f.read_exact(&mut buf).ok()?;

    let payload = box_payload(&buf)?;
    let children = parse_boxes(payload);

    // Movie-level duration (from mvhd).
    let mut duration_secs = None;
    for (typ, body) in &children {
        if typ == b"mvhd" {
            duration_secs = parse_mvhd_duration(body);
        }
    }

    // Locate the video track and pull its dimensions + codec.
    let (mut width, mut height, mut codec) = (None, None, None);
    for (typ, body) in &children {
        if typ != b"trak" {
            continue;
        }
        let trak_children = parse_boxes(body);
        let mut is_video = false;
        let (mut t_w, mut t_h, mut t_codec) = (None, None, None);

        for (ttyp, tbody) in &trak_children {
            match ttyp {
                b"tkhd" => {
                    if let Some((w, h)) = parse_tkhd_dims(tbody)
                        && w > 0 && h > 0 {
                            t_w = Some(w);
                            t_h = Some(h);
                        }
                }
                b"mdia" => {
                    for (mtyp, mbody) in &parse_boxes(tbody) {
                        match mtyp {
                            b"hdlr" if handler_is_video(mbody) => is_video = true,
                            b"minf" => t_codec = find_codec(mbody),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // Prefer a track explicitly marked as video; otherwise accept any track
        // that reported real dimensions (covers odd files with no/late hdlr).
        if is_video || t_w.is_some() {
            width = t_w;
            height = t_h;
            codec = t_codec;
            if is_video {
                break;
            }
        }
    }

    Some(VideoInfo {
        width,
        height,
        duration_secs,
        codec,
    })
}

/// Seek through the top-level boxes (reading only their headers) and return the
/// `(absolute offset, total size)` of the first box of `target` type.
fn find_top_level_box(
    f: &mut crate::archive::Reader,
    file_len: u64,
    target: &[u8; 4],
) -> Option<(u64, u64)> {
    let mut pos = 0u64;
    // Saturating: a corrupt box size can push `pos` to near u64::MAX.
    while pos.saturating_add(8) <= file_len {
        f.seek(SeekFrom::Start(pos)).ok()?;
        let mut hdr = [0u8; 8];
        f.read_exact(&mut hdr).ok()?;
        let size32 = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as u64;
        let typ = &hdr[4..8];

        let total = if size32 == 1 {
            let mut ext = [0u8; 8];
            f.read_exact(&mut ext).ok()?;
            u64::from_be_bytes(ext)
        } else if size32 == 0 {
            file_len - pos // box runs to end of file
        } else {
            size32
        };
        if total < 8 {
            return None;
        }
        if typ == target {
            return Some((pos, total));
        }
        pos = pos.checked_add(total)?;
    }
    None
}

/// The children bytes of a box buffer that still includes its own header.
fn box_payload(buf: &[u8]) -> Option<&[u8]> {
    if buf.len() < 8 {
        return None;
    }
    let size32 = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let header = if size32 == 1 { 16 } else { 8 };
    buf.get(header..)
}

/// Parse the immediate child boxes within a payload slice into `(type, body)`.
fn parse_boxes(buf: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while p + 8 <= buf.len() {
        let size32 = u32::from_be_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as usize;
        let typ = [buf[p + 4], buf[p + 5], buf[p + 6], buf[p + 7]];

        let (header, total) = if size32 == 1 {
            if p + 16 > buf.len() {
                break;
            }
            let s = u64::from_be_bytes(buf[p + 8..p + 16].try_into().ok().unwrap());
            (16usize, usize::try_from(s).unwrap_or(usize::MAX))
        } else if size32 == 0 {
            (8usize, buf.len() - p)
        } else {
            (8usize, size32)
        };

        // Compare against the bytes left rather than computing `p + total`: a
        // corrupt 64-bit size near u64::MAX wrapped that sum past the check in
        // release builds, and the slice below then panicked.
        if total < header || total > buf.len() - p {
            break;
        }
        out.push((typ, &buf[p + header..p + total]));
        p += total;
    }
    out
}

/// `mvhd` → duration in seconds (`duration / timescale`). Handles v0 and v1.
fn parse_mvhd_duration(body: &[u8]) -> Option<f64> {
    let version = *body.first()?;
    let (timescale, duration) = if version == 1 {
        // v1: 1 ver + 3 flags + 8 created + 8 modified, then timescale, duration.
        let ts = u32::from_be_bytes(body.get(20..24)?.try_into().ok()?);
        let du = u64::from_be_bytes(body.get(24..32)?.try_into().ok()?);
        (ts, du as f64)
    } else {
        // v0: 1 ver + 3 flags + 4 created + 4 modified, then timescale, duration.
        let ts = u32::from_be_bytes(body.get(12..16)?.try_into().ok()?);
        let du = u32::from_be_bytes(body.get(16..20)?.try_into().ok()?);
        (ts, du as f64)
    };
    if timescale == 0 {
        return None;
    }
    Some(duration / timescale as f64)
}

/// `tkhd` → display dimensions. Width/height are the final two 16.16 fixed-point
/// fields, regardless of box version.
fn parse_tkhd_dims(body: &[u8]) -> Option<(u32, u32)> {
    let len = body.len();
    if len < 8 {
        return None;
    }
    let w = u32::from_be_bytes(body.get(len - 8..len - 4)?.try_into().ok()?) >> 16;
    let h = u32::from_be_bytes(body.get(len - 4..len)?.try_into().ok()?) >> 16;
    Some((w, h))
}

/// `hdlr` handler type (offset 8..12) equals `vide` for a video track.
fn handler_is_video(body: &[u8]) -> bool {
    body.get(8..12).map(|h| h == b"vide").unwrap_or(false)
}

/// Walk `minf → stbl → stsd` and return the first sample entry's codec name.
fn find_codec(minf_body: &[u8]) -> Option<String> {
    let minf = parse_boxes(minf_body);
    let stbl = minf.iter().find(|(t, _)| t == b"stbl")?.1;
    let stbl = parse_boxes(stbl);
    let stsd = stbl.iter().find(|(t, _)| t == b"stsd")?.1;
    // stsd: 1 ver + 3 flags + 4 entry_count, then the first sample entry box:
    // 4 size + 4 format(fourcc).
    let fourcc = stsd.get(12..16)?;
    Some(pretty_codec(fourcc))
}

/// Map a sample-entry fourcc to a friendly name, keeping the raw code in parens.
fn pretty_codec(fourcc: &[u8]) -> String {
    let raw: String = fourcc
        .iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
        .collect();
    let pretty = match raw.as_str() {
        "avc1" | "avc3" => "H.264",
        "hev1" | "hvc1" => "H.265 (HEVC)",
        "av01" => "AV1",
        "vp09" => "VP9",
        "vp08" => "VP8",
        "mp4v" => "MPEG-4",
        "ap4h" | "apch" | "apcn" | "apcs" | "apco" => "ProRes",
        _ => return raw,
    };
    format!("{pretty} ({raw})")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box with a plain 32-bit size.
    fn bx(typ: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(typ);
        v.extend_from_slice(body);
        v
    }

    /// A box claiming the 64-bit size `size` (`size32 == 1`), whatever its real length.
    fn bx64(typ: &[u8; 4], size: u64, body: &[u8]) -> Vec<u8> {
        let mut v = 1u32.to_be_bytes().to_vec();
        v.extend_from_slice(typ);
        v.extend_from_slice(&size.to_be_bytes());
        v.extend_from_slice(body);
        v
    }

    /// `mvhd` v0 with timescale 1000 and duration 5000 (5 s).
    fn mvhd() -> Vec<u8> {
        let mut body = vec![0u8; 20];
        body[12..16].copy_from_slice(&1000u32.to_be_bytes());
        body[16..20].copy_from_slice(&5000u32.to_be_bytes());
        bx(b"mvhd", &body)
    }

    // A 64-bit size near u64::MAX made `p + total` wrap past the bounds check in
    // release builds (overflow in debug), then the slice panicked: such a box
    // must just end the parse.
    #[test]
    fn huge_64bit_child_size_is_skipped() {
        for size in [u64::MAX, u64::MAX - 7, u64::MAX - 15, 1 << 40] {
            let mut buf = bx(b"free", &[0; 4]);
            buf.extend(bx64(b"trak", size, &[0; 8]));
            let boxes = parse_boxes(&buf);
            assert_eq!(boxes.len(), 1, "size {size:#x}");
            assert_eq!(&boxes[0].0, b"free");
        }
    }

    // End to end through `probe`: a bad box inside `moov` (and inside a track)
    // is skipped and the rest still reads; a bad top-level size gives `None`.
    #[test]
    fn probe_survives_huge_box_sizes() {
        let dir = std::env::temp_dir().join(format!("ctf_mp4_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut moov_body = mvhd();
        moov_body.extend(bx(b"trak", &bx64(b"tkhd", u64::MAX - 7, &[0; 8])));
        moov_body.extend(bx64(b"trak", u64::MAX - 7, &[0; 8]));
        let mut file = bx(b"ftyp", b"isom");
        file.extend(bx(b"moov", &moov_body));
        let ok = dir.join("bad_child.mp4");
        std::fs::write(&ok, &file).unwrap();
        let info = probe(&ok).expect("moov should still parse");
        assert_eq!(info.duration_secs, Some(5.0));
        assert_eq!((info.width, info.height), (None, None));

        let mut top = bx64(b"ftyp", u64::MAX - 3, &[]);
        top.extend(bx(b"moov", &mvhd()));
        let bad = dir.join("bad_top.mp4");
        std::fs::write(&bad, &top).unwrap();
        assert!(probe(&bad).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
