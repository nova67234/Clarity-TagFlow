//! Reads embedded **Stable-Diffusion generation metadata** (the prompt / Steps /
//! Sampler / Seed block) from images, and formats it for display.
//!
//! This is a Rust port of terminus2's `RightDetailsPanel.java` metadata code
//! (`UniversalMetadataScanner`, `PngTextChunks`, `formatStableDiffusionMetadata`,
//! `formatComfyUIMetadata`), which in turn mirrors AUTOMATIC1111's
//! `read_info_from_image` (the `images.py` the user asked to convert). It reads
//! the same sources the Python/Java do:
//!
//! - **PNG**: the `tEXt` / `zTXt` (zlib) / `iTXt` text chunks — keys like
//!   `parameters` (A1111), `prompt` + `workflow` (ComfyUI), `Comment`,
//!   `Description`.
//! - **JPEG / WebP / AVIF / anything else**: a raw byte scan that decodes the
//!   file as UTF-8, then UTF-16LE, then UTF-16BE and isolates the parameter block
//!   around the `Steps:` (or `Civitai metadata:`) anchor. A1111 stores it in the
//!   EXIF `UserComment`, usually UTF-16 — the scan finds it without a full EXIF
//!   parser, exactly like the Java `UniversalMetadataScanner`.
//!
//! [`read_both`] returns both the cleaned, display-ready text and the raw original
//! metadata string (the latter for the Civitai panel) — or `None` when the image
//! has no SD metadata.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

/// Files larger than this are never byte-scanned for metadata. SD images are at
/// most a few MB; videos and other large files would otherwise be read fully into
/// memory and decoded as UTF-8/UTF-16, which froze (and could OOM-crash) the app.
const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB

/// Read the embedded metadata **once** and return both the formatted **display**
/// string (what [`read`] yields) and the **raw** original metadata string.
///
/// The Civitai panel needs the raw string: formatting splits the parameter line on
/// `", "`, which shreds blocks like `Hashes: {…}`, `Lora hashes:` and `TI hashes:`.
/// Embeddings (textual inversions) are recorded *only* in those blocks, so handing
/// the panel the formatted text left it unable to list them.
pub fn read_both(path: &Path) -> (Option<String>, Option<String>) {
    match metadata_map(path) {
        Some(map) => (format_from_map(&map), raw_from_map(&map)),
        None => (None, None),
    }
}

/// Read `path` and collect its embedded metadata into a keyword → text map (PNG
/// text chunks, or a raw byte-scan of the EXIF `UserComment` for JPEG/WebP/AVIF).
/// `None` only when the file can't be read; a readable file with no metadata
/// yields an empty (or prompt-less) map.
fn metadata_map(path: &Path) -> Option<BTreeMap<String, String>> {
    // Never touch videos — they carry no SD generation metadata, and reading a
    // multi-MB/GB file in full (then scanning it as UTF-8/UTF-16) freezes and can
    // crash the app. Bail before any read.
    if crate::is_video(path) {
        return None;
    }

    // Guard against pathologically large files (or anything mis-detected): only
    // read files small enough to scan safely.
    let mut file = crate::archive::open(path).ok()?;
    if file.len().ok()? > MAX_SCAN_BYTES {
        return None;
    }

    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut bytes).ok()?;

    let is_png = bytes.len() >= 8 && bytes[0..8] == [137, 80, 78, 71, 13, 10, 26, 10];

    // PNG: read the text chunks first (the richest, most reliable source).
    let mut text: BTreeMap<String, String> = BTreeMap::new();
    if is_png {
        text = read_png_text_chunks(&bytes);
    }

    // If the PNG had no useful key (or it isn't a PNG), fall back to the raw
    // byte scan — that's how JPEG/WebP/AVIF EXIF-UserComment metadata is found.
    if first_non_blank(&text, &["parameters", "prompt", "workflow", "Comment", "Description"]).is_none() {
        if let Some(scanned) = scan_raw_for_parameters(&bytes) {
            text.insert("parameters".to_string(), scanned);
        }
    }

    Some(text)
}

/// Pick the best metadata field from the map and format it for display.
fn format_from_map(text: &BTreeMap<String, String>) -> Option<String> {
    // Prefer the A1111 `parameters` block whenever it's a real param block (has
    // the `Steps:` anchor). Civitai/A1111 embed this clean, human-readable summary
    // — and many ComfyUI-saved images carry BOTH it AND the raw `prompt`/`workflow`
    // node graph. The graph can use custom Set/Get-node rerouting and custom
    // samplers that `format_comfyui` can't trace (yielding an empty/garbled result),
    // so the clean `parameters` block wins when present. (Mirrors A1111's own
    // `read_info_from_image`, which pops `parameters` before checking for ComfyUI.)
    if let Some(p) = text.get("parameters") {
        if has_steps_anchor(p) {
            let formatted = format_a1111(p);
            if !formatted.trim().is_empty() {
                return Some(formatted);
            }
        }
    }

    // ComfyUI workflows carry both `prompt` and `workflow`; format the node graph.
    if text.contains_key("prompt") && text.contains_key("workflow") {
        if let Some(formatted) = format_comfyui(text.get("prompt").unwrap()) {
            return Some(formatted);
        }
        // Fall through to showing the raw prompt JSON if we couldn't format it.
        return text.get("prompt").cloned().filter(|s| !s.trim().is_empty());
    }

    // A1111 / generic: format the first parameter-ish field we found.
    let raw = first_non_blank(text, &["parameters", "prompt", "workflow", "Comment", "Description"])?;
    let formatted = format_a1111(raw);
    if formatted.trim().is_empty() {
        None
    } else {
        Some(formatted)
    }
}

/// The **raw** original metadata string (no formatting), as the Civitai parser
/// expects it — preserving the `Hashes:` / `… hashes:` / `<lora:…>` blocks intact.
/// Mirrors [`format_from_map`]'s source-selection order so it returns the same
/// underlying field, just unformatted.
fn raw_from_map(text: &BTreeMap<String, String>) -> Option<String> {
    if let Some(p) = text.get("parameters") {
        if has_steps_anchor(p) {
            return Some(p.clone());
        }
    }
    if text.contains_key("prompt") && text.contains_key("workflow") {
        return text.get("prompt").cloned().filter(|s| !s.trim().is_empty());
    }
    first_non_blank(text, &["parameters", "prompt", "workflow", "Comment", "Description"])
        .map(|s| s.to_string())
}

/// Returns the first of `keys` whose value in `map` is non-blank.
fn first_non_blank<'a>(map: &'a BTreeMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    for k in keys {
        if let Some(v) = map.get(*k) {
            if !v.trim().is_empty() {
                return Some(v);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// PNG text chunks (tEXt / zTXt / iTXt)
// ---------------------------------------------------------------------------

/// Parse all PNG text chunks into a keyword → text map. Mirrors the Java
/// `PngTextChunks.readAllTextChunks`.
fn read_png_text_chunks(bytes: &[u8]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut pos = 8; // skip the 8-byte signature

    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start.checked_add(len).filter(|&e| e <= bytes.len());
        let Some(data_end) = data_end else { break };
        let data = &bytes[data_start..data_end];

        match kind {
            b"tEXt" => parse_text(data, &mut out),
            b"zTXt" => parse_ztext(data, &mut out),
            b"iTXt" => parse_itext(data, &mut out),
            b"IEND" => break,
            _ => {}
        }

        // advance past data + 4-byte CRC
        pos = data_end + 4;
    }
    out
}

/// `tEXt`: keyword \0 text  (both Latin-1).
fn parse_text(data: &[u8], out: &mut BTreeMap<String, String>) {
    let Some(nul) = data.iter().position(|&b| b == 0) else { return };
    if nul == 0 {
        return;
    }
    let keyword = latin1(&data[..nul]);
    let text = latin1(&data[nul + 1..]);
    out.insert(keyword, text);
}

/// `zTXt`: keyword \0 compression_method zlib(text).
fn parse_ztext(data: &[u8], out: &mut BTreeMap<String, String>) {
    let Some(nul) = data.iter().position(|&b| b == 0) else { return };
    if nul == 0 || nul + 2 > data.len() {
        return;
    }
    let keyword = latin1(&data[..nul]);
    if data[nul + 1] != 0 {
        return; // only deflate (method 0) is defined
    }
    if let Some(text) = inflate(&data[nul + 2..]) {
        out.insert(keyword, latin1(&text));
    }
}

/// `iTXt`: keyword \0 compFlag compMethod langTag \0 transKeyword \0 text(UTF-8).
fn parse_itext(data: &[u8], out: &mut BTreeMap<String, String>) {
    let Some(nul1) = data.iter().position(|&b| b == 0) else { return };
    if nul1 == 0 {
        return;
    }
    let keyword = latin1(&data[..nul1]);
    let mut p = nul1 + 1;
    if p + 2 > data.len() {
        return;
    }
    let comp_flag = data[p];
    let comp_method = data[p + 1];
    p += 2;
    // skip language tag
    let Some(rel2) = data[p..].iter().position(|&b| b == 0) else { return };
    p += rel2 + 1;
    // skip translated keyword
    let Some(rel3) = data[p..].iter().position(|&b| b == 0) else { return };
    p += rel3 + 1;
    if p > data.len() {
        return;
    }
    let raw = &data[p..];
    let text = if comp_flag == 1 && comp_method == 0 {
        inflate(raw).map(|b| String::from_utf8_lossy(&b).into_owned())
    } else {
        Some(String::from_utf8_lossy(raw).into_owned())
    };
    if let Some(text) = text {
        out.insert(keyword, text);
    }
}

/// Decode Latin-1 (ISO-8859-1) bytes — every byte maps to the same code point.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

/// zlib-inflate, returning `None` on failure.
fn inflate(compressed: &[u8]) -> Option<Vec<u8>> {
    let mut decoder = flate2::read::ZlibDecoder::new(compressed);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).ok()?;
    Some(out)
}

// ---------------------------------------------------------------------------
// Raw byte scan (JPEG / WebP / AVIF / fallback) — UniversalMetadataScanner
// ---------------------------------------------------------------------------

/// How far around an anchor hit the windowed decodes reach. The prompt text
/// precedes `Steps:`; the parameter line / Civitai JSON follows it. Real-world
/// blocks are well under 100 KiB, so these are very generous.
const SCAN_BEFORE: usize = 1024 * 1024; // 1 MiB
const SCAN_AFTER: usize = 4 * 1024 * 1024; // 4 MiB

/// Find the SD parameter block around the `Steps:` / `Civitai metadata:`
/// anchor, trying UTF-8, UTF-16LE and UTF-16BE. A1111 stores it in the EXIF
/// UserComment (commonly UTF-16), so the multi-encoding pass finds it without a
/// full EXIF parser.
///
/// Decoding the *whole* file in all three encodings (with `isolate_sd`'s
/// per-char work on top) allocated several times the file size and froze the
/// UI on hi-res images. Instead, each encoding's ANCHOR BYTES are searched
/// first — a metadata-less file is never decoded at all — and a hit decodes
/// only a generous window around the anchor.
fn scan_raw_for_parameters(bytes: &[u8]) -> Option<String> {
    // EXIF stores the A1111 block in `UserComment`, prefixed by an 8-byte charset
    // id ("UNICODE\0") and then UTF-16 text. Decoding the whole file as UTF-16
    // drags that marker in as mojibake right before the prompt (e.g. a leading
    // "啎䥃佄䔀"). So if the marker is present, decode only the text that follows it
    // — that lands exactly on the prompt with no marker garbage. EXIF's text byte
    // order varies, so try big- then little-endian. (A JPEG APP1 segment caps the
    // comment at 64 KiB; the SCAN_AFTER cap is just a defensive bound.)
    if let Some(pos) = find_bytes(bytes, b"UNICODE\0") {
        let after = &bytes[pos + 8..];
        let after = &after[..after.len().min(SCAN_AFTER)];
        if let Some(s) = isolate_sd(&decode_utf16(after, true)) {
            return Some(s);
        }
        if let Some(s) = isolate_sd(&decode_utf16(after, false)) {
            return Some(s);
        }
    }

    // UTF-8, then UTF-16LE, then UTF-16BE — same order as before, but each only
    // when its encoded anchor actually occurs, and decoding only a window.
    if let Some(pos) = anchor_pos(bytes, Encoding::Utf8) {
        if let Some(s) = isolate_sd(&String::from_utf8_lossy(anchor_window(bytes, pos, false))) {
            return Some(s);
        }
    }
    if let Some(pos) = anchor_pos(bytes, Encoding::Utf16Le) {
        if let Some(s) = isolate_sd(&decode_utf16(anchor_window(bytes, pos, true), false)) {
            return Some(s);
        }
    }
    if let Some(pos) = anchor_pos(bytes, Encoding::Utf16Be) {
        if let Some(s) = isolate_sd(&decode_utf16(anchor_window(bytes, pos, true), true)) {
            return Some(s);
        }
    }
    None
}

#[derive(Clone, Copy)]
enum Encoding {
    Utf8,
    Utf16Le,
    Utf16Be,
}

/// Byte position of the first `Steps:` / `Civitai metadata:` anchor in `enc`.
fn anchor_pos(bytes: &[u8], enc: Encoding) -> Option<usize> {
    for anchor in ["Steps:", "Civitai metadata:"] {
        let needle: Vec<u8> = match enc {
            Encoding::Utf8 => anchor.as_bytes().to_vec(),
            Encoding::Utf16Le => anchor.bytes().flat_map(|b| [b, 0]).collect(),
            Encoding::Utf16Be => anchor.bytes().flat_map(|b| [0, b]).collect(),
        };
        if let Some(pos) = find_bytes(bytes, &needle) {
            return Some(pos);
        }
    }
    None
}

/// The slice to decode around an anchor at `pos`: [`SCAN_BEFORE`] bytes back and
/// [`SCAN_AFTER`] forward. For UTF-16 the start keeps the anchor's 2-byte
/// alignment so the code-unit stream stays in phase.
fn anchor_window(bytes: &[u8], pos: usize, utf16: bool) -> &[u8] {
    let mut start = pos.saturating_sub(SCAN_BEFORE);
    if utf16 {
        start += (pos - start) % 2;
    }
    let end = (pos + SCAN_AFTER).min(bytes.len());
    &bytes[start..end]
}

/// Index of the first occurrence of `needle` in `haystack` (raw bytes).
/// memchr's memmem is SIMD-accelerated — the naive `windows().position()`
/// version took ~1 s per pass on a 14 MB image.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    memchr::memmem::find(haystack, needle)
}

fn decode_utf16(bytes: &[u8], big_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| if big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
        .collect();
    String::from_utf16_lossy(&units)
}

/// Find the `Steps:` (or `Civitai metadata:`) anchor and expand outward over
/// "valid" text characters. Mirrors Java `isolateSDMetadata`. Works directly on
/// the string slice — collecting a multi-MB decode into a `Vec<char>` first
/// allocated 4 bytes per char and was a large part of the hi-res freeze.
fn isolate_sd(text: &str) -> Option<String> {
    let anchor = text.find("Steps:").or_else(|| text.find("Civitai metadata:"))?;

    let mut start = anchor;
    for (i, c) in text[..anchor].char_indices().rev() {
        if is_valid_text(c) {
            start = i;
        } else {
            break;
        }
    }
    let mut end = anchor;
    for (i, c) in text[anchor..].char_indices() {
        if is_valid_text(c) {
            end = anchor + i + c.len_utf8();
        } else {
            break;
        }
    }

    let s = text[start..end].trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn is_valid_text(c: char) -> bool {
    if c == '\u{FFFD}' || c == '\u{0}' {
        return false;
    }
    c >= ' ' || c == '\n' || c == '\r' || c == '\t'
}

// ---------------------------------------------------------------------------
// A1111 formatting — formatStableDiffusionMetadata
// ---------------------------------------------------------------------------

/// Preferred key order for the parameter line, matching the Java version.
const A1111_ORDER: &[&str] = &[
    "Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size", "Model hash", "Model", "Clip skip", "Version",
];

fn format_a1111(raw: &str) -> String {
    let s = raw.trim().replace("\r\n", "\n").replace('\r', "\n");
    if s.is_empty() {
        return String::new();
    }

    // The parameter line begins at "\nSteps:". Without it, show the text as-is.
    let Some(idx_steps) = index_of_ignore_case(&s, "\nsteps:") else {
        return s;
    };
    let pre = s[..idx_steps].trim().to_string();
    let mut params_line = s[idx_steps + 1..].trim().to_string();

    // Pull any Civitai JSON tail off the end of the parameter line.
    let res_idx = params_line.find("Civitai resources:");
    let meta_idx = params_line.find("Civitai metadata:");
    let first_json = match (res_idx, meta_idx) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let mut json_part = String::new();
    if let Some(j) = first_json {
        json_part = params_line[j..]
            .replace('\n', "")
            .replace('\r', "")
            .replace("Civitai resources:", "Civitai resources:\n")
            .replace("Civitai metadata:", "\n\nCivitai metadata:\n")
            .trim()
            .to_string();
        // The byte-scan can capture a few stray characters past the final `]`/`}`
        // (image data decoded as text); clip the tail to the last JSON bracket.
        if let Some(cut) = json_part.rfind([']', '}']) {
            json_part.truncate(cut + 1);
        }
        params_line = params_line[..j].trim().to_string();
        if params_line.ends_with(',') {
            params_line.pop();
            params_line = params_line.trim().to_string();
        }
    }

    // Split the prompt from the negative prompt.
    let (prompt_text, negative_text) = split_negative(&pre);

    let kv = parse_params(&params_line);

    let mut out = String::new();
    if !prompt_text.is_empty() {
        out.push_str(&prompt_text);
        out.push_str("\n\n");
    }
    if !negative_text.is_empty() {
        out.push_str("Negative Prompt:\n");
        out.push_str(&negative_text);
        out.push_str("\n\n");
    }

    if !kv.is_empty() {
        out.push_str("Parameters:\n");
        let mut used: Vec<String> = Vec::new();
        for key in A1111_ORDER {
            if let Some((actual, val)) = kv.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)) {
                out.push_str(&format!("  \u{2022} {key}: {val}\n"));
                used.push(actual.clone());
            }
        }
        for (k, v) in &kv {
            if used.iter().any(|u| u == k) {
                continue;
            }
            out.push_str(&format!("  \u{2022} {k}: {v}\n"));
        }
    }

    if !json_part.is_empty() {
        out.push_str("\n\n");
        out.push_str(&json_part);
    }

    out.trim().to_string()
}

/// Split prompt vs "Negative prompt:" (newline-prefixed form preferred).
fn split_negative(pre: &str) -> (String, String) {
    if let Some(idx) = index_of_ignore_case(pre, "\nnegative prompt:") {
        let prompt = pre[..idx].trim().to_string();
        let neg = pre[idx + "\nNegative prompt:".len()..].trim();
        return (prompt, strip_leading_colon(neg));
    }
    if let Some(idx) = index_of_ignore_case(pre, "negative prompt:") {
        if idx > 0 {
            let prompt = pre[..idx].trim().to_string();
            let neg = pre[idx + "Negative prompt:".len()..].trim();
            return (prompt, strip_leading_colon(neg));
        }
    }
    (pre.trim().to_string(), String::new())
}

fn strip_leading_colon(s: &str) -> String {
    let s = s.trim();
    s.strip_prefix(':').unwrap_or(s).trim().to_string()
}

/// Parse the comma-separated "key: value" parameter line, preserving order.
fn parse_params(params: &str) -> Vec<(String, String)> {
    let s = params.replace('\n', " ");
    let mut out = Vec::new();
    for part in s.split(", ") {
        let part = part.trim();
        if let Some(colon) = part.find(':') {
            if colon == 0 {
                continue;
            }
            let key = part[..colon].trim().to_string();
            let val = part[colon + 1..].trim().to_string();
            if !key.is_empty() && !val.is_empty() {
                out.push((key, val));
            }
        }
    }
    out
}

/// Case-insensitive substring search returning a byte index into `haystack`.
fn index_of_ignore_case(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_lowercase().find(&needle.to_lowercase())
}

/// True when `raw` looks like a genuine A1111 parameter block (it carries the
/// `Steps:` key). Lets `read` distinguish a real `parameters` chunk from an empty
/// or prompt-only one before preferring it over a ComfyUI node graph.
fn has_steps_anchor(raw: &str) -> bool {
    index_of_ignore_case(raw, "steps:").is_some()
}

// ---------------------------------------------------------------------------
// ComfyUI formatting — formatComfyUIMetadata
// ---------------------------------------------------------------------------

/// Format a ComfyUI `prompt` node-graph JSON into a readable summary, walking the
/// graph the same way the Java `formatComfyUIMetadata` does.
fn format_comfyui(prompt_json: &str) -> Option<String> {
    let root: serde_json::Value = parse_comfy_json(prompt_json)?;
    let nodes = root.as_object()?;

    let mut positive = String::new();
    let mut negative = String::new();
    let (mut steps, mut sampler, mut scheduler, mut cfg, mut seed, mut size, mut model) =
        (None, None, None, None, None, None, None);
    let mut loras: Vec<String> = Vec::new();

    let s = |v: &serde_json::Value| -> Option<String> {
        match v {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    };

    for node in nodes.values() {
        let class_type = node.get("class_type").and_then(|v| v.as_str()).unwrap_or("");
        let inputs = node.get("inputs").cloned().unwrap_or(serde_json::Value::Null);

        match class_type {
            "CheckpointLoaderSimple" | "CheckpointLoader" => {
                if model.is_none() {
                    model = inputs.get("ckpt_name").and_then(s);
                }
            }
            "UNETLoader" => {
                if model.is_none() {
                    model = inputs.get("unet_name").and_then(s);
                }
            }
            "EmptyLatentImage" | "EmptySD3LatentImage" => {
                let w = inputs.get("width").and_then(|v| v.as_i64());
                let h = inputs.get("height").and_then(|v| v.as_i64());
                if let (Some(w), Some(h)) = (w, h) {
                    if w > 0 && h > 0 {
                        size = Some(format!("{w}x{h}"));
                    }
                }
            }
            "KSampler" | "KSamplerAdvanced" => {
                if seed.is_none() {
                    seed = inputs.get("seed").and_then(s).or_else(|| inputs.get("noise_seed").and_then(s));
                }
                if steps.is_none() {
                    steps = inputs.get("steps").and_then(s);
                }
                if cfg.is_none() {
                    cfg = inputs.get("cfg").and_then(s);
                }
                if sampler.is_none() {
                    sampler = inputs.get("sampler_name").and_then(s);
                }
                if scheduler.is_none() {
                    scheduler = inputs.get("scheduler").and_then(s);
                }
                if positive.is_empty() {
                    positive = trace_prompt(nodes, inputs.get("positive"));
                }
                if negative.is_empty() {
                    negative = trace_prompt(nodes, inputs.get("negative"));
                }
            }
            ct if ct.contains("LoraLoader") => {
                if let Some(name) = inputs.get("lora_name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        loras.push(name.to_string());
                    }
                }
            }
            "Power Lora Loader (rgthree)" => {
                if let Some(obj) = inputs.as_object() {
                    for (field, lnode) in obj {
                        if field.starts_with("lora_") && lnode.is_object() {
                            let on = lnode.get("on").and_then(|v| v.as_bool()).unwrap_or(true);
                            if on {
                                if let Some(name) = lnode.get("lora").and_then(|v| v.as_str()) {
                                    if !name.is_empty() {
                                        loras.push(name.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut out = String::from("ComfyUI Generation\n\n");
    if !positive.is_empty() {
        out.push_str(&positive);
        out.push_str("\n\n");
    }
    if !negative.is_empty() {
        out.push_str("Negative Prompt:\n");
        out.push_str(&negative);
        out.push_str("\n\n");
    }
    out.push_str("Parameters:\n");
    let mut line = |label: &str, val: &Option<String>| {
        if let Some(v) = val {
            out.push_str(&format!("  \u{2022} {label}: {v}\n"));
        }
    };
    line("Steps", &steps);
    line("Sampler", &sampler);
    line("Scheduler", &scheduler);
    line("CFG scale", &cfg);
    line("Seed", &seed);
    line("Size", &size);
    line("Model", &model);
    for lora in &loras {
        out.push_str(&format!("  \u{2022} LoRA: {lora}\n"));
    }

    Some(out.trim().to_string())
}

/// Follow a ComfyUI input link (`[node_id, slot]`) back to the text that feeds a
/// CLIP encode node. Mirrors the Java `tracePrompt`.
fn trace_prompt(nodes: &serde_json::Map<String, serde_json::Value>, link: Option<&serde_json::Value>) -> String {
    let Some(link) = link else { return String::new() };
    let Some(arr) = link.as_array() else { return String::new() };
    let Some(node_id) = arr.first().and_then(|v| v.as_str().map(|s| s.to_string()).or_else(|| v.as_i64().map(|n| n.to_string()))) else {
        return String::new();
    };
    let Some(node) = nodes.get(&node_id) else { return String::new() };

    let class_type = node.get("class_type").and_then(|v| v.as_str()).unwrap_or("");
    let inputs = node.get("inputs").cloned().unwrap_or(serde_json::Value::Null);

    if class_type == "CLIPTextEncode" || class_type == "CLIPTextEncodeSDXL" {
        if let Some(text) = inputs.get("text") {
            if let Some(t) = text.as_str() {
                return t.trim().to_string();
            }
            if text.is_array() {
                return trace_prompt(nodes, Some(text));
            }
        }
    }
    if class_type == "ConditioningZeroOut" {
        return String::new();
    }
    if let Some(cond) = inputs.get("conditioning") {
        return trace_prompt(nodes, Some(cond));
    }
    if let Some(tg) = inputs.get("text_g") {
        if let Some(t) = tg.as_str() {
            return t.trim().to_string();
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Structured extraction for the generator's drag-and-drop import — the prompt
// text, the core generation settings, and the ComfyUI custom nodes a workflow
// depends on (so they can be downloaded/installed on drop).
// ---------------------------------------------------------------------------

/// One custom node a dropped ComfyUI workflow depends on. Populated from the
/// `workflow` chunk's per-node `properties`: `aux_id` is the GitHub `owner/repo`
/// it was installed from, `cnr_id` its Comfy-registry id. Core ComfyUI nodes
/// carry neither, so collecting only the nodes that have one naturally yields
/// just the custom (downloadable) ones.
#[derive(Clone, Debug)]
pub struct CustomNodeRef {
    /// Display / install-folder name (the repo name, or the registry id).
    pub name: String,
    /// `owner/repo` when known (from `aux_id`) — directly clonable from GitHub.
    pub repo: Option<String>,
    /// Comfy-registry id (from `cnr_id`), resolved to a repo via the registry
    /// API when no `aux_id` is present.
    pub cnr_id: Option<String>,
}

/// A Civitai resource the image was made with, from the `Civitai resources:`
/// block an Image-Saver node writes (`urn:air:…:<kind>:civitai:<model>@<version>`).
/// Lets the importer resolve+download the exact model/LoRA files by version id.
#[derive(Clone, Debug)]
pub struct CivitaiRef {
    pub version_id: i64,
    /// AIR resource kind: `checkpoint`, `lora`, `vae`, … (drives the target folder).
    pub kind: String,
}

/// Everything a drag-and-drop import pulls out of a generated image/video: the
/// prompt text, the core settings, and the custom nodes the workflow used.
#[derive(Clone, Debug, Default)]
pub struct DroppedMeta {
    pub positive: String,
    pub negative: String,
    pub steps: Option<i64>,
    pub cfg: Option<f64>,
    pub seed: Option<i64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub custom_nodes: Vec<CustomNodeRef>,
    /// LoRAs the generation used: `(name, strength)`. `name` is as recorded
    /// (often with a `.safetensors` extension); the importer matches it against
    /// installed LoRAs by file stem and auto-selects them.
    pub loras: Vec<(String, f32)>,
    /// The **runnable** ComfyUI API graph (the `prompt` chunk) — the exact
    /// node graph that produced the image, custom nodes and all. When present,
    /// Generate can run *this* graph (with the prompt/seed swapped in) instead of
    /// the app's built-in pipeline, so the imported custom nodes are actually
    /// used. `None` for files that only carry a UI `workflow` graph or A1111 text.
    pub workflow_api: Option<String>,
    /// Civitai resources the image was generated with (checkpoint + LoRAs, by
    /// version id) — used to auto-download the missing model files.
    pub civitai: Vec<CivitaiRef>,
}

impl DroppedMeta {
    /// Nothing worth importing (no prompt, no settings, no nodes, no LoRAs).
    pub fn is_empty(&self) -> bool {
        self.positive.trim().is_empty()
            && self.negative.trim().is_empty()
            && self.steps.is_none()
            && self.cfg.is_none()
            && self.seed.is_none()
            && self.width.is_none()
            && self.height.is_none()
            && self.custom_nodes.is_empty()
            && self.loras.is_empty()
    }
}

/// Read a file's embedded generation metadata into a structured form for the
/// generator's drag-and-drop import. Handles both ComfyUI (`prompt` + `workflow`
/// node graphs) and A1111-style `parameters` blocks. `None` only when the file
/// can't be read or carries no recognisable metadata.
pub fn read_generation(path: &Path) -> Option<DroppedMeta> {
    // A dropped ComfyUI workflow / API-prompt `.json` (exported from ComfyUI),
    // not an image: parse it directly.
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
        return read_generation_json(path);
    }

    // Videos carry the workflow inside the container (Matroska/MP4 tags), not in
    // PNG/EXIF, and are too large for `metadata_map` (which refuses them). Scan a
    // bounded window of the file for the embedded ComfyUI `workflow` object.
    if crate::is_video(path) {
        return read_generation_video(path);
    }

    let map = metadata_map(path)?;
    let mut out = DroppedMeta::default();

    // ComfyUI: the API `prompt` graph gives the prompt + settings; the `workflow`
    // graph names the custom nodes it depends on.
    if let Some(prompt_json) = map.get("prompt").filter(|s| !s.trim().is_empty()) {
        comfy_prompt_into(prompt_json, &mut out);
        // The API graph is runnable — keep it (sanitised of NaN/Infinity so it
        // re-parses) so Generate can run this exact workflow (custom nodes
        // included) rather than the built-in pipeline.
        out.workflow_api = Some(sanitize_json(prompt_json));
    }
    if let Some(workflow_json) = map.get("workflow").filter(|s| !s.trim().is_empty()) {
        out.custom_nodes = comfy_custom_nodes(workflow_json);
        // If the API graph had no prompt (rare), recover it from the UI graph.
        if out.positive.trim().is_empty() {
            workflow_widgets_into(workflow_json, &mut out);
        }
    }

    // A1111: a `parameters` block with a `Steps:` anchor (no custom nodes).
    if out.positive.trim().is_empty() {
        if let Some(p) = map.get("parameters").filter(|p| has_steps_anchor(p)) {
            a1111_into(p, &mut out);
        }
    }

    // Civitai resources (checkpoint + LoRAs by version id) — written by the
    // Image-Saver node into the `parameters` block; independent of prompt-fill.
    if let Some(p) = map.get("parameters") {
        out.civitai = parse_civitai_resources(p);
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse the `Civitai resources: [ {…"air":"urn:air:…:<kind>:civitai:<m>@<v>"} ]`
/// block from an A1111 parameter string into resolvable `(version_id, kind)`s.
fn parse_civitai_resources(raw: &str) -> Vec<CivitaiRef> {
    let Some(i) = ci_find(raw, "civitai resources:") else { return Vec::new() };
    let after = &raw["civitai resources:".len() + i..];
    let Some(open) = after.find('[') else { return Vec::new() };
    let Some(arr_str) = balanced_span(after, open, b'[', b']') else { return Vec::new() };
    let Ok(arr) = serde_json::from_str::<serde_json::Value>(&arr_str) else { return Vec::new() };
    let Some(items) = arr.as_array() else { return Vec::new() };

    let mut out = Vec::new();
    for it in items {
        let Some(air) = it.get("air").and_then(|v| v.as_str()) else { continue };
        // urn:air:<eco>:<kind>:civitai:<modelId>@<versionId>
        let parts: Vec<&str> = air.split(':').collect();
        let kind = parts.get(3).copied().unwrap_or("").to_string();
        if let Some(tail) = air.rsplit("civitai:").next() {
            if let Some(ver) = tail.split('@').nth(1) {
                if let Ok(version_id) = ver.trim().parse::<i64>() {
                    out.push(CivitaiRef { version_id, kind });
                }
            }
        }
    }
    out
}

/// Balanced `open_char … close_char` slice starting at byte `open`, string-aware
/// (so brackets inside JSON strings don't unbalance it). `None` if never closed.
fn balanced_span(text: &str, open: usize, open_char: u8, close_char: u8) -> Option<String> {
    let bytes = text.as_bytes();
    let (mut depth, mut in_str, mut escaped) = (0i32, false, false);
    let mut i = open;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else if b == b'"' {
            in_str = true;
        } else if b == open_char {
            depth += 1;
        } else if b == close_char {
            depth -= 1;
            if depth == 0 {
                return text.get(open..=i).map(str::to_string);
            }
        }
        i += 1;
    }
    None
}

/// Import a dropped ComfyUI `.json` directly: a **UI workflow** export (has a
/// top-level `nodes` array → custom nodes + best-effort prompt/settings) or an
/// **API prompt** graph (a map of id → `{class_type, inputs}` → prompt/settings,
/// but no custom-node source info).
fn read_generation_json(path: &Path) -> Option<DroppedMeta> {
    let text = std::fs::read_to_string(path).ok()?;
    let v: serde_json::Value = parse_comfy_json(&text)?;
    let mut out = DroppedMeta::default();
    if v.get("nodes").and_then(|n| n.as_array()).is_some() {
        out.custom_nodes = comfy_custom_nodes(&text);
        workflow_widgets_into(&text, &mut out);
    } else if v.is_object() {
        comfy_prompt_into(&text, &mut out);
        // An API-format graph is runnable (sanitised so it re-parses cleanly).
        out.workflow_api = Some(sanitize_json(&text));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Bounded read of the head + tail of a (potentially huge) video file. ComfyUI
/// writes the workflow/prompt as container tags near the start or end, so a
/// capped window finds them without ever loading the whole clip into memory.
const VIDEO_HEAD: usize = 24 * 1024 * 1024;
const VIDEO_TAIL: usize = 8 * 1024 * 1024;

/// Drag-and-drop import for a generated **video**: pull the ComfyUI `workflow`
/// object out of the container and read its custom nodes + (best-effort) prompt
/// and settings from the nodes' `widgets_values`.
fn read_generation_video(path: &Path) -> Option<DroppedMeta> {
    use std::io::{Read, Seek, SeekFrom};
    let len = std::fs::metadata(path).ok()?.len();
    let mut file = std::fs::File::open(path).ok()?;

    let head_n = (len as usize).min(VIDEO_HEAD);
    let mut head = vec![0u8; head_n];
    file.read_exact(&mut head).ok()?;
    let mut text = String::from_utf8_lossy(&head).into_owned();
    if len as usize > VIDEO_HEAD {
        let tail_start = len.saturating_sub(VIDEO_TAIL as u64).max(VIDEO_HEAD as u64);
        if file.seek(SeekFrom::Start(tail_start)).is_ok() {
            let mut tail = vec![0u8; (len - tail_start) as usize];
            if file.read_exact(&mut tail).is_ok() {
                text.push_str(&String::from_utf8_lossy(&tail));
            }
        }
    }

    // The workflow object opens with `{"last_node_id": …}` — a key unique to the
    // top level — so the brace right before it is the object's start.
    let marker = ci_find(&text, "\"last_node_id\"")?;
    let open = text[..marker].rfind('{')?;
    let workflow_json = balanced_object(&text, open)?;

    let mut out = DroppedMeta::default();
    out.custom_nodes = comfy_custom_nodes(&workflow_json);
    workflow_widgets_into(&workflow_json, &mut out);
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Return the balanced `{ … }` slice starting at byte `open` (which must index a
/// `{`), tracking JSON string state + escapes so braces inside strings don't
/// throw off the depth count. `None` if the object never closes within `text`.
fn balanced_object(text: &str, open: usize) -> Option<String> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    let mut i = open;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else {
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return text.get(open..=i).map(str::to_string);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Best-effort prompt + settings from a ComfyUI **UI** (`workflow`) graph, read
/// from each node's positional `widgets_values`. Used for videos (and images
/// lacking an API graph), where the clean API `prompt` graph isn't available.
/// Widget layouts vary, so this is a heuristic: the positive prompt is taken as
/// the longest `CLIPTextEncode` text.
fn workflow_widgets_into(workflow_json: &str, out: &mut DroppedMeta) {
    let Some(root) = parse_comfy_json(workflow_json) else { return };
    let Some(nodes) = root.get("nodes").and_then(|v| v.as_array()) else { return };

    let num_i = |v: Option<&serde_json::Value>| -> Option<i64> {
        v.and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())))
    };
    let num_f = |v: Option<&serde_json::Value>| -> Option<f64> {
        v.and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())))
    };

    let mut prompts: Vec<String> = Vec::new();
    for node in nodes {
        let ty = node.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let wv = node.get("widgets_values").and_then(|v| v.as_array());
        match ty {
            "CLIPTextEncode" | "CLIPTextEncodeSDXL" => {
                if let Some(s) = wv.and_then(|a| a.first()).and_then(|v| v.as_str()) {
                    if !s.trim().is_empty() {
                        prompts.push(s.trim().to_string());
                    }
                }
            }
            // KSampler widgets: [seed, control_after_generate, steps, cfg, …].
            "KSampler" | "KSamplerAdvanced" => {
                if let Some(a) = wv {
                    if out.seed.is_none() {
                        out.seed = num_i(a.first());
                    }
                    if out.steps.is_none() {
                        out.steps = num_i(a.get(2));
                    }
                    if out.cfg.is_none() {
                        out.cfg = num_f(a.get(3));
                    }
                }
            }
            "EmptyLatentImage" | "EmptySD3LatentImage" => {
                if let Some(a) = wv {
                    if out.width.is_none() {
                        out.width = num_i(a.first()).filter(|&w| w > 0);
                    }
                    if out.height.is_none() {
                        out.height = num_i(a.get(1)).filter(|&h| h > 0);
                    }
                }
            }
            // LoraLoader UI widgets: [lora_name, strength_model, strength_clip].
            ty if ty.contains("LoraLoader") => {
                if let Some(a) = wv {
                    if let Some(name) = a.first().and_then(|v| v.as_str()) {
                        if !name.is_empty() {
                            let strength = num_f(a.get(1)).unwrap_or(1.0) as f32;
                            out.loras.push((name.to_string(), strength));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // Positive = the longest text-encode value; negative = the next longest.
    prompts.sort_by_key(|s| std::cmp::Reverse(s.len()));
    if out.positive.trim().is_empty() {
        if let Some(p) = prompts.first() {
            out.positive = p.clone();
        }
    }
    if out.negative.trim().is_empty() {
        if let Some(n) = prompts.get(1) {
            out.negative = n.clone();
        }
    }
}

/// Walk a ComfyUI API `prompt` graph, filling the prompt text + generation
/// settings into `out`. Mirrors [`format_comfyui`]'s node walk.
fn comfy_prompt_into(prompt_json: &str, out: &mut DroppedMeta) {
    let Some(root) = parse_comfy_json(prompt_json) else { return };
    let Some(nodes) = root.as_object() else { return };

    let as_i = |v: &serde_json::Value| -> Option<i64> {
        v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    };
    let as_f = |v: &serde_json::Value| -> Option<f64> {
        v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    };

    for node in nodes.values() {
        let class_type = node.get("class_type").and_then(|v| v.as_str()).unwrap_or("");
        let inputs = node.get("inputs").cloned().unwrap_or(serde_json::Value::Null);
        match class_type {
            "EmptyLatentImage" | "EmptySD3LatentImage" => {
                if out.width.is_none() {
                    out.width = inputs.get("width").and_then(&as_i).filter(|&w| w > 0);
                }
                if out.height.is_none() {
                    out.height = inputs.get("height").and_then(&as_i).filter(|&h| h > 0);
                }
            }
            "KSampler" | "KSamplerAdvanced" => {
                if out.seed.is_none() {
                    out.seed = inputs.get("seed").and_then(&as_i).or_else(|| inputs.get("noise_seed").and_then(&as_i));
                }
                if out.steps.is_none() {
                    out.steps = inputs.get("steps").and_then(&as_i);
                }
                if out.cfg.is_none() {
                    out.cfg = inputs.get("cfg").and_then(&as_f);
                }
                if out.positive.trim().is_empty() {
                    let p = trace_prompt(nodes, inputs.get("positive"));
                    if !p.is_empty() {
                        out.positive = p;
                    }
                }
                if out.negative.trim().is_empty() {
                    let n = trace_prompt(nodes, inputs.get("negative"));
                    if !n.is_empty() {
                        out.negative = n;
                    }
                }
            }
            ct if ct.contains("LoraLoader") => {
                if let Some(name) = inputs.get("lora_name").and_then(|v| v.as_str()) {
                    if !name.is_empty() {
                        let strength = inputs.get("strength_model").and_then(&as_f).unwrap_or(1.0) as f32;
                        out.loras.push((name.to_string(), strength));
                    }
                }
            }
            "Power Lora Loader (rgthree)" => {
                if let Some(obj) = inputs.as_object() {
                    for (field, lnode) in obj {
                        if field.starts_with("lora_") && lnode.is_object() {
                            let on = lnode.get("on").and_then(|v| v.as_bool()).unwrap_or(true);
                            if on {
                                if let Some(name) = lnode.get("lora").and_then(|v| v.as_str()) {
                                    if !name.is_empty() {
                                        let strength = lnode.get("strength").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                                        out.loras.push((name.to_string(), strength));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Collect the custom nodes a ComfyUI `workflow` (UI) graph depends on, from each
/// node's `properties.aux_id` / `properties.cnr_id`. Core ComfyUI nodes carry
/// neither (or `cnr_id == "comfy-core"`), so they're skipped — only the
/// installable custom nodes come back, de-duplicated.
fn comfy_custom_nodes(workflow_json: &str) -> Vec<CustomNodeRef> {
    let Some(root) = parse_comfy_json(workflow_json) else { return Vec::new() };
    let Some(nodes) = root.get("nodes").and_then(|v| v.as_array()) else { return Vec::new() };

    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for node in nodes {
        let Some(props) = node.get("properties") else { continue };
        let aux = props
            .get("aux_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty() && s.contains('/') && s != "comfyanonymous/ComfyUI");
        let cnr = props
            .get("cnr_id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty() && s != "comfy-core");
        if aux.is_none() && cnr.is_none() {
            continue;
        }
        // De-dupe by repo when known, else by registry id.
        let key = aux.clone().or_else(|| cnr.clone()).unwrap();
        if !seen.insert(key) {
            continue;
        }
        let name = match (&aux, &cnr) {
            (Some(repo), _) => repo.rsplit('/').next().unwrap_or(repo).to_string(),
            (None, Some(id)) => id.clone(),
            _ => continue,
        };
        out.push(CustomNodeRef { name, repo: aux, cnr_id: cnr });
    }
    out
}

/// Parse an A1111 `parameters` block into the prompt + negative + settings +
/// LoRAs. The positive prompt's `<lora:name:weight>` tags are pulled into
/// `out.loras` and stripped from the text (the app applies LoRAs via node
/// selection, not inline tags).
fn a1111_into(raw: &str, out: &mut DroppedMeta) {
    const NEG_KEY: &str = "\nnegative prompt:";
    let neg_idx = ci_find(raw, NEG_KEY);
    let steps_idx = ci_find(raw, "steps:");

    // Positive = everything before the "Negative prompt:" line or the "Steps:"
    // line, whichever comes first.
    let pos_end = [neg_idx, steps_idx].into_iter().flatten().min();
    let pos = match pos_end {
        Some(i) => &raw[..i],
        None => raw,
    };
    out.positive = extract_and_strip_loras(pos.trim(), out);

    // Negative = between the "Negative prompt:" line and the "Steps:" line.
    if let Some(ni) = neg_idx {
        let after = &raw[ni + NEG_KEY.len()..];
        let neg = match ci_find(after, "steps:") {
            Some(j) => &after[..j],
            None => after,
        };
        out.negative = neg.trim().to_string();
    }

    out.steps = a1111_field(raw, "Steps:").and_then(|s| s.parse().ok());
    out.cfg = a1111_field(raw, "CFG scale:").and_then(|s| s.parse().ok());
    out.seed = a1111_field(raw, "Seed:").and_then(|s| s.parse().ok());
    if let Some(size) = a1111_field(raw, "Size:") {
        if let Some((w, h)) = size.split_once('x') {
            out.width = w.trim().parse().ok();
            out.height = h.trim().parse().ok();
        }
    }
}

/// Pull `<lora:name:weight>` tags out of `text` into `out.loras` and return the
/// text with those tags removed (A1111 inline-LoRA syntax; the app selects the
/// matching installed LoRAs instead of passing the tags to the encoder).
fn extract_and_strip_loras(text: &str, out: &mut DroppedMeta) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<lora:") {
        result.push_str(&rest[..start]);
        let after = &rest[start + "<lora:".len()..];
        let Some(end) = after.find('>') else {
            // Unterminated tag — keep the remainder verbatim and stop.
            result.push_str(&rest[start..]);
            return result.trim().to_string();
        };
        let inner = &after[..end];
        let (name, weight) = match inner.rsplit_once(':') {
            Some((n, w)) => (n.trim(), w.trim().parse::<f32>().unwrap_or(1.0)),
            None => (inner.trim(), 1.0),
        };
        if !name.is_empty() {
            out.loras.push((name.to_string(), weight));
        }
        rest = &after[end + 1..];
    }
    result.push_str(rest);
    result.trim().to_string()
}

/// Read a single A1111 key's value (`Steps: 30, ...`) — the text after the key up
/// to the next comma or newline.
fn a1111_field(raw: &str, key: &str) -> Option<String> {
    let i = ci_find(raw, key)?;
    let after = &raw[i + key.len()..];
    let val: String = after.trim_start().chars().take_while(|&c| c != ',' && c != '\n').collect();
    let val = val.trim().to_string();
    if val.is_empty() {
        None
    } else {
        Some(val)
    }
}

/// Case-insensitive substring search that returns a **byte index valid in the
/// original** `haystack`. Unlike [`index_of_ignore_case`] it only folds ASCII, so
/// the index never lands mid-multibyte-character (safe for slicing prompts that
/// contain Unicode).
fn ci_find(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_ascii_lowercase().find(&needle.to_ascii_lowercase())
}

/// Parse a ComfyUI graph string, tolerating the non-standard `NaN` / `Infinity`
/// literals that Python's `json.dumps` (ComfyUI's writer) emits but `serde_json`
/// rejects. Returns `None` only on genuinely broken JSON.
pub(crate) fn parse_comfy_json(s: &str) -> Option<serde_json::Value> {
    serde_json::from_str(s).ok().or_else(|| serde_json::from_str(&sanitize_json(s)).ok())
}

/// Replace the bare `NaN`, `Infinity`, `-Infinity` value literals (illegal in
/// strict JSON) with `null`, but only **outside** string literals so prompt text
/// containing those words is untouched. ComfyUI emits these for fields like a
/// node's `is_changed`; `null` parses and the executor ignores the field anyway.
pub(crate) fn sanitize_json(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut escaped = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            i += 1;
            continue;
        }
        // Outside a string, JSON has no bare identifiers except true/false/null
        // and numbers — so these tokens are always the illegal literals.
        let matches = |lit: &str| chars[i..].iter().take(lit.len()).copied().eq(lit.chars());
        if matches("-Infinity") {
            out.push_str("null");
            i += "-Infinity".len();
        } else if matches("Infinity") {
            out.push_str("null");
            i += "Infinity".len();
        } else if matches("NaN") {
            out.push_str("null");
            i += "NaN".len();
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod drop_import_tests {
    use super::*;

    /// The committed sample image under `tests/example/` (whatever it is) must
    /// parse into a prompt + a runnable API graph for drag-drop import. Found
    /// dynamically so swapping the sample image doesn't break the test.
    pub(crate) fn example_png() -> Option<std::path::PathBuf> {
        std::fs::read_dir("tests/example")
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
    }

    #[test]
    fn reads_example_png() {
        let Some(p) = example_png() else { return };
        let meta = read_generation(&p).expect("example png should have metadata");
        assert!(!meta.positive.trim().is_empty(), "a prompt was extracted");
        assert!(meta.workflow_api.is_some(), "runnable API graph captured");
        // Inline <lora:> tags must never survive in the prompt text.
        assert!(!meta.positive.contains("<lora:"), "lora tags stripped");
    }

    #[test]
    fn parses_civitai_resources() {
        let Some(p) = example_png() else { return };
        let meta = read_generation(&p).expect("metadata");
        // Skip gracefully if the committed sample has no Civitai resources block.
        if meta.civitai.is_empty() {
            return;
        }
        assert!(meta.civitai.iter().all(|c| c.version_id > 0), "version ids parsed");
        assert!(
            meta.civitai.iter().any(|c| c.kind == "checkpoint"),
            "a checkpoint resource present"
        );
    }

    #[test]
    fn strips_inline_lora_tags() {
        let mut m = DroppedMeta::default();
        let out = extract_and_strip_loras("a cat <lora:Foo:0.8> on a mat <lora:Bar Baz:0.5>", &mut m);
        assert_eq!(out, "a cat  on a mat");
        assert_eq!(m.loras, vec![("Foo".to_string(), 0.8), ("Bar Baz".to_string(), 0.5)]);
    }
}
