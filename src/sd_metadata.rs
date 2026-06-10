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
    match std::fs::metadata(path) {
        Ok(m) if m.len() > MAX_SCAN_BYTES => return None,
        Ok(_) => {}
        Err(_) => return None,
    }

    let bytes = std::fs::read(path).ok()?;

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

/// Decode the whole file as UTF-8, then UTF-16LE, then UTF-16BE, and isolate the
/// SD parameter block around the `Steps:` / `Civitai metadata:` anchor. A1111
/// stores it in the EXIF UserComment (commonly UTF-16), so the multi-encoding
/// pass finds it without a full EXIF parser.
fn scan_raw_for_parameters(bytes: &[u8]) -> Option<String> {
    // EXIF stores the A1111 block in `UserComment`, prefixed by an 8-byte charset
    // id ("UNICODE\0") and then UTF-16 text. Decoding the *whole* file as UTF-16
    // drags that marker in as mojibake right before the prompt (e.g. a leading
    // "啎䥃佄䔀"). So if the marker is present, decode only the text that follows it
    // — that lands exactly on the prompt with no marker garbage. EXIF's text byte
    // order varies, so try big- then little-endian.
    if let Some(pos) = find_bytes(bytes, b"UNICODE\0") {
        let after = &bytes[pos + 8..];
        if let Some(s) = isolate_sd(&decode_utf16(after, true)) {
            return Some(s);
        }
        if let Some(s) = isolate_sd(&decode_utf16(after, false)) {
            return Some(s);
        }
    }
    if let Some(s) = isolate_sd(&String::from_utf8_lossy(bytes)) {
        return Some(s);
    }
    if let Some(s) = isolate_sd(&decode_utf16(bytes, false)) {
        return Some(s);
    }
    if let Some(s) = isolate_sd(&decode_utf16(bytes, true)) {
        return Some(s);
    }
    None
}

/// Index of the first occurrence of `needle` in `haystack` (raw bytes).
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn decode_utf16(bytes: &[u8], big_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| if big_endian { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) })
        .collect();
    String::from_utf16_lossy(&units)
}

/// Find the `Steps:` (or `Civitai metadata:`) anchor and expand outward over
/// "valid" text characters. Mirrors Java `isolateSDMetadata`.
fn isolate_sd(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let anchor = find_subsequence(&chars, "Steps:").or_else(|| find_subsequence(&chars, "Civitai metadata:"))?;

    let mut start = anchor;
    while start > 0 && is_valid_text(chars[start - 1]) {
        start -= 1;
    }
    let mut end = anchor;
    while end < chars.len() && is_valid_text(chars[end]) {
        end += 1;
    }

    let s: String = chars[start..end].iter().collect();
    let s = s.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Index (in `chars`) of the first occurrence of `needle`, or `None`.
fn find_subsequence(chars: &[char], needle: &str) -> Option<usize> {
    let n: Vec<char> = needle.chars().collect();
    if n.is_empty() || chars.len() < n.len() {
        return None;
    }
    (0..=chars.len() - n.len()).find(|&i| chars[i..i + n.len()] == n[..])
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
    let root: serde_json::Value = serde_json::from_str(prompt_json).ok()?;
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
