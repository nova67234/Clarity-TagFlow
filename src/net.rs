//! Shared resumable HTTP download used by every model/setup installer (AI
//! models, the ComfyUI setup, Pixal3D, voice). One agent config and one
//! retry-with-resume loop, so a broken multi-GB pull continues from its
//! `.part` file instead of starting over — HuggingFace's Xet CDN has flaky
//! spells, and these are the app's biggest transfers.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Attempts per file before a download gives up (first try + retries).
pub const ATTEMPTS: u32 = 5;

/// What [`download`] reports through its `notify` callback.
pub enum Note<'a> {
    /// Bytes so far and the expected total (0 when the server didn't say).
    /// After a resume, `got` starts at the resumed offset, not zero.
    Progress { got: u64, total: u64 },
    /// An attempt failed; the next starts after a backoff sleep.
    Retry { attempt: u32, of: u32, err: &'a str },
}

/// The ureq agent every downloader shares. native-tls => Windows SChannel;
/// ureq 3.x defaults to rustls even with the native-tls feature on, so the
/// provider must be selected explicitly (avoids rustls/ring needing nasm on
/// MSVC — see Cargo.toml). Certificates validate against the OS store (with
/// AIA intermediate fetching) rather than ureq's bundled webpki roots — see
/// civitai.rs for the CDN incomplete-chain failure this avoids.
///
/// Timeouts: ureq 3 sets none, so a stalled connection blocked forever and the
/// retry/resume loop never ran. ureq 3.3 restarts a phase's limit on every
/// read, but ALSO cuts a phase off at the previous phase's end plus *that*
/// phase's limit (checked against a local test server). So `send_request`
/// doubles as the cap on waiting for the response headers, `recv_body` is a
/// per-read stall limit that never caps a long download's total time, and
/// `recv_response` must stay unset: it would cut the body off that long after
/// the headers (then allow 1 s per read).
pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_send_request(Some(std::time::Duration::from_secs(60)))
        .timeout_recv_body(Some(std::time::Duration::from_secs(60)))
        .build()
        .into()
}

/// Stream `url` into `dest` via a `<dest>.part` temp, renaming on success so a
/// partial file never looks installed. Retries with backoff, resuming the
/// `.part` where the stream broke; a leftover `.part` from an earlier failed
/// run is resumed too — same URL, so same content barring an upstream
/// re-upload mid-download. `bearer` adds an `Authorization` header (HF token)
/// when non-empty.
pub fn download(
    url: &str,
    dest: &Path,
    bearer: &str,
    notify: &mut dyn FnMut(Note),
) -> Result<(), String> {
    let agent = agent();
    let tmp = part_path(dest);
    let mut last_err = String::new();
    for attempt in 1..=ATTEMPTS {
        if attempt > 1 {
            notify(Note::Retry { attempt, of: ATTEMPTS, err: &last_err });
            std::thread::sleep(std::time::Duration::from_secs((1u64 << (attempt - 1)).min(10)));
        }
        match fetch_into(&agent, url, bearer, &tmp, notify) {
            Ok(()) => return std::fs::rename(&tmp, dest).map_err(|e| format!("save: {e}")),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

/// `model.safetensors` → `model.safetensors.part`: the extension is appended,
/// not replaced, so same-stem files can't clobber each other's temps. (Also
/// the temp name of the Civitai and booru/stock downloaders.)
pub(crate) fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// One streaming GET of `url` into `tmp`. When `tmp` already holds bytes from
/// a broken earlier attempt, asks the server to resume with a `Range` header —
/// appending on 206 Partial Content, restarting from zero when the server
/// ignores the range and sends the whole file (200).
fn fetch_into(
    agent: &ureq::Agent,
    url: &str,
    bearer: &str,
    tmp: &Path,
    notify: &mut dyn FnMut(Note),
) -> Result<(), String> {
    let have = std::fs::metadata(tmp).map(|m| m.len()).unwrap_or(0);

    // Error statuses come back as responses (not `Err`), so a 416's
    // Content-Range can be read below.
    let mut req = agent.get(url).config().http_status_as_error(false).build();
    if !bearer.is_empty() {
        req = req.header("Authorization", &format!("Bearer {bearer}"));
    }
    if have > 0 {
        req = req.header("Range", &format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    if status == 416 && have > 0 {
        // The requested range starts at/after the end. The .part holds the
        // whole file (the previous attempt died between the last byte and
        // the rename) only if it's exactly as long as the file; one that's
        // longer — the upstream file was replaced by a smaller one — or that
        // the server won't size is stale, and renaming it would install a
        // corrupt file. Drop it and fetch from zero (no Range, so no repeat).
        let size = resp.headers().get("Content-Range").and_then(|v| v.to_str().ok());
        if part_is_complete(have, size) {
            return Ok(());
        }
        std::fs::remove_file(tmp).map_err(|e| e.to_string())?;
        return fetch_into(agent, url, bearer, tmp, notify);
    }
    if status >= 400 {
        return Err(ureq::Error::StatusCode(status).to_string());
    }

    let resumed = resp.status() == 206;
    let body_len: u64 = resp
        .headers()
        .get("Content-Length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let total = if resumed { have + body_len } else { body_len };

    let mut out = if resumed {
        std::fs::OpenOptions::new().append(true).open(tmp)
    } else {
        std::fs::File::create(tmp) // also truncates a .part the server wouldn't resume
    }
    .map_err(|e| e.to_string())?;
    let mut got: u64 = if resumed { have } else { 0 };

    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let read = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        out.write_all(&buf[..read]).map_err(|e| e.to_string())?;
        got += read as u64;
        notify(Note::Progress { got, total });
    }
    out.flush().ok();
    // A truncated body (connection died before Content-Length bytes) must fail
    // this attempt so the retry resumes it — EOF alone doesn't mean complete.
    if total > 0 && got < total {
        return Err(format!("connection lost at {got} of {total} bytes"));
    }
    Ok(())
}

/// Whether a 416 for a `.part` of `have` bytes means the part is already the
/// complete file: the server's `Content-Range: bytes */<size>` equals `have`.
fn part_is_complete(have: u64, content_range: Option<&str>) -> bool {
    content_range
        .map(|s| s.trim().to_ascii_lowercase())
        .and_then(|s| s.strip_prefix("bytes */").and_then(|n| n.trim().parse::<u64>().ok()))
        == Some(have)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_part_extension() {
        assert_eq!(part_path(Path::new("m/x.safetensors")), Path::new("m/x.safetensors.part"));
        assert_ne!(part_path(Path::new("x.safetensors")), part_path(Path::new("x.ckpt")));
    }

    #[test]
    fn range_not_satisfiable_means_complete_only_at_exact_size() {
        assert!(part_is_complete(1000, Some("bytes */1000")));
        assert!(part_is_complete(1000, Some(" Bytes */1000 ")));
        // An over-long stale part, or a 416 that doesn't say the size: restart.
        assert!(!part_is_complete(1500, Some("bytes */1000")));
        assert!(!part_is_complete(1000, None));
        assert!(!part_is_complete(1000, Some("bytes 0-999/1000")));
    }
}
