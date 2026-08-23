//! Safebooru source for the Downloader (<https://safebooru.org/index.php?page=help&topic=dapi>).
//!
//! Safebooru runs the same Gelbooru 0.2 engine and `dapi` API as gelbooru.com,
//! but is safe-for-work only and fully anonymous — no account or API key
//! exists for the API, so this source skips the Account card entirely. Search
//! is `GET index.php?page=dapi&s=post&q=index&json=1&tags=…&pid=N` (0-based
//! pages); posts carry a full-resolution `file_url`, an md5 `hash` (kept as
//! the file stem, matching the Gelbooru convention), and the space-separated
//! `tags` string written to the sidecar like the other boorus.

use crate::download::Item;

pub const NAME: &str = "Safebooru";
pub const SUBTITLE: &str = "Tag Downloader";
pub const HOME: &str = "https://safebooru.org/";
pub const INFO: &str = "Safebooru is a safe-for-work anime image board running the same engine \
     as Gelbooru. It works fully anonymously — no account or API key needed.";
pub const TAGS_HINT: &str = "space-separated, e.g. blue_sky 1girl";

/// Results per page (the dapi default; kept modest to be polite).
const PER_PAGE: u32 = 100;

/// Search URL for one (1-based) page — the dapi's `pid` counts from 0.
pub fn page_url(tags: &str, page: u32) -> String {
    format!(
        "{HOME}index.php?page=dapi&s=post&q=index&json=1&limit={PER_PAGE}&tags={}&pid={}",
        crate::download::percent_encode(tags.trim()),
        page.saturating_sub(1),
    )
}

/// Parse one search page into source-neutral items. A search with no matches
/// returns an empty body rather than `[]`.
pub fn parse(body: &str) -> Result<Vec<Item>, String> {
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    let posts = v.as_array().ok_or("no posts array in response")?;

    let mut out = Vec::new();
    for p in posts {
        let Some(id) = p.get("id").and_then(|v| v.as_i64()) else { continue };
        // Modern Safebooru includes file_url directly; older mirrors of the
        // engine need it built from directory + image.
        let url = match p.get("file_url").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
            Some(u) => u.to_string(),
            None => {
                let (Some(dir), Some(image)) = (
                    p.get("directory").map(|d| d.to_string()),
                    p.get("image").and_then(|v| v.as_str()),
                ) else {
                    continue;
                };
                format!("{HOME}images/{}/{image}", dir.trim_matches('"'))
            }
        };
        let ext = url
            .split('?')
            .next()
            .unwrap_or(&url)
            .rsplit('.')
            .next()
            .unwrap_or("jpg")
            .to_ascii_lowercase();
        // Prefer the md5 hash as the file stem (matches the Gelbooru
        // convention); fall back to the post id.
        let stem = p
            .get("hash")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("safebooru-{id}"));
        let tags = p
            .get("tags")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        out.push(Item { url, key: format!("safebooru:{id}"), stem, ext, tags });
    }
    Ok(out)
}
