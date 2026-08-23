//! Pixabay source for the Downloader — free stock photos, illustrations,
//! vectors and videos via the official API (<https://pixabay.com/api/docs/>).
//!
//! Auth is a `key` query parameter (free key, required — it's shown on the API
//! docs page while logged in). Search is `GET pixabay.com/api/?key=…&q=…&page=N`;
//! videos come from the separate `pixabay.com/api/videos/` endpoint. Each hit
//! carries a real comma-separated `tags` field, kept as the sidecar. The
//! `largeImageURL` (up to 1280 px) is the biggest rendition open to every key —
//! the original-size URLs require special approval from Pixabay. Rate limit:
//! 100 requests per 60 seconds; the media files themselves don't count.

use crate::download::Item;

pub const NAME: &str = "Pixabay";
pub const SUBTITLE: &str = "Stock Media Downloader";
pub const HOME: &str = "https://pixabay.com/";
pub const CRED_URL: &str = "https://pixabay.com/api/docs/";
pub const CRED_INFO: &str = "Pixabay needs a free API key: sign up at pixabay.com, then the key \
     is shown on the API docs page (pixabay.com/api/docs). Content is free to use under the \
     Pixabay Content License. The API allows 100 requests per minute — each page of 200 results \
     is one request, the media files themselves don't count.";
pub const TAGS_HINT: &str = "search terms, e.g. yellow flowers";

/// API maximum results per page (docs allow 3–200).
const PER_PAGE: u32 = 200;

/// Search URL for one (1-based) page of images (photos + illustrations +
/// vectors — the API's default `image_type=all`).
pub fn page_url(query: &str, page: u32, key: &str) -> String {
    format!(
        "https://pixabay.com/api/?key={}&q={}&per_page={PER_PAGE}&page={page}",
        crate::download::percent_encode(key.trim()),
        crate::download::percent_encode(query.trim()),
    )
}

/// Search URL for one (1-based) page of **videos** — same shape, separate
/// endpoint.
pub fn video_page_url(query: &str, page: u32, key: &str) -> String {
    format!(
        "https://pixabay.com/api/videos/?key={}&q={}&per_page={PER_PAGE}&page={page}",
        crate::download::percent_encode(key.trim()),
        crate::download::percent_encode(query.trim()),
    )
}

/// A minimal authenticated call for the account check — one tiny page, so it
/// costs a single API request and returns the rate-limit headers.
pub fn check_url(key: &str) -> String {
    format!(
        "https://pixabay.com/api/?key={}&per_page=3",
        crate::download::percent_encode(key.trim())
    )
}

/// Parse one image-search page into source-neutral items.
pub fn parse(body: &str) -> Result<Vec<Item>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    let hits = v.get("hits").and_then(|h| h.as_array()).ok_or("no hits array in response")?;

    let mut out = Vec::new();
    for hit in hits {
        let Some(id) = hit.get("id").and_then(|v| v.as_i64()) else { continue };
        // largeImageURL (≤1280 px) is the best rendition every key can access;
        // webformatURL is the fallback on older/odd hits.
        let Some(url) = ["largeImageURL", "webformatURL"]
            .iter()
            .find_map(|k| hit.get(*k).and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        else {
            continue;
        };
        out.push(Item {
            url: url.to_string(),
            key: format!("pixabay:{id}"),
            stem: format!("pixabay-{id}"),
            ext: url_ext(url, "jpg"),
            tags: hit_tags(hit),
        });
    }
    Ok(out)
}

/// Parse one video-search page: each hit offers `large`/`medium`/`small`/`tiny`
/// renditions — the widest one with a URL is downloaded.
pub fn parse_videos(body: &str) -> Result<Vec<Item>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    let hits = v.get("hits").and_then(|h| h.as_array()).ok_or("no hits array in response")?;

    let mut out = Vec::new();
    for hit in hits {
        let Some(id) = hit.get("id").and_then(|v| v.as_i64()) else { continue };
        let Some(sizes) = hit.get("videos").and_then(|s| s.as_object()) else { continue };
        let best = sizes
            .values()
            .filter(|s| s.get("url").and_then(|u| u.as_str()).is_some_and(|u| !u.is_empty()))
            .max_by_key(|s| s.get("width").and_then(|w| w.as_i64()).unwrap_or(0));
        let Some(url) = best.and_then(|s| s.get("url")).and_then(|u| u.as_str()) else { continue };
        out.push(Item {
            url: url.to_string(),
            key: format!("pixabay:v{id}"),
            stem: format!("pixabay-video-{id}"),
            ext: url_ext(url, "mp4"),
            tags: hit_tags(hit),
        });
    }
    Ok(out)
}

/// The hit's comma-separated `tags` string ("flowers, yellow, blossom"),
/// already in the sidecar format.
fn hit_tags(hit: &serde_json::Value) -> Option<String> {
    hit.get("tags")
        .and_then(|t| t.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// File extension from a CDN URL, ignoring any query string.
fn url_ext(url: &str, fallback: &str) -> String {
    let path = url.split('?').next().unwrap_or(url);
    match path.rsplit_once('.') {
        Some((_, e)) if !e.is_empty() && e.len() <= 5 && !e.contains('/') => e.to_ascii_lowercase(),
        _ => fallback.to_string(),
    }
}
