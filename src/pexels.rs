//! Pexels source for the Downloader — free stock photos via the official API
//! (<https://www.pexels.com/api/documentation/>).
//!
//! Auth is an `Authorization: <API key>` header (free key, required). Search is
//! `GET api.pexels.com/v1/search?query=…&per_page=80&page=N`; each photo's
//! `src.original` is the full-resolution file on the images CDN (no auth needed
//! for the file itself). The `alt` description is kept as the image's caption
//! sidecar. Free tier: 200 requests/hour, 20 000/month — page fetches count,
//! CDN downloads don't.

use crate::download::Item;

pub const NAME: &str = "Pexels";
pub const SUBTITLE: &str = "Stock Photo Downloader";
pub const HOME: &str = "https://www.pexels.com/";
pub const CRED_URL: &str = "https://www.pexels.com/api/";
pub const CRED_INFO: &str = "Pexels needs a free API key: create one at pexels.com/api (free \
     account). Photos are free to use under the Pexels license. The free tier allows 200 API \
     requests per hour — each page of 80 results is one request, the photo files themselves \
     don't count.";
pub const TAGS_HINT: &str = "search terms, e.g. mountain sunrise";

/// API maximum results per page.
const PER_PAGE: u32 = 80;

/// Search URL for one (1-based) page of photos.
pub fn page_url(query: &str, page: u32) -> String {
    format!(
        "https://api.pexels.com/v1/search?query={}&per_page={PER_PAGE}&page={page}",
        crate::download::percent_encode(query.trim())
    )
}

/// Search URL for one (1-based) page of **videos** — Pexels serves those from
/// a separate endpoint with the same auth and shape conventions.
pub fn video_page_url(query: &str, page: u32) -> String {
    format!(
        "https://api.pexels.com/videos/search?query={}&per_page={PER_PAGE}&page={page}",
        crate::download::percent_encode(query.trim())
    )
}

/// A minimal authenticated call for the account check — one curated photo, so
/// it costs a single API request and returns the quota headers.
pub fn check_url() -> String {
    "https://api.pexels.com/v1/curated?per_page=1".to_string()
}

/// Request headers: the API key goes in `Authorization`, bare.
pub fn headers(key: &str) -> Vec<(&'static str, String)> {
    vec![("Authorization", key.trim().to_string())]
}

/// Parse one search page into source-neutral items.
pub fn parse(body: &str) -> Result<Vec<Item>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string()); // e.g. an invalid API key
    }
    let photos = v.get("photos").and_then(|p| p.as_array()).ok_or("no photos array in response")?;

    let mut out = Vec::new();
    for p in photos {
        let Some(id) = p.get("id").and_then(|v| v.as_i64()) else { continue };
        let Some(url) = p.pointer("/src/original").and_then(|v| v.as_str()) else { continue };
        let ext = url
            .split('?')
            .next()
            .unwrap_or(url)
            .rsplit('.')
            .next()
            .unwrap_or("jpg")
            .to_ascii_lowercase();
        // The alt text is a human caption ("Woman in red dress on a cliff…");
        // saved as the sidecar so the browser's search can find it.
        let tags = p
            .get("alt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        out.push(Item {
            url: url.to_string(),
            key: format!("pexels:{id}"),
            stem: format!("pexels-{id}"),
            ext,
            tags,
        });
    }
    Ok(out)
}

/// Parse one video-search page: each video offers several renditions in
/// `video_files` — the widest one is downloaded.
pub fn parse_videos(body: &str) -> Result<Vec<Item>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string());
    }
    let videos = v.get("videos").and_then(|p| p.as_array()).ok_or("no videos array in response")?;

    let mut out = Vec::new();
    for vid in videos {
        let Some(id) = vid.get("id").and_then(|v| v.as_i64()) else { continue };
        let Some(files) = vid.get("video_files").and_then(|f| f.as_array()) else { continue };
        let best = files
            .iter()
            .filter(|f| f.get("link").and_then(|l| l.as_str()).is_some())
            .max_by_key(|f| f.get("width").and_then(|w| w.as_i64()).unwrap_or(0));
        let Some(link) = best.and_then(|f| f.get("link")).and_then(|l| l.as_str()) else { continue };
        let ext = best
            .and_then(|f| f.get("file_type"))
            .and_then(|t| t.as_str())
            .and_then(|t| t.rsplit('/').next())
            .unwrap_or("mp4")
            .to_ascii_lowercase();
        out.push(Item {
            url: link.to_string(),
            key: format!("pexels:v{id}"),
            stem: format!("pexels-video-{id}"),
            ext,
            tags: None, // Pexels videos carry no alt text
        });
    }
    Ok(out)
}
