//! Wallhaven source for the Downloader (<https://wallhaven.cc/help/api>).
//!
//! Search is `GET wallhaven.cc/api/v1/search?q=…&page=N`; each result's `path`
//! is the direct full-resolution file. Anonymous use is fine for SFW content —
//! an API key (from the account settings) rides as `apikey` and applies the
//! account's own browsing/purity settings. Rate limit: 45 requests/minute.

use crate::download::Item;

pub const NAME: &str = "Wallhaven";
pub const SUBTITLE: &str = "Wallpaper Downloader";
pub const HOME: &str = "https://wallhaven.cc/";
pub const CRED_URL: &str = "https://wallhaven.cc/settings/account";
pub const CRED_INFO: &str = "Wallhaven works without an account for safe-for-work wallpapers. \
     An API key (free, from Settings → Account) is optional: with it, searches use your \
     account's browsing and purity settings.";
pub const TAGS_HINT: &str = "search terms, e.g. landscape mountains";

/// Search URL for one (1-based) page; the key rides along when present.
/// Wallhaven supports tag exclusion natively (`-tag` in the query), so the
/// blacklist folds straight into the search instead of filtering afterwards.
pub fn page_url(q: &str, page: u32, key: &str, blacklist: &str) -> String {
    let mut query = q.trim().to_string();
    for term in blacklist.split(',') {
        let term = term.trim();
        if !term.is_empty() {
            query.push_str(" -");
            // Wallhaven tags are word-joined; a "two words" entry becomes one tag.
            query.push_str(&term.replace(char::is_whitespace, "_"));
        }
    }
    let mut s = format!(
        "https://wallhaven.cc/api/v1/search?q={}&page={page}",
        crate::download::percent_encode(&query)
    );
    if !key.trim().is_empty() {
        use std::fmt::Write as _;
        let _ = write!(s, "&apikey={}", crate::download::percent_encode(key.trim()));
    }
    s
}

/// The authenticated account-settings endpoint — 200 with a `data` object
/// means the key is accepted; 401 means it isn't.
pub fn settings_url(key: &str) -> String {
    format!(
        "https://wallhaven.cc/api/v1/settings?apikey={}",
        crate::download::percent_encode(key.trim())
    )
}

/// Parse one search page into source-neutral items. Search results don't carry
/// tags (only the per-wallpaper endpoint does), so no sidecars here.
pub fn parse(body: &str) -> Result<Vec<Item>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string()); // e.g. an invalid API key
    }
    let data = v.get("data").and_then(|d| d.as_array()).ok_or("no data array in response")?;

    let mut out = Vec::new();
    for w in data {
        let Some(id) = w.get("id").and_then(|v| v.as_str()) else { continue };
        let Some(path) = w.get("path").and_then(|v| v.as_str()) else { continue };
        let ext = path.rsplit('.').next().unwrap_or("jpg").to_ascii_lowercase();
        out.push(Item {
            url: path.to_string(),
            key: format!("wallhaven:{id}"),
            stem: format!("wallhaven-{id}"),
            ext,
            tags: None,
        });
    }
    Ok(out)
}
