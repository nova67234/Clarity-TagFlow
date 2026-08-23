//! Danbooru source for the Downloader (<https://danbooru.donmai.us/wiki_pages/help:api>).
//!
//! Works anonymously (rate-limited, and free accounts may use at most 2 tags
//! per query); a Login + API key raises the account's limits and unlocks its
//! favorites etc. Auth rides as `login`/`api_key` URL parameters. Posts come
//! from `GET /posts.json?tags=…&limit=200&page=N`; the flat `tag_string` is
//! written to the sidecar just like Gelbooru's tag list.

use crate::download::Item;

pub const NAME: &str = "Danbooru";
pub const SUBTITLE: &str = "Tag Downloader";
pub const HOME: &str = "https://danbooru.donmai.us/";
pub const CRED_URL: &str = "https://danbooru.donmai.us/profile";
pub const CRED_INFO: &str = "Danbooru works without an account, but anonymous/free users are \
     limited to 2 tags per search and stricter rate limits. To authenticate, open your Danbooru \
     profile and generate an API key, then enter your login name and that key (both together).";
pub const TAGS_HINT: &str = "space-separated; 2 tags max on a free account";

/// API maximum results per page.
const PER_PAGE: u32 = 200;

/// Search URL for one (1-based) page; credentials ride along when both halves
/// are present.
pub fn page_url(tags: &str, page: u32, login: &str, key: &str) -> String {
    let mut s = format!(
        "https://danbooru.donmai.us/posts.json?tags={}&limit={PER_PAGE}&page={page}",
        crate::download::percent_encode(tags.trim())
    );
    if !login.trim().is_empty() && !key.trim().is_empty() {
        use std::fmt::Write as _;
        let _ = write!(
            s,
            "&login={}&api_key={}",
            crate::download::percent_encode(login.trim()),
            crate::download::percent_encode(key.trim())
        );
    }
    s
}

/// The authenticated profile endpoint — returns the account's own record,
/// including its ban status.
pub fn profile_url(login: &str, key: &str) -> String {
    format!(
        "https://danbooru.donmai.us/profile.json?login={}&api_key={}",
        crate::download::percent_encode(login.trim()),
        crate::download::percent_encode(key.trim())
    )
}

/// Parse the profile: `(name, level, is_banned)`.
pub fn parse_profile(body: &str) -> Result<(String, String, bool), String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
        return Err(msg.to_string());
    }
    let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("(unknown)").to_string();
    let level = v
        .get("level_string")
        .and_then(|l| l.as_str())
        .unwrap_or("Member")
        .to_string();
    let banned = v.get("is_banned").and_then(|b| b.as_bool()).unwrap_or(false);
    Ok((name, level, banned))
}

/// Parse one `posts.json` page into source-neutral items.
pub fn parse(body: &str) -> Result<Vec<Item>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    // Errors come back as an object ({"success":false,"message":…}) instead of
    // the posts array.
    if let Some(obj) = v.as_object() {
        let msg = obj
            .get("message")
            .or_else(|| obj.get("error"))
            .and_then(|m| m.as_str())
            .unwrap_or("unexpected response");
        return Err(msg.to_string());
    }
    let posts = v.as_array().ok_or("no posts array in response")?;

    let mut out = Vec::new();
    for p in posts {
        // Restricted/banned posts come back without a file_url — skip them.
        let Some(url) = p.get("file_url").and_then(|v| v.as_str()) else { continue };
        let Some(id) = p.get("id").and_then(|v| v.as_i64()) else { continue };
        let ext = p
            .get("file_ext")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| url.split('?').next().unwrap_or(url).rsplit('.').next().unwrap_or("jpg"))
            .to_ascii_lowercase();
        // Prefer the md5 as the file stem (matches the Gelbooru convention and
        // the shared tag_roles keying); fall back to the post id.
        let stem = p
            .get("md5")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("danbooru-{id}"));
        let tags = p
            .get("tag_string")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        out.push(Item {
            url: url.to_string(),
            key: format!("danbooru:{id}"),
            stem,
            ext,
            tags,
        });
    }
    Ok(out)
}
