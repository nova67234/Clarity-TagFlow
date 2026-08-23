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

/// `sorting` values the API accepts, paired with their UI labels.
pub const SORTINGS: [(&str, &str); 6] = [
    ("date_added", "Date added"),
    ("relevance", "Relevance"),
    ("random", "Random"),
    ("views", "Views"),
    ("favorites", "Favourites"),
    ("toplist", "Toplist"),
];

/// `topRange` values (only honoured when sorting is `toplist`).
pub const TOP_RANGES: [(&str, &str); 7] = [
    ("1d", "Last day"),
    ("3d", "Last 3 days"),
    ("1w", "Last week"),
    ("1M", "Last month"),
    ("3M", "Last 3 months"),
    ("6M", "Last 6 months"),
    ("1y", "Last year"),
];

/// Search-filter selections from the Options card, mapped 1:1 onto the API's
/// search parameters. Persisted with the rest of the downloader config.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Opts {
    pub cat_general: bool,
    pub cat_anime: bool,
    pub cat_people: bool,
    pub pur_sfw: bool,
    pub pur_sketchy: bool,
    /// NSFW results require a valid API key.
    pub pur_nsfw: bool,
    /// One of [`SORTINGS`]'s API values.
    pub sorting: String,
    /// One of [`TOP_RANGES`]'s API values; sent only with `toplist` sorting.
    pub top_range: String,
    /// Minimum resolution, e.g. `1920x1080` (empty = any).
    pub atleast: String,
    /// Comma-separated aspect ratios, e.g. `16x9,16x10` (empty = any).
    pub ratios: String,
    /// Write each wallpaper's tag list to a `.txt` sidecar (like the boorus).
    /// Search results don't carry tags, so this costs one extra API call per
    /// file against the per-wallpaper endpoint.
    pub save_tags: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            cat_general: true,
            cat_anime: true,
            cat_people: true,
            pur_sfw: true,
            pur_sketchy: false,
            pur_nsfw: false,
            sorting: "date_added".to_string(),
            top_range: "1M".to_string(),
            atleast: String::new(),
            ratios: String::new(),
            save_tags: true,
        }
    }
}

/// Search URL for one (1-based) page; the key rides along when present.
/// Wallhaven supports tag exclusion natively (`-tag` in the query), so the
/// blacklist folds straight into the search instead of filtering afterwards.
pub fn page_url(q: &str, page: u32, key: &str, blacklist: &str, opts: &Opts, seed: &str) -> String {
    let mut query = q.trim().to_string();
    for term in blacklist.split(',') {
        let term = term.trim();
        if !term.is_empty() {
            query.push_str(" -");
            // Wallhaven tags are word-joined; a "two words" entry becomes one tag.
            query.push_str(&term.replace(char::is_whitespace, "_"));
        }
    }
    use std::fmt::Write as _;
    let bit = |on: bool| if on { '1' } else { '0' };
    let mut s = format!(
        "https://wallhaven.cc/api/v1/search?q={}&page={page}",
        crate::download::percent_encode(&query)
    );
    let _ = write!(
        s,
        "&categories={}{}{}&purity={}{}{}",
        bit(opts.cat_general),
        bit(opts.cat_anime),
        bit(opts.cat_people),
        bit(opts.pur_sfw),
        bit(opts.pur_sketchy),
        bit(opts.pur_nsfw),
    );
    if !opts.sorting.is_empty() {
        let _ = write!(s, "&sorting={}", opts.sorting);
    }
    if opts.sorting == "toplist" && !opts.top_range.is_empty() {
        let _ = write!(s, "&topRange={}", opts.top_range);
    }
    // A fixed seed keeps `random` pages from repeating between requests.
    if opts.sorting == "random" && !seed.is_empty() {
        let _ = write!(s, "&seed={}", crate::download::percent_encode(seed));
    }
    if !opts.atleast.trim().is_empty() {
        let _ = write!(s, "&atleast={}", crate::download::percent_encode(opts.atleast.trim()));
    }
    if !opts.ratios.trim().is_empty() {
        // The API takes a comma list; strip any stray whitespace around entries.
        let ratios: Vec<&str> =
            opts.ratios.split(',').map(str::trim).filter(|r| !r.is_empty()).collect();
        let _ = write!(s, "&ratios={}", crate::download::percent_encode(&ratios.join(",")));
    }
    if !key.trim().is_empty() {
        let _ = write!(s, "&apikey={}", crate::download::percent_encode(key.trim()));
    }
    s
}

/// The per-wallpaper info endpoint — unlike search results it carries the
/// full tag list. The key rides along when present (NSFW wallpapers 401
/// without it).
pub fn info_url(id: &str, key: &str) -> String {
    let mut s = format!("https://wallhaven.cc/api/v1/w/{}", crate::download::percent_encode(id));
    if !key.trim().is_empty() {
        use std::fmt::Write as _;
        let _ = write!(s, "?apikey={}", crate::download::percent_encode(key.trim()));
    }
    s
}

/// Pull the tag names out of an [`info_url`] response, comma-joined ready for
/// the `.txt` sidecar (Wallhaven tag names already contain spaces).
pub fn parse_tags(body: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("bad JSON: {e}"))?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string());
    }
    let tags = v
        .get("data")
        .and_then(|d| d.get("tags"))
        .and_then(|t| t.as_array())
        .ok_or("no tags array in response")?;
    let names: Vec<&str> = tags.iter().filter_map(|t| t.get("name").and_then(|n| n.as_str())).collect();
    Ok(names.join(", "))
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
/// tags — with `save_tags` on, the worker fills them per download via
/// [`info_url`] instead.
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
