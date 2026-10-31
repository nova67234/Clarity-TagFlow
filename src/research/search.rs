//! The search services web research can use, and reading what they found.
//!
//! * **Wikipedia** — free, no setup: its own search, and articles read as
//!   plain text through its API.
//! * **Tavily**, **Brave Search** — the whole web, with the user's own key.
//! * **SearXNG** — the user's own metasearch server (it may run on this PC or
//!   the home network: only *its* address may be local; the pages it finds
//!   are checked like any other).
//!
//! Every result's URL passes [`check_url`](super::net::check_url) before it's
//! kept, so a search answer can't send the reading step anywhere local.

use super::net::{self, check_url, FetchError};
use super::{Hit, Page, Provider, ResearchSettings};

/// A configured search service.
pub(super) enum Service {
    Wikipedia { lang: String, api: url::Url },
    Tavily { key: String, endpoint: url::Url },
    Brave { key: String, endpoint: url::Url },
    Searxng { search: url::Url },
}

impl Service {
    /// The service `settings` choose, or why it can't be used yet.
    pub(super) fn from_settings(s: &ResearchSettings) -> Result<Service, String> {
        Ok(match s.provider {
            Provider::Wikipedia => {
                let lang = s.wiki_lang.trim().to_ascii_lowercase();
                let lang = if lang.is_empty() { "en".to_string() } else { lang };
                if !valid_lang(&lang) {
                    return Err(format!("\"{lang}\" isn't a Wikipedia language code (like en, de or ja)"));
                }
                let api = url::Url::parse(&format!("https://{lang}.wikipedia.org/w/api.php"))
                    .map_err(|_| "bad Wikipedia language".to_string())?;
                Service::Wikipedia { lang, api }
            }
            Provider::Tavily => {
                let key = s.tavily_key();
                if key.trim().is_empty() {
                    return Err(need_key(Provider::Tavily));
                }
                Service::Tavily { key: key.trim().to_string(), endpoint: url::Url::parse(TAVILY).expect("const url") }
            }
            Provider::Brave => {
                let key = s.brave_key();
                if key.trim().is_empty() {
                    return Err(need_key(Provider::Brave));
                }
                Service::Brave { key: key.trim().to_string(), endpoint: url::Url::parse(BRAVE).expect("const url") }
            }
            Provider::Searxng => Service::Searxng { search: searxng_url(&s.searxng_url)? },
        })
    }

    /// For showing ("Wikipedia (en)", "SearXNG at localhost").
    pub(super) fn describe(&self) -> String {
        match self {
            Service::Wikipedia { lang, .. } => format!("Wikipedia ({lang})"),
            Service::Tavily { .. } => "Tavily".to_string(),
            Service::Brave { .. } => "Brave Search".to_string(),
            Service::Searxng { search } => format!("SearXNG at {}", search.host_str().unwrap_or("?")),
        }
    }

    /// Short name for progress lines ("Searching Wikipedia for …").
    pub(super) fn short(&self) -> &'static str {
        match self {
            Service::Wikipedia { .. } => "Wikipedia",
            Service::Tavily { .. } => "the web",
            Service::Brave { .. } => "the web",
            Service::Searxng { .. } => "the web",
        }
    }
}

/// What to do when `provider` has no key.
pub(super) fn need_key(provider: Provider) -> String {
    format!("add your {} key in AI Settings → Web research", provider.label())
}

const TAVILY: &str = "https://api.tavily.com/search";
const BRAVE: &str = "https://api.search.brave.com/res/v1/web/search";

/// "en", "de", "zh-yue", "simple"…: what can sit before ".wikipedia.org".
fn valid_lang(lang: &str) -> bool {
    let mut parts = lang.split('-');
    let first = parts.next().unwrap_or("");
    (2..=12).contains(&first.len())
        && first.bytes().all(|b| b.is_ascii_lowercase())
        && parts.all(|p| (1..=8).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()))
}

/// The SearXNG server's search URL from what the user typed
/// ("localhost:8888", "https://search.example.org/").
fn searxng_url(typed: &str) -> Result<url::Url, String> {
    let typed = typed.trim();
    if typed.is_empty() {
        return Err("add your SearXNG server's address in AI Settings → Web research".to_string());
    }
    let with_scheme =
        if typed.contains("://") { typed.to_string() } else { format!("http://{typed}") };
    let mut u = url::Url::parse(&with_scheme).map_err(|_| format!("\"{typed}\" isn't a valid server address"))?;
    if !matches!(u.scheme(), "http" | "https") || u.host_str().is_none() {
        return Err(format!("\"{typed}\" isn't a web address"));
    }
    if !u.username().is_empty() || u.password().is_some() {
        return Err("leave the user name and password out of the SearXNG address".to_string());
    }
    u.set_query(None);
    u.set_fragment(None);
    let path = u.path().trim_end_matches('/').trim_end_matches("/search").to_string();
    u.set_path(&format!("{path}/search"));
    Ok(u)
}

/// What [`Service::search`] returns per query.
pub(super) const RESULTS: usize = 6;

/// The two HTTP clients the research uses: `strict` (public internet only:
/// search APIs and every page) and `own` (also local addresses: only ever
/// pointed at the user's own SearXNG server).
pub(super) struct Clients {
    pub strict: ureq::Agent,
    pub own: ureq::Agent,
}

impl Service {
    /// Search for `query`.
    pub(super) fn search(&self, http: &Clients, query: &str) -> Result<Vec<Hit>, String> {
        let query: String = query.chars().take(200).collect();
        let accept_json = [("Accept", "application/json")];
        match self {
            Service::Wikipedia { lang, api } => {
                let mut u = api.clone();
                u.query_pairs_mut()
                    .append_pair("action", "query")
                    .append_pair("list", "search")
                    .append_pair("srsearch", &query)
                    .append_pair("srlimit", &RESULTS.to_string())
                    .append_pair("srprop", "snippet")
                    .append_pair("format", "json")
                    .append_pair("formatversion", "2")
                    .append_pair("utf8", "1");
                let body = net::get(&http.strict, u, &accept_json, &no_redirects).map_err(|e| search_err(e, self))?;
                parse_wikipedia(&whole(body)?, lang)
            }
            Service::Tavily { key, endpoint } => {
                let auth = format!("Bearer {key}");
                let body = serde_json::json!({
                    "query": query,
                    "max_results": RESULTS,
                    "search_depth": "basic",
                    "include_answer": false,
                    "include_raw_content": false,
                    "include_images": false,
                });
                let raw = net::post_json(&http.strict, endpoint, &[("Authorization", auth.as_str())], &body)
                    .map_err(|e| search_err(e, self))?;
                parse_tavily(&raw)
            }
            Service::Brave { key, endpoint } => {
                let mut u = endpoint.clone();
                u.query_pairs_mut().append_pair("q", &query).append_pair("count", &RESULTS.to_string());
                let headers = [("Accept", "application/json"), ("X-Subscription-Token", key.as_str())];
                let body = net::get(&http.strict, u, &headers, &no_redirects).map_err(|e| search_err(e, self))?;
                parse_brave(&whole(body)?)
            }
            Service::Searxng { search } => {
                let mut u = search.clone();
                u.query_pairs_mut().append_pair("q", &query).append_pair("format", "json");
                let body = net::get(&http.own, u, &accept_json, &no_redirects).map_err(|e| search_err(e, self))?;
                parse_searxng(&whole(body)?)
            }
        }
    }

    /// Read a found page: Wikipedia articles as plain text from its API,
    /// anything else fetched and turned into text.
    pub(super) fn read(&self, http: &Clients, hit: &Hit) -> Result<Page, String> {
        if let (Service::Wikipedia { api, .. }, Some(pageid)) = (self, hit.wiki_page) {
            let mut u = api.clone();
            u.query_pairs_mut()
                .append_pair("action", "query")
                .append_pair("prop", "extracts")
                .append_pair("explaintext", "1")
                .append_pair("exsectionformat", "plain")
                .append_pair("redirects", "1")
                .append_pair("pageids", &pageid.to_string())
                .append_pair("format", "json")
                .append_pair("formatversion", "2");
            let body = net::get(&http.strict, u, &[("Accept", "application/json")], &no_redirects)
                .map_err(|e| e.why())?;
            let (title, text) = parse_wikipedia_extract(&whole(body)?)?;
            return Ok(Page { title, url: hit.url.clone(), text });
        }
        read_page(&http.strict, &hit.url)
    }
}

/// An API's answer, whole: one cut off at the size cap isn't valid JSON.
fn whole(f: net::Fetched) -> Result<Vec<u8>, String> {
    if f.truncated {
        return Err("the answer was too big".to_string());
    }
    Ok(f.body)
}

/// Refuse every redirect (API answers don't move).
fn no_redirects(_: &str) -> Result<url::Url, net::Refused> {
    Err(net::Refused::Invalid)
}

/// A search request's failure, worded for the user.
fn search_err(e: FetchError, service: &Service) -> String {
    let searxng = matches!(service, Service::Searxng { .. });
    match e {
        FetchError::Status(401 | 403) if searxng => {
            "the SearXNG server refused — turn on JSON results (search → formats: json in its settings.yml)"
                .to_string()
        }
        FetchError::Status(401 | 403) => "the search service refused the key".to_string(),
        FetchError::Status(429) => "too many searches — the free allowance may be used up; try again later".to_string(),
        FetchError::Status(432 | 433) => "the search allowance is used up".to_string(),
        FetchError::Status(s) => format!("the search service answered {s}"),
        other => format!("search failed: {}", other.why()),
    }
}

/// Fetch a web page and turn it into text (HTML or plain text only).
pub(super) fn read_page(agent: &ureq::Agent, url: &str) -> Result<Page, String> {
    read_page_with(agent, url, &check_url)
}

/// [`read_page`], with `check` passing the first URL and every redirect.
pub(super) fn read_page_with(
    agent: &ureq::Agent,
    url: &str,
    check: &dyn Fn(&str) -> Result<url::Url, net::Refused>,
) -> Result<Page, String> {
    let start = check(url).map_err(|r| r.why().to_string())?;
    let headers = [
        ("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1"),
        ("Accept-Language", "en;q=0.8,*;q=0.5"),
    ];
    let got = net::get(agent, start, &headers, check).map_err(|e| e.why())?;
    let ct = got.content_type.split(';').next().unwrap_or("").trim().to_string();
    let looks_html = || {
        let head = String::from_utf8_lossy(&got.body[..got.body.len().min(512)]).to_ascii_lowercase();
        head.trim_start().starts_with('<')
    };
    let (title, text) = match ct.as_str() {
        "text/html" | "application/xhtml+xml" => {
            let e = super::html::to_text(&net::decode(&got.body, &got.content_type));
            (e.title, e.text)
        }
        "text/plain" | "text/markdown" => (String::new(), super::html::clean(&net::decode(&got.body, &got.content_type))),
        "" if looks_html() => {
            let e = super::html::to_text(&net::decode(&got.body, ""));
            (e.title, e.text)
        }
        other => return Err(format!("not a web page ({})", if other.is_empty() { "unknown type" } else { other })),
    };
    if text.trim().is_empty() {
        return Err("no readable text on the page".to_string());
    }
    Ok(Page { title, url: got.url.to_string(), text })
}

// ---------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------

/// A field as trimmed text, at most `max` characters, tags stripped (search
/// snippets carry `<strong>`/`<span>` highlighting).
fn field(v: &serde_json::Value, key: &str, max: usize) -> String {
    let raw = v.get(key).and_then(serde_json::Value::as_str).unwrap_or("");
    let text = if raw.contains('<') || raw.contains('&') {
        super::html::to_text(&format!("<p>{raw}</p>")).text
    } else {
        super::html::clean(raw)
    };
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    text.chars().take(max).collect()
}

/// Keep a result only when its URL may be visited (normalized); no duplicates.
fn keep(hits: &mut Vec<Hit>, title: String, url: &str, snippet: String, wiki_page: Option<u64>) {
    let Ok(u) = check_url(url) else { return };
    let key = net::normalized(&u);
    if hits.iter().any(|h| check_url(&h.url).is_ok_and(|v| net::normalized(&v) == key)) {
        return;
    }
    let title = if title.is_empty() { net::site_of(u.as_str()) } else { title };
    hits.push(Hit { title, url: u.to_string(), snippet, wiki_page });
}

fn json(body: &[u8]) -> Result<serde_json::Value, String> {
    serde_json::from_slice(body).map_err(|_| "the search service sent something unreadable".to_string())
}

pub(super) fn parse_wikipedia(body: &[u8], lang: &str) -> Result<Vec<Hit>, String> {
    let v = json(body)?;
    if let Some(err) = v.get("error") {
        return Err(format!("Wikipedia: {}", field(err, "info", 120)));
    }
    let mut hits = Vec::new();
    for r in v.pointer("/query/search").and_then(serde_json::Value::as_array).into_iter().flatten().take(20) {
        let title = field(r, "title", 200);
        let Some(pageid) = r.get("pageid").and_then(serde_json::Value::as_u64) else { continue };
        if title.is_empty() {
            continue;
        }
        let mut u = url::Url::parse(&format!("https://{lang}.wikipedia.org/wiki/")).map_err(|_| "bad language")?;
        u.path_segments_mut().map_err(|_| "bad url")?.pop().push(&title.replace(' ', "_"));
        let snippet = field(r, "snippet", 600);
        keep(&mut hits, title, u.as_str(), snippet, Some(pageid));
    }
    Ok(hits)
}

pub(super) fn parse_wikipedia_extract(body: &[u8]) -> Result<(String, String), String> {
    let v = json(body)?;
    let page = v
        .pointer("/query/pages/0")
        .ok_or_else(|| "Wikipedia sent no article".to_string())?;
    if page.get("missing").is_some() {
        return Err("the article is gone".to_string());
    }
    let title = field(page, "title", 200);
    let text = super::html::clean(page.get("extract").and_then(serde_json::Value::as_str).unwrap_or(""));
    if text.trim().is_empty() {
        return Err("the article has no text".to_string());
    }
    Ok((title, text))
}

pub(super) fn parse_tavily(body: &[u8]) -> Result<Vec<Hit>, String> {
    let v = json(body)?;
    let mut hits = Vec::new();
    for r in v.get("results").and_then(serde_json::Value::as_array).into_iter().flatten().take(20) {
        let url = r.get("url").and_then(serde_json::Value::as_str).unwrap_or("");
        keep(&mut hits, field(r, "title", 200), url, field(r, "content", 900), None);
    }
    Ok(hits)
}

pub(super) fn parse_brave(body: &[u8]) -> Result<Vec<Hit>, String> {
    let v = json(body)?;
    let mut hits = Vec::new();
    for r in v.pointer("/web/results").and_then(serde_json::Value::as_array).into_iter().flatten().take(20) {
        let url = r.get("url").and_then(serde_json::Value::as_str).unwrap_or("");
        keep(&mut hits, field(r, "title", 200), url, field(r, "description", 600), None);
    }
    Ok(hits)
}

pub(super) fn parse_searxng(body: &[u8]) -> Result<Vec<Hit>, String> {
    let v = json(body)?;
    let mut hits = Vec::new();
    for r in v.get("results").and_then(serde_json::Value::as_array).into_iter().flatten().take(20) {
        let url = r.get("url").and_then(serde_json::Value::as_str).unwrap_or("");
        keep(&mut hits, field(r, "title", 200), url, field(r, "content", 600), None);
    }
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wikipedia_results_become_article_links() {
        let body = br#"{"batchcomplete":true,"query":{"searchinfo":{"totalhits":2},"search":[
            {"ns":0,"title":"Eiffel Tower","pageid":9232,"snippet":"The <span class=\"searchmatch\">Eiffel</span> Tower is a &quot;wrought-iron&quot; tower"},
            {"ns":0,"title":"Rust (programming language)","pageid":29414838,"snippet":"Rust is"},
            {"ns":0,"title":"No id"},
            {"ns":0,"title":"","pageid":5}]}}"#;
        let hits = parse_wikipedia(body, "en").unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://en.wikipedia.org/wiki/Eiffel_Tower");
        assert_eq!(hits[0].snippet, "The Eiffel Tower is a \"wrought-iron\" tower");
        assert_eq!(hits[0].wiki_page, Some(9232));
        assert_eq!(hits[1].url, "https://en.wikipedia.org/wiki/Rust_(programming_language)");
        // A title can't escape the article path.
        let body = br#"{"query":{"search":[{"title":"../../w/api.php?x=1#y","pageid":1}]}}"#;
        let hits = parse_wikipedia(body, "de").unwrap();
        assert!(hits[0].url.starts_with("https://de.wikipedia.org/wiki/"), "{}", hits[0].url);
        assert!(!hits[0].url.contains("/w/api.php"), "{}", hits[0].url);
        assert!(parse_wikipedia(br#"{"error":{"code":"x","info":"Bad thing"}}"#, "en").unwrap_err().contains("Bad thing"));
    }

    #[test]
    fn wikipedia_extracts_read_as_text() {
        let body = "{\"query\":{\"pages\":[{\"pageid\":9232,\"title\":\"Eiffel Tower\",\"extract\":\"The Eiffel Tower is in Paris.\\n\\nHistory\\nBuilt 1887\\u2013189\\u200b9.\"}]}}";
        let body = body.as_bytes();
        let (title, text) = parse_wikipedia_extract(body).unwrap();
        assert_eq!(title, "Eiffel Tower");
        assert!(text.contains("Built 1887–1899."), "{text}");
        assert!(parse_wikipedia_extract(br#"{"query":{"pages":[{"pageid":1,"missing":true}]}}"#).is_err());
        assert!(parse_wikipedia_extract(b"not json").is_err());
    }

    #[test]
    fn web_search_answers_keep_only_visitable_links() {
        let tavily = br#"{"query":"x","results":[
            {"title":"Good","url":"https://example.com/a","content":"Fine <b>text</b>","score":0.9},
            {"title":"Local","url":"http://192.168.1.1/admin","content":"router"},
            {"title":"File","url":"file:///etc/passwd","content":"x"},
            {"title":"Script","url":"javascript:alert(1)","content":"x"},
            {"title":"Port","url":"https://example.com:8443/","content":"x"},
            {"title":"Dup","url":"https://example.com/a#section","content":"same page"},
            {"title":"Creds","url":"https://u:p@example.org/","content":"x"},
            {"url":"https://example.net/untitled","content":"no title"},
            {"title":42,"url":17},
            {"title":"Huge","url":"https://example.org/huge","content":"%HUGE%"}]}"#;
        let tavily = String::from_utf8_lossy(tavily).replace("%HUGE%", &"y".repeat(50_000));
        let hits = parse_tavily(tavily.as_bytes()).unwrap();
        let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).collect();
        assert_eq!(urls, ["https://example.com/a", "https://example.net/untitled", "https://example.org/huge"]);
        assert_eq!(hits[0].snippet, "Fine text");
        assert_eq!(hits[1].title, "example.net");
        assert!(hits[2].snippet.chars().count() <= 900);

        let brave = br#"{"web":{"results":[{"title":"A <strong>b</strong>","url":"https://a.example.com/","description":"d"},
            {"title":"L","url":"http://localhost:3000/"}]}}"#;
        let hits = parse_brave(brave).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "A b");

        let searx = br#"{"query":"q","results":[{"url":"https://x.example.org/p","title":"T","content":"C"},
            {"url":"http://[::1]/","title":"loop"}]}"#;
        let hits = parse_searxng(searx).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(parse_searxng(b"<html>nope</html>").is_err());
        assert!(parse_brave(b"{}").unwrap().is_empty());
    }

    #[test]
    fn settings_validate_language_keys_and_servers() {
        let mut s = ResearchSettings::default();
        assert!(matches!(Service::from_settings(&s), Ok(Service::Wikipedia { .. })));
        s.wiki_lang = "evil.com/x?".into();
        assert!(Service::from_settings(&s).is_err());
        s.wiki_lang = "zh-yue".into();
        assert!(Service::from_settings(&s).is_ok());
        s.provider = Provider::Tavily;
        assert!(Service::from_settings(&s).err().unwrap().contains("Tavily key"));
        s.set_tavily_key("tvly-123");
        assert!(matches!(Service::from_settings(&s), Ok(Service::Tavily { .. })));
        s.provider = Provider::Searxng;
        assert!(Service::from_settings(&s).is_err());
        s.searxng_url = "localhost:8888".into();
        match Service::from_settings(&s) {
            Ok(Service::Searxng { search }) => assert_eq!(search.as_str(), "http://localhost:8888/search"),
            _ => panic!(),
        }
        s.searxng_url = "https://search.example.org/searxng/search/?q=old#x".into();
        match Service::from_settings(&s) {
            Ok(Service::Searxng { search }) => assert_eq!(search.as_str(), "https://search.example.org/searxng/search"),
            _ => panic!(),
        }
        s.searxng_url = "ftp://x".into();
        assert!(Service::from_settings(&s).is_err());
        s.searxng_url = "http://admin:pw@192.168.1.5:8080".into();
        assert!(Service::from_settings(&s).is_err());
    }
}
