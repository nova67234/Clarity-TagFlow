//! The network side of web research: which addresses it may visit, and
//! fetching within limits.
//!
//! * Only `http`/`https` URLs on the default ports, with no user name or
//!   password in them, whose host is a public internet name or address —
//!   never this PC, the local network or a special-purpose range
//!   ([`check_url`]).
//! * Every connection goes through [`SafeResolver`]: whatever a name resolves
//!   to, only the public addresses among the answers are dialed. A public
//!   name that DNS points at 192.168.1.1 — or answers differently on a second
//!   lookup — can't steer a request into the home network: the check is on
//!   the very addresses that are connected to.
//! * Redirects are followed here, not by the HTTP client, each new URL
//!   checked like the first, at most [`MAX_REDIRECTS`].
//! * No proxy (a proxy would resolve names itself, out of reach of the
//!   check), no cookies, no compression (nothing to inflate), at most
//!   [`MAX_BYTES`] read, all within one time limit per request — a hard one
//!   ([`HardDeadline`]): a server that drips its answer a byte at a time is
//!   cut off when the time is up, redirects included — and Stop ends a
//!   request that's waiting within a fraction of a second.

use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ureq::unversioned::resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver};
use ureq::unversioned::transport::{
    time, Buffers, ConnectionDetails, Connector, NativeTlsConnector, NextTimeout, TcpConnector, Transport,
};

/// Most of a response that's read; the rest is dropped (a page's text is in
/// its first couple of megabytes, and this bounds what a hostile server can
/// make the app hold).
pub(super) const MAX_BYTES: u64 = 2 * 1024 * 1024;
/// Redirects followed for one page.
pub(super) const MAX_REDIRECTS: usize = 5;
/// Longest URL visited.
const MAX_URL: usize = 2048;

/// What the research identifies as, so sites (and Wikipedia's API rules) can
/// tell what's asking.
pub(super) fn user_agent() -> String {
    format!(
        "ClarityTagFlow/{} (web research; +https://github.com/nova67234/Clarity-TagFlow)",
        env!("CARGO_PKG_VERSION")
    )
}

// ---------------------------------------------------------------------------
// Addresses
// ---------------------------------------------------------------------------

/// A public internet address — the only kind web research connects to.
pub(crate) fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_public(v4),
        IpAddr::V6(v6) => v6_public(v6),
    }
}

fn v4_public(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0 // "this network"
        || a == 10 // private
        || (a == 100 && (64..=127).contains(&b)) // carrier-grade NAT
        || a == 127 // this PC
        || (a == 169 && b == 254) // link-local (and cloud metadata servers)
        || (a == 172 && (16..=31).contains(&b)) // private
        || (a == 192 && b == 0 && c == 0) // protocol assignments
        || (a == 192 && b == 0 && c == 2) // documentation
        || (a == 192 && b == 88 && c == 99) // 6to4 relays
        || (a == 192 && b == 168) // private
        || (a == 198 && (b == 18 || b == 19)) // benchmarking
        || (a == 198 && b == 51 && c == 100) // documentation
        || (a == 203 && b == 0 && c == 113) // documentation
        || a >= 224) // multicast, reserved, broadcast
}

fn v6_public(ip: Ipv6Addr) -> bool {
    // An IPv4 address carried in IPv6 is judged as that IPv4 address.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return v4_public(v4);
    }
    let s = ip.segments();
    // NAT64's well-known prefix (64:ff9b::/96) carries one too.
    if s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0; 4] {
        let [a, b] = s[6].to_be_bytes();
        let [c, d] = s[7].to_be_bytes();
        return v4_public(Ipv4Addr::new(a, b, c, d));
    }
    // Otherwise only global unicast (2000::/3), less its special blocks.
    (s[0] & 0xe000) == 0x2000
        && s[0] != 0x2002 // 6to4
        && !(s[0] == 0x2001 && s[1] < 0x0200) // protocol assignments, Teredo
        && !(s[0] == 0x2001 && s[1] == 0x0db8) // documentation
        && !(s[0] == 0x3fff && s[1] < 0x1000) // documentation
}

/// Names that only mean something on a local network (or never resolve).
const LOCAL_NAMES: [&str; 15] = [
    "localhost",
    "local",
    "localdomain",
    "internal",
    "intranet",
    "lan",
    "home",
    "corp",
    "private",
    "home.arpa",
    "arpa",
    "test",
    "invalid",
    "example",
    "onion",
];

/// A dotted internet name that isn't one of the names local networks use: at
/// least two labels, ending in a letters-only (or punycode) top-level label.
fn public_name(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    let Some((_, tld)) = host.rsplit_once('.') else {
        return false; // "router", "nas": a single label resolves on the LAN
    };
    let tld_ok = tld.len() >= 2 && (tld.bytes().all(|b| b.is_ascii_alphabetic()) || tld.starts_with("xn--"));
    tld_ok && !LOCAL_NAMES.iter().any(|n| host == *n || host.ends_with(&format!(".{n}")))
}

/// Why a URL isn't visited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refused {
    /// Not a web address (`file:`, `javascript:`, `ftp:`…).
    NotWeb,
    /// A user name or password in it.
    Credentials,
    /// A port other than the scheme's own.
    Port,
    /// This PC, the local network, or a reserved address.
    Local,
    /// Not a URL at all.
    Invalid,
    TooLong,
}

impl Refused {
    pub(crate) fn why(self) -> &'static str {
        match self {
            Refused::NotWeb => "not a web address",
            Refused::Credentials => "has a user name or password in it",
            Refused::Port => "uses an unusual port",
            Refused::Local => "points at this PC or the local network",
            Refused::Invalid => "isn't a valid address",
            Refused::TooLong => "address too long",
        }
    }
}

/// A URL web research may visit (see the module docs). Where a name actually
/// resolves is checked again on connecting ([`SafeResolver`]).
pub(crate) fn check_url(raw: &str) -> Result<url::Url, Refused> {
    let raw = raw.trim();
    if raw.len() > MAX_URL {
        return Err(Refused::TooLong);
    }
    let url = url::Url::parse(raw).map_err(|_| Refused::Invalid)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Refused::NotWeb);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(Refused::Credentials);
    }
    match url.host() {
        Some(url::Host::Ipv4(ip)) if v4_public(ip) => {}
        Some(url::Host::Ipv6(ip)) if v6_public(ip) => {}
        Some(url::Host::Domain(name)) if public_name(name) => {}
        Some(_) => return Err(Refused::Local),
        None => return Err(Refused::Invalid),
    }
    // `url` leaves the port out when it's the scheme's default.
    if url.port().is_some() {
        return Err(Refused::Port);
    }
    if url.as_str().len() > MAX_URL {
        return Err(Refused::TooLong);
    }
    Ok(url)
}

/// A resolver that dials only the addresses `allow` accepts (normally
/// [`is_public`]), whatever DNS answered — the guard against names that
/// point into the local network.
#[derive(Debug)]
pub(super) struct SafeResolver {
    pub(super) allow: fn(IpAddr) -> bool,
}

/// The error a [`SafeResolver`] refuses with.
#[derive(Debug)]
pub(super) struct Blocked;

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("points at this PC or the local network")
    }
}

impl std::error::Error for Blocked {}

impl Resolver for SafeResolver {
    fn resolve(
        &self,
        uri: &ureq::http::Uri,
        config: &ureq::config::Config,
        timeout: NextTimeout,
    ) -> Result<ResolvedSocketAddrs, ureq::Error> {
        let found = DefaultResolver::default().resolve(uri, config, timeout)?;
        let mut kept = self.empty();
        for addr in found.iter().filter(|a| (self.allow)(a.ip())) {
            kept.push(*addr);
        }
        if kept.is_empty() {
            return Err(ureq::Error::Other(Box::new(Blocked)));
        }
        Ok(kept)
    }
}

/// An HTTP client for research: native TLS against the OS's certificate
/// store (like the app's downloads), no proxy, no redirects of its own, no
/// compression, one overall time limit per request, connecting only to
/// addresses `allow` accepts.
/// The time limit when an agent has none set (they all do).
const FALLBACK_LIMIT: Duration = Duration::from_secs(30);

/// How often a request that's waiting looks at the Stop flag.
const STOP_CHECK: Duration = Duration::from_millis(200);

/// An HTTP client that only connects to addresses `allow` passes, gives each
/// request `time_limit` at most, and gives up waiting once `stop` is set.
pub(super) fn agent(allow: fn(IpAddr) -> bool, time_limit: Duration, stop: &'static AtomicBool) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .accept_encoding("identity")
        .user_agent(user_agent())
        .max_response_header_size(64 * 1024)
        .timeout_global(Some(time_limit))
        .timeout_resolve(Some(Duration::from_secs(5)))
        .timeout_connect(Some(Duration::from_secs(6)))
        // Every request on a connection of its own: a connection's deadline
        // is its request's (see HardDeadline).
        .max_idle_connections(0)
        .max_idle_connections_per_host(0)
        .build();
    // A socket, the deadline over it, then TLS (so a slow handshake is cut
    // off too).
    let connector = ()
        .chain(TcpConnector::default())
        .chain(HardDeadline { stop })
        .chain(NativeTlsConnector::default())
        .chain(RequireTls);
    ureq::Agent::with_parts(config, connector, SafeResolver { allow })
}

/// An `https` address is only ever reached over TLS: should a change to the
/// connectors above ever leave one bare, the request fails instead of
/// going out in plain text.
#[derive(Debug)]
struct RequireTls;

impl<In: Transport> Connector<In> for RequireTls {
    type Out = In;

    fn connect(&self, details: &ConnectionDetails, chained: Option<In>) -> Result<Option<Self::Out>, ureq::Error> {
        if details.needs_tls() && chained.as_ref().is_some_and(|t| !t.is_tls()) {
            return Err(ureq::Error::Tls("an https address without TLS"));
        }
        Ok(chained)
    }
}

/// Ends a connection when its request's time limit is up, however slowly
/// the other end drips bytes. ureq's own limits only bound each wait, and
/// once its deadline has passed a wait may still last a second — so a
/// server sending a byte a second could hold a request for hours. Waits
/// also end soon after `stop` is set. (The limit runs from when the
/// connection is made: looking up the name before it has a limit of its
/// own, a few seconds.)
#[derive(Debug)]
struct HardDeadline {
    stop: &'static AtomicBool,
}

impl<In: Transport> Connector<In> for HardDeadline {
    type Out = Deadlined<In>;

    fn connect(&self, details: &ConnectionDetails, chained: Option<In>) -> Result<Option<Self::Out>, ureq::Error> {
        let start = match details.now {
            time::Instant::Exact(t) => t,
            _ => Instant::now(),
        };
        let limit = details.config.timeouts().global.unwrap_or(FALLBACK_LIMIT);
        Ok(chained.map(|inner| Deadlined { inner, deadline: start + limit, stop: self.stop }))
    }
}

/// A connection that won't wait past `deadline`, or once `stop` is set.
#[derive(Debug)]
struct Deadlined<T> {
    inner: T,
    deadline: Instant,
    stop: &'static AtomicBool,
}

impl<T> Deadlined<T> {
    /// When a wait of `timeout` must end: its own limit or the deadline,
    /// whichever comes first — a timeout error if that's already passed or
    /// Stop was pressed.
    fn until(&self, timeout: &NextTimeout) -> Result<Instant, ureq::Error> {
        let now = Instant::now();
        if now >= self.deadline || self.stop.load(Ordering::Relaxed) {
            return Err(ureq::Error::Timeout(ureq::Timeout::Global));
        }
        let own = now.checked_add(*timeout.after).unwrap_or(self.deadline);
        Ok(own.min(self.deadline))
    }
}

/// A wait of `d` (at least a millisecond: zero would mean none at all).
fn wait(d: Duration, reason: ureq::Timeout) -> NextTimeout {
    NextTimeout { after: time::Duration::Exact(d.max(Duration::from_millis(1))), reason }
}

impl<T: Transport> Transport for Deadlined<T> {
    fn buffers(&mut self) -> &mut dyn Buffers {
        self.inner.buffers()
    }

    fn transmit_output(&mut self, amount: usize, timeout: NextTimeout) -> Result<(), ureq::Error> {
        // Sent in one go (a write cut short can't be taken up again).
        let until = self.until(&timeout)?;
        self.inner.transmit_output(amount, wait(until.saturating_duration_since(Instant::now()), timeout.reason))
    }

    fn await_input(&mut self, timeout: NextTimeout) -> Result<bool, ureq::Error> {
        // Waited for in short slices, to look at the Stop flag between them
        // (a read that times out has taken nothing, so it can be tried again).
        let until = self.until(&timeout)?;
        loop {
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() || self.stop.load(Ordering::Relaxed) {
                return Err(ureq::Error::Timeout(timeout.reason));
            }
            let slice = left.min(STOP_CHECK);
            match self.inner.await_input(wait(slice, timeout.reason)) {
                Err(ureq::Error::Timeout(_)) if slice < left => continue,
                other => return other,
            }
        }
    }

    fn is_open(&mut self) -> bool {
        Instant::now() < self.deadline && self.inner.is_open()
    }

    fn is_tls(&self) -> bool {
        self.inner.is_tls()
    }
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

/// What a request brought back.
pub(super) struct Fetched {
    /// Where it ended up, after redirects.
    pub url: url::Url,
    /// The Content-Type header, lowercased ("" when there was none).
    pub content_type: String,
    pub body: Vec<u8>,
    /// More than [`MAX_BYTES`] came; the rest was dropped.
    pub truncated: bool,
}

/// Why a request failed.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum FetchError {
    Refused(Refused),
    /// The name resolved only to local addresses.
    Blocked,
    Status(u16),
    Timeout,
    TooManyRedirects,
    Network(String),
}

impl FetchError {
    pub(super) fn why(&self) -> String {
        match self {
            FetchError::Refused(r) => r.why().to_string(),
            FetchError::Blocked => Blocked.to_string(),
            FetchError::Status(s) => match *s {
                401 | 403 => format!("the site refused ({s})"),
                404 | 410 => "page not found".to_string(),
                429 => "the site asked to slow down (429)".to_string(),
                _ => format!("the site answered {s}"),
            },
            FetchError::Timeout => "took too long".to_string(),
            FetchError::TooManyRedirects => "too many redirects".to_string(),
            FetchError::Network(e) => e.clone(),
        }
    }
}

/// A ureq error as a [`FetchError`].
pub(super) fn classify(e: ureq::Error) -> FetchError {
    match e {
        ureq::Error::Other(inner) if inner.downcast_ref::<Blocked>().is_some() => FetchError::Blocked,
        ureq::Error::Timeout(_) => FetchError::Timeout,
        ureq::Error::Io(io) if io.kind() == std::io::ErrorKind::TimedOut => FetchError::Timeout,
        ureq::Error::HostNotFound => FetchError::Network("couldn't find the site".to_string()),
        ureq::Error::ConnectionFailed | ureq::Error::Io(_) => FetchError::Network("couldn't connect".to_string()),
        ureq::Error::Tls(_) | ureq::Error::NativeTls(_) => {
            FetchError::Network("the secure connection failed".to_string())
        }
        ureq::Error::BadUri(_) => FetchError::Refused(Refused::Invalid),
        other => {
            let mut s = other.to_string();
            s.truncate(120);
            FetchError::Network(s)
        }
    }
}

/// GET `url` with `headers`, following up to [`MAX_REDIRECTS`] redirects by
/// hand — each new location passes `check` (normally [`check_url`]) before
/// it's visited. Reads at most [`MAX_BYTES`] of the body.
pub(super) fn get(
    agent: &ureq::Agent,
    url: url::Url,
    headers: &[(&str, &str)],
    check: &dyn Fn(&str) -> Result<url::Url, Refused>,
) -> Result<Fetched, FetchError> {
    let mut url = url;
    // One time limit for the whole chain of redirects.
    let start = Instant::now();
    let limit = agent.config().timeouts().global.unwrap_or(FALLBACK_LIMIT);
    for _ in 0..=MAX_REDIRECTS {
        let left = limit.saturating_sub(start.elapsed());
        if left.is_zero() {
            return Err(FetchError::Timeout);
        }
        let mut req = agent.get(url.as_str()).config().timeout_global(Some(left)).build();
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let mut resp = req.call().map_err(classify)?;
        let status = resp.status().as_u16();
        if matches!(status, 301 | 302 | 303 | 307 | 308) {
            let location = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or(FetchError::Status(status))?;
            let next = url.join(location).map_err(|_| FetchError::Refused(Refused::Invalid))?;
            url = check(next.as_str()).map_err(FetchError::Refused)?;
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(FetchError::Status(status));
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let (body, truncated) = read_capped(resp.body_mut().as_reader())?;
        return Ok(Fetched { url, content_type, body, truncated });
    }
    Err(FetchError::TooManyRedirects)
}

/// POST a JSON body (a search API; no redirects). The response body, capped.
pub(super) fn post_json(
    agent: &ureq::Agent,
    url: &url::Url,
    headers: &[(&str, &str)],
    body: &serde_json::Value,
) -> Result<Vec<u8>, FetchError> {
    let mut req = agent.post(url.as_str()).header("Content-Type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let mut resp = req.send(body.to_string()).map_err(classify)?;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(FetchError::Status(status));
    }
    let (body, truncated) = read_capped(resp.body_mut().as_reader())?;
    if truncated {
        return Err(FetchError::Network("the answer was too big".to_string()));
    }
    Ok(body)
}

/// Read at most [`MAX_BYTES`]; true when there was more.
fn read_capped(reader: impl Read) -> Result<(Vec<u8>, bool), FetchError> {
    let mut body = Vec::new();
    reader.take(MAX_BYTES + 1).read_to_end(&mut body).map_err(|e| match e.kind() {
        std::io::ErrorKind::TimedOut => FetchError::Timeout,
        _ => {
            // ureq reports its own timeouts through io errors wrapping one.
            let s = e.to_string();
            if s.to_ascii_lowercase().contains("timeout") || s.to_ascii_lowercase().contains("timed out") {
                FetchError::Timeout
            } else {
                FetchError::Network("the download broke off".to_string())
            }
        }
    })?;
    let truncated = body.len() as u64 > MAX_BYTES;
    body.truncate(MAX_BYTES as usize);
    Ok((body, truncated))
}

/// The body as text: UTF-8 unless the Content-Type (or, without one there,
/// a `<meta charset>` near the top) says Latin-1 / Windows-1252.
pub(super) fn decode(body: &[u8], content_type: &str) -> String {
    let header = charset_of(content_type);
    let charset = header.or_else(|| {
        let head = String::from_utf8_lossy(&body[..body.len().min(2048)]).to_ascii_lowercase();
        let at = head.find("charset=")?;
        charset_of(&head[at..])
    });
    match charset.as_deref() {
        Some("iso-8859-1" | "latin1" | "latin-1" | "windows-1252" | "cp1252" | "us-ascii" | "ascii") => {
            body.iter().map(|&b| cp1252(b)).collect()
        }
        _ => String::from_utf8_lossy(body).into_owned(),
    }
}

/// The `charset=` value in a header-like string.
fn charset_of(s: &str) -> Option<String> {
    let at = s.find("charset=")?;
    let v: String = s[at + 8..]
        .trim_start_matches(['"', '\''])
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    (!v.is_empty()).then(|| v.to_ascii_lowercase())
}

/// One Windows-1252 byte as a char (what browsers do for Latin-1 pages too).
pub(super) fn cp1252(b: u8) -> char {
    const HIGH: [char; 32] = [
        '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}', '\u{90}', '‘',
        '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ',
    ];
    match b {
        0x80..=0x9f => HIGH[(b - 0x80) as usize],
        _ => b as char,
    }
}

/// The site a URL is on, for showing ("en.wikipedia.org"), without "www.".
pub(crate) fn site_of(url: &str) -> String {
    url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.trim_start_matches("www.").to_string()))
        .unwrap_or_default()
}

/// A URL in a comparable form: no fragment, lowercased scheme and host (the
/// `url` crate does those), no trailing slash on the path, no tracking
/// parameters. For telling two search results apart.
pub(super) fn normalized(url: &url::Url) -> String {
    let mut u = url.clone();
    u.set_fragment(None);
    let kept: Vec<(String, String)> = u
        .query_pairs()
        .filter(|(k, _)| !k.starts_with("utm_") && k != "fbclid" && k != "gclid")
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect();
    if kept.is_empty() {
        u.set_query(None);
    } else {
        u.query_pairs_mut().clear().extend_pairs(kept);
    }
    let mut s = u.to_string();
    if s.ends_with('/') && u.path() != "/" {
        s.pop();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(s: &str) -> IpAddr {
        IpAddr::V4(s.parse().unwrap())
    }
    fn v6(s: &str) -> IpAddr {
        IpAddr::V6(s.parse().unwrap())
    }

    #[test]
    fn private_and_special_ipv4_ranges_are_not_public() {
        for ip in [
            "0.0.0.0", "0.1.2.3", "10.0.0.1", "10.255.255.255", "100.64.0.1", "100.127.255.254", "127.0.0.1",
            "127.255.255.255", "169.254.169.254", "169.254.0.1", "172.16.0.1", "172.31.255.255", "192.0.0.1",
            "192.0.2.10", "192.88.99.1", "192.168.0.1", "192.168.255.255", "198.18.0.1", "198.19.255.255",
            "198.51.100.7", "203.0.113.9", "224.0.0.1", "239.255.255.250", "240.0.0.1", "255.255.255.255",
        ] {
            assert!(!is_public(v4(ip)), "{ip} counted as public");
        }
    }

    #[test]
    fn ordinary_ipv4_addresses_are_public() {
        for ip in ["8.8.8.8", "1.1.1.1", "93.184.216.34", "172.15.255.255", "172.32.0.1", "100.63.255.255",
            "100.128.0.1", "192.167.1.1", "192.169.1.1", "198.17.255.255", "198.20.0.1", "223.255.255.254",
            "11.0.0.1", "126.255.255.255", "128.0.0.1", "169.253.1.1", "169.255.1.1"]
        {
            assert!(is_public(v4(ip)), "{ip} counted as local");
        }
    }

    #[test]
    fn special_ipv6_ranges_are_not_public() {
        for ip in [
            "::", "::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:192.168.1.1", "::ffff:169.254.169.254",
            "::127.0.0.1", "64:ff9b::127.0.0.1", "64:ff9b::10.1.1.1", "64:ff9b:1::1", "100::1", "2001::1",
            "2001:0:4136:e378:8000:63bf:3fff:fdd2", "2001:1ff::1", "2001:db8::1", "2002:c0a8:101::1", "3fff::1",
            "3fff:fff::1", "fc00::1", "fd12:3456::1", "fe80::1", "fec0::1", "ff02::1", "5f00::1", "1::1",
        ] {
            assert!(!is_public(v6(ip)), "{ip} counted as public");
        }
    }

    #[test]
    fn global_ipv6_and_mapped_public_v4_are_public() {
        for ip in [
            "2606:4700:4700::1111", "2a00:1450:4001:80b::200e", "2001:4860:4860::8888", "2001:200::1",
            "::ffff:8.8.8.8", "64:ff9b::8.8.8.8", "3fff:1000::1", "2003::1",
        ] {
            assert!(is_public(v6(ip)), "{ip} counted as local");
        }
    }

    #[test]
    fn web_urls_to_public_hosts_pass() {
        for u in [
            "https://en.wikipedia.org/wiki/Rust_(programming_language)",
            "http://example.com/",
            "https://example.com:443/a?b=c#d",
            "http://example.com:80/x",
            "https://8.8.8.8/",
            "https://[2606:4700:4700::1111]/dns",
            "https://bücher.de/",
            "https://news.ycombinator.com./item?id=1",
            "  https://example.org/padded  ",
        ] {
            assert!(check_url(u).is_ok(), "{u} refused: {:?}", check_url(u));
        }
    }

    #[test]
    fn urls_into_this_pc_or_the_lan_are_refused() {
        for u in [
            "http://localhost/",
            "http://localhost./",
            "http://LOCALHOST/",
            "http://app.localhost/",
            "http://127.0.0.1/",
            "http://127.1/",
            "http://2130706433/",
            "http://0x7f000001/",
            "http://0x7f.0.0.1/",
            "http://017700000001/",
            "http://0177.0.0.1/",
            "http://0/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:7f00:1]/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.1/admin",
            "http://10.0.0.5/",
            "http://172.20.0.1/",
            "http://router/",
            "http://nas/",
            "http://printer.local/",
            "http://box.lan/",
            "http://srv.internal/",
            "http://x.home.arpa/",
            "http://1.0.0.127.in-addr.arpa/",
            "http://myhost.localdomain/",
            "http://test.test/",
        ] {
            assert_eq!(check_url(u).err(), Some(Refused::Local), "{u}");
        }
    }

    #[test]
    fn non_web_and_odd_urls_are_refused() {
        assert_eq!(check_url("file:///C:/Windows/win.ini").err(), Some(Refused::NotWeb));
        assert_eq!(check_url("ftp://example.com/x").err(), Some(Refused::NotWeb));
        assert_eq!(check_url("javascript:alert(1)").err(), Some(Refused::NotWeb));
        assert_eq!(check_url("data:text/html,<b>x</b>").err(), Some(Refused::NotWeb));
        assert_eq!(check_url("ms-msdt:/id PCWDiagnostic").err(), Some(Refused::NotWeb));
        assert_eq!(check_url("gopher://example.com/").err(), Some(Refused::NotWeb));
        assert_eq!(check_url("https://user:pw@example.com/").err(), Some(Refused::Credentials));
        assert_eq!(check_url("https://example.com@127.0.0.1/").err(), Some(Refused::Credentials));
        assert_eq!(check_url("https://example.com:8080/").err(), Some(Refused::Port));
        assert_eq!(check_url("http://example.com:443/").err(), Some(Refused::Port));
        assert_eq!(check_url("http://example.com:22/").err(), Some(Refused::Port));
        assert_eq!(check_url("not a url").err(), Some(Refused::Invalid));
        assert_eq!(check_url("").err(), Some(Refused::Invalid));
        assert_eq!(check_url("https://").err(), Some(Refused::Invalid));
        let long = format!("https://example.com/{}", "a".repeat(3000));
        assert_eq!(check_url(&long).err(), Some(Refused::TooLong));
        // A numeric "TLD" isn't a name url could turn into an address.
        assert!(check_url("http://1.2.3.4.5/").is_err());
    }

    #[test]
    fn the_resolver_refuses_names_that_land_on_this_pc() {
        let r = SafeResolver { allow: is_public };
        let config = ureq::config::Config::default();
        let uri: ureq::http::Uri = "http://localhost:80/".parse().unwrap();
        let err = r.resolve(&uri, &config, NextTimeout { after: ureq::unversioned::transport::time::Duration::NotHappening, reason: ureq::Timeout::Resolve }).unwrap_err();
        assert!(matches!(classify(err), FetchError::Blocked));
        // A literal private address too.
        let uri: ureq::http::Uri = "http://192.168.1.1:80/".parse().unwrap();
        let err = r.resolve(&uri, &config, NextTimeout { after: ureq::unversioned::transport::time::Duration::NotHappening, reason: ureq::Timeout::Resolve }).unwrap_err();
        assert!(matches!(classify(err), FetchError::Blocked));
    }

    #[test]
    fn text_decodes_utf8_and_windows_1252() {
        assert_eq!(decode("café — ok".as_bytes(), "text/html; charset=utf-8"), "café — ok");
        assert_eq!(decode(&[0x63, 0x61, 0x66, 0xe9, 0x20, 0x93, 0x71, 0x94], "text/html; charset=ISO-8859-1".to_ascii_lowercase().as_str()), "café “q”");
        let meta = b"<html><head><meta charset=\"windows-1252\"></head><body>\x80 5</body>";
        assert!(decode(meta, "text/html").contains("€ 5"));
        // Broken UTF-8 never panics.
        assert_eq!(decode(&[0xff, 0xfe, b'a'], ""), "\u{fffd}\u{fffd}a");
    }

    #[test]
    fn urls_normalize_for_deduplication() {
        let n = |s: &str| normalized(&url::Url::parse(s).unwrap());
        assert_eq!(n("https://Example.com/a/?utm_source=x#frag"), n("https://example.com/a"));
        assert_eq!(n("https://example.com/?q=1&utm_medium=y"), "https://example.com/?q=1");
        assert_ne!(n("https://example.com/a?id=1"), n("https://example.com/a?id=2"));
        assert_eq!(site_of("https://www.bbc.co.uk/news"), "bbc.co.uk");
    }
}
