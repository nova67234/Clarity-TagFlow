//! The Downloader panel — a multi-source media downloader.
//!
//! Sources: **Pexels** (default; stock photos, API-key auth), **Gelbooru**
//! (the original, ported from terminus2's `Gelbooru.java`), **Danbooru** and
//! **Wallhaven** — each source's API specifics live in its own module
//! (`pexels.rs` / `danbooru.rs` / `wallhaven.rs`); this module owns the shared
//! form UI, per-source credentials (encrypted at rest), the background worker
//! that streams files to disk with `.txt` sidecars where the source provides
//! tags/captions, and the de-duplication log so re-runs skip files already
//! pulled. Gelbooru keeps its original specialised worker (daily quota +
//! artist/character tag-role resolution).

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::theme::{ACCENT1, EDGE, FIELD, FIELD2, MUTED, PANEL, TEXT};

const API_URL: &str = "https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1";
/// Tag-info endpoint: returns each tag's `type` (0 general, 1 artist, 3 copyright,
/// 4 character, 5 metadata). Used to split a post's flat tag list into artist /
/// character roles for the `{md5}.json` sidecar.
const TAG_API_URL: &str = "https://gelbooru.com/index.php?page=dapi&s=tag&q=index&json=1";
const SITE_HOME: &str = "https://gelbooru.com/";
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
     (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

const PAGE_LIMIT: u32 = 100;
const MAX_TRANSIENT_RETRIES: u32 = 5;

/// Minimum delay (seconds) between downloads. Enforced everywhere — the user
/// can't go below this — to avoid hammering Gelbooru and getting rate-limited.
const MIN_DELAY: f32 = 3.0;

/// Maximum files a user may download per calendar day. A courtesy guard-rail so
/// the app can't be used (or accidentally left running) to mass-pull from
/// Gelbooru. The running count is kept in an *encrypted* file (DPAPI, same as the
/// API key) so it can't simply be edited back down — though it's a soft limit:
/// deleting the file or changing the system clock resets it.
const DAILY_CAP: u32 = 2000;

/// Full height of the Activity console's scrollable content when open.
const CONSOLE_H: f32 = 120.0;

const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "webp"];
const VIDEO_EXTS: &[&str] = &["mp4", "webm", "avi"];
const GIF_EXTS: &[&str] = &["gif"];

/// Messages the background download thread sends back to the UI.
enum DlMsg {
    Log(String),
    /// (downloaded so far, target total)
    Progress(u32, u32),
    Done,
}

/// One downloadable item from any source, in source-neutral form. Produced by
/// the per-source `parse` functions, consumed by the shared worker.
pub(crate) struct Item {
    /// Direct file URL.
    pub url: String,
    /// Stable de-duplication key recorded in the download log
    /// (e.g. `"pexels:12345"`).
    pub key: String,
    /// Output file stem; the extension is appended from `ext`.
    pub stem: String,
    /// Lower-case file extension (`"jpg"`, …).
    pub ext: String,
    /// Booru tag string or a human caption, written to the `.txt` sidecar
    /// when present.
    pub tags: Option<String>,
}

/// The selectable download sources. Pexels is the default.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Source {
    #[default]
    Pexels,
    Pixabay,
    Gelbooru,
    Safebooru,
    Danbooru,
    Wallhaven,
}

impl Source {
    const ALL: [Source; 6] = [
        Source::Pexels,
        Source::Pixabay,
        Source::Gelbooru,
        Source::Safebooru,
        Source::Danbooru,
        Source::Wallhaven,
    ];

    fn name(self) -> &'static str {
        match self {
            Source::Pexels => crate::pexels::NAME,
            Source::Pixabay => crate::pixabay::NAME,
            Source::Gelbooru => "Gelbooru",
            Source::Safebooru => crate::safebooru::NAME,
            Source::Danbooru => crate::danbooru::NAME,
            Source::Wallhaven => crate::wallhaven::NAME,
        }
    }

    fn subtitle(self) -> &'static str {
        match self {
            Source::Pexels => crate::pexels::SUBTITLE,
            Source::Pixabay => crate::pixabay::SUBTITLE,
            Source::Gelbooru => "Tag Downloader",
            Source::Safebooru => crate::safebooru::SUBTITLE,
            Source::Danbooru => crate::danbooru::SUBTITLE,
            Source::Wallhaven => crate::wallhaven::SUBTITLE,
        }
    }

    fn home(self) -> &'static str {
        match self {
            Source::Pexels => crate::pexels::HOME,
            Source::Pixabay => crate::pixabay::HOME,
            Source::Gelbooru => SITE_HOME,
            Source::Safebooru => crate::safebooru::HOME,
            Source::Danbooru => crate::danbooru::HOME,
            Source::Wallhaven => crate::wallhaven::HOME,
        }
    }

    fn cred_url(self) -> &'static str {
        match self {
            Source::Pexels => crate::pexels::CRED_URL,
            Source::Pixabay => crate::pixabay::CRED_URL,
            Source::Gelbooru => "https://gelbooru.com/index.php?page=account&s=options",
            // No credential system — never shown (the Account card is hidden).
            Source::Safebooru => crate::safebooru::HOME,
            Source::Danbooru => crate::danbooru::CRED_URL,
            Source::Wallhaven => crate::wallhaven::CRED_URL,
        }
    }

    fn cred_info(self) -> &'static str {
        match self {
            Source::Pexels => crate::pexels::CRED_INFO,
            Source::Pixabay => crate::pixabay::CRED_INFO,
            Source::Gelbooru => {
                "Gelbooru no longer allows anonymous downloads — you must log in with \
                 your account's User ID and API key. Get them from gelbooru.com → \
                 Account → Options → 'API Access Credentials' (free account required)."
            }
            Source::Safebooru => crate::safebooru::INFO,
            Source::Danbooru => crate::danbooru::CRED_INFO,
            Source::Wallhaven => crate::wallhaven::CRED_INFO,
        }
    }

    fn tags_hint(self) -> &'static str {
        match self {
            Source::Pexels => crate::pexels::TAGS_HINT,
            Source::Pixabay => crate::pixabay::TAGS_HINT,
            Source::Gelbooru => "space-separated, e.g. blue_sky 1girl",
            Source::Safebooru => crate::safebooru::TAGS_HINT,
            Source::Danbooru => crate::danbooru::TAGS_HINT,
            Source::Wallhaven => crate::wallhaven::TAGS_HINT,
        }
    }

    /// Label of the username field, for sources that pair one with the key.
    fn user_field(self) -> Option<&'static str> {
        match self {
            Source::Gelbooru => Some("User ID"),
            Source::Danbooru => Some("Login"),
            Source::Pexels | Source::Pixabay | Source::Safebooru | Source::Wallhaven => None,
        }
    }

    /// Whether the API key is mandatory (vs an optional account upgrade).
    fn key_required(self) -> bool {
        matches!(self, Source::Pexels | Source::Pixabay | Source::Gelbooru)
    }

    /// Whether the source has any credential system at all — Safebooru's API
    /// is fully anonymous, so the Account card is skipped for it.
    fn has_account(self) -> bool {
        !matches!(self, Source::Safebooru)
    }

    /// Booru-style sources: per-post tag strings (comma-formatted sidecars,
    /// tag-search syntax).
    fn is_booru(self) -> bool {
        matches!(self, Source::Gelbooru | Source::Safebooru | Source::Danbooru)
    }

    /// Whether the Options card shows file-type chips at all (Wallhaven is
    /// wallpapers-only, so there's nothing to choose).
    fn has_type_chips(self) -> bool {
        !matches!(self, Source::Wallhaven)
    }

    /// Whether the source can return GIFs (booru posts only).
    fn offers_gif(self) -> bool {
        self.is_booru()
    }

    /// Whether the source can return videos — the boorus mix them into posts,
    /// Pexels and Pixabay serve them from their separate videos endpoints.
    fn offers_video(self) -> bool {
        self.is_booru() || matches!(self, Source::Pexels | Source::Pixabay)
    }

    /// Per-source blacklist hint — the mechanism differs (tag match, caption
    /// match, or query-level exclusion) even though the field is shared.
    fn blacklist_hint(self) -> &'static str {
        match self {
            Source::Pexels => "comma-separated words to skip (matched against the description)",
            Source::Pixabay => "comma-separated words to skip (matched against the tags)",
            Source::Wallhaven => "comma-separated tags to exclude from the search",
            Source::Gelbooru | Source::Safebooru | Source::Danbooru => {
                "comma-separated tags to skip"
            }
        }
    }

    /// Stable key for the saved config.
    fn config_key(self) -> &'static str {
        match self {
            Source::Pexels => "pexels",
            Source::Pixabay => "pixabay",
            Source::Gelbooru => "gelbooru",
            Source::Safebooru => "safebooru",
            Source::Danbooru => "danbooru",
            Source::Wallhaven => "wallhaven",
        }
    }

    fn from_config_key(s: &str) -> Source {
        match s {
            "pixabay" => Source::Pixabay,
            "gelbooru" => Source::Gelbooru,
            "safebooru" => Source::Safebooru,
            "danbooru" => Source::Danbooru,
            "wallhaven" => Source::Wallhaven,
            _ => Source::Pexels,
        }
    }

    /// Self-imposed daily download allowance — a courtesy cap in the spirit
    /// of each service's own limits, tracked in an encrypted per-source file.
    fn daily_cap(self) -> u32 {
        match self {
            // Pexels caps API requests (200/hour, 20 000/month; one request
            // lists 80 photos), not downloads — 600/day keeps a full month of
            // daily runs comfortably inside the free tier.
            Source::Pexels => 600,
            // Pixabay allows 100 API requests/minute (one lists 200 hits) but
            // frowns on automated mass-downloading — same order of restraint.
            Source::Pixabay => 600,
            Source::Gelbooru | Source::Safebooru | Source::Danbooru => DAILY_CAP,
            // Wallhaven rate-limits the API at 45 requests/minute.
            Source::Wallhaven => 1000,
        }
    }

    /// Hover explanation for the allowance meter.
    fn cap_note(self) -> &'static str {
        match self {
            Source::Pexels => {
                "Pexels doesn't cap downloads per day — its free tier caps API requests \
                 (200/hour, 20,000/month; one request lists 80 photos). A 600/day allowance \
                 keeps a full month of daily runs comfortably inside the free tier."
            }
            Source::Pixabay => {
                "Pixabay allows 100 API requests per minute, but asks that content not be \
                 mass-downloaded — a 600/day allowance keeps runs to a considerate scale."
            }
            Source::Gelbooru => {
                "A courtesy guard-rail so the app can't be used (or accidentally left \
                 running) to mass-pull from Gelbooru."
            }
            Source::Safebooru => {
                "A courtesy guard-rail matching the Gelbooru cap — Safebooru runs the \
                 same engine and asks automated clients for the same restraint."
            }
            Source::Danbooru => {
                "A courtesy guard-rail matching the Gelbooru cap — Danbooru asks bots for \
                 restraint (about one request per second, sustained)."
            }
            Source::Wallhaven => {
                "A courtesy guard-rail — Wallhaven rate-limits its API at 45 requests per \
                 minute and asks for considerate use."
            }
        }
    }

    /// Index into [`Source::ALL`] — how the UI tells the API monitor thread
    /// which site to ping.
    fn index(self) -> u8 {
        Source::ALL.iter().position(|s| *s == self).unwrap_or(0) as u8
    }

    fn from_index(i: u8) -> Source {
        Source::ALL.get(i as usize).copied().unwrap_or(Source::Pexels)
    }
}

/// Persisted-across-runs form values (credentials + last inputs). The
/// unprefixed `user_id`/`api_key` are Gelbooru's (pre-multi-source configs).
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SavedConfig {
    #[serde(default)]
    user_id: String,
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    tags: String,
    #[serde(default)]
    blacklist: String,
    #[serde(default)]
    output_dir: String,
    /// The Activity console is shown (toggled by the Activity button; hidden
    /// by default).
    #[serde(default)]
    show_log: bool,
    /// The selected source's [`Source::config_key`]; absent/unknown = Pexels.
    #[serde(default)]
    source: String,
    #[serde(default)]
    pexels_key: String,
    #[serde(default)]
    pixabay_key: String,
    #[serde(default)]
    danbooru_login: String,
    #[serde(default)]
    danbooru_key: String,
    #[serde(default)]
    wallhaven_key: String,
    /// Wallhaven's search-filter selections (categories/purity/sorting/…).
    #[serde(default)]
    wallhaven_opts: crate::wallhaven::Opts,
}

/// All UI + runtime state for the downloader view. Lives on `RightPanelState`.
pub struct DownloaderState {
    /// The selected download source (persisted; Pexels by default).
    source: Source,
    output_dir: String,
    /// Gelbooru credentials (the unprefixed pair, matching older configs).
    user_id: String,
    api_key: String,
    pexels_key: String,
    pixabay_key: String,
    danbooru_login: String,
    danbooru_key: String,
    wallhaven_key: String,
    /// Wallhaven's search-filter selections (its Options card).
    wh: crate::wallhaven::Opts,
    tags: String,
    blacklist: String,
    limit: u32,
    delay: f32,
    include_img: bool,
    include_gif: bool,
    include_vid: bool,

    /// Rolling log lines shown in the console box.
    log: Vec<String>,
    /// `(done, total)` for the progress bar; `total == 0` means idle.
    progress: (u32, u32),
    status: String,

    /// `true` while a download thread is active.
    running: bool,
    /// Flipped to request the running thread stop.
    cancel: Arc<AtomicBool>,
    rx: Option<Receiver<DlMsg>>,

    /// Loaded once so credentials populate on first show.
    loaded: bool,

    /// Gelbooru reachability: 0 = checking, 1 = online, 2 = offline. Updated by a
    /// background monitor thread, read each frame to draw the status pill.
    api_status: Arc<AtomicU8>,
    /// Whether the monitor thread has been spawned (only once per session).
    monitor_started: bool,
    /// Flipped true every frame the view renders; the API monitor only pings
    /// while it keeps getting set, so leaving the view pauses the polling
    /// (matching the Civitai panel's monitor).
    api_view_visible: Arc<AtomicBool>,
    /// [`Source::index`] of the source the monitor should ping (updated each
    /// frame from the selection).
    api_source: Arc<AtomicU8>,

    /// In-flight account/key check (the "Check account" button): a one-shot
    /// worker that makes a single authenticated call and reports back.
    key_check_rx: Option<Receiver<(Source, bool, String)>>,
    /// The last check's (source, ok, message) — shown while that source is
    /// selected.
    key_check_result: Option<(Source, bool, String)>,

    /// Cached (source, used-today, read-at) for the daily-allowance meter —
    /// the count lives in an encrypted file, so it's re-read at most once a
    /// second instead of every frame (and re-read on source switch).
    quota_cache: Option<(Source, u32, std::time::Instant)>,

    /// The Activity console is shown (toggled by the Activity button). Hidden
    /// by default; the choice persists in the saved config.
    log_shown: bool,
    /// Measured height of the footer minus the animated console block, so the
    /// form can reserve exactly the right space same-frame (no nested panels).
    footer_base: f32,
}

/// API-status codes shared with the monitor thread.
const API_CHECKING: u8 = 0;
const API_ONLINE: u8 = 1;
const API_OFFLINE: u8 = 2;

impl Default for DownloaderState {
    fn default() -> Self {
        Self {
            source: Source::Pexels,
            output_dir: String::new(),
            user_id: String::new(),
            api_key: String::new(),
            pexels_key: String::new(),
            pixabay_key: String::new(),
            danbooru_login: String::new(),
            danbooru_key: String::new(),
            wallhaven_key: String::new(),
            wh: crate::wallhaven::Opts::default(),
            tags: "example_tag".to_string(),
            blacklist: String::new(),
            limit: 100,
            delay: MIN_DELAY,
            include_img: true,
            include_gif: false,
            include_vid: false,
            log: Vec::new(),
            progress: (0, 0),
            status: "Idle".to_string(),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            loaded: false,
            api_status: Arc::new(AtomicU8::new(API_CHECKING)),
            monitor_started: false,
            api_view_visible: Arc::new(AtomicBool::new(false)),
            api_source: Arc::new(AtomicU8::new(0)),
            key_check_rx: None,
            key_check_result: None,
            quota_cache: None,
            log_shown: false,
            footer_base: 78.0,
        }
    }
}

impl DownloaderState {
    /// Snapshot the persisted form values (all sources' credentials included).
    fn saved(&self) -> SavedConfig {
        SavedConfig {
            user_id: self.user_id.trim().to_string(),
            api_key: self.api_key.trim().to_string(),
            tags: self.tags.clone(),
            blacklist: self.blacklist.clone(),
            output_dir: self.output_dir.clone(),
            show_log: self.log_shown,
            source: self.source.config_key().to_string(),
            pexels_key: self.pexels_key.trim().to_string(),
            pixabay_key: self.pixabay_key.trim().to_string(),
            danbooru_login: self.danbooru_login.trim().to_string(),
            danbooru_key: self.danbooru_key.trim().to_string(),
            wallhaven_key: self.wallhaven_key.trim().to_string(),
            wallhaven_opts: self.wh.clone(),
        }
    }

    /// The selected source's (user, key) credentials.
    fn creds(&self) -> (String, String) {
        match self.source {
            Source::Pexels => (String::new(), self.pexels_key.trim().to_string()),
            Source::Pixabay => (String::new(), self.pixabay_key.trim().to_string()),
            Source::Gelbooru => (self.user_id.trim().to_string(), self.api_key.trim().to_string()),
            Source::Safebooru => (String::new(), String::new()),
            Source::Danbooru => (self.danbooru_login.trim().to_string(), self.danbooru_key.trim().to_string()),
            Source::Wallhaven => (String::new(), self.wallhaven_key.trim().to_string()),
        }
    }

    /// Today's used download count for the selected source's allowance meter,
    /// re-read from disk at most once a second (or on source switch).
    fn quota_today(&mut self) -> u32 {
        let stale = self
            .quota_cache
            .is_none_or(|(src, _, at)| src != self.source || at.elapsed().as_secs_f32() > 1.0);
        if stale {
            self.quota_cache =
                Some((self.source, quota_used_today(self.source), std::time::Instant::now()));
        }
        self.quota_cache.map_or(0, |(_, n, _)| n)
    }

    fn push_log(&mut self, line: impl Into<String>) {
        self.log.push(line.into());
        // Cap the in-memory log so a long run can't grow unbounded.
        if self.log.len() > 1000 {
            let overflow = self.log.len() - 1000;
            self.log.drain(0..overflow);
        }
    }
}

/// Render the downloader view. Drains background messages, draws the form, and
/// starts / cancels the worker thread.
pub fn show(ui: &mut egui::Ui, state: &mut DownloaderState) {
    if !state.loaded {
        state.loaded = true;
        if let Some(cfg) = load_config() {
            state.user_id = cfg.user_id;
            state.api_key = cfg.api_key;
            if !cfg.tags.is_empty() {
                state.tags = cfg.tags;
            }
            state.blacklist = cfg.blacklist;
            state.output_dir = cfg.output_dir;
            state.log_shown = cfg.show_log;
            state.source = Source::from_config_key(&cfg.source);
            state.pexels_key = cfg.pexels_key;
            state.pixabay_key = cfg.pixabay_key;
            state.danbooru_login = cfg.danbooru_login;
            state.danbooru_key = cfg.danbooru_key;
            state.wallhaven_key = cfg.wallhaven_key;
            state.wh = cfg.wallhaven_opts;
        }
        // Never let a persisted/old value drop below the safety floor.
        if state.delay < MIN_DELAY {
            state.delay = MIN_DELAY;
        }
    }

    // Mark the view visible this frame — the monitor pauses when this stops
    // getting set (i.e. when another right-panel view is selected) — and keep
    // it pointed at the selected source's site.
    state.api_view_visible.store(true, Ordering::Relaxed);
    state.api_source.store(state.source.index(), Ordering::Relaxed);
    // Spawn the API-status monitor once: it polls the source's homepage every
    // few seconds and updates `api_status`, which drives the hero's capsule.
    if !state.monitor_started {
        state.monitor_started = true;
        start_api_monitor(
            Arc::clone(&state.api_status),
            Arc::clone(&state.api_view_visible),
            Arc::clone(&state.api_source),
            ui.ctx().clone(),
        );
    }

    // Drain any messages from the worker.
    if let Some(rx) = &state.rx {
        let mut msgs = Vec::new();
        while let Ok(m) = rx.try_recv() {
            msgs.push(m);
        }
        for m in msgs {
            match m {
                DlMsg::Log(line) => state.push_log(line),
                DlMsg::Progress(done, total) => {
                    state.progress = (done, total);
                    // The count is shown by the percentage label in the Log header,
                    // so keep the status word generic rather than duplicating it.
                    state.status = "Downloading".to_string();
                }
                DlMsg::Done => {
                    state.running = false;
                    state.rx = None;
                    // A cancel leaves the status at the in-progress "Cancelling…";
                    // flip it to the finished "Cancelled" so it doesn't look stuck.
                    if state.status.starts_with("Cancel") {
                        state.status = "Cancelled".to_string();
                    } else {
                        state.status = "Done".to_string();
                    }
                }
            }
        }
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }

    // Drain a finished account/key check.
    if let Some(rx) = &state.key_check_rx {
        match rx.try_recv() {
            Ok(res) => {
                state.key_check_result = Some(res);
                state.key_check_rx = None;
            }
            Err(mpsc::TryRecvError::Empty) => {
                ui.ctx().request_repaint_after(Duration::from_millis(200));
            }
            Err(mpsc::TryRecvError::Disconnected) => state.key_check_rx = None,
        }
    }

    // Round every widget in this view; text fields get the theme FIELD well
    // via `field_edit` so they read against the PANEL section cards.
    let radius = egui::CornerRadius::same(10);
    {
        let v = ui.visuals_mut();
        v.widgets.inactive.corner_radius = radius;
        v.widgets.hovered.corner_radius = radius;
        v.widgets.active.corner_radius = radius;
        v.widgets.noninteractive.corner_radius = radius;
        v.widgets.open.corner_radius = radius;
    }

    // Hero header: identity badge + status capsule + daily-allowance meter.
    ui.add_space(2.0);
    hero_card(ui, state);
    ui.add_space(2.0);

    // Adult-capable selections (the boorus with NSFW content; Wallhaven only
    // once Sketchy/NSFW purity is switched on) need the one-time 18+
    // confirmation. Declining falls back to a safe selection; the form stays
    // disabled while the dialog is up.
    let adult = matches!(state.source, Source::Gelbooru | Source::Danbooru)
        || (state.source == Source::Wallhaven && (state.wh.pur_sketchy || state.wh.pur_nsfw));
    let mut gated = adult && !crate::age_gate::acknowledged();
    if gated {
        match crate::age_gate::modal(ui.ctx(), "downloader_age_gate") {
            Some(true) => gated = false,
            Some(false) => {
                if state.source == Source::Wallhaven {
                    state.wh.pur_sketchy = false;
                    state.wh.pur_nsfw = false;
                } else {
                    state.source = Source::Pexels;
                }
                save_config(&state.saved());
                gated = false;
            }
            None => {}
        }
    }

    let enabled = !state.running && !gated;

    // No nested panels here (the Generate views lay out plainly and never
    // glitch): a bottom panel's content overflowing its one-frame-stale height
    // bubbles `expand_to_include_rect` up to the ROOT ui, which briefly
    // inflated the centre viewer — the flicker on the Activity toggle. Instead
    // the form scroll area takes exactly the height the footer doesn't need,
    // computed same-frame (the console's animated height is deterministic and
    // the rest of the footer is a measured constant).
    let openness = ui.ctx().animate_bool(ui.id().with("dl_log_open"), state.log_shown);
    if openness > 0.0 && openness < 1.0 {
        ui.ctx().request_repaint(); // keep the console slide animating
    }
    // Console block: 2 top gap + animated well (content + 2×10 inner margin)
    // + the item-spacing the extra widget introduces.
    let console_h = if openness > 0.0 {
        2.0 + CONSOLE_H * openness + 20.0 + ui.spacing().item_spacing.y
    } else {
        0.0
    };
    let form_h = (ui.available_height() - state.footer_base - console_h).max(80.0);

    // Form — scrolls if it's too tall. The scrollbar is pushed into the card's
    // right margin so it rides the panel edge instead of sitting on the
    // controls (same treatment as the gallery).
    const SCROLL_GUTTER: f32 = 12.0;
    let mut scroll_ui = crate::edge_scroll_ui(ui, SCROLL_GUTTER);
    egui::ScrollArea::vertical()
        .id_salt("dl_form")
        .max_height(form_h)
        .min_scrolled_height(form_h)
        .auto_shrink([false, false])
        .show(&mut scroll_ui, |ui| {
            form_sections(ui, state, enabled);
        });
    crate::edge_scroll_done(ui, &scroll_ui, SCROLL_GUTTER);

    // --- Footer: Activity toggle, the sliding console, the action button ---
    // Rendered into a child pinned to exactly the remaining card space: any
    // sub-pixel estimate drift clips inside this region instead of expanding
    // the parent (which would bubble up and nudge the centre viewer).
    let footer_rect = egui::Rect::from_min_max(ui.cursor().min, ui.max_rect().max);
    let parent_clip = ui.clip_rect();
    let ui = &mut ui.new_child(
        egui::UiBuilder::new()
            .max_rect(footer_rect)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    ui.set_clip_rect(footer_rect.intersect(parent_clip));
    let footer_top = ui.cursor().min.y;
    {
            ui.add_space(8.0);

            // Console / log — an inset well, hidden by default. The Activity
            // button reveals it (extending the footer upward) and hides it
            // again; the choice persists in the saved config.
            ui.horizontal(|ui| {
                let chev = if state.log_shown {
                    egui::include_image!("../icons/arrow_down.svg")
                } else {
                    egui::include_image!("../icons/arrow_up.svg")
                };
                let btn = egui::Button::image_and_text(
                    egui::Image::new(chev)
                        .fit_to_exact_size(egui::vec2(12.0, 12.0))
                        .tint(crate::theme::icon_tint(MUTED())),
                    egui::RichText::new("Activity").color(MUTED()).size(12.0),
                )
                .frame(false);
                if ui.add(btn).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    state.log_shown = !state.log_shown;
                    save_config(&state.saved());
                }
            });
            // The console slides open/closed (`openness`, computed above so
            // the form could reserve space for it this same frame).
            if openness > 0.0 {
                let log_h = CONSOLE_H * openness;
                ui.add_space(2.0);
                egui::Frame::new()
                    .fill(crate::theme::console_bg())
                    .corner_radius(egui::CornerRadius::same(22))
                    .inner_margin(egui::Margin::same(10))
                    .stroke(egui::Stroke::new(1.0, EDGE()))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        egui::ScrollArea::vertical()
                            .id_salt("dl_log")
                            .max_height(log_h)
                            .min_scrolled_height(log_h)
                            .auto_shrink([false, false])
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                // Keep scrolling while a text selection is dragged past the edge.
                                crate::drag_select_autoscroll(ui);
                                if state.log.is_empty() {
                                    ui.label(
                                        egui::RichText::new("Log output will appear here.")
                                            .color(MUTED())
                                            .monospace()
                                            .size(12.0),
                                    );
                                } else {
                                    for line in &state.log {
                                        ui.label(egui::RichText::new(line).color(TEXT()).monospace().size(12.0));
                                    }
                                }
                            });
                    });
            }

            ui.add_space(10.0);

            // One full-width action button that morphs: accent "Start Download"
            // while idle, red "Cancel" while a run is active.
            let size = egui::vec2(ui.available_width(), 40.0);
            let (label, fill) = if state.running {
                ("Cancel", egui::Color32::from_rgb(180, 40, 40))
            } else {
                ("Start Download", ACCENT1())
            };
            let btn = egui::Button::new(egui::RichText::new(label).color(egui::Color32::WHITE).strong())
                .fill(fill)
                .corner_radius(egui::CornerRadius::same(14));
            if ui.add_sized(size, btn).clicked() {
                if state.running {
                    state.cancel.store(true, Ordering::SeqCst);
                    state.status = "Cancelling…".to_string();
                    state.push_log("Cancel requested…");
                } else {
                    start_download(state, ui.ctx());
                }
            }
            ui.add_space(2.0);
    }

    // Remember the footer's constant height (everything except the animated
    // console block) so next frame's form sizing stays exact — the same
    // measure-and-repaint trick the Generate views use for their prompt box.
    let footer_base = (ui.cursor().min.y - footer_top) - console_h;
    if (footer_base - state.footer_base).abs() > 0.5 {
        state.footer_base = footer_base;
        ui.ctx().request_repaint();
    }
}

/// The scrollable form: Destination / Account / Search / Options cards.
fn form_sections(ui: &mut egui::Ui, state: &mut DownloaderState, enabled: bool) {
                    ui.set_max_width(ui.available_width() - 12.0);
                    let blue = egui::Color32::from_rgb(90, 150, 230);
                    let purple = egui::Color32::from_rgb(150, 120, 220);
                    let orange = egui::Color32::from_rgb(232, 160, 60);
                    let slate = egui::Color32::from_rgb(130, 140, 160);
                    section_card(ui, egui::include_image!("../icons/folder.svg"), blue, "Destination", None, |ui| {
                        field_label(ui, "Output folder");
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            let folder_svg = egui::include_image!("../icons/folder.svg");
                            if crate::svg_button(ui, folder_svg, "Choose output folder", 34.0, crate::theme::icon_tint(MUTED()))
                                .clicked()
                                && let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                    state.output_dir = dir.display().to_string();
                                }
                            field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.output_dir)
                                .hint_text("Where files are saved"));
                        });
                    });

                    // Per-source credentials — which fields show, whether the
                    // key is mandatory, and where to get one all come from the
                    // selected source (info via the ⓘ next to the title).
                    // Fully anonymous sources (Safebooru) have no card at all.
                    let source = state.source;
                    if source.has_account() {
                    section_card(ui, egui::include_image!("../icons/encrypted.svg"), purple, "Account",
                        Some(source.cred_info()),
                        |ui| {
                        if let Some(user_label) = source.user_field() {
                            field_label(ui, user_label);
                            let user = match source {
                                Source::Danbooru => &mut state.danbooru_login,
                                _ => &mut state.user_id,
                            };
                            field_edit(ui, enabled, egui::TextEdit::singleline(user)
                                .hint_text(if source.key_required() { "Required" } else { "Optional" }));
                            ui.add_space(8.0);
                        }
                        field_label(ui, "API key");
                        let key = match source {
                            Source::Pexels => &mut state.pexels_key,
                            Source::Pixabay => &mut state.pixabay_key,
                            Source::Gelbooru => &mut state.api_key,
                            Source::Danbooru => &mut state.danbooru_key,
                            Source::Wallhaven => &mut state.wallhaven_key,
                            // Card hidden for accountless sources — harmless filler.
                            Source::Safebooru => &mut state.api_key,
                        };
                        let key_hint = if source.key_required() {
                            format!("Paste your {} API key", source.name())
                        } else {
                            format!("Optional — {} works without one", source.name())
                        };
                        field_edit(ui, enabled, egui::TextEdit::singleline(key)
                            .password(true)
                            .hint_text(key_hint));
                        ui.add_space(3.0);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("Stored encrypted on this device.")
                                    .color(MUTED())
                                    .size(10.5),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                crate::arrow_link(ui, "Get credentials", source.cred_url(), Some(10.5));
                            });
                        });

                        // Account check: one authenticated call that verifies
                        // the key — Danbooru's profile even reports ban status.
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            let checking = state.key_check_rx.is_some();
                            let (user, key) = state.creds();
                            let can_check = !checking
                                && !key.is_empty()
                                && (source.user_field().is_none() || !user.is_empty());
                            let btn = egui::Button::new(egui::RichText::new("Check account").size(11.0));
                            if ui
                                .add_enabled(can_check, btn)
                                .on_hover_text(
                                    "Makes one authenticated API call to verify the key. \
                                     Danbooru also reports whether the account is banned; the \
                                     other sources can only tell accepted vs rejected.",
                                )
                                .clicked()
                            {
                                start_key_check(state, ui.ctx());
                            }
                            if checking {
                                ui.add(egui::Spinner::new().size(12.0).color(MUTED()));
                            }
                        });
                        if let Some((src, ok, msg)) = &state.key_check_result
                            && *src == source
                        {
                            let color = if *ok {
                                egui::Color32::from_rgb(46, 160, 67)
                            } else {
                                egui::Color32::from_rgb(210, 70, 70)
                            };
                            ui.add_space(3.0);
                            ui.label(egui::RichText::new(msg).color(color).size(10.5));
                        }
                    });
                    } // if source.has_account()

                    section_card(ui, egui::include_image!("../icons/tag.svg"), orange, "Search", None, |ui| {
                        field_label(ui, if source.is_booru() { "Tags" } else { "Search" });
                        field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.tags)
                            .hint_text(source.tags_hint()));
                        ui.add_space(8.0);
                        field_label(ui, "Blacklist");
                        field_edit(ui, enabled, egui::TextEdit::singleline(&mut state.blacklist)
                            .hint_text(source.blacklist_hint()));
                    });

                    section_card(ui, egui::include_image!("../icons/settings.svg"), slate, "Options", None, |ui| {
                        // The right edge is reserved for the status readout; the
                        // controls keep to its left.
                        const STAT_W: f32 = 74.0;
                        let body_right = ui.max_rect().right();
                        let top = ui.cursor().min.y;
                        ui.scope(|ui| {
                            ui.set_max_width(ui.available_width() - STAT_W - 8.0);
                            ui.horizontal(|ui| {
                                ui.label(egui::RichText::new("Limit").color(TEXT()));
                                ui.add_enabled(
                                    enabled,
                                    egui::DragValue::new(&mut state.limit).range(1..=10000).speed(1.0),
                                );
                                ui.add_space(20.0);
                                ui.label(egui::RichText::new("Delay (s)").color(TEXT()));
                                ui.add_enabled(
                                    enabled,
                                    egui::DragValue::new(&mut state.delay).range(MIN_DELAY..=60.0).speed(0.1),
                                );
                                // Info icon explaining the enforced minimum delay.
                                info_icon(
                                    ui,
                                    "The delay is the wait between downloads. It can't go below 3 \
                                     seconds: Gelbooru rate-limits frequent requests, so a shorter \
                                     delay risks being throttled or temporarily blocked.",
                                );
                            });
                            // Chips per capability: boorus mix all three,
                            // Pexels has photos + a separate videos API, and
                            // Wallhaven (wallpapers-only) gets its own search
                            // filters instead of file-type chips.
                            if source.has_type_chips() {
                                ui.add_space(8.0);
                                field_label(ui, "File types");
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    ui.add_enabled_ui(enabled, |ui| {
                                        type_chip(ui, &mut state.include_img, egui::include_image!("../icons/image.svg"), "Image");
                                        if source.offers_gif() {
                                            type_chip(ui, &mut state.include_gif, egui::include_image!("../icons/gif.svg"), "Gif");
                                        }
                                        if source.offers_video() {
                                            type_chip(ui, &mut state.include_vid, egui::include_image!("../icons/video.svg"), "Video");
                                        }
                                    });
                                });
                            } else {
                                wallhaven_options(ui, enabled, &mut state.wh);
                            }
                        });
                        // Run-status readout — the download glyph (spinner while
                        // running, green check when done), the status word, and
                        // the live % — anchored flush to the card's right edge
                        // and centred on the controls' height.
                        let stat = egui::Rect::from_min_max(
                            egui::pos2(body_right - STAT_W, top),
                            egui::pos2(body_right, ui.min_rect().bottom()),
                        );
                        let mut child = ui.new_child(
                            egui::UiBuilder::new()
                                .max_rect(stat)
                                .layout(egui::Layout::top_down(egui::Align::Center)),
                        );
                        status_stat(&mut child, state.running, &state.status, state.progress, stat.height());
                    });
                    ui.add_space(8.0);
}

/// The bundled info SVG at 16 px, tinted with the muted theme colour, with a
/// hover tooltip. Returns the response so callers can lay it out alongside text.
fn info_icon(ui: &mut egui::Ui, tooltip: &str) -> egui::Response {
    ui.add(
        egui::Image::new(egui::include_image!("../icons/info.svg"))
            .fit_to_exact_size(egui::vec2(16.0, 16.0))
            .tint(crate::theme::icon_tint(MUTED())),
    )
    .on_hover_text(tooltip)
}

/// The hero header card: the source selector (title + chevron, like the
/// generator's model picker), a live Online/Offline capsule, and the source's
/// daily-allowance meter.
fn hero_card(ui: &mut egui::Ui, state: &mut DownloaderState) {
    let api = state.api_status.load(Ordering::Relaxed);
    let source = state.source;
    let used = state.quota_today();
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(egui::CornerRadius::same(18))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .inner_margin(egui::Margin::symmetric(14, 12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 9.0;
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    source_selector(ui, state);
                    ui.label(egui::RichText::new(state.source.subtitle()).color(MUTED()).size(11.0));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    status_capsule(ui, api);
                });
            });

            ui.add_space(11.0);
            allowance_meter(ui, used, source);
        });
}

/// The hero's "N / cap today" meter: a hairline bar that shifts green →
/// amber → red as the day's allowance is spent. The cap is per source;
/// hovering the label explains where each number comes from.
fn allowance_meter(ui: &mut egui::Ui, used: u32, source: Source) {
    let cap = source.daily_cap();
    let frac = (used as f32 / cap as f32).clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Daily allowance").color(MUTED()).size(10.5))
            .on_hover_text(source.cap_note());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{used} / {cap}")).color(MUTED()).size(10.5).strong(),
            );
        });
    });
    ui.add_space(4.0);
    let h = 5.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let r = egui::CornerRadius::same((h / 2.0) as u8);
    ui.painter().rect_filled(rect, r, FIELD());
    if frac > 0.0 {
        let color = if frac < 0.7 {
            egui::Color32::from_rgb(46, 160, 67)
        } else if frac < 0.95 {
            egui::Color32::from_rgb(240, 198, 60)
        } else {
            egui::Color32::from_rgb(210, 70, 70)
        };
        let fill = egui::Rect::from_min_size(rect.min, egui::vec2((rect.width() * frac).max(h), h));
        ui.painter().rect_filled(fill, r, color);
    }
}

/// The hero's source title with an up/down chevron, mirroring the generator's
/// model selector. Clicking opens the source list; picking one switches the
/// whole form (credentials, hints, sections) to that source.
fn source_selector(ui: &mut egui::Ui, state: &mut DownloaderState) {
    let menu_id = ui.id().with("dl_source_menu");
    let open = egui::Popup::is_id_open(ui.ctx(), menu_id);

    let galley = egui::WidgetText::from(egui::RichText::new(state.source.name()).color(TEXT()).strong().size(16.0))
        .into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Body);
    let (arrow, gap) = (14.0, 3.0);
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(galley.size().x + gap + arrow, galley.size().y),
        egui::Sense::click(),
    );
    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let text_pos = egui::pos2(rect.left(), rect.center().y - galley.size().y / 2.0);
        ui.painter().galley(text_pos, galley, TEXT());
        let arrow_src = if open {
            egui::include_image!("../icons/arrow_up.svg")
        } else {
            egui::include_image!("../icons/arrow_down.svg")
        };
        let arrow_rect = egui::Rect::from_center_size(
            egui::pos2(rect.right() - arrow / 2.0, rect.center().y + 1.0),
            egui::vec2(arrow, arrow),
        );
        egui::Image::new(arrow_src)
            .tint(crate::theme::icon_tint(MUTED()))
            .paint_at(ui, arrow_rect);
    }
    egui::Popup::menu(&resp).id(menu_id).frame(crate::zoom::menu_frame()).show(|ui| {
        ui.set_min_width(160.0);
        let radius = egui::CornerRadius::same(6);
        ui.visuals_mut().widgets.inactive.corner_radius = radius;
        ui.visuals_mut().widgets.hovered.corner_radius = radius;
        for s in Source::ALL {
            if ui.selectable_label(state.source == s, s.name()).clicked() {
                if state.source != s {
                    state.source = s;
                    // The capsule re-checks the new site; remember the choice.
                    state.api_status.store(API_CHECKING, Ordering::Relaxed);
                    save_config(&state.saved());
                }
                ui.close();
            }
        }
    });
}

/// The compact run-status block (Options card, right edge): the download glyph
/// (spinner ring while running, green check once done), the status word
/// ("Idle" / "Downloading" / …), and the live percentage during/after a run.
/// `avail_h` is the reserved region's height, used to centre the stack in it.
fn status_stat(ui: &mut egui::Ui, running: bool, status: &str, progress: (u32, u32), avail_h: f32) {
    let frac = if progress.1 > 0 {
        (progress.0 as f32 / progress.1 as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let show_pct = running || progress.1 > 0;
    // Estimated stack height (icon + labels + gaps) → top pad that centres it.
    let mut h = 20.0;
    if !status.is_empty() {
        h += 2.0 + 14.0;
    }
    if show_pct {
        h += 2.0 + 16.0;
    }
    ui.add_space(((avail_h - h) / 2.0).max(0.0));
    ui.spacing_mut().item_spacing.y = 2.0;
    download_indicator(ui, running, status == "Done");
    if !status.is_empty() {
        ui.label(egui::RichText::new(status).color(MUTED()).size(10.5));
    }
    if show_pct {
        let pct = (frac * 100.0).round() as u32;
        let pct_color = if running { ACCENT1() } else { MUTED() };
        ui.label(egui::RichText::new(format!("{pct}%")).color(pct_color).size(12.0).strong());
    }
}

/// The status block's download glyph: the Arrow Downward Alt icon, tinted blue
/// and wrapped in an animated blue ring while `running`. When idle it's just
/// the muted arrow; after a successful run it's a green check-circle. The ring
/// is a rotating arc (a spinner), so it reads as "working" even at 0% — the
/// exact progress is the percentage label beside it.
fn download_indicator(ui: &mut egui::Ui, running: bool, done: bool) {
    let size = 20.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let painter = ui.painter().clone();
    let center = rect.center();

    // Finished (and not mid-run): a green check-circle in place of the download
    // glyph, signalling the run completed successfully.
    if done && !running {
        let green = egui::Color32::from_rgb(80, 200, 120);
        egui::Image::new(egui::include_image!("../icons/check_circle.svg"))
            .tint(green)
            .paint_at(ui, rect);
        return;
    }

    if running {
        let radius = size * 0.5 - 1.0;
        let t = ui.input(|i| i.time) as f32;
        // Faint full ring underneath, then a brighter rotating arc on top.
        painter.circle_stroke(center, radius, egui::Stroke::new(2.0, ACCENT1().gamma_multiply(0.25)));
        let start = (t * 3.0) % std::f32::consts::TAU;
        let sweep = std::f32::consts::PI * 0.6;
        let pts: Vec<egui::Pos2> = (0..=24)
            .map(|k| {
                let a = start + sweep * (k as f32 / 24.0);
                center + radius * egui::vec2(a.cos(), a.sin())
            })
            .collect();
        painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0, ACCENT1())));
        ui.ctx().request_repaint(); // keep the spinner animating
    }

    // Arrow glyph in the centre (blue while downloading, muted otherwise).
    let tint = if running { ACCENT1() } else { MUTED() };
    let icon = size * 0.6;
    let icon_rect = egui::Rect::from_center_size(center, egui::vec2(icon, icon));
    egui::Image::new(egui::include_image!("../icons/Arrow Downward Alt.svg"))
        .tint(tint)
        .paint_at(ui, icon_rect);
}

/// The API status as a small capsule chip: "● Online" on a FIELD well.
fn status_capsule(ui: &mut egui::Ui, api: u8) {
    let (text, dot) = match api {
        API_ONLINE => ("Online", egui::Color32::from_rgb(46, 160, 67)),
        API_OFFLINE => ("Offline", egui::Color32::from_rgb(210, 70, 70)),
        _ => ("Checking…", egui::Color32::from_rgb(150, 150, 150)),
    };
    egui::Frame::new()
        .fill(FIELD())
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .corner_radius(egui::CornerRadius::same(10))
        .inner_margin(egui::Margin::symmetric(8, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                // This sits in a right-to-left layout, so add the label first
                // (it lands rightmost), then the dot, to read "● Online".
                ui.label(egui::RichText::new(text).color(MUTED()).size(10.5).strong());
                let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 3.5, dot);
            });
        });
}

/// A titled, rounded group card holding related controls: a colour-tinted icon
/// chip + normal-case title over the card body (macOS Settings style). `info`
/// adds a hover ⓘ after the title.
fn section_card(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'_>,
    tint: egui::Color32,
    title: &str,
    info: Option<&str>,
    add: impl FnOnce(&mut egui::Ui),
) {
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 7.0;
        // The bare icon, colour-tinted — no chip box behind it.
        let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
        egui::Image::new(icon)
            .tint(tint)
            .paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(15.0, 15.0)));
        ui.label(egui::RichText::new(title).color(TEXT()).strong().size(12.5));
        if let Some(info) = info {
            info_icon(ui, info);
        }
    });
    ui.add_space(5.0);
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(egui::CornerRadius::same(14))
        .inner_margin(egui::Margin::symmetric(12, 10))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// A file-type toggle capsule: icon + label, accent-filled with white ink when
/// on, a quiet FIELD well when off.
fn type_chip(ui: &mut egui::Ui, on: &mut bool, icon: egui::ImageSource<'_>, label: &str) -> egui::Response {
    let font = egui::FontId::proportional(12.0);
    let icon_s = 13.0;
    let (pad_x, pad_y, gap) = (10.0, 6.0, 5.0);
    let galley = ui.fonts_mut(|f| f.layout_no_wrap(label.to_string(), font.clone(), egui::Color32::PLACEHOLDER));
    let size = egui::vec2(
        pad_x * 2.0 + icon_s + gap + galley.size().x,
        pad_y * 2.0 + galley.size().y.max(icon_s),
    );
    let (rect, mut resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, label));
    if ui.is_rect_visible(rect) {
        let r = egui::CornerRadius::same((rect.height() / 2.0) as u8);
        let (fill, ink) = if *on {
            (ACCENT1(), egui::Color32::WHITE)
        } else if resp.hovered() {
            (FIELD2(), TEXT())
        } else {
            (FIELD(), MUTED())
        };
        ui.painter().rect_filled(rect, r, fill);
        if !*on {
            ui.painter().rect_stroke(rect, r, egui::Stroke::new(1.0, EDGE()), egui::StrokeKind::Inside);
        }
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(rect.left() + pad_x + icon_s / 2.0, rect.center().y),
            egui::vec2(icon_s, icon_s),
        );
        egui::Image::new(icon).tint(ink).paint_at(ui, icon_rect);
        let text_pos = egui::pos2(
            rect.left() + pad_x + icon_s + gap,
            rect.center().y - galley.size().y / 2.0,
        );
        ui.painter().galley(text_pos, galley, ink);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A text-only toggle capsule matching [`type_chip`]'s look, for filter rows
/// that don't have a natural icon (Wallhaven's categories/purity).
fn opt_chip(ui: &mut egui::Ui, on: &mut bool, label: &str) -> egui::Response {
    let font = egui::FontId::proportional(12.0);
    let (pad_x, pad_y) = (11.0, 6.0);
    let galley = ui.fonts_mut(|f| f.layout_no_wrap(label.to_string(), font, egui::Color32::PLACEHOLDER));
    let size = egui::vec2(pad_x * 2.0 + galley.size().x, pad_y * 2.0 + galley.size().y);
    let (rect, mut resp) = ui.allocate_exact_size(size, egui::Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, label));
    if ui.is_rect_visible(rect) {
        let r = egui::CornerRadius::same((rect.height() / 2.0) as u8);
        let (fill, ink) = if *on {
            (ACCENT1(), egui::Color32::WHITE)
        } else if resp.hovered() {
            (FIELD2(), TEXT())
        } else {
            (FIELD(), MUTED())
        };
        ui.painter().rect_filled(rect, r, fill);
        if !*on {
            ui.painter().rect_stroke(rect, r, egui::Stroke::new(1.0, EDGE()), egui::StrokeKind::Inside);
        }
        let text_pos = egui::pos2(rect.left() + pad_x, rect.center().y - galley.size().y / 2.0);
        ui.painter().galley(text_pos, galley, ink);
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Wallhaven's slice of the Options card: category and purity chips, the sort
/// method (plus toplist range when relevant), and the resolution/ratio
/// filters — each maps straight onto a search parameter (see `wallhaven.rs`).
fn wallhaven_options(ui: &mut egui::Ui, enabled: bool, wh: &mut crate::wallhaven::Opts) {
    ui.add_space(8.0);
    field_label(ui, "Categories");
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_enabled_ui(enabled, |ui| {
            opt_chip(ui, &mut wh.cat_general, "General");
            opt_chip(ui, &mut wh.cat_anime, "Anime");
            opt_chip(ui, &mut wh.cat_people, "People");
        });
    });
    ui.add_space(8.0);
    field_label(ui, "Purity");
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.add_enabled_ui(enabled, |ui| {
            opt_chip(ui, &mut wh.pur_sfw, "SFW");
            opt_chip(ui, &mut wh.pur_sketchy, "Sketchy")
                .on_hover_text("Borderline content — needs a Wallhaven account's API key to \
                                actually show up if the account has it disabled.");
            opt_chip(ui, &mut wh.pur_nsfw, "NSFW")
                .on_hover_text("NSFW results require a Wallhaven API key (free — see Account).");
        });
    });
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Sort").color(TEXT()));
        ui.add_enabled_ui(enabled, |ui| {
            let label = crate::wallhaven::SORTINGS
                .iter()
                .find(|(v, _)| *v == wh.sorting)
                .map_or("Date added", |(_, l)| *l);
            egui::ComboBox::from_id_salt("wallhaven_sorting")
                .width(112.0)
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for (v, l) in crate::wallhaven::SORTINGS {
                        ui.selectable_value(&mut wh.sorting, v.to_string(), l);
                    }
                });
        });
        if wh.sorting == "toplist" {
            ui.add_space(12.0);
            ui.label(egui::RichText::new("Range").color(TEXT()));
            ui.add_enabled_ui(enabled, |ui| {
                let label = crate::wallhaven::TOP_RANGES
                    .iter()
                    .find(|(v, _)| *v == wh.top_range)
                    .map_or("Last month", |(_, l)| *l);
                egui::ComboBox::from_id_salt("wallhaven_toprange")
                    .width(112.0)
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        for (v, l) in crate::wallhaven::TOP_RANGES {
                            ui.selectable_value(&mut wh.top_range, v.to_string(), l);
                        }
                    });
            });
        }
    });
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.scope(|ui| {
            ui.visuals_mut().extreme_bg_color = FIELD();
            ui.label(egui::RichText::new("Min res").color(TEXT()))
                .on_hover_text("Only wallpapers at least this large, e.g. 1920x1080. Blank = any size.");
            ui.add_enabled(
                enabled,
                egui::TextEdit::singleline(&mut wh.atleast)
                    .hint_text("any")
                    .desired_width(78.0)
                    .margin(egui::Margin::symmetric(8, 5)),
            );
            ui.add_space(12.0);
            ui.label(egui::RichText::new("Ratios").color(TEXT()))
                .on_hover_text("Comma-separated aspect ratios, e.g. 16x9,16x10. Blank = any shape.");
            ui.add_enabled(
                enabled,
                egui::TextEdit::singleline(&mut wh.ratios)
                    .hint_text("any")
                    .desired_width(90.0)
                    .margin(egui::Margin::symmetric(8, 5)),
            );
        });
    });
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.add_enabled_ui(enabled, |ui| {
            opt_chip(ui, &mut wh.save_tags, "Save tags").on_hover_text(
                "Writes each wallpaper's tag list to a matching .txt sidecar, like the \
                 booru sources. Search results don't include tags, so this makes one \
                 extra API call per downloaded file (still well inside Wallhaven's \
                 45 requests/minute limit at the 3-second delay).",
            );
        });
    });
}

/// Spawn a daemon-style thread that probes the selected source's homepage
/// every 5s while the view is visible and stores the result in `status`,
/// repainting the UI when it changes. Polling pauses whenever `visible` stops
/// being set (view not shown); `source` (a [`Source::index`]) picks the site.
fn start_api_monitor(
    status: Arc<AtomicU8>,
    visible: Arc<AtomicBool>,
    source: Arc<AtomicU8>,
    ctx: egui::Context,
) {
    std::thread::spawn(move || {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::NativeTls)
                    // Validate against the OS cert store (with AIA intermediate
                    // fetching) instead of ureq's bundled webpki roots — see
                    // civitai.rs for the CDN incomplete-chain failure this avoids.
                    .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                    .build(),
            )
            .timeout_global(Some(Duration::from_secs(8)))
            .http_status_as_error(false)
            .build()
            .into();

        loop {
            // Only ping while the Downloader view has rendered since the last
            // cycle — leaving the view pauses the polling entirely.
            if !visible.swap(false, Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
            let online = agent
                .get(Source::from_index(source.load(Ordering::Relaxed)).home())
                .header("User-Agent", USER_AGENT)
                .call()
                .map(|r| {
                    let s = r.status().as_u16();
                    (200..500).contains(&s)
                })
                .unwrap_or(false);

            let new = if online { API_ONLINE } else { API_OFFLINE };
            if status.swap(new, Ordering::Relaxed) != new {
                ctx.request_repaint();
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    });
}

/// A small muted caption shown above a field.
fn field_label(ui: &mut egui::Ui, label: &str) {
    ui.label(egui::RichText::new(label).color(MUTED()).size(12.0));
    ui.add_space(2.0);
}

/// A full-width text field with the theme's FIELD well background so it stands
/// out against the PANEL section card (PANEL-on-PANEL made the box invisible).
fn field_edit(ui: &mut egui::Ui, enabled: bool, edit: egui::TextEdit<'_>) {
    ui.scope(|ui| {
        ui.visuals_mut().extreme_bg_color = FIELD();
        ui.add_enabled(
            enabled,
            edit.desired_width(f32::INFINITY).margin(egui::Margin::symmetric(10, 6)),
        );
    });
}

/// Validate the form and spawn the background worker.
fn start_download(state: &mut DownloaderState, ctx: &egui::Context) {
    if state.running {
        return;
    }
    // 18+ gate backstop: an adult-capable run never starts unconfirmed, even
    // if a click races the dialog.
    let adult = matches!(state.source, Source::Gelbooru | Source::Danbooru)
        || (state.source == Source::Wallhaven && (state.wh.pur_sketchy || state.wh.pur_nsfw));
    if adult && !crate::age_gate::acknowledged() {
        state.push_log("Please confirm the adult-content notice first.");
        return;
    }
    state.log.clear();
    state.progress = (0, 0);

    // Hard floor on the delay — protects Gelbooru from being overloaded even if a
    // stale config or edge case slipped a smaller value through.
    if state.delay < MIN_DELAY {
        state.delay = MIN_DELAY;
    }

    let source = state.source;
    let (user, key) = state.creds();

    // Per-source credential rules.
    match source {
        Source::Gelbooru => {
            if user.is_empty() || key.is_empty() {
                state.push_log("Error: User ID and API Key are required.");
                state.status = "Idle".to_string();
                return;
            }
        }
        Source::Pexels | Source::Pixabay => {
            if key.is_empty() {
                state.push_log(format!(
                    "Error: a {} API key is required (free — see Account).",
                    source.name()
                ));
                state.status = "Idle".to_string();
                return;
            }
        }
        Source::Danbooru => {
            // Anonymous is fine; a half-entered pair silently downgrades to
            // anonymous, so catch it.
            if user.is_empty() != key.is_empty() {
                state.push_log("Error: Danbooru needs BOTH Login and API key (or neither).");
                state.status = "Idle".to_string();
                return;
            }
        }
        Source::Safebooru => {} // fully anonymous
        Source::Wallhaven => {
            // Key optional — but the filter selections must make sense.
            if !(state.wh.cat_general || state.wh.cat_anime || state.wh.cat_people) {
                state.push_log("No categories selected. Nothing to download.");
                state.status = "Idle".to_string();
                return;
            }
            if !(state.wh.pur_sfw || state.wh.pur_sketchy || state.wh.pur_nsfw) {
                state.push_log("No purity levels selected. Nothing to download.");
                state.status = "Idle".to_string();
                return;
            }
            if state.wh.pur_nsfw && key.is_empty() {
                state.push_log("Error: NSFW results require a Wallhaven API key (see Account).");
                state.status = "Idle".to_string();
                return;
            }
        }
    }
    // At least one of the types the source actually offers must be on.
    if source.has_type_chips() {
        let any_type = state.include_img
            || (source.offers_gif() && state.include_gif)
            || (source.offers_video() && state.include_vid);
        if !any_type {
            state.push_log("No file types selected. Nothing to download.");
            state.status = "Idle".to_string();
            return;
        }
    }
    if state.output_dir.trim().is_empty() {
        state.push_log("Error: Output folder is blank.");
        state.status = "Idle".to_string();
        return;
    }
    if source != Source::Gelbooru && state.tags.trim().is_empty() {
        state.push_log("Error: enter a search first.");
        state.status = "Idle".to_string();
        return;
    }

    // Persist the inputs for next time.
    save_config(&state.saved());

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    state.cancel = Arc::clone(&cancel);
    state.rx = Some(rx);
    state.running = true;
    state.status = "Connecting…".to_string();
    let ctx = ctx.clone();

    if source == Source::Gelbooru {
        let cfg = WorkerCfg {
            user_id: user,
            api_key: key,
            tags: state.tags.clone(),
            blacklist: state.blacklist.clone(),
            limit: state.limit,
            delay: state.delay,
            include_img: state.include_img,
            include_gif: state.include_gif,
            include_vid: state.include_vid,
            output_dir: PathBuf::from(state.output_dir.trim()),
        };
        std::thread::spawn(move || {
            run_download(cfg, tx, cancel, ctx);
        });
    } else {
        let cfg = SrcCfg {
            source,
            user,
            key,
            query: state.tags.clone(),
            blacklist: state.blacklist.clone(),
            limit: state.limit,
            delay: state.delay,
            include_img: state.include_img,
            include_gif: state.include_gif,
            include_vid: state.include_vid,
            output_dir: PathBuf::from(state.output_dir.trim()),
            wh: state.wh.clone(),
            seed: random_seed(),
        };
        std::thread::spawn(move || {
            run_download_src(cfg, tx, cancel, ctx);
        });
    }
}

/// Immutable settings handed to the worker thread.
struct WorkerCfg {
    user_id: String,
    api_key: String,
    tags: String,
    blacklist: String,
    limit: u32,
    delay: f32,
    include_img: bool,
    include_gif: bool,
    include_vid: bool,
    output_dir: PathBuf,
}

/// A parsed Gelbooru post.
struct Post {
    md5: String,
    file_url: String,
    raw_tags: String,
}

fn run_download(cfg: WorkerCfg, tx: Sender<DlMsg>, cancel: Arc<AtomicBool>, ctx: egui::Context) {
    let log = |s: String| {
        let _ = tx.send(DlMsg::Log(s));
        ctx.request_repaint();
    };

    let mut downloaded_log = load_download_log();
    log(format!("Loaded {} previously downloaded file records.", downloaded_log.len()));

    let final_tags = build_final_tags(&cfg.tags, cfg.include_img, cfg.include_gif, cfg.include_vid);
    if final_tags.trim().is_empty() {
        log("No tags provided (and/or all filtered). Nothing to do.".into());
        let _ = tx.send(DlMsg::Done);
        return;
    }
    log(format!("Starting download with tags: {final_tags}"));

    if let Err(e) = std::fs::create_dir_all(&cfg.output_dir) {
        log(format!("Error: cannot create output folder: {e}"));
        let _ = tx.send(DlMsg::Done);
        return;
    }

    let agent = make_agent();

    // Session cache of tag name -> Gelbooru type, so common tags are only looked
    // up once across the whole run (most tags repeat across posts).
    let mut tag_types: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    // Shared artist/character map (md5 -> roles), loaded once and accumulated into
    // as images download, then saved back to one tag_roles.json in the config dir.
    let mut roles_map = load_tag_roles_map();

    // Enforce the daily cap: today's remaining allowance bounds this run.
    let mut used_today = quota_used_today(Source::Gelbooru);
    let remaining_today = DAILY_CAP.saturating_sub(used_today);
    if remaining_today == 0 {
        log(format!(
            "Daily limit reached ({DAILY_CAP}/day). Try again tomorrow."
        ));
        let _ = tx.send(DlMsg::Done);
        return;
    }
    let cap = cfg.limit.min(remaining_today);
    if cap < cfg.limit {
        log(format!(
            "Note: only {remaining_today} of today's {DAILY_CAP} daily allowance remain — \
             this run is capped at {cap}."
        ));
    }

    let _ = tx.send(DlMsg::Progress(0, cap));
    ctx.request_repaint();

    let mut total_downloaded: u32 = 0;
    let mut page: u32 = 0;

    'outer: while total_downloaded < cap && !cancel.load(Ordering::SeqCst) {
        let posts = match fetch_page(&agent, &final_tags, page, &cfg, &cancel, &log) {
            Some(p) => p,
            None => break,
        };
        if posts.is_empty() {
            log("No more posts found.".into());
            break;
        }

        for post in posts {
            if cancel.load(Ordering::SeqCst) || total_downloaded >= cap {
                break 'outer;
            }
            if post.file_url.is_empty() || post.md5.is_empty() {
                continue;
            }
            if post.file_url.to_lowercase().ends_with(".zip") {
                log(format!("Skipped zip file: {}", post.file_url));
                continue;
            }
            if is_blacklisted(&post.raw_tags, &cfg.blacklist) {
                log(format!("Skipped (blacklisted): {}", post.md5));
                continue;
            }
            if downloaded_log.contains(&post.md5) {
                log(format!("Skipped (already downloaded): {}", post.md5));
                continue;
            }

            let clean = post.file_url.split('?').next().unwrap_or(&post.file_url);
            let ext = clean.rsplit('.').next().unwrap_or("bin").to_lowercase();

            if !is_allowed_by_selection(&ext, cfg.include_img, cfg.include_gif, cfg.include_vid) {
                log(format!("Skipped (type not selected): {}.{}", post.md5, ext));
                continue;
            }

            let file_name = format!("{}.{}", post.md5, ext);
            let img_path = cfg.output_dir.join(&file_name);
            let txt_path = cfg.output_dir.join(format!("{}.txt", post.md5));

            if img_path.exists() {
                log(format!("Skipped (file exists): {file_name}"));
                append_download_log(&post.md5, &mut downloaded_log);
                continue;
            }

            log(format!("Downloading: {file_name}"));
            match download_file(&agent, &post.file_url, &img_path, SITE_HOME, &cancel) {
                Ok(true) => {}
                Ok(false) => {
                    if cancel.load(Ordering::SeqCst) {
                        break 'outer;
                    }
                    continue;
                }
                Err(e) => {
                    log(format!("Error downloading {}: {e}", post.md5));
                    let _ = std::fs::remove_file(&img_path);
                    continue;
                }
            }

            let formatted = format_gelbooru_tags(&post.raw_tags);
            if let Err(e) = std::fs::write(&txt_path, formatted) {
                log(format!("Warning: could not write tags for {}: {e}", post.md5));
            }

            // Record the artist (username) + character tags into the shared
            // tag_roles.json so the viewer can colour them. Resolve each tag's type
            // via the tag API, caching results across the run to minimise requests.
            let names: Vec<&str> = post.raw_tags.split_whitespace().collect();
            let unknown: Vec<String> = names
                .iter()
                .filter(|n| !tag_types.contains_key(**n))
                .map(|n| n.to_string())
                .collect();
            if !unknown.is_empty() {
                let fetched = fetch_tag_types(&agent, &unknown, &cfg.user_id, &cfg.api_key);
                for (k, v) in fetched {
                    tag_types.insert(k, v);
                }
                // Mark any tag the API didn't return as general (0), so we don't
                // re-query it for every subsequent post.
                for n in &unknown {
                    tag_types.entry(n.clone()).or_insert(0);
                }
            }
            let artist: Vec<String> =
                names.iter().filter(|n| tag_types.get(**n) == Some(&1)).map(|n| n.to_string()).collect();
            let character: Vec<String> =
                names.iter().filter(|n| tag_types.get(**n) == Some(&4)).map(|n| n.to_string()).collect();
            if !artist.is_empty() || !character.is_empty() {
                roles_map.insert(post.md5.clone(), TagRoles { artist, character });
                save_tag_roles_map(&roles_map);
            }

            append_download_log(&post.md5, &mut downloaded_log);
            total_downloaded += 1;
            // Count it against today's quota and persist immediately, so a crash
            // mid-run can't reset the running total.
            used_today += 1;
            quota_save(Source::Gelbooru, used_today);
            let _ = tx.send(DlMsg::Progress(total_downloaded, cap));
            ctx.request_repaint();

            if cfg.delay > 0.0 {
                // Sleep in small slices so Cancel feels responsive.
                let mut slept = 0.0;
                while slept < cfg.delay && !cancel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(100));
                    slept += 0.1;
                }
            }
        }

        page += 1;
        std::thread::sleep(Duration::from_millis(500)); // polite pacing
    }

    if cancel.load(Ordering::SeqCst) {
        log("Cancelled.".into());
    } else {
        log(format!("Download finished: {total_downloaded} new files this session."));
        let left = DAILY_CAP.saturating_sub(used_today);
        log(format!("Daily allowance remaining: {left} of {DAILY_CAP}."));
    }
    let _ = tx.send(DlMsg::Done);
    ctx.request_repaint();
}

/// The download workers' HTTP agent.
///
/// native-tls => Windows SChannel. ureq 3.x defaults to rustls even with the
/// native-tls feature on, so the provider must be selected explicitly (rustls
/// isn't compiled in — see Cargo.toml / ai_models.rs). Without this the agent
/// fails on every HTTPS call, which on a worker thread looked like a silent
/// no-op. `http_status_as_error(false)` lets us inspect 4xx/5xx ourselves.
/// Only the request-setup phases are bounded (DNS, connect, send, response
/// headers) — those are blocking calls that can't see the cancel flag; the
/// body is intentionally uncapped since it streams in 64 KB chunks that poll
/// the cancel flag themselves.
fn make_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                // Use the OS cert store (with AIA intermediate fetching) rather
                // than ureq's bundled webpki roots — see civitai.rs.
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .http_status_as_error(false)
        .timeout_resolve(Some(Duration::from_secs(10)))
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_send_request(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(20)))
        .build()
        .into()
}

/// Immutable settings handed to the generic (non-Gelbooru) worker thread.
struct SrcCfg {
    source: Source,
    user: String,
    key: String,
    query: String,
    blacklist: String,
    limit: u32,
    delay: f32,
    include_img: bool,
    include_gif: bool,
    include_vid: bool,
    output_dir: PathBuf,
    /// Wallhaven's filter selections (defaulted for the other sources).
    wh: crate::wallhaven::Opts,
    /// Per-run seed so Wallhaven's `random` sorting doesn't repeat across pages.
    seed: String,
}

/// Six alphanumeric characters derived from the clock — Wallhaven's `seed`
/// format — so one run's `random` pages never overlap.
fn random_seed() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() as u64 ^ d.as_secs());
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut n = n;
    (0..6)
        .map(|_| {
            let c = CHARS[(n % CHARS.len() as u64) as usize] as char;
            n /= CHARS.len() as u64;
            c
        })
        .collect()
}

/// Kick off the one-shot account/key check for the selected source.
fn start_key_check(state: &mut DownloaderState, ctx: &egui::Context) {
    if state.key_check_rx.is_some() {
        return;
    }
    let source = state.source;
    let (user, key) = state.creds();
    let (tx, rx) = mpsc::channel();
    state.key_check_rx = Some(rx);
    state.key_check_result = None;
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let res = run_key_check(source, &user, &key);
        let _ = tx.send(res);
        ctx.request_repaint();
    });
}

/// One authenticated API call that verifies the stored credentials. Danbooru's
/// profile endpoint even reports the account's ban flag; the other sources
/// can only distinguish accepted vs rejected (invalid / revoked / blocked) vs
/// rate-limited.
fn run_key_check(source: Source, user: &str, key: &str) -> (Source, bool, String) {
    // Accountless sources have no Account card, so the button can't fire.
    if !source.has_account() {
        return (source, true, format!("{} has no accounts — nothing to check.", source.name()));
    }
    let agent = make_agent();
    let (url, headers) = match source {
        Source::Pexels => (crate::pexels::check_url(), crate::pexels::headers(key)),
        Source::Pixabay => (crate::pixabay::check_url(key), Vec::new()),
        Source::Danbooru => (crate::danbooru::profile_url(user, key), Vec::new()),
        Source::Wallhaven => (crate::wallhaven::settings_url(key), Vec::new()),
        // No profile endpoint on the dapi — a 1-post authed query stands in.
        Source::Gelbooru => (build_api_url("id:>0", 1, 0, user, key), Vec::new()),
        Source::Safebooru => unreachable!("accountless — handled above"),
    };

    let mut req = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/json,text/plain,*/*");
    for (k, v) in headers {
        req = req.header(k, &v);
    }
    let mut resp = match req.call() {
        Ok(r) => r,
        Err(e) => return (source, false, format!("Network error: {e}")),
    };

    let status = resp.status().as_u16();
    // Pexels reports the remaining API-request quota in a response header.
    let quota_left = resp
        .headers()
        .get("X-Ratelimit-Remaining")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body = resp.body_mut().read_to_string().unwrap_or_default();

    match status {
        200 => match source {
            Source::Danbooru => match crate::danbooru::parse_profile(&body) {
                Ok((name, _, true)) => {
                    (source, false, format!("Signed in as {name} — this account is BANNED on Danbooru."))
                }
                Ok((name, level, false)) => {
                    (source, true, format!("Signed in as {name} ({level}) — account in good standing."))
                }
                Err(e) => (source, false, format!("Unexpected profile response: {e}")),
            },
            Source::Pexels => {
                let extra = quota_left
                    .map(|n| format!(" — {n} API requests left this period"))
                    .unwrap_or_default();
                (source, true, format!("Key accepted{extra}."))
            }
            Source::Pixabay => {
                let extra = quota_left
                    .map(|n| format!(" — {n} API requests left this minute"))
                    .unwrap_or_default();
                (source, true, format!("Key accepted{extra}."))
            }
            Source::Wallhaven => {
                if body.contains("\"data\"") {
                    (source, true, "Key accepted — your account's settings apply to searches.".to_string())
                } else {
                    (source, false, "Unexpected response — the key may be rejected.".to_string())
                }
            }
            Source::Gelbooru => {
                if serde_json::from_str::<serde_json::Value>(&body).is_ok() {
                    (source, true, "Credentials accepted.".to_string())
                } else {
                    (source, false, "Unexpected response — credentials may be rejected.".to_string())
                }
            }
            Source::Safebooru => unreachable!("accountless — handled above"),
        },
        401 | 403 => (
            source,
            false,
            format!("Rejected (HTTP {status}) — key invalid, revoked, or the account is blocked."),
        ),
        // Pixabay reports a bad key as a 400 with a plain-text explanation.
        400 if source == Source::Pixabay => {
            (source, false, format!("Rejected — {}.", body.trim().trim_start_matches("[ERROR 400] ")))
        }
        429 => (source, false, "Rate limited — try again in a minute.".to_string()),
        s => (source, false, format!("Unexpected response (HTTP {s}).")),
    }
}

/// One paged result stream. Most sources have a single feed; Pexels and
/// Pixabay split photos and videos across two endpoints, walked one after the
/// other.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Feed {
    Primary,
    Videos,
}

/// Which feeds this run walks, in order, honouring the type selection.
fn source_feeds(cfg: &SrcCfg) -> Vec<Feed> {
    match cfg.source {
        Source::Pexels | Source::Pixabay => {
            let mut feeds = Vec::new();
            if cfg.include_img {
                feeds.push(Feed::Primary);
            }
            if cfg.include_vid {
                feeds.push(Feed::Videos);
            }
            feeds
        }
        _ => vec![Feed::Primary],
    }
}

/// The selected feed's search URL for one (1-based) page.
fn source_page_url(cfg: &SrcCfg, feed: Feed, page: u32) -> String {
    match (cfg.source, feed) {
        (Source::Pexels, Feed::Videos) => crate::pexels::video_page_url(&cfg.query, page),
        (Source::Pexels, Feed::Primary) => crate::pexels::page_url(&cfg.query, page),
        (Source::Pixabay, Feed::Videos) => crate::pixabay::video_page_url(&cfg.query, page, &cfg.key),
        (Source::Pixabay, Feed::Primary) => crate::pixabay::page_url(&cfg.query, page, &cfg.key),
        (Source::Safebooru, _) => crate::safebooru::page_url(&cfg.query, page),
        (Source::Danbooru, _) => crate::danbooru::page_url(&cfg.query, page, &cfg.user, &cfg.key),
        (Source::Wallhaven, _) => {
            crate::wallhaven::page_url(&cfg.query, page, &cfg.key, &cfg.blacklist, &cfg.wh, &cfg.seed)
        }
        (Source::Gelbooru, _) => unreachable!("gelbooru uses its own worker"),
    }
}

/// Extra request headers the source needs (Pexels carries its key here).
fn source_headers(cfg: &SrcCfg) -> Vec<(&'static str, String)> {
    match cfg.source {
        Source::Pexels => crate::pexels::headers(&cfg.key),
        _ => Vec::new(),
    }
}

fn source_parse(source: Source, feed: Feed, body: &str) -> Result<Vec<Item>, String> {
    match (source, feed) {
        (Source::Pexels, Feed::Videos) => crate::pexels::parse_videos(body),
        (Source::Pexels, Feed::Primary) => crate::pexels::parse(body),
        (Source::Pixabay, Feed::Videos) => crate::pixabay::parse_videos(body),
        (Source::Pixabay, Feed::Primary) => crate::pixabay::parse(body),
        (Source::Safebooru, _) => crate::safebooru::parse(body),
        (Source::Danbooru, _) => crate::danbooru::parse(body),
        (Source::Wallhaven, _) => crate::wallhaven::parse(body),
        (Source::Gelbooru, _) => unreachable!("gelbooru uses its own worker"),
    }
}

/// The generic multi-source worker: page through search results, de-duplicate
/// against the shared download log, stream files, and write tag/caption
/// sidecars where the source provides them. (Gelbooru keeps [`run_download`],
/// which adds the daily quota and artist/character tag-role resolution.)
fn run_download_src(cfg: SrcCfg, tx: Sender<DlMsg>, cancel: Arc<AtomicBool>, ctx: egui::Context) {
    let log = |s: String| {
        let _ = tx.send(DlMsg::Log(s));
        ctx.request_repaint();
    };

    let mut downloaded_log = load_download_log();
    log(format!("Loaded {} previously downloaded file records.", downloaded_log.len()));
    log(format!("Searching {} for: {}", cfg.source.name(), cfg.query.trim()));

    if let Err(e) = std::fs::create_dir_all(&cfg.output_dir) {
        log(format!("Error: cannot create output folder: {e}"));
        let _ = tx.send(DlMsg::Done);
        return;
    }

    let agent = make_agent();

    // Enforce the source's daily allowance: today's remainder bounds this run.
    let daily_cap = cfg.source.daily_cap();
    let mut used_today = quota_used_today(cfg.source);
    let remaining_today = daily_cap.saturating_sub(used_today);
    if remaining_today == 0 {
        log(format!("Daily limit reached ({daily_cap}/day). Try again tomorrow."));
        let _ = tx.send(DlMsg::Done);
        return;
    }
    let cap = cfg.limit.min(remaining_today);
    if cap < cfg.limit {
        log(format!(
            "Note: only {remaining_today} of today's {daily_cap} daily allowance remain — \
             this run is capped at {cap}."
        ));
    }

    let _ = tx.send(DlMsg::Progress(0, cap));
    ctx.request_repaint();

    let mut total_downloaded: u32 = 0;

    // Walk each feed (photos, then videos for Pexels) until the cap is hit.
    'outer: for feed in source_feeds(&cfg) {
        let mut page: u32 = 1; // these APIs are all 1-based
        'feed: while total_downloaded < cap && !cancel.load(Ordering::SeqCst) {
        let items = match fetch_items(&agent, &cfg, feed, page, &cancel, &log) {
            Some(i) => i,
            None => break 'outer,
        };
        if items.is_empty() {
            log("No more results.".into());
            break 'feed;
        }

        for item in items {
            if cancel.load(Ordering::SeqCst) || total_downloaded >= cap {
                break 'outer;
            }
            if item.url.is_empty() {
                continue;
            }
            // Tag strings (boorus) and captions (Pexels alt) both honour the
            // blacklist; Wallhaven excludes at the query level instead.
            if let Some(tags) = &item.tags
                && is_blacklisted(tags, &cfg.blacklist)
            {
                log(format!("Skipped (blacklisted): {}", item.stem));
                continue;
            }
            if downloaded_log.contains(&item.key) {
                log(format!("Skipped (already downloaded): {}", item.stem));
                continue;
            }
            // Boorus mix media; the image-only sources always pass this.
            if cfg.source.is_booru()
                && !is_allowed_by_selection(&item.ext, cfg.include_img, cfg.include_gif, cfg.include_vid)
            {
                log(format!("Skipped (type not selected): {}.{}", item.stem, item.ext));
                continue;
            }

            let file_name = format!("{}.{}", item.stem, item.ext);
            let dest = cfg.output_dir.join(&file_name);
            if dest.exists() {
                log(format!("Skipped (file exists): {file_name}"));
                append_download_log(&item.key, &mut downloaded_log);
                continue;
            }

            log(format!("Downloading: {file_name}"));
            match download_file(&agent, &item.url, &dest, cfg.source.home(), &cancel) {
                Ok(true) => {}
                Ok(false) => {
                    if cancel.load(Ordering::SeqCst) {
                        break 'outer;
                    }
                    continue;
                }
                Err(e) => {
                    log(format!("Error downloading {}: {e}", item.stem));
                    let _ = std::fs::remove_file(&dest);
                    continue;
                }
            }

            // Wallhaven search results carry no tags — with the option on,
            // one extra call to the per-wallpaper endpoint fills them in
            // (already comma-joined, so it skips the booru reformatting).
            let mut item = item;
            if cfg.source == Source::Wallhaven && cfg.wh.save_tags && item.tags.is_none() {
                match fetch_wallhaven_tags(&agent, &item, &cfg.key) {
                    Ok(tags) if !tags.is_empty() => item.tags = Some(tags),
                    Ok(_) => {}
                    Err(e) => log(format!("Warning: could not fetch tags for {}: {e}", item.stem)),
                }
            }

            // Sidecar: booru tag strings get the comma format; captions
            // (e.g. Pexels alt text) are written as-is.
            if let Some(tags) = &item.tags {
                let text = if cfg.source.is_booru() { format_gelbooru_tags(tags) } else { tags.clone() };
                if !text.is_empty() {
                    let txt_path = cfg.output_dir.join(format!("{}.txt", item.stem));
                    if let Err(e) = std::fs::write(&txt_path, text) {
                        log(format!("Warning: could not write tags for {}: {e}", item.stem));
                    }
                }
            }

            append_download_log(&item.key, &mut downloaded_log);
            total_downloaded += 1;
            // Count it against today's allowance and persist immediately, so a
            // crash mid-run can't reset the running total.
            used_today += 1;
            quota_save(cfg.source, used_today);
            let _ = tx.send(DlMsg::Progress(total_downloaded, cap));
            ctx.request_repaint();

            if cfg.delay > 0.0 {
                // Sleep in small slices so Cancel feels responsive.
                let mut slept = 0.0;
                while slept < cfg.delay && !cancel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(100));
                    slept += 0.1;
                }
            }
        }

        page += 1;
        std::thread::sleep(Duration::from_millis(500)); // polite pacing
        }
    }

    if cancel.load(Ordering::SeqCst) {
        log("Cancelled.".into());
    } else {
        log(format!("Download finished: {total_downloaded} new files this session."));
        let left = daily_cap.saturating_sub(used_today);
        log(format!("Daily allowance remaining: {left} of {daily_cap}."));
    }
    let _ = tx.send(DlMsg::Done);
    ctx.request_repaint();
}

/// One call to Wallhaven's per-wallpaper endpoint for the tag sidecar —
/// search results don't include tags. Best-effort: a failure only costs the
/// sidecar, never the download.
fn fetch_wallhaven_tags(agent: &ureq::Agent, item: &Item, key: &str) -> Result<String, String> {
    let id = item.stem.strip_prefix("wallhaven-").unwrap_or(&item.stem);
    let mut resp = agent
        .get(&crate::wallhaven::info_url(id, key))
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/json")
        .call()
        .map_err(|e| format!("network error: {e}"))?;
    let status = resp.status().as_u16();
    if status != 200 {
        return Err(format!("HTTP {status}"));
    }
    let body = resp.body_mut().read_to_string().map_err(|e| format!("read error: {e}"))?;
    crate::wallhaven::parse_tags(&body)
}

/// Fetch one search page for the generic worker, with the same retry/backoff
/// policy as the Gelbooru path. `None` means a fatal error (caller stops).
fn fetch_items(
    agent: &ureq::Agent,
    cfg: &SrcCfg,
    feed: Feed,
    page: u32,
    cancel: &AtomicBool,
    log: &impl Fn(String),
) -> Option<Vec<Item>> {
    let mut transient = 0u32;
    while !cancel.load(Ordering::SeqCst) {
        let url = source_page_url(cfg, feed, page);
        let mut req = agent
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "application/json,text/plain,*/*");
        for (k, v) in source_headers(cfg) {
            req = req.header(k, &v);
        }

        let mut resp = match req.call() {
            Ok(r) => r,
            Err(e) => {
                transient += 1;
                if transient > MAX_TRANSIENT_RETRIES {
                    log(format!("Error: network failure (max retries): {e}"));
                    return None;
                }
                let wait = backoff_ms(transient);
                log(format!("Network issue, retrying in {:.1}s…", wait as f64 / 1000.0));
                sleep_cancellable(wait, cancel);
                continue;
            }
        };

        let status = resp.status().as_u16();
        if status == 200 {
            let body = match resp.body_mut().read_to_string() {
                Ok(b) => b,
                Err(e) => {
                    log(format!("Error reading API response: {e}"));
                    return None;
                }
            };
            return match source_parse(cfg.source, feed, &body) {
                Ok(items) => Some(items),
                Err(e) => {
                    log(format!("Error: {} API: {e}", cfg.source.name()));
                    None
                }
            };
        }
        if status == 401 || status == 403 {
            log(format!(
                "Error: {} rejected the request (HTTP {status}) — check the API key.",
                cfg.source.name()
            ));
            return None;
        }
        if status == 429 || status == 408 || (500..=599).contains(&status) {
            transient += 1;
            if transient > MAX_TRANSIENT_RETRIES {
                log(format!("Error: API returned {status} repeatedly (max retries)."));
                return None;
            }
            let wait = backoff_ms(transient);
            log(format!("API busy (HTTP {status}), retrying in {:.1}s…", wait as f64 / 1000.0));
            sleep_cancellable(wait, cancel);
            continue;
        }

        log(format!("Error: API returned status {status}."));
        return None;
    }
    None
}

/// Fetch one API page, with retry/backoff on transient failures. `None` means a
/// fatal error (caller should stop).
fn fetch_page(
    agent: &ureq::Agent,
    final_tags: &str,
    page: u32,
    cfg: &WorkerCfg,
    cancel: &AtomicBool,
    log: &impl Fn(String),
) -> Option<Vec<Post>> {
    let mut transient = 0u32;
    while !cancel.load(Ordering::SeqCst) {
        let url = build_api_url(final_tags, PAGE_LIMIT, page, &cfg.user_id, &cfg.api_key);
        let resp = agent
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "application/json,text/plain,*/*")
            .call();

        let mut resp = match resp {
            Ok(r) => r,
            Err(e) => {
                transient += 1;
                if transient > MAX_TRANSIENT_RETRIES {
                    log(format!("Error: network failure (max retries): {e}"));
                    return None;
                }
                let wait = backoff_ms(transient);
                log(format!("Network issue, retrying in {:.1}s…", wait as f64 / 1000.0));
                sleep_cancellable(wait, cancel);
                continue;
            }
        };

        let status = resp.status().as_u16();
        if status == 200 {
            let body = match resp.body_mut().read_to_string() {
                Ok(b) => b,
                Err(e) => {
                    log(format!("Error reading API response: {e}"));
                    return None;
                }
            };
            return Some(parse_posts(&body, log));
        }

        if status == 429 || status == 408 || (500..=599).contains(&status) {
            transient += 1;
            if transient > MAX_TRANSIENT_RETRIES {
                log(format!("Error: API returned {status} repeatedly (max retries)."));
                return None;
            }
            let wait = backoff_ms(transient);
            log(format!("API busy (HTTP {status}), retrying in {:.1}s…", wait as f64 / 1000.0));
            sleep_cancellable(wait, cancel);
            continue;
        }

        log(format!("Error: API returned status {status}."));
        return None;
    }
    None
}

/// Stream a file to `dest`, honouring cancellation. Returns `Ok(true)` on
/// success, `Ok(false)` on a non-success status (partial file removed).
/// `referer` is the source's homepage — many CDNs 403 a hotlink without a
/// matching Referer / Origin.
fn download_file(
    agent: &ureq::Agent,
    file_url: &str,
    dest: &Path,
    referer: &str,
    cancel: &AtomicBool,
) -> Result<bool, String> {
    let url = normalize_file_url(file_url);
    let mut resp = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "*/*")
        .header("Referer", referer)
        .header("Origin", referer.trim_end_matches('/'))
        .call()
        .map_err(|e| e.to_string())?;

    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        return Ok(false);
    }

    let mut file = std::fs::File::create(dest).map_err(|e| e.to_string())?;
    let mut reader = resp.body_mut().as_reader();
    let mut buf = [0u8; 64 * 1024];
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file);
            let _ = std::fs::remove_file(dest);
            return Ok(false);
        }
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        use std::io::Write;
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
    Ok(true)
}

fn sleep_cancellable(total_ms: u64, cancel: &AtomicBool) {
    let mut slept = 0u64;
    while slept < total_ms && !cancel.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));
        slept += 100;
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (ported 1:1 from Gelbooru.java)
// ---------------------------------------------------------------------------

fn build_final_tags(user_tags: &str, img: bool, gif: bool, vid: bool) -> String {
    let mut tags: Vec<String> = Vec::new();
    let trimmed = user_tags.trim();
    if !trimmed.is_empty() {
        tags.extend(trimmed.split_whitespace().map(|s| s.to_string()));
    }

    let selected = img as u8 + gif as u8 + vid as u8;
    if selected == 1 && gif {
        // Gif-only: a single positive "gif" tag is far more reliable than
        // negating every other extension (which Gelbooru's tag limit truncates).
        tags.push("gif".to_string());
    } else {
        if !img {
            for e in IMAGE_EXTS {
                tags.push(format!("-{e}"));
            }
        }
        if !vid {
            for e in VIDEO_EXTS {
                tags.push(format!("-{e}"));
            }
        }
        if !gif {
            for e in GIF_EXTS {
                tags.push(format!("-{e}"));
            }
        }
    }
    tags.join(" ").trim().to_string()
}

fn is_blacklisted(raw_tags: &str, blacklist: &str) -> bool {
    let tags_lower = raw_tags.to_lowercase();
    blacklist
        .split(',')
        .map(|b| b.trim().to_lowercase())
        .filter(|b| !b.is_empty())
        .any(|b| tags_lower.contains(&b))
}

fn format_gelbooru_tags(raw: &str) -> String {
    let s = raw.trim();
    if s.is_empty() {
        return String::new();
    }
    s.split_whitespace().collect::<Vec<_>>().join(", ")
}

fn is_allowed_by_selection(ext: &str, img: bool, gif: bool, vid: bool) -> bool {
    let ext = ext.to_lowercase();
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return img;
    }
    if GIF_EXTS.contains(&ext.as_str()) {
        return gif;
    }
    if VIDEO_EXTS.contains(&ext.as_str()) {
        return vid;
    }
    false
}

/// Artist (username) + character tags for one image. Stored in a single shared
/// `tag_roles.json` (in the app's config dir), keyed by md5, that accumulates as
/// more images are downloaded — so the viewer can colour those tags.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct TagRoles {
    #[serde(default)]
    artist: Vec<String>,
    #[serde(default)]
    character: Vec<String>,
}

/// Path of the shared tag-roles map (md5 -> {artist, character}) in the config dir.
pub(crate) fn tag_roles_path() -> PathBuf {
    config_dir().join("tag_roles.json")
}

/// Load the shared tag-roles map, or an empty map if none/invalid.
fn load_tag_roles_map() -> std::collections::HashMap<String, TagRoles> {
    std::fs::read_to_string(tag_roles_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persist the shared tag-roles map.
fn save_tag_roles_map(map: &std::collections::HashMap<String, TagRoles>) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(s) = serde_json::to_string_pretty(map) {
        let _ = std::fs::write(tag_roles_path(), s);
    }
}

/// Look up each tag's Gelbooru `type` via the tag endpoint, returning name->type
/// (0 general · 1 artist · 3 copyright · 4 character · 5 metadata). Chunked, since
/// the endpoint takes space-separated `names`. Best-effort: any network/parse
/// failure just yields fewer entries (those tags fall back to "general").
fn fetch_tag_types(
    agent: &ureq::Agent,
    names: &[String],
    user_id: &str,
    api_key: &str,
) -> std::collections::HashMap<String, i64> {
    let mut out = std::collections::HashMap::new();
    let add = |t: &serde_json::Value, out: &mut std::collections::HashMap<String, i64>| {
        let name = t.get("name").and_then(|v| v.as_str());
        // `type` is an integer with json=1, but tolerate a stringified form too.
        let ty = t
            .get("type")
            .and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok())));
        if let (Some(name), Some(ty)) = (name, ty) {
            out.insert(name.to_string(), ty);
        }
    };
    for chunk in names.chunks(100) {
        let mut url = format!("{TAG_API_URL}&names={}", percent_encode(&chunk.join(" ")));
        if !user_id.trim().is_empty() {
            url.push_str("&user_id=");
            url.push_str(&percent_encode(user_id.trim()));
        }
        if !api_key.trim().is_empty() {
            url.push_str("&api_key=");
            url.push_str(&percent_encode(api_key.trim()));
        }
        let resp = agent
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .header("Accept", "application/json,text/plain,*/*")
            .call();
        let Ok(mut resp) = resp else { continue };
        if resp.status().as_u16() != 200 {
            continue;
        }
        let Ok(body) = resp.body_mut().read_to_string() else { continue };
        let Ok(root) = serde_json::from_str::<serde_json::Value>(&body) else { continue };
        // { "@attributes": {...}, "tag": [ {name,type,...}, ... ] } — "tag" can be a
        // single object instead of an array when there's one result.
        match root.get("tag") {
            Some(serde_json::Value::Array(arr)) => {
                for t in arr {
                    add(t, &mut out);
                }
            }
            Some(obj @ serde_json::Value::Object(_)) => add(obj, &mut out),
            _ => {}
        }
    }
    out
}

fn build_api_url(final_tags: &str, per_page: u32, pid: u32, user_id: &str, api_key: &str) -> String {
    use std::fmt::Write as _;
    let mut s = String::from(API_URL);
    s.push_str("&tags=");
    s.push_str(&percent_encode(final_tags));
    let _ = write!(s, "&limit={per_page}&pid={pid}");
    if !user_id.trim().is_empty() {
        s.push_str("&user_id=");
        s.push_str(&percent_encode(user_id.trim()));
    }
    if !api_key.trim().is_empty() {
        s.push_str("&api_key=");
        s.push_str(&percent_encode(api_key.trim()));
    }
    s
}

fn normalize_file_url(raw: &str) -> String {
    let s = raw.trim();
    if let Some(rest) = s.strip_prefix("//") {
        format!("https://{rest}")
    } else if s.starts_with('/') {
        format!("https://gelbooru.com{s}")
    } else {
        s.to_string()
    }
}

/// Percent-encode a query value (RFC 3986 unreserved set kept literal).
pub(crate) fn percent_encode(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

fn parse_posts(body: &str, log: &impl Fn(String)) -> Vec<Post> {
    let root: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => {
            log("Error: failed to parse API JSON.".into());
            return Vec::new();
        }
    };
    let post_node = match root.get("post") {
        Some(n) if !n.is_null() => n,
        _ => return Vec::new(),
    };

    let mut out = Vec::new();
    let add = |p: &serde_json::Value, out: &mut Vec<Post>| {
        let mut file_url = p.get("file_url").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if let Some(rest) = file_url.strip_prefix("//") {
            file_url = format!("https://{rest}");
        } else if file_url.starts_with('/') {
            file_url = format!("https://gelbooru.com{file_url}");
        }
        let md5 = p.get("md5").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let raw_tags = p.get("tags").and_then(|v| v.as_str()).unwrap_or("").to_string();

        if file_url.is_empty() || file_url.eq_ignore_ascii_case("null") {
            return;
        }
        if md5.is_empty() || md5.eq_ignore_ascii_case("null") {
            return;
        }
        out.push(Post { md5, file_url, raw_tags });
    };

    if let Some(arr) = post_node.as_array() {
        for p in arr {
            add(p, &mut out);
        }
    } else if post_node.is_object() {
        add(post_node, &mut out);
    }
    out
}

fn backoff_ms(retry: u32) -> u64 {
    let shift = retry.saturating_sub(1).min(5);
    (1000u64 << shift).min(20_000)
}

// ---------------------------------------------------------------------------
// Config + download-log persistence
// ---------------------------------------------------------------------------

// pub(crate): the age gate (src/age_gate.rs) keeps its marker file here too.
pub(crate) fn config_dir() -> PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow"))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn config_path() -> PathBuf {
    config_dir().join("gelbooru_credentials.json")
}

fn download_log_path() -> PathBuf {
    config_dir().join("gelbooru_download_log.json")
}

fn load_config() -> Option<SavedConfig> {
    let json = std::fs::read_to_string(config_path()).ok()?;
    let mut cfg: SavedConfig = serde_json::from_str(&json).ok()?;
    // The API keys are stored encrypted (DPAPI on Windows); decrypt them back.
    cfg.api_key = crate::secret::unprotect(&cfg.api_key);
    cfg.pexels_key = crate::secret::unprotect(&cfg.pexels_key);
    cfg.pixabay_key = crate::secret::unprotect(&cfg.pixabay_key);
    cfg.danbooru_key = crate::secret::unprotect(&cfg.danbooru_key);
    cfg.wallhaven_key = crate::secret::unprotect(&cfg.wallhaven_key);
    Some(cfg)
}

fn save_config(cfg: &SavedConfig) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    // Never write API keys as plaintext: encrypt them (DPAPI on Windows, tied to
    // the current user account) so they can't be read straight out of the JSON.
    let on_disk = SavedConfig {
        user_id: cfg.user_id.clone(),
        api_key: crate::secret::protect(&cfg.api_key),
        tags: cfg.tags.clone(),
        blacklist: cfg.blacklist.clone(),
        output_dir: cfg.output_dir.clone(),
        show_log: cfg.show_log,
        source: cfg.source.clone(),
        pexels_key: crate::secret::protect(&cfg.pexels_key),
        pixabay_key: crate::secret::protect(&cfg.pixabay_key),
        danbooru_login: cfg.danbooru_login.clone(),
        danbooru_key: crate::secret::protect(&cfg.danbooru_key),
        wallhaven_key: crate::secret::protect(&cfg.wallhaven_key),
        wallhaven_opts: cfg.wallhaven_opts.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&on_disk) {
        let _ = std::fs::write(config_path(), json);
    }
}

fn load_download_log() -> HashSet<String> {
    let mut set = HashSet::new();
    if let Ok(json) = std::fs::read_to_string(download_log_path())
        && let Ok(v) = serde_json::from_str::<Vec<String>>(&json) {
            for s in v {
                let t = s.trim().to_string();
                if !t.is_empty() {
                    set.insert(t);
                }
            }
        }
    set
}

// ---------------------------------------------------------------------------
// Daily download quota (encrypted, per calendar day)
// ---------------------------------------------------------------------------

/// `{ date, count }` for the daily cap, stored encrypted on disk.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct QuotaData {
    #[serde(default)]
    date: String,
    #[serde(default)]
    count: u32,
}

/// Per-source daily-allowance file. Gelbooru keeps its historical name so
/// pre-multi-source counts carry over.
fn quota_path(source: Source) -> PathBuf {
    let name = match source {
        Source::Gelbooru => "gelbooru_quota.dat".to_string(),
        s => format!("{}_quota.dat", s.config_key()),
    };
    config_dir().join(name)
}

/// Local calendar day as `YYYY-MM-DD`.
fn today_str() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// How many files have already been downloaded from `source` *today*. Returns
/// 0 if the file is missing, can't be decrypted, or holds an older date.
fn quota_used_today(source: Source) -> u32 {
    let Ok(stored) = std::fs::read_to_string(quota_path(source)) else {
        return 0;
    };
    let json = crate::secret::unprotect(stored.trim());
    let Ok(q) = serde_json::from_str::<QuotaData>(&json) else {
        return 0;
    };
    if q.date == today_str() {
        q.count
    } else {
        0
    }
}

/// Persist today's running count for `source`, encrypted.
fn quota_save(source: Source, count: u32) {
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    let q = QuotaData { date: today_str(), count };
    if let Ok(json) = serde_json::to_string(&q) {
        let enc = crate::secret::protect(&json);
        let _ = std::fs::write(quota_path(source), enc);
    }
}

fn append_download_log(md5: &str, set: &mut HashSet<String>) {
    if md5.trim().is_empty() {
        return;
    }
    set.insert(md5.trim().to_string());
    let dir = config_dir();
    let _ = std::fs::create_dir_all(&dir);
    let list: Vec<&String> = set.iter().collect();
    if let Ok(json) = serde_json::to_string(&list) {
        let _ = std::fs::write(download_log_path(), json);
    }
}
