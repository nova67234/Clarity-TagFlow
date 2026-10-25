//! Link (Settings → Link): another PC on the local network uses this one's
//! AI chat model and image / video generators — its GPU — and the two send
//! each other images and videos.
//!
//! One PC is the **host** (shares its models), the other a **receiver**
//! (uses them). The first time, the receiver types the host's address and
//! the 6-digit code it shows; after that both remember each other and
//! reconnect by themselves. Connections come only from the local network
//! (private address ranges, including private VPNs like Tailscale), and
//! everything between the two apps is encrypted end to end (Noise, with the
//! pairing code turned into a key by SPAKE2 — see `crypto`). The host can
//! keep serving with its window closed, as a background process (see
//! `background`).
//!
//! The receiver's AI chat and generators call the host through
//! [`remote_chat`] / [`remote_gen`]; the host serves those calls through
//! [`Backend`], which runs them on the very same code a local chat or
//! generation uses.

mod addr;
pub(crate) mod background;
mod client;
mod crypto;
mod discover;
pub(crate) mod files;
mod host;
mod peer;
mod trust;
mod wire;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) use peer::{BlobSrc, Call, CallEvent, HostInfo, Incoming, Peer, Services};

use crate::theme::{ACCENT1, MUTED, TEXT};

/// What this PC does in Link.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum LinkMode {
    /// Not linked: nothing listens, nothing connects.
    #[default]
    Off,
    /// Shares this PC's models.
    Host,
    /// Uses another PC's models.
    Receiver,
}

/// Settings → Link, saved with the other settings.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
#[serde(default)]
pub struct LinkSettings {
    pub mode: LinkMode,
    /// The host's port (TCP for sessions, UDP for being found).
    pub port: u16,
    // --- Host: what it shares. ---
    pub share_chat: bool,
    pub share_images: bool,
    pub share_video: bool,
    pub share_files: bool,
    /// Save files from paired devices without asking.
    pub accept_files_unasked: bool,
    /// Keep hosting (in the background) when the app is closed.
    pub background: bool,
    /// Start the background host when this user logs in.
    pub autostart: bool,
    // --- Receiver. ---
    /// The host address last typed.
    pub host_addr: String,
    /// Run the AI chat on the host.
    pub use_chat: bool,
    /// Run image / video generation on the host.
    pub use_gen: bool,
    /// Where files from the other PC are saved ("" = the default folder).
    pub received_folder: String,
}

impl Default for LinkSettings {
    fn default() -> Self {
        LinkSettings {
            mode: LinkMode::Off,
            port: addr::DEFAULT_PORT,
            share_chat: true,
            share_images: true,
            share_video: true,
            share_files: true,
            accept_files_unasked: false,
            background: false,
            autostart: false,
            host_addr: String::new(),
            use_chat: true,
            use_gen: true,
            received_folder: String::new(),
        }
    }
}

/// Link's own folder (identity key, remembered devices, background host
/// state, temporary files).
pub(crate) fn link_dir() -> PathBuf {
    crate::tagger::models_root().join("link")
}

/// Where files in transit land (chat attachments, start images, outputs on
/// their way back) before they're used or moved.
pub(crate) fn temp_dir() -> PathBuf {
    link_dir().join("temp").join(std::process::id().to_string())
}

/// Clear this process's temp folder, and what crashed ones left.
pub(crate) fn clear_temp() {
    let root = link_dir().join("temp");
    let own = std::process::id().to_string();
    for e in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let old = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > Duration::from_secs(24 * 3600));
        if e.file_name().to_string_lossy() == own || old {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// A fresh folder in `temp_dir` for one job ("gen-12"). Call ids repeat
/// across devices and sessions, so they can't name it.
pub(crate) fn job_dir(kind: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    temp_dir().join(format!("{kind}-{}", N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)))
}

/// Image and video generation exist here (not on macOS).
const GEN_AVAILABLE: bool = cfg!(not(target_os = "macos"));

/// Where a host's LoRA / embedding previews are cached on a receiver.
pub(crate) fn host_cache_dir(key: &[u8]) -> PathBuf {
    let hex = crypto::to_hex(key);
    link_dir().join("hosts").join(&hex[..hex.len().min(16)])
}

fn received_dir(s: &LinkSettings) -> PathBuf {
    if s.received_folder.trim().is_empty() {
        files::default_received_dir()
    } else {
        PathBuf::from(s.received_folder.trim())
    }
}

fn host_config(s: &LinkSettings, background: bool) -> host::HostConfig {
    host::HostConfig {
        name: addr::device_name(),
        port: s.port,
        services: Services {
            chat: s.share_chat,
            images: s.share_images && GEN_AVAILABLE,
            video: s.share_video && GEN_AVAILABLE,
            files: s.share_files,
        },
        accept_files_unasked: s.accept_files_unasked,
        background,
    }
}

// ---------------------------------------------------------------------------
// What the host runs for its devices.
// ---------------------------------------------------------------------------

/// The app's services, as a host serves them. The link layer checks what the
/// host shares before calling any of these.
pub(crate) trait Backend: Send + Sync + 'static {
    /// Repaint the app (a status changed).
    fn wake(&self);
    fn temp_dir(&self) -> PathBuf;
    /// The chat model's label, when one is set up.
    fn chat_model(&self) -> Option<String>;
    /// That model thinks before answering.
    fn chat_thinks(&self) -> bool {
        false
    }
    /// Whether a `gen.run` request is for video (`None`: not a generator).
    fn gen_is_video(&self, body: &Value) -> Option<bool>;
    fn llm_ask(&self, call: &Incoming) -> Result<Value, String>;
    fn gen_catalog(&self) -> Result<Value, String>;
    fn gen_run(&self, call: &Incoming) -> Result<Value, String>;
    fn gen_thumbs(&self, call: &Incoming) -> Result<Value, String>;
    fn gen_lora_info(&self, call: &Incoming) -> Result<Value, String>;
}

/// The real thing: this app's chat model and generators.
pub(crate) struct AppBackend {
    ctx: egui::Context,
}

impl AppBackend {
    pub fn new(ctx: egui::Context) -> AppBackend {
        AppBackend { ctx }
    }
}

impl Backend for AppBackend {
    fn wake(&self) {
        self.ctx.request_repaint();
    }

    fn temp_dir(&self) -> PathBuf {
        temp_dir()
    }

    fn chat_model(&self) -> Option<String> {
        crate::llm::host_model_label()
    }

    fn chat_thinks(&self) -> bool {
        host_llm().is_some_and(|(model, _)| model.thinks())
    }

    fn gen_is_video(&self, body: &Value) -> Option<bool> {
        #[cfg(not(target_os = "macos"))]
        {
            crate::generate::remote_is_video(body)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = body;
            None
        }
    }

    fn llm_ask(&self, call: &Incoming) -> Result<Value, String> {
        crate::llm::serve_remote(call, &self.ctx)
    }

    fn gen_catalog(&self) -> Result<Value, String> {
        #[cfg(not(target_os = "macos"))]
        {
            crate::generate::remote_catalog()
        }
        #[cfg(target_os = "macos")]
        {
            Err("Generation isn't available on macOS".into())
        }
    }

    fn gen_run(&self, call: &Incoming) -> Result<Value, String> {
        #[cfg(not(target_os = "macos"))]
        {
            crate::generate::remote_run(call, &self.ctx)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = call;
            Err("Generation isn't available on macOS".into())
        }
    }

    fn gen_thumbs(&self, call: &Incoming) -> Result<Value, String> {
        #[cfg(not(target_os = "macos"))]
        {
            crate::generate::remote_thumbs(call, &self.ctx)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = call;
            Err("Generation isn't available on macOS".into())
        }
    }

    fn gen_lora_info(&self, call: &Incoming) -> Result<Value, String> {
        #[cfg(not(target_os = "macos"))]
        {
            crate::generate::remote_lora_info(call)
        }
        #[cfg(target_os = "macos")]
        {
            let _ = call;
            Err("Generation isn't available on macOS".into())
        }
    }
}

/// The chat model and image detail a host serves with: its own AI Model
/// settings (pushed every frame by the app, or read once by the background
/// host).
static HOST_LLM: Mutex<Option<(crate::llm::GemmaModel, crate::llm::ImageDetail)>> = Mutex::new(None);

pub(crate) fn set_host_llm(model: crate::llm::GemmaModel, detail: crate::llm::ImageDetail) {
    *HOST_LLM.lock().unwrap_or_else(|e| e.into_inner()) = Some((model, detail));
}

pub(crate) fn host_llm() -> Option<(crate::llm::GemmaModel, crate::llm::ImageDetail)> {
    *HOST_LLM.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// What the receiver offers the rest of the app.
// ---------------------------------------------------------------------------

struct ReceiverView {
    peer: Arc<Peer>,
    info: HostInfo,
    use_chat: bool,
    use_gen: bool,
}

static RECEIVER: Mutex<Option<ReceiverView>> = Mutex::new(None);

/// The host the app handed over to its background copy on closing, while
/// jobs devices have running on it finish.
static HANDED_OVER: Mutex<Option<host::Host>> = Mutex::new(None);

/// The app is closing and handing its host over (see `finish_handover`).
pub(crate) fn handing_over() -> bool {
    HANDED_OVER.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// After the window has closed: jobs other PCs still have running here
/// finish (up to 30 minutes, or until an update is being installed) before
/// the app exits — their devices then move to the background copy, like the
/// others already did. Also waited for: files still arriving for a job, and
/// the ComfyUI this app started (the background copy may be using it too:
/// it stops with this process).
pub fn finish_handover() {
    let Some(h) = HANDED_OVER.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let started = Instant::now();
    let working = |h: &host::Host| {
        h.shared.busy.load(std::sync::atomic::Ordering::SeqCst) > 0
            || h.shared.sessions.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|p| p.serving())
    };
    while started.elapsed() < Duration::from_secs(30 * 60) && !background::installing() {
        h.shared.wind_down();
        #[cfg(not(target_os = "macos"))]
        let comfy_busy = crate::generate::server_busy_if_ours();
        #[cfg(target_os = "macos")]
        let comfy_busy = false;
        if !working(&h) && !comfy_busy {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    h.stop();
    // Left for the jobs above when the chat closed (see `LlmState`'s drop).
    crate::llm::clear_ai_temp();
}

/// Who images can be sent to right now (identity key hex, name) — refreshed
/// each frame for the browser's and viewer's menus.
static SEND_TARGETS: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

pub(crate) fn send_targets_now() -> Vec<(String, String)> {
    SEND_TARGETS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The host the AI chat runs on: this PC is a receiver connected to a host
/// that shares its chat and has a model set up, and the owner wants to use
/// it. (Peer, host name.)
pub(crate) fn remote_chat() -> Option<(Arc<Peer>, String)> {
    let r = RECEIVER.lock().unwrap_or_else(|e| e.into_inner());
    let v = r.as_ref()?;
    (v.use_chat && v.info.services.chat && v.info.chat_model.is_some() && !v.peer.is_closed())
        .then(|| (v.peer.clone(), v.info.name.clone()))
}

/// The host image (`video` false) or video generation runs on.
pub(crate) fn remote_gen(video: bool) -> Option<(Arc<Peer>, String)> {
    let r = RECEIVER.lock().unwrap_or_else(|e| e.into_inner());
    let v = r.as_ref()?;
    let shared = if video { v.info.services.video } else { v.info.services.images };
    (v.use_gen && shared && !v.peer.is_closed()).then(|| (v.peer.clone(), v.info.name.clone()))
}

/// The host's chat model label, while the chat runs there.
pub(crate) fn remote_chat_model() -> Option<String> {
    let r = RECEIVER.lock().unwrap_or_else(|e| e.into_inner());
    let v = r.as_ref()?;
    (v.use_chat && v.info.services.chat).then(|| v.info.chat_model.clone()).flatten()
}

/// The host's chat model thinks before it answers.
pub(crate) fn remote_chat_thinks() -> bool {
    let r = RECEIVER.lock().unwrap_or_else(|e| e.into_inner());
    r.as_ref().is_some_and(|v| v.info.chat_thinks)
}

/// Wait for a call's last reply, passing progress events and files to
/// `on_event` (the caller owns any file it's handed). Stops the call when
/// `stop()` turns true.
pub(crate) fn finish_call(
    call: &mut Call,
    stop: &dyn Fn() -> bool,
    on_event: &mut dyn FnMut(CallEvent),
) -> Result<Value, String> {
    let mut stopping = false;
    loop {
        if !stopping && stop() {
            stopping = true;
            call.cancel();
        }
        match call.next(Duration::from_millis(200)) {
            Some(CallEvent::Done(v)) => return Ok(v),
            Some(CallEvent::Fail(e)) => return Err(e),
            Some(CallEvent::Closed) => return Err("The connection to the host was lost".into()),
            Some(ev) => on_event(ev),
            None => {}
        }
    }
}

// ---------------------------------------------------------------------------
// The app's Link state.
// ---------------------------------------------------------------------------

/// Link's live state, owned by the app.
pub struct LinkState {
    ctx: egui::Context,
    /// This install's identity, loaded (or made) when Link is first turned on.
    identity: Option<crypto::Identity>,
    identity_err: Option<String>,
    backend: Arc<dyn Backend>,
    files: Arc<files::FileHub>,
    host: Option<host::Host>,
    client: Option<client::Client>,
    /// The settings the running host / receiver were started with.
    applied: Option<LinkSettings>,
    /// What the background host was last told (its config file).
    background_cfg: Option<background::BackgroundConfig>,
    /// The start-with-the-computer registration as last applied.
    autostart_applied: Option<bool>,
    autostart_err: Option<String>,
    /// Receiver: the code being typed, and hosts found on the network.
    pub code_input: String,
    found: Vec<discover::Found>,
    finding: Option<std::sync::mpsc::Receiver<Vec<discover::Found>>>,
    found_once: bool,
    /// Messages for the app's status line.
    notices: Vec<String>,
    /// Folders files were just received into (the app refreshes a browser
    /// showing one).
    received: Vec<PathBuf>,
    /// The remembered devices / hosts, re-read now and then for the tab.
    trust: trust::Trust,
    trust_read: Option<Instant>,
    /// When loading the identity last failed (tried again every 10 s while
    /// Link is on: a key file locked for a moment by a virus scanner).
    identity_failed: Option<Instant>,
    /// Host: the chat model its devices were last told about, and when it
    /// was last checked (a model set up or picked in AI Model reaches them).
    told_model: Option<(Option<String>, bool)>,
    model_checked: Option<Instant>,
}

impl LinkState {
    /// Use the app's real egui context (the state is made before the window
    /// exists). Call before the first `tick`.
    pub fn set_ctx(&mut self, ctx: egui::Context) {
        let wake_ctx = ctx.clone();
        self.backend = Arc::new(AppBackend::new(ctx.clone()));
        self.files = Arc::new(files::FileHub::new(Box::new(move || wake_ctx.request_repaint())));
        self.ctx = ctx;
    }

    /// Nothing is read or written until Link is turned on (but for clearing
    /// the installer's marker, see `background::installing`).
    pub fn new(ctx: egui::Context) -> LinkState {
        background::clear_installing();
        let wake_ctx = ctx.clone();
        LinkState {
            backend: Arc::new(AppBackend::new(ctx.clone())),
            files: Arc::new(files::FileHub::new(Box::new(move || wake_ctx.request_repaint()))),
            ctx,
            identity: None,
            identity_err: None,
            host: None,
            client: None,
            applied: None,
            background_cfg: None,
            autostart_applied: None,
            autostart_err: None,
            code_input: String::new(),
            found: Vec::new(),
            finding: None,
            found_once: false,
            notices: Vec::new(),
            received: Vec::new(),
            trust: trust::Trust::default(),
            trust_read: None,
            told_model: None,
            model_checked: None,
            identity_failed: None,
        }
    }

    /// The identity, loaded or made on first use (and this process's temp
    /// folder cleared then).
    fn ensure_identity(&mut self) -> Option<crypto::Identity> {
        if self.identity.is_none() {
            clear_temp();
            match crypto::Identity::load_or_create(&link_dir()) {
                Ok(id) => {
                    self.identity = Some(id);
                    self.identity_err = None;
                }
                Err(e) => {
                    self.identity_err = Some(e);
                    self.identity_failed = Some(Instant::now());
                }
            }
        }
        self.identity.clone()
    }

    /// Bring Link in line with the settings and collect what happened.
    /// Call once a frame.
    pub fn tick(&mut self, settings: &crate::settings::Settings) {
        let s = &settings.link;
        let on = s.mode != LinkMode::Off;
        set_host_llm(settings.ai_gemma_model, settings.ai_image_detail);
        if on {
            self.files.set_received_dir(received_dir(s));
        }

        // The background host's config follows the settings (written once
        // Link has been used, so it can also learn Link was turned off).
        let cfg = background::BackgroundConfig {
            link: s.clone(),
            chat_model: settings.ai_gemma_model,
            image_detail: settings.ai_image_detail,
        };
        if self.background_cfg.as_ref() != Some(&cfg) {
            if on || background::config_exists() {
                background::save_config(&cfg);
            }
            self.background_cfg = Some(cfg);
        }
        let want_autostart = s.mode == LinkMode::Host && s.background && s.autostart;
        if self.autostart_applied != Some(want_autostart) {
            self.autostart_err = background::sync_autostart(want_autostart).err();
            self.autostart_applied = Some(want_autostart);
        }

        if on
            && self.identity.is_none()
            && self.identity_failed.is_some_and(|t| t.elapsed() > Duration::from_secs(10))
        {
            // Apply the mode again: that loads the identity again.
            self.identity_failed = None;
            self.applied = None;
        }
        self.apply_mode(s);
        self.collect();
        self.publish_receiver(s);
        *SEND_TARGETS.lock().unwrap_or_else(|e| e.into_inner()) = self.send_targets();
        if on && self.trust_read.is_none_or(|t| t.elapsed() > Duration::from_secs(2)) {
            self.trust = trust::Trust::load(&link_dir());
            self.trust_read = Some(Instant::now());
        }
        if let Some(h) = &self.host
            && self.model_checked.is_none_or(|t| t.elapsed() > Duration::from_secs(2))
        {
            self.model_checked = Some(Instant::now());
            let now = (self.backend.chat_model(), self.backend.chat_thinks());
            if self.told_model.as_ref().is_some_and(|told| *told != now) {
                h.shared.announce();
            }
            self.told_model = Some(now);
        }
        if let Some(rx) = &self.finding
            && let Ok(found) = rx.try_recv()
        {
            self.found = found;
            self.finding = None;
        }
        // Keep the tab's timers (code expiry, "connected 3 min") ticking.
        if self.host.is_some() || self.client.is_some() {
            self.ctx.request_repaint_after(Duration::from_secs(1));
        }
    }

    fn apply_mode(&mut self, s: &LinkSettings) {
        // (The port only matters to a host: a receiver's is where it looks.)
        let changed = self
            .applied
            .as_ref()
            .is_none_or(|a| a.mode != s.mode || (s.mode == LinkMode::Host && a.port != s.port));
        if changed {
            self.stop_all();
            self.applied = Some(s.clone());
            // Off means off: no background copy either. As Host this copy
            // takes over from one (it stops taking devices at once, finishes
            // any job, and this host listens as soon as the port is free).
            background::stop_running();
            if s.mode == LinkMode::Off {
                return;
            }
            let Some(identity) = self.ensure_identity() else { return };
            self.files.set_quiet(false);
            match s.mode {
                LinkMode::Off => {}
                LinkMode::Host => {
                    self.host = Some(host::Host::start(
                        identity,
                        link_dir(),
                        host_config(s, false),
                        self.backend.clone(),
                        self.files.clone(),
                    ));
                }
                LinkMode::Receiver => {
                    // Reconnect to the host used last, if there is one.
                    let t = trust::Trust::load(&link_dir());
                    if let Some(h) = t.hosts.iter().max_by_key(|h| h.last_seen) {
                        self.connect_known(h.key.clone(), h.addr.clone());
                    }
                }
            }
            return;
        }
        // Same mode: a host picks up changed shares at once.
        if let Some(h) = &self.host {
            let new = host_config(s, false);
            let mut cur = h.shared.config.lock().unwrap_or_else(|e| e.into_inner());
            if *cur != new {
                *cur = new;
                drop(cur);
                h.shared.announce();
            }
        }
        self.applied = Some(s.clone());
    }

    fn stop_all(&mut self) {
        if let Some(h) = self.host.take() {
            h.stop();
        }
        if let Some(c) = self.client.take() {
            c.stop();
        }
        *RECEIVER.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Gather events from the host / receiver / file sharing as notices.
    fn collect(&mut self) {
        if let Some(h) = &self.host {
            let events = std::mem::take(&mut *h.shared.events.lock().unwrap_or_else(|e| e.into_inner()));
            for e in events {
                self.notices.push(match e {
                    host::HostEvent::Paired(name) => format!("Paired with {name} — it can now use this PC"),
                    host::HostEvent::WrongCode { name, closed: false } => {
                        format!("{name} entered a wrong pairing code")
                    }
                    host::HostEvent::WrongCode { name, closed: true } => {
                        format!("{name} entered wrong codes too often — pairing stopped. Click Add a device for a new code.")
                    }
                });
            }
        }
        for e in self.files.take_events() {
            match e {
                files::FileEvent::Received { from, count, dir } => {
                    self.notices.push(format!(
                        "Received {count} file{} from {from} — in {}",
                        if count == 1 { "" } else { "s" },
                        dir.display()
                    ));
                    self.received.push(dir);
                }
                files::FileEvent::Sent { to, count } => {
                    self.notices.push(format!("Sent {count} file{} to {to}", if count == 1 { "" } else { "s" }))
                }
                files::FileEvent::NotSent { to, why } => self.notices.push(format!("Couldn't send to {to}: {why}")),
            }
        }
    }

    fn publish_receiver(&mut self, s: &LinkSettings) {
        let view = self.client.as_ref().and_then(|c| {
            let peer = c.shared.peer()?;
            let info = c.shared.info()?;
            Some(ReceiverView { peer, info, use_chat: s.use_chat, use_gen: s.use_gen })
        });
        *RECEIVER.lock().unwrap_or_else(|e| e.into_inner()) = view;
    }

    fn connect_known(&mut self, key: String, addr: String) {
        let Some(id) = self.identity.clone() else { return };
        if let Some(c) = self.client.take() {
            c.stop();
        }
        self.client = Some(client::Client::start(
            id,
            link_dir(),
            addr::device_name(),
            addr.clone(),
            Some(key),
            None,
            self.backend.clone(),
            self.files.clone(),
        ));
    }

    fn connect_to(&mut self, target: &str, code: Option<String>) {
        let Some(id) = self.identity.clone() else { return };
        if let Some(c) = self.client.take() {
            c.stop();
        }
        self.client = Some(client::Client::start(
            id,
            link_dir(),
            addr::device_name(),
            target.trim().to_string(),
            None,
            code,
            self.backend.clone(),
            self.files.clone(),
        ));
    }

    fn disconnect(&mut self) {
        if let Some(c) = self.client.take() {
            c.stop();
        }
        *RECEIVER.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    /// Messages for the status line.
    pub fn take_notices(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }

    /// Folders files were received into since the last call.
    pub fn take_received(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.received)
    }

    /// Who files can be sent to right now: (identity key hex, name).
    pub fn send_targets(&self) -> Vec<(String, String)> {
        let mut out = Vec::new();
        if let Some(h) = &self.host {
            let sharing = h.shared.config.lock().unwrap_or_else(|e| e.into_inner()).services.files;
            if sharing {
                for p in h.shared.sessions.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|p| !p.is_closed()) {
                    out.push((crypto::to_hex(&p.key), p.name.clone()));
                }
            }
        }
        if let Some(c) = &self.client
            && let (Some(p), Some(info)) = (c.shared.peer(), c.shared.info())
            && info.services.files
        {
            out.push((crypto::to_hex(&p.key), p.name.clone()));
        }
        out
    }

    /// Send images / videos (with their .txt) to the device with key `to`.
    pub fn send(&mut self, to: &str, paths: Vec<PathBuf>) {
        let peer = self
            .host
            .as_ref()
            .and_then(|h| {
                h.shared
                    .sessions
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .find(|p| !p.is_closed() && crypto::to_hex(&p.key) == to)
                    .cloned()
            })
            .or_else(|| self.client.as_ref().and_then(|c| c.shared.peer()).filter(|p| crypto::to_hex(&p.key) == to));
        let Some(peer) = peer else {
            self.notices.push("That PC isn't connected any more".into());
            return;
        };
        self.notices.push(format!(
            "Offering {} file{} to {}…",
            paths.len(),
            if paths.len() == 1 { "" } else { "s" },
            peer.name
        ));
        let hub = self.files.clone();
        std::thread::spawn(move || hub.send(&peer, &paths, &temp_dir()));
    }

    /// Offers waiting for this PC's owner.
    pub fn pending_offers(&self) -> Vec<files::OfferView> {
        self.files.pending()
    }

    pub fn answer_offer(&self, id: u64, answer: files::Answer) {
        self.files.answer(id, answer);
    }

    /// The app is closing. A host with background hosting on hands over to
    /// a background copy of the app: devices with nothing running here move
    /// to it at once, and jobs devices have running here finish first (see
    /// `finish_handover`).
    pub fn on_exit(&mut self, settings: &crate::settings::Settings) {
        let s = &settings.link;
        // Not while an update is being installed: the background copy would
        // hold the program open.
        let to_background =
            s.mode == LinkMode::Host && s.background && self.host.is_some() && !background::installing();
        // Files offered while the app was open can't be answered any more.
        self.files.set_quiet(true);
        for offer in self.files.pending() {
            self.files.answer(offer.id, files::Answer::Decline);
        }
        if to_background && let Some(h) = self.host.take() {
            h.shared.wind_down();
            *HANDED_OVER.lock().unwrap_or_else(|e| e.into_inner()) = Some(h);
        }
        self.stop_all();
        if to_background {
            let cfg = background::BackgroundConfig {
                link: s.clone(),
                chat_model: settings.ai_gemma_model,
                image_detail: settings.ai_image_detail,
            };
            background::save_config(&cfg);
            if let Err(e) = background::spawn() {
                eprintln!("[link] {e}");
            }
        }
    }

    /// The devices connected to this host: (key hex, address, what they're
    /// doing here).
    fn host_sessions(&self) -> Vec<(String, String, Vec<String>)> {
        let Some(h) = &self.host else { return Vec::new() };
        h.shared
            .sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|p| {
                let doing = p.activity.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(_, l)| l.clone()).collect();
                (crypto::to_hex(&p.key), p.addr.ip().to_string(), doing)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Settings → Link.
// ---------------------------------------------------------------------------

const GREEN: egui::Color32 = egui::Color32::from_rgb(46, 160, 67);
const RED: egui::Color32 = egui::Color32::from_rgb(210, 70, 70);
const AMBER: egui::Color32 = egui::Color32::from_rgb(214, 150, 40);

fn status_line(ui: &mut egui::Ui, color: egui::Color32, text: &str) {
    ui.label(egui::RichText::new(text).color(color).size(12.0));
}

/// A filled pill button (white text on the accent), like the other tabs'.
fn pill(ui: &mut egui::Ui, text: &str, width: f32, enabled: bool) -> bool {
    let btn = egui::Button::new(egui::RichText::new(text).color(egui::Color32::WHITE).strong())
        .fill(ACCENT1())
        .corner_radius(egui::CornerRadius::same(255));
    ui.add_enabled_ui(enabled, |ui| ui.add_sized(egui::vec2(width, 28.0), btn)).inner.clicked()
}

fn ago(secs: u64) -> String {
    let now = trust::now();
    if secs == 0 || secs > now {
        return "never".into();
    }
    let d = now - secs;
    match d {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", d / 60),
        3600..=86_399 => format!("{} h ago", d / 3600),
        _ => format!("{} days ago", d / 86_400),
    }
}

fn mode_picker(ui: &mut egui::Ui, mode: &mut LinkMode) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        for (m, label) in [(LinkMode::Off, "Off"), (LinkMode::Host, "Host"), (LinkMode::Receiver, "Receiver")] {
            let on = *mode == m;
            let text = egui::RichText::new(label).color(if on { egui::Color32::WHITE } else { TEXT() }).strong();
            let mut btn = egui::Button::new(text).corner_radius(egui::CornerRadius::same(255));
            if on {
                btn = btn.fill(ACCENT1());
            }
            if ui.add_sized(egui::vec2(78.0, 28.0), btn).clicked() && !on {
                *mode = m;
                changed = true;
            }
        }
    });
    changed
}

/// The Link tab of the Settings window.
pub fn settings_tab(ui: &mut egui::Ui, settings: &mut crate::settings::Settings, link: &mut LinkState) {
    use crate::settings::{hint, row, section};
    section(ui, "Link", |ui| {
        ui.label(
            egui::RichText::new(
                "Use another PC's AI models and GPU — or share this one's — and send images, \
                 videos and zip archives between the two. Local network only; everything between \
                 the two apps is encrypted end to end.",
            )
            .color(MUTED())
            .size(11.5),
        );
        ui.add_space(8.0);
        mode_picker(ui, &mut settings.link.mode);
        ui.add_space(4.0);
        hint(
            ui,
            match settings.link.mode {
                LinkMode::Off => "Nothing can connect to this PC, and it connects to nothing.",
                LinkMode::Host => "This PC shares its models with the PCs you pair with it.",
                LinkMode::Receiver => "This PC uses the models of a host on your network.",
            },
        );
        if let Some(e) = &link.identity_err {
            status_line(ui, RED, &format!("Link can't start: {e}"));
        }
    });
    match settings.link.mode {
        LinkMode::Off => {}
        LinkMode::Host => host_ui(ui, settings, link),
        LinkMode::Receiver => receiver_ui(ui, settings, link),
    }
    if settings.link.mode != LinkMode::Off {
        // Zips never show in the browser (whose right-click menu sends
        // images and videos), so they're sent from here.
        let targets = link.send_targets();
        if !targets.is_empty() {
            section(ui, "Send files", |ui| {
                for (key, name) in targets {
                    row(ui, &format!("Send a zip to {name}"), Some("Images and videos: right-click them in the browser"), |ui| {
                        if ui.button("Choose zip…").clicked()
                            && let Some(path) = rfd::FileDialog::new().add_filter("Zip archive", &["zip"]).pick_file()
                        {
                            link.send(&key, vec![path]);
                        }
                    });
                }
            });
        }
        section(ui, "Received files", |ui| {
            let dir = received_dir(&settings.link);
            row(ui, "Save files to", Some(&dir.display().to_string()), |ui| {
                if ui.button("Change…").clicked()
                    && let Some(d) = rfd::FileDialog::new().set_directory(&dir).pick_folder()
                {
                    settings.link.received_folder = d.display().to_string();
                }
                if !settings.link.received_folder.is_empty() && ui.button("Default").clicked() {
                    settings.link.received_folder.clear();
                }
            });
        });
    }
}

fn host_ui(ui: &mut egui::Ui, settings: &mut crate::settings::Settings, link: &mut LinkState) {
    use crate::settings::{hint, row, row_sep, section, switch};
    let Some(h) = link.host.as_ref() else { return };
    let shared = h.shared.clone();
    section(ui, "This PC", |ui| {
        row(ui, "Name", None, |ui| {
            let fp = link.identity.as_ref().map(|i| crypto::fingerprint(&i.public)).unwrap_or_default();
            ui.label(egui::RichText::new(addr::device_name()).color(TEXT()).strong().size(13.0))
                .on_hover_text(format!("Identity {fp}"));
        });
        row_sep(ui);
        let addrs = addr::local_addresses();
        let port = settings.link.port;
        let shown: Vec<String> = addrs
            .iter()
            .map(|(a, _)| if port == addr::DEFAULT_PORT { a.to_string() } else { format!("{a}:{port}") })
            .collect();
        row(
            ui,
            "Address",
            Some("Type this on the other PC (Link → Receiver), or find it there with Find hosts."),
            |ui| {
                if shown.is_empty() {
                    ui.label(egui::RichText::new("Not on a network").color(AMBER).size(13.0));
                } else {
                    ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                        for a in &shown {
                            ui.label(egui::RichText::new(a).color(TEXT()).strong().size(13.0));
                        }
                    });
                }
            },
        );
        row_sep(ui);
        let listening = shared.listening.lock().unwrap_or_else(|e| e.into_inner()).clone();
        row(ui, "Status", None, |ui| match listening {
            host::Listening::On => status_line(ui, GREEN, "Ready for your devices"),
            host::Listening::Starting => status_line(ui, MUTED(), "Starting…"),
            host::Listening::WaitingForPort => {
                status_line(ui, AMBER, "Waiting for the background host to finish…")
            }
            host::Listening::Failed(ref e) => status_line(ui, RED, e),
        });
        row_sep(ui);
        row(ui, "Port", Some("Change it only if another program uses this one (the same on both PCs)."), |ui| {
            port_field(ui, &mut settings.link.port);
        });
    });

    section(ui, "Add a device", |ui| {
        let pairing = shared
            .pairing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .filter(|p| p.until > Instant::now())
            .map(|p| (p.code.clone(), p.until.saturating_duration_since(Instant::now())));
        match pairing {
            Some((code, left)) => {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(crypto::show_code(&code)).color(TEXT()).strong().size(30.0).monospace());
                    ui.add_space(10.0);
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new(format!("Expires in {}:{:02}", left.as_secs() / 60, left.as_secs() % 60))
                                .color(MUTED())
                                .size(11.5),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("New code").clicked() {
                                shared.open_pairing();
                            }
                            if ui.button("Stop").clicked() {
                                shared.close_pairing();
                            }
                        });
                    });
                });
                hint(
                    ui,
                    "On the other PC: Settings → Link → Receiver, enter this PC's address, then this \
                     code when it asks. The code works once, for one device.",
                );
            }
            None => {
                if pill(ui, "Add a device", 140.0, true) {
                    shared.open_pairing();
                }
                hint(ui, "Shows a code to type on the other PC. Paired devices are remembered.");
            }
        }
    });

    section(ui, "Share", |ui| {
        let chat_sub = match crate::llm::host_model_label() {
            Some(m) => format!("Runs {m} for them."),
            None => "Set up a model in AI Model first.".to_string(),
        };
        row(ui, "AI chat", Some(&chat_sub), |ui| {
            switch(ui, &mut settings.link.share_chat);
        });
        row_sep(ui);
        if GEN_AVAILABLE {
            row(ui, "Image generation", Some("Flux, Z-Image, SDXL, Anima and Krea 2, with the models set up here."), |ui| {
                switch(ui, &mut settings.link.share_images);
            });
            row_sep(ui);
            row(ui, "Video generation", Some("LTX and Wan."), |ui| {
                switch(ui, &mut settings.link.share_video);
            });
        } else {
            row(ui, "Image and video generation", Some("Not available on macOS."), |_| {});
        }
        row_sep(ui);
        row(ui, "Send and receive files", Some("Images and videos, with their tags."), |ui| {
            switch(ui, &mut settings.link.share_files);
        });
        row_sep(ui);
        row(
            ui,
            "Accept files without asking",
            Some("From paired devices. Needed to receive files while hosting in the background."),
            |ui| {
                switch(ui, &mut settings.link.accept_files_unasked);
            },
        );
    });

    section(ui, "When the app is closed", |ui| {
        row(
            ui,
            "Keep hosting in the background",
            Some("Closing the window leaves a small background copy serving your paired devices. Opening the app takes over."),
            |ui| {
                if switch(ui, &mut settings.link.background).changed() && !settings.link.background {
                    settings.link.autostart = false;
                }
            },
        );
        row_sep(ui);
        row(
            ui,
            "Start with the computer",
            Some("Starts the background host when you log in, so the app never has to be opened here."),
            |ui| {
                if switch(ui, &mut settings.link.autostart).changed() && settings.link.autostart {
                    settings.link.background = true;
                }
            },
        );
        if let Some(e) = &link.autostart_err {
            status_line(ui, RED, &format!("Couldn't set that up: {e}"));
        }
        hint(
            ui,
            "New devices can only pair while the app is open here (the code shows in this tab). \
             Turning Link off stops the background host too.",
        );
    });

    let sessions = link.host_sessions();
    let devices = link.trust.devices.clone();
    section(ui, "Paired devices", |ui| {
        if devices.is_empty() {
            ui.label(egui::RichText::new("None yet").color(MUTED()).size(12.0));
            return;
        }
        let n = devices.len();
        for (i, d) in devices.iter().enumerate() {
            let live = sessions.iter().find(|(k, ..)| *k == d.key);
            let fp = crypto::from_hex(&d.key).map(|k| crypto::fingerprint(&k)).unwrap_or_default();
            let sub = match live {
                Some((_, ip, doing)) if !doing.is_empty() => format!("Connected from {ip} — {}", doing.join(", ")),
                Some((_, ip, _)) => format!("Connected from {ip}"),
                None => format!("Last seen {}", ago(d.last_seen)),
            };
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new(&d.name).color(TEXT()).size(13.0)).on_hover_text(format!("Identity {fp}"));
                    ui.label(
                        egui::RichText::new(&sub).color(if live.is_some() { GREEN } else { MUTED() }).size(10.5),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Forget").on_hover_text("It has to pair again to use this PC").clicked() {
                        let _ = trust::Trust::update(&link_dir(), |t| t.devices.retain(|x| x.key != d.key));
                        if let Some(k) = crypto::from_hex(&d.key) {
                            shared.drop_device(&k);
                        }
                        link.trust_read = None;
                    }
                });
            });
            if i + 1 < n {
                row_sep(ui);
            }
        }
    });
}

fn receiver_ui(ui: &mut egui::Ui, settings: &mut crate::settings::Settings, link: &mut LinkState) {
    use crate::settings::{hint, row, row_sep, section, switch};
    let status = link.client.as_ref().map(|c| c.shared.status());
    let info = link.client.as_ref().and_then(|c| c.shared.info());
    section(ui, "Host", |ui| {
        row(ui, "Address", Some("Shown on the host: Settings → Link → Host."), |ui| {
            let busy = matches!(status, Some(client::Status::Connecting | client::Status::Retrying { .. } | client::Status::Connected));
            if busy {
                if ui.button("Disconnect").clicked() {
                    link.disconnect();
                }
            } else if ui.button("Connect").clicked() {
                let target = addr::with_port(&settings.link.host_addr, settings.link.port);
                // A remembered host at that address reconnects without a code.
                let t = trust::Trust::load(&link_dir());
                match t.hosts.iter().find(|h| addr_matches(&h.addr, &target)) {
                    Some(h) => link.connect_known(h.key.clone(), target),
                    None => link.connect_to(&target, None),
                }
            }
            ui.add(
                egui::TextEdit::singleline(&mut settings.link.host_addr)
                    .desired_width(130.0)
                    .margin(egui::Margin::symmetric(8, 5))
                    .hint_text("192.168.1.23"),
            );
        });
        // On this network.
        ui.horizontal(|ui| {
            let finding = link.finding.is_some();
            if ui.add_enabled(!finding, egui::Button::new("Find hosts")).clicked() {
                let (tx, rx) = std::sync::mpsc::channel();
                let port = settings.link.port;
                let ctx = link.ctx.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(discover::find(port, Duration::from_millis(1500)));
                    ctx.request_repaint();
                });
                link.finding = Some(rx);
                link.found_once = true;
            }
            if finding {
                ui.add(egui::Spinner::new().size(14.0).color(MUTED()));
            } else if link.found_once && link.found.is_empty() {
                ui.label(
                    egui::RichText::new("No hosts answered — check Link is set to Host there")
                        .color(MUTED())
                        .size(11.0),
                );
            }
        });
        for f in link.found.clone() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(&f.name).color(TEXT()).size(12.5)).on_hover_text(format!("Identity {}", f.fp));
                ui.label(egui::RichText::new(&f.ip).color(MUTED()).size(11.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Use").clicked() {
                        settings.link.host_addr = if f.port == addr::DEFAULT_PORT { f.ip.clone() } else { f.addr() };
                        let t = trust::Trust::load(&link_dir());
                        let known = t.hosts.iter().find(|h| {
                            crypto::from_hex(&h.key).is_some_and(|k| crypto::fingerprint(&k) == f.fp)
                        });
                        match known {
                            Some(h) => link.connect_known(h.key.clone(), f.addr()),
                            None => link.connect_to(&f.addr(), None),
                        }
                    }
                });
            });
        }
        row_sep(ui);
        row(ui, "Port", Some("The host's — change it only if it was changed there."), |ui| {
            port_field(ui, &mut settings.link.port);
        });
        row_sep(ui);
        match &status {
            None => status_line(ui, MUTED(), "Not connected"),
            Some(client::Status::Connecting) => status_line(ui, MUTED(), "Connecting…"),
            Some(client::Status::Connected) => {
                let i = info.clone().unwrap_or_default();
                let mut shares = Vec::new();
                if i.services.chat {
                    shares.push("AI chat");
                }
                if i.services.images {
                    shares.push("images");
                }
                if i.services.video {
                    shares.push("video");
                }
                if i.services.files {
                    shares.push("files");
                }
                let what = if shares.is_empty() { "nothing right now".to_string() } else { shares.join(", ") };
                status_line(ui, GREEN, &format!("Connected to {} — it shares {what}", i.name));
            }
            Some(client::Status::Retrying { why }) => status_line(ui, AMBER, why),
            Some(client::Status::Failed(why)) => status_line(ui, RED, why),
            Some(client::Status::NeedCode { host }) => {
                status_line(ui, TEXT(), &format!("{host} doesn't know this PC yet — enter the code it shows"));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut link.code_input)
                            .desired_width(110.0)
                            .margin(egui::Margin::symmetric(8, 5))
                            .font(egui::TextStyle::Monospace)
                            .hint_text("123 456"),
                    );
                    let code = crypto::normalize_code(&link.code_input);
                    let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if (pill(ui, "Pair", 80.0, code.is_some()) || enter)
                        && let Some(code) = code
                    {
                        let target = link.client.as_ref().map(|c| c.target.clone()).unwrap_or_default();
                        link.code_input.clear();
                        link.connect_to(&target, Some(code));
                    }
                });
                hint(ui, &format!("On {host}: Settings → Link → Add a device."));
            }
        }
    });

    let host_name = info.as_ref().map(|i| i.name.clone()).unwrap_or_else(|| "the host".to_string());
    section(ui, &format!("Use {host_name} for"), |ui| {
        let i = info.clone().unwrap_or_default();
        let chat_sub = match (&i.chat_model, i.services.chat) {
            _ if info.is_none() => "When the host shares it.".to_string(),
            (Some(m), true) => format!("The AI chat runs {m} there."),
            (None, true) => "The host has no chat model set up.".to_string(),
            _ => "Not shared by the host.".to_string(),
        };
        row(ui, "AI chat", Some(&chat_sub), |ui| {
            switch(ui, &mut settings.link.use_chat);
        });
        row_sep(ui);
        let gen_sub = if info.is_none() {
            "When the host shares it.".to_string()
        } else if i.services.images || i.services.video {
            "The Generate panels run there, with its models.".to_string()
        } else {
            "Not shared by the host.".to_string()
        };
        row(ui, "Image and video generation", Some(&gen_sub), |ui| {
            switch(ui, &mut settings.link.use_gen);
        });
        hint(ui, "Off runs it on this PC instead (with its own models). Your messages, pictures and prompts go to the host encrypted, and nothing of them is kept there.");
    });

    let hosts = link.trust.hosts.clone();
    section(ui, "Remembered hosts", |ui| {
        if hosts.is_empty() {
            ui.label(egui::RichText::new("None yet").color(MUTED()).size(12.0));
            return;
        }
        let current = link.client.as_ref().and_then(|c| c.shared.peer()).map(|p| crypto::to_hex(&p.key));
        let n = hosts.len();
        for (i, h) in hosts.iter().enumerate() {
            let fp = crypto::from_hex(&h.key).map(|k| crypto::fingerprint(&k)).unwrap_or_default();
            let on = current.as_deref() == Some(h.key.as_str());
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new(&h.name).color(TEXT()).size(13.0)).on_hover_text(format!("Identity {fp}"));
                    let sub = if on { format!("{} · connected", h.addr) } else { format!("{} · last used {}", h.addr, ago(h.last_seen)) };
                    ui.label(egui::RichText::new(sub).color(if on { GREEN } else { MUTED() }).size(10.5));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Forget").on_hover_text("Pairing again will need a new code").clicked() {
                        let _ = trust::Trust::update(&link_dir(), |t| t.hosts.retain(|x| x.key != h.key));
                        if on {
                            link.disconnect();
                        }
                        link.trust_read = None;
                    }
                    if !on && ui.button("Connect").clicked() {
                        settings.link.host_addr = h.addr.clone();
                        link.connect_known(h.key.clone(), h.addr.clone());
                    }
                });
            });
            if i + 1 < n {
                row_sep(ui);
            }
        }
    });
}

/// The port, typed (not dragged), and taken when typing ends: each change
/// restarts a host.
fn port_field(ui: &mut egui::Ui, port: &mut u16) {
    let mut p = *port as u32;
    let field = egui::DragValue::new(&mut p).range(1024..=65535).speed(0.0).update_while_editing(false);
    if ui.add(field).changed() {
        *port = p as u16;
    }
}

/// Two addresses name the same place ("192.168.1.5" = "192.168.1.5:47821").
fn addr_matches(stored: &str, typed: &str) -> bool {
    let norm = |a: &str| {
        addr::parse_host(a).ok().and_then(|v| v.first().copied()).map(|s| s.to_string()).unwrap_or_else(|| a.trim().to_string())
    };
    !typed.trim().is_empty() && norm(stored) == norm(typed)
}

/// The prompt for files another PC offers (shown over everything).
pub fn offer_prompt(ctx: &egui::Context, link: &mut LinkState) {
    let Some(offer) = link.pending_offers().into_iter().next() else { return };
    let id = egui::Id::new(("link_offer", offer.id));
    let mut always = ctx.data(|d| d.get_temp::<bool>(id).unwrap_or(false));
    let mut answer: Option<files::Answer> = None;
    egui::Window::new("link_offer_window")
        .id(id)
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(crate::settings::window_frame())
        .show(ctx, |ui| {
            ui.set_width(360.0);
            // The images and videos; their .txt tags come along.
            let is_txt = |n: &String| n.to_lowercase().ends_with(".txt");
            let media: Vec<&String> = offer.names.iter().filter(|n| !is_txt(n)).collect();
            let tags = offer.names.len() - media.len();
            let n = media.len();
            ui.label(
                egui::RichText::new(format!("{} wants to send you {n} file{}", offer.from, if n == 1 { "" } else { "s" }))
                    .color(TEXT())
                    .strong()
                    .size(15.0),
            );
            ui.add_space(4.0);
            let size = human_size(offer.total);
            let sub = if tags > 0 { format!("{size} · with their tags") } else { size };
            ui.label(egui::RichText::new(sub).color(MUTED()).size(11.5));
            ui.add_space(6.0);
            for name in media.iter().take(6) {
                ui.label(egui::RichText::new(name.as_str()).color(TEXT()).size(12.0));
            }
            if n > 6 {
                ui.label(egui::RichText::new(format!("and {} more", n - 6)).color(MUTED()).size(12.0));
            }
            ui.add_space(8.0);
            ui.checkbox(&mut always, format!("Always accept files from {}", offer.from));
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.add_sized(egui::vec2(100.0, 30.0), egui::Button::new("Decline")).clicked() {
                    answer = Some(files::Answer::Decline);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if pill(ui, "Accept", 100.0, true) {
                        answer = Some(files::Answer::Accept { always });
                    }
                });
            });
        });
    ctx.data_mut(|d| d.insert_temp(id, always));
    if let Some(a) = answer {
        link.answer_offer(offer.id, a);
    }
}

fn human_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} KB", b / 1e3)
    } else {
        format!("{bytes} bytes")
    }
}

#[cfg(test)]
mod tests;
