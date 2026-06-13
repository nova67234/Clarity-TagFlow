//! Update checker. Polls GitHub for the latest Clarity TagFlow release and the
//! latest ComfyUI release, compares them against what's installed, and drives the
//! "Updates" tab in Settings plus the red-dot badge on the top-bar gear.
//!
//! The app update is notify-only: it shows the release notes ("what's new /
//! fixed") and a link to the GitHub download. ComfyUI can be updated in place via
//! [`crate::generate::update_comfyui`].

use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use eframe::egui;
use egui::{Color32, CornerRadius, Margin, RichText, Stroke};

use crate::settings::Settings;
use crate::theme::*;

/// This build's version (compile-time, from Cargo.toml).
const CURRENT: &str = env!("CARGO_PKG_VERSION");
/// The app's own GitHub repo (owner/name).
const APP_REPO: &str = "nova67234/Clarity-TagFlow";
/// ComfyUI's upstream repo.
const COMFY_REPO: &str = "comfyanonymous/ComfyUI";
/// GitHub requires a User-Agent on API requests.
const UA: &str = concat!("Clarity-TagFlow/", env!("CARGO_PKG_VERSION"));

const GREEN: Color32 = Color32::from_rgb(46, 160, 67);

/// A GitHub release: its tag, notes body, and web URL.
#[derive(Clone)]
pub struct ReleaseInfo {
    pub tag: String,
    pub notes: String,
    pub url: String,
}

impl ReleaseInfo {
    fn from_json(v: &serde_json::Value) -> Option<ReleaseInfo> {
        let tag = v.get("tag_name").and_then(|t| t.as_str())?.to_string();
        Some(ReleaseInfo {
            tag,
            notes: v.get("body").and_then(|b| b.as_str()).unwrap_or("").to_string(),
            url: v.get("html_url").and_then(|u| u.as_str()).unwrap_or("").to_string(),
        })
    }
}

/// The result of one update check.
#[derive(Default, Clone)]
pub struct CheckResult {
    pub app_latest: Option<ReleaseInfo>,
    pub comfy_installed: Option<String>,
    pub comfy_latest: Option<ReleaseInfo>,
    /// A non-fatal note shown in the tab if the check (partly) failed.
    pub error: Option<String>,
}

impl CheckResult {
    fn app_update(&self) -> bool {
        self.app_latest.as_ref().is_some_and(|r| is_newer(&r.tag, CURRENT))
    }
    fn comfy_update(&self) -> bool {
        match (&self.comfy_installed, &self.comfy_latest) {
            (Some(inst), Some(latest)) => is_newer(&latest.tag, inst),
            _ => false,
        }
    }
}

/// Messages from the in-place ComfyUI update worker.
enum ComfyMsg {
    Line(String),
    Done(bool),
}

/// Live state for the update feature, owned by `ViewerApp`.
#[derive(Default)]
pub struct UpdateState {
    /// Set once we've kicked off the automatic check for this launch.
    started: bool,
    checking: bool,
    check_rx: Option<Receiver<CheckResult>>,
    pub result: Option<CheckResult>,

    // In-place ComfyUI update worker.
    comfy_updating: bool,
    comfy_rx: Option<Receiver<ComfyMsg>>,
    comfy_log: Vec<String>,
    comfy_done: Option<bool>,
}

impl UpdateState {
    /// Called every frame: kicks off the one-shot launch check and drains the
    /// background workers so the badge + tab stay current even when Settings is
    /// closed.
    pub fn tick(&mut self, ctx: &egui::Context) {
        if !self.started {
            self.started = true;
            self.start_check(ctx);
        }
        if let Some(rx) = &self.check_rx {
            if let Ok(res) = rx.try_recv() {
                self.result = Some(res);
                self.checking = false;
                self.check_rx = None;
                ctx.request_repaint();
            }
        }
        if let Some(rx) = &self.comfy_rx {
            let msgs: Vec<ComfyMsg> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
            let mut finished = false;
            for m in msgs {
                match m {
                    ComfyMsg::Line(s) => self.comfy_log.push(s),
                    ComfyMsg::Done(ok) => {
                        self.comfy_done = Some(ok);
                        self.comfy_updating = false;
                        finished = true;
                    }
                }
            }
            if finished {
                self.comfy_rx = None;
                // A successful update bumps the installed version — re-check so the
                // "update available" state clears.
                if self.comfy_done == Some(true) {
                    self.start_check(ctx);
                }
            }
            ctx.request_repaint();
        }
    }

    /// Should the red dot show on the settings gear? True when an update is
    /// available that the user hasn't dismissed.
    pub fn badge(&self, settings: &Settings) -> bool {
        let Some(r) = &self.result else { return false };
        let app = r.app_update()
            && r.app_latest.as_ref().map(|ri| ri.tag != settings.dismissed_app_version).unwrap_or(false);
        let comfy = r.comfy_update()
            && r.comfy_latest.as_ref().map(|ri| ri.tag != settings.dismissed_comfy_version).unwrap_or(false);
        app || comfy
    }

    /// Spawn the version check in the background.
    fn start_check(&mut self, ctx: &egui::Context) {
        if self.checking {
            return;
        }
        self.checking = true;
        let (tx, rx) = mpsc::channel();
        self.check_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(run_check());
            ctx.request_repaint();
        });
    }

    /// Spawn the in-place ComfyUI update in the background.
    fn start_comfy_update(&mut self, ctx: &egui::Context) {
        if self.comfy_updating {
            return;
        }
        self.comfy_updating = true;
        self.comfy_done = None;
        self.comfy_log.clear();
        let (tx, rx) = mpsc::channel();
        self.comfy_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let send = |s: String| {
                let _ = tx.send(ComfyMsg::Line(s));
                ctx.request_repaint();
            };
            let ok = crate::generate::update_comfyui(&send);
            let _ = tx.send(ComfyMsg::Done(ok));
            ctx.request_repaint();
        });
    }
}

/// Run both version checks (blocking; called on a worker thread).
fn run_check() -> CheckResult {
    // Must select NativeTls explicitly: the crate is built without the rustls
    // feature, so the default provider would panic on an https connection.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(15)))
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .build()
        .into();

    let mut out = CheckResult::default();
    let mut errs: Vec<String> = Vec::new();

    match fetch_release(&agent, APP_REPO) {
        Ok(r) => out.app_latest = Some(r),
        Err(e) => errs.push(format!("app: {e}")),
    }

    out.comfy_installed = crate::generate::comfyui_installed_version();
    // Only bother fetching ComfyUI's latest if it's actually installed.
    if out.comfy_installed.is_some() {
        match fetch_release(&agent, COMFY_REPO) {
            Ok(r) => out.comfy_latest = Some(r),
            Err(e) => errs.push(format!("comfyui: {e}")),
        }
    }

    if !errs.is_empty() {
        out.error = Some(errs.join("; "));
    }
    out
}

/// The newest release for a repo: prefer `/releases/latest` (stable releases),
/// fall back to the newest of all releases (includes pre-releases like betas).
fn fetch_release(agent: &ureq::Agent, repo: &str) -> Result<ReleaseInfo, String> {
    let (code, v) = get_json(agent, &format!("https://api.github.com/repos/{repo}/releases/latest"))?;
    if code == 200 {
        if let Some(r) = ReleaseInfo::from_json(&v) {
            return Ok(r);
        }
    }
    // 404 here means "no non-prerelease release" — fall back to the full list.
    let (code, v) = get_json(agent, &format!("https://api.github.com/repos/{repo}/releases?per_page=1"))?;
    if code == 200 {
        if let Some(r) = v.as_array().and_then(|a| a.first()).and_then(ReleaseInfo::from_json) {
            return Ok(r);
        }
    }
    Err(format!("no releases found (HTTP {code})"))
}

/// GET a GitHub API URL, returning (status, parsed JSON). Status errors are not
/// treated as transport errors so callers can branch on 404.
fn get_json(agent: &ureq::Agent, url: &str) -> Result<(u16, serde_json::Value), String> {
    let mut resp = agent
        .get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .header("User-Agent", UA)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| e.to_string())?;
    let code = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
    let v = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    Ok((code, v))
}

/// True if `latest` is a strictly newer version than `current`. Parses a leading
/// dotted-numeric core (ignoring a leading `v`); a final release outranks a
/// pre-release of the same core, and two pre-releases compare by their numbers.
fn is_newer(latest: &str, current: &str) -> bool {
    let (lc, cc) = (core_nums(latest), core_nums(current));
    let n = lc.len().max(cc.len());
    for i in 0..n {
        let (l, c) = (lc.get(i).copied().unwrap_or(0), cc.get(i).copied().unwrap_or(0));
        if l != c {
            return l > c;
        }
    }
    // Equal cores: handle pre-release ordering (e.g. 5.3.1 > 5.3.1-beta.2).
    match (latest.contains('-'), current.contains('-')) {
        (false, true) => true,
        (true, false) => false,
        (false, false) => false,
        (true, true) => {
            let (lp, cp) = (pre_nums(latest), pre_nums(current));
            let m = lp.len().max(cp.len());
            for i in 0..m {
                let (l, c) = (lp.get(i).copied().unwrap_or(0), cp.get(i).copied().unwrap_or(0));
                if l != c {
                    return l > c;
                }
            }
            false
        }
    }
}

/// Numeric components of a version's core (before any `-pre`).
fn core_nums(s: &str) -> Vec<u64> {
    s.trim()
        .trim_start_matches(['v', 'V'])
        .split('-')
        .next()
        .unwrap_or("")
        .split('.')
        .map(|p| p.parse::<u64>().unwrap_or(0))
        .collect()
}

/// Digit groups within the pre-release part (after `-`), e.g. "beta.2" -> [2].
fn pre_nums(s: &str) -> Vec<u64> {
    match s.split_once('-') {
        Some((_, pre)) => pre
            .split(|c: char| !c.is_ascii_digit())
            .filter(|x| !x.is_empty())
            .map(|x| x.parse().unwrap_or(0))
            .collect(),
        None => Vec::new(),
    }
}

/// The "Updates" tab body in Settings.
pub fn updates_tab(ui: &mut egui::Ui, state: &mut UpdateState, settings: &mut Settings) {
    let mut do_check = false;
    let mut do_comfy_update = false;
    let mut do_dismiss = false;
    let checking = state.checking;
    let comfy_updating = state.comfy_updating;

    // --- App update ---
    crate::settings::section(ui, "App update", |ui| {
        ui.label(RichText::new("Clarity TagFlow").color(TEXT()).strong());
        ui.label(RichText::new(format!("Installed: v{CURRENT}")).color(MUTED()).size(12.0));
        ui.add_space(4.0);
        match &state.result {
            Some(r) => {
                if let Some(ri) = &r.app_latest {
                    if is_newer(&ri.tag, CURRENT) {
                        ui.label(RichText::new(format!("Update available — {}", ri.tag)).color(GREEN).strong());
                        ui.add_space(2.0);
                        crate::arrow_link(ui, "Download from GitHub", &ri.url, None);
                        ui.add_space(6.0);
                        ui.label(RichText::new("What's new").color(TEXT()).strong().size(12.0));
                        ui.add_space(2.0);
                        notes_box(ui, "app_notes", &ri.notes);
                    } else {
                        ui.label(RichText::new("You're on the latest version.").color(MUTED()).size(12.0));
                    }
                } else if let Some(e) = &r.error {
                    ui.label(RichText::new(format!("Couldn't check: {e}")).color(MUTED()).size(11.5));
                }
            }
            None if checking => {
                ui.label(RichText::new("Checking…").color(MUTED()).size(12.0));
            }
            None => {
                ui.label(RichText::new("Not checked yet.").color(MUTED()).size(12.0));
            }
        }
    });

    // --- ComfyUI ---
    crate::settings::section(ui, "ComfyUI", |ui| {
        let installed = state.result.as_ref().and_then(|r| r.comfy_installed.clone());
        match &installed {
            Some(v) => ui.label(RichText::new(format!("Installed: {v}")).color(MUTED()).size(12.0)),
            None => ui.label(RichText::new("Not installed (set it up from a Generate tab).").color(MUTED()).size(12.0)),
        };
        if let Some(r) = &state.result {
            if let (Some(inst), Some(latest)) = (&r.comfy_installed, &r.comfy_latest) {
                if is_newer(&latest.tag, inst) {
                    ui.add_space(2.0);
                    ui.label(RichText::new(format!("Update available — {}", latest.tag)).color(GREEN).strong());
                    ui.add_space(6.0);
                    if !latest.notes.trim().is_empty() {
                        ui.label(RichText::new("What's new").color(TEXT()).strong().size(12.0));
                        ui.add_space(2.0);
                        notes_box(ui, "comfy_notes", &latest.notes);
                        ui.add_space(6.0);
                    }
                    let btn = egui::Button::new(
                        RichText::new(if comfy_updating { "Updating…" } else { "Update ComfyUI" })
                            .color(Color32::WHITE)
                            .strong(),
                    )
                    .fill(ACCENT1());
                    if ui.add_enabled(!comfy_updating, btn).clicked() {
                        do_comfy_update = true;
                    }
                } else {
                    ui.add_space(2.0);
                    ui.label(RichText::new("ComfyUI is up to date.").color(MUTED()).size(12.0));
                }
            }
        }
        // Live progress / result of the in-place update.
        if comfy_updating || state.comfy_done.is_some() {
            ui.add_space(6.0);
            if let Some(last) = state.comfy_log.last() {
                ui.label(RichText::new(last).color(MUTED()).size(11.0));
            }
            if let Some(ok) = state.comfy_done {
                let (txt, col) = if ok {
                    ("ComfyUI updated.", GREEN)
                } else {
                    ("Update failed — see the log above.", Color32::from_rgb(220, 70, 70))
                };
                ui.label(RichText::new(txt).color(col).size(11.5));
            }
        }
    });

    // --- Footer: re-check + dismiss ---
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        if ui.add_enabled(!checking, egui::Button::new(RichText::new("Check now").size(12.0))).clicked() {
            do_check = true;
        }
        let can_dismiss = state.result.as_ref().map(|r| r.app_update() || r.comfy_update()).unwrap_or(false);
        if can_dismiss {
            ui.add_space(6.0);
            if ui.add(egui::Button::new(RichText::new("Dismiss").size(12.0)).frame(false)).clicked() {
                do_dismiss = true;
            }
        }
        if checking {
            ui.add_space(6.0);
            ui.label(RichText::new("checking…").color(MUTED()).size(11.0));
        }
    });
    hint(ui, "Clarity TagFlow updates link to the GitHub download. ComfyUI updates in place, keeping your downloaded models.");

    if do_check {
        state.start_check(ui.ctx());
    }
    if do_comfy_update {
        state.start_comfy_update(ui.ctx());
    }
    if do_dismiss {
        if let Some(r) = &state.result {
            if let Some(ri) = &r.app_latest {
                settings.dismissed_app_version = ri.tag.clone();
            }
            if let Some(ri) = &r.comfy_latest {
                settings.dismissed_comfy_version = ri.tag.clone();
            }
        }
    }
}

/// A bordered, scrollable box for release notes.
fn notes_box(ui: &mut egui::Ui, id: &str, text: &str) {
    let body = text.trim();
    let body = if body.is_empty() { "(no notes provided)" } else { body };
    egui::Frame::new()
        .fill(FIELD())
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(8))
        .stroke(Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt(id)
                .max_height(150.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(egui::Label::new(RichText::new(body).color(MUTED()).size(11.5)).wrap());
                });
        });
}

/// A small muted explanatory line (mirrors settings::hint).
fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(text).color(MUTED()).size(11.0));
}
