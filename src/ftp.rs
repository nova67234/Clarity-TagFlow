//! FTP/FTPS remote-folder browsing (Settings → FTP/FTPS).
//!
//! When FTP mode is enabled, the top bar's folder button becomes a remote
//! browser: navigate the server's directories and "Load images" pulls that
//! directory's media into a local cache folder, which then loads through the
//! normal folder pipeline — so viewing, tagging, favorites, and generation all
//! work unchanged on the cached copies. (Tags written while browsing land next
//! to the cached files; nothing is uploaded back to the server.)
//!
//! Connections use `suppaftp` (sync, run on background threads) with
//! native-tls for FTPS — the same TLS stack as the app's HTTP client. Each
//! operation opens a fresh connection: simpler than a connection pool, and
//! robust against servers that drop idle control channels.
//!
//! The password is stored encrypted at rest via `src/secret.rs` (Windows
//! DPAPI), like the Civitai/Gelbooru keys — never in the plain settings file.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};

use eframe::egui;
use suppaftp::native_tls::TlsConnector;
use suppaftp::{NativeTlsConnector, NativeTlsFtpStream};

use crate::theme::{ACCENT1, EDGE, FIELD, MUTED, PANEL, TEXT};

/// Everything needed to open one connection. Snapshotted from `Settings` (+ the
/// decrypted password) when an operation starts, so background threads never
/// touch UI state.
#[derive(Clone)]
pub struct FtpParams {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub pass: String,
    pub secure: bool,
}

/// One remote directory entry, parsed from the server's LIST output.
struct RemoteEntry {
    name: String,
    is_dir: bool,
    size: usize,
}

/// Progress messages from the background download worker.
enum DlMsg {
    /// (files done, files total, current file name)
    Progress(usize, usize, String),
    /// All files fetched into the returned local cache directory.
    Done(PathBuf),
    Err(String),
}

/// UI + worker state for the FTP browser. Lives on `ViewerApp`.
#[derive(Default)]
pub struct FtpState {
    /// Decrypted password (stored encrypted on disk; loaded on first use).
    pub password: String,
    pass_loaded: bool,

    /// Whether the remote-browser popup is open (the top bar's folder button
    /// opens it while FTP mode is enabled).
    pub browser_open: bool,
    /// Current remote directory ("/" rooted).
    cwd: String,
    /// Entries of `cwd`, once listed.
    entries: Option<Vec<RemoteEntry>>,
    list_rx: Option<Receiver<Result<Vec<RemoteEntry>, String>>>,
    error: Option<String>,

    /// In-flight directory download.
    dl_rx: Option<Receiver<DlMsg>>,
    dl_progress: (usize, usize),
    dl_current: String,
    /// Set when a download finishes — `main.rs` takes it and loads the folder.
    loaded_dir: Option<PathBuf>,

    /// "Test connection" result for the settings tab.
    test_rx: Option<Receiver<Result<usize, String>>>,
    pub test_status: Option<Result<String, String>>,
}

impl FtpState {
    /// The decrypted password, loading it from the encrypted file on first use.
    pub fn ensure_password(&mut self) -> &mut String {
        if !self.pass_loaded {
            self.pass_loaded = true;
            self.password = load_password();
        }
        &mut self.password
    }

    /// Build connection params from the settings + stored password.
    pub fn params(&mut self, settings: &crate::settings::Settings) -> FtpParams {
        self.ensure_password();
        FtpParams {
            host: settings.ftp_host.trim().to_string(),
            port: settings.ftp_port,
            user: settings.ftp_user.trim().to_string(),
            pass: self.password.clone(),
            secure: settings.ftp_secure,
        }
    }

    /// Take a finished download's local folder (the app then loads it).
    pub fn take_loaded(&mut self) -> Option<PathBuf> {
        self.loaded_dir.take()
    }

    /// Kick off a "test connection" for the settings tab.
    pub fn start_test(&mut self, params: FtpParams) {
        let (tx, rx) = mpsc::channel();
        self.test_rx = Some(rx);
        self.test_status = None;
        std::thread::spawn(move || {
            let result = (|| {
                let mut ftp = connect(&params)?;
                let n = list_dir(&mut ftp, "/")?.len();
                let _ = ftp.quit();
                Ok(n)
            })();
            let _ = tx.send(result);
        });
    }

    /// Poll the test worker (call each frame the settings tab shows).
    pub fn poll_test(&mut self) -> bool {
        let mut done = false;
        if let Some(rx) = &self.test_rx {
            if let Ok(r) = rx.try_recv() {
                self.test_status = Some(match r {
                    Ok(n) => Ok(format!("Connected — {n} entries in /")),
                    Err(e) => Err(e),
                });
                self.test_rx = None;
                done = true;
            }
        }
        done
    }

    /// True while the test worker runs.
    pub fn testing(&self) -> bool {
        self.test_rx.is_some()
    }

    fn start_list(&mut self, params: FtpParams, path: String) {
        let (tx, rx) = mpsc::channel();
        self.list_rx = Some(rx);
        self.entries = None;
        self.error = None;
        std::thread::spawn(move || {
            let result = (|| {
                let mut ftp = connect(&params)?;
                let entries = list_dir(&mut ftp, &path)?;
                let _ = ftp.quit();
                Ok(entries)
            })();
            let _ = tx.send(result);
        });
    }

    fn start_download(&mut self, params: FtpParams, path: String, files: Vec<String>) {
        let (tx, rx) = mpsc::channel();
        self.dl_rx = Some(rx);
        self.dl_progress = (0, files.len());
        self.dl_current.clear();
        std::thread::spawn(move || {
            let dest = cache_dir(&params.host, &path);
            if let Err(e) = std::fs::create_dir_all(&dest) {
                let _ = tx.send(DlMsg::Err(format!("Create cache dir: {e}")));
                return;
            }
            let result = (|| -> Result<(), String> {
                let mut ftp = connect(&params)?;
                ftp.cwd(&path).map_err(|e| format!("cd {path}: {e}"))?;
                let total = files.len();
                for (i, name) in files.iter().enumerate() {
                    let _ = tx.send(DlMsg::Progress(i, total, name.clone()));
                    let local = dest.join(name);
                    // Skip files already cached (size unknown here — presence is
                    // the cheap check; delete the cache folder to force refetch).
                    if local.exists() {
                        continue;
                    }
                    let data = ftp
                        .retr_as_buffer(name)
                        .map_err(|e| friendly(format!("{name}: {e}")))?
                        .into_inner();
                    // Stage to .part then rename, so an aborted transfer never
                    // looks like a cached file.
                    let part = dest.join(format!("{name}.part"));
                    std::fs::write(&part, &data).map_err(|e| format!("{name}: {e}"))?;
                    std::fs::rename(&part, &local).map_err(|e| format!("{name}: {e}"))?;
                }
                let _ = ftp.quit();
                Ok(())
            })();
            let _ = tx.send(match result {
                Ok(()) => DlMsg::Done(dest),
                Err(e) => DlMsg::Err(e),
            });
        });
    }
}

/// Open (and log into) a connection. FTPS upgrades the control channel with
/// native-tls before login, so credentials never travel in the clear.
fn connect(p: &FtpParams) -> Result<NativeTlsFtpStream, String> {
    if p.host.is_empty() {
        return Err("No host configured (Settings → FTP/FTPS)".into());
    }
    let addr = format!("{}:{}", p.host, p.port);
    let ftp = NativeTlsFtpStream::connect(&addr).map_err(|e| format!("Connect {addr}: {e}"))?;
    let mut ftp = if p.secure {
        let tls = TlsConnector::new().map_err(|e| format!("TLS init: {e}"))?;
        ftp.into_secure(NativeTlsConnector::from(tls), &p.host)
            .map_err(|e| format!("FTPS handshake: {e}"))?
    } else {
        ftp
    };
    let user = if p.user.is_empty() { "anonymous" } else { &p.user };
    ftp.login(user, &p.pass).map_err(|e| format!("Login: {e}"))?;
    Ok(ftp)
}

/// Make an FTP error message actionable. The notable case: hardened FTPS
/// servers that require TLS session resumption on data connections — a feature
/// native-tls can't provide — fail with a cryptic 425; explain the workaround.
fn friendly(e: String) -> String {
    if e.contains("resumption") {
        format!(
            "{e} — this server requires TLS session resumption, which isn't \
             supported yet. Use plain FTP, or relax the server's TLS reuse rule."
        )
    } else {
        e
    }
}

/// LIST a directory into parsed entries (dirs first, then files, both sorted).
fn list_dir(ftp: &mut NativeTlsFtpStream, path: &str) -> Result<Vec<RemoteEntry>, String> {
    let lines = ftp.list(Some(path)).map_err(|e| friendly(format!("List {path}: {e}")))?;
    let mut entries: Vec<RemoteEntry> = lines
        .iter()
        .filter_map(|l| suppaftp::list::File::try_from(l.as_str()).ok())
        .filter(|f| {
            let n = f.name();
            n != "." && n != ".."
        })
        .map(|f| RemoteEntry {
            name: f.name().to_string(),
            is_dir: f.is_directory(),
            size: f.size(),
        })
        .collect();
    entries.sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.to_lowercase().cmp(&b.name.to_lowercase())));
    Ok(entries)
}

/// The local cache folder mirroring `host:path` — under the per-user cache dir
/// so it's writable and survivable, but disposable.
fn cache_dir(host: &str, path: &str) -> PathBuf {
    let safe: String = path
        .trim_matches('/')
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect();
    let leaf = if safe.is_empty() { "root".to_string() } else { safe };
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("Clarity TagFlow")
        .join("ftp")
        .join(host)
        .join(leaf)
}

/// Path of the encrypted FTP password file in the app config dir.
fn password_path() -> PathBuf {
    dirs::config_dir()
        .map(|p| p.join("Clarity TagFlow").join("ftp_password.dat"))
        .unwrap_or_else(|| PathBuf::from("ftp_password.dat"))
}

fn load_password() -> String {
    std::fs::read_to_string(password_path())
        .ok()
        .map(|s| crate::secret::unprotect(s.trim()))
        .unwrap_or_default()
}

/// Save the FTP password encrypted. An empty password removes the file.
pub fn save_password(pass: &str) {
    let path = password_path();
    if pass.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, crate::secret::protect(pass));
}

/// Join a child onto a "/"-rooted remote path.
fn join_remote(cwd: &str, child: &str) -> String {
    if cwd == "/" {
        format!("/{child}")
    } else {
        format!("{}/{child}", cwd.trim_end_matches('/'))
    }
}

/// Parent of a "/"-rooted remote path ("/" stays "/").
fn parent_remote(cwd: &str) -> String {
    let trimmed = cwd.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(0) | None => "/".to_string(),
        Some(i) => trimmed[..i].to_string(),
    }
}

/// Render the remote-browser popup (when open) and drive the workers. Call once
/// per frame; returns nothing — a finished download lands in [`FtpState::take_loaded`].
pub fn show_browser(ctx: &egui::Context, state: &mut FtpState, settings: &crate::settings::Settings) {
    // Drain the list worker.
    if let Some(rx) = &state.list_rx {
        if let Ok(result) = rx.try_recv() {
            state.list_rx = None;
            match result {
                Ok(entries) => state.entries = Some(entries),
                Err(e) => state.error = Some(e),
            }
        }
    }
    // Drain the download worker (into a Vec first so the `&state.dl_rx` borrow
    // ends before the handlers mutate `state`).
    let msgs: Vec<DlMsg> = match &state.dl_rx {
        Some(rx) => std::iter::from_fn(|| rx.try_recv().ok()).collect(),
        None => Vec::new(),
    };
    for msg in msgs {
        match msg {
            DlMsg::Progress(done, total, name) => {
                state.dl_progress = (done, total);
                state.dl_current = name;
            }
            DlMsg::Done(dir) => {
                state.dl_rx = None;
                state.loaded_dir = Some(dir);
                state.browser_open = false;
            }
            DlMsg::Err(e) => {
                state.dl_rx = None;
                state.error = Some(e);
            }
        }
    }
    if state.list_rx.is_some() || state.dl_rx.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(120));
    }

    if !state.browser_open {
        return;
    }

    // First open: start at "/".
    if state.entries.is_none() && state.list_rx.is_none() && state.error.is_none() {
        if state.cwd.is_empty() {
            state.cwd = "/".into();
        }
        let params = state.params(settings);
        let cwd = state.cwd.clone();
        state.start_list(params, cwd);
    }

    let mut open = state.browser_open;
    let mut navigate: Option<String> = None;
    let mut load_files: Option<Vec<String>> = None;

    egui::Window::new("")
        .id(egui::Id::new("ftp_browser"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .frame(
            egui::Frame::new()
                .fill(PANEL())
                .corner_radius(egui::CornerRadius::same(16))
                .inner_margin(egui::Margin::same(18))
                .stroke(egui::Stroke::new(1.0, EDGE()))
                .shadow(egui::epaint::Shadow {
                    offset: [0, 6],
                    blur: 20,
                    spread: 0,
                    color: egui::Color32::from_black_alpha(150),
                }),
        )
        .show(ctx, |ui| {
            ui.set_width(430.0);

            // Title row: globe icon + host, close button on the right.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/ftp_browser.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(TEXT()),
                );
                let title = if settings.ftp_host.trim().is_empty() {
                    "FTP Browser".to_string()
                } else {
                    format!("FTP — {}", settings.ftp_host.trim())
                };
                ui.heading(egui::RichText::new(title).color(TEXT()).strong().size(17.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/close.svg"))
                                .fit_to_exact_size(egui::vec2(22.0, 22.0))
                                .tint(TEXT()),
                        ).frame(false))
                        .clicked()
                    {
                        open = false;
                    }
                });
            });
            ui.add_space(10.0);

            // Path row: up button + current remote path.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                let up = crate::svg_button(
                    ui,
                    egui::include_image!("../icons/folderup.svg"),
                    "Up one directory",
                    22.0,
                    crate::theme::icon_tint(MUTED()),
                );
                if up.clicked() && state.cwd != "/" {
                    navigate = Some(parent_remote(&state.cwd));
                }
                ui.label(egui::RichText::new(&state.cwd).color(MUTED()).size(12.5));
            });
            ui.add_space(6.0);

            // Directory listing.
            egui::Frame::new()
                .fill(FIELD())
                .corner_radius(egui::CornerRadius::same(12))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::ScrollArea::vertical().max_height(300.0).auto_shrink([false, true]).show(ui, |ui| {
                        if let Some(err) = &state.error {
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new(err).color(egui::Color32::from_rgb(210, 70, 70)).size(12.0));
                            ui.add_space(4.0);
                            if ui.button("Retry").clicked() {
                                state.error = None;
                            }
                        } else if state.entries.is_none() {
                            ui.add_space(12.0);
                            ui.horizontal(|ui| {
                                ui.add(egui::Spinner::new().size(16.0).color(MUTED()));
                                ui.label(egui::RichText::new("Listing…").color(MUTED()).size(12.0));
                            });
                            ui.add_space(12.0);
                        } else if let Some(entries) = &state.entries {
                            if entries.is_empty() {
                                ui.add_space(12.0);
                                ui.label(egui::RichText::new("(empty directory)").color(MUTED()).size(12.0));
                                ui.add_space(12.0);
                            }
                            for e in entries {
                                let icon = if e.is_dir {
                                    egui::include_image!("../icons/folder.svg")
                                } else if crate::is_media(std::path::Path::new(&e.name)) {
                                    egui::include_image!("../icons/image.svg")
                                } else {
                                    egui::include_image!("../icons/metadata.svg")
                                };
                                let resp = ui
                                    .horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 8.0;
                                        ui.add(
                                            egui::Image::new(icon)
                                                .fit_to_exact_size(egui::vec2(16.0, 16.0))
                                                .tint(crate::theme::icon_tint(MUTED())),
                                        );
                                        ui.label(egui::RichText::new(&e.name).color(TEXT()).size(13.0));
                                        if !e.is_dir && e.size > 0 {
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                ui.label(
                                                    egui::RichText::new(human_size(e.size)).color(MUTED()).size(11.0),
                                                );
                                            });
                                        }
                                    })
                                    .response;
                                if e.is_dir {
                                    let resp = resp.interact(egui::Sense::click());
                                    if resp.hovered() {
                                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                    }
                                    if resp.clicked() {
                                        navigate = Some(join_remote(&state.cwd, &e.name));
                                    }
                                }
                            }
                        }
                    });
                });
            ui.add_space(10.0);

            // Footer: media count + the Load button (or download progress).
            let media: Vec<String> = state
                .entries
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .filter(|e| !e.is_dir && crate::is_media(std::path::Path::new(&e.name)))
                .map(|e| e.name.clone())
                .collect();

            if state.dl_rx.is_some() {
                let (done, total) = state.dl_progress;
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(16.0).color(ACCENT1()));
                    ui.label(
                        egui::RichText::new(format!("Downloading {}/{} — {}", done + 1, total, state.dl_current))
                            .color(MUTED())
                            .size(12.0),
                    );
                });
                let frac = if total == 0 { 0.0 } else { done as f32 / total as f32 };
                ui.add(egui::ProgressBar::new(frac).desired_height(6.0));
            } else {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("{} media file(s) here", media.len()))
                            .color(MUTED())
                            .size(12.0),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let btn = egui::Button::new(
                            egui::RichText::new(format!("Load {} images", media.len()))
                                .color(egui::Color32::WHITE)
                                .strong(),
                        )
                        .fill(ACCENT1())
                        .corner_radius(egui::CornerRadius::same(10));
                        if ui.add_enabled(!media.is_empty(), btn).clicked() {
                            load_files = Some(media);
                        }
                    });
                });
            }
        });

    // Apply the deferred actions (borrowck: outside the closure).
    if let Some(dir) = navigate {
        state.cwd = dir.clone();
        let params = state.params(settings);
        state.start_list(params, dir);
    }
    if let Some(files) = load_files {
        let params = state.params(settings);
        let cwd = state.cwd.clone();
        state.start_download(params, cwd, files);
    }
    state.browser_open = open;
    // Re-list on next open so a fresh session starts from a live listing.
    if !state.browser_open && state.dl_rx.is_none() {
        state.entries = None;
        state.error = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_path_helpers() {
        assert_eq!(join_remote("/", "pub"), "/pub");
        assert_eq!(join_remote("/pub", "example"), "/pub/example");
        assert_eq!(parent_remote("/pub/example"), "/pub");
        assert_eq!(parent_remote("/pub"), "/");
        assert_eq!(parent_remote("/"), "/");
    }

    /// Live integration test against Rebex's public read-only FTP test server
    /// (network + external service → `#[ignore]`; run explicitly with
    /// `cargo test ftp -- --ignored`). Exercises connect, login, LIST parsing,
    /// directory navigation, and a real file download.
    #[test]
    #[ignore]
    fn lists_and_downloads_from_public_test_server() {
        let params = FtpParams {
            host: "test.rebex.net".into(),
            port: 21,
            user: "demo".into(),
            pass: "password".into(),
            secure: false,
        };
        let mut ftp = connect(&params).expect("connect + login");
        let root = list_dir(&mut ftp, "/").expect("list /");
        eprintln!("root: {:?}", root.iter().map(|e| (&e.name, e.is_dir)).collect::<Vec<_>>());
        assert!(root.iter().any(|e| e.name == "readme.txt"), "readme.txt expected in /");

        let data = ftp
            .retr_as_buffer("readme.txt")
            .expect("download readme.txt")
            .into_inner();
        assert!(!data.is_empty(), "readme.txt should have content");
        eprintln!("downloaded readme.txt: {} bytes", data.len());
        let _ = ftp.quit();
    }

    /// FTPS (explicit TLS): the control-channel upgrade + login must work. The
    /// Rebex server then demands TLS session resumption on DATA connections —
    /// a hardening native-tls can't satisfy — so a LIST either succeeds (a
    /// laxer server) or fails with our friendly resumption message; anything
    /// else is a real regression.
    #[test]
    #[ignore]
    fn connects_over_ftps() {
        let params = FtpParams {
            host: "test.rebex.net".into(),
            port: 21,
            user: "demo".into(),
            pass: "password".into(),
            secure: true,
        };
        let mut ftp = connect(&params).expect("FTPS connect + login");
        match list_dir(&mut ftp, "/") {
            Ok(root) => {
                eprintln!("ftps root entries: {}", root.len());
                assert!(!root.is_empty());
            }
            Err(e) => {
                eprintln!("ftps list: {e}");
                assert!(e.contains("resumption"), "unexpected FTPS error: {e}");
            }
        }
        let _ = ftp.quit();
    }
}

/// "12.3 MB"-style size.
fn human_size(bytes: usize) -> String {
    let b = bytes as f64;
    if b >= 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1} GB", b / (1024.0 * 1024.0 * 1024.0))
    } else if b >= 1024.0 * 1024.0 {
        format!("{:.1} MB", b / (1024.0 * 1024.0))
    } else if b >= 1024.0 {
        format!("{:.0} KB", b / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
