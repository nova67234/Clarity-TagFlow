//! Hosting with no window: the same program started with
//! `--link-background`. It starts when the app closes with "Keep hosting
//! when the app is closed" on, or at login with "Start with the computer".
//! It serves remembered devices only (pairing needs the app open, where the
//! code shows), and stops when the app opens (handing the port over) or when
//! Link is turned off there.
//!
//! The app talks to it over a control socket on 127.0.0.1, whose port and a
//! random token are in `background.json` in the Link folder.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::{link_dir, LinkMode, LinkSettings};

/// The command-line switch that starts the background host.
pub(crate) const FLAG: &str = "--link-background";

/// What the background host needs from the app's settings (written by the
/// app whenever they change — the background copy can't read eframe's own
/// settings store).
#[derive(Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(default)]
pub(crate) struct BackgroundConfig {
    pub link: LinkSettings,
    pub chat_model: crate::llm::GemmaModel,
    pub image_detail: crate::llm::ImageDetail,
}

#[derive(Serialize, Deserialize)]
struct RunningState {
    pid: u32,
    port: u16,
    token: String,
}

fn config_path() -> PathBuf {
    link_dir().join("background-config.json")
}

fn state_path() -> PathBuf {
    link_dir().join("background.json")
}

fn log_path() -> PathBuf {
    link_dir().join("background.log")
}

/// Append a line to the background host's log (it has no window to show it).
fn log(line: &str) {
    let path = log_path();
    // Keep it small: start over past 1 MB.
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > 1024 * 1024) {
        let _ = std::fs::remove_file(&path);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(f, "{stamp}  {line}");
    }
}

pub(crate) fn save_config(cfg: &BackgroundConfig) {
    let _ = std::fs::create_dir_all(link_dir());
    if let Ok(json) = serde_json::to_string_pretty(cfg) {
        let _ = crate::secret::write_private(config_path(), json);
    }
}

/// Written by the Windows installer (packaging/windows-installer.iss) before
/// it replaces the program: a copy of the app closed for the update mustn't
/// start a background host, which would hold the program open.
fn installing_marker() -> PathBuf {
    link_dir().join("installing")
}

/// An update is being installed right now.
pub(crate) fn installing() -> bool {
    std::fs::metadata(installing_marker())
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < Duration::from_secs(30 * 60))
}

/// The app started (after an update, say): the marker has done its job.
pub(crate) fn clear_installing() {
    let _ = std::fs::remove_file(installing_marker());
}

/// Link has been used on this PC (its background config was written).
pub(crate) fn config_exists() -> bool {
    config_path().is_file()
}

fn load_config() -> Option<BackgroundConfig> {
    serde_json::from_str(&std::fs::read_to_string(config_path()).ok()?).ok()
}

/// The background host that's running, if any: its control port and token.
fn running() -> Option<RunningState> {
    let state: RunningState = serde_json::from_str(&std::fs::read_to_string(state_path()).ok()?).ok()?;
    // A stale file (the process died): nothing answers on its port.
    TcpStream::connect_timeout(&([127, 0, 0, 1], state.port).into(), Duration::from_millis(300)).ok()?;
    Some(state)
}

/// Is a background host running?
pub(crate) fn is_running() -> bool {
    running().is_some()
}

/// Tell a running background host to stop: it stops taking new devices at
/// once (freeing the port) and exits when its running jobs are done. True if
/// one was told.
pub(crate) fn stop_running() -> bool {
    let Some(state) = running() else { return false };
    let Ok(mut s) = TcpStream::connect_timeout(&([127, 0, 0, 1], state.port).into(), Duration::from_millis(500)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    if writeln!(s, "{} quit", state.token).is_err() {
        return false;
    }
    let mut reply = String::new();
    let _ = BufReader::new(s).read_line(&mut reply);
    reply.trim() == "ok"
}

/// The program to start for hosting: this one — or, for an AppImage, the
/// AppImage file itself (the running copy lives in a temporary mount).
fn program() -> Option<PathBuf> {
    std::env::var_os("APPIMAGE").map(PathBuf::from).or_else(|| std::env::current_exe().ok())
}

/// Start the background host (the app is closing with background hosting on).
pub(crate) fn spawn() -> Result<(), String> {
    let exe = program().ok_or("Couldn't find the program")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg(FLAG)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
        // Outlive the app even when a launcher started it inside a job that
        // ends with it; a job that doesn't allow leaving refuses the flag, so
        // then it's tried without.
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB);
        if cmd.spawn().is_ok() {
            return Ok(());
        }
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Its own session: closing the terminal the app was started from
        // doesn't take the host with it.
        cmd.process_group(0);
    }
    cmd.spawn().map(|_| ()).map_err(|e| format!("Couldn't start the background host: {e}"))
}

#[cfg(windows)]
const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
#[cfg(windows)]
const RUN_VALUE: &str = "Clarity TagFlow Link";

/// Marks that the login entry is in place (so turning it off — or never
/// having turned it on — costs nothing at each start).
fn autostart_marker() -> PathBuf {
    link_dir().join("autostart-on")
}

/// Make the start-at-login entry match `on`. On, it's written again each
/// run (the program may have moved since).
pub(crate) fn sync_autostart(on: bool) -> Result<(), String> {
    let marker = autostart_marker();
    if !on && !marker.exists() {
        return Ok(());
    }
    set_autostart(on)?;
    if on {
        let _ = std::fs::create_dir_all(link_dir());
        let _ = std::fs::write(&marker, b"");
    } else {
        let _ = std::fs::remove_file(&marker);
    }
    Ok(())
}

/// Start the background host at login (`on`), or stop doing that.
fn set_autostart(on: bool) -> Result<(), String> {
    let exe = program().ok_or("Couldn't find the program")?;
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut cmd = std::process::Command::new("reg");
        if on {
            cmd.args(["add", RUN_KEY, "/v", RUN_VALUE, "/t", "REG_SZ", "/f", "/d"])
                .arg(format!("\"{}\" {FLAG}", exe.display()));
        } else {
            cmd.args(["delete", RUN_KEY, "/v", RUN_VALUE, "/f"]);
        }
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        let out = cmd.output().map_err(|e| e.to_string())?;
        // Deleting a value that isn't there "fails" — that's fine.
        if on && !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
        }
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        let dir = dirs::home_dir().ok_or("No home folder")?.join("Library").join("LaunchAgents");
        let path = dir.join("com.claritytagflow.link.plist");
        if !on {
            let _ = std::fs::remove_file(&path);
            return Ok(());
        }
        let esc = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
        let plist = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
             <plist version=\"1.0\"><dict>\n\
             <key>Label</key><string>com.claritytagflow.link</string>\n\
             <key>ProgramArguments</key><array><string>{}</string><string>{FLAG}</string></array>\n\
             <key>RunAtLoad</key><true/>\n\
             </dict></plist>\n",
            esc(&exe.to_string_lossy())
        );
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(&path, plist).map_err(|e| e.to_string())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // XDG autostart: started by the desktop session at login.
        let dir = dirs::config_dir().ok_or("No config folder")?.join("autostart");
        let path = dir.join("clarity-tagflow-link.desktop");
        if !on {
            let _ = std::fs::remove_file(&path);
            return Ok(());
        }
        let quoted = format!(
            "\"{}\"",
            exe.to_string_lossy()
                .replace('\\', "\\\\\\\\")
                .replace('"', "\\\\\"")
                .replace('`', "\\\\`")
                .replace('$', "\\\\$")
                .replace('%', "%%")
        );
        let entry = format!(
            "[Desktop Entry]\nType=Application\nName=Clarity TagFlow Link\n\
             Comment=Lets your other PCs use this one's AI models (Clarity TagFlow → Settings → Link)\n\
             Exec={quoted} {FLAG}\nTerminal=false\nNoDisplay=true\nX-GNOME-Autostart-enabled=true\n"
        );
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(&path, entry).map_err(|e| e.to_string())
    }
}

/// Run as the background host. Returns the process exit code.
pub(crate) fn run() -> i32 {
    let Some(cfg) = load_config() else {
        return 0;
    };
    if cfg.link.mode != LinkMode::Host || !(cfg.link.background || cfg.link.autostart) || installing() {
        return 0;
    }
    // One background host at a time.
    if is_running() {
        return 0;
    }
    let Ok(control) = TcpListener::bind("127.0.0.1:0") else { return 1 };
    let token = super::crypto::to_hex(&{
        let mut b = [0u8; 16];
        let _ = getrandom::getrandom(&mut b);
        b
    });
    let state = RunningState { pid: std::process::id(), port: control.local_addr().map(|a| a.port()).unwrap_or(0), token };
    let _ = std::fs::create_dir_all(link_dir());
    if let Ok(json) = serde_json::to_string(&state) {
        let _ = crate::secret::write_private(state_path(), json);
    }
    let _ = control.set_nonblocking(true);
    log("background host started");

    let ctx = eframe::egui::Context::default();
    super::set_host_llm(cfg.chat_model, cfg.image_detail);
    let backend: Arc<dyn super::Backend> = Arc::new(super::AppBackend::new(ctx.clone()));
    let files = Arc::new(super::files::FileHub::new(Box::new(|| {})));
    files.set_quiet(true);
    files.set_received_dir(super::received_dir(&cfg.link));
    super::clear_temp();
    // (At login a virus scanner may hold the key file for a moment.)
    let mut tries = 0;
    let identity = loop {
        match super::crypto::Identity::load_or_create(&link_dir()) {
            Ok(id) => break id,
            Err(e) if tries < 15 => {
                tries += 1;
                log(&format!("no identity key yet: {e}"));
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(e) => {
                log(&format!("no identity key: {e}"));
                let _ = std::fs::remove_file(state_path());
                return 1;
            }
        }
    };
    let host = super::host::Host::start(
        identity,
        link_dir(),
        super::host_config(&cfg.link, true),
        backend,
        files,
    );

    let mut quitting: Option<Instant> = None;
    let mut last_busy = Instant::now();
    let mut last_check = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(400));
        // Control commands from the app.
        while let Ok((s, _)) = control.accept() {
            if handle_control(s, &state.token) {
                log("asked to stop");
                quitting.get_or_insert_with(Instant::now);
                // Stop taking devices and calls now: the port is free for
                // the app, and idle devices move over to it.
                host.shared.wind_down();
            }
        }
        if let super::host::Listening::Failed(why) = host.shared.listening.lock().unwrap_or_else(|e| e.into_inner()).clone() {
            log(&format!("couldn't listen: {why}"));
            break;
        }
        let busy = host.shared.busy.load(Ordering::SeqCst) > 0;
        if busy {
            last_busy = Instant::now();
        }
        if let Some(since) = quitting {
            // Devices that aren't waiting on anything here move over to the
            // app now (they reconnect to it); running jobs finish first (up
            // to 30 minutes for a long video).
            for p in host.shared.sessions.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                if !p.serving() {
                    p.close();
                }
            }
            if !busy || since.elapsed() > Duration::from_secs(30 * 60) {
                break;
            }
            continue;
        }
        // Give the GPU back while nobody uses it (checked now and then).
        if last_check.elapsed() < Duration::from_secs(20) {
            continue;
        }
        last_check = Instant::now();
        let idle = last_busy.elapsed();
        // (`None`: a question is running.)
        let chat_idle = crate::llm::remote_idle_for().is_some_and(|d| d > Duration::from_secs(10 * 60));
        if chat_idle && crate::llm::worker_loaded() {
            log("unloading the AI model (unused for 10 minutes)");
            crate::llm::release_shared_worker();
        }
        #[cfg(not(target_os = "macos"))]
        if idle > Duration::from_secs(15 * 60) {
            crate::generate::stop_server_if_ours();
        }
    }
    host.stop();
    #[cfg(not(target_os = "macos"))]
    crate::generate::stop_server_if_ours();
    let _ = std::fs::remove_file(state_path());
    log("background host stopped");
    0
}

/// Read one control command; true for "quit" (with the right token).
fn handle_control(s: TcpStream, token: &str) -> bool {
    let _ = s.set_nonblocking(false);
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    let mut line = String::new();
    let mut reader = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return false,
    });
    if reader.read_line(&mut line).is_err() {
        return false;
    }
    let mut parts = line.split_whitespace();
    let (Some(t), Some(cmd)) = (parts.next(), parts.next()) else { return false };
    if t != token {
        return false;
    }
    let mut s = s;
    let _ = writeln!(s, "ok");
    cmd == "quit"
}
