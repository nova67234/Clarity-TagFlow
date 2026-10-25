//! The top bar — an open-folder button, the live CPU / RAM / GPU stats, and
//! the backup / find-issues / settings action buttons. Split out of
//! `main.rs`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, CornerRadius, Stroke};

use crate::theme::*;
use crate::{card_frame, svg_button};

/// Per-metric accent colours for the stats (icon, value, lit LEDs).
const CPU_COLOR: Color32 = Color32::from_rgb(96, 165, 250); // blue
const RAM_COLOR: Color32 = Color32::from_rgb(192, 132, 252); // violet
const GPU_COLOR: Color32 = Color32::from_rgb(52, 211, 153); // green

// ---------------------------------------------------------------------------
// Live CPU / RAM / GPU sampling for the top-bar stats.
// ---------------------------------------------------------------------------
pub struct SystemStats {
    sys: sysinfo::System,
    last_sample: Instant,
    cpu_pct: f32,
    ram_used_gb: f64,
    ram_total_gb: f64,
    gpu_pct: f32,
    gpu_mem_used_gb: f64,
    gpu_mem_total_gb: f64,
    /// GPU core temperature (°C) — the GPU section's badge.
    gpu_temp_c: f32,
    /// CPU temperature (°C; 0 = unknown) — the CPU section's badge.
    cpu_temp_c: f32,
    /// Latest CPU temperature, written by its background sampler thread
    /// (started with the GPU one).
    cpu_temp_shared: Arc<Mutex<f32>>,
    /// CPU brand string (from sysinfo, captured once) and the GPU's product name
    /// (from nvidia-smi) — shown beside the metric names in the stat labels.
    cpu_name: String,
    gpu_name: String,
    /// True once the background sampler has seen a GPU (drives whether the GPU
    /// section is shown at all).
    gpu_available: bool,
    /// The GPU sampler thread is spawned lazily on first `update()`.
    gpu_started: bool,
    /// Latest GPU reading, written by the background sampler thread.
    gpu_shared: Arc<Mutex<GpuInfo>>,
}

impl Default for SystemStats {
    fn default() -> Self {
        Self {
            // OPTIMIZATION 3: Use new() instead of new_all() to prevent massive startup lag!
            sys: sysinfo::System::new(),

            // Force the first `update()` to take a fresh sample immediately.
            last_sample: Instant::now() - Duration::from_secs(60),
            cpu_pct: 0.0,
            ram_used_gb: 0.0,
            ram_total_gb: 0.0,
            gpu_pct: 0.0,
            gpu_mem_used_gb: 0.0,
            gpu_mem_total_gb: 0.0,
            gpu_temp_c: 0.0,
            cpu_temp_c: 0.0,
            cpu_temp_shared: Arc::new(Mutex::new(0.0)),
            cpu_name: String::new(),
            gpu_name: String::new(),
            gpu_available: false,
            gpu_started: false,
            gpu_shared: Arc::new(Mutex::new(GpuInfo::default())),
        }
    }
}

impl SystemStats {
    /// Re-sample no more than a few times a second; cheap to call every frame.
    pub fn update(&mut self) {
        if self.last_sample.elapsed() < Duration::from_millis(400) {
            return;
        }
        self.last_sample = Instant::now();

        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        // The CPU brand string never changes — grab it from the first refresh.
        if self.cpu_name.is_empty() {
            self.cpu_name = self
                .sys
                .cpus()
                .first()
                .map(|c| c.brand().trim().to_string())
                .unwrap_or_default();
        }

        self.cpu_pct = self.sys.global_cpu_usage();
        let used = self.sys.used_memory() as f64;
        let total = self.sys.total_memory().max(1) as f64;
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        self.ram_used_gb = used / GIB;
        self.ram_total_gb = total / GIB;

        // GPU and CPU temperature are sampled off-thread (nvidia-smi blocks for
        // tens of ms), started lazily so they only run while the stats are
        // actually being shown.
        if !self.gpu_started {
            self.gpu_started = true;
            spawn_gpu_sampler(Arc::clone(&self.gpu_shared));
            spawn_cpu_temp_sampler(Arc::clone(&self.cpu_temp_shared));
        }
        self.cpu_temp_c = *self.cpu_temp_shared.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(g) = self.gpu_shared.lock() {
            self.gpu_available = g.available;
            self.gpu_pct = g.util * 100.0;
            self.gpu_mem_used_gb = g.mem_used_gb;
            self.gpu_mem_total_gb = g.mem_total_gb;
            self.gpu_temp_c = g.temp_c;
            if self.gpu_name != g.name {
                self.gpu_name = g.name.clone();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU sampling (NVIDIA via nvidia-smi). Runs on a background thread so the
// process spawn never stalls the UI. AMD/Intel GPUs aren't covered — the graph
// simply doesn't appear when no NVIDIA GPU is present.
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct GpuInfo {
    available: bool,
    /// Product name as nvidia-smi reports it (e.g. "NVIDIA GeForce RTX 4080").
    name: String,
    util: f32, // 0.0..=1.0
    mem_used_gb: f64,
    mem_total_gb: f64,
    /// Core temperature in °C.
    temp_c: f32,
}

/// One `nvidia-smi` reading. `None` means the tool is missing / errored (no
/// usable NVIDIA GPU); `Some` carries the current utilisation and VRAM.
fn query_gpu() -> Option<GpuInfo> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]);
    // Suppress the console window (Windows). The helper lives in pixal3d, which
    // is compiled out on macOS — where it's a no-op anyway (no NVIDIA support).
    #[cfg(not(target_os = "macos"))]
    crate::pixal3d::hide_window(&mut cmd);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // First GPU only (multi-GPU boxes report one line each).
    let line = text.lines().next()?;
    let mut parts = line.split(',').map(|s| s.trim());
    let name = parts.next()?.to_string();
    let util = parts.next()?.parse::<f32>().ok()?;
    let used_mib = parts.next()?.parse::<f64>().ok()?;
    let total_mib = parts.next()?.parse::<f64>().ok()?;
    // Temperature is best-effort: some GPUs report "N/A", which shouldn't drop
    // the whole reading — show 0 (hidden) instead.
    let temp_c = parts.next().and_then(|t| t.parse::<f32>().ok()).unwrap_or(0.0);
    Some(GpuInfo {
        available: true,
        name,
        util: (util / 100.0).clamp(0.0, 1.0),
        mem_used_gb: used_mib / 1024.0,
        mem_total_gb: total_mib / 1024.0,
        temp_c,
    })
}

/// Poll the GPU on a background thread, publishing the latest reading into
/// `shared`. Stops itself if the very first query fails (no NVIDIA GPU); once it
/// has seen a GPU it keeps retrying through transient failures.
fn spawn_gpu_sampler(shared: Arc<Mutex<GpuInfo>>) {
    std::thread::spawn(move || {
        let mut seen_ok = false;
        loop {
            match query_gpu() {
                Some(info) => {
                    seen_ok = true;
                    if let Ok(mut g) = shared.lock() {
                        *g = info;
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
                None if !seen_ok => {
                    // No NVIDIA GPU / tool unavailable — leave it hidden and stop.
                    if let Ok(mut g) = shared.lock() {
                        g.available = false;
                    }
                    return;
                }
                None => std::thread::sleep(Duration::from_secs(1)),
            }
        }
    });
}

// ---------------------------------------------------------------------------
// CPU temperature. Windows: the ACPI thermal zones through the performance
// counters (readable without admin, unlike WMI's MSAcpi_ThermalZoneTemperature);
// the hottest zone stands in for the CPU. Elsewhere: sysinfo's sensors.
// ---------------------------------------------------------------------------

/// Poll the CPU temperature every 2 s into `shared` (°C). Stops if the first
/// reading finds no sensor — the badge then stays hidden.
fn spawn_cpu_temp_sampler(shared: Arc<Mutex<f32>>) {
    std::thread::spawn(move || {
        let Some(mut sensor) = CpuTemp::open() else {
            return;
        };
        let mut seen_ok = false;
        loop {
            match sensor.read() {
                Some(t) => {
                    seen_ok = true;
                    *shared.lock().unwrap_or_else(|e| e.into_inner()) = t;
                }
                None if !seen_ok => return,
                None => {}
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    });
}

#[cfg(windows)]
struct CpuTemp {
    query: isize,
    counter: isize,
}

#[cfg(windows)]
mod pdh {
    pub const FMT_DOUBLE: u32 = 0x0000_0200;
    pub const MORE_DATA: u32 = 0x8000_07D2;

    #[repr(C)]
    pub struct FmtValue {
        pub status: u32,
        pub value: f64,
    }

    #[repr(C)]
    pub struct FmtItem {
        pub name: *mut u16,
        pub value: FmtValue,
    }

    #[link(name = "pdh")]
    unsafe extern "system" {
        pub fn PdhOpenQueryW(source: *const u16, user: usize, query: *mut isize) -> u32;
        pub fn PdhAddEnglishCounterW(query: isize, path: *const u16, user: usize, counter: *mut isize) -> u32;
        pub fn PdhCollectQueryData(query: isize) -> u32;
        pub fn PdhGetFormattedCounterArrayW(
            counter: isize,
            format: u32,
            size: *mut u32,
            count: *mut u32,
            items: *mut FmtItem,
        ) -> u32;
        pub fn PdhCloseQuery(query: isize) -> u32;
    }
}

#[cfg(windows)]
impl CpuTemp {
    fn open() -> Option<CpuTemp> {
        let path: Vec<u16> = "\\Thermal Zone Information(*)\\High Precision Temperature\0".encode_utf16().collect();
        let (mut query, mut counter) = (0isize, 0isize);
        // SAFETY: plain PDH calls with valid out-pointers; the query is closed
        // on the failure path here and in Drop otherwise.
        unsafe {
            if pdh::PdhOpenQueryW(std::ptr::null(), 0, &mut query) != 0 {
                return None;
            }
            if pdh::PdhAddEnglishCounterW(query, path.as_ptr(), 0, &mut counter) != 0 {
                pdh::PdhCloseQuery(query);
                return None;
            }
        }
        Some(CpuTemp { query, counter })
    }

    /// The hottest thermal zone, in °C (the counter reads tenths of a kelvin).
    fn read(&mut self) -> Option<f32> {
        // SAFETY: the buffer is sized by PDH's own first call, and u64-backed
        // so the items (which hold f64s) are aligned.
        unsafe {
            if pdh::PdhCollectQueryData(self.query) != 0 {
                return None;
            }
            let (mut size, mut count) = (0u32, 0u32);
            let r = pdh::PdhGetFormattedCounterArrayW(
                self.counter,
                pdh::FMT_DOUBLE,
                &mut size,
                &mut count,
                std::ptr::null_mut(),
            );
            if r != pdh::MORE_DATA || size == 0 {
                return None;
            }
            let mut buf = vec![0u64; (size as usize).div_ceil(8)];
            let items = buf.as_mut_ptr() as *mut pdh::FmtItem;
            if pdh::PdhGetFormattedCounterArrayW(self.counter, pdh::FMT_DOUBLE, &mut size, &mut count, items) != 0 {
                return None;
            }
            let items = std::slice::from_raw_parts(items, count as usize);
            let tenths_k =
                items.iter().filter(|i| i.value.status == 0).map(|i| i.value.value).fold(f64::NAN, f64::max);
            let c = (tenths_k / 10.0 - 273.15) as f32;
            (c.is_finite() && (1.0..150.0).contains(&c)).then_some(c)
        }
    }
}

#[cfg(windows)]
impl Drop for CpuTemp {
    fn drop(&mut self) {
        // SAFETY: the query was opened in `open`.
        unsafe {
            pdh::PdhCloseQuery(self.query);
        }
    }
}

#[cfg(not(windows))]
struct CpuTemp {
    components: sysinfo::Components,
}

#[cfg(not(windows))]
impl CpuTemp {
    fn open() -> Option<CpuTemp> {
        Some(CpuTemp { components: sysinfo::Components::new_with_refreshed_list() })
    }

    /// The hottest CPU-looking sensor ("Package id 0", "Tctl", "coretemp"…).
    fn read(&mut self) -> Option<f32> {
        self.components.refresh(false);
        self.components
            .iter()
            .filter(|c| {
                let l = c.label().to_ascii_lowercase();
                ["cpu", "package", "tctl", "tdie", "coretemp", "k10temp"].iter().any(|k| l.contains(k))
            })
            .filter_map(|c| c.temperature())
            .filter(|t| t.is_finite() && *t > 0.0)
            .reduce(f32::max)
    }
}

// ---------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------

/// What the top bar is asking the app to do this frame.
pub enum TopBarAction {
    None,
    /// The user clicked the open-folder button. Carries the button's
    /// bottom-left so the folder popup can drop down under it.
    OpenFolder(egui::Pos2),
    /// The user clicked the settings gear.
    OpenSettings,
    /// The user clicked "Create Backup".
    CreateBackup,
    /// The user clicked "Find Issues". Carries the button's bottom-right so the
    /// Deep Scan window can drop down right-aligned under it (extending left).
    FindIssues(egui::Pos2),
}

/// Render the top bar. Returns any action the app should perform. `show_stats`
/// toggles the CPU / RAM / GPU stats; `update_badge` paints a red dot on the
/// settings gear when an app/ComfyUI update is available; `ftp_mode` swaps the
/// folder button for the FTP remote-browser button (Settings → FTP/FTPS).
pub fn show(ui: &mut egui::Ui, stats: &SystemStats, show_stats: bool, update_badge: bool, ftp_mode: bool) -> TopBarAction {
    let mut action = TopBarAction::None;

    egui::Panel::top("topbar")
        .resizable(false) // This locks the height and removes the drag handle
        .show_separator_line(false)
        .frame(egui::Frame::new().fill(BG()).inner_margin(egui::Margin::same(10)))
        .show_inside(ui, |ui| {
            card_frame(22).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(56.0);

                    // LEFT: folder icon -> open a folder of images. In FTP mode
                    // the icon becomes a globe and opens the remote browser
                    // instead (main.rs routes the action accordingly).
                    let (folder_svg, tip) = if ftp_mode {
                        (egui::include_image!("../icons/ftp_browser.svg"), "Browse FTP server")
                    } else {
                        (egui::include_image!("../icons/folder.svg"), "Open folder")
                    };

                    let fb = svg_button(ui, folder_svg, tip, 37.0, icon_tint(Color32::GRAY));
                    if fb.clicked() {
                        action = TopBarAction::OpenFolder(fb.rect.left_bottom());
                    }

                    // RIGHT (laid out right-to-left): settings, the two
                    // action buttons, and then the stats fill the gap (from the left).
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let settings_svg = egui::include_image!("../icons/top_settings.svg");
                            let gear = svg_button(ui, settings_svg, "Settings", 37.0, MUTED());
                            if gear.clicked() {
                                action = TopBarAction::OpenSettings;
                            }
                            // Red "update available" dot on the gear's top-right corner.
                            if update_badge {
                                let center = gear.rect.center();
                                let dot = egui::pos2(center.x + 14.0, center.y - 14.0);
                                let p = ui.painter();
                                p.circle_filled(dot, 5.5, BG());
                                p.circle_filled(dot, 4.0, Color32::from_rgb(230, 70, 70));
                            }

                            ui.add_space(6.0);
                            let backup = bar_button_icon(
                                ui,
                                egui::include_image!("../icons/backup.svg"),
                                "Create Backup",
                                150.0,
                            );
                            if backup.clicked() {
                                action = TopBarAction::CreateBackup;
                            }
                            ui.add_space(6.0);
                            let fi = bar_button_icon(
                                ui,
                                egui::include_image!("../icons/frame_inspect.svg"),
                                "Find Issues",
                                130.0,
                            );
                            if fi.clicked() {
                                action = TopBarAction::FindIssues(fi.rect.right_bottom());
                            }
                            ui.add_space(14.0);

                            // LEFT, after the folder button: the CPU / RAM / GPU
                            // stats in the leftover space, straight on the bar's
                            // panel. Hidden when the user disables stats in Settings;
                            // the GPU section only appears when an NVIDIA GPU is found.
                            if show_stats {
                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                    hud_stats(ui, &hud_sections(stats));
                                });
                            }
                        },
                    );
                });
            });
        });

    action
}

/// A fixed-width top-bar button styled exactly like the right-details panel
/// buttons: theme fill and hover (no accent, no drop shadow) and the theme text
/// color — just rendered at a fixed width.
/// A top-bar action button with a leading SVG icon (tinted to the button's label
/// colour so it matches in every theme).
fn bar_button_icon(ui: &mut egui::Ui, icon: egui::ImageSource<'_>, label: &str, width: f32) -> egui::Response {
    let tint = ui.visuals().widgets.inactive.fg_stroke.color;
    let img = egui::Image::new(icon).fit_to_exact_size(egui::vec2(16.0, 16.0)).tint(tint);
    ui.add_sized(
        egui::vec2(width, 34.0),
        egui::Button::image_and_text(img, egui::RichText::new(label).size(15.0))
            .corner_radius(CornerRadius::same(12)),
    )
}

/// A small rounded badge after a section's name.
struct HudBadge {
    text: String,
    /// Tinted in this colour (the RAM percentage, the VRAM tier) instead of
    /// neutral grey.
    tint: Option<Color32>,
    /// Neutral, but with its text in this colour (a warm / hot temperature).
    warn: Option<Color32>,
}

/// One stats section: a round icon, the name with its badges (a
/// temperature, the RAM percentage, the VRAM size), and the value beside a
/// five-LED strip.
struct HudSection {
    name: &'static str,
    color: Color32,
    /// Left out when unknown (e.g. no CPU temperature sensor).
    badges: Vec<HudBadge>,
    value: String,
    /// Muted unit after the value ("/ 31 GB").
    unit: Option<String>,
    /// The widest value this section shows (e.g. "100%"), so the LEDs and the
    /// sections don't shift as the digits change.
    widest: String,
    /// How full the LED strip is, 0.0..=1.0.
    frac: f32,
    tooltip: String,
}

/// The stats sections for this frame's readings.
fn hud_sections(stats: &SystemStats) -> Vec<HudSection> {
    // Neutral while cool; orange / red once warm or hot (see `temp_color`).
    let temp_badge = |t: f32| {
        (t > 0.0).then(|| HudBadge { text: format!("{t:.0}°C"), tint: None, warn: (t >= 65.0).then(|| temp_color(t)) })
    };
    let ram_frac = (stats.ram_used_gb / stats.ram_total_gb.max(0.001)) as f32;
    let mut out = vec![
        HudSection {
            name: "CPU",
            color: CPU_COLOR,
            badges: temp_badge(stats.cpu_temp_c).into_iter().collect(),
            value: format!("{:.0}%", stats.cpu_pct),
            unit: None,
            widest: "100%".into(),
            frac: stats.cpu_pct / 100.0,
            tooltip: if stats.cpu_temp_c > 0.0 {
                format!("{}\nTemperature: the hottest of this PC's thermal sensors", stats.cpu_name)
            } else {
                stats.cpu_name.clone()
            },
        },
        HudSection {
            name: "RAM",
            color: RAM_COLOR,
            badges: vec![HudBadge { text: format!("{:.0}%", ram_frac * 100.0), tint: Some(RAM_COLOR), warn: None }],
            value: format!("{:.1}", stats.ram_used_gb),
            unit: Some(format!("/ {:.0} GB", stats.ram_total_gb)),
            widest: format!("{:.1}", stats.ram_total_gb),
            frac: ram_frac,
            tooltip: format!("{:.1} / {:.1} GB used", stats.ram_used_gb, stats.ram_total_gb),
        },
    ];
    if stats.gpu_available {
        out.push(HudSection {
            name: "GPU",
            color: GPU_COLOR,
            // Temperature, then the card's VRAM in its tier colour (how big a
            // model fits, at a glance).
            badges: temp_badge(stats.gpu_temp_c)
                .into_iter()
                .chain((stats.gpu_mem_total_gb > 0.0).then(|| HudBadge {
                    text: format!("{:.0} GB", stats.gpu_mem_total_gb),
                    tint: Some(vram_color(stats.gpu_mem_total_gb)),
                    warn: None,
                }))
                .collect(),
            value: format!("{:.0}%", stats.gpu_pct),
            unit: None,
            widest: "100%".into(),
            frac: stats.gpu_pct / 100.0,
            tooltip: format!(
                "{}\n{:.0}% · VRAM {:.1} / {:.1} GB",
                stats.gpu_name, stats.gpu_pct, stats.gpu_mem_used_gb, stats.gpu_mem_total_gb
            ),
        });
    }
    out
}

/// Colour-code the VRAM badge by how much VRAM the card has: 16 GB or more is
/// green, 8–16 GB orange, under 8 GB red. (Thresholds use a small tolerance —
/// a "16 GB" card can report 15.99 GB.)
fn vram_color(total_gb: f64) -> Color32 {
    if total_gb >= 15.5 {
        Color32::from_rgb(46, 160, 67) // green
    } else if total_gb >= 7.5 {
        Color32::from_rgb(235, 150, 45) // orange
    } else {
        Color32::from_rgb(210, 70, 70) // red
    }
}

/// Colour-code the GPU temperature: cool (under 65 °C) green, warm (65–79 °C)
/// orange, hot (80 °C and up — where laptop GPUs start throttling) red.
fn temp_color(temp_c: f32) -> Color32 {
    if temp_c < 65.0 {
        Color32::from_rgb(46, 160, 67) // green
    } else if temp_c < 80.0 {
        Color32::from_rgb(235, 150, 45) // orange
    } else {
        Color32::from_rgb(210, 70, 70) // red
    }
}

/// A tinted badge's text: its colour, lifted for contrast against its own
/// faint fill on the dark themes (as the design's #d8b4fe on violet).
fn badge_text_tint(c: Color32) -> Color32 {
    if crate::theme::is_light() { c } else { c.lerp_to_gamma(Color32::WHITE, 0.35) }
}

/// The stats' neutral colours: the dark HUD look, or a light counterpart on
/// the light themes.
struct HudInk {
    label: Color32,
    badge_fill: Color32,
    badge_text: Color32,
    unit: Color32,
    led_off: Color32,
    divider: Color32,
}

impl HudInk {
    fn new() -> HudInk {
        let a = Color32::from_rgba_unmultiplied;
        if crate::theme::is_light() {
            HudInk {
                label: Color32::from_rgb(100, 116, 139),
                badge_fill: a(0, 0, 0, 14),
                badge_text: Color32::from_rgb(51, 65, 85),
                unit: Color32::from_rgb(100, 116, 139),
                led_off: a(0, 0, 0, 26),
                divider: a(0, 0, 0, 22),
            }
        } else {
            HudInk {
                label: Color32::from_rgb(148, 163, 184),
                badge_fill: a(255, 255, 255, 15),
                badge_text: Color32::from_rgb(203, 213, 225),
                unit: Color32::from_rgb(100, 116, 139),
                led_off: a(255, 255, 255, 26),
                divider: a(255, 255, 255, 20),
            }
        }
    }
}

/// The stats' measurements (CSS px from the design).
const HUD_H: f32 = 56.0;
/// Space between the folder button and the stats (on top of the row's item
/// spacing): close to the button, but clear of it.
const HUD_LEFT_GAP: f32 = 20.0;
const HUD_GAP: f32 = 24.0;
const HUD_ICON: f32 = 32.0;
const HUD_ICON_GAP: f32 = 12.0;
const HUD_LEDS: usize = 5;
const HUD_LED: egui::Vec2 = egui::vec2(5.0, 10.0);
const HUD_LED_GAP: f32 = 3.0;

/// One section's text, laid out.
struct HudText {
    label: Arc<egui::Galley>,
    badges: Vec<Arc<egui::Galley>>,
    value: Arc<egui::Galley>,
    unit: Option<Arc<egui::Galley>>,
    /// Value + unit, reserved at the widest value.
    value_w: f32,
    info_w: f32,
}

/// The HUD telemetry, drawn straight on the top bar's panel: one section per
/// metric (round icon, name + badge over the value and a five-LED strip),
/// split by hairline dividers. Drops the icons when the space is short, and
/// hides itself when even that doesn't fit (rather than overlapping the
/// action buttons).
fn hud_stats(ui: &mut egui::Ui, sections: &[HudSection]) {
    if sections.is_empty() {
        return;
    }
    let ink = HudInk::new();
    let mono = egui::FontId::monospace;
    let prop = egui::FontId::proportional;
    let led_w = HUD_LEDS as f32 * HUD_LED.x + (HUD_LEDS - 1) as f32 * HUD_LED_GAP;

    let texts: Vec<HudText> = sections
        .iter()
        .map(|s| {
            let p = ui.painter();
            let label = p.layout_no_wrap(s.name.to_string(), prop(11.0), ink.label);
            let badges: Vec<_> = s
                .badges
                .iter()
                .map(|b| {
                    let color = b.warn.or(b.tint.map(badge_text_tint)).unwrap_or(ink.badge_text);
                    p.layout_no_wrap(b.text.clone(), mono(10.0), color)
                })
                .collect();
            let value = p.layout_no_wrap(s.value.clone(), mono(15.0), s.color);
            let unit = s.unit.as_ref().map(|u| p.layout_no_wrap(u.clone(), prop(12.0), ink.unit));
            let unit_w = unit.as_ref().map_or(0.0, |u| 4.0 + u.size().x);
            let widest = p.layout_no_wrap(s.widest.clone(), mono(15.0), s.color).size().x;
            let value_w = value.size().x.max(widest) + unit_w;
            let top_w = label.size().x + badges.iter().map(|b| 6.0 + b.size().x + 12.0).sum::<f32>();
            let info_w = top_w.max(value_w + 10.0 + led_w);
            HudText { label, badges, value, unit, value_w, info_w }
        })
        .collect();

    let n = sections.len() as f32;
    let dividers = (n - 1.0) * (HUD_GAP * 2.0 + 1.0);
    let infos: f32 = texts.iter().map(|t| t.info_w).sum();
    let width = |icons: bool| {
        infos + dividers + if icons { n * (HUD_ICON + HUD_ICON_GAP) } else { 0.0 }
    };
    let avail = ui.available_width() - HUD_LEFT_GAP;
    let icons = width(true) <= avail;
    if !icons && width(false) > avail {
        return;
    }
    let w = width(icons);
    // Left-aligned, next to the folder button.
    ui.add_space(HUD_LEFT_GAP);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, HUD_H), egui::Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }

    let painter = ui.painter();

    let cy = rect.center().y;
    let mut x = rect.left();
    for (i, (s, t)) in sections.iter().zip(&texts).enumerate() {
        if i > 0 {
            x += HUD_GAP;
            painter.rect_filled(
                egui::Rect::from_min_size(egui::pos2(x, cy - 14.0), egui::vec2(1.0, 28.0)),
                0,
                ink.divider,
            );
            x += 1.0 + HUD_GAP;
        }
        let start = x;

        if icons {
            let c = egui::pos2(x + HUD_ICON / 2.0, cy);
            painter.circle(c, HUD_ICON / 2.0 - 0.5, s.color.gamma_multiply(0.15), Stroke::new(1.0, s.color.gamma_multiply(0.3)));
            painter.text(c, egui::Align2::CENTER_CENTER, s.name, prop(9.0), s.color);
            x += HUD_ICON + HUD_ICON_GAP;
        }

        // Name + badges, then value + LEDs 2px below.
        let badge_h = t.badges.iter().map(|b| b.size().y + 2.0).fold(0.0, f32::max);
        let top_h = t.label.size().y.max(badge_h);
        let bottom_h = t.value.size().y.max(HUD_LED.y);
        let y0 = cy - (top_h + 2.0 + bottom_h) / 2.0;
        painter.galley(egui::pos2(x, y0 + (top_h - t.label.size().y) / 2.0), t.label.clone(), ink.label);
        let mut bx = x + t.label.size().x;
        for (b, g) in s.badges.iter().zip(&t.badges) {
            let br = egui::Rect::from_min_size(
                egui::pos2(bx + 6.0, y0 + (top_h - badge_h) / 2.0),
                egui::vec2(g.size().x + 12.0, badge_h),
            );
            let fill = b.tint.map_or(ink.badge_fill, |c| c.gamma_multiply(0.15));
            painter.rect_filled(br, CornerRadius::same((badge_h / 2.0) as u8), fill);
            painter.galley(egui::pos2(br.left() + 6.0, br.top() + 1.0), g.clone(), ink.badge_text);
            bx = br.right();
        }

        let y1 = y0 + top_h + 2.0;
        let vy = y1 + (bottom_h - t.value.size().y) / 2.0;
        painter.galley(egui::pos2(x, vy), t.value.clone(), s.color);
        if let Some(u) = &t.unit {
            // Bottoms aligned with the bigger value text.
            let ux = x + t.value.size().x + 4.0;
            painter.galley(egui::pos2(ux, vy + t.value.size().y - u.size().y - 1.0), u.clone(), ink.unit);
        }

        let lit = ((s.frac.clamp(0.0, 1.0) * HUD_LEDS as f32).ceil() as usize).min(HUD_LEDS);
        let mut lx = x + t.value_w + 10.0;
        let ly = y1 + (bottom_h - HUD_LED.y) / 2.0;
        for k in 0..HUD_LEDS {
            let led = egui::Rect::from_min_size(egui::pos2(lx, ly), HUD_LED);
            if k < lit {
                painter.add(
                    egui::epaint::Shadow { offset: [0, 0], blur: 6, spread: 0, color: s.color.gamma_multiply(0.7) }
                        .as_shape(led, CornerRadius::same(2)),
                );
                painter.rect_filled(led, CornerRadius::same(2), s.color);
            } else {
                painter.rect_filled(led, CornerRadius::same(2), ink.led_off);
            }
            lx += HUD_LED.x + HUD_LED_GAP;
        }
        x += t.info_w;

        let area = egui::Rect::from_x_y_ranges(start..=x, rect.y_range());
        ui.interact(area, ui.id().with(("hud", i)), egui::Sense::hover()).on_hover_text(&s.tooltip);
    }
}
// ---------------------------------------------------------------------------
// Folder popup — dropped down from the folder button.
// ---------------------------------------------------------------------------

/// State for the folder button's popup (owned by `ViewerApp`).
#[derive(Default)]
pub struct FolderPopup {
    pub open: bool,
    /// The folder button's bottom-left, captured on click — the popup drops
    /// down from there.
    pub anchor: Option<egui::Pos2>,
    /// True on the frame the popup was opened, so the opening click on the
    /// folder button isn't counted as a click outside (which would close it
    /// immediately).
    pub just_opened: bool,
}

/// What the folder popup asked the app to do this frame.
pub enum FolderMenuAction {
    None,
    /// Load this folder as the new input (browser contents).
    LoadInput(std::path::PathBuf),
    /// Open this zip archive as the input (via the password prompt; viewed
    /// in memory, never extracted — see `src/archive.rs`).
    OpenZip(std::path::PathBuf),
    /// Unload the current folder, emptying the browser.
    ClearInput,
}

/// The folder button's popup — a compact menu: a status line with the current
/// input folder, then Select Input / Select Output / Clear Input / Clear
/// Output rows. The Output folder is where the Move button (Details &
/// Actions) sends images; empty asks with a dialog on every move.
pub fn folder_popup(
    ctx: &egui::Context,
    state: &mut FolderPopup,
    current_folder: Option<&std::path::Path>,
    output_folder: &mut String,
) -> FolderMenuAction {
    if !state.open {
        return FolderMenuAction::None;
    }

    let mut action = FolderMenuAction::None;
    let mut close = ctx.input(|i| i.key_pressed(egui::Key::Escape));

    // Drop down from the folder button, left-aligned under it (mirrors the
    // Find Issues placement on the bar's other side). Clamped on-screen.
    let screen = ctx.content_rect();
    let win_w = 340.0_f32.min(screen.width() - 40.0);
    let anchor = state
        .anchor
        .unwrap_or_else(|| egui::pos2(screen.left() + 10.0, screen.top() + 80.0));
    let x = anchor.x.min(screen.right() - win_w - 10.0).max(screen.left() + 10.0);
    let y = anchor.y.min(screen.bottom() - 280.0).max(screen.top() + 10.0);

    use crate::PopupPlacement;
    let win = egui::Window::new("Open Folder")
        .id(egui::Id::new("folder_popup"))
        .title_bar(false) // custom header inside (matches Find Issues)
        .collapsible(false)
        .resizable(false)
        .placed_at([x, y])
        .frame(popup_frame())
        .show(ctx, |ui| {
            // Only the top strip drags the popup — not stray drags on the body.
            crate::popup_drag_strip(ui, 30.0);
            // Width pinned; the height hugs the content.
            ui.set_width(win_w - 36.0);

            // Title row: folder icon + "Open Folder" (no close button — click
            // anywhere outside the popup, or press Escape, to dismiss).
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/folder.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(icon_tint(TEXT())),
                );
                ui.heading(egui::RichText::new("Open Folder").color(TEXT()).strong().size(17.0));
            });
            ui.add_space(10.0);

            // Menu rows — plain text like a context menu, with the selected
            // folder shown right-aligned on its row (full path on hover).
            let input_path = current_folder.map(|p| p.display().to_string());
            let mut sel_in = menu_row_value(
                ui,
                "Select Input",
                input_path.as_deref().unwrap_or("No folder selected"),
                true,
            );
            if let Some(p) = &input_path {
                sel_in = sel_in.on_hover_text(p);
            }
            if sel_in.clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    action = FolderMenuAction::LoadInput(dir);
                }
                close = true;
            }
            let mut sel_out_tip = String::from("Where the Move button (Details & Actions) sends images");
            if !output_folder.trim().is_empty() {
                sel_out_tip = format!("{output_folder}\n{sel_out_tip}");
            }
            let sel_out = menu_row_value(
                ui,
                "Select Output",
                if output_folder.trim().is_empty() { "No folder selected" } else { output_folder },
                true,
            );
            if sel_out.on_hover_text(sel_out_tip).clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    *output_folder = dir.display().to_string();
                }
                close = true;
            }
            if menu_row(ui, "Open .Zip", true)
                .on_hover_text(
                    "Browse a zip archive — password-protected too — straight from \
                     the file, without extracting anything",
                )
                .clicked()
            {
                if let Some(f) = rfd::FileDialog::new()
                    .add_filter("Zip archive", &["zip"])
                    .pick_file()
                {
                    action = FolderMenuAction::OpenZip(f);
                }
                close = true;
            }

            ui.separator();

            if menu_row(ui, "Clear Input", current_folder.is_some()).clicked() {
                action = FolderMenuAction::ClearInput;
                close = true;
            }
            if menu_row(ui, "Clear Output", !output_folder.trim().is_empty()).clicked() {
                output_folder.clear();
                close = true;
            }
        });

    // Click anywhere outside the popup to dismiss it, like a native menu —
    // except on the frame it opened, when the folder-button click that spawned
    // it would register as "outside" and close it straight away.
    if state.just_opened {
        state.just_opened = false;
    } else if let Some(win) = &win
        && win.response.clicked_elsewhere()
    {
        close = true;
    }

    if close {
        state.open = false;
    }
    action
}

/// A folder-menu row with a right-aligned readout of its current selection.
/// The value is muted and elided from the left, keeping the tail (the folder
/// name) visible.
fn menu_row_value(ui: &mut egui::Ui, label: &str, value: &str, enabled: bool) -> egui::Response {
    let resp = menu_row(ui, label, enabled);
    let rect = resp.rect;
    if ui.is_rect_visible(rect) {
        let font = egui::FontId::proportional(12.0);
        let label_w = ui
            .painter()
            .layout_no_wrap(label.to_string(), egui::FontId::proportional(14.0), TEXT())
            .size()
            .x;
        let max_w = rect.width() - label_w - 30.0;
        let chars: Vec<char> = value.chars().collect();
        let mut galley = ui.painter().layout_no_wrap(value.to_string(), font.clone(), MUTED());
        let mut skip = 0;
        while galley.size().x > max_w && skip < chars.len() {
            // Drop characters from the front until the tail fits.
            skip += (chars.len() - skip).div_ceil(8).max(1);
            let shown: String = std::iter::once('…').chain(chars[skip..].iter().copied()).collect();
            galley = ui.painter().layout_no_wrap(shown, font.clone(), MUTED());
        }
        if max_w > 20.0 {
            ui.painter().galley(
                egui::pos2(rect.right() - 10.0 - galley.size().x, rect.center().y - galley.size().y / 2.0),
                galley,
                MUTED(),
            );
        }
    }
    resp
}

/// One left-aligned text row of the folder menu; highlights on hover, muted
/// when `enabled` is false.
fn menu_row(ui: &mut egui::Ui, label: &str, enabled: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 34.0),
        if enabled { egui::Sense::click() } else { egui::Sense::hover() },
    );
    if ui.is_rect_visible(rect) {
        if enabled && resp.hovered() {
            ui.painter().rect_filled(
                rect,
                CornerRadius::same(10),
                ui.visuals().widgets.hovered.weak_bg_fill,
            );
        }
        ui.painter().text(
            egui::pos2(rect.left() + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(14.0),
            if enabled { TEXT() } else { MUTED() },
        );
    }
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// A themed frame for the folder popup — same card styling as the other
/// popups, but with the softer rounded-22 corners of the gallery detail view.
fn popup_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(FLOAT())
        .corner_radius(CornerRadius::same(22))
        .inner_margin(egui::Margin::same(18))
        .stroke(Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 6],
            blur: 20,
            spread: 0,
            color: Color32::from_black_alpha(150),
        })
}
