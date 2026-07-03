//! The top bar — an open-folder button, live CPU/RAM graphs, and the
//! backup / find-issues / settings action buttons. Split out of `main.rs`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, CornerRadius, Stroke};

use crate::theme::*;
use crate::{card_frame, svg_button};

const GRAPH_POINTS: usize = 120;

/// Per-metric accent colours for the live graphs (line + gradient fill + value).
const CPU_COLOR: Color32 = Color32::from_rgb(83, 156, 255); // blue
const RAM_COLOR: Color32 = Color32::from_rgb(180, 132, 255); // violet
const GPU_COLOR: Color32 = Color32::from_rgb(60, 210, 140); // green

// ---------------------------------------------------------------------------
// Live CPU / RAM sampling for the center graphs.
// ---------------------------------------------------------------------------
pub struct SystemStats {
    sys: sysinfo::System,
    last_sample: Instant,
    cpu: VecDeque<f32>, // each value 0.0..=1.0
    ram: VecDeque<f32>, // each value 0.0..=1.0
    gpu: VecDeque<f32>, // each value 0.0..=1.0 (only filled when a GPU is found)
    cpu_pct: f32,
    ram_used_gb: f64,
    ram_total_gb: f64,
    gpu_pct: f32,
    gpu_mem_used_gb: f64,
    gpu_mem_total_gb: f64,
    /// GPU core temperature (°C) — shown under the GPU name, colour-coded.
    gpu_temp_c: f32,
    /// CPU brand string (from sysinfo, captured once) and the GPU's product name
    /// (from nvidia-smi) — shown beside the metric names in the stat labels.
    cpu_name: String,
    gpu_name: String,
    /// True once the background sampler has seen a GPU (drives whether the GPU
    /// graph is shown at all).
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
            cpu: VecDeque::with_capacity(GRAPH_POINTS),
            ram: VecDeque::with_capacity(GRAPH_POINTS),
            gpu: VecDeque::with_capacity(GRAPH_POINTS),
            cpu_pct: 0.0,
            ram_used_gb: 0.0,
            ram_total_gb: 0.0,
            gpu_pct: 0.0,
            gpu_mem_used_gb: 0.0,
            gpu_mem_total_gb: 0.0,
            gpu_temp_c: 0.0,
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

        push_capped(&mut self.cpu, (self.cpu_pct / 100.0).clamp(0.0, 1.0));
        push_capped(&mut self.ram, (used / total).clamp(0.0, 1.0) as f32);

        // GPU is sampled off-thread (nvidia-smi blocks for tens of ms), started
        // lazily so it only runs while the stats are actually being shown.
        if !self.gpu_started {
            self.gpu_started = true;
            spawn_gpu_sampler(Arc::clone(&self.gpu_shared));
        }
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
        if self.gpu_available {
            push_capped(&mut self.gpu, (self.gpu_pct / 100.0).clamp(0.0, 1.0));
        }
    }
}

fn push_capped(buf: &mut VecDeque<f32>, v: f32) {
    if buf.len() >= GRAPH_POINTS {
        buf.pop_front();
    }
    buf.push_back(v);
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
// Top bar
// ---------------------------------------------------------------------------

/// What the top bar is asking the app to do this frame.
pub enum TopBarAction {
    None,
    /// The user clicked the open-folder button.
    OpenFolder,
    /// The user clicked the settings gear.
    OpenSettings,
    /// The user clicked "Create Backup".
    CreateBackup,
    /// The user clicked "Find Issues". Carries the button's bottom-right so the
    /// Deep Scan window can drop down right-aligned under it (extending left).
    FindIssues(egui::Pos2),
}

/// Render the top bar. Returns any action the app should perform. `show_stats`
/// toggles the centre CPU/RAM graphs; `update_badge` paints a red dot on the
/// settings gear when an app/ComfyUI update is available.
pub fn show(ui: &mut egui::Ui, stats: &SystemStats, show_stats: bool, update_badge: bool) -> TopBarAction {
    let mut action = TopBarAction::None;

    egui::Panel::top("topbar")
        .resizable(false) // This locks the height and removes the drag handle
        .show_separator_line(false)
        .frame(egui::Frame::new().fill(BG()).inner_margin(egui::Margin::same(10)))
        .show_inside(ui, |ui| {
            card_frame(22).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(56.0);

                    // LEFT: folder icon -> open a folder of images.
                    let folder_svg = egui::include_image!("../icons/folder.svg");

                    if svg_button(ui, folder_svg, "Open folder", 37.0, icon_tint(Color32::GRAY)).clicked() {
                        action = TopBarAction::OpenFolder;
                    }

                    // Hardware names stacked in the left corner next to the folder
                    // button: CPU brand (muted) over the GPU product name, which is
                    // colour-coded by its VRAM tier (green / orange / red). Names
                    // truncate to the column; hover shows the full string.
                    if show_stats && !stats.cpu_name.is_empty() {
                        ui.add_space(8.0);
                        ui.allocate_ui_with_layout(
                            egui::vec2(150.0, 42.0),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                ui.set_max_width(150.0);
                                ui.spacing_mut().item_spacing.y = 2.0;
                                let show_gpu = stats.gpu_available && !stats.gpu_name.is_empty();
                                let show_temp = show_gpu && stats.gpu_temp_c > 0.0;
                                // Optically centre 1–3 10.5px lines in the row.
                                ui.add_space(if show_temp {
                                    0.0
                                } else if show_gpu {
                                    6.0
                                } else {
                                    13.0
                                });
                                let cpu = ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(short_cpu_name(&stats.cpu_name))
                                            .color(MUTED())
                                            .size(10.5)
                                            .strong(),
                                    )
                                    .truncate(),
                                );
                                cpu.on_hover_text(&stats.cpu_name);
                                if show_gpu {
                                    let gpu = ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(short_gpu_name(&stats.gpu_name))
                                                .color(vram_color(stats.gpu_mem_total_gb))
                                                .size(10.5)
                                                .strong(),
                                        )
                                        .truncate(),
                                    );
                                    gpu.on_hover_text(format!(
                                        "{} — {:.0} GB VRAM",
                                        stats.gpu_name, stats.gpu_mem_total_gb
                                    ));
                                }
                                // Core temperature under the GPU name — a muted
                                // "Temp" label plus the value, colour-coded:
                                // green cool, orange warm, red hot (80 °C+).
                                if show_temp {
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 4.0;
                                        ui.label(
                                            egui::RichText::new("Temp")
                                                .color(MUTED())
                                                .size(10.5)
                                                .strong(),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:.0}°C", stats.gpu_temp_c))
                                                .color(temp_color(stats.gpu_temp_c))
                                                .size(10.5)
                                                .strong(),
                                        );
                                    });
                                }
                            },
                        );
                    }

                    // RIGHT (laid out right-to-left): settings, the two
                    // action buttons, and then the center stats fill the gap.
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

                            // CENTRE: CPU / RAM / GPU live graphs in the leftover
                            // space. Hidden when the user disables stats in Settings;
                            // the GPU cell only appears when an NVIDIA GPU is found.
                            if show_stats {
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    let avail = ui.available_width();

                                    // The metrics to show this frame.
                                    let mut cells: Vec<StatCell> = vec![
                                        StatCell {
                                            name: "CPU",
                                            value: format!("{:.0}%", stats.cpu_pct),
                                            data: &stats.cpu,
                                            color: CPU_COLOR,
                                            label_w: 44.0,
                                            tooltip: None,
                                        },
                                        StatCell {
                                            name: "RAM",
                                            value: format!(
                                                "{:.1}/{:.0} GB",
                                                stats.ram_used_gb, stats.ram_total_gb
                                            ),
                                            data: &stats.ram,
                                            color: RAM_COLOR,
                                            label_w: 74.0,
                                            tooltip: Some(format!(
                                                "{:.1} / {:.1} GB used",
                                                stats.ram_used_gb, stats.ram_total_gb
                                            )),
                                        },
                                    ];
                                    if stats.gpu_available {
                                        cells.push(StatCell {
                                            name: "GPU",
                                            value: format!("{:.0}%", stats.gpu_pct),
                                            data: &stats.gpu,
                                            color: GPU_COLOR,
                                            label_w: 44.0,
                                            tooltip: Some(format!(
                                                "{:.0}% · VRAM {:.1} / {:.1} GB",
                                                stats.gpu_pct,
                                                stats.gpu_mem_used_gb,
                                                stats.gpu_mem_total_gb
                                            )),
                                        });
                                    }

                                    let n = cells.len() as f32;
                                    let gap = 16.0;
                                    let inner = 8.0; // between a label and its chart
                                    let sum_label: f32 = cells.iter().map(|c| c.label_w).sum();
                                    // Charts share the leftover space equally so all fit.
                                    let chart_w = ((avail - sum_label - n * inner
                                        - (n - 1.0) * gap)
                                        / n)
                                        .clamp(90.0, 240.0);
                                    // As the window narrows, the charts shrink all the
                                    // way down (not stopping at the comfortable 90px
                                    // minimum). Below a useful width they're dropped
                                    // entirely (labels + values remain), and if even
                                    // those don't fit the stats are skipped rather than
                                    // overlapping the action buttons.
                                    let raw =
                                        (avail - sum_label - n * inner - (n - 1.0) * gap) / n;
                                    let show_charts = raw >= 28.0;
                                    let chart_w = if show_charts { chart_w.min(raw) } else { 0.0 };
                                    let inner_w = if show_charts { inner } else { 0.0 };
                                    let total =
                                        sum_label + n * inner_w + chart_w * n + gap * (n - 1.0);

                                    if total <= avail + 0.5 {
                                        // Bias the block toward the right (closer to the
                                        // action buttons) instead of centring it in the
                                        // leftover gap.
                                        let slack = (avail - total).max(0.0);
                                        let left_padding = slack * 0.54;
                                        if left_padding > 0.0 {
                                            ui.add_space(left_padding);
                                        }

                                        for (i, c) in cells.iter().enumerate() {
                                            if i > 0 {
                                                ui.add_space(gap);
                                            }
                                            stat_cell(ui, c, chart_w);
                                        }
                                    }
                                },
                            );
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

/// One metric to render in the centre stats: a short name, its formatted value,
/// the history buffer, and the accent colour.
struct StatCell<'a> {
    name: &'a str,
    value: String,
    data: &'a VecDeque<f32>,
    color: Color32,
    /// Fixed width of this metric's label column. Sized per-metric (narrow for a
    /// "9%" value, wider for "15.0/31 GB") so the number sits snug to its chart
    /// and the metrics stay evenly spaced.
    label_w: f32,
    /// Extra detail shown when hovering the sparkline (e.g. exact GB).
    tooltip: Option<String>,
}

/// Trim marketing noise from a CPU brand string so it fits the label
/// ("AMD Ryzen 9 7950X 16-Core Processor" → "AMD Ryzen 9 7950X",
///  "Intel(R) Core(TM) i7-12700H @ 2.30GHz" → "Intel Core i7-12700H").
fn short_cpu_name(full: &str) -> String {
    let mut s = full.replace("(R)", "").replace("(TM)", "").replace("(tm)", "");
    if let Some(at) = s.find('@') {
        s.truncate(at);
    }
    s.split_whitespace()
        .filter(|w| !matches!(*w, "CPU" | "Processor") && !w.ends_with("-Core"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Trim the vendor prefixes from a GPU name ("NVIDIA GeForce RTX 4080" → "RTX 4080").
fn short_gpu_name(full: &str) -> String {
    full.split_whitespace()
        .filter(|w| !matches!(*w, "NVIDIA" | "GeForce"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Colour-code the GPU name by how much VRAM the card has: 16 GB or more is
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

/// A stat cell: a two-line label (muted name over coloured value) beside a modern
/// gradient-area sparkline.
fn stat_cell(ui: &mut egui::Ui, cell: &StatCell, chart_w: f32) {
    let label_w = cell.label_w;
    ui.allocate_ui_with_layout(
        egui::vec2(label_w, 42.0),
        // Right-align so the value hugs the chart to its right (a tight, readable
        // pairing) rather than floating far to the left of it.
        egui::Layout::top_down(egui::Align::Max),
        |ui| {
            // Pin the label column to a fixed width so the chart never shifts when
            // the value's digit count changes (e.g. 8% → 10% → 100%).
            ui.set_min_width(label_w);
            ui.set_max_width(label_w);
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(cell.name)
                    .color(MUTED())
                    .size(10.5)
                    .strong()
                    .extra_letter_spacing(1.0),
            );
            ui.label(egui::RichText::new(&cell.value).color(cell.color).size(13.5).strong());
        },
    );
    // On very narrow windows the charts are dropped (chart_w == 0) and only the
    // label + value remain.
    if chart_w > 0.0 {
        ui.add_space(8.0);
        let resp = modern_chart(ui, cell.data, cell.color, chart_w);
        if let Some(tip) = &cell.tooltip {
            resp.on_hover_text(tip);
        }
    }
}

/// A modern sparkline: rounded translucent card, a gradient area fading down from
/// the accent-coloured line, faint guides, and a glowing dot at the latest value.
fn modern_chart(ui: &mut egui::Ui, data: &VecDeque<f32>, accent: Color32, width: f32) -> egui::Response {
    let size = egui::vec2(width, 42.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let radius = CornerRadius::same(12);

    // Card + hairline border — white-tinted on dark themes, black-tinted on the
    // light ones (white-on-light would be invisible).
    let (card, border, guide) = if crate::theme::is_light() {
        (
            Color32::from_rgba_unmultiplied(0, 0, 0, 10),
            Color32::from_rgba_unmultiplied(0, 0, 0, 24),
            Color32::from_rgba_unmultiplied(0, 0, 0, 14),
        )
    } else {
        (
            Color32::from_rgba_unmultiplied(255, 255, 255, 10),
            Color32::from_rgba_unmultiplied(255, 255, 255, 20),
            Color32::from_rgba_unmultiplied(255, 255, 255, 12),
        )
    };
    painter.rect_filled(rect, radius, card);
    painter.rect_stroke(rect, radius, Stroke::new(1.0, border), egui::StrokeKind::Inside);

    let pad = 6.0;
    let top = rect.top() + pad;
    let bottom = rect.bottom() - pad;
    let inner_h = bottom - top;

    // Two faint horizontal guides at 1/3 and 2/3.
    for k in 1..=2 {
        let y = top + inner_h * k as f32 / 3.0;
        painter.hline(rect.left() + pad..=rect.right() - pad, y, Stroke::new(1.0, guide));
    }

    if data.len() >= 2 {
        let n = data.len();
        let step = (rect.width() - 2.0 * pad) / (n - 1) as f32;
        let pts: Vec<egui::Pos2> = data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.left() + pad + i as f32 * step;
                let y = bottom - v.clamp(0.0, 1.0) * inner_h;
                egui::pos2(x, y)
            })
            .collect();

        // Gradient area: accent near the line, fading to transparent at the base.
        let fill_top =
            Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 70);
        let fill_bot = Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 0);
        let mut mesh = egui::Mesh::default();
        for (i, p) in pts.iter().enumerate() {
            let base = mesh.vertices.len() as u32;
            mesh.colored_vertex(*p, fill_top);
            mesh.colored_vertex(egui::pos2(p.x, bottom), fill_bot);
            if i > 0 {
                mesh.add_triangle(base - 2, base - 1, base);
                mesh.add_triangle(base - 1, base + 1, base);
            }
        }
        painter.add(egui::Shape::mesh(mesh));

        // The line itself.
        painter.add(egui::Shape::line(pts.clone(), Stroke::new(1.8, accent)));

        // Glowing dot at the current (rightmost) value.
        if let Some(last) = pts.last() {
            painter.circle_filled(
                *last,
                5.0,
                Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 55),
            );
            painter.circle_filled(*last, 2.6, accent);
        }
    }

    resp
}