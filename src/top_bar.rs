//! The top bar — an open-folder button, live CPU/RAM graphs, and the
//! backup / find-issues / settings action buttons. Split out of `main.rs`.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use eframe::egui;
use egui::{Color32, CornerRadius, Stroke};

use crate::theme::*;
use crate::{card_frame, svg_button};

const GRAPH_POINTS: usize = 120;

// ---------------------------------------------------------------------------
// Live CPU / RAM sampling for the center graphs.
// ---------------------------------------------------------------------------
pub struct SystemStats {
    sys: sysinfo::System,
    last_sample: Instant,
    cpu: VecDeque<f32>, // each value 0.0..=1.0
    ram: VecDeque<f32>, // each value 0.0..=1.0
    cpu_pct: f32,
    ram_used_gb: f64,
    ram_total_gb: f64,
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
            cpu_pct: 0.0,
            ram_used_gb: 0.0,
            ram_total_gb: 0.0,
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

        self.cpu_pct = self.sys.global_cpu_usage();
        let used = self.sys.used_memory() as f64;
        let total = self.sys.total_memory().max(1) as f64;
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        self.ram_used_gb = used / GIB;
        self.ram_total_gb = total / GIB;

        push_capped(&mut self.cpu, (self.cpu_pct / 100.0).clamp(0.0, 1.0));
        push_capped(&mut self.ram, (used / total).clamp(0.0, 1.0) as f32);
    }
}

fn push_capped(buf: &mut VecDeque<f32>, v: f32) {
    if buf.len() >= GRAPH_POINTS {
        buf.pop_front();
    }
    buf.push_back(v);
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
/// toggles the centre CPU/RAM graphs.
pub fn show(ui: &mut egui::Ui, stats: &SystemStats, show_stats: bool) -> TopBarAction {
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

                    // RIGHT (laid out right-to-left): settings, the two
                    // action buttons, and then the center stats fill the gap.
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            let settings_svg = egui::include_image!("../icons/top_settings.svg");
                            if svg_button(ui, settings_svg, "Settings", 37.0, MUTED()).clicked() {
                                action = TopBarAction::OpenSettings;
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

                            // CENTRE: CPU / RAM live graphs in the leftover space.
                            // Hidden when the user disables stats in Settings.
                            if show_stats {
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    let avail = ui.available_width();

                                    // Stretch dynamically, but max out at 280px so it isn't massive.
                                    let chart_w = (avail * 0.35).clamp(150.0, 280.0);

                                    // Math to perfectly center the two stats in the remaining gap
                                    let text_w = 65.0; // Estimate for label width
                                    let gap_w = 32.0;  // Gap between CPU and RAM
                                    let total_center_w = (text_w + 8.0 + chart_w) * 2.0 + gap_w;

                                    let left_padding = (avail - total_center_w) / 2.0;
                                    if left_padding > 0.0 {
                                        ui.add_space(left_padding);
                                    }

                                    stat_graph(
                                        ui,
                                        &format!("CPU: {:.0}%", stats.cpu_pct),
                                        &stats.cpu,
                                        chart_w
                                    );
                                    ui.add_space(gap_w);
                                    stat_graph(
                                        ui,
                                        &format!(
                                            "RAM: {:.1} / {:.1} GB",
                                            stats.ram_used_gb, stats.ram_total_gb
                                        ),
                                        &stats.ram,
                                        chart_w
                                    );
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

/// CPU/RAM label + live line graph, matching the top bar's center stats.
fn stat_graph(ui: &mut egui::Ui, label: &str, data: &VecDeque<f32>, graph_width: f32) {
    ui.label(egui::RichText::new(label).color(MUTED()).size(12.0));
    ui.add_space(8.0);

    // Replaced the hardcoded 150.0 width with our dynamic parameter
    let size = egui::vec2(graph_width, 36.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let bg = Color32::from_rgba_unmultiplied(235, 235, 235, 15);
    let guide = Color32::from_rgba_unmultiplied(235, 235, 235, 25);
    let border = Color32::from_rgba_unmultiplied(235, 235, 235, 35);
    let line = Color32::from_rgba_unmultiplied(64, 140, 255, 220);

    painter.rect_filled(rect, CornerRadius::same(10), bg);

    let pad = 6.0;
    for k in 1..=2 {
        let y = rect.top() + pad + (rect.height() - 2.0 * pad) * k as f32 / 3.0;
        painter.hline(rect.left() + pad..=rect.right() - pad, y, Stroke::new(1.0, guide));
    }
    painter.rect_stroke(
        rect,
        CornerRadius::same(10),
        Stroke::new(1.0, border),
        egui::StrokeKind::Inside,
    );

    if data.len() >= 2 {
        let n = data.len();
        let step = (rect.width() - 2.0 * pad) / (n - 1) as f32;
        let points: Vec<egui::Pos2> = data
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = rect.left() + pad + i as f32 * step;
                let y = rect.top() + pad + (1.0 - v) * (rect.height() - 2.0 * pad);
                egui::pos2(x, y)
            })
            .collect();
        painter.add(egui::Shape::line(points, Stroke::new(1.4, line)));
    }
}