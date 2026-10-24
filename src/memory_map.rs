//! The memory map: the AI's long-term memories (src/memory.rs) drawn as a
//! cloudy 3D brain, from the "Creating 3D visualizations in chat" design.
//! Each bright point is one memory; points of one colour share a topic and
//! gather in one region of the brain, linked to their nearest neighbours.
//! The brain sways gently while idle; drag to turn it, scroll to zoom,
//! double-click to reset, and click a point to read the memory in the panel
//! on the right, where it can be edited, forgotten, or asked about in chat.
//!
//! Everything is drawn with egui's painter: a fixed, seeded cloud of points
//! inside a brain-shaped volume (two hemispheres split by the central fissure,
//! temporal lobes, cerebellum and brainstem), rotated and projected with a
//! simple perspective every frame, far points dimmer and smaller than near
//! ones. Opened from AI Settings → Memory (src/ai_settings.rs), or → Role
//! play for role play's own memories (the same map, a separate store).

use eframe::egui;
use egui::{Color32, CornerRadius, Pos2, Rect, Stroke, Vec2};

use crate::memory::{format_date, MemoryKind, MemoryStore};
use crate::theme::*;

/// Topic colours, in topic order (wraps after ten).
const TOPIC_COLORS: [Color32; 10] = [
    Color32::from_rgb(212, 83, 126),  // pink
    Color32::from_rgb(127, 119, 221), // purple
    Color32::from_rgb(29, 158, 117),  // teal
    Color32::from_rgb(214, 140, 40),  // amber
    Color32::from_rgb(55, 138, 221),  // blue
    Color32::from_rgb(226, 75, 74),   // red
    Color32::from_rgb(59, 179, 195),  // cyan
    Color32::from_rgb(139, 195, 74),  // green
    Color32::from_rgb(242, 169, 59),  // orange
    Color32::from_rgb(178, 106, 217), // violet
];

/// The colour of topic `i` (also used by the settings tab's topic list).
pub fn topic_color(i: usize) -> Color32 {
    TOPIC_COLORS[i % TOPIC_COLORS.len()]
}

/// The haze and neuron colour: the mockup's soft purple.
const HAZE: Color32 = Color32::from_rgb(127, 119, 221);

/// The brain volume as ellipsoids (centre, radii) in brain space: x runs
/// left → right, y down → up, z back → front.
const PARTS: [([f32; 3], [f32; 3]); 6] = [
    ([-0.30, 0.10, 0.00], [0.40, 0.52, 0.92]), // left hemisphere
    ([0.30, 0.10, 0.00], [0.40, 0.52, 0.92]),  // right hemisphere
    ([-0.40, -0.22, 0.18], [0.22, 0.24, 0.48]), // left temporal lobe
    ([0.40, -0.22, 0.18], [0.22, 0.24, 0.48]), // right temporal lobe
    ([0.00, -0.36, -0.66], [0.46, 0.22, 0.30]), // cerebellum
    ([0.00, -0.62, -0.30], [0.11, 0.30, 0.12]), // brainstem
];

/// Where each topic's memories gather, in topic order: the frontal, parietal,
/// temporal and occipital lobes on each side, the cerebellum, then the core.
const ANCHORS: [[f32; 3]; 10] = [
    [-0.28, 0.28, 0.55],  // left frontal
    [0.28, 0.28, 0.55],   // right frontal
    [-0.30, 0.42, -0.15], // left parietal
    [0.30, 0.42, -0.15],  // right parietal
    [-0.42, -0.20, 0.22], // left temporal
    [0.42, -0.20, 0.22],  // right temporal
    [-0.28, 0.10, -0.68], // left occipital
    [0.28, 0.10, -0.68],  // right occipital
    [0.00, -0.36, -0.66], // cerebellum
    [0.00, -0.15, 0.05],  // core
];

/// The view's starting angle: a three-quarter side view, front to the right,
/// looking slightly down.
const START_YAW: f32 = -1.30;
const START_PITCH: f32 = 0.24;

/// A tiny deterministic RNG (xorshift64*), so the cloud and every memory's
/// position come out the same on every run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng((seed ^ 0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

/// How deep `p` sits in the brain: `None` outside, else the smallest
/// normalised ellipsoid distance (0 at a part's centre, 1 at its surface).
fn depth_in_brain(p: [f32; 3]) -> Option<f32> {
    // The longitudinal fissure between the hemispheres, open at the top.
    if p[0].abs() < 0.035 && p[1] > -0.05 {
        return None;
    }
    let mut best: Option<f32> = None;
    for (c, r) in PARTS {
        let q = ((p[0] - c[0]) / r[0]).powi(2) + ((p[1] - c[1]) / r[1]).powi(2) + ((p[2] - c[2]) / r[2]).powi(2);
        if q <= 1.0 {
            best = Some(best.map_or(q, |b: f32| b.min(q)));
        }
    }
    best
}

/// Share of the points each part gets, in `PARTS` order. Sampling by volume
/// alone would leave the small parts (cerebellum, brainstem) nearly empty,
/// and they're what make the cloud read as a brain rather than an oval.
const PART_SHARE: [f32; 6] = [0.31, 0.31, 0.08, 0.08, 0.15, 0.07];

/// About `n` points in the brain with the part each came from, favouring the
/// outer surface by `bias` (0 = uniform through the volume; higher = more of
/// a shell, which shows the shape).
fn sample_brain(rng: &mut Rng, n: usize, bias: f32) -> Vec<([f32; 3], usize)> {
    let mut out = Vec::with_capacity(n);
    for (k, &(c, r)) in PARTS.iter().enumerate() {
        let want = (n as f32 * PART_SHARE[k]).round() as usize;
        let (mut got, mut tries) = (0, 0);
        while got < want && tries < want * 400 + 1000 {
            tries += 1;
            let p = [
                c[0] + r[0] * rng.range(-1.0, 1.0),
                c[1] + r[1] * rng.range(-1.0, 1.0),
                c[2] + r[2] * rng.range(-1.0, 1.0),
            ];
            let own = ((p[0] - c[0]) / r[0]).powi(2) + ((p[1] - c[1]) / r[1]).powi(2) + ((p[2] - c[2]) / r[2]).powi(2);
            if own > 1.0 {
                continue;
            }
            // Depth in the whole brain, so points buried where parts
            // overlap stay rare and the outer surface carries the shape.
            let Some(q) = depth_in_brain(p) else { continue };
            if rng.f() < (1.0 - bias) + bias * q * q {
                out.push((p, k));
                got += 1;
            }
        }
    }
    out
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Each point linked to its `k` nearest neighbours (pairs listed once).
fn nearest_links(points: &[[f32; 3]], k: usize) -> Vec<(usize, usize)> {
    let mut links = Vec::new();
    for i in 0..points.len() {
        let mut near: Vec<(f32, usize)> =
            (0..points.len()).filter(|&j| j != i).map(|j| (dist2(points[i], points[j]), j)).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, j) in near.iter().take(k) {
            let pair = (i.min(j), i.max(j));
            if !links.contains(&pair) {
                links.push(pair);
            }
        }
    }
    links
}

/// The fixed backdrop: soft haze, faint neurons and their wiring.
struct Cloud {
    /// Position, radius (brain units) and opacity.
    haze: Vec<([f32; 3], f32, f32)>,
    /// Position and radius (screen points).
    dots: Vec<([f32; 3], f32)>,
    links: Vec<(usize, usize)>,
}

fn build_cloud() -> Cloud {
    let mut rng = Rng::new(7);
    // Haze puffs shrink with the part they sit in, so the small parts keep
    // a crisp outline instead of dissolving into one blob.
    const PUFF: [f32; 6] = [1.0, 1.0, 0.75, 0.75, 0.7, 0.5];
    let haze = sample_brain(&mut rng, 420, 0.8)
        .into_iter()
        .map(|(p, k)| {
            let r = rng.range(0.045, 0.095) * PUFF[k];
            let a = rng.range(0.035, 0.075);
            (p, r, a)
        })
        .collect();
    let dot_pts: Vec<[f32; 3]> = sample_brain(&mut rng, 300, 0.9).into_iter().map(|(p, _)| p).collect();
    let links = nearest_links(&dot_pts, 2);
    let dots = dot_pts.into_iter().map(|p| (p, rng.range(1.0, 2.0))).collect();
    Cloud { haze, dots, links }
}

/// One memory's place in the brain.
struct Node {
    topic: usize,
    id: u64,
    pos: [f32; 3],
}

/// Where every memory sits, rebuilt when the store changes.
struct Layout {
    revision: u64,
    nodes: Vec<Node>,
    links: Vec<(usize, usize)>,
}

fn build_layout(store: &MemoryStore) -> Layout {
    let mut nodes: Vec<Node> = Vec::new();
    let mut links = Vec::new();
    for (ti, topic) in store.topics.iter().enumerate() {
        let mut anchor = ANCHORS[ti % ANCHORS.len()];
        // Topics past the tenth reuse a region, shifted so they don't overlap.
        let wrap = (ti / ANCHORS.len()) as f32;
        anchor[1] += 0.12 * wrap * if ti % 2 == 0 { 1.0 } else { -1.0 };
        let radius = 0.10 + 0.025 * (topic.memories.len() as f32).sqrt().min(8.0);
        let first = nodes.len();
        for m in &topic.memories {
            // Seeded by the memory's id: a memory keeps its spot across runs.
            let mut rng = Rng::new(m.id.wrapping_mul(0x9E37_79B9) ^ ((ti as u64) << 40));
            let mut pos = anchor;
            for attempt in 0..120 {
                let v = loop {
                    let v = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
                    if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] <= 1.0 {
                        break v;
                    }
                };
                let cand = [anchor[0] + v[0] * radius, anchor[1] + v[1] * radius, anchor[2] + v[2] * radius];
                if depth_in_brain(cand).is_none() {
                    continue;
                }
                let crowded = nodes[first..].iter().any(|n| dist2(n.pos, cand) < 0.05 * 0.05);
                if crowded && attempt < 100 {
                    continue;
                }
                pos = cand;
                break;
            }
            nodes.push(Node { topic: ti, id: m.id, pos });
        }
        let pts: Vec<[f32; 3]> = nodes[first..].iter().map(|n| n.pos).collect();
        links.extend(nearest_links(&pts, 2).into_iter().map(|(a, b)| (first + a, first + b)));
    }
    Layout { revision: store.revision, nodes, links }
}

/// Rotation + perspective for one frame.
struct View {
    cy: f32,
    sy: f32,
    cp: f32,
    sp: f32,
    unit: f32,
    center: Pos2,
}

impl View {
    fn new(yaw: f32, pitch: f32, rect: Rect, zoom: f32) -> Self {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let unit = rect.width().min(rect.height() * 1.45) * 0.36 * zoom;
        // The brain spans y = -0.94 … 0.64; lift it a little so that span sits
        // centred once the downward tilt (pitch) is applied.
        let center = rect.center() - egui::vec2(0.0, 0.08 * unit);
        View { cy, sy, cp, sp, unit, center }
    }

    /// Screen position, perspective scale and nearness (0 far … 1 near).
    fn project(&self, p: [f32; 3]) -> (Pos2, f32, f32) {
        let x1 = p[0] * self.cy - p[2] * self.sy;
        let z1 = p[0] * self.sy + p[2] * self.cy;
        let y2 = p[1] * self.cp - z1 * self.sp;
        let z2 = p[1] * self.sp + z1 * self.cp;
        let s = 3.4 / (3.4 - z2);
        let pos = self.center + egui::vec2(x1 * s * self.unit, -y2 * s * self.unit);
        (pos, s, ((z2 + 1.0) * 0.5).clamp(0.0, 1.0))
    }
}

/// A radial glow: `col` at the centre, half strength at 45% of the radius,
/// fully clear at the rim, so there are no hard ring edges.
fn soft_glow(painter: &egui::Painter, c: Pos2, r: f32, col: Color32) {
    let n = 40u32;
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(c, col);
    let mid = col.gamma_multiply(0.5);
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        let d = egui::vec2(a.cos(), a.sin());
        mesh.colored_vertex(c + d * (r * 0.45), mid);
        mesh.colored_vertex(c + d * r, Color32::TRANSPARENT);
    }
    for i in 0..n {
        let (m0, o0) = (1 + 2 * i, 2 + 2 * i);
        let (m1, o1) = (1 + 2 * ((i + 1) % n), 2 + 2 * ((i + 1) % n));
        mesh.add_triangle(0, m0, m1);
        mesh.add_triangle(m0, o0, o1);
        mesh.add_triangle(m0, o1, m1);
    }
    painter.add(mesh);
}

/// What the right-hand panel shows.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Panel {
    /// Nothing selected: how to read the map, and the topic list.
    #[default]
    Overview,
    /// The selected memory.
    View,
    Edit,
    ConfirmDelete,
    /// Writing a new memory by hand.
    New,
}

/// The memory map window's state, owned by `LlmState`.
pub struct MapState {
    pub open: bool,
    /// Which store the map shows: the main memory or role play's.
    kind: MemoryKind,
    /// Role play: the character's name, for the title ("Mira's memories").
    ai_name: String,
    /// A memory just forgotten, for the window to finish off (role play
    /// deletes the album picture it was about).
    forgotten: Option<crate::memory::Memory>,
    /// Rotation the user dragged to (the idle sway swings around it).
    base_yaw: f32,
    pitch: f32,
    zoom: f32,
    /// Idle-sway clock (seconds); frozen while the pointer is over the brain
    /// so points hold still to be clicked.
    sway: f64,
    last_time: Option<f64>,
    search: String,
    /// Topic names switched off with the pills.
    hidden: Vec<String>,
    /// The selected memory's id (ids are unique across topics).
    selected: Option<u64>,
    panel: Panel,
    draft_topic: String,
    draft_new_topic: String,
    draft_title: String,
    draft_text: String,
    cloud: Option<Cloud>,
    layout: Option<Layout>,
}

impl Default for MapState {
    fn default() -> Self {
        Self {
            open: false,
            kind: MemoryKind::Main,
            ai_name: String::new(),
            forgotten: None,
            base_yaw: START_YAW,
            pitch: START_PITCH,
            zoom: 1.0,
            sway: 0.0,
            last_time: None,
            search: String::new(),
            hidden: Vec::new(),
            selected: None,
            panel: Panel::Overview,
            draft_topic: String::new(),
            draft_new_topic: String::new(),
            draft_title: String::new(),
            draft_text: String::new(),
            cloud: None,
            layout: None,
        }
    }
}

/// Where the selected memory lives: (topic index, memory index).
fn find(store: &MemoryStore, id: u64) -> Option<(usize, usize)> {
    store
        .topics
        .iter()
        .enumerate()
        .find_map(|(ti, t)| t.memories.iter().position(|m| m.id == id).map(|mi| (ti, mi)))
}

/// Draw the memory map window when it's open. "Ask about this" puts a
/// question into the chat's input box and closes the map.
pub fn window(ctx: &egui::Context, llm: &mut crate::llm::LlmState) {
    if !llm.memory_map.open {
        return;
    }
    let mut map = std::mem::take(&mut llm.memory_map);
    map.ai_name = llm.roleplay.ai_display().to_string();
    let mut ask: Option<String> = None;
    let screen = ctx.content_rect();
    let size = egui::vec2(
        (screen.width() - 120.0).clamp(640.0, 980.0),
        (screen.height() - 140.0).clamp(420.0, 640.0),
    );
    egui::Window::new("Memory map")
        .id(egui::Id::new("memory_map_window"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(crate::settings::window_frame())
        .fixed_size(size)
        .show(ctx, |ui| {
            let store = match map.kind {
                MemoryKind::Main => &mut llm.memory,
                MemoryKind::Roleplay => &mut llm.rp_memory,
            };
            map.ui(ui, store, &mut ask);
        });
    // Role play: forgetting a memory about a kept picture deletes the picture.
    if let Some(m) = map.forgotten.take()
        && map.kind == MemoryKind::Roleplay
        && let Some(image) = m.image
    {
        crate::roleplay::remove_album_picture(&image);
    }
    if let Some(q) = ask {
        llm.draft = q;
        map.open = false;
    }
    llm.memory_map = map;
}

impl MapState {
    /// Open the map on `kind`'s memories (switching stores starts afresh:
    /// no selection, search or hidden topics carried over).
    pub fn show(&mut self, kind: MemoryKind) {
        if self.kind != kind {
            *self = MapState { kind, ..Default::default() };
        }
        self.open = true;
    }

    /// The window's title: "Memory map", or the character's memories.
    fn title(&self) -> String {
        match self.kind {
            MemoryKind::Main => "Memory map".to_string(),
            MemoryKind::Roleplay if self.ai_name.is_empty() || self.ai_name == "the AI" => {
                "Role play memories".to_string()
            }
            MemoryKind::Roleplay => format!("{}'s memories", self.ai_name),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, store: &mut MemoryStore, ask: &mut Option<String>) {
        // Same rounded inputs as the Settings window.
        let radius = CornerRadius::same(10);
        let v = ui.visuals_mut();
        v.widgets.noninteractive.corner_radius = radius;
        v.widgets.inactive.corner_radius = radius;
        v.widgets.hovered.corner_radius = radius;
        v.widgets.active.corner_radius = radius;
        v.widgets.open.corner_radius = radius;

        if self.cloud.is_none() {
            self.cloud = Some(build_cloud());
        }
        if self.layout.as_ref().map(|l| l.revision) != Some(store.revision) {
            self.layout = Some(build_layout(store));
            if self.selected.is_some_and(|id| find(store, id).is_none()) {
                self.selected = None;
                if matches!(self.panel, Panel::View | Panel::Edit | Panel::ConfirmDelete) {
                    self.panel = Panel::Overview;
                }
            }
        }

        // Esc backs out of an edit first, then closes the map.
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            match self.panel {
                Panel::Edit | Panel::ConfirmDelete => self.panel = Panel::View,
                Panel::New => self.panel = Panel::Overview,
                _ => self.open = false,
            }
        }

        self.header(ui, store);
        ui.add_space(10.0);
        self.filters(ui, store);
        ui.add_space(8.0);

        let avail = ui.available_size();
        let detail_w = 280.0;
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 16.0;
            self.canvas(ui, egui::vec2(avail.x - detail_w - 16.0, avail.y), store);
            ui.vertical(|ui| {
                ui.set_width(detail_w);
                ui.set_height(avail.y);
                egui::ScrollArea::vertical()
                    .id_salt("memory_detail")
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.detail(ui, store, ask));
            });
        });
    }

    /// Icon + title + counts on the left; "New memory" and close on the right.
    fn header(&mut self, ui: &mut egui::Ui, store: &MemoryStore) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            ui.add(
                egui::Image::new(egui::include_image!("../icons/ai-brain.svg"))
                    .fit_to_exact_size(egui::vec2(22.0, 22.0))
                    .tint(TEXT()),
            );
            ui.label(egui::RichText::new(self.title()).color(TEXT()).strong().size(17.0));
            let (n, t) = (store.count(), store.topics.len());
            let counts = format!(
                "{n} {} · {t} {}",
                if n == 1 { "memory" } else { "memories" },
                if t == 1 { "topic" } else { "topics" }
            );
            ui.label(egui::RichText::new(counts).color(MUTED()).size(12.5));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(
                        egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/close.svg"))
                                .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                .tint(TEXT()),
                        )
                        .frame(false),
                    )
                    .on_hover_text("Close")
                    .clicked()
                {
                    self.open = false;
                }
                if !store.load_failed {
                    let new = egui::Button::image_and_text(
                        egui::Image::new(egui::include_image!("../icons/add.svg"))
                            .fit_to_exact_size(egui::vec2(14.0, 14.0))
                            .tint(TEXT()),
                        egui::RichText::new("New memory").size(12.5),
                    );
                    if ui.add(new).on_hover_text("Write a memory yourself").clicked() {
                        self.start_new(store);
                    }
                }
            });
        });
    }

    fn start_new(&mut self, store: &MemoryStore) {
        self.panel = Panel::New;
        self.draft_topic = match store.topics.first() {
            Some(t) => t.name.clone(),
            None if self.kind == MemoryKind::Roleplay => "Story".to_string(),
            None => String::new(),
        };
        self.draft_new_topic.clear();
        self.draft_title.clear();
        self.draft_text.clear();
    }

    /// The search box and one pill per topic (click to show / hide it).
    fn filters(&mut self, ui: &mut egui::Ui, store: &MemoryStore) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .hint_text("Search memories")
                    .desired_width(220.0),
            );
            ui.add_space(6.0);
            for (ti, topic) in store.topics.iter().enumerate() {
                let on = !self.hidden.contains(&topic.name);
                let label = format!("{}  {}", topic.name, topic.memories.len());
                if pill(ui, topic_color(ti), &label, on).clicked() {
                    if on {
                        self.hidden.push(topic.name.clone());
                    } else {
                        self.hidden.retain(|h| h != &topic.name);
                    }
                }
            }
        });
    }

    fn matches(&self, title: &str, text: &str) -> bool {
        let q = self.search.trim().to_lowercase();
        q.is_empty() || title.to_lowercase().contains(&q) || text.to_lowercase().contains(&q)
    }

    /// The brain itself.
    fn canvas(&mut self, ui: &mut egui::Ui, size: Vec2, store: &MemoryStore) {
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let light = is_light();
        let bg = if light { Color32::from_rgb(247, 247, 251) } else { Color32::from_rgb(16, 16, 20) };
        painter.rect_filled(rect, CornerRadius::same(14), bg);

        // --- Motion: idle sway, drag to turn, scroll / pinch to zoom. ---
        let now = ui.input(|i| i.time);
        let dt = self.last_time.map_or(0.0, |t| (now - t).clamp(0.0, 0.1));
        self.last_time = Some(now);
        let hovering = resp.hovered() || resp.dragged();
        if !hovering {
            self.sway += dt;
        }
        if resp.dragged() {
            let d = resp.drag_delta();
            self.base_yaw += d.x * 0.008;
            self.pitch = (self.pitch + d.y * 0.006).clamp(-0.9, 0.9);
        }
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            self.zoom = (self.zoom * (1.0 + scroll * 0.0015) * pinch).clamp(0.6, 2.6);
        }
        if resp.double_clicked() {
            self.base_yaw = START_YAW;
            self.pitch = START_PITCH;
            self.zoom = 1.0;
        }
        let yaw = self.base_yaw + 0.5 * (self.sway * 0.35).sin() as f32;
        let view = View::new(yaw, self.pitch, rect, self.zoom);
        let size_k = self.zoom.sqrt();

        // --- The cloud: a backdrop glow, haze, then faint neurons. ---
        soft_glow(
            &painter,
            view.center + egui::vec2(0.0, 0.1 * view.unit),
            view.unit * 1.2,
            HAZE.gamma_multiply(if light { 0.10 } else { 0.16 }),
        );
        let cloud = self.cloud.as_ref().expect("built above");
        for &(p, r, a) in &cloud.haze {
            let (pos, s, d) = view.project(p);
            painter.circle_filled(pos, r * view.unit * s, HAZE.gamma_multiply(a * (0.45 + 0.55 * d)));
        }
        let dot_col = if light { Color32::from_rgb(96, 88, 190) } else { HAZE.lerp_to_gamma(Color32::WHITE, 0.25) };
        let projected: Vec<(Pos2, f32, f32)> = cloud.dots.iter().map(|&(p, _)| view.project(p)).collect();
        for &(a, b) in &cloud.links {
            let (pa, _, da) = projected[a];
            let (pb, _, db) = projected[b];
            let k = 0.06 + 0.10 * (da + db) * 0.5;
            painter.line_segment([pa, pb], Stroke::new(0.6_f32, dot_col.gamma_multiply(k)));
        }
        for (&(_, r), &(pos, s, d)) in cloud.dots.iter().zip(&projected) {
            painter.circle_filled(pos, r * s * size_k, dot_col.gamma_multiply(0.25 + 0.4 * d));
        }

        // --- Memories: project, hit-test, then draw far to near. ---
        let layout = self.layout.as_ref().expect("built above");
        struct Shown {
            node: usize,
            pos: Pos2,
            s: f32,
            d: f32,
            visible: bool,
            hit: bool,
        }
        let mut shown: Vec<Shown> = Vec::with_capacity(layout.nodes.len());
        for (i, n) in layout.nodes.iter().enumerate() {
            let (pos, s, d) = view.project(n.pos);
            let topic = &store.topics[n.topic];
            let visible = !self.hidden.contains(&topic.name);
            let hit = visible
                && topic.memories.iter().find(|m| m.id == n.id).is_some_and(|m| self.matches(&m.title, &m.text));
            shown.push(Shown { node: i, pos, s, d, visible, hit });
        }

        let pointer = resp.hover_pos();
        let hovered: Option<usize> = pointer.and_then(|p| {
            shown
                .iter()
                .filter(|sh| sh.visible && sh.hit)
                .map(|sh| (sh.pos.distance(p), sh))
                .filter(|(dist, sh)| *dist < 9.0 + 3.0 * sh.s)
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, sh)| sh.node)
        });
        if hovered.is_some() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        } else if resp.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }
        if resp.clicked() {
            match hovered {
                Some(i) => {
                    self.selected = Some(layout.nodes[i].id);
                    self.panel = Panel::View;
                }
                // A click on empty space clears the selection, but never
                // throws away an edit in progress.
                None if self.panel == Panel::View => {
                    self.selected = None;
                    self.panel = Panel::Overview;
                }
                None => {}
            }
        }

        for &(a, b) in &layout.links {
            let (sa, sb) = (&shown[a], &shown[b]);
            if !(sa.visible && sb.visible) {
                continue;
            }
            let col = topic_color(layout.nodes[a].topic);
            let k = if sa.hit && sb.hit { 0.18 + 0.30 * (sa.d + sb.d) * 0.5 } else { 0.05 };
            painter.line_segment([sa.pos, sb.pos], Stroke::new(0.9_f32, col.gamma_multiply(k)));
        }
        let mut order: Vec<&Shown> = shown.iter().filter(|sh| sh.visible).collect();
        order.sort_by(|a, b| a.d.total_cmp(&b.d));
        for sh in order {
            let n = &layout.nodes[sh.node];
            let col = topic_color(n.topic);
            let near = 0.45 + 0.55 * sh.d;
            let is_sel = self.selected == Some(n.id);
            let is_hov = hovered == Some(sh.node);
            if !sh.hit {
                painter.circle_filled(sh.pos, 2.6 * sh.s * size_k, col.gamma_multiply(0.18));
                continue;
            }
            soft_glow(&painter, sh.pos, 13.0 * sh.s * size_k, col.gamma_multiply(0.42 * near));
            let r = if is_sel || is_hov { 5.6 } else { 3.8 } * sh.s * size_k;
            painter.circle_filled(sh.pos, r, col.gamma_multiply(0.55 + 0.45 * near));
            if is_sel {
                let ring = if light { Color32::from_gray(30) } else { Color32::WHITE };
                painter.circle_stroke(sh.pos, r + 2.5, Stroke::new(1.5_f32, ring));
            }
        }

        // The hovered memory's title, in a chip beside it.
        if let Some(i) = hovered {
            let n = &layout.nodes[i];
            if let Some(m) = store.topics[n.topic].memories.iter().find(|m| m.id == n.id) {
                let mut title: String = m.title.chars().take(48).collect();
                if m.title.chars().count() > 48 {
                    title.push('…');
                }
                let galley = painter.layout_no_wrap(title, egui::FontId::proportional(12.5), TEXT());
                let pad = egui::vec2(8.0, 4.0);
                let mut chip = Rect::from_min_size(shown[i].pos + egui::vec2(12.0, -28.0), galley.size() + pad * 2.0);
                if chip.right() > rect.right() - 6.0 {
                    chip = chip.translate(egui::vec2(-(chip.width() + 24.0), 0.0));
                }
                if chip.top() < rect.top() + 6.0 {
                    chip = chip.translate(egui::vec2(0.0, 40.0));
                }
                painter.rect_filled(chip, CornerRadius::same(8), PANEL());
                painter.rect_stroke(chip, CornerRadius::same(8), Stroke::new(1.0_f32, topic_color(n.topic)), egui::StrokeKind::Inside);
                painter.galley(chip.min + pad, galley, TEXT());
            }
        }

        // Nothing remembered yet: say how memories get here.
        if store.count() == 0 {
            let msg = painter.layout(
                "No memories yet".to_string(),
                egui::FontId::proportional(15.0),
                TEXT(),
                rect.width() * 0.7,
            );
            let how = match self.kind {
                MemoryKind::Main => {
                    "Turn on Remember across chats in AI Settings, and the AI will file what it learns \
                     here after each chat. You can also add one yourself with New memory."
                }
                MemoryKind::Roleplay => {
                    "The AI writes memories here as the story unfolds. You can also add one yourself \
                     with New memory."
                }
            };
            let sub = painter.layout(
                how.to_string(),
                egui::FontId::proportional(12.0),
                MUTED(),
                rect.width() * 0.7,
            );
            let y = rect.bottom() - 24.0 - sub.size().y - msg.size().y - 6.0;
            painter.galley(egui::pos2(rect.center().x - msg.size().x / 2.0, y), msg.clone(), TEXT());
            painter.galley(
                egui::pos2(rect.center().x - sub.size().x / 2.0, y + msg.size().y + 6.0),
                sub,
                MUTED(),
            );
        }

        // Hint in the corner.
        painter.text(
            rect.left_bottom() + egui::vec2(12.0, -10.0),
            egui::Align2::LEFT_BOTTOM,
            if store.count() == 0 { "" } else { "Drag to turn · scroll to zoom · double-click to reset" },
            egui::FontId::proportional(11.0),
            MUTED().gamma_multiply(0.8),
        );

        // Keep the idle sway moving (~30 fps); while the pointer is over the
        // brain it holds still and redraws only on input.
        if !hovering {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
        }
    }

    /// The right-hand panel: overview, the selected memory, or a form.
    fn detail(&mut self, ui: &mut egui::Ui, store: &mut MemoryStore, ask: &mut Option<String>) {
        ui.spacing_mut().item_spacing.y = 6.0;
        if store.load_failed {
            ui.label(egui::RichText::new("Memories couldn't be read").color(TEXT()).strong().size(14.0));
            ui.label(
                egui::RichText::new(match self.kind {
                    MemoryKind::Main => {
                        "They were probably saved on another computer or user account. They're left \
                         untouched; AI Settings → Memory can start fresh."
                    }
                    MemoryKind::Roleplay => {
                        "They were probably saved on another computer or user account. They're left \
                         untouched; AI Settings → Role play can start fresh."
                    }
                })
                .color(MUTED())
                .size(12.0),
            );
            return;
        }
        match self.panel {
            Panel::Overview => self.overview(ui, store),
            Panel::View => self.view(ui, store, ask),
            Panel::Edit => self.edit(ui, store),
            Panel::ConfirmDelete => self.confirm_delete(ui, store),
            Panel::New => self.new_form(ui, store),
        }
    }

    fn overview(&mut self, ui: &mut egui::Ui, store: &MemoryStore) {
        caption(ui, "HOW TO READ IT");
        ui.label(
            egui::RichText::new(
                "Each point is one memory. Points of the same colour share a topic and sit together \
                 in one part of the brain, linked to their closest neighbours.",
            )
            .color(TEXT())
            .size(12.5),
        );
        ui.label(egui::RichText::new("Click a point to read it.").color(MUTED()).size(12.0));
        ui.add_space(8.0);
        if !store.topics.is_empty() {
            caption(ui, "TOPICS");
            for (ti, t) in store.topics.iter().enumerate() {
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 16.0), egui::Sense::hover());
                    ui.painter().circle_filled(r.center(), 4.0, topic_color(ti));
                    ui.label(egui::RichText::new(&t.name).color(TEXT()).size(12.5));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new(t.memories.len().to_string()).color(MUTED()).size(12.0));
                    });
                });
            }
            ui.add_space(8.0);
        }
        if let Some(t) = store.last_updated() {
            ui.label(egui::RichText::new(format!("Last updated {}", format_date(t))).color(MUTED()).size(11.5));
        }
    }

    fn view(&mut self, ui: &mut egui::Ui, store: &MemoryStore, ask: &mut Option<String>) {
        let Some((ti, mi)) = self.selected.and_then(|id| find(store, id)) else {
            self.panel = Panel::Overview;
            return;
        };
        let topic = &store.topics[ti];
        let m = &topic.memories[mi];
        ui.horizontal(|ui| {
            chip(ui, topic_color(ti), &topic.name);
            let when = if m.updated > m.created {
                format!("Updated {}", format_date(m.updated))
            } else {
                format!("Saved {}", format_date(m.created))
            };
            let who = match (m.by_ai, self.kind) {
                (false, _) => "added by you".to_string(),
                (true, MemoryKind::Main) => "by the AI".to_string(),
                (true, MemoryKind::Roleplay) => format!("by {}", self.ai_name),
            };
            ui.label(egui::RichText::new(format!("{when} · {who}")).color(MUTED()).size(11.5));
        });
        ui.add_space(4.0);
        ui.label(egui::RichText::new(&m.title).color(TEXT()).strong().size(15.0));
        if !m.text.is_empty() {
            ui.label(egui::RichText::new(&m.text).color(MUTED()).size(12.5));
        }
        if m.image.is_some() {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/image.svg"))
                        .fit_to_exact_size(egui::vec2(14.0, 14.0))
                        .tint(MUTED()),
                );
                ui.label(
                    egui::RichText::new("About a picture in the album (forgetting this deletes it)")
                        .color(MUTED())
                        .size(11.5),
                );
            });
        }
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            if ui
                .button(egui::RichText::new("Ask about this").size(12.5))
                .on_hover_text("Put a question about this memory in the chat box")
                .clicked()
            {
                *ask = Some(match self.kind {
                    // The topic helps the model look up the right memories.
                    MemoryKind::Main => format!("What do you remember about \"{}\" ({})?", m.title, topic.name),
                    MemoryKind::Roleplay => format!("Do you remember this: \"{}\"?", m.title),
                });
            }
            if ui.button(egui::RichText::new("Edit").size(12.5)).clicked() {
                self.draft_title.clone_from(&m.title);
                self.draft_text.clone_from(&m.text);
                self.panel = Panel::Edit;
            }
            if ui.button(egui::RichText::new("Forget").size(12.5)).clicked() {
                self.panel = Panel::ConfirmDelete;
            }
        });
    }

    fn edit(&mut self, ui: &mut egui::Ui, store: &mut MemoryStore) {
        let Some((ti, _)) = self.selected.and_then(|id| find(store, id)) else {
            self.panel = Panel::Overview;
            return;
        };
        caption(ui, "EDIT MEMORY");
        field_label(ui, "Title");
        ui.add(egui::TextEdit::singleline(&mut self.draft_title).desired_width(f32::INFINITY));
        field_label(ui, "Details");
        ui.add(egui::TextEdit::multiline(&mut self.draft_text).desired_rows(5).desired_width(f32::INFINITY));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let ok = !self.draft_title.trim().is_empty();
            if ui.add_enabled(ok, egui::Button::new(egui::RichText::new("Save").size(12.5))).clicked() {
                if let Some(id) = self.selected {
                    store.update(ti, id, &self.draft_title, &self.draft_text);
                }
                self.panel = Panel::View;
            }
            if ui.button(egui::RichText::new("Cancel").size(12.5)).clicked() {
                self.panel = Panel::View;
            }
        });
    }

    fn confirm_delete(&mut self, ui: &mut egui::Ui, store: &mut MemoryStore) {
        let Some((ti, mi)) = self.selected.and_then(|id| find(store, id)) else {
            self.panel = Panel::Overview;
            return;
        };
        let title = store.topics[ti].memories[mi].title.clone();
        caption(ui, "FORGET THIS MEMORY?");
        ui.label(egui::RichText::new(&title).color(TEXT()).strong().size(14.0));
        ui.label(
            egui::RichText::new("The AI won't remember it in future chats. This can't be undone.")
                .color(MUTED())
                .size(12.0),
        );
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            let forget = egui::Button::new(egui::RichText::new("Forget").color(Color32::WHITE).size(12.5))
                .fill(Color32::from_rgb(180, 40, 40));
            if ui.add(forget).clicked() {
                if let Some(id) = self.selected.take() {
                    self.forgotten = store.delete(ti, id);
                }
                self.panel = Panel::Overview;
            }
            if ui.button(egui::RichText::new("Cancel").size(12.5)).clicked() {
                self.panel = Panel::View;
            }
        });
    }

    fn new_form(&mut self, ui: &mut egui::Ui, store: &mut MemoryStore) {
        caption(ui, "NEW MEMORY");
        field_label(ui, "Topic");
        const NEW: &str = "\u{0}new";
        let shown = if self.draft_topic == NEW || self.draft_topic.is_empty() {
            "New topic…".to_string()
        } else {
            self.draft_topic.clone()
        };
        // Role play offers its categories even before they have memories.
        let mut choices: Vec<String> = store.topics.iter().map(|t| t.name.clone()).collect();
        if self.kind == MemoryKind::Roleplay {
            for t in crate::roleplay::TOPICS {
                if store.find_topic(t).is_none() {
                    choices.push(t.to_string());
                }
            }
        }
        egui::ComboBox::from_id_salt("memory_new_topic")
            .width(ui.available_width())
            .selected_text(shown)
            .show_ui(ui, |ui| {
                for name in &choices {
                    ui.selectable_value(&mut self.draft_topic, name.clone(), name);
                }
                ui.selectable_value(&mut self.draft_topic, NEW.to_string(), "New topic…");
            });
        let new_topic = self.draft_topic == NEW || self.draft_topic.is_empty();
        if new_topic {
            ui.add(
                egui::TextEdit::singleline(&mut self.draft_new_topic)
                    .hint_text("Topic name, e.g. a project")
                    .desired_width(f32::INFINITY),
            );
        }
        field_label(ui, "Title");
        ui.add(
            egui::TextEdit::singleline(&mut self.draft_title)
                .hint_text("A short headline")
                .desired_width(f32::INFINITY),
        );
        field_label(ui, "Details");
        ui.add(
            egui::TextEdit::multiline(&mut self.draft_text)
                .hint_text("What should the AI remember?")
                .desired_rows(4)
                .desired_width(f32::INFINITY),
        );
        ui.add_space(6.0);
        let topic = if new_topic { self.draft_new_topic.trim().to_string() } else { self.draft_topic.clone() };
        ui.horizontal(|ui| {
            let ok = !topic.is_empty() && !self.draft_title.trim().is_empty();
            if ui.add_enabled(ok, egui::Button::new(egui::RichText::new("Save").size(12.5))).clicked()
                && let Some((_, id)) = store.add(&topic, &self.draft_title, &self.draft_text)
            {
                self.selected = Some(id);
                self.panel = Panel::View;
            }
            if ui.button(egui::RichText::new("Cancel").size(12.5)).clicked() {
                self.panel = Panel::Overview;
            }
        });
    }
}

/// A small uppercase caption, like the Settings section headers.
fn caption(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(MUTED()).strong().size(11.0));
}

fn field_label(ui: &mut egui::Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(egui::RichText::new(text).color(TEXT()).size(12.5));
}

/// A rounded topic chip: tinted fill, coloured name.
fn chip(ui: &mut egui::Ui, color: Color32, label: &str) {
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(11.5), color);
    let size = galley.size() + egui::vec2(16.0, 6.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::same(10), color.gamma_multiply(0.16));
    ui.painter().galley(rect.min + egui::vec2(8.0, 3.0), galley, color);
}

/// A topic filter pill: colour dot + name; dimmed when switched off.
fn pill(ui: &mut egui::Ui, color: Color32, label: &str, on: bool) -> egui::Response {
    let text_col = if on { TEXT() } else { MUTED() };
    let galley = ui.painter().layout_no_wrap(label.to_string(), egui::FontId::proportional(12.0), text_col);
    let size = egui::vec2(galley.size().x + 30.0, 24.0);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter();
    if on {
        painter.rect_filled(rect, CornerRadius::same(12), color.gamma_multiply(0.14));
    }
    let edge = if on { color.gamma_multiply(0.55) } else { EDGE() };
    painter.rect_stroke(rect, CornerRadius::same(12), Stroke::new(1.0_f32, edge), egui::StrokeKind::Inside);
    painter.circle_filled(egui::pos2(rect.left() + 13.0, rect.center().y), 4.0, if on { color } else { color.gamma_multiply(0.35) });
    painter.galley(egui::pos2(rect.left() + 22.0, rect.center().y - galley.size().y / 2.0), galley, text_col);
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(if on { "Hide this topic" } else { "Show this topic" })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every anchor sits inside the brain, so topics start in the volume.
    #[test]
    fn anchors_are_inside_the_brain() {
        for a in ANCHORS {
            assert!(depth_in_brain(a).is_some(), "anchor {a:?} is outside the brain");
        }
    }

    /// The cloud is the same every run (seeded) and fills its quota.
    #[test]
    fn cloud_is_deterministic() {
        let (a, b) = (build_cloud(), build_cloud());
        assert!(a.haze.len() >= 410, "haze: {}", a.haze.len());
        assert!(a.dots.len() >= 295, "dots: {}", a.dots.len());
        assert_eq!(a.haze[17].0, b.haze[17].0);
    }
}
