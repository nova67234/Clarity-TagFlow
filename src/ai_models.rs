//! AI Model Manager — the "Get Models" dialog.
//!
//! A Rust/egui port of terminus2's `TagDownloader.java`. Presents a catalog of
//! supported tagger models (PixAI, JoyTag, the WD14 family) and downloads their
//! ONNX weights + tag lists from HuggingFace into `tools/<folder>/`. The actual
//! ONNX inference lives elsewhere — this dialog only fetches the files, so it
//! needs no ML runtime, just an HTTP client.
//!
//! Downloads run on a background thread (large files, ~1.3 GB for some models),
//! reporting progress through a shared atomic so the egui frame can paint a
//! live progress bar without blocking.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui::{Color32, RichText};

use crate::theme::*;

const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(220, 70, 70);

/// Where a model's files are written: the shared, writable models root (a
/// per-user data dir) joined with the model's folder.
fn target_dir(folder: &str) -> PathBuf {
    crate::tagger::models_root().join(folder)
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

/// One downloadable model. `files` pairs the saved file name with its URL.
pub struct ModelInfo {
    name: &'static str,
    tab: &'static str,
    folder: &'static str,
    desc: &'static str,
    note: &'static str,
    repo: &'static str,
    /// The tagger family this model belongs to, or `None` for non-tagger models
    /// (e.g. the depth estimator that powers Spatial Scene). `None` keeps the
    /// model out of the tagger dropdown while still listing it in Get Models.
    kind: Option<crate::tagger::TaggerKind>,
    files: &'static [(&'static str, &'static str)],
}

impl ModelInfo {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn folder(&self) -> &'static str {
        self.folder
    }
    pub fn kind(&self) -> Option<crate::tagger::TaggerKind> {
        self.kind
    }
    /// The tag-list file for this model (the catalog entry's non-`model.onnx` file).
    pub fn tags_file(&self) -> &'static str {
        self.files
            .iter()
            .map(|(n, _)| *n)
            .find(|n| *n != "model.onnx")
            .unwrap_or("selected_tags.csv")
    }
}

/// Catalog **tagger** models whose files are present on disk (checked live).
/// Non-tagger models (`kind == None`, e.g. the depth estimator) are excluded so
/// they never appear in the tagger dropdown.
pub fn installed_models() -> Vec<&'static ModelInfo> {
    CATALOG.iter().filter(|m| m.kind.is_some() && check_installed(m)).collect()
}

/// Find a catalog model by its display name.
pub fn find(name: &str) -> Option<&'static ModelInfo> {
    CATALOG.iter().find(|m| m.name == name)
}

const CATALOG: &[ModelInfo] = &[
    ModelInfo {
        name: "PixAI Tagger v0.9",
        tab: "PixAI",
        folder: "pixai-tagger-v0.9",
        desc: "EVA02-based, ~13.4k tags. Finetuned from WD-EVA02 with standout \
               character/series recognition. Recommended.",
        note: "Downloads model.onnx (~1.3 GB) & selected_tags.csv (DeepGHS ONNX export).",
        repo: "https://huggingface.co/pixai-labs/pixai-tagger-v0.9",
        kind: Some(crate::tagger::TaggerKind::Pixai),
        files: &[
            ("model.onnx", "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/model.onnx"),
            ("selected_tags.csv", "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/selected_tags.csv"),
        ],
    },
    ModelInfo {
        name: "JoyTag (SOTA)",
        tab: "JoyTag",
        folder: "joytag",
        desc: "State-of-the-Art (2024). Huge vocabulary (5k+ tags). Best for general use.",
        note: "Downloads model.onnx & top_tags.txt.",
        repo: "https://huggingface.co/fancyfeast/joytag",
        kind: Some(crate::tagger::TaggerKind::JoyTag),
        files: &[
            ("model.onnx", "https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx"),
            ("top_tags.txt", "https://huggingface.co/fancyfeast/joytag/resolve/main/top_tags.txt"),
        ],
    },
    ModelInfo {
        name: "WD14 v3 SwinV2",
        tab: "Swin V3",
        folder: "wd14-swinv2-v3",
        desc: "Highest Accuracy WD14. Slower but very smart.",
        note: "Downloads model.onnx & selected_tags.csv.",
        repo: "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3",
        kind: Some(crate::tagger::TaggerKind::Wd14),
        files: &[
            ("model.onnx", "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx"),
            ("selected_tags.csv", "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv"),
        ],
    },
    ModelInfo {
        name: "WD14 v3 EVA02",
        tab: "EVA02",
        folder: "wd14-eva02-v3",
        desc: "Newer EVA02 architecture. Very high accuracy, similar to SwinV2.",
        note: "Downloads model.onnx & selected_tags.csv.",
        repo: "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
        kind: Some(crate::tagger::TaggerKind::Wd14),
        files: &[
            ("model.onnx", "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx"),
            ("selected_tags.csv", "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv"),
        ],
    },
    ModelInfo {
        name: "WD14 v3 ConvNext",
        tab: "Conv V3",
        folder: "wd14-convnext-v3",
        desc: "Balanced WD14. Faster than Swin, better than v2.",
        note: "Downloads model.onnx & selected_tags.csv.",
        repo: "https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3",
        kind: Some(crate::tagger::TaggerKind::Wd14),
        files: &[
            ("model.onnx", "https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3/resolve/main/model.onnx"),
            ("selected_tags.csv", "https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3/resolve/main/selected_tags.csv"),
        ],
    },
    ModelInfo {
        name: "WD14 v2 ConvNext",
        tab: "WD14 v2",
        folder: "wd14-convnext-v2",
        desc: "Legacy Standard WD14. Fast and lightweight.",
        note: "Downloads model.onnx & selected_tags.csv.",
        repo: "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2",
        kind: Some(crate::tagger::TaggerKind::Wd14),
        files: &[
            ("model.onnx", "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx"),
            ("selected_tags.csv", "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/selected_tags.csv"),
        ],
    },
    ModelInfo {
        name: "Depth Anything V2 Base",
        tab: "Depth",
        folder: "depth-anything-v2-base-onnx",
        desc: "Monocular depth estimation (Base — sharper than Small). Powers the \
               Spatial Scene 3D parallax viewer (right-click an image → Spatial \
               Scene). Not a tagger.",
        note: "Downloads model.onnx + weights (~370 MB).",
        repo: "https://huggingface.co/onnx-community/depth-anything-v2-base-ONNX",
        kind: None,
        // The onnx-community export stores weights in an external-data sidecar
        // (`model.onnx` graph + `model.onnx_data` weights); both must land in the
        // same folder so ONNX Runtime can resolve the weights.
        files: &[
            ("model.onnx", "https://huggingface.co/onnx-community/depth-anything-v2-base-ONNX/resolve/main/onnx/model.onnx"),
            ("model.onnx_data", "https://huggingface.co/onnx-community/depth-anything-v2-base-ONNX/resolve/main/onnx/model.onnx_data"),
        ],
    },
    ModelInfo {
        name: "BiRefNet Lite (background removal)",
        tab: "Background",
        folder: "birefnet-lite-onnx",
        desc: "State-of-the-art background removal. Right-click an image → Remove \
               Background to save a transparent-PNG cutout of the subject. Not a tagger.",
        note: "Downloads model.onnx (~224 MB).",
        repo: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX",
        kind: None,
        // Self-contained export (no external weights sidecar).
        files: &[
            ("model.onnx", "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx"),
        ],
    },
    ModelInfo {
        name: "Gemma 4 E4B (local AI, vision)",
        tab: "Gemma 4",
        folder: crate::llm::FOLDER,
        desc: "Google's Gemma 4 vision language model, run fully inside the app \
               by the AI Model tab in Settings — describe images, answer \
               questions, no server or account. Not a tagger.",
        note: "Downloads the Q4_K_M weights (~5 GB) + vision projector (~1 GB).",
        repo: "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF",
        kind: None,
        // Two GGUFs: the instruct weights and the mmproj vision projector that
        // llama.cpp's multimodal pipeline uses to encode images (see src/llm.rs).
        files: &[
            (crate::llm::MODEL_FILE, "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"),
            (crate::llm::MMPROJ_FILE, "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf"),
        ],
    },
    ModelInfo {
        name: "Region Detection (faces / hands / people / feet / age)",
        tab: "Detect",
        folder: "region-detect",
        desc: "Detectors that draw labelled boxes over the image — faces, hands, \
               people, and feet, trained on both anime and photos (DeepGHS) — plus \
               age estimation on detected faces (InsightFace). Right-click an \
               image → Detect Regions / Detect Age. Not a tagger.",
        note: "Downloads five small ONNX models (~74 MB total).",
        repo: "https://huggingface.co/deepghs",
        kind: None,
        // One folder holds all the detectors (see src/detect.rs).
        files: &[
            ("face.onnx", "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx"),
            ("hand.onnx", "https://huggingface.co/deepghs/anime_hand_detection/resolve/main/hand_detect_v0.8_s/model.onnx"),
            // v1.3 of the anime person detector (the real-photo yolo-person misses
            // anime/illustration people entirely; this one handles both).
            ("person.onnx", "https://huggingface.co/deepghs/anime_person_detection/resolve/main/person_detect_v1.3_s/model.onnx"),
            // NudeNet 320n — used ONLY for its two feet classes (src/detect.rs
            // filters the rest out).
            ("feet.onnx", "https://huggingface.co/deepghs/nudenet_onnx/resolve/main/320n.onnx"),
            // InsightFace buffalo_l genderage — age estimation on face crops.
            ("age.onnx", "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx"),
        ],
    },
];

/// True when every file for a model is present in any of the searched model
/// roots (data dir, local `tools/`, or terminus2 resources).
fn check_installed(info: &ModelInfo) -> bool {
    info.files
        .iter()
        .all(|(name, _)| crate::tagger::resolve(info.folder, name).is_some())
}

// ---------------------------------------------------------------------------
// Download worker state
// ---------------------------------------------------------------------------

/// Shared between the UI thread (reads) and the download thread (writes).
#[derive(Default)]
struct Progress {
    pct: AtomicU32,            // 0..100
    label: Mutex<String>,      // e.g. "Downloading model.onnx…"
    done: AtomicBool,
    ok: AtomicBool,
    err: Mutex<Option<String>>,
}

impl Progress {
    fn set_label(&self, s: String) {
        *self.label.lock().unwrap() = s;
    }
    fn finish_ok(&self) {
        self.pct.store(100, Relaxed);
        self.ok.store(true, Relaxed);
        self.done.store(true, Relaxed);
    }
    fn finish_err(&self, e: String) {
        *self.err.lock().unwrap() = Some(e);
        self.done.store(true, Relaxed);
    }
}

#[derive(Clone)]
enum Status {
    NotInstalled,
    Installed,
    Downloading,
    Error(String),
}

struct Entry {
    info: &'static ModelInfo,
    status: Status,
    progress: Option<Arc<Progress>>,
}

// ---------------------------------------------------------------------------
// Public manager
// ---------------------------------------------------------------------------

pub struct ModelManager {
    pub open: bool,
    entries: Vec<Entry>,
    selected: usize,
    close_requested: bool,
}

impl Default for ModelManager {
    fn default() -> Self {
        let entries = CATALOG
            .iter()
            .map(|info| Entry {
                info,
                status: if check_installed(info) {
                    Status::Installed
                } else {
                    Status::NotInstalled
                },
                progress: None,
            })
            .collect();
        Self { open: false, entries, selected: 0, close_requested: false }
    }
}

impl ModelManager {
    /// Toggle the dropdown open/closed (driven by the "Get Models" button click).
    pub fn toggle(&mut self) {
        self.open = !self.open;
    }

    /// Drive the manager each frame: poll active downloads and, when open, draw
    /// the dropdown anchored under `anchor` (the "Get Models" button). Call every
    /// frame regardless of `open` so downloads still finalize while it's closed.
    pub fn show(&mut self, anchor: &egui::Response) {
        self.poll();

        if self.open {
            let mut open = self.open;
            egui::Popup::from_response(anchor)
                .open_bool(&mut open)
                .align(egui::RectAlign::BOTTOM_START) // drop down under the button
                .width(400.0)
                .gap(6.0)
                .frame(crate::card_frame(22))
                .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
                .show(|ui| self.body(ui));
            // Closed by clicking outside, the in-body Close button, or re-clicking.
            self.open = open && !self.close_requested;
            self.close_requested = false;
        }

        // Keep repainting while any download is in flight so the bar animates.
        if self.entries.iter().any(|e| e.progress.is_some()) {
            anchor.ctx.request_repaint();
        }
    }

    /// Promote finished downloads to Installed / Error and drop their handles.
    fn poll(&mut self) {
        for e in &mut self.entries {
            let finished = e.progress.as_ref().is_some_and(|p| p.done.load(Relaxed));
            if !finished {
                continue;
            }
            let p = e.progress.take().unwrap();
            e.status = if p.ok.load(Relaxed) {
                Status::Installed
            } else {
                let msg = p.err.lock().unwrap().clone().unwrap_or_else(|| "Download failed".into());
                Status::Error(msg)
            };
        }
    }

    fn body(&mut self, ui: &mut egui::Ui) {
        // No title bar on a popup — render our own header.
        ui.label(RichText::new("AI Model Manager").color(TEXT()).strong().size(14.0));
        ui.add_space(1.0);
        ui.label(RichText::new("Download high-performance tagging models.").color(MUTED()).size(11.0));
        ui.add_space(8.0);

        // Tab pills.
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(5.0, 5.0);
            for i in 0..self.entries.len() {
                let selected = self.selected == i;
                if tab_pill(ui, self.entries[i].info.tab, selected).clicked() {
                    self.selected = i;
                }
            }
        });
        ui.add_space(8.0);

        let idx = self.selected;
        let info = self.entries[idx].info; // &'static, cheap Copy of the reference

        // Model card.
        egui::Frame::new()
            .fill(FIELD())
            .corner_radius(egui::CornerRadius::same(22))
            .stroke(egui::Stroke::new(1.0, EDGE()))
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());

                ui.label(RichText::new(info.name).color(TEXT()).strong().size(14.5));
                ui.add_space(3.0);
                ui.label(RichText::new(info.desc).color(MUTED()).size(11.5));
                ui.add_space(5.0);
                ui.hyperlink_to(RichText::new("Visit Repository  ↗").color(ACCENT1()).size(11.5), info.repo);

                ui.add_space(9.0);

                // Status badge.
                match &self.entries[idx].status {
                    Status::Installed => {
                        badge(ui, GREEN, "Installed", Some(egui::include_image!("../icons/check.svg")))
                    }
                    Status::NotInstalled => badge(ui, RED, "Not Installed", None),
                    Status::Downloading => badge(ui, ACCENT1(), "Downloading…", None),
                    Status::Error(e) => badge(ui, RED, &format!("Error: {e}"), None),
                }

                // Live progress bar while downloading.
                if let Some(p) = &self.entries[idx].progress {
                    let frac = p.pct.load(Relaxed) as f32 / 100.0;
                    let label = p.label.lock().unwrap().clone();
                    ui.add_space(8.0);
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .desired_width(ui.available_width())
                            .text(RichText::new(label).color(TEXT()).size(10.5)),
                    );
                }

                // Action button. Installed models show no button (the badge above
                // already conveys their state). The button hugs its label (height
                // pinned, width auto) so the text stays centred.
                match self.entries[idx].status {
                    Status::Installed => {}
                    Status::Downloading => {
                        ui.add_space(10.0);
                        let btn = egui::Button::new(RichText::new("Downloading…").color(Color32::WHITE).strong())
                            .fill(ACCENT1().gamma_multiply(0.5))
                            .corner_radius(egui::CornerRadius::same(10))
                            .min_size(egui::vec2(0.0, 28.0));
                        ui.add_enabled(false, btn);
                    }
                    _ => {
                        ui.add_space(10.0);
                        let btn = egui::Button::new(RichText::new("Download").color(Color32::WHITE).strong())
                            .fill(ACCENT1())
                            .corner_radius(egui::CornerRadius::same(10))
                            .min_size(egui::vec2(0.0, 28.0));
                        if ui.add(btn).clicked() {
                            self.start_download(idx);
                        }
                    }
                }

                ui.add_space(6.0);
                ui.label(RichText::new(info.note).color(MUTED()).italics().size(10.0));
            });

        ui.add_space(8.0);

        // Footer: Close button on the right. The row height is pinned via
        // allocate_ui_with_layout — a bare right_to_left layout would otherwise
        // anchor to the right and stretch to fill the (screen-tall) auto_sized
        // region, leaving a large dead space below the window.
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), 28.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                let close = egui::Button::new(RichText::new("Close").color(TEXT()))
                    .corner_radius(egui::CornerRadius::same(10))
                    .min_size(egui::vec2(78.0, 26.0));
                if ui.add(close).clicked() {
                    self.close_requested = true;
                }
            },
        );
    }

    fn start_download(&mut self, idx: usize) {
        let info = self.entries[idx].info;
        let shared = Arc::new(Progress::default());
        shared.set_label("Connecting…".to_string());
        self.entries[idx].progress = Some(shared.clone());
        self.entries[idx].status = Status::Downloading;

        let files: Vec<(String, String)> =
            info.files.iter().map(|(n, u)| (n.to_string(), u.to_string())).collect();
        let folder = info.folder.to_string();

        std::thread::spawn(move || {
            if let Err(e) = download_all(&files, &folder, &shared) {
                shared.finish_err(e);
            } else {
                shared.finish_ok();
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Headless download — lets the Spatial Scene viewer fetch the depth model on
// first use, without opening the Model Manager dropdown. Reuses the same
// streaming downloader (`download_all`) and `Progress` as the manager.
// ---------------------------------------------------------------------------

/// A pollable handle to a background model download started by
/// [`start_model_download`]. Poll it each frame; it never blocks.
pub struct DownloadHandle {
    progress: Arc<Progress>,
}

impl DownloadHandle {
    /// Overall progress across all of the model's files, 0..100.
    pub fn pct(&self) -> u32 {
        self.progress.pct.load(Relaxed)
    }
    /// True once the download has finished (successfully or not).
    pub fn done(&self) -> bool {
        self.progress.done.load(Relaxed)
    }
    /// True if it finished successfully (all files present).
    pub fn ok(&self) -> bool {
        self.progress.ok.load(Relaxed)
    }
    /// The error message if it failed.
    pub fn error(&self) -> Option<String> {
        self.progress.err.lock().unwrap().clone()
    }
}

/// Start downloading the catalog model in `folder` on a background thread,
/// returning a handle to poll. `None` if no catalog entry has that folder.
pub fn start_model_download(folder: &str) -> Option<DownloadHandle> {
    let info = CATALOG.iter().find(|m| m.folder == folder)?;
    let shared = Arc::new(Progress::default());
    shared.set_label("Connecting…".to_string());
    let files: Vec<(String, String)> =
        info.files.iter().map(|(n, u)| (n.to_string(), u.to_string())).collect();
    let folder = info.folder.to_string();
    let worker = shared.clone();
    std::thread::spawn(move || {
        if let Err(e) = download_all(&files, &folder, &worker) {
            worker.finish_err(e);
        } else {
            worker.finish_ok();
        }
    });
    Some(DownloadHandle { progress: shared })
}

/// A small rounded status pill: translucent tinted fill with matching text,
/// optionally preceded by a (tinted) icon.
fn badge(ui: &mut egui::Ui, color: Color32, text: &str, icon: Option<egui::ImageSource<'_>>) {
    egui::Frame::new()
        .fill(color.gamma_multiply(0.16))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(10, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if let Some(src) = icon {
                    ui.add(egui::Image::new(src).fit_to_exact_size(egui::vec2(13.0, 13.0)).tint(color));
                }
                ui.label(RichText::new(text).color(color).strong().size(12.0));
            });
        });
}

/// A rounded tab pill — accent-tinted when active, muted text otherwise.
fn tab_pill(ui: &mut egui::Ui, text: &str, selected: bool) -> egui::Response {
    let (fill, fg) = if selected {
        (ACCENT1().gamma_multiply(0.30), TEXT())
    } else {
        (Color32::TRANSPARENT, MUTED())
    };
    ui.add(
        egui::Button::new(RichText::new(text).color(fg).size(11.0))
            .fill(fill)
            .corner_radius(egui::CornerRadius::same(9))
            .min_size(egui::vec2(58.0, 24.0)),
    )
}

/// Fetch every file for a model into `tools/<folder>/`, streaming each to a
/// `.part` temp first and renaming on success so partial files never look
/// installed. Updates `prog` as bytes arrive.
fn download_all(files: &[(String, String)], folder: &str, prog: &Progress) -> Result<(), String> {
    let dir = target_dir(folder);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Create {}: {e}", dir.display()))?;

    // native-tls => Windows SChannel; follow HuggingFace's redirect to the CDN.
    // ureq 3.x defaults to rustls even with the native-tls feature on, so the
    // provider must be selected explicitly (avoids rustls/ring needing nasm on
    // MSVC — see Cargo.toml). The connector is managed by ureq internally now.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                // Validate against the OS cert store (with AIA intermediate
                // fetching) rather than ureq's bundled webpki roots — see
                // civitai.rs for the CDN incomplete-chain failure this avoids.
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .build()
        .into();

    let n = files.len() as f32;
    for (i, (name, url)) in files.iter().enumerate() {
        prog.set_label(format!("Downloading {name}…"));

        let resp = agent.get(url).call().map_err(|e| format!("{name}: {e}"))?;
        let total_len: u64 = resp
            .headers()
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let tmp = dir.join(format!("{name}.part"));
        let mut out = std::fs::File::create(&tmp).map_err(|e| format!("{name}: {e}"))?;
        let mut reader = resp.into_body().into_reader();
        let mut buf = vec![0u8; 1 << 16];
        let mut got: u64 = 0;
        loop {
            let read = reader.read(&mut buf).map_err(|e| format!("{name}: {e}"))?;
            if read == 0 {
                break;
            }
            out.write_all(&buf[..read]).map_err(|e| format!("{name}: {e}"))?;
            got += read as u64;
            if total_len > 0 {
                // Each file owns an equal 1/n slice of the overall bar.
                let frac = (i as f32 + got as f32 / total_len as f32) / n;
                prog.pct.store((frac * 100.0) as u32, Relaxed);
            }
        }
        out.flush().ok();
        drop(out);
        std::fs::rename(&tmp, dir.join(name)).map_err(|e| format!("Save {name}: {e}"))?;
    }
    Ok(())
}