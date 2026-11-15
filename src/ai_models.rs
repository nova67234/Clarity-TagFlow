//! AI Model Manager — the "Models" tab of the Tag Manager settings popup.
//!
//! A Rust/egui port of terminus2's `TagDownloader.java`. Presents a catalog of
//! supported tagger models (PixAI, JoyTag, the WD14 family) and downloads their
//! ONNX weights + tag lists from HuggingFace into `tools/<folder>/`. The actual
//! ONNX inference lives elsewhere — this dialog only fetches the files, so it
//! needs no ML runtime, just an HTTP client.
//!
//! The tab lists every model, in three groups: its name over one line about
//! it and, at the row's end, where it stands — as on the Civitai browser's
//! cards: a download arrow, the percentage while it downloads, a green tick
//! once it's installed, an orange warning if it failed (a click tries again).
//!
//! Downloads run on a background thread (large files, ~1.3 GB for some models),
//! reporting progress through a shared atomic so the egui frame can paint the
//! percentage without blocking.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use eframe::egui;
use egui::Color32;

use crate::theme::*;

/// A row's slot once the model is installed, and after a download failed
/// (the Civitai browser's green and orange).
const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const ORANGE: Color32 = Color32::from_rgb(235, 150, 45);

/// Where a model's files are written: the shared, writable models root (a
/// per-user data dir) joined with the model's folder.
fn target_dir(folder: &str) -> PathBuf {
    crate::tagger::models_root().join(folder)
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

/// The Models tab's groups, in the order it lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Group {
    Taggers,
    LocalAi,
    Tools,
}

impl Group {
    const ALL: [Group; 3] = [Group::Taggers, Group::LocalAi, Group::Tools];

    fn title(self) -> &'static str {
        match self {
            Group::Taggers => "Taggers",
            Group::LocalAi => "Local AI",
            Group::Tools => "Tools",
        }
    }
}

/// The tagger the Models tab marks "Recommended" (its `desc` says why).
const RECOMMENDED: &str = "pixai-tagger-v0.9";

/// One downloadable model. `files` pairs the saved file name with its URL.
pub struct ModelInfo {
    name: &'static str,
    /// Where the Models tab lists it.
    group: Group,
    /// The one line under its name there: what it's for, after the size of
    /// the download where the app states one.
    caption: &'static str,
    folder: &'static str,
    /// The longer account of it, and what a download fetches: the row's
    /// hover text.
    desc: &'static str,
    note: &'static str,
    repo: &'static str,
    /// The tagger family this model belongs to, or `None` for non-tagger models
    /// (e.g. the depth estimator that powers Spatial Scene). `None` keeps the
    /// model out of the tagger dropdown while still listing it in Models.
    kind: Option<crate::tagger::TaggerKind>,
    files: &'static [(&'static str, &'static str)],
}

impl ModelInfo {
    pub fn name(&self) -> &'static str {
        self.name
    }
    /// The name as the Models tab shows it: without what's in brackets after
    /// it (the row's caption says that).
    fn title(&self) -> &'static str {
        self.name.split(" (").next().unwrap_or(self.name)
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
        group: Group::Taggers,
        caption: "1.3 GB · 13.4k tags, best at characters",
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
        group: Group::Taggers,
        caption: "5k+ tags · best for general use",
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
        group: Group::Taggers,
        caption: "The most accurate WD14, a little slower",
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
        group: Group::Taggers,
        caption: "Newer design, as accurate as SwinV2",
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
        group: Group::Taggers,
        caption: "Balanced: faster than SwinV2",
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
        group: Group::Taggers,
        caption: "The older standard, fast and light",
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
        group: Group::Tools,
        caption: "370 MB · depth for Spatial Scene",
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
        group: Group::Tools,
        caption: "224 MB · background removal",
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
        group: Group::LocalAi,
        caption: "6 GB · fast, fits 8 GB graphics cards",
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
        name: "Gemma 3 27B (local AI, vision)",
        group: Group::LocalAi,
        caption: "16 GB · very strong, wants 20 GB of VRAM",
        folder: crate::llm::FOLDER_27B,
        desc: "The previous-generation flagship Gemma — much smarter than E4B; \
               wants ~20 GB of VRAM (spills to RAM otherwise). Selectable in \
               Settings → AI Model. Not a tagger.",
        note: "Downloads the Q4_K_M weights (~15.5 GB) + vision projector (~0.8 GB).",
        repo: "https://huggingface.co/unsloth/gemma-3-27b-it-GGUF",
        kind: None,
        files: &[
            (crate::llm::MODEL_FILE_27B, "https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q4_K_M.gguf"),
            (crate::llm::MMPROJ_FILE, "https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/mmproj-F16.gguf"),
        ],
    },
    ModelInfo {
        name: "Gemma 4 31B (local AI, vision)",
        group: Group::LocalAi,
        caption: "19.5 GB · the strongest",
        folder: crate::llm::FOLDER_31B,
        desc: "The strongest Gemma 4 (dense). On a 16 GB GPU it loads partly \
               on the CPU — slower, but the highest quality. Selectable in \
               Settings → AI Model. Not a tagger.",
        note: "Downloads the Q4_K_M weights (~18.5 GB) + vision projector (~1.2 GB).",
        repo: "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF",
        kind: None,
        files: &[
            (crate::llm::MODEL_FILE_31B, "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf"),
            (crate::llm::MMPROJ_FILE, "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/mmproj-F16.gguf"),
        ],
    },
    ModelInfo {
        name: "Qwen3-VL 8B (local AI, vision)",
        group: Group::LocalAi,
        caption: "6 GB · a different flavour to Gemma",
        folder: crate::llm::FOLDER_QWEN,
        desc: "Alibaba's Qwen3-VL vision language model — about E4B's size and \
               speed, with a different flavour to Gemma. Selectable in \
               Settings → AI Model. Not a tagger.",
        note: "Downloads the Q4_K_M weights (~4.7 GB) + vision projector (~1.1 GB).",
        repo: "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF",
        kind: None,
        files: &[
            (crate::llm::MODEL_FILE_QWEN, "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"),
            (crate::llm::MMPROJ_FILE, "https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF/resolve/main/mmproj-F16.gguf"),
        ],
    },
    ModelInfo {
        name: "Qwen3-VL 30B Thinking (local AI, vision)",
        group: Group::LocalAi,
        caption: "18.6 GB · reasons before it answers",
        folder: crate::llm::FOLDER_QWEN_30B,
        desc: "The big Qwen3-VL — reasons in a thinking block before answering \
               (shown behind the chat's collapsible Thinking row). A \
               mixture-of-experts with only 3B active, so it stays quick even \
               spilling to RAM. Selectable in Settings → AI Model. Not a tagger.",
        note: "Downloads the Q4_K_M weights (~18.6 GB) + vision projector (~1.1 GB).",
        repo: "https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF",
        kind: None,
        files: &[
            (crate::llm::MODEL_FILE_QWEN_30B, "https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/resolve/main/Qwen3-VL-30B-A3B-Thinking-Q4_K_M.gguf"),
            (crate::llm::MMPROJ_FILE, "https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/resolve/main/mmproj-F16.gguf"),
        ],
    },
    ModelInfo {
        name: "Region Detection (faces / hands / people / feet / age)",
        group: Group::Tools,
        caption: "74 MB · finds faces, hands, people and feet",
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
    ModelInfo {
        name: "Whisper Base (Speech to Text)",
        group: Group::Tools,
        caption: "148 MB · speech to text for the chat",
        folder: crate::dictate::FOLDER,
        desc: "OpenAI's Whisper base model — powers the AI chat's mic button. \
               Multilingual dictation that runs fully locally on the CPU.",
        note: "Downloads ggml-base.bin (~148 MB). Also fetched automatically \
               the first time the mic button is used.",
        repo: "https://huggingface.co/ggerganov/whisper.cpp",
        kind: None,
        files: &[(
            crate::dictate::MODEL_FILE,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        )],
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

/// Where a model stands — what its row's slot shows.
#[derive(Clone, Debug, PartialEq)]
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

/// What the manager asks of the world outside it (the tests put their own
/// answers in its place).
#[derive(Clone, Copy)]
struct Backend {
    /// Whether all of a model's files are on disk.
    installed: fn(&ModelInfo) -> bool,
    /// Start the download of a model's files, or join the one running.
    fetch: fn(&'static ModelInfo) -> Arc<Progress>,
    /// The downloads in flight right now, whoever started them.
    running: fn() -> Vec<(&'static str, Arc<Progress>)>,
}

impl Backend {
    const REAL: Backend = Backend { installed: check_installed, fetch: spawn_download, running: in_flight };
}

// ---------------------------------------------------------------------------
// Public manager
// ---------------------------------------------------------------------------

pub struct ModelManager {
    entries: Vec<Entry>,
    backend: Backend,
    /// The frame the list was last drawn in: back in view after a while out
    /// of it, it looks again at what's on disk.
    drawn: Option<u64>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::with(Backend::REAL)
    }
}

/// The list's rows: this tall, their text this far in from the card's edge,
/// and at their end the slot — an icon, or up to "100%" — with the same
/// margin after it.
const ROW_H: f32 = 54.0;
const ROW_PAD: f32 = 16.0;
const SLOT_W: f32 = 52.0;
const ICON: f32 = 20.0;
/// Between one group's card and the next group's heading.
const GROUP_GAP: f32 = 18.0;

impl ModelManager {
    fn with(backend: Backend) -> Self {
        let entries = CATALOG
            .iter()
            .map(|info| Entry {
                info,
                status: if (backend.installed)(info) { Status::Installed } else { Status::NotInstalled },
                progress: None,
            })
            .collect();
        Self { entries, backend, drawn: None }
    }

    /// Drive the manager each frame, whether or not its UI is visible: poll
    /// active downloads so they finalize, and keep repainting while any is in
    /// flight so the percentages keep up.
    pub fn tick(&mut self, ctx: &egui::Context) {
        self.poll();
        if self.entries.iter().any(|e| e.progress.is_some()) {
            ctx.request_repaint();
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

    /// Catch up with what happened outside the list. A download something
    /// else started (Settings' AI Model tab, the first use of the mic or of
    /// Spatial Scene) is followed here, so its row shows the percentage
    /// rather than offering a second download. And when the list comes back
    /// into view (`frame` isn't the one after the last it was drawn in),
    /// each model is looked for on disk again: one that arrived meanwhile is
    /// installed, one whose files have gone is not.
    fn sync(&mut self, frame: u64) {
        let back = self.drawn.is_none_or(|last| frame > last + 1);
        self.drawn = Some(frame);
        let running = (self.backend.running)();
        for e in &mut self.entries {
            if e.progress.is_some() {
                continue;
            }
            if let Some((_, p)) = running.iter().find(|(folder, _)| *folder == e.info.folder) {
                e.progress = Some(p.clone());
                e.status = Status::Downloading;
            } else if back {
                let on_disk = (self.backend.installed)(e.info);
                // A failed download keeps its warning until it's tried again
                // — or the files are there after all.
                if on_disk || !matches!(e.status, Status::Error(_)) {
                    e.status = if on_disk { Status::Installed } else { Status::NotInstalled };
                }
            }
        }
    }

    /// Draw the model list — the body of the Tag Manager settings popup's
    /// "Models" tab: a heading and a card for each group, a row for each of
    /// its models. Host must call [`Self::tick`] every frame.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        use crate::tag_manager_settings::{CARD_RADIUS, caps_label, card_fill};
        self.sync(ui.ctx().cumulative_frame_nr());
        ui.spacing_mut().item_spacing.y = 0.0;
        let mut first = true;
        for group in Group::ALL {
            let rows: Vec<usize> = (0..self.entries.len()).filter(|&i| self.entries[i].info.group == group).collect();
            if rows.is_empty() {
                continue;
            }
            if !std::mem::take(&mut first) {
                ui.add_space(GROUP_GAP);
            }
            caps_label(ui, group.title());
            ui.add_space(8.0);

            // The group's card: its rows, a hairline between each two, and
            // the card's outline round them.
            let line = egui::Stroke::new(1.0_f32, crate::settings::hairline());
            let height = rows.len() as f32 * ROW_H + (rows.len() - 1) as f32 + 2.0;
            let (card, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
            if ui.is_rect_visible(card) {
                ui.painter().rect(card, CARD_RADIUS, card_fill(), line, egui::StrokeKind::Inside);
            }
            for (k, &idx) in rows.iter().enumerate() {
                let top = card.top() + 1.0 + k as f32 * (ROW_H + 1.0);
                let row = egui::Rect::from_min_size(
                    egui::pos2(card.left() + 1.0, top),
                    egui::vec2(card.width() - 2.0, ROW_H),
                );
                if k > 0 && ui.is_rect_visible(row) {
                    ui.painter().hline((row.left() + ROW_PAD)..=(row.right() - ROW_PAD), top - 0.5, line);
                }
                // The first and the last row follow the card's corners.
                let round = |on: bool| if on { CARD_RADIUS - 1 } else { 0 };
                let (head, foot) = (round(k == 0), round(k + 1 == rows.len()));
                let corners = egui::CornerRadius { nw: head, ne: head, sw: foot, se: foot };
                self.row(ui, row, idx, corners);
            }
        }
    }

    /// One model's row at `rect`: its name over its caption — a click there
    /// opens the model's page, as a click on a Civitai card does — and its
    /// slot at the end: the download arrow, the percentage on the way, a
    /// green tick once it's installed, an orange warning if it failed. The
    /// arrow and the warning take the click that starts the download.
    fn row(&mut self, ui: &mut egui::Ui, rect: egui::Rect, idx: usize, corners: egui::CornerRadius) {
        use crate::tag_manager::{bold, medium, semibold};
        let info = self.entries[idx].info;
        let id = ui.id().with(info.folder);
        let slot = egui::Rect::from_min_max(egui::pos2(rect.right() - SLOT_W, rect.top()), rect.max);
        let body = egui::Rect::from_min_max(rect.min, egui::pos2(slot.left(), rect.bottom()));

        // What the tab's cards used to spell out is the row's hover text.
        let page = ui
            .interact(body, id.with("page"), egui::Sense::click())
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(format!("{}\n{}\nClick to open its page.", info.desc, info.note));
        page.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Link, true, info.title()));
        if page.clicked() {
            ui.ctx().open_url(egui::OpenUrl::new_tab(info.repo));
        }

        // The slot: what a hover says, and whether a click starts the download.
        let status = self.entries[idx].status.clone();
        let (pct, doing) = match &self.entries[idx].progress {
            Some(p) => (p.pct.load(Relaxed).min(100), p.label.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            None => (0, String::new()),
        };
        let (starts, tip) = match &status {
            Status::NotInstalled => (true, "Download".to_string()),
            Status::Downloading => (false, doing),
            Status::Installed => (false, "Installed".to_string()),
            Status::Error(e) => (true, format!("{e}\nClick to try again.")),
        };
        let mut get = ui.interact(slot, id.with("get"), if starts { egui::Sense::click() } else { egui::Sense::hover() });
        if !tip.is_empty() {
            get = get.on_hover_text(tip);
        }
        if starts {
            get = get.on_hover_cursor(egui::CursorIcon::PointingHand);
            get.widget_info(|| {
                egui::WidgetInfo::labeled(egui::WidgetType::Button, true, format!("Download {}", info.title()))
            });
        }
        if get.clicked() {
            self.start_download(idx);
        }

        if !ui.is_rect_visible(rect) {
            return;
        }
        if page.hovered() || page.has_focus() {
            ui.painter().rect_filled(rect, corners, TEXT().gamma_multiply(0.04));
        }

        // The name ("Recommended" after the one that is) over the caption,
        // each on one line of the room before the slot.
        let left = rect.left() + ROW_PAD;
        let room = (body.right() - left).max(0.0);
        let (name, caption) = ui.fonts_mut(|f| {
            let face = |font_id: egui::FontId, color: Color32| egui::TextFormat { font_id, color, ..Default::default() };
            let mut name = egui::text::LayoutJob { break_on_newline: false, ..Default::default() };
            name.append(info.title(), 0.0, face(semibold(13.5), TEXT()));
            if info.folder == RECOMMENDED {
                name.append("Recommended", 8.0, face(semibold(11.5), ACCENT1()));
            }
            name.wrap = egui::text::TextWrapping::truncate_at_width(room);
            let mut caption = egui::text::LayoutJob::simple_singleline(info.caption.to_string(), medium(12.0), MUTED());
            caption.wrap = egui::text::TextWrapping::truncate_at_width(room);
            (f.layout_job(name), f.layout_job(caption))
        });
        // A line of 18 points and one of 16, 2 apart, centred on the row.
        let top = rect.center().y - 18.0;
        ui.painter().galley(egui::pos2(left, top + (18.0 - name.size().y) / 2.0), name, TEXT());
        ui.painter().galley(egui::pos2(left, top + 20.0 + (16.0 - caption.size().y) / 2.0), caption, MUTED());

        // The slot's icon ends where the percentage does.
        let end = rect.right() - ROW_PAD;
        let icon_at = egui::Rect::from_center_size(egui::pos2(end - ICON / 2.0, rect.center().y), egui::Vec2::splat(ICON));
        let icon = |ui: &egui::Ui, image: egui::ImageSource<'static>, colour: Color32| {
            // What a click reaches lights up under the pointer.
            if starts && (get.hovered() || get.has_focus()) {
                ui.painter().circle_filled(icon_at.center(), ICON / 2.0 + 6.0, colour.gamma_multiply(0.16));
            }
            egui::Image::new(image).tint(colour).paint_at(ui, icon_at);
        };
        match status {
            Status::NotInstalled => icon(ui, egui::include_image!("../icons/arrow_circle_down.svg"), ACCENT1()),
            Status::Downloading => {
                let at = egui::pos2(end, rect.center().y);
                ui.painter().text(at, egui::Align2::RIGHT_CENTER, format!("{pct}%"), bold(13.0), ACCENT1());
            }
            Status::Installed => icon(ui, egui::include_image!("../icons/checkmark.svg"), GREEN),
            Status::Error(_) => icon(ui, egui::include_image!("../icons/warning.svg"), ORANGE),
        }
    }

    fn start_download(&mut self, idx: usize) {
        self.entries[idx].progress = Some((self.backend.fetch)(self.entries[idx].info));
        self.entries[idx].status = Status::Downloading;
    }
}

/// Model downloads in flight, process-wide, by model folder. The Settings
/// Download button, this tab and the first-use downloads (the mic, Spatial
/// Scene, …) don't know about each other, and two transfers of one file
/// write the same `.part` and race to rename it — which can leave a corrupt
/// model that counts as installed. A second request joins the running
/// download (and shows its progress) instead.
static IN_FLIGHT: Mutex<Vec<(&'static str, Arc<Progress>)>> = Mutex::new(Vec::new());

/// The running download of `folder`, or a fresh one registered for it
/// (`true`: the caller starts it).
fn claim(folder: &'static str) -> (Arc<Progress>, bool) {
    let mut running = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = running.iter().find(|(f, _)| *f == folder) {
        return (p.clone(), false);
    }
    let p = Arc::new(Progress::default());
    p.set_label("Connecting…".to_string());
    running.push((folder, p.clone()));
    (p, true)
}

fn release(folder: &str) {
    IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).retain(|(f, _)| *f != folder);
}

/// The downloads in flight right now, by model folder.
fn in_flight() -> Vec<(&'static str, Arc<Progress>)> {
    IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Download `info`'s files on a background thread (or join the download of
/// them already running), returning the progress to poll.
fn spawn_download(info: &'static ModelInfo) -> Arc<Progress> {
    let (shared, start) = claim(info.folder);
    if start {
        let files: Vec<(String, String)> =
            info.files.iter().map(|(n, u)| (n.to_string(), u.to_string())).collect();
        let folder = info.folder;
        let worker = shared.clone();
        std::thread::spawn(move || {
            let res = download_all(&files, folder, &worker);
            // Off the list before it reports done, so a retry after a
            // failure starts a new download.
            release(folder);
            match res {
                Ok(()) => worker.finish_ok(),
                Err(e) => worker.finish_err(e),
            }
        });
    }
    shared
}

// ---------------------------------------------------------------------------
// Headless download — lets the Spatial Scene viewer fetch the depth model on
// first use, without opening the Models tab. Reuses the same
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

/// Start downloading the catalog model in `folder` on a background thread
/// (or join its download already running), returning a handle to poll.
/// `None` if no catalog entry has that folder.
pub fn start_model_download(folder: &str) -> Option<DownloadHandle> {
    let info = CATALOG.iter().find(|m| m.folder == folder)?;
    Some(DownloadHandle { progress: spawn_download(info) })
}

/// Fetch every file for a model into `tools/<folder>/` through the shared
/// resumable downloader (net.rs: `.part` temp, retry with backoff, Range
/// resume). Updates `prog` as bytes arrive.
fn download_all(files: &[(String, String)], folder: &str, prog: &Progress) -> Result<(), String> {
    let dir = target_dir(folder);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Create {}: {e}", dir.display()))?;

    let n = files.len() as f32;
    for (i, (name, url)) in files.iter().enumerate() {
        prog.set_label(format!("Downloading {name}…"));
        crate::net::download(url, &dir.join(name), "", &mut |note| match note {
            crate::net::Note::Progress { got, total } => {
                if total > 0 {
                    // Each file owns an equal 1/n slice of the overall bar.
                    let frac = (i as f32 + got as f32 / total as f32) / n;
                    prog.pct.store((frac * 100.0) as u32, Relaxed);
                }
            }
            crate::net::Note::Retry { attempt, of, .. } => {
                prog.set_label(format!("Downloading {name}… (retry {}/{})", attempt - 1, of - 1));
            }
        })
        .map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

/// For the tests of what shows the list (the Tag Manager's settings).
#[cfg(test)]
impl ModelManager {
    /// A manager with nothing on disk and nothing in flight, whose downloads
    /// go nowhere.
    pub(crate) fn offline() -> Self {
        Self::with(tests::OFFLINE)
    }

    /// The model in `folder`, after a download of it failed with `error`.
    pub(crate) fn fail(&mut self, folder: &str, error: &str) {
        let entry = self.entries.iter_mut().find(|e| e.info.folder == folder).expect("a catalog folder");
        entry.status = Status::Error(error.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slider::rig::Rig;

    /// Nothing on disk, nothing in flight; a download is a progress that
    /// stays where it is until the test moves it.
    pub(super) const OFFLINE: Backend = Backend {
        installed: |_| false,
        fetch: |_| Arc::new(Progress::default()),
        running: Vec::new,
    };

    /// The list as the pop-up shows it — a column as wide as its cards —
    /// starting `offset` down it at the top of the rig's screen.
    struct List {
        /// The Tag Manager's faces are installed (the first frame does it).
        fonts: bool,
        models: ModelManager,
        offset: f32,
        /// Not drawn: the pop-up is closed, or on its other tab.
        hidden: bool,
        /// The pages the browser was asked for.
        opened: Vec<String>,
    }

    const LEFT: f32 = 20.0;
    const WIDTH: f32 = 354.0;

    fn scene(ui: &mut egui::Ui, s: &mut List) {
        if !s.fonts {
            let mut fonts = egui::FontDefinitions::default();
            crate::tag_manager::add_fonts(&mut fonts);
            ui.ctx().set_fonts(fonts);
            s.fonts = true;
            return;
        }
        s.models.tick(ui.ctx());
        if s.hidden {
            return;
        }
        let column = egui::Rect::from_min_size(egui::pos2(LEFT, -s.offset), egui::vec2(WIDTH, 2000.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(column), |ui| s.models.ui(ui));
        let asked = ui.ctx().output(|o| {
            let urls = o.commands.iter().filter_map(|c| match c {
                egui::OutputCommand::OpenUrl(open) => Some(open.url.clone()),
                _ => None,
            });
            urls.collect::<Vec<_>>()
        });
        s.opened.extend(asked);
    }

    /// The list on `backend`, settled.
    fn list(backend: Backend) -> Rig<List> {
        let models = ModelManager::with(backend);
        let mut rig = Rig::new(List { fonts: false, models, offset: 0.0, hidden: false, opened: vec![] }, scene);
        rig.idle();
        rig
    }

    /// Every text the last frame painted.
    fn galleys(rig: &Rig<List>) -> Vec<std::sync::Arc<egui::Galley>> {
        fn find(shape: &egui::Shape, found: &mut Vec<std::sync::Arc<egui::Galley>>) {
            match shape {
                egui::Shape::Text(t) => found.push(t.galley.clone()),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, found)),
                _ => {}
            }
        }
        let mut found = vec![];
        rig.shapes.iter().for_each(|c| find(&c.shape, &mut found));
        found
    }

    /// Where `text` is painted (once).
    fn text(rig: &Rig<List>, text: &str) -> egui::Rect {
        let found = rig.texts(text);
        assert_eq!(found.len(), 1, "{text:?} is painted {} times", found.len());
        found[0]
    }

    /// The middle of the slot of the row named `name`.
    fn slot(rig: &Rig<List>, name: &str) -> egui::Pos2 {
        // The name's line is the upper of the row's two.
        egui::pos2(LEFT + WIDTH - 1.0 - ROW_PAD - ICON / 2.0, text(rig, name).center().y + 10.0)
    }

    fn entry<'a>(rig: &'a Rig<List>, folder: &str) -> &'a Entry {
        rig.state.models.entries.iter().find(|e| e.info.folder == folder).expect("a catalog folder")
    }

    /// The pointer off the list, and long enough for a hover text to go.
    fn away(rig: &mut Rig<List>) {
        rig.frame(vec![egui::Event::PointerMoved(egui::pos2(520.0, 300.0))]);
        rig.idle();
    }

    /// The pointer resting at `pos` until its hover text is up (which takes
    /// a frame to size itself before it shows).
    fn rest(rig: &mut Rig<List>, pos: egui::Pos2) {
        rig.frame(vec![egui::Event::PointerMoved(pos)]);
        rig.idle();
        rig.frame(vec![]);
    }

    /// How far down the list a group starts: after the groups of `before`
    /// rows each.
    fn group_top(before: &[usize]) -> f32 {
        before.iter().map(|&rows| 14.0 + 8.0 + rows as f32 * ROW_H + (rows - 1) as f32 + 2.0 + GROUP_GAP).sum()
    }

    /// Every model has its row, under its group's heading, in the
    /// catalog's order — and the groups come one after another.
    #[test]
    fn every_model_is_listed_under_its_group() {
        let mut rig = list(OFFLINE);
        let groups: [(&str, &[&str]); 3] = [
            (
                "TAGGERS",
                &[
                    "PixAI Tagger v0.9Recommended",
                    "JoyTag",
                    "WD14 v3 SwinV2",
                    "WD14 v3 EVA02",
                    "WD14 v3 ConvNext",
                    "WD14 v2 ConvNext",
                ],
            ),
            ("LOCAL AI", &["Gemma 4 E4B", "Gemma 3 27B", "Gemma 4 31B", "Qwen3-VL 8B", "Qwen3-VL 30B Thinking"]),
            ("TOOLS", &["Depth Anything V2 Base", "BiRefNet Lite", "Region Detection", "Whisper Base"]),
        ];
        let mut before = vec![];
        for (heading, names) in groups {
            rig.state.offset = group_top(&before);
            rig.idle();
            let head = text(&rig, heading);
            assert!(head.top() >= 0.0 && head.top() < 8.0, "{heading} starts at {}", head.top());
            let mut last = head.bottom();
            for name in names {
                let at = text(&rig, name);
                assert!(at.top() > last, "{name} is under what's before it: {at:?} after {last}");
                assert!((at.left() - (LEFT + 1.0 + ROW_PAD)).abs() < 2.0, "{name} starts at {}", at.left());
                last = at.bottom();
            }
            before.push(names.len());
        }
        assert_eq!(before.iter().sum::<usize>(), CATALOG.len(), "every model in the catalog");
        // The next group would start past the list's end: there is none.
        rig.state.offset = group_top(&before);
        rig.idle();
        assert!(galleys(&rig).is_empty(), "nothing after the last group");
    }

    /// A row's two lines show whole at the pop-up's width: none is cut
    /// short, and none runs into the slot.
    #[test]
    fn every_name_and_caption_fits_its_row() {
        let mut rig = list(OFFLINE);
        let slot_starts = LEFT + WIDTH - 1.0 - SLOT_W;
        let mut seen = std::collections::HashSet::new();
        for before in [&[][..], &[6], &[6, 5]] {
            rig.state.offset = group_top(before);
            rig.idle();
            for galley in galleys(&rig) {
                let text = galley.text();
                let Some(info) = CATALOG.iter().find(|m| m.caption == text || text.starts_with(m.title())) else {
                    continue;
                };
                assert!(!galley.elided, "{text:?} is cut short");
                assert!(galley.size().x <= slot_starts - (LEFT + 1.0 + ROW_PAD), "{text:?} is {} wide", galley.size().x);
                if info.caption == text {
                    seen.insert(info.folder);
                }
            }
        }
        assert_eq!(seen.len(), CATALOG.len(), "every caption was looked at");
    }

    /// A download is the row's percentage and nothing else: where the arrow
    /// was, ending where it ended — no bar, no "Downloading" on the row, and
    /// the name and its line where they were.
    #[test]
    fn a_download_shows_as_its_percentage() {
        let mut rig = list(OFFLINE);
        let lines = |rig: &Rig<List>| (text(rig, "JoyTag"), text(rig, "5k+ tags · best for general use"));
        let before = lines(&rig);
        assert!(galleys(&rig).iter().all(|g| !g.text().ends_with('%')), "nothing downloads yet");

        rig.click(slot(&rig, "JoyTag"));
        away(&mut rig);
        let progress = entry(&rig, "joytag").progress.clone().expect("a click on the arrow starts the download");
        assert_eq!(entry(&rig, "joytag").status, Status::Downloading);
        let zero = text(&rig, "0%");

        progress.set_label("Downloading model.onnx…".to_string());
        progress.pct.store(42, Relaxed);
        away(&mut rig);
        let pct = text(&rig, "42%");
        let end = LEFT + WIDTH - 1.0 - ROW_PAD;
        assert!((pct.right() - end).abs() < 1.5 && (zero.right() - end).abs() < 1.5, "{zero:?}, {pct:?}: ends at {end}");
        assert!((pct.center().y - (before.0.center().y + 10.0)).abs() < 4.0, "{pct:?} is in the middle of the row");
        assert_eq!(lines(&rig), before, "the row's text stays where it was");
        let said: Vec<String> = galleys(&rig).iter().map(|g| g.text().to_string()).collect();
        assert!(said.iter().all(|t| !t.contains("Downloading")), "{said:?}");
        // Only that row: one percentage on the list.
        assert_eq!(said.iter().filter(|t| t.ends_with('%')).count(), 1, "{said:?}");

        // What it's doing is the slot's hover text.
        let at = slot(&rig, "JoyTag");
        rest(&mut rig, at);
        assert_eq!(rig.texts("Downloading model.onnx…").len(), 1, "the hover text");
        // The percentage takes no click: nothing starts over.
        rig.click(slot(&rig, "JoyTag"));
        away(&mut rig);
        assert!(Arc::ptr_eq(entry(&rig, "joytag").progress.as_ref().unwrap(), &progress));

        progress.finish_ok();
        away(&mut rig);
        assert_eq!(entry(&rig, "joytag").status, Status::Installed);
        assert!(galleys(&rig).iter().all(|g| !g.text().ends_with('%')), "the tick takes the percentage's place");
        assert_eq!(lines(&rig), before);
        // Installed, its slot starts nothing.
        rig.click(slot(&rig, "JoyTag"));
        away(&mut rig);
        assert_eq!(entry(&rig, "joytag").status, Status::Installed);
        assert!(rig.state.opened.is_empty(), "and no click on the slot opened a page");
    }

    /// A download that failed leaves its warning in the slot and nothing of
    /// its message on the row (a long one widened the old pop-up); the
    /// message is the slot's hover text, and a click there tries again.
    #[test]
    fn a_failed_download_is_tried_again_from_its_slot() {
        let mut rig = list(OFFLINE);
        let lines = |rig: &Rig<List>| (text(rig, "JoyTag"), text(rig, "5k+ tags · best for general use"));
        let before = lines(&rig);
        rig.click(slot(&rig, "JoyTag"));
        away(&mut rig);
        let first = entry(&rig, "joytag").progress.clone().expect("the download started");
        let error = "model.onnx: https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx: \
                     Connection Failed: Connect error: connection timed out";
        first.finish_err(error.to_string());
        away(&mut rig);
        assert_eq!(entry(&rig, "joytag").status, Status::Error(error.to_string()));
        assert!(entry(&rig, "joytag").progress.is_none());
        assert!(galleys(&rig).iter().all(|g| !g.text().contains("Connection Failed")), "the message isn't on the row");
        assert_eq!(lines(&rig), before, "the row's text stays where it was");

        let at = slot(&rig, "JoyTag");
        rest(&mut rig, at);
        assert_eq!(rig.texts(&format!("{error}\nClick to try again.")).len(), 1, "the hover text says what went wrong");

        rig.click(slot(&rig, "JoyTag"));
        away(&mut rig);
        let second = entry(&rig, "joytag").progress.clone().expect("a click on the warning tries again");
        assert!(!Arc::ptr_eq(&first, &second), "with a download of its own");
        assert_eq!(entry(&rig, "joytag").status, Status::Downloading);
        text(&rig, "0%");
    }

    /// A click on a row — anywhere before its slot — opens the model's
    /// page and downloads nothing; a click on the slot opens no page.
    #[test]
    fn a_click_on_a_row_opens_the_models_page() {
        let mut rig = list(OFFLINE);
        rig.click(text(&rig, "JoyTag").center());
        away(&mut rig);
        assert_eq!(rig.state.opened, ["https://huggingface.co/fancyfeast/joytag"]);
        assert_eq!(entry(&rig, "joytag").status, Status::NotInstalled);

        // The row's last point before the slot is still the row's.
        rig.state.opened.clear();
        let name = text(&rig, "JoyTag");
        rig.click(egui::pos2(LEFT + WIDTH - 1.0 - SLOT_W - 1.0, name.center().y + 10.0));
        away(&mut rig);
        assert_eq!(rig.state.opened.len(), 1);
        assert_eq!(entry(&rig, "joytag").status, Status::NotInstalled);

        rig.state.opened.clear();
        rig.click(slot(&rig, "JoyTag"));
        away(&mut rig);
        assert!(rig.state.opened.is_empty(), "the slot's click is the download's");
        assert_eq!(entry(&rig, "joytag").status, Status::Downloading);
    }

    /// A download something else started — Settings' AI Model tab, the
    /// mic's first use — is on the list too: its row follows it instead of
    /// offering a second one.
    #[test]
    fn a_download_started_elsewhere_shows_on_the_list() {
        static ELSEWHERE: Mutex<Vec<(&'static str, Arc<Progress>)>> = Mutex::new(Vec::new());
        let mut rig = list(Backend { running: || ELSEWHERE.lock().unwrap().clone(), ..OFFLINE });
        assert_eq!(entry(&rig, "joytag").status, Status::NotInstalled);

        let theirs = Arc::new(Progress::default());
        theirs.pct.store(7, Relaxed);
        ELSEWHERE.lock().unwrap().push(("joytag", theirs.clone()));
        rig.idle();
        assert_eq!(entry(&rig, "joytag").status, Status::Downloading);
        assert!(Arc::ptr_eq(entry(&rig, "joytag").progress.as_ref().unwrap(), &theirs), "the one running");
        text(&rig, "7%");

        ELSEWHERE.lock().unwrap().clear();
        theirs.finish_ok();
        rig.idle();
        assert_eq!(entry(&rig, "joytag").status, Status::Installed);
    }

    /// Back in view after a while out of it, the list looks at the disk
    /// again: a model that arrived meanwhile is installed, one whose files
    /// have gone is not. A failed download keeps its warning.
    #[test]
    fn back_in_view_the_list_looks_at_the_disk_again() {
        static ON_DISK: AtomicBool = AtomicBool::new(false);
        let installed = |m: &ModelInfo| m.folder == "joytag" && ON_DISK.load(Relaxed);
        let mut rig = list(Backend { installed, ..OFFLINE });
        rig.state.models.fail("wd14-swinv2-v3", "no route to host");
        let failed = Status::Error("no route to host".to_string());
        let reopen = |rig: &mut Rig<List>| {
            rig.state.hidden = true;
            rig.idle();
            rig.state.hidden = false;
            rig.idle();
        };

        ON_DISK.store(true, Relaxed);
        rig.idle();
        assert_eq!(entry(&rig, "joytag").status, Status::NotInstalled, "in view, it isn't looked for every frame");
        reopen(&mut rig);
        assert_eq!(entry(&rig, "joytag").status, Status::Installed);
        assert_eq!(entry(&rig, "wd14-swinv2-v3").status, failed);
        assert_eq!(entry(&rig, "wd14-eva02-v3").status, Status::NotInstalled);

        ON_DISK.store(false, Relaxed);
        reopen(&mut rig);
        assert_eq!(entry(&rig, "joytag").status, Status::NotInstalled);
        assert_eq!(entry(&rig, "wd14-swinv2-v3").status, failed);
    }

    /// A second request for a model that's already downloading joins that
    /// download (same progress) instead of starting a second transfer.
    #[test]
    fn one_download_per_model() {
        let (a, started) = claim("test-in-flight-a");
        let (b, again) = claim("test-in-flight-a");
        assert!(started && !again, "the second request must not start another download");
        assert!(Arc::ptr_eq(&a, &b), "it shows the running download's progress");
        let (_, other) = claim("test-in-flight-b");
        assert!(other, "another model downloads on its own");
        release("test-in-flight-a");
        release("test-in-flight-b");
        let (c, fresh) = claim("test-in-flight-a");
        assert!(fresh && !Arc::ptr_eq(&a, &c), "once it ended, a retry starts afresh");
        release("test-in-flight-a");
    }
}
