//! Flux text-to-image generation via a bundled ComfyUI backend — a sibling of the
//! Pixal3D view. "Setup Requirements" provisions a self-contained Python + CUDA
//! PyTorch + ComfyUI + ComfyUI-GGUF + the chosen Flux model under
//! `…/models/comfyui/`; the app then launches ComfyUI as a local server and
//! drives it over its HTTP API.
//!
//! STATUS: this turn ships the panel, the model picker, and the installer. The
//! ComfyUI server launch + the generation workflow/API are wired next.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};

use eframe::egui;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText};

use crate::theme::*;

const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(220, 70, 70);

/// ComfyUI download (master zip) and the GGUF custom-node.
const COMFYUI_ZIP: &str = "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip";
const COMFYUI_GGUF_ZIP: &str = "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip";
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";

/// Shared Flux text-encoders + VAE (all ungated).
const CLIP_L: &str =
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true";
const T5XXL_FP8: &str =
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true";
// The Flux VAE (ae.safetensors) — black-forest-labs' own repo is gated (401), so
// pull it from the public ComfyUI auto-installer asset mirror instead.
const FLUX_VAE: &str =
    "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/models/vae/ae.safetensors?download=true";

/// Z-Image Turbo files (Comfy-Org, ungated). The variant picks the text encoder.
const ZIMAGE_DIFFUSION: &str = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors?download=true";
const ZIMAGE_VAE: &str = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors?download=true";

/// Which model family a Generate tab drives.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenFamily {
    Flux,
    ZImage,
}

impl GenFamily {
    fn title(self) -> &'static str {
        match self {
            GenFamily::Flux => "Generate",
            GenFamily::ZImage => "Z-Image",
        }
    }
    /// Per-family outputs sub-folder, so each tab keeps its own history.
    fn out_dir(self) -> &'static str {
        match self {
            GenFamily::Flux => "flux",
            GenFamily::ZImage => "zimage",
        }
    }
    fn default_model(self) -> GenModel {
        match self {
            GenFamily::Flux => GenModel::SchnellQ4,
            GenFamily::ZImage => GenModel::ZImageFast,
        }
    }
}

/// A specific model variant offered in a family's picker.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenModel {
    // Flux (GGUF UNet)
    SchnellQ4,
    SchnellQ8,
    DevQ4,
    DevQ8,
    // Z-Image Turbo (bf16 diffusion; variant picks the text-encoder precision)
    ZImageFast,    // fp8 text encoder — lower VRAM
    ZImageQuality, // bf16 text encoder
}

impl GenModel {
    fn family(self) -> GenFamily {
        match self {
            GenModel::ZImageFast | GenModel::ZImageQuality => GenFamily::ZImage,
            _ => GenFamily::Flux,
        }
    }

    fn all_for(family: GenFamily) -> &'static [GenModel] {
        match family {
            GenFamily::Flux => &[GenModel::SchnellQ4, GenModel::SchnellQ8, GenModel::DevQ4, GenModel::DevQ8],
            GenFamily::ZImage => &[GenModel::ZImageFast, GenModel::ZImageQuality],
        }
    }

    fn label(self) -> &'static str {
        match self {
            GenModel::SchnellQ4 => "Flux schnell · GGUF Q4 (8GB+, fast, free)",
            GenModel::SchnellQ8 => "Flux schnell · GGUF Q8 (12GB+, free)",
            GenModel::DevQ4 => "Flux dev · GGUF Q4 (8GB+, gated)",
            GenModel::DevQ8 => "Flux dev · GGUF Q8 (12GB+, gated)",
            GenModel::ZImageFast => "Z-Image Turbo · fp8 encoder (low VRAM)",
            GenModel::ZImageQuality => "Z-Image Turbo · bf16 encoder (quality)",
        }
    }

    /// Flux: the UNet `.gguf` filename in `models/unet/`.
    fn unet_file(self) -> &'static str {
        match self {
            GenModel::SchnellQ4 => "flux1-schnell-Q4_K_S.gguf",
            GenModel::SchnellQ8 => "flux1-schnell-Q8_0.gguf",
            GenModel::DevQ4 => "flux1-dev-Q4_K_S.gguf",
            GenModel::DevQ8 => "flux1-dev-Q8_0.gguf",
            _ => "",
        }
    }

    fn unet_url(self) -> String {
        let (repo, file) = match self {
            GenModel::SchnellQ4 | GenModel::SchnellQ8 => ("city96/FLUX.1-schnell-gguf", self.unet_file()),
            _ => ("city96/FLUX.1-dev-gguf", self.unet_file()),
        };
        format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true")
    }

    /// Z-Image: the Qwen text-encoder filename in `models/text_encoders/`.
    fn zimage_te_file(self) -> &'static str {
        match self {
            GenModel::ZImageFast => "qwen_3_4b_fp8_mixed.safetensors",
            GenModel::ZImageQuality => "qwen_3_4b.safetensors",
            _ => "",
        }
    }

    /// Flux dev variants are gated and need an HF token; everything else is public.
    fn gated(self) -> bool {
        matches!(self, GenModel::DevQ4 | GenModel::DevQ8)
    }

    fn default_steps(self) -> i32 {
        match self {
            GenModel::SchnellQ4 | GenModel::SchnellQ8 => 4,
            GenModel::DevQ4 | GenModel::DevQ8 => 20,
            GenModel::ZImageFast | GenModel::ZImageQuality => 8,
        }
    }

    /// Flux uses FluxGuidance (~3.5); Z-Image Turbo runs at low CFG (~1).
    fn default_cfg(self) -> f32 {
        match self.family() {
            GenFamily::Flux => 3.5,
            GenFamily::ZImage => 1.0,
        }
    }
}

enum RunnerMsg {
    Line(String),
    Status(String),
    Output(PathBuf),
    Done(bool),
}

/// One LoRA found in ComfyUI's `models/loras/` dir, with its selection + weight.
struct LoraEntry {
    file: String,
    selected: bool,
    strength: f32,
}

/// UI + runtime state for a Generate view. Lives on `RightPanelState` (one per
/// family — Flux and Z-Image each get their own).
pub struct GenerateState {
    family: GenFamily,
    model: GenModel,
    prompt: String,
    steps: i32,
    cfg: f32,
    width: i32,
    height: i32,
    seed: i64,
    randomize_seed: bool,
    hf_token: String,

    status: String,
    status_err: bool,
    log: Vec<String>,
    rx: Option<Receiver<RunnerMsg>>,
    running: bool,
    orb: crate::ai_orb::AiOrb,
    last_image: Option<PathBuf>,
    /// Every image generated this session (plus any already in the outputs dir),
    /// oldest→newest — shown in the browser/viewer while the Generate view is open.
    gen_images: Vec<PathBuf>,
    /// Installed LoRAs (Z-Image), with their multi-select + weight state.
    loras: Vec<LoraEntry>,
    show_lora_popup: bool,
    show_log: bool,
}

impl Default for GenerateState {
    fn default() -> Self {
        Self::new(GenFamily::Flux)
    }
}

impl GenerateState {
    /// Build the state for a given model family (Flux or Z-Image).
    pub fn new(family: GenFamily) -> Self {
        let model = family.default_model();
        Self {
            family,
            model,
            prompt: String::new(),
            steps: model.default_steps(),
            cfg: model.default_cfg(),
            width: 1024,
            height: 1024,
            seed: 0,
            randomize_seed: true,
            hf_token: crate::pixal3d::load_hf_token(),
            status: "Ready".to_string(),
            status_err: false,
            log: Vec::new(),
            rx: None,
            running: false,
            orb: crate::ai_orb::AiOrb::default(),
            last_image: None,
            gen_images: load_outputs(family),
            loras: Vec::new(),
            show_lora_popup: false,
            show_log: false,
        }
    }

    /// The session's generated images (browser/viewer source while open).
    pub fn gen_images(&self) -> &[PathBuf] {
        &self.gen_images
    }
}

/// All `.png` files already in this family's outputs dir, oldest→newest.
fn load_outputs(family: GenFamily) -> Vec<PathBuf> {
    let dir = comfy_base().join("outputs").join(family.out_dir());
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
        .collect();
    v.sort_by_key(|p| p.metadata().and_then(|m| m.modified()).ok());
    v
}

/// Base install dir for the ComfyUI backend.
fn comfy_base() -> PathBuf {
    crate::tagger::models_root().join("comfyui")
}

/// LoRA `.safetensors` files in ComfyUI's `models/loras/` dir, alphabetical.
fn scan_loras() -> Vec<String> {
    let dir = comfy_base().join("ComfyUI").join("models").join("loras");
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("safetensors")))
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();
    v.sort();
    v
}

/// Re-scan the loras dir, preserving selection + weight of files still present.
fn refresh_loras(state: &mut GenerateState) {
    state.loras = scan_loras()
        .into_iter()
        .map(|f| {
            let prev = state.loras.iter().find(|l| l.file == f);
            LoraEntry {
                selected: prev.map(|l| l.selected).unwrap_or(false),
                strength: prev.map(|l| l.strength).unwrap_or(1.0),
                file: f,
            }
        })
        .collect();
}

/// The LoRA multi-select popup: a checkbox + weight slider per installed LoRA.
fn lora_popup(ctx: &egui::Context, state: &mut GenerateState) {
    egui::Window::new("")
        .id(egui::Id::new("zimage_lora_popup"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(window_frame())
        .show(ctx, |ui| {
            ui.set_width(380.0);
            let radius = CornerRadius::same(10);
            {
                let v = ui.visuals_mut();
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.active.corner_radius = radius;
            }

            // Title row (matches the Find Issues / Backup popups): icon + title +
            // a Refresh action and a close ✕.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/lora.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(TEXT()),
                );
                ui.heading(RichText::new("LoRAs").color(TEXT()).strong().size(17.0));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(RichText::new("✕").size(14.0)).frame(false))
                        .on_hover_text("Close")
                        .clicked()
                    {
                        state.show_lora_popup = false;
                    }
                    ui.add_space(2.0);
                    if ui.button(RichText::new("Refresh").size(11.0)).clicked() {
                        refresh_loras(state);
                    }
                });
            });
            ui.add_space(14.0);

            if state.loras.is_empty() {
                ui.label(
                    RichText::new("No LoRAs found. Drop .safetensors files into models/loras/.")
                        .color(MUTED())
                        .size(12.0),
                );
            } else {
                let bg = if crate::theme::is_light() { FIELD() } else { Color32::from_rgb(15, 15, 17) };
                egui::Frame::new()
                    .fill(bg)
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(Margin::same(8))
                    .stroke(egui::Stroke::new(1.0, EDGE()))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        egui::ScrollArea::vertical().max_height(320.0).auto_shrink([false, false]).show(ui, |ui| {
                            for l in &mut state.loras {
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut l.selected, "");
                                    let name = l.file.trim_end_matches(".safetensors");
                                    let col = if l.selected { TEXT() } else { MUTED() };
                                    ui.label(RichText::new(name).color(col).size(12.0));
                                });
                                if l.selected {
                                    ui.horizontal(|ui| {
                                        ui.add_space(24.0);
                                        ui.label(RichText::new("weight").color(MUTED()).size(10.5));
                                        ui.add(egui::Slider::new(&mut l.strength, 0.0..=2.0));
                                    });
                                }
                                ui.add_space(4.0);
                            }
                        });
                    });
            }

            ui.add_space(10.0);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let done = egui::Button::new(RichText::new("Done").color(Color32::WHITE).strong()).fill(ACCENT1());
                if ui.add_sized(egui::vec2(90.0, 30.0), done).clicked() {
                    state.show_lora_popup = false;
                }
            });
        });
}

fn window_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(CornerRadius::same(16))
        .inner_margin(Margin::same(18))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .shadow(egui::epaint::Shadow {
            offset: [0, 6],
            blur: 22,
            spread: 0,
            color: Color32::from_black_alpha(150),
        })
}

/// True once ComfyUI's source is present (a rough "installed" check).
fn is_installed() -> bool {
    comfy_base().join("ComfyUI").join("main.py").exists()
}

/// Render the Generate view into the right panel.
pub fn show(ui: &mut egui::Ui, state: &mut GenerateState) {
    // Drain background-runner messages.
    if let Some(rx) = &state.rx {
        let mut finished = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                RunnerMsg::Line(line) => state.log.push(line),
                RunnerMsg::Status(s) => state.status = s,
                RunnerMsg::Output(p) => {
                    state.last_image = Some(p.clone());
                    state.gen_images.push(p);
                }
                RunnerMsg::Done(ok) => {
                    state.running = false;
                    finished = true;
                    state.status_err = !ok;
                }
            }
        }
        if finished {
            state.rx = None;
        }
        if state.running {
            ui.ctx().request_repaint();
        }
    }

    // --- Header (title + status + orb), mirroring Pixal3D. ---
    egui::Frame::new()
        .fill(FIELD())
        .corner_radius(CornerRadius::same(18))
        .inner_margin(Margin::symmetric(12, 6))
        .show(ui, |ui| {
            ui.set_height(40.0);
            ui.horizontal_centered(|ui| {
                ui.label(RichText::new(state.family.title()).color(TEXT()).strong().size(14.0));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let color = if state.status_err {
                        RED
                    } else if state.running {
                        ACCENT1()
                    } else {
                        GREEN
                    };
                    ui.label(RichText::new(&state.status).color(color).size(12.0));
                    ui.add_space(6.0);
                    state.orb.set_state(if state.status_err {
                        crate::ai_orb::OrbState::Error
                    } else if state.running {
                        crate::ai_orb::OrbState::Thinking
                    } else {
                        crate::ai_orb::OrbState::Idle
                    });
                    state.orb.show(ui, 30.0, None);
                });
            });
        });

    ui.add_space(8.0);

    // --- Setup row. ---
    ui.horizontal(|ui| {
        let setup = egui::Button::new(RichText::new("Setup Requirements").color(Color32::WHITE))
            .fill(Color32::from_rgb(96, 99, 105))
            .corner_radius(CornerRadius::same(12));
        if ui.add_enabled_ui(!state.running, |ui| ui.add_sized(egui::vec2(150.0, 28.0), setup)).inner.clicked() {
            start_setup(state, ui.ctx());
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let txt = if is_installed() { "Installed" } else { "NVIDIA GPU required" };
            ui.label(RichText::new(txt).color(MUTED()).size(11.0));
        });
    });

    ui.add_space(8.0);

    // --- Model picker (+ a LoRA button for Z-Image). ---
    ui.horizontal(|ui| {
        ui.label(RichText::new("Model").color(MUTED()).size(12.0));
        if state.family == GenFamily::ZImage {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let n = state.loras.iter().filter(|l| l.selected).count();
                let label = if n > 0 { format!("LoRA ({n})") } else { "LoRA".to_string() };
                let icon = egui::Image::new(egui::include_image!("../icons/lora.svg"))
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(crate::theme::icon_tint(MUTED()));
                let btn = egui::Button::image_and_text(icon, RichText::new(label).size(11.0))
                    .corner_radius(CornerRadius::same(10));
                if ui.add(btn).clicked() {
                    refresh_loras(state);
                    state.show_lora_popup = true;
                }
            });
        }
    });
    ui.add_space(2.0);
    egui::ComboBox::from_id_salt("gen_model")
        .width(ui.available_width())
        .selected_text(state.model.label())
        .show_ui(ui, |ui| {
            for &m in GenModel::all_for(state.family) {
                if ui.selectable_label(state.model == m, m.label()).clicked() {
                    state.model = m;
                    state.steps = m.default_steps();
                    state.cfg = m.default_cfg();
                }
            }
        });
    if state.model.gated() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(RichText::new("HF token (dev is gated)").color(MUTED()).size(11.0));
            ui.add(
                egui::Image::new(egui::include_image!("../icons/info.svg"))
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(crate::theme::icon_tint(MUTED())),
            )
            .on_hover_ui(|ui| {
                ui.set_max_width(260.0);
                ui.label("Flux dev is a gated model: accept its license on Hugging Face and paste a token.");
                ui.hyperlink_to("Manage tokens →", "https://huggingface.co/settings/tokens");
            });
            let resp = ui.add(
                egui::TextEdit::singleline(&mut state.hf_token).password(true).desired_width(f32::INFINITY),
            );
            if resp.lost_focus() {
                crate::pixal3d::save_hf_token(&state.hf_token);
            }
        });
    }

    // LoRA selection popup (Z-Image).
    if state.show_lora_popup {
        lora_popup(ui.ctx(), state);
    }

    ui.add_space(10.0);

    // --- Prompt (fixed height; long text scrolls inside instead of pushing the
    // Generate button off the panel). ---
    ui.label(RichText::new("Prompt").color(MUTED()).size(12.0));
    ui.add_space(2.0);
    let bg = if crate::theme::is_light() { FIELD() } else { Color32::from_rgb(15, 15, 17) };
    egui::Frame::new()
        .fill(bg)
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(2))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical()
                .id_salt("flux_prompt")
                .max_height(108.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut state.prompt)
                            .desired_width(f32::INFINITY)
                            .desired_rows(4)
                            .frame(egui::Frame::NONE)
                            .hint_text("a cinematic photo of…"),
                    );
                });
        });

    ui.add_space(8.0);

    // --- Settings. ---
    int_slider(ui, "Steps", &mut state.steps, 1..=50);
    slider(ui, "Guidance (CFG)", &mut state.cfg, 1.0..=8.0);
    ui.horizontal(|ui| {
        int_slider(ui, "Width", &mut state.width, 512..=1536);
    });
    ui.horizontal(|ui| {
        int_slider(ui, "Height", &mut state.height, 512..=1536);
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.randomize_seed, "");
        ui.label(RichText::new("Randomize seed").color(TEXT()).size(12.0));
    });
    if !state.randomize_seed {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Seed").color(MUTED()).size(12.0));
            ui.add(egui::DragValue::new(&mut state.seed));
        });
    }

    ui.add_space(10.0);

    // --- Generate. ---
    let gen_btn = egui::Button::new(RichText::new("Generate").color(Color32::WHITE).strong())
        .fill(ACCENT1())
        .corner_radius(CornerRadius::same(12));
    let can_gen = !state.running && is_installed() && !state.prompt.trim().is_empty();
    if ui.add_enabled_ui(can_gen, |ui| ui.add_sized(egui::vec2(ui.available_width(), 34.0), gen_btn)).inner.clicked() {
        start_generate(state, ui.ctx());
    }

    ui.add_space(8.0);

    // Generated images appear in the left browser + centre viewer while this view
    // is open (see ViewerApp::sync_flux_browser).
    if !state.gen_images.is_empty() {
        ui.label(
            RichText::new(format!(
                "{} image(s) this session — browse them on the left.",
                state.gen_images.len()
            ))
            .color(MUTED())
            .size(11.0),
        );
        ui.add_space(8.0);
    }

    // --- Log (collapsible). ---
    ui.horizontal(|ui| {
        let arrow = if state.show_log { "▾" } else { "▸" };
        if ui.button(RichText::new(format!("{arrow} Log")).size(12.0)).clicked() {
            state.show_log = !state.show_log;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let copy = egui::Button::new(RichText::new("Copy").size(11.0)).corner_radius(CornerRadius::same(8));
            if ui.add_enabled(!state.log.is_empty(), copy).on_hover_text("Copy the log").clicked() {
                ui.ctx().copy_text(state.log.join("\n"));
            }
        });
    });
    if state.show_log {
        let bg = if crate::theme::is_light() { FIELD() } else { Color32::from_rgb(15, 15, 17) };
        egui::Frame::new()
            .fill(bg)
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::same(10))
            .stroke(egui::Stroke::new(1.0, EDGE()))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::ScrollArea::vertical().max_height(180.0).stick_to_bottom(true).show(ui, |ui| {
                    for l in &state.log {
                        ui.label(RichText::new(l).color(TEXT()).monospace().size(11.0));
                    }
                });
            });
    }
}

fn slider(ui: &mut egui::Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(MUTED()).size(12.0));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(egui::Slider::new(value, range));
        });
    });
}

fn int_slider(ui: &mut egui::Ui, label: &str, value: &mut i32, range: std::ops::RangeInclusive<i32>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(MUTED()).size(12.0));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(egui::Slider::new(value, range));
        });
    });
}

fn start_setup(state: &mut GenerateState, ctx: &egui::Context) {
    let (tx, rx) = mpsc::channel();
    state.rx = Some(rx);
    state.running = true;
    state.status = "Setting up…".into();
    state.status_err = false;
    state.log.clear();
    state.show_log = true; // surface progress + any error

    let model = state.model;
    let token = state.hf_token.trim().to_string();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let ok = run_setup(model, &token, &tx, &ctx);
        let _ = tx.send(RunnerMsg::Done(ok));
        ctx.request_repaint();
    });
}

fn run_setup(
    model: GenModel,
    token: &str,
    tx: &mpsc::Sender<RunnerMsg>,
    ctx: &egui::Context,
) -> bool {
    let send = |s: String| {
        let _ = tx.send(RunnerMsg::Line(s));
        ctx.request_repaint();
    };
    let status = |s: &str| {
        let _ = tx.send(RunnerMsg::Status(s.to_string()));
        ctx.request_repaint();
    };

    let base = comfy_base();
    if let Err(e) = std::fs::create_dir_all(&base) {
        send(format!("ERROR: could not create {}: {e}", base.display()));
        return false;
    }
    send(format!("== Install dir: {}", base.display()));

    // 1 — standalone Python.
    let py = if cfg!(windows) {
        base.join("python").join("python.exe")
    } else {
        base.join("python").join("bin").join("python3")
    };
    if !py.exists() {
        status("Downloading Python…");
        send("== [1/6] Downloading standalone Python…".into());
        let tarball = base.join("python.tar.gz");
        if let Err(e) = download(&crate::pixal3d::py_tarball_url(), &tarball, &send) {
            send(format!("ERROR: Python download failed: {e}"));
            return false;
        }
        send("== Extracting Python…".into());
        if !run_streamed(&base, "tar", &["-xzf", "python.tar.gz"], &send) {
            send("ERROR: failed to extract Python".into());
            return false;
        }
        let _ = std::fs::remove_file(&tarball);
    } else {
        send("== Python already present".into());
    }
    let py = py.to_string_lossy().to_string();

    // 2 — PyTorch (CUDA 12.8).
    status("Installing PyTorch…");
    send("== [2/6] Installing PyTorch 2.8 (CUDA 12.8)…".into());
    if !run_streamed(&base, &py, &["-m", "pip", "install", "--upgrade", "pip"], &send) {
        send("WARNING: pip upgrade failed".into());
    }
    // torch + torchvision + torchaudio MUST come from the same cu128 build, or
    // torchaudio's native DLL fails to load on Windows (WinError 127) and crashes
    // ComfyUI at startup (it imports torchaudio).
    if !run_streamed(
        &base,
        &py,
        &["-m", "pip", "install", "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "--index-url", TORCH_INDEX],
        &send,
    ) {
        send("ERROR: PyTorch install failed".into());
        return false;
    }

    // 3 — ComfyUI source.
    let comfy = base.join("ComfyUI");
    if !comfy.join("main.py").exists() {
        status("Downloading ComfyUI…");
        send("== [3/6] Downloading ComfyUI…".into());
        let zip = base.join("comfyui.zip");
        if let Err(e) = download(COMFYUI_ZIP, &zip, &send) {
            send(format!("ERROR: ComfyUI download failed: {e}"));
            return false;
        }
        if let Err(e) = unzip(&zip, &base) {
            send(format!("ERROR: failed to extract ComfyUI: {e}"));
            return false;
        }
        let _ = std::fs::remove_file(&zip);
        // The zip extracts as ComfyUI-master; rename to ComfyUI.
        let extracted = base.join("ComfyUI-master");
        if extracted.exists() {
            let _ = std::fs::rename(&extracted, &comfy);
        }
    } else {
        send("== ComfyUI already present".into());
    }

    // 4 — ComfyUI requirements + the GGUF node.
    status("Installing ComfyUI deps…");
    send("== [4/6] Installing ComfyUI requirements…".into());
    let reqs = comfy.join("requirements.txt");
    if reqs.exists() {
        let reqs_s = reqs.to_string_lossy().to_string();
        if !run_streamed(&base, &py, &["-m", "pip", "install", "-r", &reqs_s], &send) {
            send("WARNING: some ComfyUI requirements failed".into());
        }
    }
    // ComfyUI's requirements list torchaudio unpinned, which can pull a newer build
    // than torch 2.8.0 — re-pin it (no-deps so torch/vision aren't disturbed) to
    // fix the Windows DLL-load crash.
    send("== Re-pinning torchaudio to 2.8.0 (cu128) to match torch…".into());
    run_streamed(
        &base,
        &py,
        &["-m", "pip", "install", "--force-reinstall", "--no-deps", "torchaudio==2.8.0", "--index-url", TORCH_INDEX],
        &send,
    );
    // ComfyUI-GGUF custom node (for the .gguf UNet loader) + the gguf lib.
    let nodes = comfy.join("custom_nodes");
    let _ = std::fs::create_dir_all(&nodes);
    if !nodes.join("ComfyUI-GGUF").exists() {
        send("== Installing ComfyUI-GGUF node…".into());
        let zip = base.join("gguf_node.zip");
        if download(COMFYUI_GGUF_ZIP, &zip, &send).is_ok() {
            let _ = unzip(&zip, &nodes);
            let _ = std::fs::remove_file(&zip);
            let extracted = nodes.join("ComfyUI-GGUF-main");
            if extracted.exists() {
                let _ = std::fs::rename(&extracted, nodes.join("ComfyUI-GGUF"));
            }
        }
    }
    run_streamed(&base, &py, &["-m", "pip", "install", "gguf"], &send);

    // 5/6 — the chosen family's model files.
    let m = |sub: &str| comfy.join("models").join(sub);
    let fetch = |url: &str, dest: PathBuf, name: &str, auth: &str, send: &dyn Fn(String)| -> bool {
        if dest.exists() {
            send(format!("== {name} already present"));
            return true;
        }
        send(format!("== Downloading {name}…"));
        match download_auth(url, &dest, auth, send) {
            Ok(()) => true,
            Err(e) => {
                send(format!("ERROR: {name} failed: {e}"));
                false
            }
        }
    };
    let mut ok = true;
    match model.family() {
        GenFamily::Flux => {
            status("Downloading Flux model + encoders…");
            let _ = std::fs::create_dir_all(m("clip"));
            let _ = std::fs::create_dir_all(m("vae"));
            let _ = std::fs::create_dir_all(m("unet"));
            ok &= fetch(CLIP_L, m("clip").join("clip_l.safetensors"), "clip_l", "", &send);
            ok &= fetch(T5XXL_FP8, m("clip").join("t5xxl_fp8_e4m3fn.safetensors"), "t5xxl_fp8", "", &send);
            ok &= fetch(FLUX_VAE, m("vae").join("ae.safetensors"), "Flux VAE", "", &send);
            let unet_dest = m("unet").join(model.unet_file());
            if unet_dest.exists() {
                send("== UNet already present".into());
            } else if model.gated() && token.is_empty() {
                send("ERROR: this dev model is gated — add an HF token above and retry".into());
                ok = false;
            } else {
                let auth = if model.gated() { token } else { "" };
                ok &= fetch(&model.unet_url(), unet_dest, model.unet_file(), auth, &send);
            }
        }
        GenFamily::ZImage => {
            status("Downloading Z-Image model + encoder…");
            let _ = std::fs::create_dir_all(m("text_encoders"));
            let _ = std::fs::create_dir_all(m("diffusion_models"));
            let _ = std::fs::create_dir_all(m("vae"));
            let te = model.zimage_te_file();
            let te_url = format!(
                "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/{te}?download=true"
            );
            ok &= fetch(&te_url, m("text_encoders").join(te), te, "", &send);
            ok &= fetch(ZIMAGE_DIFFUSION, m("diffusion_models").join("z_image_turbo_bf16.safetensors"), "Z-Image diffusion", "", &send);
            ok &= fetch(ZIMAGE_VAE, m("vae").join("ae_zimage.safetensors"), "Z-Image VAE", "", &send);
        }
    }

    if ok {
        status("Setup complete");
        send("== Setup complete. Click Generate.".into());
    } else {
        status("Setup finished with errors — see log");
    }
    ok
}

const SERVER_URL: &str = "http://127.0.0.1:8188";

/// The long-lived ComfyUI server process (started on first Generate, reused after).
static SERVER: std::sync::Mutex<Option<std::process::Child>> = std::sync::Mutex::new(None);

fn start_generate(state: &mut GenerateState, ctx: &egui::Context) {
    let (tx, rx) = mpsc::channel();
    state.rx = Some(rx);
    state.running = true;
    state.status = "Generating…".into();
    state.status_err = false;

    let seed = if state.randomize_seed {
        (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
            & 0x7FFF_FFFF_FFFF) as i64
    } else {
        state.seed
    };
    let job = GenJob {
        model: state.model,
        prompt: state.prompt.clone(),
        steps: state.steps,
        guidance: state.cfg,
        width: state.width,
        height: state.height,
        seed,
        loras: state
            .loras
            .iter()
            .filter(|l| l.selected)
            .map(|l| (l.file.clone(), l.strength))
            .collect(),
    };
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let ok = run_generate(job, &tx, &ctx);
        let _ = tx.send(RunnerMsg::Done(ok));
        ctx.request_repaint();
    });
}

struct GenJob {
    model: GenModel,
    prompt: String,
    steps: i32,
    guidance: f32,
    width: i32,
    height: i32,
    seed: i64,
    /// Selected LoRAs (filename, weight) — applied for Z-Image.
    loras: Vec<(String, f32)>,
}

fn run_generate(job: GenJob, tx: &mpsc::Sender<RunnerMsg>, ctx: &egui::Context) -> bool {
    let send = |s: String| {
        let _ = tx.send(RunnerMsg::Line(s));
        ctx.request_repaint();
    };
    let status = |s: &str| {
        let _ = tx.send(RunnerMsg::Status(s.to_string()));
        ctx.request_repaint();
    };

    let base = comfy_base();
    let comfy = base.join("ComfyUI");
    let py = if cfg!(windows) {
        base.join("python").join("python.exe")
    } else {
        base.join("python").join("bin").join("python3")
    };

    // 1 — make sure the ComfyUI server is up.
    if !ping() {
        status("Starting ComfyUI…");
        if let Err(e) = start_server(&comfy, &py, &send) {
            send(format!("ERROR: {e}"));
            return false;
        }
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build()
        .into();

    // Remember where the server log is now, so we only read THIS run's progress.
    let log_path = base.join("comfyui-server.log");
    let log_offset = std::fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);

    // 2 — queue the workflow.
    status("Queuing…");
    let wf = build_workflow(&job);
    let body = serde_json::json!({ "prompt": wf, "client_id": "clarity-tagflow" }).to_string();
    let resp = match agent
        .post(&format!("{SERVER_URL}/prompt"))
        .header("Content-Type", "application/json")
        .send(body.as_str())
    {
        Ok(r) => r,
        Err(e) => {
            send(format!("ERROR: queue request failed: {e}"));
            return false;
        }
    };
    let json: serde_json::Value = match read_json(resp) {
        Ok(j) => j,
        Err(e) => {
            send(format!("ERROR: bad queue response: {e}"));
            return false;
        }
    };
    if let Some(errs) = json.get("node_errors").filter(|v| v.is_object() && !v.as_object().unwrap().is_empty()) {
        send(format!("ERROR: workflow rejected: {errs}"));
        return false;
    }
    let Some(prompt_id) = json.get("prompt_id").and_then(|v| v.as_str()).map(str::to_string) else {
        send("ERROR: no prompt_id in response".into());
        return false;
    };

    // 3 — poll /history until the image is ready (Flux is slow on first load).
    status("Generating…");
    let mut images: Option<serde_json::Value> = None;
    for _ in 0..600 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        // Live step progress, parsed from the server's tqdm output → shown by the orb.
        if let Some(pct) = read_progress(&log_path, log_offset) {
            status(&format!("Generating… {pct}%"));
        }
        let Ok(h) = agent.get(&format!("{SERVER_URL}/history/{prompt_id}")).call() else { continue };
        let Ok(hist) = read_json(h) else { continue };
        let Some(entry) = hist.get(&prompt_id) else { continue };
        // Surface a node error if the run failed.
        if entry.get("status").and_then(|s| s.get("status_str")).and_then(|v| v.as_str()) == Some("error") {
            send(format!("ERROR: generation failed: {}", entry.get("status").cloned().unwrap_or_default()));
            return false;
        }
        if let Some(imgs) = entry.get("outputs").and_then(|o| o.get("10")).and_then(|n| n.get("images")) {
            images = Some(imgs.clone());
            break;
        }
    }
    let Some(images) = images else {
        send("ERROR: timed out waiting for the image".into());
        return false;
    };
    let Some(img0) = images.as_array().and_then(|a| a.first()) else {
        send("ERROR: no image in output".into());
        return false;
    };
    let filename = img0.get("filename").and_then(|v| v.as_str()).unwrap_or("");
    let subfolder = img0.get("subfolder").and_then(|v| v.as_str()).unwrap_or("");
    let kind = img0.get("type").and_then(|v| v.as_str()).unwrap_or("output");

    // 4 — fetch the PNG bytes and save them into our outputs dir.
    status("Saving…");
    let view = format!(
        "{SERVER_URL}/view?filename={}&subfolder={}&type={}",
        urlencode(filename),
        urlencode(subfolder),
        kind
    );
    let bytes = match agent.get(&view).call().map_err(|e| e.to_string()).and_then(read_bytes) {
        Ok(b) => b,
        Err(e) => {
            send(format!("ERROR: fetching image failed: {e}"));
            return false;
        }
    };
    let outdir = base.join("outputs").join(job.model.family().out_dir());
    let _ = std::fs::create_dir_all(&outdir);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dest = outdir.join(format!("gen_{stamp}.png"));
    if let Err(e) = std::fs::write(&dest, &bytes) {
        send(format!("ERROR: could not save image: {e}"));
        return false;
    }
    send(format!("== Saved {}", dest.display()));
    let _ = tx.send(RunnerMsg::Output(dest));
    status("Done");
    true
}

/// Build the ComfyUI API workflow for the job's model family.
fn build_workflow(job: &GenJob) -> serde_json::Value {
    match job.model.family() {
        GenFamily::Flux => flux_workflow(job),
        GenFamily::ZImage => zimage_workflow(job),
    }
}

/// Z-Image Turbo: UNETLoader + CLIPLoader(lumina2, Qwen) + VAELoader + KSampler,
/// with any selected LoRAs chained in via LoraLoader nodes.
fn zimage_workflow(job: &GenJob) -> serde_json::Value {
    use serde_json::json;
    let mut wf = json!({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": job.model.zimage_te_file(), "type": "lumina2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae_zimage.safetensors"}},
    });
    let obj = wf.as_object_mut().unwrap();

    // Chain LoraLoader nodes; model/clip start at the loaders and thread through.
    let mut model_ref = json!(["1", 0]);
    let mut clip_ref = json!(["2", 0]);
    let mut id = 100;
    for (file, strength) in &job.loras {
        let node = id.to_string();
        id += 1;
        obj.insert(
            node.clone(),
            json!({"class_type": "LoraLoader", "inputs": {
                "model": model_ref, "clip": clip_ref,
                "lora_name": file, "strength_model": strength, "strength_clip": strength
            }}),
        );
        model_ref = json!([node, 0]);
        clip_ref = json!([node, 1]);
    }

    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": clip_ref.clone()}}));
    obj.insert("6".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": clip_ref}}));
    obj.insert("7".into(), json!({"class_type": "EmptySD3LatentImage", "inputs": {"width": job.width, "height": job.height, "batch_size": 1}}));
    obj.insert("8".into(), json!({"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["4", 0], "negative": ["6", 0], "latent_image": ["7", 0],
        "seed": job.seed, "steps": job.steps, "cfg": job.guidance,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
    }}));
    obj.insert("9".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}));
    obj.insert("10".into(), json!({"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ClarityZImage"}}));
    wf
}

/// Flux GGUF text-to-image workflow as a ComfyUI API prompt.
fn flux_workflow(job: &GenJob) -> serde_json::Value {
    serde_json::json!({
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": job.model.unet_file()}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"
        }},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": ["2", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": job.guidance}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": job.width, "height": job.height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
            "seed": job.seed, "steps": job.steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
        }},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ClarityFlux"}}
    })
}

/// Is the ComfyUI server answering on its port?
fn ping() -> bool {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(2)))
        .build()
        .into();
    agent.get(&format!("{SERVER_URL}/system_stats")).call().is_ok()
}

/// Launch ComfyUI (server output to a log file) and wait until it answers.
fn start_server(comfy: &Path, py: &Path, send: &dyn Fn(String)) -> Result<(), String> {
    use std::process::{Command, Stdio};

    let mut lock = SERVER.lock().unwrap();
    if ping() {
        return Ok(());
    }
    send("== Launching ComfyUI server (first run takes ~30-60s)…".into());
    #[cfg(windows)]
    crate::pixal3d::ensure_kill_on_exit_job();

    let logf = std::fs::File::create(comfy.parent().unwrap_or(comfy).join("comfyui-server.log"))
        .map_err(|e| e.to_string())?;
    let logf2 = logf.try_clone().map_err(|e| e.to_string())?;

    let mut cmd = Command::new(py);
    cmd.arg("main.py")
        .arg("--port")
        .arg("8188")
        .current_dir(comfy)
        // Force UTF-8 I/O: stdout is a file here, so Python would otherwise use the
        // Windows locale (cp1252) and crash when tqdm writes Unicode progress bars.
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdout(Stdio::from(logf))
        .stderr(Stdio::from(logf2));
    crate::pixal3d::hide_window(&mut cmd);
    crate::pixal3d::kill_on_parent_exit(&mut cmd);

    let child = cmd.spawn().map_err(|e| format!("could not start ComfyUI: {e}"))?;
    *lock = Some(child);
    drop(lock);

    // Wait for readiness (server.log has details if it never comes up).
    for _ in 0..180 {
        std::thread::sleep(std::time::Duration::from_millis(1000));
        if ping() {
            send("== ComfyUI server ready".into());
            return Ok(());
        }
    }
    Err("ComfyUI server did not become ready (see comfyui-server.log)".into())
}

fn read_json(resp: ureq::http::Response<ureq::Body>) -> Result<serde_json::Value, String> {
    let bytes = read_bytes(resp)?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn read_bytes(resp: ureq::http::Response<ureq::Body>) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut buf = Vec::new();
    resp.into_body().into_reader().read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Read the newest step-percentage from the ComfyUI server's tqdm output, looking
/// only at log written since `offset` (this run). Returns e.g. 75 for "75%|…".
fn read_progress(path: &Path, offset: u64) -> Option<u32> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    if len <= offset {
        return None;
    }
    // Only the tail of this run's output — the latest progress line is at the end.
    let start = offset.max(len.saturating_sub(8192));
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut s = String::new();
    f.take(len - start).read_to_string(&mut s).ok()?;
    parse_progress(&s)
}

/// The last `NN%|` (tqdm bar) percentage in `s`, if any.
fn parse_progress(s: &str) -> Option<u32> {
    let mut last = None;
    let mut i = 0;
    while let Some(pos) = s[i..].find("%|") {
        let abs = i + pos;
        let mut start = abs;
        let b = s.as_bytes();
        while start > 0 && b[start - 1].is_ascii_digit() {
            start -= 1;
        }
        if start < abs {
            if let Ok(n) = s[start..abs].parse::<u32>() {
                last = Some(n);
            }
        }
        i = abs + 2;
    }
    last
}

/// Minimal percent-encoding for the /view query params.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Provisioning helpers (self-contained; log via a closure)
// ---------------------------------------------------------------------------

fn run_streamed(cwd: &Path, program: &str, args: &[&str], send: &dyn Fn(String)) -> bool {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    send(format!("$ {program} {}", args.join(" ")));
    #[cfg(windows)]
    crate::pixal3d::ensure_kill_on_exit_job();

    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd).stdout(Stdio::piped()).stderr(Stdio::piped());
    crate::pixal3d::hide_window(&mut cmd);
    crate::pixal3d::kill_on_parent_exit(&mut cmd);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            send(format!("ERROR: cannot run {program}: {e}"));
            return false;
        }
    };
    let stderr = child.stderr.take();
    let err_handle = stderr.map(|err| {
        // stderr lines are funnelled back through a channel into `send` on the
        // main reader thread to keep `send` single-threaded.
        let (etx, erx) = mpsc::channel::<String>();
        let h = std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let _ = etx.send(line);
            }
        });
        (erx, h)
    });
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            send(line);
        }
    }
    if let Some((erx, h)) = err_handle {
        let _ = h.join();
        while let Ok(line) = erx.try_recv() {
            send(line);
        }
    }
    matches!(child.wait(), Ok(s) if s.success())
}

fn download(url: &str, dest: &Path, send: &dyn Fn(String)) -> Result<(), String> {
    download_auth(url, dest, "", send)
}

/// Stream `url` to `dest` (via `.part` then rename); `bearer` adds an HF token.
fn download_auth(url: &str, dest: &Path, bearer: &str, send: &dyn Fn(String)) -> Result<(), String> {
    use std::io::{Read, Write};

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .build(),
        )
        .max_redirects(10)
        .build()
        .into();

    let mut req = agent.get(url);
    if !bearer.is_empty() {
        req = req.header("Authorization", &format!("Bearer {bearer}"));
    }
    let resp = req.call().map_err(|e| e.to_string())?;
    let total: u64 = resp
        .headers()
        .get("Content-Length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let tmp = dest.with_extension("part");
    let mut out = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 16];
    let mut got: u64 = 0;
    let mut last_pct = 0u64;
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        got += n as u64;
        if total > 0 {
            let pct = got * 100 / total;
            if pct >= last_pct + 5 {
                last_pct = pct;
                send(format!("   {pct}%  ({:.1}/{:.1} MB)", got as f64 / 1e6, total as f64 / 1e6));
            }
        }
    }
    out.flush().ok();
    drop(out);
    std::fs::rename(&tmp, dest).map_err(|e| e.to_string())
}

fn unzip(zip_path: &Path, dest_dir: &Path) -> Result<(), String> {
    let file = std::fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let Some(rel) = entry.enclosed_name() else { continue };
        let out = dest_dir.join(rel);
        if entry.is_dir() {
            let _ = std::fs::create_dir_all(&out);
        } else {
            if let Some(parent) = out.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut f = std::fs::File::create(&out).map_err(|e| e.to_string())?;
            std::io::copy(&mut entry, &mut f).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
