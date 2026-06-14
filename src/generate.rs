//! Flux text-to-image generation via a bundled ComfyUI backend — a sibling of the
//! Pixal3D view. "Setup Requirements" provisions a self-contained Python + CUDA
//! PyTorch + ComfyUI + ComfyUI-GGUF + the chosen Flux model under
//! `…/models/comfyui/`; the app then launches ComfyUI as a local server and
//! drives it over its HTTP API.
//!
//! STATUS: this turn ships the panel, the model picker, and the installer. The
//! ComfyUI server launch + the generation workflow/API are wired next.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use eframe::egui;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText};

use crate::theme::*;

const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(220, 70, 70);

/// ComfyUI download (master zip) and the GGUF custom-node.
const COMFYUI_ZIP: &str = "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip";
const COMFYUI_GGUF_ZIP: &str = "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip";
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";

/// LTX-Video 0.9.6 2B distilled checkpoint (Lightricks, ungated). Video output
/// uses ComfyUI's native SaveWEBM node, so no custom node is required.
const LTX_CKPT: &str =
    "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-distilled-04-25.safetensors?download=true";
/// LTX spatial latent upscaler — drives the two-stage (upscale + refine) pipeline
/// that makes the video sharp instead of soft/blurry. Loaded by
/// LatentUpscaleModelLoader from models/latent_upscale_models/.
const LTX_UPSCALER: &str =
    "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-spatial-upscaler-0.9.7.safetensors?download=true";

/// LTX-2.3 22B distilled (quality option): the full fp8 checkpoint (bundles the
/// video+audio VAE), the Gemma-3 12B text encoder, and the 2.3 spatial upscaler.
/// All native ComfyUI — no custom nodes. ~40 GB total.
const LTX2_CKPT: &str =
    "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors?download=true";
const LTX2_GEMMA: &str =
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors?download=true";
const LTX2_UPSCALER: &str =
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors?download=true";

/// Wan 2.2 TI2V 5B (Alibaba, repackaged by Comfy-Org, ungated). A single model
/// that does BOTH text- and image-to-video — `Wan22ImageToVideoLatent` takes an
/// optional start image — so it slots straight into the t2v/i2v toggle. Ships its
/// own VAE; shares the umt5-xxl text encoder. All native ComfyUI nodes (~10 GB).
const WAN_TI2V_5B: &str =
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors?download=true";
const WAN_UMT5: &str =
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true";
const WAN_VAE: &str =
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors?download=true";

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

/// SDXL Base 1.0 checkpoint (stabilityai, ungated). One file bundles the UNet,
/// the dual CLIP text encoders, and the VAE — all native ComfyUI nodes (~6.9 GB).
const SDXL_CKPT: &str = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true";

/// Anima Base v1.0 (CircleStone Labs / Comfy Org, built on NVIDIA Cosmos 2) — a
/// 2B anime text-to-image model. Three split files loaded with native ComfyUI
/// UNETLoader + CLIPLoader(stable_diffusion) + VAELoader nodes.
const ANIMA_UNET: &str = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors?download=true";
const ANIMA_TE: &str = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors?download=true";
const ANIMA_VAE: &str = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors?download=true";

/// Which model family a Generate tab drives.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenFamily {
    Flux,
    ZImage,
    /// LTX-Video: text/image-to-video generation (outputs an .mp4, not an image).
    Ltx,
    /// Wan 2.2: text/image-to-video generation (single 5B TI2V model; .webm out).
    Wan,
    /// SDXL: Stable Diffusion XL text-to-image (single checkpoint, native nodes).
    Sdxl,
    /// Anima: 2B anime text-to-image (Cosmos-2 DiT; UNet + Qwen3 encoder + VAE).
    Anima,
}

impl GenFamily {
    fn title(self) -> &'static str {
        match self {
            GenFamily::Flux => "Flux",
            GenFamily::ZImage => "Z-Image",
            GenFamily::Ltx => "LTX Director",
            GenFamily::Wan => "Wan Director",
            GenFamily::Sdxl => "SDXL",
            GenFamily::Anima => "Anima",
        }
    }
    /// Per-family outputs sub-folder, so each tab keeps its own history.
    fn out_dir(self) -> &'static str {
        match self {
            GenFamily::Flux => "flux",
            GenFamily::ZImage => "zimage",
            GenFamily::Ltx => "ltx",
            GenFamily::Wan => "wan",
            GenFamily::Sdxl => "sdxl",
            GenFamily::Anima => "anima",
        }
    }
    fn default_model(self) -> GenModel {
        match self {
            GenFamily::Flux => GenModel::SchnellQ4,
            GenFamily::ZImage => GenModel::ZImageFast,
            GenFamily::Ltx => GenModel::LtxDistilled,
            GenFamily::Wan => GenModel::WanTi2v5bFast,
            GenFamily::Sdxl => GenModel::SdxlBase,
            GenFamily::Anima => GenModel::AnimaBase,
        }
    }
    /// This family produces video (an .mp4) rather than a still image — drives
    /// the video-only UI controls and the run loop's output handling.
    fn is_video(self) -> bool {
        matches!(self, GenFamily::Ltx | GenFamily::Wan)
    }
    /// This family loads a single swappable model file, so the picker can offer
    /// auto-detected installed models. SDXL swaps a full checkpoint (CheckpointLoaderSimple);
    /// Anima swaps the diffusion model (UNETLoader). The other families use
    /// multi-file GGUF/UNet setups that can't be swapped wholesale.
    fn uses_checkpoint_picker(self) -> bool {
        matches!(self, GenFamily::Sdxl | GenFamily::Anima)
    }
    /// The `models/` sub-dir holding this family's swappable model file, and that
    /// file's built-in (downloaded) default name — used to route the installed-model
    /// picker and skip the default from the installed list.
    fn model_subdir(self) -> &'static str {
        match self {
            GenFamily::Anima => "diffusion_models",
            _ => "checkpoints",
        }
    }
    fn default_model_file(self) -> &'static str {
        match self {
            GenFamily::Sdxl => "sd_xl_base_1.0.safetensors",
            GenFamily::Anima => "anima-base-v1.0.safetensors",
            _ => "",
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
    // LTX-Video 0.9.6 2B distilled (video; checkpoint + shared t5xxl encoder)
    LtxDistilled,
    // LTX-2.3 22B distilled fp8 (video+audio; Gemma-3 encoder, two-stage upscale)
    Ltx2Distilled,
    // Wan 2.2 TI2V 5B (video; one model does t2v+i2v). The two variants share the
    // same files and workflow — they differ only in the UNETLoader weight_dtype
    // (fp8 cast for lower VRAM vs. full fp16 for quality).
    WanTi2v5bFast,
    WanTi2v5bQuality,
    // SDXL Base 1.0 (single checkpoint: UNet + dual CLIP + VAE).
    SdxlBase,
    // Anima Base v1.0 (UNet + Qwen3 0.6B encoder + Qwen-Image VAE).
    AnimaBase,
}

impl GenModel {
    fn family(self) -> GenFamily {
        match self {
            GenModel::ZImageFast | GenModel::ZImageQuality => GenFamily::ZImage,
            GenModel::LtxDistilled | GenModel::Ltx2Distilled => GenFamily::Ltx,
            GenModel::WanTi2v5bFast | GenModel::WanTi2v5bQuality => GenFamily::Wan,
            GenModel::SdxlBase => GenFamily::Sdxl,
            GenModel::AnimaBase => GenFamily::Anima,
            _ => GenFamily::Flux,
        }
    }

    fn all_for(family: GenFamily) -> &'static [GenModel] {
        match family {
            GenFamily::Flux => &[GenModel::SchnellQ4, GenModel::SchnellQ8, GenModel::DevQ4, GenModel::DevQ8],
            GenFamily::ZImage => &[GenModel::ZImageFast, GenModel::ZImageQuality],
            GenFamily::Ltx => &[GenModel::LtxDistilled, GenModel::Ltx2Distilled],
            GenFamily::Wan => &[GenModel::WanTi2v5bFast, GenModel::WanTi2v5bQuality],
            GenFamily::Sdxl => &[GenModel::SdxlBase],
            GenFamily::Anima => &[GenModel::AnimaBase],
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
            GenModel::LtxDistilled => "LTX-Video 0.9.6 · 2B distilled (fast video)",
            GenModel::Ltx2Distilled => "LTX-2.3 · 22B distilled (quality video+audio, ~3min)",
            GenModel::WanTi2v5bFast => "Wan 2.2 · 5B TI2V · fp8 load (low VRAM)",
            GenModel::WanTi2v5bQuality => "Wan 2.2 · 5B TI2V · fp16 (quality)",
            GenModel::SdxlBase => "SDXL Base 1.0 (~6.9 GB)",
            GenModel::AnimaBase => "Anima Base v1.0 (~5 GB)",
        }
    }

    /// Wan: the UNETLoader `weight_dtype`. The "fast" variant casts the fp16
    /// checkpoint to fp8 at load time to fit smaller cards; "quality" keeps fp16.
    fn wan_weight_dtype(self) -> &'static str {
        match self {
            GenModel::WanTi2v5bFast => "fp8_e4m3fn",
            _ => "default",
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
            // Distilled runs at 8, but ~20 steps with a little CFG + a negative
            // prompt gives noticeably more motion and detail (the 8-step path
            // tends toward near-static, low-contrast video).
            GenModel::LtxDistilled => 20,
            // LTX-2.3 uses a fixed two-stage manual-sigma schedule baked into its
            // workflow template; this value is unused by it.
            GenModel::Ltx2Distilled => 8,
            // Wan 2.2 5B: the official template runs 20 steps with uni_pc/simple.
            GenModel::WanTi2v5bFast | GenModel::WanTi2v5bQuality => 20,
            // SDXL: 25-30 steps with a moderate CFG is the sweet spot.
            GenModel::SdxlBase => 28,
            // Anima: the official template runs 30 steps (er_sde / simple).
            GenModel::AnimaBase => 30,
        }
    }

    /// Flux uses FluxGuidance (~3.5); Z-Image Turbo runs at low CFG (~1); LTX
    /// distilled looks best with a small amount of guidance (~2).
    fn default_cfg(self) -> f32 {
        match self.family() {
            GenFamily::Flux => 3.5,
            GenFamily::ZImage => 1.0,
            GenFamily::Ltx => 2.0,
            GenFamily::Wan => 5.0,
            GenFamily::Sdxl => 7.0,
            GenFamily::Anima => 4.0,
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
    /// Which model family this LoRA was trained for (sniffed from its header).
    base: LoraBase,
    /// The ℹ details block under this row is expanded.
    info_open: bool,
    /// Decoded metadata, read lazily the first time ℹ is clicked.
    info: Option<LoraInfo>,
    /// Civitai preview thumbnail (loaded from the on-disk cache when present).
    thumb: Option<egui::TextureHandle>,
    /// The cache says this file has no Civitai preview — stop checking.
    thumb_missing: bool,
}

/// One full checkpoint found in ComfyUI's `models/checkpoints/` dir, with its
/// sniffed architecture so each tab only offers compatible files (like LoRAs).
struct CkptEntry {
    file: String,
    /// Architecture sniffed from the file header (reuses [`LoraBase`]'s families).
    base: LoraBase,
}

/// One textual-inversion embedding found in ComfyUI's `models/embeddings/` dir.
/// Like a LoRA it carries a multi-select state; the selected ones are injected as
/// `embedding:<name>` tokens into the workflow's positive or negative encoder at
/// generation time (never shown in the prompt box).
struct EmbeddingEntry {
    file: String,
    /// Architecture sniffed from the embedding's tensor shapes (reuses [`LoraBase`]).
    base: LoraBase,
    /// Included in the next generation.
    selected: bool,
    /// Apply to the negative prompt instead of the positive one.
    negative: bool,
    /// Token weight — emitted as `(embedding:name:weight)` when not 1.0.
    strength: f32,
    /// Civitai preview thumbnail (loaded from the on-disk cache when present).
    thumb: Option<egui::TextureHandle>,
    /// The cache says this file has no Civitai preview — stop checking.
    thumb_missing: bool,
}

/// Human-readable metadata decoded from a LoRA's safetensors header.
#[derive(Clone, Default)]
struct LoraInfo {
    /// Most frequent training tags — the best stand-in for trigger words —
    /// plus any explicit trigger phrase the trainer stamped.
    triggers: Vec<String>,
    /// Labelled facts (base model, resolution, steps, trainer, notes, …).
    facts: Vec<(String, String)>,
}

/// Decode the `__metadata__` block of a LoRA into display-ready info.
fn read_lora_info(path: &Path) -> LoraInfo {
    let mut info = LoraInfo::default();
    let Some(meta) = read_safetensors_header(path)
        .and_then(|h| h.get("__metadata__").cloned())
        .and_then(|m| m.as_object().cloned())
    else {
        return info;
    };

    // Trigger words: an explicit trigger phrase first, then the training tags
    // aggregated across datasets, most-used first.
    for key in ["modelspec.trigger_phrase", "ss_trigger_phrase", "trigger_phrase", "trigger"] {
        if let Some(v) = meta.get(key).and_then(|v| v.as_str()) {
            let v = v.trim();
            if !v.is_empty() {
                info.triggers.push(v.to_string());
            }
        }
    }
    if let Some(tf) = meta.get("ss_tag_frequency").and_then(|v| v.as_str()) {
        if let Ok(j) = serde_json::from_str::<serde_json::Value>(tf) {
            let mut counts: Vec<(String, u64)> = Vec::new();
            for tags in j.as_object().map(|o| o.values()).into_iter().flatten() {
                for (tag, n) in tags.as_object().map(|o| o.iter()).into_iter().flatten() {
                    let tag = tag.trim();
                    if tag.is_empty() {
                        continue;
                    }
                    match counts.iter_mut().find(|(t, _)| t == tag) {
                        Some((_, c)) => *c += n.as_u64().unwrap_or(0),
                        None => counts.push((tag.to_string(), n.as_u64().unwrap_or(0))),
                    }
                }
            }
            counts.sort_by(|a, b| b.1.cmp(&a.1));
            for (t, _) in counts.into_iter().take(24) {
                if !info.triggers.contains(&t) {
                    info.triggers.push(t);
                }
            }
        }
    }

    // Facts worth reading, in display order.
    let mut fact = |label: &str, key: &str| {
        if let Some(v) = meta.get(key).and_then(|v| v.as_str()) {
            let v = v.trim();
            if !v.is_empty() {
                info.facts.push((label.to_string(), v.to_string()));
            }
        }
    };
    fact("Name", "name");
    fact("Base model", "ss_base_model_version");
    fact("Architecture", "modelspec.architecture");
    fact("Resolution", "modelspec.resolution");
    fact("Network dim", "ss_network_dim");
    fact("Network alpha", "ss_network_alpha");
    fact("Steps", "ss_steps");
    fact("Epochs", "ss_num_epochs");
    fact("Training images", "ss_num_train_images");
    fact("Learning rate", "ss_learning_rate");
    fact("Trained", "modelspec.date");
    fact("Notes", "notes");
    // The trainer field is itself JSON: {"name": "ai-toolkit", "version": "…"}.
    if let Some(j) = meta
        .get("software")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
    {
        let name = j.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if !name.is_empty() {
            let ver = j.get("version").and_then(|v| v.as_str()).unwrap_or("");
            info.facts.push(("Trained with".into(), format!("{name} {ver}").trim().to_string()));
        }
    }
    info
}

/// A LoRA's base model, sniffed from its safetensors header — so each tab's
/// picker only offers compatible files (a Z-Image LoRA on Flux produces noise).
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoraBase {
    Flux,
    ZImage,
    /// SDXL family (incl. Pony, Illustrious, NoobAI — all SDXL-based).
    Sdxl,
    /// Anima — a 2B anime model (CircleStone Labs / Comfy Org, built on NVIDIA
    /// Cosmos 2). Its own DiT architecture; not usable by the current tabs.
    Anima,
    /// Identified as some other architecture (SD 1.5, …).
    Other,
    /// No recognisable signals — offered in every tab, marked with a caveat.
    Unknown,
}

impl LoraBase {
    fn matches(self, family: GenFamily) -> bool {
        match self {
            LoraBase::Flux => family == GenFamily::Flux,
            LoraBase::ZImage => family == GenFamily::ZImage,
            LoraBase::Sdxl => family == GenFamily::Sdxl,
            LoraBase::Anima => family == GenFamily::Anima,
            LoraBase::Other => false,
            LoraBase::Unknown => true,
        }
    }

    /// Short human label for the base-model family.
    fn label(self) -> &'static str {
        match self {
            LoraBase::Flux => "Flux",
            LoraBase::ZImage => "Z-Image",
            LoraBase::Sdxl => "SDXL",
            LoraBase::Anima => "Anima",
            LoraBase::Other => "SD / other",
            LoraBase::Unknown => "Unknown",
        }
    }

    /// A status-dot colour identifying the base family at a glance (used to flag
    /// off-family LoRAs in the picker).
    fn dot_color(self) -> Color32 {
        match self {
            LoraBase::Flux => Color32::from_rgb(232, 160, 60),
            LoraBase::ZImage => Color32::from_rgb(70, 180, 160),
            LoraBase::Sdxl => Color32::from_rgb(90, 150, 230),
            LoraBase::Anima => Color32::from_rgb(220, 110, 180),
            LoraBase::Other => Color32::from_rgb(150, 120, 220),
            LoraBase::Unknown => MUTED(),
        }
    }
}

/// Sniff a LoRA's base model from its safetensors JSON header. Civitai LoRAs
/// are trained with kohya / ai-toolkit, which stamp `ss_base_model_version`
/// (e.g. "zimage", "flux1", "sdxl_base_v1-0") and/or `modelspec.architecture`
/// (e.g. "zimageturbo/lora", "flux-1-dev/lora"); when both are missing, the
/// tensor names still fingerprint the architecture.
fn sniff_lora_base(path: &Path) -> LoraBase {
    let Some(header) = read_safetensors_header(path) else {
        return LoraBase::Unknown;
    };
    let Some(obj) = header.as_object() else {
        return LoraBase::Unknown;
    };

    // 1 — explicit training metadata.
    if let Some(meta) = obj.get("__metadata__").and_then(|m| m.as_object()) {
        for key in ["ss_base_model_version", "modelspec.architecture", "ss_sd_model_name"] {
            if let Some(v) = meta.get(key).and_then(|v| v.as_str()) {
                let v = v.to_ascii_lowercase();
                if v.contains("zimage") || v.contains("z-image") || v.contains("z_image") {
                    return LoraBase::ZImage;
                }
                if v.contains("flux") {
                    return LoraBase::Flux;
                }
                if ["sdxl", "sd_xl", "pony", "illustrious", "noobai", "animagine"]
                    .iter()
                    .any(|s| v.contains(s))
                {
                    return LoraBase::Sdxl;
                }
                // Anima (CircleStone Labs / Comfy Org, built on NVIDIA Cosmos 2):
                // architecture "Anima/lora". Checked after the SDXL list so
                // "animagine" (SDXL) above isn't misread as Anima.
                if v.contains("anima") || v.contains("cosmos") {
                    return LoraBase::Anima;
                }
                if ["stable-diffusion", "sd_v1", "sd_1", "sd15"]
                    .iter()
                    .any(|s| v.contains(s))
                {
                    return LoraBase::Other;
                }
            }
        }
    }

    // 2 — tensor-name fingerprints.
    let mut saw_flat_layers = false;
    for k in obj.keys() {
        if k == "__metadata__" {
            continue;
        }
        // Flux DiT block naming (kohya: lora_unet_double_blocks_…; diffusers:
        // transformer.single_transformer_blocks…; comfy: diffusion_model.double_blocks…).
        if k.contains("double_blocks") || k.contains("single_blocks") || k.contains("single_transformer_blocks") {
            return LoraBase::Flux;
        }
        // Anima (Cosmos-2 DiT): flat block stack with split self/cross attention,
        // e.g. lora_unet_blocks_0_cross_attn_k_proj / _self_attn_q_proj.
        if k.contains("lora_unet_blocks_") && (k.contains("_cross_attn_") || k.contains("_self_attn_")) {
            return LoraBase::Anima;
        }
        // SD / SDXL UNet naming.
        if k.contains("input_blocks") || k.contains("down_blocks") || k.contains("lora_te1") || k.contains("mid_block") {
            return LoraBase::Other;
        }
        // Z-Image (S3-DiT): a flat layer stack — diffusion_model.layers.N.attention…
        if k.starts_with("diffusion_model.layers.") || k.contains("lora_unet_layers_") {
            saw_flat_layers = true;
        }
    }
    if saw_flat_layers {
        LoraBase::ZImage
    } else {
        LoraBase::Unknown
    }
}

/// Sniff a *full checkpoint's* architecture from its safetensors header — used to
/// filter the model picker to compatible files (an SD 1.5 checkpoint on the SDXL
/// tab produces garbage). The signals differ from LoRAs: full checkpoints carry
/// the whole UNet + text-encoder(s), so the text-encoder layout is the giveaway —
/// SDXL is the only family with a *dual* encoder (`conditioner.embedders.1`).
fn sniff_checkpoint_base(path: &Path) -> LoraBase {
    // Non-safetensors checkpoints (.ckpt pickles) can't be header-sniffed.
    let Some(header) = read_safetensors_header(path) else {
        return LoraBase::Unknown;
    };
    let Some(obj) = header.as_object() else {
        return LoraBase::Unknown;
    };

    // 1 — explicit modelspec metadata, when the trainer stamped it.
    if let Some(meta) = obj.get("__metadata__").and_then(|m| m.as_object()) {
        for key in ["modelspec.architecture", "ss_base_model_version", "modelspec.title"] {
            if let Some(v) = meta.get(key).and_then(|v| v.as_str()) {
                let v = v.to_ascii_lowercase();
                if v.contains("flux") {
                    return LoraBase::Flux;
                }
                if v.contains("zimage") || v.contains("z-image") || v.contains("z_image") {
                    return LoraBase::ZImage;
                }
                if ["sdxl", "sd_xl", "stable-diffusion-xl", "pony", "illustrious", "noobai", "animagine"]
                    .iter()
                    .any(|s| v.contains(s))
                {
                    return LoraBase::Sdxl;
                }
                // Anima (Cosmos-2). After the SDXL list so "animagine" isn't caught.
                if v.contains("anima") || v.contains("cosmos") {
                    return LoraBase::Anima;
                }
            }
        }
    }

    // 2 — tensor-name fingerprints.
    let mut saw_cond_stage = false; // SD 1.x / 2.x single encoder
    for k in obj.keys() {
        if k == "__metadata__" {
            continue;
        }
        // Flux full checkpoint: DiT double/single blocks in the diffusion model.
        if k.contains("double_blocks") || k.contains("single_blocks") {
            return LoraBase::Flux;
        }
        // Z-Image (S3-DiT): flat layer stack.
        if k.starts_with("model.diffusion_model.layers.") || k.starts_with("diffusion_model.layers.") {
            return LoraBase::ZImage;
        }
        // SDXL: the second text encoder (OpenCLIP bigG) — unique to SDXL.
        if k.starts_with("conditioner.embedders.1") {
            return LoraBase::Sdxl;
        }
        // SD 1.x/2.x single encoder naming.
        if k.starts_with("cond_stage_model.") || k.starts_with("conditioner.embedders.0") {
            saw_cond_stage = true;
        }
    }
    if saw_cond_stage {
        // A UNet checkpoint with a single CLIP encoder — SD 1.5 / 2.x family.
        LoraBase::Other
    } else {
        LoraBase::Unknown
    }
}

/// The `models/` sub-dirs scanned for full checkpoints. Users (reasonably) drop
/// SDXL models into whichever folder they think of as "models", so we look in
/// all three — and register the same set with ComfyUI (see [`ensure_checkpoint_paths`])
/// so `CheckpointLoaderSimple` can actually load them wherever they landed.
const CKPT_DIRS: [&str; 3] = ["checkpoints", "diffusion_models", "unet"];

/// Full `.safetensors` / `.ckpt` checkpoints across all [`CKPT_DIRS`], as
/// (filename, full path), de-duplicated by filename (first folder wins) and sorted.
fn scan_checkpoints() -> Vec<(String, PathBuf)> {
    let models = comfy_base().join("ComfyUI").join("models");
    let mut seen = std::collections::HashSet::new();
    let mut v: Vec<(String, PathBuf)> = Vec::new();
    for sub in CKPT_DIRS {
        let entries = std::fs::read_dir(models.join(sub))
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("safetensors") || x.eq_ignore_ascii_case("ckpt"))
            });
        for p in entries {
            if let Some(name) = p.file_name().and_then(|n| n.to_str()).map(String::from) {
                if seen.insert(name.clone()) {
                    v.push((name, p));
                }
            }
        }
    }
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// Re-scan the checkpoint dirs, sniffing the architecture of new files (cheap
/// header read) and keeping cached results for known ones. Drops the current
/// custom-checkpoint selection if that file has gone away.
fn refresh_checkpoints(state: &mut GenerateState) {
    state.checkpoints = scan_checkpoints()
        .into_iter()
        .map(|(f, path)| {
            let prev = state.checkpoints.iter().find(|c| c.file == f);
            CkptEntry {
                base: prev.map(|c| c.base).unwrap_or_else(|| sniff_checkpoint_base(&path)),
                file: f,
            }
        })
        .collect();
    if let Some(sel) = &state.checkpoint {
        if !state.checkpoints.iter().any(|c| &c.file == sel) {
            state.checkpoint = None;
        }
    }
}

/// Make both ComfyUI loaders find a model wherever it landed: cross-register all
/// [`CKPT_DIRS`] under BOTH the `checkpoints` key (SDXL's CheckpointLoaderSimple)
/// and the `diffusion_models` key (Anima's UNETLoader), via an `extra_model_paths
/// .yaml` next to `main.py`. Returns `true` if the file was created or changed —
/// the caller restarts the server so the new folders take effect (ComfyUI reads
/// paths only at startup).
fn ensure_checkpoint_paths(comfy: &Path) -> bool {
    let rel_dirs: String = CKPT_DIRS.iter().map(|d| format!("    models/{d}\n")).collect();
    let base = comfy.to_string_lossy().replace('\\', "/");
    let yaml = format!(
        "# Auto-written by Clarity TagFlow — lets the checkpoint (SDXL) and\n\
         # diffusion-model (Anima) loaders find models in any of these folders.\n\
         clarity_tagflow:\n  base_path: {base}\n  checkpoints: |\n{rel_dirs}  diffusion_models: |\n{rel_dirs}"
    );
    let path = comfy.join("extra_model_paths.yaml");
    if std::fs::read_to_string(&path).ok().as_deref() == Some(yaml.as_str()) {
        return false; // already current
    }
    std::fs::write(&path, &yaml).is_ok()
}

/// Sniff a textual-inversion embedding's architecture from its tensor shapes.
/// SDXL embeddings carry a 1280-dim vector (the OpenCLIP-bigG encoder) — often
/// alongside a 768-dim one; SD 1.5 embeddings are 768-dim only. `.pt`/`.bin`
/// pickles can't be header-read, so they come back Unknown (still offered).
fn sniff_embedding_base(path: &Path) -> LoraBase {
    let Some(header) = read_safetensors_header(path) else {
        return LoraBase::Unknown;
    };
    let Some(obj) = header.as_object() else {
        return LoraBase::Unknown;
    };
    let mut saw_768 = false;
    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        if let Some(shape) = v.get("shape").and_then(|s| s.as_array()) {
            for dim in shape {
                match dim.as_i64() {
                    Some(1280) | Some(2560) => return LoraBase::Sdxl, // bigG (SDXL)
                    Some(4096) => return LoraBase::Flux,              // T5-XXL (Flux)
                    Some(768) | Some(1024) => saw_768 = true,        // CLIP-L / SD 2.x
                    _ => {}
                }
            }
        }
    }
    if saw_768 {
        LoraBase::Other
    } else {
        LoraBase::Unknown
    }
}

/// Textual-inversion embeddings in ComfyUI's `models/embeddings/` dir, as
/// (filename, full path), alphabetical. Accepts `.safetensors`, `.pt`, `.bin`.
fn scan_embeddings() -> Vec<(String, PathBuf)> {
    let dir = comfy_base().join("ComfyUI").join("models").join("embeddings");
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|x| {
                x.eq_ignore_ascii_case("safetensors")
                    || x.eq_ignore_ascii_case("pt")
                    || x.eq_ignore_ascii_case("bin")
            })
        })
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(|n| (n.to_string(), p.clone())))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// Re-scan the embeddings dir, sniffing the architecture of new files and keeping
/// cached results (incl. selection/weight/thumbnail) for known ones. Kicks off the
/// background Civitai-thumbnail fetch for any file without a cached preview.
fn refresh_embeddings(state: &mut GenerateState, ctx: &egui::Context) {
    state.embeddings = scan_embeddings()
        .into_iter()
        .map(|(f, path)| {
            let prev = state.embeddings.iter().find(|e| e.file == f);
            EmbeddingEntry {
                base: prev.map(|e| e.base).unwrap_or_else(|| sniff_embedding_base(&path)),
                selected: prev.map(|e| e.selected).unwrap_or(false),
                negative: prev.map(|e| e.negative).unwrap_or(false),
                strength: prev.map(|e| e.strength).unwrap_or(1.0),
                thumb: prev.and_then(|e| e.thumb.clone()),
                thumb_missing: prev.map(|e| e.thumb_missing).unwrap_or(false),
                file: f,
            }
        })
        .collect();
    spawn_embedding_thumb_fetch(state.embeddings.iter().map(|e| e.file.clone()).collect(), ctx.clone(), false);
}

/// The filename with any embedding extension stripped — the cache key + the name
/// shown in the picker.
fn embed_stem(file: &str) -> &str {
    file.trim_end_matches(".safetensors").trim_end_matches(".pt").trim_end_matches(".bin")
}

/// Where downloaded embedding preview thumbnails live: `<stem>.img` is the raw
/// JPEG; `<stem>.none` remembers "this hash isn't on Civitai".
fn embedding_thumbs_dir() -> PathBuf {
    comfy_base().join("embedding_thumbs")
}

/// Set when the embeddings picker closes; the thumb worker checks it between files.
static EMBED_THUMB_CANCEL: AtomicBool = AtomicBool::new(false);
/// True while the embedding thumbnail worker is running (drives the spinner).
static EMBED_THUMB_RUNNING: AtomicBool = AtomicBool::new(false);

/// Stop the background embedding-thumbnail fetch (called when the picker closes).
fn cancel_embedding_thumb_fetch() {
    EMBED_THUMB_CANCEL.store(true, Ordering::SeqCst);
}

/// Whether the embedding preview fetch is currently running.
fn embeddings_fetching() -> bool {
    EMBED_THUMB_RUNNING.load(Ordering::SeqCst)
}

/// Background pass fetching Civitai preview thumbnails for embeddings without one:
/// SHA-256 of the file → Civitai's by-hash endpoint → first still preview, saved
/// to the thumbs cache. Single-flight; repaints as each thumb lands. Embeddings
/// are tiny so hashing is near-instant. Runs only while the picker is open.
/// `force` re-attempts files previously marked "not on Civitai".
fn spawn_embedding_thumb_fetch(files: Vec<String>, ctx: egui::Context, force: bool) {
    EMBED_THUMB_CANCEL.store(false, Ordering::SeqCst);
    let tdir = embedding_thumbs_dir();
    let todo: Vec<String> = files
        .into_iter()
        .filter(|f| {
            let stem = embed_stem(f);
            let have_img = tdir.join(format!("{stem}.img")).exists();
            let have_none = tdir.join(format!("{stem}.none")).exists();
            !have_img && (force || !have_none)
        })
        .collect();
    if todo.is_empty() || EMBED_THUMB_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                EMBED_THUMB_RUNNING.store(false, Ordering::SeqCst);
            }
        }
        let _reset = Reset;

        let dir = comfy_base().join("ComfyUI").join("models").join("embeddings");
        let _ = std::fs::create_dir_all(&tdir);
        for f in todo {
            if EMBED_THUMB_CANCEL.load(Ordering::SeqCst) {
                break; // picker closed — finish later, when it reopens
            }
            let stem = embed_stem(&f);
            let bytes = crate::scan::sha256_file(&dir.join(&f))
                .and_then(|sha| crate::civitai::preview_image_by_hash(&sha));
            let write = match &bytes {
                Some(b) => std::fs::write(tdir.join(format!("{stem}.img")), b),
                None => std::fs::write(tdir.join(format!("{stem}.none")), b""),
            };
            let _ = write;
            ctx.request_repaint();
        }
    });
}

/// The prompt token for a textual inversion (extension stripped): bare
/// `embedding:<name>` at weight 1.0, or the weighted form `(embedding:<name>:w)`
/// otherwise — ComfyUI's emphasis syntax.
fn embedding_token(file: &str, strength: f32) -> String {
    let stem = file
        .trim_end_matches(".safetensors")
        .trim_end_matches(".pt")
        .trim_end_matches(".bin");
    if (strength - 1.0).abs() < 0.001 {
        format!("embedding:{stem}")
    } else {
        format!("(embedding:{stem}:{strength:.2})")
    }
}

/// The space-joined `embedding:<name>` tokens for the selected embeddings of the
/// requested polarity (`negative`), filtered to the current family — what gets
/// appended to that polarity's encoder text at generation time.
fn embed_tokens(state: &GenerateState, negative: bool) -> String {
    state
        .embeddings
        .iter()
        .filter(|e| e.selected && e.negative == negative && e.base.matches(state.family))
        .map(|e| embedding_token(&e.file, e.strength))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Append embedding tokens to a prompt-encode text (positive or negative), so the
/// embeddings affect generation without ever appearing in the user's prompt box.
fn with_embeds(base: &str, tokens: &str) -> String {
    match (base.is_empty(), tokens.is_empty()) {
        (_, true) => base.to_string(),
        (true, false) => tokens.to_string(),
        (false, false) => format!("{base} {tokens}"),
    }
}

/// Kill the running ComfyUI server (if any) and wait until it stops responding,
/// so a fresh start picks up changed config. Used after [`ensure_checkpoint_paths`].
fn stop_server() {
    if let Ok(mut lock) = SERVER.lock() {
        if let Some(mut child) = lock.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    for _ in 0..30 {
        if !ping() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[cfg(test)]
mod lora_sniff_tests {
    use super::*;

    /// Write a minimal fake .safetensors (header only) and return its path.
    fn fake_safetensors(name: &str, header: serde_json::Value) -> PathBuf {
        let path = std::env::temp_dir().join(format!("ctf_lora_test_{name}.safetensors"));
        let json = serde_json::to_vec(&header).unwrap();
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&json);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn metadata_wins() {
        let p = fake_safetensors(
            "zit_meta",
            serde_json::json!({"__metadata__": {"ss_base_model_version": "zimage"}}),
        );
        assert!(matches!(sniff_lora_base(&p), LoraBase::ZImage));
        let p = fake_safetensors(
            "flux_meta",
            serde_json::json!({"__metadata__": {"modelspec.architecture": "flux-1-dev/lora"}}),
        );
        assert!(matches!(sniff_lora_base(&p), LoraBase::Flux));
        let p = fake_safetensors(
            "sdxl_meta",
            serde_json::json!({"__metadata__": {"ss_base_model_version": "sdxl_base_v1-0"}}),
        );
        assert!(matches!(sniff_lora_base(&p), LoraBase::Other));
    }

    #[test]
    fn tensor_names_as_fallback() {
        let p = fake_safetensors(
            "flux_tensors",
            serde_json::json!({"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}),
        );
        assert!(matches!(sniff_lora_base(&p), LoraBase::Flux));
        let p = fake_safetensors(
            "zit_tensors",
            serde_json::json!({"diffusion_model.layers.0.attention.to_k.lora_A.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}),
        );
        assert!(matches!(sniff_lora_base(&p), LoraBase::ZImage));
        let p = fake_safetensors(
            "sd_tensors",
            serde_json::json!({"lora_unet_down_blocks_0_attentions_0.lora_down.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}),
        );
        assert!(matches!(sniff_lora_base(&p), LoraBase::Other));
    }

    #[test]
    fn unreadable_or_signal_free_is_unknown() {
        let p = fake_safetensors("bare", serde_json::json!({"some.unrecognised.tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]}}));
        assert!(matches!(sniff_lora_base(&p), LoraBase::Unknown));
        assert!(matches!(sniff_lora_base(Path::new("Z:/does/not/exist.safetensors")), LoraBase::Unknown));
    }
}

/// The JSON header of a `.safetensors` file: an 8-byte little-endian length,
/// then that many bytes of JSON (tensor table + optional `__metadata__`).
fn read_safetensors_header(path: &Path) -> Option<serde_json::Value> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let len = u64::from_le_bytes(len8);
    if len == 0 || len > 16 * 1024 * 1024 {
        return None; // corrupt or absurd header — not a real safetensors file
    }
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).ok()?;
    serde_json::from_slice(&buf).ok()
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
    /// Reveal LoRAs trained for other model families in the picker (off by
    /// default — they're hidden behind a checkbox at the bottom of the popup).
    show_other_loras: bool,
    /// Installed full checkpoints (auto-detected from `models/checkpoints/`),
    /// sniffed + filtered per family — offered alongside the built-in download
    /// options in the model picker.
    checkpoints: Vec<CkptEntry>,
    /// The selected installed checkpoint filename, or `None` to use the built-in
    /// model's default checkpoint. Only meaningful for families that use a single
    /// swappable checkpoint (SDXL).
    checkpoint: Option<String>,
    /// Reveal checkpoints sniffed as other families in the model picker.
    show_other_ckpts: bool,
    /// Tracks the model-picker open/closed edge so the checkpoint scan runs once
    /// per open, not every frame.
    ckpt_menu_was_open: bool,
    /// Installed textual-inversion embeddings (auto-detected from
    /// `models/embeddings/`), filtered per family — inserted into the prompt as
    /// `embedding:<name>` tokens from the + menu.
    embeddings: Vec<EmbeddingEntry>,
    show_embeddings_popup: bool,
    /// Reveal embeddings sniffed as other families in the picker.
    show_other_embeddings: bool,
    /// When the embeddings-popup Refresh icon was last clicked (one-shot spin).
    embeddings_refresh_spin: Option<std::time::Instant>,
    show_log: bool,
    /// Brief tint flash on the Copy-log button: (when, was_ok). Green on success,
    /// red on failure; fades out after a short window.
    copy_flash: Option<(std::time::Instant, bool)>,
    /// Spell-check state for the prompt box (right-click suggestion menu).
    spell: crate::spellcheck::SpellcheckState,
    /// Cooperative cancel flag for the in-flight generation (the in-prompt stop
    /// button sets it; the runner thread polls it and interrupts ComfyUI).
    cancel: Arc<AtomicBool>,
    /// Z-Image: measured height (last frame) of everything between the prompt box
    /// and the end of the Log row. The prompt stretches by this much short of the
    /// panel bottom, so the Log row + copy icon sit exactly on the bottom edge.
    below_h: f32,
    /// Video families (LTX): number of frames to render (snapped to 8k+1 in the
    /// workflow), the playback frame rate, and whether to animate the selected
    /// image (image-to-video) instead of pure text-to-video.
    frames: i32,
    fps: i32,
    i2v: bool,
    /// Z-Image: the reference square-edge size (px) the aspect tiles scale to, so
    /// one "Size" slider resizes every ratio at once. Unused by other families.
    size: i32,
    /// When the LoRA-popup Refresh icon was last clicked — drives its one-shot
    /// spin animation. `None` when not spinning.
    lora_refresh_spin: Option<std::time::Instant>,
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
        // Video models are trained on landscape; a square/portrait canvas degrades
        // badly (washed-out, incoherent). LTX defaults to 768×512; Wan 2.2 5B is
        // trained at 1280×704, its native (and best-looking) resolution.
        let (width, height) = match family {
            GenFamily::Wan => (1280, 704),
            GenFamily::Ltx => (768, 512),
            _ => (1024, 1024),
        };
        // Default clip length (snapped to each model's stride later). Wan 2.2 5B at
        // its native 1280×704 is heavy (~45 s/step on a 16 GB laptop), so default
        // to a short 2-second clip (49 = 4·12+1) for a fast first run; LTX is far
        // lighter, so it keeps the fuller 97-frame default.
        let frames = match family {
            GenFamily::Wan => 49,
            _ => 97,
        };
        Self {
            family,
            model,
            prompt: String::new(),
            steps: model.default_steps(),
            cfg: model.default_cfg(),
            width,
            height,
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
            show_other_loras: false,
            checkpoints: Vec::new(),
            checkpoint: None,
            show_other_ckpts: false,
            ckpt_menu_was_open: false,
            embeddings: Vec::new(),
            show_embeddings_popup: false,
            show_other_embeddings: false,
            embeddings_refresh_spin: None,
            show_log: false,
            copy_flash: None,
            spell: crate::spellcheck::SpellcheckState::default(),
            cancel: Arc::new(AtomicBool::new(false)),
            below_h: 210.0,
            frames,
            fps: 24,
            i2v: false,
            size: 1024,
            lora_refresh_spin: None,
        }
    }

    /// The session's generated images (browser/viewer source while open).
    pub fn gen_images(&self) -> &[PathBuf] {
        &self.gen_images
    }
}

/// All output files already in this family's outputs dir, oldest→newest —
/// `.png` for image families, video containers for video families (LTX).
fn load_outputs(family: GenFamily) -> Vec<PathBuf> {
    let dir = comfy_base().join("outputs").join(family.out_dir());
    let exts: &[&str] = if family.is_video() { &["mp4", "webm"] } else { &["png"] };
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| exts.iter().any(|e| x.eq_ignore_ascii_case(e))))
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
/// New files get their base model sniffed (a cheap header read); known files
/// keep the cached result. Also kicks off the background Civitai-thumbnail
/// fetch for any file without a cached preview.
fn refresh_loras(state: &mut GenerateState, ctx: &egui::Context) {
    let dir = comfy_base().join("ComfyUI").join("models").join("loras");
    state.loras = scan_loras()
        .into_iter()
        .map(|f| {
            let prev = state.loras.iter().find(|l| l.file == f);
            LoraEntry {
                selected: prev.map(|l| l.selected).unwrap_or(false),
                strength: prev.map(|l| l.strength).unwrap_or(1.0),
                base: prev.map(|l| l.base).unwrap_or_else(|| sniff_lora_base(&dir.join(&f))),
                info_open: prev.map(|l| l.info_open).unwrap_or(false),
                info: prev.and_then(|l| l.info.clone()),
                thumb: prev.and_then(|l| l.thumb.clone()),
                thumb_missing: prev.map(|l| l.thumb_missing).unwrap_or(false),
                file: f,
            }
        })
        .collect();
    spawn_lora_thumb_fetch(state.loras.iter().map(|l| l.file.clone()).collect(), ctx.clone(), false);
}

/// Where downloaded LoRA preview thumbnails live: `<stem>.img` is the raw
/// downloaded JPEG; `<stem>.none` remembers "this hash isn't on Civitai".
fn lora_thumbs_dir() -> PathBuf {
    comfy_base().join("lora_thumbs")
}

/// Set when the LoRA picker closes; the thumb worker checks it between files,
/// so hashing/downloading only happens while the picker is actually open.
static THUMB_FETCH_CANCEL: AtomicBool = AtomicBool::new(false);
/// True while the LoRA thumbnail worker is running (drives the download glyph's
/// spinner).
static LORA_THUMB_RUNNING: AtomicBool = AtomicBool::new(false);

/// Stop the background thumbnail fetch (called when the picker closes; the
/// current file finishes, then the worker exits).
fn cancel_lora_thumb_fetch() {
    THUMB_FETCH_CANCEL.store(true, Ordering::SeqCst);
}

/// Whether the LoRA preview fetch is currently running.
fn lora_thumbs_fetching() -> bool {
    LORA_THUMB_RUNNING.load(Ordering::SeqCst)
}

/// Background pass fetching Civitai preview thumbnails for LoRAs without one:
/// SHA-256 of the file → Civitai's by-hash endpoint → first still preview,
/// saved to the thumbs cache. Single-flight; repaints as each thumb lands.
/// Runs only while the picker is open (see [`cancel_lora_thumb_fetch`]). `force`
/// re-attempts files previously marked "not on Civitai" (the download button).
fn spawn_lora_thumb_fetch(files: Vec<String>, ctx: egui::Context, force: bool) {
    // (Re)opening the picker un-cancels, which also lets a still-running worker
    // from a quick close/reopen resume instead of dying.
    THUMB_FETCH_CANCEL.store(false, Ordering::SeqCst);
    let tdir = lora_thumbs_dir();
    let todo: Vec<String> = files
        .into_iter()
        .filter(|f| {
            let stem = f.trim_end_matches(".safetensors");
            let have_img = tdir.join(format!("{stem}.img")).exists();
            let have_none = tdir.join(format!("{stem}.none")).exists();
            !have_img && (force || !have_none)
        })
        .collect();
    if todo.is_empty() || LORA_THUMB_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(move || {
        // Clear RUNNING on every exit path (incl. a panic), so one bad file
        // can't permanently wedge thumbnail fetching for the session.
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                LORA_THUMB_RUNNING.store(false, Ordering::SeqCst);
            }
        }
        let _reset = Reset;

        let dir = comfy_base().join("ComfyUI").join("models").join("loras");
        let _ = std::fs::create_dir_all(&tdir);
        for f in todo {
            if THUMB_FETCH_CANCEL.load(Ordering::SeqCst) {
                break; // picker closed — finish later, when it reopens
            }
            let stem = f.trim_end_matches(".safetensors");
            // Hashing reads the whole file (LoRAs are ~100-300 MB) — that's why
            // this runs here and the result is cached forever.
            let bytes = crate::scan::sha256_file(&dir.join(&f))
                .and_then(|sha| crate::civitai::preview_image_by_hash(&sha));
            let write = match &bytes {
                Some(b) => std::fs::write(tdir.join(format!("{stem}.img")), b),
                None => std::fs::write(tdir.join(format!("{stem}.none")), b""),
            };
            let _ = write;
            ctx.request_repaint();
        }
    });
}

/// A LoRA card frame (the Civitai-card look), highlighted with an accent border
/// + faint accent wash when selected — selection is shown by the whole card, not
/// a checkbox.
fn lora_card(ui: &mut egui::Ui, selected: bool, add: impl FnOnce(&mut egui::Ui)) -> egui::Response {
    let (fill, stroke) = if selected {
        (
            lerp_color(FIELD(), ACCENT1(), 0.14),
            egui::Stroke::new(2.0, ACCENT1()),
        )
    } else {
        (FIELD(), egui::Stroke::new(1.0, EDGE()))
    };
    let r = egui::Frame::new()
        .fill(fill)
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::same(10))
        .stroke(stroke)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
    ui.add_space(4.0);
    r.response
}

/// Blend `a` toward `b` by `t` (0..=1), per channel.
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(l(a.r(), b.r()), l(a.g(), b.g()), l(a.b(), b.b()), a.a())
}

/// Render one LoRA as a Civitai-style card: preview on the left, a top-right ℹ
/// action slot (same place + 20px size as the Civitai panel's Download arrow),
/// the name, and a weight slider when selected. The whole card is clickable —
/// selection is shown by an accent border (no checkbox). `off_family` flags
/// LoRAs trained for a different model than the current tab, marking them with a
/// coloured base-family dot.
fn lora_row(
    ui: &mut egui::Ui,
    l: &mut LoraEntry,
    loras_dir: &Path,
    tdir: &Path,
    off_family: bool,
) {
    // Pick up a freshly downloaded thumbnail from the cache.
    if l.thumb.is_none() && !l.thumb_missing {
        let stem = l.file.trim_end_matches(".safetensors");
        let img = tdir.join(format!("{stem}.img"));
        if img.exists() {
            match std::fs::read(&img).ok().and_then(|b| crate::civitai::decode_thumb(&b)) {
                Some(ci) => {
                    l.thumb = Some(ui.ctx().load_texture(
                        format!("lora_thumb_{stem}"),
                        ci,
                        Default::default(),
                    ));
                }
                None => l.thumb_missing = true, // corrupt download
            }
        } else if tdir.join(format!("{stem}.none")).exists() {
            l.thumb_missing = true;
        }
    }

    let mut info_rect: Option<egui::Rect> = None;
    let mut copy_rect: Option<egui::Rect> = None;
    let mut slider_rect: Option<egui::Rect> = None;
    let inner = lora_card(ui, l.selected, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            if let Some(tex) = l.thumb.clone() {
                ui.add(
                    egui::Image::new(&tex)
                        .max_height(80.0)
                        .max_width(130.0)
                        .corner_radius(8),
                );
            }
            // Right slot added FIRST (reserves its width), then the name/weight
            // fill and wrap in what's left — same trick as the Civitai cards.
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_space(2.0);
                let tint = icon_tint(if l.info_open { TEXT() } else { MUTED() });
                let r = ui
                    .add(
                        egui::Image::new(egui::include_image!("../icons/info.svg"))
                            .fit_to_exact_size(egui::vec2(20.0, 20.0))
                            .tint(tint)
                            .sense(egui::Sense::click()),
                    )
                    .on_hover_text("Details & trigger words");
                if r.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                info_rect = Some(r.rect);
                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        // Off-family LoRAs get a coloured base dot + family label.
                        if off_family {
                            let (rect, dr) = ui
                                .allocate_exact_size(egui::vec2(10.0, 12.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 4.0, l.base.dot_color());
                            dr.on_hover_text(format!(
                                "Trained for {} — may not work on this model.",
                                l.base.label()
                            ));
                        }
                        let name = l.file.trim_end_matches(".safetensors");
                        let col = if l.selected { TEXT() } else { MUTED() };
                        ui.add(egui::Label::new(RichText::new(name).color(col).size(12.0)));
                        if off_family {
                            ui.label(RichText::new(l.base.label()).color(l.base.dot_color()).size(10.0));
                        } else if l.base == LoraBase::Unknown {
                            ui.label(RichText::new("base?").color(MUTED()).size(10.0)).on_hover_text(
                                "Couldn't identify this LoRA's base model from its file — it may not be made for this model.",
                            );
                        }
                    });
                    if l.selected {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("weight").color(MUTED()).size(10.5));
                            let sr = ui.add(egui::Slider::new(&mut l.strength, 0.0..=2.0));
                            slider_rect = Some(sr.rect);
                        });
                    }
                });
            });
        });
        if l.info_open {
            if let Some(info) = &l.info {
                ui.add_space(4.0);
                if info.triggers.is_empty() && info.facts.is_empty() {
                    ui.label(RichText::new("No metadata in this file.").color(MUTED()).size(10.5));
                }
                if !info.triggers.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Trigger words").color(TEXT()).strong().size(11.0));
                        // Image (not a Button) detected by position, since the
                        // card-level interact swallows child button clicks.
                        let cr = ui
                            .add(
                                egui::Image::new(egui::include_image!("../icons/copy.svg"))
                                    .fit_to_exact_size(egui::vec2(12.0, 12.0))
                                    .tint(icon_tint(MUTED()))
                                    .sense(egui::Sense::click()),
                            )
                            .on_hover_text("Copy trigger words");
                        if cr.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        copy_rect = Some(cr.rect);
                    });
                    ui.add(egui::Label::new(
                        RichText::new(info.triggers.join(", ")).color(MUTED()).size(11.0),
                    ));
                }
                for (k, v) in &info.facts {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{k}:")).color(MUTED()).size(10.5));
                        ui.add(egui::Label::new(RichText::new(v).color(TEXT()).size(10.5)));
                    });
                }
            }
        }
    });
    // Whole card clickable: the ℹ slot toggles details, the copy glyph copies
    // triggers, the weight slider keeps its own drag, and a click anywhere else
    // (de)selects the LoRA.
    let resp = inner.interact(egui::Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if resp.clicked() {
        let p = resp.interact_pointer_pos();
        let hit = |r: Option<egui::Rect>| matches!((p, r), (Some(p), Some(r)) if r.contains(p));
        if hit(info_rect) {
            l.info_open = !l.info_open;
            if l.info_open && l.info.is_none() {
                l.info = Some(read_lora_info(&loras_dir.join(&l.file)));
            }
        } else if hit(copy_rect) {
            if let Some(info) = &l.info {
                let _ = arboard::Clipboard::new().and_then(|mut c| c.set_text(info.triggers.join(", ")));
            }
        } else if hit(slider_rect) {
            // The slider handled its own drag.
        } else {
            l.selected = !l.selected;
        }
    }
}

/// The LoRA multi-select popup: an accent-highlighted card + weight slider per
/// installed LoRA.
fn lora_popup(ctx: &egui::Context, state: &mut GenerateState) {
    use crate::PopupPlacement;
    egui::Window::new("")
        .id(egui::Id::new("zimage_lora_popup"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .placed_centered(ctx)
        .frame(window_frame().corner_radius(CornerRadius::same(22)))
        .show(ctx, |ui| {
            // Only the top strip drags the popup — not stray drags on the body.
            crate::popup_drag_strip(ui, 30.0);
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
                // While the background fetch is pulling Civitai previews, show a
                // spinning download glyph so it's clear thumbnails are loading.
                if lora_thumbs_fetching() {
                    ui.add_space(4.0);
                    fetch_indicator(ui);
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // click_and_drag so a click that slips a pixel is swallowed
                    // by the button instead of dragging the popup.
                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/close.svg"))
                                .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                .tint(TEXT()),
                        ).frame(false).sense(egui::Sense::click_and_drag()))
                        .on_hover_text("Close")
                        .clicked()
                    {
                        state.show_lora_popup = false;
                        cancel_lora_thumb_fetch();
                    }
                    ui.add_space(2.0);
                    // One-shot spin animation: 0.5s ease-out single turn on click.
                    let spin = match state.lora_refresh_spin {
                        Some(start) => {
                            let t = start.elapsed().as_secs_f32() / 0.5;
                            if t >= 1.0 {
                                state.lora_refresh_spin = None;
                                0.0
                            } else {
                                ui.ctx().request_repaint();
                                let ease = 1.0 - (1.0 - t).powi(3); // ease-out
                                ease * std::f32::consts::TAU
                            }
                        }
                        None => 0.0,
                    };
                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/refresh.svg"))
                                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                .rotate(spin, egui::Vec2::splat(0.5))
                                .tint(icon_tint(TEXT())),
                        ).frame(false).sense(egui::Sense::click_and_drag()))
                        .on_hover_text("Refresh")
                        .clicked()
                    {
                        state.lora_refresh_spin = Some(std::time::Instant::now());
                        refresh_loras(state, ui.ctx());
                    }
                });
            });
            ui.add_space(14.0);

            // LoRAs for this tab's model family are listed first (unknowns are
            // offered too — they may just lack metadata). Those for other models
            // stay hidden until the "Show LoRAs for other models" checkbox below
            // is ticked, then appear under a divider, flagged with a coloured
            // base-family dot.
            let family = state.family;
            if state.loras.is_empty() {
                ui.label(
                    RichText::new("No LoRAs found. Drop .safetensors files into models/loras/.")
                        .color(MUTED())
                        .size(12.0),
                );
            } else {
                let loras_dir = comfy_base().join("ComfyUI").join("models").join("loras");
                let shown = state.loras.iter().filter(|l| l.base.matches(family)).count();
                let hidden = state.loras.len() - shown;
                let show_other = state.show_other_loras;
                egui::ScrollArea::vertical().max_height(360.0).auto_shrink([false, false]).show(ui, |ui| {
                    let tdir = lora_thumbs_dir();
                    if shown == 0 {
                        ui.label(
                            RichText::new(format!("No {} LoRAs found.", family.title()))
                                .color(MUTED())
                                .size(12.0),
                        );
                        ui.add_space(6.0);
                    }
                    for l in state.loras.iter_mut().filter(|l| l.base.matches(family)) {
                        lora_row(ui, l, &loras_dir, &tdir, false);
                    }
                    if hidden > 0 && show_other {
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(format!("For other models ({hidden})"))
                                .color(MUTED())
                                .strong()
                                .size(11.0),
                        );
                        ui.add_space(6.0);
                        for l in state.loras.iter_mut().filter(|l| !l.base.matches(family)) {
                            lora_row(ui, l, &loras_dir, &tdir, true);
                        }
                    }
                });
                // Reveal/hide the other-model LoRAs (where the "… hidden" note used
                // to be).
                if hidden > 0 {
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.checkbox(
                            &mut state.show_other_loras,
                            RichText::new(format!("Show {hidden} LoRA(s) for other models"))
                                .color(MUTED())
                                .size(11.5),
                        );
                        // Only once the others are revealed: a warning glyph that
                        // pops a disclaimer about cross-architecture LoRAs.
                        if state.show_other_loras {
                            let warn = ui
                                .add(
                                    egui::Image::new(egui::include_image!("../icons/warning.svg"))
                                        .fit_to_exact_size(egui::vec2(16.0, 16.0))
                                        .tint(Color32::from_rgb(235, 150, 45))
                                        .sense(egui::Sense::click()),
                                )
                                .on_hover_text("Why these may not work");
                            if warn.hovered() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                            egui::Popup::from_toggle_button_response(&warn)
                                .align(egui::RectAlign::TOP_START)
                                .width(300.0)
                                .gap(6.0)
                                .frame(window_frame())
                                .show(|ui| {
                                    ui.label(RichText::new("Heads up").color(TEXT()).strong().size(12.5));
                                    ui.add_space(4.0);
                                    ui.label(
                                        RichText::new(
                                            "LoRAs are tied to a model's architecture. One trained \
                                             for another base (SD 1.5, SDXL, Z-Image, …) won't apply \
                                             to this model — ComfyUI loads it but none of its weights \
                                             match, so it has no effect on the image (you'll just see \
                                             \"lora key not loaded\" warnings in the log).\n\nThey're \
                                             shown here only so you can pick one you know is \
                                             mislabelled.",
                                        )
                                        .color(MUTED())
                                        .size(11.0),
                                    );
                                });
                        }
                    });
                }
            }

            ui.add_space(10.0);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let done = egui::Button::new(RichText::new("Done").color(Color32::WHITE).strong()).fill(ACCENT1());
                if ui.add_sized(egui::vec2(90.0, 30.0), done).clicked() {
                    state.show_lora_popup = false;
                    cancel_lora_thumb_fetch();
                }
            });
        });
}

/// The embeddings (textual-inversion) picker. Mirrors the LoRA popup, but each
/// row inserts an `embedding:<name>` token into the prompt on click (embeddings
/// are prompt tokens, not separate loader nodes). Compatible embeddings list
/// first; off-family ones hide behind a checkbox.
fn embeddings_popup(ctx: &egui::Context, state: &mut GenerateState) {
    use crate::PopupPlacement;
    egui::Window::new("")
        .id(egui::Id::new("embeddings_popup"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .placed_centered(ctx)
        .frame(window_frame().corner_radius(CornerRadius::same(22)))
        .show(ctx, |ui| {
            crate::popup_drag_strip(ui, 30.0);
            ui.set_width(380.0);
            let radius = CornerRadius::same(10);
            {
                let v = ui.visuals_mut();
                v.widgets.inactive.corner_radius = radius;
                v.widgets.hovered.corner_radius = radius;
                v.widgets.active.corner_radius = radius;
            }

            // Title row: icon + title + Refresh + close ✕.
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/plugin.svg"))
                        .fit_to_exact_size(egui::vec2(20.0, 20.0))
                        .tint(TEXT()),
                );
                ui.heading(RichText::new("Embeddings").color(TEXT()).strong().size(17.0));
                if embeddings_fetching() {
                    ui.add_space(4.0);
                    fetch_indicator(ui);
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/close.svg"))
                                .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                .tint(TEXT()),
                        ).frame(false).sense(egui::Sense::click_and_drag()))
                        .on_hover_text("Close")
                        .clicked()
                    {
                        state.show_embeddings_popup = false;
                        cancel_embedding_thumb_fetch();
                    }
                    ui.add_space(2.0);
                    // One-shot 0.5s ease-out spin on click (matches the LoRA popup).
                    let spin = match state.embeddings_refresh_spin {
                        Some(start) => {
                            let t = start.elapsed().as_secs_f32() / 0.5;
                            if t >= 1.0 {
                                state.embeddings_refresh_spin = None;
                                0.0
                            } else {
                                ui.ctx().request_repaint();
                                let ease = 1.0 - (1.0 - t).powi(3);
                                ease * std::f32::consts::TAU
                            }
                        }
                        None => 0.0,
                    };
                    if ui
                        .add(egui::Button::image(
                            egui::Image::new(egui::include_image!("../icons/refresh.svg"))
                                .fit_to_exact_size(egui::vec2(20.0, 20.0))
                                .rotate(spin, egui::Vec2::splat(0.5))
                                .tint(icon_tint(TEXT())),
                        ).frame(false).sense(egui::Sense::click_and_drag()))
                        .on_hover_text("Refresh")
                        .clicked()
                    {
                        state.embeddings_refresh_spin = Some(std::time::Instant::now());
                        refresh_embeddings(state, ui.ctx());
                    }
                });
            });
            ui.add_space(6.0);
            ui.label(
                RichText::new("Select embeddings, then set each to Positive or Negative. They're applied at generation — not added to your prompt.")
                    .color(MUTED())
                    .size(11.0),
            );
            ui.add_space(10.0);

            let family = state.family;
            if state.embeddings.is_empty() {
                ui.label(
                    RichText::new("No embeddings found. Drop files into models/embeddings/.")
                        .color(MUTED())
                        .size(12.0),
                );
            } else {
                let shown = state.embeddings.iter().filter(|e| e.base.matches(family)).count();
                let hidden = state.embeddings.len() - shown;
                let show_other = state.show_other_embeddings;
                egui::ScrollArea::vertical().max_height(360.0).auto_shrink([false, false]).show(ui, |ui| {
                    if shown == 0 {
                        ui.label(
                            RichText::new(format!("No {} embeddings found.", family.title()))
                                .color(MUTED())
                                .size(12.0),
                        );
                        ui.add_space(6.0);
                    }
                    for e in state.embeddings.iter_mut().filter(|e| e.base.matches(family)) {
                        embedding_row(ui, e, false);
                    }
                    if hidden > 0 && show_other {
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(format!("For other models ({hidden})"))
                                .color(MUTED())
                                .strong()
                                .size(11.0),
                        );
                        ui.add_space(6.0);
                        for e in state.embeddings.iter_mut().filter(|e| !e.base.matches(family)) {
                            embedding_row(ui, e, true);
                        }
                    }
                });
                if hidden > 0 {
                    ui.add_space(6.0);
                    ui.checkbox(
                        &mut state.show_other_embeddings,
                        RichText::new(format!("Show {hidden} embedding(s) for other models"))
                            .color(MUTED())
                            .size(11.5),
                    );
                }
            }

            ui.add_space(10.0);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let done = egui::Button::new(RichText::new("Done").color(Color32::WHITE).strong()).fill(ACCENT1());
                if ui.add_sized(egui::vec2(90.0, 30.0), done).clicked() {
                    state.show_embeddings_popup = false;
                    cancel_embedding_thumb_fetch();
                }
            });
        });
}

/// One embedding row: a LoRA-style card. A click anywhere (de)selects it; when
/// selected, a Positive / Negative pill pair appears — clicking a pill assigns the
/// embedding to that encoder. Off-family entries get a coloured base dot + label.
fn embedding_row(ui: &mut egui::Ui, e: &mut EmbeddingEntry, off_family: bool) {
    let name = embed_stem(&e.file).to_string();

    // Pick up a freshly downloaded thumbnail from the cache (lazy, like LoRAs).
    if e.thumb.is_none() && !e.thumb_missing {
        let tdir = embedding_thumbs_dir();
        let stem = embed_stem(&e.file);
        let img = tdir.join(format!("{stem}.img"));
        if img.exists() {
            match std::fs::read(&img).ok().and_then(|b| crate::civitai::decode_thumb(&b)) {
                Some(ci) => {
                    e.thumb = Some(ui.ctx().load_texture(format!("embed_thumb_{stem}"), ci, Default::default()));
                }
                None => e.thumb_missing = true,
            }
        } else if tdir.join(format!("{stem}.none")).exists() {
            e.thumb_missing = true;
        }
    }

    let base = e.base;
    let selected = e.selected;
    let negative = e.negative;
    let thumb = e.thumb.clone();
    // Rects captured so the card-level click can route a hit to a control instead
    // of toggling selection (the card swallows child clicks).
    let mut pos_rect = None;
    let mut neg_rect = None;
    let mut slider_rect = None;

    let inner = lora_card(ui, selected, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            if let Some(tex) = thumb {
                ui.add(egui::Image::new(&tex).max_height(64.0).max_width(64.0).corner_radius(8));
            } else {
                ui.add(
                    egui::Image::new(egui::include_image!("../icons/plugin.svg"))
                        .fit_to_exact_size(egui::vec2(14.0, 14.0))
                        .tint(icon_tint(if selected { TEXT() } else { MUTED() })),
                );
            }
            ui.add(egui::Label::new(RichText::new(&name).color(TEXT()).size(12.5)).truncate());
            if off_family {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new(base.label()).color(base.dot_color()).size(10.5));
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 4.0, base.dot_color());
                });
            }
        });
        if selected {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                pos_rect = Some(polarity_pill(ui, "Positive", !negative, Color32::from_rgb(70, 160, 90)));
                neg_rect = Some(polarity_pill(ui, "Negative", negative, Color32::from_rgb(200, 80, 80)));
            });
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("weight").color(MUTED()).size(10.5));
                let sr = ui.add(egui::Slider::new(&mut e.strength, 0.0..=2.0));
                slider_rect = Some(sr.rect);
            });
        }
    });

    let resp = inner.interact(egui::Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    if resp.clicked() {
        let p = resp.interact_pointer_pos();
        let hit = |r: Option<egui::Rect>| matches!((p, r), (Some(p), Some(r)) if r.contains(p));
        if hit(slider_rect) {
            // The slider handled its own drag — don't toggle selection.
        } else if hit(pos_rect) {
            e.selected = true;
            e.negative = false;
        } else if hit(neg_rect) {
            e.selected = true;
            e.negative = true;
        } else {
            e.selected = !e.selected;
        }
    }
}

/// Draw a small rounded "Positive"/"Negative" pill (active = filled with `accent`,
/// inactive = faint). Returns its rect for the parent card's click hit-testing.
fn polarity_pill(ui: &mut egui::Ui, text: &str, active: bool, accent: Color32) -> egui::Rect {
    let font = egui::FontId::proportional(11.0);
    let galley = ui.fonts_mut(|f| f.layout_no_wrap(text.to_string(), font, if active { Color32::WHITE } else { MUTED() }));
    let pad = egui::vec2(10.0, 4.0);
    let (rect, _) = ui.allocate_exact_size(galley.size() + pad * 2.0, egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let fill = if active { accent } else { PANEL().gamma_multiply(1.4) };
        ui.painter().rect_filled(rect, CornerRadius::same(9), fill);
        if !active {
            ui.painter().rect_stroke(rect, CornerRadius::same(9), egui::Stroke::new(1.0, EDGE()), egui::StrokeKind::Inside);
        }
        ui.painter().galley(rect.min + pad, galley, Color32::WHITE);
    }
    rect
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

/// The installed ComfyUI version, read from its `comfyui_version.py`
/// (`__version__ = "x.y.z"`). `None` if not installed or unparsable.
pub fn comfyui_installed_version() -> Option<String> {
    let text = std::fs::read_to_string(comfy_base().join("ComfyUI").join("comfyui_version.py")).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("__version__") {
            // grab the contents of the first quoted string on the line
            let bytes = line.as_bytes();
            if let Some(q1) = line.find(['"', '\'']) {
                let quote = bytes[q1];
                if let Some(rel) = line[q1 + 1..].bytes().position(|b| b == quote) {
                    return Some(line[q1 + 1..q1 + 1 + rel].to_string());
                }
            }
        }
    }
    None
}

/// Re-download ComfyUI (master) and refresh the code in place, preserving the
/// user's `models/`, `custom_nodes/`, `user/`, `input/`, and `output/` dirs, then
/// re-install requirements. Returns true on success. Runs on a worker thread; all
/// progress goes through `send`. The running server (if any) is stopped first so
/// its files aren't locked on Windows.
pub fn update_comfyui(send: &dyn Fn(String)) -> bool {
    let base = comfy_base();
    let comfy = base.join("ComfyUI");
    if !comfy.join("main.py").exists() {
        send("ERROR: ComfyUI isn't installed yet — run Setup Requirements first.".into());
        return false;
    }

    // Stop the bundled server so its source files aren't held open while we copy.
    if let Ok(mut guard) = SERVER.lock() {
        if let Some(mut child) = guard.take() {
            send("== Stopping ComfyUI server…".into());
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    send("== Downloading latest ComfyUI…".into());
    let zip = base.join("comfyui_update.zip");
    if let Err(e) = download(COMFYUI_ZIP, &zip, send) {
        send(format!("ERROR: download failed: {e}"));
        return false;
    }
    // Extract to a temp dir so we can copy the code over selectively.
    let tmp = base.join("comfyui_update_tmp");
    let _ = std::fs::remove_dir_all(&tmp);
    if let Err(e) = unzip(&zip, &tmp) {
        send(format!("ERROR: extract failed: {e}"));
        let _ = std::fs::remove_file(&zip);
        return false;
    }
    let _ = std::fs::remove_file(&zip);
    let src = tmp.join("ComfyUI-master");
    if !src.join("main.py").exists() {
        send("ERROR: unexpected archive layout — aborting.".into());
        let _ = std::fs::remove_dir_all(&tmp);
        return false;
    }

    // Copy the fresh code over the install, but never touch the user's data dirs
    // (downloaded models, installed nodes, generated outputs).
    const KEEP: &[&str] = &["models", "custom_nodes", "user", "input", "output"];
    send("== Updating ComfyUI files…".into());
    if let Err(e) = copy_tree_skip(&src, &comfy, KEEP) {
        send(format!("ERROR: copy failed: {e}"));
        let _ = std::fs::remove_dir_all(&tmp);
        return false;
    }
    let _ = std::fs::remove_dir_all(&tmp);

    // Requirements may have changed between versions; refresh them.
    let py = if cfg!(windows) {
        base.join("python").join("python.exe")
    } else {
        base.join("python").join("bin").join("python3")
    };
    let py = py.to_string_lossy().to_string();
    let req = comfy.join("requirements.txt");
    if req.exists() {
        send("== Updating ComfyUI requirements…".into());
        let _ = run_streamed(&base, &py, &["-m", "pip", "install", "-r", &req.to_string_lossy()], send);
    }
    match comfyui_installed_version() {
        Some(v) => send(format!("== ComfyUI updated to {v}. It will relaunch on the next generation.")),
        None => send("== ComfyUI updated. It will relaunch on the next generation.".into()),
    }
    true
}

/// Recursively copy `src` into `dst`, overwriting, but skip any top-level entry
/// whose name is in `skip_top` (case-insensitive). Used to refresh ComfyUI's code
/// without clobbering the user's data dirs.
fn copy_tree_skip(src: &Path, dst: &Path, skip_top: &[&str]) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if skip_top.iter().any(|s| s.eq_ignore_ascii_case(&name.to_string_lossy())) {
            continue;
        }
        let (from, to) = (entry.path(), dst.join(&name));
        if from.is_dir() {
            copy_tree(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Recursively copy `src` into `dst`, overwriting existing files.
fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if from.is_dir() {
            copy_tree(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Render the Generate view into the right panel. `current_image` is the
/// browser's selected file, used as the source for LTX image-to-video (None for
/// the image families).
pub fn show(ui: &mut egui::Ui, state: &mut GenerateState, current_image: Option<&Path>) {
    // The whole view is one scrollable column. Normally everything fits exactly
    // (the prompt box is stretched so the Log row bottoms out at the panel
    // edge); expanding the log console overflows the panel and this scroll area
    // lets you scroll down to read it.
    let panel_h = ui.available_height();
    egui::ScrollArea::vertical()
        .id_salt(state.family.title()) // per-tab scroll position
        .auto_shrink([false, false])
        .show(ui, |ui| show_inner(ui, state, panel_h, current_image));
}

/// The view body. `fill_h` is the panel height to fill: the prompt box
/// stretches so the content below it ends exactly at that height.
fn show_inner(ui: &mut egui::Ui, state: &mut GenerateState, fill_h: f32, current_image: Option<&Path>) {
    let top_y = ui.cursor().min.y;
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

    // (The model picker sits inside the prompt box footer, next to the send
    // icon; Z-Image's LoRA picker is in the + menu there.)
    if state.model.gated() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(RichText::new("HF token (dev is gated)").color(MUTED()).size(11.0));
            ui.add(
                egui::Image::new(egui::include_image!("../icons/info.svg"))
                    .fit_to_exact_size(egui::vec2(14.0, 14.0))
                    .tint(icon_tint(MUTED())),
            )
            .on_hover_ui(|ui| {
                ui.set_max_width(260.0);
                ui.label("Flux dev is a gated model: accept its license on Hugging Face and paste a token.");
                crate::arrow_link(ui, "Manage tokens", "https://huggingface.co/settings/tokens", None);
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
    // Embeddings (textual inversion) picker.
    if state.show_embeddings_popup {
        embeddings_popup(ui.ctx(), state);
    }

    ui.add_space(10.0);

    // --- Prompt (stretched to the panel; long text scrolls inside). ---
    ui.label(RichText::new("Prompt").color(MUTED()).size(12.0));
    ui.add_space(2.0);
    // Both Flux and Z-Image dress the prompt box in the "Details & Actions" panel
    // styling (PANEL fill, rounded-22 corners, faint edge). No drop shadow: the
    // panel's shadow paints below the box and isn't clickable, which made the
    // textbox feel like you had to "click higher" to focus it.
    let prompt_frame = egui::Frame::new()
        .fill(PANEL())
        .corner_radius(CornerRadius::same(22))
        .inner_margin(Margin::same(12))
        .stroke(egui::Stroke::new(1.0, EDGE()));
    // The box stretches downward so the settings + Log row below it (height
    // measured last frame → state.below_h) bottom out exactly at the panel's
    // edge. 24 = the prompt frame's own top+bottom inner margins.
    let prompt_max_h = (fill_h - (ui.cursor().min.y - top_y) - state.below_h - 24.0).max(140.0);
    // A strip at the bottom of the box is reserved for the footer buttons
    // (+ menu / model selector / send) so prompt text never hides under them.
    let btn_strip = 34.0;
    // Enough rows to fill the box's visible height, so a click anywhere inside
    // lands on the text. (egui 0.34 ignores TextEdit::min_size's height — the
    // allocation comes from desired_rows — so the row count is the only lever.)
    let row_h = ui.text_style_height(&egui::TextStyle::Body);
    let prompt_rows = (((prompt_max_h - btn_strip - 4.0) / row_h).floor() as usize).max(4);
    prompt_frame
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            egui::ScrollArea::vertical()
                .id_salt("flux_prompt")
                .max_height(prompt_max_h - btn_strip)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // Video models are very prompt-length-sensitive — nudge toward
                    // long, motion-describing prompts.
                    let hint = if state.family.is_video() {
                        "Describe the scene AND the motion, in a few detailed sentences — video models need long prompts…"
                    } else {
                        "a cinematic photo of…"
                    };
                    let out = egui::TextEdit::multiline(&mut state.prompt)
                        .desired_width(f32::INFINITY)
                        .desired_rows(prompt_rows)
                        .frame(egui::Frame::NONE)
                        .hint_text(hint)
                        .show(ui);
                    // Live spell-check: red squiggles + right-click suggestions.
                    crate::spellcheck::attach(ui, &out, &mut state.prompt, &mut state.spell);
                });
            {
                // Footer buttons pinned inside the box's bottom corners.
                // Allocate an exact-height strip: a bare `with_layout` here would
                // claim all remaining height (unbounded inside the outer scroll
                // area) and balloon the prompt box to thousands of pixels.
                ui.allocate_ui_with_layout(egui::vec2(ui.available_width(), 28.0), Layout::left_to_right(Align::Center), |ui| {
                    // "+" menu in the bottom-LEFT corner (flips to an × while
                    // open) — home of the LoRA picker and future tools.
                    {
                        let menu_id = ui.id().with(("add_menu", state.family.title()));
                        let open = egui::Popup::is_id_open(ui.ctx(), menu_id);
                        let (icon, tip) = if open {
                            (egui::include_image!("../icons/close.svg"), "Close")
                        } else {
                            (egui::include_image!("../icons/add.svg"), "Add")
                        };
                        let add_resp = round_icon_button(ui, icon, 18.0, tip, true);
                        egui::Popup::menu(&add_resp)
                            .id(menu_id)
                            .align(egui::RectAlign::TOP_START)
                            .frame(menu_frame())
                            .show(|ui| {
                                ui.set_min_width(150.0);
                                round_menu_rows(ui);
                                let n = state.loras.iter().filter(|l| l.selected && l.base.matches(state.family)).count();
                                let label = if n > 0 { format!("LoRA ({n})") } else { "LoRA".to_string() };
                                let licon = egui::Image::new(egui::include_image!("../icons/lora.svg"))
                                    .fit_to_exact_size(egui::vec2(16.0, 16.0))
                                    .tint(icon_tint(TEXT()));
                                if ui.add(egui::Button::image_and_text(licon, RichText::new(label).size(12.0)).frame(false)).clicked() {
                                    refresh_loras(state, ui.ctx());
                                    state.show_lora_popup = true;
                                    ui.close();
                                }

                                // Embeddings (textual inversion): selected ones are
                                // injected into the workflow's pos/neg encoder.
                                let en = state.embeddings.iter().filter(|e| e.selected && e.base.matches(state.family)).count();
                                let elabel = if en > 0 { format!("Embeddings ({en})") } else { "Embeddings".to_string() };
                                let eicon = egui::Image::new(egui::include_image!("../icons/plugin.svg"))
                                    .fit_to_exact_size(egui::vec2(16.0, 16.0))
                                    .tint(icon_tint(TEXT()));
                                if ui.add(egui::Button::image_and_text(eicon, RichText::new(elabel).size(12.0)).frame(false)).clicked() {
                                    refresh_embeddings(state, ui.ctx());
                                    state.show_embeddings_popup = true;
                                    ui.close();
                                }
                            });
                    }

                    // Send / stop pinned to the bottom-RIGHT corner, with the
                    // compact model selector to their left.
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if state.running {
                            if round_icon_button(ui, egui::include_image!("../icons/stop.svg"), 18.0, "Stop generating", true).clicked() {
                                state.cancel.store(true, Ordering::SeqCst);
                                state.status = "Cancelling…".into();
                            }
                        } else if !state.prompt.trim().is_empty() {
                            // The send button only appears once you start typing.
                            let inst = is_installed();
                            // Image-to-video needs a selected source image.
                            let needs_img = state.family.is_video() && state.i2v && current_image.is_none();
                            let ready = inst && !needs_img;
                            let tip = if !inst {
                                "Run Setup Requirements first"
                            } else if needs_img {
                                "Select an image in the browser first"
                            } else {
                                "Generate"
                            };
                            if round_icon_button(ui, egui::include_image!("../icons/send.svg"), 18.0, tip, ready).clicked() && ready {
                                start_generate(state, ui.ctx(), current_image);
                            }
                        } else {
                            // Invisible stand-in so the model selector doesn't
                            // drift into the corner while the send icon is hidden.
                            ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::hover());
                        }
                        ui.add_space(2.0);
                        model_selector(ui, state);
                    });
                });
            }
        });
    let after_prompt_y = ui.cursor().min.y;

    ui.add_space(8.0);

    // --- Settings. ---
    slider(ui, "Steps", &mut state.steps, 1..=50);
    slider(ui, "Guidance (CFG)", &mut state.cfg, 1.0..=8.0);
    // Resolution: the image families (Z-Image, Flux) use a single "Size" slider +
    // quick aspect-ratio tiles (square/landscape/portrait) instead of fine
    // Width/Height sliders — the slider scales every preset live, and one click on
    // a tile picks the ratio. Video families keep the sliders.
    if matches!(state.family, GenFamily::ZImage | GenFamily::Flux | GenFamily::Sdxl | GenFamily::Anima) {
        // The Size slider scales all three presets; recompute the active dims so the
        // current selection tracks the slider live (snapped to multiples of 64).
        let mut size = state.size;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Size").color(MUTED()).size(12.0));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add(egui::Slider::new(&mut size, 512..=2048).step_by(64.0));
            });
        });
        if size != state.size {
            state.size = size;
            let idx = aspect_idx(state.width, state.height);
            let (w, h) = zimage_aspect_dims(idx, size);
            state.width = w;
            state.height = h;
        }
        aspect_selector(ui, state);
    } else {
        ui.horizontal(|ui| {
            slider(ui, "Width", &mut state.width, 512..=1536);
        });
        ui.horizontal(|ui| {
            slider(ui, "Height", &mut state.height, 512..=1536);
        });
    }
    // Video families (LTX): length + frame rate, and the text/image-to-video
    // mode toggle.
    if state.family.is_video() {
        slider(ui, "Frames", &mut state.frames, 25..=257);
        slider(ui, "FPS", &mut state.fps, 8..=30);
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().icon_width_inner = 11.0;
            if ui.radio(!state.i2v, "").clicked() {
                state.i2v = false;
            }
            ui.label(RichText::new("Text → video").color(TEXT()).size(12.0));
            ui.add_space(10.0);
            if ui.radio(state.i2v, "").clicked() {
                state.i2v = true;
            }
            ui.label(RichText::new("Image → video").color(TEXT()).size(12.0));
        });
        if state.i2v {
            let txt = match current_image {
                Some(p) => format!(
                    "Source: {}",
                    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                ),
                None => "Select an image in the browser to animate.".to_string(),
            };
            ui.label(RichText::new(txt).color(MUTED()).size(11.0));
        }
        ui.add_space(2.0);
    }
    ui.horizontal(|ui| {
        // A radio-style filled dot instead of a checkmark for the toggle.
        // Only the inner dot is enlarged; the outer circle keeps its default size.
        ui.spacing_mut().icon_width_inner = 11.0;
        if ui.radio(state.randomize_seed, "").clicked() {
            state.randomize_seed = !state.randomize_seed;
        }
        ui.label(RichText::new("Randomize seed").color(TEXT()).size(12.0));
    });
    if !state.randomize_seed {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Seed").color(MUTED()).size(12.0));
            ui.add(egui::DragValue::new(&mut state.seed));
        });
    }

    ui.add_space(10.0);

    // Generated images appear in the left browser + centre viewer while this view
    // is open (see ViewerApp::sync_flux_browser).
    if !state.gen_images.is_empty() {
        let noun = if state.family.is_video() { "video" } else { "image" };
        ui.label(
            RichText::new(format!(
                "{} {noun}(s) this session — browse them on the left.",
                state.gen_images.len()
            ))
            .color(MUTED())
            .size(11.0),
        );
        ui.add_space(8.0);
    }

    // --- Log (collapsible). ---
    ui.horizontal(|ui| {
        // SVG disclosure arrow: drop-down when open, right when collapsed.
        // `image_and_text` vertically centres the icon against the label.
        let arrow_src = if state.show_log {
            egui::include_image!("../icons/arrow_drop_down.svg")
        } else {
            egui::include_image!("../icons/arrow_right.svg")
        };
        // Fixed-size pill drawn by hand: the icon + label are painted centered
        // *inside* it, so the pill never moves or resizes with the content (egui's
        // Button auto-sizes to its content, which made it track the text).
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(46.0, 18.0), egui::Sense::click());
        let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
        if ui.is_rect_visible(rect) {
            let visuals = *ui.style().interact(&resp);
            let txt = visuals.text_color();
            ui.painter().rect(
                rect,
                CornerRadius::same(10), // pill (radius = half the 20px height)
                visuals.weak_bg_fill,
                visuals.bg_stroke,
                egui::StrokeKind::Inside,
            );

            // Lay out the "Log" label, then centre [icon | gap | text] as a group.
            let icon = 14.0_f32;
            let gap = 3.0_f32;
            let galley = ui.painter().layout_no_wrap("Log".to_owned(), egui::FontId::proportional(11.0), txt);
            let content_w = icon + gap + galley.size().x;
            let x0 = rect.center().x - content_w / 1.7;
            let cy = rect.center().y;

            let icon_rect = egui::Rect::from_min_size(egui::pos2(x0, cy - icon / 2.0), egui::vec2(icon, icon));
            egui::Image::new(arrow_src)
                .tint(icon_tint(txt))
                .paint_at(ui, icon_rect);
            ui.painter()
                .galley(egui::pos2(x0 + icon + gap, cy - galley.size().y / 2.0), galley, txt);
        }
        if resp.clicked() {
            state.show_log = !state.show_log;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            // Flash the icon green (copied) or red (failed) for ~0.9s after a click.
            const COPY_FLASH_SECS: f32 = 0.9;
            let flash_tint = match state.copy_flash {
                Some((when, ok)) if when.elapsed().as_secs_f32() < COPY_FLASH_SECS => {
                    ui.ctx().request_repaint(); // fade out smoothly / clear when done
                    Some(if ok { GREEN } else { RED })
                }
                _ => None,
            };
            let tint = flash_tint.unwrap_or_else(|| icon_tint(MUTED()));
            let copy_icon = egui::Image::new(egui::include_image!("../icons/copy.svg"))
                .fit_to_exact_size(egui::vec2(16.0, 16.0))
                .tint(tint);
            let copy = egui::Button::image(copy_icon).frame(false);
            let tip = match state.copy_flash {
                Some((_, true)) if flash_tint.is_some() => "Copied!",
                Some((_, false)) if flash_tint.is_some() => "Copy failed",
                _ => "Copy the log",
            };
            if ui.add_enabled(!state.log.is_empty(), copy).on_hover_text(tip).clicked() {
                // Use arboard directly so we get a success/failure result to flash on.
                let ok = arboard::Clipboard::new()
                    .and_then(|mut c| c.set_text(state.log.join("\n")))
                    .is_ok();
                state.copy_flash = Some((std::time::Instant::now(), ok));
            }
        });
    });
    // Remember how tall everything between the prompt box and the end of the Log
    // row is — next frame the Z-Image prompt stretches to push this row to the
    // panel's bottom edge (the open log console below is deliberately excluded:
    // it overflows and is reached by scrolling). Both cursor reads sit one
    // item-spacing past their widget, so the difference needs no correction.
    let below_h = ui.cursor().min.y - after_prompt_y;
    if (below_h - state.below_h).abs() > 0.5 {
        // Sizing settles one frame after a layout change — repaint to get there.
        state.below_h = below_h;
        ui.ctx().request_repaint();
    }
    if state.show_log {
        let bg = if is_light() { FIELD() } else { Color32::from_rgb(15, 15, 17) };
        egui::Frame::new()
            .fill(bg)
            .corner_radius(CornerRadius::same(22))
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

/// Rounded-22 card frame for the prompt-box popup menus (+ menu, model list),
/// matching the app's panel styling.
fn menu_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(CornerRadius::same(22))
        .inner_margin(Margin::same(12))
        .stroke(egui::Stroke::new(1.0, EDGE()))
}

/// Round the hover highlight of menu rows (egui's default is square).
fn round_menu_rows(ui: &mut egui::Ui) {
    let radius = CornerRadius::same(6);
    ui.visuals_mut().widgets.inactive.corner_radius = radius;
    ui.visuals_mut().widgets.hovered.corner_radius = radius;
}

/// Compact model selector for the prompt-box footer: the model name (truncated
/// with … past 150px — hover shows the full name) plus an up/down chevron.
/// Clicking opens the model list in a rounded-22 popup above the footer.
fn model_selector(ui: &mut egui::Ui, state: &mut GenerateState) {
    // When an installed checkpoint is selected, the footer shows its (extension-
    // stripped) filename; otherwise the built-in model's label.
    let full: String = match &state.checkpoint {
        Some(f) => ckpt_display_name(f).to_string(),
        None => state.model.label().to_string(),
    };
    let menu_id = ui.id().with(("model_menu", state.family.title()));
    let open = egui::Popup::is_id_open(ui.ctx(), menu_id);

    // Auto-detect installed checkpoints once each time the picker opens (cheap
    // header reads), so newly-downloaded checkpoints show up without a restart.
    if state.family.uses_checkpoint_picker() && open && !state.ckpt_menu_was_open {
        refresh_checkpoints(state);
    }
    state.ckpt_menu_was_open = open;

    let mut job = egui::text::LayoutJob::simple_singleline(full.clone(), egui::FontId::proportional(12.0), TEXT());
    job.wrap = egui::text::TextWrapping::truncate_at_width(150.0);
    let galley = ui.fonts_mut(|f| f.layout_job(job));
    let (arrow, gap) = (14.0, 2.0);
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(galley.size().x + gap + arrow, 28.0), egui::Sense::click());
    let resp = resp.on_hover_text(full.clone()).on_hover_cursor(egui::CursorIcon::PointingHand);
    if ui.is_rect_visible(rect) {
        let text_pos = egui::pos2(rect.left(), rect.center().y - galley.size().y / 2.0);
        ui.painter().galley(text_pos, galley, TEXT());
        let arrow_src = if open {
            egui::include_image!("../icons/arrow_up.svg")
        } else {
            egui::include_image!("../icons/arrow_down.svg")
        };
        let arrow_rect = egui::Rect::from_center_size(
            egui::pos2(rect.right() - arrow / 2.0, rect.center().y),
            egui::vec2(arrow, arrow),
        );
        egui::Image::new(arrow_src).tint(icon_tint(MUTED())).paint_at(ui, arrow_rect);
    }
    egui::Popup::menu(&resp)
        .id(menu_id)
        .align(egui::RectAlign::TOP_END)
        .frame(menu_frame())
        .show(|ui| {
            ui.set_min_width(240.0);
            round_menu_rows(ui);
            // Built-in (downloadable) model variants. Selecting one clears any
            // custom-checkpoint override.
            for &m in GenModel::all_for(state.family) {
                let selected = state.checkpoint.is_none() && state.model == m;
                if ui.selectable_label(selected, m.label()).clicked() {
                    state.model = m;
                    state.checkpoint = None;
                    state.steps = m.default_steps();
                    state.cfg = m.default_cfg();
                    ui.close();
                }
            }

            // Auto-detected installed checkpoints, filtered to this family (like
            // the LoRA picker). Only families that load a single swappable
            // checkpoint show this section.
            if state.family.uses_checkpoint_picker() {
                let family = state.family;
                let show_other = state.show_other_ckpts;
                let sel = state.checkpoint.clone();
                // Partition into compatible (this family / unknown) vs off-family.
                // Owned copies so the list isn't borrowed while we mutate `state`.
                let mut compatible: Vec<(String, LoraBase)> = Vec::new();
                let mut others: Vec<(String, LoraBase)> = Vec::new();
                let base_file = family.default_model_file();
                for c in &state.checkpoints {
                    // The built-in base is already listed above — don't duplicate it.
                    if c.file == base_file {
                        continue;
                    }
                    if c.base.matches(family) || c.base == LoraBase::Unknown {
                        compatible.push((c.file.clone(), c.base));
                    } else {
                        others.push((c.file.clone(), c.base));
                    }
                }

                let mut pick: Option<String> = None;
                if !compatible.is_empty() || (!others.is_empty() && show_other) {
                    ui.separator();
                    ui.label(RichText::new("Installed checkpoints").color(MUTED()).size(11.0));
                }
                for (file, _) in &compatible {
                    let selected = sel.as_deref() == Some(file.as_str());
                    if ui.selectable_label(selected, ckpt_display_name(file)).clicked() {
                        pick = Some(file.clone());
                    }
                }
                if show_other {
                    for (file, base) in &others {
                        let selected = sel.as_deref() == Some(file.as_str());
                        let label = format!("{}  ·  {}", ckpt_display_name(file), base.label());
                        let resp = ui.selectable_label(selected, RichText::new(label).color(MUTED()));
                        if resp.clicked() {
                            pick = Some(file.clone());
                        }
                    }
                }
                if let Some(file) = pick {
                    // A custom checkpoint runs through the family's default model
                    // node config; keep that model's default steps/cfg.
                    let m = family.default_model();
                    state.model = m;
                    state.steps = m.default_steps();
                    state.cfg = m.default_cfg();
                    state.checkpoint = Some(file);
                    ui.close();
                }

                if !others.is_empty() {
                    ui.separator();
                    let mut v = state.show_other_ckpts;
                    if ui.checkbox(&mut v, RichText::new("Show other-family checkpoints").size(11.0)).changed() {
                        state.show_other_ckpts = v;
                    }
                }
            }
        });
}

/// A checkpoint's display name: the filename with its `.safetensors` / `.ckpt`
/// extension stripped.
fn ckpt_display_name(file: &str) -> &str {
    file.trim_end_matches(".safetensors").trim_end_matches(".ckpt")
}

/// A bare icon button on a 28px click target: the SVG is painted dead-centre,
/// tinted to the theme's text colour (dark in light mode, white in dark mode,
/// pink on Aurora — the white-authored SVGs take the tint 1:1). Dims when
/// disabled.
fn round_icon_button(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'_>,
    icon_size: f32,
    tip: &str,
    enabled: bool,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::click());
    let resp = resp.on_hover_text(tip);
    let resp = if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp };
    if ui.is_rect_visible(rect) {
        let tint = icon_tint(TEXT());
        let tint = if enabled { tint } else { tint.gamma_multiply(0.45) };
        egui::Image::new(icon)
            .tint(tint)
            .paint_at(ui, egui::Rect::from_center_size(rect.center(), egui::vec2(icon_size, icon_size)));
    }
    resp
}

/// A non-interactive Gelbooru-style "downloading previews" indicator: a blue
/// filled circle with a white down-arrow, wrapped in an animated spinning ring.
/// Shown in the LoRA / embeddings picker titles only while the background preview
/// fetch is running, so it's clear thumbnails are being pulled automatically.
fn fetch_indicator(ui: &mut egui::Ui) -> egui::Response {
    let size = 24.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let resp = resp.on_hover_text("Downloading preview cards…");
    let painter = ui.painter().clone();
    let center = rect.center();

    // Filled blue circle + white down-arrow.
    painter.circle_filled(center, 9.0, ACCENT1());
    let icon = 13.0;
    egui::Image::new(egui::include_image!("../icons/Arrow Downward Alt.svg"))
        .tint(Color32::WHITE)
        .paint_at(ui, egui::Rect::from_center_size(center, egui::vec2(icon, icon)));
    // Spinning arc around it (this is only drawn while a fetch is in flight).
    let radius = 11.0;
    let t = ui.input(|i| i.time) as f32;
    painter.circle_stroke(center, radius, egui::Stroke::new(2.0, ACCENT1().gamma_multiply(0.25)));
    let start = (t * 3.0) % std::f32::consts::TAU;
    let sweep = std::f32::consts::PI * 0.6;
    let pts: Vec<egui::Pos2> = (0..=24)
        .map(|k| {
            let a = start + sweep * (k as f32 / 24.0);
            center + radius * egui::vec2(a.cos(), a.sin())
        })
        .collect();
    painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0, ACCENT1())));
    ui.ctx().request_repaint();
    resp
}

/// The aspect tile index for a (width, height): 0 square, 1 landscape, 2 portrait.
/// Matched by orientation so a tweaked size still selects the right tile.
fn aspect_idx(width: i32, height: i32) -> usize {
    match width.cmp(&height) {
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => 2,
    }
}

/// Z-Image: the (width, height) for an aspect tile at the reference `size`.
/// `idx`: 0 square, 1 landscape, 2 portrait. Square is `size`² exactly; the others
/// scale the 1216×832 (≈3:2) bucket proportionally, snapped to multiples of 64.
fn zimage_aspect_dims(idx: usize, size: i32) -> (i32, i32) {
    let snap = |v: f32| (((v / 64.0).round() as i32).max(1)) * 64;
    let scale = size as f32 / 1024.0;
    match idx {
        1 => (snap(1216.0 * scale), snap(832.0 * scale)),
        2 => (snap(832.0 * scale), snap(1216.0 * scale)),
        _ => (size, size),
    }
}

/// Z-Image resolution picker: three equal-width tiles (square / landscape /
/// portrait) replacing the Width/Height sliders. Each shows its live pixel size
/// (scaled by the Size slider) and sets the dimensions on click; the tile matching
/// the current orientation is highlighted with the accent border.
fn aspect_selector(ui: &mut egui::Ui, state: &mut GenerateState) {
    let tiles = [
        (egui::include_image!("../icons/square.svg"), "Square"),
        (egui::include_image!("../icons/landscape.svg"), "Landscape"),
        (egui::include_image!("../icons/portrait.svg"), "Portrait"),
    ];
    let active = aspect_idx(state.width, state.height);

    ui.label(RichText::new("Aspect ratio").color(MUTED()).size(12.0));
    ui.add_space(4.0);
    ui.columns(3, |cols| {
        for (i, (icon, label)) in tiles.into_iter().enumerate() {
            let (w, h) = zimage_aspect_dims(i, state.size);
            if aspect_tile(&mut cols[i], icon, label, &format!("{w}×{h}"), i == active) {
                state.width = w;
                state.height = h;
            }
        }
    });
}

/// One aspect-ratio tile: a card-framed icon + label + live `dims` text that
/// highlights when active (same accent treatment as the LoRA cards). Returns true
/// on click.
fn aspect_tile(ui: &mut egui::Ui, icon: egui::ImageSource<'_>, label: &str, dims: &str, selected: bool) -> bool {
    let (fill, stroke) = if selected {
        (lerp_color(FIELD(), ACCENT1(), 0.14), egui::Stroke::new(2.0, ACCENT1()))
    } else {
        (FIELD(), egui::Stroke::new(1.0, EDGE()))
    };
    let inner = egui::Frame::new()
        .fill(fill)
        .corner_radius(CornerRadius::same(12))
        .inner_margin(Margin::symmetric(8, 10))
        .stroke(stroke)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| {
                let col = if selected { TEXT() } else { MUTED() };
                ui.add(
                    egui::Image::new(icon)
                        .fit_to_exact_size(egui::vec2(26.0, 26.0))
                        .tint(icon_tint(col)),
                );
                ui.add_space(3.0);
                ui.label(RichText::new(label).color(col).size(11.0));
                ui.label(RichText::new(dims).color(MUTED()).size(9.5));
            });
        });
    let resp = inner.response.interact(egui::Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp.clicked()
}

/// Label + right-aligned slider row; works for both the f32 and i32 settings.
fn slider<N: egui::emath::Numeric>(ui: &mut egui::Ui, label: &str, value: &mut N, range: std::ops::RangeInclusive<N>) {
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

/// The model files a variant needs at generation time, as (sub-dir, filename)
/// under ComfyUI's `models/` dir. Mirrors exactly what `run_setup` downloads — the
/// single source of truth for the pre-flight check.
fn required_files(model: GenModel) -> Vec<(&'static str, String)> {
    match model.family() {
        GenFamily::Flux => vec![
            ("clip", "clip_l.safetensors".into()),
            ("clip", "t5xxl_fp8_e4m3fn.safetensors".into()),
            ("vae", "ae.safetensors".into()),
            ("unet", model.unet_file().to_string()),
        ],
        GenFamily::ZImage => vec![
            ("text_encoders", model.zimage_te_file().to_string()),
            ("diffusion_models", "z_image_turbo_bf16.safetensors".into()),
            ("vae", "ae_zimage.safetensors".into()),
        ],
        GenFamily::Ltx if model == GenModel::Ltx2Distilled => vec![
            ("checkpoints", "ltx-2.3-22b-distilled-fp8.safetensors".into()),
            ("text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors".into()),
            ("latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors".into()),
        ],
        GenFamily::Ltx => vec![
            ("checkpoints", "ltxv-2b-0.9.6-distilled-04-25.safetensors".into()),
            ("text_encoders", "t5xxl_fp8_e4m3fn.safetensors".into()),
            ("latent_upscale_models", "ltxv-spatial-upscaler-0.9.7.safetensors".into()),
        ],
        GenFamily::Wan => vec![
            ("diffusion_models", "wan2.2_ti2v_5B_fp16.safetensors".into()),
            ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors".into()),
            ("vae", "wan2.2_vae.safetensors".into()),
        ],
        GenFamily::Sdxl => vec![("checkpoints", "sd_xl_base_1.0.safetensors".into())],
        GenFamily::Anima => vec![
            ("diffusion_models", "anima-base-v1.0.safetensors".into()),
            ("text_encoders", "qwen_3_06b_base.safetensors".into()),
            ("vae", "qwen_image_vae.safetensors".into()),
        ],
    }
}

/// Filenames `model` needs that aren't on disk yet — empty means good to generate.
fn missing_files(model: GenModel, custom_ckpt: Option<&str>) -> Vec<String> {
    let models = comfy_base().join("ComfyUI").join("models");
    // A user-picked installed checkpoint can live in any of the checkpoint dirs
    // (we register them all with ComfyUI), so look for it across the whole set.
    if let Some(c) = custom_ckpt {
        let present = CKPT_DIRS.iter().any(|sub| models.join(sub).join(c).exists());
        let mut missing: Vec<String> = if present { Vec::new() } else { vec![c.to_string()] };
        // The custom file replaces the family's swappable model entry; still verify
        // the other required files (text encoder, VAE, …) for this family.
        let model_sub = model.family().model_subdir();
        for (sub, file) in required_files(model) {
            if sub != model_sub && !file.is_empty() && !models.join(sub).join(&file).exists() {
                missing.push(file);
            }
        }
        return missing;
    }
    required_files(model)
        .into_iter()
        .filter(|(sub, file)| !file.is_empty() && !models.join(sub).join(file).exists())
        .map(|(_, file)| file)
        .collect()
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
            status("Downloading Z-Image model + encoders…");
            let _ = std::fs::create_dir_all(m("text_encoders"));
            let _ = std::fs::create_dir_all(m("diffusion_models"));
            let _ = std::fs::create_dir_all(m("vae"));
            // Fetch BOTH encoders (fp8 + bf16) so either variant works after a
            // single Setup — they share the same diffusion model + VAE, and the
            // only difference between the two tab options is which one is loaded.
            for te in [
                GenModel::ZImageFast.zimage_te_file(),
                GenModel::ZImageQuality.zimage_te_file(),
            ] {
                let te_url = format!(
                    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/{te}?download=true"
                );
                ok &= fetch(&te_url, m("text_encoders").join(te), te, "", &send);
            }
            ok &= fetch(ZIMAGE_DIFFUSION, m("diffusion_models").join("z_image_turbo_bf16.safetensors"), "Z-Image diffusion", "", &send);
            ok &= fetch(ZIMAGE_VAE, m("vae").join("ae_zimage.safetensors"), "Z-Image VAE", "", &send);
        }
        GenFamily::Ltx if model == GenModel::Ltx2Distilled => {
            // LTX-2.3 22B quality option (~40 GB). The full checkpoint bundles the
            // video+audio VAE, so only checkpoint + encoder + upscaler are needed.
            status("Downloading LTX-2.3 (22B) — large, ~40 GB…");
            let _ = std::fs::create_dir_all(m("checkpoints"));
            let _ = std::fs::create_dir_all(m("text_encoders"));
            let _ = std::fs::create_dir_all(m("latent_upscale_models"));
            ok &= fetch(LTX2_CKPT, m("checkpoints").join("ltx-2.3-22b-distilled-fp8.safetensors"), "LTX-2.3 22B checkpoint", "", &send);
            ok &= fetch(LTX2_GEMMA, m("text_encoders").join("gemma_3_12B_it_fp4_mixed.safetensors"), "Gemma-3 12B encoder", "", &send);
            ok &= fetch(LTX2_UPSCALER, m("latent_upscale_models").join("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"), "LTX-2.3 spatial upscaler", "", &send);
        }
        GenFamily::Ltx => {
            status("Downloading LTX-Video model + encoder…");
            let _ = std::fs::create_dir_all(m("checkpoints"));
            let _ = std::fs::create_dir_all(m("text_encoders"));
            let _ = std::fs::create_dir_all(m("latent_upscale_models"));
            ok &= fetch(LTX_CKPT, m("checkpoints").join("ltxv-2b-0.9.6-distilled-04-25.safetensors"), "LTX checkpoint", "", &send);
            ok &= fetch(T5XXL_FP8, m("text_encoders").join("t5xxl_fp8_e4m3fn.safetensors"), "t5xxl_fp8", "", &send);
            ok &= fetch(LTX_UPSCALER, m("latent_upscale_models").join("ltxv-spatial-upscaler-0.9.7.safetensors"), "LTX spatial upscaler", "", &send);
            // Video output uses ComfyUI's native SaveWEBM node — no custom node.
        }
        GenFamily::Wan => {
            // Wan 2.2 TI2V 5B — one model + the umt5 encoder + the Wan VAE, all
            // native ComfyUI nodes. Both tab variants (fast/quality) share these
            // files; only the load-time precision differs.
            status("Downloading Wan 2.2 (5B) model + encoder…");
            let _ = std::fs::create_dir_all(m("diffusion_models"));
            let _ = std::fs::create_dir_all(m("text_encoders"));
            let _ = std::fs::create_dir_all(m("vae"));
            ok &= fetch(WAN_TI2V_5B, m("diffusion_models").join("wan2.2_ti2v_5B_fp16.safetensors"), "Wan 2.2 5B model", "", &send);
            ok &= fetch(WAN_UMT5, m("text_encoders").join("umt5_xxl_fp8_e4m3fn_scaled.safetensors"), "umt5-xxl encoder", "", &send);
            ok &= fetch(WAN_VAE, m("vae").join("wan2.2_vae.safetensors"), "Wan 2.2 VAE", "", &send);
        }
        GenFamily::Sdxl => {
            // SDXL Base 1.0 — a single checkpoint (UNet + dual CLIP + VAE).
            status("Downloading SDXL Base 1.0 (~6.9 GB)…");
            let _ = std::fs::create_dir_all(m("checkpoints"));
            ok &= fetch(SDXL_CKPT, m("checkpoints").join("sd_xl_base_1.0.safetensors"), "SDXL Base 1.0", "", &send);
        }
        GenFamily::Anima => {
            // Anima Base v1.0 — diffusion model + Qwen3 0.6B encoder + Qwen-Image VAE.
            status("Downloading Anima Base v1.0 (~5 GB)…");
            let _ = std::fs::create_dir_all(m("diffusion_models"));
            let _ = std::fs::create_dir_all(m("text_encoders"));
            let _ = std::fs::create_dir_all(m("vae"));
            ok &= fetch(ANIMA_UNET, m("diffusion_models").join("anima-base-v1.0.safetensors"), "Anima Base v1.0", "", &send);
            ok &= fetch(ANIMA_TE, m("text_encoders").join("qwen_3_06b_base.safetensors"), "Qwen3 0.6B encoder", "", &send);
            ok &= fetch(ANIMA_VAE, m("vae").join("qwen_image_vae.safetensors"), "Qwen-Image VAE", "", &send);
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

fn start_generate(state: &mut GenerateState, ctx: &egui::Context, cur_image: Option<&Path>) {
    let (tx, rx) = mpsc::channel();
    state.rx = Some(rx);
    state.running = true;
    state.status = "Generating…".into();
    state.status_err = false;
    state.cancel.store(false, Ordering::SeqCst);
    let cancel = state.cancel.clone();

    let seed = if state.randomize_seed {
        (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
            & 0x7FFF_FFFF_FFFF) as i64
    } else {
        state.seed
    };
    // Image-to-video uses the currently-selected image as the first frame.
    let i2v = state.family.is_video() && state.i2v;
    let job = GenJob {
        model: state.model,
        prompt: state.prompt.clone(),
        steps: state.steps,
        guidance: state.cfg,
        width: state.width,
        height: state.height,
        seed,
        checkpoint: state.checkpoint.clone(),
        loras: state
            .loras
            .iter()
            .filter(|l| l.selected && l.base.matches(state.family))
            .map(|l| (l.file.clone(), l.strength))
            .collect(),
        pos_embeds: embed_tokens(state, false),
        neg_embeds: embed_tokens(state, true),
        frames: state.frames,
        fps: state.fps,
        i2v,
        src_image: if i2v { cur_image.map(|p| p.to_path_buf()) } else { None },
    };
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let ok = run_generate(job, &tx, &ctx, &cancel);
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
    /// An auto-detected installed checkpoint to use instead of the model's
    /// built-in default (SDXL only); `None` uses the built-in checkpoint.
    checkpoint: Option<String>,
    /// Selected LoRAs (filename, weight) — chained into the workflow.
    loras: Vec<(String, f32)>,
    /// Selected embeddings as `embedding:<name>` tokens, split by polarity and
    /// space-joined — appended to the positive / negative encoder text.
    pos_embeds: String,
    neg_embeds: String,
    /// Video (LTX): frame count + playback rate.
    frames: i32,
    fps: i32,
    /// Video (LTX): animate `src_image` (image-to-video) instead of text-to-video.
    i2v: bool,
    src_image: Option<PathBuf>,
}

fn run_generate(job: GenJob, tx: &mpsc::Sender<RunnerMsg>, ctx: &egui::Context, cancel: &AtomicBool) -> bool {
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

    // 0 — pre-flight: make sure this variant's model files are actually on disk.
    // Without this, a missing file only surfaces as an opaque HTTP 400 from the
    // queue endpoint (e.g. picking the bf16 encoder before it was downloaded).
    let missing = missing_files(job.model, job.checkpoint.as_deref());
    if !missing.is_empty() {
        send(format!(
            "ERROR: this model isn't fully downloaded — select this variant and click \
             Setup Requirements. Missing file(s): {}",
            missing.join(", ")
        ));
        return false;
    }

    // 1 — make sure the ComfyUI server is up. For checkpoint-picker families,
    // register the extra checkpoint folders first; if that config changed, the
    // running server must be restarted to pick it up (paths load at startup only).
    if job.model.family().uses_checkpoint_picker() && ensure_checkpoint_paths(&comfy) && ping() {
        status("Updating model folders — restarting ComfyUI…");
        stop_server();
    }
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

    // 2 — image source. Image-to-video uploads the selected frame. LTX-2.3's
    // unified graph also needs *a* valid image for text-to-video (it's bypassed),
    // so upload a neutral blank in that case.
    let mut image_name: Option<String> = None;
    let ltx2 = job.model == GenModel::Ltx2Distilled;
    if job.i2v {
        if let Some(src) = &job.src_image {
            status("Uploading source image…");
            match upload_image(&agent, src) {
                Ok(name) => {
                    send(format!("== Uploaded {name}"));
                    image_name = Some(name);
                }
                Err(e) => {
                    send(format!("ERROR: image upload failed: {e}"));
                    return false;
                }
            }
        }
    } else if ltx2 {
        // Neutral 1280×704 gray frame; bypassed by the workflow but must exist.
        let png = blank_png(1280, 704);
        match upload_image_bytes(&agent, "clarity_blank.png", png) {
            Ok(name) => image_name = Some(name),
            Err(e) => {
                send(format!("ERROR: blank-image upload failed: {e}"));
                return false;
            }
        }
    }

    // Queue the workflow. Scope `http_status_as_error(false)` to THIS request so a
    // 4xx returns a readable body instead of a bare status code — ComfyUI puts the
    // real reason (missing model, bad node input) in the body's `error`/`node_errors`.
    status("Queuing…");
    let wf = build_workflow(&job, image_name.as_deref());
    let body = serde_json::json!({ "prompt": wf, "client_id": "clarity-tagflow" }).to_string();
    let resp = match agent
        .post(&format!("{SERVER_URL}/prompt"))
        .config()
        .http_status_as_error(false)
        .build()
        .header("Content-Type", "application/json")
        .send(body.as_str())
    {
        Ok(r) => r,
        Err(e) => {
            send(format!("ERROR: queue request failed: {e}"));
            return false;
        }
    };
    let code = resp.status().as_u16();
    let json: serde_json::Value = match read_json(resp) {
        Ok(j) => j,
        Err(e) => {
            send(format!("ERROR: bad queue response (HTTP {code}): {e}"));
            return false;
        }
    };
    // A rejected job comes back as 4xx and/or a populated `node_errors` — pull out
    // ComfyUI's human-readable reason rather than reporting just the status.
    let node_errs = json.get("node_errors").filter(|v| v.is_object() && !v.as_object().unwrap().is_empty());
    if code >= 400 || node_errs.is_some() {
        // Prefer node_errors (names the offending node + input/file), then the
        // generic top-level error message, then the raw error object.
        let detail = node_errs
            .map(|e| e.to_string())
            .or_else(|| {
                json.get("error")
                    .and_then(|e| e.get("message").and_then(|m| m.as_str()).map(str::to_string))
            })
            .or_else(|| json.get("error").map(|e| e.to_string()))
            .unwrap_or_else(|| "no detail".into());
        send(format!("ERROR: ComfyUI rejected the job (HTTP {code}): {detail}"));
        return false;
    }
    let Some(prompt_id) = json.get("prompt_id").and_then(|v| v.as_str()).map(str::to_string) else {
        send("ERROR: no prompt_id in response".into());
        return false;
    };

    // 3 — poll /history until the output is ready. Save nodes report their files
    // under "images" (SaveImage/SaveWEBM/SaveVideo) or "gifs"; the node id varies
    // per workflow, so scan all output nodes. Images allow ~10 min (Flux is slow
    // on first load); video allows ~60 min — Wan 2.2 5B at its native 1280×704
    // can run ~45 s/step on a 16 GB laptop (no cu130 fp8 kernels → fp16 fallback
    // + VRAM offloading), so a full clip is 20-30 min. The user can Stop anytime.
    let is_video = job.model.family().is_video();
    let max_polls = if is_video { 7200 } else { 1200 };
    status("Generating…");
    let mut images: Option<serde_json::Value> = None;
    for _ in 0..max_polls {
        if cancel.load(Ordering::SeqCst) {
            // Abort the running job server-side (and drop it from the queue in
            // case it hadn't started), then bail out without flagging an error.
            let _ = agent.post(&format!("{SERVER_URL}/interrupt")).send_empty();
            let _ = agent
                .post(&format!("{SERVER_URL}/queue"))
                .header("Content-Type", "application/json")
                .send(serde_json::json!({ "delete": [prompt_id] }).to_string().as_str());
            send("== Cancelled".into());
            status("Cancelled");
            return true;
        }
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
        if let Some(outs) = entry.get("outputs").and_then(|o| o.as_object()) {
            let found = outs.values().find_map(|n| {
                n.get("images").or_else(|| n.get("gifs")).filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
            });
            if let Some(imgs) = found {
                images = Some(imgs.clone());
                break;
            }
        }
    }
    let Some(images) = images else {
        send("ERROR: timed out waiting for the output".into());
        return false;
    };
    let Some(img0) = images.as_array().and_then(|a| a.first()) else {
        send("ERROR: no output produced".into());
        return false;
    };
    let filename = img0.get("filename").and_then(|v| v.as_str()).unwrap_or("");
    let subfolder = img0.get("subfolder").and_then(|v| v.as_str()).unwrap_or("");
    let kind = img0.get("type").and_then(|v| v.as_str()).unwrap_or("output");

    // 4 — fetch the bytes and save them into our outputs dir.
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
            send(format!("ERROR: fetching output failed: {e}"));
            return false;
        }
    };
    let outdir = base.join("outputs").join(job.model.family().out_dir());
    let _ = std::fs::create_dir_all(&outdir);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if is_video {
        // Video: keep ComfyUI's container extension (webm) and write raw — the PNG
        // metadata stamping below is image-only.
        let ext = Path::new(filename).extension().and_then(|e| e.to_str()).unwrap_or("webm");
        let dest = outdir.join(format!("gen_{stamp}.{ext}"));
        if let Err(e) = std::fs::write(&dest, &bytes) {
            send(format!("ERROR: could not save video: {e}"));
            return false;
        }
        send(format!("== Saved {}", dest.display()));
        let _ = tx.send(RunnerMsg::Output(dest));
        status("Done");
        return true;
    }
    let dest = outdir.join(format!("gen_{stamp}.png"));
    // Stamp A1111-style generation metadata (incl. the Clarity TagFlow version) so
    // sites like Civitai surface the prompt/settings and attribute the tool.
    let bytes = add_png_text(&bytes, "parameters", &build_params(&job));
    if let Err(e) = std::fs::write(&dest, &bytes) {
        send(format!("ERROR: could not save image: {e}"));
        return false;
    }
    send(format!("== Saved {}", dest.display()));
    let _ = tx.send(RunnerMsg::Output(dest));
    status("Done");
    true
}

/// Upload an image file to ComfyUI's input dir so a LoadImage node can reference
/// it by the returned name. Used for LTX image-to-video.
fn upload_image(agent: &ureq::Agent, path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "source.png".to_string());
    upload_image_bytes(agent, &name, bytes)
}

/// Upload raw image bytes (multipart/form-data) under `name`; returns the
/// server-side filename.
fn upload_image_bytes(agent: &ureq::Agent, name: &str, bytes: Vec<u8>) -> Result<String, String> {
    let boundary = "----claritytagflowboundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(&bytes);
    body.extend_from_slice(format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").as_bytes());

    let resp = agent
        .post(&format!("{SERVER_URL}/upload/image"))
        .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
        .send(&body[..])
        .map_err(|e| e.to_string())?;
    let json = read_json(resp)?;
    // ComfyUI returns {"name": "...", "subfolder": "", "type": "input"}.
    json.get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| "no name in upload response".to_string())
}

/// A solid-gray PNG of the given size — used as the (bypassed) LoadImage source
/// for LTX-2.3 text-to-video, whose unified graph still requires a valid image.
fn blank_png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([127, 127, 127]));
    let mut buf = std::io::Cursor::new(Vec::new());
    let _ = image::DynamicImage::ImageRgb8(img).write_to(&mut buf, image::ImageFormat::Png);
    buf.into_inner()
}

/// Build an A1111-style `parameters` string for the PNG metadata. The trailing
/// `Version:` field is how generators attribute themselves (Civitai shows it).
fn build_params(job: &GenJob) -> String {
    let model = match job.model {
        GenModel::SchnellQ4 | GenModel::SchnellQ8 => "FLUX.1-schnell",
        GenModel::DevQ4 | GenModel::DevQ8 => "FLUX.1-dev",
        GenModel::ZImageFast | GenModel::ZImageQuality => "Z-Image Turbo",
        // Video models don't go through this PNG-metadata path, but keep the
        // match exhaustive.
        GenModel::LtxDistilled => "LTX-Video",
        GenModel::Ltx2Distilled => "LTX-2.3",
        GenModel::WanTi2v5bFast | GenModel::WanTi2v5bQuality => "Wan 2.2 5B",
        GenModel::SdxlBase => "SDXL 1.0",
        GenModel::AnimaBase => "Anima v1.0",
    };
    let mut prompt = job.prompt.clone();
    // Surface selected LoRAs as A1111 tags so they show up too.
    for (file, strength) in &job.loras {
        let name = file.trim_end_matches(".safetensors");
        prompt.push_str(&format!(" <lora:{name}:{strength}>"));
    }
    format!(
        "{prompt}\nNegative prompt: \nSteps: {}, Sampler: Euler, CFG scale: {}, Seed: {}, Size: {}x{}, Model: {}, Version: Clarity TagFlow v{}",
        job.steps,
        job.guidance,
        job.seed,
        job.width,
        job.height,
        model,
        env!("CARGO_PKG_VERSION"),
    )
}

/// Insert a PNG `tEXt` chunk (keyword + text) right after IHDR, preserving every
/// existing chunk (e.g. ComfyUI's own `prompt`/`workflow` metadata).
fn add_png_text(png: &[u8], keyword: &str, text: &str) -> Vec<u8> {
    // Bail unless this is a real PNG (8-byte signature + at least the IHDR chunk).
    if png.len() < 33 || &png[..8] != b"\x89PNG\r\n\x1a\n" {
        return png.to_vec();
    }
    let mut data = Vec::with_capacity(keyword.len() + 1 + text.len());
    data.extend_from_slice(keyword.as_bytes());
    data.push(0);
    data.extend_from_slice(text.as_bytes());

    let mut crc = flate2::Crc::new();
    crc.update(b"tEXt");
    crc.update(&data);

    let mut chunk = Vec::with_capacity(12 + data.len());
    chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(b"tEXt");
    chunk.extend_from_slice(&data);
    chunk.extend_from_slice(&crc.sum().to_be_bytes());

    // After the signature (8) + IHDR chunk (4 len + 4 type + 13 data + 4 crc = 25).
    let at = 33;
    let mut out = Vec::with_capacity(png.len() + chunk.len());
    out.extend_from_slice(&png[..at]);
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[at..]);
    out
}

/// Build the ComfyUI API workflow for the job's model family. `image_name` is the
/// server-side filename of an uploaded source image (LTX image-to-video only).
fn build_workflow(job: &GenJob, image_name: Option<&str>) -> serde_json::Value {
    match job.model {
        GenModel::Ltx2Distilled => ltx2_workflow(job, image_name),
        _ => match job.model.family() {
            GenFamily::Flux => flux_workflow(job),
            GenFamily::ZImage => zimage_workflow(job),
            GenFamily::Ltx => ltx_workflow(job, image_name),
            GenFamily::Wan => wan_workflow(job, image_name),
            GenFamily::Sdxl => sdxl_workflow(job),
            GenFamily::Anima => anima_workflow(job),
        },
    }
}

/// Anima Base v1.0 text-to-image (CircleStone Labs / Comfy Org, NVIDIA Cosmos 2):
/// UNETLoader + CLIPLoader(qwen3, type "stable_diffusion") + VAELoader(qwen image)
/// + dual CLIPTextEncode + EmptyLatentImage + KSampler (er_sde/simple) + VAEDecode
/// + SaveImage, with any selected LoRAs chained in model+clip. Mirrors ComfyUI's
/// official `image_anima_base_v1` template.
fn anima_workflow(job: &GenJob) -> serde_json::Value {
    use serde_json::json;
    let width = (job.width / 8).max(8) * 8;
    let height = (job.height / 8).max(8) * 8;

    // An auto-detected installed Anima model overrides the built-in base.
    let unet = job.checkpoint.as_deref().unwrap_or("anima-base-v1.0.safetensors");
    let mut wf = json!({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
    });
    let obj = wf.as_object_mut().unwrap();

    // Chain LoraLoader nodes (model + clip thread through), starting at the loaders.
    let mut model_ref = json!(["1", 0]);
    let mut clip_ref = json!(["2", 0]);
    let mut id = 100;
    for (file, strength) in &job.loras {
        let node = id.to_string();
        id += 1;
        obj.insert(
            node.clone(),
            json!({"class_type": "LoraLoader", "inputs": {
                "model": model_ref, "clip": clip_ref, "lora_name": file,
                "strength_model": strength, "strength_clip": strength
            }}),
        );
        model_ref = json!([node, 0]);
        clip_ref = json!([node, 1]);
    }

    // A light anime-oriented negative (cfg 4 uses the negative conditioning).
    const ANIMA_NEG: &str = "lowres, worst quality, low quality, bad anatomy, bad hands, text, watermark, jpeg artifacts, signature";
    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(&job.prompt, &job.pos_embeds), "clip": clip_ref.clone()}}));
    obj.insert("6".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(ANIMA_NEG, &job.neg_embeds), "clip": clip_ref}}));
    obj.insert("7".into(), json!({"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}));
    obj.insert("8".into(), json!({"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["4", 0], "negative": ["6", 0], "latent_image": ["7", 0],
        "seed": job.seed, "steps": job.steps, "cfg": job.guidance,
        "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0
    }}));
    obj.insert("9".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}));
    obj.insert("10".into(), json!({"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ClarityAnima"}}));
    wf
}

/// SDXL Base 1.0 text-to-image: CheckpointLoaderSimple (UNet + dual CLIP + VAE) +
/// CLIPTextEncode ×2 + EmptyLatentImage + KSampler + VAEDecode + SaveImage, with
/// any selected LoRAs chained in model+clip via LoraLoader nodes.
fn sdxl_workflow(job: &GenJob) -> serde_json::Value {
    use serde_json::json;
    // SDXL is trained at ~1 MP; width/height should be multiples of 8.
    let width = (job.width / 8).max(8) * 8;
    let height = (job.height / 8).max(8) * 8;

    // An auto-detected installed checkpoint overrides the built-in base.
    let ckpt = job.checkpoint.as_deref().unwrap_or("sd_xl_base_1.0.safetensors");
    let mut wf = json!({
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
    });
    let obj = wf.as_object_mut().unwrap();

    // Chain LoraLoader nodes (model + clip thread through), starting at the ckpt.
    let mut model_ref = json!(["1", 0]);
    let mut clip_ref = json!(["1", 1]);
    let mut id = 100;
    for (file, strength) in &job.loras {
        let node = id.to_string();
        id += 1;
        obj.insert(
            node.clone(),
            json!({"class_type": "LoraLoader", "inputs": {
                "model": model_ref, "clip": clip_ref, "lora_name": file,
                "strength_model": strength, "strength_clip": strength
            }}),
        );
        model_ref = json!([node, 0]);
        clip_ref = json!([node, 1]);
    }

    let vae_ref = json!(["1", 2]);
    // A standard SDXL quality negative.
    const SDXL_NEG: &str = "ugly, low quality, worst quality, blurry, jpeg artifacts, watermark, text, deformed, bad anatomy";
    obj.insert("3".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(&job.prompt, &job.pos_embeds), "clip": clip_ref.clone()}}));
    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(SDXL_NEG, &job.neg_embeds), "clip": clip_ref}}));
    obj.insert("5".into(), json!({"class_type": "EmptyLatentImage", "inputs": {
        "width": width, "height": height, "batch_size": 1
    }}));
    obj.insert("6".into(), json!({"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
        "seed": job.seed, "steps": job.steps, "cfg": job.guidance,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0
    }}));
    obj.insert("7".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": vae_ref}}));
    obj.insert("8".into(), json!({"class_type": "SaveImage", "inputs": {
        "images": ["7", 0], "filename_prefix": "ClaritySDXL"
    }}));
    wf
}

/// LTX-2.3 22B distilled (video **with audio**). This pipeline is far more
/// involved than 0.9.x (two-stage AV sampling + spatial latent upscaler + tiled
/// decode), so rather than hand-build it we embed the validated 35-node graph
/// (exported from ComfyUI's bundled LTX-2.3 blueprint, see assets/ltx2_workflow
/// .json) and patch the few user-controlled fields by node id. Stage-1 renders at
/// half resolution and the upscaler ×2 brings it to 1280×704. The graph is a
/// unified t2v/i2v graph: `LTXVImgToVideoInplace.bypass` controls whether the
/// LoadImage source is used (t2v bypasses it but still needs *a* valid image).
fn ltx2_workflow(job: &GenJob, image_name: Option<&str>) -> serde_json::Value {
    let mut wf: serde_json::Value =
        serde_json::from_str(include_str!("../assets/ltx2_workflow.json")).expect("valid ltx2 template");
    let obj = wf.as_object_mut().unwrap();
    let frames = (((job.frames - 1).max(8) / 8) * 8) + 1;
    let fps = job.fps;
    let set = |obj: &mut serde_json::Map<String, serde_json::Value>, node: &str, key: &str, val: serde_json::Value| {
        if let Some(inputs) = obj.get_mut(node).and_then(|n| n.get_mut("inputs")).and_then(|i| i.as_object_mut()) {
            inputs.insert(key.to_string(), val);
        }
    };
    use serde_json::json;
    set(obj, "305", "text", json!(with_embeds(&job.prompt, &job.pos_embeds))); // positive prompt
    set(obj, "278", "noise_seed", json!(job.seed)); // stage-1 noise
    set(obj, "279", "noise_seed", json!(job.seed)); // stage-2 noise
    set(obj, "297", "length", json!(frames)); // base video latent
    set(obj, "307", "frames_number", json!(frames)); // audio latent length
    set(obj, "306", "frame_rate", json!(fps));
    set(obj, "307", "frame_rate", json!(fps));
    set(obj, "312", "fps", json!(fps)); // CreateVideo
    // Image source: image-to-video uses the uploaded frame (bypass off); text-to
    // -video keeps the bypass on (the uploaded blank is ignored).
    if let Some(name) = image_name {
        set(obj, "322", "image", json!(name));
    }
    let bypass = !job.i2v;
    set(obj, "290", "bypass", json!(bypass));
    set(obj, "298", "bypass", json!(bypass));
    wf
}

/// LTX-Video 0.9.6 distilled text/image-to-video workflow, all native ComfyUI
/// nodes. Stage 1: CheckpointLoaderSimple (transformer + VAE) + CLIPLoader(ltxv,
/// t5xxl) → CLIPTextEncode; latent is EmptyLTXVLatentVideo (t2v) or LTXVImgToVideo
/// from the uploaded image (i2v); LTXVConditioning sets frame rate; LTXVScheduler
/// + KSamplerSelect drive SamplerCustom. Stage 2 (the key to sharp output):
/// LTXVLatentUpsampler ×2 + a short refine SamplerCustom, then VAEDecode →
/// SaveWEBM (.webm). Selected LoRAs chain in model-only.
fn ltx_workflow(job: &GenJob, image_name: Option<&str>) -> serde_json::Value {
    use serde_json::json;
    // LTX constraints: latent length must be 8k+1; width/height multiples of 32.
    let length = (((job.frames - 1).max(8) / 8) * 8) + 1;
    let width = (job.width / 32).max(8) * 32;
    let height = (job.height / 32).max(8) * 32;
    let fps = job.fps;

    let mut wf = json!({
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltxv-2b-0.9.6-distilled-04-25.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp8_e4m3fn.safetensors", "type": "ltxv"}},
    });
    let obj = wf.as_object_mut().unwrap();

    // Model-only LoRA chain (LTX has no CLIP-side LoRA path), starting at the ckpt.
    let mut model_ref = json!(["1", 0]);
    let mut id = 100;
    for (file, strength) in &job.loras {
        let node = id.to_string();
        id += 1;
        obj.insert(
            node.clone(),
            json!({"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model_ref, "lora_name": file, "strength_model": strength
            }}),
        );
        model_ref = json!([node, 0]);
    }

    let clip_ref = json!(["2", 0]);
    let vae_ref = json!(["1", 2]);
    // A standard video negative (used because LTX runs with a little CFG here).
    const LTX_NEG: &str = "worst quality, inconsistent motion, blurry, jittery, distorted, washed out, overexposed";
    obj.insert("3".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(&job.prompt, &job.pos_embeds), "clip": clip_ref.clone()}}));
    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(LTX_NEG, &job.neg_embeds), "clip": clip_ref}}));

    // Conditioning + latent: image-to-video seeds the latent from the uploaded
    // image (LTXVImgToVideo also rewrites the conditioning); text-to-video uses an
    // empty latent.
    let (pos_in, neg_in, latent_ref) = if job.i2v {
        let img = image_name.unwrap_or("");
        obj.insert("5".into(), json!({"class_type": "LoadImage", "inputs": {"image": img}}));
        obj.insert("6".into(), json!({"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "vae": vae_ref.clone(), "image": ["5", 0],
            "width": width, "height": height, "length": length, "batch_size": 1, "strength": 1.0
        }}));
        (json!(["6", 0]), json!(["6", 1]), json!(["6", 2]))
    } else {
        obj.insert("5".into(), json!({"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": length, "batch_size": 1
        }}));
        (json!(["3", 0]), json!(["4", 0]), json!(["5", 0]))
    };

    obj.insert("7".into(), json!({"class_type": "LTXVConditioning", "inputs": {
        "positive": pos_in, "negative": neg_in, "frame_rate": fps
    }}));
    // Stage 1: sample at the base resolution.
    obj.insert("8".into(), json!({"class_type": "LTXVScheduler", "inputs": {
        "steps": job.steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": true,
        "terminal": 0.1, "latent": latent_ref.clone()
    }}));
    obj.insert("9".into(), json!({"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}));
    obj.insert("11".into(), json!({"class_type": "SamplerCustom", "inputs": {
        "model": model_ref.clone(), "add_noise": true, "noise_seed": job.seed, "cfg": job.guidance,
        "positive": ["7", 0], "negative": ["7", 1], "sampler": ["9", 0], "sigmas": ["8", 0],
        "latent_image": latent_ref
    }}));

    // Stage 2: LTX spatial latent upscaler (×2) + a short refine pass. This is the
    // key to sharp video — single-pass LTX is soft/blurry. The refine pass denoises
    // only the tail of a fresh schedule (SplitSigmasDenoise) so stage-1 motion is
    // preserved while spatial detail is added at 2× resolution.
    obj.insert("20".into(), json!({"class_type": "LatentUpscaleModelLoader", "inputs": {
        "model_name": "ltxv-spatial-upscaler-0.9.7.safetensors"
    }}));
    obj.insert("21".into(), json!({"class_type": "LTXVLatentUpsampler", "inputs": {
        "samples": ["11", 0], "upscale_model": ["20", 0], "vae": vae_ref.clone()
    }}));
    obj.insert("22".into(), json!({"class_type": "LTXVScheduler", "inputs": {
        "steps": job.steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": true,
        "terminal": 0.1, "latent": ["21", 0]
    }}));
    obj.insert("23".into(), json!({"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": ["22", 0], "denoise": 0.4}}));
    obj.insert("24".into(), json!({"class_type": "SamplerCustom", "inputs": {
        "model": model_ref, "add_noise": true, "noise_seed": job.seed, "cfg": job.guidance,
        "positive": ["7", 0], "negative": ["7", 1], "sampler": ["9", 0], "sigmas": ["23", 1],
        "latent_image": ["21", 0]
    }}));
    obj.insert("12".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": vae_ref}}));
    // Native ComfyUI SaveWEBM (vp9 .webm) — no custom node needed; the app plays
    // webm via its existing video support. crf 18 keeps the upscaled detail.
    obj.insert("10".into(), json!({"class_type": "SaveWEBM", "inputs": {
        "images": ["12", 0], "filename_prefix": "ClarityLTX", "codec": "vp9", "fps": fps, "crf": 18.0
    }}));
    wf
}

/// Wan 2.2 TI2V 5B text/image-to-video workflow, all native ComfyUI nodes.
/// UNETLoader (the 5B model; weight_dtype picks fp8-load vs fp16) + CLIPLoader
/// (umt5, type "wan") + VAELoader (Wan VAE) → CLIPTextEncode ×2; the model passes
/// through any model-only LoRAs and then ModelSamplingSD3 (shift 8). The latent
/// comes from Wan22ImageToVideoLatent — its optional `start_image` makes the SAME
/// node do text-to-video (omitted) or image-to-video (the uploaded frame). A
/// single KSampler (uni_pc / simple, 20 steps, cfg 5) → VAEDecode → SaveWEBM.
fn wan_workflow(job: &GenJob, image_name: Option<&str>) -> serde_json::Value {
    use serde_json::json;
    // Wan constraints: latent length must be 4k+1; width/height multiples of 32.
    let length = (((job.frames - 1).max(4) / 4) * 4) + 1;
    let width = (job.width / 32).max(8) * 32;
    let height = (job.height / 32).max(8) * 32;
    let fps = job.fps;

    let mut wf = json!({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": job.model.wan_weight_dtype()}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
    });
    let obj = wf.as_object_mut().unwrap();

    // Model-only LoRA chain (Wan LoRAs are model-side), starting at the UNet.
    let mut model_ref = json!(["1", 0]);
    let mut id = 100;
    for (file, strength) in &job.loras {
        let node = id.to_string();
        id += 1;
        obj.insert(
            node.clone(),
            json!({"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model_ref, "lora_name": file, "strength_model": strength
            }}),
        );
        model_ref = json!([node, 0]);
    }

    let clip_ref = json!(["2", 0]);
    let vae_ref = json!(["3", 0]);
    // Wan's recommended (Chinese) negative — the one shipped in the official
    // template; it markedly improves motion stability and reduces artifacts.
    const WAN_NEG: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(&job.prompt, &job.pos_embeds), "clip": clip_ref.clone()}}));
    obj.insert("5".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(WAN_NEG, &job.neg_embeds), "clip": clip_ref}}));

    // ModelSamplingSD3 (shift 8) — Wan's sampling-shift node.
    obj.insert("6".into(), json!({"class_type": "ModelSamplingSD3", "inputs": {"model": model_ref, "shift": 8.0}}));

    // Latent: the one node does both modes — image-to-video seeds it from the
    // uploaded frame (start_image), text-to-video omits it.
    let mut latent_inputs = json!({
        "vae": vae_ref.clone(), "width": width, "height": height, "length": length, "batch_size": 1
    });
    if job.i2v {
        if let Some(name) = image_name {
            obj.insert("8".into(), json!({"class_type": "LoadImage", "inputs": {"image": name}}));
            latent_inputs["start_image"] = json!(["8", 0]);
        }
    }
    obj.insert("7".into(), json!({"class_type": "Wan22ImageToVideoLatent", "inputs": latent_inputs}));

    obj.insert("9".into(), json!({"class_type": "KSampler", "inputs": {
        "model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0],
        "seed": job.seed, "steps": job.steps, "cfg": job.guidance,
        "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0
    }}));
    obj.insert("10".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae_ref}}));
    // Native ComfyUI SaveWEBM (vp9 .webm) — same as LTX; the app plays webm.
    obj.insert("11".into(), json!({"class_type": "SaveWEBM", "inputs": {
        "images": ["10", 0], "filename_prefix": "ClarityWan", "codec": "vp9", "fps": fps, "crf": 18.0
    }}));
    wf
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

    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(&job.prompt, &job.pos_embeds), "clip": clip_ref.clone()}}));
    obj.insert("6".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds("", &job.neg_embeds), "clip": clip_ref}}));
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

/// Flux GGUF text-to-image workflow as a ComfyUI API prompt, with any selected
/// LoRAs chained in via LoraLoader nodes (same pattern as Z-Image).
fn flux_workflow(job: &GenJob) -> serde_json::Value {
    use serde_json::json;
    let mut wf = json!({
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": job.model.unet_file()}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"
        }},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
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

    obj.insert("4".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds(&job.prompt, &job.pos_embeds), "clip": clip_ref.clone()}}));
    obj.insert("5".into(), json!({"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": job.guidance}}));
    obj.insert("6".into(), json!({"class_type": "CLIPTextEncode", "inputs": {"text": with_embeds("", &job.neg_embeds), "clip": clip_ref}}));
    obj.insert("7".into(), json!({"class_type": "EmptySD3LatentImage", "inputs": {"width": job.width, "height": job.height, "batch_size": 1}}));
    obj.insert("8".into(), json!({"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
        "seed": job.seed, "steps": job.steps, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
    }}));
    obj.insert("9".into(), json!({"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}}));
    obj.insert("10".into(), json!({"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ClarityFlux"}}));
    wf
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
