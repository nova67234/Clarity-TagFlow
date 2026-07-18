//! Pixal3D — image→3D generation panel, Linux/Windows only (this whole module is
//! compiled out on macOS via `#[cfg(not(target_os = "macos"))]` on its `mod`
//! declaration, since Pixal3D needs an NVIDIA CUDA GPU).
//!
//! UI mirrors TencentARC's Pixal3D web app (the controls map to its `/generate_3d`
//! and `/extract_glb` API params): a source image, generation settings (seed,
//! resolution, sparse-structure / shape / texture guidance + steps), a
//! Generate 3D button, and GLB extraction settings. Styled after the Tag Manager.
//!
//! Backend status: fully wired. "Setup Requirements" provisions a self-contained
//! runtime (standalone Python + cu128 PyTorch + prebuilt CUDA-kernel wheels +
//! nvdiffrast + the vendored o_voxel GLB exporter) and downloads all model weights
//! from public sources — no Hugging Face login needed (DINOv3 comes from the open
//! camenduru mirror; the gated RMBG-2.0 is swapped for ZhengPeng7/BiRefNet).
//! Generate runs real inference and writes a PNG-textured GLB, shown in the centre
//! viewer (src/scene3d.rs). Requires an NVIDIA GPU + CUDA Toolkit + MSVC (the
//! latter two only for nvdiffrast's first-use compile, as with NATTEN).

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::SystemTime;

use eframe::egui;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText};

use crate::theme::{ACCENT1, EDGE, FIELD, MUTED, PANEL, TEXT};

const GREEN: Color32 = Color32::from_rgb(46, 160, 67);
const RED: Color32 = Color32::from_rgb(220, 70, 70);

/// Messages streamed from a background runner (setup / generate) to the UI.
enum RunnerMsg {
    Line(String),
    /// Update the header status text.
    Status(String),
    /// A generation finished and wrote this GLB — show it in the centre viewer.
    Output(PathBuf),
    Done(bool),
}

pub struct Pixal3DState {
    // --- Generation settings (match the Pixal3D /generate_3d API) ---
    seed: i64,
    randomize_seed: bool,
    resolution: i32,
    ss_guidance: f32,
    ss_steps: i32,
    shape_guidance: f32,
    shape_steps: i32,
    tex_guidance: f32,
    tex_steps: i32,
    // --- GLB extraction (match /extract_glb) ---
    decimation: i32,
    texture_size: i32,
    /// Hugging Face token (for gated models like briaai/RMBG-2.0). Session-only.
    hf_token: String,
    /// Low-VRAM mode (loads models on-demand; ~fits 16 GB vs ~18 GB standard).
    low_vram: bool,
    // --- Log (collapsible, Z-Image style) ---
    show_log: bool,
    /// Brief green/red flash on the copy-log button after a click.
    copy_flash: Option<(std::time::Instant, bool)>,
    // --- Status / background runner ---
    status: String,
    status_err: bool,
    log: Vec<String>,
    rx: Option<Receiver<RunnerMsg>>,
    running: bool,
    orb: crate::ai_orb::AiOrb,
    /// The most recently generated GLB, shown in the centre 3D viewer. Seeded on
    /// startup with the newest model already in the outputs dir (if any).
    pub last_glb: Option<PathBuf>,
}

impl Default for Pixal3DState {
    fn default() -> Self {
        Self {
            // Defaults from the Pixal3D space.
            seed: 0,
            randomize_seed: true,
            resolution: 1024,
            ss_guidance: 7.5,
            ss_steps: 12,
            shape_guidance: 7.5,
            shape_steps: 12,
            tex_guidance: 1.0,
            tex_steps: 12,
            decimation: 50_000,
            texture_size: 1024,
            hf_token: load_hf_token(),
            low_vram: true,
            show_log: false,
            copy_flash: None,
            status: "Ready".to_string(),
            status_err: false,
            log: Vec::new(),
            rx: None,
            running: false,
            orb: crate::ai_orb::AiOrb::default(),
            last_glb: latest_glb(),
        }
    }
}

/// Newest `.glb` in the Pixal3D outputs dir, so a previously generated model is
/// shown in the centre viewer when the app restarts.
fn latest_glb() -> Option<PathBuf> {
    let dir = crate::tagger::models_root().join("pixal3d").join("outputs");
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("glb")))
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok())
}

/// Render the Pixal3D generation view into the right panel.
pub fn show(ui: &mut egui::Ui, state: &mut Pixal3DState, current_image: Option<&Path>) {
    // Drain any background-runner messages (the Setup Requirements job).
    if let Some(rx) = &state.rx {
        let mut finished = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                RunnerMsg::Line(line) => state.log.push(line),
                RunnerMsg::Status(s) => state.status = s,
                RunnerMsg::Output(path) => state.last_glb = Some(path),
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

    // --- Header bar (title + status + orb), mirroring the Tag Manager. ---
    egui::Frame::new()
        .fill(PANEL())
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .corner_radius(CornerRadius::same(18))
        .inner_margin(Margin::symmetric(12, 6))
        .show(ui, |ui| {
            ui.set_height(40.0);
            ui.horizontal_centered(|ui| {
                ui.label(RichText::new("Pixal3D").color(TEXT()).strong().size(14.0));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let color = if state.status_err { RED } else if state.running { ACCENT1() } else { GREEN };
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

    // --- Setup row (Z-Image style: button + right-aligned status). ---
    ui.horizontal(|ui| {
        let setup = egui::Button::new(RichText::new("Setup Requirements").color(Color32::WHITE))
            .fill(Color32::from_rgb(96, 99, 105))
            .corner_radius(CornerRadius::same(12));
        if ui.add_enabled_ui(!state.running, |ui| ui.add_sized(egui::vec2(150.0, 28.0), setup)).inner.clicked() {
            start_setup(state, ui.ctx());
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(RichText::new("NVIDIA GPU required").color(MUTED()).size(11.0));
        });
    });

    ui.add_space(8.0);

    // --- HF token (optional). Setup Requirements pulls every model from public
    // sources (DINOv3 via the open camenduru mirror, BiRefNet for background
    // removal), so no Hugging Face login is needed — the field only exists in case
    // someone wants to use their own gated model. ---
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label(RichText::new("HF token (optional)").color(MUTED()).size(11.0));
        // Info icon: hover for an explanation + a clickable link.
        ui.add(
            egui::Image::new(egui::include_image!("../icons/info.svg"))
                .fit_to_exact_size(egui::vec2(14.0, 14.0))
                .tint(crate::theme::icon_tint(MUTED())),
        )
        .on_hover_ui(|ui| {
            ui.set_max_width(260.0);
            ui.label(
                "No Hugging Face login is needed — \"Setup Requirements\" downloads \
                 every model from public sources. Only add a token if you want to use \
                 your own gated model.",
            );
            crate::arrow_link(ui, "Manage Hugging Face tokens", "https://huggingface.co/settings/tokens", None);
        });
    });
    ui.add_space(2.0);
    let resp = ui.add(
        egui::TextEdit::singleline(&mut state.hf_token)
            .password(true)
            .desired_width(f32::INFINITY),
    );
    // Persist (encrypted) once the user finishes editing, so it survives restarts.
    if resp.lost_focus() {
        save_hf_token(&state.hf_token);
    }

    ui.add_space(8.0);

    // --- Source image card. ---
    section(ui, "Source Image", |ui| {
        match current_image {
            Some(p) => {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("<unknown>");
                ui.label(RichText::new(name).color(TEXT()).size(12.5));
            }
            None => {
                ui.label(RichText::new("No image selected — pick one in the browser.").color(MUTED()).size(12.0));
            }
        }
    });

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.low_vram, RichText::new("Low VRAM mode").color(TEXT()).size(12.0));
        ui.label(RichText::new("recommended for ≤16 GB").color(MUTED()).size(11.0));
    });

    ui.add_space(8.0);

    // --- Generation settings (inline, Z-Image style). ---
    labeled(ui, "Resolution", |ui| {
        egui::ComboBox::from_id_salt("pixal3d_res")
            .selected_text(state.resolution.to_string())
            .show_ui(ui, |ui| {
                for r in [512, 1024, 1536] {
                    ui.selectable_value(&mut state.resolution, r, r.to_string());
                }
            });
    });

    ui.add_space(4.0);
    ui.label(RichText::new("Sparse structure").color(MUTED()).size(11.0));
    slider(ui, "Guidance", &mut state.ss_guidance, 0.0..=12.0);
    int_slider(ui, "Steps", &mut state.ss_steps, 1..=50);
    ui.add_space(2.0);
    ui.label(RichText::new("Shape").color(MUTED()).size(11.0));
    slider(ui, "Guidance", &mut state.shape_guidance, 0.0..=12.0);
    int_slider(ui, "Steps", &mut state.shape_steps, 1..=50);
    ui.add_space(2.0);
    ui.label(RichText::new("Texture").color(MUTED()).size(11.0));
    slider(ui, "Guidance", &mut state.tex_guidance, 0.0..=12.0);
    int_slider(ui, "Steps", &mut state.tex_steps, 1..=50);

    ui.add_space(6.0);
    // Randomize seed — radio-style filled dot, matching Z-Image.
    ui.horizontal(|ui| {
        ui.spacing_mut().icon_width_inner = 11.0;
        if ui.radio(state.randomize_seed, "").clicked() {
            state.randomize_seed = !state.randomize_seed;
        }
        ui.label(RichText::new("Randomize seed").color(TEXT()).size(12.0));
    });
    if !state.randomize_seed {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Seed").color(MUTED()).size(12.0));
            ui.add(egui::DragValue::new(&mut state.seed).range(0..=i64::MAX));
        });
    }

    ui.add_space(8.0);
    // --- GLB extraction (inline). ---
    ui.label(RichText::new("GLB Extraction").color(MUTED()).size(11.0));
    int_slider(ui, "Simplify (tris)", &mut state.decimation, 5_000..=200_000);
    labeled(ui, "Texture size", |ui| {
        egui::ComboBox::from_id_salt("pixal3d_texsize")
            .selected_text(state.texture_size.to_string())
            .show_ui(ui, |ui| {
                for t in [512, 1024, 2048] {
                    ui.selectable_value(&mut state.texture_size, t, t.to_string());
                }
            });
    });
    ui.add_space(4.0);
    let extract = egui::Button::new(RichText::new("Extract GLB").color(TEXT()))
        .corner_radius(CornerRadius::same(10));
    if ui.add_enabled(!state.running, extract).clicked() {
        state.status = "Pixal3D runtime not installed — Extract GLB unavailable".into();
        state.status_err = true;
    }

    ui.add_space(10.0);

    // --- Generate (primary), after the settings like Z-Image. ---
    let gen_btn = egui::Button::new(RichText::new("Generate 3D").color(Color32::WHITE).strong())
        .fill(ACCENT1())
        .corner_radius(CornerRadius::same(12));
    let enabled = !state.running && current_image.is_some();
    if ui.add_enabled_ui(enabled, |ui| ui.add_sized(egui::vec2(ui.available_width(), 34.0), gen_btn)).inner.clicked() {
        start_generate(state, ui.ctx(), current_image);
    }

    ui.add_space(8.0);

    ui.add_space(8.0);

    // --- Log (collapsible, Z-Image style: disclosure pill + flashing copy). ---
    ui.horizontal(|ui| {
        // SVG disclosure arrow: drop-down when open, right when collapsed.
        let arrow_src = if state.show_log {
            egui::include_image!("../icons/arrow_drop_down.svg")
        } else {
            egui::include_image!("../icons/arrow_right.svg")
        };
        // Fixed-size pill drawn by hand so it never resizes with the content.
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(46.0, 18.0), egui::Sense::click());
        let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
        if ui.is_rect_visible(rect) {
            let visuals = *ui.style().interact(&resp);
            let txt = visuals.text_color();
            ui.painter().rect(
                rect,
                egui::CornerRadius::same(10),
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
                .tint(crate::theme::icon_tint(txt))
                .paint_at(ui, icon_rect);
            ui.painter()
                .galley(egui::pos2(x0 + icon + gap, cy - galley.size().y / 2.0), galley, txt);
        }
        if resp.clicked() {
            state.show_log = !state.show_log;
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            // Flash the copy icon green (copied) or red (failed) for ~0.9s.
            const COPY_FLASH_SECS: f32 = 0.9;
            let flash_tint = match state.copy_flash {
                Some((when, ok)) if when.elapsed().as_secs_f32() < COPY_FLASH_SECS => {
                    ui.ctx().request_repaint();
                    Some(if ok { GREEN } else { RED })
                }
                _ => None,
            };
            let tint = flash_tint.unwrap_or_else(|| crate::theme::icon_tint(MUTED()));
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
                let ok = arboard::Clipboard::new()
                    .and_then(|mut c| c.set_text(state.log.join("\n")))
                    .is_ok();
                state.copy_flash = Some((std::time::Instant::now(), ok));
            }
        });
    });
    if state.show_log {
        let log_bg = if crate::theme::is_light() { FIELD() } else { Color32::from_rgb(15, 15, 17) };
        egui::Frame::new()
            .fill(log_bg)
            .corner_radius(CornerRadius::same(22))
            .inner_margin(Margin::same(10))
            .stroke(egui::Stroke::new(1.0, EDGE()))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                egui::ScrollArea::vertical()
                    .id_salt("pixal3d_log")
                    .max_height(180.0)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        // Keep scrolling while a text selection is dragged past the edge.
                        crate::drag_select_autoscroll(ui);
                        if state.log.is_empty() {
                            ui.label(RichText::new("Output will appear here.").color(MUTED()).monospace().size(12.0));
                        } else {
                            for line in &state.log {
                                ui.label(RichText::new(line).color(TEXT()).monospace().size(12.0));
                            }
                        }
                    });
            });
    }
}

/// A titled card (PANEL fill, rounded) wrapping `contents`.
fn section(ui: &mut egui::Ui, title: &str, contents: impl FnOnce(&mut egui::Ui)) {
    ui.label(RichText::new(title).color(MUTED()).size(11.0));
    ui.add_space(2.0);
    egui::Frame::new()
        .fill(PANEL())
        .corner_radius(CornerRadius::same(12))
        .stroke(egui::Stroke::new(1.0, EDGE()))
        .inner_margin(Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            contents(ui);
        });
}

/// A label on the left, a right-aligned control on the right.
fn labeled(ui: &mut egui::Ui, label: &str, contents: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(TEXT()).size(12.0));
        ui.with_layout(Layout::right_to_left(Align::Center), contents);
    });
}

fn slider(ui: &mut egui::Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(MUTED()).size(12.0));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(egui::Slider::new(value, range).fixed_decimals(1));
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

/// Run Pixal3D inference on the selected image: `inference.py --image … --output
/// …`, streaming output to the log. Produces a GLB under `tools/pixal3d/outputs`.
/// `ATTN_BACKEND=sdpa` avoids the flash-attn dependency.
fn start_generate(state: &mut Pixal3DState, ctx: &egui::Context, current_image: Option<&Path>) {
    let Some(img) = current_image else {
        state.status = "No image selected".into();
        state.status_err = true;
        return;
    };
    let base = crate::tagger::models_root().join("pixal3d");
    let src = base.join("Pixal3D-master");
    let py = if cfg!(windows) {
        base.join("python").join("python.exe")
    } else {
        base.join("python").join("bin").join("python3")
    };
    if !src.join("inference.py").exists() || !py.exists() {
        state.status = "Requirements not installed — click Setup Requirements".into();
        state.status_err = true;
        return;
    }

    if state.randomize_seed {
        // Derive a fresh seed; no rng crate, so fold the wall clock.
        let n = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0);
        state.seed = (n.unsigned_abs() % 1_000_000) as i64;
    }
    // Persist the token (encrypted) whenever a run starts, in case it was typed
    // and the user hit Generate without the field losing focus first.
    save_hf_token(&state.hf_token);

    let seed = state.seed;
    let resolution = state.resolution;
    let low_vram = state.low_vram;
    let token = state.hf_token.trim().to_string();
    let img = img.to_path_buf();
    let weights = src.join("weights");
    let out_dir = base.join("outputs");
    let _ = std::fs::create_dir_all(&out_dir);
    let stem = img.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let out = out_dir.join(format!("{stem}_{seed}.glb"));

    let (tx, rx) = mpsc::channel();
    state.rx = Some(rx);
    state.running = true;
    state.status = "Generating 3D…".into();
    state.status_err = false;
    state.log.clear();

    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let send = |line: String| {
            let _ = tx.send(RunnerMsg::Line(line));
            ctx.request_repaint();
        };
        send(format!("Generating 3D from {}", img.display()));
        send(format!("Output: {}", out.display()));

        let py = py.to_string_lossy().to_string();
        let img_s = img.to_string_lossy().to_string();
        let out_s = out.to_string_lossy().to_string();
        let res_s = resolution.to_string();
        let seed_s = seed.to_string();
        let weights_s = weights.to_string_lossy().to_string();

        let mut env: Vec<(&str, &str)> = vec![("ATTN_BACKEND", "sdpa")];
        if !token.is_empty() {
            env.push(("HF_TOKEN", &token));
        }
        let mut args: Vec<&str> = vec![
            "inference.py",
            "--image", &img_s,
            "--output", &out_s,
            "--seed", &seed_s,
            "--resolution", &res_s,
            "--model_path", &weights_s,
        ];
        if low_vram {
            args.push("--low_vram");
        }
        let ok = run_streamed(&tx, &ctx, &src, &py, &args, &env);

        if ok {
            send(format!("== Done. GLB saved to {}", out.display()));
            let _ = tx.send(RunnerMsg::Output(out.clone()));
            let _ = tx.send(RunnerMsg::Status("3D generated".into()));
        } else {
            send("== Generation failed — see errors above.".into());
            let _ = tx.send(RunnerMsg::Status("Generation failed — see log".into()));
        }
        let _ = tx.send(RunnerMsg::Done(ok));
        ctx.request_repaint();
    });
}

// --- Real setup pipeline ---------------------------------------------------

/// Standalone Python (astral-sh/python-build-standalone), pinned to a verified
/// release. The "install_only" tarball extracts to a `python/` dir with pip.
const PY_TAG: &str = "20260602";
const PY_VER: &str = "3.12.13";
/// Pixal3D source (GitHub zip — extracts to `Pixal3D-master/`).
const SRC_ZIP: &str = "https://github.com/TencentARC/Pixal3D/archive/refs/heads/master.zip";
/// PyTorch CUDA wheel index. cu128 (PyTorch 2.7+) is required for Blackwell
/// (RTX 50-series, sm_120); older cu124 maxes out at sm_90.
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";
/// utils3d wheel required by Pixal3D (per its README).
const UTILS3D_WHL: &str =
    "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl";

/// Prebuilt Pixal3D CUDA-kernel wheels (cumesh, o_voxel, drtk, flex_gemm) from
/// PozzettiAndrea/cuda-wheels — the set the official ComfyUI installer uses,
/// matched to Python 3.12 + torch 2.8 + cu128. Prebuilt = no local CUDA compile.
/// `%2B` is the URL-encoded `+` local-version separator.
fn cuda_kernel_wheels() -> [&'static str; 4] {
    let plat = if cfg!(windows) { "win_amd64" } else { "linux_x86_64" };
    // Leak the formatted URLs to get `'static` strs (called once per setup run).
    let url = |pkg: &str, file: &str| -> &'static str {
        Box::leak(
            format!("https://github.com/PozzettiAndrea/cuda-wheels/releases/download/{pkg}-latest/{file}")
                .into_boxed_str(),
        )
    };
    [
        url("flex_gemm_ap", &format!("flex_gemm_ap-1.0.0%2Bcu128torch2.8-cp312-cp312-{plat}.whl")),
        url("cumesh_vb", &format!("cumesh_vb-1.0%2Bcu128torch2.8-cp312-cp312-{plat}.whl")),
        url("o_voxel_vb_ap", &format!("o_voxel_vb_ap-0.0.1%2Bcu128torch2.8-cp312-cp312-{plat}.whl")),
        url("drtk", &format!("drtk-0.1.0%2Bcu128torch2.8-cp312-cp312-{plat}.whl")),
    ]
}

/// Python patch: make the NAF upsampler fall back to the low-res projection when
/// NATTEN is unavailable (no Windows NATTEN wheel exists), so inference runs.
/// Run as `python patch_naf.py <image_conditioned_proj.py>`. Idempotent.
const NAF_PATCH_PY: &str = r#"import io, sys
f = sys.argv[1]
s = io.open(f, encoding='utf-8').read()
changed = False
# Guard: _load_naf returns early (naf_model=None) when NATTEN isn't importable,
# instead of crashing in torch.hub.load -> NAF -> import natten.
g_need = "        if self.naf_model is None:\n            import torch.hub\n"
g_repl = ("        if self.naf_model is None:\n"
"            try:\n"
"                import natten  # NAF load patch\n"
"            except Exception:\n"
"                self.naf_model = None\n"
"                return\n"
"            import torch.hub\n")
if 'NAF load patch' not in s and g_need in s:
    s = s.replace(g_need, g_repl); changed = True
# Forward: reuse the low-res projection for the hr branch when NAF is unavailable.
if 'NAF fallback patch' not in s:
    needle = "            if self.use_naf_upsample:\n                self._load_naf()\n"
    i = s.find(needle)
    if i >= 0:
        endmark = "z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)"
        j = s.find(endmark, i) + len(endmark)
        repl = (
"            if self.use_naf_upsample:  # NAF fallback patch (no NATTEN -> reuse lr)\n"
"                z_proj_hr = z_proj_lr\n"
"                try:\n"
"                    self._load_naf()\n"
"                    lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)\n"
"                    hr_features = self.naf_model(image_for_naf, lr_features_bchw, self.naf_target_size)\n"
"                    z_proj_hr = self.proj_grid(hr_features, camera_angle_x, distance, mesh_scale, transform_matrix, BHWC=False)\n"
"                except Exception as _naf_exc:\n"
"                    print('[Pixal3D] NAF unavailable (%s: %s); using lr fallback.' % (type(_naf_exc).__name__, _naf_exc))\n"
"                z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)"
)
        s = s[:i] + repl + s[j:]; changed = True
io.open(f, 'w', encoding='utf-8').write(s)
print('NAF patched OK' if changed else 'NAF already patched')
"#;

pub(crate) fn py_tarball_url() -> String {
    let triple = if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    format!(
        "https://github.com/astral-sh/python-build-standalone/releases/download/{PY_TAG}/\
         cpython-{PY_VER}+{PY_TAG}-{triple}-install_only.tar.gz"
    )
}

/// Spawn the real setup pipeline on a background thread, streaming every line of
/// output to the panel log. Bootstrap steps (Python, source) are fatal; the
/// GPU-dependent pip steps are best-effort (logged, but don't abort) since they
/// need the user's NVIDIA GPU + CUDA Toolkit.
fn start_setup(state: &mut Pixal3DState, ctx: &egui::Context) {
    let (tx, rx) = mpsc::channel();
    state.rx = Some(rx);
    state.running = true;
    state.status = "Setting up…".into();
    state.status_err = false;
    state.log.clear();

    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let ok = run_setup(&tx, &ctx);
        let _ = tx.send(RunnerMsg::Done(ok));
        ctx.request_repaint();
    });
}

fn run_setup(tx: &mpsc::Sender<RunnerMsg>, ctx: &egui::Context) -> bool {
    let send = |line: String| {
        let _ = tx.send(RunnerMsg::Line(line));
        ctx.request_repaint();
    };

    let base = crate::tagger::models_root().join("pixal3d");
    if let Err(e) = std::fs::create_dir_all(&base) {
        send(format!("ERROR: could not create {}: {e}", base.display()));
        return false;
    }
    send(format!("== Install dir: {}", base.display()));

    // Step 1 — standalone Python (fatal if it fails).
    let py = if cfg!(windows) {
        base.join("python").join("python.exe")
    } else {
        base.join("python").join("bin").join("python3")
    };
    if py.exists() {
        send("== Python already present, skipping download".into());
    } else {
        send("== [1/6] Downloading standalone Python…".into());
        let tarball = base.join("python.tar.gz");
        if let Err(e) = download(&py_tarball_url(), &tarball, tx, ctx) {
            send(format!("ERROR: Python download failed: {e}"));
            return false;
        }
        send("== Extracting Python…".into());
        if !run_streamed(tx, ctx, &base, "tar", &["-xzf", "python.tar.gz"], &[]) {
            send("ERROR: failed to extract Python".into());
            return false;
        }
        let _ = std::fs::remove_file(&tarball);
    }
    if !run_streamed(tx, ctx, &base, py.to_str().unwrap_or("python"), &["--version"], &[]) {
        send("ERROR: standalone Python is not runnable".into());
        return false;
    }
    let py = py.to_string_lossy().to_string();

    // Step 2 — Pixal3D source (fatal if it fails).
    let src = base.join("Pixal3D-master");
    if src.join("requirements.txt").exists() {
        send("== Source already present, skipping download".into());
    } else {
        send("== [2/6] Downloading Pixal3D source…".into());
        let zip = base.join("src.zip");
        if let Err(e) = download(SRC_ZIP, &zip, tx, ctx) {
            send(format!("ERROR: source download failed: {e}"));
            return false;
        }
        send("== Extracting source…".into());
        if let Err(e) = unzip(&zip, &base) {
            send(format!("ERROR: failed to extract source: {e}"));
            return false;
        }
        let _ = std::fs::remove_file(&zip);
    }

    // Patch the NAF upsampler to fall back to low-res features when NATTEN is
    // unavailable (no Windows NATTEN wheel), so inference runs without it.
    send("== Patching NAF for no-NATTEN fallback".into());
    let patch_py = base.join("patch_naf.py");
    let icp = base
        .join("Pixal3D-master")
        .join("pixal3d")
        .join("trainers")
        .join("flow_matching")
        .join("mixins")
        .join("image_conditioned_proj.py");
    if std::fs::write(&patch_py, NAF_PATCH_PY).is_ok() {
        let patch_s = patch_py.to_string_lossy().to_string();
        let icp_s = icp.to_string_lossy().to_string();
        run_streamed(tx, ctx, &base, &py, &[&patch_s, &icp_s], &[]);
    }

    // Embed PNG textures in the exported GLB instead of WebP: the in-app 3D
    // viewer (three-d) can't decode the glTF EXT_texture_webp extension, so a
    // WebP-textured model would render untextured. PNG is a bit larger but
    // universally readable.
    let inf = src.join("inference.py");
    if let Ok(s) = std::fs::read_to_string(&inf) {
        let s2 = s.replace("extension_webp=True", "extension_webp=False");
        if s2 != s && std::fs::write(&inf, s2).is_ok() {
            send("== Set GLB export to PNG textures (in-app viewer can't read WebP)".into());
        }
    }

    let src = src.to_string_lossy().to_string();

    // Steps 3–6 — pip installs (best-effort: log failures, keep going so the user
    // sees every error in one run). These need an NVIDIA GPU + CUDA Toolkit.
    // `required` steps mark the run failed; "optional" GPU-kernel steps only warn
    // (they need a system CUDA Toolkit + compiler we can't bundle), so the core
    // install + weights still report success.
    let mut core_ok = true;
    let mut step = |label: &str, args: &[&str], env: &[(&str, &str)], required: bool| {
        send(format!("== {label}"));
        if !run_streamed(tx, ctx, &base, &py, args, env) {
            send(format!("WARNING: step failed: {label}"));
            if required {
                core_ok = false;
            }
        }
    };

    step("[3/6] Upgrading pip", &["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], &[], true);
    step(
        "[3/6] Installing PyTorch 2.8 (CUDA 12.8)",
        &["-m", "pip", "install", "--upgrade", "torch==2.8.0", "torchvision", "--index-url", TORCH_INDEX],
        &[],
        true,
    );
    step(
        "[4/6] Installing Pixal3D requirements",
        &["-m", "pip", "install", "-r", &format!("{src}/requirements.txt")],
        &[],
        true,
    );
    // Extra deps not pinned in requirements.txt but needed at runtime (e.g. the
    // BiRefNet remote modeling code imports einops).
    step("[4/6] Installing extra deps (einops)", &["-m", "pip", "install", "einops"], &[], true);
    step("[5/6] Installing utils3d", &["-m", "pip", "install", UTILS3D_WHL], &[], true);

    // Prebuilt Pixal3D CUDA kernels (cumesh, o_voxel, drtk, flex_gemm). `--no-deps`
    // so pip doesn't try to swap the pinned cu128 torch for a PyPI build.
    let wheels = cuda_kernel_wheels();
    let mut wheel_args: Vec<&str> = vec!["-m", "pip", "install", "--no-deps"];
    wheel_args.extend_from_slice(&wheels);
    step("[5/6] Installing Pixal3D CUDA kernels", &wheel_args, &[], true);

    // Triton — the kernels' triton path needs it. Windows uses the triton-windows
    // build; Linux uses upstream triton.
    let triton_pkg = if cfg!(windows) { "triton-windows" } else { "triton" };
    step("[5/6] Installing Triton", &["-m", "pip", "install", triton_pkg], &[], true);

    // Alias shims: the wheels install as cumesh_vb / o_voxel_vb_ap / flex_gemm_ap,
    // but Pixal3D imports them as cumesh / o_voxel / flex_gemm.
    send("== Creating kernel import aliases (cumesh, o_voxel, flex_gemm)".into());
    if let Err(e) = write_kernel_shims(&base) {
        send(format!("WARNING: could not create import aliases: {e}"));
    }

    // GLB export needs o_voxel.postprocess (TRELLIS.2's texture-baking GLB
    // exporter), which the prebuilt o_voxel *kernel* wheel doesn't ship, plus
    // nvdiffrast (its differentiable rasterizer). nvdiffrast isn't on PyPI; its
    // build needs `--no-build-isolation` to see the installed torch, and it
    // JIT-compiles a CUDA plugin on first use (needs the CUDA Toolkit + MSVC,
    // like NATTEN), so it's best-effort. `--no-deps` keeps pip from swapping the
    // pinned cu128 torch. Without these, generation runs but fails at the final
    // "Extracting GLB" step.
    step(
        "[5/6] Installing nvdiffrast (GLB export — compiles on first use, needs CUDA Toolkit + MSVC)",
        &[
            "-m", "pip", "install", "--no-deps", "--no-build-isolation",
            "https://github.com/NVlabs/nvdiffrast/archive/refs/heads/main.zip",
        ],
        &[],
        false,
    );
    send("== Adding o_voxel.postprocess (the GLB exporter) to the kernel package".into());
    if let Err(e) = vendor_o_voxel_postprocess(&base, tx, ctx) {
        send(format!("WARNING: could not add o_voxel.postprocess: {e}"));
    }

    // NATTEN compiles from source (CUDA Toolkit + MSVC) — best-effort, and only
    // needed if a neighborhood-attention backend is selected (we use sdpa).
    step("[5/6] Installing build tools (cmake, ninja)", &["-m", "pip", "install", "cmake", "ninja"], &[], false);
    step(
        "[5/6] Installing NATTEN (compiles — needs CUDA Toolkit + MSVC)",
        &["-m", "pip", "install", "natten==0.21.0", "--no-build-isolation"],
        &[],
        false,
    );
    step("[6/6] Installing huggingface_hub", &["-m", "pip", "install", "huggingface_hub"], &[], true);
    let dl = format!(
        "from huggingface_hub import snapshot_download; \
         snapshot_download('TencentARC/Pixal3D', local_dir=r'{src}/weights')"
    );
    step("[6/6] Downloading model weights", &["-c", &dl], &[], true);

    // Swap the gated background-removal model (briaai/RMBG-2.0) for the open,
    // non-gated ZhengPeng7/BiRefNet so that piece needs no Hugging Face login.
    // (DINOv3, the image encoder, is gated by Meta and has no open swap — it
    // still needs a one-time token, then caches locally.)
    let pj = base.join("Pixal3D-master").join("weights").join("pipeline.json");
    if let Ok(s) = std::fs::read_to_string(&pj) {
        let s2 = s.replace("briaai/RMBG-2.0", "ZhengPeng7/BiRefNet");
        if s2 != s && std::fs::write(&pj, s2).is_ok() {
            send("== Set background-removal model to open ZhengPeng7/BiRefNet".into());
        }
    }

    if core_ok {
        send("== Core install + weights complete. (NATTEN, if it failed, needs the CUDA Toolkit + MSVC Build Tools.)".into());
        let _ = tx.send(RunnerMsg::Status("Requirements installed".into()));
    } else {
        send("== Setup failed — see the errors above.".into());
        let _ = tx.send(RunnerMsg::Status("Setup failed — see log".into()));
    }
    core_ok
}

/// Run a command in `cwd`, streaming stdout+stderr line-by-line to the log.
/// Returns true on a zero exit status.
fn run_streamed(
    tx: &mpsc::Sender<RunnerMsg>,
    ctx: &egui::Context,
    cwd: &Path,
    program: &str,
    args: &[&str],
    env: &[(&str, &str)],
) -> bool {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let _ = tx.send(RunnerMsg::Line(format!("$ {program} {}", args.join(" "))));
    ctx.request_repaint();

    // Make sure this child (and anything it spawns) is bound to the app's
    // lifetime, so closing Clarity_TagFlow can't orphan a running Python/CUDA job.
    #[cfg(windows)]
    ensure_kill_on_exit_job();

    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    hide_window(&mut cmd);
    kill_on_parent_exit(&mut cmd);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(RunnerMsg::Line(format!("ERROR: cannot run {program}: {e}")));
            ctx.request_repaint();
            return false;
        }
    };

    // stderr on its own thread; stdout on this one.
    let stderr = child.stderr.take();
    let err_handle = stderr.map(|err| {
        let txe = tx.clone();
        let cte = ctx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let _ = txe.send(RunnerMsg::Line(line));
                cte.request_repaint();
            }
        })
    });
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let _ = tx.send(RunnerMsg::Line(line));
            ctx.request_repaint();
        }
    }
    if let Some(h) = err_handle {
        let _ = h.join();
    }
    matches!(child.wait(), Ok(s) if s.success())
}

/// Stream a URL to `dest` through the shared resumable downloader (net.rs:
/// `.part` temp, retry with backoff, Range resume), logging periodic progress.
fn download(url: &str, dest: &Path, tx: &mpsc::Sender<RunnerMsg>, ctx: &egui::Context) -> Result<(), String> {
    let mut last_pct = 0u64;
    crate::net::download(url, dest, "", &mut |note| match note {
        crate::net::Note::Progress { got, total } => {
            if let Some(pct) = (got * 100).checked_div(total)
                && pct >= last_pct + 5
            {
                last_pct = pct;
                let _ = tx.send(RunnerMsg::Line(format!(
                    "   {pct}%  ({:.1}/{:.1} MB)",
                    got as f64 / 1e6,
                    total as f64 / 1e6
                )));
                ctx.request_repaint();
            }
        }
        crate::net::Note::Retry { attempt, of, err } => {
            let _ = tx.send(RunnerMsg::Line(format!(
                "   connection dropped ({err}) — retry {}/{}",
                attempt - 1,
                of - 1
            )));
            ctx.request_repaint();
        }
    })
}

/// Create alias shim packages so Pixal3D's `import cumesh` / `o_voxel` /
/// `flex_gemm` resolve to the installed `*_vb` / `*_ap` wheels.
fn write_kernel_shims(base: &Path) -> std::io::Result<()> {
    let site = if cfg!(windows) {
        base.join("python").join("Lib").join("site-packages")
    } else {
        base.join("python").join("lib").join("python3.12").join("site-packages")
    };
    for (alias, target) in [
        ("cumesh", "cumesh_vb"),
        ("o_voxel", "o_voxel_vb_ap"),
        ("flex_gemm", "flex_gemm_ap"),
    ] {
        let dir = site.join(alias);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join("__init__.py"),
            format!("import sys\nimport {target} as _m\nsys.modules[__name__] = _m\n"),
        )?;
    }
    Ok(())
}

/// Vendor TRELLIS.2's `o_voxel.postprocess` (the GLB exporter, `to_glb`) onto the
/// installed `o_voxel_vb_ap` kernel package, which ships only the CUDA kernels
/// (`convert` / `io` / `serialize`) and omits the pure-Python postprocess module.
/// Drops `postprocess.py` into the package and rewrites its `__init__.py` to
/// expose it, so `import o_voxel; o_voxel.postprocess.to_glb(...)` resolves.
fn vendor_o_voxel_postprocess(base: &Path, tx: &mpsc::Sender<RunnerMsg>, ctx: &egui::Context) -> Result<(), String> {
    let site = if cfg!(windows) {
        base.join("python").join("Lib").join("site-packages")
    } else {
        base.join("python").join("lib").join("python3.12").join("site-packages")
    };
    let pkg = site.join("o_voxel_vb_ap");
    if !pkg.exists() {
        return Err("o_voxel_vb_ap is not installed".into());
    }

    // postprocess.py from TRELLIS.2 (the upstream o_voxel source).
    const PP_URL: &str =
        "https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/o-voxel/o_voxel/postprocess.py";
    download(PP_URL, &pkg.join("postprocess.py"), tx, ctx)?;

    // Rewrite __init__.py so `import o_voxel` also imports postprocess. Guard that
    // import: a missing/uncompilable nvdiffrast must not break the (working)
    // sampling stages — it then fails later at to_glb with a clear nvdiffrast
    // error instead of crashing the whole run at `import o_voxel`.
    let init_body = "from . import (\n    convert,\n    io,\n    serialize,\n)\n\
        try:\n    from . import postprocess  # GLB exporter (needs nvdiffrast)\n\
        except Exception as _e:\n    import sys as _sys\n\
        \x20\x20\x20\x20print('[Pixal3D] o_voxel.postprocess unavailable: %r' % (_e,), file=_sys.stderr)\n";
    std::fs::write(pkg.join("__init__.py"), init_body).map_err(|e| e.to_string())?;
    Ok(())
}

/// Extract a zip archive into `dest_dir` using the `zip` crate.
fn unzip(zip_path: &Path, dest_dir: &Path) -> Result<(), String> {
    let file = std::fs::File::open(zip_path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).map_err(|e| e.to_string())?;
        let Some(rel) = entry.enclosed_name() else { continue };
        let out_path = dest_dir.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path).map_err(|e| e.to_string())?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut out = std::fs::File::create(&out_path).map_err(|e| e.to_string())?;
            std::io::copy(&mut entry, &mut out).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

// --- Hugging Face token persistence ----------------------------------------
//
// Stored encrypted at rest (DPAPI on Windows, via src/secret.rs — same scheme as
// the Gelbooru API key) so it isn't kept as plaintext and survives restarts.

fn hf_token_path() -> std::path::PathBuf {
    crate::tagger::models_root().join("pixal3d").join("hf_token.dat")
}

/// Load the saved HF token (decrypted). Returns "" if none is stored or it was
/// protected by a different user/machine.
pub(crate) fn load_hf_token() -> String {
    std::fs::read_to_string(hf_token_path())
        .ok()
        .map(|s| crate::secret::unprotect(s.trim()))
        .unwrap_or_default()
}

/// Save the HF token, encrypted. An empty token removes the stored file.
pub(crate) fn save_hf_token(token: &str) {
    let path = hf_token_path();
    let trimmed = token.trim();
    if trimmed.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, crate::secret::protect(trimmed));
}

/// On Windows, don't pop a console window for each spawned child process.
#[cfg(windows)]
pub(crate) fn hide_window(cmd: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
}
#[cfg(not(windows))]
pub(crate) fn hide_window(_cmd: &mut std::process::Command) {}

/// Tie every process this module spawns to the app's lifetime, so closing
/// Clarity_TagFlow (or it crashing) can never leave an orphaned Python/CUDA job
/// running. Pixal3D inference holds gigabytes of GPU + system memory; without
/// this, closing the app mid-generation leaks it all until the machine falls over.
///
/// Windows: assign the *current* process to a Job Object flagged
/// `KILL_ON_JOB_CLOSE`. Child processes spawned afterwards inherit the job, so
/// when the app's last handle to the job closes — on a clean exit OR a crash, the
/// OS tears down the handle table either way — the whole job tree (the Python
/// child and all its worker grandchildren) is terminated. Runs once; the job
/// handle is intentionally never closed so it stays open for the process lifetime.
#[cfg(windows)]
pub(crate) fn ensure_kill_on_exit_job() {
    use std::ffi::c_void;
    use std::sync::Once;

    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        type Handle = *mut c_void;
        const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION: i32 = 9;
        const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;

        #[repr(C)]
        struct BasicLimit {
            per_process_user_time_limit: i64,
            per_job_user_time_limit: i64,
            limit_flags: u32,
            minimum_working_set_size: usize,
            maximum_working_set_size: usize,
            active_process_limit: u32,
            affinity: usize,
            priority_class: u32,
            scheduling_class: u32,
        }
        #[repr(C)]
        struct IoCounters {
            read_op: u64,
            write_op: u64,
            other_op: u64,
            read_xfer: u64,
            write_xfer: u64,
            other_xfer: u64,
        }
        #[repr(C)]
        struct ExtendedLimit {
            basic: BasicLimit,
            io: IoCounters,
            process_memory_limit: usize,
            job_memory_limit: usize,
            peak_process_memory_used: usize,
            peak_job_memory_used: usize,
        }

        // SAFETY: standard kernel32 Job Object signatures; all pointers below are
        // valid for the duration of each call.
        unsafe extern "system" {
            fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> Handle;
            fn SetInformationJobObject(job: Handle, class: i32, info: *const c_void, len: u32) -> i32;
            fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
            fn GetCurrentProcess() -> Handle;
        }

        unsafe {
            let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if job.is_null() {
                return;
            }
            let mut info: ExtendedLimit = std::mem::zeroed();
            info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                std::ptr::addr_of!(info) as *const c_void,
                std::mem::size_of::<ExtendedLimit>() as u32,
            );
            // If the app is already inside a no-breakaway job this can fail; in
            // that case we simply degrade to the prior (unprotected) behaviour.
            AssignProcessToJobObject(job, GetCurrentProcess());
            // `job` is deliberately not closed — its open handle is what ties the
            // job's lifetime (and the kill-on-close trigger) to this process.
        }
    });
}

/// Linux: best-effort — ask the kernel to SIGKILL the child if the spawning
/// process dies (`PR_SET_PDEATHSIG`), so a closed/crashed app doesn't orphan the
/// Python inference child. Set in the child via `pre_exec` (post-fork).
#[cfg(target_os = "linux")]
pub(crate) fn kill_on_parent_exit(cmd: &mut std::process::Command) {
    use std::os::unix::process::CommandExt;
    // SAFETY: prctl is async-signal-safe, which is all pre_exec permits.
    unsafe {
        cmd.pre_exec(|| {
            unsafe extern "C" {
                fn prctl(option: i32, arg2: usize, arg3: usize, arg4: usize, arg5: usize) -> i32;
            }
            const PR_SET_PDEATHSIG: i32 = 1;
            const SIGKILL: usize = 9;
            prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);
            Ok(())
        });
    }
}
#[cfg(not(target_os = "linux"))]
pub(crate) fn kill_on_parent_exit(_cmd: &mut std::process::Command) {}
