# 🖼️ Clarity TagFlow

**A fast, native image viewer, tagger, and AI toolbox for your image library — built in Rust.**

Browse and tag thousands of images, auto-tag them with built-in AI models, read Stable Diffusion metadata, generate new images with Flux or Z-Image, turn photos into 3D models, and more — all in one app, no Python environment or command line required.

[![Latest release](https://img.shields.io/github/v/release/nova67234/Clarity-TagFlow?include_prereleases&label=download)](https://github.com/nova67234/Clarity-TagFlow/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-blue)

<!-- Drop a screenshot here: ![Clarity TagFlow](docs/screenshot.png) -->

---

## ✨ Features

### 🏷️ Tagging & organizing
- **AI auto-tagging** — PixAI, JoyTag, and the WD14 family run in-process (ONNX); one click downloads the models, no Hugging Face login needed.
- **Tag Manager** — edit, search, and batch-manage tags; favorites; tag-based filtering.
- **Stable Diffusion metadata reader** — see the prompt, seed, sampler, and model of generated images at a glance.
- **Deep Scan** — find corrupt images and duplicates in your library.

### 🖼️ Viewing
- **Fast viewer** — zoom, pan, crop, copy-to-clipboard (browser-paste friendly).
- **Gallery layout** with quick search and a detail popup.
- **Wide format support** — JPG, PNG, WebP, **AVIF, HEIC, HDR**, and camera RAW (**DNG, ARW, CR2, NEF**) with proper color handling.
- **Video playback** via your system's VLC (Windows), with thumbnails.
- **Five themes** — Dark, Light, frosted **Glass**, animated **Space**, and **Aurora**. Full CJK + color-emoji rendering.

### 🤖 AI tools
- **Image generation** — Flux (schnell/dev) and **Z-Image Turbo** with LoRA support, driven by a self-installing ComfyUI backend. Click *Setup Requirements* and it downloads everything.
- **Background removal** — one-click subject cutouts (BiRefNet).
- **Pixal3D** — turn a single image into a textured **3D model** (GLB), with an interactive orbit/zoom viewer and Apple-style spatial depth parallax.
- **Prompt spell-check** — squiggles and right-click suggestions in the prompt boxes.

### 🌐 Integrations
- **Gelbooru downloader** — tag search with blacklist support; API key encrypted at rest.
- **Civitai panel** — resource info for the models behind your generated images.
- **Backups** — archive your library and its tags in a couple of clicks.

---

## 📥 Download

Grab the latest version from the **[Releases page](https://github.com/nova67234/Clarity-TagFlow/releases/latest)**:

| Platform | File | Notes |
|---|---|---|
| 🪟 Windows | `ClarityTagFlow-Setup.exe` | Installer (recommended) |
| 🪟 Windows | `ClarityTagFlow-windows.zip` | Portable — unzip and run |
| 🍎 macOS (Apple Silicon) | macOS artifact | Core app; GPU-only features compiled out |
| 🐧 Linux | Linux artifact | Built on Ubuntu 24.04 |

### Requirements
- **Just run it** — no Java, no Python, no runtimes to install.
- 🎬 **Video playback** (Windows): uses your installed [VLC](https://www.videolan.org/); the app offers an install link if it's missing.
- 🤖 **Image generation & Pixal3D**: need an **NVIDIA GPU**. The built-in *Setup Requirements* button downloads everything else automatically (Pixal3D additionally needs the CUDA Toolkit + MSVC Build Tools to compile its GPU plugin once).

---

## 🐞 Bugs & ideas

Found a bug or want a feature? **[Open an issue](https://github.com/nova67234/Clarity-TagFlow/issues/new/choose)** — there are short forms for both. Screenshots and the file type involved help a lot.

---

## 🛠️ Building from source

```bash
git clone https://github.com/nova67234/Clarity-TagFlow.git
cd Clarity-TagFlow
cargo build --release
```

Rust (2024 edition) is all you need; the first build pulls everything else via Cargo.

---

## 📜 History & license

Clarity TagFlow is a ground-up Rust rewrite of the Java app **terminus2** — same workflow, native speed, and a stack of new AI features on top. See [EDITIONS.md](EDITIONS.md) for the full comparison.

Licensed under the [MIT License](LICENSE).

*Same workflow. Native speed. More AI.* 🦀✨
