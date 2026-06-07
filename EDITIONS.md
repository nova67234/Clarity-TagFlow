# 🦀 Clarity TagFlow (Rust Edition) vs ☕ terminus2 (Java Edition)

**Clarity TagFlow** is a ground-up **Rust** rewrite of the original **terminus2** Java app — the same image-tagging workflow you know, rebuilt as a fast, native desktop application with a stack of new AI features on top. 🚀

---

## ⚡ Why the Rust Edition?

- 🏎️ **Native speed** — compiled to a single native binary; no JVM, no warm-up.
- 🪶 **Lean & light** — lower memory use and a fast, instant startup.
- 📦 **One file, zero setup** — ships as a standalone executable; no Java runtime to install.
- 🎨 **Modern UI** — five themes including frosted-**Glass**, animated **Space**, and **Aurora**.
- 🧩 **Built-in AI** — ONNX models run in-process; one click downloads everything (no Hugging Face login needed).

---

## 📊 Feature Comparison

| Feature | ☕ Java (terminus2) | 🦀 Rust (Clarity TagFlow) |
|---|:---:|:---:|
| 🏷️ AI auto-tagging (PixAI · JoyTag · WD14 family) | ✅ | ✅ |
| 🖼️ Image viewer — zoom / pan / crop / copy | ✅ | ✅ |
| ⭐ Favorites & tag editing | ✅ | ✅ |
| ⬇️ Gelbooru downloader (tags + blacklist) | ✅ | ✅ |
| 🔎 Civitai resource-info panel | ✅ | ✅ *(improved — parses model data more reliably than the Java version)* |
| 📝 Stable Diffusion metadata reader | ✅ | ✅ |
| 🔐 Encrypted credentials (DPAPI at rest) | — | ✅ |
| 🧹 Deep Scan — corrupt-image & duplicate finder | — | ✅ |
| 🖌️ **Background removal** (BiRefNet, 1-click cutouts) | ❌ | ✅ |
| 🧊 **Pixal3D** — image → 3D model generation | ❌ | ✅ |
| 🪟 **Spatial Scene** — Apple-style depth parallax | ❌ | ✅ |
| 🎥 **Interactive 3D viewer** (orbit / zoom / PBR) | ❌ | ✅ |
| 🎬 In-app video playback (VLC) + thumbnails | ✅ | ✅ |
| 🧪 Extended formats — AVIF · HEIC · RAW (DNG/ARW/CR2/NEF) · HDR | — | ✅ |
| 🌈 Color-emoji & CJK / symbol font rendering | — | ✅ |
| 🎨 Themes (Dark · Light · Glass · Space · Aurora) | — | ✅ |
| 🤖 **LLM captioning** | ✅ | 🚧 *Coming soon* |
| 🪄 **Stable Diffusion generation** | ✅ | 🚧 *Coming soon* |

> ✅ available · 🚧 planned · ❌ not present · — not applicable / N-A

---

## 🚧 On the Rust Roadmap

These are in the Java edition today and are **coming to the Rust edition** in a future update:

- 🤖 **LLM captioning** — natural-language descriptions of your images.
- 🪄 **Stable Diffusion generation** — generate images right inside the app.

Everything else above already ships in the Rust edition. 🎉

---

## 🖥️ Requirements

- **Windows / Linux** for the full feature set. 🪟🐧
- 🟢 **No extra runtime to install** — the Windows download bundles the Microsoft Visual C++ runtime DLLs right alongside the app, so there's nothing to install: unzip and run. *(This is separate from the build tools below.)*
- 🧊 **Pixal3D** (image → 3D) needs an **NVIDIA GPU**, plus the **CUDA Toolkit + MSVC Build Tools** — these are a *compiler* (used once to build nvdiffrast's GPU plugin), not the same as the VC++ runtime. Its *Setup Requirements* button downloads everything else automatically, with **no Hugging Face login required**.
- 🍎 macOS builds run the core app; the NVIDIA-only 3D features are compiled out.

---

*Same workflow. Native speed. More AI.* 🦀✨
