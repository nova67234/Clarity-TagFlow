# Clarity TagFlow (Rust) vs. terminus2 (Java) — What's New

Clarity TagFlow is the ground-up Rust rewrite of **terminus2**, the original Java
Swing image-dataset tagger/viewer. This document is the detailed "what do I get
by switching" comparison: what the Rust version adds that the Java app never
had, what got dramatically better, and (honestly) what the Java app still does
that hasn't been ported yet.

---

## At a glance

| | terminus2 (Java) | Clarity TagFlow (Rust) |
|---|---|---|
| UI framework | Swing + FlatLaf, JavaFX embeds | egui/eframe (GPU-rendered, immediate mode) |
| Language / runtime | Java 21 on the JVM, 2 GB heap pre-allocated | Native Rust binary, no VM, no heap ceiling |
| Codebase | 105 classes | 37 modules, ~22,000 lines |
| Install size | ~300–400 MB (jpackage bundles a JRE) | A few tens of MB; proper Inno Setup installer |
| Startup | JVM warm-up | Near-instant, with an animated splash |
| Platforms | Windows-centric | Windows installer, macOS .dmg (Apple Silicon), Linux .tar.gz — built by CI on every release tag |
| AI inference | ONNX Runtime via Java bindings | ONNX Runtime statically staged with the app, Level-3 graph optimization |

---

## Brand-new features (not in the Java version at all)

### 1. Gallery layout
A second, full-window **masonry grid** layout (Pinterest-style balanced
columns) selectable in Appearance settings, alongside the classic 3-panel
layout. Includes:
- Lazy, viewport-only thumbnail decoding — a 10,000-image folder scrolls smoothly.
- Click-to-open **detail popup** with the full image on the left and
  Details / Civitai info on the right.
- A floating, draggable **search pill** anchored to the bottom-right corner
  (remembers its spot relative to the corner across resizes and restarts),
  with a built-in media-type filter and thumbnail-size slider.

### 2. Camera RAW support (DNG / Sony ARW / Canon CR2 / Nikon NEF)
The Java app couldn't open raw files at all. The Rust version runs a **full
pure-Rust develop pipeline** (zenraw): camera matrix, as-shot white balance,
white-preserving color transform, then an affine color match against the
embedded preview so hues land exactly where the camera vendor intended.
Plus **Radiance HDR** (.hdr) with tone mapping. All decode off-thread.

### 3. Spatial Scene (depth parallax)
An Apple-Photos-style **3D depth effect for ordinary photos**: Depth Anything
V2 runs in ONNX to estimate a depth map, and the viewer warps the image on a
96×64 mesh that follows the mouse with idle sway. Entirely new capability.

### 4. AI background removal
Right-click → **Remove Background**: BiRefNet (ONNX) computes a saliency
matte and writes a transparent-background PNG next to the original, with live
progress overlay. The Java app had no image-editing AI at all.

### 5. Pixal3D — image → 3D model
One image in, a textured **GLB 3D model** out. The app self-installs an
isolated Python + CUDA PyTorch runtime, DINOv3, BiRefNet and nvdiffrast,
runs the full sparse-structure → shape → texture pipeline with tunable
guidance/steps/decimation, and displays the result in a built-in **3D viewer**
(orbit, zoom, PBR lighting) rendered inside the app's own OpenGL context.
Subprocesses are tied to a Windows Job Object so nothing survives app exit.

### 6. Local text-to-image generation (Flux / Z-Image Turbo)
The Java app could only *talk to* an existing A1111/ComfyUI server. The Rust
app **installs and manages its own ComfyUI backend**, downloads GGUF-quantized
Flux.1 (schnell/dev) and Z-Image Turbo models, and gives you prompt/negative/
steps/guidance/seed controls with live logs, progress, and per-model output
history — zero manual setup.

### 7. Deep Scan ("Find Issues")
A folder health tool the Java app never had:
- **Corruption scan** — actually decodes every image and lists the failures
  for review/deletion.
- **Exact-duplicate finder** — SHA-256 hashes everything and moves duplicates
  (keeping one) into a `duplicates/` folder.
- Background-threaded, cancellable, with a collapsible log.

### 8. New themes and animated backdrops
Java had 4 static FlatLaf palettes. Rust has 5 themes including three that
didn't exist before:
- **Space** — transparent gutters reveal an animated twinkling starfield.
- **Aurora** — pastel aurora blobs drifting behind a light theme.
- **Glass** — frosted translucent panels over a user-picked background color,
  with a choice of Solid / Starfield / Aurora backdrop.

### 9. Gelbooru bulk downloader (rebuilt and hardened)
The Java app had a basic Gelbooru fetcher. The Rust downloader adds:
- **Tag-role sidecars** — every downloaded file gets a `.txt` with tags
  classified as artist/character/copyright/general via the tag-info API.
- **Encrypted 2000/day cap** and a 3-second minimum request delay (good-citizen
  rate limiting; the counter itself is DPAPI-encrypted so it can't be reset by
  editing a file).
- Blacklist, file-type toggles, dedup log, live progress + log.

### 10. SD generation metadata, decoded properly
Java did regex scraping of PNG text chunks. Rust has a real parser
(`sd_metadata.rs`) that handles **A1111 `parameters` blocks, ComfyUI
prompt/workflow JSON, and Civitai metadata**, and additionally byte-scans
JPEG/WebP/AVIF EXIF `UserComment` (UTF-8/UTF-16 both endians) — so generation
data is found in *any* container, not just PNG. A switchable Tags/Metadata box
shows it, and the raw block feeds the Civitai panel including Hashes/TI-hashes.

### 11. Quality-of-life additions with no Java equivalent
- **Crop tool** (drag a rectangle, saves beside the original).
- **Copy image** that actually pastes into Chrome/Gemini (adds legacy CF_DIB
  alongside PNG on the clipboard).
- **Dominant color palette** extracted per image in the details card.
- **Video metadata reader** — pure-Rust ISO-BMFF parser pulls resolution,
  duration and codec from MP4/MOV/M4V headers without touching the stream.
- **Movable popups** — drag any floating panel by its top strip; positions are
  remembered across runs (toggleable in settings).
- **Color emoji** (Twemoji SVGs) and **CJK + math/symbol font fallback** so
  Japanese/Chinese/Korean tags and fancy Unicode prompts never render as tofu.
- **Animated cursive splash screen** (skippable).
- **HD thumbnails** toggle (768 px tiles for high-DPI displays).
- **Live CPU/RAM graphs** in the top bar (120-point rolling history).

---

## Same feature, dramatically better in Rust

### AI tagging
- Same model families (WD14 v3 SwinV2/EVA02/ConvNext, JoyTag, PixAI v0.9)
  but with a **model manager UI**: catalogued downloads with progress bars into
  the per-user data dir, automatic discovery of models already downloaded for
  terminus2.
- ONNX Runtime is staged with the app (no separate install) and runs at
  Level-3 graph optimization on a background thread — the UI never stutters
  during inference.
- The **AI orb** got a full rewrite: a 3D Fibonacci-lattice particle sphere
  with breathing/thinking/error states that morphs into a cube → torus → helix
  during long waits.

### Thumbnails & image cache
- Java: one 400-entry LRU of Swing `ImageIcon`s.
- Rust: a **dual-cache architecture** — a 400-entry browser cache (320/768 px)
  and a separate 8-entry viewer cache (2048 px, full GIF animation) with
  semaphore-gated decode permits so a huge viewer decode can never starve
  thumbnail loading. LRU eviction bounds GPU memory; in-flight requests
  dedupe; offscreen thumbnails can be unloaded immediately (toggleable).

### Backups
- Same AES-256 zip concept, but with **pre-flight decode validation** (corrupt
  images are skipped and reported, not archived), dated filenames, a live
  progress bar, and a worker wrapped in panic-catching so a failure surfaces
  as a clean error instead of a hung dialog.

### Civitai integration
- Java looked up basic model info. Rust resolves **models, LoRAs, LyCORIS,
  VAEs and embeddings** by version ID, file hash, or name; renders preview
  cards with trigger words and download links; detects "original upload"
  links; and shows a live online/offline reachability indicator. All lookups
  and preview decodes are off-thread.

### Favorites
- Both use content hashing (survives renames), but Rust shows a heart badge in
  the browser and gallery, and the store is a simple `hearted.json`.

### Video
- Both use libVLC, but the Rust build **delay-loads** it on Windows: the app
  runs fine without VLC installed, politely offers an install link, and the
  no-VLC build flag removes the dependency entirely. Poster frames are
  extracted off-thread into their own cache. Loop playback is a setting.

### Image formats
- Java (via TwelveMonkeys/native readers): PNG, JPEG, BMP, WebP, AVIF, HEIC,
  TIFF, GIF.
- Rust: PNG, JPEG, GIF, BMP, WebP, ICO, TIFF, **HDR**, plus pure-Rust
  **AVIF (rav1d)**, **HEIC**, and **camera raw** — no native codec DLLs, no
  ImageIO plugin loading order issues, everything decodes identically on all
  three OSes.

### Security
- Same DPAPI-based secret storage on Windows, now also covering the Civitai
  API key, Hugging Face token, and the Gelbooru daily-cap counter.

---

## Architecture & performance wins

- **No JVM.** No 2 GB pre-allocated heap, no GC pauses during scroll, no
  JavaFX/Swing EDT threading hazards. The Java app had known spots where
  metadata reads blocked the UI thread; in Rust every decode, scan, download,
  inference and network call is on a worker thread by design — the UI thread
  only paints.
- **GPU-rendered UI.** egui draws the whole interface as triangles each frame;
  scrolling a masonry wall of thumbnails is fundamentally smoother than
  Swing's CPU compositing.
- **Native binary distribution.** One small executable + assets vs. a shaded
  fat-JAR or a 300+ MB jpackage image. CI builds a signed-layout Windows
  installer (Inno Setup), a macOS .dmg and a Linux tarball from every `v*` tag.
- **Memory behavior.** Typical working set is a few hundred MB with caches
  full, and the caches are explicitly bounded (counted entries + LRU), not
  "whatever the GC tolerates".
- **Crash containment.** Background workers run under `catch_unwind`; a panic
  in a backup or tagger surfaces as an in-dialog error message instead of
  killing the process.

---

## Still Java-only (not yet ported)

For honesty's sake — features terminus2 has that Clarity TagFlow currently
does not:

- **LLM chat assistant** (Ollama / local llama.cpp), the role-play assistant,
  memory store/persona panel, and the chat UI (markdown, syntax highlighting,
  spell-check, attachments).
- **Text-to-speech** (Kokoro sidecar).
- **Remote browsing**: SFTP/FTP/FTPS client panel and the embedded SFTP
  receiver server.
- **Danbooru and Pexels downloaders** (Rust has Gelbooru only).
- **EXIF GPS / Geo location panel** with map links.
- **Live folder watching** (Java auto-refreshes when files appear; Rust
  requires re-opening the folder).
- **Encrypted folder archives as a browsing source** (Java could open an
  AES zip as a read-only library; Rust's encryption is backup-only).
- **Animated WebP playback** (Rust plays animated GIFs; WebP shows the first
  frame).

These are candidates for future ports — the architecture (background workers +
egui panels) has a slot for each.

---

## Bottom line

The Rust version keeps everything central to the workflow — browse, tag with
ONNX models, edit sidecars, back up, pull from boorus, inspect SD metadata —
and adds an entire creative layer the Java app never had: RAW photography
support, depth-parallax viewing, AI background removal, image-to-3D with an
in-app GLB viewer, self-hosted Flux/Z-Image generation, a duplicate/corruption
scanner, a gallery layout, and a far more polished visual identity (glass/
space/aurora themes, emoji, splash, movable popups). It does this in a binary
roughly a tenth the size of the Java bundle, with no runtime to install, on
all three desktop OSes.
