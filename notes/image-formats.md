# Image Format Support

How Clarity TagFlow decodes each image format, and the reasoning behind the
AVIF/HEIC approach (which was non-obvious and worth recording).

---

## 1. What the app accepts

Defined in `src/main.rs`:
- `IMAGE_EXTENSIONS` — always available, pure-Rust via the `image` crate:
  `png, jpg, jpeg, jfif, gif, bmp, webp, ico, tif, tiff`
  (`jfif` is just JPEG with a different extension — the decoder content-sniffs it.)
- `EXTENDED_IMAGE_EXTENSIONS` — `avif, heic, heif, dng`. Only recognised when
  **both** the `avif` cargo feature is compiled in **and** the user enables
  "extended formats" in Settings. Gated so a normal build never lists files it
  can't decode.
- `VIDEO_EXTENSIONS` — listed/taggable, played via libVLC (the `vlc` feature).

`is_image()` checks `IMAGE_EXTENSIONS` always, and the extended list only when
`EXTENDED_FORMATS_ON` (a runtime `AtomicBool` mirroring the setting) is true. The
whole extended branch is `#[cfg(feature = "avif")]`, so without the feature the
code/constants don't exist and a stale persisted "on" setting can't do anything.

---

## 2. The AVIF/HEIC problem

Neither AVIF nor HEIC has support in the `image` crate without a **C library**:
- AVIF decoding in `image` = the `avif-native` feature → `dav1d` (C) → needs
  `pkg-config` + a prebuilt dav1d. Not available on this Windows box and a pain on
  CI. (The `image` crate's `avif` feature is the *encoder* only — useless here.)
- HEIC has no `image` support at all; the usual route is `libheif` (C).

So both were done with **pure-Rust** crates instead — no C toolchain, no nasm,
nothing extra for CI to install. All the decode logic lives in `src/avif.rs`
(misnamed slightly — it handles AVIF *and* HEIC), behind the `avif` feature.

Cargo features (`Cargo.toml`):
```toml
avif = ["dep:avif-parse", "dep:rav1d", "dep:heic", "dep:zenraw", "dep:enough"]
default = ["vlc", "avif"]   # on by default
```
(The `avif` feature is the umbrella for *all* extended formats, DNG included —
the name is historical.)

---

## 3. AVIF — avif-parse + rav1d (the hard one)

AVIF = an AV1 keyframe wrapped in an HEIF container.

- **`avif-parse`** — pulls the AV1 OBU bytes out of the container
  (`read_avif()` → `AvifData { primary_item, alpha_item, .. }`). Pure Rust.
- **`rav1d`** — a Rust port of dav1d that decodes the AV1 OBU to YUV. Built
  `default-features = false, features = ["bitdepth_8","bitdepth_16"]` so the
  **asm is off → no nasm needed** (slightly slower, fine for stills).

### Key gotcha: rav1d has no Rust API
rav1d only exposes the **dav1d C ABI** (`dav1d_open`, `dav1d_send_data`,
`dav1d_get_picture`, …). BUT those functions and their structs are `pub`, so we
call them **through their Rust paths** (`rav1d::src::lib::dav1d_*`,
`rav1d::include::dav1d::*`) instead of re-declaring an `extern "C"` block.

**Why that matters:** a foreign `extern "C" { fn dav1d_open... }` block does NOT
create a cargo dependency edge, so the linker never pulls in librav1d →
`unresolved external symbol` errors. Calling by Rust path links the crate AND
reuses rav1d's own struct definitions, so there's zero hand-written ABI to get
wrong. (First attempt used an `extern` block and failed to link — don't repeat.)

The decode loop: zero-init `Dav1dSettings` → `dav1d_default_settings` →
`n_threads = 1` → `dav1d_open` → `dav1d_data_create` + copy OBU in →
`dav1d_send_data` → `dav1d_get_picture` → read the `Dav1dPicture` planes. A
`Decoded` wrapper `impl Drop` calls `dav1d_picture_unref` + `dav1d_close`.

### YUV → RGBA
Handled manually in `decode_obu_to_rgba`: per-pixel sample with bit-depth
downshift (10/12-bit → 8), chroma upsampling per layout (I400/I420/I422/I444),
limited-range BT.601 coefficients, and the separate `alpha_item` OBU decoded as
luma and merged into the alpha channel.

**Caveats (it's a "make it viewable" decoder, not colour-exact):** always uses
limited-range BT.601 regardless of the file's signalled matrix/range, and
container transforms (rotate/mirror/crop) are **not** applied.

---

## 4. HEIC/HEIF — the `heic` crate (the easy one)

HEIC = HEVC (H.265) keyframe in an HEIF container.

- **`heic`** crate (imazen, pure Rust, `#![forbid(unsafe_code)]`, SIMD).
- Dead-simple API — no FFI, no manual colour conversion:
```rust
let out = heic::DecoderConfig::new().decode(&bytes, heic::PixelLayout::Rgba8)?;
RgbaImage::from_raw(out.width, out.height, out.data)
```
It returns RGBA8 directly, so it drops straight into the pipeline.

### Routing (in `decode_avif`)
- `.heic` → `decode_heic()` (HEVC) directly.
- `.heif` / `.avif` → try the AV1 path first; if `avif-parse` fails, fall back to
  `decode_heic()`. (`.heif` can legitimately be either AV1- or HEVC-coded.)

### Performance caveat (important)
Pure-Rust HEVC is **heavy** — ~seconds per 12MP photo in a debug build (much
faster in release, but still the slowest format we support). Big HEIC folders
populate thumbnails gradually. The background decode worker keeps the UI
responsive; it's a known, accepted cost of avoiding the C library.

---

## 5. DNG (camera raw) — zenraw + our own develop

DNG is a camera-raw container: a Bayer/X-Trans sensor mosaic plus metadata
(black/white levels, white-balance coefficients, a camera→XYZ colour matrix).
Turning that into a viewable sRGB image is a *develop* pipeline, not just a decode.

- **`zenraw`** (imazen, pure Rust, rawloader backend) parses the raw, normalises
  sensor data, and demosaics. Pulled in by the `avif` feature alongside `enough`
  (which supplies zenraw's `Unstoppable` cancellation token).

### Why we don't use zenraw's built-in `Develop` mode
On real DNGs `OutputMode::Develop` produces a strong **magenta cast**: its
camera→sRGB matrix isn't white-preserving — it row-normalises the XYZ→camera
matrix *before* inverting, instead of building the matrix so camera-white maps to
display-white. Neutral tones come out tinted and there's no config knob to fix it.

### `decode_dng` — two-tier strategy
1. **Full raw develop** (`decode_dng_raw`) — the path below, for DNGs zenraw can
   decode.
2. **Embedded-JPEG-preview fallback** (`decode_dng_embedded_jpeg`) — when the raw
   develop returns `None` because zenraw/rawloader can't read the file. Common
   triggers: an unsupported camera, or **lossy DNG** (TIFF compression `34892`,
   used by DJI drones and Lightroom's lossy-DNG export) which rawloader rejects
   outright. Every DNG embeds at least one camera-rendered baseline JPEG preview;
   we byte-scan for `FF D8 FF` start markers, decode each candidate, and keep the
   largest that decodes. DJI's lossy DNGs embed a **full-resolution** preview
   (e.g. test2.dng → 6720×4480), so these still display at full size — just from
   the camera's JPEG rather than a from-scratch raw develop. (We cap the scan at
   64 marker candidates so a run of false `FF D8 FF` byte coincidences inside
   compressed data can't make it loop forever.)

   The preview JPEGs are stored in *sensor* orientation and carry **no EXIF
   orientation of their own** — the rotation lives in the DNG container's TIFF
   `Orientation` tag (274). `tiff_orientation` hand-parses the TIFF IFD0 for that
   tag and we apply it to the decoded preview, so a portrait drone shot comes out
   upright (e.g. test2.dng → 4480×6720) instead of sideways.

### Full raw develop (`decode_dng_raw`)
Request `OutputMode::CameraRaw` — demosaiced, cropped, oriented camera-RGB as f32
in `[0,1]`, with **no** WB or colour matrix applied — then run our own develop:

1. **Gray-world white balance.** The as-shot WB coefficients rawloader reads from
   older DNGs are unreliable — the Canon EOS 350D sample reports `[2.25, 1.0, 1.49]`,
   which over-boosts red/blue into the same magenta cast. Rather than trust them,
   we scale R and B so the image's *mean* matches green. Metadata-independent and
   reproduces the camera's near-neutral average. (Tradeoff: gray-world slightly
   neutralises genuinely warm/cool scenes — a sunset loses a little warmth.)
2. **White-preserving camera→sRGB matrix** (`camera_to_srgb_matrix`): the dcraw
   `cam_xyz_coeff` algorithm — `cam_rgb = xyz_to_cam · XYZ_RGB`, normalise each row
   to sum to 1 (camera-white → display-white), then invert. Built from the DNG's
   own `color_matrix`.
3. **sRGB transfer curve** (`linear_to_srgb8`).

**Caveats:** the raw develop is validated against one sample (`tests/DNG/sample1.dng`,
Canon EOS 350D, full-res 3474×2314); `tests/DNG/test2.dng` (DJI, lossy DNG) goes
through the embedded-JPEG fallback. The gray-world WB in particular may want
revisiting (daylight-fixed, or derived from the embedded preview) as more cameras'
raw-decodable DNGs are tested.

---

## 6. Where the decode is wired in
All call sites route extended formats to `crate::avif::decode_avif()`:
- `image_cache::decode_still` — thumbnails + the centre viewer.
- `right_details::load_meta` — the Image Info panel's Dimensions + colour palette
  (the `image` crate can't even read AVIF/HEIC dimensions, so we decode once and
  reuse for both).
- `is_image()` / folder scan — recognition, gated on the setting.

Toggling the Settings checkbox re-scans the open folder (`main.rs` tracks
`last_extended_formats`) so files appear/disappear without reopening.

---

## 7. Build / CI notes
- `avif` is in `default` features, so normal `cargo run`, the IDE Run button, and
  the release workflows all compile rav1d + heic. **No nasm / no C libs required.**
- Trade-off: clean builds are ~30-60s longer (rav1d is large).
- The local sample sets (`/avif`, `/tests/avif`, `/tests/HEIC`, `/tests/DNG`) and
  the `*.decoded.png` smoke-test outputs are gitignored — not committed.
- Smoke tests live in `src/avif.rs` (`avif_smoke`, `heic_smoke`, `dng_smoke`) —
  they skip cleanly when the sample folders are absent, so they never break CI.

### If you ever want max AVIF speed
Enable rav1d's `asm` feature and install `nasm` on every build machine + CI
runner. Not worth it for a still-image viewer; the no-asm fallback is fine.
