# Clarity TagFlow — UI Style Guide

How pop-up dialogs and panels should look and lay out. Use this as the reference
when adding any new window, popup, or dialog so everything stays consistent with
the Settings window, the Tag Manager settings, the AI Model Manager ("Get
Models"), and the Backup dialog.

> Quick rule of thumb: **window/section cards = rounded 22**, **buttons/inputs =
> rounded 10–12**, **checkboxes = rounded 4**, **pop-up action buttons = 90 × 32**.

---

## 1. Colors (the theme)

Defined in `src/main.rs` → `mod theme`. Always pull these from `crate::theme::*`
rather than hard-coding RGB.

| Name | RGB | Use |
|------|-----|-----|
| `BG` | 24, 24, 26 | App background (darkest). |
| `PANEL` | 32, 32, 34 | Window/dialog body fill, cards. |
| `FIELD` | 45, 47, 50 | Section cards inside a dialog, input wells. |
| `TEXT` | 235, 235, 235 | Primary text. |
| `MUTED` | 170, 170, 170 | Secondary text, labels, hints, section titles. |
| `ACCENT1` | 64, 140, 255 | Primary buttons, selection, links, active state. |
| `EDGE` | premultiplied 18,18,18 @ 20 | Faint 1px outline around rounded panels. |

Status colors (from `ai_models.rs`, reuse for badges/results):
- **Green** = `Color32::from_rgb(46, 160, 67)` — success / installed.
- **Red** = `Color32::from_rgb(220, 70, 70)` — error / not installed.
- **Warning amber** = `Color32::from_rgb(220, 180, 90)` — non-fatal warnings.
- **Error text** = `Color32::from_rgb(235, 110, 110)` — inline form errors.

---

## 2. Rounded corner sizes (THE important part)

| Element | Corner radius | Notes |
|---------|---------------|-------|
| **Window / dialog frame** | **22** | Outer body of every pop-up. |
| **Section cards** (grouped controls inside a dialog) | **22** | `FIELD` fill + `EDGE` stroke. |
| **Tag list / large content boxes** | **22** | e.g. the Tag Manager tag list. |
| **Header bars / pills inside a panel** | **18** | e.g. Tag Manager header bar. |
| **Buttons** | **10–12** | 10 for pop-up footer buttons, 12 for full-width/in-panel buttons. |
| **Text inputs / combo boxes** | **8–12** | Match the buttons near them. |
| **Checkboxes** | **4** | Square with slightly rounded edges — NOT pills. |
| **Small badges / tab pills** | **8–9** | Status pills, tab selectors. |

### Why checkboxes need a manual fix
The global theme rounds widgets into pills. In any dialog with checkboxes,
override just the corner radius to `4` so they're square-with-round-edges (see
Settings window and Backup dialog):

```rust
let sq = egui::CornerRadius::same(4);
for w in [
    &mut style.visuals.widgets.noninteractive,
    &mut style.visuals.widgets.inactive,
    &mut style.visuals.widgets.hovered,
    &mut style.visuals.widgets.active,
    &mut style.visuals.widgets.open,
] {
    w.corner_radius = sq;
}
```
Leave the checkmark color and fills at the theme default — don't darken them.

---

## 3. Button sizes

### Pop-up / dialog footer buttons (the default)
**90 × 32**, corner radius **10**, 8px gap between buttons, right-aligned. Matches
the Tag Manager settings Save/Cancel buttons exactly.

**Keep the app's hover glow — don't hardcode `.fill()`.** A button with a fixed
`.fill(color)` shows that one color in *every* state (idle/hover/press), which
kills the hover glow. The app's panel buttons (right-details Copy/Edit/Move) are
just `egui::Button::new(label)` with **no fill** — that's what gives them the
theme's built-in per-state hover brighten. So for footer buttons, prefer a plain
themed button; only set a fill for a deliberate accent or a flash.

### Click-flash confirmation (the app's affirm/cancel language)
Primary/secondary dialog actions **flash a colour on click**, then fire the action
after ~450ms — the established pattern from `tag_manager_settings.rs`:
- **Primary (Create/Save)** → flashes **green** `rgb(46,160,67)` then proceeds.
- **Secondary (Cancel)** → flashes **red** `rgb(200,55,55)` then closes.

Implement with an `Option<Instant>` timer per button: on click set the timer;
while `Some`, draw the button with the flash fill and `request_repaint()`; once
`elapsed() >= 450ms`, run the action and clear the timer. The flash is a *click*
confirmation, not a hover effect — hovering still shows the normal theme glow.

### Other button sizes (for reference, not the pop-up default)
- **In-panel full-width action row** (e.g. Tag Manager Add/Remove): height **35**,
  width split evenly across the panel, radius **12**.
- **Utility buttons** (e.g. "Get Models"): **110 × 30**, radius **12**.
- **Icon-only buttons**: icon ~16–20px inside a borderless button.

---

## 4. Pop-up dialog layout (the standard recipe)

Two kinds of pop-ups:
- **Centered window** (Settings, Backup) — floats in the middle, modal-feeling.
- **Anchored popup/dropdown** (Get Models, Tag Manager settings) — drops down
  under the button that opened it.

### Centered window structure (top → bottom)
1. **Title** — lives in the window title bar only. Do NOT repeat it in the body.
   Make it phase-aware if the dialog has stages (e.g. "New Backup" → "Creating
   Backup" → "Backup Complete").
2. **Sub-header** — one muted line (size 11) describing the dialog. No bold title
   above it (the title bar already has the name).
3. **Section cards** — each is a titled group: a `MUTED` strong size-12 title,
   then a `FIELD`-filled, radius-22 frame with `EDGE` stroke holding the controls.
4. **Inline error line** (if any) — error-text color, size 12, above the footer.
5. **Separator** then the **footer** with right-aligned buttons.

### Window frame spec
```rust
egui::Frame::new()
    .fill(PANEL)
    .corner_radius(egui::CornerRadius::same(22))
    .inner_margin(egui::Margin::same(16))
    .stroke(egui::Stroke::new(1.0, EDGE))
    .shadow(egui::epaint::Shadow {
        offset: [0, 4], blur: 16, spread: 0,
        color: egui::Color32::from_black_alpha(140),
    })
```

### Section card spec
```rust
egui::Frame::new()
    .fill(FIELD)
    .corner_radius(egui::CornerRadius::same(22))
    .inner_margin(egui::Margin::symmetric(14, 12))
    .stroke(egui::Stroke::new(1.0, EDGE))
```

### Standard width — use a fixed constant, never infinity
Body width ~**380** for centered dialogs. Define it once and derive child widths
from it (window margin 16 + card margin 14 each side → inner ≈ 320):
```rust
const DIALOG_WIDTH: f32 = 380.0;
const FIELD_WIDTH:  f32 = DIALOG_WIDTH - 16.0 * 2.0 - 14.0 * 2.0; // ≈ 320
```
Inside the window: `ui.set_width(DIALOG_WIDTH); ui.set_max_width(DIALOG_WIDTH);`
(set BOTH — see the infinity gotcha below). Anchored popups use `.width(...)` on
the popup builder (~320–400).

---

## 5. Critical layout gotchas

### ⚠️ NEVER use infinite width in a centered/persisted window (causes a NaN crash)
This one cost hours. In an **auto-sized** `egui::Window`, the available width is
**infinite**. If any child reports an infinite width, that infinity propagates to
the window's `last_content_size`, which eframe (with the `persistence` feature)
**saves to disk as `inf`** — and on the next launch it reloads as **NaN**, then
panics deep in egui (`max_rect is not NaN: [[NaN ...]]`) *before any of your code
runs*. The app then crash-loops every launch until you delete the saved state.

The infinity sneaks in through **any** of these — avoid all of them in a dialog:
- `widget.desired_width(f32::INFINITY)` — use `.desired_width(FIELD_WIDTH)`.
- `ui.set_width(ui.available_width())` — use `ui.set_width(FIELD_WIDTH)`.
- `allocate_ui_with_layout(vec2(ui.available_width(), h), …)` — don't pass an
  infinite extent into a layout; use `with_layout` (below) or a fixed width.
- `ProgressBar::new(..).desired_width(f32::INFINITY)` — give it `FIELD_WIDTH`.

Also: cap the width on BOTH sides. `ui.set_width(380.0)` only sets the *minimum*
in egui 0.34 — the max stays infinite — so always pair it with
`ui.set_max_width(380.0)`.

**Recovering from a poisoned state file:** the persisted file lives at
`%APPDATA%\<App Name>\data\app.ron` (the app name is the `run_native` title, e.g.
`Clarity TagFlow — Image Viewer`). If it contains `inf` or `NaN`, delete it; the
app writes a fresh one on next launch.

### No dead space below the footer
A bare `ui.with_layout(right_to_left, …)` is the right tool for a footer and does
NOT cause dead space *as long as the window width is fixed* (see above). Because
the dialog is a fixed `DIALOG_WIDTH` with auto height, the window shrinks/grows to
its real content — conditional sections (e.g. the password fields) just work:

```rust
ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
    if accent_button(ui, "Create").clicked() { /* ... */ }
    ui.add_space(8.0);
    if secondary_button(ui, "Cancel").clicked() { /* ... */ }
});
```
> Do NOT "pin" the footer with `allocate_ui_with_layout(vec2(ui.available_width(),
> 32.0), …)` — in an auto-sized window that available width is infinite and
> triggers the NaN crash above. A fixed window width is what actually removes the
> gap.

### No window X button on dialogs with their own buttons
Don't use `.open(&mut open)` if the dialog has Cancel/Close buttons in the body —
that adds a redundant title-bar X. Dismiss via the in-body buttons only. (A
dialog with a running background job MUST be closed via Cancel so the worker is
never orphaned.)

### Inputs that should "blend" into a card
Set `style.visuals.extreme_bg_color = FIELD` so a text box matches the card it
sits on, then give it a thin outline (`Stroke::new(1.0, Color32::from_gray(80))`,
brightening to ~110 on hover) so it still reads as an input. Add a touch of
vertical padding for height: `.margin(egui::Margin::symmetric(8, 7))`.

---

## 6. Spacing conventions
- Between a label and its control: `add_space(4.0)`.
- Between controls in a card: `add_space(6.0–8.0)`.
- Between section cards: `add_space(10.0)`.
- Before the footer: `add_space(12.0)` → `ui.separator()` → `add_space(10.0)`.
- Footer button gap: `add_space(8.0)`.

---

## 7. Text sizes
- Window/dialog title (title bar): default window title.
- Section titles: **12**, `MUTED`, strong.
- Body labels: **12–13**, `MUTED` for secondary, `TEXT` for primary.
- Hints (explanatory lines under a control): **11**, `MUTED`.
- Button labels: **14**.
- Result/summary headings inside a card: **15–16**, `TEXT`, strong.

---

## 8. Background work + progress
Long actions (zipping, downloading, tagging) run on a `std::thread`, never on the
UI thread. Share state via a struct of atomics + `Mutex` (`current`, `total`,
`done`, `cancel`, a label, and an outcome); the dialog reads it each frame and
calls `request_repaint()` while a job is in flight so the bar animates.

- **Determinate bar:** set `total` after the scan, bump `current` per item, draw
  `ProgressBar::new(cur/total).text("{cur} / {total}")`. Show an indeterminate
  `.animate(true)` bar while `total == 0` (still scanning).
- **Cancel** is cooperative: a `cancel: AtomicBool` the worker checks in its loop.
- **Wrap the worker in `std::panic::catch_unwind`** so a bad item surfaces as a
  clean error in the dialog instead of silently killing the thread (UI spins
  forever) or aborting the process.

---

## 9. Reference implementations
When in doubt, copy the patterns from these files:
- `src/settings.rs` — the canonical centered window + `section()` / `hint()`.
- `src/backup.rs` — multi-phase centered dialog, fixed width, background worker
  with progress + cancel, click-flash buttons.
- `src/ai_models.rs` — anchored popup, badges, progress bar.
- `src/tag_manager_settings.rs` — anchored popup, 90×32 footer buttons, click-flash.
