//! One-time adult-content acknowledgment for sources that can return NSFW
//! material (Gelbooru, Danbooru, Wallhaven's Sketchy/NSFW purities).
//!
//! Shown the first time such a source is opened, then remembered — a marker
//! file in the app config dir, checked lazily and cached. This is an
//! attestation (the same kind of gate the booru sites themselves show), not
//! verification. Settings → General has a status row with a Reset button for
//! shared machines.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};

use eframe::egui;

use crate::theme::{ACCENT1, MUTED, TEXT};

/// Cached marker state: 0 = not checked yet, 1 = not acknowledged, 2 = acknowledged.
static STATE: AtomicU8 = AtomicU8::new(0);

fn flag_path() -> PathBuf {
    crate::download::config_dir().join("adult_ack")
}

/// Whether the user has confirmed the 18+ notice (cached after first check).
pub fn acknowledged() -> bool {
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let ok = flag_path().exists();
            STATE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Record the confirmation (marker file + cache).
pub fn acknowledge() {
    let _ = std::fs::create_dir_all(crate::download::config_dir());
    let _ = std::fs::write(flag_path(), "confirmed");
    STATE.store(2, Ordering::Relaxed);
}

/// Clear the confirmation (Settings → Reset, e.g. on a shared machine).
pub fn reset() {
    let _ = std::fs::remove_file(flag_path());
    STATE.store(1, Ordering::Relaxed);
}

/// The centred confirmation dialog, styled after the delete-confirm popup.
/// `Some(true)` = confirmed (already persisted), `Some(false)` = declined,
/// `None` = still open.
pub fn modal(ctx: &egui::Context, id: &str) -> Option<bool> {
    let mut result = None;

    egui::Window::new("Adult content")
        .id(egui::Id::new(id))
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .frame(crate::card_frame(22))
        .show(ctx, |ui| {
            ui.set_max_width(300.0);
            ui.vertical_centered(|ui| {
                let icon = egui::include_image!("../icons/age.svg");
                ui.add(
                    egui::Image::new(icon)
                        .fit_to_exact_size(egui::vec2(32.0, 32.0))
                        .tint(egui::Color32::from_rgb(220, 160, 50)),
                );
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("This source can contain adult content")
                        .size(15.0)
                        .strong()
                        .color(TEXT()),
                );
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "By continuing you confirm you are 18 or older. \
                         You'll only be asked once — this can be reset in \
                         Settings → General.",
                    )
                    .size(12.0)
                    .color(MUTED()),
                );
                ui.add_space(18.0);
                ui.horizontal(|ui| {
                    let btn_w = 110.0;
                    let gap = 12.0;
                    let total_w = btn_w * 2.0 + gap;
                    ui.add_space((ui.available_width() - total_w) / 2.0);
                    ui.spacing_mut().item_spacing.x = gap;
                    let r = egui::CornerRadius::same(8);
                    ui.visuals_mut().widgets.inactive.corner_radius = r;
                    ui.visuals_mut().widgets.hovered.corner_radius = r;
                    ui.visuals_mut().widgets.active.corner_radius = r;

                    if ui.add_sized(egui::vec2(btn_w, 30.0), egui::Button::new("Not now")).clicked() {
                        result = Some(false);
                    }
                    let ok_btn = egui::Button::new(
                        egui::RichText::new("I'm 18 or older").color(egui::Color32::WHITE),
                    )
                    .fill(ACCENT1());
                    if ui.add_sized(egui::vec2(btn_w, 30.0), ok_btn).clicked() {
                        acknowledge();
                        result = Some(true);
                    }
                });
                ui.add_space(4.0);
            });
        });

    result
}
