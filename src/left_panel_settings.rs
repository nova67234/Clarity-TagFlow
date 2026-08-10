//! The "Filter Settings" panel — a Rust port of terminus2's `SettingsLeftPanel`.
//! It narrows the left browser to a single media type (or favorites).
//!
//! Like the Java version it lives in the left browser, opened by the gear button
//! in the search bar (see `left_browser.rs`). Here it's a floating popup anchored
//! to the gear (the app supplies the rounded card frame around this content), so
//! it matches the app's layout while behaving like a real popup — a second gear
//! click or a click outside dismisses it. The chosen filter is applied to the
//! browser list — see `ViewerApp::update_filtered`.

use eframe::egui;

use crate::theme::TEXT;

/// Which media the browser list is narrowed to. Multi-select: the type kinds
/// (Images/Videos/GIFs) union together, while Favorites intersects — so
/// Images + Favorites means "favorite images". Nothing selected (the default)
/// is the "All" state: everything shows.
///
/// Not persisted (`#[serde(skip)]` on the field in `Settings`): like the Java
/// dialog, the filter resets to `All` on each launch so a stored "Favorites" can't
/// make the browser look empty after a restart.
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct MediaFilter {
    pub images: bool,
    pub videos: bool,
    pub gifs: bool,
    pub favorites: bool,
}

impl MediaFilter {
    /// True when nothing is selected — the "All" state.
    pub fn is_all(self) -> bool {
        !(self.images || self.videos || self.gifs || self.favorites)
    }

    /// Label + toggle for each option, in display order.
    pub fn toggles(&mut self) -> [(&'static str, &mut bool); 4] {
        [
            ("Images", &mut self.images),
            ("Videos", &mut self.videos),
            ("GIFs", &mut self.gifs),
            ("Favorites", &mut self.favorites),
        ]
    }
}

/// Render the Filter Settings popup contents. `filter` is the live media-type
/// filter; mutating it here re-filters the browser (the app watches for changes).
/// The caller wraps this in the popup's rounded card frame.
pub fn panel(ui: &mut egui::Ui, filter: &mut MediaFilter) {
    ui.horizontal(|ui| {
        ui.add(
            egui::Image::new(egui::include_image!("../icons/filter.svg"))
                .fit_to_exact_size(egui::vec2(16.0, 16.0))
                .tint(crate::theme::icon_tint(TEXT())),
        );
        ui.label(
            egui::RichText::new("Filter Settings")
                .color(TEXT())
                .strong()
                .size(14.0),
        );
    });

    ui.add_space(8.0);
    // Main-settings-style switch rows (label left, toggle right, inset
    // hairlines between). Multi-select: each kind toggles independently. "All"
    // reflects the empty selection — switching it on clears the rest; it can't
    // be switched off directly (deselect the kinds instead).
    crate::settings::card(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("All").color(TEXT()).size(13.0));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut on = filter.is_all();
                if crate::settings::switch(ui, &mut on).changed() && on {
                    *filter = MediaFilter::default();
                }
            });
        });
        for (label, on) in filter.toggles() {
            crate::settings::row_sep(ui);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(label).color(TEXT()).size(13.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    crate::settings::switch(ui, on);
                });
            });
        }
    });
}
