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

use crate::theme::{MUTED, TEXT};

/// Which media type the browser list is narrowed to. Single-select, mirroring the
/// Java `MediaTypeFilter`. `All` (the default) shows everything.
///
/// Not persisted (`#[serde(skip)]` on the field in `Settings`): like the Java
/// dialog, the filter resets to `All` on each launch so a stored "Favorites" can't
/// make the browser look empty after a restart.
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MediaFilter {
    #[default]
    All,
    Images,
    Videos,
    Gifs,
    Favorites,
}

impl MediaFilter {
    /// Every option, in display order (matches the Java enum order).
    pub const OPTIONS: [MediaFilter; 5] = [
        MediaFilter::All,
        MediaFilter::Images,
        MediaFilter::Videos,
        MediaFilter::Gifs,
        MediaFilter::Favorites,
    ];

    /// The label shown in the UI.
    pub fn label(self) -> &'static str {
        match self {
            MediaFilter::All => "All",
            MediaFilter::Images => "Images",
            MediaFilter::Videos => "Videos",
            MediaFilter::Gifs => "GIFs",
            MediaFilter::Favorites => "Favorites",
        }
    }
}

/// Render the Filter Settings popup contents. `filter` is the live media-type
/// filter; mutating it here re-filters the browser (the app watches for changes).
/// The caller wraps this in the popup's rounded card frame.
pub fn panel(ui: &mut egui::Ui, filter: &mut MediaFilter) {
    ui.label(
        egui::RichText::new("Filter Settings")
            .color(TEXT())
            .strong()
            .size(14.0),
    );

    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("MEDIA CONTENT")
            .color(MUTED())
            .strong()
            .size(10.0),
    );
    ui.add_space(6.0);

    // Single-select rows, mirroring the Java filter's options.
    for (idx, opt) in MediaFilter::OPTIONS.iter().enumerate() {
        if idx > 0 {
            ui.add_space(4.0);
        }
        ui.radio_value(filter, *opt, egui::RichText::new(opt.label()).color(TEXT()));
    }
}
