//! Inline color-emoji rendering for egui, via the bundled Twemoji SVG assets.
//!
//! egui's text rasterizer only produces monochrome glyph outlines, so true color
//! emoji can't come from a font. Instead we split a string into emoji / non-emoji
//! runs (by grapheme cluster), draw each emoji as its Twemoji SVG image (sized to
//! the surrounding text), and draw the rest as a normal styled label. This mirrors
//! the approach of the `egui-twemoji` crate, reimplemented here against this
//! project's egui version (that crate is pinned to an older egui and can't be used
//! directly). Rendering the SVG bytes needs `egui_extras`' image loaders, which
//! `main.rs` already installs.

use eframe::egui::{self, Color32};
use unicode_segmentation::UnicodeSegmentation;

/// The Twemoji SVG bytes for an emoji grapheme, if one exists.
fn emoji_svg(grapheme: &str) -> Option<&'static [u8]> {
    twemoji_assets::svg::SvgTwemojiAsset::from_emoji(grapheme).map(|a| a.as_bytes())
}

/// A stable image-cache URI for an emoji grapheme (its codepoints in hex, so the
/// `.svg` suffix routes it to egui_extras' SVG loader).
fn emoji_uri(grapheme: &str) -> String {
    let mut s = String::from("twemoji-");
    for (i, c) in grapheme.chars().enumerate() {
        if i > 0 {
            s.push('-');
        }
        s.push_str(&format!("{:x}", c as u32));
    }
    s.push_str(".svg");
    s
}

/// Render `text`, replacing emoji graphemes with inline color Twemoji images and
/// styling the remaining runs with the given colour / size / weight. Wraps to the
/// available width.
pub fn label(ui: &mut egui::Ui, text: &str, color: Color32, size: f32, strong: bool) {
    // Size the emoji image a touch larger than the font so it reads at the same
    // visual weight as the surrounding text.
    let box_size = size * 1.15;

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;

        let push_text = |ui: &mut egui::Ui, run: &mut String| {
            if run.is_empty() {
                return;
            }
            let mut rt = egui::RichText::new(run.as_str()).color(color).size(size);
            if strong {
                rt = rt.strong();
            }
            ui.label(rt);
            run.clear();
        };

        let mut run = String::new();
        for g in UnicodeSegmentation::graphemes(text, true) {
            match emoji_svg(g) {
                Some(bytes) => {
                    push_text(ui, &mut run);
                    let src = egui::ImageSource::Bytes {
                        uri: emoji_uri(g).into(),
                        bytes: egui::load::Bytes::Static(bytes),
                    };
                    ui.add(egui::Image::new(src).fit_to_exact_size(egui::vec2(box_size, box_size)));
                }
                None => run.push_str(g),
            }
        }
        push_text(ui, &mut run);
    });
}
