//! Web research in the chat view (the looking up itself is src/research.rs):
//! the Tools menu's switch and the composer's globe, the sources card under
//! a researched reply, and the window that shows a lookup's notes.

use std::io::Read;
use std::path::{Path, PathBuf};

use eframe::egui;

use crate::llm::LlmState;
use crate::research::{Outcome, Provider, Report, Source};
use crate::settings::Settings;
use crate::theme::*;

/// Web research's own colour: the globe and the source numbers.
fn ink() -> egui::Color32 {
    if crate::theme::is_light() {
        egui::Color32::from_rgb(0, 118, 124)
    } else {
        egui::Color32::from_rgb(70, 196, 194)
    }
}

const ERROR_RED: egui::Color32 = egui::Color32::from_rgb(210, 70, 70);

fn globe(ui: &mut egui::Ui, size: f32, tint: egui::Color32) -> egui::Response {
    ui.add(
        egui::Image::new(egui::include_image!("../../icons/globe.svg"))
            .fit_to_exact_size(egui::vec2(size, size))
            .tint(tint),
    )
}

/// Where the chosen service looks, for sentences ("searches Wikipedia").
fn service_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Wikipedia => "Wikipedia",
        Provider::Tavily => "the web (Tavily)",
        Provider::Brave => "the web (Brave Search)",
        Provider::Searxng => "your SearXNG server",
    }
}

/// Web research is on and can run (it never does in Tag images).
pub(super) fn active(settings: &Settings) -> bool {
    settings.ai_web_research && !settings.ai_tag_images
}

/// The Tools menu's row: the switch, what it does, and what's still missing
/// before it can search.
pub(super) fn tools_row(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut Settings) {
    let blocked = settings.ai_tag_images;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Web research").color(TEXT()).size(13.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut on = active(settings);
            let resp = ui.add_enabled_ui(!blocked, |ui| crate::settings::switch(ui, &mut on));
            if resp.inner.changed() {
                settings.ai_web_research = on;
                llm.web_research = on;
            }
        });
    });
    let hint = if blocked {
        "Off while Tag images is on: nothing a web page says gets near your tags.".to_string()
    } else {
        format!(
            "Before answering, the AI searches {} and answers from what it finds, citing \
             numbered sources. Every lookup is saved as notes you can review. Replies take \
             longer.",
            service_name(llm.research.provider)
        )
    };
    crate::settings::hint(ui, &hint);
    if active(settings)
        && let Some(why) = llm.research.missing()
    {
        ui.add_space(2.0);
        ui.add(
            egui::Label::new(egui::RichText::new(format!("Not set up yet: {why} (the gear above).")).color(ERROR_RED).size(11.0))
                .wrap(),
        );
    }
}

/// The composer's globe while web research is on — clicking it turns it off.
pub(super) fn composer_globe(ui: &mut egui::Ui, llm: &mut LlmState, settings: &mut Settings) {
    if !active(settings) {
        return;
    }
    let tip = match llm.research.missing() {
        Some(why) => format!("Web research is on, but not set up yet: {why}. Click to turn it off."),
        None => format!(
            "Web research is on: replies are looked up on {} first. Click to turn it off.",
            service_name(llm.research.provider)
        ),
    };
    if super::icon_button_tinted(ui, egui::include_image!("../../icons/globe.svg"), 17.0, &tip, true, ink()).clicked() {
        settings.ai_web_research = false;
        llm.web_research = false;
    }
}

/// What goes under a researched reply: the sources it was given (or why it
/// had none), and its notes.
pub(super) fn card(ui: &mut egui::Ui, llm: &mut LlmState, report: &Report, streaming: bool) {
    match &report.outcome {
        Outcome::NotNeeded => quiet_line(ui, "No web search needed", "The AI decided this message needs nothing from the web."),
        Outcome::Stopped => quiet_line(ui, "Web research stopped", "Stopped before the lookup finished."),
        Outcome::Failed(why) => framed(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                globe(ui, 14.0, MUTED());
                ui.add(
                    egui::Label::new(egui::RichText::new(format!("Couldn't look this up: {why}")).color(chat_text()).size(12.5))
                        .wrap(),
                );
            });
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("The AI answered from what it already knows.").color(MUTED()).size(11.5),
                );
                notes_button(ui, llm, report, streaming);
            });
        }),
        Outcome::Searched => framed(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                globe(ui, 14.0, ink());
                ui.label(
                    egui::RichText::new(format!("Looked up on {}", report.service)).color(chat_text()).size(12.5).strong(),
                );
                notes_button(ui, llm, report, streaming);
            });
            searches_line(ui, report);
            ui.add_space(3.0);
            for s in &report.sources {
                source_row(ui, s);
            }
            if !report.skipped.is_empty() {
                let n = report.skipped.len();
                let tip: Vec<String> = report.skipped.iter().map(|s| format!("{} — {}", s.url, s.why)).collect();
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(if n == 1 {
                        "1 page couldn't be used".to_string()
                    } else {
                        format!("{n} pages couldn't be used")
                    })
                    .color(MUTED())
                    .size(11.0),
                )
                .on_hover_text(tip.join("\n"));
            }
        }),
    }
}

/// The searches made, on one line ("“eiffel tower height” · “…”").
fn searches_line(ui: &mut egui::Ui, report: &Report) {
    if report.queries.is_empty() {
        return;
    }
    let quoted: Vec<String> = report.queries.iter().map(|q| format!("“{q}”")).collect();
    let n = report.queries.len();
    let label = format!("{} {} · {}", n, if n == 1 { "search" } else { "searches" }, quoted.join(" · "));
    ui.add(egui::Label::new(egui::RichText::new(label).color(MUTED()).size(11.0)).truncate())
        .on_hover_text(format!("Searched for:\n{}", quoted.join("\n")));
}

/// A rounded, faintly tinted card the width of a short reply.
fn framed(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    let c = ink();
    egui::Frame::new()
        .fill(c.gamma_multiply(0.06))
        .stroke(egui::Stroke::new(1.0_f32, c.gamma_multiply(0.30)))
        .corner_radius(egui::CornerRadius::same(12))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width().min(560.0));
            ui.spacing_mut().item_spacing.y = 3.0;
            add(ui);
        });
}

/// A small muted line with the globe (nothing was looked up).
fn quiet_line(ui: &mut egui::Ui, text: &str, tip: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        let col = chat_text_override().or_else(chat_aurora_ink).unwrap_or(MUTED());
        globe(ui, 12.0, col);
        ui.label(egui::RichText::new(text).color(col).size(11.5)).on_hover_text(tip);
    });
}

/// One source: its number, its title (opens the page in the browser) and
/// its site.
fn source_row(ui: &mut egui::Ui, s: &Source) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 8.5, ink().gamma_multiply(0.22));
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            s.n.to_string(),
            egui::FontId::proportional(10.5),
            chat_text(),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(&s.site).color(MUTED()).size(11.0));
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let link = crate::research::link_for(&s.url);
                let col = ui.visuals().hyperlink_color;
                let resp = ui.add(
                    egui::Label::new(egui::RichText::new(&s.title).color(col).size(12.5))
                        .truncate()
                        .sense(egui::Sense::click()),
                );
                if let Some(url) = link {
                    if resp.hovered() {
                        ui.painter().hline(
                            resp.rect.x_range(),
                            resp.rect.bottom() - 1.0,
                            egui::Stroke::new(1.0_f32, col.gamma_multiply(0.7)),
                        );
                    }
                    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(&url);
                    if resp.clicked() {
                        ui.ctx().open_url(egui::OpenUrl::new_tab(url));
                    }
                }
            });
        });
    });
}

/// "Notes": opens the lookup's notes in the viewer (or says why there are
/// none), right-aligned in its row.
fn notes_button(ui: &mut egui::Ui, llm: &mut LlmState, report: &Report, streaming: bool) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match &report.notes {
        Some(Ok(path)) => {
            let resp = ui
                .add(egui::Button::new(egui::RichText::new("Notes").size(11.5)).corner_radius(egui::CornerRadius::same(9)))
                .on_hover_text(format!(
                    "Review this lookup: the searches, the answer, and the exact text the AI was \
                     given from each page.\nSaved as {}",
                    path.display()
                ));
            if resp.clicked() {
                llm.notes_view = Some(NotesView::open(path, &report.sources));
            }
        }
        Some(Err(why)) => {
            ui.label(egui::RichText::new("Notes not saved").color(ERROR_RED).size(11.0))
                .on_hover_text(format!("The notes couldn't be saved: {why}"));
        }
        // Written once the reply ends.
        None if streaming && report.has_notes() => {
            ui.label(egui::RichText::new("Notes when done").color(MUTED()).size(11.0));
        }
        None => {}
    });
}

/// A lookup's notes, open in the viewer.
pub struct NotesView {
    path: PathBuf,
    /// The notes made safe to show, or why they couldn't be read.
    text: Result<String, String>,
    cache: egui_commonmark::CommonMarkCache,
}

/// Biggest notes file the viewer shows.
const MAX_NOTES: u64 = 4 * 1024 * 1024;

impl NotesView {
    fn open(path: &Path, sources: &[Source]) -> NotesView {
        let text = read_notes(path).map(|t| crate::research::clean_answer(&t, sources));
        NotesView { path: path.to_path_buf(), text, cache: Default::default() }
    }
}

fn read_notes(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(MAX_NOTES).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The notes viewer: the notes as they're saved, rendered — with the file
/// a click away in another app or its folder. Esc or ✕ closes it.
pub(super) fn notes_window(ctx: &egui::Context, llm: &mut LlmState) {
    let Some(view) = &mut llm.notes_view else { return };
    let mut close = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    let (w, h) = crate::settings::settings_window_size(ctx.content_rect());
    egui::Window::new("Research notes")
        .id(egui::Id::new("research_notes_window"))
        .title_bar(false)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .frame(crate::settings::window_frame())
        .fixed_size(egui::vec2(w, h))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                globe(ui, 20.0, ink());
                ui.heading(egui::RichText::new("Research notes").color(TEXT()).strong().size(17.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let x = ui
                        .add(
                            egui::Button::image(
                                egui::Image::new(egui::include_image!("../../icons/close.svg"))
                                    .fit_to_exact_size(egui::vec2(24.0, 24.0))
                                    .tint(TEXT()),
                            )
                            .frame(false),
                        )
                        .on_hover_text("Close");
                    if x.clicked() {
                        close = true;
                    }
                    let exists = view.path.is_file();
                    if ui
                        .add_enabled(exists, egui::Button::new(egui::RichText::new("Show in folder").size(12.5)))
                        .clicked()
                        && let Some(dir) = view.path.parent()
                    {
                        crate::scan::open_in_default(dir);
                    }
                    if ui
                        .add_enabled(exists, egui::Button::new(egui::RichText::new("Open in another app").size(12.5)))
                        .on_hover_text("Open the Markdown file with the app your computer uses for it")
                        .clicked()
                    {
                        crate::scan::open_in_default(&view.path);
                    }
                });
            });
            let name = view.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            ui.add(egui::Label::new(egui::RichText::new(name).color(MUTED()).size(11.5)).truncate())
                .on_hover_text(view.path.display().to_string());
            ui.add_space(6.0);
            let w = ui.available_width();
            let (line, _) = ui.allocate_exact_size(egui::vec2(w, 1.0), egui::Sense::hover());
            ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0_f32, EDGE()));
            ui.add_space(6.0);
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match &view.text {
                Ok(text) => {
                    ui.scope(|ui| {
                        if let Some(body) = ui.style_mut().text_styles.get_mut(&egui::TextStyle::Body) {
                            body.size = 13.5;
                        }
                        ui.visuals_mut().override_text_color = Some(TEXT());
                        egui_commonmark::CommonMarkViewer::new().show(ui, &mut view.cache, text);
                    });
                }
                Err(e) => {
                    ui.label(egui::RichText::new(format!("Couldn't open the notes: {e}")).color(ERROR_RED).size(13.0));
                }
            });
        });
    if close {
        llm.notes_view = None;
    }
}

/// A researched reply's text to copy: the answer, then its sources.
pub(super) fn copy_text(text: &str, report: &Report) -> String {
    if report.sources.is_empty() {
        return text.to_string();
    }
    let mut out = text.trim_end().to_string();
    out.push_str("\n\nSources:\n");
    for s in &report.sources {
        out.push_str(&format!("[{}] {} — {}\n", s.n, s.title, s.url));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copied_replies_list_their_sources() {
        let report = Report {
            sources: vec![Source {
                n: 1,
                title: "Eiffel Tower".into(),
                url: "https://example.com/eiffel".into(),
                site: "example.com".into(),
                excerpts: vec![],
            }],
            ..Report::default()
        };
        let c = copy_text("It is 330 m tall [1].\n", &report);
        assert_eq!(c, "It is 330 m tall [1].\n\nSources:\n[1] Eiffel Tower — https://example.com/eiffel\n");
        assert_eq!(copy_text("Hi", &Report::default()), "Hi");
    }

    #[test]
    fn the_viewer_shows_notes_inert() {
        let dir = std::env::temp_dir().join(format!("ctf_notes_view_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("n.md");
        std::fs::write(&path, "# Notes\n\n![x](https://evil.example/p.png) [a](file:///C:/x.exe) <b>hi</b>").unwrap();
        let view = NotesView::open(&path, &[]);
        let text = view.text.unwrap();
        assert!(!text.contains("evil.example/p.png") && !text.contains("file:") && !text.contains("<b>"), "{text}");
        let missing = NotesView::open(&dir.join("gone.md"), &[]);
        assert!(missing.text.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
