//! The Text to Image view, drawn headless in the column the right panel
//! gives it and driven with clicks and typing.

use super::*;
use crate::slider::rig::Rig;

/// Where the view is drawn on the rig's screen: the column the right panel
/// gives it, 376 points wide.
const LEFT: f32 = 20.0;
const TOP: f32 = 4.0;
const WIDTH: f32 = 376.0;

/// The view of a family's tab in a column `height` tall.
struct View {
    /// The view's own faces are installed (the first frame does it, as the
    /// app does at startup: they're there from the next one).
    fonts: bool,
    state: GenerateState,
    height: f32,
    /// What the view took of its column, last frame.
    used: egui::Rect,
}

fn scene(ui: &mut egui::Ui, v: &mut View) {
    if !v.fonts {
        let mut fonts = egui::FontDefinitions::default();
        crate::tag_manager::add_fonts(&mut fonts);
        ui.ctx().set_fonts(fonts);
        v.fonts = true;
        return;
    }
    let column = egui::Rect::from_min_size(egui::pos2(LEFT, TOP), egui::vec2(WIDTH, v.height));
    v.used = ui
        .scope_builder(egui::UiBuilder::new().max_rect(column), |ui| {
            // (Through the generators' own entry, as the right panel calls it.)
            super::super::show(ui, &mut v.state, None);
            ui.min_rect()
        })
        .inner;
}

/// `family`'s view in a column `height` tall, settled: nothing generated
/// yet, and generation set up on this PC or not as `installed` says
/// (whatever the machine running the tests has).
fn view(family: GenFamily, height: f32, installed: bool) -> Rig<View> {
    INSTALLED_FOR_TEST.set(Some(installed));
    let mut state = GenerateState::new(family);
    state.gen_images.clear();
    state.hf_token.clear();
    let mut rig = Rig::new(View { fonts: false, state, height, used: egui::Rect::NOTHING }, scene);
    rig.screen = egui::vec2(640.0, height + 2.0 * TOP);
    rig.idle();
    rig
}

/// Where `text` is painted (once).
fn text(rig: &Rig<View>, text: &str) -> egui::Rect {
    let found = rig.texts(text);
    assert_eq!(found.len(), 1, "{text:?} is painted {} times", found.len());
    found[0]
}

/// The topmost of the places `text` is painted.
fn topmost(rig: &Rig<View>, text: &str) -> egui::Rect {
    let found = rig.texts(text);
    assert!(!found.is_empty(), "{text:?} isn't painted");
    found.into_iter().min_by(|a, b| a.top().total_cmp(&b.top())).unwrap()
}

fn shown(rig: &Rig<View>, text: &str) -> bool {
    !rig.texts(text).is_empty()
}

/// The colour `text` is painted in (once).
fn ink(rig: &Rig<View>, text: &str) -> Color32 {
    fn find(shape: &egui::Shape, text: &str, found: &mut Vec<Color32>) {
        match shape {
            egui::Shape::Text(t) if t.galley.text() == text => found.push(t.galley.job.sections[0].format.color),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, text, found)),
            _ => {}
        }
    }
    let mut found = vec![];
    rig.shapes.iter().for_each(|c| find(&c.shape, text, &mut found));
    assert_eq!(found.len(), 1, "{text:?} is painted {} times", found.len());
    found[0]
}

/// The middle of the settings bar's row of segments, from the two lines of
/// its first one.
fn segments_mid(rig: &Rig<View>) -> f32 {
    let state = &rig.state.state;
    let value = text(rig, &format!("{} × {}", state.width, state.height));
    (text(rig, "Image").top() + value.bottom()) / 2.0
}

/// Click `text` (painted once), and settle.
fn click_text(rig: &mut Rig<View>, what: &str) {
    let at = text(rig, what).center();
    rig.click(at);
    rig.idle();
}

/// Click a number where it's painted topmost (a slider's pill, over the
/// segment that shows the same number), type `typed` and press Enter.
fn type_over(rig: &mut Rig<View>, number: &str, typed: &str) {
    let at = topmost(rig, number).center();
    rig.click(at);
    rig.idle();
    rig.type_text(typed);
    rig.key(egui::Key::Enter);
    rig.idle();
}

/// The prompt box, as last drawn.
fn prompt_box(rig: &Rig<View>) -> egui::Rect {
    rig.state.state.prompt_rect.expect("the prompt box was drawn")
}

/// The middle of the send icon, and of the "+".
fn send_at(rig: &Rig<View>) -> egui::Pos2 {
    prompt_box(rig).right_bottom() - egui::vec2(12.0 + 14.0, 12.0 + FOOTER_H / 2.0)
}

fn plus_at(rig: &Rig<View>) -> egui::Pos2 {
    prompt_box(rig).left_bottom() + egui::vec2(12.0 + 14.0, -12.0 - FOOTER_H / 2.0)
}

/// The middle of the header's setup button.
fn gear_at() -> egui::Pos2 {
    egui::pos2(LEFT + WIDTH - 18.0, TOP + HEADER_H / 2.0)
}

fn assert_fills(rig: &Rig<View>, what: &str) {
    let used = rig.state.used;
    let height = rig.state.height;
    assert!(
        (used.top() - TOP).abs() < 0.01 && (used.height() - height).abs() < 0.01 && (used.width() - WIDTH).abs() < 0.01,
        "{what}: the view took {used:?} of a {WIDTH} × {height} column"
    );
}

/// The rows over and under the prompt box add up to the column's height
/// whatever is open under it: the box takes what they leave, and the last
/// row ends where the card does. (A row a point taller than the sum
/// assumes would push the log past the card's edge.)
#[test]
fn the_view_fills_its_card_and_no_more() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    assert_fills(&rig, "as it opens");
    let closed = prompt_box(&rig).height();

    for setting in SETTINGS {
        rig.state.state.open_setting = Some(setting);
        rig.idle();
        assert_fills(&rig, &format!("with {setting:?} open"));
        let lost = closed - prompt_box(&rig).height();
        assert!((lost - editor_height(setting)).abs() < 0.01, "{setting:?} took {lost} from the prompt box");
        // What the bar grew by is what the open setting's rows take: its
        // last row sits its own half, the space under it and half the
        // segments' height over the middle of the segments.
        let last_row = match setting {
            Setting::Image => text(&rig, "Size"),
            Setting::Steps => topmost(&rig, "28"),
            Setting::Guidance => text(&rig, "7.00"),
            Setting::Seed => topmost(&rig, "Random"),
        };
        let over = segments_mid(&rig) - last_row.center().y;
        let wanted = EDITOR_ROW_H / 2.0 + EDITOR_FOOT + (BAR_H - 2.0 * INSET) / 2.0;
        assert!((over - wanted).abs() < 2.5, "{setting:?}'s last row is {over} over the segments' middle, not {wanted}");
    }
    rig.state.state.randomize_seed = false;
    rig.idle();
    assert_fills(&rig, "with the seed's number box");
    rig.state.state.open_setting = None;

    rig.state.state.show_log = true;
    rig.idle();
    assert_fills(&rig, "with the log open");
    assert!((closed - prompt_box(&rig).height() - CONSOLE_GAP - CONSOLE_H).abs() < 0.01);

    rig.state.state.gen_images.push("a.png".into());
    rig.state.state.open_setting = Some(Setting::Image);
    rig.idle();
    assert_fills(&rig, "with the log, a setting and the session's count");
    text(&rig, "1 image(s) this session — browse them on the left.");
    assert!(prompt_box(&rig).height() >= PROMPT_MIN_H);
}

/// In a column too short for all of it the prompt box keeps its three
/// lines and the view scrolls: nothing runs past the card.
#[test]
fn a_short_card_scrolls() {
    let mut rig = view(GenFamily::Sdxl, 300.0, true);
    rig.state.state.open_setting = Some(Setting::Image);
    rig.state.state.show_log = true;
    rig.idle();
    let used = rig.state.used;
    assert!(used.bottom() <= TOP + 300.0 + 0.01 && (used.width() - WIDTH).abs() < 0.01, "the view took {used:?}");
    assert!((prompt_box(&rig).height() - PROMPT_MIN_H).abs() < 0.01);
}

/// The header says what the app does; before generation is set up here it
/// says that instead of "Ready", and the setup button's card has the way.
#[test]
fn the_status_says_when_setup_is_needed() {
    let mut rig = view(GenFamily::Sdxl, 640.0, false);
    text(&rig, "Text to Image");
    assert_eq!(ink(&rig, "Setup needed"), ACCENT1());
    assert!(!shown(&rig, "Ready"));

    rig.click(gear_at());
    rig.idle();
    assert!(rig.state.state.show_setup);
    text(&rig, "Requirements");
    text(&rig, "NVIDIA GPU required");
    text(&rig, "Setup Requirements");
    // (SDXL isn't gated: no token is asked for.)
    assert!(!shown(&rig, "HF token (dev is gated)"));
    // The button again puts the card away.
    rig.click(gear_at());
    rig.idle();
    assert!(!rig.state.state.show_setup);
    assert!(!shown(&rig, "Setup Requirements"));

    // A job's status and an error both show as they are.
    rig.state.state.status = "Downloading SDXL Base 1.0 (~6.9 GB)…".into();
    rig.state.state.running = true;
    rig.idle();
    assert_eq!(ink(&rig, "Downloading SDXL Base 1.0 (~6.9 GB)…"), ACCENT1());
    rig.state.state.running = false;
    rig.state.state.status_err = true;
    rig.state.state.status = "Setup failed".into();
    rig.idle();
    assert_eq!(ink(&rig, "Setup failed"), RED);
    // Only "Ready" gives way: what a job left there stays.
    rig.state.state.status_err = false;
    rig.state.state.status = "Cancelled".into();
    rig.idle();
    assert_eq!(ink(&rig, "Cancelled"), GREEN);
    assert!(!shown(&rig, "Setup needed"));
}

#[test]
fn set_up_it_is_ready() {
    let mut rig = view(GenFamily::Flux, 640.0, true);
    assert_eq!(ink(&rig, "Ready"), GREEN);
    assert!(!shown(&rig, "Setup needed"));
    rig.click(gear_at());
    rig.idle();
    text(&rig, "Installed");
    assert!(!shown(&rig, "NVIDIA GPU required"));
    // Flux schnell is free; dev is gated, and asks for the token there
    // (the button says so until there is one).
    assert!(!shown(&rig, "HF token (dev is gated)"));
    assert!(!needs_token(&rig.state.state));
    rig.state.state.model = GenModel::DevQ4;
    rig.idle();
    text(&rig, "HF token (dev is gated)");
    assert!(needs_token(&rig.state.state));
    rig.state.state.hf_token = " hf_example ".into();
    assert!(!needs_token(&rig.state.state));
    rig.state.state.hf_token = "  ".into();
    assert!(needs_token(&rig.state.state));
}

/// A token typed into the setup card is kept for the next launch as it's
/// typed: the card's button can put the card away with the box still being
/// typed in, and nothing is lost.
#[test]
fn a_typed_token_is_kept() {
    fn kept() -> Option<String> {
        KEPT_TOKEN_FOR_TEST.with_borrow(|kept| kept.clone())
    }
    KEPT_TOKEN_FOR_TEST.set(None);
    let mut rig = view(GenFamily::Flux, 640.0, true);
    rig.state.state.model = GenModel::DevQ4;
    rig.click(gear_at());
    rig.idle();
    // (The box is the row under its label.)
    let label = text(&rig, "HF token (dev is gated)");
    rig.click(egui::pos2(label.left() + 40.0, label.bottom() + 6.0 + 16.0));
    rig.idle();
    assert_eq!(kept(), None, "nothing is kept until something is typed");
    rig.type_text("hf_example");
    rig.frame(vec![]);
    assert_eq!(rig.state.state.hf_token, "hf_example");
    assert_eq!(kept().as_deref(), Some("hf_example"));
    assert!(!needs_token(&rig.state.state));

    rig.click(gear_at());
    rig.idle();
    assert!(!rig.state.state.show_setup);
    assert!(!shown(&rig, "HF token (dev is gated)"));
    assert_eq!(rig.state.state.hf_token, "hf_example");
    assert_eq!(kept().as_deref(), Some("hf_example"));
}

/// A status too long for its line is cut short there: it widens nothing.
#[test]
fn a_long_status_keeps_to_its_line() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    let long = "Another ComfyUI Setup or import is already running, and this sentence goes on well past the header";
    rig.state.state.status = long.into();
    rig.state.state.status_err = true;
    rig.idle();
    assert_fills(&rig, "with a long status");
    let line = text(&rig, long);
    assert!(line.right() <= LEFT + WIDTH - 36.0, "the status runs under the setup button: {line:?}");
}

/// The model bar names the family and the model once each: the family's
/// name isn't said again in the model's.
#[test]
fn built_in_models_drop_the_family_from_their_names() {
    let cases = [
        (GenModel::SdxlBase, "Base 1.0", Some("6.9 GB")),
        (GenModel::AnimaBase, "Base v1.0", Some("5 GB")),
        (GenModel::Krea2Turbo, "Turbo · 12B fp8", Some("16GB+, ~17 GB")),
        (GenModel::SchnellQ4, "schnell · GGUF Q4", Some("8GB+, fast, free")),
        (GenModel::SchnellQ8, "schnell · GGUF Q8", Some("12GB+, free")),
        (GenModel::DevQ4, "dev · GGUF Q4", Some("8GB+, gated")),
        (GenModel::DevQ8, "dev · GGUF Q8", Some("12GB+, gated")),
        (GenModel::ZImageFast, "Turbo · fp8 encoder", Some("low VRAM")),
        (GenModel::ZImageQuality, "Turbo · bf16 encoder", Some("quality")),
    ];
    for (model, name, note) in cases {
        let (got_name, got_note) = short_label(model.label(), model.family().title());
        assert_eq!((got_name.as_str(), got_note.as_deref()), (name, note), "{}", model.label());
    }
    // Every image model is covered, so a new one isn't shown unchecked.
    let all: usize = FAMILIES.iter().map(|f| GenModel::all_for(*f).len()).sum();
    assert_eq!(all, cases.len());
    // A label that isn't in that shape is shown as it is.
    assert_eq!(short_label("Something else", "SDXL"), ("Something else".to_string(), None));
    assert_eq!(short_label("SDXL", "SDXL"), ("SDXL".to_string(), None));
    assert_eq!(short_label("SDXL (beta)", "SDXL"), ("(beta)".to_string(), None));
    assert_eq!(short_label("SDXLent 2 (~1 GB)", "SDXL"), ("SDXLent 2".to_string(), Some("1 GB".to_string())));
    assert_eq!(short_label("Flux ", "Flux"), ("Flux ".to_string(), None));
}

/// The family list switches the right panel to the chosen family's tab.
#[test]
fn the_family_list_switches_tabs() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    text(&rig, "Family");
    click_text(&mut rig, "Family");
    for family in ["Anima", "Krea 2", "Flux", "Z-Image"] {
        text(&rig, family);
    }
    // (Its own name is in the bar and in the list.)
    assert_eq!(rig.texts("SDXL").len(), 2);
    // The family it's on changes nothing.
    let own = topmost(&rig, "Anima").center() - egui::vec2(0.0, 35.0);
    rig.click(own);
    rig.idle();
    assert!(rig.state.state.family_switch.is_none());
    assert!(!shown(&rig, "Anima"), "the list closed");

    click_text(&mut rig, "Family");
    click_text(&mut rig, "Flux");
    assert!(rig.state.state.family_switch == Some(GenFamily::Flux));
    assert!(!shown(&rig, "Anima"), "the list closed");
}

/// The model list picks a built-in model, with its own steps and guidance.
#[test]
fn the_model_list_picks_a_model() {
    let mut rig = view(GenFamily::Flux, 640.0, true);
    text(&rig, "Model");
    // The bar has room for the name; its note is in the list.
    text(&rig, "schnell · GGUF Q4");
    assert!(!shown(&rig, "schnell · GGUF Q4 · 8GB+, fast, free"), "a note cut short is left out");
    click_text(&mut rig, "Model");
    // (The chosen one is in the bar and in the list.)
    assert_eq!(rig.texts("schnell · GGUF Q4").len(), 2);
    text(&rig, "8GB+, fast, free");
    for (name, note) in
        [("schnell · GGUF Q8", "12GB+, free"), ("dev · GGUF Q4", "8GB+, gated"), ("dev · GGUF Q8", "12GB+, gated")]
    {
        text(&rig, name);
        text(&rig, note);
    }
    click_text(&mut rig, "dev · GGUF Q8");
    let state = &rig.state.state;
    assert!(state.model == GenModel::DevQ8);
    assert_eq!((state.steps, state.cfg), (20, 3.5));
    // (This one's note fits.)
    text(&rig, "dev · GGUF Q8 · 12GB+, gated");
    assert!(!shown(&rig, "schnell · GGUF Q8"), "the list closed");
    // The bar's segments follow.
    text(&rig, "20");
    text(&rig, "3.5");
}

/// A checkpoint's long file name is cut short in the bar: it widens nothing.
#[test]
fn a_long_checkpoint_name_keeps_to_the_bar() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    let file = "an_installed_checkpoint_with_a_file_name_that_goes_on_and_on_v12_final_fp16.safetensors";
    rig.state.state.checkpoint = Some(file.into());
    rig.idle();
    assert_fills(&rig, "with a long checkpoint name");
    let name = text(&rig, file.trim_end_matches(".safetensors"));
    assert!(name.right() <= LEFT + WIDTH - INSET - 24.0, "the name runs under the chevron: {name:?}");
}

/// The settings bar shows each setting's value, and opens one at a time.
#[test]
fn a_segment_opens_its_setting() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    for shown_text in ["Image", "1024 × 1024", "Steps", "28", "Guidance", "7.0", "Seed", "Random"] {
        text(&rig, shown_text);
    }
    assert_eq!(rig.state.state.open_setting, None);

    click_text(&mut rig, "Steps");
    assert_eq!(rig.state.state.open_setting, Some(Setting::Steps));
    // Its slider's pill, and the segment.
    assert_eq!(rig.texts("28").len(), 2);

    click_text(&mut rig, "Guidance");
    assert_eq!(rig.state.state.open_setting, Some(Setting::Guidance));
    assert_eq!(rig.texts("28").len(), 1, "the steps closed");
    text(&rig, "7.00");

    click_text(&mut rig, "Image");
    assert_eq!(rig.state.state.open_setting, Some(Setting::Image));
    for shape in ["Square", "Landscape", "Portrait", "Size"] {
        text(&rig, shape);
    }

    // The open one again closes it.
    click_text(&mut rig, "Image");
    assert_eq!(rig.state.state.open_setting, None);
    assert!(!shown(&rig, "Landscape"));
}

/// The shapes and the size set the image's width and height as the tiles
/// and the slider did: the size scales the chosen shape.
#[test]
fn shape_and_size_set_the_image() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    click_text(&mut rig, "Image");
    click_text(&mut rig, "Landscape");
    assert_eq!((rig.state.state.width, rig.state.state.height), (1216, 832));
    text(&rig, "1216 × 832");
    click_text(&mut rig, "Portrait");
    assert_eq!((rig.state.state.width, rig.state.state.height), (832, 1216));

    type_over(&mut rig, "1024", "512");
    let state = &rig.state.state;
    assert_eq!(state.size, 512);
    assert_eq!((state.width, state.height), (448, 640), "the portrait followed the size");
    text(&rig, "448 × 640");

    click_text(&mut rig, "Square");
    assert_eq!((rig.state.state.width, rig.state.state.height), (512, 512));
    // Off the slider's steps, a typed size lands on one.
    type_over(&mut rig, "512", "1000");
    assert_eq!(rig.state.state.size, 1024);
    text(&rig, "1024 × 1024");
}

#[test]
fn steps_and_guidance_take_typed_values() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    click_text(&mut rig, "Steps");
    type_over(&mut rig, "28", "33");
    assert_eq!(rig.state.state.steps, 33);
    assert_eq!(rig.texts("33").len(), 2, "the pill and the segment");
    type_over(&mut rig, "33", "500");
    assert_eq!(rig.state.state.steps, 50, "kept to the slider's range");

    click_text(&mut rig, "Guidance");
    type_over(&mut rig, "7.00", "3.5");
    assert_eq!(rig.state.state.cfg, 3.5);
    text(&rig, "3.50");
    text(&rig, "3.5");
    type_over(&mut rig, "3.50", "7.25");
    assert_eq!(rig.state.state.cfg, 7.25);
    assert_eq!(rig.texts("7.25").len(), 2, "the pill and the segment say the same");
}

#[test]
fn guidance_is_shown_without_a_last_zero() {
    for (cfg, text) in [(7.0, "7.0"), (3.5, "3.5"), (7.25, "7.25"), (1.0, "1.0"), (8.0, "8.0"), (4.1, "4.1")] {
        assert_eq!(guidance_text(cfg), text);
    }
}

/// The seed is random until that's switched off; then it's the number in
/// the box, which the segment shows.
#[test]
fn the_seed_is_random_or_the_one_typed() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    click_text(&mut rig, "Seed");
    assert_eq!(rig.texts("Random").len(), 2, "the switch's label, and the segment's value");
    assert!(rig.state.state.randomize_seed);
    assert!(!shown(&rig, "0"), "no number box while the seed is random");

    // The switch is after its label.
    let switch = topmost(&rig, "Random").right_center() + egui::vec2(12.0 + 20.0, 0.0);
    rig.click(switch);
    rig.idle();
    assert!(!rig.state.state.randomize_seed);
    assert_eq!(rig.texts("Random").len(), 1, "the segment shows the seed now");
    assert_eq!(rig.texts("0").len(), 2, "the number box and the segment");

    type_over(&mut rig, "0", "123456789");
    assert_eq!(rig.state.state.seed, 123_456_789);
    assert_eq!(rig.texts("123456789").len(), 2);
    assert_fills(&rig, "with a seed");

    // A seed as long as they come is cut short in its segment.
    rig.state.state.seed = i64::MAX;
    rig.idle();
    assert_fills(&rig, "with the longest seed");

    rig.click(switch);
    rig.idle();
    assert!(rig.state.state.randomize_seed);
    assert_eq!(rig.state.state.seed, i64::MAX, "the number is kept for when it's switched off again");
}

/// Send is there once there's a prompt. Before generation is set up, a
/// click on it opens the setup card rather than doing nothing.
#[test]
fn send_before_setup_shows_the_way() {
    let mut rig = view(GenFamily::Sdxl, 640.0, false);
    // No prompt: nothing to click where send will be.
    rig.click(send_at(&rig));
    rig.idle();
    assert!(!rig.state.state.show_setup);

    rig.state.state.prompt = "a lighthouse".into();
    rig.idle();
    rig.click(send_at(&rig));
    rig.idle();
    assert!(rig.state.state.show_setup);
    assert!(!rig.state.state.running);
    text(&rig, "Setup Requirements");
}

/// The "+" menu is the prompt box's own: the pickers and the negative
/// prompt, with how many of each are chosen.
#[test]
fn the_plus_menu_has_the_extras() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    rig.state.state.negative = "blurry".into();
    rig.click(plus_at(&rig));
    rig.idle();
    for entry in ["LoRA", "Embeddings", "Negative prompt (set)"] {
        text(&rig, entry);
    }
    assert!(!shown(&rig, "Clear imported workflow"));
    rig.key(egui::Key::Escape);
    rig.idle();
    assert!(!shown(&rig, "Embeddings"));

    // With an imported workflow the menu lets go of it, and the box says
    // which graph Generate runs.
    rig.state.state.imported_workflow = Some("{}".into());
    rig.idle();
    text(&rig, "⚡ Custom nodes active — using imported workflow");
    assert_fills(&rig, "with the import's note");
    rig.click(plus_at(&rig));
    rig.idle();
    click_text(&mut rig, "Clear imported workflow");
    assert_eq!(rig.state.state.imported_workflow, None);
    assert!(!shown(&rig, "⚡ Custom nodes active — using imported workflow"));
}

/// What the box says of an import: a file over it first, then the job's
/// own progress, then which graph Generate runs.
#[test]
fn the_box_says_what_an_import_is_doing() {
    let mut state = GenerateState::new(GenFamily::Sdxl);
    assert!(import_note(&state, false).is_none());

    state.nodes_status = Some("3 custom nodes added".into());
    let (note, color, why) = import_note(&state, false).unwrap();
    assert_eq!((note.as_str(), color, why), ("3 custom nodes added", GREEN, None));

    state.imported_workflow = Some("{}".into());
    let (note, color, why) = import_note(&state, false).unwrap();
    assert_eq!((note.as_str(), color), ("⚡ Custom nodes active — using imported workflow", GREEN));
    assert!(why.is_some_and(|why| why.contains("+ menu")));

    state.running = true;
    state.nodes_status = Some("Installing 3 custom nodes…".into());
    let (note, color, _) = import_note(&state, false).unwrap();
    assert_eq!((note.as_str(), color), ("Installing 3 custom nodes…", DROP_BLUE));
    state.nodes_status = None;
    assert!(import_note(&state, false).is_none(), "a job with nothing to say of nodes says nothing");

    let (note, color, _) = import_note(&state, true).unwrap();
    assert_eq!((note.as_str(), color), ("Drop to import prompt + install its tools", DROP_BLUE));
}

/// What an import says sits between the "+" and send, cut short there.
#[test]
fn the_imports_note_keeps_between_the_buttons() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    rig.state.state.prompt = "a lighthouse".into();
    let long = "Installing 14 custom nodes for the dropped workflow, which takes a while on a slow connection…";
    rig.state.state.nodes_status = Some(long.into());
    rig.idle();
    assert_fills(&rig, "with a long note");
    let note = text(&rig, long);
    let room = prompt_box(&rig);
    assert!(
        note.left() >= room.left() + 12.0 + 28.0 && note.right() <= room.right() - 12.0 - 28.0,
        "the note {note:?} runs under a button of {room:?}"
    );
}

/// "Log" opens the console under it; Copy only works with something logged.
#[test]
fn the_log_opens_under_its_row() {
    let mut rig = view(GenFamily::Sdxl, 640.0, true);
    assert!(!shown(&rig, "Log output will appear here."));
    click_text(&mut rig, "Log");
    assert!(rig.state.state.show_log);
    let empty = text(&rig, "Log output will appear here.");
    assert!(empty.top() > text(&rig, "Log").bottom());

    // Nothing to copy: a click on the icon does nothing.
    let copy = egui::pos2(LEFT + WIDTH - 13.0, text(&rig, "Log").center().y);
    rig.click(copy);
    rig.idle();
    assert_eq!(rig.state.state.copy_flash.map(|(_, ok)| ok), None);

    rig.state.state.log = vec!["got prompt".into(), "Prompt executed in 9.41 seconds".into()];
    rig.idle();
    text(&rig, "got prompt");
    text(&rig, "Prompt executed in 9.41 seconds");
    assert!(!shown(&rig, "Log output will appear here."));
    assert_fills(&rig, "with the log open");

    click_text(&mut rig, "Log");
    assert!(!rig.state.state.show_log);
    assert!(!shown(&rig, "got prompt"));
}

/// The video Directors keep the layout they had.
#[test]
fn the_directors_keep_their_layout() {
    let rig = view(GenFamily::Ltx, 640.0, true);
    assert!(!shown(&rig, "Text to Image"));
    text(&rig, "Prompt");
    text(&rig, "Setup Requirements");
    text(&rig, "Frames");
}
