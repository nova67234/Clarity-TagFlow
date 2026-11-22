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
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    text(&rig, "Model");
    // The bar has room for the name; its note is in the list.
    text(&rig, "schnell · GGUF Q4");
    assert!(!shown(&rig, "schnell · GGUF Q4 · 8GB+, fast, free"), "a note cut short is left out");
    click_text(&mut rig, "Model");
    // (The chosen one is in the bar and on its tile.)
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
    assert!(!rig.state.state.model_list_open);
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
    text(&rig, "Negative prompt (set)");
    // (The LoRAs and embeddings have their own pill, beside it.)
    assert!(!shown(&rig, "LoRA"));
    assert_eq!(rig.texts("Embeddings").len(), 1, "the pill's, not the menu's");
    assert!(!shown(&rig, "Clear imported workflow"));
    rig.key(egui::Key::Escape);
    rig.idle();
    assert!(!shown(&rig, "Negative prompt (set)"));

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
    // Over the row of buttons, inside the box.
    assert!(
        note.left() >= room.left() + 12.0 && note.right() <= room.right() - 12.0,
        "the note {note:?} runs out of {room:?}"
    );
    assert!(note.bottom() <= room.bottom() - 12.0 - FOOTER_H, "the note {note:?} runs into the buttons");
    assert!(note.bottom() > room.bottom() - 12.0 - FOOTER_H - 30.0, "just over them");
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

// --- The model list's tiles. ---

/// The names' blue on the tiles.
const NAME_BLUE: Color32 = Color32::from_rgb(110, 170, 255);

/// A new, empty folder for the tiles' pictures in this test.
fn thumbs_in(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("clarity_model_thumbs_{test}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    models::THUMBS_DIR_FOR_TEST.set(Some(dir.clone()));
    dir
}

/// A `w` × `h` picture at `path`.
fn picture_file(path: &Path, w: u32, h: u32) {
    image::RgbImage::from_pixel(w, h, image::Rgb([200, 120, 40])).save_with_format(path, image::ImageFormat::Png).unwrap();
}

/// Where `text` is painted in the tiles' blue (once).
fn on_tile(rig: &Rig<View>, text: &str) -> egui::Rect {
    fn find(shape: &egui::Shape, text: &str, found: &mut Vec<egui::Rect>) {
        match shape {
            egui::Shape::Text(t) if t.galley.text() == text && t.galley.job.sections[0].format.color == NAME_BLUE => {
                found.push(t.visual_bounding_rect())
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, text, found)),
            _ => {}
        }
    }
    let mut found = vec![];
    rig.shapes.iter().for_each(|c| find(&c.shape, text, &mut found));
    assert_eq!(found.len(), 1, "{text:?} is on {} tiles", found.len());
    found[0]
}

/// The ring round the chosen tile, if the name at `name` has one.
fn ringed(rig: &Rig<View>, name: egui::Rect) -> bool {
    fn find(shape: &egui::Shape, name: egui::Rect, found: &mut bool) {
        match shape {
            egui::Shape::Rect(r)
                if r.stroke.width == 3.0
                    && r.stroke.color == crate::theme::selection_outline()
                    && r.rect.contains_rect(name) =>
            {
                *found = true
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, name, found)),
            _ => {}
        }
    }
    let mut found = false;
    rig.shapes.iter().for_each(|c| find(&c.shape, name, &mut found));
    found
}

fn right_click(rig: &mut Rig<View>, pos: egui::Pos2) {
    for pressed in [true, false] {
        rig.frame(vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Secondary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
    }
    rig.idle();
}

/// The list is tiles down a column: each model's name in blue at its foot,
/// what it's for in its corner, and the chosen one ringed.
#[test]
fn the_model_list_is_tiles() {
    thumbs_in("tiles");
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    click_text(&mut rig, "Model");
    let names = ["schnell · GGUF Q4", "schnell · GGUF Q8", "dev · GGUF Q4", "dev · GGUF Q8"];
    let at: Vec<egui::Rect> = names.iter().map(|name| on_tile(&rig, name)).collect();
    for pair in at.windows(2) {
        assert_eq!(pair[0].left(), pair[1].left(), "a column");
        // A tile's height and the gap.
        assert!((pair[1].top() - pair[0].top() - ((WIDTH - 24.0) * 0.56).round() - 12.0).abs() < 1.0, "{at:?}");
    }
    // Under its name, its note, in white.
    let note = text(&rig, "12GB+, free");
    assert!(note.top() >= at[1].bottom() - 1.0 && note.left() == at[1].left(), "{note:?} {:?}", at[1]);
    assert_eq!(ink(&rig, "12GB+, free"), Color32::from_white_alpha(200));
    // "Flux" in each tile's top left corner (and in the bar).
    let badges = rig.texts("Flux");
    assert_eq!(badges.len(), 5);
    for name in &at {
        assert!(
            badges.iter().any(|b| b.top() < name.top()
                && b.left() < LEFT + 60.0
                && name.top() - b.top() < (WIDTH - 24.0) * 0.56),
            "no badge over {name:?}"
        );
    }
    assert!(ringed(&rig, at[0]));
    for name in &at[1..] {
        assert!(!ringed(&rig, *name));
    }
}

/// The list goes as far down as the window, and scrolls past it.
#[test]
fn the_model_list_keeps_to_the_window() {
    thumbs_in("window");
    let mut rig = view(GenFamily::Flux, 640.0, true);
    click_text(&mut rig, "Model");
    let first = on_tile(&rig, "schnell · GGUF Q4");
    on_tile(&rig, "schnell · GGUF Q8");
    assert!(!shown(&rig, "dev · GGUF Q8"), "the last one is below");
    for _ in 0..10 {
        let at = first.center();
        rig.frame(vec![
            egui::Event::PointerMoved(at),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -120.0),
                phase: egui::TouchPhase::Move,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
    }
    rig.idle();
    let last = on_tile(&rig, "dev · GGUF Q8");
    assert!(last.bottom() <= rig.screen.y, "{last:?}");
    assert!(rig.state.state.model_list_open);
}

/// It closes on a click outside it, on Escape, and on its picker.
#[test]
fn the_model_list_closes_as_a_list_does() {
    thumbs_in("closes");
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    let open = |rig: &mut Rig<View>| {
        click_text(rig, "Model");
        assert!(rig.state.state.model_list_open);
        on_tile(rig, "dev · GGUF Q4");
    };
    open(&mut rig);
    // The gap between two tiles isn't outside. (Just over the second
    // tile's badge.)
    let mut badges = rig.texts("Flux");
    badges.sort_by(|a, b| a.top().total_cmp(&b.top()));
    rig.click(egui::pos2(LEFT + 100.0, badges[2].top() - 16.0));
    rig.idle();
    assert!(rig.state.state.model_list_open, "a click between tiles is in the list");
    // Outside: the header.
    rig.click(egui::pos2(LEFT + 120.0, TOP + 20.0));
    rig.idle();
    assert!(!rig.state.state.model_list_open);
    assert!(!shown(&rig, "dev · GGUF Q4"));
    open(&mut rig);
    rig.key(egui::Key::Escape);
    rig.idle();
    assert!(!rig.state.state.model_list_open);
    open(&mut rig);
    click_text(&mut rig, "Model");
    assert!(!rig.state.state.model_list_open);
    assert!(rig.state.state.model == GenModel::SchnellQ4, "nothing was picked");
}

/// A tile shows its picture: Civitai's or one the user set, else the first
/// image made with it; a tile with none (or a broken one) is blank.
#[test]
fn a_tile_shows_its_picture() {
    let dir = thumbs_in("pictures");
    picture_file(&dir.join("builtin-SchnellQ8.img"), 64, 32);
    picture_file(&dir.join("builtin-DevQ4.gen.jpg"), 32, 64);
    std::fs::write(dir.join("builtin-DevQ8.img"), b"not a picture").unwrap();
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    click_text(&mut rig, "Model");
    for _ in 0..4 {
        rig.idle();
    }
    let size = |rig: &Rig<View>, key: &str| {
        rig.state.state.model_thumbs.get(key).and_then(|t| t.tex.as_ref()).map(|t| t.size())
    };
    assert_eq!(size(&rig, "builtin-SchnellQ8"), Some([64, 32]));
    assert_eq!(size(&rig, "builtin-DevQ4"), Some([32, 64]));
    assert_eq!(size(&rig, "builtin-SchnellQ4"), None);
    assert_eq!(size(&rig, "builtin-DevQ8"), None);
    // A picture of its own goes over the first image made with it.
    picture_file(&dir.join("builtin-DevQ4.img"), 48, 48);
    rig.idle();
    rig.idle();
    assert_eq!(size(&rig, "builtin-DevQ4"), Some([48, 48]));
    // Each picture is drawn on its tile.
    let ids: Vec<egui::TextureId> =
        rig.state.state.model_thumbs.values().filter_map(|t| t.tex.as_ref().map(|t| t.id())).collect();
    let drawn = rig
        .shapes
        .iter()
        .filter(|c| matches!(&c.shape, egui::Shape::Rect(r) if r.brush.as_ref().is_some_and(|b| ids.contains(&b.fill_texture_id))))
        .count();
    assert_eq!((ids.len(), drawn), (2, 2));
}

/// A tile's right-click menu sets its picture, or takes it away — and the
/// list stays open for it.
#[test]
fn a_tiles_menu_takes_its_picture_away() {
    let dir = thumbs_in("menu");
    picture_file(&dir.join("builtin-SchnellQ4.img"), 40, 30);
    picture_file(&dir.join("builtin-SchnellQ4.gen.jpg"), 40, 30);
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    click_text(&mut rig, "Model");
    rig.idle();
    // (By the tile's right edge: the menu opens out past the list's.)
    let tile = egui::pos2(LEFT + WIDTH - 24.0, on_tile(&rig, "schnell · GGUF Q4").center().y - 60.0);
    right_click(&mut rig, tile);
    text(&rig, "Set Thumbnail…");
    assert!(text(&rig, "Remove Thumbnail").right() > LEFT + WIDTH, "the menu is out past the list");
    click_text(&mut rig, "Remove Thumbnail");
    assert!(rig.state.state.model_list_open, "the list stays open");
    on_tile(&rig, "schnell · GGUF Q4");
    assert!(!dir.join("builtin-SchnellQ4.img").exists() && !dir.join("builtin-SchnellQ4.gen.jpg").exists());
    assert!(dir.join("builtin-SchnellQ4.none").exists(), "Civitai's mustn't come back");
    assert!(rig.state.state.model_thumbs["builtin-SchnellQ4"].tex.is_none());
    // A tile with no picture has nothing to take away.
    rig.key(egui::Key::Escape);
    rig.idle();
    if !rig.state.state.model_list_open {
        click_text(&mut rig, "Model");
    }
    let tile = on_tile(&rig, "schnell · GGUF Q8").center() - egui::vec2(0.0, 60.0);
    right_click(&mut rig, tile);
    text(&rig, "Set Thumbnail…");
    assert!(!shown(&rig, "Remove Thumbnail"));
    assert!(!dir.join("builtin-SchnellQ8.none").exists());
}

/// The first image made with a model becomes its tile's picture, and stays.
#[test]
fn the_first_image_made_with_a_model_becomes_its_picture() {
    let dir = thumbs_in("first");
    let (made, later) = (dir.join("made.png"), dir.join("later.png"));
    picture_file(&made, 300, 200);
    picture_file(&later, 100, 100);
    let mut rig = view(GenFamily::Flux, 640.0, true);
    let (tx, rx) = std::sync::mpsc::channel();
    rig.state.state.rx = Some(rx);
    rig.state.state.running = true;
    let run = |rig: &mut Rig<View>, key: &str, images: &[&PathBuf]| {
        rig.state.state.run_thumb_key = Some(key.into());
        for image in images {
            tx.send(RunnerMsg::Output((*image).clone())).unwrap();
        }
        rig.frame(vec![]);
    };
    let kept = |key: &str| image::open(dir.join(format!("{key}.gen.jpg"))).ok().map(|i| (i.width(), i.height()));
    run(&mut rig, "builtin-DevQ4", &[&made, &later]);
    assert_eq!(kept("builtin-DevQ4"), Some((300, 200)));
    run(&mut rig, "builtin-DevQ4", &[&later]);
    assert_eq!(kept("builtin-DevQ4"), Some((300, 200)), "a later image doesn't replace it");
    // Not for a model whose picture was taken away, nor from a video.
    std::fs::write(dir.join("builtin-DevQ8.none"), b"").unwrap();
    run(&mut rig, "builtin-DevQ8", &[&made]);
    assert_eq!(kept("builtin-DevQ8"), None);
    run(&mut rig, "builtin-SchnellQ8", &[&dir.join("clip.webm")]);
    assert!(!dir.join("builtin-SchnellQ8.gen.jpg").exists());
    // Nor when no model's run is going.
    rig.state.state.run_thumb_key = None;
    tx.send(RunnerMsg::Output(made.clone())).unwrap();
    rig.frame(vec![]);
    assert_eq!(std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().ends_with(".gen.jpg")).count(), 1);
    // The first image large is kept small.
    let big = dir.join("big.png");
    picture_file(&big, 2000, 1000);
    run(&mut rig, "builtin-SchnellQ4", &[&big]);
    assert_eq!(kept("builtin-SchnellQ4"), Some((768, 384)));
}

/// A generation's first image is for the model picked — not a video's, and
/// not an imported workflow's.
#[test]
fn a_run_is_for_the_model_picked() {
    let mut state = GenerateState::new(GenFamily::Flux);
    assert_eq!(models::run_key(&state).as_deref(), Some("builtin-SchnellQ4"));
    state.model = GenModel::DevQ8;
    assert_eq!(models::run_key(&state).as_deref(), Some("builtin-DevQ8"));
    let mut state = GenerateState::new(GenFamily::Sdxl);
    state.checkpoint = Some("juggernautXL_v9.safetensors".into());
    assert_eq!(models::run_key(&state).as_deref(), Some("ckpt-juggernautXL_v9"));
    state.imported_workflow = Some("{}".into());
    assert_eq!(models::run_key(&state), None);
    assert_eq!(models::run_key(&GenerateState::new(GenFamily::Ltx)), None);
}

/// SDXL's installed checkpoints: under a caption, marked with what they're
/// for (when it's known), other families' behind the switch; a click picks one.
#[test]
fn installed_checkpoints_are_tiles_too() {
    thumbs_in("checkpoints");
    let found = ["sd_xl_base_1.0", "juggernautXL_v9", "my_merge", "fluxFusion_v2"]
        .map(|f| (format!("{f}.safetensors"), PathBuf::from(format!("/nowhere/{f}.safetensors"))));
    CHECKPOINTS_FOR_TEST.set(Some(found.to_vec()));
    let mut rig = view(GenFamily::Sdxl, 1400.0, true);
    rig.state.state.checkpoints = vec![
        CkptEntry { file: "sd_xl_base_1.0.safetensors".into(), base: LoraBase::Sdxl },
        CkptEntry { file: "juggernautXL_v9.safetensors".into(), base: LoraBase::Sdxl },
        CkptEntry { file: "my_merge.safetensors".into(), base: LoraBase::Unknown },
        CkptEntry { file: "fluxFusion_v2.safetensors".into(), base: LoraBase::Flux },
    ];
    click_text(&mut rig, "Model");
    let caption = text(&rig, "Installed checkpoints");
    let base = on_tile(&rig, "Base 1.0");
    let jug = on_tile(&rig, "juggernautXL_v9");
    let merge = on_tile(&rig, "my_merge");
    assert!(base.top() < caption.top() && caption.top() < jug.top() && jug.top() < merge.top());
    assert!(!shown(&rig, "sd_xl_base_1.0"), "the built-in one is listed once");
    // (In the bar, on the built-in tile and on Juggernaut's; not on the merge.)
    assert_eq!(rig.texts("SDXL").len(), 3);
    assert!(!shown(&rig, "Unknown"), "what isn't known isn't marked");
    assert!(!shown(&rig, "fluxFusion_v2") && !shown(&rig, "Flux"));
    text(&rig, "Show other-family checkpoints");
    let switch = text(&rig, "Show other-family checkpoints").right_center() + egui::vec2(WIDTH - 260.0, 0.0);
    rig.click(egui::pos2(LEFT + WIDTH - 40.0, switch.y));
    rig.idle();
    assert!(rig.state.state.show_other_ckpts);
    on_tile(&rig, "fluxFusion_v2");
    text(&rig, "Flux");
    click_text(&mut rig, "juggernautXL_v9");
    let state = &rig.state.state;
    assert_eq!(state.checkpoint.as_deref(), Some("juggernautXL_v9.safetensors"));
    assert!(state.model == GenModel::SdxlBase && !state.model_list_open);
    text(&rig, "juggernautXL_v9");
    CHECKPOINTS_FOR_TEST.set(None);
}

/// Opening the list looks up Civitai's previews for this family's files on
/// this PC that have no picture yet — not for another PC's.
#[test]
fn opening_the_list_looks_up_the_missing_previews() {
    let dir = thumbs_in("lookup");
    std::fs::write(dir.join("ckpt-juggernautXL_v9.img"), b"x").unwrap();
    std::fs::write(dir.join("ckpt-nothere.none"), b"").unwrap();
    let found = ["sd_xl_base_1.0", "juggernautXL_v9", "dreamshaper", "nothere", "fluxFusion_v2"]
        .map(|f| (format!("{f}.safetensors"), PathBuf::from(format!("/nowhere/{f}.safetensors"))));
    CHECKPOINTS_FOR_TEST.set(Some(found.to_vec()));
    let mut rig = view(GenFamily::Sdxl, 1100.0, true);
    rig.state.state.checkpoints = ["sd_xl_base_1.0", "juggernautXL_v9", "dreamshaper", "nothere"]
        .map(|f| CkptEntry { file: format!("{f}.safetensors"), base: LoraBase::Sdxl })
        .into_iter()
        .chain([CkptEntry { file: "fluxFusion_v2.safetensors".into(), base: LoraBase::Flux }])
        .collect();
    models::FETCH_FOR_TEST.set(None);
    click_text(&mut rig, "Model");
    let wanted = |keys: &[&str]| Some(keys.iter().map(|k| k.to_string()).collect::<Vec<_>>());
    assert_eq!(models::FETCH_FOR_TEST.take(), wanted(&["builtin-SdxlBase", "ckpt-dreamshaper"]));
    rig.idle();
    assert_eq!(models::FETCH_FOR_TEST.take(), None, "once an opening");
    // Shown, another family's is looked up too.
    rig.state.state.show_other_ckpts = true;
    rig.key(egui::Key::Escape);
    rig.idle();
    click_text(&mut rig, "Model");
    assert_eq!(models::FETCH_FOR_TEST.take(), wanted(&["builtin-SdxlBase", "ckpt-dreamshaper", "ckpt-fluxFusion_v2"]));
    // Another PC's models are on that PC.
    let mut state = GenerateState::new(GenFamily::Sdxl);
    state.remote_host = Some("Studio".into());
    models::opened(&state, &rig.ctx);
    assert_eq!(models::FETCH_FOR_TEST.take(), None);
    state.remote_host = None;
    models::opened(&state, &rig.ctx);
    assert_eq!(models::FETCH_FOR_TEST.take(), wanted(&["builtin-SdxlBase"]));
    CHECKPOINTS_FOR_TEST.set(None);
}

/// A tile's picture is cropped to fill it, not stretched: its middle
/// across, nearer its top than its bottom.
#[test]
fn a_picture_fills_its_tile() {
    let tile = egui::vec2(200.0, 100.0);
    let uv = models::cover_uv(egui::vec2(400.0, 100.0), tile);
    assert_eq!((uv.min.x, uv.max.x, uv.min.y, uv.max.y), (0.25, 0.75, 0.0, 1.0));
    let uv = models::cover_uv(egui::vec2(100.0, 200.0), tile);
    assert_eq!((uv.min.x, uv.max.x), (0.0, 1.0));
    assert!((uv.height() - 0.25).abs() < 1e-6 && (uv.min.y - 0.225).abs() < 1e-6, "{uv:?}");
    let uv = models::cover_uv(egui::vec2(200.0, 100.0), tile);
    assert_eq!(uv, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)));
}

/// The list opens with the chosen model in sight (from the top when it's
/// in sight there), not where it was left.
#[test]
fn the_model_list_opens_at_the_chosen_model() {
    thumbs_in("opens_at");
    let mut rig = view(GenFamily::Flux, 640.0, true);
    let wheel = |rig: &mut Rig<View>, at: egui::Pos2, dy: f32| {
        for _ in 0..10 {
            rig.frame(vec![
                egui::Event::PointerMoved(at),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, dy),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        rig.idle();
    };
    click_text(&mut rig, "Model");
    let at = on_tile(&rig, "schnell · GGUF Q4").center();
    wheel(&mut rig, at, -120.0);
    assert!(!shown(&rig, "8GB+, fast, free"), "scrolled past the first");
    rig.key(egui::Key::Escape);
    rig.idle();
    click_text(&mut rig, "Model");
    on_tile(&rig, "schnell · GGUF Q4");
    // The last one chosen, it opens there.
    rig.state.state.model = GenModel::DevQ8;
    rig.key(egui::Key::Escape);
    rig.idle();
    click_text(&mut rig, "Model");
    let last = on_tile(&rig, "dev · GGUF Q8");
    assert!(last.bottom() <= rig.screen.y);
    assert!(!shown(&rig, "8GB+, fast, free"));
    // Scrolled on from there, it stays where it's put while it's open.
    wheel(&mut rig, at, 120.0);
    text(&rig, "8GB+, fast, free");
}

// --- The tiles' loading rings. ---

/// The ring's shapes on screen: its faint tracks, and its blue arcs.
fn rings(rig: &Rig<View>) -> (usize, usize) {
    fn find(shape: &egui::Shape, found: &mut (usize, usize)) {
        match shape {
            egui::Shape::Circle(c) if c.stroke.width == 3.0 && c.stroke.color == Color32::from_white_alpha(40) => {
                found.0 += 1
            }
            egui::Shape::Path(p) if p.stroke.width == 3.0 && p.stroke.color == egui::epaint::ColorMode::Solid(NAME_BLUE) => {
                found.1 += 1
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, found)),
            _ => {}
        }
    }
    let mut found = (0, 0);
    rig.shapes.iter().for_each(|c| find(&c.shape, &mut found));
    found
}

/// A tile whose picture is on its way has a ring: a faint track while its
/// turn's to come, filling with the per cent while its file is read, turning
/// while the preview downloads. A tile with a picture has none.
#[test]
fn a_tile_rings_while_its_picture_is_on_its_way() {
    let dir = thumbs_in("rings");
    picture_file(&dir.join("builtin-SchnellQ4.img"), 40, 30);
    models::set_fetching_for_test(&["builtin-SchnellQ4", "builtin-DevQ4"], Some(("builtin-SchnellQ8", Some(0.42))));
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    click_text(&mut rig, "Model");
    rig.idle();
    // (Two tracks: the one being read, and the one whose turn's to come.)
    assert_eq!(rings(&rig), (2, 1));
    let percent = text(&rig, "42%");
    let name = on_tile(&rig, "schnell · GGUF Q8");
    assert!(percent.bottom() < name.top() && name.top() - percent.bottom() < 160.0, "{percent:?} {name:?}");
    assert!((percent.center().x - (LEFT + WIDTH / 2.0)).abs() < 2.0, "in the middle");
    assert_eq!(models::loading("builtin-DevQ4"), Some(models::Loading::Waiting));

    models::set_fetching_for_test(&["builtin-SchnellQ4", "builtin-DevQ4"], Some(("builtin-SchnellQ8", None)));
    rig.frame(vec![]);
    assert!(!shown(&rig, "42%"));
    assert_eq!(rings(&rig), (2, 1), "it turns");

    models::set_fetching_for_test(&[], None);
    rig.frame(vec![]);
    assert_eq!(rings(&rig), (0, 0));
}

/// The look-up reads each file (saying how far), then downloads its
/// preview: kept when Civitai has one, marked when it hasn't.
#[test]
fn the_look_up_reads_each_file_then_downloads_its_preview() {
    let dir = thumbs_in("look_up");
    let files = dir.join("files");
    std::fs::create_dir_all(&files).unwrap();
    let file = |name: &str, bytes: &[u8]| {
        let path = files.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    };
    // (Big enough to be read in many blocks.)
    let big = file("big.safetensors", &vec![7u8; 3 * 1024 * 1024]);
    let none = file("none.safetensors", b"b");
    let fails = file("fails.safetensors", b"c");
    let sha = |p: &Path| crate::scan::sha256_file(p).unwrap();
    let (big_sha, none_sha) = (sha(&big), sha(&none));
    let todo = vec![
        ("ckpt-big".to_string(), big.clone()),
        ("ckpt-none".to_string(), none.clone()),
        ("ckpt-fails".to_string(), fails.clone()),
    ];
    let asked = std::cell::RefCell::new(vec![]);
    let lookup = |s: &str| {
        let key = if s == big_sha { "ckpt-big" } else if s == none_sha { "ckpt-none" } else { "ckpt-fails" };
        // While it downloads, its ring turns and the rest wait their turn.
        assert_eq!(models::loading(key), Some(models::Loading::Downloading));
        asked.borrow_mut().push((key, models::loading("ckpt-fails")));
        match key {
            "ckpt-big" => Ok(Some(b"PICTURE".to_vec())),
            "ckpt-none" => Ok(None),
            _ => Err(crate::civitai::PreviewFail::Item),
        }
    };
    let read = std::cell::RefCell::new(vec![]);
    let repaint = || {
        if let Some(models::Loading::Reading(part)) = models::loading("ckpt-big") {
            read.borrow_mut().push(part);
        }
    };
    models::work(todo, &dir, &lookup, &repaint);
    assert_eq!(
        *asked.borrow(),
        [
            ("ckpt-big", Some(models::Loading::Waiting)),
            ("ckpt-none", Some(models::Loading::Waiting)),
            ("ckpt-fails", Some(models::Loading::Downloading)),
        ]
    );
    let read = read.borrow();
    assert!(read.len() > 10 && read.windows(2).all(|w| w[0] < w[1]) && *read.last().unwrap() >= 0.99, "{read:?}");
    assert_eq!(std::fs::read(dir.join("ckpt-big.img")).unwrap(), b"PICTURE");
    assert!(dir.join("ckpt-none.none").exists());
    assert!(!dir.join("ckpt-fails.img").exists() && !dir.join("ckpt-fails.none").exists(), "tried again next time");
    for key in ["ckpt-big", "ckpt-none", "ckpt-fails"] {
        assert_eq!(models::loading(key), None, "no ring is left turning");
    }
}

/// Offline, the look-up stops (and leaves no rings); so it does when the
/// list closes.
#[test]
fn the_look_up_stops_offline_or_when_the_list_closes() {
    let dir = thumbs_in("look_up_stops");
    let a = dir.join("a.safetensors");
    let b = dir.join("b.safetensors");
    std::fs::write(&a, b"a").unwrap();
    std::fs::write(&b, b"b").unwrap();
    let todo = || vec![("ckpt-a".to_string(), a.clone()), ("ckpt-b".to_string(), b.clone())];
    let asked = std::cell::Cell::new(0);
    let offline = |_: &str| {
        asked.set(asked.get() + 1);
        Err(crate::civitai::PreviewFail::Offline)
    };
    models::work(todo(), &dir, &offline, &|| {});
    assert_eq!(asked.get(), 1);
    assert_eq!((models::loading("ckpt-a"), models::loading("ckpt-b")), (None, None));
    assert!(!dir.join("ckpt-a.none").exists(), "offline isn't \"none\"");

    models::CANCEL_FOR_TEST.set(true);
    let found = |_: &str| {
        asked.set(asked.get() + 1);
        Ok(None)
    };
    models::work(todo(), &dir, &found, &|| {});
    assert_eq!(asked.get(), 1, "closed, nothing more is looked up");
    assert_eq!(models::loading("ckpt-b"), None);
    models::CANCEL_FOR_TEST.set(false);
}

// --- The text box's LoRAs | Embeddings pill and its lists. ---

use super::extras::Extra;

/// A LoRA made for `base`, with a `w` × `h` preview (none at 0 × 0).
fn lora(rig: &Rig<View>, file: &str, base: LoraBase, (w, h): (usize, usize)) -> LoraEntry {
    LoraEntry {
        file: file.into(),
        selected: false,
        strength: DEFAULT_STRENGTH,
        base,
        sniffed: true,
        info_open: false,
        info: None,
        thumb: preview(rig, file, w, h),
        // (Not looked for in this PC's folder.)
        thumb_missing: true,
        anim: None,
        anim_missing: true,
    }
}

fn embedding(rig: &Rig<View>, file: &str, base: LoraBase, (w, h): (usize, usize)) -> EmbeddingEntry {
    EmbeddingEntry {
        file: file.into(),
        base,
        sniffed: true,
        selected: false,
        negative: false,
        strength: DEFAULT_STRENGTH,
        thumb: preview(rig, file, w, h),
        thumb_missing: true,
    }
}

fn preview(rig: &Rig<View>, name: &str, w: usize, h: usize) -> Option<egui::TextureHandle> {
    (w > 0).then(|| {
        let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &vec![90; w * h * 4]);
        rig.ctx.load_texture(format!("test_preview_{name}"), image, Default::default())
    })
}

/// Flux's text box with these LoRAs (and no embeddings).
fn with_loras(test: &str, loras: &[(&str, LoraBase, (usize, usize))]) -> Rig<View> {
    thumbs_in(test);
    let mut rig = view(GenFamily::Flux, 1000.0, true);
    rig.state.state.loras = loras.iter().map(|(file, base, size)| lora(&rig, file, *base, *size)).collect();
    rig.state.state.embeddings.clear();
    rig.idle();
    rig
}

/// The pill's half for `kind`: its caption.
fn pill_half(rig: &Rig<View>, kind: &str) -> egui::Rect {
    let found: Vec<egui::Rect> = rig.texts(kind).into_iter().filter(|r| r.top() > prompt_box(rig).top()).collect();
    assert_eq!(found.len(), 1, "{kind:?} in the pill");
    found[0]
}

fn open_list(rig: &mut Rig<View>, kind: &str) {
    let at = pill_half(rig, kind).center();
    rig.click(at);
    rig.idle();
}

/// The ⓘ buttons' centres, top to bottom.
fn infos(rig: &Rig<View>) -> Vec<egui::Pos2> {
    fn find(shape: &egui::Shape, found: &mut Vec<egui::Pos2>) {
        match shape {
            egui::Shape::Circle(c) if c.radius == 14.0 && c.fill.a() >= 150 && c.fill.r() == 0 => found.push(c.center),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, found)),
            _ => {}
        }
    }
    let mut found = vec![];
    rig.shapes.iter().for_each(|c| find(&c.shape, &mut found));
    found.sort_by(|a, b| a.y.total_cmp(&b.y));
    found
}

/// Between the "+" and send: LoRAs | Embeddings, each saying how many are on.
#[test]
fn the_text_box_has_a_loras_and_embeddings_pill() {
    let mut rig = with_loras("pill", &[("a.safetensors", LoraBase::Flux, (0, 0))]);
    let (loras, embeddings) = (pill_half(&rig, "LoRAs"), pill_half(&rig, "Embeddings"));
    assert!(plus_at(&rig).x < loras.left() && loras.right() < embeddings.left() && embeddings.right() < send_at(&rig).x);
    assert!((loras.center().y - plus_at(&rig).y).abs() < 14.0, "on the buttons' row");
    assert_eq!(rig.texts("None").len(), 2);
    rig.state.state.loras[0].selected = true;
    rig.idle();
    text(&rig, "1 on");
    assert_eq!(rig.texts("None").len(), 1);
    // Each opens its own list, and a second click shuts it.
    open_list(&mut rig, "LoRAs");
    assert_eq!(rig.state.state.extras_open, Some(Extra::Loras));
    open_list(&mut rig, "Embeddings");
    assert_eq!(rig.state.state.extras_open, Some(Extra::Embeddings));
    open_list(&mut rig, "Embeddings");
    assert_eq!(rig.state.state.extras_open, None);
}

/// The LoRAs are tiles like the models', each its picture's own shape (kept
/// between half and one and a quarter its width), its name in blue at its
/// foot, its family top left and its ⓘ top right.
#[test]
fn the_loras_are_tiles_their_pictures_shape() {
    let mut rig = with_loras(
        "lora_tiles",
        &[
            ("tall.safetensors", LoraBase::Flux, (100, 400)),
            ("wide.safetensors", LoraBase::Flux, (300, 100)),
            ("none.safetensors", LoraBase::Unknown, (0, 0)),
        ],
    );
    open_list(&mut rig, "LoRAs");
    let width = WIDTH - 24.0;
    let tops: Vec<f32> = infos(&rig).iter().map(|c| c.y - 8.0 - 14.0).collect();
    assert_eq!(tops.len(), 3, "an ⓘ a tile");
    let names: Vec<egui::Rect> = ["tall", "wide", "none"].iter().map(|n| on_tile(&rig, n)).collect();
    // Tall: cut to 1.25 its width; wide: half; no picture: the models' shape.
    for (i, aspect) in [1.25, 0.5, 0.56].into_iter().enumerate() {
        let height = (width * aspect).round();
        let next_top = tops.get(i + 1).copied().unwrap_or(f32::INFINITY);
        // (A name's ink ends a little over its line's foot.)
        assert!((names[i].bottom() + 12.0 - (tops[i] + height)).abs() < 5.0, "tile {i}: {names:?} {tops:?}");
        assert!(tops[i] + height + 12.0 <= next_top + 0.5, "the gap after tile {i}");
    }
    for info in infos(&rig) {
        assert!((info.x - (LEFT + WIDTH - 12.0 - 8.0 - 14.0)).abs() < 2.0, "top right");
    }
    // What it's made for, where it's known, top left. (Not the Family bar's.)
    let badges: Vec<egui::Rect> = rig.texts("Flux").into_iter().filter(|b| b.left() > LEFT + 24.0).collect();
    assert_eq!(badges.len(), 2, "{badges:?}");
    for (badge, top) in badges.iter().zip(&tops) {
        assert!(badge.left() < LEFT + 12.0 + 20.0 && (badge.top() - top) < 20.0, "{badge:?} on the tile at {top}");
    }
}

/// A click turns a LoRA on: ringed, its weight along its foot and its name
/// just over that. A click on the weight is the weight's.
#[test]
fn a_click_turns_a_lora_on_with_its_weight_at_its_foot() {
    let mut rig = with_loras("lora_on", &[("grain.safetensors", LoraBase::Flux, (200, 200))]);
    open_list(&mut rig, "LoRAs");
    let off = on_tile(&rig, "grain");
    assert!(!ringed(&rig, off));
    assert!(!shown(&rig, "1.00"));
    rig.click(off.center() - egui::vec2(0.0, 80.0));
    rig.idle();
    assert!(rig.state.state.loras[0].selected);
    let name = on_tile(&rig, "grain");
    let weight = text(&rig, "1.00");
    assert!(ringed(&rig, name));
    assert!(weight.top() > name.bottom() && weight.left() < name.right(), "{weight:?} under {name:?}");
    assert!(name.bottom() < off.bottom() - 20.0, "the name went up over it");
    text(&rig, "1 on");
    // (The weight's own click, not the tile's.)
    rig.click(weight.center());
    rig.idle();
    assert!(rig.state.state.loras[0].selected);
    rig.click(name.center() - egui::vec2(0.0, 80.0));
    rig.idle();
    assert!(!rig.state.state.loras[0].selected);
}

/// ⓘ fills the list with the LoRA's details; back goes back to the tiles —
/// the list stays open — and a click outside it closes it.
#[test]
fn info_fills_the_list_with_its_details() {
    let mut rig = with_loras(
        "lora_info",
        &[("grain.safetensors", LoraBase::Flux, (200, 300)), ("neon.safetensors", LoraBase::Flux, (0, 0))],
    );
    rig.state.state.loras[0].info = Some(LoraInfo {
        triggers: vec!["film grain".into(), "35mm".into()],
        facts: vec![("Steps".into(), "2400".into())],
    });
    open_list(&mut rig, "LoRAs");
    let info = infos(&rig)[0];
    rig.click(info);
    rig.idle();
    assert_eq!(rig.state.state.extras_info.as_deref(), Some("grain.safetensors"));
    for words in ["Use this LoRA", "Trigger words", "film grain, 35mm", "Steps:", "2400", "Made for:", "File:"] {
        text(&rig, words);
    }
    assert!(!shown(&rig, "neon"), "the details fill the list");
    assert!(!shown(&rig, "Weight"), "off, no weight");
    // On, from here: its weight shows.
    let use_it = text(&rig, "Use this LoRA");
    rig.click(egui::pos2(LEFT + WIDTH - 12.0 - 20.0, use_it.center().y));
    rig.idle();
    assert!(rig.state.state.loras[0].selected);
    text(&rig, "Weight");
    // Back: the tiles again, and the list's still open.
    let title = text(&rig, "grain");
    rig.click(egui::pos2(title.left() - 6.0 - 18.0, title.center().y));
    rig.idle();
    assert_eq!(rig.state.state.extras_info, None);
    assert_eq!(rig.state.state.extras_open, Some(Extra::Loras));
    on_tile(&rig, "neon");
    // Outside: the header.
    rig.click(egui::pos2(LEFT + 120.0, TOP + 20.0));
    rig.idle();
    assert_eq!(rig.state.state.extras_open, None);
}

/// Favorites (right-click, or the details' heart) come first, under their
/// own caption; they're kept for next time.
#[test]
fn favorite_loras_come_first() {
    let mut rig = with_loras(
        "lora_faves",
        &[("a.safetensors", LoraBase::Flux, (0, 0)), ("b.safetensors", LoraBase::Flux, (0, 0))],
    );
    open_list(&mut rig, "LoRAs");
    assert!(!shown(&rig, "Favorites"));
    assert!(on_tile(&rig, "a").top() < on_tile(&rig, "b").top());
    let b = on_tile(&rig, "b").center() - egui::vec2(0.0, 60.0);
    right_click(&mut rig, b);
    click_text(&mut rig, "Favorite");
    let (faves, all) = (text(&rig, "Favorites"), text(&rig, "All LoRAs"));
    assert!(faves.top() < on_tile(&rig, "b").top() && on_tile(&rig, "b").top() < all.top());
    assert!(all.top() < on_tile(&rig, "a").top());
    let kept = std::fs::read_to_string(models::THUMBS_DIR_FOR_TEST.with_borrow(|d| d.clone()).unwrap().join("favorites.txt"));
    assert_eq!(kept.unwrap().trim(), "lora:b.safetensors");
    // From its details: no longer.
    let info = infos(&rig)[0];
    rig.click(info);
    rig.idle();
    let title = text(&rig, "b");
    rig.click(egui::pos2(LEFT + WIDTH - 12.0 - 18.0, title.center().y));
    rig.idle();
    assert!(!models::is_fave(&rig.state.state, "lora:b.safetensors"));
    let title = text(&rig, "b");
    rig.click(egui::pos2(title.left() - 6.0 - 18.0, title.center().y));
    rig.idle();
    assert!(!shown(&rig, "Favorites"));
}

/// Another family's LoRAs wait behind the switch at the list's foot.
#[test]
fn other_families_loras_are_behind_the_switch() {
    let mut rig = with_loras(
        "lora_others",
        &[("ours.safetensors", LoraBase::Flux, (0, 0)), ("xl.safetensors", LoraBase::Sdxl, (0, 0))],
    );
    open_list(&mut rig, "LoRAs");
    on_tile(&rig, "ours");
    assert!(!shown(&rig, "xl"));
    let switch = text(&rig, "Show other-family LoRAs (1)");
    rig.click(egui::pos2(LEFT + WIDTH - 12.0 - 20.0, switch.center().y));
    rig.idle();
    assert!(rig.state.state.show_other_loras);
    assert!(text(&rig, "For other models").top() < on_tile(&rig, "xl").top());
    text(&rig, "SDXL");
}

/// An embedding goes in the prompt it's put in: on, its chips say which.
#[test]
fn an_embedding_goes_in_the_prompt_chosen() {
    let mut rig = with_loras("embeddings", &[]);
    rig.state.state.embeddings = vec![embedding(&rig, "film_look.safetensors", LoraBase::Unknown, (0, 0))];
    open_list(&mut rig, "Embeddings");
    let name = on_tile(&rig, "film_look");
    assert!(!shown(&rig, "Negative"));
    rig.click(name.center() - egui::vec2(0.0, 60.0));
    rig.idle();
    assert!(rig.state.state.embeddings[0].selected);
    let (positive, negative) = (text(&rig, "Positive"), text(&rig, "Negative"));
    assert!(positive.left() > on_tile(&rig, "film_look").right() && positive.right() < negative.left());
    rig.click(negative.center());
    rig.idle();
    assert!(rig.state.state.embeddings[0].negative && rig.state.state.embeddings[0].selected);
    // Its details say how to use it by name.
    let info = infos(&rig)[0];
    rig.click(info);
    rig.idle();
    text(&rig, "Use it by name");
    text(&rig, "embedding:film_look");
    text(&rig, "Goes in");
}

/// With nothing installed, the list says where the files go.
#[test]
fn an_empty_list_says_where_the_files_go() {
    let mut rig = with_loras("empty", &[]);
    open_list(&mut rig, "LoRAs");
    text(&rig, "No LoRAs yet: put .safetensors files in models/loras.");
    rig.state.state.loras = vec![lora(&rig, "xl.safetensors", LoraBase::Sdxl, (0, 0))];
    rig.idle();
    text(&rig, "No Flux LoRAs here.");
    text(&rig, "Show other-family LoRAs (1)");
}

/// A favorite model comes first, under "Favorites", the rest under
/// "Built-in"; another tab sees it (they're read when a list opens).
#[test]
fn a_favorite_model_comes_first() {
    thumbs_in("model_faves");
    let mut rig = view(GenFamily::Flux, 1100.0, true);
    click_text(&mut rig, "Model");
    let q8 = on_tile(&rig, "dev · GGUF Q8").center() - egui::vec2(0.0, 60.0);
    right_click(&mut rig, q8);
    click_text(&mut rig, "Favorite");
    let faves = text(&rig, "Favorites");
    let built_in = text(&rig, "Built-in");
    assert!(faves.top() < on_tile(&rig, "dev · GGUF Q8").top());
    assert!(on_tile(&rig, "dev · GGUF Q8").top() < built_in.top() && built_in.top() < on_tile(&rig, "schnell · GGUF Q4").top());
    let mut other = GenerateState::new(GenFamily::Flux);
    models::load_faves(&mut other);
    assert!(models::is_fave(&other, "model:builtin-DevQ8"));
    // Opened again, the favorites are in sight (the chosen model is too).
    rig.key(egui::Key::Escape);
    rig.idle();
    click_text(&mut rig, "Model");
    text(&rig, "Favorites");
    on_tile(&rig, "schnell · GGUF Q4");
    // Taken back: the list as it was.
    let q8 = on_tile(&rig, "dev · GGUF Q8").center() - egui::vec2(0.0, 60.0);
    right_click(&mut rig, q8);
    click_text(&mut rig, "Remove Favorite");
    assert!(!shown(&rig, "Favorites") && !shown(&rig, "Built-in"));
}

/// The work assistant starts a run only where the Generate button could:
/// not before Text to Image is set up, and not over a run that's going. Its
/// prompt goes in the box when it does — and stays out when it doesn't.
#[test]
fn the_assistant_generates_only_when_the_view_could() {
    let ctx = egui::Context::default();
    INSTALLED_FOR_TEST.set(Some(false));
    let mut state = GenerateState::new(GenFamily::Krea2);
    state.set_prompt("what the user typed".into());
    let why = state.assistant_generate("a fox in the snow", &ctx, None).unwrap_err();
    assert!(why.contains("isn't set up yet"), "{why}");
    assert!(!state.busy() && state.prompt == "what the user typed");

    INSTALLED_FOR_TEST.set(Some(true));
    state.running = true;
    let why = state.assistant_generate("a fox in the snow", &ctx, None).unwrap_err();
    assert_eq!(why, "Krea 2 is busy with another run.");
    assert_eq!(state.prompt, "what the user typed");
    assert_eq!(state.family_title(), "Krea 2");
    INSTALLED_FOR_TEST.set(None);
}
