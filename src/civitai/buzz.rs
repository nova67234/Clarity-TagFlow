//! Buzz — Civitai's currency — in the Civitai resources view: what the account
//! behind the saved API key has, at the top of the view; and, under a resource's
//! download button, what its creator asks before it can be downloaded.
//!
//! Civitai's public API (`/api/v1`) says only THAT a version is paid for (its
//! `paidAccess`), not for how much, and nothing of an account's Buzz. The rest is
//! read the way Civitai's own command-line tool reads it: from the site's API,
//! with the key as a bearer token.
//!
//! | what                        | call                                           |
//! |-----------------------------|------------------------------------------------|
//! | the account's Buzz          | `buzz.getBuzzAccount`                          |
//! | a version's price           | `modelVersion.getById` `{"id": …}`             |
//! | whether it's bought already | `common.getEntityAccess` `{"entityType": "ModelVersion", "entityId": […]}` |
//!
//! Each is `GET /api/trpc/<call>?input={"json": <input>}` (no `input` at all for a
//! call that takes none), answered as `{"result": {"data": {"json": <answer>}}}`.
//! Paying isn't possible from here — Civitai refuses a purchase made with an API
//! key — so that is left to the resource's page, which a click on its card opens;
//! coming back to the app from it reads the account again.
//!
//! None of this is part of the versioned public API, so every reading is allowed
//! to fail, and none is part of the lookup that fills the view: the view reads
//! them once its resources are there, and without them is what it was before,
//! less a number.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use eframe::egui;
use egui::Color32;
use serde_json::Value;

use super::{native_tls_config, percent_encode, USER_AGENT};
use crate::theme::{is_light, MUTED};

/// Where the calls go.
const SITE: &str = "https://civitai.com";

/// How long a balance on show is taken as it is; older, it's read again the
/// next time the view is looked at.
const FRESH_FOR: Duration = Duration::from_secs(5 * 60);
/// Coming back to the app reads it again (the usual reason to have left is to
/// pay for something on Civitai) — unless it was read this recently, which is
/// less than paying takes.
const JUST_READ: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// What is read
// ---------------------------------------------------------------------------

/// The Buzz an account can spend, by the colour Civitai gives each kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Yellow,
    Green,
    Blue,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Yellow, Kind::Green, Kind::Blue];

    fn name(self) -> &'static str {
        match self {
            Kind::Yellow => "Yellow",
            Kind::Green => "Green",
            Kind::Blue => "Blue",
        }
    }

    /// Its colour in the theme that's on.
    pub(super) fn colour(self) -> Color32 {
        self.colour_on(is_light())
    }

    /// Its colour: Civitai's — a shade deeper on a `light` theme, where the
    /// bright ones wash out on white.
    fn colour_on(self, light: bool) -> Color32 {
        match (self, light) {
            (Kind::Yellow, false) => Color32::from_rgb(250, 176, 5),
            (Kind::Yellow, true) => Color32::from_rgb(204, 122, 0),
            (Kind::Green, false) => Color32::from_rgb(81, 207, 102),
            (Kind::Green, true) => Color32::from_rgb(43, 138, 62),
            (Kind::Blue, false) => Color32::from_rgb(77, 171, 247),
            (Kind::Blue, true) => Color32::from_rgb(25, 113, 194),
        }
    }
}

/// An account's Buzz.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Balance {
    pub(super) yellow: u64,
    pub(super) green: u64,
    pub(super) blue: u64,
}

impl Balance {
    fn of(&self, kind: Kind) -> u64 {
        match kind {
            Kind::Yellow => self.yellow,
            Kind::Green => self.green,
            Kind::Blue => self.blue,
        }
    }
}

/// Why a reading came back with nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Problem {
    /// Civitai didn't take the key (401).
    BadKey,
    /// The key isn't allowed this reading (403): it was made without the
    /// permission for it.
    NoPermission,
    /// Anything else: no connection, an answer that isn't the one expected.
    Unreachable,
}

/// What a resource's creator asks before it can be downloaded: Civitai's "paid
/// access" on a model version (early access is the kind that ends by itself).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Gate {
    /// The model version it's on.
    pub(super) version: u64,
    /// When it lifts by itself, as Civitai gave it; `None` for one that stays.
    pub(super) ends: Option<String>,
    pub(super) price: Price,
    /// Whether the account behind the key may download it already — it bought
    /// it, or it's its own. `None` while that isn't known.
    pub(super) unlocked: Option<bool>,
}

/// What the download costs, as far as it has been read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Price {
    /// Not read: that takes an API key.
    Unread,
    /// Being read for the first time.
    Reading,
    /// Asked for, and not told.
    Unknown,
    /// The download isn't for sale on its own (its creator sells generating
    /// with it on Civitai).
    NotSold,
    /// This much Buzz — `was` more, before a sale that's on — and whether blue
    /// Buzz pays for it too.
    Buzz { now: u64, was: Option<u64>, blue: bool },
}

impl Price {
    /// Whether asking again would say the same.
    fn settled(self) -> bool {
        matches!(self, Price::Buzz { .. } | Price::NotSold)
    }
}

impl Gate {
    /// The Buzz to pay, when that's known.
    pub(super) fn buzz(&self) -> Option<u64> {
        match self.price {
            Price::Buzz { now, .. } => Some(now),
            _ => None,
        }
    }

    /// Whether it may still stand between this account and the file: not once
    /// the account has the download.
    pub(super) fn asks(&self) -> bool {
        self.unlocked != Some(true)
    }

    /// Whether the card says so. Not in the moment its price is first read:
    /// what's read with it, often enough, is that the account has it already.
    pub(super) fn shows(&self) -> bool {
        self.asks() && self.price != Price::Reading
    }
}

// ---------------------------------------------------------------------------
// Reading Civitai's answers
// ---------------------------------------------------------------------------

/// A JSON number as a count of Buzz: whole, and never below nothing.
fn whole(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().filter(|f| f.is_finite()).map(|f| f.max(0.0).round() as u64))
}

/// The gate on a model version of the PUBLIC API's (`modelVersions[]` of
/// `/api/v1/models/{id}`, or a `/api/v1/model-versions/…` answer), if it has
/// one that's still up at `now`.
///
/// The API lists `paidAccess: {permanent, endsAt}` on a version while its gate
/// is up, and nothing once it's down; a version of `/models/{id}` also carries
/// the end as `earlyAccessDeadline`. Neither says the price: that is read with
/// the key ([`Account::look`]).
pub(super) fn gate_of(version: &Value, now: chrono::DateTime<chrono::Utc>) -> Option<Gate> {
    let id = version.get("id").and_then(Value::as_u64)?;
    let text = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_owned);
    let paid = version.get("paidAccess").filter(|p| p.is_object());
    let ends = paid.and_then(|p| text(p, "endsAt")).or_else(|| text(version, "earlyAccessDeadline"));
    if paid.is_none() && ends.is_none() {
        return None;
    }
    // (A date that can't be read is no reason to call the gate down.)
    let passed = ends.as_deref().and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok()).is_some_and(|at| at <= now);
    (!passed).then_some(Gate { version: id, ends, price: Price::Unread, unlocked: None })
}

/// The account's Buzz, from `buzz.getBuzzAccount`'s answer: `{blue, green,
/// yellow}`. An answer with none of the three isn't that answer.
fn balance_of(answer: &Value) -> Option<Balance> {
    let count = |kind: &str| answer.get(kind).and_then(whole);
    let (yellow, green, blue) = (count("yellow"), count("green"), count("blue"));
    (yellow.is_some() || green.is_some() || blue.is_some()).then(|| Balance {
        yellow: yellow.unwrap_or(0),
        green: green.unwrap_or(0),
        blue: blue.unwrap_or(0),
    })
}

/// The price of a version's download, from `modelVersion.getById`'s answer: its
/// `paidAccess.terms` — already what a buyer pays, a sale that's on included
/// (`paidAccess.sale.listTerms` is then what it was).
fn price_of(answer: &Value) -> Price {
    let Some(paid) = answer.get("paidAccess").filter(|p| p.is_object()) else {
        return Price::Unknown;
    };
    let download = |terms: Option<&Value>| terms?.get("download")?.get("price").and_then(whole);
    let terms = paid.get("terms");
    match download(terms) {
        Some(now) => Price::Buzz {
            now,
            was: download(paid.get("sale").and_then(|s| s.get("listTerms"))).filter(|was| *was > now),
            blue: terms.and_then(|t| t.get("acceptsBlueBuzz")).and_then(Value::as_bool).unwrap_or(false),
        },
        // Terms that name no download price: the version sells something else.
        None if terms.is_some_and(Value::is_object) => Price::NotSold,
        None => Price::Unknown,
    }
}

/// Which versions the account may download, from `common.getEntityAccess`'s
/// answer: an entry per version, `{entityId, hasAccess, permissions}`. On a
/// gated version, `permissions` are bits — 1 to generate with it, 2 to
/// download it — or -1 without any; a version with no gate (any more) is
/// `hasAccess` with no bits at all.
fn access_of(answer: &Value) -> HashMap<u64, bool> {
    const DOWNLOAD: i64 = 2;
    let entries = answer.as_array().map(Vec::as_slice).unwrap_or_default();
    let entry = |e: &Value| {
        let version = e.get("entityId").and_then(Value::as_u64)?;
        let has = e.get("hasAccess").and_then(Value::as_bool)?;
        let bits = e.get("permissions").and_then(Value::as_i64).unwrap_or(-1);
        Some((version, has && (bits == 0 || (bits > 0 && bits & DOWNLOAD != 0))))
    };
    entries.iter().filter_map(entry).collect()
}

/// What Civitai's refusal of a download comes to when it's the one for a
/// version behind a gate — to a client that isn't a browser, a 403 with
/// `{"error": "Early Access", "message": …}` — told so it can be acted on.
/// `gate` is what the card knew of it, and `has_key` whether the download went
/// out with an API key.
pub(super) fn refusal(body: &str, gate: Option<&Gate>, has_key: bool) -> Option<String> {
    let answer: Value = serde_json::from_str(body).ok()?;
    if !answer.get("error")?.as_str()?.eq_ignore_ascii_case("early access") {
        return None;
    }
    // Its message is the offer to pay with Buzz — or, where Civitai sells
    // nothing (a base model licensed for non-commercial use only), what is the
    // matter instead: that is passed on as it is.
    let message = answer.get("message").and_then(Value::as_str).map(str::trim).unwrap_or_default();
    if !message.is_empty() && !message.to_ascii_lowercase().contains("buzz") {
        return Some(message.to_string());
    }
    let price = match gate.map(|gate| gate.price) {
        Some(Price::NotSold) => {
            return Some(
                "its creator doesn't sell the download — its Civitai page (click the card) has what's offered".to_string(),
            );
        }
        Some(Price::Buzz { now, .. }) => format!("asks {} Buzz first", grouped(now)),
        _ => "asks for Buzz first".to_string(),
    };
    // (Paid for, it downloads for the account that paid: the one the key is of.)
    let key = if has_key { "" } else { " and add your API key in settings" };
    Some(format!("its creator {price} — pay on its Civitai page (click the card){key}, then try again"))
}

// ---------------------------------------------------------------------------
// Asking
// ---------------------------------------------------------------------------

/// The agent the calls go out on: short of patience, the answers are small.
/// A call that's sent on elsewhere isn't followed: the key wouldn't go with
/// it, and what came back would read as the key being refused.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .tls_config(native_tls_config())
        .timeout_global(Some(Duration::from_secs(15)))
        .max_redirects(0)
        .http_status_as_error(false)
        .build()
        .into()
}

/// One call of the site's API at `site`, as its own clients make it (see the
/// top of this file), and its answer.
fn query(agent: &ureq::Agent, site: &str, key: &str, call: &str, input: Option<Value>) -> Result<Value, Problem> {
    let mut url = format!("{site}/api/trpc/{call}");
    if let Some(input) = input {
        url.push_str("?input=");
        url.push_str(&percent_encode(&serde_json::json!({ "json": input }).to_string()));
    }
    let mut resp = agent
        .get(&url)
        .header("User-Agent", USER_AGENT)
        .header("Accept", "application/json")
        .header("Authorization", &format!("Bearer {key}"))
        .call()
        .map_err(|_| Problem::Unreachable)?;
    match resp.status().as_u16() {
        200 => {}
        401 => return Err(Problem::BadKey),
        403 => return Err(Problem::NoPermission),
        _ => return Err(Problem::Unreachable),
    }
    let body = resp.body_mut().read_to_string().map_err(|_| Problem::Unreachable)?;
    let mut envelope: Value = serde_json::from_str(&body).map_err(|_| Problem::Unreachable)?;
    let data = envelope.get_mut("result").and_then(|r| r.get_mut("data")).ok_or(Problem::Unreachable)?;
    Ok(match data.get_mut("json") {
        Some(answer) => answer.take(),
        None => data.take(),
    })
}

fn read_balance(agent: &ureq::Agent, site: &str, key: &str) -> Result<Balance, Problem> {
    let answer = query(agent, site, key, "buzz.getBuzzAccount", None)?;
    balance_of(&answer).ok_or(Problem::Unreachable)
}

fn read_price(agent: &ureq::Agent, site: &str, key: &str, version: u64) -> Price {
    let input = serde_json::json!({ "id": version });
    query(agent, site, key, "modelVersion.getById", Some(input)).map_or(Price::Unknown, |answer| price_of(&answer))
}

fn read_access(agent: &ureq::Agent, site: &str, key: &str, versions: &[u64]) -> HashMap<u64, bool> {
    let input = serde_json::json!({ "entityType": "ModelVersion", "entityId": versions });
    query(agent, site, key, "common.getEntityAccess", Some(input)).map_or_else(|_| HashMap::new(), |answer| access_of(&answer))
}

/// Fill in what the public API leaves out of `gates`: what each costs (where
/// that isn't settled already), and whether the account behind `key` has it.
/// A call per price and one for them all; `cancelled` is asked between them.
fn fill(agent: &ureq::Agent, site: &str, key: &str, gates: &mut [Gate], cancelled: impl Fn() -> bool) {
    if key.is_empty() || gates.is_empty() {
        return;
    }
    for gate in gates.iter_mut() {
        if cancelled() {
            return;
        }
        if !gate.price.settled() {
            gate.price = read_price(agent, site, key, gate.version);
        }
    }
    if cancelled() {
        return;
    }
    let versions: Vec<u64> = gates.iter().map(|g| g.version).collect();
    let access = read_access(agent, site, key, &versions);
    for gate in gates.iter_mut() {
        gate.unlocked = access.get(&gate.version).copied();
    }
}

// ---------------------------------------------------------------------------
// The account, in the view
// ---------------------------------------------------------------------------

/// The Buzz of the account behind the API key, as the view holds it.
pub(super) struct Account {
    /// Where it's read from: Civitai, outside the tests.
    site: String,
    /// What's on show: the last balance read.
    balance: Option<Balance>,
    /// What the last reading ran into, if it did.
    problem: Option<Problem>,
    reading: Option<Receiver<Result<Balance, Problem>>>,
    /// A look at gates on show, under way (see [`Account::look`]) — and what
    /// it goes on for as long as this holds.
    looking: Option<(Receiver<Vec<Gate>>, Arc<()>)>,
    /// The key the balance was last asked for with, and when.
    key: String,
    asked: Option<Instant>,
    /// Whether the app's window had the focus last frame.
    focused: bool,
}

impl Default for Account {
    fn default() -> Self {
        Self::at(SITE)
    }
}

/// What a click on the counter's row asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ask {
    /// The settings, to put an API key in.
    Key,
    /// The balance read again.
    Again,
}

/// What [`Account::tick`] finds the view is to look at again, of the resources
/// on show that ask for Buzz.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Look {
    /// Those that still ask for it: the app is back in front — from paying for
    /// one on Civitai, perhaps.
    Unpaid,
    /// All of them, as if never looked at: the key is another, and so may the
    /// account be.
    All,
}

/// Whether the balance is to be read now: with a key it hasn't been read with,
/// when the last reading — `since` ago — has aged, and on coming `back` to the
/// app unless it was only just read.
fn due(new_key: bool, since: Option<Duration>, back: bool) -> bool {
    new_key || since.is_none_or(|t| t > FRESH_FOR) || back && since.is_some_and(|t| t > JUST_READ)
}

impl Account {
    /// The account as `site` has it.
    fn at(site: &str) -> Self {
        Self {
            site: site.to_string(),
            balance: None,
            problem: None,
            reading: None,
            looking: None,
            key: String::new(),
            asked: None,
            focused: false,
        }
    }

    /// Once a frame while the view shows: take in what the reading brought,
    /// and start one when it's due — with a new key (once it's no longer being
    /// `typing`-ed in the settings), when the balance on show has aged, and
    /// when the app comes back to the front. What the view is to look at again
    /// because of it, if anything.
    pub(super) fn tick(&mut self, ctx: &egui::Context, key: &str, typing: bool) -> Option<Look> {
        if let Some(rx) = &self.reading {
            match rx.try_recv() {
                Ok(Ok(balance)) => (self.balance, self.problem, self.reading) = (Some(balance), None, None),
                Ok(Err(problem)) => {
                    // A key Civitai won't take leaves nothing on show; a
                    // reading that didn't get through leaves the last one.
                    if problem != Problem::Unreachable {
                        self.balance = None;
                    }
                    (self.problem, self.reading) = (Some(problem), None);
                }
                Err(TryRecvError::Empty) => {}
                // The reading ended without a word.
                Err(TryRecvError::Disconnected) => (self.problem, self.reading) = (Some(Problem::Unreachable), None),
            }
        }

        let focused = ctx.input(|i| i.focused);
        let back = !std::mem::replace(&mut self.focused, focused) && focused;
        let key = key.trim();
        if key.is_empty() {
            // No key (any more): nothing of the last one's stays.
            let had = !self.key.is_empty();
            (self.balance, self.problem, self.reading, self.asked) = (None, None, None, None);
            self.key.clear();
            return had.then_some(Look::All);
        }
        if typing {
            return None;
        }
        let new_key = self.key != key;
        if new_key {
            // Another account's, for all that's known — and so is the answer
            // of a reading under way.
            (self.balance, self.problem, self.reading) = (None, None, None);
        }
        if self.reading.is_none() && due(new_key, self.asked.map(|at| at.elapsed()), back) {
            self.read(ctx, key);
        }
        if new_key { Some(Look::All) } else { back.then_some(Look::Unpaid) }
    }

    /// Read the balance now.
    pub(super) fn read(&mut self, ctx: &egui::Context, key: &str) {
        let key = key.trim().to_string();
        if key.is_empty() {
            return;
        }
        self.key.clone_from(&key);
        self.asked = Some(Instant::now());
        let (tx, rx) = mpsc::channel();
        self.reading = Some(rx);
        let (site, ctx) = (self.site.clone(), ctx.clone());
        std::thread::spawn(move || {
            let _ = tx.send(read_balance(&agent(), &site, &key));
            ctx.request_repaint();
        });
    }

    /// Look at `gates` — copies of those on show that ask for Buzz: what each
    /// costs, where that isn't settled, and whether the account has it (by
    /// now). [`Account::looked`] has them back, filled in. A look under way
    /// is dropped, and stops: it was for another moment, or another key.
    pub(super) fn look(&mut self, ctx: &egui::Context, key: &str, mut gates: Vec<Gate>) {
        self.looking = None;
        let key = key.trim().to_string();
        if key.is_empty() || gates.is_empty() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let wanted = Arc::new(());
        let still = Arc::downgrade(&wanted);
        self.looking = Some((rx, wanted));
        let (site, ctx) = (self.site.clone(), ctx.clone());
        std::thread::spawn(move || {
            fill(&agent(), &site, &key, &mut gates, || still.strong_count() == 0);
            let _ = tx.send(gates);
            ctx.request_repaint();
        });
    }

    /// Whether a look is under way.
    pub(super) fn is_looking(&self) -> bool {
        self.looking.is_some()
    }

    /// What the last look came back with, once.
    pub(super) fn looked(&mut self) -> Option<Vec<Gate>> {
        let (rx, _) = self.looking.as_ref()?;
        match rx.try_recv() {
            Ok(gates) => {
                self.looking = None;
                Some(gates)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.looking = None;
                None
            }
        }
    }

    /// The counter: a row under the view's header. A bolt and an amount for
    /// each kind of Buzz the account holds, in the kind's colour — or, in grey,
    /// what stands in the way of showing that. What a click on it asks for.
    pub(super) fn row(&self, ui: &mut egui::Ui, has_key: bool) -> Option<Ask> {
        // What the row says when it can't say a balance, what its hover adds,
        // and what a click is for.
        let (instead, tip, click) = match (self.balance, self.problem) {
            _ if !has_key => (
                Some("Add your API key to see your Buzz"),
                "Shows the Buzz on your Civitai account, and what a paid model costs.\nClick to add your API key."
                    .to_string(),
                Ask::Key,
            ),
            (Some(balance), problem) => {
                let mut tip: String =
                    Kind::ALL.map(|kind| format!("{} Buzz: {}\n", kind.name(), grouped(balance.of(kind)))).concat();
                tip.push_str(if problem.is_some() && self.reading.is_none() {
                    "Couldn't update just now. Click to try again."
                } else {
                    "Click to update."
                });
                (None, tip, Ask::Again)
            }
            (None, Some(Problem::BadKey)) => (
                Some("API key not accepted"),
                "Civitai didn't accept the saved API key.\nClick to check it.".to_string(),
                Ask::Key,
            ),
            (None, Some(Problem::NoPermission)) => (
                Some("This API key can't read Buzz"),
                "The saved key isn't allowed to read your Buzz. On Civitai, make one with the Full Access preset \
                 (or with Buzz read ticked), then click here to put it in."
                    .to_string(),
                Ask::Key,
            ),
            (None, Some(Problem::Unreachable)) => (
                Some("Buzz unavailable"),
                "Couldn't read your Buzz from Civitai.\nClick to try again.".to_string(),
                Ask::Again,
            ),
            (None, None) => (Some("…"), "Reading your Buzz from Civitai…".to_string(), Ask::Again),
        };

        let row = ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.set_min_height(ROW_H);
            match (instead, self.balance) {
                (None, Some(balance)) => {
                    // The kinds the account holds; with none, a yellow nought.
                    let empty = Kind::ALL.iter().all(|kind| balance.of(*kind) == 0);
                    let mut first = true;
                    for kind in Kind::ALL {
                        let held = balance.of(kind);
                        if held == 0 && !(empty && kind == Kind::Yellow) {
                            continue;
                        }
                        if !std::mem::take(&mut first) {
                            ui.add_space(9.0);
                        }
                        bolt(ui, ROW_BOLT, kind.colour());
                        let text = egui::RichText::new(amount(held)).color(kind.colour()).size(12.5);
                        ui.add(egui::Label::new(text).selectable(false));
                    }
                }
                (words, _) => {
                    bolt(ui, ROW_BOLT, MUTED());
                    let text = egui::RichText::new(words.unwrap_or_default()).color(MUTED()).size(11.5);
                    ui.add(egui::Label::new(text).selectable(false).truncate());
                }
            }
        });
        let resp = row.response.interact(egui::Sense::click()).on_hover_text(tip);
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        resp.clicked().then_some(click)
    }
}

// ---------------------------------------------------------------------------
// Showing it
// ---------------------------------------------------------------------------

/// The counter's row: its height, and its bolts'.
const ROW_H: f32 = 18.0;
const ROW_BOLT: f32 = 13.0;
/// The price under a download button: its line's height, and its bolt's.
pub(super) const COST_H: f32 = 14.0;
const COST_BOLT: f32 = 11.0;
const COST_FONT: f32 = 11.5;

/// A count of Buzz with its thousands apart: "12,480".
fn grouped(count: u64) -> String {
    let digits = count.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// A count of Buzz as it's shown: whole while that's short ("12,480"), then
/// rounded ("125k", "1.2M").
pub(super) fn amount(count: u64) -> String {
    match count {
        0..=99_999 => grouped(count),
        100_000..=999_499 => format!("{}k", (count + 500) / 1000),
        _ => {
            let millions = count as f64 / 1e6;
            if millions < 99.95 { format!("{millions:.1}M") } else { format!("{}M", grouped(millions.round() as u64)) }
        }
    }
}

/// The Buzz icon — a bolt — `size` points across, in `colour`.
fn bolt(ui: &mut egui::Ui, size: f32, colour: Color32) {
    ui.add(bolt_image().fit_to_exact_size(egui::Vec2::splat(size)).tint(colour));
}

fn bolt_image() -> egui::Image<'static> {
    egui::Image::new(egui::include_image!("../../icons/buzz.svg"))
}

/// What the line under the download button says of `gate`: the Buzz it asks,
/// or that it asks some.
fn cost_text(gate: &Gate) -> String {
    gate.buzz().map_or_else(|| "Paid".to_string(), amount)
}

/// The price of `gate`, for under a download button: the bolt, then the
/// amount, in Buzz's yellow. Where it was drawn.
pub(super) fn cost(ui: &mut egui::Ui, gate: &Gate) -> egui::Rect {
    let colour = Kind::Yellow.colour();
    let font = egui::FontId::proportional(COST_FONT);
    let text = ui.fonts_mut(|f| f.layout_no_wrap(cost_text(gate), font, colour));
    let size = egui::vec2(COST_BOLT + 2.0 + text.size().x, COST_H);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        let icon = egui::Rect::from_min_size(
            egui::pos2(rect.left(), rect.center().y - COST_BOLT / 2.0),
            egui::Vec2::splat(COST_BOLT),
        );
        bolt_image().tint(colour).paint_at(ui, icon);
        let at = egui::pos2(icon.right() + 2.0, rect.center().y - text.size().y / 2.0);
        ui.painter().galley(at, text, colour);
    }
    rect
}

/// The day a gate lifts — "20 Nov 2026" — from the time Civitai gave, as the
/// calendar of `zone` has it.
fn day<Tz: chrono::TimeZone>(ends: &str, zone: &Tz) -> Option<String>
where
    Tz::Offset: std::fmt::Display,
{
    let at = chrono::DateTime::parse_from_rfc3339(ends).ok()?;
    Some(at.with_timezone(zone).format("%-d %b %Y").to_string())
}

/// What a hover over the price adds to it.
pub(super) fn cost_tip(gate: &Gate) -> String {
    let mut tip = match gate.price {
        Price::Buzz { now, was, blue } => {
            let mut line = format!("{} Buzz to download", grouped(now));
            if let Some(was) = was {
                line.push_str(&format!(" — on sale, was {}", grouped(was)));
            }
            if blue {
                line.push_str(". Blue Buzz pays for it too");
            }
            line + ".\nPay on its Civitai page (click the card), then download it here."
        }
        Price::NotSold => "Its creator charges Buzz to use this on Civitai, and doesn't sell the download on its own.\n\
                           Its Civitai page (click the card) has what's offered."
            .to_string(),
        Price::Unread | Price::Reading => "Its creator asks for Buzz before it can be downloaded.\n\
                                           Add your Civitai API key in this view's settings to see how much."
            .to_string(),
        Price::Unknown => "Its creator asks for Buzz before it can be downloaded.\n\
                           The price couldn't be read — its Civitai page (click the card) has it."
            .to_string(),
    };
    if let Some(day) = gate.ends.as_deref().and_then(|ends| day(ends, &chrono::Local)) {
        tip.push_str(&format!("\nFree to download from {day}."));
    }
    tip
}

#[cfg(test)]
mod tests;
