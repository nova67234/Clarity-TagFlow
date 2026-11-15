//! Buzz in the Civitai resources view, tested: Civitai's answers as they're
//! read, the calls as they go out (to a stand-in for its site on 127.0.0.1),
//! the counter and the price as they're painted, and the view with both in it.
//! Nothing here reaches Civitai.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use egui::{Event, Pos2, Rect, pos2, vec2};
use serde_json::json;

use super::super::{
    CivitaiState, Download, DownloadDirs, DownloadRequest, UiResource, UiResult, UiSection, resource_card, show,
};
use super::*;

// ---------------------------------------------------------------------------
// Civitai's answers, as they're read
// ---------------------------------------------------------------------------

/// `modelVersion.getById`'s answer for a version with this `paidAccess`.
fn version_answer(paid: Value) -> Value {
    json!({ "id": 123, "name": "v1.0", "baseModel": "Illustrious", "paidAccess": paid })
}

/// The same, for a gate that stays, with these terms and no sale on.
fn with_terms(terms: Value) -> Value {
    version_answer(json!({ "endsAt": null, "timeframeDays": null, "terms": terms, "sale": null }))
}

/// A gate as the public API gives it: its price not read.
fn gate(version: u64) -> Gate {
    Gate { version, ends: None, price: Price::Unread, unlocked: None }
}

fn priced(version: u64, price: Price, unlocked: Option<bool>) -> Gate {
    Gate { version, ends: None, price, unlocked }
}

fn buzz(now: u64) -> Price {
    Price::Buzz { now, was: None, blue: false }
}

#[test]
fn the_balance_is_the_three_kinds_of_the_accounts_answer() {
    assert_eq!(
        balance_of(&json!({ "blue": 350, "green": 1200, "yellow": 12480 })),
        Some(Balance { yellow: 12480, green: 1200, blue: 350 })
    );
    // A kind the answer leaves out is none of that kind…
    assert_eq!(balance_of(&json!({ "yellow": 5 })), Some(Balance { yellow: 5, green: 0, blue: 0 }));
    // …and an answer with none of the three isn't the account's.
    assert_eq!(balance_of(&json!({})), None);
    assert_eq!(balance_of(&json!({ "id": 7, "username": "someone" })), None);
    assert_eq!(balance_of(&Value::Null), None);
    // Counts are whole and never below nothing, however they're written.
    assert_eq!(
        balance_of(&json!({ "yellow": 99.6, "green": -3, "blue": "12" })),
        Some(Balance { yellow: 100, green: 0, blue: 0 })
    );
}

#[test]
fn the_price_is_what_a_buyer_pays_for_the_download() {
    let both = json!({ "download": { "price": 500 }, "generation": { "price": 100, "trialLimit": 10 } });
    assert_eq!(price_of(&with_terms(both)), buzz(500));
    let blue = json!({ "download": { "price": 500 }, "generation": { "free": true }, "acceptsBlueBuzz": true });
    assert_eq!(price_of(&with_terms(blue)), Price::Buzz { now: 500, was: None, blue: true });
    // Generating with it is all that's sold: the download isn't.
    let generation = json!({ "generation": { "price": 100, "trialLimit": 10 } });
    assert_eq!(price_of(&with_terms(generation)), Price::NotSold);
}

#[test]
fn a_sale_shows_what_the_price_was() {
    let sale = |list: u64| {
        version_answer(json!({
            "endsAt": null,
            "timeframeDays": null,
            // On sale, the terms are the buyer's price already.
            "terms": { "download": { "price": 1200 }, "acceptsBlueBuzz": true },
            "sale": {
                "listTerms": { "download": { "price": list } },
                "buyerTerms": { "download": { "price": 1200 } },
                "endsAt": "2026-12-01T00:00:00.000Z",
                "discountType": "percent",
                "discountAmount": 40,
            },
        }))
    };
    assert_eq!(price_of(&sale(2000)), Price::Buzz { now: 1200, was: Some(2000), blue: true });
    // A sale that takes nothing off isn't shown as one.
    assert_eq!(price_of(&sale(1200)), Price::Buzz { now: 1200, was: None, blue: true });
    assert_eq!(price_of(&sale(900)), Price::Buzz { now: 1200, was: None, blue: true });
}

#[test]
fn a_price_the_answer_does_not_have_is_not_made_up() {
    for answer in [
        json!({ "id": 123, "name": "v1.0" }),
        version_answer(Value::Null),
        version_answer(json!({ "endsAt": null })),
        version_answer(json!({ "endsAt": null, "terms": null })),
        version_answer(json!("yes")),
        Value::Null,
    ] {
        assert_eq!(price_of(&answer), Price::Unknown, "{answer}");
    }
}

#[test]
fn a_version_is_the_accounts_once_its_download_is() {
    let entry = |version: u64, has: bool, bits: i64| json!({ "entityId": version, "entityType": "ModelVersion", "hasAccess": has, "availability": "Public", "permissions": bits });
    let answer = json!([
        entry(1, false, -1), // nothing bought
        entry(2, true, 2),   // the download bought
        entry(3, true, 3),   // everything: bought, or its own
        entry(4, true, 1),   // only generating with it bought
        entry(5, true, 0),   // no gate on it (any more)
        entry(6, false, 2),  // a grant that isn't one
        { "hasAccess": true, "permissions": 3 },
        { "entityId": 8, "permissions": 3 },
        "something else",
    ]);
    let expected = HashMap::from([(1, false), (2, true), (3, true), (4, false), (5, true), (6, false)]);
    assert_eq!(access_of(&answer), expected);
    assert_eq!(access_of(&json!({ "error": "no" })), HashMap::new());
}

fn at(time: &str) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339(time).unwrap().with_timezone(&chrono::Utc)
}

#[test]
fn a_gate_is_found_on_a_version_while_it_is_up() {
    let now = at("2026-11-15T12:00:00Z");
    let found = |version: Value| gate_of(&version, now);
    let up = |version: u64, ends: Option<&str>| Some(Gate { ends: ends.map(str::to_owned), ..gate(version) });

    // One that stays.
    assert_eq!(found(json!({ "id": 7, "paidAccess": { "permanent": true, "endsAt": null } })), up(7, None));
    // One that lifts by itself (the models answer repeats its end as the
    // version's `earlyAccessDeadline`)…
    let timed = |ends: &str| json!({ "id": 8, "earlyAccessDeadline": ends, "paidAccess": { "permanent": false, "endsAt": ends } });
    assert_eq!(found(timed("2026-11-20T03:00:00.000Z")), up(8, Some("2026-11-20T03:00:00.000Z")));
    assert_eq!(found(timed("2026-11-15T12:00:01+00:00")), up(8, Some("2026-11-15T12:00:01+00:00")));
    // …and has, by now.
    assert_eq!(found(timed("2026-11-15T12:00:00.000Z")), None);
    assert_eq!(found(timed("2026-11-01T00:00:00.000Z")), None);

    // No gate at all.
    assert_eq!(found(json!({ "id": 9, "name": "v2.0", "files": [] })), None);
    assert_eq!(found(json!({ "id": 9, "paidAccess": null })), None);
    // The deadline, without the rest.
    assert_eq!(found(json!({ "id": 10, "earlyAccessDeadline": "2026-12-01T00:00:00.000Z" })), up(10, Some("2026-12-01T00:00:00.000Z")));
    assert_eq!(found(json!({ "id": 10, "earlyAccessDeadline": "2026-11-01T00:00:00.000Z" })), None);
    // An end that can't be read doesn't take the gate down.
    assert_eq!(found(json!({ "id": 11, "paidAccess": { "permanent": false, "endsAt": "soon" } })), up(11, Some("soon")));
    // A version without its number: nothing to ask the price of.
    assert_eq!(found(json!({ "paidAccess": { "permanent": true, "endsAt": null } })), None);

    // As found, it asks — the price isn't read, nor whether it's bought.
    let gate = found(json!({ "id": 7, "paidAccess": { "permanent": true, "endsAt": null } })).unwrap();
    assert_eq!((gate.price, gate.unlocked, gate.buzz(), gate.asks()), (Price::Unread, None, None, true));
}

#[test]
fn a_gate_asks_until_the_account_has_the_download() {
    for (unlocked, asks) in [(None, true), (Some(false), true), (Some(true), false)] {
        let gate = priced(1, buzz(500), unlocked);
        assert_eq!((gate.asks(), gate.shows()), (asks, asks), "{unlocked:?}");
    }
    assert_eq!(priced(1, buzz(500), None).buzz(), Some(500));
    for price in [Price::Unread, Price::Reading, Price::Unknown, Price::NotSold] {
        assert_eq!(priced(1, price, None).buzz(), None);
    }
    // While its price is first read it asks all the same — and waits to say so.
    let reading = priced(1, Price::Reading, None);
    assert_eq!((reading.asks(), reading.shows()), (true, false));
    assert!(!priced(1, Price::Reading, Some(true)).shows());
    for price in [Price::Unread, Price::Unknown, Price::NotSold] {
        assert!(priced(1, price, Some(false)).shows(), "{price:?}");
    }
}

/// Civitai's refusal of a gated download, to a client that isn't a browser.
const EARLY_ACCESS: &str = r#"{"error":"Early Access","deadline":"2026-12-01T00:00:00.000Z","message":"This asset is in Early Access. You can use Buzz access it now!"}"#;

#[test]
fn a_refused_download_says_what_its_creator_asks() {
    // (To the last Buzz: it's what's to be paid.)
    let known = priced(1, buzz(125_250), Some(false));
    assert_eq!(
        refusal(EARLY_ACCESS, Some(&known), true).as_deref(),
        Some("its creator asks 125,250 Buzz first — pay on its Civitai page (click the card), then try again")
    );
    // Without the price — or without the card having known of a gate at all.
    let asks = "its creator asks for Buzz first — pay on its Civitai page (click the card), then try again";
    for price in [Price::Unread, Price::Reading, Price::Unknown] {
        assert_eq!(refusal(EARLY_ACCESS, Some(&priced(1, price, None)), true).as_deref(), Some(asks), "{price:?}");
    }
    assert_eq!(refusal(EARLY_ACCESS, None, true).as_deref(), Some(asks));
    // Paid for, it downloads for the account that paid — which takes the key.
    assert_eq!(
        refusal(EARLY_ACCESS, Some(&known), false).as_deref(),
        Some("its creator asks 125,250 Buzz first — pay on its Civitai page (click the card) and add your API key in settings, then try again")
    );
    // The download isn't what's sold.
    assert_eq!(
        refusal(EARLY_ACCESS, Some(&priced(1, Price::NotSold, None)), true).as_deref(),
        Some("its creator doesn't sell the download — its Civitai page (click the card) has what's offered")
    );
    // However the error is cased, and without a message.
    assert_eq!(refusal(r#"{"error":"early access"}"#, None, true).as_deref(), Some(asks));
}

#[test]
fn what_civitai_does_not_sell_is_not_offered() {
    // On a base model licensed for non-commercial use only, the same refusal
    // carries another message — and paying isn't possible.
    let message = "This model isn't commercially available at the moment. Its base model, Flux.2 Klein 9B, is licensed for non-commercial use only.";
    let body = json!({ "error": "Early Access", "deadline": "2026-12-01T00:00:00.000Z", "message": message }).to_string();
    assert_eq!(refusal(&body, Some(&priced(1, buzz(500), Some(false))), true).as_deref(), Some(message));
    assert_eq!(refusal(&body, None, false).as_deref(), Some(message));
}

#[test]
fn other_refusals_are_left_to_say_what_they_say() {
    for body in [
        r#"{"error":"You do not have access to download this file"}"#,
        r#"{"error":"This file did not pass a security scan and is not available for download"}"#,
        r#"{"error":"Forbidden"}"#,
        r#"{"error":null,"message":"This asset is in Early Access. You can use Buzz access it now!"}"#,
        r#"{"message":"This asset is in Early Access. You can use Buzz access it now!"}"#,
        r#"["Early Access"]"#,
        "Early Access",
        "<html>Forbidden</html>",
        "",
    ] {
        assert_eq!(refusal(body, Some(&priced(1, buzz(500), Some(false))), true), None, "{body}");
    }
}

// ---------------------------------------------------------------------------
// Amounts, days, colours and the price's hover text
// ---------------------------------------------------------------------------

#[test]
fn amounts_are_whole_while_that_is_short() {
    for (count, shown) in [
        (0, "0"),
        (7, "7"),
        (999, "999"),
        (1_000, "1,000"),
        (12_480, "12,480"),
        (99_999, "99,999"),
        (100_000, "100k"),
        (125_499, "125k"),
        (125_500, "126k"),
        (999_499, "999k"),
        (999_500, "1.0M"),
        (1_340_000, "1.3M"),
        (99_940_000, "99.9M"),
        (99_950_000, "100M"),
        (1_234_567_890, "1,235M"),
    ] {
        assert_eq!(amount(count), shown, "{count}");
    }
    assert_eq!(grouped(100_000), "100,000");
    assert_eq!(grouped(1_234_567_890), "1,234,567,890");
    assert_eq!(grouped(u64::MAX), "18,446,744,073,709,551,615");
}

#[test]
fn a_gates_end_is_a_day_on_the_readers_calendar() {
    let ends = "2026-11-20T03:00:00.000Z";
    assert_eq!(day(ends, &chrono::Utc).as_deref(), Some("20 Nov 2026"));
    let west = chrono::FixedOffset::west_opt(5 * 3600).unwrap();
    assert_eq!(day(ends, &west).as_deref(), Some("19 Nov 2026"));
    assert_eq!(day("2026-12-05T00:00:00Z", &chrono::Utc).as_deref(), Some("5 Dec 2026"));
    assert_eq!(day("soon", &chrono::Utc), None);
}

#[test]
fn each_kind_of_buzz_has_its_colour_on_dark_and_on_light() {
    let brightness = |c: Color32| 299 * c.r() as u32 + 587 * c.g() as u32 + 114 * c.b() as u32;
    let mut colours = Vec::new();
    for kind in Kind::ALL {
        let (dark, light) = (kind.colour_on(false), kind.colour_on(true));
        assert!(brightness(light) < brightness(dark), "{kind:?} is deeper on a light theme");
        colours.extend([dark, light]);
    }
    colours.sort_by_key(|c| c.to_array());
    colours.dedup();
    assert_eq!(colours.len(), 6, "no two alike");
    // Yellow is yellow, and so on: the strongest channels say which.
    let [y, g, b] = Kind::ALL.map(|kind| kind.colour_on(false));
    assert!(y.r() > y.b() && y.g() > y.b());
    assert!(g.g() > g.r() && g.g() > g.b());
    assert!(b.b() > b.r() && b.b() > b.g());
}

#[test]
fn the_price_says_more_on_hover() {
    let pay = "\nPay on its Civitai page (click the card), then download it here.";
    assert_eq!(cost_tip(&priced(1, buzz(125_000), None)), format!("125,000 Buzz to download.{pay}"));
    assert_eq!(
        cost_tip(&priced(1, Price::Buzz { now: 1200, was: Some(2000), blue: false }, None)),
        format!("1,200 Buzz to download — on sale, was 2,000.{pay}")
    );
    assert_eq!(
        cost_tip(&priced(1, Price::Buzz { now: 1200, was: Some(2000), blue: true }, None)),
        format!("1,200 Buzz to download — on sale, was 2,000. Blue Buzz pays for it too.{pay}")
    );
    assert_eq!(
        cost_tip(&priced(1, Price::Buzz { now: 300, was: None, blue: true }, None)),
        format!("300 Buzz to download. Blue Buzz pays for it too.{pay}")
    );
    // Without the amount, what's known instead — each its own.
    let tips = [Price::Unread, Price::Unknown, Price::NotSold].map(|price| cost_tip(&priced(1, price, None)));
    assert!(tips[0].contains("Add your Civitai API key"), "{}", tips[0]);
    assert!(tips[1].contains("couldn't be read"), "{}", tips[1]);
    assert!(tips[2].contains("doesn't sell the download"), "{}", tips[2]);
    for tip in &tips {
        assert!(!tip.contains("to download.") && !tip.contains("Free to download"), "{tip}");
    }

    // A gate that lifts by itself says when (noon in UTC: the same month on
    // every calendar).
    let timed = Gate { ends: Some("2026-11-20T12:00:00.000Z".into()), ..priced(1, buzz(500), None) };
    let tip = cost_tip(&timed);
    let (rest, day) = tip.rsplit_once("\nFree to download from ").expect("the day it lifts");
    assert_eq!(rest, format!("500 Buzz to download.{pay}"));
    assert!(day.ends_with(" Nov 2026.") && day.starts_with('2'), "{day}");
    // One whose end can't be read just doesn't.
    let odd = Gate { ends: Some("soon".into()), ..priced(1, buzz(500), None) };
    assert_eq!(cost_tip(&odd), format!("500 Buzz to download.{pay}"));
}

#[test]
fn the_balance_is_read_when_it_is_due() {
    let secs = Duration::from_secs;
    // Never read; and a key it wasn't read with, however lately.
    assert!(due(false, None, false));
    assert!(due(true, Some(secs(1)), false));
    // Read: not again until that has aged.
    assert!(!due(false, Some(secs(1)), false));
    assert!(!due(false, Some(secs(4 * 60)), false));
    assert!(due(false, Some(secs(6 * 60)), false));
    // Back in front of the other windows: again, unless it was only just read.
    assert!(due(false, Some(secs(30)), true));
    assert!(!due(false, Some(secs(2)), true));
    assert!(JUST_READ < FRESH_FOR);
}

// ---------------------------------------------------------------------------
// A stand-in for Civitai's site, on 127.0.0.1
// ---------------------------------------------------------------------------

/// One call the stand-in got.
#[derive(Clone, Debug)]
struct Asked {
    /// The request's target, as it came.
    target: String,
    /// The call it names, and its input, decoded (`Null` without one).
    call: String,
    input: Value,
    /// The bearer token it came with.
    key: String,
    agent: String,
    accept: String,
}

/// Answers every call with what `answer` makes of it — a status and a body
/// (of a redirect, where to) — and keeps the calls.
struct Site {
    /// Where it is: "http://127.0.0.1:<port>".
    at: String,
    asked: Arc<Mutex<Vec<Asked>>>,
}

impl Site {
    fn answering(answer: impl Fn(&Asked) -> (u16, String) + Send + Sync + 'static) -> Site {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let at = format!("http://{}", listener.local_addr().unwrap());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let (log, answer) = (asked.clone(), Arc::new(answer));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let (log, answer) = (log.clone(), answer.clone());
                std::thread::spawn(move || {
                    let Some(call) = read_call(&mut stream) else { return };
                    log.lock().unwrap().push(call.clone());
                    let (status, mut body) = answer(&call);
                    let to = if (300..400).contains(&status) { format!("Location: {}\r\n", std::mem::take(&mut body)) } else { String::new() };
                    let head = format!(
                        "HTTP/1.1 {status} Stand-in\r\n{to}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes()).and_then(|()| stream.write_all(body.as_bytes()));
                    let _ = stream.flush();
                });
            }
        });
        Site { at, asked }
    }

    fn asked(&self) -> Vec<Asked> {
        self.asked.lock().unwrap().clone()
    }

    /// The names of the calls it got, sorted.
    fn calls(&self) -> Vec<String> {
        let mut calls: Vec<String> = self.asked().into_iter().map(|a| a.call).collect();
        calls.sort();
        calls
    }
}

fn read_call(stream: &mut std::net::TcpStream) -> Option<Asked> {
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let mut head = Vec::new();
    let mut some = [0u8; 1024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut some).ok().filter(|n| *n > 0)?;
        head.extend_from_slice(&some[..n]);
    }
    let head = String::from_utf8_lossy(&head).into_owned();
    let target = head.split_whitespace().nth(1)?.to_string();
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let header = |name: &str| {
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim().to_string())
        })
    };
    Some(Asked {
        call: path.strip_prefix("/api/trpc/").unwrap_or(path).to_string(),
        input: query
            .strip_prefix("input=")
            .and_then(|input| serde_json::from_str(&percent_decode(input)).ok())
            .unwrap_or(Value::Null),
        key: header("authorization").and_then(|v| v.strip_prefix("Bearer ").map(str::to_owned)).unwrap_or_default(),
        agent: header("user-agent").unwrap_or_default(),
        accept: header("accept").unwrap_or_default(),
        target,
    })
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let (mut out, mut i) = (Vec::with_capacity(bytes.len()), 0);
    while i < bytes.len() {
        let hex = bytes.get(i + 1..i + 3).and_then(|hex| std::str::from_utf8(hex).ok());
        match (bytes[i], hex.and_then(|hex| u8::from_str_radix(hex, 16).ok())) {
            (b'%', Some(byte)) => {
                out.push(byte);
                i += 3;
            }
            (byte, _) => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A call answered: 200, with `answer` in the site's envelope.
fn ok(answer: Value) -> (u16, String) {
    (200, json!({ "result": { "data": { "json": answer } } }).to_string())
}

fn failing(status: u16) -> (u16, String) {
    (status, json!({ "error": { "json": { "message": "no", "code": -32001, "data": { "httpStatus": status } } } }).to_string())
}

/// `modelVersion.getById`'s answer for a version whose download costs `price`.
fn costing(price: u64) -> Value {
    with_terms(json!({ "download": { "price": price } }))
}

/// `common.getEntityAccess`'s answer: each of `versions`, and whether its
/// download is the account's.
fn access(versions: &[(u64, bool)]) -> Value {
    let entry = |(version, has): &(u64, bool)| json!({ "entityId": version, "entityType": "ModelVersion", "hasAccess": has, "availability": "Public", "permissions": if *has { 2 } else { -1 } });
    Value::Array(versions.iter().map(entry).collect())
}

/// The version a `modelVersion.getById` call asks for.
fn version_asked(call: &Asked) -> u64 {
    call.input["json"]["id"].as_u64().expect("a version's number")
}

/// Somewhere on this computer nothing listens.
fn nowhere() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    format!("http://{}", listener.local_addr().unwrap())
}

/// Keeps a stand-in's answer back while it's set — but not for ever.
fn hold_while(held: &AtomicBool) {
    let since = Instant::now();
    while held.load(Ordering::SeqCst) && since.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// `get()`, once it has something — which it must, soon.
fn soon<T>(what: &str, mut get: impl FnMut() -> Option<T>) -> T {
    let since = Instant::now();
    loop {
        if let Some(got) = get() {
            return got;
        }
        assert!(since.elapsed() < Duration::from_secs(10), "waited for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn the_calls_go_out_as_civitais_own_clients_make_them() {
    let site = Site::answering(|call| match call.call.as_str() {
        "buzz.getBuzzAccount" => ok(json!({ "blue": 350, "green": 1200, "yellow": 12480 })),
        "modelVersion.getById" => ok(costing(500)),
        "common.getEntityAccess" => ok(access(&[(123, false), (456, true)])),
        _ => failing(404),
    });
    let agent = agent();
    assert_eq!(read_balance(&agent, &site.at, "the-key"), Ok(Balance { yellow: 12480, green: 1200, blue: 350 }));
    assert_eq!(read_price(&agent, &site.at, "the-key", 123), buzz(500));
    assert_eq!(read_access(&agent, &site.at, "the-key", &[123, 456]), HashMap::from([(123, false), (456, true)]));

    let asked = site.asked();
    assert_eq!(asked.len(), 3);
    // The call's name is the path. One that takes no input has no query…
    assert_eq!(asked[0].target, "/api/trpc/buzz.getBuzzAccount");
    // …and the others have theirs as `{"json": …}`, percent-encoded.
    assert_eq!(asked[1].target, "/api/trpc/modelVersion.getById?input=%7B%22json%22%3A%7B%22id%22%3A123%7D%7D");
    assert_eq!((asked[1].call.as_str(), &asked[1].input), ("modelVersion.getById", &json!({ "json": { "id": 123 } })));
    assert_eq!(asked[2].call, "common.getEntityAccess");
    assert_eq!(asked[2].input, json!({ "json": { "entityType": "ModelVersion", "entityId": [123, 456] } }));
    let query = asked[2].target.split_once("?input=").expect("an input").1;
    assert!(query.bytes().all(|b| b.is_ascii_alphanumeric() || b"%-_.~".contains(&b)), "{query}");
    // The key is a bearer token, never part of the address.
    for call in &asked {
        assert_eq!(call.key, "the-key");
        assert!(!call.target.contains("the-key"), "{}", call.target);
        assert_eq!(call.agent, USER_AGENT);
        assert_eq!(call.accept, "application/json");
    }
}

#[test]
fn a_reading_that_fails_says_why() {
    let site = Site::answering(|call| match call.key.as_str() {
        "revoked" => failing(401),
        "narrow" => failing(403),
        "broken" => failing(500),
        "odd" => (200, "<html>Just a moment…</html>".to_string()),
        "hollow" => (200, json!({ "result": {} }).to_string()),
        "other" => ok(json!({ "id": 7, "username": "someone" })),
        // (An answer that isn't wrapped as the site's own client asks for it.)
        "bare" => (200, json!({ "result": { "data": { "yellow": 5 } } }).to_string()),
        // (What would be an answer, had the call gone well.)
        "unwell" => (500, ok(json!({ "yellow": 5 })).1),
        _ => ok(json!({ "yellow": 1 })),
    });
    let read = |key: &str| read_balance(&agent(), &site.at, key);
    assert_eq!(read("revoked"), Err(Problem::BadKey));
    assert_eq!(read("narrow"), Err(Problem::NoPermission));
    for key in ["broken", "odd", "hollow", "other", "unwell"] {
        assert_eq!(read(key), Err(Problem::Unreachable), "{key}");
    }
    assert_eq!(read("bare"), Ok(Balance { yellow: 5, green: 0, blue: 0 }));
    assert_eq!(read_balance(&agent(), &nowhere(), "any"), Err(Problem::Unreachable));

    // A call that's sent on elsewhere is left at that.
    let moving = Site::answering(|call| match call.call.as_str() {
        "buzz.getBuzzAccount" => (302, "/api/trpc/elsewhere".to_string()),
        _ => ok(json!({ "yellow": 5 })),
    });
    assert_eq!(read_balance(&agent(), &moving.at, "the-key"), Err(Problem::Unreachable));
    assert_eq!(moving.calls(), ["buzz.getBuzzAccount"]);

    // A price or a purchase that can't be read is one that isn't known.
    for key in ["revoked", "narrow", "broken", "odd"] {
        assert_eq!(read_price(&agent(), &site.at, key, 1), Price::Unknown, "{key}");
        assert_eq!(read_access(&agent(), &site.at, key, &[1]), HashMap::new(), "{key}");
    }
    assert_eq!(read_price(&agent(), &nowhere(), "any", 1), Price::Unknown);
    assert_eq!(read_access(&agent(), &nowhere(), "any", &[1]), HashMap::new());
}

#[test]
fn gates_are_filled_in_with_their_prices_and_whether_they_are_the_accounts() {
    let site = Site::answering(|call| match call.call.as_str() {
        "modelVersion.getById" => match version_asked(call) {
            1 => ok(costing(500)),
            2 => failing(500),
            3 => ok(with_terms(json!({ "generation": { "price": 50 } }))),
            6 => ok(costing(600)),
            7 => ok(costing(700)),
            _ => failing(404),
        },
        "common.getEntityAccess" => ok(access(&[(1, false), (2, true), (3, false), (6, false), (7, false)])),
        _ => failing(404),
    });
    // (The fourth's price and the fifth's lack of one are settled; the last
    // two were asked for before, one of them in vain.)
    let before = [
        gate(1),
        gate(2),
        gate(3),
        priced(4, buzz(900), Some(true)),
        priced(5, Price::NotSold, Some(true)),
        priced(6, Price::Reading, None),
        priced(7, Price::Unknown, Some(false)),
    ];
    let mut gates = before.clone();
    fill(&agent(), &site.at, "the-key", &mut gates, || false);
    assert_eq!(
        gates.each_ref().map(|gate| gate.price),
        [buzz(500), Price::Unknown, Price::NotSold, buzz(900), Price::NotSold, buzz(600), buzz(700)]
    );
    // (Where nothing says whether it's the account's, that isn't assumed.)
    assert_eq!(
        gates.each_ref().map(|gate| gate.unlocked),
        [Some(false), Some(true), Some(false), None, None, Some(false), Some(false)]
    );

    // A call for each price that isn't settled, then one for them all.
    let asked = site.asked();
    let calls: Vec<(&str, &Value)> = asked.iter().map(|a| (a.call.as_str(), &a.input["json"])).collect();
    assert_eq!(
        calls,
        [
            ("modelVersion.getById", &json!({ "id": 1 })),
            ("modelVersion.getById", &json!({ "id": 2 })),
            ("modelVersion.getById", &json!({ "id": 3 })),
            ("modelVersion.getById", &json!({ "id": 6 })),
            ("modelVersion.getById", &json!({ "id": 7 })),
            ("common.getEntityAccess", &json!({ "entityType": "ModelVersion", "entityId": [1, 2, 3, 4, 5, 6, 7] })),
        ]
    );
    assert!(asked.iter().all(|a| a.key == "the-key"));
}

#[test]
fn nothing_is_asked_without_a_key_or_a_gate_or_once_cancelled() {
    let site = Site::answering(|call| match call.call.as_str() {
        "modelVersion.getById" => ok(costing(500)),
        _ => ok(access(&[(1, true), (2, true)])),
    });
    let untouched = [gate(1), gate(2)];

    let mut gates = untouched.clone();
    fill(&agent(), &site.at, "", &mut gates, || false);
    fill(&agent(), &site.at, "the-key", &mut [], || false);
    fill(&agent(), &site.at, "the-key", &mut gates, || true);
    assert_eq!((gates, site.asked().len()), (untouched.clone(), 0));

    // Cancelled part of the way: what's under way is the last of it.
    let mut gates = untouched.clone();
    fill(&agent(), &site.at, "the-key", &mut gates, || !site.asked().is_empty());
    assert_eq!(site.calls(), ["modelVersion.getById"]);
    assert_eq!(gates, [Gate { price: buzz(500), ..gate(1) }, gate(2)]);

    // Cancelled before the last call: the prices are in, and nothing more.
    let mut gates = untouched.clone();
    fill(&agent(), &site.at, "the-key", &mut gates, || site.asked().len() == 3);
    assert_eq!(site.calls(), ["modelVersion.getById", "modelVersion.getById", "modelVersion.getById"]);
    assert_eq!(gates, [Gate { price: buzz(500), ..gate(1) }, Gate { price: buzz(500), ..gate(2) }]);
}

/// How a download of `path` at `site` ends, with what the card knew of a gate
/// on it, when the site turns it down.
fn turned_down(site: &Site, path: &str, key: &str, gate: Option<Gate>) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    let dir = std::env::temp_dir().join(format!("ctf_buzz_refused_{}", std::process::id()));
    let url = format!("{}{path}", site.at);
    super::super::run_download(url, Some("a.safetensors".to_string()), key.to_string(), gate, dir.clone(), tx, egui::Context::default());
    assert!(!dir.exists(), "nothing is saved of a download that's turned down");
    match rx.try_iter().last() {
        Some(super::super::DlMsg::Error(why)) => why,
        _ => panic!("the download of {path} didn't fail"),
    }
}

#[test]
fn a_download_civitai_turns_down_fails_with_what_is_asked() {
    let not_for_sale = "This model isn't commercially available at the moment. Its base model, Flux.2 Klein 9B, is licensed for non-commercial use only.";
    let site = Site::answering(move |call| match call.call.as_str() {
        "/api/download/models/1" => (403, EARLY_ACCESS.to_string()),
        "/api/download/models/2" => (403, json!({ "error": "Early Access", "message": not_for_sale }).to_string()),
        "/api/download/models/3" => (403, json!({ "error": "You do not have access to download this file" }).to_string()),
        "/api/download/models/4" => (401, json!({ "error": "Unauthorized", "message": "The creator of this asset requires you to be logged in to download it" }).to_string()),
        _ => (404, json!({ "error": "File not found" }).to_string()),
    });
    let asks = Some(priced(1, buzz(500), Some(false)));

    // Behind a gate: what its creator asks, as the card knew it.
    assert_eq!(
        turned_down(&site, "/api/download/models/1", "the-key", asks.clone()),
        "its creator asks 500 Buzz first — pay on its Civitai page (click the card), then try again"
    );
    assert_eq!(
        turned_down(&site, "/api/download/models/1", "the-key", None),
        "its creator asks for Buzz first — pay on its Civitai page (click the card), then try again"
    );
    assert_eq!(
        turned_down(&site, "/api/download/models/1", "", asks.clone()),
        "its creator asks 500 Buzz first — pay on its Civitai page (click the card) and add your API key in settings, then try again"
    );
    // (The key goes with a download as it did: in its address.)
    let sent: Vec<bool> = site.asked().iter().map(|a| a.target.ends_with("?token=the-key")).collect();
    assert_eq!(sent, [true, true, false]);

    // Not for sale: Civitai's own words.
    assert_eq!(turned_down(&site, "/api/download/models/2", "the-key", asks.clone()), not_for_sale);
    // Turned down for anything else, it fails as it did before — whatever
    // the card knew.
    assert_eq!(
        turned_down(&site, "/api/download/models/3", "the-key", asks.clone()),
        "access denied — this model may be early access (needs purchase) or your API key lacks permission"
    );
    assert_eq!(
        turned_down(&site, "/api/download/models/4", "", asks.clone()),
        "login required — add your Civitai API key in settings"
    );
    assert_eq!(
        turned_down(&site, "/api/download/models/5", "the-key", asks),
        "file not found on Civitai (the version may have been removed)"
    );
}

#[test]
fn the_account_is_read_from_civitai() {
    assert_eq!(Account::default().site, "https://civitai.com");
}

// ---------------------------------------------------------------------------
// A screen without a window
// ---------------------------------------------------------------------------

/// What a frame painted: its texts, and its icons (the images, tinted), with
/// where and in what colour.
#[derive(Default)]
struct Painted {
    texts: Vec<(String, Rect, Color32)>,
    icons: Vec<(Rect, Color32)>,
}

fn painted(shapes: &[egui::epaint::ClippedShape]) -> Painted {
    fn walk(shape: &egui::Shape, out: &mut Painted) {
        match shape {
            egui::Shape::Text(text) => {
                let own = text.galley.job.sections.first().map(|section| section.format.color);
                let colour = own.filter(|c| *c != Color32::PLACEHOLDER).or(text.override_text_color).unwrap_or(text.fallback_color);
                out.texts.push((text.galley.text().to_string(), text.visual_bounding_rect(), colour));
            }
            egui::Shape::Rect(rect) if rect.brush.is_some() => out.icons.push((rect.rect, rect.fill)),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| walk(shape, out)),
            _ => {}
        }
    }
    let mut out = Painted::default();
    shapes.iter().for_each(|clipped| walk(&clipped.shape, &mut out));
    out
}

/// `scene` drawn into a 640 × 360 screen a frame at a time — its window in
/// front of the others or not — keeping what it needs in `state`.
struct Screen<S> {
    ctx: egui::Context,
    state: S,
    scene: fn(&mut egui::Ui, &mut S),
    focused: bool,
    /// What the last frame painted.
    seen: Painted,
    time: f64,
}

impl<S> Screen<S> {
    fn new(state: S, scene: fn(&mut egui::Ui, &mut S)) -> Self {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let mut screen = Screen { ctx, state, scene, focused: true, seen: Painted::default(), time: 0.0 };
        screen.frame(vec![]);
        screen
    }

    fn frame(&mut self, events: Vec<Event>) {
        self.time += 1.0 / 60.0;
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(640.0, 360.0))),
            time: Some(self.time),
            focused: self.focused,
            events,
            ..Default::default()
        };
        let (scene, state) = (self.scene, &mut self.state);
        let out = self.ctx.run_ui(input, |ui| scene(ui, state));
        self.seen = painted(&out.shapes);
    }

    /// A few frames with nothing new, then long enough that a hover text
    /// shows and the next click isn't the second of two.
    fn idle(&mut self) {
        for _ in 0..3 {
            self.frame(vec![]);
        }
        self.time += 1.0;
        self.frame(vec![]);
    }

    fn click(&mut self, at: Pos2) {
        for pressed in [true, false] {
            let button = Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
            self.frame(vec![Event::PointerMoved(at), button]);
        }
    }

    /// The pointer resting at `at` until its hover text is up (which takes a
    /// frame to size itself before it shows).
    fn rest(&mut self, at: Pos2) {
        self.frame(vec![Event::PointerMoved(at)]);
        self.idle();
        self.frame(vec![]);
    }

    /// The pointer off everything, and settled.
    fn away(&mut self) {
        self.frame(vec![Event::PointerMoved(pos2(630.0, 350.0))]);
        self.idle();
    }

    /// The window behind another, then in front again.
    fn leave_and_come_back(&mut self) {
        self.focused = false;
        self.frame(vec![]);
        self.focused = true;
        self.frame(vec![]);
    }

    /// Frames until `done` — which must be soon.
    fn until(&mut self, what: &str, done: impl Fn(&Self) -> bool) {
        let since = Instant::now();
        loop {
            self.frame(vec![]);
            if done(self) {
                return;
            }
            assert!(since.elapsed() < Duration::from_secs(10), "waited for {what}; on screen: {:?}", self.all());
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Frames for a while, for what's under way elsewhere to arrive.
    fn a_while(&mut self) {
        for _ in 0..60 {
            self.frame(vec![]);
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Every text of the last frame.
    fn all(&self) -> Vec<&str> {
        self.seen.texts.iter().map(|(text, ..)| text.as_str()).collect()
    }

    fn has(&self, text: &str) -> bool {
        self.seen.texts.iter().any(|(painted, ..)| painted == text)
    }

    /// Where the last frame painted `text` — once — and in what colour.
    fn text(&self, text: &str) -> (Rect, Color32) {
        let found: Vec<_> = self.seen.texts.iter().filter(|(painted, ..)| painted == text).collect();
        assert_eq!(found.len(), 1, "{text:?} is painted {} times; on screen: {:?}", found.len(), self.all());
        (found[0].1, found[0].2)
    }

    /// The icons of the last frame that are `size` points across.
    fn icons(&self, size: f32) -> Vec<(Rect, Color32)> {
        self.seen.icons.iter().filter(|(rect, _)| (rect.width() - size).abs() < 0.01).copied().collect()
    }
}

/// `icon` then `text`, close together on one line.
fn assert_icon_then_text(icon: Rect, text: Rect) {
    assert!(icon.right() <= text.left() && text.left() - icon.right() < 8.0, "{icon:?} then {text:?}");
    assert!((icon.center().y - text.center().y).abs() < 3.0, "{icon:?} beside {text:?}");
}

// ---------------------------------------------------------------------------
// The counter
// ---------------------------------------------------------------------------

/// The counter's row by itself: the account, the key in the settings and
/// whether it's being typed — and what `tick` and clicks have asked for.
struct Counter {
    account: Account,
    key: String,
    typing: bool,
    looks: Vec<Look>,
    asks: Vec<Ask>,
}

fn counter(ui: &mut egui::Ui, c: &mut Counter) {
    c.looks.extend(c.account.tick(ui.ctx(), &c.key, c.typing));
    c.asks.extend(c.account.row(ui, !c.key.trim().is_empty()));
}

fn counter_at(site: &Site, key: &str) -> Screen<Counter> {
    let state = Counter { account: Account::at(&site.at), key: key.to_string(), typing: false, looks: vec![], asks: vec![] };
    Screen::new(state, counter)
}

/// The account's last reading, `by` longer ago.
fn age(account: &mut Account, by: Duration) {
    account.asked = account.asked.and_then(|at| at.checked_sub(by));
}

#[test]
fn the_counter_shows_each_kind_of_buzz_the_account_holds() {
    let site = Site::answering(|_| ok(json!({ "blue": 350, "green": 1200, "yellow": 12480 })));
    let mut screen = counter_at(&site, " the-key\n");
    // Until it's read there's nothing to say, in grey.
    assert_eq!(screen.text("…").1, MUTED());

    screen.until("the balance", |s| s.has("12,480"));
    assert_eq!(site.calls(), ["buzz.getBuzzAccount"]);
    assert_eq!(site.asked()[0].key, "the-key", "without the blanks around it");

    // A bolt, then the amount, for each kind in its colour — yellow, green,
    // blue, left to right on one line.
    let bolts = screen.icons(ROW_BOLT);
    assert_eq!(bolts.len(), 3);
    let mut last = f32::NEG_INFINITY;
    for ((amount, kind), (bolt, tint)) in ["12,480", "1,200", "350"].into_iter().zip(Kind::ALL).zip(bolts) {
        let (text, colour) = screen.text(amount);
        assert_eq!((tint, colour), (kind.colour(), kind.colour()), "{amount} and its bolt are {kind:?}");
        assert_icon_then_text(bolt, text);
        assert!(last + 8.0 < bolt.left(), "{kind:?} is apart from the kind before");
        last = text.right();
    }
    assert!(!screen.has("…"));

    // Hovered, it has the amounts in full and what a click does — which is
    // to ask for it again.
    let at = screen.text("12,480").0.center();
    screen.rest(at);
    assert!(screen.has("Yellow Buzz: 12,480\nGreen Buzz: 1,200\nBlue Buzz: 350\nClick to update."), "{:?}", screen.all());
    screen.click(at);
    assert_eq!(screen.state.asks, [Ask::Again]);
    assert_eq!(site.calls().len(), 1, "the row asks; reading is up to the view");
}

#[test]
fn large_amounts_are_shortened_in_the_counter_and_whole_in_its_hover_text() {
    let site = Site::answering(|_| ok(json!({ "blue": 99_999, "green": 311_400, "yellow": 1_340_000 })));
    let mut screen = counter_at(&site, "the-key");
    screen.until("the balance", |s| s.has("1.3M"));
    assert!(screen.has("311k") && screen.has("99,999"), "{:?}", screen.all());
    screen.rest(screen.text("311k").0.center());
    assert!(screen.has("Yellow Buzz: 1,340,000\nGreen Buzz: 311,400\nBlue Buzz: 99,999\nClick to update."), "{:?}", screen.all());
}

#[test]
fn an_account_with_no_buzz_shows_a_nought() {
    let site = Site::answering(|call| match call.key.as_str() {
        "green-only" => ok(json!({ "blue": 0, "green": 5, "yellow": 0 })),
        _ => ok(json!({ "blue": 0, "green": 0, "yellow": 0 })),
    });
    let mut screen = counter_at(&site, "empty");
    screen.until("the balance", |s| s.has("0"));
    assert_eq!(screen.text("0").1, Kind::Yellow.colour());
    assert_eq!(screen.icons(ROW_BOLT).len(), 1);

    // A kind it has none of isn't listed beside one it has.
    screen.state.key = "green-only".to_string();
    screen.until("the other account's", |s| s.has("5"));
    assert_eq!(screen.text("5").1, Kind::Green.colour());
    assert_eq!((screen.icons(ROW_BOLT).len(), screen.has("0")), (1, false));
}

#[test]
fn without_a_key_the_counter_asks_for_one() {
    let site = Site::answering(|_| ok(json!({ "yellow": 1 })));
    let mut screen = counter_at(&site, "  ");
    screen.idle();
    let (words, colour) = screen.text("Add your API key to see your Buzz");
    assert_eq!(colour, MUTED());
    let bolts = screen.icons(ROW_BOLT);
    assert_eq!(bolts.len(), 1);
    assert_eq!(bolts[0].1, MUTED());
    assert_icon_then_text(bolts[0].0, words);

    screen.rest(words.center());
    assert!(
        screen.has("Shows the Buzz on your Civitai account, and what a paid model costs.\nClick to add your API key."),
        "{:?}",
        screen.all()
    );
    screen.click(words.center());
    assert_eq!(screen.state.asks, [Ask::Key]);
    assert!(screen.state.looks.is_empty());
    assert!(site.asked().is_empty(), "nothing is asked of Civitai without a key");
}

#[test]
fn a_reading_that_fails_is_told_where_the_balance_would_be() {
    let site = Site::answering(|call| match call.key.as_str() {
        "revoked" => failing(401),
        "narrow" => failing(403),
        _ => failing(502),
    });
    for (key, words, tip, click) in [
        ("revoked", "API key not accepted", "Civitai didn't accept the saved API key.\nClick to check it.", Ask::Key),
        ("narrow", "This API key can't read Buzz", "", Ask::Key),
        ("any", "Buzz unavailable", "Couldn't read your Buzz from Civitai.\nClick to try again.", Ask::Again),
    ] {
        let mut screen = counter_at(&site, key);
        screen.until(words, |s| s.has(words));
        let (at, colour) = screen.text(words);
        assert_eq!(colour, MUTED());
        assert_eq!(screen.icons(ROW_BOLT).iter().map(|(_, tint)| *tint).collect::<Vec<_>>(), [MUTED()]);
        screen.rest(at.center());
        if tip.is_empty() {
            assert!(screen.all().iter().any(|text| text.contains("Full Access")), "{:?}", screen.all());
        } else {
            assert!(screen.has(tip), "{:?}", screen.all());
        }
        screen.click(at.center());
        assert_eq!(screen.state.asks, [click], "a click on {words:?}");
        assert_eq!(site.asked().iter().filter(|a| a.key == key).count(), 1, "{words:?} isn't tried over and over");
    }
}

#[test]
fn another_key_is_another_account() {
    let hold = Arc::new(AtomicBool::new(false));
    let held = hold.clone();
    let site = Site::answering(move |call| {
        if call.key == "slow" {
            hold_while(&held);
        }
        ok(json!({ "yellow": if call.key == "second" { 7 } else { 100 } }))
    });
    let mut screen = counter_at(&site, "first");
    screen.until("the first account's", |s| s.has("100"));
    // (The first key the view sees is a new one, too.)
    assert_eq!(std::mem::take(&mut screen.state.looks), [Look::All]);

    // What's on show goes with its key, and everything is looked at again.
    screen.state.key = "second".to_string();
    screen.frame(vec![]);
    assert_eq!(screen.state.looks, [Look::All]);
    assert!(screen.has("…") && !screen.has("100"), "{:?}", screen.all());
    screen.until("the second account's", |s| s.has("7"));
    assert_eq!(std::mem::take(&mut screen.state.looks), [Look::All], "once");

    // An answer still on its way for a key that's gone is never shown.
    hold.store(true, Ordering::SeqCst);
    screen.state.key = "slow".to_string();
    screen.until("the slow reading to start", |_| site.asked().iter().any(|a| a.key == "slow"));
    screen.state.key = "second".to_string();
    screen.until("the second account's again", |s| s.has("7"));
    hold.store(false, Ordering::SeqCst);
    screen.a_while();
    assert!(screen.has("7") && !screen.has("100"), "{:?}", screen.all());
    assert_eq!(screen.state.looks, [Look::All, Look::All]);
}

#[test]
fn a_key_is_not_tried_while_it_is_being_typed() {
    let site = Site::answering(|_| ok(json!({ "yellow": 42 })));
    let mut screen = counter_at(&site, "");
    screen.state.typing = true;
    for typed in ["h", "half-a-k", "the-whole-key"] {
        screen.state.key = typed.to_string();
        screen.idle();
    }
    screen.a_while();
    assert!(site.asked().is_empty() && screen.state.looks.is_empty());

    screen.state.typing = false;
    screen.until("the balance", |s| s.has("42"));
    assert_eq!(site.asked().iter().map(|a| a.key.as_str()).collect::<Vec<_>>(), ["the-whole-key"]);
    assert_eq!(screen.state.looks, [Look::All]);
}

#[test]
fn taking_the_key_out_clears_the_counter() {
    let site = Site::answering(|_| ok(json!({ "yellow": 100 })));
    let mut screen = counter_at(&site, "the-key");
    screen.until("the balance", |s| s.has("100"));
    screen.state.looks.clear();

    // (In the settings, as it would be: that doesn't keep what's shown.)
    screen.state.typing = true;
    screen.state.key.clear();
    screen.frame(vec![]);
    assert!(screen.has("Add your API key to see your Buzz") && !screen.has("100"), "{:?}", screen.all());
    assert_eq!(screen.state.looks, [Look::All], "what was the account's is no one's now");
    screen.state.typing = false;
    screen.idle();
    screen.a_while();
    assert_eq!(screen.state.looks, [Look::All], "once");
    assert_eq!(site.asked().len(), 1);

    // The same key again is read again.
    screen.state.key = "the-key".to_string();
    screen.until("the balance again", |s| s.has("100"));
    assert_eq!(site.asked().len(), 2);
}

#[test]
fn coming_back_to_the_app_reads_the_balance_again() {
    let (paid, hold) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (spent, held) = (paid.clone(), hold.clone());
    let site = Site::answering(move |_| {
        hold_while(&held);
        ok(json!({ "yellow": if spent.load(Ordering::SeqCst) { 500 } else { 1000 } }))
    });
    let mut screen = counter_at(&site, "the-key");
    screen.until("the balance", |s| s.has("1,000"));
    screen.state.looks.clear();

    // Away and straight back: the resources are looked at again — the
    // balance was only just read.
    screen.leave_and_come_back();
    screen.a_while();
    assert_eq!(screen.state.looks, [Look::Unpaid]);
    assert_eq!(site.asked().len(), 1);

    // Away for as long as paying takes. What was on show stays until the
    // new balance is in.
    paid.store(true, Ordering::SeqCst);
    age(&mut screen.state.account, JUST_READ + Duration::from_secs(1));
    screen.leave_and_come_back();
    assert_eq!(screen.state.looks, [Look::Unpaid, Look::Unpaid]);
    assert!(screen.has("1,000"));
    screen.until("the balance after paying", |s| s.has("500"));
    assert_eq!(site.asked().len(), 2);

    // In front all the while, it's read again once it has aged.
    paid.store(false, Ordering::SeqCst);
    screen.a_while();
    assert_eq!(site.asked().len(), 2);
    age(&mut screen.state.account, FRESH_FOR + Duration::from_secs(1));
    screen.until("the aged balance read again", |s| s.has("1,000"));
    assert_eq!(site.asked().len(), 3);
    assert_eq!(screen.state.looks, [Look::Unpaid, Look::Unpaid], "which is no reason to look at the resources");

    // A reading that's slow to come back isn't joined by another.
    hold.store(true, Ordering::SeqCst);
    age(&mut screen.state.account, FRESH_FOR + Duration::from_secs(1));
    screen.until("the slow reading to start", |_| site.asked().len() == 4);
    age(&mut screen.state.account, JUST_READ + Duration::from_secs(1));
    screen.leave_and_come_back();
    screen.a_while();
    assert_eq!(site.asked().len(), 4);
    hold.store(false, Ordering::SeqCst);
}

#[test]
fn a_balance_that_cannot_be_updated_stays() {
    let down = Arc::new(AtomicBool::new(false));
    let is_down = down.clone();
    let site = Site::answering(move |_| if is_down.load(Ordering::SeqCst) { failing(503) } else { ok(json!({ "yellow": 1000 })) });
    let mut screen = counter_at(&site, "the-key");
    screen.until("the balance", |s| s.has("1,000"));

    down.store(true, Ordering::SeqCst);
    let ctx = screen.ctx.clone();
    screen.state.account.read(&ctx, "the-key");
    screen.until("the failed reading", |s| s.state.account.problem.is_some());
    let (at, colour) = screen.text("1,000");
    assert_eq!(colour, Kind::Yellow.colour());
    screen.rest(at.center());
    assert!(
        screen.has("Yellow Buzz: 1,000\nGreen Buzz: 0\nBlue Buzz: 0\nCouldn't update just now. Click to try again."),
        "{:?}",
        screen.all()
    );

    // The next reading that works clears that.
    down.store(false, Ordering::SeqCst);
    screen.state.account.read(&ctx, "the-key");
    screen.until("the reading that works", |s| s.state.account.problem.is_none());
    assert!(screen.has("1,000"));
}

#[test]
fn a_key_civitai_no_longer_takes_leaves_no_balance_on_show() {
    for (status, words) in [(401, "API key not accepted"), (403, "This API key can't read Buzz")] {
        let gone = Arc::new(AtomicBool::new(false));
        let is_gone = gone.clone();
        let site = Site::answering(move |_| if is_gone.load(Ordering::SeqCst) { failing(status) } else { ok(json!({ "yellow": 1000 })) });
        let mut screen = counter_at(&site, "the-key");
        screen.until("the balance", |s| s.has("1,000"));

        gone.store(true, Ordering::SeqCst);
        let ctx = screen.ctx.clone();
        screen.state.account.read(&ctx, "the-key");
        screen.until(words, |s| s.has(words));
        assert!(!screen.has("1,000"), "{:?}", screen.all());
    }
}

#[test]
fn a_second_look_fills_in_the_gates_it_is_given() {
    let hold = Arc::new(AtomicBool::new(false));
    let held = hold.clone();
    let site = Site::answering(move |call| {
        hold_while(&held);
        match call.call.as_str() {
            "modelVersion.getById" => ok(costing(version_asked(call) * 100)),
            _ => ok(access(&[(1, true), (2, false)])),
        }
    });
    let ctx = egui::Context::default();
    let mut account = Account::at(&site.at);
    assert_eq!((account.looked(), account.is_looking()), (None, false));

    account.look(&ctx, " the-key ", vec![gate(1), priced(2, buzz(900), Some(true))]);
    assert!(account.is_looking());
    let found = soon("the look", || account.looked());
    assert!(!account.is_looking());
    assert_eq!(found, [priced(1, buzz(100), Some(true)), priced(2, buzz(900), Some(false))]);
    assert_eq!(account.looked(), None, "once");
    assert_eq!(site.calls(), ["common.getEntityAccess", "modelVersion.getById"]);
    assert!(site.asked().iter().all(|a| a.key == "the-key"));

    // Nothing to look at, or no key to look with: nothing is asked — and a
    // look under way is dropped, and goes no further than the call it's at.
    for (key, gates) in [("the-key", vec![]), ("  ", vec![gate(3)])] {
        let before = site.asked().len();
        hold.store(true, Ordering::SeqCst);
        account.look(&ctx, "the-key", vec![gate(1), gate(2)]);
        assert!(account.is_looking());
        soon("the look to start", || (site.asked().len() > before).then_some(()));
        account.look(&ctx, key, gates);
        assert!(!account.is_looking());
        hold.store(false, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(account.looked(), None);
        assert_eq!(site.asked().len(), before + 1, "{:?}", site.calls());
    }

    // One look takes the place of another the same way.
    let before = site.asked().len();
    hold.store(true, Ordering::SeqCst);
    account.look(&ctx, "the-key", vec![gate(1), gate(2)]);
    soon("the look to start", || (site.asked().len() > before).then_some(()));
    account.look(&ctx, "the-key", vec![priced(3, buzz(300), None)]);
    hold.store(false, Ordering::SeqCst);
    assert_eq!(soon("the second look", || account.looked()), [priced(3, buzz(300), None)]);
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!((account.looked(), account.is_looking()), (None, false));
    assert_eq!(site.asked().len(), before + 2, "{:?}", site.calls());
}

// ---------------------------------------------------------------------------
// The price, on a resource's card
// ---------------------------------------------------------------------------

/// The download button's size on a card, and where the card is drawn.
const BUTTON: f32 = 20.0;
const COLUMN: Rect = Rect::from_min_max(pos2(20.0, 10.0), pos2(416.0, 350.0));

fn resource(name: &str, gate: Option<Gate>) -> UiResource {
    UiResource {
        name: name.to_string(),
        url: "https://civitai.com/models/1?modelVersionId=2".to_string(),
        triggers: vec!["a trigger".to_string()],
        tex: None,
        has_video_only: false,
        download_url: "https://civitai.com/api/download/models/2".to_string(),
        download_filename: Some(format!("{name}.safetensors")),
        resource_type: "LORA".to_string(),
        gate,
    }
}

/// One card by itself, and the download its button last asked for.
struct Card {
    data: UiResource,
    downloads: Vec<Download>,
    dirs: DownloadDirs,
    asked: Option<DownloadRequest>,
    /// How tall it came out.
    height: f32,
}

fn card(ui: &mut egui::Ui, c: &mut Card) {
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(COLUMN));
    let mut asked = None;
    resource_card(&mut column, &c.data, &c.downloads, &c.dirs, 160.0, &mut asked);
    if asked.is_some() {
        c.asked = asked;
    }
    c.height = column.min_rect().height();
}

fn card_of(gate: Option<Gate>) -> Screen<Card> {
    let state = Card { data: resource("Soft Pastel", gate), downloads: vec![], dirs: DownloadDirs::default(), asked: None, height: 0.0 };
    let mut screen = Screen::new(state, card);
    screen.idle();
    screen
}

/// The card's download of the resource, in this state.
fn download(rx: Option<std::sync::mpsc::Receiver<super::super::DlMsg>>, ok: bool, status: &str) -> Download {
    Download { name: "Soft Pastel".to_string(), rx, received: 10, total: 100, status: status.to_string(), ok }
}

/// The button of the last frame's card: where, and its tint.
fn button(screen: &Screen<Card>) -> (Rect, Color32) {
    let buttons = screen.icons(BUTTON);
    assert_eq!(buttons.len(), 1, "one button on the card");
    buttons[0]
}

#[test]
fn a_paid_resource_has_its_price_under_the_download_button() {
    let free = card_of(None);
    let (free_button, arrow) = button(&free);
    assert_eq!(arrow, crate::theme::ACCENT1());
    assert!(free.icons(COST_BOLT).is_empty() && !free.has("1,200"));

    let screen = card_of(Some(priced(2, buzz(1200), Some(false))));
    let (paid_button, tint) = button(&screen);
    // The button is the one it was, where it was…
    assert_eq!((tint, paid_button.left(), paid_button.right()), (arrow, free_button.left(), free_button.right()));
    // …and under it, ending where it ends, are the bolt and then the amount,
    // in Buzz's yellow.
    let bolts = screen.icons(COST_BOLT);
    assert_eq!(bolts.len(), 1);
    let (bolt, bolt_tint) = bolts[0];
    let (text, colour) = screen.text("1,200");
    assert_eq!((bolt_tint, colour), (Kind::Yellow.colour(), Kind::Yellow.colour()));
    assert_icon_then_text(bolt, text);
    assert!(bolt.top() >= paid_button.bottom() && bolt.top() - paid_button.bottom() < 6.0, "{bolt:?} under {paid_button:?}");
    assert!((text.right() - paid_button.right()).abs() < 2.0, "{text:?} ends with {paid_button:?}");
    // A card with two lines of text is no taller for it.
    assert_eq!(screen.state.height, free.state.height);
    // The name keeps to its side of it.
    let name = screen.text("Soft Pastel").0;
    assert!(name.right() < bolt.left());
}

#[test]
fn a_price_that_is_not_known_says_paid() {
    for price in [Price::Unread, Price::Unknown, Price::NotSold] {
        let screen = card_of(Some(priced(2, price, None)));
        let (text, colour) = screen.text("Paid");
        assert_eq!(colour, Kind::Yellow.colour());
        let bolts = screen.icons(COST_BOLT);
        assert_eq!(bolts.len(), 1, "{price:?}");
        assert_icon_then_text(bolts[0].0, text);
    }
}

#[test]
fn a_price_being_read_is_waited_for() {
    let free = card_of(None);
    let mut screen = card_of(Some(priced(2, Price::Reading, None)));
    // The card is the one of a resource that asks for nothing…
    assert_eq!((screen.has("Paid"), screen.icons(COST_BOLT).len()), (false, 0));
    assert_eq!((button(&screen), screen.state.height), (button(&free), free.state.height));
    // …but a download that's refused is still told what for.
    screen.click(button(&screen).0.center());
    assert_eq!(screen.state.asked.take().expect("a download").gate, Some(priced(2, Price::Reading, None)));
}

#[test]
fn the_price_goes_once_the_resource_is_the_accounts_or_here() {
    let asks = || Some(priced(2, buzz(500), Some(false)));
    let shows = |screen: &Screen<Card>| (screen.has("500"), screen.icons(COST_BOLT).len());

    // Bought (or its own).
    assert_eq!(shows(&card_of(Some(priced(2, buzz(500), Some(true))))), (false, 0));
    // Not known to be: it shows.
    assert_eq!(shows(&card_of(Some(priced(2, buzz(500), None)))), (true, 1));

    // On its way, and arrived.
    let (tx, rx) = std::sync::mpsc::channel();
    let mut screen = card_of(asks());
    assert_eq!(shows(&screen), (true, 1));
    screen.state.downloads = vec![download(Some(rx), false, "Downloading…")];
    screen.idle();
    assert_eq!(shows(&screen), (false, 0));
    assert!(screen.has("10%"));
    screen.state.downloads = vec![download(None, true, "Saved to Soft Pastel.safetensors")];
    screen.idle();
    assert_eq!(shows(&screen), (false, 0));
    drop(tx);

    // In the models folder already.
    let folder = std::env::temp_dir().join(format!("ctf_buzz_card_{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("Soft Pastel.safetensors"), b"a model").unwrap();
    let mut screen = card_of(asks());
    screen.state.dirs.lora = folder.display().to_string();
    screen.idle();
    assert_eq!(shows(&screen), (false, 0));
    let _ = std::fs::remove_dir_all(&folder);
}

#[test]
fn a_refused_download_keeps_its_price_under_the_warning() {
    let mut screen = card_of(Some(priced(2, buzz(500), Some(false))));
    let before = button(&screen).0;
    let why = "Failed: its creator asks 500 Buzz first — pay on its Civitai page (click the card), then try again";
    screen.state.downloads = vec![download(None, false, why)];
    screen.idle();
    let (warning, tint) = button(&screen);
    assert_eq!((warning, tint == crate::theme::ACCENT1()), (before, false), "the warning is where the button was");
    let (text, _) = screen.text("500");
    assert_icon_then_text(screen.icons(COST_BOLT)[0].0, text);
    assert!(text.top() >= warning.bottom());

    // Each says its own on hover.
    screen.rest(warning.center());
    assert!(screen.has(why), "{:?}", screen.all());
    screen.away();
    screen.rest(text.center());
    assert!(screen.all().iter().any(|text| text.starts_with("500 Buzz to download.")), "{:?}", screen.all());
    assert!(!screen.has(why));
}

#[test]
fn the_price_tells_more_when_hovered_and_is_part_of_the_card_when_clicked() {
    let mut screen = card_of(Some(priced(2, Price::Buzz { now: 1200, was: Some(2000), blue: true }, Some(false))));
    let text = screen.text("1,200").0;
    screen.rest(text.center());
    let tip = "1,200 Buzz to download — on sale, was 2,000. Blue Buzz pays for it too.\nPay on its Civitai page (click the card), then download it here.";
    assert!(screen.has(tip), "{:?}", screen.all());

    // A click on it is a click on the card — which opens the resource's
    // page, where it's paid for — and not on the button above it.
    screen.away();
    screen.click(text.center());
    assert!(screen.state.asked.is_none());
}

#[test]
fn the_download_button_still_downloads_and_takes_what_is_asked_with_it() {
    let asks = priced(2, buzz(500), Some(false));
    let mut screen = card_of(Some(asks.clone()));
    screen.click(button(&screen).0.center());
    let asked = screen.state.asked.take().expect("a download");
    assert_eq!((asked.name.as_str(), asked.download_url.as_str()), ("Soft Pastel", "https://civitai.com/api/download/models/2"));
    assert_eq!(asked.gate, Some(asks));

    // Bought, or free: nothing is asked with it.
    for gate in [Some(priced(2, buzz(500), Some(true))), None] {
        let mut screen = card_of(gate);
        screen.click(button(&screen).0.center());
        let asked = screen.state.asked.take().expect("a download");
        assert_eq!(asked.gate, None);
    }
}

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

fn view(ui: &mut egui::Ui, state: &mut CivitaiState) {
    let mut column = ui.new_child(egui::UiBuilder::new().max_rect(COLUMN));
    show(&mut column, state, None, None);
}

/// The view with `key` saved and `gates` on the resources it has looked up,
/// its Buzz read from `site` — and nothing read from anywhere else.
fn view_of(site: &Site, key: &str, gates: Vec<Gate>) -> Screen<CivitaiState> {
    let items = gates.into_iter().map(|gate| resource(&format!("Resource {}", gate.version), Some(gate))).collect();
    let state = CivitaiState {
        // What `show` otherwise begins with: the saved settings from the
        // disk, and Civitai asked whether it's there.
        key_loaded: true,
        monitor_started: true,
        api_key: key.to_string(),
        buzz: Account::at(&site.at),
        result: Some(UiResult { source_url: None, sections: vec![UiSection { title: "LoRA Models".to_string(), items }] }),
        ..CivitaiState::default()
    };
    Screen::new(state, view)
}

/// The gates of the resources on show.
fn gates(screen: &Screen<CivitaiState>) -> Vec<Gate> {
    let result = screen.state.result.as_ref().expect("resources on show");
    result.sections.iter().flat_map(|s| &s.items).filter_map(|item| item.gate.clone()).collect()
}

/// A site with 1,000 Buzz on the account, versions that cost a hundred times
/// their number, and `bought` to say which the account has.
fn shop(bought: impl Fn(&Asked) -> Vec<(u64, bool)> + Send + Sync + 'static) -> Site {
    Site::answering(move |call| match call.call.as_str() {
        "buzz.getBuzzAccount" => ok(json!({ "blue": 0, "green": 0, "yellow": 1000 })),
        "modelVersion.getById" => ok(costing(version_asked(call) * 100)),
        "common.getEntityAccess" => ok(access(&bought(call))),
        _ => failing(404),
    })
}

#[test]
fn the_counter_is_under_the_views_header_and_over_the_resources() {
    let site = shop(|_| vec![(5, false)]);
    let mut screen = view_of(&site, "the-key", vec![priced(5, buzz(500), Some(false))]);
    screen.until("the balance", |s| s.has("1,000"));
    let (heading, amount, section) = (screen.text("Civitai Resources").0, screen.text("1,000").0, screen.text("LoRA Models").0);
    assert!(heading.bottom() < amount.top() && amount.bottom() < section.top(), "{heading:?}, {amount:?}, {section:?}");
    let bolt = screen.icons(ROW_BOLT)[0].0;
    assert_icon_then_text(bolt, amount);
    assert!((bolt.left() - section.left()).abs() < 3.0, "the counter starts where the sections' titles do");
    // And the resource's price is on its card.
    assert!(screen.text("500").0.top() > section.bottom());
}

#[test]
fn a_click_on_the_counter_without_a_key_opens_the_settings() {
    let site = shop(|_| vec![]);
    let mut screen = view_of(&site, "", vec![gate(5)]);
    screen.idle();
    assert!(!screen.state.show_settings && !screen.has("API KEY"));
    assert!(screen.has("Paid"), "a price isn't known without a key");

    // (The settings' pop-up, where the key goes — and says what it's for.)
    screen.click(screen.text("Add your API key to see your Buzz").0.center());
    screen.idle();
    assert!(screen.state.show_settings && screen.has("API KEY"), "{:?}", screen.all());
    assert!(screen.has("For downloads and your Buzz. Stored encrypted here."));
    assert!(site.asked().is_empty());
}

/// How many of the resources on show have a price (or "Paid") on their card.
fn priced_cards(screen: &Screen<CivitaiState>) -> usize {
    screen.icons(COST_BOLT).len()
}

#[test]
fn a_key_put_in_brings_the_balance_and_the_prices() {
    let hold = Arc::new(AtomicBool::new(false));
    let held = hold.clone();
    let site = Site::answering(move |call| {
        hold_while(&held);
        match call.call.as_str() {
            "buzz.getBuzzAccount" => ok(json!({ "yellow": 1000 })),
            "modelVersion.getById" => ok(costing(version_asked(call) * 100)),
            _ => ok(access(&[(5, false), (12, true)])),
        }
    });
    let mut screen = view_of(&site, "", vec![gate(5), gate(12)]);
    screen.idle();
    assert_eq!(screen.seen.texts.iter().filter(|(text, ..)| text == "Paid").count(), 2);
    assert_eq!(priced_cards(&screen), 2);

    // (As the settings leave it when they close.) While the prices are
    // read, the cards wait…
    hold.store(true, Ordering::SeqCst);
    screen.state.api_key = "the-key".to_string();
    screen.until("the reading to start", |_| site.asked().len() >= 2);
    assert_eq!((screen.has("Paid"), priced_cards(&screen)), (false, 0), "{:?}", screen.all());
    assert_eq!(gates(&screen), [priced(5, Price::Reading, None), priced(12, Price::Reading, None)]);

    // …and then say what's asked: of the one the account has, nothing.
    hold.store(false, Ordering::SeqCst);
    screen.until("the balance and the prices", |s| s.has("1,000") && s.has("500"));
    assert_eq!(gates(&screen), [priced(5, buzz(500), Some(false)), priced(12, buzz(1200), Some(true))]);
    assert!(!screen.has("Paid") && !screen.has("1,200"), "{:?}", screen.all());
    assert_eq!(priced_cards(&screen), 1);
    assert_eq!(site.calls(), ["buzz.getBuzzAccount", "common.getEntityAccess", "modelVersion.getById", "modelVersion.getById"]);
    assert!(site.asked().iter().all(|a| a.key == "the-key"));

    screen.a_while();
    assert_eq!(site.asked().len(), 4, "and that's all that's asked");
}

/// What the lookup's worker sends the view when it's done: a result with a
/// resource for each of `gates`.
fn looked_up(gates: Vec<Gate>) -> super::super::CivMsg {
    let items = gates
        .into_iter()
        .map(|gate| super::super::Fetched {
            name: format!("Resource {}", gate.version),
            url: format!("https://civitai.com/models/1?modelVersionId={}", gate.version),
            triggers: vec![],
            image: None,
            has_video_only: false,
            resource_type: "LORA".to_string(),
            download_url: format!("https://civitai.com/api/download/models/{}", gate.version),
            download_filename: None,
            gate: Some(gate),
        })
        .collect();
    let section = super::super::FetchedSection { title: "LoRA Models".to_string(), items };
    super::super::CivMsg::Done(Box::new(super::super::CivResult { source_url: None, sections: vec![section] }))
}

#[test]
fn resources_that_are_looked_up_have_their_prices_read() {
    let hold = Arc::new(AtomicBool::new(false));
    let held = hold.clone();
    let site = Site::answering(move |call| match call.call.as_str() {
        "buzz.getBuzzAccount" => ok(json!({ "yellow": 1000 })),
        "modelVersion.getById" => {
            hold_while(&held);
            ok(costing(version_asked(call) * 100))
        }
        _ => ok(access(&[(5, false), (12, true)])),
    });
    // The view, with a lookup under way.
    let mut screen = view_of(&site, "the-key", vec![]);
    screen.state.result = None;
    let (tx, rx) = std::sync::mpsc::channel();
    screen.state.rx = Some(rx);
    screen.until("the balance", |s| s.has("1,000"));
    assert_eq!(site.calls(), ["buzz.getBuzzAccount"]);

    // Its resources arrive as the public API has them — and are on show at
    // once, the paid-for ones without a word on that yet.
    hold.store(true, Ordering::SeqCst);
    tx.send(looked_up(vec![gate(5), gate(12)])).unwrap();
    screen.frame(vec![]);
    assert!(screen.has("Resource 5") && screen.has("Resource 12"), "{:?}", screen.all());
    assert_eq!((screen.has("Paid"), priced_cards(&screen)), (false, 0));
    assert_eq!(gates(&screen), [priced(5, Price::Reading, None), priced(12, Price::Reading, None)]);

    hold.store(false, Ordering::SeqCst);
    screen.until("the price", |s| s.has("500"));
    assert_eq!(gates(&screen), [priced(5, buzz(500), Some(false)), priced(12, buzz(1200), Some(true))]);
    assert_eq!((screen.has("Paid"), screen.has("1,200"), priced_cards(&screen)), (false, false, 1));
    screen.a_while();
    assert_eq!(site.calls(), ["buzz.getBuzzAccount", "common.getEntityAccess", "modelVersion.getById", "modelVersion.getById"]);
}

#[test]
fn without_a_key_paid_resources_say_so_and_nothing_is_asked() {
    let site = shop(|_| vec![]);
    let mut screen = view_of(&site, "", vec![]);
    screen.state.result = None;
    let (tx, rx) = std::sync::mpsc::channel();
    screen.state.rx = Some(rx);
    screen.idle();
    tx.send(looked_up(vec![gate(5)])).unwrap();
    screen.idle();
    assert_eq!(gates(&screen), [gate(5)]);
    let paid = screen.text("Paid").0;
    screen.rest(paid.center());
    let tip = "Its creator asks for Buzz before it can be downloaded.\nAdd your Civitai API key in this view's settings to see how much.";
    assert!(screen.has(tip), "{:?}", screen.all());
    screen.a_while();
    assert!(site.asked().is_empty());
}

#[test]
fn a_price_that_cannot_be_read_says_paid() {
    let site = Site::answering(|call| match call.call.as_str() {
        "buzz.getBuzzAccount" => ok(json!({ "yellow": 1000 })),
        _ => failing(500),
    });
    let mut screen = view_of(&site, "the-key", vec![gate(5)]);
    screen.until("the look to be over", |s| s.has("1,000") && gates(s) == [priced(5, Price::Unknown, None)]);
    let paid = screen.text("Paid").0;
    screen.rest(paid.center());
    assert!(screen.all().iter().any(|text| text.contains("The price couldn't be read")), "{:?}", screen.all());

    // It's asked for again with the next look (and only then).
    screen.a_while();
    assert_eq!(site.calls(), ["buzz.getBuzzAccount", "common.getEntityAccess", "modelVersion.getById"]);
    screen.away();
    screen.leave_and_come_back();
    screen.until("the next look", |_| site.asked().len() == 5);
    assert_eq!(gates(&screen), [priced(5, Price::Unknown, None)]);
    assert!(screen.has("Paid"), "all the while");
}

#[test]
fn a_look_that_is_dropped_leaves_no_card_waiting() {
    let hold = Arc::new(AtomicBool::new(true));
    let held = hold.clone();
    let site = Site::answering(move |call| {
        if call.call != "buzz.getBuzzAccount" {
            hold_while(&held);
        }
        ok(json!({ "yellow": 1000 }))
    });
    let mut screen = view_of(&site, "the-key", vec![gate(5)]);
    screen.until("the look to start", |s| s.has("1,000") && site.asked().len() == 2);
    assert_eq!((gates(&screen), priced_cards(&screen)), (vec![priced(5, Price::Reading, None)], 0));

    // The key goes: there's nothing to read the price with, and the card is
    // as it is without one.
    screen.state.api_key.clear();
    screen.frame(vec![]);
    screen.frame(vec![]);
    assert_eq!((gates(&screen), screen.has("Paid")), (vec![gate(5)], true));
    hold.store(false, Ordering::SeqCst);
    screen.a_while();
    assert_eq!((gates(&screen), screen.has("Paid")), (vec![gate(5)], true));
    assert_eq!(site.asked().len(), 2);
}

#[test]
fn coming_back_from_paying_takes_the_price_off_the_card() {
    let paid = Arc::new(AtomicBool::new(false));
    let now_paid = paid.clone();
    let site = shop(move |_| vec![(5, now_paid.load(Ordering::SeqCst)), (7, false), (9, true)]);
    let mut screen = view_of(&site, "the-key", vec![gate(5), gate(7), gate(9)]);
    // (The view's first sight of a key is of a new one: it reads everything.)
    screen.until("the balance and the prices", |s| s.has("1,000") && s.has("500") && s.has("700") && !s.has("Paid"));
    assert!(!screen.has("900"), "what the account has shows no price");
    let at_first = site.asked().len();
    assert_eq!(at_first, 5, "{:?}", site.calls());

    paid.store(true, Ordering::SeqCst);
    screen.leave_and_come_back();
    screen.until("the price to go", |s| !s.has("500"));
    assert!(screen.has("700"), "the other still asks");
    assert_eq!(gates(&screen), [priced(5, buzz(500), Some(true)), priced(7, buzz(700), Some(false)), priced(9, buzz(900), Some(true))]);

    // What was asked on coming back: about the two that asked, and no
    // price again. (The balance was only just read.)
    screen.a_while();
    let asked = site.asked();
    assert_eq!(asked.len(), at_first + 1, "{:?}", site.calls());
    assert_eq!((asked[at_first].call.as_str(), &asked[at_first].input["json"]["entityId"]), ("common.getEntityAccess", &json!([5, 7])));
}

#[test]
fn a_click_on_the_balance_reads_it_and_the_resources_again() {
    let site = shop(|_| vec![(5, false)]);
    // (The key as it may be saved: with a blank or two around it.)
    let mut screen = view_of(&site, " the-key\n", vec![priced(5, buzz(500), Some(false))]);
    screen.until("the balance and the first look", |s| s.has("1,000") && site.asked().len() == 2);
    screen.a_while();
    assert_eq!(site.calls(), ["buzz.getBuzzAccount", "common.getEntityAccess"]);

    screen.click(screen.text("1,000").0.center());
    screen.until("both asked again", |_| site.asked().len() == 4);
    screen.a_while();
    assert_eq!(site.calls(), ["buzz.getBuzzAccount", "buzz.getBuzzAccount", "common.getEntityAccess", "common.getEntityAccess"]);
    assert!(site.asked().iter().all(|a| a.key == "the-key"), "{:?}", site.asked());
    assert!(screen.has("1,000") && screen.has("500"));
}

#[test]
fn another_keys_account_is_looked_at_from_scratch() {
    let site = shop(|call| vec![(5, call.key == "the-buyer")]);
    let mut screen = view_of(&site, "the-buyer", vec![priced(5, buzz(500), Some(true))]);
    screen.until("the balance and the first look", |s| s.has("1,000") && gates(s) == [priced(5, buzz(500), Some(true))]);
    assert!(!screen.has("500"));

    // Another key: what the last account had bought isn't taken as bought.
    screen.state.api_key = "someone-else".to_string();
    screen.frame(vec![]);
    assert!(screen.has("500"), "at once");
    screen.until("the look with the other key", |s| gates(s) == [priced(5, buzz(500), Some(false))]);
    assert!(screen.has("500"));

    // And with no key, nothing is known to be bought — nor asked.
    screen.state.api_key = "the-buyer".to_string();
    screen.until("the price to go", |s| !s.has("500") && s.has("1,000"));
    let asked = site.asked().len();
    screen.state.api_key.clear();
    screen.frame(vec![]);
    screen.frame(vec![]);
    assert!(screen.has("500") && screen.has("Add your API key to see your Buzz"), "{:?}", screen.all());
    assert_eq!(gates(&screen), [priced(5, buzz(500), None)]);
    screen.a_while();
    assert_eq!(site.asked().len(), asked);
}
