//! Link end to end over loopback: a host and a receiver in one process, with
//! a stand-in for the app's models. Covers pairing with the code, wrong codes
//! closing pairing, reconnecting without a code, calls that stream events and
//! files both ways, cancelling, what the host doesn't share, file sharing in
//! both directions, and forgetting a device.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::client::{Client, Status};
use super::crypto::{self, Identity};
use super::files::{Answer, FileEvent, FileHub};
use super::host::{Host, HostConfig, Listening, MAX_TRIES};
use super::peer::{BlobSrc, CallEvent, Incoming, Services};
use super::trust::Trust;
use super::Backend;

/// Stands in for the chat model and generators.
struct Fake {
    dir: PathBuf,
}

impl Backend for Fake {
    fn wake(&self) {}

    fn temp_dir(&self) -> PathBuf {
        self.dir.join("temp")
    }

    fn chat_model(&self) -> Option<String> {
        Some("Test Model".into())
    }

    fn chat_thinks(&self) -> bool {
        true
    }

    fn gen_is_video(&self, body: &Value) -> Option<bool> {
        body.get("video").and_then(Value::as_bool)
    }

    /// Streams the question back word by word; reports the attachments' sizes.
    fn llm_ask(&self, call: &Incoming) -> Result<Value, String> {
        let q = call.body.get("q").and_then(Value::as_str).unwrap_or_default().to_string();
        for w in q.split(' ') {
            call.event(json!({ "kind": "token", "text": w }));
        }
        let sizes: Vec<u64> =
            call.blobs.iter().map(|(_, p)| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
        Ok(json!({ "sizes": sizes }))
    }

    fn gen_catalog(&self) -> Result<Value, String> {
        Ok(json!({ "installed": true }))
    }

    /// Sends back a result file — or, asked to wait, runs until cancelled.
    fn gen_run(&self, call: &Incoming) -> Result<Value, String> {
        if call.body.get("wait").is_some() {
            let started = Instant::now();
            while !call.cancelled() && started.elapsed() < Duration::from_secs(20) {
                std::thread::sleep(Duration::from_millis(20));
            }
            return Err("Stopped".into());
        }
        let _ = std::fs::create_dir_all(&self.dir);
        let out = self.dir.join(format!("result-{}.png", call.id()));
        std::fs::write(&out, pattern(200_000)).map_err(|e| e.to_string())?;
        call.event(json!({ "kind": "status", "text": "Rendering" }));
        call.blob("result.png", &out)?;
        let _ = std::fs::remove_file(&out);
        Ok(json!({ "ok": true }))
    }

    fn gen_thumbs(&self, _call: &Incoming) -> Result<Value, String> {
        Ok(json!({ "none": [] }))
    }

    fn gen_lora_info(&self, _call: &Incoming) -> Result<Value, String> {
        Ok(json!({}))
    }
}

fn pattern(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 253) as u8).collect()
}

/// A fresh folder for one side of one test.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ctf_link_e2e_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn wait_for(what: &str, timeout: Duration, mut ok: impl FnMut() -> bool) {
    let started = Instant::now();
    while !ok() {
        assert!(started.elapsed() < timeout, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// One side: its identity, folder, models and file sharing.
struct Side {
    id: Identity,
    dir: PathBuf,
    backend: Arc<dyn Backend>,
    files: Arc<FileHub>,
}

impl Drop for Side {
    fn drop(&mut self) {
        // After the test's host and receiver (declared later, so dropped
        // first) have wound down their last writes.
        std::thread::sleep(Duration::from_millis(200));
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn side(name: &str) -> Side {
    let dir = scratch(name);
    let files = Arc::new(FileHub::new(Box::new(|| {})));
    files.set_received_dir(dir.join("received"));
    Side { id: Identity::generate(), backend: Arc::new(Fake { dir: dir.join("work") }), dir, files }
}

fn all_shared() -> Services {
    Services { chat: true, images: true, video: true, files: true }
}

fn start_host(h: &Side, port: u16, services: Services, accept_files_unasked: bool) -> Host {
    let host = Host::start(
        h.id.clone(),
        h.dir.clone(),
        HostConfig { name: "HOST".into(), port, services, accept_files_unasked, background: false },
        h.backend.clone(),
        h.files.clone(),
    );
    wait_for("the host to listen", Duration::from_secs(10), || {
        *host.shared.listening.lock().unwrap() == Listening::On
    });
    host
}

fn connect(r: &Side, port: u16, known: Option<&Identity>, code: Option<String>) -> Client {
    Client::start(
        r.id.clone(),
        r.dir.clone(),
        "RECEIVER".into(),
        format!("127.0.0.1:{port}"),
        known.map(|k| crypto::to_hex(&k.public)),
        code,
        r.backend.clone(),
        r.files.clone(),
    )
}

/// Wait until the client settles (connected, or stopped trying).
fn settled(c: &Client) -> Status {
    let mut last = Status::Connecting;
    wait_for("the receiver to settle", Duration::from_secs(15), || {
        last = c.shared.status();
        !matches!(last, Status::Connecting | Status::Retrying { .. })
    });
    last
}

/// Pair `r` with the running host and return the connected client.
fn pair(host: &Host, r: &Side, port: u16) -> Client {
    let code = host.shared.open_pairing();
    let c = connect(r, port, None, Some(code));
    assert_eq!(settled(&c), Status::Connected);
    c
}

#[test]
fn pairing_with_the_code_then_reconnecting_without_it() {
    let (h, r) = (side("pair_h"), side("pair_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);

    // Not paired: the host asks for its code (pairing isn't even open).
    let c = connect(&r, port, None, None);
    assert_eq!(settled(&c), Status::NeedCode { host: "HOST".into() });
    drop(c);

    // With the code it shows: in, and both remember each other.
    let c = pair(&host, &r, port);
    let info = c.shared.info().unwrap();
    assert_eq!(info.name, "HOST");
    assert_eq!(info.chat_model.as_deref(), Some("Test Model"));
    assert!(info.chat_thinks);
    let rx_key = crypto::to_hex(&r.id.public);
    let host_key = crypto::to_hex(&h.id.public);
    assert!(Trust::load(&h.dir).device(&rx_key).is_some());
    assert!(Trust::load(&r.dir).host(&host_key).is_some());
    // One code, one device.
    assert!(host.shared.pairing.lock().unwrap().is_none());
    wait_for("the host to list the session", Duration::from_secs(5), || host.shared.sessions.lock().unwrap().len() == 1);
    c.stop();
    drop(c);

    // Later: straight back in, no code.
    let c = connect(&r, port, Some(&h.id), None);
    assert_eq!(settled(&c), Status::Connected);
    c.stop();
}

#[test]
fn wrong_codes_close_pairing() {
    let (h, r) = (side("wrong_h"), side("wrong_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    let code = host.shared.open_pairing();
    let wrong = if code == "000000" { "000001" } else { "000000" }.to_string();
    for _ in 0..MAX_TRIES {
        let c = connect(&r, port, None, Some(wrong.clone()));
        assert_eq!(settled(&c), Status::Failed("Wrong code".into()));
    }
    // Used up: even the right code doesn't get in now.
    assert!(host.shared.pairing.lock().unwrap().is_none());
    let c = connect(&r, port, None, Some(code));
    match settled(&c) {
        Status::Failed(why) => assert!(why.contains("Pairing isn't open"), "{why}"),
        other => panic!("{other:?}"),
    }
    assert!(Trust::load(&h.dir).devices.is_empty());
    assert!(Trust::load(&r.dir).hosts.is_empty());
}

#[test]
fn a_different_pc_at_a_known_address_is_refused() {
    let (h, r, impostor) = (side("imp_h"), side("imp_r"), side("imp_i"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    pair(&host, &r, port).stop();
    host.stop();
    drop(host);
    // Another PC now answers at that address: not trusted, and the receiver
    // keeps looking for its host rather than giving up.
    let port2 = free_port();
    let other = start_host(&impostor, port2, all_shared(), false);
    let c = connect(&r, port2, Some(&h.id), None);
    wait_for("the receiver to turn it down", Duration::from_secs(15), || {
        matches!(c.shared.status(), Status::Retrying { ref why } if why.contains("isn't the host this PC paired with"))
    });
    assert!(c.shared.peer().is_none());
    assert!(other.shared.sessions.lock().unwrap().is_empty());
    c.stop();
    other.stop();
}

/// Stopping a receiver (Disconnect, Off) while it's still opening leaves no
/// session behind on either side.
#[test]
fn stopping_while_connecting_leaves_nothing() {
    let (h, r) = (side("stopc_h"), side("stopc_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    pair(&host, &r, port).stop();
    wait_for("the first session to end", Duration::from_secs(5), || host.shared.sessions.lock().unwrap().is_empty());
    for _ in 0..20 {
        let c = connect(&r, port, Some(&h.id), None);
        c.stop();
    }
    std::thread::sleep(Duration::from_millis(1500));
    assert!(host.shared.sessions.lock().unwrap().is_empty(), "a session outlived its receiver");
}

/// Turning Link off while a device is still opening keeps it out.
#[test]
fn a_device_opening_while_link_stops_stays_out() {
    let (h, r) = (side("offo_h"), side("offo_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    pair(&host, &r, port).stop();
    wait_for("the first session to end", Duration::from_secs(5), || host.shared.sessions.lock().unwrap().is_empty());
    let shared = host.shared.clone();
    let c = connect(&r, port, Some(&h.id), None);
    host.stop();
    std::thread::sleep(Duration::from_millis(1500));
    assert!(shared.sessions.lock().unwrap().is_empty());
    assert!(c.shared.peer().is_none());
    c.stop();
}

#[test]
fn calls_stream_events_and_files_both_ways() {
    let (h, r) = (side("calls_h"), side("calls_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    let c = pair(&host, &r, port);
    let peer = c.shared.peer().unwrap();
    let blob_dir = r.dir.join("incoming");

    // A question with an attachment: tokens stream back, the file arrived whole.
    let attachment = pattern(300_000);
    let mut call = peer
        .call(
            "llm.ask",
            json!({ "q": "one two three" }),
            vec![BlobSrc::Bytes { name: "a.png".into(), data: attachment }],
            &blob_dir,
        )
        .unwrap();
    let mut tokens = Vec::new();
    let done = super::finish_call(&mut call, &|| false, &mut |ev| {
        if let CallEvent::Event(v) = ev {
            tokens.push(v["text"].as_str().unwrap_or_default().to_string());
        }
    })
    .unwrap();
    assert_eq!(tokens, ["one", "two", "three"]);
    assert_eq!(done["sizes"], json!([300_000]));

    // A generation: its result comes back as a file.
    let mut call = peer.call("gen.run", json!({ "video": false }), Vec::new(), &blob_dir).unwrap();
    let mut got: Option<PathBuf> = None;
    let done = super::finish_call(&mut call, &|| false, &mut |ev| {
        if let CallEvent::Blob { name, path } = ev {
            assert_eq!(name, "result.png");
            got = Some(path);
        }
    })
    .unwrap();
    assert_eq!(done["ok"], true);
    let got = got.expect("a result file");
    assert_eq!(std::fs::read(&got).unwrap(), pattern(200_000));
    let _ = std::fs::remove_file(got);

    // The host's info, on request.
    let mut call = peer.call("info", json!({}), Vec::new(), &blob_dir).unwrap();
    let info = super::finish_call(&mut call, &|| false, &mut |_| {}).unwrap();
    assert_eq!(info["name"], "HOST");

    // Unknown services are turned down.
    let mut call = peer.call("shell.run", json!({}), Vec::new(), &blob_dir).unwrap();
    assert!(super::finish_call(&mut call, &|| false, &mut |_| {}).is_err());

    // Stopping a running job: the host sees it and the call ends.
    let mut call = peer.call("gen.run", json!({ "video": true, "wait": true }), Vec::new(), &blob_dir).unwrap();
    let started = Instant::now();
    let stop = || started.elapsed() > Duration::from_millis(300);
    assert!(super::finish_call(&mut call, &stop, &mut |_| {}).is_err());
    assert!(started.elapsed() < Duration::from_secs(10));
    c.stop();
}

#[test]
fn what_the_host_doesnt_share_is_refused() {
    let (h, r) = (side("share_h"), side("share_r"));
    let port = free_port();
    let host = start_host(&h, port, Services { chat: false, images: true, video: false, files: false }, false);
    let c = pair(&host, &r, port);
    let peer = c.shared.peer().unwrap();
    let dir = r.dir.join("incoming");
    let fail = |service: &str, body: Value| {
        let mut call = peer.call(service, body, Vec::new(), &dir).unwrap();
        super::finish_call(&mut call, &|| false, &mut |_| {}).unwrap_err()
    };
    assert!(fail("llm.ask", json!({ "q": "hi" })).contains("doesn't share its AI chat"));
    assert!(fail("gen.run", json!({ "video": true })).contains("doesn't share video generation"));
    assert!(fail("files.offer", json!({ "files": [] })).contains("doesn't share files"));
    // Images are shared.
    let mut call = peer.call("gen.run", json!({ "video": false }), Vec::new(), &dir).unwrap();
    assert!(super::finish_call(&mut call, &|| false, &mut |ev| {
        if let CallEvent::Blob { path, .. } = ev {
            let _ = std::fs::remove_file(path);
        }
    })
    .is_ok());

    // Turning a share off reaches the receiver at once.
    host.shared.config.lock().unwrap().services.images = false;
    host.shared.announce();
    wait_for("the new offer", Duration::from_secs(5), || c.shared.info().is_some_and(|i| !i.services.images));
    assert!(fail("gen.run", json!({ "video": false })).contains("doesn't share image generation"));
    c.stop();
}

fn write_media(dir: &Path, name: &str, tags: Option<&str>) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join(name);
    std::fs::write(&p, pattern(150_000)).unwrap();
    if let Some(t) = tags {
        std::fs::write(p.with_extension("txt"), t).unwrap();
    }
    p
}

#[test]
fn files_go_both_ways_with_their_tags() {
    let (h, r) = (side("files_h"), side("files_r"));
    let port = free_port();
    // The host takes files from paired devices without asking.
    let host = start_host(&h, port, all_shared(), true);
    let c = pair(&host, &r, port);
    let peer = c.shared.peer().unwrap();

    // Receiver → host.
    let img = write_media(&r.dir.join("pics"), "cat.png", Some("cat, orange"));
    r.files.send(&peer, &[img], &r.dir.join("temp"));
    let events = r.files.take_events();
    assert!(matches!(events.as_slice(), [FileEvent::Sent { count: 1, .. }]), "{events:?}");
    let got = h.dir.join("received");
    assert_eq!(std::fs::read(got.join("cat.png")).unwrap(), pattern(150_000));
    assert_eq!(std::fs::read_to_string(got.join("cat.txt")).unwrap(), "cat, orange");
    // Sent again: saved beside the first, never over it — the tags too.
    let img = write_media(&r.dir.join("pics"), "cat.png", Some("second"));
    r.files.send(&peer, &[img], &r.dir.join("temp"));
    assert_eq!(std::fs::read_to_string(got.join("cat_2.txt")).unwrap(), "second");
    assert_eq!(std::fs::read_to_string(got.join("cat.txt")).unwrap(), "cat, orange");

    // The host heard about both arrivals.
    let arrived = h.files.take_events();
    assert!(matches!(arrived.as_slice(), [FileEvent::Received { count: 1, .. }, FileEvent::Received { count: 1, .. }]), "{arrived:?}");

    // Host → receiver: its owner is asked, and ticks "always".
    wait_for("the host's session", Duration::from_secs(5), || !host.shared.sessions.lock().unwrap().is_empty());
    let back = host.shared.sessions.lock().unwrap()[0].clone();
    let vid = write_media(&h.dir.join("vids"), "clip.mp4", None);
    let hub = h.files.clone();
    let temp = h.dir.join("temp");
    let sender = std::thread::spawn(move || {
        hub.send(&back, &[vid], &temp);
        hub.take_events()
    });
    let mut offer = None;
    wait_for("the offer", Duration::from_secs(10), || {
        offer = r.files.pending().into_iter().next();
        offer.is_some()
    });
    let offer = offer.unwrap();
    assert_eq!(offer.from, "HOST");
    assert_eq!(offer.names, ["clip.mp4"]);
    r.files.answer(offer.id, Answer::Accept { always: true });
    let events = sender.join().unwrap();
    assert!(matches!(events.as_slice(), [FileEvent::Sent { count: 1, .. }]), "{events:?}");
    assert!(r.dir.join("received").join("clip.mp4").is_file());
    let host_key = crypto::to_hex(&h.id.public);
    assert!(Trust::load(&r.dir).host(&host_key).unwrap().accept_files);

    // A declined offer sends nothing.
    let back = host.shared.sessions.lock().unwrap()[0].clone();
    Trust::update(&r.dir, |t| t.hosts[0].accept_files = false).unwrap();
    let vid = write_media(&h.dir.join("vids"), "other.mp4", None);
    let hub = h.files.clone();
    let temp = h.dir.join("temp");
    let sender = std::thread::spawn(move || {
        hub.send(&back, &[vid], &temp);
        hub.take_events()
    });
    let mut offer = None;
    wait_for("the second offer", Duration::from_secs(10), || {
        offer = r.files.pending().into_iter().next();
        offer.is_some()
    });
    r.files.answer(offer.unwrap().id, Answer::Decline);
    let events = sender.join().unwrap();
    assert!(matches!(events.as_slice(), [FileEvent::NotSent { .. }]), "{events:?}");
    assert!(!r.dir.join("received").join("other.mp4").exists());
    c.stop();
}

#[test]
fn files_need_an_accepted_offer() {
    let (h, r) = (side("token_h"), side("token_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    let c = pair(&host, &r, port);
    let peer = c.shared.peer().unwrap();
    // Straight to sending, with a made-up token: turned down, nothing saved.
    let mut call = peer
        .call(
            "files.send",
            json!({ "token": "00" }),
            vec![BlobSrc::Bytes { name: "x.png".into(), data: pattern(1000) }],
            &r.dir.join("incoming"),
        )
        .unwrap();
    let err = super::finish_call(&mut call, &|| false, &mut |_| {}).unwrap_err();
    assert!(err.contains("weren't accepted"), "{err}");
    assert!(!h.dir.join("received").join("x.png").exists());
    c.stop();
}

#[test]
fn a_forgotten_device_has_to_pair_again() {
    let (h, r) = (side("forget_h"), side("forget_r"));
    let port = free_port();
    let host = start_host(&h, port, all_shared(), false);
    let c = pair(&host, &r, port);
    wait_for("the host's session", Duration::from_secs(5), || !host.shared.sessions.lock().unwrap().is_empty());
    // The host's owner forgets it: disconnected, and its reconnect needs a code.
    let key = crypto::to_hex(&r.id.public);
    Trust::update(&h.dir, |t| t.devices.retain(|d| d.key != key)).unwrap();
    host.shared.drop_device(&r.id.public);
    wait_for("the receiver to need the code", Duration::from_secs(20), || {
        matches!(c.shared.status(), Status::NeedCode { .. })
    });
    c.stop();
}

/// The real background host: the app started with `--link-background` in a
/// throwaway data folder hosts a paired device, says it's in the background,
/// and stops when asked the way the app asks. Needs the built program:
/// `CTF_APP_BIN=target/debug/Clarity_TagFlow cargo test -- --ignored background_host`.
#[test]
#[ignore]
fn background_host_serves_and_hands_over() {
    let Ok(bin) = std::env::var("CTF_APP_BIN") else { return };
    // The throwaway data folder is set through XDG_DATA_HOME / HOME; Windows
    // gives the program its real %APPDATA% whatever the environment says.
    if cfg!(windows) {
        eprintln!("skipped: the data folder can't be redirected on Windows");
        return;
    }
    let data = scratch("bg_data");
    let link = data.join("Clarity TagFlow").join("tools").join("link");
    std::fs::create_dir_all(&link).unwrap();
    let port = free_port();
    // What the app would have left: Link set to Host with background hosting,
    // this host's identity, and one paired device.
    let host_id = Identity::load_or_create(&link).unwrap();
    let r = side("bg_r");
    Trust::update(&link, |t| t.add_device(&crypto::to_hex(&r.id.public), "RECEIVER")).unwrap();
    Trust::update(&r.dir, |t| t.add_host(&crypto::to_hex(&host_id.public), "HOST", &format!("127.0.0.1:{port}")))
        .unwrap();
    let mut s = super::LinkSettings { mode: super::LinkMode::Host, port, background: true, ..Default::default() };
    s.share_files = false;
    let cfg = super::background::BackgroundConfig { link: s, ..Default::default() };
    std::fs::write(link.join("background-config.json"), serde_json::to_string(&cfg).unwrap()).unwrap();

    let mut child = std::process::Command::new(&bin)
        .arg(super::background::FLAG)
        .env("XDG_DATA_HOME", &data)
        .env("HOME", &data)
        .spawn()
        .unwrap();
    wait_for("the background host's state file", Duration::from_secs(20), || link.join("background.json").is_file());

    let c = connect(&r, port, Some(&host_id), None);
    assert_eq!(settled(&c), Status::Connected);
    let info = c.shared.info().unwrap();
    assert!(info.background);
    assert!(!info.services.files);
    // No model is set up there: a question fails, readably.
    let peer = c.shared.peer().unwrap();
    let mut call = peer.call("llm.ask", json!({ "msgs": [{ "user": true, "text": "hi" }] }), Vec::new(), &r.dir).unwrap();
    let err = super::finish_call(&mut call, &|| false, &mut |_| {}).unwrap_err();
    assert!(err.contains("AI") || err.contains("set up"), "{err}");

    // The app opening asks it to stop: it frees the port, then exits.
    let state: Value = serde_json::from_str(&std::fs::read_to_string(link.join("background.json")).unwrap()).unwrap();
    let control = std::net::SocketAddr::from(([127, 0, 0, 1], state["port"].as_u64().unwrap() as u16));
    let mut ctl = std::net::TcpStream::connect(control).unwrap();
    use std::io::{BufRead, Write};
    writeln!(ctl, "{} quit", state["token"].as_str().unwrap()).unwrap();
    let mut reply = String::new();
    std::io::BufReader::new(ctl).read_line(&mut reply).unwrap();
    assert_eq!(reply.trim(), "ok");
    let started = Instant::now();
    let code = loop {
        if let Some(code) = child.try_wait().unwrap() {
            break code;
        }
        assert!(started.elapsed() < Duration::from_secs(20), "the background host didn't exit");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(code.success());
    assert!(!link.join("background.json").exists());
    // And the port is free for the app.
    assert!(TcpListener::bind(("0.0.0.0", port)).is_ok());
    c.stop();

    // Link turned off: the background host won't start at all.
    let cfg = super::background::BackgroundConfig::default();
    std::fs::write(link.join("background-config.json"), serde_json::to_string(&cfg).unwrap()).unwrap();
    let status = std::process::Command::new(&bin)
        .arg(super::background::FLAG)
        .env("XDG_DATA_HOME", &data)
        .env("HOME", &data)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(!link.join("background.json").exists());
    let _ = std::fs::remove_dir_all(&data);
}
