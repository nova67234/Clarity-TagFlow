//! Finding hosts on the local network: a receiver broadcasts a short UDP
//! probe; hosts answer with their name, port and key fingerprint. Saves
//! typing an address — and finds a remembered host again after its address
//! changed (by its fingerprint). Answers go only to local addresses.

use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::addr;
use super::host::HostShared;

const PROBE: &[u8] = b"CTF-LINK?1";

/// A host that answered.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct Found {
    pub name: String,
    /// Its identity fingerprint (see `crypto::fingerprint`).
    pub fp: String,
    pub port: u16,
    /// Filled in by the receiver: the address the answer came from.
    #[serde(default)]
    pub ip: String,
}

impl Found {
    pub fn addr(&self) -> String {
        format!("{}:{}", self.ip, self.port)
    }
}

/// Answer probes while the host runs.
pub(crate) fn respond(shared: Arc<HostShared>) {
    let port = shared.config.lock().unwrap_or_else(|e| e.into_inner()).port;
    let sock = loop {
        if shared.stop.load(Ordering::SeqCst) {
            return;
        }
        match UdpSocket::bind(("0.0.0.0", port)) {
            Ok(s) => break s,
            // Still held by the copy this one takes over from.
            Err(_) => std::thread::sleep(Duration::from_millis(500)),
        }
    };
    let _ = sock.set_read_timeout(Some(Duration::from_millis(500)));
    let fp = super::crypto::fingerprint(&shared.identity.public);
    let mut buf = [0u8; 64];
    while !shared.stop.load(Ordering::SeqCst) {
        let Ok((n, from)) = sock.recv_from(&mut buf) else { continue };
        if &buf[..n] != PROBE || !addr::is_local(from.ip()) {
            continue;
        }
        let name = shared.config.lock().unwrap_or_else(|e| e.into_inner()).name.clone();
        let reply = Found { name, fp: fp.clone(), port, ip: String::new() };
        if let Ok(bytes) = serde_json::to_vec(&reply) {
            let _ = sock.send_to(&bytes, from);
        }
    }
}

/// Look for hosts on `port` for `wait` and return the ones that answered.
pub(crate) fn find(port: u16, wait: Duration) -> Vec<Found> {
    let Ok(sock) = UdpSocket::bind(("0.0.0.0", 0)) else { return Vec::new() };
    let _ = sock.set_broadcast(true);
    let _ = sock.set_read_timeout(Some(Duration::from_millis(150)));
    let mut targets: Vec<SocketAddr> = vec![
        SocketAddr::from((Ipv4Addr::BROADCAST, port)),
        // A host on this same PC (handy for trying Link out).
        SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
    ];
    for (_, broadcast) in addr::local_addresses() {
        targets.push(SocketAddr::from((broadcast, port)));
    }
    let send_all = |sock: &UdpSocket| {
        for t in &targets {
            let _ = sock.send_to(PROBE, t);
        }
    };
    send_all(&sock);
    let started = Instant::now();
    let mut resent = false;
    let mut out: Vec<Found> = Vec::new();
    let mut buf = [0u8; 512];
    while started.elapsed() < wait {
        // One more round halfway, for a probe lost on busy Wi-Fi.
        if !resent && started.elapsed() > wait / 2 {
            resent = true;
            send_all(&sock);
        }
        let Ok((n, from)) = sock.recv_from(&mut buf) else { continue };
        if !addr::is_local(from.ip()) {
            continue;
        }
        let Ok(mut f) = serde_json::from_slice::<Found>(&buf[..n]) else { continue };
        f.name = super::host::clean_name(&f.name);
        f.ip = from.ip().to_string();
        // This PC's own host heard on loopback and on the LAN: one entry,
        // the LAN address (another PC could use it too). Other answers are
        // all kept — anyone can claim a fingerprint, so one answering first
        // mustn't hide the real host.
        let loopback = |o: &Found| o.ip == "127.0.0.1";
        match out.iter_mut().find(|o| o.fp == f.fp && o.port == f.port && (o.ip == f.ip || loopback(o) || loopback(&f))) {
            Some(o) if loopback(o) => *o = f,
            Some(_) => {}
            None => out.push(f),
        }
    }
    out.sort_by_key(|f| f.name.to_lowercase());
    out
}
