//! The encrypted connection: the Noise handshake, then frames.
//!
//! On the TCP stream every message is a 2-byte big-endian length and that many
//! bytes. The handshake's three messages go first; after it, each message is
//! one Noise transport message whose plaintext is a frame:
//!
//! ```text
//! kind (1 byte) · call id (4 bytes, big-endian) · body
//! ```
//!
//! * kind 0 — a JSON message, complete or the last piece of one
//! * kind 2 — a piece of a longer JSON message, more pieces follow
//! * kind 1 — raw bytes of the file the call is streaming
//!
//! Call 0 is the control channel (pings, cancels, info updates — always one
//! frame each). A receiver numbers the calls it starts odd, a host even, so
//! either side can start calls without the numbers ever clashing.
//!
//! Each direction counts its own nonces from 0; a frame that's replayed,
//! dropped or reordered fails to decrypt and ends the session.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

use super::crypto::{noise_params, Identity, PROLOGUE};

/// Largest Noise message.
const MAX_MSG: usize = 65535;
/// Noise's authentication tag on every transport message.
const TAG: usize = 16;
/// Frame header: kind + call id.
const HEADER: usize = 5;
/// Largest frame body that fits one Noise message.
pub(crate) const MAX_BODY: usize = MAX_MSG - TAG - HEADER;
/// Largest JSON message accepted (pieced together from frames).
pub(crate) const MAX_JSON: usize = 16 * 1024 * 1024;
/// All the JSON one session may have half pieced together at once (over
/// every call), and in how many pieces-in-progress.
const MAX_JSON_PENDING: usize = 32 * 1024 * 1024;
const MAX_JSON_PARTS: usize = 64;

pub(crate) const KIND_JSON: u8 = 0;
pub(crate) const KIND_DATA: u8 = 1;
pub(crate) const KIND_JSON_MORE: u8 = 2;

fn write_raw(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    let len = u16::try_from(bytes.len()).map_err(|_| std::io::Error::other("message too long"))?;
    let mut out = Vec::with_capacity(2 + bytes.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    stream.write_all(&out)
}

fn read_raw(stream: &mut TcpStream, buf: &mut Vec<u8>) -> std::io::Result<()> {
    let mut len = [0u8; 2];
    stream.read_exact(&mut len)?;
    let len = u16::from_be_bytes(len) as usize;
    buf.resize(len, 0);
    stream.read_exact(buf)
}

/// What a finished handshake knows about the other side.
pub(crate) struct Session {
    pub reader: Reader,
    pub writer: Arc<Writer>,
    /// The other side's identity key.
    pub remote_key: Vec<u8>,
    /// The handshake hash — unique to this session (pairing proofs bind to it).
    pub hash: Vec<u8>,
}

/// Run the Noise XX handshake as the side that connected (the receiver, or a
/// host sending files — always the TCP client). The stream should have a read
/// timeout set so a silent peer can't hang it.
pub(crate) fn handshake_initiator(mut stream: TcpStream, id: &Identity) -> Result<Session, String> {
    let mut hs = snow::Builder::new(noise_params())
        .local_private_key(&id.private)
        .and_then(|b| b.prologue(PROLOGUE))
        .and_then(|b| b.build_initiator())
        .map_err(|e| format!("handshake setup: {e}"))?;
    let mut buf = vec![0u8; MAX_MSG];
    let mut msg = Vec::new();
    // -> e
    let n = hs.write_message(&[], &mut buf).map_err(|e| format!("handshake: {e}"))?;
    write_raw(&mut stream, &buf[..n]).map_err(io_err)?;
    // <- e, ee, s, es
    read_raw(&mut stream, &mut msg).map_err(io_err)?;
    hs.read_message(&msg, &mut buf).map_err(|_| "The other side isn't Clarity TagFlow Link (or a different version)".to_string())?;
    // -> s, se
    let n = hs.write_message(&[], &mut buf).map_err(|e| format!("handshake: {e}"))?;
    write_raw(&mut stream, &buf[..n]).map_err(io_err)?;
    finish(stream, hs)
}

/// The handshake as the side that was connected to (the host).
pub(crate) fn handshake_responder(mut stream: TcpStream, id: &Identity) -> Result<Session, String> {
    let mut hs = snow::Builder::new(noise_params())
        .local_private_key(&id.private)
        .and_then(|b| b.prologue(PROLOGUE))
        .and_then(|b| b.build_responder())
        .map_err(|e| format!("handshake setup: {e}"))?;
    let mut buf = vec![0u8; MAX_MSG];
    let mut msg = Vec::new();
    // -> e
    read_raw(&mut stream, &mut msg).map_err(io_err)?;
    hs.read_message(&msg, &mut buf).map_err(|_| "not a Link handshake".to_string())?;
    // <- e, ee, s, es
    let n = hs.write_message(&[], &mut buf).map_err(|e| format!("handshake: {e}"))?;
    write_raw(&mut stream, &buf[..n]).map_err(io_err)?;
    // -> s, se
    read_raw(&mut stream, &mut msg).map_err(io_err)?;
    hs.read_message(&msg, &mut buf).map_err(|_| "handshake failed".to_string())?;
    finish(stream, hs)
}

fn finish(stream: TcpStream, hs: snow::HandshakeState) -> Result<Session, String> {
    let remote_key = hs.get_remote_static().ok_or("no remote key")?.to_vec();
    let hash = hs.get_handshake_hash().to_vec();
    let transport = Arc::new(hs.into_stateless_transport_mode().map_err(|e| format!("handshake: {e}"))?);
    let write_stream = stream.try_clone().map_err(io_err)?;
    let control = stream.try_clone().map_err(io_err)?;
    // A frame that can't go out within this (the other side stopped
    // reading) ends the session rather than blocking every sender.
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
    Ok(Session {
        reader: Reader {
            stream,
            transport: transport.clone(),
            nonce: 0,
            raw: vec![0u8; 2 + MAX_MSG],
            have: 0,
            plain: vec![0u8; MAX_MSG],
        },
        writer: Arc::new(Writer {
            inner: Mutex::new(WriterInner { stream: write_stream, nonce: 0, buf: vec![0u8; MAX_MSG] }),
            transport,
            control,
        }),
        remote_key,
        hash,
    })
}

/// How long one frame may take to go out.
const WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Ends a connection whose opening (the handshake, then the hello and any
/// pairing) isn't over within its time limit. A read timeout applies to each
/// read, so a peer sending a byte now and then could otherwise hold the
/// connection — and one of the host's few opening slots — for ever. Dropping
/// it (the opening is over) stops the watch.
pub(crate) struct OpeningDeadline {
    _over: std::sync::mpsc::Sender<()>,
}

impl OpeningDeadline {
    pub fn start(stream: &TcpStream, limit: std::time::Duration) -> OpeningDeadline {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        if let Ok(guard) = stream.try_clone() {
            let _ = std::thread::Builder::new().name("link-opening".into()).spawn(move || {
                if rx.recv_timeout(limit) == Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
                    let _ = guard.shutdown(std::net::Shutdown::Both);
                }
            });
        }
        OpeningDeadline { _over: tx }
    }
}

fn io_err(e: std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => "The other PC stopped answering".to_string(),
        std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset => {
            "The connection was closed".to_string()
        }
        _ => e.to_string(),
    }
}

/// The receiving half of a session.
pub(crate) struct Reader {
    stream: TcpStream,
    transport: Arc<snow::StatelessTransportState>,
    nonce: u64,
    /// The message being read (its 2-byte length first) and how much of it
    /// is in: a read timeout part-way keeps what arrived.
    raw: Vec<u8>,
    have: usize,
    plain: Vec<u8>,
}

/// One received frame.
pub(crate) struct Frame {
    pub kind: u8,
    pub call: u32,
    pub body: Vec<u8>,
}

impl Reader {
    pub fn set_timeout(&self, t: Option<std::time::Duration>) {
        let _ = self.stream.set_read_timeout(t);
    }

    /// Read until `want` bytes of the current message are in. A timeout
    /// returns the error but keeps what arrived for the next call.
    ///
    /// It waits with a peek, which takes nothing: on Windows a read that
    /// times out may lose bytes that were arriving, a peek can't. With bytes
    /// waiting, the read then returns at once.
    fn fill(&mut self, want: usize) -> std::io::Result<()> {
        while self.have < want {
            match self.stream.peek(&mut [0u8; 1]) {
                Ok(0) => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
            match self.stream.read(&mut self.raw[self.have..want]) {
                Ok(0) => return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
                Ok(n) => self.have += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// The next frame. A timeout comes back as `ErrorKind::WouldBlock` /
    /// `TimedOut` (call again — a message that was half in carries on);
    /// anything that doesn't decrypt is an error (the session is over).
    pub fn next(&mut self) -> std::io::Result<Frame> {
        self.fill(2)?;
        let len = u16::from_be_bytes([self.raw[0], self.raw[1]]) as usize;
        self.fill(2 + len)?;
        self.have = 0;
        let n = self
            .transport
            .read_message(self.nonce, &self.raw[2..2 + len], &mut self.plain)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "a message failed to decrypt"))?;
        self.nonce += 1;
        if n < HEADER {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "short frame"));
        }
        let call = u32::from_be_bytes([self.plain[1], self.plain[2], self.plain[3], self.plain[4]]);
        Ok(Frame { kind: self.plain[0], call, body: self.plain[HEADER..n].to_vec() })
    }
}

struct WriterInner {
    stream: TcpStream,
    nonce: u64,
    buf: Vec<u8>,
}

/// The sending half of a session, shared by every thread that sends on it.
/// Each frame goes out whole under the lock, so frames from different calls
/// interleave but never mix.
pub(crate) struct Writer {
    inner: Mutex<WriterInner>,
    transport: Arc<snow::StatelessTransportState>,
    /// The same socket, for ending the session without waiting for the lock
    /// (a send stuck on a full connection holds it).
    control: TcpStream,
}

impl Writer {
    /// Send one frame (`body` ≤ [`MAX_BODY`]).
    pub fn frame(&self, kind: u8, call: u32, body: &[u8]) -> std::io::Result<()> {
        debug_assert!(body.len() <= MAX_BODY);
        let mut plain = Vec::with_capacity(HEADER + body.len());
        plain.push(kind);
        plain.extend_from_slice(&call.to_be_bytes());
        plain.extend_from_slice(body);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let WriterInner { stream, nonce, buf } = &mut *inner;
        let n = self
            .transport
            .write_message(*nonce, &plain, buf)
            .map_err(|e| std::io::Error::other(format!("encrypt: {e}")))?;
        *nonce += 1;
        let sent = write_raw(stream, &buf[..n]);
        if sent.is_err() {
            // Part of a frame may have gone out: nothing after it would
            // decrypt, so the session ends here.
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
        sent
    }

    /// Send a JSON message on `call`, split into pieces when it's long.
    pub fn json<T: serde::Serialize>(&self, call: u32, msg: &T) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(msg).map_err(std::io::Error::other)?;
        if bytes.len() > MAX_JSON {
            return Err(std::io::Error::other("message too long"));
        }
        let mut chunks = bytes.chunks(MAX_BODY).peekable();
        if chunks.peek().is_none() {
            return self.frame(KIND_JSON, call, &[]);
        }
        while let Some(piece) = chunks.next() {
            let kind = if chunks.peek().is_some() { KIND_JSON_MORE } else { KIND_JSON };
            self.frame(kind, call, piece)?;
        }
        Ok(())
    }

    /// Stream `len` bytes from `src` as data frames on `call`. Stops early
    /// (with an error) when `stop` says so — the receiving side then sees the
    /// call fail.
    pub fn data(&self, call: u32, src: &mut dyn Read, len: u64, stop: &dyn Fn() -> bool) -> std::io::Result<()> {
        let mut left = len;
        let mut buf = vec![0u8; MAX_BODY];
        while left > 0 {
            if stop() {
                return Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "stopped"));
            }
            let want = (left as usize).min(MAX_BODY);
            src.read_exact(&mut buf[..want])?;
            self.frame(KIND_DATA, call, &buf[..want])?;
            left -= want as u64;
        }
        Ok(())
    }

    pub fn shutdown(&self) {
        let _ = self.control.shutdown(std::net::Shutdown::Both);
    }
}

/// Pieces a JSON message back together from its frames.
#[derive(Default)]
pub(crate) struct JsonAssembler {
    pending: std::collections::HashMap<u32, Vec<u8>>,
    /// Bytes held in `pending`.
    held: usize,
}

impl JsonAssembler {
    /// Feed a JSON frame; the whole message once its last piece is in. Too
    /// much at once (one message, or many half-sent ones) is an error: the
    /// session ends rather than the memory running out.
    pub fn push(&mut self, frame: &Frame) -> std::io::Result<Option<Vec<u8>>> {
        let too_much = || std::io::Error::new(std::io::ErrorKind::InvalidData, "message too long");
        if !self.pending.contains_key(&frame.call) {
            if frame.kind != KIND_JSON_MORE {
                // The usual case: a message in one frame.
                return Ok(Some(frame.body.clone()));
            }
            if self.pending.len() >= MAX_JSON_PARTS {
                return Err(too_much());
            }
        }
        let buf = self.pending.entry(frame.call).or_default();
        if buf.len() + frame.body.len() > MAX_JSON || self.held + frame.body.len() > MAX_JSON_PENDING {
            return Err(too_much());
        }
        buf.extend_from_slice(&frame.body);
        self.held += frame.body.len();
        if frame.kind == KIND_JSON_MORE {
            return Ok(None);
        }
        let whole = self.pending.remove(&frame.call).unwrap_or_default();
        self.held -= whole.len();
        Ok(Some(whole))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A session over loopback: both sides learn each other's keys and the
    /// same hash, and frames — short, long and data — arrive intact.
    #[test]
    fn handshake_and_frames() {
        let host_id = Identity::generate();
        let rx_id = Identity::generate();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let host_pub = host_id.public.clone();
        let rx_pub = rx_id.public.clone();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut s = handshake_responder(stream, &host_id).unwrap();
            assert_eq!(s.remote_key, rx_pub);
            let mut asm = JsonAssembler::default();
            let mut got = Vec::new();
            loop {
                let f = s.reader.next().unwrap();
                match f.kind {
                    KIND_DATA => got.extend_from_slice(&f.body),
                    _ => {
                        if let Some(msg) = asm.push(&f).unwrap() {
                            let v: serde_json::Value = serde_json::from_slice(&msg).unwrap();
                            if v["end"] == true {
                                break;
                            }
                            assert_eq!(v["text"].as_str().unwrap().len(), 200_000);
                        }
                    }
                }
            }
            s.writer.json(3, &serde_json::json!({ "len": got.len() })).unwrap();
            (s.hash, got)
        });
        let stream = TcpStream::connect(addr).unwrap();
        let mut s = handshake_initiator(stream, &rx_id).unwrap();
        assert_eq!(s.remote_key, host_pub);
        let long = "x".repeat(200_000);
        s.writer.json(1, &serde_json::json!({ "text": long })).unwrap();
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        s.writer.data(1, &mut data.as_slice(), data.len() as u64, &|| false).unwrap();
        s.writer.json(1, &serde_json::json!({ "end": true })).unwrap();
        let f = s.reader.next().unwrap();
        let v: serde_json::Value = serde_json::from_slice(&f.body).unwrap();
        assert_eq!(v["len"], 300_000);
        let (host_hash, got) = server.join().unwrap();
        assert_eq!(host_hash, s.hash);
        assert_eq!(got, data);
    }

    /// Half-sent JSON messages can't pile up without limit.
    #[test]
    fn half_sent_json_is_capped() {
        let piece = |call: u32, kind: u8, n: usize| Frame { kind, call, body: vec![b' '; n] };
        let mut asm = JsonAssembler::default();
        // Whole messages and finished pieces hold nothing.
        assert_eq!(asm.push(&piece(1, KIND_JSON, 10)).unwrap().unwrap().len(), 10);
        assert!(asm.push(&piece(3, KIND_JSON_MORE, 5)).unwrap().is_none());
        assert_eq!(asm.push(&piece(3, KIND_JSON, 5)).unwrap().unwrap().len(), 10);
        assert_eq!(asm.held, 0);
        // Many calls each half sent: refused past the limit.
        let mut asm = JsonAssembler::default();
        let mut refused = false;
        for call in 0..(MAX_JSON_PARTS as u32 + 1) {
            if asm.push(&piece(call * 2 + 1, KIND_JSON_MORE, 100)).is_err() {
                refused = true;
                break;
            }
        }
        assert!(refused);
        // And by total size.
        let mut asm = JsonAssembler::default();
        let big = MAX_BODY;
        let mut sent = 0;
        let err = loop {
            match asm.push(&piece((sent as u32 % 60) * 2 + 1, KIND_JSON_MORE, big)) {
                Ok(_) => sent += 1,
                Err(e) => break e,
            }
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(sent * big <= MAX_JSON_PENDING);
    }

    /// A frame that arrives in pieces, with read timeouts in between, is put
    /// back together (not lost, which would end the session).
    #[test]
    fn frame_survives_a_timeout_mid_way() {
        let host_id = Identity::generate();
        let rx_id = Identity::generate();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (raw_tx, raw_rx) = std::sync::mpsc::channel::<TcpStream>();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let raw = stream.try_clone().unwrap();
            let s = handshake_responder(stream, &host_id).unwrap();
            raw_tx.send(raw).unwrap();
            s
        });
        let stream = TcpStream::connect(addr).unwrap();
        let mut rx = handshake_initiator(stream, &rx_id).unwrap();
        let host = server.join().unwrap();
        let mut raw = raw_rx.recv().unwrap();
        // Encrypt one frame by hand and send it in three pieces, pausing
        // longer than the reader's timeout between them.
        let mut plain = vec![KIND_JSON];
        plain.extend_from_slice(&7u32.to_be_bytes());
        plain.extend_from_slice(b"{\"t\":\"ping\"}");
        let mut buf = vec![0u8; MAX_MSG];
        let n = {
            let mut inner = host.writer.inner.lock().unwrap();
            let nonce = inner.nonce;
            inner.nonce += 1;
            host.writer.transport.write_message(nonce, &plain, &mut buf).unwrap()
        };
        let mut msg = (n as u16).to_be_bytes().to_vec();
        msg.extend_from_slice(&buf[..n]);
        rx.reader.set_timeout(Some(std::time::Duration::from_millis(50)));
        raw.write_all(&msg[..1]).unwrap();
        assert!(rx.reader.next().is_err());
        raw.write_all(&msg[1..10]).unwrap();
        assert!(rx.reader.next().is_err());
        raw.write_all(&msg[10..]).unwrap();
        let f = rx.reader.next().unwrap();
        assert_eq!(f.call, 7);
        assert_eq!(f.body, b"{\"t\":\"ping\"}");
        // And the stream is still in step for the next one.
        host.writer.json(9, &serde_json::json!({ "ok": true })).unwrap();
        rx.reader.set_timeout(Some(std::time::Duration::from_secs(5)));
        assert_eq!(rx.reader.next().unwrap().call, 9);
    }

    /// A stranger who isn't speaking Noise gets nowhere, and a different
    /// prologue (another protocol version) can't form a session.
    #[test]
    fn junk_is_refused() {
        let host_id = Identity::generate();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
            handshake_responder(stream, &host_id).is_err()
        });
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(b"\x00\x05hello").unwrap();
        assert!(server.join().unwrap());
    }
}
