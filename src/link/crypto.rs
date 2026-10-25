//! Keys and pairing.
//!
//! Every install has one long-term X25519 key pair (its identity). Two apps
//! talk over a Noise `XX` session, which encrypts everything end to end and
//! tells each side the other's identity key. A host lets a receiver in only
//! when it knows that key, and a receiver only trusts a host whose key it
//! knows: they learn each other's keys once, by pairing.
//!
//! Pairing turns the 6-digit code the host shows into a shared key with SPAKE2
//! (a password-authenticated key exchange): someone who records the exchange
//! can't test guesses offline, and someone who takes part gets one guess per
//! try — and the host gives the code only a few tries. Each side then proves
//! it holds that key with an HMAC over the Noise handshake hash, which also
//! ties the pairing to this very session: a machine in the middle running a
//! session with each side has two different hashes, so it can't pass either
//! proof on without knowing the code.

use std::path::Path;

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use spake2::{Ed25519Group, Identity as SpakeId, Password, Spake2};

/// The Noise protocol every Link session runs.
pub(crate) const NOISE: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
/// Mixed into every handshake: a session only forms between two copies of
/// this app speaking this version of the protocol.
pub(crate) const PROLOGUE: &[u8] = b"Clarity TagFlow Link v1";

const SPAKE_RECEIVER: &[u8] = b"clarity-tagflow link receiver";
const SPAKE_HOST: &[u8] = b"clarity-tagflow link host";

pub(crate) fn noise_params() -> snow::params::NoiseParams {
    NOISE.parse().expect("valid Noise protocol name")
}

/// This install's long-term key pair.
#[derive(Clone)]
pub(crate) struct Identity {
    pub private: Vec<u8>,
    pub public: Vec<u8>,
}

impl Identity {
    /// The key pair stored in `dir`, made on first use. The private half is
    /// stored encrypted for this user account (DPAPI on Windows, an
    /// owner-only file elsewhere). A key file that can't be read right now
    /// (locked by a virus scanner, say) is an error, tried again later; one
    /// that can be read but not decrypted, twice (copied from another
    /// account), is set aside and replaced — paired devices then pair again.
    pub fn load_or_create(dir: &Path) -> Result<Identity, String> {
        let path = dir.join("identity.key");
        match std::fs::read(&path) {
            Ok(stored) => {
                let open = || {
                    crate::secret::unprotect_bytes(&stored)
                        .filter(|b| b.len() == 64)
                        .map(|b| Identity { private: b[..32].to_vec(), public: b[32..].to_vec() })
                };
                if let Some(id) = open().or_else(|| {
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    open()
                }) {
                    return Ok(id);
                }
                let aside = dir.join(format!("identity.key.unreadable-{}", super::trust::now()));
                std::fs::rename(&path, &aside).map_err(|e| format!("Link key: {e}"))?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("Couldn't read this PC's Link key: {e}")),
        }
        let kp = snow::Builder::new(noise_params()).generate_keypair().map_err(|e| format!("new key: {e}"))?;
        let mut both = kp.private.clone();
        both.extend_from_slice(&kp.public);
        std::fs::create_dir_all(dir).map_err(|e| format!("Link folder: {e}"))?;
        crate::secret::write_private(&path, crate::secret::protect_bytes(&both)).map_err(|e| format!("save key: {e}"))?;
        Ok(Identity { private: kp.private, public: kp.public })
    }

    #[cfg(test)]
    pub fn generate() -> Identity {
        let kp = snow::Builder::new(noise_params()).generate_keypair().unwrap();
        Identity { private: kp.private, public: kp.public }
    }
}

/// A short, readable fingerprint of an identity key ("3F2A-91C0-7B44"), shown
/// next to device names so two PCs with the same name can be told apart.
pub(crate) fn fingerprint(public: &[u8]) -> String {
    let digest = Sha256::digest(public);
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02X}")).collect();
    format!("{}-{}-{}", &hex[0..4], &hex[4..8], &hex[8..12])
}

pub(crate) fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}

/// A fresh 6-digit pairing code ("042913"), uniformly random.
pub(crate) fn new_code() -> String {
    loop {
        let mut b = [0u8; 4];
        if getrandom::getrandom(&mut b).is_err() {
            continue;
        }
        let n = u32::from_le_bytes(b);
        // Reject the top sliver so every code is equally likely.
        if n < u32::MAX - (u32::MAX % 1_000_000) {
            return format!("{:06}", n % 1_000_000);
        }
    }
}

/// The code as typed ("482 913", "482-913") reduced to its 6 digits, or None
/// when it isn't one.
pub(crate) fn normalize_code(typed: &str) -> Option<String> {
    let digits: String = typed.chars().filter(|c| !c.is_whitespace() && *c != '-').collect();
    (digits.len() == 6 && digits.chars().all(|c| c.is_ascii_digit())).then_some(digits)
}

/// The code as shown: "482 913".
pub(crate) fn show_code(code: &str) -> String {
    if code.len() == 6 { format!("{} {}", &code[..3], &code[3..]) } else { code.to_string() }
}

/// The receiver's half of pairing: its first SPAKE2 message for `code`.
pub(crate) fn pair_receiver_start(code: &str) -> (Spake2<Ed25519Group>, Vec<u8>) {
    Spake2::<Ed25519Group>::start_a(
        &Password::new(code.as_bytes()),
        &SpakeId::new(SPAKE_RECEIVER),
        &SpakeId::new(SPAKE_HOST),
    )
}

/// The receiver finishes with the host's reply: the shared key.
pub(crate) fn pair_receiver_finish(state: Spake2<Ed25519Group>, host_msg: &[u8]) -> Result<Vec<u8>, String> {
    state.finish(host_msg).map_err(|e| format!("pairing: {e:?}"))
}

/// The host's half: its SPAKE2 reply to the receiver's message, and the
/// shared key (which matches the receiver's only if both used the same code).
pub(crate) fn pair_host(code: &str, receiver_msg: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (state, reply) = Spake2::<Ed25519Group>::start_b(
        &Password::new(code.as_bytes()),
        &SpakeId::new(SPAKE_RECEIVER),
        &SpakeId::new(SPAKE_HOST),
    );
    let key = state.finish(receiver_msg).map_err(|e| format!("pairing: {e:?}"))?;
    Ok((key, reply))
}

fn mac(key: &[u8], label: &[u8], handshake_hash: &[u8]) -> Hmac<Sha256> {
    let mut m = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes any key size");
    m.update(label);
    m.update(handshake_hash);
    m
}

/// Proof that the receiver holds the pairing key, bound to this session.
pub(crate) fn receiver_proof(key: &[u8], handshake_hash: &[u8]) -> Vec<u8> {
    mac(key, b"receiver proves", handshake_hash).finalize().into_bytes().to_vec()
}

/// Proof that the host holds the pairing key, bound to this session.
pub(crate) fn host_proof(key: &[u8], handshake_hash: &[u8]) -> Vec<u8> {
    mac(key, b"host proves", handshake_hash).finalize().into_bytes().to_vec()
}

/// Check the other side's proof (in constant time).
pub(crate) fn check_receiver_proof(key: &[u8], handshake_hash: &[u8], proof: &[u8]) -> bool {
    mac(key, b"receiver proves", handshake_hash).verify_slice(proof).is_ok()
}

pub(crate) fn check_host_proof(key: &[u8], handshake_hash: &[u8], proof: &[u8]) -> bool {
    mac(key, b"host proves", handshake_hash).verify_slice(proof).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_six_digits_and_normalize() {
        for _ in 0..200 {
            let c = new_code();
            assert_eq!(c.len(), 6);
            assert!(c.chars().all(|ch| ch.is_ascii_digit()));
        }
        assert_eq!(normalize_code(" 482 913 ").as_deref(), Some("482913"));
        assert_eq!(normalize_code("482-913").as_deref(), Some("482913"));
        assert_eq!(normalize_code("48291"), None);
        assert_eq!(normalize_code("48291a"), None);
        assert_eq!(show_code("482913"), "482 913");
    }

    #[test]
    fn pairing_needs_the_same_code_and_session() {
        let hash = b"handshake hash of this session";
        let (rx_state, rx_msg) = pair_receiver_start("123456");
        let (host_key, host_msg) = pair_host("123456", &rx_msg).unwrap();
        let rx_key = pair_receiver_finish(rx_state, &host_msg).unwrap();
        assert_eq!(rx_key, host_key);
        assert!(check_receiver_proof(&host_key, hash, &receiver_proof(&rx_key, hash)));
        assert!(check_host_proof(&rx_key, hash, &host_proof(&host_key, hash)));
        // A proof made for another session doesn't pass.
        assert!(!check_receiver_proof(&host_key, b"another session", &receiver_proof(&rx_key, hash)));
        // A wrong code gives different keys: the proof fails.
        let (rx_state, rx_msg) = pair_receiver_start("123457");
        let (host_key, host_msg) = pair_host("123456", &rx_msg).unwrap();
        let rx_key = pair_receiver_finish(rx_state, &host_msg).unwrap();
        assert_ne!(rx_key, host_key);
        assert!(!check_receiver_proof(&host_key, hash, &receiver_proof(&rx_key, hash)));
    }

    #[test]
    fn hex_round_trips_and_fingerprints_are_stable() {
        let bytes = [0u8, 1, 0xab, 0xff];
        assert_eq!(from_hex(&to_hex(&bytes)).unwrap(), bytes);
        assert!(from_hex("abc").is_none());
        assert!(from_hex("zz").is_none());
        let id = Identity::generate();
        assert_eq!(fingerprint(&id.public), fingerprint(&id.public));
        assert_eq!(fingerprint(&id.public).len(), 14);
    }
}
