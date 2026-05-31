//! At-rest protection for stored credentials (the Gelbooru API key).
//!
//! On **Windows** this uses DPAPI (`CryptProtectData` / `CryptUnprotectData`):
//! the secret is encrypted with a key the OS holds, tied to the current user
//! account. It can't be read as plaintext from the JSON, can't be decrypted by a
//! different user, and can't be moved to another machine. No key is stored by us.
//!
//! On **other platforms** (mac / linux CI builds) there's no DPAPI, so it falls
//! back to a light XOR obfuscation — best-effort: it keeps the value out of plain
//! sight in the file, but isn't real cryptography. The Windows path is the one
//! that matters for this app's users.
//!
//! Stored form is `enc:v1:<hex>`. [`unprotect`] also accepts bare plaintext, so
//! credentials saved before encryption was added still load.

const PREFIX: &str = "enc:v1:";

/// Encrypt a secret for storage. Empty input stays empty; on any failure the
/// plaintext is returned unchanged so saving never silently loses the value.
pub fn protect(plain: &str) -> String {
    if plain.is_empty() {
        return String::new();
    }
    match imp::encrypt(plain.as_bytes()) {
        Some(ct) => format!("{PREFIX}{}", to_hex(&ct)),
        None => plain.to_string(),
    }
}

/// Decrypt a stored secret. Accepts the `enc:v1:` form and bare plaintext (so a
/// pre-encryption config keeps working). Returns "" if the ciphertext is corrupt
/// or was protected by a different user / machine.
pub fn unprotect(stored: &str) -> String {
    let Some(hex) = stored.strip_prefix(PREFIX) else {
        return stored.to_string(); // legacy plaintext
    };
    let Some(ct) = from_hex(hex) else {
        return String::new();
    };
    imp::decrypt(&ct)
        .and_then(|b| String::from_utf8(b).ok())
        .unwrap_or_default()
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap());
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if bytes.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Windows: DPAPI via direct crypt32 FFI (no extra crates).
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod imp {
    use core::ffi::c_void;
    use core::ptr;

    #[repr(C)]
    struct DataBlob {
        cb: u32,
        pb: *mut u8,
    }

    // CRYPTPROTECT_UI_FORBIDDEN — never show a UI prompt (we run headless).
    const UI_FORBIDDEN: u32 = 0x1;

    unsafe extern "system" {
        fn CryptProtectData(
            data_in: *const DataBlob,
            data_descr: *const u16,
            optional_entropy: *const DataBlob,
            reserved: *mut c_void,
            prompt_struct: *mut c_void,
            flags: u32,
            data_out: *mut DataBlob,
        ) -> i32;

        fn CryptUnprotectData(
            data_in: *const DataBlob,
            data_descr: *mut *mut u16,
            optional_entropy: *const DataBlob,
            reserved: *mut c_void,
            prompt_struct: *mut c_void,
            flags: u32,
            data_out: *mut DataBlob,
        ) -> i32;
    }

    unsafe extern "system" {
        fn LocalFree(h: *mut c_void) -> *mut c_void;
    }

    pub fn encrypt(data: &[u8]) -> Option<Vec<u8>> {
        unsafe {
            let in_blob = DataBlob {
                cb: data.len() as u32,
                pb: data.as_ptr() as *mut u8,
            };
            let mut out = DataBlob { cb: 0, pb: ptr::null_mut() };
            let ok = CryptProtectData(
                &in_blob,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                UI_FORBIDDEN,
                &mut out,
            );
            if ok == 0 || out.pb.is_null() {
                return None;
            }
            let v = std::slice::from_raw_parts(out.pb, out.cb as usize).to_vec();
            LocalFree(out.pb as *mut c_void);
            Some(v)
        }
    }

    pub fn decrypt(data: &[u8]) -> Option<Vec<u8>> {
        unsafe {
            let in_blob = DataBlob {
                cb: data.len() as u32,
                pb: data.as_ptr() as *mut u8,
            };
            let mut out = DataBlob { cb: 0, pb: ptr::null_mut() };
            let ok = CryptUnprotectData(
                &in_blob,
                ptr::null_mut(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                UI_FORBIDDEN,
                &mut out,
            );
            if ok == 0 || out.pb.is_null() {
                return None;
            }
            let v = std::slice::from_raw_parts(out.pb, out.cb as usize).to_vec();
            LocalFree(out.pb as *mut c_void);
            Some(v)
        }
    }
}

// ---------------------------------------------------------------------------
// Non-Windows fallback: XOR obfuscation (not real crypto — see module docs).
// ---------------------------------------------------------------------------
#[cfg(not(windows))]
mod imp {
    const KEY: &[u8] = b"ClarityTagFlow/local-obfuscation/v1";

    pub fn encrypt(data: &[u8]) -> Option<Vec<u8>> {
        Some(xor(data))
    }

    pub fn decrypt(data: &[u8]) -> Option<Vec<u8>> {
        Some(xor(data))
    }

    fn xor(data: &[u8]) -> Vec<u8> {
        data.iter()
            .enumerate()
            .map(|(i, &b)| b ^ KEY[i % KEY.len()])
            .collect()
    }
}
