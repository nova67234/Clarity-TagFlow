//! View-only access to (optionally encrypted) zip archives.
//!
//! "Open Zip (.zip)..." in the folder popup mounts an archive as the browser's
//! input: its entries are listed as *virtual paths* — `<zip path>\<entry>` —
//! and every decoder that wants bytes goes through [`read`] / [`open`] here,
//! which serve zip entries straight out of the archive in memory. Nothing is
//! ever extracted to disk, and the password (AES-256 / ZipCrypto — the same
//! encryption Create Backup writes) lives only in this process for the
//! session.
//!
//! Only one archive is open at a time, mirroring the one-folder browser model.

use std::collections::HashSet;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

/// The currently mounted archive, if any.
static ARCHIVE: RwLock<Option<Arc<OpenArchive>>> = RwLock::new(None);

struct OpenArchive {
    zip_path: PathBuf,
    /// `None` for archives without encrypted entries.
    password: Option<Vec<u8>>,
    /// Entry names (zip-style `/` separators) for cheap existence checks.
    names: HashSet<String>,
    /// The zip reader needs `&mut` per entry, so concurrent decode threads
    /// serialise here — each holds the lock only while inflating one entry
    /// into memory.
    zip: Mutex<zip::ZipArchive<io::BufReader<std::fs::File>>>,
}

/// Why an archive couldn't be opened (surfaced in the password prompt).
#[derive(Debug)]
pub enum OpenError {
    /// Encrypted entries and the password is missing or wrong.
    WrongPassword,
    /// Not a readable zip at all.
    Other(String),
}

/// Open `zip_path` and mount it as THE active archive (replacing any previous
/// one). Returns every file entry as a virtual path. `password` may be empty
/// for unencrypted archives; it's verified against the first encrypted entry
/// before the archive is accepted.
pub fn open_archive(zip_path: &Path, password: &str) -> Result<Vec<PathBuf>, OpenError> {
    let file = std::fs::File::open(zip_path).map_err(|e| OpenError::Other(e.to_string()))?;
    let mut zip = zip::ZipArchive::new(io::BufReader::new(file))
        .map_err(|e| OpenError::Other(e.to_string()))?;

    // Walk the central directory (metadata only): collect file entries and
    // find whether anything is encrypted.
    let mut names = HashSet::new();
    let mut paths = Vec::new();
    let mut first_encrypted: Option<String> = None;
    for i in 0..zip.len() {
        let entry = match zip.by_index_raw(i) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_string();
        if entry.encrypted() && first_encrypted.is_none() {
            first_encrypted = Some(name.clone());
        }
        paths.push(zip_path.join(&name));
        names.insert(name);
    }

    let password = (!password.is_empty()).then(|| password.as_bytes().to_vec());

    // Verify the password by actually decrypting the start of an encrypted
    // entry: AES rejects a wrong password at open (password verifier); legacy
    // ZipCrypto catches ~255/256 wrong passwords the same way.
    if let Some(name) = &first_encrypted {
        let Some(pw) = &password else {
            return Err(OpenError::WrongPassword);
        };
        let mut probe = [0u8; 512];
        match zip.by_name_decrypt(name, pw) {
            Ok(mut entry) => {
                if entry.read(&mut probe).is_err() {
                    return Err(OpenError::WrongPassword);
                }
            }
            Err(_) => return Err(OpenError::WrongPassword),
        }
    }

    *ARCHIVE.write().unwrap() = Some(Arc::new(OpenArchive {
        zip_path: zip_path.to_path_buf(),
        password,
        names,
        zip: Mutex::new(zip),
    }));
    Ok(paths)
}

/// Unmount the active archive (its password is dropped with it).
pub fn close() {
    *ARCHIVE.write().unwrap() = None;
}

/// The path of the mounted zip, if one is open.
pub fn active() -> Option<PathBuf> {
    ARCHIVE.read().unwrap().as_ref().map(|a| a.zip_path.clone())
}

/// Resolve `path` to the open archive and an entry name when it's a virtual
/// archive path (i.e. sits under the mounted zip's path).
fn entry_of(path: &Path) -> Option<(Arc<OpenArchive>, String)> {
    let guard = ARCHIVE.read().unwrap();
    let a = guard.as_ref()?;
    let rel = path.strip_prefix(&a.zip_path).ok()?;
    if rel.as_os_str().is_empty() {
        return None;
    }
    let name = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    Some((Arc::clone(a), name))
}

/// Whether `path` points inside the mounted archive (and is therefore
/// read-only: no tag edits, moves, deletes or crops).
pub fn is_entry(path: &Path) -> bool {
    entry_of(path).is_some()
}

impl OpenArchive {
    /// Inflate one entry fully into memory (decrypting as needed).
    fn read_entry(&self, name: &str) -> io::Result<Vec<u8>> {
        if !self.names.contains(name) {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no such zip entry"));
        }
        let mut zip = self.zip.lock().unwrap();
        let mut entry = match &self.password {
            Some(pw) => zip.by_name_decrypt(name, pw),
            None => zip.by_name(name),
        }
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        let mut buf = Vec::with_capacity(entry.size().min(1 << 31) as usize);
        entry.read_to_end(&mut buf)?;
        Ok(buf)
    }
}

/// `std::fs::read`, but virtual archive paths read from the zip in memory.
pub fn read(path: &Path) -> io::Result<Vec<u8>> {
    match entry_of(path) {
        Some((a, name)) => a.read_entry(&name),
        None => std::fs::read(path),
    }
}

/// `std::fs::read_to_string`, but virtual archive paths read from the zip.
pub fn read_to_string(path: &Path) -> io::Result<String> {
    match entry_of(path) {
        Some((a, name)) => String::from_utf8(a.read_entry(&name)?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
        None => std::fs::read_to_string(path),
    }
}

/// A seekable reader over a plain file or an in-memory zip entry — what the
/// streaming decoders (jpeg/png/tiff subsampling, GIF/MP4 probes, animation
/// player) use in place of `File::open`.
pub enum Reader {
    File(std::fs::File),
    Mem(io::Cursor<Vec<u8>>),
}

/// `std::fs::File::open`, but virtual archive paths get an in-memory reader.
pub fn open(path: &Path) -> io::Result<Reader> {
    match entry_of(path) {
        Some((a, name)) => Ok(Reader::Mem(io::Cursor::new(a.read_entry(&name)?))),
        None => std::fs::File::open(path).map(Reader::File),
    }
}

impl Reader {
    /// Total length in bytes (replaces `File::metadata().len()`).
    pub fn len(&mut self) -> io::Result<u64> {
        match self {
            Reader::File(f) => f.metadata().map(|m| m.len()),
            Reader::Mem(c) => Ok(c.get_ref().len() as u64),
        }
    }
}

impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Reader::File(f) => f.read(buf),
            Reader::Mem(c) => c.read(buf),
        }
    }
}

impl Seek for Reader {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        match self {
            Reader::File(f) => f.seek(pos),
            Reader::Mem(c) => c.seek(pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Round-trip: write an AES-256 encrypted zip (as Create Backup does),
    /// mount it, and read entries back through the virtual-path API.
    #[test]
    fn encrypted_zip_roundtrip() {
        let dir = std::env::temp_dir().join("clarity_archive_test");
        let _ = std::fs::create_dir_all(&dir);
        let zip_path = dir.join("test.zip");

        let file = std::fs::File::create(&zip_path).unwrap();
        let mut w = zip::ZipWriter::new(file);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .with_aes_encryption(zip::AesMode::Aes256, "secret");
        w.start_file("photos/a.png", opts).unwrap();
        w.write_all(b"not really a png").unwrap();
        w.start_file("photos/a.txt", opts).unwrap();
        w.write_all(b"tag1, tag2").unwrap();
        w.finish().unwrap();

        // Wrong + missing passwords are rejected.
        assert!(matches!(open_archive(&zip_path, "nope"), Err(OpenError::WrongPassword)));
        assert!(matches!(open_archive(&zip_path, ""), Err(OpenError::WrongPassword)));

        // Right password lists both entries as virtual paths.
        let entries = open_archive(&zip_path, "secret").unwrap();
        assert_eq!(entries.len(), 2);
        let img = zip_path.join("photos/a.png");
        assert!(entries.contains(&img));
        assert!(is_entry(&img));
        assert!(read(&zip_path.join("photos/missing.png")).is_err());

        // Bytes come back decrypted, entirely in memory.
        assert_eq!(read(&img).unwrap(), b"not really a png");
        assert_eq!(read_to_string(&zip_path.join("photos/a.txt")).unwrap(), "tag1, tag2");

        // Real files still route to the filesystem.
        close();
        assert!(!is_entry(&img));
        std::fs::remove_file(&zip_path).unwrap();
    }
}
