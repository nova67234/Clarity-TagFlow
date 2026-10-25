//! Who this PC remembers: the devices allowed to use it (as a host) and the
//! hosts it uses (as a receiver), each by identity key. Kept in
//! `trust.json` next to the identity key, owner-only.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A receiver this host lets in.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub(crate) struct Device {
    /// Identity key (hex).
    pub key: String,
    pub name: String,
    /// Unix seconds.
    pub paired: u64,
    pub last_seen: u64,
    /// Files it sends are saved without asking.
    pub accept_files: bool,
}

/// A host this receiver uses.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub(crate) struct KnownHost {
    pub key: String,
    pub name: String,
    /// Where it was last reached ("192.168.1.23:47821").
    pub addr: String,
    pub paired: u64,
    pub last_seen: u64,
    /// Files it sends are saved without asking.
    pub accept_files: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub(crate) struct Trust {
    pub devices: Vec<Device>,
    pub hosts: Vec<KnownHost>,
}

pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Trust {
    fn path(dir: &Path) -> PathBuf {
        dir.join("trust.json")
    }

    /// The stored list (empty when there's none or it can't be read).
    pub fn load(dir: &Path) -> Trust {
        std::fs::read_to_string(Self::path(dir))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Change the stored list: load, change and save under one lock, so
    /// threads saving at once (a device pairing while another disconnects)
    /// can't undo each other's change.
    pub fn update<R>(dir: &Path, change: impl FnOnce(&mut Trust) -> R) -> Result<R, String> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _held = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut t = Trust::load(dir);
        let before = t.clone();
        let r = change(&mut t);
        if t != before {
            t.save(dir)?;
        }
        Ok(r)
    }

    fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        // Written whole to a temporary file and swapped in: the background
        // host and the app may both save it.
        let tmp = dir.join(format!(".trust-{}.tmp", std::process::id()));
        crate::secret::write_private(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, Self::path(dir)).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            e.to_string()
        })
    }

    pub fn device(&self, key: &str) -> Option<&Device> {
        self.devices.iter().find(|d| d.key == key)
    }

    pub fn host(&self, key: &str) -> Option<&KnownHost> {
        self.hosts.iter().find(|h| h.key == key)
    }

    /// Remember (or refresh) a paired receiver.
    pub fn add_device(&mut self, key: &str, name: &str) {
        let t = now();
        match self.devices.iter_mut().find(|d| d.key == key) {
            Some(d) => {
                d.name = name.to_string();
                d.paired = t;
                d.last_seen = t;
            }
            None => self.devices.push(Device {
                key: key.to_string(),
                name: name.to_string(),
                paired: t,
                last_seen: t,
                accept_files: false,
            }),
        }
    }

    /// Remember (or refresh) a paired host at `addr`.
    pub fn add_host(&mut self, key: &str, name: &str, addr: &str) {
        let t = now();
        match self.hosts.iter_mut().find(|h| h.key == key) {
            Some(h) => {
                h.name = name.to_string();
                h.addr = addr.to_string();
                h.paired = t;
                h.last_seen = t;
            }
            None => self.hosts.push(KnownHost {
                key: key.to_string(),
                name: name.to_string(),
                addr: addr.to_string(),
                paired: t,
                last_seen: t,
                accept_files: false,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_refreshes() {
        let dir = std::env::temp_dir().join(format!("ctf_link_trust_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut t = Trust::default();
        t.add_device("aa", "LAPTOP");
        t.add_host("bb", "DESKTOP", "192.168.1.5:47821");
        t.add_device("aa", "LAPTOP-2");
        assert_eq!(t.devices.len(), 1);
        assert_eq!(t.devices[0].name, "LAPTOP-2");
        t.save(&dir).unwrap();
        let back = Trust::load(&dir);
        assert_eq!(back, t);
        assert_eq!(back.host("bb").unwrap().addr, "192.168.1.5:47821");
        assert!(back.device("zz").is_none());
        // Changes from many threads at once all land.
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    Trust::update(&dir, |t| t.add_device(&format!("k{i}"), "PC")).unwrap();
                })
            })
            .collect();
        for th in threads {
            th.join().unwrap();
        }
        assert_eq!(Trust::load(&dir).devices.len(), 9);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
