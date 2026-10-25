//! Addresses: what counts as the local network, this PC's own addresses, and
//! reading what the user typed.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};

/// The port a host listens on unless set otherwise (TCP for sessions, UDP for
/// being found).
pub(crate) const DEFAULT_PORT: u16 = 47821;

/// An address on the local network: a private LAN range (10/8, 172.16/12,
/// 192.168/16), this PC, link-local, the shared address space private VPNs
/// like Tailscale use (100.64/10), or their IPv6 counterparts. Anything else
/// is the internet, and Link never talks to it.
pub(crate) fn is_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => is_local_v4(a),
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return is_local_v4(v4);
            }
            let first = a.segments()[0];
            a.is_loopback()
                || (first & 0xfe00) == 0xfc00 // unique local (fc00::/7)
                || (first & 0xffc0) == 0xfe80 // link-local (fe80::/10)
        }
    }
}

fn is_local_v4(a: Ipv4Addr) -> bool {
    let o = a.octets();
    a.is_private() || a.is_loopback() || a.is_link_local() || (o[0] == 100 && (o[1] & 0xc0) == 64)
}

/// This PC's addresses on the local network (IPv4, as other PCs would type
/// them), each with its network's broadcast address for being found.
pub(crate) fn local_addresses() -> Vec<(Ipv4Addr, Ipv4Addr)> {
    let networks = sysinfo::Networks::new_with_refreshed_list();
    let mut out: Vec<(Ipv4Addr, Ipv4Addr)> = networks
        .list()
        .values()
        .flat_map(|data| data.ip_networks().to_vec())
        .filter_map(|net| match net.addr {
            IpAddr::V4(a) if !a.is_loopback() && is_local_v4(a) => {
                let prefix = net.prefix.min(32) as u32;
                let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - prefix) };
                Some((a, Ipv4Addr::from(u32::from(a) | !mask)))
            }
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Read a typed host address — "192.168.1.23", "192.168.1.23:5000",
/// "desktop.local" — as socket addresses on the local network.
pub(crate) fn parse_host(typed: &str) -> Result<Vec<SocketAddr>, String> {
    let t = typed.trim();
    if t.is_empty() {
        return Err("Type the host's address (Link → Host on that PC shows it)".into());
    }
    // A bare IPv6 address has colons of its own.
    let with_port = if t.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("[{t}]:{DEFAULT_PORT}")
    } else if t.rsplit_once(':').is_some_and(|(_, p)| p.parse::<u16>().is_ok()) {
        t.to_string()
    } else {
        format!("{t}:{DEFAULT_PORT}")
    };
    let addrs: Vec<SocketAddr> = with_port
        .to_socket_addrs()
        .map_err(|_| format!("Couldn't find “{t}” on the network"))?
        .collect();
    let local: Vec<SocketAddr> = addrs.iter().copied().filter(|a| is_local(a.ip())).collect();
    if local.is_empty() && !addrs.is_empty() {
        return Err(format!("“{t}” isn't on your local network — Link only connects PCs on the same network"));
    }
    if local.is_empty() {
        return Err(format!("Couldn't find “{t}” on the network"));
    }
    Ok(local)
}

/// A typed address with `port` added when it names none ("192.168.1.5" →
/// "192.168.1.5:5000"; "fd00::5" → "[fd00::5]:5000").
pub(crate) fn with_port(typed: &str, port: u16) -> String {
    let t = typed.trim();
    if t.parse::<std::net::Ipv6Addr>().is_ok() {
        format!("[{t}]:{port}")
    } else if t.is_empty() || t.rsplit_once(':').is_some_and(|(_, p)| p.parse::<u16>().is_ok()) {
        t.to_string()
    } else {
        format!("{t}:{port}")
    }
}

/// This PC's name, as other PCs see it.
pub(crate) fn device_name() -> String {
    sysinfo::System::host_name()
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| "This PC".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_ranges() {
        for ok in ["192.168.1.5", "10.0.0.2", "172.16.4.1", "172.31.255.1", "127.0.0.1", "169.254.3.3", "100.101.2.3", "::1", "fd12::1", "fe80::1", "::ffff:192.168.0.9"] {
            assert!(is_local(ok.parse().unwrap()), "{ok}");
        }
        for bad in ["8.8.8.8", "172.32.0.1", "100.128.0.1", "1.1.1.1", "2001:4860::8888", "::ffff:8.8.8.8"] {
            assert!(!is_local(bad.parse().unwrap()), "{bad}");
        }
    }

    #[test]
    fn typed_addresses() {
        assert_eq!(parse_host("127.0.0.1").unwrap()[0].port(), DEFAULT_PORT);
        assert_eq!(parse_host("127.0.0.1:5000").unwrap()[0].port(), 5000);
        assert_eq!(parse_host("::1").unwrap()[0].port(), DEFAULT_PORT);
        assert!(parse_host("8.8.8.8").unwrap_err().contains("local network"));
        assert!(parse_host("  ").is_err());
        assert_eq!(with_port("192.168.1.5", 5000), "192.168.1.5:5000");
        assert_eq!(with_port(" 192.168.1.5:6000 ", 5000), "192.168.1.5:6000");
        assert_eq!(with_port("fd00::5", 5000), "[fd00::5]:5000");
        assert_eq!(with_port("[fd00::5]:7000", 5000), "[fd00::5]:7000");
        assert_eq!(with_port("desktop.local", 5000), "desktop.local:5000");
    }
}
