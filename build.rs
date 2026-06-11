//! Build script: for the libVLC-backed build on Windows, stage VLC's runtime
//! next to the executable so the app is self-contained and launches anywhere
//! (IDE, `cargo run`, or double-click).
//!
//! Two problems this solves:
//!  * `libvlc.dll`/`libvlccore.dll` are *load-time* dependencies, so they must be
//!    resolvable before `main()` runs — otherwise the loader fails with
//!    STATUS_DLL_NOT_FOUND (0xc0000135). Copying them into the exe's directory
//!    (first on the loader's search path) fixes that.
//!  * libVLC 3.x finds its plugins only in a `plugins\` folder *next to
//!    libvlc.dll* — it ignores `VLC_PLUGIN_PATH` in this build — so the plugins
//!    folder has to sit beside the copied DLLs or `libvlc_new()` returns NULL
//!    (surfaced in-app as "Couldn't start video playback.").
//!
//! The VLC install location defaults to the standard 64-bit path; override it
//! with the `VLC_RUNTIME_DIR` environment variable. The plugins copy is skipped
//! when it already exists, so only a fresh/`cargo clean` build pays the cost.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VLC_RUNTIME_DIR");

    // Embed the application icon into the Windows exe so Explorer, the taskbar,
    // and any shortcuts/installer show it. Runs regardless of the VLC feature.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rerun-if-changed=icons/app-icon.ico");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icons/app-icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=failed to embed app icon: {e}");
        }
    }

    // Only the `vlc` feature pulls in libVLC, and only the Windows target needs
    // the side-by-side runtime.
    if std::env::var_os("CARGO_FEATURE_VLC").is_none() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    // Delay-load libVLC so the exe launches even when no libVLC runtime is present
    // (src/video.rs probes for one at runtime and only then calls into it). Emitting
    // this here — rather than only via RUSTFLAGS / .cargo/config.toml — guarantees it
    // is applied in every build, including CI where RUSTFLAGS is overridden and would
    // otherwise leave libvlc.dll as a load-time import (app fails to start without it).
    println!("cargo:rustc-link-arg=/DELAYLOAD:libvlc.dll");
    println!("cargo:rustc-link-arg=delayimp.lib");

    // Never stage the VLC runtime in CI: release artifacts must not redistribute
    // VLC (LGPL licensing) — shipped builds rely on the user's own VLC install at
    // runtime (delay-load + the probe in src/video.rs, with an install prompt when
    // it's missing). The staging below is purely a local-dev convenience so
    // `cargo run` plays videos without VLC's folder on the DLL search path.
    println!("cargo:rerun-if-env-changed=CI");
    if std::env::var_os("CI").is_some() {
        return;
    }

    let vlc_dir = PathBuf::from(
        std::env::var("VLC_RUNTIME_DIR")
            .unwrap_or_else(|_| r"C:\Program Files\VideoLAN\VLC".to_string()),
    );

    // OUT_DIR is target/<profile>/build/<pkg>-<hash>/out — the exe lives three
    // levels up, in target/<profile>.
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let exe_dir = Path::new(&out_dir)
        .ancestors()
        .nth(3)
        .expect("OUT_DIR has the expected depth");

    // Core DLLs (small — copy every time so updates propagate).
    for dll in ["libvlc.dll", "libvlccore.dll"] {
        let src = vlc_dir.join(dll);
        let dst = exe_dir.join(dll);
        if !src.exists() {
            println!(
                "cargo:warning=VLC runtime DLL not found at {} — set VLC_RUNTIME_DIR to your VLC install folder.",
                src.display()
            );
            continue;
        }
        if let Err(e) = std::fs::copy(&src, &dst) {
            println!("cargo:warning=failed to copy {} -> {}: {e}", src.display(), dst.display());
        }
    }

    // Plugins folder (~130 MB) — copy only if not already staged.
    let plugins_src = vlc_dir.join("plugins");
    let plugins_dst = exe_dir.join("plugins");
    if plugins_src.is_dir() && !plugins_dst.exists() {
        if let Err(e) = copy_dir_all(&plugins_src, &plugins_dst) {
            println!("cargo:warning=failed to stage VLC plugins from {}: {e}", plugins_src.display());
        }
    } else if !plugins_src.is_dir() {
        println!(
            "cargo:warning=VLC plugins folder not found at {} — videos won't play.",
            plugins_src.display()
        );
    }
}

/// Recursively copy `src` into `dst` (std has no built-in for this).
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}
