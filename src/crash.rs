//! Crash reporting — every hard crash writes a report under
//! `%APPDATA%\Clarity TagFlow\crash reports\`, so installed-build failures
//! (which have no console to show anything) can be diagnosed after the fact:
//!
//!  * **Rust panics** (any thread): the panic message plus a full backtrace
//!    (`panic_YYYYMMDD_HHMMSS.txt`).
//!  * **Native faults** (access violation / illegal instruction inside
//!    libvlc, llama.cpp, whisper.cpp, ONNX…): a Windows SEH
//!    unhandled-exception filter writes the exception code, faulting address
//!    and module (`crash_*.txt`) plus a WinDbg/Visual-Studio-ready minidump
//!    (`crash_*.dmp`).
//!  * **abort() / C assertions** (e.g. a GGML_ASSERT): a SIGABRT handler
//!    writes `abort_*.txt` with a backtrace.
//!  * **stderr.log** (fresh each run, GUI builds only): everything native
//!    libraries print to stderr — GGML/VLC assertion messages land here,
//!    which is usually the actual explanation for an abort.

use std::path::PathBuf;

/// Where reports land: `%APPDATA%\Clarity TagFlow\crash reports`.
pub fn dir() -> PathBuf {
    dirs::data_dir()
        .map(|d| d.join("Clarity TagFlow"))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("crash reports")
}

fn stamp() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

fn write_text(kind: &str, body: &str) {
    let dir = dir();
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = std::fs::write(dir.join(format!("{kind}_{}.txt", stamp())), body);
    }
}

/// Install all the hooks. Call first thing in `main`.
pub fn install() {
    #[cfg(windows)]
    {
        redirect_stderr_to_log();
        install_seh();
        install_abort_handler();
    }

    // Rust panics: message + backtrace, then the default hook (stderr — which
    // in a GUI build now lands in stderr.log too).
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let body = format!(
            "Clarity TagFlow v{} — Rust panic\n\n{info}\n\nthread: {}\n\nbacktrace:\n{}",
            env!("CARGO_PKG_VERSION"),
            std::thread::current().name().unwrap_or("<unnamed>"),
            std::backtrace::Backtrace::force_capture(),
        );
        write_text("panic", &body);
        prev(info);
    }));
}

/// GUI builds have no console, so everything the native libraries print to
/// stderr — including the assertion text that explains an abort — vanishes.
/// Point both the Win32 stderr handle and the C runtime's fd 2 at a log file.
/// Dev runs (a console is attached) keep their console output.
#[cfg(windows)]
fn redirect_stderr_to_log() {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetConsoleWindow() -> isize;
        fn SetStdHandle(which: u32, handle: isize) -> i32;
    }
    // CRT functions resolve from the ucrt the binary already links.
    unsafe extern "C" {
        fn _open_osfhandle(handle: isize, flags: i32) -> i32;
        fn _dup2(fd1: i32, fd2: i32) -> i32;
    }
    unsafe {
        if GetConsoleWindow() != 0 {
            return;
        }
        let dir = dir();
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let Ok(f) = std::fs::File::create(dir.join("stderr.log")) else { return };
        use std::os::windows::io::IntoRawHandle as _;
        let h = f.into_raw_handle() as isize;
        const STD_ERROR_HANDLE: u32 = -12i32 as u32;
        SetStdHandle(STD_ERROR_HANDLE, h);
        let fd = _open_osfhandle(h, 0);
        if fd >= 0 {
            let _ = _dup2(fd, 2);
        }
    }
}

/// abort() (what a failed C assertion like GGML_ASSERT ends in) raises
/// SIGABRT, which bypasses the SEH filter — catch it separately. The real
/// reason is usually the assertion text in stderr.log.
#[cfg(windows)]
fn install_abort_handler() {
    unsafe extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    const SIGABRT: i32 = 22;
    extern "C" fn on_abort(_sig: i32) {
        let body = format!(
            "Clarity TagFlow v{} — abort() called (likely a native library \
             assertion; see stderr.log in this folder for the reason)\n\nbacktrace:\n{}",
            env!("CARGO_PKG_VERSION"),
            std::backtrace::Backtrace::force_capture(),
        );
        write_text("abort", &body);
    }
    unsafe {
        signal(SIGABRT, on_abort);
    }
}

/// Native faults: write a text summary + a minidump, then let the process die.
#[cfg(windows)]
fn install_seh() {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt as _;

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut c_void,
        n_params: u32,
        info: [usize; 15],
    }
    #[repr(C)]
    struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut c_void,
    }
    #[repr(C)]
    struct MinidumpExceptionInfo {
        thread_id: u32,
        pointers: *mut ExceptionPointers,
        client_pointers: i32,
    }

    type Filter = unsafe extern "system" fn(*mut ExceptionPointers) -> i32;

    // HANDLE as *mut c_void, matching src/pixal3d.rs's declarations (the
    // linker unifies extern blocks; differing signatures would warn).
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetUnhandledExceptionFilter(f: Option<Filter>) -> Option<Filter>;
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn GetCurrentThreadId() -> u32;
        fn GetModuleHandleExW(flags: u32, addr: *const u16, module: *mut isize) -> i32;
        fn GetModuleFileNameW(module: isize, name: *mut u16, size: u32) -> u32;
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            disposition: u32,
            flags: u32,
            template: *mut c_void,
        ) -> *mut c_void;
        fn CloseHandle(h: *mut c_void) -> i32;
    }
    #[link(name = "dbghelp")]
    unsafe extern "system" {
        fn MiniDumpWriteDump(
            process: *mut c_void,
            pid: u32,
            file: *mut c_void,
            dump_type: u32,
            exception: *mut MinidumpExceptionInfo,
            user_streams: *mut c_void,
            callback: *mut c_void,
        ) -> i32;
    }

    unsafe extern "system" fn filter(ptrs: *mut ExceptionPointers) -> i32 {
        unsafe {
            let (code, addr) = if !ptrs.is_null() && !(*ptrs).record.is_null() {
                ((*(*ptrs).record).code, (*(*ptrs).record).address)
            } else {
                (0, std::ptr::null_mut())
            };
            // Which DLL/EXE owns the faulting address — usually names the
            // culprit (libvlc.dll, the exe's ggml, onnxruntime…).
            let mut module = String::from("<unknown>");
            const FROM_ADDRESS: u32 = 0x4;
            const UNCHANGED_REFCOUNT: u32 = 0x2;
            let mut h: isize = 0;
            if GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT, addr as *const u16, &mut h) != 0 {
                let mut buf = [0u16; 512];
                let n = GetModuleFileNameW(h, buf.as_mut_ptr(), 512);
                module = String::from_utf16_lossy(&buf[..n as usize]);
            }
            let when = stamp();
            let body = format!(
                "Clarity TagFlow v{} — native crash\n\nexception code: {code:#010x}\n\
                 address: {addr:?}\nmodule: {module}\n\n\
                 A minidump (crash_{when}.dmp) was written alongside — open it \
                 in WinDbg or Visual Studio for the full stack.\n\
                 stderr.log in this folder holds the native libraries' output.\n",
                env!("CARGO_PKG_VERSION"),
            );
            let dir = dir();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join(format!("crash_{when}.txt")), &body);

            let dmp = dir.join(format!("crash_{when}.dmp"));
            let wide: Vec<u16> =
                dmp.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
            const GENERIC_WRITE: u32 = 0x4000_0000;
            const CREATE_ALWAYS: u32 = 2;
            const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
            let hfile = CreateFileW(
                wide.as_ptr(),
                GENERIC_WRITE,
                0,
                std::ptr::null_mut(),
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            );
            // INVALID_HANDLE_VALUE is -1.
            if hfile as isize != -1 {
                let mut mei = MinidumpExceptionInfo {
                    thread_id: GetCurrentThreadId(),
                    pointers: ptrs,
                    client_pointers: 0,
                };
                // MiniDumpWithDataSegs: globals included, file stays small.
                MiniDumpWriteDump(
                    GetCurrentProcess(),
                    GetCurrentProcessId(),
                    hfile,
                    0x0000_0002,
                    &mut mei,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                CloseHandle(hfile);
            }
        }
        1 // EXCEPTION_EXECUTE_HANDLER: die quietly, report written.
    }

    unsafe {
        SetUnhandledExceptionFilter(Some(filter));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The panic hook must leave a report file with the message + backtrace.
    #[test]
    fn panic_report_smoke() {
        install();
        let before = std::time::SystemTime::now();
        let _ = std::panic::catch_unwind(|| panic!("crash-report smoke test"));
        let newest = std::fs::read_dir(dir())
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("panic_"))
                    && p.metadata()
                        .and_then(|m| m.modified())
                        .is_ok_and(|t| t >= before - std::time::Duration::from_secs(2))
            })
            .max()
            .expect("no panic report written");
        let body = std::fs::read_to_string(&newest).unwrap();
        assert!(body.contains("crash-report smoke test"), "message missing from {newest:?}");
        assert!(body.contains("backtrace"), "backtrace missing");
        let _ = std::fs::remove_file(newest);
    }
}
