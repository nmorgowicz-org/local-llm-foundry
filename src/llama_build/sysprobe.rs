//! The seam between prerequisite/GPU detection and the machine.
//!
//! Every detector takes `&dyn SystemProbe`, so each rule has a fixture test
//! driven by [`FakeSystem`] and nothing in `cargo test` depends on what is
//! installed on the developer's machine.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::TargetOs;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CmdOutput {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl CmdOutput {
    pub fn ok(&self) -> bool {
        self.status == Some(0)
    }

    pub fn combined(&self) -> String {
        match (self.stdout.trim().is_empty(), self.stderr.trim().is_empty()) {
            (_, true) => self.stdout.clone(),
            (true, _) => self.stderr.clone(),
            _ => format!("{}\n{}", self.stdout, self.stderr),
        }
    }
}

pub trait SystemProbe: Send + Sync {
    fn os(&self) -> TargetOs;
    /// `Err` means the program could not be started or timed out.
    fn run(&self, program: &str, args: &[&str], timeout: Duration) -> Result<CmdOutput, String>;
    fn env_var(&self, key: &str) -> Option<String>;
    fn path_exists(&self, path: &Path) -> bool;
    fn which(&self, name: &str) -> Option<PathBuf>;
    /// Free bytes on the volume holding `path` (its nearest existing ancestor).
    fn free_space(&self, path: &Path) -> Option<u64>;
    fn logical_cpus(&self) -> usize;
    /// `HKLM` DWORD, Windows only.
    fn registry_dword(&self, key: &str, value: &str) -> Option<u32>;
}

/// Run a program with a hard deadline, killing it on expiry.
pub fn run_bounded(
    program: &str,
    args: &[&str],
    timeout: Duration,
    env: &[(String, String)],
) -> Result<CmdOutput, String> {
    run_bounded_with_grace(program, args, timeout, env, PROBE_EOF_GRACE)
}

/// Bound EOF confirmation after the direct child exits: a grandchild can
/// inherit the pipes and outlive it. Mirrors the contained supervisor's
/// `OUTPUT_DRAIN_GRACE`, which fixed the same join-before-EOF hang.
const PROBE_EOF_GRACE: Duration = Duration::from_secs(5);

fn run_bounded_with_grace(
    program: &str,
    args: &[&str],
    timeout: Duration,
    env: &[(String, String)],
    eof_grace: Duration,
) -> Result<CmdOutput, String> {
    use std::io::Read;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    crate::platform::no_window(&mut cmd);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start {program}: {e}"))?;
    let (out_tx, out_rx) = std::sync::mpsc::channel();
    let out = child.stdout.take();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(o) = out {
            let _ = o.take(4 * 1024 * 1024).read_to_end(&mut buf);
        }
        let _ = out_tx.send(buf);
    });
    let (err_tx, err_rx) = std::sync::mpsc::channel();
    let err = child.stderr.take();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(e) = err {
            let _ = e.take(4 * 1024 * 1024).read_to_end(&mut buf);
        }
        let _ = err_tx.send(buf);
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                // Readers may be parked on a pipe a grandchild still holds;
                // never join them unconditionally.
                return Err(format!(
                    "{program} did not finish within {}s",
                    timeout.as_secs()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(15)),
            Err(e) => return Err(format!("waiting for {program}: {e}")),
        }
    };
    // The direct child is gone, but a grandchild can still hold the pipes:
    // bound EOF confirmation instead of joining. What misses the budget is
    // lost and marked truncated; the reader thread is left detached.
    let eof_deadline = Instant::now() + eof_grace;
    let stdout = recv_bounded(&out_rx, eof_deadline);
    let stderr = recv_bounded(&err_rx, eof_deadline);
    let truncated = stdout.is_none() || stderr.is_none();
    let mut stderr = stderr.unwrap_or_default();
    if truncated {
        stderr.extend_from_slice(b" [output truncated: a descendant still holds the pipe]");
    }
    Ok(CmdOutput {
        status: status.code(),
        stdout: String::from_utf8_lossy(&stdout.unwrap_or_default()).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
    })
}

/// Take a reader's result if its EOF arrives before `deadline`. `None` means
/// the pipe was still held past the budget; the reader stays detached.
fn recv_bounded(rx: &std::sync::mpsc::Receiver<Vec<u8>>, deadline: Instant) -> Option<Vec<u8>> {
    let remaining = deadline.checked_duration_since(Instant::now())?;
    rx.recv_timeout(remaining).ok()
}

pub struct RealSystem;

impl SystemProbe for RealSystem {
    fn os(&self) -> TargetOs {
        TargetOs::host()
    }

    fn run(&self, program: &str, args: &[&str], timeout: Duration) -> Result<CmdOutput, String> {
        run_bounded(program, args, timeout, &[])
    }

    fn env_var(&self, key: &str) -> Option<String> {
        std::env::var(key).ok().filter(|v| !v.is_empty())
    }

    fn path_exists(&self, path: &Path) -> bool {
        path.exists()
    }

    fn which(&self, name: &str) -> Option<PathBuf> {
        let found = which::which(name).ok();
        #[cfg(target_os = "macos")]
        {
            // A GUI-launched app has no shell PATH: look where Homebrew installs.
            if found.is_none() {
                for dir in ["/opt/homebrew/bin", "/usr/local/bin"] {
                    let candidate = Path::new(dir).join(name);
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            }
        }
        found
    }

    fn free_space(&self, path: &Path) -> Option<u64> {
        free_space_of(path)
    }

    fn logical_cpus(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[cfg(windows)]
    fn registry_dword(&self, key: &str, value: &str) -> Option<u32> {
        winreg::RegKey::predef(winreg::enums::HKEY_LOCAL_MACHINE)
            .open_subkey(key)
            .ok()?
            .get_value::<u32, _>(value)
            .ok()
    }

    #[cfg(not(windows))]
    fn registry_dword(&self, _key: &str, _value: &str) -> Option<u32> {
        None
    }
}

/// Free space on the volume that holds `path`, walking up to the nearest
/// existing ancestor so it works before the directory has been created.
pub fn free_space_of(path: &Path) -> Option<u64> {
    let mut probe = Some(path);
    let existing = loop {
        match probe {
            Some(p) if p.exists() => break std::fs::canonicalize(p).ok()?,
            Some(p) => probe = p.parent(),
            None => return None,
        }
    };
    let existing = super::paths::without_verbatim_prefix(&existing);
    let disks = sysinfo::Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| existing.starts_with(super::paths::without_verbatim_prefix(d.mount_point())))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

#[cfg(test)]
pub use fake::FakeSystem;

#[cfg(test)]
mod fake {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    /// Scripted machine. Commands are matched on `program arg arg`.
    pub struct FakeSystem {
        pub os: TargetOs,
        pub commands: Mutex<BTreeMap<String, Result<CmdOutput, String>>>,
        pub env: BTreeMap<String, String>,
        pub existing: BTreeSet<PathBuf>,
        pub programs: BTreeMap<String, PathBuf>,
        pub free: Option<u64>,
        pub cpus: usize,
        pub registry: BTreeMap<(String, String), u32>,
        pub ran: Mutex<Vec<String>>,
    }

    impl FakeSystem {
        pub fn new(os: TargetOs) -> Self {
            Self {
                os,
                commands: Mutex::new(BTreeMap::new()),
                env: BTreeMap::new(),
                existing: BTreeSet::new(),
                programs: BTreeMap::new(),
                free: Some(500 * 1024 * 1024 * 1024),
                cpus: 8,
                registry: BTreeMap::new(),
                ran: Mutex::new(Vec::new()),
            }
        }

        pub fn on(self, command_line: &str, stdout: &str) -> Self {
            self.on_status(command_line, 0, stdout, "")
        }

        pub fn on_status(
            self,
            command_line: &str,
            status: i32,
            stdout: &str,
            stderr: &str,
        ) -> Self {
            self.commands.lock().unwrap().insert(
                command_line.to_string(),
                Ok(CmdOutput {
                    status: Some(status),
                    stdout: stdout.into(),
                    stderr: stderr.into(),
                }),
            );
            self
        }

        pub fn on_err(self, command_line: &str, error: &str) -> Self {
            self.commands
                .lock()
                .unwrap()
                .insert(command_line.to_string(), Err(error.to_string()));
            self
        }

        pub fn with_program(mut self, name: &str, path: &str) -> Self {
            self.programs.insert(name.into(), PathBuf::from(path));
            self.existing.insert(PathBuf::from(path));
            self
        }

        pub fn with_env(mut self, key: &str, value: &str) -> Self {
            self.env.insert(key.into(), value.into());
            self
        }

        pub fn with_path(mut self, path: &str) -> Self {
            self.existing.insert(PathBuf::from(path));
            self
        }

        pub fn with_registry(mut self, key: &str, value: &str, data: u32) -> Self {
            self.registry.insert((key.into(), value.into()), data);
            self
        }

        pub fn ran(&self) -> Vec<String> {
            self.ran.lock().unwrap().clone()
        }
    }

    impl SystemProbe for FakeSystem {
        fn os(&self) -> TargetOs {
            self.os
        }

        fn run(
            &self,
            program: &str,
            args: &[&str],
            _timeout: Duration,
        ) -> Result<CmdOutput, String> {
            let line = std::iter::once(program)
                .chain(args.iter().copied())
                .collect::<Vec<_>>()
                .join(" ");
            self.ran.lock().unwrap().push(line.clone());
            self.commands
                .lock()
                .unwrap()
                .get(&line)
                .cloned()
                .unwrap_or_else(|| Err(format!("could not start {program}: not scripted ({line})")))
        }

        fn env_var(&self, key: &str) -> Option<String> {
            self.env.get(key).cloned()
        }

        fn path_exists(&self, path: &Path) -> bool {
            self.existing.contains(path)
        }

        fn which(&self, name: &str) -> Option<PathBuf> {
            self.programs.get(name).cloned()
        }

        fn free_space(&self, _path: &Path) -> Option<u64> {
            self.free
        }

        fn logical_cpus(&self) -> usize {
            self.cpus
        }

        fn registry_dword(&self, key: &str, value: &str) -> Option<u32> {
            self.registry.get(&(key.into(), value.into())).copied()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_run_captures_output_and_status() {
        #[cfg(unix)]
        {
            let out = run_bounded(
                "sh",
                &["-c", "echo hi; echo err 1>&2; exit 3"],
                Duration::from_secs(10),
                &[],
            )
            .unwrap();
            assert_eq!(out.status, Some(3));
            assert_eq!(out.stdout.trim(), "hi");
            assert_eq!(out.stderr.trim(), "err");
            assert!(!out.ok());
        }
    }

    #[test]
    fn bounded_run_kills_on_timeout() {
        #[cfg(unix)]
        {
            let started = Instant::now();
            let err = run_bounded("sleep", &["30"], Duration::from_millis(150), &[]).unwrap_err();
            assert!(err.contains("did not finish"), "{err}");
            assert!(started.elapsed() < Duration::from_secs(10));
        }
    }

    #[test]
    fn a_grandchild_holding_the_pipe_cannot_hang_the_probe() {
        #[cfg(unix)]
        {
            // `sh` exits immediately; the background `sleep` inherits the
            // pipes and keeps them open past the grace budget.
            let started = Instant::now();
            let out = run_bounded_with_grace(
                "sh",
                &["-c", "sleep 2 & exit 0"],
                Duration::from_secs(10),
                &[],
                Duration::from_millis(300),
            )
            .unwrap();
            assert!(out.ok(), "{}", out.combined());
            assert!(
                out.stderr.contains("output truncated"),
                "{}",
                out.combined()
            );
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "probe took {:?}",
                started.elapsed()
            );
        }
    }

    #[test]
    fn a_missing_program_is_an_error_not_a_panic() {
        let err = run_bounded(
            "definitely-not-a-real-program-xyz",
            &[],
            Duration::from_secs(1),
            &[],
        )
        .unwrap_err();
        assert!(err.contains("could not start"));
    }

    #[test]
    fn free_space_resolves_for_a_not_yet_created_directory() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(free_space_of(&tmp.path().join("a/b/c")).is_some());
    }
}
