//! Explicit repair of `llama-build/processes.json` after a hard crash.
//!
//! A crashed app leaves process records behind. On Unix a record names the
//! supervisor, not the child's own process group, so recovery can never prove
//! the group is empty and keeps the record forever; while it exists, builds,
//! cache cleans, the staging sweep and validation all refuse to run. This is
//! the one deliberate way out: it clears the records, but only once nothing
//! alive can still be writing into the build directory. It never kills.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::paths::{BuildPaths, without_verbatim_prefix};
use super::supervise::{ProcessRecord, load_records};

/// What a refusal or a status line tells the user to do.
pub const REPAIR_ACTION: &str = "repair with POST /api/llama-build/repair-processes or `local-llm-foundry llama-build --repair-processes`";

/// One running process, as much as the check needs to know about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessInfo {
    pub pid: u32,
    /// Seconds since the epoch at creation, as the OS reports it.
    pub start_time: u64,
    /// The full command line, arguments joined by spaces.
    pub command: String,
    pub cwd: Option<PathBuf>,
}

/// The seam over the OS process table.
pub trait ProcessLister: Send + Sync {
    fn list(&self) -> Vec<ProcessInfo>;

    /// The process asking; it is never a blocker.
    fn current_pid(&self) -> u32 {
        std::process::id()
    }
}

/// A process that blocks repair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveProcess {
    pub pid: u32,
    pub command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairReport {
    /// Records removed (a corrupt file counts as none).
    pub cleared: usize,
    /// The file could not be parsed; it was removed anyway.
    pub corrupt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairError {
    Live(Vec<LiveProcess>),
    Io(String),
}

impl std::fmt::Display for RepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Live(live) => write!(
                f,
                "{} process(es) may still be using the llama-build directory; nothing was changed. Stop them, then repair again",
                live.len()
            ),
            Self::Io(e) => write!(f, "could not repair the process records: {e}"),
        }
    }
}

/// Both spellings of the directory a process could name: as configured and
/// canonical, without Windows verbatim prefixes, case-folded on Windows.
fn directory_spellings(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for candidate in [Some(root.to_path_buf()), std::fs::canonicalize(root).ok()]
        .into_iter()
        .flatten()
    {
        let text = normalize(&without_verbatim_prefix(&candidate).to_string_lossy());
        let text = text.trim_end_matches(['/', '\\']).to_string();
        if !text.is_empty() && !out.contains(&text) {
            out.push(text);
        }
    }
    out
}

fn normalize(text: &str) -> String {
    let text = text.replace(r"\\?\", "");
    if cfg!(windows) {
        text.replace('/', "\\").to_lowercase()
    } else {
        text
    }
}

/// `needle` occurs in `haystack` as a whole path: what follows it must not
/// continue a longer name (`llama-build-other` is not `llama-build`).
fn mentions_path(haystack: &str, needle: &str) -> bool {
    let mut from = 0;
    while let Some(at) = haystack[from..].find(needle) {
        let end = from + at + needle.len();
        let continues = haystack[end..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !continues {
            return true;
        }
        from = end;
    }
    false
}

fn references_directory(process: &ProcessInfo, spellings: &[String]) -> bool {
    let command = normalize(&process.command);
    let cwd = process
        .cwd
        .as_ref()
        .map(|c| normalize(&c.to_string_lossy()));
    spellings.iter().any(|dir| {
        mentions_path(&command, dir) || cwd.as_deref().is_some_and(|c| mentions_path(c, dir))
    })
}

/// Files a save may have left behind: `processes.json.tmp` (older) and
/// `processes.json.<pid>-<n>.tmp`.
fn leftover_temp_files(file: &Path) -> Vec<PathBuf> {
    let (Some(dir), Some(name)) = (file.parent(), file.file_name()) else {
        return Vec::new();
    };
    let prefix = format!("{}.", name.to_string_lossy());
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(&prefix) && n.ends_with(".tmp")
        })
        .map(|e| e.path())
        .collect()
}

fn remove_if_present(path: &Path) -> Result<(), RepairError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(RepairError::Io(format!("{}: {e}", path.display()))),
    }
}

/// Clear `processes.json` (and stray temp files) when nothing alive can still
/// own the directory. Refuses, changing nothing, while a recorded process is
/// alive with its recorded creation identity, or any other process names the
/// llama-build directory in its command line or working directory. Never
/// signals or kills anything.
pub fn repair_process_records(
    paths: &BuildPaths,
    lister: &dyn ProcessLister,
) -> Result<RepairReport, RepairError> {
    let file = &paths.process_records;
    let temps = leftover_temp_files(file);
    let present = match std::fs::symlink_metadata(file) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(RepairError::Io(format!("{}: {error}", file.display()))),
    };
    // Even a missing record is not proof that a quarantined installer has
    // stopped. Inspect directory references before releasing its lease.
    let (records, corrupt): (Vec<ProcessRecord>, bool) = match present {
        false => (Vec::new(), false),
        true => match load_records(file) {
            Ok(records) => (records, false),
            Err(_) => (Vec::new(), true),
        },
    };

    let me = lister.current_pid();
    let spellings = directory_spellings(&paths.root);
    let mut live: Vec<LiveProcess> = Vec::new();
    for process in lister.list() {
        if process.pid == me || live.iter().any(|l| l.pid == process.pid) {
            continue;
        }
        let recorded = records
            .iter()
            .any(|r| r.pid == process.pid && r.start_time.abs_diff(process.start_time) <= 1);
        if recorded || references_directory(&process, &spellings) {
            live.push(LiveProcess {
                pid: process.pid,
                command: process.command,
            });
        }
    }
    if !live.is_empty() {
        return Err(RepairError::Live(live));
    }

    remove_if_present(file)?;
    for tmp in &temps {
        remove_if_present(tmp)?;
    }
    Ok(RepairReport {
        cleared: records.len(),
        corrupt,
    })
}

/// The real process table: pid, creation time, full command line and cwd.
pub struct SysinfoLister;

impl ProcessLister for SysinfoLister {
    fn list(&self) -> Vec<ProcessInfo> {
        use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
        let mut sys = System::new();
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing()
                .with_cmd(UpdateKind::Always)
                .with_cwd(UpdateKind::Always),
        );
        sys.processes()
            .iter()
            .map(|(pid, p)| ProcessInfo {
                pid: pid.as_u32(),
                start_time: p.start_time(),
                command: p
                    .cmd()
                    .iter()
                    .map(|a| a.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(" "),
                cwd: p.cwd().map(Path::to_path_buf),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AppPaths;

    struct Fake {
        me: u32,
        procs: Vec<ProcessInfo>,
    }

    impl ProcessLister for Fake {
        fn list(&self) -> Vec<ProcessInfo> {
            self.procs.clone()
        }
        fn current_pid(&self) -> u32 {
            self.me
        }
    }

    const ME: u32 = 1;

    fn proc(pid: u32, start: u64, command: &str, cwd: Option<&Path>) -> ProcessInfo {
        ProcessInfo {
            pid,
            start_time: start,
            command: command.into(),
            cwd: cwd.map(Path::to_path_buf),
        }
    }

    fn rig() -> (tempfile::TempDir, BuildPaths) {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let paths = BuildPaths::new(&AppPaths::from_root(root));
        std::fs::create_dir_all(&paths.root).unwrap();
        (tmp, paths)
    }

    fn write_records(paths: &BuildPaths, records: &[(&str, u32, u64)]) {
        let records: Vec<ProcessRecord> = records
            .iter()
            .map(|(job, pid, start)| ProcessRecord {
                job_id: job.to_string(),
                pid: *pid,
                start_time: *start,
            })
            .collect();
        std::fs::write(
            &paths.process_records,
            serde_json::to_vec(&records).unwrap(),
        )
        .unwrap();
    }

    fn lister(procs: Vec<ProcessInfo>) -> Fake {
        Fake { me: ME, procs }
    }

    #[test]
    fn stale_records_are_cleared_and_counted() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100), ("b", 501, 100)]);
        let report = repair_process_records(&paths, &lister(vec![])).unwrap();
        assert_eq!(
            report,
            RepairReport {
                cleared: 2,
                corrupt: false
            }
        );
        assert!(!paths.process_records.exists());
    }

    #[test]
    fn missing_records_do_not_hide_a_process_still_using_the_build_directory() {
        let (_t, paths) = rig();
        let live = lister(vec![proc(77, 5, "installer", Some(&paths.root))]);
        assert!(matches!(
            repair_process_records(&paths, &live),
            Err(RepairError::Live(l)) if l.len() == 1 && l[0].pid == 77
        ));
        assert!(repair_process_records(&paths, &lister(vec![])).is_ok());
    }

    #[test]
    fn a_recorded_pid_that_is_alive_with_the_same_creation_time_refuses() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        // One second of clock slack matches `recover_record`.
        let live = lister(vec![proc(500, 101, "cmake --build x", None)]);
        match repair_process_records(&paths, &live) {
            Err(RepairError::Live(l)) => {
                assert_eq!(
                    l,
                    vec![LiveProcess {
                        pid: 500,
                        command: "cmake --build x".into()
                    }]
                );
            }
            other => panic!("{other:?}"),
        }
        assert!(paths.process_records.exists(), "refusal changes nothing");
    }

    #[test]
    fn a_reused_pid_with_a_different_creation_time_does_not_block() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let reused = lister(vec![proc(500, 9_000, "unrelated", None)]);
        assert_eq!(repair_process_records(&paths, &reused).unwrap().cleared, 1);
    }

    #[test]
    fn a_process_whose_cwd_is_inside_the_build_directory_refuses() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let inside = paths.root.join("source").join("build");
        let live = lister(vec![proc(77, 5, "ninja", Some(&inside))]);
        assert!(matches!(
            repair_process_records(&paths, &live),
            Err(RepairError::Live(l)) if l.len() == 1 && l[0].pid == 77
        ));
        assert!(paths.process_records.exists());
    }

    #[test]
    fn a_process_whose_command_line_names_the_build_directory_refuses() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let command = format!(
            "cc -c {}/source/ggml.c -o {}/x.o",
            paths.root.display(),
            paths.root.display()
        );
        let live = lister(vec![proc(78, 5, &command, None)]);
        assert!(matches!(
            repair_process_records(&paths, &live),
            Err(RepairError::Live(l)) if l[0].pid == 78
        ));
    }

    #[test]
    fn a_quoted_path_in_a_command_line_counts() {
        let (_t, paths) = rig();
        let command = format!("tool \"{}\" --flag", paths.root.display());
        let live = lister(vec![proc(79, 5, &command, None)]);
        assert!(repair_process_records(&paths, &live).is_err() || !paths.process_records.exists());
        write_records(&paths, &[("a", 500, 100)]);
        assert!(matches!(
            repair_process_records(&paths, &live),
            Err(RepairError::Live(_))
        ));
    }

    #[test]
    fn a_sibling_directory_sharing_the_prefix_does_not_match() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let sibling = PathBuf::from(format!("{}-other", paths.root.display()));
        let command = format!("tool {}/x", sibling.display());
        let live = lister(vec![
            proc(80, 5, &command, Some(&sibling)),
            proc(81, 5, "something else entirely", Some(Path::new("/"))),
        ]);
        assert_eq!(repair_process_records(&paths, &live).unwrap().cleared, 1);
    }

    #[test]
    fn the_calling_process_is_never_a_blocker() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let command = format!("local-llm-foundry --config-dir {}", paths.root.display());
        let live = lister(vec![proc(ME, 5, &command, Some(&paths.root))]);
        assert_eq!(repair_process_records(&paths, &live).unwrap().cleared, 1);
    }

    #[test]
    fn no_file_is_a_no_op_when_no_process_uses_the_directory() {
        let (_t, paths) = rig();
        let live = lister(vec![proc(77, 5, "ninja", None)]);
        assert_eq!(
            repair_process_records(&paths, &live).unwrap(),
            RepairReport {
                cleared: 0,
                corrupt: false
            }
        );
        assert!(!paths.process_records.exists());
    }

    #[test]
    fn a_corrupt_file_is_cleared_only_when_no_process_references_the_directory() {
        let (_t, paths) = rig();
        std::fs::write(&paths.process_records, "{broken").unwrap();
        let referenced = lister(vec![proc(77, 5, "ninja", Some(&paths.root))]);
        assert!(matches!(
            repair_process_records(&paths, &referenced),
            Err(RepairError::Live(_))
        ));
        assert!(paths.process_records.exists(), "kept while referenced");

        let report = repair_process_records(&paths, &lister(vec![])).unwrap();
        assert_eq!(
            report,
            RepairReport {
                cleared: 0,
                corrupt: true
            }
        );
        assert!(!paths.process_records.exists());
    }

    #[test]
    fn temp_files_from_interrupted_saves_are_removed_with_the_records() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let dir = paths.root.clone();
        for name in [
            "processes.json.tmp",
            "processes.json.123-4.tmp",
            "toolchain-cache.json",
        ] {
            std::fs::write(dir.join(name), "x").unwrap();
        }
        repair_process_records(&paths, &lister(vec![])).unwrap();
        assert!(!dir.join("processes.json.tmp").exists());
        assert!(!dir.join("processes.json.123-4.tmp").exists());
        assert!(dir.join("toolchain-cache.json").exists(), "unrelated kept");
    }

    #[test]
    fn a_leftover_temp_file_alone_is_cleaned_without_a_process_check() {
        let (_t, paths) = rig();
        std::fs::write(paths.root.join("processes.json.tmp"), "x").unwrap();
        let report = repair_process_records(&paths, &lister(vec![])).unwrap();
        assert_eq!(report.cleared, 0);
        assert!(!paths.root.join("processes.json.tmp").exists());
    }

    #[test]
    fn live_blockers_are_deduplicated_by_pid() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let live = lister(vec![proc(500, 100, "cmake", Some(&paths.root))]);
        match repair_process_records(&paths, &live) {
            Err(RepairError::Live(l)) => assert_eq!(l.len(), 1),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_match_case_insensitively_and_ignore_verbatim_prefixes() {
        let (_t, paths) = rig();
        write_records(&paths, &[("a", 500, 100)]);
        let shout = paths.root.display().to_string().to_uppercase();
        let live = lister(vec![proc(90, 5, &format!(r"\\?\{shout}\x"), None)]);
        assert!(matches!(
            repair_process_records(&paths, &live),
            Err(RepairError::Live(_))
        ));
    }
}
