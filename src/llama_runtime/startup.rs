//! Startup recovery for interrupted runtime transactions.
//!
//! Runs synchronously before anything can launch a model: a crash mid-swap
//! leaves a journal, and the live set must be made whole again (restored if
//! uncommitted, finished if committed) before the app serves a request.
//!
//! The app root, the canonical default root and the legacy default root are all
//! checked, because the 2.x root migration is intentionally restartable and a
//! managed `bin/` may live in any of them.

use std::path::PathBuf;

use crate::config::AppConfig;
use crate::llama_runtime::promote::{
    NoFaults, RecoveryReport, RuntimePaths, recover, sweep_orphans,
};
use crate::paths::AppPaths;

/// One line for the log; `warning` is set when the user must act.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartupFinding {
    pub message: String,
    pub warning: bool,
}

fn candidate_roots(config: &AppConfig) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for root in [
        config.app_paths.root.clone(),
        AppPaths::canonical_default_root(),
        AppPaths::legacy_default_root(),
    ] {
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    roots
}

/// Replay interrupted transactions in every candidate root, but only tidy
/// (`sweep_orphans`) the root this process actually manages. A root the app is
/// not using may belong to another install; its journal can only have come from
/// an interrupted transaction, so replaying it is safe, while pruning or
/// adopting its directories is not ours to do.
pub fn recover_roots(roots: &[PathBuf], active_root: &std::path::Path) -> Vec<StartupFinding> {
    let mut findings = Vec::new();
    for root in roots {
        let live = root.join("bin");
        if !root.is_dir() {
            continue;
        }
        let Ok(paths) = RuntimePaths::for_live_dir(&live) else {
            continue;
        };
        let report = match recover(&paths, &NoFaults) {
            Ok(report) => report,
            Err(e) => {
                findings.push(StartupFinding {
                    message: format!(
                        "llama.cpp runtime recovery in {} was interrupted: {e:?}; it will be retried on the next start",
                        root.display()
                    ),
                    warning: true,
                });
                continue;
            }
        };
        let blocked = matches!(report, RecoveryReport::Blocked { .. });
        match &report {
            RecoveryReport::Nothing => {}
            RecoveryReport::Restored { journal } => findings.push(StartupFinding {
                message: format!(
                    "llama.cpp runtime: an interrupted {:?} of '{}' was rolled back; the previous runtime is live",
                    journal.op, journal.label
                ),
                warning: false,
            }),
            RecoveryReport::Completed { journal } => findings.push(StartupFinding {
                message: format!(
                    "llama.cpp runtime: an interrupted {:?} of '{}' had committed and was completed",
                    journal.op, journal.label
                ),
                warning: false,
            }),
            RecoveryReport::Blocked { reason } => findings.push(StartupFinding {
                message: format!(
                    "llama.cpp runtime changes are blocked in {}: {reason}. Nothing was modified; resolve the journal file or reinstall",
                    root.display()
                ),
                warning: true,
            }),
        }
        if report.restart_pending() {
            findings.push(StartupFinding {
                message: "llama.cpp runtime: the model that was running during the interrupted swap was not restarted; start it again".into(),
                warning: true,
            });
        }
        // Orphans are only swept from a coherent runtime directory.
        if !blocked
            && root.as_path() == active_root
            && let Err(e) = sweep_orphans(&paths)
        {
            findings.push(StartupFinding {
                message: format!(
                    "llama.cpp runtime: could not clean leftover staging directories: {e}"
                ),
                warning: false,
            });
        }
    }
    findings
}

pub fn recover_for_startup(config: &AppConfig) -> Vec<StartupFinding> {
    recover_roots(&candidate_roots(config), &config.app_paths.root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_root_reports_nothing_and_a_missing_root_is_skipped() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin")).unwrap();
        let missing = root.path().join("does-not-exist");
        assert!(recover_roots(&[root.path().to_path_buf(), missing], root.path()).is_empty());
    }

    #[test]
    fn a_corrupt_journal_is_a_warning_and_leaves_every_directory_alone() {
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("llama-server"), b"live").unwrap();
        std::fs::write(root.path().join("runtime-transaction.json"), b"{broken").unwrap();
        let staging = root.path().join(".bin.staging-abc");
        std::fs::create_dir_all(&staging).unwrap();

        let findings = recover_roots(&[root.path().to_path_buf()], root.path());

        assert!(
            findings
                .iter()
                .any(|f| f.warning && f.message.contains("blocked"))
        );
        assert_eq!(std::fs::read(bin.join("llama-server")).unwrap(), b"live");
        // Blocked: even orphan sweeping is skipped.
        assert!(staging.exists());
    }

    #[test]
    fn orphaned_staging_directories_are_swept_when_the_journal_is_clean() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("bin")).unwrap();
        let staging = root.path().join(".bin.staging-abc");
        std::fs::create_dir_all(&staging).unwrap();

        recover_roots(&[root.path().to_path_buf()], root.path());

        assert!(!staging.exists());
    }

    #[test]
    fn a_non_active_root_is_recovered_but_never_swept_or_adopted() {
        let active = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(other.path().join("bin")).unwrap();
        let staging = other.path().join(".bin.staging-abc");
        let legacy = other.path().join("bin-previous-b1-1-1");
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();

        recover_roots(
            &[active.path().to_path_buf(), other.path().to_path_buf()],
            active.path(),
        );

        assert!(
            staging.exists(),
            "another install's directories are left alone"
        );
        assert!(legacy.exists());
        assert!(!other.path().join("bin-previous").exists());
    }
}
