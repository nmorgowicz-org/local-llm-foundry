//! Doctor findings for the managed llama.cpp runtime and its source builds.
//!
//! Pure: callers gather [`RuntimeFacts`] (file reads only, no subprocesses) and
//! this module decides what to say. Findings describe state and the next step;
//! none of them changes anything. Capability is never inferred here: a missing
//! probe is "unavailable", never a launch block and never a rebuild
//! recommendation (only a confirmed missing buildable CUDA kernel earns one,
//! and that comes from the K/V classifier, not from this module).

use crate::state::{DoctorFinding, DoctorFindingType, DoctorSeverity};

const SECTION: &str = "llama.cpp runtime";
/// A prerequisite report older than this no longer describes the machine well.
pub const PREREQ_STALE_SECS: u64 = 30 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalFact {
    Clean,
    /// An interrupted runtime transaction recovery has not finished.
    Pending {
        label: String,
    },
    /// Unreadable or unknown-schema journal; nothing is touched until repaired.
    Corrupt {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrereqFact {
    pub validated_at_unix: u64,
    /// Titles of checks that failed when the report was taken.
    pub failing: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFacts {
    /// `foundry_source_build`, `upstream_prebuilt`, `custom`, `unknown`; None without a managed runtime.
    pub provenance_source: Option<String>,
    /// Installed files versus the manifest hashes; None when not comparable.
    pub hashes_match: Option<bool>,
    pub journal: JournalFact,
    /// `llama-build/processes.json` exists and its writers are not proven gone.
    pub process_records_unresolved: bool,
    /// Pre-journal rollback backups left beside the live directory.
    pub legacy_backups: usize,
    /// The newest source-build prerequisite report, if a build was ever attempted.
    pub prereq: Option<PrereqFact>,
    /// A build, update or recovery is running right now: journals and process
    /// records are expected to exist, so they say nothing about a past failure.
    pub busy: bool,
    pub now_unix: u64,
}

fn finding(severity: DoctorSeverity, message: impl Into<String>) -> DoctorFinding {
    DoctorFinding {
        finding_type: DoctorFindingType::LlamaCpp,
        severity,
        message: message.into(),
        section: SECTION.into(),
        fix: None,
    }
}

pub fn runtime_findings(facts: &RuntimeFacts) -> Vec<DoctorFinding> {
    let mut out = Vec::new();

    match &facts.journal {
        _ if facts.busy => {}
        JournalFact::Clean => {}
        JournalFact::Pending { label } => out.push(finding(
            DoctorSeverity::Issue,
            format!(
                "A llama.cpp runtime update ({label}) was interrupted and recovery has not finished. \
                 Restart the app so recovery can complete before starting a model."
            ),
        )),
        JournalFact::Corrupt { reason } => out.push(finding(
            DoctorSeverity::Issue,
            format!(
                "The llama.cpp runtime transaction record is unusable ({reason}). Both runtime sets \
                 are kept untouched; do not delete the journal to bypass this, repair it or reinstall."
            ),
        )),
    }

    if facts.process_records_unresolved && !facts.busy {
        out.push(finding(
            DoctorSeverity::Issue,
            "A previous llama.cpp build left unresolved process records (llama-build/processes.json). \
             Builds, cleaning and startup cleanup are blocked until any leftover cmake/ninja/compiler \
             processes are stopped and the record is repaired. Do not delete the file just to bypass it.",
        ));
    }

    if facts.hashes_match == Some(false) && !facts.busy {
        out.push(finding(
            DoctorSeverity::Warning,
            "The installed llama.cpp files no longer match the hashes recorded when they were \
             installed, so they are treated as a custom runtime. Reinstall or rebuild to restore \
             verified provenance.",
        ));
    }

    if facts.legacy_backups > 0 {
        out.push(finding(
            DoctorSeverity::Warning,
            format!(
                "{} older llama.cpp rollback backup folder(s) from a previous updater are still next \
                 to the runtime. They are not managed and are never deleted automatically; remove \
                 them yourself if you no longer need them.",
                facts.legacy_backups
            ),
        ));
    }

    if let Some(prereq) = &facts.prereq {
        if !prereq.failing.is_empty() {
            out.push(finding(
                DoctorSeverity::Warning,
                format!(
                    "The last source-build prerequisite check failed: {}. This only matters if you \
                     want to build llama.cpp from source; the llama.cpp Runtime panel shows the fix.",
                    prereq.failing.join(", ")
                ),
            ));
        } else if facts.now_unix.saturating_sub(prereq.validated_at_unix) > PREREQ_STALE_SECS {
            out.push(finding(
                DoctorSeverity::Warning,
                "The source-build prerequisite check is more than 30 days old; tools may have changed. \
                 Re-run Validate in the llama.cpp Runtime panel before building.",
            ));
        }
    }

    out
}

/// Shown when a runtime file exists but the capability probe returned nothing.
pub fn capability_unavailable_finding() -> DoctorFinding {
    finding(
        DoctorSeverity::Warning,
        "The llama.cpp capability probe returned no usable result for this runtime. Launches are \
         not blocked, but K/V support is shown as unverified. A rebuild is not recommended from \
         this alone.",
    )
}

/// The sentence appended to a K/V fallback finding when, and only when, the
/// classifier confirmed a missing CUDA kernel that a source build adds.
pub const REBUILD_HINT: &str = " A source build with all flash-attention K/V pairs compiled would \
     add this kernel (llama.cpp Runtime panel → Build).";

#[cfg(test)]
mod tests {
    use super::*;

    fn clean() -> RuntimeFacts {
        RuntimeFacts {
            provenance_source: Some("foundry_source_build".into()),
            hashes_match: Some(true),
            journal: JournalFact::Clean,
            process_records_unresolved: false,
            legacy_backups: 0,
            prereq: None,
            busy: false,
            now_unix: 2_000_000_000,
        }
    }

    fn messages(facts: &RuntimeFacts) -> Vec<(DoctorSeverity, String)> {
        runtime_findings(facts)
            .into_iter()
            .map(|f| (f.severity, f.message))
            .collect()
    }

    #[test]
    fn a_healthy_runtime_has_no_findings() {
        assert!(runtime_findings(&clean()).is_empty());
    }

    #[test]
    fn a_runtime_that_was_never_built_from_source_has_no_prereq_noise() {
        let mut f = clean();
        f.provenance_source = Some("upstream_prebuilt".into());
        f.prereq = None;
        assert!(runtime_findings(&f).is_empty());
    }

    #[test]
    fn pending_recovery_is_an_issue_that_says_restart() {
        let mut f = clean();
        f.journal = JournalFact::Pending {
            label: "b11740 · src".into(),
        };
        let m = messages(&f);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, DoctorSeverity::Issue);
        assert!(m[0].1.contains("b11740 · src") && m[0].1.contains("Restart"));
    }

    #[test]
    fn a_corrupt_journal_is_an_issue_that_forbids_deleting_it() {
        let mut f = clean();
        f.journal = JournalFact::Corrupt {
            reason: "unreadable journal".into(),
        };
        let m = messages(&f);
        assert_eq!(m[0].0, DoctorSeverity::Issue);
        assert!(m[0].1.contains("do not delete"));
    }

    #[test]
    fn unresolved_process_records_block_and_are_never_to_be_deleted() {
        let mut f = clean();
        f.process_records_unresolved = true;
        let m = messages(&f);
        assert_eq!(m[0].0, DoctorSeverity::Issue);
        assert!(m[0].1.contains("processes.json") && m[0].1.contains("Do not delete"));
    }

    #[test]
    fn a_provenance_mismatch_is_a_warning_not_an_issue() {
        let mut f = clean();
        f.hashes_match = Some(false);
        let m = messages(&f);
        assert_eq!(m[0].0, DoctorSeverity::Warning);
        assert!(m[0].1.contains("custom"));
        // Unknown comparison (swap in progress, no manifest) says nothing.
        f.hashes_match = None;
        assert!(runtime_findings(&f).is_empty());
    }

    #[test]
    fn an_unavailable_capability_never_recommends_a_rebuild() {
        let f = capability_unavailable_finding();
        assert_eq!(f.severity, DoctorSeverity::Warning);
        assert!(f.message.contains("not blocked"));
        assert!(f.message.contains("A rebuild is not recommended"));
        assert!(!f.message.to_lowercase().contains("build with"));
    }

    #[test]
    fn a_running_build_or_update_is_not_reported_as_a_past_failure() {
        let mut f = clean();
        f.busy = true;
        f.journal = JournalFact::Pending { label: "b1".into() };
        f.process_records_unresolved = true;
        f.hashes_match = Some(false);
        assert!(runtime_findings(&f).is_empty());
    }

    #[test]
    fn a_prerequisite_failure_for_a_prebuilt_user_is_not_alarming() {
        let mut f = clean();
        f.provenance_source = Some("upstream_prebuilt".into());
        f.prereq = Some(PrereqFact {
            validated_at_unix: f.now_unix,
            failing: vec!["Ninja".into()],
        });
        let m = messages(&f);
        assert!(m[0].1.contains("only matters if you want to build"));
        assert!(!m[0].1.contains("disabled"));
    }

    #[test]
    fn legacy_backups_are_reported_with_a_count_and_never_deleted() {
        let mut f = clean();
        f.legacy_backups = 3;
        let m = messages(&f);
        assert!(m[0].1.starts_with("3 older"));
        assert!(m[0].1.contains("never deleted automatically"));
    }

    #[test]
    fn prerequisite_failures_and_staleness() {
        let mut f = clean();
        f.prereq = Some(PrereqFact {
            validated_at_unix: f.now_unix - 10,
            failing: vec!["Ninja".into(), "CMake".into()],
        });
        let m = messages(&f);
        assert!(m[0].1.contains("Ninja, CMake"));

        f.prereq = Some(PrereqFact {
            validated_at_unix: f.now_unix - PREREQ_STALE_SECS - 1,
            failing: vec![],
        });
        assert!(messages(&f)[0].1.contains("30 days"));

        f.prereq = Some(PrereqFact {
            validated_at_unix: f.now_unix - PREREQ_STALE_SECS,
            failing: vec![],
        });
        assert!(runtime_findings(&f).is_empty());
        // A report dated in the future is not stale.
        f.prereq = Some(PrereqFact {
            validated_at_unix: f.now_unix + 100,
            failing: vec![],
        });
        assert!(runtime_findings(&f).is_empty());
    }

    #[test]
    fn findings_are_llama_cpp_typed_with_a_section_and_no_auto_fix() {
        let mut f = clean();
        f.process_records_unresolved = true;
        let all = runtime_findings(&f);
        assert!(
            all.iter()
                .all(|x| x.finding_type == DoctorFindingType::LlamaCpp
                    && x.section == SECTION
                    && x.fix.is_none())
        );
    }
}
