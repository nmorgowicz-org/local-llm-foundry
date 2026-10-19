//! Running a whole build: ownership, the build phases on a blocking thread,
//! then the shared journaled promotion (plan §4.3, §4.4).
//!
//! Lock order (§4.3.1): toolchain → checkout/build/cache (the [`super::cache::Owner::Build`]
//! lease) → runtime gate → `server_lifecycle`. Compile and download never hold
//! the runtime gate; it is taken only for verify/swap/restart.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::llama_runtime::gate::{RuntimeGate, RuntimeOp};
use crate::llama_runtime::promote::{
    PromoteError, PromoteHost, PromoteOutcome, PromoteRequest, VerifyStage, promote,
};
use crate::llama_runtime::verify::{VerificationIntent, VerificationReport, verify_runtime};

use super::BuildOptions;
use super::cache::{Conflict, Ownership};
use super::job::{
    BuildFailure, JobDeps, JobEnd, JobRecord, JobStatus, Phase, SharedState, containment_clear,
    discard_unowned_staging, finish, new_job_id, new_shared, prune_records, recheck_before_journal,
    run_to_staged, start_exclusive, with_state,
};

/// A host with no model to stop: for the CLI and for custom server paths.
pub struct ModellessHost {
    pub cancel: Arc<AtomicBool>,
}

impl PromoteHost for ModellessHost {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    fn model_running(&self) -> bool {
        false
    }

    async fn stop_model(&self) -> Result<(), String> {
        Ok(())
    }

    async fn restart_model(&self) -> Result<(), String> {
        Ok(())
    }

    async fn verify(
        &self,
        dir: &Path,
        exe_name: &str,
        intent: &VerificationIntent,
        _stage: VerifyStage,
    ) -> Result<VerificationReport, String> {
        verify_runtime(dir, exe_name, intent)
            .await
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug)]
pub enum BuildStart {
    Busy(Conflict),
    RecoveryRequired(String),
}

#[derive(Debug, Clone)]
pub struct BuildResult {
    pub record: JobRecord,
}

fn promote_failure(shared: &SharedState, summary: String) -> BuildFailure {
    with_state(shared, |s| {
        let failure = BuildFailure {
            phase: s.record.phase,
            summary,
            command: None,
            exit_code: None,
            log_tail: s.log.tail(super::job::FAILURE_TAIL_LINES),
            first_error: None,
            hint: None,
            log_path: s.record.log_path.clone(),
        };
        s.record.failure = Some(failure.clone());
        failure
    })
}

/// Run one build end to end. Returns once the job is terminal.
///
/// `gate` serializes against every other runtime mutation; `ownership` makes
/// the build single-flight and exclusive with clean/validation/install.
///
/// Drive this future to completion, requesting cancellation through `cancel`.
/// The blocking worker retains a lease even if its observer is dropped, so
/// tree ownership cannot be released while a detached compiler is writing.
#[allow(clippy::too_many_arguments)]
pub async fn run_build<H: PromoteHost>(
    deps: JobDeps,
    host: &H,
    gate: &Arc<RuntimeGate>,
    ownership: &Ownership,
    options: BuildOptions,
    cancel: Arc<AtomicBool>,
    on_start: impl FnOnce(&str, &SharedState),
) -> Result<BuildResult, BuildStart> {
    // Do not even create a new job/log until prior writers are proven gone.
    if !containment_clear(&deps.paths) {
        return Err(BuildStart::RecoveryRequired(format!(
            "Unresolved process records at {}. Recovery is required before building: {}.",
            deps.paths.process_records.display(),
            super::repair::REPAIR_ACTION
        )));
    }
    let job_id = new_job_id().map_err(BuildStart::RecoveryRequired)?;
    let lease = start_exclusive(ownership, &job_id).map_err(BuildStart::Busy)?;
    if !containment_clear(&deps.paths) {
        return Err(BuildStart::RecoveryRequired(
            "Build containment changed while acquiring ownership.".into(),
        ));
    }
    let shared = match new_shared(&deps, &options, &job_id, true) {
        Ok(s) => s,
        Err(e) => {
            // Could not even create the log: report it without a persisted record.
            let s = new_shared(&deps, &options, &job_id, false).expect("in-memory job state");
            let _ = promote_failure(&s, format!("could not create the job log: {e}"));
            finish(&s, JobStatus::Failed, "could not create the job log");
            drop(lease);
            let record = with_state(&s, |s| s.record.clone());
            return Ok(BuildResult { record });
        }
    };
    on_start(&job_id, &shared);

    // Phases up to and including stage run synchronously on a blocking thread.
    let staged = {
        let (d, o, id, sh, c) = (
            deps.clone(),
            options.clone(),
            job_id.clone(),
            shared.clone(),
            cancel.clone(),
        );
        let worker_lease = lease.clone();
        tokio::task::spawn_blocking(move || {
            let _worker_lease = worker_lease;
            run_to_staged(&d, &o, &id, &sh, &|| c.load(Ordering::SeqCst))
        })
        .await
    };
    let source_dir = deps.paths.source.clone();

    // A failing step or panic may have left writers. Never discard their
    // diagnostics or source pin, even if the blocking task returned.
    if !containment_clear(&deps.paths) {
        let message = format!(
            "Build containment is unresolved; preserved the source pin and staging. Repair {} before reuse: {}.",
            deps.paths.process_records.display(),
            super::repair::REPAIR_ACTION
        );
        let _ = promote_failure(&shared, message.clone());
        finish(&shared, JobStatus::Failed, message);
        return Ok(BuildResult {
            record: with_state(&shared, |s| s.record.clone()),
        });
    }

    let staged = match staged {
        Ok(Ok(staged)) => staged,
        Ok(Err(end)) => {
            let (status, outcome) = match end {
                JobEnd::Cancelled => (
                    JobStatus::Cancelled,
                    "Cancelled. Nothing was installed.".to_string(),
                ),
                JobEnd::Failed(_) if cancel.load(Ordering::SeqCst) => (
                    JobStatus::Cancelled,
                    "Cancelled. Nothing was installed.".into(),
                ),
                JobEnd::Failed(f) => (JobStatus::Failed, f.summary),
            };
            // Ownership is released only after every subprocess is gone: the
            // step runner reaps before returning.
            deps.toolchain.unpin(&source_dir, &job_id);
            discard_unowned_staging(&deps, &deps.runtime.staging_dir(&job_id));
            finish(&shared, status, outcome);
            let _ = prune_records(&deps.paths, &[]);
            drop(lease);
            let record = with_state(&shared, |s| s.record.clone());
            return Ok(BuildResult { record });
        }
        Err(join) => {
            deps.toolchain.unpin(&source_dir, &job_id);
            discard_unowned_staging(&deps, &deps.runtime.staging_dir(&job_id));
            let _ = promote_failure(
                &shared,
                format!("the build task stopped unexpectedly: {join}"),
            );
            finish(
                &shared,
                JobStatus::Failed,
                "the build task stopped unexpectedly",
            );
            drop(lease);
            let record = with_state(&shared, |s| s.record.clone());
            return Ok(BuildResult { record });
        }
    };

    if cancel.load(Ordering::SeqCst) {
        discard_unowned_staging(&deps, &staged.dir);
        deps.toolchain.unpin(&source_dir, &job_id);
        finish(
            &shared,
            JobStatus::Cancelled,
            "Cancelled before promotion. Nothing was installed.",
        );
        return Ok(BuildResult {
            record: with_state(&shared, |s| s.record.clone()),
        });
    }
    // Promotion: recheck space immediately before the journal.
    if let Err(end) = recheck_before_journal(&deps, &staged, &shared) {
        deps.toolchain.unpin(&source_dir, &job_id);
        let (status, summary) = match end {
            JobEnd::Failed(f) => (JobStatus::Failed, f.summary),
            JobEnd::Cancelled => (JobStatus::Cancelled, "Cancelled before promotion.".into()),
        };
        finish(&shared, status, summary);
        drop(lease);
        let record = with_state(&shared, |s| s.record.clone());
        return Ok(BuildResult { record });
    }
    super::job::set_phase_public(&shared, Phase::Promote);

    let mut guard = match gate.try_begin(RuntimeOp::Promote) {
        Ok(g) => g,
        Err(busy) => {
            discard_unowned_staging(&deps, &staged.dir);
            deps.toolchain.unpin(&source_dir, &job_id);
            let msg = format!(
                "Another runtime operation ({}) is in progress. Nothing was installed.",
                busy.owner.label()
            );
            let _ = promote_failure(&shared, msg.clone());
            finish(&shared, JobStatus::Failed, msg);
            drop(lease);
            let record = with_state(&shared, |s| s.record.clone());
            return Ok(BuildResult { record });
        }
    };
    let exe = deps.toolchain.os().exe_name("llama-server");
    let request = PromoteRequest {
        staged_dir: staged.dir.clone(),
        label: staged.label.clone(),
        exe_name: exe,
        manifest: Some(staged.manifest.clone()),
        verification: staged.intent.clone(),
    };
    let result = promote(host, &mut guard, &deps.runtime, request).await;
    drop(guard);
    discard_unowned_staging(&deps, &staged.dir);
    if containment_clear(&deps.paths) {
        deps.toolchain.unpin(&source_dir, &job_id);
    }

    let (status, outcome) = match result {
        Ok(r) => match r.outcome {
            PromoteOutcome::Promoted { restarted } => (
                JobStatus::Succeeded,
                format!(
                    "Installed {}{}.",
                    staged.label,
                    if restarted {
                        " and restarted the model"
                    } else {
                        ""
                    }
                ),
            ),
            PromoteOutcome::InstalledRestartFailed { restart_error } => (
                JobStatus::SucceededRestartFailed,
                format!(
                    "Installed {}, but the model did not restart: {restart_error}",
                    staged.label
                ),
            ),
            PromoteOutcome::RolledBack {
                reason,
                restart_failed,
            } => {
                let _ = promote_failure(
                    &shared,
                    format!("The new runtime failed verification and was rolled back: {reason}"),
                );
                let text = match restart_failed {
                    None => format!("Rolled back: {reason}"),
                    Some(restart_error) => format!(
                        "Rolled back: {reason}. The previous runtime was restored but the model did not restart: {restart_error}"
                    ),
                };
                (JobStatus::Failed, text)
            }
        },
        Err(PromoteError::Cancelled) => (
            JobStatus::Cancelled,
            "Cancelled before the swap began. Nothing was installed.".into(),
        ),
        Err(e) => {
            let _ = promote_failure(&shared, e.to_string());
            (JobStatus::Failed, e.to_string())
        }
    };
    finish(&shared, status, outcome);
    let _ = prune_records(&deps.paths, &[]);
    drop(lease);
    let record = with_state(&shared, |s| s.record.clone());
    Ok(BuildResult { record })
}
