//! The in-process service behind the `/api/llama-build/*` routes (plan §7).
//!
//! The service owns what must outlive one HTTP request:
//!
//! - the [`Ownership`] table (build / clean / validation / install overlap),
//! - the active and the last build job (shared state, cancel flag),
//! - the single-flight prerequisite validation,
//! - short-lived caches that keep a polling panel from hammering the network
//!   (upstream refs) or the toolchain (fast prerequisite re-check).
//!
//! Everything machine-specific goes through [`BuildEnv`], so the routes can be
//! exercised against a fake toolchain on any host. Nothing here holds the
//! runtime gate: the build takes it only for verify/swap/restart inside
//! [`run_build`].
//!
//! Job logs are served from the job's log file through an incremental line
//! index, so a poll costs only the bytes written since the previous one and the
//! line numbers stay stable even after the in-memory ring has wrapped.

use std::collections::HashMap;
use std::io::{BufRead, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::llama_runtime::gate::RuntimeGate;
use crate::llama_runtime::promote::{JournalState, PromoteHost, RuntimePaths, read_journal};

use super::cache::{
    CleanError, CleanOptions, CleanReport, Conflict, Lease, LlamaBuildConfig, Owner, Ownership,
    cache_sizes, clean, dir_size,
};
use super::gpu;
use super::job::{
    JobDeps, JobRecord, JobStatus, Phase, SharedState, Toolchain, containment_clear, finish,
    read_record, with_state,
};
use super::orchestrate::{BuildStart, run_build};
use super::paths::{BuildPaths, list_job_records};
use super::prereqs::PrereqReport;
use super::prereqs_macos::validate_brew_packages;
use super::source::ResolvedRef;
use super::supervise::{Launcher, Line, SpawnSpec, record_for, run_contained_with, update_records};
use super::sysprobe::{RealSystem, SystemProbe};
use super::{Accelerator, BuildOptions, RefChoice, TargetOs, effective_parallelism};

/// How long `start` waits for the upstream revision to be pinned.
pub const PIN_WAIT: Duration = Duration::from_secs(20);
/// A fast prerequisite re-check result is reused for this long.
const RECHECK_TTL: Duration = Duration::from_secs(15);
/// Resolved upstream refs are reused for this long.
const REFS_TTL: Duration = Duration::from_secs(60);
/// Bound for one `brew install` run.
const BREW_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// A log page never carries more than this many lines or bytes.
const LOG_PAGE_LINES: usize = 2000;
const LOG_PAGE_BYTES: usize = 1024 * 1024;
const CHECKPOINT_EVERY: usize = 1000;
/// The line `JobLog` writes once when the per-job file cap is reached.
const TRUNCATION_MARKER: &str = "[log truncated:";
const MAX_INDEXED_LOGS: usize = 8;

/// Why a toolchain is being constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolchainMode {
    /// A real build.
    Build,
    /// The explicit "Validate" action: always re-runs the compile smoke test.
    Validate,
    /// Side-effect free: no smoke test, nothing persisted (fast re-check,
    /// upstream ref resolution).
    Probe,
}

pub struct ToolchainSpec {
    pub paths: BuildPaths,
    pub config: LlamaBuildConfig,
    pub cancel: Arc<AtomicBool>,
    pub mode: ToolchainMode,
}

/// Everything the service needs from the machine it runs on.
pub trait BuildEnv: Send + Sync {
    fn os(&self) -> TargetOs;
    fn logical_cpus(&self) -> usize;
    fn toolchain(&self, spec: ToolchainSpec) -> Arc<dyn Toolchain>;
    /// Run `brew install <packages>` with direct argv. Only called with
    /// allowlisted names on macOS, while holding install ownership.
    fn brew_install(
        &self,
        packages: &[&str],
        paths: &BuildPaths,
        owner_id: &str,
    ) -> Result<String, InstallFailure>;
    /// The CUDA architecture of the first detected NVIDIA GPU, when known.
    fn cuda_arch_auto(&self) -> Option<String>;
    /// The process table `repair-processes` consults before clearing records.
    fn process_lister(&self) -> Arc<dyn super::repair::ProcessLister> {
        Arc::new(super::repair::SysinfoLister)
    }
}

/// The real machine.
pub struct HostEnv;

impl BuildEnv for HostEnv {
    fn os(&self) -> TargetOs {
        TargetOs::host()
    }

    fn logical_cpus(&self) -> usize {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }

    fn toolchain(&self, spec: ToolchainSpec) -> Arc<dyn Toolchain> {
        Arc::new(
            super::real::RealToolchain::new(spec.paths, spec.config, Launcher::current_exe())
                .dry_run(spec.mode == ToolchainMode::Probe)
                .force_smoke(spec.mode == ToolchainMode::Validate)
                .with_cancel(spec.cancel),
        )
    }

    fn brew_install(
        &self,
        packages: &[&str],
        paths: &BuildPaths,
        owner_id: &str,
    ) -> Result<String, InstallFailure> {
        let brew = RealSystem.which("brew").ok_or_else(|| InstallFailure {
            output: "Homebrew was not found. Install it from https://brew.sh".into(),
            contained: true,
        })?;
        let mut args = vec!["install"];
        args.extend_from_slice(packages);
        run_installer(
            &brew,
            &args,
            BREW_TIMEOUT,
            &Launcher::current_exe(),
            &paths.process_records,
            owner_id,
        )
    }

    fn cuda_arch_auto(&self) -> Option<String> {
        gpu::detect_nvidia(&RealSystem)
            .ok()?
            .into_iter()
            .find_map(|g| g.cuda_architecture)
    }
}

/// An installer failure retains the cleanup proof separately from its output.
#[derive(Debug, Clone)]
pub struct InstallFailure {
    pub output: String,
    /// `false` means ownership and any process record must be retained.
    pub contained: bool,
}

const INSTALL_OUTPUT_BYTES: usize = 16 * 1024;

fn installer_env(vars: impl IntoIterator<Item = (String, String)>) -> Vec<(String, String)> {
    let mut env = super::toolchain_env::scrub_env(vars);
    env.insert("NONINTERACTIVE".into(), "1".into());
    env.insert("HOMEBREW_NO_ENV_HINTS".into(), "1".into());
    env.into_iter().collect()
}

fn append_install_output(output: &mut String, text: &str) {
    output.push_str(text);
    output.push('\n');
    if output.len() > INSTALL_OUTPUT_BYTES {
        let mut start = output.len() - INSTALL_OUTPUT_BYTES;
        while !output.is_char_boundary(start) {
            start += 1;
        }
        drop(output.drain(..start));
    }
}

/// Run an installer with direct argv, bounded output and a contained deadline.
///
/// The caller holds install ownership until this returns, retaining that lease
/// if `contained` is false. The supervisor also cleans up on app death; the
/// durable process record is removed only after successful containment.
/// Program and timeout are explicit so regressions need neither brew nor a
/// fifteen-minute wait. The deadline starts termination, including its grace.
pub fn run_installer(
    program: &Path,
    args: &[&str],
    timeout: Duration,
    launcher: &Launcher,
    process_records: &Path,
    owner_id: &str,
) -> Result<String, InstallFailure> {
    let mut spec = SpawnSpec::new(program, args);
    spec.env = Some(installer_env(std::env::vars()));
    // Establish a repair barrier before upstream code can write anything.
    // If recording the spawned identity fails, even a torn/empty barrier
    // still blocks reuse; never overwrite an earlier unresolved installation.
    let barrier = (|| -> std::io::Result<()> {
        use std::io::Write;
        if let Some(parent) = process_records.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(process_records)?;
        file.write_all(b"[]")?;
        file.sync_all()?;
        // Mirror the promote journal's durability: the barrier's directory
        // entry must survive power loss too, or brew may already be running
        // with no trace of ownership left on disk.
        if let Some(parent) = process_records.parent() {
            crate::llama_runtime::durable::sync_dir(parent)?;
        }
        Ok(())
    })();
    if let Err(error) = barrier {
        return Err(InstallFailure {
            output: format!("could not establish installer ownership: {error}"),
            contained: true,
        });
    }
    let mut output = String::new();
    let pid = std::cell::Cell::new(0);
    let started = Instant::now();
    let result = run_contained_with(
        &spec,
        launcher,
        &mut |line| {
            let (Line::Stdout(text) | Line::Stderr(text)) = line;
            append_install_output(&mut output, &text);
        },
        &|| started.elapsed() >= timeout,
        &mut |p| {
            pid.set(p);
            let record = record_for(owner_id, p).ok_or_else(|| {
                std::io::Error::other("could not establish installer process identity")
            })?;
            update_records(process_records, |all| all.push(record))
        },
    );
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            append_install_output(&mut output, &format!("installer failed: {error}"));
            if pid.get() == 0 {
                // spawn failed: nothing ever inherited the output/control pipes.
                let _ = std::fs::remove_file(process_records);
            }
            return Err(InstallFailure {
                output,
                // A post-spawn I/O error is not proof that cleanup succeeded.
                contained: pid.get() == 0,
            });
        }
    };
    if !outcome.contained {
        append_install_output(
            &mut output,
            "installer descendants could not be contained; repair is required",
        );
        return Err(InstallFailure {
            output,
            contained: false,
        });
    }
    if let Err(error) = update_records(process_records, |all| {
        all.retain(|record| record.pid != pid.get());
    }) {
        append_install_output(
            &mut output,
            &format!("could not clear installer record: {error}"),
        );
        return Err(InstallFailure {
            output,
            contained: true,
        });
    }
    if outcome.cancelled {
        append_install_output(
            &mut output,
            &format!("installer timed out after {timeout:?}"),
        );
    } else if outcome.exit_code == Some(0) {
        return Ok(output);
    }
    Err(InstallFailure {
        output,
        contained: true,
    })
}

/// Paths and configuration for one request, resolved by the route layer.
#[derive(Clone)]
pub struct BuildContext {
    pub paths: BuildPaths,
    pub runtime: RuntimePaths,
    /// Where `llama-build.json` lives.
    pub config_dir: PathBuf,
    /// Used when `llama-build.json` cannot be read.
    pub fallback_config: LlamaBuildConfig,
}

impl BuildContext {
    /// Fresh from disk so a changed cap applies to the next build without a
    /// restart (`AppConfig` itself is immutable once the app is running).
    pub fn config(&self) -> LlamaBuildConfig {
        LlamaBuildConfig::load(&self.config_dir).unwrap_or_else(|_| self.fallback_config.clone())
    }

    /// Why the build/promote machinery must be repaired before it is reused,
    /// when it must. Only meaningful while nothing is running: a live build
    /// legitimately owns process records and a journal.
    pub fn recovery_required(&self) -> Option<String> {
        if !containment_clear(&self.paths) {
            return Some(format!(
                "Unresolved build process records at {}. Recovery is required before building: {}.",
                self.paths.process_records.display(),
                super::repair::REPAIR_ACTION
            ));
        }
        match read_journal(&self.runtime) {
            JournalState::None => None,
            JournalState::Present(_) => Some(
                "An interrupted runtime transaction is pending recovery. Restart the app to recover it."
                    .into(),
            ),
            JournalState::Corrupt(reason) => Some(format!(
                "The runtime transaction journal cannot be read ({reason}). Recovery is required before building."
            )),
        }
    }
}

/// Stable label for an ownership holder in API bodies.
pub fn owner_name(owner: Owner) -> &'static str {
    match owner {
        Owner::Build => "build",
        Owner::Clean => "clean",
        Owner::Validation => "validation",
        Owner::Check => "check",
        Owner::Install => "install",
        Owner::Repair => "repair",
    }
}

pub fn default_accelerator(os: TargetOs) -> Accelerator {
    match os {
        TargetOs::Macos => Accelerator::Metal,
        TargetOs::Windows => Accelerator::Cuda,
        TargetOs::Linux => Accelerator::Cpu,
    }
}

/// `^[0-9a-f]{32}$`: exactly what `new_job_id` produces. Anything else never
/// reaches the filesystem.
pub fn is_job_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[derive(Clone)]
struct JobHandle {
    id: String,
    shared: SharedState,
    cancel: Arc<AtomicBool>,
}

struct ValidateHandle {
    id: String,
    accelerator: Accelerator,
}

#[derive(Default)]
struct Inner {
    active: Option<JobHandle>,
    last: Option<JobHandle>,
    validate: Option<ValidateHandle>,
    retained_install: Option<Lease>,
    rechecks: HashMap<Accelerator, (Instant, String)>,
    refs: HashMap<&'static str, (Instant, ResolvedRef)>,
    logs: HashMap<String, LogIndex>,
}

pub struct BuildService {
    env: Arc<dyn BuildEnv>,
    ownership: Arc<Ownership>,
    inner: Mutex<Inner>,
    pin_wait: Duration,
}

impl Default for BuildService {
    fn default() -> Self {
        Self::new(Arc::new(HostEnv))
    }
}

#[derive(Debug)]
pub enum StartError {
    Busy(Conflict),
    RecoveryRequired(String),
    /// The job could not even begin (for example its log could not be created).
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Started {
    pub job_id: String,
    pub resolved_commit: Option<String>,
}

#[derive(Debug)]
pub enum ValidateError {
    Busy(Conflict),
    Failed(String),
}

#[derive(Debug)]
pub enum FixError {
    Unsupported(String),
    Invalid(String),
    Busy(Conflict),
    Failed { error: String, output: String },
}

#[derive(Debug, Clone)]
pub struct FixOutcome {
    pub installed: Vec<String>,
    pub output: String,
}

#[derive(Debug)]
pub enum RepairFailure {
    /// A build, validation, clean or install owns the tree.
    Busy(Conflict),
    Refused(super::repair::RepairError),
}

#[derive(Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    Requested,
    NotCancellable(&'static str),
    Unknown,
}

pub const NOT_CANCELLABLE_PROMOTING: &str = "Installing/recovering — cannot cancel safely";
pub const NOT_CANCELLABLE_FINISHED: &str = "The job has already finished";

/// A job record plus what only the live service knows about it.
#[derive(Debug, Clone)]
pub struct JobView {
    pub record: JobRecord,
    pub cancellable: bool,
}

#[derive(Debug, Clone)]
pub struct PrereqView {
    pub report: Option<PrereqReport>,
    pub stale: bool,
    pub validating: bool,
    pub validate_job_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RefsView {
    pub latest_release: Option<ResolvedRef>,
    pub master: Option<ResolvedRef>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CacheView {
    pub source_bytes: u64,
    pub ccache_bytes: u64,
    pub ccache_cap_bytes: u64,
    pub build_trees: Vec<(Accelerator, u64)>,
    pub previous_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogPage {
    pub lines: Vec<String>,
    pub next: usize,
    pub truncated: bool,
}

type Ack = Result<AckOk, BuildStart>;

enum AckOk {
    Started(String, SharedState),
    /// `run_build` finished without ever starting a job.
    Finished(Box<JobRecord>),
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl BuildService {
    pub fn new(env: Arc<dyn BuildEnv>) -> Self {
        Self {
            env,
            ownership: Arc::new(Ownership::default()),
            inner: Mutex::new(Inner::default()),
            pin_wait: PIN_WAIT,
        }
    }

    pub fn with_pin_wait(mut self, wait: Duration) -> Self {
        self.pin_wait = wait;
        self
    }

    pub fn env(&self) -> &Arc<dyn BuildEnv> {
        &self.env
    }

    pub fn ownership(&self) -> &Ownership {
        &self.ownership
    }

    /// Who currently owns the build directories. A read-only prerequisite
    /// check is too short-lived to be worth reporting.
    pub fn owner(&self) -> Option<Owner> {
        self.ownership
            .holders()
            .into_iter()
            .map(|(o, _)| o)
            .find(|o| *o != Owner::Check)
    }

    /// Make `shared` the active job without running one, so route tests can
    /// place a job at an exact phase.
    #[cfg(test)]
    pub(crate) fn register_for_test(&self, id: &str, shared: SharedState, cancel: Arc<AtomicBool>) {
        lock(&self.inner).active = Some(JobHandle {
            id: id.to_string(),
            shared,
            cancel,
        });
    }

    pub fn active_job_id(&self) -> Option<String> {
        lock(&self.inner).active.as_ref().map(|h| h.id.clone())
    }

    pub fn is_idle(&self) -> bool {
        self.ownership.holders().is_empty()
    }

    pub fn installer_recovery_required(&self) -> bool {
        lock(&self.inner).retained_install.is_some()
    }

    /// What blocks starting a build right now: any holder but a read-only
    /// check. (The ownership table agrees: a validation writes the same
    /// process records and report, so it never overlaps a build.)
    pub fn conflict_for_build(&self) -> Option<Conflict> {
        self.ownership
            .holders()
            .into_iter()
            .find(|(o, _)| *o != Owner::Check)
            .map(|(holder, holder_id)| Conflict { holder, holder_id })
    }

    // ------------------------------------------------------------- build

    /// Start a build and return once its upstream revision is pinned (or the
    /// wait elapses, or the job ended). The build keeps running afterwards.
    pub async fn start_build<H>(
        self: &Arc<Self>,
        ctx: &BuildContext,
        host: H,
        cancel: Arc<AtomicBool>,
        gate: Arc<RuntimeGate>,
        options: BuildOptions,
    ) -> Result<Started, StartError>
    where
        H: PromoteHost + 'static,
    {
        let config = ctx.config();
        let toolchain = self.env.toolchain(ToolchainSpec {
            paths: ctx.paths.clone(),
            config: config.clone(),
            cancel: cancel.clone(),
            mode: ToolchainMode::Build,
        });
        let deps = JobDeps {
            toolchain,
            paths: ctx.paths.clone(),
            runtime: ctx.runtime.clone(),
            config,
        };
        let (tx, rx) = tokio::sync::oneshot::channel::<Ack>();
        let tx = Arc::new(Mutex::new(Some(tx)));
        let registered: Arc<Mutex<Option<JobHandle>>> = Arc::new(Mutex::new(None));

        let svc = Arc::clone(self);
        let task_tx = tx.clone();
        let task_registered = registered.clone();
        tokio::spawn(async move {
            let run = tokio::spawn({
                let svc = svc.clone();
                let tx = task_tx.clone();
                let registered = task_registered.clone();
                let cancel = cancel.clone();
                async move {
                    let result = run_build(
                        deps,
                        &host,
                        &gate,
                        &svc.ownership,
                        options,
                        cancel.clone(),
                        |id, shared| {
                            let handle = JobHandle {
                                id: id.to_string(),
                                shared: shared.clone(),
                                cancel: cancel.clone(),
                            };
                            lock(&svc.inner).active = Some(handle.clone());
                            *lock(&registered) = Some(handle);
                            if let Some(tx) = lock(&tx).take() {
                                let _ = tx.send(Ok(AckOk::Started(id.to_string(), shared.clone())));
                            }
                        },
                    )
                    .await;
                    // Reaching here without an acknowledgement means the job
                    // never started: report why.
                    if let Some(tx) = lock(&tx).take() {
                        let _ = tx.send(match result {
                            Err(e) => Err(e),
                            Ok(done) => Ok(AckOk::Finished(Box::new(done.record))),
                        });
                    }
                }
            });
            let joined = run.await;
            let handle = lock(&task_registered).take();
            if let Some(handle) = &handle {
                if joined.is_err() {
                    let running = with_state(&handle.shared, |s| !s.record.status.is_terminal());
                    if running {
                        finish(
                            &handle.shared,
                            JobStatus::Failed,
                            "the build task stopped unexpectedly",
                        );
                    }
                }
                let mut inner = lock(&svc.inner);
                if inner.active.as_ref().is_some_and(|a| a.id == handle.id) {
                    inner.last = inner.active.take();
                }
                inner.rechecks.clear();
            }
            if let Some(tx) = lock(&task_tx).take() {
                let _ = tx.send(Err(BuildStart::RecoveryRequired(
                    "the build task stopped before it started".into(),
                )));
            }
        });

        let ack = match tokio::time::timeout(Duration::from_secs(30), rx).await {
            Ok(Ok(ack)) => ack,
            _ => return Err(StartError::Failed("the build did not start".into())),
        };
        let (job_id, shared) = match ack {
            Ok(AckOk::Started(id, shared)) => (id, shared),
            Ok(AckOk::Finished(record)) => {
                return Err(StartError::Failed(
                    record
                        .outcome
                        .unwrap_or_else(|| "the build could not start".into()),
                ));
            }
            Err(BuildStart::Busy(conflict)) => return Err(StartError::Busy(conflict)),
            Err(BuildStart::RecoveryRequired(why)) => {
                return Err(StartError::RecoveryRequired(why));
            }
        };

        let deadline = Instant::now() + self.pin_wait;
        let resolved_commit = loop {
            let (resolved, terminal) = with_state(&shared, |s| {
                (s.record.resolved.clone(), s.record.status.is_terminal())
            });
            if let Some(r) = resolved {
                break Some(r.commit);
            }
            if terminal || Instant::now() >= deadline {
                break None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        Ok(Started {
            job_id,
            resolved_commit,
        })
    }

    /// Ask a running job to stop. Never succeeds from the promote phase on.
    pub fn cancel(&self, ctx: &BuildContext, id: &str) -> CancelOutcome {
        // The id reaches the filesystem below; never trust the caller to have
        // checked it.
        if !is_job_id(id) {
            return CancelOutcome::Unknown;
        }
        let inner = lock(&self.inner);
        if let Some(active) = inner.active.as_ref().filter(|a| a.id == id) {
            let (status, phase) = with_state(&active.shared, |s| (s.record.status, s.record.phase));
            if status.is_terminal() {
                return CancelOutcome::NotCancellable(NOT_CANCELLABLE_FINISHED);
            }
            if phase >= Phase::Promote {
                return CancelOutcome::NotCancellable(NOT_CANCELLABLE_PROMOTING);
            }
            active.cancel.store(true, Ordering::SeqCst);
            return CancelOutcome::Requested;
        }
        let known = inner.last.as_ref().is_some_and(|l| l.id == id)
            || read_record(&ctx.paths.record_file(id)).is_some();
        if known {
            CancelOutcome::NotCancellable(NOT_CANCELLABLE_FINISHED)
        } else {
            CancelOutcome::Unknown
        }
    }

    /// The job, from memory when this process ran it, else from its record.
    pub fn job(&self, ctx: &BuildContext, id: &str) -> Option<JobView> {
        if !is_job_id(id) {
            return None;
        }
        let (handle, is_active) = {
            let inner = lock(&self.inner);
            match (&inner.active, &inner.last) {
                (Some(a), _) if a.id == id => (Some(a.clone()), true),
                (_, Some(l)) if l.id == id => (Some(l.clone()), false),
                _ => (None, false),
            }
        };
        if let Some(handle) = handle {
            let record = with_state(&handle.shared, |s| s.record.clone());
            let cancellable =
                is_active && record.status == JobStatus::Running && record.phase < Phase::Promote;
            return Some(JobView {
                record,
                cancellable,
            });
        }
        let mut record = read_record(&ctx.paths.record_file(id))?;
        // Nothing in this process runs it: a record still `running` belongs to
        // a run that died (startup recovery marks it on the next launch).
        if record.status == JobStatus::Running {
            record.status = JobStatus::Interrupted;
        }
        Some(JobView {
            record,
            cancellable: false,
        })
    }

    /// The newest finished job (never the active one).
    pub fn last_job(&self, ctx: &BuildContext) -> Option<JobRecord> {
        let (active_id, last) = {
            let inner = lock(&self.inner);
            (
                inner.active.as_ref().map(|a| a.id.clone()),
                inner.last.clone(),
            )
        };
        let memory = last.map(|h| with_state(&h.shared, |s| s.record.clone()));
        let disk = list_job_records(&ctx.paths.logs)
            .ok()
            .into_iter()
            .flatten()
            .filter(|pair| Some(&pair.id) != active_id.as_ref())
            .find_map(|pair| pair.record.as_deref().and_then(read_record));
        let mut record = [memory, disk]
            .into_iter()
            .flatten()
            .max_by_key(|r| r.started_unix)?;
        if record.status == JobStatus::Running {
            record.status = JobStatus::Interrupted;
        }
        Some(record)
    }

    // --------------------------------------------------------------- log

    pub fn read_log(&self, ctx: &BuildContext, id: &str, after: usize) -> Option<LogPage> {
        if !is_job_id(id) {
            return None;
        }
        let path = ctx.paths.log_file(id);
        let memory = {
            let inner = lock(&self.inner);
            [inner.active.as_ref(), inner.last.as_ref()]
                .into_iter()
                .flatten()
                .find(|h| h.id == id)
                .map(|h| h.shared.clone())
        };
        if !path.is_file() {
            // No file (it could not be created): the in-memory ring is all
            // there is.
            let shared = memory?;
            let (lines, truncated) = with_state(&shared, |s| (s.log.all(), s.log.truncated()));
            let start = after.min(lines.len());
            return Some(LogPage {
                next: lines.len(),
                lines: lines[start..].to_vec(),
                truncated,
            });
        }
        let mut inner = lock(&self.inner);
        if inner.logs.len() >= MAX_INDEXED_LOGS && !inner.logs.contains_key(id) {
            // Active and recent jobs are the only ones polled; drop an
            // arbitrary old index rather than grow without bound.
            if let Some(victim) = inner.logs.keys().next().cloned() {
                inner.logs.remove(&victim);
            }
        }
        let index = inner.logs.entry(id.to_string()).or_default();
        index.refresh(&path).ok()?;
        let (lines, next) = index.page(&path, after).ok()?;
        Some(LogPage {
            lines,
            next,
            truncated: file_is_truncated(&path),
        })
    }

    // ---------------------------------------------------------- prereqs

    pub fn validating(&self, accelerator: Accelerator) -> Option<String> {
        lock(&self.inner)
            .validate
            .as_ref()
            .filter(|v| v.accelerator == accelerator)
            .map(|v| v.id.clone())
    }

    /// Run the full prerequisite report (including the compile smoke test) in
    /// the background. Single-flight and exclusive with build, clean and
    /// install: a build's own preflight shares the toolchain caches and the
    /// persisted report, so the two never overlap here.
    pub fn start_validation(
        self: &Arc<Self>,
        ctx: &BuildContext,
        accelerator: Accelerator,
    ) -> Result<String, ValidateError> {
        let id = super::job::new_job_id().map_err(ValidateError::Failed)?;
        // One atomic acquisition: the ownership table itself refuses a
        // validation beside a build, so there is no check-then-act window.
        let lease = self
            .ownership
            .try_acquire(Owner::Validation, &id)
            .map_err(ValidateError::Busy)?;
        lock(&self.inner).validate = Some(ValidateHandle {
            id: id.clone(),
            accelerator,
        });
        let toolchain = self.env.toolchain(ToolchainSpec {
            paths: ctx.paths.clone(),
            config: ctx.config(),
            cancel: Arc::new(AtomicBool::new(false)),
            mode: ToolchainMode::Validate,
        });
        let svc = Arc::clone(self);
        let job = id.clone();
        tokio::task::spawn_blocking(move || {
            // Release the single-flight marker even if the toolchain panics.
            struct Done<'a>(&'a BuildService, String);
            impl Drop for Done<'_> {
                fn drop(&mut self) {
                    let mut inner = lock(&self.0.inner);
                    if inner.validate.as_ref().is_some_and(|v| v.id == self.1) {
                        inner.validate = None;
                    }
                    inner.rechecks.clear();
                }
            }
            let _lease = lease;
            let _done = Done(&svc, job);
            let options = BuildOptions::new(accelerator, RefChoice::LatestTag);
            let _ = toolchain.prereqs(&options);
        });
        Ok(id)
    }

    /// The cached report plus whether it still describes this machine. Blocks:
    /// call from `spawn_blocking`.
    pub fn prereq_view(&self, ctx: &BuildContext, accelerator: Accelerator) -> PrereqView {
        let validate_job_id = self.validating(accelerator);
        let report: Option<PrereqReport> = std::fs::read(&ctx.paths.prereqs_last)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .filter(|r: &PrereqReport| r.accelerator == accelerator);
        let stale = match &report {
            None => true,
            Some(r) => self
                .recheck_fingerprint(ctx, accelerator)
                .is_some_and(|fp| fp != r.fingerprint),
        };
        PrereqView {
            report,
            stale,
            validating: validate_job_id.is_some(),
            validate_job_id,
        }
    }

    /// A fast, side-effect-free fingerprint of this machine's toolchain, or
    /// `None` when one cannot be taken now (something owns the toolchain) or
    /// cannot be compared (Windows: the SDK facts only a full pass captures).
    fn recheck_fingerprint(&self, ctx: &BuildContext, accelerator: Accelerator) -> Option<String> {
        if self.env.os() == TargetOs::Windows {
            return None;
        }
        if let Some((at, fp)) = lock(&self.inner).rechecks.get(&accelerator)
            && at.elapsed() < RECHECK_TTL
        {
            return Some(fp.clone());
        }
        // Build and validation write process records the probe would trip
        // over; an install is changing the tools themselves.
        if !self.is_idle() {
            return None;
        }
        let id = super::job::new_job_id().ok()?;
        let _lease = self.ownership.try_acquire(Owner::Check, &id).ok()?;
        let toolchain = self.env.toolchain(ToolchainSpec {
            paths: ctx.paths.clone(),
            config: ctx.config(),
            cancel: Arc::new(AtomicBool::new(false)),
            mode: ToolchainMode::Probe,
        });
        let report = toolchain.prereqs(&BuildOptions::new(accelerator, RefChoice::LatestTag));
        lock(&self.inner)
            .rechecks
            .insert(accelerator, (Instant::now(), report.fingerprint.clone()));
        Some(report.fingerprint)
    }

    /// `brew install` for allowlisted formulae (macOS only).
    pub fn brew_install(
        &self,
        ctx: &BuildContext,
        packages: &[String],
    ) -> Result<FixOutcome, FixError> {
        if self.env.os() != TargetOs::Macos {
            return Err(FixError::Unsupported(
                "one-click installs are only available on macOS".into(),
            ));
        }
        let allowed = validate_brew_packages(packages).map_err(FixError::Invalid)?;
        let id = super::job::new_job_id().map_err(|e| FixError::Failed {
            error: e,
            output: String::new(),
        })?;
        let lease = self
            .ownership
            .try_acquire(Owner::Install, &id)
            .map_err(FixError::Busy)?;
        if !containment_clear(&ctx.paths) {
            return Err(FixError::Failed {
                error: "unresolved process records block installation; repair is required".into(),
                output: String::new(),
            });
        }
        let result = self.env.brew_install(&allowed, &ctx.paths, &id);
        if result.as_ref().is_err_and(|error| !error.contained) {
            // Match build orchestration's fail-closed rule: no mutation of the
            // toolchain is safe while an unresolved installer may still write.
            lock(&self.inner).retained_install = Some(lease);
        } else {
            drop(lease);
        }
        lock(&self.inner).rechecks.clear();
        match result {
            Ok(output) => Ok(FixOutcome {
                installed: allowed.iter().map(|p| p.to_string()).collect(),
                output,
            }),
            Err(failure) => Err(FixError::Failed {
                error: if failure.contained {
                    "brew install failed".into()
                } else {
                    "brew install containment unresolved; repair is required".into()
                },
                output: failure.output,
            }),
        }
    }

    // ----------------------------------------------------------- options

    /// Upstream refs, cached briefly so a polling panel does not hammer the
    /// remote. Blocks: call from `spawn_blocking`.
    pub fn refs(&self, ctx: &BuildContext) -> RefsView {
        let mut errors = Vec::new();
        let mut resolved: [Option<ResolvedRef>; 2] = [None, None];
        let mut toolchain: Option<Arc<dyn Toolchain>> = None;
        for (slot, (key, choice)) in [
            ("latest_release", RefChoice::LatestTag),
            ("master", RefChoice::Master),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some((at, r)) = lock(&self.inner).refs.get(key)
                && at.elapsed() < REFS_TTL
            {
                resolved[slot] = Some(r.clone());
                continue;
            }
            let tc = toolchain.get_or_insert_with(|| {
                self.env.toolchain(ToolchainSpec {
                    paths: ctx.paths.clone(),
                    config: ctx.config(),
                    cancel: Arc::new(AtomicBool::new(false)),
                    mode: ToolchainMode::Probe,
                })
            });
            let options = BuildOptions::new(default_accelerator(self.env.os()), choice);
            match tc.resolve(&options) {
                Ok(r) => {
                    lock(&self.inner)
                        .refs
                        .insert(key, (Instant::now(), r.clone()));
                    resolved[slot] = Some(r);
                }
                Err(e) => errors.push(format!("{key}: {e}")),
            }
        }
        let [latest_release, master] = resolved;
        RefsView {
            latest_release,
            master,
            error: (!errors.is_empty()).then(|| errors.join("; ")),
        }
    }

    pub fn jobs_auto(&self, accelerator: Accelerator) -> usize {
        effective_parallelism(
            &BuildOptions::new(accelerator, RefChoice::LatestTag),
            self.env.logical_cpus(),
        )
    }

    // ------------------------------------------------------------- cache

    pub fn cache_view(&self, ctx: &BuildContext) -> CacheView {
        let sizes = cache_sizes(&ctx.paths, ctx.config().ccache_max_bytes);
        CacheView {
            source_bytes: sizes.source_bytes,
            ccache_bytes: sizes.ccache_bytes,
            ccache_cap_bytes: sizes.ccache_cap_bytes,
            build_trees: sizes.build_trees,
            previous_bytes: dir_size(&ctx.runtime.previous),
        }
    }

    /// Clear the process records a crash left behind, once nothing alive can
    /// still own the directory. Exclusive with every other operation. Blocks:
    /// call from `spawn_blocking`.
    pub fn repair_processes(
        &self,
        ctx: &BuildContext,
    ) -> Result<super::repair::RepairReport, RepairFailure> {
        let id = super::job::new_job_id()
            .map_err(|e| RepairFailure::Refused(super::repair::RepairError::Io(e)))?;
        // A quarantined install is no longer executing in this service, but
        // its descendants may still write. Keep that exclusive lease alive
        // while repair proves liveness and clears the durable barrier.
        struct RestoreInstall<'a> {
            inner: &'a Mutex<Inner>,
            lease: Option<Lease>,
        }
        impl Drop for RestoreInstall<'_> {
            fn drop(&mut self) {
                if let Some(lease) = self.lease.take() {
                    lock(self.inner).retained_install = Some(lease);
                }
            }
        }
        let mut retained = RestoreInstall {
            inner: &self.inner,
            lease: lock(&self.inner).retained_install.take(),
        };
        let lease = if retained.lease.is_none() {
            Some(
                self.ownership
                    .try_acquire(Owner::Repair, &id)
                    .map_err(RepairFailure::Busy)?,
            )
        } else {
            None
        };
        let lister = self.env.process_lister();
        let result = super::repair::repair_process_records(&ctx.paths, lister.as_ref());
        if result.is_ok() {
            drop(retained.lease.take());
        }
        drop(lease);
        lock(&self.inner).rechecks.clear();
        result.map_err(RepairFailure::Refused)
    }

    /// Remove inactive caches. Exclusive with build, validation and install.
    pub fn clean(
        &self,
        ctx: &BuildContext,
        options: CleanOptions,
    ) -> Result<CleanReport, CleanError> {
        let id = super::job::new_job_id().map_err(CleanError::Io)?;
        let lease = self
            .ownership
            .try_acquire(Owner::Clean, &id)
            .map_err(CleanError::Busy)?;
        let result = clean(&ctx.paths, options, &self.ownership);
        drop(lease);
        lock(&self.inner).rechecks.clear();
        result
    }
}

// ---------------------------------------------------------------------------
// Log index
// ---------------------------------------------------------------------------

/// Byte offsets of every `CHECKPOINT_EVERY`-th line start in a log file, kept
/// incrementally so a page costs the bytes written since the last poll plus at
/// most one checkpoint interval of skipping.
#[derive(Default)]
struct LogIndex {
    /// `checkpoints[k]` = offset where line `k * CHECKPOINT_EVERY` starts.
    checkpoints: Vec<u64>,
    /// Offset just past the last complete line scanned.
    scanned: u64,
    lines: usize,
}

impl LogIndex {
    fn refresh(&mut self, path: &Path) -> std::io::Result<()> {
        let mut file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        if len < self.scanned {
            *self = Self::default();
        }
        if self.checkpoints.is_empty() {
            self.checkpoints.push(0);
        }
        file.seek(SeekFrom::Start(self.scanned))?;
        let mut remaining = len - self.scanned;
        let mut pos = self.scanned;
        let mut buf = vec![0u8; 64 * 1024];
        while remaining > 0 {
            let want = buf.len().min(remaining as usize);
            let n = file.read(&mut buf[..want])?;
            if n == 0 {
                break;
            }
            for (i, byte) in buf[..n].iter().enumerate() {
                if *byte == b'\n' {
                    self.lines += 1;
                    self.scanned = pos + i as u64 + 1;
                    if self.lines.is_multiple_of(CHECKPOINT_EVERY) {
                        self.checkpoints.push(self.scanned);
                    }
                }
            }
            pos += n as u64;
            remaining -= n as u64;
        }
        Ok(())
    }

    /// Complete lines from `after` (clamped to what exists) and the next cursor.
    fn page(&self, path: &Path, after: usize) -> std::io::Result<(Vec<String>, usize)> {
        let after = after.min(self.lines);
        let checkpoint = after / CHECKPOINT_EVERY;
        let start = self.checkpoints.get(checkpoint).copied().unwrap_or(0);
        let mut file = std::fs::File::open(path)?;
        file.seek(SeekFrom::Start(start))?;
        let mut reader = std::io::BufReader::new(file.take(self.scanned.saturating_sub(start)));
        let mut lines = Vec::new();
        let mut bytes = 0usize;
        let mut index = checkpoint * CHECKPOINT_EVERY;
        let mut raw = Vec::new();
        while reader.read_until(b'\n', &mut raw)? > 0 {
            if raw.last() != Some(&b'\n') {
                break;
            }
            if index >= after {
                if lines.len() >= LOG_PAGE_LINES || bytes >= LOG_PAGE_BYTES {
                    break;
                }
                let text = String::from_utf8_lossy(&raw);
                let text = text.trim_end_matches(['\n', '\r']).to_string();
                bytes += text.len();
                lines.push(text);
            }
            index += 1;
            raw.clear();
        }
        let next = after + lines.len();
        Ok((lines, next))
    }
}

/// The truncation notice is the last thing `JobLog` writes to a file.
fn file_is_truncated(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(len) = file.metadata().map(|m| m.len()) else {
        return false;
    };
    let start = len.saturating_sub(512);
    let mut tail = Vec::new();
    if file.seek(SeekFrom::Start(start)).is_err() || file.read_to_end(&mut tail).is_err() {
        return false;
    }
    String::from_utf8_lossy(&tail).contains(TRUNCATION_MARKER)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_lines(path: &Path, count: usize) {
        let mut text = String::new();
        for i in 0..count {
            text.push_str(&format!("line {i}\n"));
        }
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn job_ids_are_exactly_32_lowercase_hex() {
        assert!(is_job_id(&"a1".repeat(16)));
        assert!(!is_job_id("../../etc/passwd"));
        assert!(!is_job_id(&"A1".repeat(16)));
        assert!(!is_job_id(&"a1".repeat(15)));
        assert!(!is_job_id(&format!("{}/", "a".repeat(31))));
        assert!(!is_job_id(""));
    }

    #[test]
    fn the_log_index_pages_across_checkpoints_and_clamps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.log");
        write_lines(&path, 2500);
        let mut index = LogIndex::default();
        index.refresh(&path).unwrap();
        assert_eq!(index.lines, 2500);
        assert_eq!(index.checkpoints.len(), 3);

        let (lines, next) = index.page(&path, 0).unwrap();
        assert_eq!((lines.len(), next), (LOG_PAGE_LINES, LOG_PAGE_LINES));
        assert_eq!(lines[0], "line 0");

        // A cursor inside a later checkpoint interval lands on the right line.
        let (lines, next) = index.page(&path, 2001).unwrap();
        assert_eq!(lines.first().map(String::as_str), Some("line 2001"));
        assert_eq!(next, 2500);

        // A cursor ahead of the file is clamped, never an error.
        let (lines, next) = index.page(&path, 9999).unwrap();
        assert!(lines.is_empty());
        assert_eq!(next, 2500);
    }

    #[test]
    fn the_log_index_is_incremental_and_ignores_a_partial_trailing_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.log");
        std::fs::write(&path, "one\ntwo\npart").unwrap();
        let mut index = LogIndex::default();
        index.refresh(&path).unwrap();
        let (lines, next) = index.page(&path, 0).unwrap();
        assert_eq!(lines, ["one", "two"]);
        assert_eq!(next, 2);

        // The writer finishes the partial line and adds another.
        std::fs::write(&path, "one\ntwo\npartial\nfour\n").unwrap();
        index.refresh(&path).unwrap();
        let (lines, next) = index.page(&path, next).unwrap();
        assert_eq!(lines, ["partial", "four"]);
        assert_eq!(next, 4);
    }

    #[test]
    fn the_log_index_rebuilds_when_the_file_shrinks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.log");
        write_lines(&path, 1200);
        let mut index = LogIndex::default();
        index.refresh(&path).unwrap();
        std::fs::write(&path, "fresh\n").unwrap();
        index.refresh(&path).unwrap();
        assert_eq!(index.lines, 1);
        assert_eq!(index.page(&path, 0).unwrap().0, ["fresh"]);
    }

    #[test]
    fn truncation_is_read_from_the_end_of_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.log");
        std::fs::write(&path, "a\nb\n").unwrap();
        assert!(!file_is_truncated(&path));
        std::fs::write(
            &path,
            "a\n[log truncated: the 64 MiB per-job limit was reached; the live view continues]\n",
        )
        .unwrap();
        assert!(file_is_truncated(&path));
    }

    #[test]
    fn the_default_accelerator_matches_the_cli() {
        assert_eq!(default_accelerator(TargetOs::Macos), Accelerator::Metal);
        assert_eq!(default_accelerator(TargetOs::Windows), Accelerator::Cuda);
        assert_eq!(default_accelerator(TargetOs::Linux), Accelerator::Cpu);
    }

    #[test]
    fn installer_environment_scrubs_secrets_and_overrides_interactive_settings() {
        let env: std::collections::BTreeMap<_, _> = installer_env([
            ("PATH".into(), "/usr/bin:/bin".into()),
            ("HOME".into(), "/home/build".into()),
            ("GITHUB_TOKEN".into(), "secret".into()),
            ("LOCAL_LLM_FOUNDRY_API_TOKEN".into(), "app-secret".into()),
            ("LLAMA_MONITOR_PORT".into(), "7778".into()),
            ("NONINTERACTIVE".into(), "0".into()),
            ("HOMEBREW_NO_ENV_HINTS".into(), "0".into()),
        ])
        .into_iter()
        .collect();
        assert_eq!(env.get("PATH").map(String::as_str), Some("/usr/bin:/bin"));
        assert_eq!(env.get("HOME").map(String::as_str), Some("/home/build"));
        assert_eq!(env.get("NONINTERACTIVE").map(String::as_str), Some("1"));
        assert_eq!(
            env.get("HOMEBREW_NO_ENV_HINTS").map(String::as_str),
            Some("1")
        );
        assert!(!env.contains_key("GITHUB_TOKEN"));
        assert!(!env.contains_key("LOCAL_LLM_FOUNDRY_API_TOKEN"));
        assert!(!env.contains_key("LLAMA_MONITOR_PORT"));
    }

    #[test]
    fn owner_names_are_stable_api_labels() {
        assert_eq!(owner_name(Owner::Build), "build");
        assert_eq!(owner_name(Owner::Install), "install");
    }
}
