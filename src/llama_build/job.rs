//! The single-flight build job (plan §4.4).
//!
//! ```text
//! preflight → fetch → configure → compile → stage → promote → done
//! ```
//!
//! Everything up to and including *stage* works only on build-owned
//! directories (`llama-build/`, a sibling staging directory), so a failure in
//! any of those phases leaves the live and previous runtime sets untouched.
//! *Promote* is the shared journaled engine; this module only hands it a
//! staged candidate and a server-built verification intent.
//!
//! The job talks to the machine through [`Toolchain`], so tests inject a fake
//! that fails at any phase.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::llama_runtime::manifest::RuntimeManifest;
use crate::llama_runtime::promote::RuntimePaths;
use crate::llama_runtime::verify::VerificationIntent;

use super::cache::{LlamaBuildConfig, Owner, Ownership, dir_size};
use super::catalog::{Hint, HintContext, first_error, match_hint};
use super::configure::{
    ConfigureEvidence, cache_value, cmake_configure_args, configure_evidence,
    cuda_configure_is_fa_all, find_stale_cache_paths, ninja_file, output_subdir,
};
use super::manifest::{SourceBuildInfo, build_label, source_manifest};
use super::paths::{BuildPaths, create_canonical_within, without_verbatim_prefix};
use super::prereqs::{DiskInputs, PrereqReport, disk_floor, estimate_disk, stage_requirement};
use super::source::{ResolvedRef, is_full_commit};
use super::stage::{StageError, StagePlan, produced_targets, source_intent, stage};
use super::{Accelerator, BuildOptions, BuildPlan, CudaPlan, TargetOs, effective_parallelism};

pub const LOG_RING_LINES: usize = 2000;
pub const FAILURE_TAIL_LINES: usize = 80;
/// Per-job log file bound (plan §4.4).
pub const LOG_FILE_CAP_BYTES: u64 = 64 * 1024 * 1024;
pub const NEEDS_RECONFIGURE_MARKER: &str = ".foundry-needs-reconfigure";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Preflight,
    Fetch,
    Configure,
    Compile,
    Stage,
    Promote,
    Done,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preflight => "preflight",
            Self::Fetch => "fetch",
            Self::Configure => "configure",
            Self::Compile => "compile",
            Self::Stage => "stage",
            Self::Promote => "promote",
            Self::Done => "done",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Running,
    Succeeded,
    /// Installed and verified, but the model restart failed.
    SucceededRestartFailed,
    Failed,
    Cancelled,
    /// The app exited while the job was running.
    Interrupted,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        self != Self::Running
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    pub done: u32,
    pub total: u32,
}

/// `[12/140] Building CXX object ...` → (12, 140).
pub fn parse_progress(line: &str) -> Option<Progress> {
    let rest = line.trim_start().strip_prefix('[')?;
    let (counts, _) = rest.split_once(']')?;
    let (done, total) = counts.split_once('/')?;
    let done: u32 = done.trim().parse().ok()?;
    let total: u32 = total.trim().parse().ok()?;
    (total > 0 && done <= total).then_some(Progress { done, total })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildFailure {
    pub phase: Phase,
    pub summary: String,
    pub command: Option<String>,
    pub exit_code: Option<i32>,
    pub log_tail: Vec<String>,
    pub first_error: Option<String>,
    pub hint: Option<Hint>,
    pub log_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub options: BuildOptions,
    pub phase: Phase,
    pub status: JobStatus,
    pub started_unix: u64,
    pub finished_unix: Option<u64>,
    /// Seconds spent per finished phase, for "your last build took ...".
    pub phase_secs: BTreeMap<String, u64>,
    pub resolved: Option<ResolvedRef>,
    pub progress: Option<Progress>,
    pub failure: Option<BuildFailure>,
    pub outcome: Option<String>,
    pub log_path: PathBuf,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Log: file tee + ring
// ---------------------------------------------------------------------------

pub struct JobLog {
    file: Option<std::fs::File>,
    cap: u64,
    written: u64,
    truncated: bool,
    ring: VecDeque<String>,
}

impl JobLog {
    pub fn new(path: Option<&Path>, cap: u64) -> std::io::Result<Self> {
        let file = match path {
            Some(p) => {
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                Some(std::fs::File::create(p)?)
            }
            None => None,
        };
        Ok(Self {
            file,
            cap,
            written: 0,
            truncated: false,
            ring: VecDeque::with_capacity(LOG_RING_LINES),
        })
    }

    pub fn push(&mut self, line: &str) {
        if self.ring.len() == LOG_RING_LINES {
            self.ring.pop_front();
        }
        self.ring.push_back(line.to_string());
        let Some(file) = self.file.as_mut() else {
            return;
        };
        if self.truncated {
            return;
        }
        let bytes = line.len() as u64 + 1;
        if self.written + bytes > self.cap {
            self.truncated = true;
            let _ = writeln!(
                file,
                "[log truncated: the {} MiB per-job limit was reached; the live view continues]",
                self.cap / (1024 * 1024)
            );
            return;
        }
        if writeln!(file, "{line}").is_ok() {
            self.written += bytes;
        }
    }

    pub fn tail(&self, n: usize) -> Vec<String> {
        let skip = self.ring.len().saturating_sub(n);
        self.ring.iter().skip(skip).cloned().collect()
    }

    pub fn all(&self) -> Vec<String> {
        self.ring.iter().cloned().collect()
    }

    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

/// State shared with whoever is polling the job.
pub struct Shared {
    pub record: JobRecord,
    pub log: JobLog,
    record_path: Option<PathBuf>,
}

impl Shared {
    pub fn new(record: JobRecord, log: JobLog, record_path: Option<PathBuf>) -> Self {
        Self {
            record,
            log,
            record_path,
        }
    }

    pub fn persist(&self) {
        let Some(path) = &self.record_path else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("json.tmp");
        if let Ok(json) = serde_json::to_vec_pretty(&self.record)
            && std::fs::write(&tmp, json).is_ok()
        {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    fn line(&mut self, text: &str) {
        if let Some(p) = parse_progress(text) {
            self.record.progress = Some(p);
        }
        self.log.push(text);
    }
}

pub type SharedState = Arc<Mutex<Shared>>;

fn with<R>(shared: &SharedState, f: impl FnOnce(&mut Shared) -> R) -> R {
    f(&mut shared.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Short, poison-tolerant access for orchestration/progress observers.
pub(crate) fn with_state<R>(shared: &SharedState, f: impl FnOnce(&mut Shared) -> R) -> R {
    with(shared, f)
}

pub(crate) fn set_phase_public(shared: &SharedState, phase: Phase) {
    with(shared, |s| {
        s.record.phase = phase;
        s.record.progress = None;
        s.line(&format!("== phase: {}", phase.as_str()));
        s.persist();
    });
}

/// Only proven absence clears the durable containment barrier.
pub fn containment_clear(paths: &BuildPaths) -> bool {
    proven_absent(&paths.process_records)
}

fn proven_absent(path: &Path) -> bool {
    matches!(std::fs::symlink_metadata(path), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
}

/// Fail closed for any journal, torn write, or unresolved child. Recovery owns
/// those paths; a build error is not permission to remove them.
pub(crate) fn discard_unowned_staging(deps: &JobDeps, dir: &Path) {
    if containment_clear(&deps.paths)
        && proven_absent(&deps.runtime.journal)
        && proven_absent(&deps.runtime.parent.join(".runtime-transaction.json.tmp"))
        && dir.parent() == Some(deps.runtime.parent.as_path())
        && dir
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with(".bin.staging-"))
    {
        let _ = std::fs::remove_dir_all(dir);
    }
}

// ---------------------------------------------------------------------------
// Toolchain seam
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult {
    Ok,
    Failed {
        exit_code: Option<i32>,
        command: String,
        detail: Option<String>,
    },
    Cancelled,
}

/// What a step may use: a line sink and the cancel flag.
pub struct StepIo<'a> {
    pub job_id: &'a str,
    pub sink: &'a mut dyn FnMut(&str),
    pub cancel: &'a dyn Fn() -> bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanInputs {
    pub ccache: bool,
    pub apple_silicon: bool,
    pub cuda: Option<CudaPlan>,
    pub cuda_root: Option<PathBuf>,
    /// Detected device id the verifier must enumerate (`CUDA0`, `MTL0`).
    pub device_id: Option<String>,
}

pub trait Toolchain: Send + Sync {
    fn os(&self) -> TargetOs;
    /// Resolved executable used for both CMake invocations.
    fn cmake_program(&self) -> PathBuf {
        PathBuf::from("cmake")
    }
    /// Exact configure argv, including resolved host-specific tool/dependency pins.
    fn configure_args(&self, plan: &BuildPlan) -> Vec<String> {
        cmake_configure_args(plan)
    }
    /// The fast prerequisite pass. The job refuses to start when it fails.
    fn prereqs(&self, options: &BuildOptions) -> PrereqReport;
    fn plan_inputs(&self, options: &BuildOptions) -> Result<PlanInputs, String>;
    fn resolve(&self, options: &BuildOptions) -> Result<ResolvedRef, String>;
    fn fetch(
        &self,
        source_dir: &Path,
        resolved: &ResolvedRef,
        job_id: &str,
        io: &mut StepIo,
    ) -> StepResult;
    fn configure(&self, plan: &BuildPlan, io: &mut StepIo) -> StepResult;
    fn compile(
        &self,
        plan: &BuildPlan,
        jobs: usize,
        low_priority: bool,
        io: &mut StepIo,
    ) -> StepResult;
    /// `ninja -t targets all` for the configure's build tree.
    fn ninja_targets(&self, plan: &BuildPlan) -> Result<String, String>;
    fn toolchain_versions(&self) -> BTreeMap<String, String>;
    fn free_space(&self, path: &Path) -> Option<u64>;
    fn logical_cpus(&self) -> usize;
    /// Release the job's pin on the checkout.
    fn unpin(&self, _source_dir: &Path, _job_id: &str) {}
}

// ---------------------------------------------------------------------------
// The job
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct JobDeps {
    pub toolchain: Arc<dyn Toolchain>,
    pub paths: BuildPaths,
    pub runtime: RuntimePaths,
    pub config: LlamaBuildConfig,
}

/// A fully staged candidate, ready for the shared promotion engine.
#[derive(Debug, Clone)]
pub struct Staged {
    pub dir: PathBuf,
    pub label: String,
    pub manifest: RuntimeManifest,
    pub intent: VerificationIntent,
    pub accelerator: Accelerator,
    pub resolved: ResolvedRef,
    pub cuda_root_free_check: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobEnd {
    Failed(Box<BuildFailure>),
    Cancelled,
}

fn set_phase(shared: &SharedState, phase: Phase, started: &mut std::time::Instant) {
    with(shared, |s| {
        let prev = s.record.phase;
        if prev != phase {
            s.record
                .phase_secs
                .insert(prev.as_str().to_string(), started.elapsed().as_secs());
            *started = std::time::Instant::now();
        }
        s.record.phase = phase;
        s.line(&format!("== phase: {}", phase.as_str()));
        s.persist();
    });
}

fn fail(
    shared: &SharedState,
    phase: Phase,
    summary: impl Into<String>,
    command: Option<String>,
    exit_code: Option<i32>,
    ctx: &HintContext,
) -> JobEnd {
    let summary = summary.into();
    let failure = with(shared, |s| {
        let lines = s.log.all();
        let hint = match_hint(&lines, ctx);
        let failure = BuildFailure {
            phase,
            summary: summary.clone(),
            command,
            exit_code,
            log_tail: s.log.tail(FAILURE_TAIL_LINES),
            first_error: first_error(&lines),
            hint,
            log_path: s.record.log_path.clone(),
        };
        s.record.failure = Some(failure.clone());
        failure
    });
    JobEnd::Failed(Box::new(failure))
}

fn step_failure(
    shared: &SharedState,
    phase: Phase,
    what: &str,
    result: StepResult,
    ctx: &HintContext,
) -> Result<(), JobEnd> {
    match result {
        StepResult::Ok => Ok(()),
        StepResult::Cancelled => Err(JobEnd::Cancelled),
        StepResult::Failed {
            exit_code,
            command,
            detail,
        } => {
            let summary = match (&detail, exit_code) {
                (Some(d), _) => format!("{what} failed: {d}"),
                (None, Some(c)) => format!("{what} failed (exit code {c})."),
                (None, None) => format!("{what} failed."),
            };
            Err(fail(shared, phase, summary, Some(command), exit_code, ctx))
        }
    }
}

fn read_cache(build_dir: &Path) -> Option<String> {
    std::fs::read_to_string(build_dir.join("CMakeCache.txt")).ok()
}

fn same_path(a: &str, b: &Path) -> bool {
    let norm = |p: &Path| {
        let s = without_verbatim_prefix(p)
            .to_string_lossy()
            .replace('\\', "/");
        let s = s.trim_end_matches('/').to_string();
        if cfg!(windows) {
            s.to_ascii_lowercase()
        } else {
            s
        }
    };
    norm(Path::new(a)) == norm(b)
}

/// Why an existing build tree cannot be reused as-is, if it cannot.
pub fn tree_needs_wipe(cache_text: Option<&str>, plan: &BuildPlan, marker: bool) -> Option<String> {
    let Some(cache) = cache_text else {
        return marker.then(|| "an interrupted configure left no usable CMake cache".to_string());
    };
    match cache_value(cache, "CMAKE_HOME_DIRECTORY") {
        Some(home) if same_path(home, &plan.source_dir) => {}
        Some(home) => {
            return Some(format!(
                "the tree was configured for a different source directory ({home})"
            ));
        }
        None => return Some("the CMake cache has no source directory".to_string()),
    }
    if let Some(generator) = cache_value(cache, "CMAKE_GENERATOR")
        && generator != plan.generator().cmake_name()
    {
        return Some(format!(
            "the tree uses the {generator} generator, not {}",
            plan.generator().cmake_name()
        ));
    }
    None
}

fn marker_path(build_dir: &Path) -> PathBuf {
    build_dir.join(NEEDS_RECONFIGURE_MARKER)
}

/// Run preflight through stage. Returns a staged candidate or why not.
pub fn run_to_staged(
    deps: &JobDeps,
    options: &BuildOptions,
    job_id: &str,
    shared: &SharedState,
    cancel: &dyn Fn() -> bool,
) -> Result<Staged, JobEnd> {
    let tc = &deps.toolchain;
    let os = tc.os();
    let mut phase_start = std::time::Instant::now();
    let none = HintContext::default();

    if !containment_clear(&deps.paths) {
        return Err(fail(
            shared,
            Phase::Preflight,
            "Unresolved build process records require recovery before reuse: repair with POST /api/llama-build/repair-processes or `local-llm-foundry llama-build --repair-processes`.",
            None,
            None,
            &none,
        ));
    }
    if cancel() {
        return Err(JobEnd::Cancelled);
    }
    // ------------------------------------------------------------ preflight
    set_phase(shared, Phase::Preflight, &mut phase_start);
    if cancel() {
        return Err(JobEnd::Cancelled);
    }
    let report = tc.prereqs(options);
    if cancel() {
        return Err(JobEnd::Cancelled);
    }
    if !report.ready {
        let failing: Vec<String> = report
            .failures()
            .map(|c| format!("{}: {}", c.title, c.detail))
            .collect();
        with(shared, |s| {
            for line in &failing {
                s.line(line);
            }
        });
        return Err(fail(
            shared,
            Phase::Preflight,
            format!("Prerequisites are not met ({} failing).", failing.len()),
            None,
            None,
            &none,
        ));
    }
    let inputs = tc
        .plan_inputs(options)
        .map_err(|e| fail(shared, Phase::Preflight, e, None, None, &none))?;
    if options.accelerator == Accelerator::Cuda && inputs.cuda.is_none() {
        return Err(fail(
            shared,
            Phase::Preflight,
            "No CUDA toolkit or architecture was resolved.",
            None,
            None,
            &none,
        ));
    }

    let io_err = |shared: &SharedState, e: std::io::Error| {
        fail(
            shared,
            Phase::Preflight,
            format!("filesystem error: {e}"),
            None,
            None,
            &HintContext::default(),
        )
    };
    let source_dir = create_canonical_within(&deps.paths.root, &deps.paths.source)
        .map(|p| without_verbatim_prefix(&p))
        .map_err(|e| io_err(shared, e))?;
    let build_dir = create_canonical_within(
        &deps.paths.root,
        &deps.paths.build_tree(options.accelerator),
    )
    .map(|p| without_verbatim_prefix(&p))
    .map_err(|e| io_err(shared, e))?;
    if inputs.ccache {
        std::fs::create_dir_all(&deps.paths.ccache).map_err(|e| io_err(shared, e))?;
    }

    let mut plan = BuildPlan {
        accelerator: options.accelerator,
        os,
        source_dir: source_dir.clone(),
        build_dir: build_dir.clone(),
        ccache: inputs.ccache,
        apple_silicon: inputs.apple_silicon,
        cuda: inputs.cuda.clone(),
        stale_cache_vars: Vec::new(),
    };

    // Disk budget: additional allocations only.
    let checkout_present = source_dir.join(".git").exists();
    let tree_present = build_dir.join("CMakeCache.txt").exists() && !options.wipe_build_dir;
    let estimate = estimate_disk(&DiskInputs {
        os,
        accelerator: options.accelerator,
        checkout_present,
        build_tree_present: tree_present,
        ccache_enabled: inputs.ccache,
        ccache_current_bytes: dir_size(&deps.paths.ccache),
        ccache_cap_bytes: deps.config.ccache_max_bytes,
    });
    let free = tc.free_space(&deps.paths.root);
    if let Some(free) = free
        && free < estimate.required_bytes
    {
        let ctx = HintContext {
            free_space_bytes: Some(free),
            ..Default::default()
        };
        with(shared, |s| {
            s.line(&format!(
                "No space left on device: {} MiB free, {} MiB needed (floor {} MiB).",
                free >> 20,
                estimate.required_bytes >> 20,
                disk_floor(os) >> 20
            ));
        });
        return Err(fail(
            shared,
            Phase::Preflight,
            "Not enough free disk space for this build.",
            None,
            None,
            &ctx,
        ));
    }

    let resolved = tc.resolve(options).map_err(|e| {
        fail(
            shared,
            Phase::Preflight,
            format!("could not resolve the revision: {e}"),
            None,
            None,
            &none,
        )
    })?;
    if !is_full_commit(&resolved.commit) {
        return Err(fail(
            shared,
            Phase::Preflight,
            "The resolved revision is not a full commit id.",
            None,
            None,
            &none,
        ));
    }
    with(shared, |s| {
        s.record.resolved = Some(resolved.clone());
        s.line(&format!(
            "Building {} at {} ({})",
            resolved.reference,
            resolved.commit,
            options.accelerator.as_str()
        ));
    });
    if cancel() {
        return Err(JobEnd::Cancelled);
    }

    // Wipe a tree that cannot be reused, or when asked.
    let cache_text = read_cache(&build_dir);
    let marker = marker_path(&build_dir).exists();
    let wipe_reason = if options.wipe_build_dir {
        Some("a clean build was requested".to_string())
    } else {
        tree_needs_wipe(cache_text.as_deref(), &plan, marker)
    };
    if let Some(reason) = wipe_reason
        && (cache_text.is_some() || marker || options.wipe_build_dir)
    {
        with(shared, |s| {
            s.line(&format!("Resetting the build tree: {reason}."))
        });
        wipe_tree(&build_dir).map_err(|e| io_err(shared, e))?;
    }

    // ---------------------------------------------------------------- fetch
    set_phase(shared, Phase::Fetch, &mut phase_start);
    let result = {
        let mut sink = |l: &str| with(shared, |s| s.line(l));
        let mut io = StepIo {
            job_id,
            sink: &mut sink,
            cancel,
        };
        tc.fetch(&source_dir, &resolved, job_id, &mut io)
    };
    let fetch_ctx = HintContext::default();
    step_failure(
        shared,
        Phase::Fetch,
        "Fetching the source",
        result,
        &fetch_ctx,
    )?;
    if cancel() {
        return Err(JobEnd::Cancelled);
    }

    // ------------------------------------------------------------ configure
    set_phase(shared, Phase::Configure, &mut phase_start);
    let existing = read_cache(&build_dir);
    let stale: Vec<(String, String)> = existing
        .as_deref()
        .map(|c| find_stale_cache_paths(c, &|p| p.exists()))
        .unwrap_or_default();
    if !stale.is_empty() {
        with(shared, |s| {
            s.line(&format!(
                "Stale CMake cache: {} cached path(s) no longer exist; purging them for this configure.",
                stale.len()
            ));
            for (var, path) in &stale {
                s.line(&format!("    {var} = {path}"));
            }
        });
    }
    plan.stale_cache_vars = stale.iter().map(|(v, _)| v.clone()).collect();
    // Dirty from here until the compile finishes.
    std::fs::write(marker_path(&build_dir), job_id).map_err(|e| io_err(shared, e))?;
    let configure_args = tc.configure_args(&plan);
    let result = {
        let mut sink = |l: &str| with(shared, |s| s.line(l));
        let mut io = StepIo {
            job_id,
            sink: &mut sink,
            cancel,
        };
        tc.configure(&plan, &mut io)
    };
    let after_configure = read_cache(&build_dir);
    let remaining_stale: Vec<(String, String)> = after_configure
        .as_deref()
        .map(|c| find_stale_cache_paths(c, &|p| p.exists()))
        .unwrap_or_default();
    let configure_ctx = HintContext {
        stale_cache_paths: remaining_stale,
        healed: !stale.is_empty(),
        free_space_bytes: tc.free_space(&deps.paths.root),
    };
    step_failure(
        shared,
        Phase::Configure,
        "CMake configure",
        result,
        &configure_ctx,
    )?;
    let Some(cache) = after_configure else {
        return Err(fail(
            shared,
            Phase::Configure,
            "CMake did not write a cache.",
            None,
            None,
            &none,
        ));
    };
    let evidence: ConfigureEvidence = configure_evidence(&cache)
        .map_err(|e| fail(shared, Phase::Configure, e.to_string(), None, None, &none))?;
    if options.accelerator == Accelerator::Cuda && !cuda_configure_is_fa_all(&evidence) {
        return Err(fail(
            shared,
            Phase::Configure,
            format!(
                "The configure recorded GGML_CUDA_FA_QUANTS={} instead of all; refusing to build a \
                 CUDA runtime without every K/V pair.",
                evidence.fa_quants_spec.as_deref().unwrap_or("(unset)")
            ),
            None,
            None,
            &none,
        ));
    }
    if cancel() {
        return Err(JobEnd::Cancelled);
    }

    // -------------------------------------------------------------- compile
    set_phase(shared, Phase::Compile, &mut phase_start);
    let jobs = effective_parallelism(options, tc.logical_cpus());
    with(shared, |s| {
        s.line(&format!("Compiling with {jobs} parallel job(s)."))
    });
    let result = {
        let mut sink = |l: &str| with(shared, |s| s.line(l));
        let mut io = StepIo {
            job_id,
            sink: &mut sink,
            cancel,
        };
        tc.compile(&plan, jobs, options.leave_headroom, &mut io)
    };
    let compile_ctx = HintContext {
        free_space_bytes: tc.free_space(&deps.paths.root),
        ..Default::default()
    };
    step_failure(shared, Phase::Compile, "Compilation", result, &compile_ctx)?;
    let _ = std::fs::remove_file(marker_path(&build_dir));
    if cancel() {
        return Err(JobEnd::Cancelled);
    }

    // ---------------------------------------------------------------- stage
    set_phase(shared, Phase::Stage, &mut phase_start);
    let targets = tc.ninja_targets(&plan).map_err(|e| {
        fail(
            shared,
            Phase::Stage,
            StageError::Targets(e).to_string(),
            None,
            None,
            &none,
        )
    })?;
    let produced: BTreeSet<String> = produced_targets(&targets);
    let staging_dir = deps.runtime.staging_dir(job_id);
    let stage_plan = StagePlan {
        os,
        accelerator: options.accelerator,
        output_dir: build_dir.join(output_subdir(&plan)),
        output_rel: output_subdir(&plan).to_string(),
        staging_dir: staging_dir.clone(),
        cuda_root: inputs.cuda_root.clone(),
    };
    let _ = ninja_file(&plan);
    let report = stage(&stage_plan, &produced, &|| {
        tc.free_space(&deps.runtime.parent)
    });
    let report = match report {
        Ok(r) => r,
        Err(e) => {
            discard_unowned_staging(deps, &staging_dir);
            let ctx = HintContext {
                free_space_bytes: tc.free_space(&deps.runtime.parent),
                ..Default::default()
            };
            if matches!(e, StageError::DiskExhausted { .. }) {
                with(shared, |s| s.line("No space left on device while staging."));
            }
            return Err(fail(shared, Phase::Stage, e.to_string(), None, None, &ctx));
        }
    };
    with(shared, |s| {
        s.line(&format!(
            "Staged {} file(s), {} link(s), {} CUDA runtime file(s); skipped {} stale.",
            report.files.len(),
            report.symlinks.len(),
            report.cuda_runtime.len(),
            report.skipped_stale.len()
        ));
    });

    let info = SourceBuildInfo {
        reference: resolved.reference.clone(),
        commit: resolved.commit.clone(),
        build_number: resolved.build_number,
        accelerator: options.accelerator,
        cmake_args: configure_args,
        toolchain: tc.toolchain_versions(),
        evidence,
    };
    let manifest = source_manifest(&staging_dir, &info).map_err(|e| {
        discard_unowned_staging(deps, &staging_dir);
        fail(
            shared,
            Phase::Stage,
            format!("could not write provenance: {e}"),
            None,
            None,
            &none,
        )
    })?;
    Ok(Staged {
        label: build_label(&info),
        dir: staging_dir,
        manifest,
        intent: source_intent(options.accelerator, &resolved.commit, inputs.device_id),
        accelerator: options.accelerator,
        resolved,
        cuda_root_free_check: stage_requirement(options.accelerator),
    })
}

fn wipe_tree(dir: &Path) -> std::io::Result<()> {
    if dir.exists() {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let meta = std::fs::symlink_metadata(&path)?;
            if meta.is_dir() {
                std::fs::remove_dir_all(&path)?;
            } else {
                std::fs::remove_file(&path)?;
            }
        }
    }
    Ok(())
}

/// Space check immediately before the journal is written (plan §4.4). Never
/// satisfied by deleting a runtime set.
pub fn recheck_before_journal(
    deps: &JobDeps,
    staged: &Staged,
    shared: &SharedState,
) -> Result<(), JobEnd> {
    if let Some(free) = deps.toolchain.free_space(&deps.runtime.parent)
        && free < staged.cuda_root_free_check
    {
        discard_unowned_staging(deps, &staged.dir);
        with(shared, |s| {
            s.line("No space left on device before installing.")
        });
        let ctx = HintContext {
            free_space_bytes: Some(free),
            ..Default::default()
        };
        return Err(fail(
            shared,
            Phase::Promote,
            "Not enough free disk space to install the new runtime. Nothing was changed.",
            None,
            None,
            &ctx,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Records, lifecycle
// ---------------------------------------------------------------------------

pub fn new_job_id() -> Result<String, String> {
    use rand::TryRng;
    let mut bytes = [0u8; 16];
    rand::rngs::SysRng
        .try_fill_bytes(&mut bytes)
        .map_err(|e| format!("could not obtain randomness for a build job: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

pub fn new_shared(
    deps: &JobDeps,
    options: &BuildOptions,
    job_id: &str,
    persist: bool,
) -> std::io::Result<SharedState> {
    let log_path = deps.paths.log_file(job_id);
    let record = JobRecord {
        id: job_id.to_string(),
        options: options.clone(),
        phase: Phase::Preflight,
        status: JobStatus::Running,
        started_unix: now_unix(),
        finished_unix: None,
        phase_secs: BTreeMap::new(),
        resolved: None,
        progress: None,
        failure: None,
        outcome: None,
        log_path: log_path.clone(),
    };
    let log = JobLog::new(persist.then_some(log_path.as_path()), LOG_FILE_CAP_BYTES)?;
    let shared = Shared::new(record, log, persist.then(|| deps.paths.record_file(job_id)));
    shared.persist();
    Ok(Arc::new(Mutex::new(shared)))
}

pub fn finish(shared: &SharedState, status: JobStatus, outcome: impl Into<String>) {
    with(shared, |s| {
        s.record.status = status;
        s.record.finished_unix = Some(now_unix());
        s.record.outcome = Some(outcome.into());
        if status == JobStatus::Succeeded || status == JobStatus::SucceededRestartFailed {
            s.record.phase = Phase::Done;
        }
        s.persist();
    });
}

pub fn read_record(file: &Path) -> Option<JobRecord> {
    serde_json::from_slice(&std::fs::read(file).ok()?).ok()
}

/// Seconds the most recent successful build took, for the panel.
pub fn last_build_seconds(paths: &BuildPaths) -> Option<u64> {
    let records = super::paths::list_job_records(&paths.logs).ok()?;
    records
        .iter()
        .filter_map(|r| r.record.as_deref().and_then(read_record))
        .find(|r| r.status == JobStatus::Succeeded)
        .and_then(|r| r.finished_unix.map(|f| f.saturating_sub(r.started_unix)))
}

/// Startup recovery for builds (plan §4.4, §4.3). Runs **before** the staging
/// sweep so a candidate belonging to a job we are about to mark interrupted is
/// not deleted out from under diagnostics.
///
/// 1. Reap recorded processes (identity-checked) so no orphan writer survives.
/// 2. Mark records still `running` as interrupted and flag their build tree
///    `needs_reconfigure`.
pub fn recover_interrupted(paths: &BuildPaths) -> Vec<String> {
    let _ = super::supervise::recover_all(&paths.process_records);
    let mut marked = Vec::new();
    let Ok(records) = super::paths::list_job_records(&paths.logs) else {
        return marked;
    };
    for pair in records {
        let Some(file) = pair.record else { continue };
        let Some(mut record) = read_record(&file) else {
            continue;
        };
        if record.status != JobStatus::Running {
            continue;
        }
        record.status = JobStatus::Interrupted;
        record.finished_unix = Some(now_unix());
        record.outcome = Some("The app exited while this build was running.".into());
        if let Ok(json) = serde_json::to_vec_pretty(&record) {
            let _ = std::fs::write(&file, json);
        }
        if matches!(
            record.phase,
            Phase::Configure | Phase::Compile | Phase::Fetch | Phase::Stage
        ) {
            let tree = paths.build_tree(record.options.accelerator);
            if tree.is_dir()
                && let Ok(tree) = create_canonical_within(&paths.root, &tree)
            {
                let _ = std::fs::write(marker_path(&tree), &record.id);
            }
        }
        marked.push(record.id);
    }
    marked
}

/// Ids the retention pass must not remove: running jobs and anything a
/// runtime journal still refers to.
pub fn protected_ids(paths: &BuildPaths, extra: &[String]) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = extra.iter().cloned().collect();
    if let Ok(records) = super::paths::list_job_records(&paths.logs) {
        for pair in records {
            if let Some(r) = pair.record.as_deref().and_then(read_record)
                && r.status == JobStatus::Running
            {
                out.insert(r.id);
            }
        }
    }
    out
}

pub fn prune_records(
    paths: &BuildPaths,
    extra_protected: &[String],
) -> std::io::Result<Vec<String>> {
    // An ambiguous journal can refer to any retained job. Until recovery retires
    // it, retaining extra diagnostics is safer than guessing which to delete.
    if !containment_clear(paths)
        || !proven_absent(&paths.root.join("runtime-transaction.json"))
        || !proven_absent(&paths.root.join(".runtime-transaction.json.tmp"))
    {
        return Ok(Vec::new());
    }
    let protected = protected_ids(paths, extra_protected);
    super::paths::prune_job_records(&paths.logs, &|id| protected.contains(id))
}

pub fn dedupe_owner() -> Owner {
    Owner::Build
}

/// Acquire single-flight ownership or name the running job (HTTP 409).
pub fn start_exclusive(
    ownership: &Ownership,
    job_id: &str,
) -> Result<super::cache::Lease, super::cache::Conflict> {
    ownership.try_acquire(Owner::Build, job_id)
}

#[cfg(test)]
#[path = "job_tests.rs"]
mod tests;
