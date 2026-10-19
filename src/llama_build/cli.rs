//! Standalone source-build command. Intercepted before ordinary app startup so
//! planning never creates tokens, configuration, logs, caches or build trees.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use clap::{Parser, ValueEnum};

use super::cache::{LlamaBuildConfig, Ownership};
use super::configure::{cmake_build_args, find_stale_cache_paths};
use super::job::{
    JobDeps, JobStatus, NEEDS_RECONFIGURE_MARKER, SharedState, Toolchain, tree_needs_wipe,
};
use super::orchestrate::{BuildStart, ModellessHost, run_build};
use super::paths::{BuildPaths, without_verbatim_prefix};
use super::real::RealToolchain;
use super::source::is_full_commit;
use super::supervise::Launcher;
use super::{Accelerator, BuildOptions, BuildPlan, RefChoice, TargetOs, effective_parallelism};
use crate::llama_runtime::gate::RuntimeGate;
use crate::llama_runtime::promote::RuntimePaths;
use crate::paths::AppPaths;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum CliAccelerator {
    Metal,
    Cuda,
    Cpu,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum CliRef {
    Latest,
    Master,
}

#[derive(Debug, Parser)]
#[command(
    name = "llama-build",
    bin_name = "local-llm-foundry llama-build",
    about = "Build and install the managed llama.cpp runtime from source. Stop the foundry and its models first."
)]
pub struct BuildCli {
    /// Print prerequisite checks, exact upstream revision and CMake argv; do not write directories.
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub config_dir: Option<PathBuf>,
    #[arg(long, value_enum)]
    pub accelerator: Option<CliAccelerator>,
    #[arg(long = "ref", value_enum, default_value = "latest")]
    pub reference: CliRef,
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..=4096), conflicts_with = "headroom")]
    pub jobs: Option<u16>,
    /// Use half the logical CPUs and lower compiler priority.
    #[arg(long)]
    pub headroom: bool,
    /// Reset only the selected managed build tree before configuring.
    #[arg(long)]
    pub wipe: bool,
}

impl BuildCli {
    pub fn options(&self, os: TargetOs, cpus: usize) -> Result<BuildOptions, String> {
        let accelerator = match self.accelerator {
            Some(CliAccelerator::Metal) => Accelerator::Metal,
            Some(CliAccelerator::Cuda) => Accelerator::Cuda,
            Some(CliAccelerator::Cpu) => Accelerator::Cpu,
            None if os == TargetOs::Macos => Accelerator::Metal,
            None if os == TargetOs::Windows => Accelerator::Cuda,
            None => Accelerator::Cpu,
        };
        if !Accelerator::available_on(os).contains(&accelerator) {
            return Err(format!(
                "{} is not supported on {os:?}",
                accelerator.as_str()
            ));
        }
        if self
            .jobs
            .is_some_and(|n| usize::from(n) > cpus.max(1).saturating_mul(2))
        {
            return Err("--jobs must not exceed twice the detected logical CPU count".into());
        }
        if let Some(root) = &self.config_dir
            && (root.as_os_str().is_empty()
                || root.components().any(|c| matches!(c, Component::ParentDir)))
        {
            return Err("--config-dir must be nonempty and must not contain '..'".into());
        }
        let mut options = BuildOptions::new(
            accelerator,
            match self.reference {
                CliRef::Latest => RefChoice::LatestTag,
                CliRef::Master => RefChoice::Master,
            },
        );
        options.parallel = self.jobs.map(usize::from);
        options.leave_headroom = self.headroom;
        options.wipe_build_dir = self.wipe;
        Ok(options)
    }
}

/// Resolve existing ancestors without creating absent descendants. Matches the
/// canonical paths passed to CMake by actual execution, including /tmp aliases.
fn planned_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let mut parent = absolute.as_path();
    let mut suffix = Vec::new();
    loop {
        match std::fs::canonicalize(parent) {
            Ok(mut p) => {
                for name in suffix.iter().rev() {
                    p.push(name);
                }
                return Ok(without_verbatim_prefix(&p));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                suffix.push(
                    parent
                        .file_name()
                        .ok_or_else(|| e.to_string())?
                        .to_os_string(),
                );
                parent = parent.parent().ok_or_else(|| e.to_string())?;
            }
            Err(e) => return Err(format!("could not resolve {}: {e}", parent.display())),
        }
    }
}

/// Read-only seam for tests. Production supplies a RealToolchain in dry-run
/// mode (no smoke test or cache/report persistence).
pub fn dry_plan(
    tc: &dyn Toolchain,
    paths: &BuildPaths,
    options: &BuildOptions,
) -> Result<(String, bool), String> {
    let report = tc.prereqs(options);
    let mut text = String::new();
    text.push_str(&format!(
        "Managed root: {}\nPrerequisites:\n",
        paths.root.display()
    ));
    for check in &report.checks {
        text.push_str(&format!(
            "  {:?} {}: {}\n",
            check.status, check.title, check.detail
        ));
    }
    // Print prerequisite evidence even if planning cannot proceed further.
    let inputs = tc
        .plan_inputs(options)
        .map_err(|e| format!("{text}\nCannot plan: {e}"))?;
    let resolved = tc
        .resolve(options)
        .map_err(|e| format!("{text}\nCannot resolve upstream revision: {e}"))?;
    if !is_full_commit(&resolved.commit) {
        return Err(format!(
            "{text}\nUpstream did not resolve to a full commit id"
        ));
    }
    let mut plan = BuildPlan {
        accelerator: options.accelerator,
        os: tc.os(),
        source_dir: planned_path(&paths.source)?,
        build_dir: planned_path(&paths.build_tree(options.accelerator))?,
        ccache: inputs.ccache,
        apple_silicon: inputs.apple_silicon,
        cuda: inputs.cuda,
        stale_cache_vars: vec![],
    };
    let cache = std::fs::read_to_string(plan.build_dir.join("CMakeCache.txt")).ok();
    let marker = plan.build_dir.join(NEEDS_RECONFIGURE_MARKER).exists();
    let wipe = options.wipe_build_dir || tree_needs_wipe(cache.as_deref(), &plan, marker).is_some();
    if !wipe {
        plan.stale_cache_vars = cache
            .as_deref()
            .map(|s| {
                find_stale_cache_paths(s, &|p| p.exists())
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect()
            })
            .unwrap_or_default();
    }
    text.push_str(&format!(
        "Ref: {}\nExact commit: {}\nAccelerator: {}\nWipe build tree: {wipe}\n",
        resolved.reference,
        resolved.commit,
        options.accelerator.as_str()
    ));
    let cmake = tc.cmake_program().to_string_lossy().into_owned();
    let mut configure = vec![cmake.clone()];
    configure.extend(tc.configure_args(&plan));
    let mut build = vec![cmake];
    build.extend(cmake_build_args(
        &plan,
        effective_parallelism(options, tc.logical_cpus()),
    ));
    // JSON arrays are lossless argv, not an ambiguously shell-quoted command.
    text.push_str(&format!(
        "configure argv: {}\nbuild argv: {}\n",
        serde_json::to_string(&configure).unwrap(),
        serde_json::to_string(&build).unwrap()
    ));
    text.push_str("No build or installation was performed. This revision is a preview; execution resolves and pins its own exact commit.\n");
    Ok((text, report.ready))
}

/// Only handles argv[1] == llama-build; ordinary dashboard flags remain owned
/// by the existing parser. Clap help and usage errors retain their exit codes.
pub fn try_run(args: impl IntoIterator<Item = OsString>) -> Option<i32> {
    let args: Vec<_> = args.into_iter().collect();
    if args.get(1).is_none_or(|a| a != "llama-build") {
        return None;
    }
    let parsed = match BuildCli::try_parse_from(
        std::iter::once(OsString::from("llama-build")).chain(args.into_iter().skip(2)),
    ) {
        Ok(p) => p,
        Err(e) => {
            let code = e.exit_code();
            let _ = e.print();
            return Some(code);
        }
    };
    Some(match execute(parsed) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("llama-build: {e}");
            1
        }
    })
}

fn execute(cli: BuildCli) -> Result<i32, String> {
    let cpus = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let options = cli.options(TargetOs::host(), cpus)?;
    let root = planned_path(&AppPaths::resolve_root(cli.config_dir.clone()))?;
    if root.parent().is_none() {
        return Err("the filesystem root cannot be used as --config-dir".into());
    }
    let config =
        LlamaBuildConfig::load(&root).map_err(|e| format!("invalid llama-build.json: {e}"))?;
    let app = AppPaths::from_root(root);
    let paths = BuildPaths::new(&app);
    let cancel = Arc::new(AtomicBool::new(false));
    let tc = Arc::new(
        RealToolchain::new(paths.clone(), config.clone(), Launcher::current_exe())
            .dry_run(cli.dry_run)
            .with_cancel(cancel.clone()),
    );
    if cli.dry_run {
        let (text, ready) = dry_plan(tc.as_ref(), &paths, &options)?;
        print!("{text}");
        return Ok(if ready { 0 } else { 1 });
    }
    let _process_lease =
        super::cache::runtime_process_lease(&app.root, true).map_err(|e| e.to_string())?;
    eprintln!(
        "Building into {}. Keep the foundry and its models stopped until this command finishes.",
        app.bin_dir().display()
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async {
        let c = cancel.clone();
        let signal = tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                c.store(true, Ordering::SeqCst);
                eprintln!("Cancellation requested. If installation/recovery has begun it must finish safely.");
            }
        });
        let observed: Arc<Mutex<Option<SharedState>>> = Arc::new(Mutex::new(None));
        let polling = observed.clone();
        let progress = tokio::spawn(async move {
            let mut previous = None;
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                let shared = polling.lock().unwrap_or_else(|e| e.into_inner()).clone();
                if let Some(shared) = shared {
                    let state = shared.lock().unwrap_or_else(|e| e.into_inner());
                    let current = (state.record.phase, state.record.progress);
                    if previous != Some(current) {
                        eprintln!("{}{}", current.0.as_str(), current.1.map(|p| format!(" {}/{}", p.done, p.total)).unwrap_or_default());
                        previous = Some(current);
                    }
                }
            }
        });
        let host = ModellessHost { cancel: cancel.clone() };
        let deps = JobDeps { toolchain: tc, paths, runtime: RuntimePaths::for_live_dir(&app.bin_dir()).map_err(|e| e.to_string())?, config };
        let result = run_build(deps, &host, &Arc::new(RuntimeGate::default()), &Ownership::default(), options, cancel, |id, state| {
            eprintln!("Job {id}; log: {}", state.lock().unwrap_or_else(|e| e.into_inner()).record.log_path.display());
            *observed.lock().unwrap_or_else(|e| e.into_inner()) = Some(state.clone());
        }).await;
        signal.abort(); progress.abort();
        match result {
            Err(BuildStart::Busy(b)) => { eprintln!("{b}"); Ok(75) }
            Err(BuildStart::RecoveryRequired(e)) => { eprintln!("{e}"); Ok(75) }
            Ok(result) => {
                let record = result.record;
                eprintln!("{}", record.outcome.as_deref().unwrap_or("Build ended without an outcome"));
                if let Some(f) = record.failure {
                    eprintln!("Failure in {}: {}\nLog: {}", f.phase.as_str(), f.summary, f.log_path.display());
                    for line in f.log_tail { eprintln!("{line}"); }
                }
                Ok(match record.status {
                    JobStatus::Succeeded => 0,
                    JobStatus::SucceededRestartFailed => 2,
                    JobStatus::Cancelled => 130,
                    _ => 1,
                })
            }
        }
    })
}
