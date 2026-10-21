//! The production [`Toolchain`]: every subprocess runs under
//! [`supervise`](super::supervise) so cancel and app death reach descendants.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::configure::{cmake_build_args, cmake_configure_args, ninja_file};
use super::gpu;
use super::job::{PlanInputs, StepIo, StepResult, Toolchain};
use super::paths::BuildPaths;
use super::prereqs::{Check, Collector, DiskInputs, PrereqReport, SmokeTest, Status, smoke_check};
use super::prereqs_macos::{MacContext, macos_checks};
use super::prereqs_windows::{WinContext, WindowsToolchain, windows_checks};
use super::source::{self, Git, ResolvedRef, SourceError};
use super::supervise::{
    Launcher, Line, SpawnSpec, load_records, record_for, run_contained_with, update_records,
};
use super::sysprobe::{CmdOutput, RealSystem, SystemProbe, free_space_of};
use super::toolchain_env::{
    BuildEnvInputs, ToolchainCache, build_env, clean_base_path, file_mtime_unix, load_cache,
    parse_env_dump, redacted_env_diff, store_cache, vcvars_capture_command,
};
use super::{Accelerator, BuildOptions, BuildPlan, CudaPlan, CudaVersion, TargetOs};
use crate::llama_build::cache::LlamaBuildConfig;

pub struct RealToolchain {
    os: TargetOs,
    paths: BuildPaths,
    launcher: Launcher,
    config: LlamaBuildConfig,
    sys: RealSystem,
    remote: String,
    windows: Mutex<Option<WindowsToolchain>>,
    build_env: Mutex<Option<BTreeMap<String, String>>>,
    /// Re-run the smoke test even if the fingerprint is unchanged.
    force_smoke: bool,
    /// Side-effect free: no smoke test, no cache or report written.
    dry_run: bool,
    cancel: Arc<AtomicBool>,
}

/// The app's environment as build children may see it: secrets and the app's
/// own settings are stripped.
fn process_env() -> BTreeMap<String, String> {
    super::toolchain_env::scrub_env(std::env::vars())
}

impl RealToolchain {
    fn query(&self, spec: &SpawnSpec, timeout: std::time::Duration) -> Result<CmdOutput, String> {
        self.query_mode(spec, timeout, true)
    }

    fn query_mode(
        &self,
        spec: &SpawnSpec,
        timeout: std::time::Duration,
        cancellable: bool,
    ) -> Result<CmdOutput, String> {
        if cancellable && self.cancel.load(Ordering::SeqCst) {
            return Err("cancelled".into());
        }
        let mut output = CmdOutput::default();
        let started = std::time::Instant::now();
        let pid = std::cell::Cell::new(0);
        let mut sink = |line| {
            let (dest, text) = match line {
                Line::Stdout(t) => (&mut output.stdout, t),
                Line::Stderr(t) => (&mut output.stderr, t),
            };
            if dest.len() < 4 * 1024 * 1024 {
                dest.extend(text.chars().take(4 * 1024 * 1024 - dest.len()));
                dest.push('\n');
            }
        };
        let result = run_contained_with(
            spec,
            &self.launcher,
            &mut sink,
            &|| started.elapsed() >= timeout || (cancellable && self.cancel.load(Ordering::SeqCst)),
            &mut |p| {
                pid.set(p);
                if self.dry_run {
                    Ok(())
                } else {
                    self.persist_record("query", p, true)
                }
            },
        );
        if !self.dry_run && pid.get() != 0 && result.as_ref().is_ok_and(|o| o.contained) {
            self.persist_record("query", pid.get(), false)
                .map_err(|e| e.to_string())?;
        }
        let result = result.map_err(|e| e.to_string())?;
        if !result.contained {
            return Err("query descendants could not be contained; repair is required".into());
        }
        if result.cancelled {
            if cancellable && self.cancel.load(Ordering::SeqCst) {
                return Err("cancelled".into());
            }
            return Err(format!(
                "{} timed out after {}s",
                spec.program.display(),
                timeout.as_secs()
            ));
        }
        output.status = result.exit_code;
        Ok(output)
    }
    pub fn new(paths: BuildPaths, config: LlamaBuildConfig, launcher: Launcher) -> Self {
        Self {
            os: TargetOs::host(),
            paths,
            launcher,
            config,
            sys: RealSystem,
            remote: source::default_remote().to_string(),
            windows: Mutex::new(None),
            build_env: Mutex::new(None),
            force_smoke: false,
            dry_run: false,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_remote(mut self, url: impl Into<String>) -> Self {
        self.remote = url.into();
        self
    }

    pub fn dry_run(mut self, dry: bool) -> Self {
        self.dry_run = dry;
        self
    }

    pub fn with_cancel(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn force_smoke(mut self, force: bool) -> Self {
        self.force_smoke = force;
        self
    }

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn tool(&self, name: &str) -> PathBuf {
        if self.os == TargetOs::Windows
            && let Some(tc) = Self::lock(&self.windows).as_ref()
        {
            let found = match name {
                "cmake" => tc.cmake.clone(),
                "ninja" => tc.ninja.clone(),
                "git" => tc.git.clone(),
                _ => None,
            };
            if let Some(path) = found {
                return path;
            }
        }
        self.sys.which(name).unwrap_or_else(|| PathBuf::from(name))
    }

    fn disk_inputs(&self, accelerator: Accelerator, ccache: bool) -> DiskInputs {
        DiskInputs {
            os: self.os,
            accelerator,
            checkout_present: self.paths.source.join(".git").exists(),
            build_tree_present: self
                .paths
                .build_tree(accelerator)
                .join("CMakeCache.txt")
                .exists(),
            ccache_enabled: ccache,
            ccache_current_bytes: super::cache::dir_size(&self.paths.ccache),
            ccache_cap_bytes: self.config.ccache_max_bytes,
        }
    }

    /// The environment every cmake/ninja/nvcc child gets.
    fn env_for(&self, cuda_root: Option<&str>, ccache: bool) -> BTreeMap<String, String> {
        let process = process_env();
        let captured = Self::lock(&self.build_env).clone();
        let mut env = build_env(&BuildEnvInputs {
            os: self.os,
            process_env: &process,
            captured: captured.as_ref(),
            cuda_root,
            ccache_dir: ccache.then_some(self.paths.ccache.as_path()),
            ccache_max_bytes: ccache.then_some(self.config.ccache_max_bytes),
        });
        if self.os == TargetOs::Macos {
            // A GUI-launched app has no shell PATH.
            let path = env.get("PATH").cloned().unwrap_or_default();
            let mut dirs: Vec<String> = ["/opt/homebrew/bin", "/usr/local/bin"]
                .iter()
                .filter(|d| !path.split(':').any(|p| p == **d))
                .map(|d| d.to_string())
                .collect();
            dirs.push(path);
            env.insert("PATH".into(), dirs.join(":"));
            // The Command Line Tools are enough; never require full Xcode.
            env.entry("DEVELOPER_DIR".into()).or_insert_with(|| {
                if Path::new("/Library/Developer/CommandLineTools").is_dir()
                    && !Path::new("/Applications/Xcode.app").is_dir()
                {
                    "/Library/Developer/CommandLineTools".into()
                } else {
                    String::new()
                }
            });
            if env.get("DEVELOPER_DIR").is_some_and(String::is_empty) {
                env.remove("DEVELOPER_DIR");
            }
        }
        // Never let a user's CMake generator preference leak in.
        env.remove("CMAKE_GENERATOR");
        env.remove("CMAKE_BUILD_TYPE");
        env
    }

    /// Durable identity record so a crashed app's orphans can be found.
    fn persist_record(&self, job_id: &str, pid: u32, add: bool) -> std::io::Result<()> {
        // Resolve identity before taking the records lock.
        let record = if add {
            Some(record_for(job_id, pid).ok_or_else(|| {
                std::io::Error::other("could not establish child process creation identity")
            })?)
        } else {
            None
        };
        update_records(&self.paths.process_records, |all| {
            all.retain(|r| r.pid != pid);
            all.extend(record);
        })
    }

    /// Run one contained step, teeing every line to the job log.
    fn step(
        &self,
        spec: &SpawnSpec,
        io: &mut StepIo,
        parent_env: Option<&BTreeMap<String, String>>,
    ) -> StepResult {
        let header = format!(
            "$ {}  (cwd: {})",
            spec.display(),
            spec.cwd
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| ".".into())
        );
        (io.sink)(&header);
        if let (Some(env), Some(parent)) = (&spec.env, parent_env) {
            let child: BTreeMap<String, String> = env.iter().cloned().collect();
            let diff = redacted_env_diff(parent, &child);
            if !diff.is_empty() {
                (io.sink)(&format!("  env: {}", diff.join(" ")));
            }
        }
        let job_id = io.job_id.to_string();
        let outcome = {
            let sink = RefCell::new(&mut *io.sink);
            let mut on_line = |line: Line| match line {
                Line::Stdout(t) | Line::Stderr(t) => (*sink.borrow_mut())(&t),
            };
            let pid = std::cell::Cell::new(0u32);
            let mut on_spawn = |p: u32| {
                pid.set(p);
                self.persist_record(&job_id, p, true)
            };
            let result =
                run_contained_with(spec, &self.launcher, &mut on_line, io.cancel, &mut on_spawn);
            // Released only after the tree is gone (the runner reaped it).
            if pid.get() != 0 && result.as_ref().is_ok_and(|o| o.contained) {
                let _ = self.persist_record(&job_id, pid.get(), false);
            }
            result
        };
        match outcome {
            Err(e) => StepResult::Failed {
                exit_code: None,
                command: spec.display(),
                detail: Some(format!("could not start: {e}")),
            },
            Ok(o) if o.cancelled && o.contained => StepResult::Cancelled,
            Ok(o) if !o.contained => StepResult::Failed {
                exit_code: o.exit_code,
                command: spec.display(),
                detail: Some(
                    "child processes could not be contained; the build tree is blocked until they exit"
                        .into(),
                ),
            },
            Ok(o) if o.exit_code == Some(0) => StepResult::Ok,
            Ok(o) => StepResult::Failed {
                exit_code: o.exit_code,
                command: spec.display(),
                detail: None,
            },
        }
    }

    fn spec(
        &self,
        program: PathBuf,
        args: Vec<String>,
        cwd: Option<PathBuf>,
        env: Vec<(String, String)>,
        low: bool,
    ) -> SpawnSpec {
        SpawnSpec {
            program,
            args,
            raw_args: None,
            cwd,
            env: Some(env),
            low_priority: low,
        }
    }

    // ----------------------------------------------------------- Windows env

    /// Capture the vcvars64 environment, using the cache when valid.
    fn ensure_vcvars(&self, tc: &WindowsToolchain) -> Result<BTreeMap<String, String>, String> {
        let (Some(vcvars), Some(instance)) = (
            &tc.vcvars,
            tc.vs_instance_id.clone().or_else(|| tc.vs_install.clone()),
        ) else {
            return Err("Visual Studio was not resolved".into());
        };
        let mtime = file_mtime_unix(vcvars);
        if let Some(cache) = load_cache(&self.paths.toolchain_cache)
            && cache.matches(&instance, vcvars, mtime)
        {
            return Ok(cache.env);
        }
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let base = clean_base_path(&root);
        let command = vcvars_capture_command(vcvars, &base);
        let mut spec = SpawnSpec::new("cmd", &["/d", "/s", "/c"]);
        spec.raw_args = Some(command);
        spec.env = Some(
            super::toolchain_env::vcvars_base_env(&process_env())
                .into_iter()
                .collect(),
        );
        let outcome = self.query(&spec, std::time::Duration::from_secs(90))?;
        if !outcome.ok() {
            return Err(format!("vcvars64.bat failed (exit {:?})", outcome.status));
        }
        let env = parse_env_dump(&outcome.stdout);
        if !env
            .get("PATH")
            .is_some_and(|p| p.to_ascii_lowercase().contains("vc"))
        {
            return Err("vcvars64.bat did not produce an MSVC environment".into());
        }
        let cache = ToolchainCache {
            vs_instance_id: instance,
            vcvars: vcvars.clone(),
            vcvars_mtime_unix: mtime,
            captured_at_unix: unix_now(),
            env: env.clone(),
        };
        let _ = store_cache(&self.paths.toolchain_cache, &cache);
        Ok(env)
    }

    fn smoke_dir(&self) -> Result<tempfile::TempDir, String> {
        std::fs::create_dir_all(&self.paths.root).map_err(|e| e.to_string())?;
        tempfile::Builder::new()
            .prefix("smoke-")
            .tempdir_in(&self.paths.root)
            .map_err(|e| e.to_string())
    }

    fn run_smoke(
        &self,
        accelerator: Accelerator,
        tc: Option<&WindowsToolchain>,
    ) -> Result<String, String> {
        let dir = self.smoke_dir()?;
        let root = std::fs::canonicalize(dir.path())
            .map(|p| super::paths::without_verbatim_prefix(&p))
            .map_err(|e| e.to_string())?;
        let cuda = accelerator == Accelerator::Cuda;
        let metal = accelerator == Accelerator::Metal;
        let (cmake_lists, source_name, source_body) = if cuda {
            let arch = tc
                .and_then(|t| t.cuda_architectures.clone())
                .ok_or("no CUDA architecture is known for the smoke test")?;
            (
                format!(
                    "cmake_minimum_required(VERSION 3.21)\nproject(smoke LANGUAGES CXX CUDA)\nset(CMAKE_CUDA_ARCHITECTURES {arch})\nadd_executable(smoke main.cu)\n"
                ),
                "main.cu",
                "__global__ void k() {}\nint main() { k<<<1,1>>>(); return 0; }\n",
            )
        } else {
            let link = if metal {
                "target_link_libraries(smoke PRIVATE \"-framework Metal\" \"-framework Foundation\")\n"
            } else {
                ""
            };
            (
                format!(
                    "cmake_minimum_required(VERSION 3.21)\nproject(smoke LANGUAGES CXX)\nadd_executable(smoke main.cpp)\n{link}"
                ),
                "main.cpp",
                "int main() { return 0; }\n",
            )
        };
        std::fs::write(root.join("CMakeLists.txt"), cmake_lists).map_err(|e| e.to_string())?;
        std::fs::write(root.join(source_name), source_body).map_err(|e| e.to_string())?;
        let build = root.join("build");
        let cuda_root = tc.and_then(|t| {
            t.cuda_root
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
        });
        let env: Vec<(String, String)> = self
            .env_for(cuda_root.as_deref().filter(|_| cuda), false)
            .into_iter()
            .collect();
        let cmake = self.tool("cmake");
        let mut log = String::new();
        let mut sink = |l: &str| {
            log.push_str(l);
            log.push('\n');
        };
        let started = std::time::Instant::now();
        let no_cancel = || {
            started.elapsed() >= std::time::Duration::from_secs(300)
                || self.cancel.load(Ordering::SeqCst)
        };
        let mut io = StepIo {
            job_id: "smoke",
            sink: &mut sink,
            cancel: &no_cancel,
        };
        let mut args = vec![
            "-S".to_string(),
            root.to_string_lossy().into_owned(),
            "-B".into(),
            build.to_string_lossy().into_owned(),
            "-G".into(),
            "Ninja".into(),
        ];
        args.push(format!(
            "-DCMAKE_MAKE_PROGRAM={}",
            self.tool("ninja").display()
        ));
        if cuda {
            if self.os == TargetOs::Windows {
                args.push("-DCMAKE_C_COMPILER=cl".into());
                args.push("-DCMAKE_CXX_COMPILER=cl".into());
                args.push("-DCMAKE_CUDA_HOST_COMPILER=cl.exe".into());
            }
            if let Some(nvcc) = tc.and_then(|t| t.nvcc.as_ref()) {
                args.push(format!("-DCMAKE_CUDA_COMPILER={}", nvcc.display()));
            }
        }
        let configure = self.step(
            &self.spec(cmake.clone(), args, Some(root.clone()), env.clone(), false),
            &mut io,
            None,
        );
        let built = match configure {
            StepResult::Ok => self.step(
                &self.spec(
                    cmake,
                    vec!["--build".into(), build.to_string_lossy().into_owned()],
                    Some(root.clone()),
                    env,
                    false,
                ),
                &mut io,
                None,
            ),
            other => other,
        };
        if std::fs::symlink_metadata(&self.paths.process_records).is_ok() {
            let retained = dir.keep();
            return Err(format!(
                "{log}\nSmoke directory retained for process repair: {}",
                retained.display()
            ));
        }
        match built {
            StepResult::Ok => Ok(log
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")),
            _ => Err(log),
        }
    }
}

/// Smoke test adapter handed to the platform check functions.
struct Deferred;

impl SmokeTest for Deferred {
    fn run(&self, _: Accelerator, _: Option<&WindowsToolchain>) -> Result<String, String> {
        Err(DEFERRED.into())
    }
}

const DEFERRED: &str = "\u{0}deferred";

/// Git whose commands are contained and stream into the job log.
struct SupervisedGit<'a, 'io> {
    tc: &'a RealToolchain,
    program: PathBuf,
    io: RefCell<&'a mut StepIo<'io>>,
    env: Vec<(String, String)>,
}

impl Git for SupervisedGit<'_, '_> {
    fn run(&self, cwd: Option<&Path>, args: &[String]) -> Result<CmdOutput, String> {
        let mut full: Vec<String> = Vec::new();
        if let Some(cwd) = cwd {
            full.push("-C".into());
            full.push(cwd.to_string_lossy().into_owned());
        }
        for c in [
            "advice.detachedHead=false",
            "gc.auto=0",
            "core.longpaths=true",
        ] {
            full.push("-c".into());
            full.push(c.into());
        }
        if source::allows_local_file_transport(Some(&self.tc.remote), args) {
            full.push("-c".into());
            full.push("protocol.file.allow=always".into());
        }
        full.extend(args.iter().cloned());
        let mut env = self.env.clone();
        env.push(("GIT_TERMINAL_PROMPT".into(), "0".into()));
        env.push(("LC_ALL".into(), "C".into()));
        let spec = self.tc.spec(self.program.clone(), full, None, env, false);
        let mut io = self.io.borrow_mut();
        let mut collected = CmdOutput::default();
        let (job_id, cancel): (String, &dyn Fn() -> bool) = (io.job_id.to_string(), io.cancel);
        let outcome = {
            let sink = RefCell::new(&mut *io.sink);
            let mut on_line = |line: Line| match line {
                Line::Stdout(t) => {
                    (*sink.borrow_mut())(&t);
                    collected.stdout.push_str(&t);
                    collected.stdout.push('\n');
                }
                Line::Stderr(t) => {
                    (*sink.borrow_mut())(&t);
                    collected.stderr.push_str(&t);
                    collected.stderr.push('\n');
                }
            };
            let pid = std::cell::Cell::new(0u32);
            let mut on_spawn = |p: u32| {
                pid.set(p);
                self.tc.persist_record(&job_id, p, true)
            };
            let started = std::time::Instant::now();
            let r = run_contained_with(
                &spec,
                &self.tc.launcher,
                &mut on_line,
                &|| cancel() || started.elapsed() >= std::time::Duration::from_secs(300),
                &mut on_spawn,
            );
            if pid.get() != 0 && r.as_ref().is_ok_and(|o| o.contained) {
                let _ = self.tc.persist_record(&job_id, pid.get(), false);
            }
            r
        }
        .map_err(|e| e.to_string())?;
        if !outcome.contained {
            return Err(
                "child containment could not be established; build reuse is blocked".into(),
            );
        }
        if outcome.cancelled {
            return Err("cancelled".into());
        }
        collected.status = outcome.exit_code;
        Ok(collected)
    }
}

impl Toolchain for RealToolchain {
    fn os(&self) -> TargetOs {
        self.os
    }

    fn cmake_program(&self) -> PathBuf {
        self.tool("cmake")
    }

    fn configure_args(&self, plan: &BuildPlan) -> Vec<String> {
        let mut args = cmake_configure_args(plan);
        args.push(format!(
            "-DCMAKE_MAKE_PROGRAM={}",
            self.tool("ninja").display()
        ));
        if self.os == TargetOs::Macos {
            let prefix = if cfg!(target_arch = "aarch64") {
                "/opt/homebrew"
            } else {
                "/usr/local"
            };
            let ssl = Path::new(prefix).join("opt/openssl@3");
            if ssl.join("lib/libcrypto.dylib").is_file() {
                // Intel and native Homebrew may coexist. Existing x86 paths
                // are not "stale" to CMake's existence check, but are unusable
                // for an arm64 link. Clear only SSL cache entries, then pin
                // the native prefix. Record these same argv in provenance.
                args.extend([
                    "-U".into(),
                    "OPENSSL_*".into(),
                    format!("-DOPENSSL_ROOT_DIR={}", ssl.display()),
                ]);
            }
        }
        args
    }

    fn prereqs(&self, options: &BuildOptions) -> PrereqReport {
        let accelerator = options.accelerator;
        let unresolved = match load_records(&self.paths.process_records) {
            Ok(records) if records.is_empty() => None,
            Ok(_) => Some(format!(
                "A previous child process has unresolved ownership; {}.",
                super::repair::REPAIR_ACTION
            )),
            Err(e) => Some(format!(
                "Cannot read process ownership records: {e}. {}.",
                super::repair::REPAIR_ACTION
            )),
        };
        if let Some(reason) = unresolved {
            let mut c = Collector::default();
            c.push(Check::fail(
                "containment",
                "Build process ownership",
                reason,
            ));
            return PrereqReport::from_collector(self.os, accelerator, c);
        }
        let ccache_wanted = self.sys.which("ccache").is_some();
        let disk = self.disk_inputs(accelerator, ccache_wanted);
        let mut collector: Collector;
        let mut windows: Option<WindowsToolchain> = None;
        match self.os {
            TargetOs::Windows => {
                let ctx = WinContext {
                    app_root: &self.paths.root,
                    disk,
                    arch_override: options.cuda_architectures.clone(),
                    check_network: true,
                };
                let (c, tc) = windows_checks(&OwnedSystem(self), &ctx, &Deferred);
                collector = c;
                windows = Some(tc);
            }
            TargetOs::Linux => {
                (collector, windows) =
                    linux_checks(&OwnedSystem(self), options, &self.paths.root, &disk);
            }
            TargetOs::Macos => {
                let ctx = MacContext {
                    accelerator,
                    app_root: &self.paths.root,
                    disk,
                    check_network: true,
                };
                collector = macos_checks(&OwnedSystem(self), &ctx, &Deferred);
            }
        }
        if let Some(tc) = windows.as_ref() {
            *Self::lock(&self.windows) = Some(tc.clone());
            if self.os == TargetOs::Windows
                && !self.dry_run
                && collector
                    .checks
                    .iter()
                    .all(|c| c.id == "smoke" || c.status != Status::Fail)
            {
                match self.ensure_vcvars(tc).and_then(|env| {
                    super::toolchain_env::validate_captured_env(&env, &|p| p.is_file())?;
                    Ok(env)
                }) {
                    Ok(env) => {
                        for key in [
                            "VCTOOLSVERSION",
                            "WINDOWSSDKVERSION",
                            "VCTOOLSINSTALLDIR",
                            "WINDOWSSDKDIR",
                        ] {
                            if let Some(value) = env.get(key) {
                                collector.fingerprint_part(key, value);
                            }
                        }
                        *Self::lock(&self.build_env) = Some(env);
                        collector.checks.push(Check::pass(
                            "sdk",
                            "MSVC and Windows SDK",
                            "Compiler and SDK headers exist in the captured environment.",
                        ));
                    }
                    Err(e) => {
                        collector
                            .checks
                            .push(Check::fail("vcvars", "MSVC build environment", e))
                    }
                }
            }
        }
        // Resolve the smoke test: reuse a passing result for an unchanged toolchain.
        let deferred = collector
            .checks
            .iter()
            .position(|c| c.id == "smoke" && c.evidence.iter().any(|e| e.contains(DEFERRED)));
        if let Some(at) = deferred {
            let fingerprint = collector.fingerprint();
            let previous: Option<PrereqReport> = std::fs::read(&self.paths.prereqs_last)
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok());
            let reusable = !self.force_smoke
                && previous.as_ref().is_some_and(|p| {
                    p.accelerator == accelerator && p.smoke_reusable(&fingerprint)
                });
            collector.checks[at] = if collector
                .checks
                .iter()
                .any(|c| c.id != "smoke" && c.status == Status::Fail)
            {
                Check::skipped(
                    "smoke",
                    "Compile smoke test",
                    "Resolve prerequisite failures before compiling.",
                )
            } else if self.dry_run {
                Check::skipped("smoke", "Compile smoke test", "Skipped in --dry-run.")
            } else if reusable {
                Check::pass(
                    "smoke",
                    "Compile smoke test",
                    "Reused: the toolchain is unchanged since the last passing run.",
                )
            } else {
                smoke_check(self.run_smoke(accelerator, windows.as_ref()), true)
            };
        }
        let report = PrereqReport::from_collector(self.os, accelerator, collector);
        if !self.dry_run
            && let Ok(json) = serde_json::to_vec_pretty(&report)
        {
            let _ = std::fs::create_dir_all(&self.paths.root);
            let _ = std::fs::write(&self.paths.prereqs_last, json);
        }
        report
    }

    fn plan_inputs(&self, options: &BuildOptions) -> Result<PlanInputs, String> {
        let ccache = self.sys.which("ccache").is_some();
        match options.accelerator {
            Accelerator::Cuda => {
                let guard = Self::lock(&self.windows);
                let tc = guard.as_ref().ok_or("the toolchain was not resolved")?;
                let (Some(nvcc), Some(version), Some(root)) =
                    (tc.nvcc.clone(), tc.cuda_version, tc.cuda_root.clone())
                else {
                    return Err("the CUDA toolkit was not resolved".into());
                };
                let architectures = match &options.cuda_architectures {
                    Some(a) => gpu::validate_cuda_architectures(a)?,
                    None => tc
                        .cuda_architectures
                        .clone()
                        .ok_or("no CUDA architecture is known; enter one")?,
                };
                Ok(PlanInputs {
                    ccache,
                    apple_silicon: false,
                    cuda: Some(CudaPlan {
                        nvcc,
                        version,
                        architectures,
                    }),
                    cuda_root: Some(root),
                    device_id: Some("CUDA0".into()),
                })
            }
            Accelerator::Metal => Ok(PlanInputs {
                ccache,
                apple_silicon: gpu::detect_apple_chip(&OwnedSystem(self))
                    .is_some_and(|c| c.apple_silicon),
                cuda: None,
                cuda_root: None,
                device_id: Some("MTL0".into()),
            }),
            Accelerator::Cpu => Ok(PlanInputs {
                ccache,
                apple_silicon: false,
                cuda: None,
                cuda_root: None,
                device_id: None,
            }),
        }
    }

    fn resolve(&self, options: &BuildOptions) -> Result<ResolvedRef, String> {
        let git = OwnedGit(self, true);
        source::resolve(&git, &self.remote, options.reference)
            .map_err(|e: SourceError| e.to_string())
    }

    fn fetch(
        &self,
        source_dir: &Path,
        resolved: &ResolvedRef,
        job_id: &str,
        io: &mut StepIo,
    ) -> StepResult {
        let git = SupervisedGit {
            tc: self,
            program: self.tool("git"),
            io: RefCell::new(io),
            env: self.env_for(None, false).into_iter().collect(),
        };
        match source::checkout_exact(&git, source_dir, &self.remote, &resolved.commit, job_id) {
            Ok(()) => StepResult::Ok,
            Err(SourceError::Git { detail, .. }) if detail == "cancelled" => StepResult::Cancelled,
            Err(e) => StepResult::Failed {
                exit_code: None,
                command: "git".into(),
                detail: Some(e.to_string()),
            },
        }
    }

    fn configure(&self, plan: &BuildPlan, io: &mut StepIo) -> StepResult {
        let cuda_root = plan.cuda.as_ref().and_then(|c| {
            c.nvcc
                .parent()?
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
        });
        let env = self.env_for(cuda_root.as_deref(), plan.ccache);
        let parent = process_env();
        let spec = self.spec(
            self.tool("cmake"),
            self.configure_args(plan),
            Some(plan.build_dir.clone()),
            env.into_iter().collect(),
            false,
        );
        self.step(&spec, io, Some(&parent))
    }

    fn compile(
        &self,
        plan: &BuildPlan,
        jobs: usize,
        low_priority: bool,
        io: &mut StepIo,
    ) -> StepResult {
        let cuda_root = plan.cuda.as_ref().and_then(|c| {
            c.nvcc
                .parent()?
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
        });
        let env = self.env_for(cuda_root.as_deref(), plan.ccache);
        let spec = self.spec(
            self.tool("cmake"),
            cmake_build_args(plan, jobs),
            Some(plan.build_dir.clone()),
            env.into_iter().collect(),
            low_priority,
        );
        self.step(&spec, io, None)
    }

    fn ninja_targets(&self, plan: &BuildPlan) -> Result<String, String> {
        let env: Vec<(String, String)> = self.env_for(None, false).into_iter().collect();
        let ninja = self.tool("ninja").to_string_lossy().into_owned();
        let build = plan.build_dir.to_string_lossy().into_owned();
        let file = ninja_file(plan);
        let out = self.query(
            &self.spec(
                PathBuf::from(ninja),
                ["-C", &build, "-f", file, "-t", "targets", "all"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                None,
                env,
                false,
            ),
            std::time::Duration::from_secs(60),
        )?;
        if out.ok() {
            Ok(out.stdout)
        } else {
            Err(out.combined().trim().chars().take(300).collect())
        }
    }

    fn toolchain_versions(&self) -> BTreeMap<String, String> {
        let mut versions = BTreeMap::new();
        for (key, program, args) in [
            ("cmake", self.tool("cmake"), "--version"),
            ("ninja", self.tool("ninja"), "--version"),
            ("git", self.tool("git"), "--version"),
        ] {
            if let Ok(out) = OwnedSystem(self).run(
                &program.to_string_lossy(),
                &[args],
                std::time::Duration::from_secs(15),
            ) && let Some(v) = super::prereqs::parse_version(&out.stdout)
            {
                versions.insert(key.to_string(), super::prereqs::version_string(v));
            }
        }
        if let Some(tc) = Self::lock(&self.windows).as_ref() {
            if let Some(v) = &tc.vs_version {
                versions.insert("visual_studio".into(), v.clone());
            }
            if let Some(v) = tc.cuda_version {
                versions.insert("nvcc".into(), v.to_string());
            }
        }
        if self.os == TargetOs::Macos
            && let Ok(out) =
                OwnedSystem(self).run("clang", &["--version"], std::time::Duration::from_secs(15))
            && let Some(line) = out.stdout.lines().next()
        {
            versions.insert("clang".into(), line.to_string());
        }
        versions
    }

    fn free_space(&self, path: &Path) -> Option<u64> {
        free_space_of(path)
    }

    fn logical_cpus(&self) -> usize {
        self.sys.logical_cpus()
    }

    fn unpin(&self, source_dir: &Path, job_id: &str) {
        let git = OwnedGit(self, false);
        let _ = source::unpin(&git, source_dir, job_id);
        if super::job::containment_clear(&self.paths) {
            // This job still owns the checkout/all trees. No other build can
            // be using a pin or fetched dependency while we prune.
            let _ = source::prune_inactive(&git, source_dir, &|_| false);
            let _ = super::cache::prune_dependency_caches(
                &self.paths,
                super::cache::DEPENDENCY_CACHE_LIMIT,
            );
            if let Some(ccache) = self.sys.which("ccache")
                && super::cache::ccache_needs_trim(
                    super::cache::dir_size(&self.paths.ccache),
                    self.config.ccache_max_bytes,
                )
            {
                let spec = self.spec(
                    ccache,
                    vec!["--cleanup".into()],
                    None,
                    self.env_for(None, true).into_iter().collect(),
                    false,
                );
                let _ = self.query_mode(&spec, std::time::Duration::from_secs(120), false);
            }
        }
    }
}

pub fn cuda_version_from(text: &str) -> Option<CudaVersion> {
    CudaVersion::parse(text)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Read-only machine discovery still launches under ownership: a hung
/// nvidia-smi/curl/vswhere cannot leave descendants behind after timeout.
struct OwnedSystem<'a>(&'a RealToolchain);

impl SystemProbe for OwnedSystem<'_> {
    fn os(&self) -> TargetOs {
        self.0.os
    }
    fn run(
        &self,
        program: &str,
        args: &[&str],
        timeout: std::time::Duration,
    ) -> Result<CmdOutput, String> {
        let program = self
            .0
            .sys
            .which(program)
            .unwrap_or_else(|| PathBuf::from(program));
        self.0.query(
            &self.0.spec(
                program,
                args.iter().map(|s| s.to_string()).collect(),
                None,
                self.0.env_for(None, false).into_iter().collect(),
                false,
            ),
            timeout,
        )
    }
    fn env_var(&self, key: &str) -> Option<String> {
        self.0.sys.env_var(key)
    }
    fn path_exists(&self, path: &Path) -> bool {
        self.0.sys.path_exists(path)
    }
    fn which(&self, name: &str) -> Option<PathBuf> {
        self.0.sys.which(name)
    }
    fn free_space(&self, path: &Path) -> Option<u64> {
        self.0.sys.free_space(path)
    }
    fn logical_cpus(&self) -> usize {
        self.0.sys.logical_cpus()
    }
    fn registry_dword(&self, key: &str, value: &str) -> Option<u32> {
        let out = self
            .run(
                "reg",
                &["query", key, "/v", value],
                std::time::Duration::from_secs(15),
            )
            .ok()?;
        let value = out
            .stdout
            .split_whitespace()
            .last()?
            .trim_start_matches("0x");
        u32::from_str_radix(value, 16).ok()
    }
}

struct OwnedGit<'a>(&'a RealToolchain, bool);

impl Git for OwnedGit<'_> {
    fn run(&self, cwd: Option<&Path>, args: &[String]) -> Result<CmdOutput, String> {
        let mut env = self.0.env_for(None, false);
        env.insert("GIT_TERMINAL_PROMPT".into(), "0".into());
        env.insert("LC_ALL".into(), "C".into());
        let mut argv = vec![
            "-c".into(),
            "gc.auto=0".into(),
            "-c".into(),
            "core.longpaths=true".into(),
        ];
        argv.extend_from_slice(args);
        self.0.query_mode(
            &self.0.spec(
                self.0.tool("git"),
                argv,
                cwd.map(Path::to_path_buf),
                env.into_iter().collect(),
                false,
            ),
            std::time::Duration::from_secs(120),
            self.1,
        )
    }
}

fn linux_checks(
    sys: &dyn SystemProbe,
    options: &BuildOptions,
    root: &Path,
    disk: &DiskInputs,
) -> (Collector, Option<WindowsToolchain>) {
    use super::prereqs::{
        disk_check, driver_check_for_os, estimate_disk, network_checks, toolkit_version_check,
    };
    let mut c = Collector::default();
    c.push(Check::warn(
        "platform",
        "Linux",
        "Linux source builds are untested.",
    ));
    for tool in ["git", "cmake", "ninja", "c++"] {
        let result = sys
            .which(tool)
            .ok_or_else(|| format!("{tool} is missing"))
            .and_then(|p| {
                sys.run(
                    &p.to_string_lossy(),
                    &["--version"],
                    std::time::Duration::from_secs(15),
                )
            });
        match result {
            Ok(out) if out.ok() => {
                c.fingerprint_part(tool, &out.stdout);
                let check = if tool == "cmake"
                    && super::prereqs::parse_version(&out.stdout)
                        .is_none_or(|v| v < super::prereqs::CMAKE_MINIMUM)
                {
                    Check::fail(tool, tool, "CMake 3.21 or newer is required.")
                } else {
                    Check::pass(tool, tool, out.stdout.lines().next().unwrap_or(tool))
                };
                c.push(check);
            }
            _ => c.push(Check::fail(tool, tool, format!("{tool} could not run"))),
        }
    }
    c.push(disk_check(&estimate_disk(disk), sys.free_space(root)));
    let mut tc = None;
    if options.accelerator == Accelerator::Cuda {
        let nvcc = sys
            .which("nvcc")
            .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)));
        let version = nvcc
            .as_ref()
            .and_then(|p| {
                sys.run(
                    &p.to_string_lossy(),
                    &["--version"],
                    std::time::Duration::from_secs(20),
                )
                .ok()
            })
            .and_then(|o| super::prereqs_windows::parse_nvcc_version(&o.stdout));
        c.push(toolkit_version_check(version));
        let gpu = gpu::detect_nvidia(sys)
            .ok()
            .and_then(|g| g.into_iter().next());
        let arch = options
            .cuda_architectures
            .clone()
            .or_else(|| gpu.as_ref()?.cuda_architecture.clone());
        if arch.is_none() {
            c.push(Check::fail(
                "cuda_arch",
                "CUDA architecture",
                "No supported device architecture was detected; supply an explicit architecture.",
            ));
        }
        if let Some(v) = version {
            c.push(driver_check_for_os(
                TargetOs::Linux,
                v,
                gpu.as_ref().and_then(|g| g.driver_version.as_deref()),
                arch.as_deref(),
            ));
            c.fingerprint_part("nvcc", &v.to_string());
        }
        tc = Some(WindowsToolchain {
            cuda_root: nvcc
                .as_ref()
                .and_then(|p| p.parent()?.parent())
                .map(Path::to_path_buf),
            nvcc,
            cuda_version: version,
            cuda_architectures: arch,
            ..Default::default()
        });
    }
    c.checks.extend(network_checks(
        sys,
        options.accelerator == Accelerator::Cuda,
    ));
    c.push(smoke_check(
        Deferred.run(options.accelerator, tc.as_ref()),
        true,
    ));
    (c, tc)
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[test]
    fn cancelled_preflight_queries_never_launch_or_create_state() {
        let root = tempfile::tempdir().unwrap();
        let app = crate::paths::AppPaths::from_root(root.path().join("app"));
        let tc = RealToolchain::new(
            BuildPaths::new(&app),
            LlamaBuildConfig::default(),
            Launcher {
                supervisor_exe: None,
            },
        )
        .with_cancel(Arc::new(AtomicBool::new(true)));
        let error = tc
            .query(
                &SpawnSpec::new("must-not-launch", &[]),
                std::time::Duration::from_secs(1),
            )
            .unwrap_err();
        assert_eq!(error, "cancelled");
        assert!(!app.root.exists());
    }
}

#[cfg(test)]
mod env_tests {
    use super::*;

    #[test]
    fn the_children_environment_is_scrubbed_but_keeps_path() {
        // Unique names: no other test reads them.
        unsafe {
            std::env::set_var("LLF_FIX4_WIRING_API_TOKEN", "secret");
            std::env::set_var("LOCAL_LLM_FOUNDRY_FIX4_WIRING", "setting");
        }
        let env = process_env();
        assert!(!env.contains_key("LLF_FIX4_WIRING_API_TOKEN"));
        assert!(!env.contains_key("LOCAL_LLM_FOUNDRY_FIX4_WIRING"));
        // Windows names it `Path`; the scrub must keep it under whatever case the OS uses.
        if std::env::vars().any(|(k, _)| k.eq_ignore_ascii_case("PATH")) {
            assert!(env.keys().any(|k| k.eq_ignore_ascii_case("PATH")));
        }
        let root = tempfile::tempdir().unwrap();
        let app = crate::paths::AppPaths::from_root(root.path().join("app"));
        let tc = RealToolchain::new(
            BuildPaths::new(&app),
            LlamaBuildConfig::default(),
            Launcher {
                supervisor_exe: None,
            },
        );
        let child = tc.env_for(None, false);
        assert!(!child.contains_key("LLF_FIX4_WIRING_API_TOKEN"));
        assert!(!child.contains_key("LOCAL_LLM_FOUNDRY_FIX4_WIRING"));
    }
}
