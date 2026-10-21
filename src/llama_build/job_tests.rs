use super::*;
use crate::llama_build::prereqs::{Check, Collector, PrereqReport};
use crate::llama_build::{RefChoice, TargetOs};
use crate::paths::AppPaths;

const COMMIT: &str = "098ce526b0000000000000000000000000000000";

#[derive(Default)]
struct Script {
    fail_at: Option<Phase>,
    /// Lines the failing step prints first.
    fail_log: Vec<String>,
    free: Option<u64>,
    prereqs_ready: bool,
    cache_extra: String,
    skip_outputs: bool,
    ninja_error: Option<String>,
}

struct Fake {
    os: TargetOs,
    script: Mutex<Script>,
    calls: Mutex<Vec<String>>,
    plans: Mutex<Vec<BuildPlan>>,
    cuda: bool,
    cuda_root: Option<PathBuf>,
}

impl Fake {
    fn metal() -> Arc<Self> {
        Arc::new(Self {
            os: TargetOs::Macos,
            script: Mutex::new(Script {
                prereqs_ready: true,
                free: Some(u64::MAX),
                ..Default::default()
            }),
            calls: Mutex::new(Vec::new()),
            plans: Mutex::new(Vec::new()),
            cuda: false,
            cuda_root: None,
        })
    }

    fn set(&self, f: impl FnOnce(&mut Script)) {
        f(&mut self.script.lock().unwrap());
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn note(&self, call: &str) {
        self.calls.lock().unwrap().push(call.to_string());
    }

    fn should_fail(&self, phase: Phase, io: &mut StepIo) -> Option<StepResult> {
        let script = self.script.lock().unwrap();
        if script.fail_at == Some(phase) {
            for line in &script.fail_log {
                (io.sink)(line);
            }
            return Some(StepResult::Failed {
                exit_code: Some(1),
                command: format!("fake {}", phase.as_str()),
                detail: None,
            });
        }
        None
    }
}

impl Toolchain for Fake {
    fn os(&self) -> TargetOs {
        self.os
    }

    fn prereqs(&self, options: &BuildOptions) -> PrereqReport {
        self.note("prereqs");
        let mut c = Collector::default();
        if self.script.lock().unwrap().prereqs_ready {
            c.push(Check::pass("git", "git", "ok"));
        } else {
            c.push(Check::fail("cmake", "CMake", "cmake was not found."));
        }
        PrereqReport::from_collector(self.os, options.accelerator, c)
    }

    fn plan_inputs(&self, options: &BuildOptions) -> Result<PlanInputs, String> {
        self.note("plan_inputs");
        Ok(PlanInputs {
            ccache: false,
            apple_silicon: true,
            cuda: (options.accelerator == Accelerator::Cuda).then(|| CudaPlan {
                nvcc: PathBuf::from(r"C:\CUDA\bin\nvcc.exe"),
                version: crate::llama_build::CudaVersion::parse("13.4.92").unwrap(),
                architectures: "120a-real".into(),
            }),
            cuda_root: self.cuda_root.clone(),
            device_id: None,
        })
    }

    fn resolve(&self, options: &BuildOptions) -> Result<ResolvedRef, String> {
        self.note("resolve");
        Ok(ResolvedRef {
            reference: match options.reference {
                RefChoice::LatestTag => "b11727".into(),
                RefChoice::Master => "master".into(),
            },
            commit: COMMIT.into(),
            build_number: Some(11727),
        })
    }

    fn fetch(&self, _: &Path, _: &ResolvedRef, _: &str, io: &mut StepIo) -> StepResult {
        self.note("fetch");
        (io.sink)("Fetching from origin");
        if let Some(r) = self.should_fail(Phase::Fetch, io) {
            return r;
        }
        StepResult::Ok
    }

    fn configure(&self, plan: &BuildPlan, io: &mut StepIo) -> StepResult {
        self.note("configure");
        self.plans.lock().unwrap().push(plan.clone());
        (io.sink)("-- Configuring");
        if let Some(r) = self.should_fail(Phase::Configure, io) {
            return r;
        }
        let extra = self.script.lock().unwrap().cache_extra.clone();
        let cache = format!(
            "CMAKE_HOME_DIRECTORY:INTERNAL={}\nCMAKE_GENERATOR:INTERNAL={}\n{}{extra}",
            plan.source_dir.display(),
            plan.generator().cmake_name(),
            if self.cuda {
                "GGML_CUDA_FA_QUANTS:STRING=all\nCMAKE_CUDA_ARCHITECTURES:STRING=120a-real\n"
            } else {
                ""
            }
        );
        std::fs::write(plan.build_dir.join("CMakeCache.txt"), cache).unwrap();
        StepResult::Ok
    }

    fn compile(&self, plan: &BuildPlan, _: usize, _: bool, io: &mut StepIo) -> StepResult {
        self.note("compile");
        (io.sink)("[1/3] Building CXX object a.o");
        (io.sink)("[2/3] Building CXX object b.o");
        if let Some(r) = self.should_fail(Phase::Compile, io) {
            return r;
        }
        (io.sink)("[3/3] Linking");
        if !self.script.lock().unwrap().skip_outputs {
            let bin = plan.build_dir.join(output_subdir(plan));
            std::fs::create_dir_all(&bin).unwrap();
            for f in [
                "llama-server",
                "llama-bench",
                "libggml-metal.so",
                "libllama.dylib",
            ] {
                std::fs::write(bin.join(f), f).unwrap();
            }
        }
        StepResult::Ok
    }

    fn ninja_targets(&self, _: &BuildPlan) -> Result<String, String> {
        self.note("ninja_targets");
        if let Some(e) = self.script.lock().unwrap().ninja_error.clone() {
            return Err(e);
        }
        Ok("bin/llama-server: L\nbin/llama-bench: L\nbin/libggml-metal.so: L\nbin/libllama.dylib: L\n".into())
    }

    fn toolchain_versions(&self) -> BTreeMap<String, String> {
        BTreeMap::from([("cmake".into(), "4.4.3".into())])
    }

    fn free_space(&self, _: &Path) -> Option<u64> {
        self.script.lock().unwrap().free
    }

    fn logical_cpus(&self) -> usize {
        8
    }
}

struct Rig {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    deps: JobDeps,
    fake: Arc<Fake>,
}

fn rig_with(fake: Arc<Fake>) -> Rig {
    let tmp = tempfile::tempdir().unwrap();
    let root = without_verbatim_prefix(&std::fs::canonicalize(tmp.path()).unwrap());
    let app = AppPaths::from_root(root.clone());
    // Live and previous runtime sets that must survive every failure.
    for dir in [app.bin_dir(), app.bin_previous_dir()] {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("llama-server"), format!("live:{}", dir.display())).unwrap();
        std::fs::write(dir.join("libllama.dylib"), "lib").unwrap();
    }
    let deps = JobDeps {
        toolchain: fake.clone(),
        paths: BuildPaths::new(&app),
        runtime: RuntimePaths::for_live_dir(&app.bin_dir()).unwrap(),
        config: LlamaBuildConfig::default(),
    };
    Rig {
        _tmp: tmp,
        root,
        deps,
        fake,
    }
}

fn new_rig() -> Rig {
    rig_with(Fake::metal())
}

impl Rig {
    fn options(&self) -> BuildOptions {
        BuildOptions::new(Accelerator::Metal, RefChoice::LatestTag)
    }

    fn shared(&self, id: &str) -> SharedState {
        new_shared(&self.deps, &self.options(), id, true).unwrap()
    }

    fn run(&self, id: &str) -> (Result<Staged, JobEnd>, SharedState) {
        let shared = self.shared(id);
        let r = run_to_staged(&self.deps, &self.options(), id, &shared, &|| false);
        (r, shared)
    }

    /// Fingerprint of both live sets.
    fn live(&self) -> Vec<(String, String)> {
        let app = AppPaths::from_root(self.root.clone());
        let mut out = Vec::new();
        for dir in [app.bin_dir(), app.bin_previous_dir()] {
            let mut names: Vec<_> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            for n in names {
                out.push((
                    format!("{}/{n}", dir.file_name().unwrap().to_string_lossy()),
                    std::fs::read_to_string(dir.join(&n)).unwrap(),
                ));
            }
        }
        out
    }

    fn staging_exists(&self, id: &str) -> bool {
        self.deps.runtime.staging_dir(id).exists()
    }
}

fn failure(r: Result<Staged, JobEnd>) -> BuildFailure {
    match r {
        Err(JobEnd::Failed(f)) => *f,
        Err(JobEnd::Cancelled) => panic!("cancelled, expected a failure"),
        Ok(_) => panic!("expected a failure, got a staged candidate"),
    }
}

#[test]
fn the_happy_path_stages_a_pinned_verified_candidate() {
    let rig = new_rig();
    let (staged, shared) = rig.run("job1");
    let staged = staged.unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(
        rig.fake.calls(),
        [
            "prereqs",
            "plan_inputs",
            "resolve",
            "fetch",
            "configure",
            "compile",
            "ninja_targets"
        ]
    );
    assert_eq!(staged.dir, rig.deps.runtime.staging_dir("job1"));
    assert_eq!(staged.dir.parent(), rig.deps.runtime.live.parent());
    assert_eq!(staged.label, "b11727 · src · metal");
    assert_eq!(staged.intent.pinned_commit.as_deref(), Some(COMMIT));
    assert_eq!(staged.manifest.llama_cpp.commit.as_deref(), Some(COMMIT));
    assert!(
        staged
            .manifest
            .cmake_args
            .iter()
            .any(|a| a == "-DGGML_METAL=ON")
    );
    assert_eq!(staged.manifest.toolchain["cmake"], "4.4.3");
    for f in [
        "llama-server",
        "llama-bench",
        "libggml-metal.so",
        "libllama.dylib",
    ] {
        assert!(staged.dir.join(f).is_file(), "{f}");
    }
    let record = with(&shared, |s| s.record.clone());
    assert_eq!(record.resolved.unwrap().commit, COMMIT);
    assert_eq!(record.progress, Some(Progress { done: 3, total: 3 }));
    assert!(record.phase_secs.contains_key("configure"));
    // The tree is no longer dirty after a good compile.
    assert!(!marker_path(&rig.deps.paths.build_tree(Accelerator::Metal)).exists());
    // And the live sets were never touched.
    assert!(
        rig.live()
            .iter()
            .all(|(_, body)| body.starts_with("live:") || body == "lib")
    );
}

#[test]
fn the_build_directories_handed_to_cmake_are_canonical() {
    let rig = new_rig();
    rig.run("job1").0.unwrap_or_else(|e| panic!("{e:?}"));
    let plan = rig.fake.plans.lock().unwrap()[0].clone();
    assert_eq!(
        plan.source_dir,
        without_verbatim_prefix(&std::fs::canonicalize(&plan.source_dir).unwrap())
    );
    assert_eq!(
        plan.build_dir,
        without_verbatim_prefix(&std::fs::canonicalize(&plan.build_dir).unwrap())
    );
    assert!(plan.build_dir.starts_with(&rig.root));
}

/// Fail every phase: the failure names the phase, carries the command and log
/// tail, leaves no staging directory and never touches a live runtime set.
#[test]
fn failing_every_phase_reports_it_and_preserves_the_live_sets() {
    for phase in [Phase::Fetch, Phase::Configure, Phase::Compile] {
        let rig = new_rig();
        let before = rig.live();
        rig.fake.set(|s| {
            s.fail_at = Some(phase);
            s.fail_log = vec!["boom line".into()];
        });
        let (r, shared) = rig.run("j");
        let f = failure(r);
        assert_eq!(f.phase, phase, "{phase:?}");
        assert_eq!(f.exit_code, Some(1));
        assert_eq!(
            f.command.as_deref(),
            Some(format!("fake {}", phase.as_str()).as_str())
        );
        assert!(f.log_tail.iter().any(|l| l == "boom line"));
        assert!(f.log_path.ends_with("j.log"));
        assert!(!rig.staging_exists("j"));
        assert_eq!(rig.live(), before, "{phase:?} modified a live set");
        assert_eq!(with(&shared, |s| s.record.failure.clone()).unwrap(), f);
    }
}

#[test]
fn failing_preflight_and_stage_also_preserve_the_live_sets() {
    let rig = new_rig();
    let before = rig.live();
    rig.fake.set(|s| s.prereqs_ready = false);
    let f = failure(rig.run("p").0);
    assert_eq!(f.phase, Phase::Preflight);
    assert!(f.summary.contains("1 failing"));
    assert!(f.log_tail.iter().any(|l| l.contains("cmake was not found")));
    assert_eq!(
        rig.fake.calls(),
        ["prereqs"],
        "nothing ran after a failed prerequisite"
    );

    let rig2 = new_rig();
    rig2.fake
        .set(|s| s.ninja_error = Some("ninja: error: loading 'build.ninja'".into()));
    let before2 = rig2.live();
    let f = failure(rig2.run("s").0);
    assert_eq!(f.phase, Phase::Stage);
    assert_eq!(rig2.live(), before2);
    assert_eq!(before.len(), rig.live().len());

    // A compile that "succeeds" without producing the executables fails stage.
    let rig3 = new_rig();
    rig3.fake.set(|s| s.skip_outputs = true);
    let before3 = rig3.live();
    let f = failure(rig3.run("m").0);
    assert_eq!(f.phase, Phase::Stage);
    assert!(f.summary.contains("did not produce"), "{}", f.summary);
    assert!(!rig3.staging_exists("m"));
    assert_eq!(rig3.live(), before3);
}

#[test]
fn cancel_at_any_boundary_stops_without_touching_anything_live() {
    for after in 1..=7 {
        let rig = new_rig();
        let before = rig.live();
        let shared = rig.shared("c");
        let fake = rig.fake.clone();
        let cancel = move || fake.calls().len() >= after;
        let r = run_to_staged(&rig.deps, &rig.options(), "c", &shared, &cancel);
        match r {
            Err(JobEnd::Cancelled) => {}
            Ok(_) => assert!(
                after >= 7,
                "cancel after {after} calls still staged a candidate"
            ),
            Err(other) => panic!("{other:?}"),
        }
        assert_eq!(rig.live(), before);
        if after < 7 {
            assert!(
                !rig.staging_exists("c"),
                "cancel after {after} left staging"
            );
        }
    }
}

#[test]
fn a_step_that_reports_cancelled_ends_the_job_as_cancelled() {
    struct Cancelling(Arc<Fake>);
    impl Toolchain for Cancelling {
        fn os(&self) -> TargetOs {
            self.0.os()
        }
        fn prereqs(&self, o: &BuildOptions) -> PrereqReport {
            self.0.prereqs(o)
        }
        fn plan_inputs(&self, o: &BuildOptions) -> Result<PlanInputs, String> {
            self.0.plan_inputs(o)
        }
        fn resolve(&self, o: &BuildOptions) -> Result<ResolvedRef, String> {
            self.0.resolve(o)
        }
        fn fetch(&self, a: &Path, b: &ResolvedRef, c: &str, io: &mut StepIo) -> StepResult {
            self.0.fetch(a, b, c, io)
        }
        fn configure(&self, p: &BuildPlan, io: &mut StepIo) -> StepResult {
            self.0.configure(p, io)
        }
        fn compile(&self, _: &BuildPlan, _: usize, _: bool, _: &mut StepIo) -> StepResult {
            StepResult::Cancelled
        }
        fn ninja_targets(&self, p: &BuildPlan) -> Result<String, String> {
            self.0.ninja_targets(p)
        }
        fn toolchain_versions(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }
        fn free_space(&self, p: &Path) -> Option<u64> {
            self.0.free_space(p)
        }
        fn logical_cpus(&self) -> usize {
            4
        }
    }
    let rig = new_rig();
    let deps = JobDeps {
        toolchain: Arc::new(Cancelling(rig.fake.clone())),
        ..rig.deps.clone()
    };
    let shared = rig.shared("x");
    let r = run_to_staged(&deps, &rig.options(), "x", &shared, &|| false);
    assert_eq!(r.err(), Some(JobEnd::Cancelled));
    // An interrupted compile leaves the tree flagged for revalidation.
    assert!(marker_path(&rig.deps.paths.build_tree(Accelerator::Metal)).exists());
}

// ------------------------------------------------------------------- disk

#[test]
fn too_little_space_at_preflight_fails_early_with_the_disk_hint() {
    let rig = new_rig();
    rig.fake.set(|s| s.free = Some(100 << 20));
    let before = rig.live();
    let f = failure(rig.run("d").0);
    assert_eq!(f.phase, Phase::Preflight);
    assert_eq!(f.hint.as_ref().map(|h| h.id.as_str()), Some("disk_full"));
    assert!(f.hint.unwrap().message.contains("never removed"));
    assert_eq!(
        rig.fake.calls(),
        ["prereqs", "plan_inputs"],
        "no checkout or compile on a full disk"
    );
    assert_eq!(rig.live(), before);
}

#[test]
fn the_windows_floor_is_enforced_even_when_the_estimate_is_smaller() {
    let fake = Arc::new(Fake {
        os: TargetOs::Windows,
        script: Mutex::new(Script {
            prereqs_ready: true,
            free: Some(10 << 30),
            ..Default::default()
        }),
        calls: Mutex::new(vec![]),
        plans: Mutex::new(vec![]),
        cuda: true,
        cuda_root: None,
    });
    let rig = rig_with(fake);
    let shared = rig.shared("w");
    let opts = BuildOptions::new(Accelerator::Cuda, RefChoice::LatestTag);
    let f = failure(run_to_staged(&rig.deps, &opts, "w", &shared, &|| false));
    assert_eq!(f.phase, Phase::Preflight);
    assert!(
        f.log_tail.iter().any(|l| l.contains("floor 20480 MiB")),
        "{:?}",
        f.log_tail
    );
}

#[test]
fn running_out_of_space_while_compiling_is_classified_and_preserves_live_sets() {
    let rig = new_rig();
    let before = rig.live();
    rig.fake.set(|s| {
        s.fail_at = Some(Phase::Compile);
        s.fail_log = include_str!("../../tests/fixtures/llama_build/disk_full.log")
            .lines()
            .map(str::to_string)
            .collect();
        s.free = Some(u64::MAX);
    });
    let f = failure(rig.run("d").0);
    assert_eq!(f.phase, Phase::Compile);
    assert_eq!(f.hint.unwrap().id, "disk_full");
    assert!(f.first_error.is_some());
    assert_eq!(rig.live(), before);
    assert!(!rig.staging_exists("d"));
    // The partial tree is flagged so the next job revalidates it.
    assert!(marker_path(&rig.deps.paths.build_tree(Accelerator::Metal)).exists());
}

#[test]
fn staging_with_too_little_room_fails_without_writing_or_deleting_runtimes() {
    // Plenty of room for preflight, none by the time staging measures it.
    struct Shrinking(Arc<Fake>, Mutex<usize>);
    impl Toolchain for Shrinking {
        fn os(&self) -> TargetOs {
            self.0.os()
        }
        fn prereqs(&self, o: &BuildOptions) -> PrereqReport {
            self.0.prereqs(o)
        }
        fn plan_inputs(&self, o: &BuildOptions) -> Result<PlanInputs, String> {
            self.0.plan_inputs(o)
        }
        fn resolve(&self, o: &BuildOptions) -> Result<ResolvedRef, String> {
            self.0.resolve(o)
        }
        fn fetch(&self, a: &Path, b: &ResolvedRef, c: &str, io: &mut StepIo) -> StepResult {
            self.0.fetch(a, b, c, io)
        }
        fn configure(&self, p: &BuildPlan, io: &mut StepIo) -> StepResult {
            self.0.configure(p, io)
        }
        fn compile(&self, p: &BuildPlan, j: usize, l: bool, io: &mut StepIo) -> StepResult {
            self.0.compile(p, j, l, io)
        }
        fn ninja_targets(&self, p: &BuildPlan) -> Result<String, String> {
            self.0.ninja_targets(p)
        }
        fn toolchain_versions(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }
        fn free_space(&self, _: &Path) -> Option<u64> {
            let mut n = self.1.lock().unwrap();
            *n += 1;
            // Calls: preflight, configure ctx, compile ctx, then staging.
            if self.0.calls().contains(&"ninja_targets".to_string()) {
                Some(1 << 20)
            } else {
                Some(u64::MAX)
            }
        }
        fn logical_cpus(&self) -> usize {
            4
        }
    }
    let rig = new_rig();
    let before = rig.live();
    let deps = JobDeps {
        toolchain: Arc::new(Shrinking(rig.fake.clone(), Mutex::new(0))),
        ..rig.deps.clone()
    };
    let shared = rig.shared("s");
    let f = failure(run_to_staged(&deps, &rig.options(), "s", &shared, &|| {
        false
    }));
    assert_eq!(f.phase, Phase::Stage);
    assert!(f.summary.contains("staging needs"), "{}", f.summary);
    assert_eq!(f.hint.unwrap().id, "disk_full");
    assert!(!rig.staging_exists("s"));
    assert_eq!(rig.live(), before);
}

#[test]
fn the_recheck_before_the_journal_discards_staging_and_keeps_every_runtime() {
    let rig = new_rig();
    let (staged, shared) = rig.run("r");
    let staged = staged.unwrap_or_else(|e| panic!("{e:?}"));
    let before = rig.live();
    assert!(recheck_before_journal(&rig.deps, &staged, &shared).is_ok());
    assert!(staged.dir.exists());
    rig.fake.set(|s| s.free = Some(1 << 20));
    let f = match recheck_before_journal(&rig.deps, &staged, &shared) {
        Err(JobEnd::Failed(f)) => *f,
        other => panic!("expected a failure, got {other:?}"),
    };
    assert_eq!(f.phase, Phase::Promote);
    assert!(f.summary.contains("Nothing was changed"));
    assert!(!staged.dir.exists());
    assert_eq!(rig.live(), before);
}

// --------------------------------------------------------------- CUDA / FA

fn cuda_rig(cache_extra_fa: &str) -> (Rig, BuildOptions) {
    let fake = Arc::new(Fake {
        os: TargetOs::Windows,
        script: Mutex::new(Script {
            prereqs_ready: true,
            free: Some(500 << 30),
            ..Default::default()
        }),
        calls: Mutex::new(vec![]),
        plans: Mutex::new(vec![]),
        cuda: cache_extra_fa.is_empty(),
        cuda_root: None,
    });
    fake.set(|s| s.cache_extra = cache_extra_fa.to_string());
    (
        rig_with(fake),
        BuildOptions::new(Accelerator::Cuda, RefChoice::Master),
    )
}

#[test]
fn a_cuda_configure_that_is_not_fa_all_is_refused_before_compiling() {
    let (rig, opts) =
        cuda_rig("GGML_CUDA_FA_QUANTS:STRING=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16\n");
    let shared = new_shared(&rig.deps, &opts, "f", true).unwrap();
    let f = failure(run_to_staged(&rig.deps, &opts, "f", &shared, &|| false));
    assert_eq!(f.phase, Phase::Configure);
    assert!(f.summary.contains("instead of all"), "{}", f.summary);
    assert!(!rig.fake.calls().contains(&"compile".to_string()));
}

#[test]
fn a_cuda_configure_with_no_fa_record_is_refused_too() {
    let (rig, opts) = cuda_rig("CMAKE_BUILD_TYPE:STRING=Release\n");
    let shared = new_shared(&rig.deps, &opts, "f", true).unwrap();
    let f = failure(run_to_staged(&rig.deps, &opts, "f", &shared, &|| false));
    assert!(f.summary.contains("(unset)"), "{}", f.summary);
}

// ---------------------------------------------------- tree reuse and healing

#[test]
fn stale_cached_paths_are_purged_for_the_next_configure() {
    let rig = new_rig();
    let tree = rig.deps.paths.build_tree(Accelerator::Metal);
    let src = std::fs::canonicalize(
        create_canonical_within(&rig.deps.paths.root, &rig.deps.paths.source).unwrap(),
    )
    .unwrap();
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(
        tree.join("CMakeCache.txt"),
        format!(
            "CMAKE_HOME_DIRECTORY:INTERNAL={}\nCMAKE_GENERATOR:INTERNAL=Ninja\nOPENSSL_ROOT_DIR:PATH=/definitely/gone/openssl\n",
            src.display()
        ),
    )
    .unwrap();
    let (r, shared) = rig.run("h");
    r.unwrap_or_else(|e| panic!("{e:?}"));
    let plan = rig.fake.plans.lock().unwrap()[0].clone();
    assert_eq!(plan.stale_cache_vars, ["OPENSSL_ROOT_DIR"]);
    let log = with(&shared, |s| s.log.all()).join("\n");
    assert!(log.contains("Stale CMake cache") && log.contains("OPENSSL_ROOT_DIR"));
}

#[test]
fn a_tree_for_another_source_directory_or_generator_is_reset() {
    let rig = new_rig();
    let tree = rig.deps.paths.build_tree(Accelerator::Metal);
    std::fs::create_dir_all(tree.join("deep")).unwrap();
    std::fs::write(tree.join("deep/old.o"), "x").unwrap();
    std::fs::write(
        tree.join("CMakeCache.txt"),
        "CMAKE_HOME_DIRECTORY:INTERNAL=/some/other/checkout\nCMAKE_GENERATOR:INTERNAL=Ninja\n",
    )
    .unwrap();
    let (r, shared) = rig.run("w");
    r.unwrap_or_else(|e| panic!("{e:?}"));
    assert!(!tree.join("deep/old.o").exists());
    assert!(
        with(&shared, |s| s.log.all())
            .iter()
            .any(|l| l.contains("different source directory"))
    );
}

#[test]
fn wipe_build_dir_clears_the_tree_before_configuring() {
    let rig = new_rig();
    let tree = rig.deps.paths.build_tree(Accelerator::Metal);
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(tree.join("old.o"), "x").unwrap();
    let mut opts = rig.options();
    opts.wipe_build_dir = true;
    let shared = new_shared(&rig.deps, &opts, "w", true).unwrap();
    run_to_staged(&rig.deps, &opts, "w", &shared, &|| false).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(!tree.join("old.o").exists());
}

#[test]
fn tree_reuse_rules() {
    let plan = BuildPlan {
        accelerator: Accelerator::Metal,
        os: TargetOs::Macos,
        source_dir: PathBuf::from("/app/llama-build/src"),
        build_dir: PathBuf::from("/app/llama-build/build-metal"),
        ccache: false,
        apple_silicon: true,
        cuda: None,
        stale_cache_vars: vec![],
    };
    let good =
        "CMAKE_HOME_DIRECTORY:INTERNAL=/app/llama-build/src\nCMAKE_GENERATOR:INTERNAL=Ninja\n";
    assert_eq!(tree_needs_wipe(Some(good), &plan, false), None);
    assert_eq!(
        tree_needs_wipe(Some(good), &plan, true),
        None,
        "a healthy cache survives a dirty marker"
    );
    assert_eq!(
        tree_needs_wipe(None, &plan, false),
        None,
        "a fresh tree is simply configured"
    );
    assert!(
        tree_needs_wipe(None, &plan, true)
            .unwrap()
            .contains("interrupted")
    );
    assert!(
        tree_needs_wipe(Some("CMAKE_GENERATOR:INTERNAL=Ninja\n"), &plan, false)
            .unwrap()
            .contains("no source directory")
    );
    assert!(tree_needs_wipe(Some("CMAKE_HOME_DIRECTORY:INTERNAL=/app/llama-build/src\nCMAKE_GENERATOR:INTERNAL=Unix Makefiles\n"), &plan, false).unwrap().contains("generator"));
    assert_eq!(
        tree_needs_wipe(
            Some(
                "CMAKE_HOME_DIRECTORY:INTERNAL=/app/llama-build/src/\nCMAKE_GENERATOR:INTERNAL=Ninja\n"
            ),
            &plan,
            false
        ),
        None,
        "trailing separators do not matter"
    );
}

#[test]
fn an_interrupted_configure_with_no_cache_resets_the_tree_on_the_next_job() {
    let rig = new_rig();
    let tree = rig.deps.paths.build_tree(Accelerator::Metal);
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(tree.join("half-written.ninja"), "x").unwrap();
    std::fs::write(marker_path(&tree), "dead-job").unwrap();
    rig.run("n").0.unwrap_or_else(|e| panic!("{e:?}"));
    assert!(!tree.join("half-written.ninja").exists());
    assert!(!marker_path(&tree).exists());
}

// ------------------------------------------------------------------ logging

#[test]
fn progress_lines_parse() {
    assert_eq!(
        parse_progress("[12/140] Building CXX object x"),
        Some(Progress {
            done: 12,
            total: 140
        })
    );
    assert_eq!(
        parse_progress("  [140/140] Linking"),
        Some(Progress {
            done: 140,
            total: 140
        })
    );
    for bad in [
        "[0/0] x",
        "[5/3] x",
        "[a/b] x",
        "no brackets",
        "[1/2",
        "[1] x",
    ] {
        assert_eq!(parse_progress(bad), None, "{bad}");
    }
}

#[test]
fn the_ring_keeps_the_last_2000_lines_and_the_failure_tail_is_80() {
    let mut log = JobLog::new(None, LOG_FILE_CAP_BYTES).unwrap();
    for i in 0..2500 {
        log.push(&format!("line {i}"));
    }
    assert_eq!(log.all().len(), LOG_RING_LINES);
    assert_eq!(log.all()[0], "line 500");
    assert_eq!(log.tail(80).len(), FAILURE_TAIL_LINES);
    assert_eq!(log.tail(80).last().unwrap(), "line 2499");
    assert_eq!(log.tail(5000).len(), LOG_RING_LINES);
}

#[test]
fn the_disk_log_is_bounded_with_an_explicit_marker_while_the_ring_continues() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("j.log");
    let mut log = JobLog::new(Some(&file), 200).unwrap();
    for i in 0..100 {
        log.push(&format!("a fairly long log line number {i}"));
    }
    assert!(log.truncated());
    let body = std::fs::read_to_string(&file).unwrap();
    assert!(body.len() < 400, "{} bytes", body.len());
    assert_eq!(body.matches("[log truncated").count(), 1);
    assert_eq!(log.tail(1)[0], "a fairly long log line number 99");
}

#[test]
fn the_job_record_is_persisted_atomically_at_each_phase() {
    let rig = new_rig();
    let (r, _) = rig.run("p1");
    r.unwrap_or_else(|e| panic!("{e:?}"));
    let record = read_record(&rig.deps.paths.record_file("p1")).unwrap();
    assert_eq!(record.id, "p1");
    assert_eq!(record.phase, Phase::Stage);
    assert_eq!(record.status, JobStatus::Running);
    assert!(rig.deps.paths.log_file("p1").is_file());
    assert!(
        std::fs::read_to_string(rig.deps.paths.log_file("p1"))
            .unwrap()
            .contains("== phase: compile")
    );
    assert!(
        !rig.deps
            .paths
            .record_file("p1")
            .with_extension("json.tmp")
            .exists()
    );
}

#[test]
fn finishing_stamps_status_outcome_and_the_last_build_duration() {
    let rig = new_rig();
    let shared = rig.shared("f1");
    with(&shared, |s| s.record.started_unix -= 42);
    finish(&shared, JobStatus::Succeeded, "installed");
    let record = read_record(&rig.deps.paths.record_file("f1")).unwrap();
    assert_eq!(record.status, JobStatus::Succeeded);
    assert_eq!(record.phase, Phase::Done);
    assert_eq!(record.outcome.as_deref(), Some("installed"));
    assert!(last_build_seconds(&rig.deps.paths).unwrap() >= 42);
    let failed = rig.shared("f2");
    finish(&failed, JobStatus::Failed, "x");
    assert!(JobStatus::Failed.is_terminal() && !JobStatus::Running.is_terminal());
}

// ---------------------------------------------------- restart and retention

#[test]
fn startup_marks_running_jobs_interrupted_and_flags_their_tree() {
    let rig = new_rig();
    let tree = rig.deps.paths.build_tree(Accelerator::Metal);
    std::fs::create_dir_all(&tree).unwrap();
    let shared = rig.shared("dead");
    with(&shared, |s| {
        s.record.phase = Phase::Compile;
        s.persist();
    });
    let done = rig.shared("fine");
    finish(&done, JobStatus::Succeeded, "ok");
    let marked = recover_interrupted(&rig.deps.paths);
    assert_eq!(marked, ["dead"]);
    let record = read_record(&rig.deps.paths.record_file("dead")).unwrap();
    assert_eq!(record.status, JobStatus::Interrupted);
    assert!(
        marker_path(&tree).exists(),
        "an interrupted compile marks needs_reconfigure"
    );
    assert_eq!(
        read_record(&rig.deps.paths.record_file("fine"))
            .unwrap()
            .status,
        JobStatus::Succeeded
    );
    assert!(
        recover_interrupted(&rig.deps.paths).is_empty(),
        "idempotent"
    );
}

#[test]
fn an_interrupted_preflight_does_not_dirty_a_tree() {
    let rig = new_rig();
    let tree = rig.deps.paths.build_tree(Accelerator::Metal);
    std::fs::create_dir_all(&tree).unwrap();
    let _ = rig.shared("early");
    recover_interrupted(&rig.deps.paths);
    assert!(!marker_path(&tree).exists());
}

#[test]
fn recovery_runs_before_the_staging_sweep_so_diagnostics_survive() {
    // The ordering contract: recover_interrupted only edits records and
    // markers; it never deletes a staging directory (that is the sweep's job,
    // which runs afterwards and keeps journal-referenced sets).
    let rig = new_rig();
    let shared = rig.shared("dead");
    with(&shared, |s| {
        s.record.phase = Phase::Stage;
        s.persist();
    });
    let staging = rig.deps.runtime.staging_dir("dead");
    std::fs::create_dir_all(&staging).unwrap();
    std::fs::write(staging.join("llama-server"), "x").unwrap();
    recover_interrupted(&rig.deps.paths);
    assert!(staging.join("llama-server").exists());
}

#[test]
fn retention_keeps_three_completed_pairs_and_exempts_running_and_referenced_jobs() {
    let rig = new_rig();
    let paths = &rig.deps.paths;
    let mut ids = Vec::new();
    for i in 0..6 {
        let id = format!("done{i}");
        let s = rig.shared(&id);
        finish(&s, JobStatus::Succeeded, "ok");
        // Distinct mtimes: oldest first.
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(1000 - i * 100);
        for f in [paths.log_file(&id), paths.record_file(&id)] {
            std::fs::File::options()
                .write(true)
                .open(&f)
                .unwrap()
                .set_modified(when)
                .unwrap();
        }
        ids.push(id);
    }
    let running = rig.shared("running-now");
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(9000);
    for f in [
        paths.log_file("running-now"),
        paths.record_file("running-now"),
    ] {
        std::fs::File::options()
            .write(true)
            .open(&f)
            .unwrap()
            .set_modified(old)
            .unwrap();
    }
    drop(running);
    let removed = prune_records(paths, &["done0".to_string()]).unwrap();
    assert_eq!(removed, vec!["done2".to_string(), "done1".to_string()]);
    for keep in ["done0", "done3", "done4", "done5", "running-now"] {
        assert!(paths.record_file(keep).exists(), "{keep}");
        assert!(paths.log_file(keep).exists(), "{keep}");
    }
    for gone in ["done1", "done2"] {
        assert!(!paths.record_file(gone).exists() && !paths.log_file(gone).exists());
    }
}

#[test]
fn single_flight_names_the_running_job() {
    let ownership = Ownership::default();
    let first = start_exclusive(&ownership, "job-a").unwrap();
    let second = start_exclusive(&ownership, "job-b").err().unwrap();
    assert_eq!(second.holder_id, "job-a");
    drop(first);
    assert!(start_exclusive(&ownership, "job-b").is_ok());
}

#[test]
fn hints_come_from_the_failed_phases_log() {
    let rig = new_rig();
    rig.fake.set(|s| {
        s.fail_at = Some(Phase::Configure);
        s.fail_log = include_str!("../../tests/fixtures/llama_build/stale_cache_openssl.log")
            .lines()
            .map(str::to_string)
            .collect();
    });
    let f = failure(rig.run("h").0);
    assert_eq!(f.hint.unwrap().id, "stale_cache");
    assert!(f.first_error.unwrap().starts_with("CMake Error"));
}
