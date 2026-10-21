use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use llama_monitor::llama_build::cache::{LlamaBuildConfig, Ownership};
use llama_monitor::llama_build::job::{
    JobDeps, JobStatus, Phase, PlanInputs, StepIo, StepResult, Toolchain,
};
use llama_monitor::llama_build::orchestrate::{BuildStart, run_build};
use llama_monitor::llama_build::paths::BuildPaths;
use llama_monitor::llama_build::prereqs::{Check, Collector, PrereqReport};
use llama_monitor::llama_build::source::ResolvedRef;
use llama_monitor::llama_build::{Accelerator, BuildOptions, BuildPlan, RefChoice, TargetOs};
use llama_monitor::llama_runtime::gate::RuntimeGate;
use llama_monitor::llama_runtime::promote::{PromoteHost, RuntimePaths, VerifyStage};
use llama_monitor::llama_runtime::verify::{VerificationIntent, VerificationReport};
use llama_monitor::paths::AppPaths;

const COMMIT: &str = "098ce526b0000000000000000000000000000000";

struct Fake {
    fail: Option<Phase>,
    cancel: Arc<AtomicBool>,
    cancel_compile: bool,
    leave_records: bool,
    panic_compile: bool,
    journal_compile: bool,
    root: std::path::PathBuf,
    unpinned: Mutex<Vec<String>>,
    block_compile: AtomicBool,
    compile_entered: AtomicBool,
}

impl Toolchain for Fake {
    fn os(&self) -> TargetOs {
        TargetOs::Linux
    }
    fn prereqs(&self, o: &BuildOptions) -> PrereqReport {
        let mut c = Collector::default();
        c.push(if self.fail == Some(Phase::Preflight) {
            Check::fail("fake", "fake", "preflight failed")
        } else {
            Check::pass("fake", "fake", "ok")
        });
        PrereqReport::from_collector(self.os(), o.accelerator, c)
    }
    fn plan_inputs(&self, _: &BuildOptions) -> Result<PlanInputs, String> {
        Ok(PlanInputs {
            ccache: false,
            apple_silicon: false,
            cuda: None,
            cuda_root: None,
            device_id: None,
        })
    }
    fn resolve(&self, _: &BuildOptions) -> Result<ResolvedRef, String> {
        Ok(ResolvedRef {
            reference: "b11727".into(),
            commit: COMMIT.into(),
            build_number: Some(11727),
        })
    }
    fn fetch(&self, _: &Path, _: &ResolvedRef, _: &str, _: &mut StepIo) -> StepResult {
        self.step(Phase::Fetch)
    }
    fn configure(&self, p: &BuildPlan, _: &mut StepIo) -> StepResult {
        if self.fail == Some(Phase::Configure) {
            return self.step(Phase::Configure);
        }
        std::fs::write(
            p.build_dir.join("CMakeCache.txt"),
            format!(
                "CMAKE_HOME_DIRECTORY:INTERNAL={}\nCMAKE_GENERATOR:INTERNAL=Ninja\n",
                p.source_dir.display()
            ),
        )
        .unwrap();
        StepResult::Ok
    }
    fn compile(&self, p: &BuildPlan, _: usize, _: bool, io: &mut StepIo) -> StepResult {
        self.compile_entered.store(true, Ordering::SeqCst);
        while self.block_compile.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if (io.cancel)() {
            return StepResult::Cancelled;
        }
        if self.leave_records {
            let paths = BuildPaths::new(&AppPaths::from_root(self.root.clone()));
            std::fs::create_dir_all(paths.process_records.parent().unwrap()).unwrap();
            std::fs::write(paths.process_records, "unresolved containment").unwrap();
            let staging = RuntimePaths::for_live_dir(&self.root.join("bin"))
                .unwrap()
                .staging_dir(io.job_id);
            std::fs::create_dir_all(&staging).unwrap();
            std::fs::write(staging.join("diagnostic"), "keep").unwrap();
        }
        if self.panic_compile {
            panic!("simulated blocking worker panic");
        }
        if self.journal_compile {
            std::fs::write(
                self.root.join("runtime-transaction.json"),
                "corrupt journal",
            )
            .unwrap();
        }
        if self.cancel_compile {
            self.cancel.store(true, Ordering::SeqCst);
            return StepResult::Cancelled;
        }
        if self.fail == Some(Phase::Compile) {
            return self.step(Phase::Compile);
        }
        let bin = p.build_dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        for name in [
            "llama-server",
            "llama-bench",
            "libggml-cpu.so",
            "libllama.so",
        ] {
            std::fs::write(bin.join(name), format!("new-{name}")).unwrap();
        }
        StepResult::Ok
    }
    fn ninja_targets(&self, _: &BuildPlan) -> Result<String, String> {
        if self.fail == Some(Phase::Stage) {
            return Err("target inventory failed".into());
        }
        Ok(
            "bin/llama-server: L\nbin/llama-bench: L\nbin/libggml-cpu.so: L\nbin/libllama.so: L\n"
                .into(),
        )
    }
    fn toolchain_versions(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }
    fn free_space(&self, _: &Path) -> Option<u64> {
        Some(u64::MAX)
    }
    fn logical_cpus(&self) -> usize {
        4
    }
    fn unpin(&self, _: &Path, id: &str) {
        self.unpinned.lock().unwrap().push(id.into());
    }
}
impl Fake {
    fn step(&self, p: Phase) -> StepResult {
        if self.fail == Some(p) {
            StepResult::Failed {
                exit_code: Some(1),
                command: "fake failure".into(),
                detail: None,
            }
        } else {
            StepResult::Ok
        }
    }
}

struct Host {
    cancel: Arc<AtomicBool>,
    calls: Mutex<Vec<VerifyStage>>,
    reject: bool,
    die_after_begin: bool,
    cancel_in_verify: bool,
    /// A model is running and its restart fails.
    model_restart_fails: bool,
}
impl llama_monitor::llama_runtime::promote::Faults for Host {
    fn at(&self, point: &str) -> llama_monitor::llama_runtime::promote::Fault {
        use llama_monitor::llama_runtime::promote::Fault;
        if self.die_after_begin && point == "journal.begin.after" {
            Fault::Die
        } else {
            Fault::None
        }
    }
}
impl PromoteHost for Host {
    fn faults(&self) -> &dyn llama_monitor::llama_runtime::promote::Faults {
        self
    }
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    fn model_running(&self) -> bool {
        self.model_restart_fails
    }
    async fn stop_model(&self) -> Result<(), String> {
        assert!(self.model_restart_fails, "no model");
        Ok(())
    }
    async fn restart_model(&self) -> Result<(), String> {
        assert!(self.model_restart_fails, "no model");
        Err("model failed to start".into())
    }
    async fn verify(
        &self,
        _: &Path,
        _: &str,
        intent: &VerificationIntent,
        stage: VerifyStage,
    ) -> Result<VerificationReport, String> {
        self.calls.lock().unwrap().push(stage);
        if self.cancel_in_verify {
            self.cancel.store(true, Ordering::SeqCst);
        }
        if self.reject {
            return Err("candidate rejected".into());
        }
        Ok(VerificationReport {
            backend: intent.expected_backend.unwrap(),
            build_number: Some(11727),
            build_commit: Some(COMMIT.into()),
            identity_fingerprint: None,
            notes: vec![],
            evidence_unavailable: false,
        })
    }
}

fn setup(
    fail: Option<Phase>,
    cancel_compile: bool,
    records: bool,
    panic: bool,
    journal: bool,
) -> (tempfile::TempDir, JobDeps, Arc<Fake>, Host) {
    let tmp = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(tmp.path()).unwrap();
    for (name, bytes) in [("bin", "old-live"), ("bin-previous", "old-previous")] {
        std::fs::create_dir(root.join(name)).unwrap();
        std::fs::write(root.join(name).join("llama-server"), bytes).unwrap();
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let fake = Arc::new(Fake {
        fail,
        cancel: cancel.clone(),
        cancel_compile,
        leave_records: records,
        panic_compile: panic,
        journal_compile: journal,
        root: root.clone(),
        unpinned: Mutex::new(vec![]),
        block_compile: AtomicBool::new(false),
        compile_entered: AtomicBool::new(false),
    });
    let app = AppPaths::from_root(root);
    let deps = JobDeps {
        toolchain: fake.clone(),
        paths: BuildPaths::new(&app),
        runtime: RuntimePaths::for_live_dir(&app.bin_dir()).unwrap(),
        config: LlamaBuildConfig::default(),
    };
    (
        tmp,
        deps,
        fake,
        Host {
            cancel,
            calls: Mutex::new(vec![]),
            reject: false,
            die_after_begin: false,
            cancel_in_verify: false,
            model_restart_fails: false,
        },
    )
}
fn options() -> BuildOptions {
    BuildOptions::new(Accelerator::Cpu, RefChoice::LatestTag)
}
fn assert_old(deps: &JobDeps) {
    assert_eq!(
        std::fs::read_to_string(deps.runtime.live.join("llama-server")).unwrap(),
        "old-live"
    );
    assert_eq!(
        std::fs::read_to_string(deps.runtime.previous.join("llama-server")).unwrap(),
        "old-previous"
    );
}

#[tokio::test]
async fn llama_build_success_uses_shared_verifier_and_retains_previous() {
    let (_tmp, deps, fake, host) = setup(None, false, false, false, false);
    let result = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(result.record.status, JobStatus::Succeeded);
    assert_eq!(
        std::fs::read_to_string(deps.runtime.live.join("llama-server")).unwrap(),
        "new-llama-server"
    );
    assert_eq!(
        std::fs::read_to_string(deps.runtime.previous.join("llama-server")).unwrap(),
        "old-live"
    );
    assert_eq!(host.calls.lock().unwrap().len(), 2);
    assert_eq!(fake.unpinned.lock().unwrap().len(), 1);
    assert!(!deps.runtime.journal.exists());
}

#[tokio::test]
async fn llama_build_each_pre_promotion_failure_preserves_live_and_previous() {
    for phase in [
        Phase::Preflight,
        Phase::Fetch,
        Phase::Configure,
        Phase::Compile,
        Phase::Stage,
    ] {
        let (_tmp, deps, _, host) = setup(Some(phase), false, false, false, false);
        let r = run_build(
            deps.clone(),
            &host,
            &Arc::new(RuntimeGate::default()),
            &Ownership::default(),
            options(),
            host.cancel.clone(),
            |_, _| {},
        )
        .await
        .unwrap();
        assert_eq!(r.record.status, JobStatus::Failed, "{phase:?}");
        assert_old(&deps);
        assert!(host.calls.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn llama_build_compile_cancellation_preserves_runtime_sets() {
    let (_tmp, deps, _, host) = setup(None, true, false, false, false);
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Cancelled);
    assert_old(&deps);
    assert!(host.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn llama_build_rejected_candidate_restores_both_sets() {
    let (_tmp, deps, _, mut host) = setup(None, false, false, false, false);
    host.reject = true;
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Failed);
    assert_old(&deps);
}

#[tokio::test]
async fn llama_build_rejected_candidate_with_a_failed_model_restart_says_so() {
    let (_tmp, deps, _, mut host) = setup(None, false, false, false, false);
    host.reject = true;
    host.model_restart_fails = true;
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Failed);
    let outcome = r.record.outcome.clone().unwrap();
    assert!(
        outcome.contains("previous runtime was restored")
            && outcome.contains("model did not restart")
            && outcome.contains("model failed to start"),
        "{outcome}"
    );
    assert_old(&deps);
}

#[tokio::test]
async fn llama_build_unresolved_records_preserve_pin_and_staging_on_error_and_panic() {
    for panic in [false, true] {
        let (_tmp, deps, fake, host) = setup(Some(Phase::Compile), false, true, panic, false);
        let ownership = Ownership::default();
        let r = run_build(
            deps.clone(),
            &host,
            &Arc::new(RuntimeGate::default()),
            &ownership,
            options(),
            host.cancel.clone(),
            |_, _| {},
        )
        .await
        .unwrap();
        assert_eq!(r.record.status, JobStatus::Failed);
        assert!(fake.unpinned.lock().unwrap().is_empty());
        assert_eq!(
            std::fs::read_to_string(deps.runtime.staging_dir(&r.record.id).join("diagnostic"))
                .unwrap(),
            "keep"
        );
        assert!(deps.paths.process_records.exists());
        let next = run_build(
            deps.clone(),
            &host,
            &Arc::new(RuntimeGate::default()),
            &ownership,
            options(),
            host.cancel.clone(),
            |_, _| panic!("must block before job creation"),
        )
        .await;
        assert!(matches!(next, Err(BuildStart::RecoveryRequired(_))));
        assert_old(&deps);
    }
}

#[tokio::test]
async fn llama_build_corrupt_journal_never_blindly_deletes_staging() {
    let (_tmp, deps, _, host) = setup(None, false, false, false, true);
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Failed);
    assert!(
        deps.runtime
            .staging_dir(&r.record.id)
            .join("llama-server")
            .exists()
    );
    assert_old(&deps);
}

#[tokio::test]
async fn llama_build_valid_journal_keeps_its_staging_after_interruption() {
    let (_tmp, deps, _, mut host) = setup(None, false, false, false, false);
    host.die_after_begin = true;
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Failed);
    assert!(matches!(
        llama_monitor::llama_runtime::promote::read_journal(&deps.runtime),
        llama_monitor::llama_runtime::promote::JournalState::Present(_)
    ));
    assert!(
        deps.runtime
            .staging_dir(&r.record.id)
            .join("llama-server")
            .exists()
    );
    assert_old(&deps);
}

#[tokio::test]
async fn llama_build_cancel_after_journal_does_not_abandon_installation() {
    let (_tmp, deps, _, mut host) = setup(None, false, false, false, false);
    host.cancel_in_verify = true;
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Succeeded);
    assert_eq!(
        std::fs::read_to_string(deps.runtime.live.join("llama-server")).unwrap(),
        "new-llama-server"
    );
    assert!(!deps.runtime.journal.exists());
}

#[tokio::test]
async fn llama_build_cancel_before_start_never_fetches_or_verifies() {
    let (_tmp, deps, fake, host) = setup(None, false, false, false, false);
    host.cancel.store(true, Ordering::SeqCst);
    let r = run_build(
        deps.clone(),
        &host,
        &Arc::new(RuntimeGate::default()),
        &Ownership::default(),
        options(),
        host.cancel.clone(),
        |_, _| {},
    )
    .await
    .unwrap();
    assert_eq!(r.record.status, JobStatus::Cancelled);
    assert!(host.calls.lock().unwrap().is_empty());
    assert!(!deps.paths.source.exists());
    assert_eq!(fake.unpinned.lock().unwrap().len(), 1);
    assert_old(&deps);
}

#[tokio::test]
async fn llama_build_dropped_observer_keeps_worker_ownership_until_writers_finish() {
    use llama_monitor::llama_build::cache::Owner;
    let (_tmp, deps, fake, host) = setup(None, false, false, false, false);
    fake.block_compile.store(true, Ordering::SeqCst);
    let ownership = Ownership::default();
    let gate = Arc::new(RuntimeGate::default());
    {
        let future = run_build(
            deps.clone(),
            &host,
            &gate,
            &ownership,
            options(),
            host.cancel.clone(),
            |_, _| {},
        );
        tokio::pin!(future);
        tokio::select! {
            _ = &mut future => panic!("blocked worker unexpectedly completed"),
            _ = async {
                while !fake.compile_entered.load(Ordering::SeqCst) {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            } => {}
        }
    }
    let clean_blocked = ownership.try_acquire(Owner::Clean, "clean").is_err();
    host.cancel.store(true, Ordering::SeqCst);
    fake.block_compile.store(false, Ordering::SeqCst);
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while ownership.is_held(Owner::Build) && std::time::Instant::now() < until {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        clean_blocked,
        "dropping an observer released a still-writing worker"
    );
    assert!(!ownership.is_held(Owner::Build));
    assert_old(&deps);
}
