//! Fault-injection tests for the promotion engine. Every journal write, rename,
//! verification, model stop/restart and cleanup step is a numbered boundary.
//! For each boundary the suite injects a failure **and** a process death, then
//! replays startup recovery twice and asserts the invariants.

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;
use crate::inference::kv_support::Backend;
use crate::llama_runtime::gate::{RuntimeGate, RuntimeOp};

type Plan = Box<dyn Fn(usize, &str) -> Fault + Send + Sync>;

struct FakeHost {
    plan: Plan,
    count: AtomicUsize,
    log: Mutex<Vec<String>>,
    running: bool,
    fail_verify: Mutex<Option<VerifyStage>>,
    fail_restart: bool,
    cancel: AtomicBool,
    cancel_when_staged_verify_starts: bool,
}

impl FakeHost {
    fn new(plan: Plan) -> Self {
        Self {
            plan,
            count: AtomicUsize::new(0),
            log: Mutex::new(Vec::new()),
            running: false,
            fail_verify: Mutex::new(None),
            fail_restart: false,
            cancel: AtomicBool::new(false),
            cancel_when_staged_verify_starts: false,
        }
    }

    fn quiet() -> Self {
        Self::new(Box::new(|_, _| Fault::None))
    }

    fn at_index(index: usize, fault: Fault) -> Self {
        Self::new(Box::new(
            move |n, _| if n == index { fault } else { Fault::None },
        ))
    }

    fn running(mut self) -> Self {
        self.running = true;
        self
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    fn calls(&self, prefix: &str) -> usize {
        self.log().iter().filter(|l| l.starts_with(prefix)).count()
    }

    fn position(&self, entry: &str) -> Option<usize> {
        self.log().iter().position(|l| l == entry)
    }
}

impl Faults for FakeHost {
    fn at(&self, p: &str) -> Fault {
        let n = self.count.fetch_add(1, Ordering::SeqCst);
        self.log.lock().unwrap().push(p.to_string());
        (self.plan)(n, p)
    }
}

impl PromoteHost for FakeHost {
    fn faults(&self) -> &dyn Faults {
        self
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    fn model_running(&self) -> bool {
        self.running
    }

    async fn stop_model(&self) -> Result<(), String> {
        self.log.lock().unwrap().push("call:stop".into());
        Ok(())
    }

    async fn restart_model(&self) -> Result<(), String> {
        self.log.lock().unwrap().push("call:restart".into());
        if self.fail_restart {
            Err("model failed to start".into())
        } else {
            Ok(())
        }
    }

    async fn verify(
        &self,
        _dir: &Path,
        _exe: &str,
        _intent: &VerificationIntent,
        stage: VerifyStage,
    ) -> Result<VerificationReport, String> {
        self.log
            .lock()
            .unwrap()
            .push(format!("call:verify:{stage:?}"));
        if stage == VerifyStage::Staged && self.cancel_when_staged_verify_starts {
            self.cancel.store(true, Ordering::SeqCst);
        }
        if *self.fail_verify.lock().unwrap() == Some(stage) {
            return Err(format!("{stage:?} verification failed"));
        }
        Ok(VerificationReport {
            backend: Backend::Cpu,
            build_number: None,
            build_commit: None,
            identity_fingerprint: None,
            notes: Vec::new(),
            evidence_unavailable: false,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Both,
    LiveOnly,
    PrevOnly,
    Fresh,
}

struct World {
    _root: tempfile::TempDir,
    paths: RuntimePaths,
    staged: std::path::PathBuf,
    kind: Kind,
}

type Tags = (Option<String>, Option<String>);

fn make_set(dir: &Path, tag: &str) {
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("llama-server"), tag).unwrap();
    std::fs::write(dir.join("libggml.so"), format!("lib-{tag}")).unwrap();
    std::fs::write(
        dir.join("sub").join("data"),
        format!("data-{tag}-{}", "x".repeat(tag.len())),
    )
    .unwrap();
}

fn tag_of(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("llama-server")).ok()
}

fn world(kind: Kind) -> World {
    let root = tempfile::tempdir().unwrap();
    let paths = RuntimePaths::for_live_dir(&root.path().join("bin")).unwrap();
    if matches!(kind, Kind::Both | Kind::LiveOnly) {
        make_set(&paths.live, "old-live");
    }
    if matches!(kind, Kind::Both | Kind::PrevOnly) {
        make_set(&paths.previous, "old-prev");
    }
    let staged = paths.staging_dir("job1");
    make_set(&staged, "new");
    World {
        _root: root,
        paths,
        staged,
        kind,
    }
}

impl World {
    fn tags(&self) -> Tags {
        (tag_of(&self.paths.live), tag_of(&self.paths.previous))
    }

    fn old(&self) -> Tags {
        let live = matches!(self.kind, Kind::Both | Kind::LiveOnly).then(|| "old-live".to_string());
        let prev = matches!(self.kind, Kind::Both | Kind::PrevOnly).then(|| "old-prev".to_string());
        (live, prev)
    }

    /// State after a committed install: the new set is live; the old live set
    /// is the one previous set; a pre-existing previous is kept only when there
    /// was no live set to retain.
    fn new_install(&self) -> Tags {
        let had_live = matches!(self.kind, Kind::Both | Kind::LiveOnly);
        let previous = if had_live {
            Some("old-live".to_string())
        } else {
            self.old().1
        };
        (Some("new".into()), previous)
    }

    fn new_rollback(&self) -> Tags {
        let (live, prev) = self.old();
        (prev, live)
    }

    fn new_uninstall(&self) -> Tags {
        (None, None)
    }

    fn assert_no_leftovers(&self) {
        assert!(!self.paths.journal.exists(), "journal must be gone");
        for entry in std::fs::read_dir(&self.paths.parent).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".bin.txn-") && !name.starts_with(".bin.staging-"),
                "leftover {name}"
            );
        }
    }
}

fn gate() -> (std::sync::Arc<RuntimeGate>, RuntimeGuard) {
    let gate = std::sync::Arc::new(RuntimeGate::default());
    let guard = gate.try_begin(RuntimeOp::Promote).unwrap();
    (gate, guard)
}

fn request(world: &World) -> PromoteRequest {
    PromoteRequest {
        staged_dir: world.staged.clone(),
        label: "test".into(),
        exe_name: "llama-server".into(),
        manifest: None,
        verification: VerificationIntent::prebuilt("cpu"),
    }
}

async fn do_install(w: &World, host: &FakeHost) -> Result<PromoteResult, PromoteError> {
    let (_gate, mut guard) = gate();
    promote(host, &mut guard, &w.paths, request(w)).await
}

/// Rollback and uninstall never use an install candidate.
fn drop_candidate(w: &World) {
    let _ = std::fs::remove_dir_all(&w.staged);
}

async fn do_rollback(w: &World, host: &FakeHost) -> Result<PromoteResult, PromoteError> {
    let (_gate, mut guard) = gate();
    rollback(host, &mut guard, &w.paths, "llama-server").await
}

async fn do_uninstall(w: &World, host: &FakeHost) -> Result<PromoteResult, PromoteError> {
    let (_gate, mut guard) = gate();
    uninstall(host, &mut guard, &w.paths, "llama-server").await
}

/// Replay startup recovery twice and assert it is idempotent.
fn replay_twice(w: &World) -> RecoveryReport {
    let first = recover(&w.paths, &NoFaults).expect("recovery must not be interrupted");
    let second = recover(&w.paths, &NoFaults).unwrap();
    assert_eq!(
        second,
        RecoveryReport::Nothing,
        "second replay must find nothing to do"
    );
    // Startup order: journal recovery first, only then orphan cleanup.
    if !matches!(first, RecoveryReport::Blocked { .. }) {
        sweep_orphans(&w.paths).unwrap();
    }
    first
}

#[derive(Clone, Copy, Debug)]
enum OpKind {
    Install,
    Rollback,
    Uninstall,
}

async fn run(op: OpKind, w: &World, host: &FakeHost) -> Result<PromoteResult, PromoteError> {
    if !matches!(op, OpKind::Install) {
        drop_candidate(w);
    }
    match op {
        OpKind::Install => do_install(w, host).await,
        OpKind::Rollback => do_rollback(w, host).await,
        OpKind::Uninstall => do_uninstall(w, host).await,
    }
}

fn expected_new(op: OpKind, w: &World) -> Tags {
    match op {
        OpKind::Install => w.new_install(),
        OpKind::Rollback => w.new_rollback(),
        OpKind::Uninstall => w.new_uninstall(),
    }
}

/// Run `op` once with no faults to learn how many boundaries it crosses, then
/// fail and kill at every one of them.
async fn sweep(op: OpKind, kind: Kind, running: bool) {
    let dry_world = world(kind);
    let dry = if running {
        FakeHost::quiet().running()
    } else {
        FakeHost::quiet()
    };
    let result = run(op, &dry_world, &dry).await.expect("fault-free run");
    assert!(
        matches!(result.outcome, PromoteOutcome::Promoted { .. }),
        "{op:?}/{kind:?}: {result:?}"
    );
    assert_eq!(dry_world.tags(), expected_new(op, &dry_world));
    let boundaries = dry.log().iter().filter(|l| !l.starts_with("call:")).count();
    assert!(
        boundaries >= 12,
        "{op:?}: only {boundaries} boundaries were instrumented"
    );

    for fault in [Fault::Fail, Fault::Die] {
        for index in 0..boundaries {
            let w = world(kind);
            let host = if running {
                FakeHost::at_index(index, fault).running()
            } else {
                FakeHost::at_index(index, fault)
            };
            let result = run(op, &w, &host).await;
            let context = format!(
                "{op:?}/{kind:?}/running={running}/{fault:?}@{index} ({}) -> {result:?}",
                host.log()
                    .iter()
                    .filter(|l| !l.starts_with("call:"))
                    .nth(index)
                    .cloned()
                    .unwrap_or_default()
            );
            if fault == Fault::Die {
                assert!(matches!(result, Err(PromoteError::Died(_))), "{context}");
            }
            replay_twice(&w);

            // The only legal end states are the old pair and the new pair, and
            // the new pair only once the commit write became durable.
            let committed = host.position("journal.committed.after").is_some();
            let tags = w.tags();
            let expected = if committed {
                expected_new(op, &w)
            } else {
                w.old()
            };
            assert_eq!(tags, expected, "{context}");
            w.assert_no_leftovers();
            let last = last_transaction(&w.paths);
            // A fault before the journal is durable leaves no record at all.
            if host.position("journal.begin.after").is_some() {
                assert!(
                    last.is_some(),
                    "{context}: terminal outcome must be recorded"
                );
            }
            if let Ok(result) = &result
                && fault == Fault::Fail
            {
                match result.outcome {
                    PromoteOutcome::RolledBack { .. } => assert!(!committed, "{context}"),
                    _ => assert!(committed, "{context}"),
                }
            }
            // Never more than one restart of the remembered model.
            assert!(host.calls("call:restart") <= 1, "{context}");
        }
    }
}

#[tokio::test]
async fn install_survives_failure_and_death_at_every_boundary() {
    for kind in [Kind::Both, Kind::LiveOnly, Kind::PrevOnly, Kind::Fresh] {
        sweep(OpKind::Install, kind, false).await;
    }
}

#[tokio::test]
async fn install_with_a_running_model_survives_every_boundary() {
    for kind in [Kind::Both, Kind::Fresh] {
        sweep(OpKind::Install, kind, true).await;
    }
}

#[tokio::test]
async fn rollback_survives_failure_and_death_at_every_boundary() {
    for kind in [Kind::Both, Kind::PrevOnly] {
        sweep(OpKind::Rollback, kind, false).await;
    }
    sweep(OpKind::Rollback, Kind::Both, true).await;
}

#[tokio::test]
async fn uninstall_survives_failure_and_death_at_every_boundary() {
    for kind in [Kind::Both, Kind::LiveOnly] {
        sweep(OpKind::Uninstall, kind, false).await;
    }
}

// -- journaling guarantees ---------------------------------------------------

#[tokio::test]
async fn journal_is_durable_before_any_mutation_and_failure_to_write_it_mutates_nothing() {
    for point_name in ["journal.begin.before", "journal.begin.tmp"] {
        let w = world(Kind::Both);
        let host = FakeHost::new(Box::new(move |_, p| {
            if p == point_name {
                Fault::Fail
            } else {
                Fault::None
            }
        }));
        let err = do_install(&w, &host).await.unwrap_err();
        assert!(
            matches!(err, PromoteError::Journal(_)),
            "{point_name}: {err:?}"
        );
        // Disk-full on the journal: nothing moved, nothing deleted.
        assert_eq!(w.tags(), w.old());
        assert!(w.staged.is_dir(), "the candidate is untouched");
        assert!(!w.paths.journal.exists());
        assert!(
            !w.paths.tmp_journal().exists(),
            "a torn temp journal is cleaned up"
        );
        assert_eq!(
            host.calls("call:"),
            0,
            "nothing ran before the journal existed"
        );
    }
}

#[tokio::test]
async fn nothing_is_mutated_before_the_journal_begin_write_completes() {
    let w = world(Kind::Both);
    let host = FakeHost::quiet();
    do_install(&w, &host).await.unwrap();
    let log = host.log();
    let begin = log.iter().position(|l| l == "journal.begin.after").unwrap();
    for (i, entry) in log.iter().enumerate() {
        if entry.starts_with("rename.") || entry.starts_with("call:") {
            assert!(i > begin, "{entry} ran before the journal was durable");
        }
    }
}

#[tokio::test]
async fn restart_intent_is_journaled_before_the_model_is_stopped() {
    let w = world(Kind::Both);
    let host = FakeHost::quiet().running();
    do_install(&w, &host).await.unwrap();
    let intent = host
        .position("journal.restart-intent.after")
        .expect("intent journaled");
    let stop = host.position("call:stop").expect("model stopped");
    assert!(intent < stop, "intent must precede the stop");
    assert!(
        host.position("call:stop").unwrap() < host.position("call:verify:Staged").unwrap(),
        "a contending model is stopped before the staged probe"
    );
}

#[tokio::test]
async fn death_after_the_model_was_stopped_leaves_a_restart_obligation() {
    let w = world(Kind::Both);
    let host = FakeHost::new(Box::new(|_, p| {
        if p == "journal.model-stopped.before" {
            Fault::Die
        } else {
            Fault::None
        }
    }))
    .running();
    let err = do_install(&w, &host).await.unwrap_err();
    assert!(matches!(err, PromoteError::Died(_)));
    // The model was stopped; the journal on disk still carries the *intent*.
    let JournalState::Present(j) = read_journal(&w.paths) else {
        panic!("journal expected")
    };
    assert!(j.restart.required);
    let report = replay_twice(&w);
    // `stopped` was never persisted, so a crash cannot claim a restart that
    // was never promised, and the old runtime is intact either way.
    assert!(matches!(report, RecoveryReport::Restored { .. }));
    assert_eq!(w.tags(), w.old());
}

#[tokio::test]
async fn death_after_commit_is_finished_by_recovery_and_reports_a_pending_restart() {
    let w = world(Kind::Both);
    let host = FakeHost::new(Box::new(|_, p| {
        if p == "model.restart.before" {
            Fault::Die
        } else {
            Fault::None
        }
    }))
    .running();
    assert!(matches!(
        do_install(&w, &host).await,
        Err(PromoteError::Died(_))
    ));
    let report = recover(&w.paths, &NoFaults).unwrap();
    assert!(matches!(report, RecoveryReport::Completed { .. }));
    assert!(
        report.restart_pending(),
        "the stopped model must be reported"
    );
    assert_eq!(w.tags(), w.new_install());
    w.assert_no_leftovers();
}

// -- outcomes ----------------------------------------------------------------

#[tokio::test]
async fn binary_failure_restores_transactionally_and_restarts_the_old_model() {
    for stage in [VerifyStage::Staged, VerifyStage::Live] {
        let w = world(Kind::Both);
        let host = FakeHost::quiet().running();
        *host.fail_verify.lock().unwrap() = Some(stage);
        let result = do_install(&w, &host).await.unwrap();
        assert!(
            matches!(result.outcome, PromoteOutcome::RolledBack { .. }),
            "{stage:?}: {result:?}"
        );
        assert_eq!(
            w.tags(),
            w.old(),
            "last usable live and old previous survive"
        );
        assert_eq!(
            host.calls("call:restart"),
            1,
            "the old model is restarted once"
        );
        w.assert_no_leftovers();
        assert!(matches!(
            last_transaction(&w.paths).unwrap().outcome,
            Some(Outcome::RolledBack { .. })
        ));
    }
}

#[tokio::test]
async fn model_restart_failure_keeps_the_installed_binaries_and_offers_rollback() {
    let w = world(Kind::Both);
    let mut host = FakeHost::quiet().running();
    host.fail_restart = true;
    let result = do_install(&w, &host).await.unwrap();
    assert_eq!(
        result.outcome,
        PromoteOutcome::InstalledRestartFailed {
            restart_error: "model failed to start".into()
        }
    );
    // Not an automatic binary rollback.
    assert_eq!(w.tags(), w.new_install());
    assert_eq!(host.calls("call:restart"), 1);
    // Manual rollback works from here, and rolling back twice is the identity.
    let host2 = FakeHost::quiet();
    do_rollback(&w, &host2).await.unwrap();
    assert_eq!(w.tags(), (Some("old-live".into()), Some("new".into())));
}

#[tokio::test]
async fn rollback_twice_returns_to_the_original() {
    let w = world(Kind::Both);
    drop_candidate(&w);
    do_rollback(&w, &FakeHost::quiet()).await.unwrap();
    assert_eq!(w.tags(), w.new_rollback());
    do_rollback(&w, &FakeHost::quiet()).await.unwrap();
    assert_eq!(w.tags(), w.old());
    w.assert_no_leftovers();
}

#[tokio::test]
async fn rollback_without_a_previous_set_is_refused_without_a_journal() {
    let w = world(Kind::LiveOnly);
    let err = do_rollback(&w, &FakeHost::quiet()).await.unwrap_err();
    assert!(matches!(err, PromoteError::Prepare(_)));
    assert!(!w.paths.journal.exists());
}

#[tokio::test]
async fn a_failed_rollback_rename_restores_both_sets() {
    let w = world(Kind::Both);
    drop_candidate(&w);
    // Rollback renames: 0 = live->old-live, 1 = previous->live. Fail the second.
    let host = FakeHost::new(Box::new(|_, p| {
        if p == "rename.1.before" {
            Fault::Fail
        } else {
            Fault::None
        }
    }));
    let result = do_rollback(&w, &host).await.unwrap();
    assert!(matches!(result.outcome, PromoteOutcome::RolledBack { .. }));
    assert_eq!(w.tags(), w.old());
    w.assert_no_leftovers();
}

#[tokio::test]
async fn a_compensating_rename_that_dies_is_finished_by_recovery() {
    // verify.live fails -> undo starts -> the process dies at each undo boundary.
    let dry_w = world(Kind::Both);
    let dry = FakeHost::quiet();
    *dry.fail_verify.lock().unwrap() = Some(VerifyStage::Live);
    do_install(&dry_w, &dry).await.unwrap();
    let undo_points: Vec<usize> = dry
        .log()
        .iter()
        .filter(|l| !l.starts_with("call:"))
        .enumerate()
        .filter(|(_, l)| l.starts_with("undo.") || l.starts_with("finalize."))
        .map(|(i, _)| i)
        .collect();
    assert!(
        undo_points.len() >= 6,
        "undo must be instrumented: {undo_points:?}"
    );
    for target in undo_points {
        let w = world(Kind::Both);
        let host = FakeHost::new(Box::new(
            move |n, _| if n == target { Fault::Die } else { Fault::None },
        ));
        *host.fail_verify.lock().unwrap() = Some(VerifyStage::Live);
        assert!(matches!(
            do_install(&w, &host).await,
            Err(PromoteError::Died(_))
        ));
        replay_twice(&w);
        assert_eq!(
            w.tags(),
            w.old(),
            "died during compensation at boundary {target}"
        );
        w.assert_no_leftovers();
    }
}

#[tokio::test]
async fn cancellation_is_honored_only_before_the_journal_exists() {
    let w = world(Kind::Both);
    let host = FakeHost::quiet();
    host.cancel.store(true, Ordering::SeqCst);
    assert_eq!(
        do_install(&w, &host).await.unwrap_err(),
        PromoteError::Cancelled
    );
    assert_eq!(w.tags(), w.old());
    assert!(!w.paths.journal.exists());

    // A cancel request that arrives once the transaction began is ignored.
    let w = world(Kind::Both);
    let mut host = FakeHost::quiet();
    host.cancel_when_staged_verify_starts = true;
    let result = do_install(&w, &host).await.unwrap();
    assert!(matches!(result.outcome, PromoteOutcome::Promoted { .. }));
    assert_eq!(w.tags(), w.new_install());
}

#[tokio::test]
async fn the_swap_window_is_open_for_the_transaction_and_closed_after() {
    let w = world(Kind::Both);
    let (_gate, mut guard) = gate();
    assert!(
        !guard.in_swap_window(),
        "not open before the journal exists"
    );
    promote(&FakeHost::quiet(), &mut guard, &w.paths, request(&w))
        .await
        .unwrap();
    assert!(
        guard.in_swap_window(),
        "opened with the journal; held until the guard drops"
    );
    // A refused promotion (bad staged dir) never opens it.
    let (_gate2, mut guard2) = gate();
    let mut bad = request(&w);
    bad.staged_dir = w.paths.parent.join("nope");
    assert!(
        promote(&FakeHost::quiet(), &mut guard2, &w.paths, bad)
            .await
            .is_err()
    );
    assert!(!guard2.in_swap_window());
}

// -- corrupt / ambiguous journals --------------------------------------------

#[tokio::test]
async fn a_corrupt_journal_blocks_everything_and_deletes_nothing() {
    let w = world(Kind::Both);
    std::fs::write(&w.paths.journal, b"{not json").unwrap();
    for _ in 0..2 {
        assert!(matches!(
            recover(&w.paths, &NoFaults).unwrap(),
            RecoveryReport::Blocked { .. }
        ));
    }
    let err = do_install(&w, &FakeHost::quiet()).await.unwrap_err();
    assert!(matches!(err, PromoteError::JournalBlocked(_)), "{err:?}");
    assert!(matches!(
        do_rollback(&w, &FakeHost::quiet()).await,
        Err(PromoteError::JournalBlocked(_))
    ));
    assert!(matches!(
        do_uninstall(&w, &FakeHost::quiet()).await,
        Err(PromoteError::JournalBlocked(_))
    ));
    // Sweeping refuses to run beside a journal, so the staged set survives.
    assert!(sweep_orphans(&w.paths).unwrap().is_empty());
    assert!(w.staged.is_dir());
    assert_eq!(w.tags(), w.old());
}

#[tokio::test]
async fn an_ambiguous_state_is_retained_not_guessed() {
    let w = world(Kind::Both);
    // Die right after the live set was parked (rename.1 is live -> old-live).
    let host = FakeHost::new(Box::new(|_, p| {
        if p == "rename.2.before" {
            Fault::Die
        } else {
            Fault::None
        }
    }));
    assert!(matches!(
        do_install(&w, &host).await,
        Err(PromoteError::Died(_))
    ));
    let JournalState::Present(j) = read_journal(&w.paths) else {
        panic!()
    };
    // Somebody deletes the parked live set: recovery cannot know what to restore.
    std::fs::remove_dir_all(w.paths.role(Role::OldLive, &j)).unwrap();
    for _ in 0..2 {
        let report = recover(&w.paths, &NoFaults).unwrap();
        assert!(
            matches!(report, RecoveryReport::Blocked { .. }),
            "{report:?}"
        );
    }
    assert!(
        w.paths.journal.exists(),
        "the journal is retained for repair"
    );
    assert!(
        w.staged.is_dir(),
        "a journal-referenced candidate is never swept"
    );
    assert!(sweep_orphans(&w.paths).unwrap().is_empty());
    assert!(matches!(
        do_install(&w, &FakeHost::quiet()).await,
        Err(PromoteError::JournalBlocked(_))
    ));
}

#[tokio::test]
async fn a_set_edited_while_the_transaction_was_in_flight_is_detected() {
    let w = world(Kind::Both);
    let host = FakeHost::new(Box::new(|_, p| {
        if p == "verify.live" {
            Fault::Die
        } else {
            Fault::None
        }
    }));
    assert!(matches!(
        do_install(&w, &host).await,
        Err(PromoteError::Died(_))
    ));
    let JournalState::Present(j) = read_journal(&w.paths) else {
        panic!()
    };
    // Tamper with the parked old live set so its fingerprint no longer matches.
    std::fs::write(w.paths.role(Role::OldLive, &j).join("extra"), b"surprise").unwrap();
    assert!(matches!(
        recover(&w.paths, &NoFaults).unwrap(),
        RecoveryReport::Blocked { .. }
    ));
}

#[tokio::test]
async fn a_torn_journal_write_is_ignored_and_the_previous_journal_wins() {
    let w = world(Kind::Both);
    let host = FakeHost::new(Box::new(|_, p| {
        if p == "journal.swapping.tmp" {
            Fault::Die
        } else {
            Fault::None
        }
    }));
    assert!(matches!(
        do_install(&w, &host).await,
        Err(PromoteError::Died(_))
    ));
    assert!(
        w.paths.tmp_journal().exists(),
        "the torn temp file is present"
    );
    let JournalState::Present(j) = read_journal(&w.paths) else {
        panic!()
    };
    assert_eq!(
        j.phase,
        Phase::Begun,
        "the durable journal is the last completed write"
    );
    replay_twice(&w);
    assert_eq!(w.tags(), w.old());
    w.assert_no_leftovers();
}

// -- startup cleanup ---------------------------------------------------------

#[test]
fn orphans_are_swept_only_when_no_journal_references_them() {
    let w = world(Kind::Both);
    let orphan_staging = w.paths.staging_dir("dead");
    let orphan_txn = w.paths.parent.join(".bin.txn-dead-live");
    let legacy_staging = w.paths.parent.join(".bin.llama-update-b1-1-2");
    let legacy_failed = w.paths.parent.join("bin-failed-b1-1-2");
    for dir in [
        &orphan_staging,
        &orphan_txn,
        &legacy_staging,
        &legacy_failed,
    ] {
        make_set(dir, "junk");
    }
    let removed = sweep_orphans(&w.paths).unwrap();
    assert_eq!(
        removed.len(),
        5,
        "the unreferenced job1 staging dir goes too: {removed:?}"
    );
    assert_eq!(w.tags(), w.old());
}

#[test]
fn the_newest_legacy_backup_is_adopted_into_an_empty_previous_slot_and_others_survive() {
    let w = world(Kind::LiveOnly);
    let older = w.paths.parent.join("bin-previous-b1-1-100");
    let newer = w.paths.parent.join("bin-previous-b2-1-200");
    make_set(&older, "legacy-old");
    std::thread::sleep(std::time::Duration::from_millis(20));
    make_set(&newer, "legacy-new");
    let removed = sweep_orphans(&w.paths).unwrap();
    assert!(
        removed
            .iter()
            .all(|p| !p.to_string_lossy().contains("bin-previous")),
        "no legacy backup is deleted: {removed:?}"
    );
    assert_eq!(
        tag_of(&w.paths.previous).as_deref(),
        Some("legacy-new"),
        "newest is adopted by rename"
    );
    assert!(!newer.exists());
    assert_eq!(tag_of(&older).as_deref(), Some("legacy-old"));

    // A managed previous slot is never displaced, and no legacy backup is deleted.
    let w = world(Kind::Both);
    let legacy = w.paths.parent.join("bin-previous-b1-1-100");
    make_set(&legacy, "legacy");
    sweep_orphans(&w.paths).unwrap();
    assert_eq!(tag_of(&w.paths.previous).as_deref(), Some("old-prev"));
    assert_eq!(tag_of(&legacy).as_deref(), Some("legacy"));
}

#[tokio::test]
async fn a_committed_install_completes_retention_and_leaves_one_live_and_one_previous() {
    let w = world(Kind::Both);
    do_install(&w, &FakeHost::quiet()).await.unwrap();
    let names: Vec<String> = std::fs::read_dir(&w.paths.parent)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.contains("runtime-transaction"))
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["bin", "bin-previous"],
        "stable state: live + at most one previous"
    );
}

#[test]
fn directory_fingerprints_follow_a_rename_and_change_with_content() {
    let root = tempfile::tempdir().unwrap();
    let a = root.path().join("a");
    make_set(&a, "one");
    let fp = dir_fingerprint(&a).unwrap();
    let b = root.path().join("b");
    std::fs::rename(&a, &b).unwrap();
    assert_eq!(
        dir_fingerprint(&b).unwrap(),
        fp,
        "a renamed set keeps its identity"
    );
    std::fs::write(b.join("extra"), b"x").unwrap();
    assert_ne!(dir_fingerprint(&b).unwrap(), fp);
}

#[tokio::test]
async fn a_staged_directory_that_is_not_a_sibling_is_refused_before_anything_runs() {
    let w = world(Kind::Both);
    let elsewhere = tempfile::tempdir().unwrap();
    make_set(&elsewhere.path().join(".bin.staging-x"), "evil");
    let (_gate, mut guard) = gate();
    let mut req = request(&w);
    req.staged_dir = elsewhere.path().join(".bin.staging-x");
    assert!(matches!(
        promote(&FakeHost::quiet(), &mut guard, &w.paths, req).await,
        Err(PromoteError::Prepare(_))
    ));
    let mut req = request(&w);
    req.staged_dir = w.paths.parent.join("bin-previous");
    assert!(matches!(
        promote(&FakeHost::quiet(), &mut guard, &w.paths, req).await,
        Err(PromoteError::Prepare(_))
    ));
    assert_eq!(w.tags(), w.old());
}

#[tokio::test]
async fn provenance_manifest_is_written_into_the_installed_set_but_never_decides_anything() {
    use crate::llama_runtime::manifest::{
        LlamaCppRef, ManifestCheck, ManifestSource, RuntimeManifest, check_manifest,
    };
    let w = world(Kind::Both);
    let manifest = RuntimeManifest::for_directory(
        &w.staged,
        ManifestSource::UpstreamPrebuilt,
        "test prebuilt",
        LlamaCppRef {
            reference: Some("b1".into()),
            commit: None,
            build_number: Some(1),
        },
        Some("cpu".into()),
    )
    .unwrap();
    let mut req = request(&w);
    req.manifest = Some(manifest);
    let (_gate, mut guard) = gate();
    promote(&FakeHost::quiet(), &mut guard, &w.paths, req)
        .await
        .unwrap();
    assert_eq!(check_manifest(&w.paths.live), ManifestCheck::Matches);
}

#[cfg(windows)]
#[tokio::test]
async fn a_locked_file_in_the_live_set_fails_the_rename_and_recovery_restores() {
    use std::os::windows::fs::OpenOptionsExt;
    let w = world(Kind::Both);
    // share_mode(0): no sharing at all, so renaming the directory is refused
    // exactly as it is while a model or a scanner has a DLL open.
    let _locked = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(w.paths.live.join("libggml.so"))
        .unwrap();
    let result = do_install(&w, &FakeHost::quiet()).await.unwrap();
    assert!(
        matches!(result.outcome, PromoteOutcome::RolledBack { .. }),
        "{result:?}"
    );
    drop(_locked);
    assert_eq!(w.tags(), w.old());
    w.assert_no_leftovers();
}

#[tokio::test]
async fn install_probes_the_candidate_exactly_once_before_and_once_after_the_swap() {
    let w = world(Kind::Both);
    let host = FakeHost::quiet();
    do_install(&w, &host).await.unwrap();
    assert_eq!(host.calls("call:verify:Staged"), 1, "{:?}", host.log());
    assert_eq!(host.calls("call:verify:Live"), 1, "{:?}", host.log());
    assert_eq!(host.calls("call:verify"), 2, "no third probe of any kind");
    assert!(
        host.position("call:verify:Staged").unwrap()
            < host.position("rename.0.before").unwrap_or(usize::MAX),
        "the staged probe precedes the first rename"
    );

    let w = world(Kind::Both);
    let host = FakeHost::quiet();
    drop_candidate(&w);
    do_rollback(&w, &host).await.unwrap();
    assert_eq!(
        host.calls("call:verify"),
        2,
        "rollback verifies before and after too"
    );

    let w = world(Kind::Both);
    let host = FakeHost::quiet();
    drop_candidate(&w);
    do_uninstall(&w, &host).await.unwrap();
    assert_eq!(host.calls("call:verify"), 0, "uninstall executes nothing");
}

#[tokio::test]
async fn the_gate_is_held_for_the_whole_model_lifecycle_inside_a_transaction() {
    // Lock order: runtime gate first, then the model lifecycle (stop/restart
    // take `server_lifecycle` inside the host). The engine must never be driven
    // without the gate, and a second mutation must be rejected throughout.
    let w = world(Kind::Both);
    let host = FakeHost::quiet().running();
    let gate = std::sync::Arc::new(RuntimeGate::default());
    let mut guard = gate.try_begin(RuntimeOp::Update).unwrap();
    let result = promote(&host, &mut guard, &w.paths, request(&w))
        .await
        .unwrap();
    assert!(matches!(
        result.outcome,
        PromoteOutcome::Promoted { restarted: true }
    ));
    assert_eq!(
        gate.current_owner(),
        Some(RuntimeOp::Update),
        "still held after the swap"
    );
    let busy = gate
        .try_begin(RuntimeOp::Rollback)
        .err()
        .expect("second mutation rejected");
    assert_eq!(busy.owner, RuntimeOp::Update);
    drop(guard);
    assert!(gate.try_begin(RuntimeOp::Rollback).is_ok());
}

// -- fix 5: the journal's staged path is untrusted input ----------------------

fn install_journal_with_staged(staged: Option<&str>) -> Journal {
    Journal {
        schema: SCHEMA,
        id: "tamper".into(),
        op: Op::Install,
        label: "tamper".into(),
        started_unix: 0,
        steps: vec![Step {
            from: Role::Staged,
            to: Role::Live,
        }],
        had_live: false,
        had_previous: false,
        live_fp: None,
        previous_fp: None,
        staged_fp: None,
        staged: staged.map(str::to_string),
        phase: Phase::Begun,
        restart: RestartRecord::default(),
        outcome: None,
    }
}

#[test]
fn a_journal_whose_staged_path_escapes_the_staging_namespace_is_corrupt_and_deletes_nothing() {
    let root = tempfile::tempdir().unwrap();
    let paths = RuntimePaths::for_live_dir(&root.path().join("app").join("bin")).unwrap();
    make_set(&paths.live, "old-live");
    let outside = root.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("keep"), "x").unwrap();
    let nested = paths.parent.join("nest").join("deep");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("keep"), "x").unwrap();
    let absolute = outside.to_string_lossy().into_owned();

    let cases: Vec<Option<String>> = vec![
        None,
        Some(String::new()),
        Some("..".into()),
        Some(".".into()),
        Some(absolute),
        Some("nest/deep".into()),
        Some("nest\\deep".into()),
        Some(".bin.staging-x/../..".into()),
        Some("bin".into()),
        Some(".bin.staging-".into()),
    ];
    for staged in cases {
        let journal = install_journal_with_staged(staged.as_deref());
        std::fs::write(&paths.journal, serde_json::to_vec(&journal).unwrap()).unwrap();
        assert!(
            matches!(read_journal(&paths), JournalState::Corrupt(_)),
            "{staged:?} must be Corrupt"
        );
        assert!(
            matches!(
                recover(&paths, &NoFaults).unwrap(),
                RecoveryReport::Blocked { .. }
            ),
            "{staged:?} must block recovery"
        );
        assert!(paths.journal.exists(), "{staged:?}: journal must stay");
        assert_eq!(tag_of(&paths.live).as_deref(), Some("old-live"));
        assert!(outside.join("keep").exists(), "{staged:?}: outside deleted");
        assert!(nested.join("keep").exists(), "{staged:?}: nested deleted");
        assert!(paths.parent.is_dir() && root.path().is_dir());
    }
}

#[test]
fn a_journal_with_a_well_formed_staged_name_is_still_accepted() {
    let root = tempfile::tempdir().unwrap();
    let paths = RuntimePaths::for_live_dir(&root.path().join("bin")).unwrap();
    let journal = install_journal_with_staged(Some(".bin.staging-job1"));
    std::fs::write(&paths.journal, serde_json::to_vec(&journal).unwrap()).unwrap();
    assert!(matches!(read_journal(&paths), JournalState::Present(_)));
}

// -- fix 2: a restart failure on the reject path is surfaced -------------------

#[tokio::test]
async fn a_rejected_install_whose_model_restart_fails_reports_the_restart_failure() {
    for stage in [VerifyStage::Staged, VerifyStage::Live] {
        let w = world(Kind::Both);
        let mut host = FakeHost::quiet().running();
        host.fail_restart = true;
        *host.fail_verify.lock().unwrap() = Some(stage);
        let result = do_install(&w, &host).await.unwrap();
        match result.outcome {
            PromoteOutcome::RolledBack {
                restart_failed: Some(e),
                ..
            } => assert_eq!(e, "model failed to start", "{stage:?}"),
            other => panic!("{stage:?}: restart failure was dropped: {other:?}"),
        }
        assert_eq!(w.tags(), w.old(), "the old runtime is still restored");
        w.assert_no_leftovers();
    }
}

#[tokio::test]
async fn a_rejected_install_whose_model_restarts_fine_reports_no_restart_failure() {
    let w = world(Kind::Both);
    let host = FakeHost::quiet().running();
    *host.fail_verify.lock().unwrap() = Some(VerifyStage::Staged);
    let result = do_install(&w, &host).await.unwrap();
    assert!(matches!(
        result.outcome,
        PromoteOutcome::RolledBack {
            restart_failed: None,
            ..
        }
    ));
}

#[test]
fn rename_retry_targets_only_windows_handle_errors() {
    assert!(is_windows_handle_error(Some(5)));
    assert!(is_windows_handle_error(Some(32)));
    assert!(is_windows_handle_error(Some(33)));
    assert!(!is_windows_handle_error(Some(13)));
    assert!(!is_windows_handle_error(None));
    // Off Windows nothing is retried: these codes never arrive as handle
    // contention there, and the delay stays `None`.
    if !cfg!(windows) {
        assert!(!rename_blocked_by_open_handle(Some(32)));
        assert_eq!(rename_retry_delay_ms(0), None);
    }
}

#[test]
fn rename_backoff_is_bounded_and_terminates() {
    if !cfg!(windows) {
        return;
    }
    assert_eq!(rename_retry_delay_ms(0), Some(250));
    assert_eq!(rename_retry_delay_ms(3), Some(1000));
    assert_eq!(rename_retry_delay_ms(4), None);
}
