//! Journaled, crash-recoverable runtime promotion (plan §4.3).
//!
//! One engine serves prebuilt installs, source promotions, rollback and
//! uninstall. It is a transaction, not a sequence of best-effort renames:
//!
//! * The complete **step list** (every rename, in order) is durably written to
//!   `runtime-transaction.json` before the first mutation, with a metadata
//!   fingerprint of every set involved.
//! * Whether a rename happened is **derived from the filesystem**, never from a
//!   flag that could be stale: step `from -> to` has happened exactly when
//!   `from` is gone. Recovery therefore reconciles real path roles and
//!   fingerprints, and replaying it any number of times is idempotent.
//! * Before the `Committed` phase is durable, recovery restores the last usable
//!   live set **and** the old previous set. After it, recovery finishes
//!   retention. Superseded sets are deleted only after commit.
//! * Once the journal exists the transaction is not cancellable, including its
//!   compensating recovery.
//!
//! Every journal write, rename, verification, model stop/restart and cleanup
//! step passes through a [`Faults`] hook so the test suite can fail or kill the
//! process at *each* boundary and assert recovery. Production uses
//! [`NoFaults`].

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llama_runtime::gate::RuntimeGuard;
use crate::llama_runtime::manifest::RuntimeManifest;
use crate::llama_runtime::verify::{VerificationIntent, VerificationReport, prepare_runtime_dir};

pub const JOURNAL_FILE: &str = "runtime-transaction.json";
const JOURNAL_TMP: &str = ".runtime-transaction.json.tmp";
const JOURNAL_LAST: &str = "runtime-transaction.last.json";
const SCHEMA: u32 = 1;

// ---------------------------------------------------------------------------
// Fault injection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    None,
    /// The operation fails; the engine runs its own recovery.
    Fail,
    /// The process dies here: nothing else runs, state is left exactly as is.
    Die,
}

pub trait Faults: Send + Sync {
    fn at(&self, point: &str) -> Fault;
}

pub struct NoFaults;

impl Faults for NoFaults {
    fn at(&self, _point: &str) -> Fault {
        Fault::None
    }
}

/// Why a step stopped. `Died` is only ever produced by a test hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interrupt {
    Failed(String),
    Died(String),
}

impl From<std::io::Error> for Interrupt {
    fn from(e: std::io::Error) -> Self {
        Interrupt::Failed(e.to_string())
    }
}

fn point(faults: &dyn Faults, name: &str) -> Result<(), Interrupt> {
    match faults.at(name) {
        Fault::None => Ok(()),
        Fault::Fail => Err(Interrupt::Failed(format!("injected failure at {name}"))),
        Fault::Die => Err(Interrupt::Died(name.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub parent: PathBuf,
    pub live: PathBuf,
    pub previous: PathBuf,
    pub journal: PathBuf,
    name: String,
}

impl RuntimePaths {
    /// Paths for the managed live directory (`<root>/bin`). The previous set,
    /// journal and every transaction directory are siblings on the same volume.
    pub fn for_live_dir(live: &Path) -> std::io::Result<Self> {
        let parent = live
            .parent()
            .ok_or_else(|| std::io::Error::other("the live runtime directory has no parent"))?
            .to_path_buf();
        let name = live
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| std::io::Error::other("the live runtime directory has no name"))?
            .to_string();
        Ok(Self {
            previous: parent.join(format!("{name}-previous")),
            journal: parent.join(JOURNAL_FILE),
            live: live.to_path_buf(),
            parent,
            name,
        })
    }

    pub fn staging_dir(&self, id: &str) -> PathBuf {
        self.parent.join(format!(".{}.staging-{id}", self.name))
    }

    fn staging_prefix(&self) -> String {
        format!(".{}.staging-", self.name)
    }

    fn txn_prefix(&self) -> String {
        format!(".{}.txn-", self.name)
    }

    fn tmp_journal(&self) -> PathBuf {
        self.parent.join(JOURNAL_TMP)
    }

    fn last_journal(&self) -> PathBuf {
        self.parent.join(JOURNAL_LAST)
    }

    fn role(&self, role: Role, j: &Journal) -> PathBuf {
        match role {
            Role::Live => self.live.clone(),
            Role::Previous => self.previous.clone(),
            Role::OldLive => self
                .parent
                .join(format!("{}{}-live", self.txn_prefix(), j.id)),
            Role::OldPrevious => self
                .parent
                .join(format!("{}{}-prev", self.txn_prefix(), j.id)),
            Role::Staged => self.parent.join(j.staged.as_deref().unwrap_or("")),
        }
    }
}

// ---------------------------------------------------------------------------
// Journal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Install,
    Rollback,
    Uninstall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Live,
    Previous,
    OldLive,
    OldPrevious,
    Staged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub from: Role,
    pub to: Role,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Journal exists; verification may be running; no set has moved.
    Begun,
    /// Renames may be in progress. Uncommitted: recovery restores.
    Swapping,
    /// The installed set is durable. Recovery finishes retention.
    Committed,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestartRecord {
    /// A running model must be restarted after the transaction.
    pub required: bool,
    /// The model was stopped (intent is persisted *before* the stop).
    pub stopped: bool,
    /// The restart attempt finished (successfully or not).
    pub done: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    Installed,
    RolledBack { reason: String },
    Uninstalled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Journal {
    pub schema: u32,
    pub id: String,
    pub op: Op,
    pub label: String,
    pub started_unix: u64,
    pub steps: Vec<Step>,
    pub had_live: bool,
    pub had_previous: bool,
    /// Metadata fingerprints (relative paths, sizes, mtimes): identity of a set
    /// that survives directory renames. Never credentials, never file content.
    pub live_fp: Option<String>,
    pub previous_fp: Option<String>,
    pub staged_fp: Option<String>,
    /// Directory name (under the runtime parent) of the install candidate.
    pub staged: Option<String>,
    pub phase: Phase,
    pub restart: RestartRecord,
    pub outcome: Option<Outcome>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalState {
    None,
    Present(Box<Journal>),
    /// Unreadable or from an unknown schema. Nothing is touched.
    Corrupt(String),
}

pub fn read_journal(paths: &RuntimePaths) -> JournalState {
    match std::fs::read(&paths.journal) {
        Ok(bytes) => match serde_json::from_slice::<Journal>(&bytes) {
            Ok(j) if j.schema == SCHEMA => JournalState::Present(Box::new(j)),
            Ok(j) => JournalState::Corrupt(format!("unknown journal schema {}", j.schema)),
            Err(e) => JournalState::Corrupt(format!("unreadable journal: {e}")),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => JournalState::None,
        Err(e) => JournalState::Corrupt(format!("cannot read journal: {e}")),
    }
}

fn sync_dir(dir: &Path) -> std::io::Result<()> {
    // Directory handles cannot be opened for sync on Windows; a rename there is
    // already write-through at the metadata level for our purposes.
    #[cfg(unix)]
    {
        std::fs::File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

/// Atomically replace the journal and flush file and directory. A failure here
/// means durability is unknown; callers re-read the journal to learn the truth.
fn write_journal(
    paths: &RuntimePaths,
    j: &Journal,
    faults: &dyn Faults,
    name: &str,
) -> Result<(), Interrupt> {
    point(faults, &format!("journal.{name}.before"))?;
    let bytes = serde_json::to_vec_pretty(j).map_err(|e| Interrupt::Failed(e.to_string()))?;
    let tmp = paths.tmp_journal();
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    point(faults, &format!("journal.{name}.tmp"))?;
    std::fs::rename(&tmp, &paths.journal)?;
    sync_dir(&paths.parent)?;
    point(faults, &format!("journal.{name}.after"))?;
    Ok(())
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn new_transaction_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}-{:x}", std::process::id())
}

// ---------------------------------------------------------------------------
// Fingerprints
// ---------------------------------------------------------------------------

/// Identity of a runtime set that survives being renamed: sorted relative
/// paths with size and modification time. Renaming a directory preserves every
/// file's mtime, so the fingerprint follows the set wherever it is moved, and
/// no multi-gigabyte hashing happens at every boundary.
pub fn dir_fingerprint(dir: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut entries: Vec<(String, u64, u128)> = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                stack.push(path);
                continue;
            }
            let meta = entry.metadata()?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let rel = path
                .strip_prefix(dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            entries.push((rel, meta.len(), mtime));
        }
    }
    entries.sort();
    let mut hasher = Sha256::new();
    for (rel, len, mtime) in &entries {
        hasher.update(rel.as_bytes());
        hasher.update([0]);
        hasher.update(len.to_le_bytes());
        hasher.update(mtime.to_le_bytes());
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn fingerprint_if_present(dir: &Path) -> std::io::Result<Option<String>> {
    if dir.is_dir() {
        dir_fingerprint(dir).map(Some)
    } else {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoverError {
    /// Ambiguous or inconsistent state. Everything is retained; mutations stay
    /// refused until the state is repaired. Never guessed, never deleted.
    Blocked(String),
    Interrupted(Interrupt),
}

impl From<Interrupt> for RecoverError {
    fn from(i: Interrupt) -> Self {
        RecoverError::Interrupted(i)
    }
}

impl From<std::io::Error> for RecoverError {
    fn from(e: std::io::Error) -> Self {
        RecoverError::Interrupted(Interrupt::Failed(e.to_string()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcileKind {
    /// Uncommitted: the last usable live set and old previous set are back.
    Restored,
    /// Committed: retention is finished.
    Completed,
}

fn rename_dir(from: &Path, to: &Path, faults: &dyn Faults, label: &str) -> Result<(), Interrupt> {
    point(faults, &format!("{label}.before"))?;
    std::fs::rename(from, to)?;
    sync_dir(to.parent().unwrap_or(to))?;
    point(faults, &format!("{label}.after"))?;
    Ok(())
}

fn remove_dir(path: &Path, faults: &dyn Faults, label: &str) -> Result<(), Interrupt> {
    point(faults, &format!("{label}.before"))?;
    if path.exists() {
        std::fs::remove_dir_all(path)?;
    }
    point(faults, &format!("{label}.after"))?;
    Ok(())
}

/// Undo performed renames in reverse. A step has happened exactly when its
/// `from` is gone. Idempotent.
fn undo(paths: &RuntimePaths, j: &Journal, faults: &dyn Faults) -> Result<(), RecoverError> {
    for (index, step) in j.steps.iter().enumerate().rev() {
        let from = paths.role(step.from, j);
        let to = paths.role(step.to, j);
        if from.exists() {
            continue; // not performed
        }
        if !to.exists() {
            return Err(RecoverError::Blocked(format!(
                "step {index}: neither {} nor {} exists; a runtime set is missing",
                from.display(),
                to.display()
            )));
        }
        rename_dir(&to, &from, faults, &format!("undo.{index}"))?;
    }
    Ok(())
}

fn expect_fp(dir: &Path, expected: &Option<String>, what: &str) -> Result<(), RecoverError> {
    match expected {
        Some(fp) => match fingerprint_if_present(dir)? {
            Some(actual) if &actual == fp => Ok(()),
            Some(_) => Err(RecoverError::Blocked(format!(
                "{what} at {} does not match the journal fingerprint",
                dir.display()
            ))),
            None => Err(RecoverError::Blocked(format!(
                "{what} is missing at {}",
                dir.display()
            ))),
        },
        None => {
            if dir.exists() {
                Err(RecoverError::Blocked(format!(
                    "{what} unexpectedly exists at {}",
                    dir.display()
                )))
            } else {
                Ok(())
            }
        }
    }
}

fn verify_restored(paths: &RuntimePaths, j: &Journal) -> Result<(), RecoverError> {
    expect_fp(&paths.live, &j.live_fp, "the live set")?;
    expect_fp(&paths.previous, &j.previous_fp, "the previous set")
}

fn verify_committed_live(paths: &RuntimePaths, j: &Journal) -> Result<(), RecoverError> {
    match j.op {
        Op::Install => expect_fp(&paths.live, &j.staged_fp, "the installed set"),
        Op::Rollback => expect_fp(&paths.live, &j.previous_fp, "the rolled-back set"),
        Op::Uninstall => expect_fp(&paths.live, &None, "the live set"),
    }
}

fn verify_retained(paths: &RuntimePaths, j: &Journal) -> Result<(), RecoverError> {
    verify_committed_live(paths, j)?;
    let previous_expected = match j.op {
        Op::Install => {
            if j.had_live {
                j.live_fp.clone()
            } else {
                j.previous_fp.clone()
            }
        }
        Op::Rollback => {
            if j.had_live {
                j.live_fp.clone()
            } else {
                None
            }
        }
        Op::Uninstall => None,
    };
    expect_fp(
        &paths.previous,
        &previous_expected,
        "the retained previous set",
    )
}

/// Finish retention for a committed transaction. Only runs after commit.
fn retain(paths: &RuntimePaths, j: &Journal, faults: &dyn Faults) -> Result<(), RecoverError> {
    let old_live = paths.role(Role::OldLive, j);
    let old_previous = paths.role(Role::OldPrevious, j);
    match j.op {
        Op::Install => {
            if old_live.exists() {
                if paths.previous.exists() {
                    return Err(RecoverError::Blocked(
                        "the previous slot is occupied while retaining the old live set".into(),
                    ));
                }
                rename_dir(&old_live, &paths.previous, faults, "retain.live")?;
            }
            if old_previous.exists() {
                if j.had_live {
                    remove_dir(&old_previous, faults, "retain.superseded")?;
                } else if !paths.previous.exists() {
                    rename_dir(
                        &old_previous,
                        &paths.previous,
                        faults,
                        "retain.keep-previous",
                    )?;
                }
            }
        }
        Op::Rollback => {
            if old_live.exists() {
                if paths.previous.exists() {
                    return Err(RecoverError::Blocked(
                        "the previous slot is occupied while retaining the replaced live set"
                            .into(),
                    ));
                }
                rename_dir(&old_live, &paths.previous, faults, "retain.live")?;
            }
        }
        Op::Uninstall => {
            remove_dir(&old_live, faults, "retain.old-live")?;
            remove_dir(&old_previous, faults, "retain.old-previous")?;
        }
    }
    Ok(())
}

/// Reconcile an on-disk journal. Leaves the journal in place; [`finalize`]
/// removes it. Never deletes a set that the journal does not mark as
/// superseded or as a rejected candidate.
pub fn reconcile(
    paths: &RuntimePaths,
    j: &Journal,
    faults: &dyn Faults,
) -> Result<ReconcileKind, RecoverError> {
    match j.phase {
        Phase::Begun | Phase::Swapping => {
            // The rejected candidate is *not* removed here: undo infers that a
            // rename happened from its source being gone, so deleting the
            // candidate inside the transaction would make a replay mistake the
            // restored live set for it. It is discarded after the journal is
            // retired (see `discard_candidate`), when it is a plain orphan.
            undo(paths, j, faults)?;
            verify_restored(paths, j)?;
            Ok(ReconcileKind::Restored)
        }
        Phase::Committed => {
            verify_committed_live(paths, j)?;
            retain(paths, j, faults)?;
            verify_retained(paths, j)?;
            Ok(ReconcileKind::Completed)
        }
    }
}

/// Best-effort removal of a rejected install candidate once the journal that
/// referenced it is gone. A failure is harmless: startup sweeps it as an orphan.
pub fn discard_candidate(paths: &RuntimePaths, j: &Journal) {
    if j.op == Op::Install {
        let _ = std::fs::remove_dir_all(paths.role(Role::Staged, j));
    }
}

/// Persist the terminal outcome, then remove the journal.
pub fn finalize(
    paths: &RuntimePaths,
    j: &Journal,
    outcome: Outcome,
    faults: &dyn Faults,
) -> Result<(), Interrupt> {
    let mut last = j.clone();
    last.outcome = Some(outcome);
    point(faults, "finalize.last.before")?;
    std::fs::write(
        paths.last_journal(),
        serde_json::to_vec_pretty(&last).map_err(|e| Interrupt::Failed(e.to_string()))?,
    )?;
    point(faults, "finalize.last.after")?;
    point(faults, "finalize.remove.before")?;
    match std::fs::remove_file(&paths.journal) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let _ = std::fs::remove_file(paths.tmp_journal());
    sync_dir(&paths.parent)?;
    point(faults, "finalize.remove.after")?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryReport {
    Nothing,
    Restored { journal: Box<Journal> },
    Completed { journal: Box<Journal> },
    Blocked { reason: String },
}

impl RecoveryReport {
    /// The model was stopped by an interrupted transaction and no restart was
    /// recorded as finished.
    pub fn restart_pending(&self) -> bool {
        match self {
            Self::Restored { journal } | Self::Completed { journal } => {
                journal.restart.required && journal.restart.stopped && !journal.restart.done
            }
            _ => false,
        }
    }
}

/// Replay whatever journal is on disk: restore if uncommitted, finish if
/// committed, then finalize. Safe to run any number of times.
pub fn recover(paths: &RuntimePaths, faults: &dyn Faults) -> Result<RecoveryReport, Interrupt> {
    let journal = match read_journal(paths) {
        JournalState::None => {
            let _ = std::fs::remove_file(paths.tmp_journal());
            return Ok(RecoveryReport::Nothing);
        }
        JournalState::Corrupt(reason) => return Ok(RecoveryReport::Blocked { reason }),
        JournalState::Present(j) => j,
    };
    match reconcile(paths, &journal, faults) {
        Ok(ReconcileKind::Restored) => {
            let outcome = Outcome::RolledBack {
                reason: "interrupted before commit; last usable runtime restored".into(),
            };
            finalize(paths, &journal, outcome, faults)?;
            discard_candidate(paths, &journal);
            Ok(RecoveryReport::Restored { journal })
        }
        Ok(ReconcileKind::Completed) => {
            let outcome = match journal.op {
                Op::Uninstall => Outcome::Uninstalled,
                _ => Outcome::Installed,
            };
            finalize(paths, &journal, outcome, faults)?;
            Ok(RecoveryReport::Completed { journal })
        }
        Err(RecoverError::Blocked(reason)) => Ok(RecoveryReport::Blocked { reason }),
        Err(RecoverError::Interrupted(i)) => Err(i),
    }
}

// ---------------------------------------------------------------------------
// Orphan and legacy cleanup (only ever after recovery succeeded)
// ---------------------------------------------------------------------------

/// Remove staging/transaction leftovers that no journal references, and fold
/// pre-journal timestamped rollback backups into the single previous slot.
/// Refuses to run while any journal (or a torn journal write) is present.
pub fn sweep_orphans(paths: &RuntimePaths) -> std::io::Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    if paths.journal.exists() {
        return Ok(removed);
    }
    let _ = std::fs::remove_file(paths.tmp_journal());
    let staging = paths.staging_prefix();
    let txn = paths.txn_prefix();
    let legacy_staging = format!(".{}.llama-update-", paths.name);
    let legacy_failed = format!("{}-failed-", paths.name);
    let legacy_previous = format!("{}-previous-", paths.name);
    let mut legacy_previous_dirs: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();
    for entry in std::fs::read_dir(&paths.parent)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        if name.starts_with(&staging)
            || name.starts_with(&txn)
            || name.starts_with(&legacy_staging)
            || name.starts_with(&legacy_failed)
        {
            std::fs::remove_dir_all(&path)?;
            removed.push(path);
        } else if name.starts_with(&legacy_previous) {
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            legacy_previous_dirs.push((path, modified));
        }
    }
    // Exactly one previous set: keep the managed slot; if it is empty, adopt
    // the newest pre-journal backup. Everything else is redundant.
    legacy_previous_dirs.sort_by_key(|(_, modified)| std::cmp::Reverse(*modified));
    let mut iter = legacy_previous_dirs.into_iter();
    if !paths.previous.exists()
        && let Some((newest, _)) = iter.next()
    {
        std::fs::rename(&newest, &paths.previous)?;
    }
    for (path, _) in iter {
        std::fs::remove_dir_all(&path)?;
        removed.push(path);
    }
    Ok(removed)
}

// ---------------------------------------------------------------------------
// Host and orchestration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyStage {
    /// The candidate in its pre-swap location.
    Staged,
    /// The candidate at the final live path.
    Live,
}

/// Everything the engine needs from the process around it. Production wires
/// this to `AppState`; tests use a recording fake.
pub trait PromoteHost: Send + Sync {
    fn faults(&self) -> &dyn Faults {
        &NoFaults
    }
    fn cancelled(&self) -> bool {
        false
    }
    fn model_running(&self) -> bool;
    fn stop_model(&self) -> impl Future<Output = Result<(), String>> + Send;
    fn restart_model(&self) -> impl Future<Output = Result<(), String>> + Send;
    fn verify(
        &self,
        dir: &Path,
        exe_name: &str,
        intent: &VerificationIntent,
        stage: VerifyStage,
    ) -> impl Future<Output = Result<VerificationReport, String>> + Send;
}

pub struct PromoteRequest {
    /// Complete candidate set, a sibling `.bin.staging-<id>` of the live dir.
    pub staged_dir: PathBuf,
    pub label: String,
    pub exe_name: String,
    /// Written into the staged set before it is verified. Provenance only.
    pub manifest: Option<RuntimeManifest>,
    pub verification: VerificationIntent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromoteOutcome {
    Promoted {
        restarted: bool,
    },
    /// The binaries are installed and verified; only the model restart failed.
    /// Rollback stays available; nothing was rolled back automatically.
    InstalledRestartFailed {
        restart_error: String,
    },
    RolledBack {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromoteResult {
    pub outcome: PromoteOutcome,
    /// Non-fatal: for example retention left for the next startup.
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromoteError {
    Cancelled,
    /// A previous transaction could not be reconciled; mutations are refused.
    JournalBlocked(String),
    /// Nothing was mutated.
    Prepare(String),
    /// The journal could not be written; nothing was mutated.
    Journal(String),
    /// Compensation did not finish. The journal is retained for startup recovery.
    RecoveryIncomplete(String),
    /// Test hook: the process "died" at this point.
    Died(String),
}

impl std::fmt::Display for PromoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "cancelled before the swap began"),
            Self::JournalBlocked(r) => write!(
                f,
                "a previous runtime transaction needs repair and blocks changes: {r}"
            ),
            Self::Prepare(r) => write!(f, "could not prepare the new runtime: {r}"),
            Self::Journal(r) => write!(f, "could not record the runtime transaction: {r}"),
            Self::RecoveryIncomplete(r) => write!(
                f,
                "the runtime swap could not be fully undone (recovery will resume at next start): {r}"
            ),
            Self::Died(p) => write!(f, "interrupted at {p}"),
        }
    }
}

impl std::error::Error for PromoteError {}

impl From<Interrupt> for PromoteError {
    fn from(i: Interrupt) -> Self {
        match i {
            Interrupt::Died(p) => PromoteError::Died(p),
            Interrupt::Failed(r) => PromoteError::RecoveryIncomplete(r),
        }
    }
}

fn recover_error(e: RecoverError) -> PromoteError {
    match e {
        RecoverError::Blocked(r) => PromoteError::RecoveryIncomplete(r),
        RecoverError::Interrupted(i) => i.into(),
    }
}

/// Recover whatever a previous transaction left, or refuse to continue.
fn ensure_clean(paths: &RuntimePaths, faults: &dyn Faults) -> Result<(), PromoteError> {
    match recover(paths, faults)? {
        RecoveryReport::Blocked { reason } => Err(PromoteError::JournalBlocked(reason)),
        _ => Ok(()),
    }
}

fn new_journal(
    op: Op,
    label: &str,
    paths: &RuntimePaths,
    staged: Option<&Path>,
) -> std::io::Result<Journal> {
    let id = new_transaction_id();
    let had_live = paths.live.is_dir();
    let had_previous = paths.previous.is_dir();
    let live_fp = fingerprint_if_present(&paths.live)?;
    let previous_fp = fingerprint_if_present(&paths.previous)?;
    let mut steps = Vec::new();
    match op {
        Op::Install => {
            if had_previous {
                steps.push(Step {
                    from: Role::Previous,
                    to: Role::OldPrevious,
                });
            }
            if had_live {
                steps.push(Step {
                    from: Role::Live,
                    to: Role::OldLive,
                });
            }
            steps.push(Step {
                from: Role::Staged,
                to: Role::Live,
            });
        }
        Op::Rollback => {
            if had_live {
                steps.push(Step {
                    from: Role::Live,
                    to: Role::OldLive,
                });
            }
            steps.push(Step {
                from: Role::Previous,
                to: Role::Live,
            });
        }
        Op::Uninstall => {
            if had_live {
                steps.push(Step {
                    from: Role::Live,
                    to: Role::OldLive,
                });
            }
            if had_previous {
                steps.push(Step {
                    from: Role::Previous,
                    to: Role::OldPrevious,
                });
            }
        }
    }
    Ok(Journal {
        schema: SCHEMA,
        id,
        op,
        label: label.to_string(),
        started_unix: now_unix(),
        steps,
        had_live,
        had_previous,
        live_fp,
        previous_fp,
        staged_fp: None,
        staged: staged
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned()),
        phase: Phase::Begun,
        restart: RestartRecord::default(),
        outcome: None,
    })
}

struct Running<'a, H: PromoteHost> {
    host: &'a H,
    paths: &'a RuntimePaths,
    exe_name: &'a str,
    intent: &'a VerificationIntent,
    journal: Journal,
    warnings: Vec<String>,
}

enum Flow {
    Done,
    /// Forward progress failed; run recovery from the on-disk journal.
    Abort(String),
    Died(String),
}

fn abort_or_die(i: Interrupt) -> Flow {
    match i {
        Interrupt::Died(p) => Flow::Died(p),
        Interrupt::Failed(r) => Flow::Abort(r),
    }
}

impl<H: PromoteHost> Running<'_, H> {
    fn faults(&self) -> &dyn Faults {
        self.host.faults()
    }

    fn write(&self, name: &str) -> Result<(), Interrupt> {
        write_journal(self.paths, &self.journal, self.faults(), name)
    }

    /// Persist restart intent *before* stopping a model that could contend
    /// for the GPU, then stop it.
    async fn stop_model_if_running(&mut self) -> Flow {
        if !self.host.model_running() {
            return Flow::Done;
        }
        self.journal.restart.required = true;
        if let Err(i) = self.write("restart-intent") {
            return abort_or_die(i);
        }
        if let Err(i) = point(self.faults(), "model.stop.before") {
            return abort_or_die(i);
        }
        if let Err(e) = self.host.stop_model().await {
            return Flow::Abort(format!("could not stop the running model: {e}"));
        }
        self.journal.restart.stopped = true;
        if let Err(i) = self.write("model-stopped") {
            return abort_or_die(i);
        }
        Flow::Done
    }

    async fn verify(&self, dir: &Path, stage: VerifyStage) -> Flow {
        let label = match stage {
            VerifyStage::Staged => "verify.staged",
            VerifyStage::Live => "verify.live",
        };
        if let Err(i) = point(self.faults(), label) {
            return abort_or_die(i);
        }
        match self
            .host
            .verify(dir, self.exe_name, self.intent, stage)
            .await
        {
            Ok(_) => Flow::Done,
            Err(e) => Flow::Abort(e),
        }
    }

    fn run_steps(&self) -> Flow {
        for (index, step) in self.journal.steps.iter().enumerate() {
            let from = self.paths.role(step.from, &self.journal);
            let to = self.paths.role(step.to, &self.journal);
            if let Err(i) = rename_dir(&from, &to, self.faults(), &format!("rename.{index}")) {
                return abort_or_die(i);
            }
        }
        Flow::Done
    }

    /// Everything from "restore the last usable runtime" to "tell the caller".
    /// The journal on disk is the truth: a failed commit write may in fact have
    /// become durable, in which case the transaction completes instead.
    async fn abort(mut self, reason: String) -> Result<PromoteResult, PromoteError> {
        let faults = self.host.faults();
        let on_disk = match read_journal(self.paths) {
            JournalState::Present(j) => *j,
            JournalState::None => self.journal.clone(),
            JournalState::Corrupt(r) => return Err(PromoteError::RecoveryIncomplete(r)),
        };
        self.journal = on_disk;
        match reconcile(self.paths, &self.journal, faults).map_err(recover_error)? {
            ReconcileKind::Completed => self.finish_success().await,
            ReconcileKind::Restored => {
                self.restart_model().await?;
                let outcome = Outcome::RolledBack {
                    reason: reason.clone(),
                };
                finalize(self.paths, &self.journal, outcome, faults)?;
                discard_candidate(self.paths, &self.journal);
                Ok(PromoteResult {
                    outcome: PromoteOutcome::RolledBack { reason },
                    warnings: self.warnings,
                })
            }
        }
    }

    /// Restart the remembered model once. A failure is reported, never turned
    /// into a binary rollback.
    async fn restart_model(&mut self) -> Result<Option<Result<(), String>>, PromoteError> {
        if !(self.journal.restart.required && self.journal.restart.stopped) {
            return Ok(None);
        }
        match point(self.host.faults(), "model.restart.before") {
            Err(Interrupt::Died(p)) => return Err(PromoteError::Died(p)),
            // An injected failure here is the restart failing, not a crash.
            Err(Interrupt::Failed(e)) => {
                self.journal.restart.done = true;
                return Ok(Some(Err(e)));
            }
            Ok(()) => {}
        }
        let result = self.host.restart_model().await;
        self.journal.restart.done = true;
        Ok(Some(result))
    }

    async fn finish_success(mut self) -> Result<PromoteResult, PromoteError> {
        let faults = self.host.faults();
        // Retention (idempotent) before the restart; leftover work is retried
        // by startup recovery while the journal stays Committed.
        let mut finalize_now = true;
        if let Err(e) = reconcile(self.paths, &self.journal, faults) {
            match e {
                RecoverError::Interrupted(Interrupt::Died(p)) => return Err(PromoteError::Died(p)),
                other => {
                    finalize_now = false;
                    self.warnings.push(format!(
                        "retention is incomplete and will finish at next start: {}",
                        match other {
                            RecoverError::Blocked(r) => r,
                            RecoverError::Interrupted(Interrupt::Failed(r)) => r,
                            RecoverError::Interrupted(Interrupt::Died(r)) => r,
                        }
                    ));
                }
            }
        }
        let restart = self.restart_model().await?;
        let outcome = match restart {
            None => PromoteOutcome::Promoted { restarted: false },
            Some(Ok(())) => PromoteOutcome::Promoted { restarted: true },
            Some(Err(e)) => PromoteOutcome::InstalledRestartFailed { restart_error: e },
        };
        if finalize_now {
            let outcome_record = match self.journal.op {
                Op::Uninstall => Outcome::Uninstalled,
                _ => Outcome::Installed,
            };
            finalize(self.paths, &self.journal, outcome_record, faults)?;
        }
        Ok(PromoteResult {
            outcome,
            warnings: self.warnings,
        })
    }
}

macro_rules! step {
    ($running:expr, $expr:expr) => {
        match $expr {
            Flow::Done => {}
            Flow::Abort(reason) => return $running.abort(reason).await,
            Flow::Died(p) => return Err(PromoteError::Died(p)),
        }
    };
}

async fn run_transaction<H: PromoteHost>(
    mut running: Running<'_, H>,
    candidate: Option<PathBuf>,
) -> Result<PromoteResult, PromoteError> {
    let op = running.journal.op;
    step!(running, running.stop_model_if_running().await);

    // Verify the candidate where it sits (the staged dir, or the previous set
    // for a rollback), before anything is renamed.
    if let Some(dir) = &candidate {
        step!(running, running.verify(dir, VerifyStage::Staged).await);
    }

    // The staged set is now final; record its identity with the swap intent.
    running.journal.staged_fp = match (op, &candidate) {
        (Op::Install, Some(dir)) => match fingerprint_if_present(dir) {
            Ok(fp) => fp,
            Err(e) => {
                return running
                    .abort(format!("cannot fingerprint the candidate: {e}"))
                    .await;
            }
        },
        _ => None,
    };
    running.journal.phase = Phase::Swapping;
    step!(
        running,
        running
            .write("swapping")
            .map_or_else(abort_or_die, |_| Flow::Done)
    );
    step!(running, running.run_steps());

    if matches!(op, Op::Install | Op::Rollback) {
        let live = running.paths.live.clone();
        step!(running, running.verify(&live, VerifyStage::Live).await);
    }

    running.journal.phase = Phase::Committed;
    step!(
        running,
        running
            .write("committed")
            .map_or_else(abort_or_die, |_| Flow::Done)
    );
    running.finish_success().await
}

#[allow(clippy::too_many_arguments)]
async fn begin<H: PromoteHost>(
    host: &H,
    guard: &mut RuntimeGuard,
    paths: &RuntimePaths,
    op: Op,
    label: &str,
    exe_name: &str,
    intent: &VerificationIntent,
    staged: Option<&Path>,
) -> Result<PromoteResult, PromoteError> {
    let faults = host.faults();
    ensure_clean(paths, faults)?;
    if host.cancelled() {
        return Err(PromoteError::Cancelled);
    }
    let owned_paths = paths.clone();
    let label_owned = label.to_string();
    let staged_owned = staged.map(Path::to_path_buf);
    let journal = tokio::task::spawn_blocking(move || {
        new_journal(op, &label_owned, &owned_paths, staged_owned.as_deref())
    })
    .await
    .map_err(|e| PromoteError::Prepare(e.to_string()))?
    .map_err(|e| PromoteError::Prepare(e.to_string()))?;

    // Journal creation begins the noncancellable transaction.
    guard.enter_swap_window();
    let candidate = match op {
        Op::Install => staged.map(Path::to_path_buf),
        Op::Rollback => Some(paths.previous.clone()),
        Op::Uninstall => None,
    };
    let running = Running {
        host,
        paths,
        exe_name,
        intent,
        journal,
        warnings: Vec::new(),
    };
    if let Err(i) = running.write("begin") {
        let _ = std::fs::remove_file(paths.tmp_journal());
        return Err(match i {
            Interrupt::Died(p) => PromoteError::Died(p),
            Interrupt::Failed(r) => PromoteError::Journal(r),
        });
    }
    run_transaction(running, candidate).await
}

/// Install a staged candidate as the live runtime.
pub async fn promote<H: PromoteHost>(
    host: &H,
    guard: &mut RuntimeGuard,
    paths: &RuntimePaths,
    req: PromoteRequest,
) -> Result<PromoteResult, PromoteError> {
    let staged_ok = req.staged_dir.parent() == Some(paths.parent.as_path())
        && req
            .staged_dir
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(&paths.staging_prefix()));
    if !staged_ok {
        return Err(PromoteError::Prepare(format!(
            "the staged runtime must be a `{}*` sibling of the live directory",
            paths.staging_prefix()
        )));
    }
    ensure_clean(paths, host.faults())?;
    if !req.staged_dir.is_dir() {
        return Err(PromoteError::Prepare(
            "the staged runtime does not exist".into(),
        ));
    }
    // Preparation precedes any execution of the candidate.
    let staged = req.staged_dir.clone();
    let exe = req.exe_name.clone();
    let manifest = req.manifest.clone();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let inventory = prepare_runtime_dir(&staged, &exe)?;
        if inventory.executable.is_none() {
            return Err(std::io::Error::other(format!(
                "{exe} is missing from the staged runtime"
            )));
        }
        if let Some(manifest) = manifest {
            manifest.write(&staged)?;
        }
        Ok(())
    })
    .await
    .map_err(|e| PromoteError::Prepare(e.to_string()))?
    .map_err(|e| PromoteError::Prepare(e.to_string()))?;
    begin(
        host,
        guard,
        paths,
        Op::Install,
        &req.label,
        &req.exe_name,
        &req.verification,
        Some(&req.staged_dir),
    )
    .await
}

/// Swap the live and previous sets. Rolling back twice restores the original.
pub async fn rollback<H: PromoteHost>(
    host: &H,
    guard: &mut RuntimeGuard,
    paths: &RuntimePaths,
    exe_name: &str,
) -> Result<PromoteResult, PromoteError> {
    ensure_clean(paths, host.faults())?;
    if !paths.previous.is_dir() {
        return Err(PromoteError::Prepare(
            "there is no previous runtime to roll back to".into(),
        ));
    }
    begin(
        host,
        guard,
        paths,
        Op::Rollback,
        "rollback",
        exe_name,
        &VerificationIntent::rollback(),
        None,
    )
    .await
}

/// Remove the live and previous sets under the same journal and gate.
pub async fn uninstall<H: PromoteHost>(
    host: &H,
    guard: &mut RuntimeGuard,
    paths: &RuntimePaths,
    exe_name: &str,
) -> Result<PromoteResult, PromoteError> {
    ensure_clean(paths, host.faults())?;
    if !paths.live.is_dir() && !paths.previous.is_dir() {
        return Err(PromoteError::Prepare(
            "there is no managed runtime to remove".into(),
        ));
    }
    begin(
        host,
        guard,
        paths,
        Op::Uninstall,
        "uninstall",
        exe_name,
        &VerificationIntent::rollback(),
        None,
    )
    .await
}

/// The last finished transaction, for status surfaces. `None` when nothing has
/// run (or the record is unreadable).
pub fn last_transaction(paths: &RuntimePaths) -> Option<Journal> {
    std::fs::read(paths.last_journal())
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}

#[cfg(test)]
#[path = "promote_tests.rs"]
mod tests;
