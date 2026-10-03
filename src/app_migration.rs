//! Pure application-root inspection and versioned migration state.
//!
//! Inspection reads metadata only. It never creates, moves, deletes, opens a
//! database, initializes tokens, or chooses between divergent roots.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::identity::PRODUCT_SLUG;
use crate::paths::AppPaths;

const STATE_MARKERS: &[&str] = &[
    "encryption-key",
    "api-token",
    "db-admin-token",
    "auth-config.json",
    "tls-config.json",
    "presets.json",
    "sessions.json",
    "chat.db",
    "models",
    "certs",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootState {
    Fresh,
    LegacyActive,
    NewActive,
    BothIdentical,
    Conflict,
    MigrationQueued,
    Migrating,
    MigrationFailed,
    RollbackAvailable,
    CustomConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootInspection {
    pub state: RootState,
    pub canonical_root: PathBuf,
    pub legacy_root: PathBuf,
    pub active_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    Critical,
    ModelRetained,
    Recreatable,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    File,
    Directory,
}

/// Stable, sanitized error identifiers for API/CLI consumers. Detailed
/// filesystem paths and OS diagnostics stay local to logs/receipts and are
/// never part of this public contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationErrorCode {
    InvalidRoot,
    UnsafeEntry,
    DestinationConflict,
    StalePlan,
    QueueConflict,
    PermissionDenied,
    IoFailure,
}

pub fn public_error_code(error: &anyhow::Error) -> MigrationErrorCode {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("stale") || message.contains("plan does not match") {
        MigrationErrorCode::StalePlan
    } else if message.contains("symlink")
        || message.contains("special filesystem")
        || message.contains("unsafe relative")
    {
        MigrationErrorCode::UnsafeEntry
    } else if message.contains("destination")
        || message.contains("overwrite")
        || message.contains("identical")
    {
        MigrationErrorCode::DestinationConflict
    } else if message.contains("permission") || message.contains("denied") {
        MigrationErrorCode::PermissionDenied
    } else if message.contains("root is not") || message.contains("root must") {
        MigrationErrorCode::InvalidRoot
    } else if message.contains("queue") {
        MigrationErrorCode::QueueConflict
    } else {
        MigrationErrorCode::IoFailure
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppHomeMigrationEntry {
    pub relative_path: PathBuf,
    pub class: ResourceClass,
    pub kind: EntryKind,
    pub bytes: u64,
    pub modified_unix_seconds: u64,
}

/// The two application-home roots participating in a migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppHomeRoots {
    pub canonical: PathBuf,
    pub legacy: PathBuf,
    pub disposable: bool,
}

/// Resolve an explicitly disposable migration root. This is intentionally
/// strict: test roots must live below the current user's temp directory and
/// may not target either real application root.
pub fn disposable_roots(root: &Path) -> Result<AppHomeRoots> {
    let root = root
        .canonicalize()
        .with_context(|| format!("migration test root is not readable: {}", root.display()))?;
    let temp = std::env::temp_dir().canonicalize()?;
    if !root.starts_with(&temp) || root == temp {
        bail!("migration test root must be a child of the temp directory");
    }
    let metadata = fs::symlink_metadata(&root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("migration test root must be a real directory");
    }
    Ok(AppHomeRoots {
        canonical: root.join("local-llm-foundry"),
        legacy: root.join("llama-monitor"),
        disposable: true,
    })
}

pub fn default_roots() -> AppHomeRoots {
    AppHomeRoots {
        canonical: AppPaths::canonical_default_root(),
        legacy: AppPaths::legacy_default_root(),
        disposable: false,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeMigrationPlan {
    pub schema_version: u32,
    pub plan_id: String,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub entries: Vec<AppHomeMigrationEntry>,
    pub retained_entries: Vec<PathBuf>,
    pub required_copy_bytes: u64,
    pub total_seen_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationJournalState {
    Planned,
    Copying,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeMigrationJournal {
    pub schema_version: u32,
    pub plan_id: String,
    pub state: MigrationJournalState,
    #[serde(default)]
    pub completed_entries: Vec<PathBuf>,
    #[serde(default)]
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeMigrationReceipt {
    pub schema_version: u32,
    pub plan_id: String,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub copied_entries: Vec<PathBuf>,
    #[serde(default)]
    pub retained_entries: Vec<PathBuf>,
    pub verified_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeMigrationRequest {
    pub schema_version: u32,
    pub plan_id: String,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub requested_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeRollbackPlan {
    pub schema_version: u32,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub receipt_path: PathBuf,
    pub legacy_root_retained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeRollbackRequest {
    pub schema_version: u32,
    pub source: PathBuf,
    pub destination: PathBuf,
    pub receipt_path: PathBuf,
    pub requested_unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppHomeCleanupPlan {
    pub schema_version: u32,
    pub legacy_root: PathBuf,
    pub canonical_root: PathBuf,
    pub receipt_path: PathBuf,
}

static MIGRATION_QUEUE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn migration_request_path() -> PathBuf {
    let canonical = AppPaths::canonical_default_root();
    let parent = canonical.parent().unwrap_or_else(|| Path::new("."));
    migration_request_path_for_parent(parent)
}

pub fn migration_request_path_for_parent(parent: &Path) -> PathBuf {
    parent.join(format!(".{PRODUCT_SLUG}-migration-request.json"))
}

pub fn queue_application_home_migration(
    plan: &AppHomeMigrationPlan,
) -> Result<AppHomeMigrationRequest> {
    let _guard = MIGRATION_QUEUE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|_| anyhow::anyhow!("migration queue is unavailable"))?;
    validate_root(&plan.source, "source")?;
    let request_path = migration_request_path_for_parent(
        plan.destination.parent().unwrap_or_else(|| Path::new(".")),
    );
    if request_path.is_file() {
        bail!("a migration is already queued");
    }
    if plan.destination.exists()
        && (!plan.destination.is_dir() || fs::read_dir(&plan.destination)?.next().is_some())
    {
        bail!(
            "migration destination is no longer empty: {}",
            plan.destination.display()
        );
    }
    let request = AppHomeMigrationRequest {
        schema_version: 1,
        plan_id: plan.plan_id.clone(),
        source: plan.source.clone(),
        destination: plan.destination.clone(),
        requested_unix_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    write_json_atomic(&request_path, &request)?;
    Ok(request)
}

pub fn load_migration_request() -> Result<Option<AppHomeMigrationRequest>> {
    load_migration_request_from_parent(
        AppPaths::canonical_default_root()
            .parent()
            .unwrap_or_else(|| Path::new(".")),
    )
}

pub fn load_migration_request_from_parent(
    parent: &Path,
) -> Result<Option<AppHomeMigrationRequest>> {
    let path = migration_request_path_for_parent(parent);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_reader(fs::File::open(path)?)?))
}

pub fn migration_journal_path(plan: &AppHomeMigrationPlan) -> PathBuf {
    plan.destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".{PRODUCT_SLUG}-migration-journal.json"))
}

pub fn migration_receipt_path(plan: &AppHomeMigrationPlan) -> PathBuf {
    plan.destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".{PRODUCT_SLUG}-migration-{}.json", plan.plan_id))
}

pub fn migration_lock_path(plan: &AppHomeMigrationPlan) -> PathBuf {
    plan.destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".{PRODUCT_SLUG}-migration.lock"))
}

struct MigrationLock {
    path: PathBuf,
}

impl Drop for MigrationLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn acquire_migration_lock(plan: &AppHomeMigrationPlan) -> Result<MigrationLock> {
    let path = migration_lock_path(plan);
    if path.exists() {
        let stale = fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > std::time::Duration::from_secs(15 * 60));
        if stale {
            fs::remove_file(&path).context("could not recover stale migration lock")?;
        } else {
            bail!("migration is already running; retry after the active maintenance process exits");
        }
    }
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("could not acquire migration lock {}", path.display()))?;
    use std::io::Write;
    writeln!(
        file,
        "pid={} started_unix_seconds={}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    )?;
    Ok(MigrationLock { path })
}

fn ensure_free_space(destination: &Path, required_bytes: u64) -> Result<()> {
    let mut probe = destination.to_path_buf();
    while !probe.exists() {
        if !probe.pop() {
            return Ok(());
        }
    }
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let available = disks
        .list()
        .iter()
        .filter(|disk| probe.starts_with(disk.mount_point()))
        .max_by_key(|disk| disk.mount_point().as_os_str().len())
        .map(sysinfo::Disk::available_space);
    if let Some(available) = available
        && available < required_bytes
    {
        bail!(
            "insufficient free space for migration ({} bytes required)",
            required_bytes
        );
    }
    Ok(())
}

/// Execute a previously previewed plan in maintenance mode. The source is
/// retained as the rollback point; no destination file is overwritten. SQLite
/// is copied through `ChatStorage::backup`, never with `fs::copy`.
pub fn execute_application_home(plan: &AppHomeMigrationPlan) -> Result<AppHomeMigrationReceipt> {
    validate_root(&plan.source, "source")?;
    if let Ok(receipt_file) = fs::File::open(migration_receipt_path(plan)) {
        let receipt: AppHomeMigrationReceipt =
            serde_json::from_reader(receipt_file).context("migration receipt is unreadable")?;
        if receipt.plan_id == plan.plan_id {
            return Ok(receipt);
        }
    }
    let current_plan = plan_application_home(&plan.source, &plan.destination)
        .context("could not revalidate migration preview")?;
    if current_plan.plan_id != plan.plan_id {
        bail!("migration preview is stale; generate a new preview");
    }
    if plan.destination.exists()
        && (!plan.destination.is_dir() || fs::read_dir(&plan.destination)?.next().is_some())
    {
        bail!(
            "migration destination is no longer empty: {}",
            plan.destination.display()
        );
    }
    let _lock = acquire_migration_lock(plan)?;
    if plan.destination == plan.source {
        bail!("migration source and destination are identical");
    }
    ensure_free_space(&plan.destination, plan.required_copy_bytes)?;
    fs::create_dir_all(&plan.destination)?;
    let journal_path = migration_journal_path(plan);
    let mut journal = load_or_create_journal(plan, &journal_path)?;
    journal.state = MigrationJournalState::Copying;
    write_json_atomic(&journal_path, &journal)?;

    for entry in &plan.entries {
        if matches!(
            entry.class,
            ResourceClass::ModelRetained | ResourceClass::Recreatable
        ) {
            continue;
        }
        if journal
            .completed_entries
            .iter()
            .any(|path| path == &entry.relative_path)
        {
            continue;
        }
        let source = plan.source.join(&entry.relative_path);
        let destination = plan.destination.join(&entry.relative_path);
        let result = if entry.kind == EntryKind::Directory {
            fs::create_dir_all(&destination).map_err(anyhow::Error::from)
        } else if entry.relative_path == Path::new("chat.db") {
            copy_sqlite_with_backup(&source, &destination)
        } else if entry.relative_path == Path::new("chat.db-wal")
            || entry.relative_path == Path::new("chat.db-shm")
        {
            // The online backup contains a consistent database; WAL sidecars
            // are intentionally not copied as independent files.
            Ok(())
        } else {
            copy_file_atomic(&source, &destination)
        };
        if let Err(error) = result {
            journal.state = MigrationJournalState::Failed;
            journal.last_error = Some(error.to_string());
            write_json_atomic(&journal_path, &journal)?;
            return Err(error);
        }
        journal.completed_entries.push(entry.relative_path.clone());
        write_json_atomic(&journal_path, &journal)?;
    }

    for entry in &plan.entries {
        if matches!(
            entry.class,
            ResourceClass::ModelRetained | ResourceClass::Recreatable
        ) || entry.relative_path == Path::new("chat.db-wal")
            || entry.relative_path == Path::new("chat.db-shm")
        {
            continue;
        }
        let destination = plan.destination.join(&entry.relative_path);
        if entry.kind == EntryKind::File && !destination.is_file() {
            bail!("migration verification missing {}", destination.display());
        }
        if entry.kind == EntryKind::Directory && !destination.is_dir() {
            bail!("migration verification missing {}", destination.display());
        }
    }
    journal.state = MigrationJournalState::Complete;
    write_json_atomic(&journal_path, &journal)?;
    let receipt = AppHomeMigrationReceipt {
        schema_version: 1,
        plan_id: plan.plan_id.clone(),
        source: plan.source.clone(),
        destination: plan.destination.clone(),
        copied_entries: journal.completed_entries.clone(),
        retained_entries: plan.retained_entries.clone(),
        verified_entries: journal.completed_entries.len(),
    };
    write_json_atomic(&migration_receipt_path(plan), &receipt)?;
    let request_path = migration_request_path_for_parent(
        plan.destination.parent().unwrap_or_else(|| Path::new(".")),
    );
    if request_path.is_file() {
        let _ = fs::remove_file(request_path);
    }
    Ok(receipt)
}

/// Build a rollback plan only when a verified Stage A receipt proves that the
/// legacy source is still intact. Planning never removes or rewrites anything.
pub fn plan_application_home_rollback(
    canonical_root: &Path,
    legacy_root: &Path,
) -> Result<AppHomeRollbackPlan> {
    if !legacy_root.is_dir() {
        bail!("legacy rollback root is not available");
    }
    let parent = canonical_root
        .parent()
        .ok_or_else(|| anyhow::anyhow!("canonical root has no parent"))?;
    let receipt_path = fs::read_dir(parent)?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with(&format!(".{PRODUCT_SLUG}-migration-"))
                        && name.ends_with(".json")
                })
                && fs::File::open(path)
                    .ok()
                    .and_then(|file| {
                        serde_json::from_reader::<_, AppHomeMigrationReceipt>(file).ok()
                    })
                    .is_some_and(|receipt| {
                        receipt.destination == canonical_root && receipt.source == legacy_root
                    })
        })
        .ok_or_else(|| {
            anyhow::anyhow!("no verified migration receipt is available for rollback")
        })?;
    Ok(AppHomeRollbackPlan {
        schema_version: 1,
        source: legacy_root.to_path_buf(),
        destination: canonical_root.to_path_buf(),
        receipt_path,
        legacy_root_retained: true,
    })
}

/// Execute an explicitly confirmed rollback. The legacy root is never touched;
/// only the receipt-scoped canonical destination is removed.
pub fn execute_application_home_rollback(plan: &AppHomeRollbackPlan) -> Result<()> {
    if !plan.legacy_root_retained || !plan.source.is_dir() {
        bail!("rollback source is not available");
    }
    if !plan.receipt_path.is_file() {
        bail!("rollback receipt is missing");
    }
    if plan.destination == plan.source || !plan.destination.is_dir() {
        bail!("rollback destination is invalid");
    }
    fs::remove_dir_all(&plan.destination).with_context(|| {
        format!(
            "could not remove migrated root {}",
            plan.destination.display()
        )
    })?;
    Ok(())
}

pub fn rollback_request_path(canonical_root: &Path) -> PathBuf {
    canonical_root
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".{PRODUCT_SLUG}-rollback-request.json"))
}

pub fn queue_application_home_rollback(
    plan: &AppHomeRollbackPlan,
) -> Result<AppHomeRollbackRequest> {
    let path = rollback_request_path(&plan.destination);
    if path.is_file() {
        bail!("a rollback is already queued");
    }
    let request = AppHomeRollbackRequest {
        schema_version: 1,
        source: plan.source.clone(),
        destination: plan.destination.clone(),
        receipt_path: plan.receipt_path.clone(),
        requested_unix_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    write_json_atomic(&path, &request)?;
    Ok(request)
}

pub fn load_rollback_request() -> Result<Option<AppHomeRollbackRequest>> {
    load_rollback_request_for_root(&AppPaths::canonical_default_root())
}

pub fn load_rollback_request_for_root(
    canonical_root: &Path,
) -> Result<Option<AppHomeRollbackRequest>> {
    let path = rollback_request_path(canonical_root);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_reader(fs::File::open(path)?)?))
}

pub fn execute_queued_rollback(request: &AppHomeRollbackRequest) -> Result<()> {
    let plan = AppHomeRollbackPlan {
        schema_version: request.schema_version,
        source: request.source.clone(),
        destination: request.destination.clone(),
        receipt_path: request.receipt_path.clone(),
        legacy_root_retained: true,
    };
    execute_application_home_rollback(&plan)?;
    let _ = fs::remove_file(rollback_request_path(&request.destination));
    Ok(())
}

pub fn plan_application_home_cleanup(
    canonical_root: &Path,
    legacy_root: &Path,
) -> Result<AppHomeCleanupPlan> {
    let rollback = plan_application_home_rollback(canonical_root, legacy_root)?;
    Ok(AppHomeCleanupPlan {
        schema_version: 1,
        legacy_root: rollback.source,
        canonical_root: rollback.destination,
        receipt_path: rollback.receipt_path,
    })
}

pub fn execute_application_home_cleanup(plan: &AppHomeCleanupPlan) -> Result<()> {
    if !plan.canonical_root.is_dir() || !plan.receipt_path.is_file() {
        bail!("cleanup requires a verified canonical migration receipt");
    }
    if !plan.legacy_root.is_dir() || plan.legacy_root == plan.canonical_root {
        bail!("cleanup legacy root is invalid");
    }
    fs::remove_dir_all(&plan.legacy_root)
        .with_context(|| format!("could not clean legacy root {}", plan.legacy_root.display()))?;
    Ok(())
}

pub fn cleanup_request_path(canonical_root: &Path) -> PathBuf {
    canonical_root
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(".{PRODUCT_SLUG}-cleanup-request.json"))
}

pub fn queue_application_home_cleanup(plan: &AppHomeCleanupPlan) -> Result<AppHomeCleanupPlan> {
    let path = cleanup_request_path(&plan.canonical_root);
    if path.is_file() {
        bail!("legacy-root cleanup is already queued");
    }
    write_json_atomic(&path, plan)?;
    Ok(plan.clone())
}

pub fn load_cleanup_request() -> Result<Option<AppHomeCleanupPlan>> {
    load_cleanup_request_for_root(&AppPaths::canonical_default_root())
}

pub fn load_cleanup_request_for_root(canonical_root: &Path) -> Result<Option<AppHomeCleanupPlan>> {
    let path = cleanup_request_path(canonical_root);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_reader(fs::File::open(path)?)?))
}

pub fn execute_queued_cleanup(plan: &AppHomeCleanupPlan) -> Result<()> {
    execute_application_home_cleanup(plan)?;
    let _ = fs::remove_file(cleanup_request_path(&plan.canonical_root));
    Ok(())
}

fn load_or_create_journal(
    plan: &AppHomeMigrationPlan,
    journal_path: &Path,
) -> Result<AppHomeMigrationJournal> {
    if journal_path.is_file() {
        let journal: AppHomeMigrationJournal =
            serde_json::from_reader(fs::File::open(journal_path)?)?;
        if journal.plan_id != plan.plan_id {
            bail!("migration journal plan does not match the requested preview");
        }
        return Ok(journal);
    }
    Ok(AppHomeMigrationJournal {
        schema_version: 1,
        plan_id: plan.plan_id.clone(),
        state: MigrationJournalState::Planned,
        completed_entries: Vec::new(),
        last_error: None,
    })
}

fn copy_file_atomic(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("destination has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = destination.with_extension("local-llm-foundry-part");
    if destination.exists() {
        bail!(
            "refusing to overwrite migration destination {}",
            destination.display()
        );
    }
    fs::copy(source, &temporary).with_context(|| format!("copying {}", source.display()))?;
    fs::rename(&temporary, destination)
        .with_context(|| format!("promoting migration file {}", destination.display()))?;
    Ok(())
}

fn copy_sqlite_with_backup(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow::anyhow!("database destination has no parent"))?;
    fs::create_dir_all(parent)?;
    if destination.exists() {
        bail!(
            "refusing to overwrite database destination {}",
            destination.display()
        );
    }
    let storage = crate::chat_storage::ChatStorage::open(&source.to_path_buf())
        .with_context(|| format!("opening SQLite source {}", source.display()))?;
    let temporary = destination.with_extension("local-llm-foundry-part");
    storage.backup(&temporary)?;
    fs::rename(&temporary, destination)?;
    Ok(())
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("JSON path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.local-llm-foundry-part");
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Build a copy-first, no-overwrite application-root plan. Model trees,
/// runtime downloads, logs, and staging are inventoried but retained at the
/// legacy root for their explicit follow-up decisions.
pub fn plan_application_home(source: &Path, destination: &Path) -> Result<AppHomeMigrationPlan> {
    validate_root(source, "source")?;
    if source == destination {
        bail!("source and destination application roots must differ");
    }
    if destination.exists()
        && (!destination.is_dir() || fs::read_dir(destination)?.next().is_some())
    {
        bail!(
            "destination exists and is not an empty directory: {}",
            destination.display()
        );
    }
    let mut entries = Vec::new();
    let mut retained_entries = Vec::new();
    collect_entries(source, source, &mut entries, &mut retained_entries)?;
    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    retained_entries.sort();
    let required_copy_bytes = entries
        .iter()
        .filter(|entry| {
            !matches!(
                entry.class,
                ResourceClass::ModelRetained | ResourceClass::Recreatable
            )
        })
        .map(|entry| entry.bytes)
        .sum();
    let total_seen_bytes = entries.iter().map(|entry| entry.bytes).sum();
    let digest = Sha256::digest(serde_json::to_vec(&(
        1u32,
        source,
        destination,
        &entries,
        &retained_entries,
    ))?);
    let plan_id = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(AppHomeMigrationPlan {
        schema_version: 1,
        plan_id,
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
        entries,
        retained_entries,
        required_copy_bytes,
        total_seen_bytes,
    })
}

fn validate_root(root: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(root)
        .with_context(|| format!("{label} root is not readable: {}", root.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        bail!("{label} root must be a real directory: {}", root.display());
    }
    Ok(())
}

fn collect_entries(
    root: &Path,
    current: &Path,
    entries: &mut Vec<AppHomeMigrationEntry>,
    retained_entries: &mut Vec<PathBuf>,
) -> Result<()> {
    for item in fs::read_dir(current)? {
        let item = item?;
        let path = item.path();
        let relative = path
            .strip_prefix(root)
            .context("migration inventory path escaped source root")?
            .to_path_buf();
        if relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::RootDir
            )
        }) {
            bail!("migration inventory produced an unsafe relative path");
        }
        let class = classify_resource(&relative);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            // Symlinks in Retained resources (models) and Recreatable resources
            // (rapid-mlx overlays, runtime venvs) are allowed. Only reject them
            // in Critical/Unknown resources where they could indicate tampering.
            if class == ResourceClass::Critical || class == ResourceClass::Unknown {
                bail!("migration refuses symlinked resource: {}", path.display());
            }
        }
        let kind = if metadata.is_dir() {
            EntryKind::Directory
        } else if metadata.is_file() {
            EntryKind::File
        } else {
            bail!(
                "migration refuses special filesystem entry: {}",
                path.display()
            );
        };
        let bytes = if metadata.is_file() {
            metadata.len()
        } else {
            0
        };
        let modified_unix_seconds = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_secs());
        entries.push(AppHomeMigrationEntry {
            relative_path: relative.clone(),
            class,
            kind,
            bytes,
            modified_unix_seconds,
        });
        if class == ResourceClass::ModelRetained || class == ResourceClass::Recreatable {
            retained_entries.push(relative);
        }
        if metadata.is_dir() {
            collect_entries(root, &path, entries, retained_entries)?;
        }
    }
    Ok(())
}

fn classify_resource(relative: &Path) -> ResourceClass {
    let components: Vec<_> = relative.components().collect();
    let first = components
        .first()
        .and_then(|component| match component {
            std::path::Component::Normal(name) => name.to_str(),
            _ => None,
        });
    match first {
        Some("models") => ResourceClass::ModelRetained,
        Some("logs" | "bin" | "binaries" | "runtimes" | "model-cache" | ".staging") => {
            ResourceClass::Recreatable
        }
        // Rapid-MLX runtime resources are generated and recreatable.
        // Template overlays contain symlinks to source templates.
        Some("rapid-mlx") => ResourceClass::Recreatable,
        Some("encryption-key" | "api-token" | "db-admin-token" | "chat.db" | "certs") => {
            ResourceClass::Critical
        }
        Some(_) => ResourceClass::Unknown,
        None => ResourceClass::Unknown,
    }
}

pub fn inspect_default_roots() -> Result<RootInspection> {
    let roots = default_roots();
    inspect_roots(&roots.canonical, &roots.legacy, false)
}

pub fn inspect_application_roots(roots: &AppHomeRoots) -> Result<RootInspection> {
    let mut inspection = inspect_roots(&roots.canonical, &roots.legacy, false)?;
    if inspection.state == RootState::LegacyActive
        && migration_request_path_for_parent(
            roots.canonical.parent().unwrap_or_else(|| Path::new(".")),
        )
        .is_file()
    {
        inspection.state = RootState::MigrationQueued;
    }
    Ok(inspection)
}

pub fn inspect_roots(
    canonical_root: &Path,
    legacy_root: &Path,
    custom_config: bool,
) -> Result<RootInspection> {
    let canonical_state = contains_app_state(canonical_root)?;
    let legacy_state = contains_app_state(legacy_root)?;
    let state = if custom_config {
        RootState::CustomConfig
    } else {
        match (canonical_state, legacy_state) {
            (false, false) => RootState::Fresh,
            (false, true) if load_migration_request()?.is_some() => RootState::MigrationQueued,
            (false, true) => RootState::LegacyActive,
            (true, false) => RootState::NewActive,
            // A completed receipt proves that the canonical root was copied
            // from the retained legacy root. That is rollback-available, not
            // an instruction to merge or delete either root.
            (true, true) if has_completed_receipt(canonical_root, legacy_root) => {
                RootState::RollbackAvailable
            }
            (true, true) => RootState::Conflict,
        }
    };
    let active_root = match state {
        RootState::LegacyActive | RootState::MigrationQueued => Some(legacy_root.to_path_buf()),
        RootState::NewActive | RootState::BothIdentical | RootState::RollbackAvailable => {
            Some(canonical_root.to_path_buf())
        }
        RootState::CustomConfig => Some(canonical_root.to_path_buf()),
        _ => None,
    };
    Ok(RootInspection {
        state,
        canonical_root: canonical_root.to_path_buf(),
        legacy_root: legacy_root.to_path_buf(),
        active_root,
    })
}

fn contains_app_state(root: &Path) -> Result<bool> {
    if !root.is_dir() {
        return Ok(false);
    }
    for marker in STATE_MARKERS {
        let path = root.join(marker);
        if fs::symlink_metadata(&path)
            .map(|metadata| !metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn has_completed_receipt(canonical_root: &Path, legacy_root: &Path) -> bool {
    let Some(parent) = canonical_root.parent() else {
        return false;
    };
    let Ok(entries) = fs::read_dir(parent) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&format!(".{PRODUCT_SLUG}-migration-")) || !name.ends_with(".json") {
            return false;
        }
        let Ok(file) = fs::File::open(entry.path()) else {
            return false;
        };
        let Ok(receipt) = serde_json::from_reader::<_, AppHomeMigrationReceipt>(file) else {
            return false;
        };
        receipt.destination == canonical_root && receipt.source == legacy_root
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "local-llm-foundry-root-state-{name}-{}",
            std::process::id()
        ));
        let canonical = root.join("canonical");
        let legacy = root.join("legacy");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&canonical).unwrap();
        fs::create_dir_all(&legacy).unwrap();
        (canonical, legacy)
    }

    #[test]
    fn classifies_fresh_and_legacy_only_without_writes() {
        let (canonical, legacy) = fixture("legacy");
        assert_eq!(
            inspect_roots(&canonical, &legacy, false).unwrap().state,
            RootState::Fresh
        );
        fs::write(legacy.join("presets.json"), b"{}").unwrap();
        let inspected = inspect_roots(&canonical, &legacy, false).unwrap();
        assert_eq!(inspected.state, RootState::LegacyActive);
        assert!(!canonical.join("presets.json").exists());
        let _ = fs::remove_dir_all(canonical.parent().unwrap());
    }

    #[test]
    fn both_roots_fail_closed_as_conflict() {
        let (canonical, legacy) = fixture("both");
        fs::write(canonical.join("presets.json"), b"{}").unwrap();
        fs::write(legacy.join("presets.json"), b"{}").unwrap();
        assert_eq!(
            inspect_roots(&canonical, &legacy, false).unwrap().state,
            RootState::Conflict
        );
        let _ = fs::remove_dir_all(canonical.parent().unwrap());
    }

    #[test]
    fn custom_config_is_authoritative() {
        let (canonical, legacy) = fixture("custom");
        fs::write(canonical.join("presets.json"), b"{}").unwrap();
        fs::write(legacy.join("presets.json"), b"{}").unwrap();
        let inspected = inspect_roots(&canonical, &legacy, true).unwrap();
        assert_eq!(inspected.state, RootState::CustomConfig);
        assert_eq!(inspected.active_root, Some(canonical));
        let _ = fs::remove_dir_all(legacy.parent().unwrap());
    }

    #[test]
    fn application_plan_retains_models_and_rejects_symlinks() {
        let (source, destination) = fixture("plan");
        fs::create_dir_all(source.join("models/gguf")).unwrap();
        fs::write(source.join("models/gguf/model.gguf"), b"model").unwrap();
        fs::write(source.join("presets.json"), b"{}").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        assert!(plan.required_copy_bytes > 0);
        assert!(
            plan.retained_entries
                .iter()
                .any(|path| path == Path::new("models"))
        );
        assert!(
            plan.entries
                .iter()
                .any(|entry| entry.relative_path == Path::new("presets.json"))
        );
        let _ = fs::remove_dir_all(source.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn application_plan_refuses_symlink_escape() {
        let (source, destination) = fixture("symlink");
        std::os::unix::fs::symlink("/tmp", source.join("escape")).unwrap();
        let error = plan_application_home(&source, &destination).unwrap_err();
        assert!(error.to_string().contains("symlink"));
        let _ = fs::remove_dir_all(source.parent().unwrap());
    }

    #[test]
    fn application_execution_is_restartable_and_copy_first() {
        let (source, destination) = fixture("execute");
        fs::write(source.join("presets.json"), b"{\"version\":1}").unwrap();
        fs::create_dir_all(source.join("models")).unwrap();
        fs::write(source.join("models/model.gguf"), b"model").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        let receipt = execute_application_home(&plan).unwrap();
        let replay = execute_application_home(&plan).unwrap();
        assert_eq!(replay.plan_id, receipt.plan_id);
        assert!(destination.join("presets.json").is_file());
        assert_eq!(
            fs::read_to_string(destination.join("presets.json")).unwrap(),
            "{\"version\":1}"
        );
        assert!(source.join("presets.json").is_file());
        assert!(migration_journal_path(&plan).is_file());
        assert!(migration_receipt_path(&plan).is_file());
        assert!(
            receipt
                .retained_entries
                .iter()
                .any(|path| path == Path::new("models"))
        );
        assert_eq!(
            inspect_roots(&destination, &source, false).unwrap().state,
            RootState::RollbackAvailable
        );
        let _ = fs::remove_dir_all(source.parent().unwrap());
        let _ = fs::remove_file(migration_journal_path(&plan));
        let _ = fs::remove_file(migration_receipt_path(&plan));
    }

    #[test]
    fn rollback_requires_receipt_and_preserves_legacy_source() {
        let (source, destination) = fixture("rollback");
        fs::write(source.join("presets.json"), b"legacy").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        execute_application_home(&plan).unwrap();
        let rollback = plan_application_home_rollback(&destination, &source).unwrap();
        execute_application_home_rollback(&rollback).unwrap();
        assert!(source.join("presets.json").is_file());
        assert!(!destination.exists());
        let _ = fs::remove_file(migration_journal_path(&plan));
        let _ = fs::remove_file(migration_receipt_path(&plan));
        let _ = fs::remove_dir_all(source.parent().unwrap());
    }

    #[test]
    fn cleanup_requires_verified_receipt_and_removes_only_legacy_root() {
        let (source, destination) = fixture("cleanup");
        fs::write(source.join("presets.json"), b"legacy").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        execute_application_home(&plan).unwrap();
        let cleanup = plan_application_home_cleanup(&destination, &source).unwrap();
        execute_application_home_cleanup(&cleanup).unwrap();
        assert!(!source.exists());
        assert!(destination.join("presets.json").is_file());
        let _ = fs::remove_file(migration_journal_path(&plan));
        let _ = fs::remove_file(migration_receipt_path(&plan));
        let _ = fs::remove_dir_all(destination.parent().unwrap());
    }

    #[test]
    fn application_execution_rejects_changed_source_after_preview() {
        let (source, destination) = fixture("stale");
        fs::write(source.join("presets.json"), b"original").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        fs::write(source.join("presets.json"), b"changed").unwrap();
        let error = execute_application_home(&plan).unwrap_err();
        assert!(error.to_string().contains("stale"));
        assert!(!destination.exists() || fs::read_dir(&destination).unwrap().next().is_none());
        let _ = fs::remove_dir_all(source.parent().unwrap());
    }

    #[test]
    fn public_error_codes_never_include_local_paths() {
        let stale =
            anyhow::anyhow!("migration preview is stale for /Users/example/.config/llama-monitor");
        assert_eq!(public_error_code(&stale), MigrationErrorCode::StalePlan);
        assert_eq!(
            serde_json::to_string(&public_error_code(&stale)).unwrap(),
            "\"stale_plan\""
        );
        let unsafe_entry = anyhow::anyhow!("migration refuses symlinked resource: /tmp/secret");
        assert_eq!(
            public_error_code(&unsafe_entry),
            MigrationErrorCode::UnsafeEntry
        );
    }

    #[test]
    fn journal_and_receipt_accept_older_payloads_with_missing_optional_fields() {
        let journal: AppHomeMigrationJournal =
            serde_json::from_str(r#"{"schema_version":1,"plan_id":"abc","state":"planned"}"#)
                .unwrap();
        assert!(journal.completed_entries.is_empty());
        assert!(journal.last_error.is_none());

        let receipt: AppHomeMigrationReceipt = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "plan_id": "abc",
            "source": "/tmp/legacy",
            "destination": "/tmp/new",
            "copied_entries": [],
            "verified_entries": 0
        }))
        .unwrap();
        assert!(receipt.retained_entries.is_empty());
    }

    #[test]
    fn active_migration_lock_blocks_a_second_executor() {
        let (source, destination) = fixture("lock");
        fs::write(source.join("presets.json"), b"lock").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        let lock_path = migration_lock_path(&plan);
        fs::write(&lock_path, b"pid=other").unwrap();
        let error = execute_application_home(&plan).unwrap_err();
        assert!(error.to_string().contains("already running"));
        assert!(lock_path.is_file());
        let _ = fs::remove_file(lock_path);
        let _ = fs::remove_dir_all(source.parent().unwrap());
    }

    #[test]
    fn free_space_preflight_fails_closed_when_requirement_exceeds_volume() {
        let (_source, destination) = fixture("free-space");
        let error = ensure_free_space(&destination, u64::MAX).unwrap_err();
        assert!(error.to_string().contains("insufficient free space"));
    }

    #[cfg(unix)]
    #[test]
    fn permission_denied_destination_fails_without_touching_legacy_source() {
        use std::os::unix::fs::PermissionsExt;
        let (source, destination) = fixture("permission");
        fs::write(source.join("presets.json"), b"legacy").unwrap();
        let plan = plan_application_home(&source, &destination).unwrap();
        let parent = destination.parent().unwrap();
        let original = fs::metadata(parent).unwrap().permissions().mode();
        let mut readonly = fs::metadata(parent).unwrap().permissions();
        readonly.set_mode(0o500);
        fs::set_permissions(parent, readonly).unwrap();
        let result = execute_application_home(&plan);
        let mut restored = fs::metadata(parent).unwrap().permissions();
        restored.set_mode(original);
        fs::set_permissions(parent, restored).unwrap();
        assert!(result.is_err());
        assert!(source.join("presets.json").is_file());
    }
}
