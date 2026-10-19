//! Managed caches and ownership (plan §4.3.1, §4.4 *Managed caches*).
//!
//! - **Cap:** the managed ccache has a finite cap, default 2 GiB, validated to
//!   256 MiB–16 GiB.
//! - **Clean:** removes inactive build trees and the managed ccache (and,
//!   optionally, the source checkout). It never touches the live or previous
//!   runtime sets, validation files or an active job's files, and it refuses to
//!   run while anything that owns those directories is active.
//! - **Ownership:** one table decides which operations may overlap.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::Accelerator;
use super::paths::{BuildPaths, is_strictly_below};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

pub const MIN_CCACHE_BYTES: u64 = 256 * MIB;
pub const DEFAULT_CCACHE_BYTES: u64 = 2 * GIB;
pub const MAX_CCACHE_BYTES: u64 = 16 * GIB;

/// `AppConfig.llama_build.ccache_max_bytes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaBuildConfig {
    #[serde(default = "default_ccache")]
    pub ccache_max_bytes: u64,
}

fn default_ccache() -> u64 {
    DEFAULT_CCACHE_BYTES
}

impl Default for LlamaBuildConfig {
    fn default() -> Self {
        Self {
            ccache_max_bytes: DEFAULT_CCACHE_BYTES,
        }
    }
}

impl LlamaBuildConfig {
    /// Read only the selected application's configuration; never fall back to
    /// a legacy/user root for build options.
    pub fn load(config_dir: &Path) -> std::io::Result<Self> {
        let file = config_dir.join("llama-build.json");
        let config: Self = match std::fs::read(file) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(std::io::Error::other)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => return Err(e),
        };
        validate_ccache_cap(config.ccache_max_bytes).map_err(std::io::Error::other)?;
        Ok(config)
    }

    pub fn store(&self, config_dir: &Path) -> std::io::Result<()> {
        use std::io::Write;
        validate_ccache_cap(self.ccache_max_bytes).map_err(std::io::Error::other)?;
        std::fs::create_dir_all(config_dir)?;
        let mut file = tempfile::NamedTempFile::new_in(config_dir)?;
        file.write_all(&serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?)?;
        file.as_file().sync_all()?;
        file.persist(config_dir.join("llama-build.json"))
            .map_err(|e| e.error)?;
        Ok(())
    }
}

pub fn validate_ccache_cap(bytes: u64) -> Result<u64, String> {
    if !(MIN_CCACHE_BYTES..=MAX_CCACHE_BYTES).contains(&bytes) {
        return Err(format!(
            "the ccache size must be between {} MiB and {} GiB",
            MIN_CCACHE_BYTES / MIB,
            MAX_CCACHE_BYTES / GIB
        ));
    }
    Ok(bytes)
}

/// Total size of a directory tree. Symlinks are counted, never followed.
pub fn dir_size(path: &Path) -> u64 {
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
        return 0;
    }
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Ownership
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    /// Owns the checkout, build trees and ccache.
    Build,
    /// Removes build trees, ccache and optionally the checkout.
    Clean,
    /// The compile smoke test.
    Validation,
    /// A read-only prerequisite check.
    Check,
    /// `brew install`: changes the toolchain itself.
    Install,
}

impl Owner {
    pub fn label(self) -> &'static str {
        match self {
            Self::Build => "a build",
            Self::Clean => "a cache clean",
            Self::Validation => "prerequisite validation",
            Self::Check => "a prerequisite check",
            Self::Install => "a tool install",
        }
    }
}

/// Whether `a` and `b` may hold ownership at the same time.
///
/// - a build owns the directories a clean removes: they never overlap, nor do
///   two builds;
/// - validation shares the ccache a clean empties;
/// - an install changes the toolchain under every other operation;
/// - a read-only check overlaps with everything except an install.
pub fn may_overlap(a: Owner, b: Owner) -> bool {
    use Owner::*;
    let conflicts = |x: Owner, y: Owner| {
        matches!(
            (x, y),
            (Install, _)
                | (Build, Build | Clean)
                | (Clean, Clean | Validation)
                | (Validation, Validation)
        )
    };
    !(conflicts(a, b) || conflicts(b, a))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    pub holder: Owner,
    pub holder_id: String,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} is running ({})", self.holder.label(), self.holder_id)
    }
}

#[derive(Default)]
pub struct Ownership {
    held: std::sync::Arc<Mutex<Vec<(Owner, String)>>>,
}

#[derive(Clone)]
pub struct Lease {
    inner: std::sync::Arc<LeaseInner>,
}

struct LeaseInner {
    table: std::sync::Arc<Mutex<Vec<(Owner, String)>>>,
    owner: Owner,
    id: String,
}

impl Lease {
    pub fn owner(&self) -> Owner {
        self.inner.owner
    }

    pub fn id(&self) -> &str {
        &self.inner.id
    }
}

impl Drop for LeaseInner {
    fn drop(&mut self) {
        let mut held = self.table.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = held
            .iter()
            .position(|(o, id)| *o == self.owner && *id == self.id)
        {
            held.remove(at);
        }
    }
}

impl Ownership {
    pub fn try_acquire(&self, owner: Owner, id: &str) -> Result<Lease, Conflict> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((holder, holder_id)) = held.iter().find(|(h, _)| !may_overlap(*h, owner)) {
            return Err(Conflict {
                holder: *holder,
                holder_id: holder_id.clone(),
            });
        }
        held.push((owner, id.to_string()));
        Ok(Lease {
            inner: std::sync::Arc::new(LeaseInner {
                table: self.held.clone(),
                owner,
                id: id.to_string(),
            }),
        })
    }

    pub fn holders(&self) -> Vec<(Owner, String)> {
        self.held.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn is_held(&self, owner: Owner) -> bool {
        self.holders().iter().any(|(o, _)| *o == owner)
    }
}

// ---------------------------------------------------------------------------
// Clean
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CleanOptions {
    /// Also remove the git checkout (it is re-downloaded on the next build).
    #[serde(default)]
    pub include_source: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct CleanReport {
    pub removed: Vec<String>,
    pub freed_bytes: u64,
    pub kept: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CleanError {
    Busy(Conflict),
    Io(String),
}

impl std::fmt::Display for CleanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(c) => write!(f, "cannot clean while {c}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Everything a clean is allowed to remove, never anything else.
pub fn cleanable(paths: &BuildPaths, options: CleanOptions) -> Vec<PathBuf> {
    let mut out = paths.cleanable_trees();
    out.push(paths.ccache.clone());
    if options.include_source {
        out.push(paths.source.clone());
    }
    out
}

/// Remove inactive build caches. The caller must hold [`Owner::Clean`]; this
/// re-checks that no build or validation is active as a second guard.
pub fn clean(
    paths: &BuildPaths,
    options: CleanOptions,
    ownership: &Ownership,
) -> Result<CleanReport, CleanError> {
    match std::fs::symlink_metadata(&paths.process_records) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => {
            return Err(CleanError::Io(
                "process containment is unresolved; repair is required before cleaning caches"
                    .into(),
            ));
        }
    }
    for other in [Owner::Build, Owner::Validation, Owner::Install] {
        if let Some((holder, id)) = ownership.holders().into_iter().find(|(h, _)| *h == other) {
            return Err(CleanError::Busy(Conflict {
                holder,
                holder_id: id,
            }));
        }
    }
    let mut report = CleanReport::default();
    for target in cleanable(paths, options) {
        // Defence in depth: never delete outside the managed build directory,
        // and never the directory itself.
        if !is_strictly_below(&paths.root, &target) {
            return Err(CleanError::Io(format!(
                "refusing to remove {}: not below {}",
                target.display(),
                paths.root.display()
            )));
        }
        if std::fs::symlink_metadata(&target).is_err() {
            continue;
        }
        let size = dir_size(&target);
        let meta = std::fs::symlink_metadata(&target).map_err(|e| CleanError::Io(e.to_string()))?;
        let result = if meta.file_type().is_symlink() || meta.is_file() {
            std::fs::remove_file(&target)
        } else {
            std::fs::remove_dir_all(&target)
        };
        result
            .map_err(|e| CleanError::Io(format!("could not remove {}: {e}", target.display())))?;
        report.freed_bytes += size;
        report.removed.push(target.to_string_lossy().into_owned());
    }
    if !options.include_source && paths.source.exists() {
        report
            .kept
            .push(paths.source.to_string_lossy().into_owned());
    }
    Ok(report)
}

/// Sizes the panel shows, reported separately (plan §4.4).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct CacheSizes {
    pub source_bytes: u64,
    pub ccache_bytes: u64,
    pub ccache_cap_bytes: u64,
    pub build_trees: Vec<(Accelerator, u64)>,
}

pub fn cache_sizes(paths: &BuildPaths, cap: u64) -> CacheSizes {
    CacheSizes {
        source_bytes: dir_size(&paths.source),
        ccache_bytes: dir_size(&paths.ccache),
        ccache_cap_bytes: cap,
        build_trees: [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu]
            .into_iter()
            .map(|a| (a, dir_size(&paths.build_tree(a))))
            .filter(|(_, size)| *size > 0)
            .collect(),
    }
}

/// `CCACHE_MAXSIZE` is applied through the child environment; this also trims
/// an oversized cache after a job so a lowered cap takes effect.
pub fn ccache_needs_trim(current: u64, cap: u64) -> bool {
    current > cap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::AppPaths;

    fn paths(root: &Path) -> BuildPaths {
        BuildPaths::new(&AppPaths::from_root(root.to_path_buf()))
    }

    fn fill(dir: &Path, name: &str, bytes: usize) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn the_ccache_cap_is_validated_and_defaults_to_2_gib() {
        assert_eq!(LlamaBuildConfig::default().ccache_max_bytes, 2 * GIB);
        assert_eq!(validate_ccache_cap(2 * GIB), Ok(2 * GIB));
        assert!(validate_ccache_cap(MIN_CCACHE_BYTES).is_ok());
        assert!(validate_ccache_cap(MAX_CCACHE_BYTES).is_ok());
        for bad in [0, MIN_CCACHE_BYTES - 1, MAX_CCACHE_BYTES + 1, u64::MAX] {
            assert!(validate_ccache_cap(bad).is_err(), "{bad}");
        }
        let cfg: LlamaBuildConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.ccache_max_bytes, DEFAULT_CCACHE_BYTES);
    }

    #[test]
    fn trim_is_needed_only_above_the_cap() {
        assert!(!ccache_needs_trim(GIB, 2 * GIB));
        assert!(ccache_needs_trim(3 * GIB, 2 * GIB));
    }

    /// The full ownership matrix, written out so a change has to be deliberate.
    #[test]
    fn ownership_matrix() {
        use Owner::*;
        let expect = [
            // (a, b, may_overlap)
            (Build, Build, false),
            (Build, Clean, false),
            (Build, Validation, true),
            (Build, Check, true),
            (Build, Install, false),
            (Clean, Clean, false),
            (Clean, Validation, false),
            (Clean, Check, true),
            (Clean, Install, false),
            (Validation, Validation, false),
            (Validation, Check, true),
            (Validation, Install, false),
            (Check, Check, true),
            (Check, Install, false),
            (Install, Install, false),
        ];
        for (a, b, want) in expect {
            assert_eq!(may_overlap(a, b), want, "{a:?} vs {b:?}");
            assert_eq!(may_overlap(b, a), want, "{b:?} vs {a:?} (symmetry)");
        }
    }

    #[test]
    fn leases_conflict_release_and_name_the_holder() {
        let table = Ownership::default();
        let build = table.try_acquire(Owner::Build, "job-1").unwrap();
        let busy = table.try_acquire(Owner::Build, "job-2").err().unwrap();
        assert_eq!(busy.holder, Owner::Build);
        assert_eq!(busy.holder_id, "job-1");
        assert!(busy.to_string().contains("job-1"));
        assert!(table.try_acquire(Owner::Clean, "c").is_err());
        assert!(table.try_acquire(Owner::Install, "i").is_err());
        // Compatible operations still run.
        let check = table.try_acquire(Owner::Check, "k").unwrap();
        drop(build);
        let clean = table.try_acquire(Owner::Clean, "c").unwrap();
        assert!(table.try_acquire(Owner::Build, "job-3").is_err());
        assert!(table.try_acquire(Owner::Validation, "v").is_err());
        drop(clean);
        drop(check);
        assert!(table.holders().is_empty());
    }

    #[test]
    fn an_install_blocks_everything_and_is_blocked_by_everything() {
        let table = Ownership::default();
        let check = table.try_acquire(Owner::Check, "k").unwrap();
        assert!(table.try_acquire(Owner::Install, "i").is_err());
        drop(check);
        let install = table.try_acquire(Owner::Install, "i").unwrap();
        for o in [Owner::Build, Owner::Clean, Owner::Validation, Owner::Check] {
            assert!(table.try_acquire(o, "x").is_err(), "{o:?}");
        }
        drop(install);
        assert!(table.try_acquire(Owner::Build, "b").is_ok());
    }

    #[test]
    fn clean_removes_build_trees_and_ccache_but_never_the_runtime_or_validation_files() {
        let tmp = tempfile::tempdir().unwrap();
        let p = paths(tmp.path());
        fill(&p.build_tree(Accelerator::Metal), "a.o", 1000);
        fill(&p.build_tree(Accelerator::Cuda), "b.o", 2000);
        fill(&p.ccache, "c", 500);
        fill(&p.source, "README", 100);
        std::fs::write(&p.prereqs_last, b"{}").unwrap();
        std::fs::write(&p.toolchain_cache, b"{}").unwrap();
        // The live and previous runtime sets and an active job's log.
        let bin = tmp.path().join("bin");
        let prev = tmp.path().join("bin-previous");
        fill(&bin, "llama-server", 10);
        fill(&prev, "llama-server", 10);
        fill(&p.logs, "job.log", 10);

        let ownership = Ownership::default();
        let report = clean(&p, CleanOptions::default(), &ownership).unwrap();
        assert_eq!(report.freed_bytes, 3500);
        assert!(!p.build_tree(Accelerator::Metal).exists());
        assert!(!p.build_tree(Accelerator::Cuda).exists());
        assert!(!p.ccache.exists());
        assert!(
            p.source.join("README").exists(),
            "the checkout is kept by default"
        );
        assert_eq!(report.kept.len(), 1);
        for survivor in [
            &p.prereqs_last,
            &p.toolchain_cache,
            &bin.join("llama-server"),
            &prev.join("llama-server"),
            &p.logs.join("job.log"),
        ] {
            assert!(survivor.exists(), "{} was removed", survivor.display());
        }
    }

    #[test]
    fn clean_can_include_the_checkout_on_request() {
        let tmp = tempfile::tempdir().unwrap();
        let p = paths(tmp.path());
        fill(&p.source, "README", 100);
        let report = clean(
            &p,
            CleanOptions {
                include_source: true,
            },
            &Ownership::default(),
        )
        .unwrap();
        assert!(!p.source.exists());
        assert_eq!(report.freed_bytes, 100);
        assert!(report.kept.is_empty());
    }

    #[test]
    fn clean_refuses_while_a_build_or_validation_or_install_is_active() {
        let tmp = tempfile::tempdir().unwrap();
        let p = paths(tmp.path());
        fill(&p.build_tree(Accelerator::Metal), "a.o", 10);
        let table = Ownership::default();
        let build = table.try_acquire(Owner::Build, "job-1").unwrap();
        let err = clean(&p, CleanOptions::default(), &table).unwrap_err();
        assert!(matches!(&err, CleanError::Busy(c) if c.holder_id == "job-1"));
        assert!(
            p.build_tree(Accelerator::Metal).exists(),
            "nothing was removed"
        );
        drop(build);
        let validation = table.try_acquire(Owner::Validation, "v").unwrap();
        assert!(clean(&p, CleanOptions::default(), &table).is_err());
        drop(validation);
        assert!(clean(&p, CleanOptions::default(), &table).is_ok());
    }

    #[test]
    fn clean_on_an_empty_root_is_a_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        let report = clean(
            &paths(tmp.path()),
            CleanOptions::default(),
            &Ownership::default(),
        )
        .unwrap();
        assert!(report.removed.is_empty());
        assert_eq!(report.freed_bytes, 0);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_inside_a_build_tree_is_removed_not_followed() {
        let tmp = tempfile::tempdir().unwrap();
        let p = paths(tmp.path());
        let precious = tmp.path().join("precious");
        fill(&precious, "model.gguf", 777);
        let tree = p.build_tree(Accelerator::Cpu);
        std::fs::create_dir_all(&tree).unwrap();
        std::os::unix::fs::symlink(&precious, tree.join("link")).unwrap();
        // The build directory itself replaced by a symlink to elsewhere.
        clean(&p, CleanOptions::default(), &Ownership::default()).unwrap();
        assert!(precious.join("model.gguf").exists());
        let cuda = p.build_tree(Accelerator::Cuda);
        std::os::unix::fs::symlink(&precious, &cuda).unwrap();
        clean(&p, CleanOptions::default(), &Ownership::default()).unwrap();
        assert!(
            precious.join("model.gguf").exists(),
            "a symlinked tree must be unlinked, not emptied"
        );
        assert!(std::fs::symlink_metadata(&cuda).is_err());
    }

    #[test]
    fn cleanable_set_is_exactly_trees_ccache_and_optionally_source() {
        let tmp = tempfile::tempdir().unwrap();
        let p = paths(tmp.path());
        let base = cleanable(&p, CleanOptions::default());
        assert_eq!(base.len(), 4);
        assert!(!base.contains(&p.source));
        assert!(
            cleanable(
                &p,
                CleanOptions {
                    include_source: true
                }
            )
            .contains(&p.source)
        );
        for path in cleanable(
            &p,
            CleanOptions {
                include_source: true,
            },
        ) {
            assert!(is_strictly_below(&p.root, &path));
            assert!(!path.starts_with(tmp.path().join("bin")));
        }
    }

    #[test]
    fn sizes_are_reported_separately() {
        let tmp = tempfile::tempdir().unwrap();
        let p = paths(tmp.path());
        fill(&p.source, "s", 100);
        fill(&p.ccache, "c", 200);
        fill(&p.build_tree(Accelerator::Metal), "m", 300);
        let sizes = cache_sizes(&p, 2 * GIB);
        assert_eq!(sizes.source_bytes, 100);
        assert_eq!(sizes.ccache_bytes, 200);
        assert_eq!(sizes.build_trees, vec![(Accelerator::Metal, 300)]);
        assert_eq!(sizes.ccache_cap_bytes, 2 * GIB);
    }
}

#[cfg(test)]
mod containment_tests {
    #[test]
    fn unresolved_records_prevent_cleaning_even_if_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let app = crate::paths::AppPaths::from_root(dir.path().to_path_buf());
        let paths = super::BuildPaths::new(&app);
        std::fs::create_dir_all(&paths.source).unwrap();
        std::fs::write(paths.source.join("keep"), "owned").unwrap();
        std::fs::write(&paths.process_records, "{broken").unwrap();
        assert!(
            super::clean(
                &paths,
                super::CleanOptions {
                    include_source: true
                },
                &super::Ownership::default()
            )
            .is_err()
        );
        assert!(paths.source.join("keep").exists());
    }
}

/// Standalone builds have no model lifecycle host. An exclusive OS lease
/// therefore excludes running app/agent processes (which hold shared leases)
/// and other standalone builders for the same managed root. Kernel release on
/// process death avoids stale PID lock files.
pub fn runtime_process_lease(
    root: &Path,
    standalone_build: bool,
) -> std::io::Result<std::fs::File> {
    std::fs::create_dir_all(root)?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(".runtime-owner.lock"))?;
    let result = if standalone_build {
        file.try_lock()
    } else {
        file.try_lock_shared()
    };
    result.map_err(|e| {
        std::io::Error::other(format!(
            "the managed runtime is in use by another foundry/build process: {e}"
        ))
    })?;
    Ok(file)
}

#[cfg(test)]
mod process_lease_tests {
    #[test]
    fn standalone_build_excludes_app_and_other_builders() {
        let root = tempfile::tempdir().unwrap();
        let app = super::runtime_process_lease(root.path(), false).unwrap();
        assert!(super::runtime_process_lease(root.path(), true).is_err());
        drop(app);
        let build = super::runtime_process_lease(root.path(), true).unwrap();
        assert!(super::runtime_process_lease(root.path(), false).is_err());
        assert!(super::runtime_process_lease(root.path(), true).is_err());
        drop(build);
        assert!(super::runtime_process_lease(root.path(), false).is_ok());
    }
}

/// Bound fetched dependency caches once the owning build has stopped writing.
/// Small caches are reused; an over-budget cache is discarded and its tree
/// marked for reconfiguration. Call only while holding checkout/build ownership.
pub fn prune_dependency_caches(paths: &BuildPaths, limit: u64) -> std::io::Result<Vec<PathBuf>> {
    if !super::job::containment_clear(paths) {
        return Err(std::io::Error::other(
            "dependency caches have unresolved writers",
        ));
    }
    let mut removed = Vec::new();
    for tree in paths.cleanable_trees() {
        let Ok(meta) = std::fs::symlink_metadata(&tree) else {
            continue;
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            continue;
        }
        let tree = super::paths::create_canonical_within(&paths.root, &tree)?;
        let deps = tree.join("_deps");
        let Ok(meta) = std::fs::symlink_metadata(&deps) else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() && dir_size(&deps) > limit {
            std::fs::remove_dir_all(&deps)?;
            std::fs::write(
                tree.join(super::job::NEEDS_RECONFIGURE_MARKER),
                "dependency cache pruned",
            )?;
            removed.push(deps);
        }
    }
    Ok(removed)
}

pub const DEPENDENCY_CACHE_LIMIT: u64 = 512 * MIB;

#[cfg(test)]
mod dependency_tests {
    #[test]
    fn dependency_retention_preserves_small_and_active_caches() {
        let root = tempfile::tempdir().unwrap();
        let paths = super::BuildPaths::new(&crate::paths::AppPaths::from_root(root.path().into()));
        let tree = paths.build_tree(crate::llama_build::Accelerator::Cpu);
        std::fs::create_dir_all(tree.join("_deps")).unwrap();
        std::fs::write(tree.join("_deps/download"), [0u8; 32]).unwrap();
        assert!(
            super::prune_dependency_caches(&paths, 64)
                .unwrap()
                .is_empty()
        );
        std::fs::write(&paths.process_records, "repair required").unwrap();
        assert!(super::prune_dependency_caches(&paths, 16).is_err());
        assert!(tree.join("_deps/download").exists());
        std::fs::remove_file(&paths.process_records).unwrap();
        assert_eq!(super::prune_dependency_caches(&paths, 16).unwrap().len(), 1);
        assert!(!tree.join("_deps").exists());
        assert!(
            tree.join(crate::llama_build::job::NEEDS_RECONFIGURE_MARKER)
                .exists()
        );
    }
}
