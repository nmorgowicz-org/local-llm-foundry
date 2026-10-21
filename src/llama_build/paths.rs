//! On-disk layout for builds, managed-path containment and job-record
//! retention (plan §4.2, §4.4).

use std::path::{Component, Path, PathBuf};

use crate::paths::AppPaths;

use super::Accelerator;

/// Completed `<id>.log` + `<id>.json` pairs kept (plan §4.4).
pub const KEEP_COMPLETED_JOBS: usize = 3;

/// All directories a build owns, resolved from the application root.
#[derive(Debug, Clone)]
pub struct BuildPaths {
    pub root: PathBuf,
    pub source: PathBuf,
    pub ccache: PathBuf,
    pub logs: PathBuf,
    pub toolchain_cache: PathBuf,
    pub prereqs_last: PathBuf,
    pub process_records: PathBuf,
    app_paths: AppPaths,
}

impl BuildPaths {
    pub fn new(app: &AppPaths) -> Self {
        let root = app.llama_build_dir();
        Self {
            source: app.llama_source_dir(),
            ccache: app.llama_build_ccache_dir(),
            logs: app.llama_build_logs_dir(),
            toolchain_cache: root.join("toolchain-cache.json"),
            prereqs_last: root.join("prereqs-last.json"),
            process_records: root.join("processes.json"),
            root,
            app_paths: app.clone(),
        }
    }

    pub fn build_tree(&self, accel: Accelerator) -> PathBuf {
        self.app_paths.llama_build_tree_dir(accel.as_str())
    }

    pub fn log_file(&self, job_id: &str) -> PathBuf {
        self.logs.join(format!("{job_id}.log"))
    }

    pub fn record_file(&self, job_id: &str) -> PathBuf {
        self.logs.join(format!("{job_id}.json"))
    }

    /// Every directory below the build root a "clean" may remove, in order of
    /// how cheaply they regenerate.
    pub fn cleanable_trees(&self) -> Vec<PathBuf> {
        [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu]
            .into_iter()
            .map(|a| self.build_tree(a))
            .collect()
    }

    pub fn app_paths(&self) -> &AppPaths {
        &self.app_paths
    }
}

/// `child` is a strict descendant of `base` after lexical normalisation.
/// Rejects `..` escapes and absolute re-rooting; never touches the disk.
pub fn is_strictly_below(base: &Path, child: &Path) -> bool {
    let norm = |p: &Path| -> Option<Vec<std::ffi::OsString>> {
        let mut out: Vec<std::ffi::OsString> = Vec::new();
        for comp in p.components() {
            match comp {
                Component::ParentDir => {
                    out.pop()?;
                }
                Component::CurDir => {}
                other => out.push(other.as_os_str().to_os_string()),
            }
        }
        Some(out)
    };
    match (norm(base), norm(child)) {
        (Some(b), Some(c)) => c.len() > b.len() && c[..b.len()] == b[..],
        _ => false,
    }
}

/// Create `dir` (and parents) and return its canonical form, checking that it
/// is still inside `base` once symlinks are resolved. A build directory reached
/// through a symlink (`/tmp` -> `/private/tmp`) breaks upstream's
/// `ui-assets.cmake`, so CMake is always given the canonical path.
pub fn create_canonical_within(base: &Path, dir: &Path) -> std::io::Result<PathBuf> {
    if !is_strictly_below(base, dir) && dir != base {
        return Err(std::io::Error::other(format!(
            "{} is outside the managed build directory {}",
            dir.display(),
            base.display()
        )));
    }
    std::fs::create_dir_all(base)?;
    let canonical_base = std::fs::canonicalize(base)?;
    let relative = dir.strip_prefix(base).map_err(std::io::Error::other)?;
    let mut canonical = canonical_base.clone();
    // Validate each existing ancestor before creating anything below it. In
    // particular, never create a child through an escaping symlink and only
    // discover the escape after the write.
    for component in relative.components() {
        match component {
            Component::CurDir => continue,
            Component::Normal(name) => canonical.push(name),
            _ => return Err(std::io::Error::other("invalid managed path component")),
        }
        match std::fs::symlink_metadata(&canonical) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir(&canonical)?;
            }
            Err(e) => return Err(e),
        }
        canonical = std::fs::canonicalize(&canonical)?;
        if canonical != canonical_base && !canonical.starts_with(&canonical_base) {
            return Err(std::io::Error::other(format!(
                "{} resolves outside the managed build directory",
                dir.display()
            )));
        }
    }
    Ok(canonical)
}

/// Windows verbatim prefixes (`\\?\C:\...`) confuse CMake; strip them.
pub fn without_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    match text.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => path.to_path_buf(),
    }
}

/// A job-record pair on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobRecordFiles {
    pub id: String,
    pub log: Option<PathBuf>,
    pub record: Option<PathBuf>,
    pub modified: std::time::SystemTime,
}

/// Group `<id>.log` / `<id>.json` files in `logs`, newest first.
pub fn list_job_records(logs: &Path) -> std::io::Result<Vec<JobRecordFiles>> {
    let mut by_id: std::collections::BTreeMap<String, JobRecordFiles> = Default::default();
    let entries = match std::fs::read_dir(logs) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let (Some(stem), Some(ext)) = (
            path.file_stem().and_then(|s| s.to_str()),
            path.extension().and_then(|s| s.to_str()),
        ) else {
            continue;
        };
        if ext != "log" && ext != "json" {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        let slot = by_id
            .entry(stem.to_string())
            .or_insert_with(|| JobRecordFiles {
                id: stem.to_string(),
                log: None,
                record: None,
                modified,
            });
        slot.modified = slot.modified.max(modified);
        if ext == "log" {
            slot.log = Some(path);
        } else {
            slot.record = Some(path);
        }
    }
    let mut out: Vec<_> = by_id.into_values().collect();
    out.sort_by(|a, b| b.modified.cmp(&a.modified).then(b.id.cmp(&a.id)));
    Ok(out)
}

/// Delete all but the newest [`KEEP_COMPLETED_JOBS`] pairs. `protected` ids
/// (the active job, anything a journal references) are never removed and do
/// not count against the limit. Returns the removed ids.
pub fn prune_job_records(
    logs: &Path,
    protected: &dyn Fn(&str) -> bool,
) -> std::io::Result<Vec<String>> {
    let mut removed = Vec::new();
    let mut kept = 0usize;
    for pair in list_job_records(logs)? {
        if protected(&pair.id) {
            continue;
        }
        if kept < KEEP_COMPLETED_JOBS {
            kept += 1;
            continue;
        }
        for file in [&pair.log, &pair.record].into_iter().flatten() {
            match std::fs::remove_file(file) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        removed.push(pair.id);
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn layout_matches_the_plan() {
        let app = AppPaths::from_root(PathBuf::from("/r"));
        let p = BuildPaths::new(&app);
        assert_eq!(p.root, PathBuf::from("/r/llama-build"));
        assert_eq!(p.source, PathBuf::from("/r/llama-build/src"));
        assert_eq!(
            p.build_tree(Accelerator::Cuda),
            PathBuf::from("/r/llama-build/build-cuda")
        );
        assert_eq!(p.ccache, PathBuf::from("/r/llama-build/ccache"));
        assert_eq!(
            p.toolchain_cache,
            PathBuf::from("/r/llama-build/toolchain-cache.json")
        );
        assert_eq!(
            p.log_file("j1"),
            PathBuf::from("/r/logs/llama-build/j1.log")
        );
        assert_eq!(
            p.record_file("j1"),
            PathBuf::from("/r/logs/llama-build/j1.json")
        );
        assert_eq!(p.cleanable_trees().len(), 3);
    }

    #[test]
    fn containment_rejects_escapes_and_equal_paths() {
        let base = Path::new("/r/llama-build");
        assert!(is_strictly_below(base, Path::new("/r/llama-build/src")));
        assert!(is_strictly_below(base, Path::new("/r/llama-build/a/./b")));
        assert!(!is_strictly_below(base, Path::new("/r/llama-build")));
        assert!(!is_strictly_below(
            base,
            Path::new("/r/llama-build/../bin/llama-server")
        ));
        assert!(!is_strictly_below(base, Path::new("/r/llama-build-evil/x")));
        assert!(!is_strictly_below(base, Path::new("/etc/passwd")));
        assert!(!is_strictly_below(base, Path::new("../../etc")));
    }

    #[test]
    fn canonical_dir_is_created_and_canonical() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("llama-build");
        let dir = base.join("build-metal");
        let got = create_canonical_within(&base, &dir).unwrap();
        assert!(got.is_dir());
        assert_eq!(got, std::fs::canonicalize(&dir).unwrap());
        // Idempotent.
        assert_eq!(create_canonical_within(&base, &dir).unwrap(), got);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_parent_is_canonicalized_for_cmake() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let base = link.join("llama-build");
        let got = create_canonical_within(&base, &base.join("build-cpu")).unwrap();
        assert!(!got.starts_with(&link));
        assert!(got.starts_with(std::fs::canonicalize(&real).unwrap()));
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_that_escapes_the_base_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("llama-build");
        std::fs::create_dir_all(&base).unwrap();
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, base.join("build-cuda")).unwrap();
        let err = create_canonical_within(&base, &base.join("build-cuda")).unwrap_err();
        assert!(err.to_string().contains("outside"), "{err}");
    }

    #[test]
    fn paths_outside_the_base_are_refused_before_creation() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("llama-build");
        let err = create_canonical_within(&base, &tmp.path().join("elsewhere")).unwrap_err();
        assert!(err.to_string().contains("outside"));
        assert!(!tmp.path().join("elsewhere").exists());
    }

    #[test]
    fn verbatim_prefixes_are_stripped() {
        assert_eq!(
            without_verbatim_prefix(Path::new(r"\\?\C:\a\b")),
            PathBuf::from(r"C:\a\b")
        );
        assert_eq!(
            without_verbatim_prefix(Path::new(r"\\?\UNC\srv\share\x")),
            PathBuf::from(r"\\srv\share\x")
        );
        assert_eq!(
            without_verbatim_prefix(Path::new("/plain/path")),
            PathBuf::from("/plain/path")
        );
    }

    fn touch(dir: &Path, name: &str, age_secs: u64) {
        let path = dir.join(name);
        std::fs::write(&path, b"x").unwrap();
        let when = SystemTime::now() - Duration::from_secs(age_secs);
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(when).unwrap();
    }

    #[test]
    fn keeps_the_newest_three_pairs_and_removes_both_files() {
        let tmp = tempfile::tempdir().unwrap();
        let logs = tmp.path();
        for (i, id) in ["a", "b", "c", "d", "e"].iter().enumerate() {
            let age = 1000 - (i as u64) * 100; // a oldest ... e newest
            touch(logs, &format!("{id}.log"), age);
            touch(logs, &format!("{id}.json"), age);
        }
        let removed = prune_job_records(logs, &|_| false).unwrap();
        assert_eq!(removed, vec!["b".to_string(), "a".to_string()]);
        for id in ["c", "d", "e"] {
            assert!(logs.join(format!("{id}.log")).exists());
            assert!(logs.join(format!("{id}.json")).exists());
        }
        for id in ["a", "b"] {
            assert!(!logs.join(format!("{id}.log")).exists());
            assert!(!logs.join(format!("{id}.json")).exists());
        }
    }

    #[test]
    fn protected_records_are_exempt_and_do_not_use_up_the_quota() {
        let tmp = tempfile::tempdir().unwrap();
        let logs = tmp.path();
        // `active` is the oldest, `journal` second oldest; both protected.
        touch(logs, "active.log", 5000);
        touch(logs, "journal.json", 4000);
        for (i, id) in ["a", "b", "c", "d"].iter().enumerate() {
            touch(logs, &format!("{id}.log"), 1000 - (i as u64) * 100);
        }
        let removed = prune_job_records(logs, &|id| id == "active" || id == "journal").unwrap();
        assert_eq!(removed, vec!["a".to_string()]);
        assert!(logs.join("active.log").exists());
        assert!(logs.join("journal.json").exists());
        assert!(logs.join("d.log").exists());
    }

    #[test]
    fn a_missing_logs_dir_is_an_empty_listing_and_unrelated_files_survive() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(
            list_job_records(&tmp.path().join("nope"))
                .unwrap()
                .is_empty()
        );
        touch(tmp.path(), "notes.txt", 9999);
        let removed = prune_job_records(tmp.path(), &|_| false).unwrap();
        assert!(removed.is_empty());
        assert!(tmp.path().join("notes.txt").exists());
    }
}

#[cfg(all(test, unix))]
mod ancestor_tests {
    #[test]
    fn escaping_ancestor_never_creates_outside_child() {
        let base = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), base.path().join("escape")).unwrap();
        assert!(
            super::create_canonical_within(base.path(), &base.path().join("escape/must-not-exist"))
                .is_err()
        );
        assert!(!outside.path().join("must-not-exist").exists());
    }
}
