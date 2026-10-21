//! Collecting a build tree's outputs into a promotable staging directory
//! (plan §4.4 *stage*, §5.2).
//!
//! The authority on "what this configure produces" is the generator's own
//! target list (`ninja -t targets all`), not file timestamps: an incremental
//! build legitimately leaves untouched files with old mtimes, while a stale
//! executable or plugin from an earlier configure (a flipped
//! `GGML_BACKEND_DL`, a tool upstream dropped) is simply not in the list and is
//! never staged. What is staged is then proven by the shared verifier, not by
//! this module.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::llama_runtime::durable;
use crate::llama_runtime::verify::{InstallKind, VerificationIntent};

use super::{Accelerator, BUILD_TARGETS, TargetOs};

const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    /// A required output is missing from the build tree.
    MissingOutput(String),
    /// The ninja target list could not be read.
    Targets(String),
    /// CUDA runtime DLLs were not found in any search directory.
    MissingCudaRuntime {
        searched: Vec<String>,
    },
    /// A runtime DLL pattern matched a `ggml-*` file (ported guard).
    RefusedGgml(String),
    DiskExhausted {
        needed: u64,
        free: u64,
    },
    Io(String),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingOutput(n) => write!(f, "the build did not produce {n}"),
            Self::Targets(e) => write!(f, "could not read the generator's target list: {e}"),
            Self::MissingCudaRuntime { searched } => write!(
                f,
                "required cudart64_*, cublas64_* and cublasLt64_* DLL families were not all found in {}",
                searched.join(", ")
            ),
            Self::RefusedGgml(n) => {
                write!(f, "refusing to copy {n} as a CUDA runtime library")
            }
            Self::DiskExhausted { needed, free } => write!(
                f,
                "staging needs about {} MiB but only {} MiB are free",
                needed / MIB,
                free / MIB
            ),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StageError {}

impl From<std::io::Error> for StageError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// Normalised relative path for set comparison (`\` and `./` removed).
fn norm(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            p => parts.push(p),
        }
    }
    parts.join("/")
}

/// Outputs listed by `ninja -t targets all` (`path: rule` per line), relative
/// to the build directory.
pub fn produced_targets(ninja_output: &str) -> BTreeSet<String> {
    ninja_output
        .lines()
        .filter_map(|line| line.split_once(": "))
        .map(|(path, _)| norm(path))
        .collect()
}

/// Library files, including the versioned and soname forms.
fn is_shared_library(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".dll")
        || lower.ends_with(".dylib")
        || lower.ends_with(".so")
        || lower
            .rsplit_once(".so.")
            .is_some_and(|(_, v)| v.split('.').all(|p| p.chars().all(|c| c.is_ascii_digit())))
}

fn exe_names(os: TargetOs) -> Vec<String> {
    BUILD_TARGETS.iter().map(|t| os.exe_name(t)).collect()
}

/// `libllama-cli-impl.dylib` → `llama-cli`. Per-tool implementation libraries
/// of tools we do not ship are left behind.
fn impl_tool(name: &str) -> Option<String> {
    let stem = name.strip_prefix("lib").unwrap_or(name);
    stem.split_once("-impl").map(|(tool, _)| tool.to_string())
}

/// Whether `name` belongs in the staged runtime, before the produced-set check.
fn wanted_name(name: &str, os: TargetOs) -> bool {
    if name.starts_with('.') {
        return false;
    }
    if exe_names(os).iter().any(|e| e == name) {
        return true;
    }
    if !is_shared_library(name) {
        return false;
    }
    match impl_tool(name) {
        Some(tool) => BUILD_TARGETS.contains(&tool.as_str()),
        None => true,
    }
}

/// The plugin a successful build for `accelerator` must contain.
pub fn required_plugin_stem(accelerator: Accelerator) -> &'static str {
    match accelerator {
        Accelerator::Metal => "ggml-metal",
        Accelerator::Cuda => "ggml-cuda",
        Accelerator::Cpu => "ggml-cpu",
    }
}

fn has_plugin(files: &BTreeSet<String>, stem: &str) -> bool {
    files.iter().any(|f| {
        let base = f.strip_prefix("lib").unwrap_or(f);
        base.strip_prefix(stem)
            .is_some_and(|rest| rest.starts_with('.') && is_shared_library(f))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan {
    pub os: TargetOs,
    pub accelerator: Accelerator,
    /// `<build>/bin` or `<build>/bin/Release`.
    pub output_dir: PathBuf,
    /// Relative to the build directory, normalised: `bin` / `bin/Release`.
    pub output_rel: String,
    pub staging_dir: PathBuf,
    pub cuda_root: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageReport {
    pub files: Vec<String>,
    pub symlinks: Vec<String>,
    /// Present in the build directory but not produced by this configure.
    pub skipped_stale: Vec<String>,
    pub cuda_runtime: Vec<String>,
    pub bytes: u64,
}

/// What would be copied, decided from the directory listing and target list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSelection {
    pub files: Vec<String>,
    pub symlinks: Vec<String>,
    pub skipped_stale: Vec<String>,
}

pub fn select_outputs(
    plan: &StagePlan,
    produced: &BTreeSet<String>,
) -> Result<StageSelection, StageError> {
    let mut files = BTreeSet::new();
    let mut symlinks = BTreeSet::new();
    let mut stale = BTreeSet::new();
    let entries = match std::fs::read_dir(&plan.output_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(StageError::MissingOutput(format!(
                "the output directory {}",
                plan.output_dir.display()
            )));
        }
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let file_type = entry.file_type()?;
        if file_type.is_dir() || !wanted_name(&name, plan.os) {
            continue;
        }
        let key = format!("{}/{name}", plan.output_rel);
        if !produced.contains(&key) {
            stale.insert(name);
            continue;
        }
        if file_type.is_symlink() {
            symlinks.insert(name);
        } else {
            files.insert(name);
        }
    }
    for exe in exe_names(plan.os) {
        if !files.contains(&exe) {
            return Err(StageError::MissingOutput(exe));
        }
    }
    let stem = required_plugin_stem(plan.accelerator);
    // A symlink is only useful with its target; drop dangling ones.
    #[cfg(unix)]
    {
        let all: BTreeSet<String> = files.union(&symlinks).cloned().collect();
        let dangling: Vec<String> = symlinks
            .iter()
            .filter(|name| {
                let path = plan.output_dir.join(name);
                match std::fs::read_link(&path) {
                    Ok(target) => {
                        target.is_absolute()
                            || target.components().count() != 1
                            || !all.contains(&target.to_string_lossy().into_owned())
                            || !path.is_file()
                    }
                    Err(_) => true,
                }
            })
            .cloned()
            .collect();
        for name in dangling {
            symlinks.remove(&name);
            stale.insert(name);
        }
    }
    let selected: BTreeSet<String> = files.union(&symlinks).cloned().collect();
    if !has_plugin(&selected, stem) {
        return Err(StageError::MissingOutput(format!("the {stem} backend")));
    }
    Ok(StageSelection {
        files: files.into_iter().collect(),
        symlinks: symlinks.into_iter().collect(),
        skipped_stale: stale.into_iter().collect(),
    })
}

/// `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`.
fn is_cuda_runtime_dll(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if !lower.ends_with(".dll") {
        return false;
    }
    ["cudart64_", "cublas64_", "cublaslt64_"]
        .iter()
        .any(|p| lower.starts_with(p))
}

/// The CUDA runtime DLLs to ship, searching `bin\x64`, `bin` and `lib` in that
/// order (the same three directories upstream release CI copies from). Each
/// DLL name is taken from the first directory that has it.
pub fn select_cuda_runtime(root: &Path) -> Result<Vec<(String, PathBuf)>, StageError> {
    let dirs = [
        root.join("bin").join("x64"),
        root.join("bin"),
        root.join("lib"),
    ];
    let mut chosen: Vec<(String, PathBuf)> = Vec::new();
    for dir in &dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            if !is_cuda_runtime_dll(&name) {
                continue;
            }
            if name.to_ascii_lowercase().starts_with("ggml-") {
                return Err(StageError::RefusedGgml(name));
            }
            if !chosen.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name)) {
                chosen.push((name.clone(), dir.join(&name)));
            }
        }
    }
    if ["cudart64_", "cublas64_", "cublaslt64_"]
        .iter()
        .any(|prefix| {
            !chosen
                .iter()
                .any(|(n, _)| n.to_ascii_lowercase().starts_with(prefix))
        })
    {
        return Err(StageError::MissingCudaRuntime {
            searched: dirs.iter().map(|d| d.display().to_string()).collect(),
        });
    }
    chosen.sort();
    Ok(chosen)
}

/// Bytes the copy will write.
fn total_bytes(plan: &StagePlan, sel: &StageSelection, runtime: &[(String, PathBuf)]) -> u64 {
    let mut total = 0u64;
    for name in &sel.files {
        total += std::fs::metadata(plan.output_dir.join(name))
            .map(|m| m.len())
            .unwrap_or(0);
    }
    for (_, path) in runtime {
        total += std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    total
}

/// Copy the selected outputs into `plan.staging_dir`. `free_space` is read
/// **after** the exact size is known, so the stage recheck uses real numbers
/// (plan §4.4: never delete a usable runtime set to make room).
pub fn stage(
    plan: &StagePlan,
    produced: &BTreeSet<String>,
    free_space: &dyn Fn() -> Option<u64>,
) -> Result<StageReport, StageError> {
    match std::fs::symlink_metadata(&plan.staging_dir) {
        Ok(meta) if !meta.is_dir() || meta.file_type().is_symlink() => {
            return Err(StageError::Io(
                "staging must be an empty real directory".into(),
            ));
        }
        Ok(_) if std::fs::read_dir(&plan.staging_dir)?.next().is_some() => {
            return Err(StageError::Io("staging directory is not empty".into()));
        }
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let selection = select_outputs(plan, produced)?;
    let runtime = if plan.accelerator == Accelerator::Cuda && plan.os == TargetOs::Windows {
        let root = plan
            .cuda_root
            .as_deref()
            .ok_or_else(|| StageError::Io("the CUDA toolkit root is unknown".into()))?;
        select_cuda_runtime(root)?
    } else {
        Vec::new()
    };
    let bytes = total_bytes(plan, &selection, &runtime);
    let needed = bytes + bytes / 4 + 64 * MIB;
    if let Some(free) = free_space()
        && free < needed
    {
        return Err(StageError::DiskExhausted { needed, free });
    }

    std::fs::create_dir_all(&plan.staging_dir)?;
    for name in &selection.files {
        durable::copy_file(&plan.output_dir.join(name), &plan.staging_dir.join(name))?;
    }
    for (name, path) in &runtime {
        durable::copy_file(path, &plan.staging_dir.join(name))?;
    }
    #[cfg(unix)]
    for name in &selection.symlinks {
        let target = std::fs::read_link(plan.output_dir.join(name))?;
        std::os::unix::fs::symlink(target, plan.staging_dir.join(name))?;
    }
    // The entries themselves (new files, symlinks) must be durable too.
    durable::sync_dir(&plan.staging_dir)?;
    Ok(StageReport {
        files: selection.files,
        symlinks: selection.symlinks,
        skipped_stale: selection.skipped_stale,
        cuda_runtime: runtime.into_iter().map(|(n, _)| n).collect(),
        bytes,
    })
}

/// What the shared verifier must prove about a source build. Built from the
/// resolved job, never from the manifest or client text.
pub fn source_intent(
    accelerator: Accelerator,
    pinned_commit: &str,
    device_id: Option<String>,
) -> VerificationIntent {
    VerificationIntent {
        kind: InstallKind::Source,
        expected_backend: Some(accelerator.backend()),
        pinned_commit: Some(pinned_commit.to_string()),
        // Decision D9: a CUDA source build is FA-all or it is not installed.
        require_fa_all: accelerator == Accelerator::Cuda,
        device_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    const NINJA_METAL: &str = "\
bin/llama-server: CXX_EXECUTABLE_LINKER__llama-server_Release
bin/llama-bench: CXX_EXECUTABLE_LINKER__llama-bench_Release
bin/libggml-metal.so: CXX_SHARED_LIBRARY_LINKER__ggml-metal_Release
bin/libggml-base.0.27.0.dylib: CXX_SHARED_LIBRARY_LINKER__ggml-base_Release
bin/libggml-base.dylib: phony
bin/libllama-server-impl.dylib: CXX_SHARED_LIBRARY_LINKER__llama-server-impl_Release
bin/libllama-bench-impl.dylib: CXX_SHARED_LIBRARY_LINKER__llama-bench-impl_Release
bin/libllama-cli-impl.dylib: CXX_SHARED_LIBRARY_LINKER__llama-cli-impl_Release
all: phony bin/llama-server
";

    fn metal_plan(build: &Path, staging: &Path) -> StagePlan {
        StagePlan {
            os: TargetOs::Macos,
            accelerator: Accelerator::Metal,
            output_dir: build.join("bin"),
            output_rel: "bin".into(),
            staging_dir: staging.to_path_buf(),
            cuda_root: None,
        }
    }

    fn metal_tree(build: &Path) {
        let bin = build.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        for f in [
            "llama-server",
            "llama-bench",
            "libggml-metal.so",
            "libggml-base.0.27.0.dylib",
            "libllama-server-impl.dylib",
            "libllama-bench-impl.dylib",
        ] {
            write(&bin, f, f);
        }
        #[cfg(unix)]
        {
            let link = bin.join("libggml-base.dylib");
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink("libggml-base.0.27.0.dylib", link).unwrap();
        }
    }

    #[test]
    fn cudart_alone_does_not_satisfy_the_cuda_runtime_closure() {
        let root = tempfile::tempdir().unwrap();
        cuda_root(root.path(), &[("bin/x64", &["cudart64_13.dll"])]);
        assert!(matches!(
            select_cuda_runtime(root.path()),
            Err(StageError::MissingCudaRuntime { .. })
        ));
    }

    #[test]
    fn nonempty_staging_is_not_merged_with_a_new_runtime() {
        let root = tempfile::tempdir().unwrap();
        let build = root.path().join("build");
        let staging = root.path().join("stage");
        metal_tree(&build);
        std::fs::create_dir(&staging).unwrap();
        write(&staging, "old-plugin.so", "old");
        assert!(
            stage(
                &metal_plan(&build, &staging),
                &produced_targets(NINJA_METAL),
                &|| Some(u64::MAX)
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(staging.join("old-plugin.so")).unwrap(),
            "old"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_plugin_symlink_cannot_satisfy_the_backend_requirement() {
        let root = tempfile::tempdir().unwrap();
        let build = root.path().join("build");
        metal_tree(&build);
        let plugin = build.join("bin/libggml-metal.so");
        std::fs::remove_file(&plugin).unwrap();
        std::os::unix::fs::symlink("absent.so", &plugin).unwrap();
        assert!(
            select_outputs(
                &metal_plan(&build, &root.path().join("stage")),
                &produced_targets(NINJA_METAL)
            )
            .is_err()
        );
    }

    #[test]
    fn the_target_list_is_normalized() {
        let t = produced_targets(
            "bin\\Release\\llama-server.exe: LINK\n./bin/x.dll: LINK\nno colon here\n",
        );
        assert!(t.contains("bin/Release/llama-server.exe"));
        assert!(t.contains("bin/x.dll"));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn library_and_impl_naming() {
        for lib in [
            "libggml.dylib",
            "libggml.so.1.2",
            "ggml-cuda.dll",
            "libllama.0.7.0.dylib",
            "libggml-metal.so",
        ] {
            assert!(is_shared_library(lib), "{lib}");
        }
        for not in ["llama-server", "notes.txt", "libx.sol", "a.so.x"] {
            assert!(!is_shared_library(not), "{not}");
        }
        assert_eq!(
            impl_tool("libllama-cli-impl.dylib").as_deref(),
            Some("llama-cli")
        );
        assert_eq!(impl_tool("libggml.dylib"), None);
        assert!(wanted_name("libllama-server-impl.dylib", TargetOs::Macos));
        assert!(!wanted_name("libllama-cli-impl.dylib", TargetOs::Macos));
        assert!(!wanted_name("llama-cli", TargetOs::Macos));
        assert!(wanted_name("llama-server.exe", TargetOs::Windows));
        assert!(!wanted_name("llama-server", TargetOs::Windows));
    }

    #[test]
    fn stages_exactly_what_this_configure_produced() {
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        metal_tree(&build);
        // Stale outputs from earlier configures.
        for stale in [
            "llama-cli",
            "libggml-vulkan.so",
            "libllama-cli-impl.dylib",
            "libmtmd.0.3.0.dylib",
        ] {
            write(&build.join("bin"), stale, "old");
        }
        let staging = tmp.path().join("staging");
        let plan = metal_plan(&build, &staging);
        let report = stage(&plan, &produced_targets(NINJA_METAL), &|| Some(u64::MAX)).unwrap();
        assert_eq!(
            report.files,
            [
                "libggml-base.0.27.0.dylib",
                "libggml-metal.so",
                "libllama-bench-impl.dylib",
                "libllama-server-impl.dylib",
                "llama-bench",
                "llama-server"
            ]
        );
        assert!(
            report
                .skipped_stale
                .contains(&"libggml-vulkan.so".to_string())
        );
        assert!(
            report
                .skipped_stale
                .contains(&"libmtmd.0.3.0.dylib".to_string())
        );
        assert!(!staging.join("libggml-vulkan.so").exists());
        assert!(!staging.join("llama-cli").exists());
        #[cfg(unix)]
        {
            assert_eq!(report.symlinks, ["libggml-base.dylib"]);
            assert_eq!(
                std::fs::read_link(staging.join("libggml-base.dylib")).unwrap(),
                PathBuf::from("libggml-base.0.27.0.dylib")
            );
        }
    }

    #[test]
    fn every_staged_file_is_synced_before_stage_returns() {
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        metal_tree(&build);
        let staging = tmp.path().join("staging");
        let plan = metal_plan(&build, &staging);
        let before = crate::llama_runtime::durable::syncs();
        let report = stage(&plan, &produced_targets(NINJA_METAL), &|| Some(u64::MAX)).unwrap();
        assert_eq!(
            crate::llama_runtime::durable::syncs() - before,
            report.files.len(),
            "one sync per copied file"
        );
    }

    #[test]
    fn a_missing_executable_or_backend_plugin_is_a_build_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        metal_tree(&build);
        std::fs::remove_file(build.join("bin/llama-bench")).unwrap();
        let plan = metal_plan(&build, &tmp.path().join("s"));
        let produced = produced_targets(NINJA_METAL);
        assert_eq!(
            stage(&plan, &produced, &|| None).unwrap_err(),
            StageError::MissingOutput("llama-bench".into())
        );
        metal_tree(&build);
        std::fs::remove_file(build.join("bin/libggml-metal.so")).unwrap();
        let err = stage(&plan, &produced, &|| None).unwrap_err();
        assert!(
            matches!(&err, StageError::MissingOutput(m) if m.contains("ggml-metal")),
            "{err:?}"
        );
    }

    #[test]
    fn a_stale_plugin_cannot_stand_in_for_the_required_one() {
        // `libggml-metal.so` exists but this configure no longer produces it.
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        metal_tree(&build);
        let listing = NINJA_METAL.replace("bin/libggml-metal.so", "bin/libggml-cpu.so");
        let err = stage(
            &metal_plan(&build, &tmp.path().join("s")),
            &produced_targets(&listing),
            &|| None,
        )
        .unwrap_err();
        assert!(
            matches!(&err, StageError::MissingOutput(m) if m.contains("ggml-metal")),
            "{err:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_pointing_outside_or_nowhere_are_not_staged() {
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        metal_tree(&build);
        let bin = build.join("bin");
        std::os::unix::fs::symlink("/etc/passwd", bin.join("libggml-evil.dylib")).unwrap();
        std::os::unix::fs::symlink("missing.dylib", bin.join("libggml-dangling.dylib")).unwrap();
        std::os::unix::fs::symlink("../escape.dylib", bin.join("libggml-up.dylib")).unwrap();
        let listing = format!(
            "{NINJA_METAL}bin/libggml-evil.dylib: phony\nbin/libggml-dangling.dylib: phony\nbin/libggml-up.dylib: phony\n"
        );
        let staging = tmp.path().join("s");
        let report = stage(
            &metal_plan(&build, &staging),
            &produced_targets(&listing),
            &|| None,
        )
        .unwrap();
        assert_eq!(report.symlinks, ["libggml-base.dylib"]);
        for bad in [
            "libggml-evil.dylib",
            "libggml-dangling.dylib",
            "libggml-up.dylib",
        ] {
            assert!(
                !staging.join(bad).exists()
                    && std::fs::symlink_metadata(staging.join(bad)).is_err(),
                "{bad}"
            );
            assert!(report.skipped_stale.contains(&bad.to_string()));
        }
    }

    #[test]
    fn disk_exhaustion_is_detected_before_anything_is_copied() {
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        metal_tree(&build);
        let staging = tmp.path().join("s");
        let err = stage(
            &metal_plan(&build, &staging),
            &produced_targets(NINJA_METAL),
            &|| Some(1024),
        )
        .unwrap_err();
        assert!(
            matches!(err, StageError::DiskExhausted { free: 1024, .. }),
            "{err:?}"
        );
        assert!(!staging.exists(), "nothing is written when space is short");
    }

    fn cuda_root(root: &Path, layout: &[(&str, &[&str])]) {
        for (dir, names) in layout {
            let d = root.join(dir);
            std::fs::create_dir_all(&d).unwrap();
            for n in *names {
                std::fs::write(d.join(n), format!("{dir}/{n}")).unwrap();
            }
        }
    }

    #[test]
    fn cuda_runtime_comes_from_bin_x64_on_cuda_13() {
        let tmp = tempfile::tempdir().unwrap();
        cuda_root(
            tmp.path(),
            &[
                (
                    "bin/x64",
                    &[
                        "cudart64_13.dll",
                        "cublas64_13.dll",
                        "cublasLt64_13.dll",
                        "cufft64_12.dll",
                        "curand64_10.dll",
                    ],
                ),
                ("bin", &[]),
            ],
        );
        let chosen = select_cuda_runtime(tmp.path()).unwrap();
        let names: Vec<_> = chosen.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            ["cublas64_13.dll", "cublasLt64_13.dll", "cudart64_13.dll"]
        );
    }

    #[test]
    fn cuda_runtime_search_order_is_x64_then_bin_then_lib() {
        let tmp = tempfile::tempdir().unwrap();
        cuda_root(
            tmp.path(),
            &[
                ("bin/x64", &["cudart64_12.dll"]),
                ("bin", &["cudart64_12.dll", "cublas64_12.dll"]),
                ("lib", &["cublasLt64_12.dll", "cublas64_12.dll"]),
            ],
        );
        let chosen = select_cuda_runtime(tmp.path()).unwrap();
        let from = |n: &str| chosen.iter().find(|(x, _)| x == n).unwrap().1.clone();
        assert!(from("cudart64_12.dll").ends_with("bin/x64/cudart64_12.dll"));
        assert!(from("cublas64_12.dll").ends_with("bin/cublas64_12.dll"));
        assert!(from("cublasLt64_12.dll").ends_with("lib/cublasLt64_12.dll"));
    }

    #[test]
    fn a_toolkit_without_cudart_is_an_error_listing_where_it_looked() {
        let tmp = tempfile::tempdir().unwrap();
        cuda_root(tmp.path(), &[("bin", &["nvcc.exe", "cublas64_13.dll"])]);
        let err = select_cuda_runtime(tmp.path()).unwrap_err();
        assert!(
            matches!(&err, StageError::MissingCudaRuntime { searched } if searched.len() == 3),
            "{err:?}"
        );
        assert!(err.to_string().contains("cudart64_"));
    }

    #[test]
    fn only_runtime_libraries_are_copied_never_ggml_or_the_rest() {
        assert!(is_cuda_runtime_dll("cudart64_13.dll"));
        assert!(is_cuda_runtime_dll("CUBLASLT64_13.DLL"));
        assert!(!is_cuda_runtime_dll("ggml-cuda.dll"));
        assert!(!is_cuda_runtime_dll("cufft64_12.dll"));
        assert!(!is_cuda_runtime_dll("cudart64_13.lib"));
    }

    #[test]
    fn windows_cuda_stage_includes_the_runtime_dlls_and_requires_the_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        let build = tmp.path().join("build");
        let out = build.join("bin/Release");
        std::fs::create_dir_all(&out).unwrap();
        for f in [
            "llama-server.exe",
            "llama-bench.exe",
            "ggml-cuda.dll",
            "ggml-base.dll",
            "ggml-cpu-haswell.dll",
            "llama.dll",
            "ggml-vulkan.dll",
        ] {
            write(&out, f, f);
        }
        let toolkit = tmp.path().join("cuda");
        cuda_root(
            &toolkit,
            &[(
                "bin/x64",
                &["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"],
            )],
        );
        let listing = "\
bin/Release/llama-server.exe: LINK
bin/Release/llama-bench.exe: LINK
bin/Release/ggml-cuda.dll: LINK
bin/Release/ggml-base.dll: LINK
bin/Release/ggml-cpu-haswell.dll: LINK
bin/Release/llama.dll: LINK
";
        let staging = tmp.path().join("s");
        let plan = StagePlan {
            os: TargetOs::Windows,
            accelerator: Accelerator::Cuda,
            output_dir: out,
            output_rel: "bin/Release".into(),
            staging_dir: staging.clone(),
            cuda_root: Some(toolkit),
        };
        let report = stage(&plan, &produced_targets(listing), &|| None).unwrap();
        assert!(report.files.contains(&"ggml-cuda.dll".to_string()));
        assert!(report.files.contains(&"ggml-cpu-haswell.dll".to_string()));
        assert_eq!(report.skipped_stale, ["ggml-vulkan.dll"]);
        assert_eq!(report.cuda_runtime.len(), 3);
        for dll in [
            "cudart64_13.dll",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "llama-server.exe",
        ] {
            assert!(staging.join(dll).is_file(), "{dll}");
        }
        assert!(!staging.join("ggml-vulkan.dll").exists());
    }

    #[test]
    fn a_windows_cuda_stage_without_a_toolkit_root_fails_loudly() {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("bin/Release");
        std::fs::create_dir_all(&out).unwrap();
        for f in ["llama-server.exe", "llama-bench.exe", "ggml-cuda.dll"] {
            write(&out, f, f);
        }
        let plan = StagePlan {
            os: TargetOs::Windows,
            accelerator: Accelerator::Cuda,
            output_dir: out,
            output_rel: "bin/Release".into(),
            staging_dir: tmp.path().join("s"),
            cuda_root: None,
        };
        let listing = "bin/Release/llama-server.exe: L\nbin/Release/llama-bench.exe: L\nbin/Release/ggml-cuda.dll: L\n";
        assert!(matches!(
            stage(&plan, &produced_targets(listing), &|| None),
            Err(StageError::Io(_))
        ));
    }

    #[test]
    fn the_source_intent_is_server_built_and_cuda_demands_fa_all() {
        let commit = "a".repeat(40);
        let cuda = source_intent(Accelerator::Cuda, &commit, Some("CUDA0".into()));
        assert!(cuda.require_fa_all);
        assert_eq!(cuda.kind, InstallKind::Source);
        assert_eq!(cuda.pinned_commit.as_deref(), Some(commit.as_str()));
        assert_eq!(cuda.device_id.as_deref(), Some("CUDA0"));
        let metal = source_intent(Accelerator::Metal, &commit, None);
        assert!(!metal.require_fa_all);
        assert_eq!(
            metal.expected_backend,
            Some(crate::inference::kv_support::Backend::Metal)
        );
    }

    /// Loader fixtures through the shared verifier: the staged set only counts
    /// if the probe proves it.
    mod verifier {
        use super::*;
        use crate::llama_runtime::probe::{RuntimeProbe, kv_basis_from_probe, parse_probe_output};
        use crate::llama_runtime::verify::{VerifyError, evaluate_probe};

        fn probe(dir: &Path, text: &str) -> RuntimeProbe {
            let parsed = parse_probe_output(
                &text.replace("/opt/llama", &dir.to_string_lossy()),
                "/nonexistent/m.gguf",
            );
            let basis = kv_basis_from_probe(&parsed, None);
            RuntimeProbe {
                parsed,
                identity: None,
                basis,
            }
        }

        const METAL: &str =
            include_str!("../../tests/fixtures/llama_probe/metal_missing_model.txt");
        const CUDA: &str = include_str!("../../tests/fixtures/llama_probe/cuda_missing_model.txt");

        #[test]
        fn a_stale_cpu_only_runtime_cannot_pass_as_the_intended_metal_build() {
            let dir = tempfile::tempdir().unwrap();
            let cpu_only = "load_backend: loaded CPU backend from /opt/llama/libggml-cpu.so\ncmn common_param: system_info: n = 1 | CPU : NEON = 1 |\n";
            let err = evaluate_probe(
                &probe(dir.path(), cpu_only),
                dir.path(),
                &source_intent(Accelerator::Metal, &"a".repeat(40), None),
            )
            .unwrap_err();
            assert!(
                matches!(err, VerifyError::BackendMismatch { .. }),
                "{err:?}"
            );
        }

        #[test]
        fn a_build_of_a_different_commit_is_rejected_even_with_the_right_backend() {
            let dir = tempfile::tempdir().unwrap();
            let p = probe(dir.path(), METAL);
            // The fixture's reported commit is not this pin.
            let err = evaluate_probe(
                &p,
                dir.path(),
                &source_intent(Accelerator::Metal, &"f".repeat(40), None),
            )
            .unwrap_err();
            assert!(matches!(err, VerifyError::CommitMismatch { .. }), "{err:?}");
        }

        #[test]
        fn a_missing_intended_device_fails_verification() {
            let dir = tempfile::tempdir().unwrap();
            let p = probe(dir.path(), METAL);
            let commit = p.parsed.build_commit.clone();
            let pinned = format!("{}{}", commit.unwrap_or_default(), "0".repeat(40));
            let pinned = &pinned[..40];
            let err = evaluate_probe(
                &p,
                dir.path(),
                &source_intent(Accelerator::Metal, pinned, Some("MTL9".into())),
            )
            .unwrap_err();
            assert!(
                matches!(err, VerifyError::DeviceNotEnumerated(_)),
                "{err:?}"
            );
        }

        fn cuda_probe(dir: &Path) -> RuntimeProbe {
            // The fixture was captured on ryne; point its plugin paths at the
            // directory under test.
            let text = CUDA.replace(
                "D:\\AI\\loaders\\llamacpp\\",
                &format!("{}/", dir.display()),
            );
            probe(dir, &text)
        }

        fn pin_of(p: &RuntimeProbe) -> String {
            let commit = p
                .parsed
                .build_commit
                .clone()
                .expect("fixture reports a commit");
            format!("{commit}{}", "0".repeat(40))[..40].to_string()
        }

        #[test]
        fn a_cuda_source_runtime_that_reports_fa_all_passes_with_its_pin() {
            let dir = tempfile::tempdir().unwrap();
            let p = cuda_probe(dir.path());
            let report = evaluate_probe(
                &p,
                dir.path(),
                &source_intent(Accelerator::Cuda, &pin_of(&p), None),
            )
            .unwrap();
            assert_eq!(report.backend, crate::inference::kv_support::Backend::Cuda);
        }

        #[test]
        fn a_cuda_source_runtime_that_reports_the_default_list_is_rejected() {
            let dir = tempfile::tempdir().unwrap();
            let text = CUDA
                .replace(
                    "D:\\AI\\loaders\\llamacpp\\",
                    &format!("{}/", dir.path().display()),
                )
                .replace(
                    "FA_QUANTS = all",
                    "FA_QUANTS = q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16",
                );
            let p = probe(dir.path(), &text);
            let err = evaluate_probe(
                &p,
                dir.path(),
                &source_intent(Accelerator::Cuda, &pin_of(&p), None),
            )
            .unwrap_err();
            assert!(matches!(err, VerifyError::FaQuantsNotAll(_)), "{err:?}");
        }

        #[test]
        fn a_cuda_source_runtime_with_no_fa_record_is_rejected() {
            let dir = tempfile::tempdir().unwrap();
            let text = CUDA
                .replace(
                    "D:\\AI\\loaders\\llamacpp\\",
                    &format!("{}/", dir.path().display()),
                )
                .replace("FA_QUANTS = all | ", "");
            let p = probe(dir.path(), &text);
            let err = evaluate_probe(
                &p,
                dir.path(),
                &source_intent(Accelerator::Cuda, &pin_of(&p), None),
            )
            .unwrap_err();
            assert!(matches!(err, VerifyError::FaQuantsNotAll(_)), "{err:?}");
        }

        #[test]
        fn a_cuda_plugin_loaded_from_outside_the_staged_directory_is_rejected() {
            let dir = tempfile::tempdir().unwrap();
            // The unrewritten fixture loads from D:\AI\loaders: not this directory.
            let p = probe(dir.path(), CUDA);
            let err = evaluate_probe(
                &p,
                dir.path(),
                &source_intent(Accelerator::Cuda, &pin_of(&p), None),
            )
            .unwrap_err();
            assert!(
                matches!(err, VerifyError::PluginOutsideRuntime(_)),
                "{err:?}"
            );
        }
    }
}
