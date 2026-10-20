//! The one verifier for a llama.cpp runtime directory (plan §4.3, §4.6.1).
//!
//! Prebuilt installs, source promotions and rollbacks all call this; `job.rs`
//! never runs its own probe. Three levels are kept separate:
//!
//! 1. **Compile identity** (source builds): the loaded runtime reports the
//!    pinned commit.
//! 2. **Loader/backend closure**: executed with a sanitized environment so a
//!    build tree, toolkit directory or inherited library override cannot
//!    satisfy the loader; every loaded backend plugin must live inside the
//!    directory being verified; the intended backend (not a CPU fallback) must
//!    be the one that loaded.
//! 3. **Model execution** is *not* proven here. A runtime with no model is
//!    verified to the loader/backend level and the rest is left to manual QA.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::inference::kv_support::{Backend, compiled_fa_pairs};
use crate::llama_runtime::probe::{
    EnvPolicy, ProbeError, RuntimeProbe, probe_runtime_with_timeout,
};
use crate::llama_runtime::promote::VerifyStage;

/// A cold macOS Metal start can take well over the 10 s ordinary probe limit
/// (the previous health check allowed 30 s for the same reason).
pub const VERIFY_PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// Prefix of the note emitted when only `--help` could be verified. A later
/// stage with full evidence drops it (see `promote::absorb_report`).
pub const EVIDENCE_NOTE_PREFIX: &str = "Runtime evidence could not be read";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstallKind {
    Prebuilt,
    Source,
    Rollback,
}

/// Server-owned expectations. Built from the resolved job/release operation,
/// never from client text or manifest claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationIntent {
    pub kind: InstallKind,
    /// The accelerator this install is supposed to drive. `None` means the
    /// accelerator is not one we have evidence rules for (ROCm, SYCL...).
    pub expected_backend: Option<Backend>,
    /// Source builds: the full pinned commit.
    pub pinned_commit: Option<String>,
    /// Source CUDA builds always require `FA_QUANTS=all` (decision D9).
    pub require_fa_all: bool,
    /// A detected device the user/job selected; must be enumerated.
    pub device_id: Option<String>,
}

impl VerificationIntent {
    pub fn prebuilt(backend: &str) -> Self {
        Self {
            kind: InstallKind::Prebuilt,
            expected_backend: backend_for_release_name(backend),
            pinned_commit: None,
            require_fa_all: false,
            device_id: None,
        }
    }

    pub fn rollback() -> Self {
        Self {
            kind: InstallKind::Rollback,
            expected_backend: None,
            pinned_commit: None,
            require_fa_all: false,
            device_id: None,
        }
    }
}

/// Map a release backend name (`metal`, `cuda13`, `vulkan`, `cpu`, ...) to the
/// backend the probe should report. Unknown accelerators get no expectation.
pub fn backend_for_release_name(name: &str) -> Option<Backend> {
    let lower = name.to_ascii_lowercase();
    if lower.starts_with("cuda") {
        Some(Backend::Cuda)
    } else {
        match lower.as_str() {
            "metal" => Some(Backend::Metal),
            "vulkan" => Some(Backend::Vulkan),
            "cpu" | "" => Some(Backend::Cpu),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerificationReport {
    pub backend: Backend,
    pub build_number: Option<u32>,
    pub build_commit: Option<String>,
    pub identity_fingerprint: Option<String>,
    /// Non-fatal observations (for example evidence that could not be read).
    pub notes: Vec<String>,
    /// True when the probe output was not recognized and only `--help` ran.
    pub evidence_unavailable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    MissingExecutable(PathBuf),
    CannotRun(String),
    BackendMismatch {
        expected: Backend,
        found: Backend,
    },
    PluginOutsideRuntime(PathBuf),
    CommitMismatch {
        pinned: String,
        found: Option<String>,
    },
    /// `llama-bench` carries a different commit than the server. Kept apart
    /// from [`VerifyError::CommitMismatch`] because the message must name the
    /// executable that disagreed.
    BenchCommitMismatch {
        pinned: String,
        found: Option<String>,
    },
    FaQuantsNotAll(String),
    DeviceNotEnumerated(String),
    EvidenceRequired(String),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingExecutable(p) => write!(f, "{} is missing", p.display()),
            Self::CannotRun(e) => write!(f, "the runtime could not be run: {e}"),
            Self::BackendMismatch { expected, found } => write!(
                f,
                "the runtime loaded the {found:?} backend but this install is for {expected:?}"
            ),
            Self::PluginOutsideRuntime(p) => write!(
                f,
                "a backend loaded from outside the runtime directory: {}",
                p.display()
            ),
            Self::CommitMismatch { pinned, found } => write!(
                f,
                "the runtime reports commit {} but {pinned} was pinned",
                found.as_deref().unwrap_or("(none)")
            ),
            Self::BenchCommitMismatch { pinned, found } => write!(
                f,
                "llama-bench reports commit {} but {pinned} was pinned",
                found.as_deref().unwrap_or("(none)")
            ),
            Self::FaQuantsNotAll(why) => {
                write!(f, "the CUDA runtime does not report FA_QUANTS=all: {why}")
            }
            Self::DeviceNotEnumerated(d) => {
                write!(
                    f,
                    "the selected device {d} was not enumerated by the runtime"
                )
            }
            Self::EvidenceRequired(why) => write!(f, "required runtime evidence is missing: {why}"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// What `prepare_runtime_dir` found. Preparation never executes anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inventory {
    pub files: usize,
    pub executable: Option<PathBuf>,
}

/// Preparation that must precede any staged execution: executable bits.
/// Synchronous and execution-free.
pub fn prepare_runtime_dir(dir: &Path, exe_name: &str) -> std::io::Result<Inventory> {
    let mut files = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                stack.push(entry.path());
            } else {
                files += 1;
            }
        }
    }
    let executable = dir.join(exe_name);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Every llama.cpp tool in the set is an executable; shared libraries
        // keep their mode.
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            let is_tool = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("llama-") && !n.contains('.'));
            if is_tool && path.is_file() {
                let mut perms = std::fs::metadata(&path)?.permissions();
                perms.set_mode(perms.mode() | 0o755);
                std::fs::set_permissions(&path, perms)?;
            }
        }
    }
    Ok(Inventory {
        files,
        executable: executable.is_file().then_some(executable),
    })
}

/// Strip Gatekeeper quarantine from the whole set (executable, co-located
/// dylibs and Metal shaders alike). Runs the `xattr` subprocess, so callers
/// keep it inside a blocking section; best-effort, like `start_server`'s own
/// strip.
#[cfg(target_os = "macos")]
pub fn strip_quarantine(dir: &Path) {
    let _ = std::process::Command::new("xattr")
        .args(["-rd", "com.apple.quarantine"])
        .arg(dir)
        .output();
}

/// Canonicalize `path`, resolving symlinks (`/var` -> `/private/var`) even when
/// the leaf does not exist, by canonicalizing the nearest existing ancestor.
fn canonical(path: &Path) -> PathBuf {
    if let Ok(resolved) = path.canonicalize() {
        return resolved;
    }
    let mut missing = Vec::new();
    let mut current = path;
    while let Some(parent) = current.parent() {
        if let Some(name) = current.file_name() {
            missing.push(name.to_os_string());
        }
        if let Ok(resolved) = parent.canonicalize() {
            let mut out = resolved;
            for part in missing.iter().rev() {
                out.push(part);
            }
            return out;
        }
        current = parent;
    }
    path.to_path_buf()
}

/// Pure evaluation of a probe against the intent. Separate from process
/// execution so every rule has a fixture test.
pub fn evaluate_probe(
    probe: &RuntimeProbe,
    dir: &Path,
    intent: &VerificationIntent,
) -> Result<VerificationReport, VerifyError> {
    let mut notes = Vec::new();
    let backend = probe.basis.backend;

    // The accelerator the user asked for must be the one that loaded, not a
    // silent CPU fallback.
    match intent.expected_backend {
        Some(expected) if expected != backend => {
            return Err(VerifyError::BackendMismatch {
                expected,
                found: backend,
            });
        }
        None if intent.kind == InstallKind::Prebuilt => notes.push(
            "No evidence rules exist for this accelerator; the loaded backend was not checked."
                .to_string(),
        ),
        _ => {}
    }

    // Every backend plugin must be part of the directory under verification.
    let root = canonical(dir);
    for loaded in &probe.parsed.loaded_backends {
        if !canonical(&loaded.path).starts_with(&root) {
            return Err(VerifyError::PluginOutsideRuntime(loaded.path.clone()));
        }
    }

    if let Some(pinned) = &intent.pinned_commit {
        let found = probe.parsed.build_commit.clone();
        let matches = found
            .as_deref()
            .is_some_and(|short| short.len() >= 7 && pinned.starts_with(short));
        if !matches {
            return Err(VerifyError::CommitMismatch {
                pinned: pinned.clone(),
                found,
            });
        }
    }

    if intent.require_fa_all {
        match probe.basis.fa_pairs.confirmed_value() {
            Some(pairs) => {
                let all = compiled_fa_pairs("all")
                    .map_err(|e| VerifyError::FaQuantsNotAll(e.to_string()))?;
                if pairs != &all {
                    return Err(VerifyError::FaQuantsNotAll(format!(
                        "{} of {} pairs compiled",
                        pairs.len(),
                        all.len()
                    )));
                }
            }
            None => {
                return Err(VerifyError::FaQuantsNotAll(
                    "no confirmed FA_QUANTS record".to_string(),
                ));
            }
        }
    }

    if let Some(device) = &intent.device_id {
        let enumerated = probe
            .basis
            .devices
            .iter()
            .any(|line| line.split(" : ").next().unwrap_or(line).trim() == device);
        if !enumerated {
            return Err(VerifyError::DeviceNotEnumerated(device.clone()));
        }
    }

    Ok(VerificationReport {
        backend,
        build_number: probe.parsed.build_number,
        build_commit: probe.parsed.build_commit.clone(),
        identity_fingerprint: probe.basis.runtime_identity.clone(),
        notes,
        evidence_unavailable: false,
    })
}

/// `--help` fallback for a prebuilt/rollback runtime whose probe output we do
/// not recognize. The runtime still has to run; what it can prove stays
/// `Unavailable`.
async fn help_runs(exe: &Path) -> Result<(), String> {
    let output = tokio::time::timeout(VERIFY_PROBE_TIMEOUT, async {
        let mut cmd = tokio::process::Command::new(exe);
        cmd.arg("--help")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        crate::platform::no_window_tokio(&mut cmd);
        cmd.output().await
    })
    .await
    .map_err(|_| "--help timed out".to_string())?
    .map_err(|e| format!("spawn error: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(if stderr.is_empty() {
            format!("exit status {:?}", output.status.code())
        } else {
            format!("exit status {:?}: {stderr}", output.status.code())
        })
    }
}

/// Verify the runtime in `dir` at the staged stage (bench cross-check
/// included). The caller has already prepared it.
pub async fn verify_runtime(
    dir: &Path,
    exe_name: &str,
    intent: &VerificationIntent,
) -> Result<VerificationReport, VerifyError> {
    verify_runtime_at(dir, exe_name, intent, VerifyStage::Staged).await
}

/// Verify the runtime in `dir`. The caller has already prepared it.
pub async fn verify_runtime_at(
    dir: &Path,
    exe_name: &str,
    intent: &VerificationIntent,
    stage: VerifyStage,
) -> Result<VerificationReport, VerifyError> {
    let exe = dir.join(exe_name);
    if !exe.is_file() {
        return Err(VerifyError::MissingExecutable(exe));
    }
    let mut report =
        match probe_runtime_with_timeout(&exe, EnvPolicy::Sanitized, VERIFY_PROBE_TIMEOUT).await {
            Ok(probe) => evaluate_probe(&probe, dir, intent)?,
            Err(ProbeError::Spawn(e)) => return Err(VerifyError::CannotRun(e)),
            Err(other) => {
                // The runtime started but we could not read its evidence.
                if intent.kind == InstallKind::Source {
                    return Err(VerifyError::EvidenceRequired(other.to_string()));
                }
                if intent.pinned_commit.is_some() || intent.require_fa_all {
                    return Err(VerifyError::EvidenceRequired(other.to_string()));
                }
                help_runs(&exe).await.map_err(VerifyError::CannotRun)?;
                VerificationReport {
                    backend: Backend::Unknown,
                    build_number: None,
                    build_commit: None,
                    identity_fingerprint: None,
                    notes: vec![format!(
                        "{EVIDENCE_NOTE_PREFIX} ({other}); only --help was verified."
                    )],
                    evidence_unavailable: true,
                }
            }
        };
    // Plan §4.6.1: both built executables carry the pinned commit. Check
    // llama-bench alongside llama-server once, before the swap; a prebuilt set
    // that does not ship it is noted, never failed.
    if stage == VerifyStage::Staged
        && let Some(note) = check_bench_commit(dir, intent).await?
    {
        report.notes.push(note);
    }
    Ok(report)
}

/// Extract the commit from llama.cpp's `--version` line
/// (`version: 0.0.0-dev (build 1234, commit 0123456789ab)`).
///
/// Anchored on the `commit` keyword. An unanchored "first hex run" would pick
/// up compiler dates (`built with AppleClang 21.0.0.21000101`) or a future
/// build number, and a metadata-less build's `commit unknown` would then
/// surface as a mismatch. `-dirty` is tolerated; anything unrecognized yields
/// `None`, which callers treat as a note rather than a failure.
fn commit_token(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    for (idx, _) in lower.match_indices("commit") {
        // `precommit`/`decommit` are not the keyword.
        if lower[..idx]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        let rest = lower[idx + "commit".len()..].trim_start_matches([' ', ':', '=']);
        let token: String = rest
            .chars()
            .take_while(|c| c.is_ascii_hexdigit())
            .take(41)
            .collect();
        if (7..=40).contains(&token.len()) {
            return Some(token);
        }
    }
    None
}

/// Commit cross-check for `llama-bench` (plan §4.6.1 "check both built
/// executables"). Runs at the staged stage only: the swapped set is the same
/// bytes, and a parsing miss after a completed swap would roll back a
/// verified install.
///
/// `Ok(None)` means nothing to report. A *parsed* mismatch is an error; an
/// absent, unrunnable or unrecognized llama-bench is a note, never a failure.
async fn check_bench_commit(
    dir: &Path,
    intent: &VerificationIntent,
) -> Result<Option<String>, VerifyError> {
    let bench = dir.join(if cfg!(windows) {
        "llama-bench.exe"
    } else {
        "llama-bench"
    });
    // Without a pin (prebuilt, rollback) there is nothing to cross-check, so
    // no note either.
    let Some(pinned) = intent.pinned_commit.clone() else {
        return Ok(None);
    };
    if !bench.is_file() {
        return Ok(Some(
            "llama-bench is absent from this runtime set; its commit was not checked.".into(),
        ));
    }
    let pinned_is_hash =
        (7..=40).contains(&pinned.len()) && pinned.chars().all(|c| c.is_ascii_hexdigit());
    if !pinned_is_hash {
        return Ok(Some(format!(
            "llama-bench was not checked: the pinned commit {pinned:?} is not a hash."
        )));
    }
    let text =
        match crate::llama_runtime::probe::capture_version(&bench, VERIFY_PROBE_TIMEOUT).await {
            Ok(text) => text,
            Err(e) => {
                return Ok(Some(format!(
                    "llama-bench could not be run to confirm its commit ({e})."
                )));
            }
        };
    let Some(found) = commit_token(&text) else {
        return Ok(Some(
            "llama-bench ran but its commit could not be read; the check was skipped.".into(),
        ));
    };
    if !(pinned.starts_with(&found) || found.starts_with(&pinned)) {
        return Err(VerifyError::BenchCommitMismatch {
            pinned,
            found: Some(found),
        });
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_runtime::probe::{
        LoadedBackend, ParsedProbe, kv_basis_from_probe, parse_probe_output,
    };

    const MODEL: &str = "/nonexistent/foundry-probe-xyz.gguf";

    #[test]
    fn commit_token_is_anchored_on_the_commit_keyword() {
        assert_eq!(
            commit_token("version: 0.0.0-dev (build 11719, commit 0123456789)"),
            Some("0123456789".into())
        );
        assert_eq!(
            commit_token("version: 0.0.0-dev (build 11719, commit abc1234-dirty)"),
            Some("abc1234".into())
        );
        // A compiler date or build number must not be mistaken for a commit.
        assert_eq!(
            commit_token(
                "version: 0.0.0-dev (build 11719, commit unknown)\nbuilt with AppleClang 21.0.0.21000101"
            ),
            None
        );
        assert_eq!(commit_token("built with AppleClang 21.0.0.21000101"), None);
        assert_eq!(
            commit_token("commit 0123456789abcdef0123456789abcdef01234567"),
            Some("0123456789abcdef0123456789abcdef01234567".into())
        );
        // Too short or too long is not a commit.
        assert_eq!(commit_token("commit abc"), None);
        assert_eq!(
            commit_token("commit 0123456789abcdef0123456789abcdef012345678"),
            None
        );
        assert_eq!(commit_token("no hex here"), None);
    }

    #[cfg(unix)]
    fn write_bench(dir: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;
        let bench = dir.join("llama-bench");
        std::fs::write(&bench, body).unwrap();
        std::fs::set_permissions(&bench, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unparseable_bench_commit_is_a_note_never_a_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut intent = VerificationIntent::prebuilt("metal");
        intent.pinned_commit = Some("deadbeef00000000000000000000000000000000".into());

        // A metadata-less build says `commit unknown`; the compiler date must
        // not be mistaken for a commit, and the result is a note.
        write_bench(
            dir.path(),
            "#!/bin/sh\necho 'version: 0.0.0-dev (build 1, commit unknown)'\necho 'built with AppleClang 21.0.0.21000101'\n",
        );
        let note = check_bench_commit(dir.path(), &intent).await.unwrap();
        assert!(note.is_some_and(|n| n.contains("could not be read")));

        // A short matching commit passes silently.
        write_bench(
            dir.path(),
            "#!/bin/sh\necho 'version: 0.0.0-dev (build 1, commit deadbee)'\n",
        );
        assert!(
            check_bench_commit(dir.path(), &intent)
                .await
                .unwrap()
                .is_none()
        );

        // A parsed, different commit is a failure that names llama-bench.
        write_bench(
            dir.path(),
            "#!/bin/sh\necho 'version: 0.0.0-dev (build 1, commit 1111111)'\n",
        );
        let err = check_bench_commit(dir.path(), &intent).await.unwrap_err();
        assert!(matches!(err, VerifyError::BenchCommitMismatch { .. }));
        assert!(err.to_string().contains("llama-bench"));
    }

    fn probe_for(dir: &Path, text: &str) -> RuntimeProbe {
        let rewritten = text.replace("/opt/llama", &dir.to_string_lossy());
        let parsed = parse_probe_output(&rewritten, MODEL);
        let basis = kv_basis_from_probe(&parsed, None);
        RuntimeProbe {
            parsed,
            identity: None,
            basis,
        }
    }

    const METAL: &str = include_str!("../../tests/fixtures/llama_probe/metal_missing_model.txt");

    fn metal(dir: &Path) -> RuntimeProbe {
        probe_for(dir, METAL)
    }

    #[test]
    fn release_names_map_to_expected_backends() {
        assert_eq!(backend_for_release_name("metal"), Some(Backend::Metal));
        assert_eq!(backend_for_release_name("cuda13"), Some(Backend::Cuda));
        assert_eq!(backend_for_release_name("CUDA12"), Some(Backend::Cuda));
        assert_eq!(backend_for_release_name("vulkan"), Some(Backend::Vulkan));
        assert_eq!(backend_for_release_name("cpu"), Some(Backend::Cpu));
        assert_eq!(backend_for_release_name("hip"), None);
        assert_eq!(backend_for_release_name("sycl"), None);
    }

    #[test]
    fn intended_backend_must_be_the_one_that_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let probe = metal(dir.path());
        assert!(evaluate_probe(&probe, dir.path(), &VerificationIntent::prebuilt("metal")).is_ok());
        // A CUDA install that came up on Metal/CPU is a failure, not a pass.
        assert_eq!(
            evaluate_probe(&probe, dir.path(), &VerificationIntent::prebuilt("cuda13")),
            Err(VerifyError::BackendMismatch {
                expected: Backend::Cuda,
                found: Backend::Metal
            })
        );
    }

    #[test]
    fn cpu_fallback_for_an_accelerated_install_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let text = "load_backend: loaded CPU backend from /opt/llama/libggml-cpu.so\ncmn common_param: system_info: n = 1 | CPU : SSE3 = 1 |\n";
        let probe = probe_for(dir.path(), text);
        assert_eq!(probe.basis.backend, Backend::Cpu);
        assert!(matches!(
            evaluate_probe(&probe, dir.path(), &VerificationIntent::prebuilt("metal")),
            Err(VerifyError::BackendMismatch { .. })
        ));
        // ...while an explicit CPU install passes.
        assert!(evaluate_probe(&probe, dir.path(), &VerificationIntent::prebuilt("cpu")).is_ok());
    }

    #[test]
    fn a_backend_loaded_from_outside_the_runtime_directory_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let mut probe = metal(dir.path());
        probe.parsed.loaded_backends = vec![LoadedBackend {
            name: "MTL".into(),
            path: other.path().join("libggml-metal.so"),
        }];
        let err =
            evaluate_probe(&probe, dir.path(), &VerificationIntent::prebuilt("metal")).unwrap_err();
        assert!(
            matches!(err, VerifyError::PluginOutsideRuntime(_)),
            "{err:?}"
        );
    }

    #[test]
    fn pinned_commit_must_prefix_match_the_reported_build() {
        let dir = tempfile::tempdir().unwrap();
        let probe = metal(dir.path());
        let mut intent = VerificationIntent::prebuilt("metal");
        intent.kind = InstallKind::Source;
        intent.pinned_commit = Some("5aeac37a3b1c2d3e4f5061728394a5b6c7d8e9f0".into());
        assert!(evaluate_probe(&probe, dir.path(), &intent).is_ok());
        intent.pinned_commit = Some("deadbeef00000000000000000000000000000000".into());
        assert!(matches!(
            evaluate_probe(&probe, dir.path(), &intent),
            Err(VerifyError::CommitMismatch { .. })
        ));
        // A runtime that reports no commit cannot satisfy a pin.
        let mut bare = metal(dir.path());
        bare.parsed.build_commit = None;
        intent.pinned_commit = Some("5aeac37a3b1c2d3e4f5061728394a5b6c7d8e9f0".into());
        assert!(matches!(
            evaluate_probe(&bare, dir.path(), &intent),
            Err(VerifyError::CommitMismatch { found: None, .. })
        ));
    }

    #[test]
    fn source_cuda_requires_all_forty_nine_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let cuda = include_str!("../../tests/fixtures/llama_probe/cuda_missing_model.txt")
            .replace("D:\\AI\\loaders\\llamacpp", "/opt/llama");
        let mut intent = VerificationIntent::prebuilt("cuda13");
        intent.kind = InstallKind::Source;
        intent.require_fa_all = true;
        let probe = probe_for(dir.path(), &cuda);
        // Plugin paths use Windows separators in the real fixture; relocate them for the test.
        let mut probe = probe;
        probe.parsed.loaded_backends.clear();
        assert!(
            evaluate_probe(&probe, dir.path(), &intent).is_ok(),
            "FA_QUANTS=all passes"
        );

        let listed = cuda.replace("FA_QUANTS = all", "FA_QUANTS = q8_0-q4_0");
        let mut probe = probe_for(dir.path(), &listed);
        probe.parsed.loaded_backends.clear();
        assert!(matches!(
            evaluate_probe(&probe, dir.path(), &intent),
            Err(VerifyError::FaQuantsNotAll(_))
        ));

        let absent = cuda.replace("FA_QUANTS = all |", "");
        let mut probe = probe_for(dir.path(), &absent);
        probe.parsed.loaded_backends.clear();
        assert!(matches!(
            evaluate_probe(&probe, dir.path(), &intent),
            Err(VerifyError::FaQuantsNotAll(_))
        ));
    }

    #[test]
    fn selected_device_must_be_enumerated() {
        let dir = tempfile::tempdir().unwrap();
        let probe = metal(dir.path());
        let mut intent = VerificationIntent::prebuilt("metal");
        intent.device_id = Some("MTL0".into());
        assert!(evaluate_probe(&probe, dir.path(), &intent).is_ok());
        intent.device_id = Some("MTL7".into());
        assert_eq!(
            evaluate_probe(&probe, dir.path(), &intent),
            Err(VerifyError::DeviceNotEnumerated("MTL7".into()))
        );
    }

    #[test]
    fn unknown_accelerators_get_a_note_not_a_false_pass() {
        let dir = tempfile::tempdir().unwrap();
        let probe = metal(dir.path());
        let report =
            evaluate_probe(&probe, dir.path(), &VerificationIntent::prebuilt("hip")).unwrap();
        assert!(report.notes.iter().any(|n| n.contains("not checked")));
    }

    #[test]
    fn preparation_sets_tool_executable_bits_and_never_runs_anything() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("llama-server"), b"x").unwrap();
        std::fs::write(dir.path().join("llama-bench"), b"x").unwrap();
        std::fs::write(dir.path().join("libggml.dylib"), b"x").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("data"), b"x").unwrap();
        let inventory = prepare_runtime_dir(dir.path(), "llama-server").unwrap();
        assert_eq!(inventory.files, 4);
        assert!(inventory.executable.is_some());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |n: &str| {
                std::fs::metadata(dir.path().join(n))
                    .unwrap()
                    .permissions()
                    .mode()
            };
            assert_eq!(mode("llama-server") & 0o111, 0o111);
            assert_eq!(mode("llama-bench") & 0o111, 0o111);
            assert_eq!(
                mode("libggml.dylib") & 0o111,
                0,
                "libraries keep their mode"
            );
        }
    }

    #[test]
    fn missing_executable_is_reported_before_anything_runs() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt
            .block_on(verify_runtime(
                dir.path(),
                "llama-server",
                &VerificationIntent::prebuilt("metal"),
            ))
            .unwrap_err();
        assert!(matches!(err, VerifyError::MissingExecutable(_)));
        let _ = ParsedProbe::default();
    }

    #[cfg(unix)]
    mod process {
        use super::*;
        use std::os::unix::fs::PermissionsExt;

        fn fake(dir: &Path, body: &str) {
            let path = dir.join("llama-server");
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        const GOOD: &str = r#"
case "$1" in
  --help) exit 0 ;;
esac
echo "load_backend: loaded MTL backend from $(dirname "$0")/libggml-metal.so" >&2
echo "cmn common_params_print_info: build 11719 (5aeac37a3) with clang" >&2
echo "cmn common_param: system_info: n = 1 | MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 |" >&2
echo "E gguf_init_from_file: failed to open GGUF file '$4' (No such file or directory)" >&2
exit 1
"#;

        #[tokio::test]
        async fn a_recognized_runtime_verifies_with_the_sanitized_environment() {
            let dir = tempfile::tempdir().unwrap();
            fake(dir.path(), GOOD);
            std::fs::write(dir.path().join("libggml-metal.so"), b"p").unwrap();
            let report = verify_runtime(
                dir.path(),
                "llama-server",
                &VerificationIntent::prebuilt("metal"),
            )
            .await
            .unwrap();
            assert_eq!(report.backend, Backend::Metal);
            assert_eq!(report.build_number, Some(11719));
            assert!(!report.evidence_unavailable);
        }

        #[tokio::test]
        async fn an_unrecognized_prebuilt_that_runs_is_unavailable_evidence_not_a_pass_of_the_backend()
         {
            let dir = tempfile::tempdir().unwrap();
            fake(
                dir.path(),
                "case \"$1\" in --help) exit 0;; esac\necho nonsense >&2\nexit 3",
            );
            let report = verify_runtime(
                dir.path(),
                "llama-server",
                &VerificationIntent::prebuilt("metal"),
            )
            .await
            .unwrap();
            assert!(report.evidence_unavailable);
            assert_eq!(report.backend, Backend::Unknown);
            assert!(report.notes[0].contains("only --help was verified"));
        }

        #[tokio::test]
        async fn source_builds_and_pinned_runs_require_recognized_evidence() {
            let dir = tempfile::tempdir().unwrap();
            fake(
                dir.path(),
                "case \"$1\" in --help) exit 0;; esac\necho nonsense >&2\nexit 3",
            );
            let mut intent = VerificationIntent::prebuilt("metal");
            intent.kind = InstallKind::Source;
            assert!(matches!(
                verify_runtime(dir.path(), "llama-server", &intent).await,
                Err(VerifyError::EvidenceRequired(_))
            ));
        }

        #[tokio::test]
        async fn a_runtime_that_cannot_run_fails_even_for_prebuilt() {
            let dir = tempfile::tempdir().unwrap();
            fake(dir.path(), "echo boom >&2\nexit 9");
            assert!(matches!(
                verify_runtime(
                    dir.path(),
                    "llama-server",
                    &VerificationIntent::prebuilt("metal")
                )
                .await,
                Err(VerifyError::CannotRun(_))
            ));
        }

        #[tokio::test]
        async fn a_plugin_from_another_directory_fails_the_real_process_path_too() {
            let dir = tempfile::tempdir().unwrap();
            let elsewhere = tempfile::tempdir().unwrap();
            let body = GOOD.replace("$(dirname \"$0\")", &elsewhere.path().to_string_lossy());
            fake(dir.path(), &body);
            std::fs::write(elsewhere.path().join("libggml-metal.so"), b"p").unwrap();
            assert!(matches!(
                verify_runtime(
                    dir.path(),
                    "llama-server",
                    &VerificationIntent::prebuilt("metal")
                )
                .await,
                Err(VerifyError::PluginOutsideRuntime(_))
            ));
        }
    }
}
