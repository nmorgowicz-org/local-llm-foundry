#![allow(clippy::collapsible_if)]

use crate::inference::rapid_mlx::capabilities as cap;
use crate::inference::rapid_mlx::runtime::RuntimeSource;
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeSet;
use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

pub const MINIMUM_VERIFIED_VERSION: (u64, u64, u64) = (0, 10, 9);
/// Latest published Rapid-MLX release selected as this repository's baseline.
/// Every installed runtime still receives the bounded live capability probe; this
/// is release metadata, not a version allowlist.
pub const LATEST_QUALIFIED_VERSION_TEXT: &str = "0.15.4";
pub const QUALIFIED_ROLLBACK_VERSION_TEXT: &str = "0.10.9";
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_PROBE_OUTPUT_BYTES: usize = 256 * 1024;
const EXECUTABLE_BUSY_RETRY_WINDOW: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityState {
    /// The managed runtime's CLI version and live interface capabilities passed.
    /// This does not authenticate the artifact or qualify it for pointer activation;
    /// the Phase 6 installer owns provenance and staged runtime gates.
    Verified,
}

impl CompatibilityState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Verified => "verified",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServeCapabilities {
    flags: BTreeSet<String>,
}

impl ServeCapabilities {
    pub fn from_help(help: &str) -> Self {
        let flags = Self::parse_flags(help);
        Self { flags }
    }

    pub fn from_flags(flags: &[String]) -> Self {
        Self {
            flags: flags.iter().cloned().collect(),
        }
    }

    fn parse_flags(help: &str) -> BTreeSet<String> {
        help.split_whitespace()
            .filter_map(|token| {
                let token = token.trim_matches(|c: char| matches!(c, ',' | '[' | ']' | '(' | ')'));
                let flag = token.split_once('=').map_or(token, |(flag, _)| flag);
                flag.starts_with("--").then(|| flag.to_string())
            })
            .collect()
    }

    pub fn contains(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    pub fn require(&self, flag: &str) -> Result<()> {
        if self.contains(flag) {
            Ok(())
        } else {
            Err(anyhow!(
                "Installed Rapid-MLX does not support configured option {flag}; select a compatible runtime or remove that option"
            ))
        }
    }

    pub fn verified_baseline() -> Self {
        Self::from_help(
            "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash --speculative-config",
        )
    }
}

#[derive(Debug, Clone)]
pub struct CompatibilityProfile {
    pub state: CompatibilityState,
    pub version: String,
    pub capabilities: ServeCapabilities,
}

impl CompatibilityProfile {
    pub fn verified_baseline() -> Self {
        Self {
            state: CompatibilityState::Verified,
            version: LATEST_QUALIFIED_VERSION_TEXT.to_string(),
            capabilities: ServeCapabilities::verified_baseline(),
        }
    }
}

pub async fn probe(binary: &Path, source: RuntimeSource) -> Result<CompatibilityProfile> {
    // Dependencies can change without changing the executable. Re-probe so an
    // identity-only cache never supplies stale capability evidence.
    probe_with_policy(binary, source, false, PROBE_TIMEOUT, MAX_PROBE_OUTPUT_BYTES).await
}

/// Probe a managed runtime selected from immutable, published release metadata.
///
/// Normal discovery remains stable-only. The runtime manager may opt a published
/// prerelease into the same minimum-version and capability gates, but local-version
/// builds remain ineligible for managed activation.
pub async fn probe_published_managed_release(
    binary: &Path,
    allow_prerelease: bool,
) -> Result<CompatibilityProfile> {
    probe_with_policy(
        binary,
        RuntimeSource::Managed,
        allow_prerelease,
        PROBE_TIMEOUT,
        MAX_PROBE_OUTPUT_BYTES,
    )
    .await
}

#[cfg(test)]
async fn probe_with_limits(
    binary: &Path,
    source: RuntimeSource,
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<CompatibilityProfile> {
    probe_with_policy(binary, source, false, timeout, max_output_bytes).await
}

async fn probe_with_policy(
    binary: &Path,
    source: RuntimeSource,
    allow_published_prerelease: bool,
    timeout: Duration,
    max_output_bytes: usize,
) -> Result<CompatibilityProfile> {
    let version_output =
        run_probe(binary, &["--version"], timeout, max_output_bytes, "version").await?;
    if !version_output.status.success() {
        anyhow::bail!(
            "Rapid-MLX version probe failed for {} with status {}",
            binary.display(),
            version_output.status
        );
    }
    let version_text = output_text(&version_output.stdout, &version_output.stderr);
    let parsed_version = parse_version(&version_text).ok_or_else(|| {
        anyhow!(
            "Rapid-MLX returned an unrecognized version from {}: {}",
            binary.display(),
            version_text.trim()
        )
    })?;
    let version = parsed_version.numbers;
    if version < MINIMUM_VERIFIED_VERSION {
        anyhow::bail!(
            "Rapid-MLX {} is unsupported; version 0.10.9 or newer is required",
            format_version(version)
        );
    }
    if source == RuntimeSource::Managed
        && !parsed_version.stable
        && !(allow_published_prerelease && parsed_version.prerelease)
    {
        anyhow::bail!(
            "Managed Rapid-MLX runtime is {}; the managed channel requires a stable release at version {} or newer (latest directly qualified: {}). Configure this build explicitly as a user-owned custom runtime to probe it provisionally",
            version_text.trim(),
            QUALIFIED_ROLLBACK_VERSION_TEXT,
            LATEST_QUALIFIED_VERSION_TEXT,
        );
    }

    let help_output = run_probe(
        binary,
        &["serve", "--help"],
        timeout,
        max_output_bytes,
        "capability",
    )
    .await?;
    if !help_output.status.success() {
        anyhow::bail!(
            "Rapid-MLX capability probe failed for {} with status {}",
            binary.display(),
            help_output.status
        );
    }
    let help = output_text(&help_output.stdout, &help_output.stderr);
    let capabilities = ServeCapabilities::from_help(&help);
    let required_capabilities: &[&str] = if source == RuntimeSource::Managed {
        // All verified_baseline flags must be present in Managed runtime upgrades.
        // --max-cache-blocks removed: replaced by --cache-memory-mb / --enable-prefix-cache.
        &[
            "--host",
            "--port",
            "--log-level",
            "--served-model-name",
            "--timeout",
            "--enable-prefix-cache",
            "--disable-prefix-cache",
            "--cache-memory-mb",
            "--hybrid-cache-entries",
            "--kv-disk-checkpoint-interval",
            "--tool-call-parser",
            "--reasoning-parser",
            "--enable-auto-tool-choice",
            "--no-thinking",
            "--reasoning",
            "--force-hybrid",
            "--no-hybrid",
            "--prefill-step-size",
            "--pflash",
        ]
    } else {
        &["--host", "--port"]
    };
    for required in required_capabilities {
        capabilities.require(required).with_context(|| {
            format!(
                "Rapid-MLX {} failed its required serve capability probe",
                format_version(version)
            )
        })?;
    }

    // Refresh the full dependency/extra snapshot after every baseline probe.
    let _ = cap::generate_snapshot(binary, source).await;

    Ok(CompatibilityProfile {
        // A successful bounded baseline probe qualifies every supported source;
        // optional uncertainty remains per-feature rather than globally.
        state: CompatibilityState::Verified,
        version: parsed_version.exact,
        capabilities,
    })
}

struct ProbeOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

async fn run_probe(
    binary: &Path,
    args: &[&str],
    timeout: Duration,
    max_output_bytes: usize,
    name: &str,
) -> Result<ProbeOutput> {
    let mut command = Command::new(binary);
    command
        .args(args)
        .kill_on_drop(true)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let started = tokio::time::Instant::now();
    let mut child = loop {
        match command.spawn() {
            Ok(child) => break child,
            Err(error)
                if executable_is_temporarily_busy(&error)
                    && started.elapsed() < EXECUTABLE_BUSY_RETRY_WINDOW =>
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("Failed to execute {} {} probe", binary.display(), name)
                });
            }
        }
    };
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("Failed to capture Rapid-MLX {name} probe stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("Failed to capture Rapid-MLX {name} probe stderr"))?;

    let capture = async move {
        let stdout_reader = read_bounded(stdout, max_output_bytes);
        let stderr_reader = read_bounded(stderr, max_output_bytes);
        let (stdout, stderr, status) = tokio::join!(stdout_reader, stderr_reader, child.wait());
        Ok::<_, anyhow::Error>(ProbeOutput {
            status: status.context("Failed waiting for Rapid-MLX probe")?,
            stdout: stdout?,
            stderr: stderr?,
        })
    };

    tokio::time::timeout(timeout, capture)
        .await
        .with_context(|| {
            format!(
                "Rapid-MLX {name} probe timed out after {:.1}s for {}",
                timeout.as_secs_f64(),
                binary.display()
            )
        })?
}

fn executable_is_temporarily_busy(error: &std::io::Error) -> bool {
    // Linux reports ETXTBSY when a freshly-created executable is briefly still
    // open for writing. This can happen on overlay filesystems used by CI and
    // during an atomic runtime replacement. It is safe to retry the spawn; all
    // other errors remain fail-closed and are returned immediately.
    #[cfg(unix)]
    {
        error.raw_os_error() == Some(26) // ETXTBSY
    }
    #[cfg(not(unix))]
    {
        let _ = error;
        false
    }
}

async fn read_bounded<R>(reader: R, max_output_bytes: usize) -> Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut bytes = Vec::with_capacity(max_output_bytes.min(8192));
    reader
        .take((max_output_bytes + 1) as u64)
        .read_to_end(&mut bytes)
        .await
        .context("Failed reading Rapid-MLX probe output")?;
    if bytes.len() > max_output_bytes {
        anyhow::bail!(
            "Rapid-MLX probe output exceeded the {} byte safety limit",
            max_output_bytes
        );
    }
    Ok(bytes)
}

/// Joins both streams because which one carries `--help` is not a stable contract.
///
/// `rapid-mlx 0.11.1` writes all 27 KB of `serve --help` to stdout and nothing to stderr.
/// Reading only one stream yields an empty capability set that looks like "this runtime
/// supports nothing" rather than like a failed probe — see `capabilities_probe_reads_the`
/// `_stream_help_actually_uses` in `rapid_mlx_runtime`.
pub(crate) fn output_text(stdout: &[u8], stderr: &[u8]) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
        (false, false) => format!("{}\n{}", stdout.trim(), stderr.trim()),
        (false, true) => stdout.trim().to_string(),
        (true, false) => stderr.trim().to_string(),
        (true, true) => String::new(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedVersion {
    numbers: (u64, u64, u64),
    stable: bool,
    prerelease: bool,
    exact: String,
}

fn parse_version(text: &str) -> Option<ParsedVersion> {
    let bytes = text.as_bytes();
    for start in 0..bytes.len() {
        if !bytes[start].is_ascii_digit() {
            continue;
        }
        let mut cursor = start;
        let Some(major) = parse_number(bytes, &mut cursor) else {
            continue;
        };
        if bytes.get(cursor) != Some(&b'.') {
            continue;
        }
        cursor += 1;
        let Some(minor) = parse_number(bytes, &mut cursor) else {
            continue;
        };
        if bytes.get(cursor) != Some(&b'.') {
            continue;
        }
        cursor += 1;
        let Some(patch) = parse_number(bytes, &mut cursor) else {
            continue;
        };
        let suffix_end = bytes[cursor..]
            .iter()
            .position(|byte| byte.is_ascii_whitespace())
            .map_or(bytes.len(), |offset| cursor + offset);
        let suffix = &bytes[cursor..suffix_end];
        let stable = !suffix.first().is_some_and(|next| {
            next.is_ascii_alphanumeric() || matches!(next, b'.' | b'+' | b'-' | b'_')
        });
        let prerelease = !stable && is_prerelease_suffix(suffix);
        return Some(ParsedVersion {
            numbers: (major, minor, patch),
            stable,
            prerelease,
            exact: String::from_utf8_lossy(&bytes[start..suffix_end]).into_owned(),
        });
    }
    None
}

fn is_prerelease_suffix(suffix: &[u8]) -> bool {
    let suffix = suffix.strip_prefix(b"-").unwrap_or(suffix);
    let Some(first) = suffix.first() else {
        return false;
    };
    if !first.is_ascii_alphabetic()
        || suffix.contains(&b'+')
        || !suffix
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return false;
    }
    let lowercase = String::from_utf8_lossy(suffix).to_ascii_lowercase();
    ["a", "alpha", "b", "beta", "rc", "dev"]
        .iter()
        .any(|prefix| lowercase.starts_with(prefix))
}

fn parse_number(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    let start = *cursor;
    while bytes.get(*cursor).is_some_and(u8::is_ascii_digit) {
        *cursor += 1;
    }
    (start != *cursor)
        .then(|| {
            std::str::from_utf8(&bytes[start..*cursor])
                .ok()?
                .parse()
                .ok()
        })
        .flatten()
}

fn format_version(version: (u64, u64, u64)) -> String {
    format!("{}.{}.{}", version.0, version.1, version.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cli_version_variants() {
        let stable = |numbers| {
            Some(ParsedVersion {
                numbers,
                stable: true,
                prerelease: false,
                exact: format!("{}.{}.{}", numbers.0, numbers.1, numbers.2),
            })
        };
        assert_eq!(parse_version("rapid-mlx 0.10.9"), stable((0, 10, 9)));
        assert_eq!(parse_version("rapid-mlx 0.10.10"), stable((0, 10, 10)));
        assert_eq!(
            parse_version("Rapid-MLX version v0.11.2\n"),
            stable((0, 11, 2))
        );
        assert_eq!(parse_version("development"), None);
        for version in [
            "0.10.10.dev1",
            "0.10.10rc1",
            "0.10.10+local",
            "0.10.10-local",
        ] {
            assert!(!parse_version(version).unwrap().stable, "{version}");
        }
        assert!(parse_version("0.10.11rc1").unwrap().prerelease);
        assert!(parse_version("0.10.11-beta.2").unwrap().prerelease);
        assert!(!parse_version("0.10.11+local").unwrap().prerelease);
    }

    #[test]
    fn help_tokens_are_exact() {
        let capabilities = ServeCapabilities::from_help(
            "--host TEXT --port INTEGER --timeout=1800 --max-cache-blocks INTEGER",
        );
        assert!(capabilities.contains("--timeout"));
        assert!(capabilities.contains("--max-cache-blocks"));
        assert!(!capabilities.contains("--request-timeout"));
        assert!(!capabilities.contains("--max-blocks"));
    }

    #[cfg(unix)]
    fn fixture_runtime(version: &str, help: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("rapid-mlx");
        std::fs::write(
            &binary,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'rapid-mlx {version}'; exit 0; fi\nif [ \"$1\" = \"serve\" ] && [ \"$2\" = \"--help\" ]; then echo '{help}'; exit 0; fi\nexit 2\n"
            ),
        )
        .unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        (dir, binary)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn live_probe_qualifies_healthy_external_profiles_without_global_warning() {
        let all_managed_flags = "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash";
        let (_dir, binary) = fixture_runtime("0.10.10", all_managed_flags);
        let profile = probe(&binary, RuntimeSource::Custom).await.unwrap();
        assert_eq!(profile.state, CompatibilityState::Verified);
        assert!(profile.capabilities.contains("--timeout"));

        let profile = probe(&binary, RuntimeSource::Managed).await.unwrap();
        assert_eq!(profile.state, CompatibilityState::Verified);
        assert_eq!(profile.version, "0.10.10");

        let (_dir, binary) = fixture_runtime("0.10.9", all_managed_flags);
        let profile = probe(&binary, RuntimeSource::Managed).await.unwrap();
        assert_eq!(profile.state, CompatibilityState::Verified);
        assert_eq!(profile.version, "0.10.9");

        let (_dir, binary) = fixture_runtime("0.10.11", all_managed_flags);
        let profile = probe(&binary, RuntimeSource::Managed).await.unwrap();
        assert_eq!(profile.state, CompatibilityState::Verified);
        assert_eq!(profile.version, "0.10.11");

        let (_dir, binary) = fixture_runtime("0.11.0", "--host --port");
        let profile = probe(&binary, RuntimeSource::Homebrew).await.unwrap();
        assert_eq!(profile.state, CompatibilityState::Verified);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[tokio::test]
    async fn managed_and_required_capability_probes_fail_closed() {
        let all_managed_flags = "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash";
        let (_dir, binary) = fixture_runtime("0.10.11", all_managed_flags);
        assert!(probe(&binary, RuntimeSource::Managed).await.is_ok());

        // Missing one required flag → fails
        let (_dir, binary) = fixture_runtime(
            "0.10.11",
            "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size",
        );
        let error = probe(&binary, RuntimeSource::Managed).await.unwrap_err();
        assert!(format!("{error:#}").contains("--pflash"));

        let (_dir, binary) = fixture_runtime("0.10.9", "--host");
        let error = probe(&binary, RuntimeSource::Custom).await.unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("--port"));

        for version in [
            "0.10.10.dev1",
            "0.10.10rc1",
            "0.10.10+local",
            "0.10.9+rollback-local",
        ] {
            let (_dir, binary) = fixture_runtime(version, "--host --port");
            let error = probe(&binary, RuntimeSource::Managed).await.unwrap_err();
            assert!(
                error.to_string().contains("requires a stable release"),
                "{version}: {error:#}"
            );

            let profile = probe(&binary, RuntimeSource::Custom).await.unwrap();
            assert_eq!(profile.state, CompatibilityState::Verified, "{version}");
        }

        for version in ["0.10.11", "0.10.10", "0.10.9"] {
            let (_dir, binary) = fixture_runtime(version, "--host --port --log-level");
            let error = probe(&binary, RuntimeSource::Managed).await.unwrap_err();
            assert!(format!("{error:#}").contains("--served-model-name"));
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn published_managed_prerelease_uses_the_full_managed_gate() {
        let all_managed_flags = "--host --port --log-level --served-model-name --timeout --enable-prefix-cache --disable-prefix-cache --cache-memory-mb --hybrid-cache-entries --kv-disk-checkpoint-interval --tool-call-parser --reasoning-parser --enable-auto-tool-choice --no-thinking --reasoning --force-hybrid --no-hybrid --prefill-step-size --pflash";
        let (_dir, binary) = fixture_runtime("0.10.11rc1", all_managed_flags);
        assert!(probe(&binary, RuntimeSource::Managed).await.is_err());
        let profile = probe_published_managed_release(&binary, true)
            .await
            .unwrap();
        assert_eq!(profile.state, CompatibilityState::Verified);
        assert_eq!(profile.version, "0.10.11rc1");

        let (_dir, binary) = fixture_runtime("0.10.11+local", all_managed_flags);
        assert!(
            probe_published_managed_release(&binary, true)
                .await
                .is_err()
        );

        let (_dir, binary) = fixture_runtime("0.10.11rc1", "--host --port");
        let error = probe_published_managed_release(&binary, true)
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("--log-level"));
    }

    #[test]
    fn qualified_versions_are_metadata_not_a_future_release_allowlist() {
        let current = CompatibilityProfile::verified_baseline();
        assert_eq!(current.version, "0.10.17");
        assert_eq!(LATEST_QUALIFIED_VERSION_TEXT, "0.10.17");
        assert_eq!(QUALIFIED_ROLLBACK_VERSION_TEXT, "0.10.9");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hung_probe_is_terminated_by_deadline() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("rapid-mlx");
        std::fs::write(&binary, "#!/bin/sh\nsleep 5\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let started = std::time::Instant::now();
        let error = probe_with_limits(
            &binary,
            RuntimeSource::Custom,
            Duration::from_millis(75),
            1024,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn oversized_probe_output_fails_at_bound() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("rapid-mlx");
        std::fs::write(
            &binary,
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then while :; do printf x; done; fi\n",
        )
        .unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let error = probe_with_limits(&binary, RuntimeSource::Custom, Duration::from_secs(5), 1024)
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("exceeded the 1024 byte safety limit"));
    }
}
