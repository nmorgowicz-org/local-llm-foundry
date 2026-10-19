//! Bounded runtime probe and loaded-closure identity.
//!
//! Runs `llama-server -lv 4 -m <unique nonexistent path>` in an owned
//! temporary directory. At trace verbosity llama-server prints the loaded
//! backend plugins, the enumerated devices and `llama_print_system_info()`
//! (including the CUDA `FA_QUANTS` feature) before it tries to load the model,
//! so no model is needed. See plan §4.6.1.
//!
//! The probe proves what the *loaded* backend reports. It is not a model
//! smoke test.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use crate::inference::kv_support::{
    Backend, Evidence, KV_BASIS_VERSION, KvSupportBasis, compiled_fa_pairs,
};

pub const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// Combined stdout+stderr cap.
pub const PROBE_MAX_OUTPUT: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvPolicy {
    /// Keep the caller's environment (launch parity for ordinary probes).
    Inherited,
    /// Minimal allowlisted environment with a system-only search path. Used by
    /// staged-candidate verification so build-tree/toolkit directories and
    /// library overrides cannot satisfy the loader.
    Sanitized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    Spawn(String),
    Timeout,
    OutputTooLarge,
    /// The process exited without the recognized model-missing diagnostic, or
    /// the output carried no system-info record.
    Unrecognized(String),
    Io(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "could not start the runtime probe: {e}"),
            Self::Timeout => write!(f, "the runtime probe timed out"),
            Self::OutputTooLarge => write!(f, "the runtime probe output exceeded its size limit"),
            Self::Unrecognized(e) => write!(f, "the runtime probe output was not recognized: {e}"),
            Self::Io(e) => write!(f, "runtime probe I/O error: {e}"),
        }
    }
}

impl std::error::Error for ProbeError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadedBackend {
    /// Registry name as printed by llama.cpp (`CUDA`, `MTL`, `CPU`, ...).
    pub name: String,
    pub path: PathBuf,
}

/// Feature groups from `system_info`, e.g. `CUDA -> {ARCHS: 1200, FA_QUANTS: all}`.
pub type FeatureGroups = BTreeMap<String, BTreeMap<String, String>>;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedProbe {
    pub loaded_backends: Vec<LoadedBackend>,
    pub devices: Vec<String>,
    pub features: FeatureGroups,
    pub system_info_seen: bool,
    pub model_missing_diagnostic: bool,
    /// Metal `has bfloat` (any device reporting false wins).
    pub metal_bf16: Option<bool>,
    pub build_number: Option<u32>,
    /// Abbreviated commit from `build N (abcdef123)`.
    pub build_commit: Option<String>,
}

/// Parse the `system_info` feature string into groups. Tokens are separated by
/// `|`; a token containing ` : ` starts a new group.
pub fn parse_system_info(info: &str) -> FeatureGroups {
    let mut groups = FeatureGroups::new();
    let mut current = String::new();
    for token in info.split('|') {
        let mut token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((group, rest)) = token.split_once(" : ") {
            current = group.trim().to_string();
            groups.entry(current.clone()).or_default();
            token = rest.trim();
        }
        if let Some((key, value)) = token.split_once(" = ") {
            groups
                .entry(current.clone())
                .or_default()
                .insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    groups
}

/// Parse probe output (stdout and stderr combined). Only recognized, complete
/// records are extracted.
pub fn parse_probe_output(output: &str, probe_model_path: &str) -> ParsedProbe {
    let mut parsed = ParsedProbe::default();
    for line in output.lines() {
        if let Some(index) = line.find("load_backend: loaded ") {
            let rest = &line[index + "load_backend: loaded ".len()..];
            if let Some((name, path)) = rest.split_once(" backend from ") {
                let path = path.trim();
                if !name.trim().is_empty() && !path.is_empty() {
                    parsed.loaded_backends.push(LoadedBackend {
                        name: name.trim().to_string(),
                        path: PathBuf::from(path),
                    });
                }
            }
        }
        if let Some(index) = line.find("system_info:") {
            let info = &line[index + "system_info:".len()..];
            parsed.system_info_seen = true;
            for (group, values) in parse_system_info(info) {
                parsed.features.entry(group).or_default().extend(values);
            }
        }
        if let Some(index) = line.find("common_param:") {
            let rest = line[index + "common_param:".len()..].trim_start();
            if let Some(device) = rest.strip_prefix("- ")
                && device.contains(" : ")
            {
                parsed.devices.push(device.trim().to_string());
            }
        }
        if line.contains("ggml_metal_device_init:")
            && let Some(index) = line.find("has bfloat")
            && let Some((_, value)) = line[index..].split_once('=')
        {
            match value.trim() {
                "true" => parsed.metal_bf16 = Some(parsed.metal_bf16.unwrap_or(true)),
                "false" => parsed.metal_bf16 = Some(false),
                _ => {}
            }
        }
        if parsed.build_number.is_none()
            && let Some(index) = line.find("common_params_print_info: build ")
        {
            let digits: String = line[index + "common_params_print_info: build ".len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            parsed.build_number = digits.parse().ok();
            let rest = &line[index + "common_params_print_info: build ".len() + digits.len()..];
            parsed.build_commit = rest
                .trim_start()
                .strip_prefix('(')
                .and_then(|r| r.split(')').next())
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty() && c.chars().all(|ch| ch.is_ascii_hexdigit()));
        }
        if line.contains("failed to open GGUF file") && line.contains(probe_model_path) {
            parsed.model_missing_diagnostic = true;
        }
    }
    parsed
}

fn backend_for_name(name: &str) -> Option<Backend> {
    match name.to_ascii_uppercase().as_str() {
        // The ungrouped leading system_info tokens (thread counts) are not a backend.
        "" => None,
        "CUDA" => Some(Backend::Cuda),
        "MTL" | "METAL" => Some(Backend::Metal),
        "VULKAN" => Some(Backend::Vulkan),
        "CPU" => Some(Backend::Cpu),
        // BLAS is an accelerator for matmul only; it never decides the backend.
        "BLAS" => None,
        // ROCm/MUSA/SYCL/etc.: an accelerator we have no K/V evidence for.
        _ => Some(Backend::Unknown),
    }
}

/// Decide the effective backend from loaded plugins and feature groups.
/// Accelerators win over CPU; unrecognized accelerators yield `Unknown`.
pub fn detect_backend(parsed: &ParsedProbe) -> Backend {
    let names = parsed
        .loaded_backends
        .iter()
        .map(|b| b.name.as_str())
        .chain(parsed.features.keys().map(String::as_str));
    let mut found: Vec<Backend> = names.filter_map(backend_for_name).collect();
    found.sort_by_key(|b| match b {
        Backend::Cuda => 0,
        Backend::Metal => 1,
        Backend::Vulkan => 2,
        Backend::Unknown => 3,
        Backend::Cpu => 4,
    });
    found.first().copied().unwrap_or(Backend::Unknown)
}

/// Build the K/V evidence basis from a parsed probe.
pub fn kv_basis_from_probe(
    parsed: &ParsedProbe,
    identity: Option<&RuntimeIdentity>,
) -> KvSupportBasis {
    let backend = detect_backend(parsed);
    let fa_pairs = if backend == Backend::Cuda {
        match parsed.features.get("CUDA").and_then(|g| g.get("FA_QUANTS")) {
            None => Evidence::unavailable("the CUDA backend reported no FA_QUANTS record"),
            Some(spec) => match compiled_fa_pairs(spec) {
                Ok(pairs) => Evidence::confirmed(pairs, "runtime probe: FA_QUANTS"),
                Err(e) => Evidence::unavailable(format!("FA_QUANTS record was not valid: {e}")),
            },
        }
    } else {
        Evidence::unavailable("compiled pair lists are probed only for CUDA")
    };
    KvSupportBasis {
        version: KV_BASIS_VERSION,
        backend,
        fa_pairs,
        runtime_identity: identity.map(RuntimeIdentity::fingerprint),
        build_number: parsed.build_number,
        metal_bf16: match (backend, parsed.metal_bf16) {
            (Backend::Metal, Some(flag)) => {
                Evidence::confirmed(flag, "runtime probe: ggml_metal_device_init has bfloat")
            }
            (Backend::Metal, None) => {
                Evidence::unavailable("the Metal backend reported no bfloat flag")
            }
            _ => Evidence::unavailable("only probed for Metal"),
        },
        devices: parsed.devices.clone(),
    }
}

// ---------------------------------------------------------------------------
// Closure identity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileIdentity {
    pub path: String,
    pub sha256: String,
    pub size: u64,
    pub mtime_unix: u64,
}

/// Identity of the executable plus the loaded backend plugins and the
/// same-directory llama/ggml shared libraries it resolves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeIdentity {
    pub executable: FileIdentity,
    pub libraries: Vec<FileIdentity>,
}

fn stat_identity(path: &Path) -> std::io::Result<(u64, u64)> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok((meta.len(), mtime))
}

type HashCache = Mutex<BTreeMap<(String, u64, u64), String>>;

static HASH_CACHE: LazyLock<HashCache> = LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Hash a file, reusing a previous hash when path, size and mtime are
/// unchanged. Backend plugins can be hundreds of megabytes, and ordinary
/// capability probes run on every launch.
fn file_identity(path: &Path) -> std::io::Result<FileIdentity> {
    let canonical = path.canonicalize()?;
    let (size, mtime) = stat_identity(&canonical)?;
    let key = (canonical.to_string_lossy().into_owned(), size, mtime);
    let cached = HASH_CACHE.lock().unwrap().get(&key).cloned();
    let sha256 = match cached {
        Some(hash) => hash,
        None => {
            let hash = sha256_file(&canonical)?;
            HASH_CACHE.lock().unwrap().insert(key.clone(), hash.clone());
            hash
        }
    };
    Ok(FileIdentity {
        path: key.0,
        sha256,
        size,
        mtime_unix: mtime,
    })
}

fn is_runtime_library(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let stem_ok = ["libllama", "libggml", "libmtmd", "llama", "ggml", "mtmd"]
        .iter()
        .any(|prefix| lower.starts_with(prefix));
    let ext_ok = [".dll", ".dylib", ".so"]
        .iter()
        .any(|ext| lower.ends_with(ext) || lower.contains(&format!("{ext}.")));
    stem_ok && ext_ok
}

impl RuntimeIdentity {
    /// Compute the closure identity. `loaded_backends` are the plugin paths the
    /// loader reported; same-directory llama/ggml shared libraries are added.
    /// Blocking: hashes files (cached by path, size and mtime).
    pub fn compute(executable: &Path, loaded_backends: &[LoadedBackend]) -> std::io::Result<Self> {
        let exe = file_identity(executable)?;
        let mut paths: BTreeMap<String, PathBuf> = BTreeMap::new();
        for backend in loaded_backends {
            if let Ok(canonical) = backend.path.canonicalize() {
                paths.insert(canonical.to_string_lossy().into_owned(), canonical);
            }
        }
        if let Some(dir) = Path::new(&exe.path).parent()
            && let Ok(entries) = std::fs::read_dir(dir)
        {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if is_runtime_library(&name)
                    && entry
                        .file_type()
                        .map(|t| t.is_file() || t.is_symlink())
                        .unwrap_or(false)
                    && let Ok(canonical) = entry.path().canonicalize()
                    && canonical.is_file()
                {
                    paths.insert(canonical.to_string_lossy().into_owned(), canonical);
                }
            }
        }
        let mut libraries = Vec::new();
        for path in paths.values() {
            libraries.push(file_identity(path)?);
        }
        Ok(Self {
            executable: exe,
            libraries,
        })
    }

    /// Stable fingerprint over every path and content hash in the closure.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        for file in std::iter::once(&self.executable).chain(self.libraries.iter()) {
            hasher.update(file.path.as_bytes());
            hasher.update([0]);
            hasher.update(file.sha256.as_bytes());
            hasher.update([0]);
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Cheap staleness check at launch/swap time: every file in the closure
    /// still exists with the same size and mtime. Does not rehash, so a
    /// same-size same-mtime replacement is not detected here; the content hash
    /// is re-verified at probe, promotion and verification time.
    pub fn still_matches_on_disk(&self) -> bool {
        std::iter::once(&self.executable)
            .chain(self.libraries.iter())
            .all(|file| {
                stat_identity(Path::new(&file.path))
                    .map(|(size, mtime)| size == file.size && mtime == file.mtime_unix)
                    .unwrap_or(false)
            })
    }
}

// ---------------------------------------------------------------------------
// Probe execution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RuntimeProbe {
    pub parsed: ParsedProbe,
    pub identity: Option<RuntimeIdentity>,
    pub basis: KvSupportBasis,
}

fn configure_env(cmd: &mut Command, policy: EnvPolicy, temp: &Path) {
    cmd.current_dir(temp);
    if policy == EnvPolicy::Sanitized {
        cmd.env_clear();
        #[cfg(windows)]
        {
            let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
            cmd.env("SystemRoot", &root)
                .env("windir", &root)
                .env("PATH", format!(r"{root}\System32;{root}"))
                .env("TEMP", temp)
                .env("TMP", temp);
        }
        #[cfg(not(windows))]
        {
            cmd.env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
                .env("LANG", "C")
                .env("TMPDIR", temp);
        }
    }
}

async fn read_budgeted<R>(mut reader: R, used: Arc<AtomicUsize>) -> Result<Vec<u8>, ProbeError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut out = Vec::with_capacity(8192);
    let mut buf = [0u8; 8192];
    loop {
        let n = reader
            .read(&mut buf)
            .await
            .map_err(|e| ProbeError::Io(e.to_string()))?;
        if n == 0 {
            return Ok(out);
        }
        if used.fetch_add(n, Ordering::SeqCst) + n > PROBE_MAX_OUTPUT {
            return Err(ProbeError::OutputTooLarge);
        }
        out.extend_from_slice(&buf[..n]);
    }
}

/// Run the bounded probe against `executable`.
pub async fn probe_runtime(
    executable: &Path,
    policy: EnvPolicy,
) -> Result<RuntimeProbe, ProbeError> {
    probe_runtime_with_timeout(executable, policy, PROBE_TIMEOUT).await
}

pub async fn probe_runtime_with_timeout(
    executable: &Path,
    policy: EnvPolicy,
    timeout: Duration,
) -> Result<RuntimeProbe, ProbeError> {
    let temp = tempfile::tempdir().map_err(|e| ProbeError::Io(e.to_string()))?;
    let model = temp
        .path()
        .join(format!("foundry-probe-missing-{}.gguf", uuid_like()));
    if model.exists() {
        return Err(ProbeError::Io(
            "probe model path unexpectedly exists".into(),
        ));
    }
    let model_str = model.to_string_lossy().into_owned();

    let mut cmd = Command::new(executable);
    cmd.args(["-lv", "4", "-m", &model_str])
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_env(&mut cmd, policy, temp.path());
    crate::platform::no_window_tokio(&mut cmd);

    let mut child = cmd.spawn().map_err(|e| ProbeError::Spawn(e.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ProbeError::Io("no stdout".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ProbeError::Io("no stderr".into()))?;
    let used = Arc::new(AtomicUsize::new(0));

    let capture = async {
        let (out, err) = tokio::join!(
            read_budgeted(stdout, used.clone()),
            read_budgeted(stderr, used.clone())
        );
        (out, err)
    };
    let (out, err) = match tokio::time::timeout(timeout, capture).await {
        Ok(result) => result,
        Err(_) => {
            let _ = child.kill().await;
            return Err(ProbeError::Timeout);
        }
    };
    let (out, err) = match (out, err) {
        (Ok(out), Ok(err)) => (out, err),
        (Err(e), _) | (_, Err(e)) => {
            let _ = child.kill().await;
            return Err(e);
        }
    };
    // Output closed; reap the process (bounded).
    match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
        Ok(_) => {}
        Err(_) => {
            let _ = child.kill().await;
        }
    }

    let mut text = String::from_utf8_lossy(&out).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&err));
    let parsed = parse_probe_output(&text, &model_str);
    if !parsed.system_info_seen {
        return Err(ProbeError::Unrecognized("no system_info record".into()));
    }
    if !parsed.model_missing_diagnostic {
        return Err(ProbeError::Unrecognized(
            "no recognized model-missing diagnostic for the probe path".into(),
        ));
    }

    let exe = executable.to_path_buf();
    let backends = parsed.loaded_backends.clone();
    let identity = tokio::task::spawn_blocking(move || RuntimeIdentity::compute(&exe, &backends))
        .await
        .ok()
        .and_then(Result::ok);
    let basis = kv_basis_from_probe(&parsed, identity.as_ref());
    Ok(RuntimeProbe {
        parsed,
        identity,
        basis,
    })
}

fn uuid_like() -> String {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}-{}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    const METAL_FIXTURE: &str =
        include_str!("../../tests/fixtures/llama_probe/metal_missing_model.txt");
    const CUDA_FIXTURE: &str =
        include_str!("../../tests/fixtures/llama_probe/cuda_missing_model.txt");
    const MODEL: &str = "/nonexistent/foundry-probe-xyz.gguf";

    #[test]
    fn system_info_groups_parse() {
        let groups = parse_system_info(
            " n_threads = 6 (n_threads_batch = 6) / 18 | MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | ",
        );
        assert_eq!(groups["MTL"]["EMBED_LIBRARY"], "1");
        assert_eq!(groups["CPU"]["ARM_FMA"], "1");
        assert_eq!(groups[""]["n_threads"], "6 (n_threads_batch = 6) / 18");
    }

    #[test]
    fn real_metal_output_is_recognized() {
        let parsed = parse_probe_output(METAL_FIXTURE, MODEL);
        assert!(parsed.system_info_seen);
        assert!(parsed.model_missing_diagnostic);
        assert_eq!(detect_backend(&parsed), Backend::Metal);
        assert!(
            parsed
                .loaded_backends
                .iter()
                .any(|b| b.name == "MTL" && b.path.ends_with("libggml-metal.so"))
        );
        assert!(parsed.devices.iter().any(|d| d.starts_with("MTL0")));
        let basis = kv_basis_from_probe(&parsed, None);
        assert_eq!(basis.backend, Backend::Metal);
        assert!(!basis.fa_pairs.is_confirmed());
    }

    #[test]
    fn real_metal_output_reports_bfloat_and_build() {
        let parsed = parse_probe_output(METAL_FIXTURE, MODEL);
        assert_eq!(parsed.metal_bf16, Some(true));
        assert_eq!(parsed.build_number, Some(11719));
        assert_eq!(parsed.build_commit.as_deref(), Some("5aeac37a3"));
        let basis = kv_basis_from_probe(&parsed, None);
        assert_eq!(basis.metal_bf16.confirmed_value(), Some(&true));
        assert_eq!(basis.build_number, Some(11719));
        assert!(basis.metal_rule_applies());
    }

    #[test]
    fn metal_bfloat_false_or_missing_never_confirms_support() {
        let base = "load_backend: loaded MTL backend from /x/libggml-metal.so\ncmn common_param: system_info: n = 1 | MTL : EMBED_LIBRARY = 1 |\n";
        let none = parse_probe_output(base, MODEL);
        assert!(!kv_basis_from_probe(&none, None).metal_bf16.is_confirmed());
        let off = format!("I ggml_metal_device_init: has bfloat            = false\n{base}");
        let basis = kv_basis_from_probe(&parse_probe_output(&off, MODEL), None);
        assert_eq!(basis.metal_bf16.confirmed_value(), Some(&false));
        // Two devices: any false wins.
        let mixed = format!(
            "I ggml_metal_device_init: has bfloat            = true\nI ggml_metal_device_init: has bfloat            = false\n{base}"
        );
        assert_eq!(parse_probe_output(&mixed, MODEL).metal_bf16, Some(false));
        // CUDA never reports a Metal flag.
        let cuda = parse_probe_output(CUDA_FIXTURE, MODEL);
        assert!(!kv_basis_from_probe(&cuda, None).metal_bf16.is_confirmed());
    }

    #[test]
    fn cuda_fa_quants_all_is_confirmed_with_forty_nine_pairs() {
        let parsed = parse_probe_output(CUDA_FIXTURE, MODEL);
        assert_eq!(detect_backend(&parsed), Backend::Cuda);
        let basis = kv_basis_from_probe(&parsed, None);
        let pairs = basis.fa_pairs.confirmed_value().expect("confirmed");
        assert_eq!(pairs.len(), 49);
    }

    #[test]
    fn cuda_missing_malformed_or_unknown_fa_quants_is_unavailable() {
        let template = |features: &str| {
            format!(
                "load_backend: loaded CUDA backend from /x/ggml-cuda.dll\ncmn common_param: system_info: n_threads = 1 | CUDA : {features} | CPU : SSE3 = 1 |\n"
            )
        };
        for features in [
            "ARCHS = 1200",
            "FA_QUANTS = q8_0",
            "FA_QUANTS = q8_0-zzz",
            "FA_QUANTS = all,q8_0-q8_0",
        ] {
            let parsed = parse_probe_output(&template(features), MODEL);
            let basis = kv_basis_from_probe(&parsed, None);
            assert_eq!(basis.backend, Backend::Cuda, "{features}");
            assert!(!basis.fa_pairs.is_confirmed(), "{features}");
        }
    }

    #[test]
    fn explicit_list_includes_f16_pair_and_is_confirmed() {
        let text = "cmn common_param: system_info: n = 1 | CUDA : FA_QUANTS = q8_0-q4_0 | CPU : SSE3 = 1 |\n";
        let parsed = parse_probe_output(text, MODEL);
        let basis = kv_basis_from_probe(&parsed, None);
        assert_eq!(basis.fa_pairs.confirmed_value().unwrap().len(), 2);
    }

    #[test]
    fn model_missing_diagnostic_must_name_the_probe_path() {
        let other =
            "E gguf_init_from_file: failed to open GGUF file '/some/other.gguf' (No such file)\n";
        assert!(!parse_probe_output(other, MODEL).model_missing_diagnostic);
    }

    #[test]
    fn cpu_only_and_blas_resolve_to_cpu() {
        let text = "load_backend: loaded BLAS backend from /x/libggml-blas.so\nload_backend: loaded CPU backend from /x/libggml-cpu.so\ncmn common_param: system_info: n = 1 | CPU : SSE3 = 1 |\n";
        assert_eq!(
            detect_backend(&parse_probe_output(text, MODEL)),
            Backend::Cpu
        );
    }

    #[test]
    fn unrecognized_accelerator_is_unknown_not_cpu() {
        let text = "load_backend: loaded ROCm backend from /x/ggml-hip.dll\ncmn common_param: system_info: n = 1 | ROCm : X = 1 | CPU : SSE3 = 1 |\n";
        assert_eq!(
            detect_backend(&parse_probe_output(text, MODEL)),
            Backend::Unknown
        );
    }

    #[test]
    fn identity_covers_plugins_and_changes_when_only_a_plugin_changes() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("llama-server");
        let plugin = dir.path().join("libggml-metal.so");
        std::fs::write(&exe, b"exe").unwrap();
        std::fs::write(&plugin, b"plugin-v1").unwrap();
        let backends = vec![LoadedBackend {
            name: "MTL".into(),
            path: plugin.clone(),
        }];
        let first = RuntimeIdentity::compute(&exe, &backends).unwrap();
        assert!(
            first
                .libraries
                .iter()
                .any(|f| f.path.ends_with("libggml-metal.so"))
        );
        std::fs::write(&plugin, b"plugin-v2-different-length").unwrap();
        let second = RuntimeIdentity::compute(&exe, &backends).unwrap();
        assert_ne!(first.fingerprint(), second.fingerprint());
        assert_eq!(first.executable, second.executable);
        assert!(!first.still_matches_on_disk());
        assert!(second.still_matches_on_disk());
    }

    #[test]
    fn identity_picks_up_same_directory_runtime_libraries_only() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("llama-server");
        std::fs::write(&exe, b"exe").unwrap();
        std::fs::write(dir.path().join("libllama.dylib"), b"a").unwrap();
        std::fs::write(dir.path().join("libggml-base.0.dylib"), b"b").unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"c").unwrap();
        std::fs::write(dir.path().join("libother.dylib"), b"d").unwrap();
        let identity = RuntimeIdentity::compute(&exe, &[]).unwrap();
        let names: Vec<_> = identity
            .libraries
            .iter()
            .map(|f| {
                Path::new(&f.path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["libggml-base.0.dylib", "libllama.dylib"]);
    }

    #[cfg(unix)]
    mod process {
        use super::*;
        use std::os::unix::fs::PermissionsExt;

        fn fake_server(dir: &Path, body: &str) -> PathBuf {
            let path = dir.join("llama-server");
            std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        }

        // The fake server echoes the -m path back inside the model-missing diagnostic.
        const GOOD: &str = r#"
echo "load_backend: loaded CUDA backend from $0" >&2
echo "cmn common_param: system_info: n = 1 | CUDA : ARCHS = 1200 | FA_QUANTS = all | CPU : SSE3 = 1 |" >&2
echo "E gguf_init_from_file: failed to open GGUF file '$4' (No such file or directory)" >&2
exit 1
"#;

        #[tokio::test]
        async fn recognized_model_missing_exit_is_accepted() {
            let dir = tempfile::tempdir().unwrap();
            let exe = fake_server(dir.path(), GOOD);
            let probe = probe_runtime(&exe, EnvPolicy::Sanitized).await.unwrap();
            assert_eq!(probe.basis.backend, Backend::Cuda);
            assert_eq!(probe.basis.fa_pairs.confirmed_value().unwrap().len(), 49);
            assert!(probe.identity.is_some());
            assert!(probe.basis.runtime_identity.is_some());
        }

        #[tokio::test]
        async fn arbitrary_nonzero_exit_without_diagnostic_is_unrecognized() {
            let dir = tempfile::tempdir().unwrap();
            let exe = fake_server(dir.path(), "echo boom >&2\nexit 3");
            let err = probe_runtime(&exe, EnvPolicy::Inherited).await.unwrap_err();
            assert!(matches!(err, ProbeError::Unrecognized(_)), "{err:?}");
        }

        #[tokio::test]
        async fn diagnostic_for_a_different_path_is_not_accepted() {
            let dir = tempfile::tempdir().unwrap();
            let body = r#"
echo "cmn common_param: system_info: n = 1 | CUDA : FA_QUANTS = all | CPU : SSE3 = 1 |" >&2
echo "E gguf_init_from_file: failed to open GGUF file '/elsewhere.gguf' (No such file)" >&2
exit 1
"#;
            let exe = fake_server(dir.path(), body);
            let err = probe_runtime(&exe, EnvPolicy::Inherited).await.unwrap_err();
            assert!(matches!(err, ProbeError::Unrecognized(_)), "{err:?}");
        }

        #[tokio::test]
        async fn hang_times_out_and_is_reaped() {
            let dir = tempfile::tempdir().unwrap();
            let exe = fake_server(dir.path(), "exec sleep 30");
            let started = std::time::Instant::now();
            let err =
                probe_runtime_with_timeout(&exe, EnvPolicy::Inherited, Duration::from_millis(300))
                    .await
                    .unwrap_err();
            assert_eq!(err, ProbeError::Timeout);
            assert!(started.elapsed() < Duration::from_secs(5));
        }

        #[tokio::test]
        async fn output_flood_is_capped() {
            let dir = tempfile::tempdir().unwrap();
            let exe = fake_server(dir.path(), "exec yes flood");
            let started = std::time::Instant::now();
            let err = probe_runtime(&exe, EnvPolicy::Inherited).await.unwrap_err();
            assert_eq!(err, ProbeError::OutputTooLarge);
            assert!(started.elapsed() < Duration::from_secs(5));
        }

        #[tokio::test]
        async fn probe_runs_in_an_owned_temp_cwd_with_a_sanitized_environment() {
            let dir = tempfile::tempdir().unwrap();
            let body = r#"
echo "cwd=$(pwd)" >&2
echo "path=$PATH home=${HOME:-unset} dyld=${DYLD_LIBRARY_PATH:-unset}" >&2
echo "cmn common_param: system_info: n = 1 | CPU : SSE3 = 1 |" >&2
echo "E gguf_init_from_file: failed to open GGUF file '$4' (x)" >&2
exit 1
"#;
            let exe = fake_server(dir.path(), body);
            // SAFETY-free: set an override in this process; the sanitized probe must not see it.
            unsafe { std::env::set_var("DYLD_LIBRARY_PATH", "/should/not/leak") };
            let probe = probe_runtime(&exe, EnvPolicy::Sanitized).await.unwrap();
            unsafe { std::env::remove_var("DYLD_LIBRARY_PATH") };
            assert_eq!(probe.basis.backend, Backend::Cpu);
        }

        #[tokio::test]
        async fn missing_executable_is_a_spawn_error() {
            let err = probe_runtime(Path::new("/definitely/not/here"), EnvPolicy::Inherited)
                .await
                .unwrap_err();
            assert!(matches!(err, ProbeError::Spawn(_)));
        }
    }
}
