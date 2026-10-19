//! Build environment vs staged-runtime environment (plan §5.2, §4.6.1).
//!
//! Two environments exist and must never be confused:
//!
//! - the **build environment**: on Windows the captured `vcvars64` environment
//!   with the selected CUDA toolkit first on `PATH`, handed to cmake/ninja/nvcc;
//! - the **staged-runtime environment**: sanitized, with no toolkit, build-tree
//!   or user `PATH` entries, used when the shared verifier executes the staged
//!   candidate. A closure that only resolves because of the build environment
//!   must fail verification.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::TargetOs;

/// Case-insensitive on Windows, so keys are normalised to upper case.
pub fn normalize_key(key: &str) -> String {
    key.to_ascii_uppercase()
}

/// Parse `set` output (`KEY=VALUE` per line). Lines without `=`, and `=C:`-style
/// drive-cwd entries, are ignored.
pub fn parse_env_dump(text: &str) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.starts_with('=') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=')
            && !key.is_empty()
            && !key.contains(char::is_whitespace)
        {
            env.insert(normalize_key(key), value.to_string());
        }
    }
    env
}

/// A successful batch exit is not proof that the compiler and SDK components
/// were installed. Validate the files a CUDA/C++ smoke configure will need.
pub fn validate_captured_env(
    env: &BTreeMap<String, String>,
    exists: &dyn Fn(&Path) -> bool,
) -> Result<(), String> {
    let required = |key: &str| {
        env.get(key)
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| format!("vcvars did not set {key}"))
    };
    let vc = PathBuf::from(required("VCTOOLSINSTALLDIR")?);
    let sdk = PathBuf::from(required("WINDOWSSDKDIR")?);
    let version = required("WINDOWSSDKVERSION")?.trim_end_matches(['\\', '/']);
    for file in [
        vc.join("bin/Hostx64/x64/cl.exe"),
        vc.join("include/vector"),
        sdk.join("Include").join(version).join("um/Windows.h"),
        sdk.join("Include").join(version).join("ucrt/stdio.h"),
    ] {
        if !exists(&file) {
            return Err(format!(
                "MSVC/Windows SDK component is missing: {}",
                file.display()
            ));
        }
    }
    Ok(())
}

/// The minimal PATH vcvars is applied to, so repeated captures do not grow it
/// (ported from the tooling's capture).
pub fn clean_base_path(system_root: &str) -> String {
    let root = system_root.trim_end_matches(['\\', '/']);
    format!(r"{root}\System32;{root};{root}\System32\Wbem")
}

/// `set` dumps every inherited variable. Start vcvars with OS discovery facts
/// only, so API tokens, proxies with credentials, etc. never enter its durable
/// environment cache. Build children separately inherit the current process.
pub fn vcvars_base_env(process: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    const KEYS: &[&str] = &[
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PROGRAMDATA",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
        "PATHEXT",
    ];
    process
        .iter()
        .filter(|(k, _)| KEYS.contains(&normalize_key(k).as_str()))
        .map(|(k, v)| (normalize_key(k), v.clone()))
        .collect()
}

/// The `cmd` command line that captures the environment vcvars produces.
/// Run with `cmd /d /s /c <this>`; the string must be passed verbatim (raw
/// argument), not re-quoted.
pub fn vcvars_capture_command(vcvars: &Path, base_path: &str) -> String {
    format!(
        "\"set \"PATH={base_path}\" && call \"{}\" >nul && set\"",
        vcvars.display()
    )
}

/// What vcvars changed or added compared with the environment it started from.
pub fn env_diff(
    base: &BTreeMap<String, String>,
    captured: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    captured
        .iter()
        .filter(|(k, v)| base.get(*k) != Some(*v))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolchainCache {
    pub vs_instance_id: String,
    pub vcvars: PathBuf,
    pub vcvars_mtime_unix: u64,
    pub captured_at_unix: u64,
    pub env: BTreeMap<String, String>,
}

impl ToolchainCache {
    /// Valid only for the same VS instance, vcvars file and modification time.
    pub fn matches(&self, instance: &str, vcvars: &Path, mtime_unix: u64) -> bool {
        self.vs_instance_id == instance
            && self.vcvars == vcvars
            && self.vcvars_mtime_unix == mtime_unix
            && self.env.contains_key("PATH")
    }
}

pub fn load_cache(file: &Path) -> Option<ToolchainCache> {
    serde_json::from_slice(&std::fs::read(file).ok()?).ok()
}

pub fn store_cache(file: &Path, cache: &ToolchainCache) -> std::io::Result<()> {
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = file.with_extension("json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(cache).map_err(std::io::Error::other)?,
    )?;
    std::fs::rename(tmp, file)
}

pub fn file_mtime_unix(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Extra PATH entries, prepended.
fn prepend_path(path_value: &str, dirs: &[String], sep: char) -> String {
    let mut parts: Vec<String> = dirs.to_vec();
    parts.extend(
        path_value
            .split(sep)
            .filter(|p| !p.is_empty())
            .map(str::to_string),
    );
    parts.join(&sep.to_string())
}

/// Inputs for the build environment.
pub struct BuildEnvInputs<'a> {
    pub os: TargetOs,
    /// The environment the app itself runs in.
    pub process_env: &'a BTreeMap<String, String>,
    /// Windows: the captured vcvars environment. `None` elsewhere.
    pub captured: Option<&'a BTreeMap<String, String>>,
    /// Windows CUDA: the toolkit root from `CUDA_PATH`.
    pub cuda_root: Option<&'a str>,
    pub ccache_dir: Option<&'a Path>,
    pub ccache_max_bytes: Option<u64>,
}

/// The environment for cmake/ninja/nvcc children.
pub fn build_env(inputs: &BuildEnvInputs) -> BTreeMap<String, String> {
    let sep = if inputs.os == TargetOs::Windows {
        ';'
    } else {
        ':'
    };
    let mut env: BTreeMap<String, String> = if inputs.os == TargetOs::Windows {
        inputs
            .process_env
            .iter()
            .map(|(k, v)| (normalize_key(k), v.clone()))
            .collect()
    } else {
        inputs.process_env.clone()
    };
    if let Some(captured) = inputs.captured {
        for (k, v) in captured {
            env.insert(k.clone(), v.clone());
        }
    }
    let path_key = "PATH";
    if let Some(root) = inputs.cuda_root {
        let root = root.trim_end_matches(['\\', '/']);
        let current = env.get(path_key).cloned().unwrap_or_default();
        // `CUDA_PATH` only: the other installed toolkits stay off PATH here.
        env.insert(
            path_key.into(),
            prepend_path(
                &current,
                &[format!(r"{root}\bin\x64"), format!(r"{root}\bin")],
                sep,
            ),
        );
        env.insert("CUDA_PATH".into(), root.to_string());
        env.insert("CUDATOOLKIT_ROOT".into(), root.to_string());
    }
    if let Some(dir) = inputs.ccache_dir {
        env.insert("CCACHE_DIR".into(), dir.to_string_lossy().into_owned());
        if let Some(max) = inputs.ccache_max_bytes {
            env.insert("CCACHE_MAXSIZE".into(), format!("{}M", max / (1024 * 1024)));
        }
    }
    env
}

/// Keys whose values are secrets or noise: omitted from the log's env diff.
fn redact(key: &str, value: &str) -> String {
    let upper = key.to_ascii_uppercase();
    if ["TOKEN", "SECRET", "PASSWORD", "KEY", "AUTH", "CREDENTIAL"]
        .iter()
        .any(|w| upper.contains(w))
    {
        return "<redacted>".into();
    }
    value.to_string()
}

/// A redacted diff of `child` against `parent` for the phase header line.
pub fn redacted_env_diff(
    parent: &BTreeMap<String, String>,
    child: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut out = Vec::new();
    for (k, v) in child {
        if parent.get(k) != Some(v) {
            out.push(format!("{k}={}", redact(k, v)));
        }
    }
    for k in parent.keys() {
        if !child.contains_key(k) {
            out.push(format!("-{k}"));
        }
    }
    out
}

/// The sanitized environment a staged runtime is executed in. Mirrors
/// `llama_runtime::probe::EnvPolicy::Sanitized`; the test below holds both to
/// the same contract.
pub fn staged_env(os: TargetOs, system_root: &str, temp: &Path) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    if os == TargetOs::Windows {
        env.insert("SYSTEMROOT".into(), system_root.into());
        env.insert("WINDIR".into(), system_root.into());
        env.insert(
            "PATH".into(),
            format!(r"{system_root}\System32;{system_root}"),
        );
        env.insert("TEMP".into(), temp.to_string_lossy().into_owned());
        env.insert("TMP".into(), temp.to_string_lossy().into_owned());
    } else {
        env.insert("PATH".into(), "/usr/bin:/bin:/usr/sbin:/sbin".into());
        env.insert("LANG".into(), "C".into());
        env.insert("TMPDIR".into(), temp.to_string_lossy().into_owned());
    }
    env
}

/// Directories the staged environment must never contain. Compared as
/// separator-agnostic, case-insensitive prefixes so `C:\\CUDA\\v13.4\\bin` is
/// caught whatever the host OS parsing the path.
pub fn path_leaks(path_value: &str, forbidden: &[&Path], sep: char) -> Vec<String> {
    let norm = |s: &str| {
        s.replace('\\', "/")
            .trim_end_matches('/')
            .to_ascii_lowercase()
    };
    let forbidden: Vec<String> = forbidden
        .iter()
        .map(|f| norm(&f.to_string_lossy()))
        .collect();
    path_value
        .split(sep)
        .filter(|entry| {
            if entry.is_empty() {
                return false;
            }
            let e = norm(entry);
            forbidden
                .iter()
                .any(|f| e == *f || e.starts_with(&format!("{f}/")))
        })
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMP: &str = "ALLUSERSPROFILE=C:\\ProgramData\r\n\
        =C:=C:\\Users\\nick\r\n\
        INCLUDE=C:\\VS\\include\r\n\
        Path=C:\\VS\\bin;C:\\Windows\\System32\r\n\
        VCINSTALLDIR=C:\\VS\\VC\\\r\n\
        WindowsSdkDir=C:\\Kits\\10\\\r\n\
        junk line without equals\r\n\
        EMPTY=\r\n";

    #[test]
    fn dump_parsing_normalizes_keys_and_skips_junk() {
        let env = parse_env_dump(DUMP);
        assert_eq!(env["PATH"], r"C:\VS\bin;C:\Windows\System32");
        assert_eq!(env["WINDOWSSDKDIR"], r"C:\Kits\10\");
        assert_eq!(env["EMPTY"], "");
        assert!(!env.keys().any(|k| k.starts_with('=') || k.contains(' ')));
        assert_eq!(env.len(), 6);
    }

    #[test]
    fn base_path_is_small_and_fixed() {
        let p = clean_base_path(r"C:\Windows\");
        assert_eq!(
            p,
            r"C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem"
        );
        assert!(!p.contains("Strawberry"));
    }

    #[test]
    fn capture_command_matches_the_ported_shape() {
        let cmd = vcvars_capture_command(
            Path::new(r"C:\VS\VC\Auxiliary\Build\vcvars64.bat"),
            r"C:\Windows\System32",
        );
        assert_eq!(
            cmd,
            r#""set "PATH=C:\Windows\System32" && call "C:\VS\VC\Auxiliary\Build\vcvars64.bat" >nul && set""#
        );
    }

    #[test]
    fn diff_keeps_only_what_vcvars_changed() {
        let base = parse_env_dump("PATH=C:\\Windows\nSYSTEMROOT=C:\\Windows\n");
        let captured =
            parse_env_dump("PATH=C:\\VS;C:\\Windows\nSYSTEMROOT=C:\\Windows\nINCLUDE=C:\\inc\n");
        let diff = env_diff(&base, &captured);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains_key("PATH") && diff.contains_key("INCLUDE"));
    }

    #[test]
    fn cache_is_keyed_by_instance_vcvars_and_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("llama-build/toolchain-cache.json");
        let cache = ToolchainCache {
            vs_instance_id: "022d1522".into(),
            vcvars: PathBuf::from(r"C:\VS\vcvars64.bat"),
            vcvars_mtime_unix: 1000,
            captured_at_unix: 2000,
            env: parse_env_dump("PATH=C:\\VS\n"),
        };
        store_cache(&file, &cache).unwrap();
        let loaded = load_cache(&file).unwrap();
        assert_eq!(loaded, cache);
        assert!(loaded.matches("022d1522", Path::new(r"C:\VS\vcvars64.bat"), 1000));
        assert!(!loaded.matches("other", Path::new(r"C:\VS\vcvars64.bat"), 1000));
        assert!(!loaded.matches("022d1522", Path::new(r"C:\VS\vcvars64.bat"), 1001));
        assert!(!loaded.matches("022d1522", Path::new(r"C:\Other\vcvars64.bat"), 1000));
        assert!(load_cache(&dir.path().join("missing.json")).is_none());
        std::fs::write(&file, b"{ not json").unwrap();
        assert!(load_cache(&file).is_none());
    }

    #[test]
    fn a_cache_without_a_path_is_never_valid() {
        let c = ToolchainCache {
            vs_instance_id: "x".into(),
            vcvars: PathBuf::from("v"),
            vcvars_mtime_unix: 1,
            captured_at_unix: 1,
            env: BTreeMap::new(),
        };
        assert!(!c.matches("x", Path::new("v"), 1));
    }

    fn proc_env() -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "Path".to_string(),
                r"C:\Strawberry\c\bin;C:\Windows".to_string(),
            ),
            ("HF_TOKEN".to_string(), "hf_secret".to_string()),
        ])
    }

    #[test]
    fn windows_build_env_layers_vcvars_then_cuda_first() {
        let captured = parse_env_dump("PATH=C:\\VS\\bin;C:\\Windows\nINCLUDE=C:\\inc\n");
        let env = build_env(&BuildEnvInputs {
            os: TargetOs::Windows,
            process_env: &proc_env(),
            captured: Some(&captured),
            cuda_root: Some(r"C:\CUDA\v13.4"),
            ccache_dir: Some(Path::new(r"C:\app\llama-build\ccache")),
            ccache_max_bytes: Some(2 * 1024 * 1024 * 1024),
        });
        let path = &env["PATH"];
        assert!(
            path.starts_with(r"C:\CUDA\v13.4\bin\x64;C:\CUDA\v13.4\bin;C:\VS\bin"),
            "{path}"
        );
        // The captured PATH replaced the process PATH (no Strawberry gcc).
        assert!(!path.contains("Strawberry"));
        assert_eq!(env["INCLUDE"], r"C:\inc");
        assert_eq!(env["CUDA_PATH"], r"C:\CUDA\v13.4");
        assert_eq!(env["CCACHE_DIR"], r"C:\app\llama-build\ccache");
        assert_eq!(env["CCACHE_MAXSIZE"], "2048M");
    }

    #[test]
    fn unix_build_env_passes_the_process_env_through() {
        let process =
            BTreeMap::from([("PATH".to_string(), "/opt/homebrew/bin:/usr/bin".to_string())]);
        let env = build_env(&BuildEnvInputs {
            os: TargetOs::Macos,
            process_env: &process,
            captured: None,
            cuda_root: None,
            ccache_dir: Some(Path::new("/app/llama-build/ccache")),
            ccache_max_bytes: Some(256 * 1024 * 1024),
        });
        assert_eq!(env["PATH"], "/opt/homebrew/bin:/usr/bin");
        assert_eq!(env["CCACHE_MAXSIZE"], "256M");
        assert!(!env.contains_key("CUDA_PATH"));
    }

    #[test]
    fn env_diff_log_redacts_secrets_and_shows_removals() {
        let parent = proc_env();
        let mut child = parent.clone();
        child.insert("CCACHE_DIR".into(), "/c".into());
        child.insert("MY_AUTH_HEADER".into(), "Bearer abc".into());
        child.remove("Path");
        let lines = redacted_env_diff(&parent, &child);
        assert!(lines.contains(&"CCACHE_DIR=/c".to_string()));
        assert!(lines.contains(&"MY_AUTH_HEADER=<redacted>".to_string()));
        assert!(lines.contains(&"-Path".to_string()));
        assert!(
            !lines
                .iter()
                .any(|l| l.contains("Bearer abc") || l.contains("hf_secret"))
        );
    }

    #[test]
    fn the_staged_environment_has_no_toolkit_or_user_path() {
        let temp = Path::new(r"C:\Temp\x");
        let win = staged_env(TargetOs::Windows, r"C:\Windows", temp);
        assert_eq!(win["PATH"], r"C:\Windows\System32;C:\Windows");
        assert!(!win.contains_key("CUDA_PATH"));
        let unix = staged_env(TargetOs::Macos, "", Path::new("/tmp/x"));
        assert_eq!(unix["PATH"], "/usr/bin:/bin:/usr/sbin:/sbin");
        let forbidden = [
            Path::new(r"C:\CUDA\v13.4"),
            Path::new(r"C:\app\llama-build"),
        ];
        assert!(path_leaks(&win["PATH"], &forbidden, ';').is_empty());
        let build_path =
            r"C:\CUDA\v13.4\bin\x64;C:\Windows;C:\app\llama-build\build-cuda\bin\Release";
        assert_eq!(path_leaks(build_path, &forbidden, ';').len(), 2);
    }

    /// The probe's own sanitized environment must match the contract above:
    /// a poisoned parent PATH, toolkit variable and build-tree directory must
    /// not reach the staged runtime.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_poisoned_parent_environment_does_not_reach_the_staged_runtime() {
        use crate::llama_runtime::probe::{EnvPolicy, probe_runtime_with_timeout};
        use std::os::unix::fs::PermissionsExt;
        if std::env::var_os("LLF_POISON_PROBE_CHILD").is_none() {
            // Poison only a new test process. Mutating the shared environment
            // in this multi-threaded test runner would be undefined behavior.
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "llama_build::toolchain_env::tests::a_poisoned_parent_environment_does_not_reach_the_staged_runtime", "--nocapture"])
                .env("LLF_POISON_PROBE_CHILD", "1")
                .env("POISON", "from-parent")
                .env("CUDA_PATH", "/poisoned/cuda")
                .env("PATH", "/poisoned/evil-build-tree/bin")
                .status().unwrap();
            assert!(status.success());
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let evil = dir.path().join("evil-build-tree/bin");
        std::fs::create_dir_all(&evil).unwrap();
        let report = dir.path().join("seen-env.txt");
        // A stand-in `llama-server` that records the environment it is given,
        // then prints a probe-shaped missing-model diagnostic.
        let exe = dir.path().join("llama-server");
        std::fs::write(
            &exe,
            format!(
                "#!/bin/sh\n{{ echo \"PATH=$PATH\"; echo \"CUDA_PATH=$CUDA_PATH\"; echo \"POISON=$POISON\"; }} > '{}'\n\
                 echo 'system_info: n_threads = 1 | CPU : NEON = 1 |' 1>&2\n\
                 echo \"error: failed to load model '$4'\" 1>&2\nexit 1\n",
                report.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();

        let _ = probe_runtime_with_timeout(
            &exe,
            EnvPolicy::Sanitized,
            std::time::Duration::from_secs(10),
        )
        .await;
        let seen = std::fs::read_to_string(&report).unwrap();
        let expected = staged_env(TargetOs::Macos, "", dir.path());
        assert!(
            seen.contains(&format!("PATH={}", expected["PATH"])),
            "{seen}"
        );
        assert!(
            seen.contains("CUDA_PATH=\n"),
            "toolkit variable leaked: {seen}"
        );
        assert!(seen.contains("POISON=\n"), "parent variable leaked: {seen}");
        assert!(!seen.contains("evil-build-tree"));
    }
}

#[cfg(test)]
mod captured_component_tests {
    use super::*;

    #[test]
    fn successful_vcvars_requires_both_compiler_and_sdk_files() {
        let env = parse_env_dump(
            "VCToolsInstallDir=C:/VS/VC\nWindowsSdkDir=C:/SDK\nWindowsSDKVersion=10.0.26100.0\\\n",
        );
        assert!(validate_captured_env(&env, &|_| true).is_ok());
        assert!(
            validate_captured_env(&env, &|p| p.file_name().is_none_or(|n| n != "Windows.h"))
                .is_err()
        );
        assert!(
            validate_captured_env(&env, &|p| p.file_name().is_none_or(|n| n != "cl.exe")).is_err()
        );
        assert!(validate_captured_env(&BTreeMap::new(), &|_| true).is_err());
    }
}

#[cfg(test)]
mod capture_secret_tests {
    #[test]
    fn vcvars_capture_never_caches_parent_tokens_or_proxy_credentials() {
        let parent = std::collections::BTreeMap::from([
            ("SystemRoot".into(), "C:\\Windows".into()),
            ("API_TOKEN".into(), "secret".into()),
            ("HTTPS_PROXY".into(), "https://user:password@proxy".into()),
            ("PATH".into(), "untrusted".into()),
        ]);
        let env = super::vcvars_base_env(&parent);
        assert_eq!(env.len(), 1);
        assert_eq!(env["SYSTEMROOT"], "C:\\Windows");
    }
}
