//! The pure `(BuildPlan) -> cmake argv` function (plan §4.5) plus the CMake
//! cache helpers: stale-path detection and configure-evidence parsing.
//!
//! Nothing here touches a toolchain. The argument list is what the manifest
//! records, so it is built in one place and covered by golden tests.

use std::collections::BTreeSet;
use std::path::Path;

use crate::inference::kv_support::{FaQuantsParseError, KvType, compiled_fa_pairs};

use super::{Accelerator, BUILD_TARGETS, BuildPlan, CudaVersion, Generator, TargetOs};

/// Decision D9: source CUDA builds always compile every K/V pair. Not a user
/// option.
pub const FA_BUILD_SPEC: &str = "all";

/// First CUDA toolkit that bundles a new enough CCCL (plan §4.5). Older
/// toolkits get CCCL fetched at configure time.
pub const CCCL_BUNDLED_FROM: CudaVersion = CudaVersion {
    major: 13,
    minor: 5,
    patch: 0,
};

/// CCCL tag fetched for older toolkits. One constant, updated alongside
/// upstream's release CI.
pub const CCCL_FETCH_VERSION: &str = "v3.4.3";

fn define(name: &str, value: &str) -> String {
    format!("-D{name}={value}")
}

fn on_off(value: bool) -> &'static str {
    if value { "ON" } else { "OFF" }
}

/// Arguments for `cmake` (without the program name) that configure `plan`.
pub fn cmake_configure_args(plan: &BuildPlan) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "-S".into(),
        plan.source_dir.to_string_lossy().into_owned(),
        "-B".into(),
        plan.build_dir.to_string_lossy().into_owned(),
        "-G".into(),
        plan.generator().cmake_name().into(),
    ];
    if plan.generator() == Generator::Ninja {
        // Multi-config generators take the configuration at build time.
        args.push(define("CMAKE_BUILD_TYPE", "Release"));
    }
    let cuda = plan.accelerator == Accelerator::Cuda;
    for (name, value) in [
        ("LLAMA_BUILD_TESTS", "OFF"),
        ("LLAMA_BUILD_EXAMPLES", "OFF"),
        ("LLAMA_BUILD_TOOLS", "ON"),
        ("LLAMA_BUILD_SERVER", "ON"),
        ("BUILD_SHARED_LIBS", "ON"),
        ("GGML_BACKEND_DL", "ON"),
    ] {
        args.push(define(name, value));
    }
    // Portable CUDA binaries must not be tuned to the build machine; the
    // local-only Metal and CPU builds may be.
    args.push(define("GGML_NATIVE", on_off(!cuda)));
    if plan.ccache {
        args.push(define("GGML_CCACHE", "ON"));
    }

    match plan.accelerator {
        Accelerator::Metal => {
            args.push(define("GGML_METAL", "ON"));
            args.push(define("GGML_METAL_EMBED_LIBRARY", "ON"));
            args.push(define("GGML_METAL_USE_BF16", on_off(plan.apple_silicon)));
            args.push(define("GGML_BLAS", "ON"));
            args.push(define("GGML_BLAS_VENDOR", "Apple"));
        }
        Accelerator::Cuda => {
            if plan.os == TargetOs::Windows {
                // Pin MSVC: Strawberry Perl's gcc is on PATH on some machines
                // and would otherwise be picked up.
                args.push(define("CMAKE_C_COMPILER", "cl"));
                args.push(define("CMAKE_CXX_COMPILER", "cl"));
            }
            if let Some(cuda) = &plan.cuda {
                args.push(define("CMAKE_CUDA_COMPILER", &cuda.nvcc.to_string_lossy()));
                if plan.os == TargetOs::Windows {
                    // The bare name, not the absolute MSVC path: nvcc/CMake can
                    // lose quoting for a -ccbin path with spaces under Ninja
                    // Multi-Config. The captured vcvars env puts `cl.exe` first.
                    args.push(define("CMAKE_CUDA_HOST_COMPILER", "cl.exe"));
                }
                if cuda.version < CCCL_BUNDLED_FROM {
                    args.push(define("GGML_CUDA_CCCL_VERSION", CCCL_FETCH_VERSION));
                }
                args.push(define("CMAKE_CUDA_ARCHITECTURES", &cuda.architectures));
            }
            args.push(define("GGML_CUDA", "ON"));
            args.push(define("GGML_CUDA_NCCL", "OFF"));
            args.push(define("GGML_CUDA_COMPRESSION_MODE", "speed"));
            args.push(define("GGML_CUDA_FA_QUANTS", FA_BUILD_SPEC));
            args.push(define("GGML_CPU_ALL_VARIANTS", "ON"));
        }
        Accelerator::Cpu => {}
    }

    for var in &plan.stale_cache_vars {
        args.push("-U".into());
        args.push(var.clone());
    }
    args
}

/// Arguments for `cmake --build` of the two targets we ship.
pub fn cmake_build_args(plan: &BuildPlan, jobs: usize) -> Vec<String> {
    let mut args = vec![
        "--build".to_string(),
        plan.build_dir.to_string_lossy().into_owned(),
    ];
    if plan.generator() == Generator::NinjaMultiConfig {
        args.push("--config".into());
        args.push("Release".into());
    }
    args.push("--target".into());
    args.extend(BUILD_TARGETS.iter().map(|t| t.to_string()));
    args.push("--parallel".into());
    args.push(jobs.max(1).to_string());
    args
}

/// Where the configure's binaries land, relative to the build directory.
pub fn output_subdir(plan: &BuildPlan) -> &'static str {
    match plan.generator() {
        Generator::NinjaMultiConfig => "bin/Release",
        Generator::Ninja => "bin",
    }
}

/// The ninja file listing the targets this configure defines.
pub fn ninja_file(plan: &BuildPlan) -> &'static str {
    match plan.generator() {
        Generator::NinjaMultiConfig => "build-Release.ninja",
        Generator::Ninja => "build.ninja",
    }
}

// ---------------------------------------------------------------------------
// CMakeCache.txt
// ---------------------------------------------------------------------------

/// Cached paths that are legitimately absent on a healthy machine.
const CACHE_PATH_ABSENT_OK: [&str; 2] = ["CMAKE_INSTALL_PREFIX", "CMAKE_INSTALL_OLDINCLUDEDIR"];

fn is_absolute_cache_path(value: &str) -> bool {
    let bytes = value.as_bytes();
    value.starts_with('/')
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/'))
}

/// `(variable, missing path)` for every absolute `PATH`/`FILEPATH` cache entry
/// that no longer exists. A dead pinned path (an upgraded Homebrew dependency)
/// otherwise surfaces as a misleading "target was not found" at configure time.
///
/// Semicolon lists are skipped: one `-U` cannot express a partial repair.
pub fn find_stale_cache_paths(
    cache_text: &str,
    exists: &dyn Fn(&Path) -> bool,
) -> Vec<(String, String)> {
    let mut stale = Vec::new();
    for line in cache_text.lines() {
        let Some((var, rest)) = line.split_once(':') else {
            continue;
        };
        let Some((kind, value)) = rest.split_once('=') else {
            continue;
        };
        if kind != "PATH" && kind != "FILEPATH" {
            continue;
        }
        let value = value.trim();
        if value.is_empty()
            || CACHE_PATH_ABSENT_OK.contains(&var)
            || !is_absolute_cache_path(value)
            || value.contains(';')
        {
            continue;
        }
        if !exists(Path::new(value)) {
            stale.push((var.to_string(), value.to_string()));
        }
    }
    stale
}

/// One cache variable's value (`NAME:TYPE=value`).
pub fn cache_value<'a>(cache_text: &'a str, name: &str) -> Option<&'a str> {
    cache_text.lines().find_map(|line| {
        let (var, rest) = line.split_once(':')?;
        if var != name {
            return None;
        }
        rest.split_once('=').map(|(_, value)| value.trim())
    })
}

/// What a configure actually recorded. Provenance only: the runtime probe, not
/// this, authorizes any capability (plan §4.6).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConfigureEvidence {
    pub fa_quants_spec: Option<String>,
    pub fa_pairs: Option<Vec<(KvType, KvType)>>,
    pub cuda_architectures: Option<String>,
    pub backend_dl: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceError {
    /// The cache holds a FA spec this build does not understand.
    FaSpec(FaQuantsParseError),
}

impl std::fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FaSpec(e) => write!(f, "the configure recorded an unreadable FA_QUANTS: {e}"),
        }
    }
}

pub fn configure_evidence(cache_text: &str) -> Result<ConfigureEvidence, EvidenceError> {
    let mut evidence = ConfigureEvidence::default();
    if let Some(spec) = cache_value(cache_text, "GGML_CUDA_FA_QUANTS") {
        let pairs: BTreeSet<(KvType, KvType)> =
            compiled_fa_pairs(spec).map_err(EvidenceError::FaSpec)?;
        evidence.fa_quants_spec = Some(spec.to_string());
        evidence.fa_pairs = Some(pairs.into_iter().collect());
    }
    evidence.cuda_architectures = cache_value(cache_text, "CMAKE_CUDA_ARCHITECTURES")
        .filter(|v| !v.is_empty())
        .map(str::to_string);
    evidence.backend_dl = cache_value(cache_text, "GGML_BACKEND_DL").map(|v| v == "ON");
    Ok(evidence)
}

/// A CUDA build must have configured `FA_QUANTS=all`, whatever we asked for.
pub fn cuda_configure_is_fa_all(evidence: &ConfigureEvidence) -> bool {
    evidence.fa_quants_spec.as_deref() == Some(FA_BUILD_SPEC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_build::CudaPlan;
    use std::path::PathBuf;

    fn plan(accelerator: Accelerator, os: TargetOs) -> BuildPlan {
        BuildPlan {
            accelerator,
            os,
            source_dir: PathBuf::from("/app/llama-build/src"),
            build_dir: PathBuf::from(format!("/app/llama-build/build-{}", accelerator.as_str())),
            ccache: false,
            apple_silicon: true,
            cuda: None,
            stale_cache_vars: vec![],
        }
    }

    fn cuda_plan(version: &str, os: TargetOs) -> BuildPlan {
        let mut p = plan(Accelerator::Cuda, os);
        p.cuda = Some(CudaPlan {
            nvcc: PathBuf::from(r"C:\CUDA\v13.4\bin\nvcc.exe"),
            version: CudaVersion::parse(version).unwrap(),
            architectures: "120a-real".into(),
        });
        p
    }

    #[test]
    fn metal_golden() {
        let args = cmake_configure_args(&plan(Accelerator::Metal, TargetOs::Macos));
        assert_eq!(
            args,
            [
                "-S",
                "/app/llama-build/src",
                "-B",
                "/app/llama-build/build-metal",
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_TOOLS=ON",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DBUILD_SHARED_LIBS=ON",
                "-DGGML_BACKEND_DL=ON",
                "-DGGML_NATIVE=ON",
                "-DGGML_METAL=ON",
                "-DGGML_METAL_EMBED_LIBRARY=ON",
                "-DGGML_METAL_USE_BF16=ON",
                "-DGGML_BLAS=ON",
                "-DGGML_BLAS_VENDOR=Apple",
            ]
        );
    }

    #[test]
    fn metal_bf16_follows_apple_silicon() {
        let mut p = plan(Accelerator::Metal, TargetOs::Macos);
        p.apple_silicon = false;
        assert!(cmake_configure_args(&p).contains(&"-DGGML_METAL_USE_BF16=OFF".to_string()));
    }

    #[test]
    fn cpu_golden_has_no_gpu_flags() {
        let args = cmake_configure_args(&plan(Accelerator::Cpu, TargetOs::Linux));
        assert!(args.contains(&"-DGGML_NATIVE=ON".to_string()));
        for forbidden in ["METAL", "CUDA", "BLAS"] {
            assert!(!args.iter().any(|a| a.contains(forbidden)), "{forbidden}");
        }
    }

    #[test]
    fn windows_cuda_golden_before_ccci_cutoff() {
        let mut p = cuda_plan("13.4.48", TargetOs::Windows);
        p.source_dir = PathBuf::from(r"C:\app\llama-build\src");
        p.build_dir = PathBuf::from(r"C:\app\llama-build\build-cuda");
        p.ccache = true;
        let args = cmake_configure_args(&p);
        assert_eq!(
            args,
            [
                "-S",
                r"C:\app\llama-build\src",
                "-B",
                r"C:\app\llama-build\build-cuda",
                "-G",
                "Ninja Multi-Config",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_TOOLS=ON",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DBUILD_SHARED_LIBS=ON",
                "-DGGML_BACKEND_DL=ON",
                "-DGGML_NATIVE=OFF",
                "-DGGML_CCACHE=ON",
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                r"-DCMAKE_CUDA_COMPILER=C:\CUDA\v13.4\bin\nvcc.exe",
                "-DCMAKE_CUDA_HOST_COMPILER=cl.exe",
                "-DGGML_CUDA_CCCL_VERSION=v3.4.3",
                "-DCMAKE_CUDA_ARCHITECTURES=120a-real",
                "-DGGML_CUDA=ON",
                "-DGGML_CUDA_NCCL=OFF",
                "-DGGML_CUDA_COMPRESSION_MODE=speed",
                "-DGGML_CUDA_FA_QUANTS=all",
                "-DGGML_CPU_ALL_VARIANTS=ON",
            ]
        );
    }

    #[test]
    fn cccl_is_only_fetched_for_toolkits_that_do_not_bundle_it() {
        let has = |v: &str| {
            cmake_configure_args(&cuda_plan(v, TargetOs::Windows))
                .iter()
                .any(|a| a.starts_with("-DGGML_CUDA_CCCL_VERSION="))
        };
        assert!(has("12.8.93"));
        assert!(has("13.4.48"));
        assert!(has("13.4.99"));
        assert!(!has("13.5.0"));
        assert!(!has("13.6.1"));
        assert!(!has("14.0.0"));
    }

    #[test]
    fn cuda_always_requests_fa_all_and_never_the_deprecated_flag() {
        for os in [TargetOs::Windows, TargetOs::Linux] {
            for v in ["12.8.93", "13.4.48", "13.5.0"] {
                let args = cmake_configure_args(&cuda_plan(v, os));
                assert!(args.contains(&"-DGGML_CUDA_FA_QUANTS=all".to_string()));
                assert!(
                    !args.iter().any(|a| a.contains("FA_ALL_QUANTS")),
                    "deprecated flag emitted: {args:?}"
                );
            }
        }
        // And no non-CUDA build mentions it either.
        for acc in [Accelerator::Metal, Accelerator::Cpu] {
            let args = cmake_configure_args(&plan(acc, TargetOs::Macos));
            assert!(!args.iter().any(|a| a.contains("FA_")), "{acc:?}");
        }
    }

    #[test]
    fn linux_cuda_drops_the_msvc_pins_and_keeps_single_config() {
        let args = cmake_configure_args(&cuda_plan("13.4.48", TargetOs::Linux));
        assert!(args.contains(&"Ninja".to_string()));
        assert!(args.contains(&"-DCMAKE_BUILD_TYPE=Release".to_string()));
        assert!(!args.iter().any(|a| a.contains("CMAKE_C_COMPILER")));
        assert!(!args.iter().any(|a| a.contains("CUDA_HOST_COMPILER")));
    }

    #[test]
    fn multi_config_never_sets_a_build_type() {
        let args = cmake_configure_args(&cuda_plan("13.4.48", TargetOs::Windows));
        assert!(!args.iter().any(|a| a.contains("CMAKE_BUILD_TYPE")));
    }

    #[test]
    fn stale_cache_vars_become_unset_pairs_at_the_end() {
        let mut p = plan(Accelerator::Metal, TargetOs::Macos);
        p.stale_cache_vars = vec!["OPENSSL_ROOT_DIR".into(), "CMAKE_OSX_SYSROOT".into()];
        let args = cmake_configure_args(&p);
        let tail = &args[args.len() - 4..];
        assert_eq!(tail, ["-U", "OPENSSL_ROOT_DIR", "-U", "CMAKE_OSX_SYSROOT"]);
    }

    #[test]
    fn only_the_two_shipped_targets_are_built() {
        let p = plan(Accelerator::Metal, TargetOs::Macos);
        assert_eq!(
            cmake_build_args(&p, 7),
            [
                "--build",
                "/app/llama-build/build-metal",
                "--target",
                "llama-server",
                "llama-bench",
                "--parallel",
                "7"
            ]
        );
        let w = cuda_plan("13.4.48", TargetOs::Windows);
        let args = cmake_build_args(&w, 0);
        assert_eq!(args[2..4], ["--config", "Release"]);
        assert_eq!(args.last().unwrap(), "1");
    }

    #[test]
    fn output_locations_follow_the_generator() {
        assert_eq!(
            output_subdir(&cuda_plan("13.4.48", TargetOs::Windows)),
            "bin/Release"
        );
        assert_eq!(
            output_subdir(&plan(Accelerator::Metal, TargetOs::Macos)),
            "bin"
        );
        assert_eq!(
            ninja_file(&cuda_plan("13.4.48", TargetOs::Windows)),
            "build-Release.ninja"
        );
    }

    const CACHE: &str = "\
CMAKE_BUILD_TYPE:STRING=Release
OPENSSL_ROOT_DIR:PATH=/opt/homebrew/Cellar/openssl@3/3.5.0
OPENSSL_SSL_LIBRARY:FILEPATH=/opt/homebrew/Cellar/openssl@3/3.5.0/lib/libssl.dylib
CMAKE_INSTALL_PREFIX:PATH=/usr/local
CMAKE_INSTALL_OLDINCLUDEDIR:PATH=/usr/include
CMAKE_PREFIX_PATH:PATH=/a/gone;/b/gone
RELATIVE_THING:PATH=relative/dir
NOTFOUND_THING:FILEPATH=OPENSSL_CRYPTO_LIBRARY-NOTFOUND
GGML_CUDA_FA_QUANTS:STRING=all
CMAKE_CUDA_ARCHITECTURES:STRING=120a-real
GGML_BACKEND_DL:BOOL=ON
";

    #[test]
    fn stale_detection_follows_the_tooling_rules() {
        let alive =
            |p: &Path| p == Path::new("/opt/homebrew/Cellar/openssl@3/3.5.0/lib/libssl.dylib");
        let stale = find_stale_cache_paths(CACHE, &alive);
        assert_eq!(
            stale,
            [(
                "OPENSSL_ROOT_DIR".to_string(),
                "/opt/homebrew/Cellar/openssl@3/3.5.0".to_string()
            )]
        );
        // Install destinations, lists, relative values and -NOTFOUND never count.
        let nothing = |_: &Path| false;
        let names: Vec<_> = find_stale_cache_paths(CACHE, &nothing)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, ["OPENSSL_ROOT_DIR", "OPENSSL_SSL_LIBRARY"]);
    }

    #[test]
    fn windows_drive_paths_count_as_absolute() {
        let cache = "CUDAToolkit_ROOT:PATH=C:/Old/CUDA\nX:PATH=D:\\gone\n";
        let stale = find_stale_cache_paths(cache, &|_| false);
        assert_eq!(stale.len(), 2);
    }

    #[test]
    fn evidence_is_parsed_from_the_cache_not_assumed() {
        let ev = configure_evidence(CACHE).unwrap();
        assert_eq!(ev.fa_quants_spec.as_deref(), Some("all"));
        assert_eq!(ev.fa_pairs.as_ref().unwrap().len(), 49);
        assert_eq!(ev.cuda_architectures.as_deref(), Some("120a-real"));
        assert_eq!(ev.backend_dl, Some(true));
        assert!(cuda_configure_is_fa_all(&ev));
    }

    #[test]
    fn upstream_default_fa_list_is_not_all() {
        let cache = "GGML_CUDA_FA_QUANTS:STRING=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16\n";
        let ev = configure_evidence(cache).unwrap();
        assert!(!cuda_configure_is_fa_all(&ev));
        assert!(ev.fa_pairs.unwrap().len() < 49);
    }

    #[test]
    fn absent_or_malformed_fa_spec() {
        let ev = configure_evidence("CMAKE_BUILD_TYPE:STRING=Release\n").unwrap();
        assert_eq!(ev.fa_quants_spec, None);
        assert!(!cuda_configure_is_fa_all(&ev));
        let err = configure_evidence("GGML_CUDA_FA_QUANTS:STRING=q9_9-q4_0\n").unwrap_err();
        assert!(matches!(err, EvidenceError::FaSpec(_)));
    }

    #[test]
    fn cache_value_reads_the_exact_variable() {
        assert_eq!(cache_value(CACHE, "CMAKE_BUILD_TYPE"), Some("Release"));
        assert_eq!(cache_value(CACHE, "CMAKE_BUILD"), None);
    }
}
