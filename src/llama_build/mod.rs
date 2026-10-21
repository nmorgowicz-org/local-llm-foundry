//! Compiling llama.cpp from source inside the foundry (plan §4).
//!
//! The build backend is a set of small units with one job each:
//!
//! - `paths`: on-disk layout, containment, log retention.
//! - `gpu`: `nvidia-smi` parsing, compute capability → CUDA architecture.
//! - `configure`: the pure `(BuildPlan) -> cmake argv` function.
//! - `manifest`: provenance for a source build (never capability).
//! - `prereqs*`: the prerequisite checks and the disk-budget estimator.
//! - `toolchain_env`: the captured build environment vs the sanitized staged one.
//! - `source`: the managed git checkout and exact-commit pinning.
//! - `stage`: collecting the build outputs into a promotable candidate.
//! - `supervise`: owned process containment (Job Object / process group).
//! - `job`: the single-flight job state machine over a `Toolchain`.
//! - `service`: the process-wide owner of the active job, validation and caches
//!   behind the HTTP routes.
//!
//! Promotion itself is `crate::llama_runtime::promote`; this module only
//! produces a staged candidate and hands it over.

pub mod cache;
pub mod catalog;
pub mod cli;
pub mod configure;
pub mod doctor;
pub mod gpu;
pub mod job;
pub mod manifest;
pub mod orchestrate;
pub mod paths;
pub mod prereqs;
pub mod prereqs_macos;
pub mod prereqs_windows;
pub mod real;
pub mod repair;
pub mod service;
pub mod source;
pub mod stage;
pub mod supervise;
pub mod sysprobe;
pub mod toolchain_env;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::inference::kv_support::Backend;

/// What a build targets. The name is also the build-tree suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    Metal,
    Cuda,
    Cpu,
}

impl Accelerator {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    pub fn backend(self) -> Backend {
        match self {
            Self::Metal => Backend::Metal,
            Self::Cuda => Backend::Cuda,
            Self::Cpu => Backend::Cpu,
        }
    }

    /// Accelerators this host can plausibly build.
    pub fn available_on(os: TargetOs) -> &'static [Accelerator] {
        match os {
            TargetOs::Macos => &[Self::Metal, Self::Cpu],
            TargetOs::Windows | TargetOs::Linux => &[Self::Cuda, Self::Cpu],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetOs {
    Macos,
    Windows,
    Linux,
}

impl TargetOs {
    pub fn host() -> Self {
        if cfg!(target_os = "macos") {
            Self::Macos
        } else if cfg!(windows) {
            Self::Windows
        } else {
            Self::Linux
        }
    }

    pub fn exe_name(self, stem: &str) -> String {
        if self == Self::Windows {
            format!("{stem}.exe")
        } else {
            stem.to_string()
        }
    }
}

/// Which upstream revision to build (decision D5: only these two).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefChoice {
    LatestTag,
    Master,
}

/// The targets we build (decision D12). Nothing else is compiled or staged.
pub const BUILD_TARGETS: [&str; 2] = ["llama-server", "llama-bench"];

/// User-selectable options. Everything else about a build is derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildOptions {
    pub accelerator: Accelerator,
    pub reference: RefChoice,
    /// Overrides the architecture derived from the detected GPU.
    #[serde(default)]
    pub cuda_architectures: Option<String>,
    /// `None` = `max(1, logical_cpus - 1)`.
    #[serde(default)]
    pub parallel: Option<usize>,
    /// Leave headroom for a running model: half the CPUs and lowered priority.
    #[serde(default)]
    pub leave_headroom: bool,
    /// Discard the build tree and reconfigure from scratch.
    #[serde(default)]
    pub wipe_build_dir: bool,
}

impl BuildOptions {
    pub fn new(accelerator: Accelerator, reference: RefChoice) -> Self {
        Self {
            accelerator,
            reference,
            cuda_architectures: None,
            parallel: None,
            leave_headroom: false,
            wipe_build_dir: false,
        }
    }
}

/// Compile concurrency (plan §4.4).
pub fn effective_parallelism(options: &BuildOptions, logical_cpus: usize) -> usize {
    let cpus = logical_cpus.max(1);
    if let Some(explicit) = options.parallel {
        return explicit.clamp(1, cpus.max(1) * 2);
    }
    if options.leave_headroom {
        return (cpus / 2).max(1);
    }
    cpus.saturating_sub(1).max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generator {
    Ninja,
    NinjaMultiConfig,
}

impl Generator {
    pub fn cmake_name(self) -> &'static str {
        match self {
            Self::Ninja => "Ninja",
            Self::NinjaMultiConfig => "Ninja Multi-Config",
        }
    }
}

/// A parsed CUDA toolkit version, `major.minor.patch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CudaVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl CudaVersion {
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.trim().trim_start_matches('v').split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or_default();
        Some(Self {
            major,
            minor,
            patch,
        })
    }
}

impl std::fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaPlan {
    /// `<CUDA_PATH>\bin\nvcc.exe`, always absolute.
    pub nvcc: PathBuf,
    pub version: CudaVersion,
    /// `CMAKE_CUDA_ARCHITECTURES`.
    pub architectures: String,
}

/// Everything `configure::cmake_args` needs. Pure data: no probing happens
/// while turning a plan into arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildPlan {
    pub accelerator: Accelerator,
    pub os: TargetOs,
    /// Canonical, already created.
    pub source_dir: PathBuf,
    /// Canonical, already created.
    pub build_dir: PathBuf,
    pub ccache: bool,
    pub apple_silicon: bool,
    pub cuda: Option<CudaPlan>,
    /// Cached absolute paths that no longer exist (`cmake -U` each).
    pub stale_cache_vars: Vec<String>,
}

impl BuildPlan {
    pub fn generator(&self) -> Generator {
        if self.os == TargetOs::Windows && self.accelerator == Accelerator::Cuda {
            Generator::NinjaMultiConfig
        } else {
            Generator::Ninja
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallelism_defaults_leave_one_cpu() {
        let o = BuildOptions::new(Accelerator::Metal, RefChoice::Master);
        assert_eq!(effective_parallelism(&o, 18), 17);
        assert_eq!(effective_parallelism(&o, 1), 1);
        assert_eq!(effective_parallelism(&o, 2), 1);
    }

    #[test]
    fn headroom_halves_and_explicit_wins_within_bounds() {
        let mut o = BuildOptions::new(Accelerator::Cuda, RefChoice::LatestTag);
        o.leave_headroom = true;
        assert_eq!(effective_parallelism(&o, 32), 16);
        assert_eq!(effective_parallelism(&o, 1), 1);
        o.parallel = Some(4);
        assert_eq!(effective_parallelism(&o, 32), 4);
        o.parallel = Some(0);
        assert_eq!(effective_parallelism(&o, 8), 1);
        o.parallel = Some(10_000);
        assert_eq!(effective_parallelism(&o, 8), 16);
    }

    #[test]
    fn cuda_versions_parse_and_order() {
        let v = CudaVersion::parse("v13.4").unwrap();
        assert_eq!((v.major, v.minor, v.patch), (13, 4, 0));
        assert!(CudaVersion::parse("13.4.48").unwrap() < CudaVersion::parse("13.5.0").unwrap());
        assert!(CudaVersion::parse("garbage").is_none());
        assert_eq!(
            CudaVersion::parse("12.8.93").unwrap().to_string(),
            "12.8.93"
        );
    }

    #[test]
    fn windows_cuda_uses_multi_config_everything_else_ninja() {
        let mk = |accelerator, os| BuildPlan {
            accelerator,
            os,
            source_dir: "/s".into(),
            build_dir: "/b".into(),
            ccache: false,
            apple_silicon: false,
            cuda: None,
            stale_cache_vars: vec![],
        };
        assert_eq!(
            mk(Accelerator::Cuda, TargetOs::Windows).generator(),
            Generator::NinjaMultiConfig
        );
        assert_eq!(
            mk(Accelerator::Cuda, TargetOs::Linux).generator(),
            Generator::Ninja
        );
        assert_eq!(
            mk(Accelerator::Metal, TargetOs::Macos).generator(),
            Generator::Ninja
        );
    }

    #[test]
    fn exe_names_follow_the_target_os() {
        assert_eq!(
            TargetOs::Windows.exe_name("llama-server"),
            "llama-server.exe"
        );
        assert_eq!(TargetOs::Macos.exe_name("llama-server"), "llama-server");
    }
}
