//! Prerequisite checks: the shared model, the cited CUDA driver matrix and the
//! disk-budget estimator (plan §4.4, §5).
//!
//! Each check has three tiers: *present*, *version* and *functional* (a smoke
//! test that actually compiles something). Platform modules produce the checks;
//! this module owns what they have in common.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::sysprobe::SystemProbe;
use super::{Accelerator, CudaVersion, TargetOs};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pass,
    Warn,
    Fail,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Fix {
    Copy {
        command: String,
    },
    Link {
        url: String,
        label: String,
    },
    /// A server-side action (macOS `brew install`), identified by id.
    Action {
        id: String,
        label: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Check {
    pub id: String,
    pub title: String,
    pub status: Status,
    pub detail: String,
    pub fix: Option<Fix>,
    /// Exact commands run and trimmed output, shown under "Details".
    pub evidence: Vec<String>,
}

impl Check {
    pub fn new(id: &str, title: &str, status: Status, detail: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            status,
            detail: detail.into(),
            fix: None,
            evidence: Vec::new(),
        }
    }

    pub fn pass(id: &str, title: &str, detail: impl Into<String>) -> Self {
        Self::new(id, title, Status::Pass, detail)
    }

    pub fn warn(id: &str, title: &str, detail: impl Into<String>) -> Self {
        Self::new(id, title, Status::Warn, detail)
    }

    pub fn fail(id: &str, title: &str, detail: impl Into<String>) -> Self {
        Self::new(id, title, Status::Fail, detail)
    }

    pub fn skipped(id: &str, title: &str, detail: impl Into<String>) -> Self {
        Self::new(id, title, Status::Skipped, detail)
    }

    pub fn with_fix(mut self, fix: Fix) -> Self {
        self.fix = Some(fix);
        self
    }

    pub fn with_evidence(mut self, line: impl Into<String>) -> Self {
        self.evidence.push(trim_evidence(&line.into()));
        self
    }
}

/// Evidence is shown to the user; keep it bounded.
pub fn trim_evidence(text: &str) -> String {
    const MAX: usize = 1200;
    let text = text.trim();
    if text.len() <= MAX {
        return text.to_string();
    }
    let mut end = MAX;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

pub fn copy_fix(command: &str) -> Fix {
    Fix::Copy {
        command: command.into(),
    }
}

pub fn link_fix(url: &str, label: &str) -> Fix {
    Fix::Link {
        url: url.into(),
        label: label.into(),
    }
}

pub fn action_fix(id: &str, label: &str) -> Fix {
    Fix::Action {
        id: id.into(),
        label: label.into(),
    }
}

/// Accumulates checks plus the facts that identify this toolchain. When the
/// fingerprint is unchanged a previous smoke-test result stays valid.
#[derive(Debug, Default)]
pub struct Collector {
    pub checks: Vec<Check>,
    parts: Vec<String>,
}

impl Collector {
    pub fn push(&mut self, check: Check) {
        self.checks.push(check);
    }

    pub fn fingerprint_part(&mut self, key: &str, value: &str) {
        self.parts.push(format!("{key}={value}"));
    }

    pub fn fingerprint(&self) -> String {
        let mut parts = self.parts.clone();
        parts.sort();
        let mut hasher = Sha256::new();
        for p in parts {
            hasher.update(p.as_bytes());
            hasher.update([0]);
        }
        hasher
            .finalize()
            .iter()
            .take(12)
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrereqReport {
    pub os: TargetOs,
    pub accelerator: Accelerator,
    pub checks: Vec<Check>,
    pub fingerprint: String,
    pub validated_at_unix: u64,
    /// No check failed. Warnings do not block a build.
    pub ready: bool,
}

impl PrereqReport {
    pub fn from_collector(os: TargetOs, accelerator: Accelerator, c: Collector) -> Self {
        let ready = !c.checks.iter().any(|ch| ch.status == Status::Fail);
        let fingerprint = c.fingerprint();
        Self {
            os,
            accelerator,
            checks: c.checks,
            fingerprint,
            validated_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            ready,
        }
    }

    pub fn check(&self, id: &str) -> Option<&Check> {
        self.checks.iter().find(|c| c.id == id)
    }

    pub fn failures(&self) -> impl Iterator<Item = &Check> {
        self.checks.iter().filter(|c| c.status == Status::Fail)
    }

    /// A previous passing smoke test may be reused only when nothing it
    /// depended on changed. The explicit "Validate" action never reuses it.
    pub fn smoke_reusable(&self, current_fingerprint: &str) -> bool {
        self.fingerprint == current_fingerprint
            && self
                .check("smoke")
                .is_some_and(|c| c.status == Status::Pass)
    }
}

/// Functional proof that the toolchain can compile. Implemented by the job
/// layer, which runs it under process supervision; tests inject a fake.
pub trait SmokeTest: Send + Sync {
    /// `toolchain` is the resolved Windows toolchain when there is one.
    fn run(
        &self,
        accelerator: Accelerator,
        toolchain: Option<&super::prereqs_windows::WindowsToolchain>,
    ) -> Result<String, String>;
}

/// No smoke test (fast path or fixtures).
pub struct NoSmoke;

impl SmokeTest for NoSmoke {
    fn run(
        &self,
        _accelerator: Accelerator,
        _toolchain: Option<&super::prereqs_windows::WindowsToolchain>,
    ) -> Result<String, String> {
        Err("not run".into())
    }
}

pub fn smoke_check(result: Result<String, String>, ran: bool) -> Check {
    if !ran {
        return Check::skipped(
            "smoke",
            "Compile smoke test",
            "Not run: an earlier prerequisite failed.",
        );
    }
    match result {
        Ok(log) => Check::pass(
            "smoke",
            "Compile smoke test",
            "A one-file project configured, compiled and linked.",
        )
        .with_evidence(log),
        Err(log) => Check::fail(
            "smoke",
            "Compile smoke test",
            "A one-file project did not build. The log below says why.",
        )
        .with_evidence(log),
    }
}

// ---------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------

/// The first `x.y[.z]` in `text`.
pub fn parse_version(text: &str) -> Option<(u32, u32, u32)> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric()) {
            let rest = &text[i..];
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let mut nums = token.split('.').filter(|p| !p.is_empty());
            if let (Some(a), Some(b)) = (nums.next(), nums.next())
                && let (Ok(a), Ok(b)) = (a.parse(), b.parse())
            {
                let c = nums.next().and_then(|p| p.parse().ok()).unwrap_or(0);
                return Some((a, b, c));
            }
            i += token.len().max(1);
        } else {
            i += 1;
        }
    }
    None
}

pub fn version_string(v: (u32, u32, u32)) -> String {
    format!("{}.{}.{}", v.0, v.1, v.2)
}

pub const CMAKE_MINIMUM: (u32, u32, u32) = (3, 21, 0);

// ---------------------------------------------------------------------------
// CUDA toolkit / driver compatibility (cited)
// ---------------------------------------------------------------------------

pub const CUDA_COMPAT_URL: &str =
    "https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html";

/// Minimum driver for minor-version compatibility, per toolkit major
/// (NVIDIA, "Minor Version Compatibility", Table 1: CUDA 13.x ≥ 580,
/// 12.x ≥ 525, 11.x ≥ 450). Majors not listed are unknown to this table.
pub const CUDA_MIN_DRIVER_BY_MAJOR: [(u32, u32); 3] = [(13, 580), (12, 525), (11, 450)];

/// Lowest CUDA toolkit this build supports; older fails (plan §5.2).
pub const CUDA_MINIMUM_TOOLKIT: (u32, u32) = (12, 4);
/// Newest toolkit line exercised (13.4 on ryne); 12.x is "untested".
pub const CUDA_TESTED_MAJOR: u32 = 13;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverVerdict {
    /// The driver meets the documented minimum for this toolkit major.
    Compatible {
        minimum: u32,
    },
    /// Documented incompatibility.
    Incompatible {
        minimum: u32,
    },
    /// The matrix has no entry for this toolkit major.
    Unknown,
    DriverUnparsed,
}

/// Compare a driver (`581.80`) with the documented minimum for a toolkit. The
/// `nvidia-smi` "CUDA Version" field is deliberately not an input: it is a
/// driver capability indicator, not an installed toolkit.
pub fn driver_verdict(toolkit: CudaVersion, driver: &str) -> DriverVerdict {
    let Some((driver_major, _, _)) = parse_version(driver) else {
        return DriverVerdict::DriverUnparsed;
    };
    match CUDA_MIN_DRIVER_BY_MAJOR
        .iter()
        .find(|(major, _)| *major == toolkit.major)
    {
        Some((_, minimum)) if driver_major >= *minimum => {
            DriverVerdict::Compatible { minimum: *minimum }
        }
        Some((_, minimum)) => DriverVerdict::Incompatible { minimum: *minimum },
        None => DriverVerdict::Unknown,
    }
}

/// PTX/feature caveats that apply under minor-version compatibility.
pub const MINOR_COMPAT_CAVEAT: &str = "Minor-version compatibility limits features and cannot run PTX \
     on an older driver; the backend check after the build is the real proof.";

/// Architectures that carry PTX (no `-real`) are the ones the caveat bites.
pub fn architectures_include_ptx(architectures: &str) -> bool {
    architectures
        .split(';')
        .any(|a| !a.trim().is_empty() && !a.trim().ends_with("-real"))
}

pub fn driver_check(
    toolkit: CudaVersion,
    driver: Option<&str>,
    architectures: Option<&str>,
) -> Check {
    const ID: &str = "nvidia_driver";
    const TITLE: &str = "NVIDIA driver";
    let fix = link_fix(CUDA_COMPAT_URL, "NVIDIA CUDA compatibility");
    let Some(driver) = driver else {
        return Check::warn(
            ID,
            TITLE,
            "The driver version could not be read; the post-build backend check will decide.",
        )
        .with_fix(fix);
    };
    match driver_verdict(toolkit, driver) {
        DriverVerdict::Compatible { minimum } => {
            let mut detail = format!(
                "Driver {driver} meets NVIDIA's documented minimum ({minimum}) for CUDA {}.x.",
                toolkit.major
            );
            if architectures.is_some_and(architectures_include_ptx) {
                detail.push(' ');
                detail.push_str(MINOR_COMPAT_CAVEAT);
                return Check::warn(ID, TITLE, detail).with_fix(fix);
            }
            Check::pass(ID, TITLE, detail)
        }
        DriverVerdict::Incompatible { minimum } => Check::fail(
            ID,
            TITLE,
            format!(
                "Driver {driver} is older than NVIDIA's documented minimum ({minimum}) for CUDA {}.x. \
                 Update the driver or choose an older toolkit.",
                toolkit.major
            ),
        )
        .with_fix(fix),
        DriverVerdict::Unknown => Check::warn(
            ID,
            TITLE,
            format!(
                "NVIDIA's compatibility table in use has no entry for CUDA {}.x, so the driver \
                 cannot be validated here. The post-build backend check will decide.",
                toolkit.major
            ),
        )
        .with_fix(fix),
        DriverVerdict::DriverUnparsed => Check::warn(
            ID,
            TITLE,
            format!("`{driver}` is not a recognizable driver version."),
        )
        .with_fix(fix),
    }
}

/// OS-specific CUDA 11/12 thresholds are from Table 2, not the driver bundled
/// with a toolkit (Table 3). NVIDIA's CUDA 13 table publishes branch 580.
/// https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver
/// https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver
pub fn minor_driver_minimum(os: TargetOs, major: u32) -> Option<(u32, u32, u32)> {
    match (major, os) {
        (13, TargetOs::Windows | TargetOs::Linux) => Some((580, 0, 0)),
        (12, TargetOs::Windows) => Some((528, 33, 0)),
        (12, TargetOs::Linux) => Some((525, 60, 13)),
        (11, TargetOs::Windows) => Some((452, 39, 0)),
        (11, TargetOs::Linux) => Some((450, 80, 2)),
        _ => None,
    }
}

pub fn driver_check_for_os(
    os: TargetOs,
    toolkit: CudaVersion,
    driver: Option<&str>,
    architectures: Option<&str>,
) -> Check {
    let mut check = driver_check(toolkit, driver, architectures);
    if let (Some(minimum), Some(actual)) = (
        minor_driver_minimum(os, toolkit.major),
        driver.and_then(parse_version),
    ) {
        let cited = format!(
            "OS-specific minor-compatibility minimum: {}. ",
            version_string(minimum)
        );
        if actual < minimum {
            check.status = Status::Fail;
            check.detail = format!(
                "{cited}Driver {} is too old for CUDA {}.x.",
                driver.unwrap_or_default(),
                toolkit.major
            );
        } else {
            check.detail = format!("{cited}{}", check.detail);
        }
        check.evidence.push(if toolkit.major == 13 {
            "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver"
        } else {
            "https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cuda-driver"
        }.into());
    }
    check
}

pub fn toolkit_version_check(version: Option<CudaVersion>) -> Check {
    const ID: &str = "cuda_toolkit";
    const TITLE: &str = "CUDA Toolkit";
    let fix = link_fix(
        "https://developer.nvidia.com/cuda-downloads",
        "NVIDIA CUDA downloads",
    );
    let Some(v) = version else {
        return Check::fail(
            ID,
            TITLE,
            "No CUDA Toolkit was found. Install one and set CUDA_PATH.",
        )
        .with_fix(fix);
    };
    if (v.major, v.minor) < CUDA_MINIMUM_TOOLKIT {
        return Check::fail(
            ID,
            TITLE,
            format!(
                "CUDA {v} is older than the supported minimum {}.{}.",
                CUDA_MINIMUM_TOOLKIT.0, CUDA_MINIMUM_TOOLKIT.1
            ),
        )
        .with_fix(fix);
    }
    if v.major < CUDA_TESTED_MAJOR {
        return Check::warn(
            ID,
            TITLE,
            format!(
                "CUDA {v} works but is untested here; {CUDA_TESTED_MAJOR}.x is the tested line."
            ),
        );
    }
    Check::pass(ID, TITLE, format!("CUDA {v}"))
}

// ---------------------------------------------------------------------------
// Disk budget (plan §4.4)
// ---------------------------------------------------------------------------

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Provisional floors until the FA-all peak has been measured on hardware.
pub fn disk_floor(os: TargetOs) -> u64 {
    match os {
        TargetOs::Windows => 20 * GIB,
        TargetOs::Macos => 2 * GIB,
        TargetOs::Linux => 10 * GIB,
    }
}

/// What the estimator knows. Deliberately **booleans and the ccache size only**:
/// the live and previous runtime sets and existing build trees are already
/// reflected in the free space, so their sizes are never inputs and cannot be
/// charged a second time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskInputs {
    pub os: TargetOs,
    pub accelerator: Accelerator,
    pub checkout_present: bool,
    pub build_tree_present: bool,
    pub ccache_enabled: bool,
    pub ccache_current_bytes: u64,
    pub ccache_cap_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiskEstimate {
    /// Additional allocations this job may make.
    pub additional_bytes: u64,
    pub floor_bytes: u64,
    /// `max(additional, floor)`.
    pub required_bytes: u64,
    pub assumptions: Vec<String>,
}

fn checkout_bytes() -> u64 {
    // Shallow-ish clone of the repository history.
    GIB
}

fn fresh_tree_bytes(accel: Accelerator) -> u64 {
    match accel {
        Accelerator::Metal | Accelerator::Cpu => 2 * GIB,
        // FA-all multiplies the CUDA template instances; the existing 1-2.4 GB
        // trees were not FA-all, so this stays a conservative placeholder.
        Accelerator::Cuda => 12 * GIB,
    }
}

fn staging_bytes(accel: Accelerator) -> u64 {
    match accel {
        Accelerator::Metal | Accelerator::Cpu => GIB / 2,
        Accelerator::Cuda => 3 * GIB,
    }
}

pub fn estimate_disk(inputs: &DiskInputs) -> DiskEstimate {
    let mut assumptions = Vec::new();
    let mut total = 0u64;
    if inputs.checkout_present {
        assumptions.push("Source checkout exists; only fetched objects are new.".into());
        total += 200 * MIB;
    } else {
        assumptions.push(format!(
            "New source checkout: about {} GiB.",
            checkout_bytes() / GIB
        ));
        total += checkout_bytes();
    }
    let tree = fresh_tree_bytes(inputs.accelerator);
    if inputs.build_tree_present {
        assumptions.push("Build tree exists; counted as incremental growth only.".into());
        total += tree / 4;
    } else {
        assumptions.push(format!(
            "New {} build tree: up to about {} GiB.",
            inputs.accelerator.as_str(),
            tree / GIB
        ));
        total += tree;
    }
    if inputs.ccache_enabled {
        let room = inputs
            .ccache_cap_bytes
            .saturating_sub(inputs.ccache_current_bytes);
        assumptions.push(format!(
            "ccache may grow up to its cap ({} MiB of room left).",
            room / MIB
        ));
        total += room;
    }
    total += staging_bytes(inputs.accelerator);
    assumptions.push("Candidate staging copy plus verification temporaries.".into());
    total += 256 * MIB;
    // Conservative margin rather than false precision.
    total += total / 4;
    assumptions.push("25% safety margin.".into());
    let floor = disk_floor(inputs.os);
    DiskEstimate {
        additional_bytes: total,
        floor_bytes: floor,
        required_bytes: total.max(floor),
        assumptions,
    }
}

/// Space still needed once the build has finished, checked before staging and
/// again before the journal (§4.4). Never satisfied by deleting a runtime set.
pub fn stage_requirement(accel: Accelerator) -> u64 {
    staging_bytes(accel) + 256 * MIB
}

pub fn disk_check(estimate: &DiskEstimate, free: Option<u64>) -> Check {
    const ID: &str = "disk";
    const TITLE: &str = "Disk space";
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / GIB as f64);
    let Some(free) = free else {
        return Check::warn(ID, TITLE, "Free space could not be read.");
    };
    let detail = format!(
        "{} free; this build needs about {} (floor {}).",
        gib(free),
        gib(estimate.required_bytes),
        gib(estimate.floor_bytes)
    );
    let mut check = if free >= estimate.required_bytes {
        Check::pass(ID, TITLE, detail)
    } else {
        Check::fail(
            ID,
            TITLE,
            format!(
                "{detail} Free space or use Clean build cache; runtime sets are never removed."
            ),
        )
    };
    for line in &estimate.assumptions {
        check = check.with_evidence(line.clone());
    }
    check
}

// ---------------------------------------------------------------------------
// Shared network checks
// ---------------------------------------------------------------------------

pub const LLAMA_CPP_REPO_URL: &str = "https://github.com/ggml-org/llama.cpp";
pub const HF_UI_URL: &str = "https://huggingface.co/ggml-org/llama-ui";
pub const CCCL_URL: &str = "https://github.com/nvidia/cccl";

/// `curl --head`: any HTTP answer proves the host is reachable.
pub fn https_reachable(sys: &dyn SystemProbe, url: &str) -> Result<(), String> {
    let out = sys
        .run(
            "curl",
            &[
                "-sS",
                "-I",
                "-o",
                null_device(sys.os()),
                "--max-time",
                "15",
                url,
            ],
            std::time::Duration::from_secs(20),
        )
        .map_err(|e| e.to_string())?;
    if out.ok() {
        Ok(())
    } else {
        Err(out.combined().trim().to_string())
    }
}

fn null_device(os: TargetOs) -> &'static str {
    if os == TargetOs::Windows {
        "NUL"
    } else {
        "/dev/null"
    }
}

pub fn network_checks(sys: &dyn SystemProbe, need_cccl: bool) -> Vec<Check> {
    let mut out = Vec::new();
    let mut hosts = vec![
        ("net_github", "GitHub reachable", LLAMA_CPP_REPO_URL),
        ("net_huggingface", "Hugging Face reachable", HF_UI_URL),
    ];
    if need_cccl {
        hosts.push(("net_cccl", "NVIDIA CCCL source reachable", CCCL_URL));
    }
    for (id, title, url) in hosts {
        let check = match https_reachable(sys, url) {
            Ok(()) => Check::pass(id, title, format!("{url} answered.")),
            Err(e) => Check::fail(
                id,
                title,
                format!(
                    "{url} did not answer ({e}). The build downloads from it: the web UI assets \
                     come from huggingface.co and sources from github.com."
                ),
            ),
        };
        out.push(check.with_evidence(format!("curl --head {url}")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_build::sysprobe::FakeSystem;

    fn v(text: &str) -> CudaVersion {
        CudaVersion::parse(text).unwrap()
    }

    #[test]
    fn versions_are_pulled_out_of_tool_banners() {
        assert_eq!(
            parse_version("cmake version 4.1.2\n\nCMake suite"),
            Some((4, 1, 2))
        );
        assert_eq!(
            parse_version("git version 2.50.1 (Apple Git-155)"),
            Some((2, 50, 1))
        );
        assert_eq!(parse_version("1.13.1"), Some((1, 13, 1)));
        assert_eq!(
            parse_version("Apple clang version 17.0.0 (clang-1700.0.13.5)"),
            Some((17, 0, 0))
        );
        assert_eq!(parse_version("no digits here"), None);
        assert_eq!(parse_version("build 7"), None);
        // Letters glued to digits are not a version start.
        assert_eq!(parse_version("sm_89 9.0"), Some((9, 0, 0)));
    }

    #[test]
    fn cmake_floor_is_3_21() {
        assert!(parse_version("cmake version 3.20.6").unwrap() < CMAKE_MINIMUM);
        assert!(parse_version("cmake version 3.21.0").unwrap() >= CMAKE_MINIMUM);
    }

    #[test]
    fn driver_matrix_is_the_cited_one() {
        assert_eq!(CUDA_MIN_DRIVER_BY_MAJOR, [(13, 580), (12, 525), (11, 450)]);
        assert_eq!(
            driver_verdict(v("13.4.48"), "581.80"),
            DriverVerdict::Compatible { minimum: 580 }
        );
        assert_eq!(
            driver_verdict(v("13.4.48"), "580.00"),
            DriverVerdict::Compatible { minimum: 580 }
        );
        assert_eq!(
            driver_verdict(v("13.0.0"), "576.88"),
            DriverVerdict::Incompatible { minimum: 580 }
        );
        assert_eq!(
            driver_verdict(v("12.8.93"), "560.94"),
            DriverVerdict::Compatible { minimum: 525 }
        );
        // A newer driver also satisfies an older toolkit line.
        assert_eq!(
            driver_verdict(v("12.8.93"), "581.80"),
            DriverVerdict::Compatible { minimum: 525 }
        );
        assert_eq!(
            driver_verdict(v("12.8.93"), "520.00"),
            DriverVerdict::Incompatible { minimum: 525 }
        );
    }

    #[test]
    fn unknown_toolkit_majors_warn_instead_of_inventing_a_minimum() {
        assert_eq!(
            driver_verdict(v("14.0.0"), "700.00"),
            DriverVerdict::Unknown
        );
        let check = driver_check(v("14.0.0"), Some("700.00"), Some("120a-real"));
        assert_eq!(check.status, Status::Warn);
        assert!(check.detail.contains("no entry"));
        assert_eq!(
            driver_verdict(v("13.4.0"), "garbage"),
            DriverVerdict::DriverUnparsed
        );
    }

    #[test]
    fn documented_incompatibility_fails_with_a_link() {
        let check = driver_check(v("13.4.48"), Some("560.94"), Some("120a-real"));
        assert_eq!(check.status, Status::Fail);
        assert!(matches!(check.fix, Some(Fix::Link { ref url, .. }) if url == CUDA_COMPAT_URL));
    }

    #[test]
    fn minor_compat_ptx_caveat_warns_only_when_ptx_is_in_play() {
        let real_only = driver_check(v("13.4.48"), Some("581.80"), Some("120a-real"));
        assert_eq!(real_only.status, Status::Pass);
        let with_ptx = driver_check(v("13.4.48"), Some("581.80"), Some("89;120a-real"));
        assert_eq!(with_ptx.status, Status::Warn);
        assert!(with_ptx.detail.contains("PTX"));
        assert!(architectures_include_ptx("90a"));
        assert!(!architectures_include_ptx("80-real;120a-real"));
    }

    #[test]
    fn the_smi_cuda_version_field_is_not_a_toolkit_input() {
        // The API takes a toolkit version and a driver version: there is no
        // parameter through which `CUDA Version: 13.0` from nvidia-smi could be
        // mistaken for an installed toolkit.
        let check = driver_check(v("12.8.93"), None, None);
        assert_eq!(check.status, Status::Warn);
    }

    #[test]
    fn toolkit_version_tiers() {
        assert_eq!(toolkit_version_check(None).status, Status::Fail);
        assert_eq!(
            toolkit_version_check(Some(v("12.3.0"))).status,
            Status::Fail
        );
        assert_eq!(
            toolkit_version_check(Some(v("12.4.0"))).status,
            Status::Warn
        );
        assert_eq!(
            toolkit_version_check(Some(v("13.4.48"))).status,
            Status::Pass
        );
    }

    fn inputs(os: TargetOs, accel: Accelerator) -> DiskInputs {
        DiskInputs {
            os,
            accelerator: accel,
            checkout_present: false,
            build_tree_present: false,
            ccache_enabled: true,
            ccache_current_bytes: 0,
            ccache_cap_bytes: 2 * GIB,
        }
    }

    #[test]
    fn floors_are_20_gib_windows_and_2_gib_macos() {
        assert_eq!(disk_floor(TargetOs::Windows), 20 * GIB);
        assert_eq!(disk_floor(TargetOs::Macos), 2 * GIB);
        let w = estimate_disk(&inputs(TargetOs::Windows, Accelerator::Cuda));
        assert!(w.required_bytes >= 20 * GIB);
        let m = estimate_disk(&inputs(TargetOs::Macos, Accelerator::Metal));
        assert!(m.required_bytes >= 2 * GIB);
    }

    #[test]
    fn estimate_exceeds_the_floor_when_the_additional_need_is_larger() {
        let mut big = inputs(TargetOs::Windows, Accelerator::Cuda);
        big.ccache_cap_bytes = 16 * GIB;
        let e = estimate_disk(&big);
        assert_eq!(e.required_bytes, e.additional_bytes.max(e.floor_bytes));
        assert!(e.additional_bytes > 20 * GIB);
        assert_eq!(e.required_bytes, e.additional_bytes);
    }

    #[test]
    fn existing_directories_are_not_charged_again() {
        let fresh = estimate_disk(&inputs(TargetOs::Windows, Accelerator::Cuda));
        let mut existing = inputs(TargetOs::Windows, Accelerator::Cuda);
        existing.checkout_present = true;
        existing.build_tree_present = true;
        let reuse = estimate_disk(&existing);
        assert!(reuse.additional_bytes < fresh.additional_bytes / 2);
        // The estimator has no input for live/previous runtime sizes at all.
        let a = estimate_disk(&existing);
        let b = estimate_disk(&existing);
        assert_eq!(a, b);
    }

    #[test]
    fn ccache_growth_is_bounded_by_the_remaining_room() {
        let mut full = inputs(TargetOs::Macos, Accelerator::Metal);
        full.ccache_current_bytes = 2 * GIB;
        let mut empty = full.clone();
        empty.ccache_current_bytes = 0;
        assert!(estimate_disk(&full).additional_bytes < estimate_disk(&empty).additional_bytes);
        let mut off = empty.clone();
        off.ccache_enabled = false;
        assert!(estimate_disk(&off).additional_bytes < estimate_disk(&empty).additional_bytes);
    }

    #[test]
    fn disk_check_fails_below_the_requirement_and_lists_assumptions() {
        let e = estimate_disk(&inputs(TargetOs::Windows, Accelerator::Cuda));
        let low = disk_check(&e, Some(5 * GIB));
        assert_eq!(low.status, Status::Fail);
        assert!(low.detail.contains("never removed"));
        assert!(!low.evidence.is_empty());
        assert_eq!(disk_check(&e, Some(e.required_bytes)).status, Status::Pass);
        assert_eq!(disk_check(&e, None).status, Status::Warn);
    }

    #[test]
    fn stage_requirement_is_small_next_to_the_whole_build() {
        let e = estimate_disk(&inputs(TargetOs::Windows, Accelerator::Cuda));
        assert!(stage_requirement(Accelerator::Cuda) < e.additional_bytes);
        assert!(stage_requirement(Accelerator::Cuda) > stage_requirement(Accelerator::Metal));
    }

    #[test]
    fn fingerprint_is_stable_order_independent_and_sensitive() {
        let mk = |parts: &[(&str, &str)]| {
            let mut c = Collector::default();
            for (k, v) in parts {
                c.fingerprint_part(k, v);
            }
            c.fingerprint()
        };
        let a = mk(&[("cmake", "4.1.2"), ("cuda", "13.4")]);
        assert_eq!(a, mk(&[("cuda", "13.4"), ("cmake", "4.1.2")]));
        assert_ne!(a, mk(&[("cmake", "4.1.3"), ("cuda", "13.4")]));
    }

    #[test]
    fn smoke_is_reused_only_with_an_unchanged_fingerprint() {
        let mut c = Collector::default();
        c.fingerprint_part("cmake", "4.1.2");
        c.push(smoke_check(Ok("ok".into()), true));
        let report = PrereqReport::from_collector(TargetOs::Macos, Accelerator::Metal, c);
        assert!(report.ready);
        assert!(report.smoke_reusable(&report.fingerprint));
        assert!(!report.smoke_reusable("different"));
        let mut c = Collector::default();
        c.push(smoke_check(Err("boom".into()), true));
        let failed = PrereqReport::from_collector(TargetOs::Macos, Accelerator::Metal, c);
        assert!(!failed.ready);
        assert!(!failed.smoke_reusable(&failed.fingerprint));
    }

    #[test]
    fn warnings_do_not_block_but_failures_do() {
        let mut c = Collector::default();
        c.push(Check::warn("ccache", "ccache", "missing"));
        assert!(PrereqReport::from_collector(TargetOs::Macos, Accelerator::Cpu, c).ready);
        let mut c = Collector::default();
        c.push(Check::fail("git", "git", "missing"));
        assert!(!PrereqReport::from_collector(TargetOs::Macos, Accelerator::Cpu, c).ready);
    }

    #[test]
    fn evidence_is_bounded() {
        let long = "x".repeat(5000);
        let c = Check::pass("a", "a", "ok").with_evidence(long);
        assert!(c.evidence[0].len() < 1300);
        assert!(c.evidence[0].ends_with('…'));
    }

    #[test]
    fn network_checks_name_the_hosts_the_build_downloads_from() {
        let sys = FakeSystem::new(TargetOs::Windows)
            .on(
                "curl -sS -I -o NUL --max-time 15 https://github.com/ggml-org/llama.cpp",
                "",
            )
            .on_status(
                "curl -sS -I -o NUL --max-time 15 https://huggingface.co/ggml-org/llama-ui",
                6,
                "",
                "curl: (6) Could not resolve host: huggingface.co",
            )
            .on(
                "curl -sS -I -o NUL --max-time 15 https://github.com/nvidia/cccl",
                "",
            );
        let checks = network_checks(&sys, true);
        assert_eq!(checks.len(), 3);
        assert_eq!(checks[0].status, Status::Pass);
        assert_eq!(checks[1].status, Status::Fail);
        assert!(checks[1].detail.contains("huggingface.co"));
        assert_eq!(network_checks(&sys, false).len(), 2);
    }
}

#[cfg(test)]
mod os_driver_tests {
    use super::*;

    #[test]
    fn release_note_minor_compatibility_boundaries_are_os_specific() {
        let cuda = CudaVersion::parse("12.9.86").unwrap();
        for (os, below, minimum) in [
            (TargetOs::Windows, "528.32", "528.33"),
            (TargetOs::Linux, "525.60.12", "525.60.13"),
        ] {
            assert_eq!(
                driver_check_for_os(os, cuda, Some(below), Some("89-real")).status,
                Status::Fail
            );
            assert_eq!(
                driver_check_for_os(os, cuda, Some(minimum), Some("89-real")).status,
                Status::Pass
            );
            assert_eq!(
                driver_check_for_os(os, cuda, Some(minimum), Some("89")).status,
                Status::Warn
            );
        }
        assert_eq!(
            minor_driver_minimum(TargetOs::Windows, 11),
            Some((452, 39, 0))
        );
        assert_eq!(
            minor_driver_minimum(TargetOs::Linux, 11),
            Some((450, 80, 2))
        );
        assert_eq!(minor_driver_minimum(TargetOs::Windows, 14), None);
    }
}
