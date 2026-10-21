//! GPU detection for builds: `nvidia-smi` parsing, compute capability →
//! `CMAKE_CUDA_ARCHITECTURES`, and Apple chip detection.
//!
//! Parsing is pure and fixture-tested; the `detect_*` functions only run a
//! command through a [`SystemProbe`].

use std::time::Duration;

use serde::Serialize;

use super::sysprobe::SystemProbe;

const SMI_TIMEOUT: Duration = Duration::from_secs(15);

/// NVIDIA compute capability → `CMAKE_CUDA_ARCHITECTURES`. Ported from the
/// llama.cpp tooling (upstream convention: `a` suffix for the architecture-
/// specific feature sets, `-real` where PTX is not wanted).
pub const CUDA_ARCH_BY_COMPUTE_CAP: [(&str, &str); 9] = [
    ("7.5", "75"),
    ("8.0", "80"),
    ("8.6", "86"),
    ("8.9", "89"),
    ("9.0", "90a"),
    ("10.0", "100a"),
    ("10.3", "103a"),
    ("12.0", "120a-real"),
    ("12.1", "121a-real"),
];

pub fn cuda_arch_for_compute_cap(compute_cap: &str) -> Option<&'static str> {
    let wanted = compute_cap.trim();
    CUDA_ARCH_BY_COMPUTE_CAP
        .iter()
        .find(|(cap, _)| *cap == wanted)
        .map(|(_, arch)| *arch)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NvidiaGpu {
    pub index: usize,
    pub name: String,
    pub compute_cap: String,
    pub memory_mib: u64,
    pub driver_version: Option<String>,
    /// `None` when the compute capability is not in the map: the UI then asks
    /// for a manual architecture instead of guessing.
    pub cuda_architecture: Option<String>,
}

/// Parse `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version
/// --format=csv,noheader`. Malformed lines are skipped, never guessed at.
pub fn parse_nvidia_smi(csv: &str) -> Vec<NvidiaGpu> {
    let mut gpus = Vec::new();
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 3 {
            continue;
        }
        let compute_cap = fields[1];
        if compute_cap.split('.').count() != 2
            || !compute_cap
                .split('.')
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
        {
            continue;
        }
        let memory_mib = fields[2]
            .split_whitespace()
            .next()
            .and_then(|n| n.parse::<u64>().ok());
        let Some(memory_mib) = memory_mib else {
            continue;
        };
        gpus.push(NvidiaGpu {
            index: gpus.len(),
            name: fields[0].to_string(),
            compute_cap: compute_cap.to_string(),
            memory_mib,
            driver_version: fields
                .get(3)
                .map(|d| d.to_string())
                .filter(|d| !d.is_empty()),
            cuda_architecture: cuda_arch_for_compute_cap(compute_cap).map(str::to_string),
        });
    }
    gpus
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum GpuDetectError {
    /// `nvidia-smi` is absent or could not run.
    NotAvailable(String),
    /// It ran but listed no parseable device.
    NoDevices(String),
}

impl std::fmt::Display for GpuDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable(e) => write!(f, "nvidia-smi is not available: {e}"),
            Self::NoDevices(e) => write!(f, "nvidia-smi listed no usable GPU: {e}"),
        }
    }
}

pub fn detect_nvidia(sys: &dyn SystemProbe) -> Result<Vec<NvidiaGpu>, GpuDetectError> {
    let exe = sys
        .which("nvidia-smi")
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "nvidia-smi".to_string());
    let out = sys
        .run(
            &exe,
            &[
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            SMI_TIMEOUT,
        )
        .map_err(GpuDetectError::NotAvailable)?;
    if !out.ok() {
        return Err(GpuDetectError::NotAvailable(
            out.combined().trim().chars().take(300).collect(),
        ));
    }
    let gpus = parse_nvidia_smi(&out.stdout);
    if gpus.is_empty() {
        return Err(GpuDetectError::NoDevices(
            out.stdout.trim().chars().take(300).collect(),
        ));
    }
    Ok(gpus)
}

/// A user-supplied `CMAKE_CUDA_ARCHITECTURES`. The value is passed to CMake as
/// one `-D` argument (never through a shell), but it is still restricted to the
/// grammar CMake accepts so a typo fails here rather than ten minutes in.
pub fn validate_cuda_architectures(value: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Err("the CUDA architecture list is empty".into());
    }
    if value.len() > 64 {
        return Err("the CUDA architecture list is too long".into());
    }
    let mut entries = Vec::new();
    for raw in value.split(';') {
        let entry = raw.trim();
        let (core, suffix) = match entry.find('-') {
            Some(at) => (&entry[..at], Some(&entry[at + 1..])),
            None => (entry, None),
        };
        let digits: String = core.chars().take_while(|c| c.is_ascii_digit()).collect();
        let tail = &core[digits.len()..];
        let numeric_ok = (2..=3).contains(&digits.len());
        let feature_ok = matches!(tail, "" | "a" | "f");
        let suffix_ok = matches!(suffix, None | Some("real") | Some("virtual"));
        if !(numeric_ok && feature_ok && suffix_ok) {
            return Err(format!(
                "`{entry}` is not a CUDA architecture (examples: 89, 90a, 120a-real)"
            ));
        }
        entries.push(entry.to_string());
    }
    Ok(entries.join(";"))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppleChip {
    pub name: String,
    pub apple_silicon: bool,
}

/// `Apple M5 Max` → Apple silicon; `Intel(R) Core(TM) i9` → not.
pub fn parse_apple_chip(brand: &str) -> Option<AppleChip> {
    let name = brand.trim();
    if name.is_empty() {
        return None;
    }
    Some(AppleChip {
        name: name.to_string(),
        apple_silicon: name.starts_with("Apple "),
    })
}

pub fn detect_apple_chip(sys: &dyn SystemProbe) -> Option<AppleChip> {
    let out = sys
        .run("sysctl", &["-n", "machdep.cpu.brand_string"], SMI_TIMEOUT)
        .ok()?;
    if !out.ok() {
        return None;
    }
    parse_apple_chip(&out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama_build::TargetOs;
    use crate::llama_build::sysprobe::FakeSystem;

    // Captured from ryne (RTX 5090, driver 581.x).
    const RYNE_SMI: &str = "NVIDIA GeForce RTX 5090, 12.0, 32607 MiB, 581.80\n";
    const MULTI_SMI: &str = "NVIDIA GeForce RTX 4090, 8.9, 24564 MiB, 560.94\n\
                             NVIDIA GeForce RTX 3090, 8.6, 24576 MiB, 560.94\n\
                             Tesla T4, 7.5, 15360 MiB, 560.94\n";

    #[test]
    fn parses_the_ryne_capture() {
        let gpus = parse_nvidia_smi(RYNE_SMI);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 5090");
        assert_eq!(gpus[0].compute_cap, "12.0");
        assert_eq!(gpus[0].memory_mib, 32607);
        assert_eq!(gpus[0].driver_version.as_deref(), Some("581.80"));
        assert_eq!(gpus[0].cuda_architecture.as_deref(), Some("120a-real"));
    }

    #[test]
    fn parses_several_devices_in_order() {
        let gpus = parse_nvidia_smi(MULTI_SMI);
        let archs: Vec<_> = gpus
            .iter()
            .map(|g| g.cuda_architecture.as_deref().unwrap())
            .collect();
        assert_eq!(archs, ["89", "86", "75"]);
        assert_eq!(gpus[2].index, 2);
    }

    #[test]
    fn unmapped_capability_has_no_guessed_architecture() {
        let gpus = parse_nvidia_smi("Future GPU, 14.0, 65536 MiB, 700.00\n");
        assert_eq!(gpus[0].cuda_architecture, None);
        assert_eq!(cuda_arch_for_compute_cap("7.0"), None);
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let gpus = parse_nvidia_smi(
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n\
             \n\
             Bad, notanumber, 100 MiB, 1.0\n\
             Bad2, 8.9, lots, 1.0\n\
             Good, 8.9, 100 MiB\n",
        );
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "Good");
        assert_eq!(gpus[0].driver_version, None);
    }

    #[test]
    fn the_architecture_map_is_the_ported_one() {
        for (cap, arch) in [
            ("7.5", "75"),
            ("8.0", "80"),
            ("8.6", "86"),
            ("8.9", "89"),
            ("9.0", "90a"),
            ("10.0", "100a"),
            ("10.3", "103a"),
            ("12.0", "120a-real"),
            ("12.1", "121a-real"),
        ] {
            assert_eq!(cuda_arch_for_compute_cap(cap), Some(arch));
        }
    }

    #[test]
    fn detect_nvidia_reports_missing_smi_and_empty_output() {
        let sys = FakeSystem::new(TargetOs::Windows);
        assert!(matches!(
            detect_nvidia(&sys),
            Err(GpuDetectError::NotAvailable(_))
        ));
        let sys = FakeSystem::new(TargetOs::Windows).on(
            "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader",
            "\n",
        );
        assert!(matches!(
            detect_nvidia(&sys),
            Err(GpuDetectError::NoDevices(_))
        ));
        let sys = FakeSystem::new(TargetOs::Windows).on_status(
            "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader",
            9,
            "",
            "NVIDIA-SMI has failed",
        );
        let err = detect_nvidia(&sys).unwrap_err();
        assert!(err.to_string().contains("has failed"), "{err}");
    }

    #[test]
    fn detect_nvidia_uses_the_resolved_smi_path() {
        let sys = FakeSystem::new(TargetOs::Windows)
            .with_program("nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe")
            .on(
                r"C:\Windows\System32\nvidia-smi.exe --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader",
                RYNE_SMI,
            );
        assert_eq!(detect_nvidia(&sys).unwrap().len(), 1);
    }

    #[test]
    fn architecture_overrides_follow_cmake_grammar() {
        assert_eq!(validate_cuda_architectures("89").unwrap(), "89");
        assert_eq!(
            validate_cuda_architectures(" 86 ; 120a-real ").unwrap(),
            "86;120a-real"
        );
        assert_eq!(validate_cuda_architectures("90a").unwrap(), "90a");
        for bad in [
            "",
            "native",
            "89;",
            "8",
            "1234",
            "89-fake",
            "89 -DFOO=1",
            "89;$(id)",
            "sm_89",
            &"8".repeat(80),
        ] {
            assert!(validate_cuda_architectures(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn apple_chip_detection() {
        let chip = parse_apple_chip("Apple M5 Max\n").unwrap();
        assert!(chip.apple_silicon);
        let intel = parse_apple_chip("Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz").unwrap();
        assert!(!intel.apple_silicon);
        assert!(parse_apple_chip("  \n").is_none());
        let sys = FakeSystem::new(TargetOs::Macos)
            .on("sysctl -n machdep.cpu.brand_string", "Apple M5 Max\n");
        assert_eq!(detect_apple_chip(&sys).unwrap().name, "Apple M5 Max");
        assert!(detect_apple_chip(&FakeSystem::new(TargetOs::Macos)).is_none());
    }
}
