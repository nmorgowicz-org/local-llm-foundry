#[cfg(target_os = "macos")]
pub mod apple;
pub mod dummy;
pub mod env;
#[cfg(target_os = "macos")]
pub mod mactop_cache;
pub mod nvidia;
pub mod rocm;
#[cfg(target_os = "windows")]
pub mod wmi_gpu;

use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GpuMetrics {
    #[serde(default)]
    pub temp: f32,
    #[serde(default)]
    pub load: u32,
    /// Origin of utilization, e.g. `ioreg`, `ioreg_suspect`, or `unavailable`.
    /// Absent for older snapshots and backends retaining legacy load semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_source: Option<String>,
    /// Whether load is an estimate rather than measured utilization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_estimated: Option<bool>,
    /// Explicit false means `load` is only a compatibility placeholder.
    /// None retains the legacy availability behavior for other backends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_available: Option<bool>,
    /// Power-state residency, not GPU utilization (Apple/mactop).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residency_percent: Option<f32>,
    #[serde(default)]
    pub power_consumption: f32,
    #[serde(default)]
    pub power_limit: u32,
    #[serde(default)]
    pub vram_used: u64,
    #[serde(default)]
    pub vram_total: u64,
    #[serde(default)]
    pub sclk_mhz: u32,
    #[serde(default)]
    pub mclk_mhz: u32,
    /// Apple Silicon only: current `iogpu.wired_limit_mb` sysctl value in MiB.
    /// 0 = system default (~66% of RAM for ≤36 GB systems, ~75% for larger).
    /// Non-zero = user has applied a custom Metal GPU wired memory cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub metal_gpu_limit_mb: Option<u64>,
}

pub trait GpuBackend: Send + Sync + 'static {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>>;
    /// Optional CPU/SoC temperature surfaced by the backend (Apple only for now).
    fn cpu_temp(&self) -> Option<f32> {
        None
    }
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

pub fn detect_backend(force: &str) -> Arc<dyn GpuBackend> {
    match force {
        #[cfg(target_os = "macos")]
        "apple" => Arc::new(apple::AppleBackend::new()),
        "rocm" => Arc::new(rocm::RocmBackend),
        "nvidia" => Arc::new(nvidia::NvidiaBackend),
        "none" => Arc::new(dummy::DummyBackend),
        #[cfg(target_os = "windows")]
        "wmi" => Arc::new(wmi_gpu::WmiGpuBackend),
        _ => {
            // Auto-detect: check which GPU tool is available
            #[cfg(target_os = "macos")]
            {
                if is_apple_silicon() {
                    return Arc::new(apple::AppleBackend::new());
                }
            }
            if command_exists("rocm-smi") {
                Arc::new(rocm::RocmBackend)
            } else if command_exists("nvidia-smi") {
                Arc::new(nvidia::NvidiaBackend)
            } else {
                // On Windows, fall back to WMI-based GPU discovery which surfaces
                // Intel, AMD, and other GPUs visible via Win32_VideoController.
                // Reports name and VRAM only (no temp/utilization without DXGI perf counters).
                #[cfg(target_os = "windows")]
                {
                    eprintln!(
                        "[info] No GPU CLI tool found (rocm-smi, nvidia-smi); \
                        using WMI GPU discovery (name/VRAM only)"
                    );
                    Arc::new(wmi_gpu::WmiGpuBackend)
                }
                #[cfg(not(target_os = "windows"))]
                {
                    eprintln!(
                        "[warn] No GPU monitoring tool found (apple/mactop, rocm-smi, nvidia-smi)"
                    );
                    Arc::new(dummy::DummyBackend)
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn is_apple_silicon() -> bool {
    // Check for Apple Silicon by looking at the CPU brand string
    let output = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok();

    if let Some(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        return stdout.contains("Apple")
            || stdout.contains("M")
            || stdout.contains("M1")
            || stdout.contains("M2")
            || stdout.contains("M3")
            || stdout.contains("M4")
            || stdout.contains("M5");
    }

    // Fallback: check if mactop is available
    command_exists("mactop")
}

fn command_exists(cmd: &str) -> bool {
    let finder = if cfg!(target_os = "windows") {
        "where"
    } else {
        "which"
    };

    let mut finder_cmd = std::process::Command::new(finder);
    crate::platform::no_window(&mut finder_cmd);
    finder_cmd
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[test]
    fn legacy_metrics_keep_default_availability_and_omit_provenance() {
        let metrics: GpuMetrics = serde_json::from_str(r#"{"load":37}"#).unwrap();
        assert_eq!(metrics.load, 37);
        assert_eq!(metrics.load_source, None);
        assert_eq!(metrics.load_estimated, None);
        assert_eq!(metrics.load_available, None);
        assert_eq!(metrics.residency_percent, None);
        let json = serde_json::to_value(metrics).unwrap();
        for key in [
            "load_source",
            "load_estimated",
            "load_available",
            "residency_percent",
        ] {
            assert!(
                json.get(key).is_none(),
                "{key} must not change legacy payloads"
            );
        }
    }

    #[test]
    fn unavailable_load_and_residency_round_trip_without_losing_false() {
        let metrics = GpuMetrics {
            load: 0,
            load_source: Some("ioreg_suspect".into()),
            load_estimated: Some(false),
            load_available: Some(false),
            residency_percent: Some(99.0),
            ..GpuMetrics::default()
        };
        let json = serde_json::to_value(&metrics).unwrap();
        assert_eq!(json["load"], 0);
        assert_eq!(json["load_source"], "ioreg_suspect");
        assert_eq!(json["load_estimated"], false);
        assert_eq!(json["load_available"], false);
        assert_eq!(json["residency_percent"], 99.0);
        let restored: GpuMetrics = serde_json::from_value(json).unwrap();
        assert_eq!(restored.load_available, Some(false));
        assert_eq!(restored.residency_percent, Some(99.0));
    }
}
