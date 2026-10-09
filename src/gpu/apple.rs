use anyhow::Result;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use super::mactop_cache::{self, MactopCacheEntry};
use super::{GpuBackend, GpuMetrics};

#[derive(Deserialize)]
struct MactopOutput {
    soc_metrics: SocMetrics,
    memory: MemoryMetrics,
    // gpu_usage is redundant with soc_metrics.gpu_active
}

#[derive(Deserialize)]
struct SocMetrics {
    #[serde(default)]
    gpu_power: f64,
    #[serde(default)]
    cpu_power: f64,
    #[serde(default)]
    total_power: f64,
    #[serde(default)]
    gpu_freq_mhz: f64,
    #[serde(default)]
    gpu_temp: f64,
    #[serde(default)]
    gpu_active: Option<f64>,
    #[serde(default)]
    cpu_temp: f64,
    #[serde(default)]
    dram_read_bw_gbs: f64,
    #[serde(default)]
    dram_write_bw_gbs: f64,
    /// Current P-cluster frequency (MHz)
    #[serde(default)]
    p_cluster_freq_mhz: f64,
    /// Current S-cluster frequency (MHz)
    #[serde(default)]
    s_cluster_freq_mhz: f64,
    /// Current E-cluster frequency (MHz)
    #[serde(default)]
    e_cluster_freq_mhz: f64,
    /// P-cluster utilization (%)
    #[serde(default)]
    p_cluster_active: f64,
    /// S-cluster utilization (%)
    #[serde(default)]
    s_cluster_active: f64,
    /// E-cluster utilization (%)
    #[serde(default)]
    e_cluster_active: f64,
}

#[derive(Deserialize)]
struct MemoryMetrics {
    total: u64, // bytes
    used: u64,  // bytes
}

pub struct AppleBackend {
    last_cpu_temp: Mutex<f32>,
}

impl Default for AppleBackend {
    fn default() -> Self {
        AppleBackend {
            last_cpu_temp: Mutex::new(0.0),
        }
    }
}
impl AppleBackend {
    pub fn new() -> Self {
        AppleBackend {
            last_cpu_temp: Mutex::new(0.0),
        }
    }
}

impl GpuBackend for AppleBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let output = Command::new("mactop")
            .args(["--headless", "--count", "1", "--format", "json"])
            .output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "mactop failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let mut mactop_vec: Vec<MactopOutput> = serde_json::from_slice(&output.stdout)
            .map_err(|e| anyhow::anyhow!("Failed to parse mactop JSON: {}", e))?;
        let mactop_output = mactop_vec
            .pop()
            .ok_or_else(|| anyhow::anyhow!("mactop returned empty JSON array"))?;

        // Cache CPU/SoC temperature for cpu_temp()
        if mactop_output.soc_metrics.cpu_temp > 0.0
            && let Ok(mut t) = self.last_cpu_temp.lock()
        {
            *t = mactop_output.soc_metrics.cpu_temp as f32;
        }

        // Populate shared mactop cache for system.rs to read cluster frequency / power / residency.
        let soc = &mactop_output.soc_metrics;
        mactop_cache::set_cache(MactopCacheEntry {
            power_total_w: soc.total_power as f32,
            power_cpu_w: soc.cpu_power as f32,
            power_gpu_w: soc.gpu_power as f32,
            p_cluster_freq_mhz: soc.p_cluster_freq_mhz as u32,
            s_cluster_freq_mhz: soc.s_cluster_freq_mhz as u32,
            e_cluster_freq_mhz: soc.e_cluster_freq_mhz as u32,
            p_cluster_active: soc.p_cluster_active as f32,
            s_cluster_active: soc.s_cluster_active as f32,
            e_cluster_active: soc.e_cluster_active as f32,
        });

        // Convert bytes to MB
        let vram_total_mb = mactop_output.memory.total / (1024 * 1024);
        let vram_used_mb = mactop_output.memory.used / (1024 * 1024);

        // Estimate memory clock from DRAM bandwidth
        // Approximate: MCLK = (dram_bw_gbs * 1000) / 8 / 2 (DDR)
        let mclk_mhz = (soc.dram_read_bw_gbs + soc.dram_write_bw_gbs) * 1000.0 / 16.0;

        // Utilization and power-state residency are distinct signals. In
        // particular, zero ioreg utilization under >5 W is suspect, not proof
        // of idleness or a reason to substitute residency as utilization.
        let utilization =
            apple_utilization_metrics(read_ioreg_gpu_utilization(), soc.gpu_power, soc.gpu_active);
        let metrics = GpuMetrics {
            temp: soc.gpu_temp as f32,
            power_consumption: soc.gpu_power as f32,
            power_limit: 0, // Not available from mactop
            vram_used: vram_used_mb as u64,
            vram_total: vram_total_mb as u64,
            sclk_mhz: soc.gpu_freq_mhz as u32,
            mclk_mhz: mclk_mhz as u32,
            metal_gpu_limit_mb: Some(read_iogpu_wired_limit_mb()),
            ..utilization
        };

        let mut map = BTreeMap::new();
        map.insert(format!("GPU0 {}", detect_chip_name()), metrics);
        Ok(map)
    }

    fn cpu_temp(&self) -> Option<f32> {
        let t = *self.last_cpu_temp.lock().ok()?;
        if t > 0.0 { Some(t) } else { None }
    }

    fn name(&self) -> &str {
        "apple"
    }
}

/// Populate only the utilization/provenance fields, keeping residency separate.
/// `load = 0` is a legacy scalar placeholder when availability is false.
fn apple_utilization_metrics(
    ioreg_load: Option<u32>,
    gpu_power: f64,
    residency: Option<f64>,
) -> GpuMetrics {
    let (load, source, available) = match ioreg_load {
        Some(0) if gpu_power > 5.0 => (0, "ioreg_suspect", false),
        Some(load) => (load, "ioreg", true),
        None => (0, "unavailable", false),
    };
    GpuMetrics {
        load,
        load_source: Some(source.into()),
        load_estimated: Some(false),
        load_available: Some(available),
        residency_percent: residency
            .filter(|value| value.is_finite() && (0.0..=100.0).contains(value))
            .map(|value| value as f32),
        ..GpuMetrics::default()
    }
}

fn detect_chip_name() -> &'static str {
    static CHIP_NAME: OnceLock<String> = OnceLock::new();
    CHIP_NAME.get_or_init(|| {
        std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Apple Silicon".to_string())
    })
}

/// Real GPU utilization from the IOAccelerator's PerformanceStatistics
/// ("Device Utilization %"). mactop's gpu_active is a power-state residency
/// ratio that reads 70-90% on an idle GPU, so this is the trustworthy load
/// source on Apple Silicon — no sudo required.
fn read_ioreg_gpu_utilization() -> Option<u32> {
    let output = Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-k", "PerformanceStatistics"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_ioreg_gpu_utilization(&text)
}

/// Parse the inline dictionaries emitted by `ioreg -w 0`, not renderer/tiler
/// counters or similarly named properties. Missing or invalid data stays None
/// so the caller can explicitly report unavailable utilization.
fn parse_ioreg_gpu_utilization(text: &str) -> Option<u32> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim_start_matches(|c: char| c.is_whitespace() || c == '|');
            let (key, value) = line.split_once('=')?;
            if key.trim() != "\"PerformanceStatistics\"" {
                return None;
            }
            value.trim().strip_prefix('{')?.strip_suffix('}')
        })
        .flat_map(|statistics| statistics.split(','))
        .filter_map(|field| {
            let (key, value) = field.split_once('=')?;
            if !matches!(
                key.trim(),
                "\"Device Utilization %\"" | "Device Utilization %"
            ) {
                return None;
            }
            let value = value.trim();
            // Parse the whole unsigned integer, never a numeric prefix or sign.
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            value.parse::<u32>().ok().filter(|v| *v <= 100)
        })
        // Integrated and discrete accelerators may each expose statistics.
        .max()
}

/// Read `iogpu.wired_limit_mb` from the kernel each call.
/// Returns 0 if unset (system default: ~66% for ≤36 GB RAM, ~75% for larger).
/// Not cached — the value changes when the user applies the Metal GPU limit tweak.
/// The sysctl call takes ~10–50 µs, negligible against the metrics poll interval.
pub fn read_iogpu_wired_limit_mb() -> u64 {
    std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "iogpu.wired_limit_mb"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Hard ceiling fraction for the wired limit (95% of RAM).
/// This is the absolute maximum; users who set wired limit near this value
/// are accepting the risk of system instability under heavy load.
/// The recommended default is much lower (RAM minus 8GB reserve).
const WIRED_LIMIT_HARD_CEILING_FRACTION: f64 = 0.95;

/// Minimum RAM reserve (GiB) for systems <24GB.
/// Smaller systems need more headroom for OS stability.
const WIRED_LIMIT_RESERVE_SMALL_SYSTEM_GIB: u64 = 6;

/// Default RAM reserve (GiB) for systems ≥24GB.
/// Flat 8GB reserve works from 24GB up to 192GB+; users who want more GPU
/// can override via Settings (Phase 7) or POST /api/system/wired-limit.
const WIRED_LIMIT_RESERVE_DEFAULT_GIB: u64 = 8;

/// Compute the maximum allowed wired limit in MiB for this machine.
/// Returns None if total RAM cannot be determined.
/// Hard ceiling: 95% of total RAM (absolute limit, never exceeds).
pub fn wired_limit_max_mb(total_ram_bytes: u64) -> Option<u64> {
    if total_ram_bytes == 0 {
        return None;
    }
    let total_ram_mb = total_ram_bytes / (1024 * 1024);
    let hard_ceiling_mb = (total_ram_mb as f64 * WIRED_LIMIT_HARD_CEILING_FRACTION) as u64;
    Some(hard_ceiling_mb.max(1))
}

/// Compute the RAM-relative safe default wired limit when sysctl is unset (0).
/// Tiered by RAM size:
/// - <24 GB: total - 6 GB reserve (protects small systems from swap thrashing;
///   covers real 16/18 GB Apple Silicon configs, not just ≤16 GB)
/// - ≥24 GB: total - 8 GB reserve (matches user-verified 64 GB path: 57,344 MiB)
///
/// This is the configured_ceiling_bytes default used by MemoryAvailabilitySnapshot
/// and the recommended value exposed in Settings (Phase 7).
pub fn wired_limit_safe_default_mb(total_ram_bytes: u64) -> Option<u64> {
    if total_ram_bytes == 0 {
        return None;
    }
    let total_ram_mb = total_ram_bytes / (1024 * 1024);
    let total_ram_gb = total_ram_mb / 1024;
    let reserve_mb = if total_ram_gb < 24 {
        WIRED_LIMIT_RESERVE_SMALL_SYSTEM_GIB * 1024
    } else {
        WIRED_LIMIT_RESERVE_DEFAULT_GIB * 1024
    };
    Some((total_ram_mb as i64 - reserve_mb as i64).max(1) as u64)
}

/// Documented behavior notes (for API responses and frontend teaching):
/// - Persistence: When set via set-metal-gpu-limit endpoint, the value is saved
///   to /etc/sysctl.conf and persists across reboots.
/// - MLX restart required: MLX queries `iogpu.wired_limit_mb` at Metal device
///   initialization. An existing MLX/Rapid process does NOT pick up a new value.
///   Restart the model runtime after changing this limit.
pub fn wired_limit_behavior_notes() -> &'static str {
    "The iogpu.wired_limit_mb value is saved to /etc/sysctl.conf and persists \
     across reboots. MLX reads the value at device init; restart the runtime \
     after changing this limit for it to take effect."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero_ioreg_is_measured_even_with_high_power_or_residency() {
        let metrics = apple_utilization_metrics(Some(37), 50.0, Some(99.0));
        assert_eq!(metrics.load, 37);
        assert_eq!(metrics.load_source.as_deref(), Some("ioreg"));
        assert_eq!(metrics.load_estimated, Some(false));
        assert_eq!(metrics.load_available, Some(true));
        assert_eq!(metrics.residency_percent, Some(99.0));
    }

    #[test]
    fn zero_ioreg_with_power_above_five_watts_is_suspect_not_residency() {
        let metrics = apple_utilization_metrics(Some(0), 5.01, Some(99.0));
        assert_eq!(metrics.load, 0);
        assert_eq!(metrics.load_source.as_deref(), Some("ioreg_suspect"));
        assert_eq!(metrics.load_estimated, Some(false));
        assert_eq!(metrics.load_available, Some(false));
        assert_eq!(metrics.residency_percent, Some(99.0));
    }

    #[test]
    fn zero_ioreg_at_or_below_five_watts_remains_measured_zero() {
        for power in [0.0, 4.0, 5.0] {
            let metrics = apple_utilization_metrics(Some(0), power, Some(85.0));
            assert_eq!(metrics.load, 0);
            assert_eq!(metrics.load_source.as_deref(), Some("ioreg"));
            assert_eq!(metrics.load_available, Some(true));
        }
    }

    #[test]
    fn missing_ioreg_is_unknown_regardless_of_power() {
        for power in [0.0, 4.0, 5.0, 50.0] {
            let metrics = apple_utilization_metrics(None, power, Some(99.0));
            assert_eq!(metrics.load, 0);
            assert_eq!(metrics.load_source.as_deref(), Some("unavailable"));
            assert_eq!(metrics.load_available, Some(false));
            assert_eq!(metrics.residency_percent, Some(99.0));
        }
    }

    #[test]
    fn missing_or_invalid_residency_does_not_become_zero() {
        for residency in [
            None,
            Some(-1.0),
            Some(101.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            let metrics = apple_utilization_metrics(Some(25), 10.0, residency);
            assert_eq!(metrics.residency_percent, None);
            assert_eq!(metrics.load, 25);
            assert_eq!(metrics.load_available, Some(true));
        }
        for residency in [0.0, 100.0] {
            let metrics = apple_utilization_metrics(None, 0.0, Some(residency));
            assert_eq!(metrics.residency_percent, Some(residency as f32));
        }
    }

    #[test]
    fn absent_mactop_residency_is_not_fabricated() {
        let soc: SocMetrics = serde_json::from_str("{}").unwrap();
        assert_eq!(soc.gpu_active, None);
        let metrics = apple_utilization_metrics(None, soc.gpu_power, soc.gpu_active);
        let json = serde_json::to_value(metrics).unwrap();
        assert!(json.get("residency_percent").is_none());
        assert_eq!(json["load_available"], false);
    }

    #[test]
    fn ioreg_utilization_parses_real_quoted_statistics_fixture() {
        // Captured with the production ioreg arguments; unrelated properties omitted.
        let text = include_str!("fixtures/ioreg-performance-statistics.txt");
        assert_eq!(parse_ioreg_gpu_utilization(text), Some(0));
    }

    #[test]
    fn ioreg_utilization_tolerates_whitespace() {
        let text = "\t|   \"PerformanceStatistics\" \t= \t{ \
            \"Device Utilization %\" \t = \t 42 \t, \"Renderer Utilization %\"=99 } \r\n";
        assert_eq!(parse_ioreg_gpu_utilization(text), Some(42));
    }

    #[test]
    fn ioreg_utilization_takes_maximum_across_accelerators() {
        let text = r#"
+-o AGXAccelerator  <class AGXAccelerator, registered, matched, active>
    {
      "PerformanceStatistics" = {"Device Utilization %"=17}
    }
+-o AMDAccelerator  <class AMDAccelerator, registered, matched, active>
    {
      "PerformanceStatistics" = {"Device Utilization %"=83}
    }
+-o OtherAccelerator  <class OtherAccelerator, registered, matched, active>
    {
      "PerformanceStatistics" = {"Device Utilization %"=24}
    }
"#;
        assert_eq!(parse_ioreg_gpu_utilization(text), Some(83));
    }

    #[test]
    fn ioreg_utilization_accepts_zero_and_full_utilization() {
        for value in [0, 100] {
            let text = format!("\"PerformanceStatistics\" = {{\"Device Utilization %\"={value}}}");
            assert_eq!(parse_ioreg_gpu_utilization(&text), Some(value));
        }
    }

    #[test]
    fn ioreg_utilization_accepts_legacy_unquoted_key() {
        let text = r#""PerformanceStatistics" = {Device Utilization %=31}"#;
        assert_eq!(parse_ioreg_gpu_utilization(text), Some(31));
    }

    #[test]
    fn ioreg_utilization_rejects_missing_and_unrelated_fields() {
        for text in [
            "",
            r#""PerformanceStatistics" = {}"#,
            r#""PerformanceStatistics" = {"Renderer Utilization %"=99,"Tiler Utilization %"=87}"#,
            r#""PerformanceStatistics" = {"Other Device Utilization %"=99}"#,
            r#""PerformanceStatistics" = {"Device Utilization % extra"=99}"#,
            r#""OtherStatistics" = {"Device Utilization %"=99}"#,
            r#""Device Utilization %" = 99"#,
            r#""PerformanceStatisticsExtra" = {"Device Utilization %"=99}"#,
            r#""Description" = "PerformanceStatistics = {Device Utilization %=99}""#,
        ] {
            assert_eq!(parse_ioreg_gpu_utilization(text), None, "{text}");
        }
    }

    #[test]
    fn ioreg_utilization_rejects_malformed_values() {
        for value in [
            "",
            "-1",
            "+42",
            "101",
            "4294967296",
            "18446744073709551616",
            "42.5",
            "42%",
            "42junk",
            "4 2",
            "\"42\"",
            "true",
            "<2a>",
        ] {
            let text = format!("\"PerformanceStatistics\" = {{\"Device Utilization %\"={value}}}");
            assert_eq!(parse_ioreg_gpu_utilization(&text), None, "{value}");
        }
        for text in [
            r#""PerformanceStatistics" = {"Device Utilization %"42}"#,
            r#""PerformanceStatistics" = {"Device Utilization %"==42}"#,
            r#""PerformanceStatistics" = {"Device Utilization %=42}"#,
            r#""PerformanceStatistics" = {"Device Utilization %"=42"#,
        ] {
            assert_eq!(parse_ioreg_gpu_utilization(text), None, "{text}");
        }
    }

    #[test]
    fn ioreg_utilization_keeps_valid_data_when_an_accelerator_is_malformed() {
        let text = r#"
    "PerformanceStatistics" = {"Device Utilization %"=99junk}
    "PerformanceStatistics" = {"Device Utilization %"=37,"Renderer Utilization %"=100}
    "PerformanceStatistics" = {"Device Utilization %"=101}
"#;
        assert_eq!(parse_ioreg_gpu_utilization(text), Some(37));
    }

    /// 8 GiB RAM system (base M1/M2)
    const RAM_8GB_BYTES: u64 = 8 * 1024 * 1024 * 1024;
    /// 16 GiB RAM system (common config)
    const RAM_16GB_BYTES: u64 = 16 * 1024 * 1024 * 1024;
    /// 64 GiB RAM system (M5 Max class)
    const RAM_64GB_BYTES: u64 = 64 * 1024 * 1024 * 1024;
    /// 128 GiB RAM system (M1 Ultra / M4 Max class)
    const RAM_128GB_BYTES: u64 = 128 * 1024 * 1024 * 1024;

    #[test]
    fn wired_limit_max_mb_95_pct_hard_ceiling() {
        // 64GB: hard ceiling = 95% of 65536 = 62259
        let max = wired_limit_max_mb(RAM_64GB_BYTES).unwrap();
        let expected = (65_536_f64 * 0.95) as u64;
        assert_eq!(max, expected);
    }

    #[test]
    fn wired_limit_max_mb_zero_ram() {
        assert_eq!(wired_limit_max_mb(0), None);
    }

    #[test]
    fn wired_limit_safe_default_mb_16gb_reserve_6gb() {
        // ≤16GB: total - 6GB reserve = 10240 MB
        let default_mb = wired_limit_safe_default_mb(RAM_16GB_BYTES).unwrap();
        assert_eq!(default_mb, 10_240, "16GB system should reserve 6GB");
    }

    #[test]
    fn wired_limit_safe_default_mb_8gb_reserve_6gb() {
        // ≤16GB: total - 6GB reserve = 2048 MB (8GB - 6GB)
        let default_mb = wired_limit_safe_default_mb(RAM_8GB_BYTES).unwrap();
        assert_eq!(default_mb, 2_048, "8GB system should reserve 6GB");
    }

    #[test]
    fn wired_limit_safe_default_mb_64gb_reserve_8gb() {
        // ≥24GB: total - 8GB reserve = 57344 MB (matches user-verified path)
        let default_mb = wired_limit_safe_default_mb(RAM_64GB_BYTES).unwrap();
        assert_eq!(
            default_mb, 57_344,
            "64GB system should reserve 8GB (matches user-verified 57344 path)"
        );
    }

    #[test]
    fn wired_limit_safe_default_mb_128gb_reserve_8gb() {
        // ≥24GB: total - 8GB reserve = 131072 - 8192 = 122880 MB
        let default_mb = wired_limit_safe_default_mb(RAM_128GB_BYTES).unwrap();
        assert_eq!(default_mb, 122_880, "128GB system should reserve 8GB");
    }

    #[test]
    fn wired_limit_safe_default_mb_zero_ram() {
        assert_eq!(wired_limit_safe_default_mb(0), None);
    }

    #[test]
    fn m5_max_57344_path_within_bounds() {
        let max = wired_limit_max_mb(RAM_64GB_BYTES).unwrap();
        assert!(
            57_344 <= max,
            "M5 Max verified path 57344 must be within bounds (max={})",
            max
        );
    }

    #[test]
    fn behavior_notes_contain_required_info() {
        let notes = wired_limit_behavior_notes();
        assert!(
            notes.contains("persist") || notes.contains("sysctl.conf"),
            "Notes must indicate persistent behavior"
        );
        assert!(
            notes.contains("restart") || notes.contains("MLX reads"),
            "Notes must indicate MLX restart requirement"
        );
    }
}
