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
    gpu_active: f64,
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

        // Populate shared mactop cache for system.rs to read cluster freq / power / load
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

        let metrics = GpuMetrics {
            temp: soc.gpu_temp as f32,
            load: soc.gpu_active as u32,
            power_consumption: soc.gpu_power as f32,
            power_limit: 0, // Not available from mactop
            vram_used: vram_used_mb as u64,
            vram_total: vram_total_mb as u64,
            sclk_mhz: soc.gpu_freq_mhz as u32,
            mclk_mhz: mclk_mhz as u32,
            metal_gpu_limit_mb: Some(read_iogpu_wired_limit_mb()),
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
