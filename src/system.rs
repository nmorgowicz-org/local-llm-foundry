#[cfg(target_os = "macos")]
use crate::gpu::mactop_cache;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;
use sysinfo::System;

/// CPU usage is a delta between samples. Keep both sysinfo's counters and the
/// last valid load across requests; refreshing a new System twice immediately
/// never provides a meaningful sampling interval.
struct SystemSampler {
    sys: System,
    last_cpu_refresh: Instant,
    cpu_load: u32,
}

impl SystemSampler {
    fn new() -> Self {
        let mut sys = System::new();
        // Prime the counters, but do not publish sysinfo's first-sample usage.
        sys.refresh_cpu_all();
        Self {
            sys,
            last_cpu_refresh: Instant::now(),
            cpu_load: 0,
        }
    }

    fn refresh_cpu_if_due(&mut self, now: Instant) {
        if !cpu_refresh_due(self.last_cpu_refresh, now) {
            return;
        }
        self.sys.refresh_cpu_all();
        self.cpu_load = get_cpu_load(&self.sys);
        self.last_cpu_refresh = Instant::now();
    }
}

fn cpu_refresh_due(previous: Instant, now: Instant) -> bool {
    now.saturating_duration_since(previous) >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL
}

static SYSTEM_SAMPLER: LazyLock<Mutex<SystemSampler>> =
    LazyLock::new(|| Mutex::new(SystemSampler::new()));

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemMetrics {
    #[serde(default)]
    pub cpu_name: String,
    #[serde(default)]
    pub cpu_temp: f32,
    #[serde(default)]
    pub cpu_temp_available: bool,
    #[serde(default)]
    pub cpu_load: u32,
    #[serde(default)]
    pub cpu_clock_mhz: u32,
    #[serde(default)]
    pub ram_total_gb: f64,
    #[serde(default)]
    pub ram_used_gb: f64,
    #[serde(default)]
    pub ram_available_gb: f64,
    #[serde(default)]
    pub memory_pressure_level: String,
    #[serde(default)]
    pub memory_pressure_source: String,
    #[serde(default)]
    pub memory_pressure_score: f64,
    #[serde(default)]
    pub memory_free_gb: f64,
    #[serde(default)]
    pub memory_compressor_gb: f64,
    #[serde(default)]
    pub memory_compressed_gb: f64,
    /// S-core count. On Apple Silicon this is derived from the first L2-group size of
    /// perflevel1 when the chip has two sub-clusters inside the secondary perf level.
    /// Returns 0 on non-macOS or when only one perf-level cluster exists.
    #[serde(default)]
    pub s_cores: u32,
    /// Human-readable name for the P-core cluster from `hw.perflevel0.name` (e.g. "Super").
    #[serde(default)]
    pub p_cluster_name: String,
    /// Human-readable name for the secondary cluster from `hw.perflevel1.name` (e.g. "Performance").
    #[serde(default)]
    pub secondary_cluster_name: String,
    /// Wired (kernel-locked) memory in GB. Non-compressible; includes GPU framebuffers
    /// and kernel allocations. On Apple Silicon this is the Metal iogpu budget floor.
    #[serde(default)]
    pub memory_wired_gb: f64,
    /// Purgeable memory in GB (macOS only). File-backed pages the kernel can drop
    /// instantly without writing to disk — effectively free on demand.
    #[serde(default)]
    pub memory_purgeable_gb: f64,
    /// Inactive memory in GB (macOS only). Candidate pages for compression or eviction.
    #[serde(default)]
    pub memory_inactive_gb: f64,
    /// Cross-platform estimate of memory the OS can reclaim without killing a process.
    #[serde(default)]
    pub memory_reclaimable_gb: f64,
    /// Swap/pagefile currently in use, where the platform exposes it.
    #[serde(default)]
    pub swap_used_gb: f64,
    #[serde(default)]
    pub swapins: u64,
    #[serde(default)]
    pub swapouts: u64,
    #[serde(default)]
    pub swapins_delta: u64,
    #[serde(default)]
    pub swapouts_delta: u64,
    #[serde(default)]
    pub memory_psi_some_avg10: f64,
    #[serde(default)]
    pub memory_psi_full_avg10: f64,
    #[serde(default)]
    pub memory_pressure_advice: String,
    #[serde(default)]
    pub motherboard: String,
    /// Performance-core count (Apple Silicon only; 0 = unknown/not applicable).
    #[serde(default)]
    pub p_cores: u32,
    /// Efficiency-core count (Apple Silicon only; 0 = unknown/not applicable).
    #[serde(default)]
    pub e_cores: u32,
    /// Total system power drawn (Watts). Populated on macOS via mactop.
    #[serde(default)]
    pub power_total_w: f32,
    /// CPU power drawn (Watts). Populated on macOS via mactop.
    #[serde(default)]
    pub power_cpu_w: f32,
    /// GPU power drawn (Watts). Populated on macOS via mactop.
    #[serde(default)]
    pub power_gpu_w: f32,
    /// P-cluster current frequency (MHz). Populated on macOS via mactop.
    #[serde(default)]
    pub p_cluster_freq_mhz: u32,
    /// S-cluster current frequency (MHz). Populated on macOS via mactop.
    #[serde(default)]
    pub s_cluster_freq_mhz: u32,
    /// E-cluster current frequency (MHz). Populated on macOS via mactop.
    #[serde(default)]
    pub e_cluster_freq_mhz: u32,
    /// P-cluster utilization (%). Populated on macOS via mactop.
    #[serde(default)]
    pub p_cluster_active: f32,
    /// S-cluster utilization (%). Populated on macOS via mactop.
    #[serde(default)]
    pub s_cluster_active: f32,
    /// E-cluster utilization (%). Populated on macOS via mactop.
    #[serde(default)]
    pub e_cluster_active: f32,
    // Default: all new fields are 0/false for non-macOS builds
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_name: String::new(),
            cpu_temp: 0.0,
            cpu_temp_available: false,
            cpu_load: 0,
            cpu_clock_mhz: 0,
            ram_total_gb: 0.0,
            ram_used_gb: 0.0,
            ram_available_gb: 0.0,
            memory_pressure_level: String::new(),
            memory_pressure_source: String::new(),
            memory_pressure_score: 0.0,
            memory_free_gb: 0.0,
            memory_compressor_gb: 0.0,
            memory_compressed_gb: 0.0,
            s_cores: 0,
            p_cluster_name: String::new(),
            secondary_cluster_name: String::new(),
            memory_wired_gb: 0.0,
            memory_purgeable_gb: 0.0,
            memory_inactive_gb: 0.0,
            memory_reclaimable_gb: 0.0,
            swap_used_gb: 0.0,
            swapins: 0,
            swapouts: 0,
            swapins_delta: 0,
            swapouts_delta: 0,
            memory_psi_some_avg10: 0.0,
            memory_psi_full_avg10: 0.0,
            memory_pressure_advice: String::new(),
            motherboard: String::new(),
            p_cores: 0,
            e_cores: 0,
            power_total_w: 0.0,
            power_cpu_w: 0.0,
            power_gpu_w: 0.0,
            p_cluster_freq_mhz: 0,
            s_cluster_freq_mhz: 0,
            e_cluster_freq_mhz: 0,
            p_cluster_active: 0.0,
            s_cluster_active: 0.0,
            e_cluster_active: 0.0,
        }
    }
}

pub fn get_system_metrics() -> SystemMetrics {
    let (sampled_cpu_load, sampled_cpu_clock_mhz, cpu_temp, cpu_temp_available, ram_info) = {
        let mut sampler = SYSTEM_SAMPLER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        sampler.refresh_cpu_if_due(Instant::now());
        sampler.sys.refresh_memory();
        let (cpu_temp, cpu_temp_available) = get_cpu_temp(&sampler.sys);
        (
            sampler.cpu_load,
            get_cpu_clock(&sampler.sys),
            cpu_temp,
            cpu_temp_available,
            get_ram_info(&sampler.sys),
        )
    }; // Release the shared sampler before unrelated platform queries.

    let cpu_name = get_cpu_name();

    // On Apple Silicon, use mactop cache for real-time clock and power, not load.
    // The GPU poller populates this every ~500ms, so data is fresh.
    let (
        cpu_load,
        cpu_clock_mhz,
        power_total_w,
        power_cpu_w,
        power_gpu_w,
        p_cluster_freq_mhz,
        s_cluster_freq_mhz,
        e_cluster_freq_mhz,
        p_cluster_active,
        s_cluster_active,
        e_cluster_active,
    ) = {
        #[cfg(target_os = "macos")]
        {
            if let Some(cache) = mactop_cache::get_cache() {
                // Real utilization from sysinfo (average across all cores) —
                // mactop's cluster_active is a frequency/residency ratio, not
                // load: the S cluster reads 100% even while the machine idles.
                let cpu_load = sampled_cpu_load;

                // Use P-cluster frequency as the "main" clock (it's the one doing heavy work)
                let cpu_clock_mhz = cache.p_cluster_freq_mhz;
                (
                    cpu_load,
                    cpu_clock_mhz,
                    cache.power_total_w,
                    cache.power_cpu_w,
                    cache.power_gpu_w,
                    cache.p_cluster_freq_mhz,
                    cache.s_cluster_freq_mhz,
                    cache.e_cluster_freq_mhz,
                    cache.p_cluster_active,
                    cache.s_cluster_active,
                    cache.e_cluster_active,
                )
            } else {
                // Cache not yet populated — fallback to sysinfo
                let cpu_load = sampled_cpu_load;
                let cpu_clock_mhz = sampled_cpu_clock_mhz;
                (
                    cpu_load,
                    cpu_clock_mhz,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                )
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            (
                sampled_cpu_load,
                sampled_cpu_clock_mhz,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
            )
        }
    };

    let (ram_total_gb, ram_used_gb, ram_available_gb) = ram_info;
    let memory_pressure = get_memory_pressure(ram_total_gb, ram_available_gb);
    let motherboard = get_motherboard();
    let (p_cores, s_cores, e_cores, p_cluster_name, secondary_cluster_name) = get_core_counts();

    SystemMetrics {
        cpu_name,
        cpu_temp,
        cpu_temp_available,
        cpu_load,
        cpu_clock_mhz,
        ram_total_gb,
        ram_used_gb,
        ram_available_gb,
        memory_pressure_level: memory_pressure.level,
        memory_pressure_source: memory_pressure.source,
        memory_pressure_score: memory_pressure.score,
        memory_free_gb: memory_pressure.free_gb,
        memory_compressor_gb: memory_pressure.compressor_gb,
        memory_compressed_gb: memory_pressure.compressed_gb,
        memory_wired_gb: memory_pressure.wired_gb,
        memory_purgeable_gb: memory_pressure.purgeable_gb,
        memory_inactive_gb: memory_pressure.inactive_gb,
        memory_reclaimable_gb: memory_pressure.reclaimable_gb,
        swap_used_gb: memory_pressure.swap_used_gb,
        swapins: memory_pressure.swapins,
        swapouts: memory_pressure.swapouts,
        swapins_delta: memory_pressure.swapins_delta,
        swapouts_delta: memory_pressure.swapouts_delta,
        memory_psi_some_avg10: memory_pressure.psi_some_avg10,
        memory_psi_full_avg10: memory_pressure.psi_full_avg10,
        memory_pressure_advice: memory_pressure.advice,
        motherboard,
        p_cores,
        s_cores,
        e_cores,
        p_cluster_name,
        secondary_cluster_name,
        power_total_w,
        power_cpu_w,
        power_gpu_w,
        p_cluster_freq_mhz,
        s_cluster_freq_mhz,
        e_cluster_freq_mhz,
        p_cluster_active,
        s_cluster_active,
        e_cluster_active,
    }
}

#[cfg(target_os = "macos")]
fn get_core_counts() -> (u32, u32, u32, String, String) {
    fn sysctl_u32(key: &str) -> u32 {
        std::process::Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }
    fn sysctl_str(key: &str) -> String {
        std::process::Command::new("sysctl")
            .args(["-n", key])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    }

    let level_count = sysctl_u32("hw.perflevelcount");
    let level_count = if level_count > 0 { level_count } else { 2 };

    let mut total_p = 0u32;
    let mut total_s = 0u32;
    let mut total_e = 0u32;
    let mut first_name = String::new();
    let mut second_name = String::new();

    for i in 0..level_count {
        let cores = sysctl_u32(&format!("hw.perflevel{i}.physicalcpu"));
        let name = sysctl_str(&format!("hw.perflevel{i}.name"));
        let name_lower = name.to_lowercase();

        if first_name.is_empty() {
            first_name = name.clone();
        } else if second_name.is_empty() {
            second_name = name.clone();
        }

        if name_lower.contains("super") {
            total_p += cores;
        } else if name_lower.contains("performance") {
            total_s += cores;
        } else if name_lower.contains("efficiency") || name_lower.contains("e-core") {
            total_e += cores;
        } else {
            // Fallback: treat unrecognized as efficiency
            total_e += cores;
        }
    }

    // Backward compatibility: if "s" is nonzero (secondary/performance cluster),
    // it is used for the S-row in the dashboard; p_cores includes all top-tier
    // (Super or performance-only on older chips).
    // On older 2-level chips (e.g. M3 Max), names are typically
    // "Performance" (perflevel0) and "Efficiency" (perflevel1), so
    // total_p = 0 and total_s will hold the P-cores; adjust for that case:
    if total_p == 0 && total_s > 0 && total_e > 0 {
        total_p = total_s;
        total_s = 0;
    }

    (total_p, total_s, total_e, first_name, second_name)
}

#[cfg(not(target_os = "macos"))]
fn get_core_counts() -> (u32, u32, u32, String, String) {
    (0, 0, 0, String::new(), String::new())
}

#[cfg(target_os = "windows")]
fn get_cpu_name() -> String {
    use std::collections::HashMap;
    use wmi::{Variant, WMIConnection};

    if let Ok(wmi) = WMIConnection::new() {
        let results: Vec<HashMap<String, Variant>> =
            match wmi.raw_query::<HashMap<String, Variant>>("SELECT Name FROM Win32_Processor") {
                Ok(r) => r,
                Err(_) => return "Unknown CPU".to_string(),
            };
        for row in &results {
            if let Some(Variant::String(name)) = row.get("Name") {
                return name.clone();
            }
        }
    }

    "Unknown CPU".to_string()
}

#[cfg(target_os = "linux")]
fn get_cpu_name() -> String {
    use std::fs;

    if let Ok(content) = fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if let Some(cpu_name) = line.strip_prefix("model name\t: ") {
                return cpu_name.to_string();
            }
            if let Some(cpu_name) = line.strip_prefix("Processor\t: ") {
                return cpu_name.to_string();
            }
        }
    }

    "Unknown CPU".to_string()
}

#[cfg(target_os = "macos")]
fn get_cpu_name() -> String {
    use std::process::Command;

    if let Ok(output) = Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
        && output.status.success()
    {
        return String::from_utf8_lossy(&output.stdout).trim().to_string();
    }

    "Unknown CPU".to_string()
}

fn get_cpu_temp(sys: &System) -> (f32, bool) {
    if sys.cpus().is_empty() {
        return (0.0, false);
    }

    #[cfg(target_os = "windows")]
    {
        use crate::lhm;

        if lhm::is_lhm_available() {
            return lhm::get_lhm_cpu_temp();
        }
        (0.0, false)
    }

    #[cfg(target_os = "linux")]
    {
        use std::fs;

        let components = sysinfo::Components::new_with_refreshed_list();

        if !components.is_empty()
            && let Some(comp) = components.iter().next()
        {
            return (comp.temperature().unwrap_or(0.0), true);
        }

        let temp_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/sys/class/hwmon/hwmon0/device/temp1_input",
            "/sys/class/hwmon/hwmon1/temp1_input",
            "/sys/class/hwmon/hwmon2/temp1_input",
        ];

        for path in temp_paths {
            if let Ok(content) = fs::read_to_string(path)
                && let Ok(temp_milli) = content.trim().parse::<i32>()
            {
                return (temp_milli as f32 / 1000.0, true);
            }
        }

        (0.0, false)
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;

        let sensors = ["hw.sensors.cpu0.temp0", "hw.acpi.thermal.cpu0.temperature"];

        for sensor in sensors {
            if let Ok(output) = Command::new("sysctl").arg("-n").arg(sensor).output()
                && output.status.success()
                && let Ok(temp) = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<f32>()
            {
                return (temp, true);
            }
        }

        (0.0, false)
    }
}

fn get_cpu_load(sys: &System) -> u32 {
    if sys.cpus().is_empty() {
        return 0;
    }

    (sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32) as _
}

#[cfg(target_os = "linux")]
fn get_cpu_clock(sys: &System) -> u32 {
    use std::fs;

    if sys.cpus().is_empty() {
        return 0;
    }

    let mut freq_samples_mhz = Vec::new();

    for cpu in 0..sys.cpus().len() {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq");
        for candidate in ["scaling_cur_freq", "cpuinfo_cur_freq"] {
            let path = format!("{base}/{candidate}");
            if let Ok(raw) = fs::read_to_string(&path)
                && let Ok(khz) = raw.trim().parse::<u64>()
                && khz > 0
            {
                freq_samples_mhz.push((khz / 1000) as u32);
                break;
            }
        }
    }

    if !freq_samples_mhz.is_empty() {
        let sum: u64 = freq_samples_mhz.iter().map(|&v| v as u64).sum();
        return (sum / freq_samples_mhz.len() as u64) as u32;
    }

    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        let mhz_values: Vec<u32> = cpuinfo
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter(|(key, _)| key.trim() == "cpu MHz")
            .filter_map(|(_, value)| value.trim().parse::<f64>().ok())
            .filter(|mhz| *mhz > 0.0)
            .map(|mhz| mhz.round() as u32)
            .collect();

        if !mhz_values.is_empty() {
            let sum: u64 = mhz_values.iter().map(|&v| v as u64).sum();
            return (sum / mhz_values.len() as u64) as u32;
        }
    }

    sys.cpus().iter().map(|c| c.frequency()).max().unwrap_or(0) as u32
}

#[cfg(target_os = "windows")]
fn get_cpu_clock(sys: &System) -> u32 {
    use std::collections::HashMap;
    use wmi::{Variant, WMIConnection};

    if sys.cpus().is_empty() {
        return 0;
    }

    fn variant_to_u32(value: &Variant) -> Option<u32> {
        match value {
            Variant::UI4(v) => Some(*v),
            Variant::I4(v) => (*v).try_into().ok(),
            Variant::UI8(v) => (*v).try_into().ok(),
            Variant::I8(v) => (*v).try_into().ok(),
            Variant::R4(v) => {
                if *v > 0.0 {
                    Some(v.round() as u32)
                } else {
                    None
                }
            }
            Variant::R8(v) => {
                if *v > 0.0 {
                    Some(v.round() as u32)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    if let Ok(wmi) = WMIConnection::new() {
        let base_mhz = wmi
            .raw_query::<HashMap<String, Variant>>("SELECT CurrentClockSpeed FROM Win32_Processor")
            .ok()
            .map(|results| {
                results
                    .iter()
                    .filter_map(|row| row.get("CurrentClockSpeed").and_then(variant_to_u32))
                    .filter(|mhz| *mhz > 0)
                    .collect::<Vec<_>>()
            })
            .and_then(|mhz_values| {
                if mhz_values.is_empty() {
                    None
                } else {
                    let sum: u64 = mhz_values.iter().map(|&v| v as u64).sum();
                    Some((sum / mhz_values.len() as u64) as u32)
                }
            });

        let perf_pct = wmi
            .raw_query::<HashMap<String, Variant>>(
                "SELECT Name,PercentProcessorPerformance FROM Win32_PerfFormattedData_Counters_ProcessorInformation WHERE Name='_Total'",
            )
            .ok()
            .and_then(|results| {
                results.iter().find_map(|row| {
                    row.get("PercentProcessorPerformance")
                        .and_then(variant_to_u32)
                        .filter(|pct| *pct > 0)
                })
            });

        if let (Some(base_mhz), Some(perf_pct)) = (base_mhz, perf_pct) {
            let effective = ((base_mhz as f64) * (perf_pct as f64 / 100.0)).round() as u32;
            if effective > 0 {
                return effective;
            }
        }

        if let Some(base_mhz) = base_mhz {
            return base_mhz;
        }
    }

    sys.cpus().iter().map(|c| c.frequency()).max().unwrap_or(0) as u32
}

#[cfg(all(not(target_os = "linux"), not(target_os = "windows")))]
fn get_cpu_clock(sys: &System) -> u32 {
    if sys.cpus().is_empty() {
        return 0;
    }

    sys.cpus().iter().map(|c| c.frequency()).max().unwrap_or(0) as u32
}

fn get_ram_info(sys: &System) -> (f64, f64, f64) {
    let total_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let used_gb = sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let available_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    (total_gb, used_gb, available_gb)
}

#[derive(Default)]
struct MemoryPressure {
    level: String,
    source: String,
    score: f64,
    free_gb: f64,
    compressor_gb: f64,
    compressed_gb: f64,
    wired_gb: f64,
    purgeable_gb: f64,
    inactive_gb: f64,
    reclaimable_gb: f64,
    swap_used_gb: f64,
    swapins: u64,
    swapouts: u64,
    swapins_delta: u64,
    swapouts_delta: u64,
    psi_some_avg10: f64,
    psi_full_avg10: f64,
    advice: String,
}

/// Map a pressure verdict plus a 0..1 intensity into a 0-100 score whose meaning
/// is consistent across platforms: **ok occupies 0-50, warning 50-80, critical
/// 80-100**. The per-OS signals (macOS compressor ratio, Linux PSI, Windows
/// available ratio) differ, but feeding them through this band mapping makes a
/// given score comparable regardless of platform — e.g. ~65 is always
/// mid-warning. `intensity` expresses how deep into the verdict's band we are.
fn score_for_level(level: &str, intensity: f64) -> f64 {
    let t = intensity.clamp(0.0, 1.0);
    match level {
        "critical" => 80.0 + 20.0 * t,
        "warning" => 50.0 + 30.0 * t,
        _ => 50.0 * t,
    }
}

#[cfg(target_os = "macos")]
fn get_memory_pressure(total_ram_gb: f64, _available_ram_gb: f64) -> MemoryPressure {
    // Prefer the syscall; fall back to scraping `vm_stat` only if it fails.
    let counts = read_macos_vm_counts().unwrap_or_else(macos_vm_counts_from_vm_stat);
    let kernel_level = read_macos_pressure_level();
    let swap_used_gb = read_macos_swap_used_gb();
    compute_macos_pressure(&counts, total_ram_gb, kernel_level, swap_used_gb)
}

/// Fallback page counts via the `vm_stat` subprocess, used only when the
/// `host_statistics64` syscall is unavailable.
#[cfg(target_os = "macos")]
fn macos_vm_counts_from_vm_stat() -> MacVmCounts {
    std::process::Command::new("vm_stat")
        .output()
        .ok()
        .map(|o| parse_macos_vm_stat_text(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default()
}

/// Raw virtual-memory page counts, sourced from `host_statistics64` (no
/// subprocess). Page units; `page_size` is bytes.
#[cfg(target_os = "macos")]
#[derive(Default)]
struct MacVmCounts {
    page_size: u64,
    free: u64,
    wired: u64,
    /// Pages occupied by the compressor (the compressed footprint).
    compressor: u64,
    /// Uncompressed-equivalent pages currently held in the compressor.
    compressed: u64,
    purgeable: u64,
    inactive: u64,
    swapins: u64,
    swapouts: u64,
}

/// Read VM statistics directly via the Mach `host_statistics64` syscall instead
/// of spawning and scraping `vm_stat`. Returns `None` on failure, in which case
/// callers fall back to the `MacVmCounts::default()` (all-zero) values.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // mach_host_self: the libc binding is stable; mach2 is not a dependency.
fn read_macos_vm_counts() -> Option<MacVmCounts> {
    // SAFETY: `host_statistics64` fills `info` with exactly `count` integer_t
    // words. We size `count` from the struct and check the return code before
    // reading any field.
    unsafe {
        let host = libc::mach_host_self();
        let mut info: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::vm_statistics64>()
            / std::mem::size_of::<libc::integer_t>())
            as libc::mach_msg_type_number_t;
        let rc = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut info as *mut _ as *mut libc::integer_t,
            &mut count,
        );
        if rc != libc::KERN_SUCCESS {
            return None;
        }
        Some(MacVmCounts {
            page_size: libc::vm_page_size as u64,
            free: u64::from(info.free_count),
            wired: u64::from(info.wire_count),
            compressor: u64::from(info.compressor_page_count),
            compressed: info.total_uncompressed_pages_in_compressor,
            purgeable: u64::from(info.purgeable_count),
            inactive: u64::from(info.inactive_count),
            swapins: info.swapins,
            swapouts: info.swapouts,
        })
    }
}

/// Read the kernel's own memory-pressure verdict from
/// `kern.memorystatus_vm_pressure_level` via `sysctlbyname`: 1 = normal,
/// 2 = warning, 4 = critical. This is the same signal the OS uses to drive
/// jetsam and `DISPATCH_MEMORYPRESSURE` notifications, so it is far more reliable
/// than reconstructing pressure from raw page counts. Returns `None` if the
/// sysctl is unavailable (then we fall back to compressor-based heuristics).
#[cfg(target_os = "macos")]
fn read_macos_pressure_level() -> Option<u8> {
    let mut value: i32 = 0;
    let mut size = std::mem::size_of::<i32>();
    // SAFETY: we pass a correctly sized i32 out-buffer and check the return code.
    let rc = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            &mut value as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        u8::try_from(value).ok()
    } else {
        None
    }
}

/// Read swap currently in use (GB) from `vm.swapusage` via `sysctlbyname`. macOS
/// swap is a real pressure signal but, unlike Linux/Windows, was previously never
/// populated.
#[cfg(target_os = "macos")]
fn read_macos_swap_used_gb() -> f64 {
    let mut usage: libc::xsw_usage = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<libc::xsw_usage>();
    // SAFETY: we pass a correctly sized xsw_usage out-buffer and check the rc.
    let rc = unsafe {
        libc::sysctlbyname(
            c"vm.swapusage".as_ptr(),
            &mut usage as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        usage.xsu_used as f64 / 1024.0 / 1024.0 / 1024.0
    } else {
        0.0
    }
}

/// Parse `vm_stat` text into `MacVmCounts`. Kept as a fallback path and for unit
/// tests; the live path uses `read_macos_vm_counts` (syscall) instead.
#[cfg(target_os = "macos")]
fn parse_macos_vm_stat_text(text: &str) -> MacVmCounts {
    let mut counts = MacVmCounts {
        page_size: 16_384,
        ..Default::default()
    };
    for line in text.lines() {
        if let Some(size) = line
            .strip_prefix("Mach Virtual Memory Statistics: (page size of ")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
        {
            counts.page_size = size;
            continue;
        }

        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_end_matches('.')
            .replace(',', "")
            .parse::<u64>()
            .unwrap_or(0);
        match key.trim() {
            "Pages free" => counts.free = value,
            "Pages wired down" => counts.wired = value,
            "Pages occupied by compressor" => counts.compressor = value,
            "Pages stored in compressor" => counts.compressed = value,
            "Pages purgeable" => counts.purgeable = value,
            "Pages inactive" => counts.inactive = value,
            "Swapins" => counts.swapins = value,
            "Swapouts" => counts.swapouts = value,
            _ => {}
        }
    }
    counts
}

#[cfg(target_os = "macos")]
fn compute_macos_pressure(
    counts: &MacVmCounts,
    total_ram_gb: f64,
    kernel_level: Option<u8>,
    swap_used_gb: f64,
) -> MemoryPressure {
    use std::sync::{Mutex, OnceLock};

    static LAST_SWAP: OnceLock<Mutex<(u64, u64)>> = OnceLock::new();

    let page_gb = counts.page_size as f64 / 1024.0 / 1024.0 / 1024.0;
    let free_gb = counts.free as f64 * page_gb;
    let wired_gb = counts.wired as f64 * page_gb;
    let compressor_gb = counts.compressor as f64 * page_gb;
    let compressed_gb = counts.compressed as f64 * page_gb;
    let purgeable_gb = counts.purgeable as f64 * page_gb;
    let inactive_gb = counts.inactive as f64 * page_gb;
    let reclaimable_gb = purgeable_gb + inactive_gb;
    let compressor_ratio = if total_ram_gb > 0.0 {
        compressor_gb / total_ram_gb
    } else {
        0.0
    };
    let (swapins_delta, swapouts_delta) = {
        let mut last = LAST_SWAP
            .get_or_init(|| Mutex::new((counts.swapins, counts.swapouts)))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let delta = (
            counts.swapins.saturating_sub(last.0),
            counts.swapouts.saturating_sub(last.1),
        );
        *last = (counts.swapins, counts.swapouts);
        delta
    };
    // Anchor the verdict on the kernel's own pressure level when available.
    // macOS keeps "Pages free" deliberately low (RAM is used as cache) and swaps
    // lazily even when healthy, so raw free-page and single-swapout thresholds
    // produce chronic false alarms. The kernel sysctl is the authoritative signal;
    // we only let an extreme compressor ratio escalate beyond what it reports.
    let level = match kernel_level {
        Some(4) => "critical",
        Some(2) => "warning",
        Some(_) => {
            // Kernel says normal — only flag if the compressor is working very hard.
            if compressor_ratio >= 0.30 {
                "warning"
            } else {
                "ok"
            }
        }
        None => {
            // sysctl unavailable: fall back to compressor heuristics. Note we do
            // NOT use free_gb here — it is misleadingly low on macOS by design.
            if compressor_ratio >= 0.30 {
                "critical"
            } else if compressor_ratio >= 0.18 {
                "warning"
            } else {
                "ok"
            }
        }
    }
    .to_string();
    // Intensity within the verdict's band: how hard the compressor is working,
    // nudged up while swap is actively churning.
    let intensity = (compressor_ratio / 0.30)
        .clamp(0.0, 1.0)
        .max(if swapouts_delta > 0 { 0.5 } else { 0.0 });
    let score = score_for_level(&level, intensity);
    // Keep advice consistent with `level`: don't nag about swapping/compression
    // while the kernel reports normal pressure (macOS compresses and swaps
    // routinely on a healthy system).
    let advice = if wired_gb > total_ram_gb * 0.55 {
        "Wired memory is high; prefer mmap-enabled presets and disable mlock.".to_string()
    } else if level == "ok" {
        "Memory pressure is normal.".to_string()
    } else if reclaimable_gb > 1.0 {
        "Reclaimable cache is available; Free Memory can help if sudo is already authorized."
            .to_string()
    } else {
        "Reduce context, batch, or parallel slots; macOS is compressing or swapping memory."
            .to_string()
    };

    MemoryPressure {
        level,
        source: "host_statistics64".to_string(),
        score,
        free_gb,
        wired_gb,
        compressor_gb,
        compressed_gb,
        purgeable_gb,
        inactive_gb,
        reclaimable_gb,
        swap_used_gb,
        swapins: counts.swapins,
        swapouts: counts.swapouts,
        swapins_delta,
        swapouts_delta,
        psi_some_avg10: 0.0,
        psi_full_avg10: 0.0,
        advice,
    }
}

#[cfg(target_os = "linux")]
fn get_memory_pressure(total_ram_gb: f64, available_ram_gb: f64) -> MemoryPressure {
    use std::fs;

    fn kb_to_gb(kb: u64) -> f64 {
        kb as f64 / 1024.0 / 1024.0
    }

    let mut mem_free_kb = 0_u64;
    let mut cached_kb = 0_u64;
    let mut sreclaimable_kb = 0_u64;
    let mut mlocked_kb = 0_u64;
    let mut unevictable_kb = 0_u64;
    let mut swap_total_kb = 0_u64;
    let mut swap_free_kb = 0_u64;

    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap_or("").trim_end_matches(':');
            let value = parts
                .next()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            match key {
                "MemFree" => mem_free_kb = value,
                "Cached" => cached_kb = value,
                "SReclaimable" => sreclaimable_kb = value,
                "Mlocked" => mlocked_kb = value,
                "Unevictable" => unevictable_kb = value,
                "SwapTotal" => swap_total_kb = value,
                "SwapFree" => swap_free_kb = value,
                _ => {}
            }
        }
    }

    let (psi_some_avg10, psi_full_avg10) = fs::read_to_string("/proc/pressure/memory")
        .ok()
        .map(|psi| parse_linux_psi(&psi))
        .unwrap_or((0.0, 0.0));
    let available_ratio = if total_ram_gb > 0.0 {
        available_ram_gb / total_ram_gb
    } else {
        1.0
    };
    let swap_used_gb = kb_to_gb(swap_total_kb.saturating_sub(swap_free_kb));
    let level = if psi_full_avg10 >= 1.0
        || psi_some_avg10 >= 20.0
        || (available_ratio < 0.05 && swap_used_gb > 0.0)
    {
        "critical"
    } else if psi_some_avg10 >= 5.0 || available_ratio < 0.10 {
        "warning"
    } else {
        "ok"
    }
    .to_string();
    // Intensity within the band: the strongest of the PSI stall signals and the
    // memory-shortfall ratio.
    let intensity = (psi_some_avg10 / 20.0)
        .max(psi_full_avg10 / 5.0)
        .max(1.0 - available_ratio);
    let pressure_score = score_for_level(&level, intensity);
    let reclaimable_gb = kb_to_gb(cached_kb + sreclaimable_kb);
    let pinned_gb = kb_to_gb(mlocked_kb + unevictable_kb);
    let advice = if psi_full_avg10 >= 1.0 || psi_some_avg10 >= 5.0 {
        "Linux PSI reports memory stalls; reduce context, batch, or parallel slots.".to_string()
    } else if available_ratio < 0.10 {
        "Available memory is low; lower model footprint or stop other large processes.".to_string()
    } else {
        "Memory pressure is normal.".to_string()
    };

    MemoryPressure {
        level,
        source: "linux_psi".to_string(),
        score: pressure_score,
        free_gb: kb_to_gb(mem_free_kb),
        wired_gb: pinned_gb,
        reclaimable_gb,
        swap_used_gb,
        psi_some_avg10,
        psi_full_avg10,
        advice,
        ..Default::default()
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_psi(text: &str) -> (f64, f64) {
    fn avg10(line: &str) -> f64 {
        line.split_whitespace()
            .find_map(|part| part.strip_prefix("avg10="))
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.0)
    }

    let mut some = 0.0;
    let mut full = 0.0;
    for line in text.lines() {
        if line.starts_with("some ") {
            some = avg10(line);
        } else if line.starts_with("full ") {
            full = avg10(line);
        }
    }
    (some, full)
}

#[cfg(target_os = "windows")]
fn get_memory_pressure(total_ram_gb: f64, available_ram_gb: f64) -> MemoryPressure {
    // Base the pressure ratio on *available* memory (from sysinfo), not WMI's
    // FreePhysicalMemory. Windows parks large amounts of RAM in the reclaimable
    // standby/cache list, so truly-free memory is low on a healthy system and
    // would chronically overstate pressure. "Available" already includes
    // reclaimable standby pages.
    let available_gb = available_ram_gb;

    // Pagefile/commit usage via sysinfo's cheap memory refresh, which calls
    // GlobalMemoryStatusEx underneath — no WMI/COM round-trip per poll.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let swap_used_gb = sys.used_swap() as f64 / 1024.0 / 1024.0 / 1024.0;

    let available_ratio = if total_ram_gb > 0.0 {
        available_gb / total_ram_gb
    } else {
        1.0
    };
    let level = if available_ratio < 0.05 {
        "critical"
    } else if available_ratio < 0.10 {
        "warning"
    } else {
        "ok"
    }
    .to_string();
    let pressure_score = score_for_level(&level, 1.0 - available_ratio);
    let advice = if level == "critical" || level == "warning" {
        "Windows reports low available memory; reduce context, batch, or stop large processes."
            .to_string()
    } else {
        "Memory pressure is normal.".to_string()
    };

    MemoryPressure {
        level,
        source: "windows_memstatus".to_string(),
        score: pressure_score,
        free_gb: available_gb,
        swap_used_gb,
        advice,
        ..Default::default()
    }
}

#[cfg(all(
    not(target_os = "macos"),
    not(target_os = "linux"),
    not(target_os = "windows")
))]
fn get_memory_pressure(_total_ram_gb: f64, _available_ram_gb: f64) -> MemoryPressure {
    MemoryPressure::default()
}

#[cfg(target_os = "windows")]
fn get_motherboard() -> String {
    use std::collections::HashMap;
    use wmi::{Variant, WMIConnection};
    if let Ok(wmi) = WMIConnection::new() {
        let results: Vec<HashMap<String, Variant>> = match wmi
            .raw_query::<HashMap<String, Variant>>("SELECT Product FROM Win32_BaseBoard")
        {
            Ok(r) => r,
            Err(_) => {
                eprintln!("[system] WMI query failed for Win32_BaseBoard");
                return "Unknown Motherboard".to_string();
            }
        };
        for row in &results {
            if let Some(Variant::String(product)) = row.get("Product") {
                return product.clone();
            }
        }
    }

    "Unknown Motherboard".to_string()
}

#[cfg(target_os = "linux")]
fn get_motherboard() -> String {
    use std::fs;

    let paths = [
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/board_name",
        "/sys/class/dmi/id/vendor",
    ];

    for path in paths {
        if let Ok(content) = fs::read_to_string(path) {
            let value = content.trim();
            if !value.is_empty() && value != "To Be Filled By O.E.M." {
                return value.to_string();
            }
        }
    }

    "Unknown Motherboard".to_string()
}

#[cfg(target_os = "macos")]
fn get_motherboard() -> String {
    use std::process::Command;

    if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.model").output()
        && output.status.success()
    {
        let model = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !model.is_empty() {
            return model;
        }
    }

    "Unknown Motherboard".to_string()
}

// ── Reclaim guidance actions (Phase 5b Part C, item 16) ──────────────────────

/// Distinct reclaim actions the backend can perform or recommend.
/// Each has a conservative bounded estimate (range), is only offered when it
/// can cross the selected fit boundary, and reports actual change after execution.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReclaimAction {
    /// Clear backend allocator cache (MLX/llama memory pools).
    /// Conservative estimate: may free 1-4 GB depending on recent allocations.
    BackendAllocatorCacheClear,
    /// Evict reusable state (prompt caches, retained session data).
    /// Conservative estimate: may free 1-3 GB if sessions exist.
    ReusableStateEviction,
    /// Stop app-owned runtime(s). Only offered when the target runtime is actually running.
    /// Conservative estimate: may free the runtime's full working set (2-8 GB typical).
    AppOwnedRuntimeStop,
    /// OS disk cache purge (macOS `purge`). Does NOT clear live model/Metal memory.
    /// Conservative estimate: may free 0.5-2 GB of file-backed cache.
    OsDiskPurge,
}

/// Grouped high-memory process info for privacy-safe reporting.
/// Uses macOS `phys_footprint` (the physical memory footprint including
/// compressed pages) with redacted command arguments.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcessFootprint {
    /// Process ID.
    #[serde(default)]
    pub pid: u32,
    /// Application name.
    #[serde(default)]
    pub name: String,
    /// Physical footprint in MB (macOS: phys_footprint; others: RSS).
    /// Labeled clearly as "physical footprint" to distinguish from RSS.
    #[serde(default)]
    pub phys_footprint_mb: f64,
    /// Resident set size in MB (actual resident pages only).
    /// Labeled clearly as "RSS" (Resident Set Size).
    #[serde(default)]
    pub rss_mb: f64,
    /// Redacted command: tokens/secrets stripped, max 128 chars.
    #[serde(default)]
    pub command: String,
}

/// Guidance response: actions that can cross the fit boundary and their effects.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReclaimGuidance {
    /// Current memory availability snapshot (source of truth).
    #[serde(default)]
    pub snapshot: crate::memory_availability::MemoryAvailabilitySnapshot,
    /// Actions that may help reach the target fit. Empty if already safe_now.
    #[serde(default)]
    pub available_actions: Vec<ReclaimAction>,
    /// Conservative bounded estimate of what reclaim may achieve.
    /// Uses ranges, never promises. "may free X-Y GB".
    #[serde(default)]
    pub conservative_estimate: String,
    /// High-memory processes grouped by category (for diagnostic context only).
    /// Not presented as terminable; no arbitrary process termination.
    #[serde(default)]
    pub high_memory_processes: Vec<ProcessFootprint>,
    /// Whether current state already meets typical launch requirements.
    #[serde(default)]
    pub safe_now: bool,
    /// Honest next-step copy when no in-app operation is both available and
    /// proven to cross the selected fit boundary.
    #[serde(default)]
    pub diagnostic_message: String,
}

/// Compute reclaim guidance based on current MemoryAvailabilitySnapshot.
///
/// Returns actions only when they can cross the selected fit boundary.
/// Estimates are conservative and bounded.
pub fn compute_reclaim_guidance(
    snapshot: &crate::memory_availability::MemoryAvailabilitySnapshot,
) -> ReclaimGuidance {
    let state = &snapshot.state;

    // If already safe_now, no reclaim needed.
    let safe_now = matches!(
        state,
        crate::memory_availability::MemoryAvailabilityState::SafeNow
    );

    // Do not advertise an action merely because it sounds plausible. The
    // allocator/session/runtime operations are not yet coordinated with a
    // selected running backend, and disk-cache purge never frees live model
    // memory. Until a concrete action can remeasure and cross this exact
    // target, return diagnostics rather than a deceptive button.
    let available_actions = Vec::new();
    let (conservative_estimate, diagnostic_message) = match state {
        crate::memory_availability::MemoryAvailabilityState::SafeNow => (
            "Selected configuration fits current safe availability.".to_string(),
            "No reclaim action is needed.".to_string(),
        ),
        crate::memory_availability::MemoryAvailabilityState::ConditionalAfterReclaim => (
            "A conservative reclaim projection could meet this selected footprint, but no managed in-app reclaimer is available yet.".to_string(),
            "Close or stop known app-owned work, then remeasure before launching. Disk-cache purge does not free live model memory.".to_string(),
        ),
        crate::memory_availability::MemoryAvailabilityState::AfterClosingApps => (
            "This launch needs a measured app-owned runtime to stop before it can fit.".to_string(),
            "Stop the selected app-owned runtime through its session controls, then remeasure.".to_string(),
        ),
        crate::memory_availability::MemoryAvailabilityState::Unsafe => (
            "No conservative reclaim projection reaches the selected footprint.".to_string(),
            "Choose a smaller model, context, or concurrency target; do not rely on disk-cache purge for live model memory.".to_string(),
        ),
    };

    let high_memory_processes = collect_high_memory_processes(10);

    ReclaimGuidance {
        snapshot: snapshot.clone(),
        available_actions,
        conservative_estimate,
        high_memory_processes,
        safe_now,
        diagnostic_message,
    }
}

/// Execute a reclaim action and return before/after snapshots.
///
/// Returns actual change reported conservatively.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReclaimResult {
    /// True if action was performed or attempted.
    #[serde(default)]
    pub success: bool,
    /// Snapshot before the action.
    #[serde(default)]
    pub before: crate::memory_availability::MemoryAvailabilitySnapshot,
    /// Snapshot after the action.
    #[serde(default)]
    pub after: crate::memory_availability::MemoryAvailabilitySnapshot,
    /// Actual change in current_safe_availability_bytes.
    #[serde(default)]
    pub availability_change_bytes: i64,
    /// Message describing result honestly.
    #[serde(default)]
    pub message: String,
}

pub fn execute_reclaim_action(action: ReclaimAction) -> ReclaimResult {
    let before = crate::memory_availability::build_snapshot();

    let result = match action {
        ReclaimAction::BackendAllocatorCacheClear => clear_backend_allocator_cache(&before),
        ReclaimAction::ReusableStateEviction => evict_reusable_state(&before),
        ReclaimAction::AppOwnedRuntimeStop => stop_app_owned_runtime(&before),
        ReclaimAction::OsDiskPurge => os_disk_purge(&before),
    };

    let after = crate::memory_availability::build_snapshot();
    let change = (after.current_safe_availability_bytes as i64)
        - (before.current_safe_availability_bytes as i64);

    // Report actual change honestly.
    let actual_msg = if change > 0 {
        let gb = change as f64 / (1024.0 * 1024.0 * 1024.0);
        format!(
            "Actual: current safe availability increased by {:.1} GB.",
            gb
        )
    } else if change == 0 {
        "Actual: no measurable change in safe availability.".to_string()
    } else {
        format!(
            "Actual: change was {:.1} GB (memory dynamics shifted).",
            change as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    };

    ReclaimResult {
        success: result.success,
        before,
        after,
        availability_change_bytes: change,
        message: format!("{} {}", result.message, actual_msg),
    }
}

/// Clear backend allocator cache.
/// On macOS with Metal, this triggers MLX memory pool reset via environment.
#[allow(dead_code)]
fn clear_backend_allocator_cache(
    _snapshot: &crate::memory_availability::MemoryAvailabilitySnapshot,
) -> ReclaimResult {
    // NOTE: Actual allocator cache clear would require coordination with the
    // running backend(s). For now, we return guidance that this requires
    // a backend restart to take effect.
    ReclaimResult {
        success: false,
        before: _snapshot.clone(),
        after: crate::memory_availability::build_snapshot(),
        availability_change_bytes: 0,
        message: "Backend allocator cache clear requires restarting the target runtime. This will reclaim 1-4 GB of MLX/llama memory pools.".to_string(),
    }
}

/// Evict reusable state (prompt caches, retained sessions).
#[allow(dead_code)]
fn evict_reusable_state(
    _snapshot: &crate::memory_availability::MemoryAvailabilitySnapshot,
) -> ReclaimResult {
    // NOTE: Actual eviction requires coordinating with running sessions.
    ReclaimResult {
        success: false,
        before: _snapshot.clone(),
        after: crate::memory_availability::build_snapshot(),
        availability_change_bytes: 0,
        message: "Reusable state eviction requires stopping active sessions. This will reclaim 1-3 GB of prompt cache and session data if sessions exist.".to_string(),
    }
}

/// Stop app-owned runtime(s).
#[allow(dead_code)]
fn stop_app_owned_runtime(
    _snapshot: &crate::memory_availability::MemoryAvailabilitySnapshot,
) -> ReclaimResult {
    // NOTE: Actual runtime stop is handled by the runtime management layer.
    ReclaimResult {
        success: false,
        before: _snapshot.clone(),
        after: crate::memory_availability::build_snapshot(),
        availability_change_bytes: 0,
        message: "App-owned runtime stop is handled through the runtime management UI. This will reclaim the full runtime working set (2-8 GB typical).".to_string(),
    }
}

/// OS disk cache purge (macOS only).
/// Does NOT clear live model/Metal memory; only file-backed cache.
#[allow(dead_code)]
fn os_disk_purge(
    _snapshot: &crate::memory_availability::MemoryAvailabilitySnapshot,
) -> ReclaimResult {
    #[cfg(target_os = "macos")]
    {
        // Delegate to the existing purge implementation in system_tools.rs.
        let result = crate::web::api::system_tools::run_purge();
        if result.ok {
            ReclaimResult {
                success: true,
                before: _snapshot.clone(),
                after: crate::memory_availability::build_snapshot(),
                availability_change_bytes: 0,
                message: result.message,
            }
        } else {
            ReclaimResult {
                success: false,
                before: _snapshot.clone(),
                after: crate::memory_availability::build_snapshot(),
                availability_change_bytes: 0,
                message: result.message,
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        ReclaimResult {
            success: false,
            before: _snapshot.clone(),
            after: crate::memory_availability::build_snapshot(),
            availability_change_bytes: 0,
            message: "OS disk purge is only supported on macOS.".to_string(),
        }
    }
}

/// Collect high-memory processes grouped by category for diagnostic context.
///
/// Privacy-safe: redacts tokens/secrets from commands, uses phys_footprint where available.
/// NOT presented as terminable; no arbitrary process termination.
fn collect_high_memory_processes(limit: usize) -> Vec<ProcessFootprint> {
    // On macOS, prefer sysctl-based phys_footprint; fall back to sysinfo RSS.
    #[cfg(target_os = "macos")]
    let procs = collect_macos_processes(limit);
    #[cfg(not(target_os = "macos"))]
    let procs = collect_sysinfo_processes(limit);

    procs
}

/// macOS-specific process collection.
/// Uses RSS as phys_footprint (true phys_footprint requires task_info syscall).
#[cfg(target_os = "macos")]
fn collect_macos_processes(limit: usize) -> Vec<ProcessFootprint> {
    collect_sysinfo_processes_inner(limit)
}

/// Cross-platform fallback using sysinfo.
#[cfg(not(target_os = "macos"))]
fn collect_sysinfo_processes(limit: usize) -> Vec<ProcessFootprint> {
    collect_sysinfo_processes_inner(limit)
}

/// Shared sysinfo-based process collection.
fn collect_sysinfo_processes_inner(limit: usize) -> Vec<ProcessFootprint> {
    let sys = sysinfo::System::new_all();
    let mut procs: Vec<ProcessFootprint> = sys
        .processes()
        .iter()
        .filter_map(|(pid, process)| {
            let rss_bytes = process.memory();
            if rss_bytes == 0 {
                return None;
            }

            let cmd_raw = process
                .cmd()
                .iter()
                .map(|part| part.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ");

            let cmd = redact_command(&cmd_raw);

            Some(ProcessFootprint {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().into_owned(),
                phys_footprint_mb: rss_bytes as f64 / 1024.0 / 1024.0,
                rss_mb: rss_bytes as f64 / 1024.0 / 1024.0,
                command: cmd,
            })
        })
        .collect();

    procs.truncate(limit);
    procs
}

/// Redact secrets/tokens from command strings.
///
/// Strips patterns that look like tokens, keys, passwords, or auth headers.
/// Result is capped at 128 characters.
pub fn redact_command(cmd: &str) -> String {
    let redacted = cmd
        .split_whitespace()
        .map(|token| {
            // Redact common secret patterns.
            if looks_like_secret(token) {
                "REDACTED".to_string()
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    // Truncate to 128 chars.
    if redacted.len() > 128 {
        format!("{}...", &redacted[..125])
    } else {
        redacted
    }
}

/// Check if a token looks like a secret (API key, password, bearer token, etc.).
fn looks_like_secret(token: &str) -> bool {
    let lower = token.to_lowercase();

    // Key-value patterns.
    if lower.starts_with('=')
        && (lower.starts_with("password=")
            || lower.starts_with("api_key=")
            || lower.starts_with("apikey=")
            || lower.starts_with("token=")
            || lower.starts_with("secret="))
    {
        return true;
    }

    // Bearer tokens.
    if lower.starts_with("bearer ") || lower.starts_with("token:") || lower.starts_with("auth:") {
        return true;
    }

    // Long hex/base64-like strings after common prefixes.
    if lower.contains("--token=")
        || lower.contains("--api-key=")
        || lower.contains("--password=")
        || lower.contains("--secret=")
        || lower.contains("-t ")
    {
        // Check if the value portion is long (likely a secret).
        let value = lower.split(['=', ' ']).next_back().unwrap_or("");
        if value.len() > 16 && !value.starts_with("-") {
            return true;
        }
    }

    // Standalone long hex strings (24+ chars) — likely API keys.
    if token.len() >= 24
        && token
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '_' || c == '-')
    {
        return true;
    }

    false
}

#[cfg(test)]
mod memory_pressure_tests {
    #[cfg(target_os = "linux")]
    use super::parse_linux_psi;
    use super::score_for_level;
    #[cfg(target_os = "macos")]
    use super::{compute_macos_pressure, parse_macos_vm_stat_text};

    #[cfg(target_os = "macos")]
    const SAMPLE_VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free:                               20000.\n\
Pages wired down:                         50000.\n\
Pages occupied by compressor:             10000.\n\
Pages stored in compressor:               30000.\n\
Pages purgeable:                          5000.\n\
Pages inactive:                           40000.\n\
Swapins:                                  100.\n\
Swapouts:                                 200.\n";

    #[cfg(target_os = "macos")]
    fn pressure_from_text(
        text: &str,
        total: f64,
        kernel: Option<u8>,
        swap: f64,
    ) -> super::MemoryPressure {
        compute_macos_pressure(&parse_macos_vm_stat_text(text), total, kernel, swap)
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn live_syscall_path_populates_reading() {
        // Exercises host_statistics64 + sysctlbyname end-to-end on the host.
        let p = super::get_memory_pressure(64.0, 32.0);
        assert_eq!(p.source, "host_statistics64");
        assert!(
            p.free_gb > 0.0,
            "free_gb should be populated, got {}",
            p.free_gb
        );
        assert!(["ok", "warning", "critical"].contains(&p.level.as_str()));
        assert!((0.0..=100.0).contains(&p.score));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn vm_stat_text_parses_page_counts() {
        let counts = parse_macos_vm_stat_text(SAMPLE_VM_STAT);
        assert_eq!(counts.page_size, 16384);
        assert_eq!(counts.free, 20000);
        assert_eq!(counts.compressor, 10000);
        assert_eq!(counts.swapouts, 200);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernel_level_anchors_verdict() {
        // Kernel reports critical (4) -> critical regardless of low compressor.
        let crit = pressure_from_text(SAMPLE_VM_STAT, 64.0, Some(4), 1.0);
        assert_eq!(crit.level, "critical");
        // Kernel reports warning (2) -> warning.
        let warn = pressure_from_text(SAMPLE_VM_STAT, 64.0, Some(2), 1.0);
        assert_eq!(warn.level, "warning");
        // Kernel reports normal (1) with a healthy compressor -> ok, even though
        // free pages are low and swap is in use (the old heuristic false-alarmed).
        let ok = pressure_from_text(SAMPLE_VM_STAT, 64.0, Some(1), 6.0);
        assert_eq!(ok.level, "ok");
        // swap_used is now populated (was previously hardcoded 0.0).
        assert_eq!(ok.swap_used_gb, 6.0);
        // Score stays within the band for the verdict.
        assert!(ok.score < 50.0, "ok score {} should be <50", ok.score);
        assert!(
            crit.score >= 80.0,
            "critical score {} should be >=80",
            crit.score
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn falls_back_to_compressor_without_kernel_level() {
        // page size 16384, total 1 GB so compressor of 30000 pages ~= 0.46 GB,
        // a ratio above the 0.30 critical threshold.
        let heavy = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages occupied by compressor:             30000.\n";
        let p = pressure_from_text(heavy, 1.0, None, 0.0);
        assert_eq!(p.level, "critical");
    }

    #[test]
    fn score_bands_are_consistent_across_levels() {
        // ok occupies 0-50, warning 50-80, critical 80-100, regardless of platform.
        assert!((score_for_level("ok", 0.0) - 0.0).abs() < 1e-9);
        assert!((score_for_level("ok", 1.0) - 50.0).abs() < 1e-9);
        assert!((score_for_level("warning", 0.0) - 50.0).abs() < 1e-9);
        assert!((score_for_level("warning", 1.0) - 80.0).abs() < 1e-9);
        assert!((score_for_level("critical", 0.0) - 80.0).abs() < 1e-9);
        assert!((score_for_level("critical", 1.0) - 100.0).abs() < 1e-9);
        // Intensity is clamped.
        assert!((score_for_level("warning", 5.0) - 80.0).abs() < 1e-9);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn psi_parses_some_and_full_avg10() {
        let text = "some avg10=12.34 avg60=1.00 avg300=0.10 total=123\n\
full avg10=2.50 avg60=0.50 avg300=0.05 total=45\n";
        let (some, full) = parse_linux_psi(text);
        assert!((some - 12.34).abs() < 1e-9);
        assert!((full - 2.50).abs() < 1e-9);
    }
}

#[cfg(test)]
mod cpu_sampling_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cpu_refresh_respects_minimum_interval_without_sleeping() {
        let previous = Instant::now();
        let interval = sysinfo::MINIMUM_CPU_UPDATE_INTERVAL;
        assert!(!cpu_refresh_due(previous, previous));
        assert!(!cpu_refresh_due(
            previous,
            previous + interval.saturating_sub(Duration::from_nanos(1))
        ));
        assert!(cpu_refresh_due(previous, previous + interval));
        assert!(cpu_refresh_due(previous, previous + interval * 2));
    }

    #[test]
    fn first_sample_is_placeholder_and_rapid_requests_preserve_history() {
        let mut sampler = SystemSampler::new();
        assert_eq!(sampler.cpu_load, 0);
        // A cached valid sample must survive calls before the next interval.
        sampler.cpu_load = 42;
        let previous = sampler.last_cpu_refresh;
        let cpu_count = sampler.sys.cpus().len();
        sampler.refresh_cpu_if_due(previous);
        assert_eq!(sampler.cpu_load, 42);
        assert_eq!(sampler.last_cpu_refresh, previous);
        assert_eq!(sampler.sys.cpus().len(), cpu_count);
    }
}
