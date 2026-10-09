use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use wmi::{Variant, WMIConnection};

use super::{GpuBackend, GpuMetrics};

pub struct WmiGpuBackend;

/// Try to read true VRAM in MiB from the registry for the given adapter name.
///
/// The registry key `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}`
/// contains numbered subkeys (`0000`, `0001`, …) for each display adapter. Each subkey
/// has `DriverDesc` (string) and `HardwareInformation.qwMemorySize` (QWORD, bytes).
/// `Win32_VideoController.AdapterRAM` is a UINT32 that wraps at 4 GiB, so GPUs with
/// more than 4 GiB report a wrong value. This function provides the corrected value.
///
/// Returns `None` if the registry is inaccessible or no subkey matches `adapter_name`.
fn registry_vram_mib(adapter_name: &str) -> Option<u64> {
    use winreg::RegKey;
    use winreg::enums::HKEY_LOCAL_MACHINE;

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    let class_key = hklm
        .open_subkey(
            r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}",
        )
        .ok()?;

    let adapter_name_lower = adapter_name.trim().to_lowercase();

    for subkey_name in class_key.enum_keys().flatten() {
        // Only consider numeric subkeys (0000, 0001, …)
        if !subkey_name.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let subkey = match class_key.open_subkey(&subkey_name) {
            Ok(k) => k,
            Err(_) => continue,
        };

        // Read DriverDesc for matching against WMI adapter name.
        let driver_desc: Option<String> = subkey.get_value("DriverDesc").ok();

        let matched = driver_desc.as_deref().is_some_and(|desc| {
            let desc_lower = desc.trim().to_lowercase();
            desc_lower == adapter_name_lower
                || desc_lower.contains(&adapter_name_lower)
                || adapter_name_lower.contains(&desc_lower)
        });

        if !matched {
            // Also try HardwareInformation.AdapterString as a secondary name.
            let adapter_string: Option<String> =
                subkey.get_value("HardwareInformation.AdapterString").ok();
            let matched_secondary = adapter_string.as_deref().is_some_and(|s| {
                let s_lower = s.trim().to_lowercase();
                s_lower == adapter_name_lower
                    || s_lower.contains(&adapter_name_lower)
                    || adapter_name_lower.contains(&s_lower)
            });
            if !matched_secondary {
                continue;
            }
        }

        // Read HardwareInformation.qwMemorySize (REG_QWORD, bytes).
        let qw_bytes: Result<u64, _> = subkey.get_value("HardwareInformation.qwMemorySize");
        if let Ok(bytes) = qw_bytes
            && bytes > 0
        {
            return Some(bytes / (1024 * 1024));
        }

        // If we matched but the QWORD value is missing or zero, return None so the
        // caller falls back to AdapterRAM rather than reporting 0 MiB.
        return None;
    }

    None
}

impl GpuBackend for WmiGpuBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let wmi =
            WMIConnection::new().map_err(|e| anyhow::anyhow!("WMI connection failed: {e}"))?;

        let rows: Vec<HashMap<String, Variant>> = wmi
            .raw_query::<HashMap<String, Variant>>(
                "SELECT Name, AdapterRAM FROM Win32_VideoController",
            )
            .map_err(|e| anyhow::anyhow!("Win32_VideoController query failed: {e}"))?;

        let mut metrics = BTreeMap::new();
        let mut idx = 0usize;

        for row in &rows {
            let name = match row.get("Name") {
                Some(Variant::String(s)) => s.trim().to_string(),
                _ => continue,
            };

            // Skip virtual/remote display adapters
            if name.is_empty() || name.contains("Remote Display") || name.contains("Virtual") {
                continue;
            }

            // AdapterRAM is UINT32 in WMI (bytes, wraps at ~4 GB).
            // Convert to MiB to match other backends. Treat 0 as unknown.
            let adapter_ram_mib = match row.get("AdapterRAM") {
                Some(Variant::UI4(bytes)) if *bytes > 0 => *bytes as u64 / (1024 * 1024),
                Some(Variant::UI8(bytes)) if *bytes > 0 => *bytes / (1024 * 1024),
                _ => 0,
            };

            // Prefer the registry QWORD value which is not capped at 4 GiB.
            // Fall back to AdapterRAM when the registry lookup fails or returns 0.
            let vram_total = registry_vram_mib(&name).unwrap_or(adapter_ram_mib);

            metrics.insert(
                format!("GPU{idx} {name}"),
                GpuMetrics {
                    temp: 0.0,
                    load: 0,
                    power_consumption: 0.0,
                    power_limit: 0,
                    vram_used: 0,
                    vram_total,
                    sclk_mhz: 0,
                    mclk_mhz: 0,
                    // Metal's unified-memory limit is specific to Apple GPUs.
                    metal_gpu_limit_mb: None,
                    ..GpuMetrics::default()
                },
            );
            idx += 1;
        }

        Ok(metrics)
    }

    fn name(&self) -> &str {
        "wmi"
    }
}
