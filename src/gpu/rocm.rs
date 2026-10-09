use anyhow::Result;
use std::collections::BTreeMap;
use std::process::Command;

use super::{GpuBackend, GpuMetrics};

pub struct RocmBackend;

impl GpuBackend for RocmBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let mut cmd = Command::new("rocm-smi");
        crate::platform::no_window(&mut cmd);
        let output = cmd
            .args([
                "--json",
                "--showclocks",
                "--showtemp",
                "--showuse",
                "--showpower",
                "--showmaxpower",
                "--showmeminfo",
                "vram",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("failed to run rocm-smi: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("rocm-smi failed: {stderr}");
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| anyhow::anyhow!("failed to parse rocm-smi JSON: {e}"))?;

        parse_rocm_json(&json)
    }

    fn name(&self) -> &str {
        "rocm"
    }
}

pub fn parse_rocm_json(json: &serde_json::Value) -> Result<BTreeMap<String, GpuMetrics>> {
    let mut metrics = BTreeMap::new();
    let object = json
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected JSON object"))?;

    for (card_name, card) in object {
        let temp = card
            .get("Temperature (Sensor junction) (C)")
            .or_else(|| card.get("Temperature (Sensor edge) (C)"))
            .and_then(|v| v.as_str())
            .and_then(|t| t.parse::<f32>().ok())
            .unwrap_or(0.0);

        let load = card
            .get("GPU use (%)")
            .and_then(|v| v.as_str())
            .and_then(|u| u.parse::<u32>().ok())
            .unwrap_or(0);

        let power_consumption = card
            .get("Current Socket Graphics Package Power (W)")
            .or_else(|| card.get("Average Graphics Package Power (W)"))
            .or_else(|| card.get("Average Package Power (W)"))
            .and_then(|v| v.as_str())
            .and_then(|p| p.parse::<f32>().ok())
            .unwrap_or(0.0);

        let power_limit = card
            .get("Max Graphics Package Power (W)")
            .or_else(|| card.get("Power Limit (W)"))
            .or_else(|| card.get("Power Limit (mW)"))
            .and_then(|v| v.as_str())
            .and_then(|p| p.parse::<f64>().ok())
            .map(|p| {
                if p > 1000.0 {
                    (p / 1000.0) as u32
                } else {
                    p as u32
                }
            })
            .unwrap_or(0);

        let vram_used = card
            .get("VRAM Total Used Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v / 1024 / 1024)
            .unwrap_or(0);

        let vram_total = card
            .get("VRAM Total Memory (B)")
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v / 1024 / 1024)
            .unwrap_or(0);

        let parse_clock = |key: &str| -> u32 {
            card.get(key)
                .and_then(|v| v.as_str())
                .and_then(|s| {
                    let cleaned = s.replace("(", "").replace(")", "").replace("Mhz", "");
                    cleaned.trim().parse::<u32>().ok()
                })
                .unwrap_or(0)
        };

        let sclk_mhz = parse_clock("sclk clock speed:");
        let mclk_mhz = parse_clock("mclk clock speed:");

        metrics.insert(
            card_name.clone(),
            GpuMetrics {
                temp,
                load,
                power_consumption,
                power_limit,
                vram_used,
                vram_total,
                sclk_mhz,
                mclk_mhz,
                metal_gpu_limit_mb: None,
                ..GpuMetrics::default()
            },
        );
    }

    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rocm_json() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/rocm_smi_output.json"))
                .unwrap();

        let metrics = parse_rocm_json(&json).unwrap();
        assert_eq!(metrics.len(), 1);

        let card = metrics.get("card0").unwrap();
        assert!((card.temp - 45.0).abs() < 0.1);
        assert_eq!(card.load, 87);
        assert!((card.power_consumption - 180.5).abs() < 0.1);
        assert_eq!(card.power_limit, 300);
        assert_eq!(card.vram_used, 15360); // 16106127360 / 1024 / 1024
        assert_eq!(card.vram_total, 16384); // 17179869184 / 1024 / 1024
        assert_eq!(card.sclk_mhz, 1725);
        assert_eq!(card.mclk_mhz, 1200);
    }

    #[test]
    fn test_parse_rocm_json_empty() {
        let json: serde_json::Value = serde_json::from_str("{}").unwrap();
        let metrics = parse_rocm_json(&json).unwrap();
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_parse_rocm_json_missing_fields() {
        let json: serde_json::Value =
            serde_json::from_str(r#"{"card0": {"GPU use (%)": "50"}}"#).unwrap();
        let metrics = parse_rocm_json(&json).unwrap();
        let card = metrics.get("card0").unwrap();
        assert_eq!(card.load, 50);
        assert_eq!(card.temp, 0.0);
        assert_eq!(card.power_consumption, 0.0);
    }
}
