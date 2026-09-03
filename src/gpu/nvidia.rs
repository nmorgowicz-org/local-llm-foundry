use anyhow::Result;
use std::collections::BTreeMap;

use super::{GpuBackend, GpuMetrics};

pub struct NvidiaBackend;

impl GpuBackend for NvidiaBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        let mut cmd = std::process::Command::new("nvidia-smi");
        crate::platform::no_window(&mut cmd);
        let output = cmd
            .args([
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,power.draw,power.limit,memory.used,memory.total,clocks.gr,clocks.mem",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("failed to run nvidia-smi: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("nvidia-smi failed: {stderr}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_nvidia_csv(&stdout)
    }

    fn name(&self) -> &str {
        "nvidia"
    }
}

pub fn parse_nvidia_csv(csv: &str) -> Result<BTreeMap<String, GpuMetrics>> {
    let mut metrics = BTreeMap::new();

    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() < 10 {
            continue;
        }

        // nvidia-smi reports an unsupported/unavailable metric as the literal
        // string "[N/A]" rather than omitting the field. Parsing that as a
        // number would silently fall back to 0 and read as a real (e.g. 0°C)
        // measurement, so skip the whole row instead of reporting fabricated
        // zeros for a card whose readings we don't actually have.
        if fields[2..10]
            .iter()
            .any(|field| field.eq_ignore_ascii_case("[N/A]") || field.eq_ignore_ascii_case("N/A"))
        {
            continue;
        }

        let index = fields[0];
        let name = fields[1];
        let temp = fields[2].parse::<f32>().unwrap_or(0.0);
        let load = fields[3].parse::<u32>().unwrap_or(0);
        let power_consumption = fields[4].parse::<f32>().unwrap_or(0.0);
        let power_limit = fields[5].parse::<f64>().unwrap_or(0.0) as u32;
        let vram_used = fields[6].parse::<u64>().unwrap_or(0); // MiB from nvidia-smi
        let vram_total = fields[7].parse::<u64>().unwrap_or(0); // MiB from nvidia-smi
        let sclk_mhz = fields[8].parse::<u32>().unwrap_or(0);
        let mclk_mhz = fields[9].parse::<u32>().unwrap_or(0);

        let card_name = format!("GPU{index} {name}");
        metrics.insert(
            card_name,
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
            },
        );
    }

    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_csv() {
        let csv = include_str!("../../tests/fixtures/nvidia_smi_csv.txt");
        let metrics = parse_nvidia_csv(csv).unwrap();
        assert_eq!(metrics.len(), 2);

        let gpu0 = metrics.get("GPU0 NVIDIA GeForce RTX 4090").unwrap();
        assert!((gpu0.temp - 45.0).abs() < 0.1);
        assert_eq!(gpu0.load, 87);
        assert!((gpu0.power_consumption - 320.5).abs() < 0.1);
        assert_eq!(gpu0.power_limit, 450);
        assert_eq!(gpu0.vram_used, 20480);
        assert_eq!(gpu0.vram_total, 24564);
        assert_eq!(gpu0.sclk_mhz, 2520);
        assert_eq!(gpu0.mclk_mhz, 10501);
    }

    #[test]
    fn test_parse_nvidia_csv_empty() {
        let metrics = parse_nvidia_csv("").unwrap();
        assert!(metrics.is_empty());
    }
}
