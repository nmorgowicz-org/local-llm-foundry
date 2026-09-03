//! Pure two-sided `n_cpu_moe` placement search over the Phase 4b seam.
//!
//! Device usage falls as expert layers move to host memory, while host usage
//! rises. Those are separate monotone predicates; combining them into one
//! predicate would make the interval boundary impossible to search safely.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::fit_probe::{DEFAULT_FIT_RESERVE_MIB, FitProbeError, FitReader, FitReading};

#[derive(Debug, Clone, Copy)]
pub struct FitSearchConfig {
    pub n_max: u32,
    pub device_budget_mib: u64,
    pub host_budget_mib: u64,
    pub reserve_mib: u64,
    pub timeout: Duration,
}

impl FitSearchConfig {
    pub fn with_default_reserve(
        n_max: u32,
        device_budget_mib: u64,
        host_budget_mib: u64,
        timeout: Duration,
    ) -> Self {
        Self {
            n_max,
            device_budget_mib,
            host_budget_mib,
            reserve_mib: DEFAULT_FIT_RESERVE_MIB,
            timeout,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FitPlacementProposal {
    pub n_cpu_moe: u32,
    pub reading: FitReading,
    pub device_headroom_mib: i64,
    pub host_headroom_mib: i64,
    pub n_dev_min: u32,
    pub n_host_max: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FitPlacementUnavailable {
    pub code: String,
    pub message: String,
    pub device_deficit_mib: Option<u64>,
    pub host_deficit_mib: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum FitPlacementResult {
    Proposal(FitPlacementProposal),
    Unavailable(FitPlacementUnavailable),
}

/// Search the device suffix and host prefix independently, then intersect them.
/// Every `n` is read at most once through the local cache, including overlap
/// between the two searches.
pub fn search(reader: &mut impl FitReader, config: FitSearchConfig) -> FitPlacementResult {
    let deadline = Instant::now() + config.timeout;
    let mut readings = HashMap::<u32, FitReading>::new();

    let device_min = match first_device_fit(reader, &mut readings, &config, deadline) {
        Ok(value) => value,
        Err(error) => return unavailable_from_error(error),
    };

    let host_max = match last_host_fit(reader, &mut readings, &config, deadline) {
        Ok(value) => value,
        Err(error) => return unavailable_from_error(error),
    };

    let Some(host_max) = host_max else {
        let reading = readings.get(&0).expect("host search reads n=0 first");
        return FitPlacementResult::Unavailable(FitPlacementUnavailable {
            code: "host_limited".into(),
            message: format!(
                "host budget is short by {} MiB at n_cpu_moe=0",
                reading
                    .host_total_mib
                    .saturating_sub(config.host_budget_mib)
            ),
            device_deficit_mib: None,
            host_deficit_mib: Some(
                reading
                    .host_total_mib
                    .saturating_sub(config.host_budget_mib),
            ),
        });
    };

    let Some(device_min) = device_min else {
        let reading = readings
            .get(&config.n_max)
            .expect("device search reads n_max last");
        return FitPlacementResult::Unavailable(FitPlacementUnavailable {
            code: "device_limited".into(),
            message: format!(
                "device budget is short by {} MiB at n_cpu_moe={}",
                reading
                    .device_total_mib
                    .saturating_add(config.reserve_mib)
                    .saturating_sub(config.device_budget_mib),
                config.n_max
            ),
            device_deficit_mib: Some(
                reading
                    .device_total_mib
                    .saturating_add(config.reserve_mib)
                    .saturating_sub(config.device_budget_mib),
            ),
            host_deficit_mib: None,
        });
    };

    if device_min > host_max {
        return FitPlacementResult::Unavailable(FitPlacementUnavailable {
            code: "disjoint_feasible_interval".into(),
            message: format!(
                "device requires n_cpu_moe >= {device_min}, while host permits n_cpu_moe <= {host_max}"
            ),
            device_deficit_mib: Some(device_min as u64),
            host_deficit_mib: Some(host_max as u64),
        });
    }

    let reading = readings
        .get(&device_min)
        .expect("device boundary reading is cached")
        .clone();
    FitPlacementResult::Proposal(FitPlacementProposal {
        n_cpu_moe: device_min,
        device_headroom_mib: headroom(
            config.device_budget_mib,
            reading.device_total_mib.saturating_add(config.reserve_mib),
        ),
        host_headroom_mib: headroom(config.host_budget_mib, reading.host_total_mib),
        reading,
        n_dev_min: device_min,
        n_host_max: host_max,
    })
}

fn first_device_fit(
    reader: &mut impl FitReader,
    readings: &mut HashMap<u32, FitReading>,
    config: &FitSearchConfig,
    deadline: Instant,
) -> Result<Option<u32>, FitProbeError> {
    if !device_fits(read_once(reader, readings, config.n_max, deadline)?, config) {
        return Ok(None);
    }
    let mut low = 0;
    let mut high = config.n_max;
    while low < high {
        let mid = low + (high - low) / 2;
        if device_fits(read_once(reader, readings, mid, deadline)?, config) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(Some(low))
}

fn last_host_fit(
    reader: &mut impl FitReader,
    readings: &mut HashMap<u32, FitReading>,
    config: &FitSearchConfig,
    deadline: Instant,
) -> Result<Option<u32>, FitProbeError> {
    if !host_fits(read_once(reader, readings, 0, deadline)?, config) {
        return Ok(None);
    }
    let mut low = 0;
    let mut high = config.n_max;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if host_fits(read_once(reader, readings, mid, deadline)?, config) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(Some(low))
}

fn read_once<'a>(
    reader: &mut impl FitReader,
    readings: &'a mut HashMap<u32, FitReading>,
    n: u32,
    deadline: Instant,
) -> Result<&'a FitReading, FitProbeError> {
    if Instant::now() >= deadline {
        return Err(FitProbeError::Timeout);
    }
    if let std::collections::hash_map::Entry::Vacant(entry) = readings.entry(n) {
        entry.insert(reader.read(n)?);
    }
    Ok(readings
        .get(&n)
        .expect("reading inserted or already present"))
}

fn device_fits(reading: &FitReading, config: &FitSearchConfig) -> bool {
    reading.device_total_mib.saturating_add(config.reserve_mib) <= config.device_budget_mib
}

fn host_fits(reading: &FitReading, config: &FitSearchConfig) -> bool {
    reading.host_total_mib <= config.host_budget_mib
}

fn headroom(budget: u64, used: u64) -> i64 {
    budget.saturating_sub(used) as i64 - used.saturating_sub(budget) as i64
}

fn unavailable_from_error(error: FitProbeError) -> FitPlacementResult {
    let code = match error {
        FitProbeError::Timeout => "probe_timeout",
        _ => "probe_unavailable",
    };
    FitPlacementResult::Unavailable(FitPlacementUnavailable {
        code: code.into(),
        message: error.to_string(),
        device_deficit_mib: None,
        host_deficit_mib: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::fit_probe::{FitProbeError, FixtureFitReader};
    use std::collections::{HashMap, HashSet};

    #[derive(Default)]
    struct MapReader {
        values: HashMap<u32, FitReading>,
        seen: HashSet<u32>,
    }

    impl MapReader {
        fn with_values(values: impl IntoIterator<Item = (u32, u64, u64)>) -> Self {
            Self {
                values: values
                    .into_iter()
                    .map(|(n, device, host)| {
                        (
                            n,
                            FitReading {
                                n_cpu_moe: n,
                                device_total_mib: device,
                                host_total_mib: host,
                                model_mib: device,
                                context_mib: 0,
                                compute_mib: 0,
                            },
                        )
                    })
                    .collect(),
                seen: HashSet::new(),
            }
        }
    }

    impl FitReader for MapReader {
        fn read(&mut self, n_cpu_moe: u32) -> Result<FitReading, FitProbeError> {
            assert!(self.seen.insert(n_cpu_moe), "n={n_cpu_moe} read twice");
            self.values
                .get(&n_cpu_moe)
                .cloned()
                .ok_or_else(|| FitProbeError::Unavailable(format!("missing n={n_cpu_moe}")))
        }
    }

    fn complete_cuda_fixture() -> MapReader {
        let mut captured = FixtureFitReader::embedded().unwrap();
        let n16 = captured.read(16).unwrap();
        let n17 = captured.read(17).unwrap();
        let n18 = captured.read(18).unwrap();
        let n19 = captured.read(19).unwrap();
        let n20 = captured.read(20).unwrap();
        let mut values = Vec::new();
        for n in 0..=15 {
            values.push((n, n16.device_total_mib + 4_000, 500));
        }
        values.extend([
            (16, n16.device_total_mib, n16.host_total_mib),
            (17, n17.device_total_mib, n17.host_total_mib),
            (18, n18.device_total_mib, n18.host_total_mib),
            (19, n19.device_total_mib, n19.host_total_mib),
            (20, n20.device_total_mib, n20.host_total_mib),
        ]);
        MapReader::with_values(values)
    }

    fn fixture_config(n_max: u32, device_budget_mib: u64, reserve_mib: u64) -> FitSearchConfig {
        FitSearchConfig {
            n_max,
            device_budget_mib,
            host_budget_mib: 10_000,
            reserve_mib,
            timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn cuda_fixture_proposes_exact_unit_step_boundary() {
        let mut reader = complete_cuda_fixture();
        let result = search(&mut reader, fixture_config(20, 16_384, 1_024));
        let FitPlacementResult::Proposal(proposal) = result else {
            panic!("expected proposal");
        };
        assert_eq!(proposal.n_cpu_moe, 18);
    }

    #[test]
    fn default_reserve_is_the_product_fit_target() {
        let config =
            FitSearchConfig::with_default_reserve(20, 16_384, 10_000, Duration::from_secs(1));
        assert_eq!(config.reserve_mib, DEFAULT_FIT_RESERVE_MIB);
    }

    #[test]
    fn search_reports_probe_failure_without_a_proposal() {
        let mut reader = complete_cuda_fixture();
        let result = search(&mut reader, fixture_config(20, 1, 1_024));
        assert!(matches!(
            result,
            FitPlacementResult::Unavailable(FitPlacementUnavailable { code, .. })
                if code == "device_limited"
        ));
    }

    #[test]
    fn interval_search_returns_lower_device_boundary() {
        let mut reader = MapReader::with_values((0..=32).map(|n| {
            let device = if n < 12 { 110 } else { 90 };
            let host = if n <= 24 { 100 } else { 110 };
            (n, device, host)
        }));
        let result = search(
            &mut reader,
            FitSearchConfig {
                n_max: 32,
                device_budget_mib: 100,
                host_budget_mib: 100,
                reserve_mib: 0,
                timeout: Duration::from_secs(1),
            },
        );
        let FitPlacementResult::Proposal(proposal) = result else {
            panic!("expected interval proposal");
        };
        assert_eq!(proposal.n_cpu_moe, 12);
        assert_eq!(proposal.n_host_max, 24);
    }

    #[test]
    fn disjoint_interval_names_both_boundaries() {
        let mut reader = MapReader::with_values((0..=20).map(|n| {
            (
                n,
                if n < 12 { 110 } else { 90 },
                if n <= 8 { 100 } else { 110 },
            )
        }));
        let result = search(
            &mut reader,
            FitSearchConfig {
                n_max: 20,
                device_budget_mib: 100,
                host_budget_mib: 100,
                reserve_mib: 0,
                timeout: Duration::from_secs(1),
            },
        );
        assert!(matches!(
            result,
            FitPlacementResult::Unavailable(FitPlacementUnavailable { code, message, .. })
                if code == "disjoint_feasible_interval"
                    && message.contains("12")
                    && message.contains("8")
        ));
    }

    #[test]
    fn host_limited_at_zero_reads_only_n_zero_for_host_search() {
        let mut reader = MapReader::with_values((0..=20).map(|n| (n, 90, 101 + n as u64)));
        let result = search(
            &mut reader,
            FitSearchConfig {
                n_max: 20,
                device_budget_mib: 100,
                host_budget_mib: 100,
                reserve_mib: 0,
                timeout: Duration::from_secs(1),
            },
        );
        assert!(matches!(
            result,
            FitPlacementResult::Unavailable(FitPlacementUnavailable { code, .. })
                if code == "host_limited"
        ));
        assert!(reader.seen.contains(&0));
    }

    #[test]
    fn reserve_increase_moves_boundary_up_monotonically() {
        let make_reader = || {
            MapReader::with_values((0..=24).map(|n| {
                let device = 17_000u64.saturating_sub(n as u64 * 200);
                (n, device, 100)
            }))
        };
        let mut low_reader = make_reader();
        let mut high_reader = make_reader();
        let low = search(
            &mut low_reader,
            FitSearchConfig {
                n_max: 24,
                device_budget_mib: 16_384,
                host_budget_mib: 100,
                reserve_mib: 1_024,
                timeout: Duration::from_secs(1),
            },
        );
        let high = search(
            &mut high_reader,
            FitSearchConfig {
                n_max: 24,
                device_budget_mib: 16_384,
                host_budget_mib: 100,
                reserve_mib: 3_072,
                timeout: Duration::from_secs(1),
            },
        );
        let FitPlacementResult::Proposal(low) = low else {
            panic!("expected low-reserve proposal");
        };
        let FitPlacementResult::Proposal(high) = high else {
            panic!("expected high-reserve proposal");
        };
        assert!(high.n_cpu_moe > low.n_cpu_moe);
    }
}
