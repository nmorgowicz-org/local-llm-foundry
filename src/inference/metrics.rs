use crate::inference::InferenceBackend;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub enum HealthState {
    Ok,
    Degraded,
    NotLoaded,
    Unreachable,
}

/// Availability of a normalized metric.  `Unavailable` is intentional: a
/// backend must not fabricate zeroes when its runtime does not expose a
/// metric. `Degraded` means the value is present but came from a partial or
/// fallback source.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub enum MetricState {
    Effective,
    Degraded,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MetricDescriptor {
    pub state: MetricState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
}

impl MetricDescriptor {
    fn present(value: Value, unit: &'static str) -> Self {
        Self {
            state: MetricState::Effective,
            value: Some(value),
            unit: Some(unit),
            reason: None,
        }
    }

    fn absent(reason: &'static str, unit: &'static str) -> Self {
        Self {
            state: MetricState::Unavailable,
            value: None,
            unit: Some(unit),
            reason: Some(reason),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct InferenceMetricsSnapshot {
    pub sampled_at: SystemTime,
    pub backend: InferenceBackend,
    // Health
    pub health: Option<HealthState>,
    pub ready: Option<bool>,
    // Identity
    pub model: Option<String>,
    pub uptime_seconds: Option<f64>,
    // Throughput
    pub generation_tokens_per_second: Option<f64>,
    pub prompt_tokens_per_second: Option<f64>,
    // Queue
    pub running_requests: Option<u64>,
    pub waiting_requests: Option<u64>,
    // Totals (cumulative)
    pub completed_requests_total: Option<u64>,
    pub prompt_tokens_total: Option<u64>,
    pub completion_tokens_total: Option<u64>,
    pub steps_executed: Option<u64>,
    pub global_cache_hit_rate: Option<f64>,
    pub global_cache_entries: Option<u64>,
    pub ttft: Option<f64>,
    pub speculative_acceptance_rate: Option<f64>,

    // Memory (always in bytes, regardless of backend source unit)
    pub active_memory_bytes: Option<u64>,
    pub peak_memory_bytes: Option<u64>,
    pub cache_memory_bytes: Option<u64>,
    // Structured opaque payloads — card registry maps these, not raw JSON
    pub cache_metrics: Option<serde_json::Value>,
    pub active_requests: Option<Vec<serde_json::Value>>,
    pub backend_details: Option<serde_json::Value>,
}

impl InferenceMetricsSnapshot {
    /// Start a poll with unavailable values rather than fabricating zero activity.
    pub fn empty(backend: InferenceBackend) -> Self {
        Self {
            sampled_at: SystemTime::now(),
            backend,
            health: None,
            ready: None,
            model: None,
            uptime_seconds: None,
            generation_tokens_per_second: None,
            prompt_tokens_per_second: None,
            running_requests: None,
            waiting_requests: None,
            completed_requests_total: None,
            prompt_tokens_total: None,
            completion_tokens_total: None,
            steps_executed: None,
            global_cache_hit_rate: None,
            global_cache_entries: None,
            ttft: None,
            speculative_acceptance_rate: None,
            active_memory_bytes: None,
            peak_memory_bytes: None,
            cache_memory_bytes: None,
            cache_metrics: None,
            active_requests: None,
            backend_details: None,
        }
    }

    /// A stable, privacy-safe dictionary for dashboard and API consumers.
    /// Only aggregate counters and rates are included; request ids, model
    /// names, and opaque backend payloads are deliberately excluded.
    pub fn metric_dictionary(&self) -> BTreeMap<&'static str, MetricDescriptor> {
        let mut metrics = BTreeMap::new();
        macro_rules! metric {
            ($name:literal, $field:expr, $unit:literal) => {
                metrics.insert(
                    $name,
                    match $field {
                        Some(value) => MetricDescriptor::present(serde_json::json!(value), $unit),
                        None => {
                            MetricDescriptor::absent("backend did not report this metric", $unit)
                        }
                    },
                );
            };
        }
        metric!(
            "generation_tokens_per_second",
            self.generation_tokens_per_second,
            "tokens_per_second"
        );
        metric!(
            "prompt_tokens_per_second",
            self.prompt_tokens_per_second,
            "tokens_per_second"
        );
        metric!("running_requests", self.running_requests, "requests");
        metric!("waiting_requests", self.waiting_requests, "requests");
        metric!(
            "completed_requests_total",
            self.completed_requests_total,
            "requests"
        );
        metric!("prompt_tokens_total", self.prompt_tokens_total, "tokens");
        metric!(
            "completion_tokens_total",
            self.completion_tokens_total,
            "tokens"
        );
        metric!("steps_executed", self.steps_executed, "steps");
        metric!("global_cache_hit_rate", self.global_cache_hit_rate, "ratio");
        metric!("global_cache_entries", self.global_cache_entries, "entries");
        metric!("ttft", self.ttft, "milliseconds");
        metric!(
            "speculative_acceptance_rate",
            self.speculative_acceptance_rate,
            "ratio"
        );
        metric!("active_memory_bytes", self.active_memory_bytes, "bytes");
        metric!("peak_memory_bytes", self.peak_memory_bytes, "bytes");
        metric!("cache_memory_bytes", self.cache_memory_bytes, "bytes");
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_marks_missing_values_unavailable_without_zero_filling() {
        let snapshot = InferenceMetricsSnapshot {
            sampled_at: SystemTime::UNIX_EPOCH,
            backend: InferenceBackend::RapidMlx,
            health: None,
            ready: None,
            model: None,
            uptime_seconds: None,
            generation_tokens_per_second: Some(12.5),
            prompt_tokens_per_second: None,
            running_requests: None,
            waiting_requests: None,
            completed_requests_total: None,
            prompt_tokens_total: None,
            completion_tokens_total: None,
            steps_executed: None,
            global_cache_hit_rate: None,
            global_cache_entries: None,
            ttft: None,
            speculative_acceptance_rate: None,
            active_memory_bytes: None,
            peak_memory_bytes: None,
            cache_memory_bytes: None,
            cache_metrics: None,
            active_requests: None,
            backend_details: None,
        };
        let dictionary = snapshot.metric_dictionary();
        assert_eq!(
            dictionary["generation_tokens_per_second"].state,
            MetricState::Effective
        );
        assert_eq!(
            dictionary["prompt_tokens_per_second"].state,
            MetricState::Unavailable
        );
        assert!(dictionary["prompt_tokens_per_second"].value.is_none());
    }
}
