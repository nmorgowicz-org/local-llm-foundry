#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LlamaMetrics {
    pub prompt_tokens_per_sec: f64,
    pub generation_tokens_per_sec: f64,
    pub throughput_source: String,
    pub prompt_throughput_active: bool,
    pub generation_throughput_active: bool,
    pub last_prompt_tokens_per_sec: f64,
    pub last_generation_tokens_per_sec: f64,
    pub last_prompt_throughput_unix_ms: u64,
    pub last_generation_throughput_unix_ms: u64,
    pub prompt_tokens_total: u64,
    pub generation_tokens_total: u64,
    pub tokens_per_decode: f64,
    pub speculative_acceptance_rate: Option<f64>,
    pub predicted_tokens_total: u64,
    pub kv_cache_tokens: u64,
    pub kv_cache_max: u64,
    pub kv_cache_tokens_available: bool,
    pub kv_cache_tokens_source: String,
    pub kv_cache_high_water: u64,
    pub context_live_tokens: u64,
    pub context_live_tokens_available: bool,
    pub context_live_tokens_source: String,
    pub context_capacity_tokens: u64,
    pub context_high_water_tokens: u64,
    pub slots_idle: u32,
    pub slots_processing: u32,
    pub active_task_id: Option<u64>,
    pub last_task_id: Option<u64>,
    pub slot_generation_tokens: u64,
    pub slot_generation_remaining: u64,
    pub slot_generation_limit: u64,
    pub slot_generation_active: bool,
    pub slot_generation_available: bool,
    pub slots: Vec<SlotSnapshot>,
    pub requests_processing: u32,
    pub n_busy_slots_per_decode: f64,
    pub status: String,
    pub model_name: String,
    pub model_params: Option<u64>,
    pub model_ctx_train: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MetricConfigItem {
    pub label: String,
    pub value: String,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlotSnapshot {
    pub id: Option<u64>,
    pub n_ctx: u64,
    pub is_processing: bool,
    pub id_task: Option<u64>,
    pub output_tokens: u64,
    pub output_remaining: u64,
    pub output_limit: u64,
    pub output_active: bool,
    pub output_available: bool,
    pub context_live_tokens: Option<u64>,
    pub context_live_tokens_source: Option<String>,
    pub speculative_enabled: bool,
    pub speculative_type: Option<String>,
    pub speculative_config: Vec<MetricConfigItem>,
    pub sampler_stack: Vec<String>,
    pub sampler_config: Vec<MetricConfigItem>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SlotSnapshotInput {
    is_processing: bool,
    task_id: Option<u64>,
    output_tokens: u64,
    output_remaining: u64,
    output_limit: u64,
    output_active: bool,
    output_available: bool,
    slot_context: Option<(u64, &'static str)>,
}

#[derive(Debug, Clone, Default)]
pub struct PrometheusValues {
    pub prompt_tokens_total: f64,
    pub prompt_seconds_total: f64,
    pub predicted_tokens_total: f64,
    pub predicted_seconds_total: f64,
    pub n_tokens_max: u64,
    pub requests_processing: u32,
    pub n_decode_total: f64,
    pub n_busy_slots_per_decode: f64,
    // Derived: predicted_tokens_total / n_decode_total — spec efficiency (>1 means drafts accepted)
    pub tokens_per_decode: f64,
    pub speculative_draft_tokens_total: f64,
    pub speculative_accepted_tokens_total: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlotValues {
    pub kv_cache_tokens: u64,
    pub kv_cache_max: u64,
    pub kv_cache_tokens_available: bool,
    pub kv_cache_tokens_source: String,
    pub slots_idle: u32,
    pub slots_processing: u32,
    pub active_task_id: Option<u64>,
    pub last_task_id: Option<u64>,
    pub slot_generation_tokens: u64,
    pub slot_generation_remaining: u64,
    pub slot_generation_limit: u64,
    pub slot_generation_active: bool,
    pub slot_generation_available: bool,
    pub slots: Vec<SlotSnapshot>,
}

/// Parse Prometheus text format and extract the metrics we care about.
/// llama.cpp uses colon-separated names like `llamacpp:prompt_tokens_total`.
pub fn parse_prometheus_metrics(body: &str) -> PrometheusValues {
    let mut vals = PrometheusValues::default();
    for line in body.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let name = match parts.next() {
            Some(n) => n,
            None => continue,
        };
        let name = name.split_once('{').map_or(name, |(name, _)| name);
        let value = match parts.next().and_then(|v| v.parse::<f64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        match name {
            "llamacpp:prompt_tokens_total" => vals.prompt_tokens_total = value,
            "llamacpp:prompt_seconds_total" => vals.prompt_seconds_total = value,
            "llamacpp:tokens_predicted_total" => vals.predicted_tokens_total = value,
            "llamacpp:tokens_predicted_seconds_total" => vals.predicted_seconds_total = value,
            "llamacpp:n_tokens_max" => vals.n_tokens_max = value as u64,
            "llamacpp:requests_processing" => vals.requests_processing = value as u32,
            "llamacpp:n_decode_total" => vals.n_decode_total = value,
            "llamacpp:n_busy_slots_per_decode" => vals.n_busy_slots_per_decode = value,
            "llamacpp:spec_decode_num_draft_tokens_total" => {
                vals.speculative_draft_tokens_total = value
            }
            "llamacpp:spec_decode_num_accepted_tokens_total" => {
                vals.speculative_accepted_tokens_total = value
            }
            _ => {}
        }
    }
    if vals.n_decode_total > 0.0 {
        vals.tokens_per_decode = vals.predicted_tokens_total / vals.n_decode_total;
    }
    vals
}

pub fn parse_slot_metrics(body: &str) -> Option<SlotValues> {
    let slots = serde_json::from_str::<Vec<serde_json::Value>>(body).ok()?;
    let mut vals = SlotValues::default();

    for slot in &slots {
        let is_processing = slot
            .get("is_processing")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if is_processing {
            vals.slots_processing += 1;
        } else {
            vals.slots_idle += 1;
        }

        if let Some(n_ctx) = slot.get("n_ctx").and_then(|v| v.as_u64()) {
            vals.kv_cache_max = vals.kv_cache_max.saturating_add(n_ctx);
        }

        let slot_context = slot_context_tokens(slot);
        if let Some((tokens, source)) = slot_context {
            vals.kv_cache_tokens = vals.kv_cache_tokens.saturating_add(tokens);
            vals.kv_cache_tokens_available = true;
            if vals.kv_cache_tokens_source.is_empty() {
                vals.kv_cache_tokens_source = source.to_string();
            }
        }

        let task_id = slot.get("id_task").and_then(|v| v.as_u64());
        if vals.last_task_id.is_none() {
            vals.last_task_id = task_id;
        }
        if is_processing && vals.active_task_id.is_none() {
            vals.active_task_id = task_id;
        }

        let mut output_tokens = 0;
        let mut output_remaining = 0;
        let output_limit = slot_generation_limit(slot);
        let mut output_active = false;
        let mut output_available = false;
        if let Some((decoded, remaining, active)) = slot_generation_progress(slot) {
            output_tokens = decoded;
            output_remaining = remaining;
            output_active = active;
            output_available = true;
            vals.slot_generation_tokens = vals.slot_generation_tokens.saturating_add(decoded);
            vals.slot_generation_remaining =
                vals.slot_generation_remaining.saturating_add(remaining);
            vals.slot_generation_limit = vals
                .slot_generation_limit
                .saturating_add(output_limit.unwrap_or_else(|| decoded.saturating_add(remaining)));
            vals.slot_generation_available = true;
            vals.slot_generation_active |= active;
        }

        vals.slots.push(slot_snapshot(
            slot,
            SlotSnapshotInput {
                is_processing,
                task_id,
                output_tokens,
                output_remaining,
                output_limit: output_limit
                    .unwrap_or_else(|| output_tokens.saturating_add(output_remaining)),
                output_active,
                output_available,
                slot_context,
            },
        ));
    }

    Some(vals)
}

fn slot_snapshot(slot: &serde_json::Value, input: SlotSnapshotInput) -> SlotSnapshot {
    let params = slot.get("params");
    let speculative_enabled = slot
        .get("speculative")
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| {
            params
                .and_then(|p| nested_value(p, &["speculative", "enabled"]))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        });

    SlotSnapshot {
        id: slot.get("id").and_then(|v| v.as_u64()),
        n_ctx: slot
            .get("n_ctx")
            .and_then(|v| v.as_u64())
            .unwrap_or_default(),
        is_processing: input.is_processing,
        id_task: input.task_id,
        output_tokens: input.output_tokens,
        output_remaining: input.output_remaining,
        output_limit: input.output_limit,
        output_active: input.output_active,
        output_available: input.output_available,
        context_live_tokens: input.slot_context.map(|(tokens, _)| tokens),
        context_live_tokens_source: input.slot_context.map(|(_, source)| source.to_string()),
        speculative_enabled,
        speculative_type: speculative_type(params),
        speculative_config: speculative_config(params),
        sampler_stack: sampler_stack(params),
        sampler_config: sampler_config(params),
    }
}

fn slot_context_tokens(slot: &serde_json::Value) -> Option<(u64, &'static str)> {
    for key in ["n_tokens", "n_past", "n_ctx_used", "n_cache_tokens"] {
        if let Some(value) = slot.get(key).and_then(|v| v.as_u64()) {
            return Some((value, key));
        }
    }

    None
}

fn slot_generation_progress(slot: &serde_json::Value) -> Option<(u64, u64, bool)> {
    let token = slot.get("next_token").and_then(|v| v.as_array())?.first()?;
    let decoded = token.get("n_decoded").and_then(|v| v.as_u64())?;
    let remaining = token
        .get("n_remain")
        .and_then(|v| v.as_u64())
        .unwrap_or_default();
    let active = token
        .get("has_next_token")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Some((decoded, remaining, active))
}

fn slot_generation_limit(slot: &serde_json::Value) -> Option<u64> {
    let params = slot.get("params")?;
    for key in ["n_predict", "max_tokens"] {
        if let Some(value) = params.get(key).and_then(|v| v.as_u64())
            && value > 0
        {
            return Some(value);
        }
    }

    None
}

fn metric_param_string(params: Option<&serde_json::Value>, key: &str) -> Option<String> {
    let params = params?;
    let value = params
        .get(key)
        .or_else(|| nested_value(params, &key.split('.').collect::<Vec<_>>()))?;
    match value {
        serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn metric_config_item(
    params: Option<&serde_json::Value>,
    key: &str,
    label: &str,
) -> Option<MetricConfigItem> {
    metric_param_string(params, key).map(|value| MetricConfigItem {
        label: label.to_string(),
        value,
    })
}

fn nested_value<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}

fn sampler_stack(params: Option<&serde_json::Value>) -> Vec<String> {
    params
        .and_then(|p| p.get("samplers"))
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn sampler_config(params: Option<&serde_json::Value>) -> Vec<MetricConfigItem> {
    [
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("min_p", "min_p"),
        ("typ_p", "typ_p"),
        ("temperature", "temp"),
        ("dry_multiplier", "dry"),
        ("dry_allowed_length", "dry length"),
        ("xtc_probability", "xtc prob"),
        ("xtc_threshold", "xtc threshold"),
    ]
    .into_iter()
    .filter_map(|(key, label)| metric_config_item(params, key, label))
    .collect()
}

fn speculative_type(params: Option<&serde_json::Value>) -> Option<String> {
    // Try singular first (legacy llama.cpp), then plural (current llama.cpp with MTP)
    if let Some(v) = metric_param_string(params, "speculative.type") {
        return Some(v);
    }
    // "speculative.types" is a comma-separated list like "none,draft-mtp,ngram-mod"
    // Filter out "none" entries to show only active types
    metric_param_string(params, "speculative.types").map(|s| {
        let active: Vec<&str> = s.split(',').filter(|t| *t != "none").collect();
        if active.is_empty() {
            s
        } else {
            active.join(",")
        }
    })
}

fn speculative_config(params: Option<&serde_json::Value>) -> Vec<MetricConfigItem> {
    [
        ("speculative.types", "types"),
        ("speculative.type", "type"),
        ("speculative.n_max", "n_max"),
        ("speculative.n_min", "n_min"),
        ("speculative.p_min", "p_min"),
        ("speculative.ngram_size_n", "ngram n"),
        ("speculative.ngram_size_m", "ngram m"),
    ]
    .into_iter()
    .filter_map(|(key, label)| metric_config_item(params, key, label))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_prometheus_metrics() {
        let body = include_str!("../../tests/fixtures/prometheus_metrics.txt");
        let vals = parse_prometheus_metrics(body);

        assert!((vals.prompt_tokens_total - 10000.0).abs() < 0.1);
        assert!((vals.prompt_seconds_total - 8.1).abs() < 0.1);
        assert!((vals.predicted_tokens_total - 5000.0).abs() < 0.1);
        assert!((vals.predicted_seconds_total - 88.2).abs() < 0.1);
        assert_eq!(vals.n_tokens_max, 131072);
        assert_eq!(vals.requests_processing, 1);
        assert!((vals.n_decode_total - 42000.0).abs() < 0.1);
        assert!((vals.n_busy_slots_per_decode - 1.5).abs() < 0.01);
        assert_eq!(vals.speculative_draft_tokens_total, 200.0);
        assert_eq!(vals.speculative_accepted_tokens_total, 100.0);
    }

    #[test]
    fn test_parse_prometheus_metrics_empty() {
        let vals = parse_prometheus_metrics("");
        assert_eq!(vals.prompt_tokens_total, 0.0);
        assert_eq!(vals.n_tokens_max, 0);
    }

    #[test]
    fn test_parse_prometheus_metrics_comments_only() {
        let body = "# HELP llamacpp:prompt_tokens_total Total prompt tokens\n# TYPE llamacpp:prompt_tokens_total counter\n";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.prompt_tokens_total, 0.0);
    }

    #[test]
    fn test_parse_prometheus_metrics_with_labels() {
        let body = r#"llamacpp:requests_processing{slot="0"} 1"#;
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.requests_processing, 1);
    }

    #[test]
    fn test_parse_slot_metrics_capacity_and_status() {
        let body = r#"[{"id":0,"n_ctx":4096,"is_processing":false},{"id":1,"n_ctx":4096,"is_processing":true}]"#;
        let vals = parse_slot_metrics(body).unwrap();

        assert_eq!(vals.kv_cache_max, 8192);
        assert_eq!(vals.kv_cache_tokens, 0);
        assert!(!vals.kv_cache_tokens_available);
        assert_eq!(vals.slots_idle, 1);
        assert_eq!(vals.slots_processing, 1);
    }

    #[test]
    fn test_parse_slot_metrics_current_tokens_when_exposed() {
        let body = r#"[{"id":0,"n_ctx":4096,"is_processing":true,"n_tokens":1234}]"#;
        let vals = parse_slot_metrics(body).unwrap();

        assert_eq!(vals.kv_cache_max, 4096);
        assert_eq!(vals.kv_cache_tokens, 1234);
        assert!(vals.kv_cache_tokens_available);
        assert_eq!(vals.kv_cache_tokens_source, "n_tokens");
    }

    #[test]
    fn test_parse_slot_metrics_generation_progress() {
        let body = r#"[{"id":0,"n_ctx":4096,"is_processing":true,"id_task":2667,"params":{"n_predict":32000},"next_token":[{"has_next_token":true,"n_remain":31849,"n_decoded":151}]}]"#;
        let vals = parse_slot_metrics(body).unwrap();

        assert_eq!(vals.active_task_id, Some(2667));
        assert_eq!(vals.last_task_id, Some(2667));
        assert_eq!(vals.slot_generation_tokens, 151);
        assert_eq!(vals.slot_generation_remaining, 31849);
        assert_eq!(vals.slot_generation_limit, 32000);
        assert!(vals.slot_generation_active);
        assert!(vals.slot_generation_available);
    }

    #[test]
    fn test_parse_slot_metrics_per_slot_config() {
        let body = r#"[{
            "id":0,
            "n_ctx":4096,
            "is_processing":true,
            "id_task":2667,
            "speculative":true,
            "params":{
                "samplers":["penalties","top_k","top_p","temperature"],
                "max_tokens":120,
                "top_k":40,
                "top_p":0.95,
                "temperature":0.8,
                "speculative.type":"ngram_map_k",
                "speculative.n_max":48,
                "speculative.p_min":0.75
            },
            "next_token":[{"has_next_token":true,"n_remain":32,"n_decoded":16}]
        }]"#;
        let vals = parse_slot_metrics(body).unwrap();
        let slot = vals.slots.first().unwrap();

        assert_eq!(slot.id, Some(0));
        assert_eq!(slot.id_task, Some(2667));
        assert_eq!(slot.output_tokens, 16);
        assert_eq!(slot.output_remaining, 32);
        assert_eq!(slot.output_limit, 120);
        assert!(slot.speculative_enabled);
        assert_eq!(slot.speculative_type.as_deref(), Some("ngram_map_k"));
        assert_eq!(
            slot.sampler_stack,
            vec!["penalties", "top_k", "top_p", "temperature"]
        );
        assert!(slot.sampler_config.iter().any(|item| item.label == "temp"));
        assert!(
            slot.speculative_config
                .iter()
                .any(|item| item.label == "n_max" && item.value == "48")
        );
    }

    #[test]
    fn test_parse_prometheus_metrics_tokens_per_decode() {
        let body = "llamacpp:tokens_predicted_total 118903\nllamacpp:n_decode_total 36178\n";
        let vals = parse_prometheus_metrics(body);
        assert!((vals.tokens_per_decode - 3.286).abs() < 0.001);
    }

    #[test]
    fn test_parse_prometheus_metrics_tokens_per_decode_zero_decodes() {
        let body = "llamacpp:tokens_predicted_total 1000\n";
        let vals = parse_prometheus_metrics(body);
        assert_eq!(vals.tokens_per_decode, 0.0);
    }

    #[test]
    fn test_speculative_types_plural_field() {
        // Current llama.cpp with draft-mtp uses "speculative.types" (plural)
        let body = r#"[{
            "id":0,
            "n_ctx":4096,
            "is_processing":true,
            "speculative":true,
            "params":{
                "speculative.types":"none,draft-mtp,ngram-mod"
            },
            "next_token":[{"has_next_token":false,"n_remain":0,"n_decoded":10}]
        }]"#;
        let vals = parse_slot_metrics(body).unwrap();
        let slot = vals.slots.first().unwrap();
        assert!(slot.speculative_enabled);
        // "none" should be filtered out, leaving "draft-mtp,ngram-mod"
        assert_eq!(
            slot.speculative_type.as_deref(),
            Some("draft-mtp,ngram-mod")
        );
        assert!(
            slot.speculative_config
                .iter()
                .any(|item| item.label == "types" && item.value == "none,draft-mtp,ngram-mod")
        );
    }

    #[test]
    fn test_speculative_type_singular_takes_priority() {
        // Singular "speculative.type" should win over plural if both present
        let body = r#"[{
            "id":0,
            "n_ctx":4096,
            "is_processing":false,
            "speculative":true,
            "params":{
                "speculative.type":"ngram",
                "speculative.types":"none,ngram"
            },
            "next_token":[{"has_next_token":false,"n_remain":0,"n_decoded":5}]
        }]"#;
        let vals = parse_slot_metrics(body).unwrap();
        let slot = vals.slots.first().unwrap();
        assert_eq!(slot.speculative_type.as_deref(), Some("ngram"));
    }
}
