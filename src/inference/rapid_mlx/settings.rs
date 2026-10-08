//! Semantic setting catalog for Rapid-MLX configuration.
//!
//! Per D6: backend-owned, schema/descriptor-driven, covering capability, evidence,
//! default, help, validation, serialization, command mapping, and unsupported reason.
//!
//! This is NOT a form generator. It is the authoritative Rust definition that
//! the UI consumes via API. Every exposed setting must trace:
//! capability/evidence -> typed schema -> validation -> launch mapping -> save/restore.
//!
//! Wires in types from execution_policy.rs and workload_scenarios.rs.

use crate::inference::rapid_mlx::capabilities::{CapabilitySnapshot, FeatureQualification};
use crate::llama::vram_estimator::WorkloadScenario;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::marker::PhantomData;

/// Validation context for settings that depend on runtime state.
///
/// Both fields are populated by callers and read by none: `validate` takes the context as
/// `_context` and ignores it, so every rule in the catalog is currently context-free. That is
/// a catalog defect rather than a wiring one — the API passes a real snapshot in — and it is
/// left visible here instead of being papered over, because the fix is to decide which rules
/// are genuinely capability- or workload-dependent, not to invent rules so the fields are read.
/// Note that an unsupported setting is *not* one of those cases: `effective_policy` already
/// downgrades it gracefully, and erroring in `validate` instead would contradict that design.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ValidationContext<'a> {
    pub capabilities: Option<&'a CapabilitySnapshot>,
    pub workload_scenario: Option<WorkloadScenario>,
    pub _marker: PhantomData<&'a ()>,
}

impl<'a> Default for ValidationContext<'a> {
    fn default() -> Self {
        Self {
            capabilities: None,
            workload_scenario: None,
            _marker: PhantomData,
        }
    }
}

/// Validation error for a specific setting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub setting_id: &'static str,
    pub message: String,
    pub code: &'static str,
}

/// Effective policy explanation for a setting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectivePolicyExplanation {
    pub requested: serde_json::Value,
    pub effective: serde_json::Value,
    pub reason: Option<String>,
    pub reason_code: Option<&'static str>,
}

/// A single setting in the Rapid-MLX semantic catalog.
#[derive(Debug, Clone)]
pub enum RapidMlxSetting {
    KvCacheDtype,
    TurboquantMode,
    PrefixCachePolicy,
    HybridCacheEntries,
    PflashPolicy,
    MaxNumSeqs,
    MaxConcurrentRequests,
    PrefillBatchSize,
    CompletionBatchSize,
    ContextLength,
    DefaultReasoningEffort,
    ReasoningMode,
    MllmVision,
    Embeddings,
    GpuMemoryUtilization,
    SamplingMode,
}

impl RapidMlxSetting {
    /// Stable ID used for serialization and API.
    pub fn id(&self) -> &'static str {
        match self {
            Self::KvCacheDtype => "kv_cache_dtype",
            Self::TurboquantMode => "turboquant_mode",
            Self::PrefixCachePolicy => "prefix_cache_policy",
            Self::HybridCacheEntries => "hybrid_cache_entries",
            Self::PflashPolicy => "pflash_policy",
            Self::MaxNumSeqs => "max_num_seqs",
            Self::MaxConcurrentRequests => "max_concurrent_requests",
            Self::PrefillBatchSize => "prefill_batch_size",
            Self::CompletionBatchSize => "completion_batch_size",
            Self::ContextLength => "context_length",
            Self::DefaultReasoningEffort => "default_reasoning_effort",
            Self::ReasoningMode => "reasoning_mode",
            Self::MllmVision => "mllm_vision",
            Self::Embeddings => "embeddings",
            Self::GpuMemoryUtilization => "gpu_memory_utilization",
            Self::SamplingMode => "sampling_mode",
        }
    }

    /// Whether this setting is supported by the given capability snapshot.
    pub fn capability(&self, snapshot: &CapabilitySnapshot) -> bool {
        let has_flag = |flag: &str| snapshot.serve_flags.iter().any(|f| f == flag);

        match self {
            Self::KvCacheDtype => has_flag("--kv-cache-dtype"),
            Self::TurboquantMode => has_flag("--kv-cache-turboquant"),
            Self::PrefixCachePolicy => has_flag("--max-cache-blocks"),
            Self::HybridCacheEntries => has_flag("--hybrid-cache-entries"),
            Self::PflashPolicy => has_flag("--pflash"),
            Self::MaxNumSeqs => has_flag("--max-num-seqs"),
            Self::MaxConcurrentRequests => has_flag("--max-concurrent-requests"),
            Self::PrefillBatchSize => has_flag("--prefill-batch-size"),
            Self::CompletionBatchSize => has_flag("--completion-batch-size"),
            Self::ContextLength => has_flag("--context-length"),
            Self::DefaultReasoningEffort => has_flag("--default-reasoning-effort"),
            Self::ReasoningMode => has_flag("--reasoning"),
            Self::MllmVision => {
                matches!(
                    snapshot.qualified_features.vision,
                    FeatureQualification::Available
                )
            }
            Self::Embeddings => {
                matches!(
                    snapshot.qualified_features.embeddings,
                    FeatureQualification::Available
                )
            }
            Self::GpuMemoryUtilization => has_flag("--gpu-memory-utilization"),
            Self::SamplingMode => true, // Always available via --default-* flags or request-level
        }
    }

    /// Default value for this setting (JSON-serializable).
    pub fn default_value(&self) -> serde_json::Value {
        match self {
            Self::KvCacheDtype => serde_json::json!({"effective": "int4"}),
            Self::TurboquantMode => serde_json::json!("none"),
            Self::PrefixCachePolicy => serde_json::json!("auto"),
            Self::HybridCacheEntries => serde_json::json!(0),
            // Must track `default_pflash_policy` in the config, not rapid-mlx's own default. This
            // catalog describes what llama-monitor will actually launch with, and llama-monitor
            // launches `--pflash off` on every config.
            Self::PflashPolicy => serde_json::json!("off"),
            Self::MaxNumSeqs => serde_json::json!(4),
            Self::MaxConcurrentRequests => serde_json::json!(16),
            Self::PrefillBatchSize => serde_json::json!(null),
            Self::CompletionBatchSize => serde_json::json!(null),
            // Unset lets the runtime size the window from available memory.
            Self::ContextLength => serde_json::json!(null),
            // Unset sends no flag, so the model template decides its own thinking depth.
            Self::DefaultReasoningEffort => serde_json::json!(null),
            Self::ReasoningMode => serde_json::json!("auto"),
            Self::MllmVision => serde_json::json!("auto"),
            Self::Embeddings => serde_json::json!("auto"),
            Self::GpuMemoryUtilization => serde_json::json!(0.9),
            Self::SamplingMode => serde_json::json!("auto"),
        }
    }

    /// Validate a value for this setting.
    pub fn validate<'a>(
        &self,
        value: &serde_json::Value,
        _context: &ValidationContext<'a>,
    ) -> Result<(), ValidationError> {
        let id = self.id();
        let invalid = |msg: String, code: &'static str| {
            Err(ValidationError {
                setting_id: id,
                message: msg,
                code,
            })
        };

        match self {
            Self::KvCacheDtype => {
                let effective = value
                    .get("effective")
                    .and_then(|v| v.as_str())
                    .unwrap_or("int4");
                match effective {
                    "int4" | "int8" | "bf16" => Ok(()),
                    _ => invalid(
                        format!(
                            "kv_cache_dtype.effective must be one of [int4, int8, bf16], got '{effective}'"
                        ),
                        "invalid_kv_dtype",
                    ),
                }
            }
            Self::TurboquantMode => {
                let mode = value.as_str().unwrap_or("none");
                match mode {
                    "none" | "k8v4" | "v4" => Ok(()),
                    _ => invalid(
                        format!("turboquant_mode must be one of [none, k8v4, v4], got '{mode}'"),
                        "invalid_turboquant_mode",
                    ),
                }
            }
            Self::HybridCacheEntries => {
                if let Some(n) = value.as_u64()
                    && n > 65536
                {
                    return invalid(
                        "hybrid_cache_entries exceeds maximum 65536".into(),
                        "hybrid_cache_entries_too_high",
                    );
                }
                Ok(())
            }
            Self::DefaultReasoningEffort => match value.as_str() {
                None | Some("") => Ok(()),
                Some(effort) if super::command::REASONING_EFFORTS.contains(&effort) => Ok(()),
                Some(effort) => invalid(
                    format!(
                        "default_reasoning_effort must be one of [{}], got '{effort}'",
                        super::command::REASONING_EFFORTS.join(", ")
                    ),
                    "invalid_reasoning_effort",
                ),
            },
            Self::ContextLength => {
                if let Some(n) = value.as_u64()
                    && !(512..=4_194_304).contains(&n)
                {
                    return invalid(
                        "context_length must be in range [512, 4194304]".into(),
                        "context_length_out_of_range",
                    );
                }
                Ok(())
            }
            Self::MaxNumSeqs => {
                if let Some(n) = value.as_u64()
                    && (n == 0 || n > 256)
                {
                    return invalid(
                        "max_num_seqs must be in range [1, 256]".into(),
                        "max_num_seqs_out_of_range",
                    );
                }
                Ok(())
            }
            Self::MaxConcurrentRequests => {
                if let Some(n) = value.as_u64()
                    && (n == 0 || n > 1024)
                {
                    return invalid(
                        "max_concurrent_requests must be in range [1, 1024]".into(),
                        "max_concurrent_requests_out_of_range",
                    );
                }
                Ok(())
            }
            Self::GpuMemoryUtilization => {
                if let Some(f) = value.as_f64()
                    && !(0.5..=1.0).contains(&f)
                {
                    return invalid(
                        "gpu_memory_utilization must be in range [0.5, 1.0]".into(),
                        "gpu_memory_utilization_out_of_range",
                    );
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Convert a setting value to CLI arguments.
    ///
    /// A null value means the setting is not in play — most often because `effective_policy`
    /// resolved it away on a runtime that does not support the flag. Returning early is what
    /// makes that resolution mean anything: several arms below (`KvCacheDtype` among them)
    /// fall back to a hardcoded default when the value has no recognised shape, so without
    /// this guard an unsupported setting would emit the very flag it was nulled out to avoid.
    pub fn to_cli_args(&self, value: &serde_json::Value) -> Vec<String> {
        if value.is_null() {
            return Vec::new();
        }
        let mut args = Vec::new();

        match self {
            Self::KvCacheDtype => {
                let effective = value
                    .get("effective")
                    .and_then(|v| v.as_str())
                    .unwrap_or("int4");
                args.push("--kv-cache-dtype".into());
                args.push(effective.into());
            }
            Self::TurboquantMode => {
                let mode = value.as_str().unwrap_or("none");
                if mode != "none" {
                    args.push("--kv-cache-turboquant".into());
                    args.push(mode.into());
                }
            }
            Self::HybridCacheEntries => {
                if let Some(n) = value.as_u64()
                    && n > 0
                {
                    args.push("--hybrid-cache-entries".into());
                    args.push(n.to_string());
                }
            }
            Self::MaxNumSeqs => {
                if let Some(n) = value.as_u64()
                    && n != 4
                {
                    args.push("--max-num-seqs".into());
                    args.push(n.to_string());
                }
            }
            Self::MaxConcurrentRequests => {
                if let Some(n) = value.as_u64()
                    && n != 16
                {
                    args.push("--max-concurrent-requests".into());
                    args.push(n.to_string());
                }
            }
            Self::PrefillBatchSize => {
                if let Some(n) = value.as_u64() {
                    args.push("--prefill-batch-size".into());
                    args.push(n.to_string());
                }
            }
            Self::CompletionBatchSize => {
                if let Some(n) = value.as_u64() {
                    args.push("--completion-batch-size".into());
                    args.push(n.to_string());
                }
            }
            Self::ContextLength => {
                if let Some(n) = value.as_u64() {
                    args.push("--context-length".into());
                    args.push(n.to_string());
                }
            }
            Self::DefaultReasoningEffort => {
                if let Some(effort) = value.as_str().filter(|e| !e.is_empty()) {
                    args.push("--default-reasoning-effort".into());
                    args.push(effort.to_string());
                }
            }
            Self::GpuMemoryUtilization => {
                if let Some(f) = value.as_f64()
                    && f != 0.9
                {
                    args.push("--gpu-memory-utilization".into());
                    args.push(format!("{f}"));
                }
            }
            _ => {}
        }

        args
    }

    /// Compute the effective value given a requested value and capabilities.
    pub fn effective_policy(
        &self,
        requested: &serde_json::Value,
        capabilities: &CapabilitySnapshot,
    ) -> serde_json::Value {
        match self {
            Self::KvCacheDtype => {
                let requested_eff = requested
                    .get("effective")
                    .and_then(|v| v.as_str())
                    .unwrap_or("int4");

                if !self.capability(capabilities) {
                    return serde_json::json!(null);
                }

                if requested_eff == "bf16" {
                    serde_json::json!({"effective": "int8", "reason": "model_safe_downgrade"})
                } else {
                    requested.clone()
                }
            }
            Self::TurboquantMode => {
                if !self.capability(capabilities) {
                    return serde_json::json!("none");
                }

                match requested.as_str().unwrap_or("none") {
                    "k8v4" | "v4" => {
                        serde_json::json!("none")
                    }
                    _ => requested.clone(),
                }
            }
            _ => {
                if !self.capability(capabilities) {
                    serde_json::json!(null)
                } else {
                    requested.clone()
                }
            }
        }
    }

    /// Unsupported reason when capability returns false.
    pub fn unsupported_reason(&self, snapshot: &CapabilitySnapshot) -> Option<String> {
        if self.capability(snapshot) {
            return None;
        }

        Some(match self {
            Self::KvCacheDtype => "Current runtime does not support --kv-cache-dtype".into(),
            Self::TurboquantMode => "Current runtime does not support --kv-cache-turboquant".into(),
            Self::ReasoningMode => "Current runtime does not support --reasoning".into(),
            Self::MllmVision => "Vision extra (mlx-vlm) is not installed or not qualified".into(),
            Self::Embeddings => {
                "Embeddings extra (mlx-embed) is not installed or not qualified".into()
            }
            Self::MaxNumSeqs => "Current runtime does not support --max-num-seqs".into(),
            Self::MaxConcurrentRequests => {
                "Current runtime does not support --max-concurrent-requests".into()
            }
            _ => "This setting is not supported by the current runtime".into(),
        })
    }
}

/// Complete catalog of all Rapid-MLX settings.
pub fn all_settings() -> &'static [RapidMlxSetting] {
    &[
        RapidMlxSetting::KvCacheDtype,
        RapidMlxSetting::TurboquantMode,
        RapidMlxSetting::PrefixCachePolicy,
        RapidMlxSetting::HybridCacheEntries,
        RapidMlxSetting::PflashPolicy,
        RapidMlxSetting::MaxNumSeqs,
        RapidMlxSetting::MaxConcurrentRequests,
        RapidMlxSetting::PrefillBatchSize,
        RapidMlxSetting::CompletionBatchSize,
        RapidMlxSetting::ContextLength,
        RapidMlxSetting::DefaultReasoningEffort,
        RapidMlxSetting::ReasoningMode,
        RapidMlxSetting::MllmVision,
        RapidMlxSetting::Embeddings,
        RapidMlxSetting::GpuMemoryUtilization,
        RapidMlxSetting::SamplingMode,
    ]
}

/// How one participant in a mutual-exclusion rule is recognized.
#[derive(Debug, Clone)]
pub enum ExclusionMatch {
    /// Participates only at these exact values.
    OneOf(&'static [&'static str]),
    // A `Present` variant (participates whenever set at all, for numeric settings with no "on"
    // value) lived here for the speculative_policy/max_num_seqs rule. That rule was built on
    // `--speculative`, a flag no rapid-mlx release has, so both went. Re-add it when a real
    // numeric participant needs it rather than keeping an unused mechanism warm.
}

impl ExclusionMatch {
    fn matches(&self, value: &serde_json::Value) -> bool {
        match self {
            Self::OneOf(allowed) => value.as_str().is_some_and(|found| allowed.contains(&found)),
        }
    }
}

/// Mutual exclusion rules for settings.
///
/// Each participant carries the values that make it part of the conflict. The rule fires
/// only when *every* participant matches — a rule that fires on any one of them is not an
/// exclusion, it is a ban on that single setting.
#[derive(Debug, Clone)]
pub struct MutualExclusionRule {
    pub participants: &'static [(&'static str, ExclusionMatch)],
    pub error: &'static str,
}

impl MutualExclusionRule {
    /// The setting ids this rule concerns, for API responses and error attribution.
    pub fn setting_ids(&self) -> Vec<&'static str> {
        self.participants.iter().map(|(id, _)| *id).collect()
    }
}

/// All mutual exclusion rules.
/// Validates incompatible combinations before launch per Phase 7 builder brief item 3.
pub fn mutual_exclusion_rules() -> &'static [MutualExclusionRule] {
    &[
        MutualExclusionRule {
            participants: &[
                ("reasoning_mode", ExclusionMatch::OneOf(&["on"])),
                ("sampling_mode", ExclusionMatch::OneOf(&["model_default"])),
            ],
            error: "reasoning_mode=on and sampling_mode=model_default are mutually exclusive",
        },
        MutualExclusionRule {
            participants: &[
                ("pflash_policy", ExclusionMatch::OneOf(&["on"])),
                // "auto" and "none" do not request TurboQuant, so they cannot collide.
                ("turboquant_mode", ExclusionMatch::OneOf(&["v4", "k8v4"])),
            ],
            error: "pflash_policy bypasses TurboQuant; these policies cannot be combined",
        },
    ]
}

/// Check mutual exclusions for a set of settings.
pub fn check_mutual_exclusions(
    settings: &BTreeMap<&'static str, serde_json::Value>,
) -> Result<(), ValidationError> {
    for rule in mutual_exclusion_rules() {
        // Every participant must be present and at a conflicting value. The previous
        // any-match reading meant `reasoning_mode=on` alone reported a conflict with a
        // `sampling_mode` the caller had not even submitted.
        let triggered = rule.participants.iter().all(|(setting_id, how)| {
            settings
                .get(setting_id)
                .is_some_and(|value| how.matches(value))
        });
        if triggered {
            return Err(ValidationError {
                setting_id: rule.participants[0].0,
                message: rule.error.into(),
                code: "mutual_exclusion",
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(
        pairs: &[(&'static str, serde_json::Value)],
    ) -> BTreeMap<&'static str, serde_json::Value> {
        pairs.iter().cloned().collect()
    }

    /// Every `--flag` literal this module and the command builder test against must be a flag
    /// the runtime actually has.
    ///
    /// `ServeCapabilities::from_help` derives capabilities by parsing `rapid-mlx serve --help`,
    /// so a flag that is not in that help text can never be detected. A field gated on an
    /// invented flag is therefore permanently unavailable — and worse, `require()` on such a
    /// flag returns `Err`, which aborts the *entire* command build and blanks the preview for
    /// every unrelated setting too.
    ///
    /// This shipped: `--response-cache`, `--disk-checkpoint`, `--batching-policy`,
    /// `--concurrency-policy`, `--speculative`, `--ui`, `--no-ui`, `--path`, `--ui-config` and
    /// `--ui-config-file` were all wired up against a runtime that has none of them. The unit
    /// tests passed because they built a synthetic capability set that declared the invented
    /// flags, so the tests confirmed the mistake instead of catching it. Hence checking source
    /// literals against a captured inventory rather than against another hand-written list.
    #[test]
    fn serve_flag_literals_exist_in_the_real_runtime() {
        let inventory: std::collections::BTreeSet<&str> = include_str!("testdata/serve-flags.txt")
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("--"))
            .collect();
        assert!(
            inventory.len() > 50,
            "flag inventory looks truncated: {} entries",
            inventory.len()
        );

        // Only probe and emission sites matter. A flag literal inside `assert!(!args...)` is a
        // test asserting the flag is *absent*, which is legitimate and must not trip the guard.
        let is_live_site = |line: &str| {
            let line = line.split("//").next().unwrap_or("");
            (line.contains("has_flag(")
                || line.contains(".require(")
                || line.contains(".contains(")
                || line.contains("args.push("))
                && !line.contains("assert!(!")
        };

        let mut offenders: Vec<(&str, &str)> = Vec::new();
        for (file, src) in [
            ("settings.rs", include_str!("settings.rs")),
            ("command.rs", include_str!("command.rs")),
        ] {
            // All literals on the line, not just the first: `has_flag("--ui") ||
            // has_flag("--no-ui")` hides a second flag behind the first.
            for (idx, _) in src
                .lines()
                .filter(|l| is_live_site(l))
                .flat_map(|l| l.match_indices("\"--").map(move |(i, _)| (&l[i + 1..], ())))
            {
                let Some(end) = idx.find('"') else { continue };
                let flag = &idx[..end];
                // Skip anything that is not a plain long flag (format strings, joined lists).
                if flag.len() < 3
                    || !flag[2..]
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
                {
                    continue;
                }
                if !inventory.contains(flag) {
                    offenders.push((file, flag));
                }
            }
        }
        offenders.sort_unstable();
        offenders.dedup();
        assert!(
            offenders.is_empty(),
            "flags referenced in source but absent from `rapid-mlx serve --help`: {offenders:?}\n\
             Either the flag was invented, or the runtime changed and \
             testdata/serve-flags.txt needs regenerating."
        );
    }

    #[test]
    fn one_setting_alone_is_not_a_mutual_exclusion() {
        // `reasoning_mode=on` is an ordinary choice. It used to report a conflict with a
        // `sampling_mode` the caller had not submitted, because the rule fired when any one
        // participant matched — which made the settings-validate endpoint answer
        // `valid: false` for a perfectly normal configuration.
        for lone in [
            ("reasoning_mode", serde_json::json!("on")),
            ("sampling_mode", serde_json::json!("model_default")),
            ("pflash_policy", serde_json::json!("on")),
            ("turboquant_mode", serde_json::json!("k8v4")),
        ] {
            let id = lone.0;
            assert!(
                check_mutual_exclusions(&values(&[lone])).is_ok(),
                "{id} alone should not trigger any exclusion"
            );
        }
    }

    #[test]
    fn the_full_conflicting_combination_still_fails() {
        let error = check_mutual_exclusions(&values(&[
            ("reasoning_mode", serde_json::json!("on")),
            ("sampling_mode", serde_json::json!("model_default")),
        ]))
        .expect_err("both participants at conflicting values must fail");
        assert_eq!(error.code, "mutual_exclusion");
        assert_eq!(error.setting_id, "reasoning_mode");
    }

    #[test]
    fn a_participant_at_a_harmless_value_does_not_conflict() {
        // TurboQuant `auto` and `none` do not request TurboQuant, so PFlash cannot collide
        // with them; only an explicit v4/k8v4 is a real conflict.
        for mode in ["auto", "none"] {
            assert!(
                check_mutual_exclusions(&values(&[
                    ("pflash_policy", serde_json::json!("on")),
                    ("turboquant_mode", serde_json::json!(mode)),
                ]))
                .is_ok(),
                "pflash + turboquant_mode={mode} is not a conflict"
            );
        }
        assert!(
            check_mutual_exclusions(&values(&[
                ("pflash_policy", serde_json::json!("on")),
                ("turboquant_mode", serde_json::json!("k8v4")),
            ]))
            .is_err(),
            "pflash + an explicit TurboQuant mode is a conflict"
        );
    }

    fn test_snapshot() -> CapabilitySnapshot {
        CapabilitySnapshot {
            executable_identity: Default::default(),
            rapid_mlx_version: "0.10.10".into(),
            help_hash: "hash".into(),
            serve_flags: vec![
                "--kv-cache-dtype".into(),
                "--kv-cache-turboquant".into(),
                "--max-cache-blocks".into(),
                "--reasoning".into(),
                "--max-num-seqs".into(),
                "--max-concurrent-requests".into(),
                "--prefill-batch-size".into(),
                "--completion-batch-size".into(),
                "--gpu-memory-utilization".into(),
            ],
            package_versions: vec![],
            installed_extras: Default::default(),
            qualified_features: Default::default(),
            mtp_concurrency: Default::default(),
            sampling_defaults: Default::default(),
            sampling_cascade: Default::default(),
            evidence_timestamp: 0,
            source: Default::default(),
            measured_spec_decode: None,
            superseded_spec_decode: None,
        }
    }

    #[test]
    fn all_settings_have_unique_ids() {
        let ids: std::collections::HashSet<_> = all_settings().iter().map(|s| s.id()).collect();
        assert_eq!(ids.len(), all_settings().len());
    }

    #[test]
    fn kv_cache_dtype_capability_works() {
        let snapshot = test_snapshot();
        let setting = RapidMlxSetting::KvCacheDtype;
        assert!(setting.capability(&snapshot));
        assert!(setting.unsupported_reason(&snapshot).is_none());
    }

    #[test]
    fn turboquant_capability_works() {
        let snapshot = test_snapshot();
        let setting = RapidMlxSetting::TurboquantMode;
        assert!(setting.capability(&snapshot));
    }

    #[test]
    fn unsupported_setting_returns_reason() {
        let mut snapshot = CapabilitySnapshot::default();
        snapshot.serve_flags.push("--host".into());
        snapshot.serve_flags.push("--port".into());

        let setting = RapidMlxSetting::KvCacheDtype;
        assert!(!setting.capability(&snapshot));
        assert!(setting.unsupported_reason(&snapshot).is_some());
    }

    #[test]
    fn kv_cache_dtype_validation_accepts_valid_values() {
        let setting = RapidMlxSetting::KvCacheDtype;
        let ctx = ValidationContext::default();

        assert!(
            setting
                .validate(&serde_json::json!({"effective": "int4"}), &ctx)
                .is_ok()
        );
        assert!(
            setting
                .validate(&serde_json::json!({"effective": "int8"}), &ctx)
                .is_ok()
        );
        assert!(
            setting
                .validate(&serde_json::json!({"effective": "bf16"}), &ctx)
                .is_ok()
        );
    }

    #[test]
    fn kv_cache_dtype_validation_rejects_invalid_value() {
        let setting = RapidMlxSetting::KvCacheDtype;
        let ctx = ValidationContext::default();

        let err = setting
            .validate(&serde_json::json!({"effective": "fp16"}), &ctx)
            .unwrap_err();
        assert_eq!(err.code, "invalid_kv_dtype");
    }

    #[test]
    fn turboquant_mode_validation() {
        let setting = RapidMlxSetting::TurboquantMode;
        let ctx = ValidationContext::default();

        assert!(setting.validate(&serde_json::json!("none"), &ctx).is_ok());
        assert!(setting.validate(&serde_json::json!("k8v4"), &ctx).is_ok());
        assert!(setting.validate(&serde_json::json!("v4"), &ctx).is_ok());

        let err = setting
            .validate(&serde_json::json!("invalid"), &ctx)
            .unwrap_err();
        assert_eq!(err.code, "invalid_turboquant_mode");
    }

    #[test]
    fn hybrid_cache_entries_validation() {
        let setting = RapidMlxSetting::HybridCacheEntries;
        let ctx = ValidationContext::default();

        assert!(setting.validate(&serde_json::json!(100), &ctx).is_ok());
        assert!(setting.validate(&serde_json::json!(65536), &ctx).is_ok());

        let err = setting
            .validate(&serde_json::json!(70000), &ctx)
            .unwrap_err();
        assert_eq!(err.code, "hybrid_cache_entries_too_high");
    }

    #[test]
    fn max_num_seqs_validation() {
        let setting = RapidMlxSetting::MaxNumSeqs;
        let ctx = ValidationContext::default();

        assert!(setting.validate(&serde_json::json!(1), &ctx).is_ok());
        assert!(setting.validate(&serde_json::json!(256), &ctx).is_ok());

        let err = setting.validate(&serde_json::json!(0), &ctx).unwrap_err();
        assert_eq!(err.code, "max_num_seqs_out_of_range");

        let err = setting.validate(&serde_json::json!(300), &ctx).unwrap_err();
        assert_eq!(err.code, "max_num_seqs_out_of_range");
    }

    #[test]
    fn gpu_memory_utilization_validation() {
        let setting = RapidMlxSetting::GpuMemoryUtilization;
        let ctx = ValidationContext::default();

        assert!(setting.validate(&serde_json::json!(0.5), &ctx).is_ok());
        assert!(setting.validate(&serde_json::json!(0.9), &ctx).is_ok());
        assert!(setting.validate(&serde_json::json!(1.0), &ctx).is_ok());

        let err = setting.validate(&serde_json::json!(0.4), &ctx).unwrap_err();
        assert_eq!(err.code, "gpu_memory_utilization_out_of_range");
    }

    #[test]
    fn to_cli_args_kvcache_dtype() {
        let setting = RapidMlxSetting::KvCacheDtype;
        let args = setting.to_cli_args(&serde_json::json!({"effective": "int8"}));
        assert_eq!(args, vec!["--kv-cache-dtype", "int8"]);
    }

    #[test]
    fn to_cli_args_turboquant() {
        let setting = RapidMlxSetting::TurboquantMode;
        let args_none = setting.to_cli_args(&serde_json::json!("none"));
        assert!(args_none.is_empty());

        let args_k8v4 = setting.to_cli_args(&serde_json::json!("k8v4"));
        assert_eq!(args_k8v4, vec!["--kv-cache-turboquant", "k8v4"]);
    }

    #[test]
    fn to_cli_args_max_num_seqs_default_omitted() {
        let setting = RapidMlxSetting::MaxNumSeqs;
        let args_default = setting.to_cli_args(&serde_json::json!(4));
        assert!(args_default.is_empty());

        let args_explicit = setting.to_cli_args(&serde_json::json!(8));
        assert_eq!(args_explicit, vec!["--max-num-seqs", "8"]);
    }

    #[test]
    fn effective_policy_kvcache_dtype() {
        let snapshot = test_snapshot();
        let setting = RapidMlxSetting::KvCacheDtype;

        let effective =
            setting.effective_policy(&serde_json::json!({"effective": "int4"}), &snapshot);
        assert_eq!(effective, serde_json::json!({"effective": "int4"}));
    }

    #[test]
    fn effective_policy_turboquant_downgrade() {
        let snapshot = test_snapshot();
        let setting = RapidMlxSetting::TurboquantMode;

        let effective = setting.effective_policy(&serde_json::json!("k8v4"), &snapshot);
        assert_eq!(effective, serde_json::json!("none"));
    }

    #[test]
    fn all_settings_have_default_value() {
        for setting in all_settings() {
            let default = setting.default_value();
            assert!(
                !default.is_null()
                    || matches!(
                        setting,
                        RapidMlxSetting::PrefillBatchSize
                            | RapidMlxSetting::CompletionBatchSize
                            | RapidMlxSetting::ContextLength
                            | RapidMlxSetting::DefaultReasoningEffort
                    )
            );
        }
    }
}
