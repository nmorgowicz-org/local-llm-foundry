//! Phase 1b: shared llama.cpp launch-policy validation.
//!
//! Pure structural and product-policy checks only — no binary lookup. Every
//! consumer (save, direct spawn, doctor, calibration, import) calls into here
//! so the K/V policy cannot fork.

use super::ModelPreset;
use crate::inference::llama_cpp_capabilities::CapabilitySnapshot;

/// A single actionable validation issue with a stable code for UI repair links.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ValidationIssue {
    pub field: String,
    pub code: String,
    pub message: String,
    /// Optional machine-readable repair action name (e.g. "edit_ctk_ctv").
    pub repair: Option<String>,
}

/// Flags (canonical long forms and accepted aliases) that must never appear in
/// `extra_args` because they would silently override the typed K/V fields.
const KV_FLAGS: &[&str] = &[
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "-ctkd",
    "--spec-draft-type-k",
    "--cache-type-k-draft",
    "-ctvd",
    "--spec-draft-type-v",
    "--cache-type-v-draft",
];

/// Duplicate safety flags that may not be passed via `extra_args` because the
/// typed fields already own them.
const DUPLICATE_SAFETY_FLAGS: &[&str] = &[
    "-j",
    "--jinja",
    "--no-jinja",
    "-no-warmup",
    "--no-warmup",
    "--no-context-shift",
    "--no-cont-batching",
];

/// Return the KV-family flags found in `extra_args`.
pub fn extra_args_kv_flags(extra_args: &str) -> Vec<&'static str> {
    extra_args
        .split_whitespace()
        .filter_map(|arg| KV_FLAGS.iter().find(|known| **known == arg).copied())
        .collect()
}

/// Return duplicate safety flags found in `extra_args`.
pub fn extra_args_duplicate_safety_flags(extra_args: &str) -> Vec<&'static str> {
    extra_args
        .split_whitespace()
        .filter_map(|arg| {
            DUPLICATE_SAFETY_FLAGS
                .iter()
                .find(|known| **known == arg)
                .copied()
        })
        .collect()
}

/// Whether a K/V value pair is the mixed pair that requires a special build.
fn is_mixed_main_kv_pair(k: &str, v: &str) -> bool {
    (k.eq_ignore_ascii_case("q8_0") && v.eq_ignore_ascii_case("q4_0"))
        || (k.eq_ignore_ascii_case("q4_0") && v.eq_ignore_ascii_case("q8_0"))
}

/// Validate the main-K/V policy against a capability snapshot.
///
/// This is the only production path allowed to read
/// `snapshot.mixed_main_kv`. Fixture tests that set `mixed_main_kv.supported`
/// to true exercise the accept branch directly (hard-gate invariant).
pub fn validate_main_kv_policy(
    k: &str,
    v: &str,
    snapshot: &CapabilitySnapshot,
) -> Option<ValidationIssue> {
    // Empty strings mean "use llama-server default"; no policy applies.
    if k.trim().is_empty() || v.trim().is_empty() {
        return None;
    }
    if is_mixed_main_kv_pair(k, v) && !snapshot.mixed_main_kv.supported {
        return Some(ValidationIssue {
            field: "ctk/ctv".into(),
            code: "MIXED_MAIN_KV_UNSUPPORTED".into(),
            message: format!(
                "Mixed K/V pair {k}/{v} is not supported by this binary. {}",
                snapshot.mixed_main_kv.reason
            ),
            repair: Some("edit_ctk_ctv".into()),
        });
    }
    None
}

/// Return an issue if a K/V value is not in the known allowed set.
///
/// The allowed set comes from `llama-server --help` (see architecture §6).
/// Values outside the set are "unknown" and the preset stays non-launchable
/// until the backend validator accepts them (no destructive migration).
const KNOWN_KV_VALUES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

pub fn unknown_kv_value_issue(field: &str, value: &str) -> Option<ValidationIssue> {
    if value.trim().is_empty() {
        return None;
    }
    if KNOWN_KV_VALUES
        .iter()
        .any(|known| known.eq_ignore_ascii_case(value))
    {
        return None;
    }
    Some(ValidationIssue {
        field: field.to_string(),
        code: "UNKNOWN_KV_VALUE".into(),
        message: format!(
            "Value {value:?} is not in the known allowed set for {field}. Allowed: {KNOWN_KV_VALUES:?}"
        ),
        repair: Some("edit_ctk_ctv".into()),
    })
}

/// Validate a preset's llama.cpp launch policy structurally.
///
/// Returns all structural and policy issues; empty means launchable (subject
/// to a successful launch-time binary check).
pub fn validate_llama_launch_policy(
    preset: &ModelPreset,
    snapshot: Option<&CapabilitySnapshot>,
) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // 1) `extra_args` must not reintroduce K/V flags.
    let kv_conflicts = extra_args_kv_flags(&preset.extra_args);
    if !kv_conflicts.is_empty() {
        issues.push(ValidationIssue {
            field: "extra_args".into(),
            code: "EXTRA_ARGS_KV_OVERRIDE".into(),
            message: format!(
                "extra_args contains {kv_conflicts:?}; KV flags may not override the typed ctk/ctv fields."
            ),
            repair: Some("strip_kv_flags_from_extra_args".into()),
        });
    }

    // 2) `extra_args` must not duplicate typed safety flags.
    let dup_safety = extra_args_duplicate_safety_flags(&preset.extra_args);
    if !dup_safety.is_empty() {
        issues.push(ValidationIssue {
            field: "extra_args".into(),
            code: "EXTRA_ARGS_DUPLICATE_SAFETY_FLAG".into(),
            message: format!(
                "extra_args contains {dup_safety:?} which duplicates a typed launch flag."
            ),
            repair: Some("strip_duplicate_safety_flags".into()),
        });
    }

    // 3) Conflicting deprecated K/V pair: preserve but non-launchable.
    let (canon_k, canon_v) = (preset.ctk.trim(), preset.ctv.trim());
    if !canon_k.is_empty()
        && !canon_v.is_empty()
        && let (Some(dep_k), Some(dep_v)) = (&preset.cache_type_k, &preset.cache_type_v)
        && !dep_k.trim().is_empty()
        && !dep_v.trim().is_empty()
        && (dep_k.trim() != canon_k || dep_v.trim() != canon_v)
    {
        issues.push(ValidationIssue {
            field: "cache_type_k/cache_type_v".into(),
            code: "KV_FIELD_CONFLICT".into(),
            message: format!(
                "Conflicting K/V: canonical ctk={canon_k:?}/ctv={canon_v:?} vs. \
                 deprecated cache_type_k={:?}/cache_type_v={:?}. \
                 Set the deprecated fields to match ctk/ctv or clear them.",
                dep_k.trim(),
                dep_v.trim()
            ),
            repair: Some("resolve_kv_conflict".into()),
        });
    }

    // 4) Unknown KV values (if set).
    if let Some(issue) = unknown_kv_value_issue("ctk", canon_k) {
        issues.push(issue);
    }
    if let Some(issue) = unknown_kv_value_issue("ctv", canon_v) {
        issues.push(issue);
    }

    // 5) Mixed main-K/V policy — only if we have a snapshot to consult.
    if let Some(snap) = snapshot
        && let Some(issue) = validate_main_kv_policy(canon_k, canon_v, snap)
    {
        issues.push(issue);
    }

    // 6) n_cpu_moe policy (when GGUF metadata is authoritative).
    if let Some(v) = preset.n_cpu_moe
        && v < 0
    {
        issues.push(ValidationIssue {
            field: "n_cpu_moe".into(),
            code: "NCPU_MOE_NEGATIVE".into(),
            message: format!("n_cpu_moe must be non-negative (got {v})."),
            repair: Some("fix_n_cpu_moe".into()),
        });
    }
    if let Some(v) = preset.n_cpu_moe
        && v > 0
        && let Some(layers) = preset.block_count
        && layers > 0
        && v as u32 > layers
    {
        issues.push(ValidationIssue {
            field: "n_cpu_moe".into(),
            code: "NCPU_MOE_OVER_LAYER_COUNT".into(),
            message: format!("n_cpu_moe ({v}) exceeds the model layer count ({layers})."),
            repair: Some("fix_n_cpu_moe".into()),
        });
    }
    if let Some(v) = preset.n_cpu_moe
        && v > 0
        && preset.architecture_kind.as_deref() == Some("dense")
    {
        issues.push(ValidationIssue {
            field: "n_cpu_moe".into(),
            code: "NCPU_MOE_ON_DENSE_MODEL".into(),
            message: "n_cpu_moe is not applicable to dense (non-MoE) models; set to 0.".into(),
            repair: Some("fix_n_cpu_moe".into()),
        });
    }

    // 7) Explicit bundle performance options reject zero.
    //
    // Legacy flat presets with zero/null batch/ubatch retain runtime defaults
    // (omitted argv). A new explicit choice (some non-default marker would be
    // introduced in Phase 2+) rejects zero — enforced there.

    issues
}

/// v4 → v5 K/V migration:
///
/// 1. If canonical empty and deprecated populated → copy deprecated to canonical.
/// 2. If both populated and equal → retain canonical (no-op).
/// 3. If both populated and conflicting → preserve as-is; non-launchable via
///    validation (do not guess).
///
/// Returns `true` if anything was mutated.
pub fn migrate_kv_fields(preset: &mut ModelPreset) -> bool {
    let mut migrated = false;

    let dep_k_empty = preset
        .cache_type_k
        .as_deref()
        .is_some_and(|s| s.trim().is_empty());
    let dep_v_empty = preset
        .cache_type_v
        .as_deref()
        .is_some_and(|s| s.trim().is_empty());

    // Rule 1: canonical empty AND (deprecated populated for that side) → copy.
    if preset.ctk.trim().is_empty()
        && !dep_k_empty
        && let Some(k) = preset.cache_type_k.clone()
    {
        preset.ctk = k;
        migrated = true;
    }
    if preset.ctv.trim().is_empty()
        && !dep_v_empty
        && let Some(v) = preset.cache_type_v.clone()
    {
        preset.ctv = v;
        migrated = true;
    }

    // Rule 3: conflicting → leave as-is (validation surfaces the issue).

    migrated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::llama_cpp_capabilities::{
        CacheCapabilities, CapabilitySnapshot, CapabilitySnapshotSource, ConcurrencyCapabilities,
        ContextCapabilities, EndpointCapabilities, ExecutableIdentity, MixedMainKv,
        SpeculationCapabilities, StreamingCapabilities, TemplateCapabilities, ToolCapabilities,
    };

    fn base_snapshot() -> CapabilitySnapshot {
        CapabilitySnapshot {
            executable_identity: ExecutableIdentity {
                path: "llama-server".into(),
                file_hash: "hash".into(),
                file_mtime_unix: 0,
            },
            version_text: "test".into(),
            help_hash: "help".into(),
            serve_flags: vec![],
            cache: CacheCapabilities::default(),
            context: ContextCapabilities::default(),
            concurrency: ConcurrencyCapabilities::default(),
            endpoints: EndpointCapabilities::default(),
            streaming: StreamingCapabilities::default(),
            templates: TemplateCapabilities::default(),
            tools: ToolCapabilities::default(),
            speculation: SpeculationCapabilities::default(),
            mixed_main_kv: MixedMainKv::product_default_denied(),
            evidence_timestamp: 0,
            source: CapabilitySnapshotSource::ManualOverride,
        }
    }

    fn base_preset() -> ModelPreset {
        ModelPreset {
            ctk: "f16".into(),
            ctv: "f16".into(),
            ..Default::default()
        }
    }

    #[test]
    fn kv_migration_canonical_empty_deprecated_populated() {
        let mut p = base_preset();
        p.ctk.clear();
        p.ctv.clear();
        p.cache_type_k = Some("q8_0".into());
        p.cache_type_v = Some("q8_0".into());

        assert!(migrate_kv_fields(&mut p));
        assert_eq!(p.ctk, "q8_0");
        assert_eq!(p.ctv, "q8_0");
    }

    #[test]
    fn kv_migration_both_equal_is_noop() {
        let mut p = base_preset();
        p.ctk = "q8_0".into();
        p.ctv = "q8_0".into();
        p.cache_type_k = Some("q8_0".into());
        p.cache_type_v = Some("q8_0".into());

        assert!(!migrate_kv_fields(&mut p));
        assert_eq!(p.ctk, "q8_0");
    }

    #[test]
    fn kv_migration_conflict_is_preserved() {
        let mut p = base_preset();
        p.ctk = "f16".into();
        p.ctv = "f16".into();
        p.cache_type_k = Some("q8_0".into());
        p.cache_type_v = Some("q8_0".into());

        // No migration should happen (would be guessing).
        assert!(!migrate_kv_fields(&mut p));
        assert_eq!(p.ctk, "f16");

        // But validation must flag the conflict.
        let issues = validate_llama_launch_policy(&p, None);
        assert!(issues.iter().any(|i| i.code == "KV_FIELD_CONFLICT"));
    }

    #[test]
    fn mixed_kv_rejected_by_default_snapshot() {
        let mut p = base_preset();
        p.ctk = "q8_0".into();
        p.ctv = "q4_0".into();
        let snap = base_snapshot();

        let issues = validate_llama_launch_policy(&p, Some(&snap));
        assert!(issues.iter().any(|i| i.code == "MIXED_MAIN_KV_UNSUPPORTED"));
    }

    #[test]
    fn mixed_kv_accepted_when_supported_flag_set() {
        let mut p = base_preset();
        p.ctk = "q8_0".into();
        p.ctv = "q4_0".into();
        let mut snap = base_snapshot();
        // Fixture test directly sets the field — the only allowed accept path.
        snap.mixed_main_kv = MixedMainKv {
            supported: true,
            reason: "build manifest".into(),
            source: "build_manifest".into(),
        };

        let issues = validate_llama_launch_policy(&p, Some(&snap));
        assert!(!issues.iter().any(|i| i.code == "MIXED_MAIN_KV_UNSUPPORTED"));
    }

    #[test]
    fn extra_args_kv_rejected() {
        let mut p = base_preset();
        p.extra_args = "--other -ctk q8_0 -ctv q4_0".into();

        let issues = validate_llama_launch_policy(&p, None);
        assert!(issues.iter().any(|i| i.code == "EXTRA_ARGS_KV_OVERRIDE"));
    }

    #[test]
    fn unknown_kv_value_rejected() {
        let mut p = base_preset();
        p.ctk = "weird_type".into();

        let issues = validate_llama_launch_policy(&p, None);
        assert!(
            issues
                .iter()
                .any(|i| i.code == "UNKNOWN_KV_VALUE" && i.field == "ctk")
        );
    }
}
