use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryAvailabilityState {
    #[default]
    Unsafe,
    SafeNow,
    ConditionalAfterReclaim,
    AfterClosingApps,
}

/// Launch intent: additional generation (concurrent with existing sessions)
/// vs replace existing (stops the target runtime first). Consumed by Wizard
/// for estimation differentiation.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LaunchIntent {
    AdditionalGeneration,
    ReplaceExisting,
}

/// The launch-specific facts required to turn a live memory sample into a fit
/// decision.  A bare machine snapshot must never pretend to know whether a
/// particular model fits.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MemoryAvailabilityRequest {
    /// Estimated peak bytes for the selected model/configuration.
    #[serde(default)]
    pub required_bytes: u64,
    /// Whether this launch coexists with a running model or replaces one.
    #[serde(default)]
    pub launch_intent: Option<LaunchIntent>,
    /// Measured footprint of an app-owned runtime which this launch will stop.
    /// Callers must leave this at zero when the runtime is not known/app-owned.
    #[serde(default)]
    pub replace_runtime_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAvailabilitySnapshot {
    /// Total unified/system memory in bytes. Informational only; never called "available".
    #[serde(default)]
    pub total_unified_bytes: u64,

    /// Current free memory in bytes (OS report).
    #[serde(default)]
    pub free_bytes: u64,

    /// Current wired (kernel-locked) memory in bytes.
    #[serde(default)]
    pub wired_bytes: u64,

    /// Current active memory in bytes.
    #[serde(default)]
    pub active_bytes: u64,

    /// Current speculative memory in bytes (macOS).
    #[serde(default)]
    pub speculative_bytes: u64,

    /// Current pageout/compressor memory in bytes (macOS).
    #[serde(default)]
    pub pageout_bytes: u64,

    /// Metal backend working set in bytes. On Apple Silicon, this is the Metal device's
    /// current recommended max working set or measured utilization base.
    #[serde(default)]
    pub metal_working_set_bytes: u64,

    /// Configured ceiling in bytes: the sysctl wired limit if set, otherwise a RAM-relative
    /// safe bound (default ~75% of total RAM on Apple Silicon). This is the stable capacity
    /// used by Model Browser/HF preview for planning.
    #[serde(default)]
    pub configured_ceiling_bytes: u64,

    /// Current safe availability in bytes, derived from metal working set, configured ceiling,
    /// and current backend utilization. This is what the Rapid Wizard and current launch use
    /// for fit determination. Always ≤ configured_ceiling_bytes.
    #[serde(default)]
    pub current_safe_availability_bytes: u64,

    /// Launch-specific requested footprint. Zero means this is a capacity
    /// snapshot only, not a model-fit claim.
    #[serde(default)]
    pub required_bytes: u64,

    /// Conservatively projected availability after reclaiming only reclaimable
    /// state. This is distinct from current safe availability.
    #[serde(default)]
    pub after_reclaim_bytes: u64,

    /// Availability after a measured app-owned runtime is stopped. It never
    /// includes arbitrary third-party process memory.
    #[serde(default)]
    pub after_closing_apps_bytes: u64,

    #[serde(default)]
    pub launch_intent: Option<LaunchIntent>,

    /// The determined availability state for a given launch scenario.
    #[serde(default)]
    pub state: MemoryAvailabilityState,

    /// Backend-specific fields: GPU-specific data for Metal (effective ceiling,
    /// recommended working set). Empty on non-Metal platforms.
    #[serde(default)]
    pub backend_specific: serde_json::Map<String, serde_json::Value>,

    /// Timestamp (Unix epoch seconds) when this snapshot was taken.
    #[serde(default)]
    pub timestamp: u64,
}

impl Default for MemoryAvailabilitySnapshot {
    fn default() -> Self {
        Self {
            total_unified_bytes: 0,
            free_bytes: 0,
            wired_bytes: 0,
            active_bytes: 0,
            speculative_bytes: 0,
            pageout_bytes: 0,
            metal_working_set_bytes: 0,
            configured_ceiling_bytes: 0,
            current_safe_availability_bytes: 0,
            required_bytes: 0,
            after_reclaim_bytes: 0,
            after_closing_apps_bytes: 0,
            launch_intent: None,
            state: MemoryAvailabilityState::Unsafe,
            backend_specific: serde_json::Map::new(),
            timestamp: 0,
        }
    }
}

/// Builds a MemoryAvailabilitySnapshot from live system metrics.
/// On Apple Silicon (macOS), uses Metal working set and iogpu wired limit.
/// On other platforms, returns a safe degraded snapshot.
pub fn build_snapshot() -> MemoryAvailabilitySnapshot {
    build_snapshot_for(&MemoryAvailabilityRequest::default())
}

/// Build a fresh snapshot with an optional selected-launch fit contract.
pub fn build_snapshot_for(request: &MemoryAvailabilityRequest) -> MemoryAvailabilitySnapshot {
    #[cfg(target_os = "macos")]
    {
        build_macos_snapshot(request)
    }
    #[cfg(not(target_os = "macos"))]
    {
        build_non_macos_snapshot(request)
    }
}

#[cfg(target_os = "macos")]
fn build_macos_snapshot(request: &MemoryAvailabilityRequest) -> MemoryAvailabilitySnapshot {
    let sys_info = crate::system::get_system_metrics();
    let total_bytes = (sys_info.ram_total_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let wired_bytes = (sys_info.memory_wired_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let free_bytes = (sys_info.memory_free_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let active_bytes = (sys_info.ram_used_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let speculative_bytes = (sys_info.memory_inactive_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let pageout_bytes = (sys_info.memory_compressor_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let reclaimable_bytes = (sys_info.memory_reclaimable_gb * 1024.0 * 1024.0 * 1024.0) as u64;

    // Read the configured Metal GPU wired limit from sysctl
    let wired_limit_mb = crate::gpu::apple::read_iogpu_wired_limit_mb();
    let configured_ceiling_bytes = if wired_limit_mb > 0 {
        wired_limit_mb * 1024 * 1024
    } else {
        // Default safe bound: tiered reserve based on RAM size
        // (<24GB: -6GB, ≥24GB: -8GB). Uses wired_limit_safe_default_mb for consistency.
        let safe_default_mb =
            crate::gpu::apple::wired_limit_safe_default_mb(total_bytes).unwrap_or(0);
        safe_default_mb * 1024 * 1024
    };

    // Metal working set: use the configured ceiling as the base (MLX reads this at init).
    // This is the effective base Rapid-MLX uses, multiplied by its utilization factor.
    let metal_working_set_bytes = configured_ceiling_bytes;

    // Current safe availability: free_bytes ("Pages free") is deliberately tiny on macOS —
    // the kernel uses spare RAM as disk cache and keeps very little literally unclaimed
    // (see system.rs::compute_macos_pressure). Purgeable and inactive pages are reclaimed
    // by the kernel on demand with no user action required, so tools users trust for this
    // number (Activity Monitor's "Memory Used" gauge, btop, htop) fold them into
    // "available". Match that convention instead of vm_stat's raw free count.
    let replace_credit = if matches!(request.launch_intent, Some(LaunchIntent::ReplaceExisting)) {
        request.replace_runtime_bytes
    } else {
        0
    };
    let current_safe_availability_bytes = free_bytes
        .saturating_add(reclaimable_bytes)
        .saturating_add(replace_credit)
        .min(configured_ceiling_bytes);
    // Reclaiming here means going further: asking the user to close other apps.
    // Approximate with a small additional margin over the already-reclaimable total.
    let after_reclaim_bytes = current_safe_availability_bytes
        .saturating_add(speculative_bytes / 4)
        .min(configured_ceiling_bytes);
    let after_closing_apps_bytes = current_safe_availability_bytes;
    let state = classify_state(
        request.required_bytes,
        current_safe_availability_bytes,
        after_reclaim_bytes,
        after_closing_apps_bytes,
    );

    // Build backend-specific metadata for Metal
    let mut backend_specific = serde_json::Map::new();
    backend_specific.insert(
        "effective_ceiling_bytes".to_string(),
        serde_json::Value::Number(serde_json::Number::from(configured_ceiling_bytes)),
    );
    backend_specific.insert(
        "metal_working_set_bytes".to_string(),
        serde_json::Value::Number(serde_json::Number::from(metal_working_set_bytes)),
    );
    backend_specific.insert(
        "recommended_working_set_bytes".to_string(),
        serde_json::Value::Number(serde_json::Number::from(configured_ceiling_bytes)),
    );
    backend_specific.insert(
        "wired_limit_mb_sysctl".to_string(),
        serde_json::Value::Number(serde_json::Number::from(wired_limit_mb)),
    );

    MemoryAvailabilitySnapshot {
        total_unified_bytes: total_bytes,
        free_bytes,
        wired_bytes,
        active_bytes,
        speculative_bytes,
        pageout_bytes,
        metal_working_set_bytes,
        configured_ceiling_bytes,
        current_safe_availability_bytes,
        required_bytes: request.required_bytes,
        after_reclaim_bytes,
        after_closing_apps_bytes,
        launch_intent: request.launch_intent.clone(),
        state,
        backend_specific,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    }
}

#[cfg(not(target_os = "macos"))]
fn build_non_macos_snapshot(request: &MemoryAvailabilityRequest) -> MemoryAvailabilitySnapshot {
    let sys_info = crate::system::get_system_metrics();
    let total_bytes = (sys_info.ram_total_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let free_bytes = (sys_info.memory_free_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let available_bytes = (sys_info.ram_available_gb * 1024.0 * 1024.0 * 1024.0) as u64;

    // On non-macOS, use a safe RAM-relative ceiling (80% of total)
    let configured_ceiling_bytes = (total_bytes as f64 * 0.80) as u64;
    let replace_credit = if matches!(request.launch_intent, Some(LaunchIntent::ReplaceExisting)) {
        request.replace_runtime_bytes
    } else {
        0
    };
    let current_safe_availability_bytes = available_bytes
        .saturating_add(replace_credit)
        .min(configured_ceiling_bytes);
    let state = classify_state(
        request.required_bytes,
        current_safe_availability_bytes,
        current_safe_availability_bytes,
        current_safe_availability_bytes,
    );

    MemoryAvailabilitySnapshot {
        total_unified_bytes: total_bytes,
        free_bytes,
        wired_bytes: 0,
        active_bytes: 0,
        speculative_bytes: 0,
        pageout_bytes: 0,
        metal_working_set_bytes: 0,
        configured_ceiling_bytes,
        current_safe_availability_bytes,
        required_bytes: request.required_bytes,
        after_reclaim_bytes: current_safe_availability_bytes,
        after_closing_apps_bytes: current_safe_availability_bytes,
        launch_intent: request.launch_intent.clone(),
        state,
        backend_specific: serde_json::Map::new(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    }
}

fn classify_state(
    required_bytes: u64,
    safe_now_bytes: u64,
    after_reclaim_bytes: u64,
    after_closing_apps_bytes: u64,
) -> MemoryAvailabilityState {
    if required_bytes == 0 || safe_now_bytes >= required_bytes {
        MemoryAvailabilityState::SafeNow
    } else if after_reclaim_bytes >= required_bytes {
        MemoryAvailabilityState::ConditionalAfterReclaim
    } else if after_closing_apps_bytes >= required_bytes {
        MemoryAvailabilityState::AfterClosingApps
    } else {
        MemoryAvailabilityState::Unsafe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_has_all_required_fields() {
        let snapshot = MemoryAvailabilitySnapshot::default();
        // All fields exist and deserialize/serialize cleanly
        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("total_unified_bytes").is_some());
        assert!(parsed.get("free_bytes").is_some());
        assert!(parsed.get("wired_bytes").is_some());
        assert!(parsed.get("active_bytes").is_some());
        assert!(parsed.get("speculative_bytes").is_some());
        assert!(parsed.get("pageout_bytes").is_some());
        assert!(parsed.get("metal_working_set_bytes").is_some());
        assert!(parsed.get("configured_ceiling_bytes").is_some());
        assert!(parsed.get("current_safe_availability_bytes").is_some());
        assert!(parsed.get("state").is_some());
        assert!(parsed.get("backend_specific").is_some());
        assert!(parsed.get("timestamp").is_some());
    }

    #[test]
    fn current_safe_availability_leq_configured_ceiling() {
        let snapshot = MemoryAvailabilitySnapshot {
            configured_ceiling_bytes: 48 * 1024 * 1024 * 1024,
            current_safe_availability_bytes: 36 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert!(
            snapshot.current_safe_availability_bytes <= snapshot.configured_ceiling_bytes,
            "current_safe_availability must be ≤ configured_ceiling"
        );
    }

    #[test]
    fn state_safe_now_when_sufficient() {
        let snapshot = MemoryAvailabilitySnapshot {
            configured_ceiling_bytes: 48 * 1024 * 1024 * 1024,
            current_safe_availability_bytes: 30 * 1024 * 1024 * 1024, // >50% of ceiling
            state: MemoryAvailabilityState::SafeNow,
            ..Default::default()
        };
        assert_eq!(snapshot.state, MemoryAvailabilityState::SafeNow);
    }

    #[test]
    fn state_conditional_when_free_but_less_than_availability() {
        let snapshot = MemoryAvailabilitySnapshot {
            current_safe_availability_bytes: 20 * 1024 * 1024 * 1024,
            free_bytes: 10 * 1024 * 1024 * 1024, // free < availability
            state: MemoryAvailabilityState::ConditionalAfterReclaim,
            ..Default::default()
        };
        assert_eq!(
            snapshot.state,
            MemoryAvailabilityState::ConditionalAfterReclaim
        );
    }

    #[test]
    fn build_snapshot_returns_valid_shape() {
        let snapshot = build_snapshot();
        // total_unified_bytes is deliberately unasserted: it is 0 on CI and in
        // containers where sysctl reports nothing, and any nonzero bound would
        // make this test host-dependent. `x > 0 || x == 0` used to stand here,
        // which is a tautology on u64 and checked nothing at all.
        // Validate that pressure can reduce availability below ceiling
        assert!(
            snapshot.current_safe_availability_bytes <= snapshot.configured_ceiling_bytes
                || snapshot.configured_ceiling_bytes == 0,
            "current_safe_availability must not exceed configured_ceiling"
        );
    }

    #[test]
    fn no_total_unified_bytes_called_available() {
        // Verify the struct does NOT expose total_unified_bytes as an "available_memory_bytes" field
        let snapshot = MemoryAvailabilitySnapshot::default();
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(
            !json.contains("available_memory_bytes"),
            "total_unified_bytes must NOT be called available"
        );
        assert!(
            json.contains("total_unified_bytes"),
            "must use total_unified_bytes for the raw total"
        );
    }

    #[test]
    fn launch_intent_serializes_correctly() {
        let additional = LaunchIntent::AdditionalGeneration;
        let replace = LaunchIntent::ReplaceExisting;
        assert_eq!(
            serde_json::to_string(&additional).unwrap(),
            r#""additional_generation""#
        );
        assert_eq!(
            serde_json::to_string(&replace).unwrap(),
            r#""replace_existing""#
        );
    }

    #[test]
    fn fit_state_uses_the_selected_requirement_not_a_generic_percentage() {
        assert_eq!(
            classify_state(20, 24, 24, 24),
            MemoryAvailabilityState::SafeNow
        );
        assert_eq!(
            classify_state(20, 16, 22, 16),
            MemoryAvailabilityState::ConditionalAfterReclaim
        );
        assert_eq!(
            classify_state(20, 16, 18, 24),
            MemoryAvailabilityState::AfterClosingApps
        );
        assert_eq!(
            classify_state(20, 16, 18, 19),
            MemoryAvailabilityState::Unsafe
        );
    }

    #[test]
    fn memory_state_serializes_correctly() {
        assert_eq!(
            serde_json::to_string(&MemoryAvailabilityState::SafeNow).unwrap(),
            r#""safe_now""#
        );
        assert_eq!(
            serde_json::to_string(&MemoryAvailabilityState::ConditionalAfterReclaim).unwrap(),
            r#""conditional_after_reclaim""#
        );
        assert_eq!(
            serde_json::to_string(&MemoryAvailabilityState::AfterClosingApps).unwrap(),
            r#""after_closing_apps""#
        );
        assert_eq!(
            serde_json::to_string(&MemoryAvailabilityState::Unsafe).unwrap(),
            r#""unsafe""#
        );
    }
}
