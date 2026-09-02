# Phase 9 receipt (partial) — record exact launch memory evidence

Baseline commit: `b9e53c48ba9ad7cce2723fea91d7c392f4bb336d`

## Scope of this slice

This is a bounded, checkpointed slice of Phase 9, not the full phase. It
ships the launch-evidence vocabulary, its persistence store, and the live
`/resolve` read path. It does **not** ship either platform sampler (Metal,
CUDA/nvidia-smi), so no receipt can yet be produced by an actual session —
the store and matching logic are exercised only by unit fixtures so far.

## Summary

Added `src/calibration/launch_evidence.rs`, a receipt kind layered onto the
existing Calibration fingerprint/evidence vocabulary (architecture 12), not
a separate evidence store:

- `ArgvFieldClass` (MemoryRelevant/BehaviorOnly/Secret/Forbidden) and
  `classify_argv_field`, an exhaustive, fail-closed classification over every
  `ModelPreset` field. `extra_args` classifies `Forbidden` because it can
  smuggle unclassified argv.
- `manifest_digest(preset)` — `evidence-v1:<sha256>` over the sorted
  MemoryRelevant argv triples only; fails closed on any unclassified field.
- `LaunchEvidenceMethod` (WddmTotalDeviceDelta/CudaRocmProcessDelta/
  MetalUnifiedObservation/FitProbe/EstimatorOnly) with
  `is_direct_observation()`.
- `LaunchEvidenceFingerprint`, `LaunchSample`, `FitState`,
  `LaunchObservationReceipt`, and `build_launch_observation(preset,
  fingerprint_base, fit_state, sample)` — rejects `FitState::On` and any
  non-empty `extra_args`, since fit can silently shrink context/batch and
  must be pinned off for a launch to count as exact evidence.
- `EvidenceMatchClass` (Exact/Compatible/Related/Stale) and
  `classify_evidence_match`, extending Calibration's existing three-tier
  match precedent with a fourth, age-gated `Stale` tier
  (`EVIDENCE_FRESHNESS_WINDOW_MS` = 30 days).
- `current_fingerprint(config, resolved_preset, capabilities)` — the
  fingerprint builder shared by both save and lookup call sites, backed by a
  cached `HardwareFingerprint` (`sysinfo`, no process spawning) so it is
  cheap enough for the interactive `/resolve` HTTP path.
- `pub mod store` — atomic on-disk persistence (`save`, `list`,
  `best_match`) under the new `AppPaths::launch_evidence_dir()`
  (`<root>/calibrations/launch-evidence`), writing to a `.json.tmp` then
  `fs::rename` to avoid a partial file being mistaken for evidence.

Wired the read path into the live bundle resolver: `preset_bundles.rs`'s
`resolve_response` now takes `&AppConfig` and calls a new `lookup_evidence`
helper that builds the expected fingerprint and calls `store::best_match`,
returning the resolver's pre-existing but previously-unused
`EvidenceMatch { class, summary }` shape — no new frontend contract was
needed; Phase 8c's `renderEvidence`/`EVIDENCE_LABELS` already render it.
Evidence *lookup* (disk I/O) intentionally stays out of `resolve_preset()`
in `resolver.rs`, which is documented as pure — no artifact reads, binary
probes, or network I/O.

Extended `CalibrationMeasurement` with an optional `launch_evidence` field
so a future Calibration trial can carry an exact-observation receipt
alongside its benchmark samples.

## Files changed

- `src/calibration/launch_evidence.rs` (new)
- `src/calibration/mod.rs`
- `src/calibration/executor.rs`
- `src/paths.rs`
- `src/web/api/preset_bundles.rs`

## Verification

- `rtk cargo build` — passed
- `rtk cargo clippy -- -D warnings` — no issues
- `rtk cargo fmt -- --check` — no diff
- `rtk cargo test calibration::` — 57 passed, 1 ignored (11 suites)
- `rtk cargo test preset_bundles` — 10 passed (11 suites)

## Outstanding (not in this slice)

- Metal/unified-memory sampler: requires threading the resolved
  `ModelPreset` through `launch_local` → `start_backend`
  (`src/inference/launch.rs` / `src/llama/server.rs`), which currently only
  receive the lossy `ServerConfig` subset. Design sketched but not
  implemented: pre-start snapshot, unblocked readiness return, detached
  bounded poll of `memory_availability::build_snapshot()` for peak, persist
  via `store::save` only when `fit_enabled == Some(false)` and `extra_args`
  is empty.
- Windows/CUDA `nvidia-smi` sampler (total-device delta, `--parallel 1 -fit
  off`, idle stabilization, background-process noise detection) — not
  started. Scoped to CUDA only; no ROCm hardware available.
- Frontend evidence-details drawer action from the card.
- Full platform-labelled fixture corpus (Windows/WDDM, CUDA/ROCm, Metal,
  estimator-only; exact/compatible/related/stale; noisy/implausible deltas;
  incomplete sampling/server-start failure; one mutation fixture per
  memory-relevant argv field) — only the 11 targeted unit tests in
  `launch_evidence.rs` exist today.
- `rtk cargo test inference::`, `rtk cargo test presets::evidence::` (no
  such module currently exists), and the `core/preset-flow.spec.js`
  playwright run covering exact/compatible/related/stale evidence classes.
- Real-host qualification gates — require explicit user authorization
  before starting/stopping a model server on this machine or the remote
  Windows/CUDA machine; not requested yet in this slice.

## Hard gate

Not yet applicable: no sampler exists yet to produce a receipt from an
actual model launch, so the architecture's "no observation may be recorded
as exact evidence unless fit was pinned off" gate is exercised only by the
`build_launch_observation` unit tests, not by a real host.
