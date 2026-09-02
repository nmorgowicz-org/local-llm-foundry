# Phase 9 receipt (partial) — record exact launch memory evidence

Baseline commit: `b9e53c48ba9ad7cce2723fea91d7c392f4bb336d`

## Slice 2 addendum (this checkpoint)

Baseline for this slice: `b45daeb7d6b7714d764afec5a29e95b9364d53fb` (slice 1,
below).

This slice ships the Metal/unified-memory sampler, a design (not
implementation) for the Windows/CUDA `nvidia-smi` sampler, and most of the
Phase 9 fixture corpus. It still does **not** ship the frontend
evidence-details drawer action, the Windows/CUDA sampler implementation
itself (no reachable hardware — see below), or any real-host qualification
run.

### Metal sampler (`metal_sampler` in `launch_evidence.rs`)

Bounded post-readiness sampler wired into the session-spawn lifecycle:

- **Signal pivot.** `MemoryAvailabilitySnapshot::metal_working_set_bytes`
  (`memory_availability::build_snapshot()`) turned out to be a static
  configured ceiling (the `iogpu` wired limit or a RAM-relative default),
  not a live per-process measurement — computed identically regardless of
  what is running. Pivoted to `wired_bytes`, a genuinely live but
  system-wide (not process-scoped) signal. Every macOS receipt therefore
  carries an explicit `noise_flags` caveat:
  `"macOS unified-memory sample is a system-wide wired-memory delta, not
  process-scoped"`.
- **Threading.** `launch_local` in `src/inference/launch.rs` split into a
  thin unchanged-signature wrapper plus
  `launch_local_with_resolved_preset(state, request, app_config,
  Option<ModelPreset>)`. `src/llama/server.rs::start_backend` gained an
  `EvidenceContext { app_config, resolved_preset }` parameter (bundled into
  one struct to stay under clippy's 7-argument limit). Only the one
  `sessions.rs` call site with a resolved `ModelPreset` in scope (the
  preset-bundle launch path) was switched to the new function; the other
  three `launch_local` call sites (session restore, direct payload,
  `llama_binary.rs`) are unchanged and pass `None`.
- **Timing.** `pre_launch_wired_bytes` is sampled in `start_backend`
  immediately before `supervisor.start()`. The sampler itself is spawned via
  a detached `tokio::spawn` immediately before `start_backend`'s final
  `Ok(())` — after readiness has already succeeded and
  `state.server_running` is already `true` — so it can never block normal
  session control (start/stop/restart). It polls
  `memory_availability::build_snapshot()` 6 times at 750ms intervals,
  tracks `peak_bytes = max(wired_bytes)`, takes a final `after` sample,
  rebuilds the `CapabilitySnapshot` via the existing `OnceLock` cache
  (never re-spawns `llama-server --help`), and persists via `store::save`.
  Every internal step fails closed and silently (no panic, no propagated
  error) — evidence capture is advisory only.
- **Gate.** Same shape as `build_launch_observation`'s hard gate, checked a
  second time in `spawn()` itself so a disqualified launch never even
  schedules a poll task: `cfg!(target_os = "macos")`,
  `preset.fit_enabled == Some(false)`, and empty `extra_args`.

### Windows/CUDA `nvidia-smi` sampler — design only

No Windows/CUDA hardware is reachable from this session (this machine is
Apple Silicon; the only other known machine requires explicit
Coordinator/user authorization before starting or stopping a real model
server, which was not sought or given). Wrote
`docs/plans/evidence/preset-bundles/windows-cuda-sampler-design.md`
covering: the `WddmTotalDeviceDelta`/`CudaRocmProcessDelta` method mapping
(already reserved in the `LaunchEvidenceMethod` enum), the qualifying gate
(mirrors `metal_sampler::spawn`), the before/idle-stabilization/peak/after
sampling algorithm with background-process noise detection and repeated
observation (per Phase 9's real-host-gate requirements, which go beyond
what the single-shot Metal sampler needs), receipt shape (reuses
`LaunchSample`/`LaunchObservationReceipt` unchanged, no schema migration),
and explicit open questions left for whoever implements it on real
hardware. No CUDA/Windows code was written — writing untested
process-spawning logic against an assumed `nvidia-smi` output format
without hardware to verify it against would be fabricated evidence, not a
design.

### Fixture corpus additions (`launch_evidence.rs` test module)

Added to the 11 pre-existing unit tests:

- `manifest_digest_changes_for_every_memory_relevant_field` — one mutation
  fixture per `MemoryRelevant` field (44 fields), with the fixture set
  itself asserted equal to `classify_argv_field`'s `MemoryRelevant` set, so
  a newly added memory-relevant field with no mutation fixture fails this
  test rather than going unverified.
- `windows_wddm_and_cuda_rocm_methods_are_direct_observations_with_distinct_identity`
  — platform-labelled fixtures for the two non-macOS direct-observation
  methods; confirms method is part of launch identity (a receipt cannot
  match across methods even with an identical manifest digest).
- `estimator_only_and_fit_probe_never_match_or_power_measured_evidence`.
- `negative_delta_from_noisy_sampling_never_underflows_into_a_bogus_positive_number`
  — noisy background usage / implausible (negative) delta fixture; confirms
  `checked_sub` yields `None` rather than a wrapped bogus positive number.
- `incomplete_sampling_window_is_recorded_not_hidden` — a short sample
  window (partial `sample_count`, e.g. process exited mid-poll) still
  builds a receipt with the shortfall visible in `noise_flags`, rather than
  the incompleteness being silently dropped.
- `metal_sampler_spawn_is_a_noop_when_launch_does_not_qualify_for_exact_evidence`
  — calls the real `metal_sampler::spawn` outside a tokio runtime for two
  disqualifying presets (fit not pinned off; non-empty `extra_args`) and
  relies on a panic (from reaching `tokio::spawn` with no runtime) to prove
  the gate returns before ever scheduling a task.
- The existing `every_model_preset_field_has_an_argv_classification` test
  already satisfies Phase 9's "fail-closed test that adding a typed argv
  field without fingerprint classification breaks the manifest validator"
  requirement; no new test was needed for it.

Not added (deferred, see Outstanding below): the frontend evidence-details
drawer action and its fixtures, and the `core/preset-flow.spec.js`
playwright run.

### Verification (slice 2)

- `rtk cargo build` — passed
- `rtk cargo clippy -- -D warnings` — no issues
- `rtk cargo fmt -- --check` — no diff
- `rtk cargo test calibration::` — 63 passed, 1 ignored (11 suites)
- `rtk cargo test preset_bundles` — 10 passed (11 suites)
- `rtk cargo test inference::` — 377 passed, 2 ignored (11 suites)
- `rtk cargo test llama::` — 305 passed, 3 ignored (11 suites)
- `rtk cargo test sessions` — 15 passed (11 suites)

### Files changed (slice 2)

- `src/calibration/launch_evidence.rs` (new `metal_sampler` module, new
  fixture corpus tests)
- `src/inference/launch.rs` (`launch_local_with_resolved_preset`)
- `src/llama/server.rs` (`EvidenceContext`, sampler-spawn hook)
- `src/web/api/sessions.rs` (bundle-launch call site)
- `docs/plans/evidence/preset-bundles/windows-cuda-sampler-design.md` (new)

## Slice 1 — original scope

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

## Outstanding (after slice 2)

- Windows/CUDA `nvidia-smi` sampler implementation — design only (see
  slice 2 addendum); no Windows/CUDA hardware reachable from this session.
  Scoped to CUDA only; no ROCm hardware available anywhere in this project.
- Frontend evidence-details drawer action from the card.
- `rtk cargo test presets::evidence::` (no such module exists — the launch-
  evidence tests live in `calibration::launch_evidence` by design, per this
  doc's opening note that this is a receipt kind layered onto Calibration's
  vocabulary, not a separate module; Phase 9's plan text naming
  `presets::evidence::` predates that design decision) and the
  `core/preset-flow.spec.js` playwright run covering exact/compatible/
  related/stale evidence classes on a live card.
- Real-host qualification gates — require explicit user authorization
  before starting/stopping a model server on this machine or the remote
  Windows/CUDA machine; not requested yet in either slice.

## Hard gate

Exercised by unit fixtures (`build_launch_observation_rejects_fit_on`,
`build_launch_observation_rejects_extra_args`, and `metal_sampler`'s own
`spawn`-time gate check) and, on macOS, by the live `metal_sampler` path
end to end — but only up to persistence; no real-host qualification run has
produced a receipt from an actual model launch yet, so the architecture's
"no observation may be recorded as exact evidence unless fit was pinned
off" gate has not been exercised against a real host's launch, on either
platform.
