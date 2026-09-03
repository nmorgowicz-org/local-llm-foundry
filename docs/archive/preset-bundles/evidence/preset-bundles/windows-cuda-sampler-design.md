# Windows/CUDA `nvidia-smi` sampler — design

Status: `WddmTotalDeviceDelta` is implemented (`nvidia_sampler` in
`src/calibration/launch_evidence.rs`, phase-9-receipt.md slice 3) and passes
build/clippy/fmt/unit tests, including against a real `nvidia-smi` capture
from Ryne (`tests/fixtures/nvidia_smi_compute_apps_csv.txt`, RTX 5090, driver
616.56). That real capture shows `used_memory` as `[N/A]` for every process
under WDDM, so `CudaRocmProcessDelta` (per-process attribution) remains
deliberately unimplemented — see "Method mapping" below. This machine is
Apple Silicon, so the sampler has never *executed* (its `cfg!(target_os =
"windows")` gate is always false here); no Windows/CUDA hardware has run it,
and no real model server has been started or stopped anywhere. A real-host
qualification run on Ryne still requires explicit Coordinator/user
authorization, per architecture section 12 and execution plan Phase 9's
"Real-host gates" — not sought or granted as of this status update.

## Method mapping

`LaunchEvidenceMethod` already reserves the two variants this sampler feeds
(`src/calibration/launch_evidence.rs:200-206`):

- `WddmTotalDeviceDelta` — Windows, total-device before/peak/after delta via
  `nvidia-smi`. WDDM does not expose reliable per-process VRAM accounting on
  Windows, so this is device-wide, matching the enum's existing
  "WDDM presented as per-process" stop condition (execution plan Phase 9).
- `CudaRocmProcessDelta` — used only where the driver's `nvidia-smi
  --query-compute-apps` (or ROCm's equivalent, out of scope: no ROCm
  hardware available anywhere in this project) reliably attributes memory to
  the launched process. Where that attribution is not reliably available,
  the sampler must fall back to `WddmTotalDeviceDelta`-style total-device
  accounting and say so in `noise_flags` rather than claim process-scoped
  precision it cannot back up.

`current_platform_method()` (`launch_evidence.rs:483-495`) already selects
`WddmTotalDeviceDelta` on `cfg!(target_os = "windows")` and
`CudaRocmProcessDelta` on every other non-macOS target. This sampler's `spawn`
gate should mirror the same `cfg!(target_os = "windows")` split the Metal
sampler uses for `target_os = "macos"`, so the qualifying condition and the
fingerprint's `method` field can never disagree.

## Qualifying gate (mirrors `metal_sampler::spawn`)

Same shape as the Metal sampler's gate
(`src/calibration/launch_evidence.rs`, `metal_sampler::spawn`):

- `cfg!(target_os = "windows")` (or the CUDA/ROCm non-macOS branch — see
  above);
- `BackendAdapter::LlamaCpp` only;
- `evidence.resolved_preset` is `Some`;
- `fit_enabled == Some(false)` on the resolved preset (fit must be pinned
  off — architecture 12's hard gate);
- `extra_args` empty (same "no unclassified argv smuggled in" rule the
  Metal sampler and `build_launch_observation` both enforce).

Threading is already in place end to end for this gate — `EvidenceContext`,
`launch_local_with_resolved_preset`, and the `sessions.rs` bundle-launch call
site all exist and are platform-agnostic. Only a new
`nvidia_sampler::spawn(app_config, preset, before)` sibling to
`metal_sampler::spawn`, called from the same site in
`src/llama/server.rs::start_backend`, is needed — the call site would gate on
`(&adapter, evidence.resolved_preset)` exactly as it does now and simply
also invoke `nvidia_sampler::spawn` when the platform matches. No further
`sessions.rs`/`launch.rs` changes are required.

## What `nvidia-smi` actually provides

`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`
gives a total-device snapshot (WDDM path). `nvidia-smi
--query-compute-apps=pid,used_memory --format=csv,noheader,nounits` gives
per-process figures where the driver supports it, keyed by PID — usable for
`CudaRocmProcessDelta` when the launched llama.cpp PID appears in that list.
Both are external-process invocations (`tokio::process::Command`), unlike the
Metal sampler's in-process `memory_availability::build_snapshot()` call, so
every sample carries real process-spawn latency and failure modes (binary not
on `PATH`, driver not loaded, permission failure) that the Metal path does
not.

## Sampling algorithm

Distinct from the Metal sampler's single post-readiness poll loop, because
architecture 12 and Phase 9 require a full before/peak/after shape with idle
stabilization and noise detection for this method, not just a peak-only
proxy:

1. **Pre-existing inventory check.** Before spawn (in `start_backend`,
   mirroring the Metal sampler's `pre_launch_wired_bytes` capture), record
   the full `nvidia-smi --query-compute-apps` process list and the
   total-device `memory.used`. This is the "pre-existing
   server/port/process inventory" the Phase 9 real-host gate requires — it
   lets the receipt later show whether some other process's memory churn
   (not this launch) explains an implausible delta.
2. **Idle stabilization before the `before` sample.** Poll total-device
   `memory.used` at a short fixed interval (proposed: 5 samples at 200ms)
   and only accept the `before` baseline once consecutive samples agree
   within a small tolerance (proposed: 16 MiB). An unstabilized `before`
   sample is the single largest source of a false "implausible delta" the
   Phase 9 fixture corpus is required to cover.
3. **`before` sample.** Timestamped raw `nvidia-smi` total-device (and, when
   using `CudaRocmProcessDelta`, per-process) sample, taken immediately
   before spawn — same timing slot as the Metal sampler's
   `pre_launch_wired_bytes`.
4. **`readiness` sample.** Taken the instant `start_backend`'s readiness
   race (`adapter.await_ready` vs. `supervisor.wait_for_exit()`) resolves
   successfully — i.e., at the same point the Metal sampler hook fires
   today, right before `state.server_running` flips to `true`. This is a
   named point in the receipt distinct from `peak`, because llama.cpp may
   continue allocating (KV cache warmup, first-request buffers) after the
   readiness probe succeeds but before steady state.
5. **Bounded post-readiness peak poll.** Same shape as the Metal sampler:
   fixed sample count at a fixed interval, run in a detached
   `tokio::spawn` task so it cannot block normal session control. Track
   `peak = max(total-device memory.used)` (or `max(per-process used_memory)`
   for `CudaRocmProcessDelta`) across the window. Proposed starting point:
   6 samples at 750ms, matching the Metal sampler's cadence, tunable once a
   real corpus exists.
6. **`after` sample.** Final sample at the end of the poll window, same
   role as the Metal sampler's `after`.
7. **Background-process change/noise detection.** Re-read the full
   `--query-compute-apps` process list at the `after` point and diff PIDs
   against the step-1 inventory (excluding the launched llama.cpp PID
   itself). Any PID that appeared, disappeared, or whose own
   `used_memory` moved by more than the stabilization tolerance during the
   sampling window is recorded in `noise_flags` — e.g. `"background CUDA
   process pid=<n> changed by <bytes> during sampling window"`. This is
   what lets `classify_evidence_match`-adjacent logic keep a noisy result at
   `related` rather than `exact`, per the Phase 9 stop condition ("a noisy
   or non-repeatable result remains related/noisy, never exact").
8. **Repeated observation.** Phase 9's real-host gate explicitly requires
   "repeated observation," unlike the Metal sampler (single-shot). Proposed
   shape: run the full before/peak/after cycle described above N times
   (starting point N=3) against the *same already-running* server — i.e.
   repeat only steps 2-6 against the live process, not full
   start/stop/restart cycles — and require the resulting deltas to agree
   within tolerance before the receipt is written as a direct observation;
   otherwise mark it noisy in `noise_flags` and downgrade confidence exactly
   as step 7 does. This still respects "no blocking of normal session
   control," since it is the same detached background task the Metal
   sampler uses, just running a longer bounded loop.
9. **Guaranteed cleanup.** No process, file handle, or lock is held across
   samples — each `nvidia-smi` invocation is a fresh short-lived child
   process reaped by `tokio::process::Command`'s normal `.output()` await, so
   there is no persistent state to clean up on the sampler's own side. The
   receipt-write path reuses `store::save`'s existing atomic
   `.json.tmp` + `fs::rename`, identical to the Metal sampler, so a crash
   mid-sample never leaves a partial evidence file. "Guaranteed cleanup"
   here is a property of the design (no held resources), not a `Drop` guard
   that needs writing.
10. **Model delta.** Compute `peak - before` (device-wide for
    `WddmTotalDeviceDelta`; the launched PID's own delta for
    `CudaRocmProcessDelta`) as the launch's attributable memory, following
    the Metal sampler's existing `peak.saturating_sub(before)` pattern in
    `launch_evidence.rs`.

## Receipt shape

Reuses the existing `LaunchSample`/`LaunchObservationReceipt` types
unchanged — no new evidence-store schema is needed, matching Phase 9's "do
not create an unrelated evidence store" instruction. Method-specific detail
(raw per-sample `nvidia-smi` output, the background-process diff, the
idle-stabilization sample count actually needed, the repeat count and
per-repeat deltas) belongs in `LaunchSample.noise_flags` as human-readable
strings, the same mechanism the Metal sampler already uses for its
system-wide-vs-process-scoped caveat — not as new typed fields, so this
sampler needs no schema migration of already-persisted Metal receipts.

`memory_peak_bytes` on `CalibrationMeasurement` is populated the same way
the architecture allows for any direct-observation method: only when the
receipt's `method.is_direct_observation()` is true, which both
`WddmTotalDeviceDelta` and `CudaRocmProcessDelta` satisfy per the existing
enum (`launch_evidence.rs:208-215`) — no new logic needed there either.

## What this design explicitly does not do

- It does not implement `nvidia_sampler::spawn`/`run` — there is no way to
  compile-test, let alone run, CUDA-path code without the target hardware,
  and writing untested process-spawning logic against an assumed
  `nvidia-smi` output format would be exactly the kind of speculative code
  the project's standing constraints (no fabricated evidence, fail closed on
  the unverifiable) argue against.
- It does not start or stop any real model server on Ryne or any other
  machine. That remains gated on explicit Coordinator/user authorization
  per Phase 9's "Real-host gates" and the standing instruction reaffirmed
  this session.
- It does not attempt ROCm. No ROCm hardware exists anywhere in this
  project's reach (execution plan Phase 9 and this document's method
  mapping both note CUDA-only scope), so `CudaRocmProcessDelta`'s ROCm half
  is named in the enum for architecture completeness but has no design
  detail here beyond "the same process-accounting reliability check applies
  if ROCm tooling is ever available."

## Open questions for whoever implements this on real hardware

- Exact `nvidia-smi` invocation flags and CSV parsing robustness across
  driver versions (locale-dependent number formatting, `[N/A]` cells) need
  a real corpus, not assumption.
- Whether `--query-compute-apps` reliably attributes memory to a llama.cpp
  child process specifically (vs. reporting the shell or launcher instead)
  needs to be checked against the actual spawn topology
  (`crate::inference::supervisor::Supervisor`) on the target machine before
  `CudaRocmProcessDelta` can be trusted over the `WddmTotalDeviceDelta`
  fallback.
- Concrete stabilization tolerance, sample counts, and intervals proposed
  above are starting points only, explicitly "tunable once a real corpus
  exists" — they must not be treated as validated numbers until a real-host
  qualification run (under explicit authorization) produces the fixture
  corpus that this document's "Repeated observation" step depends on.
