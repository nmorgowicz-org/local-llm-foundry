# Phase 4d receipt — probe-backed resolve wiring

Extended `ResolvedLaunch` with the frozen estimate status and replaced the
placeholder resolve response with tagged `EstimateStatus` JSON. Explicit
`fit_automatically` requests now use the optional configured probe. Dense
models perform one point read at `n_cpu_moe=0`; MoE models use the 4c search.
The selected result is re-resolved before returning so hashes and change
records describe the exact proposed selection.

Probe readings are enriched into the existing VRAM breakdown shape with
`method=fit_probe`, device/host totals, per-component formula divergence, and
named additions for probe-unaccepted mmproj, draft, cache-RAM, SWA, thread,
and tensor-split settings. The result is estimate-class only: it is not
runtime evidence and does not enter calibration `memory_peak_bytes` or an
evidence match. Missing probe configuration, invalid artifact provenance,
probe failures, and unavailable search results remain disabled-with-reason
statuses; they never gate or auto-apply a launch.

The zero-placement result remains represented by the exact selection identity,
while llama.cpp argument emission suppresses `--n-cpu-moe 0`.

Focused evidence:

- `rtk cargo test presets::probe_estimate::` — 2 passed
- `rtk cargo test presets::resolver::` — 6 passed
- `rtk cargo test web::api::preset_bundles::` — 10 passed
- `rtk cargo clippy -- -D warnings` — passed
