# Phase 4c receipt — two-sided MoE placement search

Implemented `src/presets/fit_search.rs` as a pure search over the Phase 4b
`FitReader` seam. Device fit is searched as a suffix from `n_max` downward to
find the smallest accepted `n_cpu_moe`; host fit is searched as a prefix from
zero upward to find the largest accepted placement. The result is the lower
bound of the intersected interval, never a combined non-monotone predicate.

The search uses every integer boundary, applies the caller-provided reserve,
returns device/host headroom, and caches each `n` locally so the two searches
never revisit a point. Probe failures and wall-clock exhaustion return an
explicit unavailable result. Placement search itself never spawns a process or
applies a launch change.

Focused evidence:

- `rtk cargo test presets::fit_search::` — 7 passed
- CUDA Phase 0 boundary: `n_cpu_moe=18` for a 16,384 MiB device budget and
  1,024 MiB reserve (17 rejected, 18 accepted)
- Interval fixture: device boundary 12 and host boundary 24 returns 12
- Disjoint, host-limited, reserve-monotonicity, default-reserve, and duplicate
  read protection are covered.
