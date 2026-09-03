# Phase 4b receipt — bounded llama-fit-params probe

Implemented `src/presets/fit_probe.rs` and the optional
`AppConfig::llama_fit_params_path` field (default `None`). The seam exposes the
frozen `FitReading` and `FitReader` contracts plus production
`ProcessFitReader` and first-class embedded `FixtureFitReader` implementations.

The process reader invokes the fixed `--fit off -lm none -lv 4 -fitp on`
contract with the resolved model, context, K/V, draft K/V, batch/ubatch, and
`--n-cpu-moe` values. Stdout and stderr are captured independently and bounded.
The parser sums device rows from the full table and every non-device row for
host memory, including `CPU_REPACK`; it does not use the `free` column or a
single named host row. Identity binds canonical path, SHA-256, mtime, and the
bounded version line. Cache keys include artifact digest, the probe-accepted
resolved configuration, probe SHA-256, and `n_cpu_moe`.

The embedded Phase 0 corpus is parsed without a binary. Successful fixture
points are round-tripped, while non-zero/invalid fixtures remain actionable
errors and are never converted to zero memory. Timeout, oversized output,
non-zero exit, parse, and identity failures all remain disabled-with-reason
states for callers.

Focused evidence:

- `rtk cargo test presets::fit_probe::` — 7 passed
- Coverage includes Metal `CPU_REPACK`, CUDA host growth, all successful Phase
  0 fixture points, parse failure retention, SHA/mtime/version identity
  failures, and bounded timeout behavior.

The reader is estimate-class evidence (`fit_probe`); it is not runtime measured
memory evidence and is not itself a placement search. Phase 4c consumes this
seam for the two-sided `n_cpu_moe` search and reserve application.
