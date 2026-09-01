# Phase 4a receipt — deterministic fit intents

Implemented `src/presets/intent.rs` as a pure, deterministic proposal
function. It returns one exact bundle selection, the resolver's complete
`ResolvedChange` list, and the existing estimator breakdown without probing a
binary, reading the filesystem, or mutating the saved default.

The proposal order is fixed: exact lower local artifact, listed context
reduction (bounded by artifact native context), listed batch/ubatch reduction,
then qualified discrete-memory MoE placement. Local down-selection requires a
local weights path, positive size, and a full-file SHA-256 digest. Curated-only
bundles remain curated; validated-custom bundles still pass exact resolver
validation. Agentic/unknown workload policy restrictions prevent silent
`q4/q4` selection, and mixed `q8/q4` is never proposed. Unified-memory MoE
automatic placement returns `n_cpu_moe_unified_memory_unqualified`.

Focused evidence:

- `rtk cargo test presets::intent::` — 4 passed
- Tests cover complete change explanations, policy quality floor, native
  context limit, deterministic low-VRAM ordering, and unified-memory MoE
  unavailability.

The implementation reuses `full_estimate`; it adds no client-side VRAM formula
and does not invoke `llama-fit-params`.
