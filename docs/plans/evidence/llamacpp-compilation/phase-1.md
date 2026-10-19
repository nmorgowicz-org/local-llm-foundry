# Phase 1 receipt: K/V classification backend

Baseline: `fcf2b878` (main). Branch `feat/llamacpp-source-builds`, result `826756e7` plus this receipt.

## Delivered

- `src/inference/kv_support.rs`: pure `classify_kv_pair` (Native / Fallback / Unverified / WontStart),
  typed `Evidence`, `compiled_fa_pairs` / `parse_fa_quants_record`, log observation parser and
  session store.
- `src/llama_runtime/probe.rs`: bounded runtime probe (10 s, 1 MiB combined cap, owned temp cwd,
  unique missing model path, recognized model-missing diagnostic, inherited or sanitized env) and
  closure identity (executable + loaded backend plugins + same-directory llama/ggml libraries).
- `CapabilitySnapshot`: `MixedMainKv` removed; `kv_basis` and `runtime_identity` added;
  `is_valid_for` and `fingerprint` include the closure.
- Callers moved to the one classifier: preset validation, launch, calibration, qualification,
  bundle resolver and API, Doctor, VRAM estimate. Only quantized V with FA off blocks
  (`KV_PAIR_WONT_START`). The no-snapshot q8_0/q4_0 launch block is deleted.
- `GET /api/llama-binary/kv-support` (authenticated; model id is a path relative to the models
  directory, contained server-side; device id validated against the runtime's enumeration).
- Docs: `docs/reference/api.md`, `docs/reference/inference-tuning.md`.

## Evidence

- Real Metal probe fixture (b11719, M5 Max) and real CUDA probe fixture (ryne, RTX 5090,
  `FA_QUANTS = all`) parse to 49 confirmed pairs.
- `LLAMA_SERVER_UNDER_TEST=~/AI/llamacpp/llama-server cargo test --lib real_binary -- --ignored`
  reports backend Metal, devices enumerated, closure identity present.

## Commands (final run)

| Command | Exit |
| --- | --- |
| `cargo fmt --check` | 0 |
| `cargo clippy --all-targets -- -D warnings` | 0 |
| `cargo test --lib` (1668 passed, 15 ignored) | 0 |
| `git diff --check` | 0 |

## Deviations from the plan, to confirm

1. **Launch-time identity recheck uses size + mtime**, not a rehash. CUDA backends are hundreds of
   MB and capability probes run on every launch. Content hashes are cached by (path, size, mtime)
   and recomputed at probe time. A same-size, same-mtime plugin swap is not detected between probes.
2. **Endpoint lives in `src/web/api/llama_kv_support.rs`**, not `llama_binary.rs`.
3. **`fa` is the preset value mapped with launch semantics** (only `off` is off), so the answer
   matches what a launch does.
4. **Log observations are returned beside the classification and do not feed it**, because a log
   line is only valid for the session's own model, device and pair.
5. **Doctor warns only for Fallback with SlowerDecode/CpuAttention**, never Unverified (noise) or an
   explicit FA-off choice.
6. **The legacy snapshot is never persisted**, so migration is a serde default plus tests, not a
   DTO.

## Addendum: Metal bf16 and mixed pairs (same day)

- Probe now parses `ggml_metal_device_init: has bfloat` and the build number; the real install
  reports `build=11719 metal_bf16=Confirmed(true) metal_rule_verified=true`.
- Depth benchmarks (plan S1b) show every mixed pair on Metal runs CPU attention, including
  bf16/f16. Source confirms the K==V requirement. The classifier now returns
  `Fallback{CpuAttention}` on a verified build instead of `Unverified`.
- The verified range is a single build (`METAL_FA_RULE_VERIFIED_BUILDS = 11719..=11719`); newer
  builds stay `Unverified` until re-checked.
- The final lib run had one failure, `rapid_mlx::model_resolver::tests::template_overlay_does_not_mutate_original_model_dir`,
  in code this branch does not touch. It passed alone and in two further full runs (1673 passed).

## Not done in Phase 1

- Advisories (Phase 2); the endpoint returns `advisories: []`.
- Per-device (multi-GPU) Metal bf16; any device reporting false wins.
- Device/model evidence in non-endpoint callers: those stay Unverified when they would need it.
