# Phase 3 receipt — resolver, revisioned selection, and authenticated APIs

Baseline: `10d373999494f4e04bd4615f36ffc7315e3ed51c`

Implementation commit: `c332e6da036cbbd468d5a2e6d5f5adaeba20f289`

## Files changed

- `src/presets/resolver.rs` — pure bundle resolver, projection, typed validation, and canonical `sel-v1`/`cfg-v1` hashes.
- `src/presets/bundle.rs` — server-owned conversion/copy helpers.
- `src/web/api/preset_bundles.rs` — authenticated cards, resolve, selection, copy, and conversion routes.
- `src/web/api/presets.rs` — revisioned bundled PUT, authenticated destructive mutations, and catalog etags.
- `src/web/api/sessions.rs` — server-side bundle resolution and preview guards during spawn.
- `src/state.rs` — persisted session selection hash.
- `src/inference/llama_cpp_capabilities.rs` — product-default capability fallback.
- `src/web/api/mod.rs` — bundle route registration.
- `tests/fixtures/presets/fingerprint_golden.json` — three literal golden selections/configurations.
- `docs/reference/api.md` — cards, resolution, selection, copy, conversion, revision, etag, and destructive-auth contracts.

## Validation

| Command | Result |
| --- | --- |
| `rtk cargo test presets::resolver::` | 6 passed |
| `rtk cargo test web::api::preset_bundles` | 9 passed |
| `rtk cargo test --test auth_routing` | 39 passed |
| `rtk cargo clippy -- -D warnings` | passed |
| `rtk cargo test` | 1,422 passed, 14 ignored |
| `rtk npm run validate-js` | passed |
| `rtk npm run lint` | passed |
| `rtk git diff --check` | passed |
| `rtk cargo build --release` | passed |
| `rtk cargo fmt -- --check` | passed |

The first parallel full-suite run had one calibration fixture failure; the
test passed in isolation and the complete suite passed on the immediate rerun.

## Gates and known boundaries

- Preview, saved-default projection, direct selection spawn, and saved-default
  spawn share the named `same_selection_same_fingerprint_across_surfaces`
  contract test.
- Resolve is read-only; selection, copy, conversion, PUT, delete, and reset
  persist before swapping in-memory state and reject stale revisions/etags.
- Card and resolve responses redact local paths and API keys.
- Estimate enrichment remains `not_applicable` until Phase 4a; exact runtime
  evidence remains deferred to Phase 9.
- Artifact path canonicalization/root membership remains owned by the existing
  inventory/editor validation boundary and is not claimed as Phase 3 evidence.
- No UI surface changed in this phase; screenshot capture is not applicable.
