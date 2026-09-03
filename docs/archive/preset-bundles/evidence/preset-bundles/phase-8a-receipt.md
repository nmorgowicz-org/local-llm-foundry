# Phase 8a receipt — compact bundle launch card

Implemented the Option B compact card: a v6 bundle renders as one launch card
carrying the exact tune title and the saved selection's summary chips (KV
policy, context, MoE placement) instead of one card per artifact. The render
kill-switch (`LLAMA_MONITOR_PRESET_BUNDLE_UI=legacy|bundled`) is resolved
server-side and exposed to the client only as the closed `preset_bundle_ui`
enum field on `GET /api/preset-cards`; `initSetupView()` now awaits
`_refreshPresetBundleUi()` before the first grid paint so a bundle preset
never flashes the wrong card shape, and a late-arriving flag value triggers
one corrective re-render rather than leaving a stale grid.

Start is an atomic resolve-and-launch: the client sends only the preset ID
and `expected_revision`, never a selection or config hash. A revision
mismatch is a one-shot 409 `revision_conflict` that re-renders the card from
the server response and re-asks — it never silently retries with the stale
revision. A bundle whose saved selection has no local artifact renders the
existing no-model degraded setup state before any spawn attempt, using the
same `#preset-select` population path as flat presets.

The Configure control exists on the card as a documented seam
(`.launch-card-btn-configure`) that Phase 8b's drawer will attach to; no
drawer logic is inlined here. The render flag's legacy path reuses the
existing one-artifact card adapter rather than introducing a second
rendering path.

Also repointed test/capture infrastructure from the deprecated
`llama-monitor` binary to the canonical `local-llm-foundry` binary
(`tests/ui/run-server.mjs`, `tests/ui/capture/harness/paths.mjs`), matching
`Cargo.toml`'s `[[bin]]` rename. Config-directory names (`~/.config/llama-monitor`,
`llama-monitor-test-*`) were left unchanged — that naming already has an
independent canonical/legacy cutover in `paths.mjs` unrelated to which binary
gets launched, and will move to `local-llm-foundry` under that existing
mechanism when the config-dir migration itself happens.

## Fixes made during implementation

- KV-policy chip previously showed the raw wire value (e.g. `q8_0_q8_0`)
  because `.split('_')` on a half-dtype that itself contains an underscore
  (`q8_0`) does not yield a 2-element pair. Replaced with an explicit lookup
  covering the four known `LlamaKvPolicyId` wire values from
  `src/presets/bundle.rs`, falling back to the old heuristic only for a
  genuinely unrecognized policy string.
- The degraded-bundle-setup fixture's `#preset-select` population was
  filtering out any preset with no resolvable local model source, which
  incorrectly excluded a bundle degrading to setup (a real, usable preset —
  just missing its saved artifact locally). Filter now exempts `p.bundle`.
- `data-bundle-id` / `data-revision` were being set on the card root but read
  by tests off `.launch-card-actions`; both live on the card root now, and a
  stale release binary (built before these JS edits, since `build.rs`
  embeds `static/` at compile time with no runtime fallback) was rebuilt.

## Verification

- `rtk npm run validate-preset-bundle-contract` — passed
- `rtk npm run validate-js` — passed
- `node tests/ui/update-baseline.mjs` — baseline unchanged at 95 JS modules
- `rtk cargo fmt --check` — passed (after fixing one pre-existing formatting
  issue in `src/config.rs`)
- `rtk git diff --check` — passed
- `core/preset-flow.spec.js` — 16 passed
- Full `core/` Playwright suite — 164 passed, 0 failed
