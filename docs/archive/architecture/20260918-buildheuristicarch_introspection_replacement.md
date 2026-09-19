---
type: implementation-plan
context: Introspection-first cleanup of the HF quant advisor's stale architecture compatibility path.
decision-record: bac4e93
---

# Introspection-First Architecture Cleanup Implementation Plan

> **For agentic workers:** Execute inline, task-by-task. Do **not** delegate work to subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the last inert `buildHeuristicArch` compatibility path so the HF quant advisor uses only GGUF-derived architecture state.

**Architecture:** GGUF parsing already derives hybrid-attention, MoE, DeltaNet-state, and Gemma global/local KV geometry in Rust. Both local introspection and progressive HF-header introspection merge that metadata into `wizardState.arch`; the quant advisor must consume that state directly. If metadata is unavailable, omit the structural field and retain the existing degraded, non-guessing estimate path.

**Tech Stack:** Rust, Warp, browser ES modules, Playwright, ESLint.

**Spec:** `docs/archive/architecture/20260918-buildheuristicarch_introspection_replacement.md` (this completed execution plan)

## Global Constraints

- Scope is limited to the stale frontend compatibility helper and its only live caller. Do not alter Rust GGUF parsing, estimator formulas, or API schemas.
- `wizardState.arch` is authoritative only after local or HF GGUF-header introspection. Never derive architecture properties from a model filename, repository name, or parameter count.
- Preserve degraded behavior: a missing `globalHeadDim` serializes as no `global_head_dim` request property, not a guessed value or a numeric sentinel.
- Do not create static assets. Existing static-asset generation files must remain unchanged.
- Conventional Commit, if committed: `refactor(wizard): remove stale architecture heuristic`.
- UI verification is required: build release before capture; run the capture scenario sequentially.

---

## Current-State Audit

| Former plan assertion | Current evidence | Execution decision |
| --- | --- | --- |
| `buildHeuristicArch` is an active ~150-line filename lookup. | `static/js/features/spawn-wizard.js` now exports a zero-valued compatibility function; its historical lookup is commented out. | Delete the export and historical block. |
| The helper has two call sites. | Symbol references show one live call: `loadQuantAdvisor()` in `static/js/features/spawn-wizard-hf-browse.js`. | Migrate that caller; remove its import. |
| Hybrid, MoE, and sliding-window data still need server-side implementation. | `GgufMetadata::to_model_metadata()` supplies `n_attn_layers`, `linear_attn_state_bytes`, experts, global/local head dimensions, and sliding-window data. Both local and HF flows hydrate `wizardState.arch`. | No Rust implementation change. |
| MTP draft discovery needs this plan. | The current MTP surface determines visibility from introspected `mtpDepth` or a selected companion draft; no `detectMtpFromName` call exists. | Explicitly out of scope. |

### Success Criteria

1. `buildHeuristicArch` has no definition, import, reference, or commented historical implementation in `static/js/`.
2. `loadQuantAdvisor()` sends `global_head_dim` only when `wizardState.arch.globalHeadDim > 0`.
3. A focused browser test proves both the resolved (`512`) and degraded (omitted) request contracts.
4. The full Rust and JavaScript validation commands pass, and the existing wizard screenshot scenario completes after a release build.

## File Map

| File | Responsibility | Change |
| --- | --- | --- |
| `static/js/features/spawn-wizard-hf-browse.js` | Builds the HF quant-advisor request. | Read `globalHeadDim` from authoritative wizard state and remove the helper import. |
| `static/js/features/spawn-wizard.js` | Owns wizard state and currently retains obsolete compatibility code. | Delete `buildHeuristicArch` and its commented historical implementation. |
| `tests/ui/core/spawn-wizard.spec.js` | Browser contract tests for spawn-wizard behavior. | Add a focused intercepted-request test for resolved and degraded global-head-dimension behavior. |

## Task 1: Remove the stale helper and prove the request contract

**Files:**

- Modify: `static/js/features/spawn-wizard-hf-browse.js:19-35,231-282`
- Modify: `static/js/features/spawn-wizard.js:3958-4080`
- Modify: `tests/ui/core/spawn-wizard.spec.js`

**Interfaces:**

- Consumes: `wizardState.arch.globalHeadDim: number`; zero means introspection did not resolve a global head dimension.
- Produces: `POST /api/vram/quant-compare` with optional `global_head_dim: number`. A positive resolved value is forwarded unchanged; an unresolved value is absent from JSON.
- Invariant: the frontend never restores a filename, family, or parameter-size heuristic after this cleanup.

- [x] **Step 1: Add the failing browser contract test**

Append an `@in-memory-test` to the existing spawn-wizard suite. Intercept `**/api/vram/quant-compare`, store each parsed POST body, and return the smallest successful quant-comparison payload that lets the advisor finish rendering.

```js
test('@in-memory-test quant advisor forwards resolved global head dimension and omits degraded metadata', async ({ page }) => {
  const requests = [];
  await page.route('**/api/vram/quant-compare', async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, quants: [] }),
    });
  });
  await page.goto('/');
  await page.waitForLoadState('networkidle');

  await page.evaluate(async () => {
    const { wizardState } = await import('/js/features/spawn-wizard.js');
    const { triggerQuantAdvisor } = await import('/js/features/spawn-wizard-hf-browse.js');
    Object.assign(wizardState.model, { paramB: 31, path: 'renamed-model.gguf' });
    wizardState.vram.available = 32 * 1024 ** 3;
    wizardState.hardware.parallelSlots = 1;
    wizardState.arch.globalHeadDim = 512;
    triggerQuantAdvisor();
  });
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].global_head_dim).toBe(512);

  await page.evaluate(async () => {
    const { wizardState } = await import('/js/features/spawn-wizard.js');
    const { triggerQuantAdvisor } = await import('/js/features/spawn-wizard-hf-browse.js');
    wizardState.arch.globalHeadDim = 0;
    triggerQuantAdvisor();
  });
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[1]).not.toHaveProperty('global_head_dim');
});
```

Adjust only the state fields required by the existing `effectiveAvailBytes()` implementation; do not mock the module or add a test-only production export. Do not assert implementation details such as imports or source text.

- [x] **Step 2: Run the focused test and confirm it fails**

Run from `tests/ui`:

```bash
CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 \
  npm test core/spawn-wizard.spec.js -- --grep "quant advisor forwards resolved global head dimension"
```

Expected: FAIL because the current request gets `global_head_dim` from the zero-valued compatibility helper, not `wizardState.arch.globalHeadDim`.

- [x] **Step 3: Make the minimal production change**

In `spawn-wizard-hf-browse.js`:

```js
// Delete buildHeuristicArch from the spawn-wizard.js import.
global_head_dim: wizardState.arch.globalHeadDim || undefined,
```

In `spawn-wizard.js`, delete the entire `buildHeuristicArch` declaration and the immediately following commented historical heuristic block. Keep `getEffectiveArch()` and `getSizingArch()` unchanged; their shallow-copy behavior is the established interface.

- [x] **Step 4: Run the focused test and confirm it passes**

Repeat Step 2. Expected: PASS. The intercepted resolved request contains `512`; the degraded request omits `global_head_dim`.

- [x] **Step 5: Confirm the obsolete path is fully removed**

The focused browser test imports both affected ES modules in the application environment. Then search `static/js/` for `buildHeuristicArch`. Expected: no matches. The standalone Node import does not work for this browser module graph because it accesses `localStorage`; `npm run validate-js` supplies the syntax check without that false-negative runtime condition.

- [x] **Step 6: Run required validation and UI proof**

```bash
npm run validate-js
npm run lint
cargo test
cargo clippy -- -D warnings
cargo build --release
node tests/ui/capture/index.mjs --scenario spawn-wizard
cd tests/ui && CI=1 LLAMA_MONITOR_USE_RELEASE=1 LLAMA_MONITOR_TEST_PORT=17778 npm test
```

Run commands serially. The capture must use the release binary just built. Treat screenshots under `docs/screenshots/artifacts/` as verification artifacts only; do not promote or commit one unless documentation references it.

- [x] **Step 7: Check the final diff and commit**

```bash
git diff --check
git status --short
git add static/js/features/spawn-wizard.js \
  static/js/features/spawn-wizard-hf-browse.js \
  tests/ui/core/spawn-wizard.spec.js \
  docs/archive/architecture/20260918-buildheuristicarch_introspection_replacement.md
git commit -m "refactor(wizard): remove stale architecture heuristic"
```

Before staging, verify that only this task's implementation files and this plan appear. Do not stage unrelated work.

## Completion Evidence

- The focused Playwright test exercises real browser module state and verifies both request shapes.
- `cargo test`, `cargo clippy -- -D warnings`, `npm run validate-js`, `npm run lint`, the release build, the wizard capture, and the isolated Playwright suite pass.
- The final search has no `buildHeuristicArch` match under `static/js/`.
- No filename- or repository-name architecture inference is introduced.

## Out of Scope

- Changing `GgufMetadata`, `ModelMetadata`, `ModelArch`, estimator formulas, or Rust API response contracts.
- Adding a new GGUF field or changing progressive header-fetch behavior.
- MTP companion-file discovery or model-family display labels.
- Editing any unrelated user work already present in the working tree.
