# Phase 4.5 receipt — throwaway card and drawer UX gate

Status: contract confirmed with amendments; independent human attestation complete

Date: 2026-09-01

Baseline commit: `12d383f` (`feat(presets): render bundle presets as a single compact launch card`)

## Scope

Built a disposable prototype under the ignored `tmp/phase-4-5/` directory. It
contains a hand-authored bundle card, Configure drawer, local stubbed resolver
responses, and no production route, persistence, `sessionState`, `bootstrap.js`,
or reusable application module changes.

The prototype covers the UX-risk states required by the execution plan:

- disabled mixed-KV choice remains visible and associates its backend reason with
  `aria-describedby`;
- Low VRAM proposal changes are shown individually in the diff;
- `Custom` is a derived indicator, not an intent control;
- CPU expert placement shows a qualitative slowdown warning;
- draft changes leave the collapsed card unchanged;
- Reset discards without prompting;
- Escape and backdrop dismissal of a dirty drawer prompt;
- closing restores focus to the opener;
- Fit automatically produces a draft-only probe proposal;
- Save updates the card; Start without saving leaves the saved card unchanged;
  - light theme, narrow bottom-sheet layout, and reduced-motion state are captured;
  - dense-model no-change behavior explains that context reduction or model-card
    variant discovery is the remaining tradeoff.

## Reproduction

From the repository root:

```text
rtk proxy node --check tmp/phase-4-5/drawer.js
rtk proxy node --check tmp/phase-4-5/run.mjs
rtk proxy node tmp/phase-4-5/run.mjs
```

For manual review, serve the module over localhost rather than opening
`index.html` with `file://` (browser module security blocks that origin):

```text
rtk proxy node tmp/phase-4-5/serve.mjs
```

Then open `http://127.0.0.1:41845/` and stop the server with Ctrl-C afterward.

The first human review identified and rejected two prototype behaviors: Low VRAM
changed workload/context/KV below the Agentic quality floor, and the narrow and
reduced-motion captures were not visually distinguishable. The scratch fixture
and architecture contract were amended before this rerun. Narrow mode now uses
the canonical 430px capture viewport and reduced motion disables the drawer
opening animation.

The final run passed and produced 15 browser assertions and seven capture
artifacts. The capture runner used the repository's existing browser and
`captureShot` primitives; its local stub server used no application API.

## Capture manifest

Artifacts are intentionally unpromoted and remain under the ignored capture
directory:

- `docs/screenshots/artifacts/presets/llamacpp-local--card-dark.png`
- `docs/screenshots/artifacts/presets/llamacpp-local--drawer-default-dark.png`
- `docs/screenshots/artifacts/presets/llamacpp-local--drawer-low-vram-diff.png`
- `docs/screenshots/artifacts/presets/llamacpp-local--drawer-disabled-reason.png`
- `docs/screenshots/artifacts/presets/llamacpp-local--drawer-light.png`
- `docs/screenshots/artifacts/presets/llamacpp-local--drawer-narrow-reduced-motion.png`
- `docs/screenshots/artifacts/presets/llamacpp-local--drawer-dense-no-change.png`
- `docs/screenshots/artifacts/presets/phase-4-5--receipt.json`

## Automated checks

- `rtk proxy node --check tmp/phase-4-5/drawer.js` — exit 0
- `rtk proxy node --check tmp/phase-4-5/run.mjs` — exit 0
- `rtk proxy node tmp/phase-4-5/run.mjs` — exit 0
- Prototype assertions — 15 passed
- `rtk npm run validate-preset-bundle-contract` — passed
- `rtk npm run validate-js` — passed
- `rtk npm run lint` — passed
- `rtk git diff --check` — passed after receipt creation

## Human gate

The plan requires a person other than the implementer to open the drawer,
change quantization and context, apply Low VRAM, read the diff, and reach a
launch decision without being coached through the plan. The automated browser
pass is evidence for the interaction contract but is not that attestation.

Required outcome remains one of:

1. contract confirmed, proceed to Phase 5;
2. contract confirmed with exact amendments to the architecture; or
3. contract rejected with a reason.

Do not advance Phase 5 or treat this receipt as a completed UX gate until an
independent human records one of those outcomes.

The user reviewed the corrected Low VRAM, MoE/VRAM-buffer, dense no-change,
narrow bottom-sheet, and reduced-motion states and confirmed the contract.
