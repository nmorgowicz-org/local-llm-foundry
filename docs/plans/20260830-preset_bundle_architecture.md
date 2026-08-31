# Preset Bundle and Launch-Option Architecture Contract

Status: binding architecture decisions frozen; implementation remains gated by
Phase 0 evidence and Coordinator approval

Source baseline commit: `9332965956ef6d3dc5399ab2967e999b1b7d4669`

The source baseline identifies the code audited while writing this contract. It
does not have to equal the later commit that contains these planning documents.
Every execution receipt separately records `plan_commit` and
`phase_start_commit`; validators compare each field to its stated purpose and
must not treat the source baseline as the current branch head.

Date: 2026-08-30

Companion execution plan: `docs/plans/20260830-preset_bundle_execution.md`

## 1. Purpose

Local LLM Foundry currently treats each `ModelPreset` as both:

1. a welcome-screen card, and
2. one fully flattened llama.cpp launch configuration.

That forces a separate card for every model quantization, context size, K/V
cache choice, batch/ubatch pair, and MoE CPU-offload choice. This contract
separates the user-facing model card from the immutable configuration passed to
the launcher.

One bundled card represents one exact model/tune and may contain multiple
weight artifacts and curated launch choices. The backend resolves a saved or
one-shot selection into the existing flat launch configuration before any
process starts.

Examples:

- `Qwen3.8-27B-Brainwaves-WFH` Q4 and Q5 files belong on one card.
- Brainwaves WFH, Unsloth, ColdFusion, and Heretic remain separate cards even
  when they share the same Qwen architecture and parameter count.
- Context, main K/V policy, batch/ubatch, and `n_cpu_moe` are launch choices,
  not separate model identities.

## 2. Binding scope boundary

### 2.1 Welcome-card and Configure-drawer scope

Only the following axes belong on the bundled preset card or its Configure
drawer:

- exact model artifact / weight quantization;
- context size;
- backend-approved main K/V policy;
- explicit batch and ubatch pair;
- MoE CPU expert-layer placement (`n_cpu_moe`) when GGUF metadata proves the
  model is MoE;
- estimate, fit result, and exact/related/stale measured-memory evidence.

The drawer may offer intent helpers such as Quality first, Balanced, or Low
VRAM. An intent produces an explicit proposal. It never remains an unresolved
dynamic launch rule and never silently changes a saved selection at Start.

### 2.2 Spawn Wizard and Preset Editor scope

The following settings do not belong on the card. They are configured through
the existing canonical Spawn Wizard Guided/Pro control system and the Preset
Editor:

- multimodal projector selection and GPU offload;
- sampling and repetition settings;
- reasoning mode, effort, format, preservation, budget, and budget message;
- KV-unified policy, continuous batching, parallelism, threads, priorities;
- load, fit, cache, checkpoint, cache-reuse, and cache-idle behavior;
- image token limits;
- speculative decoding and MTP options;
- chat templates and template kwargs;
- network, security, metrics, and observability.

Guided and Pro remain two presentations of the same DOM controls, state, and
payload. No setting may receive separate Guided and Pro storage or serializers.

### 2.3 Explicitly deferred

The following are unavailable in this project phase:

- mixed main KV `q8_0/q4_0`;
- detecting or enabling `GGML_CUDA_FA_ALL_QUANTS` builds;
- custom llama.cpp compilation or integration with `../llama-local-tooling`.
  The separate `llama-fit-params` estimate probe is not part
  of this deferral: it is an estimate-class tool that never gates or selects
  launch, and its absence degrades one drawer button, never a valid launch;
- any Metal experiment intended to enable mixed `q8_0/q4_0`;
- deriving measured VRAM from filename suffixes such as `27G` or `30G`;
- blindly copying personal flags such as `--tools all -ag` or explicit
  `--kv-offload` without a separate product and compatibility decision.

## 3. Approved Option B user interface

### 3.1 Compact card

This is the binding welcome-screen direction. It preserves the current grid
density and keeps many models visible.

```text
┌─────────────────────────────────────┐
│ Qwen3.8 27B · Brainwaves WFH        │
│ Quality first                       │
│ [Q4_K_M] [200k] [q8/q8]             │
│                                     │
│ MEM  ███████████████░░  26.3 GB     │
│       Tight · est. vs 27.8 GB free  │
│                                     │
│ [Configure] [⋯]           [▶ Start] │
└─────────────────────────────────────┘
```

Binding behavior:

- The collapsed card always shows the saved default selection.
- `Start` launches that exact saved selection after backend resolution and
  validation. Card Start is an atomic resolve-and-launch: it sends the preset
  ID plus `expected_revision` only, no one-shot selection and no config
  hash. A revision mismatch is 409; a binary that no longer supports the
  selection is 422 `capability_changed` with a fresh safe preview the card
  renders. The consent hash (`cfg-v1:`) is a drawer-only
  mechanism for draft launches.
- Card chips are summaries, not independent unvalidated dropdowns.
- `Configure` opens the drawer and moves focus into it.
- The overflow action retains direct `Edit full preset` access. The Configure
  drawer also links to the full Preset Editor. Bundle axes do not replace the
  editor-only runtime controls.
- A legacy preset without a bundle renders through a one-artifact adapter and
  retains its existing Edit/Start behavior until converted.

### 3.2 Configure drawer

```text
┌ Configure Brainwaves WFH ────────────────────────────┐
│                                                      │
│ What matters most?                                   │
│ [Quality first] [Balanced] [Low VRAM]    (Custom)    │
│                                                      │
│ Model quantization                                   │
│ ● Q4_K_M   17.8 GB     ○ Q5_K_M   21.2 GB            │
│                                                      │
│ Context                                              │
│ [160k] [200k] [262k] [Custom: ______]                │
│                                                      │
│ KV quality                                           │
│ ● Quality-first     q8_0/q8_0                        │
│ ○ Lower KV memory   q4_0/q4_0  ⚠ tool-call risk      │
│ ⊘ Mixed             q8_0/q4_0                        │
│   Needs a build with GGML_CUDA_FA_ALL_QUANTS         │
│                                                      │
│ Performance                                          │
│ [Conservative] [Balanced] [Throughput]   (Custom)    │
│ Current: batch 2048 · ubatch 256                     │
│                                                      │
│ Expert placement · MoE model                         │
│ [All GPU] [Fit automatically] [Custom: 12 / 64]      │
│                                                      │
│ Workload                                             │
│ [Agentic / tool use ▾]                               │
│                                                      │
│ Predicted result                                     │
│ Weights 17.8 · KV 5.1 · Compute 3.4 = 26.3 GB        │
│ Fits after freeing cache · expected quality: High    │
│ ⚠ 12 expert layers on CPU — slower generation        │
│                                                      │
│ Changed from saved:                                  │
│ Q5→Q4 · 262k→200k · ubatch 512→256                   │
│ (applied by “Low VRAM”)                              │
│                                                      │
│ [Edit full preset…]                                  │
│ [Reset]  Start without saving  [Save] [Save & Start] │
└──────────────────────────────────────────────────────┘
```

Binding behavior:

- Drawer edits are draft state until an explicit action.
- `Start once` passes the draft selection to the authenticated spawn route but
  does not mutate the preset.
- `Save` persists the selection with optimistic revision checking.
- `Save & Start` persists successfully first, then starts the exact returned
  revision. A failed save must not launch a different or stale selection.
- Every resolver change is listed. Low VRAM may not hide changes behind one
  label.
- Invalid combinations always render disabled with the backend-provided reason.
  They are never hidden. An option unavailable today may become available later
  with no UI change — mixed `q8_0/q4_0` becomes selectable once a build
  advertising `GGML_CUDA_FA_ALL_QUANTS` is integrated through
  `../llama-local-tooling` — so the disabled state plus its reason is the
  contract, and hiding the option would erase that affordance.
- Each disabled option associates its reason programmatically
  (`aria-describedby`), not by visual proximity or hover alone.
- The frontend does not reproduce compatibility formulas.
- Closing a drawer with unsaved edits must not silently discard them. Escape,
  backdrop click, and the close control all check `dirty` first and confirm.
  Explicit `Reset` is the only zero-friction path back to the saved selection.
- `Custom` is a derived state, not a selectable intent. It renders as an
  indicator when the draft matches no intent, and it is never a button. This
  applies to the intent row and the Performance row. Where `Custom:` introduces
  an input (Context, Expert placement) it remains a real control; the two
  meanings must stay visually distinct.
- The footer ranks its actions: `Save & Start` is primary, `Save` is secondary,
  and starting without saving is demoted and labelled by what it does rather
  than named `Start once`.
- The diff block reports every divergence from the saved selection, not only
  changes an intent produced. When an intent produced them it is named as the
  cause. This makes `Reset` legible and manual edits as reviewable as proposals.
- A proposal that moves MoE expert layers to CPU must state that generation
  gets slower. The magnitude is deliberately not estimated: it varies with
  model size, host RAM and GPU bandwidth, and system load, and the project has
  no measurement that would make a number honest. The warning is qualitative
  and always shown.
- The MoE placement row binds as follows. `Fit automatically`
  runs the bounded `llama-fit-params` probe search server-side and returns a
  draft proposal (never auto-applies); with the probe unavailable it renders
  disabled with the backend reason. `Custom: n / N` accepts any integer in
  `[0, moe_layer_count]` and round-trips as a custom value.

  The proposal is a starting point, not a destination. A user who wants
  headroom beyond the reserve — to leave room for a second process, a
  display, or simple caution — must be able to keep moving the needle and
  *see what it buys*. So a custom `n` is measured, not merely bounds-checked:
  changing it issues a single probe at that value (not a search) and the row
  reports the resulting device total and headroom against the budget. This
  reuses the drawer's existing coalescing and cancellation semantics; a
  single probe returns near-instantly, so the number tracks the control.
  With the probe unavailable, the row falls back to bounds validation with
  no number rather than showing a stale or inferred one.

  The row also exposes the headroom target directly. Raising it re-runs the
  same two-sided search against the larger reserve and proposes the new
  minimal `n`, which is strictly better than hand-stepping: it preserves the
  minimality guarantee at the headroom the user actually asked for. Manual
  stepping remains available for full control.

  Both the measured custom value and the headroom-target proposal are
  estimate-class. Neither gates a launch, and neither is a performance
  prediction — the qualitative slowdown warning above still applies, and
  applies more strongly the further the needle moves.
- The Workload row exposes `workload_policy`: a dropdown of
  the four known values, part of the draft, shown in the diff block. It
  changes aggressive-KV eligibility only through the resolver; it never
  changes the stored selection of other presets.
- The drawer copies the accessibility lifecycle from `evidence-drawer.js`:
  dialog semantics, backdrop, Escape, focus trap, focus restoration, narrow
  bottom sheet, light theme, and reduced-motion support.

The drawer owns module-local draft state with this minimum shape:

```text
bundleId
serverRevision
savedSelection
draftSelection
normalizedPreview
dirty
previewRequestGeneration
previewAbortController
opener
```

`sessionState.presets` remains the saved card source of truth. Opening copies
saved state into a draft; closing discards the draft. Resolve requests capture
a monotonically increasing generation and only the newest generation may
update the preview. Save replaces saved state only from the server-returned
preset/revision through the canonical preset reload path. Start uses the saved
revision, Start once uses the normalized draft returned by resolve, and Save &
Start uses the exact revision returned by PATCH.

### 3.3 Advanced comparison reference

This matrix is retained as a reference for a later advanced comparison inside
the drawer. It is not the Spawn Wizard Pro view and is not required for the
first card release.

```text
Configure Brainwaves WFH                     VRAM available: 27 GB

                     CONTEXT
MODEL QUANT       160k          200k          262k
────────────────────────────────────────────────────────
Q4_K_M          24.1 GB ✓     26.3 GB ◐     30.2 GB ✕
Q5_K_M          27.5 GB ◐     29.7 GB ✕     33.6 GB ✕
Q6_K            31.0 GB ✕     33.2 GB ✕     37.1 GB ✕

Selected: Q4_K_M × 200k

KV policy       [Quality-first q8/f16 ▾]
Batch/ubatch    [2048] / [256]
MoE CPU layers  [12 / 64]
────────────────────────────────────────────────────────
Estimate: 26.3 GB · Tight
Measured: no exact receipt yet

[Validate this configuration]                    [Start]
```

All numbers in the ASCII render are illustrative, not benchmark evidence.

## 4. Terminology and identity

### Preset bundle

One exact model/tune card plus its compatible weight artifacts, allowed launch
choices, and saved default selection.

### Model artifact

One exact local GGUF file plus optional Hugging Face origin provenance. Local
files are the primary launch source. Identity includes:

- canonical local path;
- file size;
- parsed quantization;
- bounded GGUF metadata digest or stronger content digest when available;
- architecture metadata used by the estimator;
- exact recorded download provenance or a separately confirmed HF-origin
  suggestion;
- compatibility links to required mmproj/draft assets.

Filename parsing may suggest quantization or grouping. It is never the
authoritative identity when GGUF metadata is available.

The existing `DownloadProvenance` sidecar is authoritative when the app
downloaded the file: it already records `repo_id`, exact `remote_path`, resolved
commit when supplied by Hugging Face, byte size, and download time. For other
local files, the existing `hf_resolve_origin(filename, size_bytes)` score is a
ranked suggestion, not identity proof. Extend its candidate DTO to retain the
matched remote file path and matched byte size that the resolver already
computes. A confident suggestion may be preselected, but bundle conversion
still requires explicit user confirmation. The score and reason are retained
as provenance; they never silently merge cards.

### Bundle membership

Membership is explicit and user-confirmed. The product must never merge cards
solely by display name, filename stem, model family, architecture, parameter
count, or quant-stripped filename.

### Bundle selection

The explicit artifact, context, KV policy, batch/ubatch pair, and MoE placement
saved as the card default or supplied as a one-shot override.

### Resolved launch

An immutable ordinary `ModelPreset`/`ServerConfig` projection plus the hashes
and launch guard defined in section 7. The existing launcher consumes this
projection; it does not know how to choose bundle options.

## 5. Target data contract

The current `ModelPreset` schema version is 4. The target uses two explicit
migration steps:

- v4 to v5: consolidate canonical K/V fields and validation;
- v5 to v6: add optional bundle and newly typed llama.cpp settings.

The existing flat `ModelPreset` remains the runtime compatibility object. Add
top-level `revision` and optional `bundle` fields to it. `revision` belongs to
the whole preset, not the bundle, so editor PUT, bundle selection PATCH, delete,
copy, and conversion share one concurrency token.

The following names and JSON fields are binding. Phase 0 writes fixtures for
them; it does not redesign them:

```rust
// New fields on ModelPreset.
#[serde(default = "default_preset_revision")]
pub revision: u64,
#[serde(default, skip_serializing_if = "Option::is_none")]
pub bundle: Option<PresetBundleSpec>,

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetBundleSpec {
    pub identity: PresetBundleIdentity,
    pub artifacts: Vec<PresetModelArtifact>,
    pub context_options: Vec<u64>,
    pub kv_policy_options: Vec<LlamaKvPolicyId>,
    pub performance_options: Vec<PresetPerformanceOption>,
    pub cpu_moe_options: Vec<i32>,
    pub curated_selections: Vec<PresetBundleSelection>,
    pub allow_validated_custom: bool,
    pub workload_policy: PresetWorkloadPolicy,
    pub default_selection: PresetBundleSelection,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

pub struct PresetBundleIdentity {
    pub bundle_id: String,
    pub tune_id: String,
    pub display_name: String,
}

pub enum PresetArtifactRole {
    Weights,
    Mmproj,
    Draft,
    Unknown(String),
}

pub struct PresetHfOrigin {
    pub repo_id: String,
    pub remote_path: String,
    pub revision: Option<String>,
    pub evidence: PresetHfOriginEvidence,
    pub confidence: Option<f64>,
    pub reason: Option<String>,
    pub user_confirmed: bool,
}

pub enum PresetHfOriginEvidence {
    DownloadProvenance,
    ResolverSuggestion,
    UserEntered,
    Unknown(String),
}

pub struct PresetArtifactDigest {
    pub algorithm: String,
    pub value: String,
    pub coverage: PresetDigestCoverage,
    pub provenance: String,
}

pub enum PresetDigestCoverage {
    FullFile,
    BoundedGgufHeader,
    Unknown(String),
}

pub struct PresetArtifactQuantization {
    pub value: String,
    pub provenance: PresetQuantizationProvenance,
}

pub enum PresetQuantizationProvenance {
    GgufMetadata,
    FilenameHint,
    UserConfirmed,
    Unknown(String),
}

pub struct PresetArtifactMetadata {
    pub gguf_architecture: Option<String>,
    pub model_kind: PresetModelKind,
    pub block_count: Option<u32>,
    pub moe_layer_count: Option<u32>,
    pub native_context_limit: Option<u64>,
    pub metadata_digest: Option<String>,
}

pub enum PresetModelKind {
    Dense,
    Moe,
    Unknown(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetModelArtifact {
    pub id: String,
    pub role: PresetArtifactRole,
    pub display_name: String,
    pub local_path: Option<String>,
    pub hf_origin: Option<PresetHfOrigin>,
    pub size_bytes: Option<u64>,
    pub digest: Option<PresetArtifactDigest>,
    pub quantization: PresetArtifactQuantization,
    pub metadata: PresetArtifactMetadata,
    pub mmproj_artifact_id: Option<String>,
    pub draft_artifact_id: Option<String>,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetBundleSelection {
    pub artifact_id: String,
    pub context_size: u64,
    pub kv_policy: LlamaKvPolicyId,
    pub performance_id: String,
    pub n_cpu_moe: Option<i32>,
    pub intent_source: Option<PresetFitIntent>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PresetPerformanceOption {
    pub id: String,
    pub label: String,
    pub batch_size: u32,
    pub ubatch_size: u32,
}

pub enum LlamaKvPolicyId {
    F16F16,
    Q8Q8,
    Q4Q4,
    MixedQ8Q4,
    Unknown(String),
}

pub enum PresetFitIntent {
    QualityFirst,
    Balanced,
    LowVram,
    Unknown(String),
}
```

`LlamaKvPolicyId` names the main K/V pair as a single policy rather than two
free-form strings, so an unsupported pair is unrepresentable rather than merely
rejected. `MixedQ8Q4` is the capability-gated member described in section 6;
it is a legal stored value and a non-launchable one until a build advertises
support. `PresetFitIntent` mirrors the drawer's intent helpers and is
explanation provenance only — it never enters the selection hash (section 7)
and never substitutes for the durable `PresetWorkloadPolicy`. Phase 0 freezes
both value sets.

Because `PresetBundleSelection` doubles as the `PATCH /selection` request
body, the server strips any client-supplied `intent_source` on all mutation
writes: it is stored as `None` and only ever appears in a resolve response
when that same request computed the proposal. It is display
provenance for the change block, not stored state.

All `Unknown(String)` variants use the same bounded custom string-enum serde
helper: known values deserialize to typed variants; any other string is retained
verbatim, reserializes unchanged, and is non-launchable until supported. Every
persisted struct uses `#[serde(default)]` plus a bounded flattened extension map
so unrelated edits preserve future fields. Phase 0 freezes maximum string,
array, and extension-map sizes and includes unknown-value round-trip fixtures.

Independent axis lists are catalogs, not permission to synthesize a Cartesian
product. `curated_selections` records known operator-authored combinations.
`allow_validated_custom` is an explicit bundle policy: when false, only curated
selections may resolve; when true, the resolver may accept a recombination only
after metadata, capability, context, K/V, batch, MoE, companion, and memory
validation. A synthesized combination is never called measured or proven.

Only `Weights` artifacts may be selected. Companion IDs resolve inside the same
bundle and must have the required `Mmproj` or `Draft` role. Artifact IDs are
unique across all roles. Every weights artifact inherits the bundle's exact,
user-confirmed `tune_id`; a different tune requires a different bundle.

A bounded-header digest is sufficient for metadata provenance but never for an
`exact` runtime observation. Exact evidence requires a full-file content digest.
A filename quantization hint is provisional and cannot override GGUF metadata.
Local paths are returned only by the authenticated full editor response; card
and resolve DTOs expose artifact IDs, safe labels, availability, and redacted HF
origin views.

`PresetWorkloadPolicy` is backend-owned and persisted. At minimum it
distinguishes `agentic_tools`, `general_chat`, `roleplay_creative`, and
`custom_unknown`. It is part of the selection fingerprint. Aggressive K/V may
be eligible only for a qualified policy or an explicit custom confirmation;
`intent_source` is explanation provenance and is not a substitute for this
durable policy.

The policy's user surface is the Configure drawer's Workload row and the
Preset Editor field. The Spawn Wizard has no workload
control: wizard-created presets are flat and carry no policy; conversion
later assigns `CustomUnknown`.

Implement workload policy with the same bounded open-string enum helper:

```rust
pub enum PresetWorkloadPolicy {
    AgenticTools,
    GeneralChat,
    RoleplayCreative,
    CustomUnknown,
    Unknown(String),
}
```

An unknown stored policy round-trips but cannot authorize aggressive K/V.

### Flat compatibility projection

When `bundle` is present, these top-level `ModelPreset` fields are derived from
`default_selection` and are not an independent source of truth:

- `model_path` from the selected local weights artifact;
- `context_size`;
- canonical `ctk` / `ctv`;
- `batch_size` / `ubatch_size`;
- `n_cpu_moe`.

`materialize_default_projection()` generates them for API responses and for
launch, not at the persistence boundary. Persistence stores the bundle; the flat
projection is materialized on read. This keeps ordinary save and migration pure
in the sense section 7 requires — no runnable binary — while still letting the
projection take the same `CapabilitySnapshot` the resolver takes, so it stays
equal to `resolve_preset(preset, None, caps)`. API writes containing a bundle
and conflicting flat fields return a 400 field error. They are never silently accepted as a second configuration.

Bundle launch is local-first. A selected artifact must have an existing,
canonical local path owned by the configured model inventory. A bundle may
retain an HF origin for source links, recovery, and download, but it never
projects an ambiguous bare `hf_repo` into the launcher. If the local file is
missing, resolve returns `artifact_not_local` with the confirmed repo/file/
revision needed by the existing download workflow; after download and inventory
adoption, resolution projects the new local path. Legacy flat `hf_repo` presets
retain their current behavior but are not used as the v6 exact-artifact model.

Legacy v4/v5 presets have no bundle and remain valid single-artifact entries.
They are not automatically combined. Conversion to a multi-artifact bundle is
an explicit, transactional user action.

If a legacy flat preset uses zero/null batch or ubatch as the existing
omit/runtime-default sentinel, conversion must either ask the user to select a
concrete bounded nonzero performance option or resolve a separately modelled
`RuntimeDefault` option from exact binary default evidence. It may not copy zero
into an explicit bundle performance option and call it reproducible.

### Conversion defaults

A converted bundle is fully determined, not left to implementer inference
:

- identity: fresh `bundle_`/`tune_`-prefixed IDs; display name from the preset.
- artifacts: one `Weights` artifact from the flat fields (inventory GGUF
  metadata when present, else filename hint, else `Unknown`); existing
  `mmproj`/`draft_model` values are adopted as `Mmproj`/`Draft` companions.
- `context_options`: the saved context plus the model's native context limit
  when metadata knows it; deduplicated, ascending.
- `kv_policy_options`: the named policy ID for the stored pair when it is
  `F16F16`/`Q8Q8`/`Q4Q4`; otherwise empty, with the selection carrying the
  verbatim `Unknown(pair)` value — readable, persisted, non-launchable until
  edited. A stored `q8_0/q4_0` never becomes a `MixedQ8Q4` option; it stays a
  non-launchable stored value (invariant 7).
- `performance_options`: one option labelled with the stored nonzero pair, or
  the standard set (512/512, 2048/256, 2048/512, 4096/4096) plus a
  capability-backed `RuntimeDefault` with a required choice when the flat
  pair is zero/null.
- `cpu_moe_options`: `[0]` plus the stored value for proven MoE when nonzero
  and within bounds; otherwise empty.
- `curated_selections`: exactly one — the converted default selection.
- `allow_validated_custom`: `true` (a flat preset previously accepted
  arbitrary editor values; conversion must not narrow that).
- `workload_policy`: `CustomUnknown`.

### Go-forward migration and recovery

This is a forward-only product migration. New v5/v6 data is not supported by
older application builds, and no engineering effort is spent making those
builds preserve or edit bundle fields. Before the first automatic v4-to-v5
write, the new build creates one non-overwriting recovery copy beside the live
file named `presets.pre-v5.<UTC timestamp>.json`, fsyncs it, and records its path
in the migration receipt. Failure to create that copy aborts migration and
leaves the live file untouched. The backup is disaster recovery, not a
downgrade format; restore is an explicit documented operator action.

From v5 onward, loading a file whose top-level schema version is greater than
`PRESET_SCHEMA_VERSION` is read-only and fail-closed: preserve the raw file,
show an upgrade-required error, and never rewrite it. The Phase 8 render flag is
only a presentation rollback and is never described as schema recovery.

New preset, bundle, and artifact IDs use `getrandom` and lowercase hex with
stable prefixes (`preset_`, `bundle_`, `artifact_`). Existing legacy IDs remain
unchanged. Selection APIs accept artifact IDs, never client-supplied local
paths. Artifact creation adopts paths returned by the authenticated model
inventory/file-browser flow, canonicalizes them server-side, and requires them
to remain inside configured model roots. Manual editor paths receive the same
canonicalization and root-membership check; `..`, symlink escape, and
cross-platform path fixtures fail closed.

## 6. Canonical K/V policy

Today the schema contains two divergent K/V pairs:

- `ctk` / `ctv`, which the launcher actually emits;
- `cache_type_k` / `cache_type_v`, which some documentation and Doctor logic
  inspect but the launcher does not emit.

v5 makes `ctk` / `ctv` canonical.

Migration rules:

1. If canonical fields are empty and deprecated fields are populated, copy the
   deprecated values into canonical fields.
2. If both pairs are populated and equal, retain the canonical pair.
3. If both pairs are populated and conflict, preserve the file but mark the
   preset non-launchable with an actionable repair error. Do not guess.
4. During one compatibility release, API output may project deprecated fields
   from canonical values; no code may read them for decisions.
5. Doctor, estimator, Calibration, imports, editor, wizard, bundle resolver,
   direct spawn, and preset spawn all call the same K/V policy validator.

The backend owns named policies and exact pairs. Initial policy eligibility is
qualified by workload and runtime capability. Mixed main K/V is gated on a
separate trusted-build capability, never inferred from `--help`:

```text
q8_0 / q4_0 -> rejected unless the exact binary advertises mixed K/V support
```

`CapabilitySnapshot` contains `mixed_main_kv`, whose value includes
`supported`, `reason`, and `source`. The only shipped provider sets
`supported=false`, `source=product_default_denied`, and a stable explanation.
The bounded help probe must never set this field: separate `-ctk` and `-ctv`
value lists do not prove the fused-kernel build property. A future
`../llama-local-tooling` build manifest may set it only after the manifest
format, trust policy, and exact executable SHA-256 binding are separately
contracted. This keeps the UI capability-driven without inventing a detector.

Mixed K/V therefore renders **disabled with the backend-provided reason**, per
invariant 17. It is present in UI option payloads and never hidden. The reason
must state that the option needs a source-compiled build advertising the
capability, so an operator running such a build outside the product understands
why the product will not launch it. Existing presets deserialize but cannot
launch until edited.

`extra_args` may not reintroduce `-ctk`, `-ctv`, or aliases that override typed
K/V fields.

`q4_0/q4_0` remains an explicitly risky, workload-gated choice. Guided
agentic/tool presets may not select it silently.

The Preset Editor's `ctk`/`ctv` controls are dropdowns, not free text.
Option order is common first — `f16`, `q8_0`, `q4_0` — then a separator, then
the remaining values advertised by the exact binary's capability probe
(e.g. `f32`, `bf16`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`); the list is
capability-sourced per binary, never hard-coded. A stored
value the selected binary does not advertise renders present but disabled with
the backend reason. Valid values outside the named policies round-trip
verbatim through read/unrelated-edit/save using the same unknown-string
handling as the enums, remain non-launchable until the validated policy
accepts them, and never count as migration conflicts.

### Prompt-cache RAM is platform-conditional

`-cram` / `--cache-ram` sets the prompt-cache ceiling in MiB (`--help`: default
`8192`, `-1` no limit, `0` disable). Policy:

```text
macOS   -> always 0
Windows -> operator choice; 8192 default, 16384 typical for high context
Linux   -> operator choice
```

For newly created or explicitly converted v6 bundles, the server-owned bundle
constructor stores `cache_ram_mib=Some(0)` and disables cache-idle slots on
macOS. The control renders disabled with the platform reason and resolve preview
shows the exact argv. Legacy flat presets retain their stored behavior until the
operator explicitly converts or saves the policy; direct legacy Start is never
silently changed. Production code obtains the current platform and passes it to
a pure `resolve_cache_policy(TargetPlatform, ...)` helper so macOS, Windows, and
Linux branches are unit-testable on any host.

### `--fit` re-resolves intent and must stay off

`--fit [on|off]` (default `on`) adjusts **unset** arguments to fit device
memory, shrinking context toward `--fit-ctx` (default `4096`) while reserving
`--fit-target` MiB per device (default `1024`). It never touches an argument
that was explicitly set — which is why operators who pin context observe no
effect from toggling it.

That default is nonetheless wrong for this product. Invariant 4 is "no
unresolved intent at launch time"; `--fit on` is the binary re-resolving intent
after the resolver already decided. Fields deliberately left omitted — v4
`null`/zero batch fields retain omitted-argv behavior — are exactly what it
adjusts, so a launch can differ from the argv the estimator and receipt
describe. Measurement paths must therefore pin `--fit off`, and any bundle
leaving fit at default must not be reported as exact runtime evidence.

The default therefore changes forward-only. The single server-owned v6 bundle
constructor sets `fit_enabled: Some(false)`, so every wizard, editor conversion,
or bundle-copy surface receives the same `--fit off` default. Existing flat
presets and Quick Load retain their stored/current `null` behavior. Do not
change the global `wizardState.hardware.fitEnabled` default or
`models.js::doQuickLoad`; neither is a bundle-only constructor.

Both fit and cache-RAM policy are go-forward bundle defaults. Neither silently
rewrites or changes a legacy flat launch.

## 7. Resolver and validation boundary

All launch paths use two explicit validation layers and one resolver:

```text
ModelPreset + optional one-shot selection
    -> structural/product validation (pure; no executable required)
    -> validate bundle membership and revision
    -> resolve named policies to exact values
    -> materialize flat ModelPreset
    -> runtime validation with an explicit CapabilitySnapshot/provider
    -> build canonical ResolvedLaunchManifest
    -> optionally enrich with estimate and evidence
    -> return internal ResolvedLaunch plus a separately redacted API view
```

At minimum, validation rejects:

- unknown or duplicate artifact IDs;
- an artifact not belonging to the preset bundle;
- invalid or prohibited K/V pairs;
- zero batch/ubatch inside an explicit bundle performance option, or
  `ubatch_size > batch_size` when both values are explicit/nonzero;
- negative `n_cpu_moe`;
- `n_cpu_moe` on a proven dense model;
- `n_cpu_moe` greater than the GGUF-derived MoE layer count;
- any nonzero `n_cpu_moe` when model kind or MoE layer count is unknown;
- context above supported/native limits unless an explicit qualified extension
  policy exists;
- missing model/mmproj/draft paths or incompatible companions;
- typed flags unsupported by the exact selected llama.cpp binary;
- safety-critical typed flags duplicated through `extra_args`.

Direct `/api/sessions/spawn` payloads and preset-ID launches must pass both
layers. Load, migration, and ordinary persistence use only pure structural
validation and never require a runnable local binary. Runtime validation fails
closed for an explicitly requested unsupported flag; an unavailable probe
returns an actionable degraded/capability error rather than an invented result.
UI validation is advisory only.

Estimate and evidence enrichment are advisory for an exact saved selection. A
valid local launch may proceed when estimate metadata or remote HF enrichment is
unavailable. Fit intents require enough estimate evidence to produce a proposal
and otherwise return `unavailable`; they never block a previously saved exact
selection merely because enrichment failed.

Resolve is bounded and constant-time with respect to artifact contents: it does
not parse GGUF, read arbitrary files, or call the network. It consumes metadata
and provenance already adopted into the preset/model inventory plus a supplied
capability snapshot. The route uses `safe_json_body`, bounded strings/arrays,
per-token rate limiting, a total timeout, and singleflight capability probing.
Frontend cancellation prevents stale presentation only; it is not a backend
resource-control or correctness mechanism.

### Three identifiers with three different jobs

Daily llama.cpp beta rebuilds are expected. They are why one overloaded
"fingerprint" is insufficient. The system uses three identifiers:

1. **Selection hash — `sel-v1:`.** Stable description of resolved user choices
   and non-secret policy. It excludes executable and hardware identity, so the
   same selection can be compared across builds and machines.
2. **Resolved configuration hash — `cfg-v1:`.** Consent guard proving that the
   effective, normalized configuration shown in Preview is the configuration
   accepted for Start. It includes selected artifact/companion identities,
   workload policy, and every behavior- or memory-relevant effective launch
   value. It excludes executable and hardware identity. A daily rebuilt binary
   therefore does not interrupt Start when the same config remains supported;
   the fresh capability check still blocks or refreshes when support changed.
3. **Evidence fingerprint — `evidence-v1:`.** Phase 9 measurement identity. It
   includes the selection hash, full artifact digest, exact executable/build,
   OS/hardware, measurement method, and workload concurrency. A daily rebuild
   intentionally makes old evidence non-exact while leaving it available as
   related historical evidence.

All three derive from one `ResolvedLaunchManifest`. Every resolved field is
classified `memory-relevant`, `behavior-only`, `secret/redacted`, or
`forbidden`; an unclassified field is a contract-test failure. Workload policy
is included even though it is not argv because it changes resolver eligibility.
`intent_source`, revision, timestamps, and secrets do not enter the selection
hash.

Canonical encoding for `sel-v1` and `cfg-v1` is JSON over an ordered vector of
typed triples `[path, type, value]`, with paths sorted byte-wise ASCII. Values remain JSON
booleans/numbers/strings/arrays, so strings and list elements are escaped and
cannot collide through newline or comma text. `None` is omitted. Adding a new
optional field that remains absent does not change an existing hash; explicitly
setting a default does. SHA-256 is rendered as the full lowercase hex digest
with the version prefix. Do not use `DefaultHasher`, a derived `Hash`, an
unordered map, or an unescaped `path=value` format.

The server always obtains the current capability snapshot and re-resolves before
process creation. A UI action based on a prior preview sends
`expected_revision` and `expected_resolved_config_hash`. Revision mismatch
returns 409. A changed valid manifest returns 412 `preview_stale` with the new
safe preview. If a new daily binary no longer supports the selection, return 422
`capability_changed` with the reason. None of those responses stops an existing
server or creates a process/session. Only after guarded preflight succeeds may
replacement/stop coordination begin. A direct admin launch with no prior
preview may resolve and launch atomically and returns all identifiers.

## 8. API contract

Keep authenticated `GET /api/presets` and `GET /api/presets/{id}` as full editor
responses. Add `GET /api/preset-cards`, requiring `api-token`, returning only
`PresetCardView` values: preset ID/revision, labels, safe artifact summaries,
saved selection, fit/estimate/evidence summaries, and no local paths or secrets.
`sessionState.presetCards` becomes setup-card truth; `sessionState.presets`
remains editor truth.

`ResolvedLaunch` is internal and deliberately does not derive `Serialize`.
`POST /api/presets/{id}/resolve` returns an explicit `ResolvePresetResponse`
containing normalized selection, final `ResolvedChange` records, tagged
estimate status, capability reasons, evidence summary, selection hash, launch
configuration hash, and revision. It contains no flat `ModelPreset`, `model_path`, artifact
local path, `api_key`, or secret-bearing argv.

The response types are frozen as:

```rust
pub struct ResolvedChange {
    pub code: String,
    pub field: String,
    pub before: Option<String>,
    pub after: String,
    pub explanation: String,
    pub source_policy: Option<String>,
}

pub enum EstimateStatus {
    Available { estimate: LaunchEstimate },
    Unavailable { code: String, message: String },
    NotApplicable { code: String },
}
```

The JSON fixture uses a tagged snake-case representation for `EstimateStatus`.
Messages are bounded display text; branching uses stable `code` values.
`LaunchEstimate` is the frozen exact response shape of the existing
`/api/vram-estimate` endpoint (weights/KV/extras/VRAM/RAM breakdown), frozen
into Phase 0's `api-target-fixtures.json` and named in Phase 4a;
no new estimate fields are invented for the bundle.

The mutation surfaces are fixed:

- `POST /api/presets`: create; server assigns secure ID and `revision=1`;
- `PUT /api/presets/{id}`: `{ expected_revision, preset }`;
- `PATCH /api/presets/{id}/selection`:
  `{ expected_revision, selection }`;
- `POST /api/presets/{id}/copy`:
  `{ expected_revision, new_name }`, creating a new ID/revision 1;
- `POST /api/presets/{id}/convert-to-bundle`:
  `{ expected_revision, conversion }`;
- `DELETE /api/presets/{id}`:
  `{ expected_revision, confirmation: "DELETE PRESET" }`;
- `POST /api/presets/reset`:
  `{ expected_catalog_etag, confirmation: "RESET PRESETS" }`.

Create, ordinary PUT/PATCH, copy, conversion, card/list, and resolve require
`api-token`. Spawn, delete, and reset require `db-admin-token`; destructive
requests require the exact confirmation string. `catalog_etag` is SHA-256 over
the sorted `[preset_id, revision]` vector returned by list/card responses, so
reset detects concurrent catalog changes without another persisted counter.

`revision` is persisted to the preset file (it is part of `catalog_etag` and
must survive restart); a legacy v4 preset receives `revision = 1` at its first
v5 write — migration assigns, it does not increment.

Revision enforcement is phased: the new bundle routes
(`POST /selection`, `POST /copy`, `POST /convert-to-bundle`, `DELETE`,
`POST /reset`) require `expected_revision` from their introduction. The
existing `POST /api/presets` and `PUT /api/presets/{id}` accept their current
request shape until the Preset Editor is updated; from that commit onward a
`PUT` on any preset — flat or bundled — requires `expected_revision`. Every
response that changes a preset returns the new `revision` and a fresh
`catalog_etag`. Mismatch is 409 with the current server revision and etag;
state is unchanged.

The UI re-fetches the catalog immediately before prompting for the destructive
confirmation on reset and delete, and sends the freshly obtained
`catalog_etag`/`expected_revision`; a stale value at prompt time defeats the
guard.

All preset mutations use one `PresetStore::mutate` boundary. Under one mutation
lock it checks revision/etag, builds and validates a cloned candidate catalog,
durably replaces the file, then swaps in-memory state. Any validation, write,
fsync, or replace failure leaves memory and the live file unchanged. The
replacement helper has injected-failure and existing-destination tests on
macOS, Linux, and Windows. Client-supplied replacement revisions are rejected.

Extend `POST /api/sessions/spawn` with optional exact selection,
`expected_revision`, and `expected_resolved_config_hash`. The client never submits the
resolved flat preset as authority. The server re-resolves and records selection
hash, resolved configuration hash, capability identity, and evidence fingerprint
when present.

## 9. Spawn Wizard and Preset Editor field contract

### Already typed; repair disclosure/contract coverage

- `repeat_last_n`;
- `kv_unified` / `--no-kv-unified`;
- `no_cont_batching`;
- `swa_full`;
- `load_mode`;
- `verbosity`;
- `ctx_checkpoints`;
- `checkpoint_min_step`;
- `cache_reuse`;
- `cache_idle_slots` (editor/backend today; add to wizard);
- context, K/V, batch, ubatch, MoE, fit, cache, sampling, and image limits.

`spawn-wizard-groups.js`, `spawn-wizard-llama-ia.js`, state reset/restore,
payload builders, editor load/save, and the control-contract fixture must list
the same applicability or an explicit `not_applicable` reason.

### Add typed llama.cpp support

- `mmproj_offload: Option<bool>`
  - `None`: exact runtime default;
  - `Some(true)`: emit positive flag only when supported;
  - `Some(false)`: emit `--no-mmproj-offload` when supported.
- `llama_reasoning_effort`
  - enum: `default|minimal|low|medium|high|xhigh|max`;
  - separate from Rapid-MLX request-default `reasoning_effort`.
- `llama_reasoning_format`
  - `Option<LlamaReasoningFormat>` where `None` means runtime default/auto and
    emits no argument;
  - explicit enum values come from exact runtime capability evidence;
  - current observed explicit values: `none|deepseek|deepseek-legacy`;
  - never emit `--reasoning-format auto` unless a future exact binary advertises
    `auto` as an accepted explicit value.
- `llama_reasoning_preserve: Option<bool>`
  - requires exact binary flag support and a compatible reasoning mode;
  - current product code has no authoritative template contract for native
    preserve-reasoning support. Until a bounded template inspection rule and
    fixture are added, template compatibility is `unknown` and launch remains
    blocked with an actionable explanation;
  - distinct from `preserve_thinking` inside chat-template kwargs.

Guided uses friendly labels and backend-resolved defaults. Pro displays exact
native values and flag names. Both manipulate the same canonical controls.

## 10. Runtime capability contract

New runtime-dependent fields are gated by the exact executable identity and a
bounded `--help` probe. The existing capability snapshot is extended rather
than scattering substring checks across UI code.

Before using the cache, fix the current validity mismatch: generation stores a
hash of raw help text while lookup compares it with a hash of normalized
`serve_flags`. Add
`snapshot_for_binary(path, qualifications)`: derive canonical path plus full
executable SHA-256, return the cached snapshot only for that exact identity, or
run the bounded probe and cache the result. `help_hash` is stored probe evidence
and participates in the snapshot fingerprint; lookup never tries to recreate it
from parsed flags. Replacing a daily build at the same path changes its SHA and
forces a fresh probe. Probe failure is an actionable capability error for
capability-gated fields.

`BuildQualificationProvider` is a separate input for properties help cannot
prove. Its shipped implementation always returns unavailable for mixed main K/V
with the stable reason from section 6. A future qualified receipt must match the
exact executable SHA-256. Version strings, build counters, OS, path, and
"custom-compiled" heuristics are forbidden.

Typed capability evidence includes:

- flag supported;
- positive/negative form supported;
- accepted enum values when parseable;
- observed help default;
- exact executable path/hash/mtime/version/help hash;
- evidence timestamp and bounded-probe errors.

Production resolver/launch code consumes the returned snapshot. It must not
generate a snapshot and discard it before constructing the adapter.

When a preset contains a value unsupported by the currently selected binary,
the UI may not newly select that value and launch remains blocked with an
actionable error. Unrelated edits must still round-trip and preserve the stored
value unchanged; capability degradation is never permission to delete it.

Forward-compatible typed enums use bounded custom deserialization with an
`Unknown(String)` representation. The raw value round-trips through API reads,
unrelated edits, and saves, while runtime validation blocks launch. Plain
`#[serde(default)]` is not sufficient for a present but unknown enum string.

Help advertisement proves only argument availability. It does not prove a
CMake flag, KV behavioral safety, tool-call correctness, or performance.

## 11. Low-VRAM resolver policy

Low VRAM is a proposal algorithm, not a static switch. It operates in this
order and reports every change:

1. Prefer an explicitly lower-memory artifact in the same confirmed bundle.
2. Reduce context only to a listed/qualified option.
3. Reduce batch/ubatch while preserving `ubatch <= batch` and model-specific
   image-token constraints.
4. For proven MoE models on qualified discrete-memory systems, increase CPU
   expert-layer placement within metadata and system-RAM bounds. Unified-memory
   systems receive an unavailable reason until separate behavior is qualified.
   This restriction binds **intents** (Low VRAM) only: the drawer's explicit
   `Fit automatically` button is a user-initiated probe-backed proposal and is
   available wherever the probe runs, discrete or unified memory.
5. Preserve the workload's qualified K/V quality floor.
6. Offer aggressive K/V only as an explicit risky choice when workload policy
   permits it; never silently for agentic/tools.

The proposal becomes an exact draft selection. Start never reruns the intent
algorithm behind the user's back.

## 12. Estimate and measured-memory evidence

The drawer uses `/api/vram-estimate` and its existing weights/KV/extras/VRAM/RAM
breakdown. Estimator evidence labelled `Measured` means estimator constants are
measurement-backed; it is not proof that this exact configuration ran.

The card distinguishes:

```text
Estimated  26.3 GB  · current settings
Measured   27.1 GB  · this machine · exact · 2 days ago
Available  27.8 GB  · now
```

An `exact` receipt is only ever the viewer's own machine, because the
fingerprint includes hardware and OS. A receipt captured on another host can
reach `compatible` or `related` at best and must be labelled with that host;
it is never presented as this machine's exact measurement.

Availability is read live rather than frozen at save time. On discrete-VRAM
systems it moves little, but on unified memory it tracks whatever else the
machine is doing, so both the collapsed card's fit verdict and the drawer's
`Available` line are computed against current free memory and carry the moment
they were read. A fit verdict with no observation behind it says so instead of
implying a measurement.

Exact runtime evidence is fingerprinted by:

- model artifact digest;
- complete resolved selection and non-card launch settings;
- llama.cpp executable/build/capability identity;
- hardware and OS;
- measurement method;
- workload and concurrency;
- timestamp.

The fingerprint is derived from one canonical normalized resolved-argv
manifest. Every typed or allowed extra argument is classified as
memory-relevant, behavior-only, secret/redacted, or forbidden. Adding an argv
field without a fingerprint classification fails tests. Memory-relevant fields
include artifact and companions, mmproj identity/offload, image token bounds,
GPU layers/tensor split, K/V policy/unification/offload, load/fit/cache/SWA,
batch/concurrency, MoE placement, and all draft/speculative model/KV/placement
settings.

Evidence match classes reuse Calibration semantics:

- exact: safe to show as the primary measured value;
- compatible: visible only with a compatibility explanation;
- related: reference-only;
- stale: never displayed as current.

Measurement methods remain explicit:

- Windows/WDDM total-device before/peak/after delta;
- CUDA/ROCm per-process where the driver exposes reliable process accounting;
- Metal/unified-memory physical/system observations;
- `fit_probe`: bounded `llama-fit-params` estimate. Estimate
  class only — it feeds the drawer's predicted result and the MoE placement
  search, is never displayed as a `Measured` line, never enters
  `memory_peak_bytes`, and never upgrades an evidence match class;
- estimator-only.

The existing `CalibrationMeasurement.memory_peak_bytes` is currently not
populated for normal llama.cpp Calibration and cannot power the card until a
platform-specific sampler and receipts prove it. Filename `G` suffixes are
operator labels, not evidence.

### `llama-fit-params` fit probe

The probe is a separately built `llama-fit-params` executable, configured by
path in `AppConfig` (`llama_fit_params_path`) and absent by default. Its
absence disables `Fit automatically` with a backend-provided reason (invariant
17) and leaves custom MoE placement validated by metadata bounds only. It
never blocks a valid launch, never selects or gates one, and never sets
`mixed_main_kv`; it is not the deferred `../llama-local-tooling` build
integration of section 2.3.

It has **two distinct roles, and they must not be conflated**:

1. **Estimator input, for every architecture.** One invocation returns the
   `model`/`context`/`compute` split at the exact resolved configuration.
   This applies to dense models as much as MoE ones — a dense probe needs no
   search at all, because there is no placement to search over, and a single
   call *is* the answer.
2. **Placement search, for MoE models only.** The two-sided `n_cpu_moe`
   search described below, which exists only where `moe_layer_count` is
   present and non-zero.

Role 1 is the general case and role 2 is the special case. A dense model gets
the probe's estimate and no MoE placement row; the row's absence must not
suppress the probe.

#### Probe-backed estimates

The probe measures what the binary will actually allocate; `/api/vram-estimate`
computes what a formula predicts. Where both are available the probe is the
better number, and the estimator's role shifts from predicting the total to
correcting for what the probe cannot see.

- When the resolved configuration uses only probe-accepted flags, the probe
  result supersedes the formula total for the `Estimated` line, labelled as
  probe-backed.
- When the configuration includes flags the probe does not accept — mmproj,
  speculative draft model, cache-RAM, SWA, load mode, threads, tensor split —
  the probe total is a **floor**, and the estimator contributes only the
  deltas for those unaccepted components. The line states that it is a probe
  base plus estimated additions, and names which additions.
- With the probe unavailable or failed, the estimator behaves exactly as it
  does today. The probe is an upgrade path, never a dependency.

This is what makes a headroom target meaningful rather than a guess, and it
applies to dense models too: on a device that cannot hold a dense
configuration there is no offload lever, so the honest answer is the exact
shortfall and which of context, K/V quantization, or weight quantization
would close it — not a proposal.

#### The probe as a check on the estimator's math

Substituting the probe for the formula is the smaller win. The larger one is
that the probe gives the existing tensor/layer arithmetic something it has
never had: an independent second opinion from the binary that will do the
allocating.

Every probe result is therefore compared against the formula's prediction for
the same resolved configuration, and the signed divergence is recorded per
component — `model`, `context`, `compute` — not just on the total, because a
total can agree while two components cancel. Component-level divergence is
what localizes the error: a wrong `context` figure points at the KV
arithmetic, a wrong `model` figure at tensor-size or layer accounting.

Divergence is diagnostic, not user-facing alarm. Within tolerance the drawer
says nothing. Beyond it, the drawer shows the probe-backed number and notes
that the formula disagrees, rather than silently picking one — a disagreement
the user can see is worth more than a confident wrong number, and the
disagreement itself is the bug report.

The Phase 0 fixture corpus makes this a regression test rather than a
runtime-only signal: the estimator is run against every captured probe point
and its per-component predictions asserted within tolerance. That converts
each capture into a permanent assertion about the estimator's math, so a
future change to the KV or tensor arithmetic that drifts away from observed
reality fails a test instead of shipping. Tolerances are set from the initial
corpus and tightened as the estimator improves; they are never loosened to
make a failing build pass.

The estimator's own constants stay the source of truth for the `Measured`
line and for evidence classes. A probe-backed `Estimated` line is still
estimate class (`method = "fit_probe"`); it never becomes `Measured`, never
enters `memory_peak_bytes`, and never upgrades an evidence match class.

The probe binary carries the same identity requirements as the launch binary:
canonical path, full SHA-256, mtime, version line, and a bounded probe
timestamp. Results are cached per
`(artifact digest, resolved-config probe subset, probe SHA-256)`. A probe
binary of unknown provenance is not trusted to launch anything; it only
produces estimate-class numbers.

The probe runs server-side only, under a total timeout and bounded output
capture, with `--fit off` pinned and a fixed verbosity. Its inputs are the
subset of the resolved configuration it accepts: `-m` local weights artifact
path, `-c` context, the `-ctk`/`-ctv` main K/V pair (and `-ctkd`/`-ctvd` when
draft K/V is set), `-b`/`-ub` batch and ubatch, the `--n-cpu-moe` candidate,
and `-lm none -lv 4 -fitp on`.

Flags the probe does not accept — mmproj, speculative/draft model, cache-RAM,
SWA, load mode, threads, tensor split — are documented per version as an
accepted approximation and listed in the probe receipt. The probe therefore
produces a lower-bound-ish device estimate plus host rows,
not a faithful launch prediction. That is sufficient for placement search, and
it is the limitation the drawer states.

Output arrives in two forms that carry different rows:

- the two-line compact form on **stdout**, `<name> model context compute` in
  MiB for the device and `Host` rows only (for example `MTL0 15196 105 493`
  and `Host 515 0 137`).
- the full `common_memory_breakdown_print` table on **stderr** (the `-lv 4`
  log), which carries a `total`/`free`/`self` summary per device alongside
  the `model`/`context`/`compute` columns.

**Where the re-homed expert weights land is backend-dependent, and the parser
must not assume either shape.** Under Metal the table carries a dedicated
`CPU_REPACK` row and the `Host` row's `model` column stays flat. Under CUDA
there is no `CPU_REPACK` row at all; the same weights appear as growth in the
`Host` row's own `model` column (`515` MiB at `n_cpu_moe = 0` rising to
`8037` at `n = 16`). A parser keyed on the literal `CPU_REPACK` row would
therefore read a flat host budget on CUDA and accept placements the host
cannot hold.

The host budget is consequently defined as **the sum of every non-device row
in the stderr table**, whatever those rows are named. That definition is
correct under both backends and does not need revising when a third appears.
Both streams are captured separately, and a parse failure on either is an
actionable `probe_unavailable` state, never a guess.

The stderr table's `free` column reports live device memory at probe time
(`30927` of `32579` MiB on an otherwise-idle desktop). The probe budgets
against `total` minus the reserve, not `free`, so that a cached result stays
valid across unrelated changes in desktop GPU usage. When `free` is materially
below `total` the drawer surfaces that as a separate advisory; it never
silently shrinks the budget, because doing so would make identical inputs
produce different proposals.

The probe exists to automate a judgement the user should not have to make:
given their chosen quantization and context target, how many MoE layers (if
any) must be re-homed to host RAM for the run to fit. The answer is strongly
platform-shaped. On a unified-memory system a smaller quantization of a 35B
A3B-class model often fits entirely on device, and the correct proposal is
`n_cpu_moe = 0` with no `--n-cpu-moe` flag emitted at all. On a discrete GPU
— a 16 GiB RTX 5080 or a 32 GiB 5090 — the same model at a Q5_K_M target with
a 200k-262k context will not fit even with quantized K/V, and the useful
answer is the *smallest* offload that reaches the context target. The search
therefore proposes the feasible interval's lower bound, never a larger `n`
that also happens to fit.

Offload cost is advisory, not validated. The drawer states that throughput
degrades as more experts are moved to host RAM; the product makes no
performance guarantee at any `n_cpu_moe`, and the probe measures fit only.

On unified-memory platforms the MoE placement row is **not exposed**, pending
measurement. Device and host draw one physical pool there outside a small OS
reservation, so moving experts between them plausibly relabels memory rather
than freeing it — and the M5 Max corpus is consistent with that reading: the
device `model` column falls by exactly the `464` MiB per layer that
`CPU_REPACK` gains, leaving the sum invariant. On CUDA the same transfer
crosses genuinely separate pools and is real. Consistent is not proven, and
the control stays hidden rather than shipping a lever that may do nothing.
The probe's role-1 estimate is unaffected and still runs on these platforms.

The open experiment that would settle it is a model deliberately oversized for
the box under `-lm dio`, to observe whether placement changes spill to NVMe;
it is not scheduled and nothing here blocks on it. If it shows offload does
relieve real pressure, exposing the row is an additive change.

The placement search is two-sided. Device memory falls as `n_cpu_moe` rises
and host memory rises with it, so the feasible region is the interval between
the smallest `n_cpu_moe` the device accepts (a suffix, anchored at 0) and the
largest the host accepts (a prefix, anchored at 0). A single
binary search for "the smallest `n` that fits both" is invalid: it can probe
the over-offload tail, read false, and discard a feasible interval. The
proposal is the interval's lower bound. The reserve on both sides is the
product's existing per-device fit reserve — the `--fit-target` default of
1024 MiB from section 6 — so the probe applies the same conservatism the
binary's own fit mode would. Execution plan Phase 4c owns the algorithm, its
bounds, and its fixtures.

Probe results are estimate-class evidence with `method = "fit_probe"`. The
probe's *inputs* are memory-relevant because they derive from the resolved
configuration; the probe's *output* enters no `sel-v1:` or `cfg-v1:` hash,
and `evidence-v1:` fingerprints a probe result only when a Phase 9 receipt
explicitly records it as one.

## 13. Real-corpus findings that shape the design

The read-only Ryne inventory contained 26 q36/q38 launch files and established:

- contexts: 131072, 160000, 200000, 212992, 262000, 262144;
- batch/ubatch: 4096/4096, runtime-default/1024, 512/512, 2048/512,
  2048/256;
- `n_cpu_moe`: 0, 6, 8, 14, 16, 30;
- repeated exact tunes with Q4/Q5 artifacts;
- repeated exact artifact/context choices with different MoE placement;
- main q8/q8 and mixed q8/q4 pairs, where mixed q8/q4 remains excluded;
- names such as `I-Quality` and `I-Balanced` that cannot be treated as
  authoritative quant labels without GGUF introspection.

Therefore the bundle stores explicit resolved values. It does not reconstruct
settings from filenames, assumed runtime defaults, or card labels.

Two `llama-fit-params` reference corpora were captured under `-fit off -fitp
on -lm none -lv 4`, one per backend, and they agree on the physics while
differing in output shape.

On an Apple M5 Max with `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, across `--n-cpu-moe`
0/12/24/36, the device `model` column falls by `464` MiB per offloaded layer
while a dedicated stderr `CPU_REPACK` row rises by the same `464` MiB per
layer (`5568` at 12, `11136` at 24, `16738` at 36, absent at 0) and the `Host`
row's `model` column stays flat at `515` MiB.

On an RTX 5090 (32 GiB) with `Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf` at
`-c 200000 -ctk q8_0 -ctv q8_0 -ctkd q4_0 -ctvd q4_0 -b 2048 -ub 256`, across
`--n-cpu-moe` 0/4/8/12/16, the device `model` column falls by the same `464`
MiB per layer — the constant is model-determined, not platform-determined —
but there is no `CPU_REPACK` row and the `Host` row's `model` column carries
the growth instead (`515` → `2469` → `4325` → `6181` → `8037`).

Three properties of the CUDA sweep shape the search:

- **Enabling offload carries a fixed cost on top of the per-layer cost.**
  The `n = 0 → 4` step moves `1954` MiB where every later 4-layer step moves
  `1856`, and device `compute` jumps `515 → 677` once at `n > 0` and is then
  flat. Both sweeps sampled coarsely (steps of 4 and 12) purely to survey
  the curve; `n_cpu_moe` is a per-layer integer and the real search space is
  every value in `[0, moe_layer_count]`. The `98` MiB excess on the first
  step is therefore a one-time transition cost, not evidence of a larger
  first layer — the coarse sampling cannot distinguish the two, and the
  search does not need to. A linear model fitted from two probes
  mispredicts near zero, so the search relies on monotonicity in each
  direction and never on a fitted slope.
- **`ubatch` is a real but second-order device lever.** At `n = 0`, device
  `compute` runs `515`/`638`/`986` MiB for `-ub` `256`/`512`/`1024`, and
  host `compute` `102`/`203`/`407`. The full `256 → 1024` range is worth
  about `471` MiB of device memory — roughly one offloaded layer. `ubatch`
  is a user-chosen configuration value, so the probe holds it fixed at the
  resolved value and never searches over it; the drawer may note that
  lowering it is worth approximately one layer.
A third capture on the same 5090 covers a **dense** model,
`Qwen3.8-27B-...-Q4_K_M.gguf`, at the same `-c 200000` and K/V quantization.
It confirms the probe is architecture-agnostic: `model` `16518`, `context`
`6796`, `compute` `920` at `-ub 256` and `1333` at `-ub 1024`, host `790` and
`1113`. Two things follow. The `context` term is `6796` MiB here against
`2140` MiB for the 35B A3B at identical `-c` and K/V quantization, so KV
footprint is an architecture property that no per-parameter rule of thumb
predicts — precisely the quantity a formula estimator gets wrong. And the
dense configuration needs `24234` MiB at `-ub 256`, so on a 16 GiB card it
overshoots by `7850` MiB with no placement lever available; the useful output
there is the shortfall and its remedies, not a proposal.

- **The 16 GiB case is the design target, and it lands between the sampled
  points.** Extrapolating the sweep at `-ub 256`, a 16 GiB card with the
  `1024` MiB reserve (`15360` usable) is not satisfied at `n = 16`
  (`16093` MiB) or `n = 17` (`15629`), and is first satisfied at `n = 18`
  (`15165` MiB, host `10923`). A search restricted to the sampled multiples
  of 4 would answer `20` and offload two layers more than necessary. This
  is the concrete reason the proposal is the interval's lower bound over
  unit steps rather than the first sampled value that happens to fit, and
  the Phase 0 fixtures must carry unit-step points around the boundary
  rather than only the surveyed ones.

## 14. Non-negotiable invariants

1. One exact model/tune per bundle; no automatic cross-tune grouping.
2. One backend resolver and validator for every launch path.
3. No Cartesian product persisted as separate cards.
4. No unresolved intent at launch time.
5. No Guided/Pro state fork.
6. Wizard-created, editor-edited, reopened, and launched presets round-trip
   without loss.
7. `q8_0/q4_0` is rejected centrally and cannot return through `extra_args`.
8. Existing invalid presets are preserved and blocked, never silently changed
   or deleted.
9. Persistence succeeds before memory mutation or success response.
10. Estimation and exact runtime measurement are labelled separately.
11. All API/DB fields use `#[serde(default)]` with degraded behavior.
12. UI option availability comes from backend policy/capability responses, not
    duplicated client formulas.
13. UI changes require dark, light, narrow, and reduced-motion screenshot
    evidence from the current release build.
14. Legacy flat zero batch/ubatch retains its existing meaning: omit the argv
    and use the runtime default. Explicit bundle performance options must store
    nonzero resolved values.
15. Schema migration is forward-only and not downgrade-supported. A durable
    pre-v5 recovery copy is mandatory before the first migration write. Future
    schema versions are preserved read-only and never rewritten.
16. Schema version and card presentation are independently controllable. A
    startup setting `LLAMA_MONITOR_PRESET_BUNDLE_UI=legacy|bundled` defaults to
    `bundled`, is parsed server-side into `AppConfig`, and is exposed to the UI
    only as a safe enum. `legacy` forces one-artifact card rendering even when
    v6 bundles exist, without changing preset data. Tests inject the config;
    localStorage and ad hoc window globals are forbidden.
17. Unavailable options render disabled with a reason and are never hidden.
18. A drawer with unsaved edits cannot be dismissed silently by Escape,
    backdrop click, or the close control.
19. Fit verdicts are computed against currently available memory, not memory
    observed when the selection was saved.
20. Every resolved launch field has a fingerprint classification; an
    unclassified field fails the contract validator.
21. A guarded preflight succeeds before any existing server is stopped or any
    new process/session is created.
22. A valid local artifact remains locally launched and launchable offline;
    optional HF-origin scoring or enrichment never becomes launch authority.
23. Card Start is an atomic resolve-and-launch with `expected_revision` only;
    the `cfg-v1:` consent hash is a drawer-only mechanism for draft launches.
24. `llama-fit-params` probe results are estimate-class only: they never gate
    or select launch, never appear as measured evidence, and their absence
    degrades `Fit automatically` to a disabled-with-reason state without
    blocking any valid launch.

## 15. Acceptance boundary

This architecture is ready for implementation only when Phase 0 of the
companion execution plan freezes current-state evidence and proves the binding
target contracts above can be represented without invention:

- the v4 current fixtures and proposed v5/v6 JSON fixtures;
- the field-parity matrix;
- the batch-import and memory-fingerprint classification columns;
- the allowed K/V policy fixture;
- exact existing and binding new API request/response fixtures;
- the 26-file redacted runtime inventory receipt;
- baseline current-source test and screenshot manifests.

After that gate, lower-tier implementation agents may not rename fields,
invent alternate stores, add independent Guided/Pro controls, or relax a hard
exclusion without revising this architecture contract first.

Phase 0 is discovery and transcription, not architecture design. If source
evidence contradicts a binding field or route, it produces a proposed contract
amendment and stops for Coordinator approval. External Ryne acquisition and the
Phase 4.5 second-person UX gate remain explicit human/Coordinator gates; a
Luna/Qwen worker must never self-attest them.
