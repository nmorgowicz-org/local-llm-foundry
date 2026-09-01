#!/usr/bin/env node
// Lint the spawn-wizard control-tier registry (plan §3.1/§5 Phase 4 item 5):
//   1. Every Quick-tier control carries a quickValue (I2), except the
//      documented pre-existing exceptions.
//   2. Every registered control id resolves to a real DOM node in
//      static/index.html for at least one loader.
// Usage: node scripts/validate-wizard-groups.mjs

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const contractPath = path.join(root, "tests", "ui", "core", "fixtures", "spawn-wizard-control-contract.json");
const fixturePath = path.join(root, 'docs/archive/rapid-mlx/evidence/spawn-wizard-guided-pro/fixture-freeze.json');

const { CONTROLS, PRESENTATION_CONTROLS, assertQuickValueCoverage, validatePresentationDescriptors } = await import(
  path.join(root, "static", "js", "features", "spawn-wizard-groups.js")
);

let failed = false;

try {
  assertQuickValueCoverage();
  console.log(`✓ quickValue coverage (I2): ${CONTROLS.length} controls checked`);
} catch (err) {
  console.error(`✗ ${err.message}`);
  failed = true;
}

for (const loader of ['llama_cpp', 'rapid_mlx']) {
  const result = validatePresentationDescriptors(loader);
  if (!result.ok) {
    console.error(`✗ presentation descriptors invalid for ${loader}: ${result.errors.join('; ')}`);
    failed = true;
  } else {
    console.log(`✓ presentation descriptors (${loader}): ${result.controls.length} canonical mounts`);
  }
}

const html = fs.readFileSync(path.join(root, "static", "index.html"), "utf-8");
const llamaGroupsSource = path.join(root, 'static/js/features/spawn-wizard-llama-ia.js');
const mlxGroupsSource = path.join(root, 'static/js/features/spawn-wizard-mlx-ia.js');
const groupFiles = [llamaGroupsSource, mlxGroupsSource];
const wizardId = (control) => control.wizardId === undefined ? control.id : control.wizardId;
const idCount = (id) => id ? (html.match(new RegExp(`\\bid=["']${id}["']`, "g")) || []).length : 0;
const missing = CONTROLS.filter((c) => wizardId(c) && idCount(wizardId(c)) === 0);
const duplicate = CONTROLS.filter((c) => wizardId(c) && idCount(wizardId(c)) !== 1);
if (missing.length) {
  console.error(`✗ registry ids missing from index.html: ${missing.map((c) => c.id).join(", ")}`);
  failed = true;
} else {
  console.log(`✓ all ${CONTROLS.length} registry ids resolve in index.html`);
}

if (duplicate.length) {
  console.error(`✗ registry ids must occur exactly once: ${duplicate.map((c) => c.id).join(", ")}`);
  failed = true;
}

const classificationFor = (control) => {
  if (control.effective) return 'read-only-effective';
  if (control.view === 'card') return 'guided-card';
  return 'guided-drawer';
};
// Explicit audit map: this is intentionally not inferred from the DOM id.  Empty arrays mean
// a wrapper/read-only control whose launch value is owned by the named child/state path.
const mapping = {
  'spawn-context-size':['context_size'],'hw-mtp-depth':['spec_draft_n_max'],'spawn-batch-size':['batch_size'],'spawn-gpu-layers':['gpu_layers'],
  'spawn-cache-type-k':['ctk'],'spawn-cache-type-v':['ctv'],'spawn-kv-unified':['kv_unified'],'hw-quant-select':[],
  'hw-mmproj-select':['mmproj'],'hw-use-mtp':['spec_type','spec_draft_n_max'],'spawn-parallel-slots':['parallel_slots'],
  'spawn-ubatch-size':['ubatch_size'],'spawn-flash-attn':['flash_attn'],'spawn-prio':['prio'],'spawn-threads':['threads'],
  'spawn-threads-batch':['threads_batch'],'spawn-n-cpu-moe':['n_cpu_moe'],'spawn-tensor-split':['tensor_split'],
  'spawn-cache-mode':['cache_mode'],'spawn-cache-ram':['cache_ram_mib'],'spawn-fit-enable':['fit_enabled'],
  'spawn-fit-target':['fit_target'],'spawn-mlock':['mlock'],'spawn-spec-type':['spec_type'],
  'spawn-spec-draft-type-k':['spec_draft_type_k'],'spawn-spec-draft-type-v':['spec_draft_type_v'],
  'spawn-draft-model':['draft_model'],'spawn-spec-draft-n-min':['spec_draft_n_min'],'spawn-spec-draft-p-min':['spec_draft_p_min'],
  'spawn-rapid-reasoning-mode':['reasoning_mode','no_thinking'],'spawn-rapid-tool-call-parser':['tool_call_parser'],
  'spawn-rapid-reasoning-parser':['reasoning_parser'],'spawn-rapid-hybrid-mode':['hybrid_mode'],'spawn-sampling-mode':['sampling_mode'],
  'spawn-kv-cache-dtype':['kv_cache_dtype'],'spawn-rapid-prefill-step-size':['prefill_step_size'],
  'spawn-turboquant-mode':['turboquant_mode'],'spawn-retained-cache-mib':['retained_cache_mib','prefix_cache_enabled'],
  'spawn-rapid-hybrid-cache-entries':['hybrid_cache_entries'],'spawn-rapid-gpu-memory-utilization':['gpu_memory_utilization'],
  'spawn-rapid-max-num-seqs':['max_num_seqs'],'spawn-rapid-max-concurrent-requests':['max_concurrent_requests'],
  'spawn-rapid-pflash-policy':['pflash_policy'],'spawn-rapid-prefill-batch-size':['prefill_batch_size'],
  'spawn-rapid-completion-batch-size':['completion_batch_size'],'spawn-rapid-auto-tool-choice':['auto_tool_choice'],
  'spawn-rapid-speculative-enabled':['speculative_config']
  ,'spawn-no-cont-batching':['no_cont_batching'],'spawn-swa-full':['swa_full'],'spawn-load-mode':['load_mode'],
  'spawn-verbosity':['verbosity'],'spawn-ctx-checkpoints':['ctx_checkpoints'],'spawn-checkpoint-min-step':['checkpoint_min_step'],
  'spawn-cache-reuse':['cache_reuse'],'spawn-repeat-last-n':['repeat_last_n'],
  'preset-mmproj-offload':['mmproj_offload'],'preset-llama-reasoning-effort':['llama_reasoning_effort'],
  'preset-llama-reasoning-format':['llama_reasoning_format'],'preset-llama-reasoning-preserve':['llama_reasoning_preserve'],
  'preset-workload-policy':['bundle.workload_policy']
};
const descriptorById = new Map(PRESENTATION_CONTROLS.map(control => [control.id, control]));
for (const control of CONTROLS) {
  if (!(control.id in mapping)) {
    const descriptor = descriptorById.get(control.id);
    mapping[control.id] = descriptor?.presetKey ? [descriptor.presetKey] : [];
  }
}
const groupMembership = Object.fromEntries(CONTROLS.map(c => [c.id, groupFiles.flatMap(file => {
  const text = fs.readFileSync(file, 'utf8');
  return [...text.matchAll(new RegExp(`id: '([^']+)'[\\s\\S]{0,700}?controls: \\[([^\\]]*)\\]`, 'g'))]
    .filter(m => m[2].includes(`'${c.id}'`)).map(m => `${path.basename(file)}:${m[1]}`);
})]));
const generatedContract = {
  schema_version: 2,
  generated_from: [
    'static/js/features/spawn-wizard-groups.js::CONTROLS',
    'static/js/features/spawn-wizard-llama-ia.js::GROUPS',
    'static/js/features/spawn-wizard-mlx-ia.js::GROUPS',
  ],
  controls: CONTROLS.map((control) => ({
    ...(() => {
      const descriptor = descriptorById.get(control.id);
      return {
        semantic_id: descriptor.semanticId,
        mount_id: descriptor.mountId,
        mount_kind: descriptor.mountKind,
        guided_placement: descriptor.guidedPlacement,
        pro_category: descriptor.proCategory,
        aliases: descriptor.aliases,
      };
    })(),
    id: control.id,
    loaders: control.loaders,
    classification: classificationFor(control),
    presentation_view: control.view,
    effective_tag: control.effective || null,
    source: 'static/js/features/spawn-wizard-groups.js::CONTROLS',
    canonical_dom_id: wizardId(control),
    payload_keys: mapping[control.id],
    preset_keys: control.loaders[0] === 'rapid_mlx' ? ['rapid_mlx', ...mapping[control.id]] : mapping[control.id],
    effective_read_path: control.effective ? `spawn-wizard-groups.js::EFFECTIVE_COPY.${control.effective}` : 'wizardState -> canonical payload; no separate effective value',
    group_membership: groupMembership[control.id],
    pro_classification: control.wizardStatus === 'planned' ? 'pro-pane (planned; current renderer absent)' : 'pro-pane',
    editor_id: descriptorById.get(control.id).editorId,
    editor_status: descriptorById.get(control.id).editorStatus,
    preset_key: descriptorById.get(control.id).presetKey,
    value_type: descriptorById.get(control.id).valueType,
    wizard_status: descriptorById.get(control.id).wizardStatus,
    launch_classification: mapping[control.id].length ? 'payload-backed' : 'read-only or model-selection wrapper',
  })),
  human_audited: {
    llama_groups: 'static/js/features/spawn-wizard-llama-ia.js::GROUPS',
    rapid_mlx_groups: 'static/js/features/spawn-wizard-mlx-ia.js::GROUPS',
    preset_serializer: 'src/presets/mod.rs::ModelPreset',
  },
};
if (CONTROLS.some(c => !['card', 'both'].includes(c.view))) {
  console.error('✗ unknown control presentation view'); failed = true;
}
for (const c of CONTROLS) {
  if (!mapping[c.id].length && !['hw-quant-select', 'spawn-output-mode'].includes(c.id)) {
    console.error(`✗ empty payload mapping without explicit non-serialized meaning: ${c.id}`); failed = true;
  }
}
// Group membership can be empty only for the explicit shell/card controls which have no IA wrapper.
const permittedUngrouped = new Set(['spawn-context-size','spawn-gpu-layers','spawn-cache-type-k','spawn-cache-type-v','spawn-kv-unified','hw-quant-select','hw-mmproj-select','hw-use-mtp','spawn-spec-type','spawn-spec-draft-type-k','spawn-spec-draft-type-v','spawn-draft-model','spawn-spec-draft-n-min','spawn-spec-draft-p-min','preset-mmproj-offload','preset-llama-reasoning-effort','preset-llama-reasoning-format','preset-llama-reasoning-preserve','preset-workload-policy','hw-mtp-depth','spawn-repeat-last-n','spawn-temperature','spawn-top-p','spawn-min-p','spawn-repeat-penalty','spawn-presence-penalty','spawn-top-k','spawn-max-tokens','spawn-seed','spawn-enable-thinking','spawn-preserve-thinking','spawn-reasoning-mode','spawn-reasoning-budget','spawn-output-mode','spawn-grammar','spawn-json-schema','spawn-tool-call-format','spawn-port','spawn-bind-host','spawn-alias','spawn-api-key','spawn-extra-args']);
for (const c of CONTROLS) if (!groupMembership[c.id].length && !permittedUngrouped.has(c.id)) {
  console.error(`✗ group extraction drift: ${c.id} is neither in GROUPS nor an approved shell/card control`); failed = true;
}
const frozenFixtures = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
if (!Array.isArray(frozenFixtures.fixtures) || frozenFixtures.fixtures.some(f => !f.id || !f.engine)) {
  console.error('✗ fixture freeze schema requires fixtures[] with id and engine'); failed = true;
}
if (process.argv.includes('--write-contract')) {
  fs.mkdirSync(path.dirname(contractPath), { recursive: true });
  fs.writeFileSync(contractPath, `${JSON.stringify(generatedContract, null, 2)}\n`);
  console.log(`✓ wrote ${path.relative(root, contractPath)}`);
} else if (fs.existsSync(contractPath)) {
  const checkedIn = JSON.parse(fs.readFileSync(contractPath, 'utf8'));
  if (JSON.stringify(checkedIn) !== JSON.stringify(generatedContract)) {
    console.error('✗ checked-in control contract differs; run node scripts/validate-wizard-groups.mjs --write-contract');
    failed = true;
  } else {
    console.log(`✓ checked-in control contract matches ${CONTROLS.length} controls`);
  }
} else {
  console.error('✗ missing checked-in spawn wizard control contract');
  failed = true;
}

if (failed) {
  console.log("");
  console.log("spawn-wizard-groups validation FAILED.");
  process.exit(1);
}

console.log("");
console.log("spawn-wizard-groups validated successfully.");
process.exit(0);
