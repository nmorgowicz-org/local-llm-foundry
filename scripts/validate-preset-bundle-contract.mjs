#!/usr/bin/env node
// Validate the Phase 5 frontend field catalog. The catalog is derived from
// CONTROLS; this fixture is a reviewable snapshot, not a second runtime source.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fixturePath = path.join(root, 'tests/ui/core/fixtures/llama-config-field-catalog.json');
const { FIELD_CATALOG } = await import(path.join(root, 'static/js/features/spawn-wizard-groups.js'));

const llama = FIELD_CATALOG.filter(field => field.loaders.includes('llama_cpp'));
const errors = [];
const unique = (values, label) => {
  const seen = new Set();
  for (const value of values) {
    if (value == null) continue;
    if (seen.has(value)) errors.push(`duplicate ${label}: ${value}`);
    seen.add(value);
  }
};

unique(llama.map(field => field.semanticId), 'semantic id');
unique(llama.map(field => field.editorId), 'editor id');
unique(llama.map(field => field.presetKey), 'preset key');
for (const field of llama) {
  for (const required of ['semanticId', 'editorStatus', 'wizardStatus', 'valueType', 'proCategory']) {
    if (!field[required]) errors.push(`${field.semanticId}: missing ${required}`);
  }
  if (field.presetKey === 'cache_type_k' || field.presetKey === 'cache_type_v') {
    errors.push(`${field.semanticId}: deprecated cache_type_k/v cannot be a canonical field`);
  }
}
const requiredKeys = [
  'context_size', 'ctk', 'ctv', 'batch_size', 'ubatch_size', 'n_cpu_moe',
  'repeat_last_n', 'kv_unified', 'no_cont_batching', 'swa_full', 'load_mode',
  'verbosity', 'ctx_checkpoints', 'checkpoint_min_step', 'cache_reuse',
  'mmproj_offload', 'llama_reasoning_effort', 'llama_reasoning_format',
  'llama_reasoning_preserve', 'bundle.workload_policy',
];
for (const key of requiredKeys) {
  if (!llama.some(field => field.presetKey === key)) errors.push(`missing required catalog key: ${key}`);
}

const generated = {
  schema_version: 1,
  generated_from: 'static/js/features/spawn-wizard-groups.js::FIELD_CATALOG',
  fields: llama.map(field => ({
    semantic_id: field.semanticId,
    wizard_id: field.wizardId,
    wizard_status: field.wizardStatus,
    editor_id: field.editorId,
    editor_status: field.editorStatus,
    preset_key: field.presetKey,
    value_type: field.valueType,
    loaders: field.loaders,
    presentation_view: field.view,
    critical: field.critical ?? false,
    pro_category: field.proCategory,
    guided_placement: field.guidedPlacement,
  })),
};

if (errors.length) {
  console.error(errors.map(error => `✗ ${error}`).join('\n'));
  process.exit(1);
}
if (process.argv.includes('--write')) {
  fs.mkdirSync(path.dirname(fixturePath), { recursive: true });
  fs.writeFileSync(fixturePath, `${JSON.stringify(generated, null, 2)}\n`);
  console.log(`✓ wrote ${path.relative(root, fixturePath)}`);
} else if (!fs.existsSync(fixturePath)) {
  console.error(`✗ missing ${path.relative(root, fixturePath)}; run with --write`);
  process.exit(1);
} else if (JSON.stringify(JSON.parse(fs.readFileSync(fixturePath, 'utf8'))) !== JSON.stringify(generated)) {
  console.error(`✗ ${path.relative(root, fixturePath)} is stale; run with --write`);
  process.exit(1);
} else {
  console.log(`✓ field catalog matches ${llama.length} llama.cpp rows`);
}

console.log('✓ preset-bundle frontend contract validated');
