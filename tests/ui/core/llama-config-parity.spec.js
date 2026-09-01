import fs from 'node:fs';
import path from 'node:path';
import { test, expect } from '@playwright/test';

const catalog = JSON.parse(fs.readFileSync(
  path.resolve(process.cwd(), 'core/fixtures/llama-config-field-catalog.json'),
  'utf8',
));

async function openEditor(page, seed = {}) {
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await page.evaluate(async (preset) => {
    const { openPresetModal } = await import('/js/features/presets.js');
    openPresetModal('new', null, {
      backend: 'llama_cpp',
      name: 'Parity fixture',
      model_path: '/models/parity.gguf',
      ...preset,
    });
  }, seed);
}

test.describe('Phase 5 llama.cpp preset editor parity', () => {
  test('catalog has unique semantic/editor keys and every editor control exists', async ({ page }) => {
    await openEditor(page);
    const result = await page.evaluate(() => import('/js/features/spawn-wizard-groups.js').then(({ FIELD_CATALOG }) => ({
      fields: FIELD_CATALOG.filter(field => field.loaders.includes('llama_cpp')),
      ids: [...document.querySelectorAll('#preset-form [id]')].map(element => element.id),
    })));
    const fields = result.fields;
    expect(fields.length).toBe(catalog.fields.length);
    expect(new Set(fields.map(field => field.semanticId)).size).toBe(fields.length);
    const editorFields = fields.filter(field => field.editorId);
    expect(new Set(editorFields.map(field => field.editorId)).size).toBe(editorFields.length);
    for (const field of editorFields) expect(result.ids).toContain(field.editorId);
    expect(fields.map(field => field.presetKey)).not.toContain('cache_type_k');
    expect(fields.map(field => field.presetKey)).not.toContain('cache_type_v');
  });

  test('capability-sourced KV controls put common values first and preserve unknown values', async ({ page }) => {
    await page.route('**/api/llama-binary/capabilities', route => route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ ok: true, snapshot: {
        cache: { kv_type_values: ['f16', 'q8_0', 'q4_0', 'iq4_xl'] },
        typed: {
          mmproj_offload: { positive: 'Available', negative: 'Available' },
          reasoning_effort: { supported: 'Available', accepted_values: ['low', 'high'] },
          reasoning_format: { supported: 'Available', accepted_values: ['none'] },
          reasoning_preserve: { positive: 'Available', negative: 'Available' },
        },
      } }),
    }));
    await openEditor(page, { ctk: 'future_kv', ctv: 'iq4_xl' });
    await page.waitForTimeout(50);
    const result = await page.evaluate(() => {
      const read = id => [...document.getElementById(id).options].map(option => ({
        value: option.value, disabled: option.disabled,
      }));
      return {
        ctk: read('modal-ctk'),
        ctv: read('modal-ctv'),
        effort: read('modal-llama-reasoning-effort'),
        unknownSelected: document.getElementById('modal-ctk').value,
      };
    });
    expect(result.ctk.slice(0, 3).map(option => option.value)).toEqual(['f16', 'q8_0', 'q4_0']);
    expect(result.ctk[3].disabled).toBe(true);
    expect(result.ctk.at(-1)).toEqual({ value: 'future_kv', disabled: true });
    expect(result.unknownSelected).toBe('future_kv');
    expect(result.ctv.at(-1)).toEqual({ value: 'iq4_xl', disabled: false });
    expect(result.effort.find(option => option.value === 'low').disabled).toBe(false);
    expect(result.effort.find(option => option.value === 'minimal').disabled).toBe(true);
  });

  test('bundle workload policy is visible while flat presets keep it absent', async ({ page }) => {
    await openEditor(page);
    const flat = await page.evaluate(() => ({
      display: getComputedStyle(document.getElementById('modal-workload-policy-wrap')).display,
      value: document.getElementById('modal-workload-policy').value,
    }));
    expect(flat.display).toBe('none');
    expect(flat.value).toBe('general_chat');

    await page.evaluate(async () => {
      const { openPresetModal } = await import('/js/features/presets.js');
      openPresetModal('new', null, {
        backend: 'llama_cpp', name: 'Bundled', model_path: '/models/bundle.gguf',
        bundle: { workload_policy: 'agentic_tools' },
      });
    });
    const bundle = await page.evaluate(() => ({
      display: getComputedStyle(document.getElementById('modal-workload-policy-wrap')).display,
      value: document.getElementById('modal-workload-policy').value,
    }));
    expect(bundle.display).toBe('flex');
    expect(bundle.value).toBe('agentic_tools');
  });

  test('variants editor keeps explicit artifacts and launch choices together', async ({ page }) => {
    await openEditor(page, {
      bundle: {
        identity: { bundle_id: 'bundle-q4q5', tune_id: 'tune-1', display_name: 'Q4/Q5 exact tune' },
        artifacts: [
          { id: 'weights-q4', role: 'weights', display_name: 'Model Q4_K_M', local_path: '/models/q4.gguf', quantization: { value: 'q4_k_m', provenance: 'gguf_metadata' }, metadata: { gguf_architecture: 'qwen3', model_kind: 'moe', block_count: 40 } },
          { id: 'weights-q5', role: 'weights', display_name: 'Model Q5_K_M', local_path: '/models/q5.gguf', quantization: { value: 'q5_k_m', provenance: 'gguf_metadata' }, metadata: { gguf_architecture: 'qwen3', model_kind: 'moe', block_count: 40 } },
        ],
        context_options: [160000, 200000],
        kv_policy_options: ['q4_0_q4_0', 'q8_0_q8_0'],
        performance_options: [{ id: 'balanced', label: '2048 / 256', batch_size: 2048, ubatch_size: 256 }],
        cpu_moe_options: [0, 6],
        curated_selections: [],
        allow_validated_custom: true,
        workload_policy: 'general_chat',
        default_selection: { artifact_id: 'weights-q5', context_size: 200000, kv_policy: 'q4_0_q4_0', performance_id: 'balanced', n_cpu_moe: 6 },
      },
    });
    await page.locator('.preset-nav-item[data-section="variants"]').click();
    await expect(page.locator('#preset-bundle-editor')).toBeVisible();
    await expect(page.locator('#preset-bundle-artifacts .preset-bundle-artifact')).toHaveCount(2);
    await expect(page.locator('#modal-bundle-artifact')).toHaveValue('weights-q5');
    await expect(page.locator('#modal-bundle-context')).toHaveValue('200000');
    await expect(page.locator('#modal-bundle-performance')).toHaveValue('balanced');
    await expect(page.locator('#preset-bundle-moe-wrap')).toBeVisible();
    await expect(page.locator('#modal-bundle-cpu-moe')).toHaveValue('6');
  });

  test('dense bundle artifacts do not expose CPU MoE choices', async ({ page }) => {
    await openEditor(page, {
      bundle: {
        artifacts: [{ id: 'weights', role: 'weights', display_name: 'Dense', local_path: '/models/dense.gguf', metadata: { gguf_architecture: 'llama', model_kind: 'dense', block_count: 32 } }],
        context_options: [8192], kv_policy_options: ['f16_f16'],
        performance_options: [{ id: 'default', label: '512 / 512', batch_size: 512, ubatch_size: 512 }],
        cpu_moe_options: [0], allow_validated_custom: true, workload_policy: 'general_chat',
        default_selection: { artifact_id: 'weights', context_size: 8192, kv_policy: 'f16_f16', performance_id: 'default', n_cpu_moe: 0 },
      },
    });
    await page.locator('.preset-nav-item[data-section="variants"]').click();
    await expect(page.locator('#preset-bundle-moe-wrap')).toBeHidden();
  });

  test('preset payload does not overwrite spawn sampling values a second time', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    const result = await page.evaluate(async () => {
      const { buildPresetPayload } = await import('/js/features/spawn-wizard-review-step.js');
      const { buildSpawnPayload } = await import('/js/features/spawn-wizard-spawn.js');
      const { wizardState } = await import('/js/features/spawn-wizard.js');
      wizardState.engine.selected = 'llama_cpp';
      wizardState.engine.explicit = true;
      wizardState.model.path = '/models/parity.gguf';
      wizardState.hardware.temperature = 0.37;
      wizardState.hardware.topP = 0.91;
      wizardState.hardware.repeatPenalty = 1.08;
      const spawn = buildSpawnPayload();
      const preset = buildPresetPayload();
      return ['temperature', 'top_p', 'repeat_penalty', 'presence_penalty', 'max_tokens', 'seed']
        .every(key => spawn[key] === preset[key]);
    });
    expect(result).toBe(true);
  });
});
