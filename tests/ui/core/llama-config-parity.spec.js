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
