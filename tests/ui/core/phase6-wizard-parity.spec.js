import { test, expect } from '@playwright/test';

const PHASE6_CONTROLS = [
  'spawn-repeat-last-n',
  'spawn-no-cont-batching',
  'spawn-swa-full',
  'spawn-load-mode',
  'spawn-verbosity',
  'spawn-ctx-checkpoints',
  'spawn-checkpoint-min-step',
  'spawn-cache-reuse',
  'spawn-cache-idle-slots',
  'spawn-mmproj-offload',
  'spawn-reasoning-effort',
  'spawn-reasoning-format',
  'spawn-reasoning-preserve',
];

async function openLlamaWizard(page, capabilities = {}) {
  await page.route('**/api/llama-binary/capabilities', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, snapshot: capabilities }),
  }));
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await page.evaluate(async () => {
    const { openSpawnWizard, selectWizardEngine, showStep, wizardState } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    selectWizardEngine('llama_cpp', true);
    wizardState.model.source = 'local';
    wizardState.model.path = '/models/phase6.gguf';
    showStep(1);
  });
  await page.waitForTimeout(100);
}

test.describe('Phase 6 Spawn Wizard Guided/Pro parity', () => {
  test('all binding rows have one canonical wizard node and physical group', async ({ page }) => {
    await openLlamaWizard(page);
    const result = await page.evaluate(async controls => {
      const { PRESENTATION_CONTROLS } = await import('/js/features/spawn-wizard-groups.js');
      return controls.map(id => {
        const node = document.getElementById(id);
        return {
          id,
          count: document.querySelectorAll(`#wizard-step-1 [id="${id}"]`).length,
          category: PRESENTATION_CONTROLS.find(control => control.id === id)?.proCategory,
          group: node?.closest('.mlx-wiz-group')?.dataset.mlxWizGroup || null,
          inGuided: node?.closest('#wizard-step-1')?.style.display !== 'none',
        };
      });
    }, PHASE6_CONTROLS);
    expect(result.every(row => row.count === 1)).toBe(true);
    expect(result.every(row => row.group || row.id === 'spawn-repeat-last-n'), JSON.stringify(result)).toBe(true);
    expect(result.find(row => row.id === 'spawn-swa-full').category).toBe('Memory & context');
    expect(result.find(row => row.id === 'spawn-load-mode').category).toBe('Memory & context');
    expect(result.find(row => row.id === 'spawn-verbosity').category).toBe('Network & observability');
    expect(result.find(row => row.id === 'spawn-mmproj-offload').category).toBe('Model & compatibility');
    expect(result.find(row => row.id === 'spawn-reasoning-effort').category).toBe('Generation & reasoning');
  });

  test('unsupported native values remain visible, disabled, and reasoned', async ({ page }) => {
    await openLlamaWizard(page, {
      cache: { idle_slot_cache: { Unavailable: 'flag absent' } },
      typed: {
        mmproj_offload: {
          positive: { Unavailable: 'flag absent' },
          negative: { Unavailable: 'flag absent' },
        },
        reasoning_effort: { supported: { Unavailable: 'flag absent' }, accepted_values: [] },
        reasoning_format: { supported: 'Available', accepted_values: ['none'] },
        reasoning_preserve: {
          positive: { Unavailable: 'flag absent' },
          negative: { Unavailable: 'flag absent' },
        },
        reasoning_preserve_template: { Unavailable: 'template support is unverified' },
      },
    });
    await expect(page.locator('#spawn-reasoning-preserve-hint')).toContainText('unverified', { timeout: 5000 });
    const result = await page.evaluate(() => Object.fromEntries([
      'spawn-cache-idle-slots', 'spawn-mmproj-offload', 'spawn-reasoning-effort',
      'spawn-reasoning-format', 'spawn-reasoning-preserve',
    ].map(id => {
      const node = document.getElementById(id);
      return [id, {
        disabled: node?.disabled,
        title: node?.title || '',
        hint: node?.parentElement?.querySelector('.field-hint')?.textContent || '',
        options: node?.tagName === 'SELECT' ? [...node.options].map(option => ({ value: option.value, disabled: option.disabled })) : [],
      }];
    })));
    expect(result['spawn-cache-idle-slots'].disabled).toBe(true);
    expect(result['spawn-mmproj-offload'].options.filter(option => option.value).every(option => option.disabled)).toBe(true);
    expect(result['spawn-reasoning-effort'].options.find(option => option.value === 'low').disabled).toBe(true);
    expect(result['spawn-reasoning-format'].options.find(option => option.value === 'deepseek').disabled).toBe(true);
    expect(result['spawn-reasoning-preserve'].options.filter(option => option.value).every(option => option.disabled)).toBe(true);
    expect(result['spawn-reasoning-preserve'].hint).toContain('unverified');
  });

  test('Guided and Pro serialize the same Phase 6 values', async ({ page }) => {
    await openLlamaWizard(page, {
      cache: { idle_slot_cache: 'Available' },
      typed: {
        mmproj_offload: { positive: 'Available', negative: 'Available' },
        reasoning_effort: { supported: 'Available', accepted_values: ['low', 'high'] },
        reasoning_format: { supported: 'Available', accepted_values: ['deepseek'] },
        reasoning_preserve: { positive: 'Available', negative: 'Available' },
        reasoning_preserve_template: { Unavailable: 'template support is unverified' },
      },
    });
    const result = await page.evaluate(async () => {
      const { wizardState } = await import('/js/features/spawn-wizard.js');
      const { buildSpawnPayload } = await import('/js/features/spawn-wizard-spawn.js');
      wizardState.hardware.repeatLastN = 77;
      wizardState.hardware.noContBatching = true;
      wizardState.hardware.swaFull = true;
      wizardState.hardware.loadMode = 'mmap';
      wizardState.hardware.verbosity = 5;
      wizardState.hardware.ctxCheckpoints = 48;
      wizardState.hardware.checkpointMinStep = 4096;
      wizardState.hardware.cacheReuse = 3;
      wizardState.hardware.cacheIdleSlots = false;
      wizardState.hardware.mmprojOffload = true;
      wizardState.hardware.llamaReasoningEffort = 'low';
      wizardState.hardware.llamaReasoningFormat = 'deepseek';
      wizardState.hardware.llamaReasoningPreserve = true;
      wizardState.viewMode = 'guided';
      const guided = buildSpawnPayload();
      wizardState.viewMode = 'pro';
      document.getElementById('view-mode-select').value = 'pro';
      document.getElementById('view-mode-select').dispatchEvent(new Event('change', { bubbles: true }));
      const pro = buildSpawnPayload();
      return { guided, pro };
    });
    expect(result.pro).toEqual(result.guided);
    expect(result.guided).toMatchObject({
      repeat_last_n: 77,
      no_cont_batching: true,
      swa_full: true,
      load_mode: 'mmap',
      verbosity: 5,
      ctx_checkpoints: 48,
      checkpoint_min_step: 4096,
      cache_reuse: 3,
      cache_idle_slots: false,
      mmproj_offload: true,
      llama_reasoning_effort: 'low',
      llama_reasoning_format: 'deepseek',
      llama_reasoning_preserve: true,
    });
  });
});
