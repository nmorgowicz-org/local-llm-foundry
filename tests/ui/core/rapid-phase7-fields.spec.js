import { test, expect } from '@playwright/test';

// These five RapidMlxConfig fields were accepted by the backend and documented in the plan
// docs, but had no control anywhere in the UI, so no user could ever set them. The point of
// these tests is that the controls reach buildSpawnPayload -- a rendered <select> that never
// arrives in the payload is the same defect in a new coat.
test.describe('Rapid-MLX Phase 7 throughput fields', () => {
  async function openRapidHardware(page, templatePreset = null) {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.evaluate(async (templatePreset) => {
      const { openSpawnWizard, wizardState } = await import('/js/features/spawn-wizard.js');
      openSpawnWizard(templatePreset ? { templatePreset } : {});
      wizardState.engine.selected = 'rapid_mlx';
      wizardState.model.source = 'local';
      wizardState.model.path = '/tmp/Qwen3-8B-4bit';
    }, templatePreset);
  }

  test('@in-memory-test each control reaches the launch payload', async ({ page }) => {
    await openRapidHardware(page);

    const payload = await page.evaluate(async () => {
      const { buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
      const set = (id, value) => {
        const el = document.getElementById(id);
        el.value = value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      };
      set('spawn-rapid-gpu-memory-utilization', '0.85');
      set('spawn-rapid-max-num-seqs', '8');
      set('spawn-rapid-max-concurrent-requests', '32');
      set('spawn-rapid-pflash-policy', 'auto');
      return buildSpawnPayload().rapid_mlx;
    });

    expect(payload.gpu_memory_utilization).toBe(0.85);
    expect(payload.max_num_seqs).toBe(8);
    expect(payload.max_concurrent_requests).toBe(32);
    expect(payload.pflash_policy).toBe('auto');
  });

  test('@in-memory-test untouched controls launch with safe concurrency defaults', async ({ page }) => {
    await openRapidHardware(page);
    await expect(page.locator('#spawn-rapid-gpu-memory-utilization')).toHaveValue('');
    await expect(page.locator('#spawn-rapid-max-num-seqs')).toHaveValue('1');
    await expect(page.locator('#spawn-rapid-max-concurrent-requests')).toHaveValue('4');

    const payload = await page.evaluate(async () => {
      const { buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
      return buildSpawnPayload().rapid_mlx;
    });

    // New launches pin a single running stream and a small request queue. Untouched
    // concurrency controls are not Auto; GPU memory utilization still is.
    expect(payload).not.toHaveProperty('gpu_memory_utilization');
    expect(payload.max_num_seqs).toBe(1);
    expect(payload.max_concurrent_requests).toBe(4);
  });

  test('@in-memory-test explicitly selecting Auto omits the keys rather than sending defaults', async ({ page }) => {
    await openRapidHardware(page);

    const payload = await page.evaluate(async () => {
      const { buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
      const set = (id, value) => {
        const el = document.getElementById(id);
        el.value = value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      };
      // Exercise clearing numeric choices, not merely leaving the controls untouched.
      set('spawn-rapid-gpu-memory-utilization', '0.85');
      set('spawn-rapid-max-num-seqs', '8');
      set('spawn-rapid-max-concurrent-requests', '32');
      for (const id of ['spawn-rapid-gpu-memory-utilization', 'spawn-rapid-max-num-seqs',
                       'spawn-rapid-max-concurrent-requests']) {
        set(id, '');
      }
      return buildSpawnPayload().rapid_mlx;
    });

    await expect(page.locator('#spawn-rapid-gpu-memory-utilization')).toHaveValue('');
    await expect(page.locator('#spawn-rapid-max-num-seqs')).toHaveValue('');
    await expect(page.locator('#spawn-rapid-max-concurrent-requests')).toHaveValue('');
    expect(payload).not.toHaveProperty('gpu_memory_utilization');
    expect(payload).not.toHaveProperty('max_num_seqs');
    expect(payload).not.toHaveProperty('max_concurrent_requests');
  });

  for (const [label, concurrency] of [
    ['missing', {}],
    ['null', { max_num_seqs: null, max_concurrent_requests: null }],
  ]) {
    test(`@in-memory-test ${label} template concurrency stays Auto instead of taking new-launch defaults`, async ({ page }) => {
      await openRapidHardware(page, {
        name: 'Auto template', backend: 'rapid_mlx',
        rapid_mlx: { port: 8080, model_source: '/tmp/Qwen3-8B-4bit', ...concurrency },
      });
      await expect(page.locator('#spawn-rapid-max-num-seqs')).toHaveValue('');
      await expect(page.locator('#spawn-rapid-max-concurrent-requests')).toHaveValue('');
      const payload = await page.evaluate(async () => {
        const { buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
        return buildSpawnPayload().rapid_mlx;
      });
      expect(payload).not.toHaveProperty('max_num_seqs');
      expect(payload).not.toHaveProperty('max_concurrent_requests');
    });
  }

  for (const effort of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh']) {
    test(`@in-memory-test template reasoning effort ${effort} survives hardware visits and reaches launch/save payloads`, async ({ page }) => {
      await openRapidHardware(page, {
        name: 'Reasoning template', backend: 'rapid_mlx',
        rapid_mlx: {
          model_source: { kind: 'mlx_directory', path: '/tmp/Qwen3-8B-4bit' },
          default_reasoning_effort: effort,
        },
      });
      await expect(page.locator('#spawn-rapid-reasoning-effort')).toHaveValue(effort);
      const result = await page.evaluate(async () => {
        const wiz = await import('/js/features/spawn-wizard.js');
        wiz.showStep(1);
        wiz.showStep(0);
        wiz.showStep(1);
        return {
          state: wiz.wizardState.hardware.reasoningEffort,
          launch: wiz.buildSpawnPayload().rapid_mlx.default_reasoning_effort,
          saved: wiz.buildPresetPayload().rapid_mlx.default_reasoning_effort,
        };
      });
      await expect(page.locator('#spawn-rapid-reasoning-effort')).toHaveValue(effort);
      expect(result).toEqual({ state: effort, launch: effort, saved: effort });
    });
  }

  for (const [label, config] of [
    ['missing', {}],
    ['null', { default_reasoning_effort: null }],
    ['empty', { default_reasoning_effort: '' }],
  ]) {
    test(`@in-memory-test ${label} template reasoning effort clears an old selection to Auto`, async ({ page }) => {
      await openRapidHardware(page);
      const result = await page.evaluate(async rapid => {
        const wiz = await import('/js/features/spawn-wizard.js');
        const control = document.getElementById('spawn-rapid-reasoning-effort');
        control.value = 'high';
        control.dispatchEvent(new Event('change', { bubbles: true }));
        wiz.openSpawnWizard({ templatePreset: {
          backend: 'rapid_mlx',
          rapid_mlx: { model_source: { kind: 'mlx_directory', path: '/tmp/model' }, ...rapid },
        } });
        return {
          state: wiz.wizardState.hardware.reasoningEffort,
          launch: wiz.buildSpawnPayload().rapid_mlx.default_reasoning_effort,
        };
      }, config);
      await expect(page.locator('#spawn-rapid-reasoning-effort')).toHaveValue('');
      expect(result).toEqual({ state: '', launch: null });
    });
  }

  test('@in-memory-test explicit Auto survives Rapid defaults and a fresh wizard resets reasoning effort', async ({ page }) => {
    await openRapidHardware(page);
    const result = await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      const { applyRapidMlxDefaults } = await import('/js/features/spawn-wizard-rapid-mlx.js');
      const control = document.getElementById('spawn-rapid-reasoning-effort');
      const choose = value => {
        control.value = value;
        control.dispatchEvent(new Event('change', { bubbles: true }));
      };
      choose('xhigh');
      choose('');
      applyRapidMlxDefaults();
      const auto = wiz.buildSpawnPayload().rapid_mlx.default_reasoning_effort;
      choose('high');
      wiz.openSpawnWizard();
      wiz.wizardState.engine.selected = 'rapid_mlx';
      applyRapidMlxDefaults();
      return { auto, fresh: wiz.buildSpawnPayload().rapid_mlx.default_reasoning_effort,
        state: wiz.wizardState.hardware.reasoningEffort };
    });
    await expect(page.locator('#spawn-rapid-reasoning-effort')).toHaveValue('');
    expect(result).toEqual({ auto: null, fresh: null, state: '' });
  });

  // Only one exclusion rule remains. The second paired speculative_policy with max_num_seqs;
  // it was built on --speculative, which no rapid-mlx release has, and went with the field.
  test('@in-memory-test the mutual-exclusion rule is surfaced', async ({ page }) => {
    await openRapidHardware(page);

    const set = async (id, value) => {
      await page.evaluate(({ id, value }) => {
        const el = document.getElementById(id);
        el.value = value;
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }, { id, value });
    };

    // Rule 1: pflash_policy=on bypasses TurboQuant.
    await set('spawn-rapid-pflash-policy', 'on');
    await set('spawn-turboquant-mode', 'k8v4');
    await expect(page.locator('#spawn-rapid-exclusion-warning')).toContainText('PFlash bypasses TurboQuant');

    // Clearing one side clears the warning.
    await set('spawn-turboquant-mode', 'none');
    await expect(page.locator('#spawn-rapid-exclusion-warning')).toHaveCount(0);
  });
});
