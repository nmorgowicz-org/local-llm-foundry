import { test, expect } from '@playwright/test';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

// Picking a model from the HF hub cache used to lose its memory estimate, and
// once the estimate appeared, the context picker on the hardware step did not
// rescale active KV (the workload scenario's planning target silently replaced
// the user's context) and the current-context rail label stayed at the 8k
// default on the Rapid-MLX path. The real backend serves this spec — no mocks —
// so it needs a real MLX model directory from the HF hub cache.
//
// Point LLAMA_MONITOR_TEST_HF_MODEL_DIR at any cached MLX model directory
// (models--<owner>--<name>) to run it. Without that, it falls back to the model the
// feature was developed against under the default app home, and skips when that
// directory is absent, so CI and other machines never fail on a missing cache.
const DEFAULT_HUB_MODEL = join(
  homedir(),
  '.config/local-llm-foundry/models/cache/huggingface/hub/models--nightmedia--Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx',
);
const HUB_PATH = process.env.LLAMA_MONITOR_TEST_HF_MODEL_DIR || DEFAULT_HUB_MODEL;

test.describe('wizard context wiring for HF-cache models', () => {
  test.skip(!existsSync(HUB_PATH), `requires a local HF-cache MLX model (looked in ${HUB_PATH}); set LLAMA_MONITOR_TEST_HF_MODEL_DIR`);

  test('context picker rescales active KV and the rail label', async ({ page }) => {
    const estimates = [];
    page.on('response', async (res) => {
      if (res.url().includes('/api/vram-estimate') && res.request().method() === 'POST') {
        let req = {}; let resp = {};
        try { req = res.request().postDataJSON(); } catch { /* not JSON */ }
        try { resp = await res.json(); } catch { /* not JSON */ }
        if (resp.ok) estimates.push({ n_ctx: req.n_ctx, planning: req.rapid_planning_context_tokens, kv: resp.kv_cache_bytes });
      }
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.evaluate(() => {
      (document.getElementById('open-spawn-wizard')
        || document.querySelector('[data-action="open-spawn-wizard"]'))?.click();
    });
    await expect(page.locator('#spawn-model-path')).toBeAttached();
    await page.evaluate((p) => {
      const input = document.getElementById('spawn-model-path');
      document.querySelector('.model-source-card[data-source="rapid_mlx"], [data-engine="rapid_mlx"]')?.click();
      input.value = p;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }, HUB_PATH);
    // The wizard module is imported only once the Rapid-MLX engine path is live.
    await expect.poll(() => page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      return document.getElementById('spawn-model-path').value !== '' && !!wiz.wizardState?.engine;
    })).toBe(true);
    await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      wiz.wizardState.engine.recommendation = { state: 'ok' };
      wiz.wizardState.engine.rapidMlxLocalAvailable = true;
      wiz.showStep(1);
    });
    // The first estimate for the hardware step arrives from the real backend.
    await expect.poll(() => estimates.length, { timeout: 30_000 }).toBeGreaterThan(0);

    // The user picks 200k on the left column.
    await page.evaluate(() => {
      const input = document.getElementById('spawn-context-size');
      input.value = '200000';
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });

    // The label reflects the selection instead of the 8k default.
    await expect(page.locator('#ctx-rail-summary-value')).toHaveText(/200K/i);

    // Planning tokens follow the context, and KV rescales with it.
    await expect.poll(() => estimates.some((e) => e.planning === 200000), { timeout: 30_000 }).toBe(true);
    const byCtx = new Map(estimates.map((e) => [e.n_ctx, e.kv]));
    expect(byCtx.size).toBeGreaterThanOrEqual(2);
    const kvs = [...byCtx.values()];
    expect(new Set(kvs).size).toBe(kvs.length);
  });
});
