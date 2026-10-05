import { test, expect } from '@playwright/test';

// Picking a model from the HF hub cache used to lose its memory estimate, and
// once the estimate appeared, the context picker on the hardware step did not
// rescale active KV (the workload scenario's planning target silently replaced
// the user's context) and the current-context rail label stayed at the 8k
// default on the Rapid-MLX path. The real backend serves this spec — no mocks —
// gated behind the local test server so it never runs in CI without a model.
const HUB_PATH = '/Users/nick/.config/local-llm-foundry/models/cache/huggingface/hub/models--nightmedia--Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx';

test.describe('wizard context wiring for HF-cache models', () => {
  test.skip(!HUB_PATH, 'requires the local hub cache');

  test('context picker rescales active KV and the rail label', async ({ page }) => {
    const estimates = [];
    page.on('response', async (res) => {
      if (res.url().includes('/api/vram-estimate') && res.request().method() === 'POST') {
        let req = {}; let resp = {};
        try { req = res.request().postDataJSON(); } catch {}
        try { resp = await res.json(); } catch {}
        if (resp.ok) estimates.push({ n_ctx: req.n_ctx, planning: req.rapid_planning_context_tokens, kv: resp.kv_cache_bytes });
      }
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.evaluate(() => {
      (document.getElementById('open-spawn-wizard')
        || document.querySelector('[data-action="open-spawn-wizard"]'))?.click();
    });
    await page.waitForTimeout(600);
    await page.evaluate((p) => {
      const input = document.getElementById('spawn-model-path');
      document.querySelector('.model-source-card[data-source="rapid_mlx"], [data-engine="rapid_mlx"]')?.click();
      input.value = p;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }, HUB_PATH);
    await page.waitForTimeout(1200);
    await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      wiz.wizardState.engine.recommendation = { state: 'ok' };
      wiz.wizardState.engine.rapidMlxLocalAvailable = true;
      wiz.showStep(1);
    });
    await page.waitForTimeout(3000);

    // The user picks 200k on the left column.
    await page.evaluate(() => {
      const input = document.getElementById('spawn-context-size');
      input.value = '200000';
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForTimeout(2500);

    // The label reflects the selection instead of the 8k default.
    await expect(page.locator('#ctx-rail-summary-value')).toHaveText(/200K/i);

    // Planning tokens follow the context, and KV rescales with it.
    expect(estimates.some((e) => e.planning === 200000)).toBe(true);
    const byCtx = new Map(estimates.map((e) => [e.n_ctx, e.kv]));
    expect(byCtx.size).toBeGreaterThanOrEqual(2);
    const kvs = [...byCtx.values()];
    expect(new Set(kvs).size).toBe(kvs.length);
  });
});
