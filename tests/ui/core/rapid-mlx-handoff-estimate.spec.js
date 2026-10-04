import { test, expect } from '@playwright/test';

// Handing an MLX repo from the Models tab to the Spawn Wizard used to leave the sidebar on the
// llama.cpp answer ("Q5_K_M recommended · 19.2 GB") for a repo that has no such file. It now
// shows the MLX estimate for the repo's real size. Every API the flow touches is mocked, and
// any other Hugging Face call is refused.
test.describe('Rapid-MLX handoff memory estimate', () => {
  const REPO = 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx';
  const GB = 1024 ** 3;
  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  test('@in-memory-test the sidebar reports the MLX estimate and no quant recommendation', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    let quantCompareCalls = 0;
    await page.route('**/api/hf/**', route => route.abort('blockedbyclient'));
    await page.route('**/api/hf/community-sources', route => json(route, {
      ok: true, catalog: { entries: [], preferences: {}, version: 1 }, roles: [],
    }));
    await page.route('**/api/hf/files', route => json(route, {
      ok: true, files: [{ path: 'model.safetensors', size: 15 * GB }],
    }));
    await page.route('**/api/hf/mlx-derivatives', route => json(route, { native_mlx_derivatives: [] }));
    await page.route('**/api/hf/qualify', route => json(route, {}));
    await page.route('**/api/models/mlx-introspect', route => json(route, {
      ok: true, data: { recursive_size_bytes: 15 * GB, config: { quantization: { bits: 4, group_size: 32 } } },
    }));
    await page.route('**/metrics/gpu', route => json(route, [
      { name: 'Apple M-series', vendor: 'apple', backend: 'metal', vram_total_mb: 57344, metal_gpu_limit_mb: 57344 },
    ]));
    await page.route('**/api/memory-availability', route => json(route, {
      ok: true, snapshot: { current_safe_availability_bytes: 40 * GB, configured_ceiling_bytes: 48 * GB },
    }));
    await page.route('**/api/vram/quant-compare', route => {
      quantCompareCalls += 1;
      return json(route, { ok: true, quants: [{ label: 'Q5_K_M', model_size_gb: 19.2, fits_vram: true, recommended: true, max_ctx_q8: 100000, max_ctx_q4: 200000, quality: 'good', quality_label: 'Good' }] });
    });
    await page.route('**/api/vram-estimate', route => json(route, {
      ok: true, weights_bytes: 15 * GB, kv_cache_bytes: 2 * GB, overhead_bytes: 1 * GB,
      total_bytes: 18 * GB, available_bytes: 40 * GB, headroom_bytes: 22 * GB,
      recommendation: 'fits',
    }));

    await page.evaluate(async (repo) => {
      const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
      openSpawnWizard({
        templatePreset: { backend: 'rapid_mlx', rapid_mlx: { model_source: { kind: 'hugging_face_repo', repo_id: repo } } },
      });
    }, REPO);

    const hint = page.locator('#wizard-sidebar-vram-hint');
    await expect(hint).toContainText('Fits', { timeout: 15000 });
    await expect(hint).not.toContainText('recommended');
    await expect(page.locator('#mlx-sidebar-body')).toBeVisible();
    await expect(page.locator('#mlx-sidebar-repo-info')).toContainText('4-bit');
    expect(quantCompareCalls).toBe(0);
  });
});
