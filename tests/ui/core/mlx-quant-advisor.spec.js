import { test, expect } from '@playwright/test';

// The Download tab's quant advisor for MLX repos compares the sibling conversions of one model
// (mxfp4 / mxfp8 / qx64 ...). It is a single table, so picking another variant must visibly
// move the focus rather than look frozen, and a variant that cannot fit says so instead of
// showing a bare dash. Every API the flow touches is mocked; anything else under /api/hf is
// refused so a test can never reach the real Hugging Face.
test.describe('MLX quant advisor', () => {
  const FREE_BYTES = 28 * 1024 ** 3;
  const MODELS = [
    { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx', tags: ['mlx'], format: 'mlx', param_b: 27, downloads: 5, model_size_bytes: 15_200_000_000 },
    { id: 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp8-mlx', tags: ['mlx'], format: 'mlx', param_b: 27, downloads: 4, model_size_bytes: 28_700_000_000 },
    { id: 'nightmedia/Qwen3.8-27B-MindMeld-qx64-hi-mlx', tags: ['mlx'], format: 'mlx', param_b: 27, downloads: 3, model_size_bytes: 20_900_000_000 },
  ];

  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  test('@in-memory-test the selected variant is highlighted and one that cannot fit says so', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.route('**/api/hf/**', route => route.abort('blockedbyclient'));
    await page.route('**/api/models/mlx-introspect', route => json(route, { ok: false, error: 'mocked' }));
    await page.route('**/api/hf/community-sources', route => json(route, {
      ok: true, catalog: { entries: [], preferences: {}, version: 1 }, roles: [],
    }));
    await page.route('**/api/hf/search', route => json(route, { models: MODELS, next_cursor: null }));
    await page.route('**/metrics/gpu', route => json(route, [
      { name: 'Apple M-series', vendor: 'apple', backend: 'metal', vram_total_mb: 57344, metal_gpu_limit_mb: 57344 },
    ]));
    await page.route('**/api/memory-availability', route => json(route, {
      ok: true, snapshot: { current_safe_availability_bytes: FREE_BYTES },
    }));
    await page.route('**/api/vram/quant-compare', route => {
      const body = route.request().postDataJSON();
      const quants = (body.available_files || []).map(f => {
        const fits = f.size_bytes + 2e9 < body.available_vram_bytes;
        return {
          quant: f.name, label: f.name, model_size_gb: f.size_bytes / 1e9, fits_vram: fits,
          max_ctx_q8: fits ? 548_864 : 0, max_ctx_f16: fits ? 274_432 : 0, max_ctx_q4: fits ? 1_098_752 : 0,
          quality: 'good', quality_label: 'Good', is_imatrix: false, recommended: false, notes: [],
        };
      });
      return json(route, { ok: true, quants });
    });

    await page.evaluate(() => window.openModelsModal?.());
    await page.waitForSelector('#models-modal.open');
    await page.locator('.mm-tab[data-tab="download"]').click();
    await page.locator('.mm-tab-panel--download .hf-scope-selector').waitFor();
    await page.locator('#mm-hf-search-input').fill('mindmeld');
    await page.waitForSelector('.hf-search-group');
    await page.locator('.hf-sg-toggle').first().click();

    // One group, three rows, each led by its quant pill.
    await expect(page.locator('.hf-search-group')).toHaveCount(1);
    await expect(page.locator('.hf-sg-variant .hf-sg-quant-pill')).toHaveText(['mxfp4', 'mxfp8', 'qx64-hi']);

    const selectedRow = page.locator('#mm-quant-advisor-table tr.qa-row-selected');
    const pick = id => page.locator(`.hf-sg-variant:has(.hf-sg-variant-name[title="${id}"])`).click();

    await pick('nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx');
    await expect(selectedRow).toContainText('AREX mxfp4');
    await expect(page.locator('#mm-quant-advisor-table')).toContainText('268k');

    // Picking the variant that does not fit moves the highlight and explains the missing numbers.
    await pick('nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp8-mlx');
    await expect(selectedRow).toContainText('AREX mxfp8');
    await expect(selectedRow.locator('td.qa-ctx-na').first()).toHaveText('won\u2019t fit');

    await pick('nightmedia/Qwen3.8-27B-MindMeld-qx64-hi-mlx');
    await expect(selectedRow).toContainText('qx64-hi');
    await expect(selectedRow).not.toContainText('won\u2019t fit');
  });
});
