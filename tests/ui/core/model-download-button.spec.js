import { test, expect } from '@playwright/test';

// The MLX "Download to models folder" button runs a real download job and reports it:
// live percentage, a stall notice, Retry after failure, and "Open in Spawn Wizard" when done.
// Only the endpoints the flow touches are mocked; every other HF call is refused.
test.describe('MLX model download button', () => {
  const REPO = 'nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx';
  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  test('@in-memory-test shows progress, retries after a failure, then offers the wizard', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    let posts = 0;
    let polls = 0;
    const GB = 1024 ** 3;
    await page.route('**/api/hf/**', route => route.abort('blockedbyclient'));
    await page.route('**/api/hf/community-sources', route => json(route, {
      ok: true, catalog: { entries: [], preferences: {}, version: 1 }, roles: [],
    }));
    await page.route('**/api/hf/search', route => json(route, {
      models: [{ id: REPO, tags: ['mlx'], format: 'mlx', param_b: 27, downloads: 5, model_size_bytes: 15_200_000_000 }],
      next_cursor: null,
    }));
    await page.route('**/api/models/mlx-introspect', route => json(route, { ok: false, error: 'mocked' }));
    await page.route('**/metrics/gpu', route => json(route, [
      { name: 'Apple M-series', vendor: 'apple', backend: 'metal', vram_total_mb: 57344, metal_gpu_limit_mb: 57344 },
    ]));
    await page.route('**/api/memory-availability', route => json(route, {
      ok: true, snapshot: { current_safe_availability_bytes: 40 * GB },
    }));
    await page.route('**/api/vram/quant-compare', route => json(route, { ok: true, quants: [] }));
    await page.route('**/api/models/downloads', route => {
      posts += 1;
      return json(route, { ok: true, job_id: `job-${posts}`, repo_id: REPO, engine: 'rapid-mlx' });
    });
    await page.route('**/api/models/downloads/*', route => {
      polls += 1;
      const job = route.request().url().endsWith('job-1')
        ? { state: polls < 2 ? 'running' : 'failed', bytes_done: 4 * GB, bytes_total: 16 * GB, error: 'connection reset' }
        : { state: polls < 4 ? 'running' : 'complete', bytes_done: 8 * GB, bytes_total: 16 * GB, stalled: polls === 3 };
      return json(route, { ok: true, job });
    });

    await page.evaluate(() => window.openModelsModal?.());
    await page.waitForSelector('#models-modal.open');
    await page.locator('.mm-tab[data-tab="download"]').click();
    await page.locator('.mm-tab-panel--download .hf-scope-selector').waitFor();
    await page.locator('#mm-hf-search-input').fill('mindmeld');
    await page.waitForSelector('.hf-search-group');
    await page.locator('.hf-sg-toggle').first().click();
    await page.locator('.hf-sg-variant').first().click();

    const btn = page.locator('#mm-hf-dlp-download-btn');
    await expect(btn).toHaveText('Download to models folder');
    await btn.click();
    await expect(btn).toContainText('Downloading 25%');
    await expect(btn).toHaveText('Retry download', { timeout: 15000 });

    await btn.click();
    await expect(btn).toContainText('Downloading 50%', { timeout: 15000 });
    await expect(btn).toHaveText('Open in Spawn Wizard', { timeout: 15000 });
    expect(posts).toBe(2);
  });
});
