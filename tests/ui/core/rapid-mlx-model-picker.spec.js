import { test, expect } from '@playwright/test';

// Rapid-MLX downloads land in the app HF cache, not mlx/native. The picker opens on mlx/native,
// so with that folder empty it used to show "Empty directory" for a model the user had just
// downloaded. It now falls back to the app HF cache when that cache has something in it.
test.describe('Rapid-MLX model picker', () => {
  const NATIVE = '/fake/models/mlx/native';
  const CACHE = '/fake/models/cache/huggingface/hub';
  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  async function openPicker(page, cacheEntries) {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.route('**/api/models', route => json(route, []));
    await page.route('**/api/hf/download-dir', route => json(route, {
      dir: '/fake/models',
      locations: { mlx_native: NATIVE, managed_hf_cache: CACHE, library_root: '/fake/models' },
    }));
    await page.route('**/api/browse?*', route => {
      const path = new URL(route.request().url()).searchParams.get('path');
      if (path === NATIVE) return json(route, { path: NATIVE, parent: '/fake/models/mlx', entries: [] });
      if (path === CACHE) return json(route, { path: CACHE, parent: '/fake/models/cache', entries: cacheEntries });
      return json(route, { path: path || '/', parent: '/', entries: [] });
    });
    await page.evaluate(({ NATIVE }) => {
      const input = document.createElement('input');
      input.id = 'picker-test-target';
      document.body.appendChild(input);
      return import('/js/features/file-browser.js').then(m => m.openFileBrowser(
        'picker-test-target', 'dir', NATIVE, { kind: 'model', engine: 'rapid_mlx' },
      ));
    }, { NATIVE });
  }

  test('@in-memory-test an empty mlx/native shows the app HF cache instead', async ({ page }) => {
    await openPicker(page, [
      { name: 'models--owner--repo-mxfp4-mlx', is_dir: true, path: `${CACHE}/models--owner--repo-mxfp4-mlx`, size: 0, size_display: '' },
    ]);
    await expect(page.locator('#fb-path-input')).toHaveValue(CACHE);
    await expect(page.locator('#fb-entries')).toContainText('models--owner--repo-mxfp4-mlx');
    await expect(page.locator('#fb-entries')).not.toContainText('Empty directory');
  });

  test('@in-memory-test with nothing downloaded either, it still says the folder is empty', async ({ page }) => {
    await openPicker(page, []);
    await expect(page.locator('#fb-entries')).toContainText('Empty directory');
    await expect(page.locator('#fb-path-input')).toHaveValue(NATIVE);
  });
  for (const suffix of ['', '/snapshots/abc123']) {
    test(`@in-memory-test cached MLX provenance resolves without HF search ${suffix || 'container'}`, async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      let searches = 0;
      await page.route('**/api/**', route => {
        const url = new URL(route.request().url());
        if (url.pathname === '/api/hf/resolve-origin') searches++;
        if (url.pathname === '/api/models/tags') return json(route, { ok: true, tags: {} });
        return route.abort();
      });
      const result = await page.evaluate(async suffix => {
        const { wizardState } = await import('/js/features/spawn-wizard.js');
        const origin = await import('/js/features/spawn-wizard-hf-origin.js');
        wizardState.model.source = 'local';
        wizardState.model.path = '/fake/hub/models--nightmedia--Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx' + suffix;
        wizardState.model.originRepo = '';
        origin.resetOriginState();
        await origin._autoResolveHfOrigin();
        return { repo: wizardState.model.originRepo, card: wizardState.model.cardUrl };
      }, suffix);
      expect(result.repo).toBe('nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx');
      expect(result.card).toBe('https://huggingface.co/nightmedia/Qwen3.8-27B-MindMeld-AREX-mxfp4-mlx');
      expect(searches).toBe(0);
      await expect(page.locator('#hf-origin-section')).not.toContainText('Not found automatically');
    });
  }
});
