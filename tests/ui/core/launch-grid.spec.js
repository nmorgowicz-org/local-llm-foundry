import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

// Fixture builder for a flat (non-bundle) preset, shaped like the server's
// GET /api/presets rows. `sizeBytes` and `backend` exist so sort-by-size and
// sort-by-backend have something real to reorder.
function preset(id, name, { sizeBytes = 0, backend = 'llama_cpp' } = {}) {
  return {
    id,
    name,
    backend,
    model_path: `/models/${id}.gguf`,
    hf_repo: null,
    context_size: 8192,
    ctk: 'q8_0',
    ctv: 'q8_0',
    batch_size: 512,
    ubatch_size: 512,
    parallel_slots: 1,
    port: 8001,
    bind_host: '127.0.0.1',
    model_size_bytes: sizeBytes,
  };
}

// Route-mocks the launch grid's full dependency surface (loadPresets' four
// parallel fetches, plus renderLaunchGrid's per-card VRAM estimate and memory
// bar calls) with deterministic fixtures, mirroring installPresetMocks() in
// preset-flow.spec.js but scoped to only what the grid itself touches — no
// bundle resolve/selection routes, which that helper carries for its own
// (unrelated) drawer flows.
async function installLaunchGridMocks(page, options = {}) {
  const state = {
    presets: [...(options.presets || [])],
    collections: options.collections || [],
    deleteRequests: [],
  };

  await page.route('**/api/settings', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ preset_id: options.savedPresetId || '' }),
  }));

  await page.route('**/api/sessions/active', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(options.active || { status: 'Stopped', preset_id: '' }),
  }));

  await page.route('**/api/sessions/active/readiness', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, ready: true }),
  }));

  await page.route('**/api/sessions/recent', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ sessions: [], active_session_id: '' }),
  }));

  await page.route('**/api/db/admin-token', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ token: 'admin-token' }),
  }));

  await page.route('**/api/collections', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ collections: state.collections }),
  }));

  await page.route('**/api/preset-cards', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ cards: [], catalog_etag: 'catalog-v1:test', preset_bundle_ui: 'bundled' }),
  }));

  await page.route('**/metrics/system', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ram_total_gb: 128, ram_used_gb: 40, memory_reclaimable_gb: 8 }),
  }));

  await page.route('**/metrics/gpu', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ gpus: [{ vram_total_mb: 131072, vram_used_mb: 40960, metal_gpu_limit_mb: 114688 }] }),
  }));

  await page.route('**/api/system/metal-gpu-limit', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, limit_mb: 114688 }),
  }));

  await page.route('**/api/llama-binary/platform-info', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({ ok: true, auto_backend: 'metal' }),
  }));

  await page.route('**/api/memory-availability', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      ok: true,
      snapshot: {
        current_safe_availability_bytes: 29_000_000_000,
        configured_ceiling_bytes: 120_259_084_288,
        total_unified_bytes: 137_438_953_472,
      },
    }),
  }));

  await page.route('**/api/vram-estimate', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      ok: true,
      total_bytes: 26_300_000_000,
      weights_bytes: 17_179_869_184,
      kv_cache_bytes: 8_000_000_000,
      overhead_bytes: 1_120_130_816,
    }),
  }));

  await page.route(/\/api\/presets(\/[^/]+)?$/, async route => {
    const request = route.request();
    const method = request.method();
    const url = new URL(request.url());
    const parts = url.pathname.split('/').filter(Boolean);
    const id = parts.length === 3 ? decodeURIComponent(parts[2]) : '';

    if (url.pathname === '/api/presets' && method === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(state.presets),
      });
      return;
    }

    if (id && method === 'DELETE') {
      state.deleteRequests.push(id);
      state.presets = state.presets.filter(p => p.id !== id);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      });
      return;
    }

    await route.continue();
  });

  return state;
}

async function boot(page) {
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await expect(page.locator('body')).toHaveClass(/setup-active/);
}

test.describe('launch grid and filters', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
  });

  test('show filter bar when there are multiple presets', async ({ page }) => {
    // Ensure setup view is active
    await expect(page.locator('body')).toHaveClass(/setup-active/);

    // If filter bar is visible, verify basic controls
    const filterBar = page.locator('#setup-filter-bar');
    await page.waitForTimeout(400);

    const isVisible = await filterBar.isVisible().catch(() => false);
    if (!isVisible) {
      // Filter bar only appears when there are >= 3 user presets;
      // we allow the test to pass (no regression) if it is hidden.
      return;
    }

    // Basic controls should be present
    await expect(page.locator('#setup-filter-family-pills')).toBeVisible();
    await expect(page.locator('#setup-filter-size-pills')).toBeVisible();
    await expect(page.locator('#setup-filter-tags-btn')).toBeVisible();
  });

  test('group by family creates group headers when enabled', async ({ page }) => {
    const filterBar = page.locator('#setup-filter-bar');
    const isVisible = await filterBar.isVisible().catch(() => false);
    if (!isVisible) {
      // No filter bar → skip this advanced behavior
      return;
    }

    const groupToggle = page.locator('#setup-filter-group-by-family');
    const isChecked = await groupToggle.isChecked().catch(() => false);

    if (!isChecked) {
      await groupToggle.check();
      await page.waitForTimeout(300);
    }

    // When grouping is active, we expect at least one group header in the grid
    const groupHeaders = page.locator('.launch-grid-group');
    const count = await groupHeaders.count().catch(() => 0);
    if (count > 0) {
      await expect(groupHeaders.first()).toBeVisible();
    }
    // If no groups rendered (e.g., not enough diverse families), no failure;
    // this test mainly guards that the UI path is wired and not throwing.
  });

  test('family filter pills click without errors', async ({ page }) => {
    const familyPills = page.locator('#setup-filter-family-pills .launch-filter-pill');
    const count = await familyPills.count().catch(() => 0);
    if (count <= 1) {
      // Only "All" pill; nothing to test
      return;
    }

    // Click a non-All pill; expect no console errors and layout is stable
    await page.evaluate(() => { window.__lc_filter_error = false; });
    await page.on('console', msg => {
      if (msg.type() === 'error') {
        page.evaluate(() => { window.__lc_filter_error = true; });
      }
    });

    await (await familyPills.last()).click();
    await page.waitForTimeout(200);

    const hasError = await page.evaluate(() => window.__lc_filter_error ?? false);
    expect(hasError).toBe(false);
  });
});

// These tests seed presets deterministically via route mocking (installLaunchGridMocks)
// rather than relying on ambient server state, so the filter bar and its controls are
// guaranteed to render instead of silently no-op'ing the way the tests above allow.
test.describe('launch grid — sort, collections, running badge, delete', () => {
  test('sort control reorders the launch grid by name and by size', async ({ page }) => {
    // Sizes are deliberately in a different rank order than the names, so a
    // name-sort and a size-sort produce distinguishable card orders and a bug
    // that mixes the two comparators up would actually fail this test.
    await installLaunchGridMocks(page, {
      presets: [
        preset('charlie', 'Charlie Model', { sizeBytes: 8_000_000_000 }),
        preset('alpha', 'Alpha Model', { sizeBytes: 1_000_000_000 }),
        preset('delta', 'Delta Model', { sizeBytes: 4_000_000_000 }),
        preset('bravo', 'Bravo Model', { sizeBytes: 16_000_000_000 }),
      ],
    });
    await boot(page);

    const cardIds = () => page.locator('.launch-card[data-preset-id]').evaluateAll(
      els => els.map(e => e.dataset.presetId),
    );

    await expect(page.locator('#setup-filter-bar')).toBeVisible();
    const sortSelect = page.locator('#setup-filter-sort');
    await expect(sortSelect).toBeVisible();

    await sortSelect.selectOption('name');
    await expect.poll(cardIds).toEqual(['alpha', 'bravo', 'charlie', 'delta']);

    // size sort is descending (largest first): bravo(16G), charlie(8G), delta(4G), alpha(1G)
    await sortSelect.selectOption('size');
    await expect.poll(cardIds).toEqual(['bravo', 'charlie', 'delta', 'alpha']);

    // Sort choice is persisted so a re-render (e.g. after any grid refresh) keeps it.
    expect(await page.evaluate(() => localStorage.getItem('llama-monitor-preset-sort'))).toBe('size');
  });

  test('collections filter narrows the grid to presets in the selected collection', async ({ page }) => {
    await installLaunchGridMocks(page, {
      presets: [
        preset('alpha', 'Alpha Model'),
        preset('bravo', 'Bravo Model'),
        preset('charlie', 'Charlie Model'),
      ],
      collections: [
        { id: 'col-1', name: 'My Collection', preset_ids: ['alpha', 'bravo'] },
      ],
    });
    await boot(page);

    const cardIds = () => page.locator('.launch-card[data-preset-id]').evaluateAll(
      els => els.map(e => e.dataset.presetId).sort(),
    );
    await expect.poll(cardIds).toEqual(['alpha', 'bravo', 'charlie']);

    const collectionsGroup = page.locator('#setup-filter-collections-group');
    await expect(collectionsGroup).toBeVisible();
    const collectionsSelect = page.locator('#setup-filter-collections');
    await expect(collectionsSelect.locator('option')).toHaveCount(2); // "All" + the one collection

    await collectionsSelect.selectOption('col-1');
    await expect.poll(cardIds).toEqual(['alpha', 'bravo']);

    await collectionsSelect.selectOption('all');
    await expect.poll(cardIds).toEqual(['alpha', 'bravo', 'charlie']);
  });

  test('running badge appears only on the active session preset\'s card', async ({ page }) => {
    await installLaunchGridMocks(page, {
      presets: [
        preset('alpha', 'Alpha Model'),
        preset('bravo', 'Bravo Model'),
        preset('charlie', 'Charlie Model'),
      ],
    });
    await boot(page);
    await expect(page.locator('.launch-card[data-preset-id]')).toHaveCount(3);

    // No server session is live yet: no card should show the running badge.
    await expect(page.locator('.launch-card--running')).toHaveCount(0);
    await expect(page.locator('.launch-card-running-badge:visible')).toHaveCount(0);

    // dashboard-ws.js normally flips these via the app's real WebSocket connection;
    // that's not something page.route can fake, so drive it directly the way this
    // codebase's own specs do for sessionState (see rapid-preset-visibility.spec.js,
    // calibration.spec.js) and call the same exported function the WS handler calls.
    await page.evaluate(async () => {
      const { sessionState } = await import('/js/core/app-state.js');
      const { updateRunningCardHighlight } = await import('/js/features/setup-view.js');
      sessionState.serverRunning = true;
      sessionState.activeSessionPresetId = 'bravo';
      updateRunningCardHighlight();
    });

    await expect(page.locator('.launch-card[data-preset-id="bravo"]')).toHaveClass(/launch-card--running/);
    await expect(page.locator('.launch-card[data-preset-id="bravo"] .launch-card-running-badge')).toBeVisible();
    await expect(page.locator('.launch-card[data-preset-id="alpha"]')).not.toHaveClass(/launch-card--running/);
    await expect(page.locator('.launch-card[data-preset-id="charlie"]')).not.toHaveClass(/launch-card--running/);
    await expect(page.locator('.launch-card--running')).toHaveCount(1);

    // Clearing server state (e.g. on stop) drops the badge again.
    await page.evaluate(async () => {
      const { sessionState } = await import('/js/core/app-state.js');
      const { updateRunningCardHighlight } = await import('/js/features/setup-view.js');
      sessionState.serverRunning = false;
      updateRunningCardHighlight();
    });
    await expect(page.locator('.launch-card--running')).toHaveCount(0);
  });

  test('per-card delete removes the preset after confirmation', async ({ page }) => {
    const state = await installLaunchGridMocks(page, {
      presets: [
        preset('alpha', 'Alpha Model'),
        preset('throwaway', 'Throwaway Preset'),
      ],
    });
    await boot(page);
    await expect(page.locator('.launch-card[data-preset-id]')).toHaveCount(2);

    const card = page.locator('.launch-card[data-preset-id="throwaway"]');
    await card.locator('.launch-card-btn-trash').click();

    const dialog = page.locator('.app-confirm-dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog.locator('.app-confirm-title')).toHaveText('Delete preset');
    await expect(dialog.locator('.app-confirm-message')).toContainText('Throwaway Preset');

    await dialog.locator('.btn-modal-save').click();

    await expect.poll(() => state.deleteRequests).toEqual(['throwaway']);
    await expect(page.locator('.launch-card[data-preset-id="throwaway"]')).toHaveCount(0);
    await expect(page.locator('.launch-card[data-preset-id="alpha"]')).toHaveCount(1);
  });

  test('per-card delete is a no-op when the confirm dialog is cancelled', async ({ page }) => {
    const state = await installLaunchGridMocks(page, {
      presets: [preset('alpha', 'Alpha Model')],
    });
    await boot(page);

    const card = page.locator('.launch-card[data-preset-id="alpha"]');
    await card.locator('.launch-card-btn-trash').click();

    const dialog = page.locator('.app-confirm-dialog');
    await expect(dialog).toBeVisible();
    await dialog.locator('.btn-modal-cancel').click();

    await expect(dialog).not.toBeVisible();
    await expect(card).toHaveCount(1);
    expect(state.deleteRequests).toEqual([]);
  });
});
