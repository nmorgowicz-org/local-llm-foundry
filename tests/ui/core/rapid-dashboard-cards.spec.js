import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

const GiB = 1024 ** 3;
const optional = ['mtp', 'runtime-memory', 'cache'];
const gpu = { load: 25, temp: 42, power: 15, name: 'Fixture GPU', metalUnified: true,
  vramUsed: 4 * GiB, vramTotal: 12 * GiB, unifiedTotal: 16 * GiB };

// Exercise the same normalized view.rapid shared update as dashboard-ws.
// No legacy renderRapidMlxCards or detail-panel renderer is involved.
async function update(page, sample, overrides = {}, invalid = false) {
  await page.evaluate(async ({ sample, overrides, invalid, gpu }) => {
    if (invalid) {
      sample = {
        speculative_acceptance_rate: Infinity,
        completed_requests_total: NaN,
        active_memory_bytes: -1, peak_memory_bytes: '42', cache_memory_bytes: false,
        backend_details: { memory_limit_bytes: {}, telemetry: {
          speculative_acceptance_rate: 1.1, succeeded_requests_total: '0',
          failed_requests_total: -1, cancelled_requests_total: Infinity,
        } },
        cache_metrics: { hits: '0', misses: -1, hit_rate: 2, entry_count: NaN,
          current_memory_bytes: false, multimodal_cache_kinds: [null, 12, ''] },
        global_cache_hit_rate: -0.1, global_cache_entries: Infinity,
      };
    }
    const { normalizeRapidDashboard } = await import('/js/features/rapid-dashboard.js');
    const { updateMetricCards } = await import('/js/features/metric-cards.js');
    const rapid = normalizeRapidDashboard(sample);
    updateMetricCards({ ...rapid, rapid, backend: 'rapid_mlx',
      attached: true, sessionId: 'rapid-a', endpointTag: 'target-a', gpu,
      sys: { cpu: 10, cpuName: 'Fixture CPU' }, ...overrides });
  }, { sample, overrides, invalid, gpu });
}

async function expectCleared(page) {
  for (const key of optional) {
    await expect(page.locator(`#mc-card-${key}`)).toBeHidden();
    await expect(page.locator(`#mv-${key}`)).toHaveText('–');
    await expect(page.locator(`#ms-${key}`)).toBeEmpty();
    await expect(page.locator(`#mu-${key}`)).toBeEmpty();
    await expect(page.locator(`#mn-${key}`)).toBeEmpty();
  }
}

const full = {
  speculative_acceptance_rate: 0.25,
  completed_requests_total: 10,
  active_memory_bytes: GiB, peak_memory_bytes: 3 * GiB, cache_memory_bytes: 2 * GiB,
  backend_details: { memory_limit_bytes: 8 * GiB, telemetry: {
    speculative_acceptance_rate: 0.75, succeeded_requests_total: 7,
    failed_requests_total: 2, cancelled_requests_total: 1,
  } },
  cache_metrics: { hits: 3, misses: 1, hit_rate: 0.6, entry_count: 4,
    current_memory_bytes: GiB / 2, multimodal_cache_kinds: ['image', 'audio'] },
  global_cache_hit_rate: 0.9, global_cache_entries: 99,
};

test.describe('@fake-data-bypass Rapid optional shared main cards', () => {
  test.beforeEach(async ({ page }) => {
    await page.routeWebSocket('**/ws', ws => ws.close());
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await page.evaluate(async () => {
      const { switchView } = await import('/js/features/setup-view.js');
      switchView('monitor');
      document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === 'page-server'));
    });
    await expect(page.locator('#view-monitor')).toBeVisible();
    // Not waitForFunction(async …): a Promise predicate is always truthy, so it
    // would resolve while switchView is still 'transitioning' and frames drop.
    await expect.poll(() => page.evaluate(async () =>
      (await import('/js/core/app-state.js')).setupViewState.view)).toBe('monitor');
  });
  test('reports acceptance, distinct outcome totals, runtime GiB, and local cache facts', async ({ page }) => {
    await update(page, full);
    for (const key of optional) await expect(page.locator(`#mc-card-${key}`)).toBeVisible();
    await expect(page.locator('#mv-mtp')).toHaveText('75.0');
    await expect(page.locator('#mc-card-mtp')).toContainText('acceptance');
    await expect(page.locator('#mc-card-mtp')).toContainText('Not a speedup measurement');
    const runtime = page.locator('#mc-card-runtime-memory');
    await expect(page.locator('#mv-runtime-memory')).toHaveText('1.0');
    await expect(runtime).toContainText('Peak (not a cap): 3.0 GiB');
    await expect(runtime).toContainText('Reclaimable allocator cache: 2.0 GiB');
    await expect(runtime).toContainText('Explicit runtime cap: 8.0 GiB');
    const cache = page.locator('#mc-card-cache');
    await expect(cache).toContainText('Hit rate: 60.0%');
    await expect(cache).toContainText('Hits: 3');
    await expect(cache).toContainText('Misses: 1');
    await expect(cache).toContainText('Entries: 4');
    await expect(cache).toContainText('Prefix cache memory: 512 MiB');
    await expect(cache).toContainText('Multimodal kinds: image, audio');
    await expect(cache).not.toContainText('99');
  });

  test('preserves reported zero but clears every optional field on the first missing update', async ({ page }) => {
    await update(page, { speculative_acceptance_rate: 0, completed_requests_total: 0,
      active_memory_bytes: 0, peak_memory_bytes: 0, cache_memory_bytes: 0,
      backend_details: { memory_limit_bytes: 0 },
      cache_metrics: { hits: 0, misses: 0, hit_rate: 0, entry_count: 0, current_memory_bytes: 0 } });
    await expect(page.locator('#mv-mtp')).toHaveText('0.0');
    await expect(page.locator('#mv-runtime-memory')).toHaveText('0.0');
    await expect(page.locator('#mc-card-runtime-memory')).toContainText('0.0 GiB reported; usable cap unavailable');
    await expect(page.locator('#mc-card-cache')).toContainText('Hit rate: 0.0%');
    await expect(page.locator('#mc-card-cache')).toContainText('Entries: 0');
    await update(page, {});
    await expectCleared(page);
    await update(page, null);
    await expectCleared(page);
    await update(page, { ...full, telemetry_unavailable: true });
    await expectCleared(page);
  });

  test('rejects malformed metrics rather than coercing them into data', async ({ page }) => {
    await update(page, full);
    await update(page, null, {}, true);
    await expectCleared(page);
    await update(page, { speculative_acceptance_rate: 0.4,
      backend_details: { telemetry: { speculative_acceptance_rate: 'invalid' } } });
    await expect(page.locator('#mv-mtp')).toHaveText('40.0');
  });

  test('gates cache on recognized fields and renders partial actual data without inferred rates', async ({ page }) => {
    for (const cache_metrics of [{}, { future: 1 }, { memory_bytes: 50 }, { multimodal_cache_kinds: [] }]) {
      await update(page, { cache_metrics });
      await expect(page.locator('#mc-card-cache')).toBeHidden();
    }
    for (const [field, value, text] of [
      ['hits', 0, 'Hits: 0'], ['misses', 2, 'Misses: 2'], ['hit_rate', 0.2, 'Hit rate: 20.0%'],
      ['entry_count', 0, 'Entries: 0'], ['current_memory_bytes', GiB, 'Prefix cache memory: 1.0 GiB'],
      ['multimodal_cache_kinds', ['<img src=x onerror=alert(1)>'], 'Multimodal kinds: <img src=x onerror=alert(1)>'],
    ]) {
      await update(page, { cache_metrics: { [field]: value } });
      await expect(page.locator('#mc-card-cache')).toBeVisible();
      await expect(page.locator('#mc-card-cache')).toContainText(text);
      await expect(page.locator('#mc-card-cache img')).toHaveCount(0);
    }
    await update(page, { cache_metrics: { hits: 3, misses: 1 } });
    await expect(page.locator('#mc-card-cache')).toContainText('Hit rate: Unavailable');
    await expect(page.locator('#mc-card-cache')).toContainText('Entries: Unavailable');
    await update(page, { global_cache_hit_rate: 0, global_cache_entries: 0 });
    await expect(page.locator('#mc-card-cache')).toContainText('Hit rate: 0.0%');
    await expect(page.locator('#mc-card-cache')).toContainText('Entries: 0');
  });

  test('units follow the headline value: no orphan unit, MiB below 1 GiB, zero stays GiB', async ({ page }) => {
    // Cache without a hit rate: the headline is the placeholder, so no "% hit rate".
    await update(page, { cache_metrics: { hits: 3 } });
    await expect(page.locator('#mv-cache')).toHaveText('–');
    await expect(page.locator('#mu-cache')).toBeEmpty();
    await update(page, { cache_metrics: { hit_rate: 0.5 } });
    await expect(page.locator('#mv-cache')).toHaveText('50.0');
    await expect(page.locator('#mu-cache')).toHaveText('% hit rate');
    // Runtime memory without an active figure: no "GiB active" next to "–".
    await update(page, { peak_memory_bytes: 5 * GiB });
    await expect(page.locator('#mv-runtime-memory')).toHaveText('–');
    await expect(page.locator('#mu-runtime-memory')).toBeEmpty();
    await update(page, { active_memory_bytes: 300 * 1024 ** 2, peak_memory_bytes: 2048, cache_memory_bytes: 0 });
    await expect(page.locator('#mv-runtime-memory')).toHaveText('300');
    await expect(page.locator('#mu-runtime-memory')).toHaveText('MiB active');
    await expect(page.locator('#mc-card-runtime-memory')).toContainText('Peak (not a cap): <0.1 MiB');
    await expect(page.locator('#mc-card-runtime-memory')).toContainText('Reclaimable allocator cache: 0.0 GiB');
    await update(page, { active_memory_bytes: 12 * 1024 ** 2 + 1024 ** 2 / 2 });
    await expect(page.locator('#mv-runtime-memory')).toHaveText('13');
    await update(page, { active_memory_bytes: 1024 ** 2 * 3.5 });
    await expect(page.locator('#mv-runtime-memory')).toHaveText('3.5');
    await update(page, { active_memory_bytes: 0 });
    await expect(page.locator('#mv-runtime-memory')).toHaveText('0.0');
    await expect(page.locator('#mu-runtime-memory')).toHaveText('GiB active');
    await update(page, { active_memory_bytes: 2 * GiB });
    await expect(page.locator('#mv-runtime-memory')).toHaveText('2.0');
    await expect(page.locator('#mu-runtime-memory')).toHaveText('GiB active');
  });

  test('card notes are visible secondary text, not mouse-only tooltips', async ({ page }) => {
    await update(page, full);
    const card = page.locator('#mc-card-runtime-memory');
    await expect(card).not.toHaveAttribute('title');
    const note = page.locator('#mn-runtime-memory');
    await expect(note).toBeVisible();
    await expect(note).toHaveText('Runtime allocations are separate from hardware residency and wired limits.');
    // Compact: small muted text (at most two 11px lines); facts keep their layout.
    const noteHeight = await note.evaluate(el => el.getBoundingClientRect().height);
    expect(noteHeight).toBeLessThan(36);
    await expect(note).toHaveCSS('font-size', '11px');
    expect(await page.locator('#ms-runtime-memory > div').count()).toBe(3);
    await update(page, { cache_metrics: { hits: 1 } });
    await expect(page.locator('#mn-runtime-memory')).toBeEmpty();
    await expect(page.locator('#mn-cache')).toBeEmpty();
  });

  test('sparkline history takes at most one sample per second per card', async ({ page }) => {
    const points = () => page.locator('#mc-card-gpu .mcard__spark-line').getAttribute('d')
      .then(d => (d.match(/L/g) || []).length);
    await page.evaluate(() => { window.__realNow = Date.now; window.__t = 1_900_000_000_000; Date.now = () => window.__t; });
    // Rapid updates inside one second contribute a single sample.
    await update(page, null, { gpu: { ...gpu, load: 10 } });
    await update(page, null, { gpu: { ...gpu, load: 20 } });
    await update(page, null, { gpu: { ...gpu, load: 30 } });
    expect(await points()).toBe(0);                       // one sample: no line yet
    await page.evaluate(() => { window.__t += 999; });
    await update(page, null, { gpu: { ...gpu, load: 40 } });
    expect(await points()).toBe(0);                       // 999 ms: still throttled
    await page.evaluate(() => { window.__t += 1; });
    await update(page, null, { gpu: { ...gpu, load: 50 } });
    expect(await points()).toBe(1);                       // 1000 ms: second sample
    await page.evaluate(() => { window.__t += 400; });
    await update(page, null, { gpu: { ...gpu, load: 60 } });
    await page.evaluate(() => { window.__t += 400; });
    await update(page, null, { gpu: { ...gpu, load: 70 } });
    expect(await points()).toBe(1);
    await page.evaluate(() => { window.__t += 300; });
    await update(page, null, { gpu: { ...gpu, load: 80 } });
    expect(await points()).toBe(2);
    await page.evaluate(() => { Date.now = window.__realNow; });
  });

  test('CPU card shows unavailable, not 0%, until a real sample exists', async ({ page }) => {
    await update(page, null, { sys: { cpu: null, cpuName: 'Fixture CPU' } });
    await expect(page.locator('#mc-card-cpu')).toHaveClass(/mcard--na/);
    await expect(page.locator('#mv-cpu')).toHaveText('–');
    await expect(page.locator('#ms-cpu')).toHaveText('unavailable');
    await update(page, null, { sys: { cpu: 0, cpuName: 'Fixture CPU' } });
    await expect(page.locator('#mc-card-cpu')).not.toHaveClass(/mcard--na/);
    await expect(page.locator('#mv-cpu')).toHaveText('0');
    await expect(page.locator('#ms-cpu')).toHaveText('Fixture CPU');
  });

  test('peak never fabricates an allocation cap', async ({ page }) => {
    await update(page, { peak_memory_bytes: 5 * GiB,
      backend_details: { telemetry: { failed_requests_total: 2 } } });
    await expect(page.locator('#mc-card-runtime-memory')).toContainText('Peak (not a cap): 5.0 GiB');
    await expect(page.locator('#mc-card-runtime-memory')).not.toContainText('Explicit runtime cap');
    await expect(page.locator('#mv-runtime-memory')).toHaveText('–');
  });

  test('backend/detach switches clear values while generic hardware and llama cards remain intact', async ({ page }) => {
    await update(page, full);
    const hardware = await page.locator('#mc-card-vram').textContent();
    await expect(page.locator('#mc-card-vram')).toContainText('Hardware Metal wired cap 12.0 GB');
    // Deliberately leave the Rapid snapshot in state: backend gating must win.
    await update(page, full, { backend: 'llama_cpp', decodeTps: 12, prefillTps: 30, waiting: 0 });
    await expectCleared(page);
    await expect(page.locator('#mv-decode')).toHaveText('12.0');
    await expect(page.locator('#mv-queue')).toHaveText('0');
    expect(await page.locator('#mc-card-vram').textContent()).toBe(hardware);
    await expect(page.locator('#llama-runtime-card')).toHaveCount(1);
    await update(page, full, { attached: false });
    await expectCleared(page);
    await update(page, {}, { sessionId: 'rapid-b', endpointTag: 'target-b' });
    await expectCleared(page);
    await update(page, { completed_requests_total: 1 });
    await expect(page.locator('#mc-card-outcomes')).toHaveCount(0);
  });

  test('long reported multimodal kinds stay within narrow shared cards', async ({ page }) => {
    await page.setViewportSize({ width: 430, height: 900 });
    await update(page, { cache_metrics: { multimodal_cache_kinds: ['x'.repeat(128)] } });
    const cache = page.locator('#mc-card-cache');
    await expect(cache).toBeVisible();
    const overflow = await cache.evaluate(el => el.scrollWidth - el.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
