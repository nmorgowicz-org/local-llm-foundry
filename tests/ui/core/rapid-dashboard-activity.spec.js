import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

async function push(page, inference, extra = {}) {
  await page.evaluate(async frame => {
    const { initWebSocket } = await import('/js/features/dashboard-ws.js');
    const socket = initWebSocket();
    socket.onmessage({ data: JSON.stringify(frame) });
    socket.close();
  }, {
    backend: 'rapid_mlx', session_mode: 'attach', active_session_id: 'rapid-fixture',
    active_session_status: 'running', active_session_endpoint: 'http://localhost:8001',
    server_running: true, local_server_running: false, inference, llama: null,
    host_metrics_available: false, capabilities: {}, gpu: {}, system: null,
    logs: [], mode: 'off', ...extra,
  });
}

const requestId = 'request-0123456789-abcdefghijklmnopqrstuvwxyz';

test.describe('@fake-data-bypass shared dashboard request activity', () => {
  test.beforeEach(async ({ page }) => {
    await page.routeWebSocket('**/ws', ws => ws.close());
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await page.evaluate(async () => {
      (await import('/js/features/setup-view.js')).switchView('monitor');
      document.querySelectorAll('.page').forEach(el => el.classList.toggle('active', el.id === 'page-server'));
    });
    await expect(page.locator('#view-monitor')).toBeVisible();
    // Not waitForFunction(async …): an async predicate returns a Promise, which is
    // always truthy, so it resolves immediately while switchView is still
    // 'transitioning' (~0.9s). updateDashboard() drops frames in that window, and
    // the spec's single pushed frame would never render.
    await expect.poll(() => page.evaluate(async () =>
      (await import('/js/core/app-state.js')).setupViewState.view)).toBe('monitor');
  });

  test('progressbar semantics and live label are static markup, not JS-only', async ({ page }) => {
    // Before any frame: role/min/max/label exist in index.html and the bar is hidden from AT.
    const bar = page.locator('#state-progress');
    await expect(bar).toHaveAttribute('role', 'progressbar');
    await expect(bar).toHaveAttribute('aria-valuemin', '0');
    await expect(bar).toHaveAttribute('aria-valuemax', '100');
    await expect(bar).toHaveAttribute('aria-label', /.+/);
    await expect(bar).toHaveAttribute('aria-hidden', 'true');
    await expect(bar).not.toHaveAttribute('aria-valuenow');
    // Only the label (changes on state transitions) is live; per-tick detail is not.
    await expect(page.locator('#state-label')).toHaveAttribute('aria-live', 'polite');
    await expect(page.locator('#state-detail')).not.toHaveAttribute('aria-live');
    await push(page, { running_requests: 1, waiting_requests: 0,
      active_requests: [{ id: requestId, phase: 'decode', completion_tokens: 8, max_tokens: 32 }] });
    await expect(bar).toHaveAttribute('aria-hidden', 'false');
    await expect(bar).toHaveAttribute('role', 'progressbar');
    await expect(bar).toHaveAttribute('aria-valuenow', '25');
  });

  test('every model state lights exactly one pill, including busy and unavailable', async ({ page }) => {
    const activePills = () => page.locator('#state-card .state-pill.active').evaluateAll(
      nodes => nodes.map(node => node.dataset.s));
    await push(page, { running_requests: 0, waiting_requests: 0, active_requests: [] });
    await expect.poll(activePills).toEqual(['idle']);
    // Running with no reported phase is busy, not generating.
    await push(page, { running_requests: 1, waiting_requests: 0, active_requests: [{ id: requestId }] });
    await expect(page.locator('#state-label')).toHaveText('Processing');
    await expect.poll(activePills).toEqual(['busy']);
    await push(page, { telemetry_unavailable: true });
    await expect(page.locator('#state-label')).toHaveText('Telemetry unavailable');
    await expect.poll(activePills).toEqual(['unavailable']);
    await push(page, { running_requests: 0, waiting_requests: 2, active_requests: [] });
    await expect.poll(activePills).toEqual(['queued']);
    // Detached: reads as idle rather than as a telemetry fault.
    await push(page, null, { active_session_id: null, active_session_endpoint: null });
    await expect(page.locator('#state-label')).toHaveText('Waiting for a request');
    await expect.poll(activePills).toEqual(['idle']);
  });

  test('start estimates are scoped to the owning target and survive rows without elapsed_s', async ({ page }) => {
    const started = () => page.locator('#rapid-request-rows > tr').first().locator('td').nth(2).textContent();
    const clock = async t => page.evaluate(ms => { window.__t = ms; }, t);
    const T = new Date(2030, 0, 1, 12, 0, 0).getTime();
    await page.evaluate(ms => { window.__realNow = Date.now; window.__t = ms; Date.now = () => window.__t; }, T);
    try {
      const row = elapsed => ({ id: requestId, phase: 'decode', ...(elapsed == null ? {} : { elapsed_s: elapsed }) });
      await push(page, { running_requests: 1, active_requests: [row(100)] });
      const first = await started();
      expect(first).toMatch(/11:58:20/);
      // Within the tolerance the first estimate is kept (no jitter).
      await clock(T + 1000);
      await push(page, { running_requests: 1, active_requests: [row(100)] });
      expect(await started()).toBe(first);
      // A row that merely lacks elapsed_s is still present: keep its estimate.
      await push(page, { running_requests: 1, active_requests: [row(null)] });
      expect(await started()).toBe(first);
      await expect(page.locator('.rapid-request-table th')).toContainText(['Started (est.)']);
      // Same request ID under another session is a different request: re-estimate.
      await push(page, { running_requests: 1, active_requests: [row(100)] },
        { active_session_id: 'rapid-other' });
      expect(await started()).toMatch(/11:58:21/);
    } finally {
      await page.evaluate(() => { Date.now = window.__realNow; });
    }
  });

  test('volatile cell changes patch text in place: rows, IDs and selection survive ticks', async ({ page }) => {
    const sample = (elapsed, rate) => ({ running_requests: 1, active_requests: [
      { id: requestId, phase: 'decode', completion_tokens: elapsed, max_tokens: 64, elapsed_s: elapsed, tokens_per_second: rate },
    ] });
    await push(page, sample(1, 5));
    await page.evaluate(() => {
      const tr = document.querySelector('#rapid-request-rows > tr');
      tr.__marker = 'same-row';
      const short = tr.querySelector('.rapid-request-id-short');
      short.__marker = 'same-id';
      const range = document.createRange();
      range.selectNodeContents(short);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      window.__mutations = 0;
      new MutationObserver(records => { window.__mutations += records.length; })
        .observe(tr.querySelector('[data-request-id]'), { subtree: true, childList: true, characterData: true, attributes: true });
      window.__selected = selection.toString();
    });
    await push(page, sample(2, 6));
    await expect(page.locator('#rapid-request-rows > tr td').nth(3)).toHaveText('2 / 64');
    await expect(page.locator('#rapid-request-rows > tr td').nth(4)).toHaveText('6');
    const state = await page.evaluate(() => {
      const tr = document.querySelector('#rapid-request-rows > tr');
      const short = tr.querySelector('.rapid-request-id-short');
      const selection = getSelection();
      return { sameRow: tr.__marker, sameId: short.__marker, selected: selection.toString(),
        attached: selection.rangeCount > 0 && short.contains(selection.anchorNode), mutations: window.__mutations };
    });
    expect(state.sameRow).toBe('same-row');
    expect(state.sameId).toBe('same-id');
    expect(state.attached).toBe(true);
    expect(state.selected).toBe(await page.evaluate(() => window.__selected));
    expect(state.selected.length).toBeGreaterThan(0);
    expect(state.mutations).toBe(0);
    // A new request is added without touching the existing row; removal drops only that row.
    await push(page, { running_requests: 2, active_requests: [
      ...sample(3, 6).active_requests,
      { id: 'second-request', phase: 'decode', completion_tokens: 1, max_tokens: 64, elapsed_s: 1, tokens_per_second: 5 }] });
    await expect(page.locator('#rapid-request-rows > tr')).toHaveCount(2);
    expect(await page.evaluate(() => document.querySelector('#rapid-request-rows > tr').__marker)).toBe('same-row');
    await push(page, { running_requests: 1, active_requests: sample(3, 6).active_requests });
    await expect(page.locator('#rapid-request-rows > tr')).toHaveCount(1);
    expect(await page.evaluate(() => document.querySelector('#rapid-request-rows > tr').__marker)).toBe('same-row');
    // A changed column set is structural: headers and rows are rebuilt.
    await push(page, { running_requests: 1, active_requests: [
      { ...sample(3, 6).active_requests[0], prompt_tokens: 12 }] });
    await expect(page.locator('.rapid-request-table th')).toContainText(['Prompt tokens']);
    expect(await page.evaluate(() => document.querySelector('#rapid-request-rows > tr').__marker)).toBeUndefined();
    await expect(page.locator('#rapid-request-rows > tr')).toHaveCount(1);
  });

  test('request IDs use one bound for display, accessible name and row key; ID cell is a positioning context', async ({ page }) => {
    const long = 'x'.repeat(200);
    await push(page, { running_requests: 1, active_requests: [{ id: long, phase: 'decode' }] });
    const id = page.locator('#rapid-request-rows [data-request-id]');
    await expect(id).toHaveAttribute('data-request-id', 'x'.repeat(128));
    await expect(page.locator('#rapid-request-rows td').first()).toHaveAccessibleName(`Request ID ${'x'.repeat(128)}`);
    await expect(id.locator('.rapid-request-id-full')).toHaveText(`Request ID ${'x'.repeat(128)}`);
    await expect(page.locator('#rapid-request-rows td').first()).toHaveCSS('position', 'relative');
  });

  test('memory warning live region writes only on change and empties when hidden', async ({ page }) => {
    const host = { host_metrics_available: true, capabilities: { system: true } };
    const warn = (level) => ({ ...host, system: { memory_pressure_level: level, memory_pressure_source: 'fixture', swap_used_gb: 1 } });
    const idle = { running_requests: 0, waiting_requests: 0, active_requests: [] };
    await push(page, idle, warn('critical'));
    const region = page.locator('#dashboard-memory-warning');
    await expect(region).toBeVisible();
    await expect(region).toContainText('Critical host memory pressure (fixture)');
    await page.evaluate(() => {
      window.__warnWrites = 0;
      new MutationObserver(records => { window.__warnWrites += records.length; })
        .observe(document.getElementById('dashboard-memory-warning'), { subtree: true, childList: true, characterData: true });
    });
    await push(page, idle, warn('critical'));
    await push(page, idle, warn('critical'));
    await expect(region).toBeVisible();
    expect(await page.evaluate(() => window.__warnWrites)).toBe(0);
    await push(page, idle, warn('ok'));
    await expect(region).toBeHidden();
    await expect(region).toHaveText('');
  });

  test('reading without processed counters animates without claiming a fraction', async ({ page }) => {
    await push(page, {
      running_requests: 1, waiting_requests: 0, backend_details: { progress: 0.75 },
      active_requests: [{ id: requestId, phase: 'prefill', prompt_tokens: 4096, elapsed_s: 0 }],
    });
    await expect(page.locator('#state-label')).toHaveText('Reading');
    await expect(page.locator('#state-detail')).toContainText('4,096 prompt tokens');
    await expect(page.locator('#state-detail')).toContainText('0.0s elapsed');
    await expect(page.locator('#state-progress')).toHaveAttribute('data-indeterminate', 'true');
    await expect(page.locator('#state-progress')).not.toHaveAttribute('aria-valuenow');
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuetext', 'Reading activity; processed prompt tokens unavailable');
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect(page.locator('#state-bar')).toHaveCSS('animation-name', 'none');
    // Reduced motion must not leave a short static segment that reads like ~30% done:
    // the bar fills the whole track at low opacity instead.
    await expect(page.locator('#state-bar')).toHaveCSS('opacity', '0.35');
    const [bar, track] = await page.evaluate(() => ['state-bar', 'state-progress']
      .map(id => document.getElementById(id).getBoundingClientRect().width));
    expect(bar).toBeGreaterThanOrEqual(track - 3);
    await expect(page.locator('#state-bar')).toHaveCSS('transform', 'none');
    await push(page, { running_requests: 0, waiting_requests: 0, active_requests: [] });
    await expect(page.locator('#state-progress')).not.toHaveAttribute('data-indeterminate');
    await expect(page.locator('#state-progress')).toHaveClass(/state-progress--hidden/);
  });

  test('generation uses observed output budget, never backend completion prediction', async ({ page }) => {
    await push(page, {
      running_requests: 1, waiting_requests: 0, backend_details: { progress: 0.9 },
      active_requests: [{ id: requestId, phase: 'decode', completion_tokens: 8, max_tokens: 32 }],
    });
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuenow', '25');
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-label', 'Output budget used');
    await expect(page.locator('#state-detail')).toContainText('8 / 32 tokens · output budget used');
    await push(page, {
      running_requests: 1, waiting_requests: 0, backend_details: { progress: { current: 3, total: 4 } },
      active_requests: [{ phase: 'decode', completion_tokens: 0 }],
    });
    await expect(page.locator('#state-progress')).not.toHaveAttribute('aria-valuenow');
    await expect(page.locator('#state-progress')).toHaveClass(/state-progress--hidden/);
    await expect(page.locator('#state-detail')).toContainText('0 output tokens');
  });

  test('llama prefill zero is real progress and llama output limits are preserved', async ({ page }) => {
    const endpoint = 'http://localhost:8001';
    const llama = { telemetry_session_id: 'rapid-fixture', telemetry_endpoint: endpoint,
      slots_processing: 1, slot_generation_tokens: 0, slot_prompt_processed: 0, slot_prompt_total: 100,
      slots: [{ is_processing: true, prompt_tokens_processed: 0 }] };
    await push(page, null, { backend: 'llama_cpp', active_session_endpoint_tag: endpoint, llama });
    await expect(page.locator('#state-label')).toHaveText('Reading');
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuenow', '0');
    await expect(page.locator('#state-progress')).not.toHaveAttribute('data-indeterminate');
    await push(page, null, { backend: 'llama_cpp', active_session_endpoint_tag: endpoint,
      llama: { ...llama, slot_generation_tokens: 10, slot_generation_limit: 40, generation_tokens_per_sec: 20 } });
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuenow', '25');
    await expect(page.locator('#state-label')).toHaveText('Generating');
  });

  test('llama slot availability prevents legacy default zero from claiming processed progress', async ({ page }) => {
    const endpoint = 'http://localhost:8001';
    const llama = { telemetry_session_id: 'rapid-fixture', telemetry_endpoint: endpoint,
      slots_processing: 1, slot_generation_tokens: 0, slot_prompt_processed: 0, slot_prompt_total: 100,
      slots: [{ is_processing: true, prompt_tokens_processed: null }] };
    await push(page, null, { backend: 'llama_cpp', active_session_endpoint_tag: endpoint, llama });
    await expect(page.locator('#state-progress')).toHaveAttribute('data-indeterminate', 'true');
    await expect(page.locator('#state-progress')).not.toHaveAttribute('aria-valuenow');
    await push(page, null, { backend: 'llama_cpp', active_session_endpoint_tag: endpoint,
      llama: { ...llama, slot_prompt_processed: 25, slots: [{ is_processing: true, prompt_tokens_processed: 25 }] } });
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuenow', '25');
    await expect(page.locator('#state-progress')).not.toHaveAttribute('data-indeterminate');
  });

  test('table retains real rows and cells, short accessible IDs and only reported columns', async ({ page }) => {
    await push(page, { running_requests: 1, waiting_requests: 0, active_requests: [
      { id: requestId, phase: 'decode', prompt_tokens: 100, completion_tokens: 0, max_tokens: 32, elapsed_s: 0 },
    ] });
    const rows = page.locator('#rapid-request-rows > tr');
    await expect(rows).toHaveCount(1);
    await expect(rows.locator('td')).toHaveCount(6);
    const id = rows.locator('[data-request-id]');
    await expect(rows.locator('td').first()).toHaveAccessibleName(`Request ID ${requestId}`);
    await expect(id).toHaveAttribute('title', requestId);
    expect((await id.locator('.rapid-request-id-short').textContent()).length).toBeLessThan(24);
    await expect(page.locator('.rapid-request-table th')).toHaveText(['Request', 'Phase / status', 'Started (est.)', 'Prompt tokens', 'Output / limit', 'Elapsed (s)']);
    await expect(rows.locator('td').nth(2)).toHaveText(/\d{1,2}:\d{2}:\d{2}/);
    await expect(rows.locator('td').nth(2)).toHaveAttribute('title', /^Estimated: /);
    await expect(rows.locator('td').nth(4)).toHaveText('0 / 32');
    await expect(rows.locator('td').nth(5)).toHaveText('0');
    await expect(page.locator('#rapid-dashboard-heading')).toHaveCount(0);
    await expect(page.locator('#rapid-dashboard-details .mcard')).toHaveCount(0);
    await expect(page.locator('.rapid-request-table')).toHaveCount(1);
  });

  test('partial concurrent budgets never fill missing request metrics with zero', async ({ page }) => {
    await push(page, { running_requests: 2, waiting_requests: 0, active_requests: [
      { phase: 'decode', completion_tokens: 4, max_tokens: 16 },
      { phase: 'decode', completion_tokens: 2 },
    ] });
    await expect(page.locator('#state-progress')).not.toHaveAttribute('aria-valuenow');
    await push(page, { running_requests: 2, waiting_requests: 0, active_requests: [
      { phase: 'decode', completion_tokens: 4, max_tokens: 16 },
      { phase: 'decode', completion_tokens: 0, max_tokens: 16 },
    ] });
    // aria-valuenow is rounded; the precise value stays in aria-valuetext.
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuenow', '13');
    await expect(page.locator('#state-progress')).toHaveAttribute('aria-valuetext', /^12\.5% /);
    await expect(page.locator('#state-detail')).toContainText('4 / 32 tokens');
  });

  test('unreported request details and bounded snapshots do not imply zero requests', async ({ page }) => {
    await push(page, { running_requests: 1, waiting_requests: 0 });
    await expect(page.locator('#rapid-request-summary')).toHaveText('Active request details unavailable');
    await push(page, { active_requests: Array.from({ length: 40 }, (_, index) => ({ id: `r-${index}`, phase: 'queued' })) });
    await expect(page.locator('#rapid-request-rows > tr')).toHaveCount(32);
    await expect(page.locator('#rapid-request-summary')).toContainText('Showing 32 of 40');
    await push(page, { telemetry_unavailable: true, active_requests: [{ id: 'stale-request', phase: 'decode' }] });
    await expect(page.locator('#rapid-request-summary')).toHaveText('Active request details unavailable');
    await expect(page.locator('#rapid-request-rows')).not.toContainText('stale-request');
  });

  test('table optional fields, empty colspan and malicious text remain safe across snapshots', async ({ page }) => {
    await push(page, { active_requests: [{ id: '<img src=x onerror=alert(1)>', phase: '<script>x</script>', cached_tokens: 0, tokens_per_second: '8', ttft_s: -1, cache_hit_type: 'prefix' }] });
    await expect(page.locator('#rapid-request-rows img, #rapid-request-rows script')).toHaveCount(0);
    await expect(page.locator('.rapid-request-table th')).toHaveText(['Request', 'Phase / status', 'Cached tokens', 'Cache hit']);
    await expect(page.locator('#rapid-request-rows')).toContainText('<script>x</script>');
    await push(page, { active_requests: [] });
    await expect(page.locator('#rapid-request-rows td')).toHaveAttribute('colspan', '2');
    await expect(page.locator('#rapid-request-rows')).toContainText('No active request details reported');
    await push(page, null, { backend: 'llama_cpp' });
    await expect(page.locator('#rapid-dashboard-details')).toBeHidden();
  });

  test('bounded table columns remain separate on narrow layouts', async ({ page }) => {
    await page.setViewportSize({ width: 430, height: 900 });
    await push(page, { active_requests: [{ id: requestId, phase: 'decode', prompt_tokens: 100,
      completion_tokens: 2, max_tokens: 64, cached_tokens: 0, tokens_per_second: 8, ttft_s: 0.2, elapsed_s: 1, cache_hit_type: 'prefix' }] });
    const cells = await page.locator('#rapid-request-rows td').evaluateAll(nodes => nodes.map(node => {
      const rect = node.getBoundingClientRect();
      return { x: rect.x, right: rect.right, width: rect.width };
    }));
    expect(cells).toHaveLength(10);
    for (let i = 1; i < cells.length; i++) {
      expect(cells[i].width).toBeGreaterThan(0);
      expect(cells[i].x).toBeGreaterThanOrEqual(cells[i - 1].right - 1);
    }
    const overflow = await page.locator('#rapid-dashboard-details').evaluate(el => el.scrollWidth - el.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
