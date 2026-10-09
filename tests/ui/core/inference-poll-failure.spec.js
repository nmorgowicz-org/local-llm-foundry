import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

const GiB = 1024 ** 3;
const endpoint = 'http://localhost:8001';
const rapidSample = {
  running_requests: 1, waiting_requests: 2,
  generation_tokens_per_second: 24, prompt_tokens_per_second: 80,
  active_requests: [{ id: 'stale-request', phase: 'generating', completion_tokens: 10, max_tokens: 40 }],
  active_memory_bytes: GiB, peak_memory_bytes: 2 * GiB, cache_memory_bytes: GiB / 2,
  backend_details: { memory_limit_bytes: GiB },
  cache_metrics: { hits: 3, misses: 1, hit_rate: 0.75 },
  speculative_acceptance_rate: 0.5,
};
const llamaSample = {
  telemetry_session_id: 'poll-fixture', telemetry_endpoint: endpoint,
  slots_processing: 1, slots_idle: 1, active_task_id: 42, waiting_requests: 2,
  generation_tokens_per_sec: 24, prompt_tokens_per_sec: 80,
  last_generation_tokens_per_sec: 22, last_prompt_tokens_per_sec: 70,
  slot_generation_tokens: 10, slot_generation_limit: 40,
  context_capacity_tokens: 1024, context_live_tokens: 256, context_live_tokens_available: true,
  prompt_tokens_cached_total: 30, prompt_tokens_processed_total: 10,
  slots: [{ id: 0, id_task: 42, is_processing: true, output_available: true, output_tokens: 10, n_ctx: 1024 }],
};

async function push(page, extra) {
  await page.evaluate(async extra => {
    const { initWebSocket } = await import('/js/features/dashboard-ws.js');
    const socket = initWebSocket();
    socket.onmessage({ data: JSON.stringify({
      backend: 'rapid_mlx', session_mode: 'attach', active_session_id: 'poll-fixture',
      active_session_status: 'running', active_session_endpoint: 'http://localhost:8001',
      active_session_endpoint_tag: 'http://localhost:8001',
      server_running: true, local_server_running: false,
      host_metrics_available: false, capabilities: {}, gpu: {}, system: null,
      logs: [], mode: 'off', inference: null, llama: null, ...extra,
    }) });
    socket.close();
  }, extra);
}

async function retained(page) {
  return page.evaluate(async () => {
    const state = await import('/js/core/app-state.js');
    return { llama: state.lastLlamaMetrics, rapid: state.getLastRapidMlxMetrics(),
      frameLlama: state.wsData.llama, frameInference: state.wsData.inference };
  });
}

async function expectUnavailable(page) {
  await expect(page.locator('#state-label')).toHaveText('Telemetry unavailable');
  await expect(page.locator('#state-card .state-pill.active')).toHaveAttribute('data-s', 'unavailable');
  await expect(page.locator('#state-progress')).not.toHaveAttribute('aria-valuenow');
  await expect(page.locator('#mv-decode')).toHaveText('–');
  await expect(page.locator('#mv-prefill')).toHaveText('–');
  await expect(page.locator('#mv-queue')).toHaveText('–');
  expect(await retained(page)).toEqual({ llama: null, rapid: null, frameLlama: null, frameInference: null });
}

test.describe('@fake-data-bypass failed inference polls', () => {
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
    await expect.poll(() => page.evaluate(async () =>
      (await import('/js/core/app-state.js')).setupViewState.view)).toBe('monitor');
  });

  test('Rapid success → failed poll with retained snapshot → fresh success clears and restores all inference surfaces', async ({ page }) => {
    const host = { host_metrics_available: true, capabilities: { system: true }, system: { cpu_load: 37 } };
    await push(page, { inference: rapidSample, ...host }); // Older successful frames omit the flag.
    await expect(page.locator('#state-label')).toHaveText('Generating');
    await expect(page.locator('#rapid-request-rows')).toContainText('stale-request');
    await expect(page.locator('#mv-decode')).toHaveText('24.0');
    await expect(page.locator('#mc-card-runtime-memory')).not.toHaveAttribute('hidden');
    await expect(page.locator('#dashboard-memory-warning')).toContainText('near the explicitly reported allocation cap');

    // Both retained sources are populated deliberately: neither may leak into fallback consumers.
    await push(page, { inference_poll_failed: true, inference: rapidSample, llama: llamaSample, ...host });
    await expectUnavailable(page);
    await expect(page.locator('#rapid-request-rows')).not.toContainText('stale-request');
    await expect(page.locator('#rapid-request-summary')).toHaveText('Active request details unavailable');
    for (const key of ['runtime-memory', 'cache', 'mtp']) {
      await expect(page.locator(`#mc-card-${key}`)).toHaveAttribute('hidden');
      await expect(page.locator(`#mv-${key}`)).toHaveText('–');
      await expect(page.locator(`#ms-${key}`)).toHaveText('');
    }
    await expect(page.locator('#dashboard-memory-warning')).toHaveText('');
    await expect(page.locator('#dashboard-memory-warning')).toBeHidden();
    await expect(page.locator('#mv-cpu')).toHaveText('37'); // Host collection is independent.

    const fresh = { ...rapidSample, generation_tokens_per_second: 31,
      active_memory_bytes: GiB / 4,
      active_requests: [{ id: 'fresh-request', phase: 'generating', completion_tokens: 4, max_tokens: 40 }] };
    await push(page, { inference_poll_failed: false, inference: fresh, ...host });
    await expect(page.locator('#state-label')).toHaveText('Generating');
    await expect(page.locator('#mv-decode')).toHaveText('31.0');
    await expect(page.locator('#rapid-request-rows')).toContainText('fresh-request');
    await expect(page.locator('#rapid-request-rows')).not.toContainText('stale-request');
    await expect(page.locator('#mc-card-runtime-memory')).not.toHaveAttribute('hidden');
    await expect(page.locator('#mv-runtime-memory')).toHaveText('256');
    await expect(page.locator('#mu-runtime-memory')).toHaveText('MiB active');
  });

  test('llama.cpp success → failed retained poll → success clears shared rates and legacy slot/efficiency details', async ({ page }) => {
    await push(page, { backend: 'llama_cpp', llama: llamaSample });
    await expect(page.locator('#state-label')).toHaveText('Generating');
    await expect(page.locator('#m-slot-grid .slot-tile.busy')).toHaveCount(1);
    await expect(page.locator('#m-slot-grid')).toContainText('task 42');
    await expect(page.locator('#m-slot-util')).toHaveText('50%');
    await expect(page.locator('#cache-reuse-value')).toHaveText('75.0%');

    await push(page, { backend: 'llama_cpp', inference_poll_failed: true,
      llama: llamaSample, inference: rapidSample });
    await expectUnavailable(page);
    await expect(page.locator('#m-slot-grid .slot-tile.busy')).toHaveCount(0);
    await expect(page.locator('#m-slot-grid')).not.toContainText('task 42');
    await expect(page.locator('#m-slot-util')).toHaveText('—');
    await expect(page.locator('#prompt-cache-reuse')).toHaveAttribute('hidden');
    await expect(page.locator('#cache-reuse-value')).toHaveText('—');
    await expect(page.locator('#m-activity-rail .activity-segment.active')).toHaveCount(0);
    expect(await page.evaluate(async () =>
      (await import('/js/core/app-state.js')).requestActivity.length)).toBe(0);

    await push(page, { backend: 'llama_cpp', inference_poll_failed: false,
      llama: { ...llamaSample, generation_tokens_per_sec: 31, active_task_id: 43,
        slots: [{ ...llamaSample.slots[0], id_task: 43 }] } });
    await expect(page.locator('#state-label')).toHaveText('Generating');
    await expect(page.locator('#mv-decode')).toHaveText('31.0');
    await expect(page.locator('#m-slot-grid')).toContainText('task 43');
    await expect(page.locator('#prompt-cache-reuse')).not.toHaveAttribute('hidden');
  });

  test('failed frames invalidate stored sources even while hidden, then render unavailable on show', async ({ page }) => {
    await push(page, { inference: rapidSample });
    await expect(page.locator('#state-label')).toHaveText('Generating');
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { configurable: true, value: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await push(page, { inference_poll_failed: true, inference: rapidSample, llama: llamaSample });
    expect(await retained(page)).toEqual({ llama: null, rapid: null, frameLlama: null, frameInference: null });
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { configurable: true, value: false });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await expectUnavailable(page);
    await push(page, { inference_poll_failed: false, inference: rapidSample });
    await expect(page.locator('#state-label')).toHaveText('Generating');
  });

  test('explicit Rapid telemetry unavailable clears stale normalized requests, rates and runtime warnings', async ({ page }) => {
    const out = await page.evaluate(async sample => {
      const { normalizeRapidDashboard, renderRapidDashboard } = await import('/js/features/rapid-dashboard.js');
      const owner = { attached: true, backend: 'rapid_mlx', sessionId: 'direct-fixture' };
      renderRapidDashboard(sample, owner);
      renderRapidDashboard({ ...sample, telemetry_unavailable: true }, owner);
      return normalizeRapidDashboard({ ...sample, telemetry_unavailable: true });
    }, rapidSample);
    expect(out.state).toBe('unavailable');
    expect(out.requests).toEqual([]);
    expect(out.running).toBeNull();
    expect(out.waiting).toBeNull();
    expect(out.decodeTps).toBeNull();
    expect(out.prefillTps).toBeNull();
    expect(out.generatingCount).toBe(0);
    expect(out.memory.active).toBeNull();
    await expect(page.locator('#rapid-request-rows')).not.toContainText('stale-request');
    await expect(page.locator('#dashboard-memory-warning')).toHaveText('');
    await page.evaluate(async sample => {
      const { renderRapidDashboard } = await import('/js/features/rapid-dashboard.js');
      renderRapidDashboard(sample, { attached: true, backend: 'rapid_mlx', sessionId: 'direct-fixture' });
    }, rapidSample);
    await expect(page.locator('#rapid-request-rows')).toContainText('stale-request');
    await expect(page.locator('#dashboard-memory-warning')).toContainText('near the explicitly reported allocation cap');
  });
});

test('compact clears stale inference rates and context on failed poll and recovers on success', async ({ page }) => {
  let socket;
  const frame = (failed, llama) => JSON.stringify({ server_running: true, inference_poll_failed: failed,
    capabilities: { gpu: false, system: false }, llama });
  await page.routeWebSocket('**/ws', ws => {
    socket = ws;
    ws.send(frame(false, llamaSample));
  });
  await page.goto('/compact');
  await expect(page.locator('#inf-generate')).toHaveText('24.0 tok/s');
  await expect(page.locator('#inf-context')).toHaveText('256 / 1024 (25%)');
  socket.send(frame(true, llamaSample));
  await expect(page.locator('#inf-generate')).toHaveText('— tok/s');
  await expect(page.locator('#inf-prompt')).toHaveText('— tok/s');
  await expect(page.locator('#inf-context')).toHaveText('— %');
  socket.send(frame(false, { ...llamaSample, generation_tokens_per_sec: 31 }));
  await expect(page.locator('#inf-generate')).toHaveText('31.0 tok/s');
  await expect(page.locator('#inf-context')).toHaveText('256 / 1024 (25%)');
});
