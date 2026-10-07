import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

const sample = {
  prompt_tokens_cached_total: 300,
  prompt_tokens_processed_total: 100,
  speculative_draft_tokens_total: 200,
  speculative_accepted_tokens_total: 150,
  speculative_verification_steps_total: 50,
  slots: [{ speculative_enabled: true, speculative_type: 'draft-mtp' }],
  runtime_facts: {
    model_name: 'attached-model.gguf', quantization: 'Q4_K_M', model_params: 8000000000,
    server_build: 'b7000', capabilities: { vision: true, tools: true },
    adapters: [{ id: 0, name: 'style.gguf', scale: 0.8 }],
  },
};

const fullCapabilities = {
  audio: false, supports_object_arguments: true, supports_parallel_tool_calls: true,
  supports_preserve_reasoning: true, supports_reasoning_effort: true,
  supports_string_content: true, supports_system_role: true, supports_tool_calls: true,
  supports_tools: true, supports_typed_content: true, video: true, vision: true,
};

async function render(page, metrics, backend = 'llama_cpp', sessionId = 'llama-a', attached = true) {
  const endpoint = `http://localhost/${sessionId}`;
  const taggedMetrics = metrics ? { telemetry_session_id: sessionId, telemetry_endpoint: endpoint, ...metrics } : metrics;
  await page.evaluate(async args => {
    const { renderLlamaCppDetails } = await import('/js/features/llama-cpp-details.js');
    renderLlamaCppDetails(args);
  }, { metrics: taggedMetrics, backend, sessionId, endpoint, endpointTag: endpoint, attached });
}

async function pushFrame(page, frame) {
  await page.evaluate(async data => {
    const { initWebSocket } = await import('/js/features/dashboard-ws.js');
    const socket = initWebSocket();
    socket.onmessage({ data: JSON.stringify(data) });
    socket.close();
  }, {
    backend: 'llama_cpp', session_mode: 'attach', active_session_id: 'fixture-session',
    active_session_status: 'running', server_running: true, local_server_running: false,
    inference: null, host_metrics_available: false, capabilities: {}, gpu: {},
    system: null, logs: [], mode: 'off', ...frame,
  });
}

test.describe('@fake-data-bypass llama.cpp inference efficiency', () => {
  test.beforeEach(async ({ page }) => {
    // Hold the external telemetry transport so real pushes cannot replace deterministic fixtures.
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
    await page.waitForFunction(async () => {
      const { setupViewState } = await import('/js/core/app-state.js');
      return setupViewState.view === 'monitor';
    });
  });

  test('uses server totals and verification yield, not speedup', async ({ page }) => {
    await render(page, sample);
    await expect(page.locator('#inference-efficiency')).toBeVisible();
    await expect(page.locator('#prompt-cache-reuse')).toContainText('75.0%');
    await expect(page.locator('#prompt-cache-reuse')).toContainText('300 cached');
    await expect(page.locator('#prompt-cache-reuse')).toContainText('100 processed');
    await expect(page.locator('#speculative-effectiveness')).toContainText('75.0%');
    await expect(page.locator('#spec-tokens-per-verification')).toHaveText('4.00');
    await expect(page.locator('#speculative-effectiveness')).toContainText('Not a speedup measurement');
    await expect(page.locator('#inference-efficiency')).toContainText('Server totals');
    await expect(page.locator('#cache-reuse-bar')).toHaveAttribute('aria-valuenow', '75');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('role', 'progressbar');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('aria-valuenow', '75');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('aria-valuetext', '75.0% accepted');
    const fillFraction = await page.evaluate(() => {
      const bar = document.getElementById('spec-acceptance-bar');
      const fill = document.getElementById('spec-acceptance-fill');
      return fill.getBoundingClientRect().width / bar.clientWidth;
    });
    expect(fillFraction).toBeCloseTo(0.75, 2);
  });

  test('hides absent counters but supported zero counters await activity', async ({ page }) => {
    await render(page, { ...sample, prompt_tokens_cached_total: null, speculative_draft_tokens_total: null, speculative_verification_steps_total: null });
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await render(page, {
      ...sample, prompt_tokens_cached_total: 0, prompt_tokens_processed_total: 0,
      speculative_draft_tokens_total: 0, speculative_accepted_tokens_total: 0, speculative_verification_steps_total: 0,
    });
    await expect(page.locator('#prompt-cache-reuse')).toContainText('Awaiting activity');
    await expect(page.locator('#speculative-effectiveness')).toContainText('Awaiting activity');
    await expect(page.locator('#cache-reuse-bar')).not.toHaveAttribute('aria-valuenow');
    await expect(page.locator('#spec-tokens-per-verification')).toHaveText('—');
    await expect(page.locator('#spec-acceptance-bar')).toBeVisible();
    await expect(page.locator('#spec-acceptance-bar')).not.toHaveAttribute('aria-valuenow');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('aria-valuetext', 'Awaiting activity');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('data-awaiting', 'true');
    await expect(page.locator('#spec-acceptance-fill')).toHaveCSS('width', '0px');
  });

  test('disabled speculation hides companion and cache spans the row', async ({ page }) => {
    await render(page, { ...sample, slots: [{ speculative_enabled: false, speculative_type: 'none' }] });
    await expect(page.locator('#speculative-effectiveness')).toBeHidden();
    await expect(page.locator('#prompt-cache-reuse')).toBeVisible();
    await expect(page.locator('#efficiency-grid')).toHaveAttribute('data-card-count', '1');
    const widths = await page.evaluate(() => [
      document.getElementById('efficiency-grid').getBoundingClientRect().width,
      document.getElementById('prompt-cache-reuse').getBoundingClientRect().width,
    ]);
    expect(Math.abs(widths[0] - widths[1])).toBeLessThan(2);
  });

  test('partial speculation only shows available measures', async ({ page }) => {
    await render(page, { ...sample, speculative_verification_steps_total: null });
    await expect(page.locator('#speculative-effectiveness')).toContainText('75.0%');
    await expect(page.locator('#spec-verification-metric')).toBeHidden();
    await expect(page.locator('#spec-acceptance-bar')).toBeVisible();
    await render(page, { ...sample, speculative_draft_tokens_total: null });
    await expect(page.locator('#spec-acceptance-metric')).toBeHidden();
    await expect(page.locator('#spec-acceptance-bar')).toBeHidden();
    await expect(page.locator('#spec-acceptance-bar')).not.toHaveAttribute('aria-valuenow');
    await expect(page.locator('#spec-tokens-per-verification')).toHaveText('4.00');
  });

  test('runtime card shows attached facts safely without a disclosure', async ({ page }) => {
    await render(page, { ...sample, model_name: 'stale-local-preset.gguf' });
    const card = page.locator('#llama-runtime-card');
    await expect(card).toBeVisible();
    await expect(card.locator('.llama-runtime-capability')).toHaveText(['Vision', 'Tools']);
    await page.getByRole('button', { name: 'View full runtime details' }).click();
    const details = page.locator('#llama-runtime-popover');
    await expect(details).toBeVisible();
    await expect(page.locator('#server-header details')).toHaveCount(0);
    await expect(details.locator('summary')).toHaveCount(0);
    await expect(details).toContainText('8B');
    await expect(details).not.toContainText('8,000,000,000');
    await expect(details).toContainText('attached-model.gguf');
    await expect(details).toContainText('Q4_K_M');
    await expect(details).toContainText('b7000');
    await expect(details.locator('.llama-runtime-capability')).toHaveText(['Vision', 'Tools']);
    await render(page, { ...sample, runtime_facts: { ...sample.runtime_facts, capabilities: { vision: true, audio: false } } });
    await expect(details.locator('.llama-runtime-capability')).toHaveText(['Vision', 'Audio (unsupported)']);
    await expect(details).not.toContainText('stale-local-preset');
    await render(page, { ...sample, runtime_facts: { model_name: '<img src=x onerror=alert(1)>', model_path: '/private/model.gguf', context: 8192 } });
    await expect(details.locator('img')).toHaveCount(0);
    await expect(details.locator('.llama-runtime-capability')).toHaveCount(0);
    await expect(details).toContainText('<img src=x onerror=alert(1)>');
    await expect(details).not.toContainText('/private');
    await expect(details).not.toContainText('8192');
    await expect(details).toBeVisible();
  });

  test('strict backend gating clears cards and facts on switching', async ({ page }) => {
    await render(page, sample);
    for (const backend of ['rapid_mlx', 'unknown', null]) {
      await render(page, sample, backend);
      await expect(page.locator('#inference-efficiency')).toBeHidden();
      await expect(page.locator('#llama-runtime-card')).toBeHidden();
      await expect(page.locator('#llama-runtime-facts')).toBeEmpty();
      await expect(page.locator('#cache-reuse-value')).toHaveText('—');
      await expect(page.locator('#spec-acceptance-bar')).not.toHaveAttribute('aria-valuenow');
    }
    await render(page, null, 'llama_cpp', 'llama-b');
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await render(page, sample, 'llama_cpp', 'llama-b');
    await expect(page.locator('#llama-runtime-card')).toBeVisible();
    await render(page, sample, 'llama_cpp', 'llama-b', false);
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await expect(page.locator('#llama-runtime-facts')).toBeEmpty();
  });

  test('rejects malformed and negative counters without inventing activity', async ({ page }) => {
    await render(page, { ...sample, prompt_tokens_cached_total: '300', speculative_accepted_tokens_total: -1 });
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await render(page, { ...sample, runtime_facts: null, speculative_accepted_tokens_total: 0 });
    await expect(page.locator('#cache-reuse-value')).toHaveText('75.0%');
    await expect(page.locator('#spec-acceptance-value')).toHaveText('0.0%');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('aria-valuenow', '0');
    await expect(page.locator('#spec-acceptance-bar')).toHaveAttribute('data-awaiting', 'false');
    await expect(page.locator('#spec-acceptance-fill')).toHaveCSS('width', '0px');
    await expect(page.locator('#spec-tokens-per-verification')).toHaveText('1.00');
    await expect(page.locator('#llama-runtime-card')).toBeHidden();
  });

  test('explicitly disabled configuration overrides retained slot configuration', async ({ page }) => {
    await render(page, { ...sample, speculative_enabled: false });
    await expect(page.locator('#speculative-effectiveness')).toBeHidden();
    await render(page, { ...sample, speculative_enabled: true, slots: [] });
    await expect(page.locator('#speculative-effectiveness')).toBeVisible();
  });

  test('WebSocket frames drive facts in attach and spawn modes and clear foreign metrics', async ({ page }) => {
    const push = async (backend, sessionMode, metrics = sample, endpoint = 'http://127.0.0.1:8001') => {
      await page.evaluate(async frame => {
        const { initWebSocket } = await import('/js/features/dashboard-ws.js');
        const socket = initWebSocket();
        socket.onmessage({ data: JSON.stringify(frame) });
        socket.close();
      }, {
        backend, session_mode: sessionMode, active_session_id: 'fixture-session',
        active_session_endpoint: endpoint, active_session_status: 'running',
        active_session_endpoint_tag: endpoint,
        active_session_model_identity: '/private/actual-attached.gguf',
        server_running: true, local_server_running: sessionMode === 'spawn',
        llama: metrics ? { telemetry_session_id: 'fixture-session', telemetry_endpoint: endpoint, ...metrics } : metrics,
        inference: null, host_metrics_available: false,
        capabilities: {}, gpu: {}, system: null, logs: [], mode: 'off',
      });
    };
    await push('llama_cpp', 'attach');
    await expect(page.locator('#cache-reuse-value')).toHaveText('75.0%');
    await expect(page.locator('#llama-runtime-card')).toBeVisible();
    await push('llama_cpp', 'spawn');
    await expect(page.locator('#llama-runtime-card')).toBeVisible();
    await expect(page.locator('#server-endpoint-label')).toBeHidden();
    await push('llama_cpp', 'attach', null, 'http://127.0.0.1:8002');
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await expect(page.locator('#server-model-identity')).toContainText('actual-attached.gguf');
    await expect(page.locator('#llama-runtime-card')).toBeVisible();
    await expect(page.locator('#server-model-identity')).not.toContainText('/private');
    await push('rapid_mlx', 'attach');
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await expect(page.locator('#llama-runtime-facts')).toBeEmpty();
    const retained = await page.evaluate(async () => {
      const state = await import('/js/core/app-state.js');
      return state.lastLlamaMetrics;
    });
    expect(retained).toBeNull();
  });

  test('rejects tagged telemetry from an old session or endpoint', async ({ page }) => {
    const tagged = { ...sample, telemetry_session_id: 'llama-a', telemetry_endpoint: 'http://localhost/llama-a/' };
    await render(page, tagged);
    await expect(page.locator('#cache-reuse-value')).toHaveText('75.0%');
    await render(page, { ...tagged, telemetry_session_id: 'old-session' });
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await expect(page.locator('#llama-runtime-facts')).toBeEmpty();
    await render(page, { ...tagged, telemetry_endpoint: 'http://localhost/old-endpoint' });
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await expect(page.locator('#llama-runtime-facts')).toBeEmpty();
    await render(page, tagged);
    await expect(page.locator('#spec-tokens-per-verification')).toHaveText('4.00');
  });

  test('opaque endpoint tags preserve credentials and query telemetry and reject changed targets', async ({ page }) => {
    const firstTag = 'sha256:' + 'a'.repeat(64);
    const secondTag = 'sha256:' + 'b'.repeat(64);
    const metrics = {
      ...sample, telemetry_session_id: 'fixture-session', telemetry_endpoint: firstTag,
      generation_tokens_per_sec: 48, prompt_tokens_per_sec: 960, context_capacity_tokens: 4096,
    };
    for (const endpoint of [
      'http://user:password@localhost:8001/',
      'http://localhost:8001/?api_key=secret',
      'http://localhost:8001/#fragment',
    ]) {
      await pushFrame(page, {
        active_session_endpoint: endpoint, active_session_endpoint_tag: firstTag, llama: metrics,
      });
      await expect(page.locator('#cache-reuse-value')).toHaveText('75.0%');
      const retained = await page.evaluate(async () => {
        const state = await import('/js/core/app-state.js');
        return { speed: state.lastLlamaMetrics?.generation_tokens_per_sec, context: state.contextCapacityTokens };
      });
      expect(retained).toEqual({ speed: 48, context: 4096 });
    }
    await pushFrame(page, {
      active_session_endpoint: 'http://localhost:8001/?api_key=new-secret',
      active_session_endpoint_tag: secondTag, llama: metrics,
    });
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    const stale = await page.evaluate(async () => (await import('/js/core/app-state.js')).lastLlamaMetrics);
    expect(stale).toBeNull();
  });

  test('default snapshots are rejected and retained tagged legacy metrics stay scoped', async ({ page }) => {
    const tag = 'http://127.0.0.1:8001';
    await pushFrame(page, {
      active_session_endpoint: tag, active_session_endpoint_tag: tag,
      llama: { generation_tokens_per_sec: 48, telemetry_session_id: null, telemetry_endpoint: null },
    });
    const defaults = await page.evaluate(async () => (await import('/js/core/app-state.js')).lastLlamaMetrics);
    expect(defaults).toBeNull();
    const retained = {
      generation_tokens_per_sec: 48, telemetry_session_id: 'fixture-session', telemetry_endpoint: tag,
      prompt_tokens_cached_total: null, runtime_facts: null, slots: [],
    };
    await pushFrame(page, {
      active_session_endpoint: tag, active_session_endpoint_tag: tag, llama: retained,
    });
    const retainedSpeed = await page.evaluate(async () => (await import('/js/core/app-state.js')).lastLlamaMetrics?.generation_tokens_per_sec);
    expect(retainedSpeed).toBe(48);
    await expect(page.locator('#inference-efficiency')).toBeHidden();
    await pushFrame(page, {
      active_session_endpoint: 'http://127.0.0.1:8002', active_session_endpoint_tag: 'http://127.0.0.1:8002',
      llama: retained,
    });
    const stale = await page.evaluate(async () => (await import('/js/core/app-state.js')).lastLlamaMetrics);
    expect(stale).toBeNull();
  });

  test('stale facts cannot reveal the spawn identity header', async ({ page }) => {
    await pushFrame(page, {
      session_mode: 'spawn', local_server_running: true,
      active_session_endpoint: 'http://127.0.0.1:8002',
      active_session_endpoint_tag: 'http://127.0.0.1:8002',
      llama: { ...sample, telemetry_session_id: 'previous-session', telemetry_endpoint: 'http://127.0.0.1:8001' },
    });
    await expect(page.locator('#server-header')).toBeHidden();
    await expect(page.locator('#llama-runtime-facts')).toBeEmpty();
  });
  test('runtime card spans two hardware columns and survives grid initialization', async ({ page }) => {
    await render(page, sample);
    const card = page.locator('#metrics-grid > #llama-runtime-card');
    await expect(card).toBeVisible();
    const geometry = await page.evaluate(async () => {
      const { initMetricCards } = await import('/js/features/metric-cards.js');
      const card = document.getElementById('llama-runtime-card');
      initMetricCards();
      const hardware = document.querySelector('#metrics-grid > .mcard:not(.llama-runtime-card)');
      return { preserved: card === document.getElementById('llama-runtime-card'),
        width: card.getBoundingClientRect().width, hardwareWidth: hardware.getBoundingClientRect().width,
        top: card.getBoundingClientRect().top, hardwareTop: hardware.getBoundingClientRect().top,
        gap: parseFloat(getComputedStyle(card.parentElement).columnGap) };
    });
    expect(geometry.preserved).toBe(true);
    expect(geometry.width).toBeCloseTo(geometry.hardwareWidth * 2 + geometry.gap, 0);
    expect(geometry.top).toBeGreaterThan(geometry.hardwareTop);
  });

  test('narrow runtime card is full width and has no overflowing facts', async ({ page }) => {
    await page.setViewportSize({ width: 430, height: 900 });
    await render(page, sample);
    const card = page.locator('#llama-runtime-card');
    await expect(card).toBeVisible();
    const geometry = await card.evaluate(el => ({
      width: el.getBoundingClientRect().width,
      gridWidth: el.parentElement.getBoundingClientRect().width,
      overflow: el.scrollWidth - el.clientWidth,
    }));
    expect(geometry.width).toBeGreaterThan(250);
    expect(geometry.width).toBeCloseTo(geometry.gridWidth, 0);
    expect(geometry.overflow).toBeLessThanOrEqual(1);
    await expect(page.locator('#server-header details')).toHaveCount(0);
  });

  test('full capabilities never change hardware heights and details stay out of flow', async ({ page }) => {
    const heights = () => page.locator('#metrics-grid > .mcard:not(.llama-runtime-card)').evaluateAll(els =>
      els.map(el => el.getBoundingClientRect().height));
    await render(page, { ...sample, runtime_facts: null });
    const before = await heights();
    const longModel = 'a-very-long-real-attached-model-name-that-must-not-resize-the-grid.gguf';
    await render(page, { ...sample, runtime_facts: {
      model_name: longModel, quantization: 'Q4_K - Medium', model_params: 27320000000,
      server_build: 'b11436-b9a5a00b8', capabilities: fullCapabilities, adapters: [],
    } });
    const card = page.locator('#llama-runtime-card');
    expect(await heights()).toEqual(before);
    await expect(card.locator('.llama-runtime-capability')).toHaveText(['Vision', 'Video', 'Tools', 'Reasoning']);
    await expect(card).not.toContainText('supports_');
    await expect(card).not.toContainText('unsupported');
    await expect(card).not.toContainText('None');
    await expect(card).toContainText('27.32B parameters');
    await expect(card).not.toContainText('b9a5a00b8');
    await expect(page.locator('#llama-runtime-model')).toHaveAttribute('title', longModel);
    const overflow = await card.evaluate(el => el.scrollHeight - el.clientHeight);
    expect(overflow).toBeLessThanOrEqual(1);
    await page.getByRole('button', { name: 'View full runtime details' }).click();
    const popover = page.locator('#llama-runtime-popover');
    await expect(popover).toBeVisible();
    await expect(popover).toContainText('b11436-b9a5a00b8');
    await expect(popover).toContainText(longModel);
    await expect(popover).toContainText('Audio (unsupported)');
    await expect(popover).toContainText('Parallel tool calls');
    await expect(popover).toContainText('None');
    expect(await heights()).toEqual(before);
    await page.keyboard.press('Escape');
    await expect(popover).toBeHidden();
    await page.getByRole('button', { name: 'View full runtime details' }).click();
    await render(page, sample, 'rapid_mlx');
    await expect(popover).toBeHidden();
    await expect(card).toBeHidden();
  });

  test('adapter indicator only reports loaded adapters and unsupported flags make no summary badges', async ({ page }) => {
    await render(page, { ...sample, runtime_facts: { ...sample.runtime_facts,
      capabilities: { vision: false, supports_preserve_reasoning: true, supports_reasoning_effort: false },
    } });
    const card = page.locator('#llama-runtime-card');
    await expect(card).toContainText('1 adapter');
    await expect(card.locator('.llama-runtime-capability')).toHaveCount(0);
    await render(page, { ...sample, runtime_facts: { ...sample.runtime_facts, adapters: [] } });
    await expect(card).not.toContainText('adapter');
  });

  test('model filenames and aliases remain separate, including alias-only servers', async ({ page }) => {
    await render(page, { ...sample, runtime_facts: { ...sample.runtime_facts,
      model_name: '/private/models/reported-model-Q4_K_M.gguf', model_alias: '200k-emm-five',
    } });
    await expect(page.locator('#llama-runtime-model')).toHaveText('reported-model-Q4_K_M.gguf');
    await page.getByRole('button', { name: 'View full runtime details' }).click();
    const facts = page.locator('#llama-runtime-facts');
    await expect(facts.locator('.llama-runtime-fact').filter({ has: page.locator('dt', { hasText: /^Model$/ }) })).toContainText('reported-model-Q4_K_M.gguf');
    await expect(facts.locator('.llama-runtime-fact').filter({ has: page.locator('dt', { hasText: /^Alias$/ }) })).toContainText('200k-emm-five');
    await expect(facts).not.toContainText('/private');
    await render(page, { ...sample, runtime_facts: { model_alias: 'alias-only' } });
    await expect(page.locator('#llama-runtime-model')).toHaveText('Alias: alias-only');
    await expect(facts.locator('dt', { hasText: /^Model$/ })).toHaveCount(0);
    await expect(facts.locator('dt', { hasText: /^Alias$/ })).toHaveCount(1);
  });

  test('metric history is scoped to backend, session, endpoint and attachment', async ({ page }) => {
    await page.route('**/api/**', route => route.abort());
    let time = Date.parse('2026-10-07T12:00:00Z');
    const base = {
      backend: 'llama_cpp', sessionId: 'history-a', endpointTag: 'target-a', attached: true,
      state: 'generating', decodeTps: 100, prefillTps: 50, running: 1, waiting: 2,
      gpu: { load: 60, name: 'fixture GPU', vramUsed: 4, unifiedTotal: 16, vramTotal: 12, temp: 40, power: 30 },
      sys: { cpu: 20, cpuName: 'fixture CPU' },
    };
    const update = async view => {
      await page.clock.setFixedTime(new Date(time));
      await page.evaluate(async value => {
        const { updateMetricCards } = await import('/js/features/metric-cards.js');
        updateMetricCards(value);
      }, view);
    };
    const paths = async () => page.locator('#metrics-grid .mcard__spark-line').evaluateAll(
      nodes => nodes.map(node => node.getAttribute('d') || ''),
    );
    for (const changed of [
      { backend: 'rapid_mlx' }, { sessionId: 'history-b' },
      { endpointTag: 'target-b' }, { attached: false },
    ]) {
      await update(base);
      time += 1000;
      await update(base);
      expect((await paths()).every(path => path.startsWith('M'))).toBe(true);
      // Switch inside the previous speed throttle: the first new sample must
      // still be accepted, but one sample cannot draw an old-target segment.
      time += 100;
      const next = { ...base, ...changed, decodeTps: 10, waiting: 1 };
      await update(next);
      expect((await paths()).every(path => path === '')).toBe(true);
      time += 1000;
      await update(next);
      expect((await paths()).every(path => path.startsWith('M'))).toBe(true);
      await expect(page.locator('#mc-card-speed > .mcard__spark > .mcard__spark-line')).toHaveAttribute(
        'd', 'M0.00,4.00L100.00,4.00',
      );
      time += 1000;
    }
  });

  test('WebSocket target changes reset normalized dashboard histories', async ({ page }) => {
    await page.route('**/api/**', route => route.fulfill({ json: {} }));
    let time = Date.parse('2026-10-07T13:00:00Z');
    const push = async (sessionId, endpointTag, rate = 100) => {
      await page.clock.setFixedTime(new Date(time));
      await pushFrame(page, {
        mode: 'live',
        active_session_id: sessionId, active_session_endpoint: 'http://127.0.0.1:8001',
        active_session_endpoint_tag: endpointTag,
        llama: {
          telemetry_session_id: sessionId, telemetry_endpoint: endpointTag,
          slots_processing: 1, slot_generation_tokens: 20, generation_tokens_per_sec: rate,
          last_prompt_tokens_per_sec: 250,
          waiting_requests: 1,
        },
      });
    };
    const speed = page.locator('#mc-card-speed > .mcard__spark > .mcard__spark-line');
    // The first live frame starts a view transition; the owner changes below
    // reset any history it produced.
    await push('warmup', 'opaque-warmup');
    // The fixed test clock freezes the transition's timers, so settle it directly.
    await page.evaluate(async () => { (await import('/js/core/app-state.js')).setupViewState.view = 'monitor'; });
    await push('history-a', 'opaque-target-a');
    await expect(page.locator('#ms-prefill')).toHaveText('Prefill last measured');
    time += 1000;
    await push('history-a', 'opaque-target-a');
    await expect(speed).toHaveAttribute('d', /^M/);
    time += 100;
    await push('history-b', 'opaque-target-a', 10);
    await expect(speed).toHaveAttribute('d', '');
    time += 1000;
    await push('history-b', 'opaque-target-a', 10);
    await expect(speed).toHaveAttribute('d', 'M0.00,4.00L100.00,4.00');
    time += 100;
    await push('history-b', 'opaque-target-b', 10);
    await expect(speed).toHaveAttribute('d', '');
  });

});
