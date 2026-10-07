import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

test('llama updater pill remains clickable and hoverable while monitoring', async ({ page }) => {
  await page.routeWebSocket('**/ws', ws => ws.close());
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const data = path === '/api/llama-binary/version' ? { build: 11436 }
      : path === '/api/llama-binary/latest' ? { build: 11461 }
      : path === '/api/llama-binary/releases' ? { releases: [] } : null;
    if (data) await route.fulfill({ json: data });
    else await route.abort();
  });
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await page.evaluate(async () => {
    const { switchView } = await import('/js/features/setup-view.js');
    switchView('monitor');
  });
  await expect(page.locator('body')).not.toHaveClass(/setup-active/);
  await page.evaluate(async () => {
    const { initWebSocket } = await import('/js/features/dashboard-ws.js');
    const socket = initWebSocket();
    socket.onmessage({ data: JSON.stringify({ backend: 'llama_cpp', session_mode: 'spawn',
      active_session_id: 'pill-fixture', active_session_status: 'running',
      active_session_endpoint: 'http://127.0.0.1:8001', active_session_endpoint_tag: 'http://127.0.0.1:8001',
      active_session_model_identity: 'Qwopus3.8-27B-Flash-V2-Apostate-Uncensored-Q4_K_M.gguf',
      server_running: true, local_server_running: true, host_metrics_available: false,
      inference: null, llama: null, capabilities: {}, gpu: {}, system: null, logs: [], mode: 'off',
    }) });
    socket.close();
  });
  await page.evaluate(async () => {
    const { chat, setLastLlamaMetrics, setContextCapacityTokens } = await import('/js/core/app-state.js');
    const { refreshTopCockpit } = await import('/js/features/nav.js');
    const rapid = document.getElementById('rapid-mlx-pill');
    rapid.style.display = 'flex';
    document.getElementById('rapid-mlx-pill-version').textContent = 'Rapid-MLX · v0.15.6';
    chat.tabs = [{ messages: [{ role: 'assistant', input_tokens: 3000, output_tokens: 1300 }], total_output_tokens: 1300 }];
    setContextCapacityTokens(10000);
    setLastLlamaMetrics({ last_generation_tokens_per_sec: 7, last_prompt_tokens_per_sec: 131, tokens_per_decode: 1.95 });
    refreshTopCockpit();
  });
  const pill = page.locator('#llama-pill');
  await expect(pill).toBeVisible();
  await expect(pill).toHaveAttribute('title', /Update available/);
  await expect(page.locator('#nav-cockpit-gpu, #nav-cockpit-spark')).toHaveCount(0);
  await expect(page.locator('#nav-cockpit-context')).toHaveText('Ctx 43%');
  for (const width of [1920, 1568, 1024, 812, 430]) {
    await page.setViewportSize({ width, height: 950 });
    await expect(page.locator('#nav-cockpit-context')).toBeVisible();
    await expect(page.locator('#nav-home-btn')).toBeVisible();
    const cockpitLayout = await page.locator('#nav-cockpit').evaluate(el => {
      const r = el.getBoundingClientRect();
      const children = [...el.children].filter(child => getComputedStyle(child).display !== 'none');
      el.scrollLeft = 100;
      return {
        overflow: getComputedStyle(el).overflowX,
        scrollLeft: el.scrollLeft,
        lastChild: el.lastElementChild.id,
        chipsFit: children.every(child => {
          const c = child.getBoundingClientRect();
          return c.left >= r.left && c.right <= r.right + 1;
        }),
        trailingSpace: r.right - el.lastElementChild.getBoundingClientRect().right,
      };
    });
    expect(cockpitLayout.overflow).toBe('clip');
    expect(cockpitLayout.scrollLeft).toBe(0);
    expect(cockpitLayout.lastChild).toBe('nav-cockpit-context');
    expect(cockpitLayout.chipsFit).toBe(true);
    if (width >= 1568) expect(cockpitLayout.trailingSpace).toBeLessThanOrEqual(12);
    const hit = await pill.evaluate(el => {
      const r = el.getBoundingClientRect();
      return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2));
    });
    expect(hit).toBe(true);
    await pill.hover({ timeout: 3000 });
  }
  await pill.hover();
  await expect(pill).toHaveJSProperty('disabled', false);
  await pill.click({ timeout: 5000 });
  await expect(page.locator('#llama-version-modal')).toBeVisible();
});
