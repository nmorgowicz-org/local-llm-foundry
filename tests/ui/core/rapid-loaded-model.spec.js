import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

const repo = 'mlx-community/Actual-Model-4bit';

async function render(page, { sample = { backend: 'rapid_mlx', model: 'friendly-api-alias' }, identity = 'friendly-api-alias', config = null, session = 'rapid-model-session', running = true, mode = 'spawn', endpoint = 'http://127.0.0.1:8001' } = {}) {
  await page.evaluate(async data => {
    const state = await import('/js/core/app-state.js');
    const nav = await import('/js/features/nav.js');
    state.sessionState.presets = data.config ? [{ id: 'active-rapid', backend: 'rapid_mlx', rapid_mlx: data.config }] : [];
    state.setWsData({ backend: 'rapid_mlx', endpoint_kind: 'Local', session_mode: data.mode,
      active_session_id: data.session, active_session_endpoint_tag: data.endpoint,
      active_session_status: data.running ? 'running' : 'stopped',
      active_session_preset_id: 'active-rapid', active_session_model_identity: data.identity,
      inference: data.sample,
    });
    // Deliberately retain foreign llama facts: Rapid must never use them.
    state.setLastLlamaMetrics({ runtime_facts: { model_name: 'stale-llama.gguf' } });
    nav.refreshTopCockpit();
  }, { sample, identity, config, session, running, mode, endpoint });
}

async function facts(page) {
  return page.locator('#engine-model-facts').evaluate(dl => Object.fromEntries(
    [...dl.querySelectorAll('dt')].map(dt => [dt.textContent, dt.nextElementSibling.textContent])));
}

test.beforeEach(async ({ page }) => {
  await page.routeWebSocket('**/ws', ws => ws.close());
  await page.route('**/api/**', route => route.abort());
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await page.evaluate(async () => (await import('/js/features/setup-view.js')).switchView('monitor'));
  // switchView is animated: the view stays 'transitioning' (~0.9s) and the dashboard drops
  // updates in that window. Poll a real readiness condition from the test side; an async
  // predicate inside waitForFunction would be an always-truthy Promise and wait for nothing.
  await expect(page.locator('#view-monitor')).toBeVisible();
  await expect.poll(() => page.evaluate(async () =>
    (await import('/js/core/app-state.js')).setupViewState.view)).toBe('monitor');
});

test('alias-only is explicitly an alias, not the physical model; unknown facts stay absent', async ({ page }) => {
  await render(page);
  await expect(page.locator('#engine-indicator')).toContainText('Alias:');
  await expect(page.locator('#server-model-identity')).toHaveText('Rapid-MLX · Alias: friendly-api-alias');
  await page.locator('#engine-indicator').click();
  expect(await facts(page)).toEqual({ Engine: 'Rapid-MLX', Model: 'Physical model not reported', Alias: 'friendly-api-alias' });
  await expect(page.locator('#engine-model-popover')).not.toContainText('stale-llama');
  const close = page.getByRole('button', { name: 'Close loaded model details' });
  await expect(close).toHaveClass(/\bbtn\b/);
  await expect(close).toHaveClass(/\bbtn-secondary\b/);
  await close.click();
  await expect(page.locator('#engine-model-popover')).not.toBeVisible();
});

test('real repository wins over served alias and only available facts are rendered', async ({ page }) => {
  await render(page, { sample: { backend: 'rapid_mlx', model: 'friendly-api-alias', backend_details: {
    runtime_facts: { model_type: 'llm', engine_type: 'batched' },
    launch_facts: { repo_id: repo, version: '0.10.0', context_length: 32768, served_model_name: 'friendly-api-alias',
      speculative_config: { method: 'mtp', num_speculative_tokens: 3 } },
  } } });
  await expect(page.locator('#server-model-identity')).toHaveText(`Rapid-MLX · ${repo}`);
  await page.locator('#engine-indicator').click();
  const rows = await facts(page);
  expect(rows.Model).toBe(repo);
  expect(rows.Repository).toBe(repo);
  expect(rows.Alias).toBe('friendly-api-alias');
  expect(rows['Runtime version']).toBe('0.10.0');
  expect(rows['Context (configured)']).toContain('32,768');
  expect(rows['Runtime lane']).toBe('llm · batched');
  expect(rows['Speculative decoding (configured)']).toBe('mtp · 3 tokens');
  expect(rows).not.toHaveProperty('Quantization');
  expect(rows).not.toHaveProperty('Quantization (catalog)');
});

test('typed config and cached model-status resolve catalog alias once, preserving served alias', async ({ page }) => {
  let requests = 0;
  await page.route('**/api/rapid-mlx/model-status?*', route => {
    requests++;
    return route.fulfill({ json: { ok: true, source: 'catalog-alias', repo_id: repo, quant: '4-bit' } });
  });
  const data = { config: { model_source: { kind: 'alias', value: 'catalog-alias' }, served_model_name: 'friendly-api-alias' } };
  await render(page, data);
  await expect(page.locator('#server-model-identity')).toHaveText(`Rapid-MLX · ${repo}`);
  await page.locator('#engine-indicator').click();
  expect((await facts(page))['Quantization (catalog)']).toBe('4-bit');
  expect(await facts(page)).not.toHaveProperty('Quantization');
  expect((await facts(page)).Alias).toBe('friendly-api-alias');
  await render(page, data);
  await render(page, data);
  expect(requests).toBe(1);
});

test('repo config and HF cache snapshot identity beat alias without leaking local directories', async ({ page }) => {
  await render(page, { config: { model_source: { kind: 'hugging_face_repo', repo_id: repo, revision: 'abc123' } } });
  expect((await facts(page)).Model).toBe(repo);
  expect((await facts(page)).Revision).toBe('abc123');
  await render(page, { identity: '/private/cache/hub/models--mlx-community--Actual-Model-4bit/snapshots/abc123' });
  expect((await facts(page)).Model).toBe(repo);
  await expect(page.locator('#engine-model-facts')).not.toContainText('/private');
});

test('late alias lookup cannot repopulate switched or detached model details', async ({ page }) => {
  let complete;
  const pending = new Promise(resolve => { complete = resolve; });
  await page.route('**/api/rapid-mlx/model-status?*', async route => {
    await pending;
    await route.fulfill({ json: { ok: true, repo_id: 'old/Model', quant: '8-bit' } });
  });
  const requested = page.waitForRequest('**/api/rapid-mlx/model-status?*');
  await render(page, { identity: 'slow-catalog-alias' });
  await requested;
  await page.locator('#engine-indicator').click();
  await render(page, { identity: 'new/Model', session: 'new-session', sample: { backend: 'rapid_mlx', model: 'new-alias' } });
  complete();
  await expect(page.locator('#engine-model-facts')).toContainText('new/Model');
  await expect(page.locator('#engine-model-facts')).not.toContainText('old/Model');
  await render(page, { running: false });
  await expect(page.locator('#engine-model-facts')).toBeEmpty();
  await expect(page.locator('#server-model-identity')).not.toBeVisible();
  await expect(page.locator('#engine-model-popover')).not.toBeVisible();
});

test('repo-shaped served aliases remain aliases without identity proof', async ({ page }) => {
  await render(page, { identity: 'owner/served-alias', sample: { backend: 'rapid_mlx', model: 'owner/served-alias' } });
  const rows = await facts(page);
  expect(rows.Model).toBe('Physical model not reported');
  expect(rows.Alias).toBe('owner/served-alias');
  expect(rows).not.toHaveProperty('Repository');
});

test('attached sessions ignore stale spawn preset facts and failed lookups retry only on target change', async ({ page }) => {
  let requests = 0;
  await page.route('**/api/rapid-mlx/model-status?*', route => {
    requests++;
    return route.fulfill({ status: 404, json: { ok: false } });
  });
  const data = { mode: 'attach', config: {
    model_source: { kind: 'hugging_face_repo', repo_id: 'old/Spawned-Model', revision: 'old' },
    served_model_name: 'old-spawned-alias', context_length: 65536,
  } };
  const requested = page.waitForResponse('**/api/rapid-mlx/model-status?*');
  await render(page, data);
  await requested;
  await render(page, data);
  const rows = await facts(page);
  expect(rows.Model).toBe('Physical model not reported');
  expect(rows.Alias).toBe('friendly-api-alias');
  expect(rows).not.toHaveProperty('Context (configured)');
  expect(rows).not.toHaveProperty('Revision');
  expect(requests).toBe(1);
  const next = page.waitForResponse('**/api/rapid-mlx/model-status?*');
  await render(page, { ...data, endpoint: 'http://127.0.0.1:8002' });
  await next;
  expect(requests).toBe(2);
});
