import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

async function renderBadge(page, { backend = 'llama_cpp', session = 'model-session', running = true, model = 'loaded-model-Q4_K_M.gguf', alias = 'friendly-api-alias', sourceSession = session } = {}) {
  await page.evaluate(async data => {
    const state = await import('/js/core/app-state.js');
    const { refreshTopCockpit } = await import('/js/features/nav.js');
    state.setWsData({ backend: data.backend, endpoint_kind: 'Local',
      active_session_id: data.session, active_session_endpoint_tag: 'http://127.0.0.1:8001',
      active_session_status: data.running ? 'running' : 'stopped',
      active_session_model_identity: '/private/models/session-identity.gguf',
    });
    state.setLastLlamaMetrics({ telemetry_session_id: data.sourceSession,
      telemetry_endpoint: 'http://127.0.0.1:8001', runtime_facts: {
        model_name: data.model ? `/private/models/${data.model}` : null, model_alias: data.alias,
      } });
    refreshTopCockpit();
  }, { backend, session, running, model, alias, sourceSession });
}

test.beforeEach(async ({ page }) => {
  await page.routeWebSocket('**/ws', ws => ws.close());
  await page.route('**/api/**', route => route.abort());
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await page.evaluate(async () => (await import('/js/features/setup-view.js')).switchView('monitor'));
  await expect(page.locator('body')).not.toHaveClass(/setup-active/);
});

test('engine badge shows full model on hover and opens model details, not updater', async ({ page }) => {
  await renderBadge(page);
  const badge = page.getByRole('button', { name: 'View loaded model', exact: true });
  await expect(badge).toBeVisible();
  await badge.hover();
  await expect(page.locator('#global-tooltip')).toBeVisible();
  await expect(page.locator('#global-tooltip')).toContainText('loaded-model-Q4_K_M.gguf');
  await badge.click();
  const details = page.locator('#engine-model-popover');
  await expect(details).toBeVisible();
  await expect(details).toContainText('loaded-model-Q4_K_M.gguf');
  await expect(details).toContainText('friendly-api-alias');
  await expect(details).not.toContainText('/private');
  await expect(page.locator('#llama-version-modal')).not.toBeVisible();
  await page.keyboard.press('Escape');
  await expect(details).not.toBeVisible();
  await badge.focus();
  await page.keyboard.press('Enter');
  await expect(details).toBeVisible();
  await details.getByRole('button', { name: 'Close loaded model details' }).click();
  await expect(details).not.toBeVisible();
  await page.setViewportSize({ width: 430, height: 900 });
  await badge.focus();
  await page.keyboard.press('Space');
  await expect(details).toBeVisible();
  await page.screenshot({ path: 'test-results/engine-model-badge-narrow.png' });
});

test('open details track model switches and clear on detach', async ({ page }) => {
  await renderBadge(page);
  await page.locator('#engine-indicator').click();
  await renderBadge(page, { model: 'next-model.gguf', alias: 'next-alias', session: 'next-session' });
  await expect(page.locator('#engine-model-popover')).toContainText('next-model.gguf');
  await expect(page.locator('#engine-model-popover')).not.toContainText('loaded-model-Q4_K_M.gguf');
  await renderBadge(page, { running: false });
  await expect(page.locator('#engine-indicator')).not.toBeVisible();
  await expect(page.locator('#engine-model-popover')).not.toBeVisible();
  await expect(page.locator('#engine-model-facts')).toBeEmpty();
});

test('foreign telemetry and Rapid-MLX do not expose stale llama model facts', async ({ page }) => {
  await renderBadge(page, { sourceSession: 'previous-session' });
  await page.locator('#engine-indicator').click();
  await expect(page.locator('#engine-model-popover')).toContainText('Model file not reported by server');
  await expect(page.locator('#engine-model-popover')).toContainText('session-identity.gguf');
  await expect(page.locator('#engine-model-popover')).not.toContainText('loaded-model-Q4_K_M.gguf');
  await renderBadge(page, { backend: 'rapid_mlx' });
  await expect(page.locator('#engine-model-popover')).toContainText('Rapid-MLX');
  await expect(page.locator('#engine-model-popover')).not.toContainText('friendly-api-alias');
});
