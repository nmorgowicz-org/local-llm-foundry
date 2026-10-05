import { test, expect } from '@playwright/test';
import { dismissAuthShell, openSettings } from '../helpers.js';

// Only the model-root relocation API is mocked, and execute is fail-closed:
// this test can never move real model files.
const PLAN_ID = 'a'.repeat(64);

const NOTIFICATION_ID = 'model-root-move-pending';

async function notificationState(page) {
  return page.evaluate(id => {
    const saved = JSON.parse(localStorage.getItem('llama-monitor-notifications') || '{}');
    return {
      active: (saved.active || []).some(item => item.id === id),
      archived: (saved.archived || []).some(item => item.id === id),
    };
  }, NOTIFICATION_ID);
}

async function mockRelocation(page, status) {
  await page.route(/\/api\/models\/root-relocation\//, async route => {
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/status')) return route.fulfill({ json: status });
    return route.fulfill({ status: 501, json: { error: 'unmocked' } });
  });
}

const LEGACY_STATUS = {
  source: '/fixture/legacy/models', destination: '/fixture/foundry/models',
  legacy_root: '/fixture/legacy/models', source_exists: true, selection: null,
  custom_root: false, relocation_required: true, move_pending: false,
};

test('notification stays until models are moved, and survives a reload', async ({ page }) => {
  await mockRelocation(page, LEGACY_STATUS);
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await expect(page.locator('.toast', { hasText: 'Models still in the old folder' })).toBeVisible();
  expect((await notificationState(page)).active).toBe(true);
  await page.reload();
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  expect((await notificationState(page)).active).toBe(true);
});

test('choosing keep-legacy does not silence the reminder', async ({ page }) => {
  await mockRelocation(page, { ...LEGACY_STATUS, selection: { choice: 'keep_legacy' } });
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await expect(page.locator('.toast', { hasText: 'Models still in the old folder' })).toBeVisible();
  expect((await notificationState(page)).active).toBe(true);
});

test('no reminder once models live in the Foundry home', async ({ page }) => {
  await mockRelocation(page, {
    ...LEGACY_STATUS,
    source: '/fixture/foundry/models',
    relocation_required: false,
    selection: { choice: 'move_into_foundry' },
  });
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await expect(page.locator('.toast', { hasText: 'Models still in the old folder' })).toHaveCount(0);
  expect((await notificationState(page)).active).toBe(false);
});

test('model library choice is a move, never a copy', async ({ page }) => {
  const calls = [];
  await page.route(/\/api\/models\/root-relocation\//, async route => {
    const url = new URL(route.request().url());
    calls.push({ path: url.pathname, body: route.request().postData() });
    if (url.pathname.endsWith('/status')) {
      return route.fulfill({ json: {
        source: '/fixture/legacy/models', destination: '/fixture/foundry/models',
        legacy_root: '/fixture/legacy/models', source_exists: true, selection: null,
        custom_root: false, relocation_required: true, move_pending: false,
      } });
    }
    if (url.pathname.endsWith('/preview')) {
      return route.fulfill({ json: {
        plan_id: PLAN_ID, choice: 'move_into_foundry', entries: [{}, {}, {}],
        required_copy_bytes: 0, total_move_bytes: 123456, retained_external_roots: [],
      } });
    }
    if (url.pathname.endsWith('/execute')) {
      return route.fulfill({ json: { ok: true, restart_required: true, receipt: {} } });
    }
    return route.fulfill({ status: 501, json: { error: 'unmocked' } });
  });
  await page.route('**/api/db/admin-token', route => route.fulfill({ json: { token: 'fixture-token' } }));
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await openSettings(page, 'migration');

  const card = page.locator('#model-root-relocation-card');
  await expect(card).toContainText('Move into Foundry');
  await expect(card).not.toContainText('Copy into Foundry');
  await expect(card).toContainText('Nothing is copied');

  await card.locator('input[value="move_into_foundry"]').check();
  await page.locator('#model-root-relocation-preview').click();
  const plan = page.locator('#model-root-relocation-plan');
  await expect(plan).toContainText('Model preview ready');
  await expect(plan.locator('#model-root-plan-size')).toHaveText('120.6 KiB');
  await expect(plan.locator('#model-root-plan-items')).toHaveText('3');
  await expect(plan.locator('#model-root-plan-extra')).toHaveText('None');
  await expect(plan).not.toContainText('{');
  await expect(plan).not.toContainText('required_copy_bytes');

  page.once('dialog', dialog => {
    expect(dialog.message()).toContain('moved, not copied');
    return dialog.accept();
  });
  await page.locator('#model-root-relocation-execute').click();
  await expect(page.locator('#model-root-relocation-state')).toContainText('Models moved');
  const resolved = await notificationState(page);
  expect(resolved.active).toBe(false);
  expect(resolved.archived).toBe(true);
  const execute = calls.find(call => call.path.endsWith('/execute'));
  expect(JSON.parse(execute.body)).toMatchObject({
    plan_id: PLAN_ID, choice: 'move_into_foundry', confirmation: 'MOVE_MODELS_INTO_FOUNDRY',
  });
});
