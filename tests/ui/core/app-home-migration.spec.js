import { test, expect } from '@playwright/test';
import { dismissAuthShell, openSettings } from '../helpers.js';
import { createMigrationMock, migrationPlan, MIGRATION_ROUTE } from '../fixtures/app-home-migration.mjs';

const card = '#app-home-migration-card';
const preview = '#app-home-migration-preview';
const queue = '#app-home-migration-queue';
const planView = '#app-home-migration-plan';

async function boot(page, options = {}) {
  const state = createMigrationMock(options);
  // Install before boot: no status/preview/queue/rollback/cleanup request can
  // reach the live server, regardless of method. All unrelated APIs stay real.
  await page.route(MIGRATION_ROUTE, async route => {
    const request = route.request();
    const response = state.respond(new URL(request.url()).pathname, request.method(), request.postData());
    await route.fulfill({ status: response.status, contentType: 'application/json', body: JSON.stringify(response.body) });
  });
  await page.addInitScript(() => sessionStorage.setItem('local-llm-foundry-migration-toast-seen', '1'));
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await openSettings(page, 'migration');
  await expect(page.locator('#app-home-migration-state')).toContainText(
    options.queued ? 'Migration is queued' : 'Legacy application data detected',
  );
  return state;
}

async function showPreview(page) {
  await page.locator(preview).click();
  await expect(page.locator(queue)).toBeEnabled();
  await expect(page.locator(planView)).toBeVisible();
}

function acceptQueueConfirmation(page) {
  page.once('dialog', dialog => dialog.type() === 'confirm' ? dialog.accept() : dialog.dismiss());
}

test.describe('application-home migration preview', () => {
  test.use({ locale: 'en-US' });

  for (const [bytes, formatted] of [[0, '0 B'], [1536, '1.5 KiB']]) {
    test(`formats ${bytes} copy bytes as ${formatted}`, async ({ page }) => {
      const plan = migrationPlan();
      plan.required_copy_bytes = bytes;
      const state = await boot(page, { plan });
      await showPreview(page);
      await expect(page.locator('#app-home-migration-copy-size')).toHaveText(formatted);
      expect(state.unexpected).toEqual([]);
    });
  }

  test('distinguishes total inventory from copy count and formats actual copy bytes', async ({ page }) => {
    const state = await boot(page);
    await expect(page.locator(queue)).toBeDisabled();
    await showPreview(page);
    const view = page.locator(planView);
    await expect(page.locator('#app-home-migration-copy-size')).toHaveText('771 MiB');
    await expect(page.locator('#app-home-migration-inventoried-count')).toHaveText('8');
    await expect(page.locator('#app-home-migration-retained-count')).toHaveText('4');
    await expect(view).toContainText('Total inventoried items');
    await expect(view).toContainText('Retained items');
    await expect(view).toContainText(/next (?:launch|restart)/i);
    await expect(view).toContainText(/critical application state/i);
    await expect(view).toContainText(/rollback/i);
    await expect(view).toContainText(/models/i);
    await expect(view).toContainText(/generated runtimes are not copied/i);
    const technical = page.locator(`${planView} details`).filter({ has: page.locator('summary', { hasText: /technical/i }) });
    await technical.locator('summary').click();
    await expect(technical.locator('pre')).toContainText('"inventoried_entries": 8');
    await expect(technical.locator('pre')).toContainText('"retained_entries": 4');
    await expect(technical.locator('pre')).not.toContainText('copied_entries');
    // The fixture makes a mislabeled actual-copy count visibly wrong: only
    // four of the eight inventory entries are copied (critical/unknown).
    expect(state.unexpected).toEqual([]);
  });

  test('keeps native location and technical details collapsed below the human summary', async ({ page }) => {
    const state = await boot(page);
    await showPreview(page);
    const details = page.locator(`${card} details`);
    await expect(details).toHaveCount(2);
    for (const detail of await details.all()) {
      await expect(detail).not.toHaveAttribute('open', '');
      await expect(detail.locator('summary')).toBeVisible();
      await detail.locator('summary').click();
      await expect(detail).toHaveAttribute('open', '');
      await detail.locator('summary').click();
      await expect(detail).not.toHaveAttribute('open', '');
    }
    expect(await page.locator(card).evaluate(el => {
      const facts = el.querySelector('.app-home-migration-facts');
      return Boolean(facts && [...el.querySelectorAll('details')].every(detail =>
        facts.compareDocumentPosition(detail) & Node.DOCUMENT_POSITION_FOLLOWING));
    })).toBe(true);
    const locations = details.filter({ has: page.locator('summary', { hasText: /location/i }) });
    await locations.locator('summary').click();
    await expect(locations).toContainText(state.plan.source);
    await expect(locations).toContainText(state.plan.destination);
    expect(state.unexpected).toEqual([]);
  });

  test('renders API plan strings literally without creating executable markup', async ({ page }) => {
    const plan = migrationPlan();
    const hostile = '<img src=x onerror="window.__migrationXss=1"><script>window.__migrationXss=2</script>';
    plan.source += hostile;
    plan.destination += hostile;
    plan.plan_id += hostile;
    const state = await boot(page, { plan });
    await showPreview(page);
    for (const summary of await page.locator(`${card} details > summary`).all()) await summary.click();
    await expect(page.locator('#app-home-migration-source')).toHaveText(plan.source);
    await expect(page.locator('#app-home-migration-destination')).toHaveText(plan.destination);
    await expect(page.locator('#app-home-migration-plan-id')).toHaveText(plan.plan_id);
    await expect(page.locator(`${card} img, ${card} script, ${card} [onerror]`)).toHaveCount(0);
    expect(await page.evaluate(() => window.__migrationXss)).toBeUndefined();
    expect(state.unexpected).toEqual([]);
  });

  test('failed first preview never enables queue', async ({ page }) => {
    const state = await boot(page);
    state.previewError = 'Fixture preview unavailable';
    await page.locator(preview).click();
    await expect(page.locator('.toast').filter({ hasText: 'Fixture preview unavailable' })).toBeVisible();
    await expect(page.locator(preview)).toBeEnabled();
    await expect(page.locator(queue)).toBeDisabled();
    expect(state.calls.filter(call => call.pathname.endsWith('/queue'))).toHaveLength(0);
    expect(state.unexpected).toEqual([]);
  });

  test('failed refresh invalidates an earlier preview and blocks stale queue submission', async ({ page }) => {
    const state = await boot(page);
    await showPreview(page);
    state.previewError = 'Fixture refreshed preview failed';
    await page.locator(preview).click();
    await expect(page.locator('.toast').filter({ hasText: 'Fixture refreshed preview failed' })).toBeVisible();
    await expect(page.locator(queue)).toBeDisabled();
    await expect(page.locator(planView)).toBeHidden();
    // Dispatch bypasses the native disabled-button click suppression, proving
    // the stale-plan guard rather than only the button's visual state.
    let dialogs = 0;
    const handler = async dialog => { dialogs += 1; await dialog.dismiss(); };
    page.on('dialog', handler);
    try {
      await page.locator(queue).dispatchEvent('click');
      expect(dialogs).toBe(0);
    } finally {
      page.off('dialog', handler);
    }
    expect(state.calls.filter(call => call.pathname.endsWith('/queue'))).toHaveLength(0);
    expect(state.unexpected).toEqual([]);
  });

  test('queue failure is visible and permits retry without claiming success', async ({ page }) => {
    const state = await boot(page);
    await showPreview(page);
    state.queueError = 'Fixture queue rejected';
    acceptQueueConfirmation(page);
    await page.locator(queue).click();
    await expect(page.locator('.toast').filter({ hasText: 'Fixture queue rejected' })).toBeVisible();
    await expect(page.locator(queue)).toBeEnabled();
    await expect(page.locator('#app-home-migration-state')).not.toContainText('Migration is queued');
    expect(state.queued).toBe(false);
    state.queueError = null;
    acceptQueueConfirmation(page);
    await page.locator(queue).click();
    await expect(page.locator('#app-home-migration-state')).toContainText('Migration is queued');
    await expect(page.locator(queue)).toBeDisabled();
    await expect(page.locator(preview)).toBeDisabled();
    expect(state.calls.filter(call => call.pathname.endsWith('/queue'))).toHaveLength(2);
    expect(state.unexpected).toEqual([]);
  });

  test('successful queue is terminal even if disabled controls receive synthetic clicks', async ({ page }) => {
    const state = await boot(page);
    await showPreview(page);
    acceptQueueConfirmation(page);
    await page.locator(queue).click();
    await expect(page.locator('#app-home-migration-state')).toContainText('Migration is queued');
    await expect(page.locator('#app-home-migration-state')).toContainText('Restart Foundry');
    await expect(page.locator(preview)).toBeDisabled();
    await expect(page.locator(queue)).toBeDisabled();
    const callsBefore = state.calls.length;
    let dialogs = 0;
    const handler = async dialog => { dialogs += 1; await dialog.dismiss(); };
    page.on('dialog', handler);
    try {
      // A bounded negative network observation, not a render-settling sleep.
      // dispatchEvent deliberately bypasses disabled-button click suppression.
      const furtherRequest = page.waitForRequest(
        request => MIGRATION_ROUTE.test(request.url()), { timeout: 500 },
      ).then(() => true, error => {
        if (error.name === 'TimeoutError') return false;
        throw error;
      });
      await page.locator(preview).dispatchEvent('click');
      await page.locator(queue).dispatchEvent('click');
      expect(await furtherRequest).toBe(false);
      expect(dialogs).toBe(0);
      expect(state.calls).toHaveLength(callsBefore);
      expect(state.unexpected).toEqual([]);
    } finally {
      page.off('dialog', handler);
    }
  });

  for (const migrationRequired of [true, false]) {
    test(`already queued status disables controls (migration_required=${migrationRequired})`, async ({ page }) => {
      const state = await boot(page, { queued: true, migrationRequired });
      await expect(page.locator('#app-home-migration-state')).toContainText(/Restart Foundry/);
      await expect(page.locator('#app-home-migration-summary')).toContainText('Nothing has moved yet');
      await expect(page.locator('#app-home-migration-summary')).toContainText(/models and generated runtimes are not copied/i);
      await expect(page.locator(preview)).toBeDisabled();
      await expect(page.locator(queue)).toBeDisabled();
      expect(state.calls.map(call => call.pathname)).toEqual(['/api/app-home-migration/status']);
      expect(state.unexpected).toEqual([]);
    });
  }
});
