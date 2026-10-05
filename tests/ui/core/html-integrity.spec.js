import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

// Duplicate ids make getElementById bind to the first match only, which silently
// disconnected the standalone database modal from its stats, buttons and log.
test('index.html has no duplicate element ids', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  const duplicates = await page.evaluate(() => {
    const seen = new Map();
    for (const el of document.querySelectorAll('[id]')) {
      seen.set(el.id, (seen.get(el.id) || 0) + 1);
    }
    return [...seen].filter(([, count]) => count > 1).map(([id]) => id);
  });
  expect(duplicates).toEqual([]);
});

test('database admin settings card and modal expose separate controls', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  for (const id of [
    'db-stat-size', 'db-stat-fts', 'db-btn-backup', 'db-btn-check', 'db-log-content',
    'settings-db-stat-size', 'settings-db-stat-fts', 'settings-db-btn-backup',
    'settings-db-btn-check', 'settings-db-log-content',
  ]) {
    await expect(page.locator(`[id="${id}"]`)).toHaveCount(1);
  }
});

// Backup and index names come from files on disk; they must render as text and
// their buttons must still call the existing global handlers.
test('database admin lists render hostile names as text and keep handlers', async ({ page }) => {
  const hostile = '<img src=x onerror="window.__pwned=1">"\'&.db';
  await page.route('**/api/db/backups', route => route.fulfill({
    json: { backups: [{ name: `manual/${hostile}`, size: 2048, modified: Date.now() / 1000, kind: 'manual' }], total_size: 2048 },
  }));
  await page.route('**/api/db/indexes', route => route.fulfill({
    json: { indexes: [{ name: hostile, table: hostile, rebuildable: true }] },
  }));
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
  await page.evaluate(() => {
    window.__rebuilt = null;
    window.rebuildIndex = name => { window.__rebuilt = name; };
    document.getElementById('db-btn-refresh').click();
  });
  await expect(page.locator('#db-backups-list .db-backup-name')).toHaveText(hostile);
  await expect(page.locator('#db-indexes-list .db-index-name')).toHaveText(hostile);
  expect(await page.locator('#db-backups-list img, #db-indexes-list img').count()).toBe(0);
  await page.locator('#db-indexes-list button').evaluate(b => b.click());
  expect(await page.evaluate(() => window.__rebuilt)).toBe(hostile);
  expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
});
