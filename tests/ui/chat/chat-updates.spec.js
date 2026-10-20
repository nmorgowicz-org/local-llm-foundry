import { test, expect } from '@playwright/test';

test.describe('app update UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
  });

  test('update pill is present but hidden by default', async ({ page }) => {
    await expect(page.locator('#update-pill')).toBeAttached();
    await expect(page.locator('#update-pill')).not.toBeVisible();
  });

  test('app version is displayed in sidebar', async ({ page }) => {
    await expect(page.locator('#app-version')).toBeAttached();
    const version = await page.locator('#app-version').textContent();
    // Should be non-empty (e.g. "v0.10.2")
    expect(version?.trim().length).toBeGreaterThan(0);
  });

  test('release notes panel is present but off-screen', async ({ page }) => {
    await expect(page.locator('#release-notes-panel')).toBeAttached();
    await expect(page.locator('#release-notes-panel')).not.toHaveClass(/open/);
  });

  test('opening release notes panel shows version diff', async ({ page }) => {
    await page.evaluate(() => {
      localStorage.removeItem('update-dismissed');
      const pill = document.getElementById('update-pill');
      const text = document.getElementById('update-pill-text');
      if (pill && text) {
        text.textContent = 'v99.0.0 available';
        pill.style.display = 'flex';
      }
    });
    await page.locator('#update-pill').click();
    await expect(page.locator('#release-notes-panel')).toBeVisible();
  });
});
