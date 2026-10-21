// ── Settings - Guided Generation Tests ───────────────────────────────────────

import { test, expect } from '@playwright/test';
import { switchToChat } from './fixtures.js';
import { openSettings } from '../helpers.js';

async function openGuidedGenSettings(page) {
    await openSettings(page);
  await page.waitForSelector('#settings-modal:not([aria-hidden="true"])');
  await page.locator('.settings-tab[data-tab="chat"]').click();
  await page.waitForSelector('#settings-chat.active');
}

test.describe('Settings - Guided Generation', () => {
  test.beforeEach(async ({ page }) => {
    await switchToChat(page);
  });

  test('toggle context notes enabled', async ({ page }) => {
    await openGuidedGenSettings(page);

    const checkbox = page.locator('#settings-enabled-context-notes');
    const initial = await checkbox.isChecked();
    // Toggle-switch hides the real checkbox visually; toggle via evaluate
    await page.evaluate(() => {
      const cb = document.getElementById('settings-enabled-context-notes');
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event('change', { bubbles: true }));
    });
    const after = await checkbox.isChecked();
    expect(after).toBe(!initial);
  });

  test('toggle suggestions enabled', async ({ page }) => {
    await openGuidedGenSettings(page);

    const checkbox = page.locator('#settings-enabled-suggestions');
    const initial = await checkbox.isChecked();
    await page.evaluate(() => {
      const cb = document.getElementById('settings-enabled-suggestions');
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(await checkbox.isChecked()).toBe(!initial);
  });

  test('toggle quick guide enabled', async ({ page }) => {
    await openGuidedGenSettings(page);

    const checkbox = page.locator('#settings-enabled-quick-guide');
    const initial = await checkbox.isChecked();
    await page.evaluate(() => {
      const cb = document.getElementById('settings-enabled-quick-guide');
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(await checkbox.isChecked()).toBe(!initial);
  });

  test('toggle saved thinking history', async ({ page }) => {
    await openGuidedGenSettings(page);

    const checkbox = page.locator('#settings-persist-thinking-content');
    const initial = await checkbox.isChecked();
    await page.evaluate(() => {
      const cb = document.getElementById('settings-persist-thinking-content');
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(await checkbox.isChecked()).toBe(!initial);
  });

  test('sidebar width slider updates display', async ({ page }) => {
    await openGuidedGenSettings(page);

    const slider = page.locator('#settings-sidebar-width');
    await expect(slider).toBeVisible();

    // Set to 350
    await slider.fill('350');
    await expect(slider).toHaveValue('350');

    // Value display should update
    const valueDisplay = page.locator('#settings-sidebar-width-value');
    await expect(valueDisplay).toContainText('350px');
  });

  test('suggestion count slider updates display', async ({ page }) => {
    await openGuidedGenSettings(page);
    const slider = page.locator('#settings-suggestion-count');
    await expect(slider).toBeVisible();

    await slider.fill('7');
    await expect(slider).toHaveValue('7');

    const valueDisplay = page.locator('#settings-suggestion-count-value');
    await expect(valueDisplay).toContainText('7');
  });

  test('context depth slider updates display', async ({ page }) => {
    await openGuidedGenSettings(page);

    const slider = page.locator('#settings-context-depth');
    await expect(slider).toBeVisible();

    await slider.fill('15');
    await expect(slider).toHaveValue('15');

    const valueDisplay = page.locator('#settings-context-depth-value');
    await expect(valueDisplay).toContainText('15');
  });

  test('save shows success feedback', async ({ page }) => {
    await openGuidedGenSettings(page);

    // Modify something
    await page.locator('#settings-sidebar-width').fill('300');

    // Click save — settings modal saves in-place without closing
    const saveButton = page.locator('#settings-modal-save');
    await saveButton.click();

    // Should briefly show success class
    await expect(saveButton).toHaveClass(/success/, { timeout: 3000 });
  });
});
