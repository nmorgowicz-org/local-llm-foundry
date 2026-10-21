// ── Quick Guide Tests ────────────────────────────────────────────────────────

import { test, expect } from '@playwright/test';
import { switchToChat, enableGuidedGenFeatures, seedChatMessages } from './fixtures.js';

test.describe('Quick Guide', () => {
  test.beforeEach(async ({ page }) => {
    await switchToChat(page);
    await enableGuidedGenFeatures(page, { enabled_quick_guide: true });
  });

  test('toggle inline input', async ({ page }) => {
    const toggle = page.locator('#quick-guide-toggle');
    const container = page.locator('#quick-guide-container');

    // Initially collapsed
    await expect(container).not.toHaveClass(/quick-guide-expanded/);
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');

    // Toggle open
    await toggle.click();
    await expect(container).toHaveClass(/quick-guide-expanded/);
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');

    // Should show textarea
    await expect(page.locator('#quick-guide-input')).toBeVisible();

    // Toggle closed
    await toggle.click();
    await expect(container).not.toHaveClass(/quick-guide-expanded/);
  });

  test('submit instruction with Enter collapses and clears input', async ({ page }) => {
    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });

    await page.locator('#quick-guide-input').fill('Make the tone more mysterious');
    await page.locator('#quick-guide-input').press('Enter');

    // Guide should collapse synchronously
    await expect(page.locator('#quick-guide-container')).not.toHaveClass(/quick-guide-expanded/);

    // Input should be cleared
    await expect(page.locator('#quick-guide-input')).toHaveValue('');
  });

  test('submit instruction with button collapses guide', async ({ page }) => {
    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });

    await page.locator('#quick-guide-input').fill('Add more dialogue');
    await page.locator('#quick-guide-submit-btn').click();

    // Guide should collapse
    await expect(page.locator('#quick-guide-container')).not.toHaveClass(/quick-guide-expanded/);
    // Input should be cleared
    await expect(page.locator('#quick-guide-input')).toHaveValue('');
  });

  test('last used instruction displayed', async ({ page }) => {
    // Mock the chat endpoint so the submit completes immediately
    await page.route('/api/chat', route => {
      const sse = 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n';
      route.fulfill({ status: 200, contentType: 'text/event-stream', body: sse });
    });

    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });

    await page.locator('#quick-guide-input').fill('Show, don\'t tell');
    await page.locator('#quick-guide-submit-btn').click();

    // Wait for the guide to close (the container loses the expanded class)
    await page.waitForSelector('#quick-guide-container', { state: 'visible', timeout: 10000 });
    await page.waitForFunction(() => {
      return !document.getElementById('quick-guide-container')?.classList.contains('quick-guide-expanded');
    }, { timeout: 10000 });

    // Toggle again to see last used
    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });
    // Wait for the last used instruction to be displayed
    await page.waitForFunction(() => {
      const lastUsed = document.getElementById('quick-guide-last-used');
      return lastUsed && lastUsed.style.display !== 'none';
    }, { timeout: 5000 });
    await expect(page.locator('.quick-guide-last-used')).toContainText('Show, don\'t tell');
  });

  test('empty submit clears draft', async ({ page }) => {
    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });

    // Fill then clear the draft
    await page.locator('#quick-guide-input').fill('Test instruction');
    await page.locator('#quick-guide-input').fill('');

    // Input should be empty
    await expect(page.locator('#quick-guide-input')).toHaveValue('');

    // Status should show Idle when draft is empty and guide is open
    await expect(page.locator('#quick-guide-status')).toContainText('Idle');
  });

  test('Escape closes without submit', async ({ page }) => {
    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });

    await page.locator('#quick-guide-input').fill('Test instruction');
    await page.keyboard.press('Escape');

    // Should collapse
    await expect(page.locator('#quick-guide-container')).not.toHaveClass(/quick-guide-expanded/);
  });

  test('click outside closes', async ({ page }) => {
    await page.locator('#quick-guide-toggle').click();
    await page.waitForSelector('#quick-guide-input', { state: 'visible' });

    // Click chat input
    await page.locator('#chat-input').click();

    // Should collapse
    await expect(page.locator('#quick-guide-container')).not.toHaveClass(/quick-guide-expanded/);
  });

});
