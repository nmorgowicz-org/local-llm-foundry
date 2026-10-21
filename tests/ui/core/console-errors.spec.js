// ── Console Error Detection ───────────────────────────────────────────────────
// Catch JavaScript errors in the browser console before they reach users.

import { test, expect } from '@playwright/test';

test.describe('console error detection', () => {
  let consoleErrors = [];

  test.beforeEach(async ({ page }) => {
    consoleErrors = [];

    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();

      if (type === 'error' || type === 'warning') {
        consoleErrors.push({ type, text, url: msg.location() });
      }
    });

    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
  });

  test('no fatal JavaScript errors', async () => {
    const fatalErrors = consoleErrors.filter(e => {
      const ignored = [
        'Unsupported site:',
        'Fetch API',
        'content-security-policy',
        // Known CDN quirk: jsDelivr returns text/plain for some SRI scripts.
        'Refused to execute',
      ];

      return !ignored.some(pattern => e.text.includes(pattern)) &&
             e.type === 'error' &&
             !e.text.includes('Failed to load resource');
    });

    if (fatalErrors.length > 0) {
      console.log('=== Console Errors Detected ===');
      fatalErrors.forEach(e => {
        console.log(`[${e.type}] ${e.text}`);
        if (e.url) {
          console.log(`  at ${e.url.url || 'unknown'}`);
        }
      });
    }

    expect(fatalErrors.length).toBe(0);
  });
});
