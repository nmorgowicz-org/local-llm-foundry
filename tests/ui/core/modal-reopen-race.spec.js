import { test, expect } from '@playwright/test';
import { openSettings } from '../helpers.js';

// Closing a modal starts an animation-teardown timer that strips the visibility class and,
// for settings, sets aria-hidden/inert. Nothing cancelled that timer on reopen, so a
// close->open inside the animation window left the *previous* close's timer to fire against
// the now-legitimately-open modal: it reported `open` in the DOM while rendering nothing,
// then went inert. Any test that reopened a modal quickly could land on either side of that
// 260ms window, which is what made the full suite fail 2-3 tests at random.
//
// These tests reopen deliberately fast and then wait past the old teardown window.
// Resolves once the page's own 260ms teardown timer has had its chance to fire. The delay is
// scheduled in the page, after the close, so timer ordering guarantees the (earlier-deadline)
// teardown timer has run before this resolves, whatever the load on the machine. A Node-side
// waitForTimeout only guarantees wall-clock time. This is an intentional negative wait: the
// assertion is that the stale timer does NOT change the modal, so there is nothing to poll for.
const pastTeardownWindow = (page) =>
  page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 300)));

test.describe('modal reopen does not race its own close animation', () => {
  test('@in-memory-test settings survives close then immediate reopen', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Open through the router first, so the modal is in the state the app really produces.
    // The close/open dance then has to be direct: the race under test is synchronous, and a
    // Router.navigate round-trip is far too slow to land inside the 260ms animation window.
    await openSettings(page);
    await page.evaluate(async () => {
      const mod = await import('/js/features/settings.js');
      mod.closeSettingsModal();
      mod.openSettingsModal();
    });

    const modal = page.locator('#settings-modal');
    // Past the 260ms teardown: a stale timer would have fired by now.
    await pastTeardownWindow(page);
    await expect(modal).toHaveClass(/\bopen\b/);
    await expect(modal).not.toHaveClass(/\bclosing\b/);
    await expect(modal).toBeVisible();
    // The half-torn-down state was the confusing part: `open` present, but inert and
    // aria-hidden, so it rendered nothing and swallowed input.
    expect(await modal.getAttribute('aria-hidden'), 'reopened modal must not be aria-hidden')
      .toBeNull();
    expect(await modal.evaluate((el) => el.inert), 'reopened modal must not be inert')
      .toBe(false);
  });

  test('@in-memory-test repeated close does not stack teardown timers', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Esc twice, or a route change racing the X button: the second close used to queue a
    // second timer that fired later against whatever state the modal had reached by then.
    await openSettings(page);
    await page.evaluate(async () => {
      const mod = await import('/js/features/settings.js');
      mod.closeSettingsModal();
      mod.closeSettingsModal();
    });
    await pastTeardownWindow(page);
    await expect(page.locator('#settings-modal')).not.toHaveClass(/\bopen\b/);

    await page.evaluate(async () => {
      const mod = await import('/js/features/settings.js');
      mod.openSettingsModal();
    });
    await pastTeardownWindow(page);
    await expect(page.locator('#settings-modal')).toHaveClass(/\bopen\b/);
    await expect(page.locator('#settings-modal')).toBeVisible();
  });
});
