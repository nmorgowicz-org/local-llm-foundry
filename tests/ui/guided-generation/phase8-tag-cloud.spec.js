// ── Phase 8 Tag Cloud Tests ──────────────────────────────────────────────────

import { test, expect } from '@playwright/test';
import { switchToChat, enableGuidedGenFeatures, mockSuggestionsAPI } from './fixtures.js';

// Count chips not hidden by the search filter, independently of group visibility.
const countVisible = (chips) => chips.evaluateAll(
  (els) => els.filter(el => el.style.display !== 'none').length,
);

test.describe('Phase 8 - Tag Cloud', () => {
  test.beforeEach(async ({ page }) => {
    await switchToChat(page);
    await enableGuidedGenFeatures(page, { enabled_suggestions: true });
    await mockSuggestionsAPI(page);
  });

  test('tag cloud renders all 15 category chips in DOM', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    // 15 chips total in DOM (explicit is hidden until explicit mode is on)
    const chips = page.locator('.suggestion-category-btn');
    await expect(chips).toHaveCount(15);

    // The 14 standard chips should be visible
    const standardKeys = [
      'general', 'plot-twist', 'new-character', 'director',
      'action', 'comedy', 'fantasy', 'horror', 'mystery', 'noir',
      'romance', 'sci-fi', 'thriller', 'character',
    ];
    for (const key of standardKeys) {
      await expect(page.locator(`.suggestion-category-btn[data-category="${key}"]`)).toBeVisible();
    }

    // Explicit chip exists but is hidden until explicit mode enabled
    await expect(page.locator('.suggestion-category-btn[data-category="explicit"]')).toHaveCount(1);
  });

  test('story tools group contains 4 chips', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const storyToolsGroup = page.locator('.category-group[data-group="story-tools"]');
    const chips = storyToolsGroup.locator('.suggestion-category-btn');
    await expect(chips).toHaveCount(4);

    await expect(chips.filter({ hasText: /General/ })).toBeVisible();
    await expect(chips.filter({ hasText: /Plot Twist/ })).toBeVisible();
    await expect(chips.filter({ hasText: /New Character/ })).toBeVisible();
    await expect(chips.filter({ hasText: /Director/ })).toBeVisible();
  });

  test('genres group contains 10 chips', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const genresGroup = page.locator('.category-group[data-group="genres"]');
    const chips = genresGroup.locator('.suggestion-category-btn');
    await expect(chips).toHaveCount(10);

    const genreNames = ['Action', 'Comedy', 'Fantasy', 'Horror', 'Mystery', 'Noir', 'Romance', 'Sci-Fi', 'Thriller', 'Character'];
    for (const name of genreNames) {
      await expect(chips.filter({ hasText: new RegExp(name) })).toBeVisible();
    }
  });

  test('explicit group contains 1 chip when explicit mode enabled', async ({ page }) => {
    // Enable explicit mode so the group becomes visible
    await page.evaluate(() => {
      return import('/js/features/chat-state.js').then(({ activeChatTab }) => {
        const tab = activeChatTab();
        if (tab) tab.explicit_level = 1;
      });
    });

    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    // Re-trigger UI update so explicit group shows
    await page.evaluate(async () => {
      const { setSuggestionCategory } = await import('/js/features/chat-suggestions.js');
      setSuggestionCategory('general');
    });

    const explicitGroup = page.locator('.category-group[data-group="explicit"]');
    const chips = explicitGroup.locator('.suggestion-category-btn');
    await expect(chips).toHaveCount(1);
    await expect(page.locator('.suggestion-category-btn[data-category="explicit"]')).toBeVisible();
  });

  test('category groups are collapsible', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const header = page.locator('.category-group[data-group="story-tools"] .category-group-header');
    const chipsContainer = page.locator('.category-group[data-group="story-tools"] .category-group-chips');

    // Initially expanded
    await expect(header).toHaveAttribute('aria-expanded', 'true');
    await expect(chipsContainer).toBeVisible();

    // Collapse
    await header.click();
    await expect(header).toHaveAttribute('aria-expanded', 'false');

    // Expand again
    await header.click();
    await expect(header).toHaveAttribute('aria-expanded', 'true');
    await expect(chipsContainer).toBeVisible();
  });

  test('search filter narrows visible categories', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const searchInput = page.locator('#suggestion-search-input');

    // All chips pass the empty search filter, including the gated explicit chip.
    const allChips = page.locator('.suggestion-category-btn');
    await expect.poll(() => countVisible(allChips)).toBe(15);

    // Type "horror" — only Horror chip should remain visible
    await searchInput.fill('horror');
    await expect.poll(() => countVisible(allChips)).toBe(1);

    const visibleChip = allChips.filter({ hasText: /Horror/ });
    await expect(visibleChip).toBeVisible();

    // Genres group should still be visible (has Horror chip)
    const genresGroup = page.locator('.category-group[data-group="genres"]');
    await expect(genresGroup).toBeVisible();

    // Story Tools and Explicit groups should be hidden (no matching chips)
    const storyToolsGroup = page.locator('.category-group[data-group="story-tools"]');
    const storyToolsDisplay = await storyToolsGroup.evaluate(el => el.style.display);
    expect(storyToolsDisplay).toBe('none');

    const explicitGroup = page.locator('.category-group[data-group="explicit"]');
    const explicitDisplay = await explicitGroup.evaluate(el => el.style.display);
    expect(explicitDisplay).toBe('none');

    // Clear search — all chips pass the filter again.
    await searchInput.fill('');
    await expect.poll(() => countVisible(allChips)).toBe(15);
  });

  test('search filter is case-insensitive', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const searchInput = page.locator('#suggestion-search-input');
    const allChips = page.locator('.suggestion-category-btn');

    await searchInput.fill('HORROR');
    await expect.poll(() => countVisible(allChips)).toBe(1);
  });

  test('explicit group hidden when explicit_level is 0', async ({ page }) => {
    await page.evaluate(() => {
      import('/js/features/chat-state.js').then(({ activeChatTab }) => {
        const tab = activeChatTab();
        if (tab) tab.explicit_level = 0;
      });
    });

    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const explicitGroup = page.locator('#suggestions-explicit-group');
    await expect(explicitGroup).toBeHidden();
    await expect(explicitGroup).not.toHaveClass(/explicit-enabled/);
  });

  test('explicit group visible when explicit_level is 1', async ({ page }) => {
    await page.evaluate(() => {
      import('/js/features/chat-state.js').then(({ activeChatTab }) => {
        const tab = activeChatTab();
        if (tab) {
          tab.explicit_level = 1;
          window.dispatchEvent(new CustomEvent('explicitModeChanged'));
        }
      });
    });

    const explicitGroup = page.locator('#suggestions-explicit-group');
    await expect(explicitGroup).toBeVisible();
    await expect(explicitGroup).toHaveClass(/explicit-enabled/);
  });

  test('all 15 chips have correct data-category attributes', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const expected = [
      { text: 'General', category: 'general' },
      { text: 'Plot Twist', category: 'plot-twist' },
      { text: 'New Character', category: 'new-character' },
      { text: 'Director', category: 'director' },
      { text: 'Action', category: 'action' },
      { text: 'Comedy', category: 'comedy' },
      { text: 'Fantasy', category: 'fantasy' },
      { text: 'Horror', category: 'horror' },
      { text: 'Mystery', category: 'mystery' },
      { text: 'Noir', category: 'noir' },
      { text: 'Romance', category: 'romance' },
      { text: 'Sci-Fi', category: 'sci-fi' },
      { text: 'Thriller', category: 'thriller' },
      { text: 'Character', category: 'character' },
      { text: 'Explicit', category: 'explicit' },
    ];

    // Use data-category attribute for unambiguous lookup (avoids substring matching)
    for (const { category } of expected) {
      await expect(page.locator(`.suggestion-category-btn[data-category="${category}"]`))
        .toHaveAttribute('data-category', category);
    }
  });

  test('search filter partial match works', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const searchInput = page.locator('#suggestion-search-input');
    const allChips = page.locator('.suggestion-category-btn');

    await searchInput.fill('sci');
    await expect.poll(() => countVisible(allChips)).toBe(1);

    const visibleChip = allChips.filter({ hasText: /Sci-Fi/ });
    await expect(visibleChip).toBeVisible();
  });

  test('multiple group headers can be independently collapsed', async ({ page }) => {
    await page.locator('#suggestions-toggle').click();
    await page.waitForSelector('#suggestions-dropdown', { state: 'visible' });

    const storyToolsHeader = page.locator('.category-group[data-group="story-tools"] .category-group-header');
    const genresHeader = page.locator('.category-group[data-group="genres"] .category-group-header');
    const storyToolsChips = page.locator('.category-group[data-group="story-tools"] .category-group-chips');
    const genresChips = page.locator('.category-group[data-group="genres"] .category-group-chips');

    // Collapse Story Tools only
    await storyToolsHeader.click();
    await expect(storyToolsHeader).toHaveAttribute('aria-expanded', 'false');
    await expect(genresHeader).toHaveAttribute('aria-expanded', 'true');

    // Collapse Genres only
    await genresHeader.click();
    await expect(genresHeader).toHaveAttribute('aria-expanded', 'false');

    // Re-expand Story Tools
    await expect(storyToolsHeader).toHaveAttribute('data-tag-cloud-bound', '1');
    expect(await storyToolsHeader.evaluate(header => header.isConnected)).toBe(true);
    await storyToolsHeader.click();
    await expect(storyToolsHeader).toHaveAttribute('aria-expanded', 'true');
    await expect(genresHeader).toHaveAttribute('aria-expanded', 'false');
  });
});
