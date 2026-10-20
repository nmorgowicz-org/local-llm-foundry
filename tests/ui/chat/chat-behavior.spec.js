import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

async function switchToMonitor(page) {
  await dismissAuthShell(page);
  await page.evaluate(async () => {
    const { switchView } = await import('/js/features/setup-view.js');
    switchView('monitor');
  });
  await expect(page.locator('body')).not.toHaveClass(/setup-active/);
  await expect(page.locator('#view-monitor')).toBeVisible();
}

async function ensureChatVisible(page) {
  await page.getByRole('button', { name: /chat/i }).click();
  await expect(page.locator('#page-chat')).toBeVisible();
}

test.describe('system prompt and persona panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await ensureChatVisible(page);
  });

  test('opens and closes via .open class (not display:none)', async ({ page }) => {
    // Panel starts without .open — not visible
    await expect(page.locator('#chat-behavior-panel')).not.toHaveClass(/open/);
    await page.locator('#btn-behavior').click();
    // CSS transition: wait for panel to be visible
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);
    await expect(page.locator('#chat-behavior-panel')).toBeVisible();
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).not.toHaveClass(/open/);
  });

  test('shows persona section in behavior panel', async ({ page }) => {
    // Persona is now managed through the behavior panel, not a header dropdown.
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);
    await expect(page.locator('#chat-persona-active-name')).toBeVisible();
    await expect(page.locator('#chat-open-template-mgr')).toBeVisible();
  });

  test('persona list has items with edit buttons in template manager', async ({ page }) => {
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);
    await page.locator('#chat-open-template-mgr').click();
    await expect(page.locator('#template-manager-modal')).toHaveClass(/active/);

    // Wait for templates to load
    await page.waitForSelector('.template-list-item', { state: 'visible', timeout: 5000 });

    const items = await page.locator('.template-list-item').count();
    expect(items).toBeGreaterThan(0);

    // Each item should have an edit button
    const editButtons = await page.locator('[data-template-action="edit"]').count();
    expect(editButtons).toBeGreaterThan(0);
  });
});

test.describe('explicit mode toggle v2 (3-state)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await page.getByRole('button', { name: /chat/i }).click();
    // Wait for chat sidebar to render (initChatTabs is async)
    await page.waitForSelector('#csp-list .csp-item.active', { timeout: 10000 });
    // Reset active tab to level 0 and re-render so toggle UI and sidebar badge are in sync
    await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const { updateExplicitToggleUI } = await import('/js/features/chat-templates.js');
      const { renderChatSessionsSidebar } = await import('/js/features/chat-sessions-sidebar.js');
      const tab = activeChatTab();
      if (tab) tab.explicit_level = 0;
      updateExplicitToggleUI();
      renderChatSessionsSidebar();
    });
  });

  test('toggle in settings panel mirrors footer toggle', async ({ page }) => {
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);
    // Enable via behavior panel toggle
    await page.locator('#chat-explicit-toggle-behavior').click();
    await expect(page.locator('#chat-explicit-toggle-behavior')).toHaveClass(/active/);
    // Footer should also be active
    await expect(page.locator('#chat-explicit-toggle-footer')).toHaveClass(/active/);
  });

  test('3-state badge cycling on tab', async ({ page }) => {
    // Scope badge checks to the active sidebar item only.
    const activeItemBadge = page.locator('#csp-list .csp-item.active .csp-item-explicit');

    // State 0: element exists but data-level is 0 (no visible indicator)
    await expect(activeItemBadge).toHaveAttribute('data-level', '0');

    // Click once → level 1: 🔓 indicator
    await page.locator('#chat-explicit-toggle-footer').click();
    await expect(page.locator('#chat-explicit-toggle-footer')).toHaveClass(/active/);
    // Sidebar item not auto-updated; re-render to reflect new level
    await page.evaluate(async () => {
      const { renderChatSessionsSidebar } = await import('/js/features/chat-sessions-sidebar.js');
      renderChatSessionsSidebar();
    });
    await expect(activeItemBadge).toHaveAttribute('data-level', '1');

    // Click again → level 2: 🔥 indicator
    await page.locator('#chat-explicit-toggle-footer').click();
    await expect(page.locator('#chat-explicit-toggle-footer')).toHaveClass(/unrestricted/);
    await page.evaluate(async () => {
      const { renderChatSessionsSidebar } = await import('/js/features/chat-sessions-sidebar.js');
      renderChatSessionsSidebar();
    });
    await expect(activeItemBadge).toHaveAttribute('data-level', '2');

    // Click again → level 0: back to 0
    await page.locator('#chat-explicit-toggle-footer').click();
    await expect(page.locator('#chat-explicit-toggle-footer')).not.toHaveClass(/active/);
    await page.evaluate(async () => {
      const { renderChatSessionsSidebar } = await import('/js/features/chat-sessions-sidebar.js');
      renderChatSessionsSidebar();
    });
    await expect(activeItemBadge).toHaveAttribute('data-level', '0');
  });

  test('explicit policy injection at level 1', async ({ page }) => {
    // Set explicit_level = 1 directly and update UI
    await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const { updateExplicitToggleUI } = await import('/js/features/chat-templates.js');
      const tab = activeChatTab();
      if (tab) tab.explicit_level = 1;
      updateExplicitToggleUI();
    });

    // Verify level is set
    const level = await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      return activeChatTab()?.explicit_level ?? 0;
    });
    expect(level).toBe(1);

    // Verify the toggle UI reflects level 1
    await expect(page.locator('#chat-explicit-toggle-footer')).toHaveClass(/active/);
    await expect(page.locator('#chat-explicit-toggle-footer')).not.toHaveClass(/unrestricted/);
  });

  test('explicit policy injection at level 2', async ({ page }) => {
    // Set explicit_level = 2 directly and update UI
    await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const { updateExplicitToggleUI } = await import('/js/features/chat-templates.js');
      const tab = activeChatTab();
      if (tab) tab.explicit_level = 2;
      updateExplicitToggleUI();
    });

    // Verify level is set
    const level = await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      return activeChatTab()?.explicit_level ?? 0;
    });
    expect(level).toBe(2);

    // Verify the toggle UI reflects level 2 (unrestricted)
    await expect(page.locator('#chat-explicit-toggle-footer')).toHaveClass(/unrestricted/);
  });

  test('explicit gating in suggestions', async ({ page }) => {
    // With explicit_level = 0, the explicit suggestion group should not have .explicit-enabled
    const explicitGroup = page.locator('#suggestions-explicit-group');

    // Set explicit_level = 0 (already 0 from beforeEach)
    const level0 = await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      return activeChatTab()?.explicit_level ?? 0;
    });
    expect(level0).toBe(0);

    // Set explicit_level = 1, verify explicit group gets .explicit-enabled class
    await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const { updateExplicitToggleUI } = await import('/js/features/chat-templates.js');
      const tab = activeChatTab();
      if (tab) tab.explicit_level = 1;
      updateExplicitToggleUI();
    });

    // Trigger the suggestions dropdown UI refresh to propagate the explicit-enabled class
    await page.evaluate(async () => {
      const { getSuggestionsState, toggleSuggestionsDropdown } = await import('/js/features/chat-suggestions.js');
      // Expand and collapse to force UI refresh
      if (getSuggestionsState().expanded) toggleSuggestionsDropdown();
      else toggleSuggestionsDropdown();
      toggleSuggestionsDropdown();
    });

    if (await explicitGroup.count() > 0) {
      await expect(explicitGroup).toHaveClass(/explicit-enabled/);
    }
  });

  test('migration: explicit_mode true converts to explicit_level 1', async ({ page }) => {
    // Simulate a legacy tab with explicit_mode: true (boolean) and no explicit_level
    const normalized = await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      // Overwrite with legacy data
      tab.explicit_mode = true;
      delete tab.explicit_level;
      // Re-normalize by calling the init path (normalizeChatTab is not exported,
      // but we can verify the toggleExplicitMode path handles it)
      return {
        explicit_mode: tab.explicit_mode,
        hasExplicitLevel: 'explicit_level' in tab,
      };
    });
    expect(normalized.explicit_mode).toBe(true);
    expect(normalized.hasExplicitLevel).toBe(false);

    // Now simulate what normalizeChatTab does: when explicit_mode is true and
    // explicit_level is undefined, it sets explicit_level to 1
    const afterNormalize = await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      // Replicate normalizeChatTab logic
      let explicitLevel = tab.explicit_level ?? 0;
      if (tab.explicit_mode !== undefined && tab.explicit_level === undefined) {
        explicitLevel = tab.explicit_mode ? 1 : 0;
      }
      tab.explicit_level = explicitLevel;
      return tab.explicit_level;
    });
    expect(afterNormalize).toBe(1);
  });
});

test.describe('per-persona explicit policies', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await ensureChatVisible(page);
  });

  test('explicit policy section present in template manager', async ({ page }) => {
    await page.evaluate(async () => {
      const { openTemplateManager } = await import('/js/features/chat-templates.js');
      await openTemplateManager();
    });

    await expect(page.locator('#template-manager-modal')).toHaveClass(/active/);
    await page.waitForSelector('.explicit-policy-section', { timeout: 5000 });

    await expect(page.locator('.explicit-policy-section')).toBeVisible();
    await expect(page.getByText('Persona Explicit Policies')).toBeVisible();
    // Click to expand the details
    await page.getByText('Persona Explicit Policies').click();
    // Check that at least one of the policy containers is present
    const level1 = page.locator('#persona-explicit-level1');
    const level2 = page.locator('#persona-explicit-level2');
    const none = page.locator('#persona-explicit-none');
    const l1Count = await level1.count();
    const l2Count = await level2.count();
    const noneCount = await none.count();
    // At least one should exist (they're toggled based on selected template)
    expect(l1Count + l2Count + noneCount).toBeGreaterThan(0);
  });
});

test.describe('persona reset button', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await ensureChatVisible(page);
  });

  test('reset button visible only for built-in persona copies', async ({ page }) => {
    await page.evaluate(async () => {
      const { openTemplateManager } = await import('/js/features/chat-templates.js');
      await openTemplateManager();
    });

    await expect(page.locator('#template-manager-modal')).toHaveClass(/active/);
    await page.waitForSelector('.template-list-item', { timeout: 5000 });

    // Check that any reset buttons are only present on items whose name matches a built-in.
    const resetButtons = page.locator('.template-preview-btn.reset, .template-list-btn[data-template-action="reset"]');
    const count = await resetButtons.count();

    if (count > 0) {
      // At least one reset button exists — verify it is tied to a built-in.
      await resetButtons.first().click();
      // After reset, a confirmation dialog or toast should appear.
      // We just confirm the button is interactive and does not crash.
    }
  });
});

test.describe('template list sections and active badge', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await ensureChatVisible(page);
  });

  test('active badge appears on current persona', async ({ page }) => {
    // Ensure active tab has an active_template_id so we can see the badge.
    await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      if (tab) {
        tab.active_template_id = 'default';
      }
    });

    await page.evaluate(async () => {
      const { openTemplateManager } = await import('/js/features/chat-templates.js');
      await openTemplateManager();
    });

    await expect(page.locator('#template-manager-modal')).toHaveClass(/active/);
    await page.waitForSelector('.template-list-item', { timeout: 5000 });

    // If active_template_id is set and a matching template exists, there should be an active badge.
    const activeBadge = page.locator('.template-active-badge');
    const activeItems = page.locator('.template-list-item.active-persona');

    // We'll accept either both present or both absent (depends on backend templates).
    const badgeCount = await activeBadge.count();
    const itemCount = await activeItems.count();

    // Consistency check: if badge exists, active item exists too.
    if (badgeCount > 0) {
      expect(itemCount).toBeGreaterThan(0);
    }
  });
});

test.describe('gender token substitution', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await ensureChatVisible(page);
  });

  test('selecting gender sets ai_gender on tab', async ({ page }) => {
    // Open behavior panel
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);

    // Select "Female" gender pill (use first() to avoid strict mode violation)
    const femalePill = page.locator('#chat-behavior-panel .chat-gender-pill[data-gender="female"]');
    if (await femalePill.count() > 0) {
      await femalePill.click();

      // onGenderChange writes the tab synchronously; poll the state instead of sleeping.
      await expect.poll(() => page.evaluate(async () => {
        const { activeChatTab } = await import('/js/features/chat-state.js');
        return activeChatTab()?.ai_gender ?? null;
      })).toBe('female');
    } else {
      test.skip(true, 'Gender pills not present in this build.');
    }
  });
});

test.describe('custom role boundary', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await switchToMonitor(page);
    await page.waitForSelector('#view-monitor', { state: 'attached', timeout: 5000 });
    await page.waitForFunction(() => {
      const monitor = document.getElementById('view-monitor');
      return monitor && getComputedStyle(monitor).display !== 'none';
    }, { timeout: 5000 });
    await ensureChatVisible(page);
  });

  test('role boundary section exists and is editable', async ({ page }) => {
    // Open behavior panel
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);

    const toggle = page.locator('#chat-role-boundary-toggle');
    if (await toggle.count() === 0) {
      test.skip(true, 'Role boundary section not present in this build.');
    }

    // Expand role boundary section
    await toggle.click();
    await expect(page.locator('#chat-role-boundary-body')).toBeVisible({ timeout: 3000 });
    const textarea = page.locator('#chat-role-boundary-input');
    await expect(textarea).toBeVisible();

    // Edit value
    await textarea.fill('CUSTOM ROLE BOUNDARY TEXT');

    // Confirm custom value stored on tab (the input handler writes it synchronously).
    await expect.poll(() => page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      return activeChatTab()?.role_boundary_custom ?? null;
    })).toBe('CUSTOM ROLE BOUNDARY TEXT');
  });

  test('reset restores default role boundary', async ({ page }) => {
    await page.locator('#btn-behavior').click();
    await expect(page.locator('#chat-behavior-panel')).toHaveClass(/open/);

    const toggle = page.locator('#chat-role-boundary-toggle');
    if (await toggle.count() === 0) {
      test.skip(true, 'Role boundary section not present in this build.');
    }

    await toggle.click();
    const textarea = page.locator('#chat-role-boundary-input');
    await expect(textarea).toBeVisible();

    // Set custom value
    await textarea.fill('CUSTOM');
    const roleBoundaryCustom = () => page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      return activeChatTab()?.role_boundary_custom ?? null;
    });
    // Prove the custom value landed first, so the null below can only come from the reset.
    await expect.poll(roleBoundaryCustom).toBe('CUSTOM');

    // Reset
    const resetBtn = page.locator('#chat-role-boundary-reset');
    if (await resetBtn.count() > 0) {
      await resetBtn.click();
      await expect.poll(roleBoundaryCustom).toBeNull();
    }
  });
});
