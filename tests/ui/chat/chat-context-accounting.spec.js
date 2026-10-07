import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

// The second prompt already contains the first 3,000-token assistant reply.
// Current occupancy is 6,000 + 1,500 = 7,500, not 6,000 + 4,500 = 10,500.
async function seedChat(page, { overflow = false } = {}) {
  await page.evaluate(async ({ overflow }) => {
    const state = await import('/js/core/app-state.js');
    const { newChatTab, normalizeTabForSave } = await import('/js/features/chat-state.js');
    const { renderChatMessages } = await import('/js/features/chat-render.js');
    const tab = newChatTab('Context regression');
    tab.auto_compact = false;
    tab.messages = [
      { role: 'user', content: 'First question', timestamp_ms: 1 },
      { role: 'assistant', content: 'First answer', input_tokens: 1000, output_tokens: 3000, timestamp_ms: 2 },
      { role: 'user', content: 'Second question', timestamp_ms: 3 },
      { role: 'assistant', content: 'Second answer', input_tokens: overflow ? 11000 : 6000, output_tokens: 1500, timestamp_ms: 4 },
    ];
    tab.total_input_tokens = overflow ? 12000 : 7000;
    tab.total_output_tokens = 4500;
    const headers = { ...window.authHeaders(), 'Content-Type': 'application/json' };
    const created = await fetch('/api/chat/tabs', { method: 'POST', headers, body: JSON.stringify(normalizeTabForSave(tab)) });
    const createdBody = await created.json();
    if (!created.ok || createdBody.error) throw new Error(`create fixture: ${createdBody.error || created.status}`);
    state.chat.tabs = [tab];
    state.chat.activeTabId = tab.id;
    state.setWsData({ active_session_id: 'context-test', mode: 'live' });
    state.setLastLlamaMetrics({ context_capacity_tokens: 10000, kv_cache_max: 10000 });
    state.setContextCapacityTokens(10000);
    renderChatMessages();
  }, { overflow });
}

async function mockReply(page, { editDuringRequest = false } = {}) {
  await page.route('**/api/chat', async route => {
    if (editDuringRequest) {
      await page.evaluate(async () => {
        const tab = (await import('/js/features/chat-state.js')).activeChatTab();
        tab.messages[0].content = 'Edited while the request is in flight';
      });
    }
    await route.fulfill({
      contentType: 'text/event-stream',
      body: 'data: {"choices":[{"delta":{"content":"New answer"}}]}\n\n'
        + 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8000,"completion_tokens":500}}\n\n'
        + 'data: [DONE]\n\n',
    });
  });
}

async function occupancy(page) {
  return page.evaluate(async () => {
    const { activeChatTab, estimateChatContextTokens } = await import('/js/features/chat-state.js');
    return estimateChatContextTokens(activeChatTab());
  });
}

test.describe('chat context accounting @fake-data-bypass', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    await dismissAuthShell(page);
    await page.evaluate(async () => {
      const { switchView } = await import('/js/features/setup-view.js');
      switchView('monitor');
      const Router = (await import('/js/features/router.js')).default;
      Router.navigate('/chat');
    });
    await expect(page.locator('#page-chat')).toBeVisible();
    await seedChat(page);
  });

  test('guard allows 7500-token multi-turn history in a 10000-token window', async ({ page }) => {
    await mockReply(page);
    await page.locator('#chat-input').fill('Third question');
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    const result = await page.evaluate(async () => {
      const { activeChatTab, estimateChatContextTokens } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      return { contents: tab.messages.map(m => m.content), tokens: estimateChatContextTokens(tab), input: tab.total_input_tokens, output: tab.total_output_tokens };
    });
    expect(result.contents).toEqual(['First question', 'First answer', 'Second question', 'Second answer', 'Third question', 'New answer']);
    expect(result.tokens).toBe(8500);
    expect(result.input).toBe(15000);
    expect(result.output).toBe(5000);
    await expect(page.locator('.chat-msg-meta-model').last()).toContainText('85% ctx');
    // Re-rendered and live-stream footers must use the same completed request.
    await page.evaluate(async () => (await import('/js/features/chat-render.js')).renderChatMessages());
    await expect(page.locator('.chat-msg-meta-model').last()).toContainText('85% ctx');
  });

  test('cockpit, footer, telemetry and context card agree on latest prompt plus completion', async ({ page }) => {
    expect(await occupancy(page)).toBe(7500);
    await page.evaluate(async () => {
      (await import('/js/features/nav.js')).refreshTopCockpit();
      (await import('/js/features/chat-params.js')).refreshChatTelemetry();
      (await import('/js/features/context-card.js')).updateContextCard({}, { context_capacity_tokens: 10000 });
    });
    await expect(page.locator('#nav-cockpit-context')).toHaveText('Ctx 75%');
    await expect(page.locator('.chat-msg-meta-model').last()).toContainText('75% ctx');
    await expect(page.locator('#chat-telemetry-context-value')).toHaveText('75%');
    await expect(page.locator('#m-context-gauge-value')).toHaveText('75%');
  });

  test('pending prompt is included conservatively and restored as a durable draft on overflow', async ({ page }) => {
    await page.locator('#chat-input').fill('x'.repeat(12000));
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    const result = await page.evaluate(async () => {
      const { activeChatTab, estimateChatContextTokens } = await import('/js/features/chat-state.js');
      return { last: activeChatTab().messages.at(-1), tokens: estimateChatContextTokens(activeChatTab()), draft: activeChatTab().composer_draft };
    });
    expect(result.last.role).toBe('assistant');
    expect(result.draft).toHaveLength(12000);
    expect(result.tokens).toBe(7500);
    await expect(page.locator('#chat-input')).toHaveValue('x'.repeat(12000));
    await expect(page.locator('.toast').filter({ hasText: 'Context overflow' })).toBeVisible();
  });

  test('edit invalidates surviving prompt usage without destroying billing metadata', async ({ page }) => {
    await seedChat(page, { overflow: true });
    const firstUser = page.locator('.chat-message-user').first();
    await firstUser.locator('[data-chat-action="edit"]').click();
    await firstUser.locator('textarea').fill('Edited question');
    await firstUser.locator('[data-chat-edit="save"]').click();
    expect(await occupancy(page)).toBeLessThan(1000);
    const usage = await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      return { inp: tab.messages.at(-1).input_tokens, out: tab.messages.at(-1).output_tokens, total: tab.total_output_tokens };
    });
    expect(usage).toEqual({ inp: 11000, out: 1500, total: 4500 });
  });

  test('deleting an earlier message invalidates later prompt usage', async ({ page }) => {
    await seedChat(page, { overflow: true });
    await page.locator('.chat-message-user').first().locator('[data-chat-action="delete"]').click();
    await page.locator('.app-confirm-dialog').getByRole('button', { name: 'Confirm', exact: true }).click();
    await expect(page.locator('.chat-message-user')).toHaveCount(1);
    expect(await occupancy(page)).toBeLessThan(1000);
  });

  test('regeneration never uses removed request usage', async ({ page }) => {
    await seedChat(page, { overflow: true });
    await mockReply(page);
    await page.locator('.chat-message-assistant').last().locator('[data-variant-dir="1"]').click();
    await expect(page.locator('.chat-message-assistant').last()).toContainText('New answer');
    expect(await occupancy(page)).toBe(8500);
  });

  test('compact-and-send does not trap kept messages behind their old prompt usage', async ({ page }) => {
    await seedChat(page, { overflow: true });
    await page.locator('#chat-input').fill('Continue after compacting');
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    await expect(page.locator('#chat-input')).toHaveValue('Continue after compacting');
    await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      await (await import('/js/features/chat-params.js')).compactChatTab(tab, 2, false);
    });
    expect(await occupancy(page)).toBeLessThan(1000);
    await mockReply(page);
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    await expect(page.locator('.chat-message-assistant').last()).toContainText('New answer');
    expect(await occupancy(page)).toBe(8500);
  });

  test('persisted invalidation survives a real PUT/GET roundtrip', async ({ page }) => {
    await page.evaluate(async () => {
      const { activeChatTab, normalizeTabForSave, invalidateChatContext, switchChatTab } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      const headers = { ...window.authHeaders(), 'Content-Type': 'application/json' };
      tab.messages[0].content = 'Edited persisted history';
      invalidateChatContext(tab);
      const saved = await fetch(`/api/chat/tabs/${tab.id}`, { method: 'PUT', headers, body: JSON.stringify(normalizeTabForSave(tab)) });
      if (!saved.ok) throw new Error(`save: ${saved.status}`);
      const { chat } = await import('/js/core/app-state.js');
      chat.tabs = [{ id: tab.id, visibility: 'active', messages: null, _loaded: false }];
      await switchChatTab(tab.id);
    });
    expect(await occupancy(page)).toBeLessThan(1000);
    const result = await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      return { fingerprint: tab.messages.at(-1).context_fingerprint, input: tab.messages.at(-1).input_tokens, total: tab.total_output_tokens };
    });
    expect(result).toEqual({ fingerprint: '', input: 6000, total: 4500 });
  });

  test('fingerprinted usage falls back when history changes outside the editor', async ({ page }) => {
    await mockReply(page);
    await page.locator('#chat-input').fill('Third question');
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    expect(await occupancy(page)).toBe(8500);
    await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      tab.messages[0].content = 'Replaced outside the editor';
    });
    expect(await occupancy(page)).toBeLessThan(1000);
  });

  test('changed system context invalidates measured usage before the next send', async ({ page }) => {
    expect(await occupancy(page)).toBe(7500);
    await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      tab.system_prompt = 'A new shorter system prompt';
      tab.context_notes = [{ section: 'Facts', content: 'Retained context note' }];
    });
    expect(await occupancy(page)).toBeLessThan(1000);
  });

  test('lazy-loaded tabs and missing usage never borrow cumulative billing counters', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { newChatTab, estimateChatContextTokens } = await import('/js/features/chat-state.js');
      const tab = newChatTab();
      tab.total_input_tokens = 900000;
      tab.total_output_tokens = 900000;
      tab.last_ctx_pct = 150;
      tab.messages = null;
      const unloaded = estimateChatContextTokens(tab);
      tab.messages = [{ role: 'user', content: 'Short prompt' }, { role: 'assistant', content: 'Short reply', input_tokens: 0, output_tokens: 0 }];
      return { unloaded, loaded: estimateChatContextTokens(tab) };
    });
    expect(result.unloaded).toBeNull();
    expect(result.loaded).toBeLessThan(1000);
  });

  test('legacy compacted tails never reuse pre-compaction prompt counts', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { activeChatTab, estimateChatContextTokens } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      tab.messages = [
        { role: 'system', content: 'Rolling memory', compaction_marker: true },
        { role: 'user', content: 'Kept question' },
        { role: 'assistant', content: 'Kept answer', input_tokens: 11000, output_tokens: 1500 },
      ];
      return estimateChatContextTokens(tab);
    });
    expect(result).toBeLessThan(1000);
  });

  test('measured fingerprints survive lazy loading and reject later replacement', async ({ page }) => {
    await mockReply(page);
    await page.locator('#chat-input').fill('Third question');
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    await page.evaluate(async () => {
      const { activeChatTab, normalizeTabForSave, switchChatTab } = await import('/js/features/chat-state.js');
      const { chat } = await import('/js/core/app-state.js');
      const tab = activeChatTab();
      const headers = { ...window.authHeaders(), 'Content-Type': 'application/json' };
      const saved = await fetch(`/api/chat/tabs/${tab.id}`, { method: 'PUT', headers, body: JSON.stringify(normalizeTabForSave(tab)) });
      if (!saved.ok) throw new Error(`save: ${saved.status}`);
      chat.tabs = [{ id: tab.id, visibility: 'active', messages: null, _loaded: false }];
      await switchChatTab(tab.id);
    });
    expect(await occupancy(page)).toBe(8500);
    await page.evaluate(async () => {
      (await import('/js/features/chat-state.js')).activeChatTab().messages[0].content = 'Replaced after reload';
    });
    expect(await occupancy(page)).toBeLessThan(1000);
  });

  test('edits during a request cannot bless usage for a replaced prompt', async ({ page }) => {
    await mockReply(page, { editDuringRequest: true });
    await page.locator('#chat-input').fill('Third question');
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    expect(await occupancy(page)).toBeLessThan(1000);
    const usage = await page.evaluate(async () => {
      const message = (await import('/js/features/chat-state.js')).activeChatTab().messages.at(-1);
      return { fingerprint: message.context_fingerprint, input: message.input_tokens, output: message.output_tokens };
    });
    expect(usage).toEqual({ fingerprint: '', input: 8000, output: 500 });
  });

  test('consumed one-shot guidance does not become valid usage for the next prompt', async ({ page }) => {
    await mockReply(page);
    await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      tab.armed_story_beats = [{ id: 'beat', instruction: 'One-shot direction', remaining_turns: 0 }];
    });
    await page.locator('#chat-input').fill('Third question');
    await page.evaluate(async () => (await import('/js/features/chat-transport.js')).sendChat());
    expect(await occupancy(page)).toBeLessThan(1000);
    const result = await page.evaluate(async () => {
      const tab = (await import('/js/features/chat-state.js')).activeChatTab();
      return { beats: tab.armed_story_beats, input: tab.messages.at(-1).input_tokens, total: tab.total_output_tokens };
    });
    expect(result).toEqual({ beats: [], input: 8000, total: 5000 });
  });

  test('overflow on resend never removes an existing user turn', async ({ page }) => {
    await seedChat(page, { overflow: true });
    const result = await page.evaluate(async () => {
      const { activeChatTab } = await import('/js/features/chat-state.js');
      const tab = activeChatTab();
      tab.messages.push({ role: 'user', content: 'An existing unanswered turn' });
      await (await import('/js/features/chat-transport.js'))._doSendChat(tab);
      return tab.messages.at(-1);
    });
    expect(result).toMatchObject({ role: 'user', content: 'An existing unanswered turn' });
  });

  test('stripping thinking preserves legacy usage validity after system context changes', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { activeChatTab, estimateChatContextTokens, normalizeTabForSave } = await import('/js/features/chat-state.js');
      const { settingsState } = await import('/js/core/app-state.js');
      const tab = activeChatTab();
      settingsState.persist_thinking_content = true;
      tab.messages.at(-1).thinking_content = 'Legacy reasoning';
      const before = estimateChatContextTokens(tab);
      const originalFingerprint = normalizeTabForSave(tab).messages.at(-1).context_fingerprint;
      tab.system_prompt = 'A replaced system prompt';
      const afterSystemChange = estimateChatContextTokens(tab);
      settingsState.persist_thinking_content = false;
      window.dispatchEvent(new CustomEvent('settings-applied', { detail: { persist_thinking_content: false } }));
      return {
        before,
        afterSystemChange,
        afterStripping: estimateChatContextTokens(tab),
        originalFingerprint,
        retainedFingerprint: normalizeTabForSave(tab).messages.at(-1).context_fingerprint,
        thinking: tab.messages.at(-1).thinking_content,
      };
    });
    expect(result.before).toBe(7500);
    expect(result.afterSystemChange).toBeLessThan(1000);
    expect(result.afterStripping).toBeLessThan(1000);
    expect(result.retainedFingerprint).toBe(result.originalFingerprint);
    expect(result.thinking).toBeUndefined();
  });
});
