// Pure-logic checks demoted from chat/chat-context-accounting.spec.js. They only exercise
// chat-state.js (no DOM, no server), so they run under node:test instead of a browser.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { newChatTab, estimateChatContextTokens } from '../../../static/js/features/chat-state.js';

// Same fixture as seedChat() in the Playwright spec (non-overflow case).
function seedTab() {
  const tab = newChatTab('Context regression');
  tab.auto_compact = false;
  tab.messages = [
    { role: 'user', content: 'First question', timestamp_ms: 1 },
    { role: 'assistant', content: 'First answer', input_tokens: 1000, output_tokens: 3000, timestamp_ms: 2 },
    { role: 'user', content: 'Second question', timestamp_ms: 3 },
    { role: 'assistant', content: 'Second answer', input_tokens: 6000, output_tokens: 1500, timestamp_ms: 4 },
  ];
  tab.total_input_tokens = 7000;
  tab.total_output_tokens = 4500;
  return tab;
}

test('changed system context invalidates measured usage before the next send', () => {
  const tab = seedTab();
  assert.equal(estimateChatContextTokens(tab), 7500);
  tab.system_prompt = 'A new shorter system prompt';
  tab.context_notes = [{ section: 'Facts', content: 'Retained context note' }];
  assert.ok(estimateChatContextTokens(tab) < 1000);
});

test('lazy-loaded tabs and missing usage never borrow cumulative billing counters', () => {
  const tab = newChatTab();
  tab.total_input_tokens = 900000;
  tab.total_output_tokens = 900000;
  tab.last_ctx_pct = 150;
  tab.messages = null;
  const unloaded = estimateChatContextTokens(tab);
  tab.messages = [{ role: 'user', content: 'Short prompt' }, { role: 'assistant', content: 'Short reply', input_tokens: 0, output_tokens: 0 }];
  const loaded = estimateChatContextTokens(tab);
  assert.equal(unloaded, null);
  assert.ok(loaded < 1000);
});

test('legacy compacted tails never reuse pre-compaction prompt counts', () => {
  const tab = seedTab();
  tab.messages = [
    { role: 'system', content: 'Rolling memory', compaction_marker: true },
    { role: 'user', content: 'Kept question' },
    { role: 'assistant', content: 'Kept answer', input_tokens: 11000, output_tokens: 1500 },
  ];
  assert.ok(estimateChatContextTokens(tab) < 1000);
});
