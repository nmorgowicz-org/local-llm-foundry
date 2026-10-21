// Pure-logic checks demoted from chat/chat-shell.spec.js and chat/chat-behavior.spec.js.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { newChatTab, normalizeTabForSave, substituteNames } from '../../../static/js/features/chat-state.js';

test('normalizeTabForSave preserves pinned state', () => {
  const tab = newChatTab('Test');
  tab.pinned = true;
  tab.active_template_id = 'persona-test';
  const result = normalizeTabForSave(tab);

  assert.equal(result.pinned, true);
  assert.equal(result.active_template_id, 'persona-test');
});

test('normalizeTabForSave strips internal token fields from messages', () => {
  const tab = {
    id: 'test-id',
    name: 'Test',
    messages: [
      { role: 'user', content: 'Hello', cumulativeInputTokens: 10, cumulativeOutputTokens: 20 }
    ]
  };
  const result = normalizeTabForSave(tab);

  assert.equal(result.id, 'test-id');
  assert.equal(result.messages[0].cumulativeInputTokens, undefined);
  assert.equal(result.messages[0].cumulativeOutputTokens, undefined);
  assert.equal(result.messages[0].content, 'Hello');
});

test('substituteNames replaces {{gender}} with selected value', () => {
  const result = substituteNames('You are a {{gender}} assistant.', 'AI', 'User', 'male');
  assert.ok(result.includes('male'));
  assert.ok(!result.includes('{{gender}}'));
});
