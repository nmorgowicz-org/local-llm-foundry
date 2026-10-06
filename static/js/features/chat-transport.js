// ── Chat Transport & Streaming ────────────────────────────────────────────────
// /api/chat calls, streaming decode, abort controller, summarization requests.

import { chat, lastLlamaMetrics, contextCapacityTokens } from '../core/app-state.js';
import {
    activeChatTab,
    substituteNames,
    scheduleChatPersist,
    setChatBusyUI,
    setTransportGetter,
    getChatViewBindings,
    getDefaultRoleBoundaryText,
    normalizeGeneratedMessageContent,
} from './chat-state.js';
import {
    renderChatMessages,
    appendAssistantPlaceholder,
    appendThinkingBlock,
    finalizeAssistantMessage,
    incrementUnreadCount,
    chatScroll,
    chatScrollToEl,
    renderMd,
    renderMdStreaming,
    updateChatTabBadge,
    setChatTransportGetter,
} from './chat-render.js';
import { escapeHtml, formatMetricNumber } from '../core/format.js';
import { autoResizeChatInput } from './chat-state.js';
import { getExplicitModePolicy, resolveActiveTemplate } from './chat-templates.js';
import { showToast, showToastWithActions } from './toast.js';
import Router from './router.js';

// ── Summarization ──────────────────────────────────────────────────────────────

function stripLegacyCompactionPrefix(content) {
    return (content || '').replace(/^\[Context compacted[^\]]*\]\s*/i, '').trim();
}

function formatContextNotesForSummary(contextNotes) {
    const notes = (contextNotes || []).filter(note => note?.content?.trim());
    if (!notes.length) return '';

    const notesBySection = {};
    for (const note of notes) {
        const section = (note.section || 'General').trim();
        if (!notesBySection[section]) notesBySection[section] = [];
        notesBySection[section].push(note.content.trim());
    }

    return Object.entries(notesBySection)
        .map(([section, contents]) => `### ${section}\n- ${contents.join('\n- ')}`)
        .join('\n\n');
}

function buildSummaryPrompt({
    transcript,
    previousMemory,
    recentTail,
    domain,
    systemPrompt,
    contextNotes,
}) {
    const domainInstructions = {
        coding: `Prioritize project goals, files, functions, APIs, commands, error messages, fixes attempted, architectural decisions, constraints, and unresolved implementation tasks. Preserve exact filenames, endpoint names, config keys, and technical decisions whenever present.`,
        creative: `Prioritize characters, relationships, setting, tone, plot beats, promises/foreshadowing, world rules, explicit boundaries, emotional state, and unresolved scene momentum. Preserve who knows what, where the scene ended, and what escalation is pending.`,
        general: `Prioritize user goals, facts, commitments, constraints, decisions, unresolved questions, and the momentum of the latest exchange.`,
    };

    const requiredSections = domain === 'coding'
        ? [
            '## Objectives & Scope',
            '## Persistent Facts',
            '## Technical State',
            '## Files & Artifacts',
            '## Decisions & Constraints',
            '## Open Work',
            '## Recent Momentum',
        ]
        : domain === 'creative'
            ? [
                '## Participants & Dynamics',
                '## World State',
                '## Plot & Scene Memory',
                '## Boundaries & Constraints',
                '## Open Threads',
                '## Recent Momentum',
            ]
            : [
                '## Persistent Facts',
                '## Decisions & Constraints',
                '## Open Threads',
                '## Recent Momentum',
            ];

    const previousMemoryBlock = previousMemory?.trim()
        ? `EXISTING ROLLING MEMORY\n${previousMemory.trim()}`
        : 'EXISTING ROLLING MEMORY\n(none)';
    const systemPromptBlock = systemPrompt?.trim()
        ? `SYSTEM / PERSONA PROMPT\n${systemPrompt.trim()}`
        : 'SYSTEM / PERSONA PROMPT\n(none)';
    const contextNotesBlock = contextNotes?.trim()
        ? `PERSISTENT CONTEXT NOTES\n${contextNotes.trim()}`
        : 'PERSISTENT CONTEXT NOTES\n(none)';
    const recentTailBlock = recentTail?.trim()
        ? `RECENT KEPT TAIL\n${recentTail.trim()}`
        : 'RECENT KEPT TAIL\n(none)';

    return `Refresh the rolling conversation memory for an ongoing chat.

Domain: ${domain}

Goal:
- Merge the existing rolling memory with the newly compacted transcript.
- Preserve facts that must remain true later.
- Preserve unresolved threads and the momentum needed to continue naturally.
- Avoid fluff, repetition, scene-by-scene retelling, and generic summaries.
- Keep the output dense, specific, and directly reusable as memory.

Special instructions:
${domainInstructions[domain] || domainInstructions.general}

Output rules:
- Output only markdown.
- Use the exact section headings below, in order.
- If a section has nothing useful, write a short "(none)" line under it.
- Be concrete. Prefer names, files, decisions, promises, and unresolved actions over vague prose.
- Under "Recent Momentum", preserve the handoff needed for the next response.

Required sections:
${requiredSections.join('\n')}

${systemPromptBlock}

${contextNotesBlock}

${previousMemoryBlock}

${recentTailBlock}

NEWLY COMPACTED TRANSCRIPT
${transcript}`;
}

function buildRoleBoundaryInstruction(tab) {
    const text = tab?.role_boundary_custom?.trim() || getDefaultRoleBoundaryText(tab);
    return `### ROLE BOUNDARY ###\n\n${text}`;
}

function getArmedStoryBeat(tab) {
    const beats = (tab?.armed_story_beats || []).filter(beat => beat.enabled !== false);
    return beats.find(beat => (beat.remaining_turns || 0) === 0) || null;
}

export async function fetchSummary(messages, options = {}) {
    const {
        previousMemory = '',
        recentTailMessages = [],
        domain = 'general',
        systemPrompt = '',
        contextNotes = [],
    } = options;
    // Derive transcript budget from the model's context window so the summarization
    // request never overflows (small models) and doesn't under-use capacity (large models).
    // 65% of context for the transcript; 35% reserved for system prompt + summary output.
    // 3.5 chars/token is a conservative estimate for mixed English/roleplay content.
    const capacityTokens =
        lastLlamaMetrics?.context_capacity_tokens ||
        lastLlamaMetrics?.kv_cache_max ||
        0;
    const MAX_TRANSCRIPT_CHARS = capacityTokens > 0
        ? Math.min(Math.floor(capacityTokens * 3.5 * 0.65), 500_000)
        : 140_000;

    const conversational = messages.filter(m => !m.compaction_marker);
    const rawTranscript = conversational
        .map(m => {
            const label = m.role === 'user' ? 'User' : 'Assistant';
            return `${label}: ${m.content}`;
        })
        .join('\n\n');

    // Smart truncation: preserve the first 25% (setup/persona/initial context) and the last 75%
    // (recent events). Slicing only the beginning discards the most important recent turns.
    let transcript = rawTranscript;
    if (rawTranscript.length > MAX_TRANSCRIPT_CHARS) {
        const headChars = Math.floor(MAX_TRANSCRIPT_CHARS * 0.25);
        const tailChars = MAX_TRANSCRIPT_CHARS - headChars;
        transcript =
            rawTranscript.slice(0, headChars) +
            '\n\n[... middle of conversation omitted for length ...]\n\n' +
            rawTranscript.slice(-tailChars);
    }

    const recentTail = recentTailMessages
        .filter(m => !m.compaction_marker && m.role !== 'system')
        .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
        .join('\n\n');
    const formattedContextNotes = formatContextNotesForSummary(contextNotes);
    const prompt = buildSummaryPrompt({
        transcript,
        previousMemory: stripLegacyCompactionPrefix(previousMemory),
        recentTail,
        domain,
        systemPrompt,
        contextNotes: formattedContextNotes,
    });

    const summaryMessages = [
        {
            role: 'system',
            content: 'You are a precise conversation-memory engine. Rewrite chat history into dense rolling memory for future model context. Output final markdown only. Do not include reasoning, notes about the task, or preamble.',
        },
        {
            role: 'user',
            content: prompt,
        },
    ];

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: summaryMessages,
                stream: true,
                temperature: 0.2,
                thinking_budget_tokens: 0,
                chat_template_kwargs: { enable_thinking: false },
            }),
        });

        if (!resp.ok) return null;

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let summary = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const payload = line.slice(5).trim();
                if (payload === '[DONE]') continue;
                try {
                    const obj = JSON.parse(payload);
                    const delta = obj.choices?.[0]?.delta;
                    if (delta?.content) summary += delta.content;
                } catch { /* skip malformed chunks */ }
            }
        }

        return summary.trim() || null;
    } catch {
        return null;
    }
}

// ── Suggested Prompts ─────────────────────────────────────────────────────────

export async function sendSuggestedPrompt(text) {
    const input = document.getElementById('chat-input');
    if (input) input.value = text;
    sendChat();
}

// ── Send Chat ──────────────────────────────────────────────────────────────────

export async function sendChatWithContent(text, options = {}) {
    if (chat.busy || chat.compactionInProgress) return;
    const tab = activeChatTab();
    if (!tab) return;

    text = text.trim();
    if (!text) return;

    const userMsg = {
        role: 'user',
        content: text,
        timestamp_ms: Date.now(),
    };
    tab.messages.push(userMsg);
    tab.updated_at = Date.now();

    if (typeof renderChatMessages === 'function') renderChatMessages();

    _doSendChat(tab, options);
}

// Send a message that is already in tab.messages (for resend/regenerate — no duplicate push)
export async function sendChatResend(tab) {
    if (chat.busy || chat.compactionInProgress) return;

    if (typeof renderChatMessages === 'function') renderChatMessages();

    // Ensure user is at bottom so auto-scroll works during AI response
    // Use requestAnimationFrame to ensure DOM is fully updated before scrolling
    requestAnimationFrame(() => {
        if (typeof chatScroll === 'function') chatScroll(true);
    });

    _doSendChat(tab);
}

export async function sendChat() {
    if (chat.busy || chat.compactionInProgress) return;
    const tab = activeChatTab();
    if (!tab) return;

    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;
    const isSuggestionDraft = input.dataset.suggestionDraft === 'true';
    input.value = '';
    delete input.dataset.suggestionDraft;
    window.dispatchEvent(new CustomEvent('replyPlanChanged'));
    if (typeof autoResizeChatInput === 'function') autoResizeChatInput();

    if (isSuggestionDraft) {
        const instruction = `Use this user-edited suggestion draft as hidden story direction for the next reply:\n\n${text}\n\nPreserve the established POV, tense, tone, and expected response length. Turn it into a natural multi-sentence or multi-paragraph assistant response. Do not echo, quote, or label the draft note.`;
        await sendOneShotGuideReply(instruction);
        return;
    }

    const userMsg = {
        role: 'user',
        content: text,
        timestamp_ms: Date.now(),
    };
    tab.messages.push(userMsg);
    tab.updated_at = Date.now();
    tab.composer_draft = text;
    scheduleChatPersist();

    if (typeof renderChatMessages === 'function') renderChatMessages();

    // Ensure user is at bottom so auto-scroll works during AI response
    // Use requestAnimationFrame to ensure DOM is fully updated before scrolling
    requestAnimationFrame(() => {
        if (typeof chatScroll === 'function') chatScroll(true);
    });

    const result = await _doSendChat(tab);
    if (result) {
        tab.composer_draft = '';
        scheduleChatPersist();
        window.dispatchEvent(new CustomEvent('replyPlanChanged'));
    }
}

export async function sendQuickGuideReply() {
    if (chat.busy || chat.compactionInProgress) return;
    const tab = activeChatTab();
    if (!tab) return;
    if (!tab.messages?.length) {
        showToast('Quick Guide needs existing chat context before it can respond', 'warning');
        return;
    }

    const lastMsg = tab.messages.at(-1);
    const transientUserPrompt = lastMsg?.role === 'user'
        ? null
        : 'Apply the active quick guide to the existing conversation and write the next assistant reply now. Continue naturally from the latest exchange. Write only the assistant reply. Do not write dialogue, actions, thoughts, or decisions for the user unless explicitly instructed. Do not mention the quick guide unless directly relevant.';

    const result = await _doSendChat(tab, { transientUserPrompt });
    if (result) result.transientUserPrompt = transientUserPrompt;
    return result;
}

export async function sendOneShotGuideReply(instruction) {
    if (chat.busy || chat.compactionInProgress) return null;
    const tab = activeChatTab();
    if (!tab || !instruction?.trim()) return null;
    if (!tab.messages?.length) {
        showToast('Guided suggestions need existing chat context before they can respond', 'warning');
        return null;
    }

    const previousGuide = tab.quick_guide_active || '';
    tab.quick_guide_active = instruction.trim();
    scheduleChatPersist();

    try {
        const lastMsg = tab.messages.at(-1);
        const transientUserPrompt = lastMsg?.role === 'user'
            ? null
            : 'Apply the active guidance to the existing conversation and write the next assistant reply now. Continue naturally from the latest exchange. Write only the assistant reply. Do not write dialogue, actions, thoughts, or decisions for the user unless explicitly instructed.';

        const result = await _doSendChat(tab, { transientUserPrompt, guided: true });
        if (result) result.transientUserPrompt = transientUserPrompt;
        return result;
    } finally {
        tab.quick_guide_active = previousGuide;
        scheduleChatPersist();
    }
}

export async function regenerateQuickGuideReply(tab, msgIdx, quickGuideMeta, variants) {
    if (chat.busy || chat.compactionInProgress) return null;
    if (!tab || !quickGuideMeta?.instruction || typeof msgIdx !== 'number' || msgIdx < 0) return null;

    tab.messages = tab.messages.slice(0, msgIdx);
    tab.updated_at = Date.now();
    tab._pendingVariants = variants?.length ? [...variants] : null;
    renderChatMessages();

    const previousGuide = tab.quick_guide_active || '';
    tab.quick_guide_active = quickGuideMeta.instruction;
    scheduleChatPersist();

    try {
        const result = await _doSendChat(tab, {
            transientUserPrompt: quickGuideMeta.transientUserPrompt ?? null,
            guided: true,
        });
        if (result) result.transientUserPrompt = quickGuideMeta.transientUserPrompt ?? null;
        return result;
    } finally {
        tab.quick_guide_active = previousGuide;
        scheduleChatPersist();
    }
}

export async function _doSendChat(tab, options = {}) {
    const { transientUserPrompt = null, guided = false } = options;
    // Pre-send overflow guard: estimate token usage against current model capacity.
    const capacity = contextCapacityTokens || lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
    if (capacity > 0) {
        // Context occupancy = cumulative output + the LAST request's input.
        // With KV cache each request's input re-includes the whole
        // conversation, so summing total_input_tokens across turns
        // overcounts (a 30k-context chat read as ~109% of 200k here).
        const asst = (tab.messages || []).filter((m) => m.role === 'assistant' && !m.compaction_marker);
        const totalOutput = tab.total_output_tokens
            || asst.reduce((s, m) => s + (m.output_tokens || 0), 0);
        const lastInput = asst.at(-1)?.input_tokens || 0;
        const estimatedTokens = totalOutput + lastInput;
        if (estimatedTokens > capacity) {
            const pct = Math.round((estimatedTokens / capacity) * 100);
            // Restore the user's message before showing the toast
            const lastMsg = tab.messages.at(-1);
            if (lastMsg?.role === 'user') {
                tab.messages.pop();
                const input = document.getElementById('chat-input');
                if (input) input.value = lastMsg.content;
                if (typeof autoResizeChatInput === 'function') autoResizeChatInput();
            }
            chat.busy = false;
            setChatBusyUI(false);
            const toast = showToastWithActions(
                'Context overflow',
                'warning',
                `Chat is ~${pct}% of the ${formatMetricNumber(capacity)}-token window. Compact first.`,
                [{
                    id: 'compact',
                    label: 'Compact now',
                    primary: true,
                    handler: () => {
                        document.getElementById('btn-compact')?.click();
                        toast?.remove();
                    },
                }]
            );
            return null;
        }
    }

    const params = tab.model_params;
    const messages = [];
    const systemParts = [];
    const armedBeat = getArmedStoryBeat(tab);
    let systemPrompt = tab.system_prompt ? substituteNames(tab.system_prompt, tab.ai_name, tab.user_name, tab.ai_gender) : '';
    if (tab.explicit_level > 0) {
        const template = typeof resolveActiveTemplate === 'function'
            ? resolveActiveTemplate(tab.active_template_id) : null;
        const policies = template?.explicit_policies;

        if (policies) {
            if (tab.explicit_level >= 1 && policies.level1) {
                systemPrompt += `\n\n${policies.level1}`;
            }
            if (tab.explicit_level >= 2 && policies.level2) {
                systemPrompt += `\n\n${policies.level2}`;
            }
        } else {
            const explicitPolicy = typeof getExplicitModePolicy === 'function'
                ? getExplicitModePolicy() : '';
            if (explicitPolicy) {
                systemPrompt += `\n\n${explicitPolicy}`;
            }
        }
    }
    if (systemPrompt) {
        systemParts.push(systemPrompt);
    }

    // Fold all guidance into a single leading system message. Some llama.cpp
    // chat templates reject any non-leading or repeated system messages.
    const contextNotes = (tab.context_notes || []).filter(note => note.content?.trim());
    if (contextNotes.length > 0) {
        const notesBySection = {};
        contextNotes.forEach(note => {
            if (!notesBySection[note.section]) {
                notesBySection[note.section] = [];
            }
            notesBySection[note.section].push(note.content);
        });

        Object.entries(notesBySection).forEach(([section, contents]) => {
            const sectionContent = contents.join('\n\n');
            systemParts.push(`### ${section.toUpperCase()} NOTES ###\n\n${sectionContent}`);
        });
    }

    // Inject active quick guide as persistent reply context until changed or cleared.
    const quickGuideInstruction = tab.quick_guide_active || tab.quick_guide_pending || tab._quickGuideInstruction;
    if (quickGuideInstruction) {
        systemParts.push(`### QUICK GUIDE ###\n\n${quickGuideInstruction}`);
    }

    if (armedBeat?.instruction) {
        systemParts.push(`### ARMED STORY BEAT ###\n\n${armedBeat.instruction}`);
    }

    systemParts.push(buildRoleBoundaryInstruction(tab));

    const compactionMarkers = (tab.messages || []).filter(m => m.compaction_marker && m.content?.trim());
    if (compactionMarkers.length > 0) {
        const compactedMemory = compactionMarkers
            .map((marker, index) => `Memory ${index + 1}:\n${marker.content.trim()}`)
            .join('\n\n');
        systemParts.push(`### COMPACTED MEMORY ###\n\n${compactedMemory}`);
    }

    if (systemParts.length > 0) {
        messages.push({ role: 'system', content: systemParts.join('\n\n') });
    }

    // Strip transient/legacy system entries from chat history before sending.
    // The active system prompt, context notes, quick guide, and compaction
    // summaries are injected into the single leading system message above.
    const persistentHistory = (tab.messages || []).filter(m => m.role !== 'system' && !m.compaction_marker);
    messages.push(...persistentHistory.map(m => ({ role: m.role, content: m.content })));
    if (transientUserPrompt) {
        messages.push({
            role: 'user',
            content: transientUserPrompt,
        });
    }

    // Capture debug snapshot of the exact outbound request shape.
    const roughTokens = (str) => Math.max(1, Math.round((str || '').length / 4));
    const systemPartsDetailed = systemParts.map((part) => {
        const headerMatch = part.match(/^###\s*(.+?)\s*###/);
        const label = headerMatch ? headerMatch[1] : 'Base system';
        return {
            label,
            content: part,
            tokens: roughTokens(part),
        };
    });
    const systemTokenMap = {};
    for (const part of systemPartsDetailed) {
        systemTokenMap[part.label] = (systemTokenMap[part.label] || 0) + part.tokens;
    }
    const historyMessagesDetailed = persistentHistory.map((message, index) => ({
        index,
        role: message.role,
        content: message.content,
        tokens: roughTokens(message.content),
    }));
    const historyTokens = historyMessagesDetailed.reduce((sum, message) => sum + message.tokens, 0)
        + (transientUserPrompt ? roughTokens(transientUserPrompt) : 0);
    const finalSystemPrompt = systemParts.join('\n\n');
    const requestPayload = {
        messages,
        stream: true,
        ...params,
    };
    tab._lastDebugData = {
        systemTokens: systemTokenMap,
        systemPartsDetailed,
        finalSystemPrompt,
        historyMessageCount: persistentHistory.length + (transientUserPrompt ? 1 : 0),
        historyMessagesDetailed,
        historyTokens,
        finalUserPrompt: transientUserPrompt || '',
        totalSystemTokens: roughTokens(finalSystemPrompt),
        totalTokens: roughTokens(finalSystemPrompt) + historyTokens,
        modelParams: { ...params },
        capacity: lastLlamaMetrics?.context_capacity_tokens || 0,
        sentAt: Date.now(),
        requestPayload,
    };

    chat.busy = true;
    setChatBusyUI(true);
    chat.abortController = new AbortController();

    let thinkEl = null;
    let thinkContent = '';
    let msgEl = null;
    let msgContent = '';
    let tokenUsage = null;
    const streamTimeoutMs = (params.stream_timeout ?? 120) * 1000;
    let lastContentTime = Date.now();
    let firstTokenReceived = false;
    let regenReverted = false;
    let regenRevertReason = '';

    try {
        const chatEndpoint = guided ? '/api/chat/guided' : '/api/chat';
        const chatResp = await fetch(chatEndpoint, {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            signal: chat.abortController.signal,
            body: JSON.stringify({
                messages,
                stream: true,
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                min_p: params.min_p,
                repeat_penalty: params.repeat_penalty,
                max_tokens: params.max_tokens || 4096,
                thinking_budget_tokens: guided ? 0 : 2048,
                chat_template_kwargs: { enable_thinking: guided ? false : true },
            }),
        });

        if (!chatResp.ok) {
            const errText = await chatResp.text().catch(() => '');
            throw new Error(`HTTP ${chatResp.status}: ${errText}`);
        }

        const reader = chatResp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        // Throttle markdown re-renders to ≤20 fps — parsing the full accumulated
        // string on every token is O(n²) and freezes the browser on long responses.
        let _chatRenderTimer = null;
        const scheduleChatRender = () => {
            if (_chatRenderTimer) return;
            _chatRenderTimer = setTimeout(() => {
                _chatRenderTimer = null;
                if (msgEl) {
                    // eslint-disable-next-line no-unsanitized/property -- LLM output rendered via marked.js in trusted local context
                    msgEl.querySelector('.chat-msg-body').innerHTML =
                        typeof renderMdStreaming === 'function' ? renderMdStreaming(msgContent) : msgContent;
                }
            }, 50);
        };

        while (true) {
            if (firstTokenReceived && streamTimeoutMs > 0 && Date.now() - lastContentTime > streamTimeoutMs) {
                chat.abortController.abort();
                if (!msgContent && tab._pendingVariants) {
                    regenReverted = true;
                    regenRevertReason = 'Generation timed out — restored previous response';
                } else {
                    if (!msgEl && typeof appendAssistantPlaceholder === 'function') {
                        msgEl = appendAssistantPlaceholder();
                    }
                    if (msgEl) {
                        // eslint-disable-next-line no-unsanitized/property -- LLM output rendered via marked.js in trusted local context; fallback span is hardcoded
                        msgEl.querySelector('.chat-msg-body').innerHTML =
                            msgContent ? (typeof renderMd === 'function' ? renderMd(msgContent) : msgContent)
                                : '<span class="chat-stopped">[timed out — no response for too long]</span>';
                    }
                }
                break;
            }

            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop() ?? '';

            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const payload = line.slice(5).trim();
                if (payload === '[DONE]') continue;
                let obj;
                try {
                    obj = JSON.parse(payload);
                } catch { continue; }
                if (typeof obj.error === 'string' && obj.error) {
                    throw new Error(obj.error);
                }
                try {
                    if (obj.usage) {
                        tokenUsage = obj.usage;
                    }
                    if (!obj.usage && obj.timings && obj.choices?.[0]?.finish_reason) {
                        tokenUsage = {
                            prompt_tokens: obj.timings.prompt_n || 0,
                            completion_tokens: obj.timings.predicted_n || 0,
                        };
                        if (tab._lastDebugData) {
                            tab._lastDebugData.promptMs = obj.timings.prompt_ms ?? null;
                            tab._lastDebugData.genMs = obj.timings.predicted_ms ?? null;
                            tab._lastDebugData.draftN = obj.timings.draft_n ?? null;
                            tab._lastDebugData.draftNAccepted = obj.timings.draft_n_accepted ?? null;
                        }
                    }

                    const delta = obj.choices?.[0]?.delta;
                    if (!delta) continue;

                    const rc = delta.reasoning_content ?? '';
                    if (rc) {
                        firstTokenReceived = true;
                        lastContentTime = Date.now();
                        thinkContent += rc;
                        if (!msgEl && typeof appendAssistantPlaceholder === 'function') {
                            msgEl = appendAssistantPlaceholder();
                        }
                        if (msgEl && !thinkEl && typeof appendThinkingBlock === 'function') {
                            thinkEl = appendThinkingBlock(msgEl);
                            // Scroll to show the thinking block as soon as it appears
                            requestAnimationFrame(() => {
                                if (typeof chatScroll === 'function') chatScroll(true);
                            });
                        }
                        if (thinkEl) {
                            thinkEl.querySelector('.chat-thinking-body').textContent = thinkContent;
                            // Update token count in header
                            const tokenCountEl = thinkEl.querySelector('.chat-thinking-token-count');
                            if (tokenCountEl) {
                                // Rough token count: characters / 4 (average English token)
                                const tokenCount = Math.round(thinkContent.length / 4);
                                tokenCountEl.textContent = `(${tokenCount} tokens)`;
                            }
                        }
                    }

                    const c = delta.content ?? '';
                    if (c) {
                        firstTokenReceived = true;
                        msgContent += c;
                        lastContentTime = Date.now();
                        const isFirstToken = !msgEl;
                        if (!msgEl && typeof appendAssistantPlaceholder === 'function') {
                            msgEl = appendAssistantPlaceholder();
                        }
                        if (msgEl) {
                            scheduleChatRender();
                        }
                        // Increment once per response (not per token) so badge = unread message count
                        if (isFirstToken && typeof incrementUnreadCount === 'function') incrementUnreadCount();
                        // On first token, scroll so the AI bubble top sits at the viewport top.
                        // Submit already forced scroll-to-bottom; this gives maximum reading room.
                        if (isFirstToken && msgEl) chatScrollToEl(msgEl);
                    }
                } catch (error) {
                    if (error instanceof Error) throw error;
                }
            }
            if (typeof chatScroll === 'function') chatScroll();
        }
        // Flush any pending throttled render now that streaming is done.
        if (_chatRenderTimer) { clearTimeout(_chatRenderTimer); _chatRenderTimer = null; }
        if (msgEl) {
            // eslint-disable-next-line no-unsanitized/property -- LLM output rendered via marked.js in trusted local context
            msgEl.querySelector('.chat-msg-body').innerHTML =
                typeof renderMdStreaming === 'function' ? renderMdStreaming(msgContent) : msgContent;
        }

    } catch (err) {
        // Detect connection/network errors (404, 503, network failures, etc.)
        const isConnectionError = /HTTP (404|502|503|504)|Failed to fetch|network/i.test(err.message);
        
        if (!msgContent && tab._pendingVariants && err.name !== 'AbortError') {
            regenReverted = true;
            regenRevertReason = `Request failed — restored previous response`;
        } else {
            if (!msgEl && typeof appendAssistantPlaceholder === 'function') {
                msgEl = appendAssistantPlaceholder();
            }
            if (msgEl) {
                const body = msgEl.querySelector('.chat-msg-body');
                if (err.name === 'AbortError') {
                    // eslint-disable-next-line no-unsanitized/property -- LLM output rendered via marked.js in trusted local context; fallback span is hardcoded
                    body.innerHTML = msgContent
                        ? (typeof renderMd === 'function' ? renderMd(msgContent) : msgContent)
                        : '<span class="chat-stopped">[stopped]</span>';
                } else {
                    body.innerHTML = `<span class="chat-error">[error] ${escapeHtml(err.message)}</span>`;
                }
            }
            // Show modal for connection errors on non-regenerate sends (resend, initial send, etc.)
            if (isConnectionError && !regenReverted) {
                showConnectionLostModal();
            }
        }
    }

    if (regenReverted) {
        const prevVariants = tab._pendingVariants;
        const prevContent = prevVariants[prevVariants.length - 1];
        const priorVariants = prevVariants.slice(0, -1);
        const restoredMsg = { role: 'assistant', content: prevContent, timestamp_ms: Date.now() };
        if (priorVariants.length > 0) {
            restoredMsg._variants = priorVariants;
            restoredMsg._variantIndex = priorVariants.length - 1;
        }
        tab.messages.push(restoredMsg);
        tab._pendingVariants = null;
        tab.updated_at = Date.now();
        scheduleChatPersist();
        setChatBusyUI(false);
        chat.busy = false;
        chat.abortController = null;
        const isTimeout = regenRevertReason.includes('timed out');
        if (isTimeout) {
            const toast = showToastWithActions(regenRevertReason, 'warning', null, [{
                id: 'adjust-timeout',
                label: 'Adjust timeout',
                handler: () => { openTimeoutSetting(); toast?.remove(); },
            }]);
        } else {
            // Show modal for connection errors (server restarted, network issue, etc.)
            showConnectionLostModal();
        }
        renderChatMessages();
        if (typeof updateChatTabBadge === 'function') updateChatTabBadge();
        return null;
    }

    let finalMessage = null;
    if (msgContent) {
        msgContent = normalizeGeneratedMessageContent(msgContent, tab, 'assistant');
        const inp = tokenUsage ? (tokenUsage.prompt_tokens ?? 0) : 0;
        const out = tokenUsage ? (tokenUsage.completion_tokens ?? 0) : 0;
        tab.total_input_tokens = (tab.total_input_tokens || 0) + inp;
        tab.total_output_tokens = (tab.total_output_tokens || 0) + out;
        finalMessage = {
            role: 'assistant',
            content: msgContent,
            timestamp_ms: Date.now(),
            input_tokens: inp,
            output_tokens: out,
            cumulativeInputTokens: tab.total_input_tokens,
            cumulativeOutputTokens: tab.total_output_tokens,
            thinking_content: thinkContent || undefined,
        };
        tab.messages.push(finalMessage);
        tab.updated_at = Date.now();
        if (armedBeat) {
            tab.armed_story_beats = (tab.armed_story_beats || []).filter(beat => beat.id !== armedBeat.id);
        }
        tab.armed_story_beats = (tab.armed_story_beats || []).map(beat => {
            if ((beat.remaining_turns || 0) > 0) {
                return { ...beat, remaining_turns: beat.remaining_turns - 1 };
            }
            return beat;
        });
        scheduleChatPersist();
    } else if (!tab.messages.at(-1)?.content) {
        tab.messages.pop();
    }

    if (msgEl) {
        msgEl.dataset.msgIdx = tab.messages.length - 1;
    }
    if (typeof finalizeAssistantMessage === 'function') {
        finalizeAssistantMessage(msgEl, msgContent, tokenUsage, tab);
    }
    setChatBusyUI(false);
    chat.busy = false;
    chat.abortController = null;
    if (typeof updateChatTabBadge === 'function') updateChatTabBadge();
    window.dispatchEvent(new CustomEvent('chatReplyComplete'));

    // Trigger auto-compact if the tab has it enabled and the threshold was hit.
    // Runs after busy is cleared so compaction can proceed without being blocked.
    getChatViewBindings().checkAutoCompact?.(tab);
    return finalMessage ? { message: finalMessage } : null;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function openTimeoutSetting() {
    const panel = document.getElementById('chat-params-panel');
    if (!panel?.classList.contains('open')) {
        document.getElementById('btn-model-params')?.click();
    }
    setTimeout(() => {
        const el = document.getElementById('param-stream-timeout');
        el?.scrollIntoView({ block: 'nearest' });
        el?.focus();
        el?.select();
    }, 50);
}

// ── Stop Chat ──────────────────────────────────────────────────────────────────

export function stopChat() {
    if (chat.abortController) {
        chat.abortController.abort();
        chat.abortController = null;
    }
    chat.busy = false;
    setChatBusyUI(false);
}

// ── Connection Lost Modal + Banner ────────────────────────────────────────────

let connectionLostModalShown = false;

export function showConnectionLostModal() {
    if (connectionLostModalShown) return;
    connectionLostModalShown = true;

    const modal = document.getElementById('connection-lost-modal');
    if (!modal) return;

    modal.classList.add('open');

    document.getElementById('connection-lost-go-welcome-btn')?.addEventListener('click', async () => {
        closeModal();
        Router.navigate('/');
    });
    document.getElementById('connection-lost-dismiss-btn')?.addEventListener('click', () => {
        closeModal();
        showDisconnectedBanner();
    });
    document.getElementById('connection-lost-modal-close')?.addEventListener('click', () => {
        closeModal();
        showDisconnectedBanner();
    });

    function closeModal() {
        modal.classList.remove('open');
        connectionLostModalShown = false;
    }
}

function showDisconnectedBanner() {
    const banner = document.getElementById('disconnected-banner');
    if (!banner) return;
    banner.hidden = false;

    const setupBtn = document.getElementById('disconnected-banner-setup-btn');
    // Replace with a fresh clone to avoid stacking listeners across multiple disconnects
    if (setupBtn) {
        const fresh = setupBtn.cloneNode(true);
        setupBtn.replaceWith(fresh);
        fresh.addEventListener('click', () => {
            hideDisconnectedBanner();
            Router.navigate('/');
        });
    }
}

export function hideDisconnectedBanner() {
    const banner = document.getElementById('disconnected-banner');
    if (banner) banner.hidden = true;
}

// ── Init ───────────────────────────────────────────────────────────────────────

export function initChatTransport() {
    // Wire up transport getter for chat-state and chat-render (avoids circular import)
    const transport = () => ({ sendChat, sendChatResend, sendSuggestedPrompt, sendQuickGuideReply, sendOneShotGuideReply, regenerateQuickGuideReply, stopChat });
    setTransportGetter(transport);
    setChatTransportGetter(transport);
    setChatBusyUI(chat.busy);
}
