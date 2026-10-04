// ── Chat Params ───────────────────────────────────────────────────────────────
// Model parameter panel, system prompt panel, style/font/enter-to-send controls,
// and compaction settings.

import { chat, lastLlamaMetrics, monitorState, settingsState, wsData } from '../core/app-state.js';
import { setHtml } from '../core/set-html.js';
import {
    activeChatTab,
    getDefaultRoleBoundaryText,
    registerChatViewBindings,
    scheduleChatPersist,
    substituteNames,
    updateChatName,
} from './chat-state.js';
import { saveSettings } from './settings.js';
import { exportChatTab, importChatTab, renderChatMessages, renderMd } from './chat-render.js';
import { fetchSummary, sendChat } from './chat-transport.js';
import {
    loadTemplates,
    openTemplateManager,
    syncPersonaPanel,
    toggleBehaviorPanel,
    toggleExplicitMode,
} from './chat-templates.js';
import { escapeHtml } from '../core/format.js';
import { showToast, showToastWithActions } from './toast.js';

// Local state — previously on window, migrated to local variables
let chatFont = parseInt(localStorage.getItem('llama-monitor-chat-font') || '100');
let enterToSend = localStorage.getItem('llama-monitor-enter-to-send') == null
    ? settingsState.enter_to_send !== false
    : localStorage.getItem('llama-monitor-enter-to-send') !== 'false';
let paramToastTimer = null;
let chatTelemetryPopoverOpen = false;
let chatTelemetryPinned = localStorage.getItem('llama-monitor-chat-telemetry-pinned') === 'true';
let debugInspectorView = 'slice';
let debugSelectedSliceKey = null;

// ── Model params panel ────────────────────────────────────────────────────────

function toggleModelParamsPanel() {
    const panel = document.getElementById('chat-params-panel');
    const btn = document.getElementById('btn-model-params');
    const wasOpen = panel.classList.contains('open');
    const isOpen = panel.classList.toggle('open');
    if (isOpen && !wasOpen) {
        const behaviorPanel = document.getElementById('chat-behavior-panel');
        const stylePanel = document.getElementById('chat-style-panel');
        const styleLabel = document.getElementById('chat-style-label');
        if (behaviorPanel) behaviorPanel.classList.remove('open');
        if (stylePanel) stylePanel.style.display = 'none';
        if (styleLabel) styleLabel.textContent = 'Style';
        if (btn) btn.classList.add('active');
        syncParamPanelToTab();
    } else if (!isOpen && wasOpen) {
        if (btn) btn.classList.remove('active');
    }
}

function syncParamPanelToTab() {
    const tab = activeChatTab();
    if (!tab) return;
    const p = tab.model_params;
    const set = (id, val, displayId) => {
        const el = document.getElementById(id);
        if (el) { el.value = val ?? ''; }
        const disp = document.getElementById(displayId);
        if (disp) disp.textContent = val ?? '';
    };
    set('param-temperature', p.temperature, 'param-temperature-val');
    set('param-top-p', p.top_p, 'param-top-p-val');
    set('param-top-k', p.top_k, 'param-top-k-val');
    set('param-min-p', p.min_p, 'param-min-p-val');
    set('param-repeat-penalty', p.repeat_penalty, 'param-repeat-penalty-val');
    const maxTok = document.getElementById('param-max-tokens');
    if (maxTok) maxTok.value = p.max_tokens ?? 4096;
    const streamTimeout = document.getElementById('param-stream-timeout');
    if (streamTimeout) streamTimeout.value = p.stream_timeout ?? 120;
}

function onParamChange(key, value) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.model_params[key] = value;
    tab.updated_at = Date.now();
    const map = {
        temperature: 'param-temperature-val',
        top_p: 'param-top-p-val',
        top_k: 'param-top-k-val',
        min_p: 'param-min-p-val',
        repeat_penalty: 'param-repeat-penalty-val',
    };
    const dispId = map[key];
    if (dispId) {
        const el = document.getElementById(dispId);
        if (el) el.textContent = value ?? '';
    }
    scheduleChatPersist();
    clearTimeout(paramToastTimer);
    paramToastTimer = setTimeout(() => showToast('Parameter saved', 'success'), 2000);
    updateParamsDirtyIndicator();
}

function toggleAdvancedParams() {
    const panel = document.getElementById('chat-params-advanced');
    const chevron = document.querySelector('.chat-advanced-chevron');
    const isVisible = panel.style.display !== 'none';
    panel.style.display = isVisible ? 'none' : 'block';
    if (chevron) {
        chevron.style.transform = isVisible ? 'rotate(0deg)' : 'rotate(90deg)';
    }
}

function resetParamsToDefaults() {
    const tab = activeChatTab();
    if (!tab) return;
    tab.model_params = {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        min_p: 0.01,
        repeat_penalty: 1.0,
        max_tokens: 4096,
        stream_timeout: 120,
    };
    tab.updated_at = Date.now();
    syncParamPanelToTab();
    scheduleChatPersist();
    updateParamsDirtyIndicator();
    showToast('Parameters reset to defaults', 'success');
}

export function updateParamsDirtyIndicator() {
}

// ── Copy settings between tabs ────────────────────────────────────────────────

function duplicateTabSettings(sourceId) {
    const source = chat.tabs.find(t => t.id === sourceId);
    const target = activeChatTab();
    if (!source || !target || source.id === target.id) return;
    target.system_prompt = source.system_prompt;
    target.model_params = structuredClone(source.model_params);
    target.updated_at = Date.now();
    scheduleChatPersist();
    syncParamPanelToTab();
    updateParamsDirtyIndicator();
    syncPersonaPanel();
    showToast('Settings copied from "' + source.name + '"', 'success');
}

function showCopySettingsDropdown() {
    const target = activeChatTab();
    if (!target) return;
    const others = chat.tabs.filter(t => t.id !== target.id);
    if (others.length === 0) {
        showToast('No other tabs to copy from', 'info');
        return;
    }
    const toast = showToastWithActions(
        'Copy settings from',
        'info',
        'Select a tab to copy its system prompt and parameters',
        others.map(t => ({
            label: t.name,
            primary: false,
            action: () => duplicateTabSettings(t.id),
        }))
    );
    if (!toast) return;
    setTimeout(() => toast.remove(), 8000);
}

// ── Message limit ─────────────────────────────────────────────────────────────

function onMessageLimitChange(value) {
    const tab = activeChatTab();
    if (!tab) return;
    const limit = Math.max(5, Math.min(200, value));
    tab.visible_message_limit = limit;
    tab.updated_at = Date.now();
    renderChatMessages();
    scheduleChatPersist();
}

function syncMessageLimitInput() {
    const tab = activeChatTab();
    const input = document.getElementById('chat-msg-limit');
    if (tab && input) input.value = tab.visible_message_limit || 15;
}

// ── Compaction ────────────────────────────────────────────────────────────────

// Estimate ctx% from message token metadata (mirrors deriveCtxPctFromMessages in context-card.js).
function estimateCtxPct(tab, capacity) {
    if (!capacity) return null;
    const asst = (tab.messages || []).filter(m => m.role === 'assistant' && !m.compaction_marker);
    if (!asst.length) return null;
    const totalOutput = asst.reduce((sum, m) => sum + (m.output_tokens || 0), 0);
    const lastInput = asst.at(-1).input_tokens || 0;
    return Math.min(200, (totalOutput + lastInput) / capacity * 100); // allow >100 so overflow is visible
}

// Calculate how many recent messages can be kept while staying within 65% of capacity.
// This leaves 35% for the system prompt, tombstone summary, and headroom for new responses.
// Falls back to the default if capacity is unknown or token metadata is sparse.
function calcKeepTailForCapacity(conversational, capacity) {
    if (!capacity || conversational.length === 0) return 15;
    const budget = Math.floor(capacity * 0.65);
    let tokensUsed = 0;
    let keep = 0;
    for (let i = conversational.length - 1; i >= 0; i--) {
        const m = conversational[i];
        // Use recorded tokens; fall back to rough char estimate (4 chars ≈ 1 token)
        const t = (m.input_tokens || 0) + (m.output_tokens || 0)
            || Math.ceil((m.content?.length || 0) / 4);
        if (tokensUsed + t > budget) break;
        tokensUsed += t;
        keep++;
    }
    const minRecentTurns = Math.min(conversational.length - 1, 6);
    // Must keep enough recent turns for continuity and still drop at least 1.
    return Math.max(1, Math.min(Math.max(keep, minRecentTurns), conversational.length - 1));
}

function extractRollingMemory(msg) {
    if (!msg?.content) return '';
    return msg.content.replace(/^\[Context compacted[^\]]*\]\s*/i, '').trim();
}

function inferCompactionDomain(tab, dropped, kept) {
    const haystack = [
        tab?.system_prompt || '',
        ...(tab?.context_notes || []).map(note => `${note.section} ${note.content}`),
        ...dropped.map(m => m.content || ''),
        ...kept.map(m => m.content || ''),
    ].join('\n').toLowerCase();

    const codingSignals = [
        /```/,
        /\b(src\/|static\/|tests\/|cargo\.toml|package\.json|dockerfile)\b/,
        /\bfunction\b|\bclass\b|\bconst\b|\blet\b|\bvar\b|\basync\b/,
        /\bapi\b|\bendpoint\b|\bjson\b|\bsql\b|\bregex\b/,
        /\bbug\b|\berror\b|\btrace\b|\bexception\b|\bcompile\b|\bbuild\b|\blint\b|\btest\b/,
        /\bfile\b|\bpath\b|\bmodule\b|\bcomponent\b|\brefactor\b|\bcommit\b/,
    ];
    if (codingSignals.some(rx => rx.test(haystack))) return 'coding';

    const creativeSignals = [
        /\bscene\b|\bchapter\b|\bplot\b|\bstory\b|\bcharacter\b|\bdialogue\b|\bsetting\b/,
        /\bnoir\b|\bromance\b|\bhorror\b|\bthriller\b|\bfantasy\b|\bsci[- ]?fi\b/,
        /\broleplay\b|\brp\b|\bworld\b|\btone\b|\batmosphere\b|\bemotion\b/,
        /\bkiss\b|\bdesire\b|\bexplicit\b|\berotic\b/,
    ];
    if (creativeSignals.some(rx => rx.test(haystack))) return 'creative';

    return 'general';
}

export async function compactChatTab(tab, keepTail = null, summarize = true) {
    const msgs = tab.messages;
    const systemMsg = msgs[0]?.role === 'system' && !msgs[0]?.compaction_marker ? msgs[0] : null;
    const tombstones = msgs.filter(m => m.compaction_marker);
    const conversational = msgs.filter(m => m.role !== 'system' && !m.compaction_marker);

    // Resolve keepTail: capacity-aware if not specified, so a small model gets a
    // smaller tail and the compacted conversation actually fits its context window.
    const capacity = lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
    const resolvedKeepTail = keepTail ?? (capacity > 0
        ? calcKeepTailForCapacity(conversational, capacity)
        : 15);

    if (conversational.length <= resolvedKeepTail) return;

    chat.compactionInProgress = true;
    setCompactButtonBusy(true);

    const dropped = conversational.slice(0, conversational.length - resolvedKeepTail);
    const kept = conversational.slice(-resolvedKeepTail);
    console.log('[COMPACT] starting — dropped:', dropped.length, 'kept:', kept.length, 'oldTombstones:', tombstones.length);

    let placeholderEl = null;
    if (summarize) {
        const chatMsgs = document.getElementById('chat-messages-inner');
        if (chatMsgs) {
            placeholderEl = document.createElement('div');
            placeholderEl.className = 'chat-message chat-compact-marker compact-marker-summarizing';
            placeholderEl.dataset.compactState = 'loading';
            setHtml(placeholderEl, `
              <div class="compact-marker-content">
                <div class="compact-marker-rule compact-marker-rule-left"></div>
                <div class="compact-marker-pill">
                  <span class="compact-summarizing-dots"><span></span><span></span><span></span></span>
                  <span class="compact-marker-label">Summarizing conversation…</span>
                </div>
                <div class="compact-marker-rule compact-marker-rule-right"></div>
              </div>`);
            chatMsgs.appendChild(placeholderEl);
            chatMsgs.scrollTop = chatMsgs.scrollHeight;
        }
    }

    let tombstoneContent;
    let isSummarized = false;
    let memoryVersion = 1;
    let memoryDomain = 'general';
    let summaryKind = summarize ? 'snapshot-summary' : 'trim-only';
    const existingMemory = tombstones
        .map(extractRollingMemory)
        .filter(Boolean)
        .join('\n\n');
    const totalPreviouslyCompacted = tombstones.reduce((sum, marker) => sum + (marker.dropped_count || 0), 0);

    if (summarize) {
        memoryDomain = inferCompactionDomain(tab, dropped, kept);
        const summary = await fetchSummary(dropped, {
            previousMemory: existingMemory,
            recentTailMessages: kept.slice(-8),
            domain: memoryDomain,
            systemPrompt: substituteNames(tab.system_prompt || '', tab.ai_name, tab.user_name, tab.ai_gender),
            contextNotes: tab.context_notes || [],
        });
        if (summary) {
            tombstoneContent = summary.trim();
            isSummarized = true;
            memoryVersion = 2;
            summaryKind = 'rolling-memory';
        } else {
            tombstoneContent = `[Context compacted — server unavailable for summarization; ${dropped.length} messages dropped]`;
        }
    } else {
        const ctxNote = tab.last_ctx_pct > 0 ? ` · was ${tab.last_ctx_pct}% ctx` : '';
        tombstoneContent = `[Context compacted — ${dropped.length} messages removed${ctxNote}]`;
    }

    const tombstone = {
        role: 'system',
        content: tombstoneContent,
        compaction_marker: true,
        timestamp_ms: Date.now(),
        summarized: isSummarized,
        dropped_count: dropped.length,
        dropped_preview: dropped.slice(0, 8).map(m => ({ role: m.role, snippet: m.content.slice(0, 80) })),
        tokens_freed_estimate: dropped.reduce((sum, m) => sum + Math.round((m.input_tokens || 0) + (m.output_tokens || 0)), 0),
        ctx_pct_before: tab.last_ctx_pct || 0,
        memory_version: memoryVersion,
        memory_domain: memoryDomain,
        summary_kind: summaryKind,
        compacted_at: Date.now(),
        compacted_message_count_total: totalPreviouslyCompacted + dropped.length,
        recent_tail_kept: kept.length,
    };

    if (placeholderEl) placeholderEl.remove();

    tab.messages = [
        ...(systemMsg ? [systemMsg] : []),
        ...tombstones,
        tombstone,
        ...kept,
    ];
    const finalTombstones = tab.messages.filter(m => m.compaction_marker);
    console.log('[COMPACT] done — final:', tab.messages.length, 'tombstones:', finalTombstones.length);
    tab.updated_at = Date.now();
    scheduleChatPersist();
    renderChatMessages();
    setCompactButtonBusy(false);
    chat.compactionInProgress = false;

    setTimeout(() => {
        const markers = document.querySelectorAll('.chat-compact-marker');
        if (markers.length > 0) {
            markers[markers.length - 1].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }, 80);
}

function setCompactButtonBusy(isBusy) {
    const btn = document.getElementById('btn-compact');
    if (!btn) return;
    btn.classList.toggle('chat-btn-busy', isBusy);
    btn.disabled = isBusy;
}

function onManualCompact() {
    const tab = activeChatTab();
    if (!tab) return;
    showCompactConfirmation(tab);
}

function showCompactConfirmation(tab, isAuto = false) {
    const msgs = tab.messages;
    const tombstones = msgs.filter(m => m.compaction_marker);
    const conversational = msgs.filter(m => m.role !== 'system' && !m.compaction_marker);

    const capacity = lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
    const resolvedKeepTail = capacity > 0
        ? calcKeepTailForCapacity(conversational, capacity)
        : 15;

    if (conversational.length <= resolvedKeepTail) {
        showToast('Nothing to compact', 'info');
        return;
    }

    const droppedCount = conversational.length - resolvedKeepTail;
    const keptCount = resolvedKeepTail;
    const dropped = conversational.slice(0, droppedCount);
    const kept = conversational.slice(-keptCount);
    const tokensFreed = dropped.reduce((sum, m) => sum + Math.round((m.input_tokens || 0) + (m.output_tokens || 0)), 0);
    const ctxPct = tab.last_ctx_pct || 0;
    const summarize = tab.auto_compact_summarize !== false;
    const domain = inferCompactionDomain(tab, dropped, kept);
    const existingMemory = tombstones.length > 0;
    let cachedSummary = null;
    let originalSummary = null;

   const overlay = document.createElement('div');
    overlay.className = 'compact-confirm-overlay';

    setHtml(overlay, `
        <div class="compact-confirm-modal" role="dialog" aria-modal="true" aria-labelledby="compact-confirm-title">
            <div class="compact-confirm-header">
                <div class="compact-confirm-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>
                        <line x1="15" y1="9" x2="21" y2="3"/><line x1="3" y1="21" x2="9" y2="15"/>
                    </svg>
                </div>
                <h2 id="compact-confirm-title">${isAuto ? 'Auto-Compact Triggered' : 'Compact Context'}</h2>
                <button class="compact-confirm-close" aria-label="Close" title="Close">&times;</button>
            </div>
            <div class="compact-confirm-body">
                <div class="compact-confirm-warning">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                    <span>This will permanently remove older messages from the visible chat. The model will no longer see them directly.</span>
                </div>
                <div class="compact-confirm-stats">
                    <div class="compact-stat">
                        <span class="compact-stat-label">Total Messages</span>
                        <span class="compact-stat-value">${conversational.length}</span>
                    </div>
                    <div class="compact-stat">
                        <span class="compact-stat-label">Messages Dropped</span>
                        <span class="compact-stat-value compact-stat-danger">${droppedCount}</span>
                    </div>
                    <div class="compact-stat">
                        <span class="compact-stat-label">Messages Kept</span>
                        <span class="compact-stat-value compact-stat-safe">${keptCount}</span>
                    </div>
                    <div class="compact-stat">
                        <span class="compact-stat-label">Est. Tokens Freed</span>
                        <span class="compact-stat-value">${tokensFreed > 0 ? `${(tokensFreed / 1000).toFixed(1)}k` : '—'}</span>
                    </div>
                    <div class="compact-stat">
                        <span class="compact-stat-label">Context Usage</span>
                        <span class="compact-stat-value ${ctxPct > 80 ? 'compact-stat-danger' : ctxPct > 60 ? 'compact-stat-warn' : ''}">${ctxPct > 0 ? `${ctxPct.toFixed(1)}%` : '—'}</span>
                    </div>
                    <div class="compact-stat">
                        <span class="compact-stat-label">Model Capacity</span>
                        <span class="compact-stat-value">${capacity > 0 ? `${(capacity / 1000).toFixed(0)}k` : '—'}</span>
                    </div>
                </div>
                ${summarize ? `
                <div class="compact-confirm-preview">
                    <div class="compact-preview-header">
                        <h3>Summary Preview</h3>
                        <div class="compact-preview-actions" style="display:none;">
                            <button class="compact-preview-btn compact-preview-edit" title="Edit summary">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                            </button>
                            <button class="compact-preview-btn compact-preview-save" title="Save changes" style="display:none;">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                            </button>
                            <button class="compact-preview-btn compact-preview-cancel-edit" title="Cancel edit" style="display:none;">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                            </button>
                            <button class="compact-preview-btn compact-preview-restore" title="Restore default" style="display:none;">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
                            </button>
                        </div>
                    </div>
                    <div class="compact-preview-status">
                        <span class="compact-preview-dot"></span>
                        <span class="compact-preview-label">Generating summary from ${droppedCount} dropped messages…</span>
                    </div>
                    <div class="compact-preview-skeleton">
                        <div class="compact-skeleton-line" style="width:60%"></div>
                        <div class="compact-skeleton-line" style="width:85%"></div>
                        <div class="compact-skeleton-line" style="width:45%"></div>
                        <div class="compact-skeleton-line" style="width:72%"></div>
                        <div class="compact-skeleton-line" style="width:55%"></div>
                        <div class="compact-skeleton-line" style="width:68%"></div>
                    </div>
                    <div class="compact-preview-content" style="display:none;"></div>
                    <textarea class="compact-preview-editor" style="display:none;" rows="10"></textarea>
                </div>` : ''}
                <div class="compact-confirm-details">
                    <button class="compact-details-toggle" type="button">
                        <span class="compact-details-title">What happens next</span>
                        <svg class="compact-details-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="6 9 12 15 18 9"/>
                        </svg>
                    </button>
                    <div class="compact-details-body">
                        <ul>
                            <li>
                                <span class="detail-icon ${summarize ? 'detail-icon-active' : 'detail-icon-inactive'}">${summarize ? '●' : '○'}</span>
                                <span><strong>AI Summarization</strong> ${summarize ? 'enabled' : 'disabled'} — ${summarize ? 'The model will generate a rolling memory summary preserving key facts, decisions, and momentum from the dropped messages.' : 'A simple marker will be inserted noting how many messages were removed.'}</span>
                            </li>
                            <li>
                                <span class="detail-icon detail-icon-info">●</span>
                                <span><strong>Domain: ${domain}</strong> — ${domain === 'creative' ? 'Summary will prioritize characters, setting, plot beats, world rules, and emotional state.' : domain === 'coding' ? 'Summary will prioritize project goals, technical decisions, file names, APIs, and unresolved tasks.' : 'Summary will prioritize goals, facts, commitments, constraints, and unresolved questions.'}</span>
                            </li>
                            <li>
                                <span class="detail-icon ${existingMemory ? 'detail-icon-active' : 'detail-icon-inactive'}">${existingMemory ? '●' : '○'}</span>
                                <span><strong>Existing Memory</strong> — ${existingMemory ? `${tombstones.length} prior compaction${tombstones.length > 1 ? 's' : ''} will be merged into the new summary.` : 'This is the first compaction for this chat.'}</span>
                            </li>
                            <li>
                                <span class="detail-icon detail-icon-info">●</span>
                                <span><strong>Context Notes</strong> — Your ${tab.context_notes?.filter(n => n.content?.trim).length || 0} note${(tab.context_notes?.filter(n => n.content?.trim).length || 0) !== 1 ? 's' : ''} will remain active and visible in the sidebar.</span>
                            </li>
                            <li>
                                <span class="detail-icon detail-icon-info">●</span>
                                <span><strong>System Prompt</strong> — Your system/persona prompt will be preserved unchanged.</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="compact-confirm-footer">
                ${isAuto ? `<button class="btn btn-secondary compact-confirm-defer">Defer</button>` : ''}
                <button class="btn btn-secondary compact-confirm-cancel">Cancel</button>
                <button class="btn btn-danger compact-confirm-ok" disabled>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>
                        <line x1="15" y1="9" x2="21" y2="3"/><line x1="3" y1="21" x2="9" y2="15"/>
                    </svg>
                    ${summarize ? 'Generating…' : 'Compact Now'}
                </button>
            </div>
        </div>
    `);

    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add('visible'));

    const close = () => {
        overlay.classList.add('closing');
        overlay.classList.remove('visible');
        setTimeout(() => overlay.remove(), 300);
    };

    // Collapsible details
    const detailsToggle = overlay.querySelector('.compact-details-toggle');
    const detailsBody = overlay.querySelector('.compact-details-body');
    const detailsChevron = overlay.querySelector('.compact-details-chevron');
    if (detailsToggle && detailsBody && detailsChevron) {
        detailsBody.style.maxHeight = '0';
        detailsBody.style.overflow = 'hidden';
        detailsToggle.addEventListener('click', () => {
            const isExpanded = detailsChevron.classList.toggle('expanded');
            detailsBody.style.maxHeight = isExpanded ? `${detailsBody.scrollHeight}px` : '0';
            detailsBody.style.transition = `max-height 0.25s ease`;
        });
    }

    const okBtn = overlay.querySelector('.compact-confirm-ok');
    const cancelBtn = overlay.querySelector('.compact-confirm-cancel');
    overlay.querySelector('.compact-confirm-close').addEventListener('click', close);
    cancelBtn.addEventListener('click', close);

    // Summary preview handlers
    const previewActions = overlay.querySelector('.compact-preview-actions');
    const editBtn = overlay.querySelector('.compact-preview-edit');
    const saveBtn = overlay.querySelector('.compact-preview-save');
    const cancelEditBtn = overlay.querySelector('.compact-preview-cancel-edit');
    const restoreBtn = overlay.querySelector('.compact-preview-restore');
    const previewContent = overlay.querySelector('.compact-preview-content');
    const previewEditor = overlay.querySelector('.compact-preview-editor');

    if (summarize) {
        const existingMemoryText = tombstones
            .map(extractRollingMemory)
            .filter(Boolean)
            .join('\n\n');
        fetchSummary(dropped, {
            previousMemory: existingMemoryText,
            recentTailMessages: kept.slice(-8),
            domain: domain,
            systemPrompt: substituteNames(tab.system_prompt || '', tab.ai_name, tab.user_name, tab.ai_gender),
            contextNotes: tab.context_notes || [],
        }).then(summary => {
            cachedSummary = summary || null;
            originalSummary = cachedSummary;
            const skeleton = overlay.querySelector('.compact-preview-skeleton');
            const status = overlay.querySelector('.compact-preview-status');
            if (skeleton) skeleton.style.display = 'none';
            if (status) status.remove();
            if (previewContent) {
                previewContent.style.display = 'block';

                setHtml(previewContent, cachedSummary
                    ? renderMd(cachedSummary)
                    : '<p class="compact-preview-fallback">Summary generation failed — compact will still proceed with a basic marker.</p>');
            }
            if (previewActions) previewActions.style.display = 'flex';
            okBtn.disabled = false;
            setHtml(okBtn, `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>
                    <line x1="15" y1="9" x2="21" y2="3"/><line x1="3" y1="21" x2="9" y2="15"/>
                </svg>
                Compact Now
            `);
        }).catch(() => {
            const skeleton = overlay.querySelector('.compact-preview-skeleton');
            const status = overlay.querySelector('.compact-preview-status');
            if (skeleton) skeleton.style.display = 'none';
            if (status) status.remove();
            if (previewContent) {
                previewContent.style.display = 'block';
                setHtml(previewContent, '<p class="compact-preview-fallback">Summary preview failed — compact will still proceed with a basic marker.</p>');
            }
            okBtn.disabled = false;
            setHtml(okBtn, `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/>
                    <line x1="15" y1="9" x2="21" y2="3"/><line x1="3" y1="21" x2="9" y2="15"/>
                </svg>
                Compact Now
            `);
        });

        // Edit mode
        if (editBtn) {
            editBtn.addEventListener('click', () => {
                editBtn.style.display = 'none';
                restoreBtn.style.display = 'inline-flex';
                if (saveBtn) saveBtn.style.display = 'inline-flex';
                if (cancelEditBtn) cancelEditBtn.style.display = 'inline-flex';
                if (previewContent && previewEditor) {
                    previewContent.style.display = 'none';
                    previewEditor.style.display = 'block';
                    previewEditor.value = cachedSummary || '';
                }
            });
        }
        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                if (previewEditor && previewContent) {
                    cachedSummary = previewEditor.value.trim() || cachedSummary;

                    setHtml(previewContent, renderMd(cachedSummary));
                    previewEditor.style.display = 'none';
                    previewContent.style.display = 'block';
                }
                editBtn.style.display = 'inline-flex';
                saveBtn.style.display = 'none';
                cancelEditBtn.style.display = 'none';
                restoreBtn.style.display = 'none';
            });
        }
        if (cancelEditBtn) {
            cancelEditBtn.addEventListener('click', () => {
                if (previewEditor && previewContent) {
                    previewEditor.style.display = 'none';
                    previewContent.style.display = 'block';
                }
                editBtn.style.display = 'inline-flex';
                saveBtn.style.display = 'none';
                cancelEditBtn.style.display = 'none';
                restoreBtn.style.display = 'none';
            });
        }
        if (restoreBtn) {
            restoreBtn.addEventListener('click', () => {
                if (originalSummary && previewEditor) {
                    previewEditor.value = originalSummary;
                }
            });
        }
    }

    // Defer button (auto-compact only)
    const deferBtn = overlay.querySelector('.compact-confirm-defer');
    if (deferBtn) {
        deferBtn.addEventListener('click', () => {
            tab._compactDeferred = true;
            close();
            showToast('Compaction deferred — will check again after next response', 'info');
        });
    }

    okBtn.addEventListener('click', () => {
        close();
        setTimeout(() => {
            if (cachedSummary) {
                tab._compactPreviewSummary = cachedSummary;
            }
            compactChatTab(tab, null, summarize);
        }, 300);
    });
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) close();
    });
}

// Called after each response (via view binding) and after model switch.
// Fires compaction if auto_compact is on and the tab has hit its threshold.
export async function checkAutoCompact(tab) {
    if (!tab || !tab.auto_compact || chat.compactionInProgress || chat.busy) return;
    if (tab._compactDeferred) {
        tab._compactDeferred = false;
        return;
    }
    const capacity = lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
    if (!capacity) return;

    const ctxPct = estimateCtxPct(tab, capacity);
    if (ctxPct === null) return;

    const mode = tab.compact_mode || 'percent';
    let shouldCompact = false;
    if (mode === 'optimized') {
        // Compact when fewer than 25k tokens remain
        const asstMsgs = (tab.messages || []).filter(m => m.role === 'assistant' && !m.compaction_marker);
        const totalOutput = asstMsgs.reduce((sum, m) => sum + (m.output_tokens || 0), 0);
        const lastInput = asstMsgs.at(-1)?.input_tokens || 0;
        shouldCompact = capacity - (totalOutput + lastInput) < 25_000;
    } else {
        const threshold = (tab.compact_threshold || 0.8) * 100;
        shouldCompact = ctxPct >= threshold;
    }

    if (shouldCompact) {
        showCompactConfirmation(tab, true);
    }
}

function onAutoCompactChange(checked) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.auto_compact = checked;
    tab.updated_at = Date.now();
    document.getElementById('compact-threshold-field').style.opacity = checked ? '1' : '0.4';
    const summarizeField = document.getElementById('compact-summarize-field');
    if (summarizeField) summarizeField.style.opacity = checked ? '1' : '0.4';
    scheduleChatPersist();
}

function onAutoCompactSummarizeChange(checked) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.auto_compact_summarize = checked;
    tab.updated_at = Date.now();
    scheduleChatPersist();
}

function onCompactModeChange(mode) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.compact_mode = mode;
    tab.updated_at = Date.now();
    scheduleChatPersist();
    syncCompactSettingsUI(tab);
}

function onCompactThresholdChange(value) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.compact_threshold = value / 100;
    tab.updated_at = Date.now();
    document.getElementById('chat-compact-threshold-val').textContent = `${value}%`;
    scheduleChatPersist();
}

function updateCtxPressureBar(pct) {
    const bar = document.getElementById('ctx-pressure-bar');
    const fill = document.getElementById('ctx-pressure-fill');
    if (!bar || !fill) return;
    if (!pct || pct <= 0) { bar.style.display = 'none'; return; }
    bar.style.display = 'block';
    fill.style.transform = 'scaleX(' + Math.min(pct, 100) / 100 + ')';
    fill.className = 'ctx-pressure-fill' +
        (pct >= 90 ? ' ctx-pressure-critical' :
         pct >= 75 ? ' ctx-pressure-high' :
         pct >= 50 ? ' ctx-pressure-medium' : '');

    const textarea = document.getElementById('chat-input');
    if (textarea) {
        textarea.classList.remove('ctx-pressure-medium', 'ctx-pressure-high', 'ctx-pressure-critical');
        if (pct >= 90) textarea.classList.add('ctx-pressure-critical');
        else if (pct >= 75) textarea.classList.add('ctx-pressure-high');
        else if (pct >= 50) textarea.classList.add('ctx-pressure-medium');
    }
}

function setChatTelemetryPopover(open) {
    chatTelemetryPopoverOpen = open;
    const popover = document.getElementById('chat-telemetry-popover');
    const btn = document.getElementById('chat-telemetry-btn');
    popover?.classList.toggle('hidden', !open);
    btn?.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function applyChatTelemetryMode() {
    const rail = document.getElementById('chat-telemetry-rail');
    const inlineHost = document.getElementById('chat-telemetry-inline-host');
    const popoverBody = document.getElementById('chat-telemetry-popover-body');
    const pinBtn = document.getElementById('chat-telemetry-pin-btn');
    const pinLabel = document.getElementById('chat-telemetry-pin-label');
    const pinHint = document.getElementById('chat-telemetry-pin-hint');
    const note = document.getElementById('chat-telemetry-popover-note');

    if (rail && inlineHost && popoverBody) {
        rail.classList.toggle('chat-telemetry-rail-floating', !chatTelemetryPinned);
        if (chatTelemetryPinned) {
            inlineHost.appendChild(rail);
        } else {
            popoverBody.insertBefore(rail, note || null);
        }
    }

    inlineHost?.classList.toggle('hidden', !chatTelemetryPinned);
    note?.classList.toggle('hidden', !chatTelemetryPinned);

    if (pinBtn) pinBtn.setAttribute('aria-pressed', chatTelemetryPinned ? 'true' : 'false');
    if (pinLabel) pinLabel.textContent = chatTelemetryPinned ? 'Pinned Inline' : 'Popup Mode';
    if (pinHint) pinHint.textContent = chatTelemetryPinned ? 'Collapse back to popup' : 'Pin below toolbar';
}

function setChatTelemetryPinned(nextPinned) {
    chatTelemetryPinned = !!nextPinned;
    localStorage.setItem('llama-monitor-chat-telemetry-pinned', chatTelemetryPinned ? 'true' : 'false');
    applyChatTelemetryMode();
    if (chatTelemetryPinned) {
        setChatTelemetryPopover(false);
    } else {
        refreshChatTelemetry();
    }
}

export function refreshChatTelemetry() {
    const hasActiveEndpoint = !!wsData?.active_session_id;
    const l = hasActiveEndpoint ? lastLlamaMetrics : null;
    const tab = activeChatTab();
    const stateEl = document.getElementById('chat-telemetry-state');
    const promptStage = document.getElementById('chat-telemetry-stage-prompt');
    const outputStage = document.getElementById('chat-telemetry-stage-output');
    const promptValue = document.getElementById('chat-telemetry-prompt-value');
    const genValue = document.getElementById('chat-telemetry-gen-value');
    const promptBar = document.getElementById('chat-telemetry-prompt-bar');
    const genBar = document.getElementById('chat-telemetry-gen-bar');
    const contextValue = document.getElementById('chat-telemetry-context-value');
    const contextRing = document.getElementById('chat-telemetry-context-ring');
    const liveRate = document.getElementById('chat-telemetry-live-rate');
    const specMetric = document.getElementById('chat-telemetry-spec-metric');
    const specValue = document.getElementById('chat-telemetry-spec-value');
    const telemetryBtn = document.getElementById('chat-telemetry-btn');

    const promptRate = hasActiveEndpoint ? (l?.prompt_tokens_per_sec || 0) : 0;
    const genRate = l?.generation_tokens_per_sec || 0;
    const promptDisplayRate = promptRate > 0 ? promptRate : (l?.last_prompt_tokens_per_sec || 0);
    const genDisplayRate = genRate > 0 ? genRate : (l?.last_generation_tokens_per_sec || 0);
    const generationAvailable = !!l?.slot_generation_available;
    const generationActive = !!l?.slot_generation_active || (l?.slots_processing || 0) > 0;
    const generated = l?.slot_generation_tokens || 0;
    let label = 'idle';
    let stateClass = 'idle';
    if (!hasActiveEndpoint) {
        label = 'attach';
    } else if (chat.compactionInProgress) {
        label = 'compact';
        stateClass = 'warning';
    } else if (promptRate > 0 && genRate <= 0) {
        label = 'prompting';
        stateClass = 'live';
    } else if (generationActive || genRate > 0) {
        label = 'generating';
        stateClass = 'live';
    } else if (chat.busy) {
        label = 'waiting';
        stateClass = 'warning';
    } else if (promptDisplayRate > 0 || genDisplayRate > 0) {
        label = 'retained';
        stateClass = 'idle';
    }

    if (stateEl) {
        stateEl.textContent = label;
        stateEl.className = 'metric-live-chip ' + stateClass;
    }
    if (telemetryBtn) {
        telemetryBtn.classList.remove('idle', 'live');
        telemetryBtn.classList.add(stateClass === 'live' ? 'live' : 'idle');
    }

    const useThroughputFallback = !generationAvailable;
    const isPromptPhase = useThroughputFallback
        ? !!(l?.prompt_throughput_active && !l?.generation_throughput_active)
        : generationActive && generated <= 1;
    const isOutputPhase = useThroughputFallback
        ? !!l?.generation_throughput_active
        : generationActive && generated > 1;

    promptStage?.classList.toggle('active', isPromptPhase);
    promptStage?.classList.toggle('idle', !isPromptPhase && !generationActive);
    outputStage?.classList.toggle('active', isOutputPhase);
    outputStage?.classList.toggle('idle', !isOutputPhase && !generationActive);

    if (promptValue) promptValue.textContent = promptDisplayRate > 0 ? promptDisplayRate.toFixed(1) : '\u2014';
    if (genValue) genValue.textContent = genDisplayRate > 0 ? genDisplayRate.toFixed(1) : '\u2014';

    const tpd = l?.tokens_per_decode ?? 0;
    if (specMetric && specValue) {
        if (tpd > 1.05) {
            specValue.textContent = tpd.toFixed(2) + '\u00d7';
            specMetric.classList.remove('hidden');
        } else {
            specMetric.classList.add('hidden');
        }
    }

    if (liveRate) {
        liveRate.textContent = genDisplayRate > 0 ? genDisplayRate.toFixed(1) + ' t/s' : (generationActive ? 'warming' : '\u2014');
    }
    if (promptBar) {
        const promptPct = monitorState.speedMax.prompt > 0 && promptDisplayRate > 0
            ? Math.max(4, (promptDisplayRate / monitorState.speedMax.prompt) * 100)
            : 0;
        promptBar.style.transform = 'scaleX(' + (promptPct / 100) + ')';
    }
    if (genBar) {
        const genPct = monitorState.speedMax.generation > 0 && genDisplayRate > 0
            ? Math.max(4, (genDisplayRate / monitorState.speedMax.generation) * 100)
            : 0;
        genBar.style.transform = 'scaleX(' + (genPct / 100) + ')';
    }

    const capacity = hasActiveEndpoint ? (l?.context_capacity_tokens || l?.kv_cache_max || 0) : 0;
    const ctxPct = tab && capacity ? estimateCtxPct(tab, capacity) : 0;
    updateCtxPressureBar(ctxPct || 0);
    if (tab && hasActiveEndpoint) tab.last_ctx_pct = ctxPct || 0;
    if (contextValue) contextValue.textContent = ctxPct > 0 ? Math.round(ctxPct) + '%' : '\u2014';
    if (contextRing) {
        const pct = Math.max(0, Math.min(100, Math.round(ctxPct || 0)));
        const color = pct >= 90 ? '#f87171' : pct >= 75 ? '#fbbf24' : '#5eead4';
        contextRing.style.setProperty('--progress', String(pct));
        contextRing.style.setProperty('--chat-telemetry-context-color', color);
    }
}

function syncCompactSettingsUI(tab) {
    const autoToggle = document.getElementById('chat-auto-compact');
    const thresholdSlider = document.getElementById('chat-compact-threshold');
    const thresholdVal = document.getElementById('chat-compact-threshold-val');
    const thresholdField = document.getElementById('compact-threshold-field');
    if (!autoToggle || !thresholdSlider) return;

    const isOn = !!tab?.auto_compact;
    const mode = tab?.compact_mode || 'percent';
    const isOptimized = mode === 'optimized';

    autoToggle.checked = isOn;

    document.getElementById('compact-mode-percent')?.classList.toggle('compact-mode-pill-active', !isOptimized);
    document.getElementById('compact-mode-optimized')?.classList.toggle('compact-mode-pill-active', isOptimized);
    document.getElementById('compact-mode-help-percent').style.display = isOptimized ? 'none' : '';
    document.getElementById('compact-mode-help-optimized').style.display = isOptimized ? '' : 'none';

    const modeField = document.getElementById('compact-mode-field');
    if (modeField) modeField.style.opacity = isOn ? '1' : '0.4';
    thresholdField.style.display = isOptimized ? 'none' : '';
    thresholdField.style.opacity = isOn ? '1' : '0.4';
    thresholdSlider.value = (tab?.compact_threshold || 0.8) * 100;
    thresholdVal.textContent = `${thresholdSlider.value}%`;

    const summarizeToggle = document.getElementById('chat-auto-compact-summarize');
    const summarizeField = document.getElementById('compact-summarize-field');
    if (summarizeToggle) summarizeToggle.checked = !!tab?.auto_compact_summarize;
    if (summarizeField) summarizeField.style.opacity = isOn ? '1' : '0.4';

    const btn = document.getElementById('btn-compact');
    if (btn && tab) {
        const conversational = tab.messages.filter(m => m.role !== 'system' && !m.compaction_marker);
        const capacity = lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
        const resolvedKeepTail = capacity > 0
            ? calcKeepTailForCapacity(conversational, capacity)
            : 15;
        const willDrop = Math.max(0, conversational.length - resolvedKeepTail);
        btn.title = willDrop > 0
            ? `Compact context — will remove ${willDrop} oldest messages`
            : 'Compact context — nothing to remove yet';
    }
}

// ── Style / Font / Enter-to-send ──────────────────────────────────────────────

function applyChatStyle(style) {
    const page = document.getElementById('page-chat');
    if (page) {
        page.dataset.chatStyle = style;
    }
}

export { applyChatStyle };

const CHAT_STYLE_LABELS = { rounded: 'Rounded', compact: 'Compact', minimal: 'Minimal', bubbly: 'Bubbly', paper: 'Paper', terminal: 'Terminal', slate: 'Slate' };

function toggleStylePanel() {
    const panel = document.getElementById('chat-style-panel');
    const btn = document.getElementById('btn-chat-style');
    const isOpen = panel.style.display !== 'none';
    panel.style.display = isOpen ? 'none' : 'block';
    if (!isOpen) {
        const current = document.getElementById('page-chat')?.dataset.chatStyle || 'rounded';
        panel.querySelectorAll('.chat-style-card').forEach(card => {
            card.classList.toggle('active', card.dataset.style === current);
        });
        const behaviorPanel = document.getElementById('chat-behavior-panel');
        const paramsPanel = document.getElementById('chat-params-panel');
        if (behaviorPanel) behaviorPanel.classList.remove('open');
        if (paramsPanel) paramsPanel.classList.remove('open');
        const styleLabel = document.getElementById('chat-style-label');
        if (styleLabel) styleLabel.textContent = 'Style';
        if (btn) btn.classList.add('active');
    } else {
        if (btn) btn.classList.remove('active');
    }
}

function selectChatStyle(style) {
    applyChatStyle(style);
    localStorage.setItem('llama-monitor-chat-style', style);
    updateChatStyleLabel(style);
    const select = document.getElementById('pref-chat-style');
    if (select) select.value = style;
    document.getElementById('chat-style-panel').style.display = 'none';
    showToast(`Style: ${CHAT_STYLE_LABELS[style]}`, 'success');
}

function updateChatStyleLabel(style) {
    const label = document.getElementById('chat-style-label');
    if (label) label.textContent = 'Style';
    const btn = document.getElementById('btn-chat-style');
    if (btn && !style) {
        btn.classList.remove('active');
    }
}

function adjustChatFont(delta) {
    chatFont = Math.max(70, Math.min(150, chatFont + delta * 10));
    localStorage.setItem('llama-monitor-chat-font', chatFont);
    applyChatFontSize();
}

function applyChatFontSize() {
    const messages = document.getElementById('chat-messages');
    if (messages) {
        messages.style.setProperty('--chat-font-scale', chatFont / 100);
    }
    const inputRow = document.getElementById('chat-input-row');
    if (inputRow) {
        inputRow.style.setProperty('--chat-font-scale', chatFont / 100);
    }
    const label = document.getElementById('chat-font-value');
    if (label) label.textContent = chatFont + '%';
}

function onEnterToggleChange(checked) {
    enterToSend = checked;
    settingsState.enter_to_send = checked;
    const prefCheckbox = document.getElementById('pref-enter-to-send');
    if (prefCheckbox) prefCheckbox.checked = checked;
}

export function getEnterToSend() {
    return enterToSend;
}

export function setEnterToSend(checked) {
    onEnterToggleChange(checked);
}

function initEnterToggle() {
    const toggle = document.getElementById('chat-enter-toggle-input');
    if (toggle) toggle.checked = enterToSend;
}

window.addEventListener('settings-applied', () => {
    enterToSend = settingsState.enter_to_send !== false;
    const toggle = document.getElementById('chat-enter-toggle-input');
    if (toggle) toggle.checked = enterToSend;
    const prefCheckbox = document.getElementById('pref-enter-to-send');
    if (prefCheckbox) prefCheckbox.checked = enterToSend;
});

function initChatStyle() {
    const savedChatStyle = localStorage.getItem('llama-monitor-chat-style') || 'rounded';
    applyChatStyle(savedChatStyle);
    const select = document.getElementById('pref-chat-style');
    if (select) select.value = savedChatStyle;
    updateChatStyleLabel(savedChatStyle);
}

function initChatInputHandler() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && enterToSend) {
            e.preventDefault();
            sendChat();
        }
    });
}

// ── Chat names ────────────────────────────────────────────────────────────────

function loadChatNames() {
    const tab = activeChatTab();
    if (!tab) return;

    const aiInput = document.getElementById('chat-ai-name');
    const userInput = document.getElementById('chat-user-name');

    if (aiInput) aiInput.value = tab.ai_name || '';
    if (userInput) userInput.value = tab.user_name || '';
    syncPersonaPanel();
}

// ── Gender pill handler ───────────────────────────────────────────────────────

function onGenderChange(gender) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.ai_gender = gender;
    tab.updated_at = Date.now();
    scheduleChatPersist();
    document.querySelectorAll('.chat-gender-pill').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.gender === gender);
    });
    showToast(`Gender: ${gender.charAt(0).toUpperCase() + gender.slice(1)}`, 'success');
}

// ── Role boundary handlers ────────────────────────────────────────────────────

let roleBoundaryToastTimer = null;

function onRoleBoundaryChange() {
    const tab = activeChatTab();
    if (!tab) return;
    const input = document.getElementById('chat-role-boundary-input');
    if (!input) return;
    const defaultText = getDefaultRoleBoundaryText(tab);
    const typed = input.value.trim();
    tab.role_boundary_custom = (typed && typed !== defaultText) ? typed : null;
    tab.updated_at = Date.now();
    scheduleChatPersist();
    clearTimeout(roleBoundaryToastTimer);
    roleBoundaryToastTimer = setTimeout(() => showToast('Role boundary saved', 'success'), 1500);
}

function resetRoleBoundaryToDefault() {
    const tab = activeChatTab();
    if (!tab) return;
    tab.role_boundary_custom = null;
    tab.updated_at = Date.now();
    scheduleChatPersist();
    const input = document.getElementById('chat-role-boundary-input');
    if (input) input.value = getDefaultRoleBoundaryText(tab);
    showToast('Role boundary reset to default', 'success');
}

function toggleRoleBoundarySection() {
    const body = document.getElementById('chat-role-boundary-body');
    const chevron = document.getElementById('chat-role-boundary-chevron');
    if (!body) return;
    const isOpen = body.style.display !== 'none';
    body.style.display = isOpen ? 'none' : 'block';
    if (chevron) chevron.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(90deg)';
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initChatParams() {
    // Call setup functions that bind DOM event listeners
    applyChatFontSize();
    initEnterToggle();
    initChatStyle();
    initChatInputHandler();
    initChatResizeHandle();
    applyChatTelemetryMode();
    setChatTelemetryPopover(false);
    document.getElementById('chat-telemetry-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        setChatTelemetryPopover(!chatTelemetryPopoverOpen);
        refreshChatTelemetry();
    });
    document.getElementById('chat-telemetry-pin-btn')?.addEventListener('click', (e) => {
        e.stopPropagation();
        setChatTelemetryPinned(!chatTelemetryPinned);
    });
    document.addEventListener('click', e => {
        if (!e.target.closest('#chat-telemetry-btn') && !e.target.closest('#chat-telemetry-popover')) {
            setChatTelemetryPopover(false);
        }
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') setChatTelemetryPopover(false);
    });

    // Bind chat header buttons
    document.getElementById('btn-behavior')?.addEventListener('click', (_e) => {
    const btn = document.getElementById('btn-behavior');
    const panel = document.getElementById('chat-behavior-panel');
    const wasOpen = panel.classList.contains('open');
    toggleBehaviorPanel();
    setTimeout(() => {
        const isOpen = panel.classList.contains('open');
        const styleBtn = document.getElementById('btn-chat-style');
        const paramsBtn = document.getElementById('btn-model-params');
        if (isOpen && !wasOpen) {
            btn.classList.add('active');
            if (styleBtn) styleBtn.classList.remove('active');
            if (paramsBtn) paramsBtn.classList.remove('active');
        } else if (!isOpen && wasOpen) {
            btn.classList.remove('active');
        }
    }, 0);
});
    document.getElementById('btn-model-params')?.addEventListener('click', (_e) => {
    const btn = document.getElementById('btn-model-params');
    const panel = document.getElementById('chat-params-panel');
    const wasOpen = panel.classList.contains('open');
    toggleModelParamsPanel();
    setTimeout(() => {
        const isOpen = panel.classList.contains('open');
        const behaviorBtn = document.getElementById('btn-behavior');
        const styleBtn = document.getElementById('btn-chat-style');
        if (isOpen && !wasOpen) {
            btn.classList.add('active');
            if (behaviorBtn) behaviorBtn.classList.remove('active');
            if (styleBtn) styleBtn.classList.remove('active');
        } else if (!isOpen && wasOpen) {
            btn.classList.remove('active');
        }
    }, 0);
});
    document.getElementById('btn-chat-style')?.addEventListener('click', (_e) => {
    const btn = document.getElementById('btn-chat-style');
    toggleStylePanel();
    setTimeout(() => {
        const panel = document.getElementById('chat-style-panel');
        const isOpen = panel.style.display !== 'none';
        const behaviorBtn = document.getElementById('btn-behavior');
        const paramsBtn = document.getElementById('btn-model-params');
        if (isOpen) {
            btn.classList.add('active');
            if (behaviorBtn) behaviorBtn.classList.remove('active');
            if (paramsBtn) paramsBtn.classList.remove('active');
        } else {
            btn.classList.remove('active');
        }
    }, 0);
});
    document.getElementById('btn-compact')?.addEventListener('click', onManualCompact);
    registerPersonaMenuBindings();

    // Bind chat name inputs
    document.getElementById('chat-ai-name')?.addEventListener('input', (e) => updateChatName('ai_name', e.target.value));
    document.getElementById('chat-user-name')?.addEventListener('input', (e) => updateChatName('user_name', e.target.value));

    // Bind explicit toggle (footer)
    document.getElementById('chat-explicit-toggle-footer')?.addEventListener('click', toggleExplicitMode);

    // Bind font controls
    document.getElementById('chat-font-decrease')?.addEventListener('click', () => adjustChatFont(-1));
    document.getElementById('chat-font-increase')?.addEventListener('click', () => adjustChatFont(1));

    // Bind file button — unified export/import dropdown
    const fileBtn = document.getElementById('chat-file-btn');
    const fileMenu = document.getElementById('chat-file-menu');
    if (fileBtn && fileMenu) {
        fileBtn.addEventListener('click', e => {
            e.stopPropagation();
            fileMenu.classList.toggle('hidden');
        });
        fileMenu.querySelectorAll('[data-export-format]').forEach(item => {
            item.addEventListener('click', () => {
                exportChatTab(item.dataset.exportFormat);
                fileMenu.classList.add('hidden');
            });
        });
        document.getElementById('chat-file-import-item')?.addEventListener('click', () => {
            importChatTab();
            fileMenu.classList.add('hidden');
        });
    }
    document.addEventListener('click', e => {
        if (!e.target.closest('#chat-file-btn') && !e.target.closest('#chat-file-menu')) {
            document.getElementById('chat-file-menu')?.classList.add('hidden');
        }
    });

    // Bind chat style cards (event delegation)
    const styleGrid = document.getElementById('chat-style-grid');
    if (styleGrid) {
        styleGrid.addEventListener('click', (e) => {
            const card = e.target.closest('.chat-style-card');
            if (card) selectChatStyle(card.dataset.style);
        });
    }

    // Bind persona panel (formerly "system prompt panel")
    document.getElementById('chat-copy-settings-btn')?.addEventListener('click', showCopySettingsDropdown);
    document.getElementById('chat-explicit-toggle-behavior')?.addEventListener('click', toggleExplicitMode);
    document.getElementById('chat-open-template-mgr')?.addEventListener('click', () => openTemplateManager(activeChatTab()?.active_template_id || null));
    document.getElementById('chat-role-boundary-toggle')?.addEventListener('click', toggleRoleBoundarySection);
    document.getElementById('chat-role-boundary-input')?.addEventListener('input', onRoleBoundaryChange);
    document.getElementById('chat-role-boundary-reset')?.addEventListener('click', resetRoleBoundaryToDefault);
    document.querySelectorAll('.chat-gender-pill').forEach(btn => {
        btn.addEventListener('click', () => onGenderChange(btn.dataset.gender));
    });

    // Bind compact / context settings (in model panel)
    document.getElementById('chat-msg-limit')?.addEventListener('input', (e) => onMessageLimitChange(+e.target.value));
    document.getElementById('chat-auto-compact')?.addEventListener('change', (e) => onAutoCompactChange(e.target.checked));
    document.getElementById('compact-mode-percent')?.addEventListener('click', () => onCompactModeChange('percent'));
    document.getElementById('compact-mode-optimized')?.addEventListener('click', () => onCompactModeChange('optimized'));
    document.getElementById('chat-compact-threshold')?.addEventListener('input', (e) => onCompactThresholdChange(+e.target.value));
    document.getElementById('chat-auto-compact-summarize')?.addEventListener('change', (e) => onAutoCompactSummarizeChange(e.target.checked));

    // Bind model params panel
    document.getElementById('chat-advanced-toggle')?.addEventListener('click', toggleAdvancedParams);
    document.getElementById('chat-reset-params-btn')?.addEventListener('click', resetParamsToDefaults);

    // Bind param sliders
    document.getElementById('param-temperature')?.addEventListener('input', (e) => onParamChange('temperature', +e.target.value));
    document.getElementById('param-top-p')?.addEventListener('input', (e) => onParamChange('top_p', +e.target.value));
    document.getElementById('param-top-k')?.addEventListener('input', (e) => onParamChange('top_k', +e.target.value));
    document.getElementById('param-min-p')?.addEventListener('input', (e) => onParamChange('min_p', +e.target.value));
    document.getElementById('param-repeat-penalty')?.addEventListener('input', (e) => onParamChange('repeat_penalty', +e.target.value));
    document.getElementById('param-max-tokens')?.addEventListener('input', (e) => onParamChange('max_tokens', e.target.value ? +e.target.value : 4096));
    document.getElementById('param-stream-timeout')?.addEventListener('input', (e) => onParamChange('stream_timeout', +e.target.value));

    // Bind enter toggle
    document.getElementById('chat-enter-toggle-input')?.addEventListener('change', (e) => onEnterToggleChange(e.target.checked));

    registerChatViewBindings({
        loadChatNames,
        syncCompactSettingsUI,
        syncMessageLimitInput,
        updateCtxPressureBar,
        refreshChatTelemetry,
        updateParamsDirtyIndicator,
        checkAutoCompact,
        updatePersonaMenuName,
    });
    refreshChatTelemetry();
}

// ── Chat Input Resize Handle ─────────────────────────────────────────────────
let isResizing = false;
let inputRowEl = null;
let textareaEl = null;
let startY = 0;
let startHeight = 0;
const MIN_ROWS = 1;
const MAX_ROWS = 10;

function initChatResizeHandle() {
    const handle = document.getElementById('chat-resize-handle');
    if (!handle) return;

    inputRowEl = document.getElementById('chat-input-row');
    textareaEl = document.getElementById('chat-input');

    if (!inputRowEl || !textareaEl) return;

    handle.addEventListener('mousedown', startResize);
    document.addEventListener('mousemove', doResize);
    document.addEventListener('mouseup', stopResize);
}

function startResize(e) {
    if (!textareaEl) return;
    isResizing = true;
    startY = e.clientY;
    startHeight = textareaEl.getBoundingClientRect().height;
    const handle = document.getElementById('chat-resize-handle');
    if (handle) handle.classList.add('active');
    e.preventDefault();
}

function doResize(e) {
    if (!isResizing || !textareaEl) return;
    const delta = startY - e.clientY;
    const computedStyle = getComputedStyle(textareaEl);
    const minHeight = parseFloat(computedStyle.minHeight) || 42;
    const newHeight = Math.max(minHeight, startHeight + delta);
    const padding = parseFloat(computedStyle.paddingTop) + parseFloat(computedStyle.paddingBottom);
    const border = parseFloat(computedStyle.borderTopWidth) + parseFloat(computedStyle.borderBottomWidth);
    const contentHeight = newHeight - padding - border;
    const lineHeight = parseFloat(computedStyle.lineHeight) || 24;
    const rows = Math.max(1, Math.round(contentHeight / lineHeight));
    textareaEl.style.height = newHeight + 'px';
    textareaEl.rows = Math.max(MIN_ROWS, Math.min(MAX_ROWS, rows));
    updateResizeHandleUI();
}

function stopResize() {
    if (!isResizing) return;
    isResizing = false;
    const handle = document.getElementById('chat-resize-handle');
    if (handle) handle.classList.remove('active');
    saveSettings();
}

function updateResizeHandleUI() {
    const handle = document.getElementById('chat-resize-handle');
    if (handle && textareaEl) {
        const height = textareaEl.getBoundingClientRect().height;
        const max = 200;
        const pct = Math.min(100, (height - 42) / (max - 42) * 100);
        handle.style.setProperty('--resize-pct', pct / 100);
    }
}

export function resetChatInputHeight() {
    if (textareaEl) {
        textareaEl.style.height = '';
        textareaEl.rows = 1;
        updateResizeHandleUI();
        saveSettings();
    }
}

// ── Persona Menu Bindings ───────────────────────────────────────────────────

let personaMenuListEl = null;

export function registerPersonaMenuBindings() {
    const btn = document.getElementById('chat-persona-btn');
    const menu = document.getElementById('chat-persona-menu');
    const list = document.getElementById('chat-persona-menu-list');
    const name = document.getElementById('chat-persona-menu-name');
    const editBtn = document.getElementById('chat-persona-edit-prompt');

    personaMenuListEl = list;

    if (!btn || !menu || !list || !name) return;

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const isVisible = !menu.classList.toggle('hidden');
        if (isVisible) {
            loadPersonaMenuItems();
        }
    });

    editBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        menu.classList.add('hidden');
        const btnBehavior = document.getElementById('btn-behavior');
        if (btnBehavior) btnBehavior.classList.add('active');
        const activeId = activeChatTab()?.active_template_id || null;
        openTemplateManager(activeId);
    });

    document.addEventListener('click', (e) => {
        if (!menu.contains(e.target)) {
            menu.classList.add('hidden');
        }
    });
}

async function loadPersonaMenuItems() {
    if (!personaMenuListEl) return;

    personaMenuListEl.scrollTop = 0;
    setHtml(personaMenuListEl, '<div class="chat-persona-menu-loading">Loading personas...</div>');

    try {
        const personas = await loadTemplates();

        if (personas.length === 0) {
            setHtml(personaMenuListEl, '<div class="chat-persona-menu-loading">No personas found</div>');
            return;
        }

        setHtml(personaMenuListEl, '');

        const tab = activeChatTab();
        const activeTemplateId = tab?.active_template_id || null;

        // Separate into active, user (non-active), and built-in
        const activePersona = activeTemplateId ? personas.find(p => p.id === activeTemplateId) : null;
        const userPersonas = personas.filter(p => !p._isDefault && p.id !== activeTemplateId);
        const builtInPersonas = personas.filter(p => p._isDefault);

        // Show active persona at the top if exists
        if (activePersona) {
            const activeSection = document.createElement('div');
            activeSection.className = 'chat-persona-menu-section';
            activeSection.textContent = 'Active Persona';
            personaMenuListEl.appendChild(activeSection);

            personaMenuListEl.appendChild(createPersonaItem(activePersona, true));
        }

        // Show user personas (edited or new)
        if (userPersonas.length > 0) {
            const userSection = document.createElement('div');
            userSection.className = 'chat-persona-menu-section';
            userSection.textContent = 'Your Personas';
            personaMenuListEl.appendChild(userSection);

            userPersonas.forEach(persona => {
                personaMenuListEl.appendChild(createPersonaItem(persona, false));
            });
        }

        // Show built-in personas
        if (builtInPersonas.length > 0) {
            const builtInSection = document.createElement('div');
            builtInSection.className = 'chat-persona-menu-section';
            builtInSection.textContent = 'Built-in Personas';
            personaMenuListEl.appendChild(builtInSection);

            builtInPersonas.forEach(persona => {
                personaMenuListEl.appendChild(createPersonaItem(persona, false));
            });
        }
    } catch (err) {
        const errorEl = document.createElement('div');
        errorEl.className = 'chat-persona-menu-loading';
        errorEl.textContent = 'Error: ' + err.message;
        personaMenuListEl.appendChild(errorEl);
    }
}

function createPersonaItem(persona, isActive) {
    const item = document.createElement('div');
    item.className = 'chat-persona-menu-item';
    if (isActive) item.classList.add('active');

    const icon = document.createElement('span');
    icon.className = 'chat-persona-menu-item-icon';
    icon.textContent = persona._isDefault ? '🎭' : '✨';

    const content = document.createElement('div');
    content.className = 'chat-persona-menu-item-content';

    const nameEl = document.createElement('div');
    nameEl.className = 'chat-persona-menu-item-name';
    nameEl.textContent = persona.name;

    // Add badge for built-in templates
    if (persona._isDefault && !isActive) {
        const badge = document.createElement('span');
        badge.className = 'chat-persona-menu-item-badge';
        badge.textContent = 'Built-in';
        nameEl.appendChild(badge);
    }

    content.appendChild(nameEl);

    // Show description or first 60 chars of prompt as meta text
    const metaText = persona.description || (persona.prompt ? persona.prompt.substring(0, 60) + '...' : '');
    if (metaText) {
        const meta = document.createElement('div');
        meta.className = 'chat-persona-menu-item-meta';
        meta.textContent = metaText;
        content.appendChild(meta);
    }

    // Add edit button
    const editBtn = document.createElement('button');
    editBtn.className = 'chat-persona-menu-item-edit';
    editBtn.title = 'Edit persona';
    setHtml(editBtn, `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`);
    editBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        document.getElementById('chat-persona-menu').classList.add('hidden');
        openTemplateManager(persona.id);
    });

    item.appendChild(icon);
    item.appendChild(content);
    item.appendChild(editBtn);

    // Click to select (only on the item, not the edit button)
    item.addEventListener('click', () => {
        window.currentPersona = persona;
        document.getElementById('chat-persona-menu-name').textContent = persona.name;
        document.getElementById('chat-persona-menu').classList.add('hidden');
        const tab = activeChatTab();
        if (tab) {
            tab.system_prompt = persona.prompt;
            tab.active_template_id = persona.id;
            tab.updated_at = Date.now();
            scheduleChatPersist?.();
            syncPersonaPanel();
        }
    });

    return item;
}

export function setPersonaMenuActive(personaName) {
    const items = personaMenuListEl?.querySelectorAll('.chat-persona-menu-item');
    items?.forEach(item => {
        if (item.textContent.includes(personaName)) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    const nameEl = document.getElementById('chat-persona-menu-name');
    if (nameEl && personaName) {
        nameEl.textContent = personaName;
    }
}

export function updatePersonaMenuName() {
    const tab = activeChatTab();
    if (!tab) return;
    const nameEl = document.getElementById('chat-persona-menu-name');
    if (!nameEl) return;

    const activeTemplateId = tab.active_template_id;
    if (!activeTemplateId) {
        nameEl.textContent = 'None';
        return;
    }

    // Resolve the template name from the ID
    loadTemplates().then(personas => {
        const found = personas.find(p => p.id === activeTemplateId);
        if (found) {
            nameEl.textContent = found.name;
        } else {
            nameEl.textContent = '(Unknown)';
        }
    });
}

// ── Debug Prompt Modal ─────────────────────────────────────────────────────

const DEBUG_COLORS = [
    'rgba(99, 102, 241, 0.7)',
    'rgba(139, 92, 246, 0.7)',
    'rgba(236, 72, 153, 0.7)',
    'rgba(244, 114, 182, 0.7)',
    'rgba(45, 212, 191, 0.7)',
    'rgba(34, 211, 238, 0.7)',
    'rgba(251, 191, 36, 0.7)',
    'rgba(251, 146, 60, 0.7)',
];

function formatDebugTokens(value) {
    const n = Number(value) || 0;
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
    return `${n}`;
}

function formatDebugLabel(label) {
    return String(label || '—')
        .toLowerCase()
        .split(/[\s_-]+/)
        .filter(Boolean)
        .map(part => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function formatDebugShare(value) {
    const n = Number(value) || 0;
    if (n === 0) return '0%';
    if (n < 1) return `${n.toFixed(1)}%`;
    if (n < 10) return `${n.toFixed(1)}%`;
    return `${Math.round(n)}%`;
}

function getDebugUtilization(total, capacity) {
    return capacity > 0 ? (total / capacity) * 100 : 0;
}

function getDebugPressureState(utilization) {
    if (utilization >= 90) return { label: 'Critical', tone: 'critical' };
    if (utilization >= 75) return { label: 'Hot', tone: 'warning' };
    if (utilization >= 55) return { label: 'Warm', tone: 'active' };
    return { label: 'Healthy', tone: 'calm' };
}

function getDebugSliceKey(label) {
    return String(label || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
}

function formatDebugTimestamp(value) {
    if (!value) return '—';
    try {
        return new Date(value).toLocaleTimeString([], {
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit',
        });
    } catch {
        return '—';
    }
}

async function copyDebugText(text, successMessage) {
    try {
        await navigator.clipboard.writeText(text || '');
        showToast(successMessage, 'success');
    } catch (err) {
        showToast(`Copy failed: ${err.message}`, 'error');
    }
}

function getDebugSegments(data) {
    const total = Number(data.totalTokens) || 1;
    const capacity = Number(data.capacity) || 0;
    const remaining = Math.max(0, capacity - total);
    const segments = [];
    let colorIdx = 0;

    for (const part of data.systemPartsDetailed || []) {
        segments.push({
            key: getDebugSliceKey(part.label),
            label: formatDebugLabel(part.label),
            rawLabel: part.label,
            tokens: Number(part.tokens) || 0,
            color: DEBUG_COLORS[colorIdx % DEBUG_COLORS.length],
            kind: 'system',
            content: part.content || '',
        });
        colorIdx++;
    }

    const historyMessages = data.historyMessagesDetailed || [];
    const conversationPreview = historyMessages
        .slice(-4)
        .map(message => `${message.role.toUpperCase()}: ${message.content}`)
        .join('\n\n');

    segments.push({
        key: 'conversation',
        label: 'Conversation',
        rawLabel: 'Conversation',
        tokens: Number(data.historyTokens) || 0,
        color: DEBUG_COLORS[colorIdx % DEBUG_COLORS.length],
        kind: 'history',
        content: conversationPreview,
    });
    colorIdx++;

    if (remaining > 0) {
        segments.push({
            key: 'remaining',
            label: 'Remaining',
            rawLabel: 'Remaining',
            tokens: remaining,
            color: 'rgba(255,255,255,0.06)',
            kind: 'remaining',
            content: '',
        });
    }

    return segments;
}

function ensureDebugSelection(data) {
    const segments = getDebugSegments(data).filter(seg => seg.kind !== 'remaining');
    if (!segments.length) {
        debugSelectedSliceKey = null;
        return;
    }

    if (!debugSelectedSliceKey || !segments.some(seg => seg.key === debugSelectedSliceKey)) {
        const firstSystem = segments.find(seg => seg.kind === 'system');
        debugSelectedSliceKey = firstSystem?.key || segments[0].key;
    }
}

function openDebugModal() {
    const overlay = document.getElementById('debug-prompt-modal');
    if (!overlay) return;
    debugInspectorView = 'slice';
    document.getElementById('debug-payload-section')?.classList.add('hidden');
    if (debugModalCloseTimer !== null) { clearTimeout(debugModalCloseTimer); debugModalCloseTimer = null; }
    overlay.querySelector('.debug-modal')?.classList.remove('closing');
    overlay.classList.add('active');
    populateDebugModal();
}

let debugModalCloseTimer = null;

function closeDebugModal() {
    const overlay = document.getElementById('debug-prompt-modal');
    if (!overlay) return;
    const modal = overlay.querySelector('.debug-modal');
    if (modal) modal.classList.add('closing');
    // Same stale-timer race as settings.js: without cancelling, reopening inside the
    // animation window let the previous close strip `active` off the reopened overlay.
    if (debugModalCloseTimer !== null) clearTimeout(debugModalCloseTimer);
    debugModalCloseTimer = setTimeout(() => {
        debugModalCloseTimer = null;
        overlay.classList.remove('active');
        if (modal) modal.classList.remove('closing');
    }, 200);
}

function populateDebugModal() {
    const tab = activeChatTab();
    const data = tab?._lastDebugData;
    const emptyState = document.getElementById('debug-empty-state');
    const content = document.getElementById('debug-content');

    if (!data) {
        if (emptyState) emptyState.classList.remove('hidden');
        if (content) content.classList.add('hidden');
        return;
    }

    if (emptyState) emptyState.classList.add('hidden');
    if (content) content.classList.remove('hidden');

    ensureDebugSelection(data);
    populateDebugSummary(data);
    populateCtxBreakdown(data);
    populateTiming(data);
    populateParams(data);
    populateDebugInspector(data);
    populatePayloadJson(data);
}

function populateDebugSummary(data) {
    const total = Number(data.totalTokens) || 0;
    const capacity = Number(data.capacity) || 0;
    const remaining = Math.max(0, capacity - total);
    const utilization = getDebugUtilization(total, capacity);
    const pressure = getDebugPressureState(utilization);
    const systemEntries = Object.entries(data.systemTokens || {});
    const dominantEntry = [
        ...systemEntries.map(([label, tokens]) => ({ label: formatDebugLabel(label), tokens: Number(tokens) || 0 })),
        { label: 'Conversation', tokens: Number(data.historyTokens) || 0 },
    ].sort((a, b) => b.tokens - a.tokens)[0];

    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    setText('debug-stat-utilization', capacity ? `${utilization.toFixed(1)}%` : '—');
    setText('debug-stat-total', formatDebugTokens(total));
    setText('debug-stat-headroom', capacity ? `${formatDebugTokens(remaining)} tok` : '—');
    setText('debug-stat-dominant', dominantEntry?.label || '—');

    const heroChip = document.getElementById('debug-hero-chip');
    if (heroChip) {
        heroChip.textContent = pressure.label;
        heroChip.dataset.tone = pressure.tone;
    }

    const heroText = document.getElementById('debug-hero-text');
    if (heroText) {
        if (!capacity) {
            heroText.textContent = 'Capacity was unavailable for this request, so the debug view is emphasizing composition and timing instead of headroom.';
        } else {
            heroText.textContent = `${formatDebugTokens(remaining)} tokens of headroom remain after the last send. ${dominantEntry?.label || 'Conversation'} is currently the largest slice of the window.`;
        }
    }
}

function populateCtxBreakdown(data) {
    const bar = document.getElementById('debug-ctx-bar');
    const legend = document.getElementById('debug-ctx-legend');
    const summary = document.getElementById('debug-ctx-summary');
    const badge = document.getElementById('debug-ctx-badge');
    if (!bar || !legend || !summary) return;

    const total = Number(data.totalTokens) || 1;
    const capacity = Number(data.capacity) || 0;
    const utilization = getDebugUtilization(total, capacity);
    const pressure = getDebugPressureState(utilization);
    const segments = getDebugSegments(data);

    const denominator = capacity || total;
    const dominantUsed = segments
        .filter(seg => seg.kind !== 'remaining')
        .sort((a, b) => b.tokens - a.tokens)[0];
    const usedTotal = segments
        .filter(seg => seg.kind !== 'remaining')
        .reduce((sum, seg) => sum + (Number(seg.tokens) || 0), 0);
    const historyShare = usedTotal > 0 ? ((Number(data.historyTokens) || 0) / usedTotal) * 100 : 0;
    const systemTotal = Object.values(data.systemTokens || {}).reduce((sum, val) => sum + (Number(val) || 0), 0);
    const systemShare = usedTotal > 0 ? (systemTotal / usedTotal) * 100 : 0;


    setHtml(bar, segments.map(seg => {
        const pct = Math.max(0, Math.round((seg.tokens / denominator) * 100));
        const height = Math.max(28, Math.round((seg.tokens / denominator) * 240));
        const label = escapeHtml(seg.label);
        const activeClass = seg.key === debugSelectedSliceKey ? ' active' : '';
        return `<button type="button" class="debug-ctx-segment debug-ctx-segment-${seg.kind}${activeClass}" data-debug-slice="${escapeHtml(seg.key)}" style="height:${height}px; --debug-segment-color:${seg.color};" title="${label}: ${seg.tokens} tokens (${pct}%)">
            <div class="debug-ctx-segment-sheen"></div>
            <span class="debug-ctx-label">${label}</span>
            <span class="debug-ctx-tokens">${formatDebugTokens(seg.tokens)} tok</span>
        </button>`;
    }).join(''));


    setHtml(legend, segments.map(seg =>
        `<button type="button" class="debug-ctx-legend-item${seg.key === debugSelectedSliceKey ? ' active' : ''}" data-debug-slice="${escapeHtml(seg.key)}">
            <span class="debug-ctx-legend-swatch" style="background:${seg.color}"></span>
            <span class="debug-ctx-legend-copy">
                <strong>${escapeHtml(seg.label)}</strong>
                <span>${formatDebugTokens(seg.tokens)} tok</span>
            </span>
        </button>`
    ).join(''));


    setHtml(summary, `
        <div class="debug-ctx-summary-card" data-tone="${pressure.tone}">
            <span class="debug-ctx-summary-label">Pressure</span>
            <strong>${pressure.label}</strong>
            <p>${capacity ? `${utilization.toFixed(1)}% of the available context is currently occupied.` : 'Capacity is unknown, so utilization is estimated from the captured prompt only.'}</p>
        </div>
        <div class="debug-ctx-summary-card">
            <span class="debug-ctx-summary-label">Dominant Slice</span>
            <strong>${escapeHtml(dominantUsed?.label || '—')}</strong>
            <p>${dominantUsed ? `${formatDebugTokens(dominantUsed.tokens)} tokens, ${Math.round((dominantUsed.tokens / Math.max(usedTotal, 1)) * 100)}% of the used prompt.` : 'No prompt slices were recorded.'}</p>
        </div>
        <div class="debug-ctx-summary-card">
            <span class="debug-ctx-summary-label">Composition</span>
            <strong>${formatDebugShare(historyShare)} history / ${formatDebugShare(systemShare)} system</strong>
            <p>Conversation history and system scaffolding are balanced across the captured request.</p>
        </div>
    `);

    if (badge) {
        badge.textContent = pressure.label;
        badge.dataset.tone = pressure.tone;
    }
}

function populateTiming(data) {
    const grid = document.getElementById('debug-timing-grid');
    if (!grid) return;

    const cells = [];
    if (data.promptMs != null) {
        cells.push({ label: 'Prompt', value: Math.round(data.promptMs), unit: 'ms' });
    }
    if (data.genMs != null) {
        cells.push({ label: 'Generation', value: Math.round(data.genMs), unit: 'ms' });
    }
    if (data.promptMs && data.genMs) {
        const totalSec = ((data.promptMs + data.genMs) / 1000);
        cells.push({ label: 'Total', value: totalSec.toFixed(1), unit: 's' });
    }
    if (data.totalTokens && data.promptMs && data.genMs) {
        const throughput = data.totalTokens / ((data.promptMs + data.genMs) / 1000);
        cells.push({ label: 'Observed Throughput', value: throughput.toFixed(1), unit: 'tok/s' });
    }
    if (data.draftN != null && data.draftNAccepted != null && data.draftN > 0) {
        const pct = Math.round((data.draftNAccepted / data.draftN) * 100);
        cells.push({ label: 'Draft Accepted', value: data.draftNAccepted + '/' + data.draftN, unit: '(' + pct + '%)' });
    }
    if (data.modelParams?.max_tokens) {
        cells.push({ label: 'Max Tokens', value: data.modelParams.max_tokens, unit: '' });
    }


    setHtml(grid, cells.map(c =>
        `<div class="debug-timing-cell"><div class="debug-timing-cell-label">${c.label}</div><div class="debug-timing-cell-value">${c.value}<span class="debug-timing-cell-unit">${c.unit}</span></div></div>`
    ).join(''));
}

function populateParams(data) {
    const grid = document.getElementById('debug-params-grid');
    if (!grid || !data.modelParams) return;

    const params = data.modelParams;
    const entries = [
        ['Temperature', params.temperature ?? '—'],
        ['Top P', params.top_p ?? '—'],
        ['Top K', params.top_k ?? '—'],
        ['Min P', params.min_p ?? '—'],
        ['Repeat Penalty', params.repeat_penalty ?? '—'],
        ['Max Tokens', params.max_tokens ?? '—'],
    ];


    setHtml(grid, entries.map(([label, value]) =>
        `<div class="debug-params-cell"><span class="debug-params-cell-label">${label}</span><span class="debug-params-cell-value">${value}</span></div>`
    ).join(''));
}

function buildConversationInspector(data) {
    const rows = [];
    const historyMessages = data.historyMessagesDetailed || [];
    const previewMessages = historyMessages.slice(-4);
    rows.push('<div class="debug-inspector-note">Conversation history is intentionally summarized here so the modal stays readable. Use the payload viewer if you need the full message array.</div>');
    rows.push(`<div class="debug-inspector-subsection"><span class="debug-inspector-subtitle">Recent Messages</span><div class="debug-inspector-message-list">`);
    for (const message of previewMessages) {
        rows.push(
            `<div class="debug-inspector-message">
                <div class="debug-inspector-message-meta">${escapeHtml(message.role)} • ${formatDebugTokens(message.tokens)} tok</div>
                <pre class="debug-inspector-pre">${escapeHtml(message.content)}</pre>
            </div>`
        );
    }
    if (data.finalUserPrompt) {
        rows.push(
            `<div class="debug-inspector-message">
                <div class="debug-inspector-message-meta">current user prompt • ${formatDebugTokens(Math.max(1, Math.round((data.finalUserPrompt || '').length / 4)))} tok</div>
                <pre class="debug-inspector-pre">${escapeHtml(data.finalUserPrompt)}</pre>
            </div>`
        );
    }
    rows.push('</div></div>');
    return rows.join('');
}

function populateDebugInspector(data) {
    const titleEl = document.getElementById('debug-inspector-title');
    const metaEl = document.getElementById('debug-inspector-meta');
    const bodyEl = document.getElementById('debug-inspector-body');
    const sliceBtn = document.getElementById('debug-view-slice');
    const finalBtn = document.getElementById('debug-view-final');
    if (!titleEl || !metaEl || !bodyEl || !sliceBtn || !finalBtn) return;

    sliceBtn.classList.toggle('active', debugInspectorView === 'slice');
    finalBtn.classList.toggle('active', debugInspectorView === 'final');

    const segments = getDebugSegments(data);
    const selected = segments.find(seg => seg.key === debugSelectedSliceKey)
        || segments.find(seg => seg.kind === 'system')
        || segments[0];

    if (debugInspectorView === 'final') {
        titleEl.textContent = 'Final Prompt';

        setHtml(metaEl, `
            <span class="debug-inspector-chip">system ${formatDebugTokens(data.totalSystemTokens || 0)} tok</span>
            <span class="debug-inspector-chip">conversation ${formatDebugTokens(data.historyTokens || 0)} tok</span>
            <span class="debug-inspector-chip">sent ${escapeHtml(formatDebugTimestamp(data.sentAt))}</span>
        `);
        setHtml(bodyEl, `
            <div class="debug-inspector-subsection">
                <span class="debug-inspector-subtitle">Final System Message</span>
                <pre class="debug-inspector-pre">${escapeHtml(data.finalSystemPrompt || '')}</pre>
            </div>
            <div class="debug-inspector-subsection">
                <span class="debug-inspector-subtitle">Last User Message</span>
                <pre class="debug-inspector-pre">${escapeHtml(data.finalUserPrompt || '(none)')}</pre>
            </div>
        `);
        return;
    }

    titleEl.textContent = selected?.label || 'Prompt Slice';

    setHtml(metaEl, selected ? `
        <span class="debug-inspector-chip">${escapeHtml(selected.kind === 'system' ? 'system slice' : selected.kind)}</span>
        <span class="debug-inspector-chip">${formatDebugTokens(selected.tokens)} tok</span>
        <span class="debug-inspector-chip">${selected.kind === 'remaining' ? 'unused capacity' : escapeHtml(formatDebugTimestamp(data.sentAt))}</span>
    ` : '');

    if (!selected) {
        setHtml(bodyEl, '<div class="debug-inspector-note">No prompt slices were captured for this request.</div>');
        return;
    }

    if (selected.kind === 'remaining') {
        setHtml(bodyEl, '<div class="debug-inspector-note">Remaining capacity is not text that was sent. It represents unused headroom in the current context window.</div>');
        return;
    }

    if (selected.kind === 'history') {

        setHtml(bodyEl, buildConversationInspector(data));
        return;
    }

    setHtml(bodyEl, `<pre class="debug-inspector-pre">${escapeHtml(selected.content || '')}</pre>`);
}

function populatePayloadJson(data) {
    const pre = document.getElementById('debug-payload-json');
    if (!pre) return;
    pre.textContent = JSON.stringify(data.requestPayload || {}, null, 2);
}

function initDebugHandlers() {
    const btn = document.getElementById('btn-debug-prompt');
    const closeBtn = document.getElementById('debug-modal-close');
    const overlay = document.getElementById('debug-prompt-modal');
    const ctxBar = document.getElementById('debug-ctx-bar');
    const ctxLegend = document.getElementById('debug-ctx-legend');
    const sliceViewBtn = document.getElementById('debug-view-slice');
    const finalViewBtn = document.getElementById('debug-view-final');
    const copySliceBtn = document.getElementById('debug-copy-slice-btn');
    const copyFinalBtn = document.getElementById('debug-copy-final-btn');
    const viewPayloadBtn = document.getElementById('debug-view-payload-btn');
    const copyPayloadBtn = document.getElementById('debug-copy-payload-btn');
    const hidePayloadBtn = document.getElementById('debug-hide-payload-btn');

    // Dropdown toggle
    const dropdownBtn = document.getElementById('btn-debug-dropdown');
    const dropdownMenu = document.getElementById('debug-dropdown-menu');

    dropdownBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdownMenu?.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
        if (dropdownMenu && !dropdownBtn?.contains(e.target)) {
            dropdownMenu.classList.remove('open');
        }
    });

    // Dropdown menu item handlers
    document.getElementById('btn-debug-prompt')?.addEventListener('click', () => {
        dropdownMenu?.classList.remove('open');
        openDebugModal();
    });

    document.getElementById('btn-db-admin')?.addEventListener('click', () => {
        dropdownMenu?.classList.remove('open');
        openDbAdminModal();
    });

    btn?.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdownMenu?.classList.remove('open');
        openDebugModal();
    });
    closeBtn?.addEventListener('click', closeDebugModal);
    overlay?.addEventListener('click', (e) => {
        if (e.target === overlay) closeDebugModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay?.classList.contains('active')) {
            closeDebugModal();
        }
    });

    function openDbAdminModal() {
        const dbOverlay = document.getElementById('db-admin-modal');
        if (dbOverlay) {
            dbOverlay.classList.add('active');
        }
    }

    const onSliceSelect = (e) => {
        const target = e.target.closest('[data-debug-slice]');
        if (!target) return;
        debugSelectedSliceKey = target.dataset.debugSlice;
        debugInspectorView = 'slice';
        const data = activeChatTab()?._lastDebugData;
        if (!data) return;
        populateCtxBreakdown(data);
        populateDebugInspector(data);
    };

    ctxBar?.addEventListener('click', onSliceSelect);
    ctxLegend?.addEventListener('click', onSliceSelect);

    sliceViewBtn?.addEventListener('click', () => {
        debugInspectorView = 'slice';
        const data = activeChatTab()?._lastDebugData;
        if (!data) return;
        populateCtxBreakdown(data);
        populateDebugInspector(data);
    });

    finalViewBtn?.addEventListener('click', () => {
        debugInspectorView = 'final';
        const data = activeChatTab()?._lastDebugData;
        if (!data) return;
        populateCtxBreakdown(data);
        populateDebugInspector(data);
    });

    copySliceBtn?.addEventListener('click', () => {
        const data = activeChatTab()?._lastDebugData;
        if (!data) return;
        const segments = getDebugSegments(data);
        const selected = segments.find(seg => seg.key === debugSelectedSliceKey) || segments[0];
        if (!selected) return;
        if (selected.kind === 'history') {
            copyDebugText((data.historyMessagesDetailed || []).map(message => `${message.role}: ${message.content}`).join('\n\n'), 'Conversation payload copied');
            return;
        }
        if (selected.kind === 'remaining') {
            showToast('Remaining capacity is not prompt text', 'info');
            return;
        }
        copyDebugText(selected.content || '', `${selected.label} copied`);
    });

    copyFinalBtn?.addEventListener('click', () => {
        const data = activeChatTab()?._lastDebugData;
        if (!data) return;
        const parts = [];
        if (data.finalSystemPrompt) {
            parts.push(`SYSTEM\n${data.finalSystemPrompt}`);
        }
        if (data.finalUserPrompt) {
            parts.push(`USER\n${data.finalUserPrompt}`);
        }
        copyDebugText(parts.join('\n\n'), 'Final prompt copied');
    });

    copyPayloadBtn?.addEventListener('click', () => {
        const data = activeChatTab()?._lastDebugData;
        if (!data) return;
        copyDebugText(JSON.stringify(data.requestPayload || {}, null, 2), 'Payload JSON copied');
    });

    viewPayloadBtn?.addEventListener('click', () => {
        document.getElementById('debug-payload-section')?.classList.remove('hidden');
    });

    hidePayloadBtn?.addEventListener('click', () => {
        document.getElementById('debug-payload-section')?.classList.add('hidden');
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDebugHandlers);
} else {
    initDebugHandlers();
}
