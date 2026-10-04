// ── Chat State & Persistence ─────────────────────────────────────────────────
// Tab collection, active tab, busy flags, persistence scheduling, tab CRUD.

import { chat, settingsState } from '../core/app-state.js';
import { refreshTopCockpit } from './nav.js';
import { showToast, showToastWithActions } from './toast.js';
import Router from './router.js';

const CHAT_TABS_PERSIST_DEBOUNCE_MS = 500;
const TRASH_AUTO_PURGE_MS = 24 * 60 * 60 * 1000; // 24 hours
const TRASH_PURGE_CHECK_INTERVAL_MS = 60 * 60 * 1000; // check every hour
const chatViewBindings = {
    renderChatTabs: null,
    renderChatMessages: null,
    renderChatSessionsSidebar: null,
    loadChatNames: null,
    updateExplicitToggleUI: null,
    updateParamsDirtyIndicator: null,
    syncMessageLimitInput: null,
    syncCompactSettingsUI: null,
    updateCtxPressureBar: null,
    refreshChatTelemetry: null,
    updateChatTabBadge: null,
    checkAutoCompact: null,
    refreshChatHistoryQA: null,
};

export function registerChatViewBindings(bindings) {
    Object.assign(chatViewBindings, bindings);
}

export function getChatViewBindings() {
    return chatViewBindings;
}

// ── Tab Accessors ──────────────────────────────────────────────────────────────

export function activeChatTab() {
    return chat.tabs.find(t => t.id === chat.activeTabId) ?? null;
}

// ── Tab Creation ───────────────────────────────────────────────────────────────

export function newChatTab(name = 'New Chat') {
    return {
        id: crypto.randomUUID(),
        name,
        system_prompt: 'You are {{char}}, a helpful, concise assistant. You are talking to {{user}}. Provide clear, accurate answers.',
        active_template_id: '',
        ai_name: '',
        user_name: '',
        explicit_level: 0,
        auto_compact: true,
        auto_compact_summarize: true,
        messages: [],
        total_input_tokens: 0,
        total_output_tokens: 0,
        last_ctx_pct: 0,
        model_params: {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.01,
            repeat_penalty: 1.0,
            max_tokens: 32768,
            stream_timeout: 120,
        },
        context_notes: [],
        context_custom_sections: [],
        compact_mode: 'percent',
        compact_threshold: 0.8,
        tab_order: 0,
        sidebar_width: 280,
        quick_guide_draft: '',
        quick_guide_active: '',
        quick_guide_pending: '',
        armed_story_beats: [],
        created_at: Date.now(),
        updated_at: Date.now(),
        pinned: false,
        visibility: 'active',
        composer_draft: '',
    };
}

function normalizeChatTab(tab) {
    const messages = tab.messages || [];
    const total_input_tokens = messages.reduce((sum, m) => sum + (m.input_tokens || 0), 0);
    const total_output_tokens = messages.reduce((sum, m) => sum + (m.output_tokens || 0), 0);
    let explicit_level = tab.explicit_level ?? tab.explicitLevel ?? 0;
    if (tab.explicit_mode !== undefined && tab.explicit_level === undefined) {
        explicit_level = tab.explicit_mode ? 1 : 0;
    }
    return {
        ...tab,
        explicit_level: explicit_level,
        active_template_id: tab.active_template_id ?? tab.activeTemplateId ?? '',
        auto_compact: tab.auto_compact ?? true,
        auto_compact_summarize: tab.auto_compact_summarize ?? true,
        last_ctx_pct: tab.last_ctx_pct ?? tab.lastCtxPct ?? 0,
        total_input_tokens: tab.total_input_tokens ?? tab.totalInputTokens ?? total_input_tokens,
        total_output_tokens: tab.total_output_tokens ?? tab.totalOutputTokens ?? total_output_tokens,
        context_notes: tab.context_notes ?? [],
        context_custom_sections: tab.context_custom_sections ?? [],
        sidebar_width: tab.sidebar_width || tab.sidebarWidth || 280,
        quick_guide_draft: tab.quick_guide_draft ?? '',
        quick_guide_active: '',
        quick_guide_pending: '',
        armed_story_beats: tab.armed_story_beats ?? [],
        pinned: tab.pinned ?? false,
        visibility: tab.visibility || 'active',
        composer_draft: tab.composer_draft ?? '',
    };
}

function sanitizeThinkingContent(messages) {
    if (settingsState.persist_thinking_content) return messages || [];
    return (messages || []).map(message => {
        if (!message?.thinking_content) return message;
        return { ...message, thinking_content: undefined };
    });
}

// ── Tab Initialization ────────────────────────────────────────────────────────

export async function initChatTabs() {
    try {
        const resp = await fetch('/api/chat/tabs?visibility=all', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (!resp.ok) {
            let detail = `Chat tabs request failed (${resp.status})`;
            if (resp.status === 401 || resp.status === 403) {
                detail = 'This browser could not authenticate to load saved chats.';
            }
            throw new Error(detail);
        }
        const metas = await resp.json();
        if (!Array.isArray(metas)) {
            throw new Error('Chat tabs response was invalid.');
        }
        if (metas.length) {
            chat.tabs = metas.map(meta => ({
                ...meta,
                messages: null,
                _loaded: false,
            }));
        } else {
            await addChatTab();
            chatViewBindings.renderChatSessionsSidebar?.();
            chatViewBindings.renderChatMessages?.();
            chatViewBindings.loadChatNames?.();
            chatViewBindings.updateExplicitToggleUI?.();
            chatViewBindings.updateParamsDirtyIndicator?.();
            chatViewBindings.syncMessageLimitInput?.();
            chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
            chatViewBindings.refreshChatTelemetry?.();
            chatViewBindings.updatePersonaMenuName?.();
            chatViewBindings.refreshChatHistoryQA?.();
            refreshTopCockpit();
            return;
        }
    } catch (e) {
        console.error('initChatTabs failed:', e);
        chat.tabs = [];
        chat.activeTabId = null;
        chatViewBindings.renderChatTabs?.();
        chatViewBindings.renderChatSessionsSidebar?.();
        chatViewBindings.renderChatMessages?.();
        chatViewBindings.loadChatNames?.();
        chatViewBindings.updateExplicitToggleUI?.();
        chatViewBindings.updateParamsDirtyIndicator?.();
        chatViewBindings.syncMessageLimitInput?.();
        chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
        chatViewBindings.refreshChatTelemetry?.();
        chatViewBindings.updatePersonaMenuName?.();
        chatViewBindings.refreshChatHistoryQA?.();
        refreshTopCockpit();
        showToast('Could not load chats', 'error', e?.message || 'Saved chats could not be loaded.');
        return;
    }
    chat.activeTabId = chat.tabs[0].id;

    await _loadTabMessages(chat.activeTabId);

    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    chatViewBindings.renderChatMessages?.();
    chatViewBindings.loadChatNames?.();
    chatViewBindings.updateExplicitToggleUI?.();
    chatViewBindings.updateParamsDirtyIndicator?.();
    chatViewBindings.syncMessageLimitInput?.();
    chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
    chatViewBindings.refreshChatTelemetry?.();
    chatViewBindings.updatePersonaMenuName?.();
    chatViewBindings.refreshChatHistoryQA?.();
    refreshTopCockpit();

    if (typeof window.onChatTabsLoaded === 'function') {
        window.onChatTabsLoaded();
    }

    setTimeout(() => {
        window.dispatchEvent(new CustomEvent('activeTabChanged', { detail: { tabId: chat.activeTabId } }));
    }, 0);
}

// ── Lazy Tab Loading ───────────────────────────────────────────────────────────

async function _loadTabMessages(id) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab || tab._loaded) return;
    try {
        const resp = await fetch(`/api/chat/tabs/${id}`, {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const full = await resp.json();
        if (Array.isArray(full?.messages)) {
            full.messages = sanitizeThinkingContent(full.messages);
        }
        Object.assign(tab, full);
        tab._loaded = true;
    } catch (e) {
        console.error(`_loadTabMessages failed for ${id}:`, e);
    }
}

// ── Tab CRUD ───────────────────────────────────────────────────────────────────

export async function addChatTab() {
    const tab = newChatTab(`Chat ${chat.tabs.length + 1}`);
    try {
        const resp = await fetch('/api/chat/tabs', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify(tab),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const created = await resp.json();
        Object.assign(tab, created);
        tab._loaded = true;
    } catch (e) {
        console.error('addChatTab failed:', e);
        // If creation failed, do not add the tab to avoid orphaned tabs.
        return;
    }
    chat.tabs.push(tab);
    await switchChatTab(tab.id);
    chatViewBindings.renderChatSessionsSidebar?.();
}

export async function closeChatTab(id) {
    const tabIdx = chat.tabs.findIndex(t => t.id === id);
    if (tabIdx === -1) return;

    const [tab] = chat.tabs.splice(tabIdx, 1);
    chat.tabTrash.push({ tab, trashedAt: Date.now() });

    if (chat.activeTabId === id) {
        if (chat.tabs.length) {
            chat.activeTabId = chat.tabs[chat.tabs.length - 1].id;
            await _loadTabMessages(chat.activeTabId);
        } else {
            chat.activeTabId = null;
        }
    }

    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    chatViewBindings.renderChatMessages?.();
    chatViewBindings.loadChatNames?.();
    chatViewBindings.updateExplicitToggleUI?.();
    chatViewBindings.updateParamsDirtyIndicator?.();
    chatViewBindings.syncMessageLimitInput?.();
    chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
    chatViewBindings.refreshChatTelemetry?.();
    chatViewBindings.updatePersonaMenuName?.();
    chatViewBindings.refreshChatHistoryQA?.();
    refreshTopCockpit();

    try {
        const resp = await fetch(`/api/chat/tabs/${id}`, {
            method: 'DELETE',
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        const body = await resp.json().catch(() => null);
        if (!resp.ok || body?.ok === false) {
            throw new Error(body?.error || `Delete failed (${resp.status})`);
        }
    } catch (e) {
        restoreTabFromTrash(id);
        showToast('Could not delete tab', 'error', e?.message || 'The tab was restored because deletion failed.');
        return;
    }

    showToastWithActions('Tab deleted', 'info', '', [
        {
            id: 'undo',
            label: 'Undo',
            primary: true,
            handler: () => restoreTabFromTrash(id),
        },
    ]);
}

export async function deleteManyChatTabs(ids) {
    if (!ids || ids.length === 0) return;

    const toDelete = ids.filter(id => chat.tabs.some(t => t.id === id));
    if (toDelete.length === 0) return;

    // Move each tab to trash.
    const deletedIds = [];
    for (const id of toDelete) {
        const idx = chat.tabs.findIndex(t => t.id === id);
        if (idx === -1) continue;
        const [tab] = chat.tabs.splice(idx, 1);
        chat.tabTrash.push({ tab, trashedAt: Date.now() });
        deletedIds.push(id);

        if (chat.activeTabId === id) {
            if (chat.tabs.length) {
                chat.activeTabId = chat.tabs[chat.tabs.length - 1].id;
                await _loadTabMessages(chat.activeTabId);
            } else {
                chat.activeTabId = null;
            }
        }
    }

    if (deletedIds.length === 0) return;

    // Fire API deletes in parallel.
    const failures = await Promise.all(
        deletedIds.map(async (id) => {
            try {
                const resp = await fetch(`/api/chat/tabs/${id}`, {
                    method: 'DELETE',
                    headers: window.authHeaders ? window.authHeaders() : {},
                });
                const body = await resp.json().catch(() => null);
                if (!resp.ok || body?.ok === false) {
                    throw new Error(body?.error || `Delete failed (${resp.status})`);
                }
            } catch (e) {
                console.error('deleteManyChatTabs failed for', id, e);
                return id;
            }
        })
    ).then(arr => arr.filter(Boolean));

    // Restore any failed deletions from trash.
    for (const id of failures) {
        restoreTabFromTrash(id);
    }

    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    chatViewBindings.renderChatMessages?.();
    chatViewBindings.loadChatNames?.();
    chatViewBindings.updateExplicitToggleUI?.();
    chatViewBindings.syncMessageLimitInput?.();
    chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
    chatViewBindings.refreshChatTelemetry?.();
    chatViewBindings.updatePersonaMenuName?.();
    chatViewBindings.refreshChatHistoryQA?.();
    refreshTopCockpit();

    showToastWithActions(`${deletedIds.length} chat(s) deleted`, 'info', '', [
        {
            id: 'undo',
            label: 'Undo',
            primary: true,
            handler: () => {
                for (const id of deletedIds) {
                    restoreTabFromTrash(id);
                }
            },
        },
    ]);
}

export function archiveManyChatTabs(ids) {
    if (!ids || ids.length === 0) return;

    const archived = [];
    for (const id of ids) {
        const tab = chat.tabs.find(t => t.id === id);
        if (!tab || tab.visibility === 'archived') continue;
        const prevVisibility = tab.visibility;
        tab.visibility = 'archived';
        archived.push({ id, prevVisibility });

        if (chat.activeTabId === id) {
            _selectFallbackTab(id);
        }

        const headers = window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' };
        fetch(`/api/chat/tabs/${id}/archive`, {
            method: 'POST',
            headers,
        }).catch(e => {
            const entry = archived.find(a => a.id === id);
            if (entry) {
                tab.visibility = entry.prevVisibility;
            }
            console.error('archiveManyChatTabs failed for', id, e);
        });
    }

    if (archived.length > 0) {
        chatViewBindings.renderChatTabs?.();
        chatViewBindings.renderChatSessionsSidebar?.();
        showToastWithActions(`${archived.length} chat(s) archived`, 'info', '', [
            {
                id: 'undo',
                label: 'Undo',
                primary: true,
                handler: () => {
                    for (const { id, prevVisibility } of archived) {
                        const tab = chat.tabs.find(t => t.id === id);
                        if (tab && tab.visibility === 'archived') {
                            tab.visibility = prevVisibility;
                        }
                    }
                    chatViewBindings.renderChatTabs?.();
                    chatViewBindings.renderChatSessionsSidebar?.();
                },
            },
        ]);
    }
}

export function restoreTabFromTrash(id) {
    const trashIdx = chat.tabTrash.findIndex(t => t.tab.id === id);
    if (trashIdx === -1) return;

    const [trashEntry] = chat.tabTrash.splice(trashIdx, 1);
    chat.tabs.push(normalizeChatTab(trashEntry.tab));
    chat.activeTabId = trashEntry.tab.id;

    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    chatViewBindings.renderChatMessages?.();
    chatViewBindings.loadChatNames?.();
    chatViewBindings.updateExplicitToggleUI?.();
    chatViewBindings.syncMessageLimitInput?.();
    chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
    chatViewBindings.refreshChatTelemetry?.();
    chatViewBindings.updatePersonaMenuName?.();
    chatViewBindings.refreshChatHistoryQA?.();
    refreshTopCockpit();
    scheduleChatPersist(normalizeChatTab(trashEntry.tab));
}

export async function switchChatTab(id) {
    if (chat.busy) return;
    const targetTab = chat.tabs.find(t => t.id === id);
    if (targetTab && targetTab.visibility !== 'active') {
        showToast('Chat not visible', 'info', '', []);
        return;
    }
    chat.activeTabId = id;

    // Keep the URL in sync with the active session so it's bookmarkable and
    // survives reloads, regardless of how the switch was triggered (new chat,
    // closing the active tab, duplicate, search, etc.). Only rewrite when already
    // on a chat route: this must not hijack the initial load (e.g. restoring the
    // last tab while the user is on '/'). Uses Router to centralize history
    // policy; updateUrlWithoutDispatch avoids recursing through /chat/:id.
    if (location.pathname === '/chat' || location.pathname.startsWith('/chat/')) {
        const target = '/chat/' + encodeURIComponent(id);
        if (location.pathname !== target) {
            Router.updateUrlWithoutDispatch(target);
        }
    }

    await _loadTabMessages(id);
    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    chatViewBindings.renderChatMessages?.();
    chatViewBindings.loadChatNames?.();
    chatViewBindings.updateExplicitToggleUI?.();
    chatViewBindings.syncMessageLimitInput?.();
    chatViewBindings.syncCompactSettingsUI?.(activeChatTab());
    chatViewBindings.updateCtxPressureBar?.(0);
    chatViewBindings.refreshChatTelemetry?.();
    chatViewBindings.updatePersonaMenuName?.();
    chatViewBindings.refreshChatHistoryQA?.();
    refreshTopCockpit();

    const input = document.getElementById('chat-input');
    if (input && targetTab?.composer_draft) {
        input.value = targetTab.composer_draft;
        autoResizeChatInput();
    } else if (input) {
        input.value = '';
    }

    window.dispatchEvent(new CustomEvent('activeTabChanged', {
        detail: { tabId: id },
    }));
}

export function renameChatTab(id, newName) {
    const tab = chat.tabs.find(t => t.id === id);
    if (tab) {
        tab.name = newName.trim() || tab.name;
        chatViewBindings.renderChatTabs?.();
        chatViewBindings.renderChatSessionsSidebar?.();
        scheduleChatPersist(tab);
    }
}

export function togglePinTab(id) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab) return;
    tab.pinned = !tab.pinned;
    const pinned = chat.tabs.filter(t => t.pinned);
    const unpinned = chat.tabs.filter(t => !t.pinned);
    chat.tabs = [...pinned, ...unpinned];
    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    scheduleChatPersist(tab);
    persistTabOrder();
}

export async function duplicateChatTab(id) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab) return null;
    const copy = normalizeChatTab({
        ...tab,
        id: crypto.randomUUID(),
        name: `${tab.name} (copy)`,
        messages: [...(tab.messages || [])],
        created_at: Date.now(),
        updated_at: Date.now(),
    });
    chat.tabs.push(copy);
    await switchChatTab(copy.id);
    chatViewBindings.renderChatSessionsSidebar?.();
    scheduleChatPersist(copy);
    return copy;
}

// ── Visibility Actions ────────────────────────────────────────────────────────

function _selectFallbackTab(leavingId) {
    const activeTabs = chat.tabs.filter(t => t.visibility === 'active');
    if (activeTabs.length) {
        const idx = activeTabs.findIndex(t => t.id === leavingId);
        const fallbackIdx = idx >= 0 ? 0 : activeTabs.length - 1;
        switchChatTab(activeTabs[fallbackIdx].id);
    } else {
        chat.activeTabId = null;
    }
}

export function archiveChatTab(id) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab) return;
    const prevVisibility = tab.visibility;
    tab.visibility = 'archived';
    if (chat.activeTabId === id) {
        _selectFallbackTab(id);
    }
    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };
    fetch(`/api/chat/tabs/${id}/archive`, {
        method: 'POST',
        headers,
    }).catch(e => {
        tab.visibility = prevVisibility;
        chatViewBindings.renderChatTabs?.();
        chatViewBindings.renderChatSessionsSidebar?.();
        console.error('archiveChatTab failed:', e);
    });
    showToastWithActions('Chat archived', 'info', '', [{
        id: 'undo',
        label: 'Undo',
        primary: true,
        handler: () => restoreChatTab(id),
    }]);
}

export function hideChatTab(id) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab) return;
    const prevVisibility = tab.visibility;
    tab.visibility = 'hidden';
    if (chat.activeTabId === id) {
        _selectFallbackTab(id);
    }
    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };
    fetch(`/api/chat/tabs/${id}/hide`, {
        method: 'POST',
        headers,
    }).catch(e => {
        tab.visibility = prevVisibility;
        chatViewBindings.renderChatTabs?.();
        chatViewBindings.renderChatSessionsSidebar?.();
        console.error('hideChatTab failed:', e);
    });
    showToastWithActions('Chat hidden', 'info', '', [{
        id: 'undo',
        label: 'Undo',
        primary: true,
        handler: () => restoreChatTab(id),
    }]);
}

export function restoreChatTab(id) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab) return;
    tab.visibility = 'active';
    switchChatTab(id);
    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };
    fetch(`/api/chat/tabs/${id}/restore`, {
        method: 'POST',
        headers,
    }).catch(e => console.error('restoreChatTab failed:', e));
    showToast('Chat restored', 'success', '', []);
}

export async function setChatTabVisibility(id, visibility) {
    const tab = chat.tabs.find(t => t.id === id);
    if (!tab) return;
    const prevVisibility = tab.visibility;
    tab.visibility = visibility;
    chatViewBindings.renderChatTabs?.();
    chatViewBindings.renderChatSessionsSidebar?.();
    const headers = window.authHeaders
        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
        : { 'Content-Type': 'application/json' };
    try {
        const resp = await fetch(`/api/chat/tabs/${id}/meta`, {
            method: 'PATCH',
            headers,
            body: JSON.stringify({ visibility }),
        });
        if (!resp.ok) {
            tab.visibility = prevVisibility;
            chatViewBindings.renderChatTabs?.();
            chatViewBindings.renderChatSessionsSidebar?.();
        }
    } catch (e) {
        tab.visibility = prevVisibility;
        chatViewBindings.renderChatTabs?.();
        chatViewBindings.renderChatSessionsSidebar?.();
        console.error('setChatTabVisibility failed:', e);
    }
}

// ── Tab Field Updates ─────────────────────────────────────────────────────────

export function substituteNames(prompt, aiName, userName, gender) {
    if (!prompt) return prompt;
    let p = prompt;
    p = p.replace(/\{\{char\}\}/gi, aiName || 'AI');
    p = p.replace(/\{\{user\}\}/gi, userName || 'User');
    p = p.replace(/\{\{gender\}\}/gi, gender || 'neutral');
    return p;
}

function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function normalizeGeneratedMessageContent(content, tab, role = 'assistant') {
    if (!content) return content;

    const names = role === 'user'
        ? [tab?.user_name, 'You', 'User']
        : [tab?.ai_name, 'AI', 'Assistant'];
    const labels = names
        .map(name => (name || '').trim())
        .filter((name, index, arr) => name && arr.indexOf(name) === index)
        .map(escapeRegExp);

    let normalized = content;
    if (labels.length > 0) {
        const labelPattern = new RegExp(
            `^\\s*(?:\\*\\*)?(?:${labels.join('|')})(?:\\*\\*)?\\s*[:：\\-–—]\\s*`,
            'i'
        );
        normalized = normalized.replace(labelPattern, '');
    }

    return normalized.replace(/^\s*[:：]\s*(?:\n+)?/, '');
}

export function getDefaultRoleBoundaryText(tab) {
    const assistantName = (tab?.ai_name || '{{char}}').trim();
    const userName = (tab?.user_name || '{{user}}').trim();
    return `You are ${assistantName}. By default, write only ${assistantName}'s reply. Do not speak as, write dialogue for, narrate actions for, or decide choices/thoughts for ${userName} unless the latest user instruction explicitly asks you to control or write both sides.\n\nWhen the scene introduces supporting or secondary characters, you may voice those characters as needed to serve the narrative — but do not speak for or make decisions on behalf of ${userName}.`;
}

export function updateChatName(field, value) {
    const tab = activeChatTab();
    if (tab) {
        tab[field] = value.trim();
        scheduleChatPersist(tab);
        chatViewBindings.renderChatMessages?.();
    }
}

// ── Persistence ────────────────────────────────────────────────────────────────

export function normalizeTabForSave(tab) {
    const t = { ...tab };
    delete t.explicit_mode;
    delete t.explicitLevel; // legacy camelCase
    delete t._quickGuideInstruction;
    delete t.quick_guide_pending;
    t.messages = (t.messages || []).map(m => {
        const msg = { ...m };
        delete msg.cumulativeInputTokens;
        delete msg.cumulativeOutputTokens;
        if (!settingsState.persist_thinking_content) {
            delete msg.thinking_content;
        }
        return msg;
    });
    return t;
}

function stripThinkingFromLoadedTabs() {
    let changed = false;
    for (const tab of chat.tabs || []) {
        if (!Array.isArray(tab?.messages)) continue;
        let tabChanged = false;
        tab.messages = tab.messages.map(message => {
            if (!message?.thinking_content) return message;
            tabChanged = true;
            return { ...message, thinking_content: undefined };
        });
        if (tabChanged) {
            changed = true;
            scheduleChatPersist(tab);
        }
    }
    if (changed) {
        chatViewBindings.renderChatMessages?.();
    }
}

export function scheduleChatPersist(tab) {
    const t = tab || activeChatTab();
    if (!t) return;
    window.dispatchEvent(new CustomEvent('replyPlanChanged'));
    if (!chat._persistTab) {
        chat._persistTab = debounce(async (tabToSave) => {
            const normalized = normalizeTabForSave(tabToSave);
            try {
                const resp = await fetch(`/api/chat/tabs/${tabToSave.id}`, {
                    method: 'PUT',
                    headers: window.authHeaders
                        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                        : { 'Content-Type': 'application/json' },
                    body: JSON.stringify(normalized),
                });
                if (!resp.ok) {
                    if (resp.status === 404) {
                        // Tab does not exist in DB; remove from local state.
                        chat.tabs = chat.tabs.filter(tb => tb.id !== tabToSave.id);
                    } else {
                        console.error('persist tab error:', resp.status);
                    }
                }
            } catch (e) {
                console.error('persist tab error:', e);
            }
        }, CHAT_TABS_PERSIST_DEBOUNCE_MS);
    }
    chat._persistTab(t);
}

function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
    };
}

export function purgeOldTrash() {
    const cutoff = Date.now() - TRASH_AUTO_PURGE_MS;
    const before = chat.tabTrash.length;
    chat.tabTrash = chat.tabTrash.filter(entry => entry.trashedAt > cutoff);
    const purged = before - chat.tabTrash.length;
    if (purged > 0) {
        scheduleChatPersist();
    }
    return purged;
}

export function markChatTabsDirty() {
    chat.tabsDirty = true;
}

export async function persistChatTabs() {
    // No longer used — individual tab persistence via scheduleChatPersist(tab)
    // Kept for backward compatibility but does nothing.
}

export function persistTabOrder() {
    const ids = chat.tabs.map(t => t.id);
    fetch('/api/chat/tabs/order', {
        method: 'PATCH',
        headers: window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tab_order: ids }),
    }).catch(e => console.error('persistTabOrder error:', e));
}

export function flushChatPersist() {
    clearTimeout(chat.persistTimer);
    clearInterval(chat.periodicSaveTimer);
    if (chat.tabs && chat.tabs.length) {
        chat.tabs.forEach(tab => {
            const normalized = normalizeTabForSave(tab);
            fetch(`/api/chat/tabs/${tab.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(normalized),
                keepalive: true,
            }).catch(() => {});
        });
    }
}

// ── Busy UI ────────────────────────────────────────────────────────────────────

// Getter for transport functions — avoids circular import (chat-state ↔ chat-transport)
let _getTransport = null;
export function setTransportGetter(getter) {
    _getTransport = getter;
}

export function setChatBusyUI(busy) {
    const sendBtn = document.getElementById('btn-send');
    const transport = _getTransport ? _getTransport() : null;
    if (busy) {
        sendBtn.onclick = () => transport?.stopChat();
        sendBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <rect x="6" y="6" width="12" height="12" rx="2"/>
           </svg>`;
        sendBtn.classList.add('btn-chat-send-stop');
        sendBtn.title = 'Stop generating';
    } else {
        sendBtn.onclick = () => transport?.sendChat();
        sendBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
           </svg>`;
        sendBtn.classList.remove('btn-chat-send-stop');
        sendBtn.title = 'Send message';
    }

    const input = document.getElementById('chat-input');
    if (input) input.disabled = busy;
    chatViewBindings.refreshChatTelemetry?.();
    refreshTopCockpit();
}

// ── Input ──────────────────────────────────────────────────────────────────────

export function autoResizeChatInput() {
    const input = document.getElementById('chat-input');
    if (!input) return;
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
}

// ── Init ───────────────────────────────────────────────────────────────────────

  export function initChatState() {
    window.addEventListener('beforeunload', flushChatPersist);
    window.addEventListener('settings-applied', event => {
        if (event?.detail?.persist_thinking_content === false) {
            stripThinkingFromLoadedTabs();
        }
    });
    chat.trashPurgeTimer = setInterval(purgeOldTrash, TRASH_PURGE_CHECK_INTERVAL_MS);
    purgeOldTrash();

    const input = document.getElementById('chat-input');
    if (input) {
        input.addEventListener('input', () => {
            const tab = activeChatTab();
            if (tab) {
                tab.composer_draft = input.value;
                scheduleChatPersist();
            }
        });
    }
}
