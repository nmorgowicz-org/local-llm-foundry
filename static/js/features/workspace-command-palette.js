// workspace-command-palette.js — Unified command palette (Cmd+K / Ctrl+K)
// Keyboard-first search across conversations, messages, and quick actions.

import { showPromptDialog } from './toast.js';
import { chat } from '../core/app-state.js';
import {
    addChatTab,
    renameChatTab,
    togglePinTab,
    archiveChatTab,
    hideChatTab,
    restoreChatTab,
    closeChatTab,
    duplicateChatTab,
} from './chat-state.js';
import Router from './router.js';

const DEBOUNCE_MS = 150;
const FTS_LIMIT = 20;

let _overlay = null;
let _input = null;
let _results = null;
let _timer = null;
let _selectedIndex = -1;
let _items = [];
let _open = false;
let _query = '';

export function initCommandPalette() {
    _overlay = document.getElementById('command-palette-overlay');
    _input = document.getElementById('command-palette-input');
    _results = document.getElementById('command-palette-results');

    if (!_overlay || !_input || !_results) return;

    _input.addEventListener('input', onInput);
    _input.addEventListener('keydown', onInputKeydown);

    _overlay.addEventListener('click', (e) => {
        if (e.target === _overlay) {
            closeCommandPalette();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.defaultPrevented || e.repeat || e.isComposing) return;

        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            if (_open) {
                closeCommandPalette();
            } else {
                openCommandPalette();
            }
            return;
        }

        if (e.key === 'Escape' && _open) {
            e.preventDefault();
            e.stopPropagation();
            closeCommandPalette();
        }
    });
}

export function openCommandPalette() {
    if (_open) return;
    _open = true;
    _query = '';
    _selectedIndex = -1;
    _items = [];

    _overlay.style.display = 'flex';
    _input.value = '';
    _results.innerHTML = '';

    _input.focus();

    renderDefaultActions();
}

export function closeCommandPalette() {
    if (!_open) return;
    _open = false;
    _overlay.style.display = 'none';
    _input.value = '';
    _results.innerHTML = '';
    _selectedIndex = -1;
    _items = [];
    _query = '';
    clearTimeout(_timer);
}

function onInput() {
    const q = _input.value.trim();
    _query = q;
    clearTimeout(_timer);

    if (!q) {
        renderDefaultActions();
        return;
    }

    _timer = setTimeout(() => {
        performSearch(q);
    }, DEBOUNCE_MS);
}

function onInputKeydown(e) {
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        e.stopPropagation();
        _selectedIndex = Math.min(_selectedIndex + 1, _items.length - 1);
        highlightItem();
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        e.stopPropagation();
        _selectedIndex = Math.max(_selectedIndex - 1, 0);
        highlightItem();
    } else if (e.key === 'Enter' && _selectedIndex >= 0 && _selectedIndex < _items.length) {
        e.preventDefault();
        e.stopPropagation();
        activateItem(_items[_selectedIndex]);
    }
}

async function performSearch(query) {
    _results.innerHTML = '';
    _items = [];
    _selectedIndex = -1;

    const titleResults = filterTabsByTitle(query);
    const ftsResults = await fetchFTS(query);

    const allResults = buildResults(query, titleResults, ftsResults);

    if (allResults.length === 0) {
        renderEmptyState(query);
        return;
    }

    renderResults(allResults);
}

function filterTabsByTitle(query) {
    const q = query.toLowerCase();
    return chat.tabs
        .filter(tab => tab.name.toLowerCase().includes(q))
        .map(tab => ({
            type: 'conversation',
            tabId: tab.id,
            title: tab.name,
            visibility: tab.visibility,
            pinned: tab.pinned,
            messageCount: getTabMessageCount(tab),
            persona: tab.active_template_id || tab.ai_name || '',
        }));
}

async function fetchFTS(query) {
    try {
        const resp = await fetch(
            `/api/chat/search?q=${encodeURIComponent(query)}&limit=${FTS_LIMIT}&offset=0`,
            { headers: window.authHeaders ? window.authHeaders() : {} },
        );
        if (!resp.ok) return [];
        const data = await resp.json();
        return (data.results || []).map(r => ({
            type: 'message',
            tabId: r.tab_id,
            tabName: r.tab_name,
            messageId: r.message_id,
            snippet: r.snippet || '',
            role: r.role,
            timestamp: r.timestamp_ms,
        }));
    } catch {
        return [];
    }
}

function buildResults(query, titleResults, ftsResults) {
    const results = [];

    if (!query) {
        return results;
    }

    for (const r of titleResults) {
        results.push(r);
    }

    const seenTabs = new Set(titleResults.map(r => r.tabId));
    for (const r of ftsResults) {
        if (!seenTabs.has(r.tabId)) {
            const tab = getTabById(r.tabId);
            results.push({
                type: 'conversation',
                tabId: r.tabId,
                title: r.tabName,
                visibility: tab?.visibility || 'active',
                pinned: !!tab?.pinned,
                messageCount: getTabMessageCount(tab),
                persona: tab?.active_template_id || tab?.ai_name || '',
                _fromFTS: true,
            });
            seenTabs.add(r.tabId);
        }
        results.push(r);
    }

    const actionTabs = Array.from(seenTabs)
        .map(getTabById)
        .filter(Boolean)
        .slice(0, 5);
    for (const tab of actionTabs) {
        results.push(...buildTabActionResults(tab));
    }

    return results;
}

function renderDefaultActions() {
    _items = [];
    _results.innerHTML = '';

    const actions = [
        { type: 'action', label: 'New Chat', action: 'new-chat', icon: '✦' },
        { type: 'action', label: 'Search Messages', action: 'search-messages', icon: '⌕' },
    ];

    for (const act of actions) {
        const el = createActionItem(act);
        _results.appendChild(el);
        _items.push(act);
    }
}

function renderResults(results) {
    _items = results;
    _results.innerHTML = '';

    for (const r of results) {
        if (r.type === 'action') {
            _results.appendChild(createActionItem(r));
        } else if (r.type === 'tab-action') {
            _results.appendChild(createActionItem(r));
        } else if (r.type === 'conversation') {
            _results.appendChild(createConversationItem(r));
        } else if (r.type === 'message') {
            _results.appendChild(createMessageItem(r));
        }
    }
}

function renderEmptyState(query) {
    _items = [];
    const el = document.createElement('div');
    el.className = 'command-palette-empty';
    const title = document.createElement('div');
    title.className = 'command-palette-empty-title';
    title.textContent = 'No results found';
    const body = document.createElement('div');
    body.className = 'command-palette-empty-body';
    body.textContent = `No conversations or messages matched "${query}".`;
    el.appendChild(title);
    el.appendChild(body);
    _results.appendChild(el);
}

function createActionItem(action) {
    const el = document.createElement('div');
    el.className = 'command-palette-item';
    el.dataset.index = _items.length;

    const typeBadge = document.createElement('span');
    typeBadge.className = 'command-palette-item-type';
    typeBadge.textContent = action.type === 'tab-action' ? 'command' : 'action';

    const icon = document.createElement('span');
    icon.className = 'command-palette-item-icon';
    icon.textContent = action.icon || '•';

    const title = document.createElement('span');
    title.className = 'command-palette-item-title';
    title.textContent = action.label;

    el.appendChild(typeBadge);
    el.appendChild(icon);
    el.appendChild(title);

    if (action.meta) {
        const meta = document.createElement('span');
        meta.className = 'command-palette-item-meta';
        meta.textContent = action.meta;
        el.appendChild(meta);
    }

    el.addEventListener('click', () => activateItem(action));

    return el;
}

function createConversationItem(result) {
    const el = document.createElement('div');
    el.className = 'command-palette-item';
    el.dataset.index = _items.length;

    const typeBadge = document.createElement('span');
    typeBadge.className = 'command-palette-item-type';
    typeBadge.textContent = 'conversation';

    const title = document.createElement('span');
    title.className = 'command-palette-item-title';
    title.textContent = result.title;

    const meta = document.createElement('span');
    meta.className = 'command-palette-item-meta';
    const parts = [];
    if (result.visibility !== 'active') {
        parts.push(result.visibility);
    }
    if (result.pinned) {
        parts.push('pinned');
    }
    if (result.messageCount > 0) {
        parts.push(`${result.messageCount} messages`);
    }
    if (result.persona) {
        parts.push(result.persona);
    }
    meta.textContent = parts.join(' · ');

    el.appendChild(typeBadge);
    el.appendChild(title);
    el.appendChild(meta);

    el.addEventListener('click', () => {
        activateItem(result);
    });

    return el;
}

function createMessageItem(result) {
    const el = document.createElement('div');
    el.className = 'command-palette-item';
    el.dataset.index = _items.length;

    const typeBadge = document.createElement('span');
    typeBadge.className = 'command-palette-item-type';
    typeBadge.textContent = 'message';

    const title = document.createElement('span');
    title.className = 'command-palette-item-title';
    const snippet = result.snippet.length > 100
        ? result.snippet.substring(0, 100) + '…'
        : result.snippet;
    title.textContent = snippet;

    const meta = document.createElement('span');
    meta.className = 'command-palette-item-meta';
    const roleLabel = result.role === 'assistant' ? 'AI' : 'You';
    const timeStr = formatTimestamp(result.timestamp);
    meta.textContent = `${result.tabName} · ${roleLabel}${timeStr ? ` · ${timeStr}` : ''}`;

    el.appendChild(typeBadge);
    el.appendChild(title);
    el.appendChild(meta);

    el.addEventListener('click', () => {
        activateItem(result);
    });

    return el;
}

function highlightItem() {
    const domItems = _results.querySelectorAll('.command-palette-item');
    domItems.forEach((el, i) => {
        el.classList.toggle('active', i === _selectedIndex);
    });

    const selected = domItems[_selectedIndex];
    if (selected) {
        selected.scrollIntoView({ block: 'nearest' });
    }
}

function activateItem(item) {
    closeCommandPalette();

    if (item.type === 'action') {
        handleAction(item.action);
    } else if (item.type === 'tab-action') {
        handleTabAction(item);
    } else if (item.type === 'conversation') {
        switchToConversation(item);
    } else if (item.type === 'message') {
        openMessage(item);
    }
}

async function handleTabAction(item) {
    const tab = getTabById(item.tabId);
    if (!tab) return;
    switch (item.action) {
        case 'pin':
            togglePinTab(tab.id);
            break;
        case 'archive':
            archiveChatTab(tab.id);
            break;
        case 'restore':
            restoreChatTab(tab.id);
            break;
        case 'hide':
            hideChatTab(tab.id);
            break;
        case 'duplicate':
            duplicateChatTab(tab.id);
            break;
        case 'rename': {
            const newName = await showPromptDialog('Rename conversation', 'Enter a new name:', tab.name);
            if (newName && newName.trim()) renameChatTab(tab.id, newName.trim());
            break;
        }
        case 'delete':
            closeChatTab(tab.id);
            break;
    }
}

function handleAction(action) {
    switch (action) {
        case 'new-chat':
            addChatTab();
            break;
        case 'search-messages':
            import('./chat-search.js').then(mod => {
                if (typeof mod.openSearch === 'function') {
                    mod.openSearch();
                }
            });
            break;
    }
}

async function switchToConversation(item) {
    const tab = chat.tabs.find(t => t.id === item.tabId);
    if (!tab) return;

    if (tab.visibility !== 'active') {
        restoreChatTab(item.tabId);
    }

    Router.navigate('/chat/' + encodeURIComponent(item.tabId));
}

async function openMessage(item) {
    Router.navigate('/chat/' + encodeURIComponent(item.tabId));

    setTimeout(() => {
        const el = document.querySelector(`.chat-message[data-msg-id="${item.messageId}"]`);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            el.classList.add('chat-msg-highlight');
            setTimeout(() => el.classList.remove('chat-msg-highlight'), 2000);
        }
    }, 400);
}

function formatTimestamp(timestampMs) {
    if (!timestampMs) return '';
    try {
        return new Intl.DateTimeFormat([], {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
        }).format(new Date(timestampMs));
    } catch {
        return '';
    }
}


function getTabById(tabId) {
    return chat.tabs.find(tab => tab.id === tabId) || null;
}

function getTabMessageCount(tab) {
    if (!tab) return 0;
    if (tab._loaded) return (tab.messages || []).length;
    return tab.message_count || (tab.messages || []).length || 0;
}

function buildTabActionResults(tab) {
    const visibilityLabel = tab.visibility === 'active' ? tab.name : `${tab.name} · ${tab.visibility}`;
    const actions = [
        { action: 'pin', label: tab.pinned ? 'Unpin conversation' : 'Pin conversation', icon: tab.pinned ? '📌' : '⊙' },
        { action: tab.visibility === 'active' ? 'archive' : 'restore', label: tab.visibility === 'archived' ? 'Unarchive conversation' : 'Archive conversation', icon: '🗂' },
        { action: tab.visibility === 'hidden' ? 'restore' : 'hide', label: tab.visibility === 'hidden' ? 'Unhide conversation' : 'Hide conversation', icon: '🙈' },
        { action: 'duplicate', label: 'Duplicate conversation', icon: '⧉' },
        { action: 'rename', label: 'Rename conversation', icon: '✎' },
        { action: 'delete', label: 'Delete conversation', icon: '✕' },
    ];
    return actions.map(item => ({
        type: 'tab-action',
        tabId: tab.id,
        action: item.action,
        label: item.label,
        icon: item.icon,
        meta: visibilityLabel,
    }));
}
