// ── Chat Rendering ────────────────────────────────────────────────────────────
// Rendering functions for chat tabs, messages, compaction markers, and actions.
// Calls rendering functions via window.* to avoid circular imports.

import { chat, lastLlamaMetrics, contextCapacityTokens, settingsState } from '../core/app-state.js';
import { setHtml } from '../core/set-html.js';
import { escapeHtml } from '../core/format.js';
import {
  activeChatTab,
  addChatTab,
  getChatViewBindings,
  hideChatTab,
  registerChatViewBindings,
  scheduleChatPersist,
  closeChatTab,
  renameChatTab,
  normalizeTabForSave,
  togglePinTab,
} from './chat-state.js';
import { showToast, showToastWithActions, showConfirmDialog } from './toast.js';
import './chat-templates.js';
import { renderChatSessionsSidebar } from './chat-sessions-sidebar.js';
import Router from './router.js';

// Getter for transport functions — avoids circular import (chat-render ↔ chat-transport)
let _getTransport = null;
export function setChatTransportGetter(getter) {
    _getTransport = getter;
}

function getTransport() {
    return _getTransport ? _getTransport() : null;
}

// ── Date formatting ───────────────────────────────────────────────────────────

export function formatMessageDateTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const fmt = settingsState.chat_date_format || localStorage.getItem('llama-monitor-date-format') || 'MM/DD/YY';
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const yy = String(d.getFullYear()).slice(-2);
    let date;
    if (fmt === 'locale') {
        date = d.toLocaleDateString([], { month: 'numeric', day: 'numeric', year: '2-digit' });
    } else if (fmt === 'DD/MM/YY') {
        date = `${dd}/${mm}/${yy}`;
    } else if (fmt === 'YYYY-MM-DD') {
        date = `${d.getFullYear()}-${mm}-${dd}`;
    } else {
        date = `${mm}/${dd}/${yy}`;
    }
    const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return `${date} ${time}`;
}

// ── Cached DOM elements (populated at init time) ──────────────────────────────
let chatMessagesEl = null;
let chatTabBarEl = null;
let chatScrollBottomBtn = null;
let chatScrollBadge = null;
let sidebarBadgeChat = null;

function ensureChatElements() {
    if (chatMessagesEl) return;
    chatMessagesEl = document.getElementById('chat-messages-inner');
    chatTabBarEl = document.getElementById('chat-tab-bar');
    chatScrollBottomBtn = document.getElementById('chat-scroll-bottom');
    chatScrollBadge = document.getElementById('chat-scroll-badge');
    sidebarBadgeChat = document.getElementById('sidebar-badge-chat');
}

// ── Markdown rendering ────────────────────────────────────────────────────────

export function renderMd(src) {
    if (typeof marked !== 'undefined') {
        try {
            const raw = marked.parse(src);
            return (typeof window.DOMPurify !== 'undefined' ? window.DOMPurify.sanitize(raw) : raw);
        } catch(_) {}
    }
    return src.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'<br>');
}

export function renderMdStreaming(src) {
    if (typeof marked !== 'undefined') {
        try {
            const raw = marked.parse(src, { gfm: true, breaks: true, renderer: new marked.Renderer() });
            return (typeof window.DOMPurify !== 'undefined' ? window.DOMPurify.sanitize(raw) : raw);
        } catch(_) {}
    }
    return src.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'<br>');
}

// ── Roleplay text colorization ────────────────────────────────────────────────
// Walks text nodes inside `el` and wraps "quoted dialogue" in .rp-dialogue spans.
// Skips code blocks. em tags (from *asterisks* markdown) get colour via CSS only.

export function colorizeRpText(el) {
    if (!el) return;

    // Match all common Unicode double quotation marks — LLMs routinely mix variants.
    // U+0022 ", U+201C ", U+201D ", U+201E „, U+201F ‟, U+00AB «, U+00BB »
    // (single quotes omitted — U+2019/U+0027 are also apostrophes in contractions)
    // Allow newlines inside quotes (multi-line dialogue), max length 4000 chars.
    // Use non-greedy (lazy) quantifier to match shortest quote pair.
    const dialogueRe = /[\u0022\u201C\u201D\u201E\u201F\u00AB\u00BB][^\u0022\u201C\u201D\u201E\u201F\u00AB\u00BB]{1,4000}?[\u0022\u201C\u201D\u201E\u201F\u00AB\u00BB]/g;

    // Process each block-level child independently.
    // For blocks with inline formatting (em, strong), we flatten text across element
    // boundaries so the regex can match quotes that span across markdown-generated tags.
    const blockSelectors = 'p, li, blockquote, div, h1, h2, h3, h4, h5, h6';
    const blocks = el.querySelectorAll(blockSelectors);

    if (blocks.length === 0) {
        colorizeBlock(el, dialogueRe);
        return;
    }

    blocks.forEach(block => {
        if (block.closest('pre, code')) return;
        colorizeBlock(block, dialogueRe);
    });
}

function colorizeBlock(block, dialogueRe) {
    const fullText = block.textContent;
    if (!fullText) return;

    // Find all quote matches in the flattened text
    const matches = [];
    let m;
    dialogueRe.lastIndex = 0;
    while ((m = dialogueRe.exec(fullText)) !== null) {
        matches.push({ start: m.index, end: m.index + m[0].length });
    }

    if (matches.length === 0) return;

    // Skip if already colorized (idempotency)
    if (block.querySelector('.rp-dialogue')) return;

   // If no inline formatting, use simple text-node replacement
    if (!block.querySelector('em, strong, code, a, del, ins, s, u, abbr, kbd, var, sub, sup')) {
        colorizeTextNodes(block, dialogueRe);
        return;
    }

    // Has inline formatting — rebuild HTML to handle cross-boundary quotes
    colorizeWithRebuild(block, fullText, matches);
}

function colorizeTextNodes(block, dialogueRe) {
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, null);
    const textNodes = [];
    let node;
    while ((node = walker.nextNode())) textNodes.push(node);

    for (const textNode of textNodes) {
        const parent = textNode.parentNode;
        if (!parent || parent.closest('pre, code')) continue;
        const text = textNode.textContent;
        if (!dialogueRe.test(text)) { dialogueRe.lastIndex = 0; continue; }
        dialogueRe.lastIndex = 0;

        const frag = document.createDocumentFragment();
        let last = 0;
        let m;
        while ((m = dialogueRe.exec(text)) !== null) {
            if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
            const span = document.createElement('span');
            span.className = 'rp-dialogue';
            span.textContent = m[0];
            frag.appendChild(span);
            last = m.index + m[0].length;
        }
        if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
        parent.replaceChild(frag, textNode);
    }
}

function colorizeWithRebuild(block, _fullText, matches) {
    // Build a character stream that tracks dialogue state and formatting tags.
    // Then rebuild the block's HTML with <span class="rp-dialogue"> at quote boundaries,
    // preserving ALL inline formatting throughout.

    const INLINE_FMT_TAGS = new Set(['em', 'strong', 'code', 'a', 'del', 'ins', 's', 'u', 'abbr', 'kbd', 'var', 'sub', 'sup']);

    const dialogueSet = new Set();
    for (const match of matches) {
        for (let i = match.start; i < match.end; i++) {
            dialogueSet.add(i);
        }
    }

    // Walk text nodes and record each character's dialogue state and formatting context
    const stream = [];
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, null);
    let textNode;
    let textIdx = 0;

    while ((textNode = walker.nextNode())) {
        const text = textNode.textContent;
        // Collect ALL inline formatting ancestors (outermost → innermost)
        const tagStack = [];
        let ancestor = textNode;
        while (ancestor) {
            if (ancestor.nodeType === 1) {
                const tag = ancestor.tagName.toLowerCase();
                if (INLINE_FMT_TAGS.has(tag)) {
                    tagStack.unshift(tag);
                }
            }
            ancestor = ancestor.parentNode;
        }

        for (let i = 0; i < text.length; i++) {
            stream.push({
                ch: text[i],
                dialogue: dialogueSet.has(textIdx + i),
                stack: tagStack.length ? tagStack.join('|') : '',
            });
        }
        textIdx += text.length;
    }

    // Build HTML by tracking state transitions
    let html = '';
    let inDialogue = false;
    let currentStack = [];

    for (const s of stream) {
        const targetStack = s.stack ? s.stack.split('|') : [];

        // Dialogue state transition
        if (s.dialogue !== inDialogue) {
            if (inDialogue) {
                // Exiting dialogue: close all fmt tags, close span
                while (currentStack.length) {
                    html += `</${currentStack.pop()}>`;
                }
                html += '</span>';
                inDialogue = false;

                // Reopen fmt tags for current character
                for (const tag of targetStack) {
                    html += `<${tag}>`;
                    currentStack.push(tag);
                }
            } else {
                // Entering dialogue: close all fmt tags, open span, reopen fmt tags
                while (currentStack.length) {
                    html += `</${currentStack.pop()}>`;
                }
                html += '<span class="rp-dialogue">';
                inDialogue = true;
                for (const tag of targetStack) {
                    html += `<${tag}>`;
                    currentStack.push(tag);
                }
            }
        } else {
            // Formatting state transition (same dialogue state)
            // Sync currentStack → targetStack by finding common prefix
            let i = 0;
            while (i < currentStack.length && i < targetStack.length && currentStack[i] === targetStack[i]) i++;
            // Close tags after divergence point
            while (currentStack.length > i) {
                html += `</${currentStack.pop()}>`;
            }
            // Open new tags after divergence point
            for (let j = i; j < targetStack.length; j++) {
                html += `<${targetStack[j]}>`;
                currentStack.push(targetStack[j]);
            }
        }

        html += escapeHtmlChar(s.ch);
    }

    // Close remaining open tags
    while (currentStack.length) html += `</${currentStack.pop()}>`;
    if (inDialogue) html += '</span>';


    setHtml(block, html);
}

function escapeHtmlChar(c) {
    switch (c) {
        case '<': return '&lt;';
        case '>': return '&gt;';
        case '&': return '&amp;';
        case '"': return '&quot;';
        default: return c;
    }
}

// ── Scroll ────────────────────────────────────────────────────────────────────

export function chatScroll(force = false) {
    ensureChatElements();
    const c = chatMessagesEl;
    if (!c) return;

    // Don't auto-scroll if user has manually scrolled up during generation
    if (!force && chat.disableAutoScroll) return;

    const distFromBottom = c.scrollHeight - c.scrollTop - c.clientHeight;
    if (force || distFromBottom < 80) {
        c.scrollTop = c.scrollHeight;
    }
    if (force) {
        chat.unreadChatCount = 0;
        if (chatScrollBadge) chatScrollBadge.style.display = 'none';
        // Reset auto-scroll disable flag after forced scroll
        chat.disableAutoScroll = false;
    }
}

// Scroll so the top of `el` sits at the top of the chat viewport (+ 8px padding).
// Used on first AI token: after the initial forced-to-bottom on submit, this
// repositions so the response bubble starts at the top, giving maximum reading room.
export function chatScrollToEl(el) {
    ensureChatElements();
    const c = chatMessagesEl;
    if (!c || !el) return;
    if (chat.disableAutoScroll) return;
    const containerTop = c.getBoundingClientRect().top;
    const elTop = el.getBoundingClientRect().top;
    c.scrollTop = c.scrollTop + (elTop - containerTop) - 8;
}

function initChatScrollButton() {
    ensureChatElements();
    const container = chatMessagesEl;
    const btn = chatScrollBottomBtn;
    if (!container || !btn) return;

    const checkScroll = () => {
        const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
        btn.classList.toggle('visible', distFromBottom > 100);

        // Disable auto-scroll if user scrolls up more than 100px during generation
        if (distFromBottom > 100 && chat.busy) {
            chat.disableAutoScroll = true;
        }
    };

    container.addEventListener('scroll', checkScroll, { passive: true });
    requestAnimationFrame(() => requestAnimationFrame(checkScroll));
}

export function incrementUnreadCount() {
    ensureChatElements();
    const container = chatMessagesEl;
    if (!container) return;
    const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    if (distFromBottom > 80) {
        chat.unreadChatCount++;
        if (chatScrollBadge) {
            chatScrollBadge.textContent = chat.unreadChatCount;
            chatScrollBadge.style.display = 'flex';
        }
    }
}

// ── Tab rendering ─────────────────────────────────────────────────────────────

let _draggedTabId = null;

export function renderChatTabs() {
    ensureChatElements();
    const bar = chatTabBarEl;
    if (!bar) return;
    const addBtn = bar?.querySelector('.chat-tab-add');
    bar?.querySelectorAll('.chat-tab').forEach(el => el.remove());

    for (const tab of chat.tabs) {
        const el = document.createElement('div');
        const msgCount = Array.isArray(tab.messages)
            ? tab.messages.filter(m => m.role !== 'system').length
            : (tab.message_count || 0);
        let extraClasses = '';
        if (msgCount > 50) extraClasses = ' tab-hot';
        else if (msgCount > 20) extraClasses = ' tab-warm';
        el.className = 'chat-tab' + (tab.id === chat.activeTabId ? ' active' : '') + (tab.pinned ? ' chat-tab-pinned' : '') + extraClasses;
        el.dataset.tabId = tab.id;
        el.draggable = true;

        setHtml(el, `
          <div class="chat-tab-name-wrapper">
            <span class="chat-tab-name" data-chat-tab-rename="${tab.id}">${escapeHtml(tab.name)}</span>
          </div>
          <svg class="chat-tab-pin-icon ${tab.pinned ? 'pinned' : ''}" width="11" height="11" viewBox="0 0 24 24" fill="${tab.pinned ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" aria-hidden="true" data-tooltip="${tab.pinned ? 'Unpin tab' : 'Pin tab'}">
            <path d="M16 12V3h-3V2H8v1H5v9l-2 6 3 1 2-5v9h6v-9l2 5 3-1-2-6z"/>
          </svg>
          <svg class="chat-tab-edit-icon" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true">
            <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
          <span class="chat-tab-count">${msgCount || ''}</span>
          ${chat.tabs.length > 1
            ? `<button class="chat-tab-close" data-chat-tab-close="${tab.id}" title="Close tab">×</button>`
            : ''}
        `);
        el.addEventListener('click', e => {
            const closeBtn = e.target.closest('.chat-tab-close');
            if (closeBtn) return;
            if (e.target.classList.contains('chat-tab-name') && e.detail === 2) return;
            Router.navigate('/chat/' + encodeURIComponent(tab.id));
        });

        // Drag-to-reorder
        el.addEventListener('dragstart', e => {
            _draggedTabId = tab.id;
            el.classList.add('tab-dragging');
            e.dataTransfer.effectAllowed = 'move';
        });
        el.addEventListener('dragend', () => {
            _draggedTabId = null;
            bar.querySelectorAll('.chat-tab').forEach(t => {
                t.classList.remove('tab-dragging', 'tab-drop-target');
            });
        });
        el.addEventListener('dragover', e => {
            if (_draggedTabId && _draggedTabId !== tab.id) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                bar.querySelectorAll('.chat-tab').forEach(t => t.classList.remove('tab-drop-target'));
                el.classList.add('tab-drop-target');
            }
        });
        el.addEventListener('dragleave', () => el.classList.remove('tab-drop-target'));
        el.addEventListener('drop', e => {
            e.preventDefault();
            el.classList.remove('tab-drop-target');
            if (!_draggedTabId || _draggedTabId === tab.id) return;
            const draggedTab = chat.tabs.find(t => t.id === _draggedTabId);
            const targetTab = tab;
            if (!draggedTab || !targetTab) return;
            // Can't drag pinned tabs into unpinned section and vice versa
            if (draggedTab.pinned !== targetTab.pinned) return;
            const fromIdx = chat.tabs.findIndex(t => t.id === _draggedTabId);
            const toIdx = chat.tabs.findIndex(t => t.id === tab.id);
            if (fromIdx < 0 || toIdx < 0) return;
            const [moved] = chat.tabs.splice(fromIdx, 1);
            chat.tabs.splice(toIdx, 0, moved);
            renderChatTabs();
            scheduleChatPersist();
        });

        // Pin button click handler
        const pinBtn = el.querySelector('.chat-tab-pin-icon');
        if (pinBtn) {
            pinBtn.title = tab.pinned ? 'Unpin tab' : 'Pin tab';
            pinBtn.addEventListener('click', e => {
                e.stopPropagation();
                togglePinTab(tab.id);
            });
        }

        if (tab.explicit_level > 0) {
            const badge = document.createElement('span');
            badge.className = 'chat-tab-explicit-badge';
            badge.textContent = tab.explicit_level >= 2 ? '\uD83D\uDD25' : '\uD83D\uDD13';
            badge.title = tab.explicit_level >= 2 ? 'Unrestricted mode' : 'Explicit mode';
            const countEl = el.querySelector('.chat-tab-count');
            if (countEl) {
                countEl.parentNode.insertBefore(badge, countEl);
            } else {
                el.appendChild(badge);
            }
        }

        bar.insertBefore(el, addBtn);
    }
    // Add separator between pinned and unpinned tabs if both exist
    const firstUnpinned = bar.querySelector('.chat-tab:not(.chat-tab-pinned)');
    if (firstUnpinned && bar.querySelector('.chat-tab-pinned')) {
        const sep = document.createElement('div');
        sep.className = 'chat-tab-pin-sep';
        if (!bar.querySelector('.chat-tab-pin-sep')) {
            bar.insertBefore(sep, firstUnpinned);
        }
    }
}

function getTimeAgo(ts) {
    const diff = Date.now() - ts;
    const secs = Math.floor(diff / 1000);
    if (secs < 60) return `${secs}s ago`;
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
}

export function renderTrashDropdown() {
    const dropdown = document.getElementById('chat-tab-trash-dropdown');
    const trashBtn = document.getElementById('chat-tab-trash-btn');
    if (!dropdown || !trashBtn) return;

    const existingBadge = trashBtn.querySelector('.trash-badge');
    if (existingBadge) existingBadge.remove();
    if (chat.tabTrash.length > 0) {
        const badge = document.createElement('span');
        badge.className = 'trash-badge';
        badge.textContent = chat.tabTrash.length;
        trashBtn.appendChild(badge);
    }

    setHtml(dropdown, '');
    if (chat.tabTrash.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'chat-tab-trash-dropdown-empty';
        empty.textContent = 'Trash is empty';
        dropdown.appendChild(empty);
        return;
    }

    for (const entry of chat.tabTrash) {
        const item = document.createElement('div');
        item.className = 'chat-tab-trash-item';
        const timeAgo = getTimeAgo(entry.trashedAt);

        setHtml(item, `
            <span class="chat-tab-trash-item-name">${escapeHtml(entry.tab.name)}</span>
            <span class="chat-tab-trash-item-time">${timeAgo}</span>
            <button class="chat-tab-trash-item-restore" data-trash-restore="${entry.tab.id}">Restore</button>
        `);
        dropdown.appendChild(item);
    }

    const footer = document.createElement('div');
    footer.className = 'chat-tab-trash-dropdown-footer';
    const clearBtn = document.createElement('button');
    clearBtn.className = 'chat-tab-trash-clear-btn';
    clearBtn.textContent = 'Clear all';
    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        chat.tabTrash = [];
        renderTrashDropdown();
        scheduleChatPersist();
    });
    footer.appendChild(clearBtn);
    dropdown.appendChild(footer);
}

// ── Message rendering ─────────────────────────────────────────────────────────

export function renderChatMessages(optionsOrSkip = false) {
    ensureChatElements();
    const container = chatMessagesEl;
    const tab = activeChatTab();
    const options = typeof optionsOrSkip === 'object' && optionsOrSkip !== null
        ? optionsOrSkip
        : { skipAutoScroll: !!optionsOrSkip };
    const skipAutoScroll = !!options.skipAutoScroll;

    if (!tab) {
        setHtml(container, `
          <div class="chat-empty">
            <div class="chat-empty-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="1.2" opacity="0.25">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
              </svg>
            </div>
            <p class="chat-empty-title">No chats open</p>
            <p class="chat-empty-hint">Create a new chat or restore one from trash.</p>
            <button class="btn btn-primary" id="chat-empty-create-btn">New Chat</button>
          </div>`);
        document.getElementById('chat-empty-create-btn')?.addEventListener('click', () => {
            addChatTab().catch(err => console.error('chat empty create failed:', err));
        });
        return;
    }

    // Tabs loaded from the server temporarily use `messages: null` until their
    // message payload arrives. Render that state as an empty chat rather than
    // racing the async loader with an exception.
    const tabMessages = Array.isArray(tab.messages) ? tab.messages : [];
    if (tabMessages.filter(m => m.role !== 'system').length === 0) {
        const prompts = [
            { icon: '💡', text: 'Explain a complex topic simply', label: 'Learn something' },
            { icon: '✍️', text: 'Help me write an email about...', label: 'Write something' },
            { icon: '🔍', text: 'Compare the pros and cons of...', label: 'Analyze something' },
            { icon: '🎨', text: 'Give me creative ideas for...', label: 'Brainstorm' },
        ];
        const promptCards = prompts.map((p, i) => `
            <button class="chat-empty-prompt" style="animation-delay:${i * 60}ms"
                    data-prompt-text="${escapeHtml(p.text)}">
                <span class="chat-empty-prompt-icon">${p.icon}</span>
                <span class="chat-empty-prompt-text">${p.text}</span>
            </button>`).join('');

        const aiName = tab?.ai_name || 'Assistant';
        const modelName = lastLlamaMetrics?.model_name
            ? ` (${lastLlamaMetrics.model_name.split('/').pop().replace(/\.gguf$/i, '')})`
            : '';


        setHtml(container, `
          <div class="chat-empty">
            <div class="chat-empty-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="1.2" opacity="0.25">
                <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
              </svg>
            </div>
            <p class="chat-empty-title">${escapeHtml(aiName)}${escapeHtml(modelName)} is ready</p>
            <p class="chat-empty-hint">Ask anything, or try a suggestion below</p>
            <div class="chat-empty-prompts">${promptCards}</div>
          </div>`);
        return;
    }

    const allMessages = tabMessages.filter(m => m.role !== 'system' || m.compaction_marker);
    const limit = tab.visible_message_limit || 15;
    const isPaginated = allMessages.length > limit;
    const visibleMessages = isPaginated ? allMessages.slice(-limit) : allMessages;

    setHtml(container, '');

    // Add "Load More" button if paginated
    if (isPaginated) {
        const loadMoreBtn = document.createElement('button');
        loadMoreBtn.className = 'chat-load-more';
        const olderCount = allMessages.length - limit;

        setHtml(loadMoreBtn, `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 5v14M5 12l7 7 7-7"/>
            </svg>
            Load ${Math.min(limit, olderCount)} older messages
        `);
        loadMoreBtn.onclick = () => loadMoreMessages(tab, limit);
        container.appendChild(loadMoreBtn);
    }

    let idx = 0;
    for (const msg of visibleMessages) {
        const el = buildMessageElement(msg, idx, tabMessages);
        const realIdx = tabMessages.indexOf(msg);
        if (realIdx >= 0) el.dataset.msgIdx = realIdx;
        el.dataset.msgId = msg.db_id ?? '';
        container.appendChild(el);
        idx++;
    }
    if (!skipAutoScroll) setTimeout(() => chatScroll(true), 50);
    getChatViewBindings().syncCompactSettingsUI?.(activeChatTab());
}

function loadMoreMessages(tab, currentLimit) {
    const messages = Array.isArray(tab.messages) ? tab.messages : [];
    const allMessages = messages.filter(m => m.role !== 'system' || m.compaction_marker);
    tab.visible_message_limit = Math.min(currentLimit * 2, allMessages.length);

    const scrollEl = chatMessagesEl;
    const prevScrollHeight = scrollEl ? scrollEl.scrollHeight : 0;
    const prevScrollTop = scrollEl ? scrollEl.scrollTop : 0;

    renderChatMessages(true);

    // Compensate for content added above: shift scrollTop by the height delta
    // so the viewport stays anchored to the same message the user was reading.
    if (scrollEl) {
        const delta = scrollEl.scrollHeight - prevScrollHeight;
        scrollEl.scrollTop = prevScrollTop + delta;
    }
}

function buildMessageElement(msg, idx, allMessages) {
    const isUser = msg.role === 'user';
    const tab = activeChatTab();
    const wrapper = document.createElement('div');

    // Render compaction tombstone as a divider
    if (msg.compaction_marker) {
        wrapper.className = 'chat-message chat-compact-marker' + (msg.summarized ? ' compact-marker-summarized' : ' compact-marker-truncated');
        wrapper.dataset.compactState = 'final';
        wrapper.dataset.expanded = 'false';

        const isSummarized = !!msg.summarized;
        const isRollingMemory = (msg.memory_version || 0) >= 2 || msg.summary_kind === 'rolling-memory';
        const droppedCount = msg.dropped_count || 0;
        const ctxBefore = msg.ctx_pct_before || 0;
        const totalCompacted = msg.compacted_message_count_total || droppedCount;
        const memoryDomain = msg.memory_domain || '';

        let statsHtml = isRollingMemory
            ? `${droppedCount} compacted now · ${totalCompacted} total`
            : `${droppedCount} messages removed`;
        if (memoryDomain) statsHtml += ` · ${escapeHtml(memoryDomain)}`;
        if (ctxBefore > 0) statsHtml += ` · was ${ctxBefore}% ctx`;

        const labelText = isRollingMemory ? 'Conversation memory' : isSummarized ? 'Context summarized' : 'Context trimmed';
        const iconPath = isRollingMemory
            ? '<path d="M7 4h10a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6a2 2 0 012-2z"/><path d="M9 8h6M9 12h6M9 16h4"/>'
            : isSummarized
            ? '<path d="M9 12h6M9 16h6M9 8h6M5 4h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V6a2 2 0 012-2z"/>'
            : '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>';

        let bodyHtml = '';
        if (isSummarized) {
            const summaryText = msg.content.replace(/^\[Context compacted[^\]]*\]\s*/i, '').trim();
            bodyHtml = summaryText ? renderMd(summaryText) : '';
        } else if (msg.dropped_preview && msg.dropped_preview.length > 0) {
            const rows = msg.dropped_preview.map(p => {
                const label = p.role === 'user' ? 'You' : 'AI';
                return `<div class="compact-peek-row"><span class="compact-peek-role">${label}</span><span class="compact-peek-snippet">${escapeHtml(p.snippet)}${p.snippet.length >= 80 ? '…' : ''}</span></div>`;
            }).join('');
            bodyHtml = `<div class="compact-peek-list">${rows}</div>`;
        }


        setHtml(wrapper, `
          <div class="compact-marker-content">
            <div class="compact-marker-rule compact-marker-rule-left"></div>
            <div class="compact-marker-pill" data-compact-toggle="true">
              <svg class="compact-marker-icon" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${iconPath}</svg>
              <span class="compact-marker-label">${labelText}</span>
              <span class="compact-marker-stats">${statsHtml}</span>
              <svg class="compact-marker-chevron" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M6 9l6 6 6-6"/></svg>
            </div>
            <div class="compact-marker-rule compact-marker-rule-right"></div>
          </div>
          <div class="compact-marker-body" style="display:none;">${bodyHtml}</div>`);

        return wrapper;
    }

    wrapper.className = `chat-message chat-message-${msg.role}`;
    wrapper.dataset.role = msg.role;
    wrapper.dataset.msgIdx = idx;

    const ts = formatMessageDateTime(msg.timestamp_ms);
    const aiLabel = tab?.ai_name || 'AI';
    const userLabel = tab?.user_name || 'You';

    let metaHtml = '';
    if (!isUser) {
        const parts = [];
        if (msg.input_tokens > 0) parts.push(`↓${formatTokenCount(msg.input_tokens)}`);
        if (msg.output_tokens > 0) parts.push(`↑${formatTokenCount(msg.output_tokens)}`);
        let cumInput = 0, cumOutput = 0;
        for (let i = 0; i <= idx; i++) {
            const m = allMessages[i];
            if (m.role === 'assistant') {
                cumInput += m.input_tokens || 0;
                cumOutput += m.output_tokens || 0;
            }
        }
        const cumTotal = cumInput + cumOutput;
        if (cumTotal > 0) parts.push(`R${formatTokenCount(cumTotal)}`);
        const capacity = contextCapacityTokens || lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
        // ctx% = (cumulative output tokens up to this message + this message's input) / capacity.
        // KV cache means input_tokens is incremental; output tokens accumulate as the actual context content.
        const ctxTokens = cumOutput + (msg.input_tokens || 0);
        const ctxPct = capacity > 0 && ctxTokens > 0 ? Math.min(100, Math.round((ctxTokens / capacity) * 100)) : 0;
        if (ctxPct > 0) parts.push(`${ctxPct}% ctx · ${formatTokenCount(capacity)}`);
        const modelName = msg.model_name || lastLlamaMetrics?.model_name || '';
        if (modelName) parts.push(modelName);
        if (parts.length > 0) {
            metaHtml = `<span class="chat-msg-meta-sep">·</span><span class="chat-msg-meta-model" title="↓ = prompt tokens in · ↑ = tokens generated · R = running total · ctx = % of context window used">${parts.join(' · ')}</span>`;
        }
    }

 // Thinking block HTML for persisted reasoning content
    let thinkingHtml = '';
    if (!isUser && msg.thinking_content) {
        const thinkTokens = Math.round(msg.thinking_content.length / 4);
        thinkingHtml = `
        <details class="chat-thinking">
          <summary class="chat-thinking-summary">
            <svg class="chat-thinking-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 16.8l-6.2 4.5 2.4-7.4L2 9.4h7.6z"/></svg>
            <span class="chat-thinking-label">Thinking</span>
            <span class="chat-thinking-token-count" style="margin:0 8px; color:var(--text-muted);">(${thinkTokens} tokens)</span>
          </summary>
          <div class="chat-thinking-body">${escapeHtml(msg.thinking_content)}</div>
        </details>`;
    }

  // All messages rendered via marked.js for consistent styling (backticks, italics, bold); labels are user-configured display names
    const html = `
      <div class="chat-avatar">${isUser ? userLabel : aiLabel}</div>
      <div class="chat-bubble">
        ${thinkingHtml}
        <div class="chat-msg-body">${renderMd(msg.content)}</div>
        <div class="chat-msg-footer">
          <span class="chat-msg-time">${ts}</span>
          ${metaHtml}
          <div class="chat-msg-actions">
            <button class="chat-action-btn" data-chat-action="copy" title="Copy">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012-2v1"/>
              </svg>
            </button>
            ${isUser ? `
            <button class="chat-action-btn" data-chat-action="retry-send" title="Resend">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M23 4v6h-6M1 20v-6h6"/>
                <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
              </svg>
            </button>` : ''}
            ${!isUser ? (() => {
                const variants = msg._variants || [];
                const curIdx = msg._variantIndex || 0;
                const total = variants.length || 1;
                const canGoLeft = variants.length > 1 && curIdx > 0;
                const canGoRight = true;
                return `
            <button class="chat-action-btn" data-chat-action="nav-variant" data-variant-dir="-1" title="Previous response" ${canGoLeft ? '' : 'disabled'}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M15 18l-6-6 6-6"/>
              </svg>
            </button>
            <span class="chat-variant-badge">${curIdx+1}/${total}</span>
            <button class="chat-action-btn" data-chat-action="nav-variant" data-variant-dir="1" title="${canGoRight && variants.length <= 1 ? 'Regenerate' : 'Next response'}" ${canGoRight ? '' : 'disabled'}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 18l6-6-6-6"/>
              </svg>
            </button>`;
            })() : ''}
            <button class="chat-action-btn" data-chat-action="edit" title="Edit">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2">
                <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
            </button>
            <button class="chat-action-btn chat-action-btn-delete" data-chat-action="delete" title="Delete">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="2">
                <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012-2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
              </svg>
            </button>
          </div>
        </div>
      </div>`;

    setHtml(wrapper, html);
    colorizeRpText(wrapper.querySelector('.chat-msg-body'));

    return wrapper;
}

function formatTokenCount(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
}

// ── Streaming helpers ─────────────────────────────────────────────────────────

export function appendAssistantPlaceholder() {
    ensureChatElements();
    const container = chatMessagesEl;
    const tab = activeChatTab();
    const aiLabel = tab?.ai_name || 'AI';
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-message chat-message-assistant chat-message-streaming';

    setHtml(wrapper, `
      <div class="chat-avatar">${aiLabel}</div>
      <div class="chat-bubble">
        <div class="chat-msg-body"><span class="chat-cursor">▋</span></div>
        <div class="chat-msg-footer">
          <span class="chat-msg-time"></span>
          <span class="chat-msg-meta-sep">·</span>
          <span class="chat-msg-meta-model"></span>
          <div class="chat-msg-actions"></div>
        </div>
      </div>`);
    container.appendChild(wrapper);
    chatScroll(false);
    return wrapper;
}

export function appendThinkingBlock(afterEl) {
    const details = document.createElement('details');
    details.className = 'chat-thinking';
    setHtml(details, `
      <summary class="chat-thinking-summary">
        <svg class="chat-thinking-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 16.8l-6.2 4.5 2.4-7.4L2 9.4h7.6z"/></svg>
        <span class="chat-thinking-label">Thinking</span>
        <span class="chat-thinking-dots"><span>.</span><span>.</span><span>.</span></span>
        <span class="chat-thinking-token-count" style="margin:0 8px; color:var(--text-muted);"></span>
        <span class="chat-thinking-hint">(click to expand)</span>
      </summary>
      <div class="chat-thinking-body"></div>`);
    afterEl.parentElement.insertBefore(details, afterEl);
    return details;
}

export function finalizeAssistantMessage(el, content, usage, tab) {
    el.classList.remove('chat-message-streaming');
    const body = el.querySelector('.chat-msg-body');
    if (content) {

        setHtml(body, renderMd(content));
        colorizeRpText(body);
        if (typeof hljs !== 'undefined') {
            body.querySelectorAll('pre code:not(.hljs)').forEach(codeEl => {
                hljs.highlightElement(codeEl);
            });
        }
        body.querySelectorAll('pre').forEach(pre => {
            if (pre.parentElement?.classList.contains('chat-code-block')) return;
            const code = pre.querySelector('code');
            const lang = (code?.className.match(/language-(\w+)/) || [])[1] || '';
            const lineCount = (code?.innerText.match(/\n/g) || []).length + 1;

            const wrapper = document.createElement('div');
            wrapper.className = 'chat-code-block';

            const header = document.createElement('div');
            header.className = 'chat-code-header';

            setHtml(header, `
                <span class="chat-code-lang">${lang || 'code'}</span>
                <span class="chat-code-lines">${lineCount} line${lineCount !== 1 ? 's' : ''}</span>
                <button class="chat-code-copy-btn" title="Copy code">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                  </svg>
                  Copy
                </button>`);

            header.querySelector('.chat-code-copy-btn').addEventListener('click', function() {
                navigator.clipboard.writeText(code?.innerText ?? pre.innerText).then(() => {
                    this.classList.add('copied');
                    setHtml(this, '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied');
                    setTimeout(() => {
                        this.classList.remove('copied');
                        setHtml(this, '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy');
                    }, 1500);
                });
            });

            pre.parentElement.insertBefore(wrapper, pre);
            wrapper.appendChild(header);
            wrapper.appendChild(pre);
        });
    }
    const time = el.querySelector('.chat-msg-time');
    if (time) {
        time.textContent = formatMessageDateTime(Date.now());
    }
    const actions = el.querySelector('.chat-msg-actions');
    if (actions && content) {
        const tab = activeChatTab();
        const variants = tab?._pendingVariants || null;
        let msg = null;
        if (variants) {
            tab._pendingVariants = null;
            for (let i = tab.messages.length - 1; i >= 0; i--) {
                if (tab.messages[i].role === 'assistant') {
                    const fullVariants = [...variants, content];
                    tab.messages[i]._variants = fullVariants;
                    tab.messages[i]._variantIndex = fullVariants.length - 1;
                    msg = tab.messages[i];
                    break;
                }
            }
        }
        if (!msg) {
            const allMsgs = Array.from(document.querySelectorAll('#chat-messages .chat-message'));
            const idx = allMsgs.indexOf(el);
            const firstVisibleIdx = tab?.messages.findIndex(m => m.role !== 'system');
            const msgIdx = firstVisibleIdx + idx;
            msg = tab?.messages[msgIdx] || null;
        }
        const msgVariants = msg?._variants || [];
        const variantIdx = msg?._variantIndex || 0;
        const total = msgVariants.length || 1;
        const canGoLeft = msgVariants.length > 1 && variantIdx > 0;
        const canGoRight = true;


        setHtml(actions, `
          <button class="chat-action-btn" data-chat-action="copy" title="Copy">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2"/>
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
            </svg>
          </button>
          <button class="chat-action-btn" data-chat-action="nav-variant" data-variant-dir="-1" title="Previous response" ${canGoLeft ? '' : 'disabled'}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
          </button>
          <span class="chat-variant-badge">${variantIdx+1}/${total}</span>
          <button class="chat-action-btn" data-chat-action="nav-variant" data-variant-dir="1" title="${canGoRight && msgVariants.length <= 1 ? 'Regenerate' : 'Next response'}" ${canGoRight ? '' : 'disabled'}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
          <button class="chat-action-btn" data-chat-action="edit" title="Edit">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
              <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
              <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
            </svg>
          </button>
          <button class="chat-action-btn chat-action-btn-delete" data-chat-action="delete" title="Delete">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
          </button>`);
    } else if (actions && !content && tab?.messages.at(-1)?.role === 'user') {
        // Timeout or error on a fresh send — no content to show, offer retry and dismiss
        setHtml(actions, `
          <button class="chat-action-btn" data-chat-action="retry-send" title="Retry">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 1 0 .49-4.5"/>
            </svg>
          </button>
          <button class="chat-action-btn chat-action-btn-delete" data-chat-action="dismiss-error" title="Dismiss">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
            </svg>
          </button>`);
        const isMsgTimeout = el.querySelector('.chat-stopped') !== null;
        if (isMsgTimeout) {
            const t = showToastWithActions('Generation timed out', 'warning', 'Increase the timeout if your model needs more time to respond.', [{
                id: 'adjust-timeout',
                label: 'Adjust timeout',
                handler: () => { openTimeoutSetting(); t?.remove(); },
            }]);
        }
    }

    // Populate footer metadata (single line)
    const footer = el.querySelector('.chat-msg-footer');
    if (footer) {
        const modelName = lastLlamaMetrics?.model_name || '';
        const inp = usage ? (usage.prompt_tokens ?? 0) : 0;
        const out = usage ? (usage.completion_tokens ?? 0) : 0;
        const totalInput = tab ? (tab.total_input_tokens || 0) : inp;
        const totalOutput = tab ? (tab.total_output_tokens || 0) : out;
        const total = totalInput + totalOutput;
        const capacity = contextCapacityTokens || lastLlamaMetrics?.context_capacity_tokens || lastLlamaMetrics?.kv_cache_max || 0;
        // ctx% = (cumulative input + cumulative output) / capacity.
        // total_input_tokens and total_output_tokens are running tab-level sums updated each turn.
        const ctxTokens = total;
        const ctxPct = capacity > 0 && ctxTokens > 0 ? Math.min(100, Math.round((ctxTokens / capacity) * 100)) : 0;

        if (tab) tab.last_ctx_pct = ctxPct;

        const parts = [];
        if (inp > 0) parts.push(`↓${formatTokenCount(inp)}`);
        if (out > 0) parts.push(`↑${formatTokenCount(out)}`);
        if (total > 0) parts.push(`R${formatTokenCount(total)}`);
        if (ctxPct > 0) parts.push(`${ctxPct}% ctx · ${formatTokenCount(capacity)}`);
        if (modelName) parts.push(modelName);

        const metaEl = footer.querySelector('.chat-msg-meta-model');
        if (metaEl) {
            metaEl.textContent = parts.join(' · ');
            metaEl.title = '↓ = prompt tokens in · ↑ = tokens generated · R = running total · ctx = % of context window used';
        }
        const sepEl = footer.querySelector('.chat-msg-meta-sep');
        if (sepEl) sepEl.style.display = parts.length > 0 ? '' : 'none';
    }
}

// ── Message actions ───────────────────────────────────────────────────────────

function copyMessageContent(btn) {
    const body = btn.closest('.chat-bubble').querySelector('.chat-msg-body');
    navigator.clipboard.writeText(body.innerText).then(() => {
        btn.classList.add('chat-action-btn-copied');
        setTimeout(() => btn.classList.remove('chat-action-btn-copied'), 1500);
    });
}

function navigateVariant(btn, direction) {
    const msgEl = btn.closest('.chat-message');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const tab = activeChatTab();
    if (!tab || isNaN(msgIdx)) return;

    const msg = tab.messages[msgIdx];
    if (!msg || msg.role !== 'assistant') return;

    const variants = msg._variants || [];
    const curIdx = msg._variantIndex || 0;

    // Going right on the last variant (or only variant) → regenerate
    if (direction === 1 && (variants.length <= 1 || curIdx >= variants.length - 1)) {
        if (chat.busy) return;

        const newVariants = variants.length > 0 ? [...variants] : [msg.content];

        if (msg._quickGuideMeta && tab.messages[msgIdx - 1]?.role === 'assistant') {
            scheduleChatPersist();
            getTransport()?.regenerateQuickGuideReply?.(tab, msgIdx, msg._quickGuideMeta, newVariants)
                ?.then((result) => {
                    if (result?.message) {
                        result.message._quickGuideMeta = { ...msg._quickGuideMeta };
                    }
                });
            return;
        }

        // Find the user message immediately before this assistant message
        let userMsgIdx = -1;
        for (let i = msgIdx - 1; i >= 0; i--) {
            if (tab.messages[i].role === 'user') { userMsgIdx = i; break; }
        }
        if (userMsgIdx === -1) return;

        // Truncate to include the user message, remove all subsequent
        tab.messages = tab.messages.slice(0, userMsgIdx + 1);
        tab.updated_at = Date.now();

        tab._pendingVariants = newVariants;
        scheduleChatPersist();

        // User message is already in tab.messages — use sendChatResend
        getTransport()?.sendChatResend(tab);
        return;
    }

    if (!variants || variants.length <= 1) return;

    msg._variantIndex = Math.max(0, Math.min(variants.length - 1, curIdx + direction));
    msg.content = msg._variants[msg._variantIndex];
    tab.updated_at = Date.now();

    renderChatMessages();
    scheduleChatPersist();
}

function regenerateFromMessage(btn) {
    const msgEl = btn.closest('.chat-message');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const tab = activeChatTab();
    if (!tab || isNaN(msgIdx)) return;

    const msg = tab.messages[msgIdx];
    if (!msg || msg.role !== 'assistant') return;

    const variants = msg._variants && msg._variants.length > 0 ? [...msg._variants] : [msg.content];

    // Find the last user message before this assistant message
    const lastUser = [...tab.messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return;
    const userMsgIdx = tab.messages.indexOf(lastUser);

    // Truncate to include the user message, remove all subsequent
    tab.messages = tab.messages.slice(0, userMsgIdx + 1);
    tab.updated_at = Date.now();
    tab._pendingVariants = variants;
    scheduleChatPersist();

    // User message is already in tab.messages — use sendChatResend
    getTransport()?.sendChatResend(tab);
}

function editMessageContent(btn) {
    const msgEl = btn.closest('.chat-message');
    const body = msgEl.querySelector('.chat-msg-body');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const tab = activeChatTab();
    if (!tab || isNaN(msgIdx)) return;

    const msg = tab.messages[msgIdx];
    if (!msg) return;

    // Show "Save and Resend" for ALL user messages, not just the last one
    const resendBtn = msg.role === 'user'
        ? `<button class="chat-edit-btn chat-edit-btn-resend" data-chat-edit="resend">Save and Resend</button>`
        : '';

    setHtml(body, `<textarea class="chat-msg-edit-area" rows="6">${escapeHtml(msg.content)}</textarea>
      <div class="chat-msg-edit-actions">
        ${resendBtn}
        <button class="chat-edit-btn chat-edit-btn-save" data-chat-edit="save">Save</button>
        <button class="chat-edit-btn chat-edit-btn-cancel" data-chat-edit="cancel">Cancel</button>
      </div>`);
    const textarea = body.querySelector('.chat-msg-edit-area');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, window.innerHeight * 0.6) + 'px';
    textarea.focus();
    textarea.selectionStart = textarea.value.length;
}

function resendMessageEdit(btn) {
    const msgEl = btn.closest('.chat-message');
    const body = msgEl.querySelector('.chat-msg-body');
    const textarea = body.querySelector('.chat-msg-edit-area');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const tab = activeChatTab();
    if (!tab || !textarea || isNaN(msgIdx)) return;

    const msg = tab.messages[msgIdx];
    if (!msg || msg.role !== 'user') return;

    const newContent = textarea.value.trim();
    if (!newContent) return;

    msg.content = newContent;
    tab.updated_at = Date.now();

    // Truncate to include the user message, remove all subsequent messages
    tab.messages = tab.messages.slice(0, msgIdx + 1);
    scheduleChatPersist();

    // Use sendChatResend — the user message is already in tab.messages
    getTransport()?.sendChatResend(tab);
}

function saveMessageEdit(btn) {
    const msgEl = btn.closest('.chat-message');
    const body = msgEl.querySelector('.chat-msg-body');
    const textarea = body.querySelector('.chat-msg-edit-area');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const tab = activeChatTab();
    if (!tab || !textarea || isNaN(msgIdx)) return;

    const msg = tab.messages[msgIdx];
    if (!msg) return;

    const newContent = textarea.value.trim();
    if (newContent !== msg.content) {
        msg.content = newContent;
        tab.updated_at = Date.now();
        scheduleChatPersist();
    }
    // Update message in-place (safe during streaming — doesn't wipe other messages)

    setHtml(body, typeof renderMd === 'function' ? renderMd(msg.content) : escapeHtml(msg.content));
    if (msg.role === 'assistant') colorizeRpText(body);
    body.classList.add('chat-msg-body-rendered');
}

function cancelMessageEdit(btn) {
    const msgEl = btn.closest('.chat-message');
    const body = msgEl.querySelector('.chat-msg-body');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const msg = activeChatTab()?.messages[msgIdx];
    if (msg && body) {
        // Restore original content in-place (safe during streaming)

        setHtml(body, typeof renderMd === 'function' ? renderMd(msg.content) : escapeHtml(msg.content));
        if (msg.role === 'assistant') colorizeRpText(body);
        body.classList.add('chat-msg-body-rendered');
    }
}

function deleteMessage(btn) {
    const msgEl = btn.closest('.chat-message');
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    const tab = activeChatTab();
    if (!tab || isNaN(msgIdx) || msgIdx < 0 || msgIdx >= tab.messages.length) return;

    showConfirmDialog(
        'Delete message',
        'This will remove this message from the conversation.'
    ).then((ok) => {
        if (!ok) return;
        tab.messages.splice(msgIdx, 1);
        tab.updated_at = Date.now();
        scheduleChatPersist();
        // Do not re-render if AI is busy (would wipe streaming message from DOM)
        if (!chat.busy) renderChatMessages();
    });
}

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

function retrySend(btn) {
    const msgEl = btn.closest('.chat-message');
    const tab = activeChatTab();
    if (!tab || !msgEl) return;

    // Only remove error placeholder messages, not regular user messages
    if (msgEl.dataset.role === 'error') {
        msgEl.remove();
        getTransport()?.sendChatResend(tab);
        return;
    }

    // For user messages, truncate to this message and resend
    const msgIdx = parseInt(msgEl.dataset.msgIdx);
    if (!isNaN(msgIdx)) {
        tab.messages = tab.messages.slice(0, msgIdx + 1);
        tab.updated_at = Date.now();
        scheduleChatPersist();
    }

    getTransport()?.sendChatResend(tab);
}

function dismissError(btn) {
    btn.closest('.chat-message').remove();
}

// ── Export / Import ───────────────────────────────────────────────────────────

export function exportChatTab(format = 'md') {
    const tab = activeChatTab();
    if (!tab) return;

    if (format === 'json') {
        const data = JSON.stringify([normalizeTabForSave(tab)], null, 2);
        const blob = new Blob([data], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${tab.name.replace(/[^a-z0-9]/gi, '-').toLowerCase()}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
        return;
    }

    const md = tab.messages
        .filter(m => m.role !== 'system')
        .map(m => `**${m.role === 'user' ? 'You' : 'Assistant'}**\n\n${m.content}`)
        .join('\n\n---\n\n');
    const blob = new Blob([md], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${tab.name.replace(/[^a-z0-9]/gi, '-').toLowerCase()}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
}

export function importChatTab() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.md';
    input.onchange = e => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = ev => {
            try {
                if (file.name.endsWith('.json')) {
                    const data = JSON.parse(ev.target.result);
                    if (Array.isArray(data) && data.length > 0) {
                        const newTab = data[0];
                        newTab.id = crypto.randomUUID();
                        newTab.created_at = Date.now();
                        newTab.updated_at = Date.now();
                        chat.tabs.push(newTab);
                        Router.navigate('/chat/' + encodeURIComponent(newTab.id));
                        scheduleChatPersist();
                        showToast('Conversation imported', 'success');
                    }
                } else {
                    const lines = ev.target.result.split(/\n---\n/);
                    const messages = [];
                    for (const block of lines) {
                        const match = block.match(/\*\*(You|Assistant)\*\*\s*\n\n([\s\S]+)/);
                        if (match) {
                            messages.push({
                                role: match[1] === 'You' ? 'user' : 'assistant',
                                content: match[2].trim(),
                                timestamp_ms: Date.now(),
                            });
                        }
                    }
                    if (messages.length > 0) {
                        const tab = activeChatTab();
                        tab.messages = [...tab.messages, ...messages];
                        tab.updated_at = Date.now();
                        renderChatMessages();
                        scheduleChatPersist();
                        showToast(`Imported ${messages.length} messages`, 'success');
                    }
                }
            } catch (err) {
                showToast('Import failed: ' + err.message, 'error');
            }
        };
        reader.readAsText(file);
    };
    input.click();
}

// ── Tab rename ────────────────────────────────────────────────────────────────

export function startRenameTab(id) {
    const tabEl = document.querySelector(`.chat-tab[data-tab-id="${id}"] .chat-tab-name`);
    if (!tabEl) return;
    const orig = tabEl.textContent;
    tabEl.contentEditable = 'true';
    tabEl.focus();
    const range = document.createRange();
    range.selectNodeContents(tabEl);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
    const finish = () => {
        tabEl.contentEditable = 'false';
        renameChatTab(id, tabEl.textContent || orig);
    };
    tabEl.addEventListener('blur', finish, { once: true });
    tabEl.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); tabEl.blur(); }
        if (e.key === 'Escape') { tabEl.textContent = orig; tabEl.blur(); }
    }, { once: true });
}

// ── Badge ─────────────────────────────────────────────────────────────────────

export function updateChatTabBadge() {
    ensureChatElements();
    const tab = activeChatTab();
    const count = tab ? tab.messages.filter(m => m.role !== 'system').length : 0;
    if (sidebarBadgeChat) sidebarBadgeChat.textContent = count > 0 ? count : '';
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initChatRender() {
    // Call setup functions that bind DOM event listeners
    initChatScrollButton();

    // Event delegation for chat tab close buttons
    document.getElementById('chat-tab-bar')?.addEventListener('click', (e) => {
        const closeBtn = e.target.closest('.chat-tab-close');
        if (closeBtn) {
            closeChatTab(closeBtn.dataset.chatTabClose);
        }
    });

    // Event delegation for chat tab rename (dblclick)
    document.getElementById('chat-tab-bar')?.addEventListener('dblclick', (e) => {
        const renameEl = e.target.closest('[data-chat-tab-rename]');
        if (renameEl) {
            startRenameTab(renameEl.dataset.chatTabRename);
        }
    });

    // Event delegation for chat message action buttons
    document.getElementById('chat-messages')?.addEventListener('click', (e) => {
        const actionBtn = e.target.closest('[data-chat-action]');
        if (!actionBtn) return;
        const action = actionBtn.dataset.chatAction;
        const msgEl = actionBtn.closest('.chat-message');
        const isUserMessage = msgEl?.dataset.role === 'user';

        if (action === 'copy') copyMessageContent(actionBtn);
        else if (action === 'regenerate') regenerateFromMessage(actionBtn);
        else if (action === 'nav-variant') navigateVariant(actionBtn, +actionBtn.dataset.variantDir);
        else if (action === 'edit') editMessageContent(actionBtn);
        else if (action === 'delete') deleteMessage(actionBtn);
        else if (action === 'retry-send') {
            if (isUserMessage) retrySend(actionBtn);
        }
        else if (action === 'dismiss-error') dismissError(actionBtn);
    });

    // Event delegation for chat message edit buttons
    document.getElementById('chat-messages')?.addEventListener('click', (e) => {
        const editBtn = e.target.closest('[data-chat-edit]');
        if (!editBtn) return;
        const editAction = editBtn.dataset.chatEdit;
        if (editAction === 'resend') resendMessageEdit(editBtn);
        else if (editAction === 'save') saveMessageEdit(editBtn);
        else if (editAction === 'cancel') cancelMessageEdit(editBtn);
    });

    // Event delegation for suggested prompt buttons
    document.getElementById('chat-messages')?.addEventListener('click', (e) => {
        const promptBtn = e.target.closest('[data-prompt-text]');
        if (promptBtn) {
            getTransport()?.sendSuggestedPrompt?.(promptBtn.dataset.promptText);
        }
    });

    // Event delegation for compact marker toggle
    document.getElementById('chat-messages')?.addEventListener('click', (e) => {
        const pill = e.target.closest('[data-compact-toggle]');
        if (!pill) return;
        const marker = pill.closest('.chat-compact-marker');
        if (!marker) return;
        const body = marker.querySelector('.compact-marker-body');
        if (!body) return;
        const isExpanded = marker.dataset.expanded === 'true';
        body.style.display = isExpanded ? 'none' : 'block';
        marker.dataset.expanded = isExpanded ? 'false' : 'true';
    });

    document.getElementById('chat-header-hide-btn')?.addEventListener('click', () => {
        const tab = activeChatTab();
        if (tab) {
            hideChatTab(tab.id);
        }
    });

    registerChatViewBindings({
        renderChatTabs,
        renderChatMessages,
        renderChatSessionsSidebar,
        updateChatTabBadge,
    });
}
