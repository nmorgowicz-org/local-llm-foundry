// Chat Sessions Sidebar
// Renders and manages the left session panel inside #page-chat.
// Activated when the Chat nav item is selected; hidden otherwise.

import { showPromptDialog } from './toast.js';
import { chat } from '../core/app-state.js';
import {
  closeChatTab,
  addChatTab,
  renameChatTab,
  togglePinTab,
  archiveChatTab,
  hideChatTab,
  restoreChatTab,
  setChatTabVisibility,
  duplicateChatTab,
  deleteManyChatTabs,
  archiveManyChatTabs,
} from './chat-state.js';
import Router from './router.js';

const CSP_COLLAPSED_KEY = 'csp-collapsed';

// Lifecycle

export function initChatSessionsSidebar() {
    const newBtn    = document.getElementById('csp-new-btn');
    const collapseBtn = document.getElementById('csp-collapse-btn');
    const searchEl  = document.getElementById('csp-search');
    const strip = document.getElementById('csp-collapsed-strip');

    newBtn?.addEventListener('click', () => addChatTab());
    collapseBtn?.addEventListener('click', toggleSessionPanelCollapse);

    // Make whole strip clickable to expand
    strip?.addEventListener('click', (e) => {
        e.stopPropagation();
        expandSessionPanel();
    });

    searchEl?.addEventListener('input', () => {
        const q = searchEl.value.trim().toLowerCase();
        _applySearchFilter(q);
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.csp-context-menu')) _dismissContextMenu();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') _dismissContextMenu();
    });

    if (localStorage.getItem(CSP_COLLAPSED_KEY) === 'true') {
        const panel = document.getElementById('chat-sessions-panel');
        panel?.classList.add('collapsed');
    }

    updateCollapsedLabel();
}

// Show / Hide (called from nav.js)

export function showSessionPanel() {
    const panel = document.getElementById('chat-sessions-panel');
    if (!panel) return;
    panel.classList.add('visible');
    // Respect user's last collapse preference
    const shouldStayCollapsed = localStorage.getItem(CSP_COLLAPSED_KEY) === 'true';
    if (!shouldStayCollapsed) {
        panel.classList.remove('collapsed');
    }
}

export function hideSessionPanel() {
    const panel = document.getElementById('chat-sessions-panel');
    if (!panel) return;
    panel.classList.remove('visible');
}

export function expandSessionPanel() {
    const panel = document.getElementById('chat-sessions-panel');
    if (!panel) return;
    panel.classList.add('visible');
    panel.classList.remove('collapsed');
    localStorage.setItem(CSP_COLLAPSED_KEY, 'false');
}

export function toggleSessionPanelCollapse() {
    const panel = document.getElementById('chat-sessions-panel');
    if (!panel) return;
    const collapsed = panel.classList.toggle('collapsed');
    localStorage.setItem(CSP_COLLAPSED_KEY, collapsed.toString());

    const icon = document.querySelector('#csp-collapse-btn svg');
    if (icon) {
        icon.style.transform = collapsed ? 'rotate(180deg)' : '';
    }

    updateCollapsedLabel();
}

function updateCollapsedLabel() {
    const label = document.getElementById('csp-collapsed-label');
    if (!label) return;
    const tab = (chat.tabs || []).find(t => t.id === chat.activeTabId);
    if (!tab || tab.visibility !== 'active') {
        label.textContent = 'Conversations';
    } else {
        label.textContent = tab.name || 'Conversations';
    }
}

function _renderManagementPills() {
    const container = document.getElementById('csp-management-row');
    if (!container) return;

    const archivedCount = chat.tabs.filter(t => t.visibility === 'archived').length;
    const hiddenCount = chat.tabs.filter(t => t.visibility === 'hidden').length;
    const selectedIds = chat.visibilityUi?.selectedIds || new Set();

    container.innerHTML = '';

    // If items selected, show bulk actions.
    if (selectedIds.size > 0) {
        const bulkWrap = document.createElement('div');
        bulkWrap.className = 'csp-bulk-row';

        const countSpan = document.createElement('span');
        countSpan.className = 'csp-bulk-count';
        countSpan.textContent = selectedIds.size + ' selected';
        bulkWrap.appendChild(countSpan);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'csp-bulk-btn csp-bulk-btn-danger';
        deleteBtn.type = 'button';
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const ids = [...selectedIds];
            selectedIds.clear();
            deleteManyChatTabs(ids);
            renderChatSessionsSidebar();
        });
        bulkWrap.appendChild(deleteBtn);

        const archiveBtn = document.createElement('button');
        archiveBtn.className = 'csp-bulk-btn';
        archiveBtn.type = 'button';
        archiveBtn.textContent = 'Archive';
        archiveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const ids = [...selectedIds];
            selectedIds.clear();
            archiveManyChatTabs(ids);
            renderChatSessionsSidebar();
        });
        bulkWrap.appendChild(archiveBtn);

        const clearBtn = document.createElement('button');
        clearBtn.className = 'csp-bulk-btn';
        clearBtn.type = 'button';
        clearBtn.textContent = 'Clear';
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            selectedIds.clear();
            renderChatSessionsSidebar();
        });
        bulkWrap.appendChild(clearBtn);

        container.appendChild(bulkWrap);
        return;
    }

    const archivePill = document.createElement('button');
    archivePill.className = 'csp-management-pill';
    archivePill.type = 'button';
    archivePill.setAttribute('aria-label', 'Show archived chats');
    const archiveLabel = document.createElement('span');
    archiveLabel.textContent = 'Archived';
    archivePill.appendChild(archiveLabel);
    if (archivedCount > 0) {
        const archiveCount = document.createElement('span');
        archiveCount.className = 'csp-pill-count';
        archiveCount.textContent = archivedCount;
        archivePill.appendChild(archiveCount);
    }
    if (chat.visibilityUi.archiveOpen) {
        archivePill.classList.add('active');
        archivePill.setAttribute('aria-pressed', 'true');
    }
    archivePill.addEventListener('click', () => {
        chat.visibilityUi.archiveOpen = !chat.visibilityUi.archiveOpen;
        renderChatSessionsSidebar();
    });
    container.appendChild(archivePill);

    const hiddenPill = document.createElement('button');
    hiddenPill.className = 'csp-management-pill';
    hiddenPill.type = 'button';
    hiddenPill.setAttribute('aria-label', 'Show hidden chats');
    const hiddenLabel = document.createElement('span');
    hiddenLabel.textContent = 'Hidden';
    hiddenPill.appendChild(hiddenLabel);
    if (hiddenCount > 0) {
        const hiddenCountEl = document.createElement('span');
        hiddenCountEl.className = 'csp-pill-count';
        hiddenCountEl.textContent = hiddenCount;
        hiddenPill.appendChild(hiddenCountEl);
    }
    if (chat.visibilityUi.hiddenOpen) {
        hiddenPill.classList.add('active');
        hiddenPill.setAttribute('aria-pressed', 'true');
    }
    hiddenPill.addEventListener('click', () => {
        chat.visibilityUi.hiddenOpen = !chat.visibilityUi.hiddenOpen;
        renderChatSessionsSidebar();
    });
    container.appendChild(hiddenPill);
}

// Render

export function renderChatSessionsSidebar() {
    const list = document.getElementById('csp-list');
    if (!list) return;

    const selectedIds = chat.visibilityUi.selectedIds || new Set();
    // Clear selections for deleted tabs.
    const tabIds = new Set(chat.tabs.map(t => t.id));
    for (const id of selectedIds) {
        if (!tabIds.has(id)) selectedIds.delete(id);
    }

    const activeTabs = chat.tabs.filter(t => t.visibility === 'active');
    const groups = _groupTabsByRecency(activeTabs);
    const activeId = chat.activeTabId;

    const sections = [
        { key: 'pinned',    label: 'Pinned' },
        { key: 'today',     label: 'Today' },
        { key: 'yesterday', label: 'Yesterday' },
        { key: 'week',      label: 'This Week' },
        { key: 'older',     label: 'Older' },
    ];

    const frag = document.createDocumentFragment();

    for (const { key, label } of sections) {
        const tabs = groups[key];
        if (!tabs || tabs.length === 0) continue;

        const hdr = document.createElement('div');
        hdr.className = 'csp-section-header';
        hdr.textContent = label;
        frag.appendChild(hdr);

        for (const tab of tabs) {
            frag.appendChild(_buildSessionItem(tab, tab.id === activeId, selectedIds.has(tab.id)));
        }
    }

    list.innerHTML = '';
    list.appendChild(frag);

    if (chat.visibilityUi.archiveOpen) {
        _renderArchivedSection(list);
    }

    if (chat.visibilityUi.hiddenOpen) {
        _renderHiddenSection(list);
    }

    _applySearchFilter(document.getElementById('csp-search')?.value.trim().toLowerCase() || '');
    updateCollapsedLabel();
    _renderManagementPills();
}

export function updateSessionItem(tabId) {
    const list = document.getElementById('csp-list');
    const existing = list?.querySelector(`.csp-item[data-tab-id="${tabId}"]`);
    if (!existing) return;

    const tab = (chat.tabs || []).find(t => t.id === tabId);
    if (!tab) { existing.remove(); return; }

    const isActive = tab.id === chat.activeTabId;
    const selectedIds = chat.visibilityUi?.selectedIds;
    const isSelected = selectedIds?.has(tabId) || false;
    const fresh = _buildSessionItem(tab, isActive, isSelected);
    existing.replaceWith(fresh);
}

// Item builder

function _buildSessionItem(tab, isActive, isSelected) {
    const el = document.createElement('div');
    const ctxPct = tab.lastCtxPct || 0;
    const ctxLevel = ctxPct >= 90 ? 'critical' : ctxPct >= 75 ? 'high' : ctxPct >= 50 ? 'medium' : 'low';
    const msgCount = tab._loaded
        ? (tab.messages || []).filter(m => m.role !== 'system').length
        : (tab.message_count || 0);
    const initial = (tab.name || '?').charAt(0).toUpperCase();
    const hue = _avatarHue(tab.id);

    el.className = 'csp-item' + (isActive ? ' active' : '') + (isSelected ? ' selected' : '');
    el.dataset.tabId = tab.id;
    el.dataset.ctx = ctxLevel;
    if (tab.pinned) el.dataset.pinned = 'true';
    el.setAttribute('role', 'button');
    el.setAttribute('tabindex', '0');
    el.setAttribute('aria-current', isActive ? 'true' : 'false');

    el.innerHTML =
        `<button class="csp-item-checkbox" type="button" aria-label="Select chat"></button>` +
        `<div class="csp-item-avatar"><span></span></div>` +
        `<div class="csp-item-body">` +
            `<div class="csp-item-name"></div>` +
            `<div class="csp-item-meta">` +
                `<span class="csp-item-persona"></span>` +
                `<span class="csp-item-explicit"></span>` +
                `<span class="csp-item-count"></span>` +
            `</div>` +
            `<div class="csp-item-ctx-bar">` +
                `<div class="csp-item-ctx-fill"></div>` +
            `</div>` +
        `</div>` +
        `<div class="csp-item-actions">` +
            `<button class="csp-item-action-btn" data-action="pin"></button>` +
            `<button class="csp-item-action-btn" data-action="more">\u22EF</button>` +
        `</div>`;

    const avatarSpan = el.querySelector('.csp-item-avatar span');
    if (avatarSpan) avatarSpan.textContent = initial;

    const nameEl = el.querySelector('.csp-item-name');
    if (nameEl) nameEl.textContent = tab.name || 'Untitled';

    const personaEl = el.querySelector('.csp-item-persona');
    if (personaEl) {
        if (tab.active_template_id) {
            personaEl.dataset.templateId = tab.active_template_id;
            personaEl.textContent = '\u2026';
        } else {
            personaEl.textContent = 'Default';
        }
    }

    const explicitEl = el.querySelector('.csp-item-explicit');
    if (explicitEl) {
        explicitEl.dataset.level = String(tab.explicit_level || 0);
    }

    const countEl = el.querySelector('.csp-item-count');
    if (countEl) {
        if (msgCount) {
            countEl.textContent = msgCount + ' msg' + (msgCount !== 1 ? 's' : '');
        } else {
            countEl.style.display = 'none';
        }
    }

    const pinBtn = el.querySelector('button[data-action="pin"]');
    if (pinBtn) {
        pinBtn.textContent = tab.pinned ? '\u{1F4CC}' : '\u2299';
        pinBtn.title = tab.pinned ? 'Unpin' : 'Pin';
    }

    const avatar = el.querySelector('.csp-item-avatar');
    if (avatar) {
        avatar.style.setProperty('--avatar-hue', String(hue));
    }

    const ctxFill = el.querySelector('.csp-item-ctx-fill');
    if (ctxFill) {
        ctxFill.style.transform = 'scaleX(' + (ctxPct / 100) + ')';
    }

    const cb = el.querySelector('.csp-item-checkbox');
    if (cb && isSelected) {
        cb.classList.add('checked');
    }

    if (tab.active_template_id) {
        _resolvePersonaLabel(el, tab.active_template_id);
    }

    // Selection helpers
    const selectedIds = chat.visibilityUi?.selectedIds || new Set();

    const selectTab = (id, e) => {
        if (e && (e.ctrlKey || e.metaKey)) {
            // Ctrl/Cmd+click: toggle this tab without clearing others.
            if (selectedIds.has(id)) {
                selectedIds.delete(id);
            } else {
                selectedIds.add(id);
            }
        } else if (selectedIds.size > 0 && !selectedIds.has(id)) {
            // If selections exist and this tab is not selected:
            // single-click selects only this tab and switches to it.
            selectedIds.clear();
            selectedIds.add(id);
        } else if (selectedIds.size > 0 && selectedIds.has(id)) {
            // Already selected and >0 selections: keep selections, switch tab.
        } else {
            // No selections: just switch tab.
            selectedIds.clear();
        }
        Router.navigate('/chat/' + encodeURIComponent(id));
        renderChatSessionsSidebar();
    };

    // Checkbox click: toggle selection only.
    cb?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (selectedIds.has(tab.id)) {
            selectedIds.delete(tab.id);
        } else {
            selectedIds.add(tab.id);
        }
        renderChatSessionsSidebar();
    });

    el.addEventListener('click', (e) => {
        // If clicking checkbox, handled above.
        if (e.target.closest('.csp-item-checkbox')) return;

        const actionBtn = e.target.closest('[data-action]');
        if (actionBtn) {
            e.stopPropagation();
            const action = actionBtn.dataset.action;
            if (action === 'pin') {
                togglePinTab(tab.id);
                renderChatSessionsSidebar();
            } else if (action === 'more') {
                _showContextMenu(tab, actionBtn);
            }
            return;
        }
        selectTab(tab.id, e);
    });

    el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            selectTab(tab.id, e);
        }
    });

    return el;
}

// Context menu

function _renderArchivedSection(list) {
    const archived = chat.tabs.filter(t => t.visibility === 'archived');
    if (archived.length === 0) return;

    const header = document.createElement('div');
    header.className = 'csp-section-header';
    header.textContent = 'ARCHIVED';
    list.appendChild(header);

    archived.forEach(tab => {
        const item = _buildArchivedItem(tab);
        list.appendChild(item);
    });
}

function _buildArchivedItem(tab) {
    const item = document.createElement('div');
    item.className = 'csp-item csp-item-archived';
    item.dataset.tabId = tab.id;

    const body = document.createElement('div');
    body.className = 'csp-item-body';

    const name = document.createElement('div');
    name.className = 'csp-item-name';
    name.textContent = tab.name || 'Untitled';
    body.appendChild(name);

    const meta = document.createElement('div');
    meta.className = 'csp-item-meta';

    const count = document.createElement('span');
    count.className = 'csp-item-count';
    const archivedCount = tab._loaded ? (tab.messages || []).length : (tab.message_count || 0);
    count.textContent = archivedCount || '';
    meta.appendChild(count);
    body.appendChild(meta);

    item.appendChild(body);

    const actions = document.createElement('div');
    actions.className = 'csp-item-actions';

    const restoreBtn = document.createElement('button');
    restoreBtn.className = 'csp-item-action-btn';
    restoreBtn.dataset.action = 'restore';
    restoreBtn.setAttribute('aria-label', 'Restore chat');
    restoreBtn.textContent = '\u2197';
    restoreBtn.style.cssText = 'font-size:14px;';
    actions.appendChild(restoreBtn);

    const moreBtn = document.createElement('button');
    moreBtn.className = 'csp-item-action-btn';
    moreBtn.dataset.action = 'more';
    moreBtn.setAttribute('aria-label', 'More actions');
    moreBtn.textContent = '\u22EF';
    actions.appendChild(moreBtn);

    item.appendChild(actions);

    item.addEventListener('click', (e) => {
        const action = e.target.dataset.action;
        if (action) {
            e.stopPropagation();
            _handleArchivedAction(tab, action);
        }
    });

    item.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        _showContextMenu(tab, e.target);
    });

    return item;
}

function _handleArchivedAction(tab, action) {
    switch (action) {
        case 'restore':
            restoreChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
        case 'hide':
            hideChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
        case 'delete':
            closeChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
    }
}

function _renderHiddenSection(list) {
    const hidden = chat.tabs.filter(t => t.visibility === 'hidden');
    if (hidden.length === 0) return;

    const header = document.createElement('div');
    header.className = 'csp-section-header';
    header.textContent = 'HIDDEN';
    list.appendChild(header);

    if (!chat.visibilityUi.hiddenRevealed) {
        const reveal = document.createElement('button');
        reveal.className = 'csp-reveal-btn';
        reveal.type = 'button';
        reveal.setAttribute('aria-label', 'Reveal hidden chats');
        const revealText = document.createElement('span');
        revealText.textContent = 'Reveal hidden chats';
        reveal.appendChild(revealText);
        reveal.addEventListener('click', () => {
            chat.visibilityUi.hiddenRevealed = true;
            renderChatSessionsSidebar();
        });
        list.appendChild(reveal);
        return;
    }

    hidden.forEach(tab => {
        const item = _buildHiddenItem(tab);
        list.appendChild(item);
    });
}

function _buildHiddenItem(tab) {
    const item = document.createElement('div');
    item.className = 'csp-item csp-item-hidden';
    item.dataset.tabId = tab.id;

    const body = document.createElement('div');
    body.className = 'csp-item-body';

    const name = document.createElement('div');
    name.className = 'csp-item-name';
    name.textContent = tab.name || 'Untitled';
    body.appendChild(name);

    const meta = document.createElement('div');
    meta.className = 'csp-item-meta';

    const count = document.createElement('span');
    count.className = 'csp-item-count';
    const hiddenCount = tab._loaded ? (tab.messages || []).length : (tab.message_count || 0);
    count.textContent = hiddenCount || '';
    meta.appendChild(count);
    body.appendChild(meta);

    item.appendChild(body);

    const actions = document.createElement('div');
    actions.className = 'csp-item-actions';

    const restoreBtn = document.createElement('button');
    restoreBtn.className = 'csp-item-action-btn';
    restoreBtn.dataset.action = 'restore';
    restoreBtn.setAttribute('aria-label', 'Restore chat');
    restoreBtn.textContent = '\u2197';
    restoreBtn.style.cssText = 'font-size:14px;';
    actions.appendChild(restoreBtn);

    const moreBtn = document.createElement('button');
    moreBtn.className = 'csp-item-action-btn';
    moreBtn.dataset.action = 'more';
    moreBtn.setAttribute('aria-label', 'More actions');
    moreBtn.textContent = '\u22EF';
    actions.appendChild(moreBtn);

    item.appendChild(actions);

    item.addEventListener('click', (e) => {
        const action = e.target.dataset.action;
        if (action) {
            e.stopPropagation();
            _handleHiddenAction(tab, action);
        }
    });

    item.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        _showContextMenu(tab, e.target);
    });

    return item;
}

function _handleHiddenAction(tab, action) {
    switch (action) {
        case 'restore':
            restoreChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
        case 'archive':
            setChatTabVisibility(tab.id, 'archived');
            renderChatSessionsSidebar();
            break;
        case 'delete':
            closeChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
    }
}

let _activeMenu = null;

function _showContextMenu(tab, anchorEl) {
    _dismissContextMenu();

    const menu = document.createElement('div');
    menu.className = 'csp-context-menu';
    menu.setAttribute('role', 'menu');
    menu.setAttribute('tabindex', '-1');

    const isHidden = tab.visibility === 'hidden';
    const items = tab.visibility === 'archived' ? [
        { label: 'Restore',         action: 'restore' },
        { label: 'Hide',            action: 'hide' },
        { separator: true },
        { label: 'Export JSON',     action: 'export-json' },
        { label: 'Export Markdown', action: 'export-md' },
        { label: 'Duplicate',       action: 'duplicate' },
        { separator: true },
        { label: 'Delete',          action: 'delete', danger: true },
    ] : isHidden ? [
        { label: 'Restore',         action: 'restore' },
        { label: 'Archive',         action: 'archive' },
        { separator: true },
        { label: 'Export JSON',     action: 'export-json' },
        { label: 'Export Markdown', action: 'export-md' },
        { label: 'Duplicate',       action: 'duplicate' },
        { separator: true },
        { label: 'Delete',          action: 'delete', danger: true },
    ] : [
        { label: 'Rename',          action: 'rename' },
        { label: tab.pinned ? 'Unpin' : 'Pin', action: 'pin' },
        { label: 'Archive',         action: 'archive' },
        { label: 'Hide',            action: 'hide' },
        { separator: true },
        { label: 'Export JSON',     action: 'export-json' },
        { label: 'Export Markdown', action: 'export-md' },
        { label: 'Duplicate',       action: 'duplicate' },
        { separator: true },
        { label: 'Delete',          action: 'delete', danger: true },
    ];

    for (const item of items) {
        if (item.separator) {
            const sep = document.createElement('div');
            sep.className = 'csp-context-menu-separator';
            menu.appendChild(sep);
            continue;
        }
        const el = document.createElement('div');
        el.className = 'csp-context-menu-item' + (item.danger ? ' danger' : '');
        el.textContent = item.label;
        el.setAttribute('role', 'menuitem');
        el.addEventListener('click', async (e) => {
            e.stopPropagation();
            _dismissContextMenu();
            await _handleContextAction(tab, item.action);
        });
        menu.appendChild(el);
    }

    document.body.appendChild(menu);
    _activeMenu = menu;

    const rect = anchorEl.getBoundingClientRect();
    const menuRect = menu.getBoundingClientRect();
    const menuW = 170;
    const left = Math.min(rect.right + 4, window.innerWidth - menuW - 8);
    let top = rect.top;

    // If menu would overflow the bottom, position above the anchor.
    if (top + menuRect.height > window.innerHeight - 8) {
        top = rect.bottom - menuRect.height - 4;
    }

    // Clamp top so it never goes off-screen.
    if (top < 8) top = 8;

    menu.style.left = left + 'px';
    menu.style.top  = top + 'px';
    menu.focus();
}

function _dismissContextMenu() {
    _activeMenu?.remove();
    _activeMenu = null;
}

async function _handleContextAction(tab, action) {
    switch (action) {
        case 'restore':
            restoreChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
        case 'archive':
            if (tab.visibility === 'hidden') {
                setChatTabVisibility(tab.id, 'archived');
            } else {
                archiveChatTab(tab.id);
            }
            renderChatSessionsSidebar();
            break;
        case 'hide':
            hideChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
        case 'rename': {
            const newName = await showPromptDialog('Rename conversation', 'Enter a new name:', tab.name);
            if (newName && newName.trim()) {
                renameChatTab(tab.id, newName.trim());
                renderChatSessionsSidebar();
            }
            break;
        }
        case 'pin':
            togglePinTab(tab.id);
            renderChatSessionsSidebar();
            break;
        case 'export-json':
            // Delegate to existing export handler on window
            window.exportChatTab?.('json');
            break;
        case 'export-md':
            window.exportChatTab?.('md');
            break;
        case 'duplicate': {
            duplicateChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
        }
        case 'delete':
            closeChatTab(tab.id);
            renderChatSessionsSidebar();
            break;
    }
}

// Search filter

function _applySearchFilter(q) {
    const list = document.getElementById('csp-list');
    if (!list) return;

    list.querySelectorAll('.csp-item').forEach(el => {
        if (!q) { el.style.display = ''; return; }
        const name = el.querySelector('.csp-item-name')?.textContent.toLowerCase() || '';
        const persona = el.querySelector('.csp-item-persona')?.textContent.toLowerCase() || '';
        el.style.display = (name.includes(q) || persona.includes(q)) ? '' : 'none';
    });

    list.querySelectorAll('.csp-section-header').forEach(hdr => {
        let next = hdr.nextElementSibling;
        let allHidden = true;
        while (next && !next.classList.contains('csp-section-header')) {
            if (next.classList.contains('csp-item') && next.style.display !== 'none') {
                allHidden = false; break;
            }
            next = next.nextElementSibling;
        }
        hdr.style.display = allHidden ? 'none' : '';
    });
}

// Grouping & utilities

function _groupTabsByRecency(tabs) {
    const now = Date.now();

    // Local-date helpers so "Today/Yesterday/This Week" match the user's calendar.
    const startOfDay = (date) => {
        const d = new Date(date);
        d.setHours(0, 0, 0, 0);
        return d.getTime();
    };

    const todayStart = startOfDay(now);
    const yesterdayStart = todayStart - 86400000;
    const weekStart = yesterdayStart - 5 * 86400000; // 7 days including today and yesterday

    const groups = { pinned: [], today: [], yesterday: [], week: [], older: [] };

    // Use last_message_at (server-computed max message timestamp) for grouping
    // so that parameter/settings changes don't push tabs into "Today" when no
    // new messages were sent.  Fall back to updated_at → created_at → now for
    // tabs that have never had a message.
    const tabTs = (tab) =>
        tab.last_message_at || tab.updated_at || tab.created_at || now;

    for (const tab of tabs) {
        if (tab.pinned) { groups.pinned.push(tab); continue; }
        const ts = tabTs(tab);
        if (ts >= todayStart)            groups.today.push(tab);
        else if (ts >= yesterdayStart)   groups.yesterday.push(tab);
        else if (ts >= weekStart)        groups.week.push(tab);
        else                             groups.older.push(tab);
    }

    // Sort each group by most-recently-messaged first (same priority as grouping).
    const byRecency = (a, b) => tabTs(b) - tabTs(a);

    groups.pinned.sort(byRecency);
    groups.today.sort(byRecency);
    groups.yesterday.sort(byRecency);
    groups.week.sort(byRecency);
    groups.older.sort(byRecency);

    return groups;
}

function _avatarHue(id) {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) & 0xFFFF;
    return h % 360;
}

async function _resolvePersonaLabel(itemEl, templateId) {
    const span = itemEl.querySelector('.csp-item-persona');
    if (!span) return;
    const templates = await window.loadTemplates?.();
    const tmpl = templates?.find(t => t.id === templateId);
    span.textContent = tmpl?.name || '';
}
