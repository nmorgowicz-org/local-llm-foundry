// ── Toast ─────────────────────────────────────────────────────────────────────
// Toast notifications, progress toasts, and action toasts.

import { escapeHtml } from '../core/format.js';

const TOAST_AUTO_DISMISS = 6000;
const NOTIFICATIONS_STORAGE_KEY = 'llama-monitor-notifications';
const MAX_ACTIVE_NOTIFICATIONS = 5;
const MAX_ARCHIVED_NOTIFICATIONS = 50;

let notificationStateLoaded = false;
let activeNotifications = new Map();
let archivedNotifications = [];
let notificationTab = 'active';
const notificationHandlers = new Map();

function notificationPriority(type) {
    return { error: 3, warning: 2, info: 1, success: 0 }[type] || 0;
}

function notificationHandlerKey(notificationId, actionId) {
    return `${notificationId}:${actionId}`;
}

function ensureNotificationState() {
    if (notificationStateLoaded) return;
    notificationStateLoaded = true;
    try {
        const saved = JSON.parse(localStorage.getItem(NOTIFICATIONS_STORAGE_KEY) || '{}');
        activeNotifications = new Map(
            Array.isArray(saved.active)
                ? saved.active.filter(item => item?.id).map(item => [item.id, item])
                : [],
        );
        archivedNotifications = Array.isArray(saved.archived)
            ? saved.archived.filter(item => item?.id).slice(0, MAX_ARCHIVED_NOTIFICATIONS)
            : [];
    } catch {
        activeNotifications = new Map();
        archivedNotifications = [];
    }
}

function saveNotificationState() {
    try {
        localStorage.setItem(NOTIFICATIONS_STORAGE_KEY, JSON.stringify({
            active: [...activeNotifications.values()],
            archived: archivedNotifications,
        }));
    } catch {
        // Notification history is best effort and must never block the app.
    }
}

function archiveNotification(record, reason) {
    activeNotifications.delete(record.id);
    archivedNotifications = [
        { ...record, archivedAt: Date.now(), archiveReason: reason },
        ...archivedNotifications.filter(item => item.id !== record.id),
    ].slice(0, MAX_ARCHIVED_NOTIFICATIONS);
}

function enforceActiveLimit() {
    const ranked = [...activeNotifications.values()].sort((a, b) => (
        notificationPriority(b.type) - notificationPriority(a.type)
        || (b.updatedAt || b.createdAt || 0) - (a.updatedAt || a.createdAt || 0)
    ));
    ranked.slice(MAX_ACTIVE_NOTIFICATIONS).forEach(record => {
        archiveNotification(record, 'Archived because it is outside the top active issues.');
    });
}

function formatNotificationTime(timestamp) {
    if (!timestamp) return 'Unknown time';
    const elapsed = Math.max(0, Date.now() - timestamp);
    if (elapsed < 60_000) return 'Just now';
    if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
    if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
    return new Intl.DateTimeFormat(undefined, {
        month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
    }).format(timestamp);
}

function createNotificationButton(label, className, handler) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = className;
    button.textContent = label;
    button.addEventListener('click', handler);
    return button;
}

function invokeNotificationAction(record, action) {
    closeNotificationMenu();
    const handler = notificationHandlers.get(notificationHandlerKey(record.id, action.id));
    if (handler) handler();
}

export function closeNotificationMenu() {
    const menu = document.getElementById('nav-notifications-menu');
    const button = document.getElementById('nav-notifications-btn');
    if (!menu || !button) return;
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    button.closest('.top-nav-bar')?.classList.remove('nav-notifications-open');
}

function isOpenModalOverlay(element) {
    return element.classList.contains('open')
        || element.classList.contains('active')
        || (element.style.display && element.style.display !== 'none');
}

function renderNotificationCenter() {
    const root = document.getElementById('nav-notifications');
    const list = document.getElementById('nav-notifications-list');
    if (!root || !list) return;
    ensureNotificationState();

    const active = [...activeNotifications.values()].sort((a, b) => (
        notificationPriority(b.type) - notificationPriority(a.type)
        || (b.updatedAt || b.createdAt || 0) - (a.updatedAt || a.createdAt || 0)
    ));
    const records = notificationTab === 'active' ? active : archivedNotifications;
    const badge = document.getElementById('nav-notifications-badge');
    const activeCount = document.getElementById('nav-notifications-active-count');
    const archivedCount = document.getElementById('nav-notifications-archived-count');
    const clearArchived = document.getElementById('nav-notifications-clear');
    if (badge) {
        badge.hidden = active.length === 0;
        badge.textContent = active.length > 9 ? '9+' : String(active.length);
    }
    if (activeCount) activeCount.textContent = active.length ? `(${active.length})` : '';
    if (archivedCount) archivedCount.textContent = archivedNotifications.length ? `(${archivedNotifications.length})` : '';
    if (clearArchived) clearArchived.disabled = archivedNotifications.length === 0;

    document.querySelectorAll('[data-notification-tab]').forEach(tab => {
        const selected = tab.dataset.notificationTab === notificationTab;
        tab.classList.toggle('active', selected);
        tab.setAttribute('aria-selected', String(selected));
    });

    list.replaceChildren();
    if (!records.length) {
        const empty = document.createElement('div');
        empty.className = 'nav-notifications-empty';
        empty.textContent = notificationTab === 'active'
            ? 'No active issues.'
            : 'No archived notifications yet.';
        list.appendChild(empty);
        return;
    }

    records.forEach(record => {
        const item = document.createElement('article');
        item.className = `nav-notification-item nav-notification-item--${record.type || 'info'}`;

        const heading = document.createElement('div');
        heading.className = 'nav-notification-heading';
        const icon = document.createElement('span');
        icon.className = 'nav-notification-icon';
        icon.textContent = getToastIcon(record.type);
        icon.setAttribute('aria-hidden', 'true');
        const title = document.createElement('strong');
        title.className = 'nav-notification-title';
        title.textContent = record.title || 'Notification';
        const time = document.createElement('time');
        time.className = 'nav-notification-time';
        time.dateTime = record.createdAt ? new Date(record.createdAt).toISOString() : '';
        time.textContent = formatNotificationTime(record.createdAt);
        heading.append(icon, title, time);

        const message = document.createElement('p');
        message.className = 'nav-notification-message';
        message.textContent = record.message || '';

        const footer = document.createElement('div');
        footer.className = 'nav-notification-footer';
        const actions = document.createElement('div');
        actions.className = 'nav-notification-actions';
        (record.actions || []).forEach(action => {
            const handler = notificationHandlers.get(notificationHandlerKey(record.id, action.id));
            const button = createNotificationButton(
                action.label,
                action.primary ? 'nav-notification-action primary' : 'nav-notification-action',
                () => invokeNotificationAction(record, action),
            );
            button.disabled = !handler;
            button.title = handler ? '' : 'This action will be available when the related feature is ready.';
            actions.appendChild(button);
        });
        if (notificationTab === 'active') {
            actions.appendChild(createNotificationButton(
                'Archive',
                'nav-notification-action nav-notification-action--quiet',
                () => {
                    archiveNotification(record, 'Archived by user.');
                    saveNotificationState();
                    renderNotificationCenter();
                },
            ));
        }
        footer.appendChild(actions);
        if (record.archiveReason) {
            const reason = document.createElement('span');
            reason.className = 'nav-notification-archive-reason';
            reason.textContent = record.archiveReason;
            footer.appendChild(reason);
        }

        item.append(heading, message);
        if ((record.actions || []).length || record.archiveReason) item.appendChild(footer);
        list.appendChild(item);
    });
}

function registerPersistentNotification(id, title, type, message, actions) {
    ensureNotificationState();
    const existing = activeNotifications.get(id);
    const now = Date.now();
    const record = {
        id,
        title: title || 'Notification',
        type: type || 'info',
        message: message || '',
        actions: actions.map(action => ({
            id: action.id,
            label: action.label,
            primary: !!action.primary,
        })),
        createdAt: existing?.createdAt || now,
        updatedAt: now,
    };
    actions.forEach(action => {
        if (typeof action.handler === 'function') {
            notificationHandlers.set(notificationHandlerKey(id, action.id), action.handler);
        }
    });
    archivedNotifications = archivedNotifications.filter(item => item.id !== id);
    activeNotifications.set(id, record);
    enforceActiveLimit();
    saveNotificationState();
    renderNotificationCenter();
}

// Persistent notifications are restored before feature modules finish their
// startup work. Re-register handlers for an existing record so an action does
// not remain disabled merely because the page was reloaded.
export function registerNotificationActionHandlers(id, actions = []) {
    ensureNotificationState();
    actions.forEach(action => {
        if (typeof action.handler === 'function') {
            notificationHandlers.set(notificationHandlerKey(id, action.id), action.handler);
        }
    });
    renderNotificationCenter();
}

export function resolveNotification(id, reason = 'Resolved automatically.') {
    ensureNotificationState();
    const record = activeNotifications.get(id);
    if (!record) return;
    archiveNotification(record, reason);
    saveNotificationState();
    renderNotificationCenter();
}

function clearArchivedNotifications() {
    ensureNotificationState();
    archivedNotifications = [];
    saveNotificationState();
    renderNotificationCenter();
}

function getToastIcon(type) {
    const icons = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ',
        explicit: '🔒'
    };
    return icons[type] || 'ℹ';
}

const EXPLICIT_LEVEL_ICONS = { 0: '🔒', 1: '🔓', 2: '🔥' };

export function showToast(title, type = 'error', message = '', options = {}) {
    const container = document.getElementById('toast-container');
    if (!container) return null;

    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;

    let content = '';

    if (type === 'progress') {
        content = '<div class="toast-content"><div class="toast-progress-bar"><div class="toast-progress-fill" style="width:0%"></div></div></div>';
    } else {
        const iconMap = {
            success: 'success',
            error: 'error',
            warning: 'warning',
            info: 'info',
            explicit: 'explicit'
        };
        const iconType = iconMap[type] || 'info';
        content = `
            <div class="toast-icon ${type}">${getToastIcon(iconType)}</div>
            <div class="toast-content">
                ${title ? '<div class="toast-title">' + escapeHtml(title) + '</div>' : ''}
                ${message ? '<div class="toast-message">' + escapeHtml(message) + '</div>' : ''}
            </div>
            <button class="toast-close" data-toast-close="">&times;</button>
        `;
    }

    // eslint-disable-next-line no-unsanitized/property -- content is built from hardcoded template; type is a caller-controlled enum used only in CSS class; title/message wrapped in escapeHtml()
    toast.innerHTML = content;

    // Set explicit-level icon and class
    if (type === 'explicit' && options.level !== undefined) {
        const iconEl = toast.querySelector('.toast-icon');
        if (iconEl) iconEl.textContent = EXPLICIT_LEVEL_ICONS[options.level] || 'G';
        toast.classList.add('explicit-level-' + options.level);
    }

    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });

    if (type === 'progress') {
        return toast;
    } else {
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, options.duration || TOAST_AUTO_DISMISS);
        return null;
    }
}

function updateToastProgress(toastElement, percent, message) {
    if (!toastElement) return;
    const fill = toastElement.querySelector('.toast-progress-fill');
    const content = toastElement.querySelector('.toast-content');
    if (fill) fill.style.width = percent + '%';
    if (content && message) {
        content.innerHTML = '<div class="toast-title">' + escapeHtml(message) + '</div>';
    }
}

export function showToastWithActions(title, type, message, actions = [], options = {}) {
    const { notificationId = null, onDismiss = null, duration = Math.max(TOAST_AUTO_DISMISS, 5000) } = options;
    if (notificationId) {
        registerPersistentNotification(notificationId, title, type, message, actions);
    }

    const container = document.getElementById('toast-container');
    if (!container) return null;

    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type + ' toast-with-actions';

    const iconMap = {
        success: 'success',
        error: 'error',
        warning: 'warning',
        info: 'info',
        explicit: 'explicit'
    };
    const iconType = iconMap[type] || 'info';

    let actionsHtml = '';
    if (actions.length > 0) {
        actionsHtml = '<div class="toast-actions">' +
            actions.map(action => {
                const cls = action.primary ? 'btn-sm btn-primary' : 'btn-sm btn-secondary';
                return '<button class="' + cls + '" data-action="' + action.id + '">' + escapeHtml(action.label) + '</button>';
            }).join('') + '</div>';
    }

    // eslint-disable-next-line no-unsanitized/property -- type is a hardcoded enum used only in CSS class; title/message wrapped in escapeHtml(); actionsHtml uses escapeHtml(); getToastIcon returns hardcoded strings
    toast.innerHTML = `
        <div class="toast-icon ${type}">${getToastIcon(iconType)}</div>
        <div class="toast-content">
            ${title ? '<div class="toast-title">' + escapeHtml(title) + '</div>' : ''}
            ${message ? '<div class="toast-message">' + escapeHtml(message) + '</div>' : ''}
        </div>
        ${actionsHtml}
        <button class="toast-close" data-toast-close="">&times;</button>
    `;

    let actionTaken = false;

    if (actions.length > 0) {
        toast.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', () => {
                actionTaken = true;
                const action = actions.find(a => a.id === btn.dataset.action);
                if (action && action.handler) action.handler();
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            });
        });
    }

    // Close button also counts as dismissed without action
    toast.querySelector('[data-toast-close]')?.addEventListener('click', () => {
        if (!actionTaken && onDismiss) onDismiss();
    });

    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });

    setTimeout(() => {
        if (!actionTaken && onDismiss) onDismiss();
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function showToastProgress(title, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return null;

    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    // eslint-disable-next-line no-unsanitized/property -- type is a hardcoded enum used only in CSS class; title wrapped in escapeHtml(); getToastIcon returns hardcoded strings
    toast.innerHTML = `
        <div class="toast-icon ${type}">${getToastIcon(type)}</div>
        <div class="toast-content">
            ${title ? '<div class="toast-title">' + escapeHtml(title) + '</div>' : ''}
            <div class="toast-progress-bar"><div class="toast-progress-fill" style="width:0%"></div></div>
        </div>
    `;
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });
    return toast;
}

// ── Public API ────────────────────────────────────────────────────────────────

export function initToast() {
    ensureNotificationState();
    const notifications = document.getElementById('nav-notifications');
    const notificationsButton = document.getElementById('nav-notifications-btn');
    const notificationsMenu = document.getElementById('nav-notifications-menu');
    const clearArchived = document.getElementById('nav-notifications-clear');
    if (notifications && notificationsButton && notificationsMenu) {
        const navBar = notificationsButton.closest('.top-nav-bar');
        const setMenuOpen = (open) => {
            if (!open) {
                closeNotificationMenu();
                return;
            }
            notificationsMenu.hidden = !open;
            notificationsButton.setAttribute('aria-expanded', String(open));
            // Class fallback for the nav z-index elevation: :has() is not
            // supported on older WebKit builds (e.g. macOS 13–14.1 via wry).
            navBar?.classList.toggle('nav-notifications-open', open);
        };
        notificationsButton.addEventListener('click', (event) => {
            event.stopPropagation();
            const open = notificationsMenu.hidden;
            setMenuOpen(open);
            if (open) renderNotificationCenter();
        });
        const modalObserver = new MutationObserver((records) => {
            const modalOpened = records.some(record => {
                const candidates = [record.target, ...record.addedNodes];
                return candidates.some(candidate => {
                    const element = candidate instanceof Element ? candidate : null;
                    const target = element?.matches('.modal-overlay')
                        ? element
                        : element?.closest('.modal-overlay');
                    return target && isOpenModalOverlay(target);
                });
            });
            if (modalOpened) setMenuOpen(false);
        });
        modalObserver.observe(document.body, {
            subtree: true,
            attributes: true,
            childList: true,
            attributeFilter: ['class', 'style', 'aria-hidden', 'inert'],
        });
        clearArchived?.addEventListener('click', clearArchivedNotifications);
        notificationsMenu.querySelectorAll('[data-notification-tab]').forEach(tab => {
            tab.addEventListener('click', () => {
                notificationTab = tab.dataset.notificationTab || 'active';
                renderNotificationCenter();
            });
        });
        document.addEventListener('click', (event) => {
            if (!notifications.contains(event.target)) {
                setMenuOpen(false);
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && !notificationsMenu.hidden) {
                setMenuOpen(false);
                notificationsButton.focus();
            }
        });
    }
    renderNotificationCenter();
    // Event delegation for toast close buttons
    document.getElementById('toast-container')?.addEventListener('click', (e) => {
        const closeBtn = e.target.closest('[data-toast-close]');
        if (closeBtn) {
            closeBtn.closest('.toast')?.remove();
        }
    });
}

/**
 * Minimal confirmation dialog matching app style.
 * Returns true if user confirmed.
 */
export async function showConfirmDialog(title, message, confirmLabel = 'Confirm') {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay app-confirm-overlay active';
    overlay.style.zIndex = '2000';

    const dialog = document.createElement('div');
    dialog.className = 'modal app-confirm-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');

    const dialogId = `app-confirm-${Date.now()}`;
    const titleId = `${dialogId}-title`;
    const messageId = `${dialogId}-message`;
    dialog.setAttribute('aria-labelledby', titleId);
    dialog.setAttribute('aria-describedby', messageId);

    const icon = document.createElement('div');
    icon.className = 'app-confirm-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '✓';

    const copy = document.createElement('div');
    copy.className = 'app-confirm-copy';

    const titleEl = document.createElement('div');
    titleEl.className = 'app-confirm-title';
    titleEl.id = titleId;
    titleEl.textContent = title;

    const msgEl = document.createElement('div');
    msgEl.className = 'app-confirm-message';
    msgEl.id = messageId;
    msgEl.textContent = message;

    const actions = document.createElement('div');
    actions.className = 'app-confirm-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn btn-modal-cancel';
    cancelBtn.textContent = 'Cancel';

    const confirmBtn = document.createElement('button');
    confirmBtn.type = 'button';
    confirmBtn.className = 'btn btn-modal-save';
    confirmBtn.textContent = confirmLabel;

    return new Promise(resolve => {
        let decided = false;

        function onKeydown(e) {
            if (e.key === 'Escape' && !decided) {
                decided = true;
                cleanup();
                resolve(false);
            }
        }

        function cleanup() {
            document.removeEventListener('keydown', onKeydown);
            if (overlay.parentElement) overlay.remove();
        }

        cancelBtn.addEventListener('click', () => {
            if (decided) return;
            decided = true;
            cleanup();
            resolve(false);
        });

        confirmBtn.addEventListener('click', () => {
            if (decided) return;
            decided = true;
            cleanup();
            resolve(true);
        });

        overlay.addEventListener('click', (e) => {
            if (decided) return;
            if (e.target === overlay) {
                decided = true;
                cleanup();
                resolve(false);
            }
        });

        document.addEventListener('keydown', onKeydown);

        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        copy.appendChild(titleEl);
        copy.appendChild(msgEl);
        dialog.appendChild(icon);
        dialog.appendChild(copy);
        dialog.appendChild(actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        cancelBtn.focus();
    });
}

/**
 * Minimal text prompt dialog matching app style.
 * Returns user text or null if cancelled.
 */
export async function showPromptDialog(title, message, defaultValue = '', options = {}) {
    const previouslyFocused = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay app-prompt-overlay';
    overlay.style.zIndex = '2000';
    overlay.style.display = 'grid';

    const dialog = document.createElement('div');
    dialog.className = 'modal app-prompt-dialog';
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.style.width = '420px';
    dialog.style.padding = '14px 16px';

    const dialogId = `app-prompt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const titleId = `${dialogId}-title`;
    const messageId = `${dialogId}-message`;
    dialog.setAttribute('aria-labelledby', titleId);
    dialog.setAttribute('aria-describedby', messageId);

    const titleEl = document.createElement('div');
    titleEl.className = 'app-prompt-title';
    titleEl.id = titleId;
    titleEl.style.fontSize = '15px';
    titleEl.style.fontWeight = '600';
    titleEl.style.marginBottom = '8px';
    titleEl.textContent = title;

    const msgEl = document.createElement('div');
    msgEl.className = 'app-prompt-message';
    msgEl.id = messageId;
    msgEl.style.fontSize = '13px';
    msgEl.style.color = 'var(--color-text-muted)';
    msgEl.style.marginBottom = '10px';
    msgEl.textContent = message;

    const input = document.createElement('input');
    input.className = 'app-prompt-input';
    input.setAttribute('aria-label', options.inputLabel || title);
    input.type = options.type || 'text';
    input.value = defaultValue;
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.style.width = '100%';
    input.style.boxSizing = 'border-box';
    input.style.padding = '8px 10px';
    input.style.marginBottom = '12px';
    input.style.borderRadius = '999px';
    input.style.border = '1px solid var(--border-subtle)';
    input.style.background = 'var(--color-bg-surface)';
    input.style.color = 'var(--color-text-primary)';
    input.style.fontSize = '14px';
    input.style.outline = 'none';

    const actions = document.createElement('div');
    actions.className = 'app-prompt-actions';
    actions.style.display = 'flex';
    actions.style.justifyContent = 'flex-end';
    actions.style.gap = '8px';

    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn btn-modal-cancel';
    cancelBtn.textContent = 'Cancel';

    const okBtn = document.createElement('button');
    okBtn.type = 'button';
    okBtn.className = 'btn btn-modal-save';
    okBtn.textContent = options.confirmLabel || 'OK';

    return new Promise(resolve => {
        let decided = false;

        function onDocumentKeydown(e) {
            if (e.key === 'Escape') {
                e.preventDefault();
                handleCancel();
                return;
            }
            if (e.key !== 'Tab') return;
            const focusable = [input, cancelBtn, okBtn].filter(element => !element.disabled);
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (e.shiftKey && document.activeElement === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
                e.preventDefault();
                first.focus();
            }
        }

        function cleanup() {
            document.removeEventListener('keydown', onDocumentKeydown);
            if (overlay.parentElement) overlay.remove();
            if (previouslyFocused?.isConnected) previouslyFocused.focus();
        }

        function handleCancel() {
            if (decided) return;
            decided = true;
            cleanup();
            resolve(null);
        }

        function handleOk() {
            if (decided) return;
            decided = true;
            cleanup();
            resolve(input.value === '' ? null : input.value);
        }

        cancelBtn.addEventListener('click', handleCancel);
        okBtn.addEventListener('click', handleOk);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleOk();
        });

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) handleCancel();
        });

        document.addEventListener('keydown', onDocumentKeydown);

        actions.appendChild(cancelBtn);
        actions.appendChild(okBtn);
        dialog.appendChild(titleEl);
        dialog.appendChild(msgEl);
        dialog.appendChild(input);
        dialog.appendChild(actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        input.focus();
        input.select();
    });
}
