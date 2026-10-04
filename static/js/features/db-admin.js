// Database Administration Panel
// Premium modal with tabs for maintenance, backups, indexes, repair, and query

import { showConfirmDialog } from './toast.js';

const dbAdminLog = [];
let dbAdminOverlay = null;
let dbAdminToken = null;
let dbStatsInterval = null;
let dbAdminInitializedData = false;

async function ensureDbAdminToken() {
    if (dbAdminToken) return;
    try {
        const res = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (res.ok) {
            const data = await res.json();
            if (data.token) dbAdminToken = data.token;
        }
    } catch {
        // Non-critical: continue without token if fetch fails.
    }
}

export function initDbAdmin() {
    dbAdminOverlay = document.getElementById('db-admin-modal');
    if (!dbAdminOverlay) return;

    // Close handlers
    document.getElementById('db-admin-modal-close')?.addEventListener('click', closeDbAdminModal);

    // Open from settings Chat tab
    document.getElementById('settings-open-db-admin-btn')?.addEventListener('click', () => {
        openDbAdminModal();
    });
    dbAdminOverlay?.addEventListener('click', (e) => {
        if (e.target === dbAdminOverlay) closeDbAdminModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && dbAdminOverlay?.classList.contains('active')) {
            closeDbAdminModal();
        }
    });

    // Tab switching
    document.querySelectorAll('.db-tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Bind button handlers
    document.getElementById('db-btn-refresh')?.addEventListener('click', () => {
        loadDbStats();
        loadBackups();
        loadIndexes();
    });
    document.getElementById('db-btn-backup')?.addEventListener('click', handleBackup);
    document.getElementById('db-btn-check')?.addEventListener('click', handleIntegrityCheck);
    document.getElementById('settings-db-btn-backup')?.addEventListener('click', handleBackup);
    document.getElementById('settings-db-btn-check')?.addEventListener('click', handleIntegrityCheck);
    document.getElementById('db-btn-checkpoint')?.addEventListener('click', () => handleMaintenance('checkpoint'));
    document.getElementById('db-btn-vacuum')?.addEventListener('click', () => handleMaintenance('vacuum'));
    document.getElementById('db-btn-rebuild-fts')?.addEventListener('click', () => handleMaintenance('rebuild_fts'));
    document.getElementById('db-btn-analyze')?.addEventListener('click', () => handleMaintenance('analyze'));

    // Backups tab
    document.getElementById('db-btn-clear-log')?.addEventListener('click', () => {
        dbAdminLog.length = 0;
        renderLog();
    });

    // Indexes tab
    document.getElementById('db-btn-rebuild-all-indexes')?.addEventListener('click', handleRebuildAllIndexes);

    // Repair tab
    document.getElementById('db-btn-repair-indexes')?.addEventListener('click', handleRepairIndexes);
    document.getElementById('db-btn-repair-compact')?.addEventListener('click', handleRepairCompact);
    document.getElementById('db-btn-repair-emergency')?.addEventListener('click', handleRepairEmergency);

    // Query tab
    document.getElementById('db-btn-run-query')?.addEventListener('click', handleQuery);
    document.getElementById('db-btn-clear-query')?.addEventListener('click', () => {
        const input = document.getElementById('db-query-input');
        if (input) input.value = '';
        const result = document.getElementById('db-query-result');
        if (result) result.hidden = true;
    });
    document.getElementById('db-btn-copy-result')?.addEventListener('click', () => {
        const output = document.getElementById('db-query-output');
        if (output) {
            navigator.clipboard.writeText(output.textContent);
            addLogEntry('success', 'Result copied to clipboard');
        }
    });

    // Enter key for query
    document.getElementById('db-query-input')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleQuery();
        }
    });

    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopDbStatsPolling();
        } else if (dbAdminOverlay?.classList.contains('active')) {
            startDbStatsPolling();
            loadDbStats();
        }
    });
}

function openDbAdminModal() {
    if (!dbAdminOverlay) return;
    if (dbAdminCloseTimer !== null) { clearTimeout(dbAdminCloseTimer); dbAdminCloseTimer = null; }
    dbAdminOverlay.classList.remove('closing');
    dbAdminOverlay.querySelector('.db-admin-modal')?.classList.remove('closing');
    dbAdminOverlay.classList.add('active');

    if (!dbAdminInitializedData) {
        dbAdminInitializedData = true;
        ensureDbAdminToken();
    }

    loadDbStats();
    loadBackups();
    loadIndexes();
    startDbStatsPolling();
}

let dbAdminCloseTimer = null;

function closeDbAdminModal() {
    if (!dbAdminOverlay) return;
    stopDbStatsPolling();
    const modal = dbAdminOverlay.querySelector('.db-admin-modal');
    if (modal) {
        modal.classList.add('closing');
        // Same stale-timer race as settings.js: reopening inside the animation window let
        // the previous close's timer strip `active` off the newly opened overlay.
        if (dbAdminCloseTimer !== null) clearTimeout(dbAdminCloseTimer);
        dbAdminCloseTimer = setTimeout(() => {
            dbAdminCloseTimer = null;
            dbAdminOverlay.classList.remove('active');
            modal.classList.remove('closing');
        }, 250);
    } else {
        dbAdminOverlay.classList.remove('active');
    }
}

function startDbStatsPolling() {
    if (dbStatsInterval || document.hidden) return;
    dbStatsInterval = setInterval(() => {
        if (!dbAdminOverlay?.classList.contains('active')) return;
        loadDbStats();
    }, 30000);
}

function stopDbStatsPolling() {
    if (!dbStatsInterval) return;
    clearInterval(dbStatsInterval);
    dbStatsInterval = null;
}

function switchTab(tabName) {
    document.querySelectorAll('.db-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.db-tab-panel').forEach(p => p.classList.remove('active'));

    const tab = document.querySelector(`.db-tab[data-tab="${tabName}"]`);
    const panel = document.querySelector(`.db-tab-panel[data-panel="${tabName}"]`);

    if (tab) tab.classList.add('active');
    if (panel) panel.classList.add('active');

    // Load data for specific tabs
    if (tabName === 'backups') loadBackups();
    if (tabName === 'indexes') loadIndexes();
}

async function loadDbStats() {
    try {
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const [statsRes, integrityRes] = await Promise.all([
            fetch('/api/db/stats', { headers }),
            fetch('/api/db/integrity', { headers }),
        ]);

        if (!statsRes.ok || !integrityRes.ok) {
            addLogEntry('error', 'Failed to load DB stats/integrity (possible auth issue)');
            return;
        }

        const stats = await statsRes.json();
        const integrity = await integrityRes.json();

        const tabCount = stats.tab_count ?? '-';
        const msgCount = stats.message_count ?? '-';
        const sizeStr = formatBytes(stats.file_size_bytes || 0);
        const ftsCount = stats.fts_index_count ?? '-';

        // Update modal stats
        document.getElementById('db-stat-tabs').textContent = tabCount;
        document.getElementById('db-stat-messages').textContent = msgCount;
        document.getElementById('db-stat-size').textContent = sizeStr;
        document.getElementById('db-stat-fts').textContent = ftsCount;

        // Update inline settings card stats
        const sEl = (id) => document.getElementById(id);
        if (sEl('settings-db-stat-tabs')) sEl('settings-db-stat-tabs').textContent = tabCount;
        if (sEl('settings-db-stat-messages')) sEl('settings-db-stat-messages').textContent = msgCount;
        if (sEl('settings-db-stat-size')) sEl('settings-db-stat-size').textContent = sizeStr;
        if (sEl('settings-db-stat-fts')) sEl('settings-db-stat-fts').textContent = ftsCount;

        // Update status indicator (modal + settings card)
        const statusLabel = integrity.status === 'healthy' ? 'Healthy'
            : integrity.status === 'warning' ? 'Issues Detected' : 'Errors Found';
        const statusClass = integrity.status === 'healthy' ? 'db-status-dot healthy'
            : integrity.status === 'warning' ? 'db-status-dot warning' : 'db-status-dot error';

        const dot = document.getElementById('db-status-dot');
        const text = document.getElementById('db-status-text');
        if (dot) dot.className = statusClass;
        if (text) text.textContent = statusLabel;

        const sDot = document.getElementById('settings-db-status-dot');
        const sText = document.getElementById('settings-db-status-text');
        if (sDot) sDot.className = statusClass;
        if (sText) sText.textContent = statusLabel;
    } catch (error) {
        console.error('[db-admin] Failed to load stats:', error);
        addLogEntry('error', 'Failed to load database statistics');
    }
}

async function handleBackup() {
    addLogEntry('info', 'Creating backup...');
    setButtonLoading('db-btn-backup', true);

    try {
        const token = typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN
            ? window.__API_TOKEN
            : dbAdminToken;
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch('/api/db/backup', { method: 'POST', headers });
        const result = await res.json();

        if (result.status === 'backup_created') {
            addLogEntry('success', `Backup created: ${formatBytes(result.size_bytes)}`);
        } else {
            addLogEntry('error', result.error || 'Backup failed');
        }
    } catch (error) {
        addLogEntry('error', `Backup failed: ${error.message}`);
    } finally {
        setButtonLoading('db-btn-backup', false);
    }
}

async function handleIntegrityCheck() {
    addLogEntry('info', 'Running integrity check...');
    setButtonLoading('db-btn-check', true);

    try {
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch('/api/db/integrity', { headers });
        const result = await res.json();

        if (result.status === 'healthy') {
            addLogEntry('success', 'Database integrity: OK');
        } else {
            addLogEntry('warning', `Database integrity issue: ${result.detail}`);
        }
    } catch (error) {
        addLogEntry('error', `Integrity check failed: ${error.message}`);
    } finally {
        setButtonLoading('db-btn-check', false);
    }
}

async function handleMaintenance(operation) {
    const labels = {
        checkpoint: 'WAL Checkpoint',
        vacuum: 'Vacuum',
        rebuild_fts: 'Rebuild FTS Index',
        analyze: 'Analyze',
    };

    addLogEntry('info', `Running ${labels[operation] || operation}...`);

    try {
        const headers = { 'Content-Type': 'application/json' };
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch('/api/db/maintenance', {
            method: 'POST',
            headers,
            body: JSON.stringify({ operation }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', `${labels[operation] || operation}: ${result.error}`);
        } else {
            addLogEntry('success', `${labels[operation] || operation}: ${result.status || 'completed'}`);
        }
    } catch (error) {
        addLogEntry('error', `${labels[operation] || operation} failed: ${error.message}`);
    }

    loadDbStats();
}

async function loadBackups() {
    const listEl = document.getElementById('db-backups-list');
    const emptyEl = document.getElementById('db-backups-empty');
    const countEl = document.getElementById('db-backup-count');
    const sizeEl = document.getElementById('db-backup-total-size');

    if (!listEl) return;

    try {
        const token = typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN
            ? window.__API_TOKEN
            : dbAdminToken;
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch('/api/db/backups', { headers });
        if (!res.ok) {
            addLogEntry('error', 'Failed to load backups (possible auth issue)');
            if (emptyEl) emptyEl.innerHTML = '<p>Failed to load backups.</p>';
            return;
        }
        const data = await res.json();

        if (!data.backups || data.backups.length === 0) {
            emptyEl.innerHTML = '<p>No backups found. Create one with "Backup Now".</p>';
            countEl.textContent = '0';
            sizeEl.textContent = '0 B';
            return;
        }

        countEl.textContent = data.backups.length;
        sizeEl.textContent = formatBytes(data.total_size || 0);

        listEl.replaceChildren(...data.backups.map(b => {
            const displayName = b.name.includes('/') ? b.name.split('/').pop() : b.name;
            const kind = b.kind || (b.name.startsWith('auto/') ? 'auto' : b.name.startsWith('daily/') ? 'daily' : 'manual');
            const kindLabel = { auto: 'hourly', daily: 'daily', manual: 'manual' }[kind] || kind;

            const nameRow = el('div', 'db-backup-name-row');
            nameRow.append(
                el('span', 'db-backup-name', displayName),
                el('span', `db-backup-kind db-backup-kind-${String(kind).replace(/[^a-z0-9_-]/gi, '')}`, kindLabel),
            );
            const info = el('div', 'db-backup-info');
            info.append(nameRow, el('span', 'db-backup-meta', `${formatBytes(b.size)} \u00b7 ${formatDate(b.modified)}`));

            const actions = el('div', 'db-backup-actions');
            actions.append(
                iconButton('Restore', RESTORE_ICON, () => window.restoreBackup(b.name)),
                iconButton('Delete', DELETE_ICON, () => window.deleteBackup(b.name)),
            );

            const item = el('div', 'db-backup-item');
            item.append(info, actions);
            return item;
        }));
    } catch (error) {
        console.error('[db-admin] Failed to load backups:', error);
    }
}

async function loadIndexes() {
    const listEl = document.getElementById('db-indexes-list');
    if (!listEl) return;

    try {
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch('/api/db/indexes', { headers });
        if (!res.ok) {
            addLogEntry('error', 'Failed to load indexes (possible auth issue)');
            listEl.innerHTML = '<div class="db-empty-state"><p>Failed to load indexes.</p></div>';
            return;
        }

        const data = await res.json();

        if (!data.indexes || data.indexes.length === 0) {
            listEl.innerHTML = '<div class="db-empty-state"><p>No indexes found.</p></div>';
            return;
        }

        listEl.replaceChildren(...data.indexes.map(idx => {
            const info = el('div');
            info.append(el('div', 'db-index-name', idx.name), el('div', 'db-index-table', idx.table || ''));
            const actions = el('div', 'db-index-actions');
            if (idx.rebuildable) {
                const rebuild = el('button', 'db-action-btn small', 'Rebuild');
                rebuild.type = 'button';
                rebuild.addEventListener('click', () => window.rebuildIndex(idx.name));
                actions.append(rebuild);
            }
            const item = el('div', 'db-index-item');
            item.append(info, actions);
            return item;
        }));
    } catch (error) {
        console.error('[db-admin] Failed to load indexes:', error);
    }
}

async function handleRebuildAllIndexes() {
    addLogEntry('info', 'Rebuilding all indexes...');
    setButtonLoading('db-btn-rebuild-all-indexes', true);

    try {
        const headers = { 'Content-Type': 'application/json' };
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch('/api/db/maintenance', {
            method: 'POST',
            headers,
            body: JSON.stringify({ operation: 'rebuild_fts' }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', `Index rebuild: ${result.error}`);
        } else {
            addLogEntry('success', `Index rebuild: ${result.status || 'completed'}`);
        }
    } catch (error) {
        addLogEntry('error', `Index rebuild failed: ${error.message}`);
    } finally {
        setButtonLoading('db-btn-rebuild-all-indexes', false);
    }

    loadIndexes();
}

// Repair operations
async function handleRepairIndexes() {
    const ok1 = await showConfirmDialog(
        'Rebuild indexes',
        'This will drop and recreate all indexes. Continue?',
        'Rebuild'
    );
    if (!ok1) return;

    addLogEntry('info', 'Repairing indexes...');
    setButtonLoading('db-btn-repair-indexes', true);

    try {
        // First create a backup (requires api-token)
        const backupHeaders = { 'Content-Type': 'application/json' };
        const backupToken = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (backupToken) {
            backupHeaders['Authorization'] = `Bearer ${backupToken}`;
        }
        const backupRes = await fetch('/api/db/backup', { method: 'POST', headers: backupHeaders });
        if (!backupRes.ok) {
            const err = await backupRes.json().catch(() => ({}));
            addLogEntry('error', `Backup failed; aborting repair: ${err.error || 'unknown error'}`);
            setButtonLoading('db-btn-repair-indexes', false);
            return;
        }
        addLogEntry('info', 'Backup created before repair');

        // Repair requires db-admin-token
        const headers = { 'Content-Type': 'application/json' };
        if (dbAdminToken) {
            headers['Authorization'] = `Bearer ${dbAdminToken}`;
        }

        const res = await fetch('/api/db/repair', {
            method: 'POST',
            headers,
            body: JSON.stringify({ operation: 'repair_indexes' }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', `Index repair: ${result.error}`);
        } else {
            addLogEntry('success', `Index repair: ${result.status || 'completed'}`);
        }
    } catch (error) {
        addLogEntry('error', `Index repair failed: ${error.message}`);
    } finally {
        setButtonLoading('db-btn-repair-indexes', false);
    }

    loadIndexes();
}

async function handleRepairCompact() {
    const ok1 = await showConfirmDialog(
        'Compact database',
        'This will compact the database. The app may be unresponsive during this operation. Continue?',
        'Compact'
    );
    if (!ok1) return;

    addLogEntry('info', 'Compacting database...');
    setButtonLoading('db-btn-repair-compact', true);

    try {
        // Backup (requires api-token)
        const backupHeaders = { 'Content-Type': 'application/json' };
        const backupToken = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (backupToken) {
            backupHeaders['Authorization'] = `Bearer ${backupToken}`;
        }
        const backupRes = await fetch('/api/db/backup', { method: 'POST', headers: backupHeaders });
        if (!backupRes.ok) {
            const err = await backupRes.json().catch(() => ({}));
            addLogEntry('error', `Backup failed; aborting compact: ${err.error || 'unknown error'}`);
            setButtonLoading('db-btn-repair-compact', false);
            return;
        }
        addLogEntry('info', 'Backup created before compact');

        // Vacuum (requires api-token)
        const headers = { 'Content-Type': 'application/json' };
        if (backupToken) {
            headers['Authorization'] = `Bearer ${backupToken}`;
        }

        const res = await fetch('/api/db/maintenance', {
            method: 'POST',
            headers,
            body: JSON.stringify({ operation: 'vacuum' }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', `Compact: ${result.error}`);
        } else {
            addLogEntry('success', `Compact: ${result.status || 'completed'}`);
        }
    } catch (error) {
        addLogEntry('error', `Compact failed: ${error.message}`);
    } finally {
        setButtonLoading('db-btn-repair-compact', false);
    }

    loadDbStats();
}

async function handleRepairEmergency() {
    const ok1 = await showConfirmDialog(
        'Emergency recovery',
        'EMERGENCY RECOVERY: This attempts to recover data from a corrupted database. A backup will be created first. Continue?',
        'Recover'
    );
    if (!ok1) return;

    addLogEntry('warning', 'Starting emergency recovery...');
    setButtonLoading('db-btn-repair-emergency', true);

    try {
        // Backup (requires api-token)
        const backupHeaders = { 'Content-Type': 'application/json' };
        const backupToken = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (backupToken) {
            backupHeaders['Authorization'] = `Bearer ${backupToken}`;
        }
        const backupRes = await fetch('/api/db/backup', { method: 'POST', headers: backupHeaders });
        if (!backupRes.ok) {
            const err = await backupRes.json().catch(() => ({}));
            addLogEntry('error', `Backup failed; aborting recovery: ${err.error || 'unknown error'}`);
            setButtonLoading('db-btn-repair-emergency', false);
            return;
        }
        addLogEntry('info', 'Backup created before recovery');

        // Repair (requires db-admin-token)
        const headers = { 'Content-Type': 'application/json' };
        if (dbAdminToken) {
            headers['Authorization'] = `Bearer ${dbAdminToken}`;
        }

        const res = await fetch('/api/db/repair', {
            method: 'POST',
            headers,
            body: JSON.stringify({ operation: 'emergency_recovery' }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', `Emergency recovery: ${result.error}`);
        } else {
            addLogEntry('success', `Emergency recovery: ${result.status || 'completed'}`);
        }
    } catch (error) {
        addLogEntry('error', `Emergency recovery failed: ${error.message}`);
    } finally {
        setButtonLoading('db-btn-repair-emergency', false);
    }

    loadDbStats();
}

async function handleQuery() {
    const input = document.getElementById('db-query-input');
    const resultEl = document.getElementById('db-query-result');
    const outputEl = document.getElementById('db-query-output');

    if (!input || !resultEl || !outputEl) return;

    const sql = input.value.trim();
    if (!sql) return;

    // Safety check - only allow SELECT and PRAGMA
    const upperSql = sql.toUpperCase().trim();
    if (!upperSql.startsWith('SELECT') && !upperSql.startsWith('PRAGMA')) {
        addLogEntry('error', 'Only SELECT and PRAGMA queries are allowed');
        return;
    }

    addLogEntry('info', `Executing: ${sql.substring(0, 50)}...`);

    try {
        await ensureDbAdminToken();
        const headers = { 'Content-Type': 'application/json' };
        // Use global api-token if available; fall back to db-admin-token.
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        const res = await fetch('/api/db/query', {
            method: 'POST',
            headers,
            body: JSON.stringify({ sql }),
        });
        const result = await res.json();

        if (result.error) {
            addLogEntry('error', result.error);
            outputEl.textContent = result.error;
        } else {
            addLogEntry('success', `Query returned ${result.rows?.length || 0} rows`);
            outputEl.textContent = JSON.stringify(result, null, 2);
        }
        resultEl.hidden = false;
    } catch (error) {
        addLogEntry('error', `Query failed: ${error.message}`);
        outputEl.textContent = error.message;
        resultEl.hidden = false;
    }
}

// Global functions for inline handlers
window.restoreBackup = async function(name) {
    const ok1 = await showConfirmDialog(
        'Restore from backup',
        `Restore from backup "${name}"? Current data will be replaced.`,
        'Restore'
    );
    if (!ok1) return;

    addLogEntry('warning', `Restoring from ${name}...`);
    try {
        await ensureDbAdminToken();
        if (!dbAdminToken) {
            addLogEntry('error', 'Restore failed: db-admin-token not available');
            return;
        }
        const headers = { 'Content-Type': 'application/json' };
        headers['Authorization'] = `Bearer ${dbAdminToken}`;
        const res = await fetch('/api/db/restore', {
            method: 'POST',
            headers,
            body: JSON.stringify({ backup_name: name }),
        });

        const result = await res.json();
        if (result.error) {
            addLogEntry('error', result.error);
        } else {
            addLogEntry('success', `Restored from ${name}`);
            loadDbStats();
        }
    } catch (error) {
        addLogEntry('error', `Restore failed: ${error.message}`);
    }
};

window.deleteBackup = async function(name) {
    const ok1 = await showConfirmDialog(
        'Delete backup',
        `Delete backup "${name}"?`,
        'Delete'
    );
    if (!ok1) return;

    try {
        await ensureDbAdminToken();
        if (!dbAdminToken) {
            addLogEntry('error', 'Delete failed: db-admin-token not available');
            return;
        }
        const headers = { 'Content-Type': 'application/json' };
        headers['Authorization'] = `Bearer ${dbAdminToken}`;
        const res = await fetch('/api/db/backup', {
            method: 'DELETE',
            headers,
            body: JSON.stringify({ backup_name: name }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', result.error);
        } else {
            addLogEntry('success', `Deleted ${name}`);
            loadBackups();
        }
    } catch (error) {
        addLogEntry('error', `Delete failed: ${error.message}`);
    }
};

window.rebuildIndex = async function(name) {
    addLogEntry('info', `Rebuilding index ${name}...`);
    try {
        const headers = { 'Content-Type': 'application/json' };
        const token = (typeof window.__API_TOKEN !== 'undefined' && window.__API_TOKEN)
            ? window.__API_TOKEN
            : dbAdminToken;
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch('/api/db/maintenance', {
            method: 'POST',
            headers,
            body: JSON.stringify({ operation: 'rebuild_fts' }),
        });
        const result = await res.json();
        if (result.error) {
            addLogEntry('error', `Index rebuild: ${result.error}`);
        } else {
            addLogEntry('success', `Index rebuild: ${result.status || 'completed'}`);
        }
        loadIndexes();
    } catch (error) {
        addLogEntry('error', `Index rebuild failed: ${error.message}`);
    }
};

function addLogEntry(type, message) {
    const timestamp = new Date().toLocaleTimeString();
    dbAdminLog.push({ type, message, timestamp });

    // Keep only last 100 entries
    while (dbAdminLog.length > 100) {
        dbAdminLog.shift();
    }

    renderLog();
}

function buildLogEntry(entry) {
    const row = el('div', `db-log-entry ${String(entry.type)}`);
    row.append(
        el('span', 'db-log-time', entry.timestamp),
        el('span', 'db-log-message', entry.message),
    );
    return row;
}

function renderLog() {
    // Modal log and the inline settings-card log share one entry list.
    for (const id of ['db-log-content', 'settings-db-log-content']) {
        const logContent = document.getElementById(id);
        if (!logContent) continue;
        logContent.replaceChildren(...dbAdminLog.map(buildLogEntry));
        // Scroll to bottom
        logContent.scrollTop = logContent.scrollHeight;
    }
}

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an element; `text` is assigned via textContent so it is never parsed as markup. */
function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

/** Small stroked SVG icon built from path/polyline definitions (static, trusted). */
function svgIcon(size, shapes) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('width', String(size));
    svg.setAttribute('height', String(size));
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    for (const [tag, attr, value] of shapes) {
        const shape = document.createElementNS(SVG_NS, tag);
        shape.setAttribute(attr, value);
        svg.append(shape);
    }
    return svg;
}

function iconButton(title, shapes, onClick) {
    const btn = el('button', 'db-action-btn small');
    btn.type = 'button';
    btn.title = title;
    btn.append(svgIcon(12, shapes));
    btn.addEventListener('click', onClick);
    return btn;
}

const RESTORE_ICON = [
    ['path', 'd', 'M3 12a9 9 0 019-9 9.75 9.75 0 016.74 2.74L21 8'],
    ['path', 'd', 'M21 3v5h-5'],
    ['path', 'd', 'M21 12a9 9 0 01-9 9 9.75 9.75 0 01-6.74-2.74L3 16'],
    ['path', 'd', 'M3 21v-5h5'],
];
const DELETE_ICON = [
    ['polyline', 'points', '3 6 5 6 21 6'],
    ['path', 'd', 'M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2'],
];

function setButtonLoading(btnId, loading) {
    // Each action exists in the modal and in the settings card (`settings-` prefix).
    for (const id of [btnId, `settings-${btnId}`]) {
        const btn = document.getElementById(id);
        if (!btn) continue;

        if (loading) {
            btn.disabled = true;
            btn.dataset.originalText = btn.textContent;
            btn.innerHTML = '<svg class="db-spinner" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 11-6.219-8.56"/></svg> Working...';
        } else {
            btn.disabled = false;
            btn.textContent = btn.dataset.originalText || btn.textContent;
        }
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatDate(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;

    // Use local-time day boundaries for "today/yesterday/Xd ago" decisions.
    const todayStart = new Date(now); todayStart.setHours(0, 0, 0, 0);
    const dateStart = new Date(date); dateStart.setHours(0, 0, 0, 0);
    const dayDiff = Math.floor((todayStart - dateStart) / 86400000);

    if (dayDiff === 0) return 'today';
    if (dayDiff === 1) return 'yesterday';
    if (dayDiff < 7) return `${dayDiff}d ago`;
    return date.toLocaleDateString();
}
