import { registerNotificationActionHandlers, resolveNotification, showToast, showToastWithActions } from './toast.js';
import { switchTab } from './nav.js';

const TOAST_SEEN_KEY = 'local-llm-foundry-migration-toast-seen';
const QUEUED_STATUS = 'Migration is queued. Restart Foundry to begin migration on the next launch.';
const QUEUED_SUMMARY = 'Nothing has moved yet. Critical application state will be copied on the next launch; models and generated runtimes are not copied, and the legacy home stays available for rollback.';

function formatCopySize(bytes) {
    if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return 'Unavailable';
    const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
    let amount = bytes;
    let unit = 0;
    while (amount >= 1024 && unit < units.length - 1) {
        amount /= 1024;
        unit += 1;
    }
    return `${amount.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${units[unit]}`;
}

/**
 * Surface a non-blocking migration hint. Detection is read-only; models and
 * application state stay on the legacy root until the user explicitly queues
 * migration for the next launch.
 */
export async function initAppHomeMigration() {
    let status;
    try {
        const response = await window.authFetch('/api/app-home-migration/status', {
            headers: window.authHeaders(),
        });
        if (!response.ok) return;
        status = await response.json();
    } catch {
        return;
    }
    const stateEl = document.getElementById('app-home-migration-state');
    const summaryEl = document.getElementById('app-home-migration-summary');
    const previewBtn = document.getElementById('app-home-migration-preview');
    const queueBtn = document.getElementById('app-home-migration-queue');
    const planEl = document.getElementById('app-home-migration-plan');
    const sourceEl = document.getElementById('app-home-migration-source');
    const destinationEl = document.getElementById('app-home-migration-destination');
    const locationsEl = document.getElementById('app-home-migration-locations');
    let queued = status?.state === 'migration_queued';
    const migrationRequired = !!status?.migration_required;
    if (stateEl) {
        stateEl.textContent = queued
            ? QUEUED_STATUS
            : (migrationRequired
                ? 'Legacy application data detected. Nothing has moved yet.'
                : 'This installation is already using the Foundry application home.');
    }
    if (summaryEl) {
        summaryEl.textContent = queued
            ? QUEUED_SUMMARY
            : (migrationRequired
                ? 'Preview the upgrade before queueing it. No data is copied until the next launch.'
                : '');
    }
    if (sourceEl) sourceEl.textContent = status?.legacy_root || 'Unavailable';
    if (destinationEl) destinationEl.textContent = status?.canonical_root || 'Unavailable';
    if (locationsEl) locationsEl.hidden = !status?.legacy_root;
    if (planEl) planEl.hidden = true;
    if (previewBtn) previewBtn.disabled = !migrationRequired || queued;
    if (queueBtn) queueBtn.disabled = true;

    const migrationActions = [{
        id: 'review-migration',
        label: 'Review migration',
        primary: true,
        handler: () => {
            switchTab('settings');
            import('./settings.js').then(({ openSettingsModal }) => openSettingsModal('migration'));
        },
    }];
    const notificationId = 'app-home-migration-pending';
    // Register before any early return so a notification saved from an earlier
    // session (including while queued) stays clickable after a reload.
    registerNotificationActionHandlers(notificationId, migrationActions);
    if (!migrationRequired && !queued) {
        resolveNotification(notificationId, 'Application data now uses the Foundry home.');
    }
    if (!migrationRequired || queued) return;

    let plan = null;
    let previewing = false;
    let queueing = false;
    previewBtn?.addEventListener('click', async () => {
        if (queued || queueing || previewing) return;
        previewing = true;
        plan = null;
        previewBtn.disabled = true;
        if (queueBtn) queueBtn.disabled = true;
        if (planEl) planEl.hidden = true;
        if (stateEl) stateEl.textContent = 'Preparing migration preview. Nothing has moved yet.';
        try {
            const response = await window.authFetch('/api/app-home-migration/preview', {
                headers: window.authHeaders(),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload.plan?.plan_id) throw new Error(payload.error || 'Preview unavailable');
            plan = payload.plan;
            // entries includes retained models and recreatable runtimes, not just
            // copied resources. Present it as inventory, never as a copy count.
            const inventoriedCount = plan.entries?.length || 0;
            const retainedCount = plan.retained_entries?.length || 0;
            const copySizeEl = document.getElementById('app-home-migration-copy-size');
            const inventoriedEl = document.getElementById('app-home-migration-inventoried-count');
            const retainedEl = document.getElementById('app-home-migration-retained-count');
            const planIdEl = document.getElementById('app-home-migration-plan-id');
            const technicalEl = document.getElementById('app-home-migration-technical-summary');
            if (copySizeEl) copySizeEl.textContent = formatCopySize(plan.required_copy_bytes);
            if (inventoriedEl) inventoriedEl.textContent = inventoriedCount.toLocaleString();
            if (retainedEl) retainedEl.textContent = retainedCount.toLocaleString();
            if (sourceEl) sourceEl.textContent = plan.source || status.legacy_root || 'Unavailable';
            if (destinationEl) destinationEl.textContent = plan.destination || status.canonical_root || 'Unavailable';
            if (locationsEl) locationsEl.hidden = false;
            if (planIdEl) planIdEl.textContent = plan.plan_id;
            if (technicalEl) {
                technicalEl.textContent = JSON.stringify({
                    schema_version: plan.schema_version,
                    required_copy_bytes: plan.required_copy_bytes,
                    total_seen_bytes: plan.total_seen_bytes,
                    inventoried_entries: inventoriedCount,
                    retained_entries: retainedCount,
                }, null, 2);
            }
            if (planEl) planEl.hidden = false;
            if (stateEl) stateEl.textContent = 'Migration preview ready. Nothing has moved yet.';
            if (queueBtn) queueBtn.disabled = false;
        } catch (error) {
            if (stateEl) stateEl.textContent = 'Migration preview failed. Try previewing again before queueing.';
            showToast('Migration preview failed', 'error', error.message || 'Try again later.');
        } finally {
            previewing = false;
            previewBtn.disabled = queued || queueing;
        }
    });
    queueBtn?.addEventListener('click', async () => {
        if (!plan || queued || queueing || previewing) return;
        if (!window.confirm('Queue the Foundry migration for the next launch? Critical application state will be copied then. Models and generated runtimes will not be copied, and your legacy home will be retained for rollback.')) return;
        queueing = true;
        queueBtn.disabled = true;
        if (previewBtn) previewBtn.disabled = true;
        if (stateEl) stateEl.textContent = 'Queueing migration for next launch. Nothing has moved yet.';
        try {
            const tokenResponse = await window.authFetch('/api/db/admin-token', {
                headers: window.authHeaders(),
            });
            const tokenPayload = await tokenResponse.json().catch(() => ({}));
            if (!tokenResponse.ok || !tokenPayload.token) throw new Error('Administrator authorization is unavailable.');
            const response = await fetch('/api/app-home-migration/queue', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${tokenPayload.token}` },
                body: JSON.stringify({ plan_id: plan.plan_id, confirmation: 'MIGRATE TO LOCAL LLM FOUNDRY' }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload.ok) throw new Error(payload.error || 'Could not queue migration.');
            queued = true;
            showToast('Migration queued', 'success', 'Restart Foundry when you are ready to complete the upgrade.');
            if (stateEl) stateEl.textContent = QUEUED_STATUS;
            if (summaryEl) summaryEl.textContent = QUEUED_SUMMARY;
        } catch (error) {
            if (stateEl) stateEl.textContent = 'Migration could not be queued. Review the preview or try again.';
            showToast('Migration could not be queued', 'error', error.message || 'Try again later.');
        } finally {
            queueing = false;
            queueBtn.disabled = queued;
            if (previewBtn) previewBtn.disabled = queued;
        }
    });


    let toastSeen = false;
    try {
        toastSeen = sessionStorage.getItem(TOAST_SEEN_KEY) === '1';
        if (!toastSeen) sessionStorage.setItem(TOAST_SEEN_KEY, '1');
    } catch {
        // A storage failure must not block the dashboard or migration hint.
    }
    if (toastSeen) return;

    showToastWithActions(
        'Upgrade ready',
        'info',
        'Nothing moves until you approve.',
        migrationActions,
        { notificationId, duration: 12000 },
    );
}
