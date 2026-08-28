import { registerNotificationActionHandlers, showToast, showToastWithActions } from './toast.js';
import { switchTab } from './nav.js';

const TOAST_SEEN_KEY = 'local-llm-foundry-migration-toast-seen';

/**
 * Surface a non-blocking migration hint. Detection is read-only; models and
 * application state stay on the legacy root until the user explicitly starts
 * the authenticated preview/execute flow.
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
    if (stateEl) {
        stateEl.textContent = status?.migration_required
            ? (status.state === 'migration_queued'
                ? 'Migration is queued. Restart Foundry to complete it.'
                : 'Legacy application data detected. Nothing has moved yet.')
            : 'This installation is already using the Foundry application home.';
    }
    if (summaryEl) {
        summaryEl.textContent = status?.legacy_root
            ? `Legacy home: ${status.legacy_root} · Foundry home: ${status.canonical_root}`
            : '';
    }
    if (!status?.migration_required || status.state === 'migration_queued') return;

    let plan = null;
    previewBtn?.addEventListener('click', async () => {
        previewBtn.disabled = true;
        try {
            const response = await window.authFetch('/api/app-home-migration/preview', {
                headers: window.authHeaders(),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload.plan) throw new Error(payload.error || 'Preview unavailable');
            plan = payload.plan;
            if (planEl) {
                planEl.style.display = 'block';
                planEl.textContent = JSON.stringify({
                    plan_id: plan.plan_id,
                    required_copy_bytes: plan.required_copy_bytes,
                    copied_entries: plan.entries?.length || 0,
                    retained_entries: plan.retained_entries?.length || 0,
                }, null, 2);
            }
            if (queueBtn) queueBtn.disabled = false;
        } catch (error) {
            showToast('Migration preview failed', 'error', error.message || 'Try again later.');
        } finally {
            previewBtn.disabled = false;
        }
    });
    queueBtn?.addEventListener('click', async () => {
        if (!plan) return;
        if (!window.confirm('Queue the Foundry migration for the next restart? Your legacy root will be retained for rollback.')) return;
        queueBtn.disabled = true;
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
            showToast('Migration queued', 'success', 'Restart Foundry when you are ready to complete the upgrade.');
            if (stateEl) stateEl.textContent = 'Migration is queued. Restart Foundry to complete it.';
        } catch (error) {
            showToast('Migration could not be queued', 'error', error.message || 'Try again later.');
            queueBtn.disabled = false;
        }
    });

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
    registerNotificationActionHandlers(notificationId, migrationActions);

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
