import { registerNotificationActionHandlers, resolveNotification, showToast, showToastWithActions } from './toast.js';
import { switchTab } from './nav.js';

const MOVE_NOTIFICATION_ID = 'model-root-move-pending';
const MOVE_TOAST_SEEN_KEY = 'local-llm-foundry-model-move-toast-seen';

const CHOICES = {
    keep_legacy: {
        label: 'KEEP_LEGACY_MODEL_ROOT',
        confirmation: 'KEEP_LEGACY_MODEL_ROOT',
    },
    move_into_foundry: {
        label: 'MOVE_MODELS_INTO_FOUNDRY',
        confirmation: 'MOVE_MODELS_INTO_FOUNDRY',
    },
};

function headers() {
    return window.authHeaders ? window.authHeaders({ 'Content-Type': 'application/json' }) : {
        'Content-Type': 'application/json',
    };
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function formatSize(bytes) {
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

function selectedChoice() {
    return document.querySelector('input[name="model-root-choice"]:checked')?.value || 'keep_legacy';
}

// Models still in the legacy folder: keep a persistent notification until
// they are actually moved. Choosing "Keep legacy location" does NOT clear it.
function syncMovePendingNotification(pending) {
    const actions = [{
        id: 'review-model-move',
        label: 'Review model location',
        primary: true,
        handler: () => {
            switchTab('settings');
            import('./settings.js').then(({ openSettingsModal }) => openSettingsModal('migration'));
        },
    }];
    // Always register so a notification saved by an earlier session stays clickable.
    registerNotificationActionHandlers(MOVE_NOTIFICATION_ID, actions);
    if (!pending) {
        resolveNotification(MOVE_NOTIFICATION_ID, 'Models now live in the Foundry home.');
        return;
    }
    let seen = false;
    try {
        seen = sessionStorage.getItem(MOVE_TOAST_SEEN_KEY) === '1';
        if (!seen) sessionStorage.setItem(MOVE_TOAST_SEEN_KEY, '1');
    } catch {
        // Storage failure must not block the notification itself.
    }
    // The notification list entry persists in localStorage across reloads and
    // sessions; the pop-up toast is shown once per browser session.
    if (seen) return;
    showToastWithActions(
        'Models still in the old folder',
        'warning',
        'Move your model library into the Foundry home. Nothing moves until you confirm.',
        actions,
        { notificationId: MOVE_NOTIFICATION_ID, duration: 12000 },
    );
}

export async function initModelRootMigration() {
    const card = document.getElementById('model-root-relocation-card');
    if (!card) return;
    const previewButton = document.getElementById('model-root-relocation-preview');
    const executeButton = document.getElementById('model-root-relocation-execute');
    const planElement = document.getElementById('model-root-relocation-plan');
    let plan = null;
    let status = null;

    try {
        const response = await window.authFetch('/api/models/root-relocation/status', {
            headers: window.authHeaders(),
        });
        if (!response.ok) return;
        status = await response.json();
        setText('model-root-relocation-state', status.relocation_required
            ? 'Legacy model resources are available for an explicit choice.'
            : 'No legacy model-root relocation is required.');
        setText('model-root-relocation-summary', status.source
            ? `Current source: ${status.source} · Foundry destination: ${status.destination}`
            : 'Model-root status is unavailable.');
        if (status.custom_root) {
            setText('model-root-relocation-state', 'A custom model root is active; it will not be moved implicitly.');
            previewButton.disabled = true;
        }
        if (!status.relocation_required) {
            previewButton.disabled = true;
        }
        // A recorded "keep legacy" selection still counts as pending: the goal
        // is for everyone to end up on the Foundry home.
        syncMovePendingNotification(!!status.relocation_required && !status.custom_root);
    } catch {
        setText('model-root-relocation-state', 'Could not read model-root status.');
        previewButton.disabled = true;
    }

    document.querySelectorAll('input[name="model-root-choice"]').forEach((input) => {
        input.addEventListener('change', () => {
            plan = null;
            executeButton.disabled = true;
            if (planElement) planElement.hidden = true;
        });
    });

    previewButton?.addEventListener('click', async () => {
        const choice = selectedChoice();
        previewButton.disabled = true;
        try {
            const response = await window.authFetch('/api/models/root-relocation/preview', {
                method: 'POST',
                headers: headers(),
                body: JSON.stringify({ choice }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload.plan_id) throw new Error(payload.error || 'Preview unavailable');
            plan = payload;
            if (planElement) {
                planElement.hidden = false;
                const moving = payload.choice === 'move_into_foundry';
                const external = Array.isArray(payload.retained_external_roots) ? payload.retained_external_roots : [];
                const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
                set('model-root-plan-size-label', moving ? 'Size to move' : 'Size staying in place');
                set('model-root-plan-size', formatSize(payload.total_move_bytes));
                set('model-root-plan-items', (payload.entries?.length || 0).toLocaleString());
                set('model-root-plan-extra', 'None');
                set('model-root-plan-external', external.length ? `${external.length}: ${external.join(', ')}` : 'None');
                set('model-root-plan-id', payload.plan_id);
                const notes = document.getElementById('model-root-plan-notes');
                if (notes) {
                    const lines = moving
                        ? ['The whole model folder is moved in one step. Nothing is copied.',
                           'The legacy model folder is gone afterward.',
                           'Restart Foundry after saving to use the new location.']
                        : ['Models stay where they are.',
                           'You will keep being reminded to move them into the Foundry home.'];
                    notes.replaceChildren(...lines.map((text) => {
                        const li = document.createElement('li');
                        li.textContent = text;
                        return li;
                    }));
                }
            }
            executeButton.disabled = false;
        } catch (error) {
            showToast('Model-root preview failed', 'error', error.message || 'Try again later.');
        } finally {
            previewButton.disabled = false;
        }
    });

    executeButton?.addEventListener('click', async () => {
        if (!plan) return;
        const choice = selectedChoice();
        const choiceInfo = CHOICES[choice];
        if (!choiceInfo || !window.confirm(choice === 'move_into_foundry'
            ? 'Move the model library into Foundry? Files are moved, not copied, so no second model set is created. The legacy model folder will no longer exist.'
            : 'Keep models in the legacy location?')) return;
        executeButton.disabled = true;
        try {
            const tokenResponse = await window.authFetch('/api/db/admin-token', {
                headers: window.authHeaders(),
            });
            const tokenPayload = await tokenResponse.json().catch(() => ({}));
            if (!tokenResponse.ok || !tokenPayload.token) throw new Error('Administrator authorization is unavailable.');
            const response = await fetch('/api/models/root-relocation/execute', {
                method: 'POST',
                headers: { Authorization: `Bearer ${tokenPayload.token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    plan_id: plan.plan_id,
                    choice,
                    confirmation: choiceInfo.confirmation,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload.ok) throw new Error(payload.error || 'Could not save model-root choice.');
            if (choice === 'move_into_foundry') {
                resolveNotification(MOVE_NOTIFICATION_ID, 'Models now live in the Foundry home.');
            }
            setText('model-root-relocation-state', choice === 'move_into_foundry'
                ? 'Models moved. Restart Foundry to use the new model location.'
                : 'Choice saved. Restart Foundry to activate the selected model root.');
            showToast('Model-root choice saved', 'success', choice === 'move_into_foundry'
                ? 'The model library now lives in the Foundry home; no duplicate copy was created.'
                : 'Models remain in the legacy location.');
            executeButton.disabled = true;
            plan = null;
        } catch (error) {
            showToast('Model-root relocation failed', 'error', error.message || 'Try again later.');
            executeButton.disabled = false;
        }
    });
}
