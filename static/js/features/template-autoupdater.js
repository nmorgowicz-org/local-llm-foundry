// ── Template Autoupdater ───────────────────────────────────────────────────────
// Quiet, 12-hour checker for installed community chat templates (Qwen, Gemma, and anything else
// installed with an upstream source). Uses /api/chat-template/active and /check-update.
// Modeled after llama-updater.js: visibility-aware, no spam.
//
// When upstream has moved, the user is told in a way they can act on:
//   - one toast, with Update (or Update all) and Review buttons;
//   - one persistent notification-bell entry per template, with the same actions.
// Nothing is installed without the user choosing Update. An update the user has already been
// told about is not announced again, so re-checking never repeats a toast.

import { applyTemplateUpdate, describeUpdate, isRetiredVariant, templateLabel } from './chat-template-update.js';
import {
  registerNotificationActionHandlers,
  registerPersistentNotification,
  resolveNotification,
  showToastWithActions,
} from './toast.js';

const STORAGE_LAST_CHECK = 'template_autoupdater_lastCheck';
const STORAGE_BUSY = 'template_autoupdater_busy';
const STORAGE_LAST_STATUS = 'template_autoupdater_lastStatus';
const STORAGE_ANNOUNCED = 'template_autoupdater_announced';
const STORAGE_LAST_ATTEMPT = 'template_autoupdater_lastAttempt';

const INTERVAL_MS = 12 * 60 * 60 * 1000; // 12 hours
const BUSY_TTL_MS = 60 * 1000; // busy guard: 1 minute
const RETRY_BACKOFF_MS = 15 * 60 * 1000; // after a failed check, try again no sooner than this
const TOAST_DURATION_MS = 30 * 1000; // long enough to read and act; the bell entry persists anyway

let _intervalId = null;

export function readLastStatus() {
  try {
    const v = localStorage.getItem(STORAGE_LAST_STATUS);
    if (!v) return { templates_with_updates: [] };
    const obj = JSON.parse(v);
    if (!obj || !Array.isArray(obj.templates_with_updates)) {
      return { templates_with_updates: [] };
    }
    return obj;
  } catch {
    return { templates_with_updates: [] };
  }
}

export function initTemplateAutoupdater() {
  // Bell entries survive a reload, but their buttons are bound in memory. Rebind them now so an
  // Update button in the bell works before the first check has run.
  for (const update of readLastStatus().templates_with_updates) {
    registerNotificationActionHandlers(notificationIdFor(update.name), actionsFor(update));
  }

  window.addEventListener('chatTemplateUpdated', (event) => {
    const name = event.detail?.name;
    if (!name) return;
    resolveNotification(notificationIdFor(name), 'Updated.');
    forgetUpdate(name);
  });

  // Start after a short delay; respect visibility.
  setTimeout(() => {
    if (!document.hidden) {
      scheduleNextCheck();
    }
  }, 3000);

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      // Respect the 12-hour interval. Re-checking on every tab focus produced repeated toasts.
      scheduleNextCheck();
    } else {
      stopInterval();
    }
  });
}

function nowTs() {
  return Date.now();
}

function isBusy() {
  try {
    const ts = Number(localStorage.getItem(STORAGE_BUSY) || '0');
    if (!ts) return false;
    return (nowTs() - ts) < BUSY_TTL_MS;
  } catch {
    return false;
  }
}

function setBusy() {
  try {
    localStorage.setItem(STORAGE_BUSY, String(nowTs()));
  } catch {
    // ignore storage errors
  }
}

function clearBusy() {
  try {
    localStorage.removeItem(STORAGE_BUSY);
  } catch {
    // ignore
  }
}

function getLastCheck() {
  try {
    return Number(localStorage.getItem(STORAGE_LAST_CHECK) || '0');
  } catch {
    return 0;
  }
}

// Only a check that reached the server and got a template list counts as "checked". A failed
// attempt is recorded separately so it backs off briefly instead of waiting out the 12 hours.
function setLastCheck(ts) {
  try {
    localStorage.setItem(STORAGE_LAST_CHECK, String(ts));
  } catch {
    // ignore
  }
}

function getLastAttempt() {
  try {
    return Number(localStorage.getItem(STORAGE_LAST_ATTEMPT) || '0');
  } catch {
    return 0;
  }
}

function setLastAttempt(ts) {
  try {
    localStorage.setItem(STORAGE_LAST_ATTEMPT, String(ts));
  } catch {
    // ignore
  }
}

function shouldRunCheck() {
  if (document.hidden) return false;
  if (isBusy()) return false;

  const last = getLastCheck();
  if (last > 0 && nowTs() - last < INTERVAL_MS) return false;
  const attempt = getLastAttempt();
  if (attempt > 0 && nowTs() - attempt < RETRY_BACKOFF_MS) return false;
  return true;
}

function scheduleNextCheck() {
  stopInterval();

  // If we should run now, do it, then schedule future interval.
  if (shouldRunCheck()) {
    // Small debounce so we don't fire immediately on every visibility change.
    setTimeout(() => {
      checkTemplateUpdates().finally(() => {
        startInterval();
      });
    }, 1500);
  } else {
    startInterval();
  }
}

function startInterval() {
  stopInterval();
  _intervalId = setInterval(() => {
    if (document.hidden) return;
    if (!shouldRunCheck()) return;
    checkTemplateUpdates();
  }, INTERVAL_MS);
}

function stopInterval() {
  if (_intervalId != null) {
    clearInterval(_intervalId);
    _intervalId = null;
  }
}

// ── What the user is told, and what they can do about it ────────────────────

function notificationIdFor(name) {
  return `chat-template-update:${name}`;
}

function readAnnounced() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_ANNOUNCED) || '{}');
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeAnnounced(value) {
  try {
    localStorage.setItem(STORAGE_ANNOUNCED, JSON.stringify(value));
  } catch {
    // ignore
  }
}

function writeStatus(updates) {
  try {
    localStorage.setItem(STORAGE_LAST_STATUS, JSON.stringify({ templates_with_updates: updates }));
  } catch {
    // ignore
  }
}

// Forget one template's pending update once it has been applied.
function forgetUpdate(name) {
  writeStatus(readLastStatus().templates_with_updates.filter(item => item.name !== name));
  const announced = readAnnounced();
  delete announced[name];
  writeAnnounced(announced);
}

function messageFor(update) {
  return `${templateLabel(update.name)} has a new upstream version (${describeUpdate(update)}).`;
}

// Cleanup after a successful update (resolving the bell entry, forgetting the pending status)
// happens in the `chatTemplateUpdated` listener below, so an update applied from anywhere (the
// toast, the bell, the Manage modal, the wizard) leaves the same state behind.
async function updateTemplate(update) {
  return applyTemplateUpdate({ name: update.name, sourceUrl: update.source_url, info: update });
}

async function updateAll(updates) {
  for (const update of updates) {
    await updateTemplate(update);
  }
}

// Opens the Manage template modal, where the user can read the version history and the upstream
// commits before deciding.
async function reviewTemplate(update) {
  const { openChatTemplateManageModal, repoFromSourceUrl } = await import('./chat-template-panel.js');
  await openChatTemplateManageModal({
    tplName: update.name,
    tplRepo: repoFromSourceUrl(update.source_url),
    currentPath: update.path,
    activePath: update.path,
    onActivated: async () => {
      await checkTemplateUpdates();
    },
  });
}

function actionsFor(update) {
  return [
    { id: 'update', label: 'Update', primary: true, handler: () => updateTemplate(update) },
    { id: 'review', label: 'Review', handler: () => reviewTemplate(update) },
  ];
}

function showSummaryToast(fresh) {
  const single = fresh.length === 1;
  showToastWithActions(
    single ? 'Chat template update available' : `${fresh.length} chat template updates available`,
    'info',
    `${fresh.map(update => `${templateLabel(update.name)} (${describeUpdate(update)})`).join('; ')}.`,
    [
      { id: 'update', label: single ? 'Update' : 'Update all', primary: true, handler: () => updateAll(fresh) },
      { id: 'review', label: 'Review', handler: () => reviewTemplate(fresh[0]) },
    ],
    { duration: TOAST_DURATION_MS },
  );
}

function announce(updates, unverified = new Set()) {
  const announced = readAnnounced();
  const fresh = updates.filter(update => announced[update.name] !== update.current_sha256);

  for (const update of updates) {
    if (fresh.includes(update)) {
      registerPersistentNotification(
        notificationIdFor(update.name),
        'Chat template update available',
        'info',
        messageFor(update),
        actionsFor(update),
      );
    } else {
      // Already announced. The user may have archived it, which must stick, so only make sure a
      // restored entry's buttons work.
      registerNotificationActionHandlers(notificationIdFor(update.name), actionsFor(update));
    }
  }

  const next = {};
  for (const update of updates) next[update.name] = update.current_sha256;
  // A template whose check failed this time is unknown, not up to date: keep what the user was
  // already told so the next successful check does not announce it a second time.
  for (const name of unverified) {
    if (!(name in next) && name in announced) next[name] = announced[name];
  }
  writeAnnounced(next);

  if (fresh.length > 0) showSummaryToast(fresh);
}

// ── The check ────────────────────────────────────────────────────────────────

/**
 * Check every installed template against upstream and announce what has changed. Resolves to
 * `{ changedTemplates }`. Network and parse failures are swallowed; the next interval retries.
 */
export async function checkTemplateUpdates() {
  const changedTemplates = [];
  if (document.hidden) return { changedTemplates };
  setBusy();
  setLastAttempt(nowTs());

  try {
    const resp = await (await fetch('/api/chat-template/active', {
      headers: window.authHeaders ? window.authHeaders() : {},
    })).json();

    if (!resp.ok || !Array.isArray(resp.templates)) {
      return { changedTemplates };
    }
    setLastCheck(nowTs());

    const seen = new Set();
    const unverified = new Set();
    for (const tpl of resp.templates) {
      if (!tpl.path) continue;
      // Leftovers of the retired no-JSON transform have no upstream of their own to update from.
      if (tpl.legacy_variant || isRetiredVariant(tpl.name)) continue;
      seen.add(tpl.name);
      try {
        const checkResp = await fetch('/api/chat-template/check-update', {
          method: 'POST',
          headers: {
            ...(window.authHeaders ? window.authHeaders() : {}),
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ path: tpl.path }),
        });

        if (!checkResp.ok) {
          unverified.add(tpl.name);
          continue;
        }
        const data = await checkResp.json();

        if (data.ok === true && data.changed === true) {
          changedTemplates.push({
            name: tpl.name,
            path: tpl.path,
            source_url: tpl.source_url,
            current_sha256: data.current_sha256,
            installed_version: data.installed_version || null,
            upstream_version: data.upstream_version || null,
            installed_revision: data.installed_revision || null,
            upstream_revision: data.upstream_revision || null,
          });
        } else if (data.ok === true) {
          // Up to date (for instance, updated from the Manage modal). Clear any stale entry.
          resolveNotification(notificationIdFor(tpl.name), 'Up to date.');
        }
      } catch {
        // Per-template errors are not reported, and must not make the template look up to date.
        unverified.add(tpl.name);
      }
    }

    // A template that is no longer installed cannot be updated; drop its entry.
    for (const name of Object.keys(readAnnounced())) {
      if (!seen.has(name)) resolveNotification(notificationIdFor(name), 'This template is no longer installed.');
    }

    // Keep the pending entry of any template we could not re-check this time.
    const carried = readLastStatus().templates_with_updates.filter(
      item => unverified.has(item.name) && !changedTemplates.some(changed => changed.name === item.name),
    );
    writeStatus([...changedTemplates, ...carried]);
    announce(changedTemplates, unverified);

    // Dispatch event for UI consumers (e.g., spawn wizard)
    window.dispatchEvent(
      new CustomEvent('templateAutoupdateResult', {
        detail: {
          changedTemplates,
          lastCheck: nowTs(),
        },
      })
    );
  } catch {
    // Network or parse errors: silently ignored; will retry at next interval.
  } finally {
    clearBusy();
  }
  return { changedTemplates };
}
