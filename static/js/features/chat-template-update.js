// Taking action on a chat-template update.
//
// A community template is installed under a stable name and tracked against its upstream
// repo's moving `main`. When upstream moves, the checker reports it. This module is the one
// place that knows how to apply that update, so the toast, the notification-bell entry, the
// Manage template modal and the wizard all do exactly the same thing.
//
// Applying an update reinstalls the template in place, under the same file name. Presets that
// point at it pick up the new content, and the previous content stays in Version history so it
// can be rolled back.

import { showToast } from './toast.js';
import { buildCommunityTemplateInstallRequest, findTemplateByName } from './chat-template-registry.js';

/** "qwen3.8-froggeric-v22.5" -> "v22.5". Empty when the string has no version tail. */
export function shortVersion(version) {
  const match = String(version || '').match(/v\d+(?:\.\d+)*$/i);
  return match ? match[0] : '';
}

/** A short phrase for what changed: "v21.3 → v22.5", else commit ids, else plain wording. */
export function describeUpdate(info = {}) {
  const from = shortVersion(info.installed_version);
  const to = shortVersion(info.upstream_version);
  if (from && to && from !== to) return `${from} → ${to}`;
  if (to && !from) return `newer: ${to}`;
  const fromRevision = String(info.installed_revision || '').slice(0, 7);
  const toRevision = String(info.upstream_revision || '').slice(0, 7);
  if (fromRevision && toRevision && fromRevision !== toRevision) return `${fromRevision} → ${toRevision}`;
  return 'newer upstream version';
}

/** The label to show for an installed template: the registry's, else a readable form of its name. */
export function templateLabel(name) {
  const known = findTemplateByName(name);
  if (known) return known.display;
  return String(name || 'template').replace(/[-_]+/g, ' ').trim();
}

/** A variant left over from the retired no-JSON transform. It has no upstream of its own. */
export function isRetiredVariant(name) {
  return String(name || '').endsWith('-no_json');
}

/**
 * The install request that refreshes `name` from upstream, or null when it cannot be refreshed
 * (a retired variant, or a template not installed from a tracked source). The request never
 * carries a revision, so it always resolves the current `main`.
 */
export function updateRequestFor({ name, sourceUrl }) {
  if (!name || isRetiredVariant(name)) return null;
  const known = findTemplateByName(name);
  if (known) return buildCommunityTemplateInstallRequest(known, true);
  const hf = String(sourceUrl || '').match(/^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/blob\/main\/(.+)$/);
  if (hf) {
    return { endpoint: '/api/chat-template/install-hf', body: { repo: hf[1], file: hf[2], name, force: true } };
  }
  return null;
}

/**
 * Reinstall a template from upstream. Resolves to the install response, `{ ok: false, error }`
 * on failure. Reports the outcome itself, so callers only react to `ok`.
 */
export async function applyTemplateUpdate({ name, sourceUrl, info }) {
  const label = templateLabel(name);
  const request = updateRequestFor({ name, sourceUrl });
  if (!request) {
    const error = isRetiredVariant(name)
      ? 'This no-JSON variant is retired and is no longer updated. Switch to the plain template.'
      : 'This template was not installed from a tracked upstream source. Open Manage template to review it.';
    showToast('Cannot update this template', 'warning', error);
    return { ok: false, error };
  }

  try {
    const response = await fetch(request.endpoint, {
      method: 'POST',
      headers: {
        ...(window.authHeaders ? window.authHeaders() : {}),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request.body),
    });
    const data = response.ok ? await response.json() : { ok: false, error: `HTTP ${response.status}` };
    if (!data.ok) {
      showToast(`Could not update ${label}`, 'error', data.error || 'The download failed.');
      return { ok: false, error: data.error || 'The download failed.' };
    }
    const message = info ? `Updated (${describeUpdate(info)}). The previous version stays in Version history.`
      : 'Updated. The previous version stays in Version history.';
    showToast(`${label} updated`, 'success', message);
    window.dispatchEvent(new CustomEvent('chatTemplateUpdated', {
      detail: { name, path: data.path, revision: data.revision || null },
    }));
    return data;
  } catch (error) {
    const text = error?.message || String(error);
    showToast(`Could not update ${label}`, 'error', text);
    return { ok: false, error: text };
  }
}
