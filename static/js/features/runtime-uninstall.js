// Shared runtime uninstall flow.
//
// One contract for every managed runtime loader (Rapid-MLX environments and
// the llama.cpp binary): show the bytes the runtime occupies, require an
// explicit typed-free confirm, then remove the runtime itself. Downloaded
// models are never touched — they live outside the runtime roots and are the
// expensive part users want to keep.

import { showToast } from './toast.js';

const UNINSTALL_ENDPOINTS = {
  rapid_mlx: '/api/rapid-mlx/runtime/uninstall',
  llama_cpp: '/api/llama-binary/uninstall',
};

const UNINSTALL_LABELS = {
  rapid_mlx: 'Rapid-MLX',
  llama_cpp: 'llama.cpp',
};

// Must match the backend constants in src/web/api/runtime_uninstall.rs.
const UNINSTALL_CONFIRM_PHRASES = {
  rapid_mlx: 'UNINSTALL_RAPID_MLX',
  llama_cpp: 'UNINSTALL_LLAMA_CPP',
};

// Uninstall is db-admin gated server-side; trade the api token for the
// db-admin token the same way the database admin panel does.
async function ensureDbAdminToken() {
  const headers = window.authHeaders ? window.authHeaders() : {};
  const res = await fetch('/api/db/admin-token', { headers });
  if (!res.ok) throw new Error('Could not obtain admin authorization for uninstall');
  const data = await res.json();
  if (!data.token) throw new Error('Admin authorization unavailable');
  return data.token;
}

export function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export async function fetchRuntimeStorage() {
  const headers = window.authHeaders ? window.authHeaders() : {};
  const resp = await fetch('/api/runtimes/storage', { headers });
  if (!resp.ok) throw new Error(`Storage query failed (${resp.status})`);
  return resp.json();
}

// Confirm overlay: shared look for both modals, no browser confirm().
function confirmUninstall(kind, bytes, onConfirm) {
  const label = UNINSTALL_LABELS[kind] || kind;
  const overlay = document.createElement('div');
  overlay.className = 'runtime-uninstall-confirm-overlay';
  const dialog = document.createElement('div');
  dialog.className = 'runtime-uninstall-confirm';
  dialog.setAttribute('role', 'alertdialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-label', `Uninstall ${label}`);
  const title = document.createElement('h3');
  title.textContent = `Uninstall ${label}?`;
  const body = document.createElement('p');
  body.textContent = `This removes the ${label} runtime (${formatBytes(bytes)}) from this machine. Downloaded models are kept. You can reinstall at any time.`;
  const phrase = UNINSTALL_CONFIRM_PHRASES[kind] || kind.toUpperCase();
  const prompt = document.createElement('p');
  prompt.className = 'runtime-uninstall-confirm-prompt';
  prompt.textContent = `Type ${phrase} to confirm.`;
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'runtime-uninstall-confirm-input';
  input.setAttribute('aria-label', `Type ${phrase} to confirm`);
  input.autocomplete = 'off';
  input.spellcheck = false;
  const actions = document.createElement('div');
  actions.className = 'runtime-uninstall-confirm-actions';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.className = 'runtime-uninstall-cancel';
  cancel.textContent = 'Cancel';
  const go = document.createElement('button');
  go.type = 'button';
  go.className = 'runtime-uninstall-go runtime-uninstall-go--danger';
  go.textContent = 'Uninstall';
  go.disabled = true;
  input.addEventListener('input', () => {
    go.disabled = input.value.trim() !== phrase;
  });
  actions.append(cancel, go);
  dialog.append(title, body, prompt, input, actions);
  overlay.appendChild(dialog);
  const opener = document.activeElement;
  let busy = false;
  // Registered on window in the capture phase so Escape and Tab are handled here before the
  // runtime manage modal underneath (which traps keys on document) can react to them.
  const onKeydown = (ev) => {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      ev.stopPropagation();
      if (!busy) close();
      return;
    }
    if (ev.key !== 'Tab') return;
    const stops = [input, cancel, go].filter((el) => !el.disabled);
    ev.preventDefault();
    ev.stopPropagation();
    if (!stops.length) return;
    const index = stops.indexOf(document.activeElement);
    const next = ev.shiftKey ? (index <= 0 ? stops.length - 1 : index - 1) : (index + 1) % stops.length;
    stops[next].focus();
  };
  function close() {
    window.removeEventListener('keydown', onKeydown, true);
    overlay.remove();
    if (opener && opener.isConnected && typeof opener.focus === 'function') opener.focus();
  }
  window.addEventListener('keydown', onKeydown, true);
  overlay.addEventListener('click', (ev) => {
    if (ev.target === overlay && !busy) close();
  });
  cancel.addEventListener('click', () => { if (!busy) close(); });
  go.addEventListener('click', async (ev) => {
    const btn = ev.currentTarget;
    busy = true;
    cancel.disabled = true;
    btn.disabled = true;
    btn.textContent = 'Uninstalling…';
    try {
      await onConfirm();
      close();
    } catch (err) {
      busy = false;
      cancel.disabled = false;
      btn.textContent = 'Uninstall';
      btn.disabled = false;
      showToast(`Uninstall failed: ${err.message}`, 'error');
    }
  });
  document.body.appendChild(overlay);
  // A destructive dialog opens on the safe choice.
  input.focus();
}

export async function uninstallRuntime(kind, { onDone } = {}) {
  const endpoint = UNINSTALL_ENDPOINTS[kind];
  if (!endpoint) throw new Error(`Unknown runtime: ${kind}`);
  let bytes = 0;
  try {
    const storage = await fetchRuntimeStorage();
    bytes = kind === 'rapid_mlx'
      ? (storage.rapid_mlx_bytes || 0)
      : (storage.llama_bin_bytes || 0);
  } catch { /* size is advisory */ }

  confirmUninstall(kind, bytes, async () => {
    const adminToken = await ensureDbAdminToken();
    const resp = await fetch(endpoint, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${adminToken}` },
      body: JSON.stringify({ confirm: UNINSTALL_CONFIRM_PHRASES[kind] }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.ok === false) {
      throw new Error(data.error || `Uninstall failed (${resp.status})`);
    }
    showToast(`${UNINSTALL_LABELS[kind]} uninstalled. Downloaded models were kept.`, 'success');
    onDone?.();
  });
}

// Wire an uninstall button that lives inside a runtime manage modal.
export function wireUninstallButton(button, kind, { onDone } = {}) {
  if (!button) return;
  button.addEventListener('click', () => {
    uninstallRuntime(kind, { onDone }).catch((err) => {
      showToast(`Uninstall failed: ${err.message}`, 'error');
    });
  });
}
