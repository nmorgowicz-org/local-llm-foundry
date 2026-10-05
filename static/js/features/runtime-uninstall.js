// Shared runtime uninstall flow.
//
// One contract for every managed runtime loader (Rapid-MLX environments and
// the llama.cpp binary): show the bytes the runtime occupies, require an
// explicit typed-free confirm, then remove the runtime itself. Downloaded
// models are never touched — they live outside the runtime roots and are the
// expensive part users want to keep.

const UNINSTALL_ENDPOINTS = {
  rapid_mlx: '/api/rapid-mlx/runtime/uninstall',
  llama_cpp: '/api/llama-binary/uninstall',
};

const UNINSTALL_LABELS = {
  rapid_mlx: 'Rapid-MLX',
  llama_cpp: 'llama.cpp',
};

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
  actions.append(cancel, go);
  dialog.append(title, body, actions);
  overlay.appendChild(dialog);
  const close = () => overlay.remove();
  overlay.addEventListener('click', (ev) => {
    if (ev.target === overlay) close();
  });
  overlay.querySelector('.runtime-uninstall-cancel').addEventListener('click', close);
  overlay.querySelector('.runtime-uninstall-go').addEventListener('click', async (ev) => {
    const btn = ev.currentTarget;
    btn.disabled = true;
    btn.textContent = 'Uninstalling…';
    try {
      await onConfirm();
      close();
    } catch (err) {
      btn.textContent = 'Uninstall';
      btn.disabled = false;
      showToast(`Uninstall failed: ${err.message}`, 'error');
    }
  });
  document.body.appendChild(overlay);
  overlay.querySelector('.runtime-uninstall-go').focus();
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
    const headers = { ...(window.authHeaders ? window.authHeaders() : {}) };
    const resp = await fetch(endpoint, { method: 'DELETE', headers });
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

function showToast(message, type) {
  if (typeof window.showToast === 'function') {
    window.showToast(message, type);
  } else {
    import('./toast.js').then(m => m.showToast?.(message, type)).catch(() => {});
  }
}
