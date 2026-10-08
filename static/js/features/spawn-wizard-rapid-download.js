// Rapid-MLX model download step (wizard page 2).
//
// llama.cpp/GGUF models are downloaded from the wizard before the server starts.
// Rapid-MLX does the same: if the selected alias/repo isn't complete in the cache the
// launch environment uses, this panel offers a tracked download (progress + cancel)
// and holds Next until it finishes, so the launch never blocks on rapid-mlx's
// interactive "Continue? [Y/n]" prompt.
import { showToast } from './toast.js';
import { wizardState, refreshStepGuardrails } from './spawn-wizard.js';

const POLL_MS = 1500;
let currentJob = null;
let pollTimer = null;
let checkSeq = 0;

const $ = (id) => document.getElementById(id);

function headers(json = false) {
  const base = window.authHeaders ? window.authHeaders() : {};
  return json ? { ...base, 'Content-Type': 'application/json' } : base;
}

function fmtGiB(bytes) {
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

/** Alias or owner/repo the Rapid-MLX launch will resolve, or '' for local paths. */
function selectedSource() {
  const model = wizardState.model;
  const typed = model.rapidMlxSource || model.localMeta?.model_source || null;
  if (typed?.kind === 'alias') return String(typed.value || '').trim();
  if (typed?.kind === 'hugging_face_repo') return String(typed.repo_id || '').trim();
  if (model.source === 'hf' && model.hfRepo) return String(model.hfRepo).trim();
  return '';
}

/** True while the selected Rapid-MLX model still needs downloading. */
export function isRapidModelMissing() {
  return wizardState.engine.selected === 'rapid_mlx' && !!wizardState.model.rapidDownload?.missing;
}

function setView(view) {
  for (const id of ['idle', 'progress', 'complete']) {
    const node = $(`rapid-dlp-${id}`);
    if (node) node.style.display = id === view ? '' : 'none';
  }
}

function stopPolling() {
  clearTimeout(pollTimer);
  pollTimer = null;
}

function setMissing(info) {
  wizardState.model.rapidDownload = info;
  refreshStepGuardrails();
}

function showIdle(info, note) {
  const panel = $('rapid-dl-panel');
  if (!panel) return;
  panel.style.display = '';
  setView('idle');
  const size = info.size_bytes ? ` (${fmtGiB(info.size_bytes)})` : '';
  $('rapid-dlp-repo').textContent = `${info.repo_id}${size}`;
  const noteEl = $('rapid-dlp-note');
  noteEl.textContent = note || 'Download the model to your models folder before starting the server.';
  const btn = $('rapid-dlp-download-btn');
  btn.disabled = false;
}

/** Re-check the selected model; show the download panel only when it is missing. */
export async function refreshRapidModelDownload() {
  const panel = $('rapid-dl-panel');
  if (!panel) return;
  stopPolling();
  const seq = ++checkSeq;
  const source = wizardState.engine.selected === 'rapid_mlx' ? selectedSource() : '';
  if (!source || source.startsWith('/') || source.startsWith('~') || source.startsWith('.')) {
    panel.style.display = 'none';
    setMissing(null);
    return;
  }
  try {
    const resp = await fetch(`/api/rapid-mlx/model-status?source=${encodeURIComponent(source)}`, { headers: headers() });
    const info = resp.ok ? await resp.json() : null;
    if (seq !== checkSeq) return;
    if (!info || !info.ok || info.cached) {
      panel.style.display = 'none';
      setMissing(null);
      return;
    }
    setMissing({ missing: true, repo_id: info.repo_id, size_bytes: info.size_bytes });
    showIdle(wizardState.model.rapidDownload);
  } catch {
    if (seq !== checkSeq) return;
    // Status unavailable: don't trap the user; launch still fails fast with a clear message.
    panel.style.display = 'none';
    setMissing(null);
  }
}

async function poll() {
  if (!currentJob) return;
  try {
    const resp = await fetch(`/api/models/downloads/${encodeURIComponent(currentJob)}`, { headers: headers() });
    const job = (await resp.json().catch(() => ({}))).job || {};
    const info = wizardState.model.rapidDownload;
    if (job.state === 'running') {
      const pct = job.bytes_total > 0 ? Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100)) : 0;
      $('rapid-dlp-bar').style.width = `${pct}%`;
      $('rapid-dlp-progress-pct').textContent = job.bytes_total > 0 ? `${pct}%` : '';
      $('rapid-dlp-progress-file').textContent = job.current_file || info?.repo_id || '';
      $('rapid-dlp-stats').textContent = job.bytes_total > 0
        ? `${fmtGiB(job.bytes_done)} / ${fmtGiB(job.bytes_total)}${job.stalled ? ' \u00b7 stalled, resuming\u2026' : ''}`
        : 'Starting\u2026';
      pollTimer = setTimeout(poll, POLL_MS);
      return;
    }
    currentJob = null;
    if (job.state === 'complete') {
      setView('complete');
      setMissing(null);
      showToast(`Downloaded ${info?.repo_id || 'model'}`, 'success');
      return;
    }
    if (info) {
      showIdle(info, job.state === 'cancelled'
        ? 'Download cancelled. Finished files are kept; download again to resume.'
        : `Download failed${job.error ? `: ${job.error}` : ''}. Try again.`);
    }
  } catch {
    pollTimer = setTimeout(poll, POLL_MS * 2);
  }
}

async function startDownload() {
  const info = wizardState.model.rapidDownload;
  if (!info?.repo_id) return;
  const btn = $('rapid-dlp-download-btn');
  btn.disabled = true;
  try {
    const resp = await fetch('/api/models/downloads', {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify({ repo_id: info.repo_id, engine: 'rapid-mlx' }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) {
      showIdle(info, data.error || 'Download failed to start.');
      return;
    }
    currentJob = data.job_id;
    $('rapid-dlp-bar').style.width = '0%';
    $('rapid-dlp-progress-pct').textContent = '';
    $('rapid-dlp-progress-file').textContent = info.repo_id;
    $('rapid-dlp-stats').textContent = 'Starting\u2026';
    setView('progress');
    poll();
  } catch (err) {
    showIdle(info, `Download request failed: ${err.message || err}`);
  }
}

export function bindRapidDownloadPanel() {
  $('rapid-dlp-download-btn')?.addEventListener('click', startDownload);
  $('rapid-dlp-cancel-btn')?.addEventListener('click', async () => {
    if (!currentJob) return;
    $('rapid-dlp-cancel-btn').disabled = true;
    try {
      await fetch(`/api/models/downloads/${encodeURIComponent(currentJob)}/cancel`, {
        method: 'POST',
        headers: headers(),
      });
    } catch { /* the next poll reports the real state */ }
    $('rapid-dlp-cancel-btn').disabled = false;
  });
}
