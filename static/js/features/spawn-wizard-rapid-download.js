// Rapid-MLX model download step (wizard page 2).
//
// llama.cpp/GGUF models are downloaded from the wizard before the server starts.
// Rapid-MLX does the same: if the selected alias/repo isn't complete in the cache the
// launch environment uses, this panel offers a tracked download (progress + cancel)
// and holds Next until it finishes, so the launch never blocks on rapid-mlx's
// interactive "Continue? [Y/n]" prompt.
import { showToast } from './toast.js';
import { wizardState, refreshStepGuardrails } from './spawn-wizard.js';
import {
  activeRapidDownloadJob,
  notifyRapidModelDownloaded,
  trackRapidDownloadJob,
  untrackRapidDownloadJob,
} from './rapid-model-download.js';

const POLL_MS = 1500;
// The download this panel is currently attached to: { source, info, id }. `id` is null
// while the start request is in flight. Every async continuation compares against the
// object it captured, so a stale job can never touch the panel or wizardState after the
// user moved to another model.
let currentJob = null;
let pollTimer = null;
let checkSeq = 0;
// repo_id of a download this panel detached from because the source changed; the job keeps
// running server-side (and stays tracked), the panel says so instead of silently dropping it.
let detachedRepo = null;

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
  const base = note || 'Download the model to your models folder before starting the server.';
  noteEl.textContent = detachedRepo && detachedRepo !== info.repo_id
    ? `${base} Download of ${detachedRepo} continues in the background.`
    : base;
  const btn = $('rapid-dlp-download-btn');
  btn.disabled = false;
}

/** Progress view for a running (or starting) download; Download stays disabled. */
function showProgress(job, text) {
  const panel = $('rapid-dl-panel');
  if (!panel) return;
  panel.style.display = '';
  setView('progress');
  $('rapid-dlp-download-btn').disabled = true;
  $('rapid-dlp-bar').style.width = '0%';
  $('rapid-dlp-progress-pct').textContent = '';
  $('rapid-dlp-progress-file').textContent = job.info.repo_id;
  $('rapid-dlp-stats').textContent = text || 'Starting\u2026';
}

/** Stop polling and release the current job without cancelling it; it keeps running. */
function detachCurrentJob() {
  stopPolling();
  if (currentJob) {
    detachedRepo = currentJob.info?.repo_id || null;
    currentJob = null;
  }
}

/** Re-check the selected model; show the download panel only when it is missing. */
export async function refreshRapidModelDownload() {
  const panel = $('rapid-dl-panel');
  if (!panel) return;
  const seq = ++checkSeq;
  const source = wizardState.engine.selected === 'rapid_mlx' ? selectedSource() : '';
  const local = !source || source.startsWith('/') || source.startsWith('~') || source.startsWith('.');

  // Same source with a download already running/starting: keep it. Re-checking would flip
  // the panel to idle and invite a duplicate start while the first job keeps running.
  if (currentJob && !local && currentJob.source === source) {
    setMissing(currentJob.info);
    panel.style.display = '';
    setView('progress');
    $('rapid-dlp-download-btn').disabled = true;
    if (currentJob.id) void poll(currentJob); // no-op while a poll chain is already live
    return;
  }
  // Source changed (or became local / non-Rapid): detach explicitly rather than orphaning.
  detachCurrentJob();
  if (local) {
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
    const missing = wizardState.model.rapidDownload;
    // A download for this repo may already be running (started here earlier, or on a card).
    const running = activeRapidDownloadJob(missing.repo_id);
    if (running) {
      if (detachedRepo === missing.repo_id) detachedRepo = null;
      currentJob = { source, info: missing, id: running };
      showProgress(currentJob, 'Resuming\u2026');
      void poll(currentJob);
      return;
    }
    showIdle(missing);
  } catch {
    if (seq !== checkSeq) return;
    // Status unavailable: don't trap the user; launch still fails fast with a clear message.
    panel.style.display = 'none';
    setMissing(null);
  }
}

async function poll(job, fromTimer = false) {
  if (currentJob !== job || !job.id) return;
  if (job.polling && !fromTimer) return; // one poll chain per job
  job.polling = true;
  stopPolling();
  let rescheduled = false;
  const again = (ms) => {
    rescheduled = true;
    pollTimer = setTimeout(() => poll(job, true), ms);
  };
  try {
    const resp = await fetch(`/api/models/downloads/${encodeURIComponent(job.id)}`, { headers: headers() });
    const data = await resp.json().catch(() => ({}));
    // Stale guard: the user switched model/source (or restarted) while this was in flight.
    if (currentJob !== job || selectedSource() !== job.source) return;
    const status = data.job || {};
    if (status.state === 'running') {
      const pct = status.bytes_total > 0 ? Math.min(100, Math.round((status.bytes_done / status.bytes_total) * 100)) : 0;
      $('rapid-dlp-bar').style.width = `${pct}%`;
      $('rapid-dlp-progress-pct').textContent = status.bytes_total > 0 ? `${pct}%` : '';
      $('rapid-dlp-progress-file').textContent = status.current_file || job.info.repo_id || '';
      $('rapid-dlp-stats').textContent = status.bytes_total > 0
        ? `${fmtGiB(status.bytes_done)} / ${fmtGiB(status.bytes_total)}${status.stalled ? ' \u00b7 stalled, resuming\u2026' : ''}`
        : 'Starting\u2026';
      again(POLL_MS);
      return;
    }
    currentJob = null;
    untrackRapidDownloadJob(job.info.repo_id, job.id);
    if (status.state === 'complete') {
      setView('complete');
      $('rapid-dlp-download-btn').disabled = false;
      setMissing(null);
      notifyRapidModelDownloaded(job.info.repo_id);
      showToast(`Downloaded ${job.info.repo_id || 'model'}`, 'success');
      return;
    }
    showIdle(job.info, status.state === 'cancelled'
      ? 'Download cancelled. Finished files are kept; download again to resume.'
      : `Download failed${status.error ? `: ${status.error}` : ''}. Try again.`);
  } catch {
    if (currentJob === job) again(POLL_MS * 2);
  } finally {
    if (!rescheduled) job.polling = false;
  }
}

async function startDownload() {
  const info = wizardState.model.rapidDownload;
  if (!info?.repo_id || currentJob) return; // a download is already running/starting
  const job = { source: selectedSource(), info, id: null };
  currentJob = job;
  detachedRepo = null;
  showProgress(job);
  try {
    const resp = await fetch('/api/models/downloads', {
      method: 'POST',
      headers: headers(true),
      body: JSON.stringify({ repo_id: info.repo_id, engine: 'rapid-mlx' }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || !data.ok) {
      if (currentJob === job) {
        currentJob = null;
        showIdle(info, data.error || 'Download failed to start.');
      }
      return;
    }
    trackRapidDownloadJob(info.repo_id, data.job_id);
    // The user moved to another model while the request was in flight: the job is tracked,
    // so returning to this source resumes it; nothing here may touch the new selection.
    if (currentJob !== job) return;
    job.id = data.job_id;
    void poll(job);
  } catch (err) {
    if (currentJob === job) {
      currentJob = null;
      showIdle(info, `Download request failed: ${err.message || err}`);
    }
  }
}

export function bindRapidDownloadPanel() {
  $('rapid-dlp-download-btn')?.addEventListener('click', startDownload);
  $('rapid-dlp-cancel-btn')?.addEventListener('click', async () => {
    const job = currentJob;
    if (!job?.id) return; // nothing to cancel yet (start request in flight, or already ended)
    const btn = $('rapid-dlp-cancel-btn');
    btn.disabled = true;
    try {
      await fetch(`/api/models/downloads/${encodeURIComponent(job.id)}/cancel`, {
        method: 'POST',
        headers: headers(),
      });
    } catch { /* the next poll reports the real state */ }
    btn.disabled = false;
  });
}
