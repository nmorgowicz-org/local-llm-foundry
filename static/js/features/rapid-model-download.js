// Download affordance for Rapid-MLX preset cards.
//
// Rapid-MLX serves models from the app-scoped HF cache. A preset whose model isn't
// complete there would block `rapid-mlx serve` on an interactive prompt, so the card
// checks first (GET /api/rapid-mlx/model-status) and offers a tracked download
// (POST /api/models/downloads) instead of letting Start fail.
import { showToast } from './toast.js';

const POLL_MS = 1500;

function headers(json = false) {
    const base = window.authHeaders ? window.authHeaders() : {};
    return json ? { ...base, 'Content-Type': 'application/json' } : base;
}

function fmtGiB(bytes) {
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

function isLocalPath(source) {
    return /^[/\\.~]/.test(source) || /^[a-zA-Z]:[\\/]/.test(source);
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

// Memoized model-status lookups: { promise, at }. Entries expire after INFO_TTL_MS (the
// cache can change outside this page, e.g. a CLI download) and are dropped on failure and
// whenever any surface in this page finishes a download (notifyRapidModelDownloaded).
const INFO_TTL_MS = 60_000;
const infoCache = new Map();

// repo_id -> job_id for downloads started from this page, so a re-rendered card or a wizard
// step change resumes the running job instead of orphaning it and offering a duplicate start.
const activeJobs = new Map();
const RAPID_MODEL_DOWNLOADED_EVENT = 'rapid-model-downloaded';

export function trackRapidDownloadJob(repoId, jobId) {
    if (repoId && jobId) activeJobs.set(repoId, jobId);
}

export function untrackRapidDownloadJob(repoId, jobId) {
    if (!repoId) return;
    if (jobId == null || activeJobs.get(repoId) === jobId) activeJobs.delete(repoId);
}

export function activeRapidDownloadJob(repoId) {
    return activeJobs.get(repoId) || null;
}

/**
 * Announce that a Rapid-MLX model finished downloading from any surface (card or wizard):
 * invalidates memoized status and lets other cards for the same repo drop their download row.
 */
export function notifyRapidModelDownloaded(repoId) {
    activeJobs.delete(repoId);
    infoCache.clear();
    window.dispatchEvent(new CustomEvent(RAPID_MODEL_DOWNLOADED_EVENT, { detail: { repoId } }));
}

/**
 * Resolve a Rapid-MLX source (catalog alias or owner/repo) to its repo, size, quantization
 * label and cache state via /api/rapid-mlx/model-status. Memoized per source with a TTL; the
 * entry is dropped on failure and when a download completes so state never goes stale.
 */
export function rapidModelInfo(source) {
    const src = String(source || '').trim();
    if (!src || isLocalPath(src)) return Promise.resolve(null);
    const existing = infoCache.get(src);
    if (existing && Date.now() - existing.at < INFO_TTL_MS) return existing.promise;
    const entry = { at: Date.now(), promise: null };
    const drop = () => { if (infoCache.get(src) === entry) infoCache.delete(src); };
    entry.promise = fetch(`/api/rapid-mlx/model-status?source=${encodeURIComponent(src)}`, { headers: headers() })
        .then(resp => (resp.ok ? resp.json() : null))
        .then(info => {
            if (!info || !info.ok) drop();
            return info && info.ok ? info : null;
        })
        .catch(() => {
            drop();
            return null;
        });
    infoCache.set(src, entry);
    return entry.promise;
}

/**
 * Decorate a Rapid-MLX launch card: add the quantization chip (the MLX analogue of a
 * GGUF Q4_K_M tag) and, when the model isn't in a usable cache, the download row.
 * @param {HTMLElement} card
 * @param {string} source catalog alias or `owner/repo`
 */
export async function attachRapidDownloadState(card, source) {
    const src = String(source || '').trim();
    const status = await rapidModelInfo(src);
    if (!status || !document.contains(card)) return;

    if (status.quant && !card.querySelector('.launch-chip--quant')) {
        const chip = el('span', 'launch-chip launch-chip--quant', status.quant);
        chip.title = `Quantization: ${status.quant}`;
        const chips = card.querySelector('.launch-card-chips');
        if (chips) chips.insertBefore(chip, chips.children[2] || null);
    }
    if (status.cached) return;

    const repoId = status.repo_id;
    const row = el('div', 'launch-card-dl');
    const label = el('div', 'launch-card-dl-label');
    const sizeText = status.size_bytes ? ` · ${fmtGiB(status.size_bytes)}` : '';
    label.textContent = `Not downloaded${sizeText}`;
    label.title = repoId;
    const bar = el('div', 'launch-card-dl-bar');
    const fill = el('div', 'launch-card-dl-fill');
    bar.appendChild(fill);
    bar.hidden = true;
    const actions = el('div', 'launch-card-dl-actions');
    const dlBtn = el('button', 'launch-card-dl-btn', 'Download');
    dlBtn.type = 'button';
    const cancelBtn = el('button', 'launch-card-dl-cancel', 'Cancel');
    cancelBtn.type = 'button';
    cancelBtn.hidden = true;
    actions.append(dlBtn, cancelBtn);
    row.append(label, bar, actions);
    const cardActions = card.querySelector('.launch-card-actions');
    if (cardActions) cardActions.before(row);
    else card.appendChild(row);

    card.dataset.modelMissing = '1';
    let jobId = null;
    let busy = false;
    let alive = true;
    let pollTimer = null;

    // Stops this card's poll loop and listener. Does not cancel the job: it keeps running
    // server-side and stays tracked, so a re-rendered card for the same repo resumes it.
    const teardown = () => {
        alive = false;
        clearTimeout(pollTimer);
        pollTimer = null;
        window.removeEventListener(RAPID_MODEL_DOWNLOADED_EVENT, onDownloaded);
    };

    const finish = (state) => {
        busy = false;
        jobId = null;
        cancelBtn.hidden = true;
        cancelBtn.disabled = false;
        dlBtn.disabled = false;
        bar.hidden = state === 'idle' || state === 'complete';
        if (state === 'complete') {
            delete card.dataset.modelMissing;
            row.remove();
        }
    };

    // Another surface (wizard or another card) finished downloading this repo.
    function onDownloaded(event) {
        if (!alive || event.detail?.repoId !== repoId) return;
        teardown();
        finish('complete');
    }
    window.addEventListener(RAPID_MODEL_DOWNLOADED_EVENT, onDownloaded);

    const schedule = (ms) => {
        clearTimeout(pollTimer);
        pollTimer = setTimeout(poll, ms);
    };

    async function poll() {
        if (!alive || !jobId) return;
        if (!document.contains(card)) { teardown(); return; }
        const polledJob = jobId;
        try {
            const resp = await fetch(`/api/models/downloads/${encodeURIComponent(polledJob)}`, { headers: headers() });
            const data = await resp.json().catch(() => ({}));
            if (!alive || jobId !== polledJob) return;
            const job = data.job || {};
            if (job.state === 'running') {
                if (job.bytes_total > 0) {
                    const pct = Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
                    fill.style.width = `${pct}%`;
                    label.textContent = `Downloading ${pct}% · ${fmtGiB(job.bytes_done)} / ${fmtGiB(job.bytes_total)}`;
                } else {
                    label.textContent = 'Downloading…';
                }
                schedule(POLL_MS);
                return;
            }
            untrackRapidDownloadJob(repoId, polledJob);
            if (job.state === 'complete') {
                showToast(`Downloaded ${repoId}`, 'success');
                teardown();
                finish('complete');
                notifyRapidModelDownloaded(repoId);
                return;
            }
            if (job.state === 'cancelled') {
                label.textContent = 'Download cancelled · finished files are kept';
                dlBtn.textContent = 'Resume';
                finish('idle');
                return;
            }
            label.textContent = `Download failed${job.error ? `: ${job.error}` : ''}`;
            dlBtn.textContent = 'Retry';
            finish('idle');
        } catch {
            if (alive && jobId === polledJob) schedule(POLL_MS * 2);
        }
    }

    const start = async () => {
        if (busy) return;
        busy = true;
        dlBtn.disabled = true;
        label.textContent = 'Starting download…';
        bar.hidden = false;
        fill.style.width = '0%';
        try {
            const resp = await fetch('/api/models/downloads', {
                method: 'POST',
                headers: headers(true),
                body: JSON.stringify({ repo_id: repoId, engine: 'rapid-mlx' }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok || !data.ok) {
                showToast(data.error || 'Download failed to start', 'error');
                label.textContent = 'Download failed to start';
                finish('idle');
                return;
            }
            trackRapidDownloadJob(repoId, data.job_id);
            if (!alive) return; // card was re-rendered mid-request; the tracked job is resumed by its replacement
            jobId = data.job_id;
            cancelBtn.hidden = false;
            poll();
        } catch (err) {
            showToast(`Download request failed: ${err.message || err}`, 'error');
            finish('idle');
        }
    };

    dlBtn.addEventListener('click', (event) => {
        event.stopPropagation();
        start();
    });
    cancelBtn.addEventListener('click', async (event) => {
        event.stopPropagation();
        if (!jobId) return; // nothing to cancel (no job yet, or it already ended)
        const cancelling = jobId;
        cancelBtn.disabled = true;
        try {
            await fetch(`/api/models/downloads/${encodeURIComponent(cancelling)}/cancel`, {
                method: 'POST',
                headers: headers(),
            });
        } catch {
            // The next poll reports the real state; let the user retry the cancel.
            if (jobId === cancelling) cancelBtn.disabled = false;
        }
    });

    // Resume a download already running for this repo (card re-rendered, or started from
    // the wizard / another card) instead of offering a second start.
    const running = activeRapidDownloadJob(repoId);
    if (running) {
        busy = true;
        jobId = running;
        dlBtn.disabled = true;
        bar.hidden = false;
        label.textContent = 'Downloading…';
        cancelBtn.hidden = false;
        poll();
    }

    // Start on a card whose model is missing downloads instead of failing at launch.
    // Capture phase on the card runs before the Start button's own handler.
    card.addEventListener('click', (event) => {
        if (card.dataset.modelMissing !== '1') return;
        if (!event.target.closest('.launch-card-btn-start')) return;
        event.preventDefault();
        event.stopPropagation();
        if (!busy) start();
        else showToast('Download in progress. Start will be available when it finishes.', 'info');
    }, true);
}
