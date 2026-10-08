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

const infoCache = new Map();

/**
 * Resolve a Rapid-MLX source (catalog alias or owner/repo) to its repo, size, quantization
 * label and cache state via /api/rapid-mlx/model-status. Memoized per source; the cache is
 * dropped on failure and after a download completes so state never goes stale.
 */
export function rapidModelInfo(source) {
    const src = String(source || '').trim();
    if (!src || isLocalPath(src)) return Promise.resolve(null);
    if (!infoCache.has(src)) {
        const pending = fetch(`/api/rapid-mlx/model-status?source=${encodeURIComponent(src)}`, { headers: headers() })
            .then(resp => (resp.ok ? resp.json() : null))
            .then(info => {
                if (!info || !info.ok) infoCache.delete(src);
                return info && info.ok ? info : null;
            })
            .catch(() => {
                infoCache.delete(src);
                return null;
            });
        infoCache.set(src, pending);
    }
    return infoCache.get(src);
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

    const finish = (state) => {
        busy = false;
        cancelBtn.hidden = true;
        cancelBtn.disabled = false;
        dlBtn.disabled = false;
        bar.hidden = state === 'idle' || state === 'complete';
        if (state === 'complete') {
            delete card.dataset.modelMissing;
            row.remove();
        }
    };

    const poll = async () => {
        if (!jobId) return;
        try {
            const resp = await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}`, { headers: headers() });
            const data = await resp.json().catch(() => ({}));
            const job = data.job || {};
            if (job.state === 'running') {
                if (job.bytes_total > 0) {
                    const pct = Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
                    fill.style.width = `${pct}%`;
                    label.textContent = `Downloading ${pct}% · ${fmtGiB(job.bytes_done)} / ${fmtGiB(job.bytes_total)}`;
                } else {
                    label.textContent = 'Downloading…';
                }
                setTimeout(poll, POLL_MS);
                return;
            }
            if (job.state === 'complete') {
                showToast(`Downloaded ${repoId}`, 'success');
                infoCache.delete(src);
                finish('complete');
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
            setTimeout(poll, POLL_MS * 2);
        }
    };

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
        cancelBtn.disabled = true;
        try {
            await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}/cancel`, {
                method: 'POST',
                headers: headers(),
            });
        } catch { /* the next poll reports the real state */ }
    });

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
