// ── Setup / Monitor View ──────────────────────────────────────────────────────
// View transitions, animations, quick stats, and view state initialization.

import { setupViewState, chat, sessionState } from '../core/app-state.js';
import { getPlatformInfo } from '../core/platform-info.js';
import { doAttachFromSetup } from './attach-detach.js';
import { presetModelSource } from './presets.js';
import { escapeHtml } from '../core/format.js';
import { showToast, showConfirmDialog, showPromptDialog } from './toast.js';
import Router from './router.js';
import { buildEstimateBody, rapidEstimatePolicyFromConfig } from './vram-estimate.js';

// ── Model / preset classification (from GGUF-derived metadata) ────────────────
// No name-based guessing: labels come from preset.family and preset.size_class
// which are derived from GGUF metadata (architecture, parameter_count).
// For presets missing metadata, we intentionally show no family/size badge
// instead of guessing.

const FAMILY_LABEL_MAP = {
    qwen36: 'Qwen3.6',
    qwen35: 'Qwen3.5',
    qwen3: 'Qwen3',
    qwen2: 'Qwen2',
    qwen: 'Qwen',
    llama3: 'Llama',
    gemma4: 'Gemma4',
    gemma: 'Gemma',
    mistral: 'Mistral',
    exaone: 'EXAONE',
    deepseek: 'DeepSeek',
    phi: 'Phi',
    falcon: 'Falcon',
    grok: 'Grok',
    mamba: 'Mamba',
    rwkv: 'RWKV',
    olmo: 'OLMo',
    stablelm: 'StableLM',
    granite: 'Granite',
    starcoder: 'StarCoder',
};

export function confirmFreeCacheCleanup(reclaimableBytes) {
    const reclaimableGb = (reclaimableBytes / (1024 ** 3)).toFixed(1);
    return showConfirmDialog(
        'Free system cache',
        `macOS can reclaim about ${reclaimableGb} GB of cached data. Running apps are not closed, but files and apps may load a little more slowly until the cache is rebuilt.`,
        'Free cache'
    );
}

function classifyPreset(preset) {
    const family = preset.family || null;
    const sizeClass = preset.size_class || 'unknown';
    return { family, sizeClass };
}

/**
 * Build an architecture label for a preset/model.
 * Returns { display, tooltip } or null.
 * Exposed globally so spawn-wizard and presets can reuse without duplication.
 */
export function buildArchitectureLabel(p, c) {
    var kind = p.architecture_kind;
    var activeB = p.active_params_b;
    var totalB = p.param_count
        ? p.param_count / 1e9
        : (c && c.paramB) || null;

    if (!kind) {
        return null;
    }

    var fmt = function (v) {
        if (v == null) return null;
        var r = Number(v);
        if (Number.isFinite(r)) {
            if (r >= 10) return Math.round(r) + 'B';
            if (r % 1 === 0) return r + 'B';
            return r.toFixed(1) + 'B';
        }
        return null;
    };

    var activeStr = fmt(activeB);
    var totalStr = fmt(totalB);

    // Append "— N experts, M active per token" when either count is known.
    var withExpertSuffix = function (tip) {
        if (p.expert_count == null && p.expert_used_count == null) return tip;
        var ec = p.expert_count != null ? p.expert_count : '*';
        var uc = p.expert_used_count != null ? p.expert_used_count : '*';
        return tip + ' — ' + ec + ' experts, ' + uc + ' active per token';
    };

    if (kind === 'dense') {
        if (totalStr) {
            return {
                display: 'Dense • ' + totalStr,
                tooltip: 'All ' + totalStr + ' parameters active per token'
            };
        }
        return { display: 'Dense', tooltip: 'Dense (all parameters active per token)' };
    }

    if (kind === 'moe' || kind === 'hybrid_moe') {
        var isHybrid = kind === 'hybrid_moe';
        var tip = isHybrid
            ? 'MoE + hybrid attention (fewer full-KV layers) — often faster at long context'
            : 'MoE: only a subset of parameters active per token';
        if (totalStr && activeStr) {
            return {
                display: (isHybrid ? 'Hybrid MoE • ' : 'MoE • ') +
                    totalStr + ' (' + activeStr + ' active)',
                tooltip: withExpertSuffix(tip)
            };
        }
        if (totalStr) {
            return {
                display: (isHybrid ? 'Hybrid MoE • ' : 'MoE • ') + totalStr,
                tooltip: withExpertSuffix(tip)
            };
        }
        return { display: isHybrid ? 'Hybrid MoE' : 'MoE', tooltip: withExpertSuffix(tip) };
    }

    return null;
}

// ── MoE CPU offload eligibility ───────────────────────────────────────────────
// --n-cpu-moe is only valid for MoE / hybrid-moE models with real experts.
// For dense models, it either does nothing or misbehaves in llama.cpp.
export function isMoEEligible(p) {
    if (!p) return false;
    const kind = p.architecture_kind;
    const experts = p.expert_count || 0;
    return (kind === 'moe' || kind === 'hybrid_moe') && experts > 0;
}

// ── Quantization tag extraction ───────────────────────────────────────────────

function extractQuantFromFilename(filename) {
    if (!filename) return null;
    if (/mmproj/i.test(filename)) return null;
    const base = filename.replace(/\.gguf$/i, '');
    // APEX-MTP draft models: extract quality label after -APEX-MTP-
    const mtp = base.match(/-APEX-(?:MTP-)?(.+)$/i);
    if (mtp) return mtp[1];
    // Standard GGUF quant suffix after - or . separator: Q4_K_M, IQ2_XXS, BF16, F16, F32 …
    const m = base.match(/[-.]((IQ\d+|Q\d+)(?:_[A-Z0-9]+)*|BF16|F16|F32)$/i);
    return m ? m[1].toUpperCase() : null;
}

// ── Launch filters state (for filter bar + grouping) ──────────────────────────

const launchFilters = {
    family: null,
    size: null,
    tags: [],
    collection: null,
    groupByFamily: false,
    sortBy: 'last_launched',
};
// ── Memory bar (segmented, platform-aware) ─────────────────────────────────────
// Unified (macOS): single pool, Metal cap, reclaimable cache.
// Discrete GPU (Win/Linux): VRAM + system RAM as overflow.

const _MEM_SAFETY_MARGIN = 512 * 1024 * 1024; // 512 MB safety margin for allocations

// Cached memory state shared between the memory bar and card VRAM estimates.
// readAt is the timestamp of the live /api/memory-availability read the verdict
// was computed against (architecture invariant 19) — surfaced on each card's
// VRAM element so a stale card is visibly stale.
let _memState = {
    availBytes: 0,
    budgetIfPurgedBytes: 0,
    metalCapBytes: 0,
    availRamBytes: 0,
    isUnified: false,
    reclaimableBytes: 0,
    readAt: null,
};

// Architecture invariant 19: the availability read is fetched once per render
// of the memory bar, before the unified/discrete dispatch, so BOTH branches
// (unified unified-memory and discrete-VRAM) consume the same live snapshot
// rather than each deriving availability from their own metrics.
async function _fetchMemoryAvailability() {
    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const resp = await fetch('/api/memory-availability', { headers });
        if (!resp.ok) return null;
        const data = await resp.json();
        const snapshot = (data && data.ok && data.snapshot) ? data.snapshot : null;
        return snapshot
            ? { bytes: snapshot.current_safe_availability_bytes || 0, readAt: new Date().toISOString() }
            : null;
    } catch {
        return null;
    }
}

// Architecture invariant 16: the render kill-switch reaches the UI only as a
// closed enum resolved server-side. 'bundled' renders the compact launch card;
// 'legacy' (the pre-bundle fallback) renders through the flat preset shape. The
// value is fetched from /api/preset-cards, validated against the closed enum,
// and applied once. Unknown values fail closed to 'bundled'.
let _presetBundleUi = 'bundled';
let _presetBundleUiLoaded = false;
let _presetBundleUiPromise = null;

async function _refreshPresetBundleUi() {
    if (_presetBundleUiLoaded) return _presetBundleUi;
    if (!_presetBundleUiPromise) {
        _presetBundleUiPromise = (async () => {
            try {
                const headers = window.authHeaders ? window.authHeaders() : {};
                const resp = await fetch('/api/preset-cards', { headers });
                if (resp.ok) {
                    const data = await resp.json();
                    const mode = data && data.preset_bundle_ui;
                    if (mode === 'bundled' || mode === 'legacy') _presetBundleUi = mode;
                }
            } catch {
                // fail closed to the default 'bundled'
            }
            _presetBundleUiLoaded = true;
            // If presets already rendered the grid under the default 'bundled'
            // assumption, refresh once the authoritative mode is known so a legacy
            // deployment never lingers on the compact bundle card (or vice versa).
            if (document.getElementById('setup-launch-grid')) renderLaunchGrid();
            return _presetBundleUi;
        })();
    }
    return _presetBundleUiPromise;
}

function _osReserveForUnified(ramTotalBytes) {
    // OS reserve: how much of total RAM must never be treated as usable for inference.
    // On unified memory, the GPU and OS share one pool; if Metal takes too much,
    // the system starves. We keep a realistic reserve so "available" isn't a lie.
    const gb = ramTotalBytes / (1024 ** 3);
    if (gb >= 96) return 8 * 1024 ** 3;
    if (gb >= 48) return 6 * 1024 ** 3;
    if (gb >= 32) return 5 * 1024 ** 3;
    return 4 * 1024 ** 3; // default for smaller machines
}

function _metalCap(totalBytes, metalGpuLimitMb) {
    if (metalGpuLimitMb > 0) return metalGpuLimitMb * 1024 * 1024;
    const fraction = totalBytes <= 36 * 1024 ** 3 ? 2 / 3 : 3 / 4;
    return Math.floor(totalBytes * fraction);
}

function _fmtGb(bytes) {
    const gb = bytes / 1024 ** 3;
    return gb >= 10 ? gb.toFixed(0) : gb.toFixed(1);
}

function _setBarSegment(el, pct) {
    if (!el) return;
    el.style.width = pct + '%';
}

function _renderUnifiedBar(segs, labels, purgeBtn, metalCapBytes, freeNow, reclaimable, availLabel, totalLabel) {
    // Segments now: GPU limit | In use | Available | Freeable
    // GPU limit is the non-GPU-usable portion (total - Metal cap).
    const gpuLimitEl = document.getElementById('setup-mem-bar-seg-gpulimit');
    const gpuLimitLabelEl = document.getElementById('setup-mem-bar-seg-gpulimit-label');
    const sep1 = document.getElementById('setup-mem-bar-sep1');

    if (segs.gpuLimitGb > 0) {
        _setBarSegment(gpuLimitEl, segs.gpuLimitPct);
        const fmt = (v) => (v >= 10 ? Math.round(v) : Math.round(v * 10) / 10);
        gpuLimitLabelEl.textContent = fmt(segs.gpuLimitGb) + ' GB GPU limit';
        if (sep1) sep1.style.display = '';
    } else {
        _setBarSegment(gpuLimitEl, 0);
        gpuLimitLabelEl.textContent = '';
        if (sep1) sep1.style.display = 'none';
    }

    _setBarSegment(document.getElementById('setup-mem-bar-seg-inuse'), segs.inuse);
    _setBarSegment(document.getElementById('setup-mem-bar-seg-avail'), segs.avail);
    _setBarSegment(document.getElementById('setup-mem-bar-seg-freeable'), segs.freeable);

    // Inline labels in segments using actual GB
    const fmt = (v) => (v >= 10 ? Math.round(v) : Math.round(v * 10) / 10);
    document.getElementById('setup-mem-bar-seg-inuse-label').textContent = fmt(segs.inuseGb) + ' GB in use';
    document.getElementById('setup-mem-bar-seg-avail-label').textContent = fmt(segs.availGb) + ' GB available';
    document.getElementById('setup-mem-bar-seg-freeable-label').textContent = fmt(segs.freeableGb) + ' GB freeable';

    // Build left label (now fully prepared by caller with clear wording)
    if (labels) {
        const availEl = document.getElementById('setup-mem-bar-avail');
        availEl.textContent = availLabel;
    }

    // Show "Free cache" button if reclaimable is meaningful
    if (purgeBtn && reclaimable >= 3 * 1024 ** 3) {
        purgeBtn.style.display = 'inline-flex';
    } else if (purgeBtn) {
        purgeBtn.style.display = 'none';
    }
}

function _renderDiscreteBar(segs, labels, purgeBtn, availLabel, totalLabel) {
    // Discrete GPU bar: in_use + available (VRAM only)
    _setBarSegment(document.getElementById('setup-mem-bar-seg-inuse'), segs.inuse);
    _setBarSegment(document.getElementById('setup-mem-bar-seg-avail'), segs.avail);
    _setBarSegment(document.getElementById('setup-mem-bar-seg-freeable'), 0);

    // Inline labels in segments (e.g., "1.5 GB in use")
    const inuseGb = segs.inuseGb != null ? (segs.inuseGb >= 10 ? Math.round(segs.inuseGb) : segs.inuseGb.toFixed(1)) : 0;
    const availGb = segs.availGb != null ? (segs.availGb >= 10 ? Math.round(segs.availGb) : segs.availGb.toFixed(1)) : 0;

    document.getElementById('setup-mem-bar-seg-inuse-label').textContent = inuseGb + ' GB in use';
    document.getElementById('setup-mem-bar-seg-avail-label').textContent = availGb + ' GB available';
    document.getElementById('setup-mem-bar-seg-freeable-label').textContent = '';

    if (labels) {
        const availEl = document.getElementById('setup-mem-bar-avail');
        availEl.textContent = availLabel;
    }

    if (purgeBtn) purgeBtn.style.display = 'none';
}

export async function fetchAndRenderMemoryBar() {
    const bar = document.getElementById('setup-mem-bar');
    if (!bar) return;

    const purgeBtn = document.getElementById('setup-mem-bar-purge');

    try {
        const headers = window.authHeaders ? window.authHeaders() : {};
        const [sysResp, gpuResp, platformInfo, limResp] = await Promise.all([
            fetch('/metrics/system', { headers }),
            fetch('/metrics/gpu', { headers }),
            getPlatformInfo().catch(() => null),
            fetch('/api/system/metal-gpu-limit', { headers }),
        ]);

        let ramTotalBytes = 0;
        let vramTotalBytes = 0;
        let ramUsedBytes = 0;
        let metalGpuLimitMb = 0;
        let isUnified = false;
        let vramUsedBytes = 0;
        let reclaimableBytes = 0;

        let sysData = null;
        if (sysResp.ok) {
            sysData = await sysResp.json();
            ramTotalBytes = (sysData.ram_total_gb || 0) * 1024 ** 3;
            ramUsedBytes = (sysData.ram_used_gb || 0) * 1024 ** 3;
            reclaimableBytes = (sysData.memory_reclaimable_gb || 0) * 1024 ** 3;
        }

        if (gpuResp.ok) {
            const data = await gpuResp.json();
            const gpus = Array.isArray(data) ? data : (data.gpus ? data.gpus : Object.values(data));
            for (const g of gpus) {
                const tMb = g.vram_total_mb || g.total_mb || g.total_memory_mb || g.vram_total || 0;
                const uMb = g.vram_used_mb || g.used_mb || g.vram_used || 0;
                if (g.metal_gpu_limit_mb !== undefined) {
                    isUnified = true;
                    metalGpuLimitMb = g.metal_gpu_limit_mb || 0;
                    if (!ramTotalBytes) ramTotalBytes = tMb * 1024 * 1024;
                } else {
                    vramTotalBytes += tMb * 1024 * 1024;
                    vramUsedBytes += uMb * 1024 * 1024;
                }
            }
        }

        // Fallback: if GPU metrics haven't populated yet (mactop race on startup),
        // detect unified memory from platform-info (independent of mactop).
        if (!isUnified && platformInfo) {
            if (platformInfo.auto_backend === 'metal') {
                isUnified = true;
                if (!metalGpuLimitMb && limResp.ok) {
                    const lim = await limResp.json();
                    metalGpuLimitMb = lim.limit_mb || 0;
                }
            }
        }

        // Fetch the live availability snapshot once, before dispatching, so the
        // unified and discrete branches share the same read (invariant 19). The
        // read timestamp is stamped on _memState NOW — before the card VRAM
        // estimates render — so every card's fit verdict carries the time it was
        // computed against.
        const availability = await _fetchMemoryAvailability();
        _memState.readAt = availability ? availability.readAt : null;

        if (isUnified) {
            await _renderUnifiedMemoryBar(bar, purgeBtn, metalGpuLimitMb, ramTotalBytes, ramUsedBytes, reclaimableBytes, sysData, availability);
        } else if (vramTotalBytes > 0) {
            await _renderDiscreteMemoryBar(bar, purgeBtn, vramTotalBytes, vramUsedBytes, ramTotalBytes, ramUsedBytes, availability);
        } else {
            // no usable metrics
        }
    } catch {
        // leave bar hidden if metrics unavailable
    }
}

// Unified (macOS) path: single pool, Metal cap, reclaimable cache.
async function _renderUnifiedMemoryBar(bar, purgeBtn, metalGpuLimitMb, ramTotalBytes, ramUsedBytes, reclaimableBytes, sysData, availability) {
    if (!ramTotalBytes) return;

    const cap = _metalCap(ramTotalBytes, metalGpuLimitMb);
    const osReserve = _osReserveForUnified(ramTotalBytes);

    // GPU limit is the non-GPU-usable portion: total - Metal cap.
    const gpuLimitBytes = (cap < ramTotalBytes) ? (ramTotalBytes - cap) : 0;
    const gpuLimitPct = (gpuLimitBytes / ramTotalBytes) * 100;
    const gpuLimitGb = gpuLimitBytes / (1024 ** 3);

    // Derive segments: in_use (actual + wired), available_now, freeable_cache.
    // macOS "ram_used_gb" = total - available; includes cache that is reclaimable.
    const nonReclaimUsed = Math.max(0, ramUsedBytes - reclaimableBytes);
    let inUseBytes = Math.max(nonReclaimUsed, (sysData && sysData.memory_wired_gb > 0)
        ? (sysData.memory_wired_gb * 1024 ** 3)
        : nonReclaimUsed);

    // Constrain inUse + reclaimable within the GPU-usable cap so the bar
    // partitions 100%: GPU limit + inUse + available + freeable = total.
    if (inUseBytes > cap) inUseBytes = cap;
    let reclaimableWithinCap = Math.min(reclaimableBytes, cap - inUseBytes);
    const freeNow = cap - inUseBytes - reclaimableWithinCap;

    // "Available now" = what we can realistically allocate for inference without purging:
    //   min(metal_cap, freeNow + partial_reclaimable) - reserve
    // We don't want to assume all reclaimable will be freed unless user purges, so
    // take a fraction (60%) as "likely reclaimable under pressure".
    const likelyReclaimable = reclaimableWithinCap * 0.6;
    const safeLimit = Math.min(cap, ramTotalBytes - osReserve);
    let availNow = Math.max(0, Math.min(safeLimit, freeNow + likelyReclaimable - _MEM_SAFETY_MARGIN));

    // Phase 5b Part C: prefer the backend's MemoryAvailabilitySnapshot for
    // current_safe_availability_bytes — the single source of truth. The read is
    // shared with the discrete branch and the card timestamps (invariant 19).
    if (availability && availability.bytes > 0) {
        availNow = availability.bytes;
    }

    // If user purges: all reclaimable becomes free; new "available" =:
    const totalAfterPurge = cap - inUseBytes; // only GPU-usable matters
    const availIfPurged = Math.max(0, Math.min(safeLimit, totalAfterPurge - _MEM_SAFETY_MARGIN));

    // Percentages of total RAM for bar segments.
    const inusePct = (inUseBytes / ramTotalBytes) * 100;
    const availPct = (Math.max(0, freeNow) / ramTotalBytes) * 100;
    const freeablePct = (reclaimableWithinCap / ramTotalBytes) * 100;

    // Labels: clear and aligned with card budgets
    const availGb = availNow > 0 ? Math.round(availNow / (1024 ** 3)) : 0;
    const ifPurgedGb = availIfPurged > 0 ? Math.round(availIfPurged / (1024 ** 3)) : 0;

    let availLabel;
    if (availGb > 0) {
        if (ifPurgedGb > availGb && reclaimableBytes >= 3 * 1024 ** 3) {
            availLabel = availGb + ' GB available now · Up to ' + ifPurgedGb + ' GB after freeing cache';
        } else {
            availLabel = availGb + ' GB available now';
        }
    } else {
        availLabel = 'Very little memory available';
    }

    const totalLabel = _fmtGb(ramTotalBytes) + ' GB unified';

    _renderUnifiedBar(
        {
            gpuLimitPct,
            gpuLimitGb,
            inuse: inusePct,
            avail: availPct,
            freeable: freeablePct,
            inuseGb: inUseBytes / (1024 ** 3),
            availGb: Math.max(0, freeNow) / (1024 ** 3),
            freeableGb: reclaimableBytes / (1024 ** 3),
        },
        true,
        purgeBtn,
        cap,
        freeNow,
        reclaimableBytes,
        availLabel,
        totalLabel,
    );

    if (document.getElementById('setup-mem-bar-total')) {
        document.getElementById('setup-mem-bar-total').textContent = totalLabel;
    }
    bar.style.display = '';

    // Metal limit teaching: if current cap is below recommended, show increase button next to Metal GPU cap text
    const recommendedMb = Math.round(ramTotalBytes / (1024 * 1024)) - 8192; // total_MB - 8GB
    if (metalGpuLimitMb > 0 && metalGpuLimitMb < recommendedMb) {
        const currentGb = Math.round(metalGpuLimitMb / 1024);
        const recGb = Math.round(recommendedMb / 1024);
        const totalGb = Math.round(ramTotalBytes / (1024 ** 3));
        // Create inline row below VRAM bar for welcome screen
        const existingRow = document.getElementById('setup-metal-limit-row');
        if (!existingRow) {
            const row = document.createElement('div');
            row.id = 'setup-metal-limit-row';
            row.className = 'metal-limit-row';
            row.style.cssText = 'margin-top:8px;display:flex;align-items:center;gap:6px;font-size:13px;color:#cbd5e1;width:fit-content;';
            const labelSpan = document.createElement('span');
            labelSpan.className = 'metal-limit-text';
            labelSpan.style.color = '#cbd5e1';
            labelSpan.textContent = 'Metal GPU cap: ' + currentGb + ' GB (of ' + totalGb + ' GB total)';
            row.appendChild(labelSpan);
            const btn = document.createElement('button');
            btn.id = 'setup-metal-limit-btn';
            btn.type = 'button';
            btn.className = 'btn-metal-increase';
            btn.style.cssText = 'background:none;border:1px solid #475569;color:#cbd5e1;padding:3px 10px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:500;transition:all 0.15s ease;';
            btn.textContent = 'Increase to ' + recGb + ' GB';
            row.appendChild(btn);
            bar.parentNode.insertBefore(row, bar.nextSibling);
        }
        const btn = document.getElementById('setup-metal-limit-btn');
        if (btn && !btn.dataset.wired) {
            btn.dataset.wired = '1';
            const row = btn.parentElement;
            btn.addEventListener('click', async () => {
                btn.disabled = true;
                btn.textContent = 'Updating…';
                try {
                    // Fetch db-admin-token (required for system-level changes)
                    const tokenHeaders = window.authHeaders ? window.authHeaders() : {};
                    const tokenResp = await fetch('/api/db/admin-token', { headers: tokenHeaders });
                    let adminToken = '';
                    if (tokenResp.ok) {
                        const data = await tokenResp.json().catch(() => ({}));
                        adminToken = data.token || '';
                    }
                    const headers = {
                        'Content-Type': 'application/json',
                        ...(adminToken ? { 'Authorization': `Bearer ${adminToken}` } : {}),
                    };
                    const resp = await fetch('/api/system/set-metal-gpu-limit', {
                        method: 'POST',
                        headers,
                        body: JSON.stringify({ limit_mb: recommendedMb, confirm: 'set-metal-gpu-limit' }),
                    });
                    if (!resp.ok) {
                        const data = await resp.json().catch(() => ({}));
                        showToast('Authentication required: ' + (data.error?.reason || 'db-admin-token needed'), 'error');
                        btn.disabled = false;
                        btn.textContent = 'Increase to ' + recGb + ' GB';
                        return;
                    }
                    const data = await resp.json();
                    if (data.ok) {
                        showToast('Metal GPU cap updated to ' + Math.round(data.limit_mb / 1024) + ' GB — persists across reboots', 'success');
                        row?.remove();
                    } else {
                        showToast('Failed: ' + (data.error || 'unknown'), 'error');
                        btn.disabled = false;
                        btn.textContent = 'Increase to ' + recGb + ' GB';
                    }
                } catch {
                    showToast('Error updating Metal GPU cap', 'error');
                    btn.disabled = false;
                    btn.textContent = 'Increase to ' + recGb + ' GB';
                }
            });
        }
    }

    // Wire "Free cache" button (macOS only)
    if (purgeBtn && reclaimableBytes >= 3 * 1024 ** 3) {
        purgeBtn.onclick = async () => {
            const confirmed = await confirmFreeCacheCleanup(reclaimableBytes);
            if (!confirmed) return;

            const originalLabel = purgeBtn.textContent;
            purgeBtn.disabled = true;
            purgeBtn.setAttribute('aria-busy', 'true');
            purgeBtn.textContent = 'Freeing…';
            try {
                const token = await _fetchDbAdminTokenForSystemAction();
                if (!token) {
                    showToast(
                        'Could not authorize cache cleanup',
                        'error',
                        'Open Configuration and confirm the administrator token is available.'
                    );
                    return;
                }
                const resp = await fetch('/system/purge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + token,
                    },
                    body: JSON.stringify({ confirm: 'purge-memory' }),
                });
                const out = await resp.json().catch(() => null);
                if (!resp.ok || out?.error) {
                    showToast('Cache cleanup failed', 'error', out?.error || 'macOS could not free the cache.');
                } else {
                    showToast(
                        'Cache cleanup complete',
                        'success',
                        'Memory availability is being refreshed.'
                    );
                    // Re-render bar + cards to reflect new free memory.
                    setTimeout(fetchAndRenderMemoryBar, 600);
                }
            } catch {
                showToast('Cache cleanup failed', 'error', 'The system action could not be completed.');
            } finally {
                purgeBtn.disabled = false;
                purgeBtn.removeAttribute('aria-busy');
                purgeBtn.textContent = originalLabel;
            }
        };
    }

    _memState = {
        availBytes: availNow,
        budgetIfPurgedBytes: availIfPurged,
        metalCapBytes: cap,
        availRamBytes: 0,
        isUnified: true,
        reclaimableBytes,
        readAt: availability ? availability.readAt : null,
    };

    // Card VRAM estimates use:
    //   - budgetNow (availNow)
    //   - budgetIfPurged (for "Free cache to run this" state)
    _fetchCardVramEstimates(availNow, 0, true, availIfPurged);
}

// Helper used by the "Free cache" button to call /system/purge.
async function _fetchDbAdminTokenForSystemAction() {
    try {
        const tokenResp = await fetch('/api/db/admin-token', {
            headers: window.authHeaders ? window.authHeaders() : {},
        });
        if (!tokenResp.ok) return null;
        const tokenData = await tokenResp.json().catch(() => ({}));
        return tokenData.token || null;
    } catch {
        return null;
    }
}

// Discrete GPU path (Win/Linux): VRAM bar + system RAM as overflow.
async function _renderDiscreteMemoryBar(bar, purgeBtn, vramTotalBytes, vramUsedBytes, ramTotalBytes, ramUsedBytes, availability) {
    const vramFree = Math.max(0, vramTotalBytes - vramUsedBytes);
    const ramFreeBytes = Math.max(0, ramTotalBytes - ramUsedBytes);

    const inusePct = vramTotalBytes > 0 ? ((vramUsedBytes / vramTotalBytes) * 100) : 0;
    const availPct = 100 - inusePct;

    const availLabel = _fmtGb(vramFree) + ' GB available';
    const totalLabel =
        _fmtGb(vramTotalBytes) + ' GB VRAM · ' +
        _fmtGb(ramTotalBytes) + ' GB system RAM';

    _renderDiscreteBar(
        {
            inuse: inusePct,
            avail: availPct,
            inuseGb: vramUsedBytes / (1024 ** 3),
            availGb: vramFree / (1024 ** 3),
        },
        true,
        purgeBtn,
        availLabel,
        totalLabel,
    );

    if (document.getElementById('setup-mem-bar-total')) {
        document.getElementById('setup-mem-bar-total').textContent = totalLabel;
    }
    bar.style.display = '';

    _memState = {
        availBytes: vramFree,
        budgetIfPurgedBytes: vramFree,
        metalCapBytes: 0,
        availRamBytes: ramFreeBytes,
        isUnified: false,
        reclaimableBytes: 0,
        readAt: availability ? availability.readAt : null,
    };

    _fetchCardVramEstimates(vramFree, ramFreeBytes, false, vramFree);
}

// ── Drop zone ─────────────────────────────────────────────────────────────────

export function initSetupDropZone() {
    const zone = document.getElementById('setup-drop-zone');
    if (!zone) return;

    zone.addEventListener('dragover', e => {
        const items = Array.from(e.dataTransfer?.items || []);
        const hasGguf = items.some(i => i.kind === 'file');
        if (!hasGguf) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        zone.classList.add('is-over');
    });

    zone.addEventListener('dragleave', e => {
        if (!zone.contains(e.relatedTarget)) zone.classList.remove('is-over');
    });

    zone.addEventListener('drop', async e => {
        e.preventDefault();
        zone.classList.remove('is-over');
        const file = e.dataTransfer?.files?.[0];
        if (!file || !file.name.toLowerCase().endsWith('.gguf')) return;

        // Try to resolve the full filesystem path from the known models library
        try {
            const headers = window.authHeaders ? window.authHeaders() : {};
            const resp = await fetch('/api/models', { headers });
            if (resp.ok) {
                const models = await resp.json();
                const match = models.find(m =>
                    (m.path || '').split(/[/\\]/).pop() === file.name
                );
                if (match) {
                    window.__spawnWizardOpts = { localPath: match.path, localModel: match };
                    Router.navigate('/spawn');
                    return;
                }
            }
        } catch {}

        // File not in library — open wizard on the model step so user can browse
        Router.navigate('/spawn');
    });

    // Also open wizard on click (as an explicit affordance)
    zone.addEventListener('click', () => {
        Router.navigate('/spawn');
    });
}

function setAttachButtonLabel(button, label) {
    if (!button) return;
    const icon = document.createElement('span');
    icon.className = 'btn-icon';
    icon.textContent = '⚡';
    button.replaceChildren(icon, document.createTextNode(` ${label}`));
}

// ── View Switching ────────────────────────────────────────────────────────────

export function switchView(targetView) {
    if (setupViewState.view === 'transitioning') return;

    const previousView = setupViewState.view;
    setupViewState.view = 'transitioning';

    if (targetView === 'setup' && previousView === 'monitor') {
        savePreviousPosition();
    }

    const currentViewEl = document.getElementById('view-' + previousView);
    const targetViewEl = document.getElementById('view-' + targetView);
    const setupStrip = document.getElementById('endpoint-strip-setup');
    const monitorStrip = document.getElementById('endpoint-strip-monitor');

    if (!currentViewEl || !targetViewEl) {
        setupViewState.view = targetView;
        return;
    }

    if (targetView === 'monitor') {
        currentViewEl.classList.add('exiting');
        setTimeout(() => {
            currentViewEl.style.display = 'none';
            currentViewEl.classList.remove('exiting');
            targetViewEl.style.display = '';
            targetViewEl.classList.add('entering');
            showFlashOverlay();
            animateCardsEnter();
            if (setupStrip) setupStrip.style.display = 'none';
            if (monitorStrip) monitorStrip.style.display = '';
            document.body.classList.remove('setup-active');
            setTimeout(() => {
                targetViewEl.classList.remove('entering');
                setupViewState.view = 'monitor';
            }, 500);
        }, 400);
    } else {
        animateCardsExit();
        if (setupStrip) setupStrip.style.display = '';
        if (monitorStrip) monitorStrip.style.display = 'none';
        document.body.classList.add('setup-active');
        setTimeout(() => {
            currentViewEl.style.display = 'none';
            currentViewEl.classList.remove('exiting');
            targetViewEl.style.display = '';
            targetViewEl.classList.add('entering');
            animateSetupCardsEnter();
            setTimeout(() => {
                targetViewEl.classList.remove('entering');
                setupViewState.view = 'setup';
            }, 400);
        }, 600);
    }
}

// Idempotent view guards used by the router so route handlers can ensure the
// correct top-level view without triggering a redundant (and visually jarring)
// transition when already there. `switchView` itself no-ops while transitioning.
export function ensureMonitorView() {
    if (setupViewState.view === 'setup') switchView('monitor');
}

export function ensureSetupView() {
    if (setupViewState.view === 'monitor') switchView('setup');
}

// ── Connecting State ──────────────────────────────────────────────────────────

export function showConnectingState() {
    const connectingDots = document.getElementById('connecting-dots');
    if (connectingDots) connectingDots.style.display = '';
}

export function hideConnectingState() {
    const connectingDots = document.getElementById('connecting-dots');
    if (connectingDots) connectingDots.style.display = 'none';
}

// ── Animations ────────────────────────────────────────────────────────────────

function showFlashOverlay() {
    const existing = document.querySelector('.view-flash');
    if (existing) existing.remove();
    const flash = document.createElement('div');
    flash.className = 'view-flash';
    document.body.appendChild(flash);
    setTimeout(() => flash.remove(), 800);
}

function animateCardsEnter() {
    const cards = document.querySelectorAll('.view-monitor .widget-card');
    cards.forEach((card, i) => {
        card.classList.add('entrance');
        setTimeout(() => card.classList.add('active'), 120 * i);
    });
}

function animateCardsExit() {
    const cards = [...document.querySelectorAll('.view-monitor .widget-card')].reverse();
    cards.forEach((card, i) => {
        card.style.transition = `opacity 0.3s ease ${60 * i}ms, transform 0.3s ease ${60 * i}ms`;
        card.style.opacity = '0';
        card.style.transform = 'translateY(16px)';
    });
}

function animateSetupCardsEnter() {
    // no-op: launch cards use CSS animation-delay via inline style
}

// Map of preset_id → last_connected_at (ms), populated by renderRecentEndpoints
const _spawnLastLaunched = new Map();

function _applyLastLaunchedToCards() {
    document.querySelectorAll('.launch-card[data-preset-id]').forEach(card => {
        const presetId = card.dataset.presetId;
        const ts = _spawnLastLaunched.get(presetId);
        let el = card.querySelector('.launch-card-last-launched');
        if (!ts) { if (el) el.remove(); return; }
        if (!el) {
            el = document.createElement('div');
            el.className = 'launch-card-last-launched';
            const modelEl = card.querySelector('.launch-card-model');
            if (modelEl) modelEl.after(el);
            else card.querySelector('.launch-card-chips')?.before(el);
        }
        el.textContent = 'Last launched · ' + formatRelativeTime(ts);
    });
}



// ── Launch Grid — Preset Cards ────────────────────────────────────────────────

function _visiblePresetsLocal(presets) {
    const user = presets.filter(p => !p.id.startsWith('default-'));
    return user.length > 0 ? user : presets;
}

export function renderLaunchGrid() {
    const grid = document.getElementById('setup-launch-grid');
    if (!grid) return;
    grid.innerHTML = '';

    const allPresets = sessionState.presets || [];
    const userPresets = allPresets.filter(p => !p.id.startsWith('default-'));
    const hasUserPresets = userPresets.length > 0;
    const filterBar = document.getElementById('setup-filter-bar');
    if (filterBar && filterBar.dataset.initialized !== '1') initLaunchFilters();
    let presets = _visiblePresetsLocal(allPresets);
    const activePresetId = sessionState.activeSessionPresetId || '';
    const showNewConfigCard = presets.length <= 2;

    // Apply filters and the user-selected ordering. Sorting is intentionally
    // done after filtering so the visible list remains stable while browsing.
    presets = _filterPresets(presets);
    presets = _sortPresets(presets);

    // Optional grouped view
    if (launchFilters.groupByFamily && presets.length > 4) {
        _renderGroupedLaunchGrid(grid, presets, activePresetId, hasUserPresets, showNewConfigCard);
    } else {
        _renderFlatLaunchGrid(grid, presets, activePresetId, hasUserPresets, showNewConfigCard);
    }

    // Stamp last-launched timestamps if session data is already available
    _applyLastLaunchedToCards();
    // Memory bar and drop zone: init once (idempotent after first call)
    if (!document.getElementById('setup-drop-zone')?._dzInited) {
        initSetupDropZone();
        const dz = document.getElementById('setup-drop-zone');
        if (dz) dz._dzInited = true;
    }
    fetchAndRenderMemoryBar();
}

function _sortPresets(presets) {
    const sorted = [...presets];
    const name = p => String(p.name || p.model_path || p.hf_repo || '').toLocaleLowerCase();
    const size = p => Number(
        p.model_size_bytes || p.file_size_bytes || p.size_bytes || p.model_size || 0,
    );
    const lastLaunched = p => _spawnLastLaunched.get(p.id) || Number(p.last_launched_at || 0);
    sorted.sort((a, b) => {
        let delta;
        switch (launchFilters.sortBy) {
            case 'name':
                delta = name(a).localeCompare(name(b));
                break;
            case 'size':
                delta = size(b) - size(a);
                break;
            case 'backend':
                delta = String(a.backend || '').localeCompare(String(b.backend || '')) || name(a).localeCompare(name(b));
                break;
            case 'last_launched':
            default:
                delta = lastLaunched(b) - lastLaunched(a);
                break;
        }
        return delta || name(a).localeCompare(name(b));
    });
    return sorted;
}

function _filterPresets(presets) {
    const f = launchFilters;
    if (!f.family && !f.size && !f.collection && (f.tags || []).length === 0) return presets;

    return presets.filter(p => {
        const c = classifyPreset(p);
        if (f.family && c.family !== f.family) return false;
        if (f.size && c.sizeClass !== f.size) return false;

        if (f.collection) {
            const col = (sessionState.collections || []).find(co => co.id === f.collection);
            if (!col || !col.preset_ids.includes(p.id)) return false;
        }

        if ((f.tags || []).length > 0) {
            const presetTags = (p.tags || []);
            if (!f.tags.some(t => presetTags.includes(t))) return false;
        }

        return true;
    });
}

function _renderFlatLaunchGrid(grid, presets, activePresetId, hasUserPresets, showNewConfigCard) {
    if (!hasUserPresets) {
        const hint = document.createElement('div');
        hint.className = 'launch-grid-hint';
        hint.style.gridColumn = '1 / -1';
        hint.textContent = 'New here? Open the setup wizard to pick a model and start your first local AI in a few steps.';
        grid.appendChild(hint);

        if (showNewConfigCard) {
            const newCard = _buildNewConfigCard(true);
            newCard.style.animationDelay = '80ms';
            grid.appendChild(newCard);
        }
        presets.forEach((preset, i) => {
            const card = _buildLaunchCard(preset, activePresetId);
            card.style.animationDelay = `${(i + (showNewConfigCard ? 2 : 1)) * 55}ms`;
            grid.appendChild(card);
        });
    } else {
        presets.forEach((preset, i) => {
            const card = _buildLaunchCard(preset, activePresetId);
            card.style.animationDelay = `${i * 55}ms`;
            grid.appendChild(card);
        });
        if (showNewConfigCard) {
            const newCard = _buildNewConfigCard(false);
            newCard.style.animationDelay = `${presets.length * 55}ms`;
            grid.appendChild(newCard);
        }
    }
}

function _renderGroupedLaunchGrid(grid, presets, activePresetId, hasUserPresets, showNewConfigCard) {
    const byFamily = {};
    presets.forEach(p => {
        const key = p.family || 'other';
        if (!byFamily[key]) byFamily[key] = [];
        byFamily[key].push(p);
    });

    let i = 0;
    const sections = Object.keys(byFamily);
    for (const fam of sections) {
        const header = document.createElement('div');
        header.className = 'launch-grid-group';
        header.style.gridColumn = '1 / -1';
        const label = fam === 'other' ? 'Other' : (FAMILY_LABEL_MAP[fam] || fam);
        const list = byFamily[fam];
        header.textContent = `${label} (${list.length})`;
        grid.appendChild(header);

        list.forEach(p => {
            const card = _buildLaunchCard(p, activePresetId);
            card.style.animationDelay = `${i * 55}ms`;
            grid.appendChild(card);
            i++;
        });
    }

    if (showNewConfigCard) {
        const newCard = _buildNewConfigCard(false);
        newCard.style.animationDelay = `${i * 55}ms`;
        grid.appendChild(newCard);
    }
}

// Phase 8a: project a v6 preset bundle onto the compact launch card. The card
// shows the bundle identity (tune display name), the saved default selection's
// chips, and the selected artifact's quantization — never the resolved-config
// hash (that is drawer-only). `degraded` is true when the selected artifact has
// no local file, which downgrades the Start button to the setup-wizard path.
function _bundleCardView(preset) {
    const bundle = preset.bundle;
    if (!bundle || !bundle.artifacts || !bundle.artifacts.length) return null;
    const sel = bundle.default_selection || {};
    const selected =
        bundle.artifacts.find(a => a.id === sel.artifact_id) ||
        bundle.artifacts.find(a => a.role === 'weights') ||
        bundle.artifacts[0];
    const perf = (bundle.performance_options || []).find(p => p.id === sel.performance_id) || null;
    // The KV policy is a wire id of the form `CTK_CTV` (e.g. `q8_0_q8_0`); each
    // dtype can itself contain an underscore, so it is not safe to split on '_'.
    // The known policies mirror server LlamaKvPolicyId; the card chip shows the
    // readable `CTK/CTV` form. Unknown policies fall back to the flat preset KV.
    const kvPolicyDisplay = {
        f16_f16: 'f16/f16',
        q8_0_q8_0: 'q8_0/q8_0',
        q4_0_q4_0: 'q4_0/q4_0',
        q8_0_q4_0: 'q8_0/q4_0',
    };
    const kvDisplay = kvPolicyDisplay[sel.kv_policy] || (sel.kv_policy || (preset.ctk || 'q8_0') + '/' + (preset.ctv || 'f16'));
    const ctxDisplay = preset.context_size
        ? (Math.round(preset.context_size / 1024) >= 1000
            ? `${(Math.round(preset.context_size / 1024) / 1024).toFixed(1)}M context`
            : `${Math.round(preset.context_size / 1024)}k context`)
        : '128k context';
    const quantDisplay = (selected.quantization && selected.quantization.value) ? selected.quantization.value.toUpperCase() : '';
    return {
        identity: bundle.identity || null,
        bundleId: (bundle.identity && bundle.identity.bundle_id) || '',
        tuneLabel: perf ? perf.label : null,
        ctxDisplay,
        kvDisplay,
        quantDisplay,
        selected,
        degraded: !(selected && selected.local_path),
    };
}

function _buildLaunchCard(preset, activePresetId) {
    const isExample = preset.id.startsWith('default-');
    const card = document.createElement('div');
    card.className = 'launch-card';
    card.dataset.presetId = preset.id;
    if (isExample) card.classList.add('launch-card--example');

    // Only show running if the server is actually live and this preset is the active one
    const isRunning = !isExample && sessionState.serverRunning && preset.id === activePresetId && activePresetId;
    if (isRunning) card.classList.add('launch-card--running');

    const rapidMlx = preset.rapid_mlx;
    // Phase 8a: a v6 bundle renders one compact card (tune name + saved selection
    // chips) instead of the flat one-artifact shape. The kill-switch (_presetBundleUi)
    // collapses it back through the flat path. Example cards never take the bundle
    // branch.
    const bundleView = (isExample || _presetBundleUi === 'legacy') ? null : _bundleCardView(preset);

    const modelSource = preset.backend === 'rapid_mlx'
        ? (rapidMlx?.model_source_view?.canonical_identity
            || rapidMlx?.model_source_view?.display_name || '')
        : (preset.model_path || preset.hf_repo || '');
    const flatModelFile = (preset.model_path || '').split(/[/\\]/).pop() ||
                      (preset.backend === 'rapid_mlx' ? modelSource.split(/[/\\]/).pop() : '') ||
                      (preset.hf_repo ? preset.hf_repo.split('/').slice(-1)[0] : '');
    const modelFile = bundleView ? (bundleView.selected?.display_name || flatModelFile) : flatModelFile;
    const hasModel = !!modelFile && !(bundleView && bundleView.degraded);
    const backendLabel = preset.backend === 'rapid_mlx' ? 'Rapid-MLX' : 'llama-server';
    // A bundle card leads with the tune's display name, not the preset's stored name.
    const displayName = (bundleView && bundleView.identity?.display_name) || preset.name;

    const ctxK = preset.context_size ? Math.round(preset.context_size / 1024) : 128;
    let ctxDisplay = ctxK >= 1000 ? `${(ctxK / 1024).toFixed(1)}M context` : `${ctxK}k context`;
    const isRapidMlx = preset.backend === 'rapid_mlx';
    let ctkDisplay;
    if (isRapidMlx) {
        const rapidKv = (rapidMlx && rapidMlx.kv_cache_dtype) || 'int4';
        const reasoningOn = rapidMlx && (rapidMlx.reasoning_mode === true || rapidMlx.reasoning_mode === 'on');
        if (reasoningOn && rapidKv !== 'int8') {
            ctkDisplay = `${rapidKv.toUpperCase()} → INT8`;
        } else {
            ctkDisplay = rapidKv.toUpperCase();
        }
    } else {
        ctkDisplay = (preset.ctk || 'q8_0') + '/' + (preset.ctv || 'f16');
    }
    let quantTag = extractQuantFromFilename(modelFile);
    if (bundleView) {
        // Bundle chips summarize the saved default selection, not the bundle's
        // full option lists. The quantization comes from the selected artifact's
        // declared quant, not a filename guess.
        ctxDisplay = bundleView.ctxDisplay;
        ctkDisplay = bundleView.kvDisplay;
        quantTag = bundleView.quantDisplay;
    }
    const rapidCacheMib = preset.backend === 'rapid_mlx' && rapidMlx?.prefix_cache_enabled !== false
        ? (rapidMlx?.retained_cache_mib ?? 8192) : 0;
    const rapidCacheChip = rapidCacheMib > 0
        ? `<span class="launch-chip launch-chip--accent" title="Optional Rapid retained prefix-cache memory">${rapidCacheMib / 1024} GiB cache</span>` : '';

    if (isExample) {
        // Example card: dimmed, no edit button, use-wizard CTA only
        // eslint-disable-next-line no-unsanitized/property -- content sanitized via escapeHtml
        card.innerHTML = `
            <div class="launch-card-top">
                <div class="launch-card-name" title="${escapeHtml(preset.name)}">${escapeHtml(preset.name)}</div>
                <span class="launch-card-example-badge">Example</span>
            </div>
            <div class="launch-card-model launch-card-model--empty">Add a model to use these settings</div>
            <div class="launch-card-chips">
                <span class="launch-chip">${ctxDisplay}</span>
                <span class="launch-chip">${ctkDisplay}</span>
                ${preset.backend === 'rapid_mlx' ? '<span class="launch-chip launch-chip--accent">Rapid-MLX</span>' : ''}
            </div>
            <div class="launch-card-actions">
                <button class="launch-card-btn-start launch-card-btn-start--configure" type="button"
                    title="Open the setup wizard with this preset's settings pre-loaded as a starting point">
                    Use as template →
                </button>
            </div>
        `;
        card.querySelector('.launch-card-btn-start').addEventListener('click', () => {
            window.__spawnWizardOpts = { templatePreset: preset };
            Router.navigate('/spawn');
        });
    } else {
        const c = classifyPreset(preset);
        const familyLabel = c.family ? FAMILY_LABEL_MAP[c.family] || c.family.toUpperCase() : '';
        const sizeLabel =
            c.sizeClass && c.sizeClass !== 'unknown' && c.sizeClass !== 'huge'
                ? c.sizeClass.charAt(0).toUpperCase() + c.sizeClass.slice(1)
                : '';
        const presetTags = (preset.tags || []);
        const tagPills = presetTags.length > 0
            ? '<div class="launch-card-tags">' +
              presetTags.slice(0, 3).map(t => `<span class="launch-tag">${escapeHtml(t)}</span>`).join('') +
              (presetTags.length > 3 ? `<span class="launch-tag launch-tag--more">+${presetTags.length - 3}</span>` : '') +
              '</div>' : '';

        // Architecture label from preset metadata (backend-driven)
        const arch = buildArchitectureLabel(preset, c);
        const archHtml = arch
            ? `<div class="launch-card-arch" title="${escapeHtml(arch.tooltip)}">${escapeHtml(arch.display)}</div>`
            : '';

        // Phase 8a bundle affordances. data-bundle-id / data-revision (set on the
        // card root below) carry the revision handshake; .launch-card-tune shows
        // the saved performance label; .launch-card-btn-configure opens the bundle
        // drawer (phase 8b) in place of the flat Edit button.
        const tuneHtml = bundleView && bundleView.tuneLabel
            ? `<div class="launch-card-tune" title="Saved tune (performance) selection">${escapeHtml(bundleView.tuneLabel)}</div>`
            : '';
        const configButtonHtml = bundleView
            ? '<button class="launch-card-btn-configure" type="button">Configure</button>'
            : '';

        // eslint-disable-next-line no-unsanitized/property -- content sanitized via escapeHtml
        card.innerHTML = `
            <div class="launch-card-top">
                <div class="launch-card-name" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</div>
                ${isRunning ? '<span class="launch-card-running-badge">● Running</span>' : ''}
            </div>
            <div class="launch-card-meta">
                ${familyLabel ? `<span class="launch-meta-badge launch-meta-badge--family" title="Model family">${escapeHtml(familyLabel)}</span>` : ''}
                ${sizeLabel ? `<span class="launch-meta-badge launch-meta-badge--size" title="Size class">${escapeHtml(sizeLabel)}</span>` : ''}
            </div>
            <div class="launch-card-model ${hasModel ? '' : 'launch-card-model--empty'}" title="${escapeHtml(modelFile || '')}">${escapeHtml(modelFile || 'No model configured')}</div>
            ${archHtml}
            ${tuneHtml}
            <div class="launch-card-chips">
                <span class="launch-chip">${ctxDisplay}</span>
                <span class="launch-chip">${ctkDisplay}</span>
                ${quantTag ? `<span class="launch-chip launch-chip--quant" title="Quantization: ${escapeHtml(quantTag)}">${escapeHtml(quantTag)}</span>` : ''}
                ${rapidCacheChip}
                ${preset.ngram_spec ? '<span class="launch-chip launch-chip--accent">n-gram</span>' : ''}
            </div>
            ${tagPills}
            ${hasModel ? '<div class="launch-card-vram launch-card-vram--loading"><span class="launch-card-vram-total">…</span></div>' : ''}
            <div class="launch-card-actions${bundleView ? ' launch-card-actions--bundle' : ''}">
                ${configButtonHtml}
                <button class="launch-card-btn-edit" type="button">Edit</button>
                <button class="launch-card-btn-start ${hasModel ? '' : 'launch-card-btn-start--configure'}" type="button"
                    title="${hasModel ? `Start ${backendLabel} with this preset` : 'Open the setup wizard to set up a model for this preset'}">
                    ${hasModel ? '▶ Start' : 'Set up model →'}
                </button>
                <button class="launch-card-btn-trash" type="button" title="Delete preset">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 6h18"/>
                        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                        <line x1="10" y1="11" x2="10" y2="17"/>
                        <line x1="14" y1="11" x2="14" y2="17"/>
                    </svg>
                </button>
            </div>
        `;

        // Revision handshake lives on the card root: the resolve-and-launch
        // payload and the conflict re-render both key off these attributes.
        if (bundleView && bundleView.bundleId) {
            card.dataset.bundleId = bundleView.bundleId;
            card.dataset.revision = String(preset.revision ?? '');
        }

        card.querySelector('.launch-card-btn-edit').addEventListener('click', () => {
            import('./presets.js').then(({ openPresetModal, syncSelectedPresetSelection }) => {
                syncSelectedPresetSelection(preset.id, { userIntent: true, persist: true });
                openPresetModal('edit');
            });
        });

        // Phase 8a: bundle cards carry a dedicated Configure affordance that opens
        // the bundle drawer (phase 8b). Until the drawer module is present we fall
        // back to the existing bundle editor so the button is never a dead end.
        const configBtn = card.querySelector('.launch-card-btn-configure');
        if (configBtn) configBtn.addEventListener('click', () => {
            import('./presets.js').then(({ syncSelectedPresetSelection, openPresetModal }) => {
                syncSelectedPresetSelection(preset.id, { userIntent: true, persist: true });
                import('./preset-bundle-drawer.js')
                    .then(({ openBundleDrawer }) => openBundleDrawer(preset))
                    .catch(() => openPresetModal('edit'));
            });
        });

        // Quick-edit: click context or KV chip to open preset modal focused on context section.
        // Bundle chips summarize the saved selection and are edited through the drawer,
        // not the flat context quick-edit, so they don't inherit this handler.
        if (!bundleView) {
            card.querySelectorAll('.launch-chip:not(.launch-chip--quant)').forEach(chip => {
                chip.style.cursor = 'pointer';
                chip.title = 'Click to quickly edit context or KV cache settings';
                chip.addEventListener('click', () => {
                    import('./presets.js').then(({ openPresetModal, syncSelectedPresetSelection }) => {
                        syncSelectedPresetSelection(preset.id, { userIntent: true, persist: true });
                        openPresetModal('edit', 'context');
                    });
                });
            });
        }

        card.querySelector('.launch-card-btn-trash').addEventListener('click', async (e) => {
            e.stopPropagation();
            const ok = await showConfirmDialog(
                'Delete preset',
                `Delete preset "${escapeHtml(preset.name)}"? This cannot be undone.`,
                'Delete'
            );
            if (!ok) return;
            try {
                const headers = window.authHeaders ? window.authHeaders() : {};
                const resp = await fetch(`/api/presets/${preset.id}`, { method: 'DELETE', headers });
                if (resp.ok) {
                    await import('./presets.js').then(({ loadPresets }) => loadPresets());
                    renderLaunchGrid();
                }
            } catch (err) {
                console.error('Delete preset failed:', err);
                showToast('Failed to delete preset', 'error', err.message || String(err));
            }
        });

        card.querySelector('.launch-card-btn-start').addEventListener('click', () => {
            if (!hasModel) {
                window.__spawnWizardOpts = { templatePreset: preset };
                Router.navigate('/spawn');
                return;
            }
            // Phase 8a: resolve-and-launch is server-owned. The card ships only the
            // preset id + expected revision; the server resolves the bundle's saved
            // selection and launches. Bundles carry the revision; flat presets don't.
            const expectedRevision = bundleView ? (preset.revision ?? null) : null;
            import('./presets.js').then(({ syncSelectedPresetSelection }) => {
                syncSelectedPresetSelection(preset.id, { userIntent: true, persist: true });
                import('./attach-detach.js').then(({ doStartFromSetup }) => doStartFromSetup({ expectedRevision }));
            });
        });
    }

    return card;
}

// ── Card VRAM estimates (4-state: fit / tight / conditional / no-fit) ──────────

async function _fetchCardVramEstimates(availBytes, availRamBytes, isUnified, budgetIfPurgedBytes) {
    const cards = document.querySelectorAll('.launch-card[data-preset-id]');
    const presets = sessionState.presets || [];

    await Promise.all([...cards].map(async (card) => {
        const preset = presets.find(p => p.id === card.dataset.presetId);
        const modelPath = presetModelSource(preset);
        if (!modelPath) return;
        const isRapidMlx = preset?.backend === 'rapid_mlx';
        const vramEl = card.querySelector('.launch-card-vram');
        if (!vramEl) return;

        try {
            // Builder item 6: canonical body builder for cross-surface equality.
            const body = buildEstimateBody({
                backend: isRapidMlx ? 'rapid_mlx' : 'llama_cpp',
                model_path: modelPath,
                n_ctx: preset.context_size || 131072,
                parallel_slots: preset.parallel_slots || 1,
                ubatch_size: preset.ubatch_size || 1024,
                ctk: isRapidMlx ? undefined : (preset.ctk || 'q8_0'),
                ctv: isRapidMlx ? undefined : (preset.ctv || 'q8_0'),
                n_cpu_moe: preset.n_cpu_moe || 0,
                gpu_layers: preset.gpu_layers ?? -1,
                available_vram_bytes: availBytes,
                available_ram_bytes: availRamBytes,
                is_unified_memory: isUnified,
                ...(isRapidMlx ? rapidEstimatePolicyFromConfig(preset.rapid_mlx) : {}),
            });
            const headers = window.authHeaders ? window.authHeaders() : {};
            const resp = await fetch('/api/vram-estimate', {
                method: 'POST',
                headers: { ...headers, 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!resp.ok) return;
            const data = await resp.json();
            if (!data.ok) return;
            // Guard: card may have been re-rendered while awaiting
            if (!document.contains(vramEl)) return;
            if (isRapidMlx && data.effective_kv_dtype) {
                const kvChip = card.querySelector('.launch-card-chips .launch-chip:nth-child(2)');
                if (kvChip && data.execution_policy && data.execution_policy.kv_cache_dtype) {
                    const requested = data.execution_policy.kv_cache_dtype || 'int4';
                    const effective = data.effective_kv_dtype;
                    const reasoningOn = data.execution_policy.reasoning_mode === true;
                    if (reasoningOn && requested !== effective) {
                        kvChip.textContent = `${requested.toUpperCase()} → INT8 (reasoning)`;
                    } else {
                        kvChip.textContent = effective.toUpperCase();
                    }
                }
            }
            _renderCardVram(vramEl, data, availBytes, availRamBytes, isUnified, budgetIfPurgedBytes);
        } catch {
            // silently skip — VRAM row stays in loading state
        }
    }));
}

function _renderCardVram(el, data, availBytes, availRamBytes, isUnified, budgetIfPurgedBytes) {
    const hasAvail = availBytes > 0;
    const totalGb = data.total_bytes / 1e9;
    const weightsGb = data.weights_bytes / 1e9;
    const kvGb = data.kv_cache_bytes / 1e9;
    const extrasBytes = (data.mmproj_bytes || 0) + (data.mtp_bytes || 0) +
                        (data.linear_attn_state_bytes || 0) + (data.overhead_bytes || 0);

    // Determine 4-state classification.
    let state; // 'fit' | 'tight' | 'conditional' | 'nofit'
    let dotClass; // CSS class
    let hint; // short human-readable hint
    let dotTitle;

    if (isUnified) {
        const metalCap = _memState.metalCapBytes || 0;
        if (hasAvail && data.total_bytes <= (availBytes * 0.82)) {
            state = 'fit';
        } else if (hasAvail && data.total_bytes <= availBytes) {
            state = 'tight';
        } else if (budgetIfPurgedBytes > 0 &&
                   data.total_bytes > availBytes &&
                   data.total_bytes <= (budgetIfPurgedBytes * 0.85) &&
                   data.total_bytes <= metalCap) {
            // Fits after purging
            state = 'conditional';
        } else {
            state = 'nofit';
        }
    } else {
        // Discrete GPU: fit / tight / slower / nofit
        if (hasAvail && data.total_bytes <= (availBytes * 0.82)) {
            state = 'fit';
        } else if (hasAvail && data.total_bytes <= availBytes) {
            state = 'tight';
        } else if (availRamBytes > 0 &&
                   data.total_bytes > availBytes &&
                   data.total_bytes <= (availBytes + availRamBytes * 0.8)) {
            // Overflows to system RAM, but still fits
            state = 'conditional';
        } else {
            state = 'nofit';
        }
    }

    switch (state) {
        case 'fit':
            dotClass = 'fit';
            hint = '';
            dotTitle = 'Fits comfortably in memory';
            break;
        case 'tight':
            dotClass = 'tight';
            hint = 'Tight — may be slow or unstable';
            dotTitle = 'Tight fit — may work but leaves little headroom';
            break;
        case 'conditional':
            dotClass = 'conditional';
            if (isUnified) {
                hint = 'Free cache to run this';
                dotTitle = 'Will fit after flushing system caches';
            } else {
                hint = 'Runs slower — uses system RAM';
                dotTitle = 'Will run, but some layers use system memory and it will be slower';
            }
            break;
        default:
            dotClass = 'risk';
            hint = 'Over memory limit';
            dotTitle = 'Exceeds available memory — consider a smaller model or context';
            break;
    }

    // Denominator for bar: machine budget or reasonable fallback.
    const fitsInBudget = hasAvail && data.total_bytes <= availBytes;
    const denominator = hasAvail ? availBytes : data.total_bytes * 1.25;
    const toWidth = (b) => Math.min(100, (b / denominator) * 100).toFixed(1) + '%';

    // Free headroom segment (for fit/tight)
    const freeBytes = fitsInBudget ? (availBytes - data.total_bytes) : 0;
    const freeSegment = freeBytes > 0
        ? `<div class="launch-card-vram-seg launch-card-vram-seg--free launch-card-vram-seg--free-${dotClass}" style="width:${toWidth(freeBytes)}"></div>`
        : '';

    const availGb = hasAvail ? (availBytes / 1e9) : 0;

    const parts = [
        `Weights ${weightsGb.toFixed(1)} GB`,
        `KV ${kvGb.toFixed(1)} GB`,
    ];
    if (data.mmproj_bytes > 0) parts.push(`mmproj ${(data.mmproj_bytes / 1e9).toFixed(1)} GB`);
    if (data.mtp_bytes > 0)    parts.push(`MTP ${(data.mtp_bytes / 1e9).toFixed(1)} GB`);
    parts.push(`overhead ${(data.overhead_bytes / 1e9).toFixed(2)} GB`);
    if (hasAvail) parts.push(`${availGb.toFixed(1)} GB currently available`);
    if ((data.ram_bytes || 0) > 0) {
        parts.push(`CPU weights ${(data.ram_bytes / 1e9).toFixed(1)} GB`);
    }

    // Label: "approx. 27.0 / 26.8 GB" makes the over-budget case immediately obvious.
    const totalLabel = hasAvail
        ? `approx. ${totalGb.toFixed(1)} / ${availGb.toFixed(1)} GB`
        : `approx. ${totalGb.toFixed(1)} GB`;
    const overflowLabel = hasAvail && data.total_bytes > availBytes
        ? ` (+${((data.total_bytes - availBytes) / 1e9).toFixed(1)} GB)`
        : '';

    // RAM row (discrete GPU)
    const ramBytes = data.ram_bytes || 0;
    const ramDenominator = availRamBytes > 0 ? availRamBytes : ramBytes;
    const ramWidth = ramDenominator > 0
        ? Math.min(100, (ramBytes / ramDenominator) * 100).toFixed(1) + '%'
        : '0%';
    const ramOver = availRamBytes > 0 && ramBytes > availRamBytes;
    const ramLabel = availRamBytes > 0
        ? `${(ramBytes / 1e9).toFixed(1)} / ${(availRamBytes / 1e9).toFixed(1)} GB`
        : `${(ramBytes / 1e9).toFixed(1)} GB`;
    const ramRow = !isUnified && ramBytes > 0
        ? `<div class="launch-card-memory-row">
            <span class="launch-card-memory-kind">RAM</span>
            <div class="launch-card-vram-bar${ramOver ? ' launch-card-vram-bar--over' : ''}">
                <div class="launch-card-vram-seg launch-card-vram-seg--ram" style="width:${ramWidth}"></div>
            </div>
            <span class="launch-card-vram-total">${ramLabel}</span>
        </div>`
        : '';

    // Hint row (for conditional / nofit)
    const hintRow = hint
        ? `<div class="launch-card-memory-hint" title="${dotTitle}">${hint}</div>`
        : '';

    el.classList.remove('launch-card-vram--loading');
    el.title = parts.join(' · ');
    // Architecture invariant 19: surface the timestamp of the live availability
    // read the fit verdict was computed against, so a stale card is visibly stale.
    if (_memState.readAt) el.dataset.availabilityReadAt = _memState.readAt;
    else delete el.dataset.availabilityReadAt;
    // eslint-disable-next-line no-unsanitized/property -- all values are numeric; no user strings
    el.innerHTML = `
        <div class="launch-card-memory-bars">
            <div class="launch-card-memory-row">
                <span class="launch-card-memory-kind">${isUnified ? 'MEM' : 'VRAM'}</span>
                <div class="launch-card-vram-bar${hasAvail && !fitsInBudget ? ' launch-card-vram-bar--over' : ''}">
                    <div class="launch-card-vram-seg launch-card-vram-seg--weights" style="width:${toWidth(data.weights_bytes)}"></div>
                    <div class="launch-card-vram-seg launch-card-vram-seg--kv" style="width:${toWidth(data.kv_cache_bytes)}"></div>
                    <div class="launch-card-vram-seg launch-card-vram-seg--extras" style="width:${toWidth(extrasBytes)}"></div>
                    ${freeSegment}
                </div>
                <span class="launch-card-vram-dot launch-card-vram-dot--${dotClass}" title="${dotTitle}"></span>
            </div>
            <div class="launch-card-memory-total-row">
                <span class="launch-card-vram-total" title="Approximate total memory: ${parts.join(' · ')}">${totalLabel}${overflowLabel}</span>
            </div>
            ${hintRow}
            ${ramRow}
        </div>
    `;
}

function _buildNewConfigCard(isPrimary = false) {
    const card = document.createElement('div');
    card.className = 'launch-card launch-card--new' + (isPrimary ? ' launch-card--new-primary' : '');
    // eslint-disable-next-line no-unsanitized/property -- static HTML, no user data
        card.innerHTML = `
        <div class="launch-card-new-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        </div>
        <div class="launch-card-new-label">New model</div>
        ${isPrimary ? '<div class="launch-card-new-hint">Set up your first local model</div>' : ''}
    `;
    card.addEventListener('click', () => {
        Router.navigate('/spawn');
    });
    return card;
}

export function updateRunningCardHighlight() {
    const activePresetId = sessionState.activeSessionPresetId || '';
    document.querySelectorAll('.launch-card[data-preset-id]').forEach(card => {
        const isRunning = sessionState.serverRunning && card.dataset.presetId === activePresetId && activePresetId;
        card.classList.toggle('launch-card--running', !!isRunning);
        const badge = card.querySelector('.launch-card-running-badge');
        if (badge) badge.style.display = isRunning ? '' : 'none';
    });

    // Show the "Dashboard" toolbar button whenever a server is live or a remote endpoint is attached
    const dashBtn = document.getElementById('setup-dashboard-btn');
    if (dashBtn) {
        // Only show when server is confirmed running — activeSessionId defaults to 'default'
        // so it can't be used as a live-session signal.
        dashBtn.style.display = sessionState.serverRunning ? '' : 'none';
    }
}

// ── Recent Sessions ───────────────────────────────────────────────────────────

export async function loadRecentSessions() {
    try {
        const headers = (typeof window.authHeaders === 'function') ? window.authHeaders() : {};
        const resp = await fetch('/api/sessions/recent', { headers });
        if (!resp.ok) {
            // Recent sessions are optional; silently ignore 4xx/5xx to avoid noisy errors.
            return;
        }
        const data = await resp.json();
        renderRecentEndpoints(data.sessions, data.active_session_id);
    } catch {
        // Silent fail — first-run users or unreachable backend won't have this endpoint.
    }
}

export function renderRecentEndpoints(sessions, activeId) {
    const list = document.getElementById('setup-endpoint-list');
    const container = document.getElementById('setup-recent-endpoints');
    const attachBtn = document.getElementById('setup-attach-btn');
    const lastSession = setupViewState.lastSessionData || loadLastSessionData();
    if (!list || !container) return;

    const allSessions = Array.isArray(sessions) ? sessions : [];
    // Only remote attach sessions belong in this reconnect list. Local presets
    // already have first-class cards in the right pane and should not be
    // duplicated as stale "recent servers" here.
    const attachSessions = allSessions.filter(session => !!session.mode?.Attach);

    const hasAttachSessions = attachSessions.length > 0;
    container.style.display = hasAttachSessions ? '' : 'none';
    if (!hasAttachSessions) setAttachButtonLabel(attachBtn, 'Attach');
    list.innerHTML = '';

    // Stamp preset cards with the last time they were spawned (for "last launched" display).
    _spawnLastLaunched.clear();
    for (const session of allSessions.filter(s => !!s.mode?.Spawn)) {
        if (!session.preset_id || !session.last_connected_at) continue;
        const ts = session.last_connected_at * 1000;
        if (ts > (_spawnLastLaunched.get(session.preset_id) || 0)) {
            _spawnLastLaunched.set(session.preset_id, ts);
        }
    }
    _applyLastLaunchedToCards();
    // Refresh the right-pane order now that session-backed launch timestamps
    // are available for the default "Last launched" sort.
    renderLaunchGrid();

    if (!hasAttachSessions) return;

    // Attach and Spawn cards use distinct reconnect/restore paths below.
    const buildCard = (session) => {
        const card = document.createElement('div');
        card.className = 'setup-endpoint-card';
        if (activeId && session.id === activeId) {
            card.classList.add('is-active-session');
        }

        const endpoint = session.mode?.Attach?.endpoint || '';
        const apiKey = session.mode?.Attach?.api_key || '';

        const statusClass = session.status === 'Running' ? 'status-running' :
                            session.status === 'Error' ? 'status-error' : 'status-disconnected';

        const lastConnected = session.last_connected_at ? formatRelativeTime(session.last_connected_at * 1000) : 'Never';
        const connectCount = session.connect_count || 0;

        const statusDot = document.createElement('div');
        statusDot.className = 'setup-endpoint-status ' + statusClass;

        const infoWrap = document.createElement('div');
        infoWrap.className = 'setup-endpoint-info';

        const nameEl = document.createElement('div');
        nameEl.className = 'setup-endpoint-name';
        nameEl.textContent = session.name || endpoint || 'Unnamed';

        const endpointEl = document.createElement('div');
        endpointEl.className = 'setup-endpoint-url';
        endpointEl.textContent = endpoint;
        endpointEl.title = endpoint;

        const metaEl = document.createElement('div');
        metaEl.className = 'setup-endpoint-meta';
        const metaParts = [session.backend === 'rapid_mlx' ? 'Rapid-MLX' : 'llama.cpp'];
        if (activeId && session.id === activeId) metaParts.push('Active workspace');
        else if (session.status === 'Running') metaParts.push('Last seen running');
        else if (session.status === 'Disconnected') metaParts.push('Ready to reconnect');
        else if (session.status === 'Error') metaParts.push(session.last_error || 'Needs attention');
        if (lastSession?.endpoint && endpoint && lastSession.endpoint === endpoint && lastSession.telemetryLabel) {
            metaParts.push(lastSession.telemetryLabel);
        }
        if (lastConnected !== 'Never') metaParts.push(lastConnected);
        if (connectCount > 0) metaParts.push(connectCount + 'x');
        let meta = metaParts.join(' · ');
        if (!meta) meta = 'Saved endpoint';
        metaEl.textContent = meta;

        infoWrap.appendChild(nameEl);
        infoWrap.appendChild(endpointEl);
        infoWrap.appendChild(metaEl);

        const connectBtn = document.createElement('button');
        connectBtn.className = 'setup-endpoint-connect';

        const doConnect = async () => {
            let reconnectApiKey = apiKey;
            if (session.launch_requires_api_key && !reconnectApiKey) {
                reconnectApiKey = await showPromptDialog(
                    'Reconnect to protected endpoint',
                    'Enter the API key for this endpoint. It is used only for this connection and is not saved.',
                    '',
                    { type: 'password', confirmLabel: 'Reconnect' },
                );
                if (reconnectApiKey == null) return;
            }
            const urlInput = document.getElementById('setup-endpoint-url');
            if (urlInput) urlInput.value = endpoint;
            const apiKeyInput = document.getElementById('setup-endpoint-api-key');
            if (apiKeyInput) apiKeyInput.value = reconnectApiKey;
            const backendInput = document.getElementById('setup-endpoint-backend');
            if (backendInput) {
                backendInput.value = session.backend === 'rapid_mlx' ? 'rapid_mlx' : 'llama_cpp';
                backendInput.dispatchEvent(new Event('change'));
            }
            const modelInput = document.getElementById('setup-endpoint-model');
            if (modelInput) modelInput.value = session.model_identity || '';
            await doAttachFromSetup();
        };

        connectBtn.textContent = activeId && session.id === activeId
            ? 'Resume'
            : (session.last_connected_at ? 'Reconnect' : 'Connect');

        const dismissBtn = document.createElement('button');
        dismissBtn.type = 'button';
        dismissBtn.className = 'setup-endpoint-dismiss';
        dismissBtn.textContent = '×';
        dismissBtn.title = 'Dismiss saved endpoint';
        dismissBtn.setAttribute('aria-label', `Dismiss saved endpoint ${session.name || endpoint || ''}`.trim());
        dismissBtn.addEventListener('click', async (event) => {
            event.stopPropagation();
            dismissBtn.disabled = true;
            try {
                const baseHeaders = window.authHeaders ? window.authHeaders() : {};
                const tokenResponse = await fetch('/api/db/admin-token', { headers: baseHeaders });
                const tokenData = tokenResponse.ok ? await tokenResponse.json().catch(() => ({})) : {};
                const headers = tokenData.token
                    ? { Authorization: `Bearer ${tokenData.token}` }
                    : baseHeaders;
                const response = await fetch(`/api/sessions/${encodeURIComponent(session.id)}`, {
                    method: 'DELETE',
                    headers,
                });
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.error || `HTTP ${response.status}`);
                }
                card.remove();
                if (!list.children.length) {
                    container.style.display = 'none';
                    setAttachButtonLabel(attachBtn, 'Attach');
                }
                showToast('Saved endpoint dismissed', 'success');
            } catch (error) {
                dismissBtn.disabled = false;
                showToast(`Could not dismiss endpoint: ${error.message}`, 'error');
            }
        });
        card.appendChild(statusDot);
        card.appendChild(infoWrap);
        card.appendChild(connectBtn);
        card.appendChild(dismissBtn);

        connectBtn.addEventListener('click', (e) => { e.stopPropagation(); doConnect(); });
        card.addEventListener('click', doConnect);

        return card;
    };

    attachSessions.forEach(session => list.appendChild(buildCard(session)));
    // Live health-check attach sessions that aren't already confirmed Running
    attachSessions.forEach((session, i) => {
        if (session.status === 'Running') return;
        const endpoint = session.mode?.Attach?.endpoint;
        if (!endpoint) return;
        const card = list.children[i];
        if (!card) return;
        const dot = card.querySelector('.setup-endpoint-status');
        if (!dot) return;
        const authHdrs = window.authHeaders ? window.authHeaders() : {};
        const checkUrl = '/api/sessions/check-endpoint?url=' + encodeURIComponent(endpoint);
        fetch(checkUrl, { headers: authHdrs, signal: AbortSignal.timeout(6000) })
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data?.reachable) dot.className = 'setup-endpoint-status status-running'; })
            .catch(() => {});
    });
}

function formatRelativeTime(ts) {
    const diff = Date.now() - ts;
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return 'Just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + 'm ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + 'h ago';
    const days = Math.floor(hours / 24);
    return days + 'd ago';
}

// ── Session Data ──────────────────────────────────────────────────────────────

export function saveLastSessionData(data) {
    const payload = { ...data, timestamp: Date.now() };
    localStorage.setItem('llama-monitor-last-session', JSON.stringify(payload));
    setupViewState.lastSessionData = payload;
}

export function loadLastSessionData() {
    try {
        const raw = localStorage.getItem('llama-monitor-last-session');
        if (!raw) return null;
        const data = JSON.parse(raw);
        if (Date.now() - data.timestamp > 24 * 60 * 60 * 1000) {
            localStorage.removeItem('llama-monitor-last-session');
            return null;
        }
        return data;
    } catch {
        return null;
    }
}

// ── Previous Position ─────────────────────────────────────────────────────────

export function savePreviousPosition() {
    const activePage = document.querySelector('.page.active');
    const navTab = activePage?.id?.replace('page-', '') || 'server';
    const chatTabId = navTab === 'chat' ? chat.activeTabId : null;
    const scrollPosition = activePage?.scrollTop || 0;

    const position = {
        view: setupViewState.view,
        navTab,
        chatTabId,
        scrollPosition,
        timestamp: Date.now(),
    };

    localStorage.setItem('llama-monitor-previous-position', JSON.stringify(position));
    setupViewState.previousPosition = position;
}

export function loadPreviousPosition() {
    try {
        const raw = localStorage.getItem('llama-monitor-previous-position');
        if (!raw) return null;
        const data = JSON.parse(raw);
        if (Date.now() - data.timestamp > 24 * 60 * 60 * 1000) {
            localStorage.removeItem('llama-monitor-previous-position');
            return null;
        }
        return data;
    } catch {
        return null;
    }
}

export function clearPreviousPosition() {
    localStorage.removeItem('llama-monitor-previous-position');
    setupViewState.previousPosition = null;
}

export async function restorePreviousPosition() {
    const position = loadPreviousPosition();
    if (!position) return;

    // Switch to saved nav tab via Router
    if (position.navTab && position.navTab !== 'server') {
        if (position.navTab === 'chat') {
            Router.navigate('/chat');
        } else if (position.navTab === 'logs') {
            Router.navigate('/logs');
        } else {
            Router.navigate('/');
        }
    }

    // Switch to saved chat tab via Router
    if (position.chatTabId && position.navTab === 'chat') {
        Router.navigate('/chat/' + encodeURIComponent(position.chatTabId));
    }

    // Restore scroll position
    const activePage = document.querySelector('.page.active');
    if (activePage && position.scrollPosition > 0) {
        activePage.scrollTop = position.scrollPosition;
    }

    clearPreviousPosition();
}

// ── Quick Stats ───────────────────────────────────────────────────────────────

export function renderQuickStats() {
    const data = loadLastSessionData();
    const statsEl = document.getElementById('setup-stats');
    if (!statsEl) return;

    if (data) {
        const promptRate = document.getElementById('setup-last-prompt-rate');
        const genRate = document.getElementById('setup-last-gen-rate');
        const session = document.getElementById('setup-last-session');
        if (promptRate) promptRate.textContent = data.promptRate || '—';
        if (genRate) genRate.textContent = data.genRate || '—';
        if (session) session.textContent = data.sessionName || '—';
        statsEl.style.display = 'flex';
    } else {
        statsEl.style.display = 'none';
    }
}

export function syncSetupPresetSelect() {
    const setupSelect = document.getElementById('setup-preset-select');
    const mainSelect = document.getElementById('preset-select');
    if (!setupSelect || !mainSelect) return;

    // Mirror the main select (already filtered to visible presets by presets.js)
    setupSelect.innerHTML = '';
    const options = mainSelect.querySelectorAll('option');
    options.forEach(opt => {
        const clone = document.createElement('option');
        clone.value = opt.value;
        clone.textContent = opt.textContent;
        setupSelect.appendChild(clone);
    });
    setupSelect.value = mainSelect.value;

    // Also re-render the launch grid when presets change
    renderLaunchGrid();
}

// ── Launch Filters ────────────────────────────────────────────────────────────

export async function initLaunchFilters() {
    const bar = document.getElementById('setup-filter-bar');
    if (!bar) return;

    // Keep the sort control available as soon as there is a user preset. The
    // right pane is the canonical local-preset list, even when there are only
    // one or two cards; hiding the entire toolbar made ordering undiscoverable.
    const presets = sessionState.presets || [];
    const userPresets = presets.filter(p => !p.id.startsWith('default-'));
    if (userPresets.length === 0) {
        bar.style.display = 'none';
        return;
    }
    bar.style.display = '';

    // Populate family pills based on available families (from GGUF metadata)
    const families = new Set();
    userPresets.forEach(p => {
        const f = p.family;
        if (f) families.add(f);
    });

    const familyContainer = document.getElementById('setup-filter-family-pills');
    if (familyContainer) {
        // Keep "All" button
        const allBtn = familyContainer.querySelector('[data-filter="family-all"]');
        if (allBtn) allBtn.addEventListener('click', () => {
            launchFilters.family = null;
            updateFilterPillActive(familyContainer, 'family-all');
            renderLaunchGrid();
        });
        for (const fam of families) {
            const btn = document.createElement('button');
            btn.className = 'launch-filter-pill';
            btn.dataset.filter = fam;
            btn.type = 'button';
            btn.textContent = FAMILY_LABEL_MAP[fam] || fam;
            btn.addEventListener('click', () => {
                launchFilters.family = fam;
                updateFilterPillActive(familyContainer, fam);
                renderLaunchGrid();
            });
            familyContainer.appendChild(btn);
        }
    }

    // Size pills
    const sizeContainer = document.getElementById('setup-filter-size-pills');
    if (sizeContainer) {
        const sizes = ['tiny', 'small', 'medium', 'large'];
        const allBtn = sizeContainer.querySelector('[data-filter="size-all"]');
        if (allBtn) allBtn.addEventListener('click', () => {
            launchFilters.size = null;
            updateFilterPillActive(sizeContainer, 'size-all');
            renderLaunchGrid();
        });
        for (const s of sizes) {
            const btn = document.createElement('button');
            btn.className = 'launch-filter-pill';
            btn.dataset.filter = s;
            btn.type = 'button';
            btn.textContent = s.charAt(0).toUpperCase() + s.slice(1);
            btn.addEventListener('click', () => {
                launchFilters.size = s;
                updateFilterPillActive(sizeContainer, s);
                renderLaunchGrid();
            });
            sizeContainer.appendChild(btn);
        }
    }

    // Tags button: simple pill picker popup
    const tagsBtn = document.getElementById('setup-filter-tags-btn');
    if (tagsBtn) {
        const knownTags = ['coding', 'agentic', 'roleplay', 'creative', 'vision', 'fast'];
        tagsBtn.addEventListener('click', () => {
            if (document.querySelector('.launch-tags-popup')) return;
            const popup = document.createElement('div');
            popup.className = 'launch-tags-popup';
            // Build via DOM to satisfy no-unsanitized rule
            knownTags.forEach(t => {
                const label = document.createElement('label');
                label.className = 'launch-tag-option';
                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = t;
                label.appendChild(input);
                label.appendChild(document.createTextNode(t));
                popup.appendChild(label);
            });
            const apply = () => {
                launchFilters.tags = [...popup.querySelectorAll('input:checked')].map(i => i.value);
                if (launchFilters.tags.length > 0) {
                    tagsBtn.classList.add('launch-filter-pill--active');
                    tagsBtn.textContent = `+ ${launchFilters.tags.length}`;
                } else {
                    tagsBtn.classList.remove('launch-filter-pill--active');
                    tagsBtn.textContent = '+ Tags';
                }
                popup.remove();
                renderLaunchGrid();
            };
            popup.addEventListener('click', e => {
                if (e.target.tagName === 'INPUT') {
                    setTimeout(apply, 50);
                }
            });
            tagsBtn.parentElement.appendChild(popup);
        });
    }

    // Collections dropdown
    const collectionsGroup = document.getElementById('setup-filter-collections-group');
    const collectionsSelect = document.getElementById('setup-filter-collections');
    if (collectionsSelect && collectionsGroup) {
        const collections = sessionState.collections || [];
        if (collections.length > 0) {
            collectionsGroup.style.display = '';
            collectionsSelect.innerHTML = '<option value="all">All</option>';
            collections.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c.id;
                opt.textContent = c.name;
                collectionsSelect.appendChild(opt);
            });
            collectionsSelect.addEventListener('change', () => {
                launchFilters.collection = collectionsSelect.value === 'all' ? null : collectionsSelect.value;
                renderLaunchGrid();
            });
        }
    }

    // Group by family toggle
    const groupToggle = document.getElementById('setup-filter-group-by-family');
    if (groupToggle) {
        const saved = localStorage.getItem('llama-monitor-group-by-family');
        if (saved === '1') {
            launchFilters.groupByFamily = true;
            groupToggle.checked = true;
        }
        groupToggle.addEventListener('change', () => {
            launchFilters.groupByFamily = !!groupToggle.checked;
            localStorage.setItem('llama-monitor-group-by-family', launchFilters.groupByFamily ? '1' : '0');
            renderLaunchGrid();
        });
    }
    const sortSelect = document.getElementById('setup-filter-sort');
    if (sortSelect) {
        const savedSort = localStorage.getItem('llama-monitor-preset-sort');
        if (savedSort && ['last_launched', 'name', 'size', 'backend'].includes(savedSort)) {
            launchFilters.sortBy = savedSort;
            sortSelect.value = savedSort;
        }
        sortSelect.addEventListener('change', () => {
            launchFilters.sortBy = sortSelect.value;
            localStorage.setItem('llama-monitor-preset-sort', launchFilters.sortBy);
            renderLaunchGrid();
        });
    }
    bar.dataset.initialized = '1';
}

function updateFilterPillActive(container, activeId) {
    if (!container) return;
    container.querySelectorAll('.launch-filter-pill').forEach(btn => {
        btn.classList.toggle('launch-filter-pill--active', btn.dataset.filter === activeId);
    });
}

// ── Initialization ────────────────────────────────────────────────────────────

export function initViewState() {
    if (document.body.classList.contains('setup-active')) return; // already initialized
    renderQuickStats();
    syncSetupPresetSelect(); // also calls renderLaunchGrid
    const lastEndpoint = localStorage.getItem('llama-monitor-last-endpoint');
    if (lastEndpoint) {
        const input = document.getElementById('setup-endpoint-url');
        if (input) input.value = lastEndpoint;
    }

    // Bind models button
    document.getElementById('setup-models-btn')?.addEventListener('click', () => {
        import('./models.js').then(({ openModelsModal }) => openModelsModal());
    });

    // Dashboard button: visible when a server is running or remote endpoint attached
    const dashBtn = document.getElementById('setup-dashboard-btn');
    if (dashBtn) {
        dashBtn.addEventListener('click', () => Router.navigate('/server'));
    }

    // Init filter bar after presets are loaded
    initLaunchFilters();

    document.body.classList.add('setup-active');
    const setupView = document.getElementById('view-setup');
    const monitorView = document.getElementById('view-monitor');
    if (setupView) {
        setupView.style.display = '';
        setupView.classList.add('entering');
        setTimeout(() => setupView.classList.remove('entering'), 600);
    }
    if (monitorView) monitorView.style.display = 'none';
    loadRecentSessions();

// ── New Configuration button (header) ──────────────────────────────────

    const btn = document.getElementById('setup-new-config-btn');
    if (btn) {
        btn.addEventListener('click', async () => {
            try {
                Router.navigate('/spawn');
            } catch (e) {
                console.error('Failed to open spawn wizard from new-config button:', e);
            }
        });
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function initSetupView() {
    // Resolve the render kill-switch before the first grid paint. The flag is a
    // closed enum from /api/preset-cards; a bundle preset must not flash as the
    // compact card (or the legacy flat card) until we know which mode to render.
    await _refreshPresetBundleUi();
    // Initialize view state immediately — defensive functions return early if DOM not ready
    initViewState();
}
