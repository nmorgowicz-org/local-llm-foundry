// Hardware-step VRAM/RAM breakdown rendering for the spawn wizard.
// Backend-agnostic: all sizing math comes from /api/vram-estimate (Rust
// vram_estimator, single source of truth). This module only renders whichever
// fields the estimate response includes -- e.g. llama.cpp's unified
// kv_cache_bytes vs. Rapid-MLX's active/retained KV split and prefix-cache
// budget -- it does not branch on engine itself.
import {
  dom, wizardState,
  effectiveAvailBytes, getSizingArch, getModelBytes, isUnifiedMemory,
  metalCap, suggestedMetalLimitMb,
  cachedRamTotal, cachedRamUsed, cachedMetalGpuLimitMb,
  maybeResetHardwareStepScroll, maybeRestoreHardwareStepScroll,
  updateAdvisor, fetchGpuVram, fetchMetalGpuLimit, scheduleVramUpdate,
} from './spawn-wizard.js';
import { _platformInfo } from './spawn-wizard-binary-prereq.js';
import {
  CTX_TARGETS, updateCtxModelMaxHint, updateCtxQuickPickActive, updateCtxTrainWarning,
} from './spawn-wizard-context-fit.js';
import {
  formatCtx, formatGB, formatVramTotal,
} from './spawn-wizard-format.js';
import { scheduleEstimate, buildEstimateBody, rapidEstimatePolicyFromWizardHardware } from './vram-estimate.js';
import { openEstimateEvidenceDrawer } from './evidence-drawer.js';
import { showToast } from './toast.js';
import { lastSystemMetrics } from '../core/app-state.js';
import { renderStickyVramBar, initStickyVramBar } from './spawn-wizard-sticky-vram.js';

// Initialize sticky VRAM bar visibility
initStickyVramBar();

function updateMlockWarning(availBytes = 0, freeBytes = null) {
  const el = document.getElementById('spawn-mlock-warning');
  if (!el) return;
  if (!dom.mlockCheck?.checked) {
    el.style.display = 'none';
    el.textContent = '';
    return;
  }

  const tight = freeBytes != null && availBytes > 0 && freeBytes < availBytes * 0.18;
  const unified = isUnifiedMemory();
  const platform = unified ? 'On unified-memory Macs, ' : '';

  // Check if this model would push total system RAM above 90% when mlock is on.
  // Above that threshold all model memory is non-compressible and macOS can stall.
  const sys = lastSystemMetrics;
  const modelGib = getModelBytes() / (1024 ** 3);
  const totalRamGb = sys?.ram_total_gb || 0;
  const usedRamGb = sys?.ram_used_gb || 0;
  const projectedPct = totalRamGb > 0
    ? Math.round(((usedRamGb + modelGib) / totalRamGb) * 100)
    : 0;
  const wiredOverload = unified && totalRamGb > 0 && projectedPct >= 90;

  let tail;
  if (wiredOverload) {
    const wiredGb = (sys?.memory_wired_gb || 0) + modelGib;
    tail = ` Loading this model will use ~${projectedPct}% of system RAM.`
      + ` With mlock on, all of that is non-compressible — macOS cannot relieve pressure and the desktop may become unresponsive.`
      + ` Consider disabling mlock: on Apple Silicon, Metal keeps model memory resident while the server is running.`
      + ` (System wired would be ~${wiredGb.toFixed(1)} GiB after loading.)`;
  } else if (tight) {
    tail = ' This configuration is already close to the memory budget, so pinned memory can push macOS into compression or swap and make the desktop unresponsive.';
  } else {
    tail = ' Leave enough free system memory for macOS, browsers, downloads, and build tools.';
  }
  el.textContent = `${platform}mlock pins model memory instead of letting the OS reclaim it.${tail}`;
  el.className = wiredOverload
    ? 'ctx-fit-warning ctx-fit-error ctx-fit-mlock-suggest-off'
    : tight ? 'ctx-fit-warning ctx-fit-error' : 'ctx-fit-warning';
  el.style.display = '';
}

export function updateVramDisplay() {
  const availVram = effectiveAvailBytes();
  if (!dom.vramPanel) return;

  const hw = wizardState.hardware;
  const arch = getSizingArch();
  const modelBytes = getModelBytes();
  const nCpuMoe = hw.nCpuMoe || 0;

    // Ensure vram-estimate uses accurate effective values
    wizardState.vram.available = availVram;
    wizardState.vram.availableRam = isUnifiedMemory()
      ? 0
      : Math.max(0, cachedRamTotal - cachedRamUsed);
    wizardState.vram.isUnifiedMemory = isUnifiedMemory();

  // Always render scenario cards so the UI doesn't go blank when the backend is unavailable.
  // renderScenarioCards will degrade gracefully if individual estimates fail.
  renderScenarioCards(modelBytes, arch, availVram);

  scheduleEstimate(wizardState, (est) => {
    if (!est || !est.total_bytes) {
      // Fallback: clear or dim the panel; don't show misleading numbers.
      if (dom.vramBar) dom.vramBar.classList.remove('has-data', 'tight', 'over');
      return;
    }

    const nativeContextLimit = Number(est.native_context_limit || 0);
    if (nativeContextLimit > 0 && wizardState.model.nCtxTrain !== nativeContextLimit) {
      wizardState.model.nCtxTrain = nativeContextLimit;
      updateCtxModelMaxHint();
      updateCtxQuickPickActive();
      updateCtxTrainWarning();
    }

    const total = est.total_bytes;
    const headroom = est.headroom_bytes || 0;
    const free = headroom; // backend headroom_bytes = available - total
    const weightVram = est.weights_bytes || 0;

    // Builder item 6: Rapid-MLX active/retained KV split — distinct totals.
    // When Rapid-MLX with workload_scenario returns separate active/retained values,
    // display them distinctly. Otherwise use unified kv_cache_bytes.
    const activeKV = est.active_kv_bytes || 0;
    const retainedKV = est.retained_kv_bytes || 0;
    const hasKVSplit = activeKV > 0 && retainedKV > 0;
    const kv = hasKVSplit ? 0 : (est.kv_cache_bytes || 0); // unified KV when no split
    const mmproj = est.mmproj_bytes || 0;
    const mtp = est.mtp_bytes || 0;
    const tqTransient = est.turboquant_transient_peak_bytes || 0;
    const oh = est.overhead_bytes || 0;
    // Phase 6 Part B: prefix cache budget display (informational, not consumed until active).
    // Show when budget exists (backend returns > 0 when configured_ceiling_bytes > 0).
    const prefixCacheBudget = est.mlx_prefix_cache_bytes || 0;
    // Rapid-MLX overhead is a documented formula-based approximation (not yet calibrated
    // against real Apple Silicon measurements); surface that in the tooltip rather than
    // presenting it as precisely measured. Backend-driven — no local VRAM math here.
    const evidenceSuffix = est.evidence === 'approximate'
      ? ' (Rapid-MLX overhead is an approximation, not yet hardware-calibrated.)'
      : est.evidence === 'degraded'
        ? ' (Architecture metadata incomplete — this estimate is a rough heuristic.)'
        : '';
    const note = (est.note || '') + evidenceSuffix;

    const explain = document.getElementById('wizard-vram-explain');
    if (explain) explain.onclick = () => openEstimateEvidenceDrawer(est, 'Setup memory estimate', explain);

    updateMlockWarning(availVram, free);

    // Update total label — leads with the model's estimated usage (what the bar
    // graph actually represents) rather than just the ceiling, since the ceiling
    // alone gives no sense of how much of it this model will consume.
    if (dom.vramPanelTotal) {
      if (availVram > 0) {
        const usedOfTotal = `${formatGB(total)} / ${formatVramTotal(availVram)}`;
        if (isUnifiedMemory() && cachedRamTotal > 0) {
          dom.vramPanelTotal.textContent =
            usedOfTotal + ' (Metal cap of ' + formatVramTotal(cachedRamTotal) + ' total)';
        } else {
          dom.vramPanelTotal.textContent = usedOfTotal;
        }
      } else {
        dom.vramPanelTotal.textContent = formatGB(total) + ' estimated — ' +
          (isUnifiedMemory() ? 'unified memory total unknown' : 'GPU VRAM total unknown');
      }
      dom.vramPanelTotal.title = note;
    }

    // Update bar segments (width as % of availVram or total, whichever is larger)
    const denom = availVram > 0 ? availVram : total;
    if (denom > 0) {
      setSegWidth(dom.vSegWeights,  weightVram / denom);
      // Builder item 6: show active/retained split for Rapid-MLX when available.
      // When split is present, vSegKv shows active KV; vSegOverhead absorbs retained.
      // This maintains visual distinction without requiring new DOM segments.
      if (hasKVSplit) {
        setSegWidth(dom.vSegKv,       (activeKV + retainedKV) / denom);
        if (dom.vLegKvLabel) dom.vLegKvLabel.textContent = `KV ${formatGB(activeKV + retainedKV)} (active ${formatGB(activeKV)})`;
      } else {
        setSegWidth(dom.vSegKv,       kv / denom);
        if (dom.vLegKvLabel) dom.vLegKvLabel.textContent = `KV ${formatGB(kv)}`;
      }
      setSegWidth(dom.vSegMmproj,   mmproj / denom);
      setSegWidth(dom.vSegMtp,      mtp / denom);
      setSegWidth(dom.vSegOverhead, (oh + tqTransient) / denom);
      setSegWidth(dom.vSegFree,     Math.max(0, free) / denom);
      if (dom.vSegFree) dom.vSegFree.classList.toggle('over-budget', free < 0);
    }

    // Bar state class
    if (dom.vramBar) {
      const ratio = availVram > 0 ? total / availVram : 0;
      dom.vramBar.classList.toggle('tight', ratio >= 0.88 && ratio < 1.0);
      dom.vramBar.classList.toggle('over', ratio >= 1.0);
      dom.vramBar.classList.toggle('has-data', total > 0);
    }

    // Render sticky header inline VRAM bar (M3-A)
    renderStickyVramBar({
      vramTotal: availVram,
      vramWeights: weightVram,
      vramKv: hasKVSplit ? activeKV + retainedKV : kv,
      vramMmproj: mmproj,
      vramMtp: mtp,
      vramOverhead: oh + tqTransient,
    });

    // Update legend labels
    if (dom.vLegWeightsLabel) dom.vLegWeightsLabel.textContent = `Weights ${formatGB(weightVram)}`;
    // KV label set above (with split annotation if applicable).
    if (mmproj > 0) {
      if (dom.vLegMmprojItem)  dom.vLegMmprojItem.style.display = '';
      if (dom.vLegMmprojLabel) dom.vLegMmprojLabel.textContent  = `mmproj ${formatGB(mmproj)}`;
    } else {
      if (dom.vLegMmprojItem) dom.vLegMmprojItem.style.display = 'none';
    }
    if (mtp > 0) {
      if (dom.vLegMtpItem)  dom.vLegMtpItem.style.display = '';
      if (dom.vLegMtpLabel) dom.vLegMtpLabel.textContent  = `MTP ${formatGB(mtp)}`;
    } else {
      if (dom.vLegMtpItem) dom.vLegMtpItem.style.display = 'none';
    }
    if (dom.vLegOverheadLabel) dom.vLegOverheadLabel.textContent = `OH ${formatGB(oh)}`;
    if (dom.vLegFreeLabel) {
      const freeAbs = Math.abs(free);
      dom.vLegFreeLabel.textContent = free >= 0 ? `Free ${formatGB(free)}` : `Over ${formatGB(freeAbs)}`;
      if (dom.vLegFreeDot) dom.vLegFreeDot.style.background = free >= 0 ? '' : 'var(--color-error)';
    }

    // Phase 6 Part B: show prefix cache budget legend when budget exists.
    if (prefixCacheBudget > 0) {
      if (dom.vLegPrefixCacheItem) dom.vLegPrefixCacheItem.style.display = '';
      if (dom.vLegPrefixCacheLabel) dom.vLegPrefixCacheLabel.textContent = `Rapid retained cache ${formatGB(prefixCacheBudget)}`;
    } else {
      if (dom.vLegPrefixCacheItem) dom.vLegPrefixCacheItem.style.display = 'none';
    }

    // Show/hide MoE panel
    if (arch.nExperts > 1) {
      if (dom.moeOffloadPanel) dom.moeOffloadPanel.style.display = '';
      if (dom.moeOffloadSlider) {
        dom.moeOffloadSlider.max = arch.nLayers || arch.nExperts;
        dom.moeOffloadSlider.value = nCpuMoe;
      }
      updateMoeSliderVisuals();
    } else {
      if (dom.moeOffloadPanel) dom.moeOffloadPanel.style.display = 'none';
    }

    // Config-time performance advisor (dense-vs-MoE, KV type, MTP)
    updateAdvisor();

    // Legacy VRAM pill (backward compat)
    if (dom.vramPill || dom.vramEstimateText) {
      updateLegacyVramPill(total, availVram);
    }

    // Unified memory label (Apple Silicon / DGX Spark — VRAM and RAM are the same pool)
    const isUnified = _platformInfo?.auto_backend === 'metal';
    if (dom.vramPanelLabel) {
      dom.vramPanelLabel.textContent = isUnified ? 'Unified Memory' : 'VRAM budget';
    }

    // Metal GPU limit row — Apple Silicon only
    if (dom.metalLimitRow) {
      if (isUnified && cachedRamTotal > 0) {
        dom.metalLimitRow.style.display = '';
        const currentCapMb = Math.round(metalCap(cachedRamTotal) / (1024 * 1024));
        const isCustom = cachedMetalGpuLimitMb > 0;
        const capGb = (currentCapMb / 1024).toFixed(0);
        const totalGb = (cachedRamTotal / (1024 ** 3)).toFixed(0);
        const label = isCustom
          ? `Metal GPU cap: ${capGb} GB (custom) — of ${totalGb} GB total`
          : `Metal GPU cap: ${capGb} GB (default) — of ${totalGb} GB total`;
        if (dom.metalLimitText) dom.metalLimitText.textContent = label;

        // Show "Increase" button if a meaningfully larger cap is achievable
        const suggested = suggestedMetalLimitMb(cachedRamTotal);
        if (dom.metalLimitBtn) {
          if (suggested > 0) {
            const suggestedGb = Math.round(suggested / 1024);
            dom.metalLimitBtn.disabled = false; // clear disabled from any previous attempt
            dom.metalLimitBtn.style.display = '';
            dom.metalLimitBtn.textContent = `Increase to ${suggestedGb} GB`;
            dom.metalLimitBtn.onclick = () => applyMetalGpuLimit(suggested);
            // Remove any stale fallback panel from a previous failed attempt
            dom.metalLimitRow?.querySelector('.metal-limit-fallback')?.remove();
          } else {
            dom.metalLimitBtn.style.display = 'none';
          }
        }
      } else {
        dom.metalLimitRow.style.display = 'none';
      }
    }

    // RAM bar — only shown on discrete GPU systems; on unified the VRAM bar already covers it
    if (dom.ramPanel) {
      if (isUnified || cachedRamTotal === 0) {
        dom.ramPanel.style.display = 'none';
      } else {
        dom.ramPanel.style.display = '';
        const cramMib = (hw.cacheRam !== null && hw.cacheRam !== undefined) ? hw.cacheRam : 8192;
        const cramBytes = cramMib < 0 ? 0 : cramMib * 1024 * 1024;
        const ramDenom = cachedRamTotal;
        const inUsePct   = cachedRamUsed / ramDenom;
        const moePct     = (est.ram_bytes || 0) / ramDenom;
        const cramPct    = cramBytes / ramDenom;
        const freePct    = Math.max(0, (cachedRamTotal - cachedRamUsed - (est.ram_bytes || 0) - cramBytes) / ramDenom);
        setSegWidth(dom.rSegUsed, inUsePct);
        setSegWidth(dom.rSegMoe,  moePct);
        setSegWidth(dom.rSegCram, cramPct);
        setSegWidth(dom.rSegFree, freePct);
        const totalNeeded = cachedRamUsed + (est.ram_bytes || 0) + cramBytes;
        const isOver = totalNeeded > cachedRamTotal;
        if (dom.ramPanelTotal) {
          dom.ramPanelTotal.textContent = formatVramTotal(cachedRamTotal) + ' total';
        }
        if (dom.rLegUsed)  dom.rLegUsed.textContent  = `In use ${formatGB(cachedRamUsed)}`;
        if (dom.rLegCram) {
          const cramLabel = cramMib < 0 ? 'no limit' : `${formatGB(cramBytes)}`;
          dom.rLegCram.textContent = `Cache ${cramLabel}`;
        }
        if (est.ram_bytes > 0) {
          if (dom.rLegMoeItem) dom.rLegMoeItem.style.display = '';
          if (dom.rLegMoe) {
            const label = arch.nExperts > 1 ? 'MoE experts' : 'CPU weights';
            dom.rLegMoe.textContent = `${label} ${formatGB(est.ram_bytes)}`;
          }
        } else {
          if (dom.rLegMoeItem) dom.rLegMoeItem.style.display = 'none';
        }
        const freeBytes = cachedRamTotal - totalNeeded;
        if (dom.rLegFree) {
          dom.rLegFree.textContent = isOver
            ? `Over ${formatGB(Math.abs(freeBytes))}`
            : `Free ${formatGB(freeBytes)}`;
        }
        if (dom.ramPanel) dom.ramPanel.classList.toggle('over-budget', isOver);
      }
    }

    maybeResetHardwareStepScroll();
    maybeRestoreHardwareStepScroll();

    // Inline VRAM exceeded warning (red feedback under context input)
    const ctxVramEl = document.getElementById('ctx-vram-warning');
    if (ctxVramEl) {
      if (free < 0) {
        ctxVramEl.textContent = `VRAM exceeded by ${formatGB(Math.abs(free))}. Reduce context size or use KV quantization.`;
        ctxVramEl.className = 'ctx-fit-warning ctx-fit-error';
        ctxVramEl.style.display = '';
      } else {
        ctxVramEl.style.display = 'none';
      }
    }
  });
}

function setSegWidth(el, frac) {
  if (!el) return;
  const pct = Math.max(0, Math.min(1, frac)) * 100;
  el.style.width = pct.toFixed(2) + '%';
  el.style.display = pct < 0.3 ? 'none' : '';
}

export function updateMoeSliderVisuals() {
  const arch = getSizingArch();
  if (!(arch.nExperts > 0)) return; // not a MoE model
  const n = arch.nLayers || 0;
  if (!n) return;
  const cpu = wizardState.hardware.nCpuMoe || 0;
  const gpu = n - cpu;
  const pct = cpu / n * 100;

  if (dom.moeOffloadSlider) {
    dom.moeOffloadSlider.style.background =
         `linear-gradient(90deg, var(--color-purple) ${pct.toFixed(1)}%, var(--neutral-soft-bg-strong) ${pct.toFixed(1)}%)`;
  }
  if (dom.moeOffloadSubtitle) {
    dom.moeOffloadSubtitle.textContent = `${cpu} of ${n} experts on CPU · ${gpu} in VRAM`;
  }
  if (dom.moeOffloadHint) {
    if (cpu === 0) {
      dom.moeOffloadHint.textContent = 'All experts in VRAM — fastest generation.';
    } else if (cpu >= n) {
      dom.moeOffloadHint.textContent = 'All experts on CPU — slowest generation. Only use if VRAM is very tight.';
    } else {
      const speedPenalty = Math.round((cpu / n) * 60);
      dom.moeOffloadHint.textContent = `~${speedPenalty}% generation speed reduction. More context available.`;
    }
  }
}

// ── Context fit modes ────────────────────────────────────────────────────────

function updateContextRailSummary() {
  if (!dom.ctxRailSummaryValue || !dom.ctxRailSummaryStatus || !dom.ctxRailSummaryNote) return;

  const currentCtx = wizardState.hardware.contextSize || 8192;
  const nCtxTrain = wizardState.model.nCtxTrain || 0;
  const target = CTX_TARGETS[wizardState.useCase] || 0;

  dom.ctxRailSummaryValue.textContent = formatCtx(currentCtx);
  dom.ctxRailSummaryStatus.classList.remove('warning');

  if (nCtxTrain && currentCtx > nCtxTrain) {
    dom.ctxRailSummaryStatus.textContent = 'Outside native context';
    dom.ctxRailSummaryStatus.classList.add('warning');
    dom.ctxRailSummaryNote.textContent = `Model max is ${formatCtx(nCtxTrain)}. This may fit memory but is untested and requires Advanced Context extension controls, which are not configured.`;
    return;
  }

  if (nCtxTrain && currentCtx === nCtxTrain) {
    dom.ctxRailSummaryStatus.textContent = 'At model max';
    dom.ctxRailSummaryNote.textContent = 'You are using the model’s full native context. Higher values require separately qualified Advanced Context extension controls.';
    return;
  }

  if (nCtxTrain) {
    dom.ctxRailSummaryStatus.textContent = 'Within trained context';
    dom.ctxRailSummaryNote.textContent = currentCtx < target
      ? `Use-case target is ${formatCtx(target)}. Lower values save memory, but leave less room for long chats, retrieval, or tool loops.`
      : `Model max is ${formatCtx(nCtxTrain)}. Higher values are advanced-only and untested until context-extension support is qualified.`;
    return;
  }

  dom.ctxRailSummaryStatus.textContent = 'Training max unavailable';
  dom.ctxRailSummaryNote.textContent = currentCtx < target
    ? `Use-case target is ${formatCtx(target)}. Larger contexts use more KV memory.`
    : 'Larger contexts use more KV memory. Stay conservative unless you know the model’s training limit.';
}

// Dispatches to the per-engine card set. The two branches diverge structurally, not just in
// styling: llama.cpp's cards vary the KV-quant axis at a fixed context, while MLX's cards vary
// concurrency/retained-cache at a fixed KV dtype (int8, pinned by the reasoning profile — §2.6).
// A shared "gate on modelBytes only" entry point keeps that divergence from leaking upward.
// Both render*ScenarioCards() functions clear dom.vramScenarios synchronously,
// then await network estimates before appending. If a second call starts
// before the first's fetches resolve, the second call's clear wipes an
// already-empty container and both calls go on to append their own set of
// cards — doubling the grid. This token makes a stale call's append a no-op.
let scenarioCardsRenderToken = 0;

async function renderScenarioCards(modelBytes, arch, availVram) {
  if (!dom.vramScenarios || !availVram || !modelBytes) return;
  const token = ++scenarioCardsRenderToken;
  if (wizardState.engine.selected === 'rapid_mlx') {
    return renderMlxScenarioCards(modelBytes, arch, availVram, token);
  }
  return renderLlamaCppScenarioCards(modelBytes, arch, availVram, token);
}

async function renderLlamaCppScenarioCards(_modelBytes, _arch, availVram, token) {
  const hw = wizardState.hardware;
  const uc = wizardState.useCase;
  const nCtxTrain = wizardState.model.nCtxTrain || 0;
  const currentCtx = hw.contextSize || 8192;

  updateContextRailSummary();

  const scenarios = [
    {
      key: 'q8_0',
      mode: 'Reliable agents',
      detail: 'High-precision KV cache',
      kk: 'q8_0',
      kv: 'q8_0',
      desc: uc === 'roleplay' ? 'Best long-context quality with less headroom for very large transcripts.' : 'Best long-context quality for tools, retrieval, and multi-step work.',
      rec: uc !== 'roleplay',
    },
    {
      key: 'q4_0',
      mode: 'More context',
      detail: 'Lower-precision KV cache',
      kk: 'q4_0',
      kv: 'q4_0',
      desc: uc === 'agentic' ? 'Fits more tokens, but lower cache precision can hurt tool-call coherence.' : 'Fits the most context if you care more about length than cache quality.',
      rec: uc === 'roleplay',
      warnAgentic: uc === 'agentic',
    },
    {
      key: 'f16',
      mode: 'Full precision',
      detail: 'Lossless KV cache',
      kk: 'f16',
      kv: 'f16',
      desc: 'Uses the most KV memory. Best reserved for comparison or when you want the most exact cache.',
      rec: false,
    },
  ];

  dom.vramScenarios.innerHTML = '';
  const activeQuant = hw.cacheTypeK === '' ? 'f16' : (hw.cacheTypeK || 'q8_0');

  // For each scenario, ask the backend if current context fits with that KV quant.
  const fitResults = await Promise.all(
    scenarios.map(async (s) => {
      try {
        // Builder item 6: canonical body builder for cross-surface equality.
        const body = buildEstimateBody({
          backend: wizardState.engine.selected || 'llama_cpp',
          model_path: wizardState.model.path || '',
          n_ctx: currentCtx,
          parallel_slots: hw.parallelSlots || 1,
          ubatch_size: hw.ubatchSize || 2048,
          ctk: s.kk,
          ctv: s.kv,
          n_cpu_moe: hw.nCpuMoe || 0,
          available_vram_bytes: availVram,
          is_unified_memory: isUnifiedMemory(),
          mmproj_path: wizardState.model.mmprojPath || null,
          mmproj_bytes: wizardState.arch.mmprojBytes || 0,
          ...(wizardState.engine.selected === 'rapid_mlx' ? rapidEstimatePolicyFromWizardHardware(hw) : {}),
        });
        const headers = (window.authHeaders ? window.authHeaders() : {});
        const res = await fetch('/api/vram-estimate', {
          method: 'POST',
          headers: { ...headers, 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) return null;
        const d = await res.json();
        if (!d.ok || d.total_bytes == null) return null;
        return d;
      } catch {
        return null;
      }
    }),
  );

  if (token !== scenarioCardsRenderToken) return;
  dom.vramScenarios.innerHTML = '';

  for (let i = 0; i < scenarios.length; i++) {
    const s = scenarios[i];
    const est = fitResults[i];
    const headroom = est ? (est.headroom_bytes ?? 0) : 0;
    const rec = est ? (est.recommendation || 'risk') : 'risk';

    const fits = headroom >= 0;
    const isTight = rec === 'tight';
    const over = !fits || rec === 'wont_fit';

    const ctx = over ? null : currentCtx;
    const cappedByModel = nCtxTrain > 0 && currentCtx >= nCtxTrain;
    const selectable = !!ctx && !over;

    const card = document.createElement('div');
    const isActive = s.key === activeQuant;
    card.className = 'vram-scenario-card' + (s.rec ? ' scenario-rec' : '') + (isActive ? ' selected' : '');
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `${s.mode}${selectable ? ': ' + formatCtx(ctx) + ' tokens' : ' — may not fit current context'} — ${s.desc}`);

    let desc = s.desc;
    if (!selectable) {
      desc = 'At your current context, this may not fit VRAM. Lower context or use a more efficient KV cache.';
    } else if (cappedByModel) {
      if (s.key === 'q8_0') desc = 'Best long-context quality. VRAM is no longer the limit.';
      else if (s.key === 'q4_0') desc = 'More headroom, but the model max is already the real ceiling.';
      else if (s.key === 'f16') desc = 'Full precision cache. Context is capped by the model, not VRAM.';
    } else if (isTight) {
      desc = 'Fits, but leaves little headroom. Watch GPU memory or reduce context.';
    }

    const limitNote = cappedByModel && selectable ? '<span class="vsc-limit-note">model max</span>' : '';

    // All values are internal constants — no user input reaches this template.
    // eslint-disable-next-line no-unsanitized/property
    card.innerHTML = `
      <div class="vsc-mode-name">${s.mode}</div>
      <div class="vsc-mode-detail">${s.detail}</div>
      <div class="vsc-ctx-row">
        <span class="vsc-ctx">${selectable ? formatCtx(ctx) : '—'}</span>
        ${selectable ? '<span class="vsc-ctx-unit">tokens</span>' : ''}
        ${limitNote}
      </div>
      <div class="vsc-desc">${desc}</div>
      ${s.rec ? '<span class="vsc-rec-badge">★ Recommended</span>' : ''}
      ${isActive ? '<span class="vsc-active-badge">✓ Active</span>' : ''}
      ${s.warnAgentic ? '<span class="vsc-warn">⚠ Not ideal for tool-heavy agents</span>' : ''}
      ${over ? '<span class="vsc-warn">⚠ may not fit current context</span>' : ''}
      <span class="vsc-footnote">KV cache: ${s.kk}/${s.kv}</span>
    `;

    if (selectable) {
      const applyScenario = () => {
        // Picking a KV scenario card is an explicit KV choice; stop seeding from use case.
        wizardState.hardware.kvDtypeUserSet = true;
        wizardState.hardware.cacheTypeK = s.kk;
        wizardState.hardware.cacheTypeV = s.kv;
        // Keep current context (already validated to fit).
        wizardState.hardware.contextSize = ctx;

        if (dom.cacheTypeKSelect) dom.cacheTypeKSelect.value = s.kk;
        if (dom.cacheTypeVSelect) dom.cacheTypeVSelect.value = s.kv;
        if (dom.contextSizeInput) dom.contextSizeInput.value = ctx;

        card.querySelector('.vsc-ctx')?.classList.add('counting');
        setTimeout(() => card.querySelector('.vsc-ctx')?.classList.remove('counting'), 300);

        dom.vramScenarios.querySelectorAll('.vram-scenario-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        updateVramDisplay();
      };
      card.addEventListener('click', e => { e.stopPropagation(); applyScenario(); });
      card.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          applyScenario();
        }
      });
    }

    dom.vramScenarios.appendChild(card);
  }
}

// MLX context-fit cards (§2.6). With kv_cache_dtype pinned to int8 by the reasoning profile,
// TurboQuant force-Off, and PFlash steered off (§2.4/§2.5), KV quant is not a scenario axis on
// this engine — concurrency and retained-cache budget are. Each card asks for an estimate at the
// user's *current* context with different policy values, rather than varying context itself.
const MLX_SCENARIOS = [
  {
    key: 'single',
    mode: 'Single interactive user',
    detail: 'One conversation at a time, warm prompts reused',
    maxNumSeqs: 1,
    retainedCacheMib: 8192,
    rec: true,
  },
  {
    key: 'long-single',
    mode: 'Long single context',
    detail: 'Maximum room for one very long conversation; nothing retained between prompts',
    maxNumSeqs: 1,
    retainedCacheMib: 0,
    rec: false,
  },
  {
    key: 'shared',
    mode: 'Shared / multi-client',
    detail: 'Several clients at once; each admitted request reserves its own active KV',
    maxNumSeqs: 4,
    retainedCacheMib: 8192,
    rec: false,
  },
];

async function renderMlxScenarioCards(modelBytes, _arch, availVram, token) {
  const hw = wizardState.hardware;
  const currentCtx = hw.contextSize || 8192;
  const activeMaxNumSeqs = hw.parallelSlots || 1;
  const activeRetained = Number(hw.retainedCacheMib ?? 8192);

  const hasLocalPath = !!wizardState.model.path;
  const estimates = await Promise.all(
    MLX_SCENARIOS.map(async (s) => {
      try {
        const body = buildEstimateBody({
          backend: 'rapid_mlx',
          model_path: hasLocalPath ? wizardState.model.path : '',
          hf_repo_id: hasLocalPath ? null : (wizardState.model.hfRepo || wizardState.model.originRepo || null),
          hf_repo_revision: hasLocalPath ? null : (wizardState.model.hfRevision || 'main'),
          model_size_bytes: modelBytes,
          n_ctx: currentCtx,
          parallel_slots: s.maxNumSeqs,
          available_vram_bytes: availVram,
          is_unified_memory: isUnifiedMemory(),
          mmproj_path: wizardState.model.mmprojPath || null,
          mmproj_bytes: wizardState.arch.mmprojBytes || 0,
          kv_cache_dtype: 'int8',
          reasoning_mode: true,
          turboquant_mode: 'none',
          retained_cache_mib: s.retainedCacheMib,
          prefill_step_size: hw.prefillStepSize || 512,
        });
        const headers = (window.authHeaders ? window.authHeaders() : {});
        const res = await fetch('/api/vram-estimate', {
          method: 'POST',
          headers: { ...headers, 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) return null;
        const d = await res.json();
        if (!d.ok || d.total_bytes == null) return null;
        return d;
      } catch {
        return null;
      }
    }),
  );

  if (token !== scenarioCardsRenderToken) return;
  dom.vramScenarios.innerHTML = '';

  // Fixed facts, rendered once — not per-card, since none of them vary across these cards.
  const facts = document.createElement('div');
  facts.className = 'vram-mlx-fixed-facts';
  facts.textContent = 'KV: int8 (pinned by reasoning profile) · TurboQuant: off (awaiting receipt) · PFlash: off';
  dom.vramScenarios.appendChild(facts);

  const cardsWrap = document.createElement('div');
  cardsWrap.className = 'vram-mlx-scenario-cards';
  dom.vramScenarios.appendChild(cardsWrap);

  for (let i = 0; i < MLX_SCENARIOS.length; i++) {
    const s = MLX_SCENARIOS[i];
    const est = estimates[i];
    const headroom = est ? (est.headroom_bytes ?? 0) : 0;
    const rec = est ? (est.recommendation || 'risk') : 'risk';
    const fits = est && headroom >= 0;
    const isTight = rec === 'tight';
    const over = !est || !fits || rec === 'wont_fit';

    const isActive = s.maxNumSeqs === activeMaxNumSeqs && s.retainedCacheMib === activeRetained;
    const selectable = !over;

    const card = document.createElement('div');
    card.className = 'vram-scenario-card vram-mlx-scenario-card'
      + (s.rec ? ' scenario-rec' : '') + (isActive ? ' selected' : '');
    card.setAttribute('tabindex', '0');
    card.setAttribute('role', 'button');
    card.setAttribute('aria-label', `${s.mode}: ${s.detail}`);

    let desc = s.detail;
    if (over) desc = 'At your current context, this may not fit VRAM. Lower context or reduce concurrency.';
    else if (isTight) desc = `${s.detail} Fits, but leaves little headroom.`;

    // eslint-disable-next-line no-unsanitized/property
    card.innerHTML = `
      <div class="vsc-mode-name">${s.mode}</div>
      <div class="vsc-mode-detail">max_num_seqs: ${s.maxNumSeqs} · retained: ${s.retainedCacheMib > 0 ? formatGB(s.retainedCacheMib * 1024 * 1024) : '0'}</div>
      <div class="vsc-desc">${desc}</div>
      ${s.rec ? '<span class="vsc-rec-badge">★ Recommended</span>' : ''}
      ${isActive ? '<span class="vsc-active-badge">✓ Active</span>' : ''}
      ${over ? '<span class="vsc-warn">⚠ may not fit current context</span>' : ''}
    `;

    if (selectable) {
      const applyScenario = () => {
        wizardState.hardware.parallelSlots = s.maxNumSeqs;
        wizardState.hardware.retainedCacheMib = s.retainedCacheMib;
        if (dom.parallelSlotsInput) dom.parallelSlotsInput.value = s.maxNumSeqs;
        cardsWrap.querySelectorAll('.vram-mlx-scenario-card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        updateVramDisplay();
      };
      card.addEventListener('click', e => { e.stopPropagation(); applyScenario(); });
      card.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); applyScenario(); }
      });
    }

    cardsWrap.appendChild(card);
  }
}

function updateLegacyVramPill(total, avail) {
  if (dom.vramEstimateText) {
    dom.vramEstimateText.textContent = avail > 0
      ? `${formatGB(total)} / ${formatVramTotal(avail)}`
      : formatGB(total);
  }
  if (!dom.vramPill) return;
  const ratio = avail > 0 ? total / avail : 0;
  const map = [
    [0.82, 'fit', 'Fits'],
    [1.00, 'tight', 'Tight'],
    [1.20, 'risk', 'At risk'],
    [Infinity, 'wont-fit', "Won't fit"],
  ];
  let cls = '', lbl = '';
  for (const [thr, c, l] of map) { if (ratio <= thr) { cls = c; lbl = l; break; } }
  dom.vramPill.className = cls ? `vram-pill-${cls}` : '';
  dom.vramPill.textContent = lbl;
}

// ── Auto-size (server-side recommendation) ────────────────────────────────────

// ── Apple Silicon: apply Metal GPU wired limit via osascript ─────────────────

async function applyMetalGpuLimit(limitMb) {
  if (!dom.metalLimitBtn) return;
  const btn = dom.metalLimitBtn;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Applying…';

  try {
    const tokenResp = await fetch('/api/db/admin-token', {
      headers: window.authHeaders ? window.authHeaders() : {},
    });
    const tokenData = tokenResp.ok ? await tokenResp.json().catch(() => ({})) : {};
    const adminToken = tokenData.token;
    const headers = {
      'Content-Type': 'application/json',
      ...(adminToken ? { 'Authorization': `Bearer ${adminToken}` } : {}),
    };

    const resp = await fetch('/api/system/set-metal-gpu-limit', {
      method: 'POST',
      headers,
      body: JSON.stringify({ limit_mb: limitMb }),
    });
    const data = await resp.json();

    if (data.ok) {
      const gb = Math.round((data.limit_mb || limitMb) / 1024);
      showToast(`Metal GPU limit set to ${gb} GB — saved to /etc/sysctl.conf, survives reboots.`, 'success');
      await fetchGpuVram();
      await fetchMetalGpuLimit();
      scheduleVramUpdate();
    } else {
      const msg = data.error || 'Failed to apply Metal GPU limit.';
      if (msg.toLowerCase().includes('cancel')) {
        showToast('Cancelled — no changes made.', 'info');
      } else {
        // osascript failed — show the error and a Terminal fallback the user can copy
        const manualCmd = data.manual_cmd || `sudo /usr/sbin/sysctl -w iogpu.wired_limit_mb=${limitMb}`;
        _showMetalLimitFallback(btn, msg, manualCmd);
      }
      btn.disabled = false;
      btn.textContent = orig;
    }
  } catch (e) {
    showToast('Failed to contact server: ' + e.message, 'error');
    btn.disabled = false;
    btn.textContent = orig;
  }
}

function _showMetalLimitFallback(btn, errorMsg, manualCmd) {
  // Replace the button row with an inline error + copyable Terminal command
  const row = btn.closest('.metal-limit-row');
  if (!row) { showToast(errorMsg, 'error'); return; }

  const fallback = document.createElement('div');
  fallback.className = 'metal-limit-fallback';

  const errorDiv = document.createElement('div');
  errorDiv.className = 'metal-limit-fallback-error';
  errorDiv.textContent = errorMsg || '';

  const hintDiv = document.createElement('div');
  hintDiv.className = 'metal-limit-fallback-hint';
  hintDiv.textContent = 'Run this in Terminal instead:';

  const cmdRow = document.createElement('div');
  cmdRow.className = 'metal-limit-fallback-cmd-row';

  const codeEl = document.createElement('code');
  codeEl.className = 'metal-limit-fallback-cmd';
  codeEl.textContent = manualCmd || '';

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'metal-limit-fallback-copy';
  copyBtn.textContent = 'Copy';

  cmdRow.appendChild(codeEl);
  cmdRow.appendChild(copyBtn);

  fallback.appendChild(errorDiv);
  fallback.appendChild(hintDiv);
  fallback.appendChild(cmdRow);
  fallback.querySelector('.metal-limit-fallback-copy').addEventListener('click', () => {
    navigator.clipboard.writeText(manualCmd).then(() => {
      showToast('Copied to clipboard', 'success');
    });
  });

  // Append below the existing row content (don't remove the row itself)
  row.appendChild(fallback);
}
