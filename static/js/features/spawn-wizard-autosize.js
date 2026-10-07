// One-click VRAM auto-size: proposes context/KV/ubatch settings via
// /api/vram/auto-size, then clamps the result against the same
// /api/vram-estimate math the hardware step's live display uses so the two
// surfaces never disagree.
import {
  dom, wizardState,
  doIntrospect, effectiveAvailBytes, getModelBytes, getSizingArch,
  isUnifiedMemory, guessQuantFromName,
} from './spawn-wizard.js';
import { updateVramDisplay } from './spawn-wizard-vram-display.js';
import { refreshGuidedCards } from './spawn-wizard-guided.js';
import { updateCtxQuickPickActive, showCtxFitWarning } from './spawn-wizard-context-fit.js';
import { formatCtx } from './spawn-wizard-format.js';
import { buildEstimateBody, rapidEstimatePolicyFromWizardHardware } from './vram-estimate.js';
import { showToast } from './toast.js';

async function clampAutoSizeResultToSizingMath(result, _arch, modelBytes, availVram) {
  if (!result || !modelBytes || !availVram) return { result, adjusted: false };

  const modelCap = wizardState.model.nCtxTrain || 0;
  const r = { ...result };

  // First, cap to model's training context if known.
  if (modelCap > 0 && r.context_size > modelCap) {
    r.context_size = modelCap;
  }

  // Use /api/vram-estimate to validate the proposed config.
  // If it doesn't fit, reduce context_size until it does.
  const hw = wizardState.hardware;
  let adjusted = false;

  const tryEstimate = (ctx) => {
    // Builder item 6: canonical body builder for cross-surface equality.
    const body = buildEstimateBody({
      backend: wizardState.engine.selected || 'llama_cpp',
      model_path: wizardState.model.path || '',
      n_ctx: ctx,
      parallel_slots: hw.parallelSlots || 1,
      ubatch_size: r.ubatch_size || hw.ubatchSize || 2048,
      ctk: r.kv_quant_k || 'q8_0',
      ctv: r.kv_quant_v || 'q8_0',
      n_cpu_moe: r.n_cpu_moe ?? hw.nCpuMoe ?? 0,
      available_vram_bytes: availVram,
      is_unified_memory: isUnifiedMemory(),
      mmproj_path: wizardState.model.mmprojPath || null,
      mmproj_bytes: wizardState.arch.mmprojBytes || 0,
      ...(wizardState.engine.selected === 'rapid_mlx' ? rapidEstimatePolicyFromWizardHardware(hw) : {}),
    });

    return (async () => {
      try {
        const headers = (window.authHeaders ? window.authHeaders() : {});
        const res = await fetch('/api/vram-estimate', {
          method: 'POST',
          headers: { ...headers, 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) return null;
        const d = await res.json();
        if (!d.ok || !d.headroom_bytes) return null;
        return d;
      } catch {
        return null;
      }
    })();
  };

  // Initial check.
  let est = await tryEstimate(r.context_size);
  if (est && est.headroom_bytes >= 0) {
    // Already fits according to backend.
    return { result: r, adjusted: adjusted || (r.context_size !== result.context_size) };
  }

  // Reduce context_size until it fits (binary search).
  let lo = 1024, hi = r.context_size;
  let bestFit = 0;

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const e = await tryEstimate(mid);
    if (!e) { hi = mid - 1; continue; }
    if (e.headroom_bytes >= 0) {
      bestFit = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  if (bestFit > 0 && bestFit !== result.context_size) {
    r.context_size = bestFit;
    r.warnings = [
      ...(r.warnings || []),
      `Adjusted to ${formatCtx(bestFit)} so auto-size matches the hardware-step fit math.`,
    ];
    adjusted = true;
  }

  return { result: r, adjusted };
}

export async function triggerAutoSize() {
  if (!dom.vramAutosizeBtn) return;
  const btn = dom.vramAutosizeBtn;
  const origText = btn.textContent;
  btn.disabled = true; btn.textContent = 'Sizing…';
  if (dom.vramAutosizeNote) dom.vramAutosizeNote.textContent = '';

  try {
    const modelPath = wizardState.model.path || '';
    if (modelPath) {
      await doIntrospect(modelPath);
    }

    const availVram = effectiveAvailBytes();
    const modelBytes = getModelBytes();
    const arch = getSizingArch();

    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };

    const body = {
      model_path: wizardState.model.path || undefined,
      model_size_bytes: modelBytes || undefined,
      param_b: wizardState.model.paramB || undefined,
      model_name: wizardState.model.path || wizardState.model.hfRepo || '',
      available_vram_bytes: availVram,
      is_unified_memory: isUnifiedMemory(),
      use_case: wizardState.useCase,
      parallel_slots: wizardState.hardware.parallelSlots,
      fit_granularity: 1024,
      quant: guessQuantFromName(wizardState.model.hfFile || wizardState.model.path || ''),
      n_layers:              arch.nLayers    || undefined,
      n_kv_heads:            arch.nKvHeads   || undefined,
      head_dim:              arch.headDim    || undefined,
      global_head_dim:       arch.globalHeadDim || undefined,
      n_attn_layers:         arch.nAttnLayers || undefined,
      linear_attn_state_bytes: arch.linearAttnStateBytes || undefined,
      n_global_attn_layers:  arch.nGlobalAttnLayers || undefined,
      local_attn_window:     arch.localAttnWindow || undefined,
      local_kv_heads:        arch.localKvHeads || undefined,
      n_experts:             arch.nExperts   || undefined,
      n_experts_used:        arch.nExpertsUsed || undefined,
      expert_fraction:       arch.expertFraction || undefined,
      mtp_depth:             arch.mtpDepth   || undefined,
      mmproj_bytes:          arch.mmprojBytes || undefined,
      backend:               wizardState.engine.selected || 'llama_cpp',
    };

    const resp = await fetch('/api/vram/auto-size', { method: 'POST', headers, body: JSON.stringify(body) });
    if (!resp.ok) { showToast('Auto-size failed', 'error'); return; }
    const data = await resp.json();
    if (!data.ok || !data.result) { showToast('Auto-size: no result', 'warning'); return; }

    const { result: r, adjusted } = await clampAutoSizeResultToSizingMath(data.result, arch, modelBytes, availVram);

    // Apply recommended settings
    wizardState.hardware.contextSize = r.context_size;
    // A recommendation carries its own KV values; they outrank the use-case seed.
    wizardState.hardware.kvDtypeUserSet = true;
    wizardState.hardware.cacheTypeK  = r.kv_quant_k;
    wizardState.hardware.cacheTypeV  = r.kv_quant_v;
    wizardState.hardware.ubatchSize  = r.ubatch_size;

    if (r.n_cpu_moe != null) wizardState.hardware.nCpuMoe = r.n_cpu_moe;

    // Unified memory shares the model and host-cache budget. Auto disables only the
    // extra host prompt-state cache; ordinary common-prefix reuse remains available.
    // Preserve every explicit saved or user-entered value, including -1 (unlimited).
    if (isUnifiedMemory() && wizardState.hardware.cacheRam == null) {
      wizardState.hardware.cacheRam = 0;
      if (dom.cacheRamInput) dom.cacheRamInput.value = '0';
    }

    // Sync form fields
    if (dom.contextSizeInput) dom.contextSizeInput.value = r.context_size;
    if (dom.cacheTypeKSelect) dom.cacheTypeKSelect.value  = r.kv_quant_k;
    if (dom.cacheTypeVSelect) dom.cacheTypeVSelect.value  = r.kv_quant_v;
    if (dom.ubatchSizeInput)  dom.ubatchSizeInput.value   = r.ubatch_size;
    if (dom.fitTargetInput)   dom.fitTargetInput.value    = wizardState.hardware.fitTarget || '';
    if (r.n_cpu_moe != null && dom.nCpuMoeInput) dom.nCpuMoeInput.value = r.n_cpu_moe;
    if (r.n_cpu_moe != null && dom.moeOffloadSlider) dom.moeOffloadSlider.value = r.n_cpu_moe;
    refreshGuidedCards();

    const note = `${adjusted ? 'Adjusted:' : 'Set:'} ${formatCtx(r.context_size)} ctx · ${r.kv_quant_k.toUpperCase()} KV · ubatch ${r.ubatch_size}`;
    if (dom.vramAutosizeNote) dom.vramAutosizeNote.textContent = note;

    updateVramDisplay();
    updateCtxQuickPickActive();
    showCtxFitWarning(r.context_size, wizardState.useCase);

    if (r.warnings?.length) showToast('Auto-size warnings', 'warning', r.warnings[0]);
    else showToast('Auto-sized', 'success', note);
  } catch (err) {
    showToast('Auto-size error: ' + (err.message || String(err)), 'error');
  } finally {
    btn.disabled = false; btn.textContent = origText;
  }
}
