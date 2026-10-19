// spawn-wizard-kv.js — K/V support line for the setup wizard (Guided + Pro).
//
// Guided and Pro share one surface: the "KV cache precision" decision card
// (#hw-decision-kv), which Pro relocates into its Memory & context category.
// The line (#hw-kv-support) lives in that card, so both profiles show the same
// server-written answer by construction. It reads the canonical selects
// (#spawn-cache-type-k/v, #spawn-flash-attn, #spawn-model-path); the old inline
// K/V row is hidden in both profiles and carries no line.

import { wizardState } from './spawn-wizard.js';
import {
  blocksStart,
  createKvSupportController,
  fetchKvSupport,
  modelIdForPath,
} from './kv-support.js';

const MIXED = { k: 'q8_0', v: 'q4_0' };

const MIXED_SUBLABEL = {
  native: 'fused path',
  fallback: 'slower path',
  unverified: 'not verified',
  wont_start: 'needs flash attention',
};

let controllers = [];
let lastResult = null;

function value(id) {
  return document.getElementById(id)?.value || '';
}

function isLlamaCpp() {
  return (wizardState.engine?.selected || 'llama_cpp') === 'llama_cpp';
}

async function currentContext() {
  if (!isLlamaCpp()) return null;
  const k = value('spawn-cache-type-k');
  const v = value('spawn-cache-type-v');
  if (!k || !v) return null;
  return {
    k,
    v,
    fa: value('spawn-flash-attn'),
    modelId: await modelIdForPath(value('spawn-model-path')),
  };
}

/** Disable Start while the pair is a known refusal; Save is never gated here. */
export function applyKvLaunchGate() {
  const button = document.getElementById('spawn-server-btn');
  if (!button) return;
  const blocked = isLlamaCpp() && blocksStart(lastResult);
  button.dataset.kvBlocked = blocked ? 'true' : 'false';
  if (blocked) {
    button.disabled = true;
    button.title = [lastResult.reason, lastResult.fix].filter(Boolean).join(' ');
  } else if (button.dataset.kvBlockedTitle === 'true') {
    button.removeAttribute('title');
  }
  button.dataset.kvBlockedTitle = blocked ? 'true' : 'false';
}

export function wizardKvBlockReason() {
  return isLlamaCpp() && blocksStart(lastResult)
    ? [lastResult.reason, lastResult.fix].filter(Boolean).join(' ')
    : '';
}

async function refreshMixedTile(ctx) {
  const tile = document.querySelector('#hw-kv-tiles [data-kv-mixed]');
  if (!tile) return;
  const sub = tile.querySelector('.hw-tile-tag');
  if (!ctx) {
    tile.hidden = true;
    return;
  }
  const result = await fetchKvSupport({ ...ctx, k: MIXED.k, v: MIXED.v });
  // Context moved on while waiting: the next refresh owns the tile.
  const now = await currentContext();
  if (!now || now.fa !== ctx.fa || now.modelId !== ctx.modelId) return;
  const active = value('spawn-cache-type-k') === MIXED.k && value('spawn-cache-type-v') === MIXED.v;
  // Metal has no mixed fused kernel and rebuilding does not add one, so the tile
  // is not offered there (an already-selected mixed pair stays visible).
  tile.hidden = result?.backend === 'metal' && !active;
  if (sub) sub.textContent = result ? MIXED_SUBLABEL[result.support.kind] : 'model-dependent';
  tile.dataset.kvMixedState = result?.support?.kind || 'unknown';
}

export function refreshWizardKvSupport() {
  controllers.forEach((controller) => controller.update());
}

export function initWizardKvSupport() {
  const container = document.getElementById('hw-kv-support');
  if (!container) return;
  controllers = [
    createKvSupportController({
      container,
      getContext: currentContext,
      onResult: async (result, ctx) => {
        lastResult = result;
        applyKvLaunchGate();
        await refreshMixedTile(ctx);
      },
    }),
  ];
  const watch = ['spawn-cache-type-k', 'spawn-cache-type-v', 'spawn-flash-attn', 'spawn-model-path'];
  for (const id of watch) {
    const control = document.getElementById(id);
    control?.addEventListener('change', refreshWizardKvSupport);
    control?.addEventListener('input', refreshWizardKvSupport);
  }
  refreshWizardKvSupport();
}
