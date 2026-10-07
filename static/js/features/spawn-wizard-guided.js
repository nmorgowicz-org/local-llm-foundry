// spawn-wizard-guided.js — Option A (Guided) decision cards wiring
import { wizardState } from './spawn-wizard.js';
import { _mtpUserConfigured } from './spawn-wizard-mtp-draft.js';

// Wire up decision cards on step entry
export function initGuidedCards() {
  wireContextTiles();
  wireKvTiles();
  wireVisionCard();
  wireSpeedBoost();
  refreshGuidedCapabilityCards();
  const overlay = document.getElementById('spawn-wizard-overlay');
  overlay?.addEventListener('input', refreshGuidedCards);
  overlay?.addEventListener('change', refreshGuidedCards);
  // Direct listeners also cover existing callers that dispatch a non-bubbling
  // input/change event on a canonical control.
  ['spawn-context-size', 'spawn-cache-type-k', 'spawn-cache-type-v', 'hw-mmproj-select', 'hw-use-mtp']
    .forEach(id => {
      const control = document.getElementById(id);
      control?.addEventListener('input', refreshGuidedCards);
      control?.addEventListener('change', refreshGuidedCards);
    });
  refreshGuidedCards();
}

// Keep the primary speed decision truthful while metadata is loading. The
// initial state is deliberately unavailable; only resolved model metadata can
// advertise built-in MTP heads. External draft-model and n-gram paths remain
// separate choices and must not be presented as MTP capability.
export function refreshGuidedCapabilityCards() {
  const onRadio = document.querySelector('input[name="hw-speed"][value="on"]');
  const offRadio = document.querySelector('input[name="hw-speed"][value="off"]');
  const label = onRadio?.closest('.hw-speed-option');
  const text = label?.querySelector('.hw-speed-label');
  if (!onRadio || !label || !text) return;

  const status = wizardState.arch?.metadataStatus || 'unknown';
  const mtpDepth = Number(wizardState.arch?.mtpDepth || 0);
  const resolved = status === 'resolved';
  const eligible = resolved && mtpDepth > 0;

  if (eligible) {
    text.textContent = `On — MTP heads detected (${mtpDepth})`;
    label.title = 'Model-native MTP heads are available.';
  } else if (resolved) {
    text.textContent = 'On — unavailable for this model';
    label.title = 'This model has no introspected built-in MTP heads.';
  } else if (status === 'degraded') {
    text.textContent = 'On — unavailable (metadata unresolved)';
    label.title = 'Resolve model metadata before enabling built-in MTP.';
  } else {
    text.textContent = 'On — checking model capability';
    label.title = 'Waiting for model-native metadata.';
  }

  onRadio.disabled = !eligible;
  onRadio.setAttribute('aria-disabled', String(!eligible));
  const ngramRadio = document.querySelector('input[name="hw-speed"][value="ngram"]');
  const speedConfigured = [...document.querySelectorAll('input[name="hw-speed"]')]
    .some(radio => radio.dataset.userConfigured);
  // Pro and Guided share the checkbox's existing authority flag. Metadata
  // rediscovery may refresh availability, but must not re-answer a user choice
  // or an explicit canonical speculative-mode selection.
  const explicitSpec = document.getElementById('spawn-spec-type')?.value || wizardState.hardware?.specType;
  const canApplyDefault = !_mtpUserConfigured && !speedConfigured && !explicitSpec;
  if (!eligible && canApplyDefault) {
    onRadio.checked = false;
    if (offRadio && !ngramRadio?.checked) offRadio.checked = true;
    if (wizardState.hardware) wizardState.hardware.mtpEnabled = false;
    const checkbox = document.getElementById('hw-use-mtp');
    if (checkbox) checkbox.checked = false;
  } else if (eligible && canApplyDefault) {
    onRadio.checked = true;
    if (wizardState.hardware) wizardState.hardware.mtpEnabled = true;
    const checkbox = document.getElementById('hw-use-mtp');
    if (checkbox) checkbox.checked = true;
  }
  refreshGuidedCards();
}

// Card 1: Context size tiles sync with #spawn-context-size
function wireContextTiles() {
  const tiles = document.querySelectorAll('#hw-ctx-tiles .hw-decision-tile');
  const customInput = document.getElementById('hw-ctx-custom');
  const origInput = document.getElementById('spawn-context-size');

  function setTileActive(ctx) {
    tiles.forEach(t => {
      t.classList.toggle('hw-decision-tile-active', t.dataset.ctx === ctx);
    });
  }

  tiles.forEach(tile => {
    tile.addEventListener('click', () => {
      const ctx = tile.dataset.ctx;
      if (customInput) customInput.value = ctx;
      if (origInput) origInput.value = ctx;
      setTileActive(ctx);
      // Trigger input event for existing handlers
      origInput?.dispatchEvent(new Event('input', { bubbles: true }));
    });
  });

  // Guided edits go through the real canonical field and its existing
  // handlers; both live typing and a committed change must update launch state.
  ['input', 'change'].forEach(type => {
    customInput?.addEventListener(type, () => {
      if (!origInput) return;
      origInput.value = customInput.value;
      origInput.dispatchEvent(new Event(type, { bubbles: true }));
    });
  });
}

// Card 2: KV precision tiles sync with #spawn-cache-type-k/v
function wireKvTiles() {
  const tiles = document.querySelectorAll('#hw-kv-tiles .hw-decision-tile');
  const kInput = document.getElementById('spawn-cache-type-k');
  const vInput = document.getElementById('spawn-cache-type-v');

  function setTileActive(kv) {
    tiles.forEach(t => {
      t.classList.toggle('hw-decision-tile-active', t.dataset.kv === kv);
    });
  }

  tiles.forEach(tile => {
    tile.addEventListener('click', () => {
      const kv = tile.dataset.kv;
      if (kInput) kInput.value = kv;
      if (vInput) vInput.value = kv;
      setTileActive(kv);
      kInput?.dispatchEvent(new Event('change', { bubbles: true }));
      vInput?.dispatchEvent(new Event('change', { bubbles: true }));
    });
  });

  // Canonical input/change events refresh both K and V through
  // refreshGuidedCards, including asymmetric cache configurations.
}

// Card 3: Vision select sync with existing mmproj controls
function wireVisionCard() {
  const visionSelect = document.getElementById('hw-vision-select');
  const origSelect = document.getElementById('hw-mmproj-select');

  if (visionSelect && origSelect) {
    // Populate vision select from original select options
    function syncOptions() {
      const options = Array.from(origSelect.options, option => option.cloneNode(true));
      if (!options.length) {
        const fallback = document.createElement('option');
        fallback.value = '';
        fallback.textContent = '( none — text only )';
        options.push(fallback);
      }
      visionSelect.replaceChildren(...options);
      visionSelect.value = origSelect.value;
    }
    syncOptions();
    // mmproj discovery populates the original select asynchronously (local
    // scan, mradermacher/HF lookup, companion download) — re-sync whenever
    // its option list changes so Guided never shows a stale "(none)".
    new MutationObserver(syncOptions).observe(origSelect, { childList: true });

    // Keep in sync on change
    visionSelect.addEventListener('change', () => {
      origSelect.value = visionSelect.value;
      origSelect.dispatchEvent(new Event('change', { bubbles: true }));
    });
    origSelect.addEventListener('change', () => {
      visionSelect.value = origSelect.value;
    });
  }
}

// Card 4: Speed boost radios sync with MTP controls
function wireSpeedBoost() {
  const radios = document.querySelectorAll('input[name="hw-speed"]');
  const mtpCheckbox = document.getElementById('hw-use-mtp');

  radios.forEach(radio => {
    radio.addEventListener('change', () => {
      radio.dataset.userConfigured = '1';
      const value = radio.value;
      if (mtpCheckbox) {
        mtpCheckbox.checked = value === 'on';
        mtpCheckbox.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  });

}

// Read-only mirror for explicit canonical writers (autosize, scenarios,
// presets, capability/discovery renders) and real input/change events. DOM
// properties are not HTML attributes: observing "value" cannot track them.
export function refreshGuidedCards(event) {
  // A radio emits input before its change handler updates the canonical
  // checkbox. Do not overwrite the new selection with that old checkbox.
  if (event?.type === 'input' && event.target?.name === 'hw-speed') return;
  const ctx = document.getElementById('spawn-context-size')?.value;
  if (ctx != null) {
    const customInput = document.getElementById('hw-ctx-custom');
    if (customInput && customInput.value !== ctx) customInput.value = ctx;
    document.querySelectorAll('#hw-ctx-tiles .hw-decision-tile').forEach(tile => {
      tile.classList.toggle('hw-decision-tile-active', tile.dataset.ctx === ctx);
    });
  }

  const k = document.getElementById('spawn-cache-type-k')?.value || '';
  const v = document.getElementById('spawn-cache-type-v')?.value || '';
  document.querySelectorAll('#hw-kv-tiles .hw-decision-tile').forEach(tile => {
    tile.classList.toggle('hw-decision-tile-active', !!k && k === v && tile.dataset.kv === k);
  });

  // Same input-before-change ordering as the speed radios: the Guided select
  // emits `input` (bubbling to the overlay) before its `change` handler writes
  // the canonical select. Mirroring now would revert the user's pick to "(none)".
  const vision = document.getElementById('hw-vision-select');
  const canonicalVision = document.getElementById('hw-mmproj-select');
  const userIsPickingVision = event?.type === 'input' && event.target === vision;
  if (vision && canonicalVision && !userIsPickingVision) vision.value = canonicalVision.value;
  const mtp = document.getElementById('hw-use-mtp');
  const ngram = document.querySelector('input[name="hw-speed"][value="ngram"]');
  if (mtp && (mtp.checked || !ngram?.checked)) {
    const selected = mtp.checked ? 'on' : 'off';
    document.querySelectorAll('input[name="hw-speed"]').forEach(radio => {
      radio.checked = radio.value === selected;
    });
  }

  const setText = (id, text) => {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
  };
  const modelName = wizardState.model.name || wizardState.model.path || '—';
  setText('hw-sticky-model-name', modelName.split('/').pop() || modelName);
  setText('hw-sticky-quant', wizardState.model.quant || '—');
  setText('hw-sticky-loader', wizardState.engine?.selected === 'rapid_mlx' ? 'Rapid-MLX' : 'llama.cpp');
  setText('hw-sticky-usecase', wizardState.useCase || 'General');
  const context = Number(ctx) || 0;
  setText('hw-sticky-ctx', context > 0 ? `ctx ${context >= 1000 ? `${(context / 1000).toFixed(0)}k` : context}` : 'ctx —');
  setText('hw-sticky-kv', k ? `KV ${k === v ? k : `${k}/${v || '—'}`}` : 'KV —');
}
