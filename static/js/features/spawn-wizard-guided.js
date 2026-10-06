// spawn-wizard-guided.js — Option A (Guided) decision cards wiring
import { wizardState } from './spawn-wizard.js';

// Wire up decision cards on step entry
export function initGuidedCards() {
  wireContextTiles();
  wireKvTiles();
  wireVisionCard();
  wireTemplateCard();
  wireSpeedBoost();
  wireStickyBar();
  refreshGuidedCapabilityCards();
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
  if (!eligible && !onRadio.dataset.userConfigured) {
    onRadio.checked = false;
    if (offRadio) offRadio.checked = true;
    if (wizardState.hardware) wizardState.hardware.mtpEnabled = false;
  } else if (eligible && !onRadio.dataset.userConfigured && !offRadio?.dataset.userConfigured) {
    onRadio.checked = true;
    if (wizardState.hardware) wizardState.hardware.mtpEnabled = true;
  }
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
      customInput?.dispatchEvent(new Event('input', { bubbles: true }));
      origInput?.dispatchEvent(new Event('input', { bubbles: true }));
    });
  });

  // Custom input updates tiles
  if (customInput) {
    customInput.addEventListener('change', () => {
      const ctx = customInput.value;
      setTileActive(ctx);
    });
  }

  // Sync from original input (when set by other code). Both events are
  // observed because several code paths set .value and dispatch only one
  // of them; programmatic writes without an event are covered by the
  // MutationObserver in syncGuidedCards().
  function syncFromOrig() {
    const ctx = origInput?.value;
    if (ctx == null) return;
    if (customInput && customInput.value !== ctx) customInput.value = ctx;
    setTileActive(ctx);
  }
  origInput?.addEventListener('input', syncFromOrig);
  origInput?.addEventListener('change', syncFromOrig);
  syncFromOrig();
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

  // Sync from original inputs
  function syncFromOrig() {
    const kv = kInput?.value || vInput?.value;
    if (kv) setTileActive(kv);
  }
  kInput?.addEventListener('change', syncFromOrig);
  vInput?.addEventListener('change', syncFromOrig);
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

  // Sync from MTP checkbox
  if (mtpCheckbox) {
    mtpCheckbox.addEventListener('change', () => {
      const checked = mtpCheckbox.checked;
      const radio = document.querySelector(`input[name="hw-speed"][value="${checked ? 'on' : 'off'}"]`);
      if (radio) radio.checked = true;
    });
  }
}

// Card 5: host the shared chat-template panel inside the Guided card. The
// panel keeps its ids (chat-template-section etc.) so the existing
// chat-template logic drives it unchanged; Pro moves it back inline.
function wireTemplateCard() {
  const host = document.getElementById('ct-card-host');
  const section = document.getElementById('chat-template-section');
  if (!host || !section) return;
  if (!host.contains(section)) host.appendChild(section);
}

export function restoreTemplateSection() {
  const section = document.getElementById('chat-template-section');
  const anchor = document.getElementById('rapid-mlx-profile-hints');
  if (section && anchor && section.parentElement !== anchor.parentElement) {
    anchor.parentElement.insertBefore(section, anchor);
  }
}

// Sticky context bar: populate from wizardState
function wireStickyBar() {
  function updateStickyBar() {
    const modelEl = document.getElementById('hw-sticky-model-name');
    const quantEl = document.getElementById('hw-sticky-quant');
    const loaderEl = document.getElementById('hw-sticky-loader');
    const usecaseEl = document.getElementById('hw-sticky-usecase');
    const ctxEl = document.getElementById('hw-sticky-ctx');
    const kvEl = document.getElementById('hw-sticky-kv');

    if (modelEl) {
      const modelName = wizardState.model.name || wizardState.model.path || '—';
      modelEl.textContent = modelName.split('/').pop() || modelName;
    }
    if (quantEl) {
      quantEl.textContent = wizardState.model.quant || '—';
    }
    if (loaderEl) {
      loaderEl.textContent = wizardState.engine?.selected === 'rapid_mlx' ? 'Rapid-MLX' : 'llama.cpp';
    }
    if (usecaseEl) {
      const useCaseLabel = wizardState.useCase || 'General';
      usecaseEl.textContent = useCaseLabel;
    }
    if (ctxEl) {
      const ctx = wizardState.hardware?.contextSize || 0;
      if (ctx > 0) {
        const display = ctx >= 1000 ? `${(ctx / 1000).toFixed(0)}k` : ctx;
        ctxEl.textContent = `ctx ${display}`;
      }
    }
    if (kvEl) {
      const kCache = document.getElementById('spawn-cache-type-k');
      if (kCache && kCache.value) {
        kvEl.textContent = `KV ${kCache.value}`;
      }
    }
  }

  // Initial update and on state changes. The same observer keeps the
  // decision cards in sync with programmatic writes to the canonical
  // controls (context size, KV precision, vision, MTP) — a single source
  // of truth with multiple writers.
  function syncGuidedCards() {
    const ctx = document.getElementById('spawn-context-size')?.value;
    if (ctx != null) {
      const customInput = document.getElementById('hw-ctx-custom');
      if (customInput && customInput.value !== ctx) customInput.value = ctx;
      document.querySelectorAll('#hw-ctx-tiles .hw-decision-tile').forEach(t => {
        t.classList.toggle('hw-decision-tile-active', t.dataset.ctx === ctx);
      });
    }
    const kv = document.getElementById('spawn-cache-type-k')?.value;
    if (kv) {
      document.querySelectorAll('#hw-kv-tiles .hw-decision-tile').forEach(t => {
        t.classList.toggle('hw-decision-tile-active', t.dataset.kv === kv);
      });
    }
    const visionSelect = document.getElementById('hw-vision-select');
    const origVision = document.getElementById('hw-mmproj-select');
    if (visionSelect && origVision && visionSelect.value !== origVision.value) {
      visionSelect.value = origVision.value;
    }
    const mtp = document.getElementById('hw-use-mtp');
    if (mtp) {
      const radio = document.querySelector(`input[name="hw-speed"][value="${mtp.checked ? 'on' : 'off'}"]`);
      if (radio && !radio.checked) radio.checked = true;
    }
  }

  updateStickyBar();
  syncGuidedCards();
  new MutationObserver(() => { updateStickyBar(); syncGuidedCards(); }).observe(document.body, {
    subtree: true,
    attributes: true,
    attributeFilter: ['value']
  });
}
