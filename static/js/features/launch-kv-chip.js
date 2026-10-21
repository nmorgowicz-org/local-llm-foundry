// launch-kv-chip.js — a small status chip on each llama.cpp launch card.
//
// Quiet when the pair is expected to use the fused path, and for an equal pair
// that is merely unverified: a chip only appears when there is something to say. The tooltip carries the same server-written
// reason the editor and wizard show. Chips never gate Start; a known refusal is
// enforced where Start happens.

import { sessionState } from '../core/app-state.js';
import { fetchKvSupport, modelIdForPath } from './kv-support.js';

const CHIP = {
    fallback: { icon: '⚠', label: 'K/V slower' },
    unverified: { icon: '?', label: 'K/V unverified' },
    wont_start: { icon: '✖', label: "K/V won't start" },
};

function chipFor(card) {
    return card.querySelector('.launch-chip--kv');
}

async function hydrateCard(card, preset) {
    const chips = card.querySelector('.launch-card-chips');
    if (!chips) return;
    chipFor(card)?.remove();
    if (preset.backend === 'rapid_mlx') return;
    if (!preset.ctk || !preset.ctv) return; // empty means the server default; nothing to classify
    const modelId = await modelIdForPath(preset.model_path || '');
    const result = await fetchKvSupport({
        k: preset.ctk,
        v: preset.ctv,
        fa: preset.flash_attn || '',
        modelId,
    });
    // The grid may have re-rendered while the request was in flight.
    if (!card.isConnected || chipFor(card)) return;
    const kind = result?.support?.kind;
    const spec = CHIP[kind];
    if (!spec) return;
    // An equal pair that merely lacks device/model evidence is the common case
    // on every card (and on any machine with no runtime yet); a "?" there is
    // noise. Unverified is only worth a chip for a mixed pair.
    if (kind === 'unverified' && String(preset.ctk).toLowerCase() === String(preset.ctv).toLowerCase()) return;
    const chip = document.createElement('span');
    chip.className = `launch-chip launch-chip--kv launch-chip--kv-${kind}`;
    chip.dataset.kvState = kind;
    chip.textContent = `${spec.icon} ${spec.label}`;
    chip.title = [result.reason, result.fix].filter(Boolean).join(' ');
    chips.append(chip);
}

export function hydrateKvChips() {
    const presets = sessionState.presets || [];
    document.querySelectorAll('.launch-card[data-preset-id]').forEach((card) => {
        const preset = presets.find((p) => p.id === card.dataset.presetId);
        if (preset) void hydrateCard(card, preset);
    });
}
