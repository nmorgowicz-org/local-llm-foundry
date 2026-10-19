// preset-editor-kv.js — K/V support line for the preset editor.
//
// Same shared client and the same server-written text as the wizard. Plain Save
// is never gated (presets are portable across machines); only Start paths are.

import { createKvSupportController, modelIdForPath } from './kv-support.js';

const WATCHED = ['modal-ctk', 'modal-ctv', 'modal-flash-attn', 'modal-model-path'];

let controller = null;

function value(id) {
    return document.getElementById(id)?.value || '';
}

async function currentContext() {
    const container = document.getElementById('modal-kv-support');
    // Rapid-MLX presets have no llama.cpp cache pair; the editor marks them with
    // the format pill. (Visibility is not a signal: the cache controls live in a
    // section tab that is usually not the open one.)
    const mlxPill = document.getElementById('preset-editor-format');
    if (!container || (mlxPill && !mlxPill.hidden)) return null;
    const k = value('modal-ctk');
    const v = value('modal-ctv');
    if (!k || !v) return null;
    return {
        k,
        v,
        fa: value('modal-flash-attn'),
        modelId: await modelIdForPath(value('modal-model-path').trim()),
    };
}

export function refreshPresetEditorKv() {
    const container = document.getElementById('modal-kv-support');
    if (!container) return;
    if (!controller) {
        controller = createKvSupportController({ container, getContext: currentContext });
        for (const id of WATCHED) {
            const control = document.getElementById(id);
            control?.addEventListener('change', () => controller.update());
            control?.addEventListener('input', () => controller.update());
        }
    }
    controller.update();
}
