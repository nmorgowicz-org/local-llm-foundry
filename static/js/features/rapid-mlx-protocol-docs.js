// In-app Model protocol reference for Rapid-MLX (spawn wizard + preset editor).
//
// Auto-detection fills the tool-call parser / reasoning parser / hybrid fields from the
// runtime's live model profile, but users launching a modified finetune — or on a machine
// where the profile is unavailable — need a way to check the documented protocol without
// leaving the app. This modal mirrors the RapidMLX model-families documentation for the
// common families and links to the live page for everything else.
//
// Grounding: https://rapidmlx.com/docs/models/families/qwen documents Qwen 3.8 as
// tool parser `qwen3_coder_xml`, reasoning parser `qwen3`, hybrid mode flag.

const FAMILIES_DOCS_URL = 'https://rapidmlx.com/docs/models/families';

// Curated from upstream's own alias registry
// (raullenchai/Rapid-MLX rapid_mlx/aliases.json — the same table `rapid-mlx serve`
// reads) and the RapidMLX families docs. `match` runs against the lowercased model
// id/repo/path; entries list the series lines so the exact parser pair is visible.
const PROTOCOL_FAMILIES = [
    {
        id: 'qwen',
        match: /\bqwen/,
        anchorFor: (modelId) => {
            const minor = modelId.match(/qwen\s*3\.(\d+)/);
            return minor ? `#qwen-3.${minor[1]}` : '#qwen';
        },
        entries: [
            { series: 'Qwen 3.8 (e.g. 3.8-27B)', tool: 'qwen3_coder_xml', reasoning: 'qwen3', flags: 'hybrid' },
            { series: 'Qwen 3.6 / 3.5 dense + MoE', tool: 'hermes', reasoning: 'qwen3', flags: 'hybrid on the MoE tiers' },
            { series: 'Qwen3 Coder', tool: 'qwen3_coder_xml', reasoning: 'qwen3', flags: '' },
        ],
        note: 'Qwen 3.8 is hybrid; 3.6/3.5 MoE tiers are hybrid, dense tiers are not.',
    },
    {
        id: 'gemma',
        match: /\bgemma/,
        anchorFor: (modelId) => {
            const major = modelId.match(/gemma\s*(\d+)/);
            return major ? `#gemma-${major[1]}` : '#gemma';
        },
        entries: [
            { series: 'Gemma 4 (incl. QAT)', tool: 'gemma4', reasoning: 'gemma4', flags: '' },
        ],
        note: '',
    },
    {
        id: 'llama',
        match: /\bllama/,
        anchorFor: () => '#llama',
        entries: [
            { series: 'Llama 3.1 / 3.2 (1B / 3B / 8B)', tool: 'llama', reasoning: '—', flags: '' },
        ],
        note: '',
    },
    {
        id: 'deepseek',
        match: /\bdeepseek/,
        anchorFor: () => '#deepseek',
        entries: [
            { series: 'DeepSeek V4 / V4-Flash', tool: 'deepseek', reasoning: 'deepseek_v4', flags: '' },
            { series: 'V4-Flash 0731 checkpoint', tool: 'deepseek_v4_0731', reasoning: 'deepseek_v4', flags: '' },
            { series: 'R1 reasoning-distilled', tool: 'deepseek_v3', reasoning: 'deepseek_r1', flags: '' },
        ],
        note: '',
    },
    {
        id: 'glm',
        match: /\bglm/,
        anchorFor: () => '#glm',
        entries: [
            { series: 'GLM-4.7 / 5.x', tool: 'glm47', reasoning: 'glm4 or glm5', flags: 'hybrid on the 5.x MoE tier' },
        ],
        note: '',
    },
    {
        id: 'mistral',
        match: /\bmistral|\bdevstral|\bministral/,
        anchorFor: () => '#mistral',
        entries: [
            { series: 'Mistral / Ministral / Devstral', tool: 'mistral', reasoning: '—', flags: '' },
        ],
        note: '',
    },
    {
        id: 'gpt-oss',
        match: /\bgpt[\s_-]?oss/,
        anchorFor: () => '#gpt-oss',
        entries: [
            { series: 'GPT-OSS 20B / 120B (MXFP4)', tool: 'harmony', reasoning: 'harmony', flags: '' },
        ],
        note: '',
    },
    {
        id: 'minimax',
        match: /\bminimax/,
        anchorFor: () => '#minimax',
        entries: [
            { series: 'MiniMax M2.5 / M2.7', tool: 'minimax', reasoning: 'minimax', flags: '' },
        ],
        note: '',
    },
    {
        id: 'hunyuan',
        match: /\bhunyuan/,
        anchorFor: () => '#hunyuan',
        entries: [
            { series: 'Hunyuan Hy3 295B MoE', tool: 'hy_v3', reasoning: 'hy_v3', flags: '' },
        ],
        note: '',
    },
    {
        id: 'granite',
        match: /\bgranite/,
        anchorFor: () => '#granite',
        entries: [
            { series: 'Granite 4.x', tool: 'qwen3_coder_xml', reasoning: 'qwen3', flags: '' },
            { series: 'Granite 4 H-Micro / Tiny (SSM hybrid)', tool: 'hermes', reasoning: '—', flags: 'hybrid' },
        ],
        note: '',
    },
    {
        id: 'muse',
        match: /\bmuse\b|\bglimmer/,
        anchorFor: () => '#muse',
        entries: [
            { series: 'Muse Glimmer 30B', tool: 'muse', reasoning: 'muse', flags: '' },
        ],
        note: '',
    },
    {
        id: 'phi',
        match: /\bphi[\s_-]?[34]/,
        anchorFor: () => '#phi',
        entries: [
            { series: 'Phi-3.5 / Phi-4 line', tool: 'hermes', reasoning: '— or deepseek_r1 on mini-reasoning', flags: '' },
        ],
        note: '',
    },
    {
        id: 'liquid',
        match: /\blfm|\bliquid/,
        anchorFor: () => '#liquid',
        entries: [
            { series: 'LiquidAI LFM2 / LFM2.5', tool: 'lfm', reasoning: 'qwen3 on some tiers', flags: 'hybrid on LFM2.5-24B-A2B' },
        ],
        note: '',
    },
    {
        id: 'tmax',
        match: /\btmax/,
        anchorFor: () => '/tmax',
        entries: [
            { series: 'Tmax-9B / 27B agent (Gated-DeltaNet)', tool: 'qwen3_xml', reasoning: '—', flags: 'hybrid' },
        ],
        note: '',
    },
    {
        id: 'ui-tars',
        match: /\bui[\s_-]?tars/,
        anchorFor: () => '/ui-tars',
        entries: [
            { series: 'UI-TARS (vision computer-use)', tool: 'ui_tars', reasoning: 'ui_tars', flags: '' },
        ],
        note: '',
    },
];

const FALLBACK_FAMILY = {
    id: null,
    anchorFor: () => '',
    entries: [],
    note: 'This family is not in the built-in table — check the live RapidMLX docs page.',
};

function inferModelId() {
    // Wizard first, then the preset editor's model field — the modal is shared.
    const wizRepo = document.getElementById('spawn-hf-repo');
    const wizPath = document.getElementById('spawn-model-path');
    const peModel = document.getElementById('modal-model-path') || document.getElementById('modal-rapid-model');
    for (const el of [wizRepo, wizPath, peModel]) {
        const value = el && el.value ? el.value.trim() : '';
        if (value) return value;
    }
    return '';
}

function detectFamily(modelId) {
    const lower = (modelId || '').toLowerCase();
    return PROTOCOL_FAMILIES.find(f => f.match.test(lower)) || FALLBACK_FAMILY;
}

function buildModal(family, modelId) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay open protocol-docs-overlay';
    overlay.id = 'protocol-docs-overlay';

    const modal = document.createElement('div');
    modal.className = 'modal-content protocol-docs-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Model protocol reference');

    const title = document.createElement('h3');
    title.textContent = 'Model protocol reference';
    modal.appendChild(title);

    const subtitle = document.createElement('p');
    subtitle.className = 'protocol-docs-subtitle';
    subtitle.textContent = modelId
        ? `Detected family for “${modelId}”: ${family.id || 'unknown'}`
        : 'No model selected yet — showing the general reference.';
    modal.appendChild(subtitle);

    if (family.entries.length) {
        const table = document.createElement('table');
        table.className = 'protocol-docs-table';
        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        for (const label of ['Series', 'Tool parser', 'Reasoning parser', 'Flags']) {
            const th = document.createElement('th');
            th.textContent = label;
            headRow.appendChild(th);
        }
        thead.appendChild(headRow);
        table.appendChild(thead);
        const tbody = document.createElement('tbody');
        for (const entry of family.entries) {
            const tr = document.createElement('tr');
            for (const value of [entry.series, entry.tool, entry.reasoning, entry.flags]) {
                const td = document.createElement('td');
                td.textContent = value;
                tr.appendChild(td);
            }
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        modal.appendChild(table);
    }

    if (family.note) {
        const note = document.createElement('p');
        note.className = 'protocol-docs-note';
        note.textContent = family.note;
        modal.appendChild(note);
    }

    const link = document.createElement('a');
    link.className = 'protocol-docs-link';
    link.href = family.id ? `${FAMILIES_DOCS_URL}/${family.id}${family.anchorFor(modelId.toLowerCase())}` : FAMILIES_DOCS_URL;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'Open the full RapidMLX families docs ↗';
    modal.appendChild(link);

    const actions = document.createElement('div');
    actions.className = 'protocol-docs-actions';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'btn-wizard-secondary';
    close.textContent = 'Close';
    close.addEventListener('click', () => overlay.remove());
    actions.appendChild(close);
    modal.appendChild(actions);

    overlay.appendChild(modal);
    // Clicking the backdrop closes; clicking inside does not.
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    document.addEventListener('keydown', function onEsc(e) {
        if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', onEsc); }
    });
    return overlay;
}

export function openProtocolDocsModal() {
    const existing = document.getElementById('protocol-docs-overlay');
    if (existing) existing.remove();
    const modelId = inferModelId();
    document.body.appendChild(buildModal(detectFamily(modelId), modelId));
}

// Delegated binding so the wizard and the preset editor can both embed a plain
// <button data-open-protocol-docs> without importing this module themselves.
document.addEventListener('click', (e) => {
    const trigger = e.target.closest('[data-open-protocol-docs]');
    if (trigger) {
        e.preventDefault();
        openProtocolDocsModal();
    }
});
