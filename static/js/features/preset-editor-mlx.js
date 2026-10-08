// MLX-family information architecture for the shared preset editor.
// Existing DOM ids remain the serialization contract; this module only groups
// those fields so Rapid-MLX and future MLX backends share a coherent layout.

const SECTION_COPY = {
  model: {
    mlx: ['Model & Fit', 'MLX Model & Fit', 'Model source, quantization, memory composition, and safe headroom'],
    llama: ['Model', 'Model & Memory', 'Model file path, GPU offloading, memory locking'],
  },
  generation: {
    mlx: ['Generation', 'Generation', 'Sampling, thinking, and model protocol defaults'],
    llama: ['Generation', 'Generation & Sampling', 'Sampling parameters, reasoning controls, and output limits'],
  },
  context: {
    mlx: ['Cache & Performance', 'Cache & Performance', 'Context window, active KV, retained prompts, scheduling, and prefill policy'],
    llama: ['Context', 'Context & KV Cache', 'Context window size, KV cache quantization, and VRAM fitting'],
  },
  advanced: {
    mlx: ['Server & Safety', 'Server & Safety', 'Network access, concurrency boundaries, and qualified companions'],
    llama: ['Advanced', 'Advanced', 'Server access, fit-to-VRAM, seed, and extra CLI flags'],
  },
};

const GROUPS = [
  {
    section: 'generation', id: 'thinking', title: 'Thinking & reasoning',
    description: 'Control model reasoning behavior independently from sampling.',
    rows: ['pe-row-rapid-reasoning', 'pe-row-rapid-reasoning-mode'],
  },
  {
    section: 'generation', id: 'protocol', title: 'Model protocol',
    description: 'Keep automatic detection unless a modified finetune requires an override.',
    rows: ['pe-row-rapid-parser-overrides', 'pe-row-rapid-workload'],
  },
  {
    section: 'generation', id: 'output', title: 'Output limit',
    description: 'Cap each response without changing the model context window.',
    rows: ['pe-row-max-tokens-seed'],
  },
  {
    section: 'context', id: 'context-window', title: 'Context window',
    description: 'Per-request window (prompt plus output). Larger windows reserve more active memory.',
    rows: ['pe-row-context-size'],
  },
  {
    section: 'context', id: 'active-memory', title: 'Active memory',
    description: 'Precision and prefill choices that affect live unified-memory pressure.',
    rows: ['pe-row-rapid-advanced', 'pe-row-rapid-architecture-overrides'],
  },
  {
    section: 'context', id: 'retained-cache', title: 'Retained prompt cache',
    description: 'Bound reusable prompt snapshots by both memory and working-set size.',
    rows: ['pe-row-rapid-cache-mode', 'pe-row-rapid-prefix-cache', 'pe-row-rapid-cache-memory', 'pe-row-rapid-hybrid-cache-entries'],
  },
  {
    section: 'context', id: 'scheduler', title: 'Scheduler & throughput',
    description: 'Advanced batching and admission limits; defaults suit one interactive user.',
    rows: ['pe-row-rapid-throughput', 'pe-row-rapid-batch-sizes'],
  },
  {
    section: 'advanced', id: 'companions', title: 'Companions & experimental acceleration',
    description: 'Only qualified local companions belong here; unsupported remote launches fail closed.',
    collapsible: true,
    rows: ['pe-row-rapid-speculative'],
  },
];

const originalPositions = new Map();
const originalNavPositions = new Map();
const originalGenerationDisplays = new Map();

const SHARED_SAMPLING_CONTROL_IDS = [
  'modal-temperature',
  'modal-top-p',
  'modal-top-k',
  'modal-min-p',
  'modal-repeat-penalty',
  'modal-presence-penalty',
];

function rememberPosition(row) {
  if (originalPositions.has(row.id) || !row.parentNode) return;
  const anchor = document.createComment(`mlx-origin:${row.id}`);
  row.parentNode.insertBefore(anchor, row);
  originalPositions.set(row.id, anchor);
}

function restorePosition(row) {
  const anchor = originalPositions.get(row.id);
  if (anchor?.parentNode) anchor.parentNode.insertBefore(row, anchor.nextSibling);
  row.classList.remove('mlx-native-row');
}

function configureSectionCopy(modal, enabled) {
  Object.entries(SECTION_COPY).forEach(([section, copy]) => {
    const navLabel = modal.querySelector(`.preset-nav-item[data-section="${section}"] .pni-label`);
    const panel = modal.querySelector(`.preset-editor-section[data-section="${section}"]`);
    const title = panel?.querySelector('.pe-section-title');
    const description = panel?.querySelector('.pe-section-desc');
    const selected = enabled ? copy.mlx : copy.llama;
    if (navLabel) navLabel.textContent = selected[0];
    if (title) title.textContent = selected[1];
    if (description) description.textContent = selected[2];
  });
}

function ensureGroup(modal, group) {
  const body = modal.querySelector(`.preset-editor-section[data-section="${group.section}"] > .pe-section-body`);
  if (!body) return null;
  let container = body.querySelector(`[data-mlx-group="${group.id}"]`);
  if (container) return container;
  container = document.createElement(group.collapsible ? 'details' : 'section');
  container.className = 'mlx-native-group';
  container.dataset.mlxGroup = group.id;
  const header = document.createElement(group.collapsible ? 'summary' : 'div');
  header.className = 'mlx-native-group-header';
  const title = document.createElement('h4');
  title.className = 'mlx-native-group-title';
  title.textContent = group.title;
  const description = document.createElement('p');
  description.className = 'mlx-native-group-description';
  description.textContent = group.description;
  header.append(title, description);
  if (group.collapsible) {
    const badge = document.createElement('span');
    badge.className = 'mlx-native-group-badge';
    badge.textContent = 'Advanced';
    header.appendChild(badge);
  }
  container.appendChild(header);
  body.appendChild(container);
  return container;
}

function activateGroups(modal) {
  GROUPS.forEach(group => {
    const container = ensureGroup(modal, group);
    if (!container) return;
    group.rows.forEach(id => {
      const row = document.getElementById(id);
      if (!row) return;
      rememberPosition(row);
      row.classList.add('mlx-native-row');
      container.appendChild(row);
    });
  });
}

function deactivateGroups(modal) {
  GROUPS.flatMap(group => group.rows).forEach(id => {
    const row = document.getElementById(id);
    if (row) restorePosition(row);
  });
  modal.querySelectorAll('.mlx-native-group').forEach(group => group.remove());
}

function directGenerationChildForControl(body, controlId) {
  const control = body.querySelector(`#${controlId}`);
  if (!control) return null;

  let child = control;
  while (child.parentElement && child.parentElement !== body) child = child.parentElement;
  return child.parentElement === body ? child : null;
}

function configureGenerationVisibility(modal, enabled) {
  const body = modal.querySelector('.preset-editor-section[data-section="generation"] > .pe-section-body');
  if (!body) return;

  if (!enabled) {
    originalGenerationDisplays.forEach(({ display }, child) => {
      child.style.display = display;
    });
    originalGenerationDisplays.clear();
    body.querySelectorAll('.mlx-shared-sampling-row, .mlx-generation-llama-only').forEach(child => {
      child.classList.remove('mlx-shared-sampling-row', 'mlx-generation-llama-only');
    });
    return;
  }

  // The six shared sampler fields are nested in two unlabelled pe-row nodes.
  // Mark their *direct* section children rather than their pe-field wrappers so
  // a layout change within either row cannot hide a sibling sampler control.
  const samplingRows = new Set(
    SHARED_SAMPLING_CONTROL_IDS
      .map(controlId => directGenerationChildForControl(body, controlId))
      .filter(Boolean),
  );

  samplingRows.forEach(row => row.classList.add('mlx-shared-sampling-row'));
  Array.from(body.children).forEach(child => {
    if (samplingRows.has(child)) return;
    if (child.classList.contains('mlx-native-group')) return;
    if (!originalGenerationDisplays.has(child)) {
      originalGenerationDisplays.set(child, { display: child.style.display });
    }
    child.classList.add('mlx-generation-llama-only');
    child.style.display = 'none';
  });
}

function activateSection(modal, section) {
  modal.querySelectorAll('.preset-nav-item').forEach(item => {
    const active = item.dataset.section === section;
    item.classList.toggle('active', active);
    item.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  modal.querySelectorAll('.preset-editor-section').forEach(panel => {
    panel.classList.toggle('active', panel.dataset.section === section);
  });
}

function configureNavOrder(modal, enabled) {
  const nav = modal.querySelector('.preset-editor-nav');
  if (!nav) return;
  ['model', 'generation', 'context', 'advanced'].forEach(section => {
    const item = nav.querySelector(`.preset-nav-item[data-section="${section}"]`);
    if (!item) return;
    if (!originalNavPositions.has(section)) {
      const anchor = document.createComment(`mlx-nav-origin:${section}`);
      item.parentNode.insertBefore(anchor, item);
      originalNavPositions.set(section, anchor);
    }
    if (enabled) {
      nav.appendChild(item);
    } else {
      const anchor = originalNavPositions.get(section);
      if (anchor?.parentNode) anchor.parentNode.insertBefore(item, anchor.nextSibling);
    }
  });
}

export function configureMlxPresetEditor(modal, enabled) {
  if (!modal) return;
  configureSectionCopy(modal, enabled);
  configureNavOrder(modal, enabled);
  if (enabled) {
    activateGroups(modal);
    configureGenerationVisibility(modal, true);
    const active = modal.querySelector('.preset-nav-item.active')?.dataset.section;
    if (!['model', 'generation', 'context', 'advanced'].includes(active)) activateSection(modal, 'model');
  } else {
    configureGenerationVisibility(modal, false);
    deactivateGroups(modal);
  }
}
