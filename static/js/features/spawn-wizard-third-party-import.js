// Third-party model import (Ollama/LM Studio/Jan/GPT4All/HuggingFace-cache discovery) for the spawn wizard.
import { formatBytes } from './spawn-wizard-format.js';
import { resetOriginState, startOriginResolve } from './spawn-wizard-hf-origin.js';
import {
  dom,
  wizardState,
  onModelPathChanged,
  renderLocalModelHint,
  refreshStepGuardrails,
} from './spawn-wizard.js';

const TOOL_ICONS = {
  'Ollama': '🦙',
  'LM Studio': '🎨',
  'Jan': '🤖',
  'GPT4All': '🌍',
  'HuggingFace': '🤗',
};

export async function loadThirdPartyModels() {
  const listLoading = document.getElementById('import-model-list-loading');
  const listEmpty = document.getElementById('import-model-list-empty');
  const listEl = document.getElementById('import-model-list');
  if (!listEl) return;

  listLoading && (listLoading.style.display = '');
  listEmpty && (listEmpty.style.display = 'none');
  listEl.style.display = 'none';
  listEl.innerHTML = '';

  try {
    const headers = window.authHeaders
      ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
      : { 'Content-Type': 'application/json' };
    const resp = await fetch('/api/third-party-models', { method: 'POST', headers, body: JSON.stringify({}) });
    if (!resp.ok) throw new Error('fetch failed');
    const data = await resp.json();
    const models = (data.models || []).filter(Boolean);

    listLoading && (listLoading.style.display = 'none');

    if (!models.length) {
      listEmpty && (listEmpty.style.display = '');
      return;
    }

    // Group by source_tool
    const grouped = {};
    for (const m of models) {
      const tool = m.source_tool || 'Other';
      if (!grouped[tool]) grouped[tool] = [];
      grouped[tool].push(m);
    }

    for (const [tool, toolModels] of Object.entries(grouped)) {
      const icon = TOOL_ICONS[tool] || '📦';
      const groupEl = document.createElement('div');
      groupEl.className = 'import-tool-group';

      const headerEl = document.createElement('div');
      headerEl.className = 'import-tool-header';
      const iconEl = document.createElement('span');
      iconEl.className = 'import-tool-icon';
      iconEl.textContent = icon;
      const nameEl = document.createElement('span');
      nameEl.className = 'import-tool-name';
      nameEl.textContent = tool;
      headerEl.appendChild(iconEl);
      headerEl.appendChild(nameEl);
      groupEl.appendChild(headerEl);

      for (const m of toolModels) {
        const itemEl = document.createElement('div');
        itemEl.className = 'import-model-item';
        itemEl.setAttribute('role', 'button');
        itemEl.setAttribute('tabindex', '0');
        itemEl.dataset.path = m.path;

        const labelEl = document.createElement('span');
        labelEl.className = 'import-model-name';
        labelEl.textContent = m.name;
        itemEl.appendChild(labelEl);

        const sizeStr = formatBytes(m.size);
        if (sizeStr) {
          const sizeEl = document.createElement('span');
          sizeEl.className = 'import-model-size';
          sizeEl.textContent = sizeStr;
          itemEl.appendChild(sizeEl);
        }

        itemEl.addEventListener('click', () => selectImportedModel(m));
        itemEl.addEventListener('keydown', e => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectImportedModel(m); }
        });
        groupEl.appendChild(itemEl);
      }
      listEl.appendChild(groupEl);
    }

    listEl.style.display = '';
  } catch {
    listLoading && (listLoading.style.display = 'none');
    listEmpty && (listEmpty.style.display = '');
  }
}

function selectImportedModel(m) {
  wizardState.model.path = m.path;
  wizardState.model.source = 'import';
  wizardState.model.delivery = 'imported_local';
  // Pre-populate localMeta with tool display info so the hint shows the
  // human-readable name rather than the raw file/blob path.
  wizardState.model.localMeta = {
    model_name: m.name,
    size_display: formatBytes(m.size),
    source_tool: m.source_tool,
    path: m.path,
  };
  // Sync the fallback text input so validation sees the path.
  if (dom.importPathInput) dom.importPathInput.value = m.path;
  // Mark the selected card visually.
  document.querySelectorAll('.import-model-item').forEach(el => el.classList.remove('selected'));
  // Find and mark the clicked item — match by path data attribute.
  document.querySelectorAll('.import-model-item').forEach(el => {
    if (el.dataset.path === m.path) el.classList.add('selected');
  });
  // Trigger arch inference + introspection using the model name for heuristics.
  resetOriginState();
  onModelPathChanged();
  renderLocalModelHint(); // also calls _refreshHfOriginSection (shows detecting)
  // Start origin resolution immediately so the widget updates without waiting for doIntrospect
  if (!wizardState.model.originRepo) {
    startOriginResolve();
  }
  refreshStepGuardrails();
}
