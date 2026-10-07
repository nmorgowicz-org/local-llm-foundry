// ── Guide AI Surface ────────────────────────────────────────────────────────
// Multi-mode assistant steering surface: Quick, Director, Surprise.

import { activeChatTab, getChatViewBindings, scheduleChatPersist, invalidateChatContext } from './chat-state.js';
import { settingsState } from '../core/app-state.js';
import { showToast } from './toast.js';

const DIRECTOR_IDEA_COUNT = 4;

const DIRECTOR_TYPE_LABELS = {
    pressure: 'Pressure',
    reveal: 'Reveal',
    escalation: 'Escalation',
    interruption: 'Interruption',
    twist: 'Twist',
    'tone-shift': 'Tone Shift',
    reversal: 'Reversal',
    intimacy: 'Intimacy',
    investigation: 'Investigation',
    confrontation: 'Confrontation',
};

const SURPRISE_KIND_META = {
    twist: 'Introduce a surprising reversal or shift in meaning.',
    interruption: 'Interrupt the current flow with outside pressure.',
    reveal: 'Surface hidden information or motive.',
    escalation: 'Raise stakes, danger, or emotional intensity.',
    'tone-shift': 'Change the scene’s emotional or stylistic temperature.',
    custom: 'Use the note exactly as written.',
};

let quickGuideState = {
    expanded: false,
    mode: 'quick',
    lastUsedInstruction: null,
    directorLoading: false,
    directorIdeas: [],
};

export function toggleQuickGuide() {
    if (settingsState.enabled_quick_guide === false) return;

    quickGuideState.expanded = !quickGuideState.expanded;
    if (quickGuideState.expanded) {
        window.dispatchEvent(new CustomEvent('quickGuideOpened'));
    }
    updateQuickGuideUI();
}

export function closeQuickGuide() {
    if (!quickGuideState.expanded) return;
    quickGuideState.expanded = false;
    updateQuickGuideUI();
}

export function isQuickGuideEnabled() {
    return settingsState.enabled_quick_guide !== false;
}

export function getQuickGuideState() {
    return quickGuideState;
}

function getModeSubtitle(mode) {
    if (mode === 'director') {
        return 'Turn one directing note into multiple continuation options.';
    }
    if (mode === 'surprise') {
        return 'Arm a hidden beat that can land on a later assistant reply.';
    }
    return 'Steer the next assistant reply without touching the main chat flow.';
}

function updateQuickGuideUI() {
    const container = document.getElementById('quick-guide-container');
    const toggleBtn = document.getElementById('quick-guide-toggle');
    const wrapper = toggleBtn?.closest('.guided-tool');
    const input = document.getElementById('quick-guide-input');
    const subtitle = document.getElementById('quick-guide-subtitle');
    const status = document.getElementById('quick-guide-status');
    const restoreBtn = document.getElementById('quick-guide-restore-btn');
    const statusRow = document.getElementById('quick-guide-status-row');
    const statusChip = document.getElementById('quick-guide-status-chip');
    const tab = activeChatTab();
    const draft = tab?.quick_guide_draft || '';
    const activeGuide = tab?._quickGuideInFlight ? (tab.quick_guide_active || draft) : '';
    const lastRevision = tab?._quickGuideLastRun || null;
    const armedCount = tab?.armed_story_beats?.filter(beat => beat.enabled !== false).length || 0;

    if (!container || !toggleBtn || !tab) return;

    if (quickGuideState.expanded) {
        container.classList.add('quick-guide-expanded');
        toggleBtn.classList.add('active');
        toggleBtn.setAttribute('aria-expanded', 'true');
        wrapper?.classList.add('is-open');

        if (quickGuideState.mode === 'quick' && input) {
            input.focus();
            input.value = draft;
        }
    } else {
        container.classList.remove('quick-guide-expanded');
        toggleBtn.classList.remove('active');
        toggleBtn.setAttribute('aria-expanded', 'false');
        wrapper?.classList.remove('is-open');
    }

    toggleBtn.classList.toggle(
        'guided-action-btn-attentive',
        !!activeGuide || (!!draft && quickGuideState.expanded) || armedCount > 0
    );

    if (status) {
        if (activeGuide) {
            status.textContent = 'Applying';
            status.hidden = false;
        } else if (armedCount > 0) {
            status.textContent = `${armedCount} Armed`;
            status.hidden = false;
        } else if (quickGuideState.expanded) {
            status.textContent = draft ? 'Draft' : 'Idle';
            status.hidden = false;
        } else {
            status.hidden = true;
        }
    }

    if (restoreBtn) {
        restoreBtn.disabled = !lastRevision;
        restoreBtn.hidden = !lastRevision;
    }

    if (subtitle) subtitle.textContent = getModeSubtitle(quickGuideState.mode);
    if (statusRow && statusChip) {
        statusRow.hidden = quickGuideState.mode === 'surprise' || armedCount === 0;
        statusChip.textContent = armedCount > 0 ? `${armedCount} surprise${armedCount === 1 ? '' : 's'} armed` : '';
    }

    document.querySelectorAll('.quick-guide-mode-btn').forEach(btn => {
        const isActive = btn.dataset.guideMode === quickGuideState.mode;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    document.querySelectorAll('.quick-guide-mode-panel').forEach(panel => {
        const isActive = panel.dataset.guidePanel === quickGuideState.mode;
        panel.hidden = !isActive;
        panel.classList.toggle('quick-guide-mode-panel-active', isActive);
    });

    const directorInput = document.getElementById('quick-guide-director-input');
    if (directorInput && typeof tab._guideDirectorDraft === 'string') {
        directorInput.value = tab._guideDirectorDraft;
    }

    const surpriseInput = document.getElementById('quick-guide-surprise-input');
    if (surpriseInput && typeof tab._guideSurpriseDraft === 'string') {
        surpriseInput.value = tab._guideSurpriseDraft;
    }

    renderDirectorIdeas();
    renderArmedSurprises();
    updateLastUsedDisplay();
}

function truncate(value, maxLength) {
    if (!value) return '';
    return value.length > maxLength ? `${value.slice(0, maxLength)}...` : value;
}

function updateLastUsedDisplay() {
    const lastUsed = document.getElementById('quick-guide-last-used');
    const tab = activeChatTab();
    const lastInstruction = tab?._quickGuideLastRun?.instruction || quickGuideState.lastUsedInstruction;
    if (!lastUsed) return;

    if (lastInstruction && quickGuideState.mode === 'quick') {
        lastUsed.textContent = `Last applied: ${truncate(lastInstruction, 60)}`;
        lastUsed.style.display = 'block';
        return;
    }

    lastUsed.style.display = 'none';
}

function setGuideMode(mode) {
    quickGuideState.mode = mode;
    updateQuickGuideUI();
}

function setupModeButtons() {
    document.querySelectorAll('.quick-guide-mode-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            setGuideMode(btn.dataset.guideMode);
        });
    });
}

function setupQuickInputHandler() {
    const input = document.getElementById('quick-guide-input');
    if (!input) return;

    input.addEventListener('input', (e) => {
        const tab = activeChatTab();
        if (tab) {
            tab.quick_guide_draft = e.target.value;
            scheduleChatPersist();
            updateQuickGuideUI();
        }
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuickGuide();
        } else if (e.key === 'Escape') {
            quickGuideState.expanded = false;
            updateQuickGuideUI();
        }
    });
}

function submitQuickGuide() {
    const tab = activeChatTab();
    const instruction = (tab?.quick_guide_draft || '').trim();

    // Always collapse the UI — even if there is no active tab.
    const inputEl = document.getElementById('quick-guide-input');
    if (inputEl) inputEl.value = '';
    quickGuideState.expanded = false;
    updateQuickGuideUI();

    if (!tab) return;

    quickGuideState.lastUsedInstruction = instruction || null;
    window.dispatchEvent(new CustomEvent('quickGuideSubmitted', {
        detail: { instruction },
    }));

    tab.quick_guide_draft = '';
    showToast(instruction ? 'Reply guide applied' : 'Reply guide cleared', 'success');
}

function restorePreviousInstructionForEdit() {
    const tab = activeChatTab();
    const input = document.getElementById('quick-guide-input');
    const lastRun = tab?._quickGuideLastRun;
    if (!tab || !lastRun?.instruction) return;

    const targetIndex = typeof lastRun.targetIndex === 'number' ? lastRun.targetIndex : -1;
    const targetMsg = targetIndex >= 0 ? tab.messages[targetIndex] : null;
    if (targetMsg && targetMsg.role === (lastRun.targetRole || 'assistant')) {
        tab.messages.splice(targetIndex, 1);
    } else {
        for (let i = tab.messages.length - 1; i >= 0; i -= 1) {
            if (tab.messages[i]?.role === (lastRun.targetRole || 'assistant')) {
                tab.messages.splice(i, 1);
                break;
            }
        }
    }
    invalidateChatContext(tab);

    tab.quick_guide_draft = lastRun.instruction;
    tab.quick_guide_active = '';
    tab.updated_at = Date.now();
    quickGuideState.mode = 'quick';
    quickGuideState.expanded = true;
    getChatViewBindings().renderChatMessages?.();
    scheduleChatPersist();
    updateQuickGuideUI();
    if (input) {
        input.value = lastRun.instruction;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
    }

    showToast('Previous guided reply removed. Edit and apply again.', 'info');
}

function setupQuickButtons() {
    const submitBtn = document.getElementById('quick-guide-submit-btn');
    const restoreBtn = document.getElementById('quick-guide-restore-btn');
    if (!submitBtn) return;

    submitBtn.addEventListener('click', submitQuickGuide);
    restoreBtn?.addEventListener('click', restorePreviousInstructionForEdit);
}

function buildDirectorPrompt(direction) {
    return `You are a story director helping shape the assistant's next reply.\n\nTASK: Based on [STORY CONTEXT], generate ${DIRECTOR_IDEA_COUNT} distinct assistant-side continuation options that obey this director note:\n\n${direction}\n\nEach option should tell the assistant how to continue the next reply, not tell the user what to type.\n\nGUIDELINES:\n- Keep each option specific and usable immediately.\n- Focus on scene control, pacing, emotional pressure, revelation, interruption, confrontation, or tone shifts.\n- Do not write as the user.\n- Do not include preamble or numbering.\n- Use strong, scene-specific language rather than generic advice.\n\n[STORY CONTEXT]`;
}

function parseSuggestionString(text) {
    const [title, ...rest] = (text || '').split('\n');
    return {
        title: (title || '').trim(),
        description: rest.join('\n').trim(),
    };
}

function normalizeDirectorIdea(raw) {
    if (!raw || typeof raw !== 'object') return null;
    const title = String(raw.title || '').trim();
    if (!title) return null;
    const effect = String(raw.effect || '').trim();
    const detail = String(raw.detail || raw.description || '').trim();
    const suggestionType = String(raw.type || raw.suggestion_type || 'pressure')
        .trim()
        .toLowerCase()
        .replaceAll('_', '-')
        .replaceAll(' ', '-');

    return {
        type: suggestionType || 'pressure',
        title,
        effect: effect || title,
        detail: detail || effect || title,
    };
}

function normalizeDirectorIdeas(cards, suggestions) {
    const normalizedCards = Array.isArray(cards)
        ? cards.map(normalizeDirectorIdea).filter(Boolean)
        : [];
    if (normalizedCards.length) return normalizedCards;

    return Array.isArray(suggestions)
        ? suggestions.map(text => {
            const { title, description } = parseSuggestionString(text);
            if (!title) return null;
            const [effectLine, ...rest] = (description || title).split('. ');
            return {
                type: 'pressure',
                title,
                effect: (effectLine || title).trim(),
                detail: (rest.join('. ') || description || effectLine || title).trim(),
            };
        }).filter(Boolean)
        : [];
}

async function fetchDirectorIdeas() {
    const tab = activeChatTab();
    const input = document.getElementById('quick-guide-director-input');
    const direction = input?.value.trim() || '';
    if (!tab || !direction) {
        showToast('Add a directing note first', 'warning');
        return;
    }

    tab._guideDirectorDraft = direction;
    quickGuideState.directorLoading = true;
    quickGuideState.directorIdeas = [];
    scheduleChatPersist();
    renderDirectorIdeas();

    try {
        const payload = {
            tab_id: tab.id,
            category: 'director',
            count: DIRECTOR_IDEA_COUNT,
            context_depth: 10,
            prompt: buildDirectorPrompt(direction),
            messages: (tab.messages || []).map(message => ({
                role: message.role,
                content: message.content,
            })),
            system_prompt: tab.system_prompt || '',
            context_notes: (tab.context_notes || []).filter(note => note?.content?.trim()),
            quick_guide_active: '',
        };

        const response = await fetch('/api/chat/suggestions', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const errText = await response.text().catch(() => '');
            throw new Error(`HTTP ${response.status}: ${errText || response.statusText}`);
        }

        const data = await response.json();
        quickGuideState.directorIdeas = normalizeDirectorIdeas(data.cards, data.suggestions);
        if (!quickGuideState.directorIdeas.length) {
            showToast('No director ideas generated', 'warning');
        }
    } catch (error) {
        showToast(`Director ideas failed: ${error.message}`, 'error');
    } finally {
        quickGuideState.directorLoading = false;
        renderDirectorIdeas();
    }
}

async function applyDirectorIdea(index) {
    const idea = quickGuideState.directorIdeas[index];
    if (!idea) return;
    const typeLabel = DIRECTOR_TYPE_LABELS[idea.type] || 'Direction';
    const instruction = `Use this ${typeLabel.toLowerCase()} steering for the next reply: ${idea.title}. ${idea.effect}. ${idea.detail}`;
    const [{ sendOneShotGuideReply }] = await Promise.all([
        import('./chat-transport.js'),
    ]);
    quickGuideState.expanded = false;
    updateQuickGuideUI();
    await sendOneShotGuideReply(instruction);
}

function renderDirectorIdeas() {
    const container = document.getElementById('quick-guide-director-results');
    if (!container) return;

    if (quickGuideState.directorLoading) {
        container.innerHTML = `<div class="quick-guide-empty-state">Generating director options…</div>`;
        return;
    }

    if (!quickGuideState.directorIdeas.length) {
        container.innerHTML = `<div class="quick-guide-empty-state">Generate a small set of assistant-side options, then apply the one that best fits the scene.</div>`;
        return;
    }

    // eslint-disable-next-line no-unsanitized/property
    container.innerHTML = quickGuideState.directorIdeas.map((idea, index) => {
        const typeLabel = DIRECTOR_TYPE_LABELS[idea.type] || 'Direction';
        return `
            <div class="quick-guide-director-item" data-director-index="${index}">
                <div class="quick-guide-director-main">
                    <div class="quick-guide-director-toprow">
                        <div class="quick-guide-director-meta">
                            <span class="quick-guide-director-badge quick-guide-director-badge-${window.escapeHtml(idea.type)}">${window.escapeHtml(typeLabel)}</span>
                        </div>
                        <button class="quick-guide-btn quick-guide-submit-btn quick-guide-director-apply-btn" data-director-apply="${index}" type="button">Apply</button>
                    </div>
                    <div class="quick-guide-director-title">${window.escapeHtml(idea.title)}</div>
                    <div class="quick-guide-director-effect">${window.escapeHtml(idea.effect)}</div>
                    <div class="quick-guide-director-copy">${window.escapeHtml(idea.detail)}</div>
                </div>
            </div>
        `;
    }).join('');

    container.querySelectorAll('[data-director-apply]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            applyDirectorIdea(parseInt(btn.dataset.directorApply, 10));
        });
    });
}

function normalizeSurpriseInstruction(kind, note) {
    const noteText = note.trim();
    const description = SURPRISE_KIND_META[kind] || SURPRISE_KIND_META.custom;
    if (kind === 'custom') return noteText;
    if (!noteText) return description;
    return `${description} Specific note: ${noteText}`;
}

function renderArmedSurprises() {
    const container = document.getElementById('quick-guide-armed-list');
    const tab = activeChatTab();
    if (!container || !tab) return;

    const beats = (tab.armed_story_beats || []).filter(beat => beat.enabled !== false);
    if (!beats.length) {
        container.innerHTML = `<div class="quick-guide-empty-state">No surprises armed. Arm one to inject a hidden future beat into a later assistant reply.</div>`;
        return;
    }

    // eslint-disable-next-line no-unsanitized/property
    container.innerHTML = beats.map(beat => `
        <div class="quick-guide-armed-item" data-armed-id="${window.escapeHtml(beat.id)}">
            <div class="quick-guide-armed-meta">
                <strong>${window.escapeHtml(beat.kind)}</strong>
                <span>${beat.remaining_turns === 0 ? 'next reply' : `after ${beat.remaining_turns} reply${beat.remaining_turns === 1 ? '' : 'ies'}`}</span>
            </div>
            <div class="quick-guide-armed-copy">${window.escapeHtml(beat.instruction)}</div>
            <button class="quick-guide-btn quick-guide-restore-btn" data-remove-armed="${window.escapeHtml(beat.id)}" type="button">Remove</button>
        </div>
    `).join('');

    container.querySelectorAll('[data-remove-armed]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeArmedSurprise(btn.dataset.removeArmed);
        });
    });
}

function armSurprise() {
    const tab = activeChatTab();
    const kindEl = document.getElementById('quick-guide-surprise-kind');
    const delayEl = document.getElementById('quick-guide-surprise-delay');
    const input = document.getElementById('quick-guide-surprise-input');
    if (!tab || !kindEl || !delayEl || !input) return;

    const kind = kindEl.value;
    const note = input.value.trim();
    if (!note) {
        showToast('Add a surprise note first', 'warning');
        return;
    }

    tab.armed_story_beats = tab.armed_story_beats || [];
    tab.armed_story_beats.push({
        id: crypto.randomUUID(),
        kind,
        instruction: normalizeSurpriseInstruction(kind, note),
        remaining_turns: parseInt(delayEl.value, 10) || 0,
        created_at: Date.now(),
        enabled: true,
    });
    tab._guideSurpriseDraft = '';
    input.value = '';
    scheduleChatPersist();
    renderArmedSurprises();
    updateQuickGuideUI();
    showToast('Surprise armed', 'success');
}

function removeArmedSurprise(id) {
    const tab = activeChatTab();
    if (!tab) return;
    tab.armed_story_beats = (tab.armed_story_beats || []).filter(beat => beat.id !== id);
    scheduleChatPersist();
    renderArmedSurprises();
    updateQuickGuideUI();
}

function setupDirectorMode() {
    const input = document.getElementById('quick-guide-director-input');
    const generateBtn = document.getElementById('quick-guide-director-generate-btn');
    if (input) {
        input.addEventListener('input', (e) => {
            const tab = activeChatTab();
            if (tab) {
                tab._guideDirectorDraft = e.target.value;
                scheduleChatPersist();
            }
        });
    }
    generateBtn?.addEventListener('click', fetchDirectorIdeas);
}

function setupSurpriseMode() {
    const input = document.getElementById('quick-guide-surprise-input');
    const armBtn = document.getElementById('quick-guide-surprise-arm-btn');
    if (input) {
        input.addEventListener('input', (e) => {
            const tab = activeChatTab();
            if (tab) {
                tab._guideSurpriseDraft = e.target.value;
                scheduleChatPersist();
            }
        });
    }
    armBtn?.addEventListener('click', armSurprise);
}

function setupClickOutside() {
    document.addEventListener('click', (e) => {
        const container = document.getElementById('quick-guide-container');
        const toggleBtn = document.getElementById('quick-guide-toggle');

        if (!container || !toggleBtn) return;

        const isClickInside = container.contains(e.target) || toggleBtn.contains(e.target);

        if (!isClickInside && quickGuideState.expanded) {
            quickGuideState.expanded = false;
            updateQuickGuideUI();
        }
    });
}

export function initQuickGuide() {
    setupModeButtons();
    setupQuickInputHandler();
    setupQuickButtons();
    setupDirectorMode();
    setupSurpriseMode();
    setupClickOutside();
    window.addEventListener('activeTabChanged', updateQuickGuideUI);
    window.addEventListener('quickGuideStateChanged', updateQuickGuideUI);
    updateQuickGuideUI();
}
