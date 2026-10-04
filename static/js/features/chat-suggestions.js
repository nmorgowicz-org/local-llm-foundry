// ── Suggestions (Dropdown) ──────────────────────────────────────────────────
// Dropdown menu with AI-generated suggestions (General, Plot Twist, New Character).

import { activeChatTab, autoResizeChatInput, normalizeGeneratedMessageContent, persistChatTabs } from './chat-state.js';
import { chat, settingsState } from '../core/app-state.js';
import { escapeHtml } from '../core/format.js';
import { showToast } from './toast.js';
import './chat-templates.js';
import { sendChatWithContent } from './chat-transport.js';
import { saveSettings } from './settings.js';

const CATEGORY_META = {
    general: { label: 'General', description: 'Versatile next-step prompts that fit almost any conversation.' },
    'plot-twist': { label: 'Plot Twist', description: 'Unexpected reversals, reveals, and pressure spikes that raise the stakes.' },
    'new-character': { label: 'New Character', description: 'Fresh entrants who create conflict, chemistry, or new information.' },
    director: { label: 'Director', description: 'High-level scene direction, pacing shifts, and cinematic steering.' },
    action: { label: 'Action', description: 'Momentum, danger, movement, and immediate physical stakes.' },
    comedy: { label: 'Comedy', description: 'Humor beats, awkward pivots, and playful escalation.' },
    fantasy: { label: 'Fantasy', description: 'Magic, myth, wonder, and worldbuilding-driven next steps.' },
    horror: { label: 'Horror', description: 'Dread, menace, and unsettling turns that tighten the atmosphere.' },
    mystery: { label: 'Mystery', description: 'Clues, suspicion, revelations, and investigative momentum.' },
    noir: { label: 'Noir', description: 'Shadowy motives, sharp dialogue, and morally messy developments.' },
    romance: { label: 'Romance', description: 'Chemistry, vulnerability, longing, and emotional tension.' },
    'sci-fi': { label: 'Sci-Fi', description: 'Futuristic complications, speculative ideas, and tech-driven stakes.' },
    thriller: { label: 'Thriller', description: 'Urgency, pressure, danger, and escalating consequences.' },
    character: { label: 'Character', description: 'Choices and beats that reveal personality, desire, or conflict.' },
    explicit: { label: 'Explicit', description: 'Unfiltered prompts for adult-only scenes when explicit mode is enabled.' },
};

let suggestionsState = {
    expanded: false,
    currentCategory: 'general',
    setupCollapsed: false,
    isLoading: false,
    hasGenerated: false,
    suggestions: [],
    mode: 'browse',
    draftSuggestion: null,
    draftText: '',
    rewrittenText: '',
    rewriteLoading: false,
    recentSuggestions: [],
    customCategories: new Map(),
    retryCount: 0,
    maxRetries: 3,
    lastError: null,
    isOffline: false,
    previewCategory: null,
};

// ── Dropdown Toggle ──────────────────────────────────────────────────────────

export function toggleSuggestionsDropdown() {
    if (settingsState.enabled_suggestions === false) return;

    suggestionsState.expanded = !suggestionsState.expanded;
    if (suggestionsState.expanded) {
        window.dispatchEvent(new CustomEvent('suggestionsOpened'));
    }
    updateDropdownUI();
}

export function closeSuggestionsDropdown() {
    if (!suggestionsState.expanded) return;
    suggestionsState.expanded = false;
    updateDropdownUI();
}

export function isSuggestionsEnabled() {
    return settingsState.enabled_suggestions !== false;
}

export function getSuggestionsState() {
    return suggestionsState;
}

// ── Dropdown UI Updates ──────────────────────────────────────────────────────

function updatePreviewText() {
    const meta = CATEGORY_META[suggestionsState.currentCategory] || {
        label: suggestionsState.currentCategory,
        description: 'Generate suggestions for the current conversation.',
    };
    const previewMeta = CATEGORY_META[suggestionsState.previewCategory || suggestionsState.currentCategory] || meta;
    const preview = document.getElementById('suggestions-category-preview');
    if (preview) {
        preview.textContent = `${previewMeta.label}: ${previewMeta.description}`;
    }

    // Update category button active states and tooltips
    document.querySelectorAll('.suggestion-category-btn').forEach(btn => {
        const category = btn.dataset.category;
        const categoryMeta = CATEGORY_META[category] || {
            label: btn.textContent.trim(),
            description: 'Generate suggestions for the current conversation.',
        };
        btn.classList.toggle('active', category === suggestionsState.currentCategory);
        btn.setAttribute('title', `${categoryMeta.label}: ${categoryMeta.description}`);
        btn.setAttribute('aria-label', `${categoryMeta.label}. ${categoryMeta.description}`);
    });
}

function updateDropdownUI() {
    const dropdown = document.getElementById('suggestions-dropdown');
    const toggleBtn = document.getElementById('suggestions-toggle');
    const wrapper = toggleBtn?.closest('.guided-tool');
    const listContainer = document.getElementById('suggestions-list');
    const explicitGroup = document.getElementById('suggestions-explicit-group');
    const status = document.getElementById('suggestions-toggle-status');
    const description = document.getElementById('suggestions-category-description');
    const setupToggle = document.getElementById('suggestions-view-toggle');
    const backBtn = document.getElementById('suggestions-editor-back');
    const meta = CATEGORY_META[suggestionsState.currentCategory] || {
        label: suggestionsState.currentCategory,
        description: 'Generate suggestions for the current conversation.',
    };

    if (!dropdown || !toggleBtn) return;

    if (suggestionsState.expanded) {
        dropdown.classList.add('dropdown-expanded');
        toggleBtn.classList.add('active');
        toggleBtn.setAttribute('aria-expanded', 'true');
        wrapper?.classList.add('is-open');
    } else {
        dropdown.classList.remove('dropdown-expanded');
        toggleBtn.classList.remove('active');
        toggleBtn.setAttribute('aria-expanded', 'false');
        wrapper?.classList.remove('is-open');
    }

    dropdown.classList.toggle('setup-collapsed', suggestionsState.setupCollapsed);
    dropdown.classList.toggle('suggestions-workspace-open', suggestionsState.mode !== 'browse');

    if (status) {
        status.textContent = suggestionsState.isLoading ? 'Loading' : meta.label;
        status.hidden = !suggestionsState.expanded;
    }
    if (description) {
        description.hidden = !suggestionsState.isLoading;
        description.textContent = suggestionsState.isLoading
            ? `Generating ${meta.label.toLowerCase()} ideas from the current conversation…`
            : '';
    }
    if (setupToggle) {
        setupToggle.textContent = suggestionsState.setupCollapsed ? 'Show Setup' : 'Hide Setup';
        setupToggle.setAttribute('aria-pressed', suggestionsState.setupCollapsed ? 'true' : 'false');
        setupToggle.hidden = suggestionsState.mode !== 'browse';
    }
    if (backBtn) {
        backBtn.hidden = suggestionsState.mode === 'browse';
    }

    // Toggle explicit group visibility based on explicit_level
    if (explicitGroup) {
        const tab = activeChatTab();
        const explicitEnabled = (tab?.explicit_level ?? 0) > 0;
        explicitGroup.classList.toggle('explicit-enabled', explicitEnabled);
    }

    // Apply search filter
    applySearchFilter();

    // Lightweight preview + button state update (does NOT rebuild custom buttons)
    updatePreviewText();

    if (listContainer) {
        renderSuggestionsList();
    }

   // Always render recent suggestions
    // (removed - no longer needed)
}

function renderCustomCategoryButtons() {
    const tagCloud = document.querySelector('.suggestions-tag-cloud');
    if (!tagCloud) return;

    const customs = Array.from(suggestionsState.customCategories.entries());
    if (customs.length === 0) return;

    // Separate explicit and non-explicit custom categories
    const explicitCustoms = customs.filter(([_key, catData]) => {
        const isExplicit = typeof catData === 'string' ? false : (catData.explicit || false);
        return isExplicit;
    });
    const nonExplicitCustoms = customs.filter(([_key, catData]) => {
        const isExplicit = typeof catData === 'string' ? false : (catData.explicit || false);
        return !isExplicit;
    });

    // Remove ALL existing custom elements (buttons + chips containers) to avoid accumulation
    tagCloud.querySelectorAll('.suggestion-category-btn[data-custom]').forEach(btn => btn.remove());
    tagCloud.querySelectorAll('.category-group-chips[data-custom-chips]').forEach(el => el.remove());
    // Clear chips containers in custom group (recreated each time)
    const customGroupEl = document.getElementById('suggestions-custom-group');
    if (customGroupEl) {
        customGroupEl.querySelectorAll('.category-group-chips').forEach(el => el.remove());
    }

    // Add non-explicit custom categories to a new "Custom" group (before Explicit group)
    if (nonExplicitCustoms.length > 0) {
        const explicitGroup = document.getElementById('suggestions-explicit-group');
        let customGroup = document.getElementById('suggestions-custom-group');
        if (!customGroup) {
            customGroup = document.createElement('div');
            customGroup.id = 'suggestions-custom-group';
            customGroup.className = 'category-group';
            customGroup.innerHTML = '<div class="category-group-title">Custom</div>';
            // Insert before the explicit group
            if (explicitGroup) {
                explicitGroup.parentNode.insertBefore(customGroup, explicitGroup);
            } else {
                tagCloud.appendChild(customGroup);
            }
        }

        // Create chips container for custom buttons
        const chips = document.createElement('div');
        chips.className = 'category-group-chips';
        const chipsInner = document.createElement('div');
        chipsInner.className = 'category-group-chips-inner';
        chips.appendChild(chipsInner);
        customGroup.appendChild(chips);

        nonExplicitCustoms.forEach(([key, _catData]) => {
            const label = key.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            const btn = document.createElement('button');
            btn.className = 'suggestion-category-btn';
            btn.dataset.category = key;
            btn.dataset.custom = 'true';
            btn.textContent = label;
            btn.setAttribute('title', `${label}: Custom category`);
            btn.setAttribute('aria-label', `${label}. Custom category`);
            chipsInner.appendChild(btn);
            setupCategoryButton(btn);
        });
    } else {
        // Remove the custom group if no non-explicit custom categories
        const customGroup = document.getElementById('suggestions-custom-group');
        if (customGroup) customGroup.remove();
    }

    // Add explicit custom categories to the Explicit group
    const explicitGroupEl = document.getElementById('suggestions-explicit-group');
    if (explicitGroupEl && explicitCustoms.length > 0) {
        // Create chips container for explicit custom buttons
        const chips = document.createElement('div');
        chips.className = 'category-group-chips';
        chips.dataset.customChips = 'true';
        const chipsInner = document.createElement('div');
        chipsInner.className = 'category-group-chips-inner';
        chips.appendChild(chipsInner);
        explicitGroupEl.appendChild(chips);

        explicitCustoms.forEach(([key, _catData]) => {
            const label = key.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            const btn = document.createElement('button');
            btn.className = 'suggestion-category-btn';
            btn.dataset.category = key;
            btn.dataset.custom = 'true';
            btn.textContent = label;
            btn.setAttribute('title', `${label}: Custom explicit category`);
            btn.setAttribute('aria-label', `${label}. Custom explicit category`);
            chipsInner.appendChild(btn);
            setupCategoryButton(btn);
        });
    }
}

function setupCategoryButton(btn) {
    btn.addEventListener('mouseenter', () => {
        suggestionsState.previewCategory = btn.dataset.category;
        updatePreviewText();
    });
    btn.addEventListener('mouseleave', () => {
        suggestionsState.previewCategory = null;
        updatePreviewText();
    });
    btn.addEventListener('focus', () => {
        suggestionsState.previewCategory = btn.dataset.category;
        updatePreviewText();
    });
    btn.addEventListener('blur', () => {
        suggestionsState.previewCategory = null;
        updatePreviewText();
    });
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        setSuggestionCategory(btn.dataset.category);
    });
}

function applySearchFilter() {
    const searchInput = document.getElementById('suggestion-search-input');
    if (!searchInput) return;

    const query = searchInput.value.toLowerCase().trim();
    const chips = document.querySelectorAll('.suggestion-category-btn');
    const groups = document.querySelectorAll('.category-group');

    chips.forEach(chip => {
        const text = chip.textContent.toLowerCase();
        const visible = !query || text.includes(query);
        chip.style.display = visible ? '' : 'none';
    });

    groups.forEach(group => {
        const visibleChips = group.querySelectorAll('.suggestion-category-btn:not([style*="display: none"])');
        const hasVisible = visibleChips.length > 0;
        group.style.display = hasVisible ? '' : 'none';
    });
}

// ── Category Switching ───────────────────────────────────────────────────────

export function setSuggestionCategory(category) {
    if (category === 'manage') {
        manageCategories();
        return;
    }

    // Silently prevent explicit category access when explicit mode is off
    if (category === 'explicit') {
        const tab = activeChatTab();
        const explicitMode = (tab?.explicit_level ?? 0) > 0;
        if (!explicitMode) {
            import('./chat-templates.js').then(({ enableExplicitMode }) => {
                enableExplicitMode();
            });
        }
    }

    suggestionsState.currentCategory = category;
    suggestionsState.previewCategory = category;
    suggestionsState.hasGenerated = false;
    suggestionsState.suggestions = [];
    suggestionsState.setupCollapsed = false;
    updateDropdownUI();
}

// ── Suggestions List Rendering ───────────────────────────────────────────────

function renderSuggestionsList() {
    const container = document.getElementById('suggestions-list');
    if (!container) return;

    if (suggestionsState.mode === 'draft') {
        renderSuggestionDraftEditor(container);
        return;
    }

    if (suggestionsState.mode === 'preview') {
        renderSuggestionDraftPreview(container);
        return;
    }

    if (!suggestionsState.isLoading && !suggestionsState.hasGenerated) {
        container.innerHTML = '';
        return;
    }

    if (suggestionsState.isLoading) {
        container.innerHTML = `
            <div class="suggestions-loading" role="status" aria-live="polite">
                <div class="spinner" aria-hidden="true"></div>
                <p>Generating suggestions...</p>
            </div>
        `;
        return;
    }

    const suggestions = suggestionsState.suggestions;

    if (suggestions.length === 0) {
        container.innerHTML = `
            <div class="suggestions-empty-state" role="status" aria-live="polite">
                <p>No suggestions yet</p>
                <p class="text-sm">Choose a category, then generate tailored prompts for this conversation.</p>
            </div>
        `;
        return;
    }

    // eslint-disable-next-line no-unsanitized/property -- User content escaped via escapeHtml()
    container.innerHTML = suggestions.map((suggestion, index) => {
        // Parse Pathweaver format: "TITLE\nDESCRIPTION"
        const parts = suggestion.split('\n');
        const title = parts[0] || suggestion;
        const description = parts.slice(1).join('\n') || '';

        return `
        <div class="suggestion-item" data-index="${index}" role="option" aria-selected="false" tabindex="0">
            <div class="suggestion-main">
                ${description ? `<div class="suggestion-title">${escapeHtml(title)}</div>` : ''}
                <div class="suggestion-content">${escapeHtml(description || title)}</div>
            </div>
            <div class="suggestion-actions">
                <button class="suggestion-btn suggestion-btn-append" data-mode="draft" aria-label="Edit draft for suggestion: ${escapeHtml(title)}">Edit Draft</button>
                <button class="suggestion-btn suggestion-btn-use" data-mode="send" aria-label="Send suggestion as user direction: ${escapeHtml(title)}">Send Direction</button>
            </div>
        </div>
    `;
    }).join('');

    // Attach use handlers
    container.querySelectorAll('.suggestion-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const item = e.target.closest('.suggestion-item');
            const index = parseInt(item.dataset.index, 10);
            useSuggestion(index, btn.dataset.mode || 'replace');
        });
    });
}

function parseSuggestionText(text) {
    const [title, ...rest] = (text || '').split('\n');
    return {
        title: (title || '').trim(),
        description: rest.join('\n').trim(),
    };
}

function detectSuggestionRewriteStyle(tab) {
    const recentUserMessages = (tab?.messages || [])
        .filter(msg => msg.role === 'user' && msg.content?.trim())
        .slice(-5)
        .map(msg => msg.content.trim());

    if (recentUserMessages.length === 0) return 'instruction';

    const joined = recentUserMessages.join('\n').toLowerCase();
    const firstPersonMatches = joined.match(/\b(i|i'm|i'd|i'll|me|my|mine)\b/g) || [];
    const directiveMatches = joined.match(/\b(write|continue|have|make|let|show|focus|use|keep|add|rewrite|respond)\b/g) || [];

    if (directiveMatches.length >= firstPersonMatches.length && directiveMatches.length >= 2) {
        return 'instruction';
    }
    if (firstPersonMatches.length >= 3) {
        return 'first_person';
    }
    return 'third_person';
}

function buildSuggestionDraftSeed(text, tab) {
    const { title, description } = parseSuggestionText(text);
    const beat = [title, description].filter(Boolean).join('. ');
    const style = detectSuggestionRewriteStyle(tab);

    if (style === 'first_person') {
        return `Use this beat for my next turn:\n${beat}\n\nI want this to land in my established first-person voice and perspective.`;
    }
    if (style === 'third_person') {
        return `Use this beat for the next scene continuation:\n${beat}\n\nKeep it aligned with the existing third-person voice, tone, and tension.`;
    }
    return `Use this beat:\n${beat}\n\nTurn it into a natural continuation that matches the current scene voice, POV, and tension.`;
}

function openSuggestionDraft(index) {
    const tab = activeChatTab();
    const suggestion = suggestionsState.suggestions[index];
    if (!tab || !suggestion) return;

    suggestionsState.mode = 'draft';
    suggestionsState.setupCollapsed = true;
    suggestionsState.draftSuggestion = suggestion;
    suggestionsState.draftText = buildSuggestionDraftSeed(suggestion, tab);
    suggestionsState.rewrittenText = '';
    suggestionsState.rewriteLoading = false;
    updateDropdownUI();
}

function resetSuggestionWorkspace({ preserveResults = true } = {}) {
    suggestionsState.mode = 'browse';
    suggestionsState.draftSuggestion = null;
    suggestionsState.draftText = '';
    suggestionsState.rewrittenText = '';
    suggestionsState.rewriteLoading = false;
    if (!preserveResults) {
        suggestionsState.hasGenerated = false;
    }
    updateDropdownUI();
}

function renderSuggestionDraftEditor(container) {
    const { title } = parseSuggestionText(suggestionsState.draftSuggestion || '');
    const helperCopy = suggestionsState.rewriteLoading
        ? 'Rewriting your draft into fuller user-side prose…'
        : 'Edit the beat, then let the AI rewrite it into a fuller user-side message in your established voice and POV.';

    // eslint-disable-next-line no-unsanitized/property -- Internal state escaped before insertion
    container.innerHTML = `
        <div class="suggestions-workspace-card">
            <div class="suggestions-workspace-kicker">Draft From Suggestion</div>
            <div class="suggestions-workspace-title">${escapeHtml(title || 'Selected Suggestion')}</div>
            <div class="suggestions-workspace-copy">${escapeHtml(helperCopy)}</div>
            <textarea class="suggestions-draft-editor" id="suggestions-draft-editor" rows="8" placeholder="Adjust the beat, POV, emotional tone, or specific action before rewriting...">${escapeHtml(suggestionsState.draftText)}</textarea>
            <div class="suggestions-workspace-actions">
                <button class="suggestion-btn suggestion-btn-append" id="suggestions-draft-cancel" type="button">Cancel</button>
                <button class="suggestion-btn suggestion-btn-use" id="suggestions-draft-rewrite" type="button" ${suggestionsState.rewriteLoading ? 'disabled' : ''}>Rewrite Draft</button>
            </div>
        </div>
    `;

    const editor = document.getElementById('suggestions-draft-editor');
    editor?.addEventListener('input', (e) => {
        suggestionsState.draftText = e.target.value;
    });
    document.getElementById('suggestions-draft-cancel')?.addEventListener('click', (e) => {
        e.stopPropagation();
        resetSuggestionWorkspace();
    });
    document.getElementById('suggestions-draft-rewrite')?.addEventListener('click', (e) => {
        e.stopPropagation();
        rewriteSuggestionDraft();
    });
}

function renderSuggestionDraftPreview(container) {
    const { title } = parseSuggestionText(suggestionsState.draftSuggestion || '');
    container.innerHTML = `
        <div class="suggestions-workspace-card">
            <div class="suggestions-workspace-kicker">Composer Preview</div>
            <div class="suggestions-workspace-title">${escapeHtml(title || 'Rewritten Draft')}</div>
            <div class="suggestions-workspace-copy">This version is ready to drop into the main composer as your next message. Make any final tweaks there before sending.</div>
            <div class="suggestions-preview-output">${escapeHtml(suggestionsState.rewrittenText)}</div>
            <div class="suggestions-workspace-actions">
                <button class="suggestion-btn suggestion-btn-append" id="suggestions-preview-edit" type="button">Edit Again</button>
                <button class="suggestion-btn suggestion-btn-use" id="suggestions-preview-use" type="button">Use In Composer</button>
            </div>
        </div>
    `;

    document.getElementById('suggestions-preview-edit')?.addEventListener('click', () => {
        suggestionsState.mode = 'draft';
        updateDropdownUI();
    });
    document.getElementById('suggestions-preview-use')?.addEventListener('click', () => {
        const input = document.getElementById('chat-input');
        if (!input) return;
        input.value = suggestionsState.rewrittenText;
        delete input.dataset.suggestionDraft;
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
        autoResizeChatInput();
        window.dispatchEvent(new CustomEvent('replyPlanChanged'));
        suggestionsState.expanded = false;
        resetSuggestionWorkspace();
    });
}

function buildSuggestionRewritePrompt(tab, suggestionText, draftText) {
    const style = detectSuggestionRewriteStyle(tab);
    const charName = (tab?.ai_name || 'the AI character').trim();
    const userName = (tab?.user_name || 'the user').trim();

    const userMessages = (tab?.messages || [])
        .filter(msg => msg.role === 'user' && msg.content?.trim())
        .slice(-6)
        .map(msg => msg.content.trim());

    const recentContext = (tab?.messages || [])
        .slice(-8)
        .map(msg => `${msg.role === 'user' ? userName : charName}: ${msg.content}`)
        .join('\n\n');

    const voiceBlock = userMessages.length > 0
        ? `\nVoice samples — how ${userName} actually writes:\n${userMessages.map((m, i) => `[${i + 1}] ${m}`).join('\n\n')}`
        : '';

    const styleInstruction = style === 'first_person'
        ? `Write in first person as ${userName}. Mirror their sentence rhythm, vocabulary, and emotional cadence from the voice samples.`
        : style === 'third_person'
            ? `Write as a ${userName}-authored third-person continuation. Match the established tone, pacing, and POV from the voice samples.`
            : `Write naturally as ${userName}'s next prose turn. Convert any planning language into scene prose that matches their voice from the samples.`;

    return `You are helping ${userName} compose their next message in an ongoing creative writing exchange with ${charName}.

CRITICAL: You are writing AS ${userName}, NOT as ${charName}. Do not write ${charName}'s dialogue, actions, or thoughts. Output only what ${userName} would type next.

Task: Expand the draft notes into polished prose in ${userName}'s established voice.

Rules:
- Output ONLY ${userName}'s message — no labels, no explanations, no meta-language.
- Strip any leading label (e.g., "Next story beat:", "Plot twist:") from the suggestion. Write the actual prose only.
- Do not mention "beat", "draft", "suggestion", "POV", or any planning terms.
- Do not write as ${charName} or continue the scene from ${charName}'s perspective.
- Match ${userName}'s voice closely using the samples below.

Style: ${styleInstruction}

Length: 2–4 paragraphs, 3–8 sentences. Rich but not overlong — ${userName}'s turns should be shorter than ${charName}'s.
${voiceBlock}

Suggestion beat to expand:
${suggestionText}

${userName}'s draft notes:
${draftText}

Recent conversation:
${recentContext}`;
}

async function fetchSuggestionRewrite(prompt) {
    const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: window.authHeaders
            ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
            : { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            messages: [
                {
                    role: 'system',
                    content: 'You rewrite user-side creative-writing drafts into polished final prose. Keep all reasoning internal. Output only the rewritten final message.',
                },
                {
                    role: 'user',
                    content: prompt,
                },
            ],
            stream: true,
            temperature: 0.75,
            thinking_budget_tokens: 0,
            chat_template_kwargs: { enable_thinking: false },
            max_tokens: 1024,
        }),
    });

    if (!resp.ok) {
        const errText = await resp.text().catch(() => '');
        throw new Error(`HTTP ${resp.status}: ${errText || resp.statusText}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let output = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') continue;
            try {
                const obj = JSON.parse(payload);
                const delta = obj.choices?.[0]?.delta;
                if (delta?.content) output += delta.content;
            } catch {
                // ignore malformed chunks
            }
        }
    }

    return normalizeGeneratedMessageContent(output.trim(), activeChatTab(), 'user');
}

async function rewriteSuggestionDraft() {
    const tab = activeChatTab();
    if (!tab || !suggestionsState.draftSuggestion) return;
    const draftText = suggestionsState.draftText.trim();
    if (!draftText) {
        showToast('Add a draft note first', 'warning');
        return;
    }

    suggestionsState.rewriteLoading = true;
    updateDropdownUI();

    try {
        const prompt = buildSuggestionRewritePrompt(tab, suggestionsState.draftSuggestion, draftText);
        const rewritten = await fetchSuggestionRewrite(prompt);
        if (!rewritten) {
            throw new Error('rewrite returned empty content');
        }

        // Drop directly into the main composer and close the popup
        const input = document.getElementById('chat-input');
        if (input) {
            input.value = rewritten;
            input.focus();
            input.setSelectionRange(input.value.length, input.value.length);
            autoResizeChatInput();
        }
        suggestionsState.expanded = false;
        resetSuggestionWorkspace({ preserveResults: true });
    } catch (error) {
        showToast(`Draft rewrite failed: ${error.message}`, 'error');
        suggestionsState.rewriteLoading = false;
        updateDropdownUI();
    }
}

// ── Fetch Suggestions from API ───────────────────────────────────────────────

function getSettingsValue() {
    return {
        context_depth: settingsState.context_depth,
        suggestion_count: settingsState.suggestion_count,
        suggestion_prompts: settingsState.suggestion_prompts,
    };
}

function buildSuggestionContext(tab) {
    return {
        messages: (tab.messages || []).map(message => ({
            role: message.role,
            content: message.content,
        })),
        system_prompt: tab.system_prompt || '',
        context_notes: (tab.context_notes || [])
            .filter(note => note?.content?.trim())
            .map(note => ({
                section: note.section || 'context',
                content: note.content,
                created_at: note.created_at || 0,
            })),
        quick_guide_active: (tab.quick_guide_active || '').trim(),
    };
}

async function fetchSuggestions() {
    const tab = activeChatTab();
    if (!tab) return;

    if (chat.busy) {
        showToast('Wait for the current reply to finish before generating suggestions', 'warning');
        return;
    }

    if (suggestionsState.isOffline) {
        showToast('Offline: suggestions unavailable until connection restored', 'error');
        return;
    }

    suggestionsState.isLoading = true;
    suggestionsState.retryCount = 0;
    suggestionsState.hasGenerated = false;
    suggestionsState.setupCollapsed = true;
    updateDropdownUI();

    // Suggestions API currently reconstructs context from persisted tab state,
    // so flush first to avoid missing the latest assistant turn or mode changes.
    try {
        await persistChatTabs();
    } catch {
        suggestionsState.isLoading = false;
        suggestionsState.setupCollapsed = false;
        updateDropdownUI();
        showToast('Failed to save the latest chat state before generating suggestions', 'error');
        return;
    }

    const settings = getSettingsValue();
    const contextDepth = settings.context_depth ?? 10;
    const suggestionCount = settings.suggestion_count ?? 5;
    const prompts = settings.suggestion_prompts ?? {};
    let promptValue = prompts[suggestionsState.currentCategory];

    // Check custom categories if not found in settings prompts
    if (!promptValue) {
        const customCat = suggestionsState.customCategories.get(suggestionsState.currentCategory);
        if (customCat && typeof customCat.prompt === 'string') {
            promptValue = customCat.prompt;
        }
    }

    const prompt = typeof promptValue === 'string' && promptValue.trim()
        ? promptValue
        : null;

    await requestSuggestions({
        tabId: tab.id,
        category: suggestionsState.currentCategory,
        contextDepth,
        suggestionCount,
        prompt,
        context: buildSuggestionContext(tab),
        retryAttempt: 0,
    });
}

async function requestSuggestions({ tabId, category, contextDepth, suggestionCount, prompt, context, retryAttempt }) {
    suggestionsState.retryCount = retryAttempt;

    try {
        const response = await fetch('/api/chat/suggestions', {
            method: 'POST',
            headers: window.authHeaders
                ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                : { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tab_id: tabId,
                category,
                context_depth: contextDepth,
                count: suggestionCount,
                ...context,
                ...(prompt ? { prompt } : {}),
            }),
        });

        if (!response.ok) {
            const errText = await response.text().catch(() => '');
            throw new Error(`HTTP ${response.status}: ${errText || response.statusText}`);
        }

        const data = await response.json();
        suggestionsState.suggestions = data.suggestions || [];
        suggestionsState.isLoading = false;
        suggestionsState.hasGenerated = true;
        suggestionsState.lastError = null;
        updateDropdownUI();
    } catch (error) {
        suggestionsState.lastError = error;
        if (retryAttempt < suggestionsState.maxRetries) {
            const nextAttempt = retryAttempt + 1;
            suggestionsState.retryCount = nextAttempt;
            const delay = 1000 * nextAttempt;
            setTimeout(() => {
                requestSuggestions({
                    tabId,
                    category,
                    contextDepth,
                    suggestionCount,
                    prompt,
                    context,
                    retryAttempt: nextAttempt,
                });
            }, delay);
            showToast(`Retrying... (${nextAttempt}/${suggestionsState.maxRetries})`, 'warning');
        } else {
            suggestionsState.isLoading = false;
            suggestionsState.hasGenerated = false;
            suggestionsState.setupCollapsed = false;
            suggestionsState.suggestions = [];
            updateDropdownUI();
            showToast(`Failed to fetch suggestions: ${error.message}`, 'error');
        }
    }
}

// ── Use Suggestion ───────────────────────────────────────────────────────────

async function useSuggestion(index, mode = 'send') {
    const suggestion = suggestionsState.suggestions[index];
    if (!suggestion) return;

    if (mode === 'draft') {
        openSuggestionDraft(index);
        return;
    }

    // Rewrite the suggestion in the user's voice before placing it in the composer.
    // The suggestion text seeds both the context and the draft so the AI knows what
    // beat to express — the rewrite just re-voices it as the user, not the character.
    const tab = activeChatTab();
    if (!tab) return;

    suggestionsState.rewriteLoading = true;
    updateDropdownUI();

    try {
        // Parse the suggestion to extract just the content, stripping any leading label like "Next story beat:" or "Plot twist:"
        const { title, description } = parseSuggestionText(suggestion);
        const beat = [title, description].filter(Boolean).join('. ');
        const prompt = buildSuggestionRewritePrompt(tab, beat, beat);
        const rewritten = await fetchSuggestionRewrite(prompt);
        if (!rewritten) throw new Error('rewrite returned empty content');

        // Send the rewritten suggestion directly as a user message
        sendChatWithContent(rewritten, { guided: true });
        suggestionsState.expanded = false;
        suggestionsState.rewriteLoading = false;
        updateDropdownUI();
    } catch (err) {
        showToast(`Send Direction rewrite failed: ${err.message}`, 'error');
        suggestionsState.rewriteLoading = false;
        updateDropdownUI();
    }
}

// ── Category Management ──────────────────────────────────────────────────────

const CATEGORY_DEFAULT_PROMPTS = {
    general: 'Generate {count} versatile next-step story beats that naturally continue the current conversation — action, emotion, dialogue, or any mix that fits the moment.',
    'plot-twist': 'Generate {count} unexpected plot twists, reversals, or reveals that raise the stakes or shift the scene\'s direction entirely.',
    'new-character': 'Generate {count} ways to introduce a new character who creates conflict, chemistry, new information, or a meaningful dynamic shift.',
    director: 'Generate {count} high-level scene direction notes — pacing changes, tonal pivots, cinematic staging, or structural beats the story needs.',
    action: 'Generate {count} action-driven beats with momentum, physical danger, immediate stakes, and visceral consequence.',
    comedy: 'Generate {count} humor beats — awkward pivots, absurd escalations, well-timed punchlines, or playful character moments.',
    fantasy: 'Generate {count} fantasy-flavored beats drawing on magic systems, myth, prophecy, or setting-specific lore.',
    horror: 'Generate {count} horror beats — creeping dread, sudden menace, psychological pressure, or atmosphere that tightens the scene.',
    mystery: 'Generate {count} mystery beats — clue drops, red herrings, suspicion shifts, or revelatory moments that reframe what we know.',
    noir: 'Generate {count} noir beats — cynical dialogue, shadowy motives, moral compromise, and rain-soaked moral ambiguity.',
    romance: 'Generate {count} romance beats — charged silence, emotional vulnerability, longing, misread signals, or turning points in intimacy.',
    'sci-fi': 'Generate {count} sci-fi beats involving futuristic dilemmas, technological stakes, or speculative world logic that complicates the scene.',
    thriller: 'Generate {count} thriller beats — escalating urgency, time pressure, dangerous information, or consequences that keep tightening.',
    character: 'Generate {count} character-focused beats that reveal hidden desire, force a difficult choice, or expose an internal contradiction.',
    explicit: 'Generate {count} explicit adult-only beats for this scene, grounded in character chemistry and emotional context.',
};

function getBuiltinPromptOverride(key) {
    return (settingsState.suggestion_prompts || {})[key] || '';
}

function saveBuiltinPromptOverride(key, value) {
    if (!settingsState.suggestion_prompts) settingsState.suggestion_prompts = {};
    if (value.trim()) {
        settingsState.suggestion_prompts[key] = value.trim();
    } else {
        delete settingsState.suggestion_prompts[key];
    }
    // Persist to backend via the settings API
    import('./settings.js').then(({ saveSettings }) => saveSettings());
}

function manageCategories() {
    const modal = document.getElementById('manage-categories-modal');
    if (!modal) return;

    modal.removeAttribute('aria-hidden');
    modal.inert = false;
    modal.classList.add('open');

    renderBuiltinCategories();
    renderCustomCategories();
}

function renderBuiltinCategories() {
    const list = document.getElementById('categories-builtin-list');
    if (!list) return;

    // eslint-disable-next-line no-unsanitized/property -- User content escaped via escapeHtml()
    list.innerHTML = Object.entries(CATEGORY_META).map(([key, meta]) => {
        const override = getBuiltinPromptOverride(key);
        const hasOverride = !!override;
        const defaultPrompt = CATEGORY_DEFAULT_PROMPTS[key] || '';
        const editValue = override || defaultPrompt;
        return `
        <div class="cat-card" data-key="${escapeHtml(key)}">
            <div class="cat-card-main" role="button" tabindex="0" aria-expanded="false">
                <div class="cat-card-info">
                    <span class="cat-card-chip">${escapeHtml(meta.label)}</span>
                    <span class="cat-card-desc">${escapeHtml(meta.description)}</span>
                </div>
                <div class="cat-card-status-wrap">
                    ${hasOverride ? '<span class="cat-card-badge cat-card-badge-custom">Custom</span>' : '<span class="cat-card-badge">Default</span>'}
                    <span class="cat-card-chevron" aria-hidden="true">›</span>
                </div>
            </div>
            <div class="cat-card-editor" hidden>
                <textarea class="cat-mgr-field cat-card-textarea" data-key="${escapeHtml(key)}">${escapeHtml(editValue)}</textarea>
                <div class="cat-card-editor-actions">
                    <button class="cat-card-save" data-key="${escapeHtml(key)}">Save Override</button>
                    <button class="cat-card-reset" data-key="${escapeHtml(key)}">Reset to Default</button>
                </div>
            </div>
        </div>`;
    }).join('');

    list.querySelectorAll('.cat-card-main').forEach(mainEl => {
        const toggle = () => {
            const card = mainEl.closest('.cat-card');
            const editor = card.querySelector('.cat-card-editor');
            const opening = editor.hidden;
            editor.hidden = !opening;
            mainEl.setAttribute('aria-expanded', String(opening));
            card.classList.toggle('is-open', opening);
            if (opening) editor.querySelector('textarea')?.focus();
        };
        mainEl.addEventListener('click', toggle);
        mainEl.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
        });
    });

    list.querySelectorAll('.cat-card-save').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.key;
            const card = btn.closest('.cat-card');
            const val = card.querySelector('.cat-card-textarea')?.value || '';
            const defaultPrompt = CATEGORY_DEFAULT_PROMPTS[key] || '';
            const isCustom = val.trim() && val.trim() !== defaultPrompt.trim();
            saveBuiltinPromptOverride(key, isCustom ? val : '');
            const badge = card.querySelector('.cat-card-badge');
            if (isCustom) {
                badge.textContent = 'Custom';
                badge.classList.add('cat-card-badge-custom');
            } else {
                badge.textContent = 'Default';
                badge.classList.remove('cat-card-badge-custom');
            }
            showToast(isCustom ? 'Prompt override saved' : 'Matches default — override cleared', 'success');
        });
    });

    list.querySelectorAll('.cat-card-reset').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.key;
            const card = btn.closest('.cat-card');
            const textarea = card.querySelector('.cat-card-textarea');
            const defaultPrompt = CATEGORY_DEFAULT_PROMPTS[key] || '';
            if (textarea) textarea.value = defaultPrompt;
            saveBuiltinPromptOverride(key, '');
            const badge = card.querySelector('.cat-card-badge');
            badge.textContent = 'Default';
            badge.classList.remove('cat-card-badge-custom');
            showToast('Reset to default', 'success');
        });
    });
}

function renderCustomCategories() {
    const list = document.getElementById('categories-custom-list');
    if (!list) return;

    const customs = Array.from(suggestionsState.customCategories.entries());

    if (customs.length === 0) {
        list.innerHTML = '<div class="cat-mgr-empty">No custom categories yet. Add one below.</div>';
        return;
    }

    // eslint-disable-next-line no-unsanitized/property -- User content escaped via escapeHtml()
    list.innerHTML = customs.map(([key, catData]) => {
        const catPrompt = typeof catData === 'string' ? catData : (catData.prompt || '');
        const isExplicit = typeof catData === 'string' ? false : (catData.explicit || false);
        const label = key.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        const preview = catPrompt.length > 90 ? catPrompt.slice(0, 90) + '...' : catPrompt;
        return `
        <div class="cat-card cat-card-custom" data-key="${escapeHtml(key)}">
            <div class="cat-card-main">
                <div class="cat-card-info">
                    <span class="cat-card-chip cat-card-chip-custom">${escapeHtml(label)}</span>
                    <span class="cat-card-desc">${escapeHtml(preview)}</span>
                </div>
                <div class="cat-card-status-wrap">
                    <button class="cat-card-toggle cat-card-edit-btn" data-key="${escapeHtml(key)}" aria-expanded="false">Edit</button>
                    <button class="cat-card-delete" data-key="${escapeHtml(key)}">Delete</button>
                </div>
            </div>
            <div class="cat-card-editor" hidden>
                <textarea class="cat-mgr-field cat-card-textarea" data-key="${escapeHtml(key)}">${escapeHtml(catPrompt)}</textarea>
                <div class="cat-card-editor-explicit">
                    <button type="button" class="cat-mgr-explicit-btn ${isExplicit ? 'active' : ''}" data-key="${escapeHtml(key)}" title="Toggle explicit mode for this category">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="cat-mgr-explicit-icon">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>
                        </svg>
                        <span class="cat-mgr-explicit-text">Explicit</span>
                    </button>
                </div>
                <div class="cat-card-editor-actions">
                    <button class="cat-card-save-custom" data-key="${escapeHtml(key)}">Save</button>
                </div>
            </div>
        </div>`;
    }).join('');

    list.querySelectorAll('.cat-card-edit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const card = btn.closest('.cat-card');
            const editor = card.querySelector('.cat-card-editor');
            const isOpen = !editor.hidden;
            editor.hidden = isOpen;
            btn.setAttribute('aria-expanded', String(!isOpen));
            btn.textContent = isOpen ? 'Edit' : 'Close';
            if (!isOpen) editor.querySelector('textarea')?.focus();
        });
    });

    list.querySelectorAll('.cat-card-save-custom').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.key;
            const card = btn.closest('.cat-card');
            const val = card.querySelector('.cat-card-textarea')?.value || '';
            const isExplicit = card.querySelector('.cat-mgr-explicit-check')?.checked || false;
            if (!val.trim()) { showToast('Prompt cannot be empty', 'error'); return; }
            suggestionsState.customCategories.set(key, { prompt: val.trim(), explicit: isExplicit });
            saveCustomCategories();
            renderCustomCategories();
            renderCustomCategoryButtons();
            showToast('Category updated', 'success');
        });
    });

    list.querySelectorAll('.cat-card-delete').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.key;
            removeCategory(key);
        });
    });
}

function addCategory(name, focusKeywords, isExplicit = false) {
    if (!name || !focusKeywords) {
        showToast('Please provide both a name and focus keywords', 'error');
        return;
    }

    // Generate the template from the focus keywords
    const catPrompt = `Generate {count} ${name.toLowerCase()} story beats. Focus on ${focusKeywords}. Build naturally from the current conversation.`;

    const key = name.toLowerCase().replace(/\s+/g, '-');
    suggestionsState.customCategories.set(key, { prompt: catPrompt, explicit: isExplicit });
    saveCustomCategories();
    renderCustomCategories();
    renderCustomCategoryButtons();
    showToast(`Category "${name}" added`, 'success');
}

function removeCategory(key) {
    if (suggestionsState.customCategories.delete(key)) {
        saveCustomCategories();
        renderCustomCategories();
        renderCustomCategoryButtons();
        showToast('Category removed', 'success');
    }
}

function saveCustomCategories() {
    try {
        const data = Object.fromEntries(suggestionsState.customCategories);
        settingsState.custom_suggestion_categories = data;
        saveSettings();
    } catch (e) {
        console.error('Failed to save custom categories:', e);
    }
}

function loadCustomCategories() {
    try {
        const shared = settingsState.custom_suggestion_categories || {};
        const data = Object.keys(shared).length > 0
            ? shared
            : JSON.parse(localStorage.getItem('suggestions_custom_categories') || '{}');
        suggestionsState.customCategories = new Map(Object.entries(data));
    } catch (e) {
        console.error('Failed to load custom categories:', e);
    }
}

// ── Generate Button ──────────────────────────────────────────────────────────

function setupGenerateButton() {
    const generateBtn = document.getElementById('suggestions-generate-btn');
    if (!generateBtn) return;

    generateBtn.addEventListener('click', () => {
        fetchSuggestions();
    });

    const setupToggle = document.getElementById('suggestions-view-toggle');
    const backBtn = document.getElementById('suggestions-editor-back');
    if (setupToggle) {
        setupToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            suggestionsState.setupCollapsed = !suggestionsState.setupCollapsed;
            updateDropdownUI();
        });
    }
    backBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (suggestionsState.mode === 'preview') {
            suggestionsState.mode = 'draft';
            updateDropdownUI();
            return;
        }
        resetSuggestionWorkspace();
    });
}

// ── Category Buttons ─────────────────────────────────────────────────────────

function setupCategoryButtons() {
    document.querySelectorAll('.suggestion-category-btn').forEach(btn => {
        const previewCategory = () => {
            suggestionsState.previewCategory = btn.dataset.category;
            updatePreviewText();
        };
        const resetPreviewCategory = () => {
            suggestionsState.previewCategory = suggestionsState.currentCategory;
            updatePreviewText();
        };

        btn.addEventListener('mouseenter', previewCategory);
        btn.addEventListener('focus', previewCategory);
        btn.addEventListener('mouseleave', resetPreviewCategory);
        btn.addEventListener('blur', resetPreviewCategory);
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const category = btn.dataset.category;
            setSuggestionCategory(category);
        });
    });

    // Manage categories modal
    const manageModal = document.getElementById('manage-categories-modal');
    const manageClose = document.getElementById('manage-categories-close');
    const addCategoryBtn = document.getElementById('add-category-btn');

    if (manageClose) {
        manageClose.addEventListener('click', () => {
            manageModal.classList.remove('open');
            manageModal.setAttribute('aria-hidden', 'true');
            manageModal.inert = true;
        });
    }

    if (addCategoryBtn) {
        addCategoryBtn.addEventListener('click', () => {
            const nameInput = document.getElementById('new-category-name');
            const focusInput = document.getElementById('new-category-focus');
            const explicitBtn = document.getElementById('new-category-explicit-btn');
            const name = nameInput?.value.trim();
            const focus = focusInput?.value.trim();
            const isExplicit = explicitBtn?.classList.contains('active') || false;

            if (name && focus) {
                addCategory(name, focus, isExplicit);
                nameInput.value = '';
                focusInput.value = '';
                explicitBtn?.classList.remove('active');
            }
        });
    }

    // Toggle explicit button
    const explicitBtn = document.getElementById('new-category-explicit-btn');
    if (explicitBtn) {
        explicitBtn.addEventListener('click', () => {
            explicitBtn.classList.toggle('active');
        });
    }

    // Auto-generate focus keywords from category name
    const autoGenBtn = document.getElementById('auto-generate-focus-btn');
    if (autoGenBtn) {
        autoGenBtn.addEventListener('click', async () => {
            const nameInput = document.getElementById('new-category-name');
            const focusInput = document.getElementById('new-category-focus');
            const categoryName = nameInput?.value.trim();

            if (!categoryName) {
                showToast('Enter a category name first', 'warning', '', { duration: 2000 });
                return;
            }

            autoGenBtn.disabled = true;
            autoGenBtn.textContent = '⏳ Generating...';

            try {
                const response = await fetch('/api/keywords/generate', {
                    method: 'POST',
                    headers: window.authHeaders
                        ? { ...window.authHeaders(), 'Content-Type': 'application/json' }
                        : { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        category: categoryName,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                const data = await response.json();
                if (data.keywords && data.keywords.length > 0) {
                    focusInput.value = data.keywords.join(', ');
                }
            } catch (e) {
                console.error('Auto-generate focus failed:', e);
                showToast('Failed to generate focus keywords', 'error', '', { duration: 2000 });
            } finally {
                autoGenBtn.disabled = false;
                autoGenBtn.textContent = '✨ Auto';
            }
        });
    }

    const manageBtn = document.getElementById('suggestions-manage-btn');
    if (manageBtn) {
        manageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            manageCategories();
        });
    }
}

// ── Keyboard Navigation ─────────────────────────────────────────────────────

function setupKeyboardNav() {
    document.addEventListener('keydown', (e) => {
        if (!suggestionsState.expanded) return;
        const searchInput = document.getElementById('suggestion-search-input');
        if (document.activeElement === searchInput && !['Escape'].includes(e.key)) return;
        // Let the draft editor handle its own arrow/enter keys
        const draftEditor = document.getElementById('suggestions-draft-editor');
        if (document.activeElement === draftEditor && e.key !== 'Escape') return;

        const items = document.querySelectorAll('.suggestion-item');
        const activeItem = document.querySelector('.suggestion-item.active');
        let currentIndex = activeItem ? parseInt(activeItem.dataset.index, 10) : -1;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            items.forEach(item => {
                item.classList.remove('active');
                item.setAttribute('aria-selected', 'false');
            });
            const nextIndex = Math.min(currentIndex + 1, items.length - 1);
            if (nextIndex >= 0) {
                items[nextIndex].classList.add('active');
                items[nextIndex].setAttribute('aria-selected', 'true');
                items[nextIndex].focus();
                items[nextIndex].scrollIntoView({ block: 'nearest' });
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            items.forEach(item => {
                item.classList.remove('active');
                item.setAttribute('aria-selected', 'false');
            });
            const prevIndex = Math.max(currentIndex - 1, 0);
            if (prevIndex >= 0) {
                items[prevIndex].classList.add('active');
                items[prevIndex].setAttribute('aria-selected', 'true');
                items[prevIndex].focus();
                items[prevIndex].scrollIntoView({ block: 'nearest' });
            }
        } else if (e.key === 'Enter' && activeItem) {
            e.preventDefault();
            const index = activeItem.dataset.index || activeItem.dataset.recentIndex;
            useSuggestion(parseInt(index, 10), 'send');
        } else if (e.key === 'Tab' && activeItem) {
            e.preventDefault();
            const index = activeItem.dataset.index || activeItem.dataset.recentIndex;
            useSuggestion(parseInt(index, 10), 'send');
        } else if (e.key === 'Escape') {
            suggestionsState.expanded = false;
            updateDropdownUI();
            document.getElementById('suggestions-toggle')?.focus();
        }
    });
}

// ── Offline Detection ───────────────────────────────────────────────────────

function setupOfflineDetection() {
    window.addEventListener('online', () => {
        suggestionsState.isOffline = false;
        if (suggestionsState.expanded) {
            showToast('Back online!', 'success');
        }
    });

    window.addEventListener('offline', () => {
        suggestionsState.isOffline = true;
        showToast('You are offline', 'warning');
    });
}

// ── Click Outside to Close ──────────────────────────────────────────────────

function setupClickOutside() {
    document.addEventListener('click', (e) => {
        const dropdown = document.getElementById('suggestions-dropdown');
        const toggleBtn = document.getElementById('suggestions-toggle');

        if (!dropdown || !toggleBtn) return;

        const isClickInside = dropdown.contains(e.target) || toggleBtn.contains(e.target);

        if (!isClickInside && suggestionsState.expanded) {
            suggestionsState.expanded = false;
            suggestionsState.previewCategory = suggestionsState.currentCategory;
            updateDropdownUI();
        }
    });
}

// ── Initialization ───────────────────────────────────────────────────────────

function setupTagCloudUI() {
    const searchInput = document.getElementById('suggestion-search-input');
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            applySearchFilter();
        });
    }

    document.querySelectorAll('.category-group-header').forEach(header => {
        if (header.dataset.tagCloudBound === '1') return;
        header.dataset.tagCloudBound = '1';
        header.addEventListener('click', () => {
            const expanded = header.getAttribute('aria-expanded') === 'true';
            header.setAttribute('aria-expanded', String(!expanded));
        });
    });
}

export function initSuggestionsDropdown() {
    loadCustomCategories();
    setupGenerateButton();
    setupCategoryButtons();
    setupKeyboardNav();
    setupClickOutside();
    setupOfflineDetection();
    setupTagCloudUI();
    window.addEventListener('activeTabChanged', updateDropdownUI);
    window.addEventListener('explicitModeChanged', updateDropdownUI);
    window.addEventListener('chatReplyComplete', () => {
        if (suggestionsState.hasGenerated) {
            suggestionsState.hasGenerated = false;
            suggestionsState.suggestions = [];
            suggestionsState.setupCollapsed = false;
            if (suggestionsState.expanded) updateDropdownUI();
        }
    });
    updateDropdownUI();
    // Render custom category buttons (only on init and when categories change)
    renderCustomCategoryButtons();
    window.addEventListener('settings-applied', () => {
        loadCustomCategories();
        renderCustomCategories();
        renderCustomCategoryButtons();
        updateDropdownUI();
    });
}
