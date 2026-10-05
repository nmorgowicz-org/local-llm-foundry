// chat-search.js — FTS cross-session search
// Provides a visible "Search messages" entry point near the title filter and
// renders a larger flyout beside the sidebar for full-text search results.

import { chat } from '../core/app-state.js';
import Router from './router.js';
import { escapeHtml } from '../core/format.js';

const SEARCH_PAGE_SIZE = 20;

let _searchActive = false;
let _searchInput = null;
let _searchTimer = null;
let _searchList = null;
let _searchCount = null;
let _searchPanel = null;
let _searchLaunchBtn = null;
let _searchSummary = null;
let _searchLoadMoreBtn = null;
let _searchStatusPill = null;
let _collapsedBeforeSearch = false;
let _searchQuery = '';
let _searchOffset = 0;
let _searchLoading = false;
let _searchVisibility = ['active'];

export function initChatSearch() {
    const panel = document.getElementById('chat-sessions-panel');
    const searchWrap = document.querySelector('.csp-search-wrap');
    const page = document.getElementById('page-chat');
    if (!panel || !searchWrap || !page) return;

    const launch = document.createElement('div');
    launch.className = 'csp-search-launch';
    launch.innerHTML =
        `<button class="csp-search-launch-btn" id="csp-message-search-btn" type="button">
            <span class="csp-search-launch-icon" aria-hidden="true">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <circle cx="11" cy="11" r="7"/>
                    <path d="M20 20l-3.5-3.5"/>
                </svg>
            </span>
            <span class="csp-search-launch-copy">
                <span class="csp-search-launch-title">Search Messages</span>
                <span class="csp-search-launch-subtitle">Full text across every chat</span>
            </span>
            <span class="csp-search-launch-pill" id="csp-message-search-pill">FTS</span>
        </button>`;
    searchWrap.after(launch);
    _searchLaunchBtn = launch.querySelector('.csp-search-launch-btn');
    _searchStatusPill = launch.querySelector('.csp-search-launch-pill');
    _searchLaunchBtn?.addEventListener('click', () => openSearch());

    _searchPanel = document.createElement('section');
    _searchPanel.className = 'csp-search-panel hidden';
    _searchPanel.setAttribute('aria-label', 'Message search');
    _searchPanel.innerHTML =
        `<div class="csp-search-panel-header">
            <div class="csp-search-panel-copy">
                <div class="csp-search-panel-eyebrow">Message Search</div>
                <div class="csp-search-panel-title">Find exact lines across your chat history</div>
                <div class="csp-search-panel-subtitle">Search message bodies, not just conversation titles.</div>
            </div>
            <button class="csp-search-close" type="button" title="Close search">✕</button>
        </div>
        <div class="csp-search-input-wrap">
            <span class="csp-search-input-icon" aria-hidden="true">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3">
                    <circle cx="11" cy="11" r="7"/>
                    <path d="M20 20l-3.5-3.5"/>
                </svg>
            </span>
            <input id="csp-search-input" type="search" class="csp-search-input" placeholder="Search messages, phrases, or fragments…" autocomplete="off">
        </div>
        <div class="csp-search-summary-row">
            <span class="csp-search-count">Type 2+ letters</span>
            <span class="csp-search-summary">Use the title filter for chat names, and message search for history.</span>
        </div>
        <div class="csp-search-filters">
            <button class="csp-search-filter-chip active" data-visibility="active" type="button">Active</button>
            <button class="csp-search-filter-chip" data-visibility="archived" type="button">Archived</button>
        </div>
        <div class="csp-search-results"></div>
        <div class="csp-search-footer">
            <button class="csp-search-load-more" type="button">Show More Results</button>
        </div>`;
    page.appendChild(_searchPanel);

    _searchInput = _searchPanel.querySelector('#csp-search-input');
    _searchList = _searchPanel.querySelector('.csp-search-results');
    _searchCount = _searchPanel.querySelector('.csp-search-count');
    _searchSummary = _searchPanel.querySelector('.csp-search-summary');
    _searchLoadMoreBtn = _searchPanel.querySelector('.csp-search-load-more');

    _searchInput?.addEventListener('input', onSearchInput);
    _searchPanel.querySelector('.csp-search-close')?.addEventListener('click', closeSearch);
    _searchLoadMoreBtn?.addEventListener('click', loadMoreResults);
    _searchPanel.querySelectorAll('.csp-search-filter-chip').forEach((chip) => {
        chip.addEventListener('click', () => onVisibilityChipClick(chip));
    });

    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'f') {
            e.preventDefault();
            openSearch();
            return;
        }
        if (e.key === 'Escape' && _searchActive) {
            closeSearch();
        }
    });

    document.addEventListener('click', (e) => {
        if (!_searchActive || !_searchPanel) return;
        if (_searchPanel.contains(e.target) || _searchLaunchBtn?.contains(e.target)) return;
        closeSearch();
    });

    window.addEventListener('resize', () => {
        if (_searchActive) updateSearchPanelPosition();
    });
}

export function openSearch() {
    const panel = document.getElementById('chat-sessions-panel');
    if (!panel || !_searchPanel) return;

    _collapsedBeforeSearch = panel.classList.contains('collapsed')
        || localStorage.getItem('csp-collapsed') === 'true';
    if (!panel.classList.contains('visible')) {
        panel.classList.add('visible');
    }
    panel.classList.remove('collapsed');

    _searchActive = true;
    panel.classList.add('search-active');
    _searchLaunchBtn?.classList.add('active');
    updateSearchPanelPosition();
    _searchPanel.classList.remove('hidden');
    _searchInput?.focus();
    _searchInput?.select();
    renderIdleState();
}

export function closeSearch() {
    if (!_searchActive) return;
    _searchActive = false;

    const panel = document.getElementById('chat-sessions-panel');
    panel?.classList.remove('search-active');
    _searchPanel?.classList.add('hidden');
    _searchLaunchBtn?.classList.remove('active');

    if (_searchInput) _searchInput.value = '';
    _searchQuery = '';
    _searchOffset = 0;
    _searchVisibility = ['active'];
    chat.visibilityUi.activeSearchVisibility = ['active'];
    _searchLoading = false;
    if (_searchList) _searchList.innerHTML = '';
    if (_searchLoadMoreBtn) _searchLoadMoreBtn.style.display = 'none';
    if (_searchStatusPill) _searchStatusPill.textContent = 'FTS';
    updateSearchCount('Type 2+ letters');
    if (_searchSummary) {
        _searchSummary.textContent = 'Use the title filter for chat names, and message search for history.';
    }

    clearTimeout(_searchTimer);

    if (_collapsedBeforeSearch) {
        panel?.classList.add('collapsed');
    }
    _collapsedBeforeSearch = false;
}

async function onSearchInput() {
    const q = _searchInput?.value?.trim();
    clearTimeout(_searchTimer);
    _searchQuery = q || '';
    _searchOffset = 0;

    if (!q || q.length < 2) {
        renderIdleState();
        return;
    }

    _searchTimer = setTimeout(async () => {
        try {
            renderLoadingState(q);
            await fetchSearchPage(q, { offset: 0, append: false });
        } catch (e) {
            console.error('[chat-search] search failed:', e);
            renderErrorState();
        }
    }, 300);
}

async function loadMoreResults() {
    if (_searchLoading || !_searchQuery) return;
    try {
        await fetchSearchPage(_searchQuery, { offset: _searchOffset, append: true });
    } catch (e) {
        console.error('[chat-search] load more failed:', e);
        renderErrorState();
    }
}

function onVisibilityChipClick(chip) {
    const vis = chip.dataset.visibility;
    const idx = _searchVisibility.indexOf(vis);
    if (idx >= 0) {
        if (_searchVisibility.length > 1) {
            _searchVisibility.splice(idx, 1);
            chip.classList.remove('active');
        }
    } else {
        _searchVisibility.push(vis);
        chip.classList.add('active');
    }
    chat.visibilityUi.activeSearchVisibility = [..._searchVisibility];
    _searchOffset = 0;
    if (_searchQuery && _searchQuery.length >= 2) {
        fetchSearchPage(_searchQuery, { offset: 0, append: false });
    }
}

async function fetchSearchPage(query, { offset, append }) {
    _searchLoading = true;
    if (_searchLoadMoreBtn) {
        _searchLoadMoreBtn.disabled = true;
        _searchLoadMoreBtn.textContent = append ? 'Loading…' : 'Show More Results';
    }
    const resp = await fetch(
        `/api/chat/search?q=${encodeURIComponent(query)}&limit=${SEARCH_PAGE_SIZE}&offset=${offset}&visibility=${encodeURIComponent(_searchVisibility.join(','))}`,
        { headers: window.authHeaders ? window.authHeaders() : {} },
    );
    _searchLoading = false;
    if (!resp.ok) {
        renderErrorState();
        return;
    }
    const page = await resp.json();
    renderResults(page, query, { append });
}

function renderResults(page, query, { append = false } = {}) {
    if (!_searchList) return;
    const results = page?.results || [];
    const total = Number(page?.total || 0);
    const offset = Number(page?.offset || 0);
    _searchOffset = offset + results.length;

    if (!append && results.length === 0) {
        updateSearchCount('0 matches');
        if (_searchSummary) {
            _searchSummary.textContent = 'No message history matched this phrase.';
        }
        _searchList.innerHTML =
            `<div class="csp-search-empty">
                <div class="csp-search-empty-title">No matches found</div>
                <div class="csp-search-empty-body">Try a broader phrase or a shorter fragment than “${escapeHtml(query)}”.</div>
            </div>`;
        if (_searchLoadMoreBtn) _searchLoadMoreBtn.style.display = 'none';
        return;
    }

    updateSearchCount(`${total} ${total === 1 ? 'match' : 'matches'}`);
    if (_searchSummary) {
        const shownBefore = append ? _searchList.querySelectorAll('.csp-search-result').length : 0;
        const shown = Math.min(shownBefore + results.length, total);
        _searchSummary.textContent = `Showing ${shown} of ${total} matches for “${query}”.`;
    }

    const frag = document.createDocumentFragment();
    for (const r of results) {
        const card = document.createElement('div');
        card.className = 'csp-search-result';

        const roleLabel = r.role === 'assistant' ? 'AI' : 'You';
        const rawSnippet = r.snippet || '';
        const snippet = (typeof window.DOMPurify !== 'undefined')
            ? window.DOMPurify.sanitize(rawSnippet, { ALLOWED_TAGS: ['mark'] })
            : escapeHtml(rawSnippet);

        // eslint-disable-next-line no-unsanitized/property
        card.innerHTML =
            `<div class="csp-search-result-header">
                <div class="csp-search-result-heading">
                    <span class="csp-search-result-tab">${escapeHtml(r.tab_name)}</span>
                    <span class="csp-search-result-role">${roleLabel}</span>
                </div>
                <span class="csp-search-result-jump">Open Match</span>
            </div>
            <div class="csp-search-result-meta">${formatTimestamp(r.timestamp_ms)}</div>
            <div class="csp-search-result-snippet">${snippet}</div>`;

        card.addEventListener('click', () => {
            // Route through the Router so the chat view, active session, and URL all
            // stay in sync, then scroll to the matched message once it has rendered.
            Router.navigate('/chat/' + encodeURIComponent(r.tab_id));
            setTimeout(() => {
                const el = document.querySelector(`.chat-message[data-msg-id="${r.message_id}"]`);
                if (el) {
                    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    el.classList.add('chat-msg-highlight');
                    setTimeout(() => el.classList.remove('chat-msg-highlight'), 2000);
                }
            }, 450);
            closeSearch();
        });

        frag.appendChild(card);
    }

    if (!append) _searchList.innerHTML = '';
    _searchList.appendChild(frag);

    const hasMore = Boolean(page?.has_more);
    if (_searchLoadMoreBtn) {
        _searchLoadMoreBtn.disabled = false;
        _searchLoadMoreBtn.textContent = 'Show More Results';
        _searchLoadMoreBtn.style.display = hasMore ? 'inline-flex' : 'none';
    }
    if (_searchStatusPill) {
        _searchStatusPill.textContent = total > 0 ? `${total}` : 'FTS';
    }
}

function renderIdleState() {
    if (!_searchList) return;
    updateSearchCount('Type 2+ letters');
    if (_searchSummary) {
        _searchSummary.textContent = 'Use the title filter for chat names, and message search for history.';
    }
    _searchList.innerHTML =
        `<div class="csp-search-empty csp-search-empty-idle">
            <div class="csp-search-empty-title">Search across messages</div>
            <div class="csp-search-empty-body">Find exact moments, themes, or fragments from any chat tab. Use the title filter when you only need to narrow the sidebar list.</div>
        </div>`;
    if (_searchLoadMoreBtn) _searchLoadMoreBtn.style.display = 'none';
}

function renderLoadingState(query) {
    if (!_searchList) return;
    updateSearchCount('Searching…');
    if (_searchSummary) {
        _searchSummary.textContent = `Scanning message history for “${query}”.`;
    }
    _searchList.innerHTML =
        `<div class="csp-search-empty csp-search-empty-loading">
            <div class="csp-search-empty-title">Searching</div>
            <div class="csp-search-empty-body">Looking for “${escapeHtml(query)}” across your conversation history.</div>
        </div>`;
    if (_searchLoadMoreBtn) _searchLoadMoreBtn.style.display = 'none';
}

function renderErrorState() {
    if (!_searchList) return;
    updateSearchCount('Search unavailable');
    if (_searchSummary) {
        _searchSummary.textContent = 'The message index could not be queried right now.';
    }
    _searchList.innerHTML =
        `<div class="csp-search-empty csp-search-empty-error">
            <div class="csp-search-empty-title">Search unavailable</div>
            <div class="csp-search-empty-body">The message index could not be queried right now. Try again in a moment.</div>
        </div>`;
    if (_searchLoadMoreBtn) _searchLoadMoreBtn.style.display = 'none';
}

function updateSearchCount(text) {
    if (_searchCount) {
        _searchCount.textContent = text;
    }
}

function updateSearchPanelPosition() {
    const page = document.getElementById('page-chat');
    const panel = document.getElementById('chat-sessions-panel');
    if (!_searchPanel || !page || !panel) return;

    const pageRect = page.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    const left = Math.max(panelRect.right - pageRect.left + 12, 24);
    const width = Math.max(Math.min(560, pageRect.width - left - 16), 320);

    _searchPanel.style.left = `${left}px`;
    _searchPanel.style.top = '12px';
    _searchPanel.style.width = `${width}px`;
    _searchPanel.style.height = `${Math.max(pageRect.height - 24, 360)}px`;
}

function formatTimestamp(timestampMs) {
    if (!timestampMs) return 'Stored message';
    try {
        return new Intl.DateTimeFormat([], {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: '2-digit',
        }).format(new Date(timestampMs));
    } catch {
        return 'Stored message';
    }
}

