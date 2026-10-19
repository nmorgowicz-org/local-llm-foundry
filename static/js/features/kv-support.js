// Shared client for GET /api/llama-binary/kv-support.
//
// Every surface that edits a main K/V cache pair (setup wizard Guided + Pro,
// preset editor, configure drawer, launch chip) goes through this module, so
// the same context always renders the same evidence and text. The server
// owns the classification and the wording; this module only fetches, caches,
// discards stale answers and draws them. Server text is inserted with
// textContent, never innerHTML.

const STATES = ['native', 'fallback', 'unverified', 'wont_start'];

const BADGE = {
    native: { icon: '✓', label: 'Fused attention' },
    fallback: { icon: '⚠', label: 'Slower path' },
    unverified: { icon: '?', label: 'Not verified' },
    wont_start: { icon: '✖', label: "Won't start" },
};

const CACHE_TTL_MS = 30_000;
const DEFAULT_DEBOUNCE_MS = 250;

const cache = new Map(); // request key -> { at, result }
const inflight = new Map(); // request key -> Promise<result|null>
let knownRuntime = null; // `${runtime_identity}|${basis_version}` of the last response
let generation = 0; // bumped by invalidate(); late responses from older generations are dropped

/** Request key: everything the caller controls. Identities come back in the response. */
export function kvKey(ctx) {
    return JSON.stringify([
        String(ctx.k || '').toLowerCase(),
        String(ctx.v || '').toLowerCase(),
        String(ctx.fa || '').toLowerCase(),
        ctx.modelId || '',
        ctx.deviceId || '',
    ]);
}

/**
 * Models are identified to the server by a path relative to the models
 * directory. Returns '' when the path is not inside it (the capability is still
 * useful; only the model-specific evidence and advisories are omitted).
 */
export function modelIdFromPath(path, modelsDir) {
    if (!path || !modelsDir) return '';
    const norm = (p) => String(p).replace(/\\/g, '/').replace(/\/+$/, '');
    const root = norm(modelsDir);
    const full = norm(path);
    const prefix = `${root}/`;
    const startsWith = /^[a-z]:/i.test(root)
        ? full.toLowerCase().startsWith(prefix.toLowerCase())
        : full.startsWith(prefix);
    if (!startsWith) return '';
    const rel = full.slice(prefix.length);
    if (!rel || rel.split('/').some((part) => part === '..' || part === '')) return '';
    return rel;
}

let modelsDirPromise = null;
export function getModelsDir() {
    if (!modelsDirPromise) {
        modelsDirPromise = (async () => {
            try {
                const headers = window.authHeaders ? window.authHeaders() : {};
                const res = await fetch('/api/hf/download-dir', { headers });
                if (!res.ok) return '';
                const body = await res.json();
                return body.dir || '';
            } catch {
                return '';
            }
        })();
    }
    return modelsDirPromise;
}

export async function modelIdForPath(path) {
    return modelIdFromPath(path, await getModelsDir());
}

/** Drop every cached answer and discard responses that are still in flight. */
export function invalidateKvSupport() {
    generation += 1;
    cache.clear();
    inflight.clear();
    knownRuntime = null;
}

function noteRuntime(result) {
    const identity = `${result.runtime_identity ?? ''}|${result.basis_version ?? ''}`;
    if (knownRuntime !== null && knownRuntime !== identity) {
        // The loaded runtime changed (plugin swap, update, rebuild): nothing
        // learned under the old identity may be reused.
        cache.clear();
    }
    knownRuntime = identity;
}

/**
 * Fetch one classification. Resolves null on any failure: a failed lookup is
 * never rendered as support, and never blocks anything.
 */
export function fetchKvSupport(ctx, { force = false } = {}) {
    const key = kvKey(ctx);
    const hit = cache.get(key);
    if (!force && hit && Date.now() - hit.at < CACHE_TTL_MS) return Promise.resolve(hit.result);
    if (inflight.has(key)) return inflight.get(key);

    const startedIn = generation;
    const params = new URLSearchParams({ k: ctx.k || '', v: ctx.v || '', fa: ctx.fa || '' });
    if (ctx.modelId) params.set('model_id', ctx.modelId);
    if (ctx.deviceId) params.set('device_id', ctx.deviceId);

    const request = (async () => {
        try {
            const headers = window.authHeaders ? window.authHeaders() : {};
            const res = await fetch(`/api/llama-binary/kv-support?${params}`, { headers });
            if (!res.ok) return null;
            const result = await res.json();
            if (!result || result.ok !== true || !STATES.includes(result.support?.kind)) return null;
            if (startedIn !== generation) return null; // invalidated while in flight
            noteRuntime(result);
            cache.set(key, { at: Date.now(), result });
            return result;
        } catch {
            return null;
        } finally {
            if (inflight.get(key) === request) inflight.delete(key);
        }
    })();
    inflight.set(key, request);
    return request;
}

function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
}

function compiledLine(result) {
    const compiled = result.compiled_pair;
    if (compiled?.state !== 'confirmed') return '';
    return compiled.value
        ? 'A fused kernel for this pair is compiled into this runtime.'
        : 'No fused kernel for this pair is compiled into this runtime.';
}

/**
 * Draw one classification into `container`. `null` clears it (no evidence is
 * shown rather than a guess). Returns the state kind or ''.
 */
export function renderKvSupport(container, result) {
    if (!container) return '';
    container.replaceChildren();
    container.removeAttribute('data-kv-state');
    container.hidden = !result;
    if (!result) {
        container.className = 'kv-support';
        return '';
    }
    const kind = result.support.kind;
    container.className = `kv-support kv-support--${kind}`;
    container.dataset.kvState = kind;
    container.setAttribute('role', kind === 'wont_start' ? 'alert' : 'status');

    const head = el('div', 'kv-support-head');
    const badge = el('span', 'kv-support-badge');
    badge.append(el('span', 'kv-support-icon', BADGE[kind].icon), el('span', 'kv-support-label', BADGE[kind].label));
    head.append(badge);
    if (result.reason) head.append(el('span', 'kv-support-reason', result.reason));
    container.append(head);

    if (kind === 'wont_start' && result.fix) {
        container.append(el('div', 'kv-support-fix', result.fix));
    }

    const compiled = compiledLine(result);
    if (compiled && kind !== 'fallback') {
        container.append(el('div', 'kv-support-compiled', compiled));
    }

    // Only a confirmed missing, buildable CUDA kernel earns a rebuild hint; the
    // Runtime panel (source builds) attaches its action to this marker.
    if (result.buildable_missing) {
        const hint = el(
            'div',
            'kv-support-rebuild',
            'A source build of llama.cpp compiles every K/V pair.',
        );
        hint.dataset.kvRebuild = 'true';
        container.append(hint);
    }

    const advisories = Array.isArray(result.advisories) ? result.advisories : [];
    if (advisories.length) {
        const list = el('ul', 'kv-support-advisories');
        for (const advisory of advisories) {
            const item = el('li', `kv-support-advisory kv-support-advisory--${advisory.severity || 'info'}`, advisory.text);
            item.dataset.advisoryId = advisory.id || '';
            if (advisory.source) item.title = advisory.source;
            list.append(item);
        }
        container.append(list);
    }
    return kind;
}

/**
 * One controller per surface. `getContext()` returns the live
 * `{k, v, fa, modelId, deviceId}`; `update()` is debounced, always revalidates
 * (so a plugin or model change shows up even for a cached key), and drops any
 * answer whose context changed while it was in flight.
 */
export function createKvSupportController({ container, getContext, onResult, debounceMs = DEFAULT_DEBOUNCE_MS }) {
    let seq = 0;
    let timer = null;
    let last = null;

    async function run() {
        const ctx = await getContext();
        const mine = ++seq;
        if (!ctx || !ctx.k || !ctx.v) {
            last = null;
            renderKvSupport(container, null);
            onResult?.(null, ctx);
            return null;
        }
        const key = kvKey(ctx);
        const cached = cache.get(key);
        if (cached) {
            // Show what we know now, then revalidate below.
            last = cached.result;
            renderKvSupport(container, cached.result);
        }
        const result = await fetchKvSupport(ctx, { force: true });
        if (mine !== seq) return last; // a newer update superseded this one
        const now = await getContext();
        if (!now || kvKey(now) !== key) return last; // context moved on while waiting
        last = result;
        renderKvSupport(container, result);
        onResult?.(result, ctx);
        return result;
    }

    return {
        update() {
            clearTimeout(timer);
            return new Promise((resolve) => {
                timer = setTimeout(() => resolve(run()), debounceMs);
            });
        },
        updateNow() {
            clearTimeout(timer);
            return run();
        },
        cancel() {
            clearTimeout(timer);
            seq += 1;
        },
        get last() {
            return last;
        },
    };
}

/** True when the classification forbids Start / Save & Start. Plain Save stays allowed. */
export function blocksStart(result) {
    return result?.support?.kind === 'wont_start' || result?.blocking === true;
}

export const KV_STATES = STATES;
