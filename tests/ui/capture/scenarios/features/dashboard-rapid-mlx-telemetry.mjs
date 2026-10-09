// DETERMINISTIC FAKE DATA: these are synthetic renderer fixtures, NOT real
// Rapid-MLX telemetry, performance evidence, or proof of runtime capability.
// Only the app UI is served. No runtime connection, model attach, or inference.
import { gotoApp } from '../../harness/browser.mjs';
import { DEFAULT_VIEWPORT, sleep } from '../../harness/paths.mjs';
import { captureShot, suppressToasts } from '../../harness/shot.mjs';

const GiB = 1024 ** 3;
const SAMPLE = {
    model: 'Synthetic Rapid capture fixture',
    running_requests: 2,
    waiting_requests: 1,
    generation_tokens_per_second: 48,
    prompt_tokens_per_second: 960,
    completed_requests_total: 120,
    active_memory_bytes: 6 * GiB,
    peak_memory_bytes: 8 * GiB,
    cache_memory_bytes: 2 * GiB,
    backend_details: {
        memory_limit_bytes: 16 * GiB,
        // Generic progress deliberately disagrees with observed output budget.
        // It must never become prompt progress or completion prediction.
        progress: 0.9,
        telemetry: {
            speculative_acceptance_rate: 0.75,
            succeeded_requests_total: 114,
            failed_requests_total: 4,
            cancelled_requests_total: 2,
        },
    },
    cache_metrics: {
        hit_rate: 0.8, hits: 80, misses: 20, entry_count: 24,
        current_memory_bytes: GiB / 2,
        multimodal_cache_kinds: ['image', 'audio'],
    },
};
const REQUESTS = [
    { id: 'fake-rapid-request-generating-000001', phase: 'generating',
        prompt_tokens: 4096, completion_tokens: 128, max_tokens: 512,
        cached_tokens: 3072, tokens_per_second: 48, ttft_s: 0.4,
        elapsed_s: 3.2, cache_hit_type: 'prefix' },
    { id: 'fake-rapid-request-reading-000002', phase: 'reading',
        prompt_tokens: 8192, cached_tokens: 2048, elapsed_s: 2.5,
        cache_hit_type: 'multimodal' },
    { id: 'fake-rapid-request-queued-000003', status: 'queued', prompt_tokens: 1024 },
];
const OVERVIEW = { ...SAMPLE, active_requests: REQUESTS };
// The synthetic `telemetry_unavailable` flag is the renderer's own "no usable poll" input.
// It carries no numbers, so nothing here can be mistaken for measured telemetry.
const UNAVAILABLE = { model: SAMPLE.model, telemetry_unavailable: true };
// Partial runtime report: only queue counts and active memory. Optional MTP/cache cards
// must stay hidden rather than show invented zeros.
const PARTIAL = { model: SAMPLE.model, running_requests: 0, waiting_requests: 0,
    active_memory_bytes: GiB, active_requests: [] };
const READING = { ...SAMPLE, running_requests: 1, waiting_requests: 0,
    active_requests: [REQUESTS[1]] };
const GENERATING = { ...SAMPLE, running_requests: 1, waiting_requests: 0,
    active_requests: [REQUESTS[0]] };

// Poll a node-side condition. Used instead of page.waitForFunction(async ...): an async
// predicate returns a Promise, which is always truthy, so it would resolve immediately.
async function pollUntil(description, condition, timeoutMs = 15000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await condition()) return;
        await sleep(50);
    }
    throw new Error(`Timed out waiting for ${description}`);
}

// Exercise the real JSON parsing, app-state ingestion, normalization, shared
// cards, and request table. The held socket never opens or reaches a server.
// `ready` is a frame-specific condition evaluated in the page: the previous check ("details
// visible and any row exists") was already true after the first frame, so later frames
// could be captured before they rendered. Selectors are ids only so markup changes to the
// table or fact cards do not break the wait.
async function inject(page, inference, ready) {
    await page.evaluate(sample => {
        const socket = window.__rapidCaptureSockets.find(ws =>
            new URL(ws.url).pathname === '/ws' && typeof ws.onmessage === 'function');
        if (!socket) throw new Error('Dashboard WebSocket handler not ready');
        socket.onmessage(new MessageEvent('message', { data: JSON.stringify({
            backend: 'rapid_mlx', session_mode: 'attach',
            active_session_id: 'synthetic-rapid-telemetry-capture',
            active_session_status: 'running',
            active_session_endpoint: 'http://rapid-telemetry-fixture.invalid:8001',
            active_session_endpoint_tag: 'synthetic-rapid-telemetry-capture',
            active_session_model_identity: 'Synthetic Rapid capture fixture',
            endpoint_kind: 'Local', server_running: true, local_server_running: false,
            inference: sample, llama: null,
            // Keep real host hardware out of these portable fixtures. Runtime
            // memory is backend-reported fixture data, not a hardware estimate.
            host_metrics_available: false, capabilities: {}, gpu: {}, system: null,
            logs: [], mode: 'off',
        }) }));
    }, inference);
    try {
        await page.waitForFunction(({ label, rows, cardsHidden = [], cardsShown = [] }) => {
            if (document.getElementById('rapid-dashboard-details')?.hidden) return false;
            if (window.__rapidDataRows() !== rows) return false;
            if (label && document.getElementById('state-label')?.textContent !== label) return false;
            return cardsHidden.every(key => document.getElementById(`mc-card-${key}`)?.hidden === true)
                && cardsShown.every(key => document.getElementById(`mc-card-${key}`)?.hidden === false);
        }, { timeout: 15000 }, ready);
    } catch (error) {
        const seen = await page.evaluate(() => ({
            label: document.getElementById('state-label')?.textContent,
            rows: window.__rapidDataRows(),
            detailsHidden: document.getElementById('rapid-dashboard-details')?.hidden,
            cards: Object.fromEntries(['mtp', 'runtime-memory', 'cache'].map(key =>
                [key, document.getElementById(`mc-card-${key}`)?.hidden])),
        }));
        throw new Error(`${error.message}; wanted ${JSON.stringify(ready)}, saw ${JSON.stringify(seen)}`);
    }
}

// Preserve production geometry, including the table's horizontal scroll region.
async function captureRegion(page, selectors, suffix) {
    await page.evaluate(sel => document.querySelector(sel).scrollIntoView({
        behavior: 'instant', block: 'start',
    }), selectors[0]);
    await sleep(300);
    // Freeze existing CSS animations at a fixed phase for reproducible stills,
    // without changing production CSS or substituting fixture markup.
    // Finite animations (entrance/transition) jump to their end state: freezing them at t=0
    // would capture a half-translated "from" frame. Only looping animations are frozen.
    await page.evaluate(() => {
        document.getAnimations().forEach(animation => {
            if (animation.effect?.getComputedTiming().iterations !== Infinity) {
                animation.finish();
                return;
            }
            animation.pause();
            animation.currentTime = animation.animationName === 'state-reading-activity' ? 800 : 0;
        });
    });
    const clip = await page.evaluate(sels => {
        const boxes = sels.map(sel => {
            const el = document.querySelector(sel);
            if (!el || el.hidden || !el.getBoundingClientRect().height) {
                throw new Error(`Missing capture region: ${sel}`);
            }
            return el.getBoundingClientRect();
        });
        const x = Math.max(0, Math.min(...boxes.map(box => box.left)) - 12);
        const y = Math.max(0, Math.min(...boxes.map(box => box.top)) - 12);
        const right = Math.min(innerWidth, Math.max(...boxes.map(box => box.right)) + 12);
        const bottom = Math.max(...boxes.map(box => box.bottom)) + 12;
        if (bottom > innerHeight || right <= x) throw new Error('Capture region exceeds viewport');
        return { x: x + scrollX, y: y + scrollY, width: right - x, height: bottom - y };
    }, selectors);
    await captureShot(page, `dashboard-rapid-mlx-telemetry-${suffix}.png`, {
        fullPage: false, expandSelector: '#rapid-dashboard-details', clip,
    });
}

export default async function({ page, baseUrl }) {
    console.log('[CAPTURE] dashboard-rapid-mlx-telemetry: DETERMINISTIC FAKE telemetry; held WS, no runtime connection or model inference.');
    // Install before navigation so no real frames/open callbacks can replace
    // fixtures or trigger restore-hint attachment. No close/reconnect events.
    await page.evaluateOnNewDocument(() => {
        window.__rapidCaptureSockets = [];
        // Request rows carrying data, excluding the single full-width placeholder row the table
        // shows when nothing is reported. Depends on the #rapid-request-rows id only.
        window.__rapidDataRows = () => [...document.querySelectorAll('#rapid-request-rows tr')]
            .filter(tr => !(tr.children.length === 1 && tr.children[0].colSpan > 1)).length;
        class HeldWebSocket extends EventTarget {
            static CONNECTING = 0;
            static OPEN = 1;
            static CLOSING = 2;
            static CLOSED = 3;
            constructor(url) {
                super();
                this.url = String(url);
                this.readyState = HeldWebSocket.CONNECTING;
                window.__rapidCaptureSockets.push(this);
            }
            send() {}
            close() { this.readyState = HeldWebSocket.CLOSED; }
        }
        window.WebSocket = HeldWebSocket;
    });
    await page.setViewport({ ...DEFAULT_VIEWPORT, width: 1280, height: 1400 });
    await gotoApp(page, baseUrl);
    await suppressToasts(page);
    await page.evaluate(async () => {
        const { switchView } = await import('/js/features/setup-view.js');
        switchView('monitor');
        document.querySelectorAll('.sidebar-btn').forEach(button =>
            button.classList.toggle('active', button.dataset.tab === 'server'));
        document.querySelectorAll('.page').forEach(panel =>
            panel.classList.toggle('active', panel.id === 'page-server'));
        document.documentElement.dataset.theme = 'dark';
    });
    await pollUntil('monitor view to finish switching', () => page.evaluate(async () =>
        (await import('/js/core/app-state.js')).setupViewState.view === 'monitor'));
    await page.waitForFunction(() => window.__rapidCaptureSockets?.some(ws =>
        new URL(ws.url).pathname === '/ws' && typeof ws.onmessage === 'function'));

    await inject(page, OVERVIEW, { rows: 3, cardsShown: ['mtp', 'runtime-memory', 'cache'] });
    await page.evaluate(() => {
        for (const key of ['mtp', 'runtime-memory', 'cache']) {
            const card = document.getElementById(`mc-card-${key}`);
            if (!card || card.hidden) throw new Error(`Missing Rapid shared card: ${key}`);
        }
        const mtp = document.getElementById('mc-card-mtp').textContent;
        if (!mtp.includes('Not a speedup measurement')) {
            throw new Error('MTP acceptance must remain distinct from a speedup measurement');
        }
        if (window.__rapidDataRows() !== 3) {
            throw new Error('Overview fixture must render three real table rows');
        }
        if (document.getElementById('mv-decode').textContent !== (48).toLocaleString(undefined, {
            minimumFractionDigits: 1, maximumFractionDigits: 1,
        })) {
            throw new Error('Rapid throughput did not reach shared metric cards');
        }
    });
    await captureRegion(page, ['#state-card', '#metrics-grid', '#rapid-dashboard-details'], 'dark-overview');

    await inject(page, READING, { label: 'Reading', rows: 1 });
    await page.evaluate(() => {
        const detail = document.getElementById('state-detail').textContent;
        if (document.getElementById('state-label').textContent !== 'Reading'
            || !detail.includes(`${(8192).toLocaleString()} prompt tokens`) || !detail.includes('2.5s elapsed')) {
            throw new Error('Reading fixture must show reported prompt size and elapsed time');
        }
        const progress = document.getElementById('state-progress');
        if (progress.dataset.indeterminate !== 'true' || progress.hasAttribute('aria-valuenow')) {
            throw new Error('Reading must not use generic backend progress as a token fraction');
        }
    });
    await captureRegion(page, ['#state-card'], 'reading-indeterminate');

    await inject(page, GENERATING, { label: 'Generating', rows: 1 });
    await page.evaluate(() => {
        const detail = document.getElementById('state-detail').textContent;
        if (document.getElementById('state-label').textContent !== 'Generating'
            || !detail.includes('128 / 512 tokens') || !detail.includes('output budget used')
            || !(Math.abs(parseFloat(document.getElementById('state-bar').style.width) - 25) <= 0.5)) {
            throw new Error('Generation must show observed output/max, not generic progress');
        }
    });
    await captureRegion(page, ['#state-card'], 'generation-observed-output');

    // Telemetry unavailable: the renderer must say so and show no request rows or optional facts.
    await inject(page, UNAVAILABLE, { label: 'Telemetry unavailable', rows: 0, cardsHidden: ['mtp', 'cache'] });
    await page.evaluate(() => {
        if (!document.getElementById('state-detail').textContent.includes('Request activity has not been reported')) {
            throw new Error('Unavailable telemetry must state that request activity was not reported');
        }
        if (window.__rapidDataRows() !== 0) {
            throw new Error('Unavailable telemetry must not render request rows');
        }
    });
    await captureRegion(page, ['#state-card', '#metrics-grid', '#rapid-dashboard-details'], 'telemetry-unavailable');

    // Partial report: runtime memory present, optional MTP and cache cards absent.
    await inject(page, PARTIAL, { rows: 0, cardsHidden: ['mtp', 'cache'], cardsShown: ['runtime-memory'] });
    await captureRegion(page, ['#state-card', '#metrics-grid', '#rapid-dashboard-details'], 'partial-card');

    await inject(page, OVERVIEW, { rows: 3, cardsShown: ['mtp', 'runtime-memory', 'cache'] });
    await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
    await captureRegion(page, ['#state-card', '#metrics-grid', '#rapid-dashboard-details'], 'light-overview');

    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
    await page.evaluate(() => {
        document.documentElement.dataset.theme = 'dark';
        document.querySelector('.rapid-request-scroll').scrollLeft = 0;
    });
    // At 430x900 the whole details panel is taller than the contract viewport, so capture the
    // heading plus the horizontally scrollable table region (the part this still documents).
    await captureRegion(page, ['#rapid-requests-heading', '.rapid-request-scroll'], 'narrow-table-reduced-motion');
}
