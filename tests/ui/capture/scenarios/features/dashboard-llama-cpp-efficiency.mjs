// Synthetic telemetry exercises production renderers, not live inference behavior.
import { gotoApp } from '../../harness/browser.mjs';
import { DEFAULT_VIEWPORT, sleep } from '../../harness/paths.mjs';
import { captureShot, suppressToasts } from '../../harness/shot.mjs';

const METRICS = {
    prompt_tokens_cached_total: 72000,
    prompt_tokens_processed_total: 24000,
    speculative_enabled: true,
    speculative_draft_tokens_total: 12000,
    speculative_accepted_tokens_total: 9000,
    speculative_verification_steps_total: 3000,
    runtime_facts: {
        model_name: 'reported-model-Q4_K_M.gguf', model_alias: '200k-emm-five', model_params: 27320000000,
        quantization: 'Q4_K - Medium', server_build: 'b11436-b9a5a00b8',
        capabilities: { audio: false, vision: true, video: true, supports_tools: true,
            supports_tool_calls: true, supports_parallel_tool_calls: true, supports_object_arguments: true,
            supports_preserve_reasoning: true, supports_reasoning_effort: true, supports_string_content: true,
            supports_system_role: true, supports_typed_content: true },
        adapters: [],
    },
};

async function render(page, metrics) {
    await page.evaluate(async data => {
        const { renderLlamaCppDetails } = await import('/js/features/llama-cpp-details.js');
        renderLlamaCppDetails({
            backend: 'llama_cpp', attached: true, sessionId: 'capture-efficiency',
            endpoint: 'http://127.0.0.1:8001', endpointTag: 'http://127.0.0.1:8001',
            attachedModel: 'attached.gguf',
            metrics: data ? { telemetry_session_id: 'capture-efficiency',
                telemetry_endpoint: 'http://127.0.0.1:8001', ...data } : data,
        });
    }, metrics);
}

// Crop the real layout, without moving cards or including unrelated lower panels.
async function captureRegion(page, selectors, filename) {
    await page.evaluate(sel => document.querySelector(sel).scrollIntoView({ behavior: 'instant', block: 'start' }), selectors[0]);
    await sleep(300);
    const clip = await page.evaluate(sels => {
        const boxes = sels.map(sel => {
            const element = document.querySelector(sel);
            if (!element || element.hidden) throw new Error(`Missing capture region: ${sel}`);
            return element.getBoundingClientRect();
        });
        const x = Math.max(0, Math.min(...boxes.map(b => b.left)) - 12);
        const y = Math.max(0, Math.min(...boxes.map(b => b.top)) - 12);
        const right = Math.min(innerWidth, Math.max(...boxes.map(b => b.right)) + 12);
        const bottom = Math.max(...boxes.map(b => b.bottom)) + 12;
        if (bottom > innerHeight) throw new Error(`Capture region exceeds viewport: ${JSON.stringify({ selectors: sels, x, y, right, bottom, height: innerHeight })}`);
        return { x: x + scrollX, y: y + scrollY, width: right - x, height: bottom - y };
    }, selectors);
    await captureShot(page, filename, { fullPage: false, expandSelector: '#server-header', clip });
}

export default async function({ page, baseUrl }) {
    console.log('[CAPTURE] dashboard-llama-cpp-efficiency: DETERMINISTIC FAKE telemetry; no model or live attachment.');
    // Install before navigation: no connection, messages, or reconnect timer can
    // replace the fixture. Keep the transport in CONNECTING rather than failing it.
    await page.evaluateOnNewDocument(() => {
        class HeldWebSocket extends EventTarget {
            static CONNECTING = 0;
            static OPEN = 1;
            static CLOSING = 2;
            static CLOSED = 3;
            constructor(url) { super(); this.url = String(url); this.readyState = 0; }
            send() {}
            close() { this.readyState = 3; }
        }
        window.WebSocket = HeldWebSocket;
    });
    await page.setViewport({ ...DEFAULT_VIEWPORT, width: 1280, height: 1400 });
    await gotoApp(page, baseUrl);
    await suppressToasts(page);
    await page.evaluate(async () => {
        const { switchView } = await import('/js/features/setup-view.js');
        const { updateMetricCards, updateStateCard } = await import('/js/features/metric-cards.js');
        switchView('monitor');
        document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === 'server'));
        document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-server'));
        document.documentElement.dataset.theme = 'dark';
        const view = {
            state: 'generating', stateLabel: 'Generating', stateDetail: 'Deterministic capture · attached.gguf',
            stateProgress: 0.5, decodeTps: 48, prefillTps: 960, running: 1, waiting: 0, queued: 0,
            gpu: { name: 'NVIDIA RTX 4090', load: 72, temp: 62, power: 220, powerLimit: 450,
                vramUsed: 8 * 1024 ** 3, vramTotal: 24 * 1024 ** 3, unifiedTotal: 24 * 1024 ** 3 },
            sys: { cpu: 24, cpuName: 'AMD Ryzen 9' },
        };
        // Populate hardware histories with a fixed sequence. Speed sampling is
        // time-throttled in production; advance its clock locally, then restore it.
        const now = Date.now;
        try {
            for (let i = 0; i < 6; i++) {
                Date.now = () => 1800000000000 + i * 1100;
                updateMetricCards({ ...view, decodeTps: 38 + i * 2, prefillTps: 860 + i * 20 });
            }
        } finally { Date.now = now; }
        updateStateCard(view);
        document.getElementById('server-endpoint').value = 'http://127.0.0.1:8001';
    });
    await render(page, METRICS);
    await page.evaluate(() => {
        if (document.getElementById('cache-reuse-value').textContent !== '75.0%'
            || document.getElementById('spec-acceptance-value').textContent !== '75.0%'
            || document.getElementById('spec-tokens-per-verification').textContent !== '4.00') {
            throw new Error('Efficiency fixture does not match the production renderer contract');
        }
        const facts = document.getElementById('llama-runtime-facts').textContent;
        for (const value of ['reported-model-Q4_K_M.gguf', '200k-emm-five', 'Q4_K - Medium', 'b11436-b9a5a00b8', 'None']) {
            if (!facts.includes(value)) throw new Error(`Missing runtime fact: ${value}`);
        }
    });
    await page.evaluate(async metrics => {
        const state = await import('/js/core/app-state.js');
        const { refreshTopCockpit } = await import('/js/features/nav.js');
        state.setWsData({ backend: 'llama_cpp', endpoint_kind: 'Local',
            active_session_id: 'capture-efficiency', active_session_status: 'running',
            active_session_endpoint_tag: 'http://127.0.0.1:8001',
            active_session_model_identity: '200k-emm-five' });
        state.chat.tabs = [{ messages: [{ role: 'assistant', input_tokens: 3000, output_tokens: 1300 }], total_output_tokens: 1300 }];
        state.setContextCapacityTokens(10000);
        state.setLastLlamaMetrics({ ...metrics, telemetry_session_id: 'capture-efficiency',
            telemetry_endpoint: 'http://127.0.0.1:8001', last_generation_tokens_per_sec: 7,
            last_prompt_tokens_per_sec: 131, tokens_per_decode: 1.95 });
        refreshTopCockpit();
    }, METRICS);
    await page.waitForFunction(() => {
        const strip = document.getElementById('endpoint-strip-monitor');
        return strip && strip.getBoundingClientRect().height > 0;
    });
    await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
    await captureRegion(page, ['.top-nav-bar'], 'dashboard-llama-cpp-efficiency-compact-cockpit.png');
    await page.setViewport({ ...DEFAULT_VIEWPORT, width: 1280, height: 1400 });
    await captureRegion(page, ['#endpoint-strip-monitor'], 'dashboard-llama-cpp-efficiency-model-badge.png');
    await page.click('#engine-indicator');
    await captureRegion(page, ['#engine-model-popover'], 'dashboard-llama-cpp-efficiency-loaded-model.png');
    await page.click('#engine-model-popover button');
    await sleep(1200);
    await captureRegion(page, ['#server-header', '#inference-efficiency'], 'dashboard-llama-cpp-efficiency-dark-overview.png');
    await captureRegion(page, ['#metrics-grid'], 'dashboard-llama-cpp-efficiency-dark-runtime-card.png');
    await page.click('#llama-runtime-details-button');
    await captureRegion(page, ['#llama-runtime-popover'], 'dashboard-llama-cpp-efficiency-runtime-details.png');
    await page.keyboard.press('Escape');
    await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
    await captureRegion(page, ['#metrics-grid'], 'dashboard-llama-cpp-efficiency-light-runtime-card.png');

    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
    await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
    await captureRegion(page, ['#nav-cockpit'], 'dashboard-llama-cpp-efficiency-narrow-cockpit.png');
    await captureRegion(page, ['#llama-runtime-card'], 'dashboard-llama-cpp-efficiency-narrow-runtime-reduced-motion.png');
    await captureRegion(page, ['#inference-efficiency'], 'dashboard-llama-cpp-efficiency-narrow-efficiency-reduced-motion.png');
    await page.setViewport({ ...DEFAULT_VIEWPORT, width: 1280, height: 1400 });
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'no-preference' }]);

    await render(page, { ...METRICS, prompt_tokens_cached_total: 0, prompt_tokens_processed_total: 0,
        speculative_draft_tokens_total: 0, speculative_accepted_tokens_total: 0, speculative_verification_steps_total: 0 });
    await page.evaluate(() => {
        if (document.getElementById('cache-reuse-status').textContent !== 'Awaiting activity'
            || document.getElementById('spec-effectiveness-status').textContent !== 'Awaiting activity') {
            throw new Error('Present zero counters must await activity');
        }
    });
    await captureRegion(page, ['#inference-efficiency'], 'dashboard-llama-cpp-efficiency-awaiting-activity.png');
    await render(page, { ...METRICS, speculative_enabled: false });
    await page.evaluate(() => {
        if (!document.getElementById('speculative-effectiveness').hidden
            || document.getElementById('prompt-cache-reuse').hidden) {
            throw new Error('Disabled speculation must leave only the cache card');
        }
    });
    await captureRegion(page, ['#inference-efficiency'], 'dashboard-llama-cpp-efficiency-cache-only.png');
    await render(page, { runtime_facts: METRICS.runtime_facts });
    await page.evaluate(() => {
        if (!document.getElementById('inference-efficiency').hidden) throw new Error('Missing counters must hide efficiency cards');
    });
}
