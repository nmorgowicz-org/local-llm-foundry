import { gotoApp } from '../../harness/browser.mjs';
import { setArtifactRuntime, sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';

// Four flat (non-bundle) presets with deliberately different names/sizes/backends
// so sort-by-name and sort-by-size produce visibly different card orders, and one
// preset (rapid-mlx) so the family/backend chips aren't all identical.
function gridPresets() {
    return [
        {
            id: 'orca-13b', name: 'Orca 13B Reasoner', backend: 'llama_cpp',
            model_path: '/models/orca-13b-q4_k_m.gguf', hf_repo: null,
            context_size: 32768, ctk: 'q8_0', ctv: 'q8_0', batch_size: 512, ubatch_size: 512,
            parallel_slots: 1, port: 8001, bind_host: '127.0.0.1',
            model_size_bytes: 8_200_000_000, family: 'llama', tags: ['agentic'],
        },
        {
            id: 'atlas-7b', name: 'Atlas 7B Coder', backend: 'llama_cpp',
            model_path: '/models/atlas-7b-q5_k_m.gguf', hf_repo: null,
            context_size: 65536, ctk: 'q8_0', ctv: 'f16', batch_size: 1024, ubatch_size: 512,
            parallel_slots: 1, port: 8001, bind_host: '127.0.0.1',
            model_size_bytes: 4_800_000_000, family: 'qwen', tags: ['coding'],
        },
        {
            id: 'nimbus-27b', name: 'Nimbus 27B Quality', backend: 'llama_cpp',
            model_path: '/models/nimbus-27b-q4_k_m.gguf', hf_repo: null,
            context_size: 131072, ctk: 'q8_0', ctv: 'q8_0', batch_size: 2048, ubatch_size: 512,
            parallel_slots: 1, port: 8001, bind_host: '127.0.0.1',
            model_size_bytes: 17_200_000_000, family: 'gemma', tags: ['creative'],
        },
        {
            id: 'sable-4b', name: 'Sable 4B Rapid', backend: 'rapid_mlx',
            model_path: '', hf_repo: null,
            rapid_mlx: {
                model_path: '/models/mlx-community/Sable-4B-4bit',
                model_source: { kind: 'mlx_directory', path: '/models/mlx-community/Sable-4B-4bit' },
                model_source_view: {
                    canonical_identity: '/models/mlx-community/Sable-4B-4bit',
                    display_name: 'Sable-4B-4bit',
                },
                host: '127.0.0.1', port: 9123, served_model_name: 'sable-rapid', log_level: 'info',
            },
            model_size_bytes: 2_400_000_000, family: 'other', tags: ['fast'],
        },
    ];
}

export default async function scenarioLaunchGrid(ctx) {
    const { page, baseUrl } = ctx;
    // See preset-bundle.mjs for why this clears the scenario-key prefix branch of
    // tagFilename — every raw filename below already starts with 'launch-grid'.
    setArtifactRuntime(null, 'llamacpp-local');

    // The delete-confirm shot below intentionally never clicks "Delete" (capture
    // runs must stay idempotent against this fixture data), so no dialog() auto-accept
    // is needed for that step. The grid itself never opens a native dialog either.

    const presets = gridPresets();
    // A throwaway preset dedicated to the delete-confirm shot, kept separate from
    // the four presets the sort/collections shots depend on so opening (and never
    // confirming) its trash dialog can't disturb them.
    const throwaway = {
        id: 'throwaway-delete-me', name: 'Throwaway (capture only)', backend: 'llama_cpp',
        model_path: '/models/throwaway.gguf', hf_repo: null,
        context_size: 8192, ctk: 'q8_0', ctv: 'q8_0', batch_size: 512, ubatch_size: 512,
        parallel_slots: 1, port: 8001, bind_host: '127.0.0.1', model_size_bytes: 1_000_000_000,
    };
    const allPresets = [...presets, throwaway];

    const collections = [
        { id: 'col-coding', name: 'Coding stack', preset_ids: ['atlas-7b', 'sable-4b'] },
    ];

    await page.setRequestInterception(true);
    page.on('request', request => {
        const url = new URL(request.url());
        const method = request.method();
        const respond = (status, body) => request.respond({ status, contentType: 'application/json', body: JSON.stringify(body) });

        if (url.pathname === '/api/settings') return respond(200, { preset_id: presets[0].id });
        if (url.pathname === '/api/sessions/active/readiness') return respond(200, { ok: true, ready: true });
        if (url.pathname === '/api/sessions/active') return respond(200, { status: 'Stopped', preset_id: '' });
        if (url.pathname === '/api/sessions/recent') return respond(200, { sessions: [], active_session_id: '' });
        if (url.pathname === '/api/db/admin-token') return respond(200, { token: 'capture-admin-token' });
        if (url.pathname === '/api/kill-llama') return respond(200, { ok: true });
        if (url.pathname === '/api/collections') return respond(200, { collections });
        if (url.pathname === '/api/preset-cards') {
            return respond(200, { cards: [], catalog_etag: 'catalog-v1:capture', preset_bundle_ui: 'bundled' });
        }
        if (url.pathname === '/metrics/system') return respond(200, { ram_total_gb: 128, ram_used_gb: 40, memory_reclaimable_gb: 8 });
        if (url.pathname === '/metrics/gpu') return respond(200, { gpus: [{ vram_total_mb: 131072, vram_used_mb: 40960, metal_gpu_limit_mb: 114688 }] });
        if (url.pathname === '/api/system/metal-gpu-limit') return respond(200, { ok: true, limit_mb: 114688 });
        if (url.pathname === '/api/llama-binary/platform-info') return respond(200, { ok: true, auto_backend: 'metal' });
        if (url.pathname === '/api/memory-availability') {
            return respond(200, { ok: true, snapshot: { current_safe_availability_bytes: 29_000_000_000, configured_ceiling_bytes: 120_259_084_288, total_unified_bytes: 137_438_953_472 } });
        }
        if (url.pathname === '/api/vram-estimate') {
            return respond(200, { ok: true, total_bytes: 12_300_000_000, weights_bytes: 8_200_000_000, kv_cache_bytes: 3_000_000_000, overhead_bytes: 1_100_000_000 });
        }
        if (url.pathname === '/api/app-home-migration/status') return respond(200, { migration_required: false });
        if (url.pathname === '/api/presets' && method === 'GET') return respond(200, allPresets);

        request.continue();
    });

    await gotoApp(page, baseUrl);
    await page.waitForSelector('.launch-card[data-preset-id]', { timeout: 10000 });
    await sleep(400);

    // 1: default grid, sort switched to "Name" so the shot's card order is
    // deterministic and matches what a viewer reading top-to-bottom expects.
    await page.select('#setup-filter-sort', 'name');
    await sleep(250);
    await captureShot(page, 'launch-grid-sort-name.png', { fullPage: true });

    // 2: same grid, re-sorted by size (largest first).
    await page.select('#setup-filter-sort', 'size');
    await sleep(250);
    await captureShot(page, 'launch-grid-sort-size.png', { fullPage: true });

    // 3: collections filter narrows the grid down to the "Coding stack" collection.
    await page.select('#setup-filter-collections', 'col-coding');
    await sleep(250);
    await captureShot(page, 'launch-grid-collections-filtered.png', { fullPage: true });
    await page.select('#setup-filter-collections', 'all');
    await sleep(250);

    // 4: running badge. dashboard-ws.js drives sessionState.serverRunning /
    // activeSessionPresetId off the app's real WebSocket connection, which this
    // capture harness (like the Playwright specs — see launch-grid.spec.js) can't
    // fake through route/request interception. Set the state directly and call the
    // same exported render function the live WS path calls.
    await page.evaluate(async presetId => {
        const { sessionState } = await import('/js/core/app-state.js');
        const { updateRunningCardHighlight } = await import('/js/features/setup-view.js');
        sessionState.serverRunning = true;
        sessionState.activeSessionPresetId = presetId;
        updateRunningCardHighlight();
    }, 'nimbus-27b');
    await sleep(300);
    await captureShot(page, 'launch-grid-running-badge.png', { fullPage: true });

    // Reset running state before the delete-confirm shot so it doesn't carry the
    // pulsing "running" border into an unrelated capture.
    await page.evaluate(async () => {
        const { sessionState } = await import('/js/core/app-state.js');
        const { updateRunningCardHighlight } = await import('/js/features/setup-view.js');
        sessionState.serverRunning = false;
        sessionState.activeSessionPresetId = '';
        updateRunningCardHighlight();
    });
    await sleep(150);

    // 5: per-card delete confirm dialog, open but never confirmed — the dedicated
    // throwaway preset means never clicking "Delete" here leaves the other four
    // presets' fixture data untouched for any capture step that runs after this one.
    await page.click('.launch-card[data-preset-id="throwaway-delete-me"] .launch-card-btn-trash');
    await page.waitForSelector('.app-confirm-dialog', { visible: true });
    await sleep(250);
    await captureShot(page, 'launch-grid-delete-confirm.png', { fullPage: true });
}
