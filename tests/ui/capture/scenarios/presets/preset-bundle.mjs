import { gotoApp } from '../../harness/browser.mjs';
import { setArtifactRuntime, sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';

// A deterministic seeded Q4/Q5 bundle, shaped like the real server's
// materialize_default_projection output (see tests/ui/core/preset-flow.spec.js
// bundlePreset()). default_selection.context_size (200704) is intentionally
// absent from context_options so the drawer's default state exercises the
// derived "Custom" context path, matching the real fixture this was ported from.
function bundlePreset() {
    return {
        id: 'bundled',
        name: 'Bundled preset',
        model_path: '/models/qwen3-27b-q4_k_m.gguf',
        hf_repo: null,
        context_size: 200704,
        ctk: 'q8_0',
        ctv: 'q8_0',
        batch_size: 2048,
        ubatch_size: 512,
        parallel_slots: 1,
        port: 8001,
        bind_host: '127.0.0.1',
        n_cpu_moe: 0,
        revision: 7,
        bundle: {
            identity: { bundle_id: 'qwen3-27b', tune_id: 'brainwaves-wfh', display_name: 'Qwen3.8 27B · Brainwaves WFH' },
            artifacts: [
                {
                    id: 'art-q4', role: 'weights', display_name: 'Qwen3.8-27B-Q4_K_M.gguf', local_path: '/models/qwen3-27b-q4_k_m.gguf',
                    hf_origin: null, size_bytes: 17179869184, digest: null,
                    quantization: { value: 'q4_k_m', provenance: 'filename' }, metadata: {}, extra: {},
                },
                {
                    id: 'art-q5', role: 'weights', display_name: 'Qwen3.8-27B-Q5_K_M.gguf', local_path: '/models/qwen3-27b-q5_k_m.gguf',
                    hf_origin: null, size_bytes: 20401094656, digest: null,
                    quantization: { value: 'q5_k_m', provenance: 'filename' }, metadata: {}, extra: {},
                },
            ],
            context_options: [160000, 200000, 262144],
            kv_policy_options: ['f16_f16', 'q8_0_q8_0', 'q4_0_q4_0'],
            performance_options: [
                { id: 'quality', label: 'Quality first', batch_size: 2048, ubatch_size: 512 },
                { id: 'balanced', label: 'Balanced', batch_size: 512, ubatch_size: 512 },
            ],
            n_cpu_moe_options: [0, 8, 16],
            selections: [],
            workload_policy: 'general_chat',
            default_selection: {
                artifact_id: 'art-q4',
                context_size: 200704,
                kv_policy: 'q8_0_q8_0',
                performance_id: 'quality',
                batch_size: 2048,
                ubatch_size: 512,
                n_cpu_moe: 0,
            },
            extra: {},
        },
    };
}

const DEFAULT_ESTIMATE = {
    status: 'available',
    estimate: { weights_bytes: 17179869184, kv_cache_bytes: 8_000_000_000, total_bytes: 26_300_000_000, headroom_bytes: 2_700_000_000 },
};

const NO_FIT_ESTIMATE = {
    status: 'available',
    estimate: { weights_bytes: 30_000_000_000, kv_cache_bytes: 10_000_000_000, total_bytes: 42_000_000_000, headroom_bytes: -12_000_000_000 },
};

export default async function scenarioPresetBundle(ctx) {
    const { page, baseUrl } = ctx;
    // Every raw filename below starts with the scenario key 'preset-bundle',
    // which would otherwise double-prefix via tagFilename's scenario-key
    // branch. Clearing the scenario key (keeping the runtime tag) makes each
    // captureShot resolve to the plan's exact 'llamacpp-local--preset-bundle-*'
    // filenames.
    setArtifactRuntime(null, 'llamacpp-local');

    // Reset/Edit-full and the low-VRAM/revision-conflict steps below dirty the
    // draft, so closing afterward raises a native confirm() dialog. Auto-accept
    // it (== the user choosing "Discard"), same as any Puppeteer capture must.
    page.on('dialog', async dialog => {
        try { await dialog.accept(); } catch { /* already handled */ }
    });

    const preset = bundlePreset();
    // Controls which state the mocked resolve/selection routes return next;
    // flipped between capture steps below.
    let resolveMode = 'default';

    await page.setRequestInterception(true);
    page.on('request', request => {
        const url = new URL(request.url());
        const method = request.method();
        const respond = (status, body) => request.respond({ status, contentType: 'application/json', body: JSON.stringify(body) });

        if (url.pathname === '/api/settings') return respond(200, { preset_id: preset.id });
        if (url.pathname === '/api/sessions/active/readiness') return respond(200, { ok: true, ready: true });
        if (url.pathname === '/api/sessions/active') return respond(200, { status: 'Stopped', preset_id: '' });
        if (url.pathname === '/api/sessions/recent') return respond(200, { sessions: [], active_session_id: '' });
        if (url.pathname === '/api/db/admin-token') return respond(200, { token: 'capture-admin-token' });
        if (url.pathname === '/api/kill-llama') return respond(200, { ok: true });
        if (url.pathname === '/metrics/system') return respond(200, { ram_total_gb: 128, ram_used_gb: 40, memory_reclaimable_gb: 8 });
        if (url.pathname === '/metrics/gpu') return respond(200, { gpus: [{ vram_total_mb: 131072, vram_used_mb: 40960, metal_gpu_limit_mb: 114688 }] });
        if (url.pathname === '/api/system/metal-gpu-limit') return respond(200, { ok: true, limit_mb: 114688 });
        if (url.pathname === '/api/llama-binary/platform-info') return respond(200, { ok: true, auto_backend: 'metal' });
        if (url.pathname === '/api/memory-availability') {
            return respond(200, { ok: true, snapshot: { current_safe_availability_bytes: 29_000_000_000, configured_ceiling_bytes: 120_259_084_288, total_unified_bytes: 137_438_953_472 } });
        }
        if (url.pathname === '/api/vram-estimate') {
            return respond(200, { ok: true, total_bytes: 26_300_000_000, weights_bytes: 17_179_869_184, kv_cache_bytes: 8_000_000_000, overhead_bytes: 1_120_130_816 });
        }
        if (url.pathname === '/api/sessions/spawn' && method === 'POST') {
            return respond(200, { ok: true, session_id: 'capture-session', backend: 'llama_cpp', port: 8001 });
        }
        if (url.pathname === '/api/attach') return respond(200, { ok: true });
        if (url.pathname === '/api/app-home-migration/status') return respond(200, { migration_required: false });

        if (url.pathname === '/api/preset-cards' && method === 'GET') {
            return respond(200, {
                cards: [{
                    id: preset.id,
                    name: preset.name,
                    revision: preset.revision,
                    artifacts: preset.bundle.artifacts.map(a => ({ id: a.id, role: a.role, display_name: a.display_name, available: true, quantization: a.quantization.value })),
                }],
                catalog_etag: 'catalog-v1:capture',
                preset_bundle_ui: 'bundled',
            });
        }

        if (url.pathname === '/api/presets' && method === 'GET') {
            return respond(200, [preset]);
        }

        if (url.pathname.endsWith('/resolve') && method === 'POST') {
            const body = JSON.parse(request.postData() || '{}');
            const selection = body.selection || preset.bundle.default_selection;
            const isLowVramIntent = selection.intent_source === 'low_vram';
            let outSelection = selection;
            let estimate = DEFAULT_ESTIMATE;
            let evidence = null;
            if (resolveMode === 'low_vram' && isLowVramIntent) {
                outSelection = { ...selection, artifact_id: 'art-q4', kv_policy: 'q4_0_q4_0', context_size: 160000 };
            } else if (resolveMode === 'no_fit') {
                estimate = NO_FIT_ESTIMATE;
            } else if (resolveMode === 'evidence_exact') {
                evidence = { class: 'exact', summary: 'Measured on this machine, 2026-08-30.' };
            } else if (resolveMode === 'evidence_related') {
                evidence = { class: 'related', summary: 'Measured on a related Q5_K_M configuration on this machine.' };
            }
            const payload = {
                ok: true,
                selection: outSelection,
                changes: [],
                estimate,
                capability_reasons: [],
                evidence,
                selection_hash: 'sel-v1:capture',
                resolved_config_hash: 'cfg-v1:capture',
                revision: preset.revision,
            };
            // Stale-response protection: in low_vram mode, the pre-intent
            // resolve (fired automatically when the drawer opens) is delayed
            // past the fast intent-click resolve that follows it. The drawer's
            // own generation guard (requestResolve/applyResolve) must keep the
            // newer low-VRAM state and discard this once it lands late.
            const delay = (resolveMode === 'low_vram' && !isLowVramIntent) ? 700 : 0;
            setTimeout(() => respond(200, payload), delay);
            return;
        }

        if (url.pathname.endsWith('/selection') && method === 'PATCH') {
            if (resolveMode === 'revision_conflict') {
                return respond(409, { ok: false, code: 'revision_conflict', error: 'preset changed elsewhere' });
            }
            const body = JSON.parse(request.postData() || '{}');
            preset.revision += 1;
            preset.bundle.default_selection = { ...(body.selection || {}) };
            return respond(200, { ok: true, preset, revision: preset.revision });
        }

        request.continue();
    });

    await gotoApp(page, baseUrl);
    await page.waitForSelector('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure', { timeout: 10000 });

    const openDrawer = async () => {
        await page.click('.launch-card[data-preset-id="bundled"] .launch-card-btn-configure');
        await page.waitForSelector('#bundle-drawer.open', { visible: true });
        await sleep(350);
    };
    const closeDrawerCapture = async () => {
        await page.click('.bundle-drawer-close');
        await page.waitForFunction(() => !document.getElementById('bundle-drawer')?.classList.contains('open'));
        await sleep(250);
    };
    // .bundle-drawer-body scrolls internally (overflow-y: auto) inside a
    // fixed-height panel, so fullPage captures never reveal content below the
    // fold there. Scroll the Predicted result row (and diff row, when present)
    // into view before every drawer capture so the screenshot actually shows
    // the content it exists to prove.
    const revealResult = () => page.evaluate(() => {
        const scroller = document.querySelector('.bundle-drawer-body');
        if (!scroller) return;
        const diff = document.querySelector('.bundle-row-diff');
        const target = diff || document.querySelector('.bundle-row-result');
        if (target) scroller.scrollTop = target.offsetTop - 16;
        else scroller.scrollTop = scroller.scrollHeight;
    });

    // 1-2: bundled grid, dark then light.
    await sleep(400);
    await captureShot(page, 'preset-bundle-grid-dark.png', { fullPage: true });
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
    await sleep(250);
    await captureShot(page, 'preset-bundle-grid-light.png', { fullPage: true });
    await page.evaluate(() => document.documentElement.removeAttribute('data-theme'));
    await sleep(150);

    // .bundle-drawer-panel has no max-height cap and is height:100% of the
    // fixed-inset drawer, so — unlike the wizard modal's capped scroll region
    // — giving it a taller (still whitelisted) viewport lets the Predicted
    // result / diff rows lay out fully in-frame instead of past the fold of
    // an internally-scrolled container that fullPage:true can't reveal.
    await page.setViewport({ width: 1280, height: 1400, deviceScaleFactor: 1 });
    await sleep(150);

    // 3: drawer default, dark.
    await openDrawer();
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-default-dark.png', { fullPage: true });

    // 4: drawer, light.
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
    await sleep(250);
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-light.png', { fullPage: true });
    await page.evaluate(() => document.documentElement.removeAttribute('data-theme'));
    await sleep(150);

    // 5: drawer, narrow bottom sheet.
    await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
    await sleep(250);
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-narrow.png', { fullPage: true });
    await page.setViewport({ width: 1280, height: 1400, deviceScaleFactor: 1 });
    await sleep(150);

    // 6: drawer, reduced motion.
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    await sleep(150);
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-reduced-motion.png', { fullPage: true });
    await page.emulateMediaFeatures([]);
    await closeDrawerCapture();

    // 7: Low VRAM change explanation.
    resolveMode = 'low_vram';
    await openDrawer();
    await page.click('.bundle-intent[data-intent="low_vram"]');
    await sleep(900);
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-low-vram-changes.png', { fullPage: true });
    await closeDrawerCapture();

    // 8: invalid/no-fit state.
    resolveMode = 'no_fit';
    await openDrawer();
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-no-fit.png', { fullPage: true });
    await closeDrawerCapture();

    // 9: exact evidence.
    resolveMode = 'evidence_exact';
    await openDrawer();
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-evidence-exact.png', { fullPage: true });
    await closeDrawerCapture();

    // 10: related evidence.
    resolveMode = 'evidence_related';
    await openDrawer();
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-evidence-related.png', { fullPage: true });
    await closeDrawerCapture();

    // 11: revision conflict on Save.
    resolveMode = 'revision_conflict';
    await openDrawer();
    await page.click('.bundle-ctx[data-context="160000"]');
    await sleep(350);
    // page.click's simulated mouse event unreliably misses the footer Save
    // button here; dispatch a real DOM click instead.
    await page.evaluate(() => document.querySelector('.bundle-save')?.click());
    await page.waitForSelector('.toast-with-actions', { visible: true });
    await sleep(200);
    await revealResult();
    await captureShot(page, 'preset-bundle-drawer-revision-conflict.png', { fullPage: true });
}
