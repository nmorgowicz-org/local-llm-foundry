import { gotoApp } from '../../harness/browser.mjs';
import { setArtifactRuntime, sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';

// Two bundled presets in one collection so the welcome grid's name, quant
// badges, context, KV policy, and the collections filter are all visible in
// a single shot — this is the README's showcase hero image for the bundled
// preset system, not a functional-coverage capture.
function heroPresets() {
    return [
        {
            id: 'qwen3-27b-brainwaves', name: 'Qwen3.8 27B · Brainwaves WFH', model_path: '/models/qwen3-27b-q4_k_m.gguf',
            hf_repo: null, context_size: 200704, ctk: 'q8_0', ctv: 'q8_0', batch_size: 2048, ubatch_size: 512,
            parallel_slots: 1, port: 8001, bind_host: '127.0.0.1', n_cpu_moe: 0, revision: 1,
            model_size_bytes: 17_179_869_184, family: 'qwen',
            bundle: {
                identity: { bundle_id: 'qwen3-27b', tune_id: 'brainwaves-wfh', display_name: 'Qwen3.8 27B · Brainwaves WFH' },
                artifacts: [{
                    id: 'art-q4', role: 'weights', display_name: 'Qwen3.8-27B-Q4_K_M.gguf', local_path: '/models/qwen3-27b-q4_k_m.gguf',
                    hf_origin: null, size_bytes: 17179869184, digest: null,
                    quantization: { value: 'q4_k_m', provenance: 'filename' }, metadata: {}, extra: {},
                }],
                context_options: [160000, 200000, 262144], kv_policy_options: ['f16_f16', 'q8_0_q8_0', 'q4_0_q4_0'],
                performance_options: [{ id: 'quality', label: 'Quality first', batch_size: 2048, ubatch_size: 512 }],
                n_cpu_moe_options: [0, 8, 16], selections: [], workload_policy: 'general_chat',
                default_selection: {
                    artifact_id: 'art-q4', context_size: 200704, kv_policy: 'q8_0_q8_0',
                    performance_id: 'quality', batch_size: 2048, ubatch_size: 512, n_cpu_moe: 0,
                },
                extra: {},
            },
        },
        {
            id: 'llama-8b-agent-desk', name: 'Llama 3.1 8B · Agent Desk', model_path: '/models/llama-8b-q5_k_m.gguf',
            hf_repo: null, context_size: 131072, ctk: 'q8_0', ctv: 'q8_0', batch_size: 1024, ubatch_size: 512,
            parallel_slots: 1, port: 8002, bind_host: '127.0.0.1', n_cpu_moe: 0, revision: 1,
            model_size_bytes: 5_700_000_000, family: 'llama',
            bundle: {
                identity: { bundle_id: 'llama-8b', tune_id: 'agent-desk', display_name: 'Llama 3.1 8B · Agent Desk' },
                artifacts: [{
                    id: 'art-q5', role: 'weights', display_name: 'Llama-3.1-8B-Q5_K_M.gguf', local_path: '/models/llama-8b-q5_k_m.gguf',
                    hf_origin: null, size_bytes: 5_700_000_000, digest: null,
                    quantization: { value: 'q5_k_m', provenance: 'filename' }, metadata: {}, extra: {},
                }],
                context_options: [65536, 131072], kv_policy_options: ['f16_f16', 'q8_0_q8_0'],
                performance_options: [{ id: 'balanced', label: 'Balanced', batch_size: 1024, ubatch_size: 512 }],
                n_cpu_moe_options: [0], selections: [], workload_policy: 'agentic',
                default_selection: {
                    artifact_id: 'art-q5', context_size: 131072, kv_policy: 'q8_0_q8_0',
                    performance_id: 'balanced', batch_size: 1024, ubatch_size: 512, n_cpu_moe: 0,
                },
                extra: {},
            },
        },
    ];
}

export default async function scenarioWelcomeHero(ctx) {
    const { page, baseUrl } = ctx;
    setArtifactRuntime(null, 'welcome-hero');

    const presets = heroPresets();
    const collections = [
        { id: 'col-agentic', name: 'Agentic workflows', preset_ids: ['llama-8b-agent-desk'] },
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
        if (url.pathname === '/api/preset-cards' && method === 'GET') {
            return respond(200, {
                cards: presets.map(p => ({
                    id: p.id, name: p.name, revision: p.revision,
                    artifacts: p.bundle.artifacts.map(a => ({ id: a.id, role: a.role, display_name: a.display_name, available: true, quantization: a.quantization.value })),
                })),
                catalog_etag: 'catalog-v1:capture', preset_bundle_ui: 'bundled',
            });
        }
        if (url.pathname === '/metrics/system') return respond(200, { ram_total_gb: 128, ram_used_gb: 40, memory_reclaimable_gb: 8 });
        if (url.pathname === '/metrics/gpu') return respond(200, { gpus: [{ vram_total_mb: 131072, vram_used_mb: 40960, metal_gpu_limit_mb: 114688 }] });
        if (url.pathname === '/api/system/metal-gpu-limit') return respond(200, { ok: true, limit_mb: 114688 });
        if (url.pathname === '/api/llama-binary/platform-info') return respond(200, { ok: true, auto_backend: 'metal' });
        if (url.pathname === '/api/memory-availability') {
            return respond(200, { ok: true, snapshot: { current_safe_availability_bytes: 29_000_000_000, configured_ceiling_bytes: 120_259_084_288, total_unified_bytes: 137_438_953_472 } });
        }
        if (url.pathname === '/api/vram-estimate') {
            // Kept comfortably under the unified-memory "fit" threshold (0.82 *
            // available) for this shot — it's a positive showcase image, not a
            // low-headroom functional case (see preset-bundle.mjs for that).
            return respond(200, { ok: true, total_bytes: 20_000_000_000, weights_bytes: 17_179_869_184, kv_cache_bytes: 1_800_000_000, overhead_bytes: 1_020_130_816 });
        }
        if (url.pathname === '/api/app-home-migration/status') return respond(200, { migration_required: false });
        if (url.pathname === '/api/presets' && method === 'GET') return respond(200, presets);

        request.continue();
    });

    await gotoApp(page, baseUrl);
    await page.waitForSelector('.launch-card[data-preset-id]', { timeout: 10000 });
    await page.select('#setup-filter-sort', 'name').catch(() => {});
    await sleep(400);
    await captureShot(page, 'preset-bundle-hero.png', { fullPage: true });
}
