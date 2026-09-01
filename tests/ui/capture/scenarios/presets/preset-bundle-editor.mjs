import { gotoApp } from '../../harness/browser.mjs';
import { captureShot } from '../../harness/shot.mjs';
import { sleep } from '../../harness/paths.mjs';

const weights = (id, name, path, kind = 'moe') => ({
    id, role: 'weights', display_name: name, local_path: path,
    quantization: { value: name.includes('Q5') ? 'q5_k_m' : 'q4_k_m', provenance: 'gguf_metadata' },
    metadata: { gguf_architecture: 'qwen3', model_kind: kind, block_count: kind === 'moe' ? 40 : 32 },
});

function bundle(kind = 'moe', onlyOne = false) {
    const artifacts = [weights('weights-q4', 'Model Q4_K_M', '/models/model-q4.gguf', kind)];
    if (!onlyOne) artifacts.push(weights('weights-q5', 'Model Q5_K_M', '/models/model-q5.gguf', kind));
    return {
        identity: { bundle_id: 'bundle-capture', tune_id: 'tune-capture', display_name: 'Q4/Q5 exact tune' },
        artifacts,
        context_options: [160000, 200000],
        kv_policy_options: ['q4_0_q4_0', 'q8_0_q8_0'],
        performance_options: [{ id: 'balanced', label: '2048 / 256', batch_size: 2048, ubatch_size: 256 }],
        cpu_moe_options: kind === 'moe' ? [0, 6] : [0],
        curated_selections: [], allow_validated_custom: true, workload_policy: 'general_chat',
        default_selection: { artifact_id: artifacts.at(-1).id, context_size: 200000, kv_policy: 'q4_0_q4_0', performance_id: 'balanced', n_cpu_moe: kind === 'moe' ? 6 : 0 },
    };
}

export default async function scenarioPresetBundleEditor(ctx) {
    const { page, baseUrl } = ctx;
    await gotoApp(page, baseUrl);
    await page.evaluate(async (data) => {
        const { openPresetModal } = await import('/js/features/presets.js');
        openPresetModal('new', null, { backend: 'llama_cpp', name: 'Bundle capture', model_path: '/models/model-q5.gguf', bundle: data });
    }, bundle());
    await page.locator('#preset-modal .preset-nav-item[data-section="variants"]').click();
    await sleep(300);
    await captureShot(page, 'preset-bundle-editor-artifacts-dark.png', { fullPage: true, runtimeTag: 'llamacpp-local' });
    await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
    await sleep(250);
    await captureShot(page, 'preset-bundle-editor-artifacts-light.png', { fullPage: true, runtimeTag: 'llamacpp-local' });
    await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
    await sleep(250);
    await captureShot(page, 'preset-bundle-editor-artifacts-narrow.png', { fullPage: true, runtimeTag: 'llamacpp-local' });
    await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
    await page.evaluate(() => document.documentElement.removeAttribute('data-theme'));
    await sleep(250);
    await captureShot(page, 'preset-bundle-editor-moe-options.png', { fullPage: true, runtimeTag: 'llamacpp-local' });

    await page.evaluate(async (data) => {
        const { closePresetModal, openPresetModal } = await import('/js/features/presets.js');
        closePresetModal();
        openPresetModal('new', null, { backend: 'llama_cpp', name: 'Dense capture', model_path: '/models/dense.gguf', bundle: data });
    }, bundle('dense'));
    await page.locator('#preset-modal .preset-nav-item[data-section="variants"]').click();
    await sleep(250);
    await captureShot(page, 'preset-bundle-editor-dense-no-moe.png', { fullPage: true, runtimeTag: 'llamacpp-local' });

    await page.evaluate(async (data) => {
        const { closePresetModal, openPresetModal } = await import('/js/features/presets.js');
        closePresetModal();
        openPresetModal('new', null, { backend: 'llama_cpp', name: 'Removal capture', model_path: '/models/model-q4.gguf', bundle: data });
    }, bundle('moe', true));
    await page.locator('#preset-modal .preset-nav-item[data-section="variants"]').click();
    await sleep(250);
    await captureShot(page, 'preset-bundle-editor-remove-warning.png', { fullPage: true, runtimeTag: 'llamacpp-local' });
}
