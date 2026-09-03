// Scenario: spawn-wizard-guided-drawer (plan §11 Phase 4 / G4).
// Exercises the single Guided All settings drawer across both loaders. The
// assertions deliberately inspect canonical DOM ownership rather than relying
// on a screenshot alone: one wrapper is mounted at a time and inactive backend
// controls remain hidden.
import { loadAppDocument } from '../../harness/browser.mjs';
import { sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';

export default async function(ctx) {
    const { page, baseUrl } = ctx;
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await loadAppDocument(page, baseUrl);

    await page.evaluate(async () => {
        const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
        openSpawnWizard({
            localPath: '/tmp/capture-fixtures/guided-drawer.gguf',
            localModel: { path: '/tmp/capture-fixtures/guided-drawer.gguf', size_bytes: 4_920_000_000, source_kind: 'gguf_file' },
        });
    });
    await page.waitForSelector('#spawn-wizard-overlay.open', { timeout: 10000 });
    await page.waitForFunction(() => document.getElementById('wizard-step-0')?.classList.contains('active'), { timeout: 5000 });

    const state = await page.evaluate(async () => {
        const mod = await import('/js/features/spawn-wizard.js');
        const registry = await import('/js/features/spawn-wizard-groups.js');
        mod.wizardState.vram.available = 64 * 1024 * 1024 * 1024;
        mod.wizardState.model.paramB = 8;
        mod.wizardState.model.modelBytes = 4_920_000_000;
        mod.selectWizardEngine('llama_cpp', true);
        mod.showStep(1);
        const stateRegistry = mod.settingStateRegistry;
        const llamaSettings = registry.controlsForView('llama_cpp', 'guided').filter(control => control.view !== 'card');
        const entries = stateRegistry.snapshot(llamaSettings);
        return {
            selected: mod.wizardState.engine.selected,
            llamaDescriptors: registry.validatePresentationDescriptors('llama_cpp'),
            rapidDescriptors: registry.validatePresentationDescriptors('rapid_mlx'),
            stateEntries: entries.length,
            dirtyEntries: entries.filter(entry => entry.dirty).length,
            stateShape: entries.every(entry => ['resolvedDefault', 'value', 'dirty', 'provenance', 'effective', 'pending'].every(key => key in entry)),
        };
    });
    if (state.selected !== 'llama_cpp' || !state.llamaDescriptors.ok || !state.rapidDescriptors.ok || state.stateEntries < 1 || !state.stateShape) {
        throw new Error(`Invalid presentation registry: ${JSON.stringify(state)}`);
    }
    await sleep(500);

    const inspectDrawer = async (expectedEngine) => page.evaluate((engine) => {
        const drawer = document.getElementById('all-settings-drawer');
        const body = document.getElementById('all-settings-body');
        const group = document.getElementById('all-settings-group');
        const llama = document.getElementById('spawn-advanced-fields');
        const rapid = document.getElementById('spawn-rapid-advanced-fields');
        const active = engine === 'rapid_mlx' ? rapid : llama;
        const inactive = engine === 'rapid_mlx' ? llama : rapid;
        const controls = active ? active.querySelectorAll('select, input, textarea').length : 0;
        const hiddenAncestors = [];
        for (let node = active?.parentElement; node; node = node.parentElement) {
            if (node.hidden || getComputedStyle(node).display === 'none') hiddenAncestors.push(node.id || node.className);
        }
        return {
            count: Number(document.getElementById('all-settings-count')?.textContent || 0),
            changed: Number(document.getElementById('all-settings-changed')?.dataset.count || 0),
            drawerVisible: drawer?.style.display !== 'none',
            bodyOpen: body?.style.display === 'block',
            activeMounted: !!active && active.parentElement === group,
            activeVisible: !!active && getComputedStyle(active).display !== 'none',
            activeParent: active?.parentElement?.id || active?.parentElement?.className || '',
            rapidPanelHidden: document.getElementById('rapid-hardware-panel')?.hidden,
            rapidPanelRect: (() => { const r = document.getElementById('rapid-hardware-panel')?.getBoundingClientRect(); return r ? { x: r.x, y: r.y, w: r.width, h: r.height } : null; })(),
            rapidPanelText: document.getElementById('rapid-hardware-panel')?.textContent?.trim().slice(0, 120) || '',
            engineClass: document.getElementById('spawn-wizard-overlay')?.className || '',
            hiddenAncestors,
            inactiveHidden: !inactive || getComputedStyle(inactive).display === 'none',
            controls,
            duplicateIds: [...(document.getElementById('spawn-wizard-overlay')?.querySelectorAll('[id]') || [])]
                .map(el => el.id).filter((id, i, ids) => ids.indexOf(id) !== i),
        };
    }, expectedEngine);

    let snapshot = await inspectDrawer('llama_cpp');
    if (!snapshot.drawerVisible || !snapshot.activeMounted || !snapshot.activeVisible || !snapshot.inactiveHidden || snapshot.count < 1 || snapshot.controls < 1 || snapshot.duplicateIds.length) {
        throw new Error(`Invalid llama drawer state: ${JSON.stringify(snapshot)}`);
    }
    // INTENT: Llama Guided drawer is closed with the computed total and zero changed settings.
    await captureShot(page, 'spawn-wizard-guided-drawer-llama-closed.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '.wizard-body' });
    await page.evaluate(() => {
        const input = document.getElementById('spawn-batch-size');
        if (input) { input.value = '1024'; input.dispatchEvent(new Event('input', { bubbles: true })); }
    });
    await sleep(200);
    const dirtyState = await page.evaluate(async () => {
        const mod = await import('/js/features/spawn-wizard.js');
        const groups = await import('/js/features/spawn-wizard-groups.js');
        const descriptor = groups.controlsForView('llama_cpp', 'guided').find(control => control.id === 'spawn-batch-size');
        const registry = mod.settingStateRegistry;
        const entry = registry.entries.get(descriptor.semanticId);
        const editedValue = entry?.value;
        registry.setResolvedDefault(descriptor.semanticId, '2048');
        const preservedDirtyEdit = entry?.value === editedValue && entry?.dirty && entry?.provenance === 'user';
        registry.reset(descriptor.semanticId);
        const resetValue = document.getElementById('spawn-batch-size')?.value;
        const resetToResolved = resetValue === '2048' && !entry?.dirty && entry?.provenance === 'resolved';
        const input = document.getElementById('spawn-batch-size');
        if (input) { input.value = editedValue; input.dispatchEvent(new Event('input', { bubbles: true })); }
        return { preservedDirtyEdit, resetToResolved };
    });
    if (!dirtyState.preservedDirtyEdit || !dirtyState.resetToResolved) {
        throw new Error(`Setting registry lost dirty/resolved state: ${JSON.stringify(dirtyState)}`);
    }
    snapshot = await inspectDrawer('llama_cpp');
    if (snapshot.changed < 1) throw new Error(`Changed count did not update: ${JSON.stringify(snapshot)}`);
    const llamaPayloadBeforeSwitch = await page.evaluate(async () => {
        const { buildSpawnPayload } = await import('/js/features/spawn-wizard.js');
        return buildSpawnPayload();
    });
    await page.evaluate(() => document.getElementById('all-settings-btn')?.click());
    await sleep(250);
    snapshot = await inspectDrawer('llama_cpp');
    if (!snapshot.bodyOpen || snapshot.count < 1) throw new Error(`Llama drawer did not open: ${JSON.stringify(snapshot)}`);
    await page.evaluate(() => document.getElementById('all-settings-group')?.scrollIntoView({ behavior: 'instant', block: 'start' }));
    // INTENT: Llama Guided drawer is open after a canonical edit and exposes the changed count.
    await captureShot(page, 'spawn-wizard-guided-drawer-llama-open.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '.wizard-body' });

    // Keep the newly canonical Phase 6 controls legible in dedicated captures.
    for (const [id, name] of [
        ['spawn-cache-idle-slots', 'spawn-wizard-guided-cache-slots.png'],
        ['spawn-mmproj-offload', 'spawn-wizard-guided-mmproj-offload.png'],
        ['spawn-reasoning-effort', 'spawn-wizard-guided-reasoning.png'],
    ]) {
        await page.evaluate((controlId) => {
            const control = document.getElementById(controlId);
            control?.closest('details')?.setAttribute('open', '');
            control?.scrollIntoView({ behavior: 'instant', block: 'center' });
        }, id);
        await captureShot(page, name, { fullPage: false, runtimeTag: 'llamacpp-local', expandSelector: `#${id}` });
    }

    await page.evaluate(async () => {
        const { selectWizardEngine } = await import('/js/features/spawn-wizard.js');
        selectWizardEngine('rapid_mlx', true);
    });
    await sleep(400);
    await page.evaluate(() => {
        const body = document.getElementById('all-settings-body');
        if (body?.style.display === 'block') document.getElementById('all-settings-btn')?.click();
        document.querySelectorAll('.wizard-main, .wizard-sidebar, .hw-vram-sidebar').forEach(el => { el.scrollTop = 0; });
    });
    await sleep(250);
    snapshot = await inspectDrawer('rapid_mlx');
    if (!snapshot.drawerVisible || snapshot.bodyOpen || !snapshot.activeMounted || !snapshot.activeVisible || snapshot.rapidPanelHidden || !snapshot.engineClass.includes('engine-rapid-mlx') || !snapshot.inactiveHidden || snapshot.count < 1 || snapshot.controls < 1 || snapshot.duplicateIds.length) {
        throw new Error(`Invalid Rapid drawer state: ${JSON.stringify(snapshot)}`);
    }
    // INTENT: Rapid Guided drawer is closed while the backend-native memory panel remains visible.
    await captureShot(page, 'spawn-wizard-guided-drawer-rapid-closed.png', { fullPage: true, runtimeTag: 'rapidmlx-local', expandSelector: '.wizard-body' });
    await page.evaluate(() => {
        document.getElementById('all-settings-btn')?.click();
        document.getElementById('all-settings-group')?.scrollIntoView({ behavior: 'instant', block: 'start' });
    });
    await sleep(250);
    snapshot = await inspectDrawer('rapid_mlx');
    if (!snapshot.bodyOpen || snapshot.count < 1) throw new Error(`Rapid drawer did not open: ${JSON.stringify(snapshot)}`);
    // INTENT: Rapid Guided drawer is open and shows native generation/protocol controls.
    await captureShot(page, 'spawn-wizard-guided-drawer-rapid-open.png', { fullPage: true, runtimeTag: 'rapidmlx-local', expandSelector: '.wizard-body' });

    await page.evaluate(async () => {
        const { selectWizardEngine } = await import('/js/features/spawn-wizard.js');
        selectWizardEngine('llama_cpp', true);
    });
    await sleep(300);
    snapshot = await inspectDrawer('llama_cpp');
    if (!snapshot.activeMounted || !snapshot.inactiveHidden || snapshot.duplicateIds.length) {
        throw new Error(`Mixed backend controls after switch-back: ${JSON.stringify(snapshot)}`);
    }
    const parity = await page.evaluate(async () => {
        const { buildSpawnPayload, buildPresetPayload } = await import('/js/features/spawn-wizard.js');
        return { payload: buildSpawnPayload(), preset: buildPresetPayload() };
    });
    if (JSON.stringify(parity.payload) !== JSON.stringify(llamaPayloadBeforeSwitch)) {
        throw new Error('Canonical llama.cpp payload changed across backend presentation switches');
    }
    if (parity.preset.batch_size !== 1024) {
        throw new Error(`Preset projection lost the canonical drawer edit: ${parity.preset.batch_size}`);
    }

    await page.evaluate(() => document.getElementById('wizard-back-btn')?.click());
    await sleep(250);
    // INTENT: The Pro selector is available while the canonical Guided state remains active.
    await captureShot(page, 'spawn-wizard-guided-drawer-pro-available.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '.wizard-body' });
 const workloadState = await page.evaluate(async () => {
     const { wizardState } = await import('/js/features/spawn-wizard.js');
     document.querySelector('.usecase-card[data-usecase="agentic"]')?.click();
     const selected = document.querySelector('.usecase-card[data-usecase="agentic"]')?.classList.contains('selected');
     const scenario = wizardState.hardware.workloadScenario;
     document.querySelector('.usecase-card[data-usecase="agentic"]')?.click();
     return { selected, scenario };
 });
 if (!workloadState.selected || workloadState.scenario !== 'interactive_coding_agent') {
     throw new Error(`Workload intent did not map to backend scenario: ${JSON.stringify(workloadState)}`);
 }
 const proState = await page.evaluate(() => ({
        value: document.getElementById('view-mode-select')?.value,
        optionDisabled: document.querySelector('#view-mode-select option[value="pro"]')?.disabled,
        text: document.querySelector('#view-mode-select option[value="pro"]')?.textContent || '',
        workloadCards: document.querySelectorAll('.usecase-card[data-usecase]').length,
    }));
    if (proState.value === 'pro' || proState.optionDisabled || !/pro/i.test(proState.text) || proState.workloadCards !== 2) {
        throw new Error(`Pro availability is not honest: ${JSON.stringify(proState)}`);
    }
}
