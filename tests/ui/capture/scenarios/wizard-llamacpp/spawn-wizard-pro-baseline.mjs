// Pro view contract (plan §14 / G7): one canonical settings surface with
// category navigation, search, modified-only filtering, reset, and responsive
// presentation. Captures are intentionally stateful so the manifest proves
// the controls remain reachable after each interaction.
import { loadAppDocument } from '../../harness/browser.mjs';
import { sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';

async function openPro(page, baseUrl) {
    await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
    await loadAppDocument(page, baseUrl);
    await page.evaluate(async () => {
        sessionStorage.clear();
        const { openSpawnWizard, wizardState, showStep } = await import('/js/features/spawn-wizard.js');
        openSpawnWizard();
        wizardState.model.source = 'local';
        wizardState.model.path = '/tmp/capture-fixtures/pro-shell-model.gguf';
        wizardState.model.paramB = 7;
        wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
        document.getElementById('wizard-binary-prereq')?.style.setProperty('display', 'none');
        showStep(1);
        const select = document.getElementById('view-mode-select');
        select.value = 'pro';
        select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForSelector('#spawn-wizard-overlay.open #pro-layout', { timeout: 10000 });
    await page.waitForFunction(() => document.querySelectorAll('#pro-rail-nav .pro-rail-item').length === 7, { timeout: 5000 });
    await sleep(250);
}

export default async function({ page, baseUrl }) {
    await openPro(page, baseUrl);

 // INTENT: Pro shell shows the seven-category rail and active pane.
 await captureShot(page, 'spawn-wizard-pro-shell.png', {
        fullPage: true,
        runtimeTag: 'llamacpp-local',
        expandSelector: '.wizard-body',
  });

  // Exercise every Pro category so each pane has fresh screenshot coverage.
  const categoryShots = [
    [0, 'spawn-wizard-pro-rail-model.png'],
    [1, 'spawn-wizard-pro-rail-memory.png'],
    [3, 'spawn-wizard-pro-rail-generation.png'],
    [4, 'spawn-wizard-pro-rail-tools.png'],
    [5, 'spawn-wizard-pro-rail-network.png'],
    [6, 'spawn-wizard-pro-rail-advanced.png'],
  ];
  for (const [index, name] of categoryShots) {
    await page.evaluate((i) => document.querySelectorAll('#pro-rail-nav .pro-rail-item')[i]?.click(), index);
    await sleep(150);
    await captureShot(page, name, {
      fullPage: true,
      runtimeTag: 'llamacpp-local',
      expandSelector: '.wizard-body',
    });
  }

  // Phase 6 adds a distinct compatibility pane and keeps native reasoning in
  // Generation & reasoning. Capture both categories at a legible scale.
  await page.evaluate(() => document.querySelectorAll('#pro-rail-nav .pro-rail-item').forEach(item => {
    if (item.textContent === 'Model & compatibility') item.click();
  }));
  await sleep(150);
  await captureShot(page, 'spawn-wizard-pro-model-compatibility.png', {
    fullPage: true,
    runtimeTag: 'llamacpp-local',
    expandSelector: '.wizard-body',
  });

  await page.evaluate(() => document.querySelectorAll('#pro-rail-nav .pro-rail-item').forEach(item => {
    if (item.textContent === 'Generation & reasoning') item.click();
  }));
  await sleep(150);
  await captureShot(page, 'spawn-wizard-pro-generation-reasoning.png', {
    fullPage: true,
    runtimeTag: 'llamacpp-local',
    expandSelector: '.wizard-body',
  });

  await page.evaluate(() => document.querySelectorAll('#pro-rail-nav .pro-rail-item')[2]?.click());
    await sleep(200);
 // INTENT: Performance category exposes llama.cpp performance controls.
 await captureShot(page, 'spawn-wizard-pro-rail-performance.png', {
        fullPage: true,
        runtimeTag: 'llamacpp-local',
        expandSelector: '.wizard-body',
    });

    await page.evaluate(() => {
        const input = document.getElementById('pro-filter-input');
        if (input) { input.value = 'batch'; input.dispatchEvent(new Event('input', { bubbles: true })); }
    });
    await sleep(150);
 // INTENT: Pro search filters the canonical batch-size control.
 await captureShot(page, 'spawn-wizard-pro-search-batch.png', {
        fullPage: true,
        runtimeTag: 'llamacpp-local',
        expandSelector: '.wizard-body',
    });

    await page.evaluate(() => {
        const filter = document.getElementById('pro-filter-input');
        if (filter) { filter.value = ''; filter.dispatchEvent(new Event('input', { bubbles: true })); }
        const batch = document.getElementById('spawn-batch-size');
        if (batch) { batch.value = '1024'; batch.dispatchEvent(new Event('input', { bubbles: true })); }
        document.querySelector('.pro-modified-only')?.click();
    });
    await sleep(150);
 // INTENT: Modified-only view shows the edited canonical setting.
 await captureShot(page, 'spawn-wizard-pro-modified-only.png', {
        fullPage: true,
        runtimeTag: 'llamacpp-local',
        expandSelector: '.wizard-body',
    });

  await page.evaluate(() => document.getElementById('pro-reset-all')?.click());
  await sleep(150);
  await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
 // INTENT: Reset and light theme leave the canonical baseline readable.
 await captureShot(page, 'spawn-wizard-pro-reset-light.png', {
        fullPage: true,
        runtimeTag: 'llamacpp-local',
    expandSelector: '.wizard-body',
  });

  await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
  await page.evaluate(() => {
    const modified = document.getElementById('pro-modified-only');
    if (modified?.checked) modified.click();
  });
  await page.evaluate(() => document.querySelector('#hw-kv-tiles [data-kv="q4_0"]')?.click());
  await sleep(150);
  await page.evaluate(() => document.querySelectorAll('[data-toast-close]').forEach(button => button.click()));
 // INTENT: Agentic workload warns when Q4 KV cache is selected.
 await captureShot(page, 'spawn-wizard-pro-agentic-q4-warning.png', {
    fullPage: true,
    runtimeTag: 'llamacpp-local',
    expandSelector: '.wizard-body',
  });

  await page.evaluate(async () => {
    const { showStep } = await import('/js/features/spawn-wizard.js');
    showStep(0);
    document.querySelector('.usecase-card[data-usecase="general"]')?.click();
    showStep(1);
    const select = document.getElementById('view-mode-select');
    if (select?.value !== 'pro') {
      select.value = 'pro';
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  await sleep(2500);
  await page.evaluate(() => document.querySelectorAll('[data-toast-close]').forEach(button => button.click()));
  await sleep(100);
 // INTENT: Roleplay workload shows the restored Q4 baseline state.
 await captureShot(page, 'spawn-wizard-pro-roleplay-q4-baseline.png', {
    fullPage: true,
    runtimeTag: 'llamacpp-local',
    expandSelector: '.wizard-body',
  });

  await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
    await page.evaluate(() => document.getElementById('pro-layout')?.scrollIntoView({ block: 'start', behavior: 'instant' }));
    await sleep(200);
 // INTENT: Narrow Pro layout remains usable without horizontal clipping.
 await captureShot(page, 'spawn-wizard-pro-narrow.png', {
        fullPage: true,
        runtimeTag: 'llamacpp-local',
        expandSelector: '.wizard-body',
    });
}
