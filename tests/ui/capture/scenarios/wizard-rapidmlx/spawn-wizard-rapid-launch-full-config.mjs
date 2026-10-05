// Scenario: Rapid-MLX launch Full config review (Phase 10 / G10).
// Captures requested settings plus the runtime-effective command-preview lane.
import { loadAppDocument } from '../../harness/browser.mjs';
import { sleep } from '../../harness/paths.mjs';
import { captureShot, suppressToasts } from '../../harness/shot.mjs';

export default async function (ctx) {
  const { page, baseUrl } = ctx;
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
  await loadAppDocument(page, baseUrl);
  await suppressToasts(page);
  await page.evaluate(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = new URL(typeof input === 'string' ? input : input.url, window.location.origin);
      if (url.pathname === '/api/rapid-mlx/command-preview') {
        const config = JSON.parse(init.body);
        return Promise.resolve(new Response(JSON.stringify({
          argv: ['serve', config.model_source.value, '--host', config.host, '--port', String(config.port), '--served-model-name', config.served_model_name || config.model_source.value],
          redacted: true,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      return originalFetch(input, init);
    };
  });
  await page.evaluate(async () => {
    const { openSpawnWizard, wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    wizardState.model.source = 'hf';
    wizardState.model.path = '';
    wizardState.model.hfRepo = 'qwen3.8-27b-4bit';
    wizardState.model.rapidMlxSource = { kind: 'alias', value: 'qwen3.8-27b-4bit' };
    wizardState.model.paramB = 7;
    wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
    selectWizardEngine('rapid_mlx', true);
    showStep(2);
  });
  await page.waitForSelector('#wizard-step-2.active', { timeout: 10000 });
  await page.locator('#spawn-alias').fill('qwen3.8-27b-4bit-205k');
  await sleep(700);
  const naming = await page.evaluate(async () => {
    const { buildSpawnPayload, buildPresetPayload } = await import('/js/features/spawn-wizard.js');
    return { spawn: buildSpawnPayload().rapid_mlx, preset: buildPresetPayload().rapid_mlx };
  });
  if (naming.spawn.served_model_name !== 'qwen3.8-27b-4bit-205k' || naming.preset.served_model_name !== naming.spawn.served_model_name || naming.spawn.model_source.value !== 'qwen3.8-27b-4bit') {
    throw new Error(`Served-name/source contract drifted: ${JSON.stringify(naming)}`);
  }
  const layout = await page.evaluate(() => {
    const drawer = document.getElementById('spawn-full-config-drawer');
    const card = document.getElementById('spawn-config-card');
    return {
      drawerClipped: drawer.scrollHeight > drawer.clientHeight + 2,
      cardHeight: card.getBoundingClientRect().height,
      cardShrink: getComputedStyle(card).flexShrink,
    };
  });
  if (layout.drawerClipped || layout.cardHeight < 100 || layout.cardShrink !== '0') {
    throw new Error(`Launch review cards collapsed: ${JSON.stringify(layout)}`);
  }
  await page.evaluate(() => document.getElementById('spawn-full-config-drawer')?.scrollIntoView({ behavior: 'instant', block: 'start' }));
 // INTENT: Rapid launch review shows requested and runtime-effective values.
 await captureShot(page, 'spawn-wizard-launch-full-config.png', {
    fullPage: true,
    runtimeTag: 'rapidmlx-local',
    expandSelector: '.wizard-body',
  });
  await page.evaluate(() => {
    const step = document.getElementById('wizard-step-2');
    if (step) step.scrollTop = step.scrollHeight;
  });
  await sleep(250);
 // INTENT: Expanded Rapid launch review exposes the complete backend config.
 await captureShot(page, 'spawn-wizard-launch-full-config-details.png', {
    fullPage: true,
    runtimeTag: 'rapidmlx-local',
    expandSelector: '#wizard-step-2',
  });
  await page.setViewport({ width: 430, height: 900, deviceScaleFactor: 1 });
  await sleep(250);
 // INTENT: Narrow Rapid launch review remains readable without clipping.
 await captureShot(page, 'spawn-wizard-launch-full-config-narrow.png', {
    fullPage: true,
    runtimeTag: 'rapidmlx-local',
    expandSelector: '#wizard-step-2',
  });
}
