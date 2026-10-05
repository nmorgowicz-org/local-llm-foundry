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
  await page.evaluate(async () => {
    const { openSpawnWizard, wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/capture-fixtures/launch-full-config-rapid';
    wizardState.model.paramB = 7;
    wizardState.model.modelBytes = 4 * 1024 * 1024 * 1024;
    selectWizardEngine('rapid_mlx', true);
    showStep(2);
  });
  await page.waitForSelector('#wizard-step-2.active', { timeout: 10000 });
  await sleep(700);
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
