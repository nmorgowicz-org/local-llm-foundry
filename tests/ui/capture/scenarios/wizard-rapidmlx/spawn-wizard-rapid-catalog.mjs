// Scenario: spawn-wizard-rapid-catalog
// SCENARIO INTENT: Document the curated-only Rapid-MLX model picker — recipe
// recommendations pinned on top, the searchable chat-model catalog, a row
// selection filling alias/size/MTP in one step, and the unvalidated-model
// review notice — plus the Rapid-MLX uninstall confirm dialog.
import { loadAppDocument } from '../../harness/browser.mjs';
import { sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';
import { suppressToasts } from '../../harness/shot.mjs';

const CATALOG = {
  ok: true,
  models: [
    { name: 'qwen3.8-27b-4bit', display_name: 'qwen3.8-27b-4bit', size_bytes: 16320875724, parser: 'qwen3_coder_xml', template: 'qwen3', hybrid: false, mtp: true, mtp_sidecar: 'rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@3' },
    { name: 'qwen3.6-35b-4bit', display_name: 'qwen3.6-35b-4bit', size_bytes: 21474836480, parser: 'qwen3_coder_xml', template: 'qwen3', hybrid: false, mtp: true, mtp_sidecar: 'rapid-mlx/Qwen3.6-35B-MTP-MLX@3' },
    { name: 'qwen3.8-27b-mixed-3.5bpw', display_name: 'qwen3.8-27b-mixed-3.5bpw', size_bytes: 14817679360, parser: 'qwen3_coder_xml', template: 'qwen3', hybrid: false, mtp: true, mtp_sidecar: 'rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX@3' },
    { name: 'gemma4-27b-4bit', display_name: 'gemma4-27b-4bit', size_bytes: 16106127360, parser: '—', template: 'gemma4', hybrid: true, mtp: false },
    { name: 'llama4-8b-4bit', display_name: 'llama4-8b-4bit', size_bytes: 4294967296, parser: 'llama3_json', template: 'llama4', hybrid: false, mtp: false },
    // Literal upstream pipeline labels; these must never appear in the picker.
    { name: 'whisper-large-v3', size_bytes: 3113851289, parser: '[audio:stt]' },
    { name: 'cogvideox-fun-5b-bf16', size_bytes: 21045339750, parser: '[video:gen]' },
    { name: 'image-generation', size_bytes: 4294967296, parser: '[image:gen]' },
  ],
  recommendations: [
    { rank: 1, label: 'Smart', name: 'qwen3.8-27b-4bit', cached: true, specs: '20.0 GB RAM · 92% capability · ~41 tok/s' },
    { rank: 2, label: 'Fast', name: 'qwen3.6-35b-4bit', cached: false, specs: '20.0 GB RAM · 87% capability · ~60 tok/s' },
  ],
};

export default async function (ctx) {
  const { page, baseUrl } = ctx;
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
  await loadAppDocument(page, baseUrl);

  // Mock the catalog plus the endpoints the wizard consults around it. The
  // catalog is the point of this capture; everything else keeps the wizard's
  // chrome quiet.
  await page.evaluate((catalog) => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = new URL(typeof input === 'string' ? input : input.url, window.location.origin);
      if (url.pathname === '/api/rapid-mlx/catalog') {
        return Promise.resolve(new Response(JSON.stringify(catalog), {
          status: 200, headers: { 'Content-Type': 'application/json' },
        }));
      }
      if (url.pathname === '/api/rapid-mlx/runtime/status') {
        return Promise.resolve(new Response(JSON.stringify({
          runtime: { supported: true, active: { version: '0.15.6' } }, jobs: [],
        }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      if (url.pathname === '/api/runtimes/storage') {
        return Promise.resolve(new Response(JSON.stringify({
          ok: true, rapid_mlx_bytes: 2415919104, llama_bin_bytes: 0,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      if (url.pathname === '/api/rapid-mlx/recommend') {
        return Promise.resolve(new Response(JSON.stringify({
          recommended_backend: 'rapid_mlx', state: 'ready',
          reason: 'Rapid-MLX is available for MLX models on this system.',
        }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      if (url.pathname.startsWith('/api/rapid-mlx/models/') && url.pathname.endsWith('/profile')) {
        return Promise.resolve(new Response(JSON.stringify({
          profile: { extras: { has_reasoning: true } },
        }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      return originalFetch(input, init);
    };
  }, CATALOG);

  await page.evaluate(async () => {
    const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard();
  });
  await page.waitForSelector('#spawn-wizard-overlay.open', { timeout: 8000 });
  await suppressToasts(page);
  await page.evaluate(() => {
    const banner = document.getElementById('wizard-binary-prereq');
    if (banner) banner.style.display = 'none';
    (document.querySelector('.profile-card[data-profile="power"]')
      || document.querySelector('.profile-card'))?.click();
    (document.querySelector('.usecase-card[data-usecase="agentic"]')
      || document.querySelector('.usecase-card'))?.click();
  });
  await sleep(300);
  await page.evaluate(() => {
    document.querySelector('.wizard-engine-card[data-engine="rapid_mlx"]')?.click();
    // The curated catalog lives in the HF model area; pick that source so the
    // panel is actually on screen.
    document.querySelector('.model-source-card[data-source="hf"]')?.click();
  });
  await page.waitForFunction(
    () => document.getElementById('rapid-catalog-panel')?.style.display !== 'none',
    { timeout: 8000 },
  );
  // Wait for the catalog fetch to render rows.
  await page.waitForFunction(
    () => document.querySelectorAll('#rapid-catalog-panel .rapid-catalog-row').length >= 5,
    { timeout: 8000 },
  );
  const visibleNames = await page.evaluate(() =>
    [...document.querySelectorAll('#rapid-catalog-panel .rapid-catalog-name')].map(row => row.textContent),
  );
  const expectedNames = CATALOG.models.slice(0, 5).map(model => model.name);
  if (visibleNames.length !== expectedNames.length || visibleNames.some(name => !expectedNames.includes(name))) {
    throw new Error(`Curated picker leaked media pipelines: ${JSON.stringify(visibleNames)}`);
  }
  await sleep(400);
  await page.evaluate(() => document.getElementById('rapid-catalog-panel')?.scrollIntoView({ behavior: 'instant', block: 'start' }));
  await sleep(250);

  // INTENT: Pinned "Recommended for your Mac" rows on top of the catalog.
  await captureShot(page, 'spawn-wizard-rapid-catalog-picker.png', {
    runtimeTag: 'rapidmlx-local',
    expandSelector: '#rapid-catalog-panel',
  });

  // INTENT: The search narrows the catalog while keeping pinned matches.
  await page.evaluate(() => {
    const search = document.querySelector('#rapid-catalog-panel .rapid-catalog-search');
    search.value = 'qwen3.8';
    search.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await sleep(500);
  await page.evaluate(() => document.getElementById('rapid-catalog-panel')?.scrollIntoView({ behavior: 'instant', block: 'start' }));
  await sleep(250);
  await captureShot(page, 'spawn-wizard-rapid-catalog-search.png', {
    runtimeTag: 'rapidmlx-local',
    expandSelector: '#rapid-catalog-panel',
  });

  // Select the pinned Smart pick and clear the filter.
  await page.evaluate(() => {
    const search = document.querySelector('#rapid-catalog-panel .rapid-catalog-search');
    search.value = '';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    document.querySelector('#rapid-catalog-panel .rapid-catalog-row--recommended')?.click();
  });
  await sleep(900);

  // INTENT: One click selects the model — alias, measured size, MTP sidecar.
  await page.evaluate(() => document.getElementById('rapid-catalog-panel')?.scrollIntoView({ behavior: 'instant', block: 'start' }));
  await sleep(250);
  await captureShot(page, 'spawn-wizard-rapid-catalog-selected.png', {
    runtimeTag: 'rapidmlx-local',
    expandSelector: '#rapid-catalog-panel',
  });

  // INTENT: The uninstall confirm names the runtime size and keeps models.
  await page.evaluate(async () => {
    const { openRapidMlxModal } = await import('/js/features/rapid-mlx-updater.js');
    openRapidMlxModal();
  });
  await sleep(600);
  await page.evaluate(() => {
    document.getElementById('rapid-mlx-uninstall-btn')?.click();
  });
  await page.waitForSelector('.runtime-uninstall-confirm-overlay', { timeout: 8000 });
  await sleep(300);
  await captureShot(page, 'runtime-uninstall-confirm.png', {
    runtimeTag: 'rapidmlx-local',
  });
  await page.evaluate(() => {
    document.querySelector('.runtime-uninstall-cancel')?.click();
  });
}
