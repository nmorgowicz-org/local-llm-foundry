import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

// Wizard (page 2) and preset-card Rapid-MLX download UI against a faked
// /api/rapid-mlx/model-status and /api/models/downloads. No real download, no runtime.

const GiB = 1024 ** 3;
const REPO_A = 'mlx-community/Model-A-4bit';
const REPO_B = 'mlx-community/Model-B-8bit';

const MODELS = {
  'alias-a': { ok: true, repo_id: REPO_A, size_bytes: 4 * GiB, cached: false, quant: '4-bit' },
  'alias-b': { ok: true, repo_id: REPO_B, size_bytes: 8 * GiB, cached: false, quant: '8-bit' },
  [REPO_A]: { ok: true, repo_id: REPO_A, size_bytes: 4 * GiB, cached: false, quant: '4-bit' },
};

/** In-test fake of the download + model-status endpoints. */
async function fakeBackend(page) {
  const fake = {
    statusRequests: [], starts: [], polls: {}, cancels: [], jobs: {},
    holdStart: null, holdPoll: null, models: structuredClone(MODELS),
  };
  await page.route('**/api/rapid-mlx/model-status?*', route => {
    const source = new URL(route.request().url()).searchParams.get('source');
    fake.statusRequests.push(source);
    const info = fake.models[source];
    return route.fulfill({ status: info ? 200 : 404, json: info || { ok: false } });
  });
  await page.route('**/api/models/downloads', async route => {
    if (route.request().method() !== 'POST') return route.fallback();
    const body = route.request().postDataJSON();
    fake.starts.push(body.repo_id);
    if (fake.holdStart) await fake.holdStart;
    const id = `job-${fake.starts.length}`;
    fake.jobs[id] = { state: 'running', bytes_done: GiB, bytes_total: 4 * GiB, current_file: 'weights.safetensors' };
    return route.fulfill({ json: { ok: true, job_id: id } });
  });
  await page.route('**/api/models/downloads/**', async route => {
    const url = new URL(route.request().url());
    const [, id, action] = url.pathname.match(/downloads\/([^/]+)(?:\/(cancel))?$/) || [];
    if (action === 'cancel') {
      fake.cancels.push(id);
      if (fake.jobs[id]) fake.jobs[id].state = 'cancelled';
      return route.fulfill({ json: { ok: true } });
    }
    fake.polls[id] = (fake.polls[id] || 0) + 1;
    if (fake.holdPoll) await fake.holdPoll;
    return route.fulfill({ json: { ok: true, job: fake.jobs[id] || {} } });
  });
  return fake;
}

test.beforeEach(async ({ page }) => {
  await page.routeWebSocket('**/ws', ws => ws.close());
  await page.route('**/api/**', route => route.abort());
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
});

test.describe('@fake-data-bypass Rapid-MLX wizard download step', () => {
  async function openWizard(page) {
    await page.evaluate(async () => {
      const wiz = await import('/js/features/spawn-wizard.js');
      wiz.openSpawnWizard({ templatePreset: { backend: 'rapid_mlx', rapid_mlx: { model_source: { kind: 'alias', value: 'alias-a' } } } });
      wiz.showStep(1);
    });
    await expect(page.locator('.wizard-body')).toBeVisible();
  }

  /** Select a Rapid-MLX alias the way the wizard stores it, then re-check like a step change. */
  const select = (page, alias) => page.evaluate(async value => {
    const { wizardState } = await import('/js/features/spawn-wizard.js');
    const { refreshRapidModelDownload } = await import('/js/features/spawn-wizard-rapid-download.js');
    wizardState.engine.selected = 'rapid_mlx';
    wizardState.model.rapidMlxSource = { kind: 'alias', value };
    await refreshRapidModelDownload();
  }, alias);

  const missingRepo = (page) => page.evaluate(async () =>
    (await import('/js/features/spawn-wizard.js')).wizardState.model.rapidDownload?.repo_id ?? null);

  test('re-checking while a download runs keeps the progress view and cannot start a duplicate', async ({ page }) => {
    const fake = await fakeBackend(page);
    await openWizard(page);
    await select(page, 'alias-a');
    await expect(page.locator('#rapid-dlp-idle')).toBeVisible();
    await expect(page.locator('#rapid-dlp-repo')).toContainText(REPO_A);

    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await expect(page.locator('#rapid-dlp-stats')).toContainText('1.0 GiB / 4.0 GiB');

    // Step navigation / source edit re-runs the check for the same source.
    await select(page, 'alias-a');
    await select(page, 'alias-a');
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await expect(page.locator('#rapid-dlp-idle')).toBeHidden();
    await expect(page.locator('#rapid-dlp-download-btn')).toBeDisabled();
    // Programmatic click on the disabled/hidden button must not start another job either.
    await page.evaluate(() => document.getElementById('rapid-dlp-download-btn').click());
    expect(fake.starts).toEqual([REPO_A]);
    expect(await missingRepo(page)).toBe(REPO_A); // Next stays held while it downloads

    fake.jobs['job-1'] = { state: 'complete', bytes_done: 4 * GiB, bytes_total: 4 * GiB };
    await expect(page.locator('#rapid-dlp-complete')).toBeVisible({ timeout: 6000 });
    expect(await missingRepo(page)).toBeNull();
    expect(fake.starts).toHaveLength(1);
  });

  test('changing the source detaches explicitly, says so, and a stale poll cannot touch the new model', async ({ page }) => {
    const fake = await fakeBackend(page);
    await openWizard(page);
    await select(page, 'alias-a');
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await expect(page.locator('#rapid-dlp-stats')).toContainText('1.0 GiB');

    // Hold the next poll response so it resolves only after the user switched models.
    let release;
    fake.holdPoll = new Promise(resolve => { release = resolve; });
    await expect.poll(() => fake.polls['job-1']).toBeGreaterThan(1);
    await select(page, 'alias-b');
    await expect(page.locator('#rapid-dlp-idle')).toBeVisible();
    await expect(page.locator('#rapid-dlp-repo')).toContainText(REPO_B);
    await expect(page.locator('#rapid-dlp-note')).toContainText(`Download of ${REPO_A} continues in the background`);
    await expect(page.locator('#rapid-dlp-download-btn')).toBeEnabled();

    // The late response for the old repo reports completion; it must be ignored.
    fake.jobs['job-1'] = { state: 'complete', bytes_done: 4 * GiB, bytes_total: 4 * GiB };
    fake.holdPoll = null;
    release();
    await page.waitForTimeout(300);
    await expect(page.locator('#rapid-dlp-idle')).toBeVisible();
    await expect(page.locator('#rapid-dlp-complete')).toBeHidden();
    expect(await missingRepo(page)).toBe(REPO_B);

    // Detached means no more polling for the old job.
    const before = fake.polls['job-1'];
    await page.waitForTimeout(1800);
    expect(fake.polls['job-1']).toBe(before);
    expect(fake.starts).toEqual([REPO_A]);
  });

  test('returning to a source with a running download resumes it instead of offering a second start', async ({ page }) => {
    const fake = await fakeBackend(page);
    await openWizard(page);
    await select(page, 'alias-a');
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await select(page, 'alias-b');
    await expect(page.locator('#rapid-dlp-idle')).toBeVisible();

    await select(page, 'alias-a');
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await expect(page.locator('#rapid-dlp-download-btn')).toBeDisabled();
    await expect(page.locator('#rapid-dlp-stats')).toContainText('1.0 GiB / 4.0 GiB');
    expect(fake.starts).toEqual([REPO_A]);
  });

  test('re-checking while the start request is in flight does not allow a second start', async ({ page }) => {
    const fake = await fakeBackend(page);
    await openWizard(page);
    await select(page, 'alias-a');
    let release;
    fake.holdStart = new Promise(resolve => { release = resolve; });
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await select(page, 'alias-a');
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await expect(page.locator('#rapid-dlp-download-btn')).toBeDisabled();
    // Cancel before a job id exists is a no-op, not a request for "null".
    await page.evaluate(() => document.getElementById('rapid-dlp-cancel-btn').click());
    release();
    await expect(page.locator('#rapid-dlp-stats')).toContainText('1.0 GiB / 4.0 GiB');
    expect(fake.starts).toEqual([REPO_A]);
    expect(fake.cancels).toEqual([]);
  });

  test('cancel reports the cancelled job and offers a resumable retry', async ({ page }) => {
    const fake = await fakeBackend(page);
    await openWizard(page);
    await select(page, 'alias-a');
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-stats')).toContainText('1.0 GiB');
    await page.locator('#rapid-dlp-cancel-btn').click();
    await expect(page.locator('#rapid-dlp-idle')).toBeVisible({ timeout: 6000 });
    await expect(page.locator('#rapid-dlp-note')).toContainText('Download cancelled');
    await expect(page.locator('#rapid-dlp-download-btn')).toBeEnabled();
    expect(fake.cancels).toEqual(['job-1']);
  });
});

test.describe('@fake-data-bypass Rapid-MLX preset card download state', () => {
  const SOURCE = REPO_A;

  /** Build a minimal launch card and attach the download state exactly like setup-view does. */
  const mountCard = (page, id) => page.evaluate(async ({ source, id }) => {
    const { attachRapidDownloadState } = await import('/js/features/rapid-model-download.js');
    document.getElementById(id)?.remove();
    const card = document.createElement('div');
    card.id = id;
    card.className = 'launch-card';
    card.innerHTML = '<div class="launch-card-chips"></div><div class="launch-card-actions"><button class="launch-card-btn-start">Start</button></div>';
    document.body.appendChild(card);
    await attachRapidDownloadState(card, source);
  }, { source: SOURCE, id });

  test('poll loop stops when the card is re-rendered, and the replacement resumes the same job', async ({ page }) => {
    const fake = await fakeBackend(page);
    await mountCard(page, 'card-1');
    await page.locator('#card-1 .launch-card-dl-btn').click();
    await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
    await page.evaluate(() => document.getElementById('card-1').remove()); // grid re-render drops the card

    // The detached card's loop tears down on its next tick instead of polling forever.
    await page.waitForTimeout(1800);
    const settled = fake.polls['job-1'];
    await page.waitForTimeout(1800);
    expect(fake.polls['job-1']).toBe(settled);

    // The replacement card resumes the running job; no duplicate start.
    await mountCard(page, 'card-2');
    await expect(page.locator('#card-2 .launch-card-dl-btn')).toBeDisabled();
    await expect(page.locator('#card-2 .launch-card-dl-label')).toContainText('Downloading');
    expect(fake.starts).toEqual([REPO_A]);
  });

  test('cancel with no job id is a no-op', async ({ page }) => {
    const fake = await fakeBackend(page);
    await mountCard(page, 'card-1');
    // The cancel button is hidden until a job exists; a stray activation must not POST /null/cancel.
    await page.evaluate(() => document.querySelector('#card-1 .launch-card-dl-cancel').click());
    await page.waitForTimeout(200);
    expect(fake.cancels).toEqual([]);
  });

  test('model-status memo expires, and invalidates when a download completes anywhere', async ({ page }) => {
    const fake = await fakeBackend(page);
    const lookups = () => fake.statusRequests.length;
    await page.evaluate(async source => {
      const mod = await import('/js/features/rapid-model-download.js');
      await mod.rapidModelInfo(source);
      await mod.rapidModelInfo(source);
    }, SOURCE);
    expect(lookups()).toBe(1); // memoized

    // Another surface (e.g. the wizard) finished the download: next lookup must refetch.
    fake.models[SOURCE] = { ...MODELS[SOURCE], cached: true };
    const cachedAfter = await page.evaluate(async source => {
      const mod = await import('/js/features/rapid-model-download.js');
      mod.notifyRapidModelDownloaded('mlx-community/Model-A-4bit');
      return (await mod.rapidModelInfo(source)).cached;
    }, SOURCE);
    expect(lookups()).toBe(2);
    expect(cachedAfter).toBe(true);

    // A download finished outside this page: only the TTL can notice.
    fake.models[SOURCE] = { ...MODELS[SOURCE], cached: false };
    const afterTtl = await page.evaluate(async source => {
      const mod = await import('/js/features/rapid-model-download.js');
      const realNow = Date.now;
      Date.now = () => realNow() + 10 * 60_000;
      try { return (await mod.rapidModelInfo(source)).cached; } finally { Date.now = realNow; }
    }, SOURCE);
    expect(lookups()).toBe(3);
    expect(afterTtl).toBe(false);
  });

  test('a download finished elsewhere removes the card download row', async ({ page }) => {
    await fakeBackend(page);
    await mountCard(page, 'card-1');
    await expect(page.locator('#card-1 .launch-card-dl')).toBeVisible();
    await page.evaluate(async () => {
      (await import('/js/features/rapid-model-download.js')).notifyRapidModelDownloaded('mlx-community/Model-A-4bit');
    });
    await expect(page.locator('#card-1 .launch-card-dl')).toHaveCount(0);
    await expect(page.locator('#card-1')).not.toHaveAttribute('data-model-missing', '1');
  });
});
