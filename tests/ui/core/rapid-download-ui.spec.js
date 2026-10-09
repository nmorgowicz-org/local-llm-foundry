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

const transientPollResponses = [
  ['HTTP 500', { status: 500, json: { ok: false, error: 'temporarily unavailable' } }],
  ['HTTP 500 with a terminal-looking body', { status: 500, json: { ok: true, job: { state: 'complete' } } }],
  ['HTTP 404', { status: 404, json: { ok: false, error: 'not found' } }],
  ['invalid JSON', { contentType: 'application/json', body: '{' }],
  ['missing job', { json: { ok: true } }],
  ['null job', { json: { ok: true, job: null } }],
  ['unknown state', { json: { ok: true, job: { state: 'unknown' } } }],
  ['unsuccessful envelope', { json: { ok: false, job: { state: 'complete' } } }],
];

const trackedJob = (page, repo = REPO_A) => page.evaluate(async repoId =>
  (await import('/js/features/rapid-model-download.js')).activeRapidDownloadJob(repoId), repo);

async function freezePollClock(page) {
  await page.clock.install({ time: new Date('2030-01-01T00:00:00Z') });
  await page.clock.pauseAt(new Date('2030-01-01T00:00:01Z'));
}

// Advance browser timers while waiting on the real poller's asynchronous fetch chain.
async function nextPoll(page, fake, id = 'job-1', waitForResponse = true) {
  const response = waitForResponse ? page.waitForResponse(resp =>
    new URL(resp.url()).pathname === `/api/models/downloads/${id}` && resp.request().method() === 'GET') : null;
  const before = fake.polls[id] || 0;
  await expect.poll(async () => {
    await page.clock.runFor(1500);
    return fake.polls[id] || 0;
  }).toBeGreaterThan(before);
  if (response) {
    await (await response).finished();
    await page.evaluate(() => {}); // let the browser consume the response before asserting retained state
  }
}

async function pollAfterDelay(page, fake, delay) {
  const before = fake.polls['job-1'];
  const response = page.waitForResponse(resp =>
    new URL(resp.url()).pathname === '/api/models/downloads/job-1' && resp.request().method() === 'GET');
  await page.clock.runFor(delay - 1);
  expect(fake.polls['job-1']).toBe(before);
  await page.clock.runFor(1);
  await (await response).finished();
  await page.evaluate(() => {});
  expect(fake.polls['job-1']).toBe(before + 1);
}

/** In-test fake of the download + model-status endpoints. */
async function fakeBackend(page) {
  const fake = {
    statusRequests: [], starts: [], polls: {}, cancels: [], jobs: {},
    holdStart: null, holdPoll: null, pollResponses: {}, cancelResponses: [], models: structuredClone(MODELS),
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
    fake.jobs[id] = {
      repo_id: body.repo_id, revision: 'main', engine: 'rapid-mlx',
      state: 'running', message: '', error: null, local_path: null,
      bytes_done: GiB, bytes_total: 4 * GiB, current_file: 'weights.safetensors',
      stalled: false, restarts: 0,
    };
    return route.fulfill({ json: { ok: true, job_id: id } });
  });
  await page.route('**/api/models/downloads/**', async route => {
    const url = new URL(route.request().url());
    const [, id, action] = url.pathname.match(/downloads\/([^/]+)(?:\/(cancel))?$/) || [];
    if (action === 'cancel') {
      fake.cancels.push(id);
      const response = fake.cancelResponses.shift();
      if (response) return route.fulfill(response);
      if (fake.jobs[id]) fake.jobs[id].state = 'cancelled';
      return route.fulfill({ json: { ok: true } });
    }
    fake.polls[id] = (fake.polls[id] || 0) + 1;
    if (fake.holdPoll) await fake.holdPoll;
    const response = fake.pollResponses[id]?.shift();
    if (response) return route.fulfill(response);
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

  for (const [label, response] of transientPollResponses) {
    test(`${label} preserves wizard progress and tracking until a valid poll completes`, async ({ page }) => {
      const fake = await fakeBackend(page);
      await freezePollClock(page);
      await openWizard(page);
      await select(page, 'alias-a');
      await page.locator('#rapid-dlp-download-btn').click();
      await expect(page.locator('#rapid-dlp-stats')).toContainText('1.0 GiB / 4.0 GiB');
      fake.pollResponses['job-1'] = [response];
      await nextPoll(page, fake);

      await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
      await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('25%');
      await expect(page.locator('#rapid-dlp-download-btn')).toBeDisabled();
      await expect(page.locator('#rapid-dlp-cancel-btn')).toBeEnabled();
      expect(await trackedJob(page)).toBe('job-1');
      expect(await missingRepo(page)).toBe(REPO_A);
      await select(page, 'alias-a');
      await page.evaluate(() => document.getElementById('rapid-dlp-download-btn').click());
      expect(fake.starts).toEqual([REPO_A]);

      fake.jobs['job-1'].bytes_done = 2 * GiB;
      await nextPoll(page, fake);
      await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('50%');
      expect(await trackedJob(page)).toBe('job-1');
      fake.jobs['job-1'].state = 'complete';
      await nextPoll(page, fake);
      await expect(page.locator('#rapid-dlp-complete')).toBeVisible();
      expect(await trackedJob(page)).toBeNull();
      expect(await missingRepo(page)).toBeNull();
      expect(fake.starts).toEqual([REPO_A]);
    });
  }

  test('queued wizard jobs remain active and an explicit failed state alone offers retry', async ({ page }) => {
    const fake = await fakeBackend(page);
    await freezePollClock(page);
    await openWizard(page);
    await select(page, 'alias-a');
    fake.pollResponses['job-1'] = [{ json: { ok: true, job: { state: 'queued' } } }];
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress')).toBeVisible();
    await expect.poll(() => trackedJob(page)).toBe('job-1');
    await expect(page.locator('#rapid-dlp-download-btn')).toBeDisabled();
    await nextPoll(page, fake);
    await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('25%');
    fake.jobs['job-1'].state = 'failed';
    fake.jobs['job-1'].error = 'disk full';
    await nextPoll(page, fake);
    await expect(page.locator('#rapid-dlp-idle')).toBeVisible();
    await expect(page.locator('#rapid-dlp-note')).toContainText('disk full');
    await expect(page.locator('#rapid-dlp-download-btn')).toBeEnabled();
    expect(await trackedJob(page)).toBeNull();
  });

  test('wizard poll errors back off to a bounded delay and success restores normal polling', async ({ page }) => {
    const fake = await fakeBackend(page);
    await freezePollClock(page);
    await openWizard(page);
    await select(page, 'alias-a');
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('25%');
    fake.pollResponses['job-1'] = Array.from({ length: 7 }, () => transientPollResponses[0][1]);
    for (const delay of [1500, 3000, 6000, 12000, 24000, 30000, 30000]) {
      await pollAfterDelay(page, fake, delay);
      expect(await trackedJob(page)).toBe('job-1');
    }
    fake.jobs['job-1'].bytes_done = 2 * GiB;
    await pollAfterDelay(page, fake, 30000);
    await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('50%');
    await pollAfterDelay(page, fake, 1500);
    expect(fake.starts).toEqual([REPO_A]);
  });

  test('switching sources during a failed poll detaches and later resumes the same job', async ({ page }) => {
    const fake = await fakeBackend(page);
    await freezePollClock(page);
    await openWizard(page);
    await select(page, 'alias-a');
    await page.locator('#rapid-dlp-download-btn').click();
    await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('25%');
    let release;
    fake.holdPoll = new Promise(resolve => { release = resolve; });
    fake.pollResponses['job-1'] = [transientPollResponses[0][1]];
    await nextPoll(page, fake, 'job-1', false);
    await select(page, 'alias-b');
    fake.holdPoll = null;
    release();
    await expect(page.locator('#rapid-dlp-repo')).toContainText(REPO_B);
    expect(await trackedJob(page)).toBe('job-1');
    const before = fake.polls['job-1'];
    await page.clock.runFor(30_000);
    expect(fake.polls['job-1']).toBe(before);
    await select(page, 'alias-a');
    await expect(page.locator('#rapid-dlp-progress-pct')).toHaveText('25%');
    expect(fake.starts).toEqual([REPO_A]);
    await page.locator('#rapid-dlp-cancel-btn').click();
    await nextPoll(page, fake);
    await expect(page.locator('#rapid-dlp-note')).toContainText('Download cancelled');
    expect(await trackedJob(page)).toBeNull();
    expect(fake.cancels).toEqual(['job-1']);
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

  for (const [label, response] of transientPollResponses) {
    test(`${label} preserves card progress and cancel until polling succeeds`, async ({ page }) => {
      const fake = await fakeBackend(page);
      await freezePollClock(page);
      await mountCard(page, 'card-1');
      await page.locator('#card-1 .launch-card-dl-btn').click();
      await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
      fake.pollResponses['job-1'] = [response];
      await nextPoll(page, fake);
      await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
      await expect(page.locator('#card-1 .launch-card-dl-btn')).toBeDisabled();
      await expect(page.locator('#card-1 .launch-card-dl-cancel')).toBeVisible();
      await expect(page.locator('#card-1 .launch-card-dl-cancel')).toBeEnabled();
      expect(await trackedJob(page)).toBe('job-1');
      await page.locator('#card-1 .launch-card-btn-start').click();
      expect(fake.starts).toEqual([REPO_A]);

      fake.jobs['job-1'].bytes_done = 2 * GiB;
      await nextPoll(page, fake);
      await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 50%');
      fake.jobs['job-1'].state = 'complete';
      await nextPoll(page, fake);
      await expect(page.locator('#card-1 .launch-card-dl')).toHaveCount(0);
      await expect(page.locator('#card-1')).not.toHaveAttribute('data-model-missing', '1');
      expect(await trackedJob(page)).toBeNull();
      expect(fake.starts).toEqual([REPO_A]);
    });
  }

  for (const state of ['failed', 'cancelled']) {
    test(`queued card jobs retain tracking until explicit ${state}`, async ({ page }) => {
      const fake = await fakeBackend(page);
      await freezePollClock(page);
      await mountCard(page, 'card-1');
      fake.pollResponses['job-1'] = [{ json: { ok: true, job: { state: 'queued' } } }];
      await page.locator('#card-1 .launch-card-dl-btn').click();
      await expect.poll(() => trackedJob(page)).toBe('job-1');
      await expect(page.locator('#card-1 .launch-card-dl-btn')).toBeDisabled();
      await expect(page.locator('#card-1 .launch-card-dl-cancel')).toBeVisible();
      await nextPoll(page, fake);
      await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
      fake.jobs['job-1'].state = state;
      await nextPoll(page, fake);
      await expect(page.locator('#card-1 .launch-card-dl-btn')).toBeEnabled();
      await expect(page.locator('#card-1 .launch-card-dl-btn')).toHaveText(state === 'failed' ? 'Retry' : 'Resume');
      expect(await trackedJob(page)).toBeNull();
    });
  }

  test('card poll errors back off to a bounded delay and success restores normal polling', async ({ page }) => {
    const fake = await fakeBackend(page);
    await freezePollClock(page);
    await mountCard(page, 'card-1');
    await page.locator('#card-1 .launch-card-dl-btn').click();
    await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
    fake.pollResponses['job-1'] = Array.from({ length: 7 }, () => transientPollResponses[0][1]);
    for (const delay of [1500, 3000, 6000, 12000, 24000, 30000, 30000]) {
      await pollAfterDelay(page, fake, delay);
      expect(await trackedJob(page)).toBe('job-1');
    }
    fake.jobs['job-1'].bytes_done = 2 * GiB;
    await pollAfterDelay(page, fake, 30000);
    await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 50%');
    await pollAfterDelay(page, fake, 1500);
    expect(fake.starts).toEqual([REPO_A]);
  });

  test('a failed cancellation request leaves card cancel usable while poll status is unavailable', async ({ page }) => {
    const fake = await fakeBackend(page);
    await freezePollClock(page);
    await mountCard(page, 'card-1');
    await page.locator('#card-1 .launch-card-dl-btn').click();
    await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
    fake.pollResponses['job-1'] = [transientPollResponses[0][1]];
    await nextPoll(page, fake);
    fake.cancelResponses = [{ status: 500, json: { ok: false } }];
    await page.locator('#card-1 .launch-card-dl-cancel').click();
    await expect(page.locator('#card-1 .launch-card-dl-cancel')).toBeEnabled();
    expect(await trackedJob(page)).toBe('job-1');
    await page.locator('#card-1 .launch-card-dl-cancel').click();
    await nextPoll(page, fake);
    await expect(page.locator('#card-1 .launch-card-dl-btn')).toHaveText('Resume');
    expect(fake.cancels).toEqual(['job-1', 'job-1']);
    expect(await trackedJob(page)).toBeNull();
  });

  test('a card removed during a failed poll keeps tracking and its replacement can cancel', async ({ page }) => {
    const fake = await fakeBackend(page);
    await freezePollClock(page);
    await mountCard(page, 'card-1');
    await page.locator('#card-1 .launch-card-dl-btn').click();
    await expect(page.locator('#card-1 .launch-card-dl-label')).toContainText('Downloading 25%');
    let release;
    fake.holdPoll = new Promise(resolve => { release = resolve; });
    fake.pollResponses['job-1'] = [transientPollResponses[0][1]];
    await nextPoll(page, fake, 'job-1', false);
    await page.evaluate(() => document.getElementById('card-1').remove());
    fake.holdPoll = null;
    release();
    const before = fake.polls['job-1'];
    await page.clock.runFor(30_000);
    expect(fake.polls['job-1']).toBe(before);
    expect(await trackedJob(page)).toBe('job-1');
    await mountCard(page, 'card-2');
    await expect(page.locator('#card-2 .launch-card-dl-label')).toContainText('Downloading 25%');
    await page.locator('#card-2 .launch-card-dl-cancel').click();
    await nextPoll(page, fake);
    await expect(page.locator('#card-2 .launch-card-dl-btn')).toHaveText('Resume');
    expect(await trackedJob(page)).toBeNull();
    expect(fake.starts).toEqual([REPO_A]);
    expect(fake.cancels).toEqual(['job-1']);
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
