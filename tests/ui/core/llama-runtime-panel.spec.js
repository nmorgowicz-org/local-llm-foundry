import { test, expect } from '@playwright/test';
import { dismissAuthShell } from '../helpers.js';

const OK_CHECK = (id, title) => ({ id, title, status: 'pass', detail: 'ok', fix: null, evidence: [] });

function report(checks) {
  return {
    os: 'macos', accelerator: 'metal', checks, fingerprint: 'f', validated_at_unix: 1_700_000_000,
    ready: !checks.some(c => c.status === 'fail'),
  };
}

const SOURCE_PROVENANCE = {
  source: 'foundry_source_build', label: 'b11739 · src · Metal', reference: 'b11739',
  commit: '0ce8cd11416c8840498e6701bcbe39ad4b858c2b', accelerator: 'metal',
  fa_quants_spec: null, cuda_architectures: null, hashes_match: true,
};

const IDLE_STATUS = {
  ok: true, os: 'macos', accelerators: ['metal'], default_accelerator: 'metal',
  active_job_id: null, last_job: null, recovery_required: null, owner: null,
};

/** One mutable fake of the build API so each test states only what differs. */
function makeApi(overrides = {}) {
  return {
    // A shallow source checkout embeds `build 1`; the pill must show the recorded tag instead.
    version: { ok: true, build: 1, provenance: SOURCE_PROVENANCE },
    status: IDLE_STATUS,
    prereqs: { ok: true, accelerator: 'metal', report: report([OK_CHECK('cmake', 'CMake 4.1')]), stale: false, validating: false },
    options: {
      ok: true, accelerator: 'metal',
      refs: [
        { id: 'latest_release', label: 'Latest release', reference: 'b11740', commit: 'a'.repeat(40) },
        { id: 'master', label: 'master HEAD', reference: 'master', commit: 'b'.repeat(40) },
      ],
      refs_error: null, cuda_arch_auto: null, cuda_arch_choices: [], jobs_auto: 8, show_cuda: false,
      ccache_cap_bytes: 2 * 1024 ** 3,
    },
    cache: {
      ok: true, source_bytes: 412 * 1024 ** 2, ccache_bytes: 0, ccache_cap_bytes: 2 * 1024 ** 3,
      build_trees: [{ accelerator: 'metal', bytes: 6 * 1024 ** 3 }], previous_bytes: 310 * 1024 ** 2, owner: null,
    },
    runtime: {
      ok: true, busy_owner: null, live: { label: 'b11739 · src · Metal' }, previous: { label: 'b11720 prebuilt' },
      previous_available: true, transaction: { state: 'clean' }, last: null,
    },
    job: null,
    log: { ok: true, lines: [], next: 0, truncated: false },
    ...overrides,
  };
}

async function boot(page, api) {
  await page.routeWebSocket('**/ws', ws => ws.close());
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const json = body => route.fulfill({ json: body });
    if (path === '/api/llama-binary/version') return json(api.version);
    if (path === '/api/llama-binary/latest') return json({ build: api.version.build });
    if (path === '/api/llama-binary/releases') return json({ releases: [] });
    if (path === '/api/llama-binary/runtime-status') return json(api.runtime);
    if (path === '/api/llama-build/status') return json(api.status);
    if (path === '/api/llama-build/prereqs') return json(api.prereqs);
    if (path === '/api/llama-build/options') return json(api.options);
    if (path === '/api/llama-build/cache') return json(api.cache);
    if (path.startsWith('/api/llama-build/jobs/') && path.endsWith('/log')) return json(api.log);
    if (path.startsWith('/api/llama-build/jobs/')) return json(api.job);
    if (path === '/api/llama-build/start') {
      api.startBody = route.request().postDataJSON();
      return route.fulfill({ status: 202, json: { ok: true, job_id: 'j1', resolved_commit: 'a'.repeat(40) } });
    }
    return route.fulfill({ status: 404, json: { ok: false } });
  });
  await page.goto('/');
  await page.waitForSelector('html.modules-ready');
  await dismissAuthShell(page);
}

const openPanel = page =>
  page.evaluate(async () => (await import('/js/features/llama-runtime-panel.js')).openRuntimePanel());

const failedJob = {
  ok: true, id: 'j1', phase: 'configure', status: 'failed', progress: null, cancellable: false,
  outcome: 'configure failed', started_at: 1, finished_at: 2, log_cursor: 3, restart_failed: false, accelerator: 'metal',
  failure: {
    phase: 'configure', summary: 'configure failed', exit_code: 1,
    first_error: 'CMake Error at vendor/cpp-httplib/CMakeLists.txt:151',
    log_tail: ['-- configuring', 'CMake Error at vendor/cpp-httplib/CMakeLists.txt:151'], log_path: '/x/j1.log',
    hint: { id: 'stale_cache', message: 'stale CMake cache', fix: { kind: 'action', id: 'wipe_build_dir', label: 'Wipe build dir & retry' } },
  },
};

const lastJobStatus = (status, phase) => ({
  ...IDLE_STATUS, last_job: { id: 'j1', status, phase, failure_phase: phase },
});

test.describe('llama.cpp runtime panel', () => {
  test('a failing prerequisite disables Build and names the check', async ({ page }) => {
    const ninja = { id: 'ninja', title: 'Ninja', status: 'fail', detail: 'Ninja not found', fix: { kind: 'copy', command: 'brew install ninja' }, evidence: [] };
    await boot(page, makeApi({
      prereqs: { ok: true, accelerator: 'metal', report: report([OK_CHECK('cmake', 'CMake 4.1'), ninja]), stale: false, validating: false },
    }));
    await openPanel(page);
    await expect(page.locator('[data-lrp="build"]')).toBeDisabled();
    await expect(page.locator('[data-lrp="build-reason"]')).toContainText('Ninja');
    await expect(page.locator('[data-check="ninja"] button', { hasText: 'Copy fix' })).toBeVisible();
  });

  test('a warning does not disable Build, and Build posts the chosen source without a time estimate', async ({ page }) => {
    const ccache = { id: 'ccache', title: 'ccache', status: 'warn', detail: 'not installed', fix: null, evidence: [] };
    const api = makeApi({
      prereqs: { ok: true, accelerator: 'metal', report: report([OK_CHECK('cmake', 'CMake 4.1'), ccache]), stale: false, validating: false },
    });
    await boot(page, api);
    await openPanel(page);
    await expect(page.locator('[data-lrp="build"]')).toBeEnabled();
    await page.locator('input[name="lrp-ref"][value="master"]').check();
    await page.locator('#lrp-headroom').check();
    await page.locator('[data-lrp="build"]').click();
    await expect.poll(() => api.startBody).toMatchObject({ accelerator: 'metal', ref: 'master', headroom: true, wipe: false });
    expect(Object.keys(api.startBody)).not.toContain('estimated_seconds');
  });

  test('a failed job turns the pill red; the panel shows full diagnostics with the hint action', async ({ page }) => {
    await boot(page, makeApi({ status: lastJobStatus('failed', 'configure'), job: failedJob }));
    const pill = page.locator('#llama-pill');
    await expect(pill).toHaveAttribute('data-state', 'error');
    await expect(pill).toContainText('Build failed: configure');
    await pill.click();
    const banner = page.locator('[data-lrp="error-banner"]');
    await expect(banner).toContainText('CONFIGURE');
    await expect(banner).toContainText('CMake Error at vendor/cpp-httplib/CMakeLists.txt:151');
    await expect(banner.locator('[data-lrp="hint-wipe_build_dir"]')).toBeVisible();
    await page.locator('[data-lrp="dismiss"]').click();
    await expect(banner).toHaveCount(0);
    await expect(pill).not.toHaveAttribute('data-state', 'error');
  });

  test('source-build provenance appears on the pill and in the panel', async ({ page }) => {
    await boot(page, makeApi());
    await expect(page.locator('#llama-pill')).toContainText('b11739 · src');
    await page.locator('#llama-pill').click();
    await page.locator('#llama-open-runtime-panel').click();
    await expect(page.locator('[data-lrp="current-line"]')).toContainText('source build');
    await expect(page.locator('[data-lrp="provenance-hashes"]')).toContainText('match');
    await expect(page.locator('[data-lrp="rollback"]')).toHaveText('Roll back to b11720 prebuilt');
  });

  test('a master source build shows its reference and never a bogus update', async ({ page }) => {
    await boot(page, makeApi({ version: { ok: true, build: 1, provenance: { ...SOURCE_PROVENANCE, reference: 'master', label: 'master · 6fa4f6c · src · Metal' } } }));
    const pill = page.locator('#llama-pill');
    await expect(pill).toContainText('llama.cpp · master · src');
    await expect(pill).not.toHaveAttribute('data-state', 'update');
  });

  test('a provenance hash mismatch is called out', async ({ page }) => {
    await boot(page, makeApi({ version: { ok: true, build: 1, provenance: { ...SOURCE_PROVENANCE, hashes_match: false } } }));
    await openPanel(page);
    await expect(page.locator('[data-lrp="provenance-hashes"]')).toContainText('do not match');
  });

  test('replacing a source build shows the conditional prebuilt caution, without a guessed pair list', async ({ page }) => {
    await boot(page, makeApi());
    await page.locator('#llama-pill').click();
    const warning = page.locator('#llama-source-warning');
    await expect(warning).toBeVisible();
    await expect(warning).toContainText('may differ');
    await expect(warning).not.toContainText(/q4_0|q8_0|f16/);
  });

  test('a prebuilt runtime gets no downgrade caution', async ({ page }) => {
    await boot(page, makeApi({ version: { ok: true, build: 11720, provenance: { source: 'upstream_prebuilt', label: 'b11720 prebuilt', hashes_match: null } } }));
    await page.locator('#llama-pill').click();
    await expect(page.locator('#llama-version-modal')).toBeVisible();
    await expect(page.locator('#llama-source-warning')).toBeHidden();
  });

  test('restart failure is an amber installed state with diagnostics and rollback', async ({ page }) => {
    const job = { ...failedJob, status: 'succeeded_restart_failed', phase: 'done', restart_failed: true, outcome: 'Installed b11740 · src · metal. The model restart failed.', failure: null };
    await boot(page, makeApi({ status: lastJobStatus('succeeded_restart_failed', 'done'), job }));
    const pill = page.locator('#llama-pill');
    await expect(pill).toHaveAttribute('data-state', 'restart-failed');
    await expect(pill).toContainText('Runtime installed; model restart failed');
    await pill.click();
    await expect(page.locator('[data-lrp="restart-banner"]')).toContainText('model restart failed');
    await expect(page.locator('[data-lrp="rollback"]')).toBeVisible();
    await expect(page.locator('[data-lrp="restart-banner"] button', { hasText: 'Copy diagnostics' })).toBeVisible();
  });

  test('a running build shows progress; Cancel is disabled once promotion begins', async ({ page }) => {
    const running = { ok: true, id: 'j2', phase: 'compile', status: 'running', progress: { done: 63, total: 100 }, cancellable: true, outcome: null, failure: null, started_at: 1, finished_at: null, log_cursor: 2, restart_failed: false, accelerator: 'metal' };
    const api = makeApi({
      status: { ...IDLE_STATUS, active_job_id: 'j2', owner: 'build' },
      job: running,
      log: { ok: true, lines: ['[63/100] Building CXX object x'], next: 1, truncated: false },
    });
    await boot(page, api);
    const pill = page.locator('#llama-pill');
    await expect(pill).toHaveAttribute('data-state', 'building');
    await expect(pill).toContainText('Building · compile 63%');
    await pill.click();
    await expect(page.locator('[data-lrp="log"]')).toContainText('[63/100] Building CXX object x');
    await expect(page.locator('[data-lrp="build"]')).toBeDisabled();
    await expect(page.locator('[data-lrp="cancel"]')).toBeEnabled();
    api.job = { ...running, phase: 'promote', progress: null, cancellable: false };
    await expect(page.locator('[data-lrp="cancel"]')).toBeDisabled({ timeout: 10_000 });
    await expect(page.locator('[data-lrp="cancel-reason"]')).toContainText('cannot cancel safely');
  });

  test('clean is disabled while another owner holds the build tree', async ({ page }) => {
    await boot(page, makeApi({
      cache: { ok: true, source_bytes: 1, ccache_bytes: 0, ccache_cap_bytes: 1, build_trees: [], previous_bytes: 0, owner: 'validate' },
    }));
    await openPanel(page);
    await expect(page.locator('[data-lrp="clean"]')).toBeDisabled();
    await expect(page.locator('[data-lrp="clean"]')).toHaveAttribute('title', /validate/);
  });

  test('CUDA controls are absent when the platform has none', async ({ page }) => {
    await boot(page, makeApi());
    await openPanel(page);
    await expect(page.locator('[data-lrp="cuda-controls"]')).toHaveCount(0);
    await expect(page.locator('#lrp-arch')).toHaveCount(0);
  });

  test('server text is never interpreted as HTML', async ({ page }) => {
    const hostile = '<img src=x onerror="window.__pwned=1">';
    await boot(page, makeApi({
      status: lastJobStatus('failed', 'configure'),
      job: { ...failedJob, failure: { ...failedJob.failure, first_error: hostile, hint: null } },
    }));
    await openPanel(page);
    await expect(page.locator('[data-lrp="error-banner"] pre.lrp-first-error')).toContainText('<img');
    expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
  });
});
