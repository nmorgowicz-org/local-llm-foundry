import { test, expect } from '@playwright/test';

// Installing a Rapid-MLX runtime from Settings used to look hung forever: the install ran and
// finished on the server, but the client polled with the wrong response shape (it looked for a
// wrapping `job` key and the state name `completed`; the server sends the bare snapshot and
// `complete`), so it stopped silently on the first tick, never refreshed, and left the button on
// "Installing…". These tests use the server's real wire shapes. Every /api/rapid-mlx route is
// mocked and anything unlisted is refused, so no test can touch a real runtime.
test.describe('Rapid-MLX runtime install', () => {
  const json = (route, body, status = 200) => route.fulfill({
    status, contentType: 'application/json', body: JSON.stringify(body),
  });

  const ACTIVE = {
    environment_id: '0.15.5-aaaa', version: '0.15.5', source_kind: 'release', release_channel: 'stable',
    extras: ['guided', 'vision'], active: true, rollback_candidate: false, complete: true,
  };

  test('@in-memory-test a finished install refreshes the header, the pill and the Install button', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    let installed = false;
    let jobPolls = 0;
    let installBody = null;
    await page.route('**/api/rapid-mlx/**', route => route.abort('blockedbyclient'));
    await page.route('**/api/db/admin-token', route => json(route, { token: 'admin-test-token' }));
    await page.route('**/api/rapid-mlx/runtime/status', route => json(route, {
      runtime: { supported: true, installer_available: true, mutation_in_progress: false, rollback_available: false,
        active: installed ? ACTIVE : null, inventory: installed ? [ACTIVE] : [] },
      jobs: [],
    }));
    await page.route('**/api/rapid-mlx/runtime/releases', route => json(route, {
      releases: [{ version: '0.15.5', channel: 'stable', published_at: new Date().toISOString(), release_notes: '' }],
    }));
    await page.route('**/api/rapid-mlx/runtime/install', route => {
      installBody = route.request().postDataJSON();
      return json(route, { job_id: 'job-1', state: 'queued' }, 202);
    });
    await page.route('**/api/rapid-mlx/runtime/jobs/job-1', route => {
      jobPolls += 1;
      if (jobPolls < 2) {
        return json(route, { id: 'job-1', operation: 'install', state: 'running', version: '0.15.5',
          message: 'Installing and validating an isolated runtime', result: null });
      }
      installed = true;
      return json(route, { id: 'job-1', operation: 'install', state: 'complete', version: '0.15.5',
        message: 'Runtime validated and activated', result: null });
    });

    await page.evaluate(() => document.getElementById('rapid-mlx-manage-btn').click());
    await expect(page.locator('#rapid-mlx-modal')).toHaveClass(/open/);
    const status = page.locator('#rapid-mlx-status-text');
    await expect(status).toHaveText('No Rapid-MLX runtime installed.');

    const installBtn = page.locator('.rapid-mlx-install-btn[data-version="0.15.5"]');
    await installBtn.click();
    await expect.poll(() => installBody?.confirm).toBe('INSTALL_RAPID_MLX_RUNTIME');

    // While it runs, the header says what is happening rather than sitting still.
    await expect(status).toContainText('Installing and validating an isolated runtime');

    // When the server reports `complete`, the whole manager updates without a reopen.
    await expect(status).toHaveText('Rapid-MLX v0.15.5 is active.', { timeout: 15000 });
    await expect(page.locator('#rapid-mlx-pill')).toBeVisible();
    await expect(page.locator('#rapid-mlx-pill-version')).toHaveText('Rapid-MLX · v0.15.5');
    await expect(installBtn).toHaveCount(0);
  });

  test('@in-memory-test a failed install says so and restores the Install button', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await page.route('**/api/rapid-mlx/**', route => route.abort('blockedbyclient'));
    await page.route('**/api/db/admin-token', route => json(route, { token: 'admin-test-token' }));
    await page.route('**/api/rapid-mlx/runtime/status', route => json(route, {
      runtime: { supported: true, installer_available: true, mutation_in_progress: false, rollback_available: false,
        active: null, inventory: [] },
      jobs: [],
    }));
    await page.route('**/api/rapid-mlx/runtime/releases', route => json(route, {
      releases: [{ version: '0.15.5', channel: 'stable', published_at: new Date().toISOString(), release_notes: '' }],
    }));
    await page.route('**/api/rapid-mlx/runtime/install', route => json(route, { job_id: 'job-2', state: 'queued' }, 202));
    await page.route('**/api/rapid-mlx/runtime/jobs/job-2', route => json(route, {
      id: 'job-2', operation: 'install', state: 'failed', version: '0.15.5',
      message: 'Managed Rapid-MLX validation failed safely; the active runtime was not changed', result: null,
    }));

    await page.evaluate(() => document.getElementById('rapid-mlx-manage-btn').click());
    await expect(page.locator('#rapid-mlx-modal')).toHaveClass(/open/);
    const installBtn = page.locator('.rapid-mlx-install-btn[data-version="0.15.5"]');
    await installBtn.click();

    await expect(page.locator('.toast, [role="status"]').filter({ hasText: 'operation failed' }).first())
      .toBeVisible({ timeout: 10000 });
    await expect(installBtn).toHaveText('Install');
    await expect(installBtn).toBeEnabled();
  });

  test('@in-memory-test the evidence drawer opens above the Settings modal', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    const covered = await page.evaluate(async () => {
      const settings = document.getElementById('settings-modal');
      settings.removeAttribute('inert');
      settings.classList.add('open');
      const { openEvidenceDrawer } = await import('/js/features/evidence-drawer.js');
      const opener = document.createElement('button');
      document.body.appendChild(opener);
      openEvidenceDrawer({
        title: 'Rapid-MLX runtime support', status: 'caution', summary: 'stacking probe',
        consequence: '', remediation: '', evidence: [], warnings: [], provenance: [],
      }, opener);
      await new Promise(resolve => setTimeout(resolve, 400));
      const hit = document.elementFromPoint(window.innerWidth - 120, 160);
      return !!hit?.closest('.evidence-drawer');
    });
    expect(covered).toBe(true);
  });
});
