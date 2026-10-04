// ── Performance Baseline ──────────────────────────────────────────────────────

import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { join, dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASELINE_FILE = join(__dirname, 'js-module-baseline.json');

function readBaseline() {
  return JSON.parse(readFileSync(BASELINE_FILE, 'utf8'));
}

// Wall-clock numbers depend on the machine as much as on the app: a busy laptop or a shared
// CI runner can double them. The JS module count is deterministic and stays a hard gate. The
// time limits are only there to catch a hang or an order-of-magnitude regression, so they are
// deliberately generous, scale with PERF_BUDGET_SCALE for slow runners, and the measured values
// are attached to the test report so a trend is still visible.
const BUDGET_SCALE = Number(process.env.PERF_BUDGET_SCALE) > 0 ? Number(process.env.PERF_BUDGET_SCALE) : 1;
const MODULES_READY_BUDGET_MS = 20_000 * BUDGET_SCALE;
const DASHBOARD_UPDATE_BUDGET_MS = 2_000 * BUDGET_SCALE;

test.describe('performance baseline', () => {
  test('cold load network and timing', async ({ page }, testInfo) => {
    const requests = [];
    page.on('request', req => {
      const url = req.url();
      if (url.endsWith('.js') || url.endsWith('.css')) {
        requests.push({ url: url.replace('http://127.0.0.1:7778', ''), method: req.method() });
      }
    });

    const startTime = Date.now();
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');
    const modulesReadyTime = Date.now() - startTime;

    const jsRequests = requests.filter(r => r.url.endsWith('.js'));
    const cssRequests = requests.filter(r => r.url.endsWith('.css'));

    console.log('=== Performance Baseline ===');
    console.log(`Time to modules-ready: ${modulesReadyTime}ms`);
    console.log(`JS requests: ${jsRequests.length}`);
    console.log(`CSS requests: ${cssRequests.length}`);
    console.log(`Total asset requests: ${requests.length}`);
    console.log('JS files loaded:');
    jsRequests.forEach(r => console.log(`  ${r.url}`));

    const baseline = readBaseline();
    expect(
      jsRequests.length,
      `JS module count regressed: ${jsRequests.length} > ${baseline.count}. ` +
      `Run \`cd tests/ui && npm run update-baseline\` after verifying the new modules are intentional.`,
    ).toBeLessThanOrEqual(baseline.count);
    testInfo.annotations.push({ type: 'modules-ready-ms', description: String(modulesReadyTime) });
    expect(
      modulesReadyTime,
      `modules-ready took ${modulesReadyTime}ms (budget ${MODULES_READY_BUDGET_MS}ms). ` +
      'That is far outside normal; set PERF_BUDGET_SCALE for a known-slow runner.',
    ).toBeLessThan(MODULES_READY_BUDGET_MS);
  });

  test('dashboard update path timing', async ({ page }, testInfo) => {
    await page.goto('/');
    await page.waitForSelector('html.modules-ready');

    const updateMs = await page.evaluate(() => {
      const start = performance.now();
      if (typeof window.updateDashboard === 'function') {
        window.updateDashboard({});
      }
      return performance.now() - start;
    });

    console.log(`Dashboard update time: ${Math.round(updateMs)}ms`);
    testInfo.annotations.push({ type: 'dashboard-update-ms', description: String(Math.round(updateMs)) });
    expect(updateMs).toBeLessThan(DASHBOARD_UPDATE_BUDGET_MS);
  });
});
