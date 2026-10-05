/**
 * update-baseline.mjs — re-measures the JS module count with a headless
 * cold load against an ephemeral static asset server and writes the new count to
 * core/js-module-baseline.json.
 *
 * Usage:  cd tests/ui && npm run update-baseline
 *
 * Set LLAMA_MONITOR_UI_URL to measure a running app; otherwise the script
 * serves static/index.html itself and never starts llama-monitor.
 */

import { chromium } from '@playwright/test';
import { createServer } from 'http';
import { writeFileSync, readFileSync, existsSync, statSync } from 'fs';
import { fileURLToPath } from 'url';
import { join, dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASELINE_FILE = join(__dirname, 'core/js-module-baseline.json');
const CONFIGURED_BASE_URL = process.env.LLAMA_MONITOR_UI_URL || '';

function startStaticServer() {
  const repoRoot = join(__dirname, '..', '..');
  const staticRoot = join(repoRoot, 'static');
  const server = createServer((req, res) => {
    const requestPath = decodeURIComponent((req.url || '/').split('?')[0]);
    const relative = requestPath === '/' ? 'index.html' : requestPath.replace(/^\/+/, '');
    const filePath = join(staticRoot, relative);
    if (!filePath.startsWith(staticRoot) || !existsSync(filePath) || !statSync(filePath).isFile()) {
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end('{}');
      return;
    }
    const contentType = filePath.endsWith('.html') ? 'text/html' : filePath.endsWith('.css') ? 'text/css' : 'application/javascript';
    res.writeHead(200, { 'content-type': contentType });
    res.end(readFileSync(filePath));
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, baseUrl: `http://127.0.0.1:${port}` });
    });
  });
}

export default async function updateBaseline() {
  const local = CONFIGURED_BASE_URL ? null : await startStaticServer();
  const baseUrl = CONFIGURED_BASE_URL || local.baseUrl;
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  const jsFiles = [];
  page.on('request', req => {
    const url = req.url();
    if (url.endsWith('.js')) {
      jsFiles.push(url.replace(baseUrl, ''));
    }
  });

  try {
    await page.goto(baseUrl, { waitUntil: 'load' });
    await page.waitForSelector('html.modules-ready', { timeout: 15000 });
    // The update checker is intentionally imported from requestIdleCallback (3s timeout)
    // after modules-ready. Wait until it has been requested so the baseline records the
    // whole normal cold-load closure instead of racing the idle callback. The performance
    // spec waits on the same condition.
    await page.waitForFunction(
      () => performance.getEntriesByType('resource').some(e => e.name.endsWith('/js/features/updates.js')),
      null,
      { timeout: 10000 },
    );
  } finally {
    await browser.close();
    local?.server.close();
  }

  const existing = JSON.parse(readFileSync(BASELINE_FILE, 'utf8'));
  const newCount = jsFiles.length;

  if (newCount === existing.count) {
    console.log(`Baseline unchanged: ${newCount} JS modules`);
    return;
  }

  console.log(`Baseline updated: ${existing.count} → ${newCount} JS modules`);
  jsFiles.forEach(f => console.log(`  ${f}`));

  writeFileSync(BASELINE_FILE, JSON.stringify(
    { count: newCount, note: existing.note },
    null, 2,
  ) + '\n');
  console.log(`Written to ${BASELINE_FILE}`);
}

// Run directly
updateBaseline().catch(err => { console.error(err); process.exit(1); });
