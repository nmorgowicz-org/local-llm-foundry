// playwright.config.js
//
// Test tags (the full CI suite still runs all non-skipped tests; focused npm scripts select them):
//   @in-memory-test     — runs JS logic against fake state in page.evaluate(); no backend
//   @fake-data-bypass   — uses mocked API responses or injected DOM data
//   @runtime-required   — needs a live model instance; gated behind env flag
//
//   npm run test:in-memory
//   npm run test:fake-data
//   npm run test:runtime
//
// LLAMA_MONITOR_HAS_RUNTIME=1 enables tests that require a real model endpoint.
// These tests are NEVER mandatory for CI.
//
// IMPORTANT (NEVER KILL PORT 7778):
//
// When running tests while a live llama-monitor (or AI coding session)
// is using port 7778, DO NOT let Playwright manage or kill that instance.
//
// Instead, use a separate port with LLAMA_MONITOR_TEST_PORT:
//
//   LLAMA_MONITOR_TEST_PORT=17778 npm test
//   LLAMA_MONITOR_TEST_PORT=17778 npx playwright test --workers=1 --retries=2
//
// Or point at an explicitly started test instance:
//
//   LLAMA_MONITOR_UI_URL=http://127.0.0.1:17778 npm test
//
// Defaulting to 7778 is only safe when no other instance is running there.

/** @type {import('@playwright/test').PlaywrightTestConfig} */
const config = {
  testDir: '.',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  // Local runs must be AT LEAST AS STRICT as CI, never looser and never differently
  // shaped. This used to read `workers: process.env.CI ? 1 : undefined`, which meant CI ran
  // serially while a local run went parallel across files. A local run could then invent
  // concurrency the suite is configured never to have, and manufacture failures in specs
  // that are fine — which cost real time chasing "races" that did not exist.
  //
  // So: one worker everywhere. Retries stay at 0 locally, which is stricter than CI's 1,
  // so a local pass implies a CI pass. Set PLAYWRIGHT_PARALLEL=1 to opt into parallelism
  // deliberately; results from such a run do not predict CI and should not be reported as
  // if they do.
  //
  // CI allows ONE retry, not two: a deterministic failure used to run three times
  // (console-errors burned three attempts to fail identically), while a genuine flake
  // almost always passes on the second attempt. Playwright reports a pass-on-retry as
  // "flaky", so it stays visible.
  retries: process.env.CI ? 1 : 0,
  workers: process.env.PLAYWRIGHT_PARALLEL ? undefined : 1,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.LLAMA_MONITOR_UI_URL ||
              (process.env.LLAMA_MONITOR_TEST_PORT
                 ? `http://127.0.0.1:${process.env.LLAMA_MONITOR_TEST_PORT}`
                 : 'http://127.0.0.1:7778'),
    actionTimeout: 15000,
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
  },
  webServer: process.env.LLAMA_MONITOR_UI_URL
              ? undefined
              : {
                  // Wrapper script creates a fresh temp config dir so tests run with a clean slate,
                  // independent of the developer's local ~/.config/llama-monitor/ data.
                  // LLAMA_MONITOR_TEST_PORT overrides the port (default 7778); run-server.mjs reads it.
                  command: 'node run-server.mjs',
                  cwd: '.',
                  url: `http://127.0.0.1:${process.env.LLAMA_MONITOR_TEST_PORT || '7778'}`,
                  reuseExistingServer: false,
                  stdout: 'pipe',
                  stderr: 'pipe',
                  timeout: 180000,
                },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    }
  ],
};

export default config;
