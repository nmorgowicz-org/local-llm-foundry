// Demoted from core/app-shell.spec.js: the live output estimate is plain arithmetic over
// tracker state in app-state.js, no DOM needed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { updateLiveOutputEstimate } from '../../../static/js/features/dashboard-render.js';
import * as appState from '../../../static/js/core/app-state.js';

test('smooths live output estimate across recent polling samples', () => {
  Object.assign(appState.liveOutputTracker, { taskId: null, previousDecoded: null, previousMs: null, latestRate: 0, rates: [] });
  appState.metricSeries.liveOutput = [];
  const now = Date.now();
  updateLiveOutputEstimate(123, 0, true, now - 3000);
  updateLiveOutputEstimate(123, 100, true, now - 2000);
  const rate = updateLiveOutputEstimate(123, 170, true, now - 1000);

  assert.ok(rate > 80);
  assert.ok(rate < 90);
});
