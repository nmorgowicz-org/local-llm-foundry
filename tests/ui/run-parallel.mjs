#!/usr/bin/env node
// run-parallel.mjs — run the UI suite as P independent Playwright processes.
//
// Why processes and not `--workers=P`: the suite shares ONE server per Playwright run and
// several specs mutate that server's state (auth, presets, sessions, settings), which is why
// the config pins `workers: 1`. Extra workers would share that server and manufacture
// failures. Extra *processes* do not: each one starts its own server (its own port and its
// own fresh config dir via run-server.mjs) and runs a disjoint `--shard`, so every process
// is exactly the serial, isolated run the suite is written for. This is the same isolation
// the CI shard matrix already relies on, applied within one machine.
//
// Usage (from tests/ui):
//   node run-parallel.mjs [--procs P] [--shard I/N] [--output DIR] [playwright args...]
//
//   --procs P    parallel processes (default: UI_PROCS, else min(4, cpus/4), at least 1)
//   --shard I/N  outer shard, as CI passes it. It is subdivided: this launcher runs global
//                shards (I-1)*P+1 .. I*P of N*P, so N=1 means "the whole suite".
//   --output DIR base for per-process test results (default: test-results)
//
// Env: LLAMA_MONITOR_TEST_PORT is the base port (default 17778, never 7778); process k of the
// whole run uses base+k. LLAMA_MONITOR_UI_URL is removed so each process owns its server.
// Exit status is non-zero if any process fails.

import { spawn } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { createInterface } from 'node:readline';
import { join } from 'node:path';

const args = process.argv.slice(2);

function take(flag) {
  const i = args.findIndex(a => a === flag || a.startsWith(`${flag}=`));
  if (i === -1) return undefined;
  const eq = args[i].indexOf('=');
  if (eq !== -1) return args.splice(i, 1)[0].slice(eq + 1);
  const value = args[i + 1];
  args.splice(i, 2);
  return value;
}

const cpuDefault = Math.max(1, Math.min(4, Math.floor(availableParallelism() / 4)));
const procs = Math.max(1, Number(take('--procs') ?? process.env.UI_PROCS ?? cpuDefault));
const [outerIndex, outerTotal] = (take('--shard') ?? '1/1').split('/').map(Number);
const outputBase = take('--output') ?? 'test-results';
const basePort = Number(process.env.LLAMA_MONITOR_TEST_PORT ?? 17778);

if (![procs, outerIndex, outerTotal, basePort].every(Number.isInteger) || outerIndex < 1 || outerIndex > outerTotal) {
  console.error('run-parallel: invalid --procs/--shard/port');
  process.exit(2);
}
if (basePort === 7778) {
  console.error('run-parallel: refusing port 7778 (reserved for a live instance)');
  process.exit(2);
}

const total = outerTotal * procs;
const children = [];
const results = [];

function launch(slot) {
  const global = (outerIndex - 1) * procs + slot + 1; // 1-based global shard
  const port = basePort + (global - 1);
  const env = {
    ...process.env,
    LLAMA_MONITOR_TEST_PORT: String(port),
    PLAYWRIGHT_HTML_OUTPUT_DIR: join(process.env.PLAYWRIGHT_HTML_OUTPUT_DIR ?? 'playwright-report', `proc-${global}`),
  };
  delete env.LLAMA_MONITOR_UI_URL;
  const child = spawn(
    'npx',
    ['playwright', 'test', '--workers=1', `--shard=${global}/${total}`,
      `--output=${join(outputBase, `proc-${global}`)}`, ...args],
    { env, stdio: ['ignore', 'pipe', 'pipe'], shell: process.platform === 'win32' },
  );
  const tag = `[${global}/${total}]`;
  for (const stream of [child.stdout, child.stderr]) {
    createInterface({ input: stream }).on('line', line => {
      // Server chatter is per-process noise; keep test results and failures.
      if (line.startsWith('[WebServer]')) return;
      console.log(`${tag} ${line}`);
    });
  }
  children.push(child);
  return new Promise(resolve => {
    child.on('exit', code => {
      results.push({ global, port, code });
      resolve();
    });
  });
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    for (const c of children) c.kill(signal);
  });
}

const started = Date.now();
console.log(`run-parallel: ${procs} process(es), shards ${(outerIndex - 1) * procs + 1}..${outerIndex * procs} of ${total}, ports ${basePort + (outerIndex - 1) * procs}+`);
await Promise.all(Array.from({ length: procs }, (_, slot) => launch(slot)));

results.sort((a, b) => a.global - b.global);
const failed = results.filter(r => r.code !== 0);
console.log(`\nrun-parallel: finished in ${Math.round((Date.now() - started) / 1000)}s — ` +
  results.map(r => `${r.global}:${r.code === 0 ? 'ok' : `exit ${r.code}`}`).join(' '));
process.exit(failed.length ? 1 : 0);
