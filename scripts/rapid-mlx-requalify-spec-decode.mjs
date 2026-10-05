#!/usr/bin/env node
// One-command requalification lane for Rapid-MLX speculative decoding.
//
// Rapid-MLX 0.11.1's MTP scheduler was directly observed to be greedy-only and
// to bail whenever a logits processor is installed. Under those two conditions
// no shipping agentic workload can ever reach it, so speculative decoding is
// held at Unavailable in the capability snapshot regardless of --speculative
// being present. See docs/reference/rapid-mlx-mtp-evidence.md.
//
// This script answers exactly one question about a new build: did upstream fix
// that yet? It runs three gates and prints one verdict per gate.
//
//   sampled      speculation is active at temperature > 0
//   constrained  speculation is active with a tool grammar installed
//   parity       greedy output is token-identical with speculation on and off
//
// The gate names and their order are mirrored in SPEC_DECODE_GATES in
// src/inference/rapid_mlx/capabilities.rs; the probe message the user sees
// after a managed upgrade points here. Keep the two in step.
//
// Usage -- no arguments needed:
//
//   node scripts/rapid-mlx-requalify-spec-decode.mjs
//
// Models, parsers and port come from scripts/spec-decode-recipe.json; --out defaults
// to tmp/requalify-<version>-<date>; and the verdict is recorded against the installed
// runtime automatically. Any field can still be overridden with a flag, and
// --recipe FILE selects a different one. See --help.
//
// Exit codes:
//   0   all gates pass; speculative decoding may be promoted for this build
//   20  gates ran cleanly but the scheduler still does not engage (still blocked)
//   1   a gate is uninterpretable: control failed, or a run errored

import { spawn, spawnSync } from 'node:child_process';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { isAbsolute, join, resolve } from 'node:path';

const SUITE = resolve('scripts/rapid-mlx-benchmark-suite.mjs');

function die(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const DEFAULT_RECIPE = resolve('scripts/spec-decode-recipe.json');

/// Where a repo-id model (the positive control) resolves from.
///
/// Defaults to the app's own cache rather than the user-wide `~/.cache/huggingface`,
/// because that is where model management is supposed to live: the resolver points
/// rapid-mlx at `<models_dir>/cache/huggingface/hub` when the app launches it, so a lane
/// reading somewhere else would be measuring a model the app cannot serve. Exported to
/// the suite, and from there to rapid-mlx, since neither reads this recipe.
const DEFAULT_HF_HUB_CACHE = join(
  homedir(), '.config', 'local-llm-foundry', 'models', 'cache', 'huggingface', 'hub',
);

const USAGE = `Requalify Rapid-MLX speculative decoding. Usually takes no arguments:

  node scripts/rapid-mlx-requalify-spec-decode.mjs

Inputs come from ${DEFAULT_RECIPE}; flags below override it.

  --recipe FILE                  use a different recipe
  --model DIR                    trunk to serve
  --speculative-model DIR        subject MTP head
  --speculative-control-model M  known-good positive control
  --tool-call-parser NAME        needed by the 'constrained' gate
  --reasoning-parser NAME
  --profile-alias REPO           HF alias to read parsers from
  --num-speculative-tokens N     requested speculative ceiling (default 3)
  --disable-auto-k               pin the requested ceiling instead of adaptive K
  --mode MODE                    screen, full, or full-diagnostic (adds greedy parity)
  --out DIR                      receipts (default: tmp/requalify-<version>-<date>)
  --port N                       (default 8110)
  --rapid-mlx-bin PATH
  --hf-hub-cache DIR             where repo-id models resolve from
                                 (default: the app's managed cache)
  --gate NAME                    run a subset; repeatable. A partial sweep is
                                 evidence, never qualification
  --no-ingest                    write the report but do not record the verdict
  --help
`;

/// Paths in a recipe are written for humans, so ~ has to work.
function expandHome(value) {
  if (typeof value !== 'string') return value;
  if (value === '~') return homedir();
  return value.startsWith('~/') ? join(homedir(), value.slice(2)) : value;
}

/// Load the recipe. A missing default recipe is not an error -- every field it would
/// have supplied can be passed as a flag -- but a named one that is missing is, because
/// the caller asked for something specific.
function loadRecipe(path, explicit) {
  if (!existsSync(path)) {
    if (explicit) die(`Recipe not found: ${path}`);
    return {};
  }
  try {
    const parsed = JSON.parse(readFileSync(path, 'utf8'));
    // `$comment` and the `*Note` fields are documentation for whoever opens the file
    // next, which is the whole reason it is worth committing. Ignore them here.
    return parsed;
  } catch (error) {
    die(`Recipe ${path} is not readable JSON: ${error.message}`);
  }
}

function parseArgs(argv) {
  const flags = {};
  const rest = argv.slice(2);
  for (let index = 0; index < rest.length; index += 1) {
    const key = rest[index];
    if (key === '--help' || key === '-h') {
      process.stdout.write(USAGE);
      process.exit(0);
    }
    if (!key.startsWith('--')) die(`Invalid argument: ${key}`);
    // Valueless flags first, so they do not swallow the next argument.
    if (key === '--no-ingest') {
      flags.ingest = false;
      continue;
    }
    if (key === '--disable-auto-k') {
      flags.disableAutoK = true;
      continue;
    }
    const value = rest[index + 1];
    if (value === undefined) die(`Missing value for ${key}`);
    index += 1;
    if (key === '--recipe') flags.recipe = value;
    else if (key === '--model') flags.model = value;
    else if (key === '--speculative-control-model') flags.controlModel = value;
    else if (key === '--speculative-model') flags.subjectModel = value;
    else if (key === '--out') flags.out = value;
    else if (key === '--rapid-mlx-bin') flags.rapidMlxBin = value;
    else if (key === '--hf-hub-cache') flags.hfHubCache = value;
    else if (key === '--port') flags.port = Number(value);
    else if (key === '--tool-call-parser') flags.toolCallParser = value;
    else if (key === '--reasoning-parser') flags.reasoningParser = value;
    else if (key === '--profile-alias') flags.profileAlias = value;
    else if (key === '--num-speculative-tokens') flags.numSpeculativeTokens = Number(value);
    else if (key === '--mode') flags.mode = value;
    else if (key === '--gate') (flags.gates ??= []).push(value);
    else die(`Unknown option: ${key}`);
  }

  const recipePath = flags.recipe ? resolve(flags.recipe) : DEFAULT_RECIPE;
  const recipe = loadRecipe(recipePath, Boolean(flags.recipe));
  const usedRecipe = Object.keys(recipe).length > 0;

  // Flags win over the recipe, field by field, so overriding one input does not mean
  // restating the rest.
  const options = {
    recipePath: usedRecipe ? recipePath : null,
    ingest: flags.ingest ?? true,
    port: flags.port ?? recipe.port ?? 8110,
    model: expandHome(flags.model ?? recipe.model),
    subjectModel: expandHome(flags.subjectModel ?? recipe.speculativeModel),
    controlModel: expandHome(flags.controlModel ?? recipe.speculativeControlModel),
    toolCallParser: flags.toolCallParser ?? recipe.toolCallParser,
    reasoningParser: flags.reasoningParser ?? recipe.reasoningParser,
    profileAlias: flags.profileAlias ?? recipe.profileAlias,
    out: flags.out,
    rapidMlxBin: flags.rapidMlxBin ?? recipe.rapidMlxBin,
    hfHubCache: expandHome(flags.hfHubCache ?? recipe.hfHubCache ?? DEFAULT_HF_HUB_CACHE),
    numSpeculativeTokens: flags.numSpeculativeTokens ?? recipe.numSpeculativeTokens ?? 3,
    disableAutoK: flags.disableAutoK ?? recipe.disableAutoK ?? false,
    mode: flags.mode ?? recipe.mode ?? 'full',
    gates: flags.gates,
  };

  if (!['screen', 'full', 'full-diagnostic'].includes(options.mode)) {
    die('--mode must be screen, full, or full-diagnostic.');
  }

  if (!Number.isInteger(options.numSpeculativeTokens)
      || options.numSpeculativeTokens < 1 || options.numSpeculativeTokens > 8) {
    die('--num-speculative-tokens must be an integer from 1 through 8.');
  }

  const missing = (field, flag) => `${field} is not set. Pass ${flag}, or add it to `
    + `${recipePath}.`;
  if (!options.model) die(missing('The trunk to serve', '--model'));
  if (!options.controlModel) {
    die(`${missing('A positive control', '--speculative-control-model')} Requalification `
      + 'without a known-good positive control cannot distinguish "upstream is still '
      + 'blocked" from "this sidecar is broken".');
  }
  // Local paths are checked before anything is served: a run that dies twenty minutes
  // in because a directory moved wastes the whole point of a cheap lane.
  for (const [label, path] of [['Trunk', options.model], ['Subject head', options.subjectModel]]) {
    if (path && (isAbsolute(path) || path.startsWith('.')) && !existsSync(path)) {
      die(`${label} does not exist: ${path}\n`
        + `Fix the path in ${recipePath}, or pass the flag explicitly. `
        + 'Artifact provenance: section 12.3 of docs/reference/rapid-mlx-mtp-evidence.md.');
    }
  }
  // Not fatal: a repo id that is absent can still be fetched. Worth saying, though,
  // because the alternative is a twenty-minute run whose first minutes are a download
  // nobody asked for.
  if (!existsSync(options.hfHubCache)) {
    process.stderr.write(`[warn] HF hub cache does not exist yet: ${options.hfHubCache}\n`);
  }
  const unknown = (options.gates ?? []).filter((name) => !GATES.some((gate) => gate.name === name));
  if (unknown.length) die(`Unknown gate(s): ${unknown.join(', ')}`);
  return options;
}

// Each gate is one suite invocation plus one predicate over its receipts.
// Deliberately the smallest cell selection that can answer the question: this
// lane is meant to be cheap enough to run on every upstream bump.
const GATES = [
  {
    name: 'sampled',
    question: 'Is speculation active at temperature > 0?',
    suite: 'spec-decode',
    cells: ['spec-off-8000-int8', 'spec-control-mtp-8000-int8'],
    // Explicit, not 'recommended', on purpose. The recommended lane resolves
    // this family's vendor profile to temperature 1.0 with reasoning ON, and
    // these cells budget 512 tokens: the model spends the whole budget inside
    // <think>, emits empty content, and the inner benchmark fails its
    // completion floor. That failure says nothing about speculation. This gate
    // owns one axis — temperature > 0 — so it pins temperature and leaves
    // reasoning off, which is also the axis the greedy-only scheduler bails on.
    // Coding temperature is the one that matters in practice.
    sampling: 'explicit',
    temperature: 0.6,
    evaluate: (facts) => {
      if (!facts.temperature) {
        return {
          verdict: 'fail',
          reason: 'The recommended sampling lane resolved to temperature 0, so '
            + 'this run measured the greedy path again and says nothing about '
            + 'sampled decode. Supply the model\'s recommended sampling or run '
            + 'the suite with an explicit temperature.',
        };
      }
      if (facts.attempts <= 0) {
        return {
          verdict: 'blocked',
          reason: `No speculative attempts recorded at temperature ${facts.temperature}: `
            + 'the scheduler still falls through for sampled requests.',
        };
      }
      return {
        verdict: 'pass',
        reason: `${facts.accepts}/${facts.attempts} draft tokens accepted at `
          + `temperature ${facts.temperature}.`,
      };
    },
  },
  {
    name: 'constrained',
    question: 'Is speculation active with a tool grammar installed?',
    // The warm suite is the one that installs tools with tool_choice=required,
    // which is what puts a logits processor in the decode path. Keep it on the
    // same non-greedy temperature as the production coding route.
    suite: 'spec-decode-warm',
    cells: ['spec-warm-off-8000-int8', 'spec-warm-control-mtp-8000-int8'],
    sampling: 'explicit',
    temperature: 0.6,
    evaluate: (facts) => {
      if (facts.attempts <= 0) {
        return {
          verdict: 'blocked',
          reason: 'No speculative attempts recorded with a tool grammar installed: '
            + 'the scheduler still bails when a logits processor is present.',
        };
      }
      return {
        verdict: 'pass',
        reason: `${facts.accepts}/${facts.attempts} draft tokens accepted with tools on.`,
      };
    },
  },
  {
    name: 'parity',
    question: 'Is greedy output token-identical with speculation on and off?',
    suite: 'spec-decode',
    cells: ['spec-off-8000-int8', 'spec-control-mtp-8000-int8'],
    sampling: 'greedy',
    // Two trials so the baseline's own reproducibility is measured; without it
    // a mismatch cannot be attributed to speculation.
    trials: 2,
    evaluate: (facts) => {
      const parity = facts.suiteIndex?.greedy_parity;
      if (!parity) {
        return { verdict: 'fail', reason: 'Run produced no greedy_parity block.' };
      }
      if (parity.baseline_deterministic === false) {
        return {
          verdict: 'fail',
          reason: 'The off-lane baseline is not reproducible across trials, so an '
            + 'off-vs-MTP difference proves nothing about losslessness.',
        };
      }
      if (facts.attempts <= 0) {
        return {
          verdict: 'blocked',
          reason: 'No speculative attempts recorded, so there is no speculative '
            + 'output to compare against the baseline.',
        };
      }
      if (parity.mismatches > 0) {
        return {
          verdict: 'fail',
          reason: `${parity.mismatches}/${parity.comparisons_total} comparisons diverged; `
            + 'greedy speculation is not lossless on this build.',
        };
      }
      return {
        verdict: 'pass',
        reason: `${parity.comparisons_total}/${parity.comparisons_total} comparisons `
          + 'token-identical.',
      };
    },
  },
];

// `rapid-mlx info` resolves tool/reasoning parsers only for HF repo aliases. A
// plain local --model directory reports the literal "(none)" for both, so a
// local trunk silently runs with no parser installed. That is fatal to the
// `constrained` gate specifically: with no tool-call parser there is no tool
// grammar in the decode path, so the gate would report "speculation works with
// tools" from a run that never installed tools. Resolve, then refuse.
const INFO_ABSENT_VALUES = new Set(['(none)', 'none', 'n/a', '-', 'unknown', 'unset', '']);

function infoParsers(model, bin) {
  const result = spawnSync(bin, ['info', model], { encoding: 'utf8', timeout: 30000 });
  const parsers = {};
  if (result.status !== 0 || !result.stdout) return parsers;
  for (const line of result.stdout.split('\n')) {
    const clean = line.replace(/^\s*│/, '').replace(/│\s*$/, '').trim();
    const colon = clean.indexOf(':');
    if (colon === -1) continue;
    const key = clean.slice(0, colon).trim().toLowerCase().replace(/[ _-]/g, '');
    const value = clean.slice(colon + 1).trim();
    if (INFO_ABSENT_VALUES.has(value.toLowerCase())) continue;
    if (key === 'toolformat' || key === 'tool') parsers.toolCallParser = value;
    else if (key === 'reasoningparser' || key === 'reasoning') parsers.reasoningParser = value;
  }
  return parsers;
}

// Ordered sources, most explicit first, with the source recorded in the report
// so a verdict can never be read without knowing where its parsers came from.
function resolveParsers(options, bin) {
  if (options.toolCallParser && options.reasoningParser) {
    return { ...options, parserSource: 'explicit flags' };
  }
  const fromModel = infoParsers(options.model, bin);
  const fromAlias = options.profileAlias ? infoParsers(options.profileAlias, bin) : {};
  const toolCallParser = options.toolCallParser ?? fromModel.toolCallParser ?? fromAlias.toolCallParser;
  const reasoningParser = options.reasoningParser ?? fromModel.reasoningParser ?? fromAlias.reasoningParser;
  const parserSource = [
    options.toolCallParser || options.reasoningParser ? 'explicit flags' : null,
    fromModel.toolCallParser || fromModel.reasoningParser ? `rapid-mlx info ${options.model}` : null,
    fromAlias.toolCallParser || fromAlias.reasoningParser ? `rapid-mlx info ${options.profileAlias}` : null,
  ].filter(Boolean).join(' + ') || 'unresolved';
  return { ...options, toolCallParser, reasoningParser, parserSource };
}

function installedVersion(bin) {
  const result = spawnSync(bin, ['--version'], { encoding: 'utf8', timeout: 30000 });
  const version = `${result.stdout ?? ''}\n${result.stderr ?? ''}`.trim();
  if (result.status !== 0 || !version) {
    die('Could not read `rapid-mlx --version`. A requalification verdict that is '
      + 'not pinned to a build is worthless.');
  }
  return version;
}

// Ordered by how likely each is to be the build the user actually means: an explicit
// override, then this checkout's newest binary, then whatever is on PATH.
//
// Newest rather than release-over-debug, learned the hard way: a stale target/release
// from before a flag existed gets picked over a debug build that has it, and rejects the
// report with an argument error that looks like the report's fault.
function resolveMonitorBin() {
  if (process.env.LLAMA_MONITOR_BIN) return process.env.LLAMA_MONITOR_BIN;
  const built = ['release', 'debug']
    .map((profile) => resolve('target', profile, 'local-llm-foundry'))
    .filter((candidate) => existsSync(candidate))
    .map((candidate) => ({ candidate, mtime: statSync(candidate).mtimeMs }))
    .sort((left, right) => right.mtime - left.mtime);
  if (built.length) return built[0].candidate;
  for (const name of ['local-llm-foundry']) {
    const which = spawnSync('command', ['-v', name], { encoding: 'utf8', shell: true });
    const found = (which.stdout ?? '').trim();
    if (found) return found;
  }
  return null;
}

/// Record the verdict against the installed runtime. Returns whether it landed.
///
/// Failure here does not invalidate the run -- the report on disk is the measurement, and
/// ingesting is bookkeeping -- so this reports and moves on rather than exiting. The exit
/// code still reflects the gates.
function ingestReport(path) {
  const bin = resolveMonitorBin();
  if (!bin) {
    process.stderr.write('\nNo local-llm-foundry binary found to record the verdict with.\n');
    return false;
  }
  process.stderr.write(`\nRecording the verdict via ${bin}\n`);
  const result = spawnSync(bin, ['--ingest-spec-decode-report', path], {
    stdio: ['ignore', 'inherit', 'inherit'],
    timeout: 120000,
  });
  if (result.error) {
    process.stderr.write(`Could not run ${bin}: ${result.error.message}\n`);
    return false;
  }
  if (result.status !== 0) {
    process.stderr.write(`${bin} refused the report (exit ${result.status}).\n`);
    return false;
  }
  return true;
}

function runSuite(args, hfHubCache) {
  return new Promise((resolveRun) => {
    const child = spawn(process.execPath, [SUITE, 'run', ...args], {
      stdio: ['ignore', 'inherit', 'inherit'],
      // An explicit HF_HUB_CACHE already in the environment wins: whoever set it knows
      // something about this run that the recipe does not.
      env: { HF_HUB_CACHE: hfHubCache, ...process.env },
    });
    child.on('error', (error) => resolveRun({ status: null, error: error.message }));
    child.on('close', (status) => resolveRun({ status, error: null }));
  });
}

// Read back what the suite recorded rather than what we asked for: a lane that
// silently fell back to greedy, or a control cell that never installed MTP,
// must not be reported as the thing we requested.
async function gateFacts(outputDir) {
  const suiteIndex = JSON.parse(await readFile(join(outputDir, 'suite-index.json'), 'utf8'));
  let controlAccepts = 0;
  let controlAttempts = 0;
  let subjectAccepts = 0;
  let subjectAttempts = 0;
  for (const entry of suiteIndex.receipts ?? []) {
    let receipt;
    try {
      receipt = JSON.parse(await readFile(join(outputDir, entry.receipt), 'utf8'));
    } catch {
      continue;
    }
    for (const cell of receipt.cells ?? []) {
      const role = cell.configuration?.speculative_role;
      if (!['control', 'subject'].includes(role)) continue;
      for (const attempt of cell.attempts ?? []) {
        const accepts = sumByPrefix(attempt.metrics_delta, 'rapid_mlx_spec_decode_accepts_total');
        const attempts = sumByPrefix(attempt.metrics_delta, 'rapid_mlx_spec_decode_attempts_total');
        if (role === 'control') {
          controlAccepts += accepts;
          controlAttempts += attempts;
        } else {
          subjectAccepts += accepts;
          subjectAttempts += attempts;
        }
      }
    }
  }
  // The control separates a broken harness from a bad subject. Once it clears,
  // the subject is the evidence that this exact managed sidecar works.
  const primary = subjectAttempts > 0 || subjectAccepts > 0
    ? { accepts: subjectAccepts, attempts: subjectAttempts }
    : { accepts: 0, attempts: 0 };
  return {
    suiteIndex,
    accepts: primary.accepts,
    attempts: primary.attempts,
    controlAccepts,
    controlAttempts,
    subjectAccepts,
    subjectAttempts,
    temperature: suiteIndex.sampling_lane?.temperature ?? 0,
  };
}

function sumByPrefix(metrics, prefix) {
  if (!metrics) return 0;
  let total = 0;
  for (const [key, value] of Object.entries(metrics)) {
    if (key.startsWith(prefix) && Number.isFinite(value)) total += value;
  }
  return total;
}

// A malformed head is a negative qualification result, not a broken harness. The
// runtime may fail before it can emit metrics, so perform the cheap safetensors
// structural check first and record that finding as StillBlocked. This also keeps a
// corrupt managed sidecar from being rendered as an unqualified mystery failure.
function subjectSidecarFailure(subjectModel) {
  if (!subjectModel || !isAbsolute(subjectModel)) return null;
  let artifact = subjectModel;
  try {
    if (statSync(artifact).isDirectory()) {
      for (const name of ['mtp.safetensors', 'model-mtp.safetensors']) {
        const candidate = join(artifact, name);
        if (existsSync(candidate)) { artifact = candidate; break; }
      }
    }
    const bytes = readFileSync(artifact);
    if (bytes.length < 16) return `Subject sidecar is too small to be a safetensors file: ${artifact}`;
    const headerLength = Number(bytes.readBigUInt64LE(0));
    if (!Number.isSafeInteger(headerLength) || headerLength <= 2 || headerLength > bytes.length - 8) {
      return `Subject sidecar has an invalid safetensors header length: ${artifact}`;
    }
    const header = JSON.parse(bytes.subarray(8, 8 + headerLength).toString('utf8'));
    const keys = Object.keys(header)
      .filter((key) => key !== '__metadata__')
      .map((key) => key.startsWith('mtp.') ? key.slice(4) : key);
    if (!keys.length || !keys.some((key) => key.startsWith('pre_fc_norm'))) {
      return `Subject sidecar has no MTP pre_fc_norm tensors: ${artifact}`;
    }
  } catch (error) {
    return `Subject sidecar is malformed or unreadable: ${artifact} (${error.message})`;
  }
  return null;
}

const parsedOptions = parseArgs(process.argv);
const rapidMlxBin = parsedOptions.rapidMlxBin ?? 'rapid-mlx';
const options = resolveParsers(parsedOptions, rapidMlxBin);
const version = installedVersion(rapidMlxBin);
// Derived rather than required: a receipt directory named after the build it measured is
// the one thing a future reader needs from it, and picking that name is not a decision
// worth asking for.
if (!options.out) {
  const slug = `${version} ${new Date().toISOString().slice(0, 10)}`
    .replace(/[^A-Za-z0-9.]+/g, '-')
    .replace(/^-|-$/g, '');
  options.out = join('tmp', `requalify-${slug}`);
}
const outRoot = resolve(options.out);
await mkdir(outRoot, { recursive: true });

const sidecarFailure = subjectSidecarFailure(options.subjectModel);

const selected = options.gates?.length
  ? GATES.filter((gate) => options.gates.includes(gate.name))
  : options.mode === 'screen'
    ? GATES.filter((gate) => gate.name === 'sampled')
    : options.mode === 'full'
      ? GATES.filter((gate) => ['sampled', 'constrained'].includes(gate.name))
      : GATES;

const completionTokens = options.mode === 'screen' ? 1024 : 8192;

if (selected.some((gate) => gate.name === 'constrained') && !options.toolCallParser) {
  die('The `constrained` gate needs a tool-call parser, and none could be resolved '
    + `for ${options.model}. \`rapid-mlx info\` reports "(none)" for local model `
    + 'directories, so a run without one installs no tool grammar and the gate would '
    + 'pass on a request that never constrained anything. Pass --tool-call-parser '
    + 'explicitly, or --profile-alias with the HF repo alias for this family '
    + '(e.g. --profile-alias unsloth/Qwen3.6-27B-MLX-8bit), or skip it with '
    + '--gate sampled --gate parity.');
}

process.stderr.write(`Requalifying speculative decoding on ${version}\n`);
process.stderr.write(`Inputs: ${options.recipePath ?? 'command-line flags only'}\n`);
process.stderr.write(`Trunk: ${options.model}\n`);
process.stderr.write(`Subject head: ${options.subjectModel ?? 'none (trunk-embedded)'}\n`);
process.stderr.write(`Control: ${options.controlModel}\n`);
process.stderr.write(`HF hub cache: ${options.hfHubCache}\n`);
process.stderr.write(`Receipts: ${outRoot}\n`);
process.stderr.write(`Mode: ${options.mode} (${options.mode === 'screen' ? 'short live probe' : options.mode === 'full' ? 'production gates' : 'production gates plus greedy safety diagnostic'})\n`);
process.stderr.write(`Parsers: tool=${options.toolCallParser ?? 'none'} `
  + `reasoning=${options.reasoningParser ?? 'none'} (source: ${options.parserSource})\n`);

const results = [];
for (const gate of selected) {
  const gateDir = join(outRoot, gate.name);
  process.stderr.write(`\n=== gate ${gate.name}: ${gate.question}\n`);
  const args = [
    '--model', options.model,
    '--suite', gate.suite,
    '--out', gateDir,
    '--port', String(options.port),
    '--rapid-mlx-bin', rapidMlxBin,
    // Never 'forced': a forced run overrides Rapid's own eligibility verdict,
    // so it can only ever be research evidence and can never requalify.
    '--spec-decode-lane', 'natural',
    // Every gate below evaluates attempts<=0 itself and calls it 'blocked'. The
    // suite's default is to fail the run on zero speculative activity, which
    // never lets those branches run: a still-blocked build would exit 1
    // (uninterpretable / harness broken) instead of 20 (upstream still limited).
    // Those two must not be confusable -- that distinction is this lane's point.
    '--spec-zero-activity', 'observed',
    '--speculative-control-model', options.controlModel,
    '--sampling', gate.sampling,
    ...(gate.temperature !== undefined
      ? ['--temperature', String(options.sampledTemperature ?? gate.temperature)]
      : []),
    ...(options.subjectModel ? ['--speculative-model', options.subjectModel] : []),
    '--speculative-tokens', String(options.numSpeculativeTokens),
    ...(options.disableAutoK ? ['--disable-speculative-auto-k'] : []),
    ...(gate.trials ? ['--trials', String(gate.trials)] : []),
    ...(options.toolCallParser ? ['--tool-call-parser', options.toolCallParser] : []),
    ...(options.reasoningParser ? ['--reasoning-parser', options.reasoningParser] : []),
    '--spec-completion-tokens', String(completionTokens),
    ...(() => {
      const cells = [...gate.cells];
      if (options.subjectModel && options.subjectModel !== options.controlModel) {
        for (const cell of gate.cells) {
          if (cell.includes('-control-')) cells.push(cell.replace('-control-', '-subject-'));
        }
      }
      return cells.flatMap((cell) => ['--cell', cell]);
    })(),
  ];

  let result;
  if (sidecarFailure) {
    result = { verdict: 'blocked', reason: sidecarFailure };
  } else {
    const run = await runSuite(args, options.hfHubCache);
    if (run.error) {
    result = { verdict: 'error', reason: `Suite failed to start: ${run.error}` };
    } else {
    let facts = null;
    try {
      facts = await gateFacts(gateDir);
    } catch (error) {
      result = { verdict: 'error', reason: `Could not read run output: ${error.message}` };
    }
    if (facts) {
      // The positive control is the precondition for every other reading. A
      // failing control means the harness or the sidecar is broken, which is a
      // different finding from upstream still being blocked.
      const control = facts.suiteIndex.positive_control;
      if (control && control.ok === false) {
        result = { verdict: 'fail', reason: `Positive control failed: ${control.reason}` };
      } else if (facts.suiteIndex.complete === false) {
        result = {
          verdict: 'error',
          reason: `Run did not complete: ${facts.suiteIndex.failure ?? 'unknown failure'}`,
        };
      } else {
        result = gate.evaluate(facts);
      }
      // A 'pass' means "speculation engaged and produced accepted tokens", which
      // is exactly what a control that recorded no activity cannot support. If a
      // gate ever returns pass on such a run, the gate's predicate and the
      // control disagree; refuse to promote capability off a contradiction.
      if (result.verdict === 'pass' && control?.code === 'no-activity') {
        result = {
          verdict: 'error',
          reason: `Gate passed but the positive control recorded zero speculative activity: ${control.reason} `
            + 'The gate predicate and the control disagree, so neither reading can be trusted.',
        };
      }
      result.control_code = control?.code ?? 'absent';
      result.accepts = facts.accepts;
      result.attempts = facts.attempts;
      result.temperature = facts.temperature;
    }
    }
  }
  result.gate = gate.name;
  result.question = gate.question;
  result.receipts = gateDir;
  results.push(result);
  process.stderr.write(`--- ${gate.name}: ${result.verdict.toUpperCase()} — ${result.reason}\n`);
}

const productionResults = results.filter((result) => ['sampled', 'constrained'].includes(result.gate));
const productionVerdicts = new Set(productionResults.map((result) => result.verdict));
const overall = productionVerdicts.has('error') || productionVerdicts.has('fail')
  ? 'uninterpretable'
  : productionVerdicts.has('blocked')
    ? 'still-blocked'
    : options.mode === 'screen' ? 'screened' : 'qualified';

const report = {
  rapid_mlx_version: version,
  model: options.model,
  speculative_control_model: options.controlModel,
  speculative_model: options.subjectModel ?? null,
  num_speculative_tokens: options.numSpeculativeTokens,
  disable_auto_k: options.disableAutoK,
  qualification_mode: options.mode,
  screen_completion_tokens: options.mode === 'screen' ? completionTokens : null,
  // A gate verdict is only readable alongside the parsers that were installed:
  // "tools did not slow speculation down" means nothing if no tool parser ran.
  tool_call_parser: options.toolCallParser ?? null,
  reasoning_parser: options.reasoningParser ?? null,
  parser_source: options.parserSource,
  // Present so a reader knows a partial run is partial.
  gates_run: selected.map((gate) => gate.name),
  gates_defined: GATES.map((gate) => gate.name),
  overall,
  // Only a full sweep with every gate passing can promote the capability;
  // a partial run is evidence, not qualification.
  promotes_capability: overall === 'qualified' && ['full', 'full-diagnostic'].includes(options.mode)
    && productionResults.length === 2,
  results,
  generated_at: new Date().toISOString(),
};
const reportPath = join(outRoot, 'requalification.json');
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);

process.stderr.write(`\nOverall: ${overall}\n`);
// Either verdict is worth recording: the app resolves spec_decode from a measurement
// against this exact install before it consults any shipped prior, so a still-blocked
// result is a fact about this box, not a no-op. Recording it is therefore the default
// rather than a step to remember -- a report nobody ingested changes nothing.
if (options.ingest) {
  const recorded = ingestReport(reportPath);
  if (!recorded) {
    process.stderr.write(
      '\nThe verdict was NOT recorded. Build the app and record it with:\n'
      + `  cargo run -- --ingest-spec-decode-report ${reportPath}\n`,
    );
  }
} else {
  process.stderr.write(
    '\n--no-ingest: verdict not recorded. To record it later:\n'
    + `  local-llm-foundry --ingest-spec-decode-report ${reportPath}\n`,
  );
}
if (report.promotes_capability) {
  process.stderr.write(
    'All gates pass. Ingesting turns multi-token prediction on for this install. '
    + 'Because scheduler behaviour is a property of the build rather than the machine, '
    + 'also add this version to SPEC_DECODE_VERSION_PRIORS in '
    + 'src/inference/rapid_mlx/capabilities.rs as SchedulerEvidence::Engages and record '
    + 'the evidence in docs/reference/rapid-mlx-mtp-evidence.md, so users who never run '
    + 'this lane benefit too.\n',
  );
} else if (overall === 'screened') {
  process.stderr.write(
    'The fast screen passed. This is live evidence for this sidecar, but it does not '
    + 'test constrained decoding or greedy parity; run full mode before treating it as qualified.\n',
  );
} else if (overall === 'still-blocked') {
  process.stderr.write(
    'Upstream has not fixed the greedy-only / logits-processor limitation. Speculative '
    + 'decoding stays unavailable, and ingesting records why against this build.\n',
  );
}
process.stderr.write(`Report: ${reportPath}\n`);

process.exit(overall === 'qualified' ? 0 : overall === 'still-blocked' ? 20 : 1);
