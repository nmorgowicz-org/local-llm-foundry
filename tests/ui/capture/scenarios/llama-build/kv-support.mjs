// SCENARIO INTENT: Show the shared K/V support line in the wizard (Guided and Pro) and the preset editor for CUDA native, CUDA missing-kernel fallback, Metal CPU attention, and a known refusal, using a mocked runtime evidence endpoint.
// Synthetic evidence only: the endpoint is mocked, so these stills document presentation, not measured behaviour.
import { loadAppDocument } from '../../harness/browser.mjs';
import { sleep } from '../../harness/paths.mjs';
import { captureShot } from '../../harness/shot.mjs';

const CASES = {
  native: {
    support: { kind: 'native' }, compiled_pair: { state: 'confirmed', value: true, source: 'probe' },
    reason: null, fix: null, blocking: false, buildable_missing: false, backend: 'cuda', advisories: [],
  },
  cudaFallback: {
    support: { kind: 'fallback', impact: 'slower_decode', reason: 'x' },
    compiled_pair: { state: 'confirmed', value: false, source: 'probe' },
    reason: 'No FlashAttention vector kernel is compiled for q8_0/q4_0. llama.cpp converts K and V to f16 instead, which is slower. A source build with every pair compiled removes this.',
    fix: null, blocking: false, buildable_missing: true, backend: 'cuda',
    advisories: [{
      id: 'gemma4-quantized-kv', severity: 'caution',
      text: 'Gemma 4 can be sensitive to quantizing either K or V. Prefer conservative cache precision and check output quality on your own prompts.',
      source: 'Maintainer/user guidance (qualitative); not a measured severity.',
    }],
  },
  metalCpu: {
    support: { kind: 'fallback', impact: 'cpu_attention', reason: 'x' },
    compiled_pair: { state: 'assumed', value: false, reason: 'source' },
    reason: 'Metal flash attention supports equal K/V types only, so q8_0/q4_0 attention runs on the CPU. On llama.cpp b11719 with an M5 Max and Qwen3.8-27B IQ4_XS (head_dim 256), a mixed quantized pair measured about 21x slower prefill and 3x slower decode than the equal pair because attention ran on the CPU. This was measured only for that device, model and version.',
    fix: null, blocking: false, buildable_missing: false, backend: 'metal', advisories: [],
  },
  wontStart: {
    support: { kind: 'wont_start', reason: 'x', fix: 'x' },
    compiled_pair: { state: 'confirmed', value: true, source: 'probe' },
    reason: 'llama.cpp refuses to start with a quantized V cache (q4_0) when flash attention is off.',
    fix: 'Set flash attention to on or auto, or use a floating V cache type (f16, bf16 or f32).',
    blocking: true, buildable_missing: false, backend: 'cuda', advisories: [],
  },
};

async function installMock(page) {
  await page.evaluate((cases) => {
    window.__kvCase = 'native';
    const original = window.fetch.bind(window);
    window.fetch = (url, options) => {
      const href = String(url);
      if (href.includes('/api/hf/download-dir')) {
        return Promise.resolve(new Response(JSON.stringify({ dir: '/tmp/capture-fixtures' }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      if (href.includes('/api/llama-binary/kv-support')) {
        const params = new URL(href, location.origin).searchParams;
        const body = {
          ok: true, k: params.get('k'), v: params.get('v'), effective_flash_attn: params.get('fa') === 'off' ? 'off' : 'on',
          runtime_identity: 'capture', device_identity: 'dev0', model_metadata_identity: 'capture-model', basis_version: 1,
          session_observations: [], ...cases[window.__kvCase],
        };
        return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      return original(url, options);
    };
  }, CASES);
}

async function setKv(page, kvCase, { k, v, fa }) {
  await page.evaluate(({ kvCase, k, v, fa }) => {
    window.__kvCase = kvCase;
    const fire = (id, value) => {
      const el = document.getElementById(id);
      if (!el || value == null) return;
      el.value = value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    };
    fire('spawn-cache-type-k', k);
    fire('spawn-cache-type-v', v);
    fire('spawn-flash-attn', fa);
  }, { kvCase, k, v, fa });
  await sleep(700);
}

async function openWizard(page, mode) {
  await page.evaluate(async (mode) => {
    const { openSpawnWizard, wizardState, selectWizardEngine, showStep } = await import('/js/features/spawn-wizard.js');
    openSpawnWizard({ localPath: '/tmp/capture-fixtures/gemma-4-fixture.gguf' });
    wizardState.model.source = 'local';
    wizardState.model.path = '/tmp/capture-fixtures/gemma-4-fixture.gguf';
    wizardState.viewMode = mode;
    selectWizardEngine('llama_cpp', true);
    const select = document.getElementById('view-mode-select');
    if (select) {
      select.value = mode;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }
    showStep(1);
  }, mode);
  await sleep(500);
}

async function reveal(page, selector, { pro = false } = {}) {
  if (pro) {
    await page.evaluate(() => document.querySelectorAll('#pro-rail-nav .pro-rail-item')[1]?.click());
    await sleep(250);
  }
  await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return;
    // The wizard scrolls inside .wizard-body, not the document.
    el.scrollIntoView({ behavior: 'instant', block: 'center' });
  }, selector);
  await sleep(250);
  if (process.env.KV_DEBUG) {
    console.log('[kv-debug]', selector, JSON.stringify(await page.evaluate((sel) => {
      const el = document.querySelector(sel);
      if (!el) return 'missing';
      const chain = [];
      for (let n = el; n && n !== document.body; n = n.parentElement) {
        const cs = getComputedStyle(n);
        if (/(auto|scroll)/.test(cs.overflowY) || n.hidden || cs.display === 'none') chain.push(`${n.id || n.className}:${cs.overflowY}:${cs.display}:${n.scrollTop}/${n.scrollHeight}/${n.clientHeight}`);
      }
      const r = el.getBoundingClientRect();
      return { top: Math.round(r.top), h: Math.round(r.height), chain };
    }, selector)));
  }
}

export default async function (ctx) {
  const { page, baseUrl } = ctx;
  await page.setViewport({ width: 1280, height: 900, deviceScaleFactor: 1 });
  await loadAppDocument(page, baseUrl);
  await installMock(page);

  await openWizard(page, 'guided');
  await setKv(page, 'native', { k: 'q8_0', v: 'q8_0', fa: 'on' });
  await reveal(page, '#hw-decision-kv');
  // INTENT: Guided KV card on CUDA with an evidenced fused path: quiet line, mixed tile offered with its sub-label.
  await captureShot(page, 'kv-support-guided-native.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '#wizard-step-1' });

  await setKv(page, 'cudaFallback', { k: 'q8_0', v: 'q4_0', fa: 'on' });
  await reveal(page, '#hw-decision-kv');
  // INTENT: Guided mixed pair on CUDA with a confirmed missing kernel: amber slower-path reason, rebuild hint, and model guidance.
  await captureShot(page, 'kv-support-guided-cuda-fallback.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '#wizard-step-1' });

  await openWizard(page, 'pro');
  await setKv(page, 'metalCpu', { k: 'q8_0', v: 'q4_0', fa: 'on' });
  await reveal(page, '#hw-decision-kv', { pro: true });
  // INTENT: Pro KV controls on Metal: contextual expected-CPU-attention warning scoped to its measurement, no rebuild suggestion.
  await captureShot(page, 'kv-support-pro-metal-cpu-attention.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '#wizard-step-1' });

  await setKv(page, 'wontStart', { k: 'q8_0', v: 'q4_0', fa: 'off' });
  await reveal(page, '#hw-decision-kv');
  // INTENT: Pro KV controls with a quantized V cache and flash attention off: red refusal with the fix, Start disabled.
  await captureShot(page, 'kv-support-pro-wont-start.png', { fullPage: true, runtimeTag: 'llamacpp-local', expandSelector: '#wizard-step-1' });
}
