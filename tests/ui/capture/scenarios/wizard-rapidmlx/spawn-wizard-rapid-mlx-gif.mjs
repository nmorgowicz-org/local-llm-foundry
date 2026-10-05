// Scenario: spawn-wizard-rapid-mlx-gif
// SCENARIO INTENT: Animate the current Rapid-MLX wizard flow for diagnostic documentation.
// Extracted from tests/ui/capture.mjs (Phase A3).
import fs from 'fs';
import { join } from 'path';
import { execFileSync } from 'child_process';
import { gotoApp } from '../../harness/browser.mjs';
import { currentArtifactsDir, tagFilename, FRAME_DIR, sleep } from '../../harness/paths.mjs';
import { cleanupFrames } from '../../harness/shot.mjs';
import { suppressToasts } from '../../harness/shot.mjs';
import { recordArtifact } from '../../harness/receipt.mjs';

export default async function(ctx, _options) {
    const { page, baseUrl } = ctx;
    const fps = 10;
    let frameIdx = 0;

    await gotoApp(page, baseUrl);
    fs.mkdirSync(FRAME_DIR, { recursive: true });
    console.log('[CAPTURE] Starting spawn-wizard-rapid-mlx-gif sequential capture...');

    // Capture N milliseconds of the current page state.
    const capture = async (durationMs) => {
        const frameMs = 1000 / fps;
        const n = Math.max(1, Math.round(durationMs / frameMs));
        for (let i = 0; i < n; i++) {
            const path = join(FRAME_DIR, `spawn-wizard-rapid-mlx-gif_${String(frameIdx).padStart(3, '0')}.png`);
            await page.screenshot({ path });
            frameIdx++;
            if (i < n - 1) await sleep(frameMs);
        }
    };

    // Mock Rapid-MLX runtime endpoints so the engine is shown as available.
    await page.evaluate(() => {
        const originalFetch = window.fetch.bind(window);
        window.fetch = (input, init) => {
            const url = new URL(typeof input === 'string' ? input : input.url, window.location.origin);
            if (url.pathname === '/api/rapid-mlx/runtime/status') {
                return Promise.resolve(new Response(JSON.stringify({
                    runtime: { supported: true, active: { version: '0.10.10' } },
                }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
            }
            if (url.pathname === '/api/rapid-mlx/catalog') {
                return Promise.resolve(new Response(JSON.stringify({
                    ok: true,
                    models: [
                        { name: 'qwen3.8-27b-4bit', display_name: 'qwen3.8-27b-4bit', size_bytes: 16320875724, parser: 'qwen3_coder_xml', template: 'qwen3', hybrid: false, mtp: true, mtp_sidecar: 'rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX@3' },
                        { name: 'qwen3.6-35b-4bit', display_name: 'qwen3.6-35b-4bit', size_bytes: 21474836480, parser: 'qwen3_coder_xml', template: 'qwen3', hybrid: false, mtp: true, mtp_sidecar: 'rapid-mlx/Qwen3.6-35B-MTP-MLX@3' },
                        { name: 'gemma4-27b-4bit', display_name: 'gemma4-27b-4bit', size_bytes: 16106127360, parser: null, template: 'gemma4', hybrid: true, mtp: false },
                        { name: 'llama4-8b-4bit', display_name: 'llama4-8b-4bit', size_bytes: 4294967296, parser: 'llama3_json', template: 'llama4', hybrid: false, mtp: false },
                    ],
                    recommendations: [
                        { rank: 1, label: 'Smart', name: 'qwen3.8-27b-4bit', cached: true, specs: '20.0 GB RAM · 92% capability · ~41 tok/s' },
                        { rank: 2, label: 'Fast', name: 'qwen3.6-35b-4bit', cached: false, specs: '20.0 GB RAM · 87% capability · ~60 tok/s' },
                    ],
                }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
            }
            if (url.pathname === '/api/rapid-mlx/recommend') {
                return Promise.resolve(new Response(JSON.stringify({
                    recommended_backend: 'rapid_mlx',
                    state: 'ready',
                    reason: 'Rapid-MLX is available for MLX models on this system.',
                }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
            }
            if (url.pathname.startsWith('/api/rapid-mlx/models/') && url.pathname.endsWith('/profile')) {
                return Promise.resolve(new Response(JSON.stringify({
                    profile: {
                        extras: { has_reasoning: true },
                        aliases: { tool_call_parser: 'qwen3_coder', reasoning_parser: 'qwen3' },
                    },
                }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
            }
            return originalFetch(input, init);
        };
    });

    // ── Welcome screen ────────────────────────────────────────────────────────
    await capture(1500);

    // ── Open wizard ───────────────────────────────────────────────────────────
    await page.evaluate(async () => {
        const { openSpawnWizard } = await import('/js/features/spawn-wizard.js');
        openSpawnWizard();
    });
    await page.waitForSelector('#spawn-wizard-overlay.open', { timeout: 8000 });
    await suppressToasts(page);
    await page.evaluate(() => {
        const banner = document.getElementById('wizard-binary-prereq');
        if (banner) banner.style.display = 'none';
    });
    await sleep(400);
    await capture(800);

    // ── Step 0: Profile ───────────────────────────────────────────────────────
    await capture(600);

    await page.evaluate(() => {
        (document.querySelector('.profile-card[data-profile="power"]')
            || document.querySelector('.profile-card'))?.click();
    });
    await sleep(200);
    await capture(700);

    await page.evaluate(() => {
        (document.querySelector('.usecase-card[data-usecase="general"]')
            || document.querySelector('.usecase-card'))?.click();
    });
    await sleep(200);
    await capture(1000);

    // Profile and engine/model selection are on the same step (Option A collapse:
    // 6 steps → 3). No navigation click needed between them.
    await capture(600);

    // ── Step 0: Select Rapid-MLX engine ───────────────────────────────────────
    await page.evaluate(() => {
        document.querySelector('.wizard-engine-card[data-engine="rapid_mlx"]')?.click();
    });
    await page.waitForFunction(
        () => document.querySelector('.wizard-engine-card[data-engine="rapid_mlx"].selected') !== null,
        { timeout: 5000 }
    );
    await sleep(400);
    await capture(1200); // Dwell on Rapid-MLX engine card selected.

    // The curated catalog lives in the HF model area; pick that source.
    await page.evaluate(() => {
        document.querySelector('.model-source-card[data-source="hf"]')?.click();
    });
    await sleep(300);

    // ── Curated model picker replaces HF browse for Rapid-MLX ─────────────────
    await page.waitForFunction(
        () => document.querySelectorAll('#rapid-catalog-panel .rapid-catalog-row').length >= 3,
        { timeout: 8000 }
    ).catch(() => console.log('[CAPTURE] catalog rows wait timed out; continuing.'));
    await page.evaluate(() => {
        document.getElementById('rapid-catalog-panel')?.scrollIntoView({ behavior: 'instant', block: 'start' });
    });
    await sleep(400);
    await capture(1600); // Dwell on pinned recommendations + curated list.

    // One click on the pinned Smart pick selects alias, measured size, and the
    // paired MTP sidecar.
    await page.evaluate(() => {
        document.querySelector('#rapid-catalog-panel .rapid-catalog-row--recommended')?.click();
    });
    await sleep(600);
    await capture(1400);

    // ── Step 0 → Step 1: Rapid-MLX Hardware ──────────────────────────────────
    await page.evaluate(() => document.getElementById('wizard-next-btn')?.click());
    await page.waitForFunction(
        () => document.getElementById('wizard-step-1')?.classList.contains('active'),
        { timeout: 5000 }
    ).catch(() => console.log('[CAPTURE] Step 1 wait timed out; continuing.'));
    await sleep(400);

    // Ensure Rapid-MLX hardware panel and advanced fields are visible, and wait
    // for the browser to reflow after unhiding.
    await page.evaluate(async () => {
        const panel = document.getElementById('rapid-hardware-panel');
        const fields = document.getElementById('spawn-rapid-advanced-fields');
        if (panel && panel.hidden) panel.hidden = false;
        if (fields) fields.style.display = 'block';
        // Wait for layout reflow so the panel is fully rendered before capture.
        await new Promise(r => setTimeout(r, 0));
        await new Promise(r => requestAnimationFrame(r));
        await new Promise(r => requestAnimationFrame(r));
    });
    await sleep(200);
    await capture(600);

    // ── Step 2: Rapid-MLX hardware panel — initial state ─────────────────────
    // Quantized KV is unlaunchable for Rapid-MLX (hybrid ArraysCache models
    // refuse it at startup), so the dtype dial is hidden; dwell on the panel.
    await page.evaluate(() => {
        const body = document.querySelector('.wizard-body');
        if (body) body.scrollTop = 300;
    });
    await sleep(300);
    await capture(2000);

    // ── Step 2: Rapid-MLX hardware panel — retained cache selector ───────────
    await page.evaluate(() => {
        const el = document.getElementById('spawn-retained-cache-mib');
        if (el) el.scrollIntoView({ behavior: 'instant', block: 'center' });
    });
    await sleep(300);
    await capture(1000);

    // ── Step 2: Show parsers and hybrid mode ─────────────────────────────────
    await page.evaluate(() => {
        const el = document.getElementById('spawn-rapid-tool-call-parser');
        if (el) el.scrollIntoView({ behavior: 'instant', block: 'center' });
    });
    await sleep(300);
    await capture(1000);

    await page.evaluate(() => {
        const el = document.getElementById('spawn-rapid-hybrid-mode');
        if (el) el.scrollIntoView({ behavior: 'instant', block: 'center' });
    });
    await sleep(300);
    await capture(1000);

    // ── Step 2: Reasoning mode toggle ────────────────────────────────────────
    await page.evaluate(() => {
        const el = document.getElementById('spawn-rapid-reasoning-mode');
        if (el) el.scrollIntoView({ behavior: 'instant', block: 'center' });
    });
    await sleep(300);
    await capture(800);

    // Enable reasoning mode to show the KV dtype lock (int4 blocked).
    await page.evaluate(() => {
        const checkbox = document.getElementById('spawn-rapid-reasoning-mode');
        if (checkbox && !checkbox.checked) checkbox.click();
    });
    await sleep(400);
    await capture(1500); // Show reasoning mode ON with KV dtype pinned.

    // Disable reasoning mode to show full KV dtype options restored.
    await page.evaluate(() => {
        const checkbox = document.getElementById('spawn-rapid-reasoning-mode');
        if (checkbox && checkbox.checked) checkbox.click();
    });
    await sleep(400);
    await capture(1000);

    // ── Step 2: Profile hints (auto-detected parsers) ────────────────────────
    await page.evaluate(() => {
        const hintsEl = document.getElementById('rapid-mlx-profile-hints');
        if (hintsEl) {
            hintsEl.style.display = 'block';
            const rect = hintsEl.getBoundingClientRect();
            const body = document.querySelector('.wizard-body');
            if (body && rect.top > 300) {
                hintsEl.scrollIntoView({ behavior: 'instant', block: 'center' });
            }
        }
    });
    await sleep(300);
    const hintsVisible = await page.evaluate(() => {
        const el = document.getElementById('rapid-mlx-profile-hints');
        return el && el.style.display !== 'none' && el.textContent.length > 10;
    });
    if (hintsVisible) {
        await capture(1500); // Show profile hints if they appeared.
    }

    // Scroll back to top of hardware panel.
    await page.evaluate(() => {
        const panel = document.getElementById('rapid-hardware-panel');
        if (panel) panel.scrollIntoView({ behavior: 'instant', block: 'start' });
        else {
            const body = document.querySelector('.wizard-body');
            if (body) body.scrollTop = 0;
        }
    });
    await sleep(200);

    // ── Same step (Hardware & memory): summary ────────────────────────────────
    // Enable reasoning mode so the review summary reflects the reasoning profile.
    // Option A collapse merged the former Summary step into this one's DOM,
    // further down the page — no navigation needed to reach it.
    await page.evaluate(() => {
        const cb = document.getElementById('spawn-rapid-reasoning-mode');
        if (cb && !cb.checked) cb.click();
    });
    await sleep(400);
    await capture(1000);

    // Scroll to the config summary list.
    await page.evaluate(() => {
        const list = document.getElementById('spawn-summary-list');
        if (list) list.scrollIntoView({ behavior: 'instant', block: 'start' });
        else {
            const main = document.querySelector('#wizard-step-1 .wizard-main');
            if (main) main.scrollTop = main.scrollHeight;
        }
    });
    await sleep(300);
    await capture(2500); // Hold on the summary list (Rapid-MLX config).

    // ── Step 1 → Step 2: Launch (preset settings + spawn) ────────────────────
    await page.evaluate(() => document.getElementById('wizard-next-btn')?.click());
    await page.waitForFunction(
        () => document.getElementById('wizard-step-2')?.classList.contains('active'),
        { timeout: 5000 }
    ).catch(() => console.log('[CAPTURE] Step 2 wait timed out; continuing.'));
    await sleep(400);
    await capture(1500);

    // Scroll to the Save Preset row.
    await page.evaluate(() => {
        const row = document.getElementById('spawn-save-preset-row');
        if (row) row.scrollIntoView({ behavior: 'instant', block: 'center' });
        else {
            const main = document.querySelector('#wizard-step-2 .wizard-main');
            if (main) main.scrollTop = main.scrollHeight;
        }
    });
    await sleep(250);
    await capture(2000);

    // Preset settings and Spawn are now on the same step (Option A collapse:
    // 6 steps → 3). No navigation click needed between them.
    await page.evaluate(() => {
        const card = document.getElementById('spawn-config-card');
        if (card) card.scrollIntoView({ behavior: 'instant', block: 'start' });
    });
    await sleep(300);
    await capture(2500); // Hold on the Spawn step — Rapid-MLX config card + Spawn Server button.

    // ── Convert frames → GIF ──────────────────────────────────────────────────
    console.log(`[CAPTURE] Converting ${frameIdx} frames to GIF at ${fps} fps (scaled to 900 px)...`);
    const output = tagFilename('spawn-wizard-rapid-mlx-flow.gif');
    execFileSync('ffmpeg', [
        '-y',
        '-framerate', String(fps),
        '-i', join(FRAME_DIR, `spawn-wizard-rapid-mlx-gif_%03d.png`),
        '-vf', 'scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5',
        join(currentArtifactsDir(), output),
    ], { stdio: 'inherit' });
    recordArtifact(output, page.viewport());
    cleanupFrames();
    console.log('[CAPTURE] spawn-wizard-rapid-mlx-flow.gif complete.');
}
