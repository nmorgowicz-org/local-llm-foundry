// Screenshot/gif capture primitives.
// Extracted from tests/ui/capture.mjs (Phase A1).
import fs from 'fs';
import { join } from 'path';
import { execFileSync } from 'child_process';
import http from 'http';
import https from 'https';
import { CAPTURE_REMOTE_IDLE_TIMEOUT_MS, CAPTURE_REMOTE_REQUEST_TIMEOUT_MS, FRAME_DIR, REMOTE_SERVER, SCREENSHOT_TAB_PREFIX, currentArtifactsDir, tagFilename, sleep } from './paths.mjs';
import { recordCapture } from './receipt.mjs';

export async function cleanupScreenshotTabs(page, { keepOne = false } = {}) {
    await page.evaluate(async ({ prefix, keepOne }) => {
        const { chat } = await import('/js/core/app-state.js');
        const { newChatTab, persistChatTabs } = await import('/js/features/chat-state.js');
        const { renderChatTabs, renderChatMessages } = await import('/js/features/chat-render.js');
        const { renderChatSessionsSidebar } = await import('/js/features/chat-sessions-sidebar.js');

        const screenshotTabs = chat.tabs.filter(tab => tab.name.startsWith(prefix));
        const keepId = keepOne ? screenshotTabs.at(-1)?.id : null;

        chat.tabs = chat.tabs.filter(tab => {
            if (!tab.name.startsWith(prefix)) return true;
            return keepOne && tab.id === keepId;
        });

        if (!chat.tabs.length) {
            const fallback = newChatTab('Chat 1');
            chat.tabs = [fallback];
            chat.activeTabId = fallback.id;
        } else if (!chat.tabs.some(tab => tab.id === chat.activeTabId)) {
            chat.activeTabId = chat.tabs[chat.tabs.length - 1].id;
        }

        renderChatTabs();
        renderChatMessages();
        renderChatSessionsSidebar();
        await persistChatTabs();
    }, { prefix: SCREENSHOT_TAB_PREFIX, keepOne });
}

export async function captureShot(page, rawFilename, options = {}) {
    const filename = tagFilename(rawFilename, options.runtimeTag);
    const { fullPage = true, expandSelector, runtimeTag, ...screenshotOptions } = options;

    // Non-full-page captures are disabled by default, except when a scenario
    // explicitly supplies an expansion selector for a real viewport capture.
    if (!fullPage && !expandSelector) {
        console.log(`[CAPTURE] Skipped non-full-page: ${filename}`);
        return;
    }

    // A prior elementHandle.click()/hover() leaves Puppeteer's virtual mouse
    // parked on that element; if it has a `title`, headless Chrome renders
    // the native tooltip into the page's own render surface and it shows up
    // in the screenshot. Park the mouse off any content before every shot.
    await page.mouse.move(0, 0).catch(() => {});

    // The spawn wizard modal (`#spawn-wizard-overlay`) is `position: fixed`,
    // so fullPage:true (which captures based on document/body scrollHeight)
    // never sees it, and `.wizard-body` scrolls internally
    // (`overflow-y: auto`) inside a modal that itself caps at
    // `max-height: min(92vh, 940px)`. We deliberately do NOT flatten/expand
    // any of that to capture the entire scrollable content in one giant
    // image — that produces unrealistic 3000-5000px screenshots nobody's
    // browser actually shows. Instead, scroll the target container to the
    // top (its natural resting scroll position) and capture at normal
    // viewport size, same as what a user actually sees.
    if (expandSelector) {
        await page.evaluate((sel) => {
            const el = document.querySelector(sel);
            if (el) {
                el.scrollTop = 0;
                el.scrollLeft = 0;
            }
        }, expandSelector);
        await sleep(200);
        await page.screenshot({ path: join(currentArtifactsDir(), filename), fullPage: false, ...screenshotOptions });
        recordCapture(filename, page.viewport());
        console.log(`[CAPTURE] Saved ${filename}`);
        return;
    }

    await page.screenshot({ path: join(currentArtifactsDir(), filename), fullPage: true, ...screenshotOptions });
    recordCapture(filename, page.viewport());
    console.log(`[CAPTURE] Saved ${filename}`);
}

export async function captureCloseUp(page, selector, rawFilename, options = {}) {
    if (!options.closeUp) return;
    const filename = tagFilename(rawFilename, options.runtimeTag);
    const padding = options.padding ?? 24;
    const handle = await page.$(selector);
    if (!handle) {
        console.log(`[CAPTURE] Close-up skipped (not found): ${selector}`);
        return;
    }
    await handle.evaluate(el => {
        el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' });
    });
    await sleep(300);
    const box = await handle.boundingBox();
    if (!box) return;
    const viewport = page.viewport();
    const clip = {
        x: Math.max(0, box.x - padding),
        y: Math.max(0, box.y - padding),
        width: Math.min((viewport?.width ?? box.width) - Math.max(0, box.x - padding), box.width + padding * 2),
        height: Math.min((viewport?.height ?? box.height) - Math.max(0, box.y - padding), box.height + padding * 2),
    };
    const cuName = filename.replace('.png', '-cu.png');
    await page.mouse.move(0, 0).catch(() => {});
    await page.screenshot({ path: join(currentArtifactsDir(), cuName), clip });
    console.log(`[CAPTURE] Close-up saved ${cuName}`);
}

export async function captureElementScreenshot(page, selector, rawFilename, options = {}) {
    // Always capture; --close-up is only for captureCloseUp helper.
    const filename = tagFilename(rawFilename, options.runtimeTag);

    const padding = options.padding ?? 20;
    const handle = await page.$(selector);
    if (!handle) {
        throw new Error(`Missing selector for screenshot capture: ${selector}`);
    }

    await handle.evaluate(el => {
        el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'nearest' });
    });
    await sleep(options.settleMs ?? 500);

    const box = await handle.boundingBox();
    if (!box) {
        throw new Error(`Selector has no visible bounds: ${selector}`);
    }

    const viewport = page.viewport();
    const clip = {
        x: Math.max(0, box.x - padding),
        y: Math.max(0, box.y - padding),
        width: Math.min((viewport?.width ?? box.width) - Math.max(0, box.x - padding), box.width + padding * 2),
        height: Math.min((viewport?.height ?? box.height) - Math.max(0, box.y - padding), box.height + padding * 2),
    };

    await page.mouse.move(0, 0).catch(() => {});
    await page.screenshot({ path: join(currentArtifactsDir(), filename), clip });
    console.log(`[CAPTURE] Saved ${filename}`);
}

export async function captureSparklineClips(page, selector) {
    const rects = await page.$$eval(selector, els => els.map((el, index) => {
        const rect = el.getBoundingClientRect();
        return {
            index,
            x: rect.x + window.scrollX,
            y: rect.y + window.scrollY,
            width: rect.width,
            height: rect.height,
        };
    }).filter(rect => rect.width > 0 && rect.height > 0));

    for (const rect of rects) {
        await page.screenshot({
            path: join(currentArtifactsDir(), `sparkline-validate-svg-${rect.index}.png`),
            clip: {
                x: Math.max(0, rect.x),
                y: Math.max(0, rect.y),
                width: Math.max(1, rect.width),
                height: Math.max(1, rect.height),
            },
        });
    }
    console.log(`[CAPTURE] Saved ${rects.length} sparkline SVG clips`);
}

// Resolve once the remote reports no processing slot, so a busy shared runtime
// (other agents, benchmarks) does not queue the capture's request invisibly.
export async function waitForRemoteIdle(remoteServer = REMOTE_SERVER, timeoutMs = CAPTURE_REMOTE_IDLE_TIMEOUT_MS) {
    const start = Date.now();
    let announced = false;
    while (Date.now() - start < timeoutMs) {
        try {
            const response = await fetch(`${remoteServer}/slots`, { signal: AbortSignal.timeout(10000) });
            if (response.ok) {
                const slots = await response.json();
                if (Array.isArray(slots) && !slots.some(slot => slot.is_processing)) return;
            } else {
                return; // /slots unavailable: nothing to wait on.
            }
        } catch {
            // Busy servers may be slow to answer /slots; keep waiting.
        }
        if (!announced) {
            console.log('[CAPTURE] Remote is busy; waiting for it to go idle...');
            announced = true;
        }
        await sleep(5000);
    }
    throw new Error(`Remote ${remoteServer} did not become idle within ${Math.ceil(timeoutMs / 1000)}s`);
}

// node:http has no header/body timeout of its own, unlike global fetch (undici
// aborts at 300s), so a long queued prefill can finish.
export async function startLiveGeneration(remoteServer = REMOTE_SERVER, timeoutMs = CAPTURE_REMOTE_REQUEST_TIMEOUT_MS) {
    const url = new URL(`${remoteServer}/v1/chat/completions`);
    const payload = JSON.stringify({
        model: 'default',
        stream: false,
        temperature: 0.7,
        max_tokens: 800,
        messages: [{
            role: 'user',
            content: 'Write a dense explanation of transformer inference performance, token throughput, KV cache behavior, and GPU offload tradeoffs.',
        }],
    });
    const client = url.protocol === 'https:' ? https : http;
    return new Promise((resolve, reject) => {
        const request = client.request(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
            timeout: timeoutMs, // socket inactivity; non-streaming replies are silent until done
        }, response => {
            response.resume();
            response.on('end', () => {
                if (response.statusCode >= 200 && response.statusCode < 300) resolve();
                else reject(new Error(`Generation request failed: ${response.statusCode} ${response.statusMessage}`));
            });
            response.on('error', reject);
        });
        request.on('timeout', () => request.destroy(new Error(`Generation request timed out after ${Math.ceil(timeoutMs / 1000)}s`)));
        request.on('error', reject);
        request.end(payload);
    });
}

export async function captureFrames(page, prefix, totalFrames, fps) {
    fs.mkdirSync(FRAME_DIR, { recursive: true });
    for (let i = 0; i < totalFrames; i += 1) {
        const path = join(FRAME_DIR, `${prefix}_${String(i).padStart(3, '0')}.png`);
        await page.screenshot({ path });
        if (process.env.CAPTURE_DEBUG_WS) {
            const dbg = await page.evaluate(async () => {
                const { lastLlamaMetrics } = await import('/js/core/app-state.js');
                return {
                    slots_processing: lastLlamaMetrics?.slots_processing,
                    prompt_tps: lastLlamaMetrics?.prompt_tokens_per_sec,
                    gen_tps: lastLlamaMetrics?.generation_tokens_per_sec,
                    status: lastLlamaMetrics?.status,
                };
            });
            console.log(`[DEBUG frame ${i}]`, JSON.stringify(dbg));
        }
        await sleep(1000 / fps);
    }
}

export function framesToGif(prefix, output, fps) {
    execFileSync('ffmpeg', [
        '-y',
        '-framerate', String(fps),
        '-i', join(FRAME_DIR, `${prefix}_%03d.png`),
        '-vf', 'scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5',
        output,
    ], { stdio: 'inherit' });
}

export function cleanupFrames() {
    fs.rmSync(FRAME_DIR, { recursive: true, force: true });
}

/**
 * Wait for Rapid-MLX real telemetry data to populate.
 * Polls /api/rapid-mlx/runtime/status until it returns active data with model and version.
 * Returns the status object or throws on timeout.
 */

export async function waitForRapidTelemetry(page, timeoutMs = 60000) {
    const start = Date.now();
    console.log(`[CAPTURE] waitForRapidTelemetry: waiting for real Rapid-MLX telemetry (${timeoutMs}ms)...`);

    while (Date.now() - start < timeoutMs) {
        try {
            const result = await page.evaluate(async () => {
                const headers = window.authHeaders ? { ...window.authHeaders() } : {};

                // Check llama-monitor's runtime/status endpoint first.
                try {
                    const statusResp = await fetch('/api/rapid-mlx/runtime/status', { headers });
                    if (statusResp.ok) {
                        const status = await statusResp.json();
                        if (status?.runtime?.active?.model && status?.runtime?.active?.version) {
                            return { type: 'manager', status };
                        }
                    }
                } catch {
                    // Ignore errors from runtime/status
                }

                // Fall back to checking the active session's rapid-mlx backend via llama-monitor.
                const activeResp = await fetch('/api/sessions/active', { headers });
                if (!activeResp.ok) return null;
                const active = await activeResp.json();
                if (active.backend !== 'rapid_mlx' || active.status !== 'Running') return null;
                const port = active.mode?.match(/:(\d+)/)?.[1];
                if (!port) return null;
                const model = active.model_identity || null;
                if (model) {
                    return { type: 'session', model, port, mode: active.mode };
                }
                return null;
            });

            if (result?.type === 'manager') {
                console.log('[CAPTURE] waitForRapidTelemetry: active (via manager) with model:', result.status.runtime.active.model);
                return result.status;
            }
            if (result?.type === 'session') {
                console.log('[CAPTURE] waitForRapidTelemetry: active (via session) with model:', result.model, 'on port', result.port);
                return { runtime: { active: { model: result.model, port: result.port, mode: result.mode } } };
            }
        } catch {
            // Keep polling
        }
        await sleep(2000);
    }

    throw new Error(`waitForRapidTelemetry: no active telemetry within ${timeoutMs}ms`);
}

/**
 * Clean up a Rapid-MLX live test preset by ID.
 */

export async function deleteRapidLiveTestPreset(page, presetId) {
    try {
        await page.evaluate(async (id) => {
            await fetch(`/api/presets/${encodeURIComponent(id)}`, {
                method: 'DELETE',
                headers: { ...(window.authHeaders ? window.authHeaders() : {}) },
            });
        }, presetId);
        console.log(`[CAPTURE] rapid-mlx-live: cleaned up preset ${presetId}`);
    } catch (e) {
        console.log(`[CAPTURE] rapid-mlx-live: preset cleanup non-fatal: ${e.message}`);
    }
}

export async function describePopover(page, toggleSelector, panelSelector) {
    return page.evaluate(({ toggleSelector, panelSelector }) => {
        const toggle = document.querySelector(toggleSelector);
        const panel = document.querySelector(panelSelector);
        if (!toggle || !panel) {
            return { missing: true, toggleFound: !!toggle, panelFound: !!panel };
        }

        const panelStyle = getComputedStyle(panel);
        const toggleRect = toggle.getBoundingClientRect();
        const panelRect = panel.getBoundingClientRect();
        return {
            missing: false,
            toggleAriaExpanded: toggle.getAttribute('aria-expanded'),
            panelClass: panel.className,
            panelOpacity: panelStyle.opacity,
            panelDisplay: panelStyle.display,
            panelPointerEvents: panelStyle.pointerEvents,
            panelHeight: Math.round(panelRect.height),
            panelWidth: Math.round(panelRect.width),
            opensUpward: panelRect.bottom <= toggleRect.top,
        };
    }, { toggleSelector, panelSelector });
}

export async function describeQuickGuideFlow(page) {
    return page.evaluate(() => {
        const errors = Array.from(document.querySelectorAll('#chat-messages .chat-error')).map(el => el.textContent?.trim()).filter(Boolean);
        const assistantMessages = Array.from(document.querySelectorAll('#chat-messages .chat-message-assistant'));
        const lastAssistant = assistantMessages.at(-1)?.querySelector('.chat-msg-body')?.textContent?.trim() ?? null;
        const container = document.getElementById('quick-guide-container');
        const activeMode = document.querySelector('.quick-guide-mode-btn.active')?.dataset.guideMode ?? null;
        const armedChip = document.getElementById('quick-guide-status-chip')?.textContent?.trim() ?? null;
        return {
            assistantCount: assistantMessages.length,
            lastAssistantPreview: lastAssistant?.slice(0, 240) ?? null,
            errorCount: errors.length,
            latestError: errors.at(-1) ?? null,
            quickGuideExpanded: container?.classList.contains('quick-guide-expanded') ?? false,
            activeMode,
            armedChip,
        };
    });
}

export async function enableGuidedGeneration(page) {
    await page.evaluate(() => {
        const settings = JSON.parse(localStorage.getItem('llama_monitor_settings') || '{}');
        settings.enabled_context_notes = true;
        settings.enabled_suggestions = true;
        settings.enabled_quick_guide = true;
        localStorage.setItem('llama_monitor_settings', JSON.stringify(settings));
    });
    await sleep(500);
}

// ── Welcome Screen ──────────────────────────────────────────────────────────────

// Hide the toast stack (persistent-notification cards included) for clean
// captures: existing toasts are removed and the container is pinned shut, so
// later showToast calls render nothing.
export async function suppressToasts(page) {
  await page.addStyleTag({ content: '#toast-container { display: none !important; }' });
  await page.evaluate(() => {
    document.getElementById('toast-container')?.replaceChildren();
  }).catch(() => {});
}
