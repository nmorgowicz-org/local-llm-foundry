// Pure view-model helpers for the llama.cpp runtime pill and panel.
//
// No DOM and no fetch here: everything takes plain API payloads and returns
// plain values, so the pill, the panel and the tests all agree on one reading
// of the server's state. The server owns every decision (what may run, what is
// cancellable); this module only chooses words and enablement from its fields.

export const PHASES = ['preflight', 'fetch', 'configure', 'compile', 'stage', 'promote'];

/** `foundry_source_build` → `src`, etc. Unknown/absent provenance has no suffix. */
export function provenanceSuffix(provenance) {
    switch (provenance?.source) {
        case 'foundry_source_build': return 'src';
        case 'upstream_prebuilt': return 'prebuilt';
        case 'custom': return 'custom';
        default: return '';
    }
}

export function isSourceBuild(provenance) {
    return provenance?.source === 'foundry_source_build';
}

/**
 * The build number to show and compare. A shallow source checkout embeds
 * `build 1` in the binary, so for a source build the recorded tag is the only
 * truthful number; a `master` build has none (it shows its reference instead).
 */
export function effectiveBuild(build, provenance) {
    if (!isSourceBuild(provenance)) return build ?? null;
    const m = /^b(\d+)$/.exec(provenance.reference || '');
    return m ? Number(m[1]) : null;
}

/** Phase + ninja `[n/m]` progress, e.g. `compile 63%`. */
export function phaseProgressText(job) {
    if (!job) return '';
    const phase = job.phase || '';
    const p = job.progress;
    if (p && p.total > 0 && phase === 'compile') {
        return `${phase} ${Math.floor((p.done / p.total) * 100)}%`;
    }
    return phase;
}

export function jobIsActive(job) {
    return !!job && job.status === 'running';
}

/**
 * The one pill description. Precedence: building, restart-failed, failed
 * build, rolled-back, update available, idle.
 *
 * `dismissed` is the set of job ids / transaction ids the user acknowledged.
 */
export function pillModel({ build, latestBuild, provenance, job, runtimeLast, dismissed }) {
    const seen = dismissed || new Set();
    if (jobIsActive(job)) {
        return {
            state: 'building',
            text: `Building · ${phaseProgressText(job)}`,
            title: 'A llama.cpp source build is running. Click to open the runtime panel.',
            opensPanel: true,
        };
    }
    if (job && job.status === 'succeeded_restart_failed' && !seen.has(job.id)) {
        return {
            state: 'restart-failed',
            text: 'Runtime installed; model restart failed',
            title: 'The new runtime is installed and verified, but the model did not restart. Open the runtime panel to retry or roll back.',
            opensPanel: true,
        };
    }
    if (job && (job.status === 'failed' || job.status === 'interrupted') && !seen.has(job.id)) {
        const where = job.failure_phase || job.phase || 'build';
        return {
            state: 'error',
            text: job.status === 'interrupted' ? 'Build interrupted' : `Build failed: ${where}`,
            title: 'The last llama.cpp build did not finish. Click for the error and the fix.',
            opensPanel: true,
        };
    }
    if (runtimeLast && runtimeLast.outcome === 'rolled_back' && !seen.has(`tx:${runtimeLast.id}`)) {
        return {
            state: 'rolled-back',
            text: 'Update rolled back',
            title: 'The last runtime update failed verification and the previous runtime was restored.',
            opensPanel: true,
        };
    }
    const suffix = provenanceSuffix(provenance);
    const eff = effectiveBuild(build, provenance);
    const name = eff ? `b${eff}` : (isSourceBuild(provenance) && provenance.reference ? provenance.reference : '');
    const base = name ? `llama.cpp · ${name}${suffix ? ` · ${suffix}` : ''}` : 'llama.cpp';
    if (eff && latestBuild && latestBuild > eff) {
        return {
            state: 'update',
            text: `llama.cpp · ↑ b${latestBuild}${suffix ? ` · ${suffix}` : ''}`,
            title: `Update available: b${eff} → b${latestBuild}. Click to manage.`,
            opensPanel: false,
        };
    }
    return {
        state: 'idle',
        text: base,
        title: 'llama-server binary version',
        opensPanel: false,
    };
}

/**
 * Why "Build & install" is disabled, or '' when it is allowed. Warnings never
 * disable it. Ownership conflicts disable it with the owner's name.
 */
export function buildDisabledReason({ report, owner, activeJobId, validating }) {
    if (activeJobId) return 'A build is already running.';
    if (owner) return `Busy: ${owner} is using the build tree.`;
    if (validating) return 'Validation is running.';
    if (!report) return 'Run Validate first.';
    const failing = (report.checks || []).filter(c => c.status === 'fail');
    if (failing.length) {
        return `Fix first: ${failing.map(c => c.title).join(', ')}.`;
    }
    return '';
}

/** Cancel is only offered before the journal begins. */
export function cancelState(job) {
    if (!jobIsActive(job)) return { visible: false, enabled: false, reason: '' };
    if (job.cancellable) return { visible: true, enabled: true, reason: '' };
    return { visible: true, enabled: false, reason: 'Installing/recovering — cannot cancel safely' };
}

/** Rollback link text, only when a previous set exists. */
export function rollbackLabel(runtimeStatus) {
    if (!runtimeStatus?.previous_available) return '';
    const label = runtimeStatus.previous?.label;
    return label ? `Roll back to ${label}` : 'Roll back to previous runtime';
}

export function formatBytes(n) {
    if (!Number.isFinite(n) || n <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}

/** One current-runtime summary line from provenance. */
export function currentRuntimeLine(provenance, build) {
    if (!provenance) return build ? `b${build}` : 'Unknown runtime';
    const parts = [provenance.label || (build ? `b${build}` : 'Unknown')];
    if (provenance.source === 'foundry_source_build') parts.push('source build');
    else if (provenance.source === 'upstream_prebuilt') parts.push('prebuilt');
    else if (provenance.source === 'custom') parts.push('custom');
    if (provenance.fa_quants_spec) parts.push(`FA ${provenance.fa_quants_spec}`);
    if (provenance.cuda_architectures) parts.push(`CUDA ${provenance.cuda_architectures}`);
    return parts.join(' · ');
}

/** The full copyable diagnostics bundle (failure JSON + prereqs + provenance + environment). */
export function diagnosticsText({ job, report, provenance, status }) {
    return JSON.stringify({
        failure: job?.failure ?? null,
        job: job ? { id: job.id, status: job.status, phase: job.phase, outcome: job.outcome } : null,
        prerequisites: report?.checks?.map(c => ({ id: c.id, status: c.status, detail: c.detail })) ?? [],
        provenance: provenance ?? null,
        environment: { os: status?.os ?? null, accelerator: job?.accelerator ?? null },
        log_path: job?.failure?.log_path ?? null,
    }, null, 2);
}
