let drawer = null;
let opener = null;

const STATUS_LABELS = {
    good: 'Qualified',
    caution: 'Use with care',
    blocked: 'Not recommended',
    info: 'How this was decided',
};

function appendText(parent, tag, className, value) {
    if (value === undefined || value === null || value === '') return null;
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = String(value);
    parent.appendChild(element);
    return element;
}

function appendList(parent, title, values, className = '') {
    const items = (values || []).filter(value => value !== undefined && value !== null && value !== '');
    if (!items.length) return;
    const section = document.createElement('section');
    section.className = `evidence-drawer-list ${className}`.trim();
    appendText(section, 'h4', '', title);
    const list = document.createElement('ul');
    for (const value of items) appendText(list, 'li', '', value);
    section.appendChild(list);
    parent.appendChild(section);
}

function buildDrawer() {
    const root = document.createElement('div');
    root.id = 'evidence-drawer';
    root.className = 'evidence-drawer';
    root.hidden = true;

    const backdrop = document.createElement('button');
    backdrop.type = 'button';
    backdrop.className = 'evidence-drawer-backdrop';
    backdrop.setAttribute('aria-label', 'Close explanation');
    backdrop.dataset.evidenceClose = 'true';

    const panel = document.createElement('aside');
    panel.className = 'evidence-drawer-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    panel.setAttribute('aria-labelledby', 'evidence-drawer-title');
    panel.setAttribute('aria-describedby', 'evidence-drawer-summary');
    panel.tabIndex = -1;

    const header = document.createElement('header');
    header.className = 'evidence-drawer-header';
    const heading = document.createElement('div');
    appendText(heading, 'div', 'evidence-drawer-eyebrow', 'Decision evidence');
    const title = appendText(heading, 'h2', '', 'How this was decided');
    title.id = 'evidence-drawer-title';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'evidence-drawer-close';
    close.setAttribute('aria-label', 'Close explanation');
    close.dataset.evidenceClose = 'true';
    close.textContent = '×';
    header.append(heading, close);

    const body = document.createElement('div');
    body.className = 'evidence-drawer-body';
    body.id = 'evidence-drawer-body';
    panel.append(header, body);
    root.append(backdrop, panel);
    document.body.appendChild(root);

    root.addEventListener('click', event => {
        if (event.target.closest('[data-evidence-close]')) closeEvidenceDrawer();
    });
    root.addEventListener('keydown', event => {
        if (event.key === 'Escape') {
            event.preventDefault();
            closeEvidenceDrawer();
            return;
        }
        if (event.key !== 'Tab') return;
        const focusable = [...panel.querySelectorAll('button:not([disabled]), a[href], summary, [tabindex]:not([tabindex="-1"])')];
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    });
    return root;
}

export function initEvidenceDrawer() {
    if (!drawer) drawer = document.getElementById('evidence-drawer') || buildDrawer();
    window.openEvidenceDrawer = openEvidenceDrawer;
    window.closeEvidenceDrawer = closeEvidenceDrawer;
    return drawer;
}

export function openEvidenceDrawer(data = {}, source = document.activeElement) {
    const root = initEvidenceDrawer();
    const body = root.querySelector('#evidence-drawer-body');
    const title = root.querySelector('#evidence-drawer-title');
    body.replaceChildren();
    title.textContent = data.title || 'How this was decided';

    const status = data.status || 'info';
    const verdict = document.createElement('section');
    verdict.className = `evidence-drawer-verdict is-${status}`;
    appendText(verdict, 'div', 'evidence-drawer-status', data.statusLabel || STATUS_LABELS[status] || STATUS_LABELS.info);
    const summary = appendText(verdict, 'p', 'evidence-drawer-summary', data.summary || 'The available evidence is shown below.');
    summary.id = 'evidence-drawer-summary';
    if (data.consequence) appendText(verdict, 'p', 'evidence-drawer-consequence', data.consequence);
    if (data.remediation) appendText(verdict, 'p', 'evidence-drawer-remediation', data.remediation);
    body.appendChild(verdict);

    const details = document.createElement('details');
    details.className = 'evidence-drawer-details';
    details.open = !!data.expanded;
    appendText(details, 'summary', '', data.technicalLabel || 'Technical evidence');
    const technical = document.createElement('div');
    technical.className = 'evidence-drawer-technical';
    appendList(technical, 'Evidence', data.evidence);
    appendList(technical, 'Requested → effective', data.adjustments, 'is-adjustments');
    appendList(technical, 'Runtime fallthroughs', data.fallthroughs, 'is-fallthroughs');
    appendList(technical, 'Warnings', data.warnings, 'is-warnings');
    appendList(technical, 'Provenance', data.provenance, 'is-provenance');
    if (!technical.childElementCount) appendText(technical, 'p', 'evidence-drawer-empty', 'No additional technical evidence was reported.');
    details.appendChild(technical);
    body.appendChild(details);

    opener = source instanceof HTMLElement ? source : null;
    root.hidden = false;
    requestAnimationFrame(() => {
        root.classList.add('open');
        root.querySelector('.evidence-drawer-close')?.focus();
    });
}

export function closeEvidenceDrawer() {
    if (!drawer || drawer.hidden) return;
    drawer.classList.remove('open');
    const restore = opener;
    opener = null;
    const finish = () => {
        drawer.hidden = true;
        restore?.focus();
    };
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) finish();
    else window.setTimeout(finish, 180);
}

function humanize(value) {
    if (value === undefined || value === null || value === '') return 'not reported';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

export function evidenceFromEstimate(estimate = {}, surface = 'Memory estimate') {
    const recommendation = estimate.recommendation || 'risk';
    let status = recommendation === 'fit' ? 'good' : recommendation === 'tight' ? 'caution' : 'blocked';
    const evidenceKind = estimate.evidence || 'degraded';
    const policy = estimate.execution_policy || {};
    const reasons = Array.isArray(policy.reasons) ? policy.reasons.map(humanize) : [];
    const admissionWarnings = Array.isArray(estimate.mtp_admission?.warnings)
        ? estimate.mtp_admission.warnings.map(humanize)
        : [];
    const admissionFallthroughs = Array.isArray(estimate.mtp_admission?.fallthroughs)
        ? estimate.mtp_admission.fallthroughs.map(reason => `MTP fallthrough: ${humanize(reason)}`)
        : [];
    const admission = estimate.mtp_admission;
    if (admission && !admission.recommended_for_workload) status = 'blocked';
    else if (admission && !admission.engages_for_workload && status === 'good') status = 'caution';
    const requested = policy.requested_policy || {};
    const effective = policy.effective_policy || {};
    const keys = new Set([...Object.keys(requested), ...Object.keys(effective)]);
    const adjustments = [...keys]
        .filter(key => humanize(requested[key]) !== humanize(effective[key]))
        .map(key => `${key}: ${humanize(requested[key])} → ${humanize(effective[key])}`);
    if (estimate.prefill_step_size) adjustments.push(`prefill_step_size: ${estimate.prefill_step_size} (Rapid-MLX-specific)`);

    return {
        title: `${surface} evidence`,
        status,
        summary: estimate.note || (status === 'good' ? 'This configuration fits the reported memory budget.' : 'This configuration needs review before launch.'),
        consequence: evidenceKind === 'measured'
            ? 'The estimate uses measured calibration evidence where available.'
            : evidenceKind === 'approximate'
                ? 'Some runtime overhead is formula-based and may differ on this machine.'
                : 'Model metadata was incomplete, so this is a rougher estimate.',
        remediation: status === 'blocked'
            ? 'Reduce context or retained cache, use a smaller model, or choose a higher-memory machine.'
            : status === 'caution' ? 'Leave additional memory for the operating system and other applications.' : '',
        evidence: [
            `Evidence quality: ${evidenceKind}`,
            admission ? `MTP admission: ${admission.engages_for_workload ? 'scheduler engages' : 'autoregressive fallthrough'} (derived from workload and request shape)` : '',
            admission ? `MTP eligible: ${humanize(admission.eligible)}; recommended for workload: ${humanize(admission.recommended_for_workload)}` : '',
            estimate.native_context_limit ? `Native context limit: ${Number(estimate.native_context_limit).toLocaleString()} tokens` : '',
            estimate.context_extension_required ? 'Selected context exceeds the model native limit.' : '',
        ],
        adjustments,
        fallthroughs: [...admissionFallthroughs, ...admissionWarnings],
        warnings: reasons,
        provenance: [
            estimate.evidence_timestamp ? `Estimator evidence timestamp: ${estimate.evidence_timestamp}` : '',
            estimate.measured_spec_decode ? `Measured speculative qualification: ${humanize(estimate.measured_spec_decode)}` : '',
            estimate.superseded_spec_decode ? `Superseded speculative evidence: ${humanize(estimate.superseded_spec_decode)}` : '',
        ],
    };
}

export async function openEstimateEvidenceDrawer(estimate = {}, surface = 'Memory estimate', trigger = null) {
    let enriched = estimate;
    if (estimate.mtp_admission) {
        try {
            const headers = window.authHeaders ? window.authHeaders() : {};
            const response = await fetch('/api/rapid-mlx/runtime/metadata', { headers });
            if (response.ok) {
                const runtime = await response.json();
                enriched = {
                    ...estimate,
                    evidence_timestamp: runtime.evidence_timestamp ?? estimate.evidence_timestamp,
                    measured_spec_decode: runtime.measured_spec_decode ?? estimate.measured_spec_decode,
                    superseded_spec_decode: runtime.superseded_spec_decode ?? estimate.superseded_spec_decode,
                };
            }
        } catch {
            // The estimator admission result remains useful when runtime metadata is unavailable.
        }
    }
    openEvidenceDrawer(evidenceFromEstimate(enriched, surface), trigger);
}

const EVIDENCE_CLASS_STATUS = {
    exact: 'good',
    compatible: 'caution',
    related: 'caution',
    stale: 'blocked',
};

const EVIDENCE_CLASS_TITLES = {
    exact: 'Measured on this machine',
    compatible: 'Compatible model evidence',
    related: 'Related model evidence',
    stale: 'Stale evidence',
};

function formatGiB(bytes) {
    if (bytes === undefined || bytes === null) return null;
    return `${(Number(bytes) / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

export function evidenceFromLaunchObservation(evidence = {}) {
    const evidenceClass = evidence.class || 'related';
    const status = EVIDENCE_CLASS_STATUS[evidenceClass] || 'info';
    const detail = evidence.detail || null;
    const capturedAt = detail?.captured_unix_ms
        ? new Date(Number(detail.captured_unix_ms)).toLocaleString()
        : null;
    return {
        title: EVIDENCE_CLASS_TITLES[evidenceClass] || 'Launch memory evidence',
        status,
        summary: evidence.summary || 'Measured launch memory evidence is available for this configuration.',
        consequence: evidenceClass === 'stale'
            ? 'This receipt has aged past the freshness window and may no longer reflect the current binary or arguments.'
            : evidenceClass === 'exact'
                ? 'This is a direct measurement of this exact launch configuration on this machine.'
                : 'This measurement comes from a related configuration, not an exact match.',
        remediation: evidenceClass === 'stale' ? 'Re-launch this configuration to refresh the evidence.' : '',
        evidence: detail ? [
            `Method: ${detail.method}`,
            `Before: ${formatGiB(detail.before_bytes)}`,
            `Peak: ${formatGiB(detail.peak_bytes)}`,
            `After: ${formatGiB(detail.after_bytes)}`,
            detail.model_delta_bytes !== undefined && detail.model_delta_bytes !== null
                ? `Model delta: ${formatGiB(detail.model_delta_bytes)}`
                : '',
            `Samples: ${detail.sample_count} (${detail.interval_ms}ms interval)`,
            capturedAt ? `Captured: ${capturedAt}` : '',
        ] : [],
        warnings: detail?.noise_flags || [],
        technicalLabel: 'Measurement detail',
        expanded: evidenceClass === 'stale',
    };
}

export function evidenceFromCommandPreview(preview = {}) {
    const diffs = preview.requested_vs_effective && typeof preview.requested_vs_effective === 'object'
        ? Object.entries(preview.requested_vs_effective)
        : [];
    const adjustments = diffs.map(([key, value]) => {
        if (value && typeof value === 'object') {
            return `${key}: ${humanize(value.requested)} → ${humanize(value.effective)}${value.reason ? ` — ${value.reason}` : ''}`;
        }
        return `${key}: ${humanize(value)}`;
    });
    const reasons = Array.isArray(preview.reasons) ? preview.reasons.map(humanize) : [];
    const policy = preview.effective_policy || {};
    const evidence = Object.entries(policy).map(([key, value]) => `${key}: ${humanize(value)}`);
    return {
        title: 'Launch policy evidence',
        status: adjustments.length || reasons.length ? 'caution' : 'good',
        summary: adjustments.length
            ? 'The launch preview differs from one or more requested settings.'
            : 'The preview found no requested settings that the runtime would change or drop.',
        consequence: 'This is the effective Rapid-MLX policy used to construct the launch command.',
        remediation: adjustments.length ? 'Review the adjustments before starting the server.' : '',
        evidence,
        adjustments,
        warnings: reasons,
    };
}
