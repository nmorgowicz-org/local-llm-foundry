// llama.cpp-only server totals and optional attached-runtime facts.
// Missing counters stay unavailable. A reported zero is supported, but inactive.
let activeIdentity = null;
let factsSignature = '';

const counter = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const count = value => value.toLocaleString(undefined, { maximumFractionDigits: 0 });
const percentage = value => `${(value * 100).toFixed(1)}%`;

function text(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function show(id, visible) {
    const el = document.getElementById(id);
    if (el) el.hidden = !visible;
}

function resetEfficiency() {
    ['inference-efficiency', 'prompt-cache-reuse', 'speculative-effectiveness'].forEach(id => show(id, false));
    ['cache-reuse-value', 'spec-acceptance-value', 'spec-tokens-per-verification'].forEach(id => text(id, '—'));
    ['cache-reuse-status', 'cache-reuse-cached', 'cache-reuse-processed', 'spec-effectiveness-status', 'spec-effectiveness-counts'].forEach(id => text(id, ''));
    const bar = document.getElementById('cache-reuse-bar');
    bar?.removeAttribute('aria-valuenow');
    bar?.removeAttribute('aria-valuetext');
    if (bar) bar.dataset.awaiting = 'true';
    const fill = document.getElementById('cache-reuse-fill');
    if (fill) fill.style.width = '0%';
    const acceptanceBar = document.getElementById('spec-acceptance-bar');
    if (acceptanceBar) {
        acceptanceBar.hidden = true;
        acceptanceBar.dataset.awaiting = 'true';
        acceptanceBar.removeAttribute('aria-valuenow');
        acceptanceBar.removeAttribute('aria-valuetext');
    }
    const acceptanceFill = document.getElementById('spec-acceptance-fill');
    if (acceptanceFill) acceptanceFill.style.width = '0%';
    const grid = document.getElementById('efficiency-grid');
    if (grid) grid.dataset.cardCount = '0';
}

function clearRuntime() {
    show('llama-runtime-card', false);
    show('server-model-identity', false);
    text('server-model-identity', '');
    document.getElementById('llama-runtime-facts')?.replaceChildren();
    ['llama-runtime-model', 'llama-runtime-meta'].forEach(id => {
        text(id, '');
        document.getElementById(id)?.removeAttribute('title');
    });
    document.getElementById('llama-runtime-badges')?.replaceChildren();
    const popover = document.getElementById('llama-runtime-popover');
    if (popover?.matches(':popover-open')) popover.hidePopover();
    factsSignature = '';
}

// Paths are not runtime facts. Keep basename-only identity/adapters on this surface.
function publicName(value) {
    if (typeof value !== 'string') return '';
    return value.trim().split(/[\\/]/).pop() || '';
}

function optionalText(value) {
    return typeof value === 'string' ? value.trim() : '';
}

function renderRuntime(facts, attachedModel) {
    const safeFacts = facts && typeof facts === 'object' ? facts : {};
    const model = publicName(safeFacts.model_name);
    const alias = publicName(safeFacts.model_alias) || publicName(attachedModel);
    const rows = [];
    if (model) rows.push(['Model', model]);
    if (alias) rows.push(['Alias', alias]);
    const quantization = optionalText(safeFacts.quantization);
    if (quantization) rows.push(['Quantization', quantization]);
    const parameters = counter(safeFacts.model_params);
    if (parameters > 0) {
        const unit = parameters >= 1e9 ? 'B' : parameters >= 1e6 ? 'M' : '';
        const divisor = unit === 'B' ? 1e9 : unit === 'M' ? 1e6 : 1;
        rows.push(['Parameters', `${(parameters / divisor).toLocaleString(undefined, { maximumFractionDigits: 2 })}${unit}`]);
    }
    const build = optionalText(safeFacts.server_build);
    if (build) rows.push(['Server build', build]);
    const capabilityFacts = safeFacts.capabilities && typeof safeFacts.capabilities === 'object' && !Array.isArray(safeFacts.capabilities)
        ? Object.entries(safeFacts.capabilities).filter(([, available]) => typeof available === 'boolean') : [];
    const labels = {
        audio: 'Audio', video: 'Video', vision: 'Vision', tools: 'Tools',
        supports_tools: 'Tools', supports_tool_calls: 'Tool calls',
        supports_parallel_tool_calls: 'Parallel tool calls', supports_object_arguments: 'Object arguments',
        supports_preserve_reasoning: 'Preserve reasoning', supports_reasoning_effort: 'Reasoning effort',
        supports_string_content: 'String content', supports_system_role: 'System role',
        supports_typed_content: 'Typed content',
    };
    const capabilities = capabilityFacts.map(([name, available]) => {
        const label = labels[name] || name.replace(/^supports_/, '').replace(/_/g, ' ');
        return available ? label : `${label} (unsupported)`;
    });
    if (capabilities.length) rows.push(['Capabilities', capabilities]);
    let adapterCount = 0;
    if (Array.isArray(safeFacts.adapters)) {
        const adapters = safeFacts.adapters.map(adapter => {
            if (!adapter || typeof adapter !== 'object') return '';
            const name = publicName(adapter.name) || (counter(adapter.id) !== null ? `Adapter ${adapter.id}` : '');
            if (!name) return '';
            return typeof adapter.scale === 'number' && Number.isFinite(adapter.scale)
                ? `${name} (scale ${adapter.scale})` : name;
        }).filter(Boolean);
        adapterCount = adapters.length;
        rows.push(['Adapters', adapters.length ? adapters.join(', ') : 'None']);
    }
    show('server-model-identity', !!(model || alias));
    text('server-model-identity', model ? `llama.cpp · ${model}` : alias ? `llama.cpp · Alias: ${alias}` : '');
    if (!rows.length) {
        clearRuntime();
        return;
    }
    show('llama-runtime-card', true);
    const signature = JSON.stringify(rows);
    if (signature === factsSignature) return;
    factsSignature = signature;
    const modelElement = document.getElementById('llama-runtime-model');
    if (modelElement) {
        modelElement.textContent = model || (alias ? `Alias: ${alias}` : 'Model unavailable');
        modelElement.title = model || (alias ? `Alias: ${alias}` : '');
    }
    const parameterLabel = rows.find(([label]) => label === 'Parameters')?.[1];
    const metadata = [quantization, parameterLabel ? `${parameterLabel} parameters` : '',
        build ? `Server ${build.split('-')[0]}` : '',
        adapterCount ? `${adapterCount} ${adapterCount === 1 ? 'adapter' : 'adapters'}` : ''].filter(Boolean);
    const metaElement = document.getElementById('llama-runtime-meta');
    if (metaElement) {
        metaElement.textContent = metadata.join(' · ');
        metaElement.title = metadata.join(' · ');
    }
    const reported = Object.fromEntries(capabilityFacts);
    const supportedBadges = [
        ['Vision', reported.vision === true], ['Video', reported.video === true],
        ['Tools', reported.tools === true || reported.supports_tools === true || reported.supports_tool_calls === true],
        ['Reasoning', reported.supports_reasoning_effort === true],
    ].filter(([, available]) => available).map(([label]) => {
        const badge = document.createElement('span');
        badge.className = 'llama-runtime-capability';
        badge.textContent = label;
        return badge;
    });
    document.getElementById('llama-runtime-badges')?.replaceChildren(...supportedBadges);
    const list = document.getElementById('llama-runtime-facts');
    if (!list) return;
    const nodes = rows.flatMap(([label, value]) => {
        const term = document.createElement('dt');
        const definition = document.createElement('dd');
        term.textContent = label;
        if (Array.isArray(value)) {
            definition.className = 'llama-runtime-capabilities';
            value.forEach(capability => {
                const badge = document.createElement('span');
                badge.className = 'llama-runtime-capability';
                badge.textContent = capability;
                definition.appendChild(badge);
            });
        } else {
            definition.textContent = value;
        }
        const fact = document.createElement('div');
        fact.className = 'llama-runtime-fact';
        if (label === 'Model' || label === 'Adapters') fact.classList.add('llama-runtime-fact--wide');
        fact.append(term, definition);
        return [fact];
    });
    list.replaceChildren(...nodes);
}

export function isLlamaTelemetryForTarget(metrics, sessionId, endpointTag) {
    const sourceSession = metrics?.telemetry_session_id;
    const sourceEndpoint = metrics?.telemetry_endpoint;
    const normalize = value => typeof value === 'string' ? value.replace(/\/+$/, '') : '';
    // Both source tags are required: cleared/default snapshots are not wildcards.
    // Opaque tags come from the same server helper for the source and WS target.
    return !!sessionId && sourceSession === sessionId
        && !!normalize(sourceEndpoint) && normalize(sourceEndpoint) === normalize(endpointTag);
}

export function renderLlamaCppDetails({ backend, attached, sessionId, endpoint, endpointTag, metrics, attachedModel }) {
    // The session model identity is current even when no telemetry is available.
    const allowed = backend === 'llama_cpp' && attached === true
        && (metrics == null || isLlamaTelemetryForTarget(metrics, sessionId, endpointTag ?? endpoint));
    const identity = allowed ? JSON.stringify([sessionId || '', endpointTag ?? endpoint ?? '']) : null;
    if (identity !== activeIdentity) {
        resetEfficiency();
        clearRuntime();
        activeIdentity = identity;
    }
    if (!allowed) {
        resetEfficiency();
        clearRuntime();
        return;
    }

    renderRuntime(metrics?.runtime_facts, attachedModel);
    // Clear absent/withdrawn measures immediately instead of retaining an old card.
    resetEfficiency();
    const cached = counter(metrics?.prompt_tokens_cached_total);
    const processed = counter(metrics?.prompt_tokens_processed_total);
    const cacheAvailable = cached !== null && processed !== null;
    if (cacheAvailable) {
        show('prompt-cache-reuse', true);
        const total = cached + processed;
        const ratio = total > 0 ? cached / total : null;
        text('cache-reuse-value', ratio === null ? '—' : percentage(ratio));
        text('cache-reuse-status', ratio === null ? 'Awaiting activity' : 'Prompt tokens reused');
        text('cache-reuse-cached', `${count(cached)} cached`);
        text('cache-reuse-processed', `${count(processed)} processed`);
        const bar = document.getElementById('cache-reuse-bar');
        const fill = document.getElementById('cache-reuse-fill');
        if (bar) {
            bar.dataset.awaiting = String(ratio === null);
            if (ratio !== null) bar.setAttribute('aria-valuenow', String(ratio * 100));
            bar.setAttribute('aria-valuetext', ratio === null ? 'Awaiting activity' : `${percentage(ratio)} cached`);
        }
        if (fill) fill.style.width = `${(ratio ?? 0) * 100}%`;
    }

    const accepted = counter(metrics?.speculative_accepted_tokens_total);
    const drafted = counter(metrics?.speculative_draft_tokens_total);
    const steps = counter(metrics?.speculative_verification_steps_total);
    const speculationEnabled = typeof metrics?.speculative_enabled === 'boolean'
        ? metrics.speculative_enabled
        : (Array.isArray(metrics?.slots) && metrics.slots.some(slot => slot?.speculative_enabled === true));
    const acceptanceAvailable = accepted !== null && drafted !== null;
    const verificationAvailable = accepted !== null && steps !== null;
    const specAvailable = speculationEnabled && (acceptanceAvailable || verificationAvailable);
    if (specAvailable) {
        show('speculative-effectiveness', true);
        show('spec-acceptance-metric', acceptanceAvailable);
        show('spec-verification-metric', verificationAvailable);
        const acceptance = acceptanceAvailable && drafted > 0 ? accepted / drafted : null;
        const tokensPerVerification = verificationAvailable && steps > 0 ? 1 + accepted / steps : null;
        text('spec-acceptance-value', acceptance === null ? '—' : percentage(acceptance));
        show('spec-acceptance-bar', acceptanceAvailable);
        const acceptanceBar = document.getElementById('spec-acceptance-bar');
        const acceptanceFill = document.getElementById('spec-acceptance-fill');
        if (acceptanceAvailable && acceptanceBar) {
            acceptanceBar.dataset.awaiting = String(acceptance === null);
            if (acceptance !== null) acceptanceBar.setAttribute('aria-valuenow', String(Math.min(100, acceptance * 100)));
            acceptanceBar.setAttribute('aria-valuetext', acceptance === null ? 'Awaiting activity' : `${percentage(acceptance)} accepted`);
        }
        if (acceptanceFill) acceptanceFill.style.width = `${Math.min(100, (acceptance ?? 0) * 100)}%`;
        text('spec-tokens-per-verification', tokensPerVerification === null ? '—' : tokensPerVerification.toFixed(2));
        text('spec-effectiveness-status', acceptance === null && tokensPerVerification === null ? 'Awaiting activity' : 'Accumulated verification results');
        const counts = [`${count(accepted)} accepted`];
        if (drafted !== null) counts.push(`${count(drafted)} drafted`);
        if (steps !== null) counts.push(`${count(steps)} verification steps`);
        text('spec-effectiveness-counts', counts.join(' · '));
    }
    const cardCount = Number(cacheAvailable) + Number(specAvailable);
    const grid = document.getElementById('efficiency-grid');
    if (grid) grid.dataset.cardCount = String(cardCount);
    show('inference-efficiency', cardCount > 0);
}
