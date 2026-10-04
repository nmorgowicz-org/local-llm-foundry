// Shared by Playwright coverage and the Puppeteer capture scenario. No disk
// fixtures: every migration response, including queue success, is synthetic.
export const MIGRATION_PATH = /^\/api\/app-home-migration(?:\/|$)/;
export const MIGRATION_ROUTE = /\/api\/app-home-migration(?:[/?]|$)/;
export const CONFIRMATION = 'MIGRATE TO LOCAL LLM FOUNDRY';

export function migrationPlan() {
    const mib = 1024 * 1024;
    const entry = (relative_path, className, bytes, kind = 'file') => ({
        relative_path, class: className, kind, bytes, modified_unix_seconds: 1700000000,
    });
    const entries = [
        entry('chat.db', 'critical', 512 * mib),
        entry('presets.json', 'unknown', 3 * mib),
        entry('settings.json', 'unknown', 0),
        entry('custom-state.bin', 'unknown', 256 * mib),
        entry('models/local.gguf', 'model_retained', 4096 * mib),
        entry('models', 'model_retained', 0, 'directory'),
        entry('logs/app.log', 'recreatable', 64 * mib),
        entry('logs', 'recreatable', 0, 'directory'),
    ];
    return {
        schema_version: 1,
        plan_id: 'fixture-plan-771-mib',
        source: '/fixture/legacy-home',
        destination: '/fixture/foundry-home',
        entries,
        retained_entries: entries.filter(row => ['model_retained', 'recreatable'].includes(row.class))
            .map(row => row.relative_path).sort(),
        required_copy_bytes: 771 * mib,
        total_seen_bytes: entries.reduce((sum, row) => sum + row.bytes, 0),
    };
}

export function createMigrationMock({ plan = migrationPlan(), queued = false, migrationRequired = true } = {}) {
    const state = {
        plan, queued, previewError: null, queueError: null, calls: [], unexpected: [],
    };
    const reply = (status, body) => ({ status, body });
    state.respond = (pathname, method, bodyText) => {
        if (!MIGRATION_PATH.test(pathname)) return null;
        const call = { pathname, method, bodyText };
        state.calls.push(call);
        if (pathname === '/api/app-home-migration/status' && method === 'GET') {
            return reply(200, {
                ok: true, state: state.queued ? 'migration_queued' : 'legacy_active',
                migration_required: migrationRequired, active_root: state.plan.source,
                legacy_root: state.plan.source, canonical_root: state.plan.destination,
            });
        }
        if (pathname === '/api/app-home-migration/preview' && method === 'GET') {
            if (state.previewError) return reply(503, { ok: false, error: state.previewError });
            return reply(200, { ok: true, state: 'legacy_active', plan: state.plan });
        }
        if (pathname === '/api/app-home-migration/queue' && method === 'POST') {
            let body;
            try { body = JSON.parse(bodyText || ''); } catch { body = null; }
            if (body?.plan_id !== state.plan.plan_id || body?.confirmation !== CONFIRMATION) {
                state.unexpected.push(call);
                return reply(400, { ok: false, error: 'Unexpected fixture queue payload' });
            }
            if (state.queueError) return reply(409, { ok: false, error: state.queueError });
            state.queued = true;
            return reply(200, {
                ok: true, restart_required: true,
                request: {
                    schema_version: 1, plan_id: state.plan.plan_id,
                    source: state.plan.source, destination: state.plan.destination,
                    requested_unix_seconds: 1700000000,
                },
            });
        }
        // Rollback, cleanup, unknown endpoints, and wrong methods MUST NOT
        // fall through to a live migration API, even when the UI changes.
        state.unexpected.push(call);
        return reply(501, { ok: false, error: 'Unmocked migration request blocked' });
    };
    return state;
}
