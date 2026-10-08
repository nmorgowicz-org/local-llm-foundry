// Shared application runner used by both the canonical and legacy entrypoints.
// GUI subsystem selection remains on each thin binary wrapper.
// We manage the console at runtime:
// - If launched from a terminal: attach to that console (AttachConsole).
// - If launched interactively (double-click / shortcut): allocate a new console
//   (AllocConsole) so logs are visible.
// - If --headless / --agent: no console; redirect to log file.
// This lets users see startup info and panics by default, while still supporting
// fully silent (headless) operation.
use anyhow::{Context, Result};
use clap::Parser;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::app_migration::RootState;
use crate::{
    agent, app_migration, cli, config, gpu, hf, inference, llama, presets, state, system, web,
};

/// Windows-only: attach to the parent console if running from a terminal.
/// Safe to call from Explorer; it will simply fail to attach in that case.
#[cfg(windows)]
fn attach_parent_console() {
    unsafe {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn AttachConsole(dwProcessId: u32) -> i32;
        }
        const ATTACH_PARENT_PROCESS: u32 = 0xFFFF_FFFF;
        let _ = AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

/// Windows-only: allocate a console window and ensure stdio routes there.
///
/// Behavior:
/// - If a console already exists (parent or prior), no-op.
/// - If is_interactive is false (e.g., headless/agent), no console (file logging).
/// - Otherwise: AllocConsole() so logs are visible when launched interactively
///   (double-click / Start menu).
///
/// Best-effort: if allocation fails, startup continues and a file-based fallback
/// is used by redirect_output_to_log_if_no_console().
#[cfg(windows)]
fn maybe_alloc_console(is_interactive: bool) -> bool {
    if !is_interactive {
        return false;
    }

    // If we already have a console (parent or otherwise), nothing to do.
    unsafe {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetConsoleWindow() -> *mut core::ffi::c_void;
        }
        if !GetConsoleWindow().is_null() {
            return true;
        }
    }

    unsafe {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn AllocConsole() -> i32;
        }
        if AllocConsole() == 0 {
            // Allocation failed; continue without console, log-file fallback applies.
            return false;
        }

        // Rust's stdio on Windows (GUI subsystem) is lazily initialized from
        // GetStdHandle(). After AllocConsole(), that now returns the console
        // handles, so subsequent println!/eprintln! go to the new console.
        // No extra handle wiring needed.
        true
    }
}

/// Windows-only: if no console is present, redirect stdout+stderr to a log file.
/// This is the fallback when:
/// - launched interactively with --headless/--agent, OR
/// - console allocation failed.
#[cfg(windows)]
fn redirect_output_to_log_if_no_console(log_dir: &std::path::Path) {
    unsafe {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetConsoleWindow() -> *mut core::ffi::c_void;
            fn SetStdHandle(n_std_handle: u32, h_handle: *mut core::ffi::c_void) -> i32;
        }
        // Have a console → keep printing there (live output, like a normal CLI).
        if !GetConsoleWindow().is_null() {
            return;
        }
        // No console → log under the already-selected application root.
        if std::fs::create_dir_all(log_dir).is_err() {
            return;
        }
        let path = log_dir.join("local-llm-foundry.log");
        let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        else {
            return;
        };
        use std::os::windows::io::AsRawHandle;
        const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
        const STD_ERROR_HANDLE: u32 = -12i32 as u32;
        let handle = file.as_raw_handle();
        SetStdHandle(STD_OUTPUT_HANDLE, handle);
        SetStdHandle(STD_ERROR_HANDLE, handle);
        // Keep the file handle open for the lifetime of the process.
        std::mem::forget(file);
        eprintln!(
            "[log] llama-monitor started (no console; logging to {})",
            path.display()
        );
    }
}

/// No-op on non-Windows: on those platforms we don't auto-allocate consoles.
#[cfg(not(windows))]
#[allow(dead_code)]
fn maybe_alloc_console() -> bool {
    false
}

use crate::chat_storage::ChatStorage;
use crate::config::{
    DashboardAuthConfig, TlsMode, clear_auth_config, harden_file_permissions, load_auth_config,
    save_auth_config,
};
use crate::web::auth::AuthManager;

const GPU_POLL_INTERVAL: Duration = Duration::from_millis(500); // fallback; dynamic interval reads ws_push_interval_ms
const SYSTEM_POLL_INTERVAL: Duration = Duration::from_secs(5);

pub fn run() -> Result<()> {
    // On Windows: console setup (before most logging).
    //
    // Steps:
    // 1) If launched from a terminal: attach to parent console (AttachConsole).
    // 2) Parse args with clap (so --headless/--agent are canonical and future-safe).
    // 3) If not headless/agent (interactive):
    //    - If no console (Explorer, Start menu, shortcut): AllocConsole so logs are visible.
    // 4) If still no console (headless/agent or AllocConsole failed):
    //    - redirect stdout/stderr to %APPDATA%\llama-monitor\logs\llama-monitor.log.
    //
    // This matches: "don't hide the terminal unless it clearly makes sense."
    #[cfg(windows)]
    attach_parent_console();

    let mut args = cli::AppArgs::parse();
    let migration_command = args.app_home_migration_status
        || args.app_home_migration_preview
        || args.app_home_migrate
        || args.app_home_rollback_preview
        || args.app_home_rollback
        || args.app_home_cleanup;
    let test_roots = args
        .migration_test_root
        .as_deref()
        .map(app_migration::disposable_roots)
        .transpose()?;
    let migration_scope = args.config_dir.is_none() || test_roots.is_some();
    let mut default_inspection = if migration_scope {
        Some(match test_roots.as_ref() {
            Some(roots) => app_migration::inspect_application_roots(roots)?,
            None => app_migration::inspect_default_roots()?,
        })
    } else {
        None
    };

    if migration_command {
        let Some(inspection) = default_inspection else {
            return Err(anyhow::anyhow!(
                "application-home migration commands require the default roots; --config-dir is explicit and cannot be reinterpreted"
            ));
        };
        if args.app_home_migration_status {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "state": inspection.state,
                    "canonical_root": inspection.canonical_root,
                    "legacy_root": inspection.legacy_root,
                    "active_root": inspection.active_root,
                }))?
            );
            return Ok(());
        }
        if args.app_home_rollback_preview || args.app_home_rollback {
            if inspection.state != RootState::RollbackAvailable {
                return Err(anyhow::anyhow!(
                    "application-root rollback requires a verified migrated canonical root (state: {:?})",
                    inspection.state
                ));
            }
            let plan = app_migration::plan_application_home_rollback(
                &inspection.canonical_root,
                &inspection.legacy_root,
            )?;
            if args.app_home_rollback_preview {
                println!("{}", serde_json::to_string_pretty(&plan)?);
                return Ok(());
            }
            if args.confirm.as_deref() != Some("ROLL BACK TO LLAMA MONITOR") {
                return Err(anyhow::anyhow!(
                    "--app-home-rollback requires --confirm 'ROLL BACK TO LLAMA MONITOR'"
                ));
            }
            app_migration::execute_application_home_rollback(&plan)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
            return Ok(());
        }
        if args.app_home_cleanup {
            if inspection.state != RootState::RollbackAvailable {
                return Err(anyhow::anyhow!(
                    "application-root cleanup requires a verified migrated canonical root (state: {:?})",
                    inspection.state
                ));
            }
            let plan = app_migration::plan_application_home_cleanup(
                &inspection.canonical_root,
                &inspection.legacy_root,
            )?;
            if args.confirm.as_deref() != Some("DELETE LEGACY ROOT AFTER VERIFIED MIGRATION") {
                return Err(anyhow::anyhow!(
                    "--app-home-cleanup requires --confirm 'DELETE LEGACY ROOT AFTER VERIFIED MIGRATION'"
                ));
            }
            app_migration::queue_application_home_cleanup(&plan)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
            return Ok(());
        }
        let (source, destination) = match inspection.state {
            RootState::LegacyActive => (inspection.legacy_root, inspection.canonical_root),
            RootState::Fresh | RootState::NewActive => {
                return Err(anyhow::anyhow!(
                    "no legacy application root requires migration (state: {:?})",
                    inspection.state
                ));
            }
            RootState::Conflict => {
                return Err(anyhow::anyhow!(
                    "application roots conflict; no migration plan will be guessed"
                ));
            }
            state => {
                return Err(anyhow::anyhow!(
                    "application-root migration state {:?} requires recovery handling",
                    state
                ));
            }
        };
        let plan = app_migration::plan_application_home(&source, &destination)?;
        if args.app_home_migration_preview {
            println!("{}", serde_json::to_string_pretty(&plan)?);
            return Ok(());
        }
        if args.confirm.as_deref() != Some("MIGRATE TO LOCAL LLM FOUNDRY") {
            return Err(anyhow::anyhow!(
                "--app-home-migrate requires --confirm 'MIGRATE TO LOCAL LLM FOUNDRY'"
            ));
        }
        let receipt = app_migration::execute_application_home(&plan)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }

    // A rollback is queued by the authenticated migration center and consumed
    // before normal root selection, so no live process ever switches roots
    // halfway through initialization.
    if migration_scope
        && let Some(request) = match test_roots.as_ref() {
            Some(roots) => app_migration::load_rollback_request_for_root(&roots.canonical)?,
            None => app_migration::load_rollback_request()?,
        }
    {
        app_migration::execute_queued_rollback(&request)
            .context("queued application-home rollback failed")?;
    }
    if migration_scope
        && let Some(request) = match test_roots.as_ref() {
            Some(roots) => app_migration::load_cleanup_request_for_root(&roots.canonical)?,
            None => app_migration::load_cleanup_request()?,
        }
    {
        app_migration::execute_queued_cleanup(&request)
            .context("queued legacy-root cleanup failed")?;
    }

    // Root inspection is pure and must happen before AppConfig can create
    // tokens, load protected config, initialize encryption, or create model
    // directories. Legacy-only installs remain on their existing root until
    // the explicit migration flow is implemented; both-root conflicts stop.
    if migration_scope {
        let inspection = default_inspection
            .take()
            .expect("default inspection captured above");
        let queued_request = if matches!(
            inspection.state,
            RootState::MigrationQueued | RootState::Conflict
        ) {
            match test_roots.as_ref() {
                Some(roots) => app_migration::load_migration_request_from_parent(
                    roots
                        .canonical
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new(".")),
                )?,
                None => app_migration::load_migration_request()?,
            }
        } else {
            None
        };
        // A Conflict with a queued migration request means a previous run
        // crashed mid-copy: the destination has partial state and no receipt.
        // The plan/executor resume from the journal instead of aborting
        // startup forever. A Conflict without a request still falls through
        // to the conflict error in the state match below.
        if inspection.state == RootState::MigrationQueued || queued_request.is_some() {
            let request = queued_request
                .ok_or_else(|| anyhow::anyhow!("migration queue marker is missing"))?;
            let plan = app_migration::plan_application_home(&request.source, &request.destination)?;
            if plan.plan_id != request.plan_id {
                return Err(anyhow::anyhow!(
                    "queued migration preview is stale; generate a new preview"
                ));
            }
            app_migration::execute_application_home(&plan)
                .context("queued application-home migration failed")?;
            default_inspection = Some(match test_roots.as_ref() {
                Some(roots) => app_migration::inspect_application_roots(roots)?,
                None => app_migration::inspect_default_roots()?,
            });
        } else {
            default_inspection = Some(inspection);
        }
        let inspection = default_inspection
            .as_ref()
            .expect("inspection restored after queued migration");
        let selected = match inspection.state {
            RootState::Fresh | RootState::NewActive => inspection.canonical_root.clone(),
            RootState::LegacyActive => inspection.legacy_root.clone(),
            RootState::Conflict => {
                return Err(anyhow::anyhow!(
                    "application roots conflict: both {} and {} contain state; run the explicit migration flow",
                    inspection.legacy_root.display(),
                    inspection.canonical_root.display()
                ));
            }
            RootState::CustomConfig
            | RootState::BothIdentical
            | RootState::MigrationQueued
            | RootState::Migrating
            | RootState::MigrationFailed
            | RootState::RollbackAvailable => inspection.canonical_root.clone(),
        };
        // A disposable migration root is only the migration fixture. Keep an
        // explicitly supplied config directory authoritative so startup
        // tokens and encrypted state are generated there, rather than inside
        // the synthetic application roots.
        if test_roots.is_none() {
            args.config_dir = Some(selected);
        }
    }

    // Establish the encryption key before AppConfig loads protected config
    // files or initializes token stores.
    if !args.clear_auth_config
        && let Some(config_dir) = args.config_dir.as_deref()
    {
        config::init_encryption_key(config_dir)
            .map_err(|_| anyhow::anyhow!("encryption environment aliases are conflicting"))?;
    }

    #[cfg(windows)]
    {
        let is_interactive = !(args.headless || args.agent);
        let _console_allocated = maybe_alloc_console(is_interactive);
    }
    #[cfg(windows)]
    redirect_output_to_log_if_no_console(
        &crate::paths::AppPaths::from_root(
            args.config_dir
                .clone()
                .expect("startup root selected before logging"),
        )
        .logs_dir(),
    );
    let app_config = Arc::new(if args.clear_auth_config {
        config::AppConfig::from_args_pure(args.clone())
    } else {
        config::AppConfig::from_args(args.clone())
    });
    // Install the selected root for helpers that run outside the request
    // context.  This is intentionally after pure root inspection and before
    // any subsystem initializes stores or certificates.
    crate::paths::AppPaths::set_active_root(&app_config.config_dir);

    if args.clear_auth_config {
        match clear_auth_config(&app_config.config_dir) {
            Ok(true) => {
                println!(
                    "[info] Cleared dashboard auth config at {}",
                    app_config.auth_config_file.display()
                );
            }
            Ok(false) => {
                println!(
                    "[info] No dashboard auth config found at {}",
                    app_config.auth_config_file.display()
                );
            }
            Err(err) => {
                eprintln!(
                    "[error] Failed to clear dashboard auth config at {}: {err}",
                    app_config.auth_config_file.display()
                );
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    // Measured speculative-decoding verdicts live beside the managed runtimes, so the
    // store has to learn the resolved config directory before any capability snapshot is
    // generated.
    inference::rapid_mlx::spec_decode_store::set_store_root(&app_config.config_dir);
    hf::mtp_pin_cache::init_pin_cache(&app_config.config_dir);
    inference::rapid_mlx::sidecar_inventory::init_sidecar_root(&app_config.config_dir);
    inference::rapid_mlx::model_resolver::init_template_overlay_root(&app_config.config_dir);

    if let Some(report) = args.ingest_spec_decode_report.clone() {
        return ingest_spec_decode_report(&report);
    }

    // Harden permissions on secret files (Unix: 0600)
    harden_file_permissions(&app_config.ui_settings_file);
    harden_file_permissions(&app_config.sessions_file);
    harden_file_permissions(&app_config.ssh_known_hosts_file);
    harden_file_permissions(&app_config.config_dir.join("db-admin-token"));
    harden_file_permissions(&app_config.config_dir.join("api-token"));
    harden_file_permissions(&app_config.config_dir.join("tls-config.json"));
    harden_file_permissions(&app_config.config_dir.join("encryption-key"));
    harden_file_permissions(&app_config.auth_config_file);

    if args.agent {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(8 * 1024 * 1024)
            .build()?;
        return runtime.block_on(agent::run_agent_server(app_config));
    }

    // Load presets from disk (or defaults)
    let initial_presets = presets::load_presets(&app_config.presets_file);
    println!(
        "[info] Loaded {} presets from {}",
        initial_presets.len(),
        app_config.presets_file.display()
    );

    // Load GPU environment config
    let mut gpu_env = gpu::env::load_gpu_env(&app_config.gpu_env_file);

    // CLI overrides take precedence
    if let Some(ref arch) = app_config.gpu_arch_override {
        gpu_env.arch = arch.clone();
    }
    if let Some(ref devices) = app_config.gpu_devices_override {
        gpu_env.devices = devices.clone();
    }

    // Auto-detect GPUs and log results
    if let Some(detected) = gpu::env::detect_gpus() {
        println!(
            "[info] Detected {}x {} GPU(s)",
            detected.count, detected.arch
        );
        // If arch is "auto" and devices is empty, suggest detected values
        if gpu_env.arch == "auto" && gpu_env.devices.is_empty() {
            gpu_env.devices = gpu::env::device_list_for_count(detected.count);
        }
    }

    println!(
        "[info] GPU env: arch={}, devices={}",
        gpu_env.arch,
        if gpu_env.devices.is_empty() {
            "all"
        } else {
            &gpu_env.devices
        }
    );

    // Load UI settings from disk (or defaults)
    let ui_settings = state::load_ui_settings(&app_config.ui_settings_file);

    // Load sessions from disk (or defaults)
    let _sessions = state::load_sessions(&app_config.sessions_file);

    // Open chat database
    let chat_db_path = app_config.config_dir.join("chat.db");
    let chat_storage = Arc::new(ChatStorage::open(&chat_db_path).context("opening chat.db")?);

    let model_library_root = app_config
        .models_dir
        .clone()
        .unwrap_or_else(|| app_config.default_models_dir.clone());
    crate::models::library::ensure_model_tree(
        &model_library_root,
        cfg!(target_os = "macos") && cfg!(target_arch = "aarch64"),
    );

    // Migrate from legacy chat-tabs.json (best-effort)
    let legacy = app_config.config_dir.join("chat-tabs.json");
    if let Err(e) = chat_storage.migrate_from_legacy(&legacy) {
        eprintln!("[warn] chat legacy migration failed: {e}");
    }

    let state = state::AppState::new(
        initial_presets,
        state::AppPaths {
            presets_path: app_config.presets_file.clone(),
            templates_path: app_config.templates_file.clone(),
            models_dir: app_config
                .models_dir
                .clone()
                .or(Some(app_config.default_models_dir.clone())),
            gpu_env_path: app_config.gpu_env_file.clone(),
            ui_settings_path: app_config.ui_settings_file.clone(),
            sessions_path: app_config.sessions_file.clone(),
            model_tags_path: app_config.config_dir.join("model-tags.json"),
        },
        gpu_env,
        ui_settings,
        chat_storage,
        app_config.tls_config.clone(),
    );

    if let Some(ref dir) = app_config.models_dir {
        match state.discovered_models.lock() {
            Ok(models) => {
                let count = models.len();
                println!("[info] Discovered {count} models in {}", dir.display());
            }
            Err(e) => {
                eprintln!("[error] Failed to acquire discovered_models lock: {e}");
            }
        }
    }

    // Detect and start GPU poller
    let backend = gpu::detect_backend(&app_config.gpu_backend);
    {
        let s = state.clone();
        thread::spawn(move || {
            loop {
                if s.active_session_uses_local_metrics() {
                    match backend.read_metrics() {
                        Ok(m) => {
                            if let Ok(mut gpu_lock) = s.gpu_metrics.lock() {
                                *gpu_lock = m;
                            } else {
                                eprintln!("[error] Failed to acquire gpu lock");
                            }
                            // Feed CPU/SoC temp from GPU backend (Apple only)
                            if let Some(t) = backend.cpu_temp()
                                && let Ok(mut sys_lock) = s.system_metrics.lock()
                            {
                                sys_lock.cpu_temp = t;
                                sys_lock.cpu_temp_available = true;
                            }
                        }
                        Err(e) => eprintln!("[error] GPU metrics: {e}"),
                    };
                }
                // POWER OPT: track the WS push interval so we don't poll GPU faster
                // than we send data to the browser. Minimum 200ms to avoid hammering
                // the GPU driver.
                // T-045: sleep guard: when asleep, slow GPU polling using config interval
                let final_ms = {
                    let settings = match s.ui_settings.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            thread::sleep(GPU_POLL_INTERVAL);
                            continue;
                        }
                    };
                    let base = (settings.ws_push_interval_ms.max(200) / 2).max(200);
                    let mode = s.sleep_mode.load(std::sync::atomic::Ordering::Relaxed);
                    if mode >= 1 {
                        // T-045: use slow GPU interval while in low-power mode
                        if let Ok(cfg) = s.sleep_mode_config.lock() {
                            let slow_ms = cfg.sleep_gpu_interval_secs.max(1) * 1000;
                            slow_ms.max(base)
                        } else {
                            base
                        }
                    } else {
                        base
                    }
                };
                thread::sleep(Duration::from_millis(final_ms));
            }
        });
    }

    // System metrics poller
    {
        let s = state.clone();
        thread::spawn(move || {
            loop {
                if s.active_session_uses_local_metrics() {
                    let mut metrics = system::get_system_metrics();
                    if let Ok(mut sys_lock) = s.system_metrics.lock() {
                        // Preserve CPU temp if the GPU backend already provided one
                        // (e.g. Apple mactop) since get_system_metrics() can't read it.
                        if !metrics.cpu_temp_available && sys_lock.cpu_temp_available {
                            metrics.cpu_temp = sys_lock.cpu_temp;
                            metrics.cpu_temp_available = true;
                        }
                        *sys_lock = metrics;
                    } else {
                        eprintln!("[error] Failed to acquire system_metrics lock");
                    }
                }
                // T-046: when in low-power mode, slow system-metrics polling using config interval
                let mode = s.sleep_mode.load(std::sync::atomic::Ordering::Relaxed);
                let interval = if mode >= 1 {
                    if let Ok(cfg) = s.sleep_mode_config.lock() {
                        let slow_secs = cfg.sleep_sys_interval_secs.max(1);
                        Duration::from_secs(slow_secs)
                    } else {
                        SYSTEM_POLL_INTERVAL
                    }
                } else {
                    SYSTEM_POLL_INTERVAL
                };
                std::thread::sleep(interval);
            }
        });
    }

    let port = app_config.port;
    let host = args.host.clone();

    let basic_auth = match args.basic_auth.as_deref() {
        Some(spec) => match AuthManager::parse_credentials(spec) {
            Some(creds) => Some(creds),
            None => {
                eprintln!("[error] Invalid --basic-auth format. Expected: user:password");
                std::process::exit(1);
            }
        },
        None => None,
    };
    let form_auth = match args.form_auth.as_deref() {
        Some(spec) => match AuthManager::parse_credentials(spec) {
            Some(creds) => Some(creds),
            None => {
                eprintln!("[error] Invalid --form-auth format. Expected: user:password");
                std::process::exit(1);
            }
        },
        None => None,
    };

    // Apply CLI TLS overrides to tls_config
    let mut tls_config = app_config.tls_config.clone();
    if args.tls {
        if args.tls_cert.is_none() || args.tls_key.is_none() {
            eprintln!("[error] --tls requires both --tls-cert and --tls-key");
            std::process::exit(1);
        }
        tls_config.mode = TlsMode::Custom;
        tls_config.custom_cert_path = args.tls_cert.clone();
        tls_config.custom_key_path = args.tls_key.clone();
    } else if args.tls_self_signed {
        tls_config.mode = TlsMode::SelfSigned;
    }

    // Update in-memory state with final TLSConfig
    state.set_tls_config(tls_config.clone());

    migrate_legacy_dashboard_auth(
        &app_config.config_dir,
        &app_config.auth_config_file,
        basic_auth.as_ref(),
        form_auth.as_ref(),
    );

    let auth_manager = if basic_auth.is_some() || form_auth.is_some() {
        AuthManager::new(basic_auth.clone(), form_auth.clone(), &tls_config.mode)
    } else {
        AuthManager::from_config(load_auth_config(&app_config.config_dir), &tls_config.mode)
    };

    let routes = web::build_routes(
        state.clone(),
        app_config.clone(),
        auth_manager.clone(),
        host.clone(),
    );

    // Warn when listening on all interfaces without TLS or auth
    if host == "0.0.0.0" && tls_config.mode == TlsMode::None && !auth_manager.has_any() {
        eprintln!(
            "[warn] Listening on all interfaces without TLS or authentication. \
            Anyone on your network can access the UI."
        );
    }

    if args.headless {
        println!("[info] Headless mode enabled (no tray, no desktop UI)");
    } else if args.no_tray {
        println!("[info] Tray disabled via --no-tray");
    }

    // Build tokio runtime for async tasks (warp server, pollers, etc.)
    // The runtime runs in background threads; the main thread is reserved
    // for the system tray, which macOS requires to be on the main thread.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(8 * 1024 * 1024)
        .build()?;

    // Llama metrics poller. It remains idle until the user starts a preset or
    // explicitly attaches to an endpoint.
    {
        let s = state.clone();
        let interval = app_config.llama_poll_interval;
        runtime.spawn(llama::poller::llama_metrics_poller(s, interval));
    }

    // Remote host metrics poller. It is gated by the same user action signal,
    // so app startup never probes saved remote endpoints automatically.
    {
        let s = state.clone();
        let app_config = app_config.clone();
        runtime.spawn(agent::remote_agent_poller(s, app_config));
    }

    // T-052: Auto-sleep background task.
    // Monitors last activity, WS connections, and streaming to auto-sleep when idle.
    {
        let s = state.clone();
        runtime.spawn(async move {
            loop {
                // Check every 30 seconds
                tokio::time::sleep(Duration::from_secs(30)).await;

                // Skip if already in low-power mode
                let mode = s.sleep_mode.load(std::sync::atomic::Ordering::Relaxed);
                if mode > 0 {
                    continue;
                }

                // T-056: do not auto-sleep while streaming is active
                let streaming_active = {
                    let llama = s.llama_metrics.lock().unwrap();
                    llama.generation_tokens_per_sec > 0.0
                };
                if streaming_active {
                    continue;
                }

                let cfg = match s.sleep_mode_config.lock() {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // T-052: Check idle time
                if let Some(idle_secs) = cfg.auto_sleep_idle_secs
                    && idle_secs > 0
                {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let last = s
                        .last_activity_at
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let elapsed = now.saturating_sub(last);

                    if elapsed >= idle_secs {
                        eprintln!(
                            "[sleep] auto-sleep: idle {}s >= threshold {}s",
                            elapsed, idle_secs
                        );
                        // Auto-sleep due to inactivity (not user-triggered)
                        s.sleep_mode_manual
                            .store(false, std::sync::atomic::Ordering::Relaxed);
                        s.sleep_mode.store(2, std::sync::atomic::Ordering::Relaxed);
                        s.sleep_notify.notify_waiters();
                        drop(cfg);
                        continue;
                    }
                }

                drop(cfg);
            }
        });
    }

    // Sessions persistence timer
    {
        let state = state.clone();
        let sessions_file = app_config.sessions_file.clone();
        runtime.spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
                if let Err(e) = state::save_sessions(&sessions_file, &state.get_sessions()) {
                    eprintln!("[error] Failed to save sessions: {}", e);
                }
            }
        });
    }

    // Hourly database maintenance: WAL checkpoint, ANALYZE, rolling backup (keep 24h)
    {
        let chat_storage = state.chat_storage.clone();
        let config_dir = app_config.config_dir.clone();
        runtime.spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;

                if let Err(e) = chat_storage.checkpoint() {
                    eprintln!("[error] WAL checkpoint failed: {}", e);
                }

                if let Err(e) = chat_storage.analyze() {
                    eprintln!("[error] ANALYZE failed: {}", e);
                }

                let auto_backup_dir = config_dir.join("backups").join("auto");
                if let Err(e) = std::fs::create_dir_all(&auto_backup_dir) {
                    eprintln!("[error] Failed to create auto backup directory: {}", e);
                    continue;
                }

                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis().to_string())
                    .unwrap_or_else(|_| "0".to_string());
                let backup_path = auto_backup_dir.join(format!("chat_auto_{}.db", timestamp));

                if let Err(e) = chat_storage.backup(&backup_path) {
                    eprintln!("[error] Hourly auto backup failed: {}", e);
                }

                // Keep the last 24 hourly backups
                if let Ok(entries) = std::fs::read_dir(&auto_backup_dir) {
                    let mut backups: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_name().to_string_lossy().starts_with("chat_auto_"))
                        .collect();
                    backups.sort_by_key(|e| e.path());
                    while backups.len() > 24 {
                        let old = backups.remove(0);
                        let _ = std::fs::remove_file(old.path());
                    }
                }
            }
        });
    }

    // Daily database backup: runs every 24 hours, keeps 7 days
    {
        let chat_storage = state.chat_storage.clone();
        let config_dir = app_config.config_dir.clone();
        runtime.spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(86400)).await;

                let daily_backup_dir = config_dir.join("backups").join("daily");
                if let Err(e) = std::fs::create_dir_all(&daily_backup_dir) {
                    eprintln!("[error] Failed to create daily backup directory: {}", e);
                    continue;
                }

                // Use local date (not UTC) so late-night backups match the user's calendar day.
                let date = {
                    use chrono::Datelike;
                    let local = chrono::Local::now();
                    let d = local.date_naive();
                    format!("{}{:02}{:02}", d.year(), d.month(), d.day())
                };
                let backup_path = daily_backup_dir.join(format!("chat_daily_{}.db", date));

                if let Err(e) = chat_storage.backup(&backup_path) {
                    eprintln!("[error] Daily backup failed: {}", e);
                }

                // Keep the last 7 daily backups
                if let Ok(entries) = std::fs::read_dir(&daily_backup_dir) {
                    let mut backups: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_name().to_string_lossy().starts_with("chat_daily_"))
                        .collect();
                    backups.sort_by_key(|e| e.path());
                    while backups.len() > 7 {
                        let old = backups.remove(0);
                        let _ = std::fs::remove_file(old.path());
                    }
                }
            }
        });
    }

    // ACME certificate renewal job (runs every 24 hours)
    {
        let state = state.clone();
        let config_dir = app_config.config_dir.clone();
        runtime.spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(86400)).await;

                let cfg = state.get_tls_config();
                if !crate::acme::should_renew(&cfg) {
                    continue;
                }

                match crate::acme::acme_renew_cert(&config_dir, &cfg) {
                    Ok(new_cfg) => {
                        eprintln!("[info] ACME renewal succeeded");
                        state.set_tls_config(new_cfg.clone());
                        if let Err(e) = crate::config::save_tls_config(&config_dir, &new_cfg) {
                            eprintln!(
                                "[error] Failed to save tls-config.json after renewal: {}",
                                e
                            );
                        }
                    }
                    Err(e) => {
                        eprintln!("[warn] ACME renewal failed: {}", e);
                    }
                }
            }
        });
    }

    // Warp server (HTTP or TLS depending on config)
    let tls_mode = tls_config.mode.clone();
    let tls_custom_cert = tls_config.custom_cert_path.clone();
    let tls_custom_key = tls_config.custom_key_path.clone();
    let tls_acme = tls_config.acme.clone();
    let tls_config_dir = app_config.config_dir.clone();
    let tls_host = host.clone();
    let tls_port = port;

    runtime.spawn(async move {
        let addr: std::net::SocketAddr = format!("{}:{}", tls_host, tls_port)
            .parse()
            .expect("Invalid host:port");

        match tls_mode {
            TlsMode::None => {
                println!(
                    "[info] Local LLM Foundry running on http://{}:{}",
                    tls_host, tls_port
                );
                warp::serve(routes).run(addr).await;
            }
            TlsMode::Acme => {
                // If ACME mode but no cert yet, start HTTP and log.
                if tls_acme.cert_path.is_none() || tls_acme.key_path.is_none() {
                    eprintln!(
                        "[info] ACME mode enabled but no certificate; start with TLS disabled \
                        until ACME request completes."
                    );
                    warp::serve(routes).run(addr).await;
                    return;
                }

                let cert_path = tls_acme.cert_path.clone().unwrap();
                let key_path = tls_acme.key_path.clone().unwrap();

                if !cert_path.exists() || !key_path.exists() {
                    eprintln!(
                        "[warn] ACME cert/key files not found; falling back to HTTP on http://{}:{}",
                        tls_host, tls_port
                    );
                    warp::serve(routes).run(addr).await;
                    return;
                }

                let tls_cfg = match build_tls_config(&cert_path, &key_path) {
                    Ok(cfg) => cfg,
                    Err(e) => {
                        eprintln!("[error] Failed to load ACME TLS config: {}", e);
                        eprintln!(
                            "[warn] Falling back to HTTP on http://{}:{}",
                            tls_host, tls_port
                        );
                        warp::serve(routes).run(addr).await;
                        return;
                    }
                };

                let tls_acceptor =
                    tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(tls_cfg));

                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[error] Failed to bind ACME TLS listener: {}", e);
                        return;
                    }
                };

                println!(
                    "[info] Local LLM Foundry running on https://{}:{} (ACME - {})",
                    tls_host,
                    tls_port,
                    if tls_acme.environment == "staging" {
                        "staging"
                    } else {
                        "production"
                    }
                );

                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[error] ACME TLS accept error: {}", e);
                            continue;
                        }
                    };

                    let acceptor = tls_acceptor.clone();
                    let routes_clone = routes.clone();
                    tokio::spawn(async move {
                        let tls_stream = match acceptor.accept(stream).await {
                            Ok(s) => s,
                            Err(e) => {
                                eprintln!("[error] ACME TLS handshake error: {}", e);
                                return;
                            }
                        };

                        let svc = warp::service(routes_clone);
                        let svc = hyper_util::service::TowerToHyperService::new(svc);
                        let io = hyper_util::rt::TokioIo::new(tls_stream);

                        if let Err(e) =
                            hyper_util::server::conn::auto::Builder::new(
                                hyper_util::rt::TokioExecutor::new(),
                            )
                            .http1()
                            .serve_connection_with_upgrades(io, svc)
                            .await
                        {
                            eprintln!("[error] ACME TLS connection error: {}", e);
                        }
                    });
                }
            }
            TlsMode::SelfSigned | TlsMode::Custom => {
                // Determine cert and key paths
                let (cert_path, key_path) = if matches!(tls_mode, TlsMode::SelfSigned) {
                    let cp = tls_config_dir.join("tls-server.pem");
                    let kp = tls_config_dir.join("tls-server.key");

                    if !cp.exists() || !kp.exists() {
                        let mut sans = vec!["localhost".to_string(), "127.0.0.1".to_string()];
                        if tls_host != "0.0.0.0"
                            && tls_host != "127.0.0.1"
                            && !tls_host.starts_with('[')
                        {
                            sans.push(tls_host.clone());
                        }
                        let cert = crate::certs::generate_self_signed(sans);
                        if let Err(e) = cert.save(&cp, &kp) {
                            eprintln!("[error] Failed to write self-signed cert: {}", e);
                            eprintln!(
                                "[warn] Falling back to HTTP on http://{}:{}",
                                tls_host, tls_port
                            );
                            warp::serve(routes).run(addr).await;
                            return;
                        }
                        println!(
                            "[info] Generated self-signed TLS certificate at {}",
                            cp.display()
                        );
                    }
                    (cp, kp)
                } else {
                    match (&tls_custom_cert, &tls_custom_key) {
                        (Some(cp), Some(kp)) => (cp.clone(), kp.clone()),
                        _ => {
                            eprintln!(
                                "[warn] TLS mode=custom but cert/key not set; falling back to HTTP"
                            );
                            warp::serve(routes).run(addr).await;
                            return;
                        }
                    }
                };

                if !cert_path.exists() || !key_path.exists() {
                    eprintln!("[warn] TLS certificate or key file not found; falling back to HTTP");
                    warp::serve(routes).run(addr).await;
                    return;
                }

                let tls_config = match build_tls_config(&cert_path, &key_path) {
                    Ok(cfg) => cfg,
                    Err(e) => {
                        eprintln!("[error] Failed to load TLS config: {}", e);
                        eprintln!(
                            "[warn] Falling back to HTTP on http://{}:{}",
                            tls_host, tls_port
                        );
                        warp::serve(routes).run(addr).await;
                        return;
                    }
                };

                let tls_acceptor = tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(tls_config));

                let listener = match tokio::net::TcpListener::bind(addr).await {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[error] Failed to bind TLS listener: {}", e);
                        return;
                    }
                };

                let tls_mode_label = if matches!(tls_mode, TlsMode::SelfSigned) {
                    "self-signed"
                } else {
                    "custom cert"
                };

                println!(
                    "[info] Local LLM Foundry running on https://{}:{} ({})",
                    tls_host, tls_port, tls_mode_label
                );

                // TLS server: replicate warp's Run pattern but with TLS-wrapped connections
                loop {
                    let (stream, _) = match listener.accept().await {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[error] TLS accept error: {}", e);
                            continue;
                        }
                    };

                    let acceptor = tls_acceptor.clone();
                    let routes_clone = routes.clone();
                    tokio::spawn(async move {
                        let tls_stream = match acceptor.accept(stream).await {
                            Ok(s) => s,
                            Err(e) => {
                                eprintln!("[error] TLS handshake error: {}", e);
                                return;
                            }
                        };

                        let svc = warp::service(routes_clone);
                        let svc = hyper_util::service::TowerToHyperService::new(svc);
                        let io = hyper_util::rt::TokioIo::new(tls_stream);

                        if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                            hyper_util::rt::TokioExecutor::new(),
                        )
                        .http1()
                        .serve_connection_with_upgrades(io, svc)
                        .await
                        {
                            eprintln!("[error] TLS connection error: {}", e);
                        }
                    });
                }
            }
        }
    });

    /// Build a rustls ServerConfig from PEM cert and key files.
    fn build_tls_config(
        cert_path: &std::path::Path,
        key_path: &std::path::Path,
    ) -> Result<rustls::ServerConfig, anyhow::Error> {
        use std::fs::File;
        use std::io::BufReader;

        let mut cert_reader = BufReader::new(File::open(cert_path)?);
        let certs: Vec<rustls::pki_types::CertificateDer> = rustls_pemfile::certs(&mut cert_reader)
            .filter_map(|c| c.ok())
            .collect();
        if certs.is_empty() {
            anyhow::bail!("No certificates found in {}", cert_path.display());
        }

        let mut key_reader = BufReader::new(File::open(key_path)?);
        let key: rustls::pki_types::PrivateKeyDer = rustls_pemfile::private_key(&mut key_reader)
            .map_err(|_| anyhow::anyhow!("Failed to read private key from {}", key_path.display()))?
            .ok_or_else(|| anyhow::anyhow!("No private key found in {}", key_path.display()))?;

        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)?;

        Ok(config)
    }

    // Clone shutdown-related fields before state is moved to tray
    let shutdown_chat_storage = state.chat_storage.clone();
    let shutdown_sessions_path = state.sessions_path.clone();
    let shutdown_state = state.clone();

    // Graceful shutdown handler. This MUST be registered before the tray block below: that block
    // never returns (the tray loop or `park_forever`), so a handler placed after it was
    // unreachable and Ctrl+C killed the app without stopping the inference backend.
    {
        let chat_storage = shutdown_chat_storage;
        let sessions_path = shutdown_sessions_path;
        let state = shutdown_state.clone();
        runtime.spawn(async move {
            // Wait for shutdown signal (platform-specific)
            #[cfg(unix)]
            {
                let mut sigint =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
                        .unwrap();
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .unwrap();

                tokio::select! {
                    _ = sigint.recv() => {},
                    _ = sigterm.recv() => {},
                }
            }

            #[cfg(windows)]
            {
                // On Windows, Ctrl+C is handled by the console handler;
                // we also listen for Ctrl+Break via console control events.
                // For now, block until a console event triggers shutdown.
                tokio::signal::ctrl_c().await.ok();
            }

            println!("\n[info] Shutdown signal received, finalizing...");

            // The backend runs in its own process group, so the terminal's Ctrl+C never
            // reaches it. Stop it here or it outlives the app and keeps its port.
            match tokio::time::timeout(
                std::time::Duration::from_secs(15),
                crate::llama::server::stop_server(&state),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("[warn] Failed to stop inference backend: {}", e),
                Err(_) => eprintln!("[warn] Timed out stopping inference backend"),
            }

            // Checkpoint WAL
            if let Err(e) = chat_storage.checkpoint() {
                eprintln!("[warn] Final checkpoint failed: {}", e);
            }

            // Save sessions
            if let Err(e) = state::save_sessions(&sessions_path, &state.get_sessions()) {
                eprintln!("[warn] Final session save failed: {}", e);
            }

            println!("[info] Shutdown complete");
            std::process::exit(0);
        });
    }

    // Run tray on the main thread when a desktop session is available.
    // Headless Linux servers still keep the web UI/API running.
    #[cfg(feature = "native-tray")]
    {
        if should_start_tray(&args) {
            match crate::tray::run_tray(state, port, app_config.config_dir.clone()) {
                Ok(()) => {
                    // A normal tray-loop return means the user selected Quit.
                    // Do not leave the API server parked alive with only its
                    // tray icon gone. Stop the backend first: it has its own
                    // process group and would otherwise outlive the app.
                    let _ = runtime.block_on(tokio::time::timeout(
                        std::time::Duration::from_secs(15),
                        crate::llama::server::stop_server(&shutdown_state),
                    ));
                    std::process::exit(0);
                }
                Err(e) => {
                    eprintln!("[warn] Tray unavailable: {e}");
                    eprintln!("[info] Continuing in headless mode with web/API server");
                }
            }
        } else {
            println!("[info] Tray disabled (no graphical session)");
            park_forever();
        }
    }

    #[cfg(not(feature = "native-tray"))]
    {
        let _ = state;
        println!("[info] Tray disabled in this build");
    }

    // Park main thread (tray or headless)
    #[cfg(feature = "native-tray")]
    {
        park_forever();
    }

    #[cfg(not(feature = "native-tray"))]
    {
        park_forever();
    }
}

/// Record a speculative-decoding requalification report against the installed
/// Rapid-MLX runtime, then exit.
///
/// The measurement is keyed by a capability snapshot's fingerprint, so this discovers
/// and probes the runtime exactly as the app would rather than trusting the report's own
/// version string alone. That is also what makes the mismatch check meaningful: the
/// report is refused unless it describes the build actually installed here.
fn ingest_spec_decode_report(report: &std::path::Path) -> Result<()> {
    use inference::rapid_mlx::capabilities;
    use inference::rapid_mlx::spec_decode_store;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let snapshot = runtime.block_on(capabilities::generate_snapshot_from_discovery())?;
    let store = spec_decode_store::process_store();
    let verdict = store.ingest_requalification_report(&snapshot, report)?;

    println!(
        "[info] Recorded {outcome:?} for Rapid-MLX {version} in {store_path}",
        outcome = verdict.outcome,
        version = verdict.rapid_mlx_version,
        store_path = store.path().display(),
    );
    println!(
        "[info] Gates run: {gates}; model: {model}; tool parser: {tools}",
        gates = verdict.gates_run.join(", "),
        model = verdict.model,
        tools = verdict.tool_call_parser.as_deref().unwrap_or("none"),
    );
    println!(
        "[info] Speculative decoding on this install is now: {qualification:?}",
        qualification = verdict.qualification(),
    );
    Ok(())
}

fn migrate_legacy_dashboard_auth(
    config_dir: &std::path::Path,
    auth_config_file: &std::path::Path,
    basic_auth: Option<&crate::web::auth::AuthCredentials>,
    form_auth: Option<&crate::web::auth::AuthCredentials>,
) {
    if auth_config_file.exists() {
        return;
    }

    let Some(creds) = basic_auth.or(form_auth) else {
        return;
    };

    if let (Some(basic), Some(form)) = (basic_auth, form_auth)
        && (basic.username != form.username || basic.password != form.password)
    {
        eprintln!(
            "[warn] Skipping auth-config migration because --basic-auth and --form-auth use different credentials."
        );
        return;
    }

    let Some(password_hash) = AuthManager::hash_password(&creds.password) else {
        eprintln!(
            "[warn] Failed to hash dashboard auth during migration; skipping auth-config migration."
        );
        return;
    };

    let cfg = DashboardAuthConfig {
        basic_enabled: basic_auth.is_some(),
        form_enabled: form_auth.is_some(),
        username: creds.username.clone(),
        password_hash,
    };

    if let Err(err) = save_auth_config(config_dir, &cfg) {
        eprintln!("[warn] Failed to migrate dashboard auth into auth-config.json: {err}");
    } else {
        eprintln!("[config] Migrated dashboard auth into auth-config.json for future builds.");
    }
}

#[cfg(target_os = "linux")]
pub fn should_start_tray(args: &cli::AppArgs) -> bool {
    if args.headless || args.no_tray {
        return false;
    }
    std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some()
}

#[cfg(not(target_os = "linux"))]
#[cfg(feature = "native-tray")]
pub fn should_start_tray(args: &cli::AppArgs) -> bool {
    if args.headless || args.no_tray {
        return false;
    }
    true
}

#[cfg(not(feature = "native-tray"))]
pub fn should_start_tray(_args: &cli::AppArgs) -> bool {
    false
}

fn park_forever() -> ! {
    loop {
        std::thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_start_tray_flag_combinations_disables_tray() {
        let test_cases = [
            (true, false, false), // headless only
            (false, true, false), // no_tray only
            (true, true, false),  // both flags
        ];

        for (headless, no_tray, expected) in test_cases {
            let args = cli::AppArgs {
                port: 7778,
                gpu_backend: "auto".to_string(),
                models_dir: None,
                gpu_arch: None,
                gpu_devices: None,
                llama_poll_interval: 1,
                llama_server_path: None,
                llama_server_cwd: None,
                presets_file: None,
                sessions_file: None,
                config_dir: None,
                migration_test_root: None,
                headless,
                no_tray,
                agent: false,
                agent_host: "127.0.0.1".to_string(),
                host: "127.0.0.1".to_string(),
                basic_auth: None,
                form_auth: None,
                ingest_spec_decode_report: None,
                clear_auth_config: false,
                app_home_migration_status: false,
                app_home_migration_preview: false,
                app_home_migrate: false,
                app_home_rollback_preview: false,
                app_home_rollback: false,
                app_home_cleanup: false,
                confirm: None,
                agent_port: 7779,
                agent_token: None,
                remote_agent_url: None,
                remote_agent_token: None,
                remote_agent_ssh_autostart: false,
                remote_agent_ssh_target: None,
                remote_agent_ssh_command: None,
                tls: false,
                tls_cert: None,
                tls_key: None,
                tls_self_signed: false,
            };
            assert_eq!(
                should_start_tray(&args),
                expected,
                "headless={headless}, no_tray={no_tray}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn should_start_tray_linux_display_variations() {
        unsafe { std::env::set_var("DISPLAY", ":0") };
        let args = cli::AppArgs {
            port: 7778,
            gpu_backend: "auto".to_string(),
            models_dir: None,
            gpu_arch: None,
            gpu_devices: None,
            llama_poll_interval: 1,
            llama_server_path: None,
            llama_server_cwd: None,
            presets_file: None,
            sessions_file: None,
            config_dir: None,
            migration_test_root: None,
            headless: false,
            no_tray: false,
            agent: false,
            agent_host: "127.0.0.1".to_string(),
            host: "127.0.0.1".to_string(),
            basic_auth: None,
            form_auth: None,
            ingest_spec_decode_report: None,
            clear_auth_config: false,
            app_home_migration_status: false,
            app_home_migration_preview: false,
            app_home_migrate: false,
            app_home_rollback_preview: false,
            app_home_rollback: false,
            app_home_cleanup: false,
            confirm: None,
            agent_port: 7779,
            agent_token: None,
            remote_agent_url: None,
            remote_agent_token: None,
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: None,
            remote_agent_ssh_command: None,
            tls: false,
            tls_cert: None,
            tls_key: None,
            tls_self_signed: false,
        };
        assert!(should_start_tray(&args));
        unsafe { std::env::remove_var("DISPLAY") };

        unsafe { std::env::set_var("WAYLAND_DISPLAY", "wayland-0") };
        let args = cli::AppArgs {
            port: 7778,
            gpu_backend: "auto".to_string(),
            models_dir: None,
            gpu_arch: None,
            gpu_devices: None,
            llama_poll_interval: 1,
            llama_server_path: None,
            llama_server_cwd: None,
            presets_file: None,
            sessions_file: None,
            config_dir: None,
            migration_test_root: None,
            headless: false,
            no_tray: false,
            agent: false,
            agent_host: "127.0.0.1".to_string(),
            host: "127.0.0.1".to_string(),
            basic_auth: None,
            form_auth: None,
            ingest_spec_decode_report: None,
            clear_auth_config: false,
            app_home_migration_status: false,
            app_home_migration_preview: false,
            app_home_migrate: false,
            app_home_rollback_preview: false,
            app_home_rollback: false,
            app_home_cleanup: false,
            confirm: None,
            agent_port: 7779,
            agent_token: None,
            remote_agent_url: None,
            remote_agent_token: None,
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: None,
            remote_agent_ssh_command: None,
            tls: false,
            tls_cert: None,
            tls_key: None,
            tls_self_signed: false,
        };
        assert!(should_start_tray(&args));
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };

        unsafe { std::env::remove_var("DISPLAY") };
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
        let args = cli::AppArgs {
            port: 7778,
            gpu_backend: "auto".to_string(),
            models_dir: None,
            gpu_arch: None,
            gpu_devices: None,
            llama_poll_interval: 1,
            llama_server_path: None,
            llama_server_cwd: None,
            presets_file: None,
            sessions_file: None,
            config_dir: None,
            migration_test_root: None,
            headless: false,
            no_tray: false,
            agent: false,
            agent_host: "127.0.0.1".to_string(),
            host: "127.0.0.1".to_string(),
            basic_auth: None,
            form_auth: None,
            ingest_spec_decode_report: None,
            clear_auth_config: false,
            app_home_migration_status: false,
            app_home_migration_preview: false,
            app_home_migrate: false,
            app_home_rollback_preview: false,
            app_home_rollback: false,
            app_home_cleanup: false,
            confirm: None,
            agent_port: 7779,
            agent_token: None,
            remote_agent_url: None,
            remote_agent_token: None,
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: None,
            remote_agent_ssh_command: None,
            tls: false,
            tls_cert: None,
            tls_key: None,
            tls_self_signed: false,
        };
        assert!(!should_start_tray(&args));
    }

    #[cfg(all(not(target_os = "linux"), feature = "native-tray"))]
    #[test]
    fn should_start_tray_non_linux_default_enabled() {
        let args = cli::AppArgs {
            port: 7778,
            gpu_backend: "auto".to_string(),
            models_dir: None,
            gpu_arch: None,
            gpu_devices: None,
            llama_poll_interval: 1,
            llama_server_path: None,
            llama_server_cwd: None,
            presets_file: None,
            sessions_file: None,
            config_dir: None,
            migration_test_root: None,
            headless: false,
            no_tray: false,
            agent: false,
            agent_host: "127.0.0.1".to_string(),
            host: "127.0.0.1".to_string(),
            basic_auth: None,
            form_auth: None,
            ingest_spec_decode_report: None,
            clear_auth_config: false,
            app_home_migration_status: false,
            app_home_migration_preview: false,
            app_home_migrate: false,
            app_home_rollback_preview: false,
            app_home_rollback: false,
            app_home_cleanup: false,
            confirm: None,
            agent_port: 7779,
            agent_token: None,
            remote_agent_url: None,
            remote_agent_token: None,
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: None,
            remote_agent_ssh_command: None,
            tls: false,
            tls_cert: None,
            tls_key: None,
            tls_self_signed: false,
        };
        assert!(should_start_tray(&args));
    }

    #[cfg(not(feature = "native-tray"))]
    #[test]
    fn should_start_tray_without_desktop_feature_disabled() {
        let args = cli::AppArgs {
            port: 7778,
            gpu_backend: "auto".to_string(),
            models_dir: None,
            gpu_arch: None,
            gpu_devices: None,
            llama_poll_interval: 1,
            llama_server_path: None,
            llama_server_cwd: None,
            presets_file: None,
            sessions_file: None,
            config_dir: None,
            headless: false,
            no_tray: false,
            agent: false,
            agent_host: "127.0.0.1".to_string(),
            host: "127.0.0.1".to_string(),
            basic_auth: None,
            form_auth: None,
            ingest_spec_decode_report: None,
            clear_auth_config: false,
            app_home_migration_status: false,
            app_home_migration_preview: false,
            app_home_migrate: false,
            app_home_rollback_preview: false,
            app_home_rollback: false,
            app_home_cleanup: false,
            confirm: None,
            agent_port: 7779,
            agent_token: None,
            remote_agent_url: None,
            remote_agent_token: None,
            remote_agent_ssh_autostart: false,
            remote_agent_ssh_target: None,
            remote_agent_ssh_command: None,
            tls: false,
            tls_cert: None,
            tls_key: None,
            tls_self_signed: false,
        };
        assert!(!should_start_tray(&args));
    }
}
