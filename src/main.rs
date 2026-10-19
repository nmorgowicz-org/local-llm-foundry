// Canonical Local LLM Foundry executable entrypoint.
#![cfg_attr(windows, windows_subsystem = "windows")]

fn main() -> anyhow::Result<()> {
    // Hidden re-exec: the llama.cpp build's process supervisor (Unix). It must
    // run before any argument parsing or application setup.
    #[cfg(unix)]
    {
        let mut args = std::env::args_os().skip(1);
        if args.next().as_deref()
            == Some(std::ffi::OsStr::new(
                llama_monitor::llama_build::supervise::SUPERVISOR_ARG,
            ))
        {
            let args: Vec<String> = args.map(|arg| arg.to_string_lossy().into_owned()).collect();
            std::process::exit(llama_monitor::llama_build::supervise::run_supervisor(&args));
        }
    }
    if let Some(code) = llama_monitor::llama_build::cli::try_run(std::env::args_os()) {
        std::process::exit(code);
    }
    llama_monitor::runner::run()
}
