//! Owned process supervision (plan §4.4 *Cancel and crash containment*).
//!
//! Every subprocess a build starts (git, cmake, compilers, smoke tests) runs
//! contained, so cancel and a crash of the app both reach every descendant:
//!
//! - **Windows:** the child is created suspended, assigned to a Job Object with
//!   `KILL_ON_JOB_CLOSE` (no breakaway flags), then resumed. Closing the job
//!   handle - which the kernel does when the app dies - kills everything.
//! - **Unix:** the app re-executes itself as a small supervisor
//!   (`__llama-build-supervise`). The supervisor owns the child's process group
//!   and watches a control pipe; when the app cancels *or dies* the pipe closes
//!   and the supervisor sends `TERM`, then `KILL` after a grace period, and
//!   does not exit until the group is empty.
//!
//! Recovery never signals a saved PID blindly: a record carries the process
//! creation time, and a mismatch means the PID was reused and nothing is killed.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// argv[1] that switches the executable into supervisor mode.
pub const SUPERVISOR_ARG: &str = "__llama-build-supervise";
/// Grace between `TERM` and `KILL`.
pub const TERM_GRACE: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Default)]
pub struct SpawnSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Windows only: appended verbatim (for `cmd /s /c "..."` quoting).
    pub raw_args: Option<String>,
    pub cwd: Option<PathBuf>,
    /// The complete environment. `None` inherits the app's.
    pub env: Option<Vec<(String, String)>>,
    /// `nice 10` / `BELOW_NORMAL_PRIORITY_CLASS`.
    pub low_priority: bool,
}

impl SpawnSpec {
    pub fn new(program: impl Into<PathBuf>, args: &[&str]) -> Self {
        Self {
            program: program.into(),
            args: args.iter().map(|s| s.to_string()).collect(),
            ..Self::default()
        }
    }

    /// Shell-quoted command line for logs.
    pub fn display(&self) -> String {
        let mut parts = vec![quote(&self.program.to_string_lossy())];
        parts.extend(self.args.iter().map(|a| quote(a)));
        if let Some(raw) = &self.raw_args {
            parts.push(raw.clone());
        }
        parts.join(" ")
    }
}

fn quote(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=+,@\\".contains(c))
    {
        arg.to_string()
    } else {
        format!("\"{}\"", arg.replace('"', "\\\""))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    Stdout(String),
    Stderr(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    /// Every descendant is gone. `false` blocks releasing ownership.
    pub contained: bool,
}

/// Where the Unix supervisor binary is. Tests point this at the built binary.
#[derive(Debug, Clone)]
pub struct Launcher {
    pub supervisor_exe: Option<PathBuf>,
}

impl Launcher {
    pub fn current_exe() -> Self {
        Self {
            supervisor_exe: std::env::current_exe().ok(),
        }
    }
}

/// A contained, running child.
pub struct Contained {
    child: Child,
    #[cfg(unix)]
    control: Option<std::process::ChildStdin>,
    #[cfg(windows)]
    job: win::Job,
}

pub fn spawn(spec: &SpawnSpec, launcher: &Launcher) -> std::io::Result<Contained> {
    #[cfg(unix)]
    {
        spawn_unix(spec, launcher)
    }
    #[cfg(windows)]
    {
        let _ = launcher;
        spawn_windows(spec)
    }
}

impl Contained {
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn take_output(
        &mut self,
    ) -> (
        Option<std::process::ChildStdout>,
        Option<std::process::ChildStderr>,
    ) {
        (self.child.stdout.take(), self.child.stderr.take())
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<i32>> {
        Ok(self.child.try_wait()?.map(|s| s.code().unwrap_or(-1)))
    }

    /// Ask for termination and wait until no descendant is left. Returns
    /// whether containment held.
    pub fn terminate(&mut self) -> bool {
        #[cfg(unix)]
        {
            // Closing the control pipe is the cancel signal: the supervisor
            // does TERM, grace, KILL and waits for an empty group.
            self.control.take();
            let deadline = Instant::now() + TERM_GRACE + Duration::from_secs(15);
            loop {
                match self.child.try_wait() {
                    Ok(Some(status)) => return status.code().is_some_and(|c| c != 125),
                    Ok(None) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(25))
                    }
                    _ => {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        return false;
                    }
                }
            }
        }
        #[cfg(windows)]
        {
            self.job.terminate();
            let _ = self.child.wait();
            self.job.wait_empty(Duration::from_secs(15))
        }
    }
}

/// Longest child output line kept whole. A child that never prints a newline
/// must not grow the supervisor's memory without bound.
pub const MAX_LINE_BYTES: usize = 64 * 1024;
const TRUNCATION_MARKER: &str = " [line truncated]";
const OUTPUT_CHANNEL_CAPACITY: usize = 256;
/// Bound EOF confirmation even if a descendant escaped the original group.
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_secs(5);

/// Yield to cancellation/deadline checks even if writers never stop producing.
fn drain_ready_output(rx: &mpsc::Receiver<Line>, sink: &mut dyn FnMut(Line)) {
    for _ in 0..OUTPUT_CHANNEL_CAPACITY {
        match rx.try_recv() {
            Ok(line) => sink(line),
            Err(_) => break,
        }
    }
}

fn finish_output(
    rx: mpsc::Receiver<Line>,
    readers: Vec<std::thread::JoinHandle<()>>,
    contained: bool,
    drain_timeout: Duration,
    sink: &mut dyn FnMut(Line),
) -> bool {
    if contained {
        // Drain BEFORE joining: a reader may be blocked sending final lines.
        // An escaped/session-changing descendant can retain a pipe even after
        // the original group is empty, so EOF confirmation has a total budget.
        let deadline = Instant::now() + drain_timeout;
        loop {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            match rx.recv_timeout(remaining) {
                Ok(line) => sink(line),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => return false,
            }
        }
        for reader in readers {
            let _ = reader.join();
        }
        true
    } else {
        // An uncontained descendant may retain or continually write a pipe.
        // Neither EOF nor an empty queue is guaranteed. Deliver at most one
        // bounded batch, then drop the receiver and detach the readers.
        // The durable ownership record still blocks reuse.
        drain_ready_output(&rx, sink);
        false
    }
}

/// Split `reader` into lines, never holding more than `cap` bytes of one line.
/// The rest of an over-long line is discarded and the kept part ends in a
/// marker; the lines after it are unaffected.
fn for_each_line(reader: impl std::io::Read, cap: usize, mut each: impl FnMut(String)) {
    let mut reader = BufReader::new(reader);
    let mut line: Vec<u8> = Vec::new();
    let mut truncated = false;
    let mut emit = |line: &mut Vec<u8>, truncated: &mut bool| {
        let mut text = String::from_utf8_lossy(line)
            .trim_end_matches('\r')
            .to_string();
        if *truncated {
            text.push_str(TRUNCATION_MARKER);
        }
        line.clear();
        *truncated = false;
        each(text);
    };
    loop {
        let buf = match reader.fill_buf() {
            Ok([]) => break,
            Ok(buf) => buf,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let newline = buf.iter().position(|b| *b == b'\n');
        let chunk = &buf[..newline.unwrap_or(buf.len())];
        let room = cap.saturating_sub(line.len());
        line.extend_from_slice(&chunk[..chunk.len().min(room)]);
        truncated |= chunk.len() > room;
        let consumed = newline.map_or(buf.len(), |i| i + 1);
        reader.consume(consumed);
        if newline.is_some() {
            emit(&mut line, &mut truncated);
        }
    }
    if !line.is_empty() || truncated {
        emit(&mut line, &mut truncated);
    }
}

/// Run to completion, teeing output line-by-line to `sink`, polling `cancel`.
pub fn run_contained(
    spec: &SpawnSpec,
    launcher: &Launcher,
    sink: &mut dyn FnMut(Line),
    cancel: &dyn Fn() -> bool,
) -> std::io::Result<RunOutcome> {
    run_contained_with(spec, launcher, sink, cancel, &mut |_| Ok(()))
}

/// As [`run_contained`], reporting the contained process id as soon as it
/// exists so the caller can persist an identity record for crash recovery.
pub fn run_contained_with(
    spec: &SpawnSpec,
    launcher: &Launcher,
    sink: &mut dyn FnMut(Line),
    cancel: &dyn Fn() -> bool,
    on_spawn: &mut dyn FnMut(u32) -> std::io::Result<()>,
) -> std::io::Result<RunOutcome> {
    let mut contained = spawn(spec, launcher)?;
    if let Err(e) = on_spawn(contained.id()) {
        let reaped = contained.terminate();
        return Err(std::io::Error::other(format!(
            "could not persist child ownership: {e}; containment reaped: {reaped}"
        )));
    }
    let (stdout, stderr) = contained.take_output();
    let (tx, rx) = mpsc::sync_channel::<Line>(OUTPUT_CHANNEL_CAPACITY);
    let mut readers = Vec::new();
    if let Some(out) = stdout {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || {
            for_each_line(out, MAX_LINE_BYTES, |line| {
                let _ = tx.send(Line::Stdout(line));
            });
        }));
    }
    if let Some(err) = stderr {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || {
            for_each_line(err, MAX_LINE_BYTES, |line| {
                let _ = tx.send(Line::Stderr(line));
            });
        }));
    }
    drop(tx);

    let mut cancelled = false;
    let mut contained_ok = true;
    let exit_code = loop {
        drain_ready_output(&rx, sink);
        if !cancelled && cancel() {
            cancelled = true;
            contained_ok = contained.terminate();
        }
        match contained.try_wait()? {
            Some(code) => break Some(code),
            None => match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(line) => sink(line),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Pipes closed: the child is exiting.
                    std::thread::sleep(Duration::from_millis(10));
                }
            },
        }
    };
    // The leader is gone; make sure nothing it started is still writing.
    if !cancelled {
        contained_ok = contained.terminate();
    }
    contained_ok &= finish_output(rx, readers, contained_ok, OUTPUT_DRAIN_GRACE, sink);
    Ok(RunOutcome {
        exit_code: if cancelled { None } else { exit_code },
        cancelled,
        contained: contained_ok,
    })
}

// ---------------------------------------------------------------------------
// Unix
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn spawn_unix(spec: &SpawnSpec, launcher: &Launcher) -> std::io::Result<Contained> {
    let exe = launcher
        .supervisor_exe
        .clone()
        .ok_or_else(|| std::io::Error::other("the supervisor executable is unknown"))?;
    let mut cmd = Command::new(exe);
    cmd.arg(SUPERVISOR_ARG);
    if spec.low_priority {
        cmd.arg("--nice");
    }
    cmd.arg("--").arg(&spec.program).args(&spec.args);
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    if let Some(env) = &spec.env {
        cmd.env_clear();
        for (k, v) in env {
            cmd.env(k, v);
        }
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let control = child.stdin.take();
    Ok(Contained { child, control })
}

/// Supervisor entry point. Never returns to the caller's `main`.
#[cfg(unix)]
pub fn run_supervisor(args: &[String]) -> i32 {
    use std::os::unix::process::CommandExt;

    let mut rest = args;
    let mut nice = false;
    if rest.first().map(String::as_str) == Some("--nice") {
        nice = true;
        rest = &rest[1..];
    }
    if rest.first().map(String::as_str) == Some("--") {
        rest = &rest[1..];
    }
    let Some((program, program_args)) = rest.split_first() else {
        eprintln!("supervisor: no command");
        return 2;
    };

    // Our own group, so a signal aimed at the app's group never reaches us and
    // ours never reaches the app.
    // SAFETY: plain libc calls with valid arguments.
    unsafe {
        libc::setpgid(0, 0);
        libc::signal(libc::SIGTERM, libc::SIG_IGN);
        libc::signal(libc::SIGINT, libc::SIG_IGN);
        libc::signal(libc::SIGHUP, libc::SIG_IGN);
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        if nice {
            libc::setpriority(libc::PRIO_PROCESS, 0, 10);
        }
    }

    let mut cmd = Command::new(program);
    cmd.args(program_args).stdin(Stdio::null());
    // The child leads a fresh group; every descendant joins it.
    cmd.process_group(0);
    // SAFETY: only async-signal-safe `signal` calls between fork and exec.
    unsafe {
        cmd.pre_exec(|| {
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGHUP, libc::SIG_DFL);
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            Ok(())
        });
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("supervisor: could not start {program}: {e}");
            return 127;
        }
    };
    let pgid = child.id() as i32;

    // Cancel and parent death are the same event: the control pipe closes.
    let (tx, rx) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut sink);
        let _ = tx.send(());
    });

    let mut kill_at: Option<Instant> = None;
    let status = loop {
        if let Ok(Some(status)) = child.try_wait() {
            break status;
        }
        if kill_at.is_none() && rx.try_recv().is_ok() {
            // SAFETY: signalling a group we created.
            unsafe { libc::killpg(pgid, libc::SIGTERM) };
            kill_at = Some(Instant::now() + TERM_GRACE);
        }
        if kill_at.is_some_and(|at| Instant::now() >= at) {
            // SAFETY: as above. Repeating KILL is harmless.
            unsafe { libc::killpg(pgid, libc::SIGKILL) };
        }
        std::thread::sleep(Duration::from_millis(40));
    };

    // The leader exited; nothing it started may outlive it.
    if !drain_group(pgid) {
        eprintln!("supervisor: descendants survived termination; repair required");
        return 125;
    }
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

/// `TERM`, wait for an empty group, then `KILL`.
#[cfg(unix)]
fn drain_group(pgid: i32) -> bool {
    let alive = || {
        // SAFETY: signal 0 only probes.
        unsafe { libc::killpg(pgid, 0) == 0 }
    };
    if !alive() {
        return true;
    }
    // SAFETY: signalling a group we created.
    unsafe { libc::killpg(pgid, libc::SIGTERM) };
    let deadline = Instant::now() + TERM_GRACE;
    while alive() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(40));
    }
    if alive() {
        // SAFETY: as above.
        unsafe { libc::killpg(pgid, libc::SIGKILL) };
        let deadline = Instant::now() + Duration::from_secs(5);
        while alive() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(40));
        }
    }
    !alive()
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

#[cfg(windows)]
fn spawn_windows(spec: &SpawnSpec) -> std::io::Result<Contained> {
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;

    const CREATE_SUSPENDED: u32 = 0x0000_0004;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;

    let job = win::Job::new()?;
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args);
    if let Some(raw) = &spec.raw_args {
        cmd.raw_arg(raw);
    }
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    if let Some(env) = &spec.env {
        cmd.env_clear();
        for (k, v) in env {
            cmd.env(k, v);
        }
    }
    let mut flags = CREATE_SUSPENDED | CREATE_NO_WINDOW;
    if spec.low_priority {
        flags |= BELOW_NORMAL_PRIORITY_CLASS;
    }
    cmd.creation_flags(flags)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = cmd.spawn()?;
    let handle = child.as_raw_handle();
    // Assign before resume: the process cannot have started a child of its own.
    if let Err(e) = job.assign(handle) {
        let mut child = child;
        let _ = child.kill();
        return Err(e);
    }
    win::resume(handle)?;
    Ok(Contained { child, job })
}

#[cfg(windows)]
pub(crate) mod win {
    use std::ffi::c_void;
    use std::time::{Duration, Instant};

    type Handle = *mut c_void;

    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x2000;
    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: i32 = 9;
    const JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS: i32 = 1;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimit {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        counters: [u64; 6],
    }

    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimit {
        basic: BasicLimit,
        io: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[repr(C)]
    #[derive(Default)]
    struct BasicAccounting {
        total_user_time: i64,
        total_kernel_time: i64,
        this_period_total_user_time: i64,
        this_period_total_kernel_time: i64,
        total_page_fault_count: u32,
        total_processes: u32,
        active_processes: u32,
        total_terminated_processes: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(job: Handle, class: i32, info: *mut c_void, len: u32) -> i32;
        fn QueryInformationJobObject(
            job: Handle,
            class: i32,
            info: *mut c_void,
            len: u32,
            returned: *mut u32,
        ) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn TerminateJobObject(job: Handle, exit_code: u32) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtResumeProcess(process: Handle) -> i32;
    }

    pub struct Job(Handle);

    // SAFETY: a job handle is a kernel object reference usable from any thread.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn new() -> std::io::Result<Self> {
            // SAFETY: null attributes/name create an anonymous job.
            let handle = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
            if handle.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let mut info = ExtendedLimit::default();
            // KILL_ON_JOB_CLOSE only: breakaway flags are deliberately absent.
            info.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `info` is a correctly laid-out ExtendedLimit of the stated size.
            let ok = unsafe {
                SetInformationJobObject(
                    handle,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                    &mut info as *mut _ as *mut c_void,
                    std::mem::size_of::<ExtendedLimit>() as u32,
                )
            };
            if ok == 0 {
                let e = std::io::Error::last_os_error();
                // SAFETY: handle came from CreateJobObjectW.
                unsafe { CloseHandle(handle) };
                return Err(e);
            }
            Ok(Self(handle))
        }

        pub fn assign(&self, process: *mut c_void) -> std::io::Result<()> {
            // SAFETY: both handles are valid for the call.
            if unsafe { AssignProcessToJobObject(self.0, process) } == 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn terminate(&self) {
            // SAFETY: valid job handle.
            unsafe { TerminateJobObject(self.0, 1) };
        }

        pub fn active_processes(&self) -> Option<u32> {
            let mut info = BasicAccounting::default();
            let mut returned = 0u32;
            // SAFETY: `info` matches the queried class and size.
            let ok = unsafe {
                QueryInformationJobObject(
                    self.0,
                    JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS,
                    &mut info as *mut _ as *mut c_void,
                    std::mem::size_of::<BasicAccounting>() as u32,
                    &mut returned,
                )
            };
            (ok != 0).then_some(info.active_processes)
        }

        pub fn wait_empty(&self, timeout: Duration) -> bool {
            let deadline = Instant::now() + timeout;
            loop {
                match self.active_processes() {
                    Some(0) => return true,
                    None => return false,
                    Some(_) if Instant::now() >= deadline => return false,
                    Some(_) => std::thread::sleep(Duration::from_millis(25)),
                }
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // KILL_ON_JOB_CLOSE: dropping the last handle ends every member.
            // SAFETY: handle is owned by this value.
            unsafe { CloseHandle(self.0) };
        }
    }

    pub fn resume(process: *mut c_void) -> std::io::Result<()> {
        // SAFETY: `process` is the suspended child's handle.
        let status = unsafe { NtResumeProcess(process) };
        if status < 0 {
            return Err(std::io::Error::other(format!(
                "NtResumeProcess failed: {status:#x}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Identity-checked process records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessRecord {
    pub job_id: String,
    pub pid: u32,
    /// Seconds since the epoch at process creation (OS-reported).
    pub start_time: u64,
}

/// Creation time of `pid` as the OS reports it, or `None` if it is not running.
pub fn process_start_time(pid: u32) -> Option<u64> {
    let mut sys = sysinfo::System::new();
    let pid = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).map(|p| p.start_time())
}

pub fn record_for(job_id: &str, pid: u32) -> Option<ProcessRecord> {
    Some(ProcessRecord {
        job_id: job_id.to_string(),
        pid,
        start_time: process_start_time(pid)?,
    })
}

pub fn load_records(file: &Path) -> std::io::Result<Vec<ProcessRecord>> {
    match std::fs::read(file) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(std::io::Error::other),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Serialises every read-modify-write of the process records in this process.
/// Concurrent owners (a build's child and a validation's smoke test) used to
/// race on one fixed temp name and on load-modify-save.
static RECORDS_LOCK: Mutex<()> = Mutex::new(());
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn records_lock() -> std::sync::MutexGuard<'static, ()> {
    RECORDS_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `<file name>.<pid>-<seq>.tmp`, unique per call and per process.
fn unique_tmp(file: &Path) -> PathBuf {
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    file.with_file_name(format!(
        "{name}.{}-{}.tmp",
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

fn save_records_locked(file: &Path, records: &[ProcessRecord]) -> std::io::Result<()> {
    if records.is_empty() {
        return match std::fs::remove_file(file) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    }
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = unique_tmp(file);
    let written = std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(records).map_err(std::io::Error::other)?,
    )
    .and_then(|()| std::fs::rename(&tmp, file));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

pub fn save_records(file: &Path, records: &[ProcessRecord]) -> std::io::Result<()> {
    let _guard = records_lock();
    save_records_locked(file, records)
}

/// Atomic load-modify-save: no other writer in this process can interleave.
pub fn update_records(
    file: &Path,
    change: impl FnOnce(&mut Vec<ProcessRecord>),
) -> std::io::Result<()> {
    let _guard = records_lock();
    let mut all = load_records(file)?;
    change(&mut all);
    save_records_locked(file, &all)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// The recorded process is gone; nothing to do.
    Gone,
    /// The PID now belongs to a different process: left alone.
    IdentityMismatch { pid: u32 },
    /// Members of the recorded group were signalled.
    Killed { pid: u32, members: usize },
    /// Containment could not be established; reuse stays blocked.
    Failed { pid: u32, reason: String },
}

/// Validate saved ownership before recovery. Unix records identify the
/// supervisor, not its child's independent process group. Without a durable
/// group handshake, even a dead/reused supervisor cannot prove its descendants
/// are gone. Preserve the record and require repair rather than signal an
/// unrelated PID or falsely release the tree.
pub fn recover_record(record: &ProcessRecord) -> RecoveryAction {
    #[cfg(unix)]
    {
        let leader = process_start_time(record.pid);
        if let Some(start) = leader
            && start.abs_diff(record.start_time) > 1
        {
            return RecoveryAction::IdentityMismatch { pid: record.pid };
        }
        RecoveryAction::Failed {
            pid: record.pid,
            reason: "the supervisor's child group cannot be proven empty; inspect descendants before repairing processes.json".into(),
        }
    }
    #[cfg(windows)]
    {
        // The Job Object dies with the app, taking every descendant with it.
        match process_start_time(record.pid) {
            None => RecoveryAction::Gone,
            Some(start) if start.abs_diff(record.start_time) > 1 => {
                RecoveryAction::IdentityMismatch { pid: record.pid }
            }
            Some(_) => RecoveryAction::Failed {
                pid: record.pid,
                reason: "the recorded process is still running outside a job".into(),
            },
        }
    }
}

/// Reap every saved record. Records whose processes are gone or reused are
/// dropped; failures are kept so reuse stays blocked.
pub fn recover_all(file: &Path) -> std::io::Result<Vec<(ProcessRecord, RecoveryAction)>> {
    let records = load_records(file)?;
    let mut keep = Vec::new();
    let mut out = Vec::new();
    for record in records {
        let action = recover_record(&record);
        if !matches!(action, RecoveryAction::Gone) {
            keep.push(record.clone());
        }
        out.push((record, action));
    }
    save_records(file, &keep)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_drains_more_than_capacity_before_joining_readers() {
        let (tx, rx) = mpsc::sync_channel(256);
        // Full before shutdown starts: the producer cannot finish until drained.
        for _ in 0..256 {
            tx.send(Line::Stdout("out".into())).unwrap();
        }
        let reader = std::thread::spawn(move || {
            for _ in 256..1024 {
                tx.send(Line::Stderr("err".into())).unwrap();
            }
        });
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut counts = (0, 0);
            let drained = finish_output(
                rx,
                vec![reader],
                true,
                Duration::from_secs(2),
                &mut |line| match line {
                    Line::Stdout(_) => counts.0 += 1,
                    Line::Stderr(_) => counts.1 += 1,
                },
            );
            assert!(drained, "output did not reach disconnect");
            let _ = done_tx.send(counts);
        });
        // A join-before-drain mutation strands only detached test threads,
        // not this test or the test process.
        assert_eq!(
            done_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("shutdown deadlocked"),
            (256, 768)
        );
    }

    #[test]
    fn unproven_containment_does_not_wait_for_an_inherited_pipe() {
        let (tx, rx) = mpsc::sync_channel(256);
        tx.send(Line::Stdout("buffered".into())).unwrap();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let reader = std::thread::spawn(move || {
            let _ = release_rx.recv_timeout(Duration::from_secs(10));
            drop(tx);
        });
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut count = 0;
            assert!(!finish_output(
                rx,
                vec![reader],
                false,
                Duration::from_secs(2),
                &mut |_| count += 1,
            ));
            let _ = done_tx.send(count);
        });
        let result = done_rx.recv_timeout(Duration::from_secs(5));
        // Release the simulated pipe even on failure.
        drop(release_tx);
        assert_eq!(result.expect("waited for uncontained reader"), 1);
    }

    #[test]
    fn apparently_contained_reader_without_eof_downgrades_containment() {
        let (tx, rx) = mpsc::sync_channel::<Line>(256);
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let drained =
                finish_output(rx, Vec::new(), true, Duration::from_millis(20), &mut |_| {});
            let _ = done_tx.send(drained);
        });
        // Keep the sender alive throughout the assertion: no scheduling race
        // can manufacture EOF. A blocking drain fails this finite outer wait.
        let result = done_rx.recv_timeout(Duration::from_secs(5));
        drop(tx);
        assert!(!result.expect("EOF confirmation exceeded its budget"));
    }

    #[test]
    fn continuous_output_yields_after_a_bounded_batch() {
        let (tx, rx) = mpsc::sync_channel(256);
        tx.send(Line::Stdout("initial".into())).unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut count = 0;
            drain_ready_output(&rx, &mut |_| {
                count += 1;
                // The queue never becomes empty, without relying on scheduling.
                tx.try_send(Line::Stdout("more".into())).unwrap();
            });
            let _ = done_tx.send(count);
        });
        let count = done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("continuous output starved control polling");
        assert!(count > 0 && count <= 256);
    }

    #[test]
    fn display_quotes_only_what_needs_it() {
        let spec = SpawnSpec::new("cmake", &["--build", "/a b/c", "-DX=1", "say \"hi\""]);
        assert_eq!(
            spec.display(),
            "cmake --build \"/a b/c\" -DX=1 \"say \\\"hi\\\"\""
        );
    }

    #[test]
    fn records_round_trip_and_an_empty_set_removes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("x/processes.json");
        let rec = ProcessRecord {
            job_id: "j".into(),
            pid: 7,
            start_time: 100,
        };
        save_records(&file, std::slice::from_ref(&rec)).unwrap();
        assert_eq!(load_records(&file).unwrap(), vec![rec]);
        save_records(&file, &[]).unwrap();
        assert!(!file.exists());
        save_records(&file, &[]).unwrap();
        assert!(load_records(&file).unwrap().is_empty());
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"{broken").unwrap();
        assert!(load_records(&file).is_err());
        assert!(recover_all(&file).is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"{broken");
    }

    fn lines_of(input: &[u8], cap: usize) -> Vec<String> {
        let mut out = Vec::new();
        for_each_line(input, cap, |l| out.push(l));
        out
    }

    #[test]
    fn short_lines_pass_through_unchanged() {
        assert_eq!(lines_of(b"a\r\nbb\n\nlast", 16), ["a", "bb", "", "last"]);
    }

    #[test]
    fn an_unterminated_endless_line_is_truncated_with_a_marker() {
        let big = vec![b'x'; MAX_LINE_BYTES * 3];
        let lines = lines_of(&big, MAX_LINE_BYTES);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].ends_with(TRUNCATION_MARKER), "marker missing");
        assert_eq!(lines[0].len(), MAX_LINE_BYTES + TRUNCATION_MARKER.len());
    }

    #[test]
    fn an_over_long_line_does_not_swallow_the_lines_after_it() {
        let mut input = vec![b'y'; 100];
        input.extend_from_slice(b"\nnext\n");
        let lines = lines_of(&input, 10);
        assert_eq!(
            lines,
            [
                format!("{}{TRUNCATION_MARKER}", "y".repeat(10)),
                "next".into()
            ]
        );
    }

    #[test]
    fn a_line_of_exactly_the_cap_is_not_truncated() {
        let mut input = vec![b'z'; 10];
        input.push(b'\n');
        assert_eq!(lines_of(&input, 10), ["z".repeat(10)]);
    }

    #[test]
    fn concurrent_saves_never_collide_on_a_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = std::sync::Arc::new(dir.path().join("processes.json"));
        let failures = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let threads: Vec<_> = (0..8u32)
            .map(|t| {
                let (file, failures) = (file.clone(), failures.clone());
                std::thread::spawn(move || {
                    for i in 0..150u32 {
                        let rec = ProcessRecord {
                            job_id: format!("t{t}"),
                            pid: t * 1000 + i + 1,
                            start_time: 1,
                        };
                        if save_records(&file, &[rec]).is_err() {
                            failures.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(failures.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(load_records(&file).unwrap().len(), 1, "never torn");
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "processes.json")
            .collect();
        assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");
    }

    #[test]
    fn concurrent_updates_never_lose_a_record() {
        let dir = tempfile::tempdir().unwrap();
        let file = std::sync::Arc::new(dir.path().join("processes.json"));
        let threads: Vec<_> = (0..8u32)
            .map(|t| {
                let file = file.clone();
                std::thread::spawn(move || {
                    for i in 0..40u32 {
                        let pid = t * 1000 + i + 1;
                        update_records(&file, |all| {
                            all.push(ProcessRecord {
                                job_id: format!("t{t}"),
                                pid,
                                start_time: 1,
                            })
                        })
                        .unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(load_records(&file).unwrap().len(), 8 * 40);
    }

    #[test]
    fn this_process_has_a_start_time() {
        let me = std::process::id();
        let t = process_start_time(me).unwrap();
        assert!(t > 1_500_000_000);
        assert!(record_for("j", me).is_some());
        assert!(process_start_time(u32::MAX - 3).is_none());
    }

    /// A live, unrelated process whose PID is in a record with the wrong
    /// creation time must never be signalled.
    #[cfg(unix)]
    #[test]
    fn a_reused_pid_is_never_signalled() {
        let mut bystander = Command::new("sleep").arg("30").spawn().unwrap();
        let real = process_start_time(bystander.id()).unwrap();
        let stale = ProcessRecord {
            job_id: "old".into(),
            pid: bystander.id(),
            start_time: real.saturating_sub(10_000),
        };
        assert_eq!(
            recover_record(&stale),
            RecoveryAction::IdentityMismatch {
                pid: bystander.id()
            }
        );
        assert!(
            bystander.try_wait().unwrap().is_none(),
            "the bystander was killed"
        );
        let _ = bystander.kill();
        let _ = bystander.wait();
    }

    #[cfg(unix)]
    #[test]
    fn a_dead_supervisor_does_not_prove_its_child_group_is_gone() {
        let rec = ProcessRecord {
            job_id: "j".into(),
            pid: 999_999,
            start_time: 1,
        };
        assert!(matches!(
            recover_record(&rec),
            RecoveryAction::Failed { .. }
        ));
    }

    #[cfg(unix)]
    #[test]
    fn recover_all_drops_resolved_records_and_keeps_failures() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("processes.json");
        save_records(
            &file,
            &[
                ProcessRecord {
                    job_id: "a".into(),
                    pid: 999_998,
                    start_time: 1,
                },
                ProcessRecord {
                    job_id: "b".into(),
                    pid: 999_997,
                    start_time: 1,
                },
            ],
        )
        .unwrap();
        let results = recover_all(&file).unwrap();
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|(_, a)| matches!(a, RecoveryAction::Failed { .. }))
        );
        assert_eq!(load_records(&file).unwrap().len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn a_missing_supervisor_executable_is_an_error_not_a_hang() {
        let launcher = Launcher {
            supervisor_exe: None,
        };
        assert!(spawn(&SpawnSpec::new("true", &[]), &launcher).is_err());
    }
}
