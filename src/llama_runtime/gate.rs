//! One runtime-mutation gate (plan §4.3.1).
//!
//! Prebuilt update, source promote, rollback and uninstall each take this gate
//! for the validation/swap window only. Downloads and compiles never hold it.
//! The gate is acquired with `try_begin`, never awaited: a second mutation gets
//! `Busy { owner }` (an HTTP 409) instead of queueing behind a model swap.
//!
//! Lock order, whenever more than one guard is needed:
//! toolchain lease -> checkout/build/cache ownership -> **this gate** ->
//! model lifecycle (`AppState::server_lifecycle`) -> session locks.
//! Nothing here may be acquired while a later lock is held.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeOp {
    Update,
    Promote,
    Rollback,
    Uninstall,
}

impl RuntimeOp {
    pub fn label(self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Promote => "promote",
            Self::Rollback => "rollback",
            Self::Uninstall => "uninstall",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Busy {
    pub owner: RuntimeOp,
}

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the llama.cpp runtime is busy ({} in progress)",
            self.owner.label()
        )
    }
}

impl std::error::Error for Busy {}

/// Number of runtime swaps (stop, verify, rename, restart) in flight in this
/// process (at most one in production; a counter so independent gates in tests
/// cannot clobber each other). Capability probes and model launches consult it
/// so they never execute or hash a half-swapped directory.
static SWAP_WINDOWS: AtomicUsize = AtomicUsize::new(0);

tokio::task_local! {
    /// Set by the promotion code around its own model restart so the launch
    /// gate lets it through.
    pub static RUNTIME_OWNER: bool;
}

pub fn swap_window_active() -> bool {
    SWAP_WINDOWS.load(Ordering::SeqCst) > 0
}

/// True when the calling task is the runtime owner (its own restart).
pub fn is_runtime_owner() -> bool {
    RUNTIME_OWNER.try_with(|owner| *owner).unwrap_or(false)
}

/// Shared leases held by anything that executes or hashes the live runtime
/// (capability and K/V probes). A swap drains them after it opens its window,
/// so it never renames files a probe has open and no probe straddles two
/// runtime versions.
static READ_LEASES: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

/// Held for the whole of a probe. Dropping it releases the runtime to a swap.
pub struct RuntimeReadLease {
    _held: Option<tokio::sync::RwLockReadGuard<'static, ()>>,
}

/// A swap is in progress; the probe must not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePaused;

impl std::fmt::Display for RuntimePaused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the llama.cpp runtime is being updated")
    }
}

impl std::error::Error for RuntimePaused {}

/// Take a read lease, or report that a swap owns the runtime. The window is
/// checked only after the lease is held: a swap opens its window and then
/// drains leases, so a probe either sees the window or is waited for.
pub async fn acquire_read_lease() -> Result<RuntimeReadLease, RuntimePaused> {
    if is_runtime_owner() {
        // The swap's own restart runs inside the window it opened.
        return Ok(RuntimeReadLease { _held: None });
    }
    let held = READ_LEASES.read().await;
    if swap_window_active() {
        return Err(RuntimePaused);
    }
    Ok(RuntimeReadLease { _held: Some(held) })
}

/// Wait for every outstanding read lease. Call after opening the swap window;
/// probes arriving later see the window and stand down.
pub async fn drain_read_leases() {
    drop(READ_LEASES.write().await);
}

#[derive(Debug)]
pub struct RuntimeGate {
    mutex: Arc<AsyncMutex<()>>,
    owner: Mutex<Option<RuntimeOp>>,
}

impl Default for RuntimeGate {
    fn default() -> Self {
        Self {
            mutex: Arc::new(AsyncMutex::new(())),
            owner: Mutex::new(None),
        }
    }
}

pub struct RuntimeGuard {
    gate: Arc<RuntimeGate>,
    op: RuntimeOp,
    _held: OwnedMutexGuard<()>,
    swap_window: bool,
}

impl RuntimeGuard {
    pub fn op(&self) -> RuntimeOp {
        self.op
    }

    /// Mark the stop/verify/swap/restart window. Cleared when the guard drops.
    pub fn enter_swap_window(&mut self) {
        if !self.swap_window {
            SWAP_WINDOWS.fetch_add(1, Ordering::SeqCst);
            self.swap_window = true;
        }
    }

    /// Whether *this* guard has opened the swap window.
    pub fn in_swap_window(&self) -> bool {
        self.swap_window
    }
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        if self.swap_window {
            SWAP_WINDOWS.fetch_sub(1, Ordering::SeqCst);
        }
        *self.gate.owner.lock().unwrap() = None;
    }
}

impl RuntimeGate {
    pub fn try_begin(self: &Arc<Self>, op: RuntimeOp) -> Result<RuntimeGuard, Busy> {
        match self.mutex.clone().try_lock_owned() {
            Ok(held) => {
                *self.owner.lock().unwrap() = Some(op);
                Ok(RuntimeGuard {
                    gate: Arc::clone(self),
                    op,
                    _held: held,
                    swap_window: false,
                })
            }
            Err(_) => Err(Busy {
                owner: self.current_owner().unwrap_or(op),
            }),
        }
    }

    pub fn current_owner(&self) -> Option<RuntimeOp> {
        *self.owner.lock().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn second_mutation_is_rejected_with_the_owner_then_allowed_after_drop() {
        let gate = Arc::new(RuntimeGate::default());
        let first = gate.try_begin(RuntimeOp::Update).unwrap();
        assert_eq!(gate.current_owner(), Some(RuntimeOp::Update));
        let busy = gate.try_begin(RuntimeOp::Rollback).err().unwrap();
        assert_eq!(busy.owner, RuntimeOp::Update);
        assert!(busy.to_string().contains("update in progress"));
        drop(first);
        assert_eq!(gate.current_owner(), None);
        assert!(gate.try_begin(RuntimeOp::Rollback).is_ok());
    }

    #[tokio::test]
    async fn swap_window_is_set_only_by_the_owner_and_cleared_on_drop() {
        let gate = Arc::new(RuntimeGate::default());
        let mut guard = gate.try_begin(RuntimeOp::Promote).unwrap();
        assert!(!guard.in_swap_window());
        guard.enter_swap_window();
        guard.enter_swap_window(); // idempotent: counted once
        assert!(guard.in_swap_window());
        assert!(swap_window_active());
        drop(guard);
    }

    #[tokio::test]
    async fn a_swap_waits_for_an_open_probe_and_later_probes_stand_down() {
        // Taken directly: other tests' swap windows share the global counter.
        let lease = READ_LEASES.read().await;
        let gate = Arc::new(RuntimeGate::default());
        let mut guard = gate.try_begin(RuntimeOp::Promote).unwrap();
        guard.enter_swap_window();
        let drained = tokio::spawn(drain_read_leases());
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!drained.is_finished(), "the swap must wait for the probe");
        drop(lease);
        drained.await.unwrap();
        assert_eq!(acquire_read_lease().await.err(), Some(RuntimePaused));
        drop(guard);
    }

    #[tokio::test]
    async fn owner_token_is_task_scoped() {
        assert!(!is_runtime_owner());
        RUNTIME_OWNER
            .scope(true, async {
                assert!(is_runtime_owner());
            })
            .await;
        assert!(!is_runtime_owner());
    }
}
