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

use std::sync::atomic::{AtomicBool, Ordering};
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

/// True while a runtime swap (stop, verify, rename, restart) is in flight in
/// this process. Capability probes and model launches consult it so they never
/// execute or hash a half-swapped directory.
static SWAP_WINDOW: AtomicBool = AtomicBool::new(false);

tokio::task_local! {
    /// Set by the promotion code around its own model restart so the launch
    /// gate lets it through.
    pub static RUNTIME_OWNER: bool;
}

pub fn swap_window_active() -> bool {
    SWAP_WINDOW.load(Ordering::SeqCst)
}

/// True when the calling task is the runtime owner (its own restart).
pub fn is_runtime_owner() -> bool {
    RUNTIME_OWNER.try_with(|owner| *owner).unwrap_or(false)
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
        SWAP_WINDOW.store(true, Ordering::SeqCst);
        self.swap_window = true;
    }
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        if self.swap_window {
            SWAP_WINDOW.store(false, Ordering::SeqCst);
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
        assert!(!swap_window_active());
        guard.enter_swap_window();
        assert!(swap_window_active());
        drop(guard);
        assert!(!swap_window_active());
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
