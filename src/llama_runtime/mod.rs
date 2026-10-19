//! Management of the installed llama.cpp runtime.
//!
//! - `probe`: bounded runtime probe and loaded-closure identity.
//! - `verify`: the one staged/live runtime verifier.
//! - `gate`: the single runtime-mutation gate and swap window.
//! - `manifest`: provenance (never capability).
//!
//! Journaled promotion/rollback lives in `promote` (plan §4.3).

pub mod gate;
pub mod host;
pub mod manifest;
pub mod probe;
pub mod promote;
pub mod startup;
pub mod verify;
