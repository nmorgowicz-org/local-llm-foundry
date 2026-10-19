//! Management of the installed llama.cpp runtime.
//!
//! Phase 1 introduces the bounded runtime probe and closure identity used by
//! K/V capability evidence. Shared promotion/rollback arrives in a later phase
//! (`docs/plans/20261019-llamacpp_compilation.md` §4.3).

pub mod probe;
