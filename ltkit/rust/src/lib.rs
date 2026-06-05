//! LTKit — backend-agnostic lottery-ticket / IMP framework (Rust core).
//!
//! The engine drives any model through the six verbs of [`PrunableModel`]
//! (see `../CONTRACT.md`). Heavy tensor math lives behind the trait; the IMP
//! loop itself is pure Rust with no external dependencies. The optional
//! `candle` feature provides a native backend over `candle_core::VarMap`.

pub mod contract;
pub mod imp;

#[cfg(feature = "candle")]
pub mod backends;

pub use contract::{Criterion, PrunableModel, RewindPolicy};
pub use imp::{run_imp, ImpConfig, ImpResult};
