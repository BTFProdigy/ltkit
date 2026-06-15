// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
//! LTKit — backend-agnostic lottery-ticket / IMP framework (Rust core).
//!
//! The engine drives any model through the six verbs of [`PrunableModel`]
//! (see `../CONTRACT.md`). Heavy tensor math lives behind the trait; the IMP
//! loop itself is pure Rust with no external dependencies. The optional
//! `candle` feature provides a native backend over `candle_core::VarMap`, and
//! the optional `tch` feature provides a libtorch backend over `tch::nn::VarStore`.

pub mod contract;
pub mod imp;

#[cfg(any(feature = "candle", feature = "tch"))]
pub mod backends;

pub use contract::{Criterion, PrunableModel, RewindPolicy};
pub use imp::{run_imp, ImpConfig, ImpResult};