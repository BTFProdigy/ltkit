// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Michael King — part of ltkit (https://github.com/BTFProdigy/ltkit)
// Licensed under the Apache License, Version 2.0. See the LICENSE and NOTICE
// files in the project root. Attribution must be retained in derivative works.
//! Backend adapters (feature-gated; each pulls a heavy tensor dependency).

#[cfg(feature = "candle")]
pub mod candle;

// Module is `tch_backend` (not `tch`) to avoid clashing with the `tch` crate.
#[cfg(feature = "tch")]
pub mod tch_backend;