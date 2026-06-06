//! Backend adapters (feature-gated; each pulls a heavy tensor dependency).

#[cfg(feature = "candle")]
pub mod candle;

// Module is `tch_backend` (not `tch`) to avoid clashing with the `tch` crate.
#[cfg(feature = "tch")]
pub mod tch_backend;
