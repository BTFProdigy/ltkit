//! Backend adapters (feature-gated; each pulls a heavy tensor dependency).

#[cfg(feature = "candle")]
pub mod candle;
