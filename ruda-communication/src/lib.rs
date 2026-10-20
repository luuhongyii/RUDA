#[macro_use]
extern crate derive_new;

mod base;
pub use base::*;

#[cfg(not(target_family = "wasm"))]
pub mod util;

#[cfg(all(feature = "websocket", not(target_family = "wasm")))]
pub mod websocket;

#[cfg(feature = "iroh")]
pub mod iroh;

#[cfg(all(feature = "iroh", feature = "websocket", not(target_family = "wasm")))]
pub mod transport;

pub mod transfer;

#[cfg(all(feature = "data-service", not(target_family = "wasm")))]
pub mod data_service;
