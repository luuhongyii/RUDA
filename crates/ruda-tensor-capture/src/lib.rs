#![cfg_attr(not(feature = "std"), no_std)]

//! Non-executing tensor backend for capturing operation graphs and initialized values.

extern crate alloc;

mod capture;
pub use capture::*;
pub use ruda_tensor::graph::TensorId;
