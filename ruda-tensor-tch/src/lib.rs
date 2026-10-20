#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::single_range_in_vec_init)]

//! Ruda LibTorch tensor backend

mod backend;
mod element;
mod ops;
mod rng;
mod tensor;

#[cfg(test)]
mod ownership_tests;
#[cfg(feature = "distributed")]
pub mod distributed;

pub use backend::*;
pub use element::*;
pub use tensor::*;
