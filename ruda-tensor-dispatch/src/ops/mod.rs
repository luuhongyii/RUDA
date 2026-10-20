use alloc::{boxed::Box, vec::Vec};
use core::{future::Future, ops::Range};
use ruda_tensor::{Backend, AutodiffContext, TensorPrimitive, TensorData, Scalar, Distribution, ExecutionError,
    FloatDType, IntDType, BoolDType, Shape, Slice};
use ruda_tensor::ops::*;
use ruda_tensor::tensor::{Device, FloatTensor, IntTensor, BoolTensor, QuantizedTensor, quantization::QuantizationParametersPrimitive};
use ruda_core::tensor::quantization::{QuantScheme, QuantPropagation};
use crate::{Dispatch, DispatchTensor, context::Selection};

mod tensor;
mod int_tensor;
mod bool_tensor;
mod qtensor;
mod activation;
mod module;
mod transaction;
