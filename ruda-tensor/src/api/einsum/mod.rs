//! Einstein summation through native differentiable tensor operations.

use core::marker::PhantomData;
use crate::api::{backend::Backend, Float, Numeric, Tensor};

mod equation;
mod execution;

/// A rank-erased einsum operand, retaining its backend, kind and dtype.
#[derive(Clone, Debug)]
pub struct EinsumOperand<B: Backend, K: Numeric<B> = Float> {
    primitive: K::Primitive,
    kind: PhantomData<(B, K)>,
}

impl<B: Backend, const D: usize, K: Numeric<B>> From<Tensor<B, D, K>> for EinsumOperand<B, K> {
    fn from(tensor: Tensor<B, D, K>) -> Self {
        Self { primitive: tensor.primitive, kind: PhantomData }
    }
}

impl<B: Backend, const D: usize, K: Numeric<B>> From<&Tensor<B, D, K>> for EinsumOperand<B, K> {
    fn from(tensor: &Tensor<B, D, K>) -> Self { tensor.clone().into() }
}

impl<B: Backend, const D: usize, K: Numeric<B>> Tensor<B, D, K> {
    /// Contract a runtime Einstein equation, with ellipses, broadcasting and repeated labels.
    /// Scalar operands and results use shape `[1]`; all inputs share a dtype and device.
    pub fn einsum(equation: &str, operands: impl IntoIterator<Item = EinsumOperand<B, K>>) -> Self {
        let equation = equation::parse(equation).unwrap_or_else(|error| panic!("einsum: {error}"));
        execution::execute(&equation.plan(), operands)
    }
}
