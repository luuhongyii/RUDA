use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::{AutodiffTensor, AutodiffValue}};
use alloc::vec::Vec;

use ruda_tensor::{
    Backend, ExecutionError, Scalar, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use ruda_core::tensor::{BoolDType, FloatDType, IntDType, Shape};

impl<B: Backend, C: CheckpointStrategy> BoolTensorOps<Self> for Autodiff<B, C> {
    fn bool_autodiff_context(tensor: &BoolTensor<Self>) -> ruda_tensor::AutodiffContext { tensor.context }

    fn bool_with_autodiff_context(tensor: BoolTensor<Self>, context: ruda_tensor::AutodiffContext) -> BoolTensor<Self> {
        tensor.set_context::<C>(context)
    }

    fn bool_from_data(data: TensorData, device: &Device<B>) -> BoolTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::bool_from_data(data, device), context)
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        let tensor = tensor.primitive;
        B::bool_into_data(tensor).await
    }

    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_into_int(tensor, out_dtype), context)
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<B>) -> BoolTensor<Self> {
        let context = crate::context::scoped::<B, C>(device).unwrap_or(tensor.context);
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_to_device(tensor, device), context)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<B> {
        let tensor = &tensor.primitive;
        B::bool_device(tensor)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_reshape(tensor, shape), context)
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[ruda_core::tensor::Slice]) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_slice(tensor, slices), context)
    }

    fn bool_empty(shape: Shape, device: &Device<B>, dtype: BoolDType) -> BoolTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::bool_empty(shape, device, dtype), context)
    }

    fn bool_zeros(shape: Shape, device: &Device<B>, dtype: BoolDType) -> BoolTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::bool_zeros(shape, device, dtype), context)
    }

    fn bool_ones(shape: Shape, device: &Device<B>, dtype: BoolDType) -> BoolTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::bool_ones(shape, device, dtype), context)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[ruda_core::tensor::Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(value.context);
        let tensor = tensor.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::bool_slice_assign(tensor, slices, value), context)
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        let context = tensors.iter().fold(ruda_tensor::AutodiffContext::Disabled, |context, tensor| context.merge(tensor.context));
        let tensors = tensors.into_iter().map(|tensor| tensor.primitive).collect();
        AutodiffValue::with_context(B::bool_cat(tensors, dim), context)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bool_equal(lhs, rhs), context)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_not(tensor), context)
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bool_and(lhs, rhs), context)
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bool_or(lhs, rhs), context)
    }

    fn bool_xor(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bool_xor(lhs, rhs), context)
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffTensor::new(B::bool_into_float(tensor, out_dtype)).with_context::<C>(context)
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_swap_dims(tensor, dim1, dim2), context)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_permute(tensor, axes), context)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_flip(tensor, axes), context)
    }

    async fn bool_argwhere(tensor: BoolTensor<Self>, out_dtype: ruda_core::tensor::IntDType) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_argwhere(tensor, out_dtype).await, context)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_expand(tensor, shape), context)
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_repeat_dim(tensor, dim, times), context)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bool_unfold(tensor, dim, size, step), context)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        source: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(mask.context).merge(source.context);
        let tensor = tensor.primitive;
        let mask = mask.primitive;
        let source = source.primitive;
        AutodiffValue::with_context(B::bool_mask_where(tensor, mask, source), context)
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(mask.context);
        let tensor = tensor.primitive;
        let mask = mask.primitive;
        AutodiffValue::with_context(B::bool_mask_fill(tensor, mask, value), context)
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(indices.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        AutodiffValue::with_context(B::bool_gather(dim, tensor, indices), context)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(indices.context).merge(value.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::bool_scatter_or(dim, tensor, indices, value), context)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::bool_equal_elem(lhs, rhs), context)
    }

    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(indices.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        AutodiffValue::with_context(B::bool_select(tensor, dim, indices), context)
    }

    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let context = tensor.context.merge(indices.context).merge(value.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::bool_select_or(tensor, dim, indices, value), context)
    }
}
