use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::{AutodiffTensor, AutodiffValue}};
use alloc::vec::Vec;

use ruda_tensor::{
    Backend, Distribution, ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use ruda_core::tensor::{BoolDType, FloatDType, IntDType, Shape};

impl<B: Backend, C: CheckpointStrategy> IntTensorOps<Self> for Autodiff<B, C> {
    fn int_autodiff_context(tensor: &IntTensor<Self>) -> ruda_tensor::AutodiffContext { tensor.context }

    fn int_with_autodiff_context(tensor: IntTensor<Self>, context: ruda_tensor::AutodiffContext) -> IntTensor<Self> {
        tensor.set_context::<C>(context)
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_from_data(data, device), context)
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        let tensor = tensor.primitive;
        B::int_into_data(tensor).await
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        let context = crate::context::scoped::<B, C>(device).unwrap_or(tensor.context);
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_to_device(tensor, device), context)
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        let tensor = &tensor.primitive;
        B::int_device(tensor)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_reshape(tensor, shape), context)
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[ruda_core::tensor::Slice]) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_slice(tensor, slices), context)
    }

    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_empty(shape, device, dtype), context)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[ruda_core::tensor::Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let context = tensor.context.merge(value.context);
        let tensor = tensor.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::int_slice_assign(tensor, slices, value), context)
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        let context = tensors.iter().fold(ruda_tensor::AutodiffContext::Disabled, |context, tensor| context.merge(tensor.context));
        let tensors = tensors.into_iter().map(|tensor| tensor.primitive).collect();
        AutodiffValue::with_context(B::int_cat(tensors, dim), context)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_equal(lhs, rhs, out_dtype), context)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_equal_elem(lhs, rhs, out_dtype), context)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_add(lhs, rhs), context)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_add_scalar(lhs, rhs), context)
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: Scalar) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_clamp_min(tensor, min), context)
    }

    fn int_clamp_max(tensor: IntTensor<Self>, max: Scalar) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_clamp_max(tensor, max), context)
    }

    fn int_clamp(tensor: IntTensor<Self>, min: Scalar, max: Scalar) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_clamp(tensor, min, max), context)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_sub(lhs, rhs), context)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_sub_scalar(lhs, rhs), context)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_mul(lhs, rhs), context)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_mul_scalar(lhs, rhs), context)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_div(lhs, rhs), context)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_div_scalar(lhs, rhs), context)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_remainder(lhs, rhs), context)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_remainder_scalar(lhs, rhs), context)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_matmul(lhs, rhs), context)
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_neg(tensor), context)
    }

    fn int_zeros(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_zeros(shape, device, dtype), context)
    }

    fn int_ones(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_ones(shape, device, dtype), context)
    }

    fn int_full(
        shape: Shape,
        fill_value: Scalar,
        device: &Device<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_full(shape, fill_value, device, dtype), context)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_sum(tensor), context)
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_sum_dim(tensor, dim), context)
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_mean(tensor), context)
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_mean_dim(tensor, dim), context)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_cumsum(tensor, dim), context)
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_cumprod(tensor, dim), context)
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_cummin(tensor, dim), context)
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_cummax(tensor, dim), context)
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_repeat_dim(tensor, dim, times), context)
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_greater(lhs, rhs, out_dtype), context)
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_greater_elem(lhs, rhs, out_dtype), context)
    }

    fn int_greater_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_greater_equal(lhs, rhs, out_dtype), context)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_greater_equal_elem(lhs, rhs, out_dtype), context)
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_lower(lhs, rhs, out_dtype), context)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_lower_elem(lhs, rhs, out_dtype), context)
    }

    fn int_lower_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::int_lower_equal(lhs, rhs, out_dtype), context)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::int_lower_equal_elem(lhs, rhs, out_dtype), context)
    }

    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context.merge(indices.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        AutodiffValue::with_context(B::int_gather(dim, tensor, indices), context)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
        update: ruda_tensor::tensor::IndexingUpdateOp,
    ) -> IntTensor<Self> {
        let context = tensor.context.merge(indices.context).merge(value.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::int_scatter(dim, tensor, indices, value, update), context)
    }

    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let context = tensor.context.merge(indices.context).merge(value.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::int_scatter_add(dim, tensor, indices, value), context)
    }

    fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context.merge(indices.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        AutodiffValue::with_context(B::int_select(tensor, dim, indices), context)
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
        update: ruda_tensor::tensor::IndexingUpdateOp,
    ) -> IntTensor<Self> {
        let context = tensor.context.merge(indices.context).merge(value.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::int_select_assign(tensor, dim, indices, value, update), context)
    }

    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let context = tensor.context.merge(indices.context).merge(value.context);
        let tensor = tensor.primitive;
        let indices = indices.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::int_select_add(tensor, dim, indices, value), context)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let context = tensor.context.merge(mask.context).merge(value.context);
        let tensor = tensor.primitive;
        let mask = mask.primitive;
        let value = value.primitive;
        AutodiffValue::with_context(B::int_mask_where(tensor, mask, value), context)
    }

    fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> IntTensor<Self> {
        let context = tensor.context.merge(mask.context);
        let tensor = tensor.primitive;
        let mask = mask.primitive;
        AutodiffValue::with_context(B::int_mask_fill(tensor, mask, value), context)
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_argmax(tensor, dim), context)
    }

    fn int_argtopk(tensor: IntTensor<Self>, dim: usize, k: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_argtopk(tensor, dim, k), context)
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_argmin(tensor, dim), context)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_max(tensor), context)
    }
    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_max_dim(tensor, dim), context)
    }
    fn int_topk(tensor: IntTensor<Self>, dim: usize, k: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_topk(tensor, dim, k), context)
    }
    fn int_max_dim_with_indices(tensor: IntTensor<Self>, dim: usize) -> (IntTensor<Self>, IntTensor<Self>) {
        let context = tensor.context;
        let tensor = tensor.primitive;
        {
            let (values, indices) = B::int_max_dim_with_indices(tensor, dim);
            (AutodiffValue::with_context(values, context), AutodiffValue::with_context(indices, context))
        }
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_min(tensor), context)
    }
    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_min_dim(tensor, dim), context)
    }
    fn int_min_dim_with_indices(tensor: IntTensor<Self>, dim: usize) -> (IntTensor<Self>, IntTensor<Self>) {
        let context = tensor.context;
        let tensor = tensor.primitive;
        {
            let (values, indices) = B::int_min_dim_with_indices(tensor, dim);
            (AutodiffValue::with_context(values, context), AutodiffValue::with_context(indices, context))
        }
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_abs(tensor), context)
    }
    fn int_into_float(tensor: IntTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffTensor::new(B::int_into_float(tensor, out_dtype)).with_context::<C>(context)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_swap_dims(tensor, dim1, dim2), context)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_random(shape, distribution, device, dtype), context)
    }

    fn int_arange(
        range: core::ops::Range<i64>,
        device: &Device<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        let context = Self::autodiff_creation_context(device);
        AutodiffValue::with_context(B::int_arange(range, device, dtype), context)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_permute(tensor, axes), context)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_flip(tensor, axes), context)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_sign(tensor), context)
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_prod(tensor), context)
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_prod_dim(tensor, dim), context)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_expand(tensor, shape), context)
    }

    fn int_sort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_sort(tensor, dim, descending), context)
    }

    fn int_sort_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        let context = tensor.context;
        let tensor = tensor.primitive;
        {
            let (values, indices) = B::int_sort_with_indices(tensor, dim, descending);
            (AutodiffValue::with_context(values, context), AutodiffValue::with_context(indices, context))
        }
    }

    fn int_argsort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_argsort(tensor, dim, descending), context)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bitwise_and(lhs, rhs), context)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::bitwise_and_scalar(lhs, rhs), context)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bitwise_or(lhs, rhs), context)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::bitwise_or_scalar(lhs, rhs), context)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bitwise_xor(lhs, rhs), context)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::bitwise_xor_scalar(lhs, rhs), context)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::bitwise_not(tensor), context)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bitwise_left_shift(lhs, rhs), context)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::bitwise_left_shift_scalar(lhs, rhs), context)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let context = lhs.context.merge(rhs.context);
        let lhs = lhs.primitive;
        let rhs = rhs.primitive;
        AutodiffValue::with_context(B::bitwise_right_shift(lhs, rhs), context)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let context = lhs.context;
        let lhs = lhs.primitive;
        AutodiffValue::with_context(B::bitwise_right_shift_scalar(lhs, rhs), context)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_cast(tensor, dtype), context)
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        let context = tensor.context;
        let tensor = tensor.primitive;
        AutodiffValue::with_context(B::int_unfold(tensor, dim, size, step), context)
    }
}
