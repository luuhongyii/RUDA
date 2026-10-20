use super::*;

impl IntTensorOps<Self> for Dispatch {
    fn int_autodiff_context(tensor: &IntTensor<Self>) -> ruda_tensor::AutodiffContext {
        tensor.context()
    }

    fn int_with_autodiff_context(tensor: IntTensor<Self>, context: ruda_tensor::AutodiffContext) -> IntTensor<Self> {
        tensor.with_context(context)
    }

    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_empty(shape, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_into_data( tensor: IntTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let future = <$B>::int_into_data(unpack!($mode, $variant, Int, tensor, $context));
                let future: core::pin::Pin<Box<dyn Future<Output = Result<TensorData, ExecutionError>> + Send>> = Box::pin(async move { future.await });
                future
            }};
        }
        dispatch!(selection, call)
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_from_data(data, $device);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        tensor.device()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        crate::transfer::int_to_device(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_reshape(unpack!($mode, $variant, Int, tensor, $context), shape);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_slice(unpack!($mode, $variant, Int, tensor, $context), slices);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_slice_assign( tensor: IntTensor<Self>, slices: &[Slice], value: IntTensor<Self>, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_slice_assign(unpack!($mode, $variant, Int, tensor, $context), slices, unpack!($mode, $variant, Int, value, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_into_float(tensor: IntTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_into_float(unpack!($mode, $variant, Int, tensor, $context), out_dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_mask_where( tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: IntTensor<Self>, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&mask)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_mask_where(unpack!($mode, $variant, Int, tensor, $context), unpack!($mode, $variant, Bool, mask, $context), unpack!($mode, $variant, Int, value, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_mask_fill(tensor: IntTensor<Self>, mask: BoolTensor<Self>, value: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&mask)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_mask_fill(unpack!($mode, $variant, Int, tensor, $context), unpack!($mode, $variant, Bool, mask, $context), value);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_gather(dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_gather(dim, unpack!($mode, $variant, Int, tensor, $context), unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_scatter_add( dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_scatter_add(dim, unpack!($mode, $variant, Int, tensor, $context), unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Int, value, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_scatter( dim: usize, tensor: IntTensor<Self>, indices: IntTensor<Self>, value: IntTensor<Self>, update: ruda_tensor::tensor::IndexingUpdateOp, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_scatter(dim, unpack!($mode, $variant, Int, tensor, $context), unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Int, value, $context), update);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_scatter_nd( _data: IntTensor<Self>, _indices: IntTensor<Self>, _values: IntTensor<Self>, _reduction: ruda_tensor::tensor::IndexingUpdateOp, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&_data).chain(core::iter::once(&_indices)).chain(core::iter::once(&_values)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_scatter_nd(unpack!($mode, $variant, Int, _data, $context), unpack!($mode, $variant, Int, _indices, $context), unpack!($mode, $variant, Int, _values, $context), _reduction);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_gather_nd(_data: IntTensor<Self>, _indices: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&_data).chain(core::iter::once(&_indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_gather_nd(unpack!($mode, $variant, Int, _data, $context), unpack!($mode, $variant, Int, _indices, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_select(tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_select(unpack!($mode, $variant, Int, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_select_add( tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_select_add(unpack!($mode, $variant, Int, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Int, value, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_select_assign( tensor: IntTensor<Self>, dim: usize, indices: IntTensor<Self>, value: IntTensor<Self>, update: ruda_tensor::tensor::IndexingUpdateOp, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_select_assign(unpack!($mode, $variant, Int, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Int, value, $context), update);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_repeat_dim(unpack!($mode, $variant, Int, tensor, $context), dim, times);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(tensors.iter());
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_cat(tensors.into_iter().map(|value| unpack!($mode, $variant, Int, value, $context)).collect(), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_equal(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_not_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_not_equal(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_equal_elem(unpack!($mode, $variant, Int, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_not_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_not_equal_elem(unpack!($mode, $variant, Int, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_greater(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_greater_elem(unpack!($mode, $variant, Int, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_greater_equal( lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_greater_equal(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_greater_equal_elem( lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_greater_equal_elem(unpack!($mode, $variant, Int, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_lower(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_lower_elem(unpack!($mode, $variant, Int, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_lower_equal(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_lower_equal_elem(unpack!($mode, $variant, Int, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_add(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_add_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_powi(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_powi_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_powi_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_powi_scalar_impl(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_powi_scalar_impl(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_clamp_min(unpack!($mode, $variant, Int, tensor, $context), min);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_clamp_max(tensor: IntTensor<Self>, max: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_clamp_max(unpack!($mode, $variant, Int, tensor, $context), max);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_clamp(tensor: IntTensor<Self>, min: Scalar, max: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_clamp(unpack!($mode, $variant, Int, tensor, $context), min, max);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sub(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sub_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_mul(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_mul_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_div(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_div_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_remainder(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_remainder_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_matmul(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_neg(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_zeros(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_zeros(shape, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_ones(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_ones(shape, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_full( shape: Shape, fill_value: Scalar, device: &Device<Self>, dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_full(shape, fill_value, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sum(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sum_dim(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_prod(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_prod_dim(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_mean(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_mean_dim(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_cumsum(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_cumprod(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_cummin(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_cummax(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_argmax(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_argtopk(tensor: IntTensor<Self>, dim: usize, k: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_argtopk(unpack!($mode, $variant, Int, tensor, $context), dim, k);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_topk(tensor: IntTensor<Self>, dim: usize, k: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_topk(unpack!($mode, $variant, Int, tensor, $context), dim, k);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_argmin(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_max(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_max_dim(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_max_dim_with_indices(tensor: IntTensor<Self>, dim: usize) -> (IntTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_max_dim_with_indices(unpack!($mode, $variant, Int, tensor, $context), dim);
                { let (value0, value1) = result; (pack!($mode, $variant, Int, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn int_max_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_max_abs(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_max_abs_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_max_abs_dim(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_min(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_min_dim(unpack!($mode, $variant, Int, tensor, $context), dim);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_min_dim_with_indices(tensor: IntTensor<Self>, dim: usize) -> (IntTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_min_dim_with_indices(unpack!($mode, $variant, Int, tensor, $context), dim);
                { let (value0, value1) = result; (pack!($mode, $variant, Int, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_abs(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_transpose(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_transpose(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_swap_dims(unpack!($mode, $variant, Int, tensor, $context), dim1, dim2);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_permute(unpack!($mode, $variant, Int, tensor, $context), axes);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_flip(unpack!($mode, $variant, Int, tensor, $context), axes);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_random( shape: Shape, distribution: Distribution, device: &Device<Self>, dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_random(shape, distribution, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_arange_step( range: Range<i64>, step: usize, device: &Device<Self>, dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_arange_step(range, step, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_arange(range: Range<i64>, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_arange(range, $device, dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_any(tensor: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_any(unpack!($mode, $variant, Int, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_any_dim(tensor: IntTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_any_dim(unpack!($mode, $variant, Int, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_all(tensor: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_all(unpack!($mode, $variant, Int, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_all_dim(tensor: IntTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_all_dim(unpack!($mode, $variant, Int, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sign(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_expand(unpack!($mode, $variant, Int, tensor, $context), shape);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sort(unpack!($mode, $variant, Int, tensor, $context), dim, descending);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_sort_with_indices( tensor: IntTensor<Self>, dim: usize, descending: bool, ) -> (IntTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_sort_with_indices(unpack!($mode, $variant, Int, tensor, $context), dim, descending);
                { let (value0, value1) = result; (pack!($mode, $variant, Int, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn int_argsort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_argsort(unpack!($mode, $variant, Int, tensor, $context), dim, descending);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_and(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_and_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_or(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_or_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_xor(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_xor_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_not(unpack!($mode, $variant, Int, tensor, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_left_shift(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_left_shift_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_right_shift(unpack!($mode, $variant, Int, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::bitwise_right_shift_scalar(unpack!($mode, $variant, Int, lhs, $context), rhs);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_cast(unpack!($mode, $variant, Int, tensor, $context), dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn int_unfold(tensor: IntTensor<Self>, dim: usize, size: usize, step: usize) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::int_unfold(unpack!($mode, $variant, Int, tensor, $context), dim, size, step);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }
}
