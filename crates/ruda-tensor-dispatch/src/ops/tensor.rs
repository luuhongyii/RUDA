use super::*;

impl FloatTensorOps<Self> for Dispatch {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_from_data(data, $device);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_random( shape: Shape, distribution: Distribution, device: &Device<Self>, dtype: FloatDType, ) -> FloatTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_random(shape, distribution, $device, dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_zeros(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_zeros(shape, $device, dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_ones(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_ones(shape, $device, dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_full( shape: Shape, fill_value: Scalar, device: &Device<Self>, dtype: FloatDType, ) -> FloatTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_full(shape, fill_value, $device, dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_into_data( tensor: FloatTensor<Self>, ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let future = <$B>::float_into_data(unpack!($mode, $variant, Float, tensor, $context));
                let future: core::pin::Pin<Box<dyn Future<Output = Result<TensorData, ExecutionError>> + Send>> = Box::pin(async move { future.await });
                future
            }};
        }
        dispatch!(selection, call)
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        crate::transfer::float_to_device(tensor, device)
    }

    fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_into_int(unpack!($mode, $variant, Float, tensor, $context), out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::factory(device);
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_empty(shape, $device, dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_repeat_dim(unpack!($mode, $variant, Float, tensor, $context), dim, times);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_add(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_add_scalar(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_clamp_min(unpack!($mode, $variant, Float, tensor, $context), min);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_clamp_max(unpack!($mode, $variant, Float, tensor, $context), max);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: Scalar, max: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_clamp(unpack!($mode, $variant, Float, tensor, $context), min, max);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sub(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sub_scalar(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_mul(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_mul_scalar(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_div(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_div_scalar(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_remainder(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_remainder_scalar(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_matmul(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cross(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cross(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_neg(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_recip(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_transpose(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_transpose(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_swap_dims(unpack!($mode, $variant, Float, tensor, $context), dim1, dim2);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_permute(unpack!($mode, $variant, Float, tensor, $context), axes);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_flip(unpack!($mode, $variant, Float, tensor, $context), axes);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_reshape(unpack!($mode, $variant, Float, tensor, $context), shape);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_gather(dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_gather(dim, unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_scatter_add( dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>, value: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_scatter_add(dim, unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Float, value, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_scatter( dim: usize, tensor: FloatTensor<Self>, indices: IntTensor<Self>, value: FloatTensor<Self>, update: ruda_tensor::tensor::IndexingUpdateOp, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_scatter(dim, unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Float, value, $context), update);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_scatter_nd( _data: FloatTensor<Self>, _indices: IntTensor<Self>, _values: FloatTensor<Self>, _reduction: ruda_tensor::tensor::IndexingUpdateOp, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&_data).chain(core::iter::once(&_indices)).chain(core::iter::once(&_values)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_scatter_nd(unpack!($mode, $variant, Float, _data, $context), unpack!($mode, $variant, Int, _indices, $context), unpack!($mode, $variant, Float, _values, $context), _reduction);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_gather_nd(_data: FloatTensor<Self>, _indices: IntTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&_data).chain(core::iter::once(&_indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_gather_nd(unpack!($mode, $variant, Float, _data, $context), unpack!($mode, $variant, Int, _indices, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_select(tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_select(unpack!($mode, $variant, Float, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_select_add( tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>, value: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_select_add(unpack!($mode, $variant, Float, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Float, value, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_select_assign( tensor: FloatTensor<Self>, dim: usize, indices: IntTensor<Self>, value: FloatTensor<Self>, update: ruda_tensor::tensor::IndexingUpdateOp, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&indices)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_select_assign(unpack!($mode, $variant, Float, tensor, $context), dim, unpack!($mode, $variant, Int, indices, $context), unpack!($mode, $variant, Float, value, $context), update);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_slice(unpack!($mode, $variant, Float, tensor, $context), slices);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_slice_assign( tensor: FloatTensor<Self>, slices: &[Slice], value: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_slice_assign(unpack!($mode, $variant, Float, tensor, $context), slices, unpack!($mode, $variant, Float, value, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_mask_where( tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: FloatTensor<Self>, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&mask)).chain(core::iter::once(&value)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_mask_where(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Bool, mask, $context), unpack!($mode, $variant, Float, value, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_mask_fill( tensor: FloatTensor<Self>, mask: BoolTensor<Self>, value: Scalar, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&mask)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_mask_fill(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Bool, mask, $context), value);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_equal(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_not_equal( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_not_equal(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_equal_elem(unpack!($mode, $variant, Float, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_not_equal_elem( lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_not_equal_elem(unpack!($mode, $variant, Float, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_greater( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_greater(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_greater_elem(unpack!($mode, $variant, Float, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_greater_equal( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_greater_equal(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_greater_equal_elem( lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_greater_equal_elem(unpack!($mode, $variant, Float, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_lower(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_lower_elem(unpack!($mode, $variant, Float, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_lower_equal( lhs: FloatTensor<Self>, rhs: FloatTensor<Self>, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_lower_equal(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_lower_equal_elem( lhs: FloatTensor<Self>, rhs: Scalar, out_dtype: BoolDType, ) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_lower_equal_elem(unpack!($mode, $variant, Float, lhs, $context), rhs, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_detach(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_set_require_grad(tensor: FloatTensor<Self>, _require_grad: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_set_require_grad(unpack!($mode, $variant, Float, tensor, $context), _require_grad);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_is_require_grad(_tensor: &FloatTensor<Self>) -> bool {
        let selection = Selection::tensors(core::iter::once(_tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_is_require_grad(borrow_tensor!($mode, $variant, Float, _tensor));
                result
            }};
        }
        dispatch!(selection, call)
    }

    fn float_is_tracked(tensor: &FloatTensor<Self>) -> bool {
        let selection = Selection::tensors(core::iter::once(tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_is_tracked(borrow_tensor!($mode, $variant, Float, tensor));
                result
            }};
        }
        dispatch!(selection, call)
    }

    fn float_gradient_checkpointing_strategy(_tensor: &FloatTensor<Self>) -> Option<ruda_tensor::GradientCheckpointingStrategy> {
        match _tensor.context() { AutodiffContext::Disabled => None, AutodiffContext::Enabled(strategy) => Some(strategy) }
    }

    fn float_autodiff_context(tensor: &FloatTensor<Self>) -> ruda_tensor::AutodiffContext {
        tensor.context()
    }

    fn float_with_autodiff_context(tensor: FloatTensor<Self>, context: ruda_tensor::AutodiffContext) -> FloatTensor<Self> {
        tensor.with_context(context)
    }

    fn float_with_gradient_checkpointing_strategy( _tensor: FloatTensor<Self>, _strategy: ruda_tensor::GradientCheckpointingStrategy, ) -> FloatTensor<Self> {
        assert!(matches!(_tensor.context(), AutodiffContext::Enabled(_)), "Checkpointing requires autodiff");
        _tensor.with_context(AutodiffContext::Enabled(_strategy))
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sum(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sum_dim(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_prod(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_prod_dim(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_mean(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_mean_dim(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cumsum(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cumprod(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cummin(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cummax(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cast(unpack!($mode, $variant, Float, tensor, $context), dtype);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_exp(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_log(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_log1p(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_powf(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_powi(lhs: FloatTensor<Self>, rhs: IntTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_powi(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Int, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_powi_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_powi_scalar(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_powi_scalar_impl(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_powi_scalar_impl(unpack!($mode, $variant, Float, lhs, $context), rhs);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_powf_scalar(unpack!($mode, $variant, Float, tensor, $context), value);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_powf_scalar_impl(unpack!($mode, $variant, Float, tensor, $context), value);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sqrt(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_hypot(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_hypot(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_rsqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_rsqrt(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_abs(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cos(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sin(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_tan(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cosh(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sinh(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_tanh(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_acos(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_acosh(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_asin(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_asinh(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_atan(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_atanh(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&lhs).chain(core::iter::once(&rhs)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_atan2(unpack!($mode, $variant, Float, lhs, $context), unpack!($mode, $variant, Float, rhs, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_round(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_floor(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_ceil(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_trunc(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_erf(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(tensors.iter());
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_cat(tensors.into_iter().map(|value| unpack!($mode, $variant, Float, value, $context)).collect(), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_argmax(unpack!($mode, $variant, Float, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_argtopk( tensor: FloatTensor<Self>, dim: usize, k: usize, out_dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_argtopk(unpack!($mode, $variant, Float, tensor, $context), dim, k, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_topk(tensor: FloatTensor<Self>, dim: usize, k: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_topk(unpack!($mode, $variant, Float, tensor, $context), dim, k);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_argmin(unpack!($mode, $variant, Float, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_max(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_max_dim(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_max_dim_with_indices( tensor: FloatTensor<Self>, dim: usize, indices_dtype: IntDType, ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_max_dim_with_indices(unpack!($mode, $variant, Float, tensor, $context), dim, indices_dtype);
                { let (value0, value1) = result; (pack!($mode, $variant, Float, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_min(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_min_dim(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_min_dim_with_indices( tensor: FloatTensor<Self>, dim: usize, indices_dtype: IntDType, ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_min_dim_with_indices(unpack!($mode, $variant, Float, tensor, $context), dim, indices_dtype);
                { let (value0, value1) = result; (pack!($mode, $variant, Float, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_max_abs(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_max_abs_dim(unpack!($mode, $variant, Float, tensor, $context), dim);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_any(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_any(unpack!($mode, $variant, Float, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_any_dim(tensor: FloatTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_any_dim(unpack!($mode, $variant, Float, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_all(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_all(unpack!($mode, $variant, Float, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_all_dim(tensor: FloatTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_all_dim(unpack!($mode, $variant, Float, tensor, $context), dim, out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sign(unpack!($mode, $variant, Float, tensor, $context));
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_expand(unpack!($mode, $variant, Float, tensor, $context), shape);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sort(tensor: FloatTensor<Self>, dim: usize, descending: bool) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sort(unpack!($mode, $variant, Float, tensor, $context), dim, descending);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_sort_with_indices( tensor: FloatTensor<Self>, dim: usize, descending: bool, indices_dtype: IntDType, ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_sort_with_indices(unpack!($mode, $variant, Float, tensor, $context), dim, descending, indices_dtype);
                { let (value0, value1) = result; (pack!($mode, $variant, Float, value0), pack!($mode, $variant, Int, value1)) }
            }};
        }
        dispatch!(selection, call)
    }

    fn float_argsort( tensor: FloatTensor<Self>, dim: usize, descending: bool, out_dtype: IntDType, ) -> IntTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_argsort(unpack!($mode, $variant, Float, tensor, $context), dim, descending, out_dtype);
                pack!($mode, $variant, Int, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_grid_sample_2d( tensor: FloatTensor<Self>, grid: FloatTensor<Self>, options: GridSampleOptions, ) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor).chain(core::iter::once(&grid)));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_grid_sample_2d(unpack!($mode, $variant, Float, tensor, $context), unpack!($mode, $variant, Float, grid, $context), options);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_unfold(tensor: FloatTensor<Self>, dim: usize, size: usize, step: usize) -> FloatTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_unfold(unpack!($mode, $variant, Float, tensor, $context), dim, size, step);
                pack!($mode, $variant, Float, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_is_nan(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_is_nan(unpack!($mode, $variant, Float, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }

    fn float_is_inf(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        let selection = Selection::tensors(core::iter::once(&tensor));
        macro_rules! call {
            ($B:ty, $variant:ident, $mode:ident, $device:expr, $context:expr) => {{
                let result = <$B>::float_is_inf(unpack!($mode, $variant, Float, tensor, $context), out_dtype);
                pack!($mode, $variant, Bool, result)
            }};
        }
        dispatch!(selection, call)
    }
}
